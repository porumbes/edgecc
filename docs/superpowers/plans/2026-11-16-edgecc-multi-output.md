# Multi-output `run()` and multiple `save:` — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An `output: B, Z;` directive makes `run()` return several tensors (a tuple in listed order), and `save:` may appear several times, once per output (DFT gap G3, batch C).

**Architecture:** The lexer/parser add an `output:` directive (`NODE_OUTPUT`). Semantic analysis resolves the list of outputs (explicit list, or today's default) and the list of saves into the `SymbolTable`. IR lowering emits one `IR_OUTPUT` per output, then one `IR_SAVE` per save. Codegen reads `IR_OUTPUT` instead of guessing "the last store", and drives `__edgecc_meta__`, the `return`, and `__main__` from that list.

**Tech Stack:** C99, Flex/Bison (Bison 2.3 compatible), CMake/ctest, Python 3.8 for generated code and test harnesses, TypeScript + `vscode-tmgrammar-test` for the VS Code grammar.

**Spec:** `docs/superpowers/specs/2026-11-16-edgecc-multi-output-design.md`

## Global Constraints

- Work only in the worktree `.worktrees/multi-output` (branch `feature/multi-output`). Never `git checkout`, edit, or build in the repo root; it belongs to the DFT exploration session (branch `dft-exploration`).
- Build: `cmake -S . -B build && make -C build` (no compiler warnings). Tests: `ctest --test-dir build --output-on-failure`.
- Generated Python and harnesses must run on Python 3.8 (no `dict |`, no `match`, no `list[int]` annotations).
- No `Co-Authored-By` trailer in commit messages. Edit files with an editor, not `sed -i` (BSD `sed -i` leaves `*-e` backups that get committed).
- Exact error messages (tests match them as substrings):
  - `output: unknown tensor '<T>'`
  - `output: '<T>' is not written by any Einsum (inputs are not outputs)`
  - `output: '<T>' listed twice`
  - `output: only one output directive is allowed`
  - `save: '<T>' is not an output of run(); add it to the output: list`
  - `save: '<T>' is saved twice`
  - `save: path "<P>" is used twice`
- Without `output:`, the outputs are today's default: the stop-watched tensor (generational) or the last Einsum's output.
- One output → `return Z` and meta keeps `"output": "Z"`; several → `return (B, Z)` and meta has no `"output"` key. Every program's meta gains `"outputs": [...]` in return order.
- `__main__` prints one line per output, in listed order, then runs every `save:` in source order.
- `output` becomes a lexer keyword.
- No `tests/reference/` case (spec: ABI plumbing; independent oracles are numpy DFT values and hand-computed BFS values).

## Review Focus

- **Generational outputs other than the watched tensor.** `output: F, P;` on reachability BFS must return both full generation lists; the trace header (`--emit-trace`) must still size `N` from the watched tensor, not from `outputs[0]`. Task 3 adds `tests/e2e/multi_output_bfs` and keeps `trace_tests` green.
- **An intermediate listed as an output** (not the last Einsum's output) must be returned with the value it has at the end of `run()`. Task 3 adds `tests/e2e/output_intermediate`.
- **Output order differs from write order** (`output: Z, B;`): return tuple, unpacking and print order follow the list, not source order. Task 3 adds `tests/e2e/output_order` (also `run_twice`).
- **Consumers that assume one output.** A multi-output kernel in the iterative driver must fail with a clear message, not a confusing tuple-combine error. Task 3 adds `tests/e2e/iterative/multi_output_rejected`.
- **Saves of a non-last output in `__main__`** must name a variable `__main__` actually defines. Task 3's `multi_output_save` e2e reads both files back.

---

### Task 1: Lexer, AST and parser for `output:`

**Files:**
- Modify: `src/lexer.l`, `src/parser.y`, `src/ast.h`, `src/ast.c`, `src/semantic.c` (statement dispatch only)
- Test: `tests/unit/test_lexer.c`, `tests/unit/test_parser.c`
- Create: `tests/negative/output_empty_list/{input.edge,expected_errors.txt}`

**Interfaces:**
- Produces: `NODE_OUTPUT`; `typedef struct { char **names; int count; } OutputData;` reachable as `node->data.output`; `AstNode *ast_output_new(char **names, int count)` (takes ownership of `names` and each string); `TOK_OUTPUT`.

- [ ] **Step 1: Write the failing tests**

In `tests/unit/test_lexer.c`, after `test_stop_keyword`, add (and `RUN_TEST(test_output_keyword);` in `main`):
```c
TEST(test_output_keyword) {
    YY_BUFFER_STATE buf = yy_scan_string("output");
    ASSERT_INT_EQ(TOK_OUTPUT, yylex());
    yy_delete_buffer(buf);
}
```

In `tests/unit/test_parser.c`, after `test_parse_save_directive`, add (and the three `RUN_TEST` lines at the end of `main`):
```c
TEST(test_parse_output_directive) {
    YY_BUFFER_STATE buf = yy_scan_string("output: B, Z;\n");
    ASSERT_INT_EQ(0, yyparse());
    ASSERT_NOT_NULL(parse_result);
    AstNode *o = parse_result->data.program.statements[0];
    ASSERT_INT_EQ(NODE_OUTPUT, o->type);
    ASSERT_INT_EQ(2, o->data.output.count);
    ASSERT_STR_EQ("B", o->data.output.names[0]);
    ASSERT_STR_EQ("Z", o->data.output.names[1]);
    ast_free(parse_result);
    parse_result = NULL;
    yy_delete_buffer(buf);
}

TEST(test_parse_output_empty_list_fails) {
    YY_BUFFER_STATE buf = yy_scan_string("output: ;\n");
    ASSERT_TRUE(yyparse() != 0);
    yy_delete_buffer(buf);
}

TEST(test_parse_output_offset_fails) {
    /* output: takes plain tensor names; a rank offset is not a tensor name. */
    YY_BUFFER_STATE buf = yy_scan_string("output: B+1;\n");
    ASSERT_TRUE(yyparse() != 0);
    yy_delete_buffer(buf);
}
```
For the two failing-parse tests, clean up `parse_result` exactly the way `test_parse_fused_missing_map_fails` does in the same file (copy its cleanup lines).

`tests/negative/output_empty_list/input.edge`:
```
// output: needs at least one tensor name.
dim N = 2;
tensor A[R=N] : float, empty = 0.0;
tensor Z[R=N] : float, empty = 0.0;
init A = [1.0, 2.0];
Z[i] = A[i] . A[i] :: map(mul, intersect) reduce(add, union);
output: ;
```
`tests/negative/output_empty_list/expected_errors.txt`:
```
syntax error
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `make -C build 2>&1 | tail -3`
Expected: `test_lexer.c` / `test_parser.c` fail to compile (`TOK_OUTPUT`, `NODE_OUTPUT` undeclared). That is the red state for this task. (`output_empty_list` already passes today because `output` lexes as an identifier and the line is a syntax error; it guards the rule after the keyword exists.)

- [ ] **Step 3: Implement**

`src/lexer.l`, after the `"save"` line:
```
"output"            { return TOK_OUTPUT; }
```

`src/ast.h`: add `NODE_OUTPUT` after `NODE_SAVE` in `NodeType`; after `SaveData` add
```c
typedef struct {
    char **names;   /* tensor names in return order (owned) */
    int count;
} OutputData;
```
add `OutputData output;` to the `AstNode` union after `SaveData save;`, and declare `AstNode *ast_output_new(char **names, int count);` after `ast_save_new`.

`src/ast.c`, after `ast_save_new`:
```c
AstNode *ast_output_new(char **names, int count) {
    AstNode *node = calloc(1, sizeof(AstNode));
    node->type = NODE_OUTPUT;
    node->data.output.names = names;
    node->data.output.count = count;
    return node;
}
```
and in `ast_free`, after the `NODE_SAVE` case:
```c
        case NODE_OUTPUT:
            for (int i = 0; i < node->data.output.count; i++)
                free(node->data.output.names[i]);
            free(node->data.output.names);
            break;
```

`src/parser.y`:
- `%token TOK_LOAD TOK_SAVE TOK_ARROW TOK_SEED TOK_OVER` → add `TOK_OUTPUT`.
- Add `output_stmt` to the `%type <node>` line and `output_names` to a `%type <ptr>` line.
- In `stmt`, add `| output_stmt` after `| save_stmt`.
- After the `save_stmt` rule:
```
output_stmt:
    TOK_OUTPUT ':' output_names ';'
    {
        StrList *sl = (StrList *)$3;
        $$ = (void *)ast_output_new(sl->items, sl->count);
        free(sl->offsets);
        free(sl);
    }
    ;

output_names:
    TOK_IDENT
    {
        StrList *sl = strlist_new();
        strlist_add(sl, $1, 0);
        $$ = (void *)sl;
    }
    | output_names ',' TOK_IDENT
    {
        StrList *sl = (StrList *)$1;
        strlist_add(sl, $3, 0);
        $$ = (void *)sl;
    }
    ;
```

`src/semantic.c`, in the statement switch, after the `NODE_SAVE` case:
```c
            case NODE_OUTPUT:
                /* Validated with the save directives after all Einsums. */
                break;
```
Grep `src/` for any other `switch` over `NodeType` with a `case NODE_SAVE:` and add `case NODE_OUTPUT:` beside it so `-Wswitch` stays quiet.

- [ ] **Step 4: Run the tests**

Run: `make -C build 2>&1 | grep -E "error|warning:"; ./build/test_lexer | tail -1; ./build/test_parser | tail -1; python3 tests/negative/run_negative.py build/edgecc 2>&1 | tail -1; ctest --test-dir build 2>&1 | tail -3`
Expected: no warnings; lexer and parser tests all pass; all negatives pass; `100% tests passed`.

- [ ] **Step 5: Commit**

```bash
git add src/lexer.l src/parser.y src/ast.h src/ast.c src/semantic.c tests/unit/test_lexer.c tests/unit/test_parser.c tests/negative/output_empty_list
git commit -m "parser: output: directive (NODE_OUTPUT, output keyword)"
```

---

### Task 2: Semantic outputs/saves and IR lowering

**Files:**
- Modify: `src/semantic.h` (`SymbolTable`), `src/semantic.c` (replace the save-directive pass), `src/ir.h`, `src/ir.c`, `src/codegen.c` (one no-op `case`)
- Test: `tests/unit/test_semantic.c`, `tests/unit/test_ir.c`
- Create: `tests/negative/{output_unknown_tensor,output_input_tensor,output_listed_twice,output_two_directives,save_non_output,save_tensor_twice,save_path_twice}/{input.edge,expected_errors.txt}`
- Modify (reconciliation): `tests/negative/{save_not_returned_intermediate,save_not_returned_nongen,save_not_returned_gen,save_not_output}/expected_errors.txt`

**Interfaces:**
- Consumes: Task 1's `NODE_OUTPUT` / `data.output.{names,count}`; existing `written_by_some_einsum(ProgramData *, const char *)` and `name_present(char *const[], int, const char *)` in `src/semantic.c`.
- Produces: `SymbolTable` fields `char *outputs[MAX_TENSORS]; int num_outputs; char *save_tensors[MAX_TENSORS]; char *save_paths[MAX_TENSORS]; int num_saves;` (all borrowed; replace `save_tensor` / `save_path`). IR: `IR_OUTPUT` with `typedef struct { char *tensor_name; /* owned */ } IROutput;` at `instr->data.output`; one `IR_OUTPUT` per output in return order, followed by one `IR_SAVE` per save in source order, all after the Einsums. `ir_dump` prints `OUTPUT(<T>)`.

The base program used by several tests below (call it **G3**):
```
dim N = 2;
tensor A[R=N] : float, empty = 0.0;
tensor B[R=N] : float, empty = 0.0;
tensor Z[R=N] : float, empty = 0.0;
init A = [1.0, 2.0];
B[i] = A[i] . A[i] :: map(mul, intersect) reduce(add, union);
Z[i] = B[i] . A[i] :: map(add, intersect) reduce(add, union);
```

- [ ] **Step 1: Write the negative tests**

Each `input.edge` is a one-line `//` comment, then the G3 program, then the lines shown. Each `expected_errors.txt` holds the line shown.

| Directory | Lines after G3 | `expected_errors.txt` |
|---|---|---|
| `output_unknown_tensor` | `output: B, Q;` | `output: unknown tensor 'Q'` |
| `output_input_tensor` | `output: A, Z;` | `output: 'A' is not written by any Einsum (inputs are not outputs)` |
| `output_listed_twice` | `output: Z, Z;` | `output: 'Z' listed twice` |
| `output_two_directives` | `output: B;` and `output: Z;` (two lines) | `output: only one output directive is allowed` |
| `save_non_output` | `output: Z;` and `save: B -> "b.mtx";` | `save: 'B' is not an output of run(); add it to the output: list` |
| `save_tensor_twice` | `output: B, Z;`, `save: Z -> "a.mtx";`, `save: Z -> "b.mtx";` | `save: 'Z' is saved twice` |
| `save_path_twice` | `output: B, Z;`, `save: B -> "x.mtx";`, `save: Z -> "x.mtx";` | `save: path "x.mtx" is used twice` |

Comments: `output_unknown_tensor`: `// output: names an undeclared tensor (unknown-value test for the output: slot).`; `output_input_tensor`: `// An input is never written, so it cannot be an output.`; `output_listed_twice`: `// Each output appears once in the list.`; `output_two_directives`: `// At most one output: directive.`; `save_non_output`: `// save: may only target one of run()'s outputs.`; `save_tensor_twice`: `// Each tensor is saved at most once.`; `save_path_twice`: `// Two saves may not write the same file.`

Reconciliation (lifted restriction "save must target the returned tensor"; each program is still invalid because it has no `output:` list): replace the single line of each `expected_errors.txt`:
- `save_not_returned_intermediate`: `save: 'T' is not an output of run(); add it to the output: list`
- `save_not_returned_nongen`: `save: 'Z1' is not an output of run(); add it to the output: list`
- `save_not_returned_gen`: `save: 'P' is not an output of run(); add it to the output: list`
- `save_not_output`: `save: 'G' is not an output of run(); add it to the output: list`

Also grep `tests/` for `only one save directive` and `is not the tensor run() returns`; any other hit gets the same treatment (list it in the report).

- [ ] **Step 2: Write the unit tests**

In `tests/unit/test_semantic.c`, before `main` (and `RUN_TEST` each):
```c
static const char *G3_SRC =
    "dim N = 2;\n"
    "tensor A[R=N] : float, empty = 0.0;\n"
    "tensor B[R=N] : float, empty = 0.0;\n"
    "tensor Z[R=N] : float, empty = 0.0;\n"
    "init A = [1.0, 2.0];\n"
    "B[i] = A[i] . A[i] :: map(mul, intersect) reduce(add, union);\n"
    "Z[i] = B[i] . A[i] :: map(add, intersect) reduce(add, union);\n";

static AstNode *parse_g3_with(const char *tail) {
    char buf[2048];
    snprintf(buf, sizeof(buf), "%s%s", G3_SRC, tail);
    return parse(buf);
}

TEST(test_semantic_outputs_default_last_einsum) {
    AstNode *ast = parse_g3_with("");
    SemanticResult sem = semantic_analyze(ast);
    ASSERT_INT_EQ(0, sem.error_count);
    ASSERT_INT_EQ(1, sem.symtab->num_outputs);
    ASSERT_STR_EQ("Z", sem.symtab->outputs[0]);
    semantic_result_free(&sem);
    ast_free(ast);
}

TEST(test_semantic_outputs_listed_in_order) {
    AstNode *ast = parse_g3_with("output: Z, B;\n");
    SemanticResult sem = semantic_analyze(ast);
    ASSERT_INT_EQ(0, sem.error_count);
    ASSERT_INT_EQ(2, sem.symtab->num_outputs);
    ASSERT_STR_EQ("Z", sem.symtab->outputs[0]);
    ASSERT_STR_EQ("B", sem.symtab->outputs[1]);
    semantic_result_free(&sem);
    ast_free(ast);
}

TEST(test_semantic_two_saves_in_source_order) {
    AstNode *ast = parse_g3_with(
        "output: B, Z;\n"
        "save: Z -> \"z.mtx\";\n"
        "save: B -> \"b.mtx\";\n");
    SemanticResult sem = semantic_analyze(ast);
    ASSERT_INT_EQ(0, sem.error_count);
    ASSERT_INT_EQ(2, sem.symtab->num_saves);
    ASSERT_STR_EQ("Z", sem.symtab->save_tensors[0]);
    ASSERT_STR_EQ("z.mtx", sem.symtab->save_paths[0]);
    ASSERT_STR_EQ("B", sem.symtab->save_tensors[1]);
    ASSERT_STR_EQ("b.mtx", sem.symtab->save_paths[1]);
    semantic_result_free(&sem);
    ast_free(ast);
}

TEST(test_semantic_outputs_default_watch) {
    AstNode *ast = parse(
        "dim N = 3;\n"
        "tensor G[R=N, C=N]  : float, empty = 0.0;\n"
        "tensor F[gen I, R=N] : float, empty = 0.0;\n"
        "tensor P[gen I, R=N] : float, empty = 0.0;\n"
        "init G = [[0.0,1.0,0.0],[0.0,0.0,1.0],[0.0,0.0,0.0]];\n"
        "init F = [1.0, 0.0, 0.0];\n"
        "init P = [1.0, 0.0, 0.0];\n"
        "F[i+1, d] = G[s,d] . F[i,s]   :: map(and, intersect) reduce(or, union);\n"
        "P[i+1, d] = P[i,d] . F[i+1,d] :: map(or, union) reduce(or, union);\n"
        "stop: stable(P);\n");
    SemanticResult sem = semantic_analyze(ast);
    ASSERT_INT_EQ(0, sem.error_count);
    ASSERT_INT_EQ(1, sem.symtab->num_outputs);
    ASSERT_STR_EQ("P", sem.symtab->outputs[0]);
    semantic_result_free(&sem);
    ast_free(ast);
}
```
(`parse` is the existing helper at the top of the file; if `snprintf` needs `<stdio.h>`, it is already included by the test header — check.)

In `tests/unit/test_ir.c`, before `main` (and `RUN_TEST`), using the same G3 text inline (the file has its own parse helper; use it):
```c
TEST(test_ir_outputs_then_saves) {
    AstNode *ast = parse(
        "dim N = 2;\n"
        "tensor A[R=N] : float, empty = 0.0;\n"
        "tensor B[R=N] : float, empty = 0.0;\n"
        "tensor Z[R=N] : float, empty = 0.0;\n"
        "init A = [1.0, 2.0];\n"
        "B[i] = A[i] . A[i] :: map(mul, intersect) reduce(add, union);\n"
        "Z[i] = B[i] . A[i] :: map(add, intersect) reduce(add, union);\n"
        "output: Z, B;\n"
        "save: B -> \"b.mtx\";\n"
        "save: Z -> \"z.mtx\";\n");
    SemanticResult sem = semantic_analyze(ast);
    ASSERT_INT_EQ(0, sem.error_count);
    IRProgram *ir = ir_lower(ast, sem.symtab);
    const char *want[] = { "OUTPUT Z", "OUTPUT B", "SAVE B b.mtx", "SAVE Z z.mtx" };
    int n = 0;
    for (int i = 0; i < ir->count; i++) {
        char got[64];
        if (ir->instrs[i].type == IR_OUTPUT)
            snprintf(got, sizeof(got), "OUTPUT %s", ir->instrs[i].data.output.tensor_name);
        else if (ir->instrs[i].type == IR_SAVE)
            snprintf(got, sizeof(got), "SAVE %s %s", ir->instrs[i].data.save.tensor_name,
                     ir->instrs[i].data.save.path);
        else continue;
        ASSERT_TRUE(n < 4);
        ASSERT_STR_EQ(want[n], got);
        n++;
    }
    ASSERT_INT_EQ(4, n);
    ir_free(ir);
    semantic_result_free(&sem);
    ast_free(ast);
}
```

- [ ] **Step 3: Run and confirm red**

Run: `make -C build 2>&1 | grep -E "error" | head -5; python3 tests/negative/run_negative.py build/edgecc 2>&1 | grep -i "fail\|passed"`
Expected: `test_semantic.c` / `test_ir.c` fail to compile (`num_outputs`, `IR_OUTPUT` undeclared); negatives: the 7 new tests and the 4 reconciled tests FAIL (old messages / programs accepted), all others pass. These are the transiently red tests until Step 5.

- [ ] **Step 4: Implement**

`src/semantic.h`: replace the two fields
```c
    char *save_tensor;   /* tensor named in `save: X -> "..."` (borrowed), or NULL */
    char *save_path;     /* path in the save directive (borrowed), or NULL */
```
with
```c
    /* Tensors run() returns, in return order (borrowed): the `output:` list,
       or the default (the stop-watched tensor, or the last Einsum's output). */
    char *outputs[MAX_TENSORS];
    int num_outputs;
    /* save: directives in source order (borrowed). */
    char *save_tensors[MAX_TENSORS];
    char *save_paths[MAX_TENSORS];
    int num_saves;
```

`src/semantic.c`: replace the whole block that starts with the comment `/* Save directive: target must be a tensor that some Einsum writes (an` and ends after `st->save_path   = s->data.save.path;          /* borrowed */` plus its closing `}` with:
```c
    /* run()'s outputs: the `output:` list if present, else the default (the
       stop-watched tensor, or the output of the last Einsum). */
    AstNode *out_dir = NULL;
    for (int i = 0; i < pd->count; i++) {
        AstNode *s = pd->statements[i];
        if (s->type != NODE_OUTPUT) continue;
        if (out_dir) {
            add_error(&res, "output: only one output directive is allowed");
            continue;
        }
        out_dir = s;
    }
    if (out_dir) {
        for (int k = 0; k < out_dir->data.output.count; k++) {
            char *nm = out_dir->data.output.names[k];
            if (!symtab_find_tensor(st, nm))
                add_error(&res, "output: unknown tensor '%s'", nm);
            else if (!written_by_some_einsum(pd, nm))
                add_error(&res, "output: '%s' is not written by any Einsum "
                                "(inputs are not outputs)", nm);
            else if (name_present(st->outputs, st->num_outputs, nm))
                add_error(&res, "output: '%s' listed twice", nm);
            else if (st->num_outputs < MAX_TENSORS)
                st->outputs[st->num_outputs++] = nm;
        }
    } else {
        char *def = NULL;
        if (st->is_iterative) {
            def = st->stop_watch;
        } else {
            for (int j = 0; j < pd->count; j++) {
                AstNode *s = pd->statements[j];
                if (s->type == NODE_EINSUM)
                    def = s->data.einsum.output.tensor_name;
                else if (s->type == NODE_EINSUM_FUSED)
                    def = s->data.einsum_fused.output.tensor_name;
            }
        }
        if (def) st->outputs[st->num_outputs++] = def;
    }

    /* save: each target is one of run()'s outputs, saved once, to its own path. */
    for (int i = 0; i < pd->count; i++) {
        AstNode *s = pd->statements[i];
        if (s->type != NODE_SAVE) continue;
        char *tgt = s->data.save.tensor_name;
        char *path = s->data.save.path;
        TensorEntry *ste = symtab_find_tensor(st, tgt);
        if (!ste) {
            add_error(&res, "save: unknown tensor '%s'", tgt);
            continue;
        }
        if (st->num_outputs == 0) continue;  /* earlier errors; don't pile on */
        if (!name_present(st->outputs, st->num_outputs, tgt)) {
            add_error(&res, "save: '%s' is not an output of run(); add it to the "
                            "output: list", tgt);
            continue;
        }
        if (name_present(st->save_tensors, st->num_saves, tgt)) {
            add_error(&res, "save: '%s' is saved twice", tgt);
            continue;
        }
        if (name_present(st->save_paths, st->num_saves, path)) {
            add_error(&res, "save: path \"%s\" is used twice", path);
            continue;
        }
        if (ste->gen_axis >= 0 && ste->num_ranks - 1 > 1) {
            add_error(&res, "save: generational tensor '%s' has more than one data rank "
                            "(with every generation it would be 3-D; .mtx stores at most 2-D)", tgt);
            continue;
        }
        if (st->num_saves < MAX_TENSORS) {
            st->save_tensors[st->num_saves] = tgt;
            st->save_paths[st->num_saves]   = path;
            st->num_saves++;
        }
    }
```

`src/ir.h`: add `IR_OUTPUT,` after `IR_SAVE,` in `IRType`; after `IRSave` add
```c
typedef struct {
    char *tensor_name;   /* owned */
} IROutput;
```
and `IROutput output;` to the `IRInstr` union after `IRSave save;`.

`src/ir.c`: replace
```c
    if (symtab->save_tensor) {
        IRInstr *s = ir_emit(ir, IR_SAVE);
        s->data.save.tensor_name = dup_str(symtab->save_tensor);
        s->data.save.path = dup_str(symtab->save_path);
    }
```
with
```c
    for (int k = 0; k < symtab->num_outputs; k++) {
        IRInstr *o = ir_emit(ir, IR_OUTPUT);
        o->data.output.tensor_name = dup_str(symtab->outputs[k]);
    }
    for (int k = 0; k < symtab->num_saves; k++) {
        IRInstr *s = ir_emit(ir, IR_SAVE);
        s->data.save.tensor_name = dup_str(symtab->save_tensors[k]);
        s->data.save.path = dup_str(symtab->save_paths[k]);
    }
```
In `ir_free`, after the `IR_SAVE` case: `case IR_OUTPUT: free(instr->data.output.tensor_name); break;`. In `ir_dump`, after the `IR_SAVE` case:
```c
            case IR_OUTPUT:
                fprintf(out, "OUTPUT(%s)\n", instr->data.output.tensor_name);
                break;
```
`src/codegen.c`, in the `run()` body switch, after `case IR_SAVE: ... break;`:
```c
            case IR_OUTPUT:
                /* Consumed for meta, return and __main__. */
                break;
```
(Codegen still returns the last stored tensor until Task 3; that is expected.)

- [ ] **Step 5: Run everything**

Run: `make -C build 2>&1 | grep -E "error|warning:"; ./build/test_semantic | tail -1; ./build/test_ir | tail -1; python3 tests/negative/run_negative.py build/edgecc 2>&1 | grep -i "fail\|passed"; ctest --test-dir build 2>&1 | tail -3`
Expected: no warnings; all unit tests pass; all negatives pass; `100% tests passed`. Also confirm nothing real is newly rejected:
```bash
for f in examples/*.edge tests/e2e/*/input.edge tests/e2e/iterative/*/kernel.edge tests/reference/kernels/*.edge; do build/edgecc "$f" -o /dev/null >/dev/null 2>&1 || echo "REJECTED $f"; done
```
Expected: no output.

- [ ] **Step 6: Commit**

```bash
git add src/semantic.h src/semantic.c src/ir.h src/ir.c src/codegen.c tests/unit/test_semantic.c tests/unit/test_ir.c tests/negative
git commit -m "semantic/ir: resolve run()'s outputs (output: list or default) and several save: directives

IR_OUTPUT per output in return order, then IR_SAVE per save in source order.
save: must target an output, once per tensor and per path; the old
'save must target the returned tensor' negatives now get the new message."
```

---

### Task 3: Codegen, harnesses and end-to-end tests

**Files:**
- Modify: `src/codegen.c` (`collect_io`, output bookkeeping, meta, tensor-alloc condition, `return`, `__main__`, trace header)
- Modify: `tests/e2e/run_e2e.py` (`run_twice` uses `outputs`; `roundtrip_check` supports several files), `tests/e2e/iterative/run_iterative.py` (single-output guard + `expect_error` marker)
- Test: `tests/unit/test_codegen.c`
- Create: `tests/e2e/{multi_output_save,output_single_directive,output_order,output_intermediate,multi_output_bfs,dft_n4_complex}/`, `tests/e2e/iterative/multi_output_rejected/`

**Interfaces:**
- Consumes: Task 2's `IR_OUTPUT` instructions (`instr->data.output.tensor_name`), in return order, and `IR_SAVE` in source order.
- Produces: generated-code ABI from the spec: `"outputs": [...]` in meta (plus `"output"` only for one output), `return X` / `return (X, Y)`, `X, Y = run(...)`, one `print` per output.

- [ ] **Step 1: Write the codegen unit tests**

In `tests/unit/test_codegen.c`, before `int main` (and `RUN_TEST` each at the end of `main`):
```c
static char *codegen_of(const char *src) {
    AstNode *ast = parse(src);
    SemanticResult sem = semantic_analyze(ast);
    if (sem.error_count != 0) { semantic_result_free(&sem); ast_free(ast); return NULL; }
    IRProgram *ir = ir_lower(ast, sem.symtab);
    char *code = codegen_python(ir);
    ir_free(ir);
    semantic_result_free(&sem);
    ast_free(ast);
    return code;
}

#define G3_PROGRAM \
    "dim N = 2;\n" \
    "tensor A[R=N] : float, empty = 0.0;\n" \
    "tensor B[R=N] : float, empty = 0.0;\n" \
    "tensor Z[R=N] : float, empty = 0.0;\n" \
    "init A = [1.0, 2.0];\n" \
    "B[i] = A[i] . A[i] :: map(mul, intersect) reduce(add, union);\n" \
    "Z[i] = B[i] . A[i] :: map(add, intersect) reduce(add, union);\n"

TEST(test_codegen_multi_output_abi) {
    char *code = codegen_of(G3_PROGRAM "output: Z, B;\n");
    ASSERT_NOT_NULL(code);
    ASSERT_TRUE(strstr(code, "\"outputs\": [\"Z\", \"B\"]") != NULL);
    ASSERT_TRUE(strstr(code, "\"output\":") == NULL);
    ASSERT_TRUE(strstr(code, "    return (Z, B)\n") != NULL);
    ASSERT_TRUE(strstr(code, "    Z, B = run(A)\n    print(Z)\n    print(B)\n") != NULL);
    free(code);
}

TEST(test_codegen_single_output_abi) {
    char *code = codegen_of(G3_PROGRAM);
    ASSERT_NOT_NULL(code);
    ASSERT_TRUE(strstr(code, "\"output\": \"Z\",\n    \"outputs\": [\"Z\"],\n") != NULL);
    ASSERT_TRUE(strstr(code, "    return Z\n") != NULL);
    ASSERT_TRUE(strstr(code, "    Z = run(A)\n    print(Z)\n") != NULL);
    free(code);
}

TEST(test_codegen_saves_follow_prints) {
    char *code = codegen_of(G3_PROGRAM
        "output: B, Z;\n"
        "save: Z -> \"z.mtx\";\n"
        "save: B -> \"b.mtx\";\n");
    ASSERT_NOT_NULL(code);
    ASSERT_TRUE(strstr(code,
        "    print(Z)\n"
        "    _edgecc_save(Z, \"z.mtx\")\n"
        "    _edgecc_save(B, \"b.mtx\")\n") != NULL);
    free(code);
}
```
(`parse` is the existing helper in this file. If the existing tests already define a helper with the same purpose, use it instead of `codegen_of`, and say so in the report.)

- [ ] **Step 2: Write the e2e tests**

Each directory gets `input.edge` and `expected_output.txt`; markers are empty files. G3 is the base program from Task 2.

`tests/e2e/multi_output_save/` (markers `needs_scipy`, `roundtrip_check` — content below):
`input.edge` = `// DFT gap G3: two outputs, both saved.` + G3 +
```
output: B, Z;
save: B -> "b.mtx";
save: Z -> "z.mtx";
```
`expected_output.txt`:
```
[1.0, 4.0]
[2.0, 6.0]
```
`roundtrip_check`:
```
b.mtx: 1.0 4.0
z.mtx: 2.0 6.0
```

`tests/e2e/output_single_directive/`: `// output: Z; is the same as the default.` + G3 + `output: Z;` → `expected_output.txt`: `[2.0, 6.0]`

`tests/e2e/output_order/` (marker `run_twice`): `// Return and print order follow the output: list, not write order.` + G3 + `output: Z, B;` → `expected_output.txt`:
```
[2.0, 6.0]
[1.0, 4.0]
```

`tests/e2e/output_intermediate/`: `// An intermediate (not the last Einsum's output) can be the output.` + G3 + `output: B;` → `expected_output.txt`: `[1.0, 4.0]`

`tests/e2e/multi_output_bfs/input.edge` (generational: both cross-generational tensors, every generation; values by hand: edges 0→1,0→2,1→3,2→3,2→4,3→5,4→5, frontier {0},{1,2},{3,4},{5},{}):
```
// Generational multi-output: frontier F and visited P, each with every generation.
dim N = 6;
tensor G[R=N, C=N]   : float, empty = 0.0;
tensor F[gen I, R=N] : float, empty = 0.0;
tensor P[gen I, R=N] : float, empty = 0.0;
init G = [[0.0,1.0,1.0,0.0,0.0,0.0],
          [0.0,0.0,0.0,1.0,0.0,0.0],
          [0.0,0.0,0.0,1.0,1.0,0.0],
          [0.0,0.0,0.0,0.0,0.0,1.0],
          [0.0,0.0,0.0,0.0,0.0,1.0],
          [0.0,0.0,0.0,0.0,0.0,0.0]];
init F = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
init P = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
F[i+1, d] = G[s,d] . F[i,s]   :: map(and, intersect) reduce(or, union);
P[i+1, d] = P[i,d] . F[i+1,d] :: map(or, union) reduce(or, union);
stop: stable(P);
output: F, P;
```
`expected_output.txt`:
```
[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]
```

`tests/e2e/dft_n4_complex/input.edge` (Re and Im of a 4-point DFT from one kernel; W = exp(-2πi kn/4); x = [1, 2+1j, 3j, 0.5+0.5j]; expected = `numpy.fft.fft(x)` = [3.5+4.5j, 1.5-4.5j, -1.5+1.5j, 0.5-1.5j]):
```
// DFT gap G3: Re and Im of a 4-point DFT from one kernel.
// x = [1, 2+1j, 3j, 0.5+0.5j]; expected = numpy.fft.fft(x).
dim N = 4;
tensor Wre[R=N, C=N] : float, empty = 0.0;
tensor Wim[R=N, C=N] : float, empty = 0.0;
tensor Xre[R=N] : float, empty = 0.0;
tensor Xim[R=N] : float, empty = 0.0;
tensor P1[R=N] : float, empty = 0.0;
tensor P2[R=N] : float, empty = 0.0;
tensor P3[R=N] : float, empty = 0.0;
tensor P4[R=N] : float, empty = 0.0;
tensor Yre[R=N] : float, empty = 0.0;
tensor Yim[R=N] : float, empty = 0.0;
init Wre = [[1.0, 1.0, 1.0, 1.0], [1.0, 0.0, -1.0, 0.0], [1.0, -1.0, 1.0, -1.0], [1.0, 0.0, -1.0, 0.0]];
init Wim = [[0.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, -1.0]];
init Xre = [1.0, 2.0, 0.0, 0.5];
init Xim = [0.0, 1.0, 3.0, 0.5];
P1[k] = Wre[k,n] . Xre[n] :: map(mul, intersect) reduce(add, union);
P2[k] = Wim[k,n] . Xim[n] :: map(mul, intersect) reduce(add, union);
P3[k] = Wre[k,n] . Xim[n] :: map(mul, intersect) reduce(add, union);
P4[k] = Wim[k,n] . Xre[n] :: map(mul, intersect) reduce(add, union);
Yre[k] = P1[k] . P2[k] :: map(sub, union) reduce(add, union);
Yim[k] = P3[k] . P4[k] :: map(add, union) reduce(add, union);
output: Yre, Yim;
```
`expected_output.txt`:
```
[3.5, 1.5, -1.5, 0.5]
[4.5, -4.5, 1.5, -1.5]
```
(The kernel math was checked on `main` as two single-output programs; each printed exactly its line above.)

`tests/e2e/iterative/multi_output_rejected/`:
`kernel.edge`:
```
// The iterative driver feeds run()'s single result back; a multi-output
// kernel must be rejected with a clear message.
dim S = 3;
dim D = 3;
tensor G[R=S, C=D] : float, empty = inf;
tensor F[R=S]      : float, empty = inf;
tensor F1[R=D]     : float, empty = inf;
tensor H[R=D]      : float, empty = inf;
init G = [[0.0, 1.0, inf], [inf, 0.0, 1.0], [inf, inf, 0.0]];
init F = [0.0, inf, inf];
F1[d] = G[s,d] . F[s] :: map(add, intersect) reduce(min, union);
H[d]  = F1[d] . F1[d] :: map(min, intersect) reduce(min, union);
output: F1, H;
```
`driver.yaml`:
```yaml
inputs:
  G: [[0.0, 1.0, .inf], [.inf, 0.0, 1.0], [.inf, .inf, 0.0]]
  F: [0.0, .inf, .inf]
feedback:
  to: F
  combine: min
termination:
  type: stable
  watch: F
max_iterations: 5
expected:
  final:
    F: [0.0, 1.0, 2.0]
```
`expect_error`:
```
iterative driver needs a single-output kernel
```

- [ ] **Step 3: Update the harnesses**

`tests/e2e/run_e2e.py`, `run_twice` snippet: replace the line `"first = g[m['output']]\n"` and the `second = ...` line with:
```python
                "outs = m['outputs']\n"
                "first = g[outs[0]] if len(outs) == 1 else tuple(g[n] for n in outs)\n"
                "second = g['run'](*[g[n] for n in m['inputs']])\n"
```
(keep the `if first != second:` lines).

`tests/e2e/run_e2e.py`, round-trip check: replace the body under `if os.path.exists(rt) and HAVE_SCIPY:` with a version that accepts either the old format (whitespace-separated numbers checked against `result.mtx`) or one `file.mtx: numbers` line per saved file:
```python
            from scipy.io import mmread
            with open(rt) as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            if any(":" in ln for ln in lines):
                checks = [(ln.split(":", 1)[0].strip(), ln.split(":", 1)[1]) for ln in lines]
            else:
                checks = [("result.mtx", " ".join(lines))]
            for fname, nums in checks:
                saved = os.path.join(test_dir, fname)
                if not os.path.exists(saved):
                    print(f"  FAIL {name}: expected saved {fname} not found")
                    return False
                m = mmread(saved)
                arr = m.toarray() if hasattr(m, "toarray") else m
                vals = arr.flatten().tolist()
                want = [float(x) for x in nums.split()]
                os.unlink(saved)
                if vals != want:
                    print(f"  FAIL {name}: {fname} round-tripped {vals} != expected {want}")
                    return False
```

`tests/e2e/iterative/run_iterative.py`, in `run_test`, directly after the `if not meta:` block:
```python
        outputs = meta.get("outputs", [meta.get("output")])
        if len(outputs) != 1:
            msg = ("iterative driver needs a single-output kernel; "
                   "got outputs %s" % outputs)
            expect = os.path.join(test_dir, "expect_error")
            if os.path.exists(expect):
                with open(expect) as f:
                    want = f.read().strip()
                if want in msg:
                    print(f"  PASS {name} (rejected as expected)")
                    return True
            print(f"  FAIL {name}: {msg}")
            return False
```
and, so an `expect_error` test cannot pass by accident, directly before the final `print(f"  PASS {name}")`:
```python
        if os.path.exists(os.path.join(test_dir, "expect_error")):
            print(f"  FAIL {name}: expected an error, but the driver ran")
            return False
```

- [ ] **Step 4: Run and confirm red for the right reasons**

Run: `make -C build && ./build/test_codegen | tail -3; python3 tests/e2e/run_e2e.py build/edgecc 2>&1 | grep -i "fail\|passed"; python3 tests/e2e/iterative/run_iterative.py build/edgecc 2>&1 | tail -3`
Expected:
- `test_codegen`: the three new tests FAIL (no `"outputs"` key, `return Z` only).
- e2e: `multi_output_save`, `output_order`, `multi_output_bfs`, `dft_n4_complex` FAIL (only the last output / watched tensor is printed; `__main__` saves `B`, which it never defined → `NameError`); `output_single_directive` PASSES (same as default); `output_intermediate` FAILS (prints `[2.0, 6.0]`). The existing `run_twice` tests (`intermediate_repeatable`, `intermediate_inferred_dim`) also FAIL with `KeyError: 'outputs'`, because the harness now reads `meta["outputs"]`, which codegen does not emit until Step 5.
- iterative: `multi_output_rejected` FAILS (no `"outputs"` in meta yet, so the guard does not fire and the expect_error check reports "expected an error").
These are the transiently red tests until Step 6.

- [ ] **Step 5: Implement in `src/codegen.c`**

(a) Replace `collect_io` (and its comment) with a version without the output out-parameter:
```c
/* Read tensors (IR loads, in load order) go to `inputs`; every tensor an
   Einsum stores to goes to `written`. Caller frees both lists. */
static void collect_io(IRProgram *ir, TensorNameList *inputs, TensorNameList *written) {
    inputs->names = NULL;  inputs->count = 0;  inputs->cap = 0;
    written->names = NULL; written->count = 0; written->cap = 0;
    for (int i = 0; i < ir->count; i++) {
        IRInstr *instr = &ir->instrs[i];
        if (instr->type == IR_LOAD) {
            add_input_name(inputs, instr->data.load.tensor_name);
        } else if (instr->type == IR_STORE) {
            add_input_name(written, instr->data.store.tensor_name);
        } else if (instr->type == IR_FUSED) {
            IRFused *fu = &instr->data.fused;
            add_input_name(inputs, fu->inner1.name);
            add_input_name(inputs, fu->inner2.name);
            add_input_name(inputs, fu->outer.name);
            add_input_name(written, fu->out_name);
        }
    }
}
```

(b) In `codegen_python_ex`, replace
```c
    TensorNameList inputs, written;
    char *output = NULL;
    collect_io(ir, &inputs, &written, &output);
```
with
```c
    TensorNameList inputs, written;
    collect_io(ir, &inputs, &written);

    /* run()'s outputs, in return order (names borrowed from the IR). */
    int num_outs = 0;
    for (int i = 0; i < ir->count; i++)
        if (ir->instrs[i].type == IR_OUTPUT) num_outs++;
    const char **outs = calloc(num_outs ? (size_t)num_outs : 1, sizeof(const char *));
    num_outs = 0;
    for (int i = 0; i < ir->count; i++)
        if (ir->instrs[i].type == IR_OUTPUT)
            outs[num_outs++] = ir->instrs[i].data.output.tensor_name;
```

(c) In the `if (iterative) { ... }` cascade I/O block, delete
```c
        if (watch) {
            if (output) free(output);
            output = strdup(watch);
        }
```
and change that block's leading comment to `/* Cascade I/O: intermediates are never parameters. */`.

(d) Replace the meta line `cb_appendf(&cb, "    \"output\": \"%s\",\n", output ? output : "");` with:
```c
    if (num_outs == 1)
        cb_appendf(&cb, "    \"output\": \"%s\",\n", outs[0]);
    cb_append(&cb, "    \"outputs\": [");
    for (int k = 0; k < num_outs; k++) {
        if (k > 0) cb_append(&cb, ", ");
        cb_appendf(&cb, "\"%s\"", outs[k]);
    }
    cb_append(&cb, "],\n");
```

(e) In the `IR_TENSOR_ALLOC` case, the condition's `output` clause only ever matched in non-generational programs, where `written` already covers it. Replace the condition with:
```c
                if (!iterative && name_in_tnl(&written, instr->data.tensor_alloc.name)) {
```

(f) Replace the `run()` return block
```c
    if (output) {
        cb_indent(&cb);
        if (iterative && watch) {
            cb_appendf(&cb, "return %s\n", watch);   /* every generation */
        } else {
            cb_appendf(&cb, "return %s\n", output);
        }
    }
```
with
```c
    /* A generational output is returned with every generation. */
    if (num_outs > 0) {
        cb_indent(&cb);
        if (num_outs == 1) {
            cb_appendf(&cb, "return %s\n", outs[0]);
        } else {
            cb_append(&cb, "return (");
            for (int k = 0; k < num_outs; k++) {
                if (k > 0) cb_append(&cb, ", ");
                cb_append(&cb, outs[k]);
            }
            cb_append(&cb, ")\n");
        }
    }
```

(g) In `__main__`, replace `if (output) {` … `cb_appendf(&cb, "%s = run(", output);` with the unpacking form, and the single `print`:
```c
    if (num_outs > 0) {
        if (emit_trace) { cb_indent(&cb); cb_append(&cb, "_tr = []\n"); }
        cb_indent(&cb);
        for (int k = 0; k < num_outs; k++) {
            if (k > 0) cb_append(&cb, ", ");
            cb_append(&cb, outs[k]);
        }
        cb_append(&cb, " = run(");
```
(the argument loop and `_trace=_tr` stay as they are), and replace `cb_indent(&cb); cb_appendf(&cb, "print(%s)\n", output);` with:
```c
        for (int k = 0; k < num_outs; k++) {
            cb_indent(&cb);
            cb_appendf(&cb, "print(%s)\n", outs[k]);
        }
```

(h) In the trace header, `"\"dims\": {\"N\": len(%s[0])}, ..."` is formatted with `output`; pass `watch` instead (the trace is generational-only, so `watch` is set):
```c
                kernel_name, watch);
```

(i) At the end, replace `free(output);` with `free(outs);`.

Build must be warning-free; grep `codegen.c` for any remaining `output` variable use (`grep -n -w output src/codegen.c`) — only comments and the `"output"` meta string may remain.

- [ ] **Step 6: Run everything**

Run: `make -C build 2>&1 | grep -E "error|warning:"; ./build/test_codegen | tail -1; python3 tests/e2e/run_e2e.py build/edgecc 2>&1 | grep -i "fail\|passed"; python3 tests/e2e/iterative/run_iterative.py build/edgecc 2>&1 | tail -2; ctest --test-dir build 2>&1 | tail -3`
Expected: no warnings; `test_codegen` all pass (existing asserts `"output": "Z"`, `"output": "D"`, `"output": "F"` still hold); all e2e pass; iterative passes including `multi_output_rejected (rejected as expected)`; `100% tests passed, 0 tests failed out of 14` (trace and viz tests included).

- [ ] **Step 7: Commit**

```bash
git add src/codegen.c tests/unit/test_codegen.c tests/e2e
git commit -m "codegen: run() returns every listed output (tuple), __main__ prints one line each (DFT gap G3)

meta gains \"outputs\" (\"output\" only for a single output); saves of any
output work in __main__. e2e: run_twice compares the whole return value;
roundtrip_check accepts one 'file.mtx: values' line per saved file.
Iterative driver rejects multi-output kernels (expect_error marker)."
```

---

### Task 4: VS Code highlighting for directive keywords

**Files:**
- Modify: `editors/vscode/syntaxes/edge.tmLanguage.json`
- Create: `editors/vscode/test/grammar/directives.test.edge`

**Interfaces:** none (editor only).

`stop`, `save` and the new `output` are all statement-leading directives and none is highlighted today. Add one pattern for the three so `output` is not the odd one out.

- [ ] **Step 1: Write the failing grammar test**

`editors/vscode/test/grammar/directives.test.edge` (keep the header exactly; the `^` columns must line up under the token — check them character by character):
```
// SYNTAX TEST "source.edge" "directive keywords"
output: B, Z;
// <- keyword.control.directive.edge
save: Z -> "z.mtx";
// <- keyword.control.directive.edge
stop: stable(P);
// <- keyword.control.directive.edge
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `cd editors/vscode && npm run test:grammar 2>&1 | tail -15`
Expected: `directives.test.edge` fails (scopes are `variable.other.edge` or none); the other grammar fixtures pass. (If `node_modules` is missing, run `npm install` first.)

- [ ] **Step 3: Implement**

In `edge.tmLanguage.json`, add to `repository`:
```json
"directives": {
  "match": "\\b(output|save|stop)\\b(?=\\s*:)",
  "name": "keyword.control.directive.edge"
}
```
and insert `{ "include": "#directives" }` into the top-level `patterns` array directly before `{ "include": "#declarations" }`.

- [ ] **Step 4: Run the grammar and unit tests**

Run: `cd editors/vscode && npm run test:grammar 2>&1 | tail -5 && npm run test:unit 2>&1 | tail -5`
Expected: all grammar fixtures pass (including the new one); unit tests pass. (The integration test needs a VS Code download; skip it unless already cached.)

- [ ] **Step 5: Commit**

```bash
git add editors/vscode/syntaxes/edge.tmLanguage.json editors/vscode/test/grammar/directives.test.edge
git commit -m "vscode: highlight the output/save/stop directive keywords"
```

---

### Task 5: Docs, DFT acceptance, and Finish

**Files:**
- Modify: `CLAUDE.md`, `docs/roadmap.md`, the spec (deviations section), this plan (checkboxes)

**Interfaces:**
- Consumes: Tasks 1–4 on the branch.

- [ ] **Step 1: Update `CLAUDE.md`**

(a) In "### File input/output", replace the sentence that begins `` `T` must be the tensor `run()` returns: `` and ends `(1-D results are stored as a column).` with:
`` `save:` may appear several times; each target must be one of `run()`'s outputs (see "Outputs" below), at most once per tensor and per path. A generational output is written with every generation (generations x N for a 1-D generational tensor; tensors with more than one data rank cannot be saved); 1-D results are stored as a column. ``

(b) Add a paragraph at the end of "## EDGE Syntax (v2)" (before "### Generational (iterative) ranks"):
`` **Outputs.** `output: B, Z;` lists the tensors `run()` returns, in order (at most one `output:` directive; each name a tensor written by some Einsum, listed once). Without it, `run()` returns the stop-watched tensor (generational) or the last Einsum's output. A generational output is returned with every generation. ``

(c) In "## Generated Python ABI", after the sentence ending `…and \`__main__\` prints it.`, add:
`` With several outputs, `run()` returns a tuple in `output:` order and `__main__` unpacks it and prints one line per output; `__edgecc_meta__["outputs"]` lists the names in return order, and `"output"` is present only when there is exactly one. The iterative driver requires a single-output kernel. ``
Also in the ABI code sample change `__edgecc_meta__ = {"inputs": ["G", "F"], "output": "F1"}` to `__edgecc_meta__ = {"inputs": ["G", "F"], "output": "F1", "outputs": ["F1"]}`.

(d) In "## v2 Scope Limitations", in the "Not yet supported" paragraph: change `(multi-output return and \`ANY\` compute;` to `(\`ANY\` compute;`, and delete `` `save:` of any tensor other than the one `run()` returns, ``.

(e) In the e2e line of "## Test", the `roundtrip_check` marker is not documented; append: `` and `roundtrip_check` (re-reads saved `.mtx` files: whitespace-separated values for `result.mtx`, or one `file.mtx: values` line per file) ``. In the iterative line, append: `` An `expect_error` file holds a message the driver must reject the kernel with. ``

- [ ] **Step 2: Update `docs/roadmap.md`**

Set `**Last updated:**` to the current date. Under "### Implemented", after the batch B bullet, add:
`` - **Multi-output `run()` (DFT gap G3, batch C):** `output: B, Z;` returns a tuple in listed order; `save:` may appear once per output; meta `"outputs"`. ``
In "### Partial 🚧" → BFS, change `the full 3-step formulation still needs multi-output return, and \`ANY\`/parent-pointer.` to `the full 3-step formulation still needs \`ANY\`/parent-pointer.`
In "Known issues & tech debt", add any review findings from this branch that were not fixed (deferred minors, parked findings, reviewer "declined to judge" items worth keeping), each tagged `(batch C)`.

- [ ] **Step 3: DFT acceptance (read-only use of the repo root)**

Mirror the DFT files into a scratch directory so `dft_ref.py` uses this branch's compiler (use the session scratchpad; never write into the repo root):
```bash
cd <this worktree>
S=<scratch dir>
ROOT=/Users/porumbes/Dropbox/Home/Serban/Projects/Owens_Group/Researchers/Toluwanimi_Odemuyiwa/code/edgecc
mkdir -p $S/repo/docs $S/repo/build
cp -R $ROOT/docs/dft $S/repo/docs/
ln -s "$PWD/build/edgecc" $S/repo/build/edgecc
cd $S/repo
K=docs/dft/kernels
python3 docs/dft/reference/dft_ref.py check $K/dft_naive_re.edge $K/dft_naive_im.edge --n 16
python3 docs/dft/reference/dft_ref.py check $K/dft_naive_n4_re.edge $K/dft_naive_n4_im.edge --x "1, 2+1j, 3j, 0.5+0.5j"
python3 docs/dft/reference/dft_ref.py check $K/dft_naive_n4_packed.edge --x "1, 2+1j, 3j, 0.5+0.5j"
python3 docs/dft/reference/dft_ref.py check $K/dft_stacked_flat.edge --layout interleaved --n 16
python3 docs/dft/reference/dft_ref.py check $K/dft_stacked.edge --layout cols --n 16 --direct
python3 docs/dft/reference/dft_ref.py check $K/dft_stacked_jx.edge --layout cols --n 16 --direct
python3 docs/dft/reference/dft_ref.py check $K/fft_split1.edge --layout split --n 16 --direct
```
Run under `bash` (unquoted variable expansion differs in zsh). Expected: every line ends `-> PASS` (all these kernels are single-output, so they are unaffected). If one fails, stop and report rather than editing DFT files. Before running, check `$ROOT/docs/dft/README.md` for the current quickstart; if its commands differ, run the README's commands and say so.

- [ ] **Step 4: Pre-merge gate**

Run: `tools/premerge.sh` (≈1 minute; long timeout). Expected: `GATE PASSED (6 steps)`. Afterwards check `ps aux | grep -E 'edgecc|run_fuzz|build/test_'` and kill any strays by PID.

- [ ] **Step 5: Record deviations, tick the plan, commit**

Append `## Deviations and findings (recorded at Finish, <date>)` to the spec. Known deviation: Task 4 also highlights `save` and `stop` (none of the three directives was highlighted). Add anything else that differed ("None." otherwise). Tick every checkbox of Tasks 1–4 and Task 5 Steps 1–5 (edit the boxes, not the header prose). Then:
```bash
git status --short     # no stray files (e.g. *-e backups)
git add CLAUDE.md docs/roadmap.md docs/superpowers
git commit -m "docs: multi-output run() (batch C) — CLAUDE.md, roadmap, spec deviations, plan ticked"
```

- [ ] **Step 6: Merge from the `main` worktree**

```bash
cd ../premerge-gate          # .worktrees/premerge-gate has main checked out
git merge --no-ff feature/multi-output -m "Merge feature/multi-output: multi-output run() and multiple save: (DFT gap G3, batch C)"
make -C build && ctest --test-dir build 2>&1 | grep "tests passed"
```
Expected: `100% tests passed`.

- [ ] **Step 7: Memory and hand-off**

Update the auto-memory `project_resume_state.md` (main's new hash, batch C done, next = batch D: 1-D / N-D load, gaps G8/G9) and tell the user that `main` moved so the DFT session can merge it: a single kernel can now return Re and Im (`output: Yre, Yim;`), `dft_ref.py --direct` receives a tuple from such a kernel, and its printed-output decoder must read one line per output.
