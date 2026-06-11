# Multi-action (parenthesized) Einsums Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let one Einsum statement fuse a parenthesized binary sub-expression and an outer binary action — `(A .1 B) .2 C` — so the paper's `(G·F)·!P` masked-frontier gather is one line instead of a 3-statement cascade.

**Architecture:** A new AST node `NODE_EINSUM_FUSED` (the single-action path is untouched), parsed from labeled `.N` operators + `map.N`/`reduce.N` specs with an optional `over=` rank tag (also added, optionally, to single-action `reduce`). Semantic analysis classifies indices per action (intermediate = `inner ∩ (outer ∪ output)`; action-1/2 reduce sets) and validates. A structured `IR_FUSED` instruction lowers it; a dedicated codegen emitter produces the two-action loop nest with a **transient scalar** intermediate (no temp tensor), reusing the existing merge-guard/compute helpers.

**Tech Stack:** C99, Flex, Bison (2.3-compatible), CMake/CTest; Python 3 for generated code and e2e/negative harnesses.

**Spec:** `docs/superpowers/specs/2026-06-10-edgecc-multi-action-einsums-design.md`

---

## Background the engineer needs

- **Pipeline:** `.edge → lexer.l → parser.y → AST (ast.h/c) → semantic.c → ir.c → codegen.c → .py`.
- **Build:** `cmake -S . -B build >/dev/null && cmake --build build` (regenerates lexer/parser).
- **All tests:** `ctest --test-dir build --output-on-failure`. One unit binary: `./build/test_parser` (also `test_lexer`, `test_ir`, `test_codegen`, `test_semantic`).
- **Unit tests** in `tests/unit/*.c`: `TEST(name){...}` macro, registered with `RUN_TEST(name)` in that file's `main()`. Lexer tests drive `yylex()`/`yy_scan_string`; parser tests call `yyparse()` and read global `parse_result`; ir/codegen/semantic tests use a local `parse()` helper then `semantic_analyze`/`ir_lower`/`codegen_python`.
- **E2E tests:** dir `tests/e2e/<name>/` with `input.edge` + `expected_output.txt` (auto-discovered; stdout compared after `.strip()`).
- **Negative tests:** dir `tests/negative/<name>/` with `input.edge` + `expected_errors.txt` (runs `edgecc input.edge -o /dev/null`, requires non-zero exit, each expected line must appear in stderr).
- **Existence model:** a coordinate exists iff `value != empty`. The merge guard works on existence Booleans (`codegen.c` `emit_general_skeleton`, helpers `emit_merge_guard`, `compute_op_pyexpr`).
- **Generational kernels** (BFS): the gen loop and `_prev`/`_next` double-buffering are emitted at program level (`IR_GEN_LOOP_BEGIN/END`); each operand load carries a `gen_read` kind (0 none / 1 `_prev` / 2 `_next`) and the store a `gen_write` flag. A fused einsum participates exactly like a single-action one — it just sets those flags on its operands/output.
- **Commit convention:** do NOT add `Co-Authored-By` trailers.

### Key reference shapes (current code)

`EinsumData` (`src/ast.h`) — single action: `output, input1, input2, map_op, map_merge_op, reduce_op, reduce_merge_op`.

`EinsumClass` (`src/semantic.h`) — per-einsum classification: `free_indices[]/num_free_indices`, `reduction_indices[]/num_reduction_indices`.

`emit_einsum` (`src/ir.c`) emits `LOOP_BEGIN(free…) LOOP_BEGIN(reduction…) LOAD LOAD MAP REDUCE STORE LOOP_END…`. `IRLoad` has `reg, tensor_name, coords, ncoords, gen_read, negate`.

`emit_general_skeleton` (`src/codegen.c`) emits one map+reduce body for the `STORE` window.

### Generated-code targets (what the fused emitter must produce)

**push** — action 1 reduces `s`, action 2 map-only (`(G.1 F).2 !P`):
```python
for d in range(N):
    _acc1 = <reduce.1 identity>
    for s in range(N):
        a_val = G[s][d]; b_val = F_prev[s]
        a_exists = a_val != 0.0; b_exists = b_val != 0.0
        if a_exists and b_exists:                 # map.1 merge=intersect
            a_op = a_val if a_exists else 0.0
            b_op = b_val if b_exists else 0.0
            _m1 = a_op and b_op                   # map.1 op=and
            _m1_emit = _m1 != 0.0
        else:
            _m1_emit = False; _m1 = 0.0
        _s1 = _acc1 != 0.0; _s2 = _m1_emit
        if _s1 or _s2:                            # reduce.1 merge=union
            _vfr = _m1 if _m1_emit else 0.0
            _acc1 = _acc1 or _vfr                 # reduce.1 op=or
        else:
            _acc1 = 0.0
    _inter = _acc1
    _inter_exists = _inter != 0.0
    c_val = P_prev[d]; c_exists = c_val == 0.0    # !P  -> negate flips to ==
    if _inter_exists and c_exists:                # map.2 merge=intersect
        l_op = _inter if _inter_exists else 0.0
        r_op = 1.0 if c_exists else 0.0           # negated operand -> tau 1.0
        _m2 = l_op                                # map.2 op=take_left
        F_next[d] = _m2 if _m2 != 0.0 else 0.0
    else:
        F_next[d] = 0.0
```

**pull** — action 1 map-only, action 2 reduces `s` (`(G.1 !P).2 F`):
```python
for d in range(N):
    _acc2 = <reduce.2 identity>
    for s in range(N):
        a_val = G[s][d]; b_val = P_prev[d]
        a_exists = a_val != 0.0; b_exists = b_val == 0.0   # !P
        if a_exists and b_exists:                          # map.1 merge=intersect
            a_op = a_val if a_exists else 0.0
            b_op = 1.0 if b_exists else 0.0
            _inter = a_op and b_op                         # map.1 op=and
            _inter_emit = _inter != 0.0
        else:
            _inter_emit = False; _inter = 0.0
        # action 2 map: operands _inter and outer F_prev[s]
        c_val = F_prev[s]; c_exists = c_val != 0.0
        if _inter_emit and c_exists:                       # map.2 merge=intersect
            l_op = _inter if _inter_emit else 0.0
            r_op = c_val if c_exists else 0.0
            _m2 = l_op and c_val                           # map.2 op=and  (placeholder shows op)
            _m2_emit = _m2 != 0.0
        else:
            _m2_emit = False; _m2 = 0.0
        _s1 = _acc2 != 0.0; _s2 = _m2_emit
        if _s1 or _s2:                                     # reduce.2 merge=union
            _vfr = _m2 if _m2_emit else 0.0
            _acc2 = _acc2 or _vfr                          # reduce.2 op=or
        else:
            _acc2 = 0.0
    F_next[d] = _acc2
```

Both-actions-reduce (matrix-chain) is **out of scope**: a clear semantic error (Task 4). Neither-reduces (both map-only) is supported (no reduction loops; `_inter` then store).

---

## File Structure

- `src/lexer.l` — add `over` keyword token.
- `src/ast.h` / `src/ast.c` — `NODE_EINSUM_FUSED` + `EinsumFusedData` + constructor + free; add optional `reduce_over` to `EinsumData`.
- `src/parser.y` — labeled fused production, `map.N`/`reduce.N` action list, `over=` arg (fused + single-action).
- `src/semantic.h` / `src/semantic.c` — extend `EinsumClass` with fused sets; `analyze_einsum_fused`; `over=` validation.
- `src/ir.h` / `src/ir.c` — `IR_FUSED` instruction + `emit_einsum_fused`.
- `src/codegen.c` — `emit_fused_skeleton` + dispatch in the main IR walk.
- `tests/unit/*.c`, `tests/e2e/*`, `tests/negative/*` — coverage.
- `examples/push_bfs_fused.edge`, `examples/pull_bfs_fused.edge`.
- `CLAUDE.md`, `docs/roadmap.md` — reconciliation.

---

## Task 1: Lexer — `over` keyword

**Files:** Modify `src/lexer.l`; Test `tests/unit/test_lexer.c`. (Also add `%token TOK_OVER` in Task 3's parser work — but the lexer references `TOK_OVER` from `parser.tab.h`, so this task's build will fail until Task 3 declares the token. To keep Task 1 self-contained, declare the token here too by adding the `%token` line to `parser.y` now.)

- [ ] **Step 1: Declare the token (parser) and write the failing lexer test**

In `src/parser.y`, add `TOK_OVER` to the token list (the line beginning `%token TOK_LOAD TOK_SAVE ...`):
```
%token TOK_LOAD TOK_SAVE TOK_ARROW TOK_SEED TOK_OVER
```
Add to `tests/unit/test_lexer.c` (after `test_bang_token`) and register in `main()`:
```c
TEST(test_over_keyword) {
    YY_BUFFER_STATE buf = yy_scan_string("over");
    ASSERT_INT_EQ(TOK_OVER, yylex());
    yy_delete_buffer(buf);
}
```
Registration: `    RUN_TEST(test_over_keyword);`

- [ ] **Step 2: Run to verify it fails**

Run: `cmake --build build && ./build/test_lexer`
Expected: FAIL — `over` currently lexes as `TOK_IDENT`, not `TOK_OVER`.

- [ ] **Step 3: Add the lexer rule**

In `src/lexer.l`, with the other keyword rules (after `"seed"`):
```
"over"              { return TOK_OVER; }
```

- [ ] **Step 4: Verify pass**

Run: `cmake --build build && ./build/test_lexer`
Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git add src/lexer.l src/parser.y tests/unit/test_lexer.c
git commit -m "lexer: add 'over' keyword for reduce rank tags"
```

---

## Task 2: AST — fused node + optional single-action over

**Files:** Modify `src/ast.h`, `src/ast.c`. Test: covered by Task 3 parser tests (struct-only changes).

- [ ] **Step 1: Add the node type and struct**

In `src/ast.h`, add to `NodeType`:
```c
    NODE_EINSUM,
    NODE_EINSUM_FUSED,
    NODE_STOP,
```
Add `char *reduce_over;` to `EinsumData` (after `reduce_merge_op`):
```c
    MergeOp      reduce_merge_op;    /* defaults to MERGE_PASSTHROUGH */
    char        *reduce_over;        /* optional `over=` rank for single-action reduce; NULL if none */
} EinsumData;
```
Add the fused struct just after `EinsumData`:
```c
typedef struct {
    TensorRef output;
    TensorRef inner1;        /* paren left operand  (action 1 left) */
    TensorRef inner2;        /* paren right operand (action 1 right) */
    TensorRef outer;         /* the non-paren outer operand */
    int       inter_is_left2;/* 1 if the intermediate is action 2's LEFT operand */
    /* action 1 (inner, label .1) */
    OperatorType map1_op;    MergeOp map1_merge;
    int          a1_has_reduce;
    OperatorType reduce1_op; MergeOp reduce1_merge;
    char        *over1;      /* optional `over=` rank, else NULL */
    /* action 2 (outer, label .2) */
    OperatorType map2_op;    MergeOp map2_merge;
    int          a2_has_reduce;
    OperatorType reduce2_op; MergeOp reduce2_merge;
    char        *over2;      /* optional `over=` rank, else NULL */
} EinsumFusedData;
```
Add to the `AstNode` union (after `EinsumData einsum;`):
```c
        EinsumFusedData einsum_fused;
```
Declare the constructor (after `ast_einsum_new`):
```c
AstNode *ast_einsum_fused_new(EinsumFusedData data);
```

- [ ] **Step 2: Implement constructor, init, and free**

In `src/ast.c`, in `ast_einsum_new`, initialize the new field (after the struct is populated, before return — find where `EinsumData` is filled and add):
```c
    node->data.einsum.reduce_over = NULL;
```
Add the constructor (near `ast_einsum_new`):
```c
AstNode *ast_einsum_fused_new(EinsumFusedData data) {
    AstNode *node = malloc(sizeof(AstNode));
    node->type = NODE_EINSUM_FUSED;
    node->data.einsum_fused = data;
    return node;
}
```
In `ast_free`, find the `NODE_EINSUM` case and extend it / add the fused case. The `NODE_EINSUM` case must also free `reduce_over`:
```c
        case NODE_EINSUM:
            tensor_ref_free_contents(&node->data.einsum.output);
            tensor_ref_free_contents(&node->data.einsum.input1);
            tensor_ref_free_contents(&node->data.einsum.input2);
            free(node->data.einsum.reduce_over);
            break;
        case NODE_EINSUM_FUSED: {
            EinsumFusedData *f = &node->data.einsum_fused;
            tensor_ref_free_contents(&f->output);
            tensor_ref_free_contents(&f->inner1);
            tensor_ref_free_contents(&f->inner2);
            tensor_ref_free_contents(&f->outer);
            free(f->over1);
            free(f->over2);
            break;
        }
```
(If the existing `NODE_EINSUM` case already frees the three refs without `reduce_over`, just add the `free(node->data.einsum.reduce_over);` line.)

- [ ] **Step 3: Verify build**

Run: `cmake --build build`
Expected: builds clean (parser doesn't yet produce the node; no behavior change).

- [ ] **Step 4: Commit**
```bash
git add src/ast.h src/ast.c
git commit -m "ast: add NODE_EINSUM_FUSED and optional single-action reduce_over"
```

---

## Task 3: Parser — fused grammar + `over=` arg

**Files:** Modify `src/parser.y`; Test `tests/unit/test_parser.c`.

This task adds: (a) an `opt_over` arg, (b) `over=` on the existing single-action reduce, (c) the fused production with a label-dispatched action list.

- [ ] **Step 1: Write the failing parser tests**

Add to `tests/unit/test_parser.c` (after the negation tests) and register all three in `main()`:
```c
TEST(test_parse_single_reduce_over) {
    YY_BUFFER_STATE buf = yy_scan_string(
        "Z[d] = G[s,d] . F[s] :: map(mul,intersect) reduce(add,union, over=s);");
    ASSERT_INT_EQ(0, yyparse());
    AstNode *es = parse_result->data.program.statements[0];
    ASSERT_INT_EQ(NODE_EINSUM, es->type);
    ASSERT_NOT_NULL(es->data.einsum.reduce_over);
    ASSERT_STR_EQ("s", es->data.einsum.reduce_over);
    ast_free(parse_result); parse_result = NULL; yy_delete_buffer(buf);
}

TEST(test_parse_fused_push) {
    YY_BUFFER_STATE buf = yy_scan_string(
        "F[d] = (G[s,d] .1 F[s]) .2 P[d] "
        ":: map.1(and,intersect) reduce.1(or,union) map.2(take_left,intersect);");
    ASSERT_INT_EQ(0, yyparse());
    AstNode *es = parse_result->data.program.statements[0];
    ASSERT_INT_EQ(NODE_EINSUM_FUSED, es->type);
    EinsumFusedData *f = &es->data.einsum_fused;
    ASSERT_STR_EQ("F", f->output.tensor_name);
    ASSERT_STR_EQ("G", f->inner1.tensor_name);
    ASSERT_STR_EQ("F", f->inner2.tensor_name);
    ASSERT_STR_EQ("P", f->outer.tensor_name);
    ASSERT_INT_EQ(1, f->inter_is_left2);     /* (inner) .2 outer */
    ASSERT_INT_EQ(OP_AND, f->map1_op);
    ASSERT_INT_EQ(MERGE_INTERSECT, f->map1_merge);
    ASSERT_INT_EQ(1, f->a1_has_reduce);
    ASSERT_INT_EQ(OP_OR, f->reduce1_op);
    ASSERT_INT_EQ(OP_TAKE_LEFT, f->map2_op);
    ASSERT_INT_EQ(0, f->a2_has_reduce);
    ast_free(parse_result); parse_result = NULL; yy_delete_buffer(buf);
}

TEST(test_parse_fused_pull_paren_right) {
    YY_BUFFER_STATE buf = yy_scan_string(
        "F[d] = P[d] .2 (G[s,d] .1 F[s]) "
        ":: map.1(and,intersect) map.2(and,intersect) reduce.2(or,union, over=s);");
    ASSERT_INT_EQ(0, yyparse());
    AstNode *es = parse_result->data.program.statements[0];
    ASSERT_INT_EQ(NODE_EINSUM_FUSED, es->type);
    EinsumFusedData *f = &es->data.einsum_fused;
    ASSERT_INT_EQ(0, f->inter_is_left2);     /* outer .2 (inner) */
    ASSERT_INT_EQ(0, f->a1_has_reduce);
    ASSERT_INT_EQ(1, f->a2_has_reduce);
    ASSERT_STR_EQ("s", f->over2);
    ast_free(parse_result); parse_result = NULL; yy_delete_buffer(buf);
}
```
Registration: `    RUN_TEST(test_parse_single_reduce_over);` etc.

- [ ] **Step 2: Run to verify failure**

Run: `cmake --build build && ./build/test_parser`
Expected: build/parse failure — fused grammar and `over=` don't exist yet.

- [ ] **Step 3: Add an action-spec accumulator in the prologue**

In `src/parser.y`, in the top C prologue (`%{ ... %}`), add:
```c
typedef struct { int kind;   /* 0=map, 1=reduce */
                 int label;  /* 1 or 2 */
                 char *op;   /* compute op name (owned) */
                 int merge;  /* MergeOp */
                 char *over; /* rank name or NULL (owned) */ } FAct;
typedef struct { FAct items[4]; int n; } FActList;
static FActList *factlist_new(void){ FActList *l=malloc(sizeof(FActList)); l->n=0; return l; }
static void factlist_add(FActList *l, FAct a){ if(l->n<4) l->items[l->n++]=a; }
```

- [ ] **Step 4: Add `opt_over` and extend the single-action reduce**

Add `%type <str_val> opt_over` near the other `%type` lines, and `%type <ptr> fused_rhs fused_action_list` and `%type <ptr> fused_action`.

Add the `opt_over` rule (near `merge_arg`):
```
/* Optional `, over=RANK` tag on a reduce. Yields the rank name or NULL. */
opt_over:
    /* empty */                       { $$ = NULL; }
    | ',' TOK_OVER '=' TOK_IDENT      { $$ = $4; }
    ;
```
In the existing single-action `einsum_stmt` rule, change the reduce part from
`TOK_REDUCE '(' TOK_IDENT merge_arg ')'` to
`TOK_REDUCE '(' TOK_IDENT merge_arg opt_over ')'`, and in its action set the field. The reduce IDENT is `$14`, merge `$15`, and the new `opt_over` becomes `$16` (the `')'` shifts to `$17`, `';'` to `$18`). After building the node:
```c
        AstNode *en = ast_einsum_new(*out, *in1, *in2,
                                     map_op, map_merge, reduce_op, red_merge);
        en->data.einsum.reduce_over = $16;   /* NULL when absent */
        $$ = (void *)en;
```
(Replace the existing direct `$$ = (void *)ast_einsum_new(...)` with the two lines above.)

- [ ] **Step 5: Add the fused production and action list**

Add a second alternative to `einsum_stmt` (after the single-action rule's `;`-terminated action, as `|`):
```
    | tensor_ref '=' fused_rhs TOK_DOUBLE_COLON fused_action_list ';'
    {
        EinsumFusedData *f = (EinsumFusedData *)$3;   /* operands + inter_is_left2 filled by fused_rhs */
        FActList *acts = (FActList *)$5;
        f->output = *(TensorRef *)$1;
        f->a1_has_reduce = 0; f->a2_has_reduce = 0;
        f->over1 = NULL; f->over2 = NULL;
        int have_map1 = 0, have_map2 = 0;
        for (int i = 0; i < acts->n; i++) {
            FAct *a = &acts->items[i];
            OperatorType op = operator_from_str(a->op);
            if (a->kind == 0 && a->label == 1) { f->map1_op = op; f->map1_merge = a->merge; have_map1 = 1; }
            else if (a->kind == 0 && a->label == 2) { f->map2_op = op; f->map2_merge = a->merge; have_map2 = 1; }
            else if (a->kind == 1 && a->label == 1) { f->reduce1_op = op; f->reduce1_merge = a->merge; f->a1_has_reduce = 1; f->over1 = a->over; }
            else if (a->kind == 1 && a->label == 2) { f->reduce2_op = op; f->reduce2_merge = a->merge; f->a2_has_reduce = 1; f->over2 = a->over; }
            free(a->op);
        }
        if (!have_map1 || !have_map2) { yyerror("fused einsum requires map.1 and map.2"); YYERROR; }
        AstNode *node = ast_einsum_fused_new(*f);
        free($1); free($3); free(acts);
        $$ = (void *)node;
    }
```
Add `fused_rhs` (both paren sides). A labeled operator is `'.' TOK_INT_LIT`; we require the inner to be `.1` and outer `.2` (checked in the actions):
```
fused_rhs:
    '(' map_operand '.' TOK_INT_LIT map_operand ')' '.' TOK_INT_LIT map_operand
    {
        if ($4 != 1 || $8 != 2) { yyerror("fused operators must be labeled .1 (inner) and .2 (outer)"); YYERROR; }
        EinsumFusedData *f = calloc(1, sizeof(EinsumFusedData));
        f->inner1 = *(TensorRef *)$2;
        f->inner2 = *(TensorRef *)$5;
        f->outer  = *(TensorRef *)$9;
        f->inter_is_left2 = 1;   /* (inner) .2 outer  -> intermediate is left */
        free($2); free($5); free($9);
        $$ = (void *)f;
    }
    | map_operand '.' TOK_INT_LIT '(' map_operand '.' TOK_INT_LIT map_operand ')'
    {
        if ($3 != 2 || $7 != 1) { yyerror("fused operators must be labeled .1 (inner) and .2 (outer)"); YYERROR; }
        EinsumFusedData *f = calloc(1, sizeof(EinsumFusedData));
        f->outer  = *(TensorRef *)$1;
        f->inner1 = *(TensorRef *)$5;
        f->inner2 = *(TensorRef *)$8;
        f->inter_is_left2 = 0;   /* outer .2 (inner)  -> intermediate is right */
        free($1); free($5); free($8);
        $$ = (void *)f;
    }
    ;

fused_action_list:
    fused_action                    { FActList *l = factlist_new(); factlist_add(l, *(FAct *)$1); free($1); $$ = (void *)l; }
    | fused_action_list fused_action{ FActList *l = (FActList *)$1; factlist_add(l, *(FAct *)$2); free($2); $$ = (void *)l; }
    ;

fused_action:
    TOK_MAP '.' TOK_INT_LIT '(' TOK_IDENT merge_arg ')'
    {
        FAct *a = malloc(sizeof(FAct));
        a->kind = 0; a->label = $3; a->op = $5; a->merge = $6; a->over = NULL;
        $$ = (void *)a;
    }
    | TOK_REDUCE '.' TOK_INT_LIT '(' TOK_IDENT merge_arg opt_over ')'
    {
        FAct *a = malloc(sizeof(FAct));
        a->kind = 1; a->label = $3; a->op = $5; a->merge = $6; a->over = $7;
        $$ = (void *)a;
    }
    ;
```

- [ ] **Step 6: Run to verify pass; check for grammar conflicts**

Run: `cmake --build build 2>&1 | grep -iE 'conflict|error'` → expect no output (no shift/reduce conflicts).
Run: `./build/test_parser` → expect PASS (including the 3 new tests and all existing).

If Bison reports a conflict between the single-action `.` operand rule and the fused `.` TOK_INT_LIT rule: confirm it's resolved — `map_operand '.' TOK_IDENT` (single) vs `map_operand '.' TOK_INT_LIT` (fused) differ on the token after `.` (IDENT vs INT_LIT), so LALR(1) distinguishes them. If a conflict persists, STOP and report it.

- [ ] **Step 7: Commit**
```bash
git add src/parser.y tests/unit/test_parser.c
git commit -m "parser: fused einsum grammar (.N labels, map.N/reduce.N) and over= arg"
```

---

## Task 4: Semantic — fused classification + validation

**Files:** Modify `src/semantic.h`, `src/semantic.c`; Test `tests/unit/test_semantic.c`.

- [ ] **Step 1: Extend `EinsumClass` (semantic.h)**

Add fused fields to `EinsumClass` (after the existing reduction fields):
```c
    int is_fused;
    IndexEntry a1_reduce[MAX_INDICES]; int num_a1_reduce;
    IndexEntry a2_reduce[MAX_INDICES]; int num_a2_reduce;
    int inter_is_left2;
```

- [ ] **Step 2: Write failing semantic tests**

Add to `tests/unit/test_semantic.c` (use the file's existing `parse()` helper; register in `main()`):
```c
TEST(test_semantic_fused_push_classification) {
    AstNode *ast = parse(
        "dim N = 3;\n"
        "tensor F[R=N] : float, empty = 0.0;\n"
        "tensor G[R=N, C=N] : float, empty = 0.0;\n"
        "tensor P[R=N] : float, empty = 0.0;\n"
        "init G = [[0,0,0],[0,0,0],[0,0,0]];\n"
        "init F = [1.0,0.0,0.0];\n"
        "init P = [1.0,0.0,0.0];\n"
        "F[d] = (G[s,d] .1 F[s]) .2 P[d] "
        ":: map.1(and,intersect) reduce.1(or,union) map.2(take_left,intersect);\n");
    SemanticResult sem = semantic_analyze(ast);
    ASSERT_INT_EQ(0, sem.error_count);
    EinsumClass *ec = &sem.symtab->einsum_class[0];
    ASSERT_INT_EQ(1, ec->is_fused);
    ASSERT_INT_EQ(1, ec->num_free_indices);      /* d */
    ASSERT_STR_EQ("d", ec->free_indices[0].name);
    ASSERT_INT_EQ(1, ec->num_a1_reduce);         /* s */
    ASSERT_STR_EQ("s", ec->a1_reduce[0].name);
    ASSERT_INT_EQ(0, ec->num_a2_reduce);
    semantic_result_free(&sem); ast_free(ast);
}

TEST(test_semantic_fused_missing_reduce_errors) {
    AstNode *ast = parse(
        "dim N = 3;\n"
        "tensor F[R=N] : float, empty = 0.0;\n"
        "tensor G[R=N, C=N] : float, empty = 0.0;\n"
        "tensor P[R=N] : float, empty = 0.0;\n"
        "init G = [[0,0,0],[0,0,0],[0,0,0]];\n"
        "init F = [1.0,0.0,0.0];\n"
        "init P = [1.0,0.0,0.0];\n"
        "F[d] = (G[s,d] .1 F[s]) .2 P[d] "
        ":: map.1(and,intersect) map.2(take_left,intersect);\n");  /* action 1 reduces s but no reduce.1 */
    SemanticResult sem = semantic_analyze(ast);
    ASSERT_TRUE(sem.error_count > 0);
    semantic_result_free(&sem); ast_free(ast);
}

TEST(test_semantic_fused_bad_over_errors) {
    AstNode *ast = parse(
        "dim N = 3;\n"
        "tensor F[R=N] : float, empty = 0.0;\n"
        "tensor G[R=N, C=N] : float, empty = 0.0;\n"
        "tensor P[R=N] : float, empty = 0.0;\n"
        "init G = [[0,0,0],[0,0,0],[0,0,0]];\n"
        "init F = [1.0,0.0,0.0];\n"
        "init P = [1.0,0.0,0.0];\n"
        "F[d] = (G[s,d] .1 F[s]) .2 P[d] "
        ":: map.1(and,intersect) reduce.1(or,union, over=d) map.2(take_left,intersect);\n"); /* d is free, not reduced by a1 */
    SemanticResult sem = semantic_analyze(ast);
    ASSERT_TRUE(sem.error_count > 0);
    semantic_result_free(&sem); ast_free(ast);
}
```
Register all three.

- [ ] **Step 3: Implement `analyze_einsum_fused` and dispatch**

In `src/semantic.c`, find where statements are dispatched in `semantic_analyze` (a switch/if on `node->type` calling `analyze_einsum`). Add a branch: when `node->type == NODE_EINSUM_FUSED`, call `analyze_einsum_fused(res, node, eidx)` and increment the einsum index counter the same way the single-action path does.

Add this function (model its tensor-lookup, dim-resolution, and gen handling on the existing `analyze_einsum`; this is the new core). Use small local helpers for index-set membership:
```c
/* Collect the spatial index names of a tensor ref (skipping its gen axis) into
   names[], returning the count. */
static int ref_spatial_indices(TensorRef *r, TensorEntry *te, char *names[], int *sizes) {
    int n = 0;
    for (int a = 0; a < r->num_vars && a < te->num_ranks; a++) {
        if (a == te->gen_axis) continue;
        names[n] = r->rank_vars[a];
        sizes[n] = te->dim_sizes[a];
        n++;
    }
    return n;
}
static int name_present(char *const names[], int n, const char *x) {
    for (int i = 0; i < n; i++) if (strcmp(names[i], x) == 0) return 1;
    return 0;
}

static void analyze_einsum_fused(SemanticResult *res, AstNode *node, int eidx) {
    SymbolTable *st = res->symtab;
    EinsumFusedData *f = &node->data.einsum_fused;
    TensorEntry *out_te = symtab_find_tensor(st, f->output.tensor_name);
    TensorEntry *i1_te  = symtab_find_tensor(st, f->inner1.tensor_name);
    TensorEntry *i2_te  = symtab_find_tensor(st, f->inner2.tensor_name);
    TensorEntry *ou_te  = symtab_find_tensor(st, f->outer.tensor_name);
    if (!out_te || !i1_te || !i2_te || !ou_te) {
        add_error(res, "fused einsum: unknown tensor"); return;
    }

    /* spatial index sets + sizes */
    char *iname1[8]; int isz1[8]; int n1 = ref_spatial_indices(&f->inner1, i1_te, iname1, isz1);
    char *iname2[8]; int isz2[8]; int n2 = ref_spatial_indices(&f->inner2, i2_te, iname2, isz2);
    char *onm[8];    int osz[8];  int no = ref_spatial_indices(&f->outer,  ou_te, onm,    osz);
    char *znm[8];    int zsz[8];  int nz = ref_spatial_indices(&f->output, out_te, znm,    zsz);

    /* size lookup helper over all refs */
    /* inner = n1 ∪ n2 ; collect distinct inner names with sizes */
    char *inner[16]; int innersz[16]; int ni = 0;
    for (int i = 0; i < n1; i++) if (!name_present(inner, ni, iname1[i])) { inner[ni]=iname1[i]; innersz[ni]=isz1[i]; ni++; }
    for (int i = 0; i < n2; i++) if (!name_present(inner, ni, iname2[i])) { inner[ni]=iname2[i]; innersz[ni]=isz2[i]; ni++; }

    EinsumClass *ec = &st->einsum_class[eidx];
    memset(ec, 0, sizeof(*ec));
    ec->is_fused = 1;
    ec->inter_is_left2 = f->inter_is_left2;

    /* free = output spatial indices */
    for (int i = 0; i < nz; i++) {
        ec->free_indices[ec->num_free_indices].name = znm[i];
        ec->free_indices[ec->num_free_indices].size = zsz[i];
        ec->free_indices[ec->num_free_indices].is_free = 1;
        ec->num_free_indices++;
    }
    /* intermediate = inner ∩ (outer ∪ output) ; action-1 reduce = inner − intermediate */
    for (int i = 0; i < ni; i++) {
        int downstream = name_present(onm, no, inner[i]) || name_present(znm, nz, inner[i]);
        if (!downstream) {
            ec->a1_reduce[ec->num_a1_reduce].name = inner[i];
            ec->a1_reduce[ec->num_a1_reduce].size = innersz[i];
            ec->num_a1_reduce++;
        }
    }
    /* intermediate set = inner − a1_reduce (kept inner indices) */
    char *inter[16]; int intersz[16]; int nint = 0;
    for (int i = 0; i < ni; i++) {
        int reduced = 0;
        for (int j = 0; j < ec->num_a1_reduce; j++) if (strcmp(ec->a1_reduce[j].name, inner[i])==0) reduced = 1;
        if (!reduced) { inter[nint]=inner[i]; intersz[nint]=innersz[i]; nint++; }
    }
    /* action-2 reduce = (intermediate ∪ outer) − output */
    /* distinct (intermediate ∪ outer) */
    char *a2in[16]; int a2sz[16]; int na2 = 0;
    for (int i = 0; i < nint; i++) if (!name_present(a2in, na2, inter[i])) { a2in[na2]=inter[i]; a2sz[na2]=intersz[i]; na2++; }
    for (int i = 0; i < no;   i++) if (!name_present(a2in, na2, onm[i]))   { a2in[na2]=onm[i];   a2sz[na2]=osz[i];   na2++; }
    for (int i = 0; i < na2; i++) {
        if (!name_present(znm, nz, a2in[i])) {
            ec->a2_reduce[ec->num_a2_reduce].name = a2in[i];
            ec->a2_reduce[ec->num_a2_reduce].size = a2sz[i];
            ec->num_a2_reduce++;
        }
    }

    /* ---- validation ---- */
    if (ec->num_a1_reduce > 0 && !f->a1_has_reduce)
        add_error(res, "fused einsum: action 1 reduces an index but has no reduce.1");
    if (ec->num_a1_reduce == 0 && f->a1_has_reduce)
        add_error(res, "fused einsum: reduce.1 given but action 1 reduces nothing");
    if (ec->num_a2_reduce > 0 && !f->a2_has_reduce)
        add_error(res, "fused einsum: action 2 reduces an index but has no reduce.2");
    if (ec->num_a2_reduce == 0 && f->a2_has_reduce)
        add_error(res, "fused einsum: reduce.2 given but action 2 reduces nothing");
    if (ec->num_a1_reduce > 0 && ec->num_a2_reduce > 0)
        add_error(res, "fused einsum: reductions on both actions (matrix-chain) are not yet supported");
    /* over= validation */
    if (f->over1) {
        int ok = 0; for (int j=0;j<ec->num_a1_reduce;j++) if (strcmp(ec->a1_reduce[j].name,f->over1)==0) ok=1;
        if (!ok) add_error(res, "fused einsum: reduce.1 over= names a rank not reduced by action 1");
    }
    if (f->over2) {
        int ok = 0; for (int j=0;j<ec->num_a2_reduce;j++) if (strcmp(ec->a2_reduce[j].name,f->over2)==0) ok=1;
        if (!ok) add_error(res, "fused einsum: reduce.2 over= names a rank not reduced by action 2");
    }
    /* take_left as a reduce op is illegal (mirror single-action rule) */
    if (f->a1_has_reduce && f->reduce1_op == OP_TAKE_LEFT) add_error(res, "fused einsum: take_left cannot be a reduce operator");
    if (f->a2_has_reduce && f->reduce2_op == OP_TAKE_LEFT) add_error(res, "fused einsum: take_left cannot be a reduce operator");

    if (eidx + 1 > st->num_einsums) st->num_einsums = eidx + 1;
}
```
Add a forward declaration near the other static prototypes if needed. Also handle the single-action `reduce_over` validation in `analyze_einsum`: after it computes `reduction_indices`, if `d->reduce_over` is set, check it's among them, else `add_error(res, "reduce over= names a rank that is not reduced")`.

- [ ] **Step 4: Run tests**

Run: `cmake --build build && ./build/test_semantic`
Expected: PASS (3 new tests + existing).

- [ ] **Step 5: Commit**
```bash
git add src/semantic.h src/semantic.c tests/unit/test_semantic.c
git commit -m "semantic: classify and validate fused einsums; single-action over= check"
```

---

## Task 5: IR — `IR_FUSED` node + lowering

**Files:** Modify `src/ir.h`, `src/ir.c`; Test `tests/unit/test_ir.c`.

- [ ] **Step 1: Define `IR_FUSED` (ir.h)**

Add `IR_FUSED` to the `IRType` enum. Add a struct carrying everything codegen needs (operands, actions, classification). Place near the other IR structs:
```c
typedef struct {
    char *name; char **coords; int ncoords; double empty; int negate; int gen_read;
} IRFusedOperand;

typedef struct {
    /* output */
    char *out_name; char **out_coords; int out_ncoords; double empty_z; int gen_write;
    /* operands */
    IRFusedOperand inner1, inner2, outer;
    int inter_is_left2;
    /* action 1 */
    OperatorType map1_op; MergeOp map1_merge;
    int a1_has_reduce; OperatorType reduce1_op; MergeOp reduce1_merge; double reduce1_identity;
    /* action 2 */
    OperatorType map2_op; MergeOp map2_merge;
    int a2_has_reduce; OperatorType reduce2_op; MergeOp reduce2_merge; double reduce2_identity;
    /* loop indices: free (output), then the single reducing action's indices */
    char *free_names[8]; int free_sizes[8]; char *free_dimnames[8]; int num_free;
    char *red_names[8];  int red_sizes[8];  char *red_dimnames[8];  int num_red;
    int reduce_on_action; /* 1 if action 1 reduces, 2 if action 2 reduces, 0 if neither */
} IRFused;
```
Add `IRFused fused;` to the IR instruction union and free its owned strings in `ir_free`'s switch.

- [ ] **Step 2: Write failing IR test**

Add to `tests/unit/test_ir.c` (register in `main()`):
```c
TEST(test_ir_fused_push_node) {
    AstNode *ast = parse(
        "dim N = 3;\n"
        "tensor F[R=N] : float, empty = 0.0;\n"
        "tensor G[R=N, C=N] : float, empty = 0.0;\n"
        "tensor P[R=N] : float, empty = 0.0;\n"
        "init G = [[0,0,0],[0,0,0],[0,0,0]];\n"
        "init F = [1.0,0.0,0.0];\n"
        "init P = [1.0,0.0,0.0];\n"
        "F[d] = (G[s,d] .1 F[s]) .2 P[d] "
        ":: map.1(and,intersect) reduce.1(or,union) map.2(take_left,intersect);\n");
    SemanticResult sem = semantic_analyze(ast);
    IRProgram *ir = ir_lower(ast, sem.symtab);
    int found = 0;
    for (int i = 0; i < ir->count; i++) if (ir->instrs[i].type == IR_FUSED) {
        IRFused *fu = &ir->instrs[i].data.fused;
        ASSERT_INT_EQ(1, fu->reduce_on_action);
        ASSERT_INT_EQ(1, fu->num_free);          /* d */
        ASSERT_INT_EQ(1, fu->num_red);           /* s */
        ASSERT_INT_EQ(OP_TAKE_LEFT, fu->map2_op);
        found = 1;
    }
    ASSERT_INT_EQ(1, found);
    ir_free(ir); semantic_result_free(&sem); ast_free(ast);
}
```

- [ ] **Step 3: Implement `emit_einsum_fused` and dispatch**

In `src/ir.c`, in the lowering walk (where it dispatches `NODE_EINSUM` to `emit_einsum`), add: `NODE_EINSUM_FUSED → emit_einsum_fused(ir, node, symtab, eidx)`. Implement it modeled on `emit_einsum` for coord-stripping (`dup_coords_strip`), `gen_read_kind`, `is_cross_gen_name`, and `compute_op_identity`:
```c
static void fill_fused_operand(IRFusedOperand *o, TensorRef *r, TensorEntry *te, SymbolTable *st) {
    o->name = dup_str(r->tensor_name);
    o->coords = dup_coords_strip(r->rank_vars, r->num_vars, te->gen_axis, &o->ncoords);
    o->empty = te->empty_val;
    o->negate = r->negate;
    o->gen_read = gen_read_kind(st, r, te);
}

static void emit_einsum_fused(IRProgram *ir, AstNode *node, SymbolTable *st, int eidx) {
    EinsumFusedData *f = &node->data.einsum_fused;
    EinsumClass *ec = &st->einsum_class[eidx];
    TensorEntry *out_te = symtab_find_tensor(st, f->output.tensor_name);
    TensorEntry *i1 = symtab_find_tensor(st, f->inner1.tensor_name);
    TensorEntry *i2 = symtab_find_tensor(st, f->inner2.tensor_name);
    TensorEntry *ou = symtab_find_tensor(st, f->outer.tensor_name);

    IRInstr *instr = ir_emit(ir, IR_FUSED);
    IRFused *fu = &instr->data.fused;
    memset(fu, 0, sizeof(*fu));
    fu->out_name = dup_str(f->output.tensor_name);
    fu->out_coords = dup_coords_strip(f->output.rank_vars, f->output.num_vars, out_te->gen_axis, &fu->out_ncoords);
    fu->empty_z = out_te->empty_val;
    fu->gen_write = is_cross_gen_name(st, f->output.tensor_name);
    fill_fused_operand(&fu->inner1, &f->inner1, i1, st);
    fill_fused_operand(&fu->inner2, &f->inner2, i2, st);
    fill_fused_operand(&fu->outer,  &f->outer,  ou, st);
    fu->inter_is_left2 = f->inter_is_left2;
    fu->map1_op = f->map1_op; fu->map1_merge = f->map1_merge;
    fu->a1_has_reduce = f->a1_has_reduce; fu->reduce1_op = f->reduce1_op; fu->reduce1_merge = f->reduce1_merge;
    fu->reduce1_identity = compute_op_identity(f->reduce1_op, out_te->datatype);
    fu->map2_op = f->map2_op; fu->map2_merge = f->map2_merge;
    fu->a2_has_reduce = f->a2_has_reduce; fu->reduce2_op = f->reduce2_op; fu->reduce2_merge = f->reduce2_merge;
    fu->reduce2_identity = compute_op_identity(f->reduce2_op, out_te->datatype);

    /* loop indices: free from ec; reduce from whichever action reduces */
    for (int i = 0; i < ec->num_free_indices; i++) {
        fu->free_names[fu->num_free] = dup_str(ec->free_indices[i].name);
        fu->free_sizes[fu->num_free] = ec->free_indices[i].size;
        const char *dn = dim_name_for_index_fused(f, out_te, i1, i2, ou, ec->free_indices[i].name);
        fu->free_dimnames[fu->num_free] = dn ? dup_str(dn) : NULL;
        fu->num_free++;
    }
    IndexEntry *red = NULL; int nred = 0;
    if (ec->num_a1_reduce > 0) { red = ec->a1_reduce; nred = ec->num_a1_reduce; fu->reduce_on_action = 1; }
    else if (ec->num_a2_reduce > 0) { red = ec->a2_reduce; nred = ec->num_a2_reduce; fu->reduce_on_action = 2; }
    else { fu->reduce_on_action = 0; }
    for (int i = 0; i < nred; i++) {
        fu->red_names[fu->num_red] = dup_str(red[i].name);
        fu->red_sizes[fu->num_red] = red[i].size;
        const char *dn = dim_name_for_index_fused(f, out_te, i1, i2, ou, red[i].name);
        fu->red_dimnames[fu->num_red] = dn ? dup_str(dn) : NULL;
        fu->num_red++;
    }
}
```
Add a `dim_name_for_index_fused` helper modeled on the existing `dim_name_for_index` but scanning the four fused refs (output, inner1, inner2, outer). Add `ir_free` cleanup for all owned strings in the `IR_FUSED` case (out_name, out_coords[], the three operands' name+coords[], free_names[]/free_dimnames[], red_names[]/red_dimnames[]).

- [ ] **Step 4: Run tests**

Run: `cmake --build build && ./build/test_ir`
Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git add src/ir.h src/ir.c tests/unit/test_ir.c
git commit -m "ir: IR_FUSED instruction and fused einsum lowering"
```

---

## Task 6: Codegen — `emit_fused_skeleton`

**Files:** Modify `src/codegen.c`; Test `tests/unit/test_codegen.c`.

- [ ] **Step 1: Write the failing codegen test**

Add to `tests/unit/test_codegen.c` (register in `main()`):
```c
TEST(test_codegen_fused_push) {
    AstNode *ast = parse(
        "dim N = 3;\n"
        "tensor F[R=N] : float, empty = 0.0;\n"
        "tensor G[R=N, C=N] : float, empty = 0.0;\n"
        "tensor P[R=N] : float, empty = 0.0;\n"
        "init G = [[0,0,0],[0,0,0],[0,0,0]];\n"
        "init F = [1.0,0.0,0.0];\n"
        "init P = [1.0,0.0,0.0];\n"
        "F[d] = (G[s,d] .1 F[s]) .2 P[d] "
        ":: map.1(and,intersect) reduce.1(or,union) map.2(take_left,intersect);\n");
    SemanticResult sem = semantic_analyze(ast);
    IRProgram *ir = ir_lower(ast, sem.symtab);
    char *code = codegen_python(ir);
    ASSERT_NOT_NULL(code);
    ASSERT_TRUE(strstr(code, "_acc1 =") != NULL);     /* action-1 accumulator */
    ASSERT_TRUE(strstr(code, "_inter =") != NULL);    /* transient scalar */
    ASSERT_TRUE(strstr(code, "c_exists = c_val == 0.0") != NULL); /* !P existence flip */
    free(code); ir_free(ir); semantic_result_free(&sem); ast_free(ast);
}
```

- [ ] **Step 2: Implement `emit_fused_skeleton` + dispatch**

In `src/codegen.c`, in the main IR walk (the `switch (instr->type)` inside `codegen_python_ex`), add a `case IR_FUSED:` that calls `emit_fused_skeleton(&cb, &instr->data.fused)`. Implement the emitter to generate the push/pull/neither structures from the plan's "Generated-code targets," reusing `emit_merge_guard` and `compute_op_pyexpr`. Use a small static helper to emit one "map body" producing a named result var, and one "reduce step" into a named accumulator, parameterized by op/merge/empties/negate and the `_prev`/`_next` operand names (derive operand names from `gen_read`/`gen_write` exactly as `emit_general_skeleton` does at codegen.c:313-329). Full structure:

```c
/* Emit `name = <gen-resolved tensor>[coords...]`. */
static void emit_operand_load(CodeBuf *cb, const char *dst, IRFusedOperand *o);  /* see below */

static void emit_fused_skeleton(CodeBuf *cb, IRFused *fu) {
    char *eZ = format_double_literal(fu->empty_z);
    /* free-index loops */
    for (int i = 0; i < fu->num_free; i++) {
        cb_indent(cb);
        if (fu->free_dimnames[i]) cb_appendf(cb, "for %s in range(%s):\n", fu->free_names[i], fu->free_dimnames[i]);
        else                      cb_appendf(cb, "for %s in range(%d):\n", fu->free_names[i], fu->free_sizes[i]);
        cb->indent++;
    }

    if (fu->reduce_on_action == 1) {
        /* push: acc1 over reduction loops, then action-2 map(-only or reduce) */
        char *id1 = format_double_literal(fu->reduce1_identity);
        cb_indent(cb); cb_appendf(cb, "_acc1 = %s\n", id1);
        for (int i = 0; i < fu->num_red; i++) {
            cb_indent(cb);
            if (fu->red_dimnames[i]) cb_appendf(cb, "for %s in range(%s):\n", fu->red_names[i], fu->red_dimnames[i]);
            else                     cb_appendf(cb, "for %s in range(%d):\n", fu->red_names[i], fu->red_sizes[i]);
            cb->indent++;
        }
        /* action-1 map body -> _m1 / _m1_emit, using inner1,inner2 */
        emit_map_body(cb, &fu->inner1, &fu->inner2, fu->map1_op, fu->map1_merge, fu->empty_z, "_m1");
        /* reduce.1 into _acc1 (scalar accumulator; empty = eZ) */
        emit_reduce_step(cb, "_acc1", "_m1", fu->reduce1_op, fu->reduce1_merge, fu->empty_z);
        for (int i = 0; i < fu->num_red; i++) cb->indent--;  /* close reduction loops */
        cb_indent(cb); cb_append(cb, "_inter = _acc1\n");
        /* action-2 map-only -> store output (a2_has_reduce==0 in push) */
        emit_action2_store(cb, fu);
        free(id1);
    } else if (fu->reduce_on_action == 2) {
        /* pull: acc2 over reduction loops; action-1 map-only -> _inter inside loop */
        char *id2 = format_double_literal(fu->reduce2_identity);
        cb_indent(cb); cb_appendf(cb, "_acc2 = %s\n", id2);
        for (int i = 0; i < fu->num_red; i++) {
            cb_indent(cb);
            if (fu->red_dimnames[i]) cb_appendf(cb, "for %s in range(%s):\n", fu->red_names[i], fu->red_dimnames[i]);
            else                     cb_appendf(cb, "for %s in range(%d):\n", fu->red_names[i], fu->red_sizes[i]);
            cb->indent++;
        }
        emit_map_body(cb, &fu->inner1, &fu->inner2, fu->map1_op, fu->map1_merge, fu->empty_z, "_inter");
        /* action-2 map between _inter and outer -> _m2 (respect inter_is_left2) */
        emit_action2_map_scalar(cb, fu, "_m2");
        emit_reduce_step(cb, "_acc2", "_m2", fu->reduce2_op, fu->reduce2_merge, fu->empty_z);
        for (int i = 0; i < fu->num_red; i++) cb->indent--;
        cb_indent(cb); emit_tensor_access(cb, /*gen-resolved out*/ fu_out_name(fu), fu->out_coords, fu->out_ncoords);
        cb_append(cb, " = _acc2\n");
        free(id2);
    } else {
        /* neither reduces: map.1 -> _inter ; action-2 map -> store */
        emit_map_body(cb, &fu->inner1, &fu->inner2, fu->map1_op, fu->map1_merge, fu->empty_z, "_inter");
        emit_action2_store(cb, fu);
    }

    for (int i = 0; i < fu->num_free; i++) cb->indent--;  /* close free loops */
    free(eZ);
}
```
Implement the helpers `emit_map_body`, `emit_reduce_step`, `emit_action2_store`, `emit_action2_map_scalar`, `emit_operand_load`, and `fu_out_name` (gen-resolves `_prev`/`_next` for operands and `_next`/plain for the output, mirroring `emit_general_skeleton`'s naming at codegen.c:313-329). `emit_map_body` emits `a_val/b_val/a_exists/b_exists` (negate-aware: `==` and `1.0`), the merge guard via `emit_merge_guard`, `a_op/b_op`, the compute via `compute_op_pyexpr`, and `<dst> = ...` plus `<dst>_emit = <dst> != eZ` (and the `else` branch setting `_emit=False`, `<dst>=eZ`). `emit_reduce_step` emits the `_s1/_s2` existence pair, the reduce merge guard, `_vfr`, the `<acc> = compute(<acc>,_vfr)` update, and the `else: <acc>=eZ`. `emit_action2_store`/`emit_action2_map_scalar` build the two action-2 operands as `(_inter, outer)` or `(outer, _inter)` per `inter_is_left2`, where `_inter`'s existence is `_inter != eZ` and the outer operand is loaded via `emit_operand_load` (negate-aware). For the store form, the result writes `out[coords] = m2 if m2 != eZ else eZ`.

(The plan's two "Generated-code targets" blocks are the exact byte target; match them.)

- [ ] **Step 3: Run codegen test, then a manual smoke**

Run: `cmake --build build && ./build/test_codegen` → PASS.
Smoke: write the push test program to `/tmp/f.edge`, `./build/edgecc /tmp/f.edge -o /tmp/f.py && python3 -c "import ast; ast.parse(open('/tmp/f.py').read()); print('py-parses')"` → expect `py-parses` (generated Python is syntactically valid).

- [ ] **Step 4: Full unit suite (no regressions)**

Run: `ctest --test-dir build --output-on-failure`
Expected: all targets pass; single-action output unchanged (the new emitter only runs for `IR_FUSED`).

- [ ] **Step 5: Commit**
```bash
git add src/codegen.c tests/unit/test_codegen.c
git commit -m "codegen: emit fused two-action skeleton (push/pull/neither)"
```

---

## Task 7: E2E — push differential oracle

**Files:** Create `examples/push_bfs_fused.edge`, `tests/e2e/push_bfs_fused/{input.edge,expected_output.txt}`.

- [ ] **Step 1: Reference output**

Run: `./build/edgecc examples/push_bfs_neg.edge -o /tmp/ref.py && python3 /tmp/ref.py`
Expected: `[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]` (record exactly).

- [ ] **Step 2: Write the fused kernel**

Create `examples/push_bfs_fused.edge` (fuses the gather `C` and mask `F` of `push_bfs_neg.edge` into one statement; `P` update stays single-action):
```edge
// Paper-faithful push-BFS, fused: the gather (G·F) and the !P mask are one
// statement -- the paper's (G·F)·¬P. Same graph/result as push_bfs_neg.edge
// (which writes the gather and mask as two cascade statements).
dim N = 6;

tensor G[R=N, C=N]   : float, empty = 0.0;
tensor F[gen I, R=N] : float, empty = 0.0;   // frontier (cross-generational)
tensor P[gen I, R=N] : float, empty = 0.0;   // visited  (cross-generational)

init G = [[0.0,1.0,1.0,0.0,0.0,0.0],
          [0.0,0.0,0.0,1.0,0.0,0.0],
          [0.0,0.0,0.0,1.0,1.0,0.0],
          [0.0,0.0,0.0,0.0,0.0,1.0],
          [0.0,0.0,0.0,0.0,0.0,1.0],
          [0.0,0.0,0.0,0.0,0.0,0.0]];
init F = [1.0,0.0,0.0,0.0,0.0,0.0];
init P = [1.0,0.0,0.0,0.0,0.0,0.0];

F[i+1, d] = (G[s,d] .1 F[i,s]) .2 !P[i,d]
  :: map.1(and,intersect) reduce.1(or,union) map.2(take_left,intersect);
P[i+1, d] = P[i,d] . F[i+1,d] :: map(or,union) reduce(or,union);
stop: stable(P);
```

- [ ] **Step 3: Generate, run, assert match**

Run:
```bash
cmake --build build
mkdir -p tests/e2e/push_bfs_fused
cp examples/push_bfs_fused.edge tests/e2e/push_bfs_fused/input.edge
./build/edgecc tests/e2e/push_bfs_fused/input.edge -o /tmp/fused.py && python3 /tmp/fused.py
diff <(python3 /tmp/ref.py) <(python3 /tmp/fused.py) && echo MATCH
```
Expected: prints `[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]` and `MATCH`. If no MATCH, STOP — do not edit expected output to paper over a real discrepancy; the fused semantics are wrong.

- [ ] **Step 4: Record expected + run e2e suite**

Write `tests/e2e/push_bfs_fused/expected_output.txt`:
```
[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
```
Run: `python3 tests/e2e/run_e2e.py build/edgecc 2>&1 | grep -i 'push_bfs_fused\|FAIL'`
Expected: `PASS push_bfs_fused`, no FAIL.

- [ ] **Step 5: Commit**
```bash
git add examples/push_bfs_fused.edge tests/e2e/push_bfs_fused/
git commit -m "test(e2e): fused push-BFS differential oracle vs cascade"
```

---

## Task 8: E2E — pull variant + `over=` round-trip; negatives

**Files:** Create `examples/pull_bfs_fused.edge`, `tests/e2e/pull_bfs_fused/*`, `tests/e2e/fused_over_roundtrip/*`, and three `tests/negative/*`.

- [ ] **Step 1: Pull kernel + capture**

Create `examples/pull_bfs_fused.edge` (same graph; pull form — inner mask, outer gather):
```edge
// Pull-BFS, fused: (G · !P) gathers candidate edges to unvisited nodes, then
// the outer action reduces over s against the frontier F. Same result as push.
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
init F = [1.0,0.0,0.0,0.0,0.0,0.0];
init P = [1.0,0.0,0.0,0.0,0.0,0.0];

F[i+1, d] = (G[s,d] .1 !P[i,d]) .2 F[i,s]
  :: map.1(and,intersect) map.2(and,intersect) reduce.2(or,union, over=s);
P[i+1, d] = P[i,d] . F[i+1,d] :: map(or,union) reduce(or,union);
stop: stable(P);
```
Run:
```bash
cmake --build build
mkdir -p tests/e2e/pull_bfs_fused
cp examples/pull_bfs_fused.edge tests/e2e/pull_bfs_fused/input.edge
./build/edgecc tests/e2e/pull_bfs_fused/input.edge -o /tmp/pull.py && python3 /tmp/pull.py
```
Expected: `[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]`. Write that into `tests/e2e/pull_bfs_fused/expected_output.txt`. If it differs from push, STOP and debug.

- [ ] **Step 2: `over=` round-trip**

Create `tests/e2e/fused_over_roundtrip/input.edge` = `push_bfs_fused.edge` but with `reduce.1(or,union, over=s)`. Its output must equal the untagged push (`[1.0,...]`). Write `expected_output.txt` with `[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]`.

- [ ] **Step 3: Run e2e suite**

Run: `python3 tests/e2e/run_e2e.py build/edgecc 2>&1 | grep -iE 'pull_bfs_fused|fused_over_roundtrip|FAIL'`
Expected: both PASS, no FAIL.

- [ ] **Step 4: Negative tests**

Create three negative dirs, each with `input.edge` + `expected_errors.txt`:

`tests/negative/fused_missing_reduce/input.edge` — push shape with no `reduce.1`:
```edge
dim N = 3;
tensor F[R=N] : float, empty = 0.0;
tensor G[R=N, C=N] : float, empty = 0.0;
tensor P[R=N] : float, empty = 0.0;
init G = [[0,0,0],[0,0,0],[0,0,0]];
init F = [1.0,0.0,0.0];
init P = [1.0,0.0,0.0];
F[d] = (G[s,d] .1 F[s]) .2 P[d] :: map.1(and,intersect) map.2(take_left,intersect);
```
`expected_errors.txt`:
```
has no reduce.1
```
`tests/negative/fused_bad_over/input.edge` — `over=d` where action 1 reduces `s`:
```edge
dim N = 3;
tensor F[R=N] : float, empty = 0.0;
tensor G[R=N, C=N] : float, empty = 0.0;
tensor P[R=N] : float, empty = 0.0;
init G = [[0,0,0],[0,0,0],[0,0,0]];
init F = [1.0,0.0,0.0];
init P = [1.0,0.0,0.0];
F[d] = (G[s,d] .1 F[s]) .2 P[d] :: map.1(and,intersect) reduce.1(or,union, over=d) map.2(take_left,intersect);
```
`expected_errors.txt`:
```
over=
```
`tests/negative/fused_both_reduce/input.edge` — both actions reduce (matrix-chain, unsupported):
```edge
dim M = 2; dim K = 2; dim NN = 2; dim Pp = 2;
tensor Z[R=M, C=Pp] : float, empty = 0.0;
tensor A[R=M, C=K] : float, empty = 0.0;
tensor B[R=K, C=NN] : float, empty = 0.0;
tensor C[R=NN, C=Pp] : float, empty = 0.0;
init A = [[1,2],[3,4]];
init B = [[1,0],[0,1]];
init C = [[1,1],[1,1]];
Z[m,p] = (A[m,k] .1 B[k,n]) .2 C[n,p] :: map.1(mul,intersect) reduce.1(add,union) map.2(mul,intersect) reduce.2(add,union);
```
`expected_errors.txt`:
```
not yet supported
```
Run: `python3 tests/negative/run_negative.py build/edgecc 2>&1 | grep -iE 'fused_|FAIL'`
Expected: all three PASS. (If a message substring doesn't match the actual stderr, adjust `expected_errors.txt` to a substring that does appear.)

- [ ] **Step 5: Commit**
```bash
git add examples/pull_bfs_fused.edge tests/e2e/pull_bfs_fused/ tests/e2e/fused_over_roundtrip/ tests/negative/fused_missing_reduce/ tests/negative/fused_bad_over/ tests/negative/fused_both_reduce/
git commit -m "test: fused pull-BFS, over= round-trip, and fused negatives"
```

---

## Task 9: Docs reconciliation

**Files:** Modify `CLAUDE.md`, `docs/roadmap.md`.

- [ ] **Step 1: CLAUDE.md — document the form**

In the EDGE Syntax section, after the unary-`!` paragraph, add:
```
Multi-action (fused) Einsum: one parenthesized binary sub-expression plus an
outer binary action in a single statement, e.g. `F[i+1,d] = (G[s,d] .1 F[i,s])
.2 !P[i,d] :: map.1(and,intersect) reduce.1(or,union) map.2(take_left,intersect);`.
Operators and specs are labeled `.1` (inner) / `.2` (outer); each action has a
`map.N` and an optional `reduce.N`. A reduce may name its rank with `over=RANK`
(optional, validated; also allowed on single-action `reduce`). The intermediate
index set is inferred and carried as a transient scalar. Scope: one paren group,
two actions, three operands; reductions on both actions (matrix-chain), general
nesting, and map rank tags are not yet supported.
```
In the v2 Scope Limitations section, remove "multi-operation expressions with compute labels" wording that implies *all* multi-operation expressions are unsupported — change it to note compute labels and `<<` remain unsupported but two-action fused Einsums are supported.

- [ ] **Step 2: roadmap.md — update BFS row + Implemented bullet**

Change the BFS row to drop "multi-action" from the remaining gaps:
```
| Paper-faithful 3-step BFS (`ANY`/parent, occupancy stop; unary `¬`=`!` ✅, multi-action fusion ✅) | 🚧 |
```
Add an Implemented bullet (after the unary `!` bullet):
```
- **Multi-action (fused) Einsums** `(A .1 B) .2 C`: paper-faithful one-line BFS gather/mask (`examples/push_bfs_fused.edge`, `pull_bfs_fused.edge`); transient-scalar intermediate, optional `over=` rank tags.
```
Update the "Partial 🚧 — BFS (paper-faithful)" detail to remove multi-action from the remaining list.

- [ ] **Step 3: Full suite + commit**

Run: `ctest --test-dir build --output-on-failure` → all green.
```bash
git add CLAUDE.md docs/roadmap.md
git commit -m "docs: document fused multi-action Einsums; update BFS roadmap"
```

---

## Self-Review (completed by plan author)

**Spec coverage:**
- Scope (one paren, two actions, three operands, both paren sides) → Tasks 2, 3 (`fused_rhs` both sides), 7, 8. ✅
- `.N` labels + `map.N`/`reduce.N` → Task 3. ✅
- Single-action unchanged/backward-compatible → new node type isolates it; regression check Task 6 Step 4. ✅
- `over=` optional + validated, uniform (single + multi) → Task 1 (token), Task 3 (grammar, single + fused), Task 4 (validation), Task 8 (round-trip + bad-over negative). ✅
- Maps untagged → grammar has no map `over=`; documented Task 9. ✅
- Index inference (intermediate = inner ∩ (outer ∪ output); a1/a2 reduce sets) → Task 4 `analyze_einsum_fused`. ✅
- Consistency/validation errors → Task 4 + Task 8 negatives. ✅
- Intermediate empty = output empty → used as `eZ`/`empty_z` throughout Tasks 5–6. ✅
- Transient scalar intermediate, push/pull loop nests → Task 6 generated-code targets. ✅
- IR structured node → Task 5. ✅
- Differential oracle + pull + regression → Tasks 6–8. ✅
- Reconciliation → Task 9. ✅

**Deviations from spec (intentional):** (1) `over` is a *reserved keyword*, not a soft keyword — the soft-keyword form creates an LALR(1) conflict; reserving `over` is the robust choice (documented in Task 1). (2) Both-actions-reduce (matrix-chain) is a clear *semantic error* rather than a supported bonus — keeps scope tight; the spec's "bonus test" becomes the `fused_both_reduce` negative test (Task 8). Update the spec's scope bullet to match during Task 9 if desired.

**Placeholder scan:** none — code steps carry concrete code; the codegen helper bodies in Task 6 are specified by the exact generated-code targets at the top of the plan plus the helper contracts. (If an implementer wants, each helper mirrors `emit_general_skeleton` line-for-line on scalars.)

**Type/name consistency:** `EinsumFusedData` fields (inner1/inner2/outer/inter_is_left2/map1_op/…/over1/over2) are used identically across ast (Task 2), parser (Task 3), semantic (Task 4 reads the AST), ir (`IRFused`, Task 5), codegen (Task 6). `EinsumClass` fused fields (`is_fused`, `a1_reduce`/`num_a1_reduce`, `a2_reduce`/`num_a2_reduce`, `inter_is_left2`) consistent between semantic.h (Task 4) and ir.c (Task 5). `reduce_on_action` (0/1/2) consistent between Task 5 and Task 6.
