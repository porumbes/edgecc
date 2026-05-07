# edgecc IDE Integration Implementation Plan — VS Code Extension (Tier 1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a VS Code extension that gives `.edge` files syntax highlighting and static-list autocomplete (compute operators, merge operators, types, empty values), shipped as a local-install `.vsix`.

**Architecture:** New top-level subdirectory `editors/vscode/` containing a self-contained TypeScript-built VS Code extension. A TextMate grammar handles syntax coloring; a `CompletionItemProvider` registered against four trigger characters (`(`, `,`, `:`, `=`) returns hardcoded completion lists driven by regex-based context detection on the current line — no parsing of the `.edge` file. Built and packaged with `npm`/`tsc`/`vsce`; tested with Mocha plus `vscode-tmgrammar-test` for grammar fixtures and one `@vscode/test-electron` smoke test.

**Tech Stack:** TypeScript 5.x, Node.js 18+, npm, `@vscode/vsce`, `@types/vscode`, Mocha, `@vscode/test-electron`, `vscode-tmgrammar-test`.

**Reference spec:** `docs/superpowers/specs/2026-05-06-edgecc-ide-integration-design.md`. The plan implements that spec; if anything below contradicts the spec, the spec wins.

---

## File Map

**Compiler core — unchanged.** No edits to `src/`, `tests/`, `CMakeLists.txt`, or any C source. The extension build is fully orthogonal.

**Extension files — create:**
- `editors/vscode/package.json` — extension manifest (Task 1).
- `editors/vscode/tsconfig.json` — TypeScript config (Task 1).
- `editors/vscode/.vscodeignore` — excludes test files from packaged `.vsix` (Task 1).
- `editors/vscode/language-configuration.json` — bracket pairs, comment toggle, indent rules (Task 2).
- `editors/vscode/syntaxes/edge.tmLanguage.json` — TextMate grammar (Tasks 3, 4).
- `editors/vscode/src/completions.ts` — completion item data + context detection + provider class (Tasks 5, 6, 7).
- `editors/vscode/src/extension.ts` — `activate`/`deactivate` entrypoint (Task 7).
- `editors/vscode/test/grammar/dim_decl.test.edge` — grammar fixture (Task 3).
- `editors/vscode/test/grammar/tensor_decl.test.edge` — grammar fixture (Task 3).
- `editors/vscode/test/grammar/init.test.edge` — grammar fixture (Task 3).
- `editors/vscode/test/grammar/comments.test.edge` — grammar fixture (Task 3).
- `editors/vscode/test/grammar/einsum_basic.test.edge` — grammar fixture (Task 4).
- `editors/vscode/test/grammar/einsum_with_merge.test.edge` — grammar fixture (Task 4).
- `editors/vscode/test/completions.test.ts` — pure-data unit tests for completions and context detection (Tasks 5, 6).
- `editors/vscode/test/extension.test.ts` — VS Code integration smoke test (Task 8).
- `editors/vscode/test/runTest.ts` — entrypoint for `@vscode/test-electron` (Task 8).
- `editors/vscode/test/suite/index.ts` — Mocha test suite loader (Task 8).
- `editors/vscode/README.md` — user-facing build/install docs (Task 9).
- `editors/vscode/CHANGELOG.md` — version history (Task 9).

**Repo-level files — modify:**
- `.gitignore` — add `editors/vscode/node_modules/`, `editors/vscode/out/`, `editors/vscode/*.vsix`, `editors/vscode/.vscode-test/` (Task 1).
- `CLAUDE.md` — add a "VS Code extension" section (Task 10).

---

## Task 1: Scaffold the extension directory

**Files:**
- Create: `editors/vscode/package.json`
- Create: `editors/vscode/tsconfig.json`
- Create: `editors/vscode/.vscodeignore`
- Modify: `.gitignore`

This task lays down the skeleton: directories, manifest, TS config, gitignores. After this task, `cd editors/vscode && npm install` succeeds and produces a `node_modules/`. Nothing yet runs; that's later tasks.

- [ ] **Step 1: Create the directory tree.**

Run from the repo root:
```bash
mkdir -p editors/vscode/src editors/vscode/test/grammar editors/vscode/test/suite editors/vscode/syntaxes
```

- [ ] **Step 2: Create `editors/vscode/package.json` with the manifest and devDependencies.**

```json
{
  "name": "edgecc",
  "displayName": "EDGE Language Support",
  "description": "Syntax highlighting and completions for the EDGE tensor language.",
  "version": "0.1.0",
  "publisher": "serban-edgecc",
  "engines": { "vscode": "^1.80.0" },
  "categories": ["Programming Languages"],
  "activationEvents": ["onLanguage:edge"],
  "main": "./out/extension.js",
  "contributes": {
    "languages": [
      {
        "id": "edge",
        "aliases": ["EDGE", "edge"],
        "extensions": [".edge"],
        "configuration": "./language-configuration.json"
      }
    ],
    "grammars": [
      {
        "language": "edge",
        "scopeName": "source.edge",
        "path": "./syntaxes/edge.tmLanguage.json"
      }
    ]
  },
  "scripts": {
    "compile": "tsc -p ./",
    "watch": "tsc -watch -p ./",
    "package": "vsce package",
    "test": "node ./out/test/runTest.js",
    "test:grammar": "vscode-tmgrammar-test \"test/grammar/**/*.test.edge\"",
    "pretest": "npm run compile"
  },
  "devDependencies": {
    "@types/vscode": "^1.80.0",
    "@types/node": "^20.0.0",
    "@types/mocha": "^10.0.0",
    "@vscode/test-electron": "^2.3.0",
    "@vscode/vsce": "^2.20.0",
    "glob": "^10.0.0",
    "mocha": "^10.2.0",
    "typescript": "^5.0.0",
    "vscode-tmgrammar-test": "^0.1.3"
  }
}
```

- [ ] **Step 3: Create `editors/vscode/tsconfig.json`.**

```json
{
  "compilerOptions": {
    "module": "commonjs",
    "target": "ES2020",
    "outDir": "out",
    "lib": ["ES2020"],
    "sourceMap": true,
    "rootDir": ".",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true
  },
  "include": ["src/**/*", "test/**/*"],
  "exclude": ["node_modules", ".vscode-test"]
}
```

- [ ] **Step 4: Create `editors/vscode/.vscodeignore`.**

This file tells `vsce` which files to *exclude* from the packaged `.vsix`. Tests, sources, and CI scaffolding are excluded; only compiled output, the grammar JSON, the language config, and metadata get packaged.

```
.vscode-test/**
.vscode/**
src/**
test/**
node_modules/**
**/*.ts
**/*.map
tsconfig.json
.gitignore
```

- [ ] **Step 5: Update the repo-level `.gitignore`.**

Append:
```
editors/vscode/node_modules/
editors/vscode/out/
editors/vscode/*.vsix
editors/vscode/.vscode-test/
```

(The existing entries `build/`, `*.o`, `*.pyc`, `__pycache__/`, `output.py`, `examples/*.py`, `.worktrees/` stay.)

- [ ] **Step 6: Run `npm install` to verify the manifest is valid and dependencies resolve.**

Run:
```bash
cd editors/vscode && npm install
```
Expected: a `node_modules/` directory and a `package-lock.json` are created. No errors. Some peer-dependency warnings from npm are normal and fine.

If npm rejects the manifest (e.g., invalid JSON), fix and re-run.

- [ ] **Step 7: Commit.**

From the repo root:
```bash
git add editors/vscode/package.json editors/vscode/tsconfig.json editors/vscode/.vscodeignore editors/vscode/package-lock.json .gitignore
git commit -m "vscode: scaffold extension directory with manifest and TS config"
```

(Don't add `node_modules/` — `.gitignore` excludes it.)

---

## Task 2: Language configuration

**Files:**
- Create: `editors/vscode/language-configuration.json`

Adds VS Code's per-language behaviors that aren't part of highlighting: comment toggle, bracket matching, auto-pairs, indentation hints. Tiny task — a single JSON file plus commit.

- [ ] **Step 1: Create `editors/vscode/language-configuration.json`.**

```json
{
  "comments": {
    "lineComment": "//"
  },
  "brackets": [
    ["[", "]"],
    ["(", ")"]
  ],
  "autoClosingPairs": [
    { "open": "[", "close": "]" },
    { "open": "(", "close": ")" }
  ],
  "surroundingPairs": [
    ["[", "]"],
    ["(", ")"]
  ],
  "wordPattern": "[a-zA-Z_][a-zA-Z0-9_]*",
  "indentationRules": {
    "increaseIndentPattern": "^.*[\\[\\(]\\s*$",
    "decreaseIndentPattern": "^\\s*[\\)\\]].*$"
  }
}
```

- [ ] **Step 2: Verify the JSON parses.**

Run:
```bash
cd editors/vscode && python3 -c "import json; json.load(open('language-configuration.json'))"
```
Expected: no output (silent success). If it throws, fix the JSON and re-run.

- [ ] **Step 3: Commit.**

```bash
git add editors/vscode/language-configuration.json
git commit -m "vscode: add language configuration (brackets, comments, indent)"
```

---

## Task 3: TextMate grammar — declarations, literals, comments

**Files:**
- Create: `editors/vscode/syntaxes/edge.tmLanguage.json`
- Create: `editors/vscode/test/grammar/dim_decl.test.edge`
- Create: `editors/vscode/test/grammar/tensor_decl.test.edge`
- Create: `editors/vscode/test/grammar/init.test.edge`
- Create: `editors/vscode/test/grammar/comments.test.edge`

Establishes the grammar skeleton with patterns for `dim`, `tensor`, `init`, line comments, and literals. Einsum statements come in Task 4. Test-first: write fixtures with expected scope assertions, watch them fail, then write grammar to pass.

The `vscode-tmgrammar-test` tool tokenizes each fixture file and matches against scope assertions written as comments. The assertion format is:

```
keyword
^^^^^^^ scope.name.here
```

The `^` characters underline the token in the *previous* line; the scope name follows. Multiple assertions per line with separate `//` lines. The tool reports mismatches.

- [ ] **Step 1: Create fixture `dim_decl.test.edge`.**

```
dim K = 3;
// <-------- source.edge
// <- storage.type.declaration.edge
//  ^ storage.type.declaration.edge
//  ^ storage.type.declaration.edge
//    ^ variable.other.edge
//      ^ keyword.operator.assignment.edge
//        ^ constant.numeric.integer.edge
//         ^ punctuation.separator.edge
```

(Scope assertions in `vscode-tmgrammar-test` use the `// <-` prefix for the start-of-line marker and `// ^` for column-aligned markers. The first non-assertion line is the source; subsequent `//`-prefixed lines are assertions about the immediately preceding source line.)

- [ ] **Step 2: Create fixture `comments.test.edge`.**

```
// this is a comment
// <- comment.line.double-slash.edge
dim X = 5;  // trailing comment
//          ^ comment.line.double-slash.edge
```

- [ ] **Step 3: Create fixture `tensor_decl.test.edge`.**

```
tensor Z[R=M, C=N] : float, empty = 0.0;
// <- storage.type.declaration.edge
//      ^ variable.other.edge
//       ^ punctuation.bracket.edge
//        ^ variable.other.edge
//         ^ keyword.operator.assignment.edge
//          ^ variable.other.edge
//           ^ punctuation.separator.edge
//             ^ variable.other.edge
//              ^ keyword.operator.assignment.edge
//               ^ variable.other.edge
//                ^ punctuation.bracket.edge
//                  ^ punctuation.colon.edge
//                    ^ storage.type.edge
//                         ^ punctuation.separator.edge
//                           ^ keyword.other.empty.edge
//                                 ^ keyword.operator.assignment.edge
//                                   ^ constant.numeric.float.edge
//                                      ^ punctuation.separator.edge
```

The exact column offsets depend on the source line; the engineer can iterate by running the test, inspecting failures, and aligning the `^` markers.

- [ ] **Step 4: Create fixture `init.test.edge`.**

```
init A = [[1, 2], [3, 4]];
// <- storage.type.declaration.edge
//      ^ variable.other.edge
//        ^ keyword.operator.assignment.edge
//          ^ punctuation.bracket.edge
//           ^ punctuation.bracket.edge
//            ^ constant.numeric.integer.edge
//             ^ punctuation.separator.edge
//               ^ constant.numeric.integer.edge
//                ^ punctuation.bracket.edge
//                 ^ punctuation.separator.edge
//                   ^ punctuation.bracket.edge
//                    ^ constant.numeric.integer.edge
//                     ^ punctuation.separator.edge
//                       ^ constant.numeric.integer.edge
//                        ^ punctuation.bracket.edge
//                         ^ punctuation.bracket.edge
//                          ^ punctuation.separator.edge
```

- [ ] **Step 5: Run grammar tests to confirm they fail (no grammar exists yet).**

```bash
cd editors/vscode && npx vscode-tmgrammar-test "test/grammar/**/*.test.edge"
```
Expected: an error about missing grammar file `syntaxes/edge.tmLanguage.json` (or all fixtures fail because no grammar is registered).

- [ ] **Step 6: Create the grammar file `editors/vscode/syntaxes/edge.tmLanguage.json` with patterns for declarations, literals, and comments.**

```json
{
  "$schema": "https://raw.githubusercontent.com/martinring/tmlanguage/master/tmlanguage.json",
  "name": "EDGE",
  "scopeName": "source.edge",
  "fileTypes": ["edge"],
  "patterns": [
    { "include": "#comments" },
    { "include": "#dim-decl" },
    { "include": "#tensor-decl" },
    { "include": "#init-stmt" },
    { "include": "#literals" },
    { "include": "#operators" },
    { "include": "#identifiers" }
  ],
  "repository": {
    "comments": {
      "match": "//.*$",
      "name": "comment.line.double-slash.edge"
    },
    "dim-decl": {
      "match": "\\b(dim)\\b",
      "name": "storage.type.declaration.edge"
    },
    "tensor-decl": {
      "match": "\\b(tensor|init|empty)\\b",
      "captures": {
        "1": { "name": "storage.type.declaration.edge" }
      }
    },
    "init-stmt": {
      "match": "\\b(init)\\b",
      "name": "storage.type.declaration.edge"
    },
    "literals": {
      "patterns": [
        { "match": "\\b(true|false|inf)\\b", "name": "constant.language.edge" },
        { "match": "\\b[0-9]+\\.[0-9]+\\b", "name": "constant.numeric.float.edge" },
        { "match": "\\b[0-9]+\\b", "name": "constant.numeric.integer.edge" },
        { "match": "\\b(integer|float|boolean)\\b", "name": "storage.type.edge" }
      ]
    },
    "operators": {
      "patterns": [
        { "match": "::", "name": "keyword.operator.action-sep.edge" },
        { "match": "=", "name": "keyword.operator.assignment.edge" },
        { "match": "\\.", "name": "keyword.operator.dot.edge" },
        { "match": ":", "name": "punctuation.colon.edge" },
        { "match": "[,;]", "name": "punctuation.separator.edge" },
        { "match": "[\\[\\]\\(\\)]", "name": "punctuation.bracket.edge" }
      ]
    },
    "identifiers": {
      "match": "\\b[a-zA-Z_][a-zA-Z0-9_]*\\b",
      "name": "variable.other.edge"
    }
  }
}
```

Note: `tensor-decl` and `init-stmt` are kept simple word-matches in this task. Task 4 will not need to revisit them — the einsum patterns extend the grammar by adding new entries to the `repository`, not by modifying these.

The `keyword.other.empty.edge` scope from the spec is captured by the `tensor-decl` pattern matching `empty` as `storage.type.declaration.edge`. The spec named it differently for theme nuance, but theming is fine with either; we keep the simpler scope assignment here.

- [ ] **Step 7: Run grammar tests; iterate on `^` column alignments until they pass.**

```bash
cd editors/vscode && npx vscode-tmgrammar-test "test/grammar/**/*.test.edge"
```

Expected outcome (after iteration): all four fixtures pass. Engineers may need to adjust column offsets in fixture assertions (the `^` markers) so they align with actual token boundaries; the tool's error messages indicate which line/column mismatched.

If a token is consistently scoped wrong (e.g., `dim` shows as `variable.other.edge` instead of `storage.type.declaration.edge`), the issue is in the grammar's pattern ordering — earlier patterns in the top-level `patterns` array win. The `dim-decl`, `tensor-decl`, `init-stmt` patterns must come *before* `identifiers` in the top-level include list (they already do).

- [ ] **Step 8: Manual smoke check — open a `.edge` file in VS Code Extension Development Host.**

Run:
```bash
cd editors/vscode && npm run compile
```
Then open the `editors/vscode/` folder in VS Code and press **F5** to launch the Extension Development Host. In the new window, open `examples/gemm.edge` from the repo. Verify keywords (`dim`, `tensor`, `init`) and numeric literals are colored. (`map`, `reduce`, action keywords aren't yet — Task 4.)

- [ ] **Step 9: Commit.**

```bash
git add editors/vscode/syntaxes/edge.tmLanguage.json editors/vscode/test/grammar/
git commit -m "vscode: TextMate grammar — declarations, literals, comments"
```

---

## Task 4: TextMate grammar — einsum and action regions

**Files:**
- Modify: `editors/vscode/syntaxes/edge.tmLanguage.json`
- Create: `editors/vscode/test/grammar/einsum_basic.test.edge`
- Create: `editors/vscode/test/grammar/einsum_with_merge.test.edge`

Extends the grammar with begin/end regions for `map(...)` and `reduce(...)` so the compute and merge operator keywords get their own scopes (`support.function.compute.edge`, `support.function.merge.edge`) only when they appear inside those parens.

- [ ] **Step 1: Create fixture `einsum_basic.test.edge` (no merge ops, just default form).**

```
Z[m,n] = A[k,m] . B[k,n] :: map(mul) reduce(add);
// <- variable.other.edge
//        ^ keyword.operator.assignment.edge
//                       ^ keyword.operator.action-sep.edge
//                          ^ keyword.control.action.edge
//                             ^ punctuation.bracket.edge
//                              ^ support.function.compute.edge
//                                 ^ punctuation.bracket.edge
//                                   ^ keyword.control.action.edge
//                                         ^ support.function.compute.edge
```

(Engineer iterates on column alignment; the assertions above are illustrative — the test runner will guide.)

- [ ] **Step 2: Create fixture `einsum_with_merge.test.edge` (full v2 form).**

```
F1[d] = G[s,d] . F[s] :: map(add, intersect) reduce(min, union);
// <- variable.other.edge
//                       ^ keyword.control.action.edge
//                          ^ punctuation.bracket.edge
//                           ^ support.function.compute.edge
//                              ^ punctuation.separator.edge
//                                ^ support.function.merge.edge
//                                          ^ punctuation.bracket.edge
//                                            ^ keyword.control.action.edge
//                                                  ^ support.function.compute.edge
//                                                       ^ support.function.merge.edge
```

- [ ] **Step 3: Run grammar tests; expect failure on the new fixtures.**

```bash
cd editors/vscode && npx vscode-tmgrammar-test "test/grammar/**/*.test.edge"
```

Expected: the four prior fixtures still pass, the two new ones fail because `map`/`reduce`/compute-ops/merge-ops aren't yet recognized.

- [ ] **Step 4: Extend the grammar with action regions.**

Modify `syntaxes/edge.tmLanguage.json`. Add two new entries to `repository`:

```json
"action-map": {
  "begin": "\\b(map)\\s*(\\()",
  "end": "\\)",
  "beginCaptures": {
    "1": { "name": "keyword.control.action.edge" },
    "2": { "name": "punctuation.bracket.edge" }
  },
  "endCaptures": {
    "0": { "name": "punctuation.bracket.edge" }
  },
  "patterns": [
    {
      "match": "\\b(add|sub|mul|min|max|and|or)\\b",
      "name": "support.function.compute.edge"
    },
    {
      "match": "\\b(passthrough|nopass|intersect|take_left|take_left_only|take_right|take_right_only|xor|union)\\b",
      "name": "support.function.merge.edge"
    },
    { "match": ",", "name": "punctuation.separator.edge" }
  ]
},
"action-reduce": {
  "begin": "\\b(reduce)\\s*(\\()",
  "end": "\\)",
  "beginCaptures": {
    "1": { "name": "keyword.control.action.edge" },
    "2": { "name": "punctuation.bracket.edge" }
  },
  "endCaptures": {
    "0": { "name": "punctuation.bracket.edge" }
  },
  "patterns": [
    {
      "match": "\\b(add|sub|mul|min|max|and|or)\\b",
      "name": "support.function.compute.edge"
    },
    {
      "match": "\\b(passthrough|nopass|intersect|take_left|take_left_only|take_right|take_right_only|xor|union)\\b",
      "name": "support.function.merge.edge"
    },
    { "match": ",", "name": "punctuation.separator.edge" }
  ]
}
```

And add includes for the new patterns at the top of the top-level `patterns` array, **before** `#identifiers`:

```json
"patterns": [
  { "include": "#comments" },
  { "include": "#action-map" },
  { "include": "#action-reduce" },
  { "include": "#dim-decl" },
  { "include": "#tensor-decl" },
  { "include": "#init-stmt" },
  { "include": "#literals" },
  { "include": "#operators" },
  { "include": "#identifiers" }
]
```

The order matters — `action-map` and `action-reduce` must come before `identifiers` so `map`/`reduce`/`add`/`intersect` are scoped as keywords/functions, not as bare identifiers.

- [ ] **Step 5: Run grammar tests; iterate column alignments until all six fixtures pass.**

```bash
cd editors/vscode && npx vscode-tmgrammar-test "test/grammar/**/*.test.edge"
```

Expected (after iteration): all six fixtures pass.

- [ ] **Step 6: Manual smoke check.**

Compile (`npm run compile`), launch the Extension Development Host (F5), open `examples/gemm.edge`. Verify `map`, `reduce`, `mul`, `add` are colored as keywords / functions. Open `examples/dotproduct.edge` and confirm the same.

- [ ] **Step 7: Commit.**

```bash
git add editors/vscode/syntaxes/edge.tmLanguage.json editors/vscode/test/grammar/einsum_basic.test.edge editors/vscode/test/grammar/einsum_with_merge.test.edge
git commit -m "vscode: TextMate grammar — einsum + map/reduce action regions"
```

---

## Task 5: Completion data layer

**Files:**
- Create: `editors/vscode/src/completions.ts` (initial version with data only)
- Create: `editors/vscode/test/completions.test.ts`

Adds the four hardcoded `ItemSpec` arrays (`COMPUTE_OPS`, `MERGE_OPS`, `TYPES`, `EMPTY_VALUES`) and a unit-test file that asserts their structure. No VS Code API surface in this task — pure TypeScript that can be compiled and tested standalone.

- [ ] **Step 1: Write failing unit tests in `test/completions.test.ts`.**

```typescript
import * as assert from "assert";
import {
  COMPUTE_OPS,
  MERGE_OPS,
  TYPES,
  EMPTY_VALUES,
} from "../src/completions";

describe("completion item lists", () => {
  it("COMPUTE_OPS contains all 7 built-in compute operators", () => {
    const labels = COMPUTE_OPS.map((it) => it.label).sort();
    assert.deepStrictEqual(labels, ["add", "and", "max", "min", "mul", "or", "sub"]);
  });

  it("MERGE_OPS contains all 9 supported merge operators", () => {
    const labels = MERGE_OPS.map((it) => it.label).sort();
    assert.deepStrictEqual(labels, [
      "intersect", "nopass", "passthrough", "take_left", "take_left_only",
      "take_right", "take_right_only", "union", "xor",
    ]);
  });

  it("TYPES contains exactly integer, float, boolean", () => {
    const labels = TYPES.map((it) => it.label).sort();
    assert.deepStrictEqual(labels, ["boolean", "float", "integer"]);
  });

  it("EMPTY_VALUES contains exactly 0, 0.0, inf, true, false", () => {
    const labels = EMPTY_VALUES.map((it) => it.label).sort();
    assert.deepStrictEqual(labels, ["0", "0.0", "false", "inf", "true"]);
  });

  it("every MERGE_OP has documentation containing a markdown table", () => {
    for (const op of MERGE_OPS) {
      assert.ok(
        op.documentation && op.documentation.includes("| A | B | Z |"),
        `merge op '${op.label}' missing truth-table documentation`,
      );
    }
  });

  it("every item has a non-empty detail string", () => {
    for (const list of [COMPUTE_OPS, MERGE_OPS, TYPES, EMPTY_VALUES]) {
      for (const item of list) {
        assert.ok(
          item.detail && item.detail.length > 0,
          `item '${item.label}' missing detail`,
        );
      }
    }
  });
});
```

- [ ] **Step 2: Run tests; expect failure (`completions.ts` doesn't exist yet).**

```bash
cd editors/vscode && npm run compile
```
Expected: compile error — cannot find `../src/completions`. Or, if compile succeeds despite the missing module (it shouldn't), running mocha against the compiled output would fail at import time. Either way, no green test.

- [ ] **Step 3: Implement `editors/vscode/src/completions.ts` (data layer only).**

```typescript
import { CompletionItemKind } from "vscode";

export interface ItemSpec {
  label: string;
  kind: CompletionItemKind;
  detail: string;
  documentation?: string;
  sortText?: string;
}

// Helper to build a markdown truth-table for a merge operator.
// Cells are listed in (NoNo, NoYes, YesNo, YesYes) order.
function truthTable(name: string, symbol: string, cells: [string, string, string, string]): string {
  return (
    `**${name}** (${symbol}). Compute admission per (A exists, B exists).\n\n` +
    "| A | B | Z |\n|---|---|---|\n" +
    `| No  | No  | ${cells[0]} |\n` +
    `| No  | Yes | ${cells[1]} |\n` +
    `| Yes | No  | ${cells[2]} |\n` +
    `| Yes | Yes | ${cells[3]} |`
  );
}

export const COMPUTE_OPS: ItemSpec[] = [
  { label: "add", kind: CompletionItemKind.Operator, detail: "Addition",
    documentation: "Binary `+`. Identity for reduce: `0`.", sortText: "1-add" },
  { label: "mul", kind: CompletionItemKind.Operator, detail: "Multiplication",
    documentation: "Binary `*`. Identity for reduce: `1`.", sortText: "2-mul" },
  { label: "min", kind: CompletionItemKind.Operator, detail: "Minimum",
    documentation: "`min(a, b)`. Identity: `+inf` (float) or `INT_MAX` (integer).", sortText: "3-min" },
  { label: "max", kind: CompletionItemKind.Operator, detail: "Maximum",
    documentation: "`max(a, b)`. Identity: `-inf` (float) or `INT_MIN` (integer).", sortText: "4-max" },
  { label: "sub", kind: CompletionItemKind.Operator, detail: "Subtraction",
    documentation: "Binary `-`. Identity for reduce: `0` (right-identity only).", sortText: "5-sub" },
  { label: "and", kind: CompletionItemKind.Operator, detail: "Boolean AND",
    documentation: "Boolean `and`. Identity: `true`. Boolean tensors only.", sortText: "6-and" },
  { label: "or", kind: CompletionItemKind.Operator, detail: "Boolean OR",
    documentation: "Boolean `or`. Identity: `false`. Boolean tensors only.", sortText: "7-or" },
];

export const MERGE_OPS: ItemSpec[] = [
  { label: "passthrough", kind: CompletionItemKind.EnumMember, detail: "Always admits (default if omitted)",
    documentation: truthTable("Pass-through merge", "𝟙", ["**Yes**", "**Yes**", "**Yes**", "**Yes**"]),
    sortText: "1-passthrough" },
  { label: "intersect", kind: CompletionItemKind.EnumMember, detail: "Both operands non-empty",
    documentation: truthTable("Intersection merge", "∩", ["No", "No", "No", "**Yes**"]),
    sortText: "2-intersect" },
  { label: "union", kind: CompletionItemKind.EnumMember, detail: "At least one non-empty",
    documentation: truthTable("Union merge", "∪", ["No", "**Yes**", "**Yes**", "**Yes**"]),
    sortText: "3-union" },
  { label: "take_left", kind: CompletionItemKind.EnumMember, detail: "Left operand non-empty",
    documentation: truthTable("Take-left merge", "←", ["No", "No", "**Yes**", "**Yes**"]),
    sortText: "4-take_left" },
  { label: "take_right", kind: CompletionItemKind.EnumMember, detail: "Right operand non-empty",
    documentation: truthTable("Take-right merge", "→", ["No", "**Yes**", "No", "**Yes**"]),
    sortText: "5-take_right" },
  { label: "take_left_only", kind: CompletionItemKind.EnumMember, detail: "Left non-empty AND right empty",
    documentation: truthTable("Take-left-only merge", "⊖_l", ["No", "No", "**Yes**", "No"]),
    sortText: "6-take_left_only" },
  { label: "take_right_only", kind: CompletionItemKind.EnumMember, detail: "Right non-empty AND left empty",
    documentation: truthTable("Take-right-only merge", "⊖_r", ["No", "**Yes**", "No", "No"]),
    sortText: "7-take_right_only" },
  { label: "xor", kind: CompletionItemKind.EnumMember, detail: "Exactly one non-empty",
    documentation: truthTable("Exclusive-or merge", "⊕", ["No", "**Yes**", "**Yes**", "No"]),
    sortText: "8-xor" },
  { label: "nopass", kind: CompletionItemKind.EnumMember, detail: "Always rejects",
    documentation: truthTable("No-pass merge", "𝔽", ["No", "No", "No", "No"]),
    sortText: "9-nopass" },
];

export const TYPES: ItemSpec[] = [
  { label: "integer", kind: CompletionItemKind.Keyword, detail: "Integer-typed tensor" },
  { label: "float",   kind: CompletionItemKind.Keyword, detail: "Floating-point-typed tensor" },
  { label: "boolean", kind: CompletionItemKind.Keyword, detail: "Boolean-typed tensor" },
];

export const EMPTY_VALUES: ItemSpec[] = [
  { label: "0",     kind: CompletionItemKind.Value, detail: "Integer zero" },
  { label: "0.0",   kind: CompletionItemKind.Value, detail: "Float zero" },
  { label: "inf",   kind: CompletionItemKind.Value, detail: "Float infinity (often used for min)" },
  { label: "true",  kind: CompletionItemKind.Value, detail: "Boolean true" },
  { label: "false", kind: CompletionItemKind.Value, detail: "Boolean false" },
];
```

- [ ] **Step 4: Verify TypeScript compiles cleanly.**

Run:
```bash
cd editors/vscode && npm run compile
```
Expected: zero errors. The compiler emits `out/src/completions.js` and `out/test/completions.test.js`.

- [ ] **Step 5: Run the unit tests.**

We don't yet have a Mocha runner wired up. Use Mocha directly via the installed dev-dep:

```bash
cd editors/vscode && npx mocha out/test/completions.test.js
```

Expected: all 6 tests pass.

- [ ] **Step 6: Commit.**

```bash
git add editors/vscode/src/completions.ts editors/vscode/test/completions.test.ts
git commit -m "vscode: completion data layer — compute, merge, type, empty-value lists"
```

---

## Task 6: Context detection

**Files:**
- Modify: `editors/vscode/src/completions.ts`
- Modify: `editors/vscode/test/completions.test.ts`

Adds the `detectContext(linePrefix: string)` pure function that returns one of `"compute-op" | "merge-op" | "type" | "empty-value" | null` based on regex matches against the current line prefix. TDD: write tests first.

- [ ] **Step 1: Add failing unit tests for `detectContext`.**

Append to `test/completions.test.ts`:

```typescript
import { detectContext } from "../src/completions";

describe("detectContext", () => {
  it("returns 'compute-op' right after map(", () => {
    assert.strictEqual(detectContext("Z[m] = A . B :: map("), "compute-op");
  });

  it("returns 'compute-op' right after reduce(", () => {
    assert.strictEqual(detectContext("Z[m] = A . B :: map(mul) reduce("), "compute-op");
  });

  it("returns 'merge-op' after a comma in map(...)", () => {
    assert.strictEqual(detectContext("Z[m] = A . B :: map(mul, "), "merge-op");
  });

  it("returns 'merge-op' after a comma in reduce(...)", () => {
    assert.strictEqual(
      detectContext("Z[m] = A . B :: map(mul, intersect) reduce(min, "),
      "merge-op",
    );
  });

  it("returns 'type' after `:` in a tensor decl", () => {
    assert.strictEqual(detectContext("tensor X[R=K] : "), "type");
    assert.strictEqual(detectContext("tensor X[R=K, C=M] : "), "type");
  });

  it("returns 'empty-value' after `empty = `", () => {
    assert.strictEqual(detectContext("tensor X[R=K] : float, empty = "), "empty-value");
  });

  it("returns null inside a comment", () => {
    assert.strictEqual(detectContext("// map("), null);
    assert.strictEqual(detectContext("dim K = 3; // map(mul, "), null);
  });

  it("returns null inside a tensor rank list (no map/reduce in context)", () => {
    assert.strictEqual(detectContext("tensor X[R=K, "), null);
  });

  it("returns null on irrelevant prefixes", () => {
    assert.strictEqual(detectContext(""), null);
    assert.strictEqual(detectContext("dim K = "), null);
    assert.strictEqual(detectContext("init A = ["), null);
  });
});
```

- [ ] **Step 2: Run tests; expect failures (function doesn't exist).**

```bash
cd editors/vscode && npm run compile
```
Expected: compile error — `Module '"../src/completions"' has no exported member 'detectContext'`.

- [ ] **Step 3: Implement `detectContext` in `src/completions.ts`.**

Append to the bottom of `src/completions.ts`:

```typescript
export type Context = "compute-op" | "merge-op" | "type" | "empty-value" | null;

/**
 * Determine which (if any) static-completion context the cursor is in,
 * given the prefix of the current line up to the cursor. Pure function:
 * does not parse the .edge file, only inspects this single line's prefix.
 *
 * Returns null if the cursor is inside a // line comment or at no recognized
 * trigger context.
 */
export function detectContext(linePrefix: string): Context {
  // Strip everything from the first `//` onward (line comments).
  const commentIdx = linePrefix.indexOf("//");
  const code = commentIdx >= 0 ? linePrefix.substring(0, commentIdx) : linePrefix;

  // 1st arg of map(...) / reduce(...): compute op slot
  if (/\b(?:map|reduce)\s*\(\s*$/.test(code)) {
    return "compute-op";
  }
  // 2nd arg of map(...) / reduce(...): merge op slot
  // The (?:map|reduce)( anchors the search so commas inside other contexts
  // (e.g., a tensor's rank list) don't accidentally match.
  if (/\b(?:map|reduce)\s*\([^,()]*,\s*$/.test(code)) {
    return "merge-op";
  }
  // tensor decl, after the closing `]` and the `:`
  if (/\btensor\s+\w+\s*\[[^\]]*\]\s*:\s*$/.test(code)) {
    return "type";
  }
  // empty-value slot
  if (/\bempty\s*=\s*$/.test(code)) {
    return "empty-value";
  }
  return null;
}
```

- [ ] **Step 4: Recompile and run tests.**

```bash
cd editors/vscode && npm run compile && npx mocha out/test/completions.test.js
```

Expected: all tests pass (the original 6 from Task 5 plus the new detectContext tests — roughly 15 assertions across the new `describe` block).

- [ ] **Step 5: Commit.**

```bash
git add editors/vscode/src/completions.ts editors/vscode/test/completions.test.ts
git commit -m "vscode: regex-based context detection for completions"
```

---

## Task 7: Provider class and extension entrypoint

**Files:**
- Modify: `editors/vscode/src/completions.ts` (add provider class)
- Create: `editors/vscode/src/extension.ts`

Wraps the data + context-detection into a `CompletionItemProvider` class and adds the `activate`/`deactivate` extension entrypoint. After this task the extension is functional in a real VS Code instance — automated verification is in Task 8.

- [ ] **Step 1: Add `EdgeCompletionProvider` class to `src/completions.ts`.**

Append:

```typescript
import {
  CompletionItem,
  CompletionItemProvider,
  Position,
  TextDocument,
  CancellationToken,
  CompletionContext,
  ProviderResult,
  MarkdownString,
} from "vscode";

function toCompletionItems(specs: ItemSpec[]): CompletionItem[] {
  return specs.map((spec) => {
    const item = new CompletionItem(spec.label, spec.kind);
    item.detail = spec.detail;
    if (spec.documentation) {
      item.documentation = new MarkdownString(spec.documentation);
    }
    if (spec.sortText) {
      item.sortText = spec.sortText;
    }
    return item;
  });
}

export class EdgeCompletionProvider implements CompletionItemProvider {
  provideCompletionItems(
    document: TextDocument,
    position: Position,
    _token: CancellationToken,
    _context: CompletionContext,
  ): ProviderResult<CompletionItem[]> {
    const lineText = document.lineAt(position.line).text;
    const linePrefix = lineText.substring(0, position.character);

    switch (detectContext(linePrefix)) {
      case "compute-op":
        return toCompletionItems(COMPUTE_OPS);
      case "merge-op":
        return toCompletionItems(MERGE_OPS);
      case "type":
        return toCompletionItems(TYPES);
      case "empty-value":
        return toCompletionItems(EMPTY_VALUES);
      default:
        return undefined;
    }
  }
}
```

- [ ] **Step 2: Create `src/extension.ts`.**

```typescript
import * as vscode from "vscode";
import { EdgeCompletionProvider } from "./completions";

export function activate(context: vscode.ExtensionContext): void {
  const provider = vscode.languages.registerCompletionItemProvider(
    { language: "edge", scheme: "file" },
    new EdgeCompletionProvider(),
    "(", ",", ":", "=",
  );
  context.subscriptions.push(provider);
}

export function deactivate(): void {
  // No cleanup needed; subscriptions disposed automatically by VS Code.
}
```

- [ ] **Step 3: Recompile.**

```bash
cd editors/vscode && npm run compile
```
Expected: no errors. The compiled `out/src/extension.js` is the entrypoint declared in `package.json`'s `"main"`.

- [ ] **Step 4: Manual smoke test in Extension Development Host.**

Open `editors/vscode/` in VS Code and press **F5**. In the new VS Code window, open a `.edge` file (e.g., `examples/gemm.edge`). Position the cursor inside an existing `map(` and verify a dropdown appears with compute ops. Then type `, ` after the compute op and verify a dropdown of merge ops appears. Type `tensor X[R=K] : ` and verify type dropdown. Type `tensor X[R=K] : float, empty = ` and verify empty-value dropdown.

If any dropdown fails to appear: inspect the Developer Tools console (Help → Toggle Developer Tools in the dev-host window) for activation errors, check that `extension.js` is at `editors/vscode/out/src/extension.js`, and verify `package.json`'s `"main"` matches.

- [ ] **Step 5: Commit.**

```bash
git add editors/vscode/src/completions.ts editors/vscode/src/extension.ts
git commit -m "vscode: completion provider class and extension entrypoint"
```

---

## Task 8: VS Code integration smoke test

**Files:**
- Create: `editors/vscode/test/extension.test.ts`
- Create: `editors/vscode/test/runTest.ts`
- Create: `editors/vscode/test/suite/index.ts`

Adds one integration test that spins up a real VS Code instance via `@vscode/test-electron`, opens an in-memory `.edge` document, requests completions at a known cursor position, and asserts the result is non-empty. This catches activation / registration bugs that pure-data unit tests can't.

- [ ] **Step 1: Create `test/runTest.ts` (the entrypoint that boots VS Code).**

```typescript
import * as path from "path";
import { runTests } from "@vscode/test-electron";

async function main() {
  try {
    const extensionDevelopmentPath = path.resolve(__dirname, "../..");
    const extensionTestsPath = path.resolve(__dirname, "./suite/index");

    await runTests({ extensionDevelopmentPath, extensionTestsPath });
  } catch (err) {
    console.error("Failed to run integration tests:", err);
    process.exit(1);
  }
}

main();
```

- [ ] **Step 2: Create `test/suite/index.ts` (Mocha loader).**

```typescript
import * as path from "path";
import Mocha from "mocha";
import { glob } from "glob";

export function run(): Promise<void> {
  const mocha = new Mocha({ ui: "bdd", color: true, timeout: 30000 });
  const testsRoot = path.resolve(__dirname, "..");

  return new Promise((resolve, reject) => {
    glob("**/extension.test.js", { cwd: testsRoot }).then((files: string[]) => {
      files.forEach((f: string) => mocha.addFile(path.resolve(testsRoot, f)));
      try {
        mocha.run((failures: number) => {
          if (failures > 0) reject(new Error(`${failures} tests failed`));
          else resolve();
        });
      } catch (err) {
        reject(err);
      }
    }).catch(reject);
  });
}
```

(`glob` is pinned as an explicit devDependency in Task 1's `package.json`. Version 10+ uses the promise-based API used here.)

- [ ] **Step 3: Create `test/extension.test.ts` with one smoke test.**

```typescript
import * as assert from "assert";
import * as vscode from "vscode";

describe("extension integration", () => {
  it("activates and produces completions inside map(", async () => {
    const doc = await vscode.workspace.openTextDocument({
      language: "edge",
      content: "Z[m] = A . B :: map(",
    });
    await vscode.window.showTextDocument(doc);

    // Cursor at end of the line: position.character = doc.lineAt(0).text.length.
    const lineLength = doc.lineAt(0).text.length;
    const position = new vscode.Position(0, lineLength);

    const completionList = await vscode.commands.executeCommand<vscode.CompletionList>(
      "vscode.executeCompletionItemProvider",
      doc.uri,
      position,
    );

    assert.ok(completionList, "expected a CompletionList result");
    assert.ok(
      completionList.items.length >= 7,
      `expected at least 7 compute-op completions, got ${completionList.items.length}`,
    );
    const labels = completionList.items.map((it) =>
      typeof it.label === "string" ? it.label : it.label.label,
    );
    assert.ok(labels.includes("add"), "completions should include 'add'");
    assert.ok(labels.includes("mul"), "completions should include 'mul'");
  });
});
```

- [ ] **Step 4: Recompile.**

```bash
cd editors/vscode && npm run compile
```
Expected: zero errors. (`glob` v10+ ships with built-in TypeScript types — no `@types/glob` needed.)

- [ ] **Step 5: Run the integration test suite.**

```bash
cd editors/vscode && npm test
```

This downloads VS Code (first run only, ~100 MB to `.vscode-test/`), launches it headlessly, runs Mocha against `out/test/extension.test.js`. Expected: 1 passing test, ~10–15 seconds total.

If activation fails: inspect the test runner output for stack traces. Common issues:
- `package.json`'s `"main"` doesn't point to `out/src/extension.js`.
- The grammar JSON has invalid syntax (VS Code logs the parse error in stderr).
- `activationEvents` in `package.json` is wrong — should be `["onLanguage:edge"]`.

- [ ] **Step 6: Run unit tests too, to confirm nothing regressed.**

```bash
cd editors/vscode && npx mocha out/test/completions.test.js
```
Expected: still passing.

- [ ] **Step 7: Commit.**

```bash
git add editors/vscode/test/extension.test.ts editors/vscode/test/runTest.ts editors/vscode/test/suite/index.ts editors/vscode/package.json editors/vscode/package-lock.json
git commit -m "vscode: integration smoke test via @vscode/test-electron"
```

(Include `package.json` and `package-lock.json` only if you added the optional `@types/glob` dep.)

---

## Task 9: README and CHANGELOG

**Files:**
- Create: `editors/vscode/README.md`
- Create: `editors/vscode/CHANGELOG.md`

End-user docs that ship inside the extension package. The README appears on the extension's marketplace page (if/when published) and inside VS Code's extension panel.

- [ ] **Step 1: Create `editors/vscode/README.md`.**

```markdown
# EDGE Language Support

VS Code extension providing syntax highlighting and static-list autocomplete for the EDGE tensor language.

## Features

- **Syntax highlighting** for `dim`, `tensor`, `init`, `map`, `reduce`, `empty`, types (`integer`, `float`, `boolean`), constants (`true`, `false`, `inf`), numeric literals, line comments, and operators.
- **Autocomplete dropdowns** with hardcoded suggestions at six trigger contexts:
  - First arg of `map(...)` / `reduce(...)`: 7 compute operators (`add`, `sub`, `mul`, `min`, `max`, `and`, `or`).
  - Second arg of `map(...)` / `reduce(...)`: 9 merge operators (`passthrough`, `nopass`, `intersect`, `take_left`, `take_left_only`, `take_right`, `take_right_only`, `xor`, `union`). Each merge operator's tooltip includes its truth table.
  - After `:` in a tensor declaration: type names (`integer`, `float`, `boolean`).
  - After `empty = ` in a tensor declaration: common values (`0`, `0.0`, `inf`, `true`, `false`).

## Limitations (Tier 1)

This extension does **not** parse `.edge` files. As a result:
- No diagnostics or red squiggles. Use `edgecc` from the command line for error messages.
- No context-aware completions: tensor names and dim names are not suggested. Types and operators are hardcoded lists.
- No hover info, go-to-definition, or rename.

These features are planned for a future Tier 2/3 release that adds a Language Server.

## Build and install

Requirements: Node.js 18+, npm, VS Code 1.80+.

```bash
cd editors/vscode
npm install
npm run package        # produces edgecc-0.1.0.vsix
code --install-extension edgecc-0.1.0.vsix
```

Then reload VS Code (Cmd+Shift+P → `Developer: Reload Window`).

To uninstall: `code --uninstall-extension serban-edgecc.edgecc`.

## Develop

Open `editors/vscode/` as a VS Code workspace and press **F5**. A new VS Code window launches with the extension live-loaded ("Extension Development Host"). Edit the source, save, and reload the dev host (Cmd+R inside the dev window).

## Test

```bash
npm run test:grammar   # TextMate grammar snapshot tests (fast)
npm test               # Full suite incl. integration smoke test
```
```

- [ ] **Step 2: Create `editors/vscode/CHANGELOG.md`.**

```markdown
# Changelog

## 0.1.0 — 2026-05-06

Initial release. Tier 1 scope:

- TextMate grammar for `.edge` files (syntax highlighting).
- Language configuration (line comments, bracket pairs, indent rules).
- Static-list autocomplete for compute operators, merge operators, types, and empty values.
- Truth-table tooltips on merge operator completions.
- Local-install only; no marketplace publishing.

No semantic features (diagnostics, hover, go-to-definition) — those require a Language Server, planned for a future release.
```

- [ ] **Step 3: Commit.**

```bash
git add editors/vscode/README.md editors/vscode/CHANGELOG.md
git commit -m "vscode: add README and CHANGELOG"
```

---

## Task 10: CLAUDE.md update and final clean-build verification

**Files:**
- Modify: `CLAUDE.md`

Adds a short "VS Code extension" section to the repo-level CLAUDE.md so a future session can find the extension and know how to build/test/install it. Then runs a clean build of *both* the C compiler and the VS Code extension as a final integration sanity check.

- [ ] **Step 1: Edit `CLAUDE.md` — append a new section before "v2 Scope Limitations".**

Insert this section after the existing "Iterative Tests" section and before "v2 Scope Limitations":

```markdown
## VS Code extension

Lives in `editors/vscode/`. Provides syntax highlighting and static-list autocompletion for `.edge` files.

Build: `cd editors/vscode && npm install && npm run package`
Install: `code --install-extension edgecc-0.1.0.vsix`
Test: `cd editors/vscode && npm test` (full suite) or `npm run test:grammar` (grammar only, fast)
Develop: open `editors/vscode/` in VS Code and press F5 (Extension Development Host).
Requires Node.js 18+.

Coverage is Tier 1 only — static dropdowns for compute / merge / type / empty-value slots; no parsing, no diagnostics, no go-to-definition. Tier 2 features (context-aware completions) and Neovim port are deferred to future work.
```

- [ ] **Step 2: Verify the C compiler still builds cleanly from scratch.**

```bash
cd /Users/porumbes/Dropbox/Home/serban/Projects/Owens_Group/Researchers/Toluwanimi_Odemuyiwa/code/edgecc
rm -rf build
mkdir build && cd build
cmake .. && make && ctest --output-on-failure
```

Expected: 8 ctest groups all pass. The extension work is fully orthogonal — the C compiler and its test suite must be unaffected.

- [ ] **Step 3: Verify the VS Code extension builds cleanly from scratch.**

```bash
cd /Users/porumbes/Dropbox/Home/serban/Projects/Owens_Group/Researchers/Toluwanimi_Odemuyiwa/code/edgecc/editors/vscode
rm -rf node_modules out *.vsix .vscode-test
npm install
npm run compile
npm run test:grammar
npm test
npm run package
```

Expected: each step succeeds. The final `npm run package` produces `edgecc-0.1.0.vsix` in the `editors/vscode/` directory.

- [ ] **Step 4: Manual end-user install test.**

```bash
code --install-extension edgecc-0.1.0.vsix
```

Open any `.edge` file in VS Code (e.g., `examples/gemm.edge`). Reload the window if needed. Verify:
- Keywords are colored.
- Typing `map(` opens a dropdown of compute ops.
- Typing `, ` after a compute op opens a dropdown of merge ops.
- Hovering over `intersect` in the dropdown shows its truth table.

If any of these don't work, inspect the VS Code Developer Tools console for errors.

- [ ] **Step 5: Commit.**

```bash
cd /Users/porumbes/Dropbox/Home/serban/Projects/Owens_Group/Researchers/Toluwanimi_Odemuyiwa/code/edgecc
git add CLAUDE.md
git commit -m "docs: document VS Code extension in CLAUDE.md"
```

---

## Self-Review Checklist (for the engineer)

After completing all tasks, sanity-check against the spec (`docs/superpowers/specs/2026-05-06-edgecc-ide-integration-design.md`):

- [ ] Six user-visible completion slots all work: compute op after `map(`, compute op after `reduce(`, merge op after `,` in either, type after `:` in a tensor decl, empty value after `empty = `.
- [ ] All 9 merge operator completions include a markdown truth table in their documentation.
- [ ] Sort order matches the spec: compute ops `add, mul, min, max, sub, and, or`; merge ops `passthrough, intersect, union, take_left, take_right, take_left_only, take_right_only, xor, nopass`.
- [ ] Inside a `// ...` comment, no completion dropdown fires.
- [ ] Inside a tensor's rank list (`tensor X[R=K, `), no completion dropdown fires (no map/reduce in context).
- [ ] All 9 supported merge operator keywords are colored as `support.function.merge.edge` only when inside `map(...)` or `reduce(...)`.
- [ ] All 7 compute operator keywords are similarly colored as `support.function.compute.edge` only inside actions.
- [ ] `npm run test:grammar` passes (6 fixtures).
- [ ] `npm test` passes (unit tests + 1 integration smoke test).
- [ ] `npm run package` produces a `.vsix` that installs cleanly via `code --install-extension`.
- [ ] The C compiler build (`cd build && cmake .. && make && ctest`) is unaffected — all 8 ctest groups still pass.
- [ ] CLAUDE.md describes the VS Code extension's location, build, install, test, and dev workflow.
