# edgecc IDE Integration Design Spec — VS Code Extension (Tier 1)

A VS Code extension that gives `.edge` files syntax highlighting and static-list autocomplete for compute operators, merge operators, types, and empty values. Tier 1 means: no parsing in the editor, no diagnostics, no semantic features. Just colors and dropdowns.

## Background

`edgecc` is a working compiler for the EDGE tensor language (v2 merged 2026-05-06; see `2026-05-06-edgecc-v2-merge-design.md` for the merge-operator work). Today, anyone editing a `.edge` file gets plain gray text with no colors, no autocomplete, and no help remembering the 9 merge operators or the 7 compute operators. The compiler's CLI is still required to find typos. This spec adds editor-side quality-of-life features for the most common authoring tasks: writing tensor declarations, einsum expressions, and choosing operators inside `map(...)` / `reduce(...)`.

The user's primary editor is VS Code; Neovim is on the future roadmap.

## Scope

**In scope:**
- VS Code extension shipped as a local-install `.vsix` (not yet marketplace-published).
- Syntax highlighting via a TextMate grammar (`source.edge`) with conventional scope names (`keyword.control.action.edge`, `support.function.merge.edge`, `constant.numeric.float.edge`, etc.).
- Language configuration: `//` line comments, bracket-pair matching for `[]` and `()`, basic indentation rules.
- Static-list completions at six user-visible slots, detected by four regex patterns (map and reduce share the compute-op and merge-op detectors):
  | Context | Suggestions | Count |
  |---|---|---|
  | After `map(` (1st arg) | `add`, `sub`, `mul`, `min`, `max`, `and`, `or` | 7 |
  | After `reduce(` (1st arg) | (same compute ops) | 7 |
  | After `,` inside `map(...)` (2nd arg) | `passthrough`, `nopass`, `intersect`, `take_left`, `take_left_only`, `take_right`, `take_right_only`, `xor`, `union` | 9 |
  | After `,` inside `reduce(...)` (2nd arg) | (same merge ops) | 9 |
  | After `:` in tensor decl | `integer`, `float`, `boolean` | 3 |
  | After `empty = ` | `0`, `0.0`, `inf`, `true`, `false` | 5 |
- Each merge-op completion item carries a markdown documentation tooltip including its 4-row truth table.
- Test coverage: TextMate grammar snapshot tests + completion-provider unit tests + one VS Code integration smoke test.

**Out of scope (deferred):**
- Tree-sitter grammar. (Re-evaluated when Neovim support is the active project; EDGE's grammar is small enough to port at that point.)
- Language Server Protocol — no diagnostics, no hover tooltips, no go-to-definition, no rename.
- Context-aware completions — suggesting tensor names declared earlier in the file, dim names from the current scope, etc.
- Type-aware filtering on the empty-value dropdown.
- Marketplace publishing (Microsoft VS Code Marketplace, Open VSX).
- Snippets (`gemm` → tab-expanded GEMM template).
- Code formatting / auto-format-on-save.
- Heuristic identifier sub-coloring (e.g., styling ALL-CAPS dim/rank names differently). All identifiers get one scope (`variable.other.edge`).

## Approach

We chose Approach A: **TextMate grammar + thin VS Code extension with hardcoded completion lists.** This is the standard VS Code path — TextMate is the editor's native highlighter, and the completion provider is a few hundred lines of TypeScript using VS Code's `CompletionItemProvider` API. No tree-sitter, no language server.

Rationale:
- The user said "standard and easiest." TextMate is both for VS Code; tree-sitter requires more setup *specifically* in VS Code (WASM bundling or third-party host extensions).
- Neovim was explicitly deferred.
- EDGE's surface grammar is small (the entire BNF fits on one paper page in `tex/notes/main.ebnf-cropped.tex`). When Neovim becomes the active target, porting to a tree-sitter grammar is a day of work, not a multi-week project. The TextMate grammar is throwaway, and the cost is low.
- The completion-provider TypeScript code in this design is fully reusable in a future LSP server — that's where the durable value lives.

## Repo Layout

```
edgecc/
├── editors/
│   └── vscode/
│       ├── package.json                 (extension manifest)
│       ├── language-configuration.json  (brackets, comments, indent rules)
│       ├── syntaxes/
│       │   └── edge.tmLanguage.json     (TextMate grammar)
│       ├── src/
│       │   ├── extension.ts             (activate/deactivate; provider registration)
│       │   └── completions.ts           (provider class + hardcoded item data)
│       ├── test/
│       │   ├── grammar/                 (TextMate snapshot fixtures)
│       │   │   ├── dim_decl.test.edge
│       │   │   ├── tensor_decl.test.edge
│       │   │   ├── init.test.edge
│       │   │   ├── einsum_basic.test.edge
│       │   │   ├── einsum_with_merge.test.edge
│       │   │   └── comments.test.edge
│       │   ├── completions.test.ts      (pure-data unit tests)
│       │   ├── extension.test.ts        (one VS Code integration smoke test)
│       │   └── runTest.ts               (entrypoint for vscode-test)
│       ├── tsconfig.json
│       ├── .vscodeignore
│       ├── README.md
│       └── CHANGELOG.md
├── src/                                 (existing C compiler — unchanged)
├── tests/                               (existing C/Python test suites — unchanged)
├── docs/
└── ...
```

The extension lives **in the same monorepo as the compiler**, not a separate repo. The keyword lists in the extension must stay in sync with the compiler's lexer/parser (e.g., adding a new merge operator means editing both); monorepo makes that drift visible in a single PR.

`editors/` is a top-level directory that leaves room for `editors/neovim/` later.

## Toolchain

Developer-side requirements (not needed by end users):
- Node.js 18+ and npm.
- TypeScript (compiled to JavaScript via `tsc`).
- `@vscode/vsce` — official "VS Code Extension Manager" CLI for packaging.
- `@types/vscode` for VS Code API typings.
- Mocha + `@vscode/test-electron` for integration tests.
- `vscode-tmgrammar-test` for grammar snapshot tests.

End users only need VS Code (1.80+) plus the `.vsix` file.

## Grammar (TextMate)

### Token → scope mapping

| EDGE construct | TextMate scope |
|---|---|
| `//` line comment | `comment.line.double-slash.edge` |
| `dim`, `tensor`, `init` | `storage.type.declaration.edge` |
| `empty` | `keyword.other.empty.edge` |
| `map`, `reduce` (action keywords) | `keyword.control.action.edge` |
| `integer`, `float`, `boolean` | `storage.type.edge` |
| `true`, `false`, `inf` | `constant.language.edge` |
| `[0-9]+\.[0-9]+` | `constant.numeric.float.edge` |
| `[0-9]+` | `constant.numeric.integer.edge` |
| Compute ops *inside `map(…)` or `reduce(…)`* | `support.function.compute.edge` |
| Merge ops *inside `map(…)` or `reduce(…)`* | `support.function.merge.edge` |
| `::` separator | `keyword.operator.action-sep.edge` |
| `=` | `keyword.operator.assignment.edge` |
| `.` (between einsum operands) | `keyword.operator.dot.edge` |
| `,`, `;` | `punctuation.separator.edge` |
| `:` (in tensor decl) | `punctuation.colon.edge` |
| `[`, `]`, `(`, `)` | `punctuation.bracket.edge` |
| All identifiers (tensor / dim / rank names, rank vars) | `variable.other.edge` |

Without parsing, the grammar cannot distinguish a tensor name from a rank variable from a dim name — they all match `[a-zA-Z_][a-zA-Z0-9_]*`. They share one scope.

### Context-aware regions

Compute ops and merge ops share their lexical form (`add`, `intersect`, etc.) with regular identifiers. To color them as operators *only when they appear inside* `map(...)` or `reduce(...)`, the grammar uses TextMate `begin`/`end` regions:

```jsonc
{
  "name": "edge-action.map",
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
    { "match": "\\b(add|sub|mul|min|max|and|or)\\b",
      "name": "support.function.compute.edge" },
    { "match": "\\b(passthrough|nopass|intersect|take_left|take_left_only|take_right|take_right_only|xor|union)\\b",
      "name": "support.function.merge.edge" },
    { "match": ",", "name": "punctuation.separator.edge" }
  ]
}
```

A symmetric `edge-action.reduce` region for `reduce(...)`. Outside these regions, `add` / `intersect` / etc. fall through to `variable.other.edge` — fine, because no real EDGE program names a tensor `add`.

A similar `edge-tensor-decl` region (begin: `\\btensor\\b`, end: `;`) ensures the `:` and the type that follows are scoped correctly.

### Top-level grammar shape

```jsonc
{
  "name": "EDGE",
  "scopeName": "source.edge",
  "fileTypes": ["edge"],
  "patterns": [
    { "include": "#comments" },
    { "include": "#dim-decl" },
    { "include": "#tensor-decl" },
    { "include": "#init-stmt" },
    { "include": "#einsum-stmt" }
  ],
  "repository": {
    "comments": { /* line comments */ },
    "dim-decl": { /* `dim X = N;` */ },
    "tensor-decl": { /* `tensor X[...] : type, empty = val;` region */ },
    "init-stmt": { /* `init X = literal;` */ },
    "einsum-stmt": {
      "patterns": [
        { "include": "#action-map" },
        { "include": "#action-reduce" },
        { "include": "#literals" },
        { "include": "#operators" },
        { "include": "#identifiers" }
      ]
    },
    "action-map": { /* begin/end as above */ },
    "action-reduce": { /* mirror */ },
    "literals": { /* numeric, true/false/inf */ },
    "operators": { /* ::, =, ., ,, ;, [, ], (, ) */ },
    "identifiers": {
      "match": "\\b[a-zA-Z_][a-zA-Z0-9_]*\\b",
      "name": "variable.other.edge"
    }
  }
}
```

Total grammar size: ~80–120 lines of JSONC (JSON-with-comments — VS Code's grammar loader supports it natively, no preprocessor needed).

## Language Configuration

`language-configuration.json`:

```jsonc
{
  "comments": { "lineComment": "//" },
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

The `indentationRules` are a courtesy for multi-line tensor inits like `init A = [\n  [1, 2],\n  [3, 4]\n];`. EDGE has no `{}` blocks and no string literals, so the configuration stays small.

## Completion Provider

### File structure

```
src/extension.ts       — activate(): registers EdgeCompletionProvider with trigger characters
src/completions.ts     — EdgeCompletionProvider class + ItemSpec data + context-detection helpers
```

Internally `completions.ts` separates the **pure data layer** (`COMPUTE_OPS`, `MERGE_OPS`, `TYPES`, `EMPTY_VALUES`, `detectContext()`) from the **provider class** (which integrates with VS Code's API). The data layer is unit-testable without a VS Code instance.

### Item data

Each completion item is described by:

```typescript
interface ItemSpec {
  label: string;
  kind: CompletionItemKind;
  detail: string;
  documentation?: string;  // markdown
  sortText?: string;
}
```

Full lists:

```typescript
const COMPUTE_OPS: ItemSpec[] = [
  { label: "add", kind: CompletionItemKind.Operator, detail: "Addition",
    documentation: "Binary `+`. Identity for reduce: `0`." },
  { label: "sub", kind: CompletionItemKind.Operator, detail: "Subtraction",
    documentation: "Binary `-`. Identity for reduce: `0` (right-identity only)." },
  { label: "mul", kind: CompletionItemKind.Operator, detail: "Multiplication",
    documentation: "Binary `*`. Identity for reduce: `1`." },
  { label: "min", kind: CompletionItemKind.Operator, detail: "Minimum",
    documentation: "`min(a, b)`. Identity: `+inf` (float) or `INT_MAX` (integer)." },
  { label: "max", kind: CompletionItemKind.Operator, detail: "Maximum",
    documentation: "`max(a, b)`. Identity: `-inf` (float) or `INT_MIN` (integer)." },
  { label: "and", kind: CompletionItemKind.Operator, detail: "Boolean AND",
    documentation: "Boolean `and`. Identity: `true`. Boolean tensors only." },
  { label: "or",  kind: CompletionItemKind.Operator, detail: "Boolean OR",
    documentation: "Boolean `or`. Identity: `false`. Boolean tensors only." },
];

const MERGE_OPS: ItemSpec[] = [
  // Each item's documentation is a markdown string with the operator's
  // truth table over (A exists, B exists). The `intersect` example below
  // shows the format; remaining entries follow the same pattern, with the
  // truth-table outputs taken from the v2 design spec
  // (docs/superpowers/specs/2026-05-06-edgecc-v2-merge-design.md, Scope table).
  {
    label: "intersect",
    kind: CompletionItemKind.EnumMember,
    detail: "Both operands non-empty",
    documentation:
      "**Intersection merge** (∩). Compute runs only where both operands are non-empty.\n\n" +
      "| A | B | Z |\n" +
      "|---|---|---|\n" +
      "| No  | No  | No  |\n" +
      "| No  | Yes | No  |\n" +
      "| Yes | No  | No  |\n" +
      "| Yes | Yes | **Yes** |"
  },
  // Remaining 8 entries use identical structure; per-operator truth tables:
  //   passthrough  (𝟙):  T T T T   "Always admits (default if omitted)"
  //   nopass       (𝔽):  F F F F   "Always rejects"
  //   take_left    (←):  F F T T   "Left operand non-empty"
  //   take_left_only (⊖_l): F F T F  "Left non-empty AND right empty"
  //   take_right   (→):  F T F T   "Right operand non-empty"
  //   take_right_only (⊖_r): F T F F  "Right non-empty AND left empty"
  //   xor          (⊕):  F T T F   "Exactly one non-empty"
  //   union        (∪):  F T T T   "At least one non-empty"
  // (truth-table cells listed in order: NoNo, NoYes, YesNo, YesYes)
];

const TYPES: ItemSpec[] = [
  { label: "integer", kind: CompletionItemKind.Keyword, detail: "Integer-typed tensor" },
  { label: "float",   kind: CompletionItemKind.Keyword, detail: "Floating-point-typed tensor" },
  { label: "boolean", kind: CompletionItemKind.Keyword, detail: "Boolean-typed tensor" },
];

const EMPTY_VALUES: ItemSpec[] = [
  { label: "0",     kind: CompletionItemKind.Value, detail: "Integer zero" },
  { label: "0.0",   kind: CompletionItemKind.Value, detail: "Float zero" },
  { label: "inf",   kind: CompletionItemKind.Value, detail: "Float infinity (often used for min)" },
  { label: "true",  kind: CompletionItemKind.Value, detail: "Boolean true" },
  { label: "false", kind: CompletionItemKind.Value, detail: "Boolean false" },
];
```

### Context detection

Pure regex against the current line's prefix (everything left of the cursor), with line comments stripped:

```typescript
type Context = "compute-op" | "merge-op" | "type" | "empty-value" | null;

function detectContext(linePrefix: string): Context {
  // Strip everything after `//`
  const commentIdx = linePrefix.indexOf("//");
  const code = commentIdx >= 0 ? linePrefix.substring(0, commentIdx) : linePrefix;

  if (/\b(?:map|reduce)\s*\(\s*$/.test(code))             return "compute-op";
  if (/\b(?:map|reduce)\s*\([^,()]*,\s*$/.test(code))     return "merge-op";
  if (/\btensor\s+\w+\s*\[[^\]]*\]\s*:\s*$/.test(code))   return "type";
  if (/\bempty\s*=\s*$/.test(code))                       return "empty-value";
  return null;
}
```

The provider consults `detectContext` and returns the corresponding `ItemSpec[]` mapped to `CompletionItem`, or `undefined`. The four regex patterns detect four distinct contexts; map and reduce share the compute-op and merge-op detectors, which is why six user-visible slots collapse to four regexes. Any spurious match (a future EDGE feature putting `,` somewhere unanticipated) just shows a dismissible dropdown, never an error.

### Trigger characters

```typescript
context.subscriptions.push(
  languages.registerCompletionItemProvider(
    { language: "edge", scheme: "file" },
    new EdgeCompletionProvider(),
    "(", ",", ":", "="          // trigger characters
  )
);
```

Manual `Ctrl+Space` invocation also works (VS Code default).

### Sorting

Default alphabetical. Two opinionated overrides via `sortText`:
- **Compute ops:** `add`, `mul`, `min`, `max`, `sub`, `and`, `or` (most-frequent first; nudges toward GEMM/BFS canonical choices).
- **Merge ops:** `passthrough`, `intersect`, `union`, `take_left`, `take_right`, `take_left_only`, `take_right_only`, `xor`, `nopass` (everyday → rare).

## Build / Install Workflow

Inside `editors/vscode/`:
```bash
npm install          # one-time: dev deps
npm run compile      # tsc → out/extension.js
npm run package      # vsce package → edgecc-0.1.0.vsix
```

End-user install:
```bash
code --install-extension edgecc-0.1.0.vsix
```

…or via Command Palette: `Extensions: Install from VSIX…`.

For active extension development: open `editors/vscode/` as a VS Code workspace and press `F5`. A new VS Code window launches with the extension live-loaded against your dev source ("Extension Development Host"). No `vsce package` round-trip needed.

To uninstall: `code --uninstall-extension serban-edgecc.edgecc`.

## `package.json` Skeleton

```jsonc
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
    "languages": [{
      "id": "edge",
      "aliases": ["EDGE", "edge"],
      "extensions": [".edge"],
      "configuration": "./language-configuration.json"
    }],
    "grammars": [{
      "language": "edge",
      "scopeName": "source.edge",
      "path": "./syntaxes/edge.tmLanguage.json"
    }]
  },
  "scripts": {
    "compile": "tsc -p ./",
    "watch": "tsc -watch -p ./",
    "package": "vsce package",
    "test": "node ./out/test/runTest.js",
    "test:grammar": "vscode-tmgrammar-test \"test/grammar/**/*.test.edge\""
  }
}
```

The publisher name `serban-edgecc` is a placeholder for local-install only. A real Microsoft Marketplace or Open VSX publisher account would be required to publish.

## Testing Strategy

### A. TextMate grammar snapshot tests

Tool: `vscode-tmgrammar-test`. Fixtures in `editors/vscode/test/grammar/` with inline scope assertions:

```edge
dim K = 3;
//  ^^^ storage.type.declaration.edge
//      ^ variable.other.edge
//        ^ keyword.operator.assignment.edge
//          ^ constant.numeric.integer.edge
```

Six fixture files cover the major constructs (dim decl, tensor decl, init, einsum basic, einsum with merge, comments). `npm run test:grammar` tokenizes each and fails on mismatch. Snapshots auto-update with `--update`.

### B. Completion provider unit tests

Pure-data tests against `detectContext()` — millisecond-fast, no VS Code instance:

```typescript
expect(detectContext("Z[m] = A . B :: map(")).toBe("compute-op");
expect(detectContext("Z[m] = A . B :: map(mul, ")).toBe("merge-op");
expect(detectContext("// map(")).toBe(null);
expect(detectContext("tensor X[R=K, ")).toBe(null);
expect(detectContext("tensor X[R=K] : ")).toBe("type");
expect(detectContext("tensor X[R=K] : float, empty = ")).toBe("empty-value");
```

Roughly 15–20 assertions covering each context plus negatives.

### C. Integration smoke test (one only)

Tool: `@vscode/test-electron`. Spins up a real VS Code instance, opens an in-memory `.edge` file, requests completions at a known cursor position, asserts `items.length > 0`. Slow (~10s startup) but catches activation/registration bugs.

`npm test` runs all three; `npm run test:grammar` runs only the grammar layer for fast iteration.

## Repo Integration

`.gitignore` additions:
```
editors/vscode/node_modules/
editors/vscode/out/
editors/vscode/*.vsix
editors/vscode/.vscode-test/
```

`CLAUDE.md` gets a new section describing where the extension lives, how to build / install / test / develop, and the Tier 1 scope. Briefly:

```markdown
## VS Code extension

Lives in `editors/vscode/`. Provides syntax highlighting and static-list autocompletion for `.edge` files.

Build: `cd editors/vscode && npm install && npm run package`
Install: `code --install-extension edgecc-0.1.0.vsix`
Test: `cd editors/vscode && npm test`
Develop: open `editors/vscode/` in VS Code and press F5.
Requires Node.js 18+.

Coverage is Tier 1 only — static dropdowns for compute / merge / type / empty-value slots; no parsing, no diagnostics, no go-to-definition. Tier 2 features (context-aware completions) and Neovim port are deferred.
```

No changes to `CMakeLists.txt`, the C source, or the existing C/Python test runners. The extension build is fully orthogonal to the compiler build — a contributor without Node installed can still build and test the compiler.

## Versioning

Start at `0.1.0`. Bump to `0.2.0` when Tier 2 features land. `1.0.0` if/when published to a marketplace. `editors/vscode/CHANGELOG.md` tracks changes — kept short.

## Future Work

Recorded here so it's clear what's deliberately deferred:

- **Tier 2: context-aware completions.** Suggest only tensor names declared above the cursor; suggest dim names in the brackets of a tensor decl. Requires light in-extension parsing (probably regex-based for the simple cases, or invoking `edgecc --parse-only` for full fidelity).
- **Tier 3: Language Server (LSP).** Real diagnostics surfaced as red squiggles, hover tooltips with rendered truth tables on merge symbols, go-to-definition on tensor names. Reuse the existing `edgecc` parser/semantic via a thin LSP wrapper. Same server works for VS Code and Neovim.
- **Neovim port.** Tree-sitter grammar (~1 day given the BNF is small) + the same LSP server when Tier 3 lands. Both editors share the LSP; only the highlighting layer differs (TextMate vs tree-sitter).
- **Marketplace publishing.** Microsoft VS Code Marketplace and/or Open VSX. Requires a real publisher account.
- **Snippets.** `gemm` → tab-expanded GEMM template, `bfs_step` → tab-expanded BFS gather kernel, etc. Cheap to add when desired.
- **Identifier sub-coloring heuristic.** Optional all-caps → `variable.other.constant.edge` rule for distinguishing dim names visually. Imperfect (not all all-caps tokens are dim names) but cheap.

## Design Alternatives Recorded

These were considered and deferred. Future work may revisit them.

- **Approach B: Tree-sitter grammar + VS Code extension that consumes it.** Rejected for v1 because tree-sitter integration in VS Code is less mature than TextMate (WASM bundling or third-party host extensions required), adding setup friction without a v1 payoff. The grammar would be reusable for Neovim, but the user explicitly deferred Neovim.
- **Approach C: Write both TextMate and tree-sitter grammars now.** Rejected because maintaining two grammars permanently is a real cost; EDGE's grammar is small enough that re-writing it in tree-sitter when Neovim is the active project is genuinely cheap.
- **Tier 2 (context-aware) included in v1.** Rejected because it requires either light in-editor parsing or a `--parse-only` flag on `edgecc`, both of which expand the project significantly. v1 stays "no parsing, hardcoded lists" by design.
- **Tier 3 (LSP) included in v1.** Rejected for the same reason — LSP is a multi-week project. The compiler core is reused, but JSON-RPC, document state management, debouncing, and diagnostic mapping all need real engineering.
- **Snippets included in v1.** Rejected as nice-to-have-but-not-asked-for; can be added cheaply later once usage patterns are clearer.
- **Top-level Makefile glue.** Rejected — `npm` workflow stays in `editors/vscode/`; the C compiler build doesn't need Node, and the extension build doesn't need CMake. CLAUDE.md documents both.
- **Heuristic ALL-CAPS identifier sub-scope.** Considered but rejected as default — imperfect (false positives on UDF names that happen to be all-caps). Captured as optional future work.
