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
npm run test:grammar   # TextMate grammar snapshot tests (fast; ~1 sec)
npm test               # Full suite incl. integration smoke test (~10 sec first run, ~5 sec after)
```
