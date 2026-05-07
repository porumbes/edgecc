/**
 * Local mirror of vscode.CompletionItemKind for the four kinds we use.
 * Numeric values match the VS Code API exactly, so the data layer can be
 * unit-tested without spinning up VS Code (no `vscode` runtime dep here).
 * The provider in extension.ts maps these to the real vscode enum.
 */
export const CompletionKind = {
  Operator: 23,
  EnumMember: 19,
  Keyword: 13,
  Value: 11,
} as const;
export type CompletionKind = typeof CompletionKind[keyof typeof CompletionKind];

export interface ItemSpec {
  label: string;
  kind: CompletionKind;
  detail: string;
  documentation?: string;
  sortText?: string;
}

/**
 * Build a markdown truth-table documentation string for a merge operator.
 * Cells are listed in (NoNo, NoYes, YesNo, YesYes) order — the same column
 * order used in the v2 design spec.
 */
function truthTable(
  name: string,
  symbol: string,
  cells: [string, string, string, string],
): string {
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
  { label: "add", kind: CompletionKind.Operator, detail: "Addition",
    documentation: "Binary `+`. Identity for reduce: `0`.", sortText: "1-add" },
  { label: "mul", kind: CompletionKind.Operator, detail: "Multiplication",
    documentation: "Binary `*`. Identity for reduce: `1`.", sortText: "2-mul" },
  { label: "min", kind: CompletionKind.Operator, detail: "Minimum",
    documentation: "`min(a, b)`. Identity: `+inf` (float) or `INT_MAX` (integer).", sortText: "3-min" },
  { label: "max", kind: CompletionKind.Operator, detail: "Maximum",
    documentation: "`max(a, b)`. Identity: `-inf` (float) or `INT_MIN` (integer).", sortText: "4-max" },
  { label: "sub", kind: CompletionKind.Operator, detail: "Subtraction",
    documentation: "Binary `-`. Identity for reduce: `0` (right-identity only).", sortText: "5-sub" },
  { label: "and", kind: CompletionKind.Operator, detail: "Boolean AND",
    documentation: "Boolean `and`. Identity: `true`. Boolean tensors only.", sortText: "6-and" },
  { label: "or", kind: CompletionKind.Operator, detail: "Boolean OR",
    documentation: "Boolean `or`. Identity: `false`. Boolean tensors only.", sortText: "7-or" },
];

export const MERGE_OPS: ItemSpec[] = [
  { label: "passthrough", kind: CompletionKind.EnumMember, detail: "Always admits (default if omitted)",
    documentation: truthTable("Pass-through merge", "𝟙", ["**Yes**", "**Yes**", "**Yes**", "**Yes**"]),
    sortText: "1-passthrough" },
  { label: "intersect", kind: CompletionKind.EnumMember, detail: "Both operands non-empty",
    documentation: truthTable("Intersection merge", "∩", ["No", "No", "No", "**Yes**"]),
    sortText: "2-intersect" },
  { label: "union", kind: CompletionKind.EnumMember, detail: "At least one non-empty",
    documentation: truthTable("Union merge", "∪", ["No", "**Yes**", "**Yes**", "**Yes**"]),
    sortText: "3-union" },
  { label: "take_left", kind: CompletionKind.EnumMember, detail: "Left operand non-empty",
    documentation: truthTable("Take-left merge", "←", ["No", "No", "**Yes**", "**Yes**"]),
    sortText: "4-take_left" },
  { label: "take_right", kind: CompletionKind.EnumMember, detail: "Right operand non-empty",
    documentation: truthTable("Take-right merge", "→", ["No", "**Yes**", "No", "**Yes**"]),
    sortText: "5-take_right" },
  { label: "take_left_only", kind: CompletionKind.EnumMember, detail: "Left non-empty AND right empty",
    documentation: truthTable("Take-left-only merge", "⊖_l", ["No", "No", "**Yes**", "No"]),
    sortText: "6-take_left_only" },
  { label: "take_right_only", kind: CompletionKind.EnumMember, detail: "Right non-empty AND left empty",
    documentation: truthTable("Take-right-only merge", "⊖_r", ["No", "**Yes**", "No", "No"]),
    sortText: "7-take_right_only" },
  { label: "xor", kind: CompletionKind.EnumMember, detail: "Exactly one non-empty",
    documentation: truthTable("Exclusive-or merge", "⊕", ["No", "**Yes**", "**Yes**", "No"]),
    sortText: "8-xor" },
  { label: "nopass", kind: CompletionKind.EnumMember, detail: "Always rejects",
    documentation: truthTable("No-pass merge", "𝔽", ["No", "No", "No", "No"]),
    sortText: "9-nopass" },
];

export const TYPES: ItemSpec[] = [
  { label: "integer", kind: CompletionKind.Keyword, detail: "Integer-typed tensor" },
  { label: "float",   kind: CompletionKind.Keyword, detail: "Floating-point-typed tensor" },
  { label: "boolean", kind: CompletionKind.Keyword, detail: "Boolean-typed tensor" },
];

export const EMPTY_VALUES: ItemSpec[] = [
  { label: "0",     kind: CompletionKind.Value, detail: "Integer zero" },
  { label: "0.0",   kind: CompletionKind.Value, detail: "Float zero" },
  { label: "inf",   kind: CompletionKind.Value, detail: "Float infinity (often used for min)" },
  { label: "true",  kind: CompletionKind.Value, detail: "Boolean true" },
  { label: "false", kind: CompletionKind.Value, detail: "Boolean false" },
];
