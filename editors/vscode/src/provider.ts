import {
  CompletionItem,
  CompletionItemKind,
  CompletionItemProvider,
  Position,
  TextDocument,
  CancellationToken,
  CompletionContext,
  ProviderResult,
  MarkdownString,
} from "vscode";

import {
  ItemSpec,
  COMPUTE_OPS,
  MERGE_OPS,
  TYPES,
  EMPTY_VALUES,
  detectContext,
} from "./completions";

function toCompletionItems(specs: ItemSpec[]): CompletionItem[] {
  return specs.map((spec) => {
    const item = new CompletionItem(spec.label, spec.kind as CompletionItemKind);
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
