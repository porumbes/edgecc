import * as vscode from "vscode";
import { EdgeCompletionProvider } from "./provider";

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
