import * as vscode from "vscode";
import { EdgeCompletionProvider } from "./provider";

export function activate(context: vscode.ExtensionContext): void {
  // No `scheme` filter so that both saved `file:` documents AND in-memory
  // `untitled:` documents (created via openTextDocument({ content, language }),
  // as the integration smoke test does) both get completions.
  const provider = vscode.languages.registerCompletionItemProvider(
    { language: "edge" },
    new EdgeCompletionProvider(),
    "(", ",", ":", "=",
  );
  context.subscriptions.push(provider);
}

export function deactivate(): void {
  // No cleanup needed; subscriptions disposed automatically by VS Code.
}
