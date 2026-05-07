import * as assert from "assert";
import * as vscode from "vscode";

describe("extension integration", () => {
  it("activates and produces completions inside map(", async () => {
    // Ensure the extension is activated before requesting completions.
    // Without this, executeCompletionItemProvider may run before the
    // activation event has fired, returning only the built-in word-based
    // completions instead of our static dropdowns.
    const ext = vscode.extensions.getExtension("serban-edgecc.edgecc");
    assert.ok(ext, "edgecc extension should be discoverable");
    await ext.activate();

    const doc = await vscode.workspace.openTextDocument({
      language: "edge",
      content: "Z[m] = A . B :: map(",
    });
    await vscode.window.showTextDocument(doc);

    // Cursor at end of the line.
    const lineLength = doc.lineAt(0).text.length;
    const position = new vscode.Position(0, lineLength);

    const completionList = await vscode.commands.executeCommand<vscode.CompletionList>(
      "vscode.executeCompletionItemProvider",
      doc.uri,
      position,
    );

    assert.ok(completionList, "expected a CompletionList result");
    const labels = completionList.items.map((it) =>
      typeof it.label === "string" ? it.label : it.label.label,
    );
    // Each compute-op label should appear. If the extension didn't activate
    // or the provider didn't fire, we'd see only built-in word completions
    // (e.g., the identifiers from the buffer text) and these would be missing.
    for (const expected of ["add", "mul", "min", "max", "sub", "and", "or"]) {
      assert.ok(
        labels.includes(expected),
        `completions should include '${expected}'; got ${JSON.stringify(labels)}`,
      );
    }
  });
});
