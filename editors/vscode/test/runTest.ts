import * as path from "path";
import { runTests } from "@vscode/test-electron";

async function main() {
  try {
    // Path to the extension manifest's directory (one above `out/`).
    // __dirname at runtime is `editors/vscode/out/test/`.
    const extensionDevelopmentPath = path.resolve(__dirname, "../..");
    // Path to the compiled test loader.
    const extensionTestsPath = path.resolve(__dirname, "./suite/index");

    await runTests({ extensionDevelopmentPath, extensionTestsPath });
  } catch (err) {
    console.error("Failed to run integration tests:", err);
    process.exit(1);
  }
}

main();
