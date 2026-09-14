import { existsSync } from "node:fs";
import { spawn } from "node:child_process";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

// Use the repository's ML environment on Windows, macOS and Linux without
// activating a shell or accidentally selecting a different Python installation.
const root = fileURLToPath(new URL("../", import.meta.url));
const python = join(root, ".venv", process.platform === "win32" ? "Scripts/python.exe" : "bin/python");
if (!existsSync(python)) {
  console.error("Create the ML environment first: python -m venv .venv");
  console.error("Then install ./ml[train] using that environment's Python. See docs/live-feed.md.");
  process.exitCode = 1;
} else {
  const child = spawn(python, process.argv.slice(2), { cwd: root, stdio: "inherit", shell: false, windowsHide: true });
  child.on("error", () => { console.error("Could not start the repository's ML environment."); process.exitCode = 1; });
  child.on("exit", (code, signal) => { process.exitCode = code ?? (signal ? 1 : 0); });
}
