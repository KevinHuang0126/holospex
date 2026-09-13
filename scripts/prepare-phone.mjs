import { copyFile, lstat, mkdir, readFile, readdir, realpath, rm, writeFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { build as bundle } from "esbuild";

const root = fileURLToPath(new URL("../", import.meta.url));
const build = join(root, "apps", "web", "dist");
const html = await readFile(join(build, "index.html"), "utf8");
const assets = await readdir(join(build, "assets"));
if (!assets.length || assets.some(name => !/^[\w-]+\.(js|css)$/.test(name))) {
  throw new Error("Review the build assets before packaging: expected bundled JavaScript and CSS only.");
}
for (const match of html.matchAll(/(?:src|href)="\/assets\/([^"]+)"/g)) {
  if (!assets.includes(match[1])) throw new Error(`Missing built asset: ${match[1]}`);
}
const fixtures = ["lesson.json", "frame-000.json", "frame-001.json", "synthetic-frame.svg", "mannequin-cutout.png"];
for (const name of fixtures) {
  const bytes = await readFile(join(build, "demo", name));
  if (!bytes.equals(await readFile(join(root, "assets", "demo", name)))) {
    throw new Error(`Demo asset differs from the selected source: ${name}`);
  }
}

// Bundle the API independently of the public client assets. Runtime environment
// variables remain process.env reads; no credential values or .env files are copied.
const placement = await bundle({
  entryPoints: [join(root, "api", "placement.ts")], bundle: true, write: false,
  platform: "node", format: "esm", target: "node22", sourcemap: false,
  outfile: "placement.mjs", logLevel: "warning",
});
if (placement.outputFiles.length !== 1) throw new Error("Expected one self-contained placement function.");

const staging = join(root, "runs", "holospex-mannequin-phone");
await mkdir(staging, { recursive: true });
const stagingRoot = await realpath(staging);
const publicDirectory = resolve(stagingRoot, "public");
if (dirname(publicDirectory) !== stagingRoot) throw new Error("Unexpected staging path.");
const existing = await lstat(publicDirectory).catch(error => { if (error.code !== "ENOENT") throw error; return null; });
if (existing?.isSymbolicLink()) throw new Error("Refusing to replace a linked staging directory.");
const apiDirectory = resolve(stagingRoot, "api");
if (dirname(apiDirectory) !== stagingRoot) throw new Error("Unexpected function staging path.");
const existingApi = await lstat(apiDirectory).catch(error => { if (error.code !== "ENOENT") throw error; return null; });
if (existingApi?.isSymbolicLink()) throw new Error("Refusing to replace a linked API staging directory.");
// Only verified generated children are replaced. Preserve Vercel's project link.
await rm(publicDirectory, { recursive: true, force: true });
await rm(apiDirectory, { recursive: true, force: true });
const files = ["index.html", ...assets.map(name => `assets/${name}`), ...fixtures.map(name => `demo/${name}`)];
for (const name of files) {
  const destination = join(publicDirectory, name);
  await mkdir(dirname(destination), { recursive: true });
  await copyFile(join(build, name), destination);
}
await mkdir(apiDirectory, { recursive: true });
await writeFile(join(apiDirectory, "placement.mjs"), placement.outputFiles[0].contents);
await writeFile(join(stagingRoot, "package.json"), JSON.stringify({
  name: "holospex-phone-demo", private: true, type: "module", engines: { node: "22.x" },
}, null, 2) + "\n");
// Upload only generated deployment inputs, even if local tooling adds other staging files.
await writeFile(join(stagingRoot, ".vercelignore"), "/*\n!public\n!api\n!package.json\n!vercel.json\n");
await writeFile(join(stagingRoot, "vercel.json"), JSON.stringify({
  $schema: "https://openapi.vercel.sh/vercel.json",
  framework: null, buildCommand: "", installCommand: "", outputDirectory: "public",
  functions: { "api/placement.mjs": { maxDuration: 60 } },
  redirects: [{ source: "/", destination: "/mannequin", permanent: false }],
  rewrites: ["/mannequin", "/samples", "/prototype"].map(source => ({ source, destination: "/index.html" })),
}, null, 2) + "\n");
console.log(`Prepared ${files.length} public files and api/placement.mjs in ${stagingRoot}`);
console.log(files.join("\n"));
