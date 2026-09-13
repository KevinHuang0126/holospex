import { mkdir, writeFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { readLocalSampleFiles } from "../apps/web/dev/localSamples";
import { importDatasetSamples } from "../apps/web/src/overlays/datasetSamples";
import { exportPortableSample } from "../apps/web/src/overlays/portableSample";

const root = fileURLToPath(new URL("../", import.meta.url));
const directory = resolve(root, process.argv[2] || "apps/web/tests/samples_tst");
const files = await readLocalSampleFiles(directory);
const pack = await importDatasetSamples(files.map(file => ({ name: file.name, size: file.bytes.length,
  arrayBuffer: async () => Uint8Array.from(file.bytes).buffer })));
if (!pack.samples.length) throw new Error(pack.issues.join(" "));
const destination = join(root, "runs", "camera-samples");
await mkdir(destination, { recursive: true });
for (const sample of pack.samples) {
  const blob = await exportPortableSample(sample);
  const path = join(destination, `case-${sample.id}.holospex.json`);
  await writeFile(path, new Uint8Array(await blob.arrayBuffer()));
  console.log(`${path} (${Math.round(blob.size / 1024)} KB)`);
}
for (const issue of pack.issues) console.warn(issue);
console.log("Copy one file to your phone, then open it under Labeled image on /mannequin. These files stay out of the public deployment and Git.");
