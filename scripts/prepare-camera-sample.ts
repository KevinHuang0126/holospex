import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { readLocalTrainingFiles, trainingReferenceDirectory } from "../apps/web/dev/localTraining";
import { importDatasetSamples } from "../apps/web/src/overlays/datasetSamples";
import { exportPortableSample } from "../apps/web/src/overlays/portableSample";
import { trainingReferencesOnly } from "../apps/web/src/overlays/trainingReferences";

const root = fileURLToPath(new URL("../", import.meta.url));
const directory = trainingReferenceDirectory(root, process.argv[2] || process.env.HOLOSPEX_TRAINING_REFERENCE_DIR);
const files = await readLocalTrainingFiles(directory);
const pack = trainingReferencesOnly(await importDatasetSamples(files.map(file => ({ name: file.name, size: file.bytes.length,
  arrayBuffer: async () => Uint8Array.from(file.bytes).buffer }))));
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
console.log("Prepared training-reference camera packs. Copy a file to your phone, then open it under Labeled image on /mannequin. These files stay out of the public deployment and Git.");
