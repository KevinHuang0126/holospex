import { resolve } from "node:path";
import type { Plugin } from "vite";
import { readLocalSampleFiles, type SampleFile } from "./localSamples";
import { importDatasetSamples, parseDatasetSample } from "../src/overlays/datasetSamples";
import { trainingReferencesOnly } from "../src/overlays/trainingReferences";

const DEFAULT_TRAINING_REFERENCE_DIRECTORY = "runs/training-reference";
/** Private server path; overrides can point to an explicitly selected prepared export. */
export function trainingReferenceDirectory(repositoryRoot: string, override?: string) {
  return resolve(repositoryRoot, override?.trim() || DEFAULT_TRAINING_REFERENCE_DIRECTORY);
}

/** Only complete, validated train-split image/label pairs are exposed automatically. */
export async function readLocalTrainingFiles(directory: string): Promise<SampleFile[]> {
  let candidates: SampleFile[];
  try { candidates = await readLocalSampleFiles(directory); }
  catch (cause) {
    if ((cause as NodeJS.ErrnoException)?.code === "ENOENT")
      throw new Error(`Prepared training references were not found at ${directory}. Prepare runs/training-reference or set HOLOSPEX_TRAINING_REFERENCE_DIR to the prepared export folder.`);
    throw cause;
  }
  const imported = await importDatasetSamples(candidates.map(file => ({ name: file.name, size: file.bytes.length,
    arrayBuffer: async () => Uint8Array.from(file.bytes).buffer })));
  const training = trainingReferencesOnly(imported);
  if (!training.samples.length) throw new Error(training.issues.join(" "));
  const identities = new Set(training.samples.map(sample => sample.id));
  const referenced = new Set(training.samples.flatMap(sample => [sample.labels.filesSha256["image.jpg"].toLowerCase(), sample.labels.filesSha256["labels-index.png"].toLowerCase()]));
  return candidates.filter(file => {
    if (referenced.has(file.hash)) return true;
    try {
      const labels = parseDatasetSample(JSON.parse(file.bytes.toString("utf8")));
      return labels.annotationSource.split === "train" && identities.has(`${labels.annotationSource.videoId}_${labels.annotationSource.sourceFrameNumber}`);
    } catch { return false; }
  });
}

/** Opt-in, loopback-only Vite middleware; absent from production builds. */
export function localTrainingPlugin(directory: string): Plugin {
  const endpoint = "/__local-training/";
  let assets = new Map<string, Buffer>();
  return {
    name: "holospex-local-training", apply: "serve",
    configureServer(server) {
      server.middlewares.use((request, response, next) => {
        if (!request.url?.startsWith(endpoint)) { next(); return; }
        response.setHeader("Cache-Control", "no-store");
        response.setHeader("X-Content-Type-Options", "nosniff");
        const address = request.socket.remoteAddress;
        const origin = request.headers.origin;
        if (!["127.0.0.1", "::1", "::ffff:127.0.0.1"].includes(address ?? "")
          || (origin && origin !== `http://${request.headers.host}` && origin !== `https://${request.headers.host}`)) {
          response.statusCode = 403; response.end("Local training access only."); return;
        }
        if (request.method !== "GET") { response.statusCode = 405; response.end(); return; }
        void (async () => {
          if (request.url === `${endpoint}manifest`) {
            assets = new Map();
            const files = await readLocalTrainingFiles(directory);
            assets = new Map(files.map(file => [file.hash, file.bytes]));
            response.setHeader("Content-Type", "application/json");
            response.end(JSON.stringify({ source: "training_reference", split: "train",
              files: files.map(file => ({ name: file.name, url: `${endpoint}${file.hash}` })) }));
          } else {
            const hash = request.url!.slice(endpoint.length);
            const bytes = /^[a-f0-9]{64}$/.test(hash) ? assets.get(hash) : undefined;
            if (!bytes) { response.statusCode = 404; response.end(); return; }
            response.setHeader("Content-Type", "application/octet-stream"); response.end(bytes);
          }
        })().catch(() => {
          response.statusCode = 503; response.setHeader("Content-Type", "application/json");
          response.end(JSON.stringify({ error: "Prepared training references are unavailable. Prepare the train-split export in runs/training-reference, or set HOLOSPEX_TRAINING_REFERENCE_DIR to its folder. No test samples are loaded automatically." }));
        });
      });
    },
  };
}
