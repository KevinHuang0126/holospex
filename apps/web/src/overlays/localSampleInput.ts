import { importDatasetSamples } from "./datasetSamples";
import { trainingReferencesOnly } from "./trainingReferences";
export { trainingReferencesOnly } from "./trainingReferences";

/** Only prepared training references load automatically; hashes and splits are rechecked here. */
export async function loadLocalDatasetSamples(signal: AbortSignal) {
  const response = await fetch("/__local-training/manifest", { signal, cache: "no-store" });
  if (!response.ok || !response.headers.get("content-type")?.includes("application/json"))
    throw new Error("Automatic training references are unavailable. Prepare runs/training-reference and run npm run dev:samples, or choose the folder below. Test samples are not loaded automatically.");
  const manifest: unknown = await response.json();
  const value = manifest as { source?: unknown; split?: unknown; files?: unknown };
  const files = value?.files;
  if (value?.source !== "training_reference" || value?.split !== "train" || !Array.isArray(files) || files.length > 200 || files.some(file => !file || typeof file.name !== "string"
    || typeof file.url !== "string" || !/^\/__local-training\/[a-f0-9]{64}$/.test(file.url)))
    throw new Error("The local training manifest is invalid.");
  const inputs = await Promise.all(files.map(async file => {
    const asset = await fetch(file.url, { signal, cache: "no-store" });
    if (!asset.ok) throw new Error("A training reference changed during loading. Reload training references to try again.");
    return new File([await asset.blob()], file.name);
  }));
  signal.throwIfAborted();
  const pack = trainingReferencesOnly(await importDatasetSamples(inputs));
  signal.throwIfAborted();
  return pack;
}
