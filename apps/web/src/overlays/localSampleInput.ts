import { importDatasetSamples } from "./datasetSamples";

/** The dev server supplies bytes; the browser still validates their hashes and envelope. */
export async function loadLocalDatasetSamples(signal: AbortSignal) {
  const response = await fetch("/__local-samples/manifest", { signal, cache: "no-store" });
  if (!response.ok || !response.headers.get("content-type")?.includes("application/json"))
    throw new Error("Automatic sample loading is unavailable. Run npm run dev:samples, or choose the folder below.");
  const manifest: unknown = await response.json();
  const files = (manifest as { files?: unknown })?.files;
  if (!Array.isArray(files) || files.length > 200 || files.some(file => !file || typeof file.name !== "string"
    || typeof file.url !== "string" || !/^\/__local-samples\/[a-f0-9]{64}$/.test(file.url)))
    throw new Error("The local sample manifest is invalid.");
  const inputs = await Promise.all(files.map(async file => {
    const asset = await fetch(file.url, { signal, cache: "no-store" });
    if (!asset.ok) throw new Error("A local sample changed during loading. Reload samples to try again.");
    return new File([await asset.blob()], file.name);
  }));
  signal.throwIfAborted();
  const pack = await importDatasetSamples(inputs);
  signal.throwIfAborted();
  return pack;
}
