import { datasetCredit, importDatasetSamples, type DatasetImport, type LoadedDatasetSample } from "./datasetSamples";

/** A local transfer file, separate from the ML result and annotation contracts. */
export const PORTABLE_SAMPLE_MAX_BYTES = 20 * 1024 * 1024;
const sizeError = () => new Error("Choose a camera sample file smaller than 20 MB.");

function encodeBase64(bytes: ArrayBuffer): string {
  const source = new Uint8Array(bytes);
  const parts: string[] = [];
  for (let offset = 0; offset < source.length; offset += 0x8000)
    parts.push(String.fromCharCode(...source.subarray(offset, offset + 0x8000)));
  return btoa(parts.join(""));
}

function decodeBase64(value: unknown, name: string): ArrayBuffer {
  if (typeof value !== "string" || !value.length || value.length > PORTABLE_SAMPLE_MAX_BYTES
    || value.length % 4 !== 0 || /[^A-Za-z0-9+/=]/.test(value))
    throw new Error(`Camera sample ${name} must contain valid base64 bytes.`);
  let binary: string;
  try { binary = atob(value); }
  catch { throw new Error(`Camera sample ${name} must contain valid base64 bytes.`); }
  // Reject noncanonical padding bits as well as URLs, whitespace and data URI prefixes.
  if (btoa(binary) !== value) throw new Error(`Camera sample ${name} has invalid base64 padding.`);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index++) bytes[index] = binary.charCodeAt(index);
  return bytes.buffer;
}

function validateParts(labels: unknown, image: ArrayBuffer, mask: ArrayBuffer): Promise<DatasetImport> {
  const labelBytes = new TextEncoder().encode(JSON.stringify(labels)).buffer;
  return importDatasetSamples([
    { name: "labels.json", size: labelBytes.byteLength, arrayBuffer: async () => labelBytes },
    { name: "image.jpg", size: image.byteLength, arrayBuffer: async () => image },
    { name: "labels-index.png", size: mask.byteLength, arrayBuffer: async () => mask },
  ]);
}

/** Packages only the selected annotation, original image and index mask for local phone transfer. */
export async function exportPortableSample(sample: LoadedDatasetSample): Promise<Blob> {
  if (sample.image.size + sample.indexMask.size > PORTABLE_SAMPLE_MAX_BYTES) throw sizeError();
  const [image, mask] = await Promise.all([sample.image.arrayBuffer(), sample.indexMask.arrayBuffer()]);
  const validated = await validateParts(sample.labels, image, mask);
  if (validated.samples.length !== 1 || validated.issues.length)
    throw new Error(validated.issues.join(" ") || "Unable to export an invalid camera sample.");
  const file = new Blob([JSON.stringify({
    artifactType: "holospex_camera_sample", version: 1, labels: sample.labels, attribution: datasetCredit,
    imageBase64: encodeBase64(image), indexMaskBase64: encodeBase64(mask),
  })], { type: "application/json" });
  if (file.size > PORTABLE_SAMPLE_MAX_BYTES) throw sizeError();
  return file;
}

/** No network access: the existing importer still checks provenance, hashes and mask headers. */
export async function importPortableSample(file: Pick<File, "size" | "text">): Promise<DatasetImport> {
  if (!Number.isFinite(file.size) || file.size < 0 || file.size > PORTABLE_SAMPLE_MAX_BYTES) throw sizeError();
  const contents = await file.text();
  if (contents.length > PORTABLE_SAMPLE_MAX_BYTES || new TextEncoder().encode(contents).byteLength > PORTABLE_SAMPLE_MAX_BYTES)
    throw sizeError();
  let envelope: unknown;
  try { envelope = JSON.parse(contents); }
  catch { throw new Error("Camera sample contains malformed JSON."); }
  if (!envelope || typeof envelope !== "object" || Array.isArray(envelope))
    throw new Error("Choose a Holospex camera sample file.");
  const value = envelope as Record<string, unknown>;
  if (value.artifactType !== "holospex_camera_sample" || value.version !== 1 || !value.labels
    || typeof value.labels !== "object" || Array.isArray(value.labels))
    throw new Error("Unsupported camera sample format or version.");
  return validateParts(value.labels, decodeBase64(value.imageBase64, "image"), decodeBase64(value.indexMaskBase64, "index mask"));
}
