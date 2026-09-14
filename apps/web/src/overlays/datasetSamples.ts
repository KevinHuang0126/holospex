import Ajv from "ajv";
import type { AnatomyId, FrameResult } from "@holospex/contracts";
// Keep this shared browser/dev-server importer independent of the contract runtime.
import anatomy from "../../../../contracts/anatomy.json";
import frameSchema from "../../../../contracts/schemas/frame-result.schema.json";
import type { DisplayedFrame } from "./selectFrame";

/** Person 1's sample envelope is deliberately separate from FrameResult. */
export interface DatasetSample {
  artifactType: "dataset_annotation_sample";
  annotationSource: {
    dataset: "Endoscapes-Seg50"; kind: "supplied_dataset_annotation";
    split: string; sourceUrl: string; videoId: number; sourceFrameNumber: number;
  };
  frame: DisplayedFrame & { coordinateSpace: "original_pixels" };
  classes: { index: number; sourceId: number; structureId: AnatomyId | "background" }[];
  raster: {
    dtype: "uint8"; shape: [number, number]; ignoreValue: 255;
    pixelCounts: Record<AnatomyId, number>; ignoredPixelCount: number;
  };
  structures: FrameResult["structures"];
  conversion: { geometry: string; visibility: string; withheldComponents: Record<string, number> };
  filesSha256: Record<string, string>;
}
export interface LoadedDatasetSample {
  id: string;
  labels: DatasetSample;
  image: Blob;
  indexMask: Blob;
}
export interface DatasetImport {
  samples: LoadedDatasetSample[];
  issues: string[];
}
export const datasetSourceLabel = "Supplied dataset annotation";
export const datasetCredit = {
  label: "Endoscapes-Seg50 · CAMMA / Endoscapes authors",
  sourceUrl: "https://github.com/CAMMA-public/Endoscapes",
  license: "CC BY-NC-SA 4.0",
  licenseUrl: "https://creativecommons.org/licenses/by-nc-sa/4.0/",
};
const integer = { type: "integer", minimum: 0 };
const text = { type: "string", minLength: 1 };
const ids = Object.keys(anatomy) as AnatomyId[];
const validate = new Ajv({ allErrors: true, strict: false }).compile<DatasetSample>({
  type: "object", required: ["artifactType", "annotationSource", "frame", "classes", "raster", "structures", "conversion", "filesSha256"],
  properties: {
    artifactType: { const: "dataset_annotation_sample" },
    annotationSource: {
      type: "object", required: ["dataset", "kind", "split", "sourceUrl", "videoId", "sourceFrameNumber"],
      properties: { dataset: { const: "Endoscapes-Seg50" }, kind: { const: "supplied_dataset_annotation" },
        split: text, sourceUrl: { const: datasetCredit.sourceUrl }, videoId: integer, sourceFrameNumber: integer },
    },
    frame: {
      type: "object", required: ["mediaId", "frameNumber", "timestampMs", "width", "height", "coordinateSpace"],
      properties: { mediaId: text, frameNumber: { const: 0 }, timestampMs: { const: 0 },
        width: { type: "integer", minimum: 1, maximum: 8192 }, height: { type: "integer", minimum: 1, maximum: 8192 },
        coordinateSpace: { const: "original_pixels" } },
    },
    classes: {
      type: "array", minItems: 7, maxItems: 7, items: {
        type: "object", required: ["index", "sourceId", "structureId"],
        properties: { index: integer, sourceId: integer, structureId: { enum: ["background", ...ids] } },
      },
    },
    raster: {
      type: "object", required: ["dtype", "shape", "ignoreValue", "pixelCounts", "ignoredPixelCount"],
      properties: { dtype: { const: "uint8" }, shape: { type: "array", minItems: 2, maxItems: 2, items: integer },
        ignoreValue: { const: 255 }, ignoredPixelCount: integer,
        pixelCounts: { type: "object", required: ids, additionalProperties: false,
          properties: Object.fromEntries(ids.map(id => [id, integer])) } },
    },
    // Reuse the canonical geometry shape without manufacturing FrameResult provenance.
    structures: frameSchema.properties.structures,
    conversion: { type: "object", required: ["geometry", "visibility", "withheldComponents"],
      properties: { geometry: text, visibility: text,
        withheldComponents: { type: "object", additionalProperties: integer } } },
    filesSha256: { type: "object", required: ["image.jpg", "labels-index.png"],
      additionalProperties: { type: "string", pattern: "^[a-fA-F0-9]{64}$" } },
  },
});

export function parseDatasetSample(value: unknown): DatasetSample {
  if (!validate(value)) throw new Error("Invalid dataset sample: " + validate.errors?.map(error => `${error.instancePath} ${error.message}`).join("; "));
  const { frame, annotationSource: source, classes, raster } = value;
  if (frame.mediaId !== `endoscapes-still-${source.videoId}_${source.sourceFrameNumber}`)
    throw new Error("Sample media identity contradicts its source case/frame.");
  if (frame.width * frame.height > 16_000_000 || raster.shape[0] !== frame.height || raster.shape[1] !== frame.width)
    throw new Error("Mask dimensions do not match the original still.");
  const expected = ["background", "gallbladder", "cystic_duct", "cystic_artery", "cystic_plate", "hepatocystic_triangle_dissection", "tool"];
  const sourceIds = [0, 5, 4, 3, 1, 2, 6];
  if (new Set(classes.map(item => item.index)).size !== 7 || classes.some(item => expected[item.index] !== item.structureId || sourceIds[item.index] !== item.sourceId))
    throw new Error("Unsupported Endoscapes class mapping.");
  if (Object.values(raster.pixelCounts).reduce((sum, count) => sum + count, raster.ignoredPixelCount) > frame.width * frame.height)
    throw new Error("Annotation pixel counts exceed the image area.");
  if (new Set(value.structures.map(item => item.instanceId)).size !== value.structures.length)
    throw new Error("Duplicate sample structure instance.");
  for (const item of value.structures) {
    if (item.confidence !== undefined) throw new Error("Supplied dataset annotations must not invent model confidence.");
    if (item.polygon.some(([x, y]) => x > frame.width || y > frame.height)) throw new Error("Sample polygon is outside the original image.");
  }
  return value;
}

export async function sha256(bytes: ArrayBuffer): Promise<string> {
  return Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)), byte => byte.toString(16).padStart(2, "0")).join("");
}
function imageType(bytes: Uint8Array): "image/png" | "image/jpeg" | null {
  if ([137, 80, 78, 71, 13, 10, 26, 10].every((byte, index) => bytes[index] === byte)) return "image/png";
  if (bytes[0] === 255 && bytes[1] === 216 && bytes[2] === 255) return "image/jpeg";
  return null;
}

/** Reads user-selected files locally. File content and supplied hashes establish the pairing. */
export async function importDatasetSamples(files: readonly Pick<File, "name" | "arrayBuffer" | "size">[]): Promise<DatasetImport> {
  if (files.length > 200 || files.reduce((sum, file) => sum + file.size, 0) > 128 * 1024 * 1024)
    throw new Error("Choose the small sample pack (up to 200 files / 128 MB), rather than the full dataset.");
  const assets = new Map<string, { bytes: ArrayBuffer; type: string | null }>();
  const records: { name: string; value: unknown }[] = [];
  const expected = new Set<string>();
  const issues: string[] = [];
  const seenRecords = new Set<string>();
  for (const file of files) {
    const bytes = await file.arrayBuffer(), hash = await sha256(bytes), type = imageType(new Uint8Array(bytes));
    assets.set(hash, { bytes, type });
    if (type) continue;
    const contents = new TextDecoder().decode(bytes).trim();
    if (!contents.startsWith("{")) continue;
    try {
      const value = JSON.parse(contents);
      if (value?.artifactType === "dataset_annotation_sample" && !seenRecords.has(hash)) {
        records.push({ name: file.name, value }); seenRecords.add(hash);
      }
      if (Array.isArray(value?.samples)) for (const item of value.samples) {
        if (typeof item?.id === "string" && /^\d+_\d+$/.test(item.id)) expected.add(item.id);
      }
    } catch { issues.push(`${file.name}: malformed JSON; skipped.`); }
  }
  const samples: LoadedDatasetSample[] = [];
  const identities = new Set<string>();
  const duplicateIds = new Set<string>();
  for (const record of records) {
    try {
      const labels = parseDatasetSample(record.value);
      const id = `${labels.annotationSource.videoId}_${labels.annotationSource.sourceFrameNumber}`;
      if (identities.has(id)) { duplicateIds.add(id); throw new Error(`Conflicting label records for ${id}. Case withheld.`); }
      identities.add(id);
      const image = assets.get(labels.filesSha256["image.jpg"].toLowerCase());
      const mask = assets.get(labels.filesSha256["labels-index.png"].toLowerCase());
      if (!image || image.type !== "image/jpeg") throw new Error(`${id}: original JPEG missing or hash mismatch.`);
      if (!mask || mask.type !== "image/png") throw new Error(`${id}: exact index mask missing or hash mismatch.`);
      const header = new DataView(mask.bytes);
      if (header.byteLength < 33 || header.getUint32(16) !== labels.frame.width || header.getUint32(20) !== labels.frame.height
        || header.getUint8(24) !== 8 || header.getUint8(25) !== 0)
        throw new Error(`${id}: expected a full-resolution, 8-bit grayscale index PNG.`);
      samples.push({ id, labels, image: new Blob([image.bytes], { type: "image/jpeg" }), indexMask: new Blob([mask.bytes], { type: "image/png" }) });
    } catch (error) { issues.push(`${record.name}: ${error instanceof Error ? error.message : "Invalid sample."}`); }
  }
  const usable = samples.filter(item => !duplicateIds.has(item.id)).sort((a, b) => a.id.localeCompare(b.id, undefined, { numeric: true }));
  for (const id of expected) if (!usable.some(item => item.id === id)) issues.push(`Case ${id}: no complete, valid label/image/mask set. Unable to assess.`);
  if (!usable.length) issues.push("No usable samples. Select the entire sample folder, including label records, original JPEGs and index masks.");
  return { samples: usable, issues };
}
