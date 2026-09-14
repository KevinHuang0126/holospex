import { createHash } from "node:crypto";
import { readFile, readdir, realpath, stat } from "node:fs/promises";
import { isAbsolute, join, relative } from "node:path";

export interface SampleFile { name: string; bytes: Buffer; hash: string }

/** Only label/index JSON and images referenced by label hashes leave the local folder. */
export async function readLocalSampleFiles(directory: string): Promise<SampleFile[]> {
  const root = await realpath(directory), candidates: SampleFile[] = [];
  let totalBytes = 0, totalFiles = 0;
  async function walk(folder: string) {
    for (const name of await readdir(folder)) {
      const path = await realpath(join(folder, name)), inside = relative(root, path);
      if (inside.startsWith("..") || isAbsolute(inside)) continue;
      const info = await stat(path);
      if (info.isDirectory()) { if (inside.split(/[\\/]/).length < 4) await walk(path); continue; }
      if (!info.isFile()) continue;
      totalBytes += info.size; totalFiles += 1;
      if (totalFiles > 200 || totalBytes > 128 * 1024 * 1024) throw new Error("Sample folder exceeds 200 files / 128 MB.");
      const bytes = await readFile(path);
      candidates.push({ name, bytes, hash: createHash("sha256").update(bytes).digest("hex") });
    }
  }
  await walk(root);
  const selected = new Map<string, SampleFile>(), referenced = new Set<string>();
  for (const file of candidates) {
    if (!file.bytes.subarray(0, 4).toString("utf8").trimStart().startsWith("{")) continue;
    try {
      const value = JSON.parse(file.bytes.toString("utf8"));
      if (value?.artifactType === "dataset_annotation_sample") {
        selected.set(file.hash, file);
        for (const key of ["image.jpg", "labels-index.png"]) {
          const hash = value.filesSha256?.[key];
          if (typeof hash === "string" && /^[a-f0-9]{64}$/i.test(hash)) referenced.add(hash.toLowerCase());
        }
      } else if (Array.isArray(value?.samples)) selected.set(file.hash, file);
    } catch { /* Invalid JSON cannot introduce a file reference. */ }
  }
  for (const file of candidates) if (referenced.has(file.hash)) selected.set(file.hash, file);
  return [...selected.values()];
}
