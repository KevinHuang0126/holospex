import type { DatasetImport } from "./datasetSamples";

/** This gate never rewrites a source split or promotes a sample into training data. */
export function trainingReferencesOnly(pack: DatasetImport): DatasetImport {
  const excluded = pack.samples.filter(sample => sample.labels.annotationSource.split !== "train");
  const samples = pack.samples.filter(sample => sample.labels.annotationSource.split === "train");
  const issues = [...pack.issues, ...excluded.map(sample => `Case ${sample.id}: ${sample.labels.annotationSource.split} split excluded from automatic training references.`)];
  if (!samples.length) issues.push("No training references are available. Prepare exports from the train split; test samples are not used as a fallback.");
  return { samples, issues: [...new Set(issues)] };
}
