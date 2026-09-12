import { parseLearnerAttempt, type LearnerAttempt } from "@holospex/contracts";

const storageKey = "holospex.demo.attempts.v1";

/** Browser-local demo persistence. Add a consented backend here if needed later.
 * No camera pixels, names, patient details, or video are stored by this module.
 */
export function readAttempts(): LearnerAttempt[] {
  const raw = localStorage.getItem(storageKey);
  if (!raw) return [];
  const values: unknown = JSON.parse(raw);
  if (!Array.isArray(values)) throw new Error("Saved attempts have an invalid format.");
  return values.map(parseLearnerAttempt);
}

export function saveAttempt(attempt: LearnerAttempt) {
  const valid = parseLearnerAttempt(attempt);
  localStorage.setItem(storageKey, JSON.stringify([...readAttempts(), valid]));
}

export function downloadAttempts() {
  const contents = JSON.stringify(readAttempts(), null, 2);
  const url = URL.createObjectURL(new Blob([contents], { type: "application/json" }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = "holospex-demo-attempts.json";
  anchor.click();
  URL.revokeObjectURL(url);
}
