import { parseFrameResult, parseLesson } from "@holospex/contracts";

async function loadJson(path: string, signal?: AbortSignal): Promise<unknown> {
  const response = await fetch(path, { signal });
  if (!response.ok) throw new Error(`Demo asset could not load (HTTP ${response.status}).`);
  return response.json();
}

export async function loadLesson(signal?: AbortSignal) {
  return parseLesson(await loadJson("/demo/lesson.json", signal));
}

export async function loadFrameResult(path: string, signal?: AbortSignal) {
  return parseFrameResult(await loadJson(path, signal));
}
