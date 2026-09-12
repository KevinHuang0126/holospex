// Publish only this small, explicitly synthetic integration fixture. Never
// recursively copy a dataset directory into the browser's public directory.
import { copyFile, mkdir, readFile } from 'node:fs/promises';
import { parseFrameResult, parseLesson } from '../contracts/src/index';
const root = new URL('../', import.meta.url);
const source = new URL('assets/demo/', root);
const destination = new URL('apps/web/public/demo/', root);
const lesson = parseLesson(JSON.parse(await readFile(new URL('lesson.json', source), 'utf8')));
for (const checkpoint of lesson.checkpoints) {
  const name = checkpoint.frameResultPath.split('/').pop()!;
  const result = parseFrameResult(JSON.parse(await readFile(new URL(name, source), 'utf8')));
  if (result.mediaId !== lesson.media.id || result.frameNumber !== checkpoint.frameNumber ||
      result.timestampMs !== checkpoint.timestampMs || result.width !== lesson.media.width ||
      result.height !== lesson.media.height) throw new Error(`Mismatched checkpoint: ${checkpoint.id}`);
}
await mkdir(destination, { recursive: true });
for (const name of ['lesson.json', 'frame-000.json', 'frame-001.json', 'synthetic-frame.svg']) {
  await copyFile(new URL(name, source), new URL(name, destination));
}
console.log('Synthetic demo assets validated and prepared.');
