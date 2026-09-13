// Publish only the named fixtures and explicitly selected mannequin reference. Never
// recursively copy a dataset directory into the browser's public directory.
import { createHash } from 'node:crypto';
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
const mannequin = await readFile(new URL('mannequin.png', source));
if (mannequin.length < 33 || !mannequin.subarray(0, 8).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10])) ||
    mannequin.toString('ascii', 12, 16) !== 'IHDR' || mannequin.readUInt32BE(16) !== 894 || mannequin.readUInt32BE(20) !== 569 ||
    createHash('sha256').update(mannequin).digest('hex') !== 'cbeb22f2c521d8ecad5be3ed9061bd3d707c065c01b02dbf5eb426f38dbf4c51')
  throw new Error('Mannequin reference differs from the selected 894 × 569 PNG. Review its provenance before replacing it.');
const cutout = await readFile(new URL('mannequin-cutout.png', source));
if (cutout.length < 33 || !cutout.subarray(0, 8).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10])) ||
    cutout.toString('ascii', 12, 16) !== 'IHDR' || cutout.readUInt32BE(16) !== 894 || cutout.readUInt32BE(20) !== 569 ||
    cutout[24] !== 8 || cutout[25] !== 6 ||
    createHash('sha256').update(cutout).digest('hex') !== 'c25ddd6715d6402786780008588763b56d9dc970dc438bf3256447d7347f12c1')
  throw new Error('Mannequin cutout differs from the approved transparent PNG. Review the matte and provenance before replacing it.');
await mkdir(destination, { recursive: true });
for (const name of ['lesson.json', 'frame-000.json', 'frame-001.json', 'synthetic-frame.svg', 'mannequin-cutout.png']) {
  await copyFile(new URL(name, source), new URL(name, destination));
}
console.log('Demo fixtures and the selected mannequin reference validated and prepared.');
