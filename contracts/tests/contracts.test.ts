import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { anatomy, parseFrameResult, parseLearnerAttempt, parseLesson } from '../src/index';
import frameSchema from '../schemas/frame-result.schema.json';

const fixture = (name: string): any => JSON.parse(readFileSync(new URL(`../../assets/demo/${name}.json`, import.meta.url), 'utf8'));

test('anatomy catalog and wire-format class IDs stay aligned', () => {
  assert.deepEqual(Object.keys(anatomy).sort(), [...frameSchema.properties.structures.items.properties.structureId.enum].sort());
});

test('shared fixtures form matching lesson checkpoints', () => {
  const lesson = parseLesson(fixture('lesson'));
  for (const checkpoint of lesson.checkpoints) {
    const frame = parseFrameResult(fixture(checkpoint.frameResultPath.split('/').pop()!.replace('.json', '')));
    assert.equal(frame.mediaId, lesson.media.id);
    assert.equal(frame.frameNumber, checkpoint.frameNumber);
    assert.equal(frame.timestampMs, checkpoint.timestampMs);
    assert.equal(frame.width, lesson.media.width);
    assert.equal(frame.height, lesson.media.height);
  }
});

test('unavailable frames cannot carry geometry', () => {
  const frame = fixture('frame-000');
  frame.status = 'unsupported';
  frame.statusReason = 'No model available';
  assert.throws(() => parseFrameResult(frame));
});

test('successful empty detections remain distinct from unavailable processing', () => {
  const frame = fixture('frame-000');
  frame.structures = [];
  assert.equal(parseFrameResult(frame).status, 'ok');
  frame.status = 'missing';
  assert.throws(() => parseFrameResult(frame));
});

test('predictions require model identity and per-structure confidence', () => {
  const frame = fixture('frame-000');
  frame.source = 'ml_prediction';
  assert.throws(() => parseFrameResult(frame));
  frame.model = { id: 'test-model', version: '1' };
  assert.throws(() => parseFrameResult(frame));
  frame.structures[0].confidence = 0.7;
  assert.equal(parseFrameResult(frame).source, 'ml_prediction');
});

test('pixel bounds, finite numbers, duplicate instances and propagation are enforced', () => {
  const outside = fixture('frame-000');
  outside.structures[0].polygon[0][0] = outside.width + 1;
  assert.throws(() => parseFrameResult(outside));
  const invalid = fixture('frame-000');
  invalid.timestampMs = Infinity;
  assert.throws(() => parseFrameResult(invalid));
  const duplicate = fixture('frame-000');
  duplicate.structures.push(structuredClone(duplicate.structures[0]));
  assert.throws(() => parseFrameResult(duplicate));
  const propagated = fixture('frame-001');
  propagated.source = 'propagated_prediction';
  propagated.model = { id: 'test-model', version: '1' };
  propagated.propagatedFromTimestampMs = propagated.timestampMs;
  assert.throws(() => parseFrameResult(propagated));
  propagated.propagatedFromTimestampMs = 0;
  assert.equal(parseFrameResult(propagated).source, 'propagated_prediction');
});

test('lesson answer must be an actual choice and expert review needs provenance', () => {
  const lesson = fixture('lesson');
  lesson.checkpoints[0].questions[0].reviewedAnswer.choiceId = 'nonexistent';
  assert.throws(() => parseLesson(lesson));
  const reviewed = fixture('lesson');
  reviewed.checkpoints[0].questions[0].reviewedAnswer.source = 'expert_reviewed';
  assert.throws(() => parseLesson(reviewed));
});

test('attempt schema rejects negative response times and invalid dates', () => {
  const attempt = {
    schemaVersion: '1.0.0', attemptId: 'a', sessionId: 's', lessonId: 'l',
    checkpointId: 'c', questionId: 'q', selectedChoiceId: 'cannot_determine',
    responseTimeMs: 250, hintsVisible: true, recordedAt: '2026-09-12T20:00:00.000Z',
  };
  assert.equal(parseLearnerAttempt(attempt).selectedChoiceId, 'cannot_determine');
  assert.throws(() => parseLearnerAttempt({ ...attempt, responseTimeMs: -1 }));
  assert.throws(() => parseLearnerAttempt({ ...attempt, recordedAt: 'yesterday' }));
});
