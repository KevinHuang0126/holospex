import Ajv from 'ajv';
import lessonSchema from '../schemas/lesson.schema.json';
import frameSchema from '../schemas/frame-result.schema.json';
import attemptSchema from '../schemas/learner-attempt.schema.json';
import anatomyCatalog from '../anatomy.json';
import type { Lesson } from './generated/lesson';
import type { FrameResult } from './generated/frame-result';
import type { LearnerAttempt } from './generated/learner-attempt';

export type { Lesson, FrameResult, LearnerAttempt };
export const anatomy = anatomyCatalog;
export type AnatomyId = keyof typeof anatomy;

// Conditional schemas refine fields declared at the parent object, so permit
// partial type declarations there while retaining other strict schema checks.
const ajv = new Ajv({ allErrors: true, strict: true, strictTypes: false, strictRequired: false });
const validateLesson = ajv.compile<Lesson>(lessonSchema);
const validateFrame = ajv.compile<FrameResult>(frameSchema);
const validateAttempt = ajv.compile<LearnerAttempt>(attemptSchema);

function unique(values: string[], label: string) {
  if (new Set(values).size !== values.length) throw new Error(`Duplicate ${label}.`);
}

export function parseLesson(value: unknown): Lesson {
  if (!validateLesson(value)) throw new Error(`Invalid lesson: ${ajv.errorsText(validateLesson.errors)}`);
  unique(value.checkpoints.map((item) => item.id), 'checkpoint ID');
  if (value.media.kind === 'video' && !value.media.src) throw new Error('Video lesson requires media.src.');
  for (const checkpoint of value.checkpoints) {
    unique(checkpoint.questions.map((item) => item.id), 'question ID');
    for (const question of checkpoint.questions) {
      unique(question.choices.map((item) => item.id), 'choice ID');
      if (!question.choices.some((choice) => choice.id === question.reviewedAnswer.choiceId)) {
        throw new Error(`Reviewed answer is not a choice for ${question.id}.`);
      }
    }
  }
  return value;
}

export function parseFrameResult(value: unknown): FrameResult {
  if (!validateFrame(value)) throw new Error(`Invalid frame: ${ajv.errorsText(validateFrame.errors)}`);
  unique(value.structures.map((item) => item.instanceId), 'instance ID');
  for (const structure of value.structures) {
    for (const [x, y] of structure.polygon) {
      // Continuous image-edge coordinates include the right/bottom boundary.
      if (x > value.width || y > value.height) throw new Error('Polygon lies outside original frame.');
    }
  }
  if (value.source === 'propagated_prediction' &&
      !(value.propagatedFromTimestampMs! < value.timestampMs)) {
    throw new Error('Propagation must refer to an earlier frame.');
  }
  return value;
}

export function parseLearnerAttempt(value: unknown): LearnerAttempt {
  if (!validateAttempt(value)) throw new Error(`Invalid learner attempt: ${ajv.errorsText(validateAttempt.errors)}`);
  return value;
}
