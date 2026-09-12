/* Generated from contracts/schemas. Run npm run contracts:generate; do not edit. */

/**
 * An immutable first submission. Do not store ML output as the reviewed answer.
 */
export interface LearnerAttempt {
  schemaVersion: '1.0.0';
  attemptId: string;
  sessionId: string;
  lessonId: string;
  checkpointId: string;
  questionId: string;
  selectedChoiceId: string;
  responseTimeMs: number;
  hintsVisible: boolean;
  recordedAt: string;
}
