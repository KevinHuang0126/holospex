import { useRef, useState } from "react";
import type { LearnerAttempt, Lesson } from "@holospex/contracts";
import { saveAttempt } from "../analytics/attemptStore";

function createId() {
  if (typeof crypto.randomUUID === "function") return crypto.randomUUID();
  // LAN HTTP previews lack randomUUID on some browsers; getRandomValues still
  // permits local anonymous lesson IDs. Camera access continues to need HTTPS.
  return Array.from(crypto.getRandomValues(new Uint8Array(16)), (value) => value.toString(16).padStart(2, "0")).join("");
}

/** Teaching logic is independent of the camera/display and perception adapters.
 * Score only against lesson-authored answers; never infer correctness from an
 * ML confidence value. A recorded attempt is evidence of an interaction, not
 * a validated learning outcome.
 */
export function useLessonSession(lesson: Lesson) {
  const steps = lesson.checkpoints.flatMap((checkpoint) =>
    checkpoint.questions.map((question) => ({ checkpoint, question })),
  );
  const [stepIndex, setStepIndex] = useState(0);
  const [selectedChoiceId, setSelectedChoiceId] = useState("");
  const [submitted, setSubmitted] = useState(false);
  const [overlaysVisible, setOverlaysVisible] = useState(false);
  const [storageError, setStorageError] = useState<string | null>(null);
  const [readyCheckpointId, setReadyCheckpointId] = useState<string | null>(null);
  const [sessionId] = useState(createId);
  const startedAt = useRef<number | null>(null);
  const hintWasSeen = useRef(false);
  const step = steps[stepIndex];
  const imageReady = !!step && readyCheckpointId === step.checkpoint.id;

  function markImageReady() {
    if (!step) return;
    setReadyCheckpointId(step.checkpoint.id);
    // Do not include network/image-loading time in a learner's response time.
    if (startedAt.current === null) startedAt.current = performance.now();
  }

  function toggleOverlays() {
    if (!overlaysVisible) hintWasSeen.current = true;
    setOverlaysVisible(!overlaysVisible);
  }

  function submit() {
    if (!step || !selectedChoiceId || submitted || !imageReady || overlaysVisible || startedAt.current === null) return;
    const attempt: LearnerAttempt = {
      schemaVersion: "1.0.0",
      attemptId: createId(),
      sessionId,
      lessonId: lesson.id,
      checkpointId: step.checkpoint.id,
      questionId: step.question.id,
      selectedChoiceId,
      responseTimeMs: Math.round(performance.now() - startedAt.current),
      // Sticky exposure: hiding a hint again does not make an unaided attempt.
      hintsVisible: hintWasSeen.current,
      recordedAt: new Date().toISOString(),
    };
    try {
      saveAttempt(attempt);
      setStorageError(null);
    } catch {
      setStorageError("Your answer is shown below, but could not be saved in this browser.");
    }
    setSubmitted(true);
  }

  function next() {
    const nextIndex = (stepIndex + 1) % steps.length;
    setStepIndex(nextIndex);
    setSelectedChoiceId("");
    setSubmitted(false);
    setOverlaysVisible(false);
    setStorageError(null);
    hintWasSeen.current = false;
    // A second question on an already-loaded checkpoint needs no new image.
    startedAt.current = steps[nextIndex]?.checkpoint.id === readyCheckpointId ? performance.now() : null;
  }

  return {
    step, stepIndex, stepCount: steps.length, selectedChoiceId, setSelectedChoiceId,
    submitted, overlaysVisible, storageError, imageReady, markImageReady, toggleOverlays, submit, next,
  };
}
