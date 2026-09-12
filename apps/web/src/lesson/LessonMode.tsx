import { useEffect, useState } from "react";
import type { FrameResult, Lesson } from "@holospex/contracts";
import { downloadAttempts } from "../analytics/attemptStore";
import { loadFrameResult } from "../data/loadDemo";
import { FrameOverlay } from "../overlays/FrameOverlay";
import { useLessonSession } from "./useLessonSession";

export function LessonMode({ lesson }: { lesson: Lesson }) {
  const session = useLessonSession(lesson);
  const [frame, setFrame] = useState<FrameResult | null>(null);
  const [frameError, setFrameError] = useState<string | null>(null);
  const [exportError, setExportError] = useState<string | null>(null);
  const [imageErrorCheckpoint, setImageErrorCheckpoint] = useState<string | null>(null);
  const resultPath = session.step?.checkpoint.frameResultPath;

  useEffect(() => {
    const controller = new AbortController();
    setFrame(null);
    setFrameError(null);
    if (resultPath) {
      loadFrameResult(resultPath, controller.signal).then(setFrame).catch((error: unknown) => {
        if (!controller.signal.aborted) setFrameError(error instanceof Error ? error.message : "Unable to load overlays.");
      });
    }
    return () => controller.abort();
  }, [resultPath]);

  if (!session.step) return <p role="alert">This lesson has no questions.</p>;
  const { checkpoint, question } = session.step;
  const correct = session.selectedChoiceId === question.reviewedAnswer.choiceId;
  const sourceLabel = frame?.source.replaceAll("_", " ") ?? "Loading overlay data";

  function exportAnswers() {
    try {
      downloadAttempts();
      setExportError(null);
    } catch {
      setExportError("Saved answers could not be read from this browser.");
    }
  }

  return (
    <>
      <section className="workspace" aria-labelledby="lesson-title">
        <div className="visual-panel">
          <div className="stage">
            <FrameOverlay key={checkpoint.id} lesson={lesson} checkpoint={checkpoint} result={frame} visible={session.overlaysVisible} onImageLoad={session.markImageReady} onImageError={() => setImageErrorCheckpoint(checkpoint.id)} />
            <span className="stage-badge">Synthetic illustration · {sourceLabel}</span>
          </div>
          <div className="stage-controls">
            <span>Checkpoint {session.stepIndex + 1} / {session.stepCount}</span>
            <button className="secondary" onClick={session.toggleOverlays} disabled={!frame || frame.status !== "ok" || !session.imageReady} aria-pressed={session.overlaysVisible}>{session.overlaysVisible ? "Hide labels" : "Reveal labels"}</button>
          </div>
          {frameError && <p className="error" role="status">{frameError} You can still explore the lesson.</p>}
          {frame && frame.status !== "ok" && <p className="notice" role="status">Overlay {frame.status}: {frame.statusReason}</p>}
          {imageErrorCheckpoint === checkpoint.id && <p className="error" role="alert">The checkpoint image could not load. Answer submission is unavailable.</p>}
          <p className="fine-print">{lesson.disclaimer}</p>
        </div>
        <aside className="lesson-panel">
          <p className="eyebrow">Learn · Observe · Recall</p>
          <h2 id="lesson-title">{lesson.title}</h2>
          <p>{lesson.description}</p>
          <form onSubmit={(event) => { event.preventDefault(); session.submit(); }}>
            <fieldset disabled={session.submitted || !session.imageReady}>
              <legend>{question.prompt}</legend>
              {question.choices.map((choice) => <label className={`choice ${session.selectedChoiceId === choice.id ? "selected" : ""}`} key={choice.id}>
                <input type="radio" name="answer" value={choice.id} checked={session.selectedChoiceId === choice.id} onChange={() => session.setSelectedChoiceId(choice.id)} />
                <span>{choice.label}</span>
              </label>)}
            </fieldset>
            {!session.submitted && <>
              {!session.imageReady && imageErrorCheckpoint !== checkpoint.id && <p className="fine-print" role="status">Loading checkpoint image…</p>}
              {session.overlaysVisible && <p className="fine-print">Hide labels before checking your answer. Hint use remains recorded.</p>}
              <button type="submit" disabled={!session.selectedChoiceId || !session.imageReady || session.overlaysVisible}>Check answer</button>
            </>}
          </form>
          {session.submitted && <div className="feedback" role="status">
            <strong>{correct ? "Matches the demo answer" : "Compare with the demo answer"}</strong>
            <p>{question.reviewedAnswer.explanation}</p>
            <p className="fine-print">Answer source: {question.reviewedAnswer.source.replaceAll("_", " ")}.</p>
            <button onClick={session.next}>{session.stepIndex + 1 === session.stepCount ? "Try again" : "Next question"}</button>
          </div>}
          {session.storageError && <p className="error" role="status">{session.storageError}</p>}
          <p className="fine-print">Answers stay in this browser. Revealing labels is recorded as hint use.</p>
        </aside>
      </section>
      <div className="export-row"><button className="text-button" onClick={exportAnswers}>Export saved answers ↓</button>{exportError && <span className="error" role="status">{exportError}</span>}</div>
    </>
  );
}
