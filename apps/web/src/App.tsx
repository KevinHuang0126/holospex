import { useEffect, useState } from "react";
import type { Lesson } from "@holospex/contracts";
import { CameraMode } from "./camera/CameraMode";
import { loadLesson } from "./data/loadDemo";
import { LessonMode } from "./lesson/LessonMode";

/** Composition root: modes share the lesson vocabulary, not tracking logic.
 * Keep device APIs and ML transport out of the teaching components.
 */
export function App() {
  const [mode, setMode] = useState<"lesson" | "camera">("lesson");
  const [lesson, setLesson] = useState<Lesson | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    loadLesson(controller.signal).then(setLesson).catch((cause: unknown) => {
      if (!controller.signal.aborted) setError(cause instanceof Error ? cause.message : "The demo lesson could not load.");
    });
    return () => controller.abort();
  }, []);

  return (
    <main>
      <header className="topbar"><a className="wordmark" href="/" aria-label="Holospex home"><span className="brand-icon">H</span>Holospex</a><span className="prototype-tag">Training prototype</span></header>
      <section className="intro"><p className="eyebrow">A new perspective on surgical learning</p><h1>See the anatomy.<br /><span>Build your understanding.</span></h1><p>Explore the learning flow today, with a path toward an augmented reality training experience.</p></section>
      <nav className="mode-switch" aria-label="Demo mode"><button className={mode === "lesson" ? "active" : ""} aria-pressed={mode === "lesson"} onClick={() => setMode("lesson")}>Video lesson scaffold</button><button className={mode === "camera" ? "active" : ""} aria-pressed={mode === "camera"} onClick={() => setMode("camera")}>Camera prototype</button></nav>
      {mode === "camera" ? <CameraMode /> : error ? <div className="notice error" role="alert">{error}</div> : lesson ? <LessonMode lesson={lesson} /> : <div className="notice" role="status">Loading the demo lesson…</div>}
      <footer>Hackathon scaffold · Synthetic content · Educational exploration only</footer>
    </main>
  );
}
