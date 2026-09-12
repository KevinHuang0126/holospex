import { useEffect, useRef } from "react";
import { useCamera } from "../input/useCamera";

/**
 * Camera-mode integration point. Replace the empty tracking state with a marker
 * adapter before drawing labels. World/marker poses are distinct from video
 * FrameResult pixel coordinates; never reuse one as the other.
 */
export function CameraMode() {
  const video = useRef<HTMLVideoElement>(null);
  const { stream, starting, error, start, stop } = useCamera();

  useEffect(() => {
    if (video.current) video.current.srcObject = stream;
  }, [stream]);

  return (
    <section className="workspace camera-workspace" aria-labelledby="camera-title">
      <div className="stage camera-stage">
        <video ref={video} autoPlay muted playsInline aria-label="Live camera preview" />
        {!stream && <div className="stage-placeholder"><span className="reticle">＋</span><h2>Bring the model into view</h2><p>Start your camera to preview the physical-model input.</p></div>}
        {stream && <span className="stage-badge">Camera preview · no tracking</span>}
      </div>
      <aside className="lesson-panel">
        <p className="eyebrow">Camera mode</p>
        <h2 id="camera-title">A window into the future headset experience.</h2>
        <p>This scaffold provides a camera preview. Marker tracking and anchored anatomy labels are the next integration step.</p>
        <div className="notice">No anatomy is detected or identified in this mode.</div>
        {error && <p role="alert" className="error">{error}</p>}
        <button onClick={stream ? stop : start} disabled={starting}>{starting ? "Starting camera…" : stream ? "Stop camera" : "Start camera"}</button>
        <p className="fine-print">Video stays in this browser. Phone camera access requires an HTTPS preview; localhost works on the same device.</p>
      </aside>
    </section>
  );
}
