import { useEffect, useRef, useState } from "react";

/** Records the composed canvas, including provenance and warnings; never raw hidden anatomy. */
export function HudRecorder({ canvas, name }: { canvas: HTMLCanvasElement | null; name: "video" | "model" }) {
  const active = useRef<MediaRecorder | null>(null);
  const stream = useRef<MediaStream | null>(null);
  const [recording, setRecording] = useState(false);
  const [download, setDownload] = useState<{ url: string; extension: string } | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => () => {
    if (active.current) { active.current.onstop = null; if (active.current.state !== "inactive") active.current.stop(); }
    stream.current?.getTracks().forEach(track => track.stop());
  }, []);
  useEffect(() => () => { if (download) URL.revokeObjectURL(download.url); }, [download]);
  function start() {
    setError(null); setDownload(null);
    if (!canvas || typeof canvas.captureStream !== "function" || typeof MediaRecorder === "undefined") { setError("Recording is unavailable in this browser. Use the device's screen recorder."); return; }
    try {
      const mimeType = ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/mp4"].find(type => MediaRecorder.isTypeSupported(type));
      if (!mimeType) throw new Error("No supported recording format. Use the device's screen recorder.");
      const capture = canvas.captureStream(30); stream.current = capture;
      const recorder = new MediaRecorder(capture, { mimeType }); active.current = recorder;
      const chunks: Blob[] = [];
      let failed = false;
      recorder.ondataavailable = event => { if (event.data.size) chunks.push(event.data); };
      recorder.onerror = () => { failed = true; setError("Recording failed. Try the device's screen recorder."); setRecording(false); capture.getTracks().forEach(track => track.stop()); };
      recorder.onstop = () => {
        capture.getTracks().forEach(track => track.stop()); setRecording(false);
        if (!failed && chunks.length) setDownload({ url: URL.createObjectURL(new Blob(chunks, { type: recorder.mimeType })), extension: mimeType.includes("mp4") ? "mp4" : "webm" });
      };
      recorder.start(250); setRecording(true);
    } catch (cause) { stream.current?.getTracks().forEach(track => track.stop()); setError(cause instanceof Error ? cause.message : "Recording failed."); }
  }
  return <div className="setup-row">
    <button onClick={recording ? () => active.current?.stop() : start} disabled={!canvas}>{recording ? "Stop backup recording" : "Record backup"}</button>
    {download && <a download={`holospex-${name}-backup.${download.extension}`} href={download.url}>Save {name} backup</a>}
    {recording && <span role="status">Recording this view. Stop and save before switching modes.</span>}
    {error && <span className="error" role="alert">{error}</span>}
  </div>;
}
