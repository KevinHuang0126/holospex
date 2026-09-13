import { useEffect, useRef, useState } from "react";
import type { FrameResult } from "@holospex/contracts";
import aruco from "js-aruco2";
import { useCamera } from "../input/useCamera";
import { useHudControls } from "../overlays/useHudControls";
import { VideoHud } from "../overlays/VideoHud";
import { parseResultSet } from "../overlays/videoResults";
import { sourceLabels, type ResultSource } from "../overlays/frameInput";
import type { HudMode } from "../overlays/hudControls";
import { ModelHud } from "./ModelHud";
import { parseModelRegistration, type ModelRegistration } from "./modelRegistration";
import { HudRecorder } from "./HudRecorder";
import "./deviceSetup.css";
const { AR } = aruco;

export function HudDemo() {
  const hud = useHudControls({ frameClock: "external" }), camera = useCamera();
  const [clip, setClip] = useState<File | null>(null), [url, setUrl] = useState<string>();
  const [results, setResults] = useState<FrameResult[]>([]);
  const [registration, setRegistration] = useState<ModelRegistration | null>(null);
  const [canvas, setCanvas] = useState<HTMLCanvasElement | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [ready, setReady] = useState(false), [playing, setPlaying] = useState(false);
  const [seek, setSeek] = useState("0"), [threshold, setThreshold] = useState("");
  const [tracked, setTracked] = useState(false);
  const [deviceId, setDeviceId] = useState("");
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const request = useRef(0);
  useEffect(() => {
    setReady(false); setPlaying(false);
    if (!clip) { setUrl(undefined); return; }
    const next = URL.createObjectURL(clip); setUrl(next);
    return () => URL.revokeObjectURL(next);
  }, [clip]);
  useEffect(() => {
    if (camera.stream) navigator.mediaDevices.enumerateDevices().then(items => setDevices(items.filter(item => item.kind === "videoinput"))).catch(() => {});
  }, [camera.stream]);
  useEffect(() => () => { request.current += 1; }, []);
  async function loadResults(file: File | undefined) {
    const generation = ++request.current;
    setResults([]); setError(null);
    if (!file) return;
    try {
      const parsed = parseResultSet(JSON.parse(await file.text()));
      if (generation !== request.current) return;
      setResults(parsed); hud.setMedia(parsed[0]?.mediaId ?? null);
    } catch (cause) { if (generation === request.current) setError(cause instanceof Error ? cause.message : "Invalid result file."); }
  }
  function saveMarker() {
    if (!registration) return;
    const svg = new AR.Dictionary(registration.dictionary).generateSVG(registration.markerId);
    const markerUrl = URL.createObjectURL(new Blob([svg], { type: "image/svg+xml" }));
    const anchor = document.createElement("a"); anchor.href = markerUrl; anchor.download = `marker-${registration.markerId}.svg`; anchor.click();
    setTimeout(() => URL.revokeObjectURL(markerUrl), 1000);
  }
  return <section className="device-setup" aria-labelledby="hud-title">
    <p className="eyebrow">AR / HUD demo</p><h2 id="hud-title">Anatomy in context</h2>
    <div className="setup-row" role="group" aria-label="HUD input">
      <button aria-pressed={hud.state.experience === "video"} onClick={() => { camera.stop(); hud.setExperience("video"); }}>Video</button>
      <button aria-pressed={hud.state.experience === "model"} onClick={() => hud.setExperience("model")}>Physical model</button>
      <label>Learning mode <select value={hud.state.mode} onChange={event => hud.setMode(event.target.value as HudMode)}><option value="learn">Learn</option><option value="identify">Identify</option><option value="assess">Assess</option><option value="feedback">Feedback</option></select></label>
      <label className="setup-check"><input type="checkbox" checked={hud.state.overlaysRequested} onChange={event => hud.showOverlays(event.target.checked)} />Show overlays</label>
    </div>
    {hud.state.experience === "video" ? <>
      <div className="setup-row">
        <label className="setup-field">Shared surgical clip<input type="file" accept="video/*" onChange={event => {
          request.current += 1; hud.pause(); setResults([]); setError(null); setReady(false);
          const file = event.target.files?.[0] ?? null; setClip(file); hud.setMedia(file ? `local:${file.name}` : null);
        }} /></label>
        <label className="setup-field">Results for this clip (JSON)<input type="file" accept="application/json,.json" disabled={!clip} onChange={event => { void loadResults(event.target.files?.[0]); event.target.value = ""; }} /></label>
        <label className="setup-field">Source<select value={hud.state.source} onChange={event => hud.setSource(event.target.value as ResultSource)}>{Object.entries(sourceLabels).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>
        <label className="setup-field">ML threshold from Person 1<input type="number" min="0" max="1" step="0.01" value={threshold} placeholder="Not supplied" onChange={event => setThreshold(event.target.value)} /></label>
      </div>
      {url ? <VideoHud key={url} src={url} mediaId={hud.state.mediaId ?? ""} results={results} source={hud.state.source} mode={hud.state.mode} visible={hud.state.overlaysRequested} minimumConfidence={threshold.trim() ? Number(threshold) : undefined}
        bindVideo={hud.bindVideo} onCanvas={setCanvas} onReady={() => setReady(true)} onPlayback={setPlaying}
        onDisplayedFrame={frame => { if (frame) hud.reportDisplayedFrame(frame, hud.state.revision); else hud.invalidate(); }} /> : <p className="notice">Choose the permitted shared clip, then its result JSON. The bundled lesson remains a synthetic illustration.</p>}
      <div className="setup-row"><button disabled={!ready || playing} onClick={() => { setError(null); void hud.play().catch(cause => setError(String(cause))); }}>Play</button><button disabled={!playing} onClick={hud.pause}>Pause</button>
        <label>Seek (ms)<input className="setup-seek" type="number" min="0" value={seek} onChange={event => setSeek(event.target.value)} /></label>
        <button disabled={!ready} onClick={() => { try { if (!seek.trim()) throw new Error("Enter a timestamp."); hud.seek(Number(seek)); setError(null); } catch (cause) { setError(String(cause)); } }}>Seek</button>
        <button disabled={!ready} onClick={() => { try { hud.seek(0); void hud.play().catch(cause => setError(String(cause))); } catch (cause) { setError(String(cause)); } }}>Replay</button>
      </div>
      <p className="fine-print">{results.length} validated frame results. Only the selected source at the displayed presentation time can draw anatomy. Files stay local to this browser.</p>
    </> : <>
      <label className="setup-field">Measured model configuration (JSON)<input type="file" accept="application/json,.json" onChange={async event => {
        const file = event.target.files?.[0]; setRegistration(null); setError(null);
        try { if (file) setRegistration(parseModelRegistration(JSON.parse(await file.text()))); } catch (cause) { setError(String(cause)); }
      }} /></label>
      <div className="setup-row"><label>Camera <select value={deviceId} onChange={event => { camera.stop(); setDeviceId(event.target.value); }}><option value="">Rear/default camera</option>{devices.map(device => <option key={device.deviceId} value={device.deviceId}>{device.label || "Camera"}</option>)}</select></label>
        <button disabled={camera.starting || !!camera.stream} onClick={() => { void camera.start(deviceId || undefined); }}>{camera.starting ? "Starting…" : "Start camera"}</button>
        <button disabled={!camera.stream && !camera.starting} onClick={camera.stop}>Stop camera</button>
        <button disabled={!registration} onClick={saveMarker}>Save printable marker</button></div>
      <ModelHud stream={camera.stream} registration={registration} visible={hud.state.overlaysRequested} mode={hud.state.mode} interrupted={camera.interrupted} onCanvas={setCanvas} onTracking={setTracked} />
      <p className="fine-print">{tracked ? "Marker pose is available." : "Labels are withheld until registration succeeds."} Attach the printed marker rigidly to the configured model at its measured position. Tracking labels use known model locations.</p>
    </>}
    {(error || camera.error) && <p className="error" role="alert">{error || camera.error}</p>}
    <HudRecorder key={hud.state.experience} canvas={canvas} name={hud.state.experience} />
    <p className="fine-print">Confirm the actual camera, model, calibration and clip on the demo device. Camera access needs HTTPS or localhost. Stop and save a backup in each mode before leaving it.</p>
  </section>;
}
