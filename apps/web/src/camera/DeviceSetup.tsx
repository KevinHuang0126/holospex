import { useEffect, useRef, useState } from "react";
import { useCamera } from "../input/useCamera";
import { sourceLabels, type ResultSource } from "../overlays/frameInput";
import type { HudMode } from "../overlays/hudControls";
import { useHudControls } from "../overlays/useHudControls";
import "./deviceSetup.css";

/** Step 1 readiness + a working control surface for Person 3 to integrate. */
export function DeviceSetup() {
  const hud = useHudControls();
  const camera = useCamera();
  const cameraVideo = useRef<HTMLVideoElement>(null);
  const [clip, setClip] = useState<File | null>(null);
  const [clipUrl, setClipUrl] = useState<string>();
  const [videoReady, setVideoReady] = useState(false);
  const [playbackPassed, setPlaybackPassed] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [cameraPlaying, setCameraPlaying] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [seekMs, setSeekMs] = useState("0");
  const [deviceName, setDeviceName] = useState("");
  const [modelName, setModelName] = useState("");
  const [modelConfirmed, setModelConfirmed] = useState(false);
  const clock = useRef<number | null>(null);

  useEffect(() => {
    setVideoReady(false);
    setPlaybackPassed(false);
    setPlaying(false);
    setError(null);
    clock.current = null;
    if (!clip) { setClipUrl(undefined); return; }
    const url = URL.createObjectURL(clip);
    setClipUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [clip]);
  useEffect(() => {
    const video = cameraVideo.current;
    if (!video) return;
    setCameraPlaying(false);
    video.srcObject = camera.stream;
    return () => { video.srcObject = null; };
  }, [camera.stream]);

  const cameraPassed = !!camera.stream && cameraPlaying && !camera.interrupted;
  async function play() {
    setError(null);
    try { await hud.play(); }
    catch (cause) { setError(cause instanceof Error ? cause.message : "Video playback failed."); }
  }
  function seek() {
    setError(null);
    try {
      if (!seekMs.trim()) throw new Error("Enter a seek time in milliseconds.");
      hud.seek(Number(seekMs));
    } catch (cause) { setError(cause instanceof Error ? cause.message : "Seek failed."); }
  }

  return <section className="device-setup" aria-labelledby="setup-title">
    <div className="setup-header"><p className="eyebrow">Demo setup · Steps 1–3</p><h2 id="setup-title">Check the demo device</h2>
      <p>React + TypeScript is confirmed. Test the camera and an approved local clip on the device you will present with.</p></div>
    <div className="setup-grid">
      <div>
        <div className="setup-row" role="group" aria-label="Input mode">
          <button aria-pressed={hud.state.experience === "video"} onClick={() => { camera.stop(); hud.setExperience("video"); }}>Video</button>
          <button aria-pressed={hud.state.experience === "model"} onClick={() => { setPlaying(false); hud.setExperience("model"); }}>Physical model</button>
        </div>
        <div hidden={hud.state.experience !== "video"}>
          <label className="setup-field">Local test clip<input type="file" accept="video/*" onChange={event => {
            const next = event.target.files?.[0] ?? null;
            hud.pause();
            setVideoReady(false);
            setPlaybackPassed(false);
            hud.setMedia(next ? `local:${next.name}:${next.size}:${next.lastModified}` : null);
            setClip(next);
          }} /></label>
          <video className="setup-video" ref={hud.bindVideo} src={clipUrl} playsInline muted preload="metadata" aria-label="Local video playback test"
            onLoadedData={() => setVideoReady(true)}
            onPlaying={() => setPlaying(true)} onPause={() => setPlaying(false)} onEnded={() => setPlaying(false)}
            onSeeking={() => { clock.current = null; }}
            onTimeUpdate={event => {
              const video = event.currentTarget;
              if (!video.paused && !video.seeking && clock.current !== null && video.currentTime > clock.current) setPlaybackPassed(true);
              clock.current = video.currentTime;
            }}
            onError={() => { setVideoReady(false); setPlaybackPassed(false); setPlaying(false); setError("This clip could not be decoded. Try the actual demo format in this browser."); }} />
          <div className="setup-row"><button onClick={play} disabled={!videoReady || playing}>Play</button><button onClick={hud.pause} disabled={!playing}>Pause</button>
            <label>Seek (ms)<input className="setup-seek" type="number" min="0" step="1" value={seekMs} onChange={event => setSeekMs(event.target.value)} /></label>
            <button onClick={seek} disabled={!videoReady}>Seek</button></div>
          <p role="status">Video: {playbackPassed ? "playback verified on this device" : videoReady ? "loaded; press Play to verify playback" : "select a local clip to test"}.</p>
          <p className="fine-print">The clip stays in this browser. No file is uploaded or added to the repository.</p>
        </div>
        <div hidden={hud.state.experience !== "model"}>
          <video className="setup-video" ref={cameraVideo} autoPlay muted playsInline aria-label="Live camera preview"
            onPlaying={event => setCameraPlaying(event.currentTarget.videoWidth > 0 && event.currentTarget.videoHeight > 0)}
            onError={() => { setCameraPlaying(false); setError("Camera preview could not play."); }} />
          <div className="setup-row"><button onClick={() => { setError(null); void camera.start(); }} disabled={camera.starting || !!camera.stream}>{camera.starting ? "Starting camera…" : "Start camera"}</button>
            <button onClick={camera.stop} disabled={!camera.stream && !camera.starting}>Stop camera</button></div>
          <p role="status">Camera: {camera.interrupted ? "interrupted; waiting for camera frames" : cameraPassed ? "preview verified on this device" : "not verified"}.</p>
          <p className="fine-print">Physical labels require marker registration in a later step. This preview does not locate anatomy.</p>
        </div>
        {(error || camera.error) && <p role="alert" className="error">{error || camera.error}</p>}
      </div>
      <div>
        <h3>Hardware readiness</h3>
        <p>Camera connection: {window.isSecureContext && !!navigator.mediaDevices?.getUserMedia ? "supported; permission and preview still need testing" : "unavailable — use HTTPS or localhost in a supported browser"}.</p>
        <label className="setup-field">Demo device and browser<input value={deviceName} onChange={event => setDeviceName(event.target.value)} placeholder="For example: laptop, Chrome" /></label>
        <label className="setup-field">Physical model and marker<input value={modelName} onChange={event => { setModelName(event.target.value); setModelConfirmed(false); }} placeholder="Model name and planned marker" /></label>
        <label className="setup-check"><input type="checkbox" checked={modelConfirmed} onChange={event => setModelConfirmed(event.target.checked)} disabled={!modelName.trim()} />I have this model and marker available.</label>
        <p role="status">Hardware details: {deviceName.trim() && modelName.trim() && modelConfirmed ? "recorded for this session; alignment still needs testing" : "awaiting device and model confirmation"}.</p>
        <hr />
        <h3>Frontend controls</h3>
        <label className="setup-field">Learning mode<select value={hud.state.mode} onChange={event => hud.setMode(event.target.value as HudMode)}>
          <option value="learn">Learn</option><option value="identify">Identify (Quiz)</option><option value="assess">Assess (Quiz)</option><option value="feedback">Feedback</option>
        </select></label>
        <label className="setup-field">Overlay source<select value={hud.state.source} onChange={event => hud.setSource(event.target.value as ResultSource)}>
          {Object.entries(sourceLabels).map(([value, label]) => <option key={value} value={value}>{label}</option>)}
        </select></label>
        <label className="setup-check"><input type="checkbox" checked={hud.state.overlaysRequested} onChange={event => hud.showOverlays(event.target.checked)} />Show overlays</label>
        <p role="status">Overlay policy: {hud.overlaysVisible ? "allowed when matching data is available" : "hidden"}.</p>
        <p className="fine-print">This setup checks controls only. No overlay data or original-frame index is loaded, so no anatomy or frame ID is displayed. Source selection applies to video; model labels will use known registered locations.</p>
      </div>
    </div>
  </section>;
}
