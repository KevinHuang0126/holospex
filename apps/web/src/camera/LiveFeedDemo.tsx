import { useEffect, useMemo, useState } from "react";
import { useCamera } from "../input/useCamera";
import { identifyLiveFrame, loadIdentificationModel, type IdentificationModel, type LiveFrameIdentifier } from "../input/liveIdentification";
import { createIdentificationReadinessMonitor } from "../input/identificationReadiness";
import { parseVideoSourceUrl, type VideoSourceFormat } from "../input/videoSource";
import { LiveFeedHud } from "./LiveFeedHud";
import { UploadedVideoHud } from "./UploadedVideoHud";
import { UploadedImageHud } from "./UploadedImageHud";
import { HudRecorder } from "./HudRecorder";
import "./deviceSetup.css";
import "./liveFeed.css";

/** Cameras, images, videos and remote streams share the trained model connection. */
export function LiveFeedDemo({ visible, source = "live" }: { visible: boolean; source?: "live" | "image" | "upload" | "url" }) {
  const camera = useCamera();
  const [deviceId, setDeviceId] = useState("");
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [model, setModel] = useState<IdentificationModel | null>(null);
  const [checking, setChecking] = useState(true);
  const [checkRevision, setCheckRevision] = useState(0);
  const [canvas, setCanvas] = useState<HTMLCanvasElement | null>(null);
  const [clip, setClip] = useState<File | null>(null);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [url, setUrl] = useState<string | null>(null);
  const [fileError, setFileError] = useState<string | null>(null);
  const [streamDraft, setStreamDraft] = useState("");
  const [streamFormat, setStreamFormat] = useState<VideoSourceFormat>("auto");
  const [connection, setConnection] = useState<{ url: string; format: VideoSourceFormat; revision: number } | null>(null);
  const [connectionRevision, setConnectionRevision] = useState(0);
  const [outlineOnly, setOutlineOnly] = useState(false);
  const [showScores, setShowScores] = useState(false);
  const [cutoff, setCutoff] = useState<number | null>(null);
  const [cutoffDraft, setCutoffDraft] = useState<string | null>(null);
  const canAdjustCutoff = model?.supportsMinimumConfidence === true;
  const minimumConfidence = model ? canAdjustCutoff ? cutoff ?? model.minimumConfidence : model.minimumConfidence : undefined;
  const cutoffValue = cutoffDraft ?? (minimumConfidence === undefined ? "" : String(Number((minimumConfidence * 100).toFixed(8))));
  const cutoffPercent = cutoffValue.trim() ? Number(cutoffValue) : NaN;
  const validCutoff = Number.isFinite(cutoffPercent) && cutoffPercent >= 0 && cutoffPercent <= 100;
  const appearance = useMemo(() => ({ fillOpacity: outlineOnly ? 0 : 0.14, showConfidence: showScores }), [outlineOnly, showScores]);
  const uploaded = source === "upload";
  const imageInput = source === "image";
  const cameraInput = source === "live";
  const remote = source === "url";
  const videoInput = uploaded || remote;
  const activeUrl = remote ? connection?.url ?? null : url;
  const mediaKey = remote ? `stream:${connection?.revision ?? "disconnected"}` : url ?? "no-video";
  const title = imageInput ? "Uploaded image identification" : remote ? "Stream link identification" : uploaded ? "Uploaded video identification" : "Camera identification";

  useEffect(() => {
    if (!clip) { setUrl(null); return; }
    const next = URL.createObjectURL(clip);
    setUrl(next);
    return () => URL.revokeObjectURL(next);
  }, [clip]);
  useEffect(() => { if (!cameraInput) camera.stop(); }, [cameraInput, camera.stop]);

  useEffect(() => {
    if (!cameraInput) return;
    let disposed = false, generation = 0;
    const refresh = async () => {
      const current = ++generation;
      try {
        const available = await navigator.mediaDevices?.enumerateDevices();
        if (!disposed && current === generation) setDevices(available?.filter(device => device.kind === "videoinput") ?? []);
      } catch { /* Starting capture reports permission and device errors. */ }
    };
    void refresh();
    navigator.mediaDevices?.addEventListener("devicechange", refresh);
    return () => { disposed = true; generation++; navigator.mediaDevices?.removeEventListener("devicechange", refresh); };
  }, [camera.stream, cameraInput]);

  useEffect(() => {
    const monitor = createIdentificationReadinessMonitor({
      load: loadIdentificationModel,
      onChecking: setChecking,
      onReady(next) {
        setModel(previous => previous?.model.id === next.model.id && previous.model.version === next.model.version
          && previous.minimumConfidence === next.minimumConfidence && previous.supportsMinimumConfidence === next.supportsMinimumConfidence
          ? previous : next);
      },
      onUnavailable: () => setModel(null),
    });
    const reconnect = () => { if (!document.hidden) monitor.check(); };
    window.addEventListener("online", reconnect);
    document.addEventListener("visibilitychange", reconnect);
    monitor.check();
    return () => {
      monitor.dispose();
      window.removeEventListener("online", reconnect);
      document.removeEventListener("visibilitychange", reconnect);
    };
  }, [checkRevision]);

  const identify = useMemo<LiveFrameIdentifier | undefined>(() => model ? async (frame, image, signal) => {
    const result = await identifyLiveFrame(frame, image, signal, model.supportsMinimumConfidence ? minimumConfidence : undefined);
    if (result.model?.id !== model.model.id || result.model?.version !== model.model.version) {
      throw new Error("Identification model changed. Refresh the model connection.");
    }
    return result;
  } : undefined, [model, minimumConfidence]);

  return <section className="live-feed-demo" aria-label={title}>
    <h2>{title}</h2>
    <p>{imageInput ? "Choose a surgical image, adjust the confidence cutoff if needed, then select Identify image to label its anatomy."
      : remote ? "Connect a live stream or direct video link to identify anatomy in its video frames."
      : uploaded ? "Choose a surgical video to identify anatomy with the trained model. Play through the clip or pause and seek to inspect a frame."
      : "Preview a camera or USB capture feed, then tap Capture & identify to freeze and label one frame. Use Retake to return to the live preview."}</p>
    {imageInput ? <div className="setup-row live-feed-controls">
      <label className="setup-field live-feed-device">Upload image<input type="file" accept="image/jpeg,image/png,image/webp,.jpg,.jpeg,.png,.webp" onChange={event => {
        const file = event.target.files?.[0]; event.target.value = "";
        if (file) setImageFile(file);
      }} /></label>
      <button className="secondary" disabled={!imageFile} onClick={() => setImageFile(null)}>Remove image</button>
      {imageFile && <span className="uploaded-video-name">{imageFile.name}</span>}
    </div> : remote ? <form className="setup-row live-feed-controls" noValidate onSubmit={event => {
      event.preventDefault();
      try {
        const next = parseVideoSourceUrl(streamDraft);
        const revision = connectionRevision + 1;
        setFileError(null); setCanvas(null); setConnectionRevision(revision);
        setConnection({ url: next, format: streamFormat, revision });
      } catch (cause) { setFileError(cause instanceof Error ? cause.message : "Enter a direct HTTPS video or HLS stream link."); }
    }}>
      <label className="setup-field live-feed-device">Video / live stream URL<input type="url" value={streamDraft} required autoComplete="off" spellCheck={false} placeholder="https://your-stream.example/live.m3u8" aria-describedby="stream-link-help" onChange={event => { setStreamDraft(event.target.value); setFileError(null); }} /></label>
      <label className="setup-field">Link format<select value={streamFormat} onChange={event => setStreamFormat(event.target.value as VideoSourceFormat)}>
        <option value="auto">Automatic</option><option value="hls">HLS stream (.m3u8)</option><option value="video">Direct video (MP4 / WebM)</option>
      </select></label>
      <button type="submit">{connection ? "Reconnect stream" : "Connect stream"}</button>
      <button type="button" className="secondary" disabled={!connection} onClick={() => { setConnection(null); setCanvas(null); setFileError(null); }}>Disconnect stream</button>
      <p id="stream-link-help" className="fine-print stream-link-help">Use a direct HTTPS media link with cross-origin access enabled by its host. YouTube pages and RTSP camera addresses need a browser-compatible HLS feed. For an HLS link without .m3u8 in its path, choose HLS stream.</p>
    </form> : uploaded ? <div className="setup-row live-feed-controls">
      <label className="setup-field live-feed-device">Upload video<input type="file" accept="video/*,.mp4,.mov,.webm,.m4v" onChange={event => {
        const file = event.target.files?.[0]; event.target.value = "";
        if (!file) return;
        setFileError(null);
        if (!file.size) { setFileError("This video is empty. Choose another file."); return; }
        setCanvas(null); setUrl(null); setClip(file);
      }} /></label>
      <button className="secondary" disabled={!clip} onClick={() => { setClip(null); setUrl(null); setCanvas(null); setFileError(null); }}>Remove video</button>
      {clip && <span className="uploaded-video-name">{clip.name}</span>}
    </div> : <div className="setup-row live-feed-controls">
      <label className="setup-field live-feed-device">Camera / capture device<select value={deviceId} onChange={event => {
        camera.stop(); setDeviceId(event.target.value);
      }}>
        <option value="">Rear / default camera</option>
        {deviceId && !devices.some(device => device.deviceId === deviceId) && <option value={deviceId}>Previously selected device</option>}
        {devices.map((device, index) => <option key={device.deviceId || index} value={device.deviceId}>{device.label || `Camera / capture device ${index + 1}`}</option>)}
      </select></label>
      <button disabled={camera.starting || !!camera.stream} onClick={() => { void camera.start(deviceId || undefined, { width: { ideal: 1280, max: 1920 }, height: { ideal: 720, max: 1920 } }); }}>{camera.starting ? "Starting camera…" : "Start camera"}</button>
      <button className="secondary" disabled={!camera.stream && !camera.starting} onClick={camera.stop}>Stop camera</button>
    </div>}
    {cameraInput && !devices.length && !camera.stream && <p className="fine-print">Start the camera to allow camera access. Connected USB capture devices appear in the device list.</p>}
    {cameraInput && camera.error && <p className="error" role="alert">{camera.error}</p>}
    {fileError && <p className="error" role="alert">{fileError}</p>}

    <p className="live-feed-status" role="status">{model
      ? !visible ? videoInput ? "Anatomy hidden; identification paused." : "Anatomy overlays hidden."
        : imageInput ? "Identification model ready. Choose an image, then select Identify image."
        : remote ? connection ? "Stream identification enabled · trained on Endoscapes-Seg50." : "Identification model ready. Connect a stream to begin."
        : uploaded ? url ? "Video identification enabled · trained on Endoscapes-Seg50." : "Identification model ready. Choose a video to begin."
        : camera.stream ? "Camera ready. Capture one frame to identify anatomy." : "Identification model ready. Start the camera, then capture a frame."
      : checking ? "Connecting to the identification model…" : "Identification model connection unavailable. Retrying automatically every 10 seconds; preview remains available."}</p>
    <button className="secondary" disabled={checking} onClick={() => { setModel(null); setCheckRevision(value => value + 1); }}>Refresh model</button>
    <div className="setup-row live-feed-controls identification-display-controls">
      <label className="setup-field">Overlay appearance<select value={outlineOnly ? "outline" : "fill"} onChange={event => setOutlineOnly(event.target.value === "outline")}>
        <option value="fill">Masks + outlines</option><option value="outline">Outlines only</option>
      </select></label>
      <label className="identification-score-toggle"><input type="checkbox" checked={showScores} onChange={event => setShowScores(event.target.checked)} /> Show model scores</label>
    </div>
    <form className="setup-row live-feed-controls identification-cutoff-controls" noValidate onSubmit={event => {
      event.preventDefault();
      if (!canAdjustCutoff || !validCutoff) return;
      setCutoff(cutoffPercent / 100); setCutoffDraft(null);
    }}>
      <label className="setup-field">Confidence cutoff (%)<input type="number" min="0" max="100" step="any" inputMode="decimal" required
        disabled={!canAdjustCutoff} value={cutoffValue} aria-describedby="confidence-cutoff-help"
        aria-invalid={cutoffDraft !== null && !validCutoff} onChange={event => setCutoffDraft(event.target.value)} /></label>
      <button type="submit" disabled={!canAdjustCutoff || !validCutoff || cutoffPercent / 100 === minimumConfidence}>Apply cutoff</button>
      {minimumConfidence !== undefined && <span className="fine-print" role="status">Applied: {Number((minimumConfidence * 100).toFixed(8))}%</span>}
      {cutoffDraft !== null && !validCutoff && <p className="error stream-link-help" role="alert">Enter a percentage from 0 to 100.</p>}
      <p id="confidence-cutoff-help" className="fine-print stream-link-help">{!model ? "Connect the identification model to set a cutoff."
        : !canAdjustCutoff ? "This model connection uses its default cutoff. Refresh the model after updating the runner to enable this input."
        : `Lower values include more uncertain regions; higher values hide them. Applying a cutoff clears the previous result.${videoInput ? "" : imageInput ? " Select Identify image afterward." : " Capture a new frame afterward."}`}</p>
    </form>
    {showScores && <p className="fine-print">Scores describe model confidence for the displayed region, not its measured accuracy.</p>}
    <p className="fine-print">{imageInput
      ? "Your image is previewed locally. Only selecting Identify image sends a resized JPEG to Holospex. Labels remain attached to that image."
      : remote
      ? "The stream loads directly from its host in your browser. With overlays on and the model ready, selected video frames are sent to Holospex for identification. The link is not sent to the model."
      : uploaded
      ? "The video plays from your device. With overlays on and the model ready, selected frames are sent to Holospex for identification. The full video is not uploaded."
      : "Only tapping Capture & identify sends a camera frame to Holospex. The captured image stays still while its anatomy is identified. The model was trained on surgical views; a room camera does not reveal internal anatomy."}</p>

    <div className="live-feed-preview">
      {imageInput ? <UploadedImageHud file={imageFile} visible={visible} identify={identify} minimumConfidence={minimumConfidence} appearance={appearance} />
        : videoInput ? <UploadedVideoHud key={mediaKey} src={activeUrl} visible={visible} sourceKind={remote ? "url" : "upload"} streamFormat={connection?.format}
        identify={identify} minimumConfidence={minimumConfidence} appearance={appearance} onCanvas={setCanvas} />
        : <LiveFeedHud stream={camera.stream} interrupted={camera.interrupted} visible={visible}
          identify={identify} minimumConfidence={minimumConfidence} appearance={appearance} onCanvas={setCanvas} />}
    </div>
    {!imageInput && <>
      <HudRecorder key={videoInput ? mediaKey : "live"} canvas={canvas} name={remote ? "stream" : uploaded ? "video" : "live"} />
      <p className="fine-print">Backup recordings stay in this browser. Stop and save the recording before switching feeds.</p>
    </>}
  </section>;
}
