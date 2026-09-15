import { useEffect, useLayoutEffect, useRef, useState } from "react";
import type { FrameResult } from "@holospex/contracts";
import { MAX_LIVE_FRAME_AGE_MS } from "../input/liveFramePipeline";
import type { LiveFrameIdentifier } from "../input/liveIdentification";
import type { VideoSourceFormat } from "../input/videoSource";
import { drawHud, type HudAppearance } from "../overlays/drawHud";
import { sourceLabels } from "../overlays/frameInput";
import type { DisplayedFrame } from "../overlays/selectFrame";
import { visibleStructures } from "../overlays/videoResults";
import { createUploadedVideoSession, videoCaptureSize, type UploadedVideoCapture } from "./uploadedVideoSession";
import { attachVideoElementSource } from "./videoElementSource";
import { freezeVideoFrame, type InspectedVideoFrame } from "./videoInspection";

export interface UploadedVideoHudProps {
  src: string | null;
  sourceKind?: "upload" | "url";
  streamFormat?: VideoSourceFormat;
  visible: boolean;
  identify?: LiveFrameIdentifier;
  minimumConfidence?: number;
  appearance?: HudAppearance;
  onCanvas?: (canvas: HTMLCanvasElement | null) => void;
  onDisplayedFrame?: (frame: DisplayedFrame | null) => void;
}
type Capture = UploadedVideoCapture<HTMLCanvasElement>;
type Identified = Capture & { result: FrameResult };
const FRAME_INTERVAL_MS = 200;
const formatTime = (seconds: number) => `${Math.floor(seconds / 60)}:${Math.floor(seconds % 60).toString().padStart(2, "0")}`;

/** Media is decoded in the browser; only sampled JPEG frames reach the existing model. */
export function UploadedVideoHud(props: UploadedVideoHudProps) {
  const video = useRef<HTMLVideoElement>(null), canvas = useRef<HTMLCanvasElement>(null);
  const current = useRef(props); current.current = props;
  const latest = useRef<Capture | null>(null), identified = useRef<Identified | null>(null);
  const displayed = useRef<Capture | null>(null), inspection = useRef<InspectedVideoFrame<HTMLCanvasElement> | null>(null);
  const status = useRef("Choose a video to identify anatomy.");
  const terminalError = useRef<string | null>(null);
  const [notice, setNotice] = useState(status.current);
  const [inspecting, setInspecting] = useState(false), [playbackRate, setPlaybackRate] = useState(1);
  const [playback, setPlayback] = useState({ time: 0, duration: 0, playing: false, ready: false, live: false, failed: false, liveEdge: 0 });
  const live = useRef(false);
  const paint = useRef(() => {}), changeSettings = useRef(() => {}), prepareSeek = useRef(() => {});
  const inspectDisplayed = useRef(() => {}), releaseInspection = useRef((_stayAtCurrentPosition = false) => {});

  paint.current = () => {
    if (!canvas.current) return;
    const active = current.current, element = video.current;
    // A deliberate pause can keep the result for the unchanged image indefinitely.
    const valid = identified.current && (element?.paused || performance.now() - identified.current.capturedAt < MAX_LIVE_FRAME_AGE_MS);
    const paired = active.src && active.identify && active.visible && !document.hidden && valid ? identified.current : null;
    const display = active.src && !document.hidden ? paired ?? inspection.current ?? latest.current : null;
    displayed.current = display;
    const overlay = visibleStructures(paired?.result ?? null, active.visible, "learn", active.minimumConfidence);
    const warning = overlay.warning?.replace(/Unable to assess/g, "Unable to identify") ?? null;
    const message = !active.src ? active.sourceKind === "url" ? "Connect a video URL or HLS stream to identify anatomy." : "Choose a video to identify anatomy."
      : document.hidden ? "Video paused while this page is hidden."
      : terminalError.current ? terminalError.current
      : !display ? status.current
      : !active.visible ? "Video preview. Anatomy hidden; identification paused."
      : !active.identify ? "Video preview. Connecting to the identification model; this frame will be identified when connected."
      : paired ? warning ?? (overlay.structures.length ? inspection.current ? "Inspecting the frozen video frame. Play to resume." : "Anatomy identified in the displayed video frame." : "No supported anatomy identified in this frame.")
      : status.current;
    setNotice(message);
    drawHud(canvas.current, display?.image ?? null, {
      width: display?.frame.width ?? 1280, height: display?.frame.height ?? 720,
      sourceLabel: paired ? sourceLabels[paired.result.source] : active.sourceKind === "url" ? "Video URL feed" : "Uploaded video",
      source: paired?.result.source,
      statusLabel: display ? `${formatTime(display.frame.timestampMs / 1000)} · ${display.frame.timestampMs.toFixed(1)} ms${paired && !element?.paused ? ` · ${Math.round(performance.now() - paired.capturedAt)} ms behind capture` : ""}` : "No video frame",
      structures: paired ? overlay.structures : [], labelSlots: 6, appearance: active.appearance,
      warning: !display || (active.identify && active.visible && !paired) ? message : paired ? warning : null,
    });
    active.onDisplayedFrame?.(display?.frame ?? null);
  };
  useLayoutEffect(() => { paint.current(); }, [props.minimumConfidence, props.appearance]);
  useEffect(() => {
    props.onCanvas?.(canvas.current);
    const observer = new ResizeObserver(() => paint.current()); observer.observe(canvas.current!);
    return () => { observer.disconnect(); current.current.onCanvas?.(null); current.current.onDisplayedFrame?.(null); };
  }, []);

  useLayoutEffect(() => {
    const element = video.current!;
    let disposed = false, callback = 0, animation = 0, frameNumber = 0, callbackGeneration = 0;
    let lastSubmittedAt = -Infinity, previousTime = -1, lastPreviewAt = -Infinity, needsPausedCapture = true;
    let seekPending = false, failed = false, awaitingDecodedFrame = props.sourceKind === "url", captureChecked = false;
    let lastDecodedAt = performance.now();
    let retryAt: number | null = null, retryAttempts = 0;
    const mediaId = `${props.sourceKind === "url" ? "url-feed" : "upload"}:${globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random().toString(36).slice(2)}`}`;
    const synchronized = typeof element.requestVideoFrameCallback === "function";
    const raw = document.createElement("canvas");
    const session = createUploadedVideoSession<HTMLCanvasElement>({
      onResult(capture, result) {
        if (disposed || failed || awaitingDecodedFrame || document.hidden || element.seeking || seekPending) return;
        retryAt = null;
        identified.current = { ...capture, result };
        status.current = "Identifying the next video frame."; paint.current();
      },
      onUnavailable(message) {
        if (disposed) return;
        identified.current = null;
        // Aborting HTTP cannot stop a server's already-running CPU inference. A selected
        // paused frame retries briefly so a transient busy response does not strand it.
        if (element.paused && !failed && !awaitingDecodedFrame && !document.hidden && !seekPending && current.current.visible && current.current.identify && retryAttempts < 3) {
          retryAt = performance.now() + 1500; retryAttempts++;
          status.current = `${message} Retrying this video frame…`;
        } else status.current = `${message} Try playing the video or refreshing the model connection.`;
        paint.current();
      },
    });
    const updatePlayback = () => {
      if (!disposed) setPlayback({ time: Number.isFinite(element.currentTime) ? element.currentTime : 0,
        duration: Number.isFinite(element.duration) ? element.duration : 0,
        playing: !element.paused && !element.ended, ready: element.readyState >= 2 && !failed,
        live: live.current || element.duration === Infinity, failed,
        liveEdge: element.seekable.length ? Math.max(0, element.seekable.end(element.seekable.length - 1) - 0.25) : 0 });
    };
    const invalidate = (message: string, clearImage = false, preserveInspection = false) => {
      session.invalidate(); identified.current = null;
      if (!preserveInspection) { inspection.current = null; setInspecting(false); }
      if (clearImage) latest.current = null;
      lastSubmittedAt = -Infinity; previousTime = -1; needsPausedCapture = true;
      retryAt = null; retryAttempts = 0;
      status.current = message; paint.current();
    };
    const capture = (timestampMs: number, decoded = false) => {
      if (disposed || document.hidden || failed || inspection.current || seekPending || element.seeking || !props.src || element.readyState < 2) return;
      const size = videoCaptureSize(element.videoWidth, element.videoHeight);
      if (!size || !Number.isFinite(timestampMs) || timestampMs < 0) return;
      if (decoded) { awaitingDecodedFrame = false; lastDecodedAt = performance.now(); }
      if (awaitingDecodedFrame) return;
      try {
        const now = performance.now();
        if (raw.width !== size.width || raw.height !== size.height) {
          invalidate("Waiting for identification at this video size."); raw.width = size.width; raw.height = size.height;
        }
        if (timestampMs < previousTime) invalidate("Waiting for identification at this video position.", true);
        previousTime = timestampMs;
        const context = raw.getContext("2d");
        if (!context) throw new Error("Video rendering is unavailable.");
        context.drawImage(element, 0, 0, size.width, size.height);
        // A remote video can render but still taint a canvas. Detect this once before
        // painting the public/recording canvas or scheduling repeated JPEG failures.
        if (!captureChecked) { context.getImageData(0, 0, 1, 1); captureChecked = true; }
        const frame: DisplayedFrame = { mediaId, frameNumber: frameNumber++, timestampMs, ...size };
        latest.current = { image: raw, frame, capturedAt: now };
        const active = current.current;
        const canIdentify = active.visible && !!active.identify && (synchronized || element.paused);
        if (canIdentify && !session.busy && now - lastSubmittedAt >= FRAME_INTERVAL_MS) {
          const image = document.createElement("canvas"); image.width = size.width; image.height = size.height;
          const pixels = image.getContext("2d");
          if (!pixels) throw new Error("Video capture is unavailable.");
          pixels.drawImage(raw, 0, 0);
          if (session.submit({ image, frame, capturedAt: now }, () => new Promise<Blob>((resolve, reject) => {
            image.toBlob(blob => blob ? resolve(blob) : reject(new Error("Video frame encoding failed.")), "image/jpeg", 0.9);
          }))) { lastSubmittedAt = now; needsPausedCapture = false; }
        } else if (!canIdentify) needsPausedCapture = false;
        if (!synchronized && !element.paused && active.visible && active.identify)
          status.current = "Video preview. Pause to identify a frame in this browser.";
        paint.current();
      } catch {
        error(props.sourceKind === "url"
          ? "This feed cannot be captured for identification. Its server must allow cross-origin video and canvas access. Reconnect an accessible feed."
          : "Unable to identify. This video frame could not be decoded or captured. Choose another supported video.");
      }
    };
    const pausedFrame = () => {
      if (!element.paused || !needsPausedCapture) return;
      const frozen = inspection.current;
      if (!frozen) { capture(element.currentTime * 1000); return; }
      const active = current.current;
      if (disposed || failed || document.hidden || seekPending || awaitingDecodedFrame || !active.identify || !active.visible || session.busy) return;
      // The still is deliberately frozen; start a fresh processing budget without
      // changing its original frame ID, media timestamp, dimensions or pixels.
      if (session.submit({ ...frozen, capturedAt: performance.now() }, () => new Promise<Blob>((resolve, reject) => {
        frozen.image.toBlob(blob => blob ? resolve(blob) : reject(new Error("Video frame encoding failed.")), "image/jpeg", 0.9);
      }))) needsPausedCapture = false;
    };
    const loaded = () => {
      if (disposed || failed || element.readyState < 2) return;
      awaitingDecodedFrame = false; lastDecodedAt = performance.now(); updatePlayback(); status.current = "Identifying this video frame.";
      pausedFrame(); paint.current();
    };
    const startSeek = () => { if (failed) return; stopFrames(); seekPending = true; invalidate("Seeking. Waiting for the selected video frame.", true); };
    const seeked = () => { if (failed) return; seekPending = false; awaitingDecodedFrame = false; lastDecodedAt = performance.now(); updatePlayback(); if (element.paused) pausedFrame(); else startFrames(); };
    const play = () => { if (failed || (inspection.current && element.paused)) return; invalidate("Identifying video frames."); lastDecodedAt = performance.now(); updatePlayback(); startFrames(); };
    const pause = () => {
      if (disposed || failed) return;
      stopFrames();
      if (inspection.current) { updatePlayback(); pausedFrame(); paint.current(); return; }
      invalidate("Identifying the paused video frame."); updatePlayback(); pausedFrame();
    };
    const buffering = () => {
      if (disposed || failed) return;
      stopFrames();
      awaitingDecodedFrame = true;
      invalidate("Video is buffering or interrupted. Waiting for a decoded frame.", true);
      if (!element.paused) startFrames();
    };
    const error = (message?: string) => {
      if (disposed) return;
      failed = true; stopFrames();
      terminalError.current = message ?? (props.sourceKind === "url"
        ? "The video URL could not be played. Check the URL, supported video format and cross-origin access, then reconnect."
        : "This video could not be played. Try an MP4 encoded with H.264 or another video supported by your browser.");
      invalidate(terminalError.current, true); needsPausedCapture = false; element.pause();
      updatePlayback();
    };
    const visibility = () => {
      if (failed) { paint.current(); return; }
      invalidate(document.hidden ? "Video paused while this page is hidden." : "Identifying the paused video frame.", true);
      if (document.hidden) element.pause(); else pausedFrame();
    };
    const stopFrames = () => {
      callbackGeneration++;
      if (callback) element.cancelVideoFrameCallback(callback);
      callback = 0;
    };
    const startFrames = () => {
      stopFrames();
      if (!synchronized || element.paused || seekPending) return;
      const generation = callbackGeneration;
      const next = (_: number, metadata: VideoFrameCallbackMetadata) => {
        if (disposed || generation !== callbackGeneration) return;
        capture(metadata.mediaTime * 1000, true);
        if (!failed) callback = element.requestVideoFrameCallback(next);
      };
      callback = element.requestVideoFrameCallback(next);
    };
    const preview = () => {
      if (disposed) return;
      const now = performance.now();
      if (!element.paused && now - lastPreviewAt >= FRAME_INTERVAL_MS) {
        lastPreviewAt = now; capture(element.currentTime * 1000);
      }
      animation = requestAnimationFrame(preview);
    };
    changeSettings.current = () => {
      if (failed) { paint.current(); return; }
      session.setIdentifier(current.current.identify);
      invalidate("Waiting for identification of this video frame.", false, true); pausedFrame();
    };
    prepareSeek.current = startSeek;
    inspectDisplayed.current = () => {
      if (disposed || failed || !displayed.current || inspection.current || seekPending || element.seeking || document.hidden) return;
      const snapshot = document.createElement("canvas");
      snapshot.width = displayed.current.image.width; snapshot.height = displayed.current.image.height;
      const pixels = snapshot.getContext("2d");
      if (!pixels) { setNotice("This frame could not be frozen. Try pausing the video."); return; }
      const frozen = freezeVideoFrame(displayed.current, identified.current, image => {
        pixels.drawImage(image, 0, 0);
        return snapshot;
      });
      stopFrames(); session.invalidate(); retryAt = null; retryAttempts = 0;
      inspection.current = frozen; latest.current = frozen;
      identified.current = frozen.result ? { ...frozen, result: frozen.result } : null;
      needsPausedCapture = !frozen.result;
      status.current = "Identifying the frozen displayed frame.";
      element.pause(); setInspecting(true); updatePlayback(); pausedFrame(); paint.current();
    };
    releaseInspection.current = (stayAtCurrentPosition = false) => {
      const frozen = inspection.current;
      if (!frozen || failed) return;
      // A live stream may no longer retain the inspected media position.
      if (stayAtCurrentPosition || live.current || element.duration === Infinity) {
        invalidate("Waiting for identification at this video position.", true); updatePlayback(); return;
      }
      // Resume from the inspected media position, rather than the newer hidden
      // playback position. Seeking clears the still before decoding resumes.
      const time = frozen.frame.timestampMs / 1000;
      if (element.currentTime === time) { invalidate("Waiting for identification of this video frame.", true); updatePlayback(); }
      else { startSeek(); element.currentTime = time; }
    };
    const events: [string, () => void][] = [["loadeddata", loaded], ["canplay", loaded], ["durationchange", updatePlayback], ["timeupdate", updatePlayback],
      ["seeking", startSeek], ["seeked", seeked], ["play", play], ["pause", pause], ["ended", pause],
      ["waiting", buffering], ["stalled", () => { if (!element.paused || element.readyState < 2) buffering(); }], ["error", () => error()]];
    events.forEach(([name, listener]) => element.addEventListener(name, listener));
    document.addEventListener("visibilitychange", visibility);
    terminalError.current = null;
    invalidate(props.src ? "Loading video." : "Choose a video to identify anatomy.", true);
    live.current = false;
    element.playbackRate = 1; setPlaybackRate(1);
    setPlayback({ time: 0, duration: 0, playing: false, ready: false, live: false, failed: false, liveEdge: 0 });
    let detachSource = () => {};
    if (props.src) {
      detachSource = attachVideoElementSource(element, { src: props.src, sourceKind: props.sourceKind ?? "upload", format: props.streamFormat ?? "auto",
        onLive(value) { if (!disposed) { live.current = value; updatePlayback(); } },
        onInterrupted: buffering, onError: error,
      });
      if (!synchronized) animation = requestAnimationFrame(preview);
    }
    const watchdog = window.setInterval(() => {
      if (document.hidden || failed) return;
      if (props.sourceKind === "url" && !element.paused && !awaitingDecodedFrame && synchronized && performance.now() - lastDecodedAt > 1000) buffering();
      if (retryAt !== null && performance.now() >= retryAt) { retryAt = null; needsPausedCapture = true; }
      pausedFrame();
      if (!element.paused && identified.current && performance.now() - identified.current.capturedAt >= MAX_LIVE_FRAME_AGE_MS) {
        identified.current = null; status.current = "Unable to identify. Waiting for a current video result."; paint.current();
      }
    }, 100);
    return () => {
      disposed = true; detachSource(); session.dispose(); clearInterval(watchdog);
      events.forEach(([name, listener]) => element.removeEventListener(name, listener));
      document.removeEventListener("visibilitychange", visibility);
      if (callback) element.cancelVideoFrameCallback(callback);
      if (animation) cancelAnimationFrame(animation);
      element.pause(); element.removeAttribute("src"); element.load();
      latest.current = null; identified.current = null; inspection.current = null; displayed.current = null;
      changeSettings.current = () => {}; prepareSeek.current = () => {}; inspectDisplayed.current = () => {}; releaseInspection.current = () => {};
    };
  }, [props.src, props.sourceKind, props.streamFormat]);
  useLayoutEffect(() => { changeSettings.current(); }, [props.identify, props.visible, props.src, props.sourceKind, props.streamFormat]);

  const seek = (time: number) => {
    if (!video.current || !playback.ready || playback.live || !Number.isFinite(time)) return;
    const destination = Math.max(0, Math.min(playback.duration, time));
    if (video.current.currentTime === destination) { if (inspection.current) releaseInspection.current(true); return; }
    prepareSeek.current(); video.current.currentTime = destination;
    setPlayback(previous => ({ ...previous, time: destination }));
  };
  const togglePlayback = () => {
    const element = video.current;
    if (!element) return;
    if (!element.paused) element.pause();
    else {
      releaseInspection.current();
      if (element.ended && !playback.live) seek(0);
      void element.play().catch(() => { status.current = "Video playback could not start. Try another supported video."; paint.current(); });
    }
  };
  const goLive = () => {
    const element = video.current;
    if (!element || !playback.live || !element.seekable.length || playback.failed) return;
    const last = element.seekable.length - 1;
    const target = Math.max(element.seekable.start(last), element.seekable.end(last) - 0.25);
    if (!Number.isFinite(target) || target <= 0) return;
    if (element.currentTime !== target) { prepareSeek.current(); element.currentTime = target; }
    else releaseInspection.current();
    void element.play().catch(() => { status.current = "Tap Play feed to resume the live stream."; paint.current(); });
  };
  return <>
    <video ref={video} muted playsInline preload="auto" aria-hidden="true" style={{ position: "absolute", width: 1, height: 1, opacity: 0, pointerEvents: "none" }} />
    <canvas ref={canvas} style={{ width: "100%", display: "block" }} role="img" aria-label={`${props.sourceKind === "url" ? "Video URL feed" : "Uploaded video"} with frame-matched anatomy identification`} />
    <div className="uploaded-video-controls">
      <div className="button-row">
        <button type="button" disabled={!props.src || playback.failed} onClick={togglePlayback}>{playback.playing ? playback.live ? "Pause feed" : "Pause video" : playback.live ? "Play feed" : "Play video"}</button>
        {!playback.live && <button type="button" disabled={!playback.ready} onClick={() => { seek(0); void video.current?.play().catch(() => { status.current = "Tap Play video to start playback."; paint.current(); }); }}>Replay video</button>}
        <button type="button" disabled={!playback.ready || inspecting} onClick={() => inspectDisplayed.current()}>{inspecting ? "Frame frozen" : "Inspect displayed frame"}</button>
        {playback.live && playback.liveEdge > 0 && <button type="button" disabled={playback.failed} onClick={goLive}>Go live</button>}
      </div>
      {inspecting && inspection.current && <p className="fine-print">Frozen displayed frame at {formatTime(inspection.current.frame.timestampMs / 1000)}. Play to resume{playback.live ? " the stream; Go live returns to its live edge when available" : " from this position"}.</p>}
      {!playback.live && <label className="uploaded-video-speed">Playback speed <select value={playbackRate} disabled={!playback.ready} onChange={event => {
        const rate = Number(event.currentTarget.value);
        if (video.current && [0.25, 0.5, 1].includes(rate)) { video.current.playbackRate = rate; setPlaybackRate(rate); }
      }}><option value="0.25">0.25×</option><option value="0.5">0.5×</option><option value="1">1×</option></select></label>}
      {playback.live ? <p className="fine-print">Live stream · {formatTime(playback.time)}</p> : <label className="uploaded-video-timeline">Video position · {formatTime(playback.time)} / {formatTime(playback.duration)}
        <input type="range" min="0" max={playback.duration || 0} step="0.01" value={Math.min(playback.time, playback.duration)}
          disabled={!playback.ready || !playback.duration} aria-label="Seek video" aria-valuetext={`${formatTime(playback.time)} of ${formatTime(playback.duration)}`}
          onChange={event => seek(Number(event.currentTarget.value))} />
      </label>}
    </div>
    <p role="status" className="notice">{notice}</p>
  </>;
}
