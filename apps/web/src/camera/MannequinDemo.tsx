import { useEffect, useMemo, useRef, useState } from "react";
import { useCamera } from "../input/useCamera";
import type { HudMode } from "../overlays/hudControls";
import { ModelHud } from "./ModelHud";
import { ImageOverlayPreview } from "./ImageOverlayPreview";
import type { ImageOverlayView } from "./imageOverlayAsset";
import { useCameraSample } from "./useCameraSample";
import { useMannequinComposite } from "./useMannequinComposite";
import { useSamplePlacement } from "./useSamplePlacement";
import { datasetCredit } from "../overlays/datasetSamples";
import { HudRecorder } from "./HudRecorder";
import { parseModelRegistration, type ModelRegistration } from "./modelRegistration";
import { createMarkerTestRegistration, downloadLocalFile, printableMarkerSvg } from "./markerSetup";
import "./deviceSetup.css";
import "./mannequin.css";

/** Standalone camera input/setup; the learning app owns mode and visibility. */
export function MannequinDemo({ mode, visible }: { mode: HudMode; visible: boolean }) {
  const camera = useCamera();
  const [deviceId, setDeviceId] = useState("");
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [size, setSize] = useState<{ width: number; height: number } | null>(null);
  const [setup, setSetup] = useState<"image" | "test" | "model">("image");
  const sample = useCameraSample();
  const [view, setView] = useState<ImageOverlayView>("scene");
  const [placement, setPlacement] = useState<"screen" | "table">("screen");
  const [opacity, setOpacity] = useState(100);
  const [imageSize, setImageSize] = useState(80);
  const [planeWidth, setPlaneWidth] = useState(240);
  const [composition, setComposition] = useState<"sample" | "mannequin">("sample");
  const anatomyPlacement = useSamplePlacement(sample.asset, setup === "image" && composition === "mannequin");
  const samplePlacement = anatomyPlacement.placement;
  const setSamplePlacement = anatomyPlacement.change;
  const [mannequinWidth, setMannequinWidth] = useState(600);
  const composite = useMannequinComposite(sample.asset, setup === "image" && composition === "mannequin", samplePlacement);
  const displayedAsset = composition === "mannequin" ? composite.asset : sample.asset;
  const [markerSize, setMarkerSize] = useState("80");
  const [model, setModel] = useState<ModelRegistration | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [canvas, setCanvas] = useState<HTMLCanvasElement | null>(null);
  const request = useRef(0);
  const presetRegistration = useMemo(() => {
    const mm = Number(markerSize);
    return Number.isFinite(mm) && mm > 0 ? createMarkerTestRegistration(size?.width, size?.height, mm) : null;
  }, [markerSize, size?.width, size?.height]);
  const registration = setup === "model" ? model : presetRegistration;
  const marker = useMemo(() => registration ? printableMarkerSvg(registration) : null, [registration]);
  const imageOverlay = setup === "image" && displayedAsset ? { asset: displayedAsset, view, placement,
    opacity: opacity / 100, widthFraction: imageSize / 100, planeWidthMm: composition === "mannequin" ? mannequinWidth : planeWidth } : null;

  useEffect(() => {
    let disposed = false;
    const refresh = async () => {
      try {
        const items = await navigator.mediaDevices?.enumerateDevices();
        if (!disposed) setDevices(items?.filter(item => item.kind === "videoinput") ?? []);
      } catch { /* Camera permission errors are reported by useCamera on Start. */ }
    };
    void refresh(); navigator.mediaDevices?.addEventListener("devicechange", refresh);
    return () => { disposed = true; navigator.mediaDevices?.removeEventListener("devicechange", refresh); };
  }, [camera.stream]);
  useEffect(() => () => { request.current += 1; }, []);

  function changeSetup(next: "image" | "test" | "model") {
    request.current += 1; setLoading(false); setError(null); setSetup(next);
  }
  async function loadConfiguration(file: File | undefined) {
    const generation = ++request.current;
    setModel(null); setSetup("model"); setError(null); setLoading(!!file);
    if (!file) return;
    try {
      const text = await file.text();
      if (generation === request.current) setModel(parseModelRegistration(JSON.parse(text)));
    } catch (cause) {
      if (generation === request.current) setError(cause instanceof Error ? cause.message : "Invalid model configuration.");
    } finally { if (generation === request.current) setLoading(false); }
  }

  return <div className="mannequin-demo">
    <div className="setup-row" role="group" aria-label="Overlay source">
      <button aria-pressed={setup === "image"} onClick={() => changeSetup("image")}>Labeled image</button>
      <button aria-pressed={setup === "test"} onClick={() => changeSetup("test")}>Marker test</button>
      <button aria-pressed={setup === "model"} onClick={() => changeSetup("model")}>Mannequin configuration</button>
    </div>
    {setup === "image" && <section className="camera-image-input" aria-labelledby="camera-image-title">
      <h3 id="camera-image-title">Choose labeled training imagery</h3>
      <p>The original surgical image supplies the tissue detail. Its matching mask supplies the boundaries and label positions.</p>
      <div className="setup-row">
        <label className="setup-field">Open camera sample or sample files<input type="file" multiple onChange={event => { sample.loadFiles(event.target.files); event.target.value = ""; }} /></label>
        <label className="setup-field">Open sample folder<input type="file" multiple ref={node => { node?.setAttribute("webkitdirectory", ""); }} onChange={event => { sample.loadFiles(event.target.files); event.target.value = ""; }} /></label>
        {import.meta.env.MODE === "samples" && <button className="secondary" disabled={sample.loading} onClick={sample.reload}>Reload training images</button>}
      </div>
      <p className="fine-print">Open a prepared training reference folder, or transfer its .holospex.json file to your phone. The image and exact annotation mask stay together. Images stay in this browser.</p>
      {sample.loading && <p className="notice" role="status">Preparing the image and its exact anatomy mask…</p>}
      {sample.error && <p className="error" role="alert">{sample.error}</p>}
      {composite.loading && <p className="notice" role="status">Preparing the mannequin and sample composite…</p>}
      {composite.error && <p className="error" role="alert">{composite.error}</p>}
      {sample.pack.issues.length > 0 && <div className="notice" role="status"><ul>{sample.pack.issues.map((issue, index) => <li key={index}>{issue}</li>)}</ul></div>}
      <div className="setup-row">
        {sample.pack.samples.length > 0 && <label className="setup-field">Sample<select value={sample.selected} onChange={event => sample.select(event.target.value)}>{sample.pack.samples.map(item => <option key={item.id} value={item.id}>Case {item.labels.annotationSource.videoId} · frame {item.labels.annotationSource.sourceFrameNumber}</option>)}</select></label>}
        <label className="setup-field">Scene<select value={composition} onChange={event => {
          const next = event.target.value as "sample" | "mannequin";
          setComposition(next); if (next === "mannequin") { setPlacement("table"); setView("anatomy"); }
        }}><option value="sample">Sample only</option><option value="mannequin">Mannequin + sample</option></select></label>
        <label className="setup-field">Image view<select value={view} onChange={event => setView(event.target.value as ImageOverlayView)}><option value="scene">Full surgical image</option><option value="anatomy">Anatomy cutout</option></select></label>
        <label className="setup-field">Place image<select value={placement} onChange={event => setPlacement(event.target.value as "screen" | "table")}><option value="screen">Camera screen</option><option value="table">Table marker</option></select></label>
        <label className="setup-field">Image opacity · {opacity}%<input aria-label="Camera image opacity" type="range" min="0" max="100" value={opacity} onChange={event => setOpacity(Number(event.target.value))} /></label>
        {placement === "screen"
          ? <label className="setup-field">Image size · {imageSize}%<input aria-label="Camera image size" type="range" min="30" max="95" value={imageSize} onChange={event => setImageSize(Number(event.target.value))} /></label>
          : composition === "mannequin"
            ? <label className="setup-field">Mannequin width on table · {mannequinWidth} mm<input aria-label="Mannequin width on table" type="range" min="240" max="1200" step="20" value={mannequinWidth} onChange={event => setMannequinWidth(Number(event.target.value))} /></label>
            : <label className="setup-field">Image width on table · {planeWidth} mm<input aria-label="Image width on table" type="range" min="120" max="400" step="10" value={planeWidth} onChange={event => setPlaneWidth(Number(event.target.value))} /></label>}
      </div>
      {composition === "mannequin" && <fieldset className="mannequin-composite-controls">
        <legend>Position the sample on the mannequin</legend>
        <p>Fit the labeled anatomy to a body region, then fine-tune it here. The image, boundaries and labels stay together.</p>
        <div className="setup-row">
          <button disabled={!sample.asset?.regions?.length || anatomyPlacement.pending || mode !== "learn"} onClick={() => { void anatomyPlacement.suggest(); }}>
            {anatomyPlacement.pending ? "Finding anatomy placement..." : "Fit anatomy with AI"}
          </button>
          <button className="secondary" onClick={anatomyPlacement.reset}>Reset placement</button>
        </div>
        <p className="fine-print">Fit anatomy sends class names and mask measurements to the placement API. It keeps the surgical image and camera feed on your device.</p>
        {anatomyPlacement.error && <p className="error" role="alert">{anatomyPlacement.error}</p>}
        {anatomyPlacement.pending && <p className="notice" role="status">Requesting an anatomy-aware position and scale...</p>}
        {mode === "learn" && anatomyPlacement.suggestion && <p className="notice" role="status">
          {anatomyPlacement.manual ? "Manually adjusted AI suggestion" : "AI-suggested placement"}: {anatomyPlacement.suggestion.focusStructureId.replaceAll("_", " ")}
          {" → "}{anatomyPlacement.suggestion.targetRegion.replaceAll("_", " ")}. {anatomyPlacement.suggestion.reason}
        </p>}
        <div className="setup-row">
          <label className="setup-field">Left / right · {Math.round(samplePlacement.centerX * 100)}%<input aria-label="Sample horizontal position" type="range" min="0" max="100" step="0.5" value={samplePlacement.centerX * 100} onChange={event => setSamplePlacement(previous => ({ ...previous, centerX: Number(event.target.value) / 100 }))} /></label>
          <label className="setup-field">Up / down · {Math.round(samplePlacement.centerY * 100)}%<input aria-label="Sample vertical position" type="range" min="0" max="100" step="0.5" value={samplePlacement.centerY * 100} onChange={event => setSamplePlacement(previous => ({ ...previous, centerY: Number(event.target.value) / 100 }))} /></label>
          <label className="setup-field">Sample width · {(samplePlacement.widthFraction * 100).toFixed(1)}%<input aria-label="Sample width on mannequin" type="range" min="0.1" max="50" step="0.1" value={samplePlacement.widthFraction * 100} onChange={event => setSamplePlacement(previous => ({ ...previous, widthFraction: Number(event.target.value) / 100 }))} /></label>
        </div>
        <p className="fine-print">Placement is an illustration on this mannequin reference, not measured anatomical registration. Check the preview before anchoring it to the marker.</p>
      </fieldset>}
      <p className="notice">{placement === "screen" ? "The image stays fixed on your camera screen. No marker is needed." : "The image lies flat beyond the printed marker on the table. Keep the marker in view."} The image and its labels move together.</p>
    </section>}
    <div className="setup-row mannequin-camera-controls">
      <label className="setup-field">Camera<select value={deviceId} onChange={event => { camera.stop(); setSize(null); setDeviceId(event.target.value); }}>
        <option value="">Rear / default camera</option>
        {devices.map((device, index) => <option key={device.deviceId || index} value={device.deviceId}>{device.label || `Camera ${index + 1}`}</option>)}
      </select></label>
      <button disabled={camera.starting || !!camera.stream} onClick={() => { void camera.start(deviceId || undefined); }}>{camera.starting ? "Starting camera…" : "Start camera"}</button>
      <button className="secondary" disabled={!camera.stream && !camera.starting} onClick={() => { camera.stop(); setSize(null); }}>Stop camera</button>
      {size && camera.stream && <span className="fine-print">Camera: {size.width} × {size.height}</span>}
    </div>
    {camera.error && <p className="error" role="alert">{camera.error}</p>}
    <div className="mannequin-preview">
      {!camera.stream && setup === "image" ? <ImageOverlayPreview asset={displayedAsset} view={view} visible={visible && opacity > 0} mode={mode} onCanvas={setCanvas} />
        : <ModelHud stream={camera.stream} registration={setup === "image" && !imageOverlay ? null : registration} visible={visible} mode={mode} imageOverlay={imageOverlay}
          interrupted={camera.interrupted} onCanvas={setCanvas} onCameraSize={setSize} />}
    </div>
    {(mode === "identify" || mode === "assess") && <p className="notice">The image, model and answer overlays are hidden in this learning mode. {setup === "image" ? "Switch to Learn to reveal them." : "Switch to Learn or Feedback to reveal them."}</p>}

    {setup === "image" && mode === "feedback" && <p className="notice">Unable to assess: reviewed feedback has not been supplied for these dataset images. Switch to Learn to inspect the annotations.</p>}

    {(setup !== "image" || placement === "table") && <section className="mannequin-setup" aria-labelledby="model-setup-title">
      <div>
        <h3 id="model-setup-title">Anchor the overlay</h3>
        {setup !== "model" ? <>
          {setup === "image"
            ? <p>Place the printed marker flat on a table with its top pointing away from you. {composition === "mannequin" ? "The mannequin, sample and labels appear together just beyond its top edge." : "The labeled surgical image appears just beyond its top edge."} Move back enough to keep both in view.</p>
            : <p>Print the marker, attach it to a firm surface, and keep it visible to the camera. Three test points follow its center, right and top as you move the camera.</p>}
          <label className="setup-field mannequin-marker-size">Printed black square (mm)<input type="number" min="1" step="1" value={markerSize} onChange={event => setMarkerSize(event.target.value)} /></label>
          {!presetRegistration && <p className="error" role="alert">Enter a positive marker size.</p>}
          <p className="notice">{setup === "image" ? "Virtual image placement with approximate camera calibration. This is a flat image surface; it does not reconstruct 3D anatomy or detect the table." : "Synthetic tracking test: these points are not anatomical locations. Camera calibration is approximate."}</p>
        </> : <>
          <p>Attach the marker rigidly to the mannequin at the position used for your measurements. Load the camera calibration and marker-relative anatomical locations.</p>
          <label className="setup-field">Mannequin configuration (JSON)<input type="file" accept="application/json,.json" onChange={event => { void loadConfiguration(event.target.files?.[0]); event.target.value = ""; }} /></label>
          {loading && <p role="status">Loading configuration…</p>}
          {model && <p className="notice">{model.modelId} · {model.anchors.length} locations · {model.provenance === "synthetic_mock" ? "Synthetic fixture" : "Supplied model measurements"}</p>}
        </>}
        {error && <p className="error" role="alert">{error}</p>}
      </div>
      <div className="mannequin-marker">
        {marker && registration ? <>
          <img src={`data:image/svg+xml,${encodeURIComponent(marker)}`} alt={`Printable tracking marker ${registration.markerId}`} width="220" height="220" />
          <strong>Marker {registration.markerId} · {registration.markerSizeMm} mm black square</strong>
          <button className="secondary" onClick={() => downloadLocalFile(marker, `holospex-marker-${registration.markerId}.svg`, "image/svg+xml")}>Download printable marker</button>
          <p>Print at 100% scale. Measure the black square and leave the white border intact. The preview on this page is scaled to fit.</p>
        </> : <p>Load a valid configuration to get its matching marker.</p>}
      </div>
    </section>}

    <HudRecorder canvas={canvas} name="model" />
    {setup === "image" && <p className="fine-print"><a href={datasetCredit.sourceUrl} target="_blank" rel="noreferrer">{datasetCredit.label}</a> · <a href={datasetCredit.licenseUrl} target="_blank" rel="noreferrer">{datasetCredit.license}</a>. Supplied dataset annotations with colored contours and optional cutout. The placement is virtual; the image retains its dataset provenance. No live anatomy recognition is running.</p>}
    <p className="fine-print">Camera processing and recordings stay in this browser. In table mode, losing the marker removes the image and labels. Camera interruption removes overlays in both placement modes.</p>
  </div>;
}
