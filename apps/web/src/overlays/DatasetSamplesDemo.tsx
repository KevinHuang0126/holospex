import { useCallback, useEffect, useRef, useState } from "react";
import { anatomy } from "@holospex/contracts";
import { DatasetSampleHud, type DatasetSelection } from "./DatasetSampleHud";
import { datasetCredit, importDatasetSamples, type DatasetImport } from "./datasetSamples";
import { datasetOverlayState, type DatasetView } from "./datasetRaster";
import { loadLocalDatasetSamples } from "./localSampleInput";
import type { HudMode } from "./hudControls";
import type { DisplayedFrame } from "./selectFrame";
import "../camera/deviceSetup.css";
import "./sampleTest.css";

/** Local integration harness; Person 3 can embed DatasetSampleHud with their own controls. */
export function DatasetSamplesDemo({ autoLoadLocal = false }: { autoLoadLocal?: boolean }) {
  const [pack, setPack] = useState<DatasetImport>({ samples: [], issues: [] });
  const [selected, setSelected] = useState("");
  const [mode, setMode] = useState<HudMode>("learn"), [visible, setVisible] = useState(true);
  const [view, setView] = useState<DatasetView>("raster");
  const [opacity, setOpacity] = useState(28);
  const [showBoundaries, setShowBoundaries] = useState(true), [showLabels, setShowLabels] = useState(true);
  const [loading, setLoading] = useState(false), [error, setError] = useState<string | null>(null);
  const [frame, setFrame] = useState<DisplayedFrame | null>(null);
  const [selection, setSelection] = useState<DatasetSelection | null>(null);
  const generation = useRef(0);
  const pending = useRef<AbortController | null>(null);
  const loadPack = useCallback(async (read: (signal: AbortSignal) => Promise<DatasetImport>) => {
    pending.current?.abort();
    const controller = new AbortController(); pending.current = controller;
    const request = ++generation.current;
    setPack({ samples: [], issues: [] }); setSelected(""); setFrame(null); setSelection(null); setError(null); setLoading(true);
    try {
      const imported = await read(controller.signal);
      if (request !== generation.current) return;
      setPack(imported); setSelected(imported.samples[0]?.id ?? "");
    } catch (cause) {
      if (request === generation.current) setError(cause instanceof Error ? cause.message : "Unable to import samples.");
    } finally { if (request === generation.current) setLoading(false); }
  }, []);
  useEffect(() => {
    if (autoLoadLocal) void loadPack(loadLocalDatasetSamples);
    return () => { generation.current += 1; pending.current?.abort(); };
  }, [autoLoadLocal, loadPack]);
  function load(files: FileList | null) {
    if (!files?.length) return;
    const inputs = Array.from(files);
    void loadPack(() => importDatasetSamples(inputs));
  }
  function chooseCase(id: string) {
    if (id === selected) return;
    setFrame(null); setSelection(null); setSelected(id);
  }
  const sample = pack.samples.find(item => item.id === selected) ?? null;
  const caseIndex = pack.samples.findIndex(item => item.id === selected);
  const overlaysShown = !!frame && datasetOverlayState(visible, mode, view).show && (opacity > 0 || showBoundaries || showLabels);
  return <section className="device-setup" aria-labelledby="samples-title">
    <div className="sample-heading"><div><p className="eyebrow">AR / HUD · React test page</p><h2 id="samples-title">Test anatomy overlays</h2></div>
      {autoLoadLocal && <button disabled={loading} onClick={() => void loadPack(loadLocalDatasetSamples)}>Reload local samples</button>}
    </div>
    <p className="sample-intro">Inspect the supplied masks on the original images. Adjust the fill, boundaries and labels, or hide all overlays to compare alignment. Identify and Assess hide every anatomical layer.</p>
    <details className="sample-import" open={!autoLoadLocal || !!error}><summary>Choose a different sample folder</summary>
    <div className="setup-row">
      <label className="setup-field">Open sample folder<input type="file" multiple ref={node => { node?.setAttribute("webkitdirectory", ""); }} onChange={event => { load(event.target.files); event.target.value = ""; }} /></label>
      <label className="setup-field">Or select the sample files together<input type="file" multiple onChange={event => { load(event.target.files); event.target.value = ""; }} /></label>
    </div>
    <p className="fine-print">Select apps/web/tests/samples_tst. Files stay in this browser; original files are unchanged. Images and masks are paired using the hashes supplied with their labels.</p>
    </details>
    {loading && <p role="status" className="notice">Checking the sample pack…</p>}
    {error && <p role="alert" className="error">{error}</p>}
    {pack.issues.length > 0 && <div className="notice" role="status"><strong>Some sample data is unavailable</strong><ul>{pack.issues.map((issue, index) => <li key={index}>{issue}</li>)}</ul></div>}
    {pack.samples.length > 0 && <>
      <div className="setup-row">
        <button className="secondary" disabled={caseIndex <= 0} onClick={() => chooseCase(pack.samples[caseIndex - 1].id)}>Previous case</button>
        <label className="setup-field">Case <select value={selected} onChange={event => chooseCase(event.target.value)}>{pack.samples.map(item => <option value={item.id} key={item.id}>Case {item.labels.annotationSource.videoId} · source frame {item.labels.annotationSource.sourceFrameNumber}</option>)}</select></label>
        <button className="secondary" disabled={caseIndex >= pack.samples.length - 1} onClick={() => chooseCase(pack.samples[caseIndex + 1].id)}>Next case</button>
        <label className="setup-field">Learning mode <select value={mode} onChange={event => { setSelection(null); setMode(event.target.value as HudMode); }}><option value="learn">Learn</option><option value="identify">Identify</option><option value="assess">Assess</option><option value="feedback">Feedback</option></select></label>
        <label>Overlay <select value={view} onChange={event => setView(event.target.value as DatasetView)}><option value="raster">Exact mask</option><option value="polygons">Approximate polygons</option></select></label>
        <label className="setup-check"><input type="checkbox" checked={visible} onChange={event => setVisible(event.target.checked)} />Show overlays</label>
      </div>
      <fieldset className="overlay-layer-controls" disabled={!frame || mode !== "learn"}>
        <legend>Overlay appearance</legend>
        <label className="overlay-opacity">Fill opacity <output>{opacity}%</output><input aria-label="Overlay fill opacity" type="range" min="0" max="100" step="1" value={opacity} onChange={event => setOpacity(Number(event.target.value))} /></label>
        <label className="setup-check"><input type="checkbox" checked={showBoundaries} onChange={event => setShowBoundaries(event.target.checked)} />Boundaries</label>
        <label className="setup-check"><input type="checkbox" checked={showLabels} onChange={event => setShowLabels(event.target.checked)} />Labels and pointers</label>
        <button className="secondary" onClick={() => { setOpacity(0); setShowBoundaries(true); setShowLabels(false); setVisible(true); }}>Boundaries only</button>
        <button className="secondary" onClick={() => { setOpacity(28); setShowBoundaries(true); setShowLabels(true); setVisible(true); setView("raster"); }}>Reset overlay</button>
      </fieldset>
      <div className="sample-status" role="status"><span>{frame ? `Case ${caseIndex + 1} of ${pack.samples.length} ready` : "Checking image and mask…"}</span><span>{overlaysShown ? "Overlay visible" : "Overlay hidden"}</span><span>Supplied dataset annotation</span></div>
      <DatasetSampleHud sample={sample} mode={mode} visible={visible} view={view}
        appearance={{ fillOpacity: opacity / 100, showBoundaries, showLabels }}
        onDisplayedFrame={next => { setFrame(next); if (!next) setSelection(null); }} onSelection={setSelection} />
      <div className="sample-readout">
        <div><h3>Displayed frame</h3><p>{frame ? `${frame.mediaId} · ${frame.width} × ${frame.height} · frame ${frame.frameNumber} · ${frame.timestampMs} ms` : "No validated still is displayed."}</p></div>
        <div><h3>Last image selection</h3><p>{selection ? `Point ${selection.point.map(value => Math.floor(value)).join(", ")}. ` : "Click a point inside the image."}
          {selection && (overlaysShown && showLabels ? (selection.structureId ? anatomy[selection.structureId].label : selection.status === "ignored" ? "Unlabelled / ignored pixel" : "Background") : "Selection received; anatomy identity hidden.")}</p></div>
      </div>
      <p className="fine-print">These are independent still images. Selections test the renderer connection and do not produce lesson scores.</p>
      {sample && view === "polygons" && mode === "learn" && visible && <p className="fine-print">{sample.labels.conversion.geometry} {sample.labels.conversion.visibility} Omitted components: {Object.entries(sample.labels.conversion.withheldComponents).map(([reason, count]) => `${reason}: ${count}`).join(", ")}.</p>}
    </>}
    {!pack.samples.length && !loading && !error && <p className="notice">Choose the sample folder above to begin.</p>}
    <p className="fine-print"><a href={datasetCredit.sourceUrl} target="_blank" rel="noreferrer">{datasetCredit.label}</a> · <a href={datasetCredit.licenseUrl} target="_blank" rel="noreferrer">{datasetCredit.license}</a>. Original images unchanged; index masks remapped by Person 1 and colored for this viewer. Ignored pixels remain unlabelled. No model confidence or clinical review is implied.</p>
  </section>;
}
