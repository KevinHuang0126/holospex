import { useCallback, useEffect, useRef, useState } from "react";
import { importDatasetSamples, type DatasetImport } from "../overlays/datasetSamples";
import { loadLocalDatasetSamples } from "../overlays/localSampleInput";
import { importPortableSample } from "../overlays/portableSample";
import { loadImageOverlayAsset, type ImageOverlayAsset } from "./imageOverlayAsset";

/** Local files only. Replacing a case withholds the previous texture immediately. */
export function useCameraSample() {
  const [pack, setPack] = useState<DatasetImport>({ samples: [], issues: [] });
  const [selected, setSelected] = useState("");
  const [decoded, setDecoded] = useState<{ sample: DatasetImport["samples"][number]; asset: ImageOverlayAsset } | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const request = useRef(0), pending = useRef<AbortController | null>(null);
  const sample = pack.samples.find(item => item.id === selected) ?? null;
  const asset = decoded?.sample === sample ? decoded.asset : null;
  const load = useCallback(async (read: (signal: AbortSignal) => Promise<DatasetImport>) => {
    pending.current?.abort();
    const controller = new AbortController(); pending.current = controller;
    const generation = ++request.current;
    setPack({ samples: [], issues: [] }); setSelected(""); setError(null); setLoading(true);
    try {
      const next = await read(controller.signal);
      if (generation !== request.current) return;
      setPack(next); setSelected(next.samples[0]?.id ?? "");
    } catch (cause) {
      if (generation === request.current) setError(cause instanceof Error ? cause.message : "Unable to load sample.");
    } finally { if (generation === request.current) setLoading(false); }
  }, []);
  useEffect(() => {
    if (import.meta.env.MODE === "samples") void load(loadLocalDatasetSamples);
    return () => { request.current += 1; pending.current?.abort(); };
  }, [load]);
  useEffect(() => {
    let disposed = false, resource: ImageOverlayAsset | null = null;
    const controller = new AbortController();
    setDecoded(null);
    if (sample) {
      setError(null);
      void loadImageOverlayAsset(sample, controller.signal).then(next => {
        if (disposed) { next.dispose(); return; }
        resource = next; setDecoded({ sample, asset: next });
      }).catch(cause => {
        if (!disposed) setError(cause instanceof Error ? cause.message : "Unable to decode the image and mask.");
      });
    }
    return () => { disposed = true; controller.abort(); resource?.dispose(); };
  }, [sample]);
  function loadFiles(files: FileList | null) {
    if (!files?.length) return;
    const inputs = Array.from(files);
    void load(() => inputs.length === 1 && inputs[0].name.toLowerCase().endsWith(".holospex.json")
      ? importPortableSample(inputs[0]) : importDatasetSamples(inputs));
  }
  return { pack, selected, select: setSelected, asset, loading: loading || (!!sample && !asset && !error),
    error, loadFiles, reload: () => void load(loadLocalDatasetSamples) };
}
