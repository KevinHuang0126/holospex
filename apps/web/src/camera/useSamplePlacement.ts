import { useCallback, useLayoutEffect, useRef, useState } from "react";
import type { ImageOverlayAsset } from "./imageOverlayAsset";
import { DEFAULT_MANNEQUIN_SAMPLE_PLACEMENT, type MannequinSamplePlacement } from "./mannequinComposite";
import { placementFromSuggestion, placementRequestFor, requestSamplePlacement } from "./samplePlacement";
import { TEMPLATE_WIDTH, TEMPLATE_HEIGHT, type PlacementResponse } from "../../../../shared/placement";

type State = { asset: ImageOverlayAsset | null; placement: MannequinSamplePlacement; pending: boolean;
  suggestion: PlacementResponse | null; manual: boolean; error: string | null };
const initial = (asset: ImageOverlayAsset | null): State => ({ asset, placement: { ...DEFAULT_MANNEQUIN_SAMPLE_PLACEMENT },
  pending: false, suggestion: null, manual: true, error: null });

/** Cancel and invalidate requests on sample/source changes, manual edits and unmount. */
export function useSamplePlacement(asset: ImageOverlayAsset | null, enabled: boolean) {
  const [state, setState] = useState<State>(() => initial(asset));
  const generation = useRef(0), controller = useRef<AbortController | null>(null);
  const invalidate = useCallback(() => { generation.current++; controller.current?.abort(); controller.current = null; }, []);
  useLayoutEffect(() => {
    invalidate(); setState(initial(asset));
    return invalidate;
  }, [asset, enabled, invalidate]);
  const current = state.asset === asset ? state : initial(asset);
  const change = (update: (placement: MannequinSamplePlacement) => MannequinSamplePlacement) => {
    invalidate();
    setState(previous => ({ ...(previous.asset === asset ? previous : initial(asset)),
      placement: update(previous.asset === asset ? previous.placement : DEFAULT_MANNEQUIN_SAMPLE_PLACEMENT),
      pending: false, manual: true, error: null }));
  };
  const suggest = async () => {
    if (!asset || !enabled || current.pending) return;
    invalidate();
    const attempt = generation.current, abort = new AbortController(); controller.current = abort;
    const timeout = window.setTimeout(() => abort.abort(), 35_000);
    setState(previous => ({ ...(previous.asset === asset ? previous : initial(asset)), pending: true, error: null }));
    try {
      const request = placementRequestFor(asset);
      const suggestion = await requestSamplePlacement(request, abort.signal);
      const placement = placementFromSuggestion(request, suggestion, TEMPLATE_WIDTH, TEMPLATE_HEIGHT);
      if (generation.current === attempt) setState({ asset, placement, suggestion, pending: false, manual: false, error: null });
    } catch (cause) {
      if (generation.current === attempt) setState(previous => ({ ...previous, pending: false,
        error: abort.signal.aborted ? "The placement request timed out. Try again or adjust manually."
          : cause instanceof Error ? cause.message : "The placement request failed." }));
    } finally {
      window.clearTimeout(timeout);
      if (generation.current === attempt) controller.current = null;
    }
  };
  return { ...current, change, suggest, reset: () => { invalidate(); setState(initial(asset)); } };
}
