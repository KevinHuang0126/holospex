import { useEffect, useState } from "react";
import type { ImageOverlayAsset } from "./imageOverlayAsset";
import { composeMannequinAsset, type MannequinSamplePlacement } from "./mannequinComposite";

/** Prepared transparent display image only: annotation indices still use the exact PNG decoder. */
export function loadMannequinReference(signal: AbortSignal): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image();
    const detach = () => { image.onload = null; image.onerror = null; signal.removeEventListener("abort", abort); };
    const fail = (cause: Error) => { detach(); image.src = ""; reject(cause); };
    const abort = () => fail(new DOMException("Mannequin loading was cancelled.", "AbortError"));
    if (signal.aborted) { abort(); return; }
    image.onload = () => {
      if (image.naturalWidth !== 894 || image.naturalHeight !== 569) {
        fail(new Error("The mannequin reference has unexpected dimensions.")); return;
      }
      detach(); resolve(image);
    };
    image.onerror = () => fail(new Error("The mannequin image could not load. Reload the page to try again."));
    signal.addEventListener("abort", abort, { once: true });
    image.src = "/demo/mannequin-cutout.png";
  });
}

/** A changed sample or placement must never reuse the previous composite's labels. */
export function useMannequinComposite(sample: ImageOverlayAsset | null, enabled: boolean, placement: MannequinSamplePlacement) {
  const [reference, setReference] = useState<HTMLImageElement | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const { centerX, centerY, widthFraction } = placement;
  const [prepared, setPrepared] = useState<{
    sample: ImageOverlayAsset; reference: HTMLImageElement;
    centerX: number; centerY: number; widthFraction: number;
    asset: ImageOverlayAsset | null; error: string | null;
  } | null>(null);

  useEffect(() => {
    let disposed = false, image: HTMLImageElement | null = null;
    const controller = new AbortController();
    setReference(null); setLoadError(null);
    if (enabled) void loadMannequinReference(controller.signal).then(next => {
      if (disposed) { next.src = ""; return; }
      image = next; setReference(next);
    }).catch(cause => {
      if (!disposed) setLoadError(cause instanceof Error ? cause.message : "Unable to load mannequin reference.");
    });
    return () => { disposed = true; controller.abort(); if (image) image.src = ""; };
  }, [enabled]);

  useEffect(() => {
    setPrepared(null);
    if (!enabled || !sample || !reference) return;
    let asset: ImageOverlayAsset | null = null, error: string | null = null;
    try { asset = composeMannequinAsset(sample, reference, reference.naturalWidth, reference.naturalHeight, { centerX, centerY, widthFraction }); }
    catch (cause) { error = cause instanceof Error ? cause.message : "Unable to prepare mannequin overlay."; }
    setPrepared({ sample, reference, centerX, centerY, widthFraction, asset, error });
    return () => asset?.dispose();
  }, [enabled, sample, reference, centerX, centerY, widthFraction]);

  const current = enabled && prepared?.sample === sample && prepared?.reference === reference
    && prepared?.centerX === centerX && prepared?.centerY === centerY && prepared?.widthFraction === widthFraction ? prepared : null;
  const asset = current?.asset ?? null, error = enabled ? loadError ?? current?.error ?? null : null;
  return { asset, error, loading: enabled && !!sample && !asset && !error };
}
