import type { FrameResult } from "@holospex/contracts";
import type { DisplayedFrame } from "../overlays/selectFrame";
import { IdentificationBusyError, validateLiveResult, type LiveFrameIdentifier } from "./liveIdentification";

// Allow CPU inference and network time while keeping the captured frame's lifetime bounded.
export const MAX_LIVE_FRAME_AGE_MS = 6000;
export interface LiveFramePipelineOptions {
  identify: LiveFrameIdentifier;
  onResult(frame: DisplayedFrame, result: FrameResult, capturedAt: number): void;
  onUnavailable(message: string): void;
  /** capturedAt and now must use the same monotonic clock. */
  now?: () => number;
  scheduleTimeout?: (callback: () => void, delayMs: number) => unknown;
  cancelTimeout?: (handle: unknown) => void;
}

/** One captured frame at a time; an uncooperative timed-out identifier still occupies its slot. */
export function createLiveFramePipeline(options: LiveFramePipelineOptions) {
  const now = options.now ?? (() => performance.now());
  const schedule = options.scheduleTimeout ?? ((callback, delayMs) => setTimeout(callback, delayMs));
  const cancel = options.cancelTimeout ?? (handle => clearTimeout(handle as ReturnType<typeof setTimeout>));
  let disposed = false;
  let retryAt = -Infinity;
  let active: { controller: AbortController; expired: boolean; timer?: unknown } | null = null;

  return {
    get busy() { return active !== null || !disposed && now() < retryAt; },
    submit(frame: DisplayedFrame, image: () => Promise<Blob>, capturedAt: number): boolean {
      if (disposed || active || now() < retryAt) return false;
      const age = now() - capturedAt;
      if (!Number.isFinite(capturedAt) || !Number.isFinite(age) || age < 0 || age >= MAX_LIVE_FRAME_AGE_MS) {
        options.onUnavailable("Unable to identify. The captured camera frame is outdated.");
        return false;
      }
      const capturedFrame = Object.freeze({ ...frame });
      const job = { controller: new AbortController(), expired: false, timer: undefined as unknown };
      active = job;
      const expire = () => {
        if (disposed || job.expired || active !== job) return;
        job.expired = true; job.controller.abort();
        options.onUnavailable("Unable to identify. Identification timed out.");
      };
      const current = () => {
        if (disposed || job.expired || active !== job) return false;
        const elapsed = now() - capturedAt;
        if (!Number.isFinite(elapsed) || elapsed < 0 || elapsed >= MAX_LIVE_FRAME_AGE_MS) { expire(); return false; }
        return true;
      };
      job.timer = schedule(expire, Math.ceil(MAX_LIVE_FRAME_AGE_MS - age));
      void (async () => {
        let phase: "encoding" | "identification" | "validation" = "encoding";
        let result: FrameResult | undefined;
        let failure: string | undefined;
        try {
          const blob = await image();
          if (!current()) return;
          phase = "identification";
          const value = await options.identify(capturedFrame, blob, job.controller.signal);
          if (!current()) return;
          phase = "validation";
          result = validateLiveResult(value, capturedFrame);
          if (!current()) result = undefined;
        } catch (cause) {
          if (current()) {
            if (phase === "identification" && cause instanceof IdentificationBusyError) {
              retryAt = now() + cause.retryAfterMs;
              failure = "Identification is busy.";
            } else failure = phase === "encoding" ? "Unable to identify. Camera frame could not be prepared."
              : phase === "validation" ? "Unable to identify. Identification returned an invalid or mismatched frame."
                : "Unable to identify. Identification failed.";
          }
        } finally {
          cancel(job.timer);
          if (active === job) active = null;
        }
        if (disposed || job.expired) return;
        if (result) options.onResult(capturedFrame, result, capturedAt);
        else if (failure) options.onUnavailable(failure);
      })();
      return true;
    },
    dispose() {
      if (disposed) return;
      disposed = true;
      if (active) { cancel(active.timer); active.controller.abort(); }
    },
  };
}
