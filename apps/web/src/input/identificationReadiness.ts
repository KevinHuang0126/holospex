import type { IdentificationModel } from "./liveIdentification";

/** Check readiness without sending pixels. Failed checks retry even before media is selected. */
export function createIdentificationReadinessMonitor(options: {
  load(signal: AbortSignal): Promise<IdentificationModel>;
  onChecking(checking: boolean): void;
  onReady(model: IdentificationModel): void;
  onUnavailable(): void;
  scheduleTimeout?: (callback: () => void, delayMs: number) => unknown;
  cancelTimeout?: (handle: unknown) => void;
}) {
  const schedule = options.scheduleTimeout ?? ((callback, delay) => setTimeout(callback, delay));
  const cancel = options.cancelTimeout ?? (handle => clearTimeout(handle as ReturnType<typeof setTimeout>));
  let disposed = false, generation = 0;
  let active: AbortController | null = null;
  let retry: { handle?: unknown } | null = null;
  const cancelRetry = () => {
    const pending = retry; retry = null;
    if (pending) cancel(pending.handle);
  };
  const check = () => {
    if (disposed) return;
    cancelRetry();
    if (active) return;
    const controller = new AbortController(), epoch = ++generation;
    const current = () => !disposed && generation === epoch;
    active = controller; options.onChecking(true);
    if (!current()) return;
    void (async () => {
      let model: IdentificationModel | undefined, failed = false;
      try { model = await options.load(controller.signal); }
      catch { failed = true; }
      if (!current()) return;
      active = null; options.onChecking(false);
      if (!current()) return;
      if (!failed) { options.onReady(model!); return; }
      options.onUnavailable();
      if (!current()) return;
      const pending: { handle?: unknown } = {}; retry = pending;
      pending.handle = schedule(() => {
        if (disposed || retry !== pending) return;
        retry = null; check();
      }, 10_000);
    })();
  };
  return {
    check,
    dispose() {
      if (disposed) return;
      disposed = true; generation++; cancelRetry(); active?.abort(); active = null;
    },
  };
}
