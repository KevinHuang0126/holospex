import assert from "node:assert/strict";
import test from "node:test";
import { createIdentificationReadinessMonitor } from "../src/input/identificationReadiness";
import type { IdentificationModel } from "../src/input/liveIdentification";

const ready: IdentificationModel = { status: "ready", model: { id: "person1-model", version: "v1" },
  minimumConfidence: 0.5, supportsMinimumConfidence: true, dataset: "Endoscapes-Seg50" };
const flush = () => new Promise<void>(resolve => setImmediate(resolve));
function deferred<T>() {
  let resolve!: (value: T) => void, reject!: (cause?: unknown) => void;
  const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}
function setup() {
  let now = 0;
  const timers = new Map<object, { due: number; run: () => void }>();
  const cancelled: unknown[] = [], events: unknown[] = [];
  const requests: { signal: AbortSignal; task: ReturnType<typeof deferred<IdentificationModel>> }[] = [];
  const monitor = createIdentificationReadinessMonitor({
    load: signal => { const task = deferred<IdentificationModel>(); requests.push({ signal, task }); return task.promise; },
    onChecking: value => events.push(value), onReady: model => events.push(model), onUnavailable: () => events.push("unavailable"),
    scheduleTimeout: (run, delay) => { const id = {}; timers.set(id, { due: now + delay, run }); return id; },
    cancelTimeout: handle => { cancelled.push(handle); timers.delete(handle as object); },
  });
  return { monitor, timers, cancelled, events, requests, advance(duration: number) {
    now += duration;
    for (const [id, timer] of timers) if (timer.due <= now) { timers.delete(id); timer.run(); }
  } };
}

test("only explicit check starts readiness and repeated checks cannot overlap the pending GET", async () => {
  const state = setup();
  await flush(); state.advance(60_000);
  assert.equal(state.requests.length, 0); assert.equal(state.timers.size, 0); assert.deepEqual(state.events, []);
  state.monitor.check(); state.monitor.check(); state.monitor.check();
  assert.equal(state.requests.length, 1); assert.deepEqual(state.events, [true]);
  state.advance(60_000); assert.equal(state.requests.length, 1);
  state.requests[0].task.resolve(ready); await flush();
  assert.deepEqual(state.events, [true, false, ready]); assert.equal(state.timers.size, 0);
  state.advance(60_000); assert.equal(state.requests.length, 1);
  state.monitor.dispose();
});

test("every failure retries after ten seconds with no media prerequisites and recovery stops retries", async () => {
  const state = setup(); state.monitor.check();
  for (let attempt = 0; attempt < 3; attempt++) {
    state.requests[attempt].task.reject(new Error("model host offline")); await flush();
    assert.equal(state.events.at(-1), "unavailable"); assert.equal(state.timers.size, 1);
    state.advance(9999); assert.equal(state.requests.length, attempt + 1);
    state.advance(1); assert.equal(state.requests.length, attempt + 2); assert.equal(state.timers.size, 0);
    state.monitor.check(); assert.equal(state.requests.length, attempt + 2);
  }
  state.requests.at(-1)!.task.resolve(ready); await flush();
  assert.equal(state.events.at(-1), ready); assert.equal(state.timers.size, 0);
  state.advance(60_000); assert.equal(state.requests.length, 4);
  state.monitor.dispose();
});

test("manual or reconnect checks cancel queued retry and ignore an already queued stale timer callback", async () => {
  const state = setup(); state.monitor.check(); state.requests[0].task.reject(); await flush();
  const obsolete = [...state.timers.values()][0].run;
  state.advance(3000); state.monitor.check();
  assert.equal(state.cancelled.length, 1); assert.equal(state.timers.size, 0); assert.equal(state.requests.length, 2);
  obsolete(); state.monitor.check(); assert.equal(state.requests.length, 2);
  state.requests[1].task.resolve(ready); await flush();
  obsolete(); state.advance(60_000); assert.equal(state.requests.length, 2);
  state.monitor.check(); assert.equal(state.requests.length, 3);
  state.requests[2].task.resolve(ready); await flush(); state.monitor.dispose();
});

test("disposal aborts pending readiness and suppresses late resolution or rejection and all future checks", async () => {
  for (const outcome of ["ready", "failed"] as const) {
    const state = setup(); state.monitor.check(); state.monitor.dispose(); state.monitor.dispose();
    assert.equal(state.requests[0].signal.aborted, true);
    const count = state.events.length;
    if (outcome === "ready") state.requests[0].task.resolve(ready); else state.requests[0].task.reject(new Error("late failure"));
    await flush(); state.monitor.check(); state.advance(60_000);
    assert.equal(state.events.length, count); assert.equal(state.requests.length, 1); assert.equal(state.timers.size, 0);
  }
});

test("disposal cancels scheduled retry even if the cancelled timer is invoked later", async () => {
  const state = setup(); state.monitor.check(); state.requests[0].task.reject(); await flush();
  const obsolete = [...state.timers.values()][0].run, count = state.events.length;
  state.monitor.dispose(); assert.equal(state.cancelled.length, 1); assert.equal(state.timers.size, 0);
  obsolete(); state.monitor.check(); state.advance(60_000); await flush();
  assert.equal(state.requests.length, 1); assert.equal(state.events.length, count);
});
