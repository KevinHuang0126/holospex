import assert from "node:assert/strict";
import test from "node:test";
import { attachVideoElementSource } from "../src/camera/videoElementSource";

const flush = () => new Promise<void>(resolve => setImmediate(resolve));
function media(nativeHls = "") {
  const events: string[] = [];
  const element = {
    set crossOrigin(value: string) { events.push(`cors:${value}`); },
    set src(_: string) { events.push("source"); },
    load() { events.push("load"); },
    removeAttribute(name: string) { events.push(`remove:${name}`); },
    canPlayType() { events.push("native-support"); return nativeHls; },
  } as unknown as HTMLVideoElement;
  return { element, events };
}
function library(supported = true) {
  const calls: string[] = [], instances: FakeHls[] = [];
  class FakeHls {
    static Events = { LEVEL_LOADED: "level", ERROR: "error" };
    static isSupported() { calls.push("mse-support"); return supported; }
    callbacks = new Map<string, (_: string, data: unknown) => void>();
    constructor(public config: { liveDurationInfinity: boolean; xhrSetup(xhr: { withCredentials: boolean }): void;
      fetchSetup(context: { url: string }, init: RequestInit): Request }) { instances.push(this); calls.push("create"); }
    on(name: string, callback: (_: string, data: unknown) => void) { this.callbacks.set(name, callback); }
    emit(name: string, data: unknown) { this.callbacks.get(name)?.(name, data); }
    loadSource() { calls.push("source"); }
    attachMedia() { calls.push("attach"); }
    destroy() { calls.push("destroy"); }
  }
  const module = { default: FakeHls } as unknown as typeof import("hls.js");
  return { module, calls, instances };
}

test("direct URL enables anonymous CORS before media loading; uploads never import HLS", async () => {
  for (const sourceKind of ["upload", "url"] as const) {
    const { element, events } = media(); let imports = 0;
    const dispose = attachVideoElementSource(element, { src: sourceKind === "url" ? "https://video.example/clip.mp4" : "blob:local-video",
      sourceKind, format: "auto", onLive: () => assert.fail(), onError: () => assert.fail(), onInterrupted: () => assert.fail(),
      loadHls: async () => { imports++; return library().module; } });
    assert.deepEqual(events, [sourceKind === "url" ? "cors:anonymous" : "remove:crossorigin", "source", "load"]);
    await flush(); assert.equal(imports, 0); dispose();
  }
});

test("HLS prefers supported MSE, propagates live metadata and closes fatally failed or disposed streams", async () => {
  const { element, events } = media("probably"), hls = library();
  const lives: boolean[] = [], errors: string[] = []; let interruptions = 0;
  const dispose = attachVideoElementSource(element, { src: "https://video.example/live.m3u8?access=private", sourceKind: "url", format: "auto",
    onLive: live => lives.push(live), onError: message => errors.push(message), onInterrupted: () => interruptions++, loadHls: async () => hls.module });
  await flush();
  assert.deepEqual(events, ["cors:anonymous"]);
  assert.deepEqual(hls.calls, ["mse-support", "create", "source", "attach"]);
  const instance = hls.instances[0];
  assert.equal(instance.config.liveDurationInfinity, true);
  const xhr = { withCredentials: true }; instance.config.xhrSetup(xhr); assert.equal(xhr.withCredentials, false);
  assert.equal(instance.config.fetchSetup({ url: "https://video.example/segment.ts" }, { credentials: "include" }).credentials, "omit");
  instance.emit("level", { details: { live: true } }); instance.emit("level", { details: { live: false } });
  assert.deepEqual(lives, [true, false]);
  instance.emit("error", { fatal: false }); assert.equal(interruptions, 1);
  instance.emit("error", { fatal: true, url: "https://video.example/live.m3u8?access=private" });
  assert.equal(errors.length, 1); assert.match(errors[0], /cross-origin/); assert.doesNotMatch(errors[0], /access=|private|example/);
  assert.equal(hls.calls.at(-1), "destroy");
  instance.emit("level", { details: { live: true } }); assert.deepEqual(lives, [true, false]);
  dispose(); instance.emit("error", { fatal: true }); assert.equal(errors.length, 1);
  assert.equal(hls.calls.filter(call => call === "destroy").length, 1);
});

test("native HLS is selected only when MSE is unavailable, and unsupported browsers receive a useful error", async () => {
  for (const native of ["", "probably"]) {
    const { element, events } = media(native), hls = library(false), errors: string[] = [];
    const dispose = attachVideoElementSource(element, { src: "https://video.example/feed", sourceKind: "url", format: "hls",
      onLive: () => {}, onError: message => errors.push(message), onInterrupted: () => {}, loadHls: async () => hls.module });
    await flush();
    assert.deepEqual(hls.calls, ["mse-support"]);
    assert.deepEqual(events, native ? ["cors:anonymous", "native-support", "source", "load"] : ["cors:anonymous", "native-support"]);
    assert.equal(errors.length, native ? 0 : 1);
    if (!native) assert.match(errors[0], /HLS playback is unavailable/);
    dispose();
  }
});

test("removing a source while the HLS chunk loads prevents late attachment and errors", async () => {
  for (const rejected of [false, true]) {
    const { element, events } = media(), hls = library(); let callbacks = 0;
    let resolve!: (module: typeof import("hls.js")) => void, reject!: (error: unknown) => void;
    const pending = new Promise<typeof import("hls.js")>((yes, no) => { resolve = yes; reject = no; });
    const dispose = attachVideoElementSource(element, { src: "https://video.example/feed.m3u8", sourceKind: "url", format: "hls",
      onLive: () => callbacks++, onError: () => callbacks++, onInterrupted: () => callbacks++, loadHls: () => pending });
    dispose();
    if (rejected) reject(new Error("private URL in provider error")); else resolve(hls.module);
    await flush();
    assert.deepEqual(events, ["cors:anonymous"]); assert.deepEqual(hls.calls, []); assert.equal(callbacks, 0);
  }
});
