import assert from "node:assert/strict";
import test from "node:test";
import { loadMannequinReference } from "../src/camera/useMannequinComposite";

function mockImages() {
  const previous = Object.getOwnPropertyDescriptor(globalThis, "Image");
  const instances: MockImage[] = [];
  class MockImage {
    naturalWidth = 894;
    naturalHeight = 569;
    onload: (() => void) | null = null;
    onerror: (() => void) | null = null;
    requests: string[] = [];
    private value = "";
    constructor() { instances.push(this); }
    set src(value: string) { this.value = value; this.requests.push(value); }
    get src() { return this.value; }
  }
  Object.defineProperty(globalThis, "Image", { configurable: true, value: MockImage });
  return { instances, restore: () => {
    if (previous) Object.defineProperty(globalThis, "Image", previous);
    else Reflect.deleteProperty(globalThis, "Image");
  } };
}

test("the mannequin loader accepts the native reference and detaches handlers after success", async () => {
  const images = mockImages();
  try {
    const controller = new AbortController();
    const loaded = loadMannequinReference(controller.signal);
    const image = images.instances[0];
    assert.deepEqual(image.requests, ["/demo/mannequin-cutout.png"]);
    image.onload?.();
    assert.equal(await loaded, image);
    assert.equal(image.onload, null); assert.equal(image.onerror, null);
    // After success the hook owns the resource. A detached loader cannot clear
    // the image later when that request's signal is aborted.
    controller.abort();
    assert.equal(image.src, "/demo/mannequin-cutout.png");
  } finally { images.restore(); }
});

test("cancelling mannequin loading clears the image and detaches callbacks", async () => {
  const images = mockImages();
  try {
    const controller = new AbortController();
    const loaded = loadMannequinReference(controller.signal);
    const image = images.instances[0];
    const rejected = assert.rejects(loaded, { name: "AbortError" });
    controller.abort();
    await rejected;
    assert.deepEqual(image.requests, ["/demo/mannequin-cutout.png", ""]);
    assert.equal(image.onload, null); assert.equal(image.onerror, null);

    const alreadyAborted = new AbortController(); alreadyAborted.abort();
    await assert.rejects(loadMannequinReference(alreadyAborted.signal), { name: "AbortError" });
    assert.deepEqual(images.instances[1].requests.filter(Boolean), [], "An aborted request must never start fetching the reference");
    assert.equal(images.instances[1].onload, null); assert.equal(images.instances[1].onerror, null);
  } finally { images.restore(); }
});

test("mannequin dimension mismatches and image failures release resources without a usable reference", async () => {
  const images = mockImages();
  try {
    for (const [width, height] of [[893, 569], [894, 568], [0, 0]]) {
      const controller = new AbortController();
      const loaded = loadMannequinReference(controller.signal);
      const image = images.instances.at(-1)!;
      image.naturalWidth = width; image.naturalHeight = height;
      const rejected = assert.rejects(loaded, /unexpected dimensions/);
      image.onload?.();
      await rejected;
      assert.equal(image.src, "");
      assert.equal(image.onload, null); assert.equal(image.onerror, null);
      const writes = image.requests.length;
      controller.abort();
      assert.equal(image.requests.length, writes, "The failed request must detach its abort listener");
    }
    const controller = new AbortController();
    const loaded = loadMannequinReference(controller.signal);
    const image = images.instances.at(-1)!;
    const rejected = assert.rejects(loaded, /could not load/);
    image.onerror?.();
    await rejected;
    assert.equal(image.src, "");
    assert.equal(image.onload, null); assert.equal(image.onerror, null);
    const writes = image.requests.length;
    controller.abort();
    assert.equal(image.requests.length, writes);
  } finally { images.restore(); }
});
