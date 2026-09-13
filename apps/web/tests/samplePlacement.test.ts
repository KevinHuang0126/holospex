import assert from "node:assert/strict";
import test from "node:test";
import { parsePlacementRequest, parsePlacementResponse, TEMPLATE_ID, type PlacementRequest, type PlacementResponse } from "../../../shared/placement";
import { placementFromSuggestion, placementRequestFor, requestSamplePlacement } from "../src/camera/samplePlacement";
import { mannequinSampleLayout, projectMannequinAnchors } from "../src/camera/mannequinComposite";
import { imagePlaneLayout, projectImagePlaneAnchors } from "../src/camera/ImagePlaneRenderer";
import { projectPoint, type ModelPose } from "../src/camera/modelRegistration";
import type { ImageOverlayAsset } from "../src/camera/imageOverlayAsset";

const close = (actual: number, expected: number) => assert.ok(Math.abs(actual - expected) < 0.000001, `${actual} != ${expected}`);
function source(): PlacementRequest {
  return parsePlacementRequest({ schemaVersion: 1, templateId: TEMPLATE_ID, sampleId: "case-asymmetric", width: 854, height: 480,
    regions: [{ structureId: "gallbladder", bounds: { x: 0.10, y: 0.10, width: 0.30, height: 0.60 }, centroid: { x: 0.16, y: 0.30 }, pixelCount: 1400 }] });
}
function response(request = source()): PlacementResponse {
  return { schemaVersion: 1, templateId: TEMPLATE_ID, sampleId: request.sampleId, source: "ai_suggested", focusStructureId: "gallbladder",
    targetRegion: "right_upper_abdomen", centerX: 0.345, centerY: 0.545, focusWidthFraction: 0.06, reason: "Illustrative placement in the authored upper abdominal region." };
}

test("placement requests serialize only anatomy measurements and never read image pixels", async () => {
  const request = source();
  const asset = { id: request.sampleId, width: request.width, height: request.height, regions: request.regions,
    get scene() { throw new Error("Image textures must not be read for the placement API."); },
    get anatomy() { throw new Error("Mask textures must not be sent to the placement API."); },
    imageBase64: "PRIVATE_IMAGE_BYTES", cameraFrame: "PRIVATE_CAMERA_BYTES", sourceLabel: "Private fixture" } as unknown as ImageOverlayAsset;
  const metadata = placementRequestFor(asset);
  assert.deepEqual(metadata, request);
  const controller = new AbortController();
  let sent: RequestInit | undefined;
  const fetcher = (async (url: string | URL | Request, init?: RequestInit) => {
    assert.equal(url, "/api/placement"); sent = init;
    return Response.json(response(request));
  }) as typeof fetch;
  assert.deepEqual(await requestSamplePlacement(metadata, controller.signal, fetcher), response(request));
  assert.equal(sent?.method, "POST"); assert.equal(sent?.signal, controller.signal);
  assert.deepEqual(sent?.headers, { "Content-Type": "application/json" });
  assert.deepEqual(JSON.parse(sent?.body as string), request);
  assert.doesNotMatch(sent?.body as string, /PRIVATE|base64|canvas|cameraFrame|imageBase64/);
});

test("an asymmetric organ's actual centroid and measured width align through mannequin and marker projection", () => {
  const request = source(), suggestion = response(request), focus = request.regions[0];
  const placement = placementFromSuggestion(request, suggestion, 894, 569);
  const rect = mannequinSampleLayout(request.width, request.height, 894, 569, placement);
  const sourceAnchor = { id: "focus", structureId: "gallbladder" as const, x: focus.centroid.x * request.width, y: focus.centroid.y * request.height };
  const [anchor] = projectMannequinAnchors([sourceAnchor], rect);
  close(anchor.x, suggestion.centerX * 894); close(anchor.y, suggestion.centerY * 569);
  close(rect.scale * focus.bounds.width * request.width, suggestion.focusWidthFraction * 894);
  assert.notEqual(placement.centerX, suggestion.centerX); // Centering the whole surgical frame would move the anatomy.
  assert.ok(rect.x >= 0 && rect.y >= 0 && rect.x + rect.width <= 894 && rect.y + rect.height <= 569);
  const plane = imagePlaneLayout(894, 569, 600, 80);
  for (const tilt of [-0.35, 0.28]) {
    const pose: ModelPose = { rotation: [[1, 0, 0], [0, Math.cos(tilt), -Math.sin(tilt)], [0, Math.sin(tilt), Math.cos(tilt)]],
      translation: [0, -180, 1100], reprojectionError: 0,
      calibration: { width: 1280, height: 720, fx: 1000, fy: 1060, cx: 610, cy: 350 } };
    const expected = projectPoint([(suggestion.centerX - 0.5) * 600, 50 + (1 - suggestion.centerY) * 600 * 569 / 894, -0.5],
      pose.rotation, pose.translation, pose.calibration)!;
    for (const [width, height] of [[1280, 720], [640, 360], [960, 540]]) {
      const projected = projectImagePlaneAnchors([anchor], 894, 569, pose, width, height, plane);
      assert.equal(projected.length, 1);
      close(projected[0].x, expected[0] * width / 1280); close(projected[0].y, expected[1] * height / 720);
    }
  }
});

test("brain metadata targets the head while preserving exact focus position when the full frame limits scale", () => {
  // This exercises placement geometry, not a brain-image import adapter.
  const request = parsePlacementRequest({ schemaVersion: 1, templateId: TEMPLATE_ID, sampleId: "brain-metadata-fixture", width: 256, height: 256,
    regions: [{ structureId: "brain", bounds: { x: 0.05, y: 0.15, width: 0.90, height: 0.70 }, centroid: { x: 0.90, y: 0.45 }, pixelCount: 3000 }] });
  const suggestion: PlacementResponse = { ...response(request), focusStructureId: "brain", targetRegion: "head", centerX: 0.05, centerY: 0.50, focusWidthFraction: 0.12 };
  const placement = placementFromSuggestion(request, suggestion, 894, 569);
  const rect = mannequinSampleLayout(256, 256, 894, 569, placement);
  close(rect.x + 0.90 * 256 * rect.scale, 0.05 * 894);
  close(rect.y + 0.45 * 256 * rect.scale, 0.50 * 569);
  assert.ok(rect.scale * 0.90 * 256 < suggestion.focusWidthFraction * 894);
  assert.ok(rect.x >= 0 && rect.y >= 0 && rect.x + rect.width <= 894 && rect.y + rect.height <= 569);
  assert.throws(() => placementFromSuggestion(request, { ...suggestion, targetRegion: "right_upper_abdomen" }, 894, 569), /contradicts/);
  assert.throws(() => placementFromSuggestion(request, suggestion, 0, 569), /dimensions/);
});

test("response association rejects stale identity, incompatible anatomy and unsupported target coordinates", () => {
  const request = source(), valid = response(request);
  for (const bad of [{ ...valid, sampleId: "old-sample" }, { ...valid, templateId: "old-template" }, { ...valid, source: "reviewed_annotation" },
    { ...valid, focusStructureId: "brain", targetRegion: "head" }, { ...valid, targetRegion: "head" }, { ...valid, centerX: 0.90 }, { ...valid, focusWidthFraction: 0.90 },
    { ...valid, imageBase64: "unexpected bytes" }]) assert.throws(() => parsePlacementResponse(bad, request));
  const ambiguous = { ...request, regions: [...request.regions, { ...request.regions[0], structureId: "brain" }] };
  assert.throws(() => parsePlacementResponse(valid, ambiguous), /ambiguous/);
});

test("API errors, invalid JSON, oversized responses and aborts never become accepted placements", async () => {
  const request = source(), signal = new AbortController().signal;
  const errors: [Response, RegExp][] = [
    [Response.json({ error: { code: "missing_key", message: "Placement API key is missing." } }, { status: 503 }), /API key is missing/],
    [Response.json({ error: "Service unavailable." }, { status: 503 }), /Service unavailable/],
    [Response.json({ unexpected: true }, { status: 500 }), /could not suggest/],
    [new Response("<html>Application shell</html>"), /API is unavailable/],
    [new Response(" ".repeat(16_385)), /oversized response/],
    [Response.json({ ...response(request), sampleId: "previous-sample" }), /does not match/],
  ];
  for (const [reply, message] of errors) await assert.rejects(requestSamplePlacement(request, signal, (async () => reply) as typeof fetch), message);
  const controller = new AbortController();
  const aborted = requestSamplePlacement(request, controller.signal, ((_input: unknown, init?: RequestInit) => new Promise((_resolve, reject) => {
    init?.signal?.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError")), { once: true });
  })) as typeof fetch);
  controller.abort(); await assert.rejects(aborted, { name: "AbortError" });
});
