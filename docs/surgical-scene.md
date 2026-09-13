# Tabletop gallbladder scene

The `/mannequin` page now uses the [labeled-image overlay](camera-image-overlay.md).
This document describes the retained procedural 3D adapter, which is not offered
by the current camera setup UI. It constructs an authored, synthetic 3D open
torso on a surgical drape, with retractors, background liver
and abdominal context, and highlighted gallbladder, cystic duct and cystic
artery. No patient image, ML result, or third-party anatomical mesh is used.
The shapes are simplified and enlarged for demonstration, not reviewed lesson
answers or a reconstruction of an actual operation.

`SurgicalPreview` can show the model while the camera is off. Integrators can
pass `surgicalScene` and `createSurgicalRegistration()` to `ModelHud` to use the
adapter with marker 7 lying flat on the table. Its black square defaults to
80 mm; the virtual platform is approximately 25 × 37 cm and begins beyond the
marker's top edge. Keep the marker and model area in view. No device-specific
WebXR support is required; rendering uses Three.js/WebGL 2 and the existing
camera/ArUco tracker.

## Registration and composition

- `surgicalScene.ts` constructs the mesh and defines the three label locations.
  Scene x points right, y runs toward the model's head, and z points above the
  table. Scene z is negated when creating POSIT marker-relative anchor values.
- `estimateModelPose` and `projectModelAnchors` expose the existing calibrated
  marker pose without changing `FrameResult` or the physical-model JSON format.
- `SurgicalRenderer` converts that pose into a Three.js model matrix and uses
  the exact fx/fy/cx/cy projection for the captured frame. The conversion is
  tested to preserve handedness and match HUD label pixel positions.
- `ModelHud` captures the camera, estimates the pose, renders the transparent
  3D layer, and copies it into the composed canvas beneath the labels. Backup
  recordings therefore include the model, camera, labels and warnings together.
- Marker loss, camera interruption/stall, configuration changes and scene
  switching invalidate the pose. Visibility off and Identify/Assess suppress
  both the model and anatomical answers. WebGL failure displays a warning
  instead of anatomy labels floating without the scene.
- GPU resources are released when switching away or stopping the renderer.
  Geometry is generated once per renderer, not per video frame.

The removed configuration editor/export section remains absent. The separate
physical-model configuration upload is retained for measured mannequin labels.

## Checks and limits

Tests cover 3D mesh presence/volume, marker clearance, synthetic provenance,
pose-to-WebGL projection under tilt and resizing, composition order, and
clearing the 3D layer without moving the camera rectangle. Offline top and
oblique geometry views were inspected with Three.js's SVG renderer; this is
not a substitute for checking the WebGL presentation on the actual phone.

Camera calibration remains approximate in the built-in scene. A single
marker does not establish automatic table tracking, real-world occlusion,
patient anatomy, or clinical accuracy. Keep the printed marker visible;
hands and objects in front of the virtual body will not occlude it. Mobile
tracking, lighting and actual-device rendering still require a physical test.

Anatomical context reference: [NIDDK biliary-system illustration](https://www.niddk.nih.gov/news/media-library/17493).
Rendering references: [Three.js camera](https://threejs.org/docs/pages/Camera.html),
[WebGLRenderer](https://threejs.org/docs/pages/WebGLRenderer.html).
