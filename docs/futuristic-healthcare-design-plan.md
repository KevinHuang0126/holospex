# Holospex futuristic healthcare design plan

## Recommendation

Create a calm, precise medical-console appearance: deep navy surfaces, restrained cyan accents, crisp typography, clear operational states, and a prominent anatomy viewport. Preserve the current routes, feature set, control groups, button order, and interaction behavior.

The presentation pass below is implemented in the six existing UI stylesheets. No button was removed or newly hidden. The original research and implementation sequence remain below for context; see the implementation verification at the end for actual coverage.

## Evidence reviewed

- Inspected the deployed `/prototype` HUD with saved video results and the main entry, which redirected to `/mannequin` during review.
- Reviewed the current React entry points, lesson UI, HUD inputs, live-feed controls, dataset samples, training-image placement, recording component, and related styles.
- The main identification page has a spacious light background, muted green branding, a large introduction, and a white control panel. The user's active HUD rendered dark, while the checked-out base stylesheet defines a light palette. Verify the browser appearance and deployment version before treating that difference as an intentional product theme.
- Existing uncommitted work in `VideoHud.tsx` and `videoHud.css` reserves space for changing warnings to keep playback controls stationary. Preserve and coordinate that work.
- Research sources below inform the proposal. The palette and styling recommendations are design judgments; the sources do not establish that a particular aesthetic improves medical outcomes.

## Research and application

| Reference | Relevant evidence | Application to Holospex |
| --- | --- | --- |
| [Intuitive da Vinci 5](https://www.intuitive.com/en-gb/products-and-services/da-vinci/5) | Describes a universal interface across system components, guided setup, and integrated controls. | Use one consistent visual language across lesson, camera, upload, stream, and sample modes; retain familiar control locations. |
| [Siemens Healthineers Cinematic Reality](https://www.siemens-healthineers.com/en-us/press-room/press-releases/cinematic-reality-app) | Presents immersive, interactive anatomical visualization for uses including medical education. | Let anatomical imagery establish the healthcare identity. Keep decorative effects outside the image and make labels legible. Do not imply Holospex has its 3D capabilities. |
| [W3C text contrast](https://www.w3.org/WAI/WCAG22/Understanding/contrast-minimum.html) | WCAG AA requires 4.5:1 for normal text and 3:1 for large text. | Validate text on actual panel and button backgrounds, including hover and selected states. Avoid faint gray clinical explanations. |
| [W3C target size](https://www.w3.org/WAI/WCAG22/Understanding/target-size-minimum.html) | WCAG 2.2 AA sets a 24-by-24 CSS pixel minimum, with defined exceptions. | Use roughly 44-pixel touch targets as the product design target. Wrap rows before shrinking controls. |
| [W3C interaction animation](https://www.w3.org/WAI/WCAG22/Understanding/animation-from-interactions.html) | The AAA criterion calls for disabling nonessential motion triggered by interaction. | Honor reduced-motion preferences and keep transitions short. Avoid continuous scanning, pulsing, and parallax. |

## Visual specification

### 1. Palette and materials

Proposed starting tokens, subject to contrast verification:

| Role | Color | Use |
| --- | --- | --- |
| Page | `#07131D` | Deep navy background |
| Panel | `#102330` | Main workspaces and control groups |
| Raised surface | `#173241` | Fields and secondary surfaces |
| Primary text | `#EAF5FA` | Headings and essential information |
| Secondary text | `#ADC2CE` | Descriptions and metadata |
| Accent | `#55DCE0` | Primary actions and selected indicators; dark text on filled accent buttons |
| Warning | `#F2BF67` | Warning icon and text with a restrained dark amber surface |
| Error | `#FF9292` | Error icon and explanatory text |

Use subtle borders and 12–16 pixel panel corners. Reserve a faint glow for a selected control or the outer viewport frame. Keep functional panels predominantly opaque. Any background gradient or technical line motif belongs in existing introductory whitespace, never over surgical media.

Keep the established anatomy ID/color mapping. UI accent colors must not recolor anatomical classes or imply that a prediction is reviewed.

### 2. Typography and hierarchy

- Keep the existing Manrope headings and DM Sans body fonts; improve hierarchy through weight, size, and spacing before adding dependencies.
- Use readable body text around 14–16 pixels, explicit labels, and stronger section headings. Keep existing heading placement and approximate height.
- Use tabular numerals for timestamps, frame numbers, and scores where they already exist.
- Keep uppercase tracking limited to short section labels. Retain visible action names even if small icons are added.

### 3. Buttons and forms

- Primary actions use a solid cyan fill with dark text. Secondary actions use opaque navy surfaces and a clear border. Selected modes also have a non-color cue such as an underline.
- Give file inputs, selects, number inputs, checkboxes, and sliders consistent sizing and spacing without replacing their native semantics.
- Define default, hover, pressed, selected, disabled, focus, and loading appearances. A disabled stop/play control remains visible in its current place.
- Preserve button labels and handlers in the initial styling pass. Do not merge Play/Pause or replace text actions with icon-only controls.

### 4. Anatomy viewport and status

- Preserve the media rectangle, aspect ratio, fit behavior, frame matching, and existing label positions. Frame the viewport with a thin technical border outside the image.
- Keep source identity and technical status distinct: a model-ready status does not indicate anatomical accuracy or lesson correctness.
- Style existing source/status information with clear labels and restrained badges; do not add decorative heart rates, patient data, confidence values, or system-readiness claims.
- Keep the existing warning area readable and stable so warnings appearing/disappearing do not move playback controls. Style warnings consistently inside and outside recorded canvases.
- Do not apply a color filter, blur, scanline effect, or animation to the media or anatomy overlays.

## Feature and placement preservation map

“Same place” means the same containing section, ordering, and relationship to the image at the same viewport width. Minor spacing changes are allowed; swapping major regions is not.

| Area | Features retained | Placement constraint |
| --- | --- | --- |
| App shell | Home link, title, prototype/source wording, route links | Header and footer remain in place; existing introduction remains above workspace. |
| Prototype modes | Video lesson scaffold, Dataset samples, AR / HUD demo | Same row above content, same order. |
| HUD inputs | Camera identification, Upload video, Stream link, Video + saved results, Training image AR | Same top control group and order; preserve route-specific differences. |
| Shared controls | Learning mode where applicable, Show overlays | Same existing rows; retain conditional availability. |
| Camera | Device selection, Start camera, Stop camera, model status, Refresh model | Same setup area above feed. |
| Live appearance | Masks + outlines / Outlines only, Show model scores | Same area above feed; same defaults. |
| Uploaded video | File chooser, Remove video, playback/seek/replay behavior | Preserve source setup and existing player controls. |
| Stream | URL, automatic/HLS/direct format, Connect/Reconnect, Disconnect, help and errors | Same source setup group; no modal or hidden settings menu. |
| Saved results | Clip file, JSON file, Source, ML threshold | Same row above video, same order. |
| Saved-results playback | Play, Pause, Seek timestamp, Seek, Replay | Same row below video, same order; reserve warning height. |
| Recording | Record backup, Stop backup recording, resulting save/download behavior and instructions | Same recording section; stop action remains directly accessible. |
| Lesson | Image, Reveal/Hide labels, checkpoint, answer choices, Check answer, feedback, Next question/Try again | Image stays left and questions right on desktop; existing stacked flow on mobile. Export saved answers stays below workspace. |
| Dataset samples | Folder/file imports, conditional reload, previous/case selector/next, learning mode, exact mask/polygons, overlay toggle | Keep existing headings and control groups. Existing sample-import disclosure behavior stays as implemented. |
| Sample appearance | Opacity, boundaries, labels/pointers, Boundaries only, Reset overlay, image point selection and metadata | Same toolbar and information regions. |
| Training image AR | Labeled image, Marker test, Mannequin configuration; file/folder input and conditional reload | Same nested setup controls and order. |
| Image placement | Sample, Scene, Image view, Place image, opacity, sizes/table dimensions, suggest/reset placement, horizontal/vertical/width adjustments | Keep within current conditional setup sections; no new collapse. |
| Marker/model setup | Camera selection and start/stop, marker size, configuration JSON, printable marker download, calibration/tracking information | Same setup context; preserve lost-tracking and unavailable states. |

### Moves and hidden controls

- **Required desktop button moves: none.**
- **Newly hidden controls: none.** Existing conditional controls and disclosures retain their behavior.
- **Mobile change:** let crowded rows wrap in their current reading order. Controls may sit one row lower, but remain labeled and accessible without opening a new menu.
- **Optional, outside the initial pass:** shorten the large introduction to bring the workspace higher. This would move the entire workspace upward and should be shown as a separate before/after proposal before implementation.
- Moving labels into a new sidebar, moving playback onto the video, introducing sticky toolbars, or collapsing calibration controls is outside this recommendation. If later necessary, report the exact control, old/new position, trigger width, and reason before making the change.

## Implementation sequence

### Phase 1 — Freeze the functional baseline

Capture desktop and phone screenshots of `/prototype`, `/mannequin`, and `/samples`, including conditional input modes. Record button names, order, accessible labels, disabled conditions, and existing disclosure behavior. Verify the deployed build versus the local checkout, and coordinate the pending HUD changes.

### Phase 2 — Establish the shared appearance

Introduce shared color, typography, border, spacing, and focus tokens in `apps/web/src/styles.css`. Apply them to the shell and common controls. Update `camera/deviceSetup.css`, `camera/liveFeed.css`, `camera/mannequin.css`, and `overlays/sampleTest.css` to consume those tokens. Prefer presentation-only changes and retain component state and event wiring.

### Phase 3 — Refine the working screens

Apply the same styles to the lesson, input toolbars, sample controls, and training-image setup. Coordinate HUD presentation changes with the AR owner. CSS styling outside a canvas will not appear in backup recordings; any desired recorded HUD styling needs explicit renderer review and a recording check. Preserve geometry and shared anatomy colors.

### Phase 4 — Responsive and accessibility pass

Review approximately 1440, 1024, 768, and 390 pixel widths, plus 320-pixel reflow and 200% zoom. Check touch targets, keyboard navigation, visible focus, long labels, warnings, native dropdowns, file inputs, and reduced motion. Ensure rows wrap without clipped labels or new menus.

### Phase 5 — Verify feature parity and hand off

Run `npm run check` and `npm run build` after implementation. Exercise the affected browser flows using existing permitted fixtures and report any hardware/model-dependent checks that could not be completed.

Acceptance criteria:

- Every baseline action remains available in the same region and order; no additional click is required to reach a currently visible control.
- Lesson submission, feedback sequencing, label/hint exposure, elapsed-time recording, progression, and answer export behave as before.
- Camera, upload, stream, saved-result loading, playback, seeking, replay, recording, and saving retain their behavior.
- Sample rendering modes, appearance controls, point selection, placement controls, calibration and marker configuration remain available.
- Frame/source replacement and tracking loss still clear invalid overlays. No stale geometry appears after seek or mode changes.
- Warning changes do not make playback buttons jump. Essential explanations stay visible or readable in the existing reserved warning area.
- Original media, image-coordinate mapping, anatomy colors, provenance, model configuration, and learner scoring are unchanged.
- Recorded output is checked separately from page appearance. No material new UI-induced playback or interaction slowdown is accepted.
- Final handoff includes before/after screenshots and an explicit list of moved/hidden controls, even if that list is empty.

## Scope and review status

This plan covers the website's existing experiences, not a new marketing homepage or a new healthcare capability. Initial research inspected visible layout. The following section records the subsequent implementation and verification.

## Implementation verification

### Delivered

- Shared navy/cyan interface tokens, opaque panels, subtle introductory background light, consistent native controls, explicit selected states, readable source/status information, and keyboard focus styles.
- Existing panel order, desktop columns, control order, route composition, and major spacing retained. Phone rows wrap and use larger touch targets; no extra menus or collapsed controls were introduced.
- Application changes are limited to `styles.css`, `camera/deviceSetup.css`, `camera/liveFeed.css`, `camera/mannequin.css`, `overlays/sampleTest.css`, and `overlays/videoHud.css`.
- Component handlers, lesson storage/scoring, API calls, schemas, inference, media processing, canvas recording, and anatomy palette were not edited by this design pass. Separate concurrent HUD renderer/test changes were preserved.
- The existing warning-area height and empty-state reservation remain intact. The warning now uses amber text and border on an opaque dark surface.

### Checks completed

- `npm run check`: passed; 8 contract tests passed; 129 web tests passed, 2 skipped; TypeScript checks passed.
- `npm run build`: passed. Vite reports a JavaScript chunk-size advisory; this pass adds no JavaScript or dependencies.
- `git diff --check`: passed.
- Exact accessible-tree comparisons matched the initial lesson, camera, and empty sample views before and after the theme change.
- Browser lesson checks: reveal/hide labels, selection, submission blocked while hints are exposed, feedback after submission, next question, unsupported-overlay state, and final retry action appearance.
- Existing synthetic video: local file selection, playback, frozen-frame inspection, playback speed change, seeking, and backup recording start/stop were exercised.
- Stream setup: all source controls remain present, and a non-HTTPS test address was rejected with the existing validation message.
- Training-image setup: scene selection exposes the existing mannequin placement and marker controls, including at 320px width.
- Existing train-split sample imported from local files with its original labels/image/index mask. Verified Boundaries only, Reset overlay, and assessment hiding/disabling all anatomical layers and appearance controls.
- Lesson, loaded sample, and uploaded-video views had no page-width overflow at 1440, 1024, 768, 390, and 320 CSS pixels. Stream and mannequin setup were also inspected at 320px.
- Saved-video HUD: the Play button's document position was identical before/after hiding the warning via the overlay toggle (2232.046875px at the tested 320px viewport). Playback remained operable.
- Keyboard focus was visibly outlined; measured 3px solid on the focused home link. Reduced-motion CSS removes button transitions.
- Token contrast checks: primary text/panel 14.51:1; secondary text/raised panel 7.26:1; secondary text/hover panel 5.65:1; primary button text/fill 10.65:1; warning text/background 8.34:1; control border/field 3.31:1.

### Verification limits

- The local model reported pending. No live inference, physical camera/marker session, or external HLS connection was started during this presentation pass. Their existing automated tests passed where enabled.
- Answer export was clicked without an application error, but the in-app browser did not report a download event. Recording start/stop worked; saved file contents and recording playback were not independently verified. Export and recording implementations are unchanged by this pass.
- Native 200% browser zoom and OS reduced-motion preferences were not exercised; responsive reflow and the reduced-motion stylesheet were checked separately.
- No production deployment was performed. Desktop and phone preview screenshots were saved in the task's visualization directory; initial baseline screenshots are in the task history.

### Movement report

No desktop controls were relocated or newly hidden. On small screens controls wrap in the same order, and some fields take a full row to preserve legibility and touch access. Header and introduction remain above the workspace.
