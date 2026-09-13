# Two-day implementation guide

Build one complete educational lesson and one small camera demonstration that explains the future glasses experience. The current scaffold is an integration starting point; real surgical assets, content review, a trained model, and working marker registration must be verified separately.

## Owners and handoffs

| Owner | Deliverable | Handoff to the others |
| --- | --- | --- |
| Person 1: ML + architecture lead | Contracts, sample results, dataset/class mapping, inference export, integration decisions | Validated result assets, media/frame identity, supported classes, source and limitations |
| Person 2: AR + rendering | Video overlay alignment and show/hide controls; camera marker registration and predefined labels | Renderer controls, tracking states, supported device setup, geometry conventions |
| Person 3: lesson + frontend | Question flow, reviewed feedback integration, attempts, export, usable navigation | Checkpoint/visibility events, learner interaction requirements, testable lesson |

Person 1 is the integration owner. Each person documents the evidence and limitations for their own component and supplies their own presentation material; do not leave all research and the presentation to Person 3. Name a clinical content reviewer before describing lesson answers or geometry as reviewed.

For agents: work within the assigned directories and existing module boundaries. Before changing a shared schema, asset format, or root command, tell the integration owner. Give the next contributor a runnable artifact, the command to exercise it, what you checked, and what remains mocked. Do not rename anatomy IDs independently or present TODOs as completed features.

## Integration gates

These are acceptance gates, not a report of features already implemented.

### Gate 1 — Contract and synthetic path · Day 1, first two hours

- Freeze anatomy IDs/colors, image coordinates, timestamp units, geometry representation, and the three record responsibilities in [the architecture](architecture.md).
- One explicitly synthetic fixture flows through export/validation, asset delivery, renderer, show/hide controls, and a recorded learner response.
- Seeking/resizing does not leave geometry on unrelated content. Unavailable results clear labels.
- Confirm the actual demo device, camera access setup, ML compute, dataset access, and permitted media. Training must not block this gate.

### Gate 2 — First real lesson · Day 1 afternoon

- Load selected permitted imagery/footage; stop at checkpoints whose annotations and answers have a named content review provenance.
- Complete instruction → hidden-label response → feedback. Export the selected answer, response time, checkpoint and hint exposure.
- Use an independently reviewed difficult example with a supported answer rubric. A low model score alone is insufficient to author `cannot_determine` feedback.
- Prepare a different case for the final hidden-label transfer question. If review or real assets are unavailable, keep the demo visibly synthetic and report that limitation.

### Gate 3 — ML and camera branches · Day 1 afternoon into Day 2 morning

- ML: export predictions for the selected checkpoints. Switch sources without changing lesson answers or renderer internals. Identify unsupported anatomy and retain a successful-empty result separately from inference failure.
- ML: inspect predictions in original-frame coordinates. If training or reporting accuracy, separate train/validation/test at the surgical-video level; choose thresholds using validation data.
- Camera: register one marker/model on the selected phone or desktop camera, display predefined labels through small motions, hide labels for identification, and clear them on tracking loss.
- The presentation visibly distinguishes a synthetic fixture, reviewed geometry, ML predictions, and marker-positioned model labels. Physical labels must not be described as surgical-image ML detections.

### Gate 4 — Device rehearsal and freeze · Day 2

- From a clean start, run the lesson, export an attempt, switch into camera mode, and recover from denied camera access or tracking loss.
- Check pause, seek, replay, source switching, resize/phone rotation, missing assets, malformed results, and late results. There must be no stale geometry on a different frame.
- Test the actual phone/browser early. Camera access on a phone typically needs a secure origin; another computer's HTTP LAN address is not the phone's localhost. Decide the development/demo HTTPS route before rehearsal.
- Check lesson clarity with available participants and describe nonclinical participant feedback as usability findings. Assessment performance on the hack demo is not evidence of clinical readiness or durable learning.
- Freeze a working build, capture backup recordings, and assign demo/recovery operators before adding polish.

## Scope decisions

Keep precomputed inference, a small checkpoint lesson, a separate-case assessment, and one marker-based camera exercise. If time shrinks, reduce the number of classes/checkpoints and camera interactions. Remove fine-tuning, continuous-video masks, temporal smoothing, additional procedures, and visual polish before disrupting the integrated demo.

Do not make reviewed content contingent on model quality. A reviewed lesson can demonstrate teaching while the prediction view demonstrates honest ML successes and failures. If either branch is unfinished, label it accurately and use its backup only as a recording of the actual feature.

No inference API, account system, cloud database, glasses SDK, or live-patient workflow is required for this hack. Add one only when a named integration need justifies the new dependency.

## Decisions the lead still needs

- **ML progress:** the local M4 GPU runs the Endoscapes baseline. Data acquisition,
  model training and offline exports are documented in [the ML runbook](../ml/TRAINING.md).
  Cloud compute is optional; retain per-class evidence before choosing a larger run.
- **Camera:** the actual device/browser, physical model, and marker/registration library; define model coordinates with Person 2.
- **Content:** reviewer availability, selected cases, asset license/permission record, and review criteria for uncertain views.
- **Product:** whether future “live practice” means simulation or actual patient operations. Current implementation assumes simulation/education until clarified.

Keep these answers with the assets or implementation notes they affect. They need not stop the synthetic integration path.
