# Working on Holospex

Read `README.md` and `docs/architecture.md` before changing module boundaries.
This is a two-day educational hackathon prototype with a future AR-glasses
direction. The web scaffold uses synthetic geometry. The ML package now has a
real Endoscapes training and export path; read `ml/TRAINING.md` and `ml/STATUS.md`
before describing its capabilities. The user leads ML and general
architecture. Dataset labels and predictions are not reviewed lesson answers.

## Ownership and parallel work

- Person 1 / ML lead: `ml/`, `contracts/`, root tooling, integration assets.
- Person 2 / AR: `apps/web/src/overlays/`, `apps/web/src/camera/`, camera input.
- Person 3 / frontend: lesson flow, analytics, app shell, styles in `apps/web/`.
- Inspect current files and work in progress before editing. Keep another
  contributor's changes. Coordinate shared contracts; do not independently
  rename fields, anatomy IDs, or exported controls.
- Prefer the smallest complete handoff. Do not introduce an API server,
  database, authentication, a glasses SDK, or heavy model dependencies unless
  the requested feature needs them. Routine fixes do not need extra approval.

## Boundaries to preserve

1. JSON Schemas in `contracts/schemas/` define the wire format. Generated
   TypeScript is derived, never hand-edited. Python validates the same schemas.
2. Keep reviewed lesson answers separate from frame predictions. An ML score,
   missing prediction, or tracking failure cannot determine a CVS answer.
3. Keep surgical-video segmentation separate from physical-model marker
   registration. Original image pixels and world/model coordinates are not
   interchangeable. Glasses need the laparoscope feed to show internal imagery.
4. Keep source labels explicit: synthetic, reviewed, predicted, propagated.
   Never relabel synthetic/model output as reviewed to complete a demo.
5. Clear overlays when media/frame identity does not match or tracking is lost.
   Do not hold a sparse annotation over unrelated video frames.
6. Store elapsed time and cumulative hint exposure with a submitted answer.
   Feedback follows commitment; renderers do not own correctness or scoring.
7. Keep datasets, weights, recordings, credentials, and learner exports out of
   Git. Only add media to the explicit demo asset list after checking its use
   and recording provenance. No automatic dataset downloads or clinical-use claims.

## Checks and handoff

- Contract change: regenerate with `npm run contracts:generate`, update the
  producer/consumer and fixtures, then run `npm run check` and Python tests.
- Web change: `npm run check` and `npm run build`. Exercise the affected flow
  in a browser when interaction or rendering changes.
- ML change: `.venv/bin/python -m unittest discover -s ml/tests -v` after
  installing the optional dependencies needed for the changed modules.
- Run only checks relevant to the change; avoid tests that merely restate code.
- Report what works, what was tested, and which adapters remain placeholders.
  Do not describe a directory, interface, or mock as a completed capability.

## Autonomous ML experiment workflow

When the user requests autonomous training or model improvement, carry the
authorized experiment window through execution, evaluation and handoff. Setting
up a script or submitting a background job does not complete that request.
This workflow does not itself authorize a new training window or cloud spend.

1. **Establish the starting evidence.** Read `ml/EXPERIMENT_LOG.md`,
   `ml/STATUS.md`, `ml/TRAINING.md` and `ml/AUTONOMOUS_TRAINING.md`. Inspect
   current code, uncommitted work, available data, checkpoints and cloud jobs.
   Verify the baseline from saved metrics and artifact identities; treat old
   status reports as historical until checked.
2. **Make the stopping rule concrete.** Record the start time, absolute UTC
   deadline, target metric, compute/concurrency limits and maximum launches in
   a new, uniquely named controller state. Use the user's existing authorization
   for routine experiment choices; do not ask again before each trial. Resume
   interruptions with the same deadline and owned job IDs. Never extend a
   deadline, increase authorized spend or reuse a completed window implicitly.
3. **Freeze the comparison.** Record training/validation identities, case
   splits, label provenance, class mapping, ignored pixels and evaluation grid.
   For the current Endoscapes task, use six-class foreground macro IoU from
   pooled confusion counts on the fixed validation set at original annotation
   resolution. Exclude background from the class average, retain its false
   positives, and ignore source 255. Track per-class IoU and pooled/equal-case
   small-anatomy IoU as secondary measures. Keep the test split out of search
   and checkpoint selection; unreviewed proposals cannot become training truth.
4. **Choose experiments from results.** Identify the strongest supported
   bottleneck using learning curves, per-class/per-case errors and prior trials.
   Before each launch, log the hypothesis, comparison, expected information,
   concrete configuration and time cost. Prefer informative matched changes;
   do not repeat a failed idea without a new reason. Distinguish fresh training,
   weight-only initialization and true resumption. Adapt the next trial after
   verified results, and reserve time to repeat the best complete recipe across
   seeds. Freeze that recipe for the repeat group; report partial schedules
   separately. Consult primary research or official implementations when a new
   method needs evidence, including pretrained-weight provenance and licenses.
5. **Use durable orchestration.** Reuse `ml/cloud/autonomous_loop.py` and the
   source packaging, collection, audit and reporting utilities in `ml/cloud/`.
   Persist launch intent before submission, reconcile ambiguous submissions by
   exact identity, prevent duplicate controllers and bind each job to immutable
   source/data/configuration hashes. Keep state, logs and artifacts under an
   ignored run directory. Bound preparation, training and collection time;
   reserve time for evaluation/upload and stop launching when useful work no
   longer fits. Do not let a blocking collection delay deadline enforcement.
6. **Stay engaged throughout the window.** Monitor actual job states and logs,
   diagnose failures, collect results and update the plan while training runs.
   Use independent agents for bounded strategy, implementation or audit work
   when useful, with clear file ownership and one coordinated launch owner.
   Give concise progress updates with measured findings and the next decision.
   Continue until the requested deadline or a verified target is reached,
   unless the user stops the work or a real blocker requires their input.
7. **Verify before claiming improvement.** Check completion receipts, artifact
   sizes/checksums, actual epochs, configuration and checkpoint selection.
   Independently recompute native-grid metrics from saved confusion counts and
   verify fixed validation identities before accepting a score or target hit.
   An input-grid training log, cached success flag or incomplete artifact is
   insufficient. Report seed variation and class/case tradeoffs; repeated
   validation selection does not establish significance or clinical validity.
8. **Close the window completely.** At the deadline or verified target, stop
   new launches and cancel only owned unfinished jobs. Confirm every owned job
   is terminal and the controller has exited; cancellation requested is not
   cancellation confirmed. Finish bounded artifact collection, record missing
   or failed audits explicitly, and remove temporary helpers created for the
   run. Do not promote a research checkpoint into the demo automatically.
9. **Leave a reproducible handoff.** Update `ml/EXPERIMENT_LOG.md` after each
   result with rationale, configuration, metrics, selected/completed epochs,
   job ID and checkpoint hash. At closure, update `ml/STATUS.md` and the run
   guide, create a verified results report and comparison chart, and mark
   operator notes historical. Report the best score against the baseline,
   whether the target was met, completed/partial/failed runs, relevant checks,
   exact checkpoint/report paths and the next evidence-backed experiment.
   Never describe an unmet target as achieved because the time budget expired.

Unresolved details are tracked in `docs/team-plan.md`; continue independent
scaffold work while awaiting compute, real data, hardware, and reviewer choices.
