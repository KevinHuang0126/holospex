"""Build a portable offline review page and ZIP from immutable mask proposals.

Only packages existing model-generated candidates; never invents review decisions,
changes proposal pixels, or enrolls annotations in a training dataset.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import zipfile

from review_io import validate_bundle


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def relative_file(root: Path, value: str) -> Path:
    path = PurePosixPath(value)
    if not isinstance(value, str) or path.is_absolute() or not path.parts or any(part in ('..', '.') for part in path.parts) or '\\' in value:
        raise ValueError('Unsafe relative image path')
    local = root.joinpath(*path.parts)
    if local.is_symlink() or not local.resolve().is_relative_to(root.resolve()) or not local.is_file():
        raise ValueError('Missing or unsafe image file: '+value)
    return local


def reviewer_instructions(bundle: dict) -> str:
    """Keep the portable handoff instructions aligned with the actual batch."""
    image_count = len(bundle['images'])
    case_count = len({row['videoId'] for row in bundle['images']})
    candidate_count = sum(len(row['candidates']) for row in bundle['images'])
    return f'''HOLOSPEX ANATOMY REVIEW — START HERE

This batch: {image_count} images from {case_count} surgical cases; {candidate_count} masks to review.
Structures: {", ".join(row['label'] for row in bundle['classes'])}.

A mask is the colored area over the image. Each candidate is one computer-made
guess for one named structure. Please check every candidate in every image.
The goal is careful decisions, not a minimum number of approvals.

1. OPEN THE REVIEW
   Save this ZIP to your computer and extract/unzip it first. On Windows,
   right-click the ZIP and choose Extract All; on Mac, double-click the ZIP.
   Open the extracted folder, then open review.html in a desktop browser
   (Chrome, Edge, Firefox or Safari). If needed, right-click review.html and
   choose Open With, then your browser. No account or installation is needed.
   You can work offline. Keep the extracted folder for later sessions.

2. START UNDER YOUR NAME
   Enter Your name. Under Review scope, select
   "Anatomy — assess correctness", then click "Start review".
   This records who checked the anatomy. If you cannot assess a structure,
   use "Needs expert" for that candidate instead of guessing.

3. CHECK ONE STRUCTURE AT A TIME
   Select a structure under "Candidates in this image" on the right.
   Read its name. Toggle "Mask" off and on to compare with the original image.
   Toggle "Source box" to hide/show the rectangle. The box is a location hint;
   it is not the correct outline. Inspect the entire mask, even outside the box.
   Use Zoom to see edges, Pan to move around, and Fit to see the whole image.
   Opacity changes only how transparent the mask looks.

   Check for missed visible parts and color spilling into neighboring anatomy.
   Pay special attention to duct and plate boundaries and triangle extent.
   Follow what is visible; do not invent a boundary hidden by tissue or tools.
   A computer score does not tell you whether the anatomy is correct.

4. FIX THE MASK IF YOU CAN
   Keep "Mask" checked. Select Brush and drag to add missing areas.
   Select Erase and drag to remove extra areas. The Brush slider changes the
   size of both tools; zoom in and use a smaller size near fine boundaries.
   Only the selected structure changes. Undo reverses the last edit;
   Reset mask returns that candidate to its original computer proposal.

5. RECORD AN EXPLICIT DECISION
   "Accept unchanged": the original mask is correct as shown.
   "Approve edit": you corrected the mask and the final version is correct.
   "Reject": you can identify that the proposal is wrong and are leaving it
     without a usable correction. Add a short explanation in Notes first.
   "Needs expert": you are unsure of the anatomy or its boundary. Describe
     the uncertainty in Notes first. It is fine to choose this when unsure.

   When a note describes a problem you fixed, say that clearly, for example:
   "Original mask included adjacent tissue; removed it. Final mask corrected."
   Finish any edits AND notes before clicking the decision button. Any later
   mask or note change puts the candidate back to Pending, so decide again.
   Looking at a mask, painting it, or writing a note alone does not approve it.

6. CONTINUE AND SAVE OFTEN
   Use "Next →" or select the next structure on the right. The arrows and
   image menu above the picture also let you move between images.
   Click "Export review JSON" every few decisions and before each break.
   This downloads a file named holospex-review-...json, usually to Downloads.
   Keep the newest file: it contains all decisions, edits and notes so far.
   The browser also tries to save locally, but that copy is not guaranteed.

7. RESUME LATER
   Open the SAME batch's review.html. Click "Resume from JSON" and choose your
   most recent downloaded review file. Confirm that it is your saved review.
   This replaces the current session and keeps the reviewer name from the file;
   export any current work you want to keep before importing a saved review.
   A "Restore saved review" banner may also offer this browser's saved copy.
   Use your downloaded JSON when moving to another browser or computer.

8. FINISH AND SEND BACK
   A completed batch shows "{candidate_count} / {candidate_count} decisions recorded" at the top.
   Reject and Needs expert count as decisions; there is no approval quota.
   You may also return a partial review if you need help or run out of time.
   Click "Export review JSON" one last time. Send ONLY that newest .json file
   back to the ML lead. There is no automatic upload or shared live syncing.
   You do not need to send the ZIP, images, screenshots or bundle.json back.

Keep the original bundle.json and images unchanged. Use a separate review file
for each reviewer. This review records anatomy masks; it does not answer lesson
questions or automatically start model training.
'''


def build_package(bundle_path: Path, output_dir: Path, *, template_path: Path | None = None, license_paths=()) -> dict:
    from PIL import Image

    bundle_path, output_dir = Path(bundle_path).resolve(), Path(output_dir).resolve()
    template_path = Path(template_path or Path(__file__).with_name('review.template.html'))
    raw = bundle_path.read_bytes()
    bundle = json.loads(raw)
    validate_bundle(bundle)
    template = template_path.read_text()
    if template.count('__HOLOSPEX_REVIEW_DATA__') != 1:
        raise ValueError('Review template must contain exactly one data token')
    if output_dir.exists() or output_dir.with_suffix('.zip').exists():
        raise ValueError('Choose a new handoff directory; existing review packages are preserved')
    originals, embedded = {}, {}
    for row in bundle['images']:
        path = relative_file(bundle_path.parent, row['imagePath'])
        data = path.read_bytes()
        if sha256(data) != row['imageSha256']:
            raise ValueError('Source image hash differs: '+row['id'])
        with Image.open(BytesIO(data)) as image:
            if image.size != (row['width'], row['height']) or image.format != 'JPEG':
                raise ValueError('Source image must be an aligned original JPEG: '+row['id'])
            image.verify()
        originals[row['imagePath']] = data
        embedded[row['id']] = 'data:image/jpeg;base64,'+base64.b64encode(data).decode('ascii')
    licenses = {}
    for path in map(Path, license_paths):
        if path.name in licenses:
            raise ValueError('Duplicate license filename; use distinct source names')
        licenses[path.name] = path.read_bytes()
    envelope = {'bundle':bundle, 'bundleSha256':sha256(raw), 'images':embedded}
    encoded = json.dumps(envelope, separators=(',', ':'), ensure_ascii=True, allow_nan=False).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    html = template.replace('__HOLOSPEX_REVIEW_DATA__', encoded)
    mask_count = sum(len(row['candidates']) for row in bundle['images'])
    readme = f'''# Holospex anatomy-mask review

This package contains {len(bundle['images'])} images from {len({row['videoId'] for row in bundle['images']})} different TRAIN cases and {mask_count} model-generated candidate masks.

Read **START-HERE.txt** for step-by-step instructions written for the reviewer.

1. Extract/unzip the package, then open **review.html** in a current desktop Chrome, Edge, Firefox or Safari browser. No installation, server, GCP login or internet connection is required.
2. Enter your reviewer name and choose your review scope. Use anatomy review only when you can assess the named structure.
3. Select an image and structure. Toggle the mask and original box; zoom in to inspect boundaries. Draw/erase to correct a proposal and use undo for mistakes.
4. Use **Accept unchanged** for a good original mask or **Approve edit** after correcting it. Use **Reject** for a wrong proposal left without a usable correction, or **Needs expert** when uncertain, adding a note first. Finish notes before clicking the decision; any later note or mask change returns it to Pending. A model score is not anatomical confidence.
5. **Export review JSON** after a few decisions and before closing. Local browser saving is best effort; the downloaded JSON is your portable backup. Use **Resume from JSON** to import that file in another browser/computer, keeping the recorded reviewer.
6. Send the exported JSON back to the ML lead. They already have the immutable image/proposal bundle. No images or credentials need to be sent with your review.

Review guidance: check the named structure, visible extent, missed branches, spill into adjacent tissue, and obscured boundaries. Keep holes and separate visible parts where appropriate. Source boxes provide location hints only. Only the selected candidate changes when painting; masks may overlap, and any overlap will need resolution before dense training labels can be created. Reject or flag uncertainty instead of inventing anatomy. Reviewing segmentation does not establish a CVS answer or a surgical lesson answer.

This batch lists {len(bundle['classes'])} anatomy classes. Unreviewed surroundings are unknown, not background negatives. Review does not start training automatically. Technical-only reviews are useful feedback but are not eligible as anatomy-approved supervision.

Keep bundle.json and images/ unchanged. bundle.json SHA-256: {sha256(raw)}
Bundle ID: {bundle['bundleId']}

Dataset source: https://github.com/CAMMA-public/Endoscapes
Dataset license: CC BY-NC-SA 4.0; retain included attribution/license files.
Proposal model: https://github.com/facebookresearch/sam2 (exact provenance in bundle.json).

Use a separate export file per reviewer. Never silently merge people\'s decisions. This is an offline review tool; there is no shared live synchronization or automatic upload.
'''
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.review-package-', dir=output_dir.parent) as temporary:
        stage=Path(temporary)/output_dir.name
        stage.mkdir()
        (stage/'review.html').write_text(html)
        (stage/'bundle.json').write_bytes(raw)
        (stage/'README.md').write_text(readme)
        (stage/'START-HERE.txt').write_text(reviewer_instructions(bundle))
        for relative,data in originals.items():
            target=stage/relative; target.parent.mkdir(parents=True, exist_ok=True);target.write_bytes(data)
        for name,data in licenses.items():
            target=stage/'licenses'/name;target.parent.mkdir(exist_ok=True);target.write_bytes(data)
        inventory=[{'path':p.relative_to(stage).as_posix(),'bytes':p.stat().st_size,'sha256':sha256(p.read_bytes())} for p in sorted(stage.rglob('*')) if p.is_file()]
        report={'formatVersion':'1.0.0','artifactType':'offline_mask_review_handoff','bundleId':bundle['bundleId'],'bundleSha256':sha256(raw),'createdAt':datetime.now(timezone.utc).isoformat(),'imageCount':len(bundle['images']),'caseCount':len({x['videoId'] for x in bundle['images']}),'candidateCount':mask_count,'reviewDecisionsCreated':0,'files':inventory}
        (stage/'handoff-manifest.json').write_text(json.dumps(report,indent=2)+'\n')
        archive=Path(temporary)/(output_dir.name+'.zip')
        with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
            for p in sorted(stage.rglob('*')):
                if p.is_file():z.write(p,arcname=output_dir.name+'/'+p.relative_to(stage).as_posix())
        # Expose only a complete handoff after source validation and ZIP creation.
        stage.rename(output_dir)
        archive.rename(output_dir.with_suffix('.zip'))
    return {**report,'directory':str(output_dir),'zip':str(output_dir.with_suffix('.zip'))}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--template',type=Path)
    parser.add_argument('--license',type=Path,action='append',default=[])
    args=parser.parse_args(argv)
    try:
        report=build_package(args.bundle,args.output_dir,template_path=args.template,license_paths=args.license)
        print(json.dumps({k:v for k,v in report.items() if k!='files'},indent=2))
    except (OSError,ValueError,KeyError,TypeError) as error:
        parser.exit(1,f'Review packaging failed: {error}\n')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
