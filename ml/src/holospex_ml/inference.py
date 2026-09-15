"""Run a trained semantic model and export frame-aligned HUD results.

The PNG is the authoritative predicted label raster. The current HUD contract
supports simple polygons only: components with holes or self-intersections are
withheld, never repaired silently. Confidence is mean softmax over accepted pixels, not calibrated
anatomical correctness or a CVS assessment.
"""

from pathlib import Path
import json
from time import perf_counter

import cv2
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from .adapters import FrameInput
from .validation import validate_frame_result


def _is_simple_polygon(polygon):
    """Reject self-touching/crossing contours before the simple-polygon HUD.

    OpenCV uses 8-connected foreground: two regions touching diagonally can
    produce a contour that revisits a vertex. A contour hierarchy without holes
    does not guarantee a simple boundary, even after approxPolyDP.
    """
    points = [tuple(map(float, point)) for point in polygon]
    count = len(points)
    if count < 3 or len(set(points)) != count:
        return False

    def cross(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def on_segment(a, b, point):
        return (min(a[0], b[0]) <= point[0] <= max(a[0], b[0])
                and min(a[1], b[1]) <= point[1] <= max(a[1], b[1]))

    # Adjacent edges may meet at one endpoint but cannot double back along each
    # other. Collinear vertices that continue forward remain valid.
    for i, b in enumerate(points):
        a, c = points[i - 1], points[(i + 1) % count]
        if cross(a, b, c) == 0 and ((b[0] - a[0]) * (c[0] - b[0])
                                  + (b[1] - a[1]) * (c[1] - b[1])) < 0:
            return False

    for i, a in enumerate(points):
        b = points[(i + 1) % count]
        for j in range(i + 2, count):
            if i == 0 and j == count - 1:
                continue  # First and last edges are adjacent.
            c, d = points[j], points[(j + 1) % count]
            if (max(a[0], b[0]) < min(c[0], d[0]) or max(c[0], d[0]) < min(a[0], b[0])
                    or max(a[1], b[1]) < min(c[1], d[1]) or max(c[1], d[1]) < min(a[1], b[1])):
                continue
            ac, ad, ca, cb = cross(a, b, c), cross(a, b, d), cross(c, d, a), cross(c, d, b)
            if ((ac > 0) != (ad > 0) and (ca > 0) != (cb > 0)
                    and ac != 0 and ad != 0 and ca != 0 and cb != 0):
                return False
            if ((ac == 0 and on_segment(a, b, c)) or (ad == 0 and on_segment(a, b, d))
                    or (ca == 0 and on_segment(c, d, a)) or (cb == 0 and on_segment(c, d, b))):
                return False
    return True


def masks_to_structures(probabilities, classes, threshold=0.5, min_area=64):
    """Convert C×H×W probabilities to conservative single-contour exports."""
    if not 0 <= threshold <= 1 or min_area < 1:
        raise ValueError('threshold must be in [0,1] and min_area must be positive')
    probabilities = np.asarray(probabilities)
    if probabilities.ndim != 3 or not np.isfinite(probabilities).all():
        raise ValueError('Expected finite C×H×W probabilities')
    if len(classes) != probabilities.shape[0]:
        raise ValueError('Checkpoint class count does not match model output')
    labels = probabilities.argmax(axis=0).astype(np.uint8)
    height, width = labels.shape
    structures = []
    withheld = {'small': 0, 'holes': 0, 'degenerate': 0, 'nonSimple': 0}
    for entry in classes:
        class_index = entry['index']
        if entry['structureId'] == 'background':
            continue
        mask = ((labels == class_index) & (probabilities[class_index] >= threshold)).astype(np.uint8)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            continue
        for contour_index, contour in enumerate(contours):
            # A child contour is a hole, not a separate detected structure.
            if hierarchy[0][contour_index][3] != -1:
                continue
            if hierarchy[0][contour_index][2] != -1:
                withheld['holes'] += 1
                continue
            if cv2.contourArea(contour) < min_area:
                withheld['small'] += 1
                continue
            polygon = cv2.approxPolyDP(contour, 0.5, True).reshape(-1, 2)
            if len(polygon) < 3:
                withheld['degenerate'] += 1
                continue
            if not _is_simple_polygon(polygon):
                withheld['nonSimple'] += 1
                continue
            region = np.zeros_like(mask)
            cv2.drawContours(region, [contour], -1, 1, cv2.FILLED)
            accepted = (region != 0) & (mask != 0)
            confidence = float(probabilities[class_index][accepted].mean())
            touches_edge = bool(((polygon[:, 0] == 0) | (polygon[:, 0] == width - 1)
                                | (polygon[:, 1] == 0) | (polygon[:, 1] == height - 1)).any())
            structures.append({
                'instanceId': f"{entry['structureId']}-{contour_index}",
                'structureId': entry['structureId'],
                'polygon': polygon.astype(float).tolist(),
                'confidence': confidence,
                # This is a frame-boundary proxy, not an occlusion estimator.
                'visibility': 'partial' if touches_edge else 'visible',
            })
    return structures, labels, withheld


def _synchronize_device(device):
    """Wait for device work only when explicitly collecting stage timings."""
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elif device.type == 'mps':
        torch.mps.synchronize()


def _record_stage(timings, name, started, device=None):
    if device is not None:
        _synchronize_device(device)
    finished = perf_counter()
    timings[name] = (finished - started) * 1000
    return finished


class SegmentationAdapter:
    def __init__(self, checkpoint_path: Path, device='auto', threshold=0.5, min_area=64,
                 *, run_auxiliary_head=False):
        from .model import load_checkpoint
        self.model, self.checkpoint = load_checkpoint(checkpoint_path, device)
        # Load every checkpoint tensor strictly before removing the unused
        # training head. Keep the full head available for baseline comparisons.
        if not run_auxiliary_head:
            self.model.aux_classifier = None
        self.device = next(self.model.parameters()).device
        self.threshold = threshold
        self.min_area = min_area

    def predict_details(self, frame: FrameInput):
        if frame.image_path is None:
            raise ValueError('Real inference requires image_path')
        with Image.open(frame.image_path) as image:
            rgb = np.array(image.convert('RGB'))
        return self.predict_rgb_details(frame, rgb)

    def predict_rgb_details(self, frame: FrameInput, rgb, *, timings=None):
        """Infer a decoded RGB frame, optionally recording synchronized stage ms.

        Timings exclude decoding and caller work. Profiling waits at accelerator
        boundaries; normal inference adds no explicit synchronization calls.
        """
        if timings is not None:
            _synchronize_device(self.device)
            started = perf_counter()
        rgb = np.asarray(rgb)
        if rgb.dtype != np.uint8 or rgb.shape != (frame.height, frame.width, 3):
            raise ValueError('RGB input must be uint8 H×W×3 matching frame dimensions')
        size = self.checkpoint['input_size']
        resized = Image.fromarray(rgb).resize((size['width'], size['height']), Image.Resampling.BILINEAR)
        pixels = np.array(resized, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0)
        normalization = self.checkpoint['normalization']
        mean = torch.tensor(normalization['mean']).view(1, 3, 1, 1)
        std = torch.tensor(normalization['std']).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std
        if timings is not None:
            started = _record_stage(timings, 'preprocess_ms', started)
        tensor = tensor.to(self.device)
        if timings is not None:
            started = _record_stage(timings, 'input_transfer_ms', started, self.device)
        with torch.inference_mode():
            logits = self.model(tensor)['out']
            if timings is not None:
                started = _record_stage(timings, 'forward_ms', started, self.device)
            # Stretch preprocessing is inverted BEFORE argmax/contour extraction.
            logits = F.interpolate(logits, size=(frame.height, frame.width), mode='bilinear', align_corners=False)
            probabilities = logits.softmax(dim=1)[0]
            if timings is not None:
                started = _record_stage(timings, 'output_resize_softmax_ms', started, self.device)
            probabilities = probabilities.cpu().numpy()
            if timings is not None:
                started = _record_stage(timings, 'output_transfer_ms', started, self.device)
        structures, labels, withheld = masks_to_structures(
            probabilities, self.checkpoint['classes'], self.threshold, self.min_area,
        )
        result = {
            'schemaVersion': '1.0.0', 'mediaId': frame.media_id,
            'frameNumber': frame.frame_number, 'timestampMs': frame.timestamp_ms,
            'width': frame.width, 'height': frame.height, 'coordinateSpace': 'original_pixels',
            'source': 'ml_prediction', 'status': 'ok',
            'model': {'id': self.checkpoint['model_id'], 'version': self.checkpoint['model_version']},
            'structures': structures,
        }
        if timings is not None:
            started = _record_stage(timings, 'geometry_ms', started)
        validate_frame_result(result)
        if timings is not None:
            _record_stage(timings, 'validation_ms', started)
        return result, labels, withheld

    def predict(self, frame: FrameInput):
        return self.predict_details(frame)[0]


def export_prediction(adapter: SegmentationAdapter, frame: FrameInput, output: Path):
    """Keep raw argmax raster and explicit export limitations alongside HUD JSON."""
    mask_path = output.with_suffix('.mask.png')
    info_path = output.with_suffix('.info.json')
    for destination in (output, mask_path, info_path):
        if destination.exists():
            raise FileExistsError(f'Refusing to replace {destination}')
    result, labels, withheld = adapter.predict_details(frame)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(labels).save(mask_path)
    info = {
        'classes': adapter.checkpoint['classes'], 'rawArgmaxMask': mask_path.name,
        'threshold': adapter.threshold, 'minimumContourAreaPixels': adapter.min_area,
        'confidenceDefinition': 'Mean uncalibrated class softmax over accepted component pixels.',
        'polygonLimitations': 'Pixel-center contours simplified by 0.5px; components with holes or non-simple/self-touching boundaries withheld. PNG retains raw argmax without thresholding.',
        'visibilityDefinition': 'partial means contour intersects image border; no occlusion estimator is implemented.',
        'withheldComponents': withheld,
        'training': adapter.checkpoint.get('training', {}),
    }
    info_path.write_text(json.dumps(info, indent=2, allow_nan=False) + '\n')
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result
