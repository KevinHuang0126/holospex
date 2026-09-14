type Point = readonly [number, number];

function edgeDistanceSquared(point: Point, a: Point, b: Point) {
  const dx = b[0] - a[0], dy = b[1] - a[1], length = dx * dx + dy * dy;
  const t = length ? Math.max(0, Math.min(1, ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / length)) : 0;
  return (point[0] - a[0] - t * dx) ** 2 + (point[1] - a[1] - t * dy) ** 2;
}

/** A pointer must terminate inside its own contour. Vertex averages can land
 * in background for curved ducts and arteries. Sample interior scan-line spans,
 * then prefer clearance from the boundary. This is a bounded label-placement
 * heuristic, not a change to the segmentation or an anatomical landmark.
 */
export function polygonAnchor(polygon: readonly Point[]): { x: number; y: number } | null {
  if (polygon.length < 3 || polygon.some(point => !point.every(Number.isFinite))) return null;
  let minY = Infinity, maxY = -Infinity, minX = Infinity, maxX = -Infinity;
  for (const [x, y] of polygon) { minX = Math.min(minX, x); maxX = Math.max(maxX, x); minY = Math.min(minY, y); maxY = Math.max(maxY, y); }
  if (minX === maxX || minY === maxY) return null;
  let best: Point | null = null, bestDistance = 0, bestCenterDistance = Infinity;
  const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
  // The center is visited first for consistent ties. A fixed number of rows
  // avoids image-size-dependent work on every video repaint.
  for (const fraction of [0.5, ...Array.from({ length: 16 }, (_, i) => (i + 0.5) / 16)]) {
    const y = minY + (maxY - minY) * fraction, crossings: number[] = [];
    for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
      const a = polygon[j], b = polygon[i];
      if ((a[1] > y) !== (b[1] > y)) crossings.push(a[0] + (y - a[1]) * (b[0] - a[0]) / (b[1] - a[1]));
    }
    crossings.sort((a, b) => a - b);
    // Only evaluate the widest inside span on this row, keeping the boundary
    // distance calculation O(vertices), even with many concave notches.
    let left = 0, right = 0;
    for (let i = 0; i + 1 < crossings.length; i += 2) {
      const span = crossings[i + 1] - crossings[i];
      if (span > right - left || (span === right - left && Math.abs((crossings[i + 1] + crossings[i]) / 2 - cx) < Math.abs((right + left) / 2 - cx))) {
        left = crossings[i]; right = crossings[i + 1];
      }
    }
    if (right <= left) continue;
    const candidate: Point = [(left + right) / 2, y];
    let distance = Infinity;
    for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) distance = Math.min(distance, edgeDistanceSquared(candidate, polygon[j], polygon[i]));
    const centerDistance = (candidate[0] - cx) ** 2 + (candidate[1] - cy) ** 2;
    if (distance > bestDistance || (distance === bestDistance && distance > 0 && centerDistance < bestCenterDistance)) {
      best = candidate; bestDistance = distance; bestCenterDistance = centerDistance;
    }
  }
  return best ? { x: best[0], y: best[1] } : null;
}
