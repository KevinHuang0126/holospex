/* Generated from contracts/schemas. Run npm run contracts:generate; do not edit. */

/**
 * One source result for one original media frame. No result implies no overlay; no detection does not imply anatomical absence.
 */
export type FrameResult = {
  [k: string]: unknown;
} & {
  schemaVersion: '1.0.0';
  mediaId: string;
  frameNumber: number;
  timestampMs: number;
  width: number;
  height: number;
  coordinateSpace: 'original_pixels';
  source: 'synthetic_mock' | 'reviewed_annotation' | 'ml_prediction' | 'propagated_prediction';
  status: 'ok' | 'unsupported' | 'missing' | 'error';
  statusReason?: string;
  model?: {
    id: string;
    version: string;
  };
  propagatedFromTimestampMs?: number;
  structures: {
    instanceId: string;
    structureId:
      'gallbladder' | 'cystic_duct' | 'cystic_artery' | 'cystic_plate' | 'hepatocystic_triangle_dissection' | 'tool';
    /**
     * @minItems 3
     */
    polygon: [[number, number], [number, number], [number, number], ...[number, number][]];
    confidence?: number;
    visibility: 'visible' | 'partial';
  }[];
};
