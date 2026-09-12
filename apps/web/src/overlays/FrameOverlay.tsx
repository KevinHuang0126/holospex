import { anatomy, type FrameResult, type Lesson } from "@holospex/contracts";
import { matchesDisplayedFrame } from "./selectFrame";

type Checkpoint = Lesson["checkpoints"][number];

/**
 * This renderer consumes ORIGINAL 2D frame pixels. The shared SVG viewBox scales
 * the image and polygons together. These are not marker/world coordinates.
 * Real video must select the correct checkpoint/result from its playback clock;
 * never retain the previous frame's polygons while awaiting a new prediction.
 */
export function FrameOverlay({ lesson, checkpoint, result, visible, onImageLoad, onImageError }: {
  lesson: Lesson;
  checkpoint: Checkpoint;
  result: FrameResult | null;
  visible: boolean;
  onImageLoad: () => void;
  onImageError: () => void;
}) {
  const aligned = matchesDisplayedFrame(result, {
    mediaId: lesson.media.id,
    frameNumber: checkpoint.frameNumber,
    timestampMs: checkpoint.timestampMs,
    width: lesson.media.width,
    height: lesson.media.height,
  });

  return (
    <>
      <svg viewBox={`0 0 ${lesson.media.width} ${lesson.media.height}`} role="img" aria-label="Synthetic lesson illustration with optional anatomy labels">
        <image href={checkpoint.imagePath} width={lesson.media.width} height={lesson.media.height} onLoad={onImageLoad} onError={onImageError} />
        {visible && aligned && result && result.structures.map((structure) => {
          const entry = anatomy[structure.structureId as keyof typeof anatomy];
          const x = structure.polygon.reduce((sum, point) => sum + point[0], 0) / structure.polygon.length;
          const y = structure.polygon.reduce((sum, point) => sum + point[1], 0) / structure.polygon.length;
          return <g key={structure.instanceId}>
            <polygon points={structure.polygon.map((point) => point.join(",")).join(" ")} fill={entry?.color ?? "#ffffff"} fillOpacity="0.22" stroke={entry?.color ?? "#ffffff"} strokeWidth="3" />
            <text x={x} y={y} textAnchor="middle" className="anatomy-label">{entry?.label ?? structure.structureId}</text>
          </g>;
        })}
      </svg>
      {visible && result && !aligned && <p className="overlay-warning" role="status">Overlay unavailable for this frame. {result.statusReason}</p>}
    </>
  );
}
