import type { AnatomyId } from "@holospex/contracts";
import type { ResultSource } from "./frameInput";
import type { DisplayedFrame } from "./selectFrame";

export type HudMode = "learn" | "identify" | "assess" | "feedback";
export type HudExperience = "video" | "model";
export interface HudState {
  mode: HudMode;
  experience: HudExperience;
  source: ResultSource;
  overlaysRequested: boolean;
  mediaId: string | null;
  displayedFrame: DisplayedFrame | null;
  /** Capture before asynchronous work. Invalidated on input/source/mode changes. */
  revision: number;
}
export const initialHudState: HudState = {
  mode: "learn", experience: "video", source: "reviewed_annotation",
  overlaysRequested: true, mediaId: null, displayedFrame: null, revision: 0,
};
export type HudAction =
  | { type: "mode"; mode: HudMode }
  | { type: "experience"; experience: HudExperience }
  | { type: "source"; source: ResultSource }
  | { type: "overlays"; visible: boolean }
  | { type: "media"; mediaId: string | null }
  | { type: "invalidate" }
  | { type: "frame"; frame: DisplayedFrame; revision: number };

export function hudReducer(state: HudState, action: HudAction): HudState {
  switch (action.type) {
    case "overlays": return { ...state, overlaysRequested: action.visible };
    case "mode": return { ...state, mode: action.mode, displayedFrame: null, revision: state.revision + 1 };
    case "source": return { ...state, source: action.source, displayedFrame: null, revision: state.revision + 1 };
    case "experience": return { ...state, experience: action.experience, displayedFrame: null, revision: state.revision + 1 };
    case "media": return { ...state, mediaId: action.mediaId, displayedFrame: null, revision: state.revision + 1 };
    case "invalidate": return { ...state, displayedFrame: null, revision: state.revision + 1 };
    case "frame": {
      const frame = action.frame;
      if (action.revision !== state.revision || state.experience !== "video" || frame.mediaId !== state.mediaId
        || !Number.isInteger(frame.frameNumber) || frame.frameNumber < 0
        || !Number.isFinite(frame.timestampMs) || frame.timestampMs < 0
        || !Number.isInteger(frame.width) || frame.width <= 0
        || !Number.isInteger(frame.height) || frame.height <= 0) return state;
      return { ...state, displayedFrame: { ...frame } };
    }
  }
}

/** Identify/Assess override show requests. Feedback requires independently supplied annotations. */
export function hudOverlaysVisible(state: HudState): boolean {
  if (!state.overlaysRequested || state.mode === "identify" || state.mode === "assess") return false;
  return state.mode !== "feedback" || state.experience === "model"
    || state.source === "reviewed_annotation" || state.source === "synthetic_mock";
}

export interface VideoAnatomySelection {
  experience: "video";
  frame: DisplayedFrame;
  source: ResultSource;
  revision: number;
  /** Null means a click outside a supplied region. No learner correctness is implied. */
  structureId: AnatomyId | null;
  instanceId: string | null;
  point: [number, number];
}
export interface ModelAnatomySelection {
  experience: "model";
  registrationId: string;
  anchorId: string;
  structureId: AnatomyId;
}
export type AnatomySelection = VideoAnatomySelection | ModelAnatomySelection;

export function isCurrentSelection(state: HudState, selection: VideoAnatomySelection): boolean {
  const frame = state.displayedFrame;
  return state.experience === "video" && frame !== null && selection.revision === state.revision
    && selection.source === state.source
    && selection.frame.mediaId === frame.mediaId && selection.frame.frameNumber === frame.frameNumber
    && selection.frame.timestampMs === frame.timestampMs
    && selection.frame.width === frame.width && selection.frame.height === frame.height
    && selection.point.every(Number.isFinite)
    && selection.point[0] >= 0 && selection.point[0] <= frame.width
    && selection.point[1] >= 0 && selection.point[1] <= frame.height;
}

/** Milliseconds in the frontend API; seconds only at the HTML media boundary. */
export function seekVideo(video: Pick<HTMLVideoElement, "duration" | "currentTime" | "readyState">, timestampMs: number) {
  if (!Number.isFinite(timestampMs) || timestampMs < 0) throw new Error("Seek time must be finite, nonnegative milliseconds.");
  if (video.readyState < 1 || !Number.isFinite(video.duration)) throw new Error("Load a seekable video first.");
  video.currentTime = Math.min(timestampMs / 1000, video.duration);
}
