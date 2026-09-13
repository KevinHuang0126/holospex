import { useCallback, useEffect, useReducer, useRef } from "react";
import { hudReducer, hudOverlaysVisible, initialHudState, isCurrentSelection, seekVideo, type HudExperience, type HudMode, type VideoAnatomySelection } from "./hudControls";
import type { ResultSource } from "./frameInput";
import type { DisplayedFrame } from "./selectFrame";

export function useHudControls(events: {
  frameClock?: "external" | "conservative";
  onDisplayedFrame?: (frame: DisplayedFrame | null) => void;
  onAnatomySelection?: (selection: VideoAnatomySelection) => void;
} = {}) {
  const [state, dispatch] = useReducer(hudReducer, initialHudState);
  const currentState = useRef(state);
  currentState.current = state;
  const [video, setVideo] = useReducer((_: HTMLVideoElement | null, next: HTMLVideoElement | null) => next, null);
  const callbacks = useRef(events);
  callbacks.current = events;
  useEffect(() => { callbacks.current.onDisplayedFrame?.(state.displayedFrame); }, [state.displayedFrame]);
  const invalidate = useCallback(() => dispatch({ type: "invalidate" }), []);
  const setMedia = useCallback((mediaId: string | null) => dispatch({ type: "media", mediaId }), []);
  const setMode = useCallback((mode: HudMode) => dispatch({ type: "mode", mode }), []);
  const setSource = useCallback((source: ResultSource) => dispatch({ type: "source", source }), []);
  const showOverlays = useCallback((visible: boolean) => dispatch({ type: "overlays", visible }), []);
  const reportDisplayedFrame = useCallback((frame: DisplayedFrame, revision: number) => dispatch({ type: "frame", frame, revision }), []);
  const reportAnatomySelection = useCallback((selection: VideoAnatomySelection) => {
    // A renderer may retain this callback across an asynchronous result request.
    if (isCurrentSelection(currentState.current, selection)) callbacks.current.onAnatomySelection?.(selection);
  }, []);
  useEffect(() => {
    if (!video) return;
    // Native controls and errors must also clear identity; timeupdate is not a frame clock.
    const names = events.frameClock === "external"
      ? ["seeking", "emptied", "loadstart", "error"]
      : ["seeking", "emptied", "loadstart", "error", "playing", "timeupdate"];
    names.forEach(name => video.addEventListener(name, invalidate));
    return () => { names.forEach(name => video.removeEventListener(name, invalidate)); video.pause(); };
  }, [video, invalidate, events.frameClock]);

  return {
    state,
    overlaysVisible: hudOverlaysVisible(state),
    bindVideo: setVideo,
    setMedia,
    setMode,
    setSource,
    setExperience: (experience: HudExperience) => { video?.pause(); dispatch({ type: "experience", experience }); },
    showOverlays,
    invalidate,
    play: async () => {
      if (!video || state.experience !== "video" || !state.mediaId) throw new Error("Select a video first.");
      invalidate();
      await video.play();
    },
    pause: () => video?.pause(),
    seek: (timestampMs: number) => {
      if (!video || state.experience !== "video") throw new Error("Select a video first.");
      invalidate();
      seekVideo(video, timestampMs);
    },
    // The future presentation-clock adapter supplies original frame identity.
    // Never infer it from timeupdate, rounded fps, or the callback count.
    reportDisplayedFrame,
    reportAnatomySelection,
  };
}
