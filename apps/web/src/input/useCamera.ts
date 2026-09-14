import { useCallback, useEffect, useRef, useState } from "react";

/**
 * Input adapter only: obtaining camera pixels does not locate anatomy.
 * Marker tracking and live surgical identification consume streams separately.
 * A selected USB capture device can supply the surgical feed.
 */
export function useCamera() {
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const [interrupted, setInterrupted] = useState(false);
  const activeStream = useRef<MediaStream | null>(null);
  const requestGeneration = useRef(0);

  const stop = useCallback(() => {
    requestGeneration.current += 1;
    activeStream.current?.getTracks().forEach((track) => track.stop());
    activeStream.current = null;
    setStream(null);
    setStarting(false);
    setInterrupted(false);
  }, []);

  const start = useCallback(async (deviceId?: string, constraints: MediaTrackConstraints = {}) => {
    if (!navigator.mediaDevices?.getUserMedia) {
      setError("Camera access needs HTTPS or localhost and a supported browser.");
      return;
    }
    const generation = ++requestGeneration.current;
    setError(null);
    setStarting(true);
    try {
      const next = await navigator.mediaDevices.getUserMedia({
        video: { ...constraints, ...(deviceId ? { deviceId: { exact: deviceId } } : { facingMode: { ideal: "environment" } }) },
        audio: false,
      });
      // A permission prompt may resolve after the user leaves this mode.
      if (generation !== requestGeneration.current) {
        next.getTracks().forEach((track) => track.stop());
        return;
      }
      activeStream.current?.getTracks().forEach((track) => track.stop());
      activeStream.current = next;
      setInterrupted(false);
      for (const track of next.getVideoTracks()) {
        track.addEventListener("mute", () => {
          if (activeStream.current === next) setInterrupted(true);
        });
        track.addEventListener("unmute", () => {
          if (activeStream.current === next) setInterrupted(false);
        });
        track.addEventListener("ended", () => {
          if (activeStream.current !== next) return;
          stop();
          setError("Camera disconnected. Reconnect it and start the camera again.");
        });
      }
      setStream(next);
    } catch (cause) {
      if (generation === requestGeneration.current) {
        setError(cause instanceof Error ? cause.message : "Camera could not be started.");
      }
    } finally {
      if (generation === requestGeneration.current) setStarting(false);
    }
  }, [stop]);

  useEffect(() => () => {
    requestGeneration.current += 1;
    activeStream.current?.getTracks().forEach((track) => track.stop());
  }, []);

  return { stream, starting, interrupted, error, start, stop };
}
