import { useCallback, useEffect, useRef, useState } from "react";

/**
 * Input adapter only: obtaining camera pixels does not locate anatomy.
 * The future marker tracker consumes this stream and produces its own poses;
 * surgical-video segmentation must not be applied to this stream by default.
 */
export function useCamera() {
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const activeStream = useRef<MediaStream | null>(null);
  const requestGeneration = useRef(0);

  const stop = useCallback(() => {
    requestGeneration.current += 1;
    activeStream.current?.getTracks().forEach((track) => track.stop());
    activeStream.current = null;
    setStream(null);
    setStarting(false);
  }, []);

  const start = useCallback(async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      setError("Camera access needs HTTPS or localhost and a supported browser.");
      return;
    }
    const generation = ++requestGeneration.current;
    setError(null);
    setStarting(true);
    try {
      const next = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: "environment" } },
        audio: false,
      });
      // A permission prompt may resolve after the user leaves this mode.
      if (generation !== requestGeneration.current) {
        next.getTracks().forEach((track) => track.stop());
        return;
      }
      activeStream.current?.getTracks().forEach((track) => track.stop());
      activeStream.current = next;
      setStream(next);
    } catch (cause) {
      if (generation === requestGeneration.current) {
        setError(cause instanceof Error ? cause.message : "Camera could not be started.");
      }
    } finally {
      if (generation === requestGeneration.current) setStarting(false);
    }
  }, []);

  useEffect(() => () => {
    requestGeneration.current += 1;
    activeStream.current?.getTracks().forEach((track) => track.stop());
  }, []);

  return { stream, starting, error, start, stop };
}
