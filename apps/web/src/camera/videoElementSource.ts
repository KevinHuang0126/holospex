import { isHlsSource, type VideoSourceFormat } from "../input/videoSource";

/** Attach browser-decoded media only. URLs never pass through the identification server. */
export function attachVideoElementSource(element: HTMLVideoElement, options: {
  src: string;
  sourceKind: "upload" | "url";
  format: VideoSourceFormat;
  onLive(live: boolean): void;
  onInterrupted(): void;
  onError(message: string): void;
  loadHls?: () => Promise<typeof import("hls.js")>;
}) {
  let disposed = false, hls: import("hls.js").default | null = null;
  const fail = (message: string) => { if (!disposed) options.onError(message); };
  if (options.sourceKind === "url") element.crossOrigin = "anonymous";
  else element.removeAttribute("crossorigin");
  const native = () => {
    if (disposed) return;
    element.src = options.src;
    element.load();
  };
  if (options.sourceKind !== "url" || !isHlsSource(options.src, options.format)) native();
  else void (async () => {
    try {
      const { default: Hls } = await (options.loadHls ?? (() => import("hls.js")))();
      if (disposed) return;
      // Prefer the library when MSE is supported; native canPlayType alone can overclaim HLS support.
      if (Hls.isSupported()) {
        hls = new Hls({ liveDurationInfinity: true,
          xhrSetup: xhr => { xhr.withCredentials = false; },
          fetchSetup: (context, init) => new Request(context.url, { ...init, credentials: "omit" }),
        });
        hls.on(Hls.Events.LEVEL_LOADED, (_, data) => { if (!disposed && hls) options.onLive(data.details.live); });
        hls.on(Hls.Events.ERROR, (_, data) => {
          if (disposed || !hls) return;
          if (data.fatal) {
            hls?.destroy(); hls = null;
            fail("The HLS feed could not be played. Check that the stream is available and permits cross-origin access, then reconnect.");
          } else options.onInterrupted();
        });
        hls.loadSource(options.src); hls.attachMedia(element);
      } else if (element.canPlayType("application/vnd.apple.mpegurl")) native();
      else fail("HLS playback is unavailable in this browser. Try a supported browser or a direct MP4 video URL.");
    } catch {
      hls?.destroy(); hls = null;
      fail("The HLS player could not start. Check your connection and reconnect the video source.");
    }
  })();
  return () => { disposed = true; hls?.destroy(); hls = null; };
}
