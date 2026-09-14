export type VideoSourceFormat = "auto" | "hls" | "video";

const hostedAt = (hostname: string, domain: string) => hostname === domain || hostname.endsWith(`.${domain}`);

/** Validate a browser media address without rewriting signed paths or query strings. */
export function parseVideoSourceUrl(value: string): string {
  const source = value.trim();
  if (!source) throw new Error("Enter a direct HTTPS video or HLS stream URL.");
  if (/[\u0000-\u001f\u007f]/.test(source)) throw new Error("The video URL contains unsupported control characters.");
  let url: URL;
  try { url = new URL(source); }
  catch { throw new Error("Enter a complete HTTPS video URL, including https:// and the host name."); }
  if (url.protocol !== "https:" || !/^https:\/\//i.test(source))
    throw new Error("Video streams must use a complete HTTPS URL to work in the phone app.");
  if (url.username || url.password)
    throw new Error("Video URLs containing a username or password are unsupported. Use a direct media link without embedded credentials.");

  const hostname = url.hostname.toLowerCase().replace(/\.$/, "");
  const youtubePage = hostedAt(hostname, "youtu.be")
    || ((hostedAt(hostname, "youtube.com") || hostedAt(hostname, "youtube-nocookie.com"))
      && /^\/(?:$|watch(?:\/|$)|shorts(?:\/|$)|live(?:\/|$)|embed(?:\/|$)|playlist(?:\/|$))/.test(url.pathname));
  const vimeoPage = hostedAt(hostname, "vimeo.com")
    && /^\/(?:\d+(?:\/|$)|video\/\d+(?:\/|$)|channels\/[^/]+\/\d+(?:\/|$)|groups\/[^/]+\/videos\/\d+(?:\/|$)|ondemand(?:\/|$))/.test(url.pathname);
  if (youtubePage || vimeoPage)
    throw new Error("This is a video watch or share page. Use the direct HTTPS media or HLS stream URL so anatomy can be identified from its frames.");
  return source;
}

/** Explicit format selection supports extensionless feeds; queries never determine format. */
export function isHlsSource(src: string, format: VideoSourceFormat): boolean {
  if (format !== "auto") return format === "hls";
  try { return /\.m3u8$/i.test(new URL(src).pathname); }
  catch { return false; }
}
