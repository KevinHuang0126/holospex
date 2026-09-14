import assert from "node:assert/strict";
import test from "node:test";
import { isHlsSource, parseVideoSourceUrl } from "../src/input/videoSource";

test("direct video addresses preserve signed paths and query encoding exactly", () => {
  const signed = "https://Media.Example:443/surgery%2Fcase/clip.mp4?key=a%2Bb+c&sig=%2fABC%3d&part=2&part=1#t=3";
  assert.equal(parseVideoSourceUrl(`  ${signed}\n`), signed);
  for (const source of ["https://example.test", "https://example.test/feed/session?token=abc", "HTTPS://example.test/live.M3U8"])
    assert.equal(parseVideoSourceUrl(source), source);
});

test("invalid and non-HTTPS addresses fail without exposing their values", () => {
  for (const source of ["", "   ", "example.test/feed", "/relative.mp4", "https:/example.test/feed",
    "http://example.test/video", "rtsp://example.test/video", "data:video/mp4;base64,secret",
    "blob:https://example.test/video", "javascript:alert(1)", "https://", "https://example.test/\nclip.mp4",
    "https://example.test/\u0000clip.mp4", "https://example.test/\u007fclip.mp4"]) {
    assert.throws(() => parseVideoSourceUrl(source), error => {
      assert.ok(error instanceof Error);
      if (source.trim().length > 12) assert.ok(!error.message.includes(source.trim()));
      return true;
    });
  }
  for (const credentials of ["user:secret", "user", ":secret", "user%40example.test:secret"]) {
    assert.throws(() => parseVideoSourceUrl(`https://${credentials}@example.test/video.mp4`), /embedded credentials/);
  }
});

test("watch/share pages explain direct media is required without blocking lookalike hosts or direct media", () => {
  for (const source of ["https://www.youtube.com/watch?v=123", "https://m.youtube.com/shorts/123",
    "https://youtube.com/live/123", "https://youtube-nocookie.com/embed/123", "https://youtu.be/123?t=4",
    "https://www.youtube.com./watch?v=123", "https://vimeo.com/12345", "https://vimeo.com/12345/unlisted",
    "https://player.vimeo.com/video/12345", "https://vimeo.com/channels/staffpicks/12345",
    "https://vimeo.com/groups/shortfilms/videos/12345", "https://vimeo.com/ondemand/example"]) {
    assert.throws(() => parseVideoSourceUrl(source), /watch or share page/);
  }
  for (const source of ["https://youtube.com.example.test/watch", "https://notvimeo.com/video/12345",
    "https://example.test/watch/stream", "https://player.vimeo.com/progressive_redirect/playback/12345/file.mp4?token=x",
    "https://cdn.vimeo.com/media/clip.mp4"]) assert.equal(parseVideoSourceUrl(source), source);
});

test("HLS auto detection uses the pathname and explicit format handles extensionless or misleading URLs", () => {
  assert.equal(isHlsSource("https://example.test/live.m3u8?sig=ABC#player", "auto"), true);
  assert.equal(isHlsSource("https://example.test/LIVE.M3U8", "auto"), true);
  assert.equal(isHlsSource("https://example.test/video.mp4?name=live.m3u8", "auto"), false);
  assert.equal(isHlsSource("https://example.test/stream?format=m3u8", "auto"), false);
  assert.equal(isHlsSource("https://example.test/stream", "hls"), true);
  assert.equal(isHlsSource("https://example.test/video.mp4", "hls"), true);
  assert.equal(isHlsSource("https://example.test/live.m3u8", "video"), false);
  assert.equal(isHlsSource("not a URL", "auto"), false);
});
