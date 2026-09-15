import { videoCaptureSize } from "./uploadedVideoSession";

export const MAX_UPLOADED_IMAGE_BYTES = 20 * 1024 * 1024;
export const MAX_UPLOADED_IMAGE_PIXELS = 40_000_000;
const MAX_IMAGE_DIMENSION = 16_384;
class ImageInputError extends Error {}
const cancelled = () => new DOMException("Image loading was cancelled.", "AbortError");
const checkAbort = (signal: AbortSignal) => { if (signal.aborted) throw cancelled(); };

/** Race decoding with replacement/removal; release even a decoder which ignores abort. */
function abortable<T>(promise: Promise<T>, signal: AbortSignal, release?: (value: T) => void): Promise<T> {
  return new Promise((resolve, reject) => {
    let finished = false;
    const abort = () => { if (!finished) { finished = true; signal.removeEventListener("abort", abort); reject(cancelled()); } };
    signal.addEventListener("abort", abort, { once: true });
    if (signal.aborted) abort();
    void promise.then(value => {
      if (finished) { release?.(value); return; }
      finished = true; signal.removeEventListener("abort", abort); resolve(value);
    }, cause => {
      if (finished) return;
      finished = true; signal.removeEventListener("abort", abort); reject(cause);
    });
  });
}

export function uploadedImageSize(width: number, height: number) {
  if (!Number.isSafeInteger(width) || !Number.isSafeInteger(height) || width < 1 || height < 1
    || width > MAX_IMAGE_DIMENSION || height > MAX_IMAGE_DIMENSION || width * height > MAX_UPLOADED_IMAGE_PIXELS)
    throw new ImageInputError("This image is too large to decode safely. Use an image up to 40 megapixels and 16,384 pixels per side.");
  return videoCaptureSize(width, height)!;
}

async function imageMime(file: File, signal: AbortSignal) {
  if (!file.size) throw new ImageInputError("This image is empty. Choose a JPEG, PNG or WebP image.");
  if (file.size > MAX_UPLOADED_IMAGE_BYTES) throw new ImageInputError("This image exceeds 20 MiB. Choose a smaller image.");
  if (/\.hei[cf]$/i.test(file.name) || /^image\/hei[cf]/i.test(file.type))
    throw new ImageInputError("HEIC and HEIF images are not supported. Convert the image to JPEG or PNG first.");
  const bytes = new Uint8Array(await abortable(file.slice(0, 12).arrayBuffer(), signal));
  if (bytes[0] === 255 && bytes[1] === 216 && bytes[2] === 255) return "image/jpeg";
  if ([137, 80, 78, 71, 13, 10, 26, 10].every((value, index) => bytes[index] === value)) return "image/png";
  if (String.fromCharCode(...bytes.slice(0, 4)) === "RIFF" && String.fromCharCode(...bytes.slice(8, 12)) === "WEBP") return "image/webp";
  throw new ImageInputError("Unsupported image format. Choose a JPEG, PNG or WebP image; convert HEIC or HEIF to JPEG or PNG.");
}

/** The oriented, opaque preview canvas is also the exact source sent as a JPEG. */
export async function decodeUploadedImage(file: File, signal: AbortSignal) {
  checkAbort(signal);
  let release = () => {};
  try {
    const mime = await imageMime(file, signal);
    checkAbort(signal);
    const blob = file.slice(0, file.size, mime);
    let source: CanvasImageSource, width: number, height: number;
    if (typeof createImageBitmap === "function") {
      const bitmap = await abortable(createImageBitmap(blob, { imageOrientation: "from-image" }), signal, value => value.close());
      release = () => bitmap.close(); source = bitmap; width = bitmap.width; height = bitmap.height;
    } else {
      // HTML image decoding applies the image's EXIF orientation before drawing.
      const image = new Image(), url = URL.createObjectURL(blob);
      release = () => { image.removeAttribute("src"); URL.revokeObjectURL(url); };
      image.src = url;
      await abortable(image.decode(), signal);
      source = image; width = image.naturalWidth; height = image.naturalHeight;
    }
    checkAbort(signal);
    const size = uploadedImageSize(width, height);
    const image = document.createElement("canvas"); image.width = size.width; image.height = size.height;
    const context = image.getContext("2d");
    if (!context) throw new ImageInputError("Image rendering is unavailable in this browser.");
    context.fillStyle = "#000"; context.fillRect(0, 0, size.width, size.height);
    context.imageSmoothingEnabled = true; context.imageSmoothingQuality = "high";
    context.drawImage(source, 0, 0, size.width, size.height);
    return { image, ...size };
  } catch (cause) {
    checkAbort(signal);
    if (cause instanceof ImageInputError) throw cause;
    throw new Error("This image could not be decoded. Convert it to JPEG, PNG or WebP and try again.");
  } finally { release(); }
}
