const MAX_MASK_BYTES = 32 * 1024 * 1024;
const PNG_SIGNATURE = [137, 80, 78, 71, 13, 10, 26, 10];
const CRC_TABLE = Uint32Array.from({ length: 256 }, (_, index) => {
  let value = index;
  for (let bit = 0; bit < 8; bit++) value = value & 1 ? 0xedb88320 ^ (value >>> 1) : value >>> 1;
  return value >>> 0;
});
function crc32(bytes: Uint8Array, start: number, end: number) {
  let value = 0xffffffff;
  for (let offset = start; offset < end; offset++) value = CRC_TABLE[(value ^ bytes[offset]) & 255] ^ (value >>> 8);
  return (value ^ 0xffffffff) >>> 0;
}
function abortIfRequested(signal?: AbortSignal) {
  if (signal?.aborted) throw new DOMException("Index mask decoding was cancelled.", "AbortError");
}
function invalid(message: string): never { throw new Error(`Invalid index PNG: ${message}`); }

function readImageData(bytes: Uint8Array<ArrayBuffer>, width: number, height: number, signal?: AbortSignal) {
  if (!PNG_SIGNATURE.every((value, index) => bytes[index] === value)) invalid("signature does not match PNG.");
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const data: Uint8Array<ArrayBuffer>[] = [];
  let offset = 8, header = false, imageData = false, dataEnded = false, ended = false;
  while (offset < bytes.length) {
    abortIfRequested(signal);
    if (bytes.length - offset < 12) invalid("truncated chunk.");
    const length = view.getUint32(offset);
    if (length > 0x7fffffff || length > bytes.length - offset - 12) invalid("chunk length exceeds available bytes.");
    const start = offset + 8, end = start + length;
    const typeBytes = bytes.subarray(offset + 4, start);
    if (!typeBytes.every(value => (value >= 65 && value <= 90) || (value >= 97 && value <= 122)) || (typeBytes[2] & 32)) invalid("unsupported chunk type.");
    const type = String.fromCharCode(...typeBytes);
    if (crc32(bytes, offset + 4, end) !== view.getUint32(end)) invalid(`${type} CRC check failed.`);
    if (!header && type !== "IHDR") invalid("IHDR must be the first chunk.");
    if (type === "IHDR") {
      if (header || length !== 13) invalid("expected one 13-byte IHDR header.");
      if (view.getUint32(start) !== width || view.getUint32(start + 4) !== height) invalid("dimensions disagree with the label record.");
      if (bytes[start + 8] !== 8 || bytes[start + 9] !== 0 || bytes[start + 10] !== 0 || bytes[start + 11] !== 0 || bytes[start + 12] !== 0)
        invalid("only non-interlaced, 8-bit grayscale masks with standard compression and filtering are supported.");
      header = true;
    } else if (type === "IDAT") {
      if (dataEnded) invalid("IDAT chunks must be consecutive.");
      imageData = true; data.push(bytes.subarray(start, end));
    } else if (type === "IEND") {
      if (!imageData || length !== 0 || end + 4 !== bytes.length) invalid("expected a final empty IEND after image data.");
      ended = true;
    } else {
      if (!(typeBytes[0] & 32)) invalid(`unsupported critical chunk ${type}.`);
      if (["tRNS", "acTL", "fcTL", "fdAT"].includes(type)) invalid("transparent or animated masks are unsupported.");
      // Ancillary display metadata (gamma/ICC/EXIF, etc.) never changes class indices.
      if (imageData) dataEnded = true;
    }
    offset = end + 4;
  }
  if (!header || !imageData || !ended) invalid("missing IHDR, IDAT or IEND.");
  return data;
}

async function inflateExact(parts: Uint8Array<ArrayBuffer>[], expected: number, signal?: AbortSignal) {
  abortIfRequested(signal);
  if (typeof DecompressionStream === "undefined") throw new Error("Exact index mask decoding requires a browser with DecompressionStream support.");
  const reader = new Blob(parts).stream().pipeThrough(new DecompressionStream("deflate")).getReader();
  const cancel = () => { void reader.cancel().catch(() => {}); };
  signal?.addEventListener("abort", cancel, { once: true });
  const output = new Uint8Array(expected);
  let length = 0;
  try {
    abortIfRequested(signal);
    while (true) {
      const next = await reader.read();
      abortIfRequested(signal);
      if (next.done) break;
      if (next.value.length > expected - length) invalid("inflated data exceeds the declared mask size.");
      output.set(next.value, length); length += next.value.length;
    }
    if (length !== expected) invalid("inflated data length does not match the declared mask size.");
    return output;
  } catch (cause) {
    abortIfRequested(signal);
    if (cause instanceof Error && cause.message.startsWith("Invalid index PNG:")) throw cause;
    invalid("compressed image data is corrupt.");
  } finally {
    signal?.removeEventListener("abort", cancel);
    try { await reader.cancel(); } catch { /* Preserve the validation/decompression error. */ }
    reader.releaseLock();
  }
}

function paeth(left: number, above: number, upperLeft: number) {
  const estimate = left + above - upperLeft;
  const dl = Math.abs(estimate - left), da = Math.abs(estimate - above), du = Math.abs(estimate - upperLeft);
  return dl <= da && dl <= du ? left : da <= du ? above : upperLeft;
}

/**
 * Read categorical bytes, not display colors. Browser image/canvas decoding may
 * color-convert or perturb near-black label indices, even for a valid PNG.
 * PNG filters and CRC: https://www.w3.org/TR/png-3/#9Filters
 */
export async function decodeIndexMask(blob: Blob, width: number, height: number, signal?: AbortSignal): Promise<Uint8ClampedArray> {
  abortIfRequested(signal);
  if (![width, height].every(value => Number.isInteger(value) && value > 0 && value <= 8192) || width * height > 16_000_000)
    invalid("mask dimensions exceed the supported limits.");
  if (!Number.isFinite(blob.size) || blob.size < 57 || blob.size > MAX_MASK_BYTES) invalid("mask file must be a PNG smaller than 32 MB.");
  const bytes = new Uint8Array(await blob.arrayBuffer());
  abortIfRequested(signal);
  const parts = readImageData(bytes, width, height, signal);
  const filtered = await inflateExact(parts, (width + 1) * height, signal);
  const rgba = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++) {
    abortIfRequested(signal);
    const row = y * (width + 1), filter = filtered[row];
    if (filter > 4) invalid(`unsupported scanline filter ${filter}.`);
    for (let x = 0; x < width; x++) {
      const output = (y * width + x) * 4;
      const left = x ? rgba[output - 4] : 0;
      const above = y ? rgba[output - width * 4] : 0;
      const upperLeft = x && y ? rgba[output - width * 4 - 4] : 0;
      const predictor = filter === 0 ? 0 : filter === 1 ? left : filter === 2 ? above : filter === 3 ? Math.floor((left + above) / 2) : paeth(left, above, upperLeft);
      const value = (filtered[row + 1 + x] + predictor) & 255;
      rgba[output] = value; rgba[output + 1] = value; rgba[output + 2] = value; rgba[output + 3] = 255;
    }
  }
  return rgba;
}
