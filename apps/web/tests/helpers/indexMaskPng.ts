import { crc32, deflateSync } from "node:zlib";

export function pngChunk(type: string, data: Uint8Array = new Uint8Array()): Buffer {
  const typeAndData = Buffer.concat([Buffer.from(type, "ascii"), data]);
  const chunk = Buffer.alloc(data.length + 12);
  chunk.writeUInt32BE(data.length, 0); typeAndData.copy(chunk, 4);
  chunk.writeUInt32BE(crc32(typeAndData), chunk.length - 4);
  return chunk;
}

/** Test fixture encoder, deliberately using unfiltered rows and Node's independent CRC. */
export function encodeIndexMaskPng(width: number, height: number, indices: number[] | Uint8Array): Buffer {
  if (indices.length !== width * height) throw new Error("Fixture dimensions do not match indices.");
  const header = Buffer.alloc(13); header.writeUInt32BE(width, 0); header.writeUInt32BE(height, 4); header[8] = 8;
  const rows = Buffer.alloc((width + 1) * height);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) rows[y * (width + 1) + 1 + x] = indices[y * width + x];
  return Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), pngChunk("IHDR", header), pngChunk("IDAT", deflateSync(rows)), pngChunk("IEND")]);
}
