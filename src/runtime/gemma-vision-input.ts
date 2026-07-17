export const GEMMA_VISION_PATCH_SIZE = 16;
export const GEMMA_VISION_POOLING_KERNEL = 3;
export const GEMMA_VISION_MAX_SOFT_TOKENS = 280;
export const GEMMA_VISION_TOKEN_BUDGETS = [70, 140, 280] as const;
export type GemmaVisionTokenBudget = typeof GEMMA_VISION_TOKEN_BUDGETS[number];
export const GEMMA_VISION_MAX_PATCHES =
  GEMMA_VISION_MAX_SOFT_TOKENS * GEMMA_VISION_POOLING_KERNEL ** 2;
export const GEMMA_VISION_PATCH_DIMENSION = 3 * GEMMA_VISION_PATCH_SIZE ** 2;

export interface GemmaVisionInput {
  identity?: string;
  patches: Float32Array;
  positions: Int32Array;
  patchRows: number;
  patchColumns: number;
  patchCount: number;
  softTokenCount: number;
}

export type GemmaVisionImageSource = Blob | ImageBitmap | ImageData;

export async function prepareGemmaVisionImage(
  source: GemmaVisionImageSource,
  signal?: AbortSignal,
  tokenBudget: GemmaVisionTokenBudget = GEMMA_VISION_MAX_SOFT_TOKENS,
): Promise<GemmaVisionInput> {
  signal?.throwIfAborted();
  const bitmap = await createImageBitmap(source);
  try {
    signal?.throwIfAborted();
    const [height, width] = gemmaVisionTargetSize(bitmap.height, bitmap.width, tokenBudget);
    const canvas = new OffscreenCanvas(width, height);
    const context = canvas.getContext("2d", { willReadFrequently: true });
    if (!context) throw new Error("Gemma vision image canvas is unavailable");
    context.imageSmoothingEnabled = true;
    context.imageSmoothingQuality = "high";
    context.drawImage(bitmap, 0, 0, width, height);
    const bytes = context.getImageData(0, 0, width, height).data;
    const rgb = new Uint8Array(height * width * 3);
    const pixels = new Float32Array(height * width * 3);
    for (let sourceIndex = 0, destination = 0;
        sourceIndex < bytes.length;
        sourceIndex += 4) {
      rgb[destination] = bytes[sourceIndex];
      pixels[destination++] = bytes[sourceIndex] / 255;
      rgb[destination] = bytes[sourceIndex + 1];
      pixels[destination++] = bytes[sourceIndex + 1] / 255;
      rgb[destination] = bytes[sourceIndex + 2];
      pixels[destination++] = bytes[sourceIndex + 2] / 255;
    }
    signal?.throwIfAborted();
    const identity = await gemmaVisionContentIdentity(rgb, height, width, tokenBudget);
    signal?.throwIfAborted();
    return { ...patchifyGemmaVisionRgb(pixels, height, width), identity };
  } finally {
    bitmap.close();
  }
}

export async function gemmaVisionContentIdentity(
  rgb: Uint8Array,
  height: number,
  width: number,
  tokenBudget: GemmaVisionTokenBudget,
): Promise<string> {
  if (rgb.length !== height * width * 3) {
    throw new Error("Gemma vision identity requires exact HWC RGB bytes");
  }
  validateGemmaVisionTokenBudget(tokenBudget);
  const header = new Uint32Array([height, width, tokenBudget]);
  const payload = new Uint8Array(header.byteLength + rgb.byteLength);
  payload.set(new Uint8Array(header.buffer));
  payload.set(rgb, header.byteLength);
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", payload));
  return Array.from(digest, (byte) => byte.toString(16).padStart(2, "0")).join("");
}

export function gemmaVisionTargetSize(
  height: number,
  width: number,
  tokenBudget: GemmaVisionTokenBudget = GEMMA_VISION_MAX_SOFT_TOKENS,
): [number, number] {
  if (!Number.isInteger(height) || height < 1 || !Number.isInteger(width) || width < 1) {
    throw new Error("Gemma vision image dimensions must be positive integers");
  }
  validateGemmaVisionTokenBudget(tokenBudget);
  const patchBudget = tokenBudget * GEMMA_VISION_POOLING_KERNEL ** 2;
  const targetPixels = patchBudget * GEMMA_VISION_PATCH_SIZE ** 2;
  const factor = Math.sqrt(targetPixels / (height * width));
  const sideMultiple = GEMMA_VISION_POOLING_KERNEL * GEMMA_VISION_PATCH_SIZE;
  let targetHeight = Math.floor(factor * height / sideMultiple) * sideMultiple;
  let targetWidth = Math.floor(factor * width / sideMultiple) * sideMultiple;
  if (targetHeight === 0 && targetWidth === 0) {
    throw new Error("Gemma vision image aspect ratio cannot produce a nonzero target size");
  }
  const maximumSide = Math.floor(
    patchBudget / GEMMA_VISION_POOLING_KERNEL ** 2,
  ) * sideMultiple;
  if (targetHeight === 0) {
    targetHeight = sideMultiple;
    targetWidth = Math.min(Math.floor(width / height) * sideMultiple, maximumSide);
  } else if (targetWidth === 0) {
    targetWidth = sideMultiple;
    targetHeight = Math.min(Math.floor(height / width) * sideMultiple, maximumSide);
  }
  return [targetHeight, targetWidth];
}

export function validateGemmaVisionTokenBudget(
  tokenBudget: number,
): asserts tokenBudget is GemmaVisionTokenBudget {
  if (!(GEMMA_VISION_TOKEN_BUDGETS as readonly number[]).includes(tokenBudget)) {
    throw new Error(
      `Gemma vision token budget must be one of ${GEMMA_VISION_TOKEN_BUDGETS.join(", ")}`,
    );
  }
}

export function patchifyGemmaVisionRgb(
  pixels: Float32Array,
  height: number,
  width: number,
): GemmaVisionInput {
  if (pixels.length !== height * width * 3) {
    throw new Error("Gemma vision input must contain an HWC RGB float tensor");
  }
  if (height % GEMMA_VISION_PATCH_SIZE !== 0 || width % GEMMA_VISION_PATCH_SIZE !== 0) {
    throw new Error("Gemma vision input dimensions must be divisible by the patch size");
  }
  const patchRows = height / GEMMA_VISION_PATCH_SIZE;
  const patchColumns = width / GEMMA_VISION_PATCH_SIZE;
  const patchCount = patchRows * patchColumns;
  if (patchCount > GEMMA_VISION_MAX_PATCHES) {
    throw new Error(`Gemma vision input has ${patchCount} patches; maximum is ${GEMMA_VISION_MAX_PATCHES}`);
  }
  const patches = new Float32Array(GEMMA_VISION_MAX_PATCHES * GEMMA_VISION_PATCH_DIMENSION);
  const positions = new Int32Array(GEMMA_VISION_MAX_PATCHES * 2).fill(-1);
  let destination = 0;
  let position = 0;
  for (let patchRow = 0; patchRow < patchRows; patchRow += 1) {
    for (let patchColumn = 0; patchColumn < patchColumns; patchColumn += 1) {
      for (let row = 0; row < GEMMA_VISION_PATCH_SIZE; row += 1) {
        const sourceRow = patchRow * GEMMA_VISION_PATCH_SIZE + row;
        const sourceStart = (sourceRow * width + patchColumn * GEMMA_VISION_PATCH_SIZE) * 3;
        const sourceEnd = sourceStart + GEMMA_VISION_PATCH_SIZE * 3;
        patches.set(pixels.subarray(sourceStart, sourceEnd), destination);
        destination += GEMMA_VISION_PATCH_SIZE * 3;
      }
      positions[position * 2] = patchColumn;
      positions[position * 2 + 1] = patchRow;
      position += 1;
    }
  }
  return {
    patches,
    positions,
    patchRows,
    patchColumns,
    patchCount,
    softTokenCount: Math.floor(patchCount / GEMMA_VISION_POOLING_KERNEL ** 2),
  };
}