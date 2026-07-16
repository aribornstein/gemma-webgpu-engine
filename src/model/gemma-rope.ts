export interface GemmaRotaryRow {
  cosine: Float32Array;
  sine: Float32Array;
}

export interface GemmaRotaryRows {
  sliding: GemmaRotaryRow;
  full: GemmaRotaryRow;
}

export interface GemmaRotaryBlock {
  sliding: GemmaRotaryRow;
  full: GemmaRotaryRow;
  rowCount: number;
}

export function createGemmaRotaryRows(position: number): GemmaRotaryRows {
  if (!Number.isInteger(position) || position < 0) {
    throw new Error("Gemma RoPE position must be a non-negative integer");
  }
  return {
    sliding: createRow(position, 256, 10_000, 128),
    full: createRow(position, 512, 1_000_000, 64),
  };
}

export function createGemmaRotaryBlock(
  startPosition: number,
  rowCount: number,
): GemmaRotaryBlock {
  if (!Number.isInteger(startPosition) || startPosition < 0 ||
      !Number.isInteger(rowCount) || rowCount < 1) {
    throw new Error("Gemma RoPE block requires a non-negative start and positive row count");
  }
  const sliding = {
    cosine: new Float32Array(rowCount * 128),
    sine: new Float32Array(rowCount * 128),
  };
  const full = {
    cosine: new Float32Array(rowCount * 256),
    sine: new Float32Array(rowCount * 256),
  };
  for (let row = 0; row < rowCount; row += 1) {
    const rotary = createGemmaRotaryRows(startPosition + row);
    sliding.cosine.set(rotary.sliding.cosine, row * 128);
    sliding.sine.set(rotary.sliding.sine, row * 128);
    full.cosine.set(rotary.full.cosine, row * 256);
    full.sine.set(rotary.full.sine, row * 256);
  }
  return { sliding, full, rowCount };
}

function createRow(
  position: number,
  headDim: number,
  theta: number,
  rotaryPairs: number,
): GemmaRotaryRow {
  const halfDim = headDim / 2;
  const cosine = new Float32Array(halfDim);
  const sine = new Float32Array(halfDim);
  for (let pair = 0; pair < halfDim; pair += 1) {
    if (pair >= rotaryPairs) {
      cosine[pair] = 1;
      sine[pair] = 0;
      continue;
    }
    const exponent = Math.fround(Math.fround(2 * pair) / Math.fround(headDim));
    const frequency = Math.fround(1 / Math.pow(theta, exponent));
    const angle = Math.fround(Math.fround(position) * frequency);
    cosine[pair] = Math.fround(Math.cos(angle));
    sine[pair] = Math.fround(Math.sin(angle));
  }
  return { cosine, sine };
}