const AXES = 2;
const AXIS_DIMENSION = 32;
const PAIRS_PER_AXIS = AXIS_DIMENSION / 2;
const THETA = 100;

export interface GemmaVisionRotaryTable {
  cosine: Float32Array;
  sine: Float32Array;
  rows: number;
}

export function createGemmaVisionRotaryTable(
  positions: Int32Array,
  rows = positions.length / AXES,
): GemmaVisionRotaryTable {
  if (!Number.isInteger(rows) || rows < 1 || positions.length < rows * AXES) {
    throw new Error("Gemma vision rotary positions are invalid");
  }
  const cosine = new Float32Array(rows * AXES * PAIRS_PER_AXIS);
  const sine = new Float32Array(cosine.length);
  for (let row = 0; row < rows; row += 1) {
    for (let axis = 0; axis < AXES; axis += 1) {
      const position = Math.max(0, positions[row * AXES + axis]);
      for (let pair = 0; pair < PAIRS_PER_AXIS; pair += 1) {
        const angle = position / THETA ** (2 * pair / AXIS_DIMENSION);
        const index = (row * AXES + axis) * PAIRS_PER_AXIS + pair;
        cosine[index] = Math.fround(Math.cos(angle));
        sine[index] = Math.fround(Math.sin(angle));
      }
    }
  }
  return { cosine, sine, rows };
}