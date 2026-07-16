export interface QatLinearFixture {
  input: Float32Array;
  packedWeights: Uint32Array;
  rowScales: Float32Array;
  inFeatures: number;
  outFeatures: number;
  bits?: 2 | 4;
  inputActivationScale?: number;
  outputActivationScale?: number;
  inputSum?: Float32Array;
  emulateBfloat16?: boolean;
}

const INT4_VALUES_PER_WORD = 8;
const INT4_VALUE_MASK = 0x0f;
const INT2_VALUES_PER_WORD = 16;
const INT2_VALUE_MASK = 0x03;

export function packInt4Rows(
  codes: Uint8Array,
  outFeatures: number,
  inFeatures: number,
): Uint32Array {
  if (inFeatures % INT4_VALUES_PER_WORD !== 0) {
    throw new Error("Int4 input width must be divisible by 8");
  }
  if (codes.length !== outFeatures * inFeatures) {
    throw new Error("Int4 code count does not match the matrix shape");
  }

  const words = new Uint32Array(codes.length / INT4_VALUES_PER_WORD);
  for (let wordIndex = 0; wordIndex < words.length; wordIndex += 1) {
    let packed = 0;
    for (let offset = 0; offset < INT4_VALUES_PER_WORD; offset += 1) {
      const code = codes[wordIndex * INT4_VALUES_PER_WORD + offset];
      if (code > INT4_VALUE_MASK) throw new Error("Int4 codes must be between 0 and 15");
      packed |= code << (offset * 4);
    }
    words[wordIndex] = packed >>> 0;
  }
  return words;
}

export function unpackInt4(packedWeights: Uint32Array, index: number): number {
  const word = packedWeights[Math.floor(index / INT4_VALUES_PER_WORD)];
  return (word >>> ((index % INT4_VALUES_PER_WORD) * 4)) & INT4_VALUE_MASK;
}

export function packInt2Rows(
  codes: Uint8Array,
  outFeatures: number,
  inFeatures: number,
): Uint32Array {
  if (inFeatures % INT2_VALUES_PER_WORD !== 0) {
    throw new Error("Int2 input width must be divisible by 16");
  }
  if (codes.length !== outFeatures * inFeatures) {
    throw new Error("Int2 code count does not match the matrix shape");
  }

  const words = new Uint32Array(codes.length / INT2_VALUES_PER_WORD);
  for (let wordIndex = 0; wordIndex < words.length; wordIndex += 1) {
    let packed = 0;
    for (let offset = 0; offset < INT2_VALUES_PER_WORD; offset += 1) {
      const code = codes[wordIndex * INT2_VALUES_PER_WORD + offset];
      if (code > INT2_VALUE_MASK) throw new Error("Int2 codes must be between 0 and 3");
      packed |= code << (offset * 2);
    }
    words[wordIndex] = packed >>> 0;
  }
  return words;
}

export function unpackInt2(packedWeights: Uint32Array, index: number): number {
  const word = packedWeights[Math.floor(index / INT2_VALUES_PER_WORD)];
  return (word >>> ((index % INT2_VALUES_PER_WORD) * 2)) & INT2_VALUE_MASK;
}

export function cpuQatLinear(fixture: QatLinearFixture): Float32Array {
  const {
    input,
    packedWeights,
    rowScales,
    inFeatures,
    outFeatures,
    bits = 4,
    inputActivationScale = 0,
    outputActivationScale = 0,
    emulateBfloat16 = false,
  } = fixture;
  const valuesPerWord = bits === 2 ? INT2_VALUES_PER_WORD : INT4_VALUES_PER_WORD;
  const zeroPoint = bits === 2 ? 2 : 8;
  if (input.length !== inFeatures) throw new Error("Input width does not match the matrix");
  if (rowScales.length !== outFeatures) throw new Error("Scale count does not match the output width");
  if (packedWeights.length * valuesPerWord !== inFeatures * outFeatures) {
    throw new Error("Packed weight count does not match the matrix shape");
  }

  const output = new Float32Array(outFeatures);
  for (let row = 0; row < outFeatures; row += 1) {
    let dot = 0;
    const rowOffset = row * inFeatures;
    for (let column = 0; column < inFeatures; column += 1) {
      const activation = applySrq(input[column], inputActivationScale, emulateBfloat16);
      const code = bits === 2
        ? unpackInt2(packedWeights, rowOffset + column)
        : unpackInt4(packedWeights, rowOffset + column);
      const integerWeight = code - zeroPoint;
      const weight = emulateBfloat16
        ? roundToBfloat16(integerWeight * roundToBfloat16(rowScales[row]))
        : integerWeight;
      dot += weight * activation;
    }
    const projected = emulateBfloat16 ? dot : dot * rowScales[row];
    output[row] = applySrq(projected, outputActivationScale, emulateBfloat16);
  }
  return output;
}

export function applySrq(value: number, scale: number, emulateBfloat16 = false): number {
  if (scale === 0) return value;
  const input = emulateBfloat16 ? roundToBfloat16(value) : value;
  const effectiveScale = emulateBfloat16 ? roundToBfloat16(scale) : scale;
  const ratio = emulateBfloat16
    ? roundToBfloat16(input / effectiveScale)
    : input / effectiveScale;
  const quantized = Math.min(127, Math.max(-128, roundTiesToEven(ratio)));
  const output = Math.fround(quantized * effectiveScale);
  return emulateBfloat16 ? roundToBfloat16(output) : output;
}

const BFLOAT_SCRATCH = new DataView(new ArrayBuffer(4));

function roundToBfloat16(value: number): number {
  BFLOAT_SCRATCH.setFloat32(0, value, true);
  const bits = BFLOAT_SCRATCH.getUint32(0, true);
  const rounded = bits + 0x7fff + ((bits >>> 16) & 1);
  BFLOAT_SCRATCH.setUint32(0, rounded & 0xffff0000, true);
  return BFLOAT_SCRATCH.getFloat32(0, true);
}

function roundTiesToEven(value: number): number {
  const lower = Math.floor(value);
  const fraction = value - lower;
  if (fraction < 0.5) return lower;
  if (fraction > 0.5) return lower + 1;
  return lower % 2 === 0 ? lower : lower + 1;
}

export function createQatLinearFixture(
  inFeatures: number,
  outFeatures: number,
  seed = 0x4e554d41,
): QatLinearFixture {
  if (inFeatures % INT4_VALUES_PER_WORD !== 0) {
    throw new Error("Int4 input width must be divisible by 8");
  }

  const input = Float32Array.from(
    { length: inFeatures },
    (_, index) => Math.fround(Math.sin(index * 0.071) * 1.75),
  );
  const rowScales = Float32Array.from(
    { length: outFeatures },
    (_, index) => Math.fround(0.0008 + (index % 29) * 0.000025),
  );
  const codes = new Uint8Array(inFeatures * outFeatures);
  let state = seed >>> 0;
  for (let index = 0; index < codes.length; index += 1) {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    codes[index] = state & INT4_VALUE_MASK;
  }

  return {
    input,
    packedWeights: packInt4Rows(codes, outFeatures, inFeatures),
    rowScales,
    inFeatures,
    outFeatures,
  };
}