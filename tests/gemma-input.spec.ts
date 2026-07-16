import { expect, test } from "@playwright/test";
import type {
  CachedTensorDescriptor,
  CachedTensorSliceRequest,
} from "../src/model/cached-safetensors";
import {
  loadGemmaTokenInputBatch,
  loadGemmaTokenInputs,
  type GemmaInputTensorSource,
} from "../src/model/gemma-input-weights";
import {
  createGemmaRotaryBlock,
  createGemmaRotaryRows,
} from "../src/model/gemma-rope";

test("dequantizes cached token and per-layer embedding rows", async () => {
  const source = createInputSource();
  const inputs = await loadGemmaTokenInputs(source, 7);

  expect(Array.from(inputs.hidden.slice(0, 8))).toEqual([
    -2 * Math.fround(Math.sqrt(1536)),
    -1 * Math.fround(Math.sqrt(1536)),
    0,
    1 * Math.fround(Math.sqrt(1536)),
    -2 * Math.fround(Math.sqrt(1536)),
    -1 * Math.fround(Math.sqrt(1536)),
    0,
    1 * Math.fround(Math.sqrt(1536)),
  ]);
  expect(Array.from(inputs.perLayerEmbedding.slice(0, 4))).toEqual([-128, -128, -112, -128]);
  expect(Array.from(inputs.perLayerEmbedding.slice(256, 260))).toEqual([-256, 0, -224, 0]);
  expect(source.slices).toEqual([
    ["model.language_model.embed_tokens.embedding_quantized", 7 * 384, 384],
    ["model.language_model.embed_tokens.embedding_scale", 7 * 4, 4],
    ["model.language_model.embed_tokens_per_layer.embedding_quantized", 7 * 4480, 4480],
    ["model.language_model.embed_tokens_per_layer.embedding_scale", 7 * 140, 140],
  ]);
});

test("loads a fixed token block with token-zero padding through one slice batch", async () => {
  const source = createInputSource();
  const batches: CachedTensorSliceRequest[][] = [];
  source.readTensorSlices = async (requests) => {
    batches.push(Array.from(requests));
    return requests.map(({ name, byteLength }) => inputSlice(name, byteLength));
  };

  const inputs = await loadGemmaTokenInputBatch(source, [7, 9], 4);

  expect(inputs).toHaveLength(4);
  expect(source.slices).toEqual([]);
  expect(batches).toHaveLength(1);
  expect(batches[0].map(({ byteOffset }) => byteOffset)).toEqual([
    7 * 384, 7 * 4, 7 * 4480, 7 * 140,
    9 * 384, 9 * 4, 9 * 4480, 9 * 140,
    0, 0, 0, 0,
    0, 0, 0, 0,
  ]);
});

test("constructs sliding and partial full RoPE rows", () => {
  const zero = createGemmaRotaryRows(0);
  expect(Array.from(zero.sliding.cosine)).toEqual(new Array(128).fill(1));
  expect(Array.from(zero.sliding.sine)).toEqual(new Array(128).fill(0));
  expect(Array.from(zero.full.cosine)).toEqual(new Array(256).fill(1));
  expect(Array.from(zero.full.sine)).toEqual(new Array(256).fill(0));

  const position = createGemmaRotaryRows(10);
  expect(position.sliding.cosine[0]).toBe(Math.fround(Math.cos(10)));
  expect(position.full.cosine[0]).toBe(Math.fround(Math.cos(10)));
  expect(Array.from(position.full.cosine.slice(64))).toEqual(new Array(192).fill(1));
  expect(Array.from(position.full.sine.slice(64))).toEqual(new Array(192).fill(0));
});

test("constructs fixed-block RoPE exactly from sequential rows", () => {
  const block = createGemmaRotaryBlock(11, 4);
  expect(block.rowCount).toBe(4);
  for (let row = 0; row < block.rowCount; row += 1) {
    const expected = createGemmaRotaryRows(11 + row);
    expect(block.sliding.cosine.slice(row * 128, (row + 1) * 128)).toEqual(
      expected.sliding.cosine,
    );
    expect(block.sliding.sine.slice(row * 128, (row + 1) * 128)).toEqual(
      expected.sliding.sine,
    );
    expect(block.full.cosine.slice(row * 256, (row + 1) * 256)).toEqual(
      expected.full.cosine,
    );
    expect(block.full.sine.slice(row * 256, (row + 1) * 256)).toEqual(
      expected.full.sine,
    );
  }
});

test("keeps 8K, 32K, and 128K RoPE rows finite and normalized", () => {
  for (const position of [8_191, 32_767, 131_071]) {
    const rotary = createGemmaRotaryRows(position);
    for (const row of [rotary.sliding, rotary.full]) {
      expect(Array.from(row.cosine).every(Number.isFinite)).toBe(true);
      expect(Array.from(row.sine).every(Number.isFinite)).toBe(true);
    }
    for (let pair = 0; pair < rotary.sliding.cosine.length; pair += 1) {
      const cosine = rotary.sliding.cosine[pair];
      const sine = rotary.sliding.sine[pair];
      expect(Math.abs(cosine * cosine + sine * sine - 1)).toBeLessThan(2e-7);
    }
    for (let pair = 0; pair < 64; pair += 1) {
      const cosine = rotary.full.cosine[pair];
      const sine = rotary.full.sine[pair];
      expect(Math.abs(cosine * cosine + sine * sine - 1)).toBeLessThan(2e-7);
    }
  }
});

test("constructs the pinned scale-specific GELU lookup table", async ({ page }) => {
  await page.goto("/");
  const result = await page.evaluate(async () => {
    const fixtureModulePath = "/src/model/decode-mlp-ple-fixture.ts";
    const lutModulePath = "/src/model/gemma-gelu-lut.ts";
    const [{ loadDecodeMlpPleFixture }, { createGemmaGeluLut }] = await Promise.all([
      import(fixtureModulePath),
      import(lutModulePath),
    ]);
    const expected = new Uint32Array((await loadDecodeMlpPleFixture()).gateGeluLut.buffer);
    const actual = new Uint32Array(createGemmaGeluLut(0.6181102395057678).buffer);
    let valueMismatches = 0;
    let signedZeroMismatches = 0;
    for (let index = 0; index < actual.length; index += 1) {
      if (actual[index] === expected[index]) continue;
      if ((actual[index] & 0x7fffffff) === 0 && (expected[index] & 0x7fffffff) === 0) {
        signedZeroMismatches += 1;
      } else {
        valueMismatches += 1;
      }
    }
    return { valueMismatches, signedZeroMismatches };
  });

  expect(result.valueMismatches).toBe(0);
  expect(result.signedZeroMismatches).toBe(120);
});

interface InputSource extends GemmaInputTensorSource {
  slices: [string, number, number][];
}

function createInputSource(): InputSource {
  const descriptors = new Map<string, CachedTensorDescriptor>([
    descriptor("model.language_model.embed_tokens.embedding_quantized", "U8", [262144, 384]),
    descriptor("model.language_model.embed_tokens.embedding_scale", "F32", [262144, 1]),
    descriptor("model.language_model.embed_tokens_per_layer.embedding_quantized", "U8", [262144, 4480]),
    descriptor("model.language_model.embed_tokens_per_layer.embedding_scale", "F32", [262144, 35]),
  ].map((value) => [value.name, value]));
  const source: InputSource = {
    descriptors,
    slices: [],
    async readTensors() {
      return new Map();
    },
    async readTensorSlice(name, byteOffset, byteLength) {
      source.slices.push([name, byteOffset, byteLength]);
      return inputSlice(name, byteLength);
    },
  };
  return source;
}

function inputSlice(name: string, byteLength: number): Uint8Array {
  if (name.endsWith("embed_tokens.embedding_quantized")) {
    return new Uint8Array(byteLength).fill(0xe4);
  }
  if (name.endsWith("embed_tokens.embedding_scale")) return floatBytes([1]);
  if (name.endsWith("embedding_quantized")) {
    return Uint8Array.from({ length: byteLength }, (_, index) => index & 0xff);
  }
  return floatBytes(Array.from({ length: 35 }, (_, index) => index + 1));
}

function descriptor(name: string, dtype: string, shape: readonly number[]): CachedTensorDescriptor {
  const bytes = shape.reduce((product, value) => product * value, 1) * (dtype === "F32" ? 4 : 1);
  return { name, dtype, shape, begin: 0, end: bytes, byteLength: bytes };
}

function floatBytes(values: readonly number[]): Uint8Array {
  return new Uint8Array(new Float32Array(values).buffer);
}