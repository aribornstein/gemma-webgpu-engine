import { expect, test } from "@playwright/test";
import {
  applySrq,
  cpuQatLinear,
  packInt2Rows,
  packInt4Rows,
  unpackInt2,
  unpackInt4,
  type QatLinearFixture,
} from "../src/reference/qat-linear";

test("packs Gemma int4 weights lower nibble first", () => {
  const codes = Uint8Array.from([0, 1, 2, 3, 4, 5, 6, 15]);
  const packed = packInt4Rows(codes, 1, 8);

  expect(packed).toEqual(Uint32Array.from([0xf6543210]));
  expect(codes.every((code, index) => unpackInt4(packed, index) === code)).toBe(true);
});

test("packs Gemma int2 weights from low to high bits", () => {
  const codes = Uint8Array.from([0, 1, 2, 3, 3, 2, 1, 0, 1, 3, 0, 2, 2, 0, 3, 1]);
  const packed = packInt2Rows(codes, 1, 16);

  expect(packed).toEqual(Uint32Array.from([0x728d1be4]));
  expect(codes.every((code, index) => unpackInt2(packed, index) === code)).toBe(true);
});

test("applies the int2 midpoint zero point and row scale", () => {
  const input = Float32Array.from([1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16]);
  const codes = Uint8Array.from([0, 1, 2, 3, 3, 2, 1, 0, 1, 3, 0, 2, 2, 0, 3, 1]);
  const explicitDot = codes.reduce(
    (sum, code, index) => sum + (code - 2) * input[index],
    0,
  );

  expect(Array.from(cpuQatLinear({
    input,
    packedWeights: packInt2Rows(codes, 1, 16),
    rowScales: Float32Array.from([0.25]),
    inFeatures: 16,
    outFeatures: 1,
    bits: 2,
  }))).toEqual([explicitDot * 0.25]);
});

test("applies midpoint zero point and per-row scales", () => {
  const fixture: QatLinearFixture = {
    input: Float32Array.from([1, -2, 3, -4, 5, -6, 7, -8]),
    packedWeights: packInt4Rows(
      Uint8Array.from([
        8, 8, 8, 8, 8, 8, 8, 8,
        9, 7, 10, 6, 11, 5, 12, 4,
      ]),
      2,
      8,
    ),
    rowScales: Float32Array.from([0.5, 0.25]),
    inFeatures: 8,
    outFeatures: 2,
  };

  expect(Array.from(cpuQatLinear(fixture))).toEqual([0, 27.5]);
});

test("applies signed 8-bit SRQ with ties-to-even rounding", () => {
  expect(applySrq(2.5, 1)).toBe(2);
  expect(applySrq(3.5, 1)).toBe(4);
  expect(applySrq(-2.5, 1)).toBe(-2);
  expect(applySrq(1000, 1)).toBe(127);
  expect(applySrq(-1000, 1)).toBe(-128);
  expect(applySrq(2.5, 0)).toBe(2.5);
});

test("rejects invalid int4 values and shapes", () => {
  expect(() => packInt4Rows(Uint8Array.from([16, 0, 0, 0, 0, 0, 0, 0]), 1, 8)).toThrow(
    "Int4 codes must be between 0 and 15",
  );
  expect(() => packInt4Rows(new Uint8Array(7), 1, 7)).toThrow(
    "Int4 input width must be divisible by 8",
  );
});

test("rejects invalid int2 values and shapes", () => {
  expect(() => packInt2Rows(Uint8Array.from([4, ...new Uint8Array(15)]), 1, 16)).toThrow(
    "Int2 codes must be between 0 and 3",
  );
  expect(() => packInt2Rows(new Uint8Array(15), 1, 15)).toThrow(
    "Int2 input width must be divisible by 16",
  );
});