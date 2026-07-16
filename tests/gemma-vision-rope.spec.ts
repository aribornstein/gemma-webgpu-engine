import { expect, test } from "@playwright/test";
import { createGemmaVisionRotaryTable } from "../src/model/gemma-vision-rope";

test("constructs independent x and y vision RoPE frequencies", () => {
  const table = createGemmaVisionRotaryTable(new Int32Array([3, 7, -1, -1]));
  expect(table.rows).toBe(2);
  expect(table.cosine[0]).toBe(Math.fround(Math.cos(3)));
  expect(table.sine[0]).toBe(Math.fround(Math.sin(3)));
  expect(table.cosine[16]).toBe(Math.fround(Math.cos(7)));
  expect(table.sine[16]).toBe(Math.fround(Math.sin(7)));
  expect(Array.from(table.cosine.slice(32))).toEqual(new Array(32).fill(1));
  expect(Array.from(table.sine.slice(32))).toEqual(new Array(32).fill(0));
});