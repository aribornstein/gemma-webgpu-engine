import { expect, test } from "@playwright/test";
import { gemmaDecodeAttentionChunkCount } from "../src/webgpu/decode-attention-chunks";

test("dispatches only active decode attention chunks", () => {
  expect(gemmaDecodeAttentionChunkCount(0, 512)).toBe(8);
  expect(gemmaDecodeAttentionChunkCount(511, 512)).toBe(8);
  expect(gemmaDecodeAttentionChunkCount(32_767, 512)).toBe(8);
  expect(gemmaDecodeAttentionChunkCount(511, 0)).toBe(8);
  expect(gemmaDecodeAttentionChunkCount(512, 0)).toBe(9);
  expect(gemmaDecodeAttentionChunkCount(2_047, 0)).toBe(32);
  expect(gemmaDecodeAttentionChunkCount(131_071, 0)).toBe(32);
});