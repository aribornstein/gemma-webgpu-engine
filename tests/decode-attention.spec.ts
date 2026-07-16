import { expect, test } from "@playwright/test";
import { createDecodeAttentionShader } from "../src/webgpu/decode-attention";

test("sizes attention workgroup activations for each head profile", () => {
  const sliding = createDecodeAttentionShader(256);
  const full = createDecodeAttentionShader(512);

  expect(sliding).toContain("var<workgroup> qn_sh: array<f32, 256>;");
  expect(sliding).toContain("var<workgroup> out_acc: array<f32, 256>;");
  expect(full).toContain("var<workgroup> qn_sh: array<f32, 512>;");
  expect(full).toContain("var<workgroup> out_acc: array<f32, 512>;");
});