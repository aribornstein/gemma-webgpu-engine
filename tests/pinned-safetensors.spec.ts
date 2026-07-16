import { expect, test } from "@playwright/test";

test("reads exact vision tensor ranges from the pinned checkpoint", async ({ page }) => {
  await page.goto("/");
  const result = await page.evaluate(async () => {
    const modulePath = "/src/model/pinned-safetensors.ts";
    const { PinnedSafetensorsSource } = await import(modulePath);
    const source = await PinnedSafetensorsSource.open();
    const names = [
      "model.vision_tower.patch_embedder.input_proj.weight",
      "model.vision_tower.patch_embedder.position_embedding_table",
      "model.vision_tower.encoder.layers.0.input_layernorm.weight",
      "model.embed_vision.embedding_projection.weight",
    ];
    const tensors = await source.readTensors(names);
    return names.map((name) => {
      const tensor = tensors.get(name);
      if (!tensor) throw new Error(`Missing tensor ${name}`);
      return {
        name,
        dtype: tensor.dtype,
        shape: tensor.shape,
        byteLength: tensor.byteLength,
        sha256: tensor.sha256,
      };
    });
  });

  expect(result).toEqual([
    {
      name: "model.vision_tower.patch_embedder.input_proj.weight",
      dtype: "BF16",
      shape: [768, 768],
      byteLength: 1_179_648,
      sha256: "cb7caa77a0a3d4a35b9cd6005c37a355177360a37511fdc76feffd8df9dc1c12",
    },
    {
      name: "model.vision_tower.patch_embedder.position_embedding_table",
      dtype: "BF16",
      shape: [2, 10_240, 768],
      byteLength: 31_457_280,
      sha256: "de759b1e5a83c56c8f27822d7acef2028e101dafa139b2e69da33deeb8d1a8f6",
    },
    {
      name: "model.vision_tower.encoder.layers.0.input_layernorm.weight",
      dtype: "BF16",
      shape: [768],
      byteLength: 1_536,
      sha256: "b113244500d12479397fb721e233b26ec59b81dc5ea7422c9e3994709f322fa0",
    },
    {
      name: "model.embed_vision.embedding_projection.weight",
      dtype: "F32",
      shape: [1536, 768],
      byteLength: 4_718_592,
      sha256: "f4408700b561f7352b4720aeaa645dc8ffadf4c12b7918483e34a62ccfe9263f",
    },
  ]);
});