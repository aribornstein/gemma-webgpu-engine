import { expect, test } from "@playwright/test";

test("loads the exact pinned Gemma audio layer and global tensors", async ({ page }) => {
  test.setTimeout(120_000);
  await page.goto("/");
  const result = await page.evaluate(async () => {
    const sourcePath = "/src/model/pinned-safetensors.ts";
    const audioPath = "/src/model/gemma-audio-weights.ts";
    const [{ PinnedSafetensorsSource }, audio] = await Promise.all([
      import(sourcePath),
      import(audioPath),
    ]);
    const source = await PinnedSafetensorsSource.open();
    const [layer, globals] = await Promise.all([
      audio.loadGemmaAudioLayerWeights(source, 0),
      audio.loadGemmaAudioGlobalWeights(source),
    ]);
    return {
      layerIndex: layer.layerIndex,
      layerSourceBytes: layer.sourceBytes,
      globalSourceBytes: globals.sourceBytes,
      queryBits: layer.attention.query.bits,
      queryPackedValues: layer.attention.query.packedWeights.length,
      convolutionInputBits: layer.convolution.input.bits,
      convolutionInputPackedValues: layer.convolution.input.packedWeights.length,
      relativeKeyValues: layer.attention.relativeKeyProjection.length,
      depthwiseValues: layer.convolution.depthwise.length,
      towerOutputValues: globals.towerOutput.weight.length,
      embeddingProjectionValues: globals.embeddingProjection.length,
    };
  });

  expect(result).toEqual({
    layerIndex: 0,
    layerSourceBytes: 10_875_472,
    globalSourceBytes: 20_081_792,
    queryBits: 2,
    queryPackedValues: 65_536,
    convolutionInputBits: 4,
    convolutionInputPackedValues: 262_144,
    relativeKeyValues: 1_048_576,
    depthwiseValues: 5_120,
    towerOutputValues: 1_572_864,
    embeddingProjectionValues: 2_359_296,
  });
});

test("reuses and clears cached audio weights", async ({ page }) => {
  test.setTimeout(120_000);
  await page.goto("/");
  const result = await page.evaluate(async () => {
    const sourcePath = "/src/model/pinned-safetensors.ts";
    const audioPath = "/src/model/gemma-audio-weights.ts";
    const [{ PinnedSafetensorsSource }, { GemmaAudioWeightCache }] = await Promise.all([
      import(sourcePath),
      import(audioPath),
    ]);
    const source = await PinnedSafetensorsSource.open();
    const cache = new GemmaAudioWeightCache();
    const firstLayer = cache.loadLayer(source, 0);
    const secondLayer = cache.loadLayer(source, 0);
    const firstGlobals = cache.loadGlobals(source);
    const secondGlobals = cache.loadGlobals(source);
    await Promise.all([firstLayer, firstGlobals]);
    const loaded = cache.estimateRetainedMemory();
    cache.clear();
    return {
      sameLayerPromise: firstLayer === secondLayer,
      sameGlobalPromise: firstGlobals === secondGlobals,
      loaded,
      cleared: cache.estimateRetainedMemory(),
    };
  });

  expect(result.sameLayerPromise).toBe(true);
  expect(result.sameGlobalPromise).toBe(true);
  expect(result.loaded).toEqual({
    loadedEntryCount: 2,
    sourceBytes: 30_957_264,
    materializedBytes: 30_957_184,
  });
  expect(result.cleared).toEqual({
    loadedEntryCount: 0,
    sourceBytes: 0,
    materializedBytes: 0,
  });
});