import { expect, test } from "@playwright/test";

test("GPU top candidates match the CPU sampler's ranking, tail mass, and draws", async ({ page }) => {
  test.setTimeout(120_000);
  await page.goto("/");
  const webGpuAvailable = await page.evaluate(() => Boolean(navigator.gpu));
  test.skip(!webGpuAvailable, "Chrome does not expose WebGPU on this machine");

  const report = await page.evaluate(async () => {
    const kernelPath = "/src/webgpu/decode-top-candidates.ts";
    const samplingPath = "/src/runtime/sampling.ts";
    const decodingPath = "/src/runtime/decoding.ts";
    const devicePath = "/src/webgpu/device.ts";
    const kernel = await import(kernelPath);
    const sampling = await import(samplingPath);
    const { DEFAULT_DECODING_CONFIG } = await import(decodingPath);
    const { getWebGpuDevice } = await import(devicePath);
    const device: GPUDevice = await getWebGpuDevice();
    const vocab = 262144;
    const softcap = 30;
    const logitsBuffer = device.createBuffer({
      size: vocab * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const makeLogits = (kind: string, seed: number) => {
      const random = new sampling.SeededRandom(seed);
      const values = new Float32Array(vocab);
      for (let index = 0; index < vocab; index += 1) {
        const u = random.next() || 1e-9;
        values[index] = 4 * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * random.next());
      }
      if (kind === "ties") for (let index = 0; index < vocab; index += 1) values[index] = Math.round(values[index] * 2) / 2;
      if (kind === "signed-zero") for (let index = 0; index < vocab; index += 5) values[index] = index % 2 ? -0 : 0;
      if (kind === "peaked") values[200_001] = 48;
      return values;
    };
    const cpuOrder = (values: Float32Array) => Array.from({ length: vocab }, (_, index) => index)
      .sort((left, right) => (values[right] - values[left]) || (left - right));
    const failures: string[] = [];
    let draws = 0;
    let drawMismatches = 0;
    let maxTailError = 0;
    const configs = [
      { temperature: 1, topK: 64, topP: 0.95 },
      { temperature: 0.8, topK: 40, topP: 0.9, minP: 0.05 },
      { temperature: 1.2, topK: 100, topP: 1, typicalP: 0.9 },
      { temperature: 0.5, topK: 5, topP: 0.5 },
      { temperature: 1, topK: 256, topP: 0.99 },
      { temperature: 0.7, topK: 1, topP: 1 },
    ];
    for (const kind of ["normal", "ties", "signed-zero", "peaked"]) {
      const values = makeLogits(kind, kind.length * 17 + 3);
      device.queue.writeBuffer(logitsBuffer, 0, values);
      const order = cpuOrder(values);
      for (const config of configs) {
        const listLength = kernel.gemmaTopCandidateListLength(config.topK);
        const pipelines = await kernel.getGemmaTopCandidatePipelines(device, listLength);
        const resources = kernel.createGemmaTopCandidateResources(device, pipelines, logitsBuffer);
        kernel.updateGemmaTopCandidates(device, resources, config.topK, softcap, config.temperature);
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        kernel.encodeGemmaTopCandidatesPass(pass, resources);
        pass.end();
        kernel.encodeGemmaTopCandidatesCopy(encoder, resources);
        device.queue.submit([encoder.finish()]);
        const result = await kernel.readGemmaTopCandidates(resources);
        kernel.destroyGemmaTopCandidateResources(resources);
        const label = `${kind} top-${config.topK}`;
        for (let rank = 0; rank < listLength; rank += 1) {
          if (result.tokenIds[rank] !== order[rank] || !Object.is(result.logits[rank], values[order[rank]])) {
            failures.push(`${label} rank ${rank}: gpu ${result.tokenIds[rank]} cpu ${order[rank]}`);
            break;
          }
        }
        if (result.nonFiniteCount !== 0) failures.push(`${label}: unexpected non-finite count`);
        const scaled = (value: number) => softcap * Math.tanh(value / softcap) / config.temperature;
        const maximum = scaled(values[order[0]]);
        let tail = 0;
        for (let rank = config.topK; rank < vocab; rank += 1) tail += Math.exp(scaled(values[order[rank]]) - maximum);
        const tailError = Math.abs(result.tailMass - tail) / Math.max(tail, 1e-30);
        maxTailError = Math.max(maxTailError, tailError);
        if (tailError > 1e-4) failures.push(`${label}: tail ${result.tailMass} vs ${tail}`);
        const decoding = { ...DEFAULT_DECODING_CONFIG, minP: 0, typicalP: 1, repetitionPenalty: 1, ...config };
        for (let seed = 1; seed <= 40; seed += 1) {
          const gpuRandom = new sampling.SeededRandom(seed);
          const cpuRandom = new sampling.SeededRandom(seed);
          const gpuToken = sampling.sampleTopCandidates(
            result.tokenIds,
            result.logits,
            result.tailMass,
            decoding,
            () => gpuRandom.next(),
            softcap,
          );
          const cpuToken = sampling.sampleToken(values, [], decoding, () => cpuRandom.next(), softcap);
          draws += 1;
          if (gpuToken !== cpuToken) drawMismatches += 1;
        }
      }
    }
    const special = makeLogits("normal", 5);
    special[17] = Number.NaN;
    special[90_000] = Number.NEGATIVE_INFINITY;
    special[123_456] = Number.POSITIVE_INFINITY;
    device.queue.writeBuffer(logitsBuffer, 0, special);
    const pipelines = await kernel.getGemmaTopCandidatePipelines(device, 64);
    const resources = kernel.createGemmaTopCandidateResources(device, pipelines, logitsBuffer);
    kernel.updateGemmaTopCandidates(device, resources, 64, softcap, 1);
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    kernel.encodeGemmaTopCandidatesPass(pass, resources);
    pass.end();
    kernel.encodeGemmaTopCandidatesCopy(encoder, resources);
    device.queue.submit([encoder.finish()]);
    const flagged = await kernel.readGemmaTopCandidates(resources);
    kernel.destroyGemmaTopCandidateResources(resources);
    logitsBuffer.destroy();
    return { failures, draws, drawMismatches, maxTailError, nonFinite: flagged.nonFiniteCount };
  });
  expect(report.failures).toEqual([]);
  expect(report.nonFinite).toBe(3);
  expect(report.draws).toBe(4 * 6 * 40);
  expect(report.drawMismatches).toBe(0);
  expect(report.maxTailError).toBeLessThan(1e-4);
});
