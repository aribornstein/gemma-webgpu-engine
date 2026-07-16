import { expect, test } from "@playwright/test";

test("owns reusable contiguous decode K/V cache storage", async ({ page }) => {
  await page.goto("/");
  const webGpuAvailable = await page.evaluate(() => Boolean(navigator.gpu));
  test.skip(!webGpuAvailable, "Chrome does not expose WebGPU on this machine");

  const result = await page.evaluate(async () => {
    const cacheModulePath = "/src/webgpu/decode-kv-cache.ts";
    const deviceModulePath = "/src/webgpu/device.ts";
    const { DecodeKvCache } = await import(cacheModulePath);
    const { getWebGpuDevice } = await import(deviceModulePath);
    const device = await getWebGpuDevice();
    const cache = new DecodeKvCache(device, {
      capacity: 4,
      kvHeads: 1,
      headDim: 4,
      label: "K/V cache test",
    });
    const readBuffer = device.createBuffer({
      label: "K/V cache test readback",
      size: cache.bytesAllocated,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    try {
      const keys = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]);
      const values = new Float32Array([-1, -2, -3, -4, -5, -6, -7, -8]);
      cache.writeTokens(device.queue, 0, keys, values);
      cache.commitWrite(2);

      const encoder = device.createCommandEncoder({ label: "K/V cache test copy" });
      encoder.copyBufferToBuffer(cache.keyBuffer, 0, readBuffer, 0, cache.bufferBytes);
      encoder.copyBufferToBuffer(
        cache.valueBuffer,
        0,
        readBuffer,
        cache.bufferBytes,
        cache.bufferBytes,
      );
      device.queue.submit([encoder.finish()]);
      await readBuffer.mapAsync(GPUMapMode.READ);
      const stored = new Float32Array(readBuffer.getMappedRange().slice(0));
      readBuffer.unmap();

      cache.reset();
      let gapRejected = false;
      try {
        cache.commitWrite(1);
      } catch {
        gapRejected = true;
      }
      let boundsRejected = false;
      try {
        cache.byteOffset(4);
      } catch {
        boundsRejected = true;
      }
      const clearEncoder = device.createCommandEncoder({ label: "K/V cache test clear" });
      cache.encodeClear(clearEncoder);
      device.queue.submit([clearEncoder.finish()]);
      await device.queue.onSubmittedWorkDone();

      return {
        keys: Array.from(stored.slice(0, keys.length)),
        values: Array.from(stored.slice(16, 16 + values.length)),
        length: cache.length,
        byteOffsetAtTwo: cache.byteOffset(2),
        bytesAllocated: cache.bytesAllocated,
        gapRejected,
        boundsRejected,
      };
    } finally {
      readBuffer.destroy();
      cache.destroy();
    }
  });

  expect(result.keys).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);
  expect(result.values).toEqual([-1, -2, -3, -4, -5, -6, -7, -8]);
  expect(result.length).toBe(0);
  expect(result.byteOffsetAtTwo).toBe(32);
  expect(result.bytesAllocated).toBe(128);
  expect(result.gapRejected).toBe(true);
  expect(result.boundsRejected).toBe(true);
});

test("wraps circular decode K/V cache storage at its physical capacity", async ({ page }) => {
  await page.goto("/");
  const webGpuAvailable = await page.evaluate(() => Boolean(navigator.gpu));
  test.skip(!webGpuAvailable, "Chrome does not expose WebGPU on this machine");

  const result = await page.evaluate(async () => {
    const cacheModulePath = "/src/webgpu/decode-kv-cache.ts";
    const deviceModulePath = "/src/webgpu/device.ts";
    const { DecodeKvCache } = await import(cacheModulePath);
    const { getWebGpuDevice } = await import(deviceModulePath);
    const device = await getWebGpuDevice();
    const cache = new DecodeKvCache(device, {
      capacity: 512,
      kvHeads: 1,
      headDim: 1,
      mode: "circular",
      label: "Circular K/V cache test",
    });
    const readBuffer = device.createBuffer({
      size: cache.bufferBytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    try {
      const keys = Float32Array.from({ length: 514 }, (_, position) => position);
      cache.writeTokens(
        device.queue,
        0,
        keys,
        new Float32Array(keys.length),
      );
      const encoder = device.createCommandEncoder();
      encoder.copyBufferToBuffer(cache.keyBuffer, 0, readBuffer, 0, cache.bufferBytes);
      device.queue.submit([encoder.finish()]);
      await readBuffer.mapAsync(GPUMapMode.READ);
      const stored = Array.from(new Float32Array(readBuffer.getMappedRange().slice(0)));
      readBuffer.unmap();
      return {
        boundary: [stored[511], stored[0], stored[1]],
        length: cache.length,
        positions: [511, 512, 513].map((position) => cache.physicalPosition(position)),
        retainsCurrent: cache.canRetainPrefix(514),
        retainsOldBranch: cache.canRetainPrefix(513),
      };
    } finally {
      readBuffer.destroy();
      cache.destroy();
    }
  });

  expect(result).toEqual({
    boundary: [511, 512, 513],
    length: 514,
    positions: [511, 0, 1],
    retainsCurrent: true,
    retainsOldBranch: false,
  });
});

test("matches the packed int4 CPU reference with cached GPU resources", async ({ page }) => {
  await page.goto("/");
  const webGpuAvailable = await page.evaluate(() => Boolean(navigator.gpu));
  test.skip(!webGpuAvailable, "Chrome does not expose WebGPU on this machine");

  const results = await page.evaluate(async () => {
    const modulePath = "/src/webgpu/qat-linear.ts";
    const { benchmarkQatLinear } = await import(modulePath);
    const first = await benchmarkQatLinear({ inFeatures: 64, outFeatures: 32, iterations: 2 });
    const second = await benchmarkQatLinear({ inFeatures: 64, outFeatures: 32, iterations: 2 });
    return { first, second };
  });

  expect(results.first.tolerancePassed).toBe(true);
  expect(results.first.maximumAbsoluteError).toBeLessThanOrEqual(results.first.absoluteTolerance);
  expect(results.first.gpuBufferAllocations).toBe(6);
  expect(results.first.allocationsPerDispatch).toBe(0);
  expect(results.first.pipelineCacheHit).toBe(false);
  expect(results.second.pipelineCacheHit).toBe(true);
  expect(results.second.shaderCompilationMs).toBe(0);
});

test("runs the verified cached layer-0 Q projection artifact", async ({ page }) => {
  await page.goto("/");
  const webGpuAvailable = await page.evaluate(() => Boolean(navigator.gpu));
  test.skip(!webGpuAvailable, "Chrome does not expose WebGPU on this machine");

  const result = await page.evaluate(async () => {
    const modulePath = "/src/webgpu/qat-linear.ts";
    const { benchmarkRealQatLinear } = await import(modulePath);
    return benchmarkRealQatLinear(2);
  });

  expect(result.artifactSource).toBe("cached-export");
  expect(result.artifactSha256).toBe(
    "932bfa1d84087dba4ef2104a801431a9b8c9a0fd7a25f7ff65d83bea1d062be6",
  );
  expect(result.referenceArtifactSha256).toBe(
    "e9ef6a3b477cda98572611840f6a87965a9460221f4bc2b5473f535b50b876f2",
  );
  expect(result.inFeatures).toBe(1536);
  expect(result.outFeatures).toBe(2048);
  expect(result.tolerancePassed).toBe(true);
  expect(result.gpuBufferAllocations).toBe(6);
  expect(result.allocationsPerDispatch).toBe(0);
});

test("matches Buza's real layer-0 decode Q output", async ({ page }) => {
  await page.goto("/");
  const webGpuAvailable = await page.evaluate(() => Boolean(navigator.gpu));
  test.skip(!webGpuAvailable, "Chrome does not expose WebGPU on this machine");

  const { result, timestampQuerySupported } = await page.evaluate(async () => {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    const modulePath = "/src/webgpu/qat-linear.ts";
    const { benchmarkCapturedQatLinear } = await import(modulePath);
    return {
      result: await benchmarkCapturedQatLinear(2),
      timestampQuerySupported: adapter?.features.has("timestamp-query") ?? false,
    };
  });

  expect(result.artifactSource).toBe("buza-capture");
  expect(result.referenceArtifactSha256).toBe(
    "78dcbadb59abdb04d51facaa1af5674fc40b7260503fcc900239cd00890f1ae9",
  );
  expect(result.inFeatures).toBe(1536);
  expect(result.outFeatures).toBe(2048);
  expect(result.tolerancePassed).toBe(true);
  expect(result.maximumAbsoluteError).toBeLessThanOrEqual(result.absoluteTolerance);
  expect(result.gpuBufferAllocations).toBe(6);
  expect(result.allocationsPerDispatch).toBe(0);
  expect(result.gpuKernelDispatchesPerSample).toBe(timestampQuerySupported ? 10 : null);
  if (timestampQuerySupported) {
    expect(result.gpuKernelMedianMs).toBeGreaterThan(0);
    expect(result.gpuKernelP95Ms).toBeGreaterThanOrEqual(result.gpuKernelMedianMs ?? 0);
  }
});

test("matches Hugging Face QatMatMul scalar_presrq for layer-0 Q", async ({ page }) => {
  await page.goto("/");
  const webGpuAvailable = await page.evaluate(() => Boolean(navigator.gpu));
  test.skip(!webGpuAvailable, "Chrome does not expose WebGPU on this machine");

  const { result, timestampQuerySupported } = await page.evaluate(async () => {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    const modulePath = "/src/webgpu/qat-linear-presrq.ts";
    const { benchmarkCapturedQatLinearPresrq } = await import(modulePath);
    return {
      result: await benchmarkCapturedQatLinearPresrq(2),
      timestampQuerySupported: adapter?.features.has("timestamp-query") ?? false,
    };
  });

  expect(result.sourceOperator).toBe("com.xenova.gemma4.QatMatMul");
  expect(result.sourceVariant).toBe("scalar_presrq");
  expect(result.workgroupSize).toBe(32);
  expect(result.rowsPerWorkgroup).toBe(2);
  expect(result.workgroupCount).toBe(1024);
  expect(result.maximumAbsoluteError).toBe(0);
  expect(result.maximumRelativeError).toBe(0);
  expect(result.gpuBufferAllocations).toBe(7);
  expect(result.allocationsPerDispatch).toBe(0);
  expect(result.gpuKernelDispatchesPerSample).toBe(timestampQuerySupported ? 100 : null);
  if (timestampQuerySupported) {
    expect(result.gpuKernelMedianMs).toBeGreaterThan(0);
    expect(result.gpuKernelP95Ms).toBeGreaterThanOrEqual(result.gpuKernelMedianMs ?? 0);
  }
});

test("matches Hugging Face DecodeQkvProj presrq for layer-0 QKV", async ({ page }) => {
  await page.goto("/");
  const webGpuAvailable = await page.evaluate(() => Boolean(navigator.gpu));
  test.skip(!webGpuAvailable, "Chrome does not expose WebGPU on this machine");

  const { result, timestampQuerySupported } = await page.evaluate(async () => {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    const modulePath = "/src/webgpu/qat-qkv-presrq.ts";
    const { benchmarkCapturedQatQkvPresrq } = await import(modulePath);
    return {
      result: await benchmarkCapturedQatQkvPresrq(2),
      timestampQuerySupported: adapter?.features.has("timestamp-query") ?? false,
    };
  });

  expect(result.sourceOperator).toBe("com.xenova.gemma4.DecodeQkvProj");
  expect(result.sourceVariant).toBe("presrq");
  expect(result.implementation).toBe("combined-storage");
  expect(result.artifactSha256).toBe(
    "63482ab46577cc82b15879a8db0b0fea4515fc690741c34f4e47fb2d6faab1e3",
  );
  expect(result.referenceArtifactSha256).toBe(
    "78dcbadb59abdb04d51facaa1af5674fc40b7260503fcc900239cd00890f1ae9",
  );
  expect(result.workgroupSize).toBe(32);
  expect(result.rowsPerWorkgroup).toBe(2);
  expect(result.qWorkgroupCount).toBe(1024);
  expect(result.kvWorkgroupCount).toBe(128);
  expect(result.workgroupCount).toBe(1280);
  expect(result.qMaximumAbsoluteError).toBe(0);
  expect(result.qMaximumRelativeError).toBe(0);
  expect(result.kMaximumAbsoluteError).toBe(0);
  expect(result.kMaximumRelativeError).toBe(0);
  expect(result.vMaximumAbsoluteError).toBe(0);
  expect(result.vMaximumRelativeError).toBe(0);
  expect(result.gpuBufferAllocations).toBe(7);
  expect(result.allocationsPerDispatch).toBe(0);
  expect(result.gpuKernelDispatchesPerSample).toBe(timestampQuerySupported ? 100 : null);
  if (timestampQuerySupported) {
    expect(result.gpuKernelMedianMs).toBeGreaterThan(0);
    expect(result.gpuKernelP95Ms).toBeGreaterThanOrEqual(result.gpuKernelMedianMs ?? 0);
  }
});

test("matches Hugging Face DecodeRmsSrq for layer-0 decode", async ({ page }) => {
  await page.goto("/");
  const webGpuAvailable = await page.evaluate(() => Boolean(navigator.gpu));
  test.skip(!webGpuAvailable, "Chrome does not expose WebGPU on this machine");

  const { result, timestampQuerySupported } = await page.evaluate(async () => {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    const modulePath = "/src/webgpu/decode-rms-srq.ts";
    const { benchmarkDecodeRmsSrq } = await import(modulePath);
    return {
      result: await benchmarkDecodeRmsSrq(2),
      timestampQuerySupported: adapter?.features.has("timestamp-query") ?? false,
    };
  });

  expect(result.sourceOperator).toBe("com.xenova.gemma4.DecodeRmsSrq");
  expect(result.sourceVariant).toBe("main");
  expect(result.artifactSha256).toBe(
    "75edf39811df47143afcf92fd8e64931820eae808e9a6a11a2b57e4464202c36",
  );
  expect(result.sourceCaptureSha256).toBe(
    "78dcbadb59abdb04d51facaa1af5674fc40b7260503fcc900239cd00890f1ae9",
  );
  expect(result.hiddenSize).toBe(1536);
  expect(result.workgroupSize).toBe(256);
  expect(result.workgroupCount).toBe(1);
  expect(result.subgroupReduction).toBe(true);
  expect(result.outputMaximumAbsoluteError).toBe(0);
  expect(result.outputMaximumRelativeError).toBe(0);
  expect(result.sumMaximumAbsoluteError).toBe(0);
  expect(result.sumMaximumRelativeError).toBe(0);
  expect(result.gpuBufferAllocations).toBe(6);
  expect(result.allocationsPerDispatch).toBe(0);
  expect(result.gpuKernelDispatchesPerSample).toBe(timestampQuerySupported ? 100 : null);
  if (timestampQuerySupported) {
    expect(result.gpuKernelMedianMs).toBeGreaterThan(0);
    expect(result.gpuKernelP95Ms).toBeGreaterThanOrEqual(result.gpuKernelMedianMs ?? 0);
  }
});

test("matches Hugging Face DecodeQkNormRope for layer-0 K", async ({ page }) => {
  await page.goto("/");
  const webGpuAvailable = await page.evaluate(() => Boolean(navigator.gpu));
  test.skip(!webGpuAvailable, "Chrome does not expose WebGPU on this machine");

  const { result, timestampQuerySupported } = await page.evaluate(async () => {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    const modulePath = "/src/webgpu/decode-k-norm-rope.ts";
    const { benchmarkDecodeKNormRope } = await import(modulePath);
    return {
      result: await benchmarkDecodeKNormRope(2),
      timestampQuerySupported: adapter?.features.has("timestamp-query") ?? false,
    };
  });

  expect(result.sourceOperator).toBe("com.xenova.gemma4.DecodeQkNormRope");
  expect(result.sourceVariant).toBe("scalar");
  expect(result.artifactSha256).toBe(
    "b0a44ef55d5dc0d9a827d7f4171fe36c15a6f37a4f74be98bae701708e88a374",
  );
  expect(result.sourceCaptureSha256).toBe(
    "adc161c83071906c53ecf521d4c9fa140d01f729e5ecb6017d06860592306106",
  );
  expect(result.headDim).toBe(256);
  expect(result.halfDim).toBe(128);
  expect(result.heads).toBe(1);
  expect(result.workgroupSize).toBe(128);
  expect(result.workgroupCount).toBe(1);
  expect(result.outputMaximumAbsoluteError).toBe(0);
  expect(result.outputMaximumRelativeError).toBe(0);
  expect(result.gpuBufferAllocations).toBe(7);
  expect(result.allocationsPerDispatch).toBe(0);
  expect(result.gpuKernelDispatchesPerSample).toBe(timestampQuerySupported ? 100 : null);
  if (timestampQuerySupported) {
    expect(result.gpuKernelMedianMs).toBeGreaterThan(0);
    expect(result.gpuKernelP95Ms).toBeGreaterThanOrEqual(result.gpuKernelMedianMs ?? 0);
  }
});

test("composes DecodeQkvProj directly into K norm RoPE cache storage", async ({ page }) => {
  await page.goto("/");
  const webGpuAvailable = await page.evaluate(() => Boolean(navigator.gpu));
  test.skip(!webGpuAvailable, "Chrome does not expose WebGPU on this machine");

  const { result, timestampQuerySupported } = await page.evaluate(async () => {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    const modulePath = "/src/webgpu/decode-qkv-k-norm-rope.ts";
    const { benchmarkDecodeQkvKNormRope } = await import(modulePath);
    return {
      result: await benchmarkDecodeQkvKNormRope(2),
      timestampQuerySupported: adapter?.features.has("timestamp-query") ?? false,
    };
  });

  expect(result.sourceOperators).toEqual([
    "com.xenova.gemma4.DecodeQkvProj",
    "com.xenova.gemma4.DecodeQkNormRope",
  ]);
  expect(result.implementation).toBe("shared-storage-k-cache");
  expect(result.qkvArtifactSha256).toBe(
    "63482ab46577cc82b15879a8db0b0fea4515fc690741c34f4e47fb2d6faab1e3",
  );
  expect(result.kNormRopeArtifactSha256).toBe(
    "b0a44ef55d5dc0d9a827d7f4171fe36c15a6f37a4f74be98bae701708e88a374",
  );
  expect(result.cachePosition).toBe(10);
  expect(result.cacheElementOffset).toBe(2560);
  expect(result.qMaximumAbsoluteError).toBe(0);
  expect(result.qMaximumRelativeError).toBe(0);
  expect(result.kMaximumAbsoluteError).toBe(0);
  expect(result.kMaximumRelativeError).toBe(0);
  expect(result.vMaximumAbsoluteError).toBe(0);
  expect(result.vMaximumRelativeError).toBe(0);
  expect(result.normalizedKMaximumAbsoluteError).toBe(0);
  expect(result.normalizedKMaximumRelativeError).toBe(0);
  expect(result.gpuBufferAllocations).toBe(12);
  expect(result.allocationsPerDispatch).toBe(0);
  expect(result.cpuReadbacksBetweenKernels).toBe(0);
  expect(result.gpuCopiesBetweenKernels).toBe(0);
  expect(result.gpuKernelPairsPerSample).toBe(timestampQuerySupported ? 100 : null);
  if (timestampQuerySupported) {
    expect(result.gpuKernelPairSamplesMs).toHaveLength(10);
    expect(result.gpuKernelPairMedianMs).toBeGreaterThan(0);
    expect(result.gpuKernelPairP95Ms).toBeGreaterThanOrEqual(result.gpuKernelPairMedianMs ?? 0);
  }
});

test("matches Hugging Face fused layer-0 decode attention", async ({ page }) => {
  await page.goto("/");
  const webGpuAvailable = await page.evaluate(() => Boolean(navigator.gpu));
  test.skip(!webGpuAvailable, "Chrome does not expose WebGPU on this machine");

  const { result, timestampQuerySupported } = await page.evaluate(async () => {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    const modulePath = "/src/webgpu/decode-attention.ts";
    const { benchmarkDecodeAttention } = await import(modulePath);
    return {
      result: await benchmarkDecodeAttention(2),
      timestampQuerySupported: adapter?.features.has("timestamp-query") ?? false,
    };
  });

  expect(result.sourceOperator).toBe("Gemma4DecodeAttentionPartial");
  expect(result.sourceVariant).toBe("fixed-subgroup-32");
  expect(result.artifactSha256).toBe(
    "4ad8f65ebbaf1f71fbcb4ea20e22906e5bd4fa2765b077a6ddf3183b22277b97",
  );
  expect(result.sourceCaptureSha256).toBe(
    "fa4d670c13f3f7e1d271040b994aefb85106fe0b0343646907fb54cc9b907f2b",
  );
  expect(result.qHeads).toBe(8);
  expect(result.kvHeads).toBe(1);
  expect(result.headDim).toBe(256);
  expect(result.keyLength).toBe(11);
  expect(result.queryOffset).toBe(10);
  expect(result.window).toBe(512);
  expect(result.workgroupSize).toBe(256);
  expect(result.chunkCount).toBe(32);
  expect(result.workgroupCount).toBe(256);
  expect(result.outputMaximumAbsoluteError).toBe(0);
  expect(result.outputMaximumRelativeError).toBe(0);
  expect(result.gpuBufferAllocations).toBe(10);
  expect(result.allocationsPerDispatch).toBe(0);
  expect(result.gpuKernelDispatchesPerSample).toBe(timestampQuerySupported ? 20 : null);
  if (timestampQuerySupported) {
    expect(result.gpuKernelSamplesMs).toHaveLength(10);
    expect(result.gpuKernelMedianMs).toBeGreaterThan(0);
    expect(result.gpuKernelP95Ms).toBeGreaterThanOrEqual(result.gpuKernelMedianMs ?? 0);
  }
});

test("composes QKV through reusable K/V cache into decode attention", async ({ page }) => {
  await page.goto("/");
  const webGpuAvailable = await page.evaluate(() => Boolean(navigator.gpu));
  test.skip(!webGpuAvailable, "Chrome does not expose WebGPU on this machine");

  const result = await page.evaluate(async () => {
    const modulePath = "/src/webgpu/decode-attention-block.ts";
    const { benchmarkDecodeAttentionBlock } = await import(modulePath);
    return benchmarkDecodeAttentionBlock(2);
  });

  expect(result.sourceOperators).toEqual([
    "com.xenova.gemma4.DecodeRmsSrq",
    "com.xenova.gemma4.DecodeQkvProj",
    "com.xenova.gemma4.DecodeQkNormRope",
    "RMSNorm",
    "Gemma4DecodeAttentionPartial",
    "com.xenova.gemma4.DecodeOprojNorm",
  ]);
  expect(result.implementation).toBe("shared-qkv-reusable-kv-cache-oproj");
  expect(result.cacheCapacity).toBe(11);
  expect(result.cacheLength).toBe(11);
  expect(result.cachePosition).toBe(10);
  expect(result.dispatchesPerToken).toBe(6);
  expect(result.qMaximumAbsoluteError).toBe(0);
  expect(result.qMaximumRelativeError).toBe(0);
  expect(result.rawKMaximumAbsoluteError).toBe(0);
  expect(result.rawKMaximumRelativeError).toBe(0);
  expect(result.cachedKMaximumAbsoluteError).toBe(0);
  expect(result.cachedKMaximumRelativeError).toBe(0);
  expect(result.cachedVMaximumAbsoluteError).toBe(0);
  expect(result.cachedVMaximumRelativeError).toBe(0);
  expect(result.attentionMaximumAbsoluteError).toBe(0);
  expect(result.attentionMaximumRelativeError).toBe(0);
  expect(result.hiddenMaximumAbsoluteError).toBe(0);
  expect(result.hiddenMaximumRelativeError).toBe(0);
  expect(result.ffnInputBitMismatches).toBe(0);
  expect(result.ffnInputSumMaximumAbsoluteError).toBe(0);
  expect(result.ffnInputSumMaximumRelativeError).toBe(0);
  expect(result.gpuBufferAllocations).toBe(28);
  expect(result.allocationsPerDispatch).toBe(0);
  expect(result.cpuReadbacksBetweenKernels).toBe(0);
  expect(result.gpuCopiesBetweenKernels).toBe(0);
});

test("matches Hugging Face fused layer-0 DecodeOprojNorm", async ({ page }) => {
  await page.goto("/");
  const webGpuAvailable = await page.evaluate(() => Boolean(navigator.gpu));
  test.skip(!webGpuAvailable, "Chrome does not expose WebGPU on this machine");

  const result = await page.evaluate(async () => {
    const modulePath = "/src/webgpu/decode-oproj-norm.ts";
    const { benchmarkDecodeOprojNorm } = await import(modulePath);
    return benchmarkDecodeOprojNorm(2);
  });

  expect(result.sourceOperator).toBe("com.xenova.gemma4.DecodeOprojNorm");
  expect(result.sourceVariant).toBe("fused-fixed-subgroup-32");
  expect(result.artifactSha256).toBe(
    "d8ec21da0edcccdfd478c76e90215b79d4bae5a4f58eebf8b1de355c474a223d",
  );
  expect(result.sourceMetadataSha256).toBe(
    "1b81dd537bc0418ce74d93ee5dcf8c0b5d4b70c4c8621bb2ec3ae15c3d0dacdf",
  );
  expect(result.sourceTensorsSha256).toBe(
    "b5eff21d1af5f8826cd00a2a01d0830462fad78505a803c50ef1ca12b8e2ac52",
  );
  expect(result.inFeatures).toBe(2048);
  expect(result.outFeatures).toBe(1536);
  expect(result.workgroupSize).toBe(256);
  expect(result.workgroupCount).toBe(192);
  expect(result.hiddenMaximumAbsoluteError).toBe(0);
  expect(result.hiddenMaximumRelativeError).toBe(0);
  expect(result.ffnInputBitMismatches).toBe(0);
  expect(result.ffnInputSumMaximumAbsoluteError).toBe(0);
  expect(result.ffnInputSumMaximumRelativeError).toBe(0);
  expect(result.allocationsPerDispatch).toBe(0);
});

test("matches Hugging Face layer-0 DecodeGateUpNormPresrq integer codes", async ({ page }) => {
  await page.goto("/");
  const webGpuAvailable = await page.evaluate(() => Boolean(navigator.gpu));
  test.skip(!webGpuAvailable, "Chrome does not expose WebGPU on this machine");

  const result = await page.evaluate(async () => {
    const modulePath = "/src/webgpu/decode-gate-up-presrq.ts";
    const { runDecodeGateUpPresrq } = await import(modulePath);
    return runDecodeGateUpPresrq();
  });

  expect(result.sourceOperator).toBe("com.xenova.gemma4.DecodeGateUpNormPresrq");
  expect(result.sourceVariant).toBe("presrq-codes-fixed-subgroup-32");
  expect(result.sourceMetadataSha256).toBe(
    "05a79837b22bfec2d8d832970f38ba624624c1db8f4abaa5c26755730c95d1d6",
  );
  expect(result.sourceTensorsSha256).toBe(
    "0ee88858daf8e800bae18bb15c8901d578e5a50a32a4e3e4ffe9e426d02d43c0",
  );
  expect(result.workgroupSize).toBe(64);
  expect(result.workgroupCount).toBe(768);
  expect(result.outputElements).toBe(6144);
  expect(result.capturedBufferElements).toBe(12288);
  expect(result.codeBitMismatches).toBe(0);
  expect(result.outputBitMismatches).toBe(result.signedZeroBitMismatches);
  expect(result.outputBitMismatches).toBe(
    result.positiveToNegativeZeroMismatches + result.negativeToPositiveZeroMismatches,
  );
  expect(result.gpuBufferAllocations).toBe(10);
  expect(result.allocationsPerDispatch).toBe(0);
});

test("matches Hugging Face layer-0 DecodeDownNormAddFused", async ({ page }) => {
  await page.goto("/");
  const webGpuAvailable = await page.evaluate(() => Boolean(navigator.gpu));
  test.skip(!webGpuAvailable, "Chrome does not expose WebGPU on this machine");

  const result = await page.evaluate(async () => {
    const modulePath = "/src/webgpu/decode-down-norm-add.ts";
    const { runDecodeDownNormAdd } = await import(modulePath);
    return runDecodeDownNormAdd();
  });

  expect(result.sourceOperator).toBe("com.xenova.gemma4.DecodeDownNormAddFused");
  expect(result.sourceVariant).toBe("codes-fixed-subgroup-32");
  expect(result.workgroupSize).toBe(256);
  expect(result.workgroupCount).toBe(384);
  expect(result.hiddenMaximumAbsoluteError).toBe(0);
  expect(result.hiddenMaximumRelativeError).toBe(0);
  expect(result.hiddenBitMismatches).toBe(0);
  expect(result.gpuBufferAllocations).toBe(8);
  expect(result.allocationsPerDispatch).toBe(0);
});

test("matches Hugging Face layer-0 DecodePleGateCodes", async ({ page }) => {
  await page.goto("/");
  const webGpuAvailable = await page.evaluate(() => Boolean(navigator.gpu));
  test.skip(!webGpuAvailable, "Chrome does not expose WebGPU on this machine");

  const result = await page.evaluate(async () => {
    const modulePath = "/src/webgpu/decode-ple-gate-codes.ts";
    const { runDecodePleGateCodes } = await import(modulePath);
    return runDecodePleGateCodes();
  });

  expect(result.sourceOperator).toBe("com.xenova.gemma4.DecodePleGateCodes");
  expect(result.sourceVariant).toBe("codes-fixed-subgroup-32");
  expect(result.workgroupSize).toBe(32);
  expect(result.workgroupCount).toBe(256);
  expect(result.outputMaximumAbsoluteError).toBe(0);
  expect(result.outputMaximumRelativeError).toBe(0);
  expect(result.nonzeroBitMismatches).toBe(0);
  expect(result.outputBitMismatches).toBe(result.signedZeroBitMismatches);
  expect(result.gpuBufferAllocations).toBe(8);
  expect(result.allocationsPerDispatch).toBe(0);
});

test("matches Hugging Face layer-0 DecodePleProjNormCodes", async ({ page }) => {
  await page.goto("/");
  const webGpuAvailable = await page.evaluate(() => Boolean(navigator.gpu));
  test.skip(!webGpuAvailable, "Chrome does not expose WebGPU on this machine");

  const result = await page.evaluate(async () => {
    const modulePath = "/src/webgpu/decode-ple-proj-norm-codes.ts";
    const { runDecodePleProjNormCodes } = await import(modulePath);
    return runDecodePleProjNormCodes();
  });

  expect(result.sourceOperator).toBe("com.xenova.gemma4.DecodePleProjNormCodes");
  expect(result.sourceVariant).toBe("codes-fixed-subgroup-32");
  expect(result.workgroupSize).toBe(256);
  expect(result.workgroupCount).toBe(96);
  expect(result.hiddenMaximumAbsoluteError).toBe(0);
  expect(result.hiddenMaximumRelativeError).toBe(0);
  expect(result.hiddenBitMismatches).toBe(0);
  expect(result.nextInputMaximumAbsoluteError).toBe(0);
  expect(result.nextInputMaximumRelativeError).toBe(0);
  expect(result.nextInputNonzeroBitMismatches).toBe(0);
  expect(result.nextInputBitMismatches).toBe(result.nextInputSignedZeroBitMismatches);
  expect(result.nextSumMaximumAbsoluteError).toBe(0);
  expect(result.nextSumMaximumRelativeError).toBe(0);
  expect(result.nextSumBitMismatches).toBe(0);
  expect(result.gpuBufferAllocations).toBe(10);
  expect(result.allocationsPerDispatch).toBe(0);
});

test("composes the complete layer-0 MLP and PLE block with shared GPU buffers", async ({ page }) => {
  await page.goto("/");
  const webGpuAvailable = await page.evaluate(() => Boolean(navigator.gpu));
  test.skip(!webGpuAvailable, "Chrome does not expose WebGPU on this machine");

  const result = await page.evaluate(async () => {
    const modulePath = "/src/webgpu/decode-mlp-ple-block.ts";
    const { runDecodeMlpPleBlock } = await import(modulePath);
    return runDecodeMlpPleBlock();
  });

  expect(result.sourceOperators).toEqual([
    "com.xenova.gemma4.DecodeGateUpNormPresrq",
    "com.xenova.gemma4.DecodeDownNormAddFused",
    "com.xenova.gemma4.DecodePleGateCodes",
    "com.xenova.gemma4.DecodePleProjNormCodes",
  ]);
  expect(result.implementation).toBe("shared-storage-four-dispatch");
  expect(result.dispatchesPerToken).toBe(4);
  expect(result.hiddenMaximumAbsoluteError).toBe(0);
  expect(result.hiddenMaximumRelativeError).toBe(0);
  expect(result.hiddenBitMismatches).toBe(0);
  expect(result.nextInputMaximumAbsoluteError).toBe(0);
  expect(result.nextInputMaximumRelativeError).toBe(0);
  expect(result.nextInputNonzeroBitMismatches).toBe(0);
  expect(result.nextSumMaximumAbsoluteError).toBe(0);
  expect(result.nextSumMaximumRelativeError).toBe(0);
  expect(result.nextSumBitMismatches).toBe(0);
  expect(result.gpuBufferAllocations).toBe(29);
  expect(result.allocationsPerDispatch).toBe(0);
  expect(result.cpuReadbacksBetweenKernels).toBe(0);
  expect(result.gpuCopiesBetweenKernels).toBe(0);
});

test("runs the layer-15 double-wide int2 MLP pipeline profile", async ({ page }) => {
  await page.goto("/");
  const webGpuAvailable = await page.evaluate(() => Boolean(navigator.gpu));
  test.skip(!webGpuAvailable, "Chrome does not expose WebGPU on this machine");

  const result = await page.evaluate(async () => {
    const blockModulePath = "/src/webgpu/decode-mlp-ple-block.ts";
    const deviceModulePath = "/src/webgpu/device.ts";
    const fixtureModulePath = "/src/model/decode-mlp-ple-fixture.ts";
    const {
      compileDecodeMlpPleBlockPipelines,
      createDecodeMlpPleBlockResources,
      encodeDecodeMlpPleBlock,
    } = await import(blockModulePath);
    const { getWebGpuDevice } = await import(deviceModulePath);
    const { loadDecodeMlpPleFixture } = await import(fixtureModulePath);
    const [device, fixture] = await Promise.all([getWebGpuDevice(), loadDecodeMlpPleFixture()]);
    const pipelines = await compileDecodeMlpPleBlockPipelines(device, "sliding-int2");
    const gateBits = new Uint32Array(12288 * 96).fill(0xaaaaaaaa);
    const upBits = new Uint32Array(12288 * 96).fill(0xaaaaaaaa);
    gateBits[0] = 0xffffffff;
    upBits[0] = 0xffffffff;
    const gateScales = new Float32Array(12288);
    const upScales = new Float32Array(12288);
    gateScales[0] = 1;
    upScales[0] = 1;
    const downBits = new Uint32Array(1536 * 768).fill(0xaaaaaaaa);
    const downScales = new Float32Array(1536);
    downBits[0] = 0xffffffff;
    downScales[0] = 1;
    const preMlpInputBits = new Uint16Array(1536);
    preMlpInputBits.fill(0x3c00, 0, 16);
    const syntheticFixture = {
      ...fixture,
      preMlpInputBits,
      preMlpSum: Float32Array.from([16]),
    };
    const neutralPleNormWeights = fixture.pleNormWeights.slice();
    neutralPleNormWeights[3072] = 1;
    const projection = (
      packedWeights: Uint32Array,
      rowScales: Float32Array,
      inputScale = 1,
      outputScale = 1,
    ) => ({ packedWeights, rowScales, inputScale, outputScale });
    const materialized = {
      layer: {
        profile: "sliding-int2",
        mlp: {
          gate: projection(gateBits, gateScales, 1, 0.6181102395057678),
          up: projection(upBits, upScales, 1, 0.6181102395057678),
          down: projection(
            downBits,
            downScales,
            27.842519760131836,
            16.64207649230957,
          ),
        },
        ple: {
          inputGate: projection(fixture.pleGateWeights, fixture.pleGateRowScales),
          projection: projection(fixture.pleProjectionWeights, new Float32Array(1536)),
        },
        norms: { postFeedforward: fixture.postFfNorm },
      },
      pleNormWeights: neutralPleNormWeights,
      nextInputScale: 0.4842597544193268,
    };
    const resources = createDecodeMlpPleBlockResources(
      device,
      pipelines,
      syntheticFixture,
      undefined,
      materialized,
    );
    device.pushErrorScope("validation");
    device.pushErrorScope("internal");
    const gateReadback = device.createBuffer({
      label: "Synthetic int2 gate/up readback",
      size: resources.gateOutput.size + fixture.hiddenBeforeDown.byteLength,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    try {
      const encoder = device.createCommandEncoder({ label: "Synthetic layer-15 MLP/PLE" });
      encodeDecodeMlpPleBlock(encoder, pipelines, resources);
      encoder.copyBufferToBuffer(
        resources.gateOutput,
        0,
        gateReadback,
        0,
        resources.gateOutput.size,
      );
      encoder.copyBufferToBuffer(
        resources.hidden,
        0,
        gateReadback,
        resources.gateOutput.size,
        fixture.hiddenBeforeDown.byteLength,
      );
      device.queue.submit([encoder.finish()]);
      await gateReadback.mapAsync(GPUMapMode.READ);
      const copied = gateReadback.getMappedRange().slice(0);
      const gateOutput = new Uint16Array(
        copied,
        0,
        resources.gateOutput.size / Uint16Array.BYTES_PER_ELEMENT,
      );
      const hidden = new Float32Array(
        copied,
        resources.gateOutput.size,
        fixture.hiddenBeforeDown.length,
      );
      gateReadback.unmap();
      const internalError = await device.popErrorScope();
      const validationError = await device.popErrorScope();
      return {
        bitWidth: pipelines.bitWidth,
        intermediateFeatures: pipelines.intermediateFeatures,
        gateUpWorkgroupCount: pipelines.gateUpWorkgroupCount,
        gateOutputBytes: resources.gateOutput.size,
        firstGateCodeBits: gateOutput[0],
        nonzeroGateCodes: gateOutput.filter((bits) => (bits & 0x7fff) !== 0).length,
        changedHiddenElements: hidden.filter(
          (value, index) => value !== fixture.hiddenBeforeDown[index],
        ).length,
        dispatches: 4,
        gpuError: internalError?.message ?? validationError?.message ?? null,
        labels: [
          pipelines.gateUp.label,
          pipelines.down.label,
          pipelines.pleGate.label,
          pipelines.pleProjection.label,
        ],
      };
    } finally {
      gateReadback.destroy();
      for (const buffer of resources.buffers) buffer.destroy();
    }
  });

  expect(result).toEqual({
    bitWidth: 2,
    intermediateFeatures: 12288,
    gateUpWorkgroupCount: 3072,
    gateOutputBytes: 24576,
    firstGateCodeBits: 0x4880,
    nonzeroGateCodes: 1,
    changedHiddenElements: 1,
    dispatches: 4,
    gpuError: null,
    labels: [
      "MLP/PLE int2 gate/up",
      "MLP/PLE int2 down",
      "MLP/PLE input gate",
      "MLP/PLE projection",
    ],
  });
});

test("compiles all four generic Gemma layer profiles", async ({ page }) => {
  await page.goto("/");
  const webGpuAvailable = await page.evaluate(() => Boolean(navigator.gpu));
  test.skip(!webGpuAvailable, "Chrome does not expose WebGPU on this machine");

  const profiles = await page.evaluate(async () => {
    const layerModulePath = "/src/webgpu/decode-layer.ts";
    const deviceModulePath = "/src/webgpu/device.ts";
    const { getGemmaDecodeLayerPipelines } = await import(layerModulePath);
    const { getWebGpuDevice } = await import(deviceModulePath);
    const device = await getWebGpuDevice();
    const names = ["sliding-int4", "full-int4", "sliding-int2", "full-int2"] as const;
    return Promise.all(names.map(async (profile) => {
      const pipelines = await getGemmaDecodeLayerPipelines(device, profile);
      return {
        profile: pipelines.profile,
        headDim: pipelines.attention.headDim,
        qOutFeatures: pipelines.attention.qOutFeatures,
        kvOutFeatures: pipelines.attention.kvOutFeatures,
        qkvWorkgroupCount: pipelines.attention.qkvWorkgroupCount,
        mlpBits: pipelines.mlp.bitWidth,
        intermediateFeatures: pipelines.mlp.intermediateFeatures,
      };
    }));
  });

  expect(profiles).toEqual([
    {
      profile: "sliding-int4",
      headDim: 256,
      qOutFeatures: 2048,
      kvOutFeatures: 256,
      qkvWorkgroupCount: 1280,
      mlpBits: 4,
      intermediateFeatures: 6144,
    },
    {
      profile: "full-int4",
      headDim: 512,
      qOutFeatures: 4096,
      kvOutFeatures: 512,
      qkvWorkgroupCount: 2560,
      mlpBits: 4,
      intermediateFeatures: 6144,
    },
    {
      profile: "sliding-int2",
      headDim: 256,
      qOutFeatures: 2048,
      kvOutFeatures: 256,
      qkvWorkgroupCount: 1280,
      mlpBits: 2,
      intermediateFeatures: 12288,
    },
    {
      profile: "full-int2",
      headDim: 512,
      qOutFeatures: 4096,
      kvOutFeatures: 512,
      qkvWorkgroupCount: 2560,
      mlpBits: 2,
      intermediateFeatures: 12288,
    },
  ]);
});

test("runs the canonical layer-4 full-attention profile", async ({ page }) => {
  await page.goto("/");
  const webGpuAvailable = await page.evaluate(() => Boolean(navigator.gpu));
  test.skip(!webGpuAvailable, "Chrome does not expose WebGPU on this machine");

  const result = await page.evaluate(async () => {
    const layerModulePath = "/src/webgpu/decode-layer.ts";
    const blockModulePath = "/src/webgpu/decode-attention-block.ts";
    const deviceModulePath = "/src/webgpu/device.ts";
    const { getGemmaDecodeLayerPipelines } = await import(layerModulePath);
    const {
      createGemmaDecodeAttentionBlockResources,
      encodeDecodeAttentionBlock,
    } = await import(blockModulePath);
    const { getWebGpuDevice } = await import(deviceModulePath);
    const device = await getWebGpuDevice();
    const pipelines = await getGemmaDecodeLayerPipelines(device, "full-int4");
    const projection = (
      packedWeights: Uint32Array,
      rowScales: Float32Array,
      inputScale = 0,
      outputScale = 0,
    ) => ({ packedWeights, rowScales, inputScale, outputScale });
    const qkvRows = 4096 + 512 + 512;
    const qkvWeights = new Uint32Array(qkvRows * 192).fill(0x88888888);
    const oprojWeights = new Uint32Array(1536 * 512).fill(0x88888888);
    const emptyProjection = projection(new Uint32Array(1), new Float32Array(1));
    const layer = {
      layerIndex: 4,
      profile: "full-int4",
      qkv: {
        packedWeights: qkvWeights,
        rowScales: new Float32Array(qkvRows),
        inputScale: 0,
        outputScales: new Float32Array(3),
      },
      outputProjection: projection(oprojWeights, new Float32Array(1536)),
      mlp: {
        gate: emptyProjection,
        up: emptyProjection,
        down: emptyProjection,
      },
      ple: {
        inputGate: emptyProjection,
        projection: emptyProjection,
      },
      norms: {
        input: new Float32Array(1536).fill(1),
        q: new Float32Array(512).fill(1),
        k: new Float32Array(512).fill(1),
        postAttention: new Float32Array(1536).fill(1),
        preFeedforward: new Float32Array(1536).fill(1),
        postFeedforward: new Float32Array(1536).fill(1),
        postPerLayerInput: new Float32Array(1536).fill(1),
        oProjectionFused: new Float32Array(3072).fill(1),
      },
      layerScalar: 1,
      sourceBytes: 0,
    };
    const runtime = {
      hidden: new Float32Array(1536),
      cosine: new Float32Array(256).fill(1),
      sine: new Float32Array(256),
      keyCache: new Float32Array(512),
      valueCache: new Float32Array(512),
      keyLength: 1,
      queryOffset: 0,
      qHeads: 8,
      kvHeads: 1,
      window: 0,
    };
    const resources = createGemmaDecodeAttentionBlockResources(
      device,
      pipelines.attention,
      layer,
      runtime,
    );
    const readback = device.createBuffer({
      label: "Synthetic full-attention readback",
      size: 1536 * 6 + 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    device.pushErrorScope("validation");
    device.pushErrorScope("internal");
    try {
      const encoder = device.createCommandEncoder({ label: "Synthetic layer-4 attention" });
      encodeDecodeAttentionBlock(encoder, pipelines.attention, resources);
      encoder.copyBufferToBuffer(resources.hiddenBuffer, 0, readback, 0, 1536 * 4);
      encoder.copyBufferToBuffer(resources.ffnInputBuffer, 0, readback, 1536 * 4, 1536 * 2);
      encoder.copyBufferToBuffer(resources.ffnInputSumBuffer, 0, readback, 1536 * 6, 4);
      device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const bytes = new Uint8Array(readback.getMappedRange().slice(0));
      readback.unmap();
      const internalError = await device.popErrorScope();
      const validationError = await device.popErrorScope();
      return {
        profile: pipelines.profile,
        dispatches: 6,
        headDim: pipelines.attention.headDim,
        qOutFeatures: pipelines.attention.qOutFeatures,
        kvOutFeatures: pipelines.attention.kvOutFeatures,
        qkvWorkgroupCount: pipelines.attention.qkvWorkgroupCount,
        nonzeroOutputBytes: bytes.filter((value) => value !== 0).length,
        gpuError: internalError?.message ?? validationError?.message ?? null,
      };
    } finally {
      readback.destroy();
      for (const buffer of resources.buffers) buffer.destroy();
      resources.cache.destroy();
    }
  });

  expect(result).toEqual({
    profile: "full-int4",
    dispatches: 6,
    headDim: 512,
    qOutFeatures: 4096,
    kvOutFeatures: 512,
    qkvWorkgroupCount: 2560,
    nonzeroOutputBytes: 0,
    gpuError: null,
  });
});

test("runs generic shared-KV layer-15 and layer-19 profiles", async ({ page }) => {
  await page.goto("/");
  const webGpuAvailable = await page.evaluate(() => Boolean(navigator.gpu));
  test.skip(!webGpuAvailable, "Chrome does not expose WebGPU on this machine");

  const results = await page.evaluate(async () => {
    const layerModulePath = "/src/webgpu/decode-layer.ts";
    const attentionModulePath = "/src/webgpu/decode-attention-block.ts";
    const mlpModulePath = "/src/webgpu/decode-mlp-ple-block.ts";
    const cacheModulePath = "/src/webgpu/decode-kv-cache.ts";
    const fixtureModulePath = "/src/model/decode-mlp-ple-fixture.ts";
    const deviceModulePath = "/src/webgpu/device.ts";
    const {
      encodeGemmaDecodeLayer,
      gemmaDecodeLayerDispatchCount,
      getGemmaDecodeLayerPipelines,
    } = await import(layerModulePath);
    const { createGemmaDecodeSharedKvAttentionBlockResources } = await import(
      attentionModulePath
    );
    const { createDecodeMlpPleBlockResources } = await import(mlpModulePath);
    const { DecodeKvCache } = await import(cacheModulePath);
    const { loadDecodeMlpPleFixture } = await import(fixtureModulePath);
    const { getWebGpuDevice } = await import(deviceModulePath);
    const [device, fixture] = await Promise.all([
      getWebGpuDevice(),
      loadDecodeMlpPleFixture(),
    ]);
    const projection = (
      packedWeights: Uint32Array,
      rowScales: Float32Array,
      inputScale = 0,
      outputScale = 0,
    ) => ({ packedWeights, rowScales, inputScale, outputScale });
    const mlpGate = projection(
      new Uint32Array(12288 * 96).fill(0xaaaaaaaa),
      new Float32Array(12288),
    );
    const mlpUp = projection(
      new Uint32Array(12288 * 96).fill(0xaaaaaaaa),
      new Float32Array(12288),
    );
    const mlpDown = projection(
      new Uint32Array(1536 * 768).fill(0xaaaaaaaa),
      new Float32Array(1536),
    );
    const configs = [
      { layerIndex: 15, profile: "sliding-int2", headDim: 256, qOut: 2048, kvOut: 256, window: 512 },
      { layerIndex: 19, profile: "full-int2", headDim: 512, qOut: 4096, kvOut: 512, window: 0 },
    ] as const;
    const executed = [];
    for (const config of configs) {
      const pipelines = await getGemmaDecodeLayerPipelines(device, config.profile);
      const qkvRows = config.qOut + 2 * config.kvOut;
      const layer = {
        layerIndex: config.layerIndex,
        profile: config.profile,
        qkv: {
          packedWeights: new Uint32Array(qkvRows * 192).fill(0x88888888),
          rowScales: new Float32Array(qkvRows),
          inputScale: 0,
          outputScales: new Float32Array(3),
        },
        outputProjection: projection(
          new Uint32Array(1536 * (config.qOut / 8)).fill(0x88888888),
          new Float32Array(1536),
        ),
        mlp: { gate: mlpGate, up: mlpUp, down: mlpDown },
        ple: {
          inputGate: projection(fixture.pleGateWeights, fixture.pleGateRowScales),
          projection: projection(fixture.pleProjectionWeights, fixture.pleProjectionRowScales),
        },
        norms: {
          input: new Float32Array(1536).fill(1),
          q: new Float32Array(config.headDim).fill(1),
          k: null,
          postAttention: new Float32Array(1536).fill(1),
          preFeedforward: new Float32Array(1536).fill(1),
          postFeedforward: new Float32Array(1536).fill(1),
          postPerLayerInput: new Float32Array(1536).fill(1),
          oProjectionFused: new Float32Array(3072).fill(1),
        },
        layerScalar: 1,
        sourceBytes: 0,
      };
      const sourceCache = new DecodeKvCache(device, {
        capacity: 1,
        kvHeads: 1,
        headDim: config.headDim,
        label: `Synthetic shared cache ${config.layerIndex}`,
      });
      sourceCache.writeTokens(
        device.queue,
        0,
        new Float32Array(config.headDim),
        new Float32Array(config.headDim),
      );
      const activationUsage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
      const activationInput = device.createBuffer({
        label: `Synthetic prior input ${config.layerIndex}`,
        size: 1536 * Uint16Array.BYTES_PER_ELEMENT,
        usage: activationUsage,
      });
      const activationSum = device.createBuffer({
        label: `Synthetic prior sum ${config.layerIndex}`,
        size: 4,
        usage: activationUsage,
      });
      const activationHidden = device.createBuffer({
        label: `Synthetic prior hidden ${config.layerIndex}`,
        size: 1536 * Float32Array.BYTES_PER_ELEMENT,
        usage: activationUsage,
      });
      const attention = createGemmaDecodeSharedKvAttentionBlockResources(
        device,
        pipelines.attention,
        layer,
        {
          hidden: new Float32Array(1536),
          cosine: new Float32Array(config.headDim / 2).fill(1),
          sine: new Float32Array(config.headDim / 2),
          keyLength: 1,
          queryOffset: 0,
          qHeads: 8,
          kvHeads: 1,
          window: config.window,
          sourceCache,
        },
        {
          input: activationInput,
          inputSum: activationSum,
          hidden: activationHidden,
        },
      );
      const mlp = createDecodeMlpPleBlockResources(
        device,
        pipelines.mlp,
        fixture,
        {
          preMlpInput: attention.ffnInputBuffer,
          preMlpSum: attention.ffnInputSumBuffer,
          hidden: attention.hiddenBuffer,
        },
        {
          layer,
          pleNormWeights: fixture.pleNormWeights,
          nextInputScale: 0,
        },
      );
      device.pushErrorScope("validation");
      device.pushErrorScope("internal");
      try {
        const encoder = device.createCommandEncoder({
          label: `Synthetic generic layer ${config.layerIndex}`,
        });
        encodeGemmaDecodeLayer(encoder, pipelines, { attention, mlp });
        device.queue.submit([encoder.finish()]);
        await device.queue.onSubmittedWorkDone();
        const internalError = await device.popErrorScope();
        const validationError = await device.popErrorScope();
        executed.push({
          layerIndex: config.layerIndex,
          profile: config.profile,
          dispatches: gemmaDecodeLayerDispatchCount({ attention, mlp }),
          attentionWritesKv: attention.writesKvCache,
          qProjectionWorkgroups: attention.qkvWorkgroupCount,
          sourceCacheLength: sourceCache.length,
          gpuError: internalError?.message ?? validationError?.message ?? null,
        });
      } finally {
        for (const buffer of mlp.buffers) buffer.destroy();
        for (const buffer of attention.buffers) buffer.destroy();
        activationInput.destroy();
        activationSum.destroy();
        activationHidden.destroy();
        sourceCache.destroy();
      }
    }
    return executed;
  });

  expect(results).toEqual([
    {
      layerIndex: 15,
      profile: "sliding-int2",
      dispatches: 7,
      attentionWritesKv: false,
      qProjectionWorkgroups: 1024,
      sourceCacheLength: 1,
      gpuError: null,
    },
    {
      layerIndex: 19,
      profile: "full-int2",
      dispatches: 7,
      attentionWritesKv: false,
      qProjectionWorkgroups: 2048,
      sourceCacheLength: 1,
      gpuError: null,
    },
  ]);
});

test("projects final activations through the int2 LM head", async ({ page }) => {
  await page.goto("/");
  const webGpuAvailable = await page.evaluate(() => Boolean(navigator.gpu));
  test.skip(!webGpuAvailable, "Chrome does not expose WebGPU on this machine");

  const result = await page.evaluate(async () => {
    const lmHeadModulePath = "/src/webgpu/decode-lm-head.ts";
    const deviceModulePath = "/src/webgpu/device.ts";
    const {
      createGemmaLmHeadResources,
      destroyGemmaLmHeadResources,
      encodeGemmaLmHead,
      getGemmaLmHeadPipeline,
    } = await import(lmHeadModulePath);
    const { getWebGpuDevice } = await import(deviceModulePath);
    const device = await getWebGpuDevice();
    const pipeline = await getGemmaLmHeadPipeline(device, 8);
    const activation = device.createBuffer({
      label: "Synthetic final activation",
      size: 1536 * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const activationSum = device.createBuffer({
      label: "Synthetic final activation sum",
      size: 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const packedWeights = new Uint32Array(8 * 96).fill(0xaaaaaaaa);
    packedWeights.fill(0xffffffff, 96, 192);
    packedWeights.fill(0x55555555, 192, 288);
    const rowScales = new Float32Array([1, 0.5, 0.25, 1, 1, 1, 1, 1]);
    device.queue.writeBuffer(activation, 0, new Float32Array(1536).fill(1));
    device.queue.writeBuffer(activationSum, 0, new Float32Array([1536]));
    const resources = createGemmaLmHeadResources(
      device,
      pipeline,
      { activation, activationSum },
      { packedWeights, rowScales, outputScale: 0 },
    );
    const readback = device.createBuffer({
      label: "Synthetic LM head readback",
      size: 8 * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    device.pushErrorScope("validation");
    device.pushErrorScope("internal");
    try {
      const encoder = device.createCommandEncoder({ label: "Synthetic int2 LM head" });
      encodeGemmaLmHead(encoder, pipeline, resources);
      encoder.copyBufferToBuffer(resources.logits, 0, readback, 0, readback.size);
      device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const logits = Array.from(new Float32Array(readback.getMappedRange().slice(0)));
      readback.unmap();
      const internalError = await device.popErrorScope();
      const validationError = await device.popErrorScope();
      return {
        logits,
        workgroupCount: pipeline.workgroupCount,
        gpuError: internalError?.message ?? validationError?.message ?? null,
      };
    } finally {
      readback.destroy();
      destroyGemmaLmHeadResources(resources);
      activation.destroy();
      activationSum.destroy();
    }
  });

  expect(result.workgroupCount).toBe(1);
  expect(result.gpuError).toBeNull();
  expect(result.logits[0]).toBeCloseTo(0, 3);
  expect(result.logits[1]).toBeCloseTo(768, 3);
  expect(result.logits[2]).toBe(-384);
  for (const logit of result.logits.slice(3)) expect(logit).toBeCloseTo(0, 3);
});

test("reduces logits to a deterministic greedy token on GPU", async ({ page }) => {
  await page.goto("/");
  const webGpuAvailable = await page.evaluate(() => Boolean(navigator.gpu));
  test.skip(!webGpuAvailable, "Chrome does not expose WebGPU on this machine");

  const result = await page.evaluate(async () => {
    const greedyModulePath = "/src/webgpu/decode-greedy.ts";
    const deviceModulePath = "/src/webgpu/device.ts";
    const {
      createGemmaGreedyResources,
      destroyGemmaGreedyResources,
      encodeGemmaGreedy,
      getGemmaGreedyPipelines,
      readGemmaGreedyResult,
    } = await import(greedyModulePath);
    const { getWebGpuDevice } = await import(deviceModulePath);
    const device = await getWebGpuDevice();
    const logits = new Float32Array(2049).fill(-4);
    logits[1] = 9;
    logits[1024] = 8;
    logits[2048] = 9;
    const logitsBuffer = device.createBuffer({
      label: "Synthetic greedy logits",
      size: logits.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(logitsBuffer, 0, logits);
    const pipelines = await getGemmaGreedyPipelines(device, logits.length);
    const resources = createGemmaGreedyResources(device, pipelines, logitsBuffer);
    device.pushErrorScope("validation");
    device.pushErrorScope("internal");
    try {
      const encoder = device.createCommandEncoder({ label: "Synthetic greedy reduction" });
      encodeGemmaGreedy(encoder, pipelines, resources, true);
      device.queue.submit([encoder.finish()]);
      const greedy = await readGemmaGreedyResult(resources);
      const internalError = await device.popErrorScope();
      const validationError = await device.popErrorScope();
      return {
        ...greedy,
        partialCount: pipelines.partialCount,
        gpuError: internalError?.message ?? validationError?.message ?? null,
      };
    } finally {
      destroyGemmaGreedyResources(resources);
      logitsBuffer.destroy();
    }
  });

  expect(result).toEqual({
    token: 1,
    logit: 9,
    partialCount: 3,
    gpuError: null,
  });
});

test("prepares token hidden and per-layer inputs on GPU", async ({ page }) => {
  await page.goto("/");
  const webGpuAvailable = await page.evaluate(() => Boolean(navigator.gpu));
  test.skip(!webGpuAvailable, "Chrome does not expose WebGPU on this machine");

  const result = await page.evaluate(async () => {
    const inputModulePath = "/src/webgpu/decode-input.ts";
    const deviceModulePath = "/src/webgpu/device.ts";
    const {
      createGemmaDecodeInputResources,
      destroyGemmaDecodeInputResources,
      encodeGemmaDecodeInput,
      getGemmaDecodeInputPipeline,
      uploadGemmaTokenInputs,
    } = await import(inputModulePath);
    const { getWebGpuDevice } = await import(deviceModulePath);
    const device = await getWebGpuDevice();
    const pipelines = await getGemmaDecodeInputPipeline(device);
    const resources = createGemmaDecodeInputResources(device, pipelines, {
      projectionBfloat16: new Uint32Array(8960 * 768),
      projectionNorm: new Float32Array(256).fill(1),
    });
    const hidden = Float32Array.from({ length: 1536 }, (_, index) => index / 1536);
    const perLayerEmbedding = new Float32Array(8960).fill(2);
    uploadGemmaTokenInputs(device, resources, { hidden, perLayerEmbedding });
    const readback = device.createBuffer({
      size: hidden.byteLength + perLayerEmbedding.byteLength,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    device.pushErrorScope("validation");
    device.pushErrorScope("internal");
    try {
      const encoder = device.createCommandEncoder();
      encodeGemmaDecodeInput(encoder, pipelines, resources);
      encoder.copyBufferToBuffer(resources.hidden, 0, readback, 0, hidden.byteLength);
      encoder.copyBufferToBuffer(
        resources.perLayerInputs,
        0,
        readback,
        hidden.byteLength,
        perLayerEmbedding.byteLength,
      );
      device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const bytes = readback.getMappedRange();
      const actualHidden = new Float32Array(bytes.slice(0, hidden.byteLength));
      const actualPle = new Float32Array(bytes.slice(hidden.byteLength));
      readback.unmap();
      const internalError = await device.popErrorScope();
      const validationError = await device.popErrorScope();
      return {
        hiddenFirst: Array.from(actualHidden.slice(0, 4)),
        pleFirst: Array.from(actualPle.slice(0, 4)),
        gpuError: internalError?.message ?? validationError?.message ?? null,
      };
    } finally {
      readback.destroy();
      destroyGemmaDecodeInputResources(resources);
    }
  });

  expect(result.gpuError).toBeNull();
  for (const [index, value] of result.hiddenFirst.entries()) {
    expect(value).toBeCloseTo(index / 1536, 8);
  }
  for (const value of result.pleFirst) expect(value).toBeCloseTo(Math.SQRT2, 6);
});

test("runs the complete ten-dispatch layer-0 decode plan", async ({ page }) => {
  await page.goto("/");
  const webGpuAvailable = await page.evaluate(() => Boolean(navigator.gpu));
  test.skip(!webGpuAvailable, "Chrome does not expose WebGPU on this machine");

  const result = await page.evaluate(async () => {
    const modulePath = "/src/webgpu/decode-layer0.ts";
    const { runDecodeLayer0 } = await import(modulePath);
    return runDecodeLayer0();
  });

  expect(result.sourceOperators).toHaveLength(10);
  expect(result.implementation).toBe("shared-hidden-pre-mlp-ten-dispatch");
  expect(result.dispatchesPerToken).toBe(10);
  expect(result.hiddenMaximumAbsoluteError).toBe(0);
  expect(result.hiddenMaximumRelativeError).toBe(0);
  expect(result.hiddenBitMismatches).toBe(0);
  expect(result.nextInputMaximumAbsoluteError).toBe(0);
  expect(result.nextInputMaximumRelativeError).toBe(0);
  expect(result.nextInputNonzeroBitMismatches).toBe(0);
  expect(result.nextSumMaximumAbsoluteError).toBe(0);
  expect(result.nextSumMaximumRelativeError).toBe(0);
  expect(result.nextSumBitMismatches).toBe(0);
  expect(result.gpuBufferAllocations).toBe(55);
  expect(result.allocationsPerDispatch).toBe(0);
  expect(result.cpuReadbacksBetweenKernels).toBe(0);
  expect(result.gpuCopiesBetweenKernels).toBe(0);
});

test("composes DecodeRmsSrq into DecodeQkvProj with shared GPU buffers", async ({ page }) => {
  await page.goto("/");
  const webGpuAvailable = await page.evaluate(() => Boolean(navigator.gpu));
  test.skip(!webGpuAvailable, "Chrome does not expose WebGPU on this machine");

  const { result, timestampQuerySupported } = await page.evaluate(async () => {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    const modulePath = "/src/webgpu/decode-rms-qkv.ts";
    const { benchmarkDecodeRmsQkv } = await import(modulePath);
    return {
      result: await benchmarkDecodeRmsQkv(2),
      timestampQuerySupported: adapter?.features.has("timestamp-query") ?? false,
    };
  });

  expect(result.sourceOperators).toEqual([
    "com.xenova.gemma4.DecodeRmsSrq",
    "com.xenova.gemma4.DecodeQkvProj",
  ]);
  expect(result.implementation).toBe("shared-storage");
  expect(result.rmsArtifactSha256).toBe(
    "75edf39811df47143afcf92fd8e64931820eae808e9a6a11a2b57e4464202c36",
  );
  expect(result.qkvArtifactSha256).toBe(
    "63482ab46577cc82b15879a8db0b0fea4515fc690741c34f4e47fb2d6faab1e3",
  );
  expect(result.qMaximumAbsoluteError).toBe(0);
  expect(result.qMaximumRelativeError).toBe(0);
  expect(result.kMaximumAbsoluteError).toBe(0);
  expect(result.kMaximumRelativeError).toBe(0);
  expect(result.vMaximumAbsoluteError).toBe(0);
  expect(result.vMaximumRelativeError).toBe(0);
  expect(result.gpuBufferAllocations).toBe(10);
  expect(result.allocationsPerDispatch).toBe(0);
  expect(result.cpuReadbacksBetweenKernels).toBe(0);
  expect(result.gpuKernelPairsPerSample).toBe(timestampQuerySupported ? 100 : null);
  if (timestampQuerySupported) {
    expect(result.gpuKernelPairMedianMs).toBeGreaterThan(0);
    expect(result.gpuKernelPairP95Ms).toBeGreaterThanOrEqual(result.gpuKernelPairMedianMs ?? 0);
  }
});