import { expect, test } from "@playwright/test";

test("projects a 32-row QAT block exactly like individual rows", async ({ page }) => {
  await page.goto("/");
  const webGpuAvailable = await page.evaluate(() => Boolean(navigator.gpu));
  test.skip(!webGpuAvailable, "Chrome does not expose WebGPU on this machine");

  const result = await page.evaluate(async () => {
    const qatModulePath = "/src/webgpu/prefill-qat-linear.ts";
    const deviceModulePath = "/src/webgpu/device.ts";
    const {
      createGemmaPrefillQatLinearResources,
      destroyGemmaPrefillQatLinearResources,
      encodeGemmaPrefillQatLinear,
      getGemmaPrefillQatLinearPipelines,
    } = await import(qatModulePath);
    const { getWebGpuDevice } = await import(deviceModulePath);
    const device = await getWebGpuDevice();
    const rows = 32;
    const inFeatures = 64;
    const outFeatures = 1024;
    const packedWeights = Uint32Array.from(
      { length: outFeatures * inFeatures / 8 },
      (_, index) => Math.imul(index + 17, 0x9e3779b1) >>> 0,
    );
    const rowScales = Float32Array.from(
      { length: outFeatures },
      (_, index) => Math.fround(0.0005 + (index % 31) * 0.00003),
    );
    const activation = Float32Array.from(
      { length: rows * inFeatures },
      (_, index) => Math.fround(Math.sin(index * 0.17) * 3.25),
    );
    const storageUpload = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    const activationBuffer = device.createBuffer({
      size: activation.byteLength,
      usage: storageUpload,
    });
    const singleActivationBuffer = device.createBuffer({
      size: inFeatures * 4,
      usage: storageUpload,
    });
    const weightBuffer = device.createBuffer({
      size: 256 + packedWeights.byteLength,
      usage: storageUpload,
    });
    const scaleBuffer = device.createBuffer({
      size: 256 + rowScales.byteLength,
      usage: storageUpload,
    });
    device.queue.writeBuffer(activationBuffer, 0, activation);
    device.queue.writeBuffer(weightBuffer, 256, packedWeights);
    device.queue.writeBuffer(scaleBuffer, 256, rowScales);
    const weights = {
      packedWeights: { buffer: weightBuffer, offset: 256, size: packedWeights.byteLength },
      rowScales: { buffer: scaleBuffer, offset: 256, size: rowScales.byteLength },
      inputScale: Math.fround(0.03125),
      outputScale: Math.fround(0.0078125),
    };

    async function run(
      input: GPUBuffer,
      rowCount: number,
    ): Promise<Uint32Array> {
      const pipelines = await getGemmaPrefillQatLinearPipelines(device, {
        rows: rowCount,
        inFeatures,
        outFeatures,
        bits: 4,
      });
      const resources = createGemmaPrefillQatLinearResources(
        device,
        pipelines,
        input,
        weights,
      );
      const outputBytes = rowCount * outFeatures * 4;
      const readback = device.createBuffer({
        size: outputBytes,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      try {
        const encoder = device.createCommandEncoder();
        encodeGemmaPrefillQatLinear(encoder, pipelines, resources);
        encoder.copyBufferToBuffer(resources.output, 0, readback, 0, outputBytes);
        device.queue.submit([encoder.finish()]);
        await readback.mapAsync(GPUMapMode.READ);
        const output = new Uint32Array(readback.getMappedRange().slice(0));
        readback.unmap();
        return output;
      } finally {
        readback.destroy();
        destroyGemmaPrefillQatLinearResources(resources);
      }
    }

    device.pushErrorScope("validation");
    device.pushErrorScope("internal");
    const batch = await run(activationBuffer, rows);
    let bitMismatches = 0;
    for (const row of [0, 7, 8, 31]) {
      device.queue.writeBuffer(
        singleActivationBuffer,
        0,
        activation.subarray(row * inFeatures, (row + 1) * inFeatures),
      );
      const single = await run(singleActivationBuffer, 1);
      for (let outputIndex = 0; outputIndex < outFeatures; outputIndex += 1) {
        if (batch[row * outFeatures + outputIndex] !== single[outputIndex]) {
          bitMismatches += 1;
        }
      }
    }
    const internalError = await device.popErrorScope();
    const validationError = await device.popErrorScope();
    activationBuffer.destroy();
    singleActivationBuffer.destroy();
    weightBuffer.destroy();
    scaleBuffer.destroy();
    return {
      bitMismatches,
      gpuError: internalError?.message ?? validationError?.message ?? null,
    };
  });

  expect(result).toEqual({ bitMismatches: 0, gpuError: null });
});

test("shares staged QKV SRQ without changing projection bits", async ({ page }) => {
  await page.goto("/");
  const webGpuAvailable = await page.evaluate(() => Boolean(navigator.gpu));
  test.skip(!webGpuAvailable, "Chrome does not expose WebGPU on this machine");

  const result = await page.evaluate(async () => {
    const linearModulePath = "/src/webgpu/prefill-qat-linear.ts";
    const deviceModulePath = "/src/webgpu/device.ts";
    const {
      createGemmaPrefillQatLinearResources,
      destroyGemmaPrefillQatLinearResources,
      encodeGemmaPrefillQatLinearPass,
      getGemmaPrefillQatLinearPipelines,
    } = await import(linearModulePath);
    const { getWebGpuDevice } = await import(deviceModulePath);
    const device = await getWebGpuDevice();
    const rows = 32;
    const inFeatures = 64;
    const outputFeatures = [128, 64, 64] as const;
    const inputScale = Math.fround(0.03125);
    const storageUpload = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    const activation = Float32Array.from(
      { length: rows * inFeatures },
      (_, index) => Math.fround(Math.sin(index * 0.17) * 3.25),
    );
    const activationBuffer = device.createBuffer({
      size: activation.byteLength,
      usage: storageUpload,
    });
    const sharedSrq = device.createBuffer({
      size: activation.byteLength,
      usage: GPUBufferUsage.STORAGE,
    });
    device.queue.writeBuffer(activationBuffer, 0, activation);
    const ownedBuffers: GPUBuffer[] = [activationBuffer, sharedSrq];
    const baselineResources: Awaited<ReturnType<
      typeof createGemmaPrefillQatLinearResources
    >>[] = [];
    const sharedResources: Awaited<ReturnType<
      typeof createGemmaPrefillQatLinearResources
    >>[] = [];
    const pipelines = [];

    for (let projectionIndex = 0; projectionIndex < outputFeatures.length;
      projectionIndex += 1) {
      const outFeatures = outputFeatures[projectionIndex];
      const packedWeights = Uint32Array.from(
        { length: outFeatures * inFeatures / 8 },
        (_, index) => Math.imul(index + 17 + projectionIndex * 101, 0x9e3779b1) >>> 0,
      );
      const rowScales = Float32Array.from(
        { length: outFeatures },
        (_, index) => Math.fround(0.0005 + ((index + projectionIndex) % 31) * 0.00003),
      );
      const weightBuffer = device.createBuffer({
        size: packedWeights.byteLength,
        usage: storageUpload,
      });
      const scaleBuffer = device.createBuffer({
        size: rowScales.byteLength,
        usage: storageUpload,
      });
      ownedBuffers.push(weightBuffer, scaleBuffer);
      device.queue.writeBuffer(weightBuffer, 0, packedWeights);
      device.queue.writeBuffer(scaleBuffer, 0, rowScales);
      const compiled = await getGemmaPrefillQatLinearPipelines(device, {
        rows,
        inFeatures,
        outFeatures,
        bits: 4,
      });
      pipelines.push(compiled);
      const weights = {
        packedWeights: weightBuffer,
        rowScales: scaleBuffer,
        inputScale,
        outputScale: Math.fround(0.0078125 + projectionIndex * 0.001),
      };
      baselineResources.push(createGemmaPrefillQatLinearResources(
        device,
        compiled,
        activationBuffer,
        weights,
      ));
      sharedResources.push(createGemmaPrefillQatLinearResources(
        device,
        compiled,
        projectionIndex === 0 ? activationBuffer : sharedSrq,
        { ...weights, inputScale: projectionIndex === 0 ? inputScale : 0 },
        undefined,
        projectionIndex === 0 ? sharedSrq : undefined,
      ));
    }

    device.pushErrorScope("validation");
    device.pushErrorScope("internal");
    const totalElements = rows * outputFeatures.reduce((total, value) => total + value, 0);
    const readback = device.createBuffer({
      size: totalElements * 8,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    try {
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      for (let index = 0; index < pipelines.length; index += 1) {
        encodeGemmaPrefillQatLinearPass(pass, pipelines[index], baselineResources[index]);
      }
      for (let index = 0; index < pipelines.length; index += 1) {
        encodeGemmaPrefillQatLinearPass(pass, pipelines[index], sharedResources[index]);
      }
      pass.end();
      let offset = 0;
      for (const resources of baselineResources) {
        const bytes = resources.output.size;
        encoder.copyBufferToBuffer(resources.output, 0, readback, offset, bytes);
        offset += bytes;
      }
      for (const resources of sharedResources) {
        const bytes = resources.output.size;
        encoder.copyBufferToBuffer(resources.output, 0, readback, offset, bytes);
        offset += bytes;
      }
      device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const outputs = new Uint32Array(readback.getMappedRange().slice(0));
      readback.unmap();
      let bitMismatches = 0;
      for (let index = 0; index < totalElements; index += 1) {
        if (outputs[index] !== outputs[totalElements + index]) bitMismatches += 1;
      }
      const internalError = await device.popErrorScope();
      const validationError = await device.popErrorScope();
      return {
        bitMismatches,
        gpuError: internalError?.message ?? validationError?.message ?? null,
      };
    } finally {
      readback.destroy();
      for (const resources of [...baselineResources, ...sharedResources]) {
        destroyGemmaPrefillQatLinearResources(resources);
      }
      for (const buffer of ownedBuffers) buffer.destroy();
    }
  });

  expect(result).toEqual({ bitMismatches: 0, gpuError: null });
});

test("fuses 32-row gate and up projections without changing output bits", async ({ page }) => {
  await page.goto("/");
  const webGpuAvailable = await page.evaluate(() => Boolean(navigator.gpu));
  test.skip(!webGpuAvailable, "Chrome does not expose WebGPU on this machine");

  const result = await page.evaluate(async () => {
    const gateUpModulePath = "/src/webgpu/prefill-qat-gate-up.ts";
    const linearModulePath = "/src/webgpu/prefill-qat-linear.ts";
    const deviceModulePath = "/src/webgpu/device.ts";
    const {
      createGemmaPrefillQatGateUpResources,
      destroyGemmaPrefillQatGateUpResources,
      encodeGemmaPrefillQatGateUp,
      getGemmaPrefillQatGateUpPipelines,
    } = await import(gateUpModulePath);
    const {
      createGemmaPrefillQatLinearResources,
      destroyGemmaPrefillQatLinearResources,
      encodeGemmaPrefillQatLinear,
      getGemmaPrefillQatLinearPipelines,
    } = await import(linearModulePath);
    const { getWebGpuDevice } = await import(deviceModulePath);
    const device = await getWebGpuDevice();
    const rows = 32;
    const inFeatures = 64;
    const outFeatures = 1024;
    const storageUpload = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    let bitMismatches = 0;

    device.pushErrorScope("validation");
    device.pushErrorScope("internal");
    for (const bits of [4, 2]) {
      const packedLength = outFeatures * inFeatures * bits / 32;
      const gatePacked = Uint32Array.from(
        { length: packedLength },
        (_, index) => Math.imul(index + 17, 0x9e3779b1) >>> 0,
      );
      const upPacked = Uint32Array.from(
        { length: packedLength },
        (_, index) => Math.imul(index + 43, 0x85ebca6b) >>> 0,
      );
      const gateScales = Float32Array.from(
        { length: outFeatures },
        (_, index) => Math.fround(0.0005 + (index % 31) * 0.00003),
      );
      const upScales = Float32Array.from(
        { length: outFeatures },
        (_, index) => Math.fround(0.0007 + (index % 29) * 0.00002),
      );
      const activation = Float32Array.from(
        { length: rows * inFeatures },
        (_, index) => Math.fround(Math.sin(index * 0.17) * 3.25),
      );
      const activationBuffer = device.createBuffer({
        size: activation.byteLength,
        usage: storageUpload,
      });
      const gateWeightBuffer = device.createBuffer({
        size: 256 + gatePacked.byteLength,
        usage: storageUpload,
      });
      const upWeightBuffer = device.createBuffer({
        size: 256 + upPacked.byteLength,
        usage: storageUpload,
      });
      const gateScaleBuffer = device.createBuffer({
        size: 256 + gateScales.byteLength,
        usage: storageUpload,
      });
      const upScaleBuffer = device.createBuffer({
        size: 256 + upScales.byteLength,
        usage: storageUpload,
      });
      device.queue.writeBuffer(activationBuffer, 0, activation);
      device.queue.writeBuffer(gateWeightBuffer, 256, gatePacked);
      device.queue.writeBuffer(upWeightBuffer, 256, upPacked);
      device.queue.writeBuffer(gateScaleBuffer, 256, gateScales);
      device.queue.writeBuffer(upScaleBuffer, 256, upScales);
      const inputScale = Math.fround(0.03125);
      const gateWeights = {
        packedWeights: {
          buffer: gateWeightBuffer,
          offset: 256,
          size: gatePacked.byteLength,
        },
        rowScales: {
          buffer: gateScaleBuffer,
          offset: 256,
          size: gateScales.byteLength,
        },
        inputScale,
        outputScale: Math.fround(0.0078125),
      };
      const upWeights = {
        packedWeights: {
          buffer: upWeightBuffer,
          offset: 256,
          size: upPacked.byteLength,
        },
        rowScales: {
          buffer: upScaleBuffer,
          offset: 256,
          size: upScales.byteLength,
        },
        inputScale,
        outputScale: Math.fround(0.015625),
      };
      const geometry = { rows, inFeatures, outFeatures, bits };
      const linearPipelines = await getGemmaPrefillQatLinearPipelines(device, geometry);
      const fusedPipelines = await getGemmaPrefillQatGateUpPipelines(device, geometry);
      const gate = createGemmaPrefillQatLinearResources(
        device,
        linearPipelines,
        activationBuffer,
        gateWeights,
      );
      const up = createGemmaPrefillQatLinearResources(
        device,
        linearPipelines,
        activationBuffer,
        upWeights,
      );
      const fused = createGemmaPrefillQatGateUpResources(
        device,
        fusedPipelines,
        activationBuffer,
        gateWeights,
        upWeights,
      );
      const outputBytes = rows * outFeatures * 4;
      const readback = device.createBuffer({
        size: outputBytes * 4,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      try {
        const encoder = device.createCommandEncoder();
        encodeGemmaPrefillQatLinear(encoder, linearPipelines, gate);
        encodeGemmaPrefillQatLinear(encoder, linearPipelines, up);
        encodeGemmaPrefillQatGateUp(encoder, fusedPipelines, fused);
        encoder.copyBufferToBuffer(gate.output, 0, readback, 0, outputBytes);
        encoder.copyBufferToBuffer(up.output, 0, readback, outputBytes, outputBytes);
        encoder.copyBufferToBuffer(fused.gateOutput, 0, readback, outputBytes * 2, outputBytes);
        encoder.copyBufferToBuffer(fused.upOutput, 0, readback, outputBytes * 3, outputBytes);
        device.queue.submit([encoder.finish()]);
        await readback.mapAsync(GPUMapMode.READ);
        const outputs = new Uint32Array(readback.getMappedRange().slice(0));
        readback.unmap();
        const outputElements = outputBytes / 4;
        for (let index = 0; index < outputElements; index += 1) {
          if (outputs[index] !== outputs[outputElements * 2 + index]) bitMismatches += 1;
          if (outputs[outputElements + index] !== outputs[outputElements * 3 + index]) {
            bitMismatches += 1;
          }
        }
      } finally {
        readback.destroy();
        destroyGemmaPrefillQatLinearResources(gate);
        destroyGemmaPrefillQatLinearResources(up);
        destroyGemmaPrefillQatGateUpResources(fused);
        activationBuffer.destroy();
        gateWeightBuffer.destroy();
        upWeightBuffer.destroy();
        gateScaleBuffer.destroy();
        upScaleBuffer.destroy();
      }
    }
    const internalError = await device.popErrorScope();
    const validationError = await device.popErrorScope();
    return {
      bitMismatches,
      gpuError: internalError?.message ?? validationError?.message ?? null,
    };
  });

  expect(result).toEqual({ bitMismatches: 0, gpuError: null });
});

test("fuses gate/up activation and down SRQ without changing down output bits", async ({ page }) => {
  await page.goto("/");
  const webGpuAvailable = await page.evaluate(() => Boolean(navigator.gpu));
  test.skip(!webGpuAvailable, "Chrome does not expose WebGPU on this machine");

  const result = await page.evaluate(async () => {
    const gateUpModulePath = "/src/webgpu/prefill-qat-gate-up.ts";
    const linearModulePath = "/src/webgpu/prefill-qat-linear.ts";
    const elementwiseModulePath = "/src/webgpu/prefill-elementwise.ts";
    const deviceModulePath = "/src/webgpu/device.ts";
    const {
      createGemmaPrefillQatGateUpActivationResources,
      createGemmaPrefillQatGateUpResources,
      destroyGemmaPrefillQatGateUpActivationResources,
      destroyGemmaPrefillQatGateUpResources,
      encodeGemmaPrefillQatGateUp,
      encodeGemmaPrefillQatGateUpActivation,
      getGemmaPrefillQatGateUpActivationPipelines,
      getGemmaPrefillQatGateUpPipelines,
    } = await import(gateUpModulePath);
    const {
      createGemmaPrefillQatLinearResources,
      destroyGemmaPrefillQatLinearResources,
      encodeGemmaPrefillQatLinear,
      getGemmaPrefillQatLinearPipelines,
    } = await import(linearModulePath);
    const {
      createGemmaPrefillGeluMultiplyResources,
      destroyGemmaPrefillElementwiseResources,
      encodeGemmaPrefillElementwise,
      getGemmaPrefillElementwisePipelines,
    } = await import(elementwiseModulePath);
    const { getWebGpuDevice } = await import(deviceModulePath);
    const device = await getWebGpuDevice();
    const rows = 32;
    const hiddenFeatures = 64;
    const intermediateFeatures = 1024;
    const storageUpload = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    let activationBitMismatches = 0;
    let downBitMismatches = 0;

    device.pushErrorScope("validation");
    device.pushErrorScope("internal");
    for (const bits of [4, 2]) {
      const upload = (data: ArrayBufferView) => {
        const buffer = device.createBuffer({ size: data.byteLength, usage: storageUpload });
        device.queue.writeBuffer(buffer, 0, data);
        return buffer;
      };
      const activation = Float32Array.from(
        { length: rows * hiddenFeatures },
        (_, index) => Math.fround(Math.sin(index * 0.17) * 3.25),
      );
      const gatePacked = Uint32Array.from(
        { length: intermediateFeatures * hiddenFeatures * bits / 32 },
        (_, index) => Math.imul(index + 17, 0x9e3779b1) >>> 0,
      );
      const upPacked = Uint32Array.from(
        { length: gatePacked.length },
        (_, index) => Math.imul(index + 43, 0x85ebca6b) >>> 0,
      );
      const gateScales = Float32Array.from(
        { length: intermediateFeatures },
        (_, index) => Math.fround(0.0005 + (index % 31) * 0.00003),
      );
      const upScales = Float32Array.from(
        { length: intermediateFeatures },
        (_, index) => Math.fround(0.0007 + (index % 29) * 0.00002),
      );
      const downPacked = Uint32Array.from(
        { length: hiddenFeatures * intermediateFeatures * bits / 32 },
        (_, index) => Math.imul(index + 71, 0xc2b2ae35) >>> 0,
      );
      const downScales = Float32Array.from(
        { length: hiddenFeatures },
        (_, index) => Math.fround(0.0009 + (index % 23) * 0.00004),
      );
      const gateOutputScale = Math.fround(0.0078125);
      const downInputScale = Math.fround(0.01953125);
      const geluLookup = Float32Array.from(
        { length: 256 },
        (_, index) => Math.fround(Math.tanh((index - 128) * gateOutputScale)),
      );
      const activationBuffer = upload(activation);
      const gateWeightBuffer = upload(gatePacked);
      const upWeightBuffer = upload(upPacked);
      const gateScaleBuffer = upload(gateScales);
      const upScaleBuffer = upload(upScales);
      const downWeightBuffer = upload(downPacked);
      const downScaleBuffer = upload(downScales);
      const lookupBuffer = upload(geluLookup);
      const inputScale = Math.fround(0.03125);
      const gateWeights = {
        packedWeights: gateWeightBuffer,
        rowScales: gateScaleBuffer,
        inputScale,
        outputScale: gateOutputScale,
      };
      const upWeights = {
        packedWeights: upWeightBuffer,
        rowScales: upScaleBuffer,
        inputScale,
        outputScale: Math.fround(0.015625),
      };
      const downWeights = {
        packedWeights: downWeightBuffer,
        rowScales: downScaleBuffer,
        inputScale: downInputScale,
        outputScale: Math.fround(0.0234375),
      };
      const gateGeometry = {
        rows,
        inFeatures: hiddenFeatures,
        outFeatures: intermediateFeatures,
        bits,
      };
      const downGeometry = {
        rows,
        inFeatures: intermediateFeatures,
        outFeatures: hiddenFeatures,
        bits,
      };
      const [gatePipelines, activationPipelines, downPipelines, elementwisePipelines] =
        await Promise.all([
          getGemmaPrefillQatGateUpPipelines(device, gateGeometry),
          getGemmaPrefillQatGateUpActivationPipelines(device, gateGeometry),
          getGemmaPrefillQatLinearPipelines(device, downGeometry),
          getGemmaPrefillElementwisePipelines(device),
        ]);
      const gate = createGemmaPrefillQatGateUpResources(
        device,
        gatePipelines,
        activationBuffer,
        gateWeights,
        upWeights,
      );
      const oldActivated = device.createBuffer({
        size: rows * intermediateFeatures * 4,
        usage: GPUBufferUsage.STORAGE,
      });
      const oldDownInput = device.createBuffer({
        size: rows * intermediateFeatures * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });
      const newDownInput = device.createBuffer({
        size: rows * intermediateFeatures * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });
      const gelu = createGemmaPrefillGeluMultiplyResources(
        device,
        elementwisePipelines.geluMultiply,
        gate.gateOutput,
        gate.upOutput,
        lookupBuffer,
        oldActivated,
        rows * intermediateFeatures,
        gateOutputScale,
      );
      const oldDown = createGemmaPrefillQatLinearResources(
        device,
        downPipelines,
        oldActivated,
        downWeights,
        undefined,
        oldDownInput,
      );
      const fused = createGemmaPrefillQatGateUpActivationResources(
        device,
        activationPipelines,
        activationBuffer,
        gateWeights,
        upWeights,
        lookupBuffer,
        downInputScale,
        newDownInput,
      );
      const newDown = createGemmaPrefillQatLinearResources(
        device,
        downPipelines,
        newDownInput,
        { ...downWeights, inputScale: 0 },
      );
      const activationBytes = rows * intermediateFeatures * 4;
      const downBytes = rows * hiddenFeatures * 4;
      const readback = device.createBuffer({
        size: activationBytes * 2 + downBytes * 2,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      try {
        const encoder = device.createCommandEncoder();
        encodeGemmaPrefillQatGateUp(encoder, gatePipelines, gate);
        encodeGemmaPrefillElementwise(encoder, elementwisePipelines.geluMultiply, gelu);
        encodeGemmaPrefillQatLinear(encoder, downPipelines, oldDown);
        encodeGemmaPrefillQatGateUpActivation(encoder, activationPipelines, fused);
        encodeGemmaPrefillQatLinear(encoder, downPipelines, newDown);
        encoder.copyBufferToBuffer(oldDownInput, 0, readback, 0, activationBytes);
        encoder.copyBufferToBuffer(newDownInput, 0, readback, activationBytes, activationBytes);
        encoder.copyBufferToBuffer(
          oldDown.output,
          0,
          readback,
          activationBytes * 2,
          downBytes,
        );
        encoder.copyBufferToBuffer(
          newDown.output,
          0,
          readback,
          activationBytes * 2 + downBytes,
          downBytes,
        );
        device.queue.submit([encoder.finish()]);
        await readback.mapAsync(GPUMapMode.READ);
        const outputs = new Uint32Array(readback.getMappedRange().slice(0));
        readback.unmap();
        const activationElements = activationBytes / 4;
        const downElements = downBytes / 4;
        for (let index = 0; index < activationElements; index += 1) {
          if (outputs[index] !== outputs[activationElements + index]) {
            activationBitMismatches += 1;
          }
        }
        for (let index = 0; index < downElements; index += 1) {
          if (outputs[activationElements * 2 + index] !==
              outputs[activationElements * 2 + downElements + index]) {
            downBitMismatches += 1;
          }
        }
      } finally {
        readback.destroy();
        destroyGemmaPrefillQatGateUpResources(gate);
        destroyGemmaPrefillElementwiseResources(gelu);
        destroyGemmaPrefillQatLinearResources(oldDown);
        destroyGemmaPrefillQatGateUpActivationResources(fused);
        destroyGemmaPrefillQatLinearResources(newDown);
        for (const buffer of [
          activationBuffer,
          gateWeightBuffer,
          upWeightBuffer,
          gateScaleBuffer,
          upScaleBuffer,
          downWeightBuffer,
          downScaleBuffer,
          lookupBuffer,
          oldActivated,
          oldDownInput,
          newDownInput,
        ]) buffer.destroy();
      }
    }
    const internalError = await device.popErrorScope();
    const validationError = await device.popErrorScope();
    return {
      activationBitMismatches,
      downBitMismatches,
      gpuError: internalError?.message ?? validationError?.message ?? null,
    };
  });

  expect(result).toEqual({
    activationBitMismatches: 0,
    downBitMismatches: 0,
    gpuError: null,
  });
});