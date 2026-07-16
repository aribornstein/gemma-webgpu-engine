let devicePromise: Promise<GPUDevice> | null = null;

export function getWebGpuDevice(): Promise<GPUDevice> {
  devicePromise ??= createDevice();
  return devicePromise;
}

async function createDevice(): Promise<GPUDevice> {
  if (!navigator.gpu) throw new Error("WebGPU is unavailable in this browser");
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("No compatible WebGPU adapter was found");
  const optionalFeatures: GPUFeatureName[] = ["timestamp-query", "subgroups", "shader-f16"];
  const requiredFeatures = optionalFeatures.filter((feature) => adapter.features.has(feature));
  const requiredLimits = adapter.limits.maxStorageBuffersPerShaderStage >= 9
    ? {
        maxStorageBuffersPerShaderStage: adapter.limits.maxStorageBuffersPerShaderStage,
        maxBufferSize: adapter.limits.maxBufferSize,
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      }
    : undefined;
  const device = await adapter.requestDevice({ requiredFeatures, requiredLimits });
  device.lost.then((info) => {
    console.error("WebGPU device lost", info);
    devicePromise = null;
  });
  return device;
}
