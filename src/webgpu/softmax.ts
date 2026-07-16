import { getWebGpuDevice } from "./device";

const shader = `
struct Data { values: array<f32> }
@group(0) @binding(0) var<storage, read> input: Data;
@group(0) @binding(1) var<storage, read_write> output: Data;
var<workgroup> scratch: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) lane: u32) {
  scratch[lane] = input.values[lane];
  workgroupBarrier();
  var stride = 128u;
  loop {
    if (lane < stride) { scratch[lane] = max(scratch[lane], scratch[lane + stride]); }
    workgroupBarrier();
    if (stride == 1u) { break; }
    stride /= 2u;
  }
  let value = exp(input.values[lane] - scratch[0]);
  scratch[lane] = value;
  workgroupBarrier();
  stride = 128u;
  loop {
    if (lane < stride) { scratch[lane] += scratch[lane + stride]; }
    workgroupBarrier();
    if (stride == 1u) { break; }
    stride /= 2u;
  }
  output.values[lane] = value / scratch[0];
}`;

export async function benchmarkSoftmax(size = 256, iterations = 50): Promise<Record<string, number>> {
  if (size !== 256) throw new Error("Kernel 001 currently supports exactly 256 values");
  const input = Float32Array.from({ length: size }, (_, index) => Math.sin(index * 0.17) * 6);
  const expected = cpuSoftmax(input);
  const device = await getWebGpuDevice();
  const bytes = input.byteLength;
  const inputBuffer = device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const outputBuffer = device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const readBuffer = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  device.queue.writeBuffer(inputBuffer, 0, input);
  const module = device.createShaderModule({ code: shader });
  const pipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });
  const bindGroup = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: inputBuffer } },
    { binding: 1, resource: { buffer: outputBuffer } },
  ] });
  const started = performance.now();
  for (let index = 0; index < iterations; index += 1) {
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(1);
    pass.end();
    if (index === iterations - 1) encoder.copyBufferToBuffer(outputBuffer, 0, readBuffer, 0, bytes);
    device.queue.submit([encoder.finish()]);
  }
  await readBuffer.mapAsync(GPUMapMode.READ);
  const elapsed = performance.now() - started;
  const actual = new Float32Array(readBuffer.getMappedRange().slice(0));
  readBuffer.unmap();
  let maximumError = 0;
  for (let index = 0; index < size; index += 1) maximumError = Math.max(maximumError, Math.abs(actual[index] - expected[index]));
  inputBuffer.destroy();
  outputBuffer.destroy();
  readBuffer.destroy();
  return { elements: size, iterations, totalMs: round(elapsed), averageMs: round(elapsed / iterations), maximumAbsoluteError: maximumError };
}

function cpuSoftmax(input: Float32Array): Float32Array {
  const maximum = Math.max(...input);
  const result = Float32Array.from(input, (value) => Math.exp(value - maximum));
  const total = result.reduce((sum, value) => sum + value, 0);
  return result.map((value) => value / total);
}

function round(value: number): number {
  return Math.round(value * 1000) / 1000;
}
