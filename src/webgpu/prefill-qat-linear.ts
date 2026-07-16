const WORKGROUP_SIZE = 32;
const SRQ_WORKGROUP_SIZE = 256;

export interface GemmaPrefillQatLinearGeometry {
  rows: number;
  inFeatures: number;
  outFeatures: number;
  bits: 2 | 4;
}

export interface GemmaPrefillQatLinearPipelines extends GemmaPrefillQatLinearGeometry {
  rowsPerTile: number;
  outputRowsPerWorkgroup: number;
  srq: GPUComputePipeline;
  projection: GPUComputePipeline;
}

export interface GemmaPrefillQatLinearResources {
  srqBindGroup: GPUBindGroup | null;
  projectionBindGroup: GPUBindGroup;
  output: GPUBuffer;
  runsSrq: boolean;
  workgroupCount: number;
  rowTileCount: number;
  ownedBuffers: GPUBuffer[];
}

export interface GemmaPrefillQatLinearWeights {
  packedWeights: GemmaPrefillBufferSlice;
  rowScales: GemmaPrefillBufferSlice;
  inputScale: number;
  outputScale: number;
}

export type GemmaPrefillBufferSlice = GPUBuffer | {
  buffer: GPUBuffer;
  offset: number;
  size: number;
};

const pipelineCache = new WeakMap<
  GPUDevice,
  Map<string, Promise<GemmaPrefillQatLinearPipelines>>
>();

export function getGemmaPrefillQatLinearPipelines(
  device: GPUDevice,
  geometry: GemmaPrefillQatLinearGeometry,
): Promise<GemmaPrefillQatLinearPipelines> {
  validateGeometry(geometry);
  let devicePipelines = pipelineCache.get(device);
  if (!devicePipelines) {
    devicePipelines = new Map();
    pipelineCache.set(device, devicePipelines);
  }
  const key = `${geometry.rows}:${geometry.inFeatures}:${geometry.outFeatures}:${geometry.bits}`;
  const cached = devicePipelines.get(key);
  if (cached) return cached;
  const compiled = compileGemmaPrefillQatLinearPipelines(device, geometry).catch((error) => {
    devicePipelines?.delete(key);
    throw error;
  });
  devicePipelines.set(key, compiled);
  return compiled;
}

export async function compileGemmaPrefillQatLinearPipelines(
  device: GPUDevice,
  geometry: GemmaPrefillQatLinearGeometry,
): Promise<GemmaPrefillQatLinearPipelines> {
  validateGeometry(geometry);
  const rowsPerTile = geometry.rows >= 8 ? 8 : geometry.rows > 2 ? 2 : geometry.rows;
  const outputRowsPerWorkgroup = geometry.outFeatures >= 32768
    ? 8
    : geometry.outFeatures >= 1024 ? 2 : 1;
  const [srq, projection] = await Promise.all([
    device.createComputePipelineAsync({
      label: "Gemma prefill staged SRQ",
      layout: "auto",
      compute: {
        module: device.createShaderModule({ code: createGemmaPrefillSrqShader() }),
        entryPoint: "main",
      },
    }),
    device.createComputePipelineAsync({
      label: "Gemma prefill exact QAT linear",
      layout: "auto",
      compute: {
        module: device.createShaderModule({
          code: createGemmaPrefillQatLinearShader(
            geometry,
            rowsPerTile,
            outputRowsPerWorkgroup,
          ),
        }),
        entryPoint: "main",
      },
    }),
  ]);
  return { ...geometry, rowsPerTile, outputRowsPerWorkgroup, srq, projection };
}

export function createGemmaPrefillQatLinearResources(
  device: GPUDevice,
  pipelines: GemmaPrefillQatLinearPipelines,
  activation: GPUBuffer,
  weights: GemmaPrefillQatLinearWeights,
  output?: GPUBuffer,
  srqOutput?: GPUBuffer,
): GemmaPrefillQatLinearResources {
  const inputBytes = pipelines.rows * pipelines.inFeatures * 4;
  const weightBytes = pipelines.outFeatures * pipelines.inFeatures * pipelines.bits / 8;
  const outputBytes = pipelines.rows * pipelines.outFeatures * 4;
    if (activation.size < inputBytes || sliceSize(weights.packedWeights) < weightBytes ||
      sliceSize(weights.rowScales) < pipelines.outFeatures * 4 ||
      (output && output.size < outputBytes) ||
      (srqOutput && srqOutput.size < inputBytes)) {
    throw new Error("Gemma prefill QAT buffers do not match pipeline geometry");
  }
  if (!Number.isFinite(weights.inputScale) || !Number.isFinite(weights.outputScale)) {
    throw new Error("Gemma prefill QAT scales must be finite");
  }

  const ownedBuffers: GPUBuffer[] = [];
  const make = (label: string, size: number, usage: GPUBufferUsageFlags) => {
    const buffer = device.createBuffer({ label, size, usage });
    ownedBuffers.push(buffer);
    return buffer;
  };
  const runsSrq = weights.inputScale !== 0;
  let projectionInput = activation;
  let srqBindGroup: GPUBindGroup | null = null;
  if (runsSrq) {
    projectionInput = srqOutput ?? make(
      "Gemma prefill SRQ activation",
      inputBytes,
      GPUBufferUsage.STORAGE,
    );
    const srqParams = make(
      "Gemma prefill SRQ parameters",
      16,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    );
    device.queue.writeBuffer(
      srqParams,
      0,
      new Float32Array([
        weights.inputScale,
        Math.fround(1 / weights.inputScale),
        0,
        0,
      ]),
    );
    srqBindGroup = device.createBindGroup({
      layout: pipelines.srq.getBindGroupLayout(0),
      entries: [
        binding(0, activation),
        binding(1, projectionInput),
        binding(2, srqParams),
      ],
    });
  }

  const outputBuffer = output ?? make(
    "Gemma prefill QAT output",
    outputBytes,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  );
  const projectionParams = make(
    "Gemma prefill QAT parameters",
    16,
    GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  );
  device.queue.writeBuffer(
    projectionParams,
    0,
    new Float32Array([0, weights.outputScale, 0, 0]),
  );
  const projectionBindGroup = device.createBindGroup({
    layout: pipelines.projection.getBindGroupLayout(0),
    entries: [
      binding(0, projectionInput),
      sliceBinding(1, weights.packedWeights),
      sliceBinding(2, weights.rowScales),
      binding(3, outputBuffer),
      binding(4, projectionParams),
    ],
  });
  return {
    srqBindGroup,
    projectionBindGroup,
    output: outputBuffer,
    runsSrq,
    workgroupCount: Math.ceil(pipelines.outFeatures / pipelines.outputRowsPerWorkgroup),
    rowTileCount: Math.ceil(pipelines.rows / pipelines.rowsPerTile),
    ownedBuffers,
  };
}

export function encodeGemmaPrefillQatLinear(
  encoder: GPUCommandEncoder,
  pipelines: GemmaPrefillQatLinearPipelines,
  resources: GemmaPrefillQatLinearResources,
): void {
  if (resources.runsSrq) {
    if (!resources.srqBindGroup) throw new Error("Gemma prefill SRQ bind group is missing");
    const srq = encoder.beginComputePass({ label: "Gemma prefill staged SRQ" });
    srq.setPipeline(pipelines.srq);
    srq.setBindGroup(0, resources.srqBindGroup);
    srq.dispatchWorkgroups(
      Math.ceil(pipelines.rows * pipelines.inFeatures / SRQ_WORKGROUP_SIZE),
    );
    srq.end();
  }
  const projection = encoder.beginComputePass({ label: "Gemma prefill exact QAT linear" });
  projection.setPipeline(pipelines.projection);
  projection.setBindGroup(0, resources.projectionBindGroup);
  projection.dispatchWorkgroups(resources.workgroupCount, 1, resources.rowTileCount);
  projection.end();
}

export function destroyGemmaPrefillQatLinearResources(
  resources: GemmaPrefillQatLinearResources,
): void {
  for (const buffer of resources.ownedBuffers) buffer.destroy();
}

export function createGemmaPrefillSrqShader(): string {
  return `struct Params {
  scale: f32,
  inverseScale: f32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

fn divExact(value: f32, scale: f32, inverseScale: f32) -> f32 {
  let quotient = value * inverseScale;
  let remainder = fma(-scale, quotient, value);
  return fma(remainder, inverseScale, quotient);
}

@compute @workgroup_size(${SRQ_WORKGROUP_SIZE}, 1, 1)
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
  let index = invocation.x;
  if (index >= arrayLength(&input)) { return; }
  let scale = params.scale;
  let value = input[index];
  output[index] = select(
    value,
    clamp(round(divExact(value, scale, params.inverseScale)), -128.0, 127.0) * scale,
    scale != 0.0,
  );
}`;
}

export function createGemmaPrefillQatLinearShader(
  geometry: GemmaPrefillQatLinearGeometry,
  rowsPerTile = geometry.rows >= 8 ? 8 : geometry.rows > 2 ? 2 : geometry.rows,
  outputRowsPerWorkgroup = geometry.outFeatures >= 32768
    ? 8
    : geometry.outFeatures >= 1024 ? 2 : 1,
): string {
  validateGeometry(geometry);
  if (![1, 2, 8].includes(rowsPerTile) || ![1, 2, 8].includes(outputRowsPerWorkgroup)) {
    throw new Error("Gemma prefill QAT tile geometry is unsupported");
  }
  const valuesPerWord = 32 / geometry.bits;
  const chunks = 8 / geometry.bits;
  const wordsPerRow = geometry.inFeatures / valuesPerWord;
  const declarations: string[] = [];
  const unpack: string[] = [];
  const accumulation: string[] = [];
  const epilogue: string[] = [];

  for (let token = 0; token < rowsPerTile; token += 1) {
    declarations.push(`  let tokenOk${token} = tokenStart + ${token}u < ROWS;`);
    declarations.push(`  var activationSum${token}: f32 = 0.0;`);
    for (let outputRow = 0; outputRow < outputRowsPerWorkgroup; outputRow += 1) {
      declarations.push(`  var dotSum${token}_${outputRow}: f32 = 0.0;`);
    }
  }
  for (let outputRow = 0; outputRow < outputRowsPerWorkgroup; outputRow += 1) {
    unpack.push(`    var packed${outputRow}: u32 = 0u;`);
    unpack.push(
      `    if (outputBase + ${outputRow}u < OUT_FEATURES) { ` +
      `packed${outputRow} = packedWeights[(outputBase + ${outputRow}u) * WORDS_PER_ROW + word]; }`,
    );
    if (geometry.bits === 4) {
      unpack.push(`    let low${outputRow} = vec4<f32>(unpack4xU8(packed${outputRow} & 0x0f0f0f0fu));`);
      unpack.push(`    let high${outputRow} = vec4<f32>(unpack4xU8((packed${outputRow} >> 4u) & 0x0f0f0f0fu));`);
      unpack.push(
        `    let codes${outputRow}_0 = vec4<f32>(low${outputRow}.x, high${outputRow}.x, ` +
        `low${outputRow}.y, high${outputRow}.y);`,
      );
      unpack.push(
        `    let codes${outputRow}_1 = vec4<f32>(low${outputRow}.z, high${outputRow}.z, ` +
        `low${outputRow}.w, high${outputRow}.w);`,
      );
    } else {
      for (let chunk = 0; chunk < 4; chunk += 1) {
        unpack.push(
          `    let unpacked${outputRow}_${chunk} = vec4<f32>(unpack4xU8(` +
          `(packed${outputRow} >> ${chunk * 2}u) & 0x03030303u));`,
        );
      }
      for (let chunk = 0; chunk < 4; chunk += 1) {
        const component = ["x", "y", "z", "w"][chunk];
        unpack.push(
          `    let codes${outputRow}_${chunk} = vec4<f32>(` +
          [0, 1, 2, 3]
            .map((part) => `unpacked${outputRow}_${part}.${component}`)
            .join(", ") +
          `);`,
        );
      }
    }
  }
  for (let token = 0; token < rowsPerTile; token += 1) {
    accumulation.push(`    if (tokenOk${token}) {`);
    accumulation.push(
      `      let activationBase${token} = (tokenStart + ${token}u) * ` +
      `(IN_FEATURES / 4u) + word * CHUNKS;`,
    );
    for (let chunk = 0; chunk < chunks; chunk += 1) {
      accumulation.push(
        `      let activation${token}_${chunk} = vec4<f32>(` +
        `activation[activationBase${token} + ${chunk}u]);`,
      );
      accumulation.push(
        `      activationSum${token} = activationSum${token} + ` +
        `activation${token}_${chunk}.x + activation${token}_${chunk}.y + ` +
        `activation${token}_${chunk}.z + activation${token}_${chunk}.w;`,
      );
      for (let outputRow = 0; outputRow < outputRowsPerWorkgroup; outputRow += 1) {
        accumulation.push(
          `      dotSum${token}_${outputRow} = dotSum${token}_${outputRow} + ` +
          `dot(codes${outputRow}_${chunk}, activation${token}_${chunk});`,
        );
      }
    }
    accumulation.push("    }");
  }
  for (let token = 0; token < rowsPerTile; token += 1) {
    epilogue.push(`  if (tokenOk${token}) {`);
    epilogue.push(`    let reducedActivation${token} = subgroupAdd(activationSum${token});`);
    for (let outputRow = 0; outputRow < outputRowsPerWorkgroup; outputRow += 1) {
      epilogue.push("    {");
      epilogue.push(
        `      let reducedDot = subgroupAdd(dotSum${token}_${outputRow});`,
      );
      epilogue.push(`      let outputRow = outputBase + ${outputRow}u;`);
      epilogue.push("      if (lane == 0u && outputRow < OUT_FEATURES) {");
      epilogue.push(
        `        output[(tokenStart + ${token}u) * OUT_FEATURES + outputRow] = srq(` +
        `rowScales[outputRow] * (reducedDot - ZERO_POINT * reducedActivation${token}), ` +
        `params.outputScale);`,
      );
      epilogue.push("      }");
      epilogue.push("    }");
    }
    epilogue.push("  }");
  }

  return `enable subgroups;

struct Params {
  inputScale: f32,
  outputScale: f32,
}

@group(0) @binding(0) var<storage, read> activation: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> packedWeights: array<u32>;
@group(0) @binding(2) var<storage, read> rowScales: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

const ROWS: u32 = ${geometry.rows}u;
const ROWS_PER_TILE: u32 = ${rowsPerTile}u;
const IN_FEATURES: u32 = ${geometry.inFeatures}u;
const OUT_FEATURES: u32 = ${geometry.outFeatures}u;
const CHUNKS: u32 = ${chunks}u;
const WORDS_PER_ROW: u32 = ${wordsPerRow}u;
const ZERO_POINT: f32 = ${geometry.bits === 4 ? 8 : 2}.0;
const OUTPUT_ROWS_PER_WORKGROUP: u32 = ${outputRowsPerWorkgroup}u;

fn srq(value: f32, scale: f32) -> f32 {
  if (scale == 0.0) { return value; }
  return clamp(round(value / scale), -128.0, 127.0) * scale;
}

@compute @workgroup_size(${WORKGROUP_SIZE}, 1, 1)
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) lane: u32,
) {
  let outputBase = workgroupId.x * OUTPUT_ROWS_PER_WORKGROUP;
  if (outputBase >= OUT_FEATURES) { return; }
  let tokenStart = workgroupId.z * ROWS_PER_TILE;
${declarations.join("\n")}

  var word = lane;
  loop {
    if (word >= WORDS_PER_ROW) { break; }
${unpack.join("\n")}
${accumulation.join("\n")}
    word = word + ${WORKGROUP_SIZE}u;
  }

${epilogue.join("\n")}
}`;
}

function binding(bindingIndex: number, buffer: GPUBuffer): GPUBindGroupEntry {
  return { binding: bindingIndex, resource: { buffer } };
}

function sliceBinding(
  bindingIndex: number,
  slice: GemmaPrefillBufferSlice,
): GPUBindGroupEntry {
  return slice instanceof GPUBuffer
    ? binding(bindingIndex, slice)
    : {
        binding: bindingIndex,
        resource: { buffer: slice.buffer, offset: slice.offset, size: slice.size },
      };
}

function sliceSize(slice: GemmaPrefillBufferSlice): number {
  if (slice instanceof GPUBuffer) return slice.size;
  if (!Number.isInteger(slice.offset) || slice.offset < 0 ||
      !Number.isInteger(slice.size) || slice.size < 1 ||
      slice.offset % 256 !== 0 || slice.offset + slice.size > slice.buffer.size) {
    throw new Error("Gemma prefill buffer slice is invalid or not 256-byte aligned");
  }
  return slice.size;
}

function validateGeometry(geometry: GemmaPrefillQatLinearGeometry): void {
  if (!Number.isInteger(geometry.rows) || geometry.rows < 1 || geometry.rows > 32 ||
      !Number.isInteger(geometry.inFeatures) || geometry.inFeatures < 4 ||
      !Number.isInteger(geometry.outFeatures) || geometry.outFeatures < 1 ||
      ![2, 4].includes(geometry.bits) ||
      (geometry.inFeatures * geometry.bits) % 32 !== 0 ||
      geometry.inFeatures % 4 !== 0) {
    throw new Error("Gemma prefill QAT geometry is unsupported");
  }
}