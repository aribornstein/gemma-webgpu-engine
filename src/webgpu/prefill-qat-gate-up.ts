import {
  createGemmaPrefillParameter,
  gemmaPrefillParameterBinding,
  writeGemmaPrefillParameter,
  type GemmaPrefillParameterArena,
} from "./prefill-parameter-arena";
import {
  getGemmaPrefillQatLinearPipelines,
  type GemmaPrefillBufferSlice,
  type GemmaPrefillQatLinearGeometry,
  type GemmaPrefillQatLinearWeights,
} from "./prefill-qat-linear";

const WORKGROUP_SIZE = 32;
const SRQ_WORKGROUP_SIZE = 256;

export interface GemmaPrefillQatGateUpPipelines extends GemmaPrefillQatLinearGeometry {
  rowsPerTile: number;
  outputRowsPerWorkgroup: number;
  srq: GPUComputePipeline;
  projection: GPUComputePipeline;
}

export interface GemmaPrefillQatGateUpResources {
  srqBindGroup: GPUBindGroup | null;
  projectionBindGroup: GPUBindGroup;
  gateOutput: GPUBuffer;
  upOutput: GPUBuffer;
  runsSrq: boolean;
  workgroupCount: number;
  rowTileCount: number;
  ownedBuffers: GPUBuffer[];
}

export interface GemmaPrefillQatGateUpActivationResources {
  srqBindGroup: GPUBindGroup | null;
  projectionBindGroup: GPUBindGroup;
  output: GPUBuffer;
  runsSrq: boolean;
  workgroupCount: number;
  rowTileCount: number;
  ownedBuffers: GPUBuffer[];
}

const pipelineCache = new WeakMap<
  GPUDevice,
  Map<string, Promise<GemmaPrefillQatGateUpPipelines>>
>();
const activationPipelineCache = new WeakMap<
  GPUDevice,
  Map<string, Promise<GemmaPrefillQatGateUpPipelines>>
>();

export function getGemmaPrefillQatGateUpPipelines(
  device: GPUDevice,
  geometry: GemmaPrefillQatLinearGeometry,
): Promise<GemmaPrefillQatGateUpPipelines> {
  validateGeometry(geometry);
  let devicePipelines = pipelineCache.get(device);
  if (!devicePipelines) {
    devicePipelines = new Map();
    pipelineCache.set(device, devicePipelines);
  }
  const key = `${geometry.rows}:${geometry.inFeatures}:${geometry.outFeatures}:${geometry.bits}`;
  const cached = devicePipelines.get(key);
  if (cached) return cached;
  const compiled = compileGemmaPrefillQatGateUpPipelines(device, geometry).catch((error) => {
    devicePipelines?.delete(key);
    throw error;
  });
  devicePipelines.set(key, compiled);
  return compiled;
}

export async function compileGemmaPrefillQatGateUpPipelines(
  device: GPUDevice,
  geometry: GemmaPrefillQatLinearGeometry,
): Promise<GemmaPrefillQatGateUpPipelines> {
  validateGeometry(geometry);
  const linear = await getGemmaPrefillQatLinearPipelines(device, geometry);
  const projection = await device.createComputePipelineAsync({
    label: "Gemma prefill exact fused gate/up QAT linear",
    layout: "auto",
    compute: {
      module: device.createShaderModule({
        code: createGemmaPrefillQatGateUpShader(
          geometry,
          linear.rowsPerTile,
          linear.outputRowsPerWorkgroup,
        ),
      }),
      entryPoint: "main",
    },
  });
  return {
    ...geometry,
    rowsPerTile: linear.rowsPerTile,
    outputRowsPerWorkgroup: linear.outputRowsPerWorkgroup,
    srq: linear.srq,
    projection,
  };
}

export function getGemmaPrefillQatGateUpActivationPipelines(
  device: GPUDevice,
  geometry: GemmaPrefillQatLinearGeometry,
): Promise<GemmaPrefillQatGateUpPipelines> {
  validateGeometry(geometry);
  let devicePipelines = activationPipelineCache.get(device);
  if (!devicePipelines) {
    devicePipelines = new Map();
    activationPipelineCache.set(device, devicePipelines);
  }
  const key = `${geometry.rows}:${geometry.inFeatures}:${geometry.outFeatures}:${geometry.bits}`;
  const cached = devicePipelines.get(key);
  if (cached) return cached;
  const compiled = compileGemmaPrefillQatGateUpActivationPipelines(
    device,
    geometry,
  ).catch((error) => {
    devicePipelines?.delete(key);
    throw error;
  });
  devicePipelines.set(key, compiled);
  return compiled;
}

export async function compileGemmaPrefillQatGateUpActivationPipelines(
  device: GPUDevice,
  geometry: GemmaPrefillQatLinearGeometry,
): Promise<GemmaPrefillQatGateUpPipelines> {
  validateGeometry(geometry);
  const linear = await getGemmaPrefillQatLinearPipelines(device, geometry);
  const projection = await device.createComputePipelineAsync({
    label: "Gemma prefill exact fused gate/up activation",
    layout: "auto",
    compute: {
      module: device.createShaderModule({
        code: createGemmaPrefillQatGateUpShader(
          geometry,
          linear.rowsPerTile,
          linear.outputRowsPerWorkgroup,
          true,
        ),
      }),
      entryPoint: "main",
    },
  });
  return {
    ...geometry,
    rowsPerTile: linear.rowsPerTile,
    outputRowsPerWorkgroup: linear.outputRowsPerWorkgroup,
    srq: linear.srq,
    projection,
  };
}

export function createGemmaPrefillQatGateUpResources(
  device: GPUDevice,
  pipelines: GemmaPrefillQatGateUpPipelines,
  activation: GPUBuffer,
  gateWeights: GemmaPrefillQatLinearWeights,
  upWeights: GemmaPrefillQatLinearWeights,
  gateOutput?: GPUBuffer,
  upOutput?: GPUBuffer,
  srqOutput?: GPUBuffer,
  parameterArena?: GemmaPrefillParameterArena,
): GemmaPrefillQatGateUpResources {
  const inputBytes = pipelines.rows * pipelines.inFeatures * 4;
  const weightBytes = pipelines.outFeatures * pipelines.inFeatures * pipelines.bits / 8;
  const outputBytes = pipelines.rows * pipelines.outFeatures * 4;
  if (activation.size < inputBytes ||
      sliceSize(gateWeights.packedWeights) < weightBytes ||
      sliceSize(upWeights.packedWeights) < weightBytes ||
      sliceSize(gateWeights.rowScales) < pipelines.outFeatures * 4 ||
      sliceSize(upWeights.rowScales) < pipelines.outFeatures * 4 ||
      (gateOutput && gateOutput.size < outputBytes) ||
      (upOutput && upOutput.size < outputBytes) ||
      (srqOutput && srqOutput.size < inputBytes)) {
    throw new Error("Gemma prefill fused gate/up buffers do not match pipeline geometry");
  }
  if (!Number.isFinite(gateWeights.inputScale) ||
      !Number.isFinite(gateWeights.outputScale) ||
      !Number.isFinite(upWeights.inputScale) ||
      !Number.isFinite(upWeights.outputScale) ||
      gateWeights.inputScale !== upWeights.inputScale) {
    throw new Error("Gemma prefill fused gate/up scales are invalid or incompatible");
  }

  const ownedBuffers: GPUBuffer[] = [];
  const make = (label: string, size: number, usage: GPUBufferUsageFlags) => {
    const buffer = device.createBuffer({ label, size, usage });
    ownedBuffers.push(buffer);
    return buffer;
  };
  const runsSrq = gateWeights.inputScale !== 0;
  let projectionInput = activation;
  let srqBindGroup: GPUBindGroup | null = null;
  if (runsSrq) {
    projectionInput = srqOutput ?? make(
      "Gemma prefill fused gate/up SRQ activation",
      inputBytes,
      GPUBufferUsage.STORAGE,
    );
    const srqParams = createGemmaPrefillParameter(
      device,
      16,
      "Gemma prefill fused gate/up SRQ parameters",
      parameterArena,
    );
    ownedBuffers.push(...srqParams.ownedBuffers);
    writeGemmaPrefillParameter(
      device,
      srqParams.slice,
      new Float32Array([
        gateWeights.inputScale,
        Math.fround(1 / gateWeights.inputScale),
        0,
        0,
      ]),
    );
    srqBindGroup = device.createBindGroup({
      layout: pipelines.srq.getBindGroupLayout(0),
      entries: [
        binding(0, activation),
        binding(1, projectionInput),
        gemmaPrefillParameterBinding(2, srqParams.slice),
      ],
    });
  }

  const gateOutputBuffer = gateOutput ?? make(
    "Gemma prefill fused gate output",
    outputBytes,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  );
  const upOutputBuffer = upOutput ?? make(
    "Gemma prefill fused up output",
    outputBytes,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  );
  const projectionParams = createGemmaPrefillParameter(
    device,
    16,
    "Gemma prefill fused gate/up parameters",
    parameterArena,
  );
  ownedBuffers.push(...projectionParams.ownedBuffers);
  writeGemmaPrefillParameter(
    device,
    projectionParams.slice,
    new Float32Array([gateWeights.outputScale, upWeights.outputScale, 0, 0]),
  );
  const projectionBindGroup = device.createBindGroup({
    layout: pipelines.projection.getBindGroupLayout(0),
    entries: [
      binding(0, projectionInput),
      sliceBinding(1, gateWeights.packedWeights),
      sliceBinding(2, gateWeights.rowScales),
      binding(3, gateOutputBuffer),
      sliceBinding(4, upWeights.packedWeights),
      sliceBinding(5, upWeights.rowScales),
      binding(6, upOutputBuffer),
      gemmaPrefillParameterBinding(7, projectionParams.slice),
    ],
  });
  return {
    srqBindGroup,
    projectionBindGroup,
    gateOutput: gateOutputBuffer,
    upOutput: upOutputBuffer,
    runsSrq,
    workgroupCount: Math.ceil(pipelines.outFeatures / pipelines.outputRowsPerWorkgroup),
    rowTileCount: Math.ceil(pipelines.rows / pipelines.rowsPerTile),
    ownedBuffers,
  };
}

export function encodeGemmaPrefillQatGateUp(
  encoder: GPUCommandEncoder,
  pipelines: GemmaPrefillQatGateUpPipelines,
  resources: GemmaPrefillQatGateUpResources,
): void {
  const pass = encoder.beginComputePass({ label: "Gemma prefill exact fused gate/up" });
  encodeGemmaPrefillQatGateUpPass(pass, pipelines, resources);
  pass.end();
}

export function encodeGemmaPrefillQatGateUpPass(
  pass: GPUComputePassEncoder,
  pipelines: GemmaPrefillQatGateUpPipelines,
  resources: GemmaPrefillQatGateUpResources,
): void {
  if (resources.runsSrq) {
    if (!resources.srqBindGroup) throw new Error("Gemma prefill fused gate/up SRQ bind group is missing");
    pass.setPipeline(pipelines.srq);
    pass.setBindGroup(0, resources.srqBindGroup);
    pass.dispatchWorkgroups(
      Math.ceil(pipelines.rows * pipelines.inFeatures / SRQ_WORKGROUP_SIZE),
    );
  }
  pass.setPipeline(pipelines.projection);
  pass.setBindGroup(0, resources.projectionBindGroup);
  pass.dispatchWorkgroups(resources.workgroupCount, 1, resources.rowTileCount);
}

export function destroyGemmaPrefillQatGateUpResources(
  resources: GemmaPrefillQatGateUpResources,
): void {
  for (const buffer of resources.ownedBuffers) buffer.destroy();
}

export function createGemmaPrefillQatGateUpActivationResources(
  device: GPUDevice,
  pipelines: GemmaPrefillQatGateUpPipelines,
  activation: GPUBuffer,
  gateWeights: GemmaPrefillQatLinearWeights,
  upWeights: GemmaPrefillQatLinearWeights,
  geluLookup: GPUBuffer,
  downInputScale: number,
  output?: GPUBuffer,
  srqOutput?: GPUBuffer,
  parameterArena?: GemmaPrefillParameterArena,
): GemmaPrefillQatGateUpActivationResources {
  const inputBytes = pipelines.rows * pipelines.inFeatures * 4;
  const weightBytes = pipelines.outFeatures * pipelines.inFeatures * pipelines.bits / 8;
  const outputBytes = pipelines.rows * pipelines.outFeatures * 4;
  if (activation.size < inputBytes ||
      sliceSize(gateWeights.packedWeights) < weightBytes ||
      sliceSize(upWeights.packedWeights) < weightBytes ||
      sliceSize(gateWeights.rowScales) < pipelines.outFeatures * 4 ||
      sliceSize(upWeights.rowScales) < pipelines.outFeatures * 4 ||
      geluLookup.size < 256 * 4 ||
      (output && output.size < outputBytes) ||
      (srqOutput && srqOutput.size < inputBytes)) {
    throw new Error("Gemma prefill fused gate/up activation buffers do not match geometry");
  }
  if (!Number.isFinite(gateWeights.inputScale) ||
      !Number.isFinite(gateWeights.outputScale) ||
      !Number.isFinite(upWeights.inputScale) ||
      !Number.isFinite(upWeights.outputScale) ||
      gateWeights.inputScale !== upWeights.inputScale ||
      !Number.isFinite(downInputScale) || downInputScale <= 0) {
    throw new Error("Gemma prefill fused gate/up activation scales are invalid");
  }

  const ownedBuffers: GPUBuffer[] = [];
  const make = (label: string, size: number, usage: GPUBufferUsageFlags) => {
    const buffer = device.createBuffer({ label, size, usage });
    ownedBuffers.push(buffer);
    return buffer;
  };
  const runsSrq = gateWeights.inputScale !== 0;
  let projectionInput = activation;
  let srqBindGroup: GPUBindGroup | null = null;
  if (runsSrq) {
    projectionInput = srqOutput ?? make(
      "Gemma prefill fused gate/up activation SRQ input",
      inputBytes,
      GPUBufferUsage.STORAGE,
    );
    const srqParams = createGemmaPrefillParameter(
      device,
      16,
      "Gemma prefill fused gate/up activation SRQ parameters",
      parameterArena,
    );
    ownedBuffers.push(...srqParams.ownedBuffers);
    writeGemmaPrefillParameter(
      device,
      srqParams.slice,
      new Float32Array([
        gateWeights.inputScale,
        Math.fround(1 / gateWeights.inputScale),
        0,
        0,
      ]),
    );
    srqBindGroup = device.createBindGroup({
      layout: pipelines.srq.getBindGroupLayout(0),
      entries: [
        binding(0, activation),
        binding(1, projectionInput),
        gemmaPrefillParameterBinding(2, srqParams.slice),
      ],
    });
  }

  const outputBuffer = output ?? make(
    "Gemma prefill fused gate/up activated output",
    outputBytes,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  );
  const projectionParams = createGemmaPrefillParameter(
    device,
    16,
    "Gemma prefill fused gate/up activation parameters",
    parameterArena,
  );
  ownedBuffers.push(...projectionParams.ownedBuffers);
  writeGemmaPrefillParameter(
    device,
    projectionParams.slice,
    new Float32Array([
      gateWeights.outputScale,
      upWeights.outputScale,
      downInputScale,
      Math.fround(1 / downInputScale),
    ]),
  );
  const projectionBindGroup = device.createBindGroup({
    layout: pipelines.projection.getBindGroupLayout(0),
    entries: [
      binding(0, projectionInput),
      sliceBinding(1, gateWeights.packedWeights),
      sliceBinding(2, gateWeights.rowScales),
      sliceBinding(3, upWeights.packedWeights),
      sliceBinding(4, upWeights.rowScales),
      binding(5, geluLookup),
      binding(6, outputBuffer),
      gemmaPrefillParameterBinding(7, projectionParams.slice),
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

export function encodeGemmaPrefillQatGateUpActivation(
  encoder: GPUCommandEncoder,
  pipelines: GemmaPrefillQatGateUpPipelines,
  resources: GemmaPrefillQatGateUpActivationResources,
): void {
  const pass = encoder.beginComputePass({ label: "Gemma prefill fused gate/up activation" });
  encodeGemmaPrefillQatGateUpActivationPass(pass, pipelines, resources);
  pass.end();
}

export function encodeGemmaPrefillQatGateUpActivationPass(
  pass: GPUComputePassEncoder,
  pipelines: GemmaPrefillQatGateUpPipelines,
  resources: GemmaPrefillQatGateUpActivationResources,
): void {
  if (resources.runsSrq) {
    if (!resources.srqBindGroup) {
      throw new Error("Gemma prefill fused gate/up activation SRQ bind group is missing");
    }
    pass.setPipeline(pipelines.srq);
    pass.setBindGroup(0, resources.srqBindGroup);
    pass.dispatchWorkgroups(
      Math.ceil(pipelines.rows * pipelines.inFeatures / SRQ_WORKGROUP_SIZE),
    );
  }
  pass.setPipeline(pipelines.projection);
  pass.setBindGroup(0, resources.projectionBindGroup);
  pass.dispatchWorkgroups(resources.workgroupCount, 1, resources.rowTileCount);
}

export function destroyGemmaPrefillQatGateUpActivationResources(
  resources: GemmaPrefillQatGateUpActivationResources,
): void {
  for (const buffer of resources.ownedBuffers) buffer.destroy();
}

export function createGemmaPrefillQatGateUpShader(
  geometry: GemmaPrefillQatLinearGeometry,
  rowsPerTile = geometry.rows >= 8 ? 8 : geometry.rows > 2 ? 2 : geometry.rows,
  outputRowsPerWorkgroup = geometry.outFeatures >= 32768
    ? 8
    : geometry.outFeatures >= 1024 ? 2 : 1,
  fuseActivation = false,
): string {
  validateGeometry(geometry);
  if (![1, 2, 8].includes(rowsPerTile) || ![1, 2, 8].includes(outputRowsPerWorkgroup)) {
    throw new Error("Gemma prefill fused gate/up tile geometry is unsupported");
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
      declarations.push(`  var gateDotSum${token}_${outputRow}: f32 = 0.0;`);
      declarations.push(`  var upDotSum${token}_${outputRow}: f32 = 0.0;`);
    }
  }
  for (let outputRow = 0; outputRow < outputRowsPerWorkgroup; outputRow += 1) {
    unpack.push(`    var gatePacked${outputRow}: u32 = 0u;`);
    unpack.push(`    var upPacked${outputRow}: u32 = 0u;`);
    unpack.push(
      `    if (outputBase + ${outputRow}u < OUT_FEATURES) { ` +
      `gatePacked${outputRow} = gatePackedWeights[(outputBase + ${outputRow}u) * WORDS_PER_ROW + word]; ` +
      `upPacked${outputRow} = upPackedWeights[(outputBase + ${outputRow}u) * WORDS_PER_ROW + word]; }`,
    );
    for (const prefix of ["gate", "up"]) {
      const packed = `${prefix}Packed${outputRow}`;
      if (geometry.bits === 4) {
        unpack.push(`    let ${prefix}Low${outputRow} = vec4<f32>(unpack4xU8(${packed} & 0x0f0f0f0fu));`);
        unpack.push(`    let ${prefix}High${outputRow} = vec4<f32>(unpack4xU8((${packed} >> 4u) & 0x0f0f0f0fu));`);
        unpack.push(
          `    let ${prefix}Codes${outputRow}_0 = vec4<f32>(${prefix}Low${outputRow}.x, ` +
          `${prefix}High${outputRow}.x, ${prefix}Low${outputRow}.y, ${prefix}High${outputRow}.y);`,
        );
        unpack.push(
          `    let ${prefix}Codes${outputRow}_1 = vec4<f32>(${prefix}Low${outputRow}.z, ` +
          `${prefix}High${outputRow}.z, ${prefix}Low${outputRow}.w, ${prefix}High${outputRow}.w);`,
        );
      } else {
        for (let chunk = 0; chunk < 4; chunk += 1) {
          unpack.push(
            `    let ${prefix}Unpacked${outputRow}_${chunk} = vec4<f32>(unpack4xU8(` +
            `(${packed} >> ${chunk * 2}u) & 0x03030303u));`,
          );
        }
        for (let chunk = 0; chunk < 4; chunk += 1) {
          const component = ["x", "y", "z", "w"][chunk];
          unpack.push(
            `    let ${prefix}Codes${outputRow}_${chunk} = vec4<f32>(` +
            [0, 1, 2, 3]
              .map((part) => `${prefix}Unpacked${outputRow}_${part}.${component}`)
              .join(", ") +
            `);`,
          );
        }
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
          `      gateDotSum${token}_${outputRow} = gateDotSum${token}_${outputRow} + ` +
          `dot(gateCodes${outputRow}_${chunk}, activation${token}_${chunk});`,
        );
        accumulation.push(
          `      upDotSum${token}_${outputRow} = upDotSum${token}_${outputRow} + ` +
          `dot(upCodes${outputRow}_${chunk}, activation${token}_${chunk});`,
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
      epilogue.push(`      let reducedGateDot = subgroupAdd(gateDotSum${token}_${outputRow});`);
      epilogue.push(`      let reducedUpDot = subgroupAdd(upDotSum${token}_${outputRow});`);
      epilogue.push(`      let outputRow = outputBase + ${outputRow}u;`);
      epilogue.push("      if (lane == 0u && outputRow < OUT_FEATURES) {");
      epilogue.push(
        `        let gateValue = srq(` +
        `gateRowScales[outputRow] * (reducedGateDot - ZERO_POINT * reducedActivation${token}), ` +
        `params.gateOutputScale);`,
      );
      epilogue.push(
        `        let upValue = srq(` +
        `upRowScales[outputRow] * (reducedUpDot - ZERO_POINT * reducedActivation${token}), ` +
        `params.upOutputScale);`,
      );
      if (fuseActivation) {
        epilogue.push(
          "        let lookupIndex = u32(clamp(round(gateValue / params.gateOutputScale), " +
          "-128.0, 127.0) + 128.0);",
        );
        epilogue.push("        let activated = gateGeluLookup[lookupIndex] * upValue;");
        epilogue.push(
          `        activatedOutput[(tokenStart + ${token}u) * OUT_FEATURES + outputRow] = ` +
          "srqExact(activated, params.downInputScale, params.downInputInverseScale);",
        );
      } else {
        epilogue.push(
          `        gateOutput[(tokenStart + ${token}u) * OUT_FEATURES + outputRow] = gateValue;`,
        );
        epilogue.push(
          `        upOutput[(tokenStart + ${token}u) * OUT_FEATURES + outputRow] = upValue;`,
        );
      }
      epilogue.push("      }");
      epilogue.push("    }");
    }
    epilogue.push("  }");
  }

  const params = fuseActivation
    ? `struct Params {
  gateOutputScale: f32,
  upOutputScale: f32,
  downInputScale: f32,
  downInputInverseScale: f32,
}`
    : `struct Params {
  gateOutputScale: f32,
  upOutputScale: f32,
}`;
  const bindings = fuseActivation
    ? `@group(0) @binding(0) var<storage, read> activation: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> gatePackedWeights: array<u32>;
@group(0) @binding(2) var<storage, read> gateRowScales: array<f32>;
@group(0) @binding(3) var<storage, read> upPackedWeights: array<u32>;
@group(0) @binding(4) var<storage, read> upRowScales: array<f32>;
@group(0) @binding(5) var<storage, read> gateGeluLookup: array<f32>;
@group(0) @binding(6) var<storage, read_write> activatedOutput: array<f32>;
@group(0) @binding(7) var<uniform> params: Params;`
    : `@group(0) @binding(0) var<storage, read> activation: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> gatePackedWeights: array<u32>;
@group(0) @binding(2) var<storage, read> gateRowScales: array<f32>;
@group(0) @binding(3) var<storage, read_write> gateOutput: array<f32>;
@group(0) @binding(4) var<storage, read> upPackedWeights: array<u32>;
@group(0) @binding(5) var<storage, read> upRowScales: array<f32>;
@group(0) @binding(6) var<storage, read_write> upOutput: array<f32>;
@group(0) @binding(7) var<uniform> params: Params;`;

  return `enable subgroups;

${params}

${bindings}

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

fn divExact(value: f32, scale: f32, inverseScale: f32) -> f32 {
  let quotient = value * inverseScale;
  let remainder = fma(-scale, quotient, value);
  return fma(remainder, inverseScale, quotient);
}

fn srqExact(value: f32, scale: f32, inverseScale: f32) -> f32 {
  return clamp(round(divExact(value, scale, inverseScale)), -128.0, 127.0) * scale;
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
    throw new Error("Gemma prefill fused gate/up geometry is unsupported");
  }
}