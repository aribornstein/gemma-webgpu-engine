export function createDecodeVRmsShader(): string {
  return `struct Params {
  rows: u32,
  dim: u32,
  dstOffset: u32,
  padding: u32,
}

@group(0) @binding(0) var<storage, read_write> values: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

const WG: u32 = 256u;
const EPS: f32 = 0.000001;

var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(
  @builtin(workgroup_id) workgroup_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  let row = workgroup_id.x;
  if (row >= params.rows) { return; }
  let thread = local_id.x;
  let base = params.dstOffset + row * params.dim;
  var square_sum = 0.0;
  var dimension = thread;
  loop {
    if (dimension >= params.dim) { break; }
    let value = values[base + dimension];
    square_sum = square_sum + value * value;
    dimension = dimension + WG;
  }
  red[thread] = square_sum;
  workgroupBarrier();
  var stride = WG / 2u;
  loop {
    if (stride == 0u) { break; }
    if (thread < stride) {
      red[thread] = red[thread] + red[thread + stride];
    }
    stride = stride / 2u;
    workgroupBarrier();
  }
  let scale = inverseSqrt(red[0] / f32(params.dim) + EPS);
  dimension = thread;
  loop {
    if (dimension >= params.dim) { break; }
    let index = base + dimension;
    values[index] = values[index] * scale;
    dimension = dimension + WG;
  }
}`;
}