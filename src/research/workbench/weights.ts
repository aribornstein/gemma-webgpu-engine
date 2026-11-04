import type { GemmaInputTensorSource } from "../../model/gemma-input-weights";
import { createGemmaLayerPlan, type GemmaProjectionPlan } from "../../model/gemma-layer-plan";

export const PROJECTIONS = ["q", "k", "v", "o", "gate", "up", "down", "pleGate", "pleProjection"] as const;
export type ProjectionName = typeof PROJECTIONS[number];
export interface WeightSliceRequest {
  layer: number;
  projection: ProjectionName;
  rowStart: number;
  rowCount: number;
  columnStart: number;
  columnCount: number;
}
export interface WeightSlice {
  tensor: string;
  scaleTensor: string;
  storageDtype: string;
  logicalShape: [number, number];
  packedBits: number;
  rowStart: number;
  columnStart: number;
  shape: [number, number];
  codes: number[];
  rowScales: number[];
  values: number[];
  inputScale: number;
  outputScale: number;
  bytesRead: number;
}

export function validateWeightSliceRequest(request: WeightSliceRequest): void {
  if (!Number.isInteger(request.layer) || request.layer < 0 || request.layer >= 35 ||
      !PROJECTIONS.includes(request.projection)) throw new Error("Unsupported layer or projection");
  for (const key of ["rowStart", "columnStart"] as const) {
    if (!Number.isSafeInteger(request[key]) || request[key] < 0) throw new Error(`Invalid ${key}`);
  }
  if (!Number.isInteger(request.rowCount) || request.rowCount < 1 || request.rowCount > 64 ||
      !Number.isInteger(request.columnCount) || request.columnCount < 1 || request.columnCount > 4096 ||
      request.rowCount * request.columnCount > 65536) throw new Error("Weight inspection is limited to 65,536 values (64 rows)");
}

export async function inspectWeightSlice(source: GemmaInputTensorSource, request: WeightSliceRequest): Promise<WeightSlice> {
  validateWeightSliceRequest(request);
  const plan = createGemmaLayerPlan(source.descriptors, request.layer);
  const projections: Record<ProjectionName, GemmaProjectionPlan | null> = {
    q: plan.attention.q, k: plan.attention.k, v: plan.attention.v, o: plan.attention.output,
    gate: plan.mlp.gate, up: plan.mlp.up, down: plan.mlp.down,
    pleGate: plan.ple.inputGate, pleProjection: plan.ple.projection,
  };
  const projection = projections[request.projection];
  if (!projection) throw new Error("This shared-KV layer has no owned K/V projection");
  const bits = projection.weight.dtype === "I8" ? 8 :
    ["gate", "up", "down"].includes(request.projection) ? plan.mlp.bits : 4;
  const rowBytes = projection.weight.shape[1];
  const columns = rowBytes * 8 / bits;
  if (request.rowStart + request.rowCount > projection.weight.shape[0] ||
      request.columnStart + request.columnCount > columns) throw new Error("Weight slice exceeds tensor bounds");
  const firstByte = Math.floor(request.columnStart * bits / 8);
  const byteCount = Math.ceil((request.columnStart + request.columnCount) * bits / 8) - firstByte;
  const rowScalesBytes = await source.readTensorSlice(projection.weightScale.name, request.rowStart * 4, request.rowCount * 4);
  const scalesView = new DataView(rowScalesBytes.buffer, rowScalesBytes.byteOffset, rowScalesBytes.byteLength);
  const rowScales = Array.from({ length: request.rowCount }, (_, i) => scalesView.getFloat32(i * 4, true));
  const codes: number[] = [];
  const values: number[] = [];
  for (let row = 0; row < request.rowCount; row += 1) {
    const bytes = await source.readTensorSlice(projection.weight.name, (request.rowStart + row) * rowBytes + firstByte, byteCount);
    for (let column = request.columnStart; column < request.columnStart + request.columnCount; column += 1) {
      const byte = bytes[Math.floor(column * bits / 8) - firstByte];
      const code = bits === 8 ? (byte << 24) >> 24 : ((byte >> ((column * bits) % 8)) & ((1 << bits) - 1)) - (1 << (bits - 1));
      codes.push(code);
      values.push(Math.fround(code * rowScales[row]));
    }
  }
  const scalar = async (name: string) => {
    const bytes = await source.readTensorSlice(name, 0, 4);
    return new DataView(bytes.buffer, bytes.byteOffset, 4).getFloat32(0, true);
  };
  return {
    tensor: projection.weight.name, scaleTensor: projection.weightScale.name,
    storageDtype: projection.weight.dtype, packedBits: bits,
    logicalShape: [projection.weight.shape[0], columns],
    rowStart: request.rowStart, columnStart: request.columnStart,
    shape: [request.rowCount, request.columnCount], codes, values, rowScales,
    inputScale: await scalar(projection.inputActivationScale.name),
    outputScale: await scalar(projection.outputActivationScale.name),
    bytesRead: byteCount * request.rowCount + rowScalesBytes.byteLength + 8,
  };
}
