import {
  checkpointFloat32,
  checkpointUint16,
  checkpointUint32,
  loadCheckpointFixture,
} from "./checkpoint-fixture";
import { modelAssetUrl } from "./model-assets";

const METADATA_URL = modelAssetUrl(
  "gemma-4-e2b/operators/decode-mlp-ple-layer0.json",
);
const METADATA_SHA256 = "05a79837b22bfec2d8d832970f38ba624624c1db8f4abaa5c26755730c95d1d6";
const TENSOR_FILE_SHA256 = "0ee88858daf8e800bae18bb15c8901d578e5a50a32a4e3e4ffe9e426d02d43c0";

export interface DecodeMlpPleFixture {
  preMlpInputBits: Uint16Array;
  preMlpSum: Float32Array;
  hiddenBeforeDown: Float32Array;
  gateBits: Uint32Array;
  gateScales: Float32Array;
  upBits: Uint32Array;
  upScales: Float32Array;
  gateGeluLut: Float32Array;
  downBits: Uint32Array;
  downScales: Float32Array;
  postFfNorm: Float32Array;
  pleInput: Float32Array;
  pleGateWeights: Uint32Array;
  pleGateRowScales: Float32Array;
  pleGeluLut: Float32Array;
  pleProjectionWeights: Uint32Array;
  pleProjectionRowScales: Float32Array;
  pleNormWeights: Float32Array;
  expectedGateUpBits: Uint16Array;
  expectedHiddenAfterDown: Float32Array;
  expectedPleGateOutput: Float32Array;
  expectedHiddenAfterPle: Float32Array;
  expectedNextLayerInput: Float32Array;
  expectedNextLayerSum: Float32Array;
  metadataSha256: string;
  tensorFileSha256: string;
}

let fixturePromise: Promise<DecodeMlpPleFixture> | null = null;

export function loadDecodeMlpPleFixture(): Promise<DecodeMlpPleFixture> {
  fixturePromise ??= loadFixture().catch((error) => {
    fixturePromise = null;
    throw error;
  });
  return fixturePromise;
}

async function loadFixture(): Promise<DecodeMlpPleFixture> {
  const fixture = await loadCheckpointFixture(
    METADATA_URL,
    METADATA_SHA256,
    TENSOR_FILE_SHA256,
  );
  return {
    preMlpInputBits: checkpointUint16(fixture, "preMlpInput"),
    preMlpSum: checkpointFloat32(fixture, "preMlpSum"),
    hiddenBeforeDown: checkpointFloat32(fixture, "hiddenBeforeDown"),
    gateBits: checkpointUint32(fixture, "gateBits"),
    gateScales: checkpointFloat32(fixture, "gateScales"),
    upBits: checkpointUint32(fixture, "upBits"),
    upScales: checkpointFloat32(fixture, "upScales"),
    gateGeluLut: checkpointFloat32(fixture, "gateGeluLut"),
    downBits: checkpointUint32(fixture, "downBits"),
    downScales: checkpointFloat32(fixture, "downScales"),
    postFfNorm: checkpointFloat32(fixture, "postFfNorm"),
    pleInput: checkpointFloat32(fixture, "pleInput"),
    pleGateWeights: checkpointUint32(fixture, "pleGateWeights"),
    pleGateRowScales: checkpointFloat32(fixture, "pleGateRowScales"),
    pleGeluLut: checkpointFloat32(fixture, "pleGeluLut"),
    pleProjectionWeights: checkpointUint32(fixture, "pleProjectionWeights"),
    pleProjectionRowScales: checkpointFloat32(fixture, "pleProjectionRowScales"),
    pleNormWeights: checkpointFloat32(fixture, "pleNormWeights"),
    expectedGateUpBits: checkpointUint16(fixture, "gateUpOutput"),
    expectedHiddenAfterDown: checkpointFloat32(fixture, "hiddenAfterDown"),
    expectedPleGateOutput: checkpointFloat32(fixture, "pleGateOutput"),
    expectedHiddenAfterPle: checkpointFloat32(fixture, "hiddenAfterPle"),
    expectedNextLayerInput: checkpointFloat32(fixture, "nextLayerInput"),
    expectedNextLayerSum: checkpointFloat32(fixture, "nextLayerSum"),
    metadataSha256: fixture.metadataSha256,
    tensorFileSha256: fixture.tensorFileSha256,
  };
}