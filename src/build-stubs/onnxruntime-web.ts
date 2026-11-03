// Production-build stand-in for `onnxruntime-web/webgpu`. The engine uses @huggingface/transformers
// only for its tokenizer, which never creates an ONNX session; aliasing the runtime here keeps the
// 23.5 MB ONNX Runtime WebAssembly binary out of the demo and library bundles.
export const env: Record<string, unknown> = { wasm: {}, webgpu: {}, versions: {} };

export class InferenceSession {
  static create(): never {
    throw new Error("ONNX Runtime is not bundled with the Gemma WebGPU engine build");
  }
}

export class Tensor {}
