import type {
  BenchmarkAdapter,
  BenchmarkCase,
  GenerationCallbacks,
  GenerationResult,
  LoadOptions,
  LoadResult,
} from "../types";

export class PinnedHuggingFaceBenchmarkAdapter implements BenchmarkAdapter {
  readonly id = "pinned-hugging-face-webgpu";
  readonly runtimeName = "Pinned Hugging Face WebGPU";
  readonly runtimeVersion = "158f16ae0f672943ca304d59c47c8e3a264e399e";
  readonly modelId = "google/gemma-4-E2B-it-qat-mobile-transformers";
  readonly modelRevision = "9fcec64df66cb1e4d972fc5cdc142afb25b2362c";
  readonly artifactType = "mobile-QAT safetensors";
  readonly artifactUrl = "https://huggingface.co/google/gemma-4-E2B-it-qat-mobile-transformers/resolve/main/model.safetensors";
  readonly artifactBytes = 2_458_111_846;
  readonly artifactEquivalence = "pinned-source-equivalent" as const;
  readonly available = false;
  readonly limitations = Object.freeze([
    "The pinned upstream browser bundle is not present in this workspace; no current-browser run is fabricated from prior evidence.",
  ]);

  async load(_options: LoadOptions): Promise<LoadResult> {
    throw new Error(this.limitations[0]);
  }
  async warmup(_testCase: BenchmarkCase): Promise<void> {
    throw new Error(this.limitations[0]);
  }
  async generate(_testCase: BenchmarkCase, _callbacks: GenerationCallbacks): Promise<GenerationResult> {
    throw new Error(this.limitations[0]);
  }
  async countTokens(_text: string): Promise<number> {
    throw new Error(this.limitations[0]);
  }
  async resetConversation(): Promise<void> {}
  async createConversation(): Promise<void> {}
  async dispose(): Promise<void> {}
}