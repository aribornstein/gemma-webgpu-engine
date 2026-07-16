export const GEMMA_MODEL_CONTEXT_CAPACITY = 131_072;
export const GEMMA_VALIDATED_CONTEXT_CAPACITY = 32_768;
export const GEMMA_SLIDING_CACHE_CAPACITY = 512;

const GEMMA_SLIDING_CACHE_OWNERS = 12;
const GEMMA_FULL_CACHE_OWNERS = 3;
const GEMMA_SLIDING_HEAD_DIMENSION = 256;
const GEMMA_FULL_HEAD_DIMENSION = 512;
const FLOAT_BYTES = Float32Array.BYTES_PER_ELEMENT;

export interface GemmaContextMemoryPlan {
  contextCapacity: number;
  slidingPhysicalCapacity: number;
  fullPhysicalCapacity: number;
  largestCacheBufferBytes: number;
  kvCacheBytes: number;
}

export interface GemmaContextDeviceLimits {
  maxBufferSize: number;
  maxStorageBufferBindingSize: number;
}

export function planGemmaContextMemory(contextCapacity: number): GemmaContextMemoryPlan {
  if (!Number.isInteger(contextCapacity) || contextCapacity < 1 ||
      contextCapacity > GEMMA_MODEL_CONTEXT_CAPACITY) {
    throw new Error(
      `Gemma context capacity must be between 1 and ${GEMMA_MODEL_CONTEXT_CAPACITY}`,
    );
  }
  const slidingPhysicalCapacity = Math.min(
    contextCapacity,
    GEMMA_SLIDING_CACHE_CAPACITY,
  );
  const fullPhysicalCapacity = contextCapacity;
  const slidingBytes = GEMMA_SLIDING_CACHE_OWNERS * 2 * slidingPhysicalCapacity *
    GEMMA_SLIDING_HEAD_DIMENSION * FLOAT_BYTES;
  const fullBytes = GEMMA_FULL_CACHE_OWNERS * 2 * fullPhysicalCapacity *
    GEMMA_FULL_HEAD_DIMENSION * FLOAT_BYTES;
  return {
    contextCapacity,
    slidingPhysicalCapacity,
    fullPhysicalCapacity,
    largestCacheBufferBytes: fullPhysicalCapacity * GEMMA_FULL_HEAD_DIMENSION * FLOAT_BYTES,
    kvCacheBytes: slidingBytes + fullBytes,
  };
}

export function assertGemmaContextSupported(
  contextCapacity: number,
  limits: GemmaContextDeviceLimits,
): GemmaContextMemoryPlan {
  const plan = planGemmaContextMemory(contextCapacity);
  const bufferLimit = Math.min(
    Number(limits.maxBufferSize),
    Number(limits.maxStorageBufferBindingSize),
  );
  if (plan.largestCacheBufferBytes > bufferLimit) {
    throw new Error(
      `Gemma ${contextCapacity}-position full-attention cache requires ` +
      `${plan.largestCacheBufferBytes} bytes per buffer, exceeding device limit ${bufferLimit}`,
    );
  }
  return plan;
}

export function availableGemmaOutputTokens(
  promptTokenCount: number,
  contextCapacity: number,
): number {
  if (!Number.isInteger(promptTokenCount) || promptTokenCount < 1) {
    throw new Error("Gemma prompt token count must be a positive integer");
  }
  if (!Number.isInteger(contextCapacity) || contextCapacity < 1) {
    throw new Error("Gemma context capacity must be a positive integer");
  }
  return Math.max(0, contextCapacity - promptTokenCount + 1);
}