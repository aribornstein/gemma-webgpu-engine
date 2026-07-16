export interface DecodeKvCacheOptions {
  capacity: number;
  kvHeads: number;
  headDim: number;
  mode?: "linear" | "circular";
  label?: string;
}

export interface DecodeKvCacheAllocation {
  capacity: number;
  mode: "linear" | "circular";
}

export interface DecodeKvCacheAllocationRequest {
  keyLength: number;
  cacheCapacity?: number;
  window: number;
}

export function canRetainDecodeKvPrefix(
  mode: "linear" | "circular",
  capacity: number,
  logicalLength: number,
  length: number,
): boolean {
  if (!Number.isInteger(length) || length < 0 || length > logicalLength) return false;
  return mode === "linear" || logicalLength <= capacity || length >= logicalLength - 1;
}

export function resolveDecodeKvCacheAllocation(
  request: DecodeKvCacheAllocationRequest,
): DecodeKvCacheAllocation {
  const logicalCapacity = request.cacheCapacity ?? request.keyLength;
  if (!Number.isInteger(logicalCapacity) || logicalCapacity < request.keyLength) {
    throw new Error("Decode attention logical cache capacity is invalid");
  }
  if (!Number.isInteger(request.window) || request.window < 0) {
    throw new Error("Decode attention window must be a non-negative integer");
  }
  return request.window > 0
    ? { capacity: Math.min(request.window, logicalCapacity), mode: "circular" }
    : { capacity: logicalCapacity, mode: "linear" };
}

export class DecodeKvCache {
  readonly capacity: number;
  readonly kvHeads: number;
  readonly headDim: number;
  readonly mode: "linear" | "circular";
  readonly tokenElements: number;
  readonly bufferBytes: number;
  readonly keyBuffer: GPUBuffer;
  readonly valueBuffer: GPUBuffer;

  private logicalLength = 0;
  private destroyed = false;

  constructor(device: GPUDevice, options: DecodeKvCacheOptions) {
    this.capacity = positiveInteger(options.capacity, "capacity");
    this.kvHeads = positiveInteger(options.kvHeads, "kvHeads");
    this.headDim = positiveInteger(options.headDim, "headDim");
    this.mode = options.mode ?? "linear";
    this.tokenElements = this.kvHeads * this.headDim;
    this.bufferBytes = this.capacity * this.tokenElements * Float32Array.BYTES_PER_ELEMENT;
    if (
      this.bufferBytes > device.limits.maxBufferSize ||
      this.bufferBytes > device.limits.maxStorageBufferBindingSize
    ) {
      throw new Error(
        `Decode K/V cache buffer requires ${this.bufferBytes} bytes, exceeding this device's storage limits`,
      );
    }

    const label = options.label ?? "Decode K/V cache";
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    this.keyBuffer = device.createBuffer({ label: `${label} keys`, size: this.bufferBytes, usage });
    this.valueBuffer = device.createBuffer({ label: `${label} values`, size: this.bufferBytes, usage });
  }

  get length(): number {
    return this.logicalLength;
  }

  get bytesAllocated(): number {
    return this.bufferBytes * 2;
  }

  get buffers(): readonly [GPUBuffer, GPUBuffer] {
    return [this.keyBuffer, this.valueBuffer];
  }

  canRetainPrefix(length: number): boolean {
    return canRetainDecodeKvPrefix(
      this.mode,
      this.capacity,
      this.logicalLength,
      length,
    );
  }

  truncate(length: number): void {
    this.assertAlive();
    if (!this.canRetainPrefix(length)) {
      throw new Error(
        `Decode K/V cache cannot retain prefix ${length} from logical length ${this.logicalLength}`,
      );
    }
    this.logicalLength = length;
  }

  elementOffset(position: number, kvHead = 0): number {
    this.assertAlive();
    const validPosition = this.physicalPosition(position);
    const validHead = boundedInteger(kvHead, "kvHead", this.kvHeads);
    return validPosition * this.tokenElements + validHead * this.headDim;
  }

  physicalPosition(position: number): number {
    if (!Number.isInteger(position) || position < 0) {
      throw new Error("Decode K/V cache position must be a non-negative integer");
    }
    if (this.mode === "circular") return position % this.capacity;
    return boundedInteger(position, "position", this.capacity);
  }

  byteOffset(position: number, kvHead = 0): number {
    return this.elementOffset(position, kvHead) * Float32Array.BYTES_PER_ELEMENT;
  }

  writeTokens(
    queue: GPUQueue,
    startPosition: number,
    keys: Float32Array,
    values: Float32Array,
  ): void {
    this.assertAlive();
    if (keys.length !== values.length || keys.length % this.tokenElements !== 0) {
      throw new Error("Decode K/V cache writes require equally shaped whole-token tensors");
    }
    const tokenCount = keys.length / this.tokenElements;
    this.validateWrite(startPosition, tokenCount);
    let writtenTokens = 0;
    while (writtenTokens < tokenCount) {
      const logicalPosition = startPosition + writtenTokens;
      const physicalPosition = this.physicalPosition(logicalPosition);
      const contiguousTokens = Math.min(tokenCount - writtenTokens, this.capacity - physicalPosition);
      const sourceStart = writtenTokens * this.tokenElements;
      const sourceEnd = sourceStart + contiguousTokens * this.tokenElements;
      const offset = physicalPosition * this.tokenElements * Float32Array.BYTES_PER_ELEMENT;
      queue.writeBuffer(this.keyBuffer, offset, keys.subarray(sourceStart, sourceEnd));
      queue.writeBuffer(this.valueBuffer, offset, values.subarray(sourceStart, sourceEnd));
      writtenTokens += contiguousTokens;
    }
    this.commitWrite(startPosition, tokenCount);
  }

  commitWrite(startPosition: number, tokenCount = 1): void {
    this.assertAlive();
    this.validateWrite(startPosition, tokenCount);
    this.logicalLength = Math.max(this.logicalLength, startPosition + tokenCount);
  }

  reset(): void {
    this.assertAlive();
    this.logicalLength = 0;
  }

  encodeClear(encoder: GPUCommandEncoder): void {
    this.assertAlive();
    encoder.clearBuffer(this.keyBuffer);
    encoder.clearBuffer(this.valueBuffer);
    this.logicalLength = 0;
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.logicalLength = 0;
    this.keyBuffer.destroy();
    this.valueBuffer.destroy();
  }

  private validateWrite(startPosition: number, tokenCount: number): void {
    if (!Number.isInteger(startPosition) || startPosition < 0) {
      throw new Error("Decode K/V cache startPosition must be a non-negative integer");
    }
    if (!Number.isInteger(tokenCount) || tokenCount < 1) {
      throw new Error("Decode K/V cache tokenCount must be a positive integer");
    }
    if (startPosition > this.logicalLength) {
      throw new Error(
        `Decode K/V cache write at ${startPosition} would leave a gap after ${this.logicalLength}`,
      );
    }
    if (this.mode === "linear" && startPosition + tokenCount > this.capacity) {
      throw new Error(
        `Decode K/V cache write ending at ${startPosition + tokenCount} exceeds capacity ${this.capacity}`,
      );
    }
  }

  private assertAlive(): void {
    if (this.destroyed) throw new Error("Decode K/V cache has been destroyed");
  }
}

function positiveInteger(value: number, name: string): number {
  if (!Number.isInteger(value) || value < 1) {
    throw new Error(`Decode K/V cache ${name} must be a positive integer`);
  }
  return value;
}

function boundedInteger(value: number, name: string, upperBound: number): number {
  if (!Number.isInteger(value) || value < 0 || value >= upperBound) {
    throw new Error(`Decode K/V cache ${name} must be an integer below ${upperBound}`);
  }
  return value;
}