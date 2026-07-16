export interface GemmaPrefillParameterSlice {
  buffer: GPUBuffer;
  offset: number;
  size: number;
}

const DEFAULT_SLOTS_PER_BUFFER = 256;

export class GemmaPrefillParameterArena {
  private readonly device: GPUDevice;
  private readonly alignment: number;
  private readonly chunkBytes: number;
  private readonly allocatedBuffers: GPUBuffer[] = [];
  private currentBuffer: GPUBuffer | null = null;
  private currentOffset = 0;

  constructor(
    device: GPUDevice,
    slotsPerBuffer = DEFAULT_SLOTS_PER_BUFFER,
  ) {
    if (!Number.isInteger(slotsPerBuffer) || slotsPerBuffer < 1) {
      throw new Error("Gemma prefill parameter arena slot count must be positive");
    }
    this.device = device;
    this.alignment = device.limits.minUniformBufferOffsetAlignment;
    this.chunkBytes = this.alignment * slotsPerBuffer;
  }

  allocate(size: number, label: string): GemmaPrefillParameterSlice {
    if (!Number.isInteger(size) || size < 1 || size > this.alignment) {
      throw new Error("Gemma prefill parameter allocation exceeds one aligned slot");
    }
    if (!this.currentBuffer || this.currentOffset + this.alignment > this.chunkBytes) {
      this.currentBuffer = this.device.createBuffer({
        label: `${label} arena`,
        size: this.chunkBytes,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      this.allocatedBuffers.push(this.currentBuffer);
      this.currentOffset = 0;
    }
    const allocation = {
      buffer: this.currentBuffer,
      offset: this.currentOffset,
      size,
    };
    this.currentOffset += this.alignment;
    return allocation;
  }

  get buffers(): readonly GPUBuffer[] {
    return this.allocatedBuffers;
  }

  destroy(): void {
    for (const buffer of this.allocatedBuffers.toReversed()) buffer.destroy();
    this.allocatedBuffers.length = 0;
    this.currentBuffer = null;
    this.currentOffset = 0;
  }
}

export function createGemmaPrefillParameter(
  device: GPUDevice,
  size: number,
  label: string,
  arena?: GemmaPrefillParameterArena,
): { slice: GemmaPrefillParameterSlice; ownedBuffers: GPUBuffer[] } {
  if (arena) return { slice: arena.allocate(size, label), ownedBuffers: [] };
  const buffer = device.createBuffer({
    label,
    size,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  return {
    slice: { buffer, offset: 0, size },
    ownedBuffers: [buffer],
  };
}

export function gemmaPrefillParameterBinding(
  binding: number,
  slice: GemmaPrefillParameterSlice,
): GPUBindGroupEntry {
  return {
    binding,
    resource: { buffer: slice.buffer, offset: slice.offset, size: slice.size },
  };
}

export function writeGemmaPrefillParameter(
  device: GPUDevice,
  slice: GemmaPrefillParameterSlice,
  data: AllowSharedBufferSource,
): void {
  device.queue.writeBuffer(slice.buffer, slice.offset, data);
}
