import type { DecodeKvCache } from "./decode-kv-cache";

const BYTES_PER_ELEMENT = Float32Array.BYTES_PER_ELEMENT;

/**
 * Per-beam storage for the K/V rows that beam search generates after a shared prompt.
 *
 * Every beam stores its generated rows `[0, rows)` (cache positions `basePosition + row`) for each
 * owner cache. Before a beam's next token is evaluated, its rows are copied back into the live
 * caches; afterwards the newly written row is saved. Circular (sliding-window) caches only receive
 * rows that are still inside the window, so prompt rows that the window still needs are never
 * overwritten. Two slot sets are kept so `reorder` can copy parents into children without
 * aliasing.
 */
export class DecodeKvBeamSlots {
  readonly beamCount: number;
  readonly maxRows: number;
  readonly basePosition: number;
  private readonly caches: readonly DecodeKvCache[];
  private current: GPUBuffer[][];
  private next: GPUBuffer[][];
  private destroyed = false;

  constructor(
    device: GPUDevice,
    caches: readonly DecodeKvCache[],
    beamCount: number,
    maxRows: number,
    basePosition: number,
  ) {
    if (!Number.isInteger(beamCount) || beamCount < 1) throw new Error("Beam slot count must be a positive integer");
    if (!Number.isInteger(maxRows) || maxRows < 1) throw new Error("Beam slot rows must be a positive integer");
    if (!Number.isInteger(basePosition) || basePosition < 0) {
      throw new Error("Beam slot base position must be a non-negative integer");
    }
    this.caches = caches;
    this.beamCount = beamCount;
    this.maxRows = maxRows;
    this.basePosition = basePosition;
    const allocate = (set: string) => Array.from({ length: beamCount }, (_, beam) =>
      caches.map((cache, cacheIndex) => device.createBuffer({
        label: `Beam ${set} slot ${beam} cache ${cacheIndex}`,
        size: 2 * maxRows * cache.tokenElements * BYTES_PER_ELEMENT,
        usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      })));
    this.current = allocate("current");
    try {
      this.next = allocate("next");
    } catch (error) {
      for (const buffers of this.current) for (const buffer of buffers) buffer.destroy();
      throw error;
    }
  }

  /** Bytes held by both slot sets. */
  get bytesAllocated(): number {
    const perBeam = this.caches.reduce(
      (sum, cache) => sum + 2 * this.maxRows * cache.tokenElements * BYTES_PER_ELEMENT,
      0,
    );
    return 2 * this.beamCount * perBeam;
  }

  /** Makes slot `beam` a copy of previous slot `parents[beam]` rows `[0, rows)`. */
  encodeReorder(encoder: GPUCommandEncoder, parents: readonly number[], rows: number): void {
    this.assertAlive();
    if (parents.length > this.beamCount) throw new Error("Beam reorder has too many beams");
    this.validateRows(rows);
    if (rows > 0) {
      parents.forEach((parent, beam) => {
        if (!Number.isInteger(parent) || parent < 0 || parent >= this.beamCount) {
          throw new Error(`Beam parent ${parent} is outside the slot range`);
        }
        this.caches.forEach((cache, cacheIndex) => {
          const rowBytes = cache.tokenElements * BYTES_PER_ELEMENT;
          for (const half of [0, 1]) {
            const offset = half * this.maxRows * rowBytes;
            encoder.copyBufferToBuffer(
              this.current[parent][cacheIndex],
              offset,
              this.next[beam][cacheIndex],
              offset,
              rows * rowBytes,
            );
          }
        });
      });
    }
    const previous = this.current;
    this.current = this.next;
    this.next = previous;
  }

  /**
   * Copies slot `beam` rows `[0, rows)` into the caches and truncates every cache so the next
   * evaluated token writes position `basePosition + rows`.
   */
  encodeRestore(encoder: GPUCommandEncoder, beam: number, rows: number): void {
    this.assertAlive();
    this.validateBeam(beam);
    this.validateRows(rows);
    const nextPosition = this.basePosition + rows;
    this.caches.forEach((cache, cacheIndex) => {
      const rowBytes = cache.tokenElements * BYTES_PER_ELEMENT;
      const firstPosition = cache.mode === "circular"
        ? Math.max(this.basePosition, nextPosition - cache.capacity + 1)
        : this.basePosition;
      let position = firstPosition;
      while (position < nextPosition) {
        const physical = cache.physicalPosition(position);
        const run = cache.mode === "circular"
          ? Math.min(nextPosition - position, cache.capacity - physical)
          : nextPosition - position;
        const row = position - this.basePosition;
        for (const [half, target] of [[0, cache.keyBuffer], [1, cache.valueBuffer]] as const) {
          encoder.copyBufferToBuffer(
            this.current[beam][cacheIndex],
            (half * this.maxRows + row) * rowBytes,
            target,
            physical * rowBytes,
            run * rowBytes,
          );
        }
        position += run;
      }
      cache.truncate(nextPosition);
    });
  }

  /** Saves cache position `basePosition + row` into slot `beam`. */
  encodeSave(encoder: GPUCommandEncoder, beam: number, row: number): void {
    this.assertAlive();
    this.validateBeam(beam);
    if (!Number.isInteger(row) || row < 0 || row >= this.maxRows) {
      throw new Error(`Beam slot row ${row} is outside 0..${this.maxRows - 1}`);
    }
    this.caches.forEach((cache, cacheIndex) => {
      const rowBytes = cache.tokenElements * BYTES_PER_ELEMENT;
      const physical = cache.physicalPosition(this.basePosition + row);
      for (const [half, source] of [[0, cache.keyBuffer], [1, cache.valueBuffer]] as const) {
        encoder.copyBufferToBuffer(
          source,
          physical * rowBytes,
          this.current[beam][cacheIndex],
          (half * this.maxRows + row) * rowBytes,
          rowBytes,
        );
      }
    });
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    for (const set of [this.current, this.next]) {
      for (const buffers of set) for (const buffer of buffers) buffer.destroy();
    }
  }

  private validateBeam(beam: number): void {
    if (!Number.isInteger(beam) || beam < 0 || beam >= this.beamCount) {
      throw new Error(`Beam ${beam} is outside the slot range`);
    }
  }

  private validateRows(rows: number): void {
    if (!Number.isInteger(rows) || rows < 0 || rows > this.maxRows) {
      throw new Error(`Beam slot rows ${rows} are outside 0..${this.maxRows}`);
    }
  }

  private assertAlive(): void {
    if (this.destroyed) throw new Error("Beam K/V slots have been destroyed");
  }
}
