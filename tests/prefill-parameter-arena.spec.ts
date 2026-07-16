import { expect, test } from "@playwright/test";
import {
  GemmaPrefillParameterArena,
  createGemmaPrefillParameter,
  gemmaPrefillParameterBinding,
  writeGemmaPrefillParameter,
} from "../src/webgpu/prefill-parameter-arena";

const originalGpuBufferUsage = Object.getOwnPropertyDescriptor(globalThis, "GPUBufferUsage");

interface FakeBuffer {
  label: string;
  size: number;
  destroyCount: number;
  destroy(): void;
}

function createFakeDevice() {
  const buffers: FakeBuffer[] = [];
  const writes: Array<{ buffer: FakeBuffer; offset: number; data: AllowSharedBufferSource }> = [];
  const device = {
    limits: { minUniformBufferOffsetAlignment: 256 },
    createBuffer(descriptor: GPUBufferDescriptor) {
      const buffer: FakeBuffer = {
        label: String(descriptor.label ?? ""),
        size: Number(descriptor.size),
        destroyCount: 0,
        destroy() {
          buffer.destroyCount += 1;
        },
      };
      buffers.push(buffer);
      return buffer as unknown as GPUBuffer;
    },
    queue: {
      writeBuffer(buffer: GPUBuffer, offset: number, data: AllowSharedBufferSource) {
        writes.push({ buffer: buffer as unknown as FakeBuffer, offset, data });
      },
    },
  } as unknown as GPUDevice;
  return { device, buffers, writes };
}

test.beforeEach(() => {
  Object.defineProperty(globalThis, "GPUBufferUsage", {
    configurable: true,
    value: { UNIFORM: 1, COPY_DST: 2 },
  });
});

test.afterEach(() => {
  if (originalGpuBufferUsage) {
    Object.defineProperty(globalThis, "GPUBufferUsage", originalGpuBufferUsage);
  } else {
    delete (globalThis as { GPUBufferUsage?: unknown }).GPUBufferUsage;
  }
});

test("packs aligned parameter slices and rolls over backing buffers", () => {
  const { device, buffers, writes } = createFakeDevice();
  const arena = new GemmaPrefillParameterArena(device, 2);

  const first = arena.allocate(16, "first");
  const second = arena.allocate(32, "second");
  const third = arena.allocate(16, "third");

  expect(first).toEqual({ buffer: buffers[0], offset: 0, size: 16 });
  expect(second).toEqual({ buffer: buffers[0], offset: 256, size: 32 });
  expect(third).toEqual({ buffer: buffers[1], offset: 0, size: 16 });
  expect(buffers.map(({ size }) => size)).toEqual([512, 512]);
  expect(arena.buffers).toEqual(buffers);
  expect(gemmaPrefillParameterBinding(3, second)).toEqual({
    binding: 3,
    resource: { buffer: buffers[0], offset: 256, size: 32 },
  });

  const data = new Uint32Array([7, 11, 13, 17]);
  writeGemmaPrefillParameter(device, second, data);
  expect(writes).toEqual([{ buffer: buffers[0], offset: 256, data }]);
  expect(() => arena.allocate(257, "too large")).toThrow(
    "Gemma prefill parameter allocation exceeds one aligned slot",
  );

  arena.destroy();
  expect(buffers.map(({ destroyCount }) => destroyCount)).toEqual([1, 1]);
  expect(arena.buffers).toEqual([]);
});

test("keeps private parameter ownership without an arena", () => {
  const { device, buffers } = createFakeDevice();
  const parameter = createGemmaPrefillParameter(device, 32, "private");

  expect(parameter.slice).toEqual({ buffer: buffers[0], offset: 0, size: 32 });
  expect(parameter.ownedBuffers).toEqual([buffers[0]]);
});
