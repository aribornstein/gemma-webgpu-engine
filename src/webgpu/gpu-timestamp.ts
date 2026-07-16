export async function measureGpuDispatches(
  device: GPUDevice,
  label: string,
  dispatchesPerSample: number,
  encodeDispatch: (pass: GPUComputePassEncoder) => void,
  sampleCount = 10,
  beforeSample?: () => void,
): Promise<number[] | null> {
  if (!device.features.has("timestamp-query")) return null;
  const querySet = device.createQuerySet({ type: "timestamp", count: 2 });
  const resolveBuffer = device.createBuffer({
    label: `${label} timestamp resolve`,
    size: 16,
    usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
  });
  const readBuffer = device.createBuffer({
    label: `${label} timestamp readback`,
    size: 16,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const samples: number[] = [];

  try {
    for (let sample = -2; sample < sampleCount; sample += 1) {
      beforeSample?.();
      const encoder = device.createCommandEncoder({ label: `${label} timestamp sample` });
      const pass = encoder.beginComputePass({
        label: `${label} timestamp batch`,
        timestampWrites: {
          querySet,
          beginningOfPassWriteIndex: 0,
          endOfPassWriteIndex: 1,
        },
      });
      for (let dispatch = 0; dispatch < dispatchesPerSample; dispatch += 1) {
        encodeDispatch(pass);
      }
      pass.end();
      encoder.resolveQuerySet(querySet, 0, 2, resolveBuffer, 0);
      encoder.copyBufferToBuffer(resolveBuffer, 0, readBuffer, 0, 16);
      device.queue.submit([encoder.finish()]);
      await readBuffer.mapAsync(GPUMapMode.READ);
      const timestamps = new BigUint64Array(readBuffer.getMappedRange().slice(0));
      readBuffer.unmap();
      if (sample >= 0) {
        samples.push(
          Number(timestamps[1] - timestamps[0]) / 1e6 / dispatchesPerSample,
        );
      }
    }
    return samples.toSorted((left, right) => left - right);
  } finally {
    querySet.destroy();
    resolveBuffer.destroy();
    readBuffer.destroy();
  }
}