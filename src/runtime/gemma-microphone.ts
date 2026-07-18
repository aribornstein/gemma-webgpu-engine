export interface GemmaMicrophoneCapture {
  readonly sampleRate: number;
  stop(): Promise<Blob>;
  discard(): Promise<void>;
}

export async function startGemmaMicrophoneCapture(
  stream: MediaStream,
): Promise<GemmaMicrophoneCapture> {
  const context = new AudioContext();
  const source = context.createMediaStreamSource(stream);
  const processor = context.createScriptProcessor(4096, 1, 1);
  const silence = context.createGain();
  const chunks: Float32Array[] = [];
  let active = true;
  silence.gain.value = 0;
  processor.onaudioprocess = ({ inputBuffer }) => {
    if (!active) return;
    const mono = new Float32Array(inputBuffer.length);
    for (let channel = 0; channel < inputBuffer.numberOfChannels; channel += 1) {
      const values = inputBuffer.getChannelData(channel);
      for (let index = 0; index < values.length; index += 1) {
        mono[index] += values[index] / inputBuffer.numberOfChannels;
      }
    }
    chunks.push(mono);
  };
  source.connect(processor);
  processor.connect(silence);
  silence.connect(context.destination);
  try {
    await context.resume();
  } catch (error) {
    for (const track of stream.getTracks()) track.stop();
    source.disconnect();
    processor.disconnect();
    silence.disconnect();
    await context.close();
    throw error;
  }

  const release = async (): Promise<void> => {
    if (!active) return;
    active = false;
    processor.onaudioprocess = null;
    for (const track of stream.getTracks()) track.stop();
    source.disconnect();
    processor.disconnect();
    silence.disconnect();
    await context.close();
  };

  return {
    sampleRate: context.sampleRate,
    async stop() {
      await release();
      const sampleCount = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
      if (sampleCount === 0) throw new Error("The recording was empty");
      const samples = new Float32Array(sampleCount);
      let offset = 0;
      for (const chunk of chunks) {
        samples.set(chunk, offset);
        offset += chunk.length;
      }
      return encodeGemmaMicrophoneWav(samples, context.sampleRate);
    },
    discard: release,
  };
}

export function encodeGemmaMicrophoneWav(
  samples: Float32Array,
  sampleRate: number,
): Blob {
  if (!Number.isInteger(sampleRate) || sampleRate < 1 || samples.length < 1) {
    throw new Error("Gemma microphone PCM geometry is invalid");
  }
  const bytes = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(bytes);
  writeAscii(view, 0, "RIFF");
  view.setUint32(4, bytes.byteLength - 8, true);
  writeAscii(view, 8, "WAVE");
  writeAscii(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeAscii(view, 36, "data");
  view.setUint32(40, samples.length * 2, true);
  for (let index = 0; index < samples.length; index += 1) {
    const value = Math.max(-1, Math.min(1, samples[index]));
    view.setInt16(44 + index * 2, value < 0 ? value * 0x8000 : value * 0x7fff, true);
  }
  return new Blob([bytes], { type: "audio/wav" });
}

function writeAscii(view: DataView, offset: number, value: string): void {
  for (let index = 0; index < value.length; index += 1) {
    view.setUint8(offset + index, value.charCodeAt(index));
  }
}