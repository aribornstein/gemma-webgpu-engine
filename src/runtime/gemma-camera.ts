export interface GemmaCameraCapture {
  readonly stream: MediaStream;
  stop(): Promise<Blob>;
  discard(): Promise<void>;
}

export function startGemmaCameraCapture(stream: MediaStream): GemmaCameraCapture {
  if (stream.getVideoTracks().length === 0) {
    for (const track of stream.getTracks()) track.stop();
    throw new Error("Camera stream has no video track");
  }
  if (typeof MediaRecorder === "undefined") {
    for (const track of stream.getTracks()) track.stop();
    throw new Error("Video recording is not supported in this browser");
  }
  const mimeType = supportedVideoMimeType();
  const recorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);
  const chunks: Blob[] = [];
  let active = true;
  recorder.addEventListener("dataavailable", ({ data }) => {
    if (data.size > 0) chunks.push(data);
  });
  recorder.start();

  const releaseTracks = () => {
    for (const track of stream.getTracks()) track.stop();
  };

  return {
    stream,
    stop() {
      if (!active) return Promise.reject(new Error("Camera recording is not active"));
      active = false;
      return new Promise<Blob>((resolve, reject) => {
        recorder.addEventListener("stop", () => {
          releaseTracks();
          const recording = new Blob(chunks, { type: recorder.mimeType || "video/webm" });
          if (recording.size === 0) reject(new Error("The video recording was empty"));
          else resolve(recording);
        }, { once: true });
        recorder.addEventListener("error", (event) => {
          releaseTracks();
          reject(event.error ?? new Error("Video recording failed"));
        }, { once: true });
        recorder.stop();
      });
    },
    async discard() {
      if (!active) return;
      active = false;
      if (recorder.state !== "inactive") recorder.stop();
      releaseTracks();
    },
  };
}

function supportedVideoMimeType(): string | undefined {
  return [
    "video/mp4;codecs=avc1.42E01E",
    "video/mp4",
    "video/webm;codecs=vp8",
    "video/webm;codecs=vp9",
    "video/webm",
  ].find((type) => MediaRecorder.isTypeSupported(type));
}
