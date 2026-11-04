import { spawn } from "node:child_process";

const [label, command, ...args] = process.argv.slice(2);
if (!label || !command) throw new Error("Usage: heartbeat.mjs <label> <command> [args...]");
const started = Date.now();
let progress = null;
let lastEvent = "starting";
let pending = "";
const child = spawn(command, args, { stdio: ["ignore", "pipe", "pipe"] });
function receive(bytes) {
  process.stdout.write(bytes);
  pending += bytes.toString();
  const lines = pending.split(/\r?\n/);
  pending = lines.pop() ?? "";
  for (const line of lines) {
    if (!line.trim()) continue;
    lastEvent = line.slice(0, 180);
    const match = /^PROGRESS (\d+)\/(\d+) (.+)$/.exec(line);
    if (match) progress = { done: Number(match[1]), total: Number(match[2]), phase: match[3] };
  }
}
child.stdout.on("data", receive);
child.stderr.on("data", receive);
const timer = setInterval(() => {
  const elapsed = Math.floor((Date.now() - started) / 1000);
  const fraction = progress?.total ? Math.min(1, progress.done / progress.total) : null;
  const filled = fraction === null ? 0 : Math.floor(fraction * 20);
  const bar = fraction === null ? "[working: no denominator]" : `[${"#".repeat(filled)}${".".repeat(20 - filled)}] ${progress.done}/${progress.total}`;
  console.log(`HEARTBEAT ${label} ${bar} elapsed=${elapsed}s last=${lastEvent}`);
}, 15_000);
child.once("error", (error) => {
  clearInterval(timer);
  console.error(error);
  process.exitCode = 1;
});
child.once("exit", (code, signal) => {
  clearInterval(timer);
  console.log(`FINISHED ${label} status=${code ?? signal} elapsed=${Math.floor((Date.now() - started) / 1000)}s`);
  process.exitCode = code ?? 1;
});
for (const signal of ["SIGINT", "SIGTERM"]) {
  process.on(signal, () => child.kill(signal));
}
