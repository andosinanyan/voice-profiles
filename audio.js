// Audio capture and phrase detection.
// Everything is converted to 16 kHz mono, the rate the voice models expect.

export const RATE = 16000;

const TAP_WORKLET = `
class Tap extends AudioWorkletProcessor {
  constructor() { super(); this.buf = new Float32Array(2048); this.n = 0; }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (ch) for (let i = 0; i < ch.length; i++) {
      this.buf[this.n++] = ch[i];
      if (this.n === this.buf.length) { this.port.postMessage(this.buf.slice(0)); this.n = 0; }
    }
    return true;
  }
}
registerProcessor("tap", Tap);`;

// Streams raw microphone samples to onSamples(Float32Array) at the context's sample rate.
export async function startTap(ctx, stream, onSamples) {
  const url = URL.createObjectURL(new Blob([TAP_WORKLET], { type: "application/javascript" }));
  try { await ctx.audioWorklet.addModule(url); } finally { URL.revokeObjectURL(url); }
  const src = ctx.createMediaStreamSource(stream);
  const node = new AudioWorkletNode(ctx, "tap");
  const sink = ctx.createGain();
  sink.gain.value = 0; // keeps the graph running without playing the mic back
  src.connect(node); node.connect(sink); sink.connect(ctx.destination);
  node.port.onmessage = (e) => onSamples(e.data);
  return {
    stop() {
      node.port.onmessage = null;
      try { src.disconnect(); node.disconnect(); sink.disconnect(); } catch { /* already closed */ }
    }
  };
}

// Streaming resampler to 16 kHz for any input rate (44.1, 48, 96 kHz...). Returns push(Float32Array) -> number[].
// Downsampling first removes everything above ~7 kHz with a low-pass filter (windowed sinc). Without it,
// sound above 8 kHz (hiss, "s" sounds, room noise) folds back into the speech band and blurs the voiceprint.
export function createResampler(inRate) {
  const ratio = inRate / RATE;
  if (ratio === 1) return (input) => input;
  const up = ratio < 1; // rare (mic below 16 kHz): plain interpolation, nothing to filter
  const fc = up ? 0 : (0.45 * RATE) / inRate; // cutoff 7.2 kHz, as cycles per input sample
  const half = up ? 0 : Math.ceil(16 * ratio); // filter half-length: 48 taps each side at 48 kHz
  const h = new Float32Array(2 * half + 1);
  if (up) h[0] = 1;
  else {
    let sum = 0;
    for (let i = -half; i <= half; i++) {
      const x = 2 * fc * i;
      const sinc = i === 0 ? 1 : Math.sin(Math.PI * x) / (Math.PI * x);
      const win = 0.42 + 0.5 * Math.cos((Math.PI * i) / half) + 0.08 * Math.cos((2 * Math.PI * i) / half); // Blackman
      h[i + half] = sinc * win; sum += sinc * win;
    }
    for (let i = 0; i < h.length; i++) h[i] /= sum;
  }
  let buf = new Float32Array(half); // unused input, starting with `half` samples of silence
  let pos = half;                   // where the next output sample falls, in buf's coordinates
  const at = (data, i) => { let s = 0; for (let k = 0; k < h.length; k++) s += h[k] * data[i - half + k]; return s; };
  return (input) => {
    const data = new Float32Array(buf.length + input.length);
    data.set(buf); data.set(input, buf.length);
    const out = [];
    while (Math.floor(pos) + 1 + half < data.length) {
      const i = Math.floor(pos), frac = pos - i;
      const a = at(data, i);
      out.push(frac ? a + (at(data, i + 1) - a) * frac : a);
      pos += ratio;
    }
    const drop = Math.max(0, Math.floor(pos) - half);
    buf = data.slice(drop); pos -= drop;
    return out;
  };
}

// Splits a stream of samples into phrases using a simple energy-based voice activity detector.
// onUtterance(audio16k: Float32Array, startSeconds) is called for each phrase;
// onActivity(isSpeaking) reports when speech starts and stops.
export function createPhraseSplitter(inRate, { onUtterance, onActivity = () => {}, minSeconds = 0.5, maxSeconds = 8 }) {
  const FRAME = 480; // 30 ms at 16 kHz
  const START_FRAMES = 3, END_FRAMES = 22, PRE_FRAMES = 10; // 90 ms to start, ~660 ms of silence to end
  const MAX_FRAMES = Math.round((maxSeconds * RATE) / FRAME);
  const resample = createResampler(inRate);
  let frame = new Float32Array(FRAME), fn = 0, frameCount = 0;
  let floor = 0.003, speechRun = 0, silenceRun = 0, active = false, startFrame = 0;
  let pre = [], chunks = [];

  function finish() {
    const ended = silenceRun >= END_FRAMES;
    const kept = ended ? chunks.slice(0, chunks.length - END_FRAMES + 6) : chunks;
    const wasCut = !ended;
    active = false; speechRun = 0; silenceRun = 0; chunks = [];
    if (!wasCut) onActivity(false);
    const audio = new Float32Array(kept.length * FRAME);
    kept.forEach((c, i) => audio.set(c, i * FRAME));
    if (audio.length / RATE >= minSeconds) onUtterance(audio, (startFrame * FRAME) / RATE);
    if (wasCut) { active = true; startFrame = frameCount; } // speech continues into the next piece
  }

  function onFrame(f) {
    frameCount++;
    let s = 0;
    for (let i = 0; i < f.length; i++) s += f[i] * f[i];
    const rms = Math.sqrt(s / f.length);
    const isSpeech = rms > Math.max(0.01, floor * 3);
    if (!active) {
      floor = rms < floor ? floor * 0.9 + rms * 0.1 : floor * 0.995 + rms * 0.005;
      pre.push(f);
      if (pre.length > PRE_FRAMES) pre.shift();
      speechRun = isSpeech ? speechRun + 1 : 0;
      if (speechRun >= START_FRAMES) {
        active = true; silenceRun = 0;
        chunks = pre; pre = [];
        startFrame = frameCount - chunks.length;
        onActivity(true);
      }
    } else {
      chunks.push(f);
      silenceRun = isSpeech ? 0 : silenceRun + 1;
      if (silenceRun >= END_FRAMES || chunks.length >= MAX_FRAMES) finish();
    }
  }

  return {
    push(input) {
      for (const v of resample(input)) {
        frame[fn++] = v;
        if (fn === FRAME) { onFrame(frame); frame = new Float32Array(FRAME); fn = 0; }
      }
    },
    flush() {
      if (active && chunks.length) { silenceRun = END_FRAMES; finish(); }
    }
  };
}

// Decodes an audio or video file to 16 kHz mono, peak-normalized.
export async function decodeFileTo16k(file) {
  const bytes = await file.arrayBuffer();
  const Ctx = window.AudioContext || window.webkitAudioContext;
  const ctx = new Ctx();
  let decoded;
  try { decoded = await ctx.decodeAudioData(bytes); } finally { ctx.close().catch(() => {}); }
  const off = new OfflineAudioContext(1, Math.max(1, Math.ceil(decoded.duration * RATE)), RATE);
  const src = off.createBufferSource();
  src.buffer = decoded;
  src.connect(off.destination);
  src.start();
  const rendered = await off.startRendering();
  const data = rendered.getChannelData(0);
  let peak = 0;
  for (let i = 0; i < data.length; i++) { const a = Math.abs(data[i]); if (a > peak) peak = a; }
  if (peak > 0 && peak < 0.9) { const g = 0.9 / peak; for (let i = 0; i < data.length; i++) data[i] *= g; }
  return { samples: data, duration: decoded.duration };
}
