// Runs a voice model in a background thread so the page stays responsive.
// The app starts two of these workers so both models work at the same time:
// - Speaker model (WeSpeaker ResNet34-LM, the one pyannote uses): a phrase of speech -> 256-number voiceprint.
//   Full precision (27 MB), on the CPU (WebAssembly): about 0.1 s of work per second of speech.
//   It replaced WavLM Base+ (8-bit), whose voiceprints jumped on some phrases and split one person into several speakers.
// - Whisper (English): speech -> text. Runs on the GPU (WebGPU) when the browser supports it, otherwise on the CPU.
// Both are open source, downloaded once from the Hugging Face hub and cached by the browser.
import { pipeline, AutoProcessor, AutoModel, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2";

env.allowLocalModels = false;

let processor = null, speaker = null;
let asr = null, asrName = null, asrDevice = null;

function progress_callback(p) {
  if (!p || !p.file) return;
  if (p.status === "initiate" || p.status === "progress" || p.status === "done") {
    postMessage({ type: "progress", status: p.status, file: `${p.name}/${p.file}`, loaded: p.loaded || 0, total: p.total || 0 });
  }
}

function normalize(v) {
  let n = 0;
  for (let i = 0; i < v.length; i++) n += v[i] * v[i];
  n = Math.sqrt(n) || 1;
  const out = new Float32Array(v.length);
  for (let i = 0; i < v.length; i++) out[i] = v[i] / n;
  return out;
}

async function embed(audio) {
  const inputs = await processor(audio.slice());
  const out = await speaker(inputs);
  return normalize((out.embeddings ?? out.last_hidden_state).data);
}

// WebGPU with 16-bit floats (the weights we use on the GPU need them).
async function webgpuAvailable() {
  try {
    const adapter = await navigator.gpu?.requestAdapter();
    return !!adapter && adapter.features.has("shader-f16");
  } catch {
    return false;
  }
}

async function loadAsr(model, device) {
  const opts = device === "webgpu"
    ? { device: "webgpu", dtype: { encoder_model: "fp16", decoder_model_merged: "q4f16" } }
    : { device: "wasm", dtype: "q8" };
  const next = await pipeline("automatic-speech-recognition", model, { ...opts, progress_callback });
  const old = asr;
  asr = next; asrName = model; asrDevice = device;
  if (old) Promise.resolve(old.dispose?.()).catch(() => {});
}

async function transcribe(audio) {
  try {
    return (await asr(audio.slice())).text || "";
  } catch (err) {
    if (asrDevice !== "webgpu") throw err;
    // The GPU failed (driver problem, device lost...): switch to the CPU for the rest of the session.
    console.warn("WebGPU speech-to-text failed, switching to the CPU:", err);
    await loadAsr(asrName, "wasm");
    return (await asr(audio.slice())).text || "";
  }
}

async function handle(m) {
  switch (m.type) {
    case "load":
      if (m.speakerModel && !speaker) {
        processor = await AutoProcessor.from_pretrained(m.speakerModel, { progress_callback });
        speaker = await AutoModel.from_pretrained(m.speakerModel, { device: "wasm", dtype: "fp32", progress_callback });
      }
      if (m.asrModel && asrName !== m.asrModel) {
        let loaded = false;
        if (m.gpu && await webgpuAvailable()) {
          try { await loadAsr(m.asrModel, "webgpu"); loaded = true; }
          catch (err) { console.warn("WebGPU speech-to-text unavailable, using the CPU:", err); }
        }
        if (!loaded) await loadAsr(m.asrModel, "wasm");
      }
      return { asrDevice };
    case "process": {
      const embedding = m.embed ? Array.from(await embed(m.audio)) : null;
      const text = m.transcribe && asr ? await transcribe(m.audio) : "";
      return { embedding, text, asrDevice };
    }
    default:
      throw new Error(`Unknown task: ${m.type}`);
  }
}

// One task at a time, in the order received.
let chain = Promise.resolve();
self.onmessage = (e) => {
  const m = e.data;
  chain = chain.then(() => handle(m)).then(
    (result) => postMessage({ type: "result", id: m.id, ...result }),
    (err) => postMessage({ type: "error", id: m.id, message: String(err?.message || err) })
  );
};
