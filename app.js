import { RATE, startTap, createPhraseSplitter, decodeFileTo16k } from "./audio.js";
import { SpeakerTracker, DEFAULT_SETTINGS, PALETTE, normalize, cosine, weightedMean } from "./speakers.js";

// ---------------------------------------------------------------------------
// Settings (some can be set from the URL, e.g. ?asr=tiny&enroll=30)
// ---------------------------------------------------------------------------
const params = new URLSearchParams(location.search);
const isMobile = matchMedia("(pointer: coarse)").matches;
const APP = {
  speakerModel: "onnx-community/wespeaker-voxceleb-resnet34-LM",
  asrModel: (params.get("asr") || (isMobile ? "tiny" : "base")) === "tiny" ? "Xenova/whisper-tiny.en" : "Xenova/whisper-base.en",
  enrollSeconds: Number(params.get("enroll")) || 20, // seconds of speech for a new profile
  minEmbedSeconds: 1.0,  // shorter phrases are transcribed but not used to tell voices apart
  minSaveSeconds: 8,     // speech needed before a voice from a meeting can be saved
  liveQueueCap: 20,      // skip phrases if analysis falls this far behind live speech
  maxFileMinutes: 20,
  gpu: params.get("gpu") !== "0" // speech-to-text on WebGPU when available; ?gpu=0 forces the CPU
};
const S = { ...DEFAULT_SETTINGS }; // live-tunable thresholds
const CONSENT_VERSION = "v1";
const MIC = { echoCancellation: false, noiseSuppression: true, autoGainControl: true, channelCount: 1 };

const $ = (id) => document.getElementById(id);
const randomId = () => crypto.randomUUID().replace(/-/g, "").slice(0, 16);
const fmt = (x) => (typeof x === "number" ? x.toFixed(2) : "–");
const mmss = (s) => `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
const when = (t) => new Date(t).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
const el = (tag, props = {}, ...kids) => {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === "class") n.className = v;
    else if (k === "style") n.setAttribute("style", v);
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else if (v !== undefined && v !== null && v !== false) n.setAttribute(k, v === true ? "" : v);
  }
  for (const k of kids.flat()) if (k !== null && k !== undefined && k !== false) n.append(k);
  return n;
};
// An icon from the sprite in index.html.
const icon = (name, cls = "ic") => {
  const s = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  s.setAttribute("class", cls); s.setAttribute("aria-hidden", "true");
  const u = document.createElementNS("http://www.w3.org/2000/svg", "use");
  u.setAttribute("href", `#i-${name}`);
  s.append(u);
  return s;
};
// "Anna Keller" -> "AK", "Speaker 2" -> "S2"
const initials = (name) => {
  const n = /^Speaker (\d+)$/.exec(name || "");
  if (n) return `S${n[1]}`;
  const w = (name || "?").trim().split(/\s+/);
  return ((w[0][0] || "?") + (w.length > 1 ? w[w.length - 1][0] : "")).toUpperCase();
};
const local = {
  get(k) { try { return localStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch { /* storage blocked */ } }
};

function consentText(name, workspace) {
  return `I, ${name || "[name]"}, agree that ${workspace} stores a voice profile of me: a set of numbers calculated from my voice, not a recording. ` +
    `It is used only to show my name in ${workspace}'s meeting transcripts and is never shared with anyone else. ` +
    `I can delete it at any time, and it is then removed immediately.`;
}

// Whisper sometimes writes sound descriptions like "[BLANK_AUDIO]" or "(laughs)".
function cleanText(t) {
  t = (t || "").replace(/\[[^\]]*\]|\([^)]*\)|\*[^*]*\*|♪/g, " ").replace(/\s+/g, " ").trim();
  if (/^[.,!?\s-]*$/.test(t) || /^(you|thank you)\.?$/i.test(t)) return "";
  return t;
}

// ---------------------------------------------------------------------------
// Storage: IndexedDB, with an in-memory fallback if the browser blocks it
// ---------------------------------------------------------------------------
class Store {
  static open(name, stores) {
    return new Promise((resolve) => {
      let req;
      try { req = indexedDB.open(name, 1); } catch { resolve(new Store(null, stores)); return; }
      req.onupgradeneeded = () => {
        for (const s of stores) if (!req.result.objectStoreNames.contains(s)) req.result.createObjectStore(s, { keyPath: "id" });
      };
      req.onsuccess = () => resolve(new Store(req.result, stores));
      req.onerror = () => resolve(new Store(null, stores));
    });
  }
  constructor(db, stores) { this.db = db; this.mem = new Map(stores.map((s) => [s, new Map()])); }
  _req(store, mode, fn) {
    return new Promise((resolve, reject) => {
      const t = this.db.transaction(store, mode);
      const r = fn(t.objectStore(store));
      t.oncomplete = () => resolve(r ? r.result : undefined);
      t.onerror = () => reject(t.error);
      t.onabort = () => reject(t.error);
    });
  }
  all(s) { return this.db ? this._req(s, "readonly", (o) => o.getAll()) : Promise.resolve([...this.mem.get(s).values()]); }
  get(s, id) { return this.db ? this._req(s, "readonly", (o) => o.get(id)) : Promise.resolve(this.mem.get(s).get(id)); }
  put(s, v) { if (this.db) return this._req(s, "readwrite", (o) => o.put(v)); this.mem.get(s).set(v.id, v); return Promise.resolve(); }
  del(s, id) { if (this.db) return this._req(s, "readwrite", (o) => o.delete(id)); this.mem.get(s).delete(id); return Promise.resolve(); }
  close() { try { this.db?.close(); } catch { /* ignore */ } }
}
function dropDatabase(name) {
  return new Promise((resolve) => {
    try {
      const r = indexedDB.deleteDatabase(name);
      r.onsuccess = r.onerror = r.onblocked = () => resolve();
    } catch { resolve(); }
  });
}

// Voiceprints are encrypted with the workspace's own key before they are stored.
async function encryptVec(key, vec) {
  const iv = crypto.getRandomValues(new Uint8Array(12));
  const ct = await crypto.subtle.encrypt({ name: "AES-GCM", iv }, key, Float32Array.from(vec).buffer);
  return { iv, ct };
}
async function decryptVec(key, enc) {
  const pt = await crypto.subtle.decrypt({ name: "AES-GCM", iv: enc.iv }, key, enc.ct);
  return new Float32Array(pt);
}

// ---------------------------------------------------------------------------
// App state
// ---------------------------------------------------------------------------
const state = {
  meta: null, workspaces: [], ws: null, // ws = { id, name, store, key }
  profiles: [],                          // decrypted, in memory: { id, name, embedding, ... }
  stale: [],                             // profiles made with an older voice model: listed, not used
  audit: [],
  enroll: null,
  persistent: true
};
let meeting = newMeeting();

function newMeeting() {
  const m = {
    id: randomId(), lines: [], lastClusterId: null,
    queue: 0, done: 0, skipped: 0, errors: 0, lastError: "",
    source: null, live: null, file: null, startedAt: null,
    seen: new Set() // lines and speakers already on screen, so only new ones animate in
  };
  m.tracker = new SpeakerTracker({
    settings: () => S,
    profiles: () => state.profiles,
    onMerge: (from, to) => {
      for (const l of m.lines) {
        if (l.clusterId === from) l.clusterId = to;
        if (l.closest?.id === from) l.closest = { ...l.closest, id: to };
      }
      if (m.lastClusterId === from) m.lastClusterId = to;
    }
  });
  return m;
}

const busy = () => !!(meeting.live || meeting.file?.running || state.enroll?.recording);

// ---------------------------------------------------------------------------
// Workspaces (one per customer)
// ---------------------------------------------------------------------------
async function initWorkspaces() {
  state.meta = await Store.open("vp-meta", ["workspaces"]);
  state.persistent = !!state.meta.db;
  state.workspaces = (await state.meta.all("workspaces")).sort((a, b) => a.createdAt - b.createdAt);
  if (!state.workspaces.length) await createWorkspace("Demo Company", false);
  const want = local.get("vp-current");
  await openWorkspace(state.workspaces.find((w) => w.id === want) ? want : state.workspaces[0].id);
  if (!state.persistent) {
    $("notice").append(el("span", { class: "badge-pill" }, "Storage blocked: voices are kept until you close this tab"));
  }
}

async function createWorkspace(name, open = true) {
  const w = { id: randomId(), name, createdAt: Date.now() };
  await state.meta.put("workspaces", w);
  state.workspaces.push(w);
  if (open) await openWorkspace(w.id);
  return w;
}

async function openWorkspace(id) {
  if (busy()) return;
  state.ws?.store.close();
  const w = state.workspaces.find((x) => x.id === id);
  const store = await Store.open(`vp-ws-${id}`, ["profiles", "audit", "keys"]);
  let k = await store.get("keys", "main");
  if (!k) {
    // non-extractable: the key can be used here but never read out of the browser
    const key = await crypto.subtle.generateKey({ name: "AES-GCM", length: 256 }, false, ["encrypt", "decrypt"]);
    k = { id: "main", key, createdAt: Date.now() };
    await store.put("keys", k);
  }
  state.ws = { id, name: w.name, store, key: k.key };
  local.set("vp-current", id);
  closeEnroll();
  await loadProfiles();
  meeting = newMeeting();
  renderAll();
}

async function deleteWorkspace() {
  if (busy()) return;
  const w = state.ws;
  if (!confirm(`Delete workspace "${w.name}" with all its voice profiles and its encryption key? This can't be undone.`)) return;
  w.store.close();
  await dropDatabase(`vp-ws-${w.id}`);
  await state.meta.del("workspaces", w.id);
  state.workspaces = state.workspaces.filter((x) => x.id !== w.id);
  state.ws = null;
  if (!state.workspaces.length) await createWorkspace("Demo Company", false);
  await openWorkspace(state.workspaces[0].id);
}

// ---------------------------------------------------------------------------
// Profiles and audit log
// ---------------------------------------------------------------------------
async function loadProfiles() {
  const { store, key } = state.ws;
  const recs = await store.all("profiles");
  const out = [];
  for (const r of recs) {
    try { out.push({ ...r, enc: undefined, embedding: normalize(await decryptVec(key, r.enc)) }); }
    catch (e) { console.warn("Could not decrypt profile", r.id, e); }
  }
  // Voiceprints from different models can't be compared, so older profiles are listed but not used.
  const byName = (a, b) => a.name.localeCompare(b.name);
  state.profiles = out.filter((p) => p.model === APP.speakerModel).sort(byName);
  state.stale = out.filter((p) => p.model !== APP.speakerModel).sort(byName);
  state.audit = (await store.all("audit")).sort((a, b) => b.at - a.at);
}

async function logAudit(action, name) {
  await state.ws.store.put("audit", { id: randomId(), at: Date.now(), action, name });
}

async function saveProfile({ id, name, embedding, seconds, consistency, source, consent, createdAt }) {
  const rec = {
    id, name, enc: await encryptVec(state.ws.key, embedding),
    seconds: Math.round(seconds), consistency: consistency == null ? null : Math.round(consistency * 1000) / 1000,
    source, consent, model: APP.speakerModel, createdAt: createdAt || Date.now(), updatedAt: Date.now()
  };
  await state.ws.store.put("profiles", rec);
}

function consentRecord(name, method) {
  return { version: CONSENT_VERSION, text: consentText(name, state.ws.name), signedName: name, method, at: Date.now() };
}

async function deleteProfile(p) {
  if (!confirm(`Delete ${p.name}'s voice profile? It is removed immediately and can't be recovered.`)) return;
  await state.ws.store.del("profiles", p.id);
  await logAudit("Voice profile deleted at the person's request", p.name);
  meeting.tracker.forgetProfile(p.id);
  await loadProfiles();
  renderAll();
}

// Save a voice heard in a meeting (after consent), or add it to an existing profile with the same name.
async function saveFromCluster(c, name) {
  const same = (p) => p.name.trim().toLowerCase() === name.trim().toLowerCase();
  const existing = state.profiles.find(same);
  const stale = !existing && state.stale.find(same);
  if (stale) { // made with an older voice model: replace it with this meeting's voice
    await saveProfile({
      ...stale, embedding: c.centroid, seconds: c.weight, consistency: null,
      source: "Learned in a meeting", consent: consentRecord(name, "confirmed in meeting (demo shortcut)")
    });
    await logAudit("Consent confirmed; voice profile replaced with a new recording from a meeting", name);
    c.profileId = stale.id;
  } else if (existing) {
    const embedding = weightedMean([{ e: existing.embedding, w: existing.seconds || 1 }, { e: c.centroid, w: c.weight }]);
    await saveProfile({
      ...existing, embedding, seconds: (existing.seconds || 0) + c.weight,
      consent: consentRecord(name, "confirmed in meeting (demo shortcut)")
    });
    await logAudit("Consent confirmed; voice from a meeting added to the profile", name);
    c.profileId = existing.id;
  } else {
    const id = randomId();
    await saveProfile({
      id, name, embedding: c.centroid, seconds: c.weight, consistency: null,
      source: "Learned in a meeting", consent: consentRecord(name, "confirmed in meeting (demo shortcut)")
    });
    await logAudit("Consent given; voice profile created from a meeting", name);
    c.profileId = id;
  }
  c.saved = true;
  await loadProfiles();
}

// ---------------------------------------------------------------------------
// Voice models (run in worker.js)
// ---------------------------------------------------------------------------
// Two workers so a phrase's voiceprint and its transcript are computed at the same time:
// "speaker" (WeSpeaker, CPU) and "asr" (Whisper, GPU when available).
const models = {
  workers: {}, ready: {}, nextId: 1, pending: new Map(), files: new Map(),
  hideTimer: null, broken: false, asrDevice: null
};

function worker(kind) {
  if (models.workers[kind]) return models.workers[kind];
  const w = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
  w.onmessage = (e) => {
    const m = e.data;
    if (m.type === "progress") { onModelProgress(m); return; }
    const p = models.pending.get(m.id);
    if (!p) return;
    models.pending.delete(m.id);
    if (m.type === "error") p.reject(new Error(m.message)); else p.resolve(m);
  };
  w.onerror = (e) => {
    showModel(`The voice models couldn't start (${e.message || "worker error"}). Please try a recent Chrome, Edge or Safari.`, null);
    for (const p of models.pending.values()) p.reject(new Error("Voice models not available"));
    models.pending.clear();
    models.broken = true;
  };
  models.workers[kind] = w;
  return w;
}

function callWorker(kind, type, payload = {}, transfer = []) {
  const id = models.nextId++;
  if (models.broken) return Promise.reject(new Error("Voice models not available"));
  return new Promise((resolve, reject) => {
    models.pending.set(id, { resolve, reject });
    worker(kind).postMessage({ type, id, ...payload }, transfer);
  });
}

function onModelProgress(m) {
  const f = models.files.get(m.file) || { loaded: 0, total: 0 };
  if (m.total) f.total = m.total;
  f.loaded = m.status === "done" ? (f.total || f.loaded) : (m.loaded || f.loaded);
  models.files.set(m.file, f);
  let loaded = 0, total = 0;
  for (const x of models.files.values()) { loaded += x.loaded; total += x.total; }
  if (total > 0 && loaded < total) renderDownload(loaded, total);
}

// ---------------------------------------------------------------------------
// First visit: a popup says "this is just a demo" (AI on this computer) vs the live version
// (rented GPUs, no download), with the download progress. It opens once, right away.
// After "Got it", progress continues in the corner toast, whose "Why?" reopens the popup.
// ---------------------------------------------------------------------------
const dl = { open: false, dismissed: local.get("vp-dl-seen") === "1", frac: 0, total: 0, isCached: null };
const mb = (b) => Math.round(b / 1e6);

// True if the weights of these models are already in the browser's cache (Transformers.js keeps
// them in Cache Storage under their Hugging Face URL), so loading them is not a real download.
async function modelsCached(kinds) {
  try {
    if (!("caches" in window)) return false;
    const urls = (await (await caches.open("transformers-cache")).keys()).map((r) => r.url);
    return kinds.map((k) => (k === "speaker" ? APP.speakerModel : APP.asrModel))
      .every((name) => urls.some((u) => u.includes(`/${name}/`) && u.endsWith(".onnx")));
  } catch { return false; }
}

function renderDownload(loaded, total) {
  // File sizes become known one by one as each file starts, so the total grows during the
  // download. Never let the percentage go backwards.
  dl.frac = Math.max(dl.frac, loaded / total); dl.total = total;
  const pct = `${Math.floor(dl.frac * 100)}%`;
  const real = dl.isCached === false;
  $("dl-dlg").classList.remove("ready", "failed");
  $("dl-progress").hidden = false;
  $("dl-pct").textContent = pct;
  $("dl-bar").style.width = pct;
  $("dl-sub").textContent = real ? `Downloading · ${mb(loaded)} MB` : "Loading…";
  if (!dl.open) showModel(`${real ? "Downloading" : "Loading"} the AI · ${pct}`, dl.frac, true);
}

function openDownload() {
  const d = $("dl-dlg");
  if (models.loading) $("dl-close").textContent = "Got it";
  hideToast(true);
  dl.open = true;
  if (!d.open) d.showModal();
}

function finishDownloadPopup() {
  $("dl-dlg").classList.add("ready");
  $("dl-progress").hidden = false;
  $("dl-pct").textContent = "100%";
  $("dl-bar").style.width = "100%";
  $("dl-sub").textContent = "Ready · saved for next time";
  $("dl-close").textContent = "Start";
}

function hideToast(now) {
  clearTimeout(models.hideTimer);
  const box = $("model-box");
  if (now) { box.hidden = true; box.classList.remove("leaving"); return; }
  box.classList.add("leaving");
  models.hideTimer = setTimeout(() => { box.hidden = true; box.classList.remove("leaving"); }, 300);
}

function showModel(text, fraction, why = false) {
  clearTimeout(models.hideTimer);
  $("model-box").classList.remove("leaving");
  $("model-box").hidden = false;
  $("model-text").textContent = text;
  $("model-why").hidden = !why;
  $("model-bar-wrap").hidden = fraction == null;
  $("model-bar-wrap").classList.toggle("ok", fraction === 1);
  if (fraction != null) $("model-bar").style.width = `${Math.round(fraction * 100)}%`;
}

function loadModel(kind) {
  if (!models.ready[kind]) {
    const payload = kind === "speaker" ? { speakerModel: APP.speakerModel } : { asrModel: APP.asrModel, gpu: APP.gpu };
    const p = callWorker(kind, "load", payload).then((r) => { if (kind === "asr") models.asrDevice = r.asrDevice; });
    p.catch(() => { if (models.ready[kind] === p) delete models.ready[kind]; });
    models.ready[kind] = p;
  }
  return models.ready[kind];
}

function ensureModels(withAsr) {
  const kinds = withAsr ? ["speaker", "asr"] : ["speaker"];
  const newKinds = kinds.filter((k) => !models.ready[k]);
  const fresh = newKinds.length > 0;
  if (fresh) { // checked before loading starts writing to the cache
    dl.isCached = null; dl.frac = 0;
    modelsCached(newKinds).then((c) => { dl.isCached = c; });
  }
  const all = Promise.all(kinds.map(loadModel));
  if (fresh) {
    models.loading = true;
    if (!dl.open) showModel("Preparing the AI… you can already start.", 0);
    all.then(() => {
      models.loading = false;
      const where = models.asrDevice === "webgpu" ? "GPU" : "CPU";
      const text = withAsr ? `Running on this computer · speech-to-text on the ${where}` : "Running on this computer";
      finishDownloadPopup(); // also when closed, so "Why?" later shows the finished state
      if (dl.open) { hideToast(true); return; }
      showModel(`Ready · ${text[0].toLowerCase()}${text.slice(1)}`, 1);
      models.hideTimer = setTimeout(() => hideToast(false), 3000);
    }, (e) => {
      console.error(e);
      models.loading = false;
      const msg = `Couldn't load the AI models: ${e.message}. Check your connection and reload the page.`;
      if (dl.open) {
        $("dl-dlg").classList.add("failed");
        $("dl-progress").hidden = false;
        $("dl-sub").textContent = msg;
        $("dl-close").textContent = "Close";
      } else showModel(msg, null);
    });
  }
  return all;
}

// ---------------------------------------------------------------------------
// Meeting: live recording or uploaded file
// ---------------------------------------------------------------------------
async function startLive() {
  if (busy()) return;
  if (meeting.lines.length && !confirm("Start a new meeting? The current transcript will be cleared.")) return;
  meeting = newMeeting();
  const m = meeting;
  let stream;
  try { stream = await navigator.mediaDevices.getUserMedia({ audio: MIC, video: false }); }
  catch { setStatus("Microphone access was blocked. Allow it in your browser's address bar and try again."); return; }
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  ctx.resume().catch(() => {});
  const splitter = createPhraseSplitter(ctx.sampleRate, {
    onUtterance: (a, t) => onMeetingPhrase(m, a, t, false),
    onActivity: (on) => { if (meeting === m) setSpeaking(on); }
  });
  let tap;
  try { tap = await startTap(ctx, stream, (s) => splitter.push(s)); }
  catch (e) {
    console.error(e);
    stream.getTracks().forEach((t) => t.stop()); ctx.close().catch(() => {});
    setStatus("This browser can't process live audio. Please use a recent Chrome, Edge or Safari.");
    return;
  }
  m.source = "live"; m.startedAt = Date.now();
  m.live = { stream, ctx, tap, splitter, timer: setInterval(renderStatus, 1000) };
  ensureModels(true).catch(() => {});
  renderMeeting();
}

function stopLive() {
  const l = meeting.live;
  if (!l) return;
  l.splitter.flush();
  l.tap.stop();
  l.stream.getTracks().forEach((t) => t.stop());
  l.ctx.close().catch(() => {});
  clearInterval(l.timer);
  meeting.live = null;
  meeting.endedAt = Date.now();
  setSpeaking(null);
  renderMeeting();
}

async function processFile(file) {
  if (busy() || !file) return;
  if (meeting.lines.length && !confirm("Analyze this file as a new meeting? The current transcript will be cleared.")) return;
  meeting = newMeeting();
  const m = meeting;
  m.source = "file";
  m.file = { name: file.name, running: true, total: 0 };
  renderMeeting();
  setStatus(`Reading "${file.name}"…`);
  ensureModels(true).catch(() => {});
  let decoded;
  try { decoded = await decodeFileTo16k(file); }
  catch (e) {
    console.error(e);
    m.file.running = false; renderMeeting();
    setStatus("Couldn't read this file. Try a common format such as .mp3, .m4a, .wav or .webm.");
    return;
  }
  if (decoded.duration > APP.maxFileMinutes * 60) {
    m.file.running = false; renderMeeting();
    setStatus(`This recording is ${Math.round(decoded.duration / 60)} minutes long. Please use one under ${APP.maxFileMinutes} minutes for this demo.`);
    return;
  }
  const phrases = [];
  const splitter = createPhraseSplitter(RATE, { onUtterance: (a, t) => phrases.push([a, t]) });
  for (let i = 0; i < decoded.samples.length; i += 4096) splitter.push(decoded.samples.subarray(i, i + 4096));
  splitter.flush();
  m.file.total = phrases.length;
  m.file.duration = decoded.duration;
  if (!phrases.length) {
    m.file.running = false; renderMeeting();
    setStatus("No speech found in this recording.");
    return;
  }
  renderMeeting();
  await Promise.all(phrases.map(([a, t]) => onMeetingPhrase(m, a, t, true)));
  m.file.running = false;
  renderMeeting();
}

async function onMeetingPhrase(m, audio, start, fromFile) {
  if (meeting !== m) return;
  const seconds = audio.length / RATE;
  if (!fromFile && m.queue >= APP.liveQueueCap) { m.skipped++; renderStatus(); return; }
  m.queue++;
  renderStatus();
  const fail = (e) => { console.error(e); m.errors++; m.lastError = e.message; };
  // The voiceprint and the transcript are computed in parallel by two workers. The line appears as soon as
  // the speaker is known (text "…" while transcribing). Every phrase goes through the speaker worker, even
  // short ones, so speakers are always assigned in the order the phrases were spoken.
  let line = null, text = null; // text: null while still transcribing, "" if there were no words
  const voiceAudio = audio.slice();
  const voice = callWorker("speaker", "process", { audio: voiceAudio, embed: seconds >= APP.minEmbedSeconds }, [voiceAudio.buffer])
    .then((res) => {
      if (meeting !== m) return;
      let cluster = null, score = null, short = false, closest = null;
      if (res.embedding) {
        const r = m.tracker.assign(Float32Array.from(res.embedding), seconds);
        cluster = r.cluster; score = r.score; closest = r.closest;
      } else {
        cluster = m.tracker.get(m.lastClusterId); // too short to tell: assume the last speaker continued
        short = true;
      }
      if (cluster) m.lastClusterId = cluster.id;
      if (text === "") return; // the transcript is already back and has no words
      line = { id: randomId(), clusterId: cluster ? cluster.id : null, text, start, end: start + seconds, score, closest, short, rating: null };
      m.lines.push(line);
      renderMeeting();
    })
    .catch(fail);
  const words = callWorker("asr", "process", { audio, transcribe: true }, [audio.buffer])
    .then((res) => { text = cleanText(res.text); })
    .catch((e) => { fail(e); text = ""; })
    .then(() => {
      if (meeting !== m || !line) return;
      if (text) { line.text = text; return; }
      const i = m.lines.indexOf(line);
      if (i >= 0) m.lines.splice(i, 1);
    })
    .catch(fail);
  try {
    await Promise.all([voice, words]);
  } finally {
    m.queue--; m.done++;
    if (meeting === m) renderMeeting();
  }
}

function clearMeeting() {
  if (busy()) return;
  if (meeting.lines.length && !confirm("Clear this meeting's transcript and speakers?")) return;
  meeting = newMeeting();
  renderMeeting();
}

function downloadTranscript() {
  const t = meeting.tracker;
  const lines = meeting.lines.filter((l) => l.text).sort((a, b) => a.start - b.start)
    .map((l) => `[${mmss(l.start)}] ${t.displayName(t.get(l.clusterId))}: ${l.text}`);
  const head = `Meeting transcript (${state.ws.name}), ${when(meeting.startedAt || Date.now())}\n\n`;
  const url = URL.createObjectURL(new Blob([head + lines.join("\n") + "\n"], { type: "text/plain" }));
  const a = el("a", { href: url, download: `transcript-${new Date().toISOString().slice(0, 16).replace(/[:T]/g, "-")}.txt` });
  document.body.append(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}

// ---------------------------------------------------------------------------
// Rename a speaker (with consent if the voice should be remembered)
// ---------------------------------------------------------------------------
let renaming = null;

function openRename(c) {
  renaming = c.id;
  const t = meeting.tracker;
  $("rename-sub").textContent = `${t.displayName(c)} · ${Math.round(c.weight)} s in this meeting`;
  $("rename-name").value = c.name || (c.profileId ? t.displayName(c) : "");
  $("rename-error").textContent = "";
  $("rename-consent").checked = false;
  const canSave = c.weight >= APP.minSaveSeconds;
  const remember = $("keep-remember");
  remember.disabled = !canSave;
  $("remember-label").classList.toggle("disabled", !canSave);
  $("remember-sub").textContent = canSave
    ? "Needs their consent."
    : `Needs ${APP.minSaveSeconds} s of speech (${Math.round(c.weight)} s so far).`;
  document.querySelector('input[name="keep"][value="meeting"]').checked = true;
  updateRenameConsent();
  $("rename-dlg").showModal();
  $("rename-name").focus();
}

function updateRenameConsent() {
  const remember = $("keep-remember").checked;
  const name = $("rename-name").value.trim();
  $("rename-consent-box").hidden = !remember;
  $("rename-consent-who").textContent = name || "This person";
  $("rename-consent-text").textContent = consentText(name, state.ws.name);
  $("rename-consent-label").textContent = `${name || "This person"} has read this and agrees.`;
}

async function submitRename(e) {
  e.preventDefault();
  const c = meeting.tracker.get(renaming);
  const name = $("rename-name").value.trim().replace(/\s+/g, " ");
  const remember = $("keep-remember").checked;
  if (!c) { $("rename-dlg").close(); return; }
  if (!name) { $("rename-error").textContent = "Enter a name."; return; }
  if (remember && !$("rename-consent").checked) { $("rename-error").textContent = `${name} has to agree before their voice can be stored.`; return; }
  $("rename-save").disabled = true;
  try {
    const kept = meeting.tracker.rename(c, name);
    if (remember) await saveFromCluster(kept, name);
    $("rename-dlg").close();
    renderAll();
  } catch (err) {
    console.error(err);
    $("rename-error").textContent = `Couldn't save: ${err.message}`;
  } finally {
    $("rename-save").disabled = false;
  }
}

// ---------------------------------------------------------------------------
// Enrollment: consent first, then about 20 seconds of speech
// ---------------------------------------------------------------------------
function openEnroll() {
  if (busy()) return;
  $("enroll-card").hidden = false;
  $("enroll-consent").hidden = false;
  $("enroll-record").hidden = true;
  $("enroll-done").hidden = true;
  $("enroll-name").value = "";
  $("enroll-agree").checked = false;
  $("enroll-error").textContent = "";
  updateEnrollConsent();
  $("enroll-card").scrollIntoView({ behavior: "smooth", block: "nearest" });
  $("enroll-name").focus();
}

function updateEnrollConsent() {
  $("enroll-consent-text").textContent = consentText($("enroll-name").value.trim(), state.ws.name);
}

async function startEnrollRecording() {
  const name = $("enroll-name").value.trim().replace(/\s+/g, " ");
  if (!name) { $("enroll-error").textContent = "Enter your name."; return; }
  if (!$("enroll-agree").checked) { $("enroll-error").textContent = "Please read the text and tick the box to agree."; return; }
  if (busy()) { $("enroll-error").textContent = "Stop the meeting recording first."; return; }
  let stream;
  try { stream = await navigator.mediaDevices.getUserMedia({ audio: MIC, video: false }); }
  catch { $("enroll-error").textContent = "Microphone access was blocked. Allow it in your browser and try again."; return; }
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  ctx.resume().catch(() => {});
  const en = { name, items: [], seconds: 0, pending: 0, recording: true, consentAt: Date.now(), stream, ctx, tap: null };
  state.enroll = en;
  const splitter = createPhraseSplitter(ctx.sampleRate, { onUtterance: (a) => onEnrollPhrase(en, a) });
  try { en.tap = await startTap(ctx, stream, (s) => splitter.push(s)); }
  catch (e) {
    console.error(e);
    stopEnrollAudio(en); state.enroll = null;
    $("enroll-error").textContent = "This browser can't process live audio. Please use a recent Chrome, Edge or Safari.";
    return;
  }
  $("enroll-consent").hidden = true;
  $("enroll-record").hidden = false;
  $("enroll-who").textContent = name;
  ensureModels(false).catch(() => {});
  renderEnroll();
  renderMeetingControls();
}

async function onEnrollPhrase(en, audio) {
  const seconds = audio.length / RATE;
  if (state.enroll !== en || !en.recording || seconds < APP.minEmbedSeconds) return;
  en.pending += seconds; renderEnroll();
  try {
    const { embedding } = await callWorker("speaker", "process", { audio, embed: true }, [audio.buffer]);
    if (state.enroll !== en || !en.recording) return;
    en.items.push({ e: Float32Array.from(embedding), w: seconds });
    en.seconds += seconds;
  } catch (e) {
    console.error(e);
    en.error = `Voice analysis error: ${e.message}`;
  } finally {
    en.pending -= seconds;
    if (state.enroll === en) renderEnroll();
  }
  if (state.enroll === en && en.recording && en.seconds >= APP.enrollSeconds) finishEnroll(en);
}

async function finishEnroll(en) {
  en.recording = false;
  stopEnrollAudio(en);
  const embedding = weightedMean(en.items);
  const total = en.items.reduce((s, it) => s + it.w, 0);
  const consistency = en.items.reduce((s, it) => s + cosine(it.e, embedding) * it.w, 0) / total;
  const same = (p) => p.name.trim().toLowerCase() === en.name.toLowerCase();
  const existing = state.profiles.find(same) || state.stale.find(same); // re-recording replaces an older-model profile
  const consent = { ...consentRecord(en.name, "given by the person themselves"), at: en.consentAt };
  try {
    if (existing) {
      await saveProfile({ ...existing, embedding, seconds: en.seconds, consistency, source: "Recorded before a meeting", consent });
      await logAudit("Consent given; voice profile re-recorded", en.name);
    } else {
      await saveProfile({ id: randomId(), name: en.name, embedding, seconds: en.seconds, consistency, source: "Recorded before a meeting", consent });
      await logAudit("Consent given; voice profile recorded", en.name);
    }
    await loadProfiles();
    // WeSpeaker: 20 s of clear speech from one person scored 0.89-0.94 in testing.
    const quality = consistency >= 0.85 ? "very clear" : consistency >= 0.75 ? "clear" : "noisy, try again somewhere quieter";
    $("enroll-result").textContent = `${en.name} · ${Math.round(en.seconds)} s · recording ${quality} (${fmt(consistency)})`;
  } catch (e) {
    console.error(e);
    $("enroll-result").textContent = `Couldn't save the profile: ${e.message}`;
  }
  state.enroll = null;
  $("enroll-record").hidden = true;
  $("enroll-done").hidden = false;
  renderAll();
}

function stopEnrollAudio(en) {
  en.tap?.stop();
  en.stream?.getTracks().forEach((t) => t.stop());
  en.ctx?.close().catch(() => {});
}

function closeEnroll() {
  const en = state.enroll;
  if (en) { en.recording = false; stopEnrollAudio(en); state.enroll = null; }
  $("enroll-card").hidden = true;
  renderMeetingControls();
}

function renderEnroll() {
  const en = state.enroll;
  if (!en) return;
  const frac = Math.min(1, en.seconds / APP.enrollSeconds);
  $("enroll-bar").style.width = `${Math.round(frac * 100)}%`;
  $("enroll-status").classList.toggle("error", !!en.error);
  $("enroll-status").textContent = en.error || `${Math.floor(en.seconds)} / ${APP.enrollSeconds} s`;
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------
function renderAll() {
  renderWorkspaces();
  renderProfiles();
  renderAudit();
  renderMeeting();
}

function renderWorkspaces() {
  const sel = $("ws-select");
  sel.replaceChildren(...state.workspaces.map((w) => el("option", { value: w.id, selected: w.id === state.ws?.id }, w.name)));
  sel.disabled = busy();
  $("ws-new").disabled = busy();
  $("ws-delete").disabled = busy();
  $("team-count").textContent = String(state.profiles.length);
}

function renderProfiles() {
  const ul = $("profiles");
  if (!state.profiles.length && !state.stale.length) {
    ul.replaceChildren(el("li", { class: "empty" }, "No voices yet."));
    return;
  }
  const rows = [...state.profiles.map((p) => [p, false]), ...state.stale.map((p) => [p, true])];
  ul.replaceChildren(...rows.map(([p, stale], i) => {
    const c = p.consent || {};
    const meta = [`${p.seconds || "?"} s`];
    if (p.consistency != null && !stale) meta.push(`quality ${fmt(p.consistency)}`);
    meta.push(when(p.updatedAt || p.createdAt));
    return el("li", {},
      el("div", { class: "p-head" },
        el("span", { class: "avatar", style: `background:${PALETTE[i % PALETTE.length]}` }, initials(p.name)),
        el("div", { class: "p-main" },
          el("strong", {}, p.name),
          el("small", { title: stale ? "Recorded with an older voice model, so it isn't used. Add this voice again with the same name to replace it." : p.source || "" },
            stale ? "Older model · not used" : meta.join(" · "))
        ),
        stale ? el("span", { class: "badge" }, "Record again") : el("span", { class: "badge ok" }, "Consent ✓"),
        el("button", { class: "icon danger", onclick: () => deleteProfile(p), title: "Delete", "aria-label": `Delete ${p.name}'s voice profile` }, icon("trash"))
      ),
      el("details", { class: "consent" },
        el("summary", {}, `Consent ${c.version || ""} · ${c.at ? when(c.at) : "unknown date"}`),
        el("p", { class: "hint", style: "margin-top:6px" }, c.method ? `Method: ${c.method}` : ""),
        el("blockquote", {}, c.text || "")
      )
    );
  }));
}

function renderAudit() {
  const ul = $("audit");
  if (!state.audit.length) { ul.replaceChildren(el("li", { class: "empty" }, "Nothing yet.")); return; }
  ul.replaceChildren(...state.audit.slice(0, 50).map((a) =>
    el("li", {}, el("time", {}, when(a.at)), `${a.action}: `, el("strong", {}, a.name))
  ));
}

function speakerStatus(c) {
  if (c.saved) return { text: "Saved with consent", cls: "ok" };
  if (c.name) return { text: "This meeting only", cls: "temp" };
  if (c.profileId) return { text: `Recognized · ${fmt(c.profileScore)}`, cls: "ok" };
  return { text: "New voice", cls: "" };
}

// Animate a list item only the first time it appears.
function enterClass(key) {
  if (meeting.seen.has(key)) return "";
  meeting.seen.add(key);
  return "enter";
}

function renderMeetingControls() {
  const live = !!meeting.live, fileRunning = !!meeting.file?.running, b = busy();
  $("start-btn").hidden = live;
  $("stop-btn").hidden = !live;
  $("start-btn").disabled = b;
  $("upload-input").disabled = b;
  $("upload-btn").disabled = b;
  $("clear-btn").disabled = b || (!meeting.lines.length && !meeting.tracker.clusters.length);
  $("download-btn").disabled = !meeting.lines.length;
  $("enroll-open").disabled = b;
  if (fileRunning || live) { $("ws-select").disabled = true; $("ws-new").disabled = true; $("ws-delete").disabled = true; }
  else renderWorkspaces();
}

function setStatus(text) { $("meeting-status").textContent = text; }

function setSpeaking(on) {
  const pill = $("speech-pill");
  pill.hidden = on === null;
  $("speech-text").textContent = on ? "Speech" : "Listening";
  pill.classList.toggle("speaking", !!on);
}

function renderStatus() {
  const m = meeting;
  const parts = [];
  if (m.live) {
    parts.push(`Recording · ${mmss((Date.now() - m.startedAt) / 1000)}`);
    if (m.queue > 0) parts.push(`analyzing ${m.queue} phrase${m.queue === 1 ? "" : "s"}`);
  } else if (m.file) {
    if (m.file.running && m.file.total) parts.push(`Analyzing "${m.file.name}": ${m.done} of ${m.file.total} phrases`);
    else if (!m.file.running && m.file.total) parts.push(`Finished "${m.file.name}" (${mmss(m.file.duration)})`);
    else if (m.file.running) return; // still decoding, status set elsewhere
  } else if (m.source === "live") {
    parts.push(m.queue > 0 ? `Recording stopped · finishing ${m.queue} phrase${m.queue === 1 ? "" : "s"}` : "Recording stopped");
  } else {
    setStatus(state.profiles.length
      ? `${state.profiles.length} saved voice${state.profiles.length === 1 ? "" : "s"} will be recognized`
      : "English · no saved voices yet");
    return;
  }
  if (m.skipped) parts.push(`${m.skipped} phrase${m.skipped === 1 ? "" : "s"} skipped because analysis fell behind`);
  if (m.errors) parts.push(`${m.errors} error${m.errors === 1 ? "" : "s"} (${m.lastError})`);
  setStatus(parts.join(" · "));
}

// "Try it in 3 steps": each step ticks off when it has been done.
function renderGuide() {
  const hidden = local.get("vp-guide") === "hidden";
  $("guide").hidden = hidden;
  $("guide-show").hidden = !hidden;
  if (hidden) return;
  if (meeting.lines.length) local.set("vp-met", "1");
  const done = [state.profiles.length > 0, local.get("vp-met") === "1", state.profiles.length > 1];
  done.forEach((d, i) => {
    const li = $(`step-${i + 1}`);
    li.classList.toggle("done", d);
    li.querySelector(".step-n").textContent = d ? "✓" : String(i + 1);
  });
  $("step-1-btn").hidden = done[0];
  $("step-2-btn").hidden = done[1] || !!meeting.live;
  $("guide-title").textContent = done.every(Boolean) ? "All three steps done" : "Try it in 3 steps";
}

function renderMeeting() {
  renderGuide();
  renderMeetingControls();
  renderStatus();
  const m = meeting, t = m.tracker;

  // speakers
  const clusters = [...t.clusters].sort((a, b) => b.weight - a.weight);
  const sp = $("speakers");
  if (!clusters.length) {
    sp.replaceChildren(el("li", { class: "empty" }, "Voices appear as people speak."));
  } else {
    sp.replaceChildren(...clusters.map((c) => {
      const st = speakerStatus(c);
      const minor = c.weight < 3 && !c.name && !c.profileId;
      const name = t.displayName(c);
      return el("li", { class: [minor ? "minor" : "", enterClass(`s${c.id}`)].join(" ").trim() },
        el("span", { class: "avatar", style: `background:${c.color}` }, initials(name)),
        el("div", { class: "sp-main" },
          el("strong", {}, name),
          el("small", { class: st.cls, title: minor ? "Very short: may be noise or one of the others" : "" },
            `${minor ? "Very short" : st.text} · ${Math.round(c.weight)} s`)
        ),
        el("button", { class: "icon", title: "Rename", "aria-label": `Rename ${name}`, onclick: () => openRename(c) }, icon("pencil"))
      );
    }));
  }
  const unnamed = clusters.filter((c) => !c.name && !c.profileId && c.weight >= 3).length;
  $("speakers-hint").textContent = unnamed ? "Rename to name someone. Same name twice merges them." : "";

  // transcript
  const ol = $("transcript");
  const atBottom = ol.scrollHeight - ol.scrollTop - ol.clientHeight < 60;
  const lines = [...m.lines].sort((a, b) => a.start - b.start);
  if (!lines.length) {
    const waiting = m.live || m.file?.running;
    ol.replaceChildren(el("li", { class: "empty-state" }, icon(waiting ? "wave" : "mic"),
      el("p", {}, waiting ? "Listening… lines appear a moment after each sentence." : "Start recording, or upload a recording.")));
  } else {
    ol.replaceChildren(...lines.map((l) => {
      const c = t.get(l.clusterId);
      const name = t.displayName(c);
      // Short score chip; the full explanation is in its tooltip.
      const [chip, why] = l.short ? ["short", "Too short to analyze: given to the previous speaker"]
        : l.score != null ? [fmt(l.score), `Voice similarity ${fmt(l.score)}`]
        : l.closest ? ["new", `New voice · closest was ${t.displayName(t.get(l.closest.id))} at ${fmt(l.closest.score)}`]
        : ["new", "First phrase from this voice"];
      const rate = (val, label, cls) => el("button", {
        class: cls, "aria-pressed": String(l.rating === val), title: label, "aria-label": label,
        onclick: () => { l.rating = l.rating === val ? null : val; renderMeeting(); }
      }, val ? "✓" : "✗");
      return el("li", { class: ["line", enterClass(l.id)].join(" ").trim() },
        el("span", { class: "avatar", style: `background:${c ? c.color : "#475569"}` }, initials(name)),
        el("div", {},
          el("div", { class: "line-head" },
            el("strong", {}, name),
            el("time", {}, mmss(l.start)),
            el("span", { class: "chip", title: why }, chip)
          ),
          l.text == null
            ? el("p", { class: "t-text pending", "aria-label": "Transcribing" }, "…")
            : el("p", { class: "t-text" }, l.text)
        ),
        el("div", { class: "rate" }, rate(true, "Right speaker", "yes"), rate(false, "Wrong speaker", "no"))
      );
    }));
    if (atBottom) ol.scrollTop = ol.scrollHeight;
  }

  // accuracy
  const rated = m.lines.filter((l) => l.rating !== null);
  const right = rated.filter((l) => l.rating).length;
  $("acc-big").textContent = rated.length ? `${Math.round((100 * right) / rated.length)}%` : "";
  $("acc-text").textContent = rated.length
    ? `right speaker · ${right} of ${rated.length} rated lines`
    : "Mark lines ✓ or ✗ to measure accuracy";
}

// ---------------------------------------------------------------------------
// Tuning sliders
// ---------------------------------------------------------------------------
const SLIDERS = [["s-profile", "o-profile", "profileThreshold", 2], ["s-join", "o-join", "joinThreshold", 2], ["s-merge", "o-merge", "mergeThreshold", 2], ["s-max", "o-max", "maxSpeakers", 0]];
function renderSliders() {
  for (const [sid, oid, key, dp] of SLIDERS) { $(sid).value = S[key]; $(oid).textContent = Number(S[key]).toFixed(dp); }
}
for (const [sid, oid, key, dp] of SLIDERS) {
  $(sid).addEventListener("input", () => { S[key] = Number($(sid).value); $(oid).textContent = S[key].toFixed(dp); });
}
$("tune-reset").addEventListener("click", () => { Object.assign(S, DEFAULT_SETTINGS); renderSliders(); });

// ---------------------------------------------------------------------------
// Wire up
// ---------------------------------------------------------------------------
function selectTab(name) {
  for (const n of ["meeting", "team", "how"]) {
    $(`tab-${n}`).setAttribute("aria-selected", String(n === name));
    $(`panel-${n}`).hidden = n !== name;
  }
}
for (const n of ["meeting", "team", "how"]) $(`tab-${n}`).addEventListener("click", () => selectTab(n));

$("dl-close").addEventListener("click", () => $("dl-dlg").close());
$("dl-x").addEventListener("click", () => $("dl-dlg").close());
$("dl-dlg").addEventListener("click", (e) => { if (e.target === $("dl-dlg")) $("dl-dlg").close(); }); // click outside
$("dl-dlg").addEventListener("close", () => {
  dl.open = false;
  if (!dl.dismissed) { dl.dismissed = true; local.set("vp-dl-seen", "1"); }
  if (models.loading) showModel(`Downloading the AI · ${Math.floor(dl.frac * 100)}%`, dl.frac, true);
});
$("model-why").addEventListener("click", openDownload);
$("step-1-btn").addEventListener("click", () => { selectTab("team"); openEnroll(); });
$("step-2-btn").addEventListener("click", () => { selectTab("meeting"); startLive(); });
$("guide-hide").addEventListener("click", () => { local.set("vp-guide", "hidden"); renderGuide(); });
$("guide-show").addEventListener("click", () => { local.set("vp-guide", ""); renderGuide(); window.scrollTo({ top: 0, behavior: "smooth" }); });
$("notice-more").addEventListener("click", () => { selectTab("how"); $("compare").scrollIntoView({ behavior: "smooth", block: "start" }); });

$("start-btn").addEventListener("click", startLive);
$("stop-btn").addEventListener("click", stopLive);
$("upload-input").addEventListener("change", (e) => { const f = e.target.files[0]; e.target.value = ""; processFile(f); });
$("clear-btn").addEventListener("click", clearMeeting);
$("download-btn").addEventListener("click", downloadTranscript);

$("enroll-open").addEventListener("click", openEnroll);
$("enroll-name").addEventListener("input", updateEnrollConsent);
$("enroll-start").addEventListener("click", startEnrollRecording);
$("enroll-cancel-1").addEventListener("click", closeEnroll);
$("enroll-cancel-2").addEventListener("click", () => { closeEnroll(); });
$("enroll-close").addEventListener("click", () => { closeEnroll(); selectTab("meeting"); });

$("rename-form").addEventListener("submit", submitRename);
$("rename-cancel").addEventListener("click", () => $("rename-dlg").close());
$("rename-name").addEventListener("input", updateRenameConsent);
for (const r of document.querySelectorAll('input[name="keep"]')) r.addEventListener("change", updateRenameConsent);

$("ws-select").addEventListener("change", (e) => openWorkspace(e.target.value));
$("ws-new").addEventListener("click", async () => {
  const name = (prompt("Name of the new workspace (one per customer):", "") || "").trim();
  if (name) await createWorkspace(name.slice(0, 40));
});
$("ws-delete").addEventListener("click", deleteWorkspace);

window.addEventListener("beforeunload", (e) => {
  if (meeting.live || state.enroll?.recording || meeting.file?.running) { e.preventDefault(); e.returnValue = ""; }
});

renderSliders();
if (!dl.dismissed) openDownload(); // first visit: say up front that this is a demo
initWorkspaces().then(() => {
  // On a computer, start downloading the models right away so the first click doesn't wait.
  // On phones, wait until they're needed (mobile data).
  if (!isMobile) ensureModels(true).catch(() => {});
}, (e) => {
  console.error(e);
  setStatus(`Couldn't start: ${e.message}`);
});
