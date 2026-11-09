// Who is speaking? Groups phrases into speakers as a meeting happens, and links
// each speaker to a saved team voice profile when the match is confident.
// Pure logic, no browser APIs, so it can be unit-tested.

export function normalize(v) {
  let n = 0;
  for (let i = 0; i < v.length; i++) n += v[i] * v[i];
  n = Math.sqrt(n) || 1;
  const out = new Float32Array(v.length);
  for (let i = 0; i < v.length; i++) out[i] = v[i] / n;
  return out;
}

// Cosine similarity of two normalized vectors.
export function cosine(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

// Average of normalized vectors weighted by seconds of speech, then normalized.
export function weightedMean(items) {
  const sum = new Float32Array(items[0].e.length);
  for (const { e, w } of items) for (let i = 0; i < e.length; i++) sum[i] += e[i] * w;
  return normalize(sum);
}

export const PALETTE = ["#2563eb", "#db2777", "#059669", "#d97706", "#7c3aed", "#0891b2", "#dc2626", "#65a30d"];

// Cosine similarities for WeSpeaker ResNet34-LM voiceprints. Chosen by a grid search on test voices
// (6 voices; clean, 48 kHz and simulated-room recordings): one person talking, two people taking turns,
// saved profiles, and a stranger who must stay unnamed. These values sit inside the range where all of
// those came out right. Measured on that set: same person, phrase vs their average ~0.85 (lowest 0.62
// for 1.5-2.5 s phrases); different people at most 0.58-0.63.
// Real voices on a real mic still need checking: every transcript line shows its score.
export const DEFAULT_SETTINGS = {
  profileThreshold: 0.65, // phrase or speaker vs saved profile: accept as that person (0.55 named strangers)
  joinThreshold: 0.60,    // phrase vs a speaker already in this meeting: same person (0.70 split people, 0.55 mixed two)
  mergeThreshold: 0.70,   // two speakers in this meeting: actually the same person (0.60 mixed two)
  margin: 0.10,           // best profile must beat the second best by this much
  // A speaker heard for only a few seconds has a rough voiceprint, so allow looser
  // matches while a speaker is new. The bonus fades as more speech arrives.
  youngBonus: 0.05,
  youngSeconds: 8,
  // A short phrase (1-2.5 s) gives a rough voiceprint: in testing its nearest speaker was always right,
  // but its score was often just under joinThreshold. So a short phrase needs less to join a speaker
  // (up to shortBonus less at 1 s, fading to 0 at shortSeconds) and rarely starts a new one.
  shortBonus: 0.15,
  shortSeconds: 2.5,
  maxSpeakers: 8          // never create more speakers than this; extra phrases go to the closest one
};

// Threshold adjusted for how much speech we have heard from a speaker.
function relaxed(base, seconds, S) {
  return base - S.youngBonus * Math.exp(-seconds / S.youngSeconds);
}

// Extra allowance for a short phrase (w seconds long).
function shortRelief(w, S) {
  if (!S.shortBonus || w >= S.shortSeconds) return 0;
  return S.shortBonus * Math.min(1, (S.shortSeconds - w) / (S.shortSeconds - 1));
}

export class SpeakerTracker {
  // settings(): current thresholds; profiles(): [{ id, name, embedding }]
  // onMerge(fromId, toId): called when two speakers turn out to be one person
  constructor({ settings, profiles, onMerge = () => {} }) {
    this.settings = settings;
    this.profiles = profiles;
    this.onMerge = onMerge;
    this.clusters = [];
    this.counter = 0;
  }

  get(id) { return this.clusters.find((c) => c.id === id) || null; }

  newCluster() {
    const n = ++this.counter;
    const c = {
      id: `s${n}`, label: `Speaker ${n}`, color: PALETTE[(n - 1) % PALETTE.length],
      sum: null, weight: 0, centroid: null,
      profileId: null, profileScore: null, // linked saved profile, if recognized
      name: null, saved: false,            // name set by the user; saved = stored as a profile
      blocked: new Set()                   // profiles the user said this is NOT
    };
    this.clusters.push(c);
    return c;
  }

  add(c, e, w) {
    if (!c.sum) c.sum = new Float32Array(e.length);
    for (let i = 0; i < e.length; i++) c.sum[i] += e[i] * w;
    c.weight += w;
    c.centroid = normalize(c.sum);
  }

  matchProfile(e, blocked = new Set()) {
    const list = this.profiles()
      .filter((p) => !blocked.has(p.id))
      .map((p) => ({ profile: p, score: cosine(e, p.embedding) }))
      .sort((a, b) => b.score - a.score);
    if (!list.length) return null;
    return { profile: list[0].profile, score: list[0].score, second: list[1] ? list[1].score : -1 };
  }

  accepted(m) {
    const S = this.settings();
    return !!m && m.score >= S.profileThreshold && m.score - m.second >= S.margin;
  }

  // Assign one phrase (normalized embedding e, w seconds long) to a speaker.
  // Returns { cluster, score, closest }: score is null when the phrase started a new speaker, and then
  // closest = { id, score } is the existing speaker it was nearest to (useful for tuning joinThreshold).
  assign(e, w) {
    const S = this.settings();
    let best = null, bestScore = -1, bestMargin = -Infinity;
    let nearest = null;
    for (const c of this.clusters) {
      if (!c.centroid) continue;
      const s = cosine(e, c.centroid);
      const m = s - (relaxed(S.joinThreshold, c.weight, S) - shortRelief(w, S)); // how far above its threshold
      if (m > bestMargin) { bestMargin = m; bestScore = s; best = c; }
      if (!nearest || s > nearest.score) nearest = { id: c.id, score: s };
    }
    let target, score, closest = null;
    if (best && bestMargin >= 0) {
      target = best; score = bestScore;
    } else {
      const m = this.matchProfile(e);
      if (this.accepted(m)) {
        target = this.clusters.find((c) => c.profileId === m.profile.id && !c.name) || this.newCluster();
        target.profileId = m.profile.id; target.profileScore = m.score;
        score = m.score;
      } else if (best && this.clusters.length >= S.maxSpeakers) {
        target = best; score = bestScore; // at the cap: closest speaker wins
      } else {
        target = this.newCluster();
        score = null;
        closest = nearest;
      }
    }
    this.add(target, e, w);
    target = this.relink(target);
    target = this.mergePass(target);
    return { cluster: target, score, closest };
  }

  // Re-check a speaker against the saved profiles now that we have more of their voice.
  relink(c) {
    if (c.name) return c; // the user named this speaker, so their word wins
    const S = this.settings();
    const m = this.matchProfile(c.centroid, c.blocked);
    if (this.accepted(m)) {
      const other = this.clusters.find((d) => d !== c && d.profileId === m.profile.id);
      if (other) return this.merge(other, c); // two speakers match the same person: they are one
      c.profileId = m.profile.id; c.profileScore = m.score;
    } else if (c.profileId) {
      const p = this.profiles().find((x) => x.id === c.profileId);
      const s = p ? cosine(c.centroid, p.embedding) : -1;
      if (s < S.profileThreshold - 0.05) { c.profileId = null; c.profileScore = null; }
      else c.profileScore = s;
    }
    return c;
  }

  // Merge speakers whose voices turned out to be the same. Returns where `track` ended up.
  mergePass(track) {
    const S = this.settings();
    let changed = true;
    while (changed) {
      changed = false;
      outer:
      for (let i = 0; i < this.clusters.length; i++) {
        for (let j = i + 1; j < this.clusters.length; j++) {
          const a = this.clusters[i], b = this.clusters[j];
          if (!a.centroid || !b.centroid) continue;
          if (a.profileId && b.profileId && a.profileId !== b.profileId) continue;
          if (a.name && b.name && a.name !== b.name) continue;
          if (cosine(a.centroid, b.centroid) < relaxed(S.mergeThreshold, Math.min(a.weight, b.weight), S)) continue;
          const bIdentified = b.name || b.profileId, aIdentified = a.name || a.profileId;
          const [keep, gone] = bIdentified && !aIdentified ? [b, a] : [a, b];
          this.merge(keep, gone);
          if (track === gone) track = keep;
          changed = true;
          break outer;
        }
      }
    }
    return track;
  }

  merge(keep, gone) {
    if (gone.sum) {
      if (!keep.sum) keep.sum = new Float32Array(gone.sum.length);
      for (let i = 0; i < gone.sum.length; i++) keep.sum[i] += gone.sum[i];
      keep.weight += gone.weight;
      keep.centroid = normalize(keep.sum);
    }
    keep.name = keep.name || gone.name;
    keep.saved = keep.saved || gone.saved;
    if (!keep.profileId && gone.profileId) { keep.profileId = gone.profileId; keep.profileScore = gone.profileScore; }
    for (const id of gone.blocked) keep.blocked.add(id);
    this.clusters = this.clusters.filter((c) => c !== gone);
    this.onMerge(gone.id, keep.id);
    return keep;
  }

  // The user typed a name for a speaker. If another speaker already has that name,
  // the user is telling us they are the same person, so merge them.
  // Returns the speaker that now carries the name.
  rename(c, name) {
    const same = (a, b) => a.trim().toLowerCase() === b.trim().toLowerCase();
    if (c.profileId) {
      const p = this.profiles().find((x) => x.id === c.profileId);
      if (!p || !same(p.name, name)) {
        c.blocked.add(c.profileId); // "this is not who you thought"
        c.profileId = null; c.profileScore = null; c.saved = false;
      }
    }
    c.name = name;
    const twin = this.clusters.find((d) => d !== c && same(this.displayName(d), name));
    if (twin) {
      const keep = twin.saved || twin.profileId ? twin : c;
      const gone = keep === twin ? c : twin;
      keep.name = name;
      return this.merge(keep, gone);
    }
    return c;
  }

  // A saved profile was deleted: forget any link to it.
  forgetProfile(profileId) {
    for (const c of this.clusters) {
      if (c.profileId === profileId) { c.profileId = null; c.profileScore = null; c.saved = false; }
      c.blocked.delete(profileId);
    }
  }

  displayName(c) {
    if (!c) return "Unknown";
    if (c.name) return c.name;
    if (c.profileId) {
      const p = this.profiles().find((x) => x.id === c.profileId);
      if (p) return p.name;
    }
    return c.label;
  }
}
