import { test } from "node:test";
import assert from "node:assert/strict";
import { SpeakerTracker, DEFAULT_SETTINGS, normalize, cosine } from "../speakers.js";

// Synthetic voices on the WeSpeaker scale (256 numbers): each speaker is a base vector sharing a common
// component (different speakers ~0.35 similar), each phrase is the base plus noise
// (noise 0.02 puts a phrase ~0.9 from its speaker's base, like clear test speech).
function rng(seed) { return () => (seed = (seed * 16807) % 2147483647) / 2147483647; }
function makeWorld(seed, nSpeakers, noise) {
  const rnd = rng(seed);
  const gauss = () => Math.sqrt(-2 * Math.log(rnd() + 1e-12)) * Math.cos(2 * Math.PI * rnd());
  const vec = (s = 1) => Float32Array.from({ length: 256 }, () => gauss() * s);
  const add = (a, b, k = 1) => a.map((x, i) => x + k * b[i]);
  const common = normalize(vec());
  const bases = Array.from({ length: nSpeakers }, () => normalize(add(common, normalize(vec()), 1.35)));
  return { rnd, bases, phrase: (who, s = noise) => normalize(add(bases[who], vec(s))) };
}

function run(world, profiles, n = 90) {
  const t = new SpeakerTracker({ settings: () => DEFAULT_SETTINGS, profiles: () => profiles });
  const said = [];
  const remap = {};
  t.onMerge = (from, to) => { remap[from] = to; };
  for (let k = 0; k < n; k++) {
    const who = Math.floor(world.rnd() * world.bases.length);
    said.push({ who, id: t.assign(world.phrase(who), 1 + world.rnd() * 5).cluster.id });
  }
  const final = (id) => { while (remap[id]) id = remap[id]; return id; };
  return { t, said: said.map((s) => ({ ...s, id: final(s.id) })) };
}

test("recognizes enrolled speakers and keeps an unknown one separate", () => {
  const w = makeWorld(7, 3, 0.016);
  const profiles = [
    { id: "pA", name: "Ani", embedding: w.phrase(0, 0.004) },
    { id: "pB", name: "Bob", embedding: w.phrase(1, 0.004) }
  ];
  const { t, said } = run(w, profiles);
  assert.equal(t.clusters.length, 3);
  for (const [who, name] of [[0, "Ani"], [1, "Bob"]]) {
    const ids = new Set(said.filter((s) => s.who === who).map((s) => s.id));
    assert.equal(ids.size, 1);
    assert.equal(t.displayName(t.get([...ids][0])), name);
  }
  const unknown = new Set(said.filter((s) => s.who === 2).map((s) => s.id));
  assert.equal(unknown.size, 1);
  assert.match(t.displayName(t.get([...unknown][0])), /^Speaker \d+$/);
});

test("discovers speakers with no profiles, without fragmenting", () => {
  const w = makeWorld(11, 4, 0.02);
  const { t } = run(w, [], 100);
  assert.equal(t.clusters.length, 4);
});

test("one person talking stays one speaker, even with noisy voiceprints", () => {
  // noise 0.04: phrase vs phrase ~0.7, the low end measured for WeSpeaker in a simulated room
  for (const seed of [3, 13, 23, 33]) {
    const w = makeWorld(seed, 1, 0.04);
    const { t } = run(w, [], 60);
    assert.equal(t.clusters.length, 1, `seed ${seed}: ${t.clusters.length} speakers`);
  }
});

test("never creates more than maxSpeakers", () => {
  const w = makeWorld(5, 3, 0.03); // very noisy
  const { t } = run(w, [], 90);
  assert.ok(t.clusters.length <= DEFAULT_SETTINGS.maxSpeakers);
});

test("renaming two speakers to the same name merges them", () => {
  const t = new SpeakerTracker({ settings: () => ({ ...DEFAULT_SETTINGS, youngBonus: 0 }), profiles: () => [] });
  const a = t.assign(Float32Array.from([1, 0, 0]), 3).cluster;
  const b = t.assign(Float32Array.from([0, 1, 0]), 3).cluster;
  t.rename(a, "Ani");
  t.rename(b, "ani");
  assert.equal(t.clusters.length, 1);
});

test("a correction blocks the wrong profile from relinking", () => {
  const profiles = [{ id: "p1", name: "Bob", embedding: Float32Array.from([0, 0, 1]) }];
  const t = new SpeakerTracker({ settings: () => DEFAULT_SETTINGS, profiles: () => profiles });
  const c = t.assign(Float32Array.from([0, 0, 1]), 3).cluster;
  assert.equal(t.displayName(c), "Bob");
  const kept = t.rename(c, "Carla");
  t.assign(Float32Array.from([0, 0, 1]), 3);
  assert.equal(t.displayName(kept), "Carla");
  assert.equal(kept.profileId, null);
});

test("deleting a profile unlinks it", () => {
  const profiles = [{ id: "p1", name: "Bob", embedding: Float32Array.from([0, 0, 1]) }];
  const t = new SpeakerTracker({ settings: () => DEFAULT_SETTINGS, profiles: () => profiles });
  const c = t.assign(Float32Array.from([0, 0, 1]), 3).cluster;
  profiles.length = 0;
  t.forgetProfile("p1");
  assert.match(t.displayName(c), /^Speaker \d+$/);
  assert.ok(cosine(c.centroid, Float32Array.from([0, 0, 1])) > 0.99);
});
