import test from "node:test";
import assert from "node:assert/strict";
import { createResampler, RATE } from "../audio.js";

const sine = (rate, freq, seconds, amp = 0.5) =>
  Float32Array.from({ length: Math.round(rate * seconds) }, (_, i) => amp * Math.sin((2 * Math.PI * freq * i) / rate));

// Pushes the input in mic-sized chunks, like the AudioWorklet does.
function run(rate, input, chunk = 2048) {
  const push = createResampler(rate);
  const out = [];
  for (let i = 0; i < input.length; i += chunk) out.push(...push(input.subarray(i, i + chunk)));
  return Float32Array.from(out);
}

// RMS of the middle of the signal, away from the filter's start-up.
function rms(x) {
  const a = Math.floor(x.length * 0.1), b = Math.floor(x.length * 0.9);
  let s = 0;
  for (let i = a; i < b; i++) s += x[i] * x[i];
  return Math.sqrt(s / (b - a));
}

for (const rate of [48000, 44100]) {
  test(`${rate} Hz: keeps speech frequencies and the right length`, () => {
    const out = run(rate, sine(rate, 1000, 1));
    // The filter looks ahead a few samples, so the last ~1 ms waits for the next chunk.
    assert.ok(RATE - out.length >= 0 && RATE - out.length <= 20, `length ${out.length}`);
    assert.ok(Math.abs(rms(out) - 0.5 / Math.SQRT2) < 0.01, `rms ${rms(out)}`);
  });

  test(`${rate} Hz: removes sound above 8 kHz instead of folding it into the speech band`, () => {
    for (const f of [9000, 12000, 15000]) {
      const out = run(rate, sine(rate, f, 1));
      assert.ok(rms(out) < 0.005, `${f} Hz leaked at rms ${rms(out)}`);
    }
  });
}

test("chunk size does not change the output", () => {
  const input = sine(48000, 440, 0.5);
  const a = run(48000, input, 128), b = run(48000, input, 4096);
  assert.equal(a.length, b.length);
  for (let i = 0; i < a.length; i++) assert.ok(Math.abs(a[i] - b[i]) < 1e-6);
});

test("16 kHz input passes through unchanged", () => {
  const input = sine(RATE, 440, 0.1);
  assert.deepEqual(run(RATE, input), input);
});
