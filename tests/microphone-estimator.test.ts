import { expect, test } from "bun:test";
import { readFileSync } from "node:fs";

interface Pose {
  weights: Float64Array;
  active: boolean;
  label: string;
  voicing: number;
  f1: number;
  f2: number;
}
interface Estimator {
  update(s: Float32Array, rate: number, dt: number): Pose;
  reset(): void;
  setVowelReference(index: number, f1: number, f2: number): boolean;
  clearReferences(): void;
}
const source = readFileSync(new URL("../src/microphone/speech-estimator.js", import.meta.url), "utf8");
const estimator: () => Estimator = new Function(source + "\nreturn createSpeechEstimator")();

function resonate(samples: Float64Array, rate: number, hz: number, bandwidth: number) {
  const radius = Math.exp(-Math.PI * bandwidth / rate);
  const a = 2 * radius * Math.cos(2 * Math.PI * hz / rate);
  let y1 = 0, y2 = 0;
  for (let i = 0; i < samples.length; i++) {
    const value = samples[i] + a * y1 - radius * radius * y2;
    samples[i] = value;
    y2 = y1;
    y1 = value;
  }
}

function latestFrame(signal: Float64Array, rate: number) {
  const frame = signal.subarray(-Math.round(rate * .048));
  const rms = Math.sqrt(frame.reduce((sum, x) => sum + x * x, 0) / frame.length);
  return Float32Array.from(frame, x => .055 * x / rms);
}

function vowel(rate: number, fundamental: number, f1: number, f2: number) {
  const signal = new Float64Array(Math.round(rate * .18));
  let phase = 0, glottal = 0;
  for (let i = 0; i < signal.length; i++) {
    phase += fundamental / rate;
    glottal *= Math.exp(-2 * Math.PI * 900 / rate);
    if (phase >= 1) { glottal++; phase--; }
    signal[i] = glottal;
  }
  resonate(signal, rate, f1, 75);
  resonate(signal, rate, f2, 105);
  resonate(signal, rate, 2980, 160);
  return latestFrame(signal, rate);
}

function noise(rate: number, frequency: number, bandwidth: number) {
  let seed = 123;
  const signal = Float64Array.from({length: rate / 5}, () => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return seed / 2 ** 31 - 1;
  });
  resonate(signal, rate, frequency, bandwidth);
  resonate(signal, rate, frequency, bandwidth);
  return latestFrame(signal, rate);
}

function sustained(e: Estimator, samples: Float32Array, rate: number, frames = 10) {
  let pose = e.update(samples, rate, 1 / 60);
  for (let i = 1; i < frames; i++) pose = e.update(samples, rate, 1 / 60);
  return pose;
}

const vowels = [[750, 1260], [300, 2310], [330, 815], [565, 1900], [510, 990]];
for (const rate of [16000, 44100, 96000, 192000]) {
  test(`equal-level vowels yield five distinct places of articulation at ${rate} Hz`, () => {
    for (const pitch of [113, 193]) {
      for (const [index, [f1, f2]] of vowels.entries()) {
        const result = sustained(estimator(), vowel(rate, pitch, f1, f2), rate);
        expect(result.active).toBe(true);
        expect(result.weights.indexOf(Math.max(...result.weights))).toBe(index);
        expect(result.weights[index]).toBeGreaterThan(.42);
      }
    }
  });
}

test("front and postalveolar frication move different constrictions at the same level", () => {
  const s = sustained(estimator(), noise(48000, 6100, 1000), 48000);
  const sh = sustained(estimator(), noise(48000, 2800, 1100), 48000);
  expect(s.weights[8]).toBeGreaterThan(.45);
  expect(sh.weights[9]).toBeGreaterThan(.45);
  expect(s.weights[8]).toBeGreaterThan(s.weights[9]);
  expect(sh.weights[9]).toBeGreaterThan(sh.weights[8]);
  expect(Math.max(...sh.weights.slice(0, 5))).toBeLessThan(.3);
});

test("the recent 32 ms wins over stale audio and smooth movement finishes within speech timing", () => {
  const e = estimator(), rate = 48000;
  const aa = vowel(rate, 115, 760, 1250), ee = vowel(rate, 115, 310, 2320);
  const mixed = new Float32Array(rate * .12);
  for (let i = 0; i < mixed.length; i++) mixed[i] = aa[i % aa.length];
  mixed.set(ee.subarray(-rate * .032), mixed.length - rate * .032);
  sustained(e, aa, rate);
  const first = e.update(mixed, rate, 1 / 60);
  expect(first.weights[1]).toBeGreaterThan(.12);
  expect(first.weights[0]).toBeGreaterThan(.12);
  const shifted = sustained(e, mixed, rate, 3);
  expect(shifted.weights[1]).toBeGreaterThan(shifted.weights[0] + .2);
  const quiet = new Float32Array(4096);
  const settled = sustained(e, quiet, rate, 12);
  expect(settled.active).toBe(false);
  expect(Math.max(...settled.weights)).toBeLessThan(.035);
  sustained(e, ee, rate);
  e.reset();
  const reset = e.update(quiet, rate, 1 / 60);
  expect(reset.weights.every(w => w === 0)).toBe(true);
});

test("higher vowel resonances can still drive close, mid and open poses", () => {
  for (const [index, [f1, f2]] of vowels.entries()) {
    const result = sustained(estimator(), vowel(48000, 245, f1 * 1.16, f2 * 1.16), 48000);
    expect(result.weights.indexOf(Math.max(...result.weights))).toBe(index);
  }
});

test("a measured vowel reference survives a pause and can be cleared", () => {
  const rate = 48000, ownVowel = vowel(rate, 170, 455, 2140);
  const e = estimator();
  const before = sustained(e, ownVowel, rate).weights[1];
  expect(e.setVowelReference(1, 455, 2140)).toBe(true);
  expect(e.setVowelReference(1, 0, Infinity)).toBe(false);
  expect(e.setVowelReference(5, 450, 2000)).toBe(false);
  e.reset();
  expect(sustained(e, ownVowel, rate).weights[1]).toBeGreaterThan(before + .17);
  e.clearReferences();
  e.reset();
  expect(sustained(e, ownVowel, rate).weights[1]).toBeCloseTo(before, 5);
});

for (const hz of [125, 400, 440, 750]) {
  test(`a ${hz} Hz tone cannot supply vowel calibration formants`, () => {
    for (const rate of [16000, 44100, 48000, 96000, 192000]) {
      for (const noiseFloor of [0, .001]) {
        let seed = 47;
        const tone = Float32Array.from({ length: Math.ceil(rate * .038) }, (_, i) => {
          seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
          return .12 * Math.sin(.81 + 2 * Math.PI * hz * i / rate) + .014 + noiseFloor * (seed / 2 ** 31 - 1);
        });
        const pose = sustained(estimator(), tone, rate);
        expect(pose.active).toBe(true);
        expect(pose.f1).toBe(0);
        expect(pose.f2).toBe(0);
        expect(Math.max(...pose.weights)).toBeGreaterThan(.1);
        expect(Math.max(...pose.weights)).toBeLessThan(.3);
        expect(pose.label.toLowerCase()).not.toMatch(/vowel|nasal/);
      }
    }
  });
}

test("broad unvoiced friction gives a diffuse pose distinct from a narrow hiss", () => {
  let seed = 324;
  const broad = Float32Array.from({length: 2048}, () => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return .055 * (seed / 2 ** 31 - 1);
  });
  const pose = sustained(estimator(), broad, 16000);
  expect(pose.weights[10]).toBeGreaterThan(pose.weights[8]);
  expect(pose.weights[10]).toBeGreaterThan(pose.weights[9]);
});
