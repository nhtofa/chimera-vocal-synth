import { expect, test } from "bun:test";
import { readFileSync } from "node:fs";

const page = readFileSync(new URL("../public/microphone.html", import.meta.url), "utf8");
test("the standalone download embeds the current model, estimator and renderer", () => {
  for (const id of ["head-model", "speech-estimator", "inside-surface"]) {
    const embedded = page.match(new RegExp('<script id="' + id + '">([\\s\\S]*?)</script>'));
    expect(embedded?.[1].trim()).toBe(readFileSync(new URL("../src/microphone/" + id + ".js", import.meta.url), "utf8").trim());
  }
  expect(page).not.toMatch(/<script[^>]+ src=/i);
});

test("192 kHz microphone input retains a full 32 ms analysis window", () => {
  const setup = page.match(/input\.analyser\.fftSize = ([^;]+);/);
  expect(setup).toBeTruthy();
  const input = {context: {sampleRate: 192000}};
  const size = new Function("input", "return " + setup![1])(input);
  expect(size).toBeGreaterThanOrEqual(input.context.sampleRate * .032);
  expect(Math.log2(size)).toBeInteger();
});
