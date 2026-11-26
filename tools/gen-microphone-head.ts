/** Bake the synth's wireframe into the offline microphone page. Run after bun run wasm:
 *  bun tools/gen-microphone-head.ts
 *  Positions are in body cm. Only moving lines need a second or third set of coordinates.
 */
import { readFileSync, writeFileSync } from "node:fs";
import { initChimera } from "../src/wasm/chimera.ts";
import { Beams, type Beam3Opts } from "../src/ui/render/beams3d.ts";
import { PRESETS } from "../src/ui/body.ts";

await initChimera(readFileSync(new URL("../src/wasm/chimera.wasm", import.meta.url)));
const { VoiceModel } = await import("../src/ui/analysis.ts");
const { Anatomy3D } = await import("../src/ui/render/anatomy3d.ts");
const { NEUTRAL } = await import("../src/wasm/engine.ts");

type Line = { xyz: number[]; color: string; width: number; alpha: number; hot: boolean };
class Capture extends Beams {
  lines: Line[] = [];
  override line(points: ArrayLike<number>, n: number, color: string, width: number, o: Beam3Opts = {}) {
    if (n < 2) return;
    const xyz = Array.from({ length: n * 3 }, (_, i) => points[i]);
    if (o.close) xyz.push(...xyz.slice(0, 3));
    this.lines.push({ xyz, color, width, alpha: (o.alpha ?? 1) * this.gain, hot: !!o.hot });
  }
}
const capture = new Capture();
const head = new Anatomy3D(capture);
const model = new VoiceModel(PRESETS.male, 48000);
head.camera.position.copy(head.poses(model).home);
const poses = [
  NEUTRAL,
  { ...NEUTRAL, jaw: 2.1, lipAp: 2.0, tbX: 2.3, tbY: 5.3, ttY: 6.2 },
  { ...NEUTRAL, jaw: 0.8, lipAp: 0.85, lipWidth: 1.0, tbX: 3.7, tbY: 7.55, ttY: 7.9 },
].map((a) => {
  capture.lines = [];
  head.update(model, model.frame(a), a, null, { pressure: false, particles: false, grid: false, calipers: false }, 0);
  return capture.lines;
});
head.dispose();
model.dispose();
const [rest, open, spread] = poses;
if (!poses.every((p) => p.length === rest.length)) throw new Error("Anatomy stroke order changed");

// The tract can add samples to a contour as the jaw opens. Match the endpoints and
// interpolate along the ordered samples when two poses have different vertex counts.
function fit(points: number[], length: number): number[] {
  if (points.length === length) return points;
  return Array.from({ length }, (_, i) => {
    const u = Math.floor(i / 3) * (points.length / 3 - 1) / (length / 3 - 1);
    const a = Math.floor(u), b = Math.min(a + 1, points.length / 3 - 1), axis = i % 3;
    return points[3 * a + axis] * (1 - (u - a)) + points[3 * b + axis] * (u - a);
  });
}

const chunks: Buffer[] = [];
for (const [i, line] of rest.entries()) {
  if (open[i].color !== line.color || spread[i].color !== line.color) throw new Error("Mismatched pose at stroke " + i);
  const quantize = (n: number) => {
    const q = Math.round(n * 512);
    if (!Number.isFinite(q) || Math.abs(q) > 32767) throw new Error("Invalid head coordinate");
    return q;
  };
  const base = line.xyz.map(quantize);
  const arrays = [base, ...[open[i], spread[i]].map((p) => fit(p.xyz, base.length).map((n, j) => quantize(n) - base[j]))];
  const moving = arrays.slice(1).some((delta) => delta.some((n) => n !== 0));
  const midline = base.every((n, j) => j % 3 !== 2 || n === 0);
  const buf = Buffer.alloc(8 + base.length * (moving ? 6 : 2));
  buf.writeUInt16LE(base.length / 3, 0);
  buf.writeUIntBE(parseInt(line.color.slice(1), 16), 2, 3);
  buf.writeUInt8(Math.round(line.width * 32), 5);
  buf.writeUInt8(Math.round(line.alpha * 255), 6);
  buf.writeUInt8(Number(moving) | (Number(midline) << 1) | (Number(line.hot) << 2), 7);
  let offset = 8;
  for (const coords of moving ? arrays : [base]) {
    for (const value of coords) {
      buf.writeInt16LE(value, offset);
      offset += 2;
    }
  }
  chunks.push(buf);
}
const page = new URL("../public/microphone.html", import.meta.url);
const html = readFileSync(page, "utf8");
const pattern = /(<script type="application\/octet-stream" id="head-mesh">)[\s\S]*?(<\/script>)/;
if (!pattern.test(html)) throw new Error("Microphone page is missing its head mesh slot");
const data = Buffer.concat(chunks).toString("base64");
writeFileSync(page, html.replace(pattern, (_, a, b) => a + data + b));
console.log(`Embedded ${rest.length} wireframe lines and two speech poses.`);
