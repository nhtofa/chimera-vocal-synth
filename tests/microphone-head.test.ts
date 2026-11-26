import { expect, test } from "bun:test";
import { readFileSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { runInNewContext } from "node:vm";

type Line = { base: Float32Array; deltas: (Float32Array | null)[]; part: string };
type Face = { refs: [number, number][]; part: string };
type Mesh = { lines: Line[]; faces: Face[] };

test("baked speech anatomy retains fixed upper teeth, rigid mandibular teeth, independent gestures and valid surface triangles", () => {
  const dir = mkdtempSync(join(tmpdir(), "speech-head-"));
  try {
    const source = readFileSync("src/microphone/head-model.js", "utf8");
    const output = join(dir, "mesh.b64");
    const result = Bun.spawnSync([process.execPath, "tools/gen-microphone-head.ts", "--output", output]);
    if (result.exitCode) throw new Error(result.stderr.toString());
    const decode = runInNewContext(source + "\ndecodeSpeechHead", { atob, Float32Array });
    const generated = readFileSync(output, "utf8");
    const page = readFileSync("public/microphone.html", "utf8");
    const embedded = page.match(/<script type="application\/octet-stream" id="head-mesh">([^<]*)<\/script>/);
    expect(embedded?.[1].trim() === generated).toBe(true);
    const { lines, faces }: Mesh = decode(generated);
    const of = (part: string) => lines.filter(l => l.part === part);
    for (const l of lines) {
      expect(l.base.length % 3).toBe(0);
      expect(l.deltas.length).toBe(12);
      for (const delta of l.deltas) if (delta) expect(delta.length).toBe(l.base.length);
    }
    for (const part of ["tongue", "palate", "velum", "lips", "upperTeeth", "lowerTeeth", "upperGum", "lowerGum"]) {
      expect(of(part).length).toBeGreaterThan(0);
      expect(faces.filter(f => f.part === part).length).toBeGreaterThan(8);
    }
    for (const f of faces) {
      expect(f.refs.length).toBe(3);
      for (const [l, p] of f.refs) {
        expect(lines[l].part).toBe(f.part);
        expect(3*p + 2).toBeLessThan(lines[l].base.length);
      }
    }
    expect(of("upperGum").every(l => l.deltas.every(d => d === null))).toBe(true);
    expect(of("upperTeeth").every(l => l.deltas.every(d => d === null))).toBe(true);
    const moving = (part: string, pose: number) => Math.max(...of(part).map(l => Math.max(...(l.deltas[pose] ?? [0]).map(Math.abs))));
    for (const pose of [0,1,2,3,4,5,6,8,9]) expect(moving("tongue", pose)).toBeGreaterThan(0.25);
    // Stop targets aim past contact in the engine. Visually, soft tissue meets
    // the undersurface of the hard palate rather than emerging through it.
    const roof = of("palate").find(l => l.base.every((v,j) => j%3 !== 2 || v === 0))!.base;
    const centre = of("tongue").find(l => l.base.length > 120 && l.base.every((v,j) => j%3 !== 2 || v === 0))!;
    for (const k of [5,6]) for (let j=0; j<centre.base.length; j+=3) {
      const x = centre.base[j] + centre.deltas[k]![j], y = centre.base[j+1] + centre.deltas[k]![j+1];
      for (let i=3; i<roof.length; i+=3) if (x >= roof[i-3] && x <= roof[i]) {
        const u=(x-roof[i-3])/(roof[i]-roof[i-3]);
        expect(y).toBeLessThanOrEqual(roof[i-2]+u*(roof[i+1]-roof[i-2])+.005);
      }
    }
    expect(moving("lips", 7)).toBeGreaterThan(0.25);
    expect(moving("lips", 10)).toBeGreaterThan(0.35);
    expect(moving("velum", 11)).toBeGreaterThan(0.6);
    expect(moving("lowerGum", 0)).toBeGreaterThan(0.5);
    expect(moving("lowerTeeth", 0)).toBeGreaterThan(0.5);
    // Pick a tooth and a distant point of the mandible. Their separation stays fixed in every target.
    const a = of("lowerTeeth")[0], b = of("jaw")[0];
    const separation = (pose: number) => Math.hypot(...[0,1,2].map(i =>
      a.base[i] + (a.deltas[pose]?.[i] ?? 0) - b.base[i] - (b.deltas[pose]?.[i] ?? 0)));
    for (let i = 0; i < 12; i++) expect(Math.abs(separation(i) - separation(-1))).toBeLessThan(0.005);
    expect(() => decode("eA==")).toThrow();
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
