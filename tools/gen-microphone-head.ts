/** Bake the engine's articulated head into the single-file microphone page.
 * bun tools/gen-microphone-head.ts [--output path-to-base64]
 */
import { readFileSync, writeFileSync } from "node:fs";
import { initChimera } from "../src/wasm/chimera.ts";
import { Beams, type Beam3Opts } from "../src/ui/render/beams3d.ts";
import type { AnatomyPart } from "../src/ui/render/anatomy3d.ts";
import { PRESETS } from "../src/ui/body.ts";
import type { HumanArtic } from "../src/wasm/types.ts";

const args = process.argv.slice(2);
if (args.length && (args.length !== 2 || args[0] !== "--output")) {
  throw new Error("Usage: bun tools/gen-microphone-head.ts [--output path-to-base64]");
}
await initChimera(readFileSync(new URL("../src/wasm/chimera.wasm", import.meta.url)));
const { VoiceModel } = await import("../src/ui/analysis.ts");
const { Anatomy3D } = await import("../src/ui/render/anatomy3d.ts");
const { NEUTRAL, VOWEL_TARGETS } = await import("../src/wasm/engine.ts");

type MeshPart = AnatomyPart | "upperGum" | "lowerGum";
type Line = { xyz: number[]; color: string; width: number; alpha: number; hot: boolean; part: MeshPart };
type Vertex = [line: number, point: number];
type Face = { refs: [Vertex, Vertex, Vertex]; color: string };
const parts: MeshPart[] = ["other", "tongue", "upperTeeth", "lowerTeeth", "jaw", "lips", "velum", "palate", "airway", "upperGum", "lowerGum"];
class Capture extends Beams {
  lines: Line[] = [];
  part: AnatomyPart = "other";
  override line(points: ArrayLike<number>, n: number, color: string, width: number, o: Beam3Opts = {}) {
    if (n < 2) return;
    const xyz = Array.from({ length: n * 3 }, (_, i) => points[i]);
    if (o.close) xyz.push(...xyz.slice(0, 3));
    this.lines.push({ xyz, color, width, alpha: (o.alpha ?? 1) * this.gain, hot: !!o.hot, part: this.part });
  }
}

const capture = new Capture();
const head = new Anatomy3D(capture, part => { capture.part = part; });
const model = new VoiceModel(PRESETS.male, 48000);
head.camera.position.copy(head.poses(model).home);
// Fixed point/triangle correspondence across poses lets the offline view blend a vowel
// with a closure. Consonant targets come from phonetics/inventory.rs.
const patches: Partial<HumanArtic>[] = [
  {}, ...["ɑ", "i", "u", "ɛ", "o"].map(ipa => ({ ...VOWEL_TARGETS.male[ipa].artic, larynx: NEUTRAL.larynx })),
  { ttX: 6.25, ttY: 9.68, tbX: 3.35, jaw: 0.35 },
  { tbX: 2.3, tbY: 8.35, jaw: 0.7 },
  { lipAp: 0, jaw: 0.28 },
  { ttX: 6.6, ttY: 8.8, tbX: 3.3, jaw: 0.3 },
  { ttX: 5.7, ttY: 9.5, jaw: 0.35, lipProt: 0.75, lipWidth: 0.2 },
  { labiodental: 1, lipAp: 0.05, jaw: 0.35 },
  { velum: 0.7 },
];

function surfaces(lines: Line[], edges: { upper: Float32Array; lower: Float32Array }) {
  const faces: Face[] = [];
  const group = (part: MeshPart, count?: number) => lines.flatMap((l, i) =>
    l.part === part && (count === undefined || l.xyz.length === 3 * count) ? [i] : []);
  const addRow = (like: number, xyz: number[], alpha = 0, part = lines[like].part) => {
    lines.push({ ...lines[like], xyz, alpha, part, hot: false });
    return lines.length - 1;
  };
  const tri = (color: string, a: Vertex, b: Vertex, c: Vertex) => faces.push({ color, refs: [a, b, c] });
  const strip = (a: number, b: number, color: string) => {
    const n = lines[a].xyz.length / 3;
    if (lines[b].xyz.length !== n * 3) throw new Error("Unmatched anatomy surface rows");
    for (let i = 1; i < n; i++) {
      tri(color, [a, i-1], [b, i-1], [a, i]);
      tri(color, [a, i], [b, i-1], [b, i]);
    }
  };
  const sheet = (rows: number[], color: string) => {
    for (let i = 1; i < rows.length; i++) strip(rows[i-1], rows[i], color);
  };

  // The source tongue is an actual transverse elliptical sweep of the engine's
  // midsagittal contour. Complete both ends from its four longitudinal rails.
  const ribs = group("tongue", 13);
  const rails = group("tongue", 22);
  if (rails.length !== 4 || ribs.length !== 10) throw new Error("Tongue sweep changed");
  const endRib = (station: number) => {
    const rail = (r: number, axis: number) => lines[rails[r]].xyz[3*station + axis];
    const xyz: number[] = [];
    for (let i = 0; i <= 12; i++) {
      const angle = (i - 6) * Math.PI / 12;
      for (const axis of [0, 1]) {
        const edge = rail(3, axis);
        xyz.push(edge + (rail(2, axis) - edge) * Math.SQRT2 * Math.cos(angle));
      }
      xyz.push(rail(3, 2) * Math.sin(angle));
    }
    return addRow(ribs[0], xyz);
  };
  const surface = Array.from({ length: 22 }, (_, i) => i % 2 && i < 21 ? ribs[(i-1)/2] : endRib(i));
  sheet(surface, "#ce848b");
  // The engine ends its midsagittal trace at the floor of the mouth: the
  // frenulum and two points leading *away* toward the lower incisors. Follow
  // the frenulum back under the muscle instead of closing that trace diagonally.
  const outline = lines[group("tongue").find(i => lines[i].hot)!].xyz;
  const fren = outline.length - 9;
  const tip = fren - 15; // five segments along the engine's ventral blade
  const root = outline.slice(0, 2);
  const ventral = surface.map((row, station) => {
    const rib = lines[row].xyz;
    const t = station / 21;
    let floor: number[];
    if (t < .73) {
      const u = t/.73, v = 1-u;
      floor = [0,1].map(axis => {
        const start = root[axis] + (axis ? -.18 : 1.12);
        const c1 = root[axis] + (axis ? -.72 : 2.2);
        const c2 = outline[fren+axis] + (axis ? -1.06 : -1.6);
        return v*v*v*start + 3*v*v*u*c1 + 3*v*u*u*c2 + u*u*u*outline[fren+axis];
      });
    } else {
      // The last lower ring sits a little behind the tip to give its cap thickness.
      const sample = (fren-tip)/3 * Math.min(.88, (t-.73)/.27);
      const index = Math.floor(sample), u = sample-index;
      floor = [0,1].map(axis => (1-u)*outline[fren-3*index+axis] + u*outline[fren-3*(index+1)+axis]);
    }
    const xyz = rib.slice();
    for (let j = 0; j <= 12; j++) {
      const d = Math.cos((j-6)*Math.PI/12);
      for (const axis of [0,1]) xyz[3*j+axis] = rib[axis] + d*(floor[axis]-rib[axis]);
    }
    return addRow(row, xyz);
  });
  sheet(ventral, "#a86475");
  strip(surface[0], ventral[0], "#a86273");
  strip(surface[21], ventral[21], "#ce828b");
  // These triangles are the real central cut surface. Give every station its
  // own floor position so both sagittal and oblique projections stay closed.
  for (let i = 1; i < surface.length; i++) {
    tri("#c8888d", [surface[i-1],6], [surface[i],6], [ventral[i-1],6]);
    tri("#c8888d", [ventral[i-1],6], [surface[i],6], [ventral[i],6]);
  }

  for (const part of ["upperTeeth", "lowerTeeth"] as const) {
    const crowns = group(part, 8), bites = group(part, 5);
    if (crowns.length !== 14 || crowns.length !== bites.length) throw new Error("Dental arch changed");
    const gums: number[][] = [[], []];
    crowns.forEach((crown, i) => {
      const front = lines[crown].xyz, bite = lines[bites[i]].xyz;
      const position = i % 7, side = Math.floor(i/7);
      const incisor = position < 2, canine = position === 2, molar = position >= 5;
      // Basis vectors already include the exact rotation of the mandible.
      const at = (along: number, height: number, depth: number): number[] =>
        [0,1,2].map(axis => {
          const mid = (front[axis]+front[18+axis])/2;
          const tangent = front[18+axis]-front[axis];
          const up = (front[9+axis]-mid)/1.08;
          const inward = (bite[9+axis]-bite[axis])/.8;
          return mid + along*tangent + height*up + depth*inward;
        });
      const row = (height: number, halfWidth: number, halfDepth: number, centre: number, relief: number) => {
        const xyz: number[] = [];
        for (let j = 0; j <= 12; j++) {
          const a = (j%12)*Math.PI/6, cs = Math.cos(a), sn = Math.sin(a);
          const power = molar ? .65 : .81;
          const u = Math.sign(cs)*Math.pow(Math.abs(cs), power);
          const d = Math.sign(sn)*Math.pow(Math.abs(sn), power);
          // Low four-cusp molars, bicuspid premolars, a single canine cusp,
          // or the broad, thin incisal blade. Height grows towards the gingiva.
          const cusp = incisor ? .06*Math.abs(u) : canine ? .33*Math.abs(u)
            : molar ? .08*(1+Math.cos(4*a)) : .13*Math.abs(u);
          xyz.push(...at(halfWidth*u, height+relief*cusp, centre+halfDepth*d));
        }
        return addRow(crown, xyz);
      };
      const biting = row(.025, incisor ? .40 : canine ? .22 : .39, incisor ? .065 : canine ? .13 : .32, .42, 1);
      const shoulder = row(.28, .47, incisor ? .20 : .43, .46, .4);
      const belly = row(.58, .49, incisor ? .31 : .48, .47, 0);
      const neck = row(.94, .39, incisor ? .26 : .38, .48, 0);
      sheet([biting, shoulder, belly, neck], "#f1e5ce");
      const centre = addRow(crown, [...at(0, incisor ? .022 : canine ? -.035 : .13,.44), ...at(0,.96,.48)]);
      for (let j = 0; j < 12; j++) {
        tri("#f5e9d3", [biting,j], [biting,j+1], [centre,0]);
        tri("#e3d2bc", [neck,j+1], [neck,j], [centre,1]);
      }
      // Gingiva wraps the cervical margin and rises between neighbours. One
      // continuous tube per arch avoids independent boxes around the roots.
      const gumPart = part === "upperTeeth" ? "upperGum" : "lowerGum";
      const gum = (u: number, margin: number) => {
        const xyz: number[] = [];
        for (const [h,d] of [[margin,.02],[1.02,-.055],[1.28,.02],[1.43,.30],[1.40,.66],[1.23,.98],[1.01,1.02],[margin,.91],[margin+.12,.52],[margin,.02]]) {
          xyz.push(...at(u,h,d));
        }
        return addRow(crown, xyz, 0, gumPart);
      };
      if (position === 0) gums[side].push(gum(-.5,.62));
      gums[side].push(gum(0,.76), gum(.5, .62));
    });
    // Bridge the central incisors and cap behind the last molars.
    const arch = [...gums[0]].reverse().concat(gums[1]);
    sheet(arch, part === "upperTeeth" ? "#bf8184" : "#be777e");
    for (const row of [arch[0], arch[arch.length-1]]) {
      for (let j = 1; j < 8; j++) tri("#ba737d", [row,0], [row,j], [row,j+1]);
    }
  }
  sheet(group("palate", 21), "#ad7079");
  sheet(group("velum", 13), "#bc8290");

  // Replace the variable-length single opening stroke by anatomical upper and
  // lower margins. The paired rows coincide continuously at bilabial closure.
  const borders = group("lips", 29);
  if (borders.length !== 4) throw new Error("Lip borders changed");
  const opening = lines.find(l => l.part === "lips" && l.width === 2.1);
  if (!opening) throw new Error("Missing lip opening");
  opening.xyz = Array.from(edges.upper);
  const upper = lines.indexOf(opening);
  const lower = addRow(upper, Array.from(edges.lower), opening.alpha);
  for (const [edge, middle, border, sign] of [[upper, borders[1], borders[0], 1], [lower, borders[3], borders[2], -1]]) {
    const a = lines[edge].xyz, b = lines[border].xyz;
    // Complete the vermilion patch on its oral side. Taper the inner roll at
    // the commissures, maintaining the exact engine-driven free margin.
    const inner = (fraction: number, thickness: number, roll: number) => {
      const xyz = a.map((v,j) => {
        const u = Math.floor(j/3)/(a.length/3-1);
        const width = Math.sin(Math.PI*u)**.65;
        const axis = j%3;
        return v + fraction*(b[j]-v) + width*(axis === 0 ? -thickness : axis === 1 ? sign*roll : 0);
      });
      return addRow(edge, xyz);
    };
    const rows = [edge, middle, border, inner(.91,.30,.07), inner(.62,.56,.08), inner(.24,.42,.05), inner(.07,.18,.025)];
    sheet([...rows, edge], sign === 1 ? "#bf737e" : "#d1868c");
    // Seal the sagittal section of each roll. The skin outlines extending to
    // the nose and chin are available as wireframe context outside this cut.
    for (let j = 1; j < rows.length-1; j++) tri("#cd8a8c", [edge,14], [rows[j],14], [rows[j+1],14]);
  }
  // Acoustic stops deliberately aim beyond contact. Limit the visible soft
  // tissue against the same hard palate and dental arches that the viewer sees.
  const roof = lines[group("palate").find(i => lines[i].xyz.every((n,j) => j%3 !== 2 || n === 0))!].xyz;
  const vault = group("palate",21).map(i => lines[i].xyz);
  const vaultDrop = (row: number[], z: number) => {
    const q = Math.min(10, 10*Math.abs(z)/row[row.length-1]);
    const j = Math.min(9,Math.floor(q)), u = q-j;
    return row[31] - ((1-u)*row[31+3*j] + u*row[34+3*j]);
  };
  const arches = (["upperTeeth","lowerTeeth"] as const).map(part => {
    // Original bite outlines (before the surface rows) run labial to lingual.
    return group(part,5).slice(0,7).flatMap(i => {
      const a = lines[i].xyz;
      return [[a[6],a[7],a[8]], [a[9],a[10],a[11]]];
    }).sort((a,b) => a[0]-b[0]);
  });
  for (const line of lines.filter(l => l.part === "tongue")) {
    const a = line.xyz;
    for (let j=0; j<a.length; j+=3) {
      const x = a[j], y = a[j+1];
      let z = Math.abs(a[j+2]);
      for (const [k, arch] of arches.entries()) {
        const i = arch.findIndex(v => v[0]>=x);
        if (i<1) continue;
        const u=(x-arch[i-1][0])/(arch[i][0]-arch[i-1][0] || 1);
        const biteY=arch[i-1][1]+u*(arch[i][1]-arch[i-1][1]);
        const biteZ=arch[i-1][2]+u*(arch[i][2]-arch[i-1][2]);
        const clear = k === 0 ? Math.max(0,biteY-y) : Math.max(0,y-biteY);
        z = Math.min(z, Math.max(0,biteZ-.045) + clear*clear*5);
      }
      a[j+2] = Math.sign(a[j+2])*z;
      for (let p=3; p<roof.length; p+=3) {
        if (x<roof[p-3] || x>roof[p]) continue;
        const u=(x-roof[p-3])/(roof[p]-roof[p-3]);
        let limit = roof[p-2]+u*(roof[p+1]-roof[p-2]);
        const i = vault.findIndex(row => row[0]>=x);
        if (i>0) {
          const v=(x-vault[i-1][0])/(vault[i][0]-vault[i-1][0]);
          limit -= (1-v)*vaultDrop(vault[i-1],z) + v*vaultDrop(vault[i],z);
        }
        a[j+1] = Math.min(y, limit-.018);
        break;
      }
    }
  }
  return faces;
}

let faces: Face[] = [];
const poses = patches.map((patch, pose) => {
  const a = { ...NEUTRAL, ...patch };
  capture.lines = [];
  head.update(model, model.frame(a), a, null, { pressure: false, particles: false, grid: false, calipers: false }, 0);
  const topology = surfaces(capture.lines, head.lipEdges(model, a, 28));
  if (!pose) faces = topology;
  else if (topology.length !== faces.length) throw new Error("Surface topology changed with pose");
  return capture.lines;
});
head.dispose();
model.dispose();
const [rest, ...targets] = poses;
if (!poses.every(p => p.length === rest.length)) throw new Error("Anatomy stroke count changed with pose");

function fit(points: number[], length: number): number[] {
  if (points.length === length) return points;
  return Array.from({ length }, (_, i) => {
    const u = Math.floor(i / 3) * (points.length / 3 - 1) / (length / 3 - 1);
    const a = Math.floor(u), b = Math.min(a + 1, points.length / 3 - 1), axis = i % 3;
    return points[3*a+axis]*(1-u+a) + points[3*b+axis]*(u-a);
  });
}
function quantize(n: number) {
  const q = Math.round(n*512);
  if (!Number.isFinite(q) || Math.abs(q) > 32767) throw new Error("Invalid head coordinate");
  return q;
}

// Little-endian. Versioned header, independently masked morphs per polyline,
// and three line/point references per face so surfaces follow those morphs exactly.
const header = Buffer.alloc(8);
header.write("CHM4");
header.writeUInt16LE(rest.length, 4);
header.writeUInt16LE(faces.length, 6);
const chunks = [header];
for (const [i, line] of rest.entries()) {
  const base = line.xyz.map(quantize);
  const arrays = [base];
  let mask = 0;
  targets.forEach((target, pose) => {
    const l = target[i];
    if (l.part !== line.part || l.color !== line.color || l.width !== line.width) throw new Error("Mismatched pose at stroke " + i);
    const delta = fit(l.xyz, base.length).map((n, j) => quantize(n) - base[j]);
    if (delta.some(x => x !== 0)) {
      mask |= 1 << pose;
      arrays.push(delta);
    }
  });
  const midline = base.every((n, j) => j % 3 !== 2 || n === 0);
  const buf = Buffer.alloc(11 + 2*base.length*arrays.length);
  buf.writeUInt16LE(base.length/3, 0);
  buf.writeUIntBE(parseInt(line.color.slice(1), 16), 2, 3);
  buf[5] = Math.round(line.width*32);
  buf[6] = Math.round(line.alpha*255);
  buf[7] = Number(midline) | (Number(line.hot)<<1);
  buf[8] = parts.indexOf(line.part);
  buf.writeUInt16LE(mask, 9);
  let offset = 11;
  for (const coords of arrays) for (const value of coords) {
    buf.writeInt16LE(value, offset);
    offset += 2;
  }
  chunks.push(buf);
}
for (const f of faces) {
  const buf = Buffer.alloc(15);
  buf.writeUIntBE(parseInt(f.color.slice(1), 16), 0, 3);
  f.refs.forEach(([l, p], i) => {
    buf.writeUInt16LE(l, 3+4*i);
    buf.writeUInt16LE(p, 5+4*i);
  });
  chunks.push(buf);
}

const data = Buffer.concat(chunks).toString("base64");
if (args.length) {
  writeFileSync(args[1], data);
} else {
  const page = new URL("../public/microphone.html", import.meta.url);
  let html = readFileSync(page, "utf8");
  function replaceScript(id: string, contents: string) {
    const pattern = new RegExp('(<script[^>]* id="' + id + '"[^>]*>)[\\s\\S]*?(<\\/script>)');
    if (!pattern.test(html)) throw new Error("Missing microphone page slot " + id);
    html = html.replace(pattern, (_, a, b) => a + contents + b);
  }
  replaceScript("head-mesh", data);
  for (const id of ["head-model", "speech-estimator", "inside-surface"]) {
    replaceScript(id, "\n" + readFileSync(new URL("../src/microphone/" + id + ".js", import.meta.url), "utf8"));
  }
  writeFileSync(page, html);
}
console.log(`Baked ${rest.length} anatomy lines, ${faces.length} triangles, and ${targets.length} morphs.`);
