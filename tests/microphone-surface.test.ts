import { expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { runInNewContext } from "node:vm";

test("intersecting tissues occlude per pixel and only the near dental half is cut", () => {
  let image = new Uint8ClampedArray(0);
  const canvas = () => ({
    width: 0, height: 0, addEventListener() {},
    getContext: (kind: string) => kind === "webgl" ? null : {
      createImageData: (w: number, h: number) => ({ data: new Uint8ClampedArray(w*h*4) }),
      putImageData: (pixels: {data: Uint8ClampedArray<ArrayBuffer>}) => { image = pixels.data; },
    },
  });
  const create = runInNewContext(readFileSync("src/microphone/inside-surface.js","utf8") + "\ncreateInsideSurface", {
    document: { createElement: canvas },
  });
  const xy = [[3,3],[97,3],[48,95]];
  const triangle = (depth: number[], side: number[]) => ({
    positions: new Float32Array(xy.flatMap(([x,y],i) => [x/10,y/10,side[i]])),
    screen: new Float32Array(xy.flatMap(([x,y],i) => [x,y,depth[i]])),
  });
  const lines = [triangle([-8,8,2],[1,1,1]), triangle([8,-8,-2],[1,1,1]), triangle([12,12,12],[1,-1,0])];
  const faces = [["tongue","#ff0000"],["velum","#00ff00"],["upperTeeth","#0000ff"]].map(([part,color],i) =>
    ({ part, color, refs: [[i,0],[i,1],[i,2]] }));
  const paint = create(lines,faces);
  const ctx = { save(){}, restore(){}, drawImage(){} };
  const channel = (x: number) => {
    const index = 4*(100*10+x);
    expect(image[index+3]).toBe(255);
    const rgb = Array.from(image.slice(index,index+3));
    return rgb.indexOf(Math.max(...rgb));
  };
  paint(ctx,100,100,1,0,0);
  expect([channel(20),channel(80)]).toEqual([1,2]);
  paint(ctx,100,100,-1,0,0);
  expect([channel(20),channel(80)]).toEqual([2,0]);
});
