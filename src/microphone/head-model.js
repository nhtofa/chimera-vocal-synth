/** Decode the quantized CHM4 anatomy packed by tools/gen-microphone-head.ts.
 * Body coordinates are centimetres from the glottis, +x anterior, +y superior.
 */
function decodeSpeechHead(base64) {
  const bytes = Uint8Array.from(atob(base64), c => c.charCodeAt(0));
  const data = new DataView(bytes.buffer);
  if (bytes.length < 8 || data.getUint32(0) !== 0x43484d34) throw new Error("Unknown head mesh format");
  const lineCount = data.getUint16(4, true), faceCount = data.getUint16(6, true);
  const parts = ["other", "tongue", "upperTeeth", "lowerTeeth", "jaw", "lips", "velum", "palate", "airway", "upperGum", "lowerGum"];
  let p = 8;
  const byte = () => data.getUint8(p++);
  const u16 = () => { const v = data.getUint16(p, true); p += 2; return v; };
  const rgb = () => "#" + ((byte()<<16) | (byte()<<8) | byte()).toString(16).padStart(6, "0");
  const coords = n => {
    if (p + n*2 > bytes.length) throw new Error("Incomplete head coordinates");
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++, p += 2) out[i] = data.getInt16(p, true)/512;
    return out;
  };
  const lines = [], faces = [];
  for (let i = 0; i < lineCount; i++) {
    const n = u16(), color = rgb(), width = byte()/32, alpha = byte()/255;
    const flags = byte(), part = parts[byte()], mask = u16();
    if (!part || n < 2 || mask >= 4096) throw new Error("Invalid head stroke");
    const base = coords(3*n);
    const deltas = Array.from({ length: 12 }, (_, j) => mask & (1<<j) ? coords(3*n) : null);
    lines.push({ base, deltas, color, width, alpha, section: !!(flags&1), hot: !!(flags&2), part });
  }
  for (let i = 0; i < faceCount; i++) {
    const color = rgb(), refs = [];
    for (let j = 0; j < 3; j++) {
      const line = u16(), point = u16();
      if (!lines[line] || point*3 >= lines[line].base.length) throw new Error("Invalid head triangle");
      refs.push([line, point]);
    }
    faces.push({ refs, color, part: lines[refs[0][0]].part });
  }
  if (p !== bytes.length) throw new Error("Trailing head data");
  return { lines, faces };
}
