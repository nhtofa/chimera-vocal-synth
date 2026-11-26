/** Render articulated mouth triangles into a transparent layer over the 2D view. */
function createInsideSurface(lines, faces) {
  const count = 3 * faces.length;
  const source = new Int32Array(count), point = new Int32Array(count);
  const vertex = new Int32Array(count), direction = new Int8Array(faces.length);
  const rgb = new Uint8Array(count), clip = new Uint8Array(faces.length);
  const joints = new Map(), edges = new Map(), neighbors = faces.map(() => []);
  // Weld normals at duplicated ring endpoints. Edge orientation also joins the
  // top and underside of the tongue even when their strips have opposite winding.
  faces.forEach((face, f) => {
    const color = parseInt(face.color.slice(1), 16);
    rgb.set([color >> 16, color >> 8 & 255, color & 255], 3*f);
    const dental = ["upperTeeth", "lowerTeeth", "upperGum", "lowerGum"].includes(face.part);
    clip[f] = dental ? 2 : Number(face.part !== "tongue" && face.part !== "velum");
    face.refs.forEach(([line, index], j) => {
      const k = 3*f+j, i = 3*index, xyz = lines[line].base || lines[line].positions;
      source[k] = line;
      point[k] = i;
      const key = [face.part, xyz[i], xyz[i+1], xyz[i+2]].join("/");
      if (!joints.has(key)) joints.set(key, joints.size);
      vertex[k] = joints.get(key);
    });
    for (let j = 0; j < 3; j++) {
      const a = vertex[3*f+j], b = vertex[3*f+(j+1)%3];
      if (a === b) continue;
      const edge = Math.min(a,b) + "/" + Math.max(a,b);
      const sign = a < b ? 1 : -1;
      const other = edges.get(edge);
      if (other) {
        const relative = -sign * other.sign;
        neighbors[f].push([other.f, relative]);
        neighbors[other.f].push([f, relative]);
      } else edges.set(edge, { f, sign });
    }
  });
  for (let f = 0; f < faces.length; f++) {
    if (direction[f]) continue;
    direction[f] = 1;
    const pending = [f];
    while (pending.length) {
      const current = pending.pop();
      for (const [next, sign] of neighbors[current]) {
        if (direction[next]) continue;
        direction[next] = direction[current] * sign;
        pending.push(next);
      }
    }
  }
  const normals = new Float32Array(3*joints.size), lights = new Float32Array(2*joints.size);
  // Per corner: CSS pixel x/y, reciprocal camera distance, clipping distance,
  // and lit RGB. Both rasterizers interpolate exactly these values.
  const data = new Float32Array(7*count);
  function prepare(cutSide, yaw, tilt, showNearTeeth) {
    normals.fill(0);
    for (let k = 0; k < count; k += 3) {
      const a = lines[source[k]].positions, b = lines[source[k+1]].positions, c = lines[source[k+2]].positions;
      const i = point[k], j = point[k+1], l = point[k+2];
      const x = b[j]-a[i], y = b[j+1]-a[i+1], z = b[j+2]-a[i+2];
      const u = c[l]-a[i], v = c[l+1]-a[i+1], w = c[l+2]-a[i+2];
      let nx = y*w-z*v, ny = z*u-x*w, nz = x*v-y*u;
      // Unit face weights prevent a large cap triangle from flattening a narrow bevel.
      const scale = direction[k/3] / (Math.sqrt(nx*nx+ny*ny+nz*nz) || 1);
      nx *= scale; ny *= scale; nz *= scale;
      for (let t = k; t < k+3; t++) {
        const n = 3*vertex[t];
        normals[n] += nx; normals[n+1] += ny; normals[n+2] += nz;
      }
    }
    const cy = Math.cos(yaw), sy = Math.sin(yaw), ct = Math.cos(tilt), st = Math.sin(tilt);
    for (let i = 0; i < normals.length; i += 3) {
      const nx = normals[i], ny = normals[i+1], nz = normals[i+2];
      const inv = 1/(Math.sqrt(nx*nx+ny*ny+nz*nz) || 1);
      const side = nx*sy+nz*cy;
      const x = (nx*cy-nz*sy)*inv, y = (ny*ct+side*st)*inv, z = (side*ct-ny*st)*inv;
      const key = -.38*x+.66*y+.65*z, half = -.21*x+.35*y+.91*z;
      const j = 2*i/3;
      lights[j] = .56 + .48*Math.max(0,key) + .11*Math.pow(Math.max(0,half),12) + .08*Math.max(0,z);
      lights[j+1] = .56 + .48*Math.max(0,-key) + .11*Math.pow(Math.max(0,-half),12) + .08*Math.max(0,-z);
    }
    for (let k = 0; k < count; k += 3) {
      const f = k/3, a = lines[source[k]].screen, b = lines[source[k+1]].screen, c = lines[source[k+2]].screen;
      const i = point[k], j = point[k+1], l = point[k+2];
      const area = (b[j]-a[i])*(c[l+1]-a[i+1]) - (b[j+1]-a[i+1])*(c[l]-a[i]);
      const back = Number(area*direction[f] > 0);
      for (let t = k; t < k+3; t++) {
        const line = lines[source[t]], s = line.screen, v = point[t], d = 7*t;
        const inv = 68/(68-s[v+2]), light = lights[2*vertex[t]+back];
        data[d] = s[v]; data[d+1] = s[v+1]; data[d+2] = inv;
        data[d+3] = (clip[f] && !(clip[f] === 2 && showNearTeeth)) ? line.positions[v+2]*cutSide*inv : -1;
        data[d+4] = rgb[k]*light; data[d+5] = rgb[k+1]*light; data[d+6] = rgb[k+2]*light;
      }
    }
  }

  const layer = document.createElement("canvas");
  let gl, gpu, tried = false;
  function setupGPU() {
    gl = layer.getContext("webgl", { alpha:true, antialias:true, depth:true, stencil:false, preserveDrawingBuffer:false });
    if (!gl) return null;
    const program = gl.createProgram();
    const compile = (type, src) => {
      const shader = gl.createShader(type);
      gl.shaderSource(shader, src);
      gl.compileShader(shader);
      gl.attachShader(program, shader);
      gl.deleteShader(shader);
    };
    compile(gl.VERTEX_SHADER, `attribute vec4 p;
      attribute vec3 rgb;
      uniform vec2 size;
      varying mediump vec3 color;
      varying mediump float cut;
      void main(){
        vec2 screen = p.xy / size * 2.0 - 1.0;
        gl_Position = vec4(screen.x, -screen.y, 1.0-p.z, 1.0);
        cut=p.w; color=rgb/255.0;
      }`);
    compile(gl.FRAGMENT_SHADER, `precision mediump float;
      varying mediump vec3 color;
      varying mediump float cut;
      void main(){if (cut>0.0) discard; gl_FragColor=vec4(color,1.0);}`);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      gl.deleteProgram(program);
      return null;
    }
    gl.useProgram(program);
    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, data.byteLength, gl.DYNAMIC_DRAW);
    for (const [name, size, offset] of [["p",4,0],["rgb",3,16]]) {
      const index = gl.getAttribLocation(program, name);
      gl.enableVertexAttribArray(index);
      gl.vertexAttribPointer(index, size, gl.FLOAT, false, 28, offset);
    }
    gl.enable(gl.DEPTH_TEST);
    return { size: gl.getUniformLocation(program, "size") };
  }
  layer.addEventListener("webglcontextlost", event => { event.preventDefault(); gpu = null; });
  layer.addEventListener("webglcontextrestored", () => { tried = false; });

  let cpu, pixels, depth, brush;
  function software(w, h) {
    if (!cpu) {
      cpu = document.createElement("canvas");
      brush = cpu.getContext("2d");
    }
    const scale = Math.min(1, Math.sqrt(600000/(w*h)));
    const width = Math.max(1, Math.round(w*scale)), height = Math.max(1, Math.round(h*scale));
    if (!pixels || cpu.width !== width || cpu.height !== height) {
      cpu.width = width; cpu.height = height;
      pixels = brush.createImageData(width,height);
      depth = new Float32Array(width*height);
    }
    const sx = width/w, sy = height/h, rgba = pixels.data;
    rgba.fill(0); depth.fill(0);
    for (let a = 0; a < data.length; a += 21) {
      const b = a+7, c = a+14;
      if (data[a+3]>0 && data[b+3]>0 && data[c+3]>0) continue;
      const x0 = data[a]*sx, y0 = data[a+1]*sy, x1 = data[b]*sx, y1 = data[b+1]*sy, x2 = data[c]*sx, y2 = data[c+1]*sy;
      const area = (x1-x0)*(y2-y0)-(y1-y0)*(x2-x0);
      if (Math.abs(area)<1e-5) continue;
      const left = Math.max(0,Math.ceil(Math.min(x0,x1,x2)-.5)), right = Math.min(width-1,Math.floor(Math.max(x0,x1,x2)-.5));
      const top = Math.max(0,Math.ceil(Math.min(y0,y1,y2)-.5)), bottom = Math.min(height-1,Math.floor(Math.max(y0,y1,y2)-.5));
      const ux = (y2-y0)/area, uy = (x0-x2)/area, vx = (y0-y1)/area, vy = (x1-x0)/area;
      for (let y = top; y <= bottom; y++) {
        let u = (left+.5-x0)*ux+(y+.5-y0)*uy, v = (left+.5-x0)*vx+(y+.5-y0)*vy;
        for (let x = left; x <= right; x++, u+=ux, v+=vx) {
          const t = 1-u-v;
          if (u<-.00001 || v<-.00001 || t<-.00001 || t*data[a+3]+u*data[b+3]+v*data[c+3]>0) continue;
          const index = y*width+x, z = t*data[a+2]+u*data[b+2]+v*data[c+2];
          if (z <= depth[index]) continue;
          depth[index] = z;
          for (let channel = 0; channel < 3; channel++) {
            const j = 4+channel;
            rgba[index*4+channel] = t*data[a+j]+u*data[b+j]+v*data[c+j];
          }
          rgba[index*4+3] = 255;
        }
      }
    }
    brush.putImageData(pixels,0,0);
    return cpu;
  }

  return function paint(ctx, w, h, cutSide, yaw, tilt, showNearTeeth = false) {
    if (!count || !w || !h) return;
    prepare(cutSide, yaw, tilt, showNearTeeth);
    if (!tried) {
      tried = true;
      try { gpu = setupGPU(); } catch { gpu = null; }
    }
    let image;
    if (gpu && !gl.isContextLost()) {
      const width = ctx.canvas.width, height = ctx.canvas.height;
      if (layer.width !== width) layer.width = width;
      if (layer.height !== height) layer.height = height;
      gl.viewport(0,0,width,height);
      gl.uniform2f(gpu.size,w,h);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.bufferSubData(gl.ARRAY_BUFFER,0,data);
      gl.drawArrays(gl.TRIANGLES,0,count);
      image = layer;
    } else image = software(w,h);
    ctx.save();
    ctx.globalAlpha = 1;
    ctx.drawImage(image,0,0,w,h);
    ctx.restore();
  };
}
