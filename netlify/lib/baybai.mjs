// BaybAI paper pipeline in JavaScript, for the Netlify function (netlify/functions/ocr.mjs).
// A line-by-line port of model-training/baybayin_preprocess.py and
// model-training/translate_pipeline.py (Section 3.5 of the paper), running the
// trained models exported by model-training/export_web_models.py.
import { readFileSync, existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

// --------------------------------------------------------------------------- data
function dataDir() {
  const here = dirname(fileURLToPath(import.meta.url));
  const candidates = [join(here, "..", "ocr-data"), join(here, "ocr-data"),
    join(process.cwd(), "netlify", "ocr-data")];
  for (const c of candidates) if (existsSync(join(c, "params.json"))) return c;
  throw new Error("OCR model files not found (netlify/ocr-data)");
}

let MODELS = null;
export function loadModels() {
  if (MODELS) return MODELS;
  const dir = dataDir();
  const json = (f) => JSON.parse(readFileSync(join(dir, f), "utf8"));
  const cnn = (name) => {
    const spec = json(`${name}.json`);
    const buf = readFileSync(join(dir, `${name}.bin`));
    const f32 = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
    const arr = (ref) => f32.subarray(ref.offset / 4, ref.offset / 4 + ref.shape.reduce((a, b) => a * b, 1));
    return { classes: spec.classes, layers: spec.layers.map((l) => ({ ...l, w: l.weight && arr(l.weight), b: l.bias && arr(l.bias) })) };
  };
  const svm = json("script.json");
  const bits = readFileSync(join(dir, "script_bits.bin"));
  const lang = json("language.json");
  MODELS = {
    params: json("params.json"),
    cnn: cnn("cnn"),
    diacritic: cnn("diacritic"),
    svm: { ...svm, bits, rowBytes: Math.ceil(svm.n_features / 8),
      // Pixels are 0/1, so a standardised difference is 0 or 1/scale: its square per feature
      weight: svm.scale.map((s) => 1 / (s * s)) },
    words: new Set(lang.words),
    lm: lang,
    dict: json("dictionary.json"),
  };
  return MODELS;
}

// --------------------------------------------------------------------------- image helpers (OpenCV-compatible)
const roundHalfEven = (x) => { const f = Math.floor(x), d = x - f; return d > 0.5 || (d === 0.5 && f % 2 !== 0) ? f + 1 : f; };
const median = (a) => { const s = [...a].sort((x, y) => x - y), n = s.length; return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2; };
const percentile = (a, q) => { const s = [...a].sort((x, y) => x - y), pos = (s.length - 1) * q / 100, lo = Math.floor(pos); return lo + 1 < s.length ? s[lo] + (s[lo + 1] - s[lo]) * (pos - lo) : s[lo]; };
const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;

/** RGB(A) raw pixels -> grayscale (cv2 BGR2GRAY weights); transparency composited on white. */
export function toGray(data, width, height, channels) {
  const g = new Uint8Array(width * height);
  for (let i = 0, p = 0; i < g.length; i++, p += channels) {
    let r = data[p], gr = data[p + 1], b = data[p + 2];
    if (channels === 4) {
      const a = data[p + 3] / 255;
      r = r * a + 255 * (1 - a); gr = gr * a + 255 * (1 - a); b = b * a + 255 * (1 - a);
      r = Math.trunc(r); gr = Math.trunc(gr); b = Math.trunc(b);
    }
    g[i] = (r * 4899 + gr * 9617 + b * 1868 + 8192) >> 14;
  }
  return { data: g, width, height };
}

function medianBlur3(img) {
  const { data, width: W, height: H } = img, out = new Uint8Array(W * H), win = new Uint8Array(9);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    let k = 0;
    for (let dy = -1; dy <= 1; dy++) {
      const yy = Math.min(H - 1, Math.max(0, y + dy));
      for (let dx = -1; dx <= 1; dx++) win[k++] = data[yy * W + Math.min(W - 1, Math.max(0, x + dx))];
    }
    win.sort();
    out[y * W + x] = win[4];
  }
  return { data: out, width: W, height: H };
}

function otsuThreshold(data) {
  const h = new Float64Array(256);
  for (const v of data) h[v]++;
  const N = data.length, scale = 1 / N;
  let mu = 0;
  for (let i = 0; i < 256; i++) mu += i * h[i];
  mu *= scale;
  let mu1 = 0, q1 = 0, maxSigma = 0, maxVal = 0;
  const EPS = 1.1920928955078125e-7;
  for (let i = 0; i < 256; i++) {
    const p = h[i] * scale;
    mu1 *= q1; q1 += p;
    const q2 = 1 - q1;
    if (Math.min(q1, q2) < EPS || Math.max(q1, q2) > 1 - EPS) continue;
    mu1 = (mu1 + i * p) / q1;
    const mu2 = (mu - q1 * mu1) / q2, sigma = q1 * q2 * (mu1 - mu2) ** 2;
    if (sigma > maxSigma) { maxSigma = sigma; maxVal = i; }
  }
  return maxVal;
}

/** Grayscale -> median filter -> Otsu. Ink = 255. lightText: null = guess (text is the smaller class). */
export function binarize(gray, p, lightText = null) {
  const g = medianBlur3(gray), t = otsuThreshold(g.data), bw = new Uint8Array(g.data.length);
  let ink = 0;
  for (let i = 0; i < bw.length; i++) { bw[i] = g.data[i] > t ? 0 : 255; ink += bw[i] ? 1 : 0; }
  if (lightText === null) lightText = ink > 0.5 * bw.length;
  if (lightText) for (let i = 0; i < bw.length; i++) bw[i] = 255 - bw[i];
  return { data: bw, width: g.width, height: g.height };
}

/** cv2.resize INTER_AREA for downscaling (bilinear when enlarging). */
function resizeArea(img, W2, H2) {
  const { data, width: W, height: H } = img, out = new Float64Array(W2 * H2);
  const sx = W / W2, sy = H / H2;
  if (sx < 1 || sy < 1) return resizeBilinear(img, W2, H2);
  const wx = [], wy = [];
  for (let x = 0; x < W2; x++) { const a = x * sx, b = a + sx, list = []; for (let s = Math.floor(a); s < Math.ceil(b) && s < W; s++) { const w = Math.min(b, s + 1) - Math.max(a, s); if (w > 1e-9) list.push([s, w / sx]); } wx.push(list); }
  for (let y = 0; y < H2; y++) { const a = y * sy, b = a + sy, list = []; for (let s = Math.floor(a); s < Math.ceil(b) && s < H; s++) { const w = Math.min(b, s + 1) - Math.max(a, s); if (w > 1e-9) list.push([s, w / sy]); } wy.push(list); }
  for (let y = 0; y < H2; y++) for (let x = 0; x < W2; x++) {
    let v = 0;
    for (const [yy, a] of wy[y]) for (const [xx, b] of wx[x]) v += data[yy * W + xx] * a * b;
    out[y * W2 + x] = v;
  }
  const o = new Uint8Array(W2 * H2);
  for (let i = 0; i < o.length; i++) o[i] = Math.min(255, Math.max(0, roundHalfEven(out[i])));
  return { data: o, width: W2, height: H2 };
}

function resizeBilinear(img, W2, H2) {
  const { data, width: W, height: H } = img, o = new Uint8Array(W2 * H2), sx = W / W2, sy = H / H2;
  for (let y = 0; y < H2; y++) {
    const fy = Math.min(H - 1, Math.max(0, (y + 0.5) * sy - 0.5)), y0 = Math.floor(fy), y1 = Math.min(H - 1, y0 + 1), dy = fy - y0;
    for (let x = 0; x < W2; x++) {
      const fx = Math.min(W - 1, Math.max(0, (x + 0.5) * sx - 0.5)), x0 = Math.floor(fx), x1 = Math.min(W - 1, x0 + 1), dx = fx - x0;
      const v = data[y0 * W + x0] * (1 - dx) * (1 - dy) + data[y0 * W + x1] * dx * (1 - dy) + data[y1 * W + x0] * (1 - dx) * dy + data[y1 * W + x1] * dx * dy;
      o[y * W2 + x] = roundHalfEven(v);
    }
  }
  return { data: o, width: W2, height: H2 };
}

/** Rotation about the image centre by `angle` degrees (counter-clockwise, as cv2). */
function rotate(img, angle, nearest) {
  const { data, width: W, height: H } = img, out = new Uint8Array(W * H);
  const a = (angle * Math.PI) / 180, c = Math.cos(a), s = Math.sin(a), cx = W / 2, cy = H / 2;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    // inverse mapping: source = R^-1 (dst - centre) + centre
    const dx = x - cx, dy = y - cy, sx = c * dx - s * dy + cx, sy = s * dx + c * dy + cy;
    if (nearest) {
      const ix = Math.round(sx), iy = Math.round(sy);
      if (ix >= 0 && iy >= 0 && ix < W && iy < H) out[y * W + x] = data[iy * W + ix];
    } else {
      const x0 = Math.floor(sx), y0 = Math.floor(sy), fx = sx - x0, fy = sy - y0;
      const px = (xx, yy) => (xx >= 0 && yy >= 0 && xx < W && yy < H ? data[yy * W + xx] : 0);
      out[y * W + x] = Math.round(px(x0, y0) * (1 - fx) * (1 - fy) + px(x0 + 1, y0) * fx * (1 - fy) + px(x0, y0 + 1) * (1 - fx) * fy + px(x0 + 1, y0 + 1) * fx * fy);
    }
  }
  return { data: out, width: W, height: H };
}

/** Minor skew correction: the angle whose row-sum profile is sharpest. */
export function deskew(bw, maxAngle) {
  const { data, width: W, height: H } = bw;
  let n = 0, x0 = W, x1 = -1, y0 = H, y1 = -1;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) if (data[y * W + x]) { n++; if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
  if (n < 50 || (x1 - x0 + 1) < 1.5 * (y1 - y0 + 1)) return { bw, angle: 0 };
  const scale = 256 / Math.max(W, H);
  const small = scale < 1 ? resizeArea(bw, Math.round(W * scale), Math.round(H * scale)) : bw;
  let best = 0, bestScore = -1;
  for (let k = 0; k <= Math.round(2 * maxAngle / 0.5); k++) {
    const ang = -maxAngle + 0.5 * k;
    const r = rotate(small, ang, true), rows = new Float64Array(r.height);
    for (let y = 0; y < r.height; y++) { let sum = 0; for (let x = 0; x < r.width; x++) sum += r.data[y * r.width + x]; rows[y] = sum; }
    const m = mean(rows), v = mean(rows.map((q) => (q - m) ** 2));
    if (v > bestScore) { bestScore = v; best = ang; }
  }
  if (Math.abs(best) < 0.25) return { bw, angle: 0 };
  const rot = rotate(bw, best, false);
  for (let i = 0; i < rot.data.length; i++) rot.data[i] = rot.data[i] > 127 ? 255 : 0;
  return { bw: rot, angle: best };
}

function cropToContent(bw, pad) {
  const { data, width: W, height: H } = bw;
  let x0 = W, x1 = -1, y0 = H, y1 = -1;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) if (data[y * W + x]) { if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
  if (x1 < 0) return { bw, offset: [0, 0] };
  x0 = Math.max(0, x0 - pad); y0 = Math.max(0, y0 - pad); x1 = Math.min(W, x1 + 1 + pad); y1 = Math.min(H, y1 + 1 + pad);
  const w = x1 - x0, h = y1 - y0, out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) out.set(data.subarray((y0 + y) * W + x0, (y0 + y) * W + x0 + w), y * w);
  return { bw: { data: out, width: w, height: h }, offset: [x0, y0] };
}

/** Resize and centre a binary mask (ink != 0) into a size x size binary image. */
export function normalizeGlyph(mask, w, h, size = 28, margin = 0.12) {
  let x0 = w, x1 = -1, y0 = h, y1 = -1;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (mask[y * w + x]) { if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
  if (x1 < 0) return new Uint8Array(size * size);
  const cw = x1 - x0 + 1, ch = y1 - y0 + 1, side = Math.max(cw, ch), border = roundHalfEven(side * margin);
  const S = side + 2 * border, canvas = new Uint8Array(S * S), oy = border + Math.floor((side - ch) / 2), ox = border + Math.floor((side - cw) / 2);
  for (let y = 0; y < ch; y++) for (let x = 0; x < cw; x++) if (mask[(y0 + y) * w + x0 + x]) canvas[(oy + y) * S + ox + x] = 255;
  const small = resizeArea({ data: canvas, width: S, height: S }, size, size);
  return small.data.map((v) => (v >= 80 ? 255 : 0));
}

// --------------------------------------------------------------------------- segmentation
function components(bw) {
  const { data, width: W, height: H } = bw, labels = new Int32Array(W * H), comps = [];
  const stack = new Int32Array(W * H);
  let next = 0;
  for (let i = 0; i < data.length; i++) {
    if (!data[i] || labels[i]) continue;
    next++;
    let top = 0, area = 0, x0 = W, x1 = 0, y0 = H, y1 = 0;
    stack[top++] = i; labels[i] = next;
    while (top) {
      const j = stack[--top], x = j % W, y = (j - x) / W;
      area++; if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
      for (let dy = -1; dy <= 1; dy++) {
        const yy = y + dy;
        if (yy < 0 || yy >= H) continue;
        for (let dx = -1; dx <= 1; dx++) {
          const xx = x + dx;
          if (xx < 0 || xx >= W) continue;
          const k = yy * W + xx;
          if (data[k] && !labels[k]) { labels[k] = next; stack[top++] = k; }
        }
      }
    }
    comps.push({ idx: next, x: x0, y: y0, w: x1 - x0 + 1, h: y1 - y0 + 1, area,
      get cx() { return this.x + this.w / 2; }, get cy() { return this.y + this.h / 2; } });
  }
  return { comps, labels };
}

function classify(comps, p) {
  if (!comps.length) return [[], [], []];
  const areas = comps.map((c) => c.area), p75 = percentile(areas, 75), ref = median(areas.filter((a) => a >= 0.5 * p75));
  const bases = [], marks = [], noise = [];
  for (const c of comps) {
    if (c.area < Math.max(p.min_abs_area, p.noise_ratio * ref)) noise.push(c);
    else if (c.area < p.mark_ratio * ref) marks.push(c);
    else bases.push(c);
  }
  return [bases, marks, noise];
}

function groupLines(bases, p) {
  if (!bases.length) return [];
  const thr = p.line_split_ratio * median(bases.map((b) => b.h)), lines = [];
  for (const b of [...bases].sort((a, c) => a.cy - c.cy)) {
    const last = lines[lines.length - 1];
    if (last && Math.abs(b.cy - mean(last.map((c) => c.cy))) <= thr) last.push(b); else lines.push([b]);
  }
  lines.sort((a, c) => mean(a.map((x) => x.cy)) - mean(c.map((x) => x.cy)));
  return lines.map((l) => [...l].sort((a, c) => a.x - c.x));
}

function assignMarks(lines, marks) {
  const perLine = lines.map(() => new Map()), orphans = [];
  for (const m of marks) {
    let best = null, bestD = null, bestLi = -1;
    lines.forEach((bases, li) => bases.forEach((b) => {
      const ov = Math.min(m.x + m.w, b.x + b.w) - Math.max(m.x, b.x);
      if (ov <= 0) return;
      const d = Math.abs(m.cy - b.cy);
      if (bestD === null || d < bestD) { best = b; bestD = d; bestLi = li; }
    }));
    if (!best) orphans.push(m);
    else { const mp = perLine[bestLi]; if (!mp.has(best.idx)) mp.set(best.idx, []); mp.get(best.idx).push(m); }
  }
  return [perLine, orphans];
}

function splitWords(bases, p) {
  if (!bases.length) return [];
  const medW = median(bases.map((b) => b.w)), words = [];
  let cur = [bases[0]];
  for (let i = 1; i < bases.length; i++) {
    const prev = bases[i - 1], nxt = bases[i], gap = nxt.x - (prev.x + prev.w);
    if (gap > p.word_gap_ratio * medW) { words.push(cur); cur = [nxt]; } else cur.push(nxt);
  }
  words.push(cur);
  return words;
}

const union = (boxes) => { const x0 = Math.min(...boxes.map((b) => b[0])), y0 = Math.min(...boxes.map((b) => b[1])); return [x0, y0, Math.max(...boxes.map((b) => b[0] + b[2])) - x0, Math.max(...boxes.map((b) => b[1] + b[3])) - y0]; };
const bbox = (c) => [c.x, c.y, c.w, c.h];

function maskOf(labels, W, box, ids) {
  const [x, y, w, h] = box, m = new Uint8Array(w * h);
  for (let r = 0; r < h; r++) for (let q = 0; q < w; q++) if (ids.includes(labels[(y + r) * W + x + q])) m[r * w + q] = 1;
  return m;
}

export function segment(bw, p, singleWord = false) {
  const { comps, labels } = components(bw);
  const [bases, marks, noise] = classify(comps, p);
  const lineGroups = singleWord && bases.length ? [[...bases].sort((a, c) => a.x - c.x)] : groupLines(bases, p);
  const [perLine, orphans] = assignMarks(lineGroups, marks);
  const lines = lineGroups.map((group, li) => {
    const wordGroups = singleWord ? [group] : splitWords(group, p);
    return { words: wordGroups.map((wg) => ({ glyphs: wg.map((b) => {
      const bb = bbox(b);
      return { bbox: bb, image: normalizeGlyph(maskOf(labels, bw.width, bb, [b.idx]), b.w, b.h, 28, p.glyph_margin),
        marks: (perLine[li].get(b.idx) || []).map((m) => {
          const ub = union([bb, bbox(m)]);
          return { bbox: bbox(m), image: normalizeGlyph(maskOf(labels, bw.width, ub, [b.idx, m.idx]), ub[2], ub[3], 28, p.glyph_margin) };
        }) };
    }) })) };
  });
  return { lines, orphans: orphans.map(bbox), noise: noise.length };
}

/** Full Section 3.5 preprocessing + segmentation of one grayscale image. */
export function processImage(gray, p, singleWord = false, lightText = null) {
  let bw = binarize(gray, p, lightText);
  const d = deskew(bw, p.deskew_max_angle);
  const c = cropToContent(d.bw, p.crop_pad);
  const res = segment(c.bw, p, singleWord);
  return { ...res, offset: c.offset, angle: d.angle, srcShape: [gray.height, gray.width] };
}

/** Map an (x, y, w, h) box from the processed image back onto the input image. */
export function toSourceBox(res, box) {
  const [x, y, w, h] = box, [ox, oy] = res.offset, [H, W] = res.srcShape;
  let pts = [[x, y], [x + w, y], [x, y + h], [x + w, y + h]].map(([a, b]) => [a + ox, b + oy]);
  if (res.angle) {
    const a = (-res.angle * Math.PI) / 180, c = Math.cos(a), s = Math.sin(a), cx = W / 2, cy = H / 2;
    pts = pts.map(([px, py]) => [c * (px - cx) + s * (py - cy) + cx, -s * (px - cx) + c * (py - cy) + cy]);
  }
  const xs = pts.map((q) => Math.min(W - 1, Math.max(0, q[0]))), ys = pts.map((q) => Math.min(H - 1, Math.max(0, q[1])));
  return [Math.trunc(Math.min(...xs)), Math.trunc(Math.min(...ys)), Math.trunc(Math.max(...xs)) , Math.trunc(Math.max(...ys))];
}

// --------------------------------------------------------------------------- models
/** CNN forward pass on a 28x28 binary image (ink 255). Returns class probabilities. */
export function cnnPredict(model, img28) {
  let C = 1, H = 28, W = 28, x = Float32Array.from(img28, (v) => v / 255);
  for (const l of model.layers) {
    if (l.type === "conv3x3_relu") {
      const O = l.weight.shape[0], out = new Float32Array(O * H * W), w = l.w;
      for (let o = 0; o < O; o++) {
        const bias = l.b[o], ob = o * H * W;
        for (let i = 0; i < H * W; i++) out[ob + i] = bias;
        for (let c = 0; c < C; c++) {
          const xb = c * H * W, wb = (o * C + c) * 9;
          for (let ky = 0; ky < 3; ky++) for (let kx = 0; kx < 3; kx++) {
            const wv = w[wb + ky * 3 + kx];
            if (wv === 0) continue;
            const dy = ky - 1, dx = kx - 1;
            for (let yy = Math.max(0, -dy); yy < Math.min(H, H - dy); yy++) {
              const src = xb + (yy + dy) * W + dx, dst = ob + yy * W;
              for (let xx = Math.max(0, -dx); xx < Math.min(W, W - dx); xx++) out[dst + xx] += wv * x[src + xx];
            }
          }
        }
      }
      for (let i = 0; i < out.length; i++) if (out[i] < 0) out[i] = 0;
      x = out; C = O;
    } else if (l.type === "maxpool2") {
      const H2 = Math.floor(H / 2), W2 = Math.floor(W / 2), out = new Float32Array(C * H2 * W2);
      for (let c = 0; c < C; c++) for (let y = 0; y < H2; y++) for (let q = 0; q < W2; q++) {
        const b = c * H * W + 2 * y * W + 2 * q;
        out[c * H2 * W2 + y * W2 + q] = Math.max(x[b], x[b + 1], x[b + W], x[b + W + 1]);
      }
      x = out; H = H2; W = W2;
    } else {
      const [O, I] = l.weight.shape, out = new Float32Array(O);
      for (let o = 0; o < O; o++) { let s = l.b[o]; const wb = o * I; for (let i = 0; i < I; i++) s += l.w[wb + i] * x[i]; out[o] = s; }
      if (l.type === "dense_relu") for (let i = 0; i < O; i++) if (out[i] < 0) out[i] = 0;
      x = out;
    }
  }
  const m = Math.max(...x), e = Array.from(x, (v) => Math.exp(v - m)), z = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / z);
}

/** Baybayin-vs-not SVM (RBF) on a 28x28 binary image. */
export function isBaybayin(svm, img28) {
  const n = svm.n_features, xb = new Uint8Array(svm.rowBytes);
  for (let j = 0; j < n; j++) if (img28[j]) xb[j >> 3] |= 0x80 >> (j & 7);
  let dec = svm.intercept;
  for (let s = 0; s < svm.n_support; s++) {
    const row = s * svm.rowBytes;
    let d2 = 0;
    for (let byte = 0; byte < svm.rowBytes; byte++) {
      let diff = svm.bits[row + byte] ^ xb[byte];
      while (diff) {
        const bit = 31 - Math.clz32(diff);   // bit 7 = first feature of this byte
        d2 += svm.weight[byte * 8 + (7 - bit)];
        diff &= ~(1 << bit);
      }
    }
    dec += svm.dual_coef[s] * Math.exp(-svm.gamma * d2);
  }
  return dec > 0;
}

// --------------------------------------------------------------------------- readings, word list, ranking
const CLASS_CONSONANT = { ba: ["b"], dara: ["d", "r"], ga: ["g"], ha: ["h"], ka: ["k"], la: ["l"], ma: ["m"], na: ["n"],
  nga: ["ng"], pa: ["p"], sa: ["s"], ta: ["t"], wa: ["w"], ya: ["y"] };
const VOWEL_CLASS = { a: ["a"], ei: ["e", "i"], ou: ["o", "u"] };
const MARK_VOWELS = { none: ["a"], above: ["e", "i"], below: ["o", "u"], virama: [""] };
const UNI_BASE = { a: "ᜀ", ei: "ᜁ", ou: "ᜂ", ka: "ᜃ", ga: "ᜄ", nga: "ᜅ", ta: "ᜆ",
  dara: "ᜇ", na: "ᜈ", pa: "ᜉ", ba: "ᜊ", ma: "ᜋ", ya: "ᜌ", la: "ᜎ", wa: "ᜏ",
  sa: "ᜐ", ha: "ᜑ" };
const UNI_MARK = { none: "", above: "ᜒ", below: "ᜓ", virama: "᜔" };

const syllableOptions = (g) => (g.cls in VOWEL_CLASS ? [...VOWEL_CLASS[g.cls]]
  : CLASS_CONSONANT[g.cls].flatMap((c) => MARK_VOWELS[g.mark].map((v) => c + v)));
export const baybayinUnicode = (obs) => obs.map((g) => UNI_BASE[g.cls] + (g.cls in CLASS_CONSONANT ? UNI_MARK[g.mark] : "")).join("");
export const mostProbableReading = (obs) => obs.map((g) => {
  const opts = syllableOptions(g), pref = opts.filter((o) => !o.startsWith("r") && !/[eo]$/.test(o));
  return (pref.length ? pref : opts)[0];
}).join("");

function searchWord(obs, words, maxCandidates) {
  if (!obs.length) return [];
  const options = obs.map(syllableOptions), found = [];
  const idx = new Array(options.length).fill(0);
  for (let n = 0; n < maxCandidates; n++) {   // itertools.product order (last position fastest)
    const w = idx.map((k, i) => options[i][k]).join("");
    if (words.has(w) && !found.includes(w)) found.push(w);
    let i = options.length - 1;
    while (i >= 0 && ++idx[i] === options[i].length) { idx[i] = 0; i--; }
    if (i < 0) break;
  }
  return found.length ? found.map((w) => ({ word: w, unverified: false })) : [{ word: mostProbableReading(obs), unverified: true }];
}

function lmLogp(lm, words, w, prev) {
  const key = (x) => (words.has(x) ? x : "<unk>");
  const pu = ((lm.unigram[key(w)] || 0) + lm.k) / (lm.total + lm.k * (lm.vocab_size + 1));
  if (prev === null) return Math.log(pu);
  const cprev = lm.prev_count[key(prev)] || 0, pb = cprev ? (lm.bigram[`${key(prev)}\t${key(w)}`] || 0) / cprev : 0;
  const lam = cprev ? lm.lam : 0;
  return Math.log(lam * pb + (1 - lam) * pu);
}

function rankSequence(wordCands, M) {
  let hyps = [[0, []]];
  for (const cands of wordCands) {
    if (!cands.length) continue;
    const nxt = [];
    for (const [sc, seq] of hyps) {
      const prev = seq.length ? seq[seq.length - 1].word : null;
      for (const c of cands) nxt.push([sc + lmLogp(M.lm, M.words, c.word, prev), [...seq, c]]);
    }
    hyps = nxt.sort((a, b) => b[0] - a[0]).slice(0, M.params.sentence_beam);
  }
  return hyps;
}

/** Dictionary English: single word -> its entry; several words -> first sense of each (fallback). */
export function dictionaryEnglish(M, words) {
  if (!words.length) return "";
  if (words.length === 1) return M.dict[words[0]] || "";
  return words.map((w) => (M.dict[w] || `[${w}]`).split(";")[0]).join(" ");
}

// --------------------------------------------------------------------------- pipeline
function observations(M, res) {
  const lines = [], kept = [], rejected = [];
  for (const line of res.lines) {
    const wordsObs = [], lineWords = [];
    for (const w of line.words) {
      if (!w.glyphs.length) continue;
      const share = w.glyphs.filter((g) => isBaybayin(M.svm, g.image)).length / w.glyphs.length;
      if (share < M.params.script_min_share) { rejected.push(w); continue; }
      lineWords.push(w);
      wordsObs.push(w.glyphs.map((g) => {
        const probs = cnnPredict(M.cnn, g.image), k = probs.indexOf(Math.max(...probs)), cls = M.cnn.classes[k];
        let mark = "none";
        if (g.marks.length && cls in CLASS_CONSONANT) {
          const m = g.marks.reduce((a, b) => (b.bbox[2] * b.bbox[3] > a.bbox[2] * a.bbox[3] ? b : a));
          const dp = cnnPredict(M.diacritic, m.image);
          mark = M.diacritic.classes[dp.indexOf(Math.max(...dp))];
        }
        return { cls, mark, prob: probs[k] };
      }));
    }
    if (wordsObs.length) { lines.push(wordsObs); kept.push(lineWords); }
  }
  return { lines, kept, rejected };
}

const wordBox = (w) => union(w.glyphs.flatMap((g) => [g.bbox, ...g.marks.map((m) => m.bbox)]));
const nChars = (lines) => lines.reduce((s, l) => s + l.reduce((t, w) => t + w.length, 0), 0);

/** Grayscale image -> recognised Baybayin, transliteration, Tagalog, English (dictionary). */
export function recognize(gray, singleWord = false) {
  const M = loadModels(), p = M.params.preprocess;
  let res = processImage(gray, p, singleWord), obs = observations(M, res);
  // Text may be darker or lighter than its background: also read the other side
  // of the Otsu threshold and keep whichever yields more Baybayin characters.
  for (const light of [true, false]) {
    const alt = processImage(gray, p, singleWord, light), altObs = observations(M, alt);
    if (nChars(altObs.lines) > nChars(obs.lines)) { res = alt; obs = altObs; }
  }
  const words = [], baybayin = [], tagalog = [], translit = [], english = [];
  obs.lines.forEach((wordsObs, li) => {
    const cands = wordsObs.map((w) => searchWord(w, M.words, M.params.max_candidates));
    const best = rankSequence(cands, M)[0]?.[1] || [];
    const chosen = best.map((c) => c.word), unverified = new Set(best.filter((c) => c.unverified).map((c) => c.word));
    wordsObs.forEach((g, wi) => {
      words.push({ line: li, baybayin: baybayinUnicode(g), reading: mostProbableReading(g), tagalog: chosen[wi] || "",
        verified: !unverified.has(chosen[wi]), confidence: Math.min(...g.map((o) => o.prob)),
        box: toSourceBox(res, wordBox(obs.kept[li][wi])) });
    });
    baybayin.push(wordsObs.map(baybayinUnicode).join(" "));
    translit.push(wordsObs.map(mostProbableReading).join(" "));
    tagalog.push(chosen.join(" "));
    english.push(dictionaryEnglish(M, chosen));
  });
  return { baybayin: baybayin.join("\n"), transliteration: translit.join("\n"), tagalog: tagalog.join("\n"),
    english: english.join("\n"), words, unverified: words.filter((w) => !w.verified).map((w) => w.tagalog),
    ignored: obs.rejected.map((w) => ({ box: toSourceBox(res, wordBox(w)), reason: "not Baybayin" })) };
}
