/* ============================================================================
 * pg_engine.js — Engine กลางสำหรับสนามทดลอง (playground) ทุกตัวในหนังสือ ML
 * โหลดด้วย <script src="pg_engine.js"></script> (global window.PG — ไม่ใช่ ES module
 * เพื่อให้เปิดผ่าน file:// ได้)
 *
 * โมดูล:
 *   PG.colors      — สีมาตรฐาน (data/model/target/...)
 *   PG.rng(seed)   — เครื่องสุ่มแบบมี seed (reproducible) + gauss noise
 *   PG.Plot(...)   — ระบบพิกัด data↔pixel บน canvas (รองรับจอ retina)
 *   PG.draw        — primitive การวาด: grid/curve/points/squares/band/coefBars
 *   PG.targets     — คลังฟังก์ชันเป้าหมาย (line/quad/sine/...)
 *   PG.scenario    — ตัวสร้างข้อมูล: regression(sample) + classification
 *   PG.fit         — least-squares / ridge polynomial (+ solve)
 *   PG.metrics     — RMSE/MSE/MAE/R²/E_in/E_out/‖w‖²
 * ========================================================================== */
(function (global) {
'use strict';

/* ---------- สีมาตรฐาน (CLAUDE.md): data แดง · model น้ำเงิน · target เขียว --- */
const colors = {
  data:   '#E07A5F',                 // coral (pastel soft-muted)
  model:  '#5B8FB9',                 // soft blue
  target: '#6BA38C',                 // sage
  band:   'rgba(91,143,185,0.16)',   // variance — soft blue จาง
  square: 'rgba(176,122,174,0.22)',  // error²/SSE — mauve
  squareEdge: 'rgba(176,122,174,0.65)',
  grid:   '#eef1f5',
  axis:   '#c7cdd6',
  bias:   '#9aa3b2',                 // bias coefficient bar (w0) — เทา
};

/* ---------- RNG แบบมี seed (mulberry32) — สำคัญกับ variance/CV ------------- */
function rng(seed) {
  let s = (seed >>> 0) || 1;
  const next = () => {
    s = (s + 0x6D2B79F5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  // Gaussian โดยประมาณ (ผลรวม uniform 4 ตัว) — เข้ากับพฤติกรรมเดิมของ playground
  const gauss = () => (next() + next() + next() + next() - 2) / 2;
  return { next, gauss };
}

/* ---------- ระบบพิกัด + canvas (รองรับ devicePixelRatio ให้เส้นคม) -------- */
function Plot(canvas, opts) {
  const o = Object.assign({ X0: 0, X1: 1, Y0: -1.6, Y1: 1.6, margin: 40 }, opts);
  const ctx = canvas.getContext('2d');
  // ขนาดเชิงตรรกะ = "กล่องเนื้อใน" ของ canvas เอง (clientWidth/Height)
  // ⚠️ ห้ามยึดจาก parent: pg_style มี *{box-sizing:border-box} + canvas{border:1px}
  // → ถ้าตั้ง backing store = ขนาด parent แล้ว style.width เท่ากัน พื้นที่วาดจริงจะเล็กกว่า 2px
  //   ภาพถูกบีบด้วยอัตราส่วนเศษ → เส้น/ตัวหนังสือเบลอทั้งแคนวาส
  const cw = Math.round(canvas.clientWidth), chh = Math.round(canvas.clientHeight);
  const W = cw > 0 ? cw : canvas.width, H = chh > 0 ? chh : canvas.height;
  const dpr = global.devicePixelRatio || 1;
  canvas.width = Math.round(W * dpr);
  canvas.height = Math.round(H * dpr);
  if (dpr !== 1) ctx.setTransform(canvas.width / W, 0, 0, canvas.height / H, 0, 0);
  const innerW = W - 2 * o.margin, innerH = H - 2 * o.margin;
  return {
    ctx, W, H, X0: o.X0, X1: o.X1, Y0: o.Y0, Y1: o.Y1, margin: o.margin,
    px: x => o.margin + (x - o.X0) / (o.X1 - o.X0) * innerW,
    py: y => H - o.margin - (y - o.Y0) / (o.Y1 - o.Y0) * innerH,
    ux: sx => o.X0 + (sx - o.margin) / innerW * (o.X1 - o.X0),
    uy: sy => o.Y0 + (H - o.margin - sy) / innerH * (o.Y1 - o.Y0),
    clear: () => ctx.clearRect(0, 0, W, H),
  };
}

/* ---------- responsive: ปรับขนาด canvas ตาม container แล้วสร้าง Plot ใหม่ ----- */
function fitPlot(canvas, opts) {
  const o = opts || {};
  const box = o.box || canvas.parentElement || canvas;
  const cssW = Math.max(o.minW || 260, Math.round(box.clientWidth || canvas.width || 640));
  let cssH = o.height || Math.round(cssW * (o.aspect || 0.67));
  if (!o.height) { var _cap = Math.round((global.innerHeight || 700) * 0.46); if (cssH > _cap) cssH = _cap; }
  canvas.style.width = cssW + 'px';
  canvas.style.height = cssH + 'px';
  canvas.width = cssW; canvas.height = cssH;   // logical; Plot คูณ dpr ให้คม
  return Plot(canvas, o);
}

/* ---------- primitive การวาด ----------------------------------------------- */
const draw = {
  // เส้นตาราง + แกน · opts:{stepX,stepY,xAxis,yAxis}
  grid(p, opts) {
    const c = Object.assign({ stepX: 0.25, stepY: 0.5, xAxis: true, yAxis: false }, opts);
    const ctx = p.ctx;
    p.clear();
    ctx.strokeStyle = colors.grid; ctx.lineWidth = 1;
    const eps = 1e-9;
    for (let gx = Math.ceil(p.X0 / c.stepX) * c.stepX; gx <= p.X1 + eps; gx += c.stepX) {
      ctx.beginPath(); ctx.moveTo(p.px(gx), p.py(p.Y0)); ctx.lineTo(p.px(gx), p.py(p.Y1)); ctx.stroke();
    }
    for (let gy = Math.ceil(p.Y0 / c.stepY) * c.stepY; gy <= p.Y1 + eps; gy += c.stepY) {
      ctx.beginPath(); ctx.moveTo(p.px(p.X0), p.py(gy)); ctx.lineTo(p.px(p.X1), p.py(gy)); ctx.stroke();
    }
    ctx.strokeStyle = colors.axis;
    if (c.xAxis && p.Y0 < 0 && p.Y1 > 0) { ctx.beginPath(); ctx.moveTo(p.px(p.X0), p.py(0)); ctx.lineTo(p.px(p.X1), p.py(0)); ctx.stroke(); }
    if (c.yAxis && p.X0 < 0 && p.X1 > 0) { ctx.beginPath(); ctx.moveTo(p.px(0), p.py(p.Y0)); ctx.lineTo(p.px(0), p.py(p.Y1)); ctx.stroke(); }
  },

  // วาดเส้นโค้งของฟังก์ชัน fn (ตัดส่วนที่หลุดกรอบ) · opts:{color,width,steps}
  curve(p, fn, opts) {
    const c = Object.assign({ color: colors.model, width: 2.5, steps: 300 }, opts);
    const ctx = p.ctx;
    ctx.strokeStyle = c.color; ctx.lineWidth = c.width; ctx.beginPath();
    let first = true;
    for (let i = 0; i <= c.steps; i++) {
      const x = p.X0 + (p.X1 - p.X0) * i / c.steps, sy = p.py(fn(x));
      if (sy < -60 || sy > p.H + 60) { first = true; continue; }
      if (first) { ctx.moveTo(p.px(x), sy); first = false; } else ctx.lineTo(p.px(x), sy);
    }
    ctx.stroke();
  },

  // จุดข้อมูล (วงกลมขอบขาว)
  points(p, pts, opts) {
    const c = Object.assign({ color: colors.data, r: 4.5 }, opts);
    const ctx = p.ctx;
    for (const pt of pts) {
      ctx.fillStyle = c.color; ctx.beginPath(); ctx.arc(p.px(pt.x), p.py(pt.y), c.r, 0, 7); ctx.fill();
      ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5; ctx.stroke();
    }
  },

  // พื้นที่สี่เหลี่ยม = ค่าคลาดเคลื่อน² (วาดในพิกัด pixel ให้ดูเป็นสี่เหลี่ยมจัตุรัส)
  squares(p, pts, predict) {
    const ctx = p.ctx;
    ctx.fillStyle = colors.square; ctx.strokeStyle = colors.squareEdge; ctx.lineWidth = 1;
    for (const pt of pts) {
      const sx = p.px(pt.x), syD = p.py(pt.y), syP = p.py(predict(pt.x)), side = Math.abs(syD - syP);
      ctx.fillRect(sx, Math.min(syD, syP), side, side);
      ctx.strokeRect(sx, Math.min(syD, syP), side, side);
    }
  },

  // แถบความแปรปรวน (สำหรับ playground bias-variance): lo(x),hi(x)
  band(p, lo, hi, opts) {
    const c = Object.assign({ color: colors.band, steps: 120 }, opts);
    const ctx = p.ctx;
    ctx.fillStyle = c.color; ctx.beginPath();
    for (let i = 0; i <= c.steps; i++) { const x = p.X0 + (p.X1 - p.X0) * i / c.steps; const sy = p.py(hi(x)); i ? ctx.lineTo(p.px(x), sy) : ctx.moveTo(p.px(x), sy); }
    for (let i = c.steps; i >= 0; i--) { const x = p.X0 + (p.X1 - p.X0) * i / c.steps; ctx.lineTo(p.px(x), p.py(lo(x))); }
    ctx.closePath(); ctx.fill();
  },

  // แผนภูมิแท่งขนาดสัมประสิทธิ์ |wⱼ| บน canvas แยก (w0 = bias เป็นสีเทา)
  coefBars(canvas, weights) {
    // ตั้ง backing store จากกล่องเนื้อในของ canvas เอง (ไม่งั้น attribute เดิม เช่น 280×90 จะถูก CSS ยืด → เบลอ/ผิดสัดส่วน)
    const ctx = canvas.getContext('2d');
    const dpr = global.devicePixelRatio || 1;
    const W = Math.round(canvas.clientWidth) || canvas.width, H = Math.round(canvas.clientHeight) || canvas.height;
    canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
    ctx.setTransform(canvas.width / W, 0, 0, canvas.height / H, 0, 0);
    ctx.clearRect(0, 0, W, H);
    if (!weights.length) return;
    const n = weights.length, gap = 4, bw = (W - gap * (n + 1)) / n;
    const mx = Math.max(0.2, ...weights.map(w => Math.abs(w)));
    for (let j = 0; j < n; j++) {
      const h = Math.abs(weights[j]) / mx * (H - 18);
      ctx.fillStyle = j === 0 ? colors.bias : colors.model;
      ctx.fillRect(gap + j * (bw + gap), H - 14 - h, bw, h);
      ctx.fillStyle = '#6b7280'; ctx.font = '14px KaTeX_Math'; ctx.textAlign = 'center';
      ctx.fillText('w' + j, gap + j * (bw + gap) + bw / 2, H - 3);
    }
  },
};

/* ---------- คลังฟังก์ชันเป้าหมาย ------------------------------------------- */
const targets = {
  line:    x => 1.5 * x - 0.75,                          // โดเมน [0,1]
  quad:    x => 2 * (x - 0.5) * (x - 0.5) * 1.6 - 0.55,  // พาราโบลา [0,1]
  sine:    x => 0.85 * Math.sin(2 * Math.PI * x),        // ไซน์ [0,1]
  step:    x => (x < 0.5 ? -0.6 : 0.6),                  // ขั้นบันได [0,1]
  sineSym: x => 0.8 * Math.sin(Math.PI * x),             // ไซน์สมมาตร [-1,1] (Bishop)
  polySym: x => 2 * x * x - 0.6,                         // พหุนามสมมาตร [-1,1]
};

/* ---------- ตัวสร้างข้อมูล -------------------------------------------------- */
const scenario = {
  /* สุ่มจุดจากฟังก์ชันเป้าหมาย trueFn (regression)
   * opts: {X0,X1,N,noise,seed,grid,jitterX,sorted}
   *   grid=false → x สุ่มสม่ำเสมอ · grid=true → x เรียงเท่ากัน + jitter เล็กน้อย */
  sample(trueFn, opts) {
    const o = Object.assign({ X0: 0, X1: 1, N: 12, noise: 0.18, seed: 1, grid: false, jitterX: 0, sorted: true }, opts);
    const r = rng(o.seed), pts = [];
    for (let i = 0; i < o.N; i++) {
      let x = o.grid
        ? o.X0 + (o.X1 - o.X0) * i / (o.N - 1) + r.gauss() * o.jitterX
        : o.X0 + (o.X1 - o.X0) * r.next();
      x = Math.max(o.X0, Math.min(o.X1, x));
      pts.push({ x, y: trueFn(x) + r.gauss() * o.noise });
    }
    if (o.sorted) pts.sort((a, b) => a.x - b.x);
    return pts;
  },

  /* ข้อมูลจำแนก 2 มิติ (เตรียมไว้สำหรับ Classifier Playground)
   * name: 'blobs'|'circles'|'spiral'|'xor' · คืน [{x,y,label}] (label 0/1) */
  classification(name, opts) {
    const o = Object.assign({ N: 120, overlap: 1, imbalance: 0.5, seed: 1, R: 1 }, opts);
    const r = rng(o.seed), pts = [];
    const n1 = Math.round(o.N * o.imbalance), n0 = o.N - n1;
    const push = (x, y, label) => pts.push({ x, y, label });
    if (name === 'blobs') {
      const sep = 1.1, sd = 0.45 * o.overlap;
      for (let i = 0; i < n0; i++) push(-sep + r.gauss() * sd, r.gauss() * sd, 0);
      for (let i = 0; i < n1; i++) push(+sep + r.gauss() * sd, r.gauss() * sd, 1);
    } else if (name === 'circles') {
      const sd = 0.12 * o.overlap;
      for (let i = 0; i < n0; i++) { const a = r.next() * 2 * Math.PI, rad = 0.4 + r.gauss() * sd; push(rad * Math.cos(a) * 2.2, rad * Math.sin(a) * 2.2, 0); }
      for (let i = 0; i < n1; i++) { const a = r.next() * 2 * Math.PI, rad = 1.0 + r.gauss() * sd; push(rad * Math.cos(a) * 2.2, rad * Math.sin(a) * 2.2, 1); }
    } else if (name === 'spiral') {
      const sd = 0.1 * o.overlap;
      for (let c = 0; c < 2; c++) {
        const cnt = c === 0 ? n0 : n1;
        for (let i = 0; i < cnt; i++) {
          const t = i / cnt * 3.2, rad = t * 0.6;
          const a = t * 2.2 + c * Math.PI + r.gauss() * sd;
          push(rad * Math.cos(a), rad * Math.sin(a), c);
        }
      }
    } else if (name === 'xor') {
      const sd = 0.4 * o.overlap;
      for (let i = 0; i < o.N; i++) {
        const qx = r.next() < 0.5 ? -1 : 1, qy = r.next() < 0.5 ? -1 : 1;
        push(qx + r.gauss() * sd, qy + r.gauss() * sd, qx * qy > 0 ? 1 : 0);
      }
    }
    return pts;
  },
};

/* ---------- พีชคณิตเชิงเส้น + การฟิตพหุนาม ---------------------------------- */
function solve(A, b) {
  const n = b.length;
  A = A.map(r => r.slice()); b = b.slice();
  for (let c = 0; c < n; c++) {
    let piv = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[piv][c])) piv = r;
    [A[c], A[piv]] = [A[piv], A[c]]; [b[c], b[piv]] = [b[piv], b[c]];
    const d = A[c][c]; if (Math.abs(d) < 1e-12) continue;
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = A[r][c] / d;
      for (let j = c; j < n; j++) A[r][j] -= f * A[c][j];
      b[r] -= f * b[c];
    }
  }
  return b.map((v, i) => Math.abs(A[i][i]) < 1e-12 ? 0 : v / A[i][i]);
}

const fit = {
  solve,
  /* ฟิตพหุนาม ridge ผ่าน normal equations — ไม่ลงโทษ bias (w0)
   * opts:{degree, lambda} · คืน {weights, predict} */
  poly(points, opts) {
    const o = Object.assign({ degree: 3, lambda: 0 }, opts);
    const m = o.degree + 1;
    if (!points.length) return { weights: [], predict: () => 0 };
    const XtX = Array.from({ length: m }, () => Array(m).fill(0)), Xty = Array(m).fill(0);
    for (const p of points) {
      const phi = []; let v = 1;
      for (let j = 0; j < m; j++) { phi.push(v); v *= p.x; }
      for (let a = 0; a < m; a++) { Xty[a] += phi[a] * p.y; for (let c = 0; c < m; c++) XtX[a][c] += phi[a] * phi[c]; }
    }
    for (let j = 0; j < m; j++) XtX[j][j] += 1e-9;       // เสถียรภาพเชิงตัวเลข
    for (let j = 1; j < m; j++) XtX[j][j] += o.lambda;   // weight decay (ข้าม bias)
    const weights = solve(XtX, Xty);
    const predict = x => { let v = 1, s = 0; for (let j = 0; j < weights.length; j++) { s += weights[j] * v; v *= x; } return s; };
    return { weights, predict };
  },
};

/* ---------- ตัววัดผล ------------------------------------------------------- */
const metrics = {
  /* points, model({predict,weights}), trueFn|null, dom{X0,X1}
   * คืน {mse,rmse,mae,r2,ein,eout,eoutRmse,wnorm}
   *   ein = mse (ในตัวอย่าง) · eout = MSE เทียบ trueFn · eoutRmse = √eout */
  regression(points, model, trueFn, dom) {
    const predict = model.predict, weights = model.weights || [];
    let se = 0, ae = 0;
    for (const p of points) { const e = predict(p.x) - p.y; se += e * e; ae += Math.abs(e); }
    const n = points.length || 1;
    const mse = se / n, rmse = Math.sqrt(mse), mae = ae / n;
    const ym = points.reduce((a, p) => a + p.y, 0) / n;
    let sst = 0; for (const p of points) sst += (p.y - ym) ** 2;
    const r2 = sst < 1e-9 ? 0 : 1 - se / sst;
    let eout = null, eoutRmse = null;
    if (trueFn) {
      const X0 = dom.X0, X1 = dom.X1, K = 200; let s = 0;
      for (let i = 0; i < K; i++) { const x = X0 + (X1 - X0) * i / (K - 1); s += (predict(x) - trueFn(x)) ** 2; }
      eout = s / K; eoutRmse = Math.sqrt(eout);
    }
    let wn = 0; for (let j = 1; j < weights.length; j++) wn += weights[j] ** 2;
    return { mse, rmse, mae, r2, ein: mse, eout, eoutRmse, wnorm: wn };
  },
};

/* ==========================================================================
 * ส่วนเสริมการจำแนก (classification) — ใช้โดย Classifier / Confusion / ROC
 * ========================================================================== */
// สีของชั้น (class): A = label 1 (แดง) · B = label 0 (น้ำเงิน) + โทนพื้นที่ตัดสินใจ
colors.classA = '#E07A5F'; colors.classB = '#5B8FB9';   // pastel: positive=coral, negative=soft-blue

/* --- ตัวจำแนก: คืน {score(x,y)→ความน่าจะเป็นของ label 1} --- */
fit.logistic = function (points, opts) {
  const o = Object.assign({ lr: 0.5, iters: 400, l2: 0.0 }, opts);
  let w = [0, 0, 0];
  const N = points.length || 1;
  for (let it = 0; it < o.iters; it++) {
    let g0 = 0, g1 = 0, g2 = 0;
    for (const p of points) {
      const s = 1 / (1 + Math.exp(-(w[0] + w[1] * p.x + w[2] * p.y)));
      const e = s - p.label; g0 += e; g1 += e * p.x; g2 += e * p.y;
    }
    w[0] -= o.lr * (g0 / N);
    w[1] -= o.lr * (g1 / N + o.l2 * w[1]);
    w[2] -= o.lr * (g2 / N + o.l2 * w[2]);
  }
  return { weights: w, score: (x, y) => 1 / (1 + Math.exp(-(w[0] + w[1] * x + w[2] * y))) };
};
fit.knn = function (points, k) {
  return { score: (x, y) => {
    const ds = points.map(p => ({ d: (p.x - x) ** 2 + (p.y - y) ** 2, l: p.label }));
    ds.sort((a, b) => a.d - b.d);
    const kk = Math.min(k, ds.length); let c = 0;
    for (let i = 0; i < kk; i++) c += ds[i].l;
    return kk ? c / kk : 0;
  } };
};

/* --- วาดพื้นที่ตัดสินใจ (เซลล์โปร่งแสง เข้มตามความมั่นใจ) + จุดตามชั้น --- */
draw.boundary = function (p, scoreFn, opts) {
  const o = Object.assign({ threshold: 0.5, cell: 5 }, opts), ctx = p.ctx;
  for (let sx = p.margin; sx < p.W - p.margin; sx += o.cell)
    for (let sy = p.margin; sy < p.H - p.margin; sy += o.cell) {
      const s = scoreFn(p.ux(sx + o.cell / 2), p.uy(sy + o.cell / 2));
      const a = Math.min(0.34, 0.10 + Math.abs(s - 0.5) * 0.55);
      ctx.fillStyle = (s >= o.threshold ? `rgba(224,122,95,${a})` : `rgba(91,143,185,${a})`);   // pastel: predicted +=coral · −=soft-blue
      ctx.fillRect(sx, sy, o.cell, o.cell);
    }
};
draw.classPoints = function (p, points, opts) {
  const o = Object.assign({ r: 4.5 }, opts), ctx = p.ctx;
  for (const pt of points) {
    ctx.fillStyle = pt.label ? colors.classA : colors.classB;
    ctx.beginPath(); ctx.arc(p.px(pt.x), p.py(pt.y), o.r, 0, 7); ctx.fill();
    ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.3; ctx.stroke();
  }
};

/* --- ข้อมูลคะแนน 1 มิติ: สอง Gaussian (สำหรับ threshold/ROC/PR) --- */
scenario.scores = function (opts) {
  const o = Object.assign({ n: 200, sep: 2, spread: 1, imbalance: 0.5, seed: 1 }, opts);
  const r = rng(o.seed), out = [];
  const n1 = Math.round(o.n * o.imbalance), n0 = o.n - n1;
  for (let i = 0; i < n0; i++) out.push({ score: -o.sep / 2 + r.gauss() * o.spread, label: 0 });
  for (let i = 0; i < n1; i++) out.push({ score: +o.sep / 2 + r.gauss() * o.spread, label: 1 });
  return out;
};

/* --- เมทริกการจำแนก: ตารางความสับสน + ROC/PR + AUC --- */
metrics.classification = function (items, threshold) {
  let tp = 0, fp = 0, fn = 0, tn = 0;
  for (const it of items) {
    const pred = it.score >= threshold ? 1 : 0;
    if (it.label === 1) (pred ? tp++ : fn++); else (pred ? fp++ : tn++);
  }
  const tpr = (tp + fn) ? tp / (tp + fn) : 0, fpr = (fp + tn) ? fp / (fp + tn) : 0;
  const prec = (tp + fp) ? tp / (tp + fp) : 0;
  const f1 = (prec + tpr) ? 2 * prec * tpr / (prec + tpr) : 0;
  return { tp, fp, fn, tn, acc: (tp + tn) / (items.length || 1), tpr, fpr, precision: prec, recall: tpr, f1 };
};
metrics.rocPoints = function (items) {
  const ths = [...new Set(items.map(i => i.score))].sort((a, b) => b - a);
  const pts = [{ fpr: 0, tpr: 0, th: Infinity }];
  for (const t of ths) { const m = metrics.classification(items, t); pts.push({ fpr: m.fpr, tpr: m.tpr, th: t }); }
  pts.push({ fpr: 1, tpr: 1, th: -Infinity });
  return pts;
};
metrics.prPoints = function (items) {
  const ths = [...new Set(items.map(i => i.score))].sort((a, b) => b - a), pts = [];
  for (const t of ths) { const m = metrics.classification(items, t); pts.push({ recall: m.recall, precision: m.precision, th: t }); }
  return pts;
};
metrics.auc = function (roc) {
  const p = [...roc].sort((a, b) => a.fpr - b.fpr); let a = 0;
  for (let i = 1; i < p.length; i++) a += (p[i].fpr - p[i - 1].fpr) * (p[i].tpr + p[i - 1].tpr) / 2;
  return a;
};

global.PG = { colors, rng, Plot, fitPlot, draw, targets, scenario, fit, metrics };

/* auto-size .mbar fills from their .mval text (same-scale metric bars).
   each .mbar has data-max (default 1); fill width = clamp(value/max)*100 percent.
   value cell carries class .mval plus the id existing JS writes to, so no per-file JS. */
if (typeof document !== 'undefined') (function () {
  function sizeBars() {
    document.querySelectorAll('.mbar').forEach(function (b) {
      var v = b.querySelector('.mval'), f = b.querySelector('.fill'); if (!v || !f) return;
      var num = parseFloat((v.textContent || '').replace(/[^0-9.\-]/g, ''));
      var max = parseFloat(b.getAttribute('data-max') || '1');
      f.style.width = isNaN(num) ? '0' : (Math.max(0, Math.min(1, num / max)) * 100).toFixed(1) + '%';
    });
  }
  function init() {
    sizeBars();
    document.querySelectorAll('.mbar .mval').forEach(function (v) {
      new MutationObserver(sizeBars).observe(v, { childList: true, characterData: true, subtree: true });
    });
    // segmented controls: .seg[data-for] buttons drive a hidden <select> (template style, not dropdown)
    document.querySelectorAll('.seg[data-for]').forEach(function (seg) {
      var sel = document.getElementById(seg.getAttribute('data-for')); if (!sel) return;
      function sync(){ seg.querySelectorAll('button').forEach(function(b){ var on=b.getAttribute('data-val')===sel.value; b.style.background=on?'#1e3a8a':'#fff'; b.style.color=on?'#fff':'#555'; }); }
      seg.querySelectorAll('button').forEach(function (b) {
        b.addEventListener('click', function () { sel.value=b.getAttribute('data-val'); sel.dispatchEvent(new Event('change',{bubbles:true})); sync(); });
      });
      sel.addEventListener('change', sync); sync();
    });
  }
  if (document.readyState !== 'loading') init(); else document.addEventListener('DOMContentLoaded', init);
})();

})(typeof window !== 'undefined' ? window : this);
