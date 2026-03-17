// ══════════════════════════════════════════════════════
// BG EMOJIS
// ══════════════════════════════════════════════════════
(function() {
  const box = document.getElementById('bgEmojis');
  const emojis = ['⭐','✨','🌟','💫','🎈','🌈','🍭','🎀'];
  for (let i = 0; i < 14; i++) {
    const el = document.createElement('div');
    el.className = 'bg-emoji';
    el.textContent = emojis[i % emojis.length];
    el.style.left = (Math.random() * 100) + 'vw';
    el.style.fontSize = (18 + Math.random() * 22) + 'px';
    el.style.animationDuration = (10 + Math.random() * 14) + 's';
    el.style.animationDelay = (Math.random() * 20) + 's';
    box.appendChild(el);
  }
})();

// ══════════════════════════════════════════════════════
// MODEL MANAGEMENT
// ══════════════════════════════════════════════════════
const CHARS = ['0','1','2','3','4','5','6','7','8','9'];
const NUM_CLASSES = 10;
const IMG_SIZE = 28;
const MODEL_KEY = 'indexeddb://math-digit-v6';
let digitModel = null;

function setTrainingUI(pct, msg) {
  document.getElementById('trainingBar').style.width = pct + '%';
  document.getElementById('trainingStatus').textContent = msg;
}

function createModel() {
  // Lighter model for faster training on mobile/iPad
  const m = tf.sequential({
    layers: [
      tf.layers.reshape({ inputShape: [IMG_SIZE * IMG_SIZE], targetShape: [IMG_SIZE, IMG_SIZE, 1] }),
      tf.layers.conv2d({ filters: 8, kernelSize: 3, activation: 'relu', padding: 'same' }),
      tf.layers.maxPooling2d({ poolSize: 2 }),
      tf.layers.conv2d({ filters: 16, kernelSize: 3, activation: 'relu', padding: 'same' }),
      tf.layers.maxPooling2d({ poolSize: 2 }),
      tf.layers.flatten(),
      tf.layers.dense({ units: 32, activation: 'relu' }),
      tf.layers.dropout({ rate: 0.2 }),
      tf.layers.dense({ units: NUM_CLASSES, activation: 'softmax' })
    ]
  });
  m.compile({ optimizer: tf.train.adam(0.001), loss: 'categoricalCrossentropy', metrics: ['accuracy'] });
  return m;
}

// Draw a single stroke-based digit on ctx, centered at (cx,cy), with given scale and line width.
// scale=1 means the digit fits in roughly a 0.6×0.8 normalized unit square.
function renderStrokeDigit(ctx, digit, cx, cy, sc, lw) {
  ctx.save();
  ctx.translate(cx, cy);
  ctx.scale(sc, sc);
  ctx.strokeStyle = '#fff';
  ctx.lineWidth = lw / sc;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';

  const s = 10; // base half-size
  ctx.beginPath();
  switch (digit) {
    case 0:
      // Oval: ellipse centered at origin
      ctx.ellipse(0, 0, s * 0.55, s * 0.82, 0, 0, Math.PI * 2);
      break;
    case 1:
      // Short serif base + vertical stroke
      ctx.moveTo(-s * 0.18, -s * 0.62);
      ctx.lineTo(s * 0.08, -s * 0.82);
      ctx.lineTo(s * 0.08, s * 0.82);
      ctx.moveTo(-s * 0.3, s * 0.82);
      ctx.lineTo(s * 0.42, s * 0.82);
      break;
    case 2:
      // Arc top-right, diagonal down-left, base
      ctx.moveTo(-s * 0.38, -s * 0.35);
      ctx.bezierCurveTo(-s * 0.38, -s * 0.88, s * 0.52, -s * 0.88, s * 0.52, -s * 0.35);
      ctx.bezierCurveTo(s * 0.52, s * 0.1, -s * 0.4, s * 0.4, -s * 0.48, s * 0.82);
      ctx.lineTo(s * 0.52, s * 0.82);
      break;
    case 3:
      // Two right-facing arcs
      ctx.moveTo(-s * 0.38, -s * 0.82);
      ctx.bezierCurveTo(s * 0.65, -s * 0.82, s * 0.65, -s * 0.08, -s * 0.08, -s * 0.08);
      ctx.bezierCurveTo(s * 0.65, -s * 0.08, s * 0.65, s * 0.82, -s * 0.38, s * 0.82);
      break;
    case 4:
      // Diagonal down, horizontal, then vertical stem
      ctx.moveTo(s * 0.35, s * 0.82);
      ctx.lineTo(s * 0.35, -s * 0.82);
      ctx.lineTo(-s * 0.48, s * 0.22);
      ctx.lineTo(s * 0.62, s * 0.22);
      break;
    case 5:
      // Top bar, down-left hook, then bump right
      ctx.moveTo(s * 0.45, -s * 0.82);
      ctx.lineTo(-s * 0.38, -s * 0.82);
      ctx.lineTo(-s * 0.38, -s * 0.08);
      ctx.bezierCurveTo(-s * 0.38, -s * 0.08, s * 0.55, -s * 0.2, s * 0.55, s * 0.35);
      ctx.bezierCurveTo(s * 0.55, s * 0.82, -s * 0.38, s * 0.92, -s * 0.42, s * 0.6);
      break;
    case 6:
      // Curved down from top, circle at bottom
      ctx.moveTo(s * 0.3, -s * 0.82);
      ctx.bezierCurveTo(-s * 0.6, -s * 0.82, -s * 0.6, s * 0.0, -s * 0.55, s * 0.28);
      ctx.bezierCurveTo(-s * 0.55, s * 0.95, s * 0.55, s * 0.95, s * 0.55, s * 0.28);
      ctx.bezierCurveTo(s * 0.55, -s * 0.18, -s * 0.55, -s * 0.18, -s * 0.55, s * 0.28);
      break;
    case 7:
      // Top bar + diagonal
      ctx.moveTo(-s * 0.42, -s * 0.82);
      ctx.lineTo(s * 0.48, -s * 0.82);
      ctx.lineTo(-s * 0.25, s * 0.82);
      break;
    case 8:
      // Two stacked ovals (figure-8)
      ctx.ellipse(0, -s * 0.38, s * 0.42, s * 0.42, 0, 0, Math.PI * 2);
      ctx.moveTo(s * 0.42, s * 0.3);
      ctx.ellipse(0, s * 0.38, s * 0.48, s * 0.42, 0, 0, Math.PI * 2);
      break;
    case 9:
      // Circle top + stem down
      ctx.ellipse(s * 0.05, -s * 0.28, s * 0.48, s * 0.48, 0, 0, Math.PI * 2);
      ctx.moveTo(s * 0.52, -s * 0.28);
      ctx.bezierCurveTo(s * 0.52, s * 0.5, s * 0.1, s * 0.92, -s * 0.28, s * 0.82);
      break;
  }
  ctx.stroke();
  ctx.restore();
}

async function generateTrainingData(samplesPerClass = 120) {
  const size = IMG_SIZE;

  const tmpCanvas = document.createElement('canvas');
  tmpCanvas.width = size; tmpCanvas.height = size;
  const tmpCtx = tmpCanvas.getContext('2d');

  const n = samplesPerClass * NUM_CLASSES;
  // Pre-allocate typed arrays (much faster than push to regular Array)
  const flatXs = new Float32Array(n * size * size);
  const allYs  = new Int32Array(n);
  let idx = 0;

  for (let ci = 0; ci < CHARS.length; ci++) {
    const digit = parseInt(CHARS[ci], 10);
    for (let s = 0; s < samplesPerClass; s++) {
      tmpCtx.fillStyle = '#000';
      tmpCtx.fillRect(0, 0, size, size);

      const rot = (Math.random() - 0.5) * 0.4;
      const tx  = (Math.random() - 0.5) * size * 0.12;
      const ty  = (Math.random() - 0.5) * size * 0.12;
      const sc  = size * (0.038 + Math.random() * 0.018);
      const lw  = size * (0.055 + Math.random() * 0.045);

      tmpCtx.save();
      tmpCtx.translate(size / 2 + tx, size / 2 + ty);
      tmpCtx.rotate(rot);
      renderStrokeDigit(tmpCtx, digit, 0, 0, sc, lw);
      tmpCtx.restore();

      const id = tmpCtx.getImageData(0, 0, size, size);
      const base = idx * size * size;
      for (let p = 0; p < size * size; p++) {
        flatXs[base + p] = Math.min(1, Math.max(0, id.data[p * 4] / 255 + (Math.random() - 0.5) * 0.06));
      }
      allYs[idx] = ci;
      idx++;
    }
    // Yield to browser every class (keeps UI responsive on iPad)
    await tf.nextFrame();
  }

  // tf.fit with shuffle:true handles shuffling per-epoch — no manual shuffle needed
  return {
    xs: tf.tensor2d(flatXs, [n, size * size]),
    ys: tf.oneHot(tf.tensor1d(allYs, 'int32'), NUM_CLASSES)
  };
}

async function trainNewModel() {
  setTrainingUI(5, 'AIを かくにんちゅう...');
  await tf.nextFrame();

  // Detect backend — CPU fallback (no WebGL) needs reduced workload
  await tf.ready();
  const useWebGL = tf.getBackend() === 'webgl';
  const samplesPerClass = useWebGL ? 120 : 70;
  const EPOCHS        = useWebGL ? 12  : 8;
  const BATCH_SIZE    = useWebGL ? 32  : 16;

  setTrainingUI(8, 'データを じゅんびちゅう...');
  await tf.nextFrame();

  const data = await generateTrainingData(samplesPerClass);
  setTrainingUI(30, 'がくしゅうを はじめます...');
  await tf.nextFrame();

  const m = createModel();
  const totalSamples = samplesPerClass * NUM_CLASSES;
  const batchesPerEpoch = Math.ceil(totalSamples / BATCH_SIZE);
  let curEpoch = 0;

  await m.fit(data.xs, data.ys, {
    epochs: EPOCHS,
    batchSize: BATCH_SIZE,
    shuffle: true,
    callbacks: {
      onEpochBegin: async (epoch) => { curEpoch = epoch; },
      // Yield to browser every 4 batches so the UI stays responsive
      onBatchEnd: async (batch) => {
        if (batch % 4 !== 0) return;
        const progress = (curEpoch + (batch + 1) / batchesPerEpoch) / EPOCHS;
        const pct = 30 + Math.round(progress * 65);
        setTrainingUI(Math.min(pct, 94), `がくしゅうちゅう... ${curEpoch + 1}/${EPOCHS} エポック`);
        await tf.nextFrame();
      },
      onEpochEnd: async (epoch, logs) => {
        const pct = 30 + Math.round((epoch + 1) / EPOCHS * 65);
        const acc = Math.round((logs.acc || 0) * 100);
        setTrainingUI(pct, `がくしゅうちゅう... ${acc}% せいかく`);
        await tf.nextFrame();
      }
    }
  });

  data.xs.dispose();
  data.ys.dispose();

  try {
    await m.save(MODEL_KEY);
  } catch (e) {
    console.warn('Could not cache model:', e);
  }

  return m;
}

async function initModel() {
  setTrainingUI(2, 'AIを よみこみちゅう...');
  await tf.nextFrame();

  try {
    digitModel = await tf.loadLayersModel(MODEL_KEY);
    setTrainingUI(100, 'じゅんびかんりょう！');
    await tf.nextFrame();
  } catch (e) {
    digitModel = await trainNewModel();
    setTrainingUI(100, 'じゅんびかんりょう！ 🎉');
    await tf.nextFrame();
  }

  // Warm up
  const dummy = tf.zeros([1, IMG_SIZE * IMG_SIZE]);
  digitModel.predict(dummy).dispose();
  dummy.dispose();

  await new Promise(r => setTimeout(r, 600));
  show('startScreen');
}

// ── CANVAS PREPROCESSING & RECOGNITION ──
function toMid64(srcCanvas) {
  const mid = document.createElement('canvas');
  mid.width = 64; mid.height = 64;
  const mc = mid.getContext('2d');
  mc.fillStyle = '#fff'; mc.fillRect(0, 0, 64, 64);
  mc.drawImage(srcCanvas, 0, 0, 64, 64);
  return { canvas: mid, data: mc.getImageData(0, 0, 64, 64).data };
}

function findBBox(d, w, h) {
  let x0=w, y0=h, x1=0, y1=0;
  for (let y=0; y<h; y++) for (let x=0; x<w; x++) {
    if (d[(y*w+x)*4] < 200) {
      if (x<x0) x0=x; if (x>x1) x1=x;
      if (y<y0) y0=y; if (y>y1) y1=y;
    }
  }
  return x1>=x0 && y1>=y0 ? {x0,y0,x1,y1} : null;
}

function cropNormalize(srcCanvas, bx0, by0, bx1, by1) {
  const pad = 3;
  bx0 = Math.max(0, bx0-pad); by0 = Math.max(0, by0-pad);
  bx1 = Math.min(srcCanvas.width-1, bx1+pad);
  by1 = Math.min(srcCanvas.height-1, by1+pad);
  if (bx1<=bx0 || by1<=by0) return null;
  const out = document.createElement('canvas');
  out.width = IMG_SIZE; out.height = IMG_SIZE;
  const oc = out.getContext('2d');
  oc.fillStyle = '#fff'; oc.fillRect(0, 0, IMG_SIZE, IMG_SIZE);
  oc.drawImage(srcCanvas, bx0, by0, bx1-bx0+1, by1-by0+1, 0, 0, IMG_SIZE, IMG_SIZE);
  const od = oc.getImageData(0, 0, IMG_SIZE, IMG_SIZE).data;
  const px = new Float32Array(IMG_SIZE * IMG_SIZE);
  for (let i=0; i<IMG_SIZE*IMG_SIZE; i++) px[i] = 1 - od[i*4]/255;
  return px;
}

async function classifyPixels(px) {
  const t = tf.tensor2d([Array.from(px)], [1, IMG_SIZE*IMG_SIZE]);
  const p = digitModel.predict(t);
  const a = await p.data();
  t.dispose(); p.dispose();
  let bi=0, bc=0;
  for (let i=0; i<a.length; i++) if (a[i]>bc) { bc=a[i]; bi=i; }
  return { digit: bi, confidence: bc };
}

// Try to classify left half as "1" and right half as "0" at a given splitX
async function trySplitAt(mid64, bbox, splitX) {
  const { x0, y0, x1, y1 } = bbox;
  const lPx = cropNormalize(mid64, x0, y0, splitX, y1);
  const rPx = cropNormalize(mid64, splitX + 1, y0, x1, y1);
  if (!lPx || !rPx) return null;
  const [L, R] = await Promise.all([classifyPixels(lPx), classifyPixels(rPx)]);
  if (L.digit === 1 && R.digit === 0) {
    return { digit: 10, confidence: Math.min(L.confidence, R.confidence) };
  }
  return null;
}

// Try multiple split strategies to detect "10" (1 on left, 0 on right)
async function tryTenRecognition(mid64, d, bbox) {
  const { x0, y0, x1, y1 } = bbox;
  const bw = x1 - x0 + 1, bh = y1 - y0 + 1;
  // "1" is very narrow, so "10" can have bw/bh as small as ~0.4
  // Only skip if truly square-ish single digit (too narrow overall)
  if (bw < bh * 0.35) return null;

  // Strategy 1: find the column with fewest dark pixels (gap between "1" and "0")
  // "1" typically occupies left 15–30% of the combined bbox
  const mS = Math.floor(x0 + bw * 0.12);
  const mE = Math.floor(x0 + bw * 0.60);
  let bestSplitX = Math.floor(x0 + bw * 0.25), minDens = Infinity;
  for (let x = mS; x <= mE; x++) {
    let dens = 0;
    for (let y = y0; y <= y1; y++) if (d[(y * 64 + x) * 4] < 200) dens++;
    if (dens < minDens) { minDens = dens; bestSplitX = x; }
  }
  if (minDens <= bh * 0.8) {
    const res = await trySplitAt(mid64, bbox, bestSplitX);
    if (res && res.confidence > 0.22) return res;
  }

  // Strategy 2: try fixed split ratios — "1" is narrow so try 18–38% of width
  for (const ratio of [0.18, 0.22, 0.26, 0.30, 0.34, 0.38]) {
    const splitX = Math.floor(x0 + bw * ratio);
    if (Math.abs(splitX - bestSplitX) < 2) continue; // skip if already tried nearby
    const res = await trySplitAt(mid64, bbox, splitX);
    if (res && res.confidence > 0.22) return res;
  }

  return null;
}

async function recognizeFromCanvas() {
  const cv = document.getElementById('drawCanvas');
  const { canvas: mid, data: d } = toMid64(cv);
  const bbox = findBBox(d, 64, 64);
  if (!bbox) return null;

  const { x0, y0, x1, y1 } = bbox;
  const fullPx = cropNormalize(mid, x0, y0, x1, y1);
  if (!fullPx) return null;

  // Try "10" split detection
  const ten = await tryTenRecognition(mid, d, bbox);
  if (ten && ten.confidence > 0.22) return ten;

  return classifyPixels(fullPx);
}

// ══════════════════════════════════════════════════════
// GAME STATE
// ══════════════════════════════════════════════════════
let problems = [], current = 0, recognizedAnswer = null, results = [];
let ctx = null, drawing = false, lx = 0, ly = 0;

function show(id) {
  document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
  document.getElementById(id).classList.add('active');
}

function genProblems() {
  const ps = [];
  while (ps.length < 5) {
    const a = 1 + Math.floor(Math.random() * 9);
    const b = 1 + Math.floor(Math.random() * (10 - a));
    if (a + b <= 10) ps.push({ a, b, ans: a + b });
  }
  return ps;
}

function startGame() {
  problems = genProblems();
  current = 0; results = []; recognizedAnswer = null;
  buildDots();
  show('questionScreen');
  loadQ(0);
}

function buildDots() {
  const box = document.getElementById('progressDots');
  box.innerHTML = '';
  for (let i = 0; i < 5; i++) {
    const d = document.createElement('div');
    d.className = 'dot' + (i === 0 ? ' active' : '');
    d.id = 'dot' + i;
    box.appendChild(d);
  }
}

function loadQ(idx) {
  const p = problems[idx];
  recognizedAnswer = null;

  for (let i = 0; i < 5; i++) {
    const d = document.getElementById('dot' + i);
    d.className = 'dot' + (i < idx ? ' done' : i === idx ? ' active' : '');
  }
  document.getElementById('progressText').textContent = (idx + 1) + 'もん め';

  const eqA = document.getElementById('eqA');
  const eqB = document.getElementById('eqB');
  eqA.textContent = p.a;
  eqB.textContent = p.b;
  document.getElementById('eqBlank').textContent = '？';
  document.getElementById('eqBlank').style.color = 'var(--purple)';

  eqA.classList.remove('pop'); eqB.classList.remove('pop');
  void eqA.offsetWidth;
  eqA.classList.add('pop');
  setTimeout(() => { eqB.classList.remove('pop'); void eqB.offsetWidth; eqB.classList.add('pop'); }, 80);

  // Reset recognition area
  const rn = document.getElementById('recogNumber');
  rn.textContent = '？';
  rn.className = 'recog-number';
  document.getElementById('submitBtn').disabled = true;
  document.getElementById('judgeBtn').disabled = false;

  clearCanvas();
  fitCanvas();
}

// ── JUDGE ──
async function judgeDrawing() {
  const cv = document.getElementById('drawCanvas');
  const { canvas: _mid, data: _d } = toMid64(cv);
  const _bbox = findBBox(_d, 64, 64);
  if (!_bbox) {
    // Nothing drawn
    const rn = document.getElementById('recogNumber');
    rn.textContent = '？';
    rn.className = 'recog-number';
    // Shake hint
    document.querySelector('.canvas-hint').textContent = '✏️ なにか かいてね！';
    setTimeout(() => document.querySelector('.canvas-hint').textContent = '✏️ こたえを かいてみよう', 1500);
    return;
  }

  // Show thinking
  const rn = document.getElementById('recogNumber');
  rn.textContent = '🤔';
  rn.className = 'recog-number recog-thinking';
  document.getElementById('judgeBtn').disabled = true;
  await tf.nextFrame();

  const result = await recognizeFromCanvas();

  if (!result) {
    rn.textContent = '？';
    rn.className = 'recog-number';
    document.getElementById('judgeBtn').disabled = false;
    return;
  }

  recognizedAnswer = result.digit;
  rn.textContent = result.digit;
  rn.className = 'recog-number';

  // Show the recognized digit in the equation
  const blank = document.getElementById('eqBlank');
  blank.textContent = result.digit;

  // Color hint based on correctness (preview)
  const correct = problems[current].ans;
  if (result.digit === correct) {
    rn.classList.add('recog-ok');
    blank.style.color = '#16A34A';
  } else {
    rn.classList.add('recog-ng');
    blank.style.color = '#DC2626';
  }

  document.getElementById('submitBtn').disabled = false;
  document.getElementById('judgeBtn').disabled = false;
}

function submitAnswer() {
  if (recognizedAnswer === null) return;
  const p = problems[current];
  const ok = recognizedAnswer === p.ans;
  results.push({ ...p, given: recognizedAnswer, ok });
  showFeedback(ok, () => {
    current++;
    if (current >= 5) showResults();
    else loadQ(current);
  });
}

// ── FEEDBACK ──
const OKS = ['すごい！', 'せいかい！', 'やったね！', 'かんぺき！', 'さすが！'];
const OKE = ['🎉','⭐','🌟','🎊','✨'];

function showFeedback(ok, cb) {
  const ov = document.getElementById('feedbackOverlay');
  const ic = document.getElementById('fbIcon');
  const tx = document.getElementById('fbText');
  if (ok) {
    ov.className = 'show correct';
    ic.textContent = OKE[Math.floor(Math.random() * OKE.length)];
    tx.textContent = OKS[Math.floor(Math.random() * OKS.length)];
    confetti();
  } else {
    ov.className = 'show wrong';
    ic.textContent = '😊';
    tx.textContent = `こたえは ${problems[current].ans} だよ！`;
  }
  ic.style.animation = 'none'; void ic.offsetWidth; ic.style.animation = '';
  tx.style.animation = 'none'; void tx.offsetWidth; tx.style.animation = '';
  setTimeout(() => { ov.className = ''; cb(); }, ok ? 1700 : 2200);
}

// ── CONFETTI ──
const COLORS = ['#FF6B4A','#3DD6C8','#9B7FE8','#FFD93D','#52D68A','#FF7BAC','#60A5FA'];
function confetti() {
  const box = document.getElementById('confettiBox');
  for (let i = 0; i < 70; i++) {
    const p = document.createElement('div');
    p.className = 'cp';
    const size = 8 + Math.random() * 14;
    p.style.cssText = `left:${Math.random()*100}vw;top:-${size}px;width:${size}px;height:${size}px;background:${COLORS[Math.floor(Math.random()*COLORS.length)]};border-radius:${Math.random()>0.5?'50%':'3px'};animation-duration:${1.4+Math.random()*1.8}s;animation-delay:${Math.random()*0.4}s;`;
    box.appendChild(p);
    setTimeout(() => p.remove(), 3200);
  }
}

// ── RESULTS ──
function showResults() {
  const score = results.filter(r => r.ok).length;
  const titles = [
    ['📚','れんしゅうしよう！'], ['💪','あとすこし！'], ['😊','がんばったね！'],
    ['🌟','よくできました！'], ['🎊','すごいね！ほぼかんぺき！'], ['🏆','かんぺき！まんてん！！']
  ];
  const [em, tt] = titles[score];
  document.getElementById('resEmoji').textContent = em;
  document.getElementById('resTitle').textContent = tt;
  document.getElementById('scoreBig').textContent  = `${score}/5`;
  document.getElementById('scoreSub').textContent  = 'もん せいかい！';

  const row = document.getElementById('answersRow');
  row.innerHTML = '';
  results.forEach(r => {
    const c = document.createElement('div');
    c.className = 'ans-card ' + (r.ok ? 'ok' : 'ng');
    c.innerHTML = `<div class="ans-prob">${r.a}＋${r.b}</div><div class="ans-mark">${r.ok ? '✅' : '❌'}</div>${!r.ok ? `<div class="ans-correct">→ ${r.ans}</div>` : ''}`;
    row.appendChild(c);
  });

  show('resultsScreen');
  if (score >= 4) { confetti(); setTimeout(confetti, 700); }
}

function restartGame() { show('startScreen'); }

// ══════════════════════════════════════════════════════
// CANVAS DRAWING
// ══════════════════════════════════════════════════════
function drawCrosshair() {
  if (!ctx) return;
  const cv = document.getElementById('drawCanvas');
  const { width: w, height: h } = cv.getBoundingClientRect();
  ctx.save();
  ctx.strokeStyle = 'rgba(155,127,232,0.25)';
  ctx.lineWidth = 1;
  ctx.setLineDash([6, 6]);
  ctx.beginPath();
  ctx.moveTo(0,     h / 2); ctx.lineTo(w,     h / 2); // 横
  ctx.moveTo(w / 2, 0);     ctx.lineTo(w / 2, h);     // 縦
  ctx.stroke();
  ctx.restore();
}

function fitCanvas() {
  const cv = document.getElementById('drawCanvas');
  const dpr = Math.min(window.devicePixelRatio || 2, 3);
  const rect = cv.getBoundingClientRect();
  cv.width  = rect.width  * dpr;
  cv.height = rect.height * dpr;
  ctx = cv.getContext('2d');
  ctx.scale(dpr, dpr);
  ctx.lineCap  = 'round';
  ctx.lineJoin = 'round';
  ctx.lineWidth = 14;
  ctx.strokeStyle = '#2A2060';
  drawCrosshair();
}

function clearCanvas() {
  if (!ctx) return;
  const cv = document.getElementById('drawCanvas');
  ctx.clearRect(0, 0, cv.width, cv.height);
  drawCrosshair();
  // Reset recognition when canvas cleared
  const rn = document.getElementById('recogNumber');
  rn.textContent = '？';
  rn.className = 'recog-number';
  document.getElementById('eqBlank').textContent = '？';
  document.getElementById('eqBlank').style.color = 'var(--purple)';
  recognizedAnswer = null;
  document.getElementById('submitBtn').disabled = true;
}

function canvasPos(e, cv) {
  const r = cv.getBoundingClientRect();
  return { x: e.clientX - r.left, y: e.clientY - r.top };
}

(function initCanvas() {
  const cv = document.getElementById('drawCanvas');
  const ro = new ResizeObserver(() => fitCanvas());
  ro.observe(cv);

  cv.addEventListener('pointerdown', e => {
    e.preventDefault();
    cv.setPointerCapture(e.pointerId);
    drawing = true;
    const { x, y } = canvasPos(e, cv);
    lx = x; ly = y;
    ctx.beginPath();
    ctx.arc(x, y, ctx.lineWidth / 2, 0, Math.PI * 2);
    ctx.fillStyle = ctx.strokeStyle;
    ctx.fill();
  }, { passive: false });

  cv.addEventListener('pointermove', e => {
    e.preventDefault();
    if (!drawing) return;
    const { x, y } = canvasPos(e, cv);
    ctx.lineWidth = e.pointerType === 'pen' ? 6 + (e.pressure || 0.5) * 16 : 14;
    ctx.beginPath(); ctx.moveTo(lx, ly); ctx.lineTo(x, y); ctx.stroke();
    lx = x; ly = y;
  }, { passive: false });

  cv.addEventListener('pointerup',     e => { e.preventDefault(); drawing = false; }, { passive: false });
  cv.addEventListener('pointercancel', () => { drawing = false; });
})();

document.addEventListener('touchmove', e => {
  if (e.target.id === 'drawCanvas') e.preventDefault();
}, { passive: false });

// ══════════════════════════════════════════════════════
// BOOT
// ══════════════════════════════════════════════════════
window.addEventListener('load', () => initModel());
