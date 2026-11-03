import { checkPalette, checkContrast, checkLayout, findLogo, checkLogo, checkBanned, toneVerdict, overlap, stretchTiles, photoMask, maskPhoto } from './lint.js';

const TF_URL = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.1';
const TESS_URL = 'https://cdn.jsdelivr.net/npm/tesseract.js@5.1.1/dist/tesseract.min.js';
const EMB_MODEL = 'Xenova/multilingual-e5-small';
const CLIP_MODEL = 'Xenova/clip-vit-base-patch32';
const SAMPLES = {
  soop: [['samples/soop-good.png', '정상 배너', 'instagram'], ['samples/soop-bad.png', '위반 배너', 'instagram'], ['samples/soop-photo.png', '사진 배너', 'web']],
  moa: [['samples/moa-good.png', '정상 배너', 'web'], ['samples/moa-bad.png', '위반 배너', 'web'], ['samples/moa-photo.png', '사진 배너', 'instagram']],
};
const SAMPLE_CHAN = { soop: 'instagram', moa: 'web' };

const $ = id => document.getElementById(id);
const S = { brand: null, logo: null, img: null, imgEl: null, ocr: null, embed: null, brandVecs: null, issues: [], sel: -1, filter: 'all', report: null };
window.__guardrail = S; // 자동 시연·시험용

// ---------- 모델 준비 ----------
const prog = { ocr: 0, emb: 0 };
const setBar = () => { $('bar').style.width = Math.round((prog.ocr + prog.emb) / 2 * 100) + '%'; };
const status = t => { $('status').textContent = t; };

async function loadOCR() {
  await new Promise((res, rej) => { const s = document.createElement('script'); s.src = TESS_URL; s.onload = res; s.onerror = rej; document.head.appendChild(s); });
  S.ocr = await Tesseract.createWorker(['kor', 'eng'], 1, {
    logger: m => { if (m.status?.includes('load')) { prog.ocr = Math.max(prog.ocr, m.progress * 0.9); setBar(); } },
  });
  await S.ocr.setParameters({ preserve_interword_spaces: '1' });
  prog.ocr = 1; setBar();
}
async function loadEmbed() {
  const { pipeline } = await import(TF_URL);
  const files = {};
  const ext = await pipeline('feature-extraction', EMB_MODEL, {
    dtype: 'q8',
    progress_callback: p => {
      if (p.status === 'progress' && p.total) { files[p.file] = p.loaded / p.total; const v = Object.values(files); prog.emb = v.reduce((a, b) => a + b, 0) / v.length * 0.95; setBar(); }
    },
  });
  S.embed = async texts => (await ext(texts.map(t => 'query: ' + t), { pooling: 'mean', normalize: true })).tolist();
  prog.emb = 1; setBar();
}
// 시각 무드 검사(CLIP)는 핵심 검사 준비 뒤 백그라운드로 불러온다
async function loadClip() {
  const { pipeline } = await import(TF_URL);
  S.clip = await pipeline('zero-shot-image-classification', CLIP_MODEL, { dtype: 'q8' });
}
async function boot() {
  status('AI 모델 내려받는 중 — 첫 방문 한 번만(약 130MB), 이후에는 브라우저에 저장된 모델을 씁니다…');
  const r = await Promise.allSettled([loadOCR(), loadEmbed()]);
  const fails = [];
  if (r[0].status === 'rejected') fails.push('문자 인식');
  if (r[1].status === 'rejected') fails.push('문장 임베딩');
  await prepBrandVecs();
  status(fails.length ? `${fails.join('·')} 모델을 불러오지 못해 해당 검사는 건너뜁니다.` : '준비 완료 — 샘플을 고르고 검수를 시작하세요.');
  $('runBtn').disabled = !S.ocr && !S.embed;
  S.ready = true;
  loadClip().then(() => { S.clipReady = true; if (!$('runBtn').disabled && !S.report) status($('status').textContent + ' (시각 무드 모델 추가 완료)'); }).catch(() => { S.clipReady = false; });
}

// ---------- 브랜드 ----------
async function setBrand(b) {
  S.brand = b; S.brandVecs = null;
  S.logo = b.logoData || await loadImageData(b.logo);
  $('guideCard').innerHTML = `
    <b>${b.name} ${b.nameEn}</b> · ${b.category}<br>
    <span class="sw">${b.palette.map(p => `<i title="${p.name} ${p.hex}" style="background:${p.hex}"></i>`).join('')}</span><br>
    톤: ${b.tone.description}<br>
    금칙·규제어 ${b.banned.length}개 · 톤 예문 ${b.tone.onBrand.length + b.tone.offBrand.length}개 · 로고 여백 높이×${b.logoClearSpace}`;
  $('chanSel').innerHTML = Object.entries(b.channels).map(([k, c]) => `<option value="${k}">${c.label}</option>`).join('');
  $('chanSel').value = SAMPLE_CHAN[b.id] || 'instagram';
  $('samples').innerHTML = (SAMPLES[b.id] || []).map(([src, label, ch]) => `<button data-src="${src}" data-ch="${ch}"><img src="${src}" alt="">${label}</button>`).join('');
  $('samples').querySelectorAll('button').forEach(btn => btn.onclick = () => {
    $('samples').querySelectorAll('button').forEach(x => x.classList.toggle('on', x === btn));
    if (b.channels[btn.dataset.ch]) $('chanSel').value = btn.dataset.ch;
    setImage(btn.dataset.src);
  });
  await prepBrandVecs();
}
async function prepBrandVecs() {
  if (!S.embed || !S.brand || S.brandVecs) return;
  const t = S.brand.tone;
  const v = await S.embed([...t.onBrand, ...t.offBrand]);
  S.brandVecs = { on: v.slice(0, t.onBrand.length), off: v.slice(t.onBrand.length) };
}
async function loadBrand(id) { setBrand(await (await fetch(`brands/${id}.json`)).json()); }

// ---------- 이미지 ----------
function loadImg(src) { return new Promise((res, rej) => { const i = new Image(); i.onload = () => res(i); i.onerror = rej; i.src = src; }); }
async function loadImageData(src) {
  const i = await loadImg(src), c = new OffscreenCanvas(i.naturalWidth, i.naturalHeight), x = c.getContext('2d');
  x.drawImage(i, 0, 0); return x.getImageData(0, 0, c.width, c.height);
}
async function setImage(src) {
  S.imgEl = await loadImg(src);
  S.img = await loadImageData(src);
  S.issues = []; S.sel = -1; S.report = null;
  $('empty').hidden = true; resetResult(); draw();
}

// ---------- 검수 ----------
async function run() {
  if (!S.img) { status('먼저 배너를 고르거나 올려 주세요.'); return; }
  $('runBtn').disabled = true;
  const t0 = performance.now();
  const b = S.brand, chan = $('chanSel').value, issues = [];
  let lines = [];
  if (S.ocr) {
    status('① 문자 인식(OCR) 중…');
    // k: 축소본을 읽었을 때 좌표를 원본 크기로 되돌리는 배율
    const ocrLines = async (im, k = 1) => {
      let c = im;
      if (!(im instanceof HTMLCanvasElement)) { c = document.createElement('canvas'); c.width = im.width; c.height = im.height; c.getContext('2d').putImageData(im, 0, 0); }
      const { data } = await S.ocr.recognize(c);
      const out = [];
      for (const l of data.lines) {
        if (l.confidence < 55) continue;
        // 같은 높이에 떨어져 있는 글(예: 본문 옆 로고)은 단어 간격으로 나눈다
        // 줄 높이는 단어 높이의 중앙값(줄 상자는 기호·겹친 글자 때문에 부풀 수 있음)
        const hs = l.words.map(w => w.bbox.y1 - w.bbox.y0).sort((a, b) => a - b), lh = hs[hs.length >> 1] || (l.bbox.y1 - l.bbox.y0), groups = [];
        for (const w of l.words) {
          const g = groups[groups.length - 1];
          if (g && w.bbox.x0 - g[g.length - 1].bbox.x1 < lh * 1.5) g.push(w); else groups.push([w]);
        }
        for (const g of groups) {
          const text = g.map((w, i) => (i && w.bbox.x0 - g[i - 1].bbox.x1 > lh * 0.25 ? ' ' : '') + w.text).join('').trim();
          if ((text.match(/[가-힣]/g) || []).length < 2 && (text.match(/[A-Za-z]/g) || []).length < 4) continue;
          const x0 = Math.min(...g.map(w => w.bbox.x0)), y0 = Math.min(...g.map(w => w.bbox.y0)), x1 = Math.max(...g.map(w => w.bbox.x1)), y1 = Math.max(...g.map(w => w.bbox.y1));
          out.push({ text, box: { x: x0 * k, y: y0 * k, w: (x1 - x0) * k, h: (y1 - y0) * k }, fontPx: k * [...g.map(w => w.bbox.y1 - w.bbox.y0)].sort((a, b) => a - b)[g.length >> 1] * 1.15, conf: l.confidence });
        }
      }
      return out;
    };
    const pm0 = photoMask(S.img);
    lines = await ocrLines(pm0.share > 0.03 ? maskPhoto(S.img, pm0) : S.img);
    status('① 문자 인식 2차(저대비·사진 위 글자 보강)…');
    // 2차 결과는 겹치지 않는 줄만 더하고, 겹치면 신뢰도가 높은 쪽을 남긴다
    // 3차: 장식 글꼴의 큰 헤드라인은 절반 크기 흑백 대비본에서 더 잘 읽힌다(큰 글자만 채택)
    status('① 문자 인식 3차(장식 헤드라인 보강)…');
    const half = document.createElement('canvas'); half.width = S.img.width / 2; half.height = S.img.height / 2;
    { const src = document.createElement('canvas'); src.width = S.img.width; src.height = S.img.height; src.getContext('2d').putImageData(S.img, 0, 0);
      const hx = half.getContext('2d'); hx.filter = 'grayscale(1) contrast(1.8)'; hx.drawImage(src, 0, 0, half.width, half.height); }
    const pass3 = (await ocrLines(half, 2)).filter(l => l.fontPx >= 40);
    for (const l of [...await ocrLines(stretchTiles(S.img)), ...pass3]) {
      const hit = lines.findIndex(k => overlap(k.box, l.box) > 0.3 || overlap(l.box, k.box) > 0.3);
      if (hit < 0) lines.push({ ...l, pass2: true });
      else if (l.conf > lines[hit].conf + 10 && l.text.length >= lines[hit].text.length) lines[hit] = { ...l, pass2: true };
    }
  }
  status('② 로고 탐지 중…'); await tick();
  const logoHit = findLogo(S.img, S.logo, b.logoMinWidth);
  if (logoHit) lines.forEach(l => { const b = logoHit.box, cx = l.box.x + l.box.w / 2, cy = l.box.y + l.box.h / 2; l.isLogo = overlap(b, l.box) > 0.4 || (cx > b.x && cx < b.x + b.w && cy > b.y && cy < b.y + b.h); });
  // 사진 위에서 읽힌 글은 오인식이 많아 신뢰도가 높을 때만 쓴다
  const pm = photoMask(S.img);
  const onPhoto = l => { let n = 0, k = 0; for (let y = l.box.y; y < l.box.y + l.box.h; y += 4) for (let x = l.box.x; x < l.box.x + l.box.w; x += 4) { n++; k += pm.isPhoto(x, y); } return n && k / n > 0.5; };
  const textLines = lines.filter(l => !l.isLogo && (!onPhoto(l) || l.conf >= 80));

  status('③ 색상·대비·규격 검사 중…'); await tick();
  const pal = checkPalette(S.img, b);
  issues.push(...pal.issues, ...checkContrast(S.img, textLines, b), ...checkLayout(S.img, textLines, b, chan), ...checkLogo(S.img, logoHit, lines, b));

  // 여러 줄로 나뉜 문장(같은 글자 크기·왼쪽 정렬·가까운 줄)은 한 문장으로 묶어 금칙어·톤을 본다
  const paras = [];
  for (const l of [...textLines].sort((a, c) => a.box.y - c.box.y)) {
    const p = paras.find(q => { const t = q.last; return Math.abs(t.box.x - l.box.x) < t.box.h * 0.6 && l.box.y - (t.box.y + t.box.h) < t.box.h * 0.8 && l.box.y >= t.box.y + t.box.h * 0.5 && l.fontPx / t.fontPx > 0.75 && l.fontPx / t.fontPx < 1.33; });
    if (p) { p.text += ' ' + l.text; const b = p.box, x1 = Math.max(b.x + b.w, l.box.x + l.box.w), y1 = Math.max(b.y + b.h, l.box.y + l.box.h); b.x = Math.min(b.x, l.box.x); b.w = x1 - b.x; b.h = y1 - b.y; p.last = l; }
    else paras.push({ text: l.text, box: { ...l.box }, last: l });
  }
  const units = [
    ...paras.map(p => ({ text: p.text, box: p.box, src: '배너' })),
    ...$('copy').value.split('\n').map(s => s.trim()).filter(Boolean).map(text => ({ text, src: '카피' })),
  ];
  // 금칙어는 놓치면 안 되므로 사진 위 저신뢰 글줄까지 본다(근사 매칭이 4음절 이상·20% 이내라 잡음이 걸릴 일은 드묾)
  const extra = lines.filter(l => !l.isLogo && !textLines.includes(l)).map(l => ({ text: l.text, box: l.box, src: '배너(사진 위)' }));
  const ban = checkBanned([...units, ...extra], b);
  issues.push(...ban.issues);

  if (S.clip && b.mood) {
    status('⑤ 시각 무드 분석(CLIP) 중…');
    const c = document.createElement('canvas'); c.width = S.img.width; c.height = S.img.height; c.getContext('2d').putImageData(S.img, 0, 0);
    const labels = [...b.mood.on, ...b.mood.off];
    const out = await S.clip(c.toDataURL('image/jpeg', 0.9), labels);
    const pOn = out.filter(o => b.mood.on.includes(o.label)).reduce((a, o) => a + o.score, 0);
    const top = out[0], mood = Math.round(pOn * 100);
    issues.push({ sev: mood >= 50 ? 'pass' : 'warn', cat: '무드', ai: 'AI 모델(CLIP 이미지-텍스트 임베딩)',
      title: `시각 무드 적합도 ${mood}점`, desc: `가장 가까운 묘사: “${top.label}”${mood < 50 ? ' — 브랜드가 피해야 할 분위기에 가깝습니다.' : ''}` });
  }
  const tone = [];
  if (S.embed && units.length) {
    status('⑥ 브랜드 톤 분석(문장 임베딩) 중…');
    await prepBrandVecs();
    const vecs = await S.embed(units.map(u => u.text));
    units.forEach((u, i) => {
      if (u.text.replace(/\s/g, '').length < 6) return;
      const v = toneVerdict(vecs[i], S.brandVecs.on, S.brandVecs.off, b);
      tone.push({ ...u, ...v });
      if (v.score < 40) ban.fixes.push({ from: u.text, to: v.nearOn, tone: true });
      if (v.score < 40) issues.push({ sev: 'fail', cat: '톤', ai: 'AI 모델(문장 임베딩)', box: u.box, title: `브랜드 톤과 어긋남(${v.score}점) — “${u.text}”`, desc: `금지 톤 예문 “${v.nearOff}”과 가깝습니다. 참고할 브랜드 문장: “${v.nearOn}”` });
      else if (v.score < 47) issues.push({ sev: 'warn', cat: '톤', ai: 'AI 모델(문장 임베딩)', box: u.box, title: `톤 확인 필요(${v.score}점) — “${u.text}”`, desc: `참고할 브랜드 문장: “${v.nearOn}”` });
    });
    if (tone.length) {
      const avg = Math.round(tone.reduce((a, x) => a + x.score, 0) / tone.length);
      issues.push({ sev: avg >= 55 ? 'pass' : 'warn', cat: '톤', ai: 'AI 모델(문장 임베딩)', title: `톤 일치도 평균 ${avg}점 (문장 ${tone.length}개)`, desc: `브랜드 톤: ${b.tone.description}` });
    }
  }
  // 심각도: 법·규제 > 접근성 > 브랜드. 점수는 그룹별 감점 상한을 둬 위반 수가 많아도 0점에 붙지 않게 한다
  issues.forEach(i => { i.grp = GROUP[i.cat] || '브랜드'; });
  const order = { fail: 0, warn: 1, pass: 2 };
  S.issues = issues.sort((a, c) => order[a.sev] - order[c.sev] || GORDER[a.grp] - GORDER[c.grp]);
  const nf = issues.filter(i => i.sev === 'fail').length, nw = issues.filter(i => i.sev === 'warn').length;
  const cnt = g => issues.filter(i => i.sev === 'fail' && i.grp === g).length;
  const legal = cnt('법·규제'), access = cnt('접근성'), brandF = cnt('브랜드');
  const score = Math.max(0, Math.round(100 - Math.min(50, legal * 25) - Math.min(25, access * 10) - Math.min(25, brandF * 6) - Math.min(10, nw * 2)));
  const verdict = legal ? '집행 불가' : access || brandF ? '수정 후 집행' : '집행 가능';
  S.report = { brand: b.name, channel: b.channels[chan].label, score, verdict, groups: { legal, access, brand: brandF }, fails: nf, warns: nw, ms: Math.round(performance.now() - t0), issues, fixes: ban.fixes, tone, ocrLines: lines.map(l => `${l.text}@${l.box.x},${l.box.y},${l.box.w}x${l.box.h}${l.isLogo ? 'L' : ''}`), logo: logoHit };
  render();
  status(`검수 완료 — ${(S.report.ms / 1000).toFixed(1)}초, 위반 ${nf} · 주의 ${nw}`);
  $('runBtn').disabled = false;
}
const tick = () => new Promise(r => setTimeout(r, 30));
const GROUP = { '규제·금칙어': '법·규제', '대비': '접근성', '가독성': '접근성' };
const GORDER = { '법·규제': 0, '접근성': 1, '브랜드': 2 };

// ---------- 출력 ----------
function resetResult() {
  $('score').innerHTML = '<b>–</b><span>종합 점수</span>'; $('score').style.borderColor = '#e3e6ec';
  $('counts').innerHTML = ''; $('issues').innerHTML = ''; $('top3').hidden = true; $('fixBox').hidden = true;
  $('btnJson').disabled = $('btnPrint').disabled = true;
}
function render() {
  const r = S.report;
  const col = r.score >= 80 ? 'var(--pass)' : r.score >= 50 ? 'var(--warn)' : 'var(--fail)';
  $('score').innerHTML = `<b style="color:${col}">${r.score}</b><span>종합 점수</span>`; $('score').style.borderColor = col;
  const vc = r.verdict === '집행 가능' ? 'c-pass' : r.verdict === '수정 후 집행' ? 'c-warn' : 'c-fail';
  $('counts').innerHTML = `<span class="verdict ${vc}">${r.verdict}</span><span class="c-fail">법·규제 ${r.groups.legal}</span><span class="c-fail">접근성 ${r.groups.access}</span><span class="c-fail">브랜드 ${r.groups.brand}</span><span class="c-warn">주의 ${r.warns}</span>`;
  const top = S.issues.filter(i => i.sev === 'fail').slice(0, 3);
  $('top3').hidden = !top.length;
  $('top3').innerHTML = '<b>먼저 고칠 것</b>' + top.map(i => `<span><em>${i.grp}</em>${esc(i.title)}</span>`).join('');
  const list = S.issues.map((x, i) => ({ x, i })).filter(({ x }) => S.filter === 'all' || x.sev === S.filter);
  $('issues').innerHTML = list.map(({ x, i }) => `<li class="${x.sev}${i === S.sel ? ' sel' : ''}" data-i="${i}">
    <div class="t">${x.box ? `<span class="tag">#${i + 1}</span>` : ''}<span class="tag">${x.grp} · ${x.cat}</span>${x.swatch ? `<i style="width:14px;height:14px;border-radius:3px;background:${x.swatch};display:inline-block"></i>` : ''}${esc(x.title)}</div>
    <div class="d">${esc(x.desc || '')}</div>${x.ai ? `<div class="ai">${x.ai.startsWith('분석') || x.ai === '규칙' ? '' : x.ai.includes('AI 모델') ? '' : 'AI·분석: '}${x.ai}</div>` : ''}</li>`).join('');
  $('issues').querySelectorAll('li').forEach(li => li.onclick = () => { S.sel = +li.dataset.i === S.sel ? -1 : +li.dataset.i; render(); });
  $('fixBox').hidden = !r.fixes.length;
  $('fixes').innerHTML = r.fixes.map(f => `<div class="pair"><del>${esc(f.from)}</del><br>${f.tone ? '<span class="tone">톤 참고(가장 가까운 브랜드 예문)</span><br>' : ''}<ins>${esc(f.to) || '(문장 삭제 권장)'}</ins></div>`).join('');
  $('btnJson').disabled = $('btnPrint').disabled = false;
  draw();
}
function draw() {
  const cv = $('cv'); if (!S.imgEl) return;
  cv.width = S.img.width; cv.height = S.img.height;
  const x = cv.getContext('2d'); x.drawImage(S.imgEl, 0, 0);
  const lw = Math.max(3, S.img.width / 300), placed = [], fs = Math.max(18, S.img.width / 45);
  // 같은 영역을 가리키는 여러 이슈는 상자 하나, 번호표 하나(#3·5·9)로 묶는다
  const groups = new Map();
  S.issues.forEach((it, i) => {
    if (!it.box || it.sev === 'pass' && it.cat !== '로고') return;
    const k = [it.box.x, it.box.y, it.box.w, it.box.h].map(v => Math.round(v / 12)).join(',');
    if (!groups.has(k)) groups.set(k, { box: it.box, ids: [], sev: it.sev, dash: it.title.includes('여백') });
    const g = groups.get(k); g.ids.push(i); if (it.sev === 'fail') g.sev = 'fail';
  });
  for (const g of groups.values()) {
    const c = g.sev === 'fail' ? '#e5484d' : g.sev === 'warn' ? '#f5a524' : '#2fa36b';
    const on = g.ids.includes(S.sel), dim = S.sel >= 0 && !on;
    x.globalAlpha = dim ? 0.2 : 1;
    x.strokeStyle = c; x.lineWidth = on ? lw * 2 : lw; x.setLineDash(g.dash ? [lw * 3, lw * 2] : []);
    x.strokeRect(g.box.x, g.box.y, g.box.w, g.box.h);
    x.font = `700 ${fs}px Pretendard, sans-serif`;
    const lab = '#' + g.ids.map(i => i + 1).join('·'), tw = x.measureText(lab).width + 10, th = fs + 6;
    let lx = Math.min(Math.max(0, g.box.x), cv.width - tw), ly = g.box.y - th < 0 ? g.box.y + g.box.h : g.box.y - th;
    // 앞서 그린 번호표와 겹치면 오른쪽(끝이면 아래)으로 민다
    while (placed.some(r => lx < r.x + r.w && lx + tw > r.x && ly < r.y + r.h && ly + th > r.y)) { lx += tw + 4; if (lx + tw > cv.width) { lx = Math.max(0, g.box.x); ly += th + 2; } }
    placed.push({ x: lx, y: ly, w: tw, h: th });
    x.fillStyle = c; x.fillRect(lx, ly, tw, th); x.fillStyle = '#fff'; x.fillText(lab, lx + 5, ly + fs);
  }
  x.globalAlpha = 1; x.setLineDash([]);
}
const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

// ---------- 이벤트 ----------
$('brandSel').onchange = e => { loadBrand(e.target.value); S.img = S.imgEl = null; $('empty').hidden = false; resetResult(); const cv = $('cv'); cv.width = cv.height = 0; };
$('guideFile').onchange = async e => {
  const f = e.target.files[0]; if (!f) return;
  try {
    const g = JSON.parse(await f.text());
    for (const k of ['name', 'palette', 'tone', 'banned', 'channels']) if (!g[k]) throw new Error(`필수 항목 ${k} 없음`);
    if (!g.logoDataUrl) throw new Error('logoDataUrl(로고 PNG data URL) 없음');
    g.logoData = await loadImageData(g.logoDataUrl);
    await setBrand(g); status(`가이드 “${g.name}”을 불러왔습니다.`);
  } catch (err) { status('가이드 JSON 오류: ' + err.message); }
};
const useFile = f => { if (f && f.type.startsWith('image/')) { $('samples').querySelectorAll('button').forEach(x => x.classList.remove('on')); setImage(URL.createObjectURL(f)); } };
$('imgFile').onchange = e => useFile(e.target.files[0]);
const drop = $('drop');
drop.ondragover = e => { e.preventDefault(); drop.classList.add('over'); };
drop.ondragleave = () => drop.classList.remove('over');
drop.ondrop = e => { e.preventDefault(); drop.classList.remove('over'); useFile(e.dataTransfer.files[0]); };
$('runBtn').onclick = run;
document.querySelectorAll('.tabs button').forEach(b => b.onclick = () => {
  document.querySelectorAll('.tabs button').forEach(x => x.classList.toggle('on', x === b)); S.filter = b.dataset.f; if (S.report) render();
});
$('btnJson').onclick = () => {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([JSON.stringify(S.report, null, 2)], { type: 'application/json' }));
  a.download = `guardrail-report-${Date.now()}.json`; a.click();
};
$('btnPrint').onclick = () => window.print();
// 현재 가이드를 로고 포함 JSON으로 내려받아 고쳐 쓰게 한다
$('guideDl').onclick = () => {
  const c = document.createElement('canvas'); c.width = S.logo.width; c.height = S.logo.height; c.getContext('2d').putImageData(S.logo, 0, 0);
  const { logoData, ...g } = S.brand;
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([JSON.stringify({ ...g, logoDataUrl: c.toDataURL('image/png') }, null, 2)], { type: 'application/json' }));
  a.download = `guardrail-guide-${g.id || 'brand'}.json`; a.click();
};

loadBrand('soop').then(boot);
