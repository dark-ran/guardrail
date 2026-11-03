// 검수 로직(화면과 분리). 입력은 ImageData·문자열, 출력은 이슈 배열.

export const hexToRgb = h => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16));
const toHex = c => '#' + c.map(v => Math.round(v).toString(16).padStart(2, '0')).join('').toUpperCase();

// sRGB → CIE Lab(D65)
export function rgbToLab([r, g, b]) {
  const f = v => { v /= 255; return v > 0.04045 ? ((v + 0.055) / 1.055) ** 2.4 : v / 12.92; };
  const R = f(r), G = f(g), B = f(b);
  let x = (R * 0.4124 + G * 0.3576 + B * 0.1805) / 0.95047;
  let y = R * 0.2126 + G * 0.7152 + B * 0.0722;
  let z = (R * 0.0193 + G * 0.1192 + B * 0.9505) / 1.08883;
  const g3 = t => t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116;
  x = g3(x); y = g3(y); z = g3(z);
  return [116 * y - 16, 500 * (x - y), 200 * (y - z)];
}
// shortcut: CIE76 거리. 채도 높은 색에서 ΔE2000보다 크게 나오므로 정밀 색 관리가 필요하면 ΔE2000으로 교체
export const deltaE = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

// WCAG 상대 휘도·명도 대비
export function luminance([r, g, b]) {
  const f = v => { v /= 255; return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4; };
  return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b);
}
export function contrast(a, b) {
  const [l1, l2] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return (l1 + 0.05) / (l2 + 0.05);
}

// 결정적 k-means(가장 먼 점으로 초기화). pts: [[r,g,b],...]
export function kmeans(pts, k, iters = 12) {
  const c = [pts[0]];
  while (c.length < k) {
    let best = null, bd = -1;
    for (const p of pts) {
      const d = Math.min(...c.map(q => (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 + (p[2] - q[2]) ** 2));
      if (d > bd) { bd = d; best = p; }
    }
    if (bd === 0) break;
    c.push(best);
  }
  const lab = new Int32Array(pts.length);
  for (let it = 0; it < iters; it++) {
    const sum = c.map(() => [0, 0, 0, 0]);
    pts.forEach((p, i) => {
      let bi = 0, bd = Infinity;
      c.forEach((q, j) => { const d = (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 + (p[2] - q[2]) ** 2; if (d < bd) { bd = d; bi = j; } });
      lab[i] = bi; const s = sum[bi]; s[0] += p[0]; s[1] += p[1]; s[2] += p[2]; s[3]++;
    });
    sum.forEach((s, j) => { if (s[3]) c[j] = [s[0] / s[3], s[1] / s[3], s[2] / s[3]]; });
  }
  const n = c.map(() => 0); lab.forEach(j => n[j]++);
  return c.map((center, j) => ({ center, share: n[j] / pts.length })).filter(x => x.share > 0);
}

function samplePixels(img, x0 = 0, y0 = 0, w = img.width, h = img.height, max = 4000, skip = null) {
  const step = Math.max(1, Math.floor(Math.sqrt((w * h) / max)));
  const out = [];
  for (let y = y0; y < y0 + h; y += step) for (let x = x0; x < x0 + w; x += step) {
    if (skip && skip(x, y)) continue;
    const i = (y * img.width + x) * 4;
    if (img.data[i + 3] > 200) out.push([img.data[i], img.data[i + 1], img.data[i + 2]]);
  }
  return out;
}

// 사진 영역 추정: 그래픽 타일은 상위 두 색(7비트 양자화)이 대부분을 차지하고, 사진은 색이 잘게 흩어진다
export function photoMask(img) {
  const T = Math.max(16, Math.round(img.width / 30)), cols = Math.ceil(img.width / T), rows = Math.ceil(img.height / T);
  const photo = new Uint8Array(cols * rows);
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
    const cnt = new Map(); let n = 0;
    for (let y = r * T; y < Math.min((r + 1) * T, img.height); y += 2) for (let x = c * T; x < Math.min((c + 1) * T, img.width); x += 2) {
      const i = (y * img.width + x) * 4, k = (img.data[i] >> 1) << 14 | (img.data[i + 1] >> 1) << 7 | img.data[i + 2] >> 1;
      cnt.set(k, (cnt.get(k) || 0) + 1); n++;
    }
    const top = [...cnt.values()].sort((a, b) => b - a);
    photo[r * cols + c] = (top[0] + (top[1] || 0)) / n < 0.5 ? 1 : 0;
  }
  // 글자 가장자리처럼 고립된 타일은 사진이 아님: 이웃 3개 이상이 사진일 때만 유지
  const keep = new Uint8Array(photo.length);
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
    let n = 0;
    for (let dr = -1; dr <= 1; dr++) for (let dc = -1; dc <= 1; dc++) if ((dr || dc) && photo[(r + dr) * cols + c + dc] && r + dr >= 0 && r + dr < rows && c + dc >= 0 && c + dc < cols) n++;
    // 고립된 타일(글자 가장자리)은 버리고, 사진에 둘러싸인 매끈한 타일(흐린 배경)은 채운다
    keep[r * cols + c] = photo[r * cols + c] ? (n >= 3 ? 1 : 0) : (n >= 6 ? 1 : 0);
  }
  const share = keep.reduce((a, b) => a + b, 0) / keep.length;
  return { share, isPhoto: (x, y) => keep[Math.floor(y / T) * cols + Math.floor(x / T)] === 1 };
}

// 1) 브랜드 색 준수: 주요 색 군집이 팔레트에서 얼마나 먼지
export function checkPalette(img, brand) {
  const pal = brand.palette.map(p => ({ ...p, lab: rgbToLab(hexToRgb(p.hex)) }));
  const pm = photoMask(img);
  const pts = samplePixels(img, 0, 0, img.width, img.height, 4000, pm.share > 0.03 ? pm.isPhoto : null);
  const issues = [];
  if (pm.share > 0.03) issues.push({ sev: 'pass', cat: '색상', ai: '분석: 타일 색 다양도', title: `사진 영역 ${Math.round(pm.share * 100)}%는 팔레트 검사에서 제외`, desc: '사진 속 자연색은 브랜드 팔레트 대상이 아니므로 그래픽 영역만 검사합니다.' });
  if (pts.length < 50) return { issues, offShare: 0 };
  const clusters = kmeans(pts, 7).filter(c => c.share >= 0.03).sort((a, b) => b.share - a.share);
  let offShare = 0;
  for (const c of clusters) {
    const lab = rgbToLab(c.center);
    const near = pal.reduce((m, p) => { const d = deltaE(lab, p.lab); return d < m.d ? { d, p } : m; }, { d: Infinity });
    const hex = toHex(c.center), pct = Math.round(c.share * 100);
    if (near.d > brand.paletteMaxDeltaE) {
      offShare += c.share;
      issues.push({ sev: c.share >= 0.1 ? 'fail' : 'warn', cat: '색상', ai: '분석: k-means 색 군집', box: colorBox(img, c.center, pm.share > 0.03 ? pm.isPhoto : null),
        title: `팔레트 밖 색상 ${hex} (화면의 ${pct}%)`,
        desc: `가장 가까운 브랜드 색은 ${near.p.name} ${near.p.hex}이며 색차 ΔE ${near.d.toFixed(0)} (허용 ${brand.paletteMaxDeltaE}).`, swatch: hex });
    }
  }
  if (!issues.some(i => i.sev !== 'pass')) issues.push({ sev: 'pass', cat: '색상', ai: '분석: k-means 색 군집', title: '주요 색상이 모두 브랜드 팔레트 안에 있습니다',
    desc: clusters.map(c => `${toHex(c.center)} ${Math.round(c.share * 100)}%`).join(' · ') });
  return { issues, offShare };
}

// 해당 색에 가까운 픽셀들이 모인 영역(상하위 2%를 잘라 잡음 제거)
function colorBox(img, rgb, skip) {
  const xs = [], ys = [], step = Math.max(1, Math.floor(img.width / 200));
  for (let y = 0; y < img.height; y += step) for (let x = 0; x < img.width; x += step) {
    if (skip && skip(x, y)) continue;
    const i = (y * img.width + x) * 4;
    if ((img.data[i] - rgb[0]) ** 2 + (img.data[i + 1] - rgb[1]) ** 2 + (img.data[i + 2] - rgb[2]) ** 2 < 900) { xs.push(x); ys.push(y); }
  }
  if (xs.length < 10) return null;
  xs.sort((a, b) => a - b); ys.sort((a, b) => a - b);
  const q = (a, t) => a[Math.floor((a.length - 1) * t)];
  const x0 = q(xs, 0.02), x1 = q(xs, 0.98) + step, y0 = q(ys, 0.02), y1 = q(ys, 0.98) + step;
  return { x: x0, y: y0, w: Math.min(img.width, x1) - x0, h: Math.min(img.height, y1) - y0 };
}

// 2) 텍스트 명도 대비: 글줄 영역을 2색으로 군집해 글자색·배경색 추정
export function checkContrast(img, lines, brand) {
  const issues = [];
  for (const l of lines) {
    const pts = samplePixels(img, l.box.x, l.box.y, l.box.w, l.box.h, 1500);
    if (pts.length < 20) continue;
    const cl = kmeans(pts, 2).sort((a, b) => a.share - b.share);
    if (cl.length < 2) continue;
    // 글자색은 소수 군집 중 배경에서 먼 쪽(90백분위) 픽셀로 추정: 안티에일리어싱 경계색이 대비를 낮게 끌어내리지 않게
    const bg = cl[1].center, d2 = p => (p[0] - bg[0]) ** 2 + (p[1] - bg[1]) ** 2 + (p[2] - bg[2]) ** 2;
    const fg = pts.filter(p => d2(p) > d2(cl[0].center) * 0.25).sort((a, b) => d2(a) - d2(b));
    const txt = fg.length ? fg[Math.floor(fg.length * 0.9)] : cl[0].center;
    const ratio = contrast(txt, bg);
    const large = l.fontPx >= 24;
    const need = large ? brand.minContrastLarge : brand.minContrast;
    if (ratio < need) issues.push({ sev: 'fail', cat: '대비', ai: 'AI 모델(OCR) + 분석(색 군집)', box: l.box,
      title: `명도 대비 부족 ${ratio.toFixed(2)}:1 — “${l.text}”`,
      desc: `${large ? '큰 글자' : '일반 글자'} 기준 ${need}:1 이상 필요(WCAG 1.4.3). 글자 ${toHex(txt)} / 배경 ${toHex(bg)}.` });
  }
  if (!issues.length && lines.length) issues.push({ sev: 'pass', cat: '대비', ai: 'AI 모델(OCR) + 분석(색 군집)', title: `글줄 ${lines.length}개 모두 명도 대비 기준 충족`, desc: 'WCAG 1.4.3 기준(일반 4.5:1, 큰 글자 3:1).' });
  return issues;
}

// 3) 최소 글자 크기·헤드라인 길이·규격
export function checkLayout(img, lines, brand, chan) {
  const ch = brand.channels[chan], issues = [];
  const scale = ch.w / img.width; // 업로드 해상도가 달라도 채널 기준 px로 환산
  for (const l of lines) {
    if (l.fontPx * scale < ch.minTextPx) issues.push({ sev: 'warn', cat: '가독성', ai: 'AI 모델(OCR)', box: l.box,
      title: `글자가 너무 작음(약 ${Math.round(l.fontPx * scale)}px) — “${l.text.slice(0, 24)}”`,
      desc: `${ch.label} 기준 최소 ${ch.minTextPx}px. 모바일 피드에서 읽기 어렵습니다.` });
  }
  const head = [...lines].sort((a, b) => b.fontPx - a.fontPx)[0];
  if (head) {
    const n = head.text.replace(/\s/g, '').length;
    if (n > ch.headlineMax) issues.push({ sev: 'warn', cat: '카피', ai: 'AI 모델(OCR)', box: head.box, title: `헤드라인이 깁니다(${n}자)`, desc: `${ch.label} 권장 ${ch.headlineMax}자 이하.` });
  }
  const sameRatio = Math.abs(img.width / img.height - ch.w / ch.h) < 0.01;
  if (img.width === ch.w && img.height === ch.h) issues.push({ sev: 'pass', cat: '규격', title: `규격 일치 ${img.width}×${img.height}`, desc: ch.label });
  else issues.push({ sev: sameRatio ? 'warn' : 'fail', cat: '규격', title: `규격 불일치 ${img.width}×${img.height}`, desc: `${ch.label}${sameRatio ? ' — 비율은 같지만 해상도가 다릅니다.' : ' — 비율이 달라 잘리거나 여백이 생깁니다.'}` });
  return issues;
}

// 4) 로고 탐지: 투명 배경을 흰색·검은색으로 채운 두 템플릿의 정규상관(NCC) 최댓값
export function findLogo(img, logo, minW = 0.12) {
  const W = 360, f = W / img.width, H = Math.round(img.height * f);
  const g = gray(resize(img, W, H));
  let best = { score: -1 };
  // 가이드의 최소 로고 폭보다 조금 작은 크기부터 찾는다(그보다 작으면 위반이므로 탐지 대상 아님)
  for (let lw = Math.round(W * minW * 0.8); lw <= W * 0.45; lw = Math.ceil(lw * 1.07)) {
    const lh = Math.max(4, Math.round(logo.height * lw / logo.width));
    if (lh >= H) break;
    const t = resize(logo, lw, lh), n = lw * lh;
    for (const fill of [255, 0]) {
      const tg = new Float32Array(n);
      for (let i = 0; i < n; i++) { const a = t.data[i * 4 + 3] / 255; tg[i] = a * (0.299 * t.data[i * 4] + 0.587 * t.data[i * 4 + 1] + 0.114 * t.data[i * 4 + 2]) + (1 - a) * fill; }
      let tm = 0; for (let i = 0; i < n; i++) tm += tg[i]; tm /= n;
      let tv = 0; for (let i = 0; i < n; i++) { tg[i] -= tm; tv += tg[i] ** 2; }
      const ncc = (x, y) => {
        let s = 0, s2 = 0, st = 0;
        for (let r = 0; r < lh; r++) { const o = (y + r) * W + x, q = r * lw; for (let c = 0; c < lw; c++) { const v = g[o + c]; s += v; s2 += v * v; st += v * tg[q + c]; } }
        const iv = s2 - s * s / n;
        return iv > 1e-6 ? st / Math.sqrt(iv * tv) : 0;
      };
      let loc = { score: -1 };
      for (let y = 0; y + lh <= H; y += 2) for (let x = 0; x + lw <= W; x += 2) { const s = ncc(x, y); if (s > loc.score) loc = { score: s, x, y }; }
      // 거친 탐색 최적점 주변 정밀 탐색
      for (let y = Math.max(0, loc.y - 1); y <= Math.min(H - lh, loc.y + 1); y++) for (let x = Math.max(0, loc.x - 1); x <= Math.min(W - lw, loc.x + 1); x++) { const s = ncc(x, y); if (s > loc.score) loc = { score: s, x, y }; }
      if (loc.score > best.score) best = { ...loc, lw, lh };
    }
  }
  if (best.score < 0.7) return null;
  return { score: best.score, box: { x: Math.round(best.x / f), y: Math.round(best.y / f), w: Math.round(best.lw / f), h: Math.round(best.lh / f) } };
}

// 사진 영역을 흰색으로 지운 이미지(사진이 OCR 글줄 분석을 흐트러뜨리지 않게)
export function maskPhoto(img, pm) {
  const out = new ImageData(new Uint8ClampedArray(img.data), img.width, img.height);
  for (let y = 0; y < img.height; y++) for (let x = 0; x < img.width; x++) if (pm.isPhoto(x, y)) { const i = (y * img.width + x) * 4; out.data[i] = out.data[i + 1] = out.data[i + 2] = 255; }
  return out;
}

// 저대비 글자도 OCR이 읽도록 타일마다 채널 대비를 늘린 보조 이미지
export function stretchTiles(img, T = 48) {
  const out = new ImageData(img.width, img.height), d = img.data, o = out.data;
  for (let ty = 0; ty < img.height; ty += T) for (let tx = 0; tx < img.width; tx += T) {
    const lo = [255, 255, 255], hi = [0, 0, 0];
    for (let y = ty; y < Math.min(ty + T, img.height); y++) for (let x = tx; x < Math.min(tx + T, img.width); x++) {
      const i = (y * img.width + x) * 4; for (let c = 0; c < 3; c++) { lo[c] = Math.min(lo[c], d[i + c]); hi[c] = Math.max(hi[c], d[i + c]); }
    }
    const c = [0, 1, 2].reduce((m, k) => hi[k] - lo[k] > hi[m] - lo[m] ? k : m, 0), r = hi[c] - lo[c];
    for (let y = ty; y < Math.min(ty + T, img.height); y++) for (let x = tx; x < Math.min(tx + T, img.width); x++) {
      const i = (y * img.width + x) * 4, v = r < 16 ? 255 : (d[i + c] - lo[c]) * 255 / r;
      o[i] = o[i + 1] = o[i + 2] = v; o[i + 3] = 255;
    }
  }
  return out;
}

export function checkLogo(img, logoHit, lines, brand) {
  if (!logoHit) return [{ sev: 'fail', cat: '로고', ai: '분석: 템플릿 정규상관', title: '브랜드 로고를 찾지 못했습니다', desc: `로고가 없거나, 최소 폭(화면의 ${Math.round((brand.logoMinWidth || 0) * 100)}%)보다 작거나, 변형·재채색되어 가이드 원본과 다릅니다.` }];
  const b = logoHit.box, pad = Math.round(b.h * brand.logoClearSpace);
  const minW = Math.round(img.width * (brand.logoMinWidth || 0));
  const z = { x: b.x - pad, y: b.y - pad, w: b.w + pad * 2, h: b.h + pad * 2 };
  const issues = [{ sev: 'pass', cat: '로고', ai: '분석: 템플릿 정규상관', box: b, title: `로고 확인(일치도 ${(logoHit.score * 100).toFixed(0)}%)`, desc: `크기 ${b.w}×${b.h}px` }];
  if (b.w < minW) issues.push({ sev: 'warn', cat: '로고', ai: '분석: 템플릿 정규상관', box: b, title: `로고가 작습니다(${b.w}px)`, desc: `가이드 최소 폭은 화면 폭의 ${Math.round(brand.logoMinWidth * 100)}%(${minW}px)입니다.` });
  const edge = z.x < 0 || z.y < 0 || z.x + z.w > img.width || z.y + z.h > img.height;
  const hit = lines.filter(l => !l.isLogo && overlap(z, l.box) > 0);
  if (edge || hit.length) issues.push({ sev: 'fail', cat: '로고', ai: '분석(템플릿 정규상관) + AI 모델(OCR)', box: z,
    title: '로고 보호 여백 침범',
    desc: `로고 높이의 ${brand.logoClearSpace}배(${pad}px) 여백이 필요합니다. ${edge ? '화면 가장자리에 붙어 있습니다. ' : ''}${hit.length ? `침범한 글: “${hit.map(l => l.text).join('”, “')}”` : ''}` });
  return issues;
}

// 5) 금칙어·규제어: 띄어쓰기 무시 + 자모 단위 근사 매칭(OCR 오인식 "손실"→"소심" 같은 경우 보정)
const CHO = 'ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ', JUNG = 'ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ', JONG = ' ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ';
export function jamo(str) {
  let o = '';
  for (const ch of str) {
    const c = ch.charCodeAt(0) - 0xAC00;
    if (c >= 0 && c < 11172) o += CHO[c / 588 | 0] + JUNG[(c % 588) / 28 | 0] + (c % 28 ? JONG[c % 28] : '');
    else o += ch;
  }
  return o;
}
function lev(a, b) {
  const d = Array.from({ length: b.length + 1 }, (_, j) => j);
  for (let i = 1; i <= a.length; i++) {
    let prev = d[0]; d[0] = i;
    for (let j = 1; j <= b.length; j++) { const t = d[j]; d[j] = Math.min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] === b[j - 1] ? 0 : 1)); prev = t; }
  }
  return d[b.length];
}
// 정확히 일치하면 'exact', 자모 편집거리가 단어 길이의 20% 이하이면 'fuzzy'(4음절 이상만)
export function matchWord(text, word) {
  const flat = text.replace(/\s/g, ''), w = word.replace(/\s/g, '');
  if (flat.includes(w)) return { kind: 'exact', found: w };
  if ([...w].length < 4) return null;
  const jw = jamo(w), tol = Math.floor(jw.length * 0.2), chars = [...flat];
  let best = null;
  for (let i = 0; i < chars.length; i++) for (const L of [w.length - 1, w.length, w.length + 1]) {
    if (i + L > chars.length) continue;
    const seg = chars.slice(i, i + L).join(''), dist = lev(jamo(seg), jw);
    if (dist <= tol && (!best || dist < best.dist)) best = { kind: 'fuzzy', found: seg, dist };
  }
  return best;
}
export function checkBanned(units, brand) {
  const issues = [], fixes = [];
  for (const u of units) {
    const hits = brand.banned.map(b => ({ b, m: matchWord(u.text, b.word) })).filter(h => h.m);
    for (const { b, m } of hits) issues.push({ sev: 'fail', cat: '규제·금칙어', ai: m.kind === 'fuzzy' ? 'AI 모델(OCR) + 자모 근사 매칭' : u.box ? 'AI 모델(OCR) + 규칙' : '규칙', box: u.box,
      title: `“${b.word}” — ${b.type}${m.kind === 'fuzzy' ? ` (“${m.found}”로 인식, 유사 표현)` : ''}`,
      desc: `근거: ${b.basis}. 위치: ${u.src} “${u.text}”${b.suggest ? ` → 대체 제안: “${b.suggest}”` : ''}` });
    if (hits.length) {
      let fixed = u.text;
      for (const { b, m } of hits) fixed = fixed.replace(new RegExp([...m.found].map(ch => ch.replace(/[.*+?^${}()|[\]\\%]/g, '\\$&')).join('\\s*'), 'g'), b.suggest);
      fixes.push({ from: u.text, to: fixed.replace(/\s{2,}/g, ' ').replace(/^[\s,]+|[\s,]+$/g, '').trim() });
    }
  }
  return { issues, fixes };
}

// 6) 톤 일치도: 브랜드 예문(onBrand)과 금지 톤 예문(offBrand)에 대한 임베딩 유사도 차이
export function toneVerdict(vec, onVecs, offVecs, brand) {
  const sims = vs => vs.map(v => dot(v, vec)).map((s, i) => ({ s, i })).sort((a, b) => b.s - a.s);
  const on = sims(onVecs), off = sims(offVecs);
  const top = a => (a[0].s + a[1].s + a[2].s) / 3;
  const margin = top(on) - top(off);
  const score = Math.max(0, Math.min(100, Math.round(50 + margin * 600)));
  return { margin, score, nearOn: brand.tone.onBrand[on[0].i], nearOff: brand.tone.offBrand[off[0].i] };
}

export const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
export function overlap(a, b) {
  const w = Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x), h = Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y);
  return w > 0 && h > 0 ? (w * h) / (b.w * b.h) : 0;
}
function resize(img, w, h) {
  const c = new OffscreenCanvas(w, h), x = c.getContext('2d');
  const s = new OffscreenCanvas(img.width, img.height); s.getContext('2d').putImageData(img, 0, 0);
  x.imageSmoothingQuality = 'high'; x.drawImage(s, 0, 0, w, h);
  return x.getImageData(0, 0, w, h);
}
function gray(img) {
  const g = new Float32Array(img.width * img.height);
  for (let i = 0; i < g.length; i++) g[i] = 0.299 * img.data[i * 4] + 0.587 * img.data[i * 4 + 1] + 0.114 * img.data[i * 4 + 2];
  return g;
}
