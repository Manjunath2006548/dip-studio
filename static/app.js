/* ============================================================
   DIP Studio — client controller
   ============================================================ */
'use strict';

const CONDITIONS = ['day', 'midday', 'evening', 'night'];
const COND_LABEL = { day: 'Day', midday: 'Midday', evening: 'Evening', night: 'Night' };
const COND_DESC = {
  day: 'Normal daylight — the real phone photograph.',
  midday: 'High illumination — histogram shifted right, highlights partly clipped.',
  evening: 'Reduced natural light — narrow histogram crowded into the low band.',
  night: 'Poor / artificial lighting — most pixels piled at the dark end and noisy.',
};

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

const state = {
  files: [],
  mode: 'simulate',
  active: 'day',
  result: null,
  busy: false,
};

/* ------------------------------ boot ------------------------------ */
window.addEventListener('DOMContentLoaded', () => {
  bindTabs();
  bindDropzone();
  bindModes();
  bindControls();
  bindCompare();
  bindLightbox();
  bindDownloads();
  bindAccount();
  fetchHealth();
  analyze();               // immediately show the report results for the sample scene
});

/* ------------------------------ helpers ------------------------------ */
function fmt(v, digits = 1) {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  if (!Number.isFinite(v)) return '∞';
  return Number(v).toFixed(digits);
}

function pct(v, digits = 1) {
  if (v === null || v === undefined) return '—';
  return `${(v * 100).toFixed(digits)} %`;
}

function sizeLabel(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function setStatus(kind, text) {
  const el = $('#status');
  el.className = `status ${kind || ''}`.trim();
  $('#statusText').textContent = text;
}

function setBusy(on, title, sub) {
  state.busy = on;
  const busy = $('#busy');
  busy.classList.toggle('open', on);
  busy.setAttribute('aria-hidden', String(!on));
  if (title) $('#busyTitle').textContent = title;
  if (sub) $('#busySub').textContent = sub;
  $('#runBtn').disabled = on;
}

function toast(message, kind = '') {
  const wrap = $('#toastWrap');
  const el = document.createElement('div');
  el.className = `toast ${kind}`.trim();
  el.textContent = message;
  wrap.appendChild(el);
  setTimeout(() => {
    el.style.opacity = '0';
    el.style.transform = 'translateY(6px)';
    el.style.transition = 'all .2s ease';
    setTimeout(() => el.remove(), 220);
  }, 4200);
}

/* ------------------------------ health ------------------------------ */
async function fetchHealth() {
  try {
    const res = await fetch('/api/health');
    const h = await res.json();
    $('#stackChip').textContent = `Python ${h.python} · OpenCV ${h.opencv} · NumPy ${h.numpy}`;
  } catch (_) { /* server is serving the page, so this is non-fatal */ }
}

/* ------------------------------ tabs ------------------------------ */
function bindTabs() {
  $$('.tab').forEach((tab) => {
    tab.addEventListener('click', () => activateTab(tab.dataset.tab));
  });
}

function activateTab(name) {
  $$('.tab').forEach((t) => {
    const on = t.dataset.tab === name;
    t.classList.toggle('active', on);
    t.setAttribute('aria-selected', String(on));
  });
  $$('.tabpane').forEach((p) => p.classList.toggle('active', p.id === `pane-${name}`));
  if (name === 'techniques' && state.result) ensureTechniques(state.result);
}

/* ------------------------------ dropzone / files ------------------------------ */
function bindDropzone() {
  const dz = $('#dropzone');
  const input = $('#fileInput');

  dz.addEventListener('click', () => input.click());
  dz.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); input.click(); }
  });
  input.addEventListener('change', () => {
    addFiles(Array.from(input.files || []));
    input.value = '';
  });

  ['dragenter', 'dragover'].forEach((ev) => dz.addEventListener(ev, (e) => {
    e.preventDefault();
    dz.classList.add('dragover');
  }));
  ['dragleave', 'drop'].forEach((ev) => dz.addEventListener(ev, (e) => {
    e.preventDefault();
    dz.classList.remove('dragover');
  }));
  dz.addEventListener('drop', (e) => {
    const files = Array.from(e.dataTransfer?.files || []);
    if (files.length) addFiles(files);
  });

  window.addEventListener('paste', (e) => {
    const items = Array.from(e.clipboardData?.items || []);
    const files = items.filter((i) => i.kind === 'file' && i.type.startsWith('image/'))
      .map((i) => i.getAsFile()).filter(Boolean);
    if (files.length) addFiles(files);
  });

  $('#sampleBtn').addEventListener('click', () => {
    clearFiles();
    setMode('simulate');
    setStatus('', 'Ready — sample scene.');
    analyze();
  });
}

function addFiles(files) {
  const images = files.filter((f) => f.type.startsWith('image/') || /\.(jpe?g|png|webp|bmp)$/i.test(f.name));
  if (!images.length) { toast('Those files are not images.', 'err'); return; }

  const room = 4 - state.files.length;
  if (room <= 0) { toast('Four images is the maximum — remove one first.', 'err'); return; }
  const accepted = images.slice(0, room);

  accepted.forEach((file) => {
    state.files.push({
      file,
      url: URL.createObjectURL(file),
      label: CONDITIONS[state.files.length] || 'day',
    });
  });
  if (images.length > room) toast('Only the first four images were added.', 'err');
  $('#detectResult').hidden = true;

  if (state.files.length > 1 && state.mode !== 'direct') {
    setMode('direct');
    toast('Several images added — switched to “use as they are”.');
  }
  assignLabels();
  renderFiles();
  setStatus('', `${state.files.length} image${state.files.length > 1 ? 's' : ''} ready — run the analysis.`);
}

function assignLabels() {
  const used = new Set();
  state.files.forEach((f, i) => {
    if (state.files.length === 1) { f.label = 'day'; return; }
    if (!CONDITIONS.includes(f.label) || used.has(f.label)) {
      f.label = CONDITIONS.find((c) => !used.has(c)) || CONDITIONS[i] || 'day';
    }
    used.add(f.label);
  });
}

function clearFiles() {
  state.files.forEach((f) => URL.revokeObjectURL(f.url));
  state.files = [];
  renderFiles();
}

function renderFiles() {
  const list = $('#fileList');
  list.textContent = '';
  state.files.forEach((entry, index) => {
    const row = document.createElement('div');
    row.className = 'file-row';

    const img = document.createElement('img');
    img.className = 'file-thumb';
    img.src = entry.url;
    img.alt = '';
    row.appendChild(img);

    const name = document.createElement('div');
    name.className = 'file-name';
    const strong = document.createElement('span');
    strong.textContent = entry.file.name;
    const meta = document.createElement('em');
    meta.textContent = `${sizeLabel(entry.file.size)} · ${COND_LABEL[entry.label] || '—'}`;
    name.append(strong, meta);
    row.appendChild(name);

    const remove = document.createElement('button');
    remove.className = 'file-remove';
    remove.type = 'button';
    remove.title = 'Remove image';
    remove.setAttribute('aria-label', `Remove ${entry.file.name}`);
    remove.textContent = '×';
    remove.addEventListener('click', () => {
      URL.revokeObjectURL(entry.url);
      state.files.splice(index, 1);
      assignLabels();
      renderFiles();
      setStatus('', state.files.length ? 'Image removed — run again.' : 'No files selected — the sample scene will be used.');
    });
    row.appendChild(remove);

    if (state.mode === 'direct' && state.files.length > 1) {
      const wrap = document.createElement('label');
      wrap.className = 'file-label';
      const span = document.createElement('span');
      span.textContent = 'Condition';
      const select = document.createElement('select');
      CONDITIONS.forEach((c) => {
        const opt = document.createElement('option');
        opt.value = c;
        opt.textContent = COND_LABEL[c];
        opt.selected = entry.label === c;
        select.appendChild(opt);
      });
      select.addEventListener('change', () => {
        const target = state.files.find((f) => f.label === select.value && f !== entry);
        if (target) {
          target.label = entry.label;
        }
        entry.label = select.value;
        renderFiles();
      });
      wrap.append(span, select);
      row.appendChild(wrap);
    }

    list.appendChild(row);
  });
}

function bindModes() {
  $$('input[name="mode"]').forEach((radio) => {
    radio.addEventListener('change', () => {
      if (radio.checked) setMode(radio.value);
    });
  });
}

function setMode(mode) {
  state.mode = mode;
  $$('input[name="mode"]').forEach((r) => { r.checked = r.value === mode; });
  const note = $('#modeNote');
  note.textContent = mode === 'simulate'
    ? 'Upload one photo — its lighting condition is detected automatically and the other three are generated from it.'
    : 'Each uploaded image is analysed as-is. Give every condition a distinct label — up to four images, one per condition.';
  if (mode === 'direct') assignLabels();
  $('#detectResult').hidden = true;
  renderFiles();
}

/* ------------------------------ controls ------------------------------ */
function bindControls() {
  $('#runBtn').addEventListener('click', analyze);
  document.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') analyze();
  });
  ['sizeSelect', 'tileSelect', 'clipInput', 'loInput', 'hiInput',
    'gainMidday', 'gainEvening', 'gainNight'].forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.addEventListener('change', () => {
      if (state.result) setStatus('', 'Parameters changed — run the analysis again.');
    });
  });
}

function collectParams() {
  const [width, height] = ($('#sizeSelect').value || '640x480').split('x').map(Number);
  return {
    width,
    height,
    tile: Number($('#tileSelect').value),
    clip_limit: Number($('#clipInput').value),
    stretch_lo: Number($('#loInput').value),
    stretch_hi: Number($('#hiInput').value),
    gains: {
      midday: Number($('#gainMidday').value),
      evening: Number($('#gainEvening').value),
      night: Number($('#gainNight').value),
    },
  };
}

/* ------------------------------ analysis ------------------------------ */
async function analyze() {
  if (state.busy) return;
  if (state.mode === 'direct' && state.files.length > 1) {
    const labels = state.files.map((f) => f.label);
    if (new Set(labels).size !== labels.length) {
      toast('Two images share the same condition label.', 'err');
      return;
    }
  }

  setBusy(true, 'Running the pipeline…', 'Decoding, cropping and resizing the images');
  setStatus('busy', 'Processing — histogram, CDF, mapping and metrics…');

  const form = new FormData();
  state.files.forEach((f) => form.append('files', f.file, f.file.name));
  form.append('mode', state.mode);
  form.append('labels', JSON.stringify(state.files.map((f) => f.label)));
  form.append('params', JSON.stringify(collectParams()));

  const hints = [
    'Decoding, cropping and resizing the images',
    'Counting grey levels and building the PDF / CDF',
    'Applying the equalization mapping and CLAHE',
    'Measuring contrast, entropy and noise',
    'Rendering figures and comparison charts',
  ];
  let hintIndex = 0;
  const hintTimer = setInterval(() => {
    hintIndex = (hintIndex + 1) % hints.length;
    $('#busySub').textContent = hints[hintIndex];
  }, 900);

  try {
    const res = await fetch('/api/run', { method: 'POST', body: form });
    const data = await res.json().catch(() => ({}));
    if (!res.ok || !data.ok) throw new Error(data.detail || data.error || `Server error (${res.status})`);
    state.result = data;
    if (!data.conditions.includes(state.active)) state.active = data.conditions[0];
    renderAll();
    renderDetection(data);
    updateFigureCaptions(data);
    const source = data.source.used_sample ? 'sample scene' : data.source.names.join(', ');
    const cacheNote = data.cached ? ` · cached (pipeline ${Math.round(data.pipeline_ms ?? data.elapsed_ms)} ms)` : '';
    setStatus('ok', `Done in ${data.elapsed_ms} ms${cacheNote} · ${data.conditions.length} condition(s) · ${source}`);
  } catch (err) {
    console.error(err);
    setStatus('err', `Analysis failed: ${err.message}`);
    toast(err.message, 'err');
  } finally {
    clearInterval(hintTimer);
    setBusy(false);
  }
}

/* ------------------------------ rendering ------------------------------ */
function updateFigureCaptions(data) {
  const p = data.params || {};
  const clip = p.clip_limit ?? 2.0;
  const tile = p.tile ?? 8;
  const lo = p.stretch_lo ?? 1;
  const hi = p.stretch_hi ?? 99;
  const src = (data.source && data.source.names && data.source.names[0]) || 'sample scene';
  const n = (data.conditions || []).length;

  const captions = {
    '#overviewCaption': ['#overviewImage',
      `Figure 7 equivalent — original grayscale, global HE, CLAHE (clip ${Number(clip).toFixed(1)}, ${tile} × ${tile} tiles) and ${lo}–${hi}% contrast stretching. Measured from ${src}.`],
    '#techniquesCaption': ['#techniquesChart',
      `Contrast, entropy and noise-based SNR for each technique across ${n} conditions, measured from ${src}.`],
    '#metricsCaption': ['#metricsChart',
      `Mean, contrast, entropy and noise-based SNR before and after global HE across ${n} conditions, measured from ${src}.`],
  };
  Object.entries(captions).forEach(([capSel, [imgSel, text]]) => {
    const cap = $(capSel);
    if (cap) cap.textContent = text;
    const img = $(imgSel);
    if (img) img.dataset.caption = text;
  });
}

function renderDetection(data) {
  const card = $('#detectResult');
  if (!card) return;
  const d = data.detection;
  if (!d) { card.hidden = true; return; }
  card.hidden = false;
  $('#detectCondition').textContent = d.label || '—';
  const f = d.features || {};
  $('#detectDetail').textContent =
    `Confidence ${Math.round((d.confidence || 0) * 100)}% · mean ${f.mean ?? '—'} · dark ${Math.round((f.dark_fraction || 0) * 100)}% · noise σ ${f.noise_sigma ?? '—'}`;
}

function renderAll() {
  const data = state.result;
  if (!data) return;
  renderKpis(data);
  renderCondSwitch(data);
  renderCondition(data);
  renderTables(data);
  renderTechniques(data);
  const source = data.source.used_sample ? 'sample' : data.source.names.join(', ');
  const cacheNote = data.cached ? ` (cached · pipeline ${Math.round(data.pipeline_ms ?? data.elapsed_ms)} ms)` : '';
  $('#runMeta').textContent = `${data.conditions.length} conditions · ${data.elapsed_ms} ms${cacheNote} · ${source}`;
}

function renderKpis(data) {
  const rs = data.results;
  const conds = data.conditions;
  const mean0 = conds.map((c) => rs[c].metrics.original.mean);
  const mean1 = conds.map((c) => rs[c].metrics.he.mean);
  const sig0 = conds.map((c) => rs[c].metrics.original.contrast_std);
  const sig1 = conds.map((c) => rs[c].metrics.he.contrast_std);
  const ratios = conds.map((c, i) => (sig0[i] > 0 ? sig1[i] / sig0[i] : 1));
  const ambe = conds.map((c) => rs[c].metrics.he.ambe).filter((v) => Number.isFinite(v));
  const maxDiff = Math.max(...conds.map((c) => rs[c].max_diff_cv));

  const spread = `${Math.min(...mean0).toFixed(0)}–${Math.max(...mean0).toFixed(0)}`;
  const spreadAfter = `${Math.min(...mean1).toFixed(0)}–${Math.max(...mean1).toFixed(0)}`;
  $('#kpiSpread').textContent = spread;
  $('#kpiSpread').title = `Original mean: ${spread} grey levels`;
  $('#kpiSpreadSub').textContent = `grey levels · after HE ${spreadAfter}`;

  const avgRatio = ratios.reduce((a, b) => a + b, 0) / ratios.length;
  $('#kpiGain').textContent = `${avgRatio.toFixed(2)}×`;
  $('#kpiGainSub').textContent = `mean σ ${(sig0.reduce((a, b) => a + b, 0) / sig0.length).toFixed(1)} → ${(sig1.reduce((a, b) => a + b, 0) / sig1.length).toFixed(1)}`;

  const avgAmbe = ambe.reduce((a, b) => a + b, 0) / Math.max(1, ambe.length);
  $('#kpiAmbe').textContent = `${avgAmbe.toFixed(1)} gl`;
  $('#kpiAmbeSub').textContent = 'average |Δmean| after global HE';

  $('#kpiVerify').textContent = `${maxDiff} gl`;
  $('#kpiVerifySub').textContent = 'max |user HE − cv2.equalizeHist|';
}

function renderCondSwitch(data) {
  const wrap = $('#condSwitch');
  wrap.textContent = '';
  data.conditions.forEach((c) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = `cond-btn ${c === state.active ? 'active' : ''}`;
    btn.textContent = COND_LABEL[c];
    btn.setAttribute('aria-pressed', String(c === state.active));
    btn.addEventListener('click', () => {
      state.active = c;
      $$('.cond-btn', wrap).forEach((b) => {
        b.classList.toggle('active', b.textContent === COND_LABEL[c]);
        b.setAttribute('aria-pressed', String(b.textContent === COND_LABEL[c]));
      });
      renderCondition(state.result);
    });
    wrap.appendChild(btn);
  });
}

function renderCondition(data) {
  const c = state.active;
  if (!data.conditions.includes(c)) return;
  const entry = data.results[c];
  const m0 = entry.metrics.original;
  const m1 = entry.metrics.he;

  const panelUrl = (data.figures && data.figures[`panel:${c}`]) || data.figure_urls.panel[c];
  const panelImg = $('#panelImage');
  panelImg.classList.add('loading');
  panelImg.onload = () => panelImg.classList.remove('loading');
  panelImg.onerror = () => {
    panelImg.classList.remove('loading');
    toast('The panel figure could not be loaded — run the analysis again.', 'err');
  };
  panelImg.src = panelUrl;
  panelImg.alt = `${COND_LABEL[c]} condition analysis panel`;
  $('#panelImage').dataset.caption = `${COND_LABEL[c]} condition — original, grayscale, equalized image, histogram + CDF, mapping T(r), equalized histogram + CDF`;
  $('#panelCaption').textContent =
    `Figure — ${COND_LABEL[c]} condition: (a) original colour, (b) grayscale, (c) equalized image, (d) original histogram with CDF, (e) transformation s = T(r), (f) equalized histogram with CDF.`;

  const gain = m0.contrast_std > 0 ? m1.contrast_std / m0.contrast_std : 1;
  const summary = document.createElement('div');
  summary.innerHTML =
    `<strong>${COND_LABEL[c]}</strong> — ${COND_DESC[c]} ` +
    `Mean <strong>${fmt(m0.mean)}</strong> → <strong>${fmt(m1.mean)}</strong>, ` +
    `σ <strong>${fmt(m0.contrast_std)}</strong> → <strong>${fmt(m1.contrast_std)}</strong> (${gain.toFixed(2)}×), ` +
    `entropy ${fmt(m0.entropy, 2)} → ${fmt(m1.entropy, 2)} bits.` +
    `<span class="badge">max |HE − cv2| = ${entry.max_diff_cv}</span>`;
  if (entry.tails.dark_frac > 0.2) {
    summary.insertAdjacentHTML('beforeend',
      `<span class="badge">${pct(entry.tails.dark_frac)} of pixels ≤ 40</span>`);
  }
  if (entry.tails.bright_frac > 0.02) {
    summary.insertAdjacentHTML('beforeend',
      `<span class="badge">${pct(entry.tails.bright_frac)} of pixels ≥ 250</span>`);
  }
  const strip = $('#condSummary');
  strip.textContent = '';
  strip.appendChild(summary);

  const thumbs = data.thumbs[c];
  $('#cmpOrig').src = thumbs.gray;
  $('#cmpHe').src = thumbs.he;
  setCompare(50);

  $('#condSize').textContent = `${m0.width} × ${m0.height}`;

  const rows = [
    ['Mean intensity μ', `${fmt(m0.mean)} → <span class="up">${fmt(m1.mean)}</span>`],
    ['Contrast σ', `${fmt(m0.contrast_std)} → <span class="up">${fmt(m1.contrast_std)}</span> <small>(${gain.toFixed(2)}×)</small>`],
    ['Entropy H', `${fmt(m0.entropy, 2)} → ${fmt(m1.entropy, 2)} bits`],
    ['Michelson contrast', `${fmt(m0.michelson, 2)} → ${fmt(m1.michelson, 2)}`],
    ['Dynamic range', `${m0.dynamic_range} → ${m1.dynamic_range}`],
    ['Grey levels used', `${m0.levels_used} → ${m1.levels_used}`],
    ['SNR μ/σ', `${fmt(m0.snr_db, 2)} → ${fmt(m1.snr_db, 2)} dB`],
    ['Noise σₙ', `${fmt(m0.noise_sigma, 2)} → ${fmt(m1.noise_sigma, 2)} gl`],
    ['SNR (noise)', `${fmt(m0.snr_noise_db)} → ${fmt(m1.snr_noise_db)} dB`],
    ['Sharpness', `${fmt(m0.sharpness)} → <span class="up">${fmt(m1.sharpness)}</span>`],
    ['MSE / PSNR', `${fmt(m1.mse, 0)} / ${fmt(m1.psnr_db)} dB`],
    ['AMBE', `${fmt(m1.ambe)} gl`],
  ];
  const list = $('#condStats');
  list.textContent = '';
  rows.forEach(([k, v]) => {
    const row = document.createElement('div');
    row.className = 'stat-row';
    const key = document.createElement('span');
    key.className = 'k';
    key.textContent = k;
    const val = document.createElement('span');
    val.className = 'v';
    val.innerHTML = v;
    row.append(key, val);
    list.appendChild(row);
  });
}

function tableHtml(headers, rows, cls = '') {
  const thead = headers.map((h) => `<th>${h}</th>`).join('');
  const body = rows.map((r) => `<tr>${r.map((cell) => {
    if (cell && typeof cell === 'object') {
      return `<td class="${cell.cls || ''}">${cell.html}</td>`;
    }
    return `<td>${cell ?? '—'}</td>`;
  }).join('')}</tr>`).join('');
  return `<table class="metric ${cls}"><thead><tr>${thead}</tr></thead><tbody>${body}</tbody></table>`;
}

function renderTables(data) {
  const conds = data.conditions;
  const rs = data.results;
  const M = (c) => rs[c].metrics;

  $('#table6').innerHTML = tableHtml(
    ['Condition', 'Size', 'Mean (orig)', 'Mean (HE)', 'σ (orig)', 'σ (HE)', 'H (orig)', 'H (HE)', 'σ gain'],
    conds.map((c) => {
      const m0 = M(c).original, m1 = M(c).he;
      const g = m0.contrast_std > 0 ? m1.contrast_std / m0.contrast_std : 1;
      return [
        { html: COND_LABEL[c], cls: 'cond' },
        `${m0.width} × ${m0.height}`,
        fmt(m0.mean), fmt(m1.mean),
        fmt(m0.contrast_std), fmt(m1.contrast_std),
        fmt(m0.entropy, 2), fmt(m1.entropy, 2),
        `${g.toFixed(2)}×`,
      ];
    })
  ) + `<div class="table-note">Global HE pulls every mean close to mid-grey (≈128) and raises σ by ${Math.min(...conds.map((c) => (M(c).he.contrast_std / M(c).original.contrast_std))).toFixed(2)}× to ${Math.max(...conds.map((c) => (M(c).he.contrast_std / M(c).original.contrast_std))).toFixed(2)}×.</div>`;

  $('#table7').innerHTML = tableHtml(
    ['Condition', 'SNR μ/σ (orig)', 'SNR μ/σ (HE)', 'σₙ (orig)', 'σₙ (HE)', 'SNR noise (orig)', 'SNR noise (HE)'],
    conds.map((c) => {
      const m0 = M(c).original, m1 = M(c).he;
      return [
        { html: COND_LABEL[c], cls: 'cond' },
        `${fmt(m0.snr_db, 2)} dB`, `${fmt(m1.snr_db, 2)} dB`,
        fmt(m0.noise_sigma, 2), fmt(m1.noise_sigma, 2),
        `${fmt(m0.snr_noise_db)} dB`, `${fmt(m1.snr_noise_db)} dB`,
      ];
    })
  );

  $('#table8').innerHTML = tableHtml(
    ['Condition', 'Dynamic range', 'Michelson', 'Sharpness', 'MSE', 'PSNR', 'AMBE'],
    conds.map((c) => {
      const m0 = M(c).original, m1 = M(c).he;
      return [
        { html: COND_LABEL[c], cls: 'cond' },
        `${m0.dynamic_range} → ${m1.dynamic_range}`,
        `${fmt(m0.michelson, 2)} → ${fmt(m1.michelson, 2)}`,
        `${fmt(m0.sharpness)} → ${fmt(m1.sharpness)}`,
        fmt(m1.mse, 0), `${fmt(m1.psnr_db)} dB`, `${fmt(m1.ambe)} gl`,
      ];
    })
  ) + `<div class="table-note">For contrast enhancement a large difference from the original is intended, so a low PSNR is not a quality loss.</div>`;

  const tech = (m) => `${fmt(m.contrast_std)} / ${fmt(m.entropy, 2)} / ${fmt(m.snr_noise_db)}`;
  $('#table9').innerHTML = tableHtml(
    ['Condition', 'Original', 'Global HE', 'CLAHE', 'Contrast stretch'],
    conds.map((c) => {
      const m = M(c);
      return [
        { html: COND_LABEL[c], cls: 'cond' },
        tech(m.original), tech(m.he), tech(m.clahe), tech(m.stretch),
      ];
    })
  ) + `<div class="table-note">Cells are contrast σ (grey levels) / entropy H (bits) / noise-based SNR (dB).</div>`;

  $('#table10').innerHTML = tableHtml(
    ['Condition', 'HE vs OpenCV', 'Levels (orig)', 'Levels (HE)', 'Pixels ≤ 40', 'Pixels ≥ 250', 'Pipeline time'],
    conds.map((c) => {
      const e = rs[c];
      const diffCls = e.max_diff_cv === 0 ? 'good' : 'warn';
      const darkCls = e.tails.dark_frac > 0.2 ? 'warn' : '';
      const brightCls = e.tails.bright_frac > 0.05 ? 'warn' : '';
      return [
        { html: COND_LABEL[c], cls: 'cond' },
        { html: `${e.max_diff_cv} gl`, cls: diffCls },
        e.levels.original, e.levels.he,
        { html: pct(e.tails.dark_frac), cls: darkCls },
        { html: pct(e.tails.bright_frac), cls: brightCls },
        `${fmt(e.elapsed_ms, 1)} ms`,
      ];
    })
  ) + `<div class="table-note">A maximum difference of 0 grey levels confirms the user-defined mapping is identical to OpenCV's implementation.</div>`;
}

function renderTechniques(data) {
  // Figures are embedded in the analysis response; only wire them up when
  // the Techniques tab is actually opened.
  if ($('#pane-techniques').classList.contains('active')) ensureTechniques(data);
}

function ensureTechniques(data) {
  const urls = data.figure_urls;
  if (!urls) return;
  // Figures are embedded in the analysis response so they survive a serverless
  // instance change; the /api/figure URLs remain as a local fallback.
  const fig = (key, fallback) => (data.figures && data.figures[key]) || fallback;
  const set = (selector, url, caption) => {
    const img = $(selector);
    if (img.dataset.url === url && img.getAttribute('src')) return;
    img.dataset.url = url;
    img.dataset.caption = caption;
    img.src = url;
  };
  const caption = (sel, fallback) => ($(sel)?.textContent || fallback);
  set('#overviewImage', fig('overview:', urls.overview),
    caption('#overviewCaption', 'Original vs global HE vs CLAHE vs contrast stretching'));
  set('#techniquesChart', fig('techniques:', urls.techniques),
    caption('#techniquesCaption', 'Contrast, entropy and noise-based SNR by technique'));
  set('#metricsChart', fig('metrics:', urls.metrics),
    caption('#metricsCaption', 'Mean, contrast, entropy and noise-based SNR before and after HE'));
}

/* ------------------------------ compare slider ------------------------------ */
function bindCompare() {
  const range = $('#cmpRange');
  range.addEventListener('input', () => setCompare(Number(range.value)));
}

function setCompare(pos) {
  $('#cmpRange').value = String(pos);
  $('#compare').style.setProperty('--pos', `${pos}%`);
}

/* ------------------------------ downloads ------------------------------ */
function bindDownloads() {
  $('#csvBtn').addEventListener('click', () => {
    if (!state.result) return;
    const conds = state.result.conditions;
    const header = ['condition', 'variant', 'width', 'height', 'mean', 'contrast_std',
      'michelson', 'dynamic_range', 'levels_used', 'entropy', 'snr_db', 'sharpness',
      'noise_sigma', 'snr_noise_db', 'mse', 'psnr_db', 'ambe'];
    const lines = [header.join(',')];
    conds.forEach((c) => {
      ['original', 'he', 'clahe', 'stretch'].forEach((variant) => {
        const m = state.result.results[c].metrics[variant];
        const row = [c, variant, m.width, m.height, m.mean, m.contrast_std, m.michelson,
          m.dynamic_range, m.levels_used, m.entropy, m.snr_db, m.sharpness,
          m.noise_sigma, m.snr_noise_db, m.mse ?? '', m.psnr_db ?? '', m.ambe ?? ''];
        lines.push(row.map((v) => (typeof v === 'number' ? v.toFixed(4) : v)).join(','));
      });
    });
    downloadBlob('dip-metrics.csv', lines.join('\r\n'), 'text/csv;charset=utf-8');
    toast('Metrics CSV downloaded.');
  });

  $('#jsonBtn').addEventListener('click', () => {
    if (!state.result) return;
    const { thumbs, figure_urls, figures, ...rest } = state.result;
    downloadBlob('dip-results.json', JSON.stringify(rest, null, 2), 'application/json');
    toast('Results JSON downloaded.');
  });
}

function downloadBlob(name, text, type) {
  const blob = new Blob([text], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1500);
}

/* ------------------------------ lightbox ------------------------------ */
function bindLightbox() {
  const box = $('#lightbox');
  document.addEventListener('click', (e) => {
    const img = e.target.closest('img.zoomable');
    if (!img || !img.src) return;
    $('#lbImage').src = img.src;
    $('#lbCaption').textContent = img.dataset.caption || img.alt || 'Figure';
    const link = $('#lbDownload');
    link.href = img.src;
    link.download = `${(img.dataset.caption || 'figure').toLowerCase().replace(/[^a-z0-9]+/g, '-').slice(0, 60)}.jpg`;
    box.classList.add('open');
    box.setAttribute('aria-hidden', 'false');
    document.body.style.overflow = 'hidden';
  });
  const close = () => {
    box.classList.remove('open');
    box.setAttribute('aria-hidden', 'true');
    document.body.style.overflow = '';
  };
  $('#lbClose').addEventListener('click', close);
  box.addEventListener('click', (e) => { if (e.target === box) close(); });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && box.classList.contains('open')) close();
  });
}

/* ------------------------------ account ------------------------------ */
function bindAccount() {
  const btn = $('#logoutBtn');
  if (btn) {
    btn.addEventListener('click', async () => {
      try { await fetch('/api/logout', { method: 'POST' }); } catch (_) {}
      location.href = '/';
    });
  }
  fetch('/api/me').then((r) => r.json()).then((data) => {
    if (data.ok && data.username) $('#userChip').textContent = data.username;
  }).catch(() => {});
}

