// Scissortail — single-window controller: drop trial files in, look at the
// automatic speech segment and landmarks of each, drag to correct, export.
'use strict';

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const num = (v, d = 3) => (v == null || !isFinite(v) ? '—' : Number(v).toFixed(d));

// Key names as the platform writes them: ⇧⌘O and ⌥ on the Mac, Ctrl+Shift+O and Alt on Windows.
const IS_MAC = /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent);
const ALT_KEY = IS_MAC ? '⌥' : 'Alt';
const shortcut = (key, shift = false) => (IS_MAC ? `${shift ? '⇧' : ''}⌘${key}` : `Ctrl+${shift ? 'Shift+' : ''}${key}`);

const DIMS = [
  { id: 'z', label: 'z', title: 'vertical (up +)' },
  { id: 'x', label: 'x', title: 'anterior–posterior (front +)' },
  { id: 'y', label: 'y', title: 'lateral' },
  { id: '3d', label: '3D', title: '3-D: path length, speed, |acceleration|' },
];
const ROWS = [
  { id: 'pos', name: 'Position', short: 'Pos', tag: 'P' },
  { id: 'vel', name: 'Velocity', short: 'Vel', tag: 'V' },
  { id: 'acc', name: 'Acceleration', short: 'Acc', tag: 'A' },
];

function loadPrefs() {
  try {
    const p = JSON.parse(localStorage.getItem('scissortail.prefs') || 'null');
    if (p && DIMS.some((d) => d.id === p.dim)) return p;
  } catch (_) { /* storage unavailable */ }
  return null;
}

const prefs = loadPrefs() || {};
const S = {
  info: null,
  files: [],
  byFile: new Map(),
  current: null,
  view: null,
  result: null,
  seg: 0,
  config: null,
  filter: '',
  statusFilter: null,
  dim: prefs.dim || 'z',
  rows: new Set(prefs.rows || ['pos', 'vel', 'acc']),
  spec: prefs.spec ?? true,
  zoomMode: prefs.zoomMode || 'fit',
  progress: { pending: 0, total: 0 },
  cache: new Map(),
  token: 0,
  selection: null,
  loaded: false,
};

function savePrefs() {
  try {
    localStorage.setItem('scissortail.prefs', JSON.stringify({ dim: S.dim, rows: [...S.rows], spec: S.spec, zoomMode: S.zoomMode }));
  } catch (_) { /* storage unavailable */ }
}

let viewer = null;

// ---------------------------------------------------------------- feedback
function toast(html, kind = 'info', ms = 5000) {
  const el = document.createElement('div');
  el.className = `toast ${kind}`;
  el.innerHTML = html;
  $('#toasts').appendChild(el);
  $$('[data-reveal]', el).forEach((a) => a.addEventListener('click', () => API.invoke('reveal', { path: a.dataset.reveal })));
  setTimeout(() => el.remove(), kind === 'bad' ? 9000 : ms);
}

function errMsg(e) {
  return typeof e === 'string' ? e : e?.message || JSON.stringify(e);
}

let closeModal = null;
function modal({ title, body, foot = '', narrow = false, onMount }) {
  const root = $('#modalRoot');
  root.innerHTML = `<div class="modal-backdrop"><div class="modal ${narrow ? 'narrow' : ''}"><div class="modal-head">${title}<button class="btn x" data-close>×</button></div><div class="modal-body">${body}</div>${foot ? `<div class="modal-foot">${foot}</div>` : ''}</div></div>`;
  const close = () => { root.innerHTML = ''; closeModal = null; };
  $$('[data-close]', root).forEach((b) => b.addEventListener('click', close));
  $('.modal-backdrop', root).addEventListener('mousedown', (e) => { if (e.target.classList.contains('modal-backdrop')) close(); });
  closeModal = close;
  onMount?.(root, close);
  return close;
}

function confirmDialog(title, text, okLabel) {
  return new Promise((resolve) => {
    let answered = false;
    modal({
      title,
      narrow: true,
      body: `<div>${text}</div>`,
      foot: `<button class="btn" data-close>Cancel</button><button class="btn primary" id="cfOk">${okLabel}</button>`,
      onMount: (root, close) => {
        $('#cfOk', root).addEventListener('click', () => { answered = true; close(); resolve(true); });
        const obs = new MutationObserver(() => { if (!root.innerHTML) { obs.disconnect(); if (!answered) resolve(false); } });
        obs.observe(root, { childList: true });
      },
    });
  });
}

// ----------------------------------------------------------------- toolbar
function renderToolbar() {
  $('#btnOpenFiles').textContent = 'Open files…';
  $('#btnOpenFiles').title = `Open .mat trial files (${shortcut('O')})`;
  $('#btnOpenFolder').textContent = 'Folder…';
  $('#btnOpenFolder').title = `Open every .mat file in a folder and its sub-folders (${shortcut('O', true)})`;
  $('#btnPrev').textContent = '◀';
  $('#btnPrev').title = `Previous file (←)`;
  $('#btnNext').textContent = '▶';
  $('#btnNext').title = `Next file (→)`;
  $('#dimSel').innerHTML = DIMS.map((d) => `<button data-dim="${d.id}" class="opt ${S.dim === d.id ? 'on' : ''}" title="${esc(d.title)}"><span class="rb"></span>${d.label}</button>`).join('');
  $('#rowSel').innerHTML = ROWS.map((r) => `<button data-row="${r.id}" class="opt ${S.rows.has(r.id) ? 'on' : ''}" title="${esc(r.name)} (${r.tag})"><span class="cb"></span>${esc(r.short)}</button>`).join('');
  $('#btnSpec').innerHTML = `<span class="cb"></span>Spectrogram`;
  $('#btnSpec').classList.toggle('on', S.spec);
  $('#btnExport').textContent = 'Export CSV…';
  $('#btnExport').title = `Landmarks of all files in the list, one CSV (${shortcut('E')})`;
  $('#btnSettings').textContent = 'Settings…';
  $('#btnSettings').title = 'Settings';
  $('#btnClear').textContent = 'Clear list';
  $('#filter').placeholder = 'Filter files…';
  renderTitle();
}

function renderTitle() {
  const list = visibleFiles();
  const i = list.findIndex((f) => f.file === S.current);
  const sum = S.byFile.get(S.current);
  $('#fileName').textContent = sum ? sum.name : '—';
  $('#fileName').title = S.current || '';
  $('#filePos').textContent = sum ? `${sum.folder}  ·  ${i >= 0 ? i + 1 : '–'} / ${list.length}` : '';
  $('#btnPrev').disabled = i <= 0;
  $('#btnNext').disabled = i < 0 || i >= list.length - 1;
  const rv = $('#btnReviewed');
  const reviewed = !!sum?.reviewed;
  rv.disabled = !S.result;
  rv.classList.toggle('on', reviewed);
  rv.innerHTML = `<span class="cb"></span>Checked`;
  rv.title = `Mark this file as checked and go to the next one (↩)`;
  $('#btnExport').disabled = !S.files.length;
}

// --------------------------------------------------------------- file list
// The four states of the status dot (and of the legend under the file list).
const STATUS_GROUPS = {
  ok: ['ok'],
  check: ['check'],
  bad: ['not_found', 'error'],
  pending: ['pending'],
};

function visibleFiles() {
  let list = S.files;
  if (S.statusFilter) list = list.filter((f) => STATUS_GROUPS[S.statusFilter].includes(f.status));
  const q = S.filter.trim().toLowerCase();
  if (!q) return list;
  if (q === '!') return list.filter((f) => f.status === 'check' || f.status === 'not_found' || f.status === 'error');
  return list.filter((f) => `${f.name} ${f.folder} ${f.word || ''}`.toLowerCase().includes(q));
}

function statusTitle(f) {
  return {
    pending: 'Not analysed yet',
    ok: 'Segment found',
    check: 'Please check (low confidence)',
    not_found: 'No speech found',
    error: 'Cannot read the file',
    excluded: 'Excluded',
  }[f.status] + (f.message ? ` — ${f.message}` : '');
}

function rowHtml(f) {
  return `<div class="frow ${f.file === S.current ? 'sel' : ''}" data-file="${esc(f.file)}" title="${esc(statusTitle(f))}">
    <span class="dot ${f.status}"></span><span class="name">${esc(f.name)}</span>
    <span class="word">${esc(f.word || '')}${f.segments > 1 ? ` ×${f.segments}` : ''}</span>
    <span class="flags">${f.edited ? '<span title="Edited by hand">✎</span>' : ''}${f.reviewed ? '<span title="Checked">✓</span>' : ''}</span>
  </div>`;
}

function renderFileList() {
  const host = $('#fileList');
  const list = visibleFiles();
  if (!S.files.length) {
    host.innerHTML = '';
  } else if (!list.length) {
    host.innerHTML = `<div class="side-empty">No file matches the filter.</div>`;
  } else {
    const counts = new Map();
    for (const f of list) counts.set(f.folder, (counts.get(f.folder) || 0) + 1);
    let html = '';
    let folder = null;
    for (const f of list) {
      if (f.folder !== folder) {
        folder = f.folder;
        html += `<div class="folder" title="${esc(f.file.replace(/[\\/][^\\/]*$/, ''))}"><span class="fname">${esc(folder)}</span><span class="fcount">${counts.get(folder)}</span></div>`;
      }
      html += rowHtml(f);
    }
    host.innerHTML = html;
  }
  renderCounts();
  renderTitle();
}

/** Legend with the number of files in each state, and the file count. */
function renderCounts() {
  const n = S.files.length;
  const legend = [
    ['ok', 'extracted'],
    ['check', 'check'],
    ['bad', 'not found'],
    ['pending', 'not yet'],
  ];
  $('#legend').innerHTML = n
    ? legend.map(([k, label]) => {
      const count = S.files.filter((f) => STATUS_GROUPS[k].includes(f.status)).length;
      return `<button class="lg ${S.statusFilter === k ? 'on' : ''}" data-st="${k}" title="Show only these files (click again to show all)"><span class="dot ${k}"></span>${esc(label)}<b>${count}</b></button>`;
    }).join('')
    : '';
  const done = S.files.filter((f) => f.reviewed).length;
  $('#fileCount').textContent = n ? `${n} files${done ? ` · ${done} ✓` : ''}` : '';
  $('#btnClear').disabled = !n;
}

function updateSummary(sum) {
  if (!sum) return;
  const i = S.files.findIndex((f) => f.file === sum.file);
  if (i < 0) return;
  S.files[i] = sum;
  S.byFile.set(sum.file, sum);
  const el = $(`#fileList .frow[data-file="${CSS.escape(sum.file)}"]`);
  if (el) el.outerHTML = rowHtml(sum);
  renderCounts();
  if (sum.file === S.current) renderTitle();
}

function setWorkspace(ws) {
  S.loaded = true;
  S.files = ws.files;
  S.byFile = new Map(ws.files.map((f) => [f.file, f]));
  S.config = ws.config;
  renderFileList();
  renderEmpty();
}

function scrollToCurrent() {
  const el = $(`#fileList .frow[data-file="${CSS.escape(S.current || '')}"]`);
  el?.scrollIntoView({ block: 'nearest' });
}

// ----------------------------------------------------------- opening files
function cachePut(file, view) {
  S.cache.delete(file);
  S.cache.set(file, view);
  while (S.cache.size > 6) S.cache.delete(S.cache.keys().next().value);
}

function setBusy(text) {
  $('#viewerBusy').classList.toggle('hidden', !text);
  $('#viewerBusyText').textContent = text || '';
}

function renderEmpty() {
  const empty = S.loaded && !S.files.length;
  $('#empty').classList.toggle('hidden', !empty);
  if (empty) {
    $('#empty').innerHTML = `<button class="btn primary" id="emFolder">Open folder…</button><button class="btn" id="emOpen">Open files…</button>`;
    $('#emFolder').addEventListener('click', openFolderDialog);
    $('#emOpen').addEventListener('click', openFilesDialog);
    S.current = null;
    S.view = null;
    S.result = null;
    renderSegBar();
    renderTable();
  }
}

function applyZoomMode() {
  if (!viewer.view) return;
  if (S.zoomMode === 'segment' && S.result?.occurrences?.length) viewer.zoomSegment();
  else viewer.zoomFit();
}

function toggleZoomMode() {
  S.zoomMode = S.zoomMode === 'segment' ? 'fit' : 'segment';
  applyZoomMode();
  savePrefs();
}

async function openFile(file) {
  if (!file) return;
  const token = ++S.token;
  S.current = file;
  hideSelBar();
  $$('#fileList .frow.sel').forEach((el) => el.classList.remove('sel'));
  $(`#fileList .frow[data-file="${CSS.escape(file)}"]`)?.classList.add('sel');
  scrollToCurrent();
  renderTitle();
  try { localStorage.setItem('scissortail.last', file); } catch (_) { /* storage unavailable */ }
  let view = S.cache.get(file);
  if (!view) {
    setBusy('Loading…');
    try {
      view = await API.invoke('load_trial', { file });
    } catch (e) {
      if (token !== S.token) return;
      setBusy(null);
      S.view = null;
      S.result = null;
      viewer.setTrial({ duration: 1, sensors: [] });
      viewer.setResult(null);
      renderSegBar(errMsg(e));
      renderTable();
      return;
    }
    cachePut(file, view);
  }
  if (token !== S.token) return;
  S.view = view;
  S.result = view.result || null;
  S.seg = 0;
  viewer.setTrial(view);
  viewer.setResult(S.result, 0);
  applyZoomMode();
  renderSegBar();
  renderTable();
  renderTitle();
  if (!S.result) {
    setBusy('Finding the speech segment…');
    try {
      const r = await API.invoke('analyze_trial', { file });
      applyResult(file, r);
      if (token === S.token) applyZoomMode();
    } catch (e) {
      if (token === S.token) toast(esc(errMsg(e)), 'bad');
    }
  }
  if (token !== S.token) return;
  setBusy(null);
  prefetch();
}

function prefetch() {
  const list = visibleFiles();
  const i = list.findIndex((f) => f.file === S.current);
  const next = list[i + 1];
  if (!next || S.cache.has(next.file)) return;
  const file = next.file;
  setTimeout(() => {
    if (S.cache.has(file)) return;
    API.invoke('load_trial', { file }).then((v) => cachePut(file, v)).catch(() => {});
  }, 150);
}

/** Takes an edit / analysis answer ({result, summary}) from the backend. */
function applyResult(file, r) {
  if (!r) return;
  updateSummary(r.summary);
  const cached = S.cache.get(file);
  if (cached) cached.result = r.result;
  if (file !== S.current) return;
  S.result = r.result;
  const n = S.result?.occurrences?.length || 0;
  S.seg = Math.max(0, Math.min(S.seg, n - 1));
  viewer.setResult(S.result, S.seg);
  renderSegBar();
  renderTable();
  renderTitle();
}

function step(delta) {
  const list = visibleFiles();
  if (!list.length) return;
  const i = list.findIndex((f) => f.file === S.current);
  const j = i < 0 ? 0 : Math.max(0, Math.min(list.length - 1, i + delta));
  if (list[j] && list[j].file !== S.current) openFile(list[j].file);
}

async function addPaths(paths) {
  if (!paths?.length) return;
  try {
    const r = await API.invoke('add_paths', { paths });
    setWorkspace(r.workspace);
    const msg = r.added ? `${r.added} files added` : 'These files are already in the list';
    toast(esc(msg));
    if (r.first) openFile(r.first);
  } catch (e) {
    toast(esc(errMsg(e)), 'bad');
  }
}

async function openFilesDialog() {
  const paths = await API.invoke('pick_files').catch((e) => { toast(esc(errMsg(e)), 'bad'); return []; });
  addPaths(paths);
}

async function openFolderDialog() {
  const p = await API.invoke('pick_folder').catch((e) => { toast(esc(errMsg(e)), 'bad'); return null; });
  if (p) addPaths([p]);
}

// ------------------------------------------------------ segment + landmarks
function occ() {
  return S.result?.occurrences?.[S.seg] || null;
}

function methodLabel(o) {
  if (o.edited && o.occurrence.engine === 'speech' && o.occurrence.confidence === 1) return 'marked by hand';
  return { phonetic: 'speech alignment', template: 'TTS template', speech: 'speech activity' }[o.occurrence.engine] || o.occurrence.engine;
}

function renderSegBar(error) {
  const bar = $('#segBar');
  if (!S.current) { bar.innerHTML = ''; return; }
  if (error) { bar.innerHTML = `<span class="note bad">${icon('warning')} ${esc(error)}</span>`; return; }
  const r = S.result;
  if (!r) { bar.innerHTML = `<span class="muted">Finding the speech segment…</span>`; return; }
  const o = occ();
  const occs = r.occurrences || [];
  let html = '';
  if (!o) {
    html = `<span class="note ${r.status === 'error' ? 'bad' : ''}">${icon('warning')} ${esc(r.message || 'No speech segment.')}</span><span class="muted">Shift-drag over the signals to mark one by hand.</span>`;
  } else {
    const oc = o.occurrence;
    html += `<span class="seg-word">${esc(oc.word)}</span>`;
    if (occs.length > 1) html += `<span class="chips">${occs.map((_, i) => `<button class="btn small toggle ${i === S.seg ? 'on' : ''}" data-seg="${i}">#${i + 1}</button>`).join('')}</span>`;
    html += `<span class="kv">onset <b>${num(oc.start_s, 3)}</b> s</span>`;
    html += `<span class="kv">offset <b>${num(oc.end_s, 3)}</b> s</span>`;
    html += `<span class="kv">duration <b>${num((oc.end_s - oc.start_s) * 1000, 0)}</b> ms</span>`;
    html += `<span class="kv muted">${methodLabel(o)}${oc.engine !== 'speech' ? ` · confidence ${num(oc.confidence, 2)}` : ''}${o.edited ? ` · ✎ edited` : ''}</span>`;
    if (r.message) html += `<span class="note">${icon('warning')} ${esc(r.message)}</span>`;
  }
  html += `<span class="right">`;
  if (o) html += `<button class="btn small" id="sbPlay" title="Play the segment (␣)">▶ Play</button>`;
  html += `<button class="btn small" id="sbZoom" title="Zoom to the segment / whole file (Z)">${S.zoomMode === 'segment' ? 'Whole file' : 'Zoom to segment'}</button>`;
  if (o && occs.length > 1) html += `<button class="btn small" id="sbDel" title="Delete this segment">Delete segment</button>`;
  html += `<button class="btn small" id="sbReset" title="Discard the edits of this file and detect again">Re-detect</button>`;
  html += `</span>`;
  bar.innerHTML = html;
  $$('[data-seg]', bar).forEach((b) => b.addEventListener('click', () => selectSegment(+b.dataset.seg)));
  $('#sbPlay')?.addEventListener('click', togglePlay);
  $('#sbZoom')?.addEventListener('click', () => { toggleZoomMode(); renderSegBar(); });
  $('#sbDel')?.addEventListener('click', deleteSegment);
  $('#sbReset')?.addEventListener('click', reanalyze);
}

function selectSegment(i) {
  S.seg = i;
  viewer.setResult(S.result, i);
  if (S.zoomMode === 'segment') viewer.zoomSegment();
  renderSegBar();
  renderTable();
}

// Measures of the protocols: short column names and units.
const MEASURE_NAMES = {
  JVC_Dur_ms: 'JVC duration', MaxJ_Disp_mm: 'Max jaw displacement', TT_Lat_ms: 'TT latency', TT_Pha_deg: 'TT phase angle',
  PV1_mm_s: 'Peak lowering vel.', PV2_mm_s: 'Peak raising vel.', Jaw_Amp_mm: 'Jaw opening',
};
const GESTURE_MEASURES = [
  ['dur_ms', 'Duration'], ['form_ms', 'Formation'], ['plateau_ms', 'Plateau'], ['release_ms', 'Release'], ['ttpv_ms', 'Time to peak vel.'],
  ['disp_mm', 'Displacement'], ['pvel_mm_s', 'Peak velocity'], ['stiff_1_s', 'Stiffness'], ['onset_lag_ms', 'Onset lag'], ['target_lag_ms', 'Target lag'], ['maxc_lag_ms', 'Constriction lag'],
];
const unitOf = (m) => (m.unit === '°' ? '°' : m.unit);
const digitsOf = (m) => (m.unit === 'mm' ? 2 : m.unit === '1/s' ? 1 : 1);

function renderProtocolTable(host, o) {
  const p = o.protocol;
  const on = o.occurrence.start_s;
  let html = '';
  if (p.kind === 'gestures') {
    const labels = [...new Set(p.measures.map((m) => m.id.slice(0, m.id.length - (GESTURE_MEASURES.find(([suf]) => m.id.endsWith('_' + suf))?.[0].length + 1 || 0))))];
    const cols = GESTURE_MEASURES.filter(([suf]) => p.measures.some((m) => m.id.endsWith('_' + suf)));
    html += `<table class="lm"><thead><tr><th>Gesture</th>${cols.map(([suf, name]) => {
      const m = p.measures.find((x) => x.id.endsWith('_' + suf));
      return `<th title="${esc(m.label)}">${esc(name)} <span class="unit">${esc(unitOf(m))}</span></th>`;
    }).join('')}</tr></thead><tbody>`;
    for (const g of labels) {
      html += `<tr><td class="sensor">${esc(g)}</td>${cols.map(([suf]) => {
        const m = p.measures.find((x) => x.id === `${g}_${suf}`);
        return `<td>${m ? num(m.value, digitsOf(m)) : ''}</td>`;
      }).join('')}</tr>`;
    }
    html += `</tbody></table>`;
  } else {
    html += `<table class="lm"><thead><tr><th>Measures</th>${p.measures.map((m) => `<th title="${esc(m.label)}">${esc(MEASURE_NAMES[m.id] || m.id)} <span class="unit">${esc(unitOf(m))}</span></th>`).join('')}</tr></thead>`;
    html += `<tbody><tr><td class="sensor">${esc(o.occurrence.word)}</td>${p.measures.map((m) => `<td title="${esc(m.label)}">${num(m.value, digitsOf(m))}</td>`).join('')}</tr></tbody></table>`;
  }
  if (p.landmarks.length) {
    html += `<table class="lm"><thead><tr><th>Landmark</th><th>Sensor</th><th>time <span class="unit">ms re onset</span></th><th>position <span class="unit">mm</span></th><th>velocity <span class="unit">mm/s</span></th><th class="left">found by</th></tr></thead><tbody>`;
    for (const l of p.landmarks) {
      const rel = (l.time_s - on) * 1000;
      const cls = l.manual ? 'manual' : '';
      html += `<tr><td class="sensor" title="${esc(l.id)}">${esc(l.label)}</td><td>${esc(l.reference ? `${l.sensor}−${l.reference}` : l.sensor)}</td>`
        + `<td class="${cls}" title="${num(l.time_s, 4)} s">${rel >= 0 ? '+' : ''}${rel.toFixed(1)}</td><td>${num(l.position, 2)}</td><td>${num(l.velocity, 1)}</td><td class="left">${esc(l.criterion)}</td></tr>`;
    }
    html += `</tbody></table>`;
  }
  const notes = [...(p.warnings || []), ...(S.result?.review || [])];
  if (notes.length) html += `<div class="table-empty">${icon('warning')} ${esc(notes.join(' · '))}</div>`;
  host.innerHTML = html;
}

function renderTable() {
  const host = $('#tableHost');
  const o = occ();
  if (!o) { host.innerHTML = ''; host.classList.add('hidden'); return; }
  host.classList.remove('hidden');
  if (o.protocol) { renderProtocolTable(host, o); return; }
  const dk = dimKey(S.dim);
  const shown = new Set((S.view?.sensors || []).filter((s) => !s.is_reference).map((s) => s.name));
  const kins = o.kinematics.filter((k) => k.dim === dk && shown.has(k.sensor));
  if (!kins.length) {
    host.innerHTML = `<div class="table-empty">No landmarks for this dimension — enable it in Settings.</div>`;
    return;
  }
  const on = o.occurrence.start_s;
  const units = { position: 'mm', velocity: 'mm/s', acceleration: 'mm/s²' };
  const cell = (p, key) => {
    if (!p) return '<td>—</td><td>—</td>';
    const cls = `${p.manual ? 'manual' : ''} ${p.at_edge ? 'edge' : ''}`;
    const rel = (p.time_s - on) * 1000;
    const tip = `${num(p.time_s, 4)} s${p.at_edge ? ' — at the window edge: the true peak may lie outside the segment' : ''}${p.manual ? ' — placed by hand' : ''}`;
    return `<td class="${cls}" title="${esc(tip)}">${rel >= 0 ? '+' : ''}${rel.toFixed(1)}</td><td class="${cls}" title="${esc(tip)}">${num(p.value, key === 'position' ? 2 : 1)}</td>`;
  };
  let html = `<table class="lm"><thead><tr><th>Sensor · ${S.dim === '3d' ? '3D' : S.dim}</th>`;
  for (const [key, name, tag] of [['position', 'Position', 'P'], ['velocity', 'Velocity', 'V'], ['acceleration', 'Acceleration', 'A']]) {
    html += `<th>${tag} · ${name} <span class="unit">ms</span></th><th>value <span class="unit">${units[key]}</span></th>`;
  }
  html += `<th>range <span class="unit">mm</span></th></tr></thead><tbody>`;
  for (const k of kins) {
    html += `<tr><td class="sensor">${esc(k.sensor)}</td>${cell(k.position, 'position')}${cell(k.velocity, 'velocity')}${cell(k.acceleration, 'acceleration')}<td>${num(k.range, 2)}</td></tr>`;
  }
  html += `</tbody></table>`;
  host.innerHTML = html;
}

async function onSegmentChange(index, start, end) {
  const file = S.current;
  try {
    applyResult(file, await API.invoke('update_segment', { file, index, startS: start, endS: end }));
  } catch (e) {
    toast(esc(errMsg(e)), 'bad');
    reloadResult(file);
  }
}

async function onLandmarkChange({ sensor, dim, kind, time_s, snap_s }) {
  const file = S.current;
  try {
    applyResult(file, await API.invoke('set_landmark', { file, index: S.seg, sensor, dim, kind, timeS: time_s, snapS: snap_s }));
  } catch (e) {
    toast(esc(errMsg(e)), 'bad');
    reloadResult(file);
  }
}

async function onLandmarkReset({ sensor, dim, kind }) {
  const file = S.current;
  try {
    applyResult(file, await API.invoke('reset_landmark', { file, index: S.seg, sensor, dim, kind }));
    toast('Landmark back at the automatic position.');
  } catch (e) {
    toast(esc(errMsg(e)), 'bad');
  }
}

async function onProtocolLandmarkChange({ id, time_s, snap_s }) {
  const file = S.current;
  try {
    applyResult(file, await API.invoke('set_protocol_landmark', { file, index: S.seg, id, timeS: time_s, snapS: snap_s }));
  } catch (e) {
    toast(esc(errMsg(e)), 'bad');
    reloadResult(file);
  }
}

async function onProtocolLandmarkReset({ id }) {
  const file = S.current;
  try {
    applyResult(file, await API.invoke('reset_protocol_landmark', { file, index: S.seg, id }));
    toast('Landmark back at the automatic position.');
  } catch (e) {
    toast(esc(errMsg(e)), 'bad');
  }
}

async function reloadResult(file) {
  try { applyResult(file, await API.invoke('analyze_trial', { file })); } catch (_) { /* keep what is shown */ }
}

function showSelBar(a, b) {
  S.selection = [a, b];
  const bar = $('#selBar');
  bar.innerHTML = `<span>Selection <b class="mono">${num(a, 3)}–${num(b, 3)} s</b> (${num((b - a) * 1000, 0)} ms)</span>
    <button class="btn small primary" id="selAdd">Use as segment</button>
    <button class="btn small" id="selPlay" title="Play">▶</button>
    <button class="btn small" id="selZoom">Zoom</button>
    <button class="btn small" id="selClose" title="Cancel">×</button>`;
  bar.classList.remove('hidden');
  $('#selAdd').addEventListener('click', addSegmentFromSelection);
  $('#selPlay').addEventListener('click', () => viewer.play(a, b));
  $('#selZoom').addEventListener('click', () => { viewer.zoomTo(a - 0.05, b + 0.05); hideSelBar(); });
  $('#selClose').addEventListener('click', hideSelBar);
}

function hideSelBar() {
  S.selection = null;
  $('#selBar').classList.add('hidden');
  viewer?.clearSelection();
}

async function addSegmentFromSelection() {
  const [a, b] = S.selection || [];
  if (a == null) return;
  const file = S.current;
  hideSelBar();
  try {
    if (!S.result) await API.invoke('analyze_trial', { file });
    const r = await API.invoke('add_segment', { file, startS: a, endS: b });
    const i = r.result.occurrences.findIndex((o) => Math.abs(o.occurrence.start_s - a) < 1e-6);
    S.seg = Math.max(0, i);
    applyResult(file, r);
  } catch (e) {
    toast(esc(errMsg(e)), 'bad');
  }
}

async function deleteSegment() {
  const file = S.current;
  const ok = await confirmDialog('Delete segment', 'Delete this segment and its landmarks?', 'Delete');
  if (!ok) return;
  try {
    const r = await API.invoke('delete_segment', { file, index: S.seg });
    S.seg = Math.max(0, S.seg - 1);
    applyResult(file, r);
  } catch (e) {
    toast(esc(errMsg(e)), 'bad');
  }
}

async function reanalyze() {
  const file = S.current;
  const edited = S.result?.occurrences?.some((o) => o.edited) || S.result?.reviewed;
  if (edited && !(await confirmDialog('Re-detect', 'Discard the changes made by hand in this file and detect the segment and landmarks again?', 'Re-detect'))) return;
  setBusy('Finding the speech segment…');
  try {
    S.seg = 0;
    applyResult(file, await API.invoke('reanalyze', { file }));
    applyZoomMode();
  } catch (e) {
    toast(esc(errMsg(e)), 'bad');
  }
  setBusy(null);
}

async function setReviewed(value, advance) {
  const file = S.current;
  if (!file || !S.result) return;
  try {
    applyResult(file, await API.invoke('set_reviewed', { file, reviewed: value }));
    if (advance) step(+1);
  } catch (e) {
    toast(esc(errMsg(e)), 'bad');
  }
}

function togglePlay() {
  if (!viewer?.view) return;
  if (viewer.isPlaying()) { viewer.stop(); viewer.render(); return; }
  if (S.selection) return viewer.play(S.selection[0], S.selection[1]);
  const o = occ()?.occurrence;
  if (o) viewer.play(o.start_s, o.end_s);
  else viewer.play(viewer.t0, viewer.t1);
}

// --------------------------------------------------------------- settings
function settingsDialog() {
  const cfg = JSON.parse(JSON.stringify(S.config));
  const speechOnly = cfg.locate.engine === 'speech';
  const wordRow = (w = { label: '', ipa: '', file_pattern: '' }) => `<tr><td><input type="text" data-k="label" value="${esc(w.label)}" placeholder="tadat"></td><td><input type="text" class="ipa" data-k="ipa" value="${esc(w.ipa)}" placeholder="tˈɑdɑt"></td><td><input type="text" class="mono" data-k="file_pattern" value="${esc(w.file_pattern)}" placeholder="ad_"></td><td><button class="btn small" data-del>×</button></td></tr>`;
  const dims = new Set(cfg.kinematics.dims);
  const proto = cfg.protocol || { kind: 'peaks', jaw_cycle: {}, gestures: { gestures: [] } };
  const jc = proto.jaw_cycle;
  const gs = proto.gestures;
  const opt = (v, cur, label) => `<option value="${v}" ${v === cur ? 'selected' : ''}>${label}</option>`;
  const gestureRow = (g = { label: '', sensor: '', reference: '', dim: 'z', direction: 'max', anchor_phone: 0 }) => `<tr>
    <td><input type="text" data-k="label" value="${esc(g.label)}" placeholder="TT" style="width:60px"></td>
    <td><input type="text" data-k="sensor" value="${esc(g.sensor)}" placeholder="TT" style="width:60px"></td>
    <td><input type="text" data-k="reference" value="${esc(g.reference)}" placeholder="—" style="width:60px"></td>
    <td><select data-k="dim">${opt('z', g.dim, 'z')}${opt('x', g.dim, 'x')}${opt('y', g.dim, 'y')}</select></td>
    <td><select data-k="direction">${opt('max', g.direction, 'raising (max)')}${opt('min', g.direction, 'lowering (min)')}</select></td>
    <td><input type="number" data-k="anchor_phone" value="${g.anchor_phone || 0}" min="0" step="1" style="width:48px"></td>
    <td><button class="btn small" data-gdel>×</button></td></tr>`;
  const body = `
    <div class="field"><label>How to find the segment</label>
      <label class="radio"><input type="radio" name="mode" value="words" ${speechOnly ? '' : 'checked'}><span><span class="t">Target word (speech alignment)</span><br><span class="d">Finds the word the participant was asked to say, even inside a carrier phrase, among noise or other speech.</span></span></label>
      <label class="radio"><input type="radio" name="mode" value="speech" ${speechOnly ? 'checked' : ''}><span><span class="t">Whole speech segment</span><br><span class="d">No word needed: marks the main stretch of speech from the sound energy.</span></span></label>
    </div>
    <div id="wordsBox" class="${speechOnly ? 'hidden' : ''}">
      <div class="field"><label>Carrier phrase (IPA, * = target word)</label><input type="text" class="ipa" id="setCarrier" style="width:100%" value="${esc(cfg.locate.carrier)}" placeholder="ɪts ə * əɡˈɛn">
        <div class="help">Leave empty when the word is said on its own.</div></div>
      <div class="field"><label>Target words</label>
        <table class="words"><thead><tr><th>Name</th><th>IPA</th><th>File-name code</th><th></th></tr></thead><tbody id="wordRows">${cfg.locate.words.map(wordRow).join('')}</tbody></table>
        <div class="inline" style="margin-top:6px"><button class="btn small" id="addWord">Add word</button><button class="btn small" id="resetWords">Default: “It’s a ___ again” (tadat / tatat / tedat / tetat)</button></div>
        <div class="help">A file whose name contains the code (e.g. ad_) is matched with that word only; otherwise the best-fitting word is used.</div></div>
    </div>
    <div class="field"><label>Measurement protocol</label>
      <label class="radio"><input type="radio" name="proto" value="peaks" ${proto.kind === 'peaks' ? 'checked' : ''}><span><span class="t">Peaks (P / V / A)</span><br><span class="d">Position, velocity and acceleration peaks of every sensor in the segment.</span></span></label>
      <label class="radio"><input type="radio" name="proto" value="jaw_cycle" ${proto.kind === 'jaw_cycle' ? 'checked' : ''}><span><span class="t">Jaw cycle + tongue-tip onset (Masapollo &amp; Nittrouer 2023)</span><br><span class="d">Six landmarks — V1, PV1, MaxDisp, PV2, TT onset, V2 — and from them the jaw vowel-cycle duration, maximum jaw displacement, TT latency and TT phase angle. For /tVCVt/-type words.</span></span></label>
      <label class="radio"><input type="radio" name="proto" value="gestures" ${proto.kind === 'gestures' ? 'checked' : ''}><span><span class="t">Gestural landmarks (velocity threshold)</span><br><span class="d">Per articulator: onset, peak velocity, target, maximum constriction, release, peak velocity, offset (the MVIEW findgest convention) — durations, displacement, peak velocity, stiffness, and the lags between articulators.</span></span></label>
    </div>
    <div id="jcBox" class="field ${proto.kind === 'jaw_cycle' ? '' : 'hidden'}"><label>Jaw-cycle rules</label>
      <div class="kin-grid">
        <span>Jaw sensor</span><input type="text" id="jcJaw" value="${esc(jc.jaw_sensor || '')}" placeholder="auto (JAW)"><span class="muted"></span>
        <span>Tongue sensor (medial consonant)</span><input type="text" id="jcTT" value="${esc(jc.tongue_sensor || '')}" placeholder="auto (TT)"><span class="muted"></span>
        <span>Velocity criterion</span><input type="number" id="jcVel" min="1" step="1" value="${jc.velocity_threshold ?? 10}"><span class="muted">mm/s (10 = 1 cm/s)</span>
        <span>Gradual onset: zero → criterion longer than</span><input type="number" id="jcGrad" min="0" step="1" value="${jc.gradual_ms ?? 10}"><span class="muted">ms (jaw)</span>
        <span></span><input type="number" id="jcGradTT" min="0" step="1" value="${jc.tongue_gradual_ms ?? 15}"><span class="muted">ms (tongue)</span>
        <span>Displacement criterion for weak lowerings</span><input type="number" id="jcWeak" min="0" step="5" value="${jc.weak_lowering_velocity ?? 0}"><span class="muted">mm/s peak (0 = off; ${jc.displacement_threshold ?? 0.5} mm below closure)</span>
      </div>
      <div class="inline" style="margin-top:8px"><label><input type="checkbox" id="jcRel" ${jc.tongue_minus_jaw !== false ? 'checked' : ''}>Time the tongue relative to the jaw (TT − JAW)</label></div>
      <div class="help">Onsets are velocity zero crossings; when the movement starts gradually, the point where the velocity passes the criterion. Validated against two human coders on 2,269 tokens (10 talkers).</div>
    </div>
    <div id="gsBox" class="field ${proto.kind === 'gestures' ? '' : 'hidden'}"><label>Gestures</label>
      <table class="words"><thead><tr><th>Name</th><th>Sensor</th><th>minus</th><th>Dim</th><th>Constriction</th><th>Phone #</th><th></th></tr></thead><tbody id="gsRows">${(gs.gestures || []).map(gestureRow).join('')}</tbody></table>
      <div class="inline" style="margin-top:6px"><button class="btn small" id="gsAdd">Add gesture</button>
        <span>Threshold</span><input type="number" id="gsThr" min="5" max="50" step="5" value="${Math.round((gs.threshold ?? 0.2) * 100)}" style="width:52px"><span class="muted">% of peak velocity</span></div>
      <div class="help">Phone # picks the constriction nearest that phone of the aligned word (0 = the largest in the word). Lags are measured from the first gesture.</div>
    </div>
    <div class="field"><label>Kinematics</label>
      <div class="kin-grid">
        <span>Low-pass filter (position)</span><input type="number" id="setLp" min="0" step="1" value="${cfg.kinematics.lowpass_hz}"><span class="muted">Hz (0 = off)</span>
        <span>Derivative window (Savitzky–Golay)</span><input type="number" id="setSg" min="10" step="5" value="${cfg.kinematics.sg_window_ms}"><span class="muted">ms</span>
      </div>
      <div class="inline" style="margin-top:8px"><span>Dimensions with landmarks:</span>${[['z', 'z'], ['x', 'x'], ['y', 'y'], ['tangential', '3D']].map(([d, l]) => `<label><input type="checkbox" data-dim="${d}" ${dims.has(d) ? 'checked' : ''}>${l}</label>`).join('')}</div>
    </div>`;
  modal({
    title: 'Settings',
    body,
    foot: `<button class="btn" data-close>Cancel</button><button class="btn primary" id="setSave">Save and re-analyse</button>`,
    onMount: (root, close) => {
      const rows = $('#wordRows', root);
      const bindRows = () => $$('[data-del]', rows).forEach((b) => (b.onclick = () => b.closest('tr').remove()));
      bindRows();
      $$('input[name="mode"]', root).forEach((r) => r.addEventListener('change', () => $('#wordsBox', root).classList.toggle('hidden', r.value === 'speech' && r.checked)));
      $$('input[name="proto"]', root).forEach((r) => r.addEventListener('change', () => {
        const k = $('input[name="proto"]:checked', root).value;
        $('#jcBox', root).classList.toggle('hidden', k !== 'jaw_cycle');
        $('#gsBox', root).classList.toggle('hidden', k !== 'gestures');
      }));
      const gsRows = $('#gsRows', root);
      const bindG = () => $$('[data-gdel]', gsRows).forEach((b) => (b.onclick = () => b.closest('tr').remove()));
      bindG();
      $('#gsAdd', root).addEventListener('click', () => { gsRows.insertAdjacentHTML('beforeend', gestureRow()); bindG(); });
      $('#addWord', root).addEventListener('click', () => { rows.insertAdjacentHTML('beforeend', wordRow()); bindRows(); });
      $('#resetWords', root).addEventListener('click', async () => {
        const def = await API.invoke('default_settings');
        $('#setCarrier', root).value = def.locate.carrier;
        rows.innerHTML = def.locate.words.map(wordRow).join('');
        bindRows();
      });
      $('#setSave', root).addEventListener('click', async () => {
        const mode = $('input[name="mode"]:checked', root).value;
        cfg.locate.engine = mode === 'speech' ? 'speech' : 'phonetic';
        cfg.locate.carrier = $('#setCarrier', root).value.trim();
        cfg.locate.words = $$('tr', rows).map((tr) => ({ label: $('[data-k="label"]', tr).value.trim(), ipa: $('[data-k="ipa"]', tr).value.trim(), file_pattern: $('[data-k="file_pattern"]', tr).value.trim() })).filter((w) => w.ipa);
        if (mode === 'words' && !cfg.locate.words.length) { toast('Enter at least one target word (IPA), or choose “Whole speech segment”.', 'bad'); return; }
        if (mode === 'words' && cfg.locate.carrier && !cfg.locate.carrier.includes('*')) { toast('Mark the place of the target word in the carrier phrase with *.', 'bad'); return; }
        cfg.kinematics.lowpass_hz = Math.max(0, +$('#setLp', root).value || 0);
        cfg.kinematics.sg_window_ms = Math.max(10, +$('#setSg', root).value || 40);
        cfg.kinematics.dims = $$('[data-dim]', root).filter((c) => c.checked).map((c) => c.dataset.dim);
        if (!cfg.kinematics.dims.length) cfg.kinematics.dims = ['z'];
        proto.kind = $('input[name="proto"]:checked', root).value;
        Object.assign(jc, {
          jaw_sensor: $('#jcJaw', root).value.trim(), tongue_sensor: $('#jcTT', root).value.trim(),
          velocity_threshold: Math.max(1, +$('#jcVel', root).value || 10), gradual_ms: Math.max(0, +$('#jcGrad', root).value || 0),
          tongue_gradual_ms: Math.max(0, +$('#jcGradTT', root).value || 0), weak_lowering_velocity: Math.max(0, +$('#jcWeak', root).value || 0),
          tongue_minus_jaw: $('#jcRel', root).checked,
        });
        gs.threshold = Math.min(0.5, Math.max(0.05, (+$('#gsThr', root).value || 20) / 100));
        gs.gestures = $$('tr', gsRows).map((tr) => ({
          label: $('[data-k="label"]', tr).value.trim(), sensor: $('[data-k="sensor"]', tr).value.trim(), reference: $('[data-k="reference"]', tr).value.trim(),
          dim: $('[data-k="dim"]', tr).value, direction: $('[data-k="direction"]', tr).value, anchor_phone: Math.max(0, Math.round(+$('[data-k="anchor_phone"]', tr).value || 0)),
        })).filter((g) => g.sensor);
        if (proto.kind === 'gestures' && !gs.gestures.length) { toast('Add at least one gesture (sensor), or choose another protocol.', 'bad'); return; }
        cfg.protocol = proto;
        close();
        try {
          const ws = await API.invoke('set_config', { config: cfg });
          S.cache.clear();
          setWorkspace(ws);
          toast('Settings saved — files are being analysed again (edited and checked files keep their segments).');
          if (S.current) openFile(S.current);
        } catch (e) {
          toast(esc(errMsg(e)), 'bad');
        }
      });
    },
  });
}

// ------------------------------------------------------------------ export
async function exportCsv() {
  if (!S.files.length) return;
  try {
    const r = await API.invoke('export_csv');
    if (!r) return;
    const skipped = r.skipped ? ` ${r.skipped} files are not analysed yet and were left out.` : '';
    const link = (p) => `<a data-reveal="${esc(p)}">${esc(p.split(/[\\/]/).pop())}</a>`;
    const measures = r.measures_path ? ` Protocol measures: ${link(r.measures_path)}.` : '';
    toast(`Landmarks of ${r.files} files exported: ${link(r.path)}.${measures}${esc(skipped)}`, 'info', 9000);
  } catch (e) {
    toast(esc(errMsg(e)), 'bad');
  }
}

async function clearList() {
  const ok = await confirmDialog('Clear list', 'Remove all files from the list? Their segments and hand edits are forgotten (the .mat files are not touched).', 'Clear list');
  if (!ok) return;
  try {
    const ws = await API.invoke('remove_files', { files: S.files.map((f) => f.file) });
    S.cache.clear();
    setWorkspace(ws);
  } catch (e) {
    toast(esc(errMsg(e)), 'bad');
  }
}

// ------------------------------------------------------------------ status
const HINTS = {
  signal: 'Drag the onset / offset line (flags at the top) to move the segment · drag a P / V / A marker to move a landmark · Shift-drag to mark a segment · double-click to zoom',
  landmark: `Drag to move this landmark — it snaps to the nearest peak (hold ${ALT_KEY} to place freely) · double-click: back to automatic`,
  'landmark-drag': `Release to place · hold ${ALT_KEY} to switch snapping off`,
  boundary: 'Drag to move the onset / offset — the landmarks are recomputed at once',
};

function setHint(kind) {
  const h = HINTS[kind];
  $('#hint').innerHTML = h
    ? esc(h)
    : `<span class="kbd">␣</span> play &nbsp; <span class="kbd">←</span><span class="kbd">→</span> files &nbsp; <span class="kbd">↩</span> checked &amp; next &nbsp; <span class="kbd">Z</span> zoom`;
}

function renderProgress() {
  const { pending, total } = S.progress;
  const el = $('#progress');
  el.classList.toggle('hidden', !pending);
  if (!pending) return;
  $('#progressText').textContent = `Analysing in the background ${total - pending} / ${total}`;
  $('.bar span', el).style.width = `${total ? ((total - pending) / total) * 100 : 0}%`;
}

function renderModelStatus() {
  const el = $('#modelStatus');
  const i = S.info;
  if (!i) return;
  el.classList.remove('warn');
  if (!i.models_ready) el.textContent = 'Loading the speech model…';
  else if (i.aligner) el.textContent = 'Speech alignment ready';
  else {
    el.textContent = 'Speech model not found — segments from sound energy';
    el.classList.add('warn');
  }
}

async function pollInfo() {
  try { S.info = await API.invoke('app_info'); } catch (_) { /* retry below */ }
  renderModelStatus();
  if (!S.info?.models_ready) setTimeout(pollInfo, 700);
}

// -------------------------------------------------------- drag-and-drop in
function bindDrop() {
  const overlay = $('#dropOverlay');
  const show = (on) => {
    overlay.classList.toggle('hidden', !on);
    $('div', overlay).textContent = 'Release to add the files';
  };
  const wv = window.__TAURI__?.webview?.getCurrentWebview?.();
  if (wv?.onDragDropEvent) {
    wv.onDragDropEvent((e) => {
      const p = e.payload;
      if (p.type === 'enter' || p.type === 'over') show(true);
      else if (p.type === 'leave') show(false);
      else if (p.type === 'drop') { show(false); addPaths(p.paths); }
    });
    return;
  }
  // plain browser (interface preview): no file paths, the mock supplies them
  let depth = 0;
  document.addEventListener('dragenter', (e) => { e.preventDefault(); depth++; show(true); });
  document.addEventListener('dragover', (e) => e.preventDefault());
  document.addEventListener('dragleave', () => { if (--depth <= 0) { depth = 0; show(false); } });
  document.addEventListener('drop', (e) => { e.preventDefault(); depth = 0; show(false); addPaths(['(dropped)']); });
}

// ---------------------------------------------------------------- keyboard
function bindKeys() {
  // Windows: a lone Alt (held to place a landmark freely) would otherwise
  // activate the window menu when it is released.
  const swallowAlt = (e) => { if (e.key === 'Alt' && !IS_MAC) e.preventDefault(); };
  document.addEventListener('keyup', swallowAlt);
  document.addEventListener('keydown', (e) => {
    swallowAlt(e);
    if (closeModal) { if (e.key === 'Escape') closeModal(); return; }
    const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName);
    if (typing) { if (e.key === 'Escape') e.target.blur(); return; }
    const cmd = e.metaKey || e.ctrlKey;
    const k = e.key.toLowerCase();
    if (cmd && k === 'o') { e.preventDefault(); e.shiftKey ? openFolderDialog() : openFilesDialog(); }
    else if (cmd && k === 'e') { e.preventDefault(); exportCsv(); }
    else if (cmd && k === 'f') { e.preventDefault(); $('#filter').focus(); }
    else if (cmd) return;
    else if (e.key === ' ') { e.preventDefault(); togglePlay(); }
    else if (e.key === 'ArrowRight' || e.key === 'ArrowDown') { e.preventDefault(); step(+1); }
    else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') { e.preventDefault(); step(-1); }
    else if (e.key === 'Enter') { e.preventDefault(); setReviewed(true, true); }
    else if (k === 'z') { toggleZoomMode(); renderSegBar(); }
    else if (e.key === 'Escape') { viewer.stop(); hideSelBar(); viewer.render(); }
  });
}

// ------------------------------------------------------------------- setup
function bindToolbar() {
  $('#btnOpenFiles').addEventListener('click', openFilesDialog);
  $('#btnOpenFolder').addEventListener('click', openFolderDialog);
  $('#btnPrev').addEventListener('click', () => step(-1));
  $('#btnNext').addEventListener('click', () => step(+1));
  $('#btnReviewed').addEventListener('click', () => {
    const sum = S.byFile.get(S.current);
    setReviewed(!sum?.reviewed, !sum?.reviewed);
  });
  $('#dimSel').addEventListener('click', (e) => {
    const b = e.target.closest('[data-dim]');
    if (!b) return;
    S.dim = b.dataset.dim;
    $$('#dimSel button').forEach((x) => x.classList.toggle('on', x === b));
    viewer.setDisplay({ dim: S.dim });
    renderTable();
    savePrefs();
  });
  $('#rowSel').addEventListener('click', (e) => {
    const b = e.target.closest('[data-row]');
    if (!b) return;
    const r = b.dataset.row;
    if (S.rows.has(r) && S.rows.size > 1) S.rows.delete(r);
    else S.rows.add(r);
    $$('#rowSel button').forEach((x) => x.classList.toggle('on', S.rows.has(x.dataset.row)));
    viewer.setDisplay({ rows: new Set(S.rows) });
    savePrefs();
  });
  $('#btnSpec').addEventListener('click', () => {
    S.spec = !S.spec;
    $('#btnSpec').classList.toggle('on', S.spec);
    viewer.setDisplay({ spec: S.spec });
    savePrefs();
  });
  $('#btnExport').addEventListener('click', exportCsv);
  $('#btnSettings').addEventListener('click', settingsDialog);
  $('#btnClear').addEventListener('click', clearList);
  $('#legend').addEventListener('click', (e) => {
    const b = e.target.closest('[data-st]');
    if (!b) return;
    S.statusFilter = S.statusFilter === b.dataset.st ? null : b.dataset.st;
    renderFileList();
    scrollToCurrent();
  });
  $('#filter').addEventListener('input', (e) => { S.filter = e.target.value; renderFileList(); scrollToCurrent(); });
  $('#fileList').addEventListener('click', (e) => {
    const row = e.target.closest('.frow');
    if (row) openFile(row.dataset.file);
  });
}

function renderAll() {
  renderToolbar();
  renderFileList();
  renderEmpty();
  renderSegBar();
  renderTable();
  renderProgress();
  renderModelStatus();
  setHint(null);
  viewer?.render();
}

async function init() {
  viewer = new SignalViewer($('#viewerHost'), {
    onSegmentChange,
    onLandmarkChange,
    onLandmarkReset,
    onProtocolLandmarkChange,
    onProtocolLandmarkReset,
    onSegmentSelect: selectSegment,
    onSelectRange: (a, b) => (a == null ? hideSelBar() : showSelBar(a, b)),
    onZoomToggle: () => { toggleZoomMode(); renderSegBar(); },
    onHint: setHint,
  });
  viewer.setDisplay({ dim: S.dim, rows: new Set(S.rows), spec: S.spec });
  bindToolbar();
  bindKeys();
  bindDrop();
  renderAll();
  API.on('trial-updated', (sum) => {
    updateSummary(sum);
    if (sum.file === S.current && !S.result) reloadResult(sum.file);
  });
  API.on('analysis-progress', (p) => {
    const had = S.progress.pending;
    S.progress = p;
    renderProgress();
    if (had && !p.pending) renderFileList();
  });
  API.on('analysis-error', (msg) => toast(esc(msg), 'bad'));
  pollInfo();
  try {
    setWorkspace(await API.invoke('get_workspace'));
  } catch (e) {
    toast(esc(errMsg(e)), 'bad');
  }
  let last = null;
  try { last = localStorage.getItem('scissortail.last'); } catch (_) { /* storage unavailable */ }
  const start = S.byFile.has(last) ? last : S.files[0]?.file;
  if (start) openFile(start);
}

document.addEventListener('DOMContentLoaded', init);
