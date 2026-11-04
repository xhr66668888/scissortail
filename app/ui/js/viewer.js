// Multi-track signal display in the style of MAVIS / MVIEW: waveform,
// spectrogram and one panel per articulator (position / velocity /
// acceleration rows) on a shared time axis. The speech-segment boundaries and
// the kinematic landmarks sit on the signals and can be dragged.
'use strict';

function b64ToBytes(b64) {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

function pcm16ToFloat(b64) {
  const bytes = b64ToBytes(b64);
  const i16 = new Int16Array(bytes.buffer, bytes.byteOffset, Math.floor(bytes.byteLength / 2));
  const f = new Float32Array(i16.length);
  for (let i = 0; i < i16.length; i++) f[i] = i16[i] / 32768;
  return f;
}

// Drawn like a classic MATLAB figure printed in black and white: grey figure,
// white axes with black frames and inward ticks, black traces. The three
// landmark signals are told apart by line style.
const SIG = {
  pos: { dash: [], key: 'position', tag: 'P', unit: 'mm' },
  vel: { dash: [5, 3], key: 'velocity', tag: 'V', unit: 'mm/s' },
  acc: { dash: [1.5, 2.5], key: 'acceleration', tag: 'A', unit: 'mm/s²' },
};
const INK = '#000000';
const PAPER = '#ffffff';
const FIGURE = '#cccccc'; // MATLAB's figure colour [0.8 0.8 0.8], as --fig in app.css
const SHADE = '#e0e0e0'; // the active speech segment
const SHADE_OTHER = '#f0f0f0';
const FONT = 'Helvetica, Arial, "PingFang SC", "Microsoft YaHei", sans-serif';
const SNAP_PX = 10;
const HIT_PX = 6;
const TAG_STRIP = 17;
const TICK = 4;

// The backend names the 3-D dimension "tangential"; tracks use "3d".
const dimKey = (dim) => (dim === '3d' ? 'tangential' : dim);

// Playback opens an AudioContext only for as long as it is needed. A context
// left open in WKWebView goes silent for good once macOS switches the output
// device (e.g. AirPods taken out): WebKit keeps rendering, the speaker gets
// zeros. An open context also keeps the device running and the Mac awake.
const AUDIO_KEEP_MS = 3000; // a replay within this time reuses the context
let audioCtx = null;
let audioCtxTimer = 0;

function openAudio() {
  clearTimeout(audioCtxTimer);
  if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  return audioCtx;
}

function closeAudioSoon() {
  clearTimeout(audioCtxTimer);
  audioCtxTimer = setTimeout(() => {
    const ctx = audioCtx;
    audioCtx = null;
    ctx?.close().catch(() => {});
  }, AUDIO_KEEP_MS);
}

function fmtVal(v) {
  if (v == null || !isFinite(v)) return '—';
  const a = Math.abs(v);
  return a >= 1000 ? v.toFixed(0) : a >= 100 ? v.toFixed(1) : a >= 10 ? v.toFixed(2) : v.toFixed(3);
}

/** Round tick values for a y range, about one per 25 px, as MATLAB picks them. */
function yTicks(mn, mx, h) {
  const raw = (mx - mn) / Math.max(2, Math.floor(h / 25));
  const p = Math.pow(10, Math.floor(Math.log10(raw)));
  const cands = [1, 2, 2.5, 5, 10, 20].map((m) => m * p);
  // the round step nearest the target, but labels never closer than 14 px
  const minStep = (14 / h) * (mx - mn);
  let step = cands.reduce((a, b) => (Math.abs(Math.log(b / raw)) < Math.abs(Math.log(a / raw)) ? b : a));
  if (step < minStep) step = cands.find((s) => s >= minStep) || step;
  let digits = Math.max(0, Math.ceil(-Math.log10(step) - 1e-9));
  while (digits < 6 && Math.abs(step * 10 ** digits - Math.round(step * 10 ** digits)) > 1e-6) digits++;
  const ticks = [];
  for (let k = Math.ceil(mn / step); k <= Math.floor(mx / step); k++) ticks.push(k === 0 ? 0 : k * step);
  return { ticks, digits };
}

class SignalViewer {
  constructor(host, opts = {}) {
    this.host = host;
    this.opts = opts;
    this.canvas = document.createElement('canvas');
    this.ctx = this.canvas.getContext('2d');
    host.appendChild(this.canvas);
    this.gutter = 112;
    this.right = 16;
    this.view = null;
    this.result = null;
    this.seg = 0;
    this.display = { dim: 'z', rows: new Set(['pos', 'vel', 'acc']), spec: true };
    this.t0 = 0;
    this.t1 = 1;
    this.hover = null;
    this.drag = null;
    this.selection = null;
    this.playhead = null;
    this.source = null;
    this.tracks = [];
    this.markers = [];
    this._frame = 0;
    this._bind();
    this.ro = new ResizeObserver(() => this.render());
    this.ro.observe(host);
  }

  // ------------------------------------------------------------------ data
  setTrial(view) {
    this.stop();
    this.view = view;
    this.samples = view.audio ? pcm16ToFloat(view.audio.pcm16) : null;
    this.fs = view.audio ? view.audio.fs : 24000;
    this.duration = view.duration || (this.samples ? this.samples.length / this.fs : 1);
    this._buildSpec(view.spec);
    this.audioBuffer = null;
    this._env = null;
    this.t0 = 0;
    this.t1 = this.duration;
    this.selection = null;
    this.hover = null;
    this.drag = null;
    this.render();
  }

  setResult(result, seg = 0) {
    this.result = result;
    const n = result?.occurrences?.length || 0;
    this.seg = Math.max(0, Math.min(seg, n - 1));
    this.render();
  }

  setDisplay(d) {
    Object.assign(this.display, d);
    this.render();
  }

  occ() {
    return this.result?.occurrences?.[this.seg] || null;
  }

  sensorsShown() {
    const dim = this.display.dim;
    return (this.view?.sensors || []).filter((s) => !s.is_reference && s.dims?.[dim]);
  }

  _buildSpec(spec) {
    this.specCanvas = null;
    this.spec = spec;
    if (!spec || !spec.data) return;
    const px = b64ToBytes(spec.data);
    const c = document.createElement('canvas');
    c.width = spec.w;
    c.height = spec.h;
    const g = c.getContext('2d');
    const img = g.createImageData(spec.w, spec.h);
    for (let i = 0; i < px.length; i++) {
      const v = 255 - px[i];
      img.data[4 * i] = v;
      img.data[4 * i + 1] = v;
      img.data[4 * i + 2] = v;
      img.data[4 * i + 3] = 255;
    }
    g.putImageData(img, 0, 0);
    this.specCanvas = c;
  }

  // ---------------------------------------------------------------- layout
  _layout(W, H) {
    const tracks = [];
    let y = 8;
    const gap = 8;
    tracks.push({ kind: 'wave', y, h: 64 });
    y += 64 + gap;
    if (this.display.spec && this.specCanvas) {
      tracks.push({ kind: 'spec', y, h: 96 });
      y += 96 + gap;
    }
    const sensors = this.sensorsShown();
    const rows = ['pos', 'vel', 'acc'].filter((r) => this.display.rows.has(r));
    const axisH = 26;
    const panelGap = 10;
    if (sensors.length && rows.length) {
      const avail = H - y - axisH - sensors.length * (panelGap + TAG_STRIP);
      const rowH = Math.max(36, Math.min(150, Math.floor(avail / (sensors.length * rows.length))));
      for (const s of sensors) {
        // the P / V / A tags sit in a strip above the axes so they never hide a peak
        const panel = { kind: 'sensor', sensor: s, y, top: y + TAG_STRIP, rows: [] };
        y += TAG_STRIP;
        const d = s.dims[this.display.dim];
        for (const r of rows) {
          panel.rows.push({ sig: r, y, h: rowH, data: d[r], fs: s.fs });
          y += rowH;
        }
        panel.h = y - panel.y;
        tracks.push(panel);
        y += panelGap;
      }
    }
    // the time axis is the bottom edge of the lowest axes, as in a MATLAB subplot column
    this.axisY = y - (sensors.length && rows.length ? panelGap : gap);
    this.totalHeight = this.axisY + axisH;
    this.tracks = tracks;
  }

  plotW() {
    return this.cssW - this.gutter - this.right;
  }

  x(t) {
    return this.gutter + ((t - this.t0) / (this.t1 - this.t0)) * this.plotW();
  }

  tAt(x) {
    return this.t0 + ((x - this.gutter) / this.plotW()) * (this.t1 - this.t0);
  }

  secPerPx() {
    return (this.t1 - this.t0) / this.plotW();
  }

  // ---------------------------------------------------------------- render
  render() {
    if (this._frame) return;
    this._frame = requestAnimationFrame(() => {
      this._frame = 0;
      this._draw();
    });
  }

  _draw() {
    const host = this.host;
    const dpr = window.devicePixelRatio || 1;
    const W = host.clientWidth;
    const H0 = host.clientHeight;
    if (!W || !H0) return;
    this.cssW = W;
    this._layout(W, H0);
    const H = Math.max(H0, this.totalHeight);
    if (this.canvas.width !== Math.round(W * dpr) || this.canvas.height !== Math.round(H * dpr)) {
      this.canvas.width = Math.round(W * dpr);
      this.canvas.height = Math.round(H * dpr);
      this.canvas.style.width = W + 'px';
      this.canvas.style.height = H + 'px';
    }
    const g = this.ctx;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.fillStyle = FIGURE;
    g.fillRect(0, 0, W, H);
    if (!this.view) return;

    this.tk = this._ticks();
    this.axes = this.tracks.map((t) => (t.kind === 'sensor' ? { y: t.top, h: t.y + t.h - t.top } : { y: t.y, h: t.h }));
    g.fillStyle = PAPER;
    for (const a of this.axes) g.fillRect(this.gutter, a.y, this.plotW(), a.h);
    this._drawSegments(g);
    for (const tr of this.tracks) {
      if (tr.kind === 'wave') this._drawWave(g, tr);
      else if (tr.kind === 'spec') this._drawSpec(g, tr);
      else this._drawSensor(g, tr);
    }
    this._drawSelection(g);
    this._drawBoundaries(g);
    this._drawMarkers(g);
    this._drawAxis(g);
    this._drawHover(g);
    if (this.playhead != null) {
      const px = Math.round(this.x(this.playhead)) + 0.5;
      if (px >= this.gutter && px <= W - this.right) {
        g.strokeStyle = INK;
        g.lineWidth = 1;
        g.beginPath();
        g.moveTo(px, 4);
        g.lineTo(px, this.axisY);
        g.stroke();
      }
    }
  }

  /** Axes frame with MATLAB's inward ticks along the top and bottom edge. */
  _box(g, y, h) {
    const x0 = this.gutter + 0.5, x1 = this.gutter + this.plotW() - 0.5;
    g.strokeStyle = INK;
    g.lineWidth = 1;
    g.strokeRect(x0, y + 0.5, x1 - x0, h - 1);
    g.beginPath();
    for (const t of this.tk.ticks) {
      const px = Math.round(this.x(t)) + 0.5;
      if (px <= x0 + 1 || px >= x1 - 1) continue;
      g.moveTo(px, y + 1);
      g.lineTo(px, y + 1 + TICK);
      g.moveTo(px, y + h - 1);
      g.lineTo(px, y + h - 1 - TICK);
    }
    g.stroke();
  }

  _text(g, s, x, y, { size = 11, weight = 400, color = INK, align = 'left', base = 'middle', halo = false } = {}) {
    g.font = `${weight} ${size}px ${FONT}`;
    g.textAlign = align;
    g.textBaseline = base;
    if (halo) {
      g.strokeStyle = PAPER;
      g.lineWidth = 3;
      g.lineJoin = 'round';
      g.strokeText(s, x, y);
    }
    g.fillStyle = color;
    g.fillText(s, x, y);
  }

  _niceStep(span, px) {
    const target = span / Math.max(2, px / 90);
    const steps = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.25, 0.5, 1, 2, 5, 10, 20, 30, 60];
    return steps.find((s) => s >= target) || 60;
  }

  _ticks() {
    const step = this._niceStep(this.t1 - this.t0, this.plotW());
    const out = [];
    for (let t = Math.ceil(this.t0 / step) * step; t <= this.t1 + 1e-9; t += step) out.push(t);
    return { step, ticks: out };
  }

  _drawSegments(g) {
    const occs = this.result?.occurrences || [];
    occs.forEach((o, i) => {
      const oc = o.occurrence;
      const xa = Math.max(this.gutter, this.x(oc.start_s));
      const xb = Math.min(this.gutter + this.plotW(), this.x(oc.end_s));
      if (xb <= xa) return;
      g.fillStyle = i === this.seg ? SHADE : SHADE_OTHER;
      for (const a of this.axes) g.fillRect(xa, a.y, xb - xa, a.h);
    });
  }

  _envelope(tr) {
    const key = `${this.t0}|${this.t1}|${this.cssW}`;
    if (this._env?.key === key) return this._env.data;
    const n = Math.ceil(this.plotW());
    const data = new Float32Array(n * 2);
    for (let k = 0; k < n; k++) {
      const ta = this.tAt(this.gutter + k), tb = this.tAt(this.gutter + k + 1);
      let a = Math.max(0, Math.floor(ta * this.fs)), b = Math.min(this.samples.length, Math.ceil(tb * this.fs));
      if (b <= a) b = Math.min(this.samples.length, a + 1);
      let mn = 0, mx = 0;
      for (let i = a; i < b; i++) {
        const v = this.samples[i];
        if (v < mn) mn = v;
        if (v > mx) mx = v;
      }
      data[2 * k] = mn;
      data[2 * k + 1] = mx;
    }
    this._env = { key, data };
    return data;
  }

  _drawWave(g, tr) {
    this._text(g, 'Audio', 10, tr.y + 11, { weight: 700 });
    this._text(g, `${Math.round(this.fs / 1000)} kHz`, 10, tr.y + 25, { size: 10 });
    if (this.samples) {
      const env = this._envelope(tr);
      const mid = tr.y + tr.h / 2;
      const amp = tr.h / 2 - 4;
      g.strokeStyle = INK;
      g.lineWidth = 1;
      g.beginPath();
      for (let k = 0; k < env.length / 2; k++) {
        const px = this.gutter + k + 0.5;
        g.moveTo(px, mid - env[2 * k + 1] * amp);
        g.lineTo(px, mid - env[2 * k] * amp + 0.5);
      }
      g.stroke();
    }
    this._box(g, tr.y, tr.h);
  }

  _drawSpec(g, tr) {
    this._text(g, 'Spectrogram', 10, tr.y + 11, { weight: 700 });
    this._text(g, `0–${Math.round(this.spec.fmax / 1000)} kHz`, 10, tr.y + 25, { size: 10 });
    const spc = this.spec.sec_per_col;
    const sx = Math.max(0, this.t0 / spc);
    const sw = Math.max(1, (this.t1 - this.t0) / spc);
    const vis = Math.min(sw, this.spec.w - sx);
    g.imageSmoothingEnabled = true;
    if (vis > 0) g.drawImage(this.specCanvas, sx, 0, vis, this.spec.h, this.gutter, tr.y, (vis / sw) * this.plotW(), tr.h);
    this._box(g, tr.y, tr.h);
  }

  _rowScale(row) {
    const [a, b] = this._visibleRange(row.data, row.fs);
    let mn = Infinity, mx = -Infinity;
    for (let i = a; i < b; i++) {
      const v = row.data[i];
      if (v < mn) mn = v;
      if (v > mx) mx = v;
    }
    if (!isFinite(mn)) { mn = -1; mx = 1; }
    if (row.sig !== 'pos' && this.display.dim !== '3d') { mn = Math.min(mn, 0); mx = Math.max(mx, 0); }
    if (mx - mn < 1e-9) { mx += 1; mn -= 1; }
    const pad = (mx - mn) * 0.1;
    return [mn - pad, mx + pad];
  }

  _visibleRange(data, fs) {
    const a = Math.max(0, Math.floor(this.t0 * fs) - 1);
    const b = Math.min(data.length, Math.ceil(this.t1 * fs) + 2);
    return [a, b];
  }

  _drawSensor(g, panel) {
    const s = panel.sensor;
    const dimLabel = { z: 'z', x: 'x', y: 'y', '3d': '3D' }[this.display.dim];
    this._text(g, s.name, 10, panel.top + 9, { weight: 700, size: 12 });
    this._text(g, dimLabel, 10 + g.measureText(s.name).width + 5, panel.top + 9);
    const x0 = this.gutter, x1 = this.gutter + this.plotW();
    for (const row of panel.rows) {
      const [mn, mx] = this._rowScale(row);
      row.yOf = (v) => row.y + row.h - ((v - mn) / (mx - mn)) * row.h;
      row.range = [mn, mx];
      const st = SIG[row.sig];
      // row label and y-axis ticks (MATLAB-style: labels left of the frame, ticks pointing in)
      const name = { pos: 'position', vel: 'velocity', acc: 'acceleration' }[row.sig];
      const labelY = row === panel.rows[0] ? Math.max(row.y + row.h / 2, row.y + 34) : row.y + row.h / 2;
      if (row.h >= 30) {
        this._text(g, name, 10, labelY - 6);
        this._text(g, `(${st.unit})`, 10, labelY + 8, { size: 10 });
      }
      const yt = yTicks(mn, mx, row.h);
      g.strokeStyle = INK;
      g.lineWidth = 1;
      g.beginPath();
      for (const v of yt.ticks) {
        const py = Math.round(row.yOf(v)) + 0.5;
        g.moveTo(x0, py);
        g.lineTo(x0 + TICK, py);
        g.moveTo(x1, py);
        g.lineTo(x1 - TICK, py);
      }
      g.stroke();
      for (const v of yt.ticks) {
        const py = row.yOf(v);
        // a label close to the row edge would run into the neighbouring row's
        if (py - row.y >= 6 && row.y + row.h - py >= 6) this._text(g, String(+v.toFixed(yt.digits)), this.gutter - 4, py, { size: 10, align: 'right' });
      }
      g.save();
      g.beginPath();
      g.rect(this.gutter, row.y, this.plotW(), row.h);
      g.clip();
      if (mn < 0 && mx > 0) {
        g.strokeStyle = '#8c8c8c';
        g.setLineDash([1, 2]);
        g.beginPath();
        g.moveTo(x0, Math.round(row.yOf(0)) + 0.5);
        g.lineTo(x1, Math.round(row.yOf(0)) + 0.5);
        g.stroke();
        g.setLineDash([]);
      }
      const [a, b] = this._visibleRange(row.data, row.fs);
      g.strokeStyle = INK;
      g.lineWidth = 1;
      g.beginPath();
      for (let i = a; i < b; i++) {
        const px = this.x(i / row.fs), py = row.yOf(row.data[i]);
        if (i === a) g.moveTo(px, py); else g.lineTo(px, py);
      }
      g.stroke();
      g.restore();
      if (row !== panel.rows[panel.rows.length - 1]) {
        g.strokeStyle = INK;
        g.beginPath();
        g.moveTo(x0, row.y + row.h + 0.5);
        g.lineTo(x1, row.y + row.h + 0.5);
        g.stroke();
      }
    }
    this._box(g, panel.top, panel.y + panel.h - panel.top);
  }

  // ------------------------------------------------------------- markers
  /** Landmarks of the active segment for the shown dimension, one per sensor × signal. */
  _collectMarkers() {
    const out = [];
    const occ = this.occ();
    if (!occ) return out;
    const dk = dimKey(this.display.dim);
    for (const panel of this.tracks) {
      if (panel.kind !== 'sensor') continue;
      const kin = occ.kinematics.find((k) => k.sensor === panel.sensor.name && k.dim === dk);
      if (!kin) continue;
      for (const sig of ['pos', 'vel', 'acc']) {
        let peak = kin[SIG[sig].key];
        if (!peak) continue;
        const dragging = this.drag?.type === 'landmark' && this.drag.sensor === panel.sensor.name && this.drag.sig === sig;
        if (dragging) peak = { ...peak, time_s: this.drag.t, value: this.drag.v, manual: true, at_edge: false };
        const row = panel.rows.find((r) => r.sig === sig) || null;
        out.push({ panel, row, sig, peak, sensor: panel.sensor.name, dragging, data: panel.sensor.dims[this.display.dim][sig], fs: panel.sensor.fs });
      }
    }
    return out;
  }

  _drawMarkers(g) {
    this.markers = this._collectMarkers();
    const occ = this.occ();
    for (const panel of this.tracks.filter((t) => t.kind === 'sensor')) {
      const ms = this.markers.filter((m) => m.panel === panel);
      // tags along the top edge of the panel; nudge overlapping ones apart
      const tags = ms.map((m) => ({ m, x: this.x(m.peak.time_s) })).sort((a, b) => a.x - b.x);
      for (let i = 1; i < tags.length; i++) if (tags[i].x - tags[i - 1].x < 15) tags[i].x = tags[i - 1].x + 15;
      for (const { m, x: tagX } of tags) {
        const px = this.x(m.peak.time_s);
        if (px < this.gutter - 1 || px > this.cssW - this.right + 1) continue;
        const st = SIG[m.sig];
        const hot = m.dragging || (this.hover?.marker && this.hover.marker.sensor === m.sensor && this.hover.marker.sig === m.sig);
        const lx = Math.round(px) + 0.5;
        g.save();
        g.strokeStyle = INK;
        g.lineWidth = hot ? 2 : 1;
        g.setLineDash(st.dash);
        g.beginPath();
        g.moveTo(lx, panel.top - 3);
        g.lineTo(lx, panel.y + panel.h - 1);
        g.stroke();
        g.setLineDash([]);
        if (m.row && m.row.yOf) {
          // round = automatic, square = placed by hand, hollow = on the segment edge
          const py = m.row.yOf(m.peak.value);
          const r = hot ? 4.5 : 3.5;
          g.beginPath();
          if (m.peak.manual) g.rect(px - r, py - r, 2 * r, 2 * r);
          else g.arc(px, py, r, 0, Math.PI * 2);
          g.fillStyle = m.peak.at_edge ? PAPER : INK;
          g.fill();
          g.lineWidth = 1.5;
          g.strokeStyle = m.peak.at_edge ? INK : PAPER;
          g.stroke();
          if (m.dragging && this.drag.snapped) {
            g.beginPath();
            g.arc(px, py, 8.5, 0, Math.PI * 2);
            g.strokeStyle = INK;
            g.lineWidth = 1;
            g.stroke();
          }
        }
        // tag: white = automatic, black = placed by hand
        const tw = 13, th = 14, ty = panel.y;
        const tx = Math.round(Math.min(Math.max(tagX - tw / 2, this.gutter + 1), this.cssW - this.right - tw - 1));
        g.fillStyle = m.peak.manual ? INK : PAPER;
        g.fillRect(tx, ty, tw, th);
        g.strokeStyle = INK;
        g.lineWidth = 1;
        g.strokeRect(tx + 0.5, ty + 0.5, tw - 1, th - 1);
        if (hot) g.strokeRect(tx - 0.5, ty - 0.5, tw + 1, th + 1);
        this._text(g, st.tag, tx + tw / 2, ty + th / 2 + 0.5, { size: 10, weight: 700, color: m.peak.manual ? PAPER : INK, align: 'center' });
        g.restore();
        if (hot && occ) this._markerLabel(g, m, px, occ);
      }
      // the original position of a landmark being dragged, as a faint ghost
      if (this.drag?.type === 'landmark' && this.drag.panelSensor === panel.sensor.name) {
        const gx = Math.round(this.x(this.drag.orig.time_s)) + 0.5;
        g.save();
        g.strokeStyle = '#8c8c8c';
        g.lineWidth = 1;
        g.setLineDash([2, 3]);
        g.beginPath();
        g.moveTo(gx, panel.top);
        g.lineTo(gx, panel.y + panel.h - 1);
        g.stroke();
        g.restore();
      }
    }
  }

  /** A MATLAB-style data tip: X relative to the onset, Y the signal value. */
  _markerLabel(g, m, px, occ) {
    const st = SIG[m.sig];
    const rel = (m.peak.time_s - occ.occurrence.start_s) * 1000;
    const txt = `${st.tag}   X: ${rel >= 0 ? '+' : ''}${rel.toFixed(0)} ms   Y: ${fmtVal(m.peak.value)} ${st.unit}`;
    g.font = `400 11px ${FONT}`;
    const w = Math.ceil(g.measureText(txt).width) + 10;
    const y = (m.row?.yOf ? m.row.yOf(m.peak.value) : m.panel.top + 20) - 26;
    const bx = Math.round(Math.min(Math.max(px + 8, this.gutter + 2), this.cssW - this.right - w - 2));
    const by = Math.round(Math.max(m.panel.top + 2, Math.min(y, m.panel.y + m.panel.h - 20)));
    g.fillStyle = PAPER;
    g.fillRect(bx, by, w, 17);
    g.strokeStyle = INK;
    g.lineWidth = 1;
    g.strokeRect(bx + 0.5, by + 0.5, w - 1, 16);
    this._text(g, txt, bx + 5, by + 9);
  }

  _drawBoundaries(g) {
    const occs = this.result?.occurrences || [];
    occs.forEach((o, i) => {
      const oc = o.occurrence;
      const active = i === this.seg;
      for (const k of ['start_s', 'end_s']) {
        let t = oc[k];
        if (active && this.drag?.type === 'boundary' && this.drag.key === k) t = this.drag.t;
        const px = Math.round(this.x(t)) + 0.5;
        if (px < this.gutter - 1 || px > this.cssW - this.right + 1) continue;
        const hot = active && (this.drag?.key === k || this.hover?.boundary === k);
        g.save();
        g.strokeStyle = active ? INK : '#8c8c8c';
        g.lineWidth = hot ? 3 : active ? 2 : 1;
        g.beginPath();
        g.moveTo(px, 6);
        g.lineTo(px, this.axisY);
        g.stroke();
        if (active) {
          // flag handle at the top, pointing into the segment
          const d = k === 'start_s' ? 1 : -1;
          g.fillStyle = INK;
          g.beginPath();
          g.moveTo(px, 3);
          g.lineTo(px + d * 11, 3);
          g.lineTo(px + d * 11, 14);
          g.lineTo(px, 18);
          g.closePath();
          g.fill();
        }
        g.restore();
      }
      if (active) {
        const xa = Math.max(this.gutter + 4, this.x(oc.start_s) + 15);
        const xb = this.x(oc.end_s) - 15;
        if (xb - xa > 30) this._text(g, `${oc.word}${occs.length > 1 ? `  #${i + 1}` : ''}`, xa, 15, { weight: 700 });
      }
    });
  }

  _drawSelection(g) {
    if (!this.selection) return;
    const [a, b] = this.selection;
    const xa = Math.round(this.x(Math.min(a, b))) + 0.5, xb = Math.round(this.x(Math.max(a, b))) + 0.5;
    g.save();
    g.fillStyle = 'rgba(0, 0, 0, 0.1)';
    for (const ax of this.axes) g.fillRect(xa, ax.y, xb - xa, ax.h);
    g.strokeStyle = INK;
    g.lineWidth = 1;
    g.setLineDash([3, 3]);
    g.strokeRect(xa, 6.5, xb - xa, this.axisY - 6);
    g.restore();
  }

  _drawAxis(g) {
    const y = this.axisY;
    const { step, ticks } = this.tk;
    const digits = step < 0.01 ? 3 : step < 0.1 ? 2 : step < 1 ? 1 : 0;
    for (const t of ticks) {
      const px = this.x(t);
      if (px < this.gutter - 0.5 || px > this.gutter + this.plotW() + 0.5) continue;
      this._text(g, t.toFixed(digits), px, y + 4, { size: 10, align: 'center', base: 'top' });
    }
    this._text(g, 'time (s)', 10, y + 4, { size: 10, base: 'top' });
  }

  _drawHover(g) {
    if (!this.hover || this.drag?.type === 'landmark' || this.drag?.type === 'boundary' || this.hover.marker) return;
    const px = Math.round(this.x(this.hover.t)) + 0.5;
    if (px < this.gutter || px > this.cssW - this.right) return;
    g.save();
    g.strokeStyle = INK;
    g.lineWidth = 1;
    g.setLineDash([1, 2]);
    g.beginPath();
    g.moveTo(px, 8);
    g.lineTo(px, this.axisY);
    g.stroke();
    g.restore();
    const label = `${this.hover.t.toFixed(3)} s`;
    g.font = `400 11px ${FONT}`;
    const tw = Math.ceil(g.measureText(label).width) + 10;
    const lx = Math.round(Math.min(px + 4, this.cssW - this.right - tw - 2));
    const ly = this.axisY - 20;
    g.fillStyle = PAPER;
    g.fillRect(lx, ly, tw, 16);
    g.strokeStyle = INK;
    g.strokeRect(lx + 0.5, ly + 0.5, tw - 1, 15);
    this._text(g, label, lx + 5, ly + 8);
    for (const panel of this.tracks) {
      if (panel.kind !== 'sensor') continue;
      for (const row of panel.rows) {
        if (!row.yOf) continue;
        const i = Math.round(this.hover.t * row.fs);
        if (i < 0 || i >= row.data.length) continue;
        const v = row.data[i];
        const py = row.yOf(v);
        g.fillStyle = INK;
        g.fillRect(Math.round(px) - 2, Math.round(py) - 2, 4, 4);
        this._text(g, fmtVal(v), Math.min(px + 6, this.cssW - this.right - 44), Math.min(row.y + row.h - 7, Math.max(row.y + 7, py - 7)), { size: 10, halo: true });
      }
    }
  }

  // ------------------------------------------------------------ zoom/pan
  zoomTo(a, b) {
    const minSpan = 0.03;
    a = Math.max(0, a);
    b = Math.min(this.duration, b);
    if (b - a < minSpan) {
      const c = (a + b) / 2;
      a = Math.max(0, c - minSpan / 2);
      b = Math.min(this.duration, a + minSpan);
    }
    this.t0 = a;
    this.t1 = b;
    this.render();
  }

  zoomFit() {
    this.zoomTo(0, this.duration);
  }

  zoomSegment(pad = 0.15) {
    const o = this.occ()?.occurrence;
    if (!o) return this.zoomFit();
    const p = Math.max(pad, (o.end_s - o.start_s) * 0.35);
    this.zoomTo(o.start_s - p, o.end_s + p);
  }

  toggleZoom() {
    const o = this.occ()?.occurrence;
    if (o && this.t1 - this.t0 > (o.end_s - o.start_s) * 2.4) this.zoomSegment();
    else this.zoomFit();
  }

  zoomBy(f, center) {
    const c = center ?? (this.t0 + this.t1) / 2;
    this.zoomTo(c - (c - this.t0) * f, c + (this.t1 - c) * f);
  }

  // ------------------------------------------------------------- playback
  _ensureBuffer() {
    if (!this.audioBuffer && this.samples) {
      this.audioBuffer = new AudioBuffer({ numberOfChannels: 1, length: this.samples.length, sampleRate: this.fs });
      this.audioBuffer.copyToChannel(this.samples, 0);
    }
    return this.audioBuffer;
  }

  play(a = this.t0, b = this.t1) {
    this.stop();
    const buf = this._ensureBuffer();
    if (!buf) return;
    const fresh = !audioCtx;
    const ctx = openAudio();
    if (ctx.state !== 'running') ctx.resume();
    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.connect(ctx.destination);
    a = Math.max(0, a);
    b = Math.min(this.duration, b);
    // a new context starts with a little silence: a Bluetooth headset that is
    // just waking up would otherwise swallow the onset
    const start = ctx.currentTime + (fresh ? 0.1 : 0.02);
    src.start(start, a, Math.max(0.01, b - a));
    this.source = src;
    const done = () => {
      if (this.source !== src) return;
      this.source = null;
      this.playhead = null;
      closeAudioSoon();
      this.render();
      this.opts.onPlayEnd?.();
    };
    const tick = () => {
      if (this.source !== src) return;
      const t = a + (ctx.currentTime - start);
      if (t >= b) return done();
      this.playhead = Math.max(a, t);
      this.render();
      requestAnimationFrame(tick);
    };
    src.onended = done;
    requestAnimationFrame(tick);
  }

  isPlaying() {
    return !!this.source;
  }

  stop() {
    if (this.source) {
      try { this.source.stop(); } catch (_) { /* already stopped */ }
      this.source = null;
      closeAudioSoon();
    }
    this.playhead = null;
  }

  // --------------------------------------------------------- interaction
  _panelAt(y) {
    return this.tracks.find((t) => t.kind === 'sensor' && y >= t.y && y <= t.y + t.h) || null;
  }

  _hitMarker(x, y) {
    const panel = this._panelAt(y);
    if (!panel || this.opts.readOnly) return null;
    const near = this.markers.filter((m) => m.panel === panel && Math.abs(this.x(m.peak.time_s) - x) <= HIT_PX);
    if (!near.length) {
      // the tags along the top edge are handles too (they may be nudged apart)
      if (y <= panel.top) {
        const tags = this.markers.filter((m) => m.panel === panel).map((m) => ({ m, x: this.x(m.peak.time_s) })).sort((a, b) => a.x - b.x);
        for (let i = 1; i < tags.length; i++) if (tags[i].x - tags[i - 1].x < 15) tags[i].x = tags[i - 1].x + 15;
        const hit = tags.find((t) => Math.abs(t.x - x) <= 7);
        return hit ? hit.m : null;
      }
      return null;
    }
    // overlapping markers: the one whose row is under the pointer wins
    const row = panel.rows.find((r) => y >= r.y && y <= r.y + r.h);
    return near.find((m) => m.row && m.row === row) || near.sort((a, b) => Math.abs(this.x(a.peak.time_s) - x) - Math.abs(this.x(b.peak.time_s) - x))[0];
  }

  _hitBoundary(x) {
    const o = this.occ()?.occurrence;
    if (!o || this.opts.readOnly) return null;
    const ds = Math.abs(x - this.x(o.start_s)), de = Math.abs(x - this.x(o.end_s));
    if (Math.min(ds, de) > HIT_PX) return null;
    return ds <= de ? 'start_s' : 'end_s';
  }

  /** Nearest local extremum of `data` within SNAP_PX of time t (sample index), or null. */
  _snapIndex(data, fs, t) {
    const c = t * fs;
    const r = (SNAP_PX * this.secPerPx()) * fs;
    const lo = Math.max(1, Math.floor(c - r)), hi = Math.min(data.length - 2, Math.ceil(c + r));
    let best = null;
    for (let i = lo; i <= hi; i++) {
      const a = data[i] - data[i - 1], b = data[i + 1] - data[i];
      if (a * b <= 0 && (a !== 0 || b !== 0)) {
        if (best == null || Math.abs(i - c) < Math.abs(best - c)) best = i;
      }
    }
    return best;
  }

  _valueAt(data, fs, t) {
    const x = Math.min(Math.max(t * fs, 0), data.length - 1);
    const i = Math.floor(x), j = Math.min(i + 1, data.length - 1);
    return data[i] + (data[j] - data[i]) * (x - i);
  }

  /** Quadratic (3-point) refinement of an extremum at sample i: [time, value]. */
  _refine(data, fs, i) {
    const y0 = data[i - 1], y1 = data[i], y2 = data[i + 1];
    const den = y0 - 2 * y1 + y2;
    if (Math.abs(den) < 1e-12) return [i / fs, y1];
    const dx = Math.max(-1, Math.min(1, (0.5 * (y0 - y2)) / den));
    return [(i + dx) / fs, y1 - 0.25 * (y0 - y2) * dx];
  }

  _bind() {
    const c = this.canvas;
    const pos = (e) => {
      const r = c.getBoundingClientRect();
      return { x: e.clientX - r.left, y: e.clientY - r.top };
    };
    c.addEventListener('wheel', (e) => {
      if (!this.view) return;
      e.preventDefault();
      const { x } = pos(e);
      if (Math.abs(e.deltaX) > Math.abs(e.deltaY) || e.shiftKey) {
        const span = this.t1 - this.t0;
        const dt = ((e.deltaX || e.deltaY) / this.plotW()) * span;
        const a = Math.min(Math.max(0, this.t0 + dt), this.duration - span);
        this.zoomTo(a, a + span);
      } else {
        const f = Math.exp(e.deltaY * 0.0022);
        this.zoomBy(f, Math.min(Math.max(this.tAt(x), 0), this.duration));
      }
    }, { passive: false });

    c.addEventListener('mousedown', (e) => {
      if (!this.view || e.button !== 0) return;
      const { x, y } = pos(e);
      if (x < this.gutter || y > this.axisY) return;
      const t = this.tAt(x);
      const marker = this._hitMarker(x, y);
      const inPanel = !!this._panelAt(y);
      const boundary = !marker ? this._hitBoundary(x) : null;
      if (marker) {
        this.drag = { type: 'landmark', sensor: marker.sensor, panelSensor: marker.sensor, sig: marker.sig, data: marker.data, fs: marker.fs, orig: { ...marker.peak }, t: marker.peak.time_s, v: marker.peak.value, moved: false, snapped: false, x0: x };
      } else if (boundary && !(inPanel && e.altKey)) {
        this.drag = { type: 'boundary', key: boundary, t: this.occ().occurrence[boundary], moved: false };
      } else if (e.shiftKey && !this.opts.readOnly) {
        this.drag = { type: 'select', a: t };
        this.selection = [t, t];
      } else {
        this.drag = { type: 'pan', x0: x, t0: this.t0, t1: this.t1, moved: false, tClick: t, y };
      }
      e.preventDefault();
    });

    this._onMove = (e) => {
      if (!this.view) return;
      const { x, y } = pos(e);
      const d = this.drag;
      if (d) {
        const t = Math.min(Math.max(this.tAt(x), 0), this.duration);
        if (d.type === 'landmark') {
          if (Math.abs(x - d.x0) > 1) d.moved = true;
          const i = e.altKey ? null : this._snapIndex(d.data, d.fs, t);
          if (i != null) {
            [d.t, d.v] = this._refine(d.data, d.fs, i);
            d.snapped = true;
          } else {
            d.t = t;
            d.v = this._valueAt(d.data, d.fs, t);
            d.snapped = false;
          }
          this.opts.onHint?.('landmark-drag');
        } else if (d.type === 'boundary') {
          const o = this.occ().occurrence;
          d.moved = true;
          d.t = d.key === 'start_s' ? Math.min(t, o.end_s - 0.01) : Math.max(t, o.start_s + 0.01);
        } else if (d.type === 'select') {
          this.selection = [d.a, t];
        } else if (d.type === 'pan') {
          const dx = x - d.x0;
          if (Math.abs(dx) > 3) d.moved = true;
          if (d.moved) {
            const span = d.t1 - d.t0;
            let a = d.t0 - (dx / this.plotW()) * span;
            a = Math.min(Math.max(0, a), this.duration - span);
            this.t0 = a;
            this.t1 = a + span;
            c.style.cursor = 'grabbing';
          }
        }
        this.render();
        return;
      }
      if (e.target === c && x >= this.gutter && y <= this.axisY) {
        const marker = this._hitMarker(x, y);
        const boundary = marker ? null : this._hitBoundary(x);
        this.hover = { t: this.tAt(x), marker, boundary };
        c.style.cursor = marker || boundary ? 'ew-resize' : 'crosshair';
        this.opts.onHint?.(marker ? 'landmark' : boundary ? 'boundary' : 'signal');
        this.render();
      } else if (this.hover) {
        this.hover = null;
        c.style.cursor = 'default';
        this.opts.onHint?.(null);
        this.render();
      }
    };
    window.addEventListener('mousemove', this._onMove);

    this._onUp = (e) => {
      const d = this.drag;
      if (!d) return;
      this.drag = null;
      c.style.cursor = 'crosshair';
      if (d.type === 'landmark' && d.moved) {
        const occ = this.occ();
        const kin = occ?.kinematics.find((k) => k.sensor === d.sensor && k.dim === dimKey(this.display.dim));
        const key = SIG[d.sig].key;
        // show the new position at once; the backend's answer replaces it
        if (kin && kin[key]) kin[key] = { ...kin[key], time_s: d.t, value: d.v, manual: true, at_edge: false };
        this.opts.onLandmarkChange?.({ sensor: d.sensor, dim: dimKey(this.display.dim), kind: key, time_s: d.t, snap_s: d.snapped ? 2.5 / d.fs : 0 });
      } else if (d.type === 'boundary' && d.moved) {
        const o = this.occ().occurrence;
        o[d.key] = d.t;
        this.opts.onSegmentChange?.(this.seg, o.start_s, o.end_s);
      } else if (d.type === 'select') {
        const [a, b] = this.selection;
        if (Math.abs(b - a) > 0.02) this.opts.onSelectRange?.(Math.min(a, b), Math.max(a, b));
        else this.selection = null;
      } else if (d.type === 'pan' && !d.moved) {
        // a click inside another segment makes it the active one
        const occs = this.result?.occurrences || [];
        const k = occs.findIndex((o) => d.tClick >= o.occurrence.start_s && d.tClick <= o.occurrence.end_s);
        if (k >= 0 && k !== this.seg) this.opts.onSegmentSelect?.(k);
        if (this.selection) { this.selection = null; this.opts.onSelectRange?.(null); }
      }
      this.render();
    };
    window.addEventListener('mouseup', this._onUp);

    c.addEventListener('dblclick', (e) => {
      const { x, y } = pos(e);
      const marker = this._hitMarker(x, y);
      if (marker) {
        if (marker.peak.manual) this.opts.onLandmarkReset?.({ sensor: marker.sensor, dim: dimKey(this.display.dim), kind: SIG[marker.sig].key });
        return;
      }
      if (this.opts.onZoomToggle) this.opts.onZoomToggle();
      else this.toggleZoom();
    });
    c.addEventListener('mouseleave', () => {
      if (!this.drag) {
        this.hover = null;
        this.opts.onHint?.(null);
        this.render();
      }
    });
  }

  clearSelection() {
    this.selection = null;
    this.render();
  }

  destroy() {
    this.stop();
    this.ro.disconnect();
    window.removeEventListener('mousemove', this._onMove);
    window.removeEventListener('mouseup', this._onUp);
    this.host.innerHTML = '';
  }
}
