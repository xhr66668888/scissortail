// Backend bridge. Inside the desktop app this calls the Rust commands via
// Tauri; in a plain browser (interface development) it serves fixtures
// recorded from the real pipeline (`scissortail-app --dump-fixtures`).
'use strict';

const API = (() => {
  const tauri = window.__TAURI__;
  const listeners = {};

  async function invoke(cmd, args = {}) {
    if (tauri) return tauri.core.invoke(cmd, args);
    return Mock.invoke(cmd, args);
  }

  function on(event, fn) {
    if (tauri) return tauri.event.listen(event, (e) => fn(e.payload));
    (listeners[event] = listeners[event] || []).push(fn);
    return Promise.resolve(() => {});
  }

  function emitLocal(event, payload) {
    (listeners[event] || []).forEach((fn) => fn(payload));
  }

  // ------------------------------------------------- browser-only fixtures
  const STATUS = { ok: 'ok', low_confidence: 'check', not_found: 'not_found', error: 'error', excluded: 'excluded' };
  const clone = (o) => JSON.parse(JSON.stringify(o));
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  const Mock = {
    base: new URLSearchParams(location.search).get('fixtures') || 'dev-fixtures/',
    ws: null,
    views: new Map(),
    results: new Map(),
    originals: new Map(),
    async json(name) {
      const r = await fetch(this.base + name);
      if (!r.ok) throw new Error(`fixture ${name} missing`);
      return r.json();
    },
    id(file) {
      return file.split(/[\\/]/).pop().replace(/\.mat$/i, '');
    },
    async workspace() {
      if (!this.ws) {
        this.ws = await this.json('workspace.json');
        this.all = clone(this.ws.files);
        if (new URLSearchParams(location.search).has('empty')) this.ws.files = [];
      }
      return this.ws;
    },
    summary(file, r) {
      const f = this.all.find((x) => x.file === file);
      return {
        ...f,
        status: r ? STATUS[r.status] : 'pending',
        word: r?.occurrences?.[0]?.occurrence.word ?? null,
        segments: r?.occurrences?.length || 0,
        edited: !!r?.occurrences?.some((o) => o.edited),
        reviewed: !!r?.reviewed,
        message: r?.message || '',
      };
    },
    view(file) {
      return this.views.get(file);
    },
    async result(file) {
      if (this.results.has(file)) return this.results.get(file);
      const v = this.views.get(file) || (await this.invoke('load_trial', { file }));
      let r = v.result;
      if (!r) r = (await this.json('analysis_default.json')).result;
      this.results.set(file, clone(r));
      this.originals.set(file, clone(r));
      return this.results.get(file);
    },
    answer(file) {
      const r = this.results.get(file);
      const sum = this.summary(file, r);
      const i = this.ws.files.findIndex((x) => x.file === file);
      if (i >= 0) this.ws.files[i] = sum;
      return { result: clone(r), summary: sum };
    },
    valueAt(file, sensor, dim, kind, t) {
      const s = this.view(file).sensors.find((x) => x.name === sensor);
      const sig = { position: 'pos', velocity: 'vel', acceleration: 'acc' }[kind];
      const data = s.dims[dim === 'tangential' ? '3d' : dim][sig];
      const x = Math.min(Math.max(t * s.fs, 0), data.length - 1);
      const i = Math.floor(x), j = Math.min(i + 1, data.length - 1);
      return data[i] + (data[j] - data[i]) * (x - i);
    },
    async invoke(cmd, a) {
      switch (cmd) {
        case 'app_info':
          return { version: '0.2.0 (preview)', platform: 'browser', models_ready: true, aligner: true, models_dir: '(fixtures)' };
        case 'get_workspace':
          return clone(await this.workspace());
        case 'pick_files':
        case 'pick_folder':
          return cmd === 'pick_files' ? ['(picked)'] : '(picked)';
        case 'add_paths': {
          const ws = await this.workspace();
          const known = new Set(ws.files.map((f) => f.file));
          const add = this.all.filter((f) => !known.has(f.file));
          ws.files.push(...add);
          return { workspace: clone(ws), added: add.length, first: (add[0] || ws.files[0]).file };
        }
        case 'remove_files': {
          const ws = await this.workspace();
          const drop = new Set(a.files);
          ws.files = ws.files.filter((f) => !drop.has(f.file));
          return clone(ws);
        }
        case 'load_trial': {
          let v = this.views.get(a.file);
          if (!v) {
            v = await this.json(`trial_${this.id(a.file)}.json`);
            this.views.set(a.file, v);
          }
          // a copy, as over IPC: the interface edits its result in place
          return { ...v, result: clone(this.results.has(a.file) ? this.results.get(a.file) : v.result) };
        }
        case 'analyze_trial': {
          if (!this.results.has(a.file)) await sleep(500);
          await this.result(a.file);
          return this.answer(a.file);
        }
        case 'reanalyze': {
          await this.result(a.file);
          await sleep(400);
          this.results.set(a.file, clone(this.originals.get(a.file)));
          return this.answer(a.file);
        }
        case 'update_segment': {
          const r = await this.result(a.file);
          const o = r.occurrences[a.index];
          o.occurrence.start_s = a.startS;
          o.occurrence.end_s = a.endS;
          o.edited = true;
          return this.answer(a.file);
        }
        case 'add_segment': {
          const r = await this.result(a.file);
          const kin = clone(r.occurrences[0]?.kinematics || []);
          r.occurrences.push({ occurrence: { word_index: 0, word: r.occurrences[0]?.occurrence.word || 'speech', start_s: a.startS, end_s: a.endS, raw_start_s: a.startS, raw_end_s: a.endS, phrase_start_s: a.startS, phrase_end_s: a.endS, confidence: 1, phones: [], engine: 'speech' }, kinematics: kin, edited: true });
          r.occurrences.sort((x, y) => x.occurrence.start_s - y.occurrence.start_s);
          return this.answer(a.file);
        }
        case 'delete_segment': {
          const r = await this.result(a.file);
          r.occurrences.splice(a.index, 1);
          return this.answer(a.file);
        }
        case 'set_landmark': {
          const r = await this.result(a.file);
          const o = r.occurrences[a.index];
          const k = o.kinematics.find((x) => x.sensor === a.sensor && x.dim === a.dim);
          k[a.kind] = { time_s: a.timeS, value: this.valueAt(a.file, a.sensor, a.dim, a.kind, a.timeS), at_edge: false, manual: true };
          o.edited = true;
          return this.answer(a.file);
        }
        case 'reset_landmark': {
          const r = await this.result(a.file);
          const orig = this.originals.get(a.file).occurrences[a.index]?.kinematics.find((x) => x.sensor === a.sensor && x.dim === a.dim);
          const k = r.occurrences[a.index].kinematics.find((x) => x.sensor === a.sensor && x.dim === a.dim);
          if (orig && k) k[a.kind] = clone(orig[a.kind]);
          return this.answer(a.file);
        }
        case 'set_reviewed': {
          const r = await this.result(a.file);
          r.reviewed = a.reviewed;
          return this.answer(a.file);
        }
        case 'set_config': {
          const ws = await this.workspace();
          ws.config = a.config;
          return clone(ws);
        }
        case 'default_settings':
          return clone((await this.workspace()).config);
        case 'export_csv':
          return { path: '/Users/demo/Desktop/scissortail_landmarks_20261103-120000.csv', files: (await this.workspace()).files.length, skipped: 0 };
        default:
          return null;
      }
    },
  };

  return { invoke, on, emitLocal, isDesktop: !!tauri };
})();
