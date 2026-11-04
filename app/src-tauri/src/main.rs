// Scissortail desktop application (Tauri 2 backend).
//
// One window in the spirit of MAVIS / MVIEW: the researcher drops trial files
// in, each one is segmented and labelled automatically in the background, and
// every boundary and landmark can then be dragged by hand.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod state;

use std::cmp::Ordering as CmpOrdering;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use base64::Engine as _;
use rayon::prelude::*;
use serde_json::{json, Value};
use tauri::{Emitter, Manager, State};
use tauri_plugin_dialog::DialogExt;

use scissortail_core::analysis::{has_target_words, kinematics_for, refresh_kinematics, AnalysisConfig, Analyzer, OccurrenceResult, TrialResult, TrialStatus};
use scissortail_core::dsp::{resample, spectrogram_u8};
use scissortail_core::export::landmarks_csv;
use scissortail_core::kinematics::{derivatives, manual_peak, Dim, KinematicsConfig, Landmark};
use scissortail_core::locate::{candidate_words, Engine, Occurrence};
use scissortail_core::trial::Trial;

use state::{engines, AppState, Foreground, Workspace};

type CmdResult<T> = Result<T, String>;

fn err<E: std::fmt::Display>(e: E) -> String {
    e.to_string()
}

fn b64(bytes: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

fn round(v: f64, digits: i32) -> f64 {
    let p = 10f64.powi(digits);
    (v * p).round() / p
}

// ------------------------------------------------------------------ info --

#[tauri::command]
fn app_info(state: State<'_, AppState>) -> Value {
    let e = state::engines_if_ready();
    json!({
        "version": env!("CARGO_PKG_VERSION"),
        "platform": std::env::consts::OS,
        "models_ready": e.is_some(),
        "aligner": e.map(|e| e.aligner.is_some()),
        "models_dir": state.models_dir.display().to_string(),
    })
}

// ------------------------------------------------------------- workspace --

fn status_key(r: Option<&TrialResult>) -> &'static str {
    match r.map(|r| r.status) {
        None => "pending",
        Some(TrialStatus::Ok) => "ok",
        Some(TrialStatus::LowConfidence) => "check",
        Some(TrialStatus::NotFound) => "not_found",
        Some(TrialStatus::Error) => "error",
        Some(TrialStatus::Excluded) => "excluded",
    }
}

/// One row of the file list.
fn file_summary(file: &str, r: Option<&TrialResult>) -> Value {
    let p = Path::new(file);
    let name = |s: Option<&std::ffi::OsStr>| s.map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
    json!({
        "file": file,
        "name": name(p.file_stem()),
        "folder": name(p.parent().and_then(|d| d.file_name())),
        "status": status_key(r),
        "word": r.and_then(|r| r.occurrences.first()).map(|o| o.occurrence.word.clone()),
        "segments": r.map(|r| r.occurrences.len()).unwrap_or(0),
        "edited": r.map(|r| r.occurrences.iter().any(|o| o.edited)).unwrap_or(false),
        "reviewed": r.map(|r| r.reviewed).unwrap_or(false),
        "message": r.map(|r| r.message.clone()).unwrap_or_default(),
    })
}

fn workspace_view(ws: &Workspace) -> Value {
    json!({
        "files": ws.files.iter().map(|f| file_summary(f, ws.results.get(f))).collect::<Vec<_>>(),
        "config": ws.config,
    })
}

/// Natural order: "T2" < "T10".
fn natural_cmp(a: &str, b: &str) -> CmpOrdering {
    let (mut x, mut y) = (a.as_bytes(), b.as_bytes());
    loop {
        match (x.first(), y.first()) {
            (None, None) => return CmpOrdering::Equal,
            (None, _) => return CmpOrdering::Less,
            (_, None) => return CmpOrdering::Greater,
            (Some(c), Some(d)) if c.is_ascii_digit() && d.is_ascii_digit() => {
                let i = x.iter().take_while(|c| c.is_ascii_digit()).count();
                let j = y.iter().take_while(|c| c.is_ascii_digit()).count();
                let (nx, ny) = (std::str::from_utf8(&x[..i]).unwrap().trim_start_matches('0'), std::str::from_utf8(&y[..j]).unwrap().trim_start_matches('0'));
                let o = nx.len().cmp(&ny.len()).then_with(|| nx.cmp(ny));
                if o != CmpOrdering::Equal {
                    return o;
                }
                x = &x[i..];
                y = &y[j..];
            }
            (Some(c), Some(d)) => {
                let o = c.to_ascii_lowercase().cmp(&d.to_ascii_lowercase());
                if o != CmpOrdering::Equal {
                    return o;
                }
                x = &x[1..];
                y = &y[1..];
            }
        }
    }
}

/// Trial files under the dropped paths: `.mat` files as they are, folders
/// searched recursively (hidden, build and package folders skipped).
fn expand_paths(paths: &[String]) -> Vec<String> {
    const MAX_FILES: usize = 50_000;
    const MAX_DEPTH: usize = 6;
    let skip = |name: &str| name.starts_with('.') || name.ends_with(".app") || matches!(name, "node_modules" | "target" | "__pycache__" | "Library");
    let mut out = Vec::new();
    let mut stack: Vec<(PathBuf, usize)> = paths.iter().rev().map(|p| (PathBuf::from(p), 0)).collect();
    while let Some((p, depth)) = stack.pop() {
        if out.len() >= MAX_FILES {
            break;
        }
        if p.is_dir() {
            if depth > MAX_DEPTH {
                continue;
            }
            let mut entries: Vec<PathBuf> = std::fs::read_dir(&p)
                .map(|rd| rd.filter_map(|e| e.ok()).map(|e| e.path()).collect())
                .unwrap_or_default();
            entries.retain(|e| !e.file_name().map(|n| skip(&n.to_string_lossy())).unwrap_or(true));
            entries.sort_by(|a, b| natural_cmp(&a.to_string_lossy(), &b.to_string_lossy()));
            stack.extend(entries.into_iter().rev().map(|e| (e, depth + 1)));
        } else if p.extension().map(|e| e.eq_ignore_ascii_case("mat")).unwrap_or(false) && p.is_file() {
            out.push(p.display().to_string());
        }
    }
    out
}

/// Shows a native file dialog. Dialogs run on the main thread, so the
/// command itself must not: synchronous Tauri commands execute on the main
/// thread, and waiting there for the dialog's answer deadlocks the window.
async fn on_dialog_thread<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> CmdResult<T> {
    tauri::async_runtime::spawn_blocking(f).await.map_err(err)
}

#[tauri::command]
async fn pick_files(app: tauri::AppHandle) -> CmdResult<Vec<String>> {
    let picked = on_dialog_thread(move || app.dialog().file().set_title("Open EMA trial files").add_filter("MATLAB data (.mat)", &["mat"]).blocking_pick_files()).await?;
    Ok(picked.unwrap_or_default().into_iter().filter_map(|p| p.into_path().ok()).map(|p| p.display().to_string()).collect())
}

#[tauri::command]
async fn pick_folder(app: tauri::AppHandle) -> CmdResult<Option<String>> {
    let picked = on_dialog_thread(move || app.dialog().file().set_title("Open a folder of EMA trials").blocking_pick_folder()).await?;
    Ok(picked.and_then(|p| p.into_path().ok()).map(|p| p.display().to_string()))
}

/// Adds dropped / picked files and folders to the workspace and starts the
/// background analysis. `first` is the first trial among `paths` (also when
/// it was already in the list), so the interface can open it.
#[tauri::command]
async fn add_paths(paths: Vec<String>, app: tauri::AppHandle) -> CmdResult<Value> {
    let app2 = app.clone();
    let out = tauri::async_runtime::spawn_blocking(move || {
        let state = app.state::<AppState>();
        let files = expand_paths(&paths);
        if files.is_empty() {
            return Err("no .mat files found".to_string());
        }
        let first = files[0].clone();
        let (view, added) = {
            let mut ws = state.ws.lock().unwrap();
            let known: HashSet<String> = ws.files.iter().cloned().collect();
            let new: Vec<String> = files.into_iter().filter(|f| !known.contains(f)).collect();
            let added = new.len();
            ws.files.extend(new);
            (workspace_view(&ws), added)
        };
        state.save();
        Ok(json!({ "workspace": view, "added": added, "first": first }))
    })
    .await
    .map_err(err)??;
    ensure_worker(&app2);
    Ok(out)
}

#[tauri::command]
fn get_workspace(app: tauri::AppHandle) -> Value {
    let view = workspace_view(&app.state::<AppState>().ws.lock().unwrap());
    ensure_worker(&app);
    view
}

#[tauri::command]
async fn remove_files(files: Vec<String>, app: tauri::AppHandle) -> CmdResult<Value> {
    tauri::async_runtime::spawn_blocking(move || {
        let state = app.state::<AppState>();
        let view = {
            let mut ws = state.ws.lock().unwrap();
            let drop: HashSet<&String> = files.iter().collect();
            ws.files.retain(|f| !drop.contains(f));
            for f in &files {
                ws.results.remove(f);
            }
            workspace_view(&ws)
        };
        state.save();
        view
    })
    .await
    .map_err(err)
}

// -------------------------------------------------------------- analysis --

/// The analyzer for the current settings (the app has no TTS, and without
/// the speech model trials are segmented by speech activity).
fn analyzer_for(state: &AppState) -> CmdResult<Arc<Analyzer<'static>>> {
    let mut cfg = state.ws.lock().unwrap().config.clone();
    let e = engines()?;
    if cfg.locate.engine == Engine::Template || e.aligner.is_none() {
        cfg.locate.engine = if e.aligner.is_some() { Engine::Phonetic } else { Engine::Speech };
    }
    let key = serde_json::to_string(&cfg.locate).unwrap_or_default();
    if let Some((k, a)) = state.analyzer_cache.lock().unwrap().as_ref() {
        if *k == key {
            let mut a2 = (**a).clone();
            a2.config = cfg;
            return Ok(Arc::new(a2));
        }
    }
    let a = Arc::new(Analyzer::new(e, cfg).map_err(err)?);
    *state.analyzer_cache.lock().unwrap() = Some((key, a.clone()));
    Ok(a)
}

/// Analyses one file right away (ahead of the background queue) and stores
/// the result unless one appeared meanwhile.
fn analyze_now(state: &AppState, file: &str) -> CmdResult<TrialResult> {
    let _fg = Foreground::enter(&state.foreground);
    let gen = state.generation.load(Ordering::SeqCst);
    let analyzer = analyzer_for(state)?;
    let r = analyzer.analyze_file(Path::new(file));
    let mut ws = state.ws.lock().unwrap();
    if state.generation.load(Ordering::SeqCst) != gen || !ws.files.iter().any(|f| f == file) {
        return Ok(r);
    }
    Ok(ws.results.entry(file.to_string()).or_insert(r).clone())
}

/// Starts the background worker unless it is running: it analyses every file
/// without a result, two at a time, pausing while the interface waits for one.
fn ensure_worker(app: &tauri::AppHandle) {
    let state = app.state::<AppState>();
    if state.worker_running.swap(true, Ordering::SeqCst) {
        return;
    }
    let app = app.clone();
    std::thread::spawn(move || {
        let state = app.state::<AppState>();
        let pool = rayon::ThreadPoolBuilder::new().num_threads(2).build().ok();
        loop {
            let ok = run_pending(&app, &state, pool.as_ref());
            state.worker_running.store(false, Ordering::SeqCst);
            let more = !state.ws.lock().unwrap().pending().is_empty();
            if !ok || !more || state.worker_running.swap(true, Ordering::SeqCst) {
                break;
            }
        }
    });
}

/// Analyses pending files until none is left; false when no analyzer could
/// be built (reported to the interface once).
fn run_pending(app: &tauri::AppHandle, state: &AppState, pool: Option<&rayon::ThreadPool>) -> bool {
    loop {
        let gen = state.generation.load(Ordering::SeqCst);
        let pending = state.ws.lock().unwrap().pending();
        let total = state.ws.lock().unwrap().files.len();
        let _ = app.emit("analysis-progress", json!({ "pending": pending.len(), "total": total }));
        if pending.is_empty() {
            return true;
        }
        let analyzer = match analyzer_for(state) {
            Ok(a) => a,
            Err(e) => {
                let _ = app.emit("analysis-error", e);
                return false;
            }
        };
        let left = AtomicUsize::new(pending.len());
        let work = || {
            pending.par_iter().for_each(|f| {
                while state.foreground.load(Ordering::SeqCst) > 0 {
                    std::thread::sleep(std::time::Duration::from_millis(15));
                }
                if state.generation.load(Ordering::SeqCst) != gen || state.ws.lock().unwrap().results.contains_key(f) {
                    return;
                }
                let r = analyzer.analyze_file(Path::new(f));
                let summary = {
                    let mut ws = state.ws.lock().unwrap();
                    if state.generation.load(Ordering::SeqCst) != gen || !ws.files.iter().any(|x| x == f) {
                        return;
                    }
                    let r = ws.results.entry(f.clone()).or_insert(r);
                    file_summary(f, Some(r))
                };
                let n = left.fetch_sub(1, Ordering::SeqCst) - 1;
                let _ = app.emit("trial-updated", summary);
                let _ = app.emit("analysis-progress", json!({ "pending": n, "total": total }));
                if n % 50 == 0 {
                    state.save();
                }
            })
        };
        match pool {
            Some(p) => p.install(work),
            None => work(),
        }
        state.save();
    }
}

// ----------------------------------------------------------------- trials --

fn signals_json(trial: &Trial, cfg: &KinematicsConfig) -> Vec<Value> {
    trial
        .sensors
        .iter()
        .map(|s| {
            let mut dims = serde_json::Map::new();
            for dim in Dim::all() {
                if let Some(d) = derivatives(s, dim, cfg) {
                    let r = |v: &[f64], k: i32| v.iter().map(|&x| round(x, k)).collect::<Vec<f64>>();
                    dims.insert(dim.label().to_string(), json!({ "pos": r(&d.pos, 3), "vel": r(&d.vel, 1), "acc": r(&d.acc, 0) }));
                }
            }
            json!({ "name": s.name, "fs": s.fs, "n": s.len(), "is_reference": s.looks_like_reference(), "dims": dims })
        })
        .collect()
}

fn trial_view_json(file: &str, cfg: &KinematicsConfig, result: Option<&TrialResult>) -> CmdResult<Value> {
    let trial = Trial::load(Path::new(file)).map_err(err)?;
    let fs_play = 24000.0;
    let a24 = resample(&trial.audio.samples, trial.audio.fs, fs_play);
    let peak = a24.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
    let pcm: Vec<u8> = a24.iter().flat_map(|&s| ((s / peak * 0.95 * 32767.0) as i16).to_le_bytes()).collect();
    let dur = trial.audio.duration();
    let hop = (dur / 1800.0).max(0.002);
    let a16 = resample(&trial.audio.samples, trial.audio.fs, 16000.0);
    let (pix, w, h, spc, fmax) = spectrogram_u8(&a16, 16000.0, 0.005, hop, 8000.0, 160, 65.0);
    Ok(json!({
        "id": trial.id, "file": file, "duration": dur,
        "audio": { "fs": fs_play, "pcm16": b64(&pcm) },
        "spec": { "w": w, "h": h, "data": b64(&pix), "sec_per_col": spc, "fmax": fmax },
        "sensors": signals_json(&trial, cfg),
        "result": result,
    }))
}

/// Signals of one file plus its result when it has been analysed already
/// (otherwise the interface asks for `analyze_trial`).
#[tauri::command]
async fn load_trial(file: String, app: tauri::AppHandle) -> CmdResult<Value> {
    tauri::async_runtime::spawn_blocking(move || {
        let state = app.state::<AppState>();
        let (cfg, result) = {
            let ws = state.ws.lock().unwrap();
            (ws.config.kinematics.clone(), ws.results.get(&file).cloned())
        };
        trial_view_json(&file, &cfg, result.as_ref())
    })
    .await
    .map_err(err)?
}

fn result_json(file: &str, r: &TrialResult) -> Value {
    json!({ "result": r, "summary": file_summary(file, Some(r)) })
}

#[tauri::command]
async fn analyze_trial(file: String, app: tauri::AppHandle) -> CmdResult<Value> {
    tauri::async_runtime::spawn_blocking(move || {
        let state = app.state::<AppState>();
        let existing = state.ws.lock().unwrap().results.get(&file).cloned();
        let r = match existing {
            Some(r) => r,
            None => analyze_now(&state, &file)?,
        };
        state.save();
        Ok(result_json(&file, &r))
    })
    .await
    .map_err(err)?
}

/// Discards every edit of a file and analyses it again.
#[tauri::command]
async fn reanalyze(file: String, app: tauri::AppHandle) -> CmdResult<Value> {
    tauri::async_runtime::spawn_blocking(move || {
        let state = app.state::<AppState>();
        state.ws.lock().unwrap().results.remove(&file);
        let r = analyze_now(&state, &file)?;
        state.save();
        Ok(result_json(&file, &r))
    })
    .await
    .map_err(err)?
}

// ------------------------------------------------------------------ edits --

/// Applies an edit to the stored result of `file` and saves the workspace.
fn edit<F: FnOnce(&mut TrialResult) -> CmdResult<()>>(state: &AppState, file: &str, f: F) -> CmdResult<Value> {
    let out = {
        let mut ws = state.ws.lock().unwrap();
        let r = ws.results.get_mut(file).ok_or("this file has not been analysed yet")?;
        f(r)?;
        result_json(file, r)
    };
    state.save();
    Ok(out)
}

fn kinematics_cfg(state: &AppState) -> KinematicsConfig {
    state.ws.lock().unwrap().config.kinematics.clone()
}

#[tauri::command]
async fn update_segment(file: String, index: usize, start_s: f64, end_s: f64, app: tauri::AppHandle) -> CmdResult<Value> {
    tauri::async_runtime::spawn_blocking(move || {
        let state = app.state::<AppState>();
        if end_s <= start_s + 0.01 {
            return Err("the offset must come after the onset".into());
        }
        let cfg = kinematics_cfg(&state);
        let trial = Trial::load(Path::new(&file)).map_err(err)?;
        edit(&state, &file, |r| {
            let o = r.occurrences.get_mut(index).ok_or("no such segment")?;
            o.occurrence.start_s = start_s;
            o.occurrence.end_s = end_s;
            if let Some(p) = o.occurrence.phones.first_mut() {
                p.start_s = start_s;
            }
            if let Some(p) = o.occurrence.phones.last_mut() {
                p.end_s = end_s;
            }
            refresh_kinematics(&trial, o, &cfg);
            o.edited = true;
            if r.status == TrialStatus::NotFound {
                r.status = TrialStatus::Ok;
            }
            Ok(())
        })
    })
    .await
    .map_err(err)?
}

/// A segment marked by hand (Shift-drag over the signals).
#[tauri::command]
async fn add_segment(file: String, start_s: f64, end_s: f64, app: tauri::AppHandle) -> CmdResult<Value> {
    tauri::async_runtime::spawn_blocking(move || {
        let state = app.state::<AppState>();
        if end_s <= start_s + 0.01 {
            return Err("the segment is too short".into());
        }
        let (cfg, words) = {
            let ws = state.ws.lock().unwrap();
            (ws.config.kinematics.clone(), ws.config.locate.clone())
        };
        let trial = Trial::load(Path::new(&file)).map_err(err)?;
        let file_name = Path::new(&file).file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
        let word = if has_target_words(&words) {
            candidate_words(&words.words, &file_name).first().map(|&i| words.words[i].display()).unwrap_or_default()
        } else {
            "speech".to_string()
        };
        let occ = Occurrence {
            word_index: 0,
            word,
            start_s,
            end_s,
            raw_start_s: start_s,
            raw_end_s: end_s,
            phrase_start_s: start_s,
            phrase_end_s: end_s,
            confidence: 1.0,
            phones: vec![],
            engine: Engine::Speech,
        };
        let kinematics = kinematics_for(&trial, &occ, &cfg);
        edit(&state, &file, |r| {
            r.occurrences.push(OccurrenceResult { occurrence: occ, kinematics, edited: true });
            r.occurrences.sort_by(|a, b| a.occurrence.start_s.partial_cmp(&b.occurrence.start_s).unwrap());
            if matches!(r.status, TrialStatus::NotFound | TrialStatus::Error) {
                r.status = TrialStatus::Ok;
                r.message.clear();
            }
            Ok(())
        })
    })
    .await
    .map_err(err)?
}

#[tauri::command]
async fn delete_segment(file: String, index: usize, app: tauri::AppHandle) -> CmdResult<Value> {
    tauri::async_runtime::spawn_blocking(move || {
        edit(&app.state::<AppState>(), &file, |r| {
            if index < r.occurrences.len() {
                r.occurrences.remove(index);
            }
            if r.occurrences.is_empty() {
                r.status = TrialStatus::NotFound;
            }
            Ok(())
        })
    })
    .await
    .map_err(err)?
}

/// Places a landmark by hand. With `snap_s` > 0 it snaps to the nearest
/// local peak of its signal within ±`snap_s` seconds.
#[tauri::command]
#[allow(clippy::too_many_arguments)]
async fn set_landmark(file: String, index: usize, sensor: String, dim: Dim, kind: Landmark, time_s: f64, snap_s: f64, app: tauri::AppHandle) -> CmdResult<Value> {
    tauri::async_runtime::spawn_blocking(move || {
        let state = app.state::<AppState>();
        let cfg = kinematics_cfg(&state);
        let trial = Trial::load(Path::new(&file)).map_err(err)?;
        let s = trial.sensor(&sensor).ok_or("no such sensor")?;
        let peak = manual_peak(s, dim, kind, time_s, snap_s.max(0.0), &cfg).ok_or("this signal is too short")?;
        edit(&state, &file, |r| {
            let o = r.occurrences.get_mut(index).ok_or("no such segment")?;
            let k = o.kinematics.iter_mut().find(|k| k.sensor.eq_ignore_ascii_case(&sensor) && k.dim == dim).ok_or("this sensor dimension is not analysed")?;
            *k.peak_mut(kind) = Some(peak);
            o.edited = true;
            Ok(())
        })
    })
    .await
    .map_err(err)?
}

/// Puts a hand-placed landmark back where the automatic detection has it.
#[tauri::command]
async fn reset_landmark(file: String, index: usize, sensor: String, dim: Dim, kind: Landmark, app: tauri::AppHandle) -> CmdResult<Value> {
    tauri::async_runtime::spawn_blocking(move || {
        let state = app.state::<AppState>();
        let cfg = kinematics_cfg(&state);
        let trial = Trial::load(Path::new(&file)).map_err(err)?;
        edit(&state, &file, |r| {
            let o = r.occurrences.get_mut(index).ok_or("no such segment")?;
            let auto = kinematics_for(&trial, &o.occurrence, &cfg)
                .into_iter()
                .find(|k| k.sensor.eq_ignore_ascii_case(&sensor) && k.dim == dim)
                .and_then(|k| k.peak(kind).cloned());
            let k = o.kinematics.iter_mut().find(|k| k.sensor.eq_ignore_ascii_case(&sensor) && k.dim == dim).ok_or("this sensor dimension is not analysed")?;
            *k.peak_mut(kind) = auto;
            Ok(())
        })
    })
    .await
    .map_err(err)?
}

#[tauri::command]
async fn set_reviewed(file: String, reviewed: bool, app: tauri::AppHandle) -> CmdResult<Value> {
    tauri::async_runtime::spawn_blocking(move || {
        edit(&app.state::<AppState>(), &file, |r| {
            r.reviewed = reviewed;
            Ok(())
        })
    })
    .await
    .map_err(err)?
}

// --------------------------------------------------------------- settings --

/// New settings. Changed target words re-run the segmentation of every file
/// not edited or reviewed by hand; changed kinematics settings only
/// recompute the landmarks (hand-placed ones stay where they are).
#[tauri::command]
async fn set_config(config: AnalysisConfig, app: tauri::AppHandle) -> CmdResult<Value> {
    let app2 = app.clone();
    let view = tauri::async_runtime::spawn_blocking(move || {
        let state = app.state::<AppState>();
        let (locate_changed, results) = {
            let mut ws = state.ws.lock().unwrap();
            let changed = serde_json::to_string(&ws.config.locate).ok() != serde_json::to_string(&config.locate).ok();
            ws.config = config.clone();
            (changed, ws.results.iter().map(|(f, r)| (f.clone(), r.clone())).collect::<Vec<_>>())
        };
        state.generation.fetch_add(1, Ordering::SeqCst);
        *state.analyzer_cache.lock().unwrap() = None;
        let cfg = config.kinematics;
        let updated: Vec<(String, Option<TrialResult>)> = results
            .into_par_iter()
            .map(|(f, mut r)| {
                let by_hand = r.reviewed || r.occurrences.iter().any(|o| o.edited);
                if locate_changed && !by_hand {
                    return (f, None);
                }
                if let Ok(trial) = Trial::load(Path::new(&f)) {
                    r.occurrences.iter_mut().for_each(|o| refresh_kinematics(&trial, o, &cfg));
                }
                (f, Some(r))
            })
            .collect();
        let view = {
            let mut ws = state.ws.lock().unwrap();
            for (f, r) in updated {
                match r {
                    Some(r) => ws.results.insert(f, r),
                    None => ws.results.remove(&f),
                };
            }
            workspace_view(&ws)
        };
        state.save();
        view
    })
    .await
    .map_err(err)?;
    ensure_worker(&app2);
    Ok(view)
}

#[tauri::command]
fn default_settings() -> AnalysisConfig {
    state::default_config()
}

// ----------------------------------------------------------------- export --

/// Writes the landmarks of every analysed file to one CSV (save dialog).
/// Returns `null` when the dialog is cancelled.
#[tauri::command]
async fn export_csv(app: tauri::AppHandle) -> CmdResult<Value> {
    let (csv, n, skipped, dir) = {
        let state = app.state::<AppState>();
        let ws = state.ws.lock().unwrap();
        let rows: Vec<(String, &TrialResult)> = ws
            .files
            .iter()
            .filter_map(|f| {
                let folder = Path::new(f).parent().and_then(|d| d.file_name()).map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
                ws.results.get(f).map(|r| (folder, r))
            })
            .collect();
        if rows.is_empty() {
            return Err("nothing has been analysed yet".into());
        }
        let refs: Vec<(&str, &TrialResult)> = rows.iter().map(|(s, r)| (s.as_str(), *r)).collect();
        let dir = ws.files.first().and_then(|f| Path::new(f).parent().and_then(|d| d.parent()).map(|d| d.to_path_buf()));
        (landmarks_csv(&refs), rows.len(), ws.files.len() - rows.len(), dir)
    };
    let name = format!("scissortail_landmarks_{}.csv", state::timestamp());
    let picked = on_dialog_thread(move || {
        let mut d = app.dialog().file().set_title("Export landmarks").add_filter("CSV", &["csv"]).set_file_name(name);
        if let Some(dir) = dir {
            d = d.set_directory(dir);
        }
        d.blocking_save_file()
    })
    .await?;
    let Some(path) = picked.and_then(|p| p.into_path().ok()) else { return Ok(Value::Null) };
    std::fs::write(&path, csv).map_err(err)?;
    Ok(json!({ "path": path.display().to_string(), "files": n, "skipped": skipped }))
}

#[tauri::command]
fn reveal(path: String) -> CmdResult<()> {
    #[cfg(target_os = "macos")]
    std::process::Command::new("open").arg("-R").arg(&path).spawn().map_err(err)?;
    // explorer parses its own command line: the path must be quoted after the comma
    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        std::process::Command::new("explorer").raw_arg(format!("/select,\"{path}\"")).spawn().map_err(err)?;
    }
    #[cfg(target_os = "linux")]
    std::process::Command::new("xdg-open").arg(Path::new(&path).parent().unwrap_or(Path::new("."))).spawn().map_err(err)?;
    Ok(())
}

/// Developer aid: `scissortail-app --dump-fixtures <data folder> <out dir>`
/// runs the real pipeline on a few trials and writes the JSON the interface
/// receives, so it can be previewed in an ordinary browser.
fn dump_fixtures(root: &str, out: &str) -> CmdResult<()> {
    let out = PathBuf::from(out);
    std::fs::create_dir_all(&out).map_err(err)?;
    let st = AppState::new(None, std::env::temp_dir().join("Scissortail-fixtures"));
    st.start_loading_engines();
    let files: Vec<String> = expand_paths(&[root.to_string()]);
    let mut by_folder: std::collections::BTreeMap<String, Vec<String>> = Default::default();
    for f in files {
        let folder = Path::new(&f).parent().map(|p| p.display().to_string()).unwrap_or_default();
        by_folder.entry(folder).or_default().push(f);
    }
    let picked: Vec<String> = by_folder.values().take(2).flat_map(|v| v.iter().take(12).cloned()).collect();
    let mut ws = Workspace { files: picked.clone(), ..Default::default() };
    let analyzer = Analyzer::new(engines()?, ws.config.clone()).map_err(err)?;
    let results: Vec<(String, TrialResult)> = picked.par_iter().skip(1).map(|f| (f.clone(), analyzer.analyze_file(Path::new(f)))).collect();
    let write = |name: String, v: &Value| std::fs::write(out.join(name), serde_json::to_string(v).unwrap()).map_err(err);
    // the first file stays unanalysed, to preview the "analysing…" state
    for (f, r) in &results {
        ws.results.insert(f.clone(), r.clone());
    }
    write("workspace.json".into(), &workspace_view(&ws))?;
    let first = analyzer.analyze_file(Path::new(&picked[0]));
    write("analysis_default.json".into(), &result_json(&picked[0], &first))?;
    for f in &picked {
        let id = Path::new(f).file_stem().unwrap().to_string_lossy().to_string();
        write(format!("trial_{id}.json"), &trial_view_json(f, &ws.config.kinematics, ws.results.get(f))?)?;
    }
    println!("wrote fixtures for {} trials to {}", picked.len(), out.display());
    Ok(())
}

/// The program used to be called ArticuMark and kept its workspace in a folder
/// named after the old identifier: on the first start under the new name, copy
/// that workspace (file list, settings, results, hand edits) over. The old
/// folder is left as it is.
fn adopt_articumark_workspace(data: &Path) {
    let new = data.join("workspace.json");
    let old = data.with_file_name("org.articumark.workbench").join("workspace.json");
    if !new.exists() && old.exists() {
        let _ = std::fs::create_dir_all(data);
        let _ = std::fs::copy(old, new);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() >= 4 && args[1] == "--dump-fixtures" {
        if let Err(e) = dump_fixtures(&args[2], &args[3]) {
            eprintln!("{e}");
            std::process::exit(1);
        }
        return;
    }
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            let resource = app.path().resource_dir().ok();
            let data = app.path().app_data_dir().unwrap_or_else(|_| std::env::temp_dir().join("Scissortail"));
            adopt_articumark_workspace(&data);
            let st = AppState::new(resource, data);
            st.start_loading_engines();
            app.manage(st);
            // Small screens, and Windows laptops at 125–150 % scaling, have less
            // room than the default window size: open maximised instead.
            if let Some(w) = app.get_webview_window("main") {
                if let (Ok(Some(m)), Ok(size)) = (w.current_monitor(), w.outer_size()) {
                    let area = m.work_area().size;
                    if size.width > area.width || size.height > area.height {
                        let _ = w.maximize();
                    }
                }
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            app_info,
            pick_files,
            pick_folder,
            add_paths,
            get_workspace,
            remove_files,
            load_trial,
            analyze_trial,
            reanalyze,
            update_segment,
            add_segment,
            delete_segment,
            set_landmark,
            reset_landmark,
            set_reviewed,
            set_config,
            default_settings,
            export_csv,
            reveal,
        ])
        .run(tauri::generate_context!())
        .expect("error while running Scissortail");
}
