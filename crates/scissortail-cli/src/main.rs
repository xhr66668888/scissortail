//! Scissortail command-line interface (batch analysis and diagnostics).
//!
//!   scissortail inspect <trial.mat>
//!   scissortail synth "<ipa>" [--voice af_heart] [--speed 1.0] --out file.wav
//!   scissortail analyze <folder> --word "<ipa>[=label[@file-pattern]]" ... [--carrier "<ipa with *>"]
//!                      [--engine phonetic|template] [--out results.json] [--csv landmarks.csv]
//!                      [--protocol peaks|jaw-cycle|gestures] [--protocol-config p.json] [--measures measures.csv]
//!   scissortail measure <results.json> --protocol jaw-cycle [--protocol-config p.json] --measures m.csv [--out r.json]
//!       (re-computes the protocol on the segments of an earlier `analyze --out`, without re-aligning)

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};
use rayon::prelude::*;

use scissortail_core::analysis::{models_dir_candidates, AnalysisConfig, Analyzer, Engines, TrialResult};
use scissortail_core::locate::{Engine, WordSpec};
use scissortail_core::protocol::{ProtocolConfig, ProtocolKind};
use scissortail_core::session::scan;
use scissortail_core::trial::Trial;

fn models_dir() -> Result<PathBuf> {
    let exe_dir = std::env::current_exe().ok().and_then(|p| p.parent().map(|p| p.to_path_buf()));
    models_dir_candidates(exe_dir)
        .into_iter()
        .find(|d| d.join("mms-fa-int8.onnx").exists() || d.join("kokoro-v1.0-int8-dur.onnx").exists())
        .ok_or_else(|| anyhow!("models folder not found (set SCISSORTAIL_MODELS)"))
}

fn arg_values(args: &[String], flag: &str) -> Vec<String> {
    let mut out = vec![];
    let mut i = 0;
    while i < args.len() {
        if args[i] == flag && i + 1 < args.len() {
            out.push(args[i + 1].clone());
            i += 2;
        } else {
            i += 1;
        }
    }
    out
}

fn arg_value(args: &[String], flag: &str) -> Option<String> {
    arg_values(args, flag).pop()
}

/// Protocol settings from `--protocol` and `--protocol-config` (a JSON
/// `ProtocolConfig`; fields left out keep their defaults).
fn protocol_config(args: &[String]) -> Result<ProtocolConfig> {
    let mut p = match arg_value(args, "--protocol-config") {
        Some(f) => serde_json::from_str(&std::fs::read_to_string(&f).with_context(|| f.clone())?).with_context(|| format!("{f}: not a protocol configuration"))?,
        None => ProtocolConfig::default(),
    };
    if let Some(k) = arg_value(args, "--protocol") {
        p.kind = match k.replace('-', "_").as_str() {
            "peaks" => ProtocolKind::Peaks,
            "jaw_cycle" | "jvc" | "mn23" => ProtocolKind::JawCycle,
            "gestures" | "findgest" => ProtocolKind::Gestures,
            other => bail!("unknown protocol {other} (peaks, jaw-cycle, gestures)"),
        };
    }
    Ok(p)
}

/// `--lowpass <Hz>` and `--sg-window <ms>`: the kinematics filters.
fn kinematics_flags(args: &[String], cfg: &mut AnalysisConfig) -> Result<()> {
    if let Some(v) = arg_value(args, "--lowpass") {
        cfg.kinematics.lowpass_hz = v.parse()?;
    }
    if let Some(v) = arg_value(args, "--sg-window") {
        cfg.kinematics.sg_window_ms = v.parse()?;
    }
    Ok(())
}

/// The review across trials (outliers among the repetitions of a condition).
fn review(results: &mut [(String, TrialResult)], cfg: &AnalysisConfig) {
    let mut refs: Vec<&mut TrialResult> = results.iter_mut().map(|(_, r)| r).collect();
    scissortail_core::review::review_outliers(&mut refs, &cfg.protocol.review);
    let flagged = results.iter().filter(|(_, r)| !r.review.is_empty()).count();
    if flagged > 0 {
        eprintln!("review: {flagged} trials far from the other repetitions of their condition");
    }
}

fn write_measures(path: &str, results: &[(String, TrialResult)]) -> Result<()> {
    let rows: Vec<(&str, &TrialResult)> = results.iter().map(|(s, r)| (s.as_str(), r)).collect();
    std::fs::write(path, scissortail_core::export::measures_csv(&rows))?;
    eprintln!("wrote {path}");
    Ok(())
}

fn write_wav(path: &Path, samples: &[f32], fs: u32) -> Result<()> {
    let mut b = Vec::with_capacity(44 + samples.len() * 2);
    let data_len = (samples.len() * 2) as u32;
    b.extend_from_slice(b"RIFF");
    b.extend_from_slice(&(36 + data_len).to_le_bytes());
    b.extend_from_slice(b"WAVEfmt ");
    b.extend_from_slice(&16u32.to_le_bytes());
    b.extend_from_slice(&1u16.to_le_bytes());
    b.extend_from_slice(&1u16.to_le_bytes());
    b.extend_from_slice(&fs.to_le_bytes());
    b.extend_from_slice(&(fs * 2).to_le_bytes());
    b.extend_from_slice(&2u16.to_le_bytes());
    b.extend_from_slice(&16u16.to_le_bytes());
    b.extend_from_slice(b"data");
    b.extend_from_slice(&data_len.to_le_bytes());
    let peak = samples.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
    for &s in samples {
        b.extend_from_slice(&((s / peak * 0.9 * 32767.0) as i16).to_le_bytes());
    }
    std::fs::write(path, b)?;
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(|s| s.as_str()).unwrap_or("help");
    match cmd {
        "inspect" => {
            let path = PathBuf::from(args.get(1).context("missing file")?);
            let t = Trial::load(&path)?;
            println!("trial {} (variable {})", t.id, t.variable);
            println!("  audio: {:.0} Hz, {} samples ({:.3} s)", t.audio.fs, t.audio.samples.len(), t.audio.duration());
            for s in t.sensor_info() {
                println!("  sensor {:6} {:.0} Hz {} samples × {} cols{}", s.name, s.fs, s.n_samples, s.n_columns, if s.is_reference { "  (reference)" } else { "" });
            }
        }
        "synth" => {
            let ipa = args.get(1).context("missing IPA")?;
            let voice = arg_value(&args, "--voice").unwrap_or_else(|| "af_heart".into());
            let speed: f32 = arg_value(&args, "--speed").map(|s| s.parse()).transpose()?.unwrap_or(1.0);
            let out = PathBuf::from(arg_value(&args, "--out").unwrap_or_else(|| "synth.wav".into()));
            let md = models_dir()?;
            let engines = Engines::load(&md, 4);
            let tts = engines.tts.as_ref().context("TTS model not available")?;
            let t0 = Instant::now();
            let syn = tts.synthesize(ipa, &voice, speed)?;
            println!("kokoro input: {}  unknown: {:?}  ({:.2} s, {:.0} ms)", syn.kokoro_text, syn.unknown_symbols, syn.samples.len() as f64 / syn.fs, t0.elapsed().as_millis());
            for p in &syn.phones {
                println!("  {:4} {:.3}–{:.3}", p.symbol, p.start_s, p.end_s);
            }
            write_wav(&out, &syn.samples, syn.fs as u32)?;
            println!("wrote {}", out.display());
        }
        "analyze" => {
            let root = PathBuf::from(args.get(1).context("missing folder")?);
            let mut cfg = AnalysisConfig::default();
            for w in arg_values(&args, "--word") {
                // "<ipa>=label@pattern"
                let (ipa_label, pattern) = w.split_once('@').map(|(a, b)| (a.to_string(), b.to_string())).unwrap_or((w.clone(), String::new()));
                let (ipa, label) = ipa_label.split_once('=').map(|(a, b)| (a.to_string(), b.to_string())).unwrap_or((ipa_label.clone(), String::new()));
                cfg.locate.words.push(WordSpec { ipa, label, file_pattern: pattern });
            }
            if cfg.locate.words.is_empty() {
                bail!("give at least one --word \"<ipa>\"");
            }
            cfg.locate.carrier = arg_value(&args, "--carrier").unwrap_or_default();
            cfg.protocol = protocol_config(&args)?;
            kinematics_flags(&args, &mut cfg)?;
            if let Some(e) = arg_value(&args, "--engine") {
                cfg.locate.engine = if e.starts_with('t') { Engine::Template } else { Engine::Phonetic };
            }
            let limit: usize = arg_value(&args, "--limit").map(|s| s.parse()).transpose()?.unwrap_or(usize::MAX);
            let every: usize = arg_value(&args, "--every").map(|s| s.parse()).transpose()?.unwrap_or(1).max(1);
            let md = models_dir()?;
            let threads = std::env::var("SCISSORTAIL_THREADS").ok().and_then(|s| s.parse().ok()).unwrap_or(2);
            let engines = Engines::load(&md, threads);
            let analyzer = Analyzer::new(&engines, cfg)?;
            let sessions = scan(&root);
            let files: Vec<(String, PathBuf)> = sessions
                .iter()
                .flat_map(|s| s.trials.iter().step_by(every).take(limit).map(move |t| (s.name.clone(), PathBuf::from(&t.file))))
                .collect();
            eprintln!("{} sessions, {} trials", sessions.len(), files.len());
            let t0 = Instant::now();
            let mut results: Vec<(String, TrialResult)> = files.par_iter().map(|(s, f)| (s.clone(), analyzer.analyze_file(f))).collect();
            eprintln!("analyzed in {:.1} s", t0.elapsed().as_secs_f64());
            review(&mut results, &analyzer.config);
            for (s, r) in &results {
                let occ: Vec<String> = r
                    .occurrences
                    .iter()
                    .map(|o| format!("{} [{:.3}–{:.3}] conf {:.2}", o.occurrence.word, o.occurrence.start_s, o.occurrence.end_s, o.occurrence.confidence))
                    .collect();
                println!("{s}\t{}\t{:?}\t{}", r.trial_id, r.status, occ.join(" | "));
            }
            if let Some(out) = arg_value(&args, "--out") {
                let json: Vec<serde_json::Value> = results.iter().map(|(s, r)| serde_json::json!({"session": s, "result": r})).collect();
                std::fs::write(&out, serde_json::to_string_pretty(&json)?)?;
                eprintln!("wrote {out}");
            }
            if let Some(csv) = arg_value(&args, "--csv") {
                let rs: Vec<TrialResult> = results.iter().map(|(_, r)| r.clone()).collect();
                scissortail_core::export::write_landmarks_csv(Path::new(&csv), "all", &rs)?;
                eprintln!("wrote {csv}");
            }
            if let Some(m) = arg_value(&args, "--measures") {
                write_measures(&m, &results)?;
            }
        }
        "measure" => {
            let src = args.get(1).context("missing results.json (from analyze --out)")?;
            let mut cfg = AnalysisConfig::default();
            cfg.protocol = protocol_config(&args)?;
            kinematics_flags(&args, &mut cfg)?;
            let items: Vec<serde_json::Value> = serde_json::from_str(&std::fs::read_to_string(src)?)?;
            let mut results: Vec<(String, TrialResult)> = items
                .into_iter()
                .map(|v| Ok((v["session"].as_str().unwrap_or("").to_string(), serde_json::from_value(v["result"].clone())?)))
                .collect::<Result<_>>()?;
            let t0 = Instant::now();
            results.par_iter_mut().for_each(|(_, r)| {
                let Ok(trial) = Trial::load(Path::new(&r.file)) else { return };
                for o in r.occurrences.iter_mut() {
                    scissortail_core::analysis::refresh_protocol(&trial, o, &cfg);
                }
            });
            eprintln!("{} trials measured in {:.1} s", results.len(), t0.elapsed().as_secs_f64());
            review(&mut results, &cfg);
            if let Some(m) = arg_value(&args, "--measures") {
                write_measures(&m, &results)?;
            }
            if let Some(out) = arg_value(&args, "--out") {
                let json: Vec<serde_json::Value> = results.iter().map(|(s, r)| serde_json::json!({"session": s, "result": r})).collect();
                std::fs::write(&out, serde_json::to_string_pretty(&json)?)?;
                eprintln!("wrote {out}");
            }
        }
        _ => {
            println!("usage:\n  scissortail inspect <trial.mat>\n  scissortail synth \"<ipa>\" [--voice v] [--speed s] --out x.wav\n  scissortail analyze <folder> --word \"<ipa>[=label[@pattern]]\" [--carrier \"<ipa with *>\"] [--engine phonetic|template] [--out r.json] [--csv l.csv] [--protocol peaks|jaw-cycle|gestures] [--protocol-config p.json] [--measures m.csv]\n  scissortail measure <r.json> --protocol jaw-cycle|gestures [--protocol-config p.json] --measures m.csv [--out r2.json]");
        }
    }
    Ok(())
}
