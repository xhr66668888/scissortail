//! Per-trial analysis: locate the word occurrence(s) — or, without a target
//! word, the main stretch of speech — then extract kinematic landmarks for
//! every enabled sensor and dimension.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::aligner::PhoneticAligner;
use crate::error::{Error, Result};
use crate::kinematics::{analyze_window, manual_peak, KinematicResult, KinematicsConfig, Landmark};
use crate::locate::{build_templates, locate_phonetic, locate_speech, locate_template, refine_boundaries, Engine, LocateConfig, Occurrence, PhraseTemplate};
use crate::trial::Trial;
use crate::tts::Tts;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct AnalysisConfig {
    pub locate: LocateConfig,
    pub kinematics: KinematicsConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OccurrenceResult {
    pub occurrence: Occurrence,
    pub kinematics: Vec<KinematicResult>,
    /// Set when the researcher moved the boundaries by hand.
    #[serde(default)]
    pub edited: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrialStatus {
    Ok,
    LowConfidence,
    NotFound,
    Error,
    Excluded,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrialResult {
    pub trial_id: String,
    pub file: String,
    pub status: TrialStatus,
    pub message: String,
    pub audio_duration_s: f64,
    pub ema_duration_s: f64,
    pub occurrences: Vec<OccurrenceResult>,
    #[serde(default)]
    pub reviewed: bool,
    #[serde(default)]
    pub note: String,
}

/// Loaded models (shared, thread-safe).
pub struct Engines {
    pub aligner: Option<PhoneticAligner>,
    pub tts: Option<Tts>,
}

impl Engines {
    pub fn load(models_dir: &Path, threads: usize) -> Engines {
        let aligner = PhoneticAligner::load(&models_dir.join("mms-fa-int8.onnx"), threads).ok();
        let tts = Tts::load(&models_dir.join("kokoro-v1.0-int8-dur.onnx"), &models_dir.join("voices"), threads).ok();
        Engines { aligner, tts }
    }

    /// The phonetic aligner only (the desktop app does not use the TTS).
    pub fn load_aligner(models_dir: &Path, threads: usize) -> Engines {
        Engines { aligner: PhoneticAligner::load(&models_dir.join("mms-fa-int8.onnx"), threads).ok(), tts: None }
    }
}

/// True when the configuration names at least one target word.
pub fn has_target_words(cfg: &LocateConfig) -> bool {
    cfg.words.iter().any(|w| !w.ipa.trim().is_empty())
}

/// Everything needed to analyse trials with one configuration
/// (templates are synthesized once and shared between clones).
#[derive(Clone)]
pub struct Analyzer<'a> {
    pub engines: &'a Engines,
    pub config: AnalysisConfig,
    templates: std::sync::Arc<Vec<PhraseTemplate>>,
}

impl<'a> Analyzer<'a> {
    /// Without target words, trials are segmented by speech activity alone.
    pub fn new(engines: &'a Engines, config: AnalysisConfig) -> Result<Self> {
        let templates = match config.locate.engine {
            _ if !has_target_words(&config.locate) => vec![],
            Engine::Speech => vec![],
            Engine::Template => {
                let tts = engines.tts.as_ref().ok_or_else(|| Error::Model("Kokoro TTS model is not available".into()))?;
                build_templates(tts, &config.locate)?
            }
            Engine::Phonetic => {
                if engines.aligner.is_none() {
                    return Err(Error::Model("phonetic aligner model (mms-fa-int8.onnx) is not available".into()));
                }
                vec![]
            }
        };
        Ok(Analyzer { engines, config, templates: std::sync::Arc::new(templates) })
    }

    pub fn analyze_file(&self, path: &Path) -> TrialResult {
        let file = path.display().to_string();
        let trial_id = path.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
        match Trial::load(path) {
            Ok(trial) => self.analyze_trial(&trial, path),
            Err(e) => TrialResult {
                trial_id,
                file,
                status: TrialStatus::Error,
                message: e.to_string(),
                audio_duration_s: 0.0,
                ema_duration_s: 0.0,
                occurrences: vec![],
                reviewed: false,
                note: String::new(),
            },
        }
    }

    pub fn analyze_trial(&self, trial: &Trial, path: &Path) -> TrialResult {
        let file_name = path.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
        let ema_dur = trial.sensors.iter().map(|s| s.duration()).fold(0.0, f64::max);
        let mut result = TrialResult {
            trial_id: trial.id.clone(),
            file: path.display().to_string(),
            status: TrialStatus::Ok,
            message: String::new(),
            audio_duration_s: trial.audio.duration(),
            ema_duration_s: ema_dur,
            occurrences: vec![],
            reviewed: false,
            note: String::new(),
        };
        let use_words = has_target_words(&self.config.locate) && self.config.locate.engine != Engine::Speech;
        let located = match self.config.locate.engine {
            _ if !use_words => Ok(vec![]),
            Engine::Phonetic => locate_phonetic(self.engines.aligner.as_ref().unwrap(), &trial.audio, &self.config.locate, &file_name),
            Engine::Template => locate_template(&self.templates, &trial.audio, &self.config.locate, &file_name),
            Engine::Speech => unreachable!(),
        };
        let mut occs = match located {
            Ok(mut o) => {
                if self.config.locate.refine_boundaries {
                    o.iter_mut().for_each(|occ| refine_boundaries(&trial.audio, occ));
                }
                o
            }
            Err(e) => {
                result.message = e.to_string();
                vec![]
            }
        };
        if occs.is_empty() {
            // no target word (configured or found): fall back to the speech segment
            occs = locate_speech(&trial.audio);
            if occs.is_empty() {
                result.status = if result.message.is_empty() { TrialStatus::NotFound } else { TrialStatus::Error };
                if result.message.is_empty() {
                    result.message = "no speech found in the recording".into();
                }
                return result;
            }
            if use_words {
                result.status = TrialStatus::LowConfidence;
                result.message = format!("target word not found — the speech segment is marked instead{}", if result.message.is_empty() { String::new() } else { format!(" ({})", result.message) });
            } else {
                result.message.clear();
            }
        } else if occs.iter().all(|o| o.confidence < 0.3) {
            result.status = TrialStatus::LowConfidence;
            result.message = "low alignment confidence — please review".into();
        }
        if ema_dur + 0.02 < occs.iter().map(|o| o.end_s).fold(0.0, f64::max) {
            result.message = format!("{} (EMA ends at {:.2} s, before the word ends)", result.message, ema_dur).trim().to_string();
        }
        result.occurrences = occs
            .into_iter()
            .map(|o| OccurrenceResult { kinematics: kinematics_for(trial, &o, &self.config.kinematics), occurrence: o, edited: false })
            .collect();
        result
    }
}

/// Kinematic landmarks of every enabled sensor/dimension for one occurrence.
pub fn kinematics_for(trial: &Trial, occ: &Occurrence, cfg: &KinematicsConfig) -> Vec<KinematicResult> {
    let t0 = (occ.start_s - cfg.pad_pre_ms / 1000.0).max(0.0);
    let t1 = occ.end_s + cfg.pad_post_ms / 1000.0;
    let mut out = Vec::new();
    for sensor in &trial.sensors {
        let rule = cfg.rule_for(sensor);
        if !rule.enabled {
            continue;
        }
        for &dim in &cfg.dims {
            if let Some(r) = analyze_window(sensor, dim, rule.position_peak, t0, t1, cfg) {
                out.push(r);
            }
        }
    }
    out
}

/// Re-computes kinematics after the researcher edited an occurrence.
pub fn recompute(path: &Path, occ: &Occurrence, cfg: &KinematicsConfig) -> Result<Vec<KinematicResult>> {
    let trial = Trial::load(path)?;
    Ok(kinematics_for(&trial, occ, cfg))
}

/// Re-applies the landmarks the researcher placed by hand (`old`) to freshly
/// computed kinematics: same time, value re-read from the current signals.
pub fn keep_manual(trial: &Trial, new: &mut [KinematicResult], old: &[KinematicResult], cfg: &KinematicsConfig) {
    for o in old {
        let Some(n) = new.iter_mut().find(|n| n.sensor == o.sensor && n.dim == o.dim) else { continue };
        let Some(sensor) = trial.sensor(&o.sensor) else { continue };
        for which in Landmark::all() {
            if let Some(p) = o.peak(which).filter(|p| p.manual) {
                if let Some(mp) = manual_peak(sensor, o.dim, which, p.time_s, 0.0, cfg) {
                    *n.peak_mut(which) = Some(mp);
                }
            }
        }
    }
}

/// Kinematics of an occurrence after its boundaries (or the settings)
/// changed; hand-placed landmarks stay where the researcher put them.
pub fn refresh_kinematics(trial: &Trial, o: &mut OccurrenceResult, cfg: &KinematicsConfig) {
    let mut fresh = kinematics_for(trial, &o.occurrence, cfg);
    keep_manual(trial, &mut fresh, &o.kinematics, cfg);
    o.kinematics = fresh;
}

pub fn models_dir_candidates(exe_dir: Option<PathBuf>) -> Vec<PathBuf> {
    let mut v = Vec::new();
    if let Ok(p) = std::env::var("SCISSORTAIL_MODELS") {
        v.push(PathBuf::from(p));
    }
    if let Some(d) = exe_dir {
        v.push(d.join("models"));
        v.push(d.join("../Resources/models"));
        v.push(d.join("../../models"));
        v.push(d.join("../../../models"));
    }
    v.push(PathBuf::from("models"));
    v
}
