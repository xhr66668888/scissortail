//! Application state: the aligner model, the workspace (the trial files the
//! researcher dropped in, their analysis results and edits) and its
//! persistence.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use scissortail_core::analysis::{models_dir_candidates, AnalysisConfig, Analyzer, Engines, TrialResult};
use scissortail_core::locate::WordSpec;

static ENGINES: OnceLock<Engines> = OnceLock::new();

/// ONNX Runtime threads for the aligner. Fixed rather than taken from the
/// number of cores: the int8 model's posteriors change slightly with the
/// thread count, enough to flip borderline trials, and results must not
/// depend on the machine. 2 is the setting the method was validated with
/// (and the command-line tool's default). The tract engine of the Windows and
/// Intel-Mac builds ignores it: it runs every inference on one thread.
const ALIGNER_THREADS: usize = 2;

/// Waits (up to two minutes) for the background model loader.
pub fn engines() -> Result<&'static Engines, String> {
    let t0 = Instant::now();
    loop {
        if let Some(e) = ENGINES.get() {
            return Ok(e);
        }
        if t0.elapsed() > Duration::from_secs(120) {
            return Err("the speech model is still loading — please try again in a moment".into());
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

pub fn engines_if_ready() -> Option<&'static Engines> {
    ENGINES.get()
}

/// The "It's a ___ again" /tVCVt/ protocol of the bundled patient data: the
/// target word is picked per trial from the file-name code (ad_, at_, ed_,
/// et_), or the best-fitting one when the name carries no code.
pub fn default_config() -> AnalysisConfig {
    let mut c = AnalysisConfig::default();
    c.locate.carrier = "ɪts ə * əɡˈɛn".into();
    c.locate.words = [("tadat", "tˈɑdɑt", "ad_"), ("tatat", "tˈɑtɑt", "at_"), ("tedat", "tˈɛdɑt", "ed_"), ("tetat", "tˈɛtɑt", "et_")]
        .iter()
        .map(|(label, ipa, pattern)| WordSpec { label: label.to_string(), ipa: ipa.to_string(), file_pattern: pattern.to_string() })
        .collect();
    c
}

#[derive(Serialize, Deserialize)]
#[serde(default)]
pub struct Workspace {
    /// Trial files, in the order they were added.
    pub files: Vec<String>,
    pub config: AnalysisConfig,
    /// Keyed by trial file path.
    pub results: HashMap<String, TrialResult>,
}

impl Default for Workspace {
    fn default() -> Self {
        Workspace { files: vec![], config: default_config(), results: HashMap::new() }
    }
}

impl Workspace {
    pub fn pending(&self) -> Vec<String> {
        self.files.iter().filter(|f| !self.results.contains_key(*f)).cloned().collect()
    }
}

pub struct AppState {
    pub models_dir: PathBuf,
    data_dir: PathBuf,
    pub ws: Mutex<Workspace>,
    pub analyzer_cache: Mutex<Option<(String, Arc<Analyzer<'static>>)>>,
    /// Bumped whenever the settings change; background results computed with
    /// older settings are discarded.
    pub generation: AtomicU64,
    /// Interface requests currently waiting for an analysis; the background
    /// worker pauses while this is non-zero so the open file comes first.
    pub foreground: AtomicUsize,
    pub worker_running: AtomicBool,
    save_lock: Mutex<()>,
}

/// Marks a foreground analysis for as long as it is alive.
pub struct Foreground<'a>(&'a AtomicUsize);

impl<'a> Foreground<'a> {
    pub fn enter(counter: &'a AtomicUsize) -> Self {
        counter.fetch_add(1, Ordering::SeqCst);
        Foreground(counter)
    }
}

impl Drop for Foreground<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

/// Local-time-independent compact timestamp (UTC): YYYYMMDD-HHMMSS.
pub fn timestamp() -> String {
    let secs = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0);
    let days = secs.div_euclid(86400);
    let rem = secs.rem_euclid(86400);
    // civil-from-days (Howard Hinnant)
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{:04}{:02}{:02}-{:02}{:02}{:02}", y, m, d, rem / 3600, (rem % 3600) / 60, rem % 60)
}

impl AppState {
    pub fn new(resource_dir: Option<PathBuf>, data_dir: PathBuf) -> AppState {
        let exe_dir = std::env::current_exe().ok().and_then(|p| p.parent().map(|p| p.to_path_buf()));
        let mut candidates = vec![];
        if let Some(r) = resource_dir {
            candidates.push(r.join("models"));
        }
        candidates.extend(models_dir_candidates(exe_dir));
        let models_dir = candidates
            .iter()
            .find(|d| d.join("mms-fa-int8.onnx").exists())
            .cloned()
            .unwrap_or_else(|| candidates.first().cloned().unwrap_or_else(|| PathBuf::from("models")));
        let _ = std::fs::create_dir_all(&data_dir);
        let ws = std::fs::read_to_string(data_dir.join("workspace.json"))
            .ok()
            .and_then(|t| serde_json::from_str::<Workspace>(&t).ok())
            .unwrap_or_default();
        AppState {
            models_dir,
            data_dir,
            ws: Mutex::new(ws),
            analyzer_cache: Mutex::new(None),
            generation: AtomicU64::new(0),
            foreground: AtomicUsize::new(0),
            worker_running: AtomicBool::new(false),
            save_lock: Mutex::new(()),
        }
    }

    /// Loads the speech model in the background. Without it the app still
    /// works: trials are then segmented by speech activity.
    pub fn start_loading_engines(&self) {
        let dir = self.models_dir.clone();
        std::thread::spawn(move || {
            let _ = ENGINES.set(Engines::load_aligner(&dir, ALIGNER_THREADS));
        });
    }

    /// Writes the workspace atomically (temp file + rename).
    pub fn save(&self) {
        let _g = self.save_lock.lock().unwrap();
        let text = serde_json::to_string(&*self.ws.lock().unwrap()).unwrap_or_default();
        let path = self.data_dir.join("workspace.json");
        let tmp = path.with_extension("json.tmp");
        if std::fs::write(&tmp, text).is_ok() {
            let _ = std::fs::rename(&tmp, &path);
        }
    }
}
