//! Measurement protocols: which articulatory landmarks are labelled in a
//! located speech segment, and which measures are derived from them.
//!
//! * [`ProtocolKind::Peaks`] — position / velocity / acceleration peaks of
//!   every sensor (see `kinematics`); no derived measures.
//! * [`ProtocolKind::JawCycle`] — the jaw vowel-to-vowel cycle and the onset
//!   of tongue-tip raising for the medial consonant of /tVCVt/-type words,
//!   following Masapollo & Nittrouer (2023, JSLHR 66, 901–915): six landmarks
//!   (V1, PV1, MaxDisp, PV2, TTonset, V2) → JVC duration, maximum jaw
//!   displacement, TT latency and TT phase angle on the jaw phase plane.
//! * [`ProtocolKind::Gestures`] — velocity-threshold gestural landmarks of
//!   one or more articulators (onset, peak velocity, target, maximum
//!   constriction, release, second peak velocity, offset; the MVIEW
//!   `findgest` convention) with per-gesture measures and inter-gestural lags.
//!
//! Every protocol is computed from the smoothed signals of `kinematics`
//! (low-pass + Savitzky–Golay velocity). Landmarks placed by hand are passed
//! back in as `fixed`: they stay where the researcher put them, and every
//! automatic landmark and measure that depends on them is recomputed.

use serde::{Deserialize, Serialize};

use crate::kinematics::{derivatives, refine, value_at, Dim, KinematicsConfig, Landmark, Polarity};
use crate::locate::Occurrence;
use crate::trial::{Sensor, Trial};

// ----------------------------------------------------------- configuration --

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProtocolKind {
    /// P / V / A peaks only (no derived measures).
    #[default]
    Peaks,
    /// Jaw vowel cycle + tongue-tip raising onset (Masapollo & Nittrouer 2023).
    JawCycle,
    /// Velocity-threshold gestural landmarks + inter-gestural lags.
    Gestures,
}

impl ProtocolKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            ProtocolKind::Peaks => "peaks",
            ProtocolKind::JawCycle => "jaw_cycle",
            ProtocolKind::Gestures => "gestures",
        }
    }
}

/// Rules of the jaw-cycle protocol. The defaults are the criteria of
/// Masapollo & Nittrouer (2023).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct JawCycleConfig {
    /// Jaw sensor; empty = the first sensor named like JAW / JA / LI / MAND / CHIN.
    pub jaw_sensor: String,
    /// Sensor whose raising onset is timed (the medial consonant's articulator); empty = TT.
    pub tongue_sensor: String,
    /// Vertical dimension.
    pub dim: Dim,
    /// Time the tongue relative to the jaw (tongue − jaw), isolating lingual
    /// movement from jaw movement, as in the paper.
    pub tongue_minus_jaw: bool,
    /// Velocity criterion (mm/s; 10 = 1 cm/s) for onsets that start gradually.
    pub velocity_threshold: f64,
    /// Displacement criterion (mm below the closure height) for weak, slow
    /// jaw lowerings (unstressed vowels).
    pub displacement_threshold: f64,
    /// An onset counts as gradual when the velocity takes longer than this
    /// (ms) to go from zero to the velocity criterion; the criterion is then
    /// used instead of the zero crossing.
    pub gradual_ms: f64,
    /// Jaw lowerings whose peak velocity is below this (mm/s) are weak: a
    /// gradual onset is then placed by the displacement criterion. 0 = never
    /// (the default: on the Masapollo & Nittrouer recordings the coders'
    /// labels are matched best with the velocity criterion throughout).
    pub weak_lowering_velocity: f64,
    /// Maximum jaw displacement on a plateau: samples within this distance
    /// (mm) of the lowest point form the plateau; the landmark goes to its
    /// temporal midpoint.
    pub plateau_mm: f64,
    /// How far before the acoustic word onset the first jaw lowering may start (ms).
    pub search_before_ms: f64,
    /// Jaw lowerings smaller than this (mm) are ignored when looking for the vowels.
    pub min_lowering_mm: f64,
    /// … and so are lowerings slower than this (mm/s).
    pub min_lowering_speed: f64,
    /// Flag the trial when the second-best pair of jaw lowerings fits the
    /// vowels within this many ms of the best.
    pub ambiguity_ms: f64,
    /// Flag the trial when another tongue-tip raising reaches this fraction
    /// of the chosen one's peak velocity.
    pub ambiguity_ratio: f64,
    /// Flag the trial when the alternative coding rules (zero crossing only,
    /// velocity criterion only, displacement criterion, tongue onset with or
    /// without the earlier raising) move JVC duration or TT latency by more
    /// than this (ms). 0 = off.
    pub rule_spread_ms: f64,
    /// Gradual-onset limit for the tongue (ms); negative = same as `gradual_ms`.
    pub tongue_gradual_ms: f64,
    /// A tongue raising that grows out of an earlier raising without slowing
    /// below the velocity criterion starts where that earlier raising starts.
    pub tongue_through_speed_minimum: bool,
}

impl Default for JawCycleConfig {
    fn default() -> Self {
        JawCycleConfig {
            jaw_sensor: String::new(),
            tongue_sensor: String::new(),
            dim: Dim::Z,
            tongue_minus_jaw: true,
            velocity_threshold: 10.0,
            displacement_threshold: 0.5,
            gradual_ms: 10.0,
            weak_lowering_velocity: 0.0,
            plateau_mm: 0.1,
            search_before_ms: 350.0,
            min_lowering_mm: 0.5,
            min_lowering_speed: 10.0,
            ambiguity_ms: 40.0,
            rule_spread_ms: 60.0,
            ambiguity_ratio: 0.7,
            tongue_gradual_ms: 15.0,
            tongue_through_speed_minimum: true,
        }
    }
}

/// One gesture of the gestures protocol.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct GestureSpec {
    /// Short name used in landmark and measure names (e.g. "TT").
    pub label: String,
    pub sensor: String,
    /// Optional sensor subtracted from `sensor` (e.g. JAW for jaw-independent tongue movement).
    pub reference: String,
    pub dim: Dim,
    /// Direction of the constriction: `Max` (raising, e.g. tongue tip for /t/)
    /// or `Min` (lowering, e.g. jaw opening for a vowel).
    pub direction: Polarity,
    /// Which extremum: 0 = the largest one inside the word; k ≥ 1 = the one
    /// nearest the k-th phone of the aligned word.
    pub anchor_phone: usize,
}

impl Default for GestureSpec {
    fn default() -> Self {
        GestureSpec { label: String::new(), sensor: String::new(), reference: String::new(), dim: Dim::Z, direction: Polarity::Max, anchor_phone: 0 }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct GesturesConfig {
    pub gestures: Vec<GestureSpec>,
    /// Onset / target / release / offset at this fraction of peak velocity (MVIEW default 0.2).
    pub threshold: f64,
    /// Search margins around the word (ms).
    pub pad_ms: f64,
    /// Minimum movement amplitude (mm) for an extremum to count as a gesture.
    pub min_amplitude: f64,
    /// Flag a gesture whose onset or offset moves by more than this (ms)
    /// when the threshold is 10 % or 30 % instead. 0 = off.
    pub rule_spread_ms: f64,
}

impl Default for GesturesConfig {
    fn default() -> Self {
        GesturesConfig {
            gestures: vec![
                GestureSpec { label: "JAW".into(), sensor: "JAW".into(), direction: Polarity::Min, anchor_phone: 2, ..Default::default() },
                GestureSpec { label: "TT".into(), sensor: "TT".into(), direction: Polarity::Max, anchor_phone: 3, ..Default::default() },
            ],
            threshold: 0.2,
            pad_ms: 150.0,
            min_amplitude: 0.5,
            rule_spread_ms: 60.0,
        }
    }
}

/// Review across trials (see `review`): a timing measure far from the other
/// repetitions of the same condition marks the trial "please check".
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ReviewConfig {
    /// Robust z (distance from the condition's median in units of 1.4826 ×
    /// median absolute deviation) above which a measure is an outlier. 0 = off.
    pub outlier_z: f64,
    /// Smallest spread assumed for a condition (ms, and degrees for phase
    /// angles), so tight conditions do not flag every small deviation.
    pub floor_ms: f64,
    pub floor_deg: f64,
    /// Conditions with fewer repetitions are not checked.
    pub min_repetitions: usize,
}

impl Default for ReviewConfig {
    fn default() -> Self {
        ReviewConfig { outlier_z: 3.0, floor_ms: 8.0, floor_deg: 8.0, min_repetitions: 5 }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ProtocolConfig {
    pub kind: ProtocolKind,
    pub jaw_cycle: JawCycleConfig,
    pub gestures: GesturesConfig,
    pub review: ReviewConfig,
}

// ----------------------------------------------------------------- results --

/// One labelled point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolLandmark {
    /// Stable identifier, e.g. "V1", "TTonset", "TT.GONS".
    pub id: String,
    /// Human-readable name.
    pub label: String,
    /// Sensor whose panel shows the landmark.
    pub sensor: String,
    /// Sensor subtracted from `sensor` ("" = none).
    #[serde(default)]
    pub reference: String,
    pub dim: Dim,
    /// Curve the landmark is read from (position or velocity).
    pub curve: Landmark,
    pub time_s: f64,
    /// Position (mm) and velocity (mm/s) at `time_s`.
    pub position: f64,
    pub velocity: f64,
    /// How it was found: "zero crossing", "velocity 10 mm/s", "displacement 0.5 mm", "minimum", …
    pub criterion: String,
    #[serde(default)]
    pub manual: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Measure {
    /// Column name, e.g. "JVC_Dur_ms".
    pub id: String,
    pub label: String,
    pub unit: String,
    pub value: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolResult {
    pub kind: ProtocolKind,
    pub landmarks: Vec<ProtocolLandmark>,
    pub measures: Vec<Measure>,
    /// Problems worth a look (fallback criteria, implausible values, …).
    pub warnings: Vec<String>,
}

impl ProtocolResult {
    pub fn landmark(&self, id: &str) -> Option<&ProtocolLandmark> {
        self.landmarks.iter().find(|l| l.id == id)
    }
    pub fn measure(&self, id: &str) -> Option<f64> {
        self.measures.iter().find(|m| m.id == id).and_then(|m| m.value)
    }
}

/// Landmarks of the protocol for one occurrence (`None` for the peaks
/// protocol). `fixed` are landmarks placed by hand; they are kept.
pub fn compute(trial: &Trial, occ: &Occurrence, cfg: &ProtocolConfig, kin: &KinematicsConfig, fixed: &[ProtocolLandmark]) -> Option<ProtocolResult> {
    match cfg.kind {
        ProtocolKind::Peaks => None,
        ProtocolKind::JawCycle => Some(jaw_cycle(trial, occ, &cfg.jaw_cycle, kin, fixed)),
        ProtocolKind::Gestures => Some(gestures(trial, occ, &cfg.gestures, kin, fixed)),
    }
}

/// Manual landmarks of an earlier result, for [`compute`].
pub fn manual_landmarks(r: Option<&ProtocolResult>) -> Vec<ProtocolLandmark> {
    r.map(|r| r.landmarks.iter().filter(|l| l.manual).cloned().collect()).unwrap_or_default()
}

// ------------------------------------------------------------------ signals --

/// Smoothed position and velocity of one sensor dimension (optionally
/// relative to a reference sensor).
pub struct Track {
    pub fs: f64,
    pub pos: Vec<f64>,
    pub vel: Vec<f64>,
}

impl Track {
    pub fn load(trial: &Trial, sensor: &str, reference: &str, dim: Dim, kin: &KinematicsConfig) -> Option<Track> {
        let s = trial.sensor(sensor)?;
        let d = if reference.trim().is_empty() {
            derivatives(s, dim, kin)?
        } else {
            let r = trial.sensor(reference)?;
            derivatives(&difference(s, r)?, dim, kin)?
        };
        Some(Track { fs: d.fs, pos: d.pos, vel: d.vel })
    }

    fn len(&self) -> usize {
        self.pos.len()
    }
    fn t(&self, i: usize) -> f64 {
        i as f64 / self.fs
    }
    /// Sample index nearest time `t`, clamped.
    fn i(&self, t: f64) -> usize {
        ((t * self.fs).round().max(0.0) as usize).min(self.len().saturating_sub(1))
    }
    fn pos_at(&self, t: f64) -> f64 {
        value_at(&self.pos, self.fs, t)
    }
    fn vel_at(&self, t: f64) -> f64 {
        value_at(&self.vel, self.fs, t)
    }
}

/// `a − b` sample by sample (both sensors must share the sampling rate).
pub fn difference(a: &Sensor, b: &Sensor) -> Option<Sensor> {
    if (a.fs - b.fs).abs() > 1e-6 {
        return None;
    }
    let n = a.len().min(b.len());
    let cols = a.columns.len().min(b.columns.len()).min(3);
    let columns = (0..cols).map(|c| (0..n).map(|i| a.columns[c][i] - b.columns[c][i]).collect()).collect();
    Some(Sensor { name: format!("{}−{}", a.name, b.name), fs: a.fs, columns })
}

/// First sensor whose name matches one of `patterns` (case-insensitive prefix or exact).
fn find_sensor(trial: &Trial, wanted: &str, patterns: &[&str]) -> Option<String> {
    if !wanted.trim().is_empty() {
        return trial.sensor(wanted.trim()).map(|s| s.name.clone());
    }
    for p in patterns {
        if let Some(s) = trial.sensors.iter().find(|s| s.name.to_ascii_uppercase() == *p) {
            return Some(s.name.clone());
        }
    }
    for p in patterns {
        if let Some(s) = trial.sensors.iter().find(|s| s.name.to_ascii_uppercase().starts_with(p) && !s.looks_like_reference()) {
            return Some(s.name.clone());
        }
    }
    None
}

/// Time where `y` crosses `level` between samples `i` and `i + 1` (linear interpolation).
fn crossing(y: &[f64], i: usize, level: f64, fs: f64) -> f64 {
    let j = (i + 1).min(y.len() - 1);
    let (a, b) = (y[i], y[j]);
    let f = if (b - a).abs() < 1e-12 { 0.0 } else { ((level - a) / (b - a)).clamp(0.0, 1.0) };
    (i as f64 + f) / fs
}

fn argmax_by(lo: usize, hi: usize, f: impl Fn(usize) -> f64) -> usize {
    (lo..=hi).fold(lo, |b, i| if f(i) > f(b) { i } else { b })
}

/// Peak (sub-sample time) of `sign · y` in [lo, hi].
fn peak(y: &[f64], lo: usize, hi: usize, sign: f64, fs: f64) -> (usize, f64) {
    let i = argmax_by(lo, hi, |k| sign * y[k]);
    let t = if i > lo && i < hi { refine(y, i, fs).0 } else { i as f64 / fs };
    (i, t)
}

/// Rules for finding where a movement starts.
#[derive(Clone, Copy)]
pub struct OnsetRules {
    /// Velocity criterion (mm/s).
    pub threshold: f64,
    /// Gradual start: zero → criterion takes longer than this (s).
    pub gradual_s: f64,
    /// Displacement criterion (mm) for weak movements; 0 = never.
    pub displacement: f64,
    /// Peak speed (mm/s) below which a movement is weak.
    pub weak_speed: f64,
    /// When the speed rises again before dropping to the criterion (the
    /// movement grows out of an earlier one), continue back through it.
    pub through_speed_minimum: bool,
}

/// Onset of the movement whose velocity peak is at sample `p`; `sign` = +1
/// for a raising movement, −1 for a lowering one. Searches back to `lo`.
///
/// The onset is the velocity zero crossing that starts the movement. When
/// the speed lingers between zero and the criterion for longer than
/// `gradual_s` (a gradual start), the onset is where the speed passes the
/// criterion — or, for a weak movement, where the position has moved
/// `displacement` mm away from where it was at the zero crossing. When the
/// movement grows out of an earlier one in the same direction without
/// stopping, the onset is the speed minimum between them.
pub fn movement_onset(tr: &Track, p: usize, lo: usize, sign: f64, rules: &OnsetRules) -> (f64, String) {
    let s = |i: usize| sign * tr.vel[i];
    let thr = rules.threshold;
    // back from the peak to the velocity criterion
    let mut i = p;
    while i > lo && s(i - 1) > thr && (rules.through_speed_minimum || s(i - 1) <= s(i)) {
        i -= 1;
    }
    if rules.through_speed_minimum && i == lo && s(lo) > thr {
        // never slowed down to the criterion within the window: the earlier
        // movement is not part of this one after all
        return movement_onset(tr, p, lo, sign, &OnsetRules { through_speed_minimum: false, ..*rules });
    }
    if i > lo && s(i - 1) > thr {
        // speed rises again before dropping to the criterion: a preceding movement
        return (tr.t(i), "speed minimum".into());
    }
    if i == lo {
        return (tr.t(lo), "window start".into());
    }
    let i_thr = i - 1; // s(i_thr) ≤ thr < s(i_thr + 1)
    let t_thr = crossing(&tr.vel.iter().map(|v| sign * v).collect::<Vec<_>>(), i_thr, thr, tr.fs);
    // further back to the zero crossing
    let mut k = i_thr;
    while k > lo && s(k) > 0.0 && s(k - 1) <= s(k) {
        k -= 1;
    }
    let (t_zero, zero_kind) = if s(k) <= 0.0 {
        let sv: Vec<f64> = tr.vel.iter().map(|v| sign * v).collect();
        (crossing(&sv, k, 0.0, tr.fs), "zero crossing")
    } else {
        // the speed never reached zero (lowest speed at k)
        (tr.t(k), "speed minimum")
    };
    if t_thr - t_zero <= rules.gradual_s {
        return (t_zero, zero_kind.into());
    }
    let peak_speed = s(p);
    if rules.displacement > 0.0 && peak_speed < rules.weak_speed {
        let p0 = tr.pos_at(t_zero);
        let start = tr.i(t_zero);
        for j in start..=p {
            if sign * (tr.pos[j] - p0) >= rules.displacement {
                let d: Vec<f64> = tr.pos.iter().map(|x| sign * (x - p0)).collect();
                let t = if j > start { crossing(&d, j - 1, rules.displacement, tr.fs) } else { tr.t(j) };
                return (t.max(t_zero), format!("displacement {} mm", rules.displacement));
            }
        }
    }
    (t_thr, format!("velocity {} mm/s", thr))
}

/// One lowering movement of the jaw (sample indices).
pub struct Movement {
    pub start: usize,
    /// Peak lowering velocity.
    pub peak: usize,
    /// Where the lowering stops (or slows down most).
    pub end: usize,
    pub amplitude: f64,
}

/// Jaw lowering movements in [lo, hi]: peaks of lowering speed of at least
/// `min_lowering_speed`; peaks the speed does not drop below 60 % of
/// in between belong to one movement.
fn lowering_movements(jaw: &Track, lo: usize, hi: usize, cfg: &JawCycleConfig) -> Vec<Movement> {
    let s = |i: usize| -jaw.vel[i];
    let n = jaw.len();
    let (lo, hi) = (lo.max(1), hi.min(n.saturating_sub(2)));
    if hi <= lo {
        return vec![];
    }
    let min_speed = cfg.min_lowering_speed;
    let mut merged: Vec<usize> = vec![];
    for p in (lo..=hi).filter(|&i| s(i) >= s(i - 1) && s(i) > s(i + 1) && s(i) >= min_speed) {
        if let Some(&q) = merged.last() {
            let dip = (q..=p).map(s).fold(f64::INFINITY, f64::min);
            if dip > 0.6 * s(q).min(s(p)) {
                if s(p) > s(q) {
                    *merged.last_mut().unwrap() = p;
                }
                continue;
            }
        }
        merged.push(p);
    }
    merged
        .into_iter()
        .filter_map(|p| {
            let mut a = p;
            while a > lo && s(a - 1) > 0.0 && s(a - 1) <= s(a) {
                a -= 1;
            }
            let mut b = p;
            while b < hi && s(b + 1) > 0.0 && s(b + 1) <= s(b) {
                b += 1;
            }
            let amplitude = jaw.pos[a] - jaw.pos[b];
            (amplitude >= cfg.min_lowering_mm).then_some(Movement { start: a, peak: p, end: b, amplitude })
        })
        .collect()
}

/// Vowel spans of the aligned word (seconds); falls back to fixed
/// proportions of the word when the alignment has fewer than two vowels.
pub fn vowel_spans(occ: &Occurrence) -> Vec<(f64, f64)> {
    let is_vowel = |s: &str| s.chars().next().map(|c| "aeiouyɑɛæəɪʊɔʌɒɜɐøœɯɤɨʉAEIOUY".contains(c)).unwrap_or(false);
    let v: Vec<(f64, f64)> = occ.phones.iter().filter(|p| is_vowel(&p.symbol)).map(|p| (p.start_s, p.end_s)).collect();
    if v.len() >= 2 {
        return v;
    }
    let (a, d) = (occ.start_s, (occ.end_s - occ.start_s).max(0.05));
    vec![(a + 0.15 * d, a + 0.45 * d), (a + 0.55 * d, a + 0.85 * d)]
}

fn pinned<'a>(fixed: &'a [ProtocolLandmark], id: &str) -> Option<&'a ProtocolLandmark> {
    fixed.iter().find(|l| l.id == id && l.manual)
}

#[allow(clippy::too_many_arguments)]
fn mk(id: &str, label: &str, sensor: &str, reference: &str, dim: Dim, curve: Landmark, tr: &Track, t: f64, criterion: &str, manual: bool) -> ProtocolLandmark {
    ProtocolLandmark {
        id: id.into(),
        label: label.into(),
        sensor: sensor.into(),
        reference: reference.into(),
        dim,
        curve,
        time_s: t,
        position: tr.pos_at(t),
        velocity: tr.vel_at(t),
        criterion: if manual { "by hand".into() } else { criterion.into() },
        manual,
    }
}

fn measure(id: &str, label: &str, unit: &str, value: Option<f64>) -> Measure {
    Measure { id: id.into(), label: label.into(), unit: unit.into(), value: value.filter(|v| v.is_finite()) }
}

// ---------------------------------------------------------------- jaw cycle --

pub const JAW_CYCLE_MEASURES: &[(&str, &str, &str)] = &[
    ("JVC_Dur_ms", "Jaw vowel-cycle duration (V2 − V1)", "ms"),
    ("MaxJ_Disp_mm", "Maximum jaw displacement", "mm"),
    ("TT_Lat_ms", "Tongue-tip onset latency (TTonset − V1)", "ms"),
    ("TT_Pha_deg", "Tongue-tip onset phase angle on the jaw phase plane", "°"),
    ("PV1_mm_s", "Peak jaw lowering velocity", "mm/s"),
    ("PV2_mm_s", "Peak jaw raising velocity", "mm/s"),
    ("Jaw_Amp_mm", "Jaw lowering amplitude (V1 − MaxDisp)", "mm"),
];

fn empty_jaw_cycle(warn: String) -> ProtocolResult {
    ProtocolResult {
        kind: ProtocolKind::JawCycle,
        landmarks: vec![],
        measures: JAW_CYCLE_MEASURES.iter().map(|(id, l, u)| measure(id, l, u, None)).collect(),
        warnings: vec![warn],
    }
}

/// Phase angle (degrees) of time `t` on the normalized jaw phase plane of the
/// cycle [v1, v2] with its lowest point at `md` (Kelso et al. 1986; Nittrouer
/// 1991): displacement normalized to ±1 over the whole cycle, velocity to ±1
/// over the half cycle containing `t`; 0° = the cycle's start (highest point),
/// 90° = peak lowering velocity, 180° = lowest point, 270° = peak raising velocity.
pub fn phase_angle(jaw: &Track, v1: f64, md: f64, v2: f64, t: f64) -> Option<f64> {
    let (a, m, b) = (jaw.i(v1), jaw.i(md), jaw.i(v2));
    if b <= a + 1 {
        return None;
    }
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for i in a..=b {
        lo = lo.min(jaw.pos[i]);
        hi = hi.max(jaw.pos[i]);
    }
    if hi - lo < 1e-6 {
        return None;
    }
    let (h0, h1) = if t <= md { (a, m) } else { (m, b) };
    let vmax = (h0..=h1.max(h0)).map(|i| jaw.vel[i].abs()).fold(0.0, f64::max);
    if vmax < 1e-9 {
        return None;
    }
    let y = 2.0 * (jaw.pos_at(t) - lo) / (hi - lo) - 1.0;
    let v = jaw.vel_at(t) / vmax;
    // 0–360° around the cycle; just after the top a slightly positive
    // velocity gives a small negative angle rather than ~360°
    let mut phi = (-v).atan2(y).to_degrees().rem_euclid(360.0);
    if t <= md && phi > 270.0 {
        phi -= 360.0;
    }
    Some(phi)
}

/// The jaw-cycle protocol of Masapollo & Nittrouer (2023). With
/// `rule_spread_ms` > 0 the landmarks are also found with the alternative
/// coding rules the paper allows; when these disagree strongly, the trial is
/// one where human coders would likely disagree too, and it is flagged.
pub fn jaw_cycle(trial: &Trial, occ: &Occurrence, cfg: &JawCycleConfig, kin: &KinematicsConfig, fixed: &[ProtocolLandmark]) -> ProtocolResult {
    let mut r = jaw_cycle_rules(trial, occ, cfg, kin, fixed);
    if cfg.rule_spread_ms <= 0.0 || r.landmarks.is_empty() {
        return r;
    }
    let variants = [
        JawCycleConfig { gradual_ms: 1e6, tongue_gradual_ms: 1e6, ..cfg.clone() },
        JawCycleConfig { gradual_ms: 0.0, tongue_gradual_ms: 0.0, ..cfg.clone() },
        JawCycleConfig { weak_lowering_velocity: if cfg.weak_lowering_velocity > 0.0 { 0.0 } else { 60.0 }, ..cfg.clone() },
        JawCycleConfig { tongue_through_speed_minimum: !cfg.tongue_through_speed_minimum, ..cfg.clone() },
    ];
    let mut spread = 0.0f64;
    for id in ["JVC_Dur_ms", "TT_Lat_ms"] {
        let Some(base) = r.measure(id) else { continue };
        let (mut lo, mut hi) = (base, base);
        for v in &variants {
            if let Some(x) = jaw_cycle_rules(trial, occ, v, kin, fixed).measure(id) {
                lo = lo.min(x);
                hi = hi.max(x);
            }
        }
        spread = spread.max(hi - lo);
    }
    if spread > cfg.rule_spread_ms {
        r.warnings.push(format!("the coding rules disagree by {spread:.0} ms here (gradual onset or two tongue-tip raisings)"));
    }
    r
}

fn jaw_cycle_rules(trial: &Trial, occ: &Occurrence, cfg: &JawCycleConfig, kin: &KinematicsConfig, fixed: &[ProtocolLandmark]) -> ProtocolResult {
    let Some(jaw_name) = find_sensor(trial, &cfg.jaw_sensor, &["JAW", "JA", "MAND", "LI", "CHIN"]) else {
        return empty_jaw_cycle("no jaw sensor (JAW) in this recording".into());
    };
    let Some(tt_name) = find_sensor(trial, &cfg.tongue_sensor, &["TT", "TTIP", "TIP", "TONGUETIP"]) else {
        return empty_jaw_cycle("no tongue-tip sensor (TT) in this recording".into());
    };
    let dim = if cfg.dim == Dim::Tangential { Dim::Z } else { cfg.dim };
    let Some(jaw) = Track::load(trial, &jaw_name, "", dim, kin) else {
        return empty_jaw_cycle(format!("the {jaw_name} signal could not be read"));
    };
    let reference = if cfg.tongue_minus_jaw { jaw_name.clone() } else { String::new() };
    let Some(tt) = Track::load(trial, &tt_name, &reference, dim, kin) else {
        return empty_jaw_cycle(format!("the {tt_name} signal could not be read"));
    };
    let n = jaw.len().min(tt.len());
    if n < 10 {
        return empty_jaw_cycle("the EMA recording is too short".into());
    }
    let rules = OnsetRules {
        threshold: cfg.velocity_threshold,
        gradual_s: cfg.gradual_ms / 1000.0,
        displacement: cfg.displacement_threshold,
        weak_speed: cfg.weak_lowering_velocity,
        through_speed_minimum: false,
    };
    // tongue onsets: zero crossing, or the velocity criterion when gradual
    let tt_rules = OnsetRules {
        threshold: cfg.velocity_threshold,
        gradual_s: if cfg.tongue_gradual_ms < 0.0 { cfg.gradual_ms } else { cfg.tongue_gradual_ms } / 1000.0,
        displacement: 0.0,
        weak_speed: 0.0,
        through_speed_minimum: cfg.tongue_through_speed_minimum,
    };
    let mut warnings = vec![];
    let vowels = vowel_spans(occ);
    let (vw1, vw2) = (vowels[0], vowels[1]);
    let last = (n - 1) as f64 / jaw.fs;
    let clamp_t = |t: f64| t.clamp(0.0, last);
    let lo_all = jaw.i(occ.start_s - cfg.search_before_ms / 1000.0);

    // The two jaw lowerings of the word: of all jaw lowering movements around
    // the word, the pair whose lowest points best match the two vowels. (The
    // acoustic vowel spans only anchor the choice; the landmarks come from
    // the jaw signal.)
    let hi_all = jaw.i(occ.end_s + 0.1).min(n - 1);
    let moves = lowering_movements(&jaw, lo_all, hi_all, cfg);
    let (c1, c2) = ((vw1.0 + vw1.1) / 2.0, (vw2.0 + vw2.1) / 2.0);
    let mut pairs: Vec<(f64, usize, usize)> = vec![];
    for i in 0..moves.len() {
        for j in i + 1..moves.len() {
            pairs.push(((jaw.t(moves[i].end) - c1).abs() + (jaw.t(moves[j].end) - c2).abs(), i, j));
        }
    }
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let best = pairs.first().copied();
    if let (Some(b0), Some(b1)) = (pairs.first(), pairs.get(1)) {
        if b1.0 - b0.0 < cfg.ambiguity_ms / 1000.0 && pinned(fixed, "V1").is_none() && pinned(fixed, "V2").is_none() {
            warnings.push("two jaw cycles fit the word about equally well".into());
        }
    }
    let Some((_, m1, m2)) = best else {
        return empty_jaw_cycle(format!("fewer than two jaw lowerings found around the word ({} found)", moves.len()));
    };
    let (mv1, mv2) = (&moves[m1], &moves[m2]);
    // V2: onset of the jaw lowering into the second vowel
    let (v2, v2_crit, v2_manual) = match pinned(fixed, "V2") {
        Some(l) => (l.time_s, String::new(), true),
        None => {
            let (t, c) = movement_onset(&jaw, mv2.peak, mv1.peak.min(mv2.peak), -1.0, &rules);
            (t, c, false)
        }
    };
    // V1: onset of the jaw lowering into the first vowel
    let (v1, v1_crit, v1_manual) = match pinned(fixed, "V1") {
        Some(l) => (l.time_s, String::new(), true),
        None => {
            let lo = if m1 > 0 { moves[m1 - 1].peak.max(lo_all) } else { lo_all };
            let (t, c) = movement_onset(&jaw, mv1.peak, lo.min(mv1.peak), -1.0, &rules);
            (t, c, false)
        }
    };
    let (v1, v2) = (clamp_t(v1), clamp_t(v2));
    if v2 <= v1 + 2.0 / jaw.fs {
        warnings.push("V2 does not follow V1 — please check".into());
    }
    let (ia, ib) = (jaw.i(v1), jaw.i(v2).max(jaw.i(v1)));

    // MaxDisp: lowest jaw position in the cycle (midpoint of a plateau)
    let (md, md_crit, md_manual) = match pinned(fixed, "MaxDisp") {
        Some(l) => (l.time_s, String::new(), true),
        None => {
            let im = argmax_by(ia, ib, |i| -jaw.pos[i]);
            let floor = jaw.pos[im] + cfg.plateau_mm;
            let (mut a, mut b) = (im, im);
            while a > ia && jaw.pos[a - 1] <= floor {
                a -= 1;
            }
            while b < ib && jaw.pos[b + 1] <= floor {
                b += 1;
            }
            if b > a + 1 {
                (jaw.t(a) + (jaw.t(b) - jaw.t(a)) / 2.0, "plateau midpoint".to_string(), false)
            } else if im > ia && im < ib {
                (refine(&jaw.pos, im, jaw.fs).0, "minimum".to_string(), false)
            } else {
                (jaw.t(im), "minimum".to_string(), false)
            }
        }
    };
    let im = jaw.i(md).clamp(ia, ib);
    // peak velocities of the lowering and raising half cycles
    let pv = |id: &str, a: usize, b: usize, sign: f64| -> (f64, String, bool) {
        if let Some(l) = pinned(fixed, id) {
            return (l.time_s, String::new(), true);
        }
        let (_, t) = peak(&jaw.vel, a, b.max(a), sign, jaw.fs);
        (t, "peak".into(), false)
    };
    let (pv1, pv1_crit, pv1_manual) = pv("PV1", ia, im, -1.0);
    let (pv2, pv2_crit, pv2_manual) = pv("PV2", im, ib, 1.0);
    if ib.saturating_sub(im) < 2 || jaw.vel_at(pv2) <= 0.0 {
        warnings.push("no jaw raising between the vowels (V2 by the velocity criterion)".into());
    }

    // TTonset: onset of tongue-tip raising for the medial consonant, within the cycle
    let (tto, tto_crit, tto_manual) = match pinned(fixed, "TTonset") {
        Some(l) => (l.time_s, String::new(), true),
        None => {
            // the fastest tongue-tip raising in the cycle (relative to the jaw)
            let a = tt.i(v1);
            let b = tt.i(v2 + 0.03).max(a + 1).min(n - 1);
            let (pt, _) = peak(&tt.vel, a, b, 1.0, tt.fs);
            if tt.vel[pt] < cfg.velocity_threshold {
                warnings.push("no tongue-tip raising found in the jaw cycle".into());
            }
            // a second raising nearly as fast (the speed dropping below the
            // criterion in between): which one the consonant's is, is unclear
            let rival = (a + 1..b).filter(|&i| tt.vel[i] >= tt.vel[i - 1] && tt.vel[i] > tt.vel[i + 1] && i != pt).any(|i| {
                let (x, y) = (i.min(pt), i.max(pt));
                tt.vel[i] >= cfg.ambiguity_ratio * tt.vel[pt] && (x..=y).any(|k| tt.vel[k] < cfg.velocity_threshold)
            });
            if rival {
                warnings.push("two tongue-tip raisings in the cycle".into());
            }
            let (t, c) = movement_onset(&tt, pt, a.min(pt), 1.0, &tt_rules);
            (t.max(v1), c, false)
        }
    };
    if tto > v2 {
        warnings.push("tongue-tip onset after V2".into());
    }

    for (id, c) in [("V1", &v1_crit), ("V2", &v2_crit), ("TTonset", &tto_crit)] {
        if c == "window start" {
            warnings.push(format!("{id}: no clear movement onset"));
        }
    }
    let ref_s = if cfg.tongue_minus_jaw { jaw_name.as_str() } else { "" };
    let landmarks = vec![
        mk("V1", "Jaw lowering onset, vowel 1", &jaw_name, "", dim, Landmark::Position, &jaw, v1, &v1_crit, v1_manual),
        mk("PV1", "Peak jaw lowering velocity", &jaw_name, "", dim, Landmark::Velocity, &jaw, pv1, &pv1_crit, pv1_manual),
        mk("MaxDisp", "Maximum jaw displacement", &jaw_name, "", dim, Landmark::Position, &jaw, md, &md_crit, md_manual),
        mk("PV2", "Peak jaw raising velocity", &jaw_name, "", dim, Landmark::Velocity, &jaw, pv2, &pv2_crit, pv2_manual),
        mk("TTonset", "Tongue-tip raising onset", &tt_name, ref_s, dim, Landmark::Position, &tt, tto, &tto_crit, tto_manual),
        mk("V2", "Jaw lowering onset, vowel 2", &jaw_name, "", dim, Landmark::Position, &jaw, v2, &v2_crit, v2_manual),
    ];
    let jvc = (v2 - v1) * 1000.0;
    if !(50.0..=900.0).contains(&jvc) {
        warnings.push(format!("unusual jaw-cycle duration ({jvc:.0} ms)"));
    }
    let values = [
        Some(jvc),
        Some(jaw.pos_at(md)),
        Some((tto - v1) * 1000.0),
        phase_angle(&jaw, v1, md, v2, tto),
        Some(jaw.vel_at(pv1)),
        Some(jaw.vel_at(pv2)),
        Some(jaw.pos_at(v1) - jaw.pos_at(md)),
    ];
    let measures = JAW_CYCLE_MEASURES.iter().zip(values).map(|((id, l, u), v)| measure(id, l, u, v)).collect();
    ProtocolResult { kind: ProtocolKind::JawCycle, landmarks, measures, warnings }
}

// ----------------------------------------------------------------- gestures --

/// Landmark ids of one gesture, in temporal order.
pub const GESTURE_POINTS: &[(&str, &str)] = &[
    ("GONS", "gesture onset"),
    ("PVEL", "peak velocity (formation)"),
    ("NONS", "target onset"),
    ("MAXC", "maximum constriction"),
    ("NOFFS", "target offset (release)"),
    ("PVEL2", "peak velocity (release)"),
    ("GOFFS", "gesture offset"),
];

/// Measure suffixes of one gesture.
pub const GESTURE_MEASURES: &[(&str, &str, &str)] = &[
    ("dur_ms", "gesture duration (GOFFS − GONS)", "ms"),
    ("form_ms", "formation duration (NONS − GONS)", "ms"),
    ("plateau_ms", "constriction plateau (NOFFS − NONS)", "ms"),
    ("release_ms", "release duration (GOFFS − NOFFS)", "ms"),
    ("ttpv_ms", "time to peak velocity (PVEL − GONS)", "ms"),
    ("disp_mm", "displacement (GONS → MAXC)", "mm"),
    ("pvel_mm_s", "peak velocity (formation)", "mm/s"),
    ("stiff_1_s", "stiffness (peak velocity / displacement)", "1/s"),
];

/// Lags of each later gesture relative to the first.
pub const LAG_MEASURES: &[(&str, &str, &str)] = &[
    ("onset_lag_ms", "onset lag (GONS − GONS of the first gesture)", "ms"),
    ("target_lag_ms", "target lag (NONS − NONS of the first gesture)", "ms"),
    ("maxc_lag_ms", "constriction lag (MAXC − MAXC of the first gesture)", "ms"),
];

fn gesture_measures(label: &str, pts: Option<&[ProtocolLandmark]>, min_disp: f64) -> Vec<Measure> {
    let g = |k: usize| pts.and_then(|p| p.get(k));
    let dt = |a: usize, b: usize| Some((g(b)?.time_s - g(a)?.time_s) * 1000.0);
    let disp = (|| Some((g(3)?.position - g(0)?.position).abs()))();
    let pvel = g(1).map(|p| p.velocity.abs());
    // stiffness of a near-zero displacement is meaningless (and explodes)
    let stiff = match (pvel, disp) {
        (Some(v), Some(d)) if d >= min_disp.max(1e-6) => Some(v / d),
        _ => None,
    };
    let values = [dt(0, 6), dt(0, 2), dt(2, 4), dt(4, 6), dt(0, 1), disp, pvel, stiff];
    GESTURE_MEASURES.iter().zip(values).map(|((suffix, l, u), v)| measure(&format!("{label}_{suffix}"), &format!("{label} {l}"), u, v)).collect()
}

/// Velocity-threshold landmarks of one gesture.
fn one_gesture(trial: &Trial, occ: &Occurrence, spec: &GestureSpec, cfg: &GesturesConfig, kin: &KinematicsConfig, fixed: &[ProtocolLandmark], warnings: &mut Vec<String>) -> Option<Vec<ProtocolLandmark>> {
    let label = if spec.label.trim().is_empty() { spec.sensor.clone() } else { spec.label.clone() };
    let dim = if spec.dim == Dim::Tangential { Dim::Z } else { spec.dim };
    let Some(sensor) = trial.sensor(&spec.sensor).map(|s| s.name.clone()) else {
        warnings.push(format!("{label}: no sensor named {}", spec.sensor));
        return None;
    };
    let Some(tr) = Track::load(trial, &sensor, &spec.reference, dim, kin) else {
        warnings.push(format!("{label}: the signal could not be read"));
        return None;
    };
    let n = tr.len();
    let sign = if spec.direction == Polarity::Min { -1.0 } else { 1.0 };
    let pad = cfg.pad_ms / 1000.0;
    let (lo, hi) = (tr.i(occ.start_s - pad), tr.i(occ.end_s + pad));
    if hi <= lo + 4 {
        warnings.push(format!("{label}: the word lies outside the EMA recording"));
        return None;
    }
    let sp = |i: usize| sign * tr.pos[i];
    let sv = |i: usize| sign * tr.vel[i];
    let id = |p: &str| format!("{label}.{p}");

    // MAXC: the constriction extremum
    let maxc = if let Some(l) = pinned(fixed, &id("MAXC")) {
        tr.i(l.time_s)
    } else {
        let span = (0.15 * tr.fs) as usize;
        let cands: Vec<usize> = (lo + 1..hi)
            .filter(|&i| sp(i) >= sp(i - 1) && sp(i) > sp(i + 1))
            .filter(|&i| {
                let a = i.saturating_sub(span).max(lo);
                let before = (a..=i).map(sp).fold(f64::INFINITY, f64::min);
                let after = (i..=(i + span).min(hi)).map(sp).fold(f64::INFINITY, f64::min);
                sp(i) - before >= cfg.min_amplitude || sp(i) - after >= cfg.min_amplitude
            })
            .collect();
        let pick = if spec.anchor_phone >= 1 && spec.anchor_phone <= occ.phones.len() {
            let ph = &occ.phones[spec.anchor_phone - 1];
            let c = (ph.start_s + ph.end_s) / 2.0;
            cands.iter().copied().min_by(|&a, &b| (tr.t(a) - c).abs().partial_cmp(&(tr.t(b) - c).abs()).unwrap())
        } else {
            let (a, b) = (tr.i(occ.start_s), tr.i(occ.end_s));
            cands.iter().copied().filter(|&i| i >= a && i <= b).max_by(|&a, &b| sp(a).partial_cmp(&sp(b)).unwrap())
        };
        match pick {
            Some(i) => i,
            None => {
                warnings.push(format!("{label}: no constriction of at least {} mm found", cfg.min_amplitude));
                return None;
            }
        }
    };
    // formation: back to where the movement toward the constriction starts
    let mut a = maxc;
    while a > lo && sv(a - 1) > 0.0 {
        a -= 1;
    }
    let mut b = maxc;
    while b < hi.min(n - 1) && sv(b + 1) < 0.0 {
        b += 1;
    }
    let pvel_i = argmax_by(a, maxc, sv);
    let pvel2_i = argmax_by(maxc, b, |i| -sv(i));
    let thr1 = cfg.threshold * sv(pvel_i).max(0.0);
    let thr2 = cfg.threshold * (-sv(pvel2_i)).max(0.0);
    let s_vel: Vec<f64> = tr.vel.iter().map(|v| sign * v).collect();
    let neg_vel: Vec<f64> = s_vel.iter().map(|v| -v).collect();
    // GONS: last crossing of thr1 before PVEL; NONS: first after
    let mut i = pvel_i;
    while i > a && s_vel[i - 1] > thr1 {
        i -= 1;
    }
    let gons = if i > a { crossing(&s_vel, i - 1, thr1, tr.fs) } else { tr.t(a) };
    let mut j = pvel_i;
    while j < maxc && s_vel[j + 1] > thr1 {
        j += 1;
    }
    let nons = crossing(&s_vel, j, thr1, tr.fs).min(tr.t(maxc));
    let mut k = pvel2_i;
    while k > maxc && neg_vel[k - 1] > thr2 {
        k -= 1;
    }
    let noffs = crossing(&neg_vel, k.saturating_sub(1).max(maxc), thr2, tr.fs).max(tr.t(maxc));
    let mut m = pvel2_i;
    while m < b && neg_vel[m + 1] > thr2 {
        m += 1;
    }
    let goffs = crossing(&neg_vel, m, thr2, tr.fs);
    let maxc_t = if maxc > lo && maxc < hi { refine(&tr.pos, maxc, tr.fs).0 } else { tr.t(maxc) };
    let times = [
        (gons, "velocity 20 %", Landmark::Position),
        (if pvel_i > a && pvel_i < maxc { refine(&tr.vel, pvel_i, tr.fs).0 } else { tr.t(pvel_i) }, "peak", Landmark::Velocity),
        (nons, "velocity 20 %", Landmark::Position),
        (maxc_t, "extremum", Landmark::Position),
        (noffs, "velocity 20 %", Landmark::Position),
        (if pvel2_i > maxc && pvel2_i < b { refine(&tr.vel, pvel2_i, tr.fs).0 } else { tr.t(pvel2_i) }, "peak", Landmark::Velocity),
        (goffs, "velocity 20 %", Landmark::Position),
    ];
    let pct = format!("velocity {:.0} %", cfg.threshold * 100.0);
    let pts = GESTURE_POINTS
        .iter()
        .zip(times)
        .map(|((p, l), (t, crit, curve))| {
            let crit = if crit == "velocity 20 %" { pct.as_str() } else { crit };
            match pinned(fixed, &id(p)) {
                Some(f) => mk(&id(p), &format!("{label} {l}"), &sensor, &spec.reference, dim, curve, &tr, f.time_s, "", true),
                None => mk(&id(p), &format!("{label} {l}"), &sensor, &spec.reference, dim, curve, &tr, t, crit, false),
            }
        })
        .collect();
    Some(pts)
}

/// Velocity-threshold gestural landmarks of every configured gesture, their
/// measures, and the lags of later gestures relative to the first.
pub fn gestures(trial: &Trial, occ: &Occurrence, cfg: &GesturesConfig, kin: &KinematicsConfig, fixed: &[ProtocolLandmark]) -> ProtocolResult {
    let mut warnings = vec![];
    let mut landmarks = vec![];
    let mut measures = vec![];
    let mut firsts: Option<Vec<ProtocolLandmark>> = None;
    for (gi, spec) in cfg.gestures.iter().enumerate() {
        let label = if spec.label.trim().is_empty() { spec.sensor.clone() } else { spec.label.clone() };
        let pts = one_gesture(trial, occ, spec, cfg, kin, fixed, &mut warnings);
        if let (Some(p), true) = (&pts, cfg.rule_spread_ms > 0.0) {
            // onset / offset at 10 % and 30 % of peak velocity
            let mut spread = 0.0f64;
            for th in [0.1, 0.3] {
                let alt = GesturesConfig { threshold: th, ..cfg.clone() };
                if let Some(q) = one_gesture(trial, occ, spec, &alt, kin, fixed, &mut vec![]) {
                    spread = spread.max((q[0].time_s - p[0].time_s).abs()).max((q[6].time_s - p[6].time_s).abs());
                }
            }
            if spread * 1000.0 > cfg.rule_spread_ms {
                warnings.push(format!("{label}: onset or offset moves {:.0} ms with the threshold (10–30 %): a slow start or end", spread * 1000.0));
            }
        }
        measures.extend(gesture_measures(&label, pts.as_deref(), cfg.min_amplitude));
        if gi == 0 {
            firsts = pts.clone();
        } else {
            let lag = |k: usize| Some((pts.as_ref()?.get(k)?.time_s - firsts.as_ref()?.get(k)?.time_s) * 1000.0);
            let values = [lag(0), lag(2), lag(3)];
            measures.extend(LAG_MEASURES.iter().zip(values).map(|((suffix, l, u), v)| measure(&format!("{label}_{suffix}"), &format!("{label} {l}"), u, v)));
        }
        if let Some(p) = pts {
            landmarks.extend(p);
        }
    }
    ProtocolResult { kind: ProtocolKind::Gestures, landmarks, measures, warnings }
}

// -------------------------------------------------------------------- tests --

#[cfg(test)]
mod tests {
    use super::*;
    use crate::locate::{Engine, PhoneSpan};
    use crate::trial::Audio;

    /// A synthetic /tadat/: two jaw openings (vowels) and a tongue-tip
    /// raising for the medial consonant.
    fn synthetic() -> (Trial, Occurrence) {
        let fs = 250.0;
        let n = 500; // 2 s
        let bump = |t: f64, c: f64, w: f64| (-(t - c).powi(2) / (2.0 * w * w)).exp();
        let jaw: Vec<f64> = (0..n).map(|i| { let t = i as f64 / fs; -5.0 - 10.0 * bump(t, 0.80, 0.07) - 8.0 * bump(t, 1.20, 0.07) }).collect();
        let tt: Vec<f64> = (0..n).map(|i| { let t = i as f64 / fs; jaw[i] + 2.0 + 6.0 * bump(t, 1.02, 0.05) }).collect();
        let zeros = vec![0.0; n];
        let trial = Trial {
            id: "syn".into(),
            variable: "syn".into(),
            audio: Audio { fs: 16000.0, samples: vec![0.0; 32000] },
            sensors: vec![
                Sensor { name: "JAW".into(), fs, columns: vec![zeros.clone(), zeros.clone(), jaw] },
                Sensor { name: "TT".into(), fs, columns: vec![zeros.clone(), zeros.clone(), tt] },
            ],
        };
        let ph = |s: &str, a: f64, b: f64| PhoneSpan { symbol: s.into(), start_s: a, end_s: b, prob: 1.0 };
        let occ = Occurrence {
            word_index: 0,
            word: "tadat".into(),
            start_s: 0.65,
            end_s: 1.40,
            raw_start_s: 0.65,
            raw_end_s: 1.40,
            phrase_start_s: 0.2,
            phrase_end_s: 1.8,
            confidence: 1.0,
            phones: vec![ph("t", 0.65, 0.72), ph("a", 0.72, 0.92), ph("d", 0.92, 1.10), ph("a", 1.10, 1.30), ph("t", 1.30, 1.40)],
            engine: Engine::Phonetic,
        };
        (trial, occ)
    }

    #[test]
    fn jaw_cycle_on_synthetic_word() {
        let (trial, occ) = synthetic();
        let kin = KinematicsConfig::default();
        let r = jaw_cycle(&trial, &occ, &JawCycleConfig::default(), &kin, &[]);
        let t = |id: &str| r.landmark(id).unwrap().time_s;
        // maximum displacement at the first opening, V1 before it, V2 between the openings
        assert!((t("MaxDisp") - 0.80).abs() < 0.01, "{}", t("MaxDisp"));
        assert!(t("V1") < 0.70 && t("V1") > 0.45, "V1 {}", t("V1"));
        assert!(t("V2") > 0.90 && t("V2") < 1.10, "V2 {}", t("V2"));
        assert!(t("PV1") > t("V1") && t("PV1") < t("MaxDisp"));
        assert!(t("PV2") > t("MaxDisp") && t("PV2") < t("V2"));
        assert!(t("TTonset") > t("V1") && t("TTonset") < 1.02, "TT {}", t("TTonset"));
        let jvc = r.measure("JVC_Dur_ms").unwrap();
        assert!((jvc - (t("V2") - t("V1")) * 1000.0).abs() < 1e-6);
        assert!((r.measure("MaxJ_Disp_mm").unwrap() + 15.0).abs() < 0.3);
        let phi = r.measure("TT_Pha_deg").unwrap();
        assert!(phi > 90.0 && phi < 330.0, "{phi}");
    }

    #[test]
    fn manual_landmark_is_kept_and_measures_follow() {
        let (trial, occ) = synthetic();
        let kin = KinematicsConfig::default();
        let auto = jaw_cycle(&trial, &occ, &JawCycleConfig::default(), &kin, &[]);
        let mut v1 = auto.landmark("V1").unwrap().clone();
        v1.time_s -= 0.02;
        v1.manual = true;
        let r = jaw_cycle(&trial, &occ, &JawCycleConfig::default(), &kin, &[v1.clone()]);
        assert!((r.landmark("V1").unwrap().time_s - v1.time_s).abs() < 1e-12);
        assert!(r.landmark("V1").unwrap().manual);
        let jvc = r.measure("JVC_Dur_ms").unwrap();
        assert!((jvc - auto.measure("JVC_Dur_ms").unwrap() - 20.0).abs() < 1e-6);
    }

    #[test]
    fn phase_angle_quadrants() {
        // one cosine cycle: top at 0, bottom at 0.5 s
        let fs = 1000.0;
        let pos: Vec<f64> = (0..=1000).map(|i| (2.0 * std::f64::consts::PI * i as f64 / 1000.0).cos()).collect();
        let vel: Vec<f64> = (0..=1000).map(|i| -2.0 * std::f64::consts::PI * (2.0 * std::f64::consts::PI * i as f64 / 1000.0).sin()).collect();
        let tr = Track { fs, pos, vel };
        for (t, want) in [(0.0, 0.0), (0.25, 90.0), (0.5, 180.0), (0.75, 270.0), (0.125, 45.0), (0.501, 180.36), (0.499, 179.64)] {
            let phi = phase_angle(&tr, 0.0, 0.5, 1.0, t).unwrap();
            assert!((phi - want).abs() < 0.5, "t {t}: {phi} vs {want}");
        }
        // the bottom is a continuous 180°, whichever half the onset was assigned to
        assert!((phase_angle(&tr, 0.0, 0.499, 1.0, 0.4995).unwrap() - 179.8).abs() < 0.5);
        assert!((phase_angle(&tr, 0.0, 0.501, 1.0, 0.5005).unwrap() - 180.2).abs() < 0.5);
    }

    #[test]
    fn gestures_on_synthetic_word() {
        let (trial, occ) = synthetic();
        let kin = KinematicsConfig::default();
        let r = gestures(&trial, &occ, &GesturesConfig::default(), &kin, &[]);
        assert!(r.warnings.is_empty(), "{:?}", r.warnings);
        let t = |id: &str| r.landmark(id).unwrap().time_s;
        // jaw opening of vowel 1 (anchor phone 2) and tongue-tip closure of the medial consonant (phone 3)
        assert!((t("JAW.MAXC") - 0.80).abs() < 0.01);
        assert!((t("TT.MAXC") - 1.02).abs() < 0.02, "{}", t("TT.MAXC"));
        for g in ["JAW", "TT"] {
            let ts: Vec<f64> = GESTURE_POINTS.iter().map(|(p, _)| t(&format!("{g}.{p}"))).collect();
            assert!(ts.windows(2).all(|w| w[0] <= w[1] + 1e-9), "{g}: {ts:?}");
        }
        let lag = r.measure("TT_onset_lag_ms").unwrap();
        assert!((lag - (t("TT.GONS") - t("JAW.GONS")) * 1000.0).abs() < 1e-6);
    }
}
