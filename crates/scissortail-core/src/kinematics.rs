//! Articulator kinematics: smoothing, regression-based derivatives
//! (Savitzky–Golay) and peak extraction with sub-sample (quadratic
//! regression) refinement.

use serde::{Deserialize, Serialize};

use crate::dsp::{lowpass_filtfilt, odd_window, polyfit, savgol};
use crate::trial::Sensor;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Dim {
    X,
    Y,
    Z,
    /// 3-D: position = distance travelled from the window start,
    /// velocity = tangential speed, acceleration = magnitude.
    Tangential,
}

impl Dim {
    pub fn label(&self) -> &'static str {
        match self {
            Dim::X => "x",
            Dim::Y => "y",
            Dim::Z => "z",
            Dim::Tangential => "3d",
        }
    }
    pub fn all() -> [Dim; 4] {
        [Dim::X, Dim::Y, Dim::Z, Dim::Tangential]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Polarity {
    /// Highest value (e.g. tongue-tip raising for an alveolar closure).
    Max,
    /// Lowest value (e.g. jaw lowering for an open vowel).
    Min,
    /// Largest absolute deviation from the window median.
    Abs,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorRule {
    pub name: String,
    pub enabled: bool,
    pub position_peak: Polarity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct KinematicsConfig {
    /// Zero-phase Butterworth low-pass applied to positions (Hz, 0 = off).
    pub lowpass_hz: f64,
    /// Savitzky–Golay regression window (ms) for velocity / acceleration.
    pub sg_window_ms: f64,
    pub sg_order: usize,
    /// Extend the acoustic word interval by these margins (ms) for peak picking.
    pub pad_pre_ms: f64,
    pub pad_post_ms: f64,
    pub dims: Vec<Dim>,
    pub sensors: Vec<SensorRule>,
}

impl Default for KinematicsConfig {
    fn default() -> Self {
        KinematicsConfig {
            lowpass_hz: 20.0,
            sg_window_ms: 40.0,
            sg_order: 3,
            pad_pre_ms: 0.0,
            pad_post_ms: 0.0,
            dims: vec![Dim::Z, Dim::X, Dim::Tangential],
            sensors: vec![],
        }
    }
}

impl KinematicsConfig {
    /// Rule for a sensor (defaults: TT → max, JAW → min, others → abs).
    pub fn rule_for(&self, sensor: &Sensor) -> SensorRule {
        if let Some(r) = self.sensors.iter().find(|r| r.name.eq_ignore_ascii_case(&sensor.name)) {
            return r.clone();
        }
        let n = sensor.name.to_ascii_uppercase();
        let polarity = if n.starts_with("TT") || n.contains("TIP") || n.starts_with("TB") || n.starts_with("TD") || n.contains("TONG") {
            Polarity::Max
        } else if n.contains("JAW") || n == "LI" || n.contains("CHIN") || n == "LL" {
            Polarity::Min
        } else {
            Polarity::Abs
        };
        SensorRule { name: sensor.name.clone(), enabled: !sensor.looks_like_reference(), position_peak: polarity }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Peak {
    pub time_s: f64,
    pub value: f64,
    /// True when the extremum sits on the analysis-window boundary (the true
    /// peak may lie outside the window).
    pub at_edge: bool,
    /// Placed (dragged) by the researcher instead of detected automatically.
    #[serde(default)]
    pub manual: bool,
}

/// The three landmarks extracted for every sensor dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Landmark {
    Position,
    Velocity,
    Acceleration,
}

impl Landmark {
    pub fn all() -> [Landmark; 3] {
        [Landmark::Position, Landmark::Velocity, Landmark::Acceleration]
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KinematicResult {
    pub sensor: String,
    pub dim: Dim,
    pub window_start_s: f64,
    pub window_end_s: f64,
    pub position: Option<Peak>,
    pub velocity: Option<Peak>,
    pub acceleration: Option<Peak>,
    /// Movement range within the window (max − min of position).
    pub range: f64,
}

impl KinematicResult {
    pub fn peak(&self, which: Landmark) -> Option<&Peak> {
        match which {
            Landmark::Position => self.position.as_ref(),
            Landmark::Velocity => self.velocity.as_ref(),
            Landmark::Acceleration => self.acceleration.as_ref(),
        }
    }

    pub fn peak_mut(&mut self, which: Landmark) -> &mut Option<Peak> {
        match which {
            Landmark::Position => &mut self.position,
            Landmark::Velocity => &mut self.velocity,
            Landmark::Acceleration => &mut self.acceleration,
        }
    }
}

/// Smoothed position, velocity and acceleration of one sensor dimension.
pub struct Derivatives {
    pub fs: f64,
    pub pos: Vec<f64>,
    pub vel: Vec<f64>,
    pub acc: Vec<f64>,
}

impl Derivatives {
    pub fn signal(&self, which: Landmark) -> &[f64] {
        match which {
            Landmark::Position => &self.pos,
            Landmark::Velocity => &self.vel,
            Landmark::Acceleration => &self.acc,
        }
    }
}

pub fn derivatives(sensor: &Sensor, dim: Dim, cfg: &KinematicsConfig) -> Option<Derivatives> {
    let fs = sensor.fs;
    let dt = 1.0 / fs;
    let w = odd_window(cfg.sg_window_ms / 1000.0, fs, cfg.sg_order + 2);
    let smooth = |c: &[f64]| if cfg.lowpass_hz > 0.0 { lowpass_filtfilt(c, cfg.lowpass_hz, fs, 4) } else { c.to_vec() };
    match dim {
        Dim::X | Dim::Y | Dim::Z => {
            let idx = match dim { Dim::X => 0, Dim::Y => 1, _ => 2 };
            let col = sensor.columns.get(idx)?;
            let pos = smooth(col);
            let vel = savgol(&pos, w, cfg.sg_order, 1, dt);
            let acc = savgol(&pos, w, cfg.sg_order, 2, dt);
            Some(Derivatives { fs, pos, vel, acc })
        }
        Dim::Tangential => {
            if sensor.columns.len() < 3 {
                return None;
            }
            let p: Vec<Vec<f64>> = (0..3).map(|i| smooth(&sensor.columns[i])).collect();
            let v: Vec<Vec<f64>> = p.iter().map(|c| savgol(c, w, cfg.sg_order, 1, dt)).collect();
            let a: Vec<Vec<f64>> = p.iter().map(|c| savgol(c, w, cfg.sg_order, 2, dt)).collect();
            let n = p[0].len();
            let speed: Vec<f64> = (0..n).map(|i| (v[0][i].powi(2) + v[1][i].powi(2) + v[2][i].powi(2)).sqrt()).collect();
            let accm: Vec<f64> = (0..n).map(|i| (a[0][i].powi(2) + a[1][i].powi(2) + a[2][i].powi(2)).sqrt()).collect();
            // cumulative path length
            let mut dist = vec![0.0; n];
            for i in 1..n {
                dist[i] = dist[i - 1] + ((p[0][i] - p[0][i - 1]).powi(2) + (p[1][i] - p[1][i - 1]).powi(2) + (p[2][i] - p[2][i - 1]).powi(2)).sqrt();
            }
            Some(Derivatives { fs, pos: dist, vel: speed, acc: accm })
        }
    }
}

/// Index of the extremum of `y[lo..=hi]` according to `mode`
/// (`Max`, `Min`, or `Abs` = largest |y − reference|).
fn extremum(y: &[f64], lo: usize, hi: usize, mode: Polarity, reference: f64) -> usize {
    let mut best = lo;
    for i in lo..=hi {
        let better = match mode {
            Polarity::Max => y[i] > y[best],
            Polarity::Min => y[i] < y[best],
            Polarity::Abs => (y[i] - reference).abs() > (y[best] - reference).abs(),
        };
        if better {
            best = i;
        }
    }
    best
}

/// Refines a discrete extremum by least-squares quadratic regression over
/// ±2 samples; returns (time, value).
fn refine(y: &[f64], i: usize, fs: f64) -> (f64, f64) {
    let lo = i.saturating_sub(2);
    let hi = (i + 2).min(y.len() - 1);
    if hi - lo < 2 {
        return (i as f64 / fs, y[i]);
    }
    let t: Vec<f64> = (lo..=hi).map(|k| k as f64 - i as f64).collect();
    let c = polyfit(&t, &y[lo..=hi], 2);
    if c[2].abs() < 1e-12 {
        return (i as f64 / fs, y[i]);
    }
    let dx = (-c[1] / (2.0 * c[2])).clamp(-1.0, 1.0);
    let v = c[0] + c[1] * dx + c[2] * dx * dx;
    ((i as f64 + dx) / fs, v)
}

fn make_peak(y: &[f64], lo: usize, hi: usize, mode: Polarity, reference: f64, fs: f64) -> Peak {
    let i = extremum(y, lo, hi, mode, reference);
    let at_edge = i == lo || i == hi;
    let (time_s, value) = if at_edge { (i as f64 / fs, y[i]) } else { refine(y, i, fs) };
    Peak { time_s, value, at_edge, manual: false }
}

/// Linear interpolation of `y` (sampled at `fs`) at time `t`.
fn value_at(y: &[f64], fs: f64, t: f64) -> f64 {
    let x = (t * fs).clamp(0.0, (y.len() - 1) as f64);
    let i = x.floor() as usize;
    let j = (i + 1).min(y.len() - 1);
    y[i] + (y[j] - y[i]) * (x - i as f64)
}

/// A landmark placed by hand at `time_s`. With `snap_s` > 0 it moves to the
/// nearest local extremum of its signal within ±`snap_s` (refined by quadratic
/// regression, like the automatic peaks); otherwise it stays where it was
/// dropped and takes the signal value there.
pub fn manual_peak(sensor: &Sensor, dim: Dim, which: Landmark, time_s: f64, snap_s: f64, cfg: &KinematicsConfig) -> Option<Peak> {
    let d = derivatives(sensor, dim, cfg)?;
    let y = d.signal(which);
    let n = y.len();
    if n < 3 {
        return None;
    }
    let t = time_s.clamp(0.0, (n - 1) as f64 / d.fs);
    let c = t * d.fs;
    let lo = (c - snap_s * d.fs).floor().max(1.0) as usize;
    let hi = ((c + snap_s * d.fs).ceil().max(0.0) as usize).min(n - 2);
    let snapped = (snap_s > 0.0 && lo <= hi)
        .then(|| {
            (lo..=hi)
                .filter(|&i| {
                    let (a, b) = (y[i] - y[i - 1], y[i + 1] - y[i]);
                    a * b <= 0.0 && (a != 0.0 || b != 0.0)
                })
                .min_by(|&i, &j| (i as f64 - c).abs().partial_cmp(&(j as f64 - c).abs()).unwrap())
        })
        .flatten();
    let (time_s, value) = match snapped {
        Some(i) => refine(y, i, d.fs),
        None => (t, value_at(y, d.fs, t)),
    };
    Some(Peak { time_s, value, at_edge: false, manual: true })
}

/// Position / velocity / acceleration peaks of one sensor dimension inside
/// the window [t0, t1] seconds.
pub fn analyze_window(sensor: &Sensor, dim: Dim, polarity: Polarity, t0: f64, t1: f64, cfg: &KinematicsConfig) -> Option<KinematicResult> {
    let d = derivatives(sensor, dim, cfg)?;
    let n = d.pos.len();
    if n < 3 {
        return None;
    }
    let lo = ((t0 * d.fs).floor().max(0.0) as usize).min(n - 1);
    let hi = ((t1 * d.fs).ceil().max(0.0) as usize).min(n - 1);
    if hi <= lo + 1 {
        return None;
    }
    let seg = &d.pos[lo..=hi];
    let mut sorted = seg.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted[sorted.len() / 2];
    let range = sorted[sorted.len() - 1] - sorted[0];
    let pos_mode = if dim == Dim::Tangential { Polarity::Max } else { polarity };
    let position = make_peak(&d.pos, lo, hi, pos_mode, median, d.fs);
    let speed_mode = if dim == Dim::Tangential { Polarity::Max } else { Polarity::Abs };
    let velocity = make_peak(&d.vel, lo, hi, speed_mode, 0.0, d.fs);
    let acceleration = make_peak(&d.acc, lo, hi, speed_mode, 0.0, d.fs);
    Some(KinematicResult {
        sensor: sensor.name.clone(),
        dim,
        window_start_s: lo as f64 / d.fs,
        window_end_s: hi as f64 / d.fs,
        position: Some(position),
        velocity: Some(velocity),
        acceleration: Some(acceleration),
        range,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_sinusoid_peaks() {
        let fs = 250.0;
        let n = 500;
        // z = sin(2π·2t): max at t = 0.125 s, |v| max at t = 0, 0.25 …
        let z: Vec<f64> = (0..n).map(|i| (2.0 * std::f64::consts::PI * 2.0 * i as f64 / fs).sin()).collect();
        let sensor = Sensor { name: "TT".into(), fs, columns: vec![vec![0.0; n], vec![0.0; n], z] };
        let cfg = KinematicsConfig { lowpass_hz: 0.0, ..Default::default() };
        let r = analyze_window(&sensor, Dim::Z, Polarity::Max, 0.05, 0.2, &cfg).unwrap();
        let p = r.position.unwrap();
        assert!((p.time_s - 0.125).abs() < 0.002, "{}", p.time_s);
        assert!((p.value - 1.0).abs() < 1e-3);
        // velocity extremum in [0.05, 0.2] is at the window start (decreasing speed) -> edge
        let v = r.velocity.unwrap();
        assert!(v.at_edge);
        let r2 = analyze_window(&sensor, Dim::Z, Polarity::Max, 0.15, 0.35, &cfg).unwrap();
        let v2 = r2.velocity.unwrap();
        assert!((v2.time_s - 0.25).abs() < 0.003, "{}", v2.time_s);
        assert!((v2.value + 4.0 * std::f64::consts::PI).abs() < 0.05, "{}", v2.value);
    }

    #[test]
    fn manual_peak_snaps_to_nearest_extremum() {
        let fs = 250.0;
        let n = 500;
        let z: Vec<f64> = (0..n).map(|i| (2.0 * std::f64::consts::PI * 2.0 * i as f64 / fs).sin()).collect();
        let sensor = Sensor { name: "TT".into(), fs, columns: vec![vec![0.0; n], vec![0.0; n], z] };
        let cfg = KinematicsConfig { lowpass_hz: 0.0, ..Default::default() };
        // dropped 20 ms after the position maximum at 0.125 s: snaps back to it
        let p = manual_peak(&sensor, Dim::Z, Landmark::Position, 0.145, 0.03, &cfg).unwrap();
        assert!(p.manual && !p.at_edge);
        assert!((p.time_s - 0.125).abs() < 0.002, "{}", p.time_s);
        assert!((p.value - 1.0).abs() < 1e-3);
        // without snapping it stays where it was dropped
        let q = manual_peak(&sensor, Dim::Z, Landmark::Position, 0.145, 0.0, &cfg).unwrap();
        assert!((q.time_s - 0.145).abs() < 1e-9);
        assert!((q.value - (2.0 * std::f64::consts::PI * 2.0 * 0.145).sin()).abs() < 1e-3);
        // no extremum within reach: stays put
        let r = manual_peak(&sensor, Dim::Z, Landmark::Position, 0.2, 0.01, &cfg).unwrap();
        assert!((r.time_s - 0.2).abs() < 1e-9);
    }
}
