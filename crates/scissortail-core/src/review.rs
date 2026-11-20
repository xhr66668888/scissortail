//! Review across trials: protocol timing measures that lie far from the
//! other repetitions of the same condition (same session, same condition code
//! in the file name, same target word) mark a trial "please check".
//!
//! Most labelling errors of the jaw-cycle protocol (a wrong jaw cycle, the
//! other of two tongue-tip raisings) give values that are plausible on their
//! own but stand out among the 10–15 repetitions of their condition. On the
//! Masapollo & Nittrouer recordings this check, together with the protocols'
//! own rule-disagreement warnings, flags about 13 % of tokens and catches about
//! two thirds of the tokens off by more than 30 ms (held-out talkers).
//! It works for any sensor and any protocol measure in ms or degrees.

use std::collections::HashMap;
use std::path::Path;

use crate::analysis::{apply_protocol_status, TrialResult, TrialStatus};
use crate::protocol::ReviewConfig;
use crate::session::condition_key;

/// Which repetitions a trial's occurrence is compared with.
pub fn group_key(r: &TrialResult, word: &str) -> String {
    let session = Path::new(&r.file).parent().map(|p| p.display().to_string()).unwrap_or_default();
    format!("{session}\u{1f}{}\u{1f}{word}", condition_key(&r.trial_id))
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { (v[n / 2 - 1] + v[n / 2]) / 2.0 }
}

/// Recomputes the review notes of every result and its "please check"
/// status; returns the indices of the results whose status or message
/// changed. Trials checked or edited by hand are compared against but never
/// flagged.
pub fn review_outliers(results: &mut [&mut TrialResult], cfg: &ReviewConfig) -> Vec<usize> {
    let before: Vec<(TrialStatus, String)> = results.iter().map(|r| (r.status, r.message.clone())).collect();
    // (group, measure) → [(result, value)]
    let mut groups: HashMap<(String, String), Vec<(usize, f64)>> = HashMap::new();
    let mut labels: HashMap<String, (String, String)> = HashMap::new();
    for (i, r) in results.iter().enumerate() {
        if matches!(r.status, TrialStatus::Error | TrialStatus::NotFound | TrialStatus::Excluded) {
            continue;
        }
        for o in &r.occurrences {
            let Some(p) = &o.protocol else { continue };
            let key = group_key(r, &o.occurrence.word);
            for m in &p.measures {
                let (Some(v), true) = (m.value, m.unit == "ms" || m.unit == "°") else { continue };
                groups.entry((key.clone(), m.id.clone())).or_default().push((i, v));
                labels.entry(m.id.clone()).or_insert_with(|| (m.label.clone(), m.unit.clone()));
            }
        }
    }
    let mut notes: Vec<Vec<String>> = vec![vec![]; results.len()];
    if cfg.outlier_z > 0.0 {
        for ((_, id), items) in &groups {
            if items.len() < cfg.min_repetitions.max(3) {
                continue;
            }
            let mut vals: Vec<f64> = items.iter().map(|x| x.1).collect();
            let med = median(&mut vals);
            let mut dev: Vec<f64> = items.iter().map(|x| (x.1 - med).abs()).collect();
            let mad = median(&mut dev);
            let (label, unit) = labels.get(id).cloned().unwrap_or_default();
            let floor = if unit == "°" { cfg.floor_deg } else { cfg.floor_ms };
            let scale = (1.4826 * mad).max(floor).max(1e-9);
            for &(i, v) in items {
                let r = &results[i];
                if r.reviewed || r.occurrences.iter().any(|o| o.edited) {
                    continue;
                }
                if (v - med).abs() / scale > cfg.outlier_z {
                    let short = label.split(" (").next().unwrap_or(&label);
                    notes[i].push(format!("{short} {v:.0} {unit} is far from the other {} repetitions of this condition (median {med:.0} {unit})", items.len() - 1));
                }
            }
        }
    }
    for (i, r) in results.iter_mut().enumerate() {
        notes[i].sort();
        r.review = std::mem::take(&mut notes[i]);
        apply_protocol_status(r);
    }
    results.iter().enumerate().filter(|(i, r)| before[*i] != (r.status, r.message.clone())).map(|(i, _)| i).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::OccurrenceResult;
    use crate::locate::{Engine, Occurrence};
    use crate::protocol::{Measure, ProtocolKind, ProtocolResult};

    fn trial(id: &str, jvc: f64) -> TrialResult {
        let occ = Occurrence { word_index: 0, word: "tadat".into(), start_s: 0.0, end_s: 1.0, raw_start_s: 0.0, raw_end_s: 1.0, phrase_start_s: 0.0, phrase_end_s: 1.0, confidence: 1.0, phones: vec![], engine: Engine::Phonetic };
        let p = ProtocolResult { kind: ProtocolKind::JawCycle, landmarks: vec![], measures: vec![Measure { id: "JVC_Dur_ms".into(), label: "Jaw vowel-cycle duration (V2 − V1)".into(), unit: "ms".into(), value: Some(jvc) }], warnings: vec![] };
        TrialResult {
            trial_id: id.into(),
            file: format!("/data/s1/{id}.mat"),
            status: TrialStatus::Ok,
            message: String::new(),
            audio_duration_s: 1.0,
            ema_duration_s: 1.0,
            occurrences: vec![OccurrenceResult { occurrence: occ, kinematics: vec![], edited: false, protocol: Some(p) }],
            reviewed: false,
            note: String::new(),
            review: vec![],
        }
    }

    #[test]
    fn condition_keys() {
        assert_eq!(condition_key("T10_T292_SSad_1_NH_A_6"), "SSad_NH_A");
        assert_eq!(condition_key("T0003"), "");
    }

    #[test]
    fn flags_the_outlier_of_a_condition_only() {
        let mut v: Vec<TrialResult> = (0..10).map(|k| trial(&format!("T1_T{k}_SSad_1"), 300.0 + k as f64)).collect();
        v.push(trial("T1_T99_SSad_2", 520.0));
        // another condition with different values is not compared with these
        v.extend((0..6).map(|k| trial(&format!("T1_T{}_FUad_1", 100 + k), 180.0 + k as f64)));
        let mut refs: Vec<&mut TrialResult> = v.iter_mut().collect();
        let changed = review_outliers(&mut refs, &ReviewConfig::default());
        assert_eq!(changed, vec![10]);
        assert_eq!(v[10].status, TrialStatus::LowConfidence);
        assert!(v[10].message.contains("520"), "{}", v[10].message);
        // once checked by hand it is no longer flagged
        v[10].reviewed = true;
        let mut refs: Vec<&mut TrialResult> = v.iter_mut().collect();
        review_outliers(&mut refs, &ReviewConfig::default());
        assert_eq!(v[10].status, TrialStatus::Ok);
        assert!(v[10].review.is_empty());
    }
}
