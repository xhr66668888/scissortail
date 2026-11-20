//! CSV export of analysis results (long format: one row per trial ×
//! occurrence × sensor × dimension; plus a per-occurrence summary).

use std::io::Write;
use std::path::Path;

use crate::analysis::TrialResult;
use crate::error::{Error, Result};

fn esc(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

fn num(v: Option<f64>, digits: usize) -> String {
    match v {
        Some(x) if x.is_finite() => format!("{:.*}", digits, x),
        _ => String::new(),
    }
}

/// Columns of the landmark table (the last five were added in 0.2; earlier
/// columns keep their order so existing scripts keep working).
pub const LANDMARK_COLUMNS: &[&str] = &[
    "session", "trial", "occurrence", "word", "status", "reviewed", "edited", "confidence",
    "word_onset_s", "word_offset_s", "word_duration_ms", "sensor", "dimension", "window_start_s", "window_end_s", "range_mm",
    "pos_peak_time_s", "pos_peak_value_mm", "pos_peak_at_edge", "vel_peak_time_s", "vel_peak_value_mm_s", "vel_peak_at_edge",
    "acc_peak_time_s", "acc_peak_value_mm_s2", "acc_peak_at_edge", "pos_peak_rel_onset_ms", "vel_peak_rel_onset_ms", "acc_peak_rel_onset_ms",
    "segment_method", "pos_peak_manual", "vel_peak_manual", "acc_peak_manual", "file",
];

/// Kinematic landmarks, long format: one row per trial × occurrence × sensor
/// × dimension. `rows` pairs each result with its session (folder) name.
pub fn landmarks_csv(rows: &[(&str, &TrialResult)]) -> String {
    let mut out = LANDMARK_COLUMNS.join(",");
    out.push('\n');
    for (session, r) in rows {
        if r.occurrences.is_empty() {
            let mut cells = vec![esc(session), esc(&r.trial_id), String::new(), String::new(), status_str(r).to_string(), r.reviewed.to_string()];
            cells.resize(LANDMARK_COLUMNS.len() - 1, String::new());
            cells.push(esc(&r.file));
            out.push_str(&cells.join(","));
            out.push('\n');
            continue;
        }
        for (k, o) in r.occurrences.iter().enumerate() {
            let oc = &o.occurrence;
            for kin in &o.kinematics {
                let rel = |t: Option<f64>| t.map(|t| (t - oc.start_s) * 1000.0);
                let p = kin.position.as_ref();
                let v = kin.velocity.as_ref();
                let a = kin.acceleration.as_ref();
                let cells = [
                    esc(session),
                    esc(&r.trial_id),
                    (k + 1).to_string(),
                    esc(&oc.word),
                    status_str(r).to_string(),
                    r.reviewed.to_string(),
                    o.edited.to_string(),
                    num(Some(oc.confidence), 3),
                    num(Some(oc.start_s), 4),
                    num(Some(oc.end_s), 4),
                    num(Some((oc.end_s - oc.start_s) * 1000.0), 1),
                    esc(&kin.sensor),
                    kin.dim.label().to_string(),
                    num(Some(kin.window_start_s), 4),
                    num(Some(kin.window_end_s), 4),
                    num(Some(kin.range), 3),
                    num(p.map(|x| x.time_s), 4),
                    num(p.map(|x| x.value), 3),
                    p.map(|x| x.at_edge).unwrap_or(false).to_string(),
                    num(v.map(|x| x.time_s), 4),
                    num(v.map(|x| x.value), 2),
                    v.map(|x| x.at_edge).unwrap_or(false).to_string(),
                    num(a.map(|x| x.time_s), 4),
                    num(a.map(|x| x.value), 1),
                    a.map(|x| x.at_edge).unwrap_or(false).to_string(),
                    num(rel(p.map(|x| x.time_s)), 1),
                    num(rel(v.map(|x| x.time_s)), 1),
                    num(rel(a.map(|x| x.time_s)), 1),
                    oc.engine.as_str().to_string(),
                    p.map(|x| x.manual).unwrap_or(false).to_string(),
                    v.map(|x| x.manual).unwrap_or(false).to_string(),
                    a.map(|x| x.manual).unwrap_or(false).to_string(),
                    esc(&r.file),
                ];
                out.push_str(&cells.join(","));
                out.push('\n');
            }
        }
    }
    out
}

/// Kinematic landmarks of one session, long format (see [`landmarks_csv`]).
pub fn write_landmarks_csv(path: &Path, session: &str, results: &[TrialResult]) -> Result<()> {
    let rows: Vec<(&str, &TrialResult)> = results.iter().map(|r| (session, r)).collect();
    let mut f = std::fs::File::create(path).map_err(|e| Error::Io(path.display().to_string(), e))?;
    f.write_all(landmarks_csv(&rows).as_bytes()).map_err(|e| Error::Io(path.display().to_string(), e))
}

fn status_str(r: &TrialResult) -> &'static str {
    use crate::analysis::TrialStatus::*;
    match r.status {
        Ok => "ok",
        LowConfidence => "low_confidence",
        NotFound => "not_found",
        Error => "error",
        Excluded => "excluded",
    }
}

/// Protocol landmarks and measures, wide format: one row per trial ×
/// occurrence. Columns: identification, then for every landmark its time
/// (s), time relative to the segment onset (ms), how it was found and
/// whether it was placed by hand, then every measure, then the protocol's
/// warnings and the review notes (a measure far from the other repetitions
/// of its condition).
pub fn measures_csv(rows: &[(&str, &TrialResult)]) -> String {
    // landmark and measure columns in order of first appearance
    let mut lm_ids: Vec<String> = vec![];
    let mut ms_ids: Vec<String> = vec![];
    for (_, r) in rows {
        for o in &r.occurrences {
            if let Some(p) = &o.protocol {
                for l in &p.landmarks {
                    if !lm_ids.contains(&l.id) {
                        lm_ids.push(l.id.clone());
                    }
                }
                for m in &p.measures {
                    if !ms_ids.contains(&m.id) {
                        ms_ids.push(m.id.clone());
                    }
                }
            }
        }
    }
    let mut header: Vec<String> = ["session", "trial", "occurrence", "word", "status", "reviewed", "edited", "protocol", "word_onset_s", "word_offset_s"].iter().map(|s| s.to_string()).collect();
    for id in &lm_ids {
        for suffix in ["time_s", "rel_onset_ms", "criterion", "manual"] {
            header.push(format!("{id}_{suffix}"));
        }
    }
    header.extend(ms_ids.iter().cloned());
    header.push("warnings".into());
    header.push("review".into());
    header.push("file".into());
    let mut out = header.join(",");
    out.push('\n');
    for (session, r) in rows {
        for (k, o) in r.occurrences.iter().enumerate() {
            let Some(p) = &o.protocol else { continue };
            let oc = &o.occurrence;
            let mut cells = vec![
                esc(session),
                esc(&r.trial_id),
                (k + 1).to_string(),
                esc(&oc.word),
                status_str(r).to_string(),
                r.reviewed.to_string(),
                o.edited.to_string(),
                p.kind.as_str().to_string(),
                num(Some(oc.start_s), 4),
                num(Some(oc.end_s), 4),
            ];
            for id in &lm_ids {
                match p.landmark(id) {
                    Some(l) => {
                        cells.push(num(Some(l.time_s), 4));
                        cells.push(num(Some((l.time_s - oc.start_s) * 1000.0), 1));
                        cells.push(esc(&l.criterion));
                        cells.push(l.manual.to_string());
                    }
                    None => cells.extend([String::new(), String::new(), String::new(), String::new()]),
                }
            }
            for id in &ms_ids {
                cells.push(num(p.measure(id), 2));
            }
            cells.push(esc(&p.warnings.join("; ")));
            cells.push(esc(&r.review.join("; ")));
            cells.push(esc(&r.file));
            out.push_str(&cells.join(","));
            out.push('\n');
        }
    }
    out
}

/// One row per occurrence with word boundaries and phone segmentation.
pub fn write_segments_csv(path: &Path, session: &str, results: &[TrialResult]) -> Result<()> {
    let mut f = std::fs::File::create(path).map_err(|e| Error::Io(path.display().to_string(), e))?;
    let mut out = String::from("session,trial,occurrence,word,status,reviewed,edited,engine,confidence,word_onset_s,word_offset_s,word_duration_ms,phrase_onset_s,phrase_offset_s,phones,note,message\n");
    for r in results {
        if r.occurrences.is_empty() {
            out.push_str(&format!("{},{},,,{},{},,,,,,,,,,{},{}\n", esc(session), esc(&r.trial_id), status_str(r), r.reviewed, esc(&r.note), esc(&r.message)));
            continue;
        }
        for (k, o) in r.occurrences.iter().enumerate() {
            let oc = &o.occurrence;
            let phones = oc.phones.iter().map(|p| format!("{}:{:.3}-{:.3}", p.symbol, p.start_s, p.end_s)).collect::<Vec<_>>().join(" ");
            out.push_str(&format!(
                "{},{},{},{},{},{},{},{},{:.3},{:.4},{:.4},{:.1},{:.4},{:.4},{},{},{}\n",
                esc(session),
                esc(&r.trial_id),
                k + 1,
                esc(&oc.word),
                status_str(r),
                r.reviewed,
                o.edited,
                oc.engine.as_str(),
                oc.confidence,
                oc.start_s,
                oc.end_s,
                (oc.end_s - oc.start_s) * 1000.0,
                oc.phrase_start_s,
                oc.phrase_end_s,
                esc(&phones),
                esc(&r.note),
                esc(&r.message),
            ));
        }
    }
    f.write_all(out.as_bytes()).map_err(|e| Error::Io(path.display().to_string(), e))
}
