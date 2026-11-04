//! Discovering sessions (folders of trial MAT-files) and trial metadata.

use std::path::{Path, PathBuf};

use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct TrialEntry {
    pub id: String,
    pub file: String,
    /// Within-session trial number (last "T<digits>" in the name), if any.
    pub number: Option<u32>,
    /// Condition code embedded in the name (e.g. "SUad"), if any.
    pub condition: String,
    pub size_bytes: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct SessionEntry {
    pub name: String,
    pub path: String,
    /// Participant identifier parsed from the folder name (e.g. "cancer-patient-3").
    pub participant: String,
    /// Time point / stage parsed from the folder name (e.g. "preoperative").
    pub timepoint: String,
    pub trials: Vec<TrialEntry>,
}

fn is_mat(p: &Path) -> bool {
    p.is_file() && p.extension().map(|e| e.eq_ignore_ascii_case("mat")).unwrap_or(false)
}

/// Last "T<digits>" group in a trial name ("T3_T137_FUad_" → 137, "T0003" → 3).
pub fn trial_number(name: &str) -> Option<u32> {
    let b = name.as_bytes();
    let mut last = None;
    let mut i = 0;
    while i < b.len() {
        if (b[i] == b'T' || b[i] == b't') && i + 1 < b.len() && b[i + 1].is_ascii_digit() {
            let mut j = i + 1;
            while j < b.len() && b[j].is_ascii_digit() {
                j += 1;
            }
            last = name[i + 1..j].parse().ok();
            i = j;
        } else {
            i += 1;
        }
    }
    last
}

/// Letters-only token after the last trial number, e.g. "T3_T137_FUad_" → "FUad".
pub fn condition_code(name: &str) -> String {
    name.split('_')
        .filter(|t| !t.is_empty() && t.chars().all(|c| c.is_ascii_alphabetic()))
        .last()
        .unwrap_or("")
        .to_string()
}

fn parse_session_name(name: &str) -> (String, String) {
    let lower = name.to_ascii_lowercase();
    let stages = ["preoperative", "postoperative", "pre-rt", "post-rt", "radiation-only", "baseline", "follow-up", "pre", "post"];
    let timepoint = stages.iter().find(|s| lower.contains(*s)).map(|s| s.to_string()).unwrap_or_default();
    // participant = leading part up to the first digit group
    let mut participant = String::new();
    let mut seen_digit = false;
    for part in name.split('-') {
        if !participant.is_empty() {
            participant.push('-');
        }
        participant.push_str(part);
        if part.chars().all(|c| c.is_ascii_digit()) {
            seen_digit = true;
            break;
        }
    }
    if !seen_digit {
        participant = name.to_string();
    }
    (participant, timepoint)
}

fn list_trials(dir: &Path) -> Vec<TrialEntry> {
    let mut v: Vec<TrialEntry> = std::fs::read_dir(dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| is_mat(p))
                .map(|p| {
                    let id = p.file_stem().unwrap().to_string_lossy().to_string();
                    TrialEntry {
                        number: trial_number(&id),
                        condition: condition_code(&id),
                        size_bytes: std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0),
                        file: p.display().to_string(),
                        id,
                    }
                })
                .collect()
        })
        .unwrap_or_default();
    v.sort_by(|a, b| a.number.cmp(&b.number).then(a.id.cmp(&b.id)));
    v
}

/// Folders never searched for trials (hidden, build output, packages, system).
fn skip_dir(name: &str) -> bool {
    name.starts_with('.')
        || name.ends_with(".app")
        || matches!(name, "node_modules" | "target" | "venv" | "__pycache__" | "Library" | "Applications" | "System" | "site-packages")
}

/// Finds sessions: folders that directly contain trial MAT-files. The chosen
/// folder itself may be a session; otherwise sub-folders are searched
/// breadth-first up to three levels deep (so a parent folder such as
/// `TTS/` → `cancer-patient-MAT-files/` → sessions also works). A session
/// folder is not searched further.
pub fn scan(root: &Path) -> Vec<SessionEntry> {
    const MAX_DEPTH: usize = 3;
    const MAX_DIRS: usize = 4000;
    let mut found: Vec<(PathBuf, Vec<TrialEntry>)> = Vec::new();
    let mut queue: std::collections::VecDeque<(PathBuf, usize)> = std::collections::VecDeque::from([(root.to_path_buf(), 0)]);
    let mut visited = 0;
    while let Some((dir, depth)) = queue.pop_front() {
        visited += 1;
        if visited > MAX_DIRS {
            break;
        }
        let trials = list_trials(&dir);
        if !trials.is_empty() {
            found.push((dir, trials));
            continue;
        }
        if depth >= MAX_DEPTH {
            continue;
        }
        let mut subs: Vec<PathBuf> = std::fs::read_dir(&dir)
            .map(|rd| {
                rd.filter_map(|e| e.ok())
                    .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
                    .map(|e| e.path())
                    .filter(|p| !p.file_name().map(|n| skip_dir(&n.to_string_lossy())).unwrap_or(true))
                    .collect()
            })
            .unwrap_or_default();
        subs.sort();
        queue.extend(subs.into_iter().map(|p| (p, depth + 1)));
    }
    found.sort_by(|a, b| a.0.cmp(&b.0));
    let base = |d: &Path| d.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| d.display().to_string());
    let names: Vec<String> = found.iter().map(|(d, _)| base(d)).collect();
    found
        .into_iter()
        .enumerate()
        .map(|(i, (dir, trials))| {
            // same folder name in different places: prefix the parent folder
            let mut name = names[i].clone();
            if names.iter().filter(|n| **n == name).count() > 1 {
                if let Some(parent) = dir.parent().and_then(|p| p.file_name()) {
                    name = format!("{}/{}", parent.to_string_lossy(), name);
                }
            }
            let (participant, timepoint) = parse_session_name(&base(&dir));
            SessionEntry { name, path: dir.display().to_string(), participant, timepoint, trials }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_sessions_in_nested_folders() {
        let root = std::env::temp_dir().join(format!("scissortail-scan-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        for (dir, file) in [("study/data/patient-1-pre", "T0001.mat"), ("study/data/patient-1-post", "T0002.mat"), ("study/.cache/x", "T9.mat"), ("study/docs", "readme.txt")] {
            std::fs::create_dir_all(root.join(dir)).unwrap();
            std::fs::write(root.join(dir).join(file), b"").unwrap();
        }
        let s = scan(&root.join("study"));
        let names: Vec<&str> = s.iter().map(|x| x.name.as_str()).collect();
        assert_eq!(names, vec!["patient-1-post", "patient-1-pre"]);
        // a session folder itself is one session
        assert_eq!(scan(&root.join("study/data/patient-1-pre")).len(), 1);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn parses_trial_names() {
        assert_eq!(trial_number("T3_T137_FUad_"), Some(137));
        assert_eq!(trial_number("T0003"), Some(3));
        assert_eq!(condition_code("T3_T137_FUad_"), "FUad");
        assert_eq!(condition_code("T0003"), "");
        assert_eq!(parse_session_name("cancer-patient-3-preoperative-MAT"), ("cancer-patient-3".into(), "preoperative".into()));
    }
}
