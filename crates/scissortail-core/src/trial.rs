//! EMA trial model: one synchronized audio channel plus N articulator sensors.
//!
//! The loader understands the MVIEW-style struct array used by Carstens
//! AG50x / NDI Wave exports: one struct element per channel with fields
//! `NAME`, `SRATE`, `SIGNAL` (and optionally `SOURCE`).

use std::path::Path;

use serde::Serialize;

use crate::error::{Error, Result};
use crate::mat::{read_mat_file, MatFile, MatValue};

#[derive(Debug, Clone)]
pub struct Audio {
    pub fs: f64,
    pub samples: Vec<f32>,
}

impl Audio {
    pub fn duration(&self) -> f64 {
        self.samples.len() as f64 / self.fs
    }
}

#[derive(Debug, Clone)]
pub struct Sensor {
    pub name: String,
    pub fs: f64,
    /// `columns[c][i]` = sample i of column c (x, y, z, then any angle columns).
    pub columns: Vec<Vec<f64>>,
}

impl Sensor {
    pub fn len(&self) -> usize {
        self.columns.first().map(|c| c.len()).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn duration(&self) -> f64 {
        self.len() as f64 / self.fs
    }

    /// Reference sensors (head-correction references, and the upper incisor
    /// which is fixed to the skull) are excluded from kinematic analysis by
    /// default; they can still be enabled in the settings.
    pub fn looks_like_reference(&self) -> bool {
        let n = self.name.to_ascii_uppercase();
        n.contains("REF") || n == "UI" || n == "NOSE" || n == "NAS" || n == "MAST" || n.starts_with("HEAD")
    }
}

#[derive(Debug, Clone)]
pub struct Trial {
    pub id: String,
    pub variable: String,
    pub audio: Audio,
    pub sensors: Vec<Sensor>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SensorInfo {
    pub name: String,
    pub fs: f64,
    pub n_samples: usize,
    pub n_columns: usize,
    pub is_reference: bool,
}

impl Trial {
    pub fn load(path: &Path) -> Result<Trial> {
        let mat = read_mat_file(path)?;
        let stem = path.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
        Trial::from_mat(&mat, &stem).map_err(|e| e.with_context(&path.display().to_string()))
    }

    pub fn from_mat(mat: &MatFile, stem: &str) -> Result<Trial> {
        // Prefer the variable named like the file; otherwise the first struct
        // array that has NAME/SIGNAL fields.
        let candidates: Vec<&(String, MatValue)> = mat
            .variables
            .iter()
            .filter(|(_, v)| matches!(v, MatValue::Struct { fields, .. } if has_field(fields, "SIGNAL") && has_field(fields, "NAME")))
            .collect();
        let (var_name, value) = candidates
            .iter()
            .find(|(n, _)| n == stem)
            .or_else(|| candidates.first())
            .map(|(n, v)| (n.clone(), v))
            .ok_or_else(|| {
                Error::Format(
                    "no EMA struct array (fields NAME / SRATE / SIGNAL) found; variables present: ".to_string()
                        + &mat.variables.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>().join(", "),
                )
            })?;

        let MatValue::Struct { fields, elements, .. } = value else { unreachable!() };
        let srate_field = fields.iter().find(|f| f.eq_ignore_ascii_case("SRATE")).cloned();
        let name_field = fields.iter().find(|f| f.eq_ignore_ascii_case("NAME")).cloned().unwrap();
        let signal_field = fields.iter().find(|f| f.eq_ignore_ascii_case("SIGNAL")).cloned().unwrap();

        let mut audio: Option<Audio> = None;
        let mut sensors = Vec::new();
        for (idx, el) in elements.iter().enumerate() {
            let name = el.get(&name_field).and_then(|v| v.as_str()).unwrap_or("").trim().to_string();
            let name = if name.is_empty() { format!("CH{}", idx + 1) } else { name };
            let fs = srate_field.as_ref().and_then(|f| el.get(f)).and_then(|v| v.as_scalar()).unwrap_or(0.0);
            let Some((rows, cols, data)) = el.get(&signal_field).and_then(|v| v.as_matrix()) else { continue };
            if rows == 0 || cols == 0 {
                continue;
            }
            let is_audio = name.eq_ignore_ascii_case("AUDIO") || name.eq_ignore_ascii_case("SOUND") || name.eq_ignore_ascii_case("MIC") || fs >= 4000.0;
            if is_audio && audio.is_none() {
                if fs <= 0.0 {
                    return Err(Error::Format(format!("audio channel '{name}' has no sampling rate")));
                }
                // First column only (mono).
                let samples = data[..rows].iter().map(|&v| v as f32).collect();
                audio = Some(Audio { fs, samples });
            } else {
                if fs <= 0.0 {
                    return Err(Error::Format(format!("sensor '{name}' has no sampling rate")));
                }
                let columns = (0..cols).map(|c| data[c * rows..(c + 1) * rows].to_vec()).collect();
                sensors.push(Sensor { name, fs, columns });
            }
        }
        let audio = audio.ok_or_else(|| Error::Format("no audio channel (NAME = 'AUDIO' or SRATE ≥ 4 kHz) in EMA struct".into()))?;
        Ok(Trial { id: stem.to_string(), variable: var_name, audio, sensors })
    }

    pub fn sensor(&self, name: &str) -> Option<&Sensor> {
        self.sensors.iter().find(|s| s.name.eq_ignore_ascii_case(name))
    }

    pub fn sensor_info(&self) -> Vec<SensorInfo> {
        self.sensors
            .iter()
            .map(|s| SensorInfo {
                name: s.name.clone(),
                fs: s.fs,
                n_samples: s.len(),
                n_columns: s.columns.len(),
                is_reference: s.looks_like_reference(),
            })
            .collect()
    }
}

fn has_field(fields: &[String], name: &str) -> bool {
    fields.iter().any(|f| f.eq_ignore_ascii_case(name))
}
