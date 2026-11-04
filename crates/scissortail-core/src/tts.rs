//! Kokoro-82M text-to-speech (ONNX) driven directly by IPA, used to
//! synthesize a reference pronunciation ("template") of the target word.
//! The bundled model exposes Kokoro's per-phoneme durations as a second
//! output, so phone boundaries inside the synthetic audio are known exactly.

use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::error::{Error, Result};
use crate::ipa::{kokoro_id, to_kokoro};
use crate::onnx::{Model, Value};

pub const TTS_FS: f64 = 24000.0;
/// Samples of audio per Kokoro duration unit.
const SAMPLES_PER_UNIT: usize = 600;

pub struct Tts {
    model: Model,
    voices_dir: PathBuf,
}

#[derive(Debug, Clone, Serialize)]
pub struct SynthPhone {
    pub symbol: String,
    pub start_s: f64,
    pub end_s: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct Synthesis {
    pub input: String,
    pub kokoro_text: String,
    pub unknown_symbols: Vec<String>,
    pub voice: String,
    pub speed: f32,
    #[serde(skip)]
    pub samples: Vec<f32>,
    pub fs: f64,
    /// Phone spans (seconds) in `samples`, after silence trimming.
    pub phones: Vec<SynthPhone>,
}

impl Tts {
    pub fn load(model: &Path, voices_dir: &Path, threads: usize) -> Result<Self> {
        if !model.exists() {
            return Err(Error::Model(format!("TTS model not found at {}", model.display())));
        }
        let model = Model::load(model, threads).map_err(|e| e.with_context(&format!("cannot load {}", model.display())))?;
        Ok(Tts { model, voices_dir: voices_dir.to_path_buf() })
    }

    pub fn voices(&self) -> Vec<String> {
        let mut v: Vec<String> = std::fs::read_dir(&self.voices_dir)
            .map(|rd| {
                rd.filter_map(|e| e.ok())
                    .filter_map(|e| {
                        let p = e.path();
                        (p.extension().map(|x| x == "bin").unwrap_or(false)).then(|| p.file_stem().unwrap().to_string_lossy().to_string())
                    })
                    .collect()
            })
            .unwrap_or_default();
        v.sort();
        v
    }

    fn style(&self, voice: &str, n_tokens: usize) -> Result<Vec<f32>> {
        let path = self.voices_dir.join(format!("{voice}.bin"));
        let bytes = std::fs::read(&path).map_err(|e| Error::Io(path.display().to_string(), e))?;
        let floats: Vec<f32> = bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let rows = floats.len() / 256;
        if rows == 0 {
            return Err(Error::Model(format!("voice file {} is empty", path.display())));
        }
        let r = n_tokens.min(rows - 1);
        Ok(floats[r * 256..(r + 1) * 256].to_vec())
    }

    /// Synthesizes an IPA string. Leading/trailing silence is trimmed and the
    /// phone spans are expressed relative to the trimmed audio.
    pub fn synthesize(&self, ipa: &str, voice: &str, speed: f32) -> Result<Synthesis> {
        let k = to_kokoro(ipa);
        if k.text.trim().is_empty() {
            return Err(Error::Invalid("the IPA transcription contains no symbols Kokoro can pronounce".into()));
        }
        let chars: Vec<char> = k.text.chars().collect();
        let mut ids: Vec<i64> = vec![0];
        ids.extend(chars.iter().filter_map(|&c| kokoro_id(c)));
        ids.push(0);
        let n_tokens = ids.len() - 2;
        let style = self.style(voice, n_tokens)?;
        let n_ids = ids.len();
        let inputs = vec![
            ("input_ids", Value::I64(vec![1, n_ids], ids)),
            ("style", Value::F32(vec![1, 256], style)),
            ("speed", Value::F32(vec![1], vec![speed])),
        ];
        let mut outputs = self.model.run(inputs, &["waveform", "durations"])?.into_iter();
        let (wav, dur) = match (outputs.next(), outputs.next()) {
            (Some(w), Some(d)) => (w.into_f32()?.1, d.into_i64()?.1),
            _ => return Err(Error::Model("TTS gave no output".into())),
        };

        // token spans (samples): index 0 and last are the '$' boundary tokens
        let mut spans = Vec::new();
        let mut pos = 0usize;
        for (i, &d) in dur.iter().enumerate() {
            let len = d.max(0) as usize * SAMPLES_PER_UNIT;
            if i > 0 && i <= chars.len() {
                spans.push((chars[i - 1], pos, pos + len));
            }
            pos += len;
        }
        // trim silence (−35 dB re. peak, 10 ms smoothing)
        let (a, b) = trim_bounds(&wav, TTS_FS, -35.0);
        let samples = wav[a..b].to_vec();
        let off = a as f64 / TTS_FS;
        let total = samples.len() as f64 / TTS_FS;
        let mut phones: Vec<SynthPhone> = Vec::new();
        let mut carry_start: Option<f64> = None;
        for (c, s, e) in spans {
            if c == ' ' {
                carry_start = None;
                continue;
            }
            let (s, e) = ((s as f64 / TTS_FS - off).clamp(0.0, total), (e as f64 / TTS_FS - off).clamp(0.0, total));
            // Stress marks precede the syllable: their frames belong to the next
            // phone. Length / aspiration marks follow: extend the previous phone.
            if matches!(c, 'ˈ' | 'ˌ') {
                carry_start = Some(carry_start.unwrap_or(s));
                continue;
            }
            if matches!(c, 'ː' | 'ʰ' | 'ʲ') {
                if let Some(p) = phones.last_mut() {
                    p.end_s = e;
                    if c != 'ː' {
                        p.symbol.push(c);
                    } else {
                        p.symbol.push('ː');
                    }
                }
                continue;
            }
            let start = carry_start.take().unwrap_or(s);
            phones.push(SynthPhone { symbol: display_symbol(c), start_s: start, end_s: e });
        }
        Ok(Synthesis {
            input: ipa.to_string(),
            kokoro_text: k.text,
            unknown_symbols: k.unknown,
            voice: voice.to_string(),
            speed,
            samples,
            fs: TTS_FS,
            phones,
        })
    }
}

fn display_symbol(c: char) -> String {
    match c {
        'A' => "eɪ".into(),
        'I' => "aɪ".into(),
        'O' => "oʊ".into(),
        'W' => "aʊ".into(),
        'Y' => "ɔɪ".into(),
        'Q' => "əʊ".into(),
        'ʧ' => "tʃ".into(),
        'ʤ' => "dʒ".into(),
        c => c.to_string(),
    }
}

/// Sample range that keeps everything louder than `thresh_db` below the peak.
pub fn trim_bounds(x: &[f32], fs: f64, thresh_db: f64) -> (usize, usize) {
    let win = ((0.01 * fs) as usize).max(1);
    let n = x.len();
    if n == 0 {
        return (0, 0);
    }
    let mut energy = vec![0.0f64; n];
    let mut acc = 0.0f64;
    let sq: Vec<f64> = x.iter().map(|&v| (v as f64) * (v as f64)).collect();
    for i in 0..n {
        acc += sq[i];
        if i >= win {
            acc -= sq[i - win];
        }
        energy[i] = acc / win as f64;
    }
    let peak = energy.iter().cloned().fold(0.0, f64::max).max(1e-20);
    let thr = peak * 10f64.powf(thresh_db / 10.0);
    let first = energy.iter().position(|&e| e > thr).unwrap_or(0);
    let last = energy.iter().rposition(|&e| e > thr).unwrap_or(n - 1);
    let pad = (0.005 * fs) as usize;
    let a = first.saturating_sub(win).saturating_sub(pad);
    let b = (last + pad).min(n);
    (a, b.max(a + 1))
}
