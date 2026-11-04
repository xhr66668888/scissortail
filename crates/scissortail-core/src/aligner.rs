//! Phonetic forced alignment with the MMS wav2vec2 CTC model (ONNX) and a
//! Viterbi aligner that supports `*` wildcard tokens (absorbing any speech or
//! noise that is not part of the transcript).

use std::path::Path;

use crate::dsp::resample;
use crate::error::{Error, Result};
use crate::ipa::{MMS_BLANK, MMS_STAR};
use crate::onnx::{Model, Value};

pub const ALIGNER_FS: f64 = 16000.0;
/// wav2vec2 frame hop (320 samples at 16 kHz).
pub const FRAME_S: f64 = 0.02;

pub struct PhoneticAligner {
    model: Model,
}

/// Frame-level log-posteriors (T × V), V includes the wildcard column.
#[derive(Debug, Clone)]
pub struct LogProbs {
    pub frames: usize,
    pub vocab: usize,
    pub data: Vec<f32>,
    pub frame_s: f64,
}

impl LogProbs {
    #[inline]
    pub fn at(&self, t: usize, v: usize) -> f32 {
        self.data[t * self.vocab + v]
    }
}

#[derive(Debug, Clone)]
pub struct Alignment {
    /// (first frame, last frame exclusive) for every token.
    pub token_frames: Vec<(usize, usize)>,
    /// Mean posterior probability of every token over its frames.
    pub token_prob: Vec<f64>,
    pub log_score: f64,
}

impl PhoneticAligner {
    pub fn load(model: &Path, threads: usize) -> Result<Self> {
        if !model.exists() {
            return Err(Error::Model(format!("aligner model not found at {}", model.display())));
        }
        let model = Model::load(model, threads).map_err(|e| e.with_context(&format!("cannot load {}", model.display())))?;
        Ok(PhoneticAligner { model })
    }

    /// Runs the acoustic model on mono audio (any rate) and returns
    /// log-posteriors with an extra all-zero column for the wildcard token.
    pub fn log_probs(&self, samples: &[f32], fs: f64) -> Result<LogProbs> {
        let mut x = resample(samples, fs, ALIGNER_FS);
        if x.len() < 800 {
            x.resize(800, 0.0);
        }
        // layer-norm the waveform (as the MMS_FA bundle does)
        let n = x.len() as f64;
        let mean = x.iter().map(|&v| v as f64).sum::<f64>() / n;
        let var = x.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
        let inv = 1.0 / (var + 1e-5).sqrt();
        let xs: Vec<f32> = x.iter().map(|&v| ((v as f64 - mean) * inv) as f32).collect();
        let len = xs.len();
        let outputs = self.model.run(vec![("audio", Value::F32(vec![1, len], xs))], &["logprobs"])?;
        let (shape, data) = outputs.into_iter().next().ok_or_else(|| Error::Model("aligner gave no output".into()))?.into_f32()?;
        let frames = shape[1];
        let v0 = shape[2];
        let vocab = v0 + 1;
        let mut out = Vec::with_capacity(frames * vocab);
        for t in 0..frames {
            out.extend_from_slice(&data[t * v0..(t + 1) * v0]);
            out.push(0.0); // wildcard
        }
        Ok(LogProbs { frames, vocab, data: out, frame_s: len as f64 / ALIGNER_FS / frames as f64 })
    }
}

/// CTC Viterbi forced alignment of `tokens` to frames `[t0, t1)` of `lp`.
pub fn forced_align(lp: &LogProbs, tokens: &[usize], t0: usize, t1: usize) -> Option<Alignment> {
    let t1 = t1.min(lp.frames);
    if tokens.is_empty() || t1 <= t0 {
        return None;
    }
    let tn = t1 - t0;
    let l = tokens.len();
    let s = 2 * l + 1;
    // minimum frames: one per token plus one blank between repeated tokens
    let repeats = tokens.windows(2).filter(|w| w[0] == w[1] && w[0] != MMS_STAR).count();
    if tn < l + repeats {
        return None;
    }
    let ext: Vec<usize> = (0..s).map(|i| if i % 2 == 0 { MMS_BLANK } else { tokens[i / 2] }).collect();
    let can_skip: Vec<bool> = (0..s).map(|i| i >= 3 && i % 2 == 1 && ext[i] != ext[i - 2]).collect();
    const NEG: f64 = -1e30;
    let mut prev = vec![NEG; s];
    let mut cur = vec![NEG; s];
    let mut bp = vec![0u8; tn * s];
    prev[0] = lp.at(t0, ext[0]) as f64;
    if s > 1 {
        prev[1] = lp.at(t0, ext[1]) as f64;
    }
    for t in 1..tn {
        let row = &mut bp[t * s..(t + 1) * s];
        for j in 0..s {
            let mut best = prev[j];
            let mut b = 0u8;
            if j >= 1 && prev[j - 1] > best {
                best = prev[j - 1];
                b = 1;
            }
            if can_skip[j] && prev[j - 2] > best {
                best = prev[j - 2];
                b = 2;
            }
            row[j] = b;
            cur[j] = if best <= NEG / 2.0 { NEG } else { best + lp.at(t0 + t, ext[j]) as f64 };
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    let end = if s >= 2 && prev[s - 2] > prev[s - 1] { s - 2 } else { s - 1 };
    if prev[end] <= NEG / 2.0 {
        return None;
    }
    let log_score = prev[end];
    let mut path = vec![0usize; tn];
    let mut j = end;
    for t in (0..tn).rev() {
        path[t] = j;
        j -= bp[t * s + j] as usize;
    }
    let mut token_frames = vec![(usize::MAX, 0usize); l];
    let mut sums = vec![0.0f64; l];
    for (t, &st) in path.iter().enumerate() {
        if st % 2 == 1 {
            let k = st / 2;
            let tf = &mut token_frames[k];
            if tf.0 == usize::MAX {
                tf.0 = t0 + t;
            }
            tf.1 = t0 + t + 1;
            sums[k] += (lp.at(t0 + t, tokens[k]) as f64).exp();
        }
    }
    let token_prob = token_frames
        .iter()
        .zip(&sums)
        .map(|(&(a, b), &sum)| if a == usize::MAX || b <= a { 0.0 } else { sum / (b - a) as f64 })
        .collect();
    Some(Alignment { token_frames, token_prob, log_score })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lp_from(probs: &[Vec<f32>]) -> LogProbs {
        let vocab = probs[0].len();
        LogProbs { frames: probs.len(), vocab, data: probs.iter().flatten().map(|p| p.ln()).collect(), frame_s: 0.02 }
    }

    #[test]
    fn aligns_simple_sequence() {
        // vocab: 0 blank, 1 'a', 2 'b' ; frames: a a - b b
        let e = 0.01f32;
        let rows = vec![
            vec![e, 1.0 - 2.0 * e, e],
            vec![e, 1.0 - 2.0 * e, e],
            vec![1.0 - 2.0 * e, e, e],
            vec![e, e, 1.0 - 2.0 * e],
            vec![e, e, 1.0 - 2.0 * e],
        ];
        let lp = lp_from(&rows);
        let a = forced_align(&lp, &[1, 2], 0, 5).unwrap();
        assert_eq!(a.token_frames, vec![(0, 2), (3, 5)]);
    }
}
