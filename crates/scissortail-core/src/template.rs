//! TTS-template matching engine: speaker-robust broad-class acoustic
//! features + subsequence dynamic time warping (DTW).

use realfft::RealFftPlanner;

use crate::dsp::{lowpass_filtfilt, percentile};

pub const FEAT_FS: f64 = 16000.0;
pub const FEAT_HOP_S: f64 = 0.01;

/// Per-frame features plus the normalized energy track (0 = noise floor, 1 = loud vowel).
pub struct Features {
    pub dim: usize,
    pub frames: usize,
    pub data: Vec<f64>,
    pub energy: Vec<f64>,
}

impl Features {
    pub fn row(&self, i: usize) -> &[f64] {
        &self.data[i * self.dim..(i + 1) * self.dim]
    }
}

fn mel_filterbank(n_mels: usize, nfft: usize, fs: f64, fmin: f64, fmax: f64) -> Vec<Vec<f64>> {
    let hz2mel = |h: f64| 2595.0 * (1.0 + h / 700.0).log10();
    let mel2hz = |m: f64| 700.0 * (10f64.powf(m / 2595.0) - 1.0);
    let (m0, m1) = (hz2mel(fmin), hz2mel(fmax));
    let pts: Vec<f64> = (0..n_mels + 2).map(|i| mel2hz(m0 + (m1 - m0) * i as f64 / (n_mels + 1) as f64)).collect();
    let bins: Vec<f64> = (0..nfft / 2 + 1).map(|k| k as f64 * fs / nfft as f64).collect();
    (0..n_mels)
        .map(|i| {
            let (l, c, r) = (pts[i], pts[i + 1], pts[i + 2]);
            bins.iter().map(|&b| ((b - l) / (c - l)).min((r - b) / (r - c)).max(0.0)).collect()
        })
        .collect()
}

/// Broad-class features of 16 kHz audio (25 ms window, 10 ms hop).
pub fn broad_features(x: &[f32]) -> Features {
    let fs = FEAT_FS;
    let win = 400usize;
    let hop = 160usize;
    let nfft = 512usize;
    let mean = x.iter().map(|&v| v as f64).sum::<f64>() / x.len().max(1) as f64;
    let xd: Vec<f64> = x.iter().map(|&v| v as f64 - mean).collect();
    let pad = win / 2;
    let mut xp = vec![0.0; pad];
    xp.extend_from_slice(&xd);
    xp.extend(std::iter::repeat(0.0).take(pad));
    let lp = lowpass_filtfilt(&xd, 1000.0, fs, 4);
    let mut lpp = vec![0.0; pad];
    lpp.extend_from_slice(&lp);
    lpp.extend(std::iter::repeat(0.0).take(pad));
    let n_frames = if xp.len() >= win { 1 + (xp.len() - win) / hop } else { 1 };

    let hann: Vec<f64> = (0..win).map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / win as f64).cos()).collect();
    let mut planner = RealFftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(nfft);
    let fft2 = planner.plan_fft_forward(2 * nfft);
    let ifft2 = planner.plan_fft_inverse(2 * nfft);
    let mel = mel_filterbank(24, nfft, fs, 60.0, 7600.0);
    let freqs: Vec<f64> = (0..nfft / 2 + 1).map(|k| k as f64 * fs / nfft as f64).collect();

    let mut e_db = vec![0.0; n_frames];
    let mut ratios = vec![[0.0f64; 3]; n_frames];
    let mut voicing = vec![0.0; n_frames];
    let mut logmel = vec![vec![0.0f64; 24]; n_frames];
    let mut inb = fft.make_input_vec();
    let mut outb = fft.make_output_vec();
    let mut in2 = fft2.make_input_vec();
    let mut out2 = fft2.make_output_vec();
    let mut back = ifft2.make_output_vec();
    for f in 0..n_frames {
        let s = f * hop;
        for i in 0..nfft {
            inb[i] = if i < win { xp[s + i] * hann[i] } else { 0.0 };
        }
        fft.process(&mut inb, &mut outb).ok();
        let p: Vec<f64> = outb.iter().map(|c| c.norm_sqr() + 1e-12).collect();
        let (mut tot, mut b1, mut b2, mut b3) = (0.0, 0.0, 0.0, 0.0);
        for (k, &pk) in p.iter().enumerate() {
            let fq = freqs[k];
            if (60.0..=7800.0).contains(&fq) {
                tot += pk;
            }
            if (60.0..900.0).contains(&fq) {
                b1 += pk;
            } else if (900.0..2800.0).contains(&fq) {
                b2 += pk;
            } else if (2800.0..7800.0).contains(&fq) {
                b3 += pk;
            }
        }
        e_db[f] = 10.0 * tot.log10();
        let r = |b: f64| ((b / tot).log10().clamp(-3.0, 0.0)) / 3.0 + 1.0;
        ratios[f] = [r(b1), r(b2), r(b3)];
        for (m, filt) in mel.iter().enumerate() {
            logmel[f][m] = (filt.iter().zip(&p).map(|(w, v)| w * v).sum::<f64>() + 1e-10).ln();
        }
        // voicing: normalized autocorrelation peak (2.5–15 ms) of the low-passed frame
        for i in 0..2 * nfft {
            in2[i] = if i < win { lpp[s + i] * hann[i] } else { 0.0 };
        }
        fft2.process(&mut in2, &mut out2).ok();
        for c in out2.iter_mut() {
            *c = realfft::num_complex::Complex::new(c.norm_sqr(), 0.0);
        }
        ifft2.process(&mut out2, &mut back).ok();
        let ac0 = back[0].max(1e-12);
        let lo = (0.0025 * fs) as usize;
        let hi = (0.015 * fs) as usize;
        let mx = back[lo..hi].iter().cloned().fold(f64::MIN, f64::max);
        voicing[f] = (mx / ac0).clamp(0.0, 1.0);
    }
    let peak = percentile(&e_db, 97.0);
    let floor = percentile(&e_db, 5.0).max(peak - 55.0);
    let span = (peak - floor).max(6.0);
    let en: Vec<f64> = e_db.iter().map(|&e| ((e - floor) / span).clamp(0.0, 1.0)).collect();
    let sp: Vec<f64> = en.iter().map(|&e| ((e - 0.2) / 0.2).clamp(0.0, 1.0)).collect();
    let mut flux = vec![0.0; n_frames];
    for f in 1..n_frames {
        let d: f64 = (0..24).map(|m| (logmel[f][m] - logmel[f - 1][m]).max(0.0)).sum::<f64>() / 24.0;
        flux[f] = (d / 1.5).clamp(0.0, 1.0);
    }
    // cepstra 1..4 (DCT-II of log-mel), normalized over speech frames
    let mut ceps = vec![[0.0f64; 4]; n_frames];
    for f in 0..n_frames {
        for (q, c) in ceps[f].iter_mut().enumerate() {
            let k = q + 1;
            *c = (0..24).map(|m| logmel[f][m] * (std::f64::consts::PI * k as f64 * (m as f64 + 0.5) / 24.0).cos()).sum::<f64>() * (2.0 / 24.0f64).sqrt();
        }
    }
    let speech: Vec<usize> = (0..n_frames).filter(|&f| sp[f] > 0.5).collect();
    let use_idx: Vec<usize> = if speech.len() > 5 { speech } else { (0..n_frames).collect() };
    for q in 0..4 {
        let m = use_idx.iter().map(|&f| ceps[f][q]).sum::<f64>() / use_idx.len() as f64;
        let sd = (use_idx.iter().map(|&f| (ceps[f][q] - m).powi(2)).sum::<f64>() / use_idx.len() as f64).sqrt() + 1e-6;
        for c in ceps.iter_mut() {
            c[q] = (c[q] - m) / sd;
        }
    }
    let dim = 1 + 3 + 1 + 1 + 4;
    let mut data = Vec::with_capacity(n_frames * dim);
    for f in 0..n_frames {
        let g = sp[f];
        data.push(2.0 * en[f]);
        for r in ratios[f] {
            data.push(1.2 * r * g);
        }
        data.push(1.0 * voicing[f] * g);
        data.push(0.6 * flux[f]);
        for c in ceps[f] {
            data.push(0.25 * c * g);
        }
    }
    Features { dim, frames: n_frames, data, energy: en }
}

/// One subsequence-DTW match: template frames 0..N aligned to recording frames.
#[derive(Debug, Clone)]
pub struct DtwMatch {
    pub start: usize,
    pub end: usize,
    pub cost: f64,
    /// (template frame, recording frame) pairs, monotonic.
    pub path: Vec<(usize, usize)>,
}

/// Subsequence DTW with step pattern {(1,1), (1,2), (2,1)} (local slope 0.5–2),
/// path-length normalized cost. Returns up to `k` non-overlapping matches.
pub fn subsequence_dtw(tpl: &Features, rec: &Features, k: usize) -> Vec<DtwMatch> {
    let n = tpl.frames;
    let m = rec.frames;
    if n < 2 || m < 2 {
        return vec![];
    }
    let mut c = vec![0.0f64; n * m];
    let scale = (tpl.dim as f64).sqrt();
    for i in 0..n {
        let a = tpl.row(i);
        for j in 0..m {
            let b = rec.row(j);
            let d: f64 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
            c[i * m + j] = d.sqrt() / scale;
        }
    }
    const INF: f64 = f64::INFINITY;
    let mut d = vec![INF; n * m];
    let mut l = vec![0u32; n * m];
    let mut bp = vec![0u8; n * m];
    for j in 0..m {
        d[j] = c[j];
        l[j] = 1;
    }
    for i in 1..n {
        for j in 1..m {
            let mut best = INF;
            let (mut bv, mut bl, mut bb) = (INF, 0u32, 0u8);
            let cand = |v: f64, len: u32, b: u8, best: &mut f64, bv: &mut f64, bl: &mut u32, bb: &mut u8| {
                if v.is_finite() && len > 0 {
                    let r = v / len as f64;
                    if r < *best {
                        *best = r;
                        *bv = v;
                        *bl = len;
                        *bb = b;
                    }
                }
            };
            let p = (i - 1) * m + (j - 1);
            cand(d[p] + c[i * m + j], l[p] + 1, 0, &mut best, &mut bv, &mut bl, &mut bb);
            if j >= 2 {
                let p = (i - 1) * m + (j - 2);
                cand(d[p] + c[i * m + j - 1] + c[i * m + j], l[p] + 2, 1, &mut best, &mut bv, &mut bl, &mut bb);
            }
            if i >= 2 {
                let p = (i - 2) * m + (j - 1);
                cand(d[p] + c[(i - 1) * m + j] + c[i * m + j], l[p] + 2, 2, &mut best, &mut bv, &mut bl, &mut bb);
            }
            if best.is_finite() {
                d[i * m + j] = bv;
                l[i * m + j] = bl;
                bp[i * m + j] = bb;
            }
        }
    }
    let last = (n - 1) * m;
    let mut ends: Vec<(usize, f64)> = (0..m)
        .filter(|&j| d[last + j].is_finite() && l[last + j] > 0)
        .map(|j| (j, d[last + j] / l[last + j] as f64))
        .collect();
    ends.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let mut out: Vec<DtwMatch> = Vec::new();
    for (j_end, cost) in ends {
        let mut path = vec![(n - 1, j_end)];
        let (mut i, mut j) = (n - 1, j_end);
        let mut ok = true;
        while i > 0 {
            match bp[i * m + j] {
                0 => {
                    i -= 1;
                    j = match j.checked_sub(1) { Some(v) => v, None => { ok = false; break } };
                }
                1 => {
                    if j < 2 { ok = false; break; }
                    path.push((i - 1, j - 1));
                    i -= 1;
                    j -= 2;
                }
                _ => {
                    if i < 2 || j < 1 { ok = false; break; }
                    path.push((i - 1, j));
                    i -= 2;
                    j -= 1;
                }
            }
            path.push((i, j));
        }
        if !ok {
            continue;
        }
        path.reverse();
        let start = path[0].1;
        if out.iter().any(|o| !(j_end < o.start || start > o.end)) {
            continue;
        }
        out.push(DtwMatch { start, end: j_end, cost, path });
        if out.len() >= k {
            break;
        }
    }
    out
}
