//! Signal-processing helpers: resampling, zero-phase filtering,
//! Savitzky–Golay regression filters, envelopes and spectrograms.

use realfft::RealFftPlanner;

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    let q = x * x / 4.0;
    for k in 1..60 {
        term *= q / (k as f64 * k as f64);
        sum += term;
        if term < 1e-12 * sum {
            break;
        }
    }
    sum
}

/// Rational-ratio polyphase resampler with a Kaiser-windowed sinc low-pass
/// (equivalent in spirit to `scipy.signal.resample_poly`).
pub fn resample(x: &[f32], fs_in: f64, fs_out: f64) -> Vec<f32> {
    let fi = fs_in.round() as u64;
    let fo = fs_out.round() as u64;
    if fi == fo || x.is_empty() {
        return x.to_vec();
    }
    let g = gcd(fi, fo);
    let up = (fo / g) as usize;
    let down = (fi / g) as usize;
    let max_rate = up.max(down);
    let half_len = 10 * max_rate;
    let n_taps = 2 * half_len + 1;
    let cutoff = 1.0 / max_rate as f64; // normalized to the upsampled Nyquist
    let beta = 5.0;
    let i0b = bessel_i0(beta);
    let h: Vec<f64> = (0..n_taps)
        .map(|i| {
            let m = i as f64 - half_len as f64;
            let sinc = if m == 0.0 { 1.0 } else { (std::f64::consts::PI * cutoff * m).sin() / (std::f64::consts::PI * cutoff * m) };
            let r = m / half_len as f64;
            let w = bessel_i0(beta * (1.0 - r * r).max(0.0).sqrt()) / i0b;
            cutoff * sinc * w * up as f64
        })
        .collect();
    let n_out = (x.len() * up).div_ceil(down);
    let mut y = Vec::with_capacity(n_out);
    for n in 0..n_out {
        // position in the upsampled signal, centred on the filter
        let centre = (n * down) as i64 + half_len as i64;
        let mut acc = 0.0f64;
        // taps k such that (centre - k) is a multiple of `up`
        let first_k = (centre.rem_euclid(up as i64)) as usize;
        let mut k = first_k;
        while k < n_taps {
            let idx = (centre - k as i64) / up as i64;
            if idx >= 0 && (idx as usize) < x.len() {
                acc += h[k] * x[idx as usize] as f64;
            }
            k += up;
        }
        y.push(acc as f32);
    }
    y
}

// ---------------------------------------------------------------- filters --

#[derive(Clone, Copy, Debug)]
struct Biquad {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
}

impl Biquad {
    fn run(&self, x: &[f64]) -> Vec<f64> {
        // Steady-state initial conditions for a constant input x[0]
        // (transposed direct form II: z1 = b1·x − a1·y + z2, z2 = b2·x − a2·y).
        let (mut z1, mut z2) = (0.0, 0.0);
        if let Some(&x0) = x.first() {
            let dc = (self.b0 + self.b1 + self.b2) / (1.0 + self.a1 + self.a2);
            z2 = x0 * (self.b2 - self.a2 * dc);
            z1 = x0 * (self.b1 - self.a1 * dc) + z2;
        }
        x.iter()
            .map(|&xi| {
                let yi = self.b0 * xi + z1;
                z1 = self.b1 * xi - self.a1 * yi + z2;
                z2 = self.b2 * xi - self.a2 * yi;
                yi
            })
            .collect()
    }
}

fn butter_lowpass_sections(order: usize, cutoff_hz: f64, fs: f64) -> Vec<Biquad> {
    let wc = (std::f64::consts::PI * cutoff_hz / fs).tan();
    let n_sec = order / 2;
    (0..n_sec)
        .map(|k| {
            let theta = std::f64::consts::PI * (2 * k + 1) as f64 / (2 * order) as f64;
            let q = 1.0 / (2.0 * theta.sin());
            let k2 = wc * wc;
            let norm = 1.0 / (1.0 + wc / q + k2);
            Biquad {
                b0: k2 * norm,
                b1: 2.0 * k2 * norm,
                b2: k2 * norm,
                a1: 2.0 * (k2 - 1.0) * norm,
                a2: (1.0 - wc / q + k2) * norm,
            }
        })
        .collect()
}

/// Zero-phase Butterworth low-pass (forward–backward, odd reflection padding).
pub fn lowpass_filtfilt(x: &[f64], cutoff_hz: f64, fs: f64, order: usize) -> Vec<f64> {
    if x.len() < 4 || cutoff_hz <= 0.0 || cutoff_hz >= fs / 2.0 {
        return x.to_vec();
    }
    let sections = butter_lowpass_sections(order.max(2) & !1, cutoff_hz, fs);
    let pad = (3 * order * 4).min(x.len() - 1);
    let mut ext = Vec::with_capacity(x.len() + 2 * pad);
    for i in (1..=pad).rev() {
        ext.push(2.0 * x[0] - x[i]);
    }
    ext.extend_from_slice(x);
    let n = x.len();
    for i in 1..=pad {
        ext.push(2.0 * x[n - 1] - x[n - 1 - i]);
    }
    let mut y = ext;
    for s in &sections {
        y = s.run(&y);
    }
    y.reverse();
    for s in &sections {
        y = s.run(&y);
    }
    y.reverse();
    y[pad..pad + n].to_vec()
}

// ------------------------------------------------- Savitzky–Golay (regression)

/// Solves the small dense system `a · x = b` (Gaussian elimination, partial pivoting).
fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Vec<f64> {
    let n = b.len();
    for col in 0..n {
        let piv = (col..n).max_by(|&i, &j| a[i][col].abs().partial_cmp(&a[j][col].abs()).unwrap()).unwrap();
        a.swap(col, piv);
        b.swap(col, piv);
        let d = a[col][col];
        if d.abs() < 1e-300 {
            continue;
        }
        for row in col + 1..n {
            let f = a[row][col] / d;
            for k in col..n {
                a[row][k] -= f * a[col][k];
            }
            b[row] -= f * b[col];
        }
    }
    let mut x = vec![0.0; n];
    for row in (0..n).rev() {
        let mut s = b[row];
        for k in row + 1..n {
            s -= a[row][k] * x[k];
        }
        x[row] = if a[row][row].abs() < 1e-300 { 0.0 } else { s / a[row][row] };
    }
    x
}

/// Least-squares polynomial fit of `y` sampled at offsets `t` (any spacing).
/// Returns coefficients c0 + c1 t + c2 t² + …
pub fn polyfit(t: &[f64], y: &[f64], order: usize) -> Vec<f64> {
    let m = order + 1;
    let mut ata = vec![vec![0.0; m]; m];
    let mut aty = vec![0.0; m];
    for (&ti, &yi) in t.iter().zip(y) {
        let mut p = vec![1.0; m];
        for k in 1..m {
            p[k] = p[k - 1] * ti;
        }
        for r in 0..m {
            aty[r] += p[r] * yi;
            for c in 0..m {
                ata[r][c] += p[r] * p[c];
            }
        }
    }
    solve(ata, aty)
}

/// Savitzky–Golay filter: local polynomial regression of order `order` in a
/// centred window of `window` samples; returns the `deriv`-th derivative
/// (per second, using sample spacing `dt`). Edges use the polynomial fitted to
/// the first/last full window, like `scipy.signal.savgol_filter(mode="interp")`.
pub fn savgol(x: &[f64], window: usize, order: usize, deriv: usize, dt: f64) -> Vec<f64> {
    let n = x.len();
    let mut window = window.max(order + 2) | 1;
    if window > n {
        window = if n % 2 == 1 { n } else { n.saturating_sub(1) };
    }
    if window < order + 1 || n == 0 {
        return vec![0.0; n];
    }
    let half = window / 2;
    let offsets: Vec<f64> = (0..window).map(|i| i as f64 - half as f64).collect();
    // Convolution weights for the centre point: row `deriv` of (AᵀA)⁻¹Aᵀ, times deriv!
    let m = order + 1;
    let mut ata = vec![vec![0.0; m]; m];
    for &t in &offsets {
        let mut p = vec![1.0; m];
        for k in 1..m {
            p[k] = p[k - 1] * t;
        }
        for r in 0..m {
            for c in 0..m {
                ata[r][c] += p[r] * p[c];
            }
        }
    }
    let mut e = vec![0.0; m];
    e[deriv.min(order)] = 1.0;
    let row = solve(ata, e); // (AᵀA)⁻¹ e_d  (AᵀA symmetric)
    let fact: f64 = (1..=deriv).map(|k| k as f64).product();
    let weights: Vec<f64> = offsets
        .iter()
        .map(|&t| {
            let mut p = 1.0;
            let mut s = 0.0;
            for &r in row.iter() {
                s += r * p;
                p *= t;
            }
            s * fact / dt.powi(deriv as i32)
        })
        .collect();
    let mut y = vec![0.0; n];
    for i in half..n - half {
        y[i] = weights.iter().zip(&x[i - half..=i + half]).map(|(w, v)| w * v).sum();
    }
    // Edges: evaluate the derivative of the polynomial fitted to the edge window.
    let edge = |start: usize, idxs: std::ops::Range<usize>, y: &mut [f64]| {
        let ts: Vec<f64> = (0..window).map(|k| k as f64).collect();
        let c = polyfit(&ts, &x[start..start + window], order);
        for i in idxs {
            let t = (i - start) as f64;
            // d^deriv/dt^deriv of sum c_k t^k
            let mut s = 0.0;
            for (k, &ck) in c.iter().enumerate() {
                if k >= deriv {
                    let coef: f64 = ((k - deriv + 1)..=k).map(|v| v as f64).product();
                    s += ck * coef * t.powi((k - deriv) as i32);
                }
            }
            y[i] = s / dt.powi(deriv as i32);
        }
    };
    edge(0, 0..half, &mut y);
    edge(n - window, n - half..n, &mut y);
    y
}

/// Odd window length (in samples) closest to `seconds * fs`, at least `min`.
pub fn odd_window(seconds: f64, fs: f64, min: usize) -> usize {
    let w = (seconds * fs).round() as usize;
    w.max(min) | 1
}

// ------------------------------------------------------------- envelopes ---

/// Short-time energy in dB (relative to full scale) with a Hann window.
pub fn frame_energy_db(x: &[f32], fs: f64, win_s: f64, hop_s: f64) -> (Vec<f64>, f64) {
    let win = ((win_s * fs) as usize).max(4);
    let hop = ((hop_s * fs) as usize).max(1);
    let w: Vec<f64> = (0..win).map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / win as f64).cos()).collect();
    let wsum: f64 = w.iter().map(|v| v * v).sum();
    let n_frames = if x.len() >= win { 1 + (x.len() - win) / hop } else { 1 };
    let mut out = Vec::with_capacity(n_frames);
    for f in 0..n_frames {
        let s = f * hop;
        let mut e = 0.0;
        for i in 0..win {
            let v = *x.get(s + i).unwrap_or(&0.0) as f64;
            e += (v * w[i]).powi(2);
        }
        out.push(10.0 * (e / wsum + 1e-12).log10());
    }
    // frame f is centred at s + win/2
    (out, hop as f64 / fs)
}

/// Percentile (0..100) of a slice (copying).
pub fn percentile(x: &[f64], p: f64) -> f64 {
    if x.is_empty() {
        return f64::NAN;
    }
    let mut v: Vec<f64> = x.iter().copied().filter(|v| v.is_finite()).collect();
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pos = (p / 100.0) * (v.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    v[lo] + (v[hi] - v[lo]) * (pos - lo as f64)
}

// ----------------------------------------------------------- spectrogram ---

/// Greyscale spectrogram for display. Returns (pixels row-major top=highest
/// frequency, width, height, seconds per column, max frequency).
pub fn spectrogram_u8(x: &[f32], fs: f64, win_s: f64, hop_s: f64, fmax: f64, height: usize, dyn_range_db: f64) -> (Vec<u8>, usize, usize, f64, f64) {
    let win = ((win_s * fs) as usize).max(16);
    let hop = ((hop_s * fs) as usize).max(1);
    let nfft = win.next_power_of_two().max(256);
    let mut planner = RealFftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(nfft);
    let window: Vec<f64> = (0..win).map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / win as f64).cos()).collect();
    let n_frames = if x.len() > win { 1 + (x.len() - win) / hop } else { 1 };
    let bin_hz = fs / nfft as f64;
    let max_bin = ((fmax.min(fs / 2.0)) / bin_hz).floor() as usize;
    let mut spec = vec![0.0f64; n_frames * height];
    let mut inbuf = fft.make_input_vec();
    let mut outbuf = fft.make_output_vec();
    // pre-emphasis to flatten the spectral tilt
    let mut prev = 0.0f64;
    let xe: Vec<f64> = x
        .iter()
        .map(|&v| {
            let v = v as f64;
            let y = v - 0.95 * prev;
            prev = v;
            y
        })
        .collect();
    for f in 0..n_frames {
        let s = f * hop;
        for i in 0..nfft {
            inbuf[i] = if i < win { xe.get(s + i).copied().unwrap_or(0.0) * window[i] } else { 0.0 };
        }
        fft.process(&mut inbuf, &mut outbuf).ok();
        for r in 0..height {
            // row 0 = top = fmax
            let frac = 1.0 - (r as f64 + 0.5) / height as f64;
            let b = ((frac * max_bin as f64) as usize).min(outbuf.len() - 1);
            let p = outbuf[b].norm_sqr();
            spec[r * n_frames + f] = 10.0 * (p + 1e-20).log10();
        }
    }
    let top = percentile(&spec, 99.5);
    let lo = top - dyn_range_db;
    let pixels = spec
        .iter()
        .map(|&v| {
            let t = ((v - lo) / (top - lo)).clamp(0.0, 1.0);
            (t * 255.0) as u8
        })
        .collect();
    (pixels, n_frames, height, hop as f64 / fs, max_bin as f64 * bin_hz)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn savgol_derivative_of_quadratic_is_exact() {
        let dt = 0.004;
        let x: Vec<f64> = (0..100).map(|i| {
            let t = i as f64 * dt;
            3.0 * t * t + 2.0 * t + 1.0
        }).collect();
        let v = savgol(&x, 9, 2, 1, dt);
        let a = savgol(&x, 9, 2, 2, dt);
        for i in [0usize, 4, 50, 95, 99] {
            let t = i as f64 * dt;
            assert!((v[i] - (6.0 * t + 2.0)).abs() < 1e-6, "v[{i}]={}", v[i]);
            assert!((a[i] - 6.0).abs() < 1e-5, "a[{i}]={}", a[i]);
        }
    }

    #[test]
    fn resample_preserves_low_frequency_sine() {
        let fs = 48000.0;
        let x: Vec<f32> = (0..48000).map(|i| (2.0 * std::f64::consts::PI * 200.0 * i as f64 / fs).sin() as f32).collect();
        let y = resample(&x, fs, 16000.0);
        assert_eq!(y.len(), 16000);
        let err: f64 = (2000..14000)
            .map(|i| (y[i] as f64 - (2.0 * std::f64::consts::PI * 200.0 * i as f64 / 16000.0).sin()).abs())
            .fold(0.0, f64::max);
        assert!(err < 0.01, "max err {err}");
    }

    #[test]
    fn filtfilt_keeps_dc_and_removes_high_freq() {
        let fs = 250.0;
        let x: Vec<f64> = (0..500).map(|i| 5.0 + (2.0 * std::f64::consts::PI * 60.0 * i as f64 / fs).sin()).collect();
        let y = lowpass_filtfilt(&x, 15.0, fs, 4);
        for v in &y[50..450] {
            assert!((v - 5.0).abs() < 0.02, "{v}");
        }
    }
}
