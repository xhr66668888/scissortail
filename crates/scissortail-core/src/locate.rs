//! Locating the target word(s) inside a trial recording.
//!
//! Two engines:
//! * `Phonetic` (default): MMS wav2vec2 CTC forced alignment of the
//!   romanized IPA (carrier phrase + target, with `*` wildcards that absorb
//!   noise and unrelated speech).
//! * `Template`: Kokoro synthesizes the phrase from IPA, and the synthetic
//!   "sound template" is matched to the recording with subsequence DTW.
//!
//! Without a target word (or when it cannot be found) the main stretch of
//! speech is segmented from the energy envelope instead (`Speech`).
//!
//! All are followed by acoustic boundary refinement (segmented regression on
//! the energy envelope).

use serde::{Deserialize, Serialize};

use crate::aligner::{forced_align, LogProbs, PhoneticAligner};
use crate::dsp::{frame_energy_db, percentile, resample};
use crate::error::{Error, Result};
use crate::ipa::{romanize, MMS_STAR};
use crate::template::{broad_features, subsequence_dtw, FEAT_FS, FEAT_HOP_S};
use crate::trial::Audio;
use crate::tts::{Synthesis, Tts};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WordSpec {
    /// Display label (e.g. "tadat"); defaults to the IPA.
    #[serde(default)]
    pub label: String,
    /// Target word in IPA, e.g. "tˈɑdɑt".
    pub ipa: String,
    /// Optional file-name fragment; when it matches a trial's file name only
    /// the matching word(s) are tried for that trial (e.g. "ad_").
    #[serde(default)]
    pub file_pattern: String,
}

impl WordSpec {
    pub fn display(&self) -> String {
        if self.label.trim().is_empty() { self.ipa.clone() } else { self.label.clone() }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Engine {
    Phonetic,
    Template,
    /// Energy-based speech segmentation (no transcript).
    Speech,
}

impl Engine {
    pub fn as_str(&self) -> &'static str {
        match self {
            Engine::Phonetic => "phonetic",
            Engine::Template => "template",
            Engine::Speech => "speech",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LocateConfig {
    pub engine: Engine,
    /// Carrier phrase in IPA with `*` standing for the target word,
    /// e.g. "ɪts ə * əɡˈɛn". Empty = the word is spoken in isolation.
    pub carrier: String,
    pub words: Vec<WordSpec>,
    pub max_occurrences: usize,
    /// Minimum mean posterior of the target's letters to accept an occurrence.
    pub min_confidence: f64,
    pub refine_boundaries: bool,
    pub tts_voices: Vec<String>,
    pub tts_speeds: Vec<f32>,
}

impl Default for LocateConfig {
    fn default() -> Self {
        LocateConfig {
            engine: Engine::Phonetic,
            carrier: String::new(),
            words: vec![],
            max_occurrences: 3,
            min_confidence: 0.25,
            refine_boundaries: true,
            tts_voices: vec!["af_heart".into(), "am_adam".into()],
            tts_speeds: vec![0.75, 1.0, 1.3],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhoneSpan {
    pub symbol: String,
    pub start_s: f64,
    pub end_s: f64,
    pub prob: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Occurrence {
    pub word_index: usize,
    pub word: String,
    pub start_s: f64,
    pub end_s: f64,
    /// Boundaries before acoustic refinement.
    pub raw_start_s: f64,
    pub raw_end_s: f64,
    pub phrase_start_s: f64,
    pub phrase_end_s: f64,
    /// 0..1 (phonetic: mean letter posterior; template: 1 − normalized DTW cost).
    pub confidence: f64,
    pub phones: Vec<PhoneSpan>,
    pub engine: Engine,
}

/// Transcript for one word: "* carrier-with-target *".
pub fn transcript(carrier: &str, target_ipa: &str) -> (String, usize) {
    let carrier = carrier.trim();
    if carrier.is_empty() || !carrier.contains('*') {
        return (format!("* {} *", target_ipa.trim()), 1);
    }
    let words: Vec<&str> = carrier.split_whitespace().collect();
    let mut out = vec!["*".to_string()];
    let mut target_word = 1;
    for w in &words {
        if *w == "*" {
            target_word = out.len();
            out.push(target_ipa.trim().to_string());
        } else {
            out.push(w.to_string());
        }
    }
    out.push("*".into());
    (out.join(" "), target_word)
}

/// Words to try for a trial: those whose pattern matches the file name, or all.
pub fn candidate_words(words: &[WordSpec], file_name: &str) -> Vec<usize> {
    let matched: Vec<usize> = words
        .iter()
        .enumerate()
        .filter(|(_, w)| !w.file_pattern.trim().is_empty() && file_name.contains(w.file_pattern.trim()))
        .map(|(i, _)| i)
        .collect();
    if matched.is_empty() { (0..words.len()).collect() } else { matched }
}

// ------------------------------------------------------------ phonetic ----

struct PhoneticHit {
    occ: Occurrence,
    /// Mean posterior of every transcript letter (carrier + target), minus a
    /// small penalty per carrier word left out (see `align_best_variant`).
    fit: f64,
}

fn align_one(lp: &LogProbs, carrier: &str, word: &WordSpec, word_index: usize, t0: usize, t1: usize) -> Option<PhoneticHit> {
    let (text, target_word) = transcript(carrier, &word.ipa);
    let rom = romanize(&text);
    if rom.tokens.iter().all(|&t| t == MMS_STAR) {
        return None;
    }
    let al = forced_align(lp, &rom.tokens, t0, t1)?;
    let fs = lp.frame_s;
    let idx_target: Vec<usize> = (0..rom.tokens.len()).filter(|&i| rom.token_word[i] == target_word).collect();
    let idx_phrase: Vec<usize> = (0..rom.tokens.len()).filter(|&i| rom.tokens[i] != MMS_STAR).collect();
    if idx_target.is_empty() {
        return None;
    }
    let first = al.token_frames[idx_target[0]].0;
    let last = al.token_frames[*idx_target.last().unwrap()].1;
    let pfirst = al.token_frames[idx_phrase[0]].0;
    let plast = al.token_frames[*idx_phrase.last().unwrap()].1;
    let conf = idx_target.iter().map(|&i| al.token_prob[i]).sum::<f64>() / idx_target.len() as f64;
    let fit = idx_phrase.iter().map(|&i| al.token_prob[i]).sum::<f64>() / idx_phrase.len() as f64;
    // phone spans: group letters by source phone
    let mut phones: Vec<PhoneSpan> = Vec::new();
    let mut cur_phone = usize::MAX;
    for &i in &idx_target {
        let (a, b) = al.token_frames[i];
        let p = rom.token_phone[i];
        let symbol = if p == usize::MAX { "?".to_string() } else { rom.phones[p].symbol.clone() };
        if p == cur_phone && p != usize::MAX {
            let last = phones.last_mut().unwrap();
            last.end_s = b as f64 * fs;
            last.prob = (last.prob + al.token_prob[i]) / 2.0;
        } else {
            phones.push(PhoneSpan { symbol, start_s: a as f64 * fs, end_s: b as f64 * fs, prob: al.token_prob[i] });
            cur_phone = p;
        }
    }
    // phones as contiguous intervals: each runs until the next one starts
    for k in 0..phones.len().saturating_sub(1) {
        phones[k].end_s = phones[k + 1].start_s.max(phones[k].end_s);
    }
    Some(PhoneticHit {
        occ: Occurrence {
            word_index,
            word: word.display(),
            start_s: first as f64 * fs,
            end_s: last as f64 * fs,
            raw_start_s: first as f64 * fs,
            raw_end_s: last as f64 * fs,
            phrase_start_s: pfirst as f64 * fs,
            phrase_end_s: plast as f64 * fs,
            confidence: conf,
            phones,
            engine: Engine::Phonetic,
        },
        fit,
    })
}

/// Aligns the target with the full carrier phrase and with variants that
/// leave out leading / trailing carrier words (recordings that start or stop
/// mid-phrase), keeping the variant whose letters fit the audio best.
fn align_best_variant(lp: &LogProbs, carrier: &str, word: &WordSpec, word_index: usize, t0: usize, t1: usize) -> Option<PhoneticHit> {
    // Only chosen when the full carrier fits clearly worse (validated: 0.15 keeps
    // full-carrier accuracy on intact recordings).
    const DROP_PENALTY: f64 = 0.15;
    let words: Vec<&str> = carrier.split_whitespace().collect();
    let Some(star) = words.iter().position(|w| *w == "*") else {
        return align_one(lp, carrier, word, word_index, t0, t1);
    };
    let (pre, post) = (&words[..star], &words[star + 1..]);
    let mut best: Option<PhoneticHit> = None;
    for k in 0..=pre.len() {
        for m in 0..=post.len() {
            if k + m > 0 && k == pre.len() && m == post.len() && !(pre.is_empty() && post.is_empty()) {
                continue; // never fall back to the bare word when a carrier is given
            }
            let mut v: Vec<&str> = pre[k..].to_vec();
            v.push("*");
            v.extend_from_slice(&post[..post.len() - m]);
            if let Some(mut hit) = align_one(lp, &v.join(" "), word, word_index, t0, t1) {
                hit.fit -= DROP_PENALTY * (k + m) as f64;
                if best.as_ref().map(|b| hit.fit > b.fit).unwrap_or(true) {
                    best = Some(hit);
                }
            }
        }
    }
    best
}

/// Speech-activity mask from audio (10 ms frames, true = speech).
fn speech_mask(audio: &Audio) -> (Vec<bool>, f64) {
    let a = speech_activity(audio);
    (a.mask, a.hop)
}

struct SpeechActivity {
    /// Frame energies (dB), 25 ms frames every `hop` seconds.
    energy: Vec<f64>,
    mask: Vec<bool>,
    hop: f64,
    floor: f64,
    peak: f64,
}

fn speech_activity(audio: &Audio) -> SpeechActivity {
    let (energy, hop) = frame_energy_db(&audio.samples, audio.fs, 0.025, 0.01);
    let peak = percentile(&energy, 97.0);
    let floor = percentile(&energy, 5.0).max(peak - 55.0);
    let span = (peak - floor).max(6.0);
    let mask = energy.iter().map(|&v| (v - floor) / span > 0.25).collect();
    SpeechActivity { energy, mask, hop, floor, peak }
}

/// The main stretch of speech in a recording, without a transcript: frames
/// above the adaptive energy threshold, pauses shorter than 250 ms bridged,
/// blips under 80 ms dropped; the segment with the most energy above the
/// noise floor wins. Boundaries are then refined like aligned words.
pub fn locate_speech(audio: &Audio) -> Vec<Occurrence> {
    const MAX_GAP_S: f64 = 0.25;
    const MIN_LEN_S: f64 = 0.08;
    let a = speech_activity(audio);
    if !a.peak.is_finite() || a.peak - a.floor < 6.0 {
        return vec![];
    }
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut i = 0;
    while i < a.mask.len() {
        if !a.mask[i] {
            i += 1;
            continue;
        }
        let start = i;
        while i < a.mask.len() && a.mask[i] {
            i += 1;
        }
        match runs.last_mut() {
            Some(last) if (start - last.1) as f64 * a.hop <= MAX_GAP_S => last.1 = i,
            _ => runs.push((start, i)),
        }
    }
    let mass = |r: &(usize, usize)| a.energy[r.0..r.1].iter().map(|v| (v - a.floor).max(0.0)).sum::<f64>();
    let Some(best) = runs
        .iter()
        .filter(|r| (r.1 - r.0) as f64 * a.hop >= MIN_LEN_S)
        .max_by(|x, y| mass(x).partial_cmp(&mass(y)).unwrap())
    else {
        return vec![];
    };
    // frame k covers [k·hop, k·hop + 25 ms]
    let start_s = best.0 as f64 * a.hop + 0.0125 - a.hop / 2.0;
    let end_s = ((best.1 - 1) as f64 * a.hop + 0.0125 + a.hop / 2.0).min(audio.duration());
    let mut occ = Occurrence {
        word_index: 0,
        word: "speech".into(),
        start_s,
        end_s,
        raw_start_s: start_s,
        raw_end_s: end_s,
        phrase_start_s: start_s,
        phrase_end_s: end_s,
        confidence: ((a.peak - a.floor - 10.0) / 30.0).clamp(0.0, 1.0),
        phones: vec![],
        engine: Engine::Speech,
    };
    refine_boundaries(audio, &mut occ);
    occ.phrase_start_s = occ.start_s;
    occ.phrase_end_s = occ.end_s;
    vec![occ]
}

pub fn locate_phonetic(aligner: &PhoneticAligner, audio: &Audio, cfg: &LocateConfig, file_name: &str) -> Result<Vec<Occurrence>> {
    if cfg.words.is_empty() {
        return Err(Error::Invalid("no target word defined".into()));
    }
    let lp = aligner.log_probs(&audio.samples, audio.fs)?;
    let cands = candidate_words(&cfg.words, file_name);
    let best_in = |t0: usize, t1: usize| -> Option<PhoneticHit> {
        cands
            .iter()
            .filter_map(|&wi| align_best_variant(&lp, &cfg.carrier, &cfg.words[wi], wi, t0, t1))
            .max_by(|a, b| a.fit.partial_cmp(&b.fit).unwrap())
    };
    let mut found: Vec<Occurrence> = Vec::new();
    let Some(first) = best_in(0, lp.frames) else { return Ok(found) };
    let phrase_dur = (first.occ.phrase_end_s - first.occ.phrase_start_s).max(0.2);
    // The best alignment is always reported (flagged later when its confidence
    // is low); further repetitions must pass the confidence threshold.
    let accept = |o: &Occurrence, reference: f64| o.confidence >= cfg.min_confidence.max(0.5 * reference);
    if first.occ.confidence >= 0.03 {
        found.push(first.occ.clone());
    }
    // Further occurrences: align inside the largest remaining speech regions.
    if cfg.max_occurrences > 1 && !found.is_empty() {
        let (speech, hop) = speech_mask(audio);
        let to_frame = |s: f64| ((s / lp.frame_s).round() as usize).min(lp.frames);
        for _ in 1..cfg.max_occurrences {
            let margin = 0.25;
            let mut used = vec![false; lp.frames];
            for o in &found {
                for f in to_frame((o.phrase_start_s - margin).max(0.0))..to_frame(o.phrase_end_s + margin) {
                    used[f] = true;
                }
            }
            // contiguous free regions with their amount of speech
            let mut best_region: Option<(usize, usize, f64)> = None;
            let mut f = 0;
            while f < lp.frames {
                if used[f] {
                    f += 1;
                    continue;
                }
                let a = f;
                while f < lp.frames && !used[f] {
                    f += 1;
                }
                let (sa, sb) = ((a as f64 * lp.frame_s / hop) as usize, (f as f64 * lp.frame_s / hop) as usize);
                let sp = speech[sa.min(speech.len())..sb.min(speech.len())].iter().filter(|&&s| s).count() as f64 * hop;
                if best_region.map(|r| sp > r.2).unwrap_or(true) {
                    best_region = Some((a, f, sp));
                }
            }
            let Some((a, b, sp)) = best_region else { break };
            if sp < 0.4 * phrase_dur {
                break;
            }
            match best_in(a, b) {
                Some(hit) if accept(&hit.occ, found[0].confidence) => found.push(hit.occ),
                _ => break,
            }
        }
    }
    found.sort_by(|a, b| a.start_s.partial_cmp(&b.start_s).unwrap());
    Ok(found)
}

// ------------------------------------------------------------ template ----

pub struct PhraseTemplate {
    pub word_index: usize,
    pub synthesis: Synthesis,
    /// Target word span inside the synthetic phrase (seconds).
    pub target: (f64, f64),
    pub target_phones: Vec<(String, f64, f64)>,
    features: crate::template::Features,
}

/// Synthesizes phrase templates (all words × voices × speeds).
pub fn build_templates(tts: &Tts, cfg: &LocateConfig) -> Result<Vec<PhraseTemplate>> {
    let mut out = Vec::new();
    for (wi, w) in cfg.words.iter().enumerate() {
        let (text, target_word) = transcript(&cfg.carrier, &w.ipa);
        let phrase_ipa: String = text.split_whitespace().filter(|t| *t != "*").collect::<Vec<_>>().join(" ");
        // index of the target among non-wildcard words
        let target_pos = target_word - 1;
        let n_before: usize = text.split_whitespace().filter(|t| *t != "*").take(target_pos).map(|t| crate::ipa::phones(&crate::ipa::to_kokoro(t).text).len()).sum();
        let n_target = crate::ipa::phones(&crate::ipa::to_kokoro(&w.ipa).text).len();
        for voice in &cfg.tts_voices {
            for &speed in &cfg.tts_speeds {
                let syn = tts.synthesize(&phrase_ipa, voice, speed)?;
                if syn.phones.len() < n_before + n_target || n_target == 0 {
                    continue;
                }
                let tp: Vec<(String, f64, f64)> = syn.phones[n_before..n_before + n_target].iter().map(|p| (p.symbol.clone(), p.start_s, p.end_s)).collect();
                let target = (tp[0].1, tp[tp.len() - 1].2);
                let x16 = resample(&syn.samples, syn.fs, FEAT_FS);
                let features = broad_features(&x16);
                out.push(PhraseTemplate { word_index: wi, synthesis: syn, target, target_phones: tp, features });
            }
        }
    }
    Ok(out)
}

pub fn locate_template(templates: &[PhraseTemplate], audio: &Audio, cfg: &LocateConfig, file_name: &str) -> Result<Vec<Occurrence>> {
    let cands = candidate_words(&cfg.words, file_name);
    let x16 = resample(&audio.samples, audio.fs, FEAT_FS);
    let rec = broad_features(&x16);
    let mut hits: Vec<Occurrence> = Vec::new();
    for t in templates.iter().filter(|t| cands.contains(&t.word_index)) {
        for m in subsequence_dtw(&t.features, &rec, cfg.max_occurrences.max(1)) {
            let map = |sec: f64, first: bool| -> f64 {
                let f = (sec / FEAT_HOP_S).round() as usize;
                let js: Vec<usize> = m.path.iter().filter(|(i, _)| if first { *i >= f } else { *i <= f }).map(|(_, j)| *j).collect();
                let j = if first { js.first().copied() } else { js.last().copied() }.unwrap_or(if first { m.start } else { m.end });
                j as f64 * FEAT_HOP_S
            };
            let (mut s, mut e) = (map(t.target.0, true), map(t.target.1, false));
            // drop leading/trailing silence inside the mapped word
            let (mut a, mut b) = ((s / FEAT_HOP_S) as usize, ((e / FEAT_HOP_S) as usize).min(rec.frames.saturating_sub(1)));
            while a < b && rec.energy[a] < 0.35 {
                a += 1;
            }
            while b > a && rec.energy[b] < 0.35 {
                b -= 1;
            }
            if b > a + 5 {
                s = a as f64 * FEAT_HOP_S;
                e = b as f64 * FEAT_HOP_S;
            }
            let phones = t
                .target_phones
                .iter()
                .map(|(sym, ps, pe)| PhoneSpan { symbol: sym.clone(), start_s: map(*ps, true), end_s: map(*pe, false), prob: f64::NAN })
                .collect();
            hits.push(Occurrence {
                word_index: t.word_index,
                word: cfg.words[t.word_index].display(),
                start_s: s,
                end_s: e,
                raw_start_s: s,
                raw_end_s: e,
                phrase_start_s: m.start as f64 * FEAT_HOP_S,
                phrase_end_s: m.end as f64 * FEAT_HOP_S,
                confidence: (1.0 - m.cost / 1.5).clamp(0.0, 1.0),
                phones,
                engine: Engine::Template,
            });
        }
    }
    hits.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
    let mut keep: Vec<Occurrence> = Vec::new();
    for h in hits {
        if keep.len() >= cfg.max_occurrences {
            break;
        }
        let overlaps = keep.iter().any(|k| !(h.phrase_end_s < k.phrase_start_s || h.phrase_start_s > k.phrase_end_s));
        let good = keep.first().map(|k0| h.confidence >= k0.confidence - 0.08).unwrap_or(true);
        if !overlaps && good {
            keep.push(h);
        }
    }
    keep.sort_by(|a, b| a.start_s.partial_cmp(&b.start_s).unwrap());
    Ok(keep)
}

// ------------------------------------------------- boundary refinement ----

/// Continuous two-segment ("hinge") regression on the dB envelope:
/// y = c for t < k, y = c + s·(t − k) for t ≥ k (rising = onset) or the
/// mirror image (falling = offset). Returns the best breakpoint index.
fn hinge_breakpoint(y: &[f64], rising: bool) -> Option<usize> {
    let n = y.len();
    if n < 6 {
        return None;
    }
    let mut best = (f64::INFINITY, 0usize);
    for k in 2..n - 2 {
        // basis: 1 and h(t) = max(0, t-k) for rising, max(0, k-t) for falling
        let h: Vec<f64> = (0..n).map(|t| if rising { (t as f64 - k as f64).max(0.0) } else { (k as f64 - t as f64).max(0.0) }).collect();
        let (mut s1, mut sh, mut shh, mut sy, mut shy) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for t in 0..n {
            s1 += 1.0;
            sh += h[t];
            shh += h[t] * h[t];
            sy += y[t];
            shy += h[t] * y[t];
        }
        let det = s1 * shh - sh * sh;
        if det.abs() < 1e-12 {
            continue;
        }
        let c = (shh * sy - sh * shy) / det;
        let s = (s1 * shy - sh * sy) / det;
        if (rising && s <= 0.0) || (!rising && s <= 0.0) {
            continue;
        }
        let sse: f64 = (0..n).map(|t| (y[t] - c - s * h[t]).powi(2)).sum();
        if sse < best.0 {
            best = (sse, k);
        }
    }
    best.0.is_finite().then_some(best.1)
}

/// Snaps word boundaries to the acoustic onset (rise of the burst/voicing
/// energy) and, when the word is followed by silence, to the energy offset.
pub fn refine_boundaries(audio: &Audio, occ: &mut Occurrence) {
    let (e, hop) = frame_energy_db(&audio.samples, audio.fs, 0.005, 0.001);
    let n = e.len();
    if n < 20 {
        return;
    }
    let half_win = 0.0025; // centre offset of a 5 ms frame
    let idx = |t: f64| (((t - half_win) / hop).round().max(0.0) as usize).min(n - 1);
    // onset: search ±30 ms around the aligned onset, fit up to the local energy max
    let (a, b) = (idx(occ.start_s - 0.03), idx(occ.start_s + 0.03));
    if b > a + 8 {
        let peak_i = a + e[a..=b].iter().enumerate().max_by(|x, y| x.1.partial_cmp(y.1).unwrap()).map(|(i, _)| i).unwrap_or(0);
        let lo = a.saturating_sub(10);
        if peak_i > lo + 6 {
            if let Some(k) = hinge_breakpoint(&e[lo..=peak_i], true) {
                let t = (lo + k) as f64 * hop + half_win;
                if (t - occ.start_s).abs() <= 0.035 {
                    occ.start_s = t;
                }
            }
        }
    }
    // offset: only when ≥ 40 ms of near-silence follows (otherwise keep alignment)
    let floor = percentile(&e, 10.0);
    let (c, d) = (idx(occ.end_s - 0.03), idx(occ.end_s + 0.05));
    if d > c + 8 {
        let after = idx(occ.end_s);
        let quiet_after = e[after..d.min(n - 1)].iter().filter(|&&v| v < floor + 12.0).count() as f64 * hop;
        if quiet_after >= 0.04 {
            let peak_i = c + e[c..=after.max(c + 1).min(n - 1)].iter().enumerate().max_by(|x, y| x.1.partial_cmp(y.1).unwrap()).map(|(i, _)| i).unwrap_or(0);
            if d > peak_i + 6 {
                if let Some(k) = hinge_breakpoint(&e[peak_i..=d], false) {
                    let t = (peak_i + k) as f64 * hop + half_win;
                    if (t - occ.end_s).abs() <= 0.045 && t > occ.start_s + 0.05 {
                        occ.end_s = t;
                    }
                }
            }
        }
    }
    if let Some(p) = occ.phones.first_mut() {
        p.start_s = occ.start_s.min(p.end_s);
    }
    if let Some(p) = occ.phones.last_mut() {
        p.end_s = occ.end_s.max(p.start_s);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transcript_inserts_target() {
        let (t, i) = transcript("ɪts ə * əɡˈɛn", "tˈɑdɑt");
        assert_eq!(t, "* ɪts ə tˈɑdɑt əɡˈɛn *");
        assert_eq!(i, 3);
        let (t, i) = transcript("", "tˈɑdɑt");
        assert_eq!(t, "* tˈɑdɑt *");
        assert_eq!(i, 1);
    }

    #[test]
    fn speech_segment_bridges_short_pauses() {
        // noise floor, then two bursts 150 ms apart (one utterance), then a
        // short click far away
        let fs = 16000.0;
        let mut x = vec![0f32; (2.0 * fs) as usize];
        let mut seed = 1u32;
        for v in x.iter_mut() {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            *v = ((seed >> 9) as f32 / (1u32 << 23) as f32 - 0.5) * 0.002;
        }
        let tone = |x: &mut Vec<f32>, a: f64, b: f64, amp: f32| {
            for i in (a * fs) as usize..(b * fs) as usize {
                x[i] += amp * (2.0 * std::f32::consts::PI * 180.0 * i as f32 / fs as f32).sin();
            }
        };
        tone(&mut x, 0.50, 0.80, 0.5);
        tone(&mut x, 0.95, 1.30, 0.5);
        tone(&mut x, 1.80, 1.83, 0.5);
        let occ = locate_speech(&Audio { fs, samples: x });
        assert_eq!(occ.len(), 1);
        let o = &occ[0];
        assert_eq!(o.engine, Engine::Speech);
        assert!((o.start_s - 0.50).abs() < 0.02, "onset {}", o.start_s);
        assert!((o.end_s - 1.30).abs() < 0.03, "offset {}", o.end_s);
        assert!(locate_speech(&Audio { fs, samples: vec![0.0; 16000] }).is_empty());
    }

    #[test]
    fn hinge_finds_rise() {
        let mut y = vec![-60.0; 20];
        for (i, v) in y.iter_mut().enumerate().skip(12) {
            *v = -60.0 + 5.0 * (i - 12) as f64;
        }
        assert_eq!(hinge_breakpoint(&y, true), Some(12));
    }
}
