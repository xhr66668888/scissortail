//! IPA handling: splitting a transcription into phones, normalizing it for
//! the Kokoro TTS vocabulary, and romanizing it for the MMS phonetic aligner.

use serde::Serialize;

/// Kokoro-82M v1.0 phoneme vocabulary (symbol, token id).
pub const KOKORO_VOCAB: &[(char, i64)] = &[
    ('$', 0), (';', 1), (':', 2), (',', 3), ('.', 4), ('!', 5), ('?', 6), ('—', 9), ('…', 10), ('"', 11),
    ('(', 12), (')', 13), ('“', 14), ('”', 15), (' ', 16), ('\u{0303}', 17), ('ʣ', 18), ('ʥ', 19), ('ʦ', 20),
    ('ʨ', 21), ('ᵝ', 22), ('ꭧ', 23), ('A', 24), ('I', 25), ('O', 31), ('Q', 33), ('S', 35), ('T', 36),
    ('W', 39), ('Y', 41), ('ᵊ', 42), ('a', 43), ('b', 44), ('c', 45), ('d', 46), ('e', 47), ('f', 48),
    ('h', 50), ('i', 51), ('j', 52), ('k', 53), ('l', 54), ('m', 55), ('n', 56), ('o', 57), ('p', 58),
    ('q', 59), ('r', 60), ('s', 61), ('t', 62), ('u', 63), ('v', 64), ('w', 65), ('x', 66), ('y', 67),
    ('z', 68), ('ɑ', 69), ('ɐ', 70), ('ɒ', 71), ('æ', 72), ('β', 75), ('ɔ', 76), ('ɕ', 77), ('ç', 78),
    ('ɖ', 80), ('ð', 81), ('ʤ', 82), ('ə', 83), ('ɚ', 85), ('ɛ', 86), ('ɜ', 87), ('ɟ', 90), ('ɡ', 92),
    ('ɥ', 99), ('ɨ', 101), ('ɪ', 102), ('ʝ', 103), ('ɯ', 110), ('ɰ', 111), ('ŋ', 112), ('ɳ', 113),
    ('ɲ', 114), ('ɴ', 115), ('ø', 116), ('ɸ', 118), ('θ', 119), ('œ', 120), ('ɹ', 123), ('ɾ', 125),
    ('ɻ', 126), ('ʁ', 128), ('ɽ', 129), ('ʂ', 130), ('ʃ', 131), ('ʈ', 132), ('ʧ', 133), ('ʊ', 135),
    ('ʋ', 136), ('ʌ', 138), ('ɣ', 139), ('ɤ', 140), ('χ', 142), ('ʎ', 143), ('ʒ', 147), ('ʔ', 148),
    ('ˈ', 156), ('ˌ', 157), ('ː', 158), ('ʰ', 162), ('ʲ', 164), ('↓', 169), ('→', 171), ('↗', 172),
    ('↘', 173), ('ᵻ', 177),
];

pub fn kokoro_id(c: char) -> Option<i64> {
    KOKORO_VOCAB.iter().find(|(s, _)| *s == c).map(|(_, id)| *id)
}

/// Characters that never carry a phone of their own.
fn is_modifier(c: char) -> bool {
    matches!(c, 'ˈ' | 'ˌ' | 'ː' | 'ˑ' | 'ʰ' | 'ʲ' | 'ʷ' | 'ˠ' | 'ˤ' | 'ⁿ' | 'ˡ' | '\'' | ':')
        || ('\u{0300}'..='\u{036F}').contains(&c) // combining diacritics, tie bars
}

fn clean(input: &str) -> String {
    input
        .chars()
        .filter(|c| !matches!(c, '/' | '[' | ']' | '{' | '}' | '|' | '‖' | '.' | '\u{200B}'))
        .map(|c| match c {
            '\'' | '’' | 'ˊ' => 'ˈ',
            ',' => 'ˌ',
            ':' => 'ː',
            'g' => 'ɡ',
            '\t' | '\n' => ' ',
            c => c,
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// One phone of an IPA transcription.
#[derive(Debug, Clone, Serialize)]
pub struct Phone {
    /// Display form, e.g. "tʰ" or "ɑː".
    pub symbol: String,
    /// Index of the word (space-separated) the phone belongs to.
    pub word: usize,
    pub stressed: bool,
}

/// Splits an IPA string into phones (base symbol + following modifiers;
/// affricates written with a tie bar or as ligatures stay together).
pub fn phones(ipa: &str) -> Vec<Phone> {
    let s = clean(ipa);
    let chars: Vec<char> = s.chars().collect();
    let mut out: Vec<Phone> = Vec::new();
    let mut word = 0;
    let mut pending_stress = false;
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c == ' ' {
            word += 1;
            i += 1;
            continue;
        }
        if c == 'ˈ' || c == 'ˌ' {
            pending_stress = pending_stress || c == 'ˈ';
            i += 1;
            continue;
        }
        if is_modifier(c) {
            if let Some(last) = out.last_mut() {
                if c == '\u{0361}' || c == '\u{035C}' {
                    // tie bar: absorb the next base character
                    last.symbol.push(c);
                    if let Some(&n) = chars.get(i + 1) {
                        last.symbol.push(n);
                        i += 1;
                    }
                } else {
                    last.symbol.push(c);
                }
            }
            i += 1;
            continue;
        }
        out.push(Phone { symbol: c.to_string(), word, stressed: pending_stress });
        pending_stress = false;
        i += 1;
    }
    out
}

#[derive(Debug, Clone, Serialize)]
pub struct KokoroInput {
    /// Normalized phoneme string fed to Kokoro.
    pub text: String,
    /// Symbols that Kokoro cannot pronounce (dropped).
    pub unknown: Vec<String>,
}

/// Maps a researcher-entered IPA string onto Kokoro's phoneme inventory.
pub fn to_kokoro(ipa: &str) -> KokoroInput {
    let mut s = clean(ipa);
    for (from, to) in [
        ("t͡ʃ", "ʧ"), ("d͡ʒ", "ʤ"), ("t͡s", "ʦ"), ("d͡z", "ʣ"), ("tʃ", "ʧ"), ("dʒ", "ʤ"),
        ("eɪ", "A"), ("aɪ", "I"), ("oʊ", "O"), ("aʊ", "W"), ("ɔɪ", "Y"), ("əʊ", "Q"),
        ("ɝ", "ɜɹ"), ("ɫ", "l"), ("ʍ", "w"), ("r", "ɹ"), ("ɘ", "ə"), ("ɵ", "ə"),
    ] {
        s = s.replace(from, to);
    }
    let mut text = String::new();
    let mut unknown = Vec::new();
    for c in s.chars() {
        if c == '\u{0361}' || c == '\u{035C}' || c == '\u{0329}' || c == '\u{032F}' {
            continue;
        }
        if kokoro_id(c).is_some() {
            text.push(c);
        } else if !unknown.contains(&c.to_string()) {
            unknown.push(c.to_string());
        }
    }
    KokoroInput { text, unknown }
}

/// MMS forced-alignment dictionary (romanized letters). `-` = CTC blank,
/// `*` = wildcard appended by Scissortail.
pub const MMS_LETTERS: &str = "-aienoutsrmklgdhybpwcvjzf'qx";
pub const MMS_BLANK: usize = 0;
pub const MMS_STAR: usize = 28;

pub fn mms_index(c: char) -> Option<usize> {
    let c = match c {
        'g' => 'g',
        other => other,
    };
    // order in MMS_FA's dictionary
    const ORDER: [char; 28] = ['-', 'a', 'i', 'e', 'n', 'o', 'u', 't', 's', 'r', 'm', 'k', 'l', 'd', 'g', 'h', 'y', 'b', 'p', 'w', 'c', 'v', 'j', 'z', 'f', '\'', 'q', 'x'];
    ORDER.iter().position(|&o| o == c)
}

fn romanize_base(c: char) -> Option<&'static str> {
    Some(match c {
        'a' | 'ɑ' | 'ɒ' | 'æ' | 'ɐ' | 'ʌ' => "a",
        'e' | 'ɛ' | 'ɜ' | 'ə' | 'ɘ' | 'ᵊ' => "e",
        'ɚ' | 'ɝ' => "er",
        'i' | 'ɪ' | 'ɨ' | 'ᵻ' => "i",
        'o' | 'ɔ' | 'ɵ' | 'ø' => "o",
        'œ' => "oe",
        'u' | 'ʊ' | 'ʉ' | 'ɯ' | 'y' => "u",
        'A' => "ei",
        'I' => "ai",
        'O' | 'Q' => "ou",
        'W' => "au",
        'Y' => "oi",
        'p' => "p",
        'b' => "b",
        't' | 'ʈ' => "t",
        'd' | 'ɖ' => "d",
        'k' | 'c' | 'q' => "k",
        'ɡ' | 'g' | 'ɟ' => "g",
        'ʔ' | 'ʕ' => "",
        'm' | 'ɱ' => "m",
        'n' | 'ɳ' | 'ɴ' => "n",
        'ŋ' => "ng",
        'ɲ' => "ny",
        'r' | 'ɹ' | 'ɾ' | 'ɻ' | 'ʁ' | 'ʀ' | 'ɽ' => "r",
        'l' | 'ɫ' | 'ɭ' | 'ɬ' => "l",
        'ʎ' => "ly",
        'f' | 'ɸ' => "f",
        'v' | 'ʋ' => "v",
        'θ' => "th",
        'ð' => "dh",
        's' => "s",
        'z' => "z",
        'ʃ' | 'ʂ' | 'ɕ' => "sh",
        'ʒ' | 'ʐ' | 'ʑ' => "zh",
        'ç' | 'h' | 'ɦ' | 'ħ' => "h",
        'x' | 'χ' => "kh",
        'ɣ' => "g",
        'j' | 'ɥ' | 'ʝ' => "y",
        'w' | 'ɰ' => "w",
        'ʍ' => "wh",
        'β' => "b",
        'ʧ' | 'ʨ' => "ch",
        'ʤ' | 'ʥ' => "j",
        'ʦ' => "ts",
        'ʣ' => "dz",
        _ => return None,
    })
}

/// A romanized transcript for the MMS aligner, with each letter linked to
/// the word and phone it came from.
#[derive(Debug, Clone, Serialize)]
pub struct Romanized {
    /// Space-separated romanized words (for display).
    pub text: String,
    /// Token ids (MMS dictionary) for every letter, in order.
    pub tokens: Vec<usize>,
    /// For each token: word index.
    pub token_word: Vec<usize>,
    /// For each token: phone index within `phones` (usize::MAX for spelled input).
    pub token_phone: Vec<usize>,
    pub phones: Vec<Phone>,
    pub unknown: Vec<String>,
}

/// True when the string looks like ordinary spelling rather than IPA.
pub fn looks_like_spelling(s: &str) -> bool {
    s.chars().all(|c| c.is_ascii_alphabetic() || c == ' ' || c == '\'' || c == '*' || c == '-' || c == '.' || c == ',')
        && s.chars().any(|c| c.is_ascii_alphabetic())
        && s.chars().any(|c| c.is_ascii_uppercase() || "cqxy".contains(c))
}

/// Romanizes a transcription. Words equal to `*` become wildcard tokens.
/// Ordinary spelling (e.g. "It's a * again") is passed through letter by letter.
pub fn romanize(text: &str) -> Romanized {
    let mut out = Romanized { text: String::new(), tokens: vec![], token_word: vec![], token_phone: vec![], phones: vec![], unknown: vec![] };
    let spelled = looks_like_spelling(text);
    let mut words_out: Vec<String> = Vec::new();
    let mut phone_offset = 0usize;
    for (wi, word) in text.split_whitespace().enumerate() {
        if word == "*" {
            out.tokens.push(MMS_STAR);
            out.token_word.push(wi);
            out.token_phone.push(usize::MAX);
            words_out.push("*".into());
            continue;
        }
        let mut rom = String::new();
        if spelled {
            for c in word.chars() {
                let lc = c.to_ascii_lowercase();
                if let Some(id) = mms_index(lc) {
                    if id != MMS_BLANK {
                        out.tokens.push(id);
                        out.token_word.push(wi);
                        out.token_phone.push(usize::MAX);
                        rom.push(lc);
                    }
                }
            }
        } else {
            let k = to_kokoro(word).text; // merges diphthongs/affricates
            let ph = phones(&k);
            for (pi, p) in ph.iter().enumerate() {
                let base = p.symbol.chars().next().unwrap_or(' ');
                match romanize_base(base) {
                    Some(r) => {
                        for lc in r.chars() {
                            if let Some(id) = mms_index(lc) {
                                out.tokens.push(id);
                                out.token_word.push(wi);
                                out.token_phone.push(phone_offset + pi);
                                rom.push(lc);
                            }
                        }
                    }
                    None => {
                        if !out.unknown.contains(&p.symbol) {
                            out.unknown.push(p.symbol.clone());
                        }
                    }
                }
            }
            for mut p in ph {
                p.word = wi;
                out.phones.push(p);
            }
            phone_offset = out.phones.len();
        }
        words_out.push(rom);
    }
    out.text = words_out.join(" ");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kokoro_normalization() {
        let k = to_kokoro("/tˈɑdɑt/");
        assert_eq!(k.text, "tˈɑdɑt");
        assert!(k.unknown.is_empty());
        assert_eq!(to_kokoro("seɪ").text, "sA");
        assert_eq!(to_kokoro("ɪts ə").text, "ɪts ə");
    }

    #[test]
    fn romanization_with_wildcards() {
        let r = romanize("* ɪts ə tˈɑdɑt əɡˈɛn *");
        assert_eq!(r.text, "* its e tadat egen *");
        assert_eq!(r.tokens.first(), Some(&MMS_STAR));
        assert_eq!(r.tokens.last(), Some(&MMS_STAR));
        let target: Vec<usize> = r.token_word.iter().enumerate().filter(|(_, &w)| w == 3).map(|(i, _)| i).collect();
        assert_eq!(target.len(), 5);
    }

    #[test]
    fn spelling_passthrough() {
        let r = romanize("It's a tadat again");
        assert_eq!(r.text, "it's a tadat again");
    }

    #[test]
    fn phones_split() {
        let p = phones("tʰˈɑːdɑt");
        let s: Vec<&str> = p.iter().map(|p| p.symbol.as_str()).collect();
        assert_eq!(s, vec!["tʰ", "ɑː", "d", "ɑ", "t"]);
        assert!(p[1].stressed);
    }
}
