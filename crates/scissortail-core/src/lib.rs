//! Scissortail core: EMA trial I/O, word localization (phonetic alignment or
//! TTS-template matching) and articulatory kinematic landmark extraction.

pub mod aligner;
pub mod analysis;
pub mod dsp;
pub mod error;
pub mod export;
pub mod ipa;
pub mod kinematics;
pub mod locate;
pub mod mat;
mod onnx;
pub mod session;
pub mod template;
pub mod trial;
pub mod tts;

pub use error::{Error, Result};
