use std::fmt;

#[derive(Debug)]
pub enum Error {
    Io(String, std::io::Error),
    Format(String),
    Model(String),
    Invalid(String),
}

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub fn with_context(self, ctx: &str) -> Error {
        match self {
            Error::Format(m) => Error::Format(format!("{ctx}: {m}")),
            Error::Model(m) => Error::Model(format!("{ctx}: {m}")),
            Error::Invalid(m) => Error::Invalid(format!("{ctx}: {m}")),
            other => other,
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(path, e) => write!(f, "cannot read {path}: {e}"),
            Error::Format(m) => write!(f, "file format error: {m}"),
            Error::Model(m) => write!(f, "model error: {m}"),
            Error::Invalid(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for Error {}

#[cfg(feature = "onnxruntime")]
impl From<ort::Error> for Error {
    fn from(e: ort::Error) -> Self {
        Error::Model(e.to_string())
    }
}
