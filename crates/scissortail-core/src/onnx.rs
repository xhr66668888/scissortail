//! The ONNX inference engine behind the speech models, chosen at build time:
//!
//! - **ONNX Runtime** (C++, feature `onnxruntime`, the default) — the build for
//!   Apple-silicon Macs, on which the method was validated.
//! - **tract** (pure Rust, feature `tract`) — the Windows and Intel-Mac builds.
//!   ONNX Runtime does not load on Windows 7 (it calls Windows 8+ functions and
//!   needs the Windows 10 C++ runtime) and is no longer published for Intel
//!   Macs. tract runs every inference on a single thread, so its results do not
//!   depend on the number of cores either.
//!
//! The int8 model's posteriors differ slightly between the two engines,
//! somewhat more than between 2 and 8 ONNX Runtime threads: on the bundled
//! data 12.5 % against 7 % of trials move by more than 20 ms, mostly at the
//! offset (see README, Method).

use crate::error::{Error, Result};

#[cfg(not(any(feature = "onnxruntime", feature = "tract")))]
compile_error!("scissortail-core needs an inference engine: enable the `onnxruntime` or the `tract` feature");

pub(crate) use backend::Model;

/// A tensor passed to or returned by a model: shape and row-major data.
pub(crate) enum Value {
    F32(Vec<usize>, Vec<f32>),
    I64(Vec<usize>, Vec<i64>),
}

impl Value {
    pub fn into_f32(self) -> Result<(Vec<usize>, Vec<f32>)> {
        match self {
            Value::F32(shape, data) => Ok((shape, data)),
            Value::I64(..) => Err(Error::Model("model output is not a float tensor".into())),
        }
    }

    pub fn into_i64(self) -> Result<(Vec<usize>, Vec<i64>)> {
        match self {
            Value::I64(shape, data) => Ok((shape, data)),
            Value::F32(..) => Err(Error::Model("model output is not an integer tensor".into())),
        }
    }
}

#[cfg(all(feature = "onnxruntime", not(feature = "tract")))]
mod backend {
    use std::borrow::Cow;
    use std::path::Path;
    use std::sync::{Mutex, OnceLock};

    use ort::session::{builder::GraphOptimizationLevel, Session, SessionInputValue};
    use ort::value::Tensor;

    use super::Value;
    use crate::error::{Error, Result};

    /// Process-wide ONNX Runtime environment (telemetry disabled).
    fn environment() -> std::result::Result<&'static ort::environment::Environment, ort::Error> {
        static ENV: OnceLock<ort::environment::Environment> = OnceLock::new();
        if let Some(e) = ENV.get() {
            return Ok(e);
        }
        let env = ort::init().with_name("scissortail").with_telemetry(false).build()?;
        Ok(ENV.get_or_init(|| env))
    }

    /// An ONNX Runtime session; one run at a time.
    pub struct Model(Mutex<Session>);

    impl Model {
        pub fn load(path: &Path, threads: usize) -> Result<Model> {
            let session = Session::builder(environment()?)?
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .map_err(<ort::Error>::from)?
                .with_intra_threads(threads)
                .map_err(<ort::Error>::from)?
                .commit_from_file(path)?;
            Ok(Model(Mutex::new(session)))
        }

        /// Runs the model on named inputs and returns the named outputs, in order.
        pub fn run(&self, inputs: Vec<(&str, Value)>, outputs: &[&str]) -> Result<Vec<Value>> {
            let mut feed: Vec<(Cow<str>, SessionInputValue)> = Vec::with_capacity(inputs.len());
            for (name, value) in inputs {
                let tensor: SessionInputValue = match value {
                    Value::F32(shape, data) => Tensor::from_array((shape, data))?.into(),
                    Value::I64(shape, data) => Tensor::from_array((shape, data))?.into(),
                };
                feed.push((Cow::from(name), tensor));
            }
            let mut session = self.0.lock().map_err(|_| Error::Model("model session poisoned".into()))?;
            let out = session.run(feed)?;
            outputs
                .iter()
                .map(|&name| {
                    let v = &out[name];
                    if let Ok((shape, data)) = v.try_extract_tensor::<f32>() {
                        return Ok(Value::F32(shape.iter().map(|&d| d as usize).collect(), data.to_vec()));
                    }
                    let (shape, data) = v.try_extract_tensor::<i64>()?;
                    Ok(Value::I64(shape.iter().map(|&d| d as usize).collect(), data.to_vec()))
                })
                .collect()
        }
    }
}

#[cfg(feature = "tract")]
mod backend {
    use std::path::Path;
    use std::sync::Arc;

    use tract_onnx::prelude::*;

    use super::Value;
    use crate::error::{Error, Result};

    fn err(e: impl std::fmt::Display) -> Error {
        Error::Model(e.to_string())
    }

    /// A tract plan, optimized once for inputs of any length. Runs share no
    /// state, so several trials can be analysed at the same time.
    pub struct Model {
        plan: Arc<TypedRunnableModel>,
        inputs: Vec<String>,
        outputs: Vec<String>,
    }

    impl Model {
        /// `threads` is ignored: tract runs each inference on one thread.
        pub fn load(path: &Path, _threads: usize) -> Result<Model> {
            let plan = tract_onnx::onnx()
                .model_for_path(path)
                .and_then(|m| m.into_optimized())
                .and_then(|m| m.into_runnable())
                .map_err(err)?;
            let m = plan.model();
            let inputs = m.input_outlets().map_err(err)?.iter().map(|o| m.node(o.node).name.clone()).collect();
            let outputs = m.output_outlets().map_err(err)?.iter().map(|&o| m.outlet_label(o).unwrap_or(&m.node(o.node).name).to_string()).collect();
            Ok(Model { plan, inputs, outputs })
        }

        /// Runs the model on named inputs and returns the named outputs, in order.
        pub fn run(&self, mut inputs: Vec<(&str, Value)>, outputs: &[&str]) -> Result<Vec<Value>> {
            let mut feed: TVec<TValue> = tvec![];
            for name in &self.inputs {
                let i = inputs.iter().position(|(n, _)| n == name).ok_or_else(|| Error::Model(format!("no value for model input {name}")))?;
                let tensor: Tensor = match inputs.swap_remove(i).1 {
                    Value::F32(shape, data) => tract_ndarray::ArrayD::from_shape_vec(shape, data).map_err(err)?.into(),
                    Value::I64(shape, data) => tract_ndarray::ArrayD::from_shape_vec(shape, data).map_err(err)?.into(),
                };
                feed.push(tensor.into());
            }
            let out = self.plan.run(feed).map_err(err)?;
            outputs
                .iter()
                .map(|&name| {
                    let i = self.outputs.iter().position(|n| n == name).ok_or_else(|| Error::Model(format!("model has no output {name}")))?;
                    let t = &out[i];
                    let shape = t.shape().to_vec();
                    if t.datum_type() == f32::datum_type() {
                        return Ok(Value::F32(shape, t.to_plain_array_view::<f32>().map_err(err)?.iter().copied().collect()));
                    }
                    let t = t.cast_to::<i64>().map_err(err)?;
                    Ok(Value::I64(shape, t.to_plain_array_view::<i64>().map_err(err)?.iter().copied().collect()))
                })
                .collect()
        }
    }
}
