use std::fmt::Display;
use std::io;

use av_toolbox_core::StageError;
use av_toolbox_cubek::cnn::{BodyPath, CnnError, LayerRun, Precision};
use av_toolbox_gpu::{Capabilities, ElementType};
use schemars::JsonSchema;
use serde::Deserialize;

/// The precision of a network's feature maps, as a stage's arguments name it.
#[derive(Debug, Default, Deserialize, JsonSchema, Copy, Clone, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub(crate) enum PrecisionName {
    #[default]
    F16,
    F32,
}

impl PrecisionName {
    pub(crate) fn precision(self) -> Precision {
        match self {
            Self::F16 => Precision::F16,
            Self::F32 => Precision::F32,
        }
    }
}

fn no_f16_arithmetic(stage: &str) -> StageError {
    let message = format!(
        "{stage} uses precision f16 by default, which needs f16 arithmetic this device lacks. \
         Use precision = \"f32\"",
    );

    StageError::msg(message)
}

pub(crate) fn unreadable_model(stage: &str, file: &str, error: io::Error) -> StageError {
    let message = format!("{stage} could not read its model {file}: {error}");

    StageError::msg(message)
}

pub(crate) fn broken_model(stage: &str, file: &str, error: impl Display) -> StageError {
    let message = format!("{stage} could not load its model {file}: {error}");

    StageError::msg(message)
}

pub(crate) fn path_name(path: BodyPath) -> &'static str {
    match path {
        BodyPath::Convolution => "fused convolution",
        BodyPath::UnfusedConvolution => "unfused convolution",
        BodyPath::Im2col => "im2col",
    }
}

fn failed_warm_up(stage: &str, path: BodyPath, error: CnnError) -> StageError {
    let name = path_name(path);
    let message = format!("{stage} could not warm up on the {name} path: {error}");

    StageError::msg(message)
}

pub(crate) fn failed_run(stage: &str, path: BodyPath, error: CnnError) -> StageError {
    let name = path_name(path);
    let message = format!("{stage} could not run on the {name} path: {error}");

    StageError::msg(message)
}

/// The error for a warm-up that failed on every path it tried, naming each path and its cause in turn.
pub(crate) fn failed_every_path(stage: &str, failures: Vec<(BodyPath, CnnError)>) -> StageError {
    let mut causes: Vec<String> = failures
        .iter()
        .map(|(path, error)| {
            let name = path_name(*path);

            format!("on the {name} path ({error})")
        })
        .collect();
    let last = causes.pop().unwrap_or_default();
    let earlier = causes.join(", ");
    let message = match causes.is_empty() {
        true => format!("{stage} could not warm up {last}"),
        false => format!("{stage} could not warm up {earlier} or {last}"),
    };

    StageError::msg(message)
}

/// The paths a warm-up tries in turn from `chosen`, each falling back to the next.
pub(crate) fn fallbacks(chosen: BodyPath) -> &'static [BodyPath] {
    match chosen {
        BodyPath::Convolution => &[
            BodyPath::Convolution,
            BodyPath::UnfusedConvolution,
            BodyPath::Im2col,
        ],
        BodyPath::UnfusedConvolution => &[BodyPath::UnfusedConvolution, BodyPath::Im2col],
        BodyPath::Im2col => &[BodyPath::Im2col],
    }
}

/// The feature maps' precision, refusing f16 on a device without f16 arithmetic.
pub(crate) fn checked_precision(
    stage: &str,
    capabilities: &Capabilities,
    precision: PrecisionName,
) -> Result<Precision, StageError> {
    let computes_f16 = capabilities.can_compute(ElementType::F16);

    if precision == PrecisionName::F16 && !computes_f16 {
        return Err(no_f16_arithmetic(stage));
    }

    let chosen = precision.precision();
    Ok(chosen)
}

/// The body path for `precision`, taking the convolution path when the device has cooperative matrices
/// for it.
pub(crate) fn body_path(capabilities: &Capabilities, precision: Precision) -> BodyPath {
    let operands = match precision {
        Precision::F16 => ElementType::F16,
        Precision::F32 => ElementType::F32,
    };
    let offered = capabilities.has_cooperative_matrix(operands, ElementType::F32);

    if offered {
        BodyPath::Convolution
    } else {
        BodyPath::Im2col
    }
}

/// The device's largest single buffer, treating an unset limit of 0 as no cap.
pub(crate) fn allocation_cap(capabilities: &Capabilities) -> usize {
    let limit = capabilities.max_allocation;

    if limit == 0 {
        return usize::MAX;
    }

    usize::try_from(limit).unwrap_or(usize::MAX)
}

/// The kernels `layers` ran on, each named once with how many layers ran on it.
pub(crate) fn kernel_summary(layers: &[LayerRun]) -> String {
    let mut counts: Vec<(String, usize)> = Vec::new();

    for layer in layers {
        let name = match layer.candidate {
            Some(candidate) => candidate.name(),
            None => "cubek".to_string(),
        };

        match counts.iter_mut().find(|(known, _)| *known == name) {
            Some((_, count)) => *count += 1,
            None => counts.push((name, 1)),
        }
    }

    let parts: Vec<String> = counts
        .iter()
        .map(|(name, count)| format!("{name} x{count}"))
        .collect();

    parts.join(", ")
}

/// Warms up on each of [fallbacks](crate::cnn::fallbacks) from `chosen` in turn with `warm`, giving the
/// first path that succeeds and what its warm-up gave.
pub(crate) fn warmed_path<T>(
    stage: &'static str,
    chosen: BodyPath,
    mut warm: impl FnMut(BodyPath) -> Result<T, CnnError>,
) -> Result<(BodyPath, T), StageError> {
    let ladder = fallbacks(chosen);
    let mut failures = Vec::new();

    for (index, path) in ladder.iter().copied().enumerate() {
        let warmed = warm(path);

        let error = match warmed {
            Ok(outcome) => return Ok((path, outcome)),
            Err(error) => error,
        };

        if let Some(next) = ladder.get(index + 1) {
            let name = path_name(path);
            let next_name = path_name(*next);
            tracing::warn!(
                stage = stage,
                reason = %error,
                "the {name} path could not warm up, so the stage falls back to the {next_name} path",
            );
        }

        failures.push((path, error));
    }

    if failures.len() == 1 {
        let (path, error) = failures.remove(0);

        return Err(failed_warm_up(stage, path, error));
    }

    Err(failed_every_path(stage, failures))
}
