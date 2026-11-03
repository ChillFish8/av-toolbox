use std::time::Duration;

use av_toolbox_core::{PipelineRegistry, RegistryError};
use av_toolbox_gpu::{Backend, Retune, Tuning};
use av_toolbox_graph::{Diagnostic, StageSpec};
use av_toolbox_pipeline::{CompiledPipeline, Pipeline, PipelineError, Supplied};

use crate::SessionError;

/// How long a source waits for room on its frame budget while a stage tunes its kernels mid-run.
///
/// It matches how long av-denoise waits for another process to finish tuning.
const TUNING_STALL_DEADLINE: Duration = Duration::from_secs(15 * 60);

/// Every built-in stage.
pub fn registry() -> Result<PipelineRegistry, SessionError> {
    let registered = av_toolbox_pipeline::registry();
    registered.map_err(registry_error)
}

fn registry_error(error: RegistryError) -> SessionError {
    let message = format!("the built-in stages could not be registered: {error}");
    SessionError::message(message)
}

/// How long a source may wait for room on its frame budget, `None` for the budget's own default.
///
/// [Tuning::Auto] and a retune may tune kernels mid-run, and a source waits on the stage tuning them.
/// An exhaustive retune can take far longer than any deadline, so it waits for as long as it takes.
pub(crate) fn stall_deadline_for(tuning: Tuning, retune: Retune) -> Option<Duration> {
    match (retune, tuning) {
        (Retune::Exhaustive, _) => Some(Duration::MAX),
        (Retune::Families, _) | (Retune::No, Tuning::Auto) => Some(TUNING_STALL_DEADLINE),
        (Retune::No, Tuning::Cached | Tuning::Off) => None,
    }
}

/// The variables `vars` gives, over those `AVTB_VAR_<NAME>` environment variables give.
pub fn supplied(vars: &[(String, String)]) -> Supplied {
    let mut supplied = Supplied::from_env();
    for (name, value) in vars {
        supplied.set_flag(name.clone(), value.clone());
    }

    supplied
}

/// Builds the pipeline, or hands back its diagnostics.
pub fn build<'r>(
    registry: &'r PipelineRegistry,
    specs: &[StageSpec],
) -> Result<Pipeline<'r>, Vec<Diagnostic>> {
    let built = Pipeline::build(registry, specs);
    built.map_err(found_in)
}

/// Compiles the pipeline, or hands back why it failed together with the warnings building it raised.
pub fn compile<'r, 'b>(
    pipeline: Pipeline<'r>,
    backend: &'b Backend,
) -> Result<CompiledPipeline<'r, 'b>, Vec<Diagnostic>> {
    let warnings = pipeline.warnings().to_vec();
    let compiled = pipeline.compile(backend);

    match compiled {
        Ok(compiled) => Ok(compiled),
        Err(error) => {
            let mut found = found_in(error);
            found.extend(warnings);

            Err(found)
        },
    }
}

fn found_in(error: PipelineError) -> Vec<Diagnostic> {
    error.into_diagnostics().into_inner()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_auto_tuning_waits_longer_on_the_frame_budget() {
        assert_eq!(
            stall_deadline_for(Tuning::Auto, Retune::No),
            Some(TUNING_STALL_DEADLINE)
        );
        assert_eq!(stall_deadline_for(Tuning::Cached, Retune::No), None);
        assert_eq!(stall_deadline_for(Tuning::Off, Retune::No), None);
    }

    #[test]
    fn a_retune_waits_as_long_as_auto_tuning() {
        for tuning in [Tuning::Auto, Tuning::Cached] {
            let deadline = stall_deadline_for(tuning, Retune::Families);

            assert_eq!(deadline, Some(TUNING_STALL_DEADLINE), "{tuning}");
        }
    }

    #[test]
    fn an_exhaustive_retune_never_gives_up_waiting() {
        for tuning in [Tuning::Auto, Tuning::Cached] {
            let deadline = stall_deadline_for(tuning, Retune::Exhaustive);

            assert_eq!(deadline, Some(Duration::MAX), "{tuning}");
        }
    }
}
