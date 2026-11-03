use std::sync::Arc;

use av_toolbox_core::PipelineRegistry;
use av_toolbox_gpu::FrameFormat;
use av_toolbox_graph::{Diagnostic, StageSpec};
use av_toolbox_pipeline::Pipeline;

use crate::input::{self, PipelineInput};
use crate::report::PipelineFiles;
use crate::{SessionError, session, steps};

/// A pipeline read from its TOML, which can be changed and checked before any device opens.
///
/// [Session::new](crate::Session::new) opens it on a backend.
pub struct PipelineSource {
    pub(crate) specs: Vec<StageSpec>,
    pub(crate) files: Arc<PipelineFiles>,
    pub(crate) registry: PipelineRegistry,
}

impl PipelineSource {
    /// Reads the pipeline and every file it calls.
    ///
    /// `vars` fills the pipeline's variables over any `AVTB_VAR_<NAME>` environment variables.
    pub fn read(input: PipelineInput, vars: &[(String, String)]) -> Result<Self, SessionError> {
        let supplied = steps::supplied(vars);
        let loaded = input::load(&input, &supplied)?;
        let files = Arc::new(loaded.files);
        let registry = steps::registry()?;

        Ok(Self {
            specs: loaded.specs,
            files,
            registry,
        })
    }

    pub fn specs(&self) -> &[StageSpec] {
        &self.specs
    }

    pub fn specs_mut(&mut self) -> &mut Vec<StageSpec> {
        &mut self.specs
    }

    /// Builds the pipeline without a device, reporting every problem building it finds.
    ///
    /// `channels` gives the format of each `channel/frames` source, as [Session::new](crate::Session::new)
    /// takes them.
    pub fn check(&self, channels: &[(String, FrameFormat)]) -> Result<(), SessionError> {
        session::check_source(self, channels)
    }

    /// Every built-in stage the pipeline is built from.
    #[doc(hidden)]
    pub fn registry(&self) -> &PipelineRegistry {
        &self.registry
    }

    /// Builds the stages as written, leaving every `channel/frames` stage unbound.
    #[doc(hidden)]
    pub fn build(&self) -> Result<Pipeline<'_>, SessionError> {
        let built = steps::build(&self.registry, &self.specs);
        match built {
            Ok(pipeline) => Ok(pipeline),
            Err(found) => Err(self.pipeline_error(found)),
        }
    }

    pub(crate) fn pipeline_error(&self, found: Vec<Diagnostic>) -> SessionError {
        let files = self.files.clone();
        SessionError::pipeline(files, found)
    }
}

#[cfg(test)]
mod tests {
    use av_toolbox_gpu::{ColourInfo, FrameLayout, FrameRate, SampleFormat, Subsampling};

    use super::*;

    const PASSTHROUGH: &str = r#"
[source.input]
type = "channel/frames"

[filter.crop]
type = "crop/manual"
inputs = ["input"]

[sink.output]
type = "channel/frames"
inputs = ["crop"]
"#;

    fn text_input(text: &str) -> PipelineInput {
        PipelineInput::Text {
            name: "pipeline.toml".to_owned(),
            text: text.to_owned(),
        }
    }

    fn format() -> FrameFormat {
        FrameFormat {
            width: 64,
            height: 64,
            layout: FrameLayout::Yuv {
                subsampling: Subsampling::Yuv420,
            },
            sample: SampleFormat::U8,
            colour: ColourInfo::default(),
            frame_rate: FrameRate::new(24, 1),
            field_order: None,
        }
    }

    #[test]
    fn a_broken_pipeline_is_reported_before_any_device_opens() {
        let text = "[source.input]\ntype = \"channel/frames\"\n\n[sink.output]\ntype = \"y4m/pipe\"\ninputs = [\"inptu\"]\n";
        let input = text_input(text);
        let source = PipelineSource::read(input, &[]).expect("the text reads");
        let channels = [("input".to_owned(), format())];

        let error = source.check(&channels).expect_err("the typo is refused");

        assert!(error.points_at_files());
        let message = error.to_string();
        assert!(message.contains("inptu"), "{message}");
    }

    #[test]
    fn a_channel_pipeline_checks_with_its_formats() {
        let input = text_input(PASSTHROUGH);
        let source = PipelineSource::read(input, &[]).expect("the text reads");
        let channels = [("input".to_owned(), format())];

        source.check(&channels).expect("the pipeline builds");
    }

    #[test]
    fn a_channel_source_without_a_format_fails_the_check() {
        let input = text_input(PASSTHROUGH);
        let source = PipelineSource::read(input, &[]).expect("the text reads");

        let error = source.check(&[]).expect_err("the missing format is refused");

        let message = error.to_string();
        assert!(message.contains("only the library can run it"), "{message}");
        assert!(message.contains("no frame format was given"), "{message}");
    }
}
