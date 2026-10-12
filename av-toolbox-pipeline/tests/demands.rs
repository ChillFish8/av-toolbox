#![cfg(all(feature = "gpu-tests", any(feature = "vulkan", feature = "metal")))]

use std::fs;

use av_toolbox_core::{
    AnyDemand,
    BoundInputs,
    Filter,
    FilterDefinition,
    InputDemand,
    InputParam,
    InputSignature,
    Inputs,
    Source,
    SourceDefinition,
    StageDefinition,
    StageError,
    StageInstance,
    StreamType,
};
use av_toolbox_gpu::{
    Backend,
    BuildContext,
    Capabilities,
    ColourInfo,
    DeviceSide,
    FormatSet,
    Frame,
    FrameContext,
    FrameFormat,
    FrameLayout,
    Offered,
    OutputFormats,
    ResolvedFormats,
    Runtime,
    SampleFormat,
    StreamFormat,
    StreamTemplate,
    Subsampling,
    test_device,
};
use av_toolbox_graph::DiagnosticCode;
use av_toolbox_pipeline::{DiskFiles, Pipeline, Supplied};
use av_toolbox_providers::{SceneBreaks, SceneRequest, Scenes};
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Debug, Deserialize, JsonSchema)]
struct NoArgs {}

struct RatelessInstance;

impl StageInstance for RatelessInstance {}

impl Source for RatelessInstance {
    fn next(&mut self, _context: &mut FrameContext) -> Result<Option<Frame>, StageError> {
        Ok(None)
    }
}

/// A device source whose stream has no frame rate.
struct Rateless;

impl StageDefinition for Rateless {
    type Args = NoArgs;

    fn package() -> &'static str {
        "pipeline-test"
    }

    fn identifier() -> &'static str {
        "rateless"
    }

    fn description() -> &'static str {
        "A stage used by tests."
    }

    fn inputs() -> &'static InputSignature {
        &InputSignature::EMPTY
    }
}

impl SourceDefinition for Rateless {
    type Side = DeviceSide;
    type Opened = ();

    fn open(_args: &Self::Args) -> Result<Self::Opened, StageError> {
        Ok(())
    }

    fn produces(_capabilities: &Capabilities, _opened: &Self::Opened) -> Result<OutputFormats, StageError> {
        let layout = FrameLayout::Yuv {
            subsampling: Subsampling::Yuv420,
        };
        let format = FrameFormat {
            width: 64,
            height: 64,
            layout,
            sample: SampleFormat::U8,
            colour: ColourInfo::default(),
            frame_rate: None,
            field_order: None,
        };

        Ok(OutputFormats {
            template: StreamTemplate::Frames(format),
            samples: FormatSet::of([format.sample]),
        })
    }

    fn build<R: Runtime>(
        _context: &BuildContext<R>,
        _opened: Self::Opened,
        _inputs: BoundInputs,
        _formats: &ResolvedFormats,
    ) -> Result<Box<dyn Source>, StageError> {
        Ok(Box::new(RatelessInstance))
    }
}

struct PassInstance;

impl StageInstance for PassInstance {}

impl Filter for PassInstance {
    fn process(&mut self, _context: &mut FrameContext, mut inputs: Inputs) -> Result<Vec<Frame>, StageError> {
        let frame = inputs
            .take_frame(0)
            .ok_or_else(|| StageError::msg("expected a frame in slot 0"))?;

        Ok(vec![frame])
    }
}

/// A device filter demanding scenes of at most 10 seconds on its input.
struct TenSecondScenes;

impl StageDefinition for TenSecondScenes {
    type Args = NoArgs;

    fn package() -> &'static str {
        "pipeline-test"
    }

    fn identifier() -> &'static str {
        "ten-second-scenes"
    }

    fn description() -> &'static str {
        "A stage used by tests."
    }

    fn demands(_args: &Self::Args) -> Vec<InputDemand> {
        let breaks = SceneBreaks::Limited {
            min_frames: 0,
            max_frames: None,
            max_seconds: Some(10.0),
        };
        let request = SceneRequest {
            breaks,
            ..SceneRequest::natural()
        };
        let demand = InputDemand {
            input: "input",
            demand: AnyDemand::new::<Scenes>(request),
        };

        vec![demand]
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature =
            &InputSignature::new(&[InputParam::required("input").accepting(StreamType::Frames)]);
        SIGNATURE
    }
}

impl FilterDefinition for TenSecondScenes {
    type Side = DeviceSide;

    fn accepts_input(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        _index: usize,
        offered: StreamFormat,
        _inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        Ok(FormatSet::of([offered.sample()]))
    }

    fn produces(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError> {
        let format = chosen.iter().flatten().next().copied();
        let format = format.ok_or_else(|| StageError::msg("no input was bound"))?;

        Ok(OutputFormats {
            template: StreamTemplate::from_format(&format),
            samples: FormatSet::of([format.sample()]),
        })
    }

    fn build<R: Runtime>(
        _context: &BuildContext<R>,
        _args: Self::Args,
        _inputs: BoundInputs,
        _formats: &ResolvedFormats,
    ) -> Result<Box<dyn Filter>, StageError> {
        Ok(Box::new(PassInstance))
    }
}

const PIPELINE: &str = r#"
[source.input]
type = "pipeline-test/rateless"

[filter.denoise]
type = "pipeline-test/ten-second-scenes"
inputs = ["input"]

[sink.output]
type = "y4m/stdout"
inputs = ["denoise"]
"#;

#[test]
fn a_seconds_limit_on_a_stream_with_no_frame_rate_names_the_stage_demanding_it() {
    let mut registry = av_toolbox_pipeline::registry().expect("the built-in stages register");
    registry.register_source::<Rateless>().expect("should register");
    registry
        .register_filter::<TenSecondScenes>()
        .expect("should register");

    let folder = tempfile::tempdir().expect("a temp dir");
    let path = folder.path().join("pipeline.toml");
    fs::write(&path, PIPELINE).expect("writes the pipeline");
    let supplied = Supplied::new();
    let loaded = av_toolbox_pipeline::load(&path, &DiskFiles, &supplied).expect("the file should load");

    let pipeline = Pipeline::build(&registry, &loaded.specs).expect("should build");
    let backend = Backend::open_default(test_device()).expect("should open a device");

    let Err(error) = pipeline.compile(&backend) else {
        panic!("a seconds limit with no frame rate should be refused");
    };

    let diagnostics: Vec<_> = error.diagnostics().iter().collect();
    let refusal = diagnostics
        .iter()
        .find(|diagnostic| diagnostic.path() == "provider.scenes(input)")
        .unwrap_or_else(|| panic!("no diagnostic names the provider: {diagnostics:?}"));
    let notes: Vec<&str> = refusal.notes().iter().map(|note| note.message.as_str()).collect();

    assert_eq!(refusal.code(), DiagnosticCode::StageBuildFailed);
    assert!(notes.iter().any(|note| note.contains("`denoise`")), "{refusal}");
}
