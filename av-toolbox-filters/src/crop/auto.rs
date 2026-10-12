use av_toolbox_core::{
    AnyProbe,
    BoundInputs,
    Filter,
    FilterDefinition,
    InputParam,
    InputProbe,
    InputSignature,
    Parallelism,
    Probed,
    StageDefinition,
    StageError,
    StreamType,
};
use av_toolbox_gpu::{
    BuildContext,
    Capabilities,
    DeviceSide,
    FormatSet,
    FrameFormat,
    FrameLayout,
    Offered,
    OutputFormats,
    ResolvedFormats,
    Runtime,
    StreamFormat,
    StreamTemplate,
};
use schemars::JsonSchema;
use serde::Deserialize;

use super::{
    Crop,
    CropBox,
    CropRequest,
    Cropper,
    Edges,
    accepts_frames_of_any_sample,
    checked_input,
    cropped,
};

const STAGE: &str = "crop/auto";

/// Arguments for `crop/auto`.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct AutoArgs {
    /// How many frames to measure, spread across the clip.
    #[serde(default = "thirteen")]
    #[schemars(range(min = 1, max = 1000))]
    pub(crate) samples: u32,
    #[serde(skip)]
    #[schemars(skip)]
    found: Option<CropBox>,
}

fn thirteen() -> u32 {
    13
}

/// Finds and removes the black bars around a clip's picture.
pub(crate) struct Auto;

fn not_probed() -> StageError {
    StageError::msg("crop/auto needs its source probed, which happens when the pipeline compiles")
}

fn other_size(measured: (u32, u32), input: &FrameFormat) -> StageError {
    let message = format!(
        "crop/auto measured {}x{} frames but receives {}x{}, so place it before the stage that changes the size",
        measured.0, measured.1, input.width, input.height
    );

    StageError::msg(message)
}

fn round_down(value: u32, factor: u32) -> u32 {
    value - value % factor
}

/// The edges to crop from `input`, rounded down to its subsampling.
fn edges_for(args: &AutoArgs, input: &FrameFormat) -> Result<Edges, StageError> {
    let Some(found) = &args.found else {
        return Err(not_probed());
    };

    if found.measured != (input.width, input.height) {
        return Err(other_size(found.measured, input));
    }

    let FrameLayout::Yuv { subsampling } = input.layout else {
        return Ok(found.edges);
    };

    let (factor_x, factor_y) = subsampling.factors();
    let edges = found.edges;

    Ok(Edges {
        left: round_down(edges.left, factor_x),
        right: round_down(edges.right, factor_x),
        top: round_down(edges.top, factor_y),
        bottom: round_down(edges.bottom, factor_y),
    })
}

impl StageDefinition for Auto {
    type Args = AutoArgs;

    fn package() -> &'static str {
        "crop"
    }

    fn identifier() -> &'static str {
        "auto"
    }

    fn description() -> &'static str {
        "Finds and removes the black bars around the picture.\n\n\
         The bars are measured once, on frames spread across the source, and the same crop applies to the \
         whole clip. Place it before any stage that changes the frame size."
    }

    fn parallelism(_args: &Self::Args) -> Parallelism {
        Parallelism::Frame
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature =
            &InputSignature::new(&[InputParam::required("input").accepting(StreamType::Frames)]);
        SIGNATURE
    }

    fn probes(args: &Self::Args) -> Vec<InputProbe> {
        let request = CropRequest {
            samples: args.samples,
        };
        let probe = AnyProbe::new::<Crop>(request);

        vec![InputProbe {
            input: "input",
            probe,
        }]
    }

    fn settle(args: &mut Self::Args, probed: &Probed) -> Result<(), StageError> {
        let Some(found) = probed.get::<Crop>("input") else {
            return Err(not_probed());
        };

        args.found = Some(found.clone());
        Ok(())
    }
}

impl FilterDefinition for Auto {
    type Side = DeviceSide;

    fn accepts_input(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        _index: usize,
        offered: StreamFormat,
        _inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        accepts_frames_of_any_sample(offered, STAGE)
    }

    fn produces(
        _capabilities: &Capabilities,
        args: &Self::Args,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError> {
        let input = checked_input(chosen, STAGE)?;
        let edges = edges_for(args, &input)?;
        let output = cropped(&input, &edges, STAGE)?;
        let format = StreamFormat::Frames(output);

        Ok(OutputFormats {
            template: StreamTemplate::from_format(&format),
            samples: FormatSet::of([output.sample]),
        })
    }

    fn build<R: Runtime>(
        context: &BuildContext<R>,
        args: Self::Args,
        _inputs: BoundInputs,
        formats: &ResolvedFormats,
    ) -> Result<Box<dyn Filter>, StageError> {
        let inputs = formats.inputs();
        let input = checked_input(inputs, STAGE)?;
        let edges = edges_for(&args, &input)?;
        let client = context.client().clone();
        let cropper = Cropper::new(client, STAGE, input, edges)?;

        Ok(Box::new(cropper))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use av_toolbox_core::ProbeKind;
    use av_toolbox_gpu::{ColourInfo, SampleFormat, Subsampling, default_accelerators};
    use serde_json::json;

    use super::*;

    fn capabilities() -> Capabilities {
        let accelerator = default_accelerators()[0];

        Capabilities::builder(accelerator)
            .plane_size(32, 32)
            .max_units_per_cube(1024)
            .max_cube_count(65535, 65535, 65535)
            .build()
    }

    fn yuv(subsampling: Subsampling, width: u32, height: u32) -> StreamFormat {
        StreamFormat::Frames(FrameFormat {
            width,
            height,
            layout: FrameLayout::Yuv { subsampling },
            sample: SampleFormat::U8,
            colour: ColourInfo::default(),
            frame_rate: None,
            field_order: None,
        })
    }

    fn crop_box(edges: Edges, measured: (u32, u32)) -> CropBox {
        CropBox { edges, measured }
    }

    fn parsed(value: serde_json::Value) -> AutoArgs {
        serde_json::from_value(value).expect("parses")
    }

    fn settled(value: serde_json::Value, found: CropBox) -> AutoArgs {
        let mut args = parsed(value);
        let mut probed = Probed::new();
        let kind = ProbeKind::of::<Crop>();
        let output = Arc::new(found);
        probed.insert("input", kind, output);

        Auto::settle(&mut args, &probed).expect("settles");

        args
    }

    fn produced(format: StreamFormat, args: &AutoArgs) -> Result<StreamFormat, StageError> {
        let capabilities = capabilities();
        let chosen = [Some(format)];
        let outputs = Auto::produces(&capabilities, args, &chosen)?;

        Ok(outputs.template.with_sample(format.sample()))
    }

    #[test]
    fn it_probes_its_input_with_its_samples() {
        let args = parsed(json!({ "samples": 5 }));

        let probes = Auto::probes(&args);

        assert_eq!(probes.len(), 1);
        assert_eq!(probes[0].input, "input");
        assert_eq!(
            probes[0].probe.request::<Crop>(),
            Some(CropRequest { samples: 5 })
        );
    }

    #[test]
    fn it_takes_13_samples_by_default() {
        let args = parsed(json!({}));

        assert_eq!(args.samples, 13);
    }

    #[test]
    fn an_odd_bar_at_420_rounds_down() {
        let found = crop_box(
            Edges {
                left: 0,
                right: 0,
                top: 141,
                bottom: 139,
            },
            (1920, 1080),
        );
        let args = settled(json!({}), found);
        let input = yuv(Subsampling::Yuv420, 1920, 1080);

        let output = produced(input, &args).expect("crops");

        let frames = output.as_frames().expect("frames");
        assert_eq!((frames.width, frames.height), (1920, 802));
    }

    #[test]
    fn an_odd_top_at_422_stays_odd() {
        let found = crop_box(
            Edges {
                left: 3,
                right: 0,
                top: 141,
                bottom: 0,
            },
            (1920, 1080),
        );
        let args = settled(json!({}), found);
        let input = yuv(Subsampling::Yuv422, 1920, 1080);

        let output = produced(input, &args).expect("crops");

        let frames = output.as_frames().expect("frames");
        assert_eq!((frames.width, frames.height), (1918, 939));
    }

    #[test]
    fn a_box_measured_at_another_size_is_refused() {
        let found = crop_box(Edges::default(), (1920, 1080));
        let args = settled(json!({}), found);
        let input = yuv(Subsampling::Yuv420, 3840, 2160);

        let error = produced(input, &args).expect_err("sizes differ");

        assert!(error.to_string().contains("1920x1080"), "{error}");
        assert!(error.to_string().contains("3840x2160"), "{error}");
    }

    #[test]
    fn an_unsettled_stage_refuses_to_produce() {
        let args = parsed(json!({}));
        let input = yuv(Subsampling::Yuv420, 1920, 1080);

        let refused = produced(input, &args);

        assert!(refused.is_err());
    }

    #[test]
    fn a_found_value_cannot_be_written_in_the_args() {
        let parsed = serde_json::from_value::<AutoArgs>(json!({ "found": {} }));

        assert!(parsed.is_err());
    }
}
