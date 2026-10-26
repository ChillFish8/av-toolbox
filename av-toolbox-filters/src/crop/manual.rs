use av_toolbox_core::{
    BoundInputs,
    Filter,
    FilterDefinition,
    InputParam,
    InputSignature,
    Parallelism,
    StageDefinition,
    StageError,
    StreamType,
};
use av_toolbox_gpu::{
    BuildContext,
    Capabilities,
    DeviceSide,
    FormatSet,
    Offered,
    OutputFormats,
    ResolvedFormats,
    Runtime,
    StreamFormat,
    StreamTemplate,
};

use super::{Cropper, Edges, accepts_frames_of_any_sample, checked_input, cropped};

const STAGE: &str = "crop/manual";

/// Removes a given number of samples from each edge of a clip.
pub(crate) struct Manual;

impl StageDefinition for Manual {
    type Args = Edges;

    fn package() -> &'static str {
        "crop"
    }

    fn identifier() -> &'static str {
        "manual"
    }

    fn description() -> &'static str {
        "Crops a fixed number of pixels from each edge of the frame."
    }

    fn parallelism(_args: &Self::Args) -> Parallelism {
        Parallelism::Frame
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature =
            &InputSignature::new(&[InputParam::required("input").accepting(StreamType::Frames)]);
        SIGNATURE
    }
}

impl FilterDefinition for Manual {
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
        let output = cropped(&input, args, STAGE)?;
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
        let client = context.client().clone();
        let cropper = Cropper::new(client, STAGE, input, args)?;

        Ok(Box::new(cropper))
    }
}

#[cfg(test)]
mod tests {
    use av_toolbox_gpu::{
        ColourInfo,
        Depth,
        FrameFormat,
        FrameLayout,
        MaskFormat,
        SampleFormat,
        Subsampling,
        default_accelerators,
    };
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

    fn frames(width: u32, height: u32, layout: FrameLayout, sample: SampleFormat) -> StreamFormat {
        StreamFormat::Frames(FrameFormat {
            width,
            height,
            layout,
            sample,
            colour: ColourInfo::default(),
            frame_rate: None,
            field_order: None,
        })
    }

    fn yuv(subsampling: Subsampling, width: u32, height: u32) -> StreamFormat {
        let layout = FrameLayout::Yuv { subsampling };

        frames(width, height, layout, SampleFormat::U8)
    }

    fn produced(format: StreamFormat, value: serde_json::Value) -> Result<StreamFormat, StageError> {
        let capabilities = capabilities();
        let chosen = [Some(format)];
        let edges: Edges = serde_json::from_value(value).expect("parses");
        let outputs = Manual::produces(&capabilities, &edges, &chosen)?;

        Ok(outputs.template.with_sample(format.sample()))
    }

    #[test]
    fn edges_shrink_the_frame_and_keep_everything_else() {
        let format = yuv(Subsampling::Yuv420, 16, 8);
        let value = json!({ "left": 2, "right": 4, "top": 2, "bottom": 2 });

        let output = produced(format, value).expect("croppable");

        let cropped = output.as_frames().expect("frames");
        let input = format.as_frames().expect("frames");
        assert_eq!((cropped.width, cropped.height), (10, 4));
        assert_eq!(cropped.layout, input.layout);
        assert_eq!(cropped.colour, input.colour);
    }

    #[test]
    fn an_odd_edge_at_420_is_refused_naming_it() {
        let format = yuv(Subsampling::Yuv420, 16, 8);
        let value = json!({ "left": 3 });

        let error = produced(format, value).expect_err("odd at 4:2:0");

        assert!(error.to_string().contains("`left`"), "{error}");
        assert!(error.to_string().contains("multiple of 2"), "{error}");
    }

    #[test]
    fn an_odd_top_at_422_is_accepted_and_an_odd_right_is_not() {
        let format = yuv(Subsampling::Yuv422, 16, 8);

        let top = produced(format, json!({ "top": 1 }));
        let right = produced(format, json!({ "right": 1 }));

        assert!(top.is_ok());
        assert!(right.is_err());
    }

    #[test]
    fn any_edge_is_accepted_on_grey_rgb_and_chroma() {
        let chroma = FrameLayout::Chroma {
            subsampling: Subsampling::Yuv420,
        };
        let layouts = [FrameLayout::Grey, FrameLayout::Rgb, chroma];

        for layout in layouts {
            let format = frames(16, 8, layout, SampleFormat::U8);
            let value = json!({ "left": 1, "top": 3 });

            let output = produced(format, value);

            assert!(output.is_ok(), "{layout:?}");
        }
    }

    #[test]
    fn a_crop_removing_every_column_is_refused() {
        let format = frames(16, 8, FrameLayout::Grey, SampleFormat::U8);
        let value = json!({ "left": 8, "right": 8 });

        let refused = produced(format, value);

        assert!(refused.is_err());
    }

    #[test]
    fn every_sample_is_accepted() {
        let capabilities = capabilities();
        let ten_bit = SampleFormat::U16 { depth: Depth::TEN };
        let samples = [SampleFormat::U8, ten_bit, SampleFormat::F16, SampleFormat::F32];

        for sample in samples {
            let offered = frames(16, 8, FrameLayout::Grey, sample);
            let inputs = [Offered::Unknown];
            let accepted = Manual::accepts_input(&capabilities, &Edges::default(), 0, offered, &inputs)
                .expect("frames are readable");

            assert!(accepted.accepts(sample), "{sample}");
        }
    }

    #[test]
    fn a_mask_is_refused() {
        let capabilities = capabilities();
        let mask = StreamFormat::Mask(MaskFormat {
            frame_rate: None,
            width: 8,
            height: 8,
            sample: SampleFormat::U8,
            chroma: None,
        });
        let inputs = [Offered::Unknown];

        let refused = Manual::accepts_input(&capabilities, &Edges::default(), 0, mask, &inputs);

        assert!(refused.is_err());
    }

    #[test]
    fn an_unknown_argument_is_refused() {
        let parsed = serde_json::from_value::<Edges>(json!({ "width": 4 }));

        assert!(parsed.is_err());
    }
}
