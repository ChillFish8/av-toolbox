mod geometry;
mod instance;
mod kernels;

use std::marker::PhantomData;

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
    FrameFormat,
    Offered,
    OutputFormats,
    ResolvedFormats,
    Runtime,
    SampleFormat,
    StreamFormat,
    StreamTemplate,
};

use self::geometry::{check_output, check_source, geometry, is_identity};
use self::instance::Resizer;
pub(crate) use self::kernels::{Bicubic, Bilinear, Lanczos, Point, Spline16, Spline36, Spline64};
use self::kernels::{ResizeKernel, stage_name};

/// Resizes every plane of a clip through the kernel `K` names.
pub(crate) struct Resize<K>(PhantomData<K>);

fn resizable(sample: SampleFormat) -> bool {
    matches!(sample, SampleFormat::U8 | SampleFormat::U16 { .. })
}

fn not_frames(stage: &str) -> StageError {
    let message = format!("{stage} resizes frames, not masks");

    StageError::msg(message)
}

fn missing_input(stage: &str) -> StageError {
    let message = format!("{stage} needs its input bound");

    StageError::msg(message)
}

/// The input format, once it is known to be frames.
fn checked_input(chosen: &[Option<StreamFormat>], stage: &str) -> Result<FrameFormat, StageError> {
    let Some(input) = chosen.first().copied().flatten() else {
        return Err(missing_input(stage));
    };

    let Some(format) = input.as_frames() else {
        return Err(not_frames(stage));
    };

    Ok(*format)
}

/// Checks the arguments against the input and gives the output format.
fn resized<K: ResizeKernel>(input: &FrameFormat, args: &K::Args) -> Result<FrameFormat, StageError> {
    let stage = stage_name::<K>();
    let window = K::window(args);
    window.check(&stage)?;
    K::kernel(args)?;
    check_output(input, &window, &stage)?;
    check_source(input, &window, &stage)?;

    Ok(FrameFormat {
        width: window.width,
        height: window.height,
        ..*input
    })
}

impl<K: ResizeKernel> StageDefinition for Resize<K> {
    type Args = K::Args;

    fn package() -> &'static str {
        "resize"
    }

    fn identifier() -> &'static str {
        K::identifier()
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

impl<K: ResizeKernel> FilterDefinition for Resize<K> {
    type Side = DeviceSide;

    fn accepts_input(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        _index: usize,
        offered: StreamFormat,
        _inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        let stage = stage_name::<K>();
        let Some(format) = offered.as_frames() else {
            return Err(not_frames(&stage));
        };

        if !resizable(format.sample) {
            return Ok(FormatSet::new());
        }

        Ok(FormatSet::of([format.sample]))
    }

    fn produces(
        _capabilities: &Capabilities,
        args: &Self::Args,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError> {
        let stage = stage_name::<K>();
        let input = checked_input(chosen, &stage)?;
        let output = resized::<K>(&input, args)?;
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
        let stage = stage_name::<K>();
        let inputs = formats.inputs();
        let input = checked_input(inputs, &stage)?;
        let output = resized::<K>(&input, &args)?;
        let window = K::window(&args);
        let kernel = K::kernel(&args)?;
        let shapes = geometry(&input, &window);
        let identity = is_identity(&input, &window);
        let client = context.client().clone();
        let resizer = Resizer::new(client, stage, output, &shapes, &kernel, identity);

        Ok(Box::new(resizer))
    }
}

#[cfg(test)]
mod tests {
    use av_toolbox_gpu::{ColourInfo, Depth, FrameLayout, MaskFormat, Subsampling, default_accelerators};
    use serde_json::json;

    use super::kernels::ResizeArgs;
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
        })
    }

    fn yuv(subsampling: Subsampling) -> StreamFormat {
        let layout = FrameLayout::Yuv { subsampling };

        frames(16, 8, layout, SampleFormat::U8)
    }

    fn args(value: serde_json::Value) -> ResizeArgs {
        serde_json::from_value(value).expect("parses")
    }

    fn produced(format: StreamFormat, value: serde_json::Value) -> Result<StreamFormat, StageError> {
        let capabilities = capabilities();
        let chosen = [Some(format)];
        let arguments = args(value);
        let outputs = Resize::<Point>::produces(&capabilities, &arguments, &chosen)?;

        Ok(outputs.template.with_sample(format.sample()))
    }

    #[test]
    fn every_layout_keeps_its_layout_at_the_new_size() {
        let layouts = [
            yuv(Subsampling::Yuv420),
            yuv(Subsampling::Yuv422),
            yuv(Subsampling::Yuv444),
            frames(16, 8, FrameLayout::Grey, SampleFormat::U8),
            frames(16, 8, FrameLayout::Rgb, SampleFormat::U8),
        ];

        for layout in layouts {
            let value = json!({ "width": 32, "height": 4 });
            let output = produced(layout, value).expect("resizable");
            let input = layout.as_frames().expect("frames");
            let resized = output.as_frames().expect("frames");

            assert_eq!((resized.width, resized.height), (32, 4));
            assert_eq!(resized.layout, input.layout);
            assert_eq!(resized.sample, input.sample);
            assert_eq!(resized.colour, input.colour);
        }
    }

    #[test]
    fn an_odd_width_at_420_is_refused_when_producing() {
        let value = json!({ "width": 15, "height": 8 });
        let format = yuv(Subsampling::Yuv420);

        let refused = produced(format, value);

        assert!(refused.is_err());
    }

    #[test]
    fn a_zero_width_is_refused_when_producing() {
        let value = json!({ "width": 0, "height": 8 });
        let format = yuv(Subsampling::Yuv444);

        let refused = produced(format, value);

        assert!(refused.is_err());
    }

    #[test]
    fn a_source_over_twice_the_input_is_refused_when_producing() {
        let value = json!({ "width": 16, "height": 8, "src_width": 40.0 });
        let format = yuv(Subsampling::Yuv444);

        let error = produced(format, value).expect_err("2.5 times the input is refused");

        assert!(error.to_string().contains("src_width"));
    }

    #[test]
    fn a_source_of_twice_the_input_is_accepted() {
        let value = json!({ "width": 16, "height": 8, "src_width": 32.0 });
        let format = yuv(Subsampling::Yuv444);

        let accepted = produced(format, value);

        assert!(accepted.is_ok());
    }

    #[test]
    fn u8_and_u16_are_accepted_and_floats_are_not() {
        let capabilities = capabilities();
        let arguments = args(json!({ "width": 8, "height": 8 }));
        let ten_bit = SampleFormat::U16 { depth: Depth::TEN };
        let cases = [
            (SampleFormat::U8, true),
            (ten_bit, true),
            (SampleFormat::F32, false),
        ];

        for (sample, expected) in cases {
            let offered = frames(16, 8, FrameLayout::Grey, sample);
            let inputs = [Offered::Unknown];
            let accepted = Resize::<Point>::accepts_input(&capabilities, &arguments, 0, offered, &inputs)
                .expect("frames are readable");

            assert_eq!(accepted.accepts(sample), expected, "{sample}");
        }
    }

    #[test]
    fn a_mask_is_refused() {
        let capabilities = capabilities();
        let arguments = args(json!({ "width": 8, "height": 8 }));
        let mask = StreamFormat::Mask(MaskFormat {
            width: 8,
            height: 8,
            sample: SampleFormat::U8,
            chroma: None,
        });
        let inputs = [Offered::Unknown];

        let refused = Resize::<Point>::accepts_input(&capabilities, &arguments, 0, mask, &inputs);

        assert!(refused.is_err());
    }

    #[test]
    fn every_resize_stage_runs_per_frame() {
        let arguments = args(json!({ "width": 8, "height": 8 }));

        assert_eq!(Resize::<Spline36>::parallelism(&arguments), Parallelism::Frame);
    }

    #[test]
    fn a_chroma_clip_resizes_as_two_full_planes() {
        let layout = FrameLayout::Chroma {
            subsampling: Subsampling::Yuv420,
        };
        let format = frames(8, 4, layout, SampleFormat::U8);
        let value = json!({ "width": 16, "height": 8 });

        let output = produced(format, value).expect("a chroma clip resizes");
        let resized = output.as_frames().expect("frames");

        assert_eq!((resized.width, resized.height), (16, 8));
        assert_eq!(resized.layout, layout);
    }
}
