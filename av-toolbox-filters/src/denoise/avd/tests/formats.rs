use av_toolbox_core::FilterDefinition;
use av_toolbox_gpu::{
    Capabilities,
    ColourInfo,
    Depth,
    FrameFormat,
    FrameLayout,
    SampleFormat,
    StreamFormat,
    Subsampling,
    default_accelerators,
};
use serde_json::json;

use crate::denoise::avd::Denoise;
use crate::denoise::avd::args::{AvdArgs, Nl4dArgs, NlmeansArgs, NlmeansHqArgs};

fn format(layout: FrameLayout, sample: SampleFormat) -> FrameFormat {
    FrameFormat {
        width: 64,
        height: 32,
        layout,
        sample,
        colour: ColourInfo::default(),
        frame_rate: None,
        field_order: None,
    }
}

fn capabilities() -> Capabilities {
    let accelerator = default_accelerators()[0];

    Capabilities::builder(accelerator)
        .plane_size(32, 32)
        .max_units_per_cube(1024)
        .max_cube_count(65535, 65535, 65535)
        .build()
}

fn assert_f32_out<A: AvdArgs>() {
    let capabilities = capabilities();
    let args: A = serde_json::from_value(json!({})).expect("the args should parse");
    let yuv420 = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv420,
    };
    let ten_bit = SampleFormat::U16 { depth: Depth::TEN };
    let inputs = [
        format(yuv420, SampleFormat::U8),
        format(FrameLayout::Grey, ten_bit),
        format(yuv420, SampleFormat::F32),
    ];

    for input in inputs {
        let chosen = [Some(StreamFormat::Frames(input))];
        let outputs = Denoise::<A>::produces(&capabilities, &args, &chosen).expect("denoisable");
        let sample = outputs.samples.preferences()[0];
        let output = outputs.template.with_sample(sample);

        let float = FrameFormat {
            sample: SampleFormat::F32,
            ..input
        };

        assert_eq!(
            output,
            StreamFormat::Frames(float),
            "{} on {}",
            A::STAGE,
            input.sample
        );
        assert_eq!(outputs.samples.preferences().len(), 1, "{}", A::STAGE);
    }
}

#[test]
fn the_output_is_f32_for_every_input() {
    assert_f32_out::<NlmeansArgs>();
    assert_f32_out::<NlmeansHqArgs>();
    assert_f32_out::<Nl4dArgs>();
}
