use av_toolbox_core::{Filter, FilterDefinition, Inputs};
use av_toolbox_gpu::{
    Capabilities,
    Channel,
    ChromaLocation,
    ColourInfo,
    Frame,
    FrameFormat,
    FrameLayout,
    FrameMeta,
    MaskFormat,
    Offered,
    SampleFormat,
    StreamFormat,
    StreamItem,
    Subsampling,
    default_accelerators,
    testing_context,
};

use crate::planes::extract::{ExtractArgs, Extractor, extracted};
use crate::planes::{Extract, PlaneName};

/// The one frame a filter emitted for one input.
fn only_frame(mut frames: Vec<Frame>) -> Frame {
    assert_eq!(
        frames.len(),
        1,
        "a filter emitted {} frames for one input",
        frames.len()
    );
    frames.remove(0)
}

fn capabilities() -> Capabilities {
    let accelerator = default_accelerators()[0];

    Capabilities::builder(accelerator)
        .plane_size(32, 32)
        .max_units_per_cube(1024)
        .max_cube_count(65535, 65535, 65535)
        .build()
}

fn yuv420(width: u32, height: u32) -> FrameFormat {
    let colour = ColourInfo {
        chroma_location: ChromaLocation::Center,
        ..ColourInfo::default()
    };

    FrameFormat {
        width,
        height,
        layout: FrameLayout::Yuv {
            subsampling: Subsampling::Yuv420,
        },
        sample: SampleFormat::U8,
        colour,
        frame_rate: None,
    }
}

#[test]
fn luma_comes_out_as_grey_at_full_size() {
    let input = yuv420(64, 32);

    let output = extracted(&input, PlaneName::Y).expect("carries Y");

    assert_eq!(output.layout, FrameLayout::Grey);
    assert_eq!((output.width, output.height), (64, 32));
}

#[test]
fn one_chroma_plane_comes_out_as_grey_at_its_own_size() {
    let input = yuv420(63, 31);

    let output = extracted(&input, PlaneName::V).expect("carries V");

    assert_eq!(output.layout, FrameLayout::Grey);
    assert_eq!((output.width, output.height), (32, 16));
}

#[test]
fn the_chroma_pair_keeps_its_subsampling_and_siting() {
    let input = yuv420(64, 32);

    let output = extracted(&input, PlaneName::Chroma).expect("yuv has chroma");

    assert_eq!(
        output.layout,
        FrameLayout::Chroma {
            subsampling: Subsampling::Yuv420
        }
    );
    assert_eq!((output.width, output.height), (32, 16));
    assert_eq!(output.colour.chroma_location, ChromaLocation::Center);
}

#[test]
fn a_channel_the_clip_lacks_is_refused() {
    let input = yuv420(64, 32);

    let error = extracted(&input, PlaneName::R).expect_err("yuv has no R");

    assert!(error.to_string().contains('R'), "{error}");
}

#[test]
fn chroma_from_anything_but_yuv_is_refused() {
    let input = FrameFormat {
        layout: FrameLayout::Rgb,
        ..yuv420(64, 32)
    };

    let refused = extracted(&input, PlaneName::Chroma);

    assert!(refused.is_err());
}

#[test]
fn one_plane_can_be_taken_from_a_chroma_clip() {
    let input = FrameFormat {
        layout: FrameLayout::Chroma {
            subsampling: Subsampling::Yuv420,
        },
        ..yuv420(32, 16)
    };

    let output = extracted(&input, PlaneName::U).expect("a chroma clip carries U");

    assert_eq!(output.layout, FrameLayout::Grey);
}

#[test]
fn extracting_shares_the_plane() {
    let mut context = testing_context();
    let input = yuv420(64, 32);
    let output = extracted(&input, PlaneName::Chroma).expect("yuv has chroma");
    let frame = context.alloc_frame(&input, FrameMeta::at(0));
    let original_u = frame.plane(Channel::U).expect("carries U").clone();
    let mut extractor = Extractor::new(output, PlaneName::Chroma);

    let slots = vec![Some(StreamItem::Frame(frame))];
    let inputs = Inputs::new(slots);
    let emitted = extractor.process(&mut context, inputs).expect("extracts");
    let produced = only_frame(emitted);

    let produced_u = produced.plane(Channel::U).expect("carries U");
    assert!(std::ptr::eq(produced_u.buffer(), original_u.buffer()));
}

#[test]
fn a_mask_is_refused() {
    let capabilities = capabilities();
    let arguments = ExtractArgs { plane: PlaneName::Y };
    let inputs = [Offered::Unknown];
    let mask = StreamFormat::Mask(MaskFormat {
        width: 16,
        height: 8,
        sample: SampleFormat::U8,
        chroma: None,
    });

    let refused = Extract::accepts_input(&capabilities, &arguments, 0, mask, &inputs);

    assert!(refused.is_err());
}
