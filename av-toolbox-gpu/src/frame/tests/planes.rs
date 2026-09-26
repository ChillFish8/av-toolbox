use crate::frame::format::{ColourInfo, FrameFormat};
use crate::frame::gpu::{Frame, FrameError, FrameMeta};
use crate::frame::layout::{Channel, FrameLayout, Subsampling};
use crate::frame::sample::SampleFormat;
use crate::testing_context;

fn yuv420(width: u32, height: u32) -> FrameFormat {
    FrameFormat {
        width,
        height,
        layout: FrameLayout::Yuv {
            subsampling: Subsampling::Yuv420,
        },
        sample: SampleFormat::U8,
        colour: ColourInfo::default(),
        frame_rate: None,
    }
}

fn chroma_of(format: &FrameFormat) -> FrameFormat {
    let (width, height) = format
        .channel_dimensions(Channel::U)
        .expect("a yuv frame carries U");

    FrameFormat {
        width,
        height,
        layout: FrameLayout::Chroma {
            subsampling: Subsampling::Yuv420,
        },
        ..*format
    }
}

#[test]
fn a_frame_built_from_another_frames_planes_shares_them() {
    let context = testing_context();
    let format = yuv420(64, 32);
    let source = context.alloc_frame(&format, FrameMeta::at(0));
    let u_plane = source.plane(Channel::U).expect("carries U").clone();
    let v_plane = source.plane(Channel::V).expect("carries V").clone();

    let chroma_format = chroma_of(&format);
    let planes = vec![u_plane, v_plane];
    let chroma = Frame::from_planes(chroma_format, planes, FrameMeta::at(0)).expect("the planes match");

    let built = chroma.plane(Channel::U).expect("carries U");
    let original = source.plane(Channel::U).expect("carries U");
    assert!(std::ptr::eq(built.buffer(), original.buffer()));
}

#[test]
fn a_wrong_plane_count_is_refused() {
    let context = testing_context();
    let format = yuv420(64, 32);
    let source = context.alloc_frame(&format, FrameMeta::at(0));
    let luma = source.plane(Channel::Y).expect("carries Y").clone();

    let refused = Frame::from_planes(format, vec![luma], FrameMeta::at(0));

    assert!(matches!(refused, Err(FrameError::PlaneCount { .. })));
}

#[test]
fn a_plane_of_the_wrong_size_is_refused() {
    let context = testing_context();
    let format = yuv420(64, 32);
    let source = context.alloc_frame(&format, FrameMeta::at(0));
    let luma = source.plane(Channel::Y).expect("carries Y").clone();
    let chroma_format = chroma_of(&format);

    let planes = vec![luma.clone(), luma];
    let refused = Frame::from_planes(chroma_format, planes, FrameMeta::at(0));

    let Err(FrameError::PlaneSize { channel, .. }) = refused else {
        panic!("expected a size refusal, got {refused:?}");
    };
    assert_eq!(channel, Channel::U);
}

#[test]
fn a_plane_of_the_wrong_sample_width_is_refused() {
    let context = testing_context();
    let format = yuv420(64, 32);
    let source = context.alloc_frame(&format, FrameMeta::at(0));
    let planes: Vec<_> = format
        .channels()
        .iter()
        .map(|channel| source.plane(*channel).expect("carries it").clone())
        .collect();
    let wide = FrameFormat {
        sample: SampleFormat::F32,
        ..format
    };

    let refused = Frame::from_planes(wide, planes, FrameMeta::at(0));

    assert!(matches!(refused, Err(FrameError::PlaneBytes { .. })));
}
