use av_toolbox_gpu::{
    ColourInfo,
    Depth,
    FrameFormat,
    FrameLayout,
    MaskFormat,
    Matrix,
    SampleFormat,
    StreamFormat,
    Subsampling,
};

use crate::template::{Literal, field_of};

fn frames(sample: SampleFormat) -> FrameFormat {
    FrameFormat {
        width: 64,
        height: 48,
        layout: FrameLayout::Yuv {
            subsampling: Subsampling::Yuv420,
        },
        sample,
        colour: ColourInfo::default(),
        frame_rate: None,
        field_order: None,
    }
}

fn mask() -> StreamFormat {
    StreamFormat::Mask(MaskFormat {
        frame_rate: None,
        width: 64,
        height: 48,
        sample: SampleFormat::U8,
        chroma: None,
    })
}

#[test]
fn a_ten_bit_frame_has_bit_depth_ten() {
    let sample = SampleFormat::U16 { depth: Depth::TEN };
    let format = StreamFormat::Frames(frames(sample));

    let depth = field_of(&format, "bit_depth").expect("should read");

    assert_eq!(depth, Literal::Integer(10));
}

#[test]
fn a_float_frame_has_bit_depth_thirty_two_and_is_float() {
    let format = StreamFormat::Frames(frames(SampleFormat::F32));

    let depth = field_of(&format, "bit_depth").expect("should read");
    let is_float = field_of(&format, "is_float").expect("should read");

    assert_eq!(depth, Literal::Integer(32));
    assert_eq!(is_float, Literal::Bool(true));
}

#[test]
fn a_yuv420_frame_reports_its_layout() {
    let format = StreamFormat::Frames(frames(SampleFormat::U8));

    let layout = field_of(&format, "layout").expect("should read");

    assert_eq!(layout, Literal::String(String::from("yuv420")));
}

#[test]
fn a_mask_has_its_dimensions() {
    let width = field_of(&mask(), "width").expect("should read");
    let height = field_of(&mask(), "height").expect("should read");

    assert_eq!(width, Literal::Integer(64));
    assert_eq!(height, Literal::Integer(48));
}

#[test]
fn a_mask_has_no_layout() {
    let error = field_of(&mask(), "layout").expect_err("should fail");

    assert!(error.contains("is a mask"), "{error}");
}

#[test]
fn an_unknown_frame_rate_is_an_error() {
    let format = StreamFormat::Frames(frames(SampleFormat::U8));

    let error = field_of(&format, "frame_rate").expect_err("should fail");

    assert!(error.contains("frame rate"), "{error}");
}

#[test]
fn an_unknown_field_lists_the_fields_there_are() {
    let format = StreamFormat::Frames(frames(SampleFormat::U8));

    let error = field_of(&format, "widht").expect_err("should fail");

    assert!(error.contains("has no field widht"), "{error}");
    assert!(error.contains("width"), "{error}");
}

#[test]
fn matrix_names_are_snake_case() {
    let mut frame = frames(SampleFormat::U8);
    frame.colour.matrix = Matrix::Bt2020Ncl;
    let format = StreamFormat::Frames(frame);

    let matrix = field_of(&format, "matrix").expect("should read");

    assert_eq!(matrix, Literal::String(String::from("bt2020_ncl")));
}
