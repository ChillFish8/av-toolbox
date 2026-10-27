use av_toolbox_cubek::matrix::SampleRange;
use av_toolbox_gpu::{ColourInfo, ColourRange, Depth, FrameFormat, FrameLayout, SampleFormat, Subsampling};

use crate::bits::bits_of;
use crate::codes::{CodeValues, FLOAT_CODE_BITS, sample_range};

const TEN_BIT: SampleFormat = SampleFormat::U16 { depth: Depth::TEN };

fn format(sample: SampleFormat) -> FrameFormat {
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

#[test]
fn an_integer_clip_keeps_its_depth() {
    let eight_bit = CodeValues::new(&format(SampleFormat::U8), SampleFormat::U8);
    let ten_bit = CodeValues::new(&format(TEN_BIT), TEN_BIT);

    assert_eq!(eight_bit.bits(), 8);
    assert_eq!(eight_bit.peak(), 255.0);
    assert_eq!(ten_bit.bits(), 10);
    assert_eq!(ten_bit.peak(), 1023.0);
}

#[test]
fn a_float_clip_works_at_sixteen_bits() {
    let float = CodeValues::new(&format(SampleFormat::F32), SampleFormat::F32);

    assert_eq!(float.bits(), 16);
    assert_eq!(float.peak(), 65535.0);
}

#[test]
fn the_output_is_the_input_sample_or_f32() {
    let to_float = CodeValues::new(&format(SampleFormat::U8), SampleFormat::F32);
    let same = CodeValues::new(&format(SampleFormat::U8), SampleFormat::U8);

    assert_eq!(to_float.output_sample(), SampleFormat::F32);
    assert_eq!(same.output_sample(), SampleFormat::U8);
}

#[test]
#[should_panic(expected = "U8")]
fn an_integer_output_of_another_depth_panics() {
    CodeValues::new(&format(SampleFormat::U8), TEN_BIT);
}

#[test]
fn an_unlabelled_range_reads_as_limited() {
    assert_eq!(sample_range(ColourRange::Unspecified), SampleRange::Limited);
    assert_eq!(sample_range(ColourRange::Full), SampleRange::Full);
}

#[test]
fn bits_of_a_float_clip_is_the_code_depth() {
    assert_eq!(bits_of(SampleFormat::F32), FLOAT_CODE_BITS);
    assert_eq!(bits_of(SampleFormat::F32), 16);
}

#[test]
fn whole_codes_keeps_the_depth_peak_and_output() {
    let float = CodeValues::new(&format(SampleFormat::F32), SampleFormat::F32);
    let whole = float.whole_codes();

    assert_eq!(whole.bits(), float.bits());
    assert_eq!(whole.peak(), float.peak());
    assert_eq!(whole.output_sample(), float.output_sample());
}
