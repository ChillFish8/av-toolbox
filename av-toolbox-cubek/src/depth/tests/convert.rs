use crate::depth::{Conversion, Destination, Source, convert_sample, convert_to_float_sample};
use crate::matrix::SampleRange;

fn through(value: u16, conversion: &Conversion) -> u16 {
    let value = f32::from(value);

    convert_sample(value, conversion, 0.5)
}

fn through_float(value: f32, conversion: &Conversion) -> u16 {
    convert_sample(value, conversion, 0.5)
}

fn round_trips(input_bits: u32, output_bits: u32, full_range: bool) -> bool {
    let up = Conversion::new(input_bits, output_bits, full_range);
    let down = Conversion::new(output_bits, input_bits, full_range);
    let largest = (1u32 << input_bits) - 1;

    (0..=largest).all(|value| {
        let raised = through(value as u16, &up);
        let lowered = through(raised, &down);

        u32::from(lowered) == value
    })
}

#[test]
fn eight_bit_round_trips_through_ten_and_sixteen() {
    assert!(round_trips(8, 10, false));
    assert!(round_trips(8, 16, false));
    assert!(round_trips(8, 10, true));
}

#[test]
fn limited_range_shifts_and_full_range_stretches() {
    let limited = Conversion::new(8, 10, false);
    let full = Conversion::new(8, 10, true);

    assert_eq!(through(235, &limited), 940);
    assert_eq!(through(255, &full), 1023);
}

#[test]
fn lowering_clamps_at_the_largest_value() {
    let limited = Conversion::new(10, 8, false);
    let full = Conversion::new(16, 8, true);

    assert_eq!(through(1023, &limited), 255);
    assert_eq!(through(65535, &full), 255);
}

#[test]
fn eight_bits_is_stored_as_u8_and_more_as_u16() {
    let conversion = Conversion::new(8, 12, false);

    assert_eq!(conversion.input, Source::U8);
    assert_eq!(conversion.output, Destination::U16);
    assert_eq!(conversion.largest, 4095.0);
}

#[test]
fn floats_stretch_one_to_the_largest_value() {
    let eight_bit = Conversion::from_float(8, SampleRange::Full, false);
    let ten_bit = Conversion::from_float(10, SampleRange::Full, false);

    assert_eq!(eight_bit.input, Source::F32);
    assert_eq!(eight_bit.output, Destination::U8);
    assert_eq!(through_float(0.0, &eight_bit), 0);
    assert_eq!(through_float(1.0, &eight_bit), 255);
    assert_eq!(ten_bit.output, Destination::U16);
    assert_eq!(through_float(1.0, &ten_bit), 1023);
}

#[test]
fn float_chroma_centres_zero_on_the_middle_value() {
    let eight_bit = Conversion::from_float(8, SampleRange::Full, true);
    let ten_bit = Conversion::from_float(10, SampleRange::Full, true);

    assert_eq!(through_float(0.0, &eight_bit), 128);
    assert_eq!(through_float(-0.5, &eight_bit), 1);
    assert_eq!(through_float(0.5, &eight_bit), 255);
    assert_eq!(through_float(0.0, &ten_bit), 512);
}

#[test]
fn floats_outside_the_range_clamp() {
    let conversion = Conversion::from_float(8, SampleRange::Full, false);

    assert_eq!(through_float(-0.5, &conversion), 0);
    assert_eq!(through_float(1.5, &conversion), 255);
}

#[test]
fn a_nan_float_reads_as_zero() {
    let luma = Conversion::from_float(8, SampleRange::Full, false);
    let chroma = Conversion::from_float(8, SampleRange::Full, true);

    assert_eq!(through_float(f32::NAN, &luma), 0);
    assert_eq!(through_float(f32::NAN, &chroma), 128);
}

#[test]
fn limited_luma_stretches_to_full_range_floats() {
    let conversion = Conversion::to_float(8, SampleRange::Limited, false);

    assert_eq!(convert_to_float_sample(16.0, &conversion), 0.0);
    assert_eq!(convert_to_float_sample(235.0, &conversion), 1.0);
}

#[test]
fn limited_chroma_centres_on_zero_as_a_float() {
    let conversion = Conversion::to_float(10, SampleRange::Limited, true);

    assert_eq!(convert_to_float_sample(512.0, &conversion), 0.0);
}

#[test]
fn a_float_round_trip_lands_where_a_shift_would() {
    let widen = Conversion::to_float(8, SampleRange::Limited, false);
    let narrow = Conversion::from_float(10, SampleRange::Limited, false);
    let shift = Conversion::new(8, 10, false);

    for code in [16.0, 128.0, 235.0] {
        let float = convert_to_float_sample(code, &widen);
        let through_float = convert_sample(float, &narrow, 0.5);
        let shifted = convert_sample(code, &shift, 0.5);

        assert_eq!(through_float, shifted);
    }
}

#[test]
fn full_range_floats_compress_into_limited_range() {
    let conversion = Conversion::from_float(10, SampleRange::Limited, false);

    assert_eq!(convert_sample(1.0, &conversion, 0.5), 940);
    assert_eq!(convert_sample(0.0, &conversion, 0.5), 64);
}
