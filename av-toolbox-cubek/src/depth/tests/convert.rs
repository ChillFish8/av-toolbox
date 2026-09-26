use crate::depth::{Conversion, Element, convert_sample};

fn through(value: u16, conversion: &Conversion) -> u16 {
    convert_sample(value, conversion.multiplier, 0.5, conversion.largest)
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

    assert_eq!(conversion.input, Element::U8);
    assert_eq!(conversion.output, Element::U16);
    assert_eq!(conversion.largest, 4095.0);
}
