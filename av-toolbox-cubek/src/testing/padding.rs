/// Grows a plane on the CPU, mirroring [pad](crate::padding::pad).
pub fn reference_pad(input: &[f32], width: u32, height: u32, amount: u32) -> Vec<f32> {
    let signed_width = width as i32;
    let signed_height = height as i32;
    let signed_amount = amount as i32;
    let output_width = width + 2 * amount;
    let output_height = height + 2 * amount;

    let mut output = Vec::with_capacity((output_width * output_height) as usize);
    for row in 0..output_height as i32 {
        for column in 0..output_width as i32 {
            let source_column = (column - signed_amount).clamp(0, signed_width - 1);
            let source_row = (row - signed_amount).clamp(0, signed_height - 1);
            let index = source_row * signed_width + source_column;

            output.push(input[index as usize]);
        }
    }

    output
}

/// Removes a plane's border on the CPU, mirroring [crop](crate::padding::crop).
pub fn reference_crop(input: &[f32], width: u32, height: u32, amount: u32) -> Vec<f32> {
    let output_width = width - 2 * amount;
    let output_height = height - 2 * amount;

    let mut output = Vec::with_capacity((output_width * output_height) as usize);
    for row in 0..output_height {
        for column in 0..output_width {
            let source_column = column + amount;
            let source_row = row + amount;
            let index = source_row * width + source_column;

            output.push(input[index as usize]);
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn padding_repeats_the_nearest_edge() {
        let input = [1.0, 2.0, 3.0, 4.0];

        let padded = reference_pad(&input, 2, 2, 1);

        let expected = [
            1.0, 1.0, 2.0, 2.0, //
            1.0, 1.0, 2.0, 2.0, //
            3.0, 3.0, 4.0, 4.0, //
            3.0, 3.0, 4.0, 4.0, //
        ];
        assert_eq!(padded, expected);
    }

    #[test]
    fn cropping_undoes_padding() {
        let input: Vec<f32> = (0..15).map(|value| value as f32).collect();

        let padded = reference_pad(&input, 5, 3, 6);
        let cropped = reference_crop(&padded, 17, 15, 6);

        assert_eq!(cropped, input);
    }

    #[test]
    fn a_single_sample_pads_to_a_flat_square() {
        let input = [7.0];

        let padded = reference_pad(&input, 1, 1, 6);

        assert_eq!(padded.len(), 13 * 13);
        assert!(padded.iter().all(|value| *value == 7.0));
    }
}
