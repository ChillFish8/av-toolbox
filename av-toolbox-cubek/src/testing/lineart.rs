use super::reflect;

/// Merges on the CPU in f64, mirroring [merge](crate::lineart::merge).
pub fn reference_merge(input: &[f32], filtered: &[f32], weight: &[f32], weight_peak: f32) -> Vec<f32> {
    let peak = f64::from(weight_peak);

    input
        .iter()
        .zip(filtered)
        .zip(weight)
        .map(|((value, target), sample_weight)| {
            let value = f64::from(*value);
            let change = f64::from(*target) - value;
            let merged = value + change * f64::from(*sample_weight) / peak;

            merged as f32
        })
        .collect()
}

/// The vertical `1 2 1` blur on the CPU, mirroring [vertical_121](crate::lineart::vertical_121).
pub fn reference_vertical_121(input: &[f32], width: u32, height: u32) -> Vec<f32> {
    let rows = height as i32;
    let mut output = Vec::with_capacity(input.len());

    for row in 0..rows {
        let above = reflect(row - 1, rows) as u32;
        let below = reflect(row + 1, rows) as u32;

        for column in 0..width {
            let top = f64::from(input[(above * width + column) as usize]);
            let centre = f64::from(input[(row as u32 * width + column) as usize]);
            let bottom = f64::from(input[(below * width + column) as usize]);
            let blurred = (top + 2.0 * centre + bottom) / 4.0;

            output.push(blurred as f32);
        }
    }

    output
}
