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

/// Runs vinverse on the CPU in f64, mirroring [vinverse](crate::lineart::vinverse).
///
/// `amnt` is already scaled to the code values of `input`.
pub fn reference_vinverse(
    input: &[f32],
    width: u32,
    height: u32,
    contra_str: f32,
    scl: f32,
    amnt: Option<f32>,
) -> Vec<f32> {
    let values: Vec<f64> = input.iter().map(|value| f64::from(*value)).collect();
    let blurred = blur_121(&values, width, height);
    let diff = subtract(&values, &blurred);
    let diff_blurred = blur_121(&diff, width, height);

    let sbr: Vec<f64> = values
        .iter()
        .zip(&diff)
        .zip(&diff_blurred)
        .map(|((value, comb), comb_blurred)| remove_comb(*value, *comb, *comb_blurred))
        .collect();
    let sbr_blurred = blur_121(&sbr, width, height);

    let contra_str = f64::from(contra_str);
    let scl = f64::from(scl);
    let limit = amnt.map_or(f64::MAX, f64::from);

    values
        .iter()
        .zip(&sbr)
        .zip(&sbr_blurred)
        .map(|((value, base), base_blurred)| {
            let contra = (base - base_blurred) * contra_str;
            let removed = value - base;
            let smaller = if contra.abs() < removed.abs() {
                contra
            } else {
                removed
            };
            let step = if contra * removed < 0.0 {
                smaller * scl
            } else {
                smaller
            };
            let result = base + step;
            let capped = result.min(value + limit).max(value - limit);

            capped as f32
        })
        .collect()
}

fn remove_comb(value: f64, comb: f64, comb_blurred: f64) -> f64 {
    let residual = comb - comb_blurred;
    if residual * comb < 0.0 {
        return value;
    }

    let removed = if residual.abs() < comb.abs() {
        residual
    } else {
        comb
    };

    value - removed
}

fn subtract(first: &[f64], second: &[f64]) -> Vec<f64> {
    first
        .iter()
        .zip(second)
        .map(|(left, right)| left - right)
        .collect()
}

fn blur_121(input: &[f64], width: u32, height: u32) -> Vec<f64> {
    let rows = height as i32;
    let mut output = Vec::with_capacity(input.len());

    for row in 0..rows {
        let above = reflect(row - 1, rows) as u32;
        let below = reflect(row + 1, rows) as u32;

        for column in 0..width {
            let top = input[(above * width + column) as usize];
            let centre = input[(row as u32 * width + column) as usize];
            let bottom = input[(below * width + column) as usize];

            output.push((top + 2.0 * centre + bottom) / 4.0);
        }
    }

    output
}
