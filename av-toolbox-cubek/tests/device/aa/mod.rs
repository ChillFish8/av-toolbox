mod eedi2;
mod eedi3;
mod merge;
mod sangnom;
mod transpose;

/// `values`, `width` samples wide, with its rows and columns swapped.
pub fn transposed_plane(values: &[f32], width: u32, height: u32) -> Vec<f32> {
    let columns = width as usize;
    let rows = height as usize;

    let mut swapped = vec![0.0; values.len()];
    for row in 0..rows {
        for column in 0..columns {
            swapped[column * rows + row] = values[row * columns + column];
        }
    }

    swapped
}
