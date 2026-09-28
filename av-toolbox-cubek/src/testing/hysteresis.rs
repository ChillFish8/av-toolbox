/// Floods every seed 8-connected through `region` on the CPU, mirroring
/// [hysteresis](crate::hysteresis::hysteresis).
///
/// A seed only counts where it is set in both `seeds_from` and `region`. Flooded pixels become
/// `peak`, everything else 0.
pub fn reference_hysteresis(
    seeds_from: &[f32],
    region: &[f32],
    width: u32,
    height: u32,
    peak: f32,
) -> Vec<f32> {
    let signed_width = width as i32;
    let signed_height = height as i32;

    let mut output = vec![0.0f32; region.len()];
    let mut visited = vec![false; region.len()];
    let mut pending = Vec::new();

    for index in 0..region.len() {
        let is_seed = seeds_from[index] > 0.0 && region[index] > 0.0;
        if is_seed && !visited[index] {
            visited[index] = true;
            pending.push(index as i32);
        }
    }

    while let Some(position) = pending.pop() {
        output[position as usize] = peak;

        let column = position % signed_width;
        let row = position / signed_width;
        for offset_y in -1..=1 {
            for offset_x in -1..=1 {
                if offset_x == 0 && offset_y == 0 {
                    continue;
                }

                let neighbour_column = column + offset_x;
                let neighbour_row = row + offset_y;
                let in_bounds = neighbour_column >= 0
                    && neighbour_column < signed_width
                    && neighbour_row >= 0
                    && neighbour_row < signed_height;
                if !in_bounds {
                    continue;
                }

                let neighbour = (neighbour_row * signed_width + neighbour_column) as usize;
                if region[neighbour] > 0.0 && !visited[neighbour] {
                    visited[neighbour] = true;
                    pending.push(neighbour as i32);
                }
            }
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A region built from `#` rows, `1.0` where set, `0.0` elsewhere.
    fn region(rows: &[&str]) -> (Vec<f32>, u32, u32) {
        let width = rows[0].len() as u32;
        let height = rows.len() as u32;
        let plane = rows
            .iter()
            .flat_map(|row| row.chars())
            .map(|cell| if cell == '#' { 1.0 } else { 0.0 })
            .collect();

        (plane, width, height)
    }

    fn all_zero(samples: usize) -> Vec<f32> {
        vec![0.0; samples]
    }

    #[test]
    fn a_component_with_a_seed_is_kept_whole() {
        let rows = [".....", ".#...", ".#...", ".###."];
        let (region, width, height) = region(&rows);
        let seed_index = (width + 1) as usize;
        let mut seeds_from = all_zero(region.len());
        seeds_from[seed_index] = 1.0;

        let output = reference_hysteresis(&seeds_from, &region, width, height, 255.0);

        let expected: Vec<f32> = region
            .iter()
            .map(|value| if *value > 0.0 { 255.0 } else { 0.0 })
            .collect();
        assert_eq!(output, expected);
    }

    #[test]
    fn a_component_without_a_seed_is_dropped() {
        let rows = [".....", ".#...", ".#...", ".###."];
        let (region, width, height) = region(&rows);
        let seeds_from = all_zero(region.len());

        let output = reference_hysteresis(&seeds_from, &region, width, height, 255.0);

        assert!(output.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn a_seed_only_in_the_seed_plane_does_not_count() {
        let rows = [".....", ".#...", ".#...", ".###."];
        let (region, width, height) = region(&rows);
        let mut seeds_from = all_zero(region.len());
        seeds_from[0] = 1.0;

        let output = reference_hysteresis(&seeds_from, &region, width, height, 255.0);

        assert!(output.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn diagonal_neighbours_join_one_component() {
        let rows = ["#.", ".#"];
        let (region, width, height) = region(&rows);
        let mut seeds_from = all_zero(region.len());
        seeds_from[0] = 1.0;

        let output = reference_hysteresis(&seeds_from, &region, width, height, 255.0);

        assert_eq!(output, vec![255.0, 0.0, 0.0, 255.0]);
    }

    #[test]
    fn components_are_independent() {
        let rows = ["#....", ".....", "....#"];
        let (region, width, height) = region(&rows);
        let mut seeds_from = all_zero(region.len());
        seeds_from[0] = 1.0;

        let output = reference_hysteresis(&seeds_from, &region, width, height, 255.0);

        let mut expected = all_zero(region.len());
        expected[0] = 255.0;
        assert_eq!(output, expected);
    }
}
