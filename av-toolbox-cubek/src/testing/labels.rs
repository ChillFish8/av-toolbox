use crate::labels::UNSET;

/// Labels every 8-connected component of the set `marks` on the CPU, mirroring
/// [label](crate::labels::label).
///
/// Each component is labelled with the smallest index inside it, and unset pixels with [UNSET].
pub fn reference_labels(marks: &[u8], width: u32, height: u32) -> Vec<u32> {
    let signed_width = width as i32;
    let signed_height = height as i32;

    let mut labels = vec![UNSET; marks.len()];
    let mut pending = Vec::new();

    for start in 0..marks.len() {
        if marks[start] == 0 || labels[start] != UNSET {
            continue;
        }

        let component = start as u32;
        labels[start] = component;
        pending.push(start as i32);

        while let Some(position) = pending.pop() {
            let column = position % signed_width;
            let row = position / signed_width;

            for offset_y in -1..=1 {
                for offset_x in -1..=1 {
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
                    if marks[neighbour] != 0 && labels[neighbour] == UNSET {
                        labels[neighbour] = component;
                        pending.push(neighbour as i32);
                    }
                }
            }
        }
    }

    labels
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Marks built from `#` rows, 255 where set and 0 elsewhere.
    fn marks(rows: &[&str]) -> (Vec<u8>, u32, u32) {
        let width = rows[0].len() as u32;
        let height = rows.len() as u32;
        let plane = rows
            .iter()
            .flat_map(|row| row.chars())
            .map(|cell| if cell == '#' { 255 } else { 0 })
            .collect();

        (plane, width, height)
    }

    #[test]
    fn each_component_takes_its_smallest_index() {
        let (plane, width, height) = marks(&["..#", "##.", "...", "#.#"]);

        let labels = reference_labels(&plane, width, height);

        assert_eq!(labels[2], 2);
        assert_eq!(labels[3], 2);
        assert_eq!(labels[4], 2);
        assert_eq!(labels[9], 9);
        assert_eq!(labels[11], 11);
    }

    #[test]
    fn unset_pixels_stay_unset() {
        let (plane, width, height) = marks(&["#.", ".."]);

        let labels = reference_labels(&plane, width, height);

        assert_eq!(labels, vec![0, UNSET, UNSET, UNSET]);
    }

    #[test]
    fn a_component_reaching_back_up_keeps_the_smallest_index() {
        let (plane, width, height) = marks(&["#...#", "#...#", "#####"]);

        let labels = reference_labels(&plane, width, height);

        assert!(
            labels
                .iter()
                .filter(|label| **label != UNSET)
                .all(|label| *label == 0)
        );
    }
}
