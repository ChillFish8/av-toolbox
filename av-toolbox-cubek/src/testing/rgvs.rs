use crate::rgvs::RepairMode;

/// Repairs `clip` from `reference` on the CPU, mirroring [repair](crate::rgvs::repair).
pub fn reference_repair(
    clip: &[f32],
    reference: &[f32],
    width: u32,
    height: u32,
    mode: RepairMode,
) -> Vec<f32> {
    let mut output = Vec::with_capacity(clip.len());

    for y in 0..height {
        for x in 0..width {
            let index = (y * width + x) as usize;
            let border = x == 0 || y == 0 || x + 1 == width || y + 1 == height;

            if border {
                output.push(clip[index]);
                continue;
            }

            let repaired = match mode {
                RepairMode::One => repair_one(clip[index], reference, width, x, y),
            };

            output.push(repaired);
        }
    }

    output
}

fn repair_one(value: f32, reference: &[f32], width: u32, x: u32, y: u32) -> f32 {
    let mut lowest = f32::MAX;
    let mut highest = f32::MIN;

    for row in y - 1..=y + 1 {
        for column in x - 1..=x + 1 {
            let neighbour = reference[(row * width + column) as usize];
            lowest = lowest.min(neighbour);
            highest = highest.max(neighbour);
        }
    }

    value.clamp(lowest, highest)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repair_clamps_the_clip_into_the_reference_neighbourhood() {
        // The centre's reference neighbourhood spans 10 to 18, so 40 clamps to 18 and 2 to 10.
        let reference: Vec<f32> = (10..19).map(|value| value as f32).collect();
        let mut high = vec![0.0; 9];
        high[4] = 40.0;
        let mut low = vec![0.0; 9];
        low[4] = 2.0;

        let raised = reference_repair(&high, &reference, 3, 3, RepairMode::One);
        let lowered = reference_repair(&low, &reference, 3, 3, RepairMode::One);

        assert_eq!(raised[4], 18.0);
        assert_eq!(lowered[4], 10.0);
    }

    #[test]
    fn repair_copies_the_border_of_the_clip() {
        let clip: Vec<f32> = (0..16).map(|value| value as f32 * 100.0).collect();
        let reference = vec![5.0; 16];

        let repaired = reference_repair(&clip, &reference, 4, 4, RepairMode::One);

        for index in [0, 1, 2, 3, 4, 7, 8, 11, 12, 13, 14, 15] {
            assert_eq!(repaired[index], clip[index], "sample {index}");
        }

        assert_eq!((repaired[5], repaired[10]), (5.0, 5.0));
    }
}
