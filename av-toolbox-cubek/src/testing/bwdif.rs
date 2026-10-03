use crate::deinterlace::{
    BwdifField,
    CUBIC_CENTRE,
    CUBIC_OUTER,
    HIGH_CENTRE,
    HIGH_FAR,
    HIGH_NEAR,
    LOW_CENTRE,
    LOW_OUTER,
};

/// One case written by the plugin's own scalar code, through the float path.
///
/// `previous`, `current`, `next` and `spatial` are little-endian u16 code values, and `output` little-endian f32.
/// `metadata` holds the `width`, `height`, `bits`, `rebuilt`, `first` and `edeint` it was written with.
pub struct BwdifFixture {
    pub name: &'static str,
    pub metadata: &'static str,
    pub previous: &'static [u8],
    pub current: &'static [u8],
    pub next: &'static [u8],
    pub spatial: Option<&'static [u8]>,
    pub output: &'static [u8],
}

macro_rules! fixture {
    ($name:literal) => {
        BwdifFixture {
            name: $name,
            metadata: include_str!(concat!("fixtures/bwdif/", $name, ".txt")),
            previous: include_bytes!(concat!("fixtures/bwdif/", $name, ".prev")),
            current: include_bytes!(concat!("fixtures/bwdif/", $name, ".cur")),
            next: include_bytes!(concat!("fixtures/bwdif/", $name, ".next")),
            spatial: None,
            output: include_bytes!(concat!("fixtures/bwdif/", $name, ".out")),
        }
    };
    ($name:literal, spatial) => {
        BwdifFixture {
            spatial: Some(include_bytes!(concat!("fixtures/bwdif/", $name, ".edeint"))),
            ..fixture!($name)
        }
    };
}

/// Every bwdif field pass fixture.
pub const BWDIF_FIXTURES: [BwdifFixture; 7] = [
    fixture!("tff_first_8"),
    fixture!("tff_second_8"),
    fixture!("bff_first_10"),
    fixture!("bff_second_10"),
    fixture!("edeint_8", spatial),
    fixture!("short_8"),
    fixture!("short_edeint_10", spatial),
];

fn sample_of(pair: &[u8; 2]) -> f32 {
    let code = u16::from_le_bytes(*pair);

    f32::from(code)
}

fn decode_samples(bytes: &[u8]) -> Vec<f32> {
    let (pairs, remainder) = bytes.as_chunks::<2>();
    assert!(
        remainder.is_empty(),
        "a plane of u16 samples has an odd {} bytes",
        bytes.len()
    );

    pairs.iter().map(sample_of).collect()
}

fn decode_values(bytes: &[u8]) -> Vec<f32> {
    let (words, remainder) = bytes.as_chunks::<4>();
    assert!(
        remainder.is_empty(),
        "a plane of f32 values has {} stray bytes",
        remainder.len()
    );

    words.iter().map(|word| f32::from_le_bytes(*word)).collect()
}

fn metadata_value(metadata: &str, key: &str) -> u32 {
    for line in metadata.lines() {
        let mut parts = line.split_whitespace();
        if parts.next() != Some(key) {
            continue;
        }

        let value = parts.next().expect("every metadata key has a value");
        return value.parse().expect("every metadata value is an integer");
    }

    panic!("the fixture metadata has no `{key}`");
}

impl BwdifFixture {
    /// The value of `key` in the metadata.
    pub fn setting(&self, key: &str) -> u32 {
        metadata_value(self.metadata, key)
    }

    pub fn previous_plane(&self) -> Vec<f32> {
        decode_samples(self.previous)
    }

    pub fn current_plane(&self) -> Vec<f32> {
        decode_samples(self.current)
    }

    pub fn next_plane(&self) -> Vec<f32> {
        decode_samples(self.next)
    }

    pub fn spatial_plane(&self) -> Option<Vec<f32>> {
        self.spatial.map(decode_samples)
    }

    pub fn output_plane(&self) -> Vec<f32> {
        decode_values(self.output)
    }

    pub fn field(&self) -> BwdifField {
        BwdifField {
            rebuilt: self.setting("rebuilt"),
            first: self.setting("first") == 1,
        }
    }
}

/// The planes one bwdif field pass reads on the CPU, as f32 code values of one size.
pub struct BwdifReferencePlanes<'a> {
    pub previous: &'a [f32],
    pub current: &'a [f32],
    pub next: &'a [f32],
    pub spatial: Option<&'a [f32]>,
}

/// The same pass as [bwdif_field](crate::deinterlace::bwdif_field), on the CPU.
pub fn reference_bwdif(
    planes: &BwdifReferencePlanes<'_>,
    width: u32,
    height: u32,
    field: BwdifField,
) -> Vec<f32> {
    let mut output = planes.current.to_vec();

    for y in (field.rebuilt..height).step_by(2) {
        for x in 0..width {
            let index = (y * width + x) as usize;
            output[index] = rebuilt_sample(planes, width, height, field, x, y);
        }
    }

    output
}

/// The sample `offset` rows from `y` in column `x` of `plane`.
fn sample_at(plane: &[f32], width: u32, x: u32, y: u32, offset: i32) -> f32 {
    let row = y as i32 + offset;
    let index = row as u32 * width + x;

    plane[index as usize]
}

/// One rebuilt sample of bwdif's float path, as `filterEdge_c` and `filterLine_c` compute it.
fn rebuilt_sample(
    planes: &BwdifReferencePlanes<'_>,
    width: u32,
    height: u32,
    field: BwdifField,
    x: u32,
    y: u32,
) -> f32 {
    let (earlier, later) = if field.first {
        (planes.previous, planes.current)
    } else {
        (planes.current, planes.next)
    };

    let above = if y > 0 { -1 } else { 1 };
    let below = if y + 1 < height { 1 } else { -1 };

    let current_above = sample_at(planes.current, width, x, y, above);
    let current_below = sample_at(planes.current, width, x, y, below);
    let earlier_here = sample_at(earlier, width, x, y, 0);
    let later_here = sample_at(later, width, x, y, 0);
    let average = (earlier_here + later_here) * 0.5;

    let temporal_diff0 = (earlier_here - later_here).abs();
    let previous_above = (sample_at(planes.previous, width, x, y, above) - current_above).abs();
    let previous_below = (sample_at(planes.previous, width, x, y, below) - current_below).abs();
    let temporal_diff1 = (previous_above + previous_below) * 0.5;
    let next_above = (sample_at(planes.next, width, x, y, above) - current_above).abs();
    let next_below = (sample_at(planes.next, width, x, y, below) - current_below).abs();
    let temporal_diff2 = (next_above + next_below) * 0.5;

    let half_diff0 = temporal_diff0 * 0.5;
    let spread = half_diff0.max(temporal_diff1);
    let mut diff = spread.max(temporal_diff2);

    if diff == 0.0 {
        return average;
    }

    let inner = y >= 4 && y + 5 <= height;
    let reaches_two = y >= 2 && y + 3 <= height;

    if reaches_two {
        let earlier_above_two = sample_at(earlier, width, x, y, -2);
        let later_above_two = sample_at(later, width, x, y, -2);
        let earlier_below_two = sample_at(earlier, width, x, y, 2);
        let later_below_two = sample_at(later, width, x, y, 2);
        let outer_above = (earlier_above_two + later_above_two) * 0.5 - current_above;
        let outer_below = (earlier_below_two + later_below_two) * 0.5 - current_below;
        let towards_above = average - current_above;
        let towards_below = average - current_below;

        let near_high = towards_below.max(towards_above);
        let outer_low = outer_above.min(outer_below);
        let maximum = near_high.max(outer_low);
        let near_low = towards_below.min(towards_above);
        let outer_high = outer_above.max(outer_below);
        let minimum = near_low.min(outer_high);

        let widened = diff.max(minimum);
        diff = widened.max(-maximum);
    }

    let pair = current_above + current_below;
    let mut interpolated = pair * 0.5;
    if let Some(spatial) = planes.spatial {
        interpolated = sample_at(spatial, width, x, y, 0);
    }

    if inner {
        let current_above_three = sample_at(planes.current, width, x, y, -3);
        let current_below_three = sample_at(planes.current, width, x, y, 3);
        let current_outer = current_above_three + current_below_three;
        let edge_crosses_rows = (current_above - current_below).abs() > temporal_diff0;

        if edge_crosses_rows {
            let centre = earlier_here + later_here;
            let near = sample_at(earlier, width, x, y, -2)
                + sample_at(later, width, x, y, -2)
                + sample_at(earlier, width, x, y, 2)
                + sample_at(later, width, x, y, 2);
            let far = sample_at(earlier, width, x, y, -4)
                + sample_at(later, width, x, y, -4)
                + sample_at(earlier, width, x, y, 4)
                + sample_at(later, width, x, y, 4);
            let temporal = (HIGH_CENTRE * centre - HIGH_NEAR * near + HIGH_FAR * far) * 0.25;
            interpolated = temporal + LOW_CENTRE * pair - LOW_OUTER * current_outer;
        } else if planes.spatial.is_none() {
            interpolated = CUBIC_CENTRE * pair - CUBIC_OUTER * current_outer;
        }
    }

    let low = average - diff;
    let high = average + diff;

    interpolated.clamp(low, high)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_of(fixture: &BwdifFixture) -> Vec<f32> {
        let previous = fixture.previous_plane();
        let current = fixture.current_plane();
        let next = fixture.next_plane();
        let spatial = fixture.spatial_plane();
        let planes = BwdifReferencePlanes {
            previous: &previous,
            current: &current,
            next: &next,
            spatial: spatial.as_deref(),
        };
        let width = fixture.setting("width");
        let height = fixture.setting("height");

        reference_bwdif(&planes, width, height, fixture.field())
    }

    #[test]
    fn every_fixture_matches_the_plugin() {
        for fixture in &BWDIF_FIXTURES {
            let found = reference_of(fixture);
            let expected = fixture.output_plane();
            let width = fixture.setting("width");

            crate::testing::assert_same_plane(&found, &expected, width, fixture.name);
        }
    }

    #[test]
    fn a_static_plane_comes_out_unchanged() {
        let width = 9;
        let height = 10;
        let plane: Vec<f32> = (0..width * height)
            .map(|index| (index * 37 % 256) as f32)
            .collect();
        let planes = BwdifReferencePlanes {
            previous: &plane,
            current: &plane,
            next: &plane,
            spatial: None,
        };

        for rebuilt in 0..2 {
            for first in [true, false] {
                let field = BwdifField { rebuilt, first };
                let found = reference_bwdif(&planes, width, height, field);

                assert_eq!(found, plane, "rebuilt {rebuilt}, first {first}");
            }
        }
    }

    #[test]
    fn the_kept_rows_are_the_current_frame() {
        let fixture = &BWDIF_FIXTURES[0];
        let found = reference_of(fixture);
        let current = fixture.current_plane();
        let width = fixture.setting("width") as usize;
        let kept = 1 - fixture.field().rebuilt as usize;

        for (row, (found_row, current_row)) in found.chunks(width).zip(current.chunks(width)).enumerate() {
            if row % 2 == kept {
                assert_eq!(found_row, current_row, "row {row}");
            }
        }
    }
}
