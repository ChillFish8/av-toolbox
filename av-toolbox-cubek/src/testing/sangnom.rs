/// The samples a window reads, from 3 columns left of its centre to 3 right.
const TAPS: usize = 7;
/// The directions SangNom weighs, one cost buffer each.
const DIRECTIONS: usize = 9;
/// The vertical direction's cost buffer.
const VERTICAL: usize = 4;
/// The directions tried after the vertical one, most preferred first.
const PRIORITY: [usize; 8] = [5, 3, 6, 2, 7, 1, 8, 0];

type Window = [i64; TAPS];

/// The kept rows of one field, read with the column clamped into the plane.
struct KeptLines<'a> {
    samples: &'a [i64],
    width: usize,
    offset: usize,
}

impl KeptLines<'_> {
    /// Kept line `line` from 3 columns left of `column` to 3 right.
    fn window(&self, line: usize, column: usize) -> Window {
        let row = self.offset + 2 * line;
        let row_start = row * self.width;
        let last_column = self.width as i64 - 1;

        let mut window = [0; TAPS];
        for (tap, value) in window.iter_mut().enumerate() {
            let reached = column as i64 + tap as i64 - 3;
            let clamped = reached.clamp(0, last_column) as usize;
            *value = self.samples[row_start + clamped];
        }

        window
    }
}

/// The SangNom predictor, `(4 * first + 5 * second - third) / 8`, capped at `max`.
///
/// A negative sum gives `max`, as the plugin's unsigned shift does.
fn predict(first: i64, second: i64, third: i64, max: i64) -> i64 {
    let sum = 4 * first + 5 * second - third;
    if sum < 0 {
        return max;
    }

    let predicted = sum >> 3;
    predicted.min(max)
}

fn average(first: i64, second: i64) -> i64 {
    (first + second + 1) >> 1
}

/// The two values each direction pairs across the gap between the `current` and `next` kept lines.
///
/// A direction's cost is their difference and its candidate is their average.
fn direction_pairs(current: &Window, next: &Window, max: i64) -> [(i64, i64); DIRECTIONS] {
    let forward_current = predict(current[2], current[3], current[4], max);
    let forward_next = predict(next[4], next[3], next[2], max);
    let backward_current = predict(current[4], current[3], current[2], max);
    let backward_next = predict(next[2], next[3], next[4], max);

    [
        (current[0], next[6]),
        (current[1], next[5]),
        (current[2], next[4]),
        (forward_current, forward_next),
        (current[3], next[3]),
        (backward_current, backward_next),
        (current[4], next[2]),
        (current[5], next[1]),
        (current[6], next[0]),
    ]
}

/// Every direction's raw cost, one buffer each of `field_height + 1` rows.
///
/// Row `gap` holds the gap between kept lines `gap - 1` and `gap`. The first and last rows stay zero.
fn raw_costs(lines: &KeptLines, field_height: usize, max: i64) -> Vec<Vec<i64>> {
    let width = lines.width;
    let buffer_size = (field_height + 1) * width;

    let mut costs = vec![vec![0; buffer_size]; DIRECTIONS];
    for gap in 1..field_height {
        for column in 0..width {
            let current = lines.window(gap - 1, column);
            let next = lines.window(gap, column);
            let pairs = direction_pairs(&current, &next, max);
            for (direction, (first, second)) in pairs.iter().enumerate() {
                costs[direction][gap * width + column] = (first - second).abs();
            }
        }
    }

    costs
}

/// The sum of 7 columns of `line` around `column`, clamping at both edges.
fn box_sum(line: &[i64], column: usize) -> i64 {
    let last_column = line.len() as i64 - 1;

    (-3..=3)
        .map(|offset| {
            let reached = column as i64 + offset;
            let clamped = reached.clamp(0, last_column);
            line[clamped as usize]
        })
        .sum()
}

/// Smooths one direction's raw costs down the rows.
///
/// Each row adds the smoothed row above to its own raw costs and to the raw costs below. It then sums 7
/// columns of that and divides by 16, capped at `max`.
fn smooth_costs(raw: &[i64], width: usize, field_height: usize, max: i64) -> Vec<i64> {
    let mut smoothed = vec![0; raw.len()];
    let mut line = vec![0; width];

    for row in 1..field_height {
        for (column, value) in line.iter_mut().enumerate() {
            let above = smoothed[(row - 1) * width + column];
            let own = raw[row * width + column];
            let below = raw[(row + 1) * width + column];
            *value = above + own + below;
        }

        for column in 0..width {
            let total = box_sum(&line, column);
            let scaled = total / 16;
            smoothed[row * width + column] = scaled.min(max);
        }
    }

    smoothed
}

/// The direction to interpolate along, from the nine smoothed costs at one pixel.
fn choose_direction(costs: &[i64; DIRECTIONS], threshold: i64) -> usize {
    let lowest = costs.iter().copied().min().expect("there are nine costs");
    if costs[VERTICAL] == lowest || lowest > threshold {
        return VERTICAL;
    }

    PRIORITY
        .into_iter()
        .find(|direction| costs[*direction] == lowest)
        .expect("one of the other directions holds the lowest cost")
}

/// The rebuilt value at `column` in the gap between kept lines `gap - 1` and `gap`.
fn interpolate(
    lines: &KeptLines,
    smoothed: &[Vec<i64>],
    gap: usize,
    column: usize,
    threshold: i64,
    max: i64,
) -> i64 {
    let index = gap * lines.width + column;
    let costs: [i64; DIRECTIONS] = std::array::from_fn(|direction| smoothed[direction][index]);
    let direction = choose_direction(&costs, threshold);

    let current = lines.window(gap - 1, column);
    let next = lines.window(gap, column);
    let pairs = direction_pairs(&current, &next, max);
    let (first, second) = pairs[direction];

    average(first, second)
}

/// One SangNom field pass on the CPU, keeping the rows of parity `offset` and rebuilding the others.
///
/// The plane holds integer code values. `aaf` is the direction threshold from
/// [sangnom_aaf](crate::aa::sangnom_aaf), and `max` is 255 for 8-bit input and 65535 above. The row
/// with no kept row on one side copies its kept neighbour.
pub fn reference_sangnom_field(
    plane: &[f32],
    width: u32,
    height: u32,
    offset: u32,
    aaf: f32,
    max: f32,
) -> Vec<f32> {
    assert!(
        height >= 2 && height.is_multiple_of(2),
        "a SangNom plane needs an even height, not {height}"
    );
    assert!(offset < 2, "a SangNom field offset is 0 or 1, not {offset}");

    let samples_expected = width as usize * height as usize;
    assert_eq!(plane.len(), samples_expected, "the plane is not {width}x{height}");

    let width = width as usize;
    let height = height as usize;
    let offset = offset as usize;
    let threshold = aaf as i64;
    let max = max as i64;
    let field_height = height / 2;
    let samples: Vec<i64> = plane.iter().map(|value| *value as i64).collect();
    let lines = KeptLines {
        samples: &samples,
        width,
        offset,
    };

    let raw = raw_costs(&lines, field_height, max);
    let smoothed: Vec<Vec<i64>> = raw
        .iter()
        .map(|buffer| smooth_costs(buffer, width, field_height, max))
        .collect();

    let mut output = vec![0; samples.len()];
    for row in (offset..height).step_by(2) {
        let row_range = row * width..(row + 1) * width;
        output[row_range.clone()].copy_from_slice(&samples[row_range]);
    }

    for gap in 1..field_height {
        let row = offset + 2 * gap - 1;
        for column in 0..width {
            output[row * width + column] = interpolate(&lines, &smoothed, gap, column, threshold, max);
        }
    }

    let (edge_row, edge_source) = if offset == 0 {
        (height - 1, height - 2)
    } else {
        (0, 1)
    };
    let source_range = edge_source * width..(edge_source + 1) * width;
    output.copy_within(source_range, edge_row * width);

    output.iter().map(|value| *value as f32).collect()
}

/// SangNom's double-rate anti-alias on the CPU, `floor((pass(0) + pass(1) + 1) / 2)`.
///
/// Each pass is a [reference_sangnom_field](crate::testing::reference_sangnom_field), one keeping the
/// even rows and one the odd rows.
pub fn reference_sangnom_antialias(plane: &[f32], width: u32, height: u32, aaf: f32, max: f32) -> Vec<f32> {
    let samples_expected = width as usize * height as usize;
    assert_eq!(plane.len(), samples_expected, "the plane is not {width}x{height}");

    let even_kept = reference_sangnom_field(plane, width, height, 0, aaf, max);
    let odd_kept = reference_sangnom_field(plane, width, height, 1, aaf, max);

    even_kept
        .iter()
        .zip(&odd_kept)
        .map(|(first, second)| {
            let merged = average(*first as i64, *second as i64);
            merged as f32
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aa::sangnom_aaf;

    const WORKED_TXT: &str = include_str!("fixtures/sangnom/worked.txt");
    const WORKED_IN: &[u8] = include_bytes!("fixtures/sangnom/worked.in");
    const WORKED_PASS0: &[u8] = include_bytes!("fixtures/sangnom/worked.pass0");
    const WORKED_PASS1: &[u8] = include_bytes!("fixtures/sangnom/worked.pass1");
    const WORKED_AA: &[u8] = include_bytes!("fixtures/sangnom/worked.aa");
    const DIAGONAL_TXT: &str = include_str!("fixtures/sangnom/diagonal.txt");
    const DIAGONAL_IN: &[u8] = include_bytes!("fixtures/sangnom/diagonal.in");
    const DIAGONAL_PASS0: &[u8] = include_bytes!("fixtures/sangnom/diagonal.pass0");
    const DIAGONAL_PASS1: &[u8] = include_bytes!("fixtures/sangnom/diagonal.pass1");
    const DIAGONAL_AA: &[u8] = include_bytes!("fixtures/sangnom/diagonal.aa");
    const EXTREMES_TXT: &str = include_str!("fixtures/sangnom/extremes.txt");
    const EXTREMES_IN: &[u8] = include_bytes!("fixtures/sangnom/extremes.in");
    const EXTREMES_PASS0: &[u8] = include_bytes!("fixtures/sangnom/extremes.pass0");
    const EXTREMES_PASS1: &[u8] = include_bytes!("fixtures/sangnom/extremes.pass1");
    const EXTREMES_AA: &[u8] = include_bytes!("fixtures/sangnom/extremes.aa");
    const NOISE_TXT: &str = include_str!("fixtures/sangnom/noise.txt");
    const NOISE_IN: &[u8] = include_bytes!("fixtures/sangnom/noise.in");
    const NOISE_PASS0: &[u8] = include_bytes!("fixtures/sangnom/noise.pass0");
    const NOISE_PASS1: &[u8] = include_bytes!("fixtures/sangnom/noise.pass1");
    const NOISE_AA: &[u8] = include_bytes!("fixtures/sangnom/noise.aa");
    const WIDE_TXT: &str = include_str!("fixtures/sangnom/wide.txt");
    const WIDE_IN: &[u8] = include_bytes!("fixtures/sangnom/wide.in");
    const WIDE_PASS0: &[u8] = include_bytes!("fixtures/sangnom/wide.pass0");
    const WIDE_PASS1: &[u8] = include_bytes!("fixtures/sangnom/wide.pass1");
    const WIDE_AA: &[u8] = include_bytes!("fixtures/sangnom/wide.aa");
    const STRESS_TXT: &str = include_str!("fixtures/sangnom/stress.txt");
    const STRESS_IN: &[u8] = include_bytes!("fixtures/sangnom/stress.in");
    const STRESS_PASS0: &[u8] = include_bytes!("fixtures/sangnom/stress.pass0");
    const STRESS_PASS1: &[u8] = include_bytes!("fixtures/sangnom/stress.pass1");
    const STRESS_AA: &[u8] = include_bytes!("fixtures/sangnom/stress.aa");

    /// One case written by the plugin's own core with the smoothing spanning the plane's width.
    struct Fixture {
        name: &'static str,
        metadata: &'static str,
        input: &'static [u8],
        pass0: &'static [u8],
        pass1: &'static [u8],
        merged: &'static [u8],
    }

    const FIXTURES: [Fixture; 6] = [
        Fixture {
            name: "worked",
            metadata: WORKED_TXT,
            input: WORKED_IN,
            pass0: WORKED_PASS0,
            pass1: WORKED_PASS1,
            merged: WORKED_AA,
        },
        Fixture {
            name: "diagonal",
            metadata: DIAGONAL_TXT,
            input: DIAGONAL_IN,
            pass0: DIAGONAL_PASS0,
            pass1: DIAGONAL_PASS1,
            merged: DIAGONAL_AA,
        },
        Fixture {
            name: "extremes",
            metadata: EXTREMES_TXT,
            input: EXTREMES_IN,
            pass0: EXTREMES_PASS0,
            pass1: EXTREMES_PASS1,
            merged: EXTREMES_AA,
        },
        Fixture {
            name: "noise",
            metadata: NOISE_TXT,
            input: NOISE_IN,
            pass0: NOISE_PASS0,
            pass1: NOISE_PASS1,
            merged: NOISE_AA,
        },
        Fixture {
            name: "wide",
            metadata: WIDE_TXT,
            input: WIDE_IN,
            pass0: WIDE_PASS0,
            pass1: WIDE_PASS1,
            merged: WIDE_AA,
        },
        Fixture {
            name: "stress",
            metadata: STRESS_TXT,
            input: STRESS_IN,
            pass0: STRESS_PASS0,
            pass1: STRESS_PASS1,
            merged: STRESS_AA,
        },
    ];

    fn decode_plane(bytes: &[u8]) -> Vec<f32> {
        let (pairs, remainder) = bytes.as_chunks::<2>();
        assert!(
            remainder.is_empty(),
            "a plane of u16 samples has an odd {} bytes",
            bytes.len()
        );

        pairs
            .iter()
            .map(|pair| {
                let sample = u16::from_le_bytes(*pair);
                f32::from(sample)
            })
            .collect()
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

    #[test]
    fn the_worked_example_matches_by_hand() {
        let plane = decode_plane(WORKED_IN);
        // The port column of section 6.4 in the SangNom reference.
        let expected: Vec<f32> = vec![
            70.0, 155.0, 200.0, 200.0, 200.0, 200.0, //
            70.0, 110.0, 200.0, 200.0, 200.0, 200.0, //
            28.0, 87.0, 135.0, 203.0, 200.0, 200.0, //
            30.0, 30.0, 110.0, 178.0, 200.0, 200.0, //
            30.0, 28.0, 47.0, 110.0, 200.0, 200.0, //
            30.0, 30.0, 30.0, 110.0, 155.0, 200.0, //
        ];

        let merged = reference_sangnom_antialias(&plane, 6, 6, 63.0, 255.0);

        assert_eq!(merged, expected);
    }

    #[test]
    fn every_fixture_matches_the_plugin() {
        for fixture in &FIXTURES {
            let width = metadata_value(fixture.metadata, "width");
            let height = metadata_value(fixture.metadata, "height");
            let bits = metadata_value(fixture.metadata, "bits");
            let aa = metadata_value(fixture.metadata, "aa");
            let aaf = sangnom_aaf(aa, bits);
            let max = if bits == 8 { 255.0 } else { 65535.0 };
            let plane = decode_plane(fixture.input);
            let expected_pass0 = decode_plane(fixture.pass0);
            let expected_pass1 = decode_plane(fixture.pass1);
            let expected_merged = decode_plane(fixture.merged);
            let samples = width as usize * height as usize;
            assert_eq!(plane.len(), samples, "{} input", fixture.name);
            assert_eq!(expected_pass0.len(), samples, "{} pass 0", fixture.name);
            assert_eq!(expected_pass1.len(), samples, "{} pass 1", fixture.name);
            assert_eq!(expected_merged.len(), samples, "{} merged", fixture.name);

            let pass0 = reference_sangnom_field(&plane, width, height, 0, aaf, max);
            let pass1 = reference_sangnom_field(&plane, width, height, 1, aaf, max);
            let merged = reference_sangnom_antialias(&plane, width, height, aaf, max);

            assert_eq!(pass0, expected_pass0, "{} at offset 0", fixture.name);
            assert_eq!(pass1, expected_pass1, "{} at offset 1", fixture.name);
            assert_eq!(merged, expected_merged, "{} merged", fixture.name);
        }
    }

    #[test]
    fn aaf_scales_to_the_bit_depth() {
        // 48 * 21 / 16 = 63 exactly, and 63 * 2^2 = 252 at 10 bits.
        // 17 * 21 / 16 = 22.3125, which truncates to 22.
        // 128 * 21 / 16 = 168, and 168 * 2^8 = 43008 at 16 bits.
        assert_eq!(sangnom_aaf(48, 8), 63.0);
        assert_eq!(sangnom_aaf(48, 10), 252.0);
        assert_eq!(sangnom_aaf(17, 8), 22.0);
        assert_eq!(sangnom_aaf(128, 16), 43008.0);
    }
}
