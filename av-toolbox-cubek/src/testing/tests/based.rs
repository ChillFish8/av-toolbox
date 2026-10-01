use crate::artcnn::{Model, Weights};
use crate::testing::based::{
    Direction,
    DoubledPlanes,
    HorizontalSclip,
    antialias,
    composed,
    peak_of,
    reached_output,
};
use crate::testing::eedi3::{decode_plane, metadata_text};
use crate::testing::{
    BasedParameters,
    BasedStages,
    Eedi3Parameters,
    assert_same_plane,
    reference_supersample,
    transposed_plane,
};

/// Builds a [BasedFixture] from the files written for `name`.
macro_rules! based_fixture {
    ($name:literal) => {
        BasedFixture {
            name: $name,
            metadata: include_str!(concat!("../fixtures/based/", $name, ".txt")),
            luma: include_bytes!(concat!("../fixtures/based/", $name, ".luma")),
            supersampled: include_bytes!(concat!("../fixtures/based/", $name, ".ss")),
            antialiased: include_bytes!(concat!("../fixtures/based/", $name, ".aa2")),
            output: include_bytes!(concat!("../fixtures/based/", $name, ".out")),
            mask: include_bytes!(concat!("../fixtures/based/", $name, ".mask")),
            mclip: include_bytes!(concat!("../fixtures/based/", $name, ".mclip")),
        }
    };
}

/// One luma plane run through a model of `aa/based` at its defaults, with a built mask.
///
/// Code-value planes hold one byte per sample at 8 bits and two little-endian bytes above. `mask` holds
/// counts `k` for a mask of `k * peak / 9`, two to a byte with the low nibble first. `mclip` holds one
/// bit per doubled sample, least significant first.
struct BasedFixture {
    name: &'static str,
    metadata: &'static str,
    luma: &'static [u8],
    supersampled: &'static [u8],
    antialiased: &'static [u8],
    output: &'static [u8],
    mask: &'static [u8],
    mclip: &'static [u8],
}

const BASED_FIXTURES: [BasedFixture; 2] = [based_fixture!("lines_8"), based_fixture!("clip_16")];

/// A 24 by 16 luma plane at 8 bits and ArtCNN's doubled luma of it, one byte per sample.
const TINY_METADATA: &str = include_str!("../fixtures/based/tiny_8.txt");
const TINY_LUMA: &[u8] = include_bytes!("../fixtures/based/tiny_8.luma");
const TINY_SUPERSAMPLED: &[u8] = include_bytes!("../fixtures/based/tiny_8.ss");

fn setting(metadata: &str, key: &str) -> u32 {
    let text = metadata_text(metadata, key);

    text.parse().expect("the metadata value is an integer")
}

fn bytes_as_codes(bytes: &[u8]) -> Vec<f32> {
    bytes.iter().map(|byte| f32::from(*byte)).collect()
}

impl BasedFixture {
    fn width(&self) -> u32 {
        setting(self.metadata, "width")
    }

    fn height(&self) -> u32 {
        setting(self.metadata, "height")
    }

    fn bits(&self) -> u32 {
        setting(self.metadata, "bits")
    }

    fn parameters(&self) -> BasedParameters {
        BasedParameters {
            bits: self.bits(),
            ..BasedParameters::default()
        }
    }

    fn codes(&self, bytes: &[u8]) -> Vec<f32> {
        if self.bits() == 8 {
            return bytes_as_codes(bytes);
        }

        decode_plane(bytes)
    }

    fn luma_plane(&self) -> Vec<f32> {
        self.codes(self.luma)
    }

    fn supersampled_plane(&self) -> Vec<f32> {
        self.codes(self.supersampled)
    }

    fn mask_plane(&self) -> Vec<f32> {
        let peak = peak_of(self.bits());

        self.mask
            .iter()
            .flat_map(|byte| [byte & 0x0f, byte >> 4])
            .map(|count| f32::from(count) * peak / 9.0)
            .collect()
    }

    fn mclip_plane(&self) -> Vec<f32> {
        self.mclip
            .iter()
            .flat_map(|byte| (0..8).map(move |bit| (byte >> bit) & 1))
            .map(|set| f32::from(set) * 255.0)
            .collect()
    }

    /// The composed reference on this fixture's luma and doubled luma.
    fn run(&self, parameters: &BasedParameters, mask: Option<&[f32]>, sclip: HorizontalSclip) -> BasedStages {
        let luma = self.luma_plane();
        let supersampled = self.supersampled_plane();

        composed(
            &luma,
            self.width(),
            self.height(),
            parameters,
            mask,
            &supersampled,
            sclip,
        )
    }

    /// EEDI3's touched set on this fixture's doubled luma and `mclip` along `direction`.
    fn touched_along(&self, direction: Direction) -> Vec<bool> {
        let supersampled = self.supersampled_plane();
        let mclip = self.mclip_plane();
        let doubled = DoubledPlanes {
            supersampled: &supersampled,
            mclip: Some(&mclip),
            width: 2 * self.width(),
            height: 2 * self.height(),
        };
        let parameters = self.eedi3_parameters();

        let antialiased = antialias(&doubled, &parameters, direction, HorizontalSclip::Supersampled);

        antialiased.touched
    }

    fn eedi3_parameters(&self) -> Eedi3Parameters {
        let parameters = self.parameters();

        Eedi3Parameters {
            bits: parameters.bits,
            ..parameters.eedi3
        }
    }
}

/// How far a merged value may lie from a half code and still round either way.
const CLOSE_ROUNDING: f32 = 1e-4;

/// Panics where `found` and `expected` differ by more than a code, or where `merged` is not close to a half
/// code.
fn assert_close_roundings(found: &[f32], expected: &[f32], merged: &[f32], width: u32, label: &str) {
    for (index, (actual, wanted)) in found.iter().zip(expected).enumerate() {
        if actual == wanted {
            continue;
        }

        let column = index as u32 % width;
        let row = index as u32 / width;
        let fraction = merged[index] - merged[index].floor();
        let distance = (fraction - 0.5).abs();
        assert!(
            (actual - wanted).abs() == 1.0 && distance <= CLOSE_ROUNDING,
            "{label} differs at {column},{row}: found {actual}, expected {wanted}, merged {}",
            merged[index],
        );
    }
}

#[test]
fn the_composed_reference_matches_the_model() {
    for fixture in &BASED_FIXTURES {
        let parameters = fixture.parameters();
        let width = fixture.width();
        let doubled_width = 2 * width;

        let stages = fixture.run(&parameters, None, HorizontalSclip::Supersampled);

        let name = fixture.name;
        let mask = fixture.mask_plane();
        let mclip = fixture.mclip_plane();
        let antialiased = fixture.codes(fixture.antialiased);
        let output = fixture.codes(fixture.output);

        assert_same_plane(&stages.mask, &mask, width, &format!("{name} mask"));
        assert_same_plane(&stages.mclip, &mclip, doubled_width, &format!("{name} mclip"));
        assert_same_plane(
            &stages.antialiased,
            &antialiased,
            doubled_width,
            &format!("{name} aa2"),
        );

        let label = format!("{name} output");
        assert_close_roundings(&stages.output, &output, &stages.merged, width, &label);
    }
}

#[test]
fn a_bound_mask_is_used_as_it_is() {
    let fixture = &BASED_FIXTURES[1];
    let parameters = fixture.parameters();
    let luma = fixture.luma_plane();
    let samples = luma.len();
    let width = fixture.width();
    let empty = vec![0.0; samples];
    let full = vec![255.0; samples];
    let unmasked = BasedParameters {
        edge_mask: false,
        ..parameters
    };

    let emptied = fixture.run(&parameters, Some(&empty), HorizontalSclip::Supersampled);
    let filled = fixture.run(&parameters, Some(&full), HorizontalSclip::Supersampled);
    let filtered = fixture.run(&unmasked, None, HorizontalSclip::Supersampled);

    assert_eq!(emptied.mask, empty);
    assert_eq!(filled.mask, full);
    assert_same_plane(&emptied.output, &luma, width, "an empty bound mask");
    assert_same_plane(&filled.output, &filtered.output, width, "a full bound mask");
    assert_ne!(filled.output, luma, "a full bound mask leaves the luma unchanged");
}

#[test]
fn edge_mask_false_skips_the_merge() {
    let fixture = &BASED_FIXTURES[1];
    let parameters = fixture.parameters();
    let unmasked = BasedParameters {
        edge_mask: false,
        ..parameters
    };
    let luma = fixture.luma_plane();

    let masked = fixture.run(&parameters, None, HorizontalSclip::Supersampled);
    let filtered = fixture.run(&unmasked, None, HorizontalSclip::Supersampled);

    assert!(filtered.mask.is_empty() && filtered.mclip.is_empty());

    let outside_mask: Vec<usize> = (0..luma.len())
        .filter(|index| masked.mask[*index] == 0.0)
        .collect();
    let masked_changes = outside_mask
        .iter()
        .filter(|index| masked.output[**index] != luma[**index])
        .count();
    let filtered_changes = outside_mask
        .iter()
        .filter(|index| filtered.output[**index] != luma[**index])
        .count();
    assert_eq!(masked_changes, 0);
    assert!(filtered_changes > 0, "nothing changed outside the built mask");
}

#[test]
fn the_horizontal_pass_uses_the_original_clips() {
    let fixture = &BASED_FIXTURES[0];
    let parameters = fixture.parameters();
    let doubled_width = 2 * fixture.width();
    let expected = fixture.codes(fixture.antialiased);

    let original = fixture.run(&parameters, None, HorizontalSclip::Supersampled);
    let derived = fixture.run(&parameters, None, HorizontalSclip::PassInput);

    assert_same_plane(
        &original.antialiased,
        &expected,
        doubled_width,
        "the original clips",
    );
    assert_ne!(
        derived.antialiased, expected,
        "clips from the vertical result match the model"
    );
}

#[test]
fn a_horizontal_run_touches_what_a_vertical_run_on_the_transposed_planes_touches() {
    let fixture = &BASED_FIXTURES[0];
    let doubled_width = 2 * fixture.width();
    let doubled_height = 2 * fixture.height();
    let supersampled = fixture.supersampled_plane();
    let mclip = fixture.mclip_plane();
    let turned_supersampled = transposed_plane(&supersampled, doubled_width, doubled_height);
    let turned_mclip = transposed_plane(&mclip, doubled_width, doubled_height);
    let turned = DoubledPlanes {
        supersampled: &turned_supersampled,
        mclip: Some(&turned_mclip),
        width: doubled_height,
        height: doubled_width,
    };
    let parameters = fixture.eedi3_parameters();

    let horizontal = fixture.touched_along(Direction::Horizontal);
    let vertical = antialias(
        &turned,
        &parameters,
        Direction::Vertical,
        HorizontalSclip::Supersampled,
    );

    let expected = transposed_plane(&vertical.touched, doubled_height, doubled_width);
    assert!(
        expected.contains(&true),
        "the transposed vertical run touches nothing"
    );
    assert_same_plane(&horizontal, &expected, doubled_width, "horizontal touched");
}

#[test]
fn a_run_in_both_directions_touches_everything_the_vertical_run_touches() {
    let fixture = &BASED_FIXTURES[0];

    let both = fixture.touched_along(Direction::Both);
    let vertical = fixture.touched_along(Direction::Vertical);

    assert!(vertical.contains(&true), "the vertical run touches nothing");

    let missing = vertical
        .iter()
        .zip(&both)
        .position(|(in_vertical, in_both)| *in_vertical && !*in_both);
    assert_eq!(
        missing, None,
        "a sample the vertical run touches is untouched in both"
    );
}

const REACH_SIZE: u32 = 12;

/// A doubled plane for a `REACH_SIZE` square luma plane, touched only at `column`, `row`.
fn doubled_touched_at(column: u32, row: u32) -> Vec<bool> {
    let doubled_size = 2 * REACH_SIZE;
    let mut touched = vec![false; (doubled_size * doubled_size) as usize];
    touched[(row * doubled_size + column) as usize] = true;

    touched
}

/// The positions of `touched` in a `REACH_SIZE` square plane.
fn touched_positions(touched: &[bool]) -> Vec<(u32, u32)> {
    touched
        .iter()
        .enumerate()
        .filter(|(_, mark)| **mark)
        .map(|(index, _)| (index as u32 % REACH_SIZE, index as u32 / REACH_SIZE))
        .collect()
}

/// `centre` and the samples 1 and 3 away from it on both axes at once.
fn bilateral_reads(centre: u32) -> Vec<(u32, u32)> {
    let lines = [centre - 3, centre - 1, centre, centre + 1, centre + 3];

    let mut reads = Vec::new();
    for row in lines {
        for column in lines {
            let centre_row = row == centre;
            let centre_column = column == centre;
            if centre_row != centre_column {
                continue;
            }

            reads.push((column, row));
        }
    }

    reads
}

#[test]
fn a_touched_sample_reaches_its_halved_sample_alone_without_the_postfilter() {
    let doubled = doubled_touched_at(13, 12);

    let reached = reached_output(&doubled, REACH_SIZE, REACH_SIZE, false, None);

    assert_eq!(touched_positions(&reached), [(6, 6)]);
}

#[test]
fn a_touched_sample_reaches_every_sample_whose_bilateral_reads_it() {
    let doubled = doubled_touched_at(13, 12);

    let reached = reached_output(&doubled, REACH_SIZE, REACH_SIZE, true, None);

    assert_eq!(touched_positions(&reached), bilateral_reads(6));
}

#[test]
fn a_touched_corner_reaches_further_through_clamped_reads() {
    let doubled = doubled_touched_at(1, 0);

    let reached = reached_output(&doubled, REACH_SIZE, REACH_SIZE, true, None);

    let mut block = Vec::new();
    for row in 0..4 {
        for column in 0..4 {
            block.push((column, row));
        }
    }
    assert_eq!(touched_positions(&reached), block);
}

#[test]
fn the_merge_keeps_marks_only_under_the_mask() {
    let doubled = doubled_touched_at(13, 12);
    let mut mask = vec![9.0; (REACH_SIZE * REACH_SIZE) as usize];
    mask[(6 * REACH_SIZE + 6) as usize] = 0.0;

    let reached = reached_output(&doubled, REACH_SIZE, REACH_SIZE, true, Some(&mask));

    let expected: Vec<(u32, u32)> = bilateral_reads(6)
        .into_iter()
        .filter(|position| *position != (6, 6))
        .collect();
    assert_eq!(touched_positions(&reached), expected);
}

#[test]
fn supersampling_rounds_artcnn_to_codes() {
    let model = Model::R8F64;
    let bytes = av_toolbox_models::builtin("artcnn", model.file_name()).expect("R8F64 is built in");
    let weights = Weights::parse(bytes, model.features(), model.blocks()).expect("the builtin weights parse");
    let width = setting(TINY_METADATA, "width");
    let height = setting(TINY_METADATA, "height");
    let bits = setting(TINY_METADATA, "bits");
    let luma = bytes_as_codes(TINY_LUMA);
    let expected = bytes_as_codes(TINY_SUPERSAMPLED);

    let supersampled = reference_supersample(&luma, width, height, &weights, bits);

    assert_same_plane(&supersampled, &expected, 2 * width, "the supersampled luma");
}
