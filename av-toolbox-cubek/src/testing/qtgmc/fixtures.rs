use std::path::Path;

use crate::testing::motion::{MotionClip, motion_clips};

/// One plane of a fixture frame as code values.
pub struct FixturePlane {
    pub width: usize,
    pub height: usize,
    pub samples: Vec<i32>,
}

/// A step's fixture, every frame of every clip, each frame its planes in Y, U, V order
/// (Y alone for luma steps).
pub struct QtgmcFixture {
    pub clip: &'static str,
    pub bits: u32,
    pub frames: Vec<Vec<FixturePlane>>,
}

/// The input clips and the frames of each that are cut out of the motion clips.
const INPUTS: [(&str, usize); 3] = [("pan_8", 0), ("cut_8", 2), ("noisy_10", 0)];

/// How many frames each input clip holds.
const INPUT_FRAMES: usize = 5;

#[derive(Clone, Copy)]
enum Planes {
    Luma,
    Full,
}

struct FixtureFile {
    step: &'static str,
    name: &'static str,
    clip: &'static str,
    planes: Planes,
    /// How many frames the file holds for each input frame.
    frames_per_input: usize,
    bytes: &'static [u8],
}

macro_rules! fixture_file {
    ($step:literal, $name:literal, $clip:literal, $planes:expr, $frames_per_input:literal) => {
        FixtureFile {
            step: $step,
            name: $name,
            clip: $clip,
            planes: $planes,
            frames_per_input: $frames_per_input,
            bytes: include_bytes!(concat!(
                "../fixtures/qtgmc/",
                $step,
                "/",
                $clip,
                ".",
                $name,
                ".u16"
            )),
        }
    };
}

macro_rules! fixture_files {
    ($step:literal, $name:literal, $planes:expr, $frames_per_input:literal) => {
        [
            fixture_file!($step, $name, "pan_8", $planes, $frames_per_input),
            fixture_file!($step, $name, "cut_8", $planes, $frames_per_input),
            fixture_file!($step, $name, "noisy_10", $planes, $frames_per_input),
        ]
    };
}

macro_rules! input_file {
    ($clip:literal) => {
        FixtureFile {
            step: "inputs",
            name: "frames",
            clip: $clip,
            planes: Planes::Full,
            frames_per_input: 1,
            bytes: include_bytes!(concat!("../fixtures/qtgmc/inputs/", $clip, ".u16")),
        }
    };
}

macro_rules! input_files {
    () => {
        [
            input_file!("pan_8"),
            input_file!("cut_8"),
            input_file!("noisy_10"),
        ]
    };
}

const FILES: &[[FixtureFile; 3]] = &[
    input_files!(),
    fixture_files!("bob", "out", Planes::Full, 2),
    fixture_files!("normalise", "out", Planes::Luma, 2),
    fixture_files!("average", "tr1", Planes::Luma, 2),
    fixture_files!("average", "tr2", Planes::Luma, 2),
    fixture_files!("average", "tr2_yuv", Planes::Full, 2),
    fixture_files!("shimmer", "out", Planes::Full, 2),
    fixture_files!("search", "out", Planes::Luma, 2),
    fixture_files!("resharpen", "out", Planes::Full, 2),
    fixture_files!("backblend", "out", Planes::Luma, 2),
    fixture_files!("clamp", "max", Planes::Full, 2),
    fixture_files!("clamp", "min", Planes::Full, 2),
    fixture_files!("clamp", "out", Planes::Full, 2),
    fixture_files!("vblur", "out", Planes::Full, 1),
    fixture_files!("repair", "tff", Planes::Full, 1),
    fixture_files!("repair", "bff", Planes::Full, 1),
];

/// The width, height, bit depth and frame count the input listing gives for `clip`.
fn input_shape(clip: &str) -> (usize, usize, u32, usize) {
    let listing = match clip {
        "pan_8" => include_str!("../fixtures/qtgmc/inputs/pan_8.txt"),
        "cut_8" => include_str!("../fixtures/qtgmc/inputs/cut_8.txt"),
        "noisy_10" => include_str!("../fixtures/qtgmc/inputs/noisy_10.txt"),
        other => panic!("{other} is not a QTGMC fixture clip"),
    };
    let words: Vec<usize> = listing
        .split_whitespace()
        .map(|word| word.parse().expect("the listing holds numbers"))
        .collect();

    (words[0], words[1], words[2] as u32, words[3])
}

fn plane_sizes(width: usize, height: usize, planes: Planes) -> Vec<(usize, usize)> {
    match planes {
        Planes::Luma => vec![(width, height)],
        Planes::Full => vec![(width, height), (width / 2, height / 2), (width / 2, height / 2)],
    }
}

fn load(file: &FixtureFile) -> QtgmcFixture {
    let (width, height, bits, input_frames) = input_shape(file.clip);
    let sizes = plane_sizes(width, height, file.planes);
    let frame_count = input_frames * file.frames_per_input;

    let (words, remainder) = file.bytes.as_chunks::<2>();
    assert!(
        remainder.is_empty(),
        "{}/{} has a stray byte",
        file.step,
        file.name
    );

    let mut samples = words.iter().map(|word| i32::from(u16::from_le_bytes(*word)));
    let mut frames = Vec::with_capacity(frame_count);
    for _ in 0..frame_count {
        let mut planes = Vec::with_capacity(sizes.len());
        for (plane_width, plane_height) in &sizes {
            let plane_samples: Vec<i32> = samples.by_ref().take(plane_width * plane_height).collect();
            planes.push(FixturePlane {
                width: *plane_width,
                height: *plane_height,
                samples: plane_samples,
            });
        }

        frames.push(planes);
    }

    assert!(
        samples.next().is_none(),
        "{}/{} for {} holds more than {frame_count} frames",
        file.step,
        file.name,
        file.clip,
    );
    assert_eq!(
        frames.last().map(|planes| planes[0].samples.len()),
        Some(width * height),
        "{}/{} for {} holds fewer than {frame_count} frames",
        file.step,
        file.name,
        file.clip,
    );

    QtgmcFixture {
        clip: file.clip,
        bits,
        frames,
    }
}

/// The fixture `step` holds under `name`, such as `("bob", "out")`, for every input clip.
/// The raw input frames are `("inputs", "frames")`.
pub fn qtgmc_fixture(step: &str, name: &str) -> Vec<QtgmcFixture> {
    let files = FILES
        .iter()
        .find(|files| files[0].step == step && files[0].name == name)
        .unwrap_or_else(|| panic!("there is no QTGMC fixture {step}/{name}"));

    files.iter().map(load).collect()
}

fn input_clip(name: &str) -> MotionClip {
    motion_clips()
        .into_iter()
        .find(|clip| clip.name == name)
        .unwrap_or_else(|| panic!("there is no motion clip {name}"))
}

/// Writes each input clip to `<directory>/<clip>.u16` as little-endian `u16`, frame after frame with its
/// three planes in order, and its width, height, bit depth and frame count to `<directory>/<clip>.txt`.
pub fn write_qtgmc_inputs(directory: &Path) {
    std::fs::create_dir_all(directory).expect("the inputs directory is created");

    for (name, first_frame) in INPUTS {
        let clip = input_clip(name);
        let frames = &clip.frames[first_frame..first_frame + INPUT_FRAMES];
        let bytes: Vec<u8> = frames
            .iter()
            .flat_map(|planes| planes.iter())
            .flat_map(|plane| plane.iter())
            .flat_map(|sample| (*sample as u16).to_le_bytes())
            .collect();
        std::fs::write(directory.join(format!("{name}.u16")), bytes).expect("the input frames are written");

        let listing = format!("{} {} {} {}\n", clip.width, clip.height, clip.bits, frames.len());
        std::fs::write(directory.join(format!("{name}.txt")), listing).expect("the input listing is written");
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    #[test]
    #[ignore = "writes the committed QTGMC fixture inputs, run by hand"]
    fn writes_the_qtgmc_fixture_inputs() {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("src/testing/fixtures/qtgmc/inputs");

        write_qtgmc_inputs(&path);
    }

    #[test]
    fn every_fixture_loads_with_its_frame_count_and_plane_sizes() {
        for files in FILES {
            let file = &files[0];
            let fixtures = qtgmc_fixture(file.step, file.name);
            assert_eq!(fixtures.len(), INPUTS.len());

            for fixture in &fixtures {
                let (width, height, bits, input_frames) = input_shape(fixture.clip);
                let sizes = plane_sizes(width, height, file.planes);
                assert_eq!(fixture.bits, bits);
                assert_eq!(fixture.frames.len(), input_frames * file.frames_per_input);

                for planes in &fixture.frames {
                    assert_eq!(planes.len(), sizes.len());
                    for (plane, (plane_width, plane_height)) in planes.iter().zip(&sizes) {
                        assert_eq!((plane.width, plane.height), (*plane_width, *plane_height));
                        assert_eq!(plane.samples.len(), plane_width * plane_height);
                    }
                }
            }
        }
    }

    #[test]
    fn the_bob_output_has_twice_the_frames_of_its_input() {
        let inputs = qtgmc_fixture("inputs", "frames");
        let outputs = qtgmc_fixture("bob", "out");

        for (input, output) in inputs.iter().zip(&outputs) {
            assert_eq!(input.clip, output.clip);
            assert_eq!(output.frames.len(), 2 * input.frames.len());
        }
    }

    #[test]
    fn the_committed_inputs_are_the_motion_clip_frames() {
        for (fixture, (name, first_frame)) in qtgmc_fixture("inputs", "frames").iter().zip(INPUTS) {
            let clip = input_clip(name);

            for (index, planes) in fixture.frames.iter().enumerate() {
                for (plane, expected) in planes.iter().zip(&clip.frames[first_frame + index]) {
                    assert_eq!(&plane.samples, expected, "{name} frame {index}");
                }
            }
        }
    }

    #[test]
    fn sample_values_stay_within_the_bit_depth() {
        for fixture in qtgmc_fixture("bob", "out") {
            let peak = (1 << fixture.bits) - 1;
            let largest = fixture
                .frames
                .iter()
                .flatten()
                .flat_map(|plane| plane.samples.iter().copied())
                .max();

            assert!(largest.is_some_and(|value| value <= peak), "{}", fixture.clip);
        }
    }

    #[test]
    fn every_new_fixture_loads_with_one_frame_per_frame() {
        let inputs = qtgmc_fixture("inputs", "frames");

        for (step, name) in [("vblur", "out"), ("repair", "tff"), ("repair", "bff")] {
            let outputs = qtgmc_fixture(step, name);
            assert_eq!(outputs.len(), inputs.len());

            for (input, output) in inputs.iter().zip(&outputs) {
                assert_eq!(input.clip, output.clip);
                assert_eq!(
                    output.frames.len(),
                    input.frames.len(),
                    "{step}/{name} {}",
                    input.clip
                );
            }
        }
    }

    #[test]
    fn the_repair_fixture_keeps_parity_n_mod_2() {
        let inputs = qtgmc_fixture("inputs", "frames");

        // Top field first keeps the top field of frame 0, bottom field first keeps the bottom field.
        for (name, first_parity) in [("tff", 0), ("bff", 1)] {
            let outputs = qtgmc_fixture("repair", name);

            for (input, output) in inputs.iter().zip(&outputs) {
                // A last frame without a partner field is not a kept field, so only whole pairs are checked.
                let paired_frames = input.frames.len() / 2 * 2;

                for index in 0..paired_frames {
                    let parity = (index + first_parity) % 2;

                    for (plane, expected) in output.frames[index].iter().zip(&input.frames[index]) {
                        let kept_rows = plane.samples.chunks(plane.width).skip(parity).step_by(2);
                        let input_rows = expected.samples.chunks(expected.width).skip(parity).step_by(2);

                        assert!(
                            kept_rows.eq(input_rows),
                            "repair/{name} {} frame {index} does not keep parity {parity}",
                            input.clip,
                        );
                    }

                    let other_parity = 1 - parity;
                    let output_luma = &output.frames[index][0];
                    let input_luma = &input.frames[index][0];
                    let other_output_rows = output_luma.samples.chunks(output_luma.width).skip(other_parity);
                    let other_input_rows = input_luma.samples.chunks(input_luma.width).skip(other_parity);

                    assert!(
                        !other_output_rows.step_by(2).eq(other_input_rows.step_by(2)),
                        "repair/{name} {} frame {index} also matches parity {other_parity}",
                        input.clip,
                    );
                }
            }
        }
    }
}
