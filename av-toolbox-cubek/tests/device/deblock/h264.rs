use av_toolbox_cubek::deblock::h264::{
    DeblockPlanes,
    Thresholds,
    deblock,
    padded_frame,
    thresholds,
    upload_layout,
};
use av_toolbox_cubek::matrix::MatrixSamples;
use av_toolbox_cubek::testing::{
    H264_FIXTURES,
    H264Plane,
    assert_h264_plane,
    blocky_plane,
    decode_samples,
    encode_samples,
    reference_h264,
};
use av_toolbox_cubek::{DevicePlane, Dispatch};
use av_toolbox_gpu::{BuildContext, BuildTask};
use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::support::{Sizing, backend, empty, plane, read_u8, upload_u8};

/// The cube sizes the wavefront is checked at, against the frame context's.
const UNIT_COUNTS: [u32; 4] = [1, 7, 64, 256];

/// Sizes every launch with this many units to a cube, in one row of cubes.
struct Units(u32);

impl Dispatch for Units {
    fn cube_count(&self, samples: usize) -> CubeCount {
        let cubes = samples.div_ceil(self.0 as usize);

        CubeCount::Static(cubes as u32, 1, 1)
    }

    fn cube_dim(&self) -> CubeDim {
        CubeDim::new_1d(self.0)
    }
}

/// The planes of one frame as code values, with each plane's size and padded size.
struct TestFrame {
    values: Vec<Vec<f64>>,
    sizes: Vec<(u32, u32)>,
    padded: Vec<(u32, u32)>,
    samples: MatrixSamples,
}

impl TestFrame {
    /// A blocky `width` by `height` frame, with chroma planes `chroma_shift` down from luma when given, each
    /// plane padded to its share of the padded frame.
    fn blocky(width: u32, height: u32, chroma_shift: Option<(u32, u32)>, samples: MatrixSamples) -> Self {
        let (padded_width, padded_height) = padded_frame(width, height);
        let mut sizes = vec![(width, height)];
        let mut padded = vec![(padded_width, padded_height)];

        if let Some((shift_x, shift_y)) = chroma_shift {
            let chroma = (width.div_ceil(1 << shift_x), height.div_ceil(1 << shift_y));
            let padded_chroma = (padded_width >> shift_x, padded_height >> shift_y);
            sizes.extend([chroma, chroma]);
            padded.extend([padded_chroma, padded_chroma]);
        }

        let values = sizes
            .iter()
            .zip(0u32..)
            .map(|((plane_width, plane_height), seed)| {
                blocky_plane(*plane_width, *plane_height, samples, seed)
            })
            .collect();

        Self {
            values,
            sizes,
            padded,
            samples,
        }
    }

    /// This frame's plane `index` on its own.
    fn only(&self, index: usize) -> Self {
        Self {
            values: vec![self.values[index].clone()],
            sizes: vec![self.sizes[index]],
            padded: vec![self.padded[index]],
            samples: self.samples,
        }
    }

    /// Every plane deblocked on the CPU.
    fn reference(&self, frame_thresholds: Thresholds) -> Vec<Vec<f64>> {
        let planes = self.values.iter().zip(&self.sizes).zip(&self.padded);

        planes
            .map(|((values, (width, height)), (padded_width, padded_height))| {
                let plane = H264Plane {
                    values,
                    width: *width,
                    height: *height,
                    padded_width: *padded_width,
                    padded_height: *padded_height,
                };

                reference_h264(plane, self.samples, &frame_thresholds)
            })
            .collect()
    }
}

/// Planes deblocked in one call on the device, given and returned as little-endian sample bytes.
struct Deblock {
    planes: Vec<Vec<u8>>,
    sizes: Vec<(u32, u32)>,
    padded: Vec<(u32, u32)>,
    samples: MatrixSamples,
    thresholds: Thresholds,
    /// The units to a cube, or the frame context's cube size when `None`.
    units: Option<u32>,
    /// The scratch's size in bytes, or the layout's own when `None`.
    scratch_bytes: Option<usize>,
}

fn device_planes<'a>(handles: &'a [Handle], sizes: &[(u32, u32)]) -> Vec<DevicePlane<'a>> {
    handles
        .iter()
        .zip(sizes)
        .map(|(handle, (width, height))| plane(handle, *width, *height))
        .collect()
}

impl BuildTask for Deblock {
    type Output = Vec<Vec<u8>>;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let layout = upload_layout(client, &self.padded);
        let layout_bytes = layout.bytes();
        let scratch_bytes = self.scratch_bytes.unwrap_or(layout_bytes);
        let scratch = empty(client, scratch_bytes);

        let inputs: Vec<Handle> = self.planes.iter().map(|bytes| upload_u8(client, bytes)).collect();
        let outputs: Vec<Handle> = self
            .planes
            .iter()
            .map(|bytes| empty(client, bytes.len()))
            .collect();
        let input_planes = device_planes(&inputs, &self.sizes);
        let output_planes = device_planes(&outputs, &self.sizes);
        let planes = DeblockPlanes {
            inputs: &input_planes,
            outputs: &output_planes,
        };

        match self.units {
            Some(units) => {
                let dispatch = Units(units);
                deblock(
                    client,
                    &dispatch,
                    planes,
                    &scratch,
                    &layout,
                    self.samples,
                    self.thresholds,
                );
            },
            None => {
                let sizing = Sizing(build.frame_context());
                deblock(
                    client,
                    &sizing,
                    planes,
                    &scratch,
                    &layout,
                    self.samples,
                    self.thresholds,
                );
            },
        }

        outputs.iter().map(|handle| read_u8(client, handle)).collect()
    }
}

fn task(frame: &TestFrame, frame_thresholds: Thresholds, units: Option<u32>) -> Deblock {
    let planes = frame
        .values
        .iter()
        .map(|values| encode_samples(values, frame.samples))
        .collect();

    Deblock {
        planes,
        sizes: frame.sizes.clone(),
        padded: frame.padded.clone(),
        samples: frame.samples,
        thresholds: frame_thresholds,
        units,
        scratch_bytes: None,
    }
}

/// `frame` deblocked on the device, as code values.
fn deblocked(frame: &TestFrame, frame_thresholds: Thresholds, units: Option<u32>) -> Vec<Vec<f64>> {
    let run = task(frame, frame_thresholds, units);
    let written = backend().build(run);

    written
        .iter()
        .map(|bytes| decode_samples(bytes, frame.samples))
        .collect()
}

#[test]
fn the_kernel_matches_every_plugin_fixture() {
    let mut largest = 0.0f32;

    for fixture in H264_FIXTURES {
        let case = fixture.case();
        let frame = TestFrame {
            values: fixture.input_planes(),
            sizes: case.plane_sizes(),
            padded: case.padded_sizes(),
            samples: case.samples,
        };
        let expected = fixture.output_planes();

        let found = deblocked(&frame, case.thresholds(), None);

        for (index, found_plane) in found.iter().enumerate() {
            let (width, _) = frame.sizes[index];
            let label = format!("{} plane {index}", fixture.name);
            let difference = assert_h264_plane(found_plane, &expected[index], width, case.samples, &label);
            largest = largest.max(difference);
        }
    }

    println!("the largest float difference of the device from the plugin is {largest:e}");
}

#[test]
fn a_1080p_plane_matches_the_reference_exactly() {
    for samples in [
        MatrixSamples::U8,
        MatrixSamples::U16 { bits: 16 },
        MatrixSamples::F32,
    ] {
        let frame = TestFrame::blocky(1920, 1080, None, samples);
        let frame_thresholds = thresholds(51, 0, 0, samples);
        let expected = frame.reference(frame_thresholds);

        let found = deblocked(&frame, frame_thresholds, None);

        let label = format!("1080p {samples:?}");
        assert_h264_plane(&found[0], &expected[0], 1920, samples, &label);
        assert_ne!(found[0], frame.values[0], "{label} came back unchanged");
    }
}

#[test]
fn luma_and_two_420_chroma_planes_in_one_launch_match_each_plane_alone() {
    let frame = TestFrame::blocky(70, 46, Some((1, 1)), MatrixSamples::U8);
    let frame_thresholds = thresholds(51, 0, 0, MatrixSamples::U8);

    let together = deblocked(&frame, frame_thresholds, None);

    for (index, together_plane) in together.iter().enumerate() {
        let single = frame.only(index);
        let alone = deblocked(&single, frame_thresholds, None);

        assert_eq!(*together_plane, alone[0], "plane {index}");
    }

    let expected = frame.reference(frame_thresholds);
    assert_eq!(together, expected);
    assert_ne!(together, frame.values);
}

#[test]
fn different_workgroup_sizes_agree() {
    let samples = MatrixSamples::U16 { bits: 10 };
    let frame = TestFrame::blocky(136, 72, Some((1, 1)), samples);
    let frame_thresholds = thresholds(40, 0, 0, samples);
    let expected = frame.reference(frame_thresholds);

    let context_sized = deblocked(&frame, frame_thresholds, None);

    assert_eq!(context_sized, expected, "the frame context's cube size");
    for units in UNIT_COUNTS {
        let found = deblocked(&frame, frame_thresholds, Some(units));

        assert_eq!(found, expected, "{units} units to a cube");
    }
}

#[test]
fn steps_wider_than_a_cube_match_the_reference_at_256_units() {
    let frame = TestFrame::blocky(1024, 1024, None, MatrixSamples::U8);
    let frame_thresholds = thresholds(51, 0, 0, MatrixSamples::U8);
    let expected = frame.reference(frame_thresholds);

    let found = deblocked(&frame, frame_thresholds, Some(256));

    assert_h264_plane(
        &found[0],
        &expected[0],
        1024,
        MatrixSamples::U8,
        "1024x1024 at 256 units",
    );
    assert_ne!(
        found[0], frame.values[0],
        "1024x1024 at 256 units came back unchanged"
    );
}

#[test]
#[should_panic(expected = "the deblocking scratch holds 4 bytes, expected 1024")]
fn a_scratch_too_small_for_the_layout_is_refused() {
    let frame = TestFrame::blocky(16, 16, None, MatrixSamples::U8);
    let frame_thresholds = thresholds(25, 0, 0, MatrixSamples::U8);
    let mut run = task(&frame, frame_thresholds, None);
    run.scratch_bytes = Some(4);

    backend().build(run);
}

#[test]
#[should_panic(expected = "thresholds peak at 255 but U16 { bits: 10 } samples peak at 1023")]
fn thresholds_made_for_other_samples_are_refused() {
    let samples = MatrixSamples::U16 { bits: 10 };
    let frame = TestFrame::blocky(16, 16, None, samples);
    let eight_bit_thresholds = thresholds(25, 0, 0, MatrixSamples::U8);
    let run = task(&frame, eight_bit_thresholds, None);

    backend().build(run);
}
