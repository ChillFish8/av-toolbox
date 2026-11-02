use av_toolbox_cubek::deblock::pp7::{Pp7Mode, Pp7Planes, aligned_frame, pp7_plane, upload_thresholds};
use av_toolbox_cubek::matrix::MatrixSamples;
use av_toolbox_cubek::testing::{
    PP7_FIXTURES,
    Pp7Plane,
    assert_same_plane,
    blocky_plane,
    count_pp7_differences,
    decode_samples,
    encode_samples,
    noise_plane,
    plugin_output_at_peak,
    reference_pp7,
    reference_pp7_plane,
};
use av_toolbox_gpu::{BuildContext, BuildTask};

use crate::support::{backend, plane, read_u8, upload_u8};

/// The byte the output starts as, and every guard byte past the plane holds.
const GUARD: u8 = 0xA5;

const MODES: [Pp7Mode; 3] = [Pp7Mode::Hard, Pp7Mode::Soft, Pp7Mode::Medium];

/// One plane deblocked on the device, given and returned as little-endian sample bytes.
struct Deblock {
    input: Vec<u8>,
    width: u32,
    height: u32,
    aligned_size: (u32, u32),
    samples: MatrixSamples,
    qp: f64,
    mode: Pp7Mode,
    /// The samples the threshold table is made for.
    table_samples: MatrixSamples,
    /// The size the output plane is given as.
    output_size: (u32, u32),
    /// How many bytes of [GUARD] follow the plane in the output buffer.
    guard_bytes: usize,
    /// Whether the output plane is given the input's buffer.
    in_place: bool,
}

impl BuildTask for Deblock {
    type Output = Vec<u8>;

    fn run(self, build: &BuildContext) -> Self::Output {
        let client = build.client();
        let table = upload_thresholds(client, self.qp, self.table_samples);
        let input = upload_u8(client, &self.input);
        let starting_bytes = self.input.len() + self.guard_bytes;
        let starting = vec![GUARD; starting_bytes];
        let output = upload_u8(client, &starting);
        let written = if self.in_place { &input } else { &output };

        let (output_width, output_height) = self.output_size;
        let planes = Pp7Planes {
            input: plane(&input, self.width, self.height),
            output: plane(written, output_width, output_height),
        };

        pp7_plane(client, planes, self.aligned_size, &table, self.mode, self.samples);

        read_u8(client, written)
    }
}

/// `plane` deblocked with `qp` and `mode` on the device, with no guard bytes.
fn task(plane: Pp7Plane<'_>, samples: MatrixSamples, qp: f64, mode: Pp7Mode) -> Deblock {
    let input = encode_samples(plane.values, samples);

    Deblock {
        input,
        width: plane.width,
        height: plane.height,
        aligned_size: (plane.aligned_width, plane.aligned_height),
        samples,
        qp,
        mode,
        table_samples: samples,
        output_size: (plane.width, plane.height),
        guard_bytes: 0,
        in_place: false,
    }
}

/// `plane`'s output from the device, as code values.
fn on_device(plane: Pp7Plane<'_>, samples: MatrixSamples, qp: f64, mode: Pp7Mode) -> Vec<f64> {
    let run = task(plane, samples, qp, mode);
    let written = backend().build(run);

    decode_samples(&written, samples)
}

/// `values` as one `width` by `height` plane, aligned the way a frame of its own size aligns it.
fn whole_plane(values: &[f64], width: u32, height: u32) -> Pp7Plane<'_> {
    let (aligned_width, aligned_height) = aligned_frame(width, height);

    Pp7Plane {
        values,
        width,
        height,
        aligned_width,
        aligned_height,
    }
}

#[test]
fn the_kernel_matches_the_reference_and_the_plugin_on_every_fixture() {
    for fixture in PP7_FIXTURES {
        let case = fixture.case();
        let inputs = fixture.input_planes();
        let outputs = fixture.output_planes();
        let sizes = case.plane_sizes();
        let aligned_sizes = case.aligned_sizes();
        let expected = reference_pp7(&case, &inputs);

        let mut differing = 0;
        for (index, (width, height)) in sizes.iter().enumerate() {
            let (aligned_width, aligned_height) = aligned_sizes[index];
            let plane = Pp7Plane {
                values: &inputs[index],
                width: *width,
                height: *height,
                aligned_width,
                aligned_height,
            };

            let found = on_device(plane, case.samples, case.qp, case.mode);

            let label = format!("{} plane {index}", fixture.name);
            let (plugin, _) = plugin_output_at_peak(&outputs[index], case.samples);
            assert_same_plane(&found, &expected[index], *width, &label);
            differing += count_pp7_differences(&found, &plugin, *width, case.samples, &label);
        }

        assert!(
            differing <= fixture.allowed,
            "{}: {differing} samples past the float tolerance, {} allowed",
            fixture.name,
            fixture.allowed,
        );
    }
}

#[test]
fn a_1080p_plane_matches_the_reference_exactly_at_every_depth_and_mode() {
    let depths = [
        MatrixSamples::U8,
        MatrixSamples::U16 { bits: 10 },
        MatrixSamples::U16 { bits: 16 },
        MatrixSamples::F32,
    ];

    for samples in depths {
        let blocky = blocky_plane(1920, 1080, samples, 21);
        let stored = encode_samples(&blocky, samples);
        let values = decode_samples(&stored, samples);
        let plane = whole_plane(&values, 1920, 1080);

        for mode in MODES {
            let expected = reference_pp7_plane(plane, samples, 2.0, mode);

            let found = on_device(plane, samples, 2.0, mode);

            let label = format!("1080p {samples:?} under {mode:?}");
            assert_same_plane(&found, &expected, 1920, &label);
            assert!(found != values, "{label}: nothing moved");
        }
    }
}

#[test]
fn full_range_16_bit_noise_needs_and_gets_64_bit_sums() {
    let samples = MatrixSamples::U16 { bits: 16 };
    let values = noise_plane(256, 128, samples, 22);
    let plane = whole_plane(&values, 256, 128);

    for mode in MODES {
        let expected = reference_pp7_plane(plane, samples, 1.0, mode);

        let found = on_device(plane, samples, 1.0, mode);

        let label = format!("16-bit noise under {mode:?}");
        assert_same_plane(&found, &expected, 256, &label);
    }
}

#[test]
fn tiles_on_every_edge_and_planes_smaller_than_a_tile_match_the_reference() {
    let sizes = [
        (1, 1),
        (5, 3),
        (17, 2),
        (15, 15),
        (16, 16),
        (47, 30),
        (46, 31),
        (33, 17),
    ];
    let depths = [
        MatrixSamples::U8,
        MatrixSamples::U16 { bits: 12 },
        MatrixSamples::F32,
    ];

    for (width, height) in sizes {
        for samples in depths {
            let values = noise_plane(width, height, samples, width * 64 + height);
            let plane = whole_plane(&values, width, height);
            let expected = reference_pp7_plane(plane, samples, 3.0, Pp7Mode::Medium);

            let found = on_device(plane, samples, 3.0, Pp7Mode::Medium);

            let label = format!("{width}x{height} {samples:?}");
            assert_same_plane(&found, &expected, width, &label);
        }
    }
}

#[test]
fn samples_above_a_10_bit_peak_match_the_reference() {
    let samples = MatrixSamples::U16 { bits: 10 };
    let twelve_bit = MatrixSamples::U16 { bits: 12 };
    let values = noise_plane(64, 48, twelve_bit, 23);
    let plane = whole_plane(&values, 64, 48);
    let expected = reference_pp7_plane(plane, samples, 2.0, Pp7Mode::Hard);

    let found = on_device(plane, samples, 2.0, Pp7Mode::Hard);

    assert_same_plane(&found, &expected, 64, "10-bit samples up to 4095");
    let within_peak = found.iter().all(|value| *value <= 1023.0);
    assert!(within_peak, "an output passed the peak");
}

#[test]
fn writes_stay_inside_the_plane() {
    let samples = MatrixSamples::U8;
    let values = noise_plane(70, 46, samples, 24);
    let plane = whole_plane(&values, 70, 46);
    let expected = reference_pp7_plane(plane, samples, 2.0, Pp7Mode::Hard);
    let mut run = task(plane, samples, 2.0, Pp7Mode::Hard);
    run.guard_bytes = 64;

    let written = backend().build(run);

    let (inside, past) = written.split_at(70 * 46);
    let found = decode_samples(inside, samples);
    assert_same_plane(&found, &expected, 70, "70x46");
    let guarded = past.iter().all(|byte| *byte == GUARD);
    assert!(guarded, "a write landed past the plane");
}

#[test]
#[should_panic(expected = "the threshold table was made for U8 samples, not U16 { bits: 10 }")]
fn a_threshold_table_for_other_samples_is_refused() {
    let samples = MatrixSamples::U16 { bits: 10 };
    let values = noise_plane(16, 16, samples, 25);
    let plane = whole_plane(&values, 16, 16);
    let mut run = task(plane, samples, 2.0, Pp7Mode::Hard);
    run.table_samples = MatrixSamples::U8;

    backend().build(run);
}

#[test]
#[should_panic(expected = "the output plane is 15x16, expected 16x16")]
fn an_output_plane_of_another_size_is_refused() {
    let samples = MatrixSamples::U8;
    let values = noise_plane(16, 16, samples, 26);
    let plane = whole_plane(&values, 16, 16);
    let mut run = task(plane, samples, 2.0, Pp7Mode::Hard);
    run.output_size = (15, 16);

    backend().build(run);
}

#[test]
#[should_panic(expected = "a 16x16 plane does not fit a 16x8 alignment")]
fn an_alignment_smaller_than_the_plane_is_refused() {
    let samples = MatrixSamples::U8;
    let values = noise_plane(16, 16, samples, 27);
    let plane = whole_plane(&values, 16, 16);
    let mut run = task(plane, samples, 2.0, Pp7Mode::Hard);
    run.aligned_size = (16, 8);

    backend().build(run);
}

#[test]
#[should_panic(expected = "pp7 cannot write the plane it reads")]
fn an_output_on_the_input_buffer_is_refused() {
    let samples = MatrixSamples::U8;
    let values = noise_plane(16, 16, samples, 28);
    let plane = whole_plane(&values, 16, 16);
    let mut run = task(plane, samples, 2.0, Pp7Mode::Hard);
    run.in_place = true;

    backend().build(run);
}

#[test]
#[should_panic(expected = "a 65536x32768 plane has more samples than pp7 can index")]
fn a_plane_past_an_i32_index_is_refused() {
    let samples = MatrixSamples::U8;
    let values = noise_plane(16, 16, samples, 29);
    let plane = whole_plane(&values, 16, 16);
    let mut run = task(plane, samples, 2.0, Pp7Mode::Hard);
    run.width = 65536;
    run.height = 32768;
    run.output_size = (65536, 32768);
    run.aligned_size = (65536, 32768);

    backend().build(run);
}

#[test]
#[should_panic(expected = "a 4294967295x16 alignment is wider than pp7 can fold")]
fn an_alignment_past_half_an_i32_is_refused() {
    let samples = MatrixSamples::U8;
    let values = noise_plane(16, 16, samples, 30);
    let plane = whole_plane(&values, 16, 16);
    let mut run = task(plane, samples, 2.0, Pp7Mode::Hard);
    run.aligned_size = (u32::MAX, 16);

    backend().build(run);
}

#[test]
#[should_panic(expected = "a pp7 qp must be between 1 and 63, not 0.5")]
fn a_qp_below_1_is_refused() {
    let samples = MatrixSamples::U8;
    let values = noise_plane(16, 16, samples, 31);
    let plane = whole_plane(&values, 16, 16);
    let run = task(plane, samples, 0.5, Pp7Mode::Hard);

    backend().build(run);
}
