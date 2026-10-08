use av_toolbox_cubek::deblock::h264::{padded_frame, thresholds};
use av_toolbox_cubek::deblock::qed::{QedPlanes, lowpass_frame, qed_interiors, upload_change_table};
use av_toolbox_cubek::matrix::MatrixSamples;
use av_toolbox_cubek::testing::{
    H264Plane,
    QED_FIXTURES,
    QedPlane,
    assert_same_plane,
    blocky_plane,
    count_qed_differences,
    decode_samples,
    encode_samples,
    half_lowpass_planes,
    is_border_sample,
    reference_h264,
    reference_qed,
    reference_qed_plane,
};
use av_toolbox_gpu::{BuildContext, BuildTask};
use cubecl::prelude::*;

use crate::support::{Sizing, backend, plane, read_u8, upload_u8};

/// The byte every guard byte past the plane holds.
const GUARD: u8 = 0xA5;

/// One plane's interiors run on the device, given and returned as little-endian sample bytes.
struct Interiors {
    source: Vec<u8>,
    strong: Vec<u8>,
    /// The normal pass, which the output starts as.
    normal: Vec<u8>,
    width: u32,
    height: u32,
    lowpass_size: (u32, u32),
    samples: MatrixSamples,
    /// The samples the change table is made for.
    table_samples: MatrixSamples,
    /// The size the strong plane is given as.
    strong_size: (u32, u32),
    /// How many bytes of [GUARD] follow the plane in the output buffer.
    guard_bytes: usize,
}

impl BuildTask for Interiors {
    type Output = Vec<u8>;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let table = upload_change_table(client, self.table_samples);
        let source = upload_u8(client, &self.source);
        let strong = upload_u8(client, &self.strong);
        let guard = std::iter::repeat_n(GUARD, self.guard_bytes);
        let mut starting = self.normal;
        starting.extend(guard);
        let output = upload_u8(client, &starting);

        let (strong_width, strong_height) = self.strong_size;
        let planes = QedPlanes {
            source: plane(&source, self.width, self.height),
            strong: plane(&strong, strong_width, strong_height),
            output: plane(&output, self.width, self.height),
        };
        let sizing = Sizing(build.frame_context());

        qed_interiors(client, &sizing, planes, self.lowpass_size, &table, self.samples);

        read_u8(client, &output)
    }
}

/// `plane`'s interiors on the device, starting from its normal pass, with a change table for `samples`.
fn task(plane: QedPlane<'_>, samples: MatrixSamples) -> Interiors {
    let source = encode_samples(plane.source, samples);
    let strong = encode_samples(plane.strong, samples);
    let normal = encode_samples(plane.normal, samples);

    Interiors {
        source,
        strong,
        normal,
        width: plane.width,
        height: plane.height,
        lowpass_size: (plane.lowpass_width, plane.lowpass_height),
        samples,
        table_samples: samples,
        strong_size: (plane.width, plane.height),
        guard_bytes: 0,
    }
}

/// `plane`'s output from the device, as code values.
fn on_device(plane: QedPlane<'_>, samples: MatrixSamples) -> Vec<f64> {
    let run = task(plane, samples);
    let written = backend().build(run);

    decode_samples(&written, samples)
}

/// A plane with both its passes, at quant 24 and 26 with offsets 1 and 2.
struct Passes {
    source: Vec<f64>,
    normal: Vec<f64>,
    strong: Vec<f64>,
    width: u32,
    height: u32,
}

impl Passes {
    /// A blocky `width` by `height` plane of `samples`, as the device stores it, deblocked as luma of a frame
    /// its own size.
    fn blocky(width: u32, height: u32, samples: MatrixSamples, seed: u32) -> Self {
        let blocky = blocky_plane(width, height, samples, seed);
        let stored = encode_samples(&blocky, samples);
        let source = decode_samples(&stored, samples);
        let normal_thresholds = thresholds(24, 1, 2, samples);
        let strong_thresholds = thresholds(26, 1, 2, samples);
        let (padded_width, padded_height) = padded_frame(width, height);
        let deblocked = H264Plane {
            values: &source,
            width,
            height,
            padded_width,
            padded_height,
        };
        let normal = reference_h264(deblocked, samples, &normal_thresholds);
        let strong = reference_h264(deblocked, samples, &strong_thresholds);

        Self {
            source,
            normal,
            strong,
            width,
            height,
        }
    }

    fn plane(&self) -> QedPlane<'_> {
        let (lowpass_width, lowpass_height) = lowpass_frame(self.width, self.height);

        QedPlane {
            source: &self.source,
            normal: &self.normal,
            strong: &self.strong,
            width: self.width,
            height: self.height,
            lowpass_width,
            lowpass_height,
        }
    }
}

#[test]
fn the_kernel_matches_the_reference_and_the_plugin_on_every_fixture() {
    for fixture in QED_FIXTURES {
        let case = fixture.case();
        let inputs = fixture.input_planes();
        let normal = fixture.normal_planes();
        let strong = fixture.strong_planes();
        let output = fixture.output_planes();
        let sizes = case.plane_sizes();
        let lowpass_sizes = case.lowpass_sizes();
        let expected = reference_qed(&case, &inputs);

        let mut differing = 0;
        for index in case.qed_planes() {
            let (width, height) = sizes[index];
            let (lowpass_width, lowpass_height) = lowpass_sizes[index];
            let plane = QedPlane {
                source: &inputs[index],
                normal: &normal[index],
                strong: &strong[index],
                width,
                height,
                lowpass_width,
                lowpass_height,
            };

            let found = on_device(plane, case.samples);

            let label = format!("{} plane {index}", fixture.name);
            assert_same_plane(&found, &expected.output[index], width, &label);
            differing += count_qed_differences(&found, &output[index], width, case.samples, &label);
        }

        assert!(
            differing <= fixture.allowed,
            "{}: {differing} samples differ from the plugin, {} allowed",
            fixture.name,
            fixture.allowed,
        );
    }
}

#[test]
fn a_1080p_plane_matches_the_reference_exactly_at_every_depth() {
    let depths = [
        MatrixSamples::U8,
        MatrixSamples::U16 { bits: 9 },
        MatrixSamples::U16 { bits: 10 },
        MatrixSamples::U16 { bits: 16 },
        MatrixSamples::F32,
    ];

    for samples in depths {
        let frame = Passes::blocky(1920, 1080, samples, 11);
        let plane = frame.plane();
        let expected = reference_qed_plane(plane, samples);

        let found = on_device(plane, samples);

        let label = format!("1080p {samples:?}");
        assert_same_plane(&found, &expected, 1920, &label);
        assert!(found != frame.normal, "{label}: no interior moved");
    }
}

#[test]
fn a_420_chroma_plane_with_a_partial_block_row_matches_the_reference() {
    let samples = MatrixSamples::U8;
    let frame = Passes::blocky(960, 540, samples, 12);
    let plane = frame.plane();
    let expected = reference_qed_plane(plane, samples);

    let found = on_device(plane, samples);

    assert_eq!((plane.lowpass_width, plane.lowpass_height), (960, 544));
    assert_same_plane(&found, &expected, 960, "960x540 chroma");
}

#[test]
fn a_lowpass_of_exactly_half_rounds_away_from_zero() {
    let (source, strong) = half_lowpass_planes();
    let plane = QedPlane {
        source: &source,
        normal: &source,
        strong: &strong,
        width: 16,
        height: 16,
        lowpass_width: 16,
        lowpass_height: 16,
    };

    let found = on_device(plane, MatrixSamples::U8);

    for (index, value) in found.iter().enumerate() {
        let column = index as u32 % 16;
        let row = index as u32 / 16;

        // Every low-pass is exactly 128.5, which rounds to 129, one above neutral.
        let expected = if is_border_sample(column, row) {
            100.0
        } else {
            99.0
        };
        assert_eq!(*value, expected, "{column},{row}");
    }
}

#[test]
fn writes_stay_inside_the_plane() {
    let samples = MatrixSamples::U8;
    let frame = Passes::blocky(70, 46, samples, 13);
    let plane = frame.plane();
    let expected = reference_qed_plane(plane, samples);
    let mut run = task(plane, samples);
    run.guard_bytes = 64;

    let written = backend().build(run);

    let (inside, past) = written.split_at(70 * 46);
    let found = decode_samples(inside, samples);
    assert_same_plane(&found, &expected, 70, "70x46");
    assert!(
        past.iter().all(|byte| *byte == GUARD),
        "a write landed past the plane"
    );
}

#[test]
fn samples_above_the_peak_stay_inside_the_change_table() {
    let samples = MatrixSamples::U16 { bits: 10 };
    let twelve_bit = MatrixSamples::U16 { bits: 12 };
    let source = blocky_plane(64, 48, twelve_bit, 14);
    let strong = vec![0.0; 64 * 48];
    let plane = QedPlane {
        source: &source,
        normal: &source,
        strong: &strong,
        width: 64,
        height: 48,
        lowpass_width: 64,
        lowpass_height: 48,
    };
    let expected = reference_qed_plane(plane, samples);

    let found = on_device(plane, samples);

    assert_same_plane(&found, &expected, 64, "10-bit samples up to 4095");
}

#[test]
#[should_panic(expected = "the change table was made for U8 samples, not U16 { bits: 10 }")]
fn a_change_table_for_other_samples_is_refused() {
    let samples = MatrixSamples::U16 { bits: 10 };
    let frame = Passes::blocky(16, 16, samples, 15);
    let mut run = task(frame.plane(), samples);
    run.table_samples = MatrixSamples::U8;

    backend().build(run);
}

#[test]
#[should_panic(expected = "the strong plane is 15x16, expected 16x16")]
fn a_strong_plane_of_another_size_is_refused() {
    let samples = MatrixSamples::U8;
    let frame = Passes::blocky(16, 16, samples, 16);
    let mut run = task(frame.plane(), samples);
    run.strong_size = (15, 16);

    backend().build(run);
}

#[test]
#[should_panic(expected = "a 16x16 plane does not fit a 16x8 low-pass")]
fn a_lowpass_size_smaller_than_the_plane_is_refused() {
    let samples = MatrixSamples::U8;
    let frame = Passes::blocky(16, 16, samples, 17);
    let mut run = task(frame.plane(), samples);
    run.lowpass_size = (16, 8);

    backend().build(run);
}
