use av_toolbox_cubek::deinterlace::qtgmc::{GaussTables, blur_and_spread};
use av_toolbox_cubek::testing::{assert_plane_within, reference_blur_and_spread};
use av_toolbox_gpu::{BuildContext, BuildTask};
use cubecl::prelude::Runtime;

use crate::support::{Sizing, backend, empty, plane, read_f32, seeded_plane, upload_f32};

/// How far the fused blur may sit from the chain it replaces, on data between 0 and 1.
const CHAIN_TOLERANCE: f32 = 1e-6;

/// The sizes the short line test crosses, each as a width and as a height.
const SHORT_LENGTHS: [u32; 5] = [1, 3, 5, 8, 9];

/// One plane to blur, with the sharpness of its gaussian.
struct SpreadCase {
    width: u32,
    height: u32,
    sharpness: f32,
}

/// Planes blurred on the device by [blur_and_spread], one per case.
struct SpreadRun {
    cases: Vec<SpreadCase>,
}

impl BuildTask for SpreadRun {
    type Output = Vec<Vec<f32>>;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let sizing = Sizing(build.frame_context());

        let mut results = Vec::with_capacity(self.cases.len());
        for case in &self.cases {
            let size = (case.width, case.height);
            let gauss = GaussTables::new(client, size, case.sharpness);
            let values = seeded_plane(case.width, case.height);
            let input = upload_f32(client, &values);
            let output = empty(client, values.len() * size_of::<f32>());

            let input_plane = plane(&input, case.width, case.height);
            let output_plane = plane(&output, case.width, case.height);
            blur_and_spread(client, &sizing, &gauss, input_plane, output_plane);

            cubecl::future::block_on(client.sync()).expect("the device should finish");

            results.push(read_f32(client, &output));
        }

        results
    }
}

/// Blurs every case on the device and checks each against the chain on the CPU.
fn assert_matches_the_chain(cases: Vec<SpreadCase>) {
    let labels: Vec<_> = cases
        .iter()
        .map(|case| (case.width, case.height, case.sharpness))
        .collect();

    let task = SpreadRun { cases };
    let outputs = backend().build(task);

    for ((width, height, sharpness), found) in labels.into_iter().zip(outputs) {
        let input = seeded_plane(width, height);
        let expected = reference_blur_and_spread(&input, width as usize, height as usize, sharpness);
        let label = format!("{width}x{height} p {sharpness}");

        assert_plane_within(&found, &expected, CHAIN_TOLERANCE, width, &label);
    }
}

#[test]
fn the_fused_blur_matches_the_chain() {
    let cases = [2.0, 5.0]
        .into_iter()
        .map(|sharpness| SpreadCase {
            width: 1921,
            height: 1081,
            sharpness,
        })
        .collect();

    assert_matches_the_chain(cases);
}

#[test]
fn the_fused_blur_matches_on_short_lines() {
    let mut cases = Vec::new();
    for width in SHORT_LENGTHS {
        for height in SHORT_LENGTHS {
            for sharpness in [2.0, 5.0] {
                cases.push(SpreadCase {
                    width,
                    height,
                    sharpness,
                });
            }
        }
    }

    assert_matches_the_chain(cases);
}

// Wide enough that each cube across walks more than one tile.
#[test]
fn the_fused_blur_matches_when_a_plane_walks_the_grid() {
    let cases = vec![SpreadCase {
        width: 65535 * 32 + 37,
        height: 3,
        sharpness: 2.0,
    }];

    assert_matches_the_chain(cases);
}
