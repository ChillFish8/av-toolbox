use av_toolbox_cubek::histogram::{BINS, PercentilePlanes, percentile};
use av_toolbox_cubek::reduce::{ReducePlanes, extremes, scratch_len};
use av_toolbox_gpu::{BuildContext, BuildTask};
use cubecl::prelude::Runtime;

use crate::support::{Sizing, backend, empty, plane, read_f32, upload_f32};

struct Percentile {
    input: Vec<f32>,
    percentile: f32,
}

impl BuildTask for Percentile {
    type Output = f32;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let frame_context = build.frame_context();
        let sizing = Sizing(frame_context);
        let samples = self.input.len();
        let width = samples as u32;
        let scratch_width = scratch_len(samples) as u32;
        let scratch_bytes = scratch_len(samples) * size_of::<f32>();

        let input = upload_f32(client, &self.input);
        let first = empty(client, scratch_bytes);
        let second = empty(client, scratch_bytes);
        let range = empty(client, 2 * size_of::<f32>());
        let bins = empty(client, BINS as usize * size_of::<u32>());
        let result = empty(client, size_of::<f32>());

        let input_plane = plane(&input, width, 1);
        let first_plane = plane(&first, scratch_width, 1);
        let second_plane = plane(&second, scratch_width, 1);
        let range_plane = plane(&range, 2, 1);
        let reduce_planes = ReducePlanes {
            input: input_plane,
            first: first_plane,
            second: second_plane,
            result: range_plane,
        };
        extremes(client, &sizing, reduce_planes);

        let input_plane = plane(&input, width, 1);
        let range_plane = plane(&range, 2, 1);
        let bins_plane = plane(&bins, BINS, 1);
        let result_plane = plane(&result, 1, 1);
        let percentile_planes = PercentilePlanes {
            input: input_plane,
            range: range_plane,
            bins: bins_plane,
            result: result_plane,
        };
        percentile(client, &sizing, percentile_planes, self.percentile);

        read_f32(client, &result)[0]
    }
}

#[test]
fn the_eighth_percentile_of_a_ramp_lands_within_a_bin() {
    let input: Vec<f32> = (0..10_000).map(|index| index as f32).collect();
    let bin_width = 9_999.0 / BINS as f32;

    let found = backend().build(Percentile {
        input,
        percentile: 8.0,
    });

    assert!((found - 800.0).abs() <= bin_width, "found {found}");
}

#[test]
fn a_flat_plane_has_its_own_value_at_every_percentile() {
    let input = vec![0.25f32; 35];

    let found = backend().build(Percentile {
        input,
        percentile: 50.0,
    });

    assert_eq!(found, 0.25);
}

#[test]
fn the_hundredth_percentile_is_near_the_largest_value() {
    let input: Vec<f32> = (0..1_000).map(|index| index as f32 / 1_000.0).collect();
    let bin_width = 0.999 / BINS as f32;

    let found = backend().build(Percentile {
        input,
        percentile: 100.0,
    });

    assert!((found - 0.999).abs() <= bin_width, "found {found}");
}

#[test]
fn a_second_run_does_not_count_the_first_runs_bins() {
    let first: Vec<f32> = (0..1_000).map(|index| index as f32).collect();
    let second = first.clone();

    let backend = backend();
    let found_first = backend.build(Percentile {
        input: first,
        percentile: 50.0,
    });
    let found_second = backend.build(Percentile {
        input: second,
        percentile: 50.0,
    });

    assert_eq!(found_first, found_second);
}
