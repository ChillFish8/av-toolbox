use av_toolbox_cubek::deinterlace::qtgmc::clamped_difference;
use av_toolbox_cubek::testing::{assert_plane_within, reference_clamped_difference};
use av_toolbox_gpu::{BuildContext, BuildTask};

use crate::support::{Sizing, backend, empty, plane, read_f32, upload_f32};

/// `first − second` on the device, as `clamped_difference` stores it.
struct DifferenceRun {
    first: Vec<f32>,
    second: Vec<f32>,
    bits: u32,
}

impl BuildTask for DifferenceRun {
    type Output = Vec<f32>;

    fn run(self, build: &BuildContext) -> Self::Output {
        let client = build.client();
        let sizing = Sizing(build.frame_context());
        let width = self.first.len() as u32;

        let first = upload_f32(client, &self.first);
        let second = upload_f32(client, &self.second);
        let output = empty(client, self.first.len() * size_of::<f32>());
        clamped_difference(
            client,
            &sizing,
            plane(&first, width, 1),
            plane(&second, width, 1),
            plane(&output, width, 1),
            self.bits,
        );

        read_f32(client, &output)
    }
}

#[test]
fn clamped_difference_matches_make_diff() {
    for bits in [8u32, 10] {
        let peak = ((1u32 << bits) - 1) as f32;
        let first = vec![0.0, peak, 100.0, 10.0, peak, 0.0, 50.0, peak / 2.0, 3.0];
        let second = vec![peak, 0.0, 100.0, 90.0, peak - 5.0, 7.0, 49.0, 0.0, 200.0];
        let expected = reference_clamped_difference(&first, &second, bits);

        let task = DifferenceRun { first, second, bits };
        let found = backend().build(task);

        let lowest = (peak + 1.0) / -2.0;
        let highest = (peak - 1.0) / 2.0;
        assert_eq!(found[0], lowest, "{bits} bits, below the clamp");
        assert_eq!(found[1], highest, "{bits} bits, above the clamp");
        assert_eq!(found[2], 0.0, "{bits} bits, no difference");
        assert_eq!(
            found[3], -80.0,
            "{bits} bits, a negative difference inside the clamp"
        );
        assert_eq!(
            found[4], 5.0,
            "{bits} bits, a positive difference inside the clamp"
        );
        assert_plane_within(&found, &expected, 0.0, found.len() as u32, "clamped difference");
    }
}
