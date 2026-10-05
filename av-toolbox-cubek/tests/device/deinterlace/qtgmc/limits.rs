use av_toolbox_cubek::deinterlace::qtgmc::temporal_limits;
use av_toolbox_cubek::testing::{assert_plane_within, reference_temporal_limits};
use av_toolbox_gpu::{BuildContext, BuildTask};
use cubecl::prelude::Runtime;

use crate::support::{Sizing, backend, empty, plane, read_f32, upload_f32};

/// The limits of a centre row and its neighbours on the device.
struct LimitsRun {
    centre: Vec<f32>,
    neighbours: Vec<Vec<f32>>,
}

impl BuildTask for LimitsRun {
    type Output = (Vec<f32>, Vec<f32>);

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let sizing = Sizing(build.frame_context());
        let width = self.centre.len() as u32;
        let bytes = self.centre.len() * size_of::<f32>();

        let centre = upload_f32(client, &self.centre);
        let handles: Vec<_> = self
            .neighbours
            .iter()
            .map(|neighbour| upload_f32(client, neighbour))
            .collect();
        let neighbours: Vec<_> = handles.iter().map(|handle| plane(handle, width, 1)).collect();
        let upper = empty(client, bytes);
        let lower = empty(client, bytes);
        temporal_limits(
            client,
            &sizing,
            plane(&centre, width, 1),
            &neighbours,
            plane(&upper, width, 1),
            plane(&lower, width, 1),
        );

        (read_f32(client, &upper), read_f32(client, &lower))
    }
}

#[test]
fn temporal_limits_bracket_every_neighbour() {
    let centre = vec![5.0, 5.0, 5.0, 5.0, 0.0, 255.0];
    let before = vec![1.0, 9.0, 5.0, 7.0, 255.0, 0.0];
    let after = vec![3.0, 2.0, 5.0, 6.0, 128.0, 128.0];

    for count in 0..=2 {
        let neighbours: Vec<Vec<f32>> = [before.clone(), after.clone()].into_iter().take(count).collect();
        let views: Vec<&[f32]> = neighbours.iter().map(Vec::as_slice).collect();
        let (expected_upper, expected_lower) = reference_temporal_limits(&centre, &views);

        let task = LimitsRun {
            centre: centre.clone(),
            neighbours: neighbours.clone(),
        };
        let (upper, lower) = backend().build(task);

        let width = centre.len() as u32;
        assert_plane_within(&upper, &expected_upper, 0.0, width, "upper");
        assert_plane_within(&lower, &expected_lower, 0.0, width, "lower");

        for (position, sample) in centre.iter().enumerate() {
            assert!(lower[position] <= *sample && *sample <= upper[position]);
            for neighbour in &neighbours {
                assert!(lower[position] <= neighbour[position] && neighbour[position] <= upper[position]);
            }
        }

        if count == 0 {
            assert_eq!(upper, centre, "no neighbours leave the upper limit at the centre");
            assert_eq!(lower, centre, "no neighbours leave the lower limit at the centre");
        }
    }
}
