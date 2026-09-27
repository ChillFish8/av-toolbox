use av_toolbox_cubek::rgvs::{RepairMode, RepairPlanes, repair};
use av_toolbox_cubek::testing::reference_repair;
use av_toolbox_gpu::{BuildContext, BuildTask};
use cubecl::prelude::Runtime;

use crate::neighbourhood::busy;
use crate::support::{Sizing, backend, empty, plane, read_f32, upload_f32};

struct Repair {
    clip: Vec<f32>,
    reference: Vec<f32>,
    width: u32,
    height: u32,
}

impl BuildTask for Repair {
    type Output = Vec<f32>;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let frame_context = build.frame_context();
        let sizing = Sizing(frame_context);
        let bytes = self.clip.len() * size_of::<f32>();

        let clip = upload_f32(client, &self.clip);
        let reference = upload_f32(client, &self.reference);
        let output = empty(client, bytes);
        let clip_plane = plane(&clip, self.width, self.height);
        let reference_plane = plane(&reference, self.width, self.height);
        let output_plane = plane(&output, self.width, self.height);
        let planes = RepairPlanes {
            clip: clip_plane,
            reference: reference_plane,
            output: output_plane,
        };
        repair(client, &sizing, planes, RepairMode::One);

        read_f32(client, &output)
    }
}

#[test]
fn repair_one_matches_the_reference() {
    for (width, height) in [(37, 23), (2, 2), (1, 6)] {
        let clip = busy(width, height);
        let reference: Vec<f32> = clip.iter().map(|value| (value * 7.0) % 256.0).collect();
        let task = Repair {
            clip: clip.clone(),
            reference: reference.clone(),
            width,
            height,
        };

        let actual = backend().build(task);
        let expected = reference_repair(&clip, &reference, width, height, RepairMode::One);

        assert_eq!(actual, expected, "{width}x{height}");
    }
}
