use av_toolbox_cubek::dehalo::yahr::{YahrPlanes, yahr_combine};
use av_toolbox_gpu::{BuildContext, BuildTask};

use super::signed_planes;
use crate::neighbourhood::busy;
use crate::support::{Sizing, assert_close, backend, empty, plane, read_f32, upload_f32};

struct YahrCombine {
    clip: Vec<f32>,
    detail: Vec<f32>,
    limited: Vec<f32>,
    width: u32,
    height: u32,
}

impl BuildTask for YahrCombine {
    type Output = Vec<f32>;

    fn run(self, build: &BuildContext) -> Self::Output {
        let client = build.client();
        let frame_context = build.frame_context();
        let sizing = Sizing(frame_context);
        let bytes = self.clip.len() * size_of::<f32>();

        let clip = upload_f32(client, &self.clip);
        let detail = upload_f32(client, &self.detail);
        let limited = upload_f32(client, &self.limited);
        let output = empty(client, bytes);
        let planes = YahrPlanes {
            clip: plane(&clip, self.width, self.height),
            detail: plane(&detail, self.width, self.height),
            limited: plane(&limited, self.width, self.height),
            output: plane(&output, self.width, self.height),
        };

        yahr_combine(client, &sizing, planes);

        read_f32(client, &output)
    }
}

#[test]
fn the_yahr_combine_matches_the_reference() {
    let width = 37;
    let height = 23;
    let (detail, limited) = signed_planes(width, height);
    let clip = busy(width, height);
    let task = YahrCombine {
        clip: clip.clone(),
        detail: detail.clone(),
        limited: limited.clone(),
        width,
        height,
    };

    let actual = backend().build(task);
    let expected: Vec<f32> = (0..(width * height) as usize)
        .map(|index| clip[index] - (detail[index] - limited[index]))
        .collect();

    assert_close(&actual, &expected, 1e-3);

    let change = detail.iter().zip(&limited).map(|(value, other)| value - other);
    let negative = change.clone().filter(|value| *value < 0.0).count();
    let zero = change.clone().filter(|value| *value == 0.0).count();
    let positive = change.filter(|value| *value > 0.0).count();
    assert!(negative > 0, "no negative differences");
    assert!(zero > 0, "no zero differences");
    assert!(positive > 0, "no positive differences");
}
