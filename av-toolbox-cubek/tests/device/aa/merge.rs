use av_toolbox_cubek::aa::merge_fields;
use av_toolbox_gpu::{BuildContext, BuildTask};

use crate::support::{Sizing, backend, empty, plane, read_f32, upload_f32};

const FIRST: [f32; 6] = [0.0, 0.0, 1.0, 254.0, 65534.0, 65535.0];
const SECOND: [f32; 6] = [0.0, 1.0, 2.0, 255.0, 65535.0, 65535.0];
const EXPECTED: [f32; 6] = [0.0, 1.0, 2.0, 255.0, 65535.0, 65535.0];

struct Merge {
    in_place: bool,
}

impl BuildTask for Merge {
    type Output = Vec<f32>;

    fn run(self, build: &BuildContext) -> Self::Output {
        let client = build.client();
        let frame_context = build.frame_context();
        let sizing = Sizing(frame_context);
        let bytes = FIRST.len() * size_of::<f32>();

        let first = upload_f32(client, &FIRST);
        let second = upload_f32(client, &SECOND);
        let output = empty(client, bytes);
        let first_plane = plane(&first, 6, 1);
        let second_plane = plane(&second, 6, 1);
        let output_plane = plane(&output, 6, 1);

        if self.in_place {
            merge_fields(client, &sizing, first_plane, second_plane, first_plane);
            cubecl::future::block_on(client.sync()).expect("the device should finish");

            return read_f32(client, &first);
        }

        merge_fields(client, &sizing, first_plane, second_plane, output_plane);
        cubecl::future::block_on(client.sync()).expect("the device should finish");

        read_f32(client, &output)
    }
}

#[test]
fn merge_fields_rounds_half_up() {
    let merged = backend().build(Merge { in_place: false });

    assert_eq!(merged, EXPECTED);
}

#[test]
fn merge_fields_works_in_place() {
    let merged = backend().build(Merge { in_place: true });

    assert_eq!(merged, EXPECTED);
}
