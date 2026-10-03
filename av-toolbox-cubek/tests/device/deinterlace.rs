use av_toolbox_cubek::deinterlace::{BwdifField, BwdifPlanes, bwdif_field};
use av_toolbox_cubek::testing::{BWDIF_FIXTURES, BwdifReferencePlanes, assert_plane_within, reference_bwdif};
use av_toolbox_gpu::{BuildContext, BuildTask};
use cubecl::prelude::Runtime;

use crate::support::{Sizing, backend, empty, plane, read_f32, upload_f32};

/// The furthest a device sample may sit from the CPU, for fused multiply-adds.
const TOLERANCE: f32 = 1e-3;

/// One bwdif field pass over planes uploaded from the host.
struct FieldPass {
    previous: Vec<f32>,
    current: Vec<f32>,
    next: Vec<f32>,
    spatial: Option<Vec<f32>>,
    width: u32,
    height: u32,
    field: BwdifField,
}

impl BuildTask for FieldPass {
    type Output = Vec<f32>;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let frame_context = build.frame_context();
        let sizing = Sizing(frame_context);
        let bytes = self.current.len() * size_of::<f32>();

        let previous = upload_f32(client, &self.previous);
        let current = upload_f32(client, &self.current);
        let next = upload_f32(client, &self.next);
        let spatial = self.spatial.as_deref().map(|values| upload_f32(client, values));
        let output = empty(client, bytes);

        let planes = BwdifPlanes {
            previous: plane(&previous, self.width, self.height),
            current: plane(&current, self.width, self.height),
            next: plane(&next, self.width, self.height),
            spatial: spatial
                .as_ref()
                .map(|handle| plane(handle, self.width, self.height)),
            output: plane(&output, self.width, self.height),
        };
        bwdif_field(client, &sizing, planes, self.field);

        cubecl::future::block_on(client.sync()).expect("the device should finish");

        read_f32(client, &output)
    }
}

#[test]
fn every_fixture_matches_on_the_device() {
    for fixture in &BWDIF_FIXTURES {
        let task = FieldPass {
            previous: fixture.previous_plane(),
            current: fixture.current_plane(),
            next: fixture.next_plane(),
            spatial: fixture.spatial_plane(),
            width: fixture.setting("width"),
            height: fixture.setting("height"),
            field: fixture.field(),
        };
        let width = task.width;

        let found = backend().build(task);
        let expected = fixture.output_plane();

        assert_plane_within(&found, &expected, TOLERANCE, width, fixture.name);
    }
}

/// A deterministic hash of a sample's position, for noise.
fn scramble(column: u32, row: u32, frame: u32) -> u32 {
    let mut state =
        column.wrapping_mul(0x9e37_79b1) ^ row.wrapping_mul(0x85eb_ca77) ^ frame.wrapping_mul(0xc2b2_ae3d);
    state ^= state >> 15;
    state = state.wrapping_mul(0x2c1b_3c6d);
    state ^= state >> 12;

    state
}

/// A combed 10-bit plane with noise, and a bar that moves 3 samples right each frame and wraps every 64
/// columns.
fn moving_plane(width: u32, height: u32, frame: u32) -> Vec<f32> {
    let mut plane = Vec::with_capacity((width * height) as usize);
    for row in 0..height {
        for column in 0..width {
            let inside = (column + 64 - frame * 3) % 64 < 12;
            let base = if inside && row % 2 == 1 { 900 } else { 200 };
            let noise = scramble(column, row, frame) % 33;
            plane.push((base + noise) as f32);
        }
    }

    plane
}

#[test]
fn a_wide_moving_plane_matches_the_reference_for_every_field() {
    let width = 1920;
    let height = 12;
    let previous = moving_plane(width, height, 0);
    let current = moving_plane(width, height, 1);
    let next = moving_plane(width, height, 2);
    let reference_planes = BwdifReferencePlanes {
        previous: &previous,
        current: &current,
        next: &next,
        spatial: None,
    };

    for rebuilt in 0..2 {
        for first in [true, false] {
            let field = BwdifField { rebuilt, first };
            let task = FieldPass {
                previous: previous.clone(),
                current: current.clone(),
                next: next.clone(),
                spatial: None,
                width,
                height,
                field,
            };

            let found = backend().build(task);
            let expected = reference_bwdif(&reference_planes, width, height, field);

            let label = format!("rebuilt {rebuilt}, first {first}");
            assert_plane_within(&found, &expected, TOLERANCE, width, &label);
        }
    }
}

#[test]
#[should_panic(expected = "bwdif needs at least 4 rows, got 2")]
fn a_plane_of_two_rows_is_refused() {
    let task = FieldPass {
        previous: vec![0.0; 8],
        current: vec![0.0; 8],
        next: vec![0.0; 8],
        spatial: None,
        width: 4,
        height: 2,
        field: BwdifField {
            rebuilt: 1,
            first: true,
        },
    };

    backend().build(task);
}
