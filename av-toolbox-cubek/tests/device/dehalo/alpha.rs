use av_toolbox_cubek::dehalo::alpha::{
    AlphaParams,
    ClampPlanes,
    HaloPlanes,
    StrengthPlanes,
    apply_strength,
    blend_halo_mask,
    clamp_between,
    halo_mask_sample,
    strength_sample,
};
use av_toolbox_gpu::{BuildContext, BuildTask};

use super::{pairwise_max, pairwise_min, ranged_plane, share_unsaturated, six_planes};
use crate::neighbourhood::busy;
use crate::support::{Sizing, assert_close, backend, empty, plane, read_f32, upload_f32};

fn defaults(bits: u32) -> AlphaParams {
    AlphaParams {
        rx: 2.0,
        ry: 2.0,
        darkstr: 1.0,
        brightstr: 1.0,
        lowsens: 50.0,
        highsens: 50.0,
        ss: 1.5,
        bits,
    }
}

struct HaloMask {
    clip: Vec<f32>,
    halos: Vec<f32>,
    clip_max: Vec<f32>,
    clip_min: Vec<f32>,
    halo_max: Vec<f32>,
    halo_min: Vec<f32>,
    width: u32,
    height: u32,
    params: AlphaParams,
}

impl BuildTask for HaloMask {
    type Output = Vec<f32>;

    fn run(self, build: &BuildContext) -> Self::Output {
        let client = build.client();
        let frame_context = build.frame_context();
        let sizing = Sizing(frame_context);
        let bytes = self.clip.len() * size_of::<f32>();

        let clip = upload_f32(client, &self.clip);
        let halos = upload_f32(client, &self.halos);
        let clip_max = upload_f32(client, &self.clip_max);
        let clip_min = upload_f32(client, &self.clip_min);
        let halo_max = upload_f32(client, &self.halo_max);
        let halo_min = upload_f32(client, &self.halo_min);
        let output = empty(client, bytes);
        let planes = HaloPlanes {
            clip: plane(&clip, self.width, self.height),
            halos: plane(&halos, self.width, self.height),
            clip_max: plane(&clip_max, self.width, self.height),
            clip_min: plane(&clip_min, self.width, self.height),
            halo_max: plane(&halo_max, self.width, self.height),
            halo_min: plane(&halo_min, self.width, self.height),
            output: plane(&output, self.width, self.height),
        };

        blend_halo_mask(client, &sizing, planes, &self.params);

        read_f32(client, &output)
    }
}

#[test]
fn the_halo_mask_matches_the_reference() {
    let width = 37;
    let height = 23;

    for bits in [8, 10] {
        let params = defaults(bits);
        let scale = params.scale(1.0);
        let peak = params.peak();

        // Each range comes from the max and min of two differently shaped planes, so it holds a
        // real spread of small values rather than one constant gap.
        let clip_a = ranged_plane(width, height, false, 0.0, 16.0, scale);
        let clip_b = ranged_plane(width, height, true, 3.0, 16.0, scale);
        let halo_a = ranged_plane(width, height, false, 5.0, 16.0, scale);
        let halo_b = ranged_plane(width, height, true, 9.0, 16.0, scale);

        let clip_max = pairwise_max(&clip_a, &clip_b);
        let clip_min = pairwise_min(&clip_a, &clip_b);
        let halo_max = pairwise_max(&halo_a, &halo_b);
        let halo_min = pairwise_min(&halo_a, &halo_b);

        let clip: Vec<f32> = busy(width, height).iter().map(|value| value * scale).collect();
        let halos: Vec<f32> = clip.iter().map(|value| (value + 40.0 * scale) % peak).collect();

        let task = HaloMask {
            clip: clip.clone(),
            halos: halos.clone(),
            clip_max: clip_max.clone(),
            clip_min: clip_min.clone(),
            halo_max: halo_max.clone(),
            halo_min: halo_min.clone(),
            width,
            height,
            params,
        };

        let actual = backend().build(task);
        let expected: Vec<f32> = (0..(width * height) as usize)
            .map(|index| {
                let clip_range = clip_max[index] - clip_min[index];
                let halo_range = halo_max[index] - halo_min[index];
                halo_mask_sample(clip[index], halos[index], clip_range, halo_range, &params)
            })
            .collect();

        assert_close(&actual, &expected, 1e-3);

        let share = share_unsaturated(&expected, peak);
        assert!(
            share > 0.1,
            "bits {bits}: only {share} of the mask sat strictly between 0 and peak"
        );
    }
}

struct Clamp {
    clip: Vec<f32>,
    highest: Vec<f32>,
    lowest: Vec<f32>,
    width: u32,
    height: u32,
}

impl BuildTask for Clamp {
    type Output = Vec<f32>;

    fn run(self, build: &BuildContext) -> Self::Output {
        let client = build.client();
        let frame_context = build.frame_context();
        let sizing = Sizing(frame_context);
        let bytes = self.clip.len() * size_of::<f32>();

        let clip = upload_f32(client, &self.clip);
        let highest = upload_f32(client, &self.highest);
        let lowest = upload_f32(client, &self.lowest);
        let output = empty(client, bytes);
        let planes = ClampPlanes {
            clip: plane(&clip, self.width, self.height),
            highest: plane(&highest, self.width, self.height),
            lowest: plane(&lowest, self.width, self.height),
            output: plane(&output, self.width, self.height),
        };

        clamp_between(client, &sizing, planes);

        read_f32(client, &output)
    }
}

#[test]
fn the_clamp_matches_the_reference() {
    let width = 37;
    let height = 23;
    let planes = six_planes(width, height);
    let task = Clamp {
        clip: planes[0].clone(),
        highest: planes[1].clone(),
        lowest: planes[2].clone(),
        width,
        height,
    };

    let actual = backend().build(task);
    let expected: Vec<f32> = (0..(width * height) as usize)
        .map(|index| planes[0][index].min(planes[1][index]).max(planes[2][index]))
        .collect();

    assert_eq!(actual, expected);
}

struct Strength {
    clip: Vec<f32>,
    remove: Vec<f32>,
    width: u32,
    height: u32,
    params: AlphaParams,
}

impl BuildTask for Strength {
    type Output = Vec<f32>;

    fn run(self, build: &BuildContext) -> Self::Output {
        let client = build.client();
        let frame_context = build.frame_context();
        let sizing = Sizing(frame_context);
        let bytes = self.clip.len() * size_of::<f32>();

        let clip = upload_f32(client, &self.clip);
        let remove = upload_f32(client, &self.remove);
        let output = empty(client, bytes);
        let planes = StrengthPlanes {
            clip: plane(&clip, self.width, self.height),
            remove: plane(&remove, self.width, self.height),
            output: plane(&output, self.width, self.height),
        };

        apply_strength(client, &sizing, planes, &self.params);

        read_f32(client, &output)
    }
}

fn run_strength_case(clip: &[f32], remove: &[f32], width: u32, height: u32, params: &AlphaParams) {
    let task = Strength {
        clip: clip.to_vec(),
        remove: remove.to_vec(),
        width,
        height,
        params: *params,
    };

    let actual = backend().build(task);
    let expected: Vec<f32> = (0..(width * height) as usize)
        .map(|index| strength_sample(clip[index], remove[index], params))
        .collect();

    assert_close(&actual, &expected, 1e-3);
}

#[test]
fn the_strength_matches_the_reference() {
    let width = 37;
    let height = 23;
    let planes = six_planes(width, height);
    let moderate = AlphaParams {
        darkstr: 0.5,
        brightstr: 0.8,
        ..defaults(8)
    };

    run_strength_case(&planes[0], &planes[1], width, height, &moderate);

    // Strengths above 1 push the pulled value past 0 or the peak, so this exercises the clamp.
    let strong = AlphaParams {
        darkstr: 1.5,
        brightstr: 1.5,
        ..defaults(8)
    };
    let near_zero_clip = vec![5.0; 16];
    let near_zero_remove = vec![200.0; 16];
    let near_peak_clip = vec![250.0; 16];
    let near_peak_remove = vec![50.0; 16];

    run_strength_case(&near_zero_clip, &near_zero_remove, 4, 4, &strong);
    run_strength_case(&near_peak_clip, &near_peak_remove, 4, 4, &strong);
}
