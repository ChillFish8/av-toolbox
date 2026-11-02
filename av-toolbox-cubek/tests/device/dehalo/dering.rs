use av_toolbox_cubek::dehalo::dering::{
    ContraPlanes,
    LimitPlanes,
    RingPlanes,
    contra_limit,
    contra_limit_sample,
    limit_filter,
    limit_sample,
    ring_mask,
    ring_mask_sample,
};
use av_toolbox_gpu::{BuildContext, BuildTask};

use super::{ranged_plane, signed_planes};
use crate::neighbourhood::busy;
use crate::support::{Sizing, assert_close, backend, empty, plane, read_f32, upload_f32};

struct ContraLimit {
    smoothed: Vec<f32>,
    sharpened: Vec<f32>,
    repaired: Vec<f32>,
    width: u32,
    height: u32,
}

impl BuildTask for ContraLimit {
    type Output = Vec<f32>;

    fn run(self, build: &BuildContext) -> Self::Output {
        let client = build.client();
        let frame_context = build.frame_context();
        let sizing = Sizing(frame_context);
        let bytes = self.smoothed.len() * size_of::<f32>();

        let smoothed = upload_f32(client, &self.smoothed);
        let sharpened = upload_f32(client, &self.sharpened);
        let repaired = upload_f32(client, &self.repaired);
        let output = empty(client, bytes);
        let planes = ContraPlanes {
            smoothed: plane(&smoothed, self.width, self.height),
            sharpened: plane(&sharpened, self.width, self.height),
            repaired: plane(&repaired, self.width, self.height),
            output: plane(&output, self.width, self.height),
        };

        contra_limit(client, &sizing, planes);

        read_f32(client, &output)
    }
}

#[test]
fn the_contra_limit_matches_the_reference() {
    let width = 37;
    let height = 23;
    let smoothed = busy(width, height);
    let (sharpened, repaired) = signed_planes(width, height);
    let task = ContraLimit {
        smoothed: smoothed.clone(),
        sharpened: sharpened.clone(),
        repaired: repaired.clone(),
        width,
        height,
    };

    let actual = backend().build(task);
    let expected: Vec<f32> = (0..(width * height) as usize)
        .map(|index| contra_limit_sample(smoothed[index], sharpened[index], repaired[index]))
        .collect();

    assert_close(&actual, &expected, 1e-3);

    let picks_sharpened = (0..(width * height) as usize)
        .filter(|index| repaired[*index].abs() > sharpened[*index].abs())
        .count();
    let picks_repaired = (0..(width * height) as usize)
        .filter(|index| repaired[*index].abs() <= sharpened[*index].abs())
        .count();

    assert!(picks_sharpened > 0, "no sample took the sharpened branch");
    assert!(picks_repaired > 0, "no sample took the repaired branch");
}

struct LimitFilter {
    plane: Vec<f32>,
    filtered: Vec<f32>,
    width: u32,
    height: u32,
    thr: f32,
    darkthr: f32,
    elast: f32,
}

impl BuildTask for LimitFilter {
    type Output = Vec<f32>;

    fn run(self, build: &BuildContext) -> Self::Output {
        let client = build.client();
        let frame_context = build.frame_context();
        let sizing = Sizing(frame_context);
        let bytes = self.plane.len() * size_of::<f32>();

        let plane_handle = upload_f32(client, &self.plane);
        let filtered = upload_f32(client, &self.filtered);
        let output = empty(client, bytes);
        let planes = LimitPlanes {
            plane: plane(&plane_handle, self.width, self.height),
            filtered: plane(&filtered, self.width, self.height),
            output: plane(&output, self.width, self.height),
        };

        limit_filter(client, &sizing, planes, self.thr, self.darkthr, self.elast);

        read_f32(client, &output)
    }
}

/// Which of `limit_sample`'s branches a sample takes, and the sign of its change.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum LimitBranch {
    BrightensPassed,
    BrightensBlended,
    BrightensFellBack,
    DarkensPassed,
    DarkensBlended,
    DarkensFellBack,
}

fn limit_branch(plane_value: f32, filtered_value: f32, thr: f32, darkthr: f32, elast: f32) -> LimitBranch {
    let difference = filtered_value - plane_value;
    let threshold = if difference > 0.0 { darkthr } else { thr };
    let size = difference.abs();
    let elastic_threshold = threshold * elast;

    if difference > 0.0 {
        if size <= threshold {
            LimitBranch::BrightensPassed
        } else if size >= elastic_threshold {
            LimitBranch::BrightensFellBack
        } else {
            LimitBranch::BrightensBlended
        }
    } else if size <= threshold {
        LimitBranch::DarkensPassed
    } else if size >= elastic_threshold {
        LimitBranch::DarkensFellBack
    } else {
        LimitBranch::DarkensBlended
    }
}

#[test]
fn the_limit_filter_matches_the_reference() {
    let width = 37;
    let height = 23;
    let thr = 20.0;
    let darkthr = 6.0;
    let elast = 2.0;
    let plane_values = busy(width, height);
    // Transposed and shifted so it differs from `plane_values` at every sample, spanning both
    // brighter and darker changes of every size.
    let filtered_values = ranged_plane(width, height, true, 73.0, 256.0, 1.0);
    let task = LimitFilter {
        plane: plane_values.clone(),
        filtered: filtered_values.clone(),
        width,
        height,
        thr,
        darkthr,
        elast,
    };

    let actual = backend().build(task);
    let expected: Vec<f32> = (0..(width * height) as usize)
        .map(|index| limit_sample(plane_values[index], filtered_values[index], thr, darkthr, elast))
        .collect();

    assert_close(&actual, &expected, 1e-3);

    let branches: Vec<LimitBranch> = (0..(width * height) as usize)
        .map(|index| limit_branch(plane_values[index], filtered_values[index], thr, darkthr, elast))
        .collect();

    for branch in [
        LimitBranch::BrightensPassed,
        LimitBranch::BrightensBlended,
        LimitBranch::BrightensFellBack,
        LimitBranch::DarkensPassed,
        LimitBranch::DarkensBlended,
        LimitBranch::DarkensFellBack,
    ] {
        let count = branches.iter().filter(|value| **value == branch).count();

        assert!(count > 0, "no sample took the {branch:?} branch");
    }
}

struct RingMask {
    outer: Vec<f32>,
    inner: Vec<f32>,
    width: u32,
    height: u32,
    incedge: bool,
    peak: f32,
}

impl BuildTask for RingMask {
    type Output = Vec<f32>;

    fn run(self, build: &BuildContext) -> Self::Output {
        let client = build.client();
        let frame_context = build.frame_context();
        let sizing = Sizing(frame_context);
        let bytes = self.outer.len() * size_of::<f32>();

        let outer = upload_f32(client, &self.outer);
        let inner = upload_f32(client, &self.inner);
        let output = empty(client, bytes);
        let planes = RingPlanes {
            outer: plane(&outer, self.width, self.height),
            inner: plane(&inner, self.width, self.height),
            output: plane(&output, self.width, self.height),
        };

        ring_mask(client, &sizing, planes, self.incedge, self.peak);

        read_f32(client, &output)
    }
}

#[test]
fn the_ring_mask_matches_the_reference_with_incedge_on_and_off() {
    let width = 37;
    let height = 23;
    let peak = 255.0;
    let outer = busy(width, height);
    let inner = ranged_plane(width, height, true, 41.0, 256.0, 1.0);
    let mut outputs = Vec::new();

    for incedge in [false, true] {
        let task = RingMask {
            outer: outer.clone(),
            inner: inner.clone(),
            width,
            height,
            incedge,
            peak,
        };

        let actual = backend().build(task);
        let expected: Vec<f32> = (0..(width * height) as usize)
            .map(|index| ring_mask_sample(outer[index], inner[index], incedge, peak))
            .collect();

        assert_close(&actual, &expected, 1e-3);

        outputs.push(actual);
    }

    assert_ne!(outputs[0], outputs[1], "incedge on and off gave the same output");
}
