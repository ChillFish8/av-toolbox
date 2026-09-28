use av_toolbox_cubek::dehalo::DifferencePlanes;
use av_toolbox_cubek::dehalo::edge_clean::{
    levels,
    levels_sample,
    offset_difference,
    threshold_mask,
    threshold_sample,
};
use av_toolbox_cubek::dehalo::fine::RampPlanes;
use av_toolbox_gpu::{BuildContext, BuildTask};
use cubecl::prelude::Runtime;

use super::signed_planes;
use crate::neighbourhood::busy;
use crate::support::{Sizing, assert_close, backend, empty, plane, read_f32, upload_f32};

struct ThresholdMask {
    input: Vec<f32>,
    width: u32,
    height: u32,
    low: f32,
    high: f32,
    peak: f32,
    invert: bool,
}

impl BuildTask for ThresholdMask {
    type Output = Vec<f32>;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let frame_context = build.frame_context();
        let sizing = Sizing(frame_context);
        let bytes = self.input.len() * size_of::<f32>();

        let input = upload_f32(client, &self.input);
        let output = empty(client, bytes);
        let planes = RampPlanes {
            input: plane(&input, self.width, self.height),
            output: plane(&output, self.width, self.height),
        };

        threshold_mask(
            client,
            &sizing,
            planes,
            self.low,
            self.high,
            self.peak,
            self.invert,
        );

        read_f32(client, &output)
    }
}

#[test]
fn the_threshold_mask_matches_the_reference() {
    let width = 37;
    let height = 23;
    let peak = 255.0;
    let low = 80.0;
    let high = 160.0;
    // `busy` spans 0..256, so this crosses below `low`, between `low` and `high`, and above `high`.
    let input = busy(width, height);

    for invert in [false, true] {
        let task = ThresholdMask {
            input: input.clone(),
            width,
            height,
            low,
            high,
            peak,
            invert,
        };

        let actual = backend().build(task);
        let expected: Vec<f32> = input
            .iter()
            .map(|value| threshold_sample(*value, low, high, peak, invert))
            .collect();

        assert_close(&actual, &expected, 1e-3);

        let below = input.iter().filter(|value| **value < low).count();
        let between = input
            .iter()
            .filter(|value| **value >= low && **value <= high)
            .count();
        let above = input.iter().filter(|value| **value > high).count();
        assert!(below > 0, "invert {invert}: no values below low");
        assert!(between > 0, "invert {invert}: no values between low and high");
        assert!(above > 0, "invert {invert}: no values above high");
    }
}

struct Levels {
    input: Vec<f32>,
    width: u32,
    height: u32,
    low: f32,
    span: f32,
    exponent: f32,
    peak: f32,
}

impl BuildTask for Levels {
    type Output = Vec<f32>;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let frame_context = build.frame_context();
        let sizing = Sizing(frame_context);
        let bytes = self.input.len() * size_of::<f32>();

        let input = upload_f32(client, &self.input);
        let output = empty(client, bytes);
        let planes = RampPlanes {
            input: plane(&input, self.width, self.height),
            output: plane(&output, self.width, self.height),
        };

        levels(
            client,
            &sizing,
            planes,
            self.low,
            self.span,
            self.exponent,
            self.peak,
        );

        read_f32(client, &output)
    }
}

#[test]
fn the_levels_match_the_reference() {
    let width = 37;
    let height = 23;
    let peak = 255.0;
    let low = 40.0;
    let span = 128.0;
    let exponent = 1.0 / 0.35;
    // `busy` spans 0..256, so this crosses below the window, inside it, and above it (the window
    // is `low..low + span`, so `40..168`).
    let input = busy(width, height);
    let task = Levels {
        input: input.clone(),
        width,
        height,
        low,
        span,
        exponent,
        peak,
    };

    let actual = backend().build(task);
    let expected: Vec<f32> = input
        .iter()
        .map(|value| levels_sample(*value, low, span, exponent, peak))
        .collect();

    assert_close(&actual, &expected, 1e-3);

    let window_high = low + span;
    let below = input.iter().filter(|value| **value < low).count();
    let inside = input
        .iter()
        .filter(|value| **value >= low && **value <= window_high)
        .count();
    let above = input.iter().filter(|value| **value > window_high).count();
    assert!(below > 0, "no values below the window");
    assert!(inside > 0, "no values inside the window");
    assert!(above > 0, "no values above the window");
}

struct OffsetDifference {
    first: Vec<f32>,
    second: Vec<f32>,
    width: u32,
    height: u32,
    offset: f32,
}

impl BuildTask for OffsetDifference {
    type Output = Vec<f32>;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let frame_context = build.frame_context();
        let sizing = Sizing(frame_context);
        let bytes = self.first.len() * size_of::<f32>();

        let first = upload_f32(client, &self.first);
        let second = upload_f32(client, &self.second);
        let output = empty(client, bytes);
        let planes = DifferencePlanes {
            first: plane(&first, self.width, self.height),
            second: plane(&second, self.width, self.height),
            output: plane(&output, self.width, self.height),
        };

        offset_difference(client, &sizing, planes, self.offset);

        read_f32(client, &output)
    }
}

#[test]
fn the_offset_difference_matches_the_reference() {
    let width = 37;
    let height = 23;
    let offset = 16.0;
    let (first, second) = signed_planes(width, height);
    let task = OffsetDifference {
        first: first.clone(),
        second: second.clone(),
        width,
        height,
        offset,
    };

    let actual = backend().build(task);
    let expected: Vec<f32> = first
        .iter()
        .zip(&second)
        .map(|(value, other)| value - other + offset)
        .collect();

    assert_close(&actual, &expected, 1e-3);

    let change = first.iter().zip(&second).map(|(value, other)| value - other);
    let negative = change.clone().filter(|value| *value < 0.0).count();
    let zero = change.clone().filter(|value| *value == 0.0).count();
    let positive = change.filter(|value| *value > 0.0).count();
    assert!(negative > 0, "no negative differences");
    assert!(zero > 0, "no zero differences");
    assert!(positive > 0, "no positive differences");
}
