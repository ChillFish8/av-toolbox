use super::reference::{
    IntPlane,
    reference_chroma_into_luma,
    reference_comb_mask,
    reference_compare_plane,
    reference_difference,
    reference_match_weave,
    reference_mic,
    scene_difference,
};
use super::synthetic::SyntheticClip;
use crate::ivtc::{ComparePass, FieldSums};

const WIDTH: usize = 64;
const HEIGHT: usize = 32;
const BLOCK: usize = 16;

/// The VFM settings [reference_vfm_metrics] measures a frame with.
///
/// `field` is VFM's resolved field and `pairs` the raw matches each field compare takes, `match1` first.
/// `exclude_from` and `exclude_to` are VFM's `y0` and `y1`, in luma rows. `mchroma` adds the chroma planes to
/// the field compares, and `chroma` marks chroma combing into the luma mask. The mics count 16x16 blocks.
pub struct VfmSettings {
    pub field: usize,
    pub pairs: [(usize, usize); 3],
    pub exclude_from: u32,
    pub exclude_to: u32,
    pub cthresh: i32,
    pub chroma: bool,
    pub mchroma: bool,
}

/// What VFM measures of one frame before it decides.
///
/// `sums` holds the field compare sums of each of the settings' pairs, in order. `mics` holds each match's
/// mic by raw match number. `current_scene` is the scene measure of the frame against the one before, and
/// `next_scene` the same for the next frame.
pub struct VfmMetrics {
    pub sums: [FieldSums; 3],
    pub mics: [u32; 5],
    pub current_scene: f64,
    pub next_scene: f64,
}

/// VFM's frames for frame `frame`, the previous, current and next frames clamped to the clip.
fn window(clip: &SyntheticClip, frame: usize) -> [&[Vec<i32>; 3]; 3] {
    let last = clip.frames.len() - 1;
    let previous = frame.saturating_sub(1);
    let next = usize::min(frame + 1, last);

    [&clip.frames[previous], &clip.frames[frame], &clip.frames[next]]
}

/// The planes `frames` holds of plane `plane_index`, chroma planes halved in both directions.
fn plane_window<'a>(frames: [&'a [Vec<i32>; 3]; 3], plane_index: usize) -> [IntPlane<'a>; 3] {
    let chroma = usize::from(plane_index > 0);

    frames.map(|planes| IntPlane {
        samples: &planes[plane_index],
        width: WIDTH >> chroma,
        height: HEIGHT >> chroma,
    })
}

fn plane_count(with_chroma: bool) -> usize {
    if with_chroma { 3 } else { 1 }
}

/// `compareFieldsSlow`'s sums on frame `frame` of a synthetic clip, for matches `first` and `second`.
pub fn reference_field_sums(
    clip: &SyntheticClip,
    frame: usize,
    settings: &VfmSettings,
    first: usize,
    second: usize,
) -> FieldSums {
    let frames = window(clip, frame);

    let mut sums = FieldSums::default();
    for plane_index in 0..plane_count(settings.mchroma) {
        let chroma = u32::from(plane_index > 0);
        let pass = ComparePass {
            first: first as u32,
            second: second as u32,
            field: settings.field as u32,
            exclude_from: settings.exclude_from >> chroma,
            exclude_to: settings.exclude_to >> chroma,
            start_x: 8 >> chroma,
            shift: clip.bits - 8,
        };
        let planes = plane_window(frames, plane_index);

        let (plane_sums, _) = reference_compare_plane(planes, &pass);
        sums += plane_sums;
    }

    sums
}

/// `calcMI`'s count for the weave of match `matched` on frame `frame` of a synthetic clip.
fn weave_mic(clip: &SyntheticClip, frame: usize, settings: &VfmSettings, matched: usize) -> u32 {
    let frames = window(clip, frame);
    let cthresh = if settings.cthresh >= 0 {
        settings.cthresh << (clip.bits - 8)
    } else {
        settings.cthresh
    };

    let mut masks = Vec::new();
    for plane_index in 0..plane_count(settings.chroma) {
        let [previous, current, next] = plane_window(frames, plane_index);
        let samples = [previous.samples, current.samples, next.samples];
        let woven = reference_match_weave(samples, current.width, current.height, matched, settings.field);
        let plane = IntPlane {
            samples: &woven,
            ..current
        };
        masks.push(reference_comb_mask(plane, cthresh));
    }

    let mut masks = masks.into_iter();
    let mut luma = masks.next().expect("a luma mask");
    if settings.chroma {
        let u_mask = masks.next().expect("a U mask");
        let v_mask = masks.next().expect("a V mask");
        reference_chroma_into_luma(&mut luma, WIDTH, &u_mask, &v_mask, WIDTH / 2, HEIGHT / 2, 1, 1);
    }

    reference_mic(&luma, WIDTH, HEIGHT, BLOCK, BLOCK)
}

/// The scene measure of frame `frame`'s luma against the frame before it, and of frame 0 against itself.
fn scene_measure(clip: &SyntheticClip, frame: usize) -> f64 {
    let current = &clip.frames[frame][0];
    let before = &clip.frames[frame.saturating_sub(1)][0];
    let sum = reference_difference(current, before);

    scene_difference(sum, WIDTH, HEIGHT, clip.bits)
}

/// Everything VFM measures of frame `frame` of a synthetic clip, on the CPU.
///
/// The previous and next frames are clamped to the clip, as `vfmProcess` clamps them, so the last frame's
/// next scene measure is its own.
pub fn reference_vfm_metrics(clip: &SyntheticClip, frame: usize, settings: &VfmSettings) -> VfmMetrics {
    let last = clip.frames.len() - 1;
    let next = usize::min(frame + 1, last);

    let sums = settings
        .pairs
        .map(|(first, second)| reference_field_sums(clip, frame, settings, first, second));
    let mics = std::array::from_fn(|matched| weave_mic(clip, frame, settings, matched));
    let current_scene = scene_measure(clip, frame);
    let next_scene = scene_measure(clip, next);

    VfmMetrics {
        sums,
        mics,
        current_scene,
        next_scene,
    }
}
