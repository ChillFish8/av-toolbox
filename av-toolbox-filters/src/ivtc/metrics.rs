use av_toolbox_core::StageError;
use av_toolbox_cubek::DevicePlane;
use av_toolbox_cubek::ivtc::{
    ChromaMasks,
    CombPlanes,
    ComparePass,
    ComparePlanes,
    DifferencePlanes,
    FieldSums,
    MicBlocks,
    MicPlanes,
    WeavePlanes,
    chroma_into_luma,
    clear_mics,
    comb_mask,
    compare_fields,
    compare_sums_len,
    difference_sums,
    mic,
    mic_count_len,
    segments,
    weave,
};
use av_toolbox_gpu::{Client, FrameContext, FrameFormat, FrameLayout};
use cubecl::prelude::CubeElement;
use cubecl::server::ServerError;

use crate::aa::eedi2::subsampling_shifts;
use crate::ivtc::decision::{B, C, FrameMetrics, N, P, PairSums, U, is_scene_change};
use crate::ivtc::matching::STAGE;
use crate::scratch::Scratch;
use crate::sizing::Sizing;

/// How many weaves VFM builds of each frame, one per match.
const MATCHES: usize = 5;

/// The first and last columns VFM's field compare leaves out of a luma row.
const COMPARE_EDGE: u32 = 8;

fn unreadable_scene_sums(error: ServerError) -> StageError {
    let message = format!("{STAGE} could not read back its scene sums");

    StageError::with_source(message, error)
}

fn unreadable_field_sums(error: ServerError) -> StageError {
    let message = format!("{STAGE} could not read back its field compare sums");

    StageError::with_source(message, error)
}

fn unreadable_mics(error: ServerError) -> StageError {
    let message = format!("{STAGE} could not read back its mics");

    StageError::with_source(message, error)
}

/// The settings VFM measures every frame with, once checked.
///
/// `field` is VFM's field, 1 when the matching field is the top field. `cthresh` is scaled to the bit depth
/// and `scthresh` is a fraction rather than a percentage.
#[derive(Debug, Clone, Copy)]
pub(crate) struct MetricSettings {
    pub(crate) field: u32,
    pub(crate) pairs: [(usize, usize); 3],
    pub(crate) mode: u32,
    pub(crate) micmatch: u32,
    pub(crate) cthresh: i32,
    pub(crate) chroma: bool,
    pub(crate) mchroma: bool,
    pub(crate) blocks: MicBlocks,
    pub(crate) y0: u32,
    pub(crate) y1: u32,
    pub(crate) scthresh: f64,
}

/// The previous, current and next frames of one window as f32 planes, each frame's planes in channel order.
///
/// `next_is_current` is set when `next` holds the current frame's own planes.
pub(crate) struct WindowPlanes {
    pub(crate) previous: Vec<Scratch>,
    pub(crate) current: Vec<Scratch>,
    pub(crate) next: Vec<Scratch>,
    pub(crate) next_is_current: bool,
}

/// What one window measured, and the five weaves it built, by raw match number and then by plane.
pub(crate) struct Gathered {
    pub(crate) metrics: FrameMetrics,
    pub(crate) woven: Vec<Vec<Scratch>>,
}

/// Measures every frame of a stream the way VFM does before it decides.
pub(crate) struct MetricGatherer {
    client: Client,
    input: FrameFormat,
    settings: MetricSettings,
    shift: u32,
    peak: u64,
}

impl MetricGatherer {
    /// VFM's field, 1 when the matching field is the top field.
    pub(crate) fn field(&self) -> u32 {
        self.settings.field
    }

    /// A gatherer for frames of `input` at `bits` bits.
    pub(crate) fn new(client: Client, input: FrameFormat, settings: MetricSettings, bits: u32) -> Self {
        let peak = (1u64 << bits) - 1;

        Self {
            client,
            input,
            settings,
            shift: bits - 8,
            peak,
        }
    }

    fn has_chroma(&self) -> bool {
        matches!(self.input.layout, FrameLayout::Yuv { .. })
    }

    /// Measures the window's current frame and builds its five weaves.
    pub(crate) fn gather(
        &self,
        context: &FrameContext,
        planes: &WindowPlanes,
    ) -> Result<Gathered, StageError> {
        let scene_change = if self.settings.micmatch == 1 {
            self.scene_change(context, planes)?
        } else {
            false
        };

        let pairs = self.pair_sums(context, planes)?;

        let woven = self.woven(context, planes);
        let mics = self.mics(context, &woven)?;

        let metrics = FrameMetrics {
            pairs,
            mics,
            scene_change,
        };

        Ok(Gathered { metrics, woven })
    }

    /// The partial sums of `|first - second|` over two luma planes.
    fn difference(&self, context: &FrameContext, first: &Scratch, second: &Scratch) -> Scratch {
        let first_plane = first.plane();
        let width = first_plane.width();
        let height = first_plane.height();
        let sums_len = segments(width) * height;
        let sums = Scratch::new::<u32>(context, sums_len, 1);
        let sizing = Sizing(context);

        let difference_planes = DifferencePlanes {
            first: first_plane,
            second: second.plane(),
            sums: sums.plane(),
        };
        difference_sums(&self.client, &sizing, difference_planes);

        sums
    }

    /// The scene measure of difference sums `sums`, VapourSynth's PlaneStats difference of the luma.
    fn scene_measure(&self, sums: &Scratch) -> Result<f64, StageError> {
        let read = self.read_u32(sums.plane());
        let partials = read.map_err(unreadable_scene_sums)?;
        let total: u64 = partials.iter().map(|partial| u64::from(*partial)).sum();

        let width = u64::from(self.input.width);
        let height = u64::from(self.input.height);
        let scale = width * height * self.peak;

        Ok(total as f64 / scale as f64)
    }

    /// VFM's scene change flag, from the current frame against the previous and the next against the current.
    ///
    /// When the next frame is the current one, its measure is the current frame's.
    fn scene_change(&self, context: &FrameContext, planes: &WindowPlanes) -> Result<bool, StageError> {
        let current_sums = self.difference(context, &planes.current[0], &planes.previous[0]);
        let next_sums = if planes.next_is_current {
            None
        } else {
            let sums = self.difference(context, &planes.next[0], &planes.current[0]);
            Some(sums)
        };

        let current_measure = self.scene_measure(&current_sums)?;
        let next_measure = match &next_sums {
            Some(sums) => self.scene_measure(sums)?,
            None => current_measure,
        };

        let scene_change = is_scene_change(current_measure, next_measure, self.settings.scthresh);

        Ok(scene_change)
    }

    /// The planes the field compare runs on, luma alone or every plane with `mchroma`.
    fn compared_planes(&self) -> usize {
        if self.settings.mchroma && self.has_chroma() {
            3
        } else {
            1
        }
    }

    /// The field compare sums of each pair the mode needs, the others left at zero.
    fn pair_sums(&self, context: &FrameContext, planes: &WindowPlanes) -> Result<PairSums, StageError> {
        let [current_previous, current_next, previous_next] = self.settings.pairs;

        let current_previous_sums = self.field_sums(context, planes, current_previous)?;
        let mut sums = PairSums {
            current_previous: current_previous_sums,
            current_next: FieldSums::default(),
            previous_next: FieldSums::default(),
        };

        if self.settings.mode >= 4 {
            sums.current_next = self.field_sums(context, planes, current_next)?;
            sums.previous_next = self.field_sums(context, planes, previous_next)?;
        }

        Ok(sums)
    }

    /// `compareFieldsSlow`'s sums for the raw matches of `pair`, added over every compared plane.
    fn field_sums(
        &self,
        context: &FrameContext,
        planes: &WindowPlanes,
        pair: (usize, usize),
    ) -> Result<FieldSums, StageError> {
        let (first, second) = pair;
        let sizing = Sizing(context);

        let mut launched = Vec::new();
        for plane_index in 0..self.compared_planes() {
            let current = planes.current[plane_index].plane();
            let width = current.width();
            let height = current.height();
            let sums_len = compare_sums_len(width, height);
            if sums_len == 0 {
                continue;
            }

            let (shift_x, shift_y) = subsampling_shifts(self.input.layout, plane_index);
            let pass = ComparePass {
                first: first as u32,
                second: second as u32,
                field: self.settings.field,
                exclude_from: self.settings.y0 >> shift_y,
                exclude_to: self.settings.y1 >> shift_y,
                start_x: COMPARE_EDGE >> shift_x,
                shift: self.shift,
            };

            let differences = Scratch::new::<u32>(context, width, height / 2);
            let map = Scratch::new::<u32>(context, width, height);
            let sums = Scratch::new::<u32>(context, sums_len as u32, 1);
            let compare_planes = ComparePlanes {
                frames: [
                    planes.previous[plane_index].plane(),
                    current,
                    planes.next[plane_index].plane(),
                ],
                differences: differences.plane(),
                map: map.plane(),
                sums: sums.plane(),
            };
            compare_fields(&self.client, &sizing, compare_planes, pass);

            launched.push([differences, map, sums]);
        }

        let mut total = FieldSums::default();
        for [_, _, sums] in &launched {
            let read = self.read_u32(sums.plane());
            let partials = read.map_err(unreadable_field_sums)?;
            let plane_sums = FieldSums::from_partials(&partials);
            total += plane_sums;
        }

        Ok(total)
    }

    /// Every match's weave, by raw match number, each plane woven from the frames the match names.
    fn woven(&self, context: &FrameContext, planes: &WindowPlanes) -> Vec<Vec<Scratch>> {
        let sizing = Sizing(context);
        let frames = [
            planes.previous.as_slice(),
            planes.current.as_slice(),
            planes.next.as_slice(),
        ];

        let mut woven = Vec::with_capacity(MATCHES);
        for matched in 0..MATCHES {
            let (matching_planes, other_planes, matching_parity) =
                match_pairing(frames, matched, self.settings.field);

            let mut match_planes = Vec::with_capacity(matching_planes.len());
            for (matching, other) in matching_planes.iter().zip(other_planes) {
                let matching_plane = matching.plane();
                let width = matching_plane.width();
                let height = matching_plane.height();
                let output = Scratch::new::<f32>(context, width, height);

                let weave_planes = WeavePlanes {
                    matching: matching_plane,
                    other: other.plane(),
                    output: output.plane(),
                };
                weave(&self.client, &sizing, weave_planes, matching_parity);

                match_planes.push(output);
            }

            woven.push(match_planes);
        }

        woven
    }

    /// The planes whose comb masks reach the mic, luma alone or every plane with `chroma`.
    fn masked_planes(&self) -> usize {
        if self.settings.chroma && self.has_chroma() {
            3
        } else {
            1
        }
    }

    /// `calcMI`'s count of each weave in `woven`, by raw match number.
    fn mics(&self, context: &FrameContext, woven: &[Vec<Scratch>]) -> Result<[u32; MATCHES], StageError> {
        let sizing = Sizing(context);
        let luma = woven[C][0].plane();
        let luma_width = luma.width();
        let luma_height = luma.height();
        let blocks = self.settings.blocks;
        let count_len = mic_count_len(luma_width, luma_height, blocks);

        let results = Scratch::new::<u32>(context, MATCHES as u32, 1);
        let counts = Scratch::new::<u32>(context, count_len as u32, 1);
        clear_mics(&self.client, &sizing, results.plane());

        let masks: Vec<Scratch> = woven[C][..self.masked_planes()]
            .iter()
            .map(|plane| {
                let device_plane = plane.plane();
                Scratch::new::<u32>(context, device_plane.width(), device_plane.height())
            })
            .collect();

        for (slot, match_planes) in woven.iter().enumerate() {
            for (plane, mask) in match_planes.iter().zip(&masks) {
                let comb_planes = CombPlanes {
                    input: plane.plane(),
                    mask: mask.plane(),
                };
                comb_mask(&self.client, &sizing, comb_planes, self.settings.cthresh);
            }

            if let [luma_mask, u_mask, v_mask] = masks.as_slice() {
                let (shift_x, shift_y) = subsampling_shifts(self.input.layout, 1);
                let chroma_masks = ChromaMasks {
                    u: u_mask.plane(),
                    v: v_mask.plane(),
                    luma: luma_mask.plane(),
                };
                chroma_into_luma(&self.client, &sizing, chroma_masks, shift_x, shift_y);
            }

            let mic_planes = MicPlanes {
                mask: masks[0].plane(),
                counts: counts.plane(),
                results: results.plane(),
            };
            mic(&self.client, &sizing, mic_planes, blocks, slot as u32);
        }

        let read = self.read_u32(results.plane());
        let values = read.map_err(unreadable_mics)?;
        let mut mics = [0u32; MATCHES];
        mics.copy_from_slice(&values[..MATCHES]);

        Ok(mics)
    }

    fn read_u32(&self, plane: DevicePlane<'_>) -> Result<Vec<u32>, ServerError> {
        let handle = plane.handle().clone();
        let bytes = self.client.read_one(handle)?;

        Ok(u32::from_bytes(&bytes).to_vec())
    }
}

/// The frame a raw match takes its matching field from, the frame it takes the other field from, and the
/// row parity of the matching field.
///
/// `frames` holds the previous, current and next frames. `field` is VFM's field, 1 when the matching field
/// is the top field.
pub(crate) fn match_pairing<T: Copy>(frames: [T; 3], matched: usize, field: u32) -> (T, T, u32) {
    let [previous, current, next] = frames;
    let (matching, other) = match matched {
        P => (current, previous),
        C => (current, current),
        N => (current, next),
        B => (previous, current),
        U => (next, current),
        _ => unreachable!("VFM has five matches, not {matched}"),
    };

    (matching, other, 1 - field)
}
