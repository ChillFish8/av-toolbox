use super::reference::merge;
use crate::degrain::DegrainSettings;
use crate::deinterlace::qtgmc::{Parity, sharpness_strength};
use crate::motion::{
    BlockGrid,
    ConfidenceScores,
    MotionSettings,
    PROGRESSIVE_SAD_MASK,
    PlaneScale,
    SceneChange,
};
use crate::testing::{
    ReferenceField,
    ReferenceNeighbour,
    reference_back_blend,
    reference_bob,
    reference_clamp,
    reference_compensate,
    reference_degrain,
    reference_normalise,
    reference_pyramid,
    reference_repair_blend,
    reference_resharpen,
    reference_sad_mask,
    reference_scene_changed,
    reference_search,
    reference_search_clip,
    reference_shimmer,
    reference_temporal_limits,
    reference_tr0,
    reference_vertical_blur,
};

/// The interpolator that builds `edi` from a frame's plane, its width, its height and the field's parity.
pub type Interpolate<'a> = &'a dyn Fn(&[f32], usize, usize, Parity) -> Vec<f32>;

/// Scores the CPU search asks for, though the chain never reads a confidence.
const UNREAD_SCORES: ConfidenceScores = ConfidenceScores {
    noise_floor: 0.0,
    thsad: 1.0,
};

/// The weight of `edi` in TR1 at a radius of 1.
const TR1_EDI_WEIGHT_1: f32 = 0.25;

/// The weight of the second degrain in TR1 at a radius of 2.
const TR1_SECOND_WEIGHT: f32 = 0.2;

/// The weight of `edi` in TR1 at a radius of 2.
const TR1_EDI_WEIGHT_2: f32 = 0.0625;

/// QTGMC's InputType.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReferenceInput {
    /// Interlaced frames, each giving its top field and then its bottom field.
    Interlaced,
    /// Progressive frames, each its own entry.
    Progressive,
    /// Badly deinterlaced progressive frames, each rebuilt from one of its fields.
    Repair,
    /// The same as [Repair](ReferenceInput::Repair), keeping the other field of every frame.
    RepairReversed,
}

/// The settings of QTGMC's chain that [reference_qtgmc](crate::testing::reference_qtgmc) runs with.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QtgmcReferenceSettings {
    pub tr0: u32,
    pub tr1: u32,
    pub tr2: u32,
    pub sharpness: f32,
    pub thsad1: f32,
    pub thsad2: f32,
    pub input: ReferenceInput,
    /// The stream index of the clip's first frame, which the kept parity of repair input counts from.
    pub first_frame: usize,
}

impl Default for QtgmcReferenceSettings {
    fn default() -> Self {
        Self {
            tr0: 2,
            tr1: 2,
            tr2: 1,
            sharpness: 1.0,
            thsad1: 640.0,
            thsad2: 256.0,
            input: ReferenceInput::Interlaced,
            first_frame: 0,
        }
    }
}

impl QtgmcReferenceSettings {
    /// MVTools' scene-change rule as QTGMC sets it, for compensation and both degrains.
    pub const SCENE_CHANGE: SceneChange = SceneChange {
        thscd1: 180.0,
        thscd2: 98,
    };

    /// MVTools' default scene-change rule, which QTGMC's repair mask runs with as it sets no other.
    pub const MASK_SCENE_CHANGE: SceneChange = SceneChange {
        thscd1: 400.0,
        thscd2: 130,
    };

    /// How far apart the furthest motion neighbours are, `max(1, tr1, tr2)` entries.
    pub fn max_tr(&self) -> u32 {
        self.tr1.max(self.tr2).max(1)
    }

    /// The settings of each TR1 degrain, which reads one distance on either side.
    pub fn tr1_degrain(&self) -> DegrainSettings {
        degrain_settings(1, self.thsad1)
    }

    /// The settings of the TR2 degrain.
    pub fn tr2_degrain(&self) -> DegrainSettings {
        degrain_settings(self.tr2.max(1), self.thsad2)
    }
}

fn degrain_settings(temporal_radius: u32, thsad: f32) -> DegrainSettings {
    let rule = QtgmcReferenceSettings::SCENE_CHANGE;

    DegrainSettings {
        temporal_radius,
        thsad,
        thsad_chroma: thsad,
        thscd1: rule.thscd1,
        thscd2: rule.thscd2,
    }
}

/// The offsets of an entry's motion neighbours up to `radius` entries away, nearest first and the later
/// entry of each pair first, as MVTools' degrains order their backward and forward vectors.
pub fn qtgmc_deltas(radius: u32) -> Vec<i32> {
    (1..=radius as i32)
        .flat_map(|distance| [distance, -distance])
        .collect()
}

/// One motion neighbour of an entry, with the neighbour's planes the chain read through it.
#[derive(Debug, Clone, PartialEq)]
pub struct ReferenceMotion {
    /// How many entries after the centre the neighbour is, negative for entries before it.
    pub delta: i32,
    /// The search from the centre to the neighbour, in half pixels.
    pub field: ReferenceField,
    /// Whether the search is a scene change.
    pub flagged: bool,
    /// The neighbour's normalised search clip.
    pub neighbour_normalised: Vec<f32>,
    pub neighbour_edi: [Vec<f32>; 3],
    /// The neighbour's planes after the limit, which TR2 degrains with.
    pub neighbour_limited: [Vec<f32>; 3],
}

/// Every intermediate of one output entry, in the chain's order.
///
/// The luma-only steps hold one plane. The rest hold Y, U and V, with an empty plane where the input
/// has none.
#[derive(Debug, Clone, PartialEq)]
pub struct QtgmcFieldSteps {
    pub bob: Vec<f32>,
    pub tr0: Vec<f32>,
    /// The TR0 average after Rep0.
    pub repaired: Vec<f32>,
    pub search_clip: Vec<f32>,
    pub normalised: Vec<f32>,
    /// The neighbours that exist, in [qtgmc_deltas] order.
    pub motion: Vec<ReferenceMotion>,
    pub edi: [Vec<f32>; 3],
    pub upper: [Vec<f32>; 3],
    pub lower: [Vec<f32>; 3],
    /// `edi` after TR1.
    pub smoothed: [Vec<f32>; 3],
    pub resharpened: [Vec<f32>; 3],
    /// The resharpened luma after the back-blend, with chroma passed through.
    pub back_blended: [Vec<f32>; 3],
    pub limited: [Vec<f32>; 3],
    /// The limited planes after TR2.
    pub stable: [Vec<f32>; 3],
    /// The stable planes after Rep2, the entry's output.
    pub output: [Vec<f32>; 3],
}

/// What every call of the chain reads.
struct Chain<'a> {
    frames: &'a [[Vec<f32>; 3]],
    sizes: [(usize, usize); 3],
    bits: u32,
    settings: &'a QtgmcReferenceSettings,
    interpolate: Interpolate<'a>,
}

/// The luma-only search path and the interpolation of one entry.
///
/// In the repair modes `edi` holds the frame itself until the entry's motion exists.
struct Prepared {
    bob: Vec<f32>,
    tr0: Vec<f32>,
    repaired: Vec<f32>,
    search_clip: Vec<f32>,
    normalised: Vec<f32>,
    pyramid: Vec<Vec<f32>>,
    edi: [Vec<f32>; 3],
}

/// One search from an entry to a neighbour that exists.
struct FieldMotion {
    delta: i32,
    neighbour: usize,
    field: ReferenceField,
    flagged: bool,
}

/// One entry's planes from the temporal limits up to the limit.
#[derive(Default)]
struct Limited {
    upper: [Vec<f32>; 3],
    lower: [Vec<f32>; 3],
    smoothed: [Vec<f32>; 3],
    resharpened: [Vec<f32>; 3],
    back_blended: [Vec<f32>; 3],
    limited: [Vec<f32>; 3],
}

/// Every step of every entry.
struct ChainRun {
    prepared: Vec<Prepared>,
    motion: Vec<Vec<FieldMotion>>,
    limited: Vec<Limited>,
    stable: Vec<[Vec<f32>; 3]>,
    output: Vec<[Vec<f32>; 3]>,
}

/// The frame holding `field` and the field's parity, top field first.
fn field_source(frames: &[[Vec<f32>; 3]], field: usize) -> (&[Vec<f32>; 3], Parity) {
    let parity = if field.is_multiple_of(2) {
        Parity::Top
    } else {
        Parity::Bottom
    };

    (&frames[field / 2], parity)
}

/// The parity of the field that a repaired `frame` keeps, top field first, by its stream index.
///
/// [RepairReversed](ReferenceInput::RepairReversed) keeps the other field.
pub(super) fn kept_parity(input: ReferenceInput, frame: usize) -> Parity {
    let reversed = input == ReferenceInput::RepairReversed;
    let top = frame.is_multiple_of(2) != reversed;

    if top { Parity::Top } else { Parity::Bottom }
}

/// A repaired frame's `edi`, its luma blended towards `interpolate`'s rebuild of the field of `parity`
/// through `mask` and its chroma the frame's own.
pub(super) fn repaired_edi(
    frame: &[Vec<f32>; 3],
    (width, height): (usize, usize),
    parity: Parity,
    mask: &[f32],
    interpolate: Interpolate<'_>,
) -> [Vec<f32>; 3] {
    let rebuilt = interpolate(&frame[0], width, height, parity);
    let luma = reference_repair_blend(&frame[0], &rebuilt, mask);

    [luma, frame[1].clone(), frame[2].clone()]
}

/// The mask that blends a repaired entry towards its rebuild, from the `sads` of its search to the next
/// entry.
///
/// It is MVTools' `Mask(kind=1)` at QTGMC's `ProgSADMask`, as havsfunc r33's line 1326 calls it. That call
/// sets no scene-change rule, so it is flagged by
/// [MASK_SCENE_CHANGE](QtgmcReferenceSettings::MASK_SCENE_CHANGE). An entry without a next entry gets a mask
/// of 0.
pub(super) fn repair_mask(sads: Option<&[f32]>, grid: &BlockGrid, luma: (usize, usize)) -> Vec<f32> {
    let Some(sads) = sads else {
        return vec![0.0; luma.0 * luma.1];
    };

    let rule = QtgmcReferenceSettings::MASK_SCENE_CHANGE;
    let flagged = reference_scene_changed(sads, grid, rule);

    reference_sad_mask(sads, grid, flagged, PROGRESSIVE_SAD_MASK, luma)
}

/// The size of a chroma plane of a `width` by `height` frame holding `samples`, by trying each
/// subsampling.
fn chroma_size(samples: usize, plane: usize, (width, height): (usize, usize)) -> (usize, usize) {
    let half_width = width.div_ceil(2);
    let half_height = height.div_ceil(2);
    let candidates = [
        (width, height),
        (half_width, height),
        (half_width, half_height),
        (0, 0),
    ];

    let found = candidates
        .into_iter()
        .find(|(plane_width, plane_height)| plane_width * plane_height == samples);
    match found {
        Some(size) => size,
        None => panic!("chroma plane {plane} holds {samples} samples, no subsampling of {width}x{height}"),
    }
}

/// The size of each plane of `frame`.
fn plane_sizes(frame: &[Vec<f32>; 3], luma: (usize, usize)) -> [(usize, usize); 3] {
    let (width, height) = luma;
    assert_eq!(frame[0].len(), width * height, "the luma is not {width}x{height}");

    let first = chroma_size(frame[1].len(), 1, luma);
    let second = chroma_size(frame[2].len(), 2, luma);

    [luma, first, second]
}

/// How much smaller a plane of `size` is than luma.
fn plane_scale(size: (usize, usize), luma: (usize, usize)) -> PlaneScale {
    PlaneScale {
        shift_x: u32::from(size.0 < luma.0),
        shift_y: u32::from(size.1 < luma.1),
    }
}

impl Chain<'_> {
    fn entry_count(&self) -> usize {
        match self.settings.input {
            ReferenceInput::Interlaced => 2 * self.frames.len(),
            _ => self.frames.len(),
        }
    }

    fn luma(&self) -> (usize, usize) {
        self.sizes[0]
    }

    /// The bob and `edi` of `entry`, with `edi` the frame itself in the repair modes.
    fn interpolated(&self, entry: usize) -> (Vec<f32>, [Vec<f32>; 3]) {
        let (width, height) = self.luma();

        match self.settings.input {
            ReferenceInput::Interlaced => self.field_interpolated(entry),
            ReferenceInput::Progressive => {
                let frame = &self.frames[entry];
                (frame[0].clone(), frame.clone())
            },
            ReferenceInput::Repair | ReferenceInput::RepairReversed => {
                let frame = &self.frames[entry];
                let bob = reference_vertical_blur(&frame[0], width, height);
                (bob, frame.clone())
            },
        }
    }

    /// The bob and `edi` of `field`.
    fn field_interpolated(&self, field: usize) -> (Vec<f32>, [Vec<f32>; 3]) {
        let (frame, parity) = field_source(self.frames, field);
        let bob = reference_bob(&frame[0], self.luma(), parity);
        let edi = std::array::from_fn(|plane| {
            let (width, height) = self.sizes[plane];
            if width == 0 {
                return Vec::new();
            }

            (self.interpolate)(&frame[plane], width, height, parity)
        });

        (bob, edi)
    }

    /// The bob, TR0, Rep0, search clip, normalised clip and `edi` of every entry.
    fn prepare(&self) -> Vec<Prepared> {
        let (width, height) = self.luma();
        let levels = MotionSettings::default().pyramid_levels;

        let interpolated: Vec<_> = (0..self.entry_count())
            .map(|entry| self.interpolated(entry))
            .collect();
        let bobs: Vec<Vec<f32>> = interpolated.iter().map(|(bob, _)| bob.clone()).collect();

        let mut prepared = Vec::with_capacity(bobs.len());
        for (field, (bob, edi)) in interpolated.into_iter().enumerate() {
            let tr0 = reference_tr0(&bobs, field, self.settings.tr0, width, height);
            let repaired = reference_shimmer(&tr0, &bob, width, height, self.bits);
            let search_clip = reference_search_clip(&repaired, &bob, width, height, self.bits);
            let normalised = reference_normalise(&search_clip, self.bits);
            let pyramid = reference_pyramid(&normalised, width as u32, height as u32, levels);

            prepared.push(Prepared {
                bob,
                tr0,
                repaired,
                search_clip,
                normalised,
                pyramid,
                edi,
            });
        }

        prepared
    }

    /// The motion search from `centre` to every neighbour that exists, with its scene-change flag.
    fn search(&self, prepared: &[Prepared], centre: usize) -> Vec<FieldMotion> {
        let (width, height) = self.luma();
        let settings = MotionSettings::default();
        let grid = BlockGrid::new(&settings, width as u32, height as u32);
        let rule = QtgmcReferenceSettings::SCENE_CHANGE;

        let mut motion = Vec::new();
        for delta in qtgmc_deltas(self.settings.max_tr()) {
            let neighbour = centre as isize + delta as isize;
            if neighbour < 0 || neighbour >= prepared.len() as isize {
                continue;
            }

            let neighbour = neighbour as usize;
            let field = reference_search(
                &prepared[centre].pyramid,
                &prepared[neighbour].pyramid,
                width as u32,
                height as u32,
                &settings,
                UNREAD_SCORES,
            );
            let flagged = reference_scene_changed(&field.sads, &grid, rule);

            motion.push(FieldMotion {
                delta,
                neighbour,
                field,
                flagged,
            });
        }

        motion
    }

    /// Rebuilds the `edi` of every repaired entry, masked by its motion to the next entry.
    fn repair(&self, prepared: &mut [Prepared], motion: &[Vec<FieldMotion>]) {
        let luma = self.luma();
        let grid = self.grid();

        for (entry, found) in motion.iter().enumerate() {
            let next = found.iter().find(|search| search.delta == 1);
            let sads = next.map(|search| search.field.sads.as_slice());
            let mask = repair_mask(sads, &grid, luma);

            let index = self.settings.first_frame + entry;
            let parity = kept_parity(self.settings.input, index);
            let frame = &self.frames[entry];
            prepared[entry].edi = repaired_edi(frame, luma, parity, &mask, self.interpolate);
        }
    }

    fn grid(&self) -> BlockGrid {
        let (width, height) = self.luma();

        BlockGrid::new(&MotionSettings::default(), width as u32, height as u32)
    }

    /// Degrains `centre` of `plane` with `neighbours`, each a search and the neighbour's plane.
    fn degrain(
        &self,
        centre: &[f32],
        plane: usize,
        neighbours: &[(&FieldMotion, &[f32])],
        settings: &DegrainSettings,
    ) -> Vec<f32> {
        let size = self.sizes[plane];
        let scale = plane_scale(size, self.luma());
        let grid = self.grid();
        let pel = MotionSettings::default().pel;

        let references: Vec<ReferenceNeighbour<'_>> = neighbours
            .iter()
            .map(|(found, neighbour)| ReferenceNeighbour {
                plane: neighbour,
                vectors: &found.field.vectors,
                sads: &found.field.sads,
            })
            .collect();
        let flagged: Vec<bool> = neighbours.iter().map(|(found, _)| found.flagged).collect();

        reference_degrain(
            centre,
            size,
            &references,
            &flagged,
            &grid,
            settings,
            pel,
            scale,
            plane > 0,
        )
    }

    /// The temporal limits and the TR1 average of one plane, as `(upper, lower, smoothed)`.
    fn smooth_plane(
        &self,
        prepared: &[Prepared],
        motion: &[FieldMotion],
        centre: usize,
        plane: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let size = self.sizes[plane];
        let scale = plane_scale(size, self.luma());
        let edi = &prepared[centre].edi[plane];
        let grid = self.grid();
        let pel = MotionSettings::default().pel;

        let compensated: Vec<Vec<f32>> = motion
            .iter()
            .filter(|found| found.delta.abs() == 1)
            .map(|found| {
                if found.flagged {
                    return edi.clone();
                }

                let neighbour = &prepared[found.neighbour].edi[plane];
                reference_compensate(neighbour, size, &found.field.vectors, &grid, pel, scale)
            })
            .collect();
        let borrowed: Vec<&[f32]> = compensated.iter().map(Vec::as_slice).collect();
        let (upper, lower) = reference_temporal_limits(edi, &borrowed);

        let settings = self.settings.tr1_degrain();
        let degrain_at = |distance: i32| {
            let neighbours: Vec<(&FieldMotion, &[f32])> = motion
                .iter()
                .filter(|found| found.delta.abs() == distance)
                .map(|found| (found, prepared[found.neighbour].edi[plane].as_slice()))
                .collect();

            self.degrain(edi, plane, &neighbours, &settings)
        };
        let smoothed = match self.settings.tr1 {
            0 => edi.clone(),
            1 => {
                let near = degrain_at(1);
                merge(&near, edi, TR1_EDI_WEIGHT_1)
            },
            _ => {
                let near = degrain_at(1);
                let far = degrain_at(2);
                let blended = merge(&near, &far, TR1_SECOND_WEIGHT);
                merge(&blended, edi, TR1_EDI_WEIGHT_2)
            },
        };

        (upper, lower, smoothed)
    }

    /// The temporal limits, TR1, resharpen, back-blend and limit of `centre`.
    fn limit(&self, prepared: &[Prepared], motion: &[FieldMotion], centre: usize) -> Limited {
        let settings = self.settings;
        let strength = sharpness_strength(settings.sharpness, settings.tr1, settings.tr2);
        let sharpened = settings.sharpness > 0.0;

        let mut limited = Limited::default();
        for plane in 0..3 {
            let (width, height) = self.sizes[plane];
            if width == 0 {
                continue;
            }

            let (upper, lower, smoothed) = self.smooth_plane(prepared, motion, centre, plane);

            let mut resharpened = smoothed.clone();
            if sharpened {
                resharpened = reference_resharpen(&smoothed, width, height, strength, self.bits);
            }

            let mut back_blended = resharpened.clone();
            if sharpened && plane == 0 {
                back_blended = reference_back_blend(&resharpened, &smoothed, width, height, self.bits);
            }

            limited.limited[plane] = reference_clamp(&back_blended, &upper, &lower);
            limited.upper[plane] = upper;
            limited.lower[plane] = lower;
            limited.smoothed[plane] = smoothed;
            limited.resharpened[plane] = resharpened;
            limited.back_blended[plane] = back_blended;
        }

        limited
    }

    /// The TR2 average and Rep2 of `centre`, as its stable planes and its output.
    fn finish(
        &self,
        prepared: &[Prepared],
        motion: &[FieldMotion],
        limited: &[Limited],
        centre: usize,
    ) -> ([Vec<f32>; 3], [Vec<f32>; 3]) {
        let radius = self.settings.tr2 as i32;
        let settings = self.settings.tr2_degrain();

        let mut stable: [Vec<f32>; 3] = Default::default();
        let mut output: [Vec<f32>; 3] = Default::default();
        for plane in 0..3 {
            let (width, height) = self.sizes[plane];
            if width == 0 {
                continue;
            }

            let centre_plane = &limited[centre].limited[plane];
            let neighbours: Vec<(&FieldMotion, &[f32])> = motion
                .iter()
                .filter(|found| found.delta.abs() <= radius)
                .map(|found| (found, limited[found.neighbour].limited[plane].as_slice()))
                .collect();
            stable[plane] = match radius {
                0 => centre_plane.clone(),
                _ => self.degrain(centre_plane, plane, &neighbours, &settings),
            };

            let edi = &prepared[centre].edi[plane];
            output[plane] = reference_shimmer(&stable[plane], edi, width, height, self.bits);
        }

        (stable, output)
    }

    fn run(&self) -> ChainRun {
        let mut prepared = self.prepare();
        let motion: Vec<Vec<FieldMotion>> = (0..prepared.len())
            .map(|centre| self.search(&prepared, centre))
            .collect();

        let input = self.settings.input;
        if input == ReferenceInput::Repair || input == ReferenceInput::RepairReversed {
            self.repair(&mut prepared, &motion);
        }

        let limited: Vec<Limited> = motion
            .iter()
            .enumerate()
            .map(|(centre, found)| self.limit(&prepared, found, centre))
            .collect();

        let mut stable = Vec::with_capacity(prepared.len());
        let mut output = Vec::with_capacity(prepared.len());
        for (centre, found) in motion.iter().enumerate() {
            let (stable_planes, output_planes) = self.finish(&prepared, found, &limited, centre);
            stable.push(stable_planes);
            output.push(output_planes);
        }

        ChainRun {
            prepared,
            motion,
            limited,
            stable,
            output,
        }
    }
}

/// Builds the chain over `frames`, which holds at least one frame.
fn chain<'a>(
    frames: &'a [[Vec<f32>; 3]],
    luma: (usize, usize),
    bits: u32,
    settings: &'a QtgmcReferenceSettings,
    interpolate: Interpolate<'a>,
) -> Chain<'a> {
    let first = frames.first().expect("the chain needs at least one frame");
    let sizes = plane_sizes(first, luma);

    Chain {
        frames,
        sizes,
        bits,
        settings,
        interpolate,
    }
}

/// Runs the whole chain on the CPU over `frames`, returning every output in order.
///
/// `frames` are Y, U and V code values. Chroma may be subsampled, or empty for gray. `settings.input`
/// sets how each frame enters the chain:
///
/// - **Interlaced.** Each frame is top field first and yields its top field and then its bottom field.
///   `interpolate` builds `edi` from a frame's plane and a field's parity.
/// - **Progressive.** Each frame yields one output, and its bob and `edi` are the frame itself.
/// - **Repair.** Each frame yields one output. Its bob is a vertical 1-2-1 blur of its luma. The frame at
///   stream index `n`, the clip starting at `settings.first_frame`, keeps the field of parity `n mod 2`, or
///   `(n + 1) mod 2` when reversed, and `interpolate` rebuilds the other rows. `edi`'s luma blends the
///   frame towards that rebuild through the SAD mask of its motion to the next frame, and its chroma is
///   the frame's. The last frame keeps its own luma.
///
/// The clip is one chunk. TR0 repeats the nearest bob past either end, a motion neighbour past either
/// end is skipped, and a degrain without neighbours is its centre. Motion searches the normalised search
/// clip at the library's defaults, and chroma reads the luma vectors.
pub fn reference_qtgmc(
    frames: &[[Vec<f32>; 3]],
    (width, height): (usize, usize),
    bits: u32,
    settings: &QtgmcReferenceSettings,
    interpolate: Interpolate<'_>,
) -> Vec<[Vec<f32>; 3]> {
    let run = reference_qtgmc_run(frames, (width, height), bits, settings, interpolate);

    run.output
}

/// One search from an entry to a neighbour that exists.
#[derive(Debug, Clone, PartialEq)]
pub struct ReferenceSearch {
    /// How many entries after the centre the neighbour is, negative for entries before it.
    pub delta: i32,
    /// The search from the centre to the neighbour, in half pixels.
    pub field: ReferenceField,
    /// Whether the search is a scene change.
    pub flagged: bool,
}

/// Every output of the chain, with every entry's searches.
#[derive(Debug, Clone, PartialEq)]
pub struct QtgmcReferenceRun {
    pub output: Vec<[Vec<f32>; 3]>,
    /// Each entry's searches, in [qtgmc_deltas] order.
    pub motion: Vec<Vec<ReferenceSearch>>,
}

/// The same as [reference_qtgmc](crate::testing::reference_qtgmc), also giving every entry's searches.
pub fn reference_qtgmc_run(
    frames: &[[Vec<f32>; 3]],
    luma: (usize, usize),
    bits: u32,
    settings: &QtgmcReferenceSettings,
    interpolate: Interpolate<'_>,
) -> QtgmcReferenceRun {
    let run = chain(frames, luma, bits, settings, interpolate).run();

    let motion = run
        .motion
        .into_iter()
        .map(|searches| {
            searches
                .into_iter()
                .map(|found| ReferenceSearch {
                    delta: found.delta,
                    field: found.field,
                    flagged: found.flagged,
                })
                .collect()
        })
        .collect();

    QtgmcReferenceRun {
        output: run.output,
        motion,
    }
}

/// Every intermediate of entry `centre` of [reference_qtgmc](crate::testing::reference_qtgmc) over the
/// same arguments.
pub fn reference_qtgmc_field(
    frames: &[[Vec<f32>; 3]],
    luma: (usize, usize),
    bits: u32,
    settings: &QtgmcReferenceSettings,
    interpolate: Interpolate<'_>,
    centre: usize,
) -> QtgmcFieldSteps {
    let mut run = chain(frames, luma, bits, settings, interpolate).run();
    assert!(
        centre < run.prepared.len(),
        "entry {centre} asked for, the clip has {}",
        run.prepared.len()
    );

    let motion = run.motion[centre]
        .iter()
        .map(|found| ReferenceMotion {
            delta: found.delta,
            field: found.field.clone(),
            flagged: found.flagged,
            neighbour_normalised: run.prepared[found.neighbour].normalised.clone(),
            neighbour_edi: run.prepared[found.neighbour].edi.clone(),
            neighbour_limited: run.limited[found.neighbour].limited.clone(),
        })
        .collect();

    let prepared = run.prepared.swap_remove(centre);
    let limited = run.limited.swap_remove(centre);
    let stable = run.stable.swap_remove(centre);
    let output = run.output.swap_remove(centre);

    QtgmcFieldSteps {
        bob: prepared.bob,
        tr0: prepared.tr0,
        repaired: prepared.repaired,
        search_clip: prepared.search_clip,
        normalised: prepared.normalised,
        motion,
        edi: prepared.edi,
        upper: limited.upper,
        lower: limited.lower,
        smoothed: limited.smoothed,
        resharpened: limited.resharpened,
        back_blended: limited.back_blended,
        limited: limited.limited,
        stable,
        output,
    }
}
