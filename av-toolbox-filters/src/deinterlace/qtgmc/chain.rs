use av_toolbox_core::StageError;
use av_toolbox_cubek::DevicePlane;
use av_toolbox_cubek::degrain::{DegrainNeighbour, DegrainSettings, degrain};
use av_toolbox_cubek::deinterlace::qtgmc::{
    BackBlendPlanes,
    BobPlanes,
    Parity,
    QtgmcTables,
    SearchPlanes,
    ShimmerPlanes,
    StepScratch,
    back_blend,
    bob,
    clamp_between,
    merge,
    search_clip_from_repaired,
    shimmer,
    smode2_resharpen,
    temporal_limits,
    tr0_taps,
    weighted_sum,
};
use av_toolbox_cubek::motion::{
    BlockGrid,
    CompensatePlanes,
    MotionSettings,
    PlaneScale,
    Pyramid,
    SadMaskTables,
    SceneChange,
    build_pyramid,
    compensate,
    level_dims,
    normalise,
    scene_change_flags,
    search,
};
use av_toolbox_gpu::{Frame, FrameContext, FrameError, FrameFormat, FrameMeta, Plane, Runtime};

use super::cache::{FieldCache, FieldState, Motion, Neighbour, Place, Progress, Reach};
use crate::deinterlace::qtgmc::{Input, STAGE};
use crate::deinterlace::rebuild::{FieldRebuilder, channel_plane};
use crate::deinterlace::spatial::Predictor;
use crate::scratch::Scratch;
use crate::sizing::Sizing;

/// The weight of `edi` in TR1 at a radius of 1.
const TR1_EDI_WEIGHT_1: f32 = 0.25;

/// The weight of the degrain at ±2 in TR1 at a radius of 2.
const TR1_SECOND_WEIGHT: f32 = 0.2;

/// The weight of `edi` in TR1 at a radius of 2.
const TR1_EDI_WEIGHT_2: f32 = 0.0625;

pub(super) fn lost_slot(slot: &str, field: usize) -> StageError {
    let message = format!("{STAGE} lost the {slot} of field {field} before every step had read it");

    StageError::msg(message)
}

pub(super) fn no_prediction(plane: usize) -> StageError {
    let message = format!("{STAGE} got no spatial prediction of plane {plane}");

    StageError::msg(message)
}

fn bad_frame(error: FrameError) -> StageError {
    let message = format!("{STAGE} built a bad frame: {error}");

    StageError::msg(message)
}

/// The slot `slot` of `field`, refusing one already freed.
pub(super) fn held<'a, T>(slot: &'a Option<T>, name: &str, field: usize) -> Result<&'a T, StageError> {
    match slot {
        Some(value) => Ok(value),
        None => Err(lost_slot(name, field)),
    }
}

/// The state of `field`, refusing one no longer held.
pub(super) fn state_of(cache: &FieldCache, field: usize) -> Result<&FieldState, StageError> {
    match cache.field(field) {
        Some(state) => Ok(state),
        None => Err(lost_slot("state", field)),
    }
}

fn edi_of(state: &FieldState) -> &Option<Vec<Scratch>> {
    &state.edi
}

fn limited_of(state: &FieldState) -> &Option<Vec<Scratch>> {
    &state.limited
}

/// The offsets of a field's motion neighbours up to `radius` fields away, nearest first and the later
/// field of each pair first.
fn neighbour_deltas(radius: u32) -> Vec<i32> {
    (1..=radius as i32)
        .flat_map(|distance| [distance, -distance])
        .collect()
}

/// The settings every step of the chain runs with.
#[derive(Debug, Clone, Copy)]
pub(super) struct ChainSettings {
    pub(super) tr0: u32,
    pub(super) tr1: u32,
    pub(super) tr2: u32,
    pub(super) max_tr: u32,
    /// The resharpen's strength, `None` when sharpening is off.
    pub(super) strength: Option<f32>,
    pub(super) motion: MotionSettings,
    pub(super) scene_change: SceneChange,
    /// The scene-change rule of repair's SAD mask.
    pub(super) mask_scene_change: SceneChange,
    pub(super) tr1_degrain: DegrainSettings,
    pub(super) tr2_degrain: DegrainSettings,
}

impl ChainSettings {
    pub(super) fn reach(&self) -> Reach {
        Reach {
            tr0: self.tr0,
            max_tr: self.max_tr,
            smoothed: self.tr1.max(1),
            tr2: self.tr2,
        }
    }
}

/// The size of one plane and how much smaller than luma it is.
#[derive(Debug, Clone, Copy)]
pub(super) struct PlaneShape {
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) scale: PlaneScale,
}

/// What every step of the chain reads, built once per format.
pub(super) struct Chain<R: Runtime> {
    pub(super) rebuilder: FieldRebuilder<R>,
    pub(super) predictor: Predictor,
    pub(super) tables: QtgmcTables,
    /// The SAD mask's tables, for repair input alone.
    pub(super) mask_tables: Option<SadMaskTables>,
    /// What the stage is given.
    pub(super) mode: Input,
    pub(super) input: FrameFormat,
    pub(super) output: FrameFormat,
    /// Each plane's shape, in channel order.
    pub(super) shapes: Vec<PlaneShape>,
    pub(super) grid: BlockGrid,
    pub(super) settings: ChainSettings,
    pub(super) bits: u32,
}

impl<R: Runtime> Chain<R> {
    pub(super) fn scratch(&self, context: &FrameContext, plane: usize) -> Scratch {
        let shape = self.shapes[plane];

        Scratch::new::<f32>(context, shape.width, shape.height)
    }

    fn scratches<const N: usize>(&self, context: &FrameContext, plane: usize) -> [Scratch; N] {
        std::array::from_fn(|_| self.scratch(context, plane))
    }

    /// Bobs the field of `parity` in `frame`, plane `plane` of a widened frame.
    pub(super) fn bobbed(
        &self,
        context: &FrameContext,
        frame: DevicePlane<'_>,
        plane: usize,
        parity: Parity,
    ) -> Scratch {
        let output = self.scratch(context, plane);
        let sizing = Sizing(context);
        let planes = BobPlanes {
            frame,
            output: output.plane(),
        };
        let weights = self.tables.bob(plane, parity);
        bob(self.rebuilder.client(), &sizing, planes, weights);

        output
    }

    /// The `edi` of both fields of one plane of a widened frame, the top field first.
    fn interpolated(
        &self,
        context: &FrameContext,
        frame: DevicePlane<'_>,
        plane: usize,
    ) -> Result<[Scratch; 2], StageError> {
        if let Predictor::Cubic = self.predictor {
            let top = self.bobbed(context, frame, plane, Parity::Top);
            let bottom = self.bobbed(context, frame, plane, Parity::Bottom);
            return Ok([top, bottom]);
        }

        let rebuilt = self.rebuilder.both_fields(context, plane, frame);
        match <[Scratch; 2]>::try_from(rebuilt) {
            Ok(fields) => Ok(fields),
            Err(_) => Err(no_prediction(plane)),
        }
    }

    /// The two field frames of `frame`, each with its bob and `edi`, in the order they are shown.
    ///
    /// `parities` gives each field's parity and `metas` the metadata of each shown field.
    pub(super) fn fields_of(
        &self,
        context: &FrameContext,
        frame: &Frame,
        parities: [Parity; 2],
        metas: [Option<FrameMeta>; 2],
    ) -> Result<[FieldState; 2], StageError> {
        let mut bobs = None;
        let mut top_edi = Vec::with_capacity(self.shapes.len());
        let mut bottom_edi = Vec::with_capacity(self.shapes.len());

        for (plane, channel) in self.input.channels().iter().enumerate() {
            let source = channel_plane(STAGE, frame, *channel)?;
            let widened = self.rebuilder.widened(context, source, *channel);

            if plane == 0 {
                let bobbed = parities.map(|parity| self.bobbed(context, widened.plane(), 0, parity));
                bobs = Some(bobbed);
            }

            let [top, bottom] = self.interpolated(context, widened.plane(), plane)?;
            top_edi.push(top);
            bottom_edi.push(bottom);
        }

        let (first_edi, second_edi) = match parities[0] {
            Parity::Top => (top_edi, bottom_edi),
            Parity::Bottom => (bottom_edi, top_edi),
        };
        let [first_meta, second_meta] = metas;
        let Some([first_bob, second_bob]) = bobs else {
            return Err(lost_slot("bob", 0));
        };

        Ok([
            FieldState::new(first_meta, first_bob, first_edi),
            FieldState::new(second_meta, second_bob, second_edi),
        ])
    }

    /// The field whose bob TR0 reads at `offset` from `field`, the nearest one past either end of the
    /// chunk, or `None` while it has not arrived.
    fn tr0_source(cache: &FieldCache, field: usize, offset: i32) -> Option<usize> {
        let wanted = (field as i64 + i64::from(offset)).max(0);

        match cache.place(wanted) {
            Place::Held | Place::Dropped => Some(wanted as usize),
            Place::Coming => None,
            Place::Outside => cache.last(),
        }
    }

    /// Builds the search pyramid of `field` once the bobs TR0 reads have arrived.
    ///
    /// It runs TR0, Rep0, the search clip and the normalise on luma, answering whether it ran.
    pub(super) fn search_clip(
        &self,
        context: &FrameContext,
        cache: &mut FieldCache,
        field: usize,
    ) -> Result<bool, StageError> {
        let state = state_of(cache, field)?;
        if state.progress != Progress::Interpolated {
            return Ok(false);
        }

        let taps = tr0_taps(self.settings.tr0);
        let mut sources = Vec::with_capacity(taps.len());
        for (offset, _) in &taps {
            let Some(source) = Self::tr0_source(cache, field, *offset) else {
                return Ok(false);
            };

            sources.push(source);
        }

        let client = self.rebuilder.client();
        let sizing = Sizing(context);
        let centre_bob = held(&state.bob, "bob", field)?.plane();

        let mut bob_planes = Vec::with_capacity(sources.len());
        for source in &sources {
            let source_state = state_of(cache, *source)?;
            let source_bob = held(&source_state.bob, "bob", *source)?;
            bob_planes.push(source_bob.plane());
        }

        let weights: Vec<f32> = taps.iter().map(|(_, weight)| *weight).collect();
        let averaged = self.scratch(context, 0);
        weighted_sum(client, &sizing, &bob_planes, &weights, averaged.plane());

        let [difference, first, positive_choke, negative_choke] = self.scratches(context, 0);
        let repaired = self.scratch(context, 0);
        let repair = ShimmerPlanes {
            input: averaged.plane(),
            reference: centre_bob,
            difference: difference.plane(),
            first: first.plane(),
            positive_choke: positive_choke.plane(),
            negative_choke: negative_choke.plane(),
            output: repaired.plane(),
        };
        shimmer(client, &sizing, repair, self.bits);

        let [spread, merged] = self.scratches(context, 0);
        let search_clip = self.scratch(context, 0);
        let searching = SearchPlanes {
            repaired: repaired.plane(),
            bob: centre_bob,
            scratch: StepScratch {
                first: spread.plane(),
                second: merged.plane(),
            },
            output: search_clip.plane(),
        };
        search_clip_from_repaired(client, &sizing, &self.tables, searching, self.bits);

        let pyramid = self.pyramid(context, search_clip.plane());

        let Some(state) = cache.field_mut(field) else {
            return Err(lost_slot("state", field));
        };
        state.pyramid = Some(pyramid);
        state.progress = Progress::Searched;

        Ok(true)
    }

    /// The normalised `search_clip` at every level the search reads.
    fn pyramid(&self, context: &FrameContext, search_clip: DevicePlane<'_>) -> Vec<Scratch> {
        let client = self.rebuilder.client();
        let sizing = Sizing(context);
        let luma = self.shapes[0];

        let levels: Vec<Scratch> = (0..self.settings.motion.pyramid_levels)
            .map(|level| {
                let (width, height) = level_dims(luma.width, luma.height, level);
                Scratch::new::<f32>(context, width, height)
            })
            .collect();
        normalise(client, &sizing, search_clip, levels[0].plane(), self.bits);

        let planes = levels.iter().map(Scratch::plane).collect();
        let pyramid = Pyramid { levels: planes };
        build_pyramid(client, &sizing, &pyramid);

        levels
    }

    /// The neighbours of `field` up to `max_tr` away that exist, as `(delta, field)`, or `None` while
    /// one has not arrived or has no pyramid yet.
    fn search_neighbours(
        &self,
        cache: &FieldCache,
        field: usize,
    ) -> Result<Option<Vec<(i32, usize)>>, StageError> {
        let mut neighbours = Vec::new();
        for delta in neighbour_deltas(self.settings.max_tr) {
            let wanted = field as i64 + i64::from(delta);
            match cache.place(wanted) {
                Place::Outside => continue,
                Place::Coming => return Ok(None),
                Place::Dropped => return Err(lost_slot("pyramid", wanted as usize)),
                Place::Held => {},
            }

            let neighbour = wanted as usize;
            let state = state_of(cache, neighbour)?;
            if state.progress < Progress::Searched {
                return Ok(None);
            }

            neighbours.push((delta, neighbour));
        }

        Ok(Some(neighbours))
    }

    /// Searches `field` against each of its neighbours that exist once their pyramids are built, and flags
    /// the scene changes among them, answering whether it ran.
    ///
    /// The search to the next field is the repair's, when the repair ran one.
    pub(super) fn find_motion(
        &self,
        context: &FrameContext,
        cache: &mut FieldCache,
        field: usize,
    ) -> Result<bool, StageError> {
        let state = state_of(cache, field)?;
        if state.progress != Progress::Repaired {
            return Ok(false);
        }

        let Some(found) = self.search_neighbours(cache, field)? else {
            return Ok(false);
        };

        let Some(state) = cache.field_mut(field) else {
            return Err(lost_slot("state", field));
        };
        let mut reused = state.next_search.take();

        let client = self.rebuilder.client();
        let sizing = Sizing(context);
        let settings = &self.settings.motion;
        let fresh_count = found.len() - usize::from(reused.is_some());
        let mut fresh = cache.take_buffers(client, &self.grid, fresh_count).into_iter();
        let flags = match found.len() {
            0 => None,
            count => Some(cache.take_flags(client, count)),
        };

        let state = state_of(cache, field)?;
        let centre_levels = held(&state.pyramid, "pyramid", field)?;
        let centre = Pyramid {
            levels: centre_levels.iter().map(Scratch::plane).collect(),
        };

        let mut neighbours = Vec::with_capacity(found.len());
        for (delta, neighbour) in found {
            let reusable = match delta {
                1 => reused.take(),
                _ => None,
            };
            if let Some(buffers) = reusable {
                neighbours.push(Neighbour {
                    delta,
                    field: neighbour,
                    buffers,
                });
                continue;
            }

            let Some(buffers) = fresh.next() else {
                return Err(lost_slot("motion buffers", field));
            };
            let neighbour_state = state_of(cache, neighbour)?;
            let neighbour_levels = held(&neighbour_state.pyramid, "pyramid", neighbour)?;
            let searched = Pyramid {
                levels: neighbour_levels.iter().map(Scratch::plane).collect(),
            };

            let motion_field = buffers.field(settings.pel, false);
            search(client, &sizing, &centre, &searched, &motion_field, settings, None);

            neighbours.push(Neighbour {
                delta,
                field: neighbour,
                buffers,
            });
        }

        if let Some(flags) = &flags {
            let fields: Vec<_> = neighbours
                .iter()
                .map(|neighbour| neighbour.buffers.field(settings.pel, false))
                .collect();
            scene_change_flags(client, &fields, flags.plane(), self.settings.scene_change);
        }

        let Some(state) = cache.field_mut(field) else {
            return Err(lost_slot("state", field));
        };
        state.motion = Some(Motion { neighbours, flags });
        state.progress = Progress::Moved;

        Ok(true)
    }

    /// The neighbours of `motion` at a distance `keep` accepts, each with its plane `plane` of the slot
    /// `pick` reads.
    fn degrain_neighbours<'a>(
        &self,
        cache: &'a FieldCache,
        motion: &'a Motion,
        keep: impl Fn(i32) -> bool,
        pick: fn(&FieldState) -> &Option<Vec<Scratch>>,
        plane: usize,
    ) -> Result<Vec<DegrainNeighbour<'a>>, StageError> {
        let pel = self.settings.motion.pel;

        let mut chosen = Vec::new();
        let kept = motion
            .neighbours
            .iter()
            .enumerate()
            .filter(|(_, neighbour)| keep(neighbour.delta));
        for (index, neighbour) in kept {
            let flags = held(&motion.flags, "flags", neighbour.field)?;
            let state = state_of(cache, neighbour.field)?;
            let planes = held(pick(state), "planes", neighbour.field)?;

            chosen.push(DegrainNeighbour {
                plane: planes[plane].plane(),
                field: neighbour.buffers.field(pel, false),
                flag: flags.flag(index as u32),
            });
        }

        Ok(chosen)
    }

    /// Degrains `centre` of plane `plane` with `neighbours` into a new plane.
    fn degrained(
        &self,
        context: &FrameContext,
        centre: DevicePlane<'_>,
        neighbours: &[DegrainNeighbour<'_>],
        settings: &DegrainSettings,
        plane: usize,
    ) -> Scratch {
        let output = self.scratch(context, plane);
        let sizing = Sizing(context);
        let scale = self.shapes[plane].scale;
        let client = self.rebuilder.client();
        degrain(
            client,
            &sizing,
            centre,
            neighbours,
            output.plane(),
            settings,
            scale,
            plane > 0,
        );

        output
    }

    fn merged(
        &self,
        context: &FrameContext,
        first: DevicePlane<'_>,
        second: DevicePlane<'_>,
        weight: f32,
        plane: usize,
    ) -> Scratch {
        let output = self.scratch(context, plane);
        let sizing = Sizing(context);
        merge(
            self.rebuilder.client(),
            &sizing,
            first,
            second,
            output.plane(),
            weight,
        );

        output
    }

    /// The upper and lower temporal limits of `edi`, from its neighbours at ±1 compensated onto it.
    fn limits(
        &self,
        context: &FrameContext,
        cache: &FieldCache,
        motion: &Motion,
        edi: DevicePlane<'_>,
        plane: usize,
    ) -> Result<(Scratch, Scratch), StageError> {
        let client = self.rebuilder.client();
        let sizing = Sizing(context);
        let pel = self.settings.motion.pel;
        let scale = self.shapes[plane].scale;

        let near = motion
            .neighbours
            .iter()
            .enumerate()
            .filter(|(_, neighbour)| neighbour.delta.abs() == 1);
        let mut compensated = Vec::new();
        for (index, neighbour) in near {
            let flags = held(&motion.flags, "flags", neighbour.field)?;
            let state = state_of(cache, neighbour.field)?;
            let neighbour_edi = held(&state.edi, "edi", neighbour.field)?;

            let output = self.scratch(context, plane);
            let planes = CompensatePlanes {
                centre: edi,
                neighbour: neighbour_edi[plane].plane(),
                output: output.plane(),
            };
            let motion_field = neighbour.buffers.field(pel, false);
            let flag = flags.flag(index as u32);
            compensate(client, &sizing, planes, &motion_field, flag, scale);

            compensated.push(output);
        }

        let upper = self.scratch(context, plane);
        let lower = self.scratch(context, plane);
        let neighbour_planes: Vec<_> = compensated.iter().map(Scratch::plane).collect();
        temporal_limits(
            client,
            &sizing,
            edi,
            &neighbour_planes,
            upper.plane(),
            lower.plane(),
        );

        Ok((upper, lower))
    }

    /// TR1 on `edi`, `None` when TR1 is off and `edi` passes through.
    fn smoothed(
        &self,
        context: &FrameContext,
        cache: &FieldCache,
        motion: &Motion,
        edi: DevicePlane<'_>,
        plane: usize,
    ) -> Result<Option<Scratch>, StageError> {
        if self.settings.tr1 == 0 {
            return Ok(None);
        }

        let settings = &self.settings.tr1_degrain;
        let near = self.degrain_neighbours(cache, motion, |delta| delta.abs() == 1, edi_of, plane)?;
        let first = self.degrained(context, edi, &near, settings, plane);
        if self.settings.tr1 == 1 {
            let smoothed = self.merged(context, first.plane(), edi, TR1_EDI_WEIGHT_1, plane);
            return Ok(Some(smoothed));
        }

        let far = self.degrain_neighbours(cache, motion, |delta| delta.abs() == 2, edi_of, plane)?;
        let second = self.degrained(context, edi, &far, settings, plane);
        let blended = self.merged(context, first.plane(), second.plane(), TR1_SECOND_WEIGHT, plane);
        let smoothed = self.merged(context, blended.plane(), edi, TR1_EDI_WEIGHT_2, plane);

        Ok(Some(smoothed))
    }

    /// The resharpen and, on luma, the back-blend of `smoothed`, `None` when sharpening is off.
    fn sharpened(&self, context: &FrameContext, smoothed: DevicePlane<'_>, plane: usize) -> Option<Scratch> {
        let strength = self.settings.strength?;
        let client = self.rebuilder.client();
        let sizing = Sizing(context);

        let [first, second] = self.scratches(context, plane);
        let resharpened = self.scratch(context, plane);
        let scratch = StepScratch {
            first: first.plane(),
            second: second.plane(),
        };
        smode2_resharpen(
            client,
            &sizing,
            smoothed,
            scratch,
            resharpened.plane(),
            strength,
            self.bits,
        );

        if plane > 0 {
            return Some(resharpened);
        }

        let back_blended = self.scratch(context, plane);
        let planes = BackBlendPlanes {
            sharpened: resharpened.plane(),
            unsharpened: smoothed,
            scratch: StepScratch {
                first: first.plane(),
                second: second.plane(),
            },
            output: back_blended.plane(),
        };
        back_blend(client, &sizing, &self.tables, planes, self.bits);

        Some(back_blended)
    }

    /// The temporal limits, TR1, resharpen, back-blend and limit of one plane of `field`.
    fn limit_plane(
        &self,
        context: &FrameContext,
        cache: &FieldCache,
        field: usize,
        plane: usize,
    ) -> Result<Scratch, StageError> {
        let state = state_of(cache, field)?;
        let motion = held(&state.motion, "motion", field)?;
        let edi_planes = held(&state.edi, "edi", field)?;
        let edi = edi_planes[plane].plane();

        let (upper, lower) = self.limits(context, cache, motion, edi, plane)?;
        let smoothed = self.smoothed(context, cache, motion, edi, plane)?;
        let smoothed_plane = smoothed.as_ref().map_or(edi, Scratch::plane);
        let sharpened = self.sharpened(context, smoothed_plane, plane);
        let sharpened_plane = sharpened.as_ref().map_or(smoothed_plane, Scratch::plane);

        let limited = self.scratch(context, plane);
        let sizing = Sizing(context);
        clamp_between(
            self.rebuilder.client(),
            &sizing,
            sharpened_plane,
            upper.plane(),
            lower.plane(),
            limited.plane(),
        );

        Ok(limited)
    }

    /// Limits every plane of `field` once it has its motion, answering whether it ran.
    pub(super) fn limit(
        &self,
        context: &FrameContext,
        cache: &mut FieldCache,
        field: usize,
    ) -> Result<bool, StageError> {
        let state = state_of(cache, field)?;
        if state.progress != Progress::Moved {
            return Ok(false);
        }

        let motion = held(&state.motion, "motion", field)?;
        if !self.edi_ready(cache, motion) {
            return Ok(false);
        }

        let mut limited = Vec::with_capacity(self.shapes.len());
        for plane in 0..self.shapes.len() {
            let plane_limited = self.limit_plane(context, cache, field, plane)?;
            limited.push(plane_limited);
        }

        let Some(state) = cache.field_mut(field) else {
            return Err(lost_slot("state", field));
        };
        state.limited = Some(limited);
        state.progress = Progress::Limited;

        Ok(true)
    }

    /// Whether every neighbour the limits and TR1 read from `motion` has its final `edi`.
    fn edi_ready(&self, cache: &FieldCache, motion: &Motion) -> bool {
        let radius = self.settings.reach().smoothed as i32;

        motion
            .neighbours
            .iter()
            .filter(|neighbour| neighbour.delta.abs() <= radius)
            .all(|neighbour| match cache.field(neighbour.field) {
                Some(state) => state.progress >= Progress::Repaired,
                None => true,
            })
    }

    /// Whether every neighbour TR2 reads from `motion` has its limited planes.
    fn tr2_ready(&self, cache: &FieldCache, motion: &Motion) -> bool {
        let radius = self.settings.tr2 as i32;

        motion
            .neighbours
            .iter()
            .filter(|neighbour| neighbour.delta.abs() <= radius)
            .all(|neighbour| match cache.field(neighbour.field) {
                Some(state) => state.progress >= Progress::Limited,
                None => true,
            })
    }

    /// TR2 and Rep2 of one plane of `field`, narrowed into a plane of the output.
    fn finish_plane(
        &self,
        context: &FrameContext,
        cache: &FieldCache,
        field: usize,
        plane: usize,
        meta: &FrameMeta,
    ) -> Result<Plane, StageError> {
        let state = state_of(cache, field)?;
        let motion = held(&state.motion, "motion", field)?;
        let limited_planes = held(&state.limited, "limited planes", field)?;
        let edi_planes = held(&state.edi, "edi", field)?;
        let limited = limited_planes[plane].plane();
        let radius = self.settings.tr2 as i32;

        let stable = match radius {
            0 => None,
            _ => {
                let within = |delta: i32| delta.abs() <= radius;
                let neighbours = self.degrain_neighbours(cache, motion, within, limited_of, plane)?;
                let settings = &self.settings.tr2_degrain;
                let degrained = self.degrained(context, limited, &neighbours, settings, plane);
                Some(degrained)
            },
        };
        let stable_plane = stable.as_ref().map_or(limited, Scratch::plane);

        let client = self.rebuilder.client();
        let sizing = Sizing(context);
        let [difference, first, positive_choke, negative_choke] = self.scratches(context, plane);
        let output = self.scratch(context, plane);
        let repair = ShimmerPlanes {
            input: stable_plane,
            reference: edi_planes[plane].plane(),
            difference: difference.plane(),
            first: first.plane(),
            positive_choke: positive_choke.plane(),
            negative_choke: negative_choke.plane(),
            output: output.plane(),
        };
        shimmer(client, &sizing, repair, self.bits);

        let channel = self.input.channels()[plane];

        self.rebuilder.stored(context, &output, meta, channel)
    }

    /// Builds the output of `field` once the neighbours TR2 reads are limited, answering whether it ran.
    ///
    /// A field that is not shown is finished without building anything.
    pub(super) fn finish(
        &self,
        context: &FrameContext,
        cache: &mut FieldCache,
        field: usize,
    ) -> Result<bool, StageError> {
        let state = state_of(cache, field)?;
        if state.progress != Progress::Limited {
            return Ok(false);
        }

        let motion = held(&state.motion, "motion", field)?;
        if !self.tr2_ready(cache, motion) {
            return Ok(false);
        }

        let output = match &state.meta {
            Some(meta) => {
                let mut planes = Vec::with_capacity(self.shapes.len());
                for plane in 0..self.shapes.len() {
                    let narrowed = self.finish_plane(context, cache, field, plane, meta)?;
                    planes.push(narrowed);
                }

                let built = Frame::from_planes(self.output, planes, meta.clone());
                Some(built.map_err(bad_frame)?)
            },
            None => None,
        };

        let Some(state) = cache.field_mut(field) else {
            return Err(lost_slot("state", field));
        };
        state.output = output;
        state.progress = Progress::Finished;

        Ok(true)
    }
}
