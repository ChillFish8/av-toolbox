use av_toolbox_core::StageError;
use av_toolbox_cubek::DevicePlane;
use av_toolbox_cubek::deinterlace::qtgmc::{Parity, repair_blend, vertical_blur};
use av_toolbox_cubek::motion::{
    FieldBuffers,
    Pyramid,
    SadMaskTables,
    SceneFlags,
    sad_mask,
    scene_change_flags,
    search,
};
use av_toolbox_gpu::{Frame, FrameContext, FrameMeta, Runtime};

use super::cache::{FieldCache, FieldState, Place, Progress};
use super::chain::{Chain, held, lost_slot, no_prediction, state_of};
use crate::deinterlace::qtgmc::STAGE;
use crate::deinterlace::rebuild::channel_plane;
use crate::deinterlace::spatial::Predictor;
use crate::scratch::Scratch;
use crate::sizing::Sizing;

fn no_luma() -> StageError {
    let message = format!("{STAGE} was handed a frame without a luma plane");

    StageError::msg(message)
}

fn no_mask_tables() -> StageError {
    let message = format!("{STAGE} has no SAD mask tables to repair with");

    StageError::msg(message)
}

/// The parity of the field that the frame at stream index `index` of repair input keeps.
///
/// Frames alternate from `first`, the parity of the field shown first, which reversed input swaps. Counting
/// from the stream follows QTGMC's `SelectEvery(4, [0, 3])`, which runs over the whole clip.
pub(super) fn kept_parity(first: Parity, reversed: bool, index: u64) -> Parity {
    let starting = match reversed {
        true => first.other(),
        false => first,
    };

    match index.is_multiple_of(2) {
        true => starting,
        false => starting.other(),
    }
}

impl<R: Runtime> Chain<R> {
    /// The entry of one progressive `frame`, shown with `meta`.
    ///
    /// Its `edi` is the frame. Its bob is the frame's luma for progressive input, and a vertical 1-2-1 blur
    /// of it for repair input, which also rebuilds the luma from the field of parity `kept`.
    pub(super) fn entry_of(
        &self,
        context: &FrameContext,
        frame: &Frame,
        kept: Parity,
        meta: FrameMeta,
    ) -> Result<FieldState, StageError> {
        let channels = self.input.channels();
        let mut edi = Vec::with_capacity(channels.len());
        for channel in channels {
            let source = channel_plane(STAGE, frame, *channel)?;
            let widened = self.rebuilder.widened(context, source, *channel);
            edi.push(widened);
        }

        let Some(luma) = edi.first() else {
            return Err(no_luma());
        };

        if !self.mode.is_repair() {
            let source = channel_plane(STAGE, frame, channels[0])?;
            let bob = self.rebuilder.widened(context, source, channels[0]);
            return Ok(FieldState::new(Some(meta), bob, edi));
        }

        let bob = self.blurred(context, luma.plane());
        let rebuilt = self.rebuilt_luma(context, luma.plane(), kept)?;
        let state = FieldState::new(Some(meta), bob, edi);

        Ok(state.with_rebuilt(rebuilt))
    }

    fn blurred(&self, context: &FrameContext, luma: DevicePlane<'_>) -> Scratch {
        let output = self.scratch(context, 0);
        let sizing = Sizing(context);
        vertical_blur(self.rebuilder.client(), &sizing, luma, output.plane());

        output
    }

    /// `luma` with the field of parity `kept` kept and the other rows rebuilt by the spatial predictor.
    fn rebuilt_luma(
        &self,
        context: &FrameContext,
        luma: DevicePlane<'_>,
        kept: Parity,
    ) -> Result<Scratch, StageError> {
        if let Predictor::Cubic = self.predictor {
            let bobbed = self.bobbed(context, luma, 0, kept);
            return Ok(bobbed);
        }

        let rebuilt = self.rebuilder.one_field(context, 0, luma, kept.row_offset());
        rebuilt.ok_or_else(|| no_prediction(0))
    }

    /// Gives `field` its final `edi` once it has its pyramid, answering whether it ran.
    ///
    /// For repair input, it waits for the next field's pyramid and searches from `field` to it. The luma
    /// then blends towards the rebuilt luma through MVTools' SAD mask of that search, which a scene change
    /// under the mask's own rule sets to 0. The last field of a chunk keeps its own luma, and chroma is
    /// always the frame's own.
    pub(super) fn repair(
        &self,
        context: &FrameContext,
        cache: &mut FieldCache,
        field: usize,
    ) -> Result<bool, StageError> {
        let state = state_of(cache, field)?;
        if state.progress != Progress::Searched {
            return Ok(false);
        }

        if !self.mode.is_repair() {
            return Self::finish_repair(cache, field);
        }

        let next = field + 1;
        let place = cache.place(next as i64);
        let ready = match place {
            Place::Coming => false,
            Place::Dropped => return Err(lost_slot("pyramid", next)),
            Place::Outside => true,
            Place::Held => state_of(cache, next)?.progress >= Progress::Searched,
        };
        if !ready {
            return Ok(false);
        }

        if place == Place::Held {
            self.blend_towards_rebuild(context, cache, field, next)?;
        }

        Self::finish_repair(cache, field)
    }

    /// Marks `field` repaired, dropping its rebuild.
    fn finish_repair(cache: &mut FieldCache, field: usize) -> Result<bool, StageError> {
        let Some(state) = cache.field_mut(field) else {
            return Err(lost_slot("state", field));
        };
        state.rebuilt = None;
        state.progress = Progress::Repaired;

        Ok(true)
    }

    /// Searches from `field` to `next`, and blends the luma of `field` towards its rebuild through the SAD
    /// mask of that search.
    fn blend_towards_rebuild(
        &self,
        context: &FrameContext,
        cache: &mut FieldCache,
        field: usize,
        next: usize,
    ) -> Result<(), StageError> {
        let Some(tables) = &self.mask_tables else {
            return Err(no_mask_tables());
        };

        let client = self.rebuilder.client();
        let sizing = Sizing(context);
        let settings = &self.settings.motion;
        let taken = cache.take_buffers(client, &self.grid, 1).pop();
        let Some(buffers) = taken else {
            return Err(lost_slot("motion buffers", field));
        };
        let flags = cache.take_flags(client, 1);

        let state = state_of(cache, field)?;
        let next_state = state_of(cache, next)?;
        let centre_levels = held(&state.pyramid, "pyramid", field)?;
        let next_levels = held(&next_state.pyramid, "pyramid", next)?;
        let centre = pyramid_of(centre_levels);
        let searched = pyramid_of(next_levels);
        let motion_field = buffers.field(settings.pel, false);
        search(client, &sizing, &centre, &searched, &motion_field, settings, None);
        scene_change_flags(
            client,
            &[motion_field],
            flags.plane(),
            self.settings.mask_scene_change,
        );

        let mask = self.masked(context, tables, &buffers, &flags);
        let edi = held(&state.edi, "edi", field)?;
        let rebuilt = held(&state.rebuilt, "rebuilt luma", field)?;
        let blended = self.scratch(context, 0);
        repair_blend(
            client,
            &sizing,
            edi[0].plane(),
            rebuilt.plane(),
            mask.plane(),
            blended.plane(),
        );

        let Some(state) = cache.field_mut(field) else {
            return Err(lost_slot("state", field));
        };
        let Some(edi) = state.edi.as_mut() else {
            return Err(lost_slot("edi", field));
        };
        edi[0] = blended;
        state.next_search = Some(buffers);
        state.mask_flag = Some(flags);

        Ok(())
    }

    /// The SAD mask of the search in `buffers`, flagged by the only flag of `flags`.
    fn masked(
        &self,
        context: &FrameContext,
        tables: &SadMaskTables,
        buffers: &FieldBuffers,
        flags: &SceneFlags,
    ) -> Scratch {
        let output = self.scratch(context, 0);
        let sizing = Sizing(context);
        let motion_field = buffers.field(self.settings.motion.pel, false);
        let flag = flags.flag(0);
        sad_mask(
            self.rebuilder.client(),
            &sizing,
            &motion_field,
            flag,
            tables,
            output.plane(),
        );

        output
    }
}

fn pyramid_of(levels: &[Scratch]) -> Pyramid<'_> {
    let planes = levels.iter().map(Scratch::plane).collect();

    Pyramid { levels: planes }
}
