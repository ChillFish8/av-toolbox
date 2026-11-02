use std::collections::VecDeque;

use av_toolbox_cubek::motion::{BlockGrid, FieldBuffers, SceneFlags};
use av_toolbox_gpu::{Client, Frame, FrameMeta};

use crate::scratch::Scratch;

/// How far through the chain a field is, each step in the order it runs.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd)]
pub(super) enum Progress {
    /// The field has its bob and `edi`.
    Interpolated,
    /// The field has its search pyramid.
    Searched,
    /// The field has its final `edi`, which repair input rebuilds where the frame moves.
    Repaired,
    /// The field has its motion to every neighbour that exists.
    Moved,
    /// The field has its limited planes.
    Limited,
    /// The field's output is built, or skipped when it is not shown.
    Finished,
}

/// The search from a field to one of its neighbours.
pub(super) struct Neighbour {
    /// How many fields after the centre the neighbour is, negative for fields before it.
    pub(super) delta: i32,
    /// The neighbour's field number.
    pub(super) field: usize,
    pub(super) buffers: FieldBuffers,
}

/// A field's searches, in the order they were run, and their scene-change flags in the same order.
pub(super) struct Motion {
    pub(super) neighbours: Vec<Neighbour>,
    /// `None` when the field has no neighbours.
    pub(super) flags: Option<SceneFlags>,
}

/// One field frame of the chunk, with a slot for each step's result.
pub(super) struct FieldState {
    /// The metadata of the field's output, `None` when the field is not shown.
    pub(super) meta: Option<FrameMeta>,
    pub(super) progress: Progress,
    /// The bob of the field's luma.
    pub(super) bob: Option<Scratch>,
    /// Every level of the field's normalised search clip.
    pub(super) pyramid: Option<Vec<Scratch>>,
    /// The field with its missing rows rebuilt, one plane per channel.
    pub(super) edi: Option<Vec<Scratch>>,
    /// The luma rebuilt from the field a repaired frame keeps, until the repair blends it in.
    pub(super) rebuilt: Option<Scratch>,
    /// The repair's search to the next field, which the motion search takes over.
    pub(super) next_search: Option<FieldBuffers>,
    /// The repair mask's scene-change flag, until the step that made it has finished.
    pub(super) mask_flag: Option<SceneFlags>,
    pub(super) motion: Option<Motion>,
    /// The field after the limit, one plane per channel.
    pub(super) limited: Option<Vec<Scratch>>,
    pub(super) output: Option<Frame>,
}

impl FieldState {
    pub(super) fn new(meta: Option<FrameMeta>, bob: Scratch, edi: Vec<Scratch>) -> Self {
        Self {
            meta,
            progress: Progress::Interpolated,
            bob: Some(bob),
            pyramid: None,
            edi: Some(edi),
            rebuilt: None,
            next_search: None,
            mask_flag: None,
            motion: None,
            limited: None,
            output: None,
        }
    }

    /// The same field, holding `rebuilt` for its repair.
    pub(super) fn with_rebuilt(self, rebuilt: Scratch) -> Self {
        Self {
            rebuilt: Some(rebuilt),
            ..self
        }
    }

    fn is_empty(&self) -> bool {
        self.bob.is_none()
            && self.pyramid.is_none()
            && self.edi.is_none()
            && self.rebuilt.is_none()
            && self.next_search.is_none()
            && self.mask_flag.is_none()
            && self.motion.is_none()
            && self.limited.is_none()
            && self.output.is_none()
    }
}

/// Where a field number falls in the chunk.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub(super) enum Place {
    /// Before the chunk's first field, or past its last once the chunk has ended.
    Outside,
    /// A field already finished and dropped.
    Dropped,
    /// A field the cache holds.
    Held,
    /// A field that has not arrived yet.
    Coming,
}

/// The temporal radii each slot is read across.
#[derive(Debug, Copy, Clone)]
pub(super) struct Reach {
    pub(super) tr0: u32,
    /// How far apart the furthest motion neighbours are.
    pub(super) max_tr: u32,
    /// How far apart the neighbours whose `edi` the limits and TR1 read are.
    pub(super) smoothed: u32,
    pub(super) tr2: u32,
}

/// The rolling window of field frames of one chunk, keyed by field number.
pub(super) struct FieldCache {
    fields: VecDeque<FieldState>,
    /// The field number of the first field held.
    first: usize,
    /// Whether every field of the chunk has arrived.
    ended: bool,
    /// The number of the next field whose output is due.
    next_output: usize,
    spare_buffers: Vec<FieldBuffers>,
    spare_flags: Vec<SceneFlags>,
}

impl FieldCache {
    pub(super) fn new() -> Self {
        Self {
            fields: VecDeque::new(),
            first: 0,
            ended: false,
            next_output: 0,
            spare_buffers: Vec::new(),
            spare_flags: Vec::new(),
        }
    }

    /// Empties the cache for a new chunk, keeping its spare motion buffers.
    pub(super) fn restart(&mut self) {
        let fields: Vec<FieldState> = self.fields.drain(..).collect();
        for state in fields {
            self.spare_buffers.extend(state.next_search);
            self.spare_flags.extend(state.mask_flag);
            self.recycle_motion(state.motion);
        }

        self.first = 0;
        self.ended = false;
        self.next_output = 0;
    }

    pub(super) fn push(&mut self, state: FieldState) {
        self.fields.push_back(state);
    }

    pub(super) fn end(&mut self) {
        self.ended = true;
    }

    /// How many fields of the chunk have arrived.
    pub(super) fn count(&self) -> usize {
        self.first + self.fields.len()
    }

    /// The numbers of every field held.
    pub(super) fn held(&self) -> std::ops::Range<usize> {
        self.first..self.count()
    }

    pub(super) fn place(&self, field: i64) -> Place {
        if field < 0 {
            return Place::Outside;
        }

        let field = field as usize;
        if field < self.first {
            return Place::Dropped;
        }

        if field < self.count() {
            return Place::Held;
        }

        if self.ended { Place::Outside } else { Place::Coming }
    }

    /// The last field of the chunk, once the chunk has ended.
    pub(super) fn last(&self) -> Option<usize> {
        let count = self.count();
        if !self.ended || count == 0 {
            return None;
        }

        Some(count - 1)
    }

    pub(super) fn field(&self, field: usize) -> Option<&FieldState> {
        let index = field.checked_sub(self.first)?;

        self.fields.get(index)
    }

    pub(super) fn field_mut(&mut self, field: usize) -> Option<&mut FieldState> {
        let index = field.checked_sub(self.first)?;

        self.fields.get_mut(index)
    }

    /// How far `field` is through the chain, finished once it is dropped.
    fn progress(&self, field: usize) -> Progress {
        match self.field(field) {
            Some(state) => state.progress,
            None => Progress::Finished,
        }
    }

    /// Whether a field within `radius` of `field` has yet to pass `step`, or may still arrive.
    fn awaited(&self, field: usize, radius: u32, step: Progress) -> bool {
        let centre = field as i64;
        let radius = i64::from(radius);

        (centre - radius..=centre + radius).any(|neighbour| match self.place(neighbour) {
            Place::Outside | Place::Dropped => false,
            Place::Coming => true,
            Place::Held => self.progress(neighbour as usize) < step,
        })
    }

    /// Drops every slot no later step reads, and every field left with nothing in it.
    pub(super) fn release(&mut self, reach: Reach) {
        for field in self.held() {
            let keep_bob = self.awaited(field, reach.tr0, Progress::Searched);
            let keep_pyramid = self.awaited(field, reach.max_tr, Progress::Moved);
            let keep_edi = self.awaited(field, reach.smoothed, Progress::Limited)
                || self.progress(field) < Progress::Finished;
            let keep_motion = self.progress(field) < Progress::Finished;
            let keep_mask_flag = self.progress(field) < Progress::Repaired;
            let keep_limited = self.awaited(field, reach.tr2, Progress::Finished);

            let Some(state) = self.field_mut(field) else {
                continue;
            };

            if !keep_bob {
                state.bob = None;
            }

            if !keep_pyramid {
                state.pyramid = None;
            }

            if !keep_edi {
                state.edi = None;
            }

            if !keep_limited {
                state.limited = None;
            }

            let mask_flag = match keep_mask_flag {
                true => None,
                false => state.mask_flag.take(),
            };
            let motion = match keep_motion {
                true => None,
                false => state.motion.take(),
            };
            self.spare_flags.extend(mask_flag);
            self.recycle_motion(motion);
        }

        while let Some(front) = self.fields.front() {
            let emitted = self.first < self.next_output;
            if !emitted || !front.is_empty() {
                break;
            }

            self.fields.pop_front();
            self.first += 1;
        }
    }

    /// Takes the outputs due, in field order, up to the first field not yet finished.
    pub(super) fn take_outputs(&mut self) -> Vec<Frame> {
        let mut outputs = Vec::new();
        while self.next_output < self.count() {
            let Some(state) = self.field_mut(self.next_output) else {
                break;
            };

            if state.progress < Progress::Finished {
                break;
            }

            if let Some(output) = state.output.take() {
                outputs.push(output);
            }

            self.next_output += 1;
        }

        outputs
    }

    /// `count` motion buffers on `grid`, reusing spare ones first.
    pub(super) fn take_buffers(
        &mut self,
        client: &Client,
        grid: &BlockGrid,
        count: usize,
    ) -> Vec<FieldBuffers> {
        let mut buffers = Vec::with_capacity(count);
        for _ in 0..count {
            let spare = self.spare_buffers.pop();
            let taken = spare.unwrap_or_else(|| FieldBuffers::new(client, grid));
            buffers.push(taken);
        }

        buffers
    }

    /// Flags for `count` fields, reusing a spare set of that size when there is one.
    pub(super) fn take_flags(&mut self, client: &Client, count: usize) -> SceneFlags {
        let found = self
            .spare_flags
            .iter()
            .position(|flags| flags.plane().width() as usize == count);

        match found {
            Some(index) => self.spare_flags.swap_remove(index),
            None => SceneFlags::new(client, count as u32),
        }
    }

    fn recycle_motion(&mut self, motion: Option<Motion>) {
        let Some(motion) = motion else {
            return;
        };

        let buffers = motion.neighbours.into_iter().map(|neighbour| neighbour.buffers);
        self.spare_buffers.extend(buffers);

        if let Some(flags) = motion.flags {
            self.spare_flags.push(flags);
        }
    }
}
