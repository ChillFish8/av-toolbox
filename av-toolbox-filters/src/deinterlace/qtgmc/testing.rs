use std::sync::{Arc, Mutex};

use av_toolbox_core::{Filter, StageError};
use av_toolbox_gpu::{BuildContext, Client, FrameFormat};
use cubecl::prelude::CubeElement;
use cubecl::server::{Handle, ServerError};
use serde::{Deserialize, Deserializer};

use super::cache::FieldCache;
use super::chain::Chain;
use super::{QtgmcArgs, STAGE, deinterlacer};

fn unreadable_args(error: impl std::fmt::Display) -> StageError {
    let message = format!("{STAGE} could not read its arguments: {error}");

    StageError::msg(message)
}

fn unreadable_motion(error: ServerError) -> StageError {
    let message = format!("{STAGE} could not read its motion back: {error}");

    StageError::msg(message)
}

fn missing_motion(field: usize) -> StageError {
    let message = format!("{STAGE} has no motion recorded for field {field}");

    StageError::msg(message)
}

/// One motion search the stage ran, read back from the device.
#[derive(Debug, Clone, PartialEq)]
pub struct QtgmcMotion {
    /// The centre's field number within its chunk.
    pub field: usize,
    /// How many fields after the centre the neighbour is, negative for fields before it.
    pub delta: i32,
    /// Each block's vector as an `x, y` pair, in half pixels.
    pub vectors: Vec<i32>,
    pub sads: Vec<f32>,
    /// Whether the neighbour was flagged as a scene change.
    pub flagged: bool,
}

/// The scene-change flag of one repair mask the stage built, read back from the device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QtgmcRepairFlag {
    /// The repaired field's number within its chunk.
    pub field: usize,
    pub flagged: bool,
}

/// Every motion search and repair mask flag a stage built by [qtgmc_with_motion_log] ran, in the order it
/// ran them.
#[derive(Debug, Clone, Default)]
pub struct QtgmcMotionLog {
    searches: Arc<Mutex<Vec<QtgmcMotion>>>,
    repairs: Arc<Mutex<Vec<QtgmcRepairFlag>>>,
}

impl QtgmcMotionLog {
    /// Takes every search recorded so far.
    pub fn take(&self) -> Vec<QtgmcMotion> {
        let mut searches = self.searches.lock().unwrap_or_else(|error| error.into_inner());

        std::mem::take(&mut *searches)
    }

    /// Takes every repair mask flag recorded so far.
    pub fn take_repairs(&self) -> Vec<QtgmcRepairFlag> {
        let mut repairs = self.repairs.lock().unwrap_or_else(|error| error.into_inner());

        std::mem::take(&mut *repairs)
    }

    /// Reads the repair mask flag of `field` back and records it, when its repair built a mask.
    pub(super) fn record_repair(
        &self,
        chain: &Chain,
        cache: &FieldCache,
        field: usize,
    ) -> Result<(), StageError> {
        let Some(flags) = cache.field(field).and_then(|state| state.mask_flag.as_ref()) else {
            return Ok(());
        };

        let client = chain.rebuilder.client();
        let read_flags = read::<u32>(client, flags.plane().handle())?;

        let flagged = read_flags.first().is_some_and(|flag| *flag != 0);
        let mut repairs = self.repairs.lock().unwrap_or_else(|error| error.into_inner());
        repairs.push(QtgmcRepairFlag { field, flagged });

        Ok(())
    }

    /// Reads the motion of `field` back and records each of its searches.
    pub(super) fn record(&self, chain: &Chain, cache: &FieldCache, field: usize) -> Result<(), StageError> {
        let Some(motion) = cache.field(field).and_then(|state| state.motion.as_ref()) else {
            return Err(missing_motion(field));
        };
        let client = chain.rebuilder.client();
        let pel = chain.settings.motion.pel;

        let flags = match &motion.flags {
            Some(flags) => read::<u32>(client, flags.plane().handle())?,
            None => Vec::new(),
        };

        let mut found = Vec::with_capacity(motion.neighbours.len());
        for (neighbour, flag) in motion.neighbours.iter().zip(&flags) {
            let planes = neighbour.buffers.field(pel, false).planes;
            let vectors = read::<i32>(client, planes.vectors.handle())?;
            let sads = read::<f32>(client, planes.sads.handle())?;

            found.push(QtgmcMotion {
                field,
                delta: neighbour.delta,
                vectors,
                sads,
                flagged: *flag != 0,
            });
        }

        let mut searches = self.searches.lock().unwrap_or_else(|error| error.into_inner());
        searches.extend(found);

        Ok(())
    }
}

fn read<E: CubeElement>(client: &Client, handle: &Handle) -> Result<Vec<E>, StageError> {
    let bytes = client.read_one(handle.clone()).map_err(unreadable_motion)?;

    Ok(E::from_bytes(&bytes).to_vec())
}

/// Builds `deinterlace/qtgmc` with `args` for frames of `input`, recording every motion search it runs.
pub fn qtgmc_with_motion_log<'de, D: Deserializer<'de>>(
    context: &BuildContext,
    args: D,
    input: FrameFormat,
) -> Result<(Box<dyn Filter>, QtgmcMotionLog), StageError> {
    let parsed = QtgmcArgs::deserialize(args);
    let args = parsed.map_err(unreadable_args)?;

    let log = QtgmcMotionLog::default();
    let mut built = deinterlacer(context, &args, input)?;
    built.log = Some(log.clone());

    Ok((Box::new(built), log))
}
