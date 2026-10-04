use av_toolbox_core::StageError;

/// The smallest block either `ivtc` stage counts in, along either axis.
const SMALLEST_BLOCK: u32 = 4;

/// The largest block either `ivtc` stage counts in, along either axis.
const LARGEST_BLOCK: u32 = 512;

fn not_a_block(stage: &str, name: &str, value: u32) -> StageError {
    let message = format!(
        "{stage} needs {name} to be a power of 2 between {SMALLEST_BLOCK} and {LARGEST_BLOCK}, got {value}"
    );

    StageError::msg(message)
}

/// Refuses a block size `value` unless it is a power of 2 between 4 and 512.
pub(crate) fn checked_block(stage: &str, name: &str, value: u32) -> Result<(), StageError> {
    let in_range = (SMALLEST_BLOCK..=LARGEST_BLOCK).contains(&value);
    if in_range && value.is_power_of_two() {
        return Ok(());
    }

    Err(not_a_block(stage, name, value))
}
