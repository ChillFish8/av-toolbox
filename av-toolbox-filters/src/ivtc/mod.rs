mod checks;
pub(crate) mod decimate;
mod decimation;
mod decision;
pub(crate) mod matching;
mod metrics;
#[cfg(feature = "test-support")]
mod testing;
#[cfg(test)]
mod tests;

pub(crate) use self::decimate::Decimate;
pub(crate) use self::matching::Match;
#[cfg(feature = "test-support")]
pub use self::testing::VfmDecision;
