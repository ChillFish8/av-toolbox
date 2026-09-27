mod extract;
mod merge;
#[cfg(test)]
mod tests;

use av_toolbox_gpu::Channel;
use schemars::JsonSchema;
use serde::Deserialize;

pub(crate) use self::extract::Extract;
pub(crate) use self::merge::Merge;

/// One channel, as a stage argument names it.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub(crate) enum ChannelName {
    Y,
    U,
    V,
    R,
    G,
    B,
}

impl ChannelName {
    pub(crate) const fn channel(self) -> Channel {
        match self {
            Self::Y => Channel::Y,
            Self::U => Channel::U,
            Self::V => Channel::V,
            Self::R => Channel::R,
            Self::G => Channel::G,
            Self::B => Channel::B,
        }
    }
}

/// What `planes/extract` pulls out, one channel or the chroma pair.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub(crate) enum PlaneName {
    Y,
    U,
    V,
    R,
    G,
    B,
    Chroma,
}
