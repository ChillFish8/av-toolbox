mod extract;
#[cfg(test)]
mod tests;

use av_toolbox_gpu::Channel;
use schemars::JsonSchema;
use serde::Deserialize;

pub(crate) use self::extract::Extract;

/// One channel, as a stage argument names it.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
#[expect(dead_code, reason = "no stage in this crate takes a ChannelName argument yet")]
pub(crate) enum ChannelName {
    Y,
    U,
    V,
    R,
    G,
    B,
}

impl ChannelName {
    #[expect(dead_code, reason = "no stage in this crate takes a ChannelName argument yet")]
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
