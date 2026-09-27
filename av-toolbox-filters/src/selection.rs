use av_toolbox_core::StageError;
use av_toolbox_gpu::{Channel, FrameLayout};

use crate::planes::ChannelName;

/// Which planes a stage works on when `args.planes` is not given.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub(crate) enum PlaneDefault {
    First,
    Every,
}

fn no_planes(stage: &str) -> StageError {
    let message = format!("{stage} needs at least one plane in planes");

    StageError::msg(message)
}

fn listed_twice(stage: &str, channel: Channel) -> StageError {
    let message = format!("{stage} lists {channel} twice in planes");

    StageError::msg(message)
}

fn not_carried(stage: &str, channel: Channel, layout: FrameLayout) -> StageError {
    let message = format!("{stage} cannot work on {channel} of a {layout} clip");

    StageError::msg(message)
}

/// Whether each plane of `layout`, in channel order, is worked on.
pub(crate) fn selected_planes(
    stage: &str,
    requested: Option<&[ChannelName]>,
    layout: FrameLayout,
    default: PlaneDefault,
) -> Result<Vec<bool>, StageError> {
    let channels = layout.channels();

    let Some(names) = requested else {
        let chosen = (0..channels.len())
            .map(|index| default == PlaneDefault::Every || index == 0)
            .collect();

        return Ok(chosen);
    };

    if names.is_empty() {
        return Err(no_planes(stage));
    }

    let mut chosen = vec![false; channels.len()];
    for name in names {
        let channel = name.channel();
        let Some(index) = channels.iter().position(|carried| *carried == channel) else {
            return Err(not_carried(stage, channel, layout));
        };

        if chosen[index] {
            return Err(listed_twice(stage, channel));
        }

        chosen[index] = true;
    }

    Ok(chosen)
}

#[cfg(test)]
mod tests {
    use av_toolbox_gpu::{FrameLayout, Subsampling};

    use super::*;

    const STAGE: &str = "test/stage";

    fn yuv420() -> FrameLayout {
        FrameLayout::Yuv {
            subsampling: Subsampling::Yuv420,
        }
    }

    fn refusal(requested: &[ChannelName]) -> String {
        let layout = yuv420();
        let error = selected_planes(STAGE, Some(requested), layout, PlaneDefault::First)
            .expect_err("should be refused");

        error.to_string()
    }

    #[test]
    fn the_first_plane_is_the_default_for_every_layout() {
        let chroma = FrameLayout::Chroma {
            subsampling: Subsampling::Yuv420,
        };
        let cases = [
            (yuv420(), vec![true, false, false]),
            (FrameLayout::Rgb, vec![true, false, false]),
            (FrameLayout::Grey, vec![true]),
            (chroma, vec![true, false]),
        ];

        for (layout, expected) in cases {
            let selected = selected_planes(STAGE, None, layout, PlaneDefault::First).expect("valid");

            assert_eq!(selected, expected, "{layout}");
        }
    }

    #[test]
    fn every_plane_can_be_the_default() {
        let layout = yuv420();
        let selected = selected_planes(STAGE, None, layout, PlaneDefault::Every).expect("valid");

        assert_eq!(selected, vec![true, true, true]);
    }

    #[test]
    fn named_planes_are_selected_in_channel_order() {
        let requested = [ChannelName::V, ChannelName::U];
        let layout = yuv420();
        let selected = selected_planes(STAGE, Some(&requested), layout, PlaneDefault::First).expect("valid");

        assert_eq!(selected, vec![false, true, true]);
    }

    #[test]
    fn an_empty_list_is_refused() {
        let message = refusal(&[]);

        assert!(message.contains(STAGE) && message.contains("planes"), "{message}");
    }

    #[test]
    fn a_plane_listed_twice_is_refused() {
        let message = refusal(&[ChannelName::U, ChannelName::U]);

        assert!(message.contains("U") && message.contains("twice"), "{message}");
    }

    #[test]
    fn a_plane_the_layout_does_not_carry_is_refused() {
        let message = refusal(&[ChannelName::R]);

        assert!(message.contains("R") && message.contains(STAGE), "{message}");
    }
}
