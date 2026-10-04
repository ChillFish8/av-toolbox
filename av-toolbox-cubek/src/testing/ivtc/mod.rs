mod reference;
mod synthetic;

pub use self::reference::{
    IntPlane,
    reference_difference,
    reference_match_weave,
    reference_weave,
    scene_difference,
};
pub use self::synthetic::{SyntheticClip, synthetic_clips};

#[cfg(test)]
mod tests {
    use super::*;

    const CLIPS: &str = include_str!("../fixtures/ivtc/clips.txt");
    const SCENE: &str = include_str!("../fixtures/ivtc/scene.txt");
    const WIDTH: usize = 64;
    const HEIGHT: usize = 32;

    struct FixtureClip {
        bits: u32,
        order: u32,
        frames: usize,
        checksum: u64,
    }

    fn fixture_clip(name: &str) -> FixtureClip {
        let line = CLIPS
            .lines()
            .find(|line| line.split_whitespace().nth(1) == Some(name))
            .unwrap_or_else(|| panic!("clips.txt has no clip {name}"));
        let words: Vec<&str> = line.split_whitespace().collect();

        FixtureClip {
            bits: words[3].parse().expect("bits"),
            order: words[5].parse().expect("order"),
            frames: words[7].parse().expect("frames"),
            checksum: words[9].parse().expect("checksum"),
        }
    }

    fn fixture_scene(name: &str, frame: usize) -> f64 {
        let frame_text = frame.to_string();
        let line = SCENE
            .lines()
            .find(|line| {
                let words: Vec<&str> = line.split_whitespace().collect();
                words[1] == name && words[2] == frame_text
            })
            .unwrap_or_else(|| panic!("scene.txt has no {name} frame {frame}"));

        line.split_whitespace()
            .nth(4)
            .expect("diff")
            .parse()
            .expect("diff is an f64")
    }

    #[test]
    fn every_synthetic_clip_matches_the_harness() {
        for clip in synthetic_clips() {
            let expected = fixture_clip(clip.name);
            let checksum = clip
                .frames
                .iter()
                .flatten()
                .flatten()
                .fold(0u64, |sum, sample| sum.wrapping_add(*sample as u64));

            assert_eq!(clip.bits, expected.bits, "{}", clip.name);
            assert_eq!(clip.order, expected.order, "{}", clip.name);
            assert_eq!(clip.frames.len(), expected.frames, "{}", clip.name);
            assert_eq!(checksum, expected.checksum, "{}", clip.name);
        }
    }

    #[test]
    fn every_scene_difference_matches_the_harness() {
        for clip in synthetic_clips() {
            for frame in 0..clip.frames.len() {
                let current = &clip.frames[frame][0];
                let before = &clip.frames[frame.saturating_sub(1)][0];
                let sum = reference_difference(current, before);
                let found = scene_difference(sum, WIDTH, HEIGHT, clip.bits);
                let expected = fixture_scene(clip.name, frame);

                assert_eq!(
                    found.to_bits(),
                    expected.to_bits(),
                    "{} frame {frame}: {found} against {expected}",
                    clip.name,
                );
            }
        }
    }

    #[test]
    fn a_match_weave_keeps_the_matching_rows_of_the_right_frame() {
        let clips = synthetic_clips();
        let clip = clips
            .iter()
            .find(|clip| clip.name == "tff_8")
            .expect("tff_8 exists");
        let previous = &clip.frames[0][0];
        let current = &clip.frames[1][0];
        let next = &clip.frames[2][0];

        let woven = reference_match_weave([previous, current, next], WIDTH, HEIGHT, 0, 1);

        for row in 0..HEIGHT {
            let source = if row % 2 == 0 { current } else { previous };
            let start = row * WIDTH;

            assert_eq!(
                woven[start..start + WIDTH],
                source[start..start + WIDTH],
                "row {row}"
            );
        }
    }
}
