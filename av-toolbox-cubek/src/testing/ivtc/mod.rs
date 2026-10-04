mod fixtures;
mod reference;
mod synthetic;

pub use self::fixtures::{MapFixture, MaskFixture, excluded_map_fixtures, map_fixtures, mask_fixtures};
pub use self::reference::{
    IntPlane,
    reference_chroma_into_luma,
    reference_comb_mask,
    reference_compare_plane,
    reference_difference,
    reference_match_weave,
    reference_mic,
    reference_weave,
    scene_difference,
};
pub use self::synthetic::{SyntheticClip, synthetic_clips};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ivtc::{ComparePass, FieldSums};

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

    fn clip_named(name: &str) -> SyntheticClip {
        synthetic_clips()
            .into_iter()
            .find(|clip| clip.name == name)
            .unwrap_or_else(|| panic!("there is no synthetic clip {name}"))
    }

    /// The luma, U and V comb masks of `fixture`'s weave, with chroma marked into luma.
    fn reference_masks(fixture: &MaskFixture) -> [Vec<u8>; 3] {
        let clip = clip_named(&fixture.clip);
        let cthresh = 9 << (clip.bits - 8);
        let previous = &clip.frames[fixture.frame - 1];
        let current = &clip.frames[fixture.frame];
        let next = &clip.frames[fixture.frame + 1];

        let mut masks = [Vec::new(), Vec::new(), Vec::new()];
        for (plane_index, mask) in masks.iter_mut().enumerate() {
            let width = if plane_index == 0 { WIDTH } else { WIDTH / 2 };
            let height = if plane_index == 0 { HEIGHT } else { HEIGHT / 2 };
            let frames = [
                previous[plane_index].as_slice(),
                current[plane_index].as_slice(),
                next[plane_index].as_slice(),
            ];
            let woven = reference_match_weave(frames, width, height, fixture.matched, fixture.field);
            let plane = IntPlane {
                samples: &woven,
                width,
                height,
            };
            *mask = reference_comb_mask(plane, cthresh);
        }

        let [mut luma, u, v] = masks;
        reference_chroma_into_luma(&mut luma, WIDTH, &u, &v, WIDTH / 2, HEIGHT / 2, 1, 1);

        [luma, u, v]
    }

    fn fixture_name(fixture: &MaskFixture) -> String {
        format!(
            "{} frame {} field {} match {}",
            fixture.clip, fixture.frame, fixture.field, fixture.matched,
        )
    }

    #[test]
    fn every_comb_mask_matches_the_harness() {
        for fixture in mask_fixtures() {
            let [luma, u, v] = reference_masks(&fixture);
            let name = fixture_name(&fixture);

            assert_eq!(u, fixture.u, "{name} U");
            assert_eq!(v, fixture.v, "{name} V");
            assert_eq!(luma, fixture.luma, "{name} luma");
        }
    }

    #[test]
    fn every_mic_matches_the_harness() {
        for fixture in mask_fixtures() {
            let [luma, _, _] = reference_masks(&fixture);
            let mic = reference_mic(&luma, WIDTH, HEIGHT, 16, 16);

            assert_eq!(mic, fixture.mic, "{}", fixture_name(&fixture));
        }
    }

    #[test]
    fn a_negative_cthresh_marks_every_sample() {
        let samples = vec![100; WIDTH * HEIGHT];
        let plane = IntPlane {
            samples: &samples,
            width: WIDTH,
            height: HEIGHT,
        };

        let mask = reference_comb_mask(plane, -1);

        assert_eq!(mask, vec![255; WIDTH * HEIGHT]);
    }

    /// `compareFieldsSlow` on all three planes of `fixture`'s frame, the sums added and the maps kept.
    fn reference_compare(fixture: &MapFixture) -> (FieldSums, [Vec<u8>; 3]) {
        let clip = clip_named(&fixture.clip);
        let previous = &clip.frames[fixture.frame - 1];
        let current = &clip.frames[fixture.frame];
        let next = &clip.frames[fixture.frame + 1];

        let mut sums = FieldSums::default();
        let mut maps = [Vec::new(), Vec::new(), Vec::new()];
        for (plane_index, map) in maps.iter_mut().enumerate() {
            let chroma = u32::from(plane_index > 0);
            let width = WIDTH >> chroma;
            let height = HEIGHT >> chroma;
            let frames = [previous, current, next].map(|frame| IntPlane {
                samples: &frame[plane_index],
                width,
                height,
            });
            let pass = ComparePass {
                first: fixture.first,
                second: fixture.second,
                field: fixture.field,
                exclude_from: fixture.exclude_from >> chroma,
                exclude_to: fixture.exclude_to >> chroma,
                start_x: 8 >> chroma,
                shift: clip.bits - 8,
            };

            let (plane_sums, plane_map) = reference_compare_plane(frames, &pass);
            sums = sums.add(plane_sums);
            *map = plane_map;
        }

        (sums, maps)
    }

    fn map_name(fixture: &MapFixture) -> String {
        format!(
            "{} frame {} field {} matches {} and {} excluding {} to {}",
            fixture.clip,
            fixture.frame,
            fixture.field,
            fixture.first,
            fixture.second,
            fixture.exclude_from,
            fixture.exclude_to,
        )
    }

    #[test]
    fn every_map_matches_the_harness() {
        let fixtures = map_fixtures().into_iter().chain(excluded_map_fixtures());
        for fixture in fixtures {
            let (_, [luma, u, v]) = reference_compare(&fixture);
            let name = map_name(&fixture);

            assert_eq!(luma, fixture.luma, "{name} luma");
            assert_eq!(u, fixture.u, "{name} U");
            assert_eq!(v, fixture.v, "{name} V");
        }
    }

    #[test]
    fn excluding_rows_changes_the_sums() {
        let plain_fixtures = map_fixtures();
        let mut compared = 0;
        for excluded in excluded_map_fixtures() {
            let plain = plain_fixtures
                .iter()
                .find(|plain| {
                    plain.clip == excluded.clip
                        && plain.frame == excluded.frame
                        && plain.field == excluded.field
                        && plain.first == excluded.first
                        && plain.second == excluded.second
                })
                .expect("every excluded map has a plain twin");

            let (plain_sums, _) = reference_compare(plain);
            let (excluded_sums, _) = reference_compare(&excluded);
            if plain_sums == FieldSums::default() {
                continue;
            }

            assert_ne!(plain_sums, excluded_sums, "{}", map_name(&excluded));
            compared += 1;
        }

        assert!(compared > 0, "every plain compare summed to nothing");
    }
}
