const WIDTH: i32 = 64;
const HEIGHT: i32 = 32;
const FILM_FRAMES: i32 = 10;
const VIDEO_FRAMES: i32 = 12;

/// A telecined clip of three planes per frame, as `i32` code values.
///
/// `order` is VFM's field order, 1 for top field first and 0 for bottom field first.
pub struct SyntheticClip {
    pub name: &'static str,
    pub bits: u32,
    pub order: u32,
    pub frames: Vec<[Vec<i32>; 3]>,
}

struct ClipSpec {
    name: &'static str,
    bits: i32,
    order: i32,
    /// The first film frame after a scene cut, or -1.
    cut: i32,
    /// The video frame whose first shown field is brightened, or -1.
    orphan: i32,
}

const CLIPS: [ClipSpec; 8] = [
    ClipSpec {
        name: "tff_8",
        bits: 8,
        order: 1,
        cut: -1,
        orphan: -1,
    },
    ClipSpec {
        name: "bff_8",
        bits: 8,
        order: 0,
        cut: -1,
        orphan: -1,
    },
    ClipSpec {
        name: "cut_8",
        bits: 8,
        order: 1,
        cut: 5,
        orphan: -1,
    },
    ClipSpec {
        name: "orphan_8",
        bits: 8,
        order: 1,
        cut: -1,
        orphan: 7,
    },
    ClipSpec {
        name: "tff_10",
        bits: 10,
        order: 1,
        cut: -1,
        orphan: -1,
    },
    ClipSpec {
        name: "bff_10",
        bits: 10,
        order: 0,
        cut: -1,
        orphan: -1,
    },
    ClipSpec {
        name: "cut_10",
        bits: 10,
        order: 1,
        cut: 5,
        orphan: -1,
    },
    ClipSpec {
        name: "orphan_10",
        bits: 10,
        order: 1,
        cut: -1,
        orphan: 7,
    },
];

fn mix(mut value: u32) -> u32 {
    value ^= value >> 15;
    value = value.wrapping_mul(0x2c1b_3c6d);
    value ^= value >> 12;
    value = value.wrapping_mul(0x297a_2d39);
    value ^= value >> 15;

    value
}

/// Sample `x`, `y` of plane `plane` of film frame `film`. Chroma planes are 32x16.
fn film_sample(clip: &ClipSpec, plane: i32, film: i32, x: i32, y: i32) -> i32 {
    let peak = (1 << clip.bits) - 1;
    let scale = 1 << (clip.bits - 8);
    let width = if plane == 0 { WIDTH } else { WIDTH / 2 };

    // A static gradient, inverted from the cut onward.
    let mut value = ((x * 3 + y * 5) % 160 + 48) * scale;
    if clip.cut >= 0 && film >= clip.cut {
        value = peak - value;
    }

    // A bar 6 luma or 3 chroma columns wide, moving right 2 luma or 1 chroma columns each film frame.
    let bar_width = if plane == 0 { 6 } else { 3 };
    let bar_speed = if plane == 0 { 2 } else { 1 };
    let bar_left = (bar_speed * film) % width;
    if (x - bar_left + width) % width < bar_width {
        value += 48 * scale;
    }

    // A luma strip 4 rows tall and 16 columns wide, stepping down 8 rows each film frame.
    let strip_top = 4 + (film % 3) * 8;
    if plane == 0 && (24..40).contains(&x) && y >= strip_top && y < strip_top + 4 {
        value += 48 * scale;
    }

    let seed = (plane * 1_000_003 + film * 7919 + y * 131 + x + clip.bits * 17) as u32;
    value += ((mix(seed) % 9) as i32 - 4) * scale;

    value.clamp(0, peak)
}

/// The film frame each field of video frame `video` shows, for the 3:2 cadence AA BB BC CD DD.
fn film_of_field(video: i32, shows_first: bool) -> i32 {
    const FIRST: [i32; 5] = [0, 1, 1, 2, 3];
    const SECOND: [i32; 5] = [0, 1, 2, 3, 3];

    let cycle = video / 5;
    let phase = (video % 5) as usize;
    let base = cycle * 4;

    base + if shows_first { FIRST[phase] } else { SECOND[phase] }
}

/// Sample `x`, `y` of plane `plane` of video frame `video`. Even rows are the top field.
fn video_sample(clip: &ClipSpec, plane: i32, video: i32, x: i32, y: i32) -> i32 {
    let top = y % 2 == 0;
    let shows_first = if clip.order != 0 { top } else { !top };
    let film = film_of_field(video, shows_first).min(FILM_FRAMES - 1);
    let value = film_sample(clip, plane, film, x, y);
    if !shows_first || video != clip.orphan {
        return value;
    }

    // The orphan field is brightened, so it matches no other field.
    let peak = (1 << clip.bits) - 1;
    let scale = 1 << (clip.bits - 8);
    let brighter = value + 16 * scale;

    brighter.min(peak)
}

fn video_plane(clip: &ClipSpec, plane: i32, video: i32) -> Vec<i32> {
    let width = if plane == 0 { WIDTH } else { WIDTH / 2 };
    let height = if plane == 0 { HEIGHT } else { HEIGHT / 2 };

    let mut samples = Vec::with_capacity((width * height) as usize);
    for y in 0..height {
        for x in 0..width {
            samples.push(video_sample(clip, plane, video, x, y));
        }
    }

    samples
}

/// The eight synthetic 3:2 telecined clips, 64x32 with 4:2:0 chroma.
pub fn synthetic_clips() -> Vec<SyntheticClip> {
    CLIPS
        .iter()
        .map(|clip| {
            let frames = (0..VIDEO_FRAMES)
                .map(|video| [0, 1, 2].map(|plane| video_plane(clip, plane, video)))
                .collect();

            SyntheticClip {
                name: clip.name,
                bits: clip.bits as u32,
                order: clip.order as u32,
                frames,
            }
        })
        .collect()
}
