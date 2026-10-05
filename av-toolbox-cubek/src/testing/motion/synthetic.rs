use std::f64::consts::{FRAC_PI_2, LN_2, PI, TAU};
use std::io;
use std::path::Path;

const WIDTH: usize = 128;
const HEIGHT: usize = 64;
const FRAMES: usize = 8;
const PAN_VELOCITY: (f64, f64) = (1.5, 0.5);
const NOISE_SIGMA_8_BIT: f64 = 3.0;
/// The side of one cell of the grain's hashed lattice, in pixels.
const GRAIN_CELL: f64 = 4.0;

/// A progressive 4:2:0 clip whose motion is known exactly.
pub struct MotionClip {
    pub name: &'static str,
    pub bits: u32,
    pub width: usize,
    pub height: usize,
    /// Every frame's luma, chroma U and chroma V as code values, 4:2:0.
    pub frames: Vec<[Vec<i32>; 3]>,
    /// The same frames without noise.
    pub clean: Vec<[Vec<i32>; 3]>,
    /// The true displacement of each luma sample from frame `t` to frame `t + 1`, in pixels.
    pub displacement: Vec<Vec<(f32, f32)>>,
}

struct ClipSpec {
    name: &'static str,
    bits: u32,
    objects: bool,
    noisy: bool,
    /// The first frame showing the second background, or `FRAMES` for none.
    cut: usize,
}

const CLIPS: [ClipSpec; 6] = [
    ClipSpec {
        name: "pan_8",
        bits: 8,
        objects: false,
        noisy: false,
        cut: FRAMES,
    },
    ClipSpec {
        name: "pan_10",
        bits: 10,
        objects: false,
        noisy: false,
        cut: FRAMES,
    },
    ClipSpec {
        name: "objects_8",
        bits: 8,
        objects: true,
        noisy: false,
        cut: FRAMES,
    },
    ClipSpec {
        name: "noisy_8",
        bits: 8,
        objects: true,
        noisy: true,
        cut: FRAMES,
    },
    ClipSpec {
        name: "noisy_10",
        bits: 10,
        objects: true,
        noisy: true,
        cut: FRAMES,
    },
    ClipSpec {
        name: "cut_8",
        bits: 8,
        objects: false,
        noisy: false,
        cut: 4,
    },
];

/// One sinusoid of a texture, running along `direction` with a period in pixels.
struct Wave {
    amplitude: f64,
    period: f64,
    direction: (f64, f64),
    phase: f64,
}

/// A smooth texture around `mean`, three waves plus a grain interpolated from a hashed lattice.
struct Texture {
    mean: f64,
    waves: [Wave; 3],
    grain_amplitude: f64,
    grain_seed: u64,
    /// The chroma U and V means, around which chroma follows the luma texture.
    chroma: (f64, f64),
}

const FIRST_BACKGROUND: Texture = Texture {
    mean: 0.5,
    waves: [
        Wave {
            amplitude: 0.14,
            period: 29.0,
            direction: (0.8, 0.6),
            phase: 0.3,
        },
        Wave {
            amplitude: 0.1,
            period: 17.0,
            direction: (-0.28, 0.96),
            phase: 1.1,
        },
        Wave {
            amplitude: 0.07,
            period: 9.5,
            direction: (1.0, 0.0),
            phase: 2.0,
        },
    ],
    grain_amplitude: 0.24,
    grain_seed: 11,
    chroma: (0.45, 0.55),
};

const SECOND_BACKGROUND: Texture = Texture {
    mean: 0.45,
    waves: [
        Wave {
            amplitude: 0.12,
            period: 21.0,
            direction: (-0.6, 0.8),
            phase: 2.4,
        },
        Wave {
            amplitude: 0.11,
            period: 13.0,
            direction: (0.96, 0.28),
            phase: 0.7,
        },
        Wave {
            amplitude: 0.08,
            period: 7.5,
            direction: (0.0, 1.0),
            phase: 1.6,
        },
    ],
    grain_amplitude: 0.26,
    grain_seed: 23,
    chroma: (0.6, 0.4),
};

const FIRST_OBJECT: Texture = Texture {
    mean: 0.3,
    waves: [
        Wave {
            amplitude: 0.1,
            period: 7.0,
            direction: (0.6, 0.8),
            phase: 0.0,
        },
        Wave {
            amplitude: 0.06,
            period: 11.0,
            direction: (1.0, 0.0),
            phase: 0.5,
        },
        Wave {
            amplitude: 0.04,
            period: 5.0,
            direction: (0.0, 1.0),
            phase: 1.0,
        },
    ],
    grain_amplitude: 0.2,
    grain_seed: 37,
    chroma: (0.35, 0.65),
};

const SECOND_OBJECT: Texture = Texture {
    mean: 0.66,
    waves: [
        Wave {
            amplitude: 0.09,
            period: 6.0,
            direction: (-0.8, 0.6),
            phase: 0.9,
        },
        Wave {
            amplitude: 0.06,
            period: 10.0,
            direction: (0.0, 1.0),
            phase: 0.2,
        },
        Wave {
            amplitude: 0.04,
            period: 4.5,
            direction: (1.0, 0.0),
            phase: 2.2,
        },
    ],
    grain_amplitude: 0.2,
    grain_seed: 41,
    chroma: (0.62, 0.3),
};

/// A textured rectangle whose top left corner sits at `origin` in frame 0.
struct MovingObject {
    texture: &'static Texture,
    origin: (f64, f64),
    size: (f64, f64),
    velocity: (f64, f64),
}

/// The objects of an `objects_*` or `noisy_*` clip, the last drawn on top.
const OBJECTS: [MovingObject; 2] = [
    MovingObject {
        texture: &FIRST_OBJECT,
        origin: (88.0, 12.0),
        size: (24.0, 16.0),
        velocity: (-2.0, 0.0),
    },
    MovingObject {
        texture: &SECOND_OBJECT,
        origin: (20.0, 6.0),
        size: (16.0, 16.0),
        velocity: (0.5, 1.5),
    },
];

/// The splitmix64 finaliser.
fn mix(value: u64) -> u64 {
    let mut mixed = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);

    mixed ^ (mixed >> 31)
}

fn hash(words: &[u64]) -> u64 {
    words.iter().fold(0, |state, word| mix(state ^ word))
}

/// A value from 0 up to but not including 1, in steps of 2^-53.
fn unit_of(hashed: u64) -> f64 {
    let top_bits = hashed >> 11;
    top_bits as f64 / (1u64 << 53) as f64
}

/// The sine of `phase`, from a Taylor series in plain arithmetic, so every machine agrees.
fn sine(phase: f64) -> f64 {
    let turns = (phase / TAU).round();
    let reduced = phase - turns * TAU;
    let folded = if reduced > FRAC_PI_2 {
        PI - reduced
    } else if reduced < -FRAC_PI_2 {
        -PI - reduced
    } else {
        reduced
    };

    let squared = folded * folded;
    let mut term = folded;
    let mut total = folded;
    for order in 1..10 {
        let divisor = (2 * order * (2 * order + 1)) as f64;
        term *= -squared / divisor;
        total += term;
    }

    total
}

fn cosine(phase: f64) -> f64 {
    sine(phase + FRAC_PI_2)
}

/// The natural logarithm of a positive normal `value`, from its exponent and an atanh series.
fn natural_log(value: f64) -> f64 {
    let bits = value.to_bits();
    let exponent = ((bits >> 52) & 0x7ff) as i64 - 1023;
    let mantissa_bits = (bits & 0x000f_ffff_ffff_ffff) | 0x3ff0_0000_0000_0000;
    let mantissa = f64::from_bits(mantissa_bits);

    let ratio = (mantissa - 1.0) / (mantissa + 1.0);
    let squared = ratio * ratio;
    let mut power = ratio;
    let mut total = 0.0;
    for order in 0..24 {
        let divisor = (2 * order + 1) as f64;
        total += power / divisor;
        power *= squared;
    }

    exponent as f64 * LN_2 + 2.0 * total
}

/// One standard normal draw, by Box-Muller from two hashed uniforms.
fn gaussian(words: &[u64]) -> f64 {
    let first = hash(words);
    let second = mix(first);

    // Shifted into 2^-53 up to 1, so the logarithm is always finite.
    let radius_uniform = unit_of(first) + 1.0 / (1u64 << 53) as f64;
    let angle_uniform = unit_of(second);
    let radius = (-2.0 * natural_log(radius_uniform)).sqrt();

    radius * cosine(TAU * angle_uniform)
}

/// A smoothstep blend of hashed lattice values around `x`, `y`, between 0 and 1.
fn grain(seed: u64, x: f64, y: f64) -> f64 {
    let cell_x = (x / GRAIN_CELL).floor();
    let cell_y = (y / GRAIN_CELL).floor();
    let fraction_x = x / GRAIN_CELL - cell_x;
    let fraction_y = y / GRAIN_CELL - cell_y;
    let weight_x = fraction_x * fraction_x * (3.0 - 2.0 * fraction_x);
    let weight_y = fraction_y * fraction_y * (3.0 - 2.0 * fraction_y);

    let corner = |offset_x: f64, offset_y: f64| {
        let lattice_x = (cell_x + offset_x) as i64 as u64;
        let lattice_y = (cell_y + offset_y) as i64 as u64;
        let hashed = hash(&[seed, lattice_x, lattice_y]);
        unit_of(hashed)
    };

    let top_left = corner(0.0, 0.0);
    let top_right = corner(1.0, 0.0);
    let bottom_left = corner(0.0, 1.0);
    let bottom_right = corner(1.0, 1.0);

    let top = top_left + (top_right - top_left) * weight_x;
    let bottom = bottom_left + (bottom_right - bottom_left) * weight_x;

    top + (bottom - top) * weight_y
}

impl Texture {
    /// The texture's luma, chroma U and chroma V at `x`, `y`, as values between 0 and 1.
    fn at(&self, x: f64, y: f64) -> [f64; 3] {
        let mut luma = self.mean;
        for wave in &self.waves {
            let along = wave.direction.0 * x + wave.direction.1 * y;
            let phase = TAU * along / wave.period + wave.phase;
            luma += wave.amplitude * sine(phase);
        }

        let grain_value = grain(self.grain_seed, x, y);
        luma += self.grain_amplitude * (grain_value - 0.5);

        let detail = luma - self.mean;
        let chroma_u = self.chroma.0 + 0.25 * detail;
        let chroma_v = self.chroma.1 - 0.2 * detail;

        [luma, chroma_u, chroma_v]
    }
}

/// The topmost object covering `x`, `y` at `time`, in frames, if any.
fn covering_object(spec: &ClipSpec, time: f64, x: f64, y: f64) -> Option<&'static MovingObject> {
    if !spec.objects {
        return None;
    }

    OBJECTS.iter().rev().find(|object| {
        let local_x = x - (object.origin.0 + object.velocity.0 * time);
        let local_y = y - (object.origin.1 + object.velocity.1 * time);

        (0.0..object.size.0).contains(&local_x) && (0.0..object.size.1).contains(&local_y)
    })
}

/// The clean luma, chroma U and chroma V at `time`, in frames, at luma position `x`, `y`.
fn scene_at(spec: &ClipSpec, time: f64, x: f64, y: f64) -> [f64; 3] {
    if let Some(object) = covering_object(spec, time, x, y) {
        let local_x = x - (object.origin.0 + object.velocity.0 * time);
        let local_y = y - (object.origin.1 + object.velocity.1 * time);
        return object.texture.at(local_x, local_y);
    }

    let background = if time >= spec.cut as f64 {
        &SECOND_BACKGROUND
    } else {
        &FIRST_BACKGROUND
    };
    let source_x = x - PAN_VELOCITY.0 * time;
    let source_y = y - PAN_VELOCITY.1 * time;

    background.at(source_x, source_y)
}

fn peak_of(bits: u32) -> u32 {
    (1 << bits) - 1
}

/// `value` between 0 and 1 as a code value, plus `noise` code values, rounded and clamped.
fn code_value(value: f64, noise: f64, bits: u32) -> i32 {
    let peak = peak_of(bits) as f64;
    let scaled = value * peak + noise;
    let rounded = scaled.round().clamp(0.0, peak);

    rounded as i32
}

/// Plane `plane` of the render numbered `frame` at `time`, with noise when `noisy` is true.
///
/// The noise is drawn for the render's number, so renders with different numbers have independent noise.
fn render_plane(
    spec: &ClipSpec,
    clip_index: usize,
    (frame, time): (usize, f64),
    plane: usize,
    noisy: bool,
) -> Vec<i32> {
    let (width, height) = if plane == 0 {
        (WIDTH, HEIGHT)
    } else {
        (WIDTH / 2, HEIGHT / 2)
    };
    let sigma = NOISE_SIGMA_8_BIT * (1u32 << (spec.bits - 8)) as f64;

    let mut samples = Vec::with_capacity(width * height);
    for row in 0..height {
        for column in 0..width {
            // A chroma sample sits at the centre of the 2x2 luma samples it covers.
            let (x, y) = if plane == 0 {
                (column as f64, row as f64)
            } else {
                (2.0 * column as f64 + 0.5, 2.0 * row as f64 + 0.5)
            };
            let values = scene_at(spec, time, x, y);

            let noise = if noisy {
                let words = [
                    clip_index as u64,
                    frame as u64,
                    plane as u64,
                    column as u64,
                    row as u64,
                ];
                sigma * gaussian(&words)
            } else {
                0.0
            };
            samples.push(code_value(values[plane], noise, spec.bits));
        }
    }

    samples
}

/// The true motion of every luma sample of frame `frame` on to the next frame.
fn displacement_of(spec: &ClipSpec, frame: usize) -> Vec<(f32, f32)> {
    let mut motion = Vec::with_capacity(WIDTH * HEIGHT);
    for row in 0..HEIGHT {
        for column in 0..WIDTH {
            let object = covering_object(spec, frame as f64, column as f64, row as f64);
            let velocity = object.map_or(PAN_VELOCITY, |object| object.velocity);
            motion.push((velocity.0 as f32, velocity.1 as f32));
        }
    }

    motion
}

/// Every plane of the render numbered `frame` at `time`, with noise when `noisy` is true.
fn render_frame(spec: &ClipSpec, clip_index: usize, render: (usize, f64), noisy: bool) -> [Vec<i32>; 3] {
    [0, 1, 2].map(|plane| render_plane(spec, clip_index, render, plane, noisy))
}

/// The Y, U and V planes of the motion clip `name`'s scene at `time`, in frames, as code values.
///
/// A noisy clip's noise is added only when `noisy` is true. It is drawn for `render`, so renders with
/// different numbers have independent noise.
pub(crate) fn render_at(name: &str, time: f64, render: usize, noisy: bool) -> [Vec<i32>; 3] {
    let (clip_index, spec) = CLIPS
        .iter()
        .enumerate()
        .find(|(_, spec)| spec.name == name)
        .expect("the clip is one of the motion clips");

    render_frame(spec, clip_index, (render, time), spec.noisy && noisy)
}

fn build_clip(clip_index: usize, spec: &ClipSpec) -> MotionClip {
    let clean: Vec<[Vec<i32>; 3]> = (0..FRAMES)
        .map(|frame| render_frame(spec, clip_index, (frame, frame as f64), false))
        .collect();
    let frames = if spec.noisy {
        (0..FRAMES)
            .map(|frame| render_frame(spec, clip_index, (frame, frame as f64), true))
            .collect()
    } else {
        clean.clone()
    };
    let displacement = (0..FRAMES - 1)
        .map(|frame| displacement_of(spec, frame))
        .collect();

    MotionClip {
        name: spec.name,
        bits: spec.bits,
        width: WIDTH,
        height: HEIGHT,
        frames,
        clean,
        displacement,
    }
}

/// The synthetic motion clips, 128x64 luma with 4:2:0 chroma and 8 frames each.
///
/// A smooth background pans by 1.5 pixels across and 0.5 down each frame, so sub-pixel positions render
/// exactly. The `objects_*` and `noisy_*` clips add two textured rectangles moving on their own, the
/// `noisy_*` clips add Gaussian noise of 3 code values on the 8-bit scale, and `cut_8` switches to an
/// unrelated background from frame 4. Across that cut the displacement still holds the pan's speed.
pub fn motion_clips() -> Vec<MotionClip> {
    CLIPS
        .iter()
        .enumerate()
        .map(|(clip_index, spec)| build_clip(clip_index, spec))
        .collect()
}

/// Which luma samples of frame `frame` of `clip` an object covers, row after row.
///
/// A clip without objects has none covered.
pub fn object_mask(clip: &MotionClip, frame: usize) -> Vec<bool> {
    let spec = CLIPS
        .iter()
        .find(|spec| spec.name == clip.name)
        .expect("every clip has a spec");

    let mut mask = Vec::with_capacity(clip.width * clip.height);
    for row in 0..clip.height {
        for column in 0..clip.width {
            let object = covering_object(spec, frame as f64, column as f64, row as f64);
            mask.push(object.is_some());
        }
    }

    mask
}

/// `samples` as values between 0 and 1, each code value divided by the peak for `bits`.
pub fn normalised(samples: &[i32], bits: u32) -> Vec<f32> {
    let peak = peak_of(bits);

    samples
        .iter()
        .map(|sample| *sample as f32 / peak as f32)
        .collect()
}

/// Writes every clip's luma to `<directory>/<clip>.y` as little-endian `u16`, frame after frame, and the
/// clip list to `<directory>/clips.txt`.
pub fn write_motion_inputs(directory: &Path) -> io::Result<()> {
    std::fs::create_dir_all(directory)?;

    let mut listing = String::new();
    for clip in motion_clips() {
        let bytes: Vec<u8> = clip
            .frames
            .iter()
            .flat_map(|planes| planes[0].iter())
            .flat_map(|sample| (*sample as u16).to_le_bytes())
            .collect();
        let path = directory.join(format!("{}.y", clip.name));
        std::fs::write(path, bytes)?;

        let line = format!(
            "{} {} {} {} {}\n",
            clip.name,
            clip.bits,
            clip.width,
            clip.height,
            clip.frames.len(),
        );
        listing.push_str(&line);
    }

    std::fs::write(directory.join("clips.txt"), listing)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    fn inputs_directory() -> PathBuf {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("src/testing/fixtures/motion/inputs");
        path
    }

    #[test]
    fn clips_are_deterministic() {
        let first = motion_clips();
        let second = motion_clips();

        for (built, rebuilt) in first.iter().zip(&second) {
            assert_eq!(built.frames, rebuilt.frames, "{} frames", built.name);
            assert_eq!(built.clean, rebuilt.clean, "{} clean frames", built.name);
            assert_eq!(
                built.displacement, rebuilt.displacement,
                "{} displacement",
                built.name
            );
        }
    }

    #[test]
    fn sine_and_log_match_the_standard_library() {
        for step in -400..400 {
            let phase = step as f64 * 0.0731;
            let difference = (sine(phase) - phase.sin()).abs();
            assert!(difference < 1e-12, "sine({phase}) is off by {difference}");
        }

        for step in 1..400 {
            let value = step as f64 * 0.0173;
            let difference = (natural_log(value) - value.ln()).abs();
            assert!(difference < 1e-12, "natural_log({value}) is off by {difference}");
        }
    }

    /// The mean and standard deviation of the noise on frame 0's luma of the clip `name`.
    fn noise_spread(clips: &[MotionClip], name: &str) -> (f64, f64) {
        let noisy = clips
            .iter()
            .find(|clip| clip.name == name)
            .expect("the noisy clip exists");

        let differences: Vec<f64> = noisy.frames[0][0]
            .iter()
            .zip(&noisy.clean[0][0])
            .map(|(sample, clean)| (sample - clean) as f64)
            .collect();
        let count = differences.len() as f64;
        let mean = differences.iter().sum::<f64>() / count;
        let variance = differences
            .iter()
            .map(|difference| (difference - mean) * (difference - mean))
            .sum::<f64>()
            / count;

        (mean, variance.sqrt())
    }

    #[test]
    fn noise_has_the_requested_spread() {
        let clips = motion_clips();

        let (mean_8, deviation_8) = noise_spread(&clips, "noisy_8");
        assert!(mean_8.abs() < 0.2, "the 8-bit noise mean is {mean_8}");
        assert!(
            (deviation_8 - 3.0).abs() < 0.2,
            "the 8-bit noise deviation is {deviation_8}"
        );

        let (mean_10, deviation_10) = noise_spread(&clips, "noisy_10");
        assert!(mean_10.abs() < 0.8, "the 10-bit noise mean is {mean_10}");
        assert!(
            (deviation_10 - 12.0).abs() < 0.8,
            "the 10-bit noise deviation is {deviation_10}"
        );
    }

    #[test]
    fn displacement_follows_the_objects() {
        let clips = motion_clips();
        let objects = clips
            .iter()
            .find(|clip| clip.name == "objects_8")
            .expect("objects_8");

        let first_object = objects.displacement[0][20 * WIDTH + 100];
        let second_object = objects.displacement[0][10 * WIDTH + 25];
        let background = objects.displacement[0][50 * WIDTH + 64];

        assert_eq!(first_object, (-2.0, 0.0));
        assert_eq!(second_object, (0.5, 1.5));
        assert_eq!(background, (1.5, 0.5));
    }

    #[test]
    fn a_whole_pixel_pan_moves_the_background_exactly() {
        let spec = &CLIPS[0];
        let earlier = scene_at(spec, 0.0, 30.0, 20.0);
        let later = scene_at(spec, 2.0, 33.0, 21.0);

        assert_eq!(earlier, later);
    }

    #[test]
    fn clips_match_the_committed_inputs() {
        const INPUTS: [(&str, &[u8]); 6] = [
            ("pan_8", include_bytes!("../fixtures/motion/inputs/pan_8.y")),
            ("pan_10", include_bytes!("../fixtures/motion/inputs/pan_10.y")),
            (
                "objects_8",
                include_bytes!("../fixtures/motion/inputs/objects_8.y"),
            ),
            ("noisy_8", include_bytes!("../fixtures/motion/inputs/noisy_8.y")),
            ("noisy_10", include_bytes!("../fixtures/motion/inputs/noisy_10.y")),
            ("cut_8", include_bytes!("../fixtures/motion/inputs/cut_8.y")),
        ];

        for clip in motion_clips() {
            let (_, bytes) = INPUTS
                .iter()
                .find(|(name, _)| *name == clip.name)
                .expect("every clip has an input fixture");
            let (pairs, _) = bytes.as_chunks::<2>();
            let committed: Vec<i32> = pairs
                .iter()
                .map(|pair| i32::from(u16::from_le_bytes(*pair)))
                .collect();
            let built: Vec<i32> = clip
                .frames
                .iter()
                .flat_map(|planes| planes[0].iter().copied())
                .collect();

            assert_eq!(built, committed, "{} differs from its committed input", clip.name);
        }
    }

    #[test]
    #[ignore = "writes the committed motion fixture inputs, run by hand"]
    fn write_the_motion_inputs() {
        write_motion_inputs(&inputs_directory()).expect("the motion inputs are written");
    }
}
