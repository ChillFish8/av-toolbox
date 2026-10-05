use crate::testing::motion::render_at;

/// The luma size the motion clips' scenes render at.
const WIDTH: usize = 128;
const HEIGHT: usize = 64;

/// How many interlaced frames each clip holds.
const FRAMES: usize = 8;

/// How many luma rows apart the static clip's lines are, an odd count so they fall in both fields.
const LINE_PERIOD: usize = 5;

/// The first luma row the static clip draws a line on.
const FIRST_LINE: usize = 2;

/// The static clip's lines, as a value between 0 and 1.
const LINE_LEVEL: f64 = 0.85;

/// An interlaced clip made by weaving a double-rate progressive render, with the true frame of every field.
pub struct FieldClip {
    pub name: &'static str,
    pub bits: u32,
    pub width: usize,
    pub height: usize,
    /// Interlaced frames, top field first, Y, U, V at 4:2:0.
    pub frames: Vec<[Vec<i32>; 3]>,
    /// The progressive render each field `2n` and `2n + 1` was taken from, the same layout.
    ///
    /// On the noisy clips it holds the same noise as the fields.
    pub truth: Vec<[Vec<i32>; 3]>,
    /// The same renders without noise, equal to `truth` on the clips without noise.
    pub clean: Vec<[Vec<i32>; 3]>,
}

/// What a field clip's renders show.
#[derive(Clone, Copy)]
enum Scene {
    /// The scene of the motion clip named, moving as it does.
    Moving(&'static str),
    /// The first frame of the motion clip named, held still, with thin horizontal lines over its luma.
    Lined(&'static str),
}

struct FieldSpec {
    name: &'static str,
    bits: u32,
    scene: Scene,
}

const CLIPS: [FieldSpec; 5] = [
    FieldSpec {
        name: "field_pan_8",
        bits: 8,
        scene: Scene::Moving("pan_8"),
    },
    FieldSpec {
        name: "field_objects_8",
        bits: 8,
        scene: Scene::Moving("objects_8"),
    },
    FieldSpec {
        name: "field_noisy_8",
        bits: 8,
        scene: Scene::Moving("noisy_8"),
    },
    FieldSpec {
        name: "field_noisy_10",
        bits: 10,
        scene: Scene::Moving("noisy_10"),
    },
    FieldSpec {
        name: "field_static_8",
        bits: 8,
        scene: Scene::Lined("pan_8"),
    },
];

/// Draws the static clip's one-pixel-high lines over `luma`.
fn draw_lines(luma: &mut [i32], bits: u32) {
    let peak = ((1u32 << bits) - 1) as f64;
    let level = (LINE_LEVEL * peak).round() as i32;

    for row in (FIRST_LINE..HEIGHT).step_by(LINE_PERIOD) {
        let start = row * WIDTH;
        luma[start..start + WIDTH].fill(level);
    }
}

/// The progressive render `render` of `spec`, half a frame after the one before, with its noise only
/// when `noisy` is true.
fn render(spec: &FieldSpec, render: usize, noisy: bool) -> [Vec<i32>; 3] {
    match spec.scene {
        Scene::Moving(name) => render_at(name, render as f64 / 2.0, render, noisy),
        Scene::Lined(name) => {
            let mut planes = render_at(name, 0.0, 0, noisy);
            draw_lines(&mut planes[0], spec.bits);
            planes
        },
    }
}

/// One interlaced frame, its even rows from `top` and its odd rows from `bottom`.
fn weave(top: &[Vec<i32>; 3], bottom: &[Vec<i32>; 3]) -> [Vec<i32>; 3] {
    [0, 1, 2].map(|plane| {
        let width = if plane == 0 { WIDTH } else { WIDTH / 2 };

        let mut woven = top[plane].clone();
        let rows = woven.len() / width;
        for row in (1..rows).step_by(2) {
            let span = row * width..(row + 1) * width;
            woven[span.clone()].copy_from_slice(&bottom[plane][span]);
        }

        woven
    })
}

fn build_clip(spec: &FieldSpec) -> FieldClip {
    let truth: Vec<[Vec<i32>; 3]> = (0..2 * FRAMES).map(|index| render(spec, index, true)).collect();
    let clean = (0..2 * FRAMES).map(|index| render(spec, index, false)).collect();
    let frames = truth.chunks(2).map(|pair| weave(&pair[0], &pair[1])).collect();

    FieldClip {
        name: spec.name,
        bits: spec.bits,
        width: WIDTH,
        height: HEIGHT,
        frames,
        truth,
        clean,
    }
}

/// The interlaced synthetic clips, 128x64 luma with 4:2:0 chroma and 8 frames each.
///
/// Each renders a motion clip's scene at twice its rate and weaves field `2n` from render `2n` and field
/// `2n + 1` from render `2n + 1`:
///
/// - `field_pan_8` pans a smooth background.
/// - `field_objects_8` adds two moving objects.
/// - `field_noisy_8` and `field_noisy_10` add noise to that, drawn afresh for every render. `truth` holds
///   the noise and `clean` does not.
/// - `field_static_8` holds one still texture with one-pixel-high lines every few rows, which a bob
///   shimmers on.
pub fn field_clips() -> Vec<FieldClip> {
    CLIPS.iter().map(build_clip).collect()
}
