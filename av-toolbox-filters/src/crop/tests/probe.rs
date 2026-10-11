use av_toolbox_core::{ProbeDefinition, Seeker, StageError};
use av_toolbox_gpu::{
    Channel,
    ColourInfo,
    ColourRange,
    Depth,
    FrameFormat,
    FrameLayout,
    FrameMeta,
    HostFrame,
    HostPool,
    SampleFormat,
    Subsampling,
};

use crate::crop::{Crop, CropBox, CropRequest, Edges, sample_indices};

const WIDTH: u32 = 32;
const HEIGHT: u32 = 24;

struct Frames {
    format: FrameFormat,
    frames: Vec<HostFrame>,
    reads: usize,
}

impl Seeker for Frames {
    fn format(&self) -> &FrameFormat {
        &self.format
    }

    fn frame_count(&self) -> usize {
        self.frames.len()
    }

    fn read(&mut self, index: usize) -> Result<HostFrame, StageError> {
        self.reads += 1;

        Ok(self.frames[index].clone())
    }
}

fn format(sample: SampleFormat, range: ColourRange, layout: FrameLayout) -> FrameFormat {
    let colour = ColourInfo {
        range,
        ..ColourInfo::default()
    };

    FrameFormat {
        width: WIDTH,
        height: HEIGHT,
        layout,
        sample,
        colour,
        frame_rate: None,
        field_order: None,
    }
}

fn yuv420() -> FrameLayout {
    FrameLayout::Yuv {
        subsampling: Subsampling::Yuv420,
    }
}

fn limited() -> FrameFormat {
    format(SampleFormat::U8, ColourRange::Limited, yuv420())
}

/// A frame whose luma is `black` outside `edges` and `content` inside, with chroma left at zero.
fn boxed(format: &FrameFormat, edges: Edges, black: u16, content: u16) -> HostFrame {
    boxed_with(format, edges, content, |_row, _column| black)
}

/// Like [boxed], with the bar value chosen per position.
fn boxed_with(format: &FrameFormat, edges: Edges, content: u16, bar: impl Fn(u32, u32) -> u16) -> HostFrame {
    let pool = HostPool::new();
    let mut frame = pool.alloc_frame(format, FrameMeta::at(0));

    for channel in format.channels() {
        let plane = frame.plane_mut(*channel).expect("listed");
        plane.fill(0);
    }

    let wide = format.sample.bytes() == 2;
    let plane = frame.plane_mut(Channel::Y).expect("luma");
    for row in 0..HEIGHT {
        for column in 0..WIDTH {
            let inside = column >= edges.left
                && column < WIDTH - edges.right
                && row >= edges.top
                && row < HEIGHT - edges.bottom;
            let value = if inside { content } else { bar(row, column) };
            let position = (row * WIDTH + column) as usize;

            match wide {
                true => plane[position * 2..position * 2 + 2].copy_from_slice(&value.to_le_bytes()),
                false => plane[position] = value as u8,
            }
        }
    }

    frame.freeze()
}

fn probe_one(seeker: &mut Frames, samples: u32) -> CropBox {
    let requests = [CropRequest { samples }];
    let mut found = Crop::probe(seeker, &requests).expect("probes");

    found.remove(0)
}

fn seeker_of(format: FrameFormat, frames: Vec<HostFrame>) -> Frames {
    Frames {
        format,
        frames,
        reads: 0,
    }
}

fn letterbox() -> Edges {
    Edges {
        left: 0,
        right: 0,
        top: 4,
        bottom: 6,
    }
}

fn top_only(top: u32) -> Edges {
    Edges {
        top,
        ..Edges::default()
    }
}

#[test]
fn a_letterbox_is_found_at_8_bit_limited_range() {
    let format = limited();
    let frame = boxed(&format, letterbox(), 16, 120);
    let mut seeker = seeker_of(format, vec![frame]);

    let found = probe_one(&mut seeker, 13);

    assert_eq!(found.edges, letterbox());
    assert_eq!(found.measured, (WIDTH, HEIGHT));
}

#[test]
fn a_pillarbox_and_a_windowbox_are_found() {
    let pillarbox = Edges {
        left: 6,
        right: 6,
        ..Edges::default()
    };
    let windowbox = Edges {
        left: 2,
        right: 4,
        top: 2,
        bottom: 2,
    };

    for edges in [pillarbox, windowbox] {
        let format = limited();
        let frame = boxed(&format, edges, 16, 120);
        let mut seeker = seeker_of(format, vec![frame]);

        let found = probe_one(&mut seeker, 13);

        assert_eq!(found.edges, edges);
    }
}

#[test]
fn a_letterbox_is_found_at_10_bit_and_in_full_range() {
    let ten_bit = SampleFormat::U16 { depth: Depth::TEN };
    let format = format(ten_bit, ColourRange::Limited, yuv420());
    let frame = boxed(&format, letterbox(), 64, 480);
    let mut seeker = seeker_of(format, vec![frame]);

    let found = probe_one(&mut seeker, 13);

    assert_eq!(found.edges, letterbox());

    let format = format_full();
    let frame = boxed(&format, letterbox(), 0, 31);
    let mut seeker = seeker_of(format, vec![frame]);

    let found = probe_one(&mut seeker, 13);

    assert_eq!(found.edges, letterbox());
}

fn format_full() -> FrameFormat {
    format(SampleFormat::U8, ColourRange::Full, yuv420())
}

#[test]
fn noisy_bars_stay_bars() {
    let format = limited();
    let noise = |row: u32, _column: u32| if row.is_multiple_of(2) { 16 } else { 22 };
    let frame = boxed_with(&format, letterbox(), 120, noise);
    let mut seeker = seeker_of(format, vec![frame]);

    let found = probe_one(&mut seeker, 13);

    assert_eq!(found.edges, letterbox());
}

#[test]
fn one_bright_sample_ends_the_scan_at_its_row() {
    let format = limited();
    let bright = |row: u32, column: u32| if row == 2 && column == 10 { 200 } else { 16 };
    let frame = boxed_with(&format, top_only(4), 120, bright);
    let mut seeker = seeker_of(format, vec![frame]);

    let found = probe_one(&mut seeker, 13);

    assert_eq!(found.edges.top, 2);
}

#[test]
fn uneven_bars_keep_their_own_edges() {
    let edges = Edges {
        top: 2,
        bottom: 8,
        ..Edges::default()
    };
    let format = limited();
    let frame = boxed(&format, edges, 16, 120);
    let mut seeker = seeker_of(format, vec![frame]);

    let found = probe_one(&mut seeker, 13);

    assert_eq!(found.edges, edges);
}

#[test]
fn the_smallest_crop_per_edge_wins() {
    let format = limited();
    let first_edges = Edges {
        top: 2,
        bottom: 6,
        ..Edges::default()
    };
    let second_edges = Edges {
        top: 4,
        bottom: 3,
        ..Edges::default()
    };
    let first = boxed(&format, first_edges, 16, 120);
    let second = boxed(&format, second_edges, 16, 120);
    let mut seeker = seeker_of(format, vec![first, second]);

    let found = probe_one(&mut seeker, 13);

    assert_eq!(found.edges.top, 2);
    assert_eq!(found.edges.bottom, 3);
}

#[test]
fn equal_requests_read_their_frames_once() {
    let format = limited();
    let frames: Vec<HostFrame> = (0..5).map(|_| boxed(&format, letterbox(), 16, 120)).collect();
    let mut seeker = seeker_of(format, frames);
    let requests = [CropRequest { samples: 13 }, CropRequest { samples: 13 }];

    let found = Crop::probe(&mut seeker, &requests).expect("probes");

    assert_eq!(seeker.reads, 5);
    assert_eq!(found[0], found[1]);
    assert_eq!(found[0].edges, letterbox());
}

#[test]
fn an_all_black_frame_counts_for_nothing() {
    let format = limited();
    let black = boxed(&format, Edges::default(), 16, 16);
    let boxed_frame = boxed(&format, letterbox(), 16, 120);
    let mut seeker = seeker_of(format, vec![black, boxed_frame]);

    let found = probe_one(&mut seeker, 13);

    assert_eq!(found.edges, letterbox());
}

#[test]
fn all_black_frames_crop_nothing() {
    let format = limited();
    let first = boxed(&format, Edges::default(), 16, 16);
    let second = boxed(&format, Edges::default(), 16, 16);
    let mut seeker = seeker_of(format, vec![first, second]);

    let found = probe_one(&mut seeker, 13);

    assert!(found.edges.is_zero());
}

#[test]
fn reading_stops_once_every_edge_is_zero() {
    let format = limited();
    let mut frames = vec![boxed(&format, Edges::default(), 16, 120)];
    for _ in 0..4 {
        frames.push(boxed(&format, letterbox(), 16, 120));
    }
    let mut seeker = seeker_of(format, frames);

    let found = probe_one(&mut seeker, 13);

    assert!(found.edges.is_zero());
    assert_eq!(seeker.reads, 1);
}

#[test]
fn rgb_and_float_sources_are_refused() {
    let requests = [CropRequest { samples: 13 }];

    let rgb = format(SampleFormat::U8, ColourRange::Full, FrameLayout::Rgb);
    let mut seeker = seeker_of(rgb, Vec::new());
    assert!(Crop::probe(&mut seeker, &requests).is_err());

    let float = format(SampleFormat::F32, ColourRange::Full, yuv420());
    let mut seeker = seeker_of(float, Vec::new());
    assert!(Crop::probe(&mut seeker, &requests).is_err());
}

#[test]
fn the_frames_read_match_the_spread() {
    let indices = sample_indices(42000, 13);

    assert_eq!(indices.len(), 13);
    assert_eq!(indices[0], 2994);
    assert_eq!(indices[12], 38923);
}

#[test]
fn a_short_clip_reads_every_frame() {
    assert_eq!(sample_indices(5, 13), vec![0, 1, 2, 3, 4]);
    assert_eq!(sample_indices(13, 13), (0..13).collect::<Vec<_>>());
}

#[test]
fn combining_keeps_the_smallest_edges() {
    let measured = (WIDTH, HEIGHT);
    let first = CropBox {
        edges: Edges {
            top: 2,
            bottom: 6,
            ..Edges::default()
        },
        measured,
    };
    let second = CropBox {
        edges: Edges {
            top: 4,
            bottom: 3,
            ..Edges::default()
        },
        measured,
    };

    let combined = Crop::combine(vec![first, second]).expect("same size");

    assert_eq!(combined.edges.top, 2);
    assert_eq!(combined.edges.bottom, 3);
}

#[test]
fn combining_boxes_of_different_sizes_names_both() {
    let small = CropBox {
        edges: Edges::default(),
        measured: (32, 24),
    };
    let large = CropBox {
        edges: Edges::default(),
        measured: (64, 48),
    };

    let error = Crop::combine(vec![small, large]).expect_err("sizes differ");
    let message = error.to_string();

    assert!(message.contains("32x24"), "{message}");
    assert!(message.contains("64x48"), "{message}");
}
