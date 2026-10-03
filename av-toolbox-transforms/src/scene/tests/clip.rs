use std::time::Duration;

use av_toolbox_gpu::{
    Channel,
    ColourInfo,
    Depth,
    FrameContext,
    FrameFormat,
    FrameLayout,
    FrameMeta,
    HostFrame,
    SampleFormat,
    Subsampling,
};

/// The width and height of every test clip.
pub(super) const SIZE: u32 = 64;

/// Three scenes of 30 frames, with hard cuts at frames 30 and 60.
pub(super) const THREE_SCENES: &[(u64, u64)] = &[(1, 30), (2, 30), (3, 30)];

pub(super) fn grey8() -> FrameFormat {
    grey8_of(SIZE, SIZE)
}

pub(super) fn grey8_of(width: u32, height: u32) -> FrameFormat {
    FrameFormat {
        width,
        height,
        layout: FrameLayout::Grey,
        sample: SampleFormat::U8,
        colour: ColourInfo::default(),
        frame_rate: None,
        field_order: None,
    }
}

pub(super) fn yuv420_ten_bit() -> FrameFormat {
    FrameFormat {
        width: SIZE,
        height: SIZE,
        layout: FrameLayout::Yuv {
            subsampling: Subsampling::Yuv420,
        },
        sample: SampleFormat::U16 { depth: Depth::TEN },
        colour: ColourInfo::default(),
        frame_rate: None,
        field_order: None,
    }
}

/// A clip of `scenes`, each a seed for its texture and a frame count, 40 ms apart.
///
/// Each scene's texture drifts one pixel a frame, so frames within a scene resemble each other
/// and a new seed makes a hard cut.
pub(super) fn clip(context: &FrameContext, format: &FrameFormat, scenes: &[(u64, u64)]) -> Vec<HostFrame> {
    let mut frames = Vec::new();
    let mut index = 0;
    for &(seed, length) in scenes {
        for offset in 0..length {
            let frame = textured(context, format, seed, offset, index);
            frames.push(frame);
            index += 1;
        }
    }

    frames
}

/// A clip of `length` frames, every sample the same value throughout, 40 ms apart.
pub(super) fn flat_clip(context: &FrameContext, format: &FrameFormat, length: u64) -> Vec<HostFrame> {
    let mut frames = Vec::new();
    for index in 0..length {
        let frame = flat(context, format, index);
        frames.push(frame);
    }

    frames
}

fn flat(context: &FrameContext, format: &FrameFormat, index: u64) -> HostFrame {
    let timestamp = Duration::from_millis(index * 40);
    let meta = FrameMeta {
        timestamp: Some(timestamp),
        ..FrameMeta::at(index)
    };

    let mut frame = context.alloc_host_frame(format, meta);
    let luma = frame.plane_mut(Channel::Y).expect("every test clip carries luma");
    fill_neutral(luma, format);

    for channel in [Channel::U, Channel::V] {
        if let Some(chroma) = frame.plane_mut(channel) {
            fill_neutral(chroma, format);
        }
    }

    frame.freeze()
}

fn textured(context: &FrameContext, format: &FrameFormat, seed: u64, offset: u64, index: u64) -> HostFrame {
    let timestamp = Duration::from_millis(index * 40);
    let meta = FrameMeta {
        timestamp: Some(timestamp),
        ..FrameMeta::at(index)
    };

    let mut frame = context.alloc_host_frame(format, meta);
    let luma = frame.plane_mut(Channel::Y).expect("every test clip carries luma");
    fill_texture(luma, format, seed, offset);

    for channel in [Channel::U, Channel::V] {
        if let Some(chroma) = frame.plane_mut(channel) {
            fill_neutral(chroma, format);
        }
    }

    frame.freeze()
}

/// Fills `plane` with 4 by 4 cells of noise from `seed`, moved `offset` pixels to the left.
fn fill_texture(plane: &mut [u8], format: &FrameFormat, seed: u64, offset: u64) {
    let width = format.width as usize;
    let height = format.height as usize;
    let largest = largest_value(format);
    let bytes_per_sample = format.sample.bytes();
    for y in 0..height {
        for x in 0..width {
            let shifted = x as u64 + offset;
            let hashed = noise(seed, shifted, y as u64);
            let value = hashed % (largest + 1);
            let at = (y * width + x) * bytes_per_sample;
            write_sample(plane, at, value, bytes_per_sample);
        }
    }
}

fn fill_neutral(plane: &mut [u8], format: &FrameFormat) {
    let largest = largest_value(format);
    let middle = largest.div_ceil(2);
    let bytes_per_sample = format.sample.bytes();
    for at in (0..plane.len()).step_by(bytes_per_sample) {
        write_sample(plane, at, middle, bytes_per_sample);
    }
}

fn largest_value(format: &FrameFormat) -> u64 {
    match format.sample {
        SampleFormat::U16 { depth } => u64::from(depth.max_value()),
        _ => 255,
    }
}

fn write_sample(plane: &mut [u8], at: usize, value: u64, bytes_per_sample: usize) {
    if bytes_per_sample == 1 {
        plane[at] = value as u8;
        return;
    }

    let bytes = (value as u16).to_le_bytes();
    plane[at..at + 2].copy_from_slice(&bytes);
}

fn noise(seed: u64, x: u64, y: u64) -> u64 {
    let cell_x = x / 4;
    let cell_y = y / 4;

    let mut hash = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ cell_x.wrapping_mul(0x85EB_CA6B)
        ^ cell_y.wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    hash ^= hash >> 33;
    hash = hash.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    hash ^= hash >> 33;

    hash
}
