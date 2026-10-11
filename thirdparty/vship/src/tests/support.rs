use crate::{Butteraugli, ChromaLocation, ColorSpace, Planes, Range, SampleType, device_count, device_info};

pub(crate) const SIZE: usize = 256;

pub(crate) fn gpu_id() -> i32 {
    let selector = std::env::var("AVTB_DEVICE").unwrap_or_default();
    let (kind, ordinal) = selector.split_once(':').unwrap_or((selector.as_str(), "0"));
    let wants_integrated = match kind {
        "discrete" => false,
        "integrated" => true,
        _ => return 0,
    };

    let ordinal: usize = ordinal.parse().expect("AVTB_DEVICE has a numeric ordinal");

    let count = device_count().expect("counts devices");
    let matching: Vec<i32> = (0..count)
        .filter(|&candidate| {
            let info = device_info(candidate).expect("describes a device");
            info.integrated == wants_integrated
        })
        .collect();

    *matching
        .get(ordinal)
        .expect("AVTB_DEVICE names a device Vship lists")
}

pub(crate) fn color_space() -> ColorSpace {
    ColorSpace {
        sample: SampleType::Uint8,
        width: SIZE as i64,
        height: SIZE as i64,
        range: Range::Limited,
        sub_sampling_w: 1,
        sub_sampling_h: 1,
        chroma_location: ChromaLocation::Left,
        matrix: 1,
        transfer: 1,
        primaries: 1,
    }
}

/// A smooth gradient with fine detail, so every metric has structure to compare.
pub(crate) fn reference_frame() -> [Vec<u8>; 3] {
    let mut y_plane = vec![0u8; SIZE * SIZE];
    for row in 0..SIZE {
        for column in 0..SIZE {
            let gradient = 16 + (row + column) * 200 / (2 * SIZE);
            let detail = if (row / 4 + column / 4) % 2 == 0 { 10 } else { 0 };
            y_plane[row * SIZE + column] = (gradient + detail) as u8;
        }
    }

    let chroma_plane = vec![128u8; (SIZE / 2) * (SIZE / 2)];

    [y_plane, chroma_plane.clone(), chroma_plane]
}

/// The reference with its luma detail blurred away.
pub(crate) fn distorted_frame() -> [Vec<u8>; 3] {
    let [mut y_plane, u_plane, v_plane] = reference_frame();
    for row in 0..SIZE {
        for column in 0..SIZE {
            let gradient = 16 + (row + column) * 200 / (2 * SIZE);
            y_plane[row * SIZE + column] = (gradient + 5) as u8;
        }
    }

    [y_plane, u_plane, v_plane]
}

pub(crate) fn planes(frame: &[Vec<u8>; 3]) -> Planes<'_> {
    Planes {
        data: [&frame[0], &frame[1], &frame[2]],
        strides: [SIZE, SIZE / 2, SIZE / 2],
    }
}

pub(crate) fn butteraugli() -> Butteraugli {
    let reference_space = color_space();
    let distorted_space = color_space();
    let gpu_id = gpu_id();

    Butteraugli::new(reference_space, distorted_space, 2, 203.0, gpu_id).expect("makes the metric")
}
