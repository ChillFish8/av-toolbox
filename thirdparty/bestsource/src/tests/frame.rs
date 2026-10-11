use std::path::PathBuf;

use crate::tests::video::open_fixture;
use crate::{ColorFamily, PlanarTarget, PlaneTarget};

const WIDTH: usize = 64;
const CHROMA_WIDTH: usize = 32;

fn expected_frame0() -> Vec<u8> {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let path = manifest.join("tests").join("media").join("yuv420p-frame0.yuv");

    std::fs::read(path).expect("reads the raw frame fixture")
}

#[test]
fn a_frame_reports_its_format_and_timing() {
    let mut source = open_fixture();

    let frame = source.frame(0).expect("decodes frame 0");

    assert_eq!(frame.format().color_family, ColorFamily::Yuv);
    assert_eq!((frame.width(), frame.height()), (64, 64));
    assert_eq!((frame.ss_mod_width(), frame.ss_mod_height()), (64, 64));
    assert_eq!(frame.pts(), 0);
    assert!(frame.key_frame());
}

#[test]
fn the_first_frame_matches_ffmpegs_decode() {
    let mut source = open_fixture();
    let frame = source.frame(0).expect("decodes frame 0");
    let mut y_plane = vec![0u8; WIDTH * 64];
    let mut u_plane = vec![0u8; CHROMA_WIDTH * 32];
    let mut v_plane = vec![0u8; CHROMA_WIDTH * 32];

    {
        let mut target = PlanarTarget {
            planes: vec![
                PlaneTarget {
                    data: &mut y_plane,
                    stride: WIDTH,
                },
                PlaneTarget {
                    data: &mut u_plane,
                    stride: CHROMA_WIDTH,
                },
                PlaneTarget {
                    data: &mut v_plane,
                    stride: CHROMA_WIDTH,
                },
            ],
            alpha: None,
        };
        frame.export_planar(&mut target).expect("exports frame 0");
    }

    let mut actual = y_plane;
    actual.extend_from_slice(&u_plane);
    actual.extend_from_slice(&v_plane);
    assert_eq!(actual, expected_frame0());
}

#[test]
fn a_padded_stride_keeps_each_row_in_place() {
    let mut source = open_fixture();
    let frame = source.frame(0).expect("decodes frame 0");
    let padded_stride = WIDTH + 16;
    let mut y_plane = vec![0u8; padded_stride * 64];
    let mut u_plane = vec![0u8; CHROMA_WIDTH * 32];
    let mut v_plane = vec![0u8; CHROMA_WIDTH * 32];

    {
        let mut target = PlanarTarget {
            planes: vec![
                PlaneTarget {
                    data: &mut y_plane,
                    stride: padded_stride,
                },
                PlaneTarget {
                    data: &mut u_plane,
                    stride: CHROMA_WIDTH,
                },
                PlaneTarget {
                    data: &mut v_plane,
                    stride: CHROMA_WIDTH,
                },
            ],
            alpha: None,
        };
        frame.export_planar(&mut target).expect("exports frame 0");
    }

    let expected = expected_frame0();
    let last_row = &y_plane[63 * padded_stride..63 * padded_stride + WIDTH];
    assert_eq!(last_row, &expected[63 * WIDTH..64 * WIDTH]);
}

#[test]
fn a_frame_outlives_its_source() {
    let mut source = open_fixture();
    let frame = source.frame(0).expect("decodes frame 0");
    drop(source);
    let mut y_plane = vec![0u8; WIDTH * 64];
    let mut u_plane = vec![0u8; CHROMA_WIDTH * 32];
    let mut v_plane = vec![0u8; CHROMA_WIDTH * 32];

    {
        let mut target = PlanarTarget {
            planes: vec![
                PlaneTarget {
                    data: &mut y_plane,
                    stride: WIDTH,
                },
                PlaneTarget {
                    data: &mut u_plane,
                    stride: CHROMA_WIDTH,
                },
                PlaneTarget {
                    data: &mut v_plane,
                    stride: CHROMA_WIDTH,
                },
            ],
            alpha: None,
        };
        frame
            .export_planar(&mut target)
            .expect("exports after the source is gone");
    }

    let expected = expected_frame0();
    assert_eq!(y_plane, expected[..WIDTH * 64]);
}

#[test]
fn a_frame_past_the_end_is_an_error() {
    let mut source = open_fixture();

    let error = source.frame(8).expect_err("frame 8 decoded");

    assert!(error.message().contains("frame 8"), "{error}");
}

#[test]
fn a_negative_frame_is_an_error() {
    let mut source = open_fixture();

    let error = source.frame(-1).expect_err("frame -1 decoded");

    assert!(error.message().contains("frame -1"), "{error}");
}

#[test]
fn a_short_plane_is_refused() {
    let mut source = open_fixture();
    let frame = source.frame(0).expect("decodes frame 0");
    let mut y_plane = vec![0u8; WIDTH * 64 - 1];
    let mut u_plane = vec![0u8; CHROMA_WIDTH * 32];
    let mut v_plane = vec![0u8; CHROMA_WIDTH * 32];
    let mut target = PlanarTarget {
        planes: vec![
            PlaneTarget {
                data: &mut y_plane,
                stride: WIDTH,
            },
            PlaneTarget {
                data: &mut u_plane,
                stride: CHROMA_WIDTH,
            },
            PlaneTarget {
                data: &mut v_plane,
                stride: CHROMA_WIDTH,
            },
        ],
        alpha: None,
    };

    let error = frame
        .export_planar(&mut target)
        .expect_err("a short plane was written");

    assert!(error.message().contains("plane 0"), "{error}");
}

#[test]
fn a_stride_below_the_row_size_is_refused() {
    let mut source = open_fixture();
    let frame = source.frame(0).expect("decodes frame 0");
    let mut y_plane = vec![0u8; WIDTH * 64];
    let mut u_plane = vec![0u8; CHROMA_WIDTH * 32];
    let mut v_plane = vec![0u8; CHROMA_WIDTH * 32];
    let mut target = PlanarTarget {
        planes: vec![
            PlaneTarget {
                data: &mut y_plane,
                stride: WIDTH,
            },
            PlaneTarget {
                data: &mut u_plane,
                stride: CHROMA_WIDTH - 1,
            },
            PlaneTarget {
                data: &mut v_plane,
                stride: CHROMA_WIDTH,
            },
        ],
        alpha: None,
    };

    let error = frame
        .export_planar(&mut target)
        .expect_err("a narrow stride was written");

    assert!(error.message().contains("plane 1"), "{error}");
}

#[test]
fn the_wrong_number_of_planes_is_refused() {
    let mut source = open_fixture();
    let frame = source.frame(0).expect("decodes frame 0");
    let mut y_plane = vec![0u8; WIDTH * 64];
    let mut target = PlanarTarget {
        planes: vec![PlaneTarget {
            data: &mut y_plane,
            stride: WIDTH,
        }],
        alpha: None,
    };

    let error = frame
        .export_planar(&mut target)
        .expect_err("one plane was enough");

    assert!(error.message().contains("3 planes"), "{error}");
}
