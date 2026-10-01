#![cfg(all(unix, feature = "gpu-tests", any(feature = "vulkan", feature = "metal")))]

use std::fs;
use std::ops::Range;
use std::path::Path;

use av_toolbox_gpu::{Backend, ChromaLocation, MaskChroma, StreamFormat, Subsampling, test_device};
use av_toolbox_pipeline::{DiskFiles, Pipeline, Supplied};
use av_toolbox_runtime::Settings;

const SIZE: usize = 64;

const FRAMES: u64 = 3;

const BASE_LUMA: u8 = 200;

const STROKE_LUMA: u8 = 20;

const BASE_CHROMA: Chroma = Chroma { u: 128, v: 128 };

const OVERLAY: Colour = Colour {
    luma: 0,
    chroma: Chroma { u: 40, v: 220 },
};

const SECOND_OVERLAY: Colour = Colour {
    luma: 100,
    chroma: Chroma { u: 90, v: 170 },
};

const STROKE: Range<usize> = 30..34;

const STROKE_COLUMN: usize = 31;

const BACKGROUND_COLUMN: usize = 5;

/// The 4:2:0 chroma column sited on luma column 32, whose bilinear taps on luma columns `31..=33`
/// all fall inside the stroke.
const STROKE_CHROMA_COLUMN: usize = 16;

/// The 4:2:0 chroma column sited on luma column 4, whose taps all fall well clear of the stroke.
const BACKGROUND_CHROMA_COLUMN: usize = 2;

/// One U and V sample pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Chroma {
    u: u8,
    v: u8,
}

/// A flat clip's luma and chroma.
#[derive(Debug, Clone, Copy)]
struct Colour {
    luma: u8,
    chroma: Chroma,
}

/// One decoded frame's planes.
struct DecodedFrame {
    luma: Vec<u8>,
    u_plane: Vec<u8>,
    v_plane: Vec<u8>,
}

/// A decoded y4m stream's header and the planes of each of its frames.
struct Decoded {
    width: usize,
    height: usize,
    colorspace: y4m::Colorspace,
    frames: Vec<DecodedFrame>,
}

/// A C420 y4m clip whose luma comes from `luma_at(column)` on every row, with flat `chroma`.
fn clip(luma_at: impl Fn(usize) -> u8, chroma: Chroma) -> Vec<u8> {
    let mut bytes = Vec::new();
    let ratio = y4m::Ratio::new(24, 1);
    let range_text = b"COLORRANGE=FULL".to_vec();
    let range = y4m::VendorExtensionString::new(range_text).expect("valid");
    let mut encoder = y4m::encode(SIZE, SIZE, ratio)
        .append_vendor_extension(range)
        .write_header(&mut bytes)
        .expect("writes the header");

    let row: Vec<u8> = (0..SIZE).map(&luma_at).collect();
    let luma = row.repeat(SIZE);
    let chroma_samples = (SIZE / 2) * (SIZE / 2);
    let u_plane = vec![chroma.u; chroma_samples];
    let v_plane = vec![chroma.v; chroma_samples];
    for _ in 0..FRAMES {
        let frame = y4m::Frame::new([&luma, &u_plane, &v_plane], None);
        encoder.write_frame(&frame).expect("writes a frame");
    }

    bytes
}

/// A bright clip with a dark vertical stroke across `STROKE`.
fn stroked_clip() -> Vec<u8> {
    let luma_at = |column| {
        if STROKE.contains(&column) {
            STROKE_LUMA
        } else {
            BASE_LUMA
        }
    };

    clip(luma_at, BASE_CHROMA)
}

fn flat_clip(colour: Colour) -> Vec<u8> {
    clip(|_| colour.luma, colour.chroma)
}

fn decode(stream: &[u8]) -> Decoded {
    let mut decoder = y4m::decode(stream).expect("reads the header");
    let width = decoder.get_width();
    let height = decoder.get_height();
    let colorspace = decoder.get_colorspace();

    let mut frames = Vec::new();
    while let Ok(frame) = decoder.read_frame() {
        let luma = frame.get_y_plane().to_vec();
        let u_plane = frame.get_u_plane().to_vec();
        let v_plane = frame.get_v_plane().to_vec();
        frames.push(DecodedFrame {
            luma,
            u_plane,
            v_plane,
        });
    }

    Decoded {
        width,
        height,
        colorspace,
        frames,
    }
}

/// Checks `path` holds every frame at the clips' format, each taking `overlay` on the stroke.
fn assert_blended(path: &Path, overlay: Colour) {
    let written = fs::read(path).expect("reads the output");
    let decoded = decode(&written);

    assert_eq!((decoded.width, decoded.height), (SIZE, SIZE));
    assert!(matches!(decoded.colorspace, y4m::Colorspace::C420));
    assert_eq!(decoded.frames.len(), FRAMES as usize);

    let row_start = SIZE / 2 * SIZE;
    let chroma_width = SIZE / 2;
    let chroma_row_start = SIZE / 4 * chroma_width;
    for frame in &decoded.frames {
        let middle_row = &frame.luma[row_start..row_start + SIZE];
        let u_row = &frame.u_plane[chroma_row_start..chroma_row_start + chroma_width];
        let v_row = &frame.v_plane[chroma_row_start..chroma_row_start + chroma_width];
        let stroke_chroma = Chroma {
            u: u_row[STROKE_CHROMA_COLUMN],
            v: v_row[STROKE_CHROMA_COLUMN],
        };

        let background_chroma = Chroma {
            u: u_row[BACKGROUND_CHROMA_COLUMN],
            v: v_row[BACKGROUND_CHROMA_COLUMN],
        };

        assert_eq!(
            middle_row[STROKE_COLUMN], overlay.luma,
            "the stroke did not take the overlay"
        );
        assert_eq!(
            middle_row[BACKGROUND_COLUMN], BASE_LUMA,
            "the background did not keep the base"
        );
        assert_eq!(
            stroke_chroma, overlay.chroma,
            "the stroke's chroma did not take the overlay"
        );
        assert_eq!(
            background_chroma, BASE_CHROMA,
            "the background's chroma did not keep the base"
        );
    }
}

fn utf8(path: &Path) -> &str {
    path.to_str().expect("a utf-8 path")
}

fn example() -> &'static Path {
    Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../examples/lineart.toml"))
}

#[test]
fn the_lineart_example_blends_the_overlay_into_the_lines() {
    let folder = tempfile::tempdir().expect("a temp dir");
    let base = folder.path().join("base.y4m");
    let overlay = folder.path().join("overlay.y4m");
    let output = folder.path().join("output.y4m");

    let base_clip = stroked_clip();
    let overlay_clip = flat_clip(OVERLAY);
    fs::write(&base, base_clip).expect("writes the base");
    fs::write(&overlay, overlay_clip).expect("writes the overlay");

    let base_path = utf8(&base);
    let overlay_path = utf8(&overlay);
    let output_path = utf8(&output);
    let mut supplied = Supplied::new();
    supplied.set_flag("base", base_path);
    supplied.set_flag("overlay", overlay_path);
    supplied.set_flag("output", output_path);

    let example_path = example();
    let loaded = av_toolbox_pipeline::load(example_path, &DiskFiles, &supplied).expect("the example loads");
    let registry = av_toolbox_pipeline::registry().expect("the built-in stages register");
    let pipeline = Pipeline::build(&registry, &loaded.specs).expect("should build");
    let backend = Backend::open_default(test_device()).expect("should open a device");
    let compiled = pipeline.compile(&backend).expect("should compile");
    let settings = Settings::default();
    let engine = compiled.engine(settings).expect("should build the engine");
    let report = engine.run().expect("should run");

    assert_eq!(report.frames, FRAMES);
    assert_blended(&output, OVERLAY);
}

#[test]
fn one_mask_read_by_two_blends_is_downscaled_once() {
    let folder = tempfile::tempdir().expect("a temp dir");
    let base = folder.path().join("base.y4m");
    let first_overlay = folder.path().join("first_overlay.y4m");
    let second_overlay = folder.path().join("second_overlay.y4m");
    let first_output = folder.path().join("first_output.y4m");
    let second_output = folder.path().join("second_output.y4m");

    let base_clip = stroked_clip();
    let first_overlay_clip = flat_clip(OVERLAY);
    let second_overlay_clip = flat_clip(SECOND_OVERLAY);
    fs::write(&base, base_clip).expect("writes the base");
    fs::write(&first_overlay, first_overlay_clip).expect("writes the first overlay");
    fs::write(&second_overlay, second_overlay_clip).expect("writes the second overlay");

    let base_path = utf8(&base);
    let first_overlay_path = utf8(&first_overlay);
    let second_overlay_path = utf8(&second_overlay);
    let first_output_path = utf8(&first_output);
    let second_output_path = utf8(&second_output);
    let text = format!(
        r#"
[source.base]
type = "y4m/pipe"
args.path = "{base_path}"

[source.first_overlay]
type = "y4m/pipe"
args.path = "{first_overlay_path}"

[source.second_overlay]
type = "y4m/pipe"
args.path = "{second_overlay_path}"

[mask.lines]
type = "lines/adaptive"
inputs = ["base"]

[filter.first_blend]
type = "blend/masked"
inputs = ["base", "first_overlay", "lines"]

[filter.second_blend]
type = "blend/masked"
inputs = ["base", "second_overlay", "lines"]

[sink.first_output]
type = "y4m/pipe"
inputs = ["first_blend"]
args.path = "{first_output_path}"

[sink.second_output]
type = "y4m/pipe"
inputs = ["second_blend"]
args.path = "{second_output_path}"
"#
    );

    let pipeline_path = folder.path().join("pipeline.toml");
    fs::write(&pipeline_path, &text).expect("writes the pipeline");
    let supplied = Supplied::new();
    let loaded =
        av_toolbox_pipeline::load(&pipeline_path, &DiskFiles, &supplied).expect("the file should load");

    let registry = av_toolbox_pipeline::registry().expect("the built-in stages register");
    let pipeline = Pipeline::build(&registry, &loaded.specs).expect("should build");
    let backend = Backend::open_default(test_device()).expect("should open a device");
    let compiled = pipeline.compile(&backend).expect("should compile");

    let mut chroma_carriers = Vec::new();
    for node in compiled.graph().nodes() {
        let node_id = node.id();
        let output = compiled.formats().output(node_id);
        if let Some(StreamFormat::Mask(mask)) = output
            && let Some(chroma) = mask.chroma
        {
            chroma_carriers.push((node.name(), chroma));
        }
    }

    let expected = MaskChroma {
        subsampling: Subsampling::Yuv420,
        location: ChromaLocation::Left,
    };

    assert_eq!(chroma_carriers, [("lines", expected)]);

    let settings = Settings::default();
    let engine = compiled.engine(settings).expect("should build the engine");
    let report = engine.run().expect("should run");

    assert_eq!(report.frames, FRAMES * 2, "each sink counts every frame it wrote");
    assert_blended(&first_output, OVERLAY);
    assert_blended(&second_output, SECOND_OVERLAY);
}
