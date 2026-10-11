use std::path::{Path, PathBuf};

use crate::{ColorFamily, OpenOptions, VideoSource};

pub(crate) fn fixture() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

    manifest
        .join("..")
        .join("..")
        .join("av-toolbox-sources")
        .join("tests")
        .join("media")
        .join("yuv420p.mkv")
}

pub(crate) fn open_fixture() -> VideoSource {
    let options = OpenOptions::default();
    let fixture = fixture();

    VideoSource::open(&fixture, &options).expect("opens the fixture")
}

#[test]
fn opening_the_fixture_reads_its_properties() {
    let source = open_fixture();
    let properties = source.properties();

    assert_eq!(properties.num_frames, 8);
    assert_eq!((properties.width, properties.height), (64, 64));
    assert_eq!((properties.ss_mod_width, properties.ss_mod_height), (64, 64));
    assert_eq!(properties.fps, (25, 1));
    assert_eq!(properties.format.color_family, ColorFamily::Yuv);
    assert_eq!(properties.format.bits, 8);
    assert!(!properties.format.float);
    assert!(!properties.format.alpha);
    assert_eq!((properties.format.sub_sampling_w, properties.format.sub_sampling_h), (1, 1));
}

#[test]
fn a_missing_file_names_the_file() {
    let folder = tempfile::tempdir().expect("a temp dir");
    let missing = folder.path().join("missing.mkv");
    let options = OpenOptions::default();

    let error = VideoSource::open(&missing, &options).expect_err("a missing file opened");

    assert!(error.message().contains("missing.mkv"), "{error}");
}

#[test]
fn a_track_the_file_does_not_have_is_an_error() {
    let options = OpenOptions {
        track: 5,
        ..OpenOptions::default()
    };
    let fixture = fixture();

    let error = VideoSource::open(&fixture, &options).expect_err("track 5 opened");

    assert!(error.message().contains("Invalid track index"), "{error}");
}

#[test]
fn a_path_with_a_nul_byte_is_refused() {
    let path = Path::new("clip\0.mkv");
    let options = OpenOptions::default();

    let error = VideoSource::open(path, &options).expect_err("a nul path opened");

    assert!(error.message().contains("nul byte"), "{error}");
}
