use std::path::{Path, PathBuf};

use crate::{ColorFamily, IndexCache, OpenOptions, VideoSource};

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
    assert_eq!(
        (properties.format.sub_sampling_w, properties.format.sub_sampling_h),
        (1, 1)
    );
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

fn find_index(directory: &Path) -> Option<PathBuf> {
    let entries = std::fs::read_dir(directory).ok()?;

    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if let Some(found) = find_index(&path) {
                return Some(found);
            }

            continue;
        }

        let is_index = path.extension().is_some_and(|extension| extension == "bsindex");
        if is_index {
            return Some(path);
        }
    }

    None
}

fn modified(path: &Path) -> std::time::SystemTime {
    let metadata = std::fs::metadata(path).expect("reads the index metadata");

    metadata.modified().expect("reads the index modified time")
}

#[test]
fn a_directory_cache_writes_an_index_and_reuses_it() {
    let folder = tempfile::tempdir().expect("a temp dir");
    let cache = folder.path().join("indexes");
    let options = OpenOptions {
        cache: IndexCache::Directory(cache.clone()),
        ..OpenOptions::default()
    };
    let fixture = fixture();

    let first = VideoSource::open(&fixture, &options).expect("first open");
    drop(first);
    let index = find_index(&cache).expect("an index under the cache directory");
    let written_at = modified(&index);

    let second = VideoSource::open(&fixture, &options).expect("second open");
    drop(second);

    assert_eq!(modified(&index), written_at);
}

#[test]
fn a_beside_source_cache_writes_the_index_next_to_the_file() {
    let folder = tempfile::tempdir().expect("a temp dir");
    let source = folder.path().join("clip.mkv");
    let fixture = fixture();
    std::fs::copy(fixture, &source).expect("copies the fixture");
    let options = OpenOptions {
        cache: IndexCache::BesideSource,
        ..OpenOptions::default()
    };

    let opened = VideoSource::open(&source, &options).expect("opens");
    drop(opened);

    let index = folder.path().join("clip.mkv.0.bsindex");
    assert!(index.exists(), "no index at {}", index.display());
}
