use std::path::PathBuf;

use av_toolbox_core::{Seeker, Source, SourceDefinition};
use av_toolbox_gpu::{Channel, Depth, FrameLayout, HostFrame, SampleFormat, testing_context};

use crate::bestsource::file::{File, FileArgs, FileSource, Opened};

fn copied(name: &str) -> (tempfile::TempDir, PathBuf) {
    let folder = tempfile::tempdir().expect("a temp dir");
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let media = manifest.join("tests").join("media").join(name);
    let source = folder.path().join(name);
    std::fs::copy(media, &source).expect("copies");

    (folder, source)
}

fn args_for(path: PathBuf) -> FileArgs {
    FileArgs {
        path,
        track: None,
        index_dir: None,
        threads: None,
    }
}

fn opened(name: &str) -> (tempfile::TempDir, Opened) {
    let (folder, source) = copied(name);
    let opened = File::open(&args_for(source)).expect("opens");

    (folder, opened)
}

fn decode_all(opened: Opened) -> Vec<HostFrame> {
    let mut context = testing_context();
    let mut source = FileSource::new(opened);
    let mut frames = Vec::new();
    while let Some(frame) = source.next(&mut context).expect("decodes") {
        frames.push(frame);
    }

    frames
}

fn seeker_over(opened: &Opened) -> Box<dyn Seeker> {
    let found = File::seeker(opened).expect("opens a second source");

    found.expect("bestsource can seek")
}

fn channels_of(frame: &HostFrame) -> Vec<Channel> {
    match frame.format().layout {
        FrameLayout::Yuv { .. } => vec![Channel::Y, Channel::U, Channel::V],
        FrameLayout::Grey => vec![Channel::Y],
        FrameLayout::Rgb => vec![Channel::R, Channel::G, Channel::B],
        FrameLayout::Chroma { .. } => unreachable!("never opened as chroma planes"),
    }
}

#[cfg(feature = "ffms2")]
fn assert_matches_ffms2(name: &str) {
    use crate::ffms2;

    let (_best_folder, best) = opened(name);
    let (_ffms2_folder, source) = copied(name);
    let ffms2_args = ffms2::FileArgs {
        path: source,
        track: None,
        index_dir: None,
        format: None,
        threads: None,
    };
    let ffms2_opened = ffms2::File::open(&ffms2_args).expect("ffms2 opens");

    let best_frames = decode_all(best);
    let mut context = testing_context();
    let mut ffms2_source = ffms2::FileSource::new(ffms2_opened);
    let mut ffms2_frames = Vec::new();
    while let Some(frame) = ffms2_source.next(&mut context).expect("ffms2 decodes") {
        ffms2_frames.push(frame);
    }

    assert_eq!(best_frames.len(), ffms2_frames.len());
    for (index, (best_frame, ffms2_frame)) in best_frames.iter().zip(&ffms2_frames).enumerate() {
        assert_eq!(best_frame.format().layout, ffms2_frame.format().layout, "{index}");
        assert_eq!(best_frame.format().sample, ffms2_frame.format().sample, "{index}");

        for channel in channels_of(best_frame) {
            let best_plane = best_frame.plane(channel).expect("carried");
            let ffms2_plane = ffms2_frame.plane(channel).expect("carried");

            assert_eq!(best_plane.bytes(), ffms2_plane.bytes(), "{index} {channel}");
        }
    }
}

#[cfg(feature = "ffms2")]
#[test]
fn a_planar_file_matches_ffms2_frame_for_frame() {
    assert_matches_ffms2("yuv420p.mkv");
}

#[cfg(feature = "ffms2")]
#[test]
fn rgb_matches_ffms2_plane_for_plane() {
    assert_matches_ffms2("rgb24.nut");
}

#[cfg(feature = "ffms2")]
#[test]
fn nv12_matches_ffms2_plane_for_plane() {
    assert_matches_ffms2("nv12.nut");
}

#[test]
fn a_ten_bit_file_keeps_its_depth() {
    let (_folder, opened) = opened("yuv420p10.mkv");
    let depth = Depth::new(10).expect("a valid depth");

    let format = *opened.format();
    let frames = decode_all(opened);

    assert_eq!(format.sample, SampleFormat::U16 { depth });
    for frame in &frames {
        let luma = frame.plane(Channel::Y).expect("luma");

        assert_eq!(luma.bytes().len(), 64 * 64 * 2);
    }
}

#[test]
fn the_seeker_reads_what_the_stream_decodes() {
    let (_folder, opened) = opened("yuv420p.mkv");
    let mut seeker = seeker_over(&opened);
    let frames = decode_all(opened);

    let read = seeker.read(3).expect("reads");

    for channel in [Channel::Y, Channel::U, Channel::V] {
        let read_plane = read.plane(channel).expect("carried");
        let streamed_plane = frames[3].plane(channel).expect("carried");

        assert_eq!(read_plane.bytes(), streamed_plane.bytes(), "{channel}");
    }
}

#[test]
fn reading_past_the_end_is_refused() {
    let (_folder, opened) = opened("yuv420p.mkv");
    let mut seeker = seeker_over(&opened);
    let frame_count = seeker.frame_count();

    let refused = seeker.read(frame_count);

    assert!(refused.is_err());
}

#[test]
fn the_stream_ends_after_the_last_frame() {
    let (_folder, opened) = opened("yuv420p.mkv");
    let mut context = testing_context();
    let mut source = FileSource::new(opened);

    let mut count = 0;
    while source.next(&mut context).expect("decodes").is_some() {
        count += 1;
    }

    assert_eq!(count, 8);
    assert!(source.next(&mut context).expect("ends").is_none());
}

#[test]
fn the_index_is_written_beside_the_file_by_default() {
    let (_folder, source) = copied("yuv420p.mkv");
    let index = PathBuf::from(format!("{}.0.bsindex", source.display()));

    File::open(&args_for(source)).expect("opens");

    assert!(index.exists(), "{}", index.display());
}

#[test]
fn an_index_dir_holds_the_index_instead() {
    let (folder, source) = copied("yuv420p.mkv");
    let index_dir = tempfile::tempdir().expect("a temp dir");
    let mut args = args_for(source);
    args.index_dir = Some(index_dir.path().to_path_buf());

    File::open(&args).expect("opens");

    assert!(contains_index(index_dir.path()));
    assert!(!contains_index(folder.path()));
}

fn contains_index(folder: &std::path::Path) -> bool {
    let entries = std::fs::read_dir(folder).expect("reads the folder");

    entries.flatten().any(|entry| {
        let path = entry.path();
        let is_index = path.extension().is_some_and(|extension| extension == "bsindex");

        is_index || (path.is_dir() && contains_index(&path))
    })
}

#[test]
fn a_missing_file_names_it() {
    let folder = tempfile::tempdir().expect("a temp dir");
    let args = args_for(folder.path().join("missing.mkv"));

    let error = File::open(&args).err().expect("the file is missing");

    assert!(error.to_string().contains("missing.mkv"));
}
