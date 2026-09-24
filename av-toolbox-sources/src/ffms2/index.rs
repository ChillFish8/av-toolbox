use std::ffi::OsStr;
use std::path::{Path, PathBuf};

use av_toolbox_core::StageError;

use super::sys::{Ffms2Error, Index, Indexer};

/// The path an index for a source file lives at.
///
/// Without an index directory the index sits beside the source. With one, the source's
/// canonical path is hashed into the name so files sharing a name in different folders do not
/// collide.
pub(crate) fn index_path(source: &Path, index_dir: Option<&Path>) -> Result<PathBuf, StageError> {
    let Some(index_dir) = index_dir else {
        let mut beside_source = source.as_os_str().to_owned();
        beside_source.push(".ffindex");

        return Ok(PathBuf::from(beside_source));
    };

    let canonicalized = source.canonicalize();
    let canonical = canonicalized.map_err(|error| {
        let display = source.display();
        let message = format!("could not canonicalise {display}: {error}");

        StageError::msg(message)
    })?;

    let file_name = source.file_name().ok_or_else(|| {
        let display = source.display();
        let message = format!("{display} has no file name");

        StageError::msg(message)
    })?;
    let indexed_name = keyed_name(file_name, &canonical);
    let path = index_dir.join(indexed_name);

    Ok(path)
}

/// The index file name for `file_name`, keyed by a 64-bit FNV-1a hash of its canonical path.
///
/// The hash is written out here rather than taken from the standard library, whose hashers can
/// change between Rust releases and would orphan every index already written.
fn keyed_name(file_name: &OsStr, canonical: &Path) -> String {
    const OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;

    let bytes = canonical.as_os_str().as_encoded_bytes();
    let mut hash = OFFSET_BASIS;

    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(PRIME);
    }

    let name = file_name.to_string_lossy();

    format!("{name}.{hash:016x}.ffindex")
}

fn indexing_error(source: &Path, error: &Ffms2Error) -> StageError {
    let display = source.display();
    let message = format!("ffms2 could not index {display}: {error}");

    StageError::msg(message)
}

/// Loads a usable index for the source file, building and writing one when needed.
///
/// An existing index is reused when FFMS2 confirms it still belongs to the source. Otherwise the
/// source is indexed again. When the rebuilt index cannot be written, it is still returned and
/// used from memory.
pub(crate) fn load_or_build(source: &Path, index_dir: Option<&Path>) -> Result<Index, StageError> {
    let path = index_path(source, index_dir)?;

    if path.exists() {
        let existing = Index::read(&path);

        if let Ok(existing) = existing {
            let belongs = existing.belongs_to(source);

            if belongs.is_ok() {
                return Ok(existing);
            }
        }
    }

    let created = Indexer::new(source);
    let indexer = created.map_err(|error| indexing_error(source, &error))?;

    let indexed = indexer.index();
    let index = indexed.map_err(|error| indexing_error(source, &error))?;

    let written = index.write(&path);

    if let Err(error) = written {
        let path_display = path.display();
        tracing::warn!(
            path = %path_display,
            %error,
            "the ffms2 index could not be written, continuing with it in memory"
        );
    }

    Ok(index)
}

#[cfg(test)]
mod tests {
    use std::ffi::OsStr;
    use std::path::PathBuf;

    use super::*;

    fn media(name: &str) -> PathBuf {
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

        manifest.join("tests").join("media").join(name)
    }

    fn copied(folder: &tempfile::TempDir, name: &str) -> PathBuf {
        let target = folder.path().join(name);
        let media = media(name);
        std::fs::copy(media, &target).expect("copies the media");

        target
    }

    #[test]
    fn the_index_is_written_beside_the_file_by_default() {
        let folder = tempfile::tempdir().expect("a temp dir");
        let source = copied(&folder, "yuv420p.mkv");

        load_or_build(&source, None).expect("indexes");

        assert!(folder.path().join("yuv420p.mkv.ffindex").exists());
    }

    #[test]
    fn an_existing_index_is_reused_unchanged() {
        let folder = tempfile::tempdir().expect("a temp dir");
        let source = copied(&folder, "yuv420p.mkv");
        load_or_build(&source, None).expect("indexes");
        let index_file = folder.path().join("yuv420p.mkv.ffindex");
        let written = std::fs::metadata(&index_file)
            .expect("exists")
            .modified()
            .expect("mtime");

        load_or_build(&source, None).expect("reuses");

        let after = std::fs::metadata(&index_file)
            .expect("exists")
            .modified()
            .expect("mtime");
        assert_eq!(written, after);
    }

    #[test]
    fn an_index_dir_holds_a_name_keyed_by_the_source_path() {
        let folder = tempfile::tempdir().expect("a temp dir");
        let index_dir = tempfile::tempdir().expect("an index dir");
        let source = copied(&folder, "yuv420p.mkv");

        let path = index_path(&source, Some(index_dir.path())).expect("a path");

        assert_eq!(path.parent(), Some(index_dir.path()));
        let name = path.file_name().expect("a name").to_string_lossy().to_string();
        assert!(name.starts_with("yuv420p.mkv."));
        assert!(name.ends_with(".ffindex"));
    }

    #[test]
    fn a_known_path_gives_a_known_index_name() {
        let name = keyed_name(OsStr::new("clip.mkv"), Path::new("/media/clip.mkv"));

        assert_eq!(name, "clip.mkv.90986a809d219fa1.ffindex");
    }

    #[test]
    fn a_stale_index_is_rebuilt() {
        let folder = tempfile::tempdir().expect("a temp dir");
        let source = copied(&folder, "yuv420p.mkv");
        let other = copied(&folder, "nv12.nut");
        load_or_build(&other, None).expect("indexes the other file");
        let other_index = folder.path().join("nv12.nut.ffindex");
        let planted = folder.path().join("yuv420p.mkv.ffindex");
        std::fs::copy(other_index, planted).expect("plants an index for the wrong file");

        let index = load_or_build(&source, None).expect("rebuilds");

        assert!(index.belongs_to(&source).is_ok());
    }

    #[test]
    fn a_garbage_index_is_rebuilt() {
        let folder = tempfile::tempdir().expect("a temp dir");
        let source = copied(&folder, "yuv420p.mkv");
        let planted = folder.path().join("yuv420p.mkv.ffindex");
        std::fs::write(&planted, b"not an index").expect("plants garbage");

        let index = load_or_build(&source, None).expect("rebuilds");

        assert!(index.belongs_to(&source).is_ok());
    }

    #[test]
    fn an_unwritable_index_dir_still_opens_the_file() {
        let folder = tempfile::tempdir().expect("a temp dir");
        let source = copied(&folder, "yuv420p.mkv");
        let missing = folder.path().join("does-not-exist");

        let index = load_or_build(&source, Some(&missing));

        assert!(index.is_ok());
    }
}
