use std::mem::ManuallyDrop;
use std::path::Path;
use std::ptr::NonNull;

use ffms2_sys::{FFMS_Index, FFMS_IndexErrorHandling, FFMS_Indexer, FFMS_TrackType};

use super::error::{ErrorBuffer, Ffms2Error};
use super::{c_path, init};

/// An FFMS2 index of the tracks in a source file.
pub(crate) struct Index {
    handle: NonNull<FFMS_Index>,
}

// SAFETY: FFMS2 handles are not tied to the thread that created them.
unsafe impl Send for Index {}

impl Index {
    pub(crate) fn read(path: &Path) -> Result<Self, Ffms2Error> {
        init();

        let c_index = c_path(path)?;
        let mut errors = ErrorBuffer::new();
        let info = errors.info();

        // SAFETY: `c_index` and `info` outlive the call.
        let raw = unsafe { ffms2_sys::FFMS_ReadIndex(c_index.as_ptr(), info) };
        let handle = NonNull::new(raw).ok_or_else(|| errors.into_error())?;

        Ok(Self { handle })
    }

    /// Succeeds when this index was made from `source` as it is now.
    pub(crate) fn belongs_to(&self, source: &Path) -> Result<(), Ffms2Error> {
        let c_source = c_path(source)?;
        let mut errors = ErrorBuffer::new();
        let info = errors.info();

        // SAFETY: `self.handle` is a live index, and `c_source` and `info` outlive the call.
        let status =
            unsafe { ffms2_sys::FFMS_IndexBelongsToFile(self.handle.as_ptr(), c_source.as_ptr(), info) };

        errors.check(status)
    }

    pub(crate) fn write(&self, path: &Path) -> Result<(), Ffms2Error> {
        let c_index = c_path(path)?;
        let mut errors = ErrorBuffer::new();
        let info = errors.info();

        // SAFETY: `self.handle` is a live index, and `c_index` and `info` outlive the call.
        let status = unsafe { ffms2_sys::FFMS_WriteIndex(c_index.as_ptr(), self.handle.as_ptr(), info) };

        errors.check(status)
    }

    pub(crate) fn first_video_track(&self) -> Result<i32, Ffms2Error> {
        let video_type = FFMS_TrackType::FFMS_TYPE_VIDEO as i32;
        let mut errors = ErrorBuffer::new();
        let info = errors.info();

        // SAFETY: `self.handle` is a live index, and `info` outlives the call.
        let track = unsafe { ffms2_sys::FFMS_GetFirstTrackOfType(self.handle.as_ptr(), video_type, info) };

        if track < 0 {
            return Err(errors.into_error());
        }

        Ok(track)
    }

    pub(super) fn as_ptr(&self) -> *mut FFMS_Index {
        self.handle.as_ptr()
    }
}

impl Drop for Index {
    fn drop(&mut self) {
        // SAFETY: `self.handle` is a live index that nothing else frees.
        unsafe { ffms2_sys::FFMS_DestroyIndex(self.handle.as_ptr()) };
    }
}

/// Reads a source file to build its [Index](crate::ffms2::sys::Index).
pub(crate) struct Indexer {
    handle: NonNull<FFMS_Indexer>,
}

// SAFETY: FFMS2 handles are not tied to the thread that created them.
unsafe impl Send for Indexer {}

impl Indexer {
    pub(crate) fn new(source: &Path) -> Result<Self, Ffms2Error> {
        init();

        let c_source = c_path(source)?;
        let mut errors = ErrorBuffer::new();
        let info = errors.info();

        // SAFETY: `c_source` and `info` outlive the call.
        let raw = unsafe { ffms2_sys::FFMS_CreateIndexer(c_source.as_ptr(), info) };
        let handle = NonNull::new(raw).ok_or_else(|| errors.into_error())?;

        Ok(Self { handle })
    }

    /// Indexes every track, stopping at the first decoding error.
    pub(crate) fn index(self) -> Result<Index, Ffms2Error> {
        let indexer = ManuallyDrop::new(self);
        let abort = FFMS_IndexErrorHandling::FFMS_IEH_ABORT as i32;
        let mut errors = ErrorBuffer::new();
        let info = errors.info();

        // SAFETY: FFMS2 frees the indexer during this call whether it succeeds or not, and
        // `ManuallyDrop` stops it being cancelled again.
        let raw = unsafe { ffms2_sys::FFMS_DoIndexing2(indexer.handle.as_ptr(), abort, info) };
        let handle = NonNull::new(raw).ok_or_else(|| errors.into_error())?;

        Ok(Index { handle })
    }
}

impl Drop for Indexer {
    fn drop(&mut self) {
        // SAFETY: `self.handle` is a live indexer that was never passed to `FFMS_DoIndexing2`.
        unsafe { ffms2_sys::FFMS_CancelIndexing(self.handle.as_ptr()) };
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    fn media(name: &str) -> PathBuf {
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

        manifest.join("tests").join("media").join(name)
    }

    fn indexed(name: &str) -> Index {
        let source = media(name);
        let indexer = Indexer::new(&source).expect("creates an indexer");

        indexer.index().expect("indexes")
    }

    #[test]
    fn an_indexed_file_has_a_first_video_track() {
        let index = indexed("yuv420p.mkv");

        assert_eq!(index.first_video_track(), Ok(0));
    }

    #[test]
    fn a_written_index_reads_back_and_belongs_to_its_file() {
        let folder = tempfile::tempdir().expect("a temp dir");
        let index_file = folder.path().join("yuv420p.ffindex");
        let index = indexed("yuv420p.mkv");

        index.write(&index_file).expect("writes");
        let read_back = Index::read(&index_file).expect("reads");

        assert!(read_back.belongs_to(&media("yuv420p.mkv")).is_ok());
    }

    #[test]
    fn an_index_does_not_belong_to_another_file() {
        let index = indexed("yuv420p.mkv");

        assert!(index.belongs_to(&media("nv12.nut")).is_err());
    }

    #[test]
    fn a_missing_file_fails_with_its_whole_message() {
        let error = Indexer::new(Path::new("/nonexistent/missing.mkv"))
            .err()
            .expect("the file is missing");

        assert!(error.to_string().contains("/nonexistent/missing.mkv"), "{error}");
    }

    #[test]
    fn reading_a_missing_index_is_an_error() {
        let read = Index::read(Path::new("/nonexistent/missing.ffindex"));

        assert!(read.is_err());
    }

    #[test]
    fn a_file_that_is_not_media_fails_to_index_cleanly() {
        let folder = tempfile::tempdir().expect("a temp dir");
        let source = folder.path().join("noise.mkv");
        std::fs::write(&source, [0x5a; 4096]).expect("writes noise");

        let indexed = Indexer::new(&source).and_then(Indexer::index);

        assert!(indexed.is_err());
    }

    #[test]
    fn an_unused_indexer_drops_cleanly() {
        let indexer = Indexer::new(&media("yuv420p.mkv")).expect("creates an indexer");

        drop(indexer);
    }
}
