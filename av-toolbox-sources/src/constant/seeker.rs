use av_toolbox_core::{Seeker, StageError};
use av_toolbox_gpu::{FrameFormat, HostFrame, HostPool};

use super::convert::Converted;
use super::repeat::meta_at;

/// Reads any frame of a clip that repeats one picture.
pub(crate) struct RepeatSeeker {
    converted: Converted,
    frames: u64,
    pool: HostPool,
}

impl RepeatSeeker {
    pub(crate) fn new(converted: Converted, frames: u64) -> Self {
        Self {
            converted,
            frames,
            pool: HostPool::new(),
        }
    }
}

fn past_end(index: usize, frames: u64) -> StageError {
    let message = format!("the clip has {frames} frames, so frame {index} cannot be read");

    StageError::msg(message)
}

impl Seeker for RepeatSeeker {
    fn format(&self) -> &FrameFormat {
        &self.converted.format
    }

    fn frame_count(&self) -> usize {
        usize::try_from(self.frames).unwrap_or(usize::MAX)
    }

    fn read(&mut self, index: usize) -> Result<HostFrame, StageError> {
        if index >= self.frame_count() {
            return Err(past_end(index, self.frames));
        }

        let format = &self.converted.format;
        let meta = meta_at(index as u64, format.frame_rate);
        let mut host = self.pool.alloc_frame(format, meta);

        for (channel, bytes) in format.channels().iter().zip(&self.converted.planes) {
            let plane = host
                .plane_mut(*channel)
                .expect("a frame carries every channel its format lists");
            plane.copy_from_slice(bytes);
        }

        Ok(host.freeze())
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::time::Duration;

    use av_toolbox_core::{Seeker, SourceDefinition};
    use av_toolbox_gpu::HostFrame;
    use image::{GrayImage, Luma};
    use serde_json::json;

    use crate::constant::{File, Fill, Prepared};

    fn fill_prepared() -> Prepared {
        let value = json!({
            "width": 4,
            "height": 4,
            "frames": 4,
            "colour": "#336699",
            "frame_rate": [24, 1],
        });
        let args = serde_json::from_value(value).expect("parses");

        Fill::open(&args).expect("opens")
    }

    fn image_prepared(folder: &Path) -> Prepared {
        let path = folder.join("small.png");
        let small = GrayImage::from_pixel(8, 4, Luma([200]));
        small.save(&path).expect("writes the image");

        let value = json!({
            "path": path,
            "frames": 4,
            "frame_rate": [24, 1],
        });
        let args = serde_json::from_value(value).expect("parses");

        File::open(&args).expect("opens")
    }

    fn assert_repeats(prepared: &Prepared, mut seeker: Box<dyn Seeker>) {
        assert_eq!(seeker.frame_count(), 4);

        let frame: HostFrame = seeker.read(3).expect("reads");

        assert_eq!(frame.meta().index, 3);
        assert_eq!(frame.meta().timestamp, Some(Duration::from_millis(125)));

        let channels = prepared.converted.format.channels();
        for (channel, bytes) in channels.iter().zip(&prepared.converted.planes) {
            let plane = frame.plane(*channel).expect("carried");

            assert_eq!(plane.bytes(), bytes.as_slice());
        }

        assert!(seeker.read(4).is_err());
    }

    #[test]
    fn solid_fill_seeks_any_frame() {
        let prepared = fill_prepared();
        let seeker = Fill::seeker(&prepared).expect("builds").expect("seeks");

        assert_repeats(&prepared, seeker);
    }

    #[test]
    fn image_file_seeks_any_frame() {
        let folder = tempfile::tempdir().expect("a folder");
        let prepared = image_prepared(folder.path());
        let seeker = File::seeker(&prepared).expect("builds").expect("seeks");

        assert_repeats(&prepared, seeker);
    }
}
