use av_toolbox_core::{Artefact, ArtefactData, ArtefactPiece, Filter, PipelineRegistry, WantedArtefacts};
use av_toolbox_gpu::{
    BuildContext,
    BuildTask,
    ChromaLocation,
    FrameFormat,
    FrameLayout,
    FrameMeta,
    FrameRate,
    SampleFormat,
};
use cubecl::prelude::Runtime;
use serde_json::json;

use super::{noisy_clip, yuv420};
use crate::stream::{Input, run_chunk};
use crate::support::{backend, built_filter, frame};

const STAGE: &str = "av-denoise/nl4d";

/// Frames in each scene of [grainy_clip].
const SCENE_LENGTH: u64 = 30;

/// Runs every chunk through one filter, taking its artefacts at the end of each.
struct GrainStream {
    filter: Box<dyn Filter>,
    format: FrameFormat,
    chunks: Vec<Vec<Input>>,
}

impl BuildTask for GrainStream {
    type Output = Vec<Vec<Artefact>>;

    fn run<R: Runtime>(mut self, build: &BuildContext<R>) -> Self::Output {
        let mut context = build.frame_context();
        let mut taken = Vec::new();

        for chunk in self.chunks {
            let formats = (&self.format, &self.format);
            run_chunk(build, &mut context, self.filter.as_mut(), formats, chunk);

            let artefacts = self
                .filter
                .take_artefacts(&mut context)
                .expect("should hand over");
            taken.push(artefacts);
        }

        taken
    }
}

fn at_25_fps(mut format: FrameFormat) -> FrameFormat {
    format.frame_rate = FrameRate::new(25, 1);
    format
}

fn grain_wanted() -> WantedArtefacts {
    WantedArtefacts::new(vec!["grain_table"])
}

/// Repeatable noise from `seed`, one byte per sample.
fn pattern(seed: u32, samples: usize) -> Vec<u8> {
    let mut state = seed.max(1);
    let mut bytes = Vec::with_capacity(samples);

    for _ in 0..samples {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        bytes.push((state & 0xff) as u8);
    }

    bytes
}

/// Three flat luma bands under fresh grain every frame, swapping sides at every scene, as one chunk per scene.
fn grainy_clip(format: &FrameFormat, scenes: u64) -> Vec<Vec<Input>> {
    let width = format.width as usize;
    let height = format.height as usize;
    let mut chunks = Vec::new();

    for scene in 0..scenes {
        let levels: [i16; 3] = if scene % 2 == 0 {
            [60, 120, 180]
        } else {
            [180, 120, 60]
        };
        let mut chunk = Vec::new();

        for offset in 0..SCENE_LENGTH {
            let index = scene * SCENE_LENGTH + offset;
            let seed = index as u32 * 7919 + 12345;
            let grain = pattern(seed, width * height);
            let luma: Vec<u8> = grain
                .iter()
                .enumerate()
                .map(|(pixel, sample)| {
                    let band = (pixel % width) * 3 / width;
                    let wobble = i16::from(*sample % 9) - 4;
                    (levels[band] + wobble) as u8
                })
                .collect();

            let mut planes = vec![luma];
            for channel in format.channels().iter().skip(1) {
                let samples = format.channel_samples(*channel).unwrap_or(0);
                planes.push(vec![128u8; samples]);
            }

            chunk.push(Input {
                meta: FrameMeta::at(index),
                planes,
            });
        }

        chunks.push(chunk);
    }

    chunks
}

fn run_wanting(format: FrameFormat, wanted: &WantedArtefacts, chunks: Vec<Vec<Input>>) -> Vec<Vec<Artefact>> {
    let backend = backend();
    let mut filter = built_filter(&backend, STAGE, format, format, json!({}));
    filter.wants_artefacts(wanted).expect("accepts");

    let stream = GrainStream {
        filter,
        format,
        chunks,
    };

    backend.build(stream)
}

fn merged_table(pieces: Vec<ArtefactPiece>) -> String {
    let mut registry = PipelineRegistry::new();
    av_toolbox_filters::register(&mut registry).expect("the filters register");
    let info = registry
        .stages()
        .find(|info| info.full_identifier() == STAGE)
        .expect("nl4d is registered");

    let merged = info.artefacts().merge("grain_table", pieces).expect("merges");
    let ArtefactData::Opaque(bytes) = merged else {
        panic!("the grain table is opaque");
    };

    String::from_utf8(bytes).expect("the table is text")
}

#[test]
fn wanting_a_grain_table_with_only_chroma_is_refused() {
    let format = at_25_fps(yuv420(64, 32, SampleFormat::U8));
    let backend = backend();
    let mut filter = built_filter(&backend, STAGE, format, format, json!({ "planes": ["u", "v"] }));

    let error = filter.wants_artefacts(&grain_wanted()).expect_err("refused");

    assert!(error.to_string().contains("leave luma out"), "{error}");
}

#[test]
fn wanting_a_grain_table_without_a_frame_rate_is_refused() {
    let format = yuv420(64, 32, SampleFormat::U8);
    let backend = backend();
    let mut filter = built_filter(&backend, STAGE, format, format, json!({}));

    let error = filter.wants_artefacts(&grain_wanted()).expect_err("refused");

    assert!(error.to_string().contains("frame rate"), "{error}");
}

#[test]
fn a_grain_table_not_wanted_hands_nothing_over() {
    let format = at_25_fps(yuv420(64, 32, SampleFormat::U8));
    let clip = noisy_clip(&format, 6);
    let nothing = WantedArtefacts::default();

    let taken = run_wanting(format, &nothing, vec![clip]);

    assert_eq!(taken, vec![Vec::<Artefact>::new()]);
}

#[test]
fn each_scene_hands_over_a_piece_starting_at_its_first_frame() {
    let format = at_25_fps(yuv420(192, 128, SampleFormat::U8));
    let chunks = grainy_clip(&format, 2);

    let taken = run_wanting(format, &grain_wanted(), chunks);

    assert_eq!(taken.len(), 2);

    let mut pieces = Vec::new();
    for (scene, artefacts) in taken.into_iter().enumerate() {
        assert_eq!(artefacts.len(), 1);

        let artefact = artefacts.into_iter().next().expect("one artefact");
        assert_eq!(artefact.name, "grain_table");

        let ArtefactData::Structured(value) = &artefact.data else {
            panic!("a piece is structured");
        };
        let first_frame = scene as u64 * SCENE_LENGTH;
        assert_eq!(value["frame_rate"], json!([25, 1]));
        assert_eq!(value["scenes"][0]["first_frame"], json!(first_frame), "{value}");
        assert!(
            !value["scenes"][0]["chunks"]
                .as_array()
                .expect("chunks")
                .is_empty()
        );

        pieces.push(ArtefactPiece {
            chunk: Some(scene as u64),
            data: artefact.data,
        });
    }

    let table = merged_table(pieces);

    assert!(table.starts_with("filmgrn1\n"), "{table}");
    assert!(table.contains("\nE "), "{table}");
}

#[test]
fn a_grey_clip_can_export_grain() {
    let grey = frame(
        192,
        128,
        FrameLayout::Grey,
        SampleFormat::U8,
        ChromaLocation::Left,
    );
    let format = at_25_fps(grey);
    let chunks = grainy_clip(&format, 1);

    let taken = run_wanting(format, &grain_wanted(), chunks);

    assert_eq!(taken.len(), 1);
    assert_eq!(taken[0].len(), 1);
}
