use av_denoise_core::{GrainChunk, SceneGrain};
use av_toolbox_core::{ArtefactData, ArtefactPiece};
use serde_json::json;

use crate::denoise::avd::grain::{GRAIN_TABLE, GrainLog, GrainPiece, joined_scenes, merge_grain_table};

fn measured_chunk(frames: u32) -> GrainChunk {
    let mut chunk = GrainChunk::empty();
    chunk.frames = frames;
    chunk.source_hist[3] = 5;
    chunk.kept_hist[7] = 2;
    chunk.autocov[0] = 0.1 + 0.2;
    chunk.pixels[1] = 1.0 / 3.0;

    chunk
}

fn piece_of(frame_rate: (u64, u64), scenes: &[SceneGrain]) -> ArtefactPiece {
    let grain = GrainPiece::new(frame_rate, scenes);
    let artefact = grain.to_artefact().expect("encodes");

    ArtefactPiece {
        chunk: Some(0),
        data: artefact.data,
    }
}

#[test]
fn a_piece_round_trips_through_json_to_the_same_chunks() {
    let scenes = vec![SceneGrain {
        first_frame: 48,
        chunks: vec![measured_chunk(24), measured_chunk(6)],
    }];
    let piece = piece_of((24000, 1001), &scenes);

    let decoded = GrainPiece::from_piece(piece).expect("decodes");

    assert_eq!(decoded.frame_rate(), (24000, 1001));
    assert_eq!(decoded.into_scenes(), scenes);
}

#[test]
fn a_piece_is_handed_over_as_structured_grain_table() {
    let grain = GrainPiece::new((25, 1), &[]);

    let artefact = grain.to_artefact().expect("encodes");

    assert_eq!(artefact.name, GRAIN_TABLE);
    assert_eq!(
        artefact.data,
        ArtefactData::Structured(json!({ "frame_rate": [25, 1], "scenes": [] }))
    );
}

#[test]
fn the_merge_joins_the_scenes_of_every_piece() {
    let later = vec![SceneGrain {
        first_frame: 30,
        chunks: vec![measured_chunk(24)],
    }];
    let earlier = vec![SceneGrain {
        first_frame: 0,
        chunks: vec![measured_chunk(24)],
    }];
    let pieces = vec![piece_of((25, 1), &later), piece_of((25, 1), &earlier)];

    let (frame_rate, scenes) = joined_scenes(pieces).expect("joins");
    let first_frames: Vec<u64> = scenes.iter().map(|scene| scene.first_frame).collect();

    assert_eq!(frame_rate, (25, 1));
    assert_eq!(first_frames, vec![30, 0]);
}

#[test]
fn pieces_at_different_frame_rates_are_refused() {
    let pieces = vec![piece_of((25, 1), &[]), piece_of((24, 1), &[])];

    let error = merge_grain_table(pieces).expect_err("refused");

    assert!(error.to_string().contains("frames per second"), "{error}");
}

#[test]
fn pieces_with_no_scenes_merge_to_an_empty_table() {
    let pieces = vec![piece_of((25, 1), &[]), piece_of((25, 1), &[])];

    let merged = merge_grain_table(pieces).expect("merges");

    assert_eq!(merged, ArtefactData::Opaque(b"filmgrn1\n".to_vec()));
}

#[test]
fn a_piece_that_does_not_decode_is_refused_naming_the_artefact() {
    let piece = ArtefactPiece {
        chunk: None,
        data: ArtefactData::Structured(json!({ "frame_rate": "fast" })),
    };

    let error = merge_grain_table(vec![piece]).expect_err("refused");

    assert!(error.to_string().contains(GRAIN_TABLE), "{error}");
}

#[test]
fn an_opaque_piece_is_refused_naming_the_artefact() {
    let piece = ArtefactPiece {
        chunk: None,
        data: ArtefactData::Opaque(b"filmgrn1\n".to_vec()),
    };

    let error = merge_grain_table(vec![piece]).expect_err("refused");

    assert!(error.to_string().contains(GRAIN_TABLE), "{error}");
}

#[test]
fn a_scene_starts_at_the_first_frame_seen() {
    let mut log = GrainLog::new(0, (25, 1));

    log.restart();
    log.saw(12);
    log.saw(13);
    log.keep(vec![measured_chunk(2)]);
    let artefact = log.take_piece().expect("encodes");
    let piece = ArtefactPiece {
        chunk: Some(1),
        data: artefact.data,
    };
    let scenes = GrainPiece::from_piece(piece).expect("decodes").into_scenes();

    assert_eq!(scenes.len(), 1);
    assert_eq!(scenes[0].first_frame, 12);
}

#[test]
fn a_stream_with_no_chunks_keeps_no_scene() {
    let mut log = GrainLog::new(0, (25, 1));

    log.restart();
    log.saw(0);
    log.keep(Vec::new());
    let artefact = log.take_piece().expect("encodes");

    assert_eq!(
        artefact.data,
        ArtefactData::Structured(json!({ "frame_rate": [25, 1], "scenes": [] }))
    );
}

#[test]
fn taking_a_piece_empties_the_log() {
    let mut log = GrainLog::new(0, (25, 1));

    log.restart();
    log.saw(0);
    log.keep(vec![measured_chunk(2)]);
    log.take_piece().expect("encodes");
    let second = log.take_piece().expect("encodes");

    assert_eq!(
        second.data,
        ArtefactData::Structured(json!({ "frame_rate": [25, 1], "scenes": [] }))
    );
}

#[test]
fn a_restart_starts_a_new_scene_at_its_own_first_frame() {
    let mut log = GrainLog::new(0, (25, 1));

    log.restart();
    log.saw(0);
    log.keep(vec![measured_chunk(2)]);
    log.restart();
    log.saw(30);
    log.keep(vec![measured_chunk(2)]);
    let artefact = log.take_piece().expect("encodes");
    let piece = ArtefactPiece {
        chunk: None,
        data: artefact.data,
    };
    let scenes = GrainPiece::from_piece(piece).expect("decodes").into_scenes();
    let first_frames: Vec<u64> = scenes.iter().map(|scene| scene.first_frame).collect();

    assert_eq!(first_frames, vec![0, 30]);
}

#[test]
fn a_piece_with_a_non_finite_statistic_is_refused() {
    let mut chunk = measured_chunk(24);
    chunk.autocov[0] = f64::NAN;
    let scenes = vec![SceneGrain {
        first_frame: 0,
        chunks: vec![chunk],
    }];
    let grain = GrainPiece::new((25, 1), &scenes);

    let error = grain.to_artefact().expect_err("refused");

    assert!(error.to_string().contains("not finite"), "{error}");
}
