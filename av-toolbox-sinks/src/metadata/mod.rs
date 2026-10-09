mod kinds;
mod line;

use std::io;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use av_toolbox_core::{
    AnyDemand,
    BoundInputs,
    InputDemand,
    InputSignature,
    KeyFilter,
    MetaInputs,
    MetaSink,
    MetaSinkDefinition,
    Pattern,
    Provision,
    StageDefinition,
    StageError,
    StageInstance,
};
use av_toolbox_gpu::{
    BuildContext,
    Capabilities,
    FormatSet,
    FrameContext,
    MetaRecord,
    Offered,
    ResolvedFormats,
    Runtime,
    StreamFormat,
};
use av_toolbox_providers::{SceneRequest, SceneScores, Scenes, ScoreRequest};
use schemars::JsonSchema;
use serde::Deserialize;

use self::kinds::Kinds;
use self::line::{write_line, write_record_line};
use crate::shared::{INPUTS, open_target};

const STAGE: &str = "metadata/ndjson";

/// Arguments for `metadata/ndjson`, which takes exactly one of `fd` or `path`.
#[derive(Debug, Deserialize, JsonSchema)]
pub(crate) struct NdjsonArgs {
    /// An inherited file descriptor, Unix only.
    pub(crate) fd: Option<u32>,
    /// A FIFO, or a file to create or truncate.
    pub(crate) path: Option<PathBuf>,
    /// Keys to keep, where `*` matches any run of characters. Every key is kept when unset.
    pub(crate) include: Option<Vec<Pattern>>,
    /// Keys to drop, which wins over `include`.
    pub(crate) exclude: Option<Vec<Pattern>>,
    /// Scene breaks to write each frame's chunk in, as its `chunk`, along with its `scene_score`.
    #[serde(default)]
    pub(crate) scenes: Option<SceneRequest>,
    /// The kinds of line to write, such as `frame` or `chunk`. Every kind is written when unset.
    #[serde(default)]
    pub(crate) kinds: Option<Kinds>,
}

pub(crate) struct Ndjson;

impl StageDefinition for Ndjson {
    type Args = NdjsonArgs;

    fn package() -> &'static str {
        "metadata"
    }

    fn identifier() -> &'static str {
        "ndjson"
    }

    /// Scene breaks and scene cut scores, both on the input, when `scenes` is set.
    fn demands(args: &Self::Args) -> Vec<InputDemand> {
        let Some(request) = args.scenes.clone() else {
            return Vec::new();
        };

        let score_request = ScoreRequest {
            speed: request.speed,
            after: request.after.clone(),
        };
        let scenes = AnyDemand::new::<Scenes>(request);
        let scores = AnyDemand::new::<SceneScores>(score_request);

        vec![
            InputDemand {
                input: "input",
                demand: scenes,
            },
            InputDemand {
                input: "input",
                demand: scores,
            },
        ]
    }

    fn inputs() -> &'static InputSignature {
        INPUTS
    }
}

impl MetaSinkDefinition for Ndjson {
    fn accepts_input(
        _capabilities: &Capabilities,
        args: &Self::Args,
        _index: usize,
        offered: StreamFormat,
        _inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        if let Some(scenes) = &args.scenes {
            scenes.validate()?;
        }

        accepts(offered)
    }

    /// Opens the descriptor or path.
    ///
    /// Opening a FIFO waits until something opens it to read.
    fn build<R: Runtime>(
        _context: &BuildContext<R>,
        args: Self::Args,
        inputs: BoundInputs,
        _formats: &ResolvedFormats,
    ) -> Result<Box<dyn MetaSink>, StageError> {
        let path = args.path.as_deref();
        let file = open_target(STAGE, args.fd, path)?;
        let filter = KeyFilter::new(args.include, args.exclude);
        let scenes = SceneProvisions::of(&inputs);
        let sink = NdjsonSink::new(file, filter, scenes).with_kinds(args.kinds);

        Ok(Box::new(sink))
    }
}

/// The samples `metadata/ndjson` will take for `offered`, which has to be a frame stream.
fn accepts(offered: StreamFormat) -> Result<FormatSet, StageError> {
    let StreamFormat::Frames(_) = offered else {
        return Err(StageError::msg("metadata/ndjson logs frames, not a mask"));
    };

    Ok(FormatSet::of([offered.sample()]))
}

/// The handles a sink given `scenes` reads each frame's chunk and scene cut score through.
#[derive(Debug, Copy, Clone)]
pub(crate) struct SceneProvisions {
    pub(crate) chunks: Provision<Scenes>,
    pub(crate) scores: Provision<SceneScores>,
}

impl SceneProvisions {
    /// The handles for what was provided on `input`, when the sink demanded both.
    fn of(inputs: &BoundInputs) -> Option<Self> {
        let chunks = inputs.provision::<Scenes>("input").ok()?;
        let scores = inputs.provision::<SceneScores>("input").ok()?;

        Some(Self { chunks, scores })
    }
}

/// A sink writing each frame's metadata as one line of JSON.
///
/// When it demands scenes, each line also carries the frame's scene chunk and scene cut score.
pub(crate) struct NdjsonSink<W: Write> {
    output: BufWriter<W>,
    filter: KeyFilter,
    scenes: Option<SceneProvisions>,
    kinds: Option<Kinds>,
    last_index: Option<u64>,
    last_chunk: Option<u64>,
}

impl<W: Write> NdjsonSink<W> {
    pub(crate) fn new(output: W, filter: KeyFilter, scenes: Option<SceneProvisions>) -> Self {
        Self {
            output: BufWriter::new(output),
            filter,
            scenes,
            kinds: None,
            last_index: None,
            last_chunk: None,
        }
    }

    /// Writes only the lines of the given kinds, or every kind when `None`.
    pub(crate) fn with_kinds(mut self, kinds: Option<Kinds>) -> Self {
        self.kinds = kinds;

        self
    }

    fn keeps(&self, kind: &str) -> bool {
        self.kinds.as_ref().is_none_or(|kinds| kinds.keeps(kind))
    }
}

impl<W: Write + Send + 'static> StageInstance for NdjsonSink<W> {}

impl<W: Write + Send + 'static> MetaSink for NdjsonSink<W> {
    fn consume(&mut self, _context: &mut FrameContext, mut inputs: MetaInputs) -> Result<(), StageError> {
        let Some(item) = inputs.take(0) else {
            return Ok(());
        };

        let meta = item.meta();
        let chunk = self
            .scenes
            .and_then(|scenes| scenes.chunks.get(meta))
            .map(|scene| scene.0);
        let score = self
            .scenes
            .and_then(|scenes| scenes.scores.get(meta))
            .and_then(|cut| cut.score);
        self.last_index = Some(meta.index);
        self.last_chunk = chunk;
        if !self.keeps("frame") {
            return Ok(());
        }

        let written = write_line(&mut self.output, meta, chunk, score, &self.filter);

        written.map_err(|error| {
            let position = format!("frame {}", meta.index);

            write_error(&position, &error)
        })
    }

    fn consume_record(&mut self, _context: &mut FrameContext, record: &MetaRecord) -> Result<(), StageError> {
        if !self.keeps(record.kind()) {
            return Ok(());
        }

        let chunk = self.scenes.map(|_| self.last_chunk);
        let written = write_record_line(&mut self.output, record, self.last_index, chunk, &self.filter);

        written.map_err(|error| {
            let position = format!("the {} record", record.kind());

            write_error(&position, &error)
        })
    }

    fn end_of_stream(&mut self, _context: &mut FrameContext) -> Result<(), StageError> {
        let flushed = self.output.flush();

        flushed.map_err(|error| flush_error(&error))
    }
}

fn write_error(position: &str, error: &io::Error) -> StageError {
    let message = match error.kind() {
        io::ErrorKind::BrokenPipe => {
            format!("the reader closed the metadata stream before {position} was written")
        },
        _ => format!("writing {position} of the metadata stream failed: {error}"),
    };

    StageError::msg(message)
}

fn flush_error(error: &io::Error) -> StageError {
    let message = match error.kind() {
        io::ErrorKind::BrokenPipe => {
            "the reader closed the metadata stream before it was flushed".to_string()
        },
        _ => format!("flushing the metadata stream failed: {error}"),
    };

    StageError::msg(message)
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::{fs, io};

    use av_toolbox_core::{
        DemandKind,
        InputArgument,
        StageHandle,
        StreamType,
        bind_inputs,
        end_meta_sink,
        record_meta_sink,
        run_meta_sink,
    };
    use av_toolbox_gpu::{
        AnyItem,
        HostStreamItem,
        LayerId,
        MaskFormat,
        MetaRecord,
        MetaValue,
        SampleFormat,
        default_accelerators,
        testing_context,
    };
    use av_toolbox_providers::{SceneChunk, SceneCut, SceneRequest, SceneScore, SceneSpeed};

    use super::*;
    use crate::shared::testing::{SharedBuffer, filled_frame, yuv420};

    fn frame_item(index: u64, tagged: bool) -> AnyItem {
        let format = yuv420(4, 2, SampleFormat::U8, None);
        let mut frame = filled_frame(&format, index, 0);

        if tagged {
            frame.meta_mut().attach("seen", MetaValue::Bool(true));
            frame.meta_mut().attachments.stamp("filter.tag");
        }

        AnyItem::Host(HostStreamItem::from(frame))
    }

    fn run_frames<W: Write + Send + 'static>(
        sink: &mut NdjsonSink<W>,
        context: &mut FrameContext,
        items: Vec<AnyItem>,
    ) {
        for item in items {
            run_meta_sink(sink, context, vec![Some(item)]).expect("consumes");
        }
    }

    fn run<W: Write + Send + 'static>(
        sink: &mut NdjsonSink<W>,
        items: Vec<AnyItem>,
    ) -> Result<(), StageError> {
        let mut context = testing_context();

        for item in items {
            run_meta_sink(sink, &mut context, vec![Some(item)])?;
        }

        end_meta_sink(sink, &mut context)
    }

    #[test]
    fn each_frame_is_one_line_in_order() {
        let buffer = SharedBuffer::default();
        let mut sink = NdjsonSink::new(buffer.clone(), KeyFilter::default(), None);

        run(&mut sink, vec![frame_item(0, true), frame_item(1, false)]).expect("runs");

        let text = String::from_utf8(buffer.contents()).expect("utf-8");
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 2);
        assert!(lines[0].starts_with("{\"kind\":\"frame\",\"index\":0,"));
        assert!(lines[0].ends_with("\"filter.tag\":{\"seen\":true}}"));
        assert!(lines[1].starts_with("{\"kind\":\"frame\",\"index\":1,"));
    }

    #[test]
    fn an_empty_stream_writes_nothing() {
        let buffer = SharedBuffer::default();
        let mut sink = NdjsonSink::new(buffer.clone(), KeyFilter::default(), None);

        run(&mut sink, Vec::new()).expect("runs");

        assert!(buffer.contents().is_empty());
    }

    #[test]
    fn ending_twice_is_harmless() {
        let buffer = SharedBuffer::default();
        let mut sink = NdjsonSink::new(buffer.clone(), KeyFilter::default(), None);
        let mut context = testing_context();

        run(&mut sink, vec![frame_item(0, false)]).expect("runs");
        end_meta_sink(&mut sink, &mut context).expect("ends again");

        let text = String::from_utf8(buffer.contents()).expect("utf-8");
        assert_eq!(text.lines().count(), 1);
    }

    #[test]
    fn a_closed_reader_gives_the_broken_pipe_error() {
        struct ClosedReader;

        impl Write for ClosedReader {
            fn write(&mut self, _buffer: &[u8]) -> io::Result<usize> {
                Err(io::ErrorKind::BrokenPipe.into())
            }

            fn flush(&mut self) -> io::Result<()> {
                Err(io::ErrorKind::BrokenPipe.into())
            }
        }

        let mut sink = NdjsonSink::new(ClosedReader, KeyFilter::default(), None);

        let error = run(&mut sink, vec![frame_item(0, false)]).expect_err("the reader is gone");

        assert!(error.to_string().contains("reader closed"), "{error}");
    }

    #[test]
    fn a_mask_is_refused() {
        let offered = StreamFormat::Mask(MaskFormat {
            width: 4,
            height: 2,
            sample: SampleFormat::U8,
            chroma: None,
        });

        let result = accepts(offered);

        assert!(result.is_err());
    }

    #[test]
    fn both_fd_and_path_is_refused() {
        let result = open_target("metadata/ndjson", Some(3), Some(Path::new("a.ndjson")));

        let error = result.expect_err("both given");
        assert!(error.to_string().contains("metadata/ndjson"), "{error}");
    }

    #[test]
    fn a_path_is_written_through() {
        let folder = tempfile::tempdir().expect("a temp dir");
        let path = folder.path().join("out.ndjson");
        let file = open_target("metadata/ndjson", None, Some(&path)).expect("opens");
        let mut sink = NdjsonSink::new(file, KeyFilter::default(), None);

        run(&mut sink, vec![frame_item(0, false)]).expect("runs");
        drop(sink);

        let text = fs::read_to_string(&path).expect("reads");
        assert_eq!(text.lines().count(), 1);
    }

    fn chunked_item(index: u64, chunk: u64) -> AnyItem {
        let format = yuv420(4, 2, SampleFormat::U8, None);
        let mut frame = filled_frame(&format, index, 0);
        frame.meta_mut().provided.set(LayerId(0), SceneChunk(chunk));

        AnyItem::Host(HostStreamItem::from(frame))
    }

    /// Handles reading chunks from layer 0 and scores from layer 1.
    fn scenes_provision() -> SceneProvisions {
        let chunks_kind = DemandKind::of::<Scenes>();
        let scores_kind = DemandKind::of::<SceneScores>();
        let argument = InputArgument::positional(StageHandle::new(0), StreamType::Frames)
            .with_demand(chunks_kind, LayerId(0))
            .with_demand(scores_kind, LayerId(1));
        let bound = bind_inputs(Ndjson::inputs(), &[argument]).expect("binds");

        SceneProvisions::of(&bound).expect("demanded")
    }

    fn scored_item(index: u64, chunk: u64, score: f32) -> AnyItem {
        let format = yuv420(4, 2, SampleFormat::U8, None);
        let mut frame = filled_frame(&format, index, 0);
        let cut = SceneCut {
            score: Some(SceneScore::new(score)),
            starts_scene: false,
        };
        let provided = &mut frame.meta_mut().provided;
        provided.set(LayerId(0), SceneChunk(chunk));
        provided.set(LayerId(1), cut);

        AnyItem::Host(HostStreamItem::from(frame))
    }

    fn written_lines(buffer: &SharedBuffer) -> Vec<serde_json::Value> {
        let text = String::from_utf8(buffer.contents()).expect("utf-8");

        text.lines()
            .map(|line| serde_json::from_str(line).expect("valid json"))
            .collect()
    }

    #[test]
    fn the_ndjson_sink_writes_the_layer_it_asks_for() {
        let buffer = SharedBuffer::default();
        let mut sink = NdjsonSink::new(buffer.clone(), KeyFilter::default(), Some(scenes_provision()));
        let items = vec![chunked_item(0, 0), chunked_item(1, 0), chunked_item(2, 1)];

        run(&mut sink, items).expect("runs");

        let lines = written_lines(&buffer);
        let chunks: Vec<&serde_json::Value> = lines.iter().map(|line| &line["chunk"]).collect();
        assert_eq!(chunks, [0, 0, 1]);
    }

    #[test]
    fn the_ndjson_sink_writes_the_score_it_asks_for() {
        let buffer = SharedBuffer::default();
        let mut sink = NdjsonSink::new(buffer.clone(), KeyFilter::default(), Some(scenes_provision()));
        let items = vec![scored_item(0, 0, 2.5), chunked_item(1, 0)];

        run(&mut sink, items).expect("runs");

        let lines = written_lines(&buffer);
        let scores: Vec<&serde_json::Value> = lines.iter().map(|line| &line["scene_score"]).collect();
        assert_eq!(scores, [&serde_json::json!(2.5), &serde_json::Value::Null]);
    }

    #[test]
    fn the_ndjson_sink_writes_no_score_without_scenes() {
        let buffer = SharedBuffer::default();
        let mut sink = NdjsonSink::new(buffer.clone(), KeyFilter::default(), None);

        run(&mut sink, vec![scored_item(0, 0, 2.5)]).expect("runs");

        let lines = written_lines(&buffer);
        assert!(lines[0]["scene_score"].is_null(), "{lines:?}");
    }

    #[test]
    fn the_ndjson_sink_writes_no_chunk_without_scenes() {
        let buffer = SharedBuffer::default();
        let mut sink = NdjsonSink::new(buffer.clone(), KeyFilter::default(), None);

        run(&mut sink, vec![chunked_item(0, 3)]).expect("runs");

        let lines = written_lines(&buffer);
        assert_eq!(lines.len(), 1);
        assert!(lines[0].get("chunk").is_none(), "{lines:?}");
    }

    #[test]
    fn scenes_demand_scene_breaks_and_scores_on_the_input() {
        let value = serde_json::json!({
            "path": "out.ndjson",
            "scenes": { "breaks": { "mode": "natural" }, "speed": "fast", "after": "denoise" },
        });
        let args: NdjsonArgs = serde_json::from_value(value).expect("parses");

        let demands = Ndjson::demands(&args);

        assert_eq!(demands.len(), 2);
        assert!(demands.iter().all(|demand| demand.input == "input"));

        let scenes = demands[0].demand.request::<Scenes>();
        let expected_scenes = SceneRequest {
            speed: SceneSpeed::Fast,
            after: Some("denoise".to_string()),
            ..SceneRequest::natural()
        };
        assert_eq!(scenes, Some(expected_scenes));

        let scores = demands[1].demand.request::<SceneScores>();
        let expected_scores = ScoreRequest {
            speed: SceneSpeed::Fast,
            after: Some("denoise".to_string()),
        };
        assert_eq!(scores, Some(expected_scores));
    }

    #[test]
    fn a_scenes_seconds_limit_that_is_not_finite_is_refused() {
        let args = NdjsonArgs {
            fd: None,
            path: Some(PathBuf::from("out.ndjson")),
            include: None,
            exclude: None,
            scenes: Some(SceneRequest::limited_to_seconds(f64::INFINITY)),
            kinds: None,
        };
        let format = yuv420(4, 2, SampleFormat::U8, None);
        let offered = StreamFormat::Frames(format);
        let accelerator = default_accelerators()[0];
        let capabilities = Capabilities::builder(accelerator)
            .plane_size(16, 16)
            .max_units_per_cube(1024)
            .max_cube_count(65535, 65535, 65535)
            .build();

        let accepted = Ndjson::accepts_input(&capabilities, &args, 0, offered, &[]);

        let error = accepted.expect_err("refused").to_string();
        assert!(error.contains("finite"), "{error}");
    }

    #[test]
    fn without_scenes_nothing_is_demanded() {
        let value = serde_json::json!({ "path": "out.ndjson" });
        let args: NdjsonArgs = serde_json::from_value(value).expect("parses");

        let demands = Ndjson::demands(&args);

        assert!(demands.is_empty());
    }

    #[test]
    fn the_arguments_deserialize_include_and_exclude() {
        let value = serde_json::json!({
            "path": "out.ndjson",
            "include": ["index", "filter.*"],
            "exclude": ["filter.debug_*"],
        });

        let args: NdjsonArgs = serde_json::from_value(value).expect("parses");

        assert_eq!(args.include.map(|patterns| patterns.len()), Some(2));
        assert_eq!(args.exclude.map(|patterns| patterns.len()), Some(1));
    }

    fn a_record() -> MetaRecord {
        let mut record = MetaRecord::new("chunk");
        record.attachments_mut().attach("frames", MetaValue::Int(24));
        record.attachments_mut().stamp("filter.stats");

        record
    }

    #[test]
    fn a_record_line_names_the_last_frame_received() {
        let buffer = SharedBuffer::default();
        let mut sink = NdjsonSink::new(buffer.clone(), KeyFilter::default(), None);
        let mut context = testing_context();

        run_frames(
            &mut sink,
            &mut context,
            vec![frame_item(0, false), frame_item(1, false)],
        );
        record_meta_sink(&mut sink, &mut context, &a_record()).expect("records");
        end_meta_sink(&mut sink, &mut context).expect("ends");

        let lines = written_lines(&buffer);
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[2]["kind"], "chunk");
        assert_eq!(lines[2]["after"], 1);
        assert!(lines[2].get("chunk").is_none(), "{lines:?}");
        assert_eq!(lines[2]["filter.stats"]["frames"], 24);
    }

    #[test]
    fn a_record_line_carries_the_last_chunk_with_scenes() {
        let buffer = SharedBuffer::default();
        let mut sink = NdjsonSink::new(buffer.clone(), KeyFilter::default(), Some(scenes_provision()));
        let mut context = testing_context();

        run_frames(
            &mut sink,
            &mut context,
            vec![chunked_item(0, 0), chunked_item(1, 2)],
        );
        record_meta_sink(&mut sink, &mut context, &a_record()).expect("records");
        end_meta_sink(&mut sink, &mut context).expect("ends");

        let lines = written_lines(&buffer);
        assert_eq!(lines[2]["chunk"], 2);
    }

    #[test]
    fn kinds_keeps_only_the_named_kinds() {
        let buffer = SharedBuffer::default();
        let kinds: Kinds = serde_json::from_value(serde_json::json!(["chunk"])).expect("parses");
        let mut sink = NdjsonSink::new(buffer.clone(), KeyFilter::default(), None).with_kinds(Some(kinds));
        let mut context = testing_context();

        run_frames(
            &mut sink,
            &mut context,
            vec![frame_item(0, false), frame_item(1, false)],
        );
        record_meta_sink(&mut sink, &mut context, &a_record()).expect("records");
        end_meta_sink(&mut sink, &mut context).expect("ends");

        let lines = written_lines(&buffer);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0]["kind"], "chunk");
        assert_eq!(lines[0]["after"], 1);
    }

    #[test]
    fn an_empty_kinds_list_fails_to_deserialize() {
        let value = serde_json::json!({ "path": "out.ndjson", "kinds": [] });

        let error = serde_json::from_value::<NdjsonArgs>(value).expect_err("empty");

        assert!(error.to_string().contains("expected a list such as"), "{error}");
    }

    #[test]
    fn an_empty_kind_name_fails_to_deserialize() {
        let value = serde_json::json!({ "path": "out.ndjson", "kinds": [""] });

        let error = serde_json::from_value::<NdjsonArgs>(value).expect_err("empty");

        assert!(error.to_string().contains("expected a name such as"), "{error}");
    }

    #[test]
    fn a_record_before_any_frame_has_a_null_chunk_with_scenes() {
        let buffer = SharedBuffer::default();
        let mut sink = NdjsonSink::new(buffer.clone(), KeyFilter::default(), Some(scenes_provision()));
        let mut context = testing_context();

        record_meta_sink(&mut sink, &mut context, &a_record()).expect("records");
        end_meta_sink(&mut sink, &mut context).expect("ends");

        let lines = written_lines(&buffer);
        assert!(lines[0]["after"].is_null(), "{lines:?}");
        assert!(lines[0]["chunk"].is_null(), "{lines:?}");
        assert!(lines[0].get("chunk").is_some(), "{lines:?}");
    }
}
