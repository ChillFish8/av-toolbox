#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(docsrs, doc(auto_cfg))]

//! Running av-toolbox pipelines from Rust
//!
//! A [Session] reads a TOML pipeline once and opens it on a [Backend], then runs it as one [Stream]
//! after another. Frames go in through the pipeline's `channel/frames` sources and come back in order
//! from its `channel/frames` sinks.
//!
//! To move to another point in the video, cancel the stream, drop its senders and receivers, join or drop its
//! handle, then start the next stream.
//!
//! Kernel compilation needs a large stack. Set `RUST_MIN_STACK` to [CODEGEN_STACK_BYTES] before any thread
//! starts, or call [raise_codegen_stack_limit] then.
//!
//! To see a pipeline's diagnostics before any GPU opens, read it as a [PipelineSource] and call
//! [PipelineSource::check], then open it with [Session::new].
//!
//! ```no_run
//! use std::thread;
//!
//! use av_toolbox::{BackendOptions, FrameFormat, InputFrame, PipelineInput, Session, SessionError, Settings};
//!
//! fn denoise(format: FrameFormat, frames: Vec<InputFrame>) -> Result<(), SessionError> {
//!     let text = r#"
//!         [source.input]
//!         type = "channel/frames"
//!
//!         [filter.denoise]
//!         type = "av-denoise/nl4d"
//!         inputs = ["input"]
//!
//!         [sink.output]
//!         type = "channel/frames"
//!         inputs = ["denoise"]
//!     "#;
//!     let input = PipelineInput::Text {
//!         name: "denoise.toml".to_owned(),
//!         text: text.to_owned(),
//!     };
//!     let options = BackendOptions::default();
//!     let backend = av_toolbox::open_backend(&options)?;
//!     let channels = [("input".to_owned(), format)];
//!     let mut session = Session::load(input, &[], backend, &channels)?;
//!
//!     let mut stream = session.start_stream(Settings::default())?;
//!     let sender = stream.take_sender("input").expect("the pipeline reads input");
//!     let receiver = stream.take_receiver("output").expect("the pipeline writes output");
//!     let handle = stream.into_handle();
//!
//!     let feeder = thread::spawn(move || {
//!         for frame in frames {
//!             if sender.send(frame).is_err() {
//!                 break;
//!             }
//!         }
//!     });
//!     for finished in receiver {
//!         println!("frame {} is done", finished.index);
//!     }
//!
//!     feeder.join().expect("the feeder finishes");
//!     let report = handle.join()?;
//!     println!("{} frames in {:?}", report.frames, report.wall);
//!
//!     Ok(())
//! }
//! ```

mod backend;
mod error;
mod input;
mod plan;
#[doc(hidden)]
pub mod report;
mod session;
mod source;
#[doc(hidden)]
pub mod steps;

pub use av_toolbox_core::{Granularity, InputFrame, OutputFrame, StageKind};
pub use av_toolbox_gpu::{
    Accelerator,
    Backend,
    CODEGEN_STACK_BYTES,
    ColourInfo,
    Device,
    FieldOrder,
    FrameFormat,
    FrameLayout,
    FrameRate,
    SampleFormat,
    Subsampling,
    Tuning,
    raise_codegen_stack_limit,
};
pub use av_toolbox_graph::{Diagnostic, DiagnosticCode, NodeId, NodeRole, Severity, StageSpec};
pub use av_toolbox_pipeline::StageRef;
pub use av_toolbox_runtime::{Cancel, Progress, RunError, RunReport, Settings};
pub use codespan_reporting::term::termcolor;

pub use self::backend::{BackendOptions, open_backend};
pub use self::error::SessionError;
pub use self::input::PipelineInput;
pub use self::plan::PlannedNode;
pub use self::session::{Session, Stream, StreamHandle};
pub use self::source::PipelineSource;
