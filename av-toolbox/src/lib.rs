#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(docsrs, doc(auto_cfg))]

//! Running av-toolbox pipelines from Rust
//!
//! A [Session] reads a TOML pipeline once and opens it on a [Backend], then runs it as one [Stream]
//! after another. Frames go in through the pipeline's `channel/frames` sources and come back in order
//! from its `channel/frames` sinks.
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
#[doc(hidden)]
pub mod report;
mod session;
#[doc(hidden)]
pub mod steps;

pub use av_toolbox_core::{InputFrame, OutputFrame};
pub use av_toolbox_gpu::{Accelerator, Backend, Device, FrameFormat, Tuning};
pub use av_toolbox_graph::{Diagnostic, NodeId, StageSpec};
pub use av_toolbox_pipeline::StageRef;
pub use av_toolbox_runtime::{Cancel, Progress, RunError, RunReport, Settings};

pub use self::backend::{BackendOptions, open_backend};
pub use self::error::SessionError;
pub use self::input::PipelineInput;
pub use self::session::{Session, Stream, StreamHandle};
