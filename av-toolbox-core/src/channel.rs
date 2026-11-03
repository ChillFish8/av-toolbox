use std::collections::HashMap;
use std::process;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, SyncSender};
use std::sync::{Arc, LazyLock, Mutex, MutexGuard, PoisonError};
use std::time::Duration;

use av_toolbox_gpu::FrameFormat;

use crate::StageError;

/// How many frames each channel holds before the sending side waits.
pub const CHANNEL_DEPTH: usize = 2;

/// The longest a channel stage waits on the caller before it checks whether its run has stopped.
pub const CHANNEL_POLL: Duration = Duration::from_millis(20);

static PROCESS_HUB: LazyLock<ChannelHub> = LazyLock::new(ChannelHub::new);
static NEXT_TOKEN: AtomicU64 = AtomicU64::new(0);

/// A frame handed to the pipeline by the caller.
///
/// It holds one byte buffer per plane, in plane order, with tightly packed rows and native-endian samples.
#[derive(Debug, Clone)]
pub struct InputFrame {
    planes: Vec<Vec<u8>>,
}

impl InputFrame {
    pub fn new(planes: Vec<Vec<u8>>) -> Self {
        Self { planes }
    }

    pub fn planes(&self) -> &[Vec<u8>] {
        &self.planes
    }
}

/// A finished frame handed back to the caller.
///
/// The planes are laid out the same way as in [InputFrame].
#[derive(Debug, Clone)]
pub struct OutputFrame {
    pub index: u64,
    pub format: FrameFormat,
    pub planes: Vec<Vec<u8>>,
}

/// Names one session, so its channels never mix with another session's in the same process.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct ChannelToken(String);

impl ChannelToken {
    pub fn new() -> Self {
        let count = NEXT_TOKEN.fetch_add(1, Ordering::Relaxed);
        let process_id = process::id();
        Self(format!("{process_id}-{count}"))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for ChannelToken {
    fn default() -> Self {
        Self::new()
    }
}

/// Tells a channel stage that its run has stopped, so it stops waiting on the caller.
#[derive(Debug, Clone, Default)]
pub struct ChannelStop(Arc<AtomicBool>);

impl ChannelStop {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn stop(&self) {
        self.0.store(true, Ordering::SeqCst);
    }

    pub fn is_stopped(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }
}

/// The receiving end of a caller's frames, with the format they have.
#[derive(Debug)]
pub struct BoundSource {
    pub format: FrameFormat,
    pub frames: Receiver<InputFrame>,
    pub stop: ChannelStop,
}

/// The sending end of the frames handed back to a caller.
#[derive(Debug)]
pub struct BoundSink {
    pub frames: SyncSender<OutputFrame>,
    pub stop: ChannelStop,
}

#[derive(Debug)]
enum Endpoint {
    Source(BoundSource),
    Sink(BoundSink),
}

type EndpointKey = (String, String);

/// Holds the channel ends a caller binds until the matching stages take them.
///
/// Each end is keyed by a session token and a stage name, and can be taken once.
#[derive(Debug, Default)]
pub struct ChannelHub {
    endpoints: Mutex<HashMap<EndpointKey, Endpoint>>,
}

impl ChannelHub {
    /// The hub shared by every stage in this process.
    pub fn process() -> &'static ChannelHub {
        &PROCESS_HUB
    }

    fn new() -> Self {
        Self::default()
    }

    pub fn bind_source(
        &self,
        token: &ChannelToken,
        stage: &str,
        format: FrameFormat,
        frames: Receiver<InputFrame>,
        stop: ChannelStop,
    ) {
        let source = BoundSource { format, frames, stop };
        self.bind(token, stage, Endpoint::Source(source));
    }

    pub fn bind_sink(
        &self,
        token: &ChannelToken,
        stage: &str,
        frames: SyncSender<OutputFrame>,
        stop: ChannelStop,
    ) {
        let sink = BoundSink { frames, stop };
        self.bind(token, stage, Endpoint::Sink(sink));
    }

    pub fn take_source(&self, token: &str, stage: &str) -> Result<BoundSource, StageError> {
        let endpoint = self.take(token, stage);
        match endpoint {
            Some(Endpoint::Source(source)) => Ok(source),
            Some(other) => {
                self.put_back(token, stage, other);
                Err(not_bound(stage))
            },
            None => Err(not_bound(stage)),
        }
    }

    pub fn take_sink(&self, token: &str, stage: &str) -> Result<BoundSink, StageError> {
        let endpoint = self.take(token, stage);
        match endpoint {
            Some(Endpoint::Sink(sink)) => Ok(sink),
            Some(other) => {
                self.put_back(token, stage, other);
                Err(not_bound(stage))
            },
            None => Err(not_bound(stage)),
        }
    }

    /// Drops every end still bound under `token`.
    pub fn release(&self, token: &ChannelToken) {
        let mut endpoints = self.lock();
        endpoints.retain(|(bound_token, _), _| bound_token != token.as_str());
    }

    fn bind(&self, token: &ChannelToken, stage: &str, endpoint: Endpoint) {
        let key = (token.as_str().to_owned(), stage.to_owned());
        let mut endpoints = self.lock();
        endpoints.insert(key, endpoint);
    }

    fn take(&self, token: &str, stage: &str) -> Option<Endpoint> {
        let key = (token.to_owned(), stage.to_owned());
        let mut endpoints = self.lock();
        endpoints.remove(&key)
    }

    fn put_back(&self, token: &str, stage: &str, endpoint: Endpoint) {
        let key = (token.to_owned(), stage.to_owned());
        let mut endpoints = self.lock();
        endpoints.insert(key, endpoint);
    }

    fn lock(&self) -> MutexGuard<'_, HashMap<EndpointKey, Endpoint>> {
        self.endpoints.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

fn not_bound(stage: &str) -> StageError {
    StageError::msg(format!(
        "stage `{stage}` has no channel bound, it only runs through the av-toolbox library"
    ))
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc::{channel, sync_channel};

    use av_toolbox_gpu::{ColourInfo, FrameLayout, SampleFormat};

    use super::*;

    fn format() -> FrameFormat {
        FrameFormat {
            width: 2,
            height: 2,
            layout: FrameLayout::Grey,
            sample: SampleFormat::U8,
            colour: ColourInfo::default(),
            frame_rate: None,
            field_order: None,
        }
    }

    #[test]
    fn a_bound_source_is_taken_once() {
        let hub = ChannelHub::new();
        let token = ChannelToken::new();
        let (_sender, receiver) = channel();
        hub.bind_source(&token, "channel/frames", format(), receiver, ChannelStop::new());

        let first = hub.take_source(token.as_str(), "channel/frames");
        let second = hub.take_source(token.as_str(), "channel/frames");

        assert!(first.is_ok());
        assert!(second.is_err());
    }

    #[test]
    fn a_bound_sink_is_taken_once() {
        let hub = ChannelHub::new();
        let token = ChannelToken::new();
        let (sender, _receiver) = sync_channel(CHANNEL_DEPTH);
        hub.bind_sink(&token, "channel/frames", sender, ChannelStop::new());

        let first = hub.take_sink(token.as_str(), "channel/frames");
        let second = hub.take_sink(token.as_str(), "channel/frames");

        assert!(first.is_ok());
        assert!(second.is_err());
    }

    #[test]
    fn an_unbound_stage_names_the_library() {
        let hub = ChannelHub::new();

        let error = hub
            .take_source("missing", "channel/frames")
            .expect_err("nothing is bound");

        let message = error.to_string();
        assert!(message.contains("channel/frames"), "{message}");
        assert!(
            message.contains("only runs through the av-toolbox library"),
            "{message}"
        );
    }

    #[test]
    fn taking_the_wrong_kind_leaves_the_endpoint_bound() {
        let hub = ChannelHub::new();
        let token = ChannelToken::new();
        let (_sender, receiver) = channel();
        hub.bind_source(&token, "channel/frames", format(), receiver, ChannelStop::new());

        let wrong = hub.take_sink(token.as_str(), "channel/frames");
        let right = hub.take_source(token.as_str(), "channel/frames");

        assert!(wrong.is_err());
        assert!(right.is_ok());
    }

    #[test]
    fn two_tokens_never_share_endpoints() {
        let hub = ChannelHub::new();
        let first_token = ChannelToken::new();
        let second_token = ChannelToken::new();
        let (first_sender, first_receiver) = channel();
        let (second_sender, second_receiver) = channel();
        hub.bind_source(
            &first_token,
            "channel/frames",
            format(),
            first_receiver,
            ChannelStop::new(),
        );
        hub.bind_source(
            &second_token,
            "channel/frames",
            format(),
            second_receiver,
            ChannelStop::new(),
        );

        first_sender
            .send(InputFrame::new(vec![vec![1]]))
            .expect("receiver is bound");
        second_sender
            .send(InputFrame::new(vec![vec![2]]))
            .expect("receiver is bound");
        let first = hub
            .take_source(first_token.as_str(), "channel/frames")
            .expect("bound");
        let second = hub
            .take_source(second_token.as_str(), "channel/frames")
            .expect("bound");

        let first_frame = first.frames.recv().expect("a frame was sent");
        let second_frame = second.frames.recv().expect("a frame was sent");
        assert_eq!(first_frame.planes(), [vec![1]]);
        assert_eq!(second_frame.planes(), [vec![2]]);
    }

    #[test]
    fn releasing_a_token_drops_its_endpoints() {
        let hub = ChannelHub::new();
        let token = ChannelToken::new();
        let (sender, receiver) = channel();
        hub.bind_source(&token, "channel/frames", format(), receiver, ChannelStop::new());

        hub.release(&token);

        let sent = sender.send(InputFrame::new(vec![vec![1]]));
        assert!(sent.is_err());
        assert!(hub.take_source(token.as_str(), "channel/frames").is_err());
    }

    #[test]
    fn tokens_are_unique() {
        let first = ChannelToken::new();
        let second = ChannelToken::new();

        assert_ne!(first, second);
    }
}
