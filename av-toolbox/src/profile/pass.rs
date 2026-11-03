use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use av_toolbox::{PlannedNode, Session};
use av_toolbox_core::{Crossing, Granularity, StageKind};
use av_toolbox_graph::{NodeId, NodeRole, StageSpec};
use av_toolbox_runtime::{Cancel, RunReport, Settings};
use codespan_reporting::term::termcolor::WriteColor;

use crate::execute::CANCELLED;
use crate::load;

/// Stops whichever pass is running on the first Ctrl+C, and exits at once on the second.
pub(super) struct Interrupt {
    current: Mutex<Option<Arc<Cancel>>>,
    interrupted: AtomicBool,
}

impl Interrupt {
    pub(super) fn install() -> Arc<Self> {
        let interrupt = Arc::new(Self {
            current: Mutex::new(None),
            interrupted: AtomicBool::new(false),
        });

        let handler = interrupt.clone();
        let installed = ctrlc::set_handler(move || handler.stop());
        if let Err(error) = installed {
            tracing::warn!(%error, "Ctrl+C will stop the process without finishing the pass");
        }

        interrupt
    }

    fn stop(&self) {
        let again = self.interrupted.swap(true, Ordering::SeqCst);
        if again {
            let code = i32::from(CANCELLED);
            std::process::exit(code);
        }

        tracing::warn!("stopping the profile, press Ctrl+C again to exit at once");

        let current = self.current.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(cancel) = current.as_ref() {
            cancel.stop();
        }
    }

    /// Makes `cancel` the run the next Ctrl+C stops, stopping it at once if one already came.
    pub(super) fn watch(&self, cancel: Arc<Cancel>) {
        let mut current = self.current.lock().unwrap_or_else(PoisonError::into_inner);
        *current = Some(cancel.clone());

        if self.interrupted() {
            cancel.stop();
        }
    }

    pub(super) fn interrupted(&self) -> bool {
        self.interrupted.load(Ordering::SeqCst)
    }
}

/// What every pass shares.
pub(super) struct PassSetup<'a> {
    pub(super) session: &'a mut Session,
    pub(super) interrupt: &'a Interrupt,
}

/// A node of a pass's graph, named for the report.
#[derive(Debug, Clone)]
pub(super) struct NodeInfo {
    pub(super) id: NodeId,
    /// `<kind>.<name>`, empty for a transfer.
    pub(super) key: String,
    pub(super) label: String,
    /// The plan's granularity for a written stage, `None` for a transfer.
    pub(super) granularity: Option<Granularity>,
    /// How many workers the plan runs it with, inherited widths included.
    pub(super) planned: usize,
    /// Whether the model may give it a width of its own.
    pub(super) widenable: bool,
}

pub(super) struct Pass {
    pub(super) report: RunReport,
    pub(super) nodes: Vec<NodeInfo>,
}

pub(super) enum PassEnd {
    Finished(Pass),
    Failed,
    Interrupted,
}

/// Whether a pass prints the warnings compiling its pipeline raises.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub(super) enum Warnings {
    Show,
    Hide,
}

/// Runs `specs` once as a stream of the session, printing whatever went wrong.
pub(super) fn run_pass(
    setup: &mut PassSetup<'_>,
    specs: &[StageSpec],
    settings: Settings,
    warnings: Warnings,
    stderr: &mut dyn WriteColor,
) -> PassEnd {
    let session = &mut *setup.session;
    *session.specs_mut() = specs.to_vec();

    let started = session.start_stream(settings);
    if warnings == Warnings::Show {
        session.render_warnings(stderr);
    }

    let stream = match started {
        Ok(stream) => stream,
        Err(error) => {
            load::report(&error, stderr);

            return PassEnd::Failed;
        },
    };

    let handle = stream.into_handle();
    let cancel = handle.cancel_handle();
    setup.interrupt.watch(cancel);

    let planned = handle.plan();
    let nodes = node_infos(planned);
    let finished = handle.join();
    let report = match finished {
        Ok(report) => report,
        Err(error) => {
            if setup.interrupt.interrupted() {
                return PassEnd::Interrupted;
            }

            load::report(&error, stderr);

            return PassEnd::Failed;
        },
    };

    if report.cancelled {
        return PassEnd::Interrupted;
    }

    PassEnd::Finished(Pass { report, nodes })
}

/// Every node in dependency order, named for the report.
fn node_infos(planned: &[PlannedNode]) -> Vec<NodeInfo> {
    let mut infos = Vec::with_capacity(planned.len());
    for node in planned {
        let info = match &node.role {
            NodeRole::Stage | NodeRole::Provider(_) => {
                let sized_kind = matches!(node.kind, StageKind::Filter | StageKind::Mask);

                NodeInfo {
                    id: node.id,
                    key: node.path.clone(),
                    label: node.path.clone(),
                    granularity: Some(node.granularity),
                    planned: node.width,
                    widenable: sized_kind && node.granularity != Granularity::Stream,
                }
            },
            NodeRole::Crossing(crossing) => NodeInfo {
                id: node.id,
                key: String::new(),
                label: transfer_label(planned, node, *crossing),
                granularity: None,
                planned: node.width,
                widenable: false,
            },
        };

        infos.push(info);
    }

    infos
}

/// Names a transfer after its direction and the stage it feeds, such as `download -> sink.output`.
fn transfer_label(planned: &[PlannedNode], transfer: &PlannedNode, crossing: Crossing) -> String {
    let direction = match crossing {
        Crossing::Upload => "upload",
        Crossing::Unload => "download",
    };

    let Some(first) = transfer.consumers.first() else {
        return direction.to_string();
    };

    let fed = planned.iter().find(|node| node.id == *first);
    let Some(fed) = fed else {
        return direction.to_string();
    };

    format!("{direction} -> {}", fed.path)
}
