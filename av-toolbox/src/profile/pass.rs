use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use av_toolbox_core::{Crossing, Granularity, PipelineRegistry, StageKind};
use av_toolbox_gpu::Backend;
use av_toolbox_graph::{CompiledGraph, NodeId, NodeRole, StageSpec};
use av_toolbox_runtime::{Cancel, RunReport, Settings};
use codespan_reporting::term::termcolor::WriteColor;

use crate::execute::CANCELLED;
use crate::load;
use crate::report::{self, PipelineFiles};

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
    pub(super) registry: &'a PipelineRegistry,
    pub(super) backend: &'a Backend,
    pub(super) files: &'a PipelineFiles,
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

/// Builds, compiles and runs `specs` once, printing whatever went wrong.
pub(super) fn run_pass(
    setup: &PassSetup<'_>,
    specs: &[StageSpec],
    settings: Settings,
    warnings: Warnings,
    stderr: &mut dyn WriteColor,
) -> PassEnd {
    let Some(pipeline) = load::build(setup.registry, specs, setup.files, stderr) else {
        return PassEnd::Failed;
    };

    let compiled = match warnings {
        Warnings::Show => load::compile(pipeline, setup.backend, setup.files, stderr),
        Warnings::Hide => load::compile_quietly(pipeline, setup.backend, setup.files, stderr),
    };
    let Some(compiled) = compiled else {
        return PassEnd::Failed;
    };

    let built = compiled.engine(settings);
    let engine = match built {
        Ok(engine) => engine,
        Err(error) => {
            let found = error.into_diagnostics().into_inner();
            report::diagnostics(stderr, setup.files, &found);

            return PassEnd::Failed;
        },
    };

    let cancel = engine.cancel_handle();
    setup.interrupt.watch(cancel);

    let nodes = node_infos(compiled.graph());
    let finished = engine.run();
    let report = match finished {
        Ok(report) => report,
        Err(error) => {
            if setup.interrupt.interrupted() {
                return PassEnd::Interrupted;
            }

            let stage = error.node().and_then(|node| compiled.stage_of(node));
            report::run_error(stderr, setup.files, &error, stage);

            return PassEnd::Failed;
        },
    };

    if report.cancelled {
        return PassEnd::Interrupted;
    }

    PassEnd::Finished(Pass { report, nodes })
}

/// Every node in topological order, named for the report.
fn node_infos(graph: &CompiledGraph) -> Vec<NodeInfo> {
    let plan = graph
        .execution_plan()
        .expect("a compiled pipeline is always planned");

    let mut infos = Vec::with_capacity(graph.len());
    for id in graph.order() {
        let node = graph.node(*id);
        let planned = plan.node(*id).width();
        let info = match node.role() {
            NodeRole::Stage | NodeRole::Provider(_) => {
                let granularity = plan.node(*id).granularity();
                let sized_kind = matches!(
                    node.kind(),
                    StageKind::Filter | StageKind::Mask | StageKind::Transform
                );

                NodeInfo {
                    id: *id,
                    key: node.path(),
                    label: node.path(),
                    granularity: Some(granularity),
                    planned,
                    widenable: sized_kind && granularity != Granularity::Stream,
                }
            },
            NodeRole::Crossing(crossing) => NodeInfo {
                id: *id,
                key: String::new(),
                label: transfer_label(graph, *id, *crossing),
                granularity: None,
                planned,
                widenable: false,
            },
        };

        infos.push(info);
    }

    infos
}

/// Names a transfer after its direction and the stage it feeds, such as `download -> sink.output`.
fn transfer_label(graph: &CompiledGraph, id: NodeId, crossing: Crossing) -> String {
    let direction = match crossing {
        Crossing::Upload => "upload",
        Crossing::Unload => "download",
    };

    let consumers = graph.node(id).consumers();
    let Some(first) = consumers.first() else {
        return direction.to_string();
    };

    let fed = graph.node(*first).path();
    format!("{direction} -> {fed}")
}
