use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::fmt;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, LazyLock, Mutex, PoisonError};

use cubecl::ir::DeviceProperties;
use cubecl::prelude::{ComputeClient, Runtime};
use cubecl::tune::{
    AutotuneKey,
    AutotuneOutput,
    InputGenerator,
    LocalTuner,
    Tunable,
    TunableSet,
    TuneInputs,
};
use serde::{Deserialize, Serialize};

use super::direct::direct_variants;
use super::winograd::winograd_variants;
use super::{BodyConvolution, Candidate, DirectVariant, launch_candidate};
use crate::cnn::body::LayerRun;
use crate::cnn::convolution::offers_cmma_16;
use crate::cnn::{BodyPath, CnnError, ConvLayer, Precision};

/// The samples at which a region leaves the smallest bucket, 64k.
const MEDIUM_SAMPLES: u64 = 1 << 16;

/// The most samples a region in the middle bucket holds, 1M.
const LARGE_SAMPLES: u64 = 1 << 20;

/// The label that keeps the tunings of the candidates a device offers apart from any other list.
const OFFERED_LABEL: &str = "offered";

/// The label that keeps the tunings of the candidates a device offers a Winograd layer apart from any other
/// list.
const OFFERED_WINOGRAD_LABEL: &str = "offered-winograd";

/// Chooses each body convolution's candidate, per device and tune key.
static TUNER: LocalTuner<TuneKey, String> = LocalTuner::new("av-toolbox-cnn-body-convolution");

/// The plan each kernel choice runs by, per runtime, device limits, Winograd offer and width.
static PLANS: LazyLock<Mutex<HashMap<PlanKey, SharedPlan>>> = LazyLock::new(Mutex::default);

/// A [Plan] for any runtime.
type SharedPlan = Arc<dyn Any + Send + Sync>;

/// How many samples a body convolution covers, in buckets coarse enough that nearby frame sizes share one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SizeBucket {
    /// Under 64k samples.
    Small,
    /// From 64k to 1M samples.
    Medium,
    /// Over 1M samples.
    Large,
}

impl SizeBucket {
    pub fn of(samples: u64) -> Self {
        if samples < MEDIUM_SAMPLES {
            return Self::Small;
        }

        if samples <= LARGE_SAMPLES {
            return Self::Medium;
        }

        Self::Large
    }
}

/// What a body convolution applies as it writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Epilogue {
    /// SiLU, on the first two convolutions of a block.
    Silu,
    /// The block's input added, on the last convolution of a block.
    Residual,
    /// The first layer's maps added, on the last body convolution.
    Join,
    /// ReLU, on the first convolution of a ResBlock.
    Relu,
}

/// What a body convolution's tuning is kept under.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TuneKey {
    /// The width of the maps.
    pub features: u32,
    pub size: SizeBucket,
    /// The device's widest subgroup.
    pub plane_width: u32,
    pub epilogue: Epilogue,
}

impl TuneKey {
    /// The key of `convolution` with `epilogue`, on a device whose widest subgroup is `plane_width`.
    pub fn of(convolution: &BodyConvolution<'_>, epilogue: Epilogue, plane_width: u32) -> Self {
        let samples = u64::from(convolution.width) * u64::from(convolution.height);
        let size = SizeBucket::of(samples);

        Self {
            features: convolution.features,
            size,
            plane_width,
            epilogue,
        }
    }
}

impl fmt::Display for TuneKey {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "width {} {:?} plane {} {:?}",
            self.features, self.size, self.plane_width, self.epilogue
        )
    }
}

impl AutotuneKey for TuneKey {}

impl AutotuneOutput for LayerRun {}

/// How a network's f16 body convolutions choose their kernel.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub enum Kernels {
    /// Autotune runs the fastest of the candidates the device offers for each tune key.
    ///
    /// The first run of a key benchmarks every candidate. The winner is kept in CubeCL's autotune cache, so
    /// later runs on a device with the same runtime and limits skip the benchmarks.
    #[default]
    Tuned,
    /// The default direct variant wherever the device offers it, and the fused cubek convolution elsewhere.
    ///
    /// Nothing is benchmarked, so the output is the same on every device that offers the variant.
    Fixed,
    /// Autotune runs the fastest of `candidates`, tuned apart from every other list by `label`.
    #[cfg(any(test, feature = "test-support"))]
    Among {
        label: String,
        candidates: Vec<Candidate>,
    },
}

/// How one body convolution ran.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TunedRun {
    pub run: LayerRun,
    /// How many times a candidate ran, which is more than once when the convolution's key was benchmarked.
    pub executions: u32,
}

/// The inputs of a tuned body convolution, at any lifetime.
struct BodyInputs<R>(PhantomData<fn() -> R>);

impl<R: Runtime> TuneInputs for BodyInputs<R> {
    type At<'a> = TuneInput<'a, R>;
}

/// One body convolution handed to the tuner, counting how many times a candidate runs it.
///
/// Outside the benchmarks, a candidate that fails to launch leaves its error in `failure` and reports a run,
/// since CubeCL panics when the candidate it chose gives an error.
struct TuneInput<'a, R: Runtime> {
    client: ComputeClient<R>,
    convolution: BodyConvolution<'a>,
    epilogue: Epilogue,
    executions: &'a AtomicU32,
    benchmarking: bool,
    failure: &'a Mutex<Option<String>>,
}

impl<R: Runtime> Clone for TuneInput<'_, R> {
    fn clone(&self) -> Self {
        Self {
            client: self.client.clone(),
            convolution: self.convolution,
            epilogue: self.epilogue,
            executions: self.executions,
            benchmarking: self.benchmarking,
            failure: self.failure,
        }
    }
}

/// Runs every candidate once before a key is benchmarked, then hands the tuner the same inputs marked as
/// benchmark inputs.
///
/// The first run compiles each candidate and brings the device's clocks up, so the timed runs that follow
/// measure the candidates alone.
struct PrimedInputs {
    candidates: Vec<Candidate>,
}

impl<R: Runtime> InputGenerator<TuneKey, BodyInputs<R>> for PrimedInputs {
    // The inputs are named through the projection so the lifetime binds as the trait's does.
    fn generate<'a>(
        &self,
        _key: &TuneKey,
        inputs: &<BodyInputs<R> as TuneInputs>::At<'a>,
    ) -> <BodyInputs<R> as TuneInputs>::At<'a> {
        for candidate in self.candidates.iter().copied() {
            inputs.executions.fetch_add(1, Ordering::Relaxed);

            // A candidate that fails here fails again when the tuner times it, which skips it.
            let _ = launch_candidate(&inputs.client, candidate, &inputs.convolution);
        }

        // A device failure here shows again when the candidates are timed.
        let pending = inputs.client.sync();
        let _ = cubecl::future::block_on(pending);

        TuneInput {
            benchmarking: true,
            ..inputs.clone()
        }
    }
}

fn tune_key<R: Runtime>(input: &TuneInput<'_, R>) -> TuneKey {
    let plane_width = input.client.properties().hardware.plane_size_max;

    TuneKey::of(&input.convolution, input.epilogue, plane_width)
}

/// Runs `input` with `candidate`, for the tuner.
///
/// A failure is an error while benchmarking, which the tuner skips. Otherwise it is left in the input's
/// failure slot.
fn run_tunable<R: Runtime>(input: TuneInput<'_, R>, candidate: Candidate) -> Result<LayerRun, String> {
    input.executions.fetch_add(1, Ordering::Relaxed);

    let launched = launch_candidate(&input.client, candidate, &input.convolution);

    let error = match launched {
        Ok(forced) => return Ok(LayerRun::of_candidate(candidate, forced)),
        Err(error) => error.to_string(),
    };

    if input.benchmarking {
        return Err(error);
    }

    let mut failure = input.failure.lock().unwrap_or_else(PoisonError::into_inner);
    *failure = Some(error);

    Ok(LayerRun::of_candidate(candidate, false))
}

fn tunable<R: Runtime>(candidate: Candidate) -> Tunable<TuneKey, BodyInputs<R>, LayerRun> {
    let name = candidate.name();

    Tunable::new(&name, move |input: TuneInput<'_, R>| {
        run_tunable(input, candidate)
    })
}

fn tunable_set<R: Runtime>(candidates: &[Candidate]) -> TunableSet<TuneKey, BodyInputs<R>, LayerRun> {
    let primed = PrimedInputs {
        candidates: candidates.to_vec(),
    };
    let mut set = TunableSet::new(tune_key::<R>, primed);

    for candidate in candidates.iter().copied() {
        let entry = tunable::<R>(candidate);
        set = set.with(entry);
    }

    set
}

/// The name the tuner keeps a device's tunings of the list `label` under.
///
/// The device is named by its runtime and the limits its candidates are chosen by.
fn tuner_id<R: Runtime>(client: &ComputeClient<R>, label: &str) -> String {
    let runtime = R::name(client);
    let hardware = &client.properties().hardware;

    format!(
        "{runtime}-subgroups-{}-{}-shared-{}-units-{}-{label}",
        hardware.plane_size_min,
        hardware.plane_size_max,
        hardware.max_shared_memory_size,
        hardware.max_units_per_cube,
    )
}

/// The candidates a device with `properties` offers maps of `features` channels: the direct variants first,
/// then the Winograd variants if `winograd` is set, and the fused cubek convolution last.
pub fn offered_candidates(properties: &DeviceProperties, features: u32, winograd: bool) -> Vec<Candidate> {
    let variants = direct_variants(properties, features);
    let mut candidates: Vec<Candidate> = variants.into_iter().map(Candidate::Direct).collect();

    if winograd {
        let variants = winograd_variants(properties, features);
        candidates.extend(variants.into_iter().map(Candidate::Winograd));
    }

    candidates.push(Candidate::FusedCubek);

    candidates
}

/// The candidate [Kernels::Fixed](crate::cnn::Kernels::Fixed) runs on a device with `properties`, for maps
/// of `features` channels.
///
/// That is the default direct variant at the device's widest subgroup where the device offers it, and the
/// fused cubek convolution elsewhere.
pub fn fixed_candidate(properties: &DeviceProperties, features: u32) -> Candidate {
    let plane_width = properties.hardware.plane_size_max;
    let variant = DirectVariant {
        plane_width,
        ..DirectVariant::default()
    };
    let offered = direct_variants(properties, features);

    if offered.contains(&variant) {
        return Candidate::Direct(variant);
    }

    Candidate::FusedCubek
}

/// What a plan is kept under: the runtime and device facts it was worked out from, whether Winograd is
/// offered, the width and the kernel choice.
#[derive(PartialEq, Eq, Hash)]
struct PlanKey {
    runtime: TypeId,
    runtime_name: &'static str,
    plane_size_min: u32,
    plane_size_max: u32,
    shared_bytes: usize,
    units: u32,
    offers_matrix: bool,
    winograd: bool,
    features: u32,
    kernels: Kernels,
}

impl PlanKey {
    fn of<R: Runtime>(client: &ComputeClient<R>, kernels: &Kernels, winograd: bool, features: u32) -> Self {
        let hardware = &client.properties().hardware;

        Self {
            runtime: TypeId::of::<R>(),
            runtime_name: R::name(client),
            plane_size_min: hardware.plane_size_min,
            plane_size_max: hardware.plane_size_max,
            shared_bytes: hardware.max_shared_memory_size,
            units: hardware.max_units_per_cube,
            offers_matrix: offers_cmma_16(client),
            winograd,
            features,
            kernels: kernels.clone(),
        }
    }
}

/// How a kernel choice runs on one device and map width.
enum Plan<R: Runtime> {
    /// No candidate can run, for the reasons given.
    Refused(String),
    /// One candidate runs without tuning. `winograd` is whether the list it came from holds a Winograd
    /// variant.
    Once { candidate: Candidate, winograd: bool },
    /// The tuner chooses from `set`, keeping its tunings under `id`. `winograd` is whether the set holds a
    /// Winograd variant.
    Tuned {
        id: String,
        set: Arc<TunableSet<TuneKey, BodyInputs<R>, LayerRun>>,
        winograd: bool,
    },
}

/// How a list of candidates runs on one device, before any tuning is set up.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum Choice {
    /// No candidate can run, for the reasons given.
    Refused(String),
    /// One candidate runs without tuning.
    Once(Candidate),
    /// The tuner chooses from the list.
    Tuned,
}

/// How `candidates` run on a device with `properties`, on maps of `features` channels.
///
/// CubeCL's tuner panics when every candidate fails while benchmarking. A direct candidate that passed its
/// check does not fail to launch, so only a list holding one is tuned. Without one, the fused cubek
/// convolution runs where it is listed, since an untuned variant may be slower than it. Otherwise the first
/// runnable candidate runs.
pub(super) fn choose(properties: &DeviceProperties, candidates: &[Candidate], features: u32) -> Choice {
    let mut runnable = Vec::new();
    let mut refusals = Vec::new();

    for candidate in candidates.iter().copied() {
        let checked = candidate.check(properties, features);

        match checked {
            Ok(()) => runnable.push(candidate),
            Err(reason) => refusals.push(format!("{} ({reason})", candidate.name())),
        }
    }

    let Some(first) = runnable.first().copied() else {
        let reasons = refusals.join(", ");
        let message = format!("no body convolution candidate can run: {reasons}");

        return Choice::Refused(message);
    };

    let holds_direct = runnable
        .iter()
        .any(|candidate| matches!(candidate, Candidate::Direct(_)));
    let holds_cubek = runnable.contains(&Candidate::FusedCubek);
    if !holds_direct && holds_cubek {
        return Choice::Once(Candidate::FusedCubek);
    }

    if !holds_direct || candidates.len() == 1 {
        return Choice::Once(first);
    }

    Choice::Tuned
}

/// The plan for `candidates`, tuned under `label`, on maps of `features` channels.
fn plan_among<R: Runtime>(
    client: &ComputeClient<R>,
    label: &str,
    candidates: &[Candidate],
    features: u32,
) -> Plan<R> {
    let properties = client.properties();
    let winograd = candidates
        .iter()
        .any(|candidate| matches!(candidate, Candidate::Winograd(_)));

    let choice = choose(properties, candidates, features);

    match choice {
        Choice::Refused(message) => Plan::Refused(message),
        Choice::Once(candidate) => Plan::Once { candidate, winograd },
        Choice::Tuned => {
            let id = tuner_id(client, label);
            let set = tunable_set::<R>(candidates);
            let shared_set = Arc::new(set);

            Plan::Tuned {
                id,
                set: shared_set,
                winograd,
            }
        },
    }
}

fn make_plan<R: Runtime>(
    client: &ComputeClient<R>,
    kernels: &Kernels,
    winograd: bool,
    features: u32,
) -> Plan<R> {
    let properties = client.properties();

    match kernels {
        Kernels::Tuned => {
            let candidates = offered_candidates(properties, features, winograd);
            let label = match winograd {
                true => OFFERED_WINOGRAD_LABEL,
                false => OFFERED_LABEL,
            };

            plan_among(client, label, &candidates, features)
        },
        Kernels::Fixed => {
            let candidate = fixed_candidate(properties, features);

            Plan::Once {
                candidate,
                winograd: false,
            }
        },
        #[cfg(any(test, feature = "test-support"))]
        Kernels::Among { label, candidates } => plan_among(client, label, candidates, features),
    }
}

/// The plan `kernels` runs by on the client's device for maps of `features` channels, worked out on its first
/// use. `winograd` is whether the Winograd candidates are offered.
fn plan_for<R: Runtime>(
    client: &ComputeClient<R>,
    kernels: &Kernels,
    winograd: bool,
    features: u32,
) -> Arc<Plan<R>> {
    let key = PlanKey::of(client, kernels, winograd, features);
    let mut plans = PLANS.lock().unwrap_or_else(PoisonError::into_inner);

    let known = plans.get(&key).cloned();
    if let Some(plan) = known.and_then(|shared| shared.downcast::<Plan<R>>().ok()) {
        return plan;
    }

    let made = make_plan(client, kernels, winograd, features);
    let plan = Arc::new(made);
    plans.insert(key, plan.clone());

    plan
}

fn run_once<R: Runtime>(
    client: &ComputeClient<R>,
    candidate: Candidate,
    convolution: &BodyConvolution<'_>,
) -> Result<TunedRun, CnnError> {
    let forced = launch_candidate(client, candidate, convolution)?;
    let run = LayerRun::of_candidate(candidate, forced);

    Ok(TunedRun { run, executions: 1 })
}

/// Runs `convolution` with the fastest candidate of `set`, benchmarking the set the first time a key is
/// seen.
///
/// Gives an error when the chosen candidate fails to launch.
fn run_tuner<R: Runtime>(
    client: &ComputeClient<R>,
    id: &String,
    set: &Arc<TunableSet<TuneKey, BodyInputs<R>, LayerRun>>,
    convolution: &BodyConvolution<'_>,
    epilogue: Epilogue,
) -> Result<TunedRun, CnnError> {
    let executions = AtomicU32::new(0);
    let failure = Mutex::new(None);
    let input = TuneInput {
        client: client.clone(),
        convolution: *convolution,
        epilogue,
        executions: &executions,
        benchmarking: false,
        failure: &failure,
    };

    let run = TUNER.execute(id, client, set.clone(), input);
    let count = executions.load(Ordering::Relaxed);
    let failed = failure.into_inner().unwrap_or_else(PoisonError::into_inner);

    if let Some(message) = failed {
        return Err(CnnError::Convolution(message));
    }

    Ok(TunedRun {
        run,
        executions: count,
    })
}

/// `tuned`, reporting whether a Winograd variant was among the candidates its kernel was chosen from.
fn offering_winograd(tuned: TunedRun, winograd: bool) -> TunedRun {
    let run = LayerRun {
        winograd_offered: winograd,
        ..tuned.run
    };

    TunedRun { run, ..tuned }
}

/// Runs `convolution`, which applies `epilogue`, with the candidate `kernels` chooses.
///
/// Gives an error when no candidate can run it, or when the chosen one fails to launch.
pub fn run_tuned<R: Runtime>(
    client: &ComputeClient<R>,
    kernels: &Kernels,
    convolution: &BodyConvolution<'_>,
    epilogue: Epilogue,
) -> Result<TunedRun, CnnError> {
    convolution.check();

    let plan = plan_for(client, kernels, convolution.winograd, convolution.features);

    match plan.as_ref() {
        Plan::Refused(reasons) => Err(CnnError::Convolution(reasons.clone())),
        Plan::Once { candidate, winograd } => {
            let once = run_once(client, *candidate, convolution)?;

            Ok(offering_winograd(once, *winograd))
        },
        Plan::Tuned { id, set, winograd } => {
            let tuned = run_tuner(client, id, set, convolution, epilogue)?;

            Ok(offering_winograd(tuned, *winograd))
        },
    }
}

/// Whether `layer` on `path` runs with the candidate its kernels choose.
///
/// That is the convolution path on f16 maps whose features are a multiple of 16.
pub(crate) fn takes_tuner(layer: &ConvLayer, path: BodyPath) -> bool {
    let tiled = layer.features.is_multiple_of(16);

    path == BodyPath::Convolution && layer.precision == Precision::F16 && tiled
}
