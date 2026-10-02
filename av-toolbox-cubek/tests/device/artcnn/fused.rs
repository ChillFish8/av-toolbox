use av_toolbox_cubek::artcnn::{
    Activation,
    BodyPath,
    Finish,
    Network,
    Precision,
    Region,
    Weights,
    body_layer_with_blueprint,
    convolve_with_blueprint,
};
use av_toolbox_gpu::{BuildContext, BuildTask};
use cubecl::prelude::Runtime;
use cubecl::server::Handle;

use super::body::{r8f64, supports};
use super::{BlueprintRuns, forced_eligible, read_maps, steps_from_separate, synced, upload_maps, wavy_maps};
use crate::support::{Sizing, backend, empty};

/// The region every fused write-out test convolves, large enough for the forced blueprint.
const REGION: Region = Region {
    x: 0,
    y: 0,
    width: 200,
    height: 120,
};

/// The R8F64 body layer each test runs.
const LAYER: usize = 1;

/// Which epilogue a fused write-out test checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Epilogue {
    Silu,
    Residual,
}

/// One body layer of the same maps run four ways: fused with the forced blueprint allowed and with the
/// inferred one, through the unfused path's separate passes, and its bare convolution and bias.
struct FusedLayer {
    weights: Weights,
    maps: Vec<f32>,
    residual: Vec<f32>,
    epilogue: Epilogue,
}

struct FusedOutputs {
    blueprints: BlueprintRuns,
    separate: Vec<f32>,
    convolved: Vec<f32>,
    forced_fused: bool,
    inferred_fused: bool,
}

impl BuildTask for FusedLayer {
    type Output = FusedOutputs;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let sizing = Sizing(build.frame_context());
        let eligible = forced_eligible(client);
        let precision = Precision::F16;
        let network = Network::upload(client, &self.weights, precision).expect("the weights are valid");
        let map_bytes = self.maps.len() * precision.bytes();

        let input = upload_maps(client, &self.maps, precision);
        let residual = upload_maps(client, &self.residual, precision);
        let forced_output = empty(client, map_bytes);
        let inferred_output = empty(client, map_bytes);
        let separate_output = empty(client, map_bytes);
        let convolved_output = empty(client, map_bytes);

        let finish = finish_for(self.epilogue, &residual);
        let layer_run = |path: BodyPath, output: &Handle, allow_forced: bool| {
            body_layer_with_blueprint(
                client,
                &sizing,
                &network,
                path,
                LAYER,
                &input,
                output,
                REGION,
                finish,
                allow_forced,
            )
        };

        let forced_ran = layer_run(BodyPath::Convolution, &forced_output, true);
        let inferred_ran = layer_run(BodyPath::Convolution, &inferred_output, false);
        let separate_ran = layer_run(BodyPath::UnfusedConvolution, &separate_output, false);
        let convolved_ran = convolve_with_blueprint(
            client,
            &sizing,
            &network,
            LAYER,
            &input,
            &convolved_output,
            REGION,
            false,
        );
        synced(client);

        let forced_run = forced_ran.expect("the forced fused layer runs");
        let inferred_run = inferred_ran.expect("the inferred fused layer runs");
        separate_ran.expect("the separate passes run");
        convolved_ran.expect("the bare convolution runs");
        assert!(!inferred_run.forced, "the inferred run took the forced blueprint");

        let blueprints = BlueprintRuns {
            forced: read_maps(client, &forced_output, precision),
            inferred: read_maps(client, &inferred_output, precision),
            forced_ran: forced_run.forced,
            eligible,
        };

        FusedOutputs {
            blueprints,
            separate: read_maps(client, &separate_output, precision),
            convolved: read_maps(client, &convolved_output, precision),
            forced_fused: forced_run.fused,
            inferred_fused: inferred_run.fused,
        }
    }
}

fn finish_for(epilogue: Epilogue, residual: &Handle) -> Finish<'_> {
    match epilogue {
        Epilogue::Silu => Finish {
            activation: Activation::Silu,
            residual: None,
        },
        Epilogue::Residual => Finish {
            activation: Activation::None,
            residual: Some(residual),
        },
    }
}

/// Runs R8F64's first body layer over maps between -2 and 2 with `epilogue`, or gives `None` without an f16
/// cooperative matrix.
fn run_fused_layer(epilogue: Epilogue) -> Option<FusedOutputs> {
    if !supports(BodyPath::Convolution, Precision::F16) {
        println!("fused {epilogue:?}: no f16 cooperative matrix, so there is nothing to compare");
        return None;
    }

    let values = (REGION.samples() * 64) as usize;
    let maps = wavy_maps(values);
    let residual: Vec<f32> = (0..values)
        .map(|index| (index as f32 * 0.11).cos() * 3.0)
        .collect();
    let task = FusedLayer {
        weights: r8f64(),
        maps,
        residual,
        epilogue,
    };

    let outputs = backend().build(task);

    Some(outputs)
}

/// Asserts the fused write-outs, forced and inferred, ran fused and lie within one f16 step of the separate
/// passes.
fn assert_within_one_step(outputs: &FusedOutputs, epilogue: Epilogue) {
    let blueprints = &outputs.blueprints;
    let runs = [
        ("forced", &blueprints.forced, outputs.forced_fused),
        ("inferred", &blueprints.inferred, outputs.inferred_fused),
    ];

    for (name, fused, fused_ran) in runs {
        let (differing, largest_share) = steps_from_separate(fused, &outputs.separate, &outputs.convolved);
        let values = fused.len();
        let shown = format!("{differing} of {values} values differ, largest {largest_share} f16 steps");
        println!("fused {epilogue:?} {name}: {shown}");

        assert!(fused_ran, "{name}: the layer ran its epilogue in a separate pass");
        // The convolution reaches the fused write-out already cast to f16, so the two compute the same. A
        // driver may fold away that cast and its widening back to f32, which rounds the value once rather
        // than twice. Measured on RADV, 25.6 % of values with SiLU and 26.8 % with the residual lie one step
        // apart.
        assert!(largest_share <= 1.0, "{name}: {largest_share} f16 steps");
    }
}

#[test]
fn fused_silu_matches_the_separate_passes_within_one_f16_step() {
    let Some(outputs) = run_fused_layer(Epilogue::Silu) else {
        return;
    };

    assert_within_one_step(&outputs, Epilogue::Silu);
}

#[test]
fn a_fused_residual_matches_the_separate_passes_within_one_f16_step() {
    let Some(outputs) = run_fused_layer(Epilogue::Residual) else {
        return;
    };

    assert_within_one_step(&outputs, Epilogue::Residual);
}

#[test]
fn fused_silu_with_the_forced_blueprint_matches_the_inferred_one_exactly() {
    let Some(outputs) = run_fused_layer(Epilogue::Silu) else {
        return;
    };

    outputs.blueprints.assert_forced_matches_inferred("fused Silu");
}

#[test]
fn a_fused_residual_with_the_forced_blueprint_matches_the_inferred_one_exactly() {
    let Some(outputs) = run_fused_layer(Epilogue::Residual) else {
        return;
    };

    outputs
        .blueprints
        .assert_forced_matches_inferred("fused Residual");
}
