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
    offers_cmma_16,
};
use av_toolbox_gpu::{BuildContext, BuildTask};
use cubecl::prelude::Runtime;
use cubecl::server::Handle;

use super::body::{bits_differ, f16_step, r8f64, supports};
use super::{read_maps, synced, upload_maps};
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
    forced: Vec<f32>,
    inferred: Vec<f32>,
    separate: Vec<f32>,
    convolved: Vec<f32>,
    forced_ran: bool,
    eligible: bool,
}

impl BuildTask for FusedLayer {
    type Output = FusedOutputs;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let sizing = Sizing(build.frame_context());
        let plane_size_max = client.properties().hardware.plane_size_max;
        let offers_matrix = offers_cmma_16(client);
        let eligible = plane_size_max == 64 && offers_matrix;
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

        let forced_ran = forced_ran.expect("the forced fused layer runs");
        let inferred_forced = inferred_ran.expect("the inferred fused layer runs");
        separate_ran.expect("the separate passes run");
        convolved_ran.expect("the bare convolution runs");
        assert!(!inferred_forced, "the inferred run took the forced blueprint");

        FusedOutputs {
            forced: read_maps(client, &forced_output, precision),
            inferred: read_maps(client, &inferred_output, precision),
            separate: read_maps(client, &separate_output, precision),
            convolved: read_maps(client, &convolved_output, precision),
            forced_ran,
            eligible,
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
    let maps: Vec<f32> = (0..values)
        .map(|index| (index as f32 * 0.37).sin() * 2.0)
        .collect();
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

/// Asserts the fused write-outs, forced and inferred, lie within one f16 step of the separate passes and
/// differ from them somewhere.
fn assert_within_one_step(outputs: &FusedOutputs, epilogue: Epilogue) {
    let runs = [("forced", &outputs.forced), ("inferred", &outputs.inferred)];

    for (name, fused) in runs {
        let (differing, largest_share) = steps_from_separate(fused, outputs);
        let values = fused.len();
        let shown = format!("{differing} of {values} values differ, largest {largest_share} f16 steps");
        println!("fused {epilogue:?} {name}: {shown}");

        assert!(largest_share <= 1.0, "{name}: {largest_share} f16 steps");
        // A write-out that ran the separate passes would match them exactly.
        assert!(
            differing > 0,
            "{name}: the fused write-out matches the separate passes exactly"
        );
    }
}

/// How many of `fused` differ from the separate passes, and the largest difference in f16 steps.
///
/// The separate passes round the convolution to f16 before the epilogue and the fused write-out does not, so
/// the step is taken at the largest of the fused, separate and convolved values.
fn steps_from_separate(fused: &[f32], outputs: &FusedOutputs) -> (usize, f32) {
    let mut differing = 0;
    let mut largest_share = 0.0f32;

    for (index, fused_value) in fused.iter().copied().enumerate() {
        let separate = outputs.separate[index];
        let convolved = outputs.convolved[index];
        let magnitude = fused_value.abs().max(separate.abs()).max(convolved.abs());
        let step = f16_step(magnitude);
        let share = (fused_value - separate).abs() / step;

        largest_share = largest_share.max(share);
        if fused_value != separate {
            differing += 1;
        }
    }

    (differing, largest_share)
}

/// Asserts the forced blueprint ran exactly where the device is eligible, and matches the inferred one bit
/// for bit there.
fn assert_forced_matches_inferred(outputs: &FusedOutputs, epilogue: Epilogue) {
    println!(
        "fused {epilogue:?}: forced ran {}, eligible {}",
        outputs.forced_ran, outputs.eligible
    );
    if !outputs.eligible {
        assert!(
            !outputs.forced_ran,
            "the forced blueprint ran on an ineligible device"
        );
        return;
    }

    assert!(
        outputs.forced_ran,
        "a large region on an eligible device kept the inferred blueprint"
    );

    let values = outputs.forced.len();
    let pairs = outputs.forced.iter().zip(&outputs.inferred);
    let differing = pairs.clone().filter(|pair| bits_differ(*pair)).count();
    let first = pairs.clone().position(bits_differ);

    if let Some(index) = first {
        let forced = outputs.forced[index];
        let inferred = outputs.inferred[index];
        panic!(
            "{differing} of {values} values differ, first at {index}: forced {forced}, inferred {inferred}"
        );
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

    assert_forced_matches_inferred(&outputs, Epilogue::Silu);
}

#[test]
fn a_fused_residual_with_the_forced_blueprint_matches_the_inferred_one_exactly() {
    let Some(outputs) = run_fused_layer(Epilogue::Residual) else {
        return;
    };

    assert_forced_matches_inferred(&outputs, Epilogue::Residual);
}
