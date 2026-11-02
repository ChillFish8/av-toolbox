use av_toolbox_cubek::artcnn::{
    BodyConvolution,
    Candidate,
    DirectVariant,
    Kernels,
    Network,
    Precision,
    TunedRun,
    WeightStaging,
    offered_candidates,
    offers_cmma_16,
    run_candidate,
    run_tuned,
};
use av_toolbox_cubek::testing::fresh_tune_label;
use av_toolbox_gpu::{BuildContext, BuildTask};
use cubecl::prelude::Client;

use super::kernels::{
    EPILOGUES,
    Epilogue,
    Runner,
    edged_maps,
    offered_variants,
    region,
    runner_matches_separate,
    weights_of,
};
use super::{synced, upload_maps};
use crate::support::{backend, empty};

/// The candidates the test device offers for maps 64 wide.
struct Offered;

impl BuildTask for Offered {
    type Output = Vec<Candidate>;

    fn run(self, build: &BuildContext) -> Self::Output {
        let properties = build.client().properties();

        offered_candidates(properties, 64, None)
    }
}

/// Runs `kernels` over a body layer, asserting the output lies within one f16 step of the separate passes,
/// and gives how the chosen candidate ran. Gives `None` without an f16 cooperative matrix.
fn chosen_run(kernels: &Kernels, width: u32, height: u32, epilogue: Epilogue) -> Option<TunedRun> {
    let runner = Runner::Chosen(kernels.clone());

    let outputs = runner_matches_separate(runner, 64, width, height, epilogue)?;
    let chosen = outputs.chosen.expect("the kernels report how they ran");
    println!("{epilogue:?} {width}x{height}: {chosen:?}");

    Some(chosen)
}

#[test]
fn the_tuner_chooses_a_candidate_for_each_key_and_does_not_benchmark_it_again() {
    let candidates = backend().build(Offered);
    if candidates.len() < 2 {
        println!("the device offers no direct variant, so there is nothing to tune");
        return;
    }

    let kernels = Kernels::Among {
        label: fresh_tune_label("each-key"),
        candidates: candidates.clone(),
    };

    // 24,000 samples are under 64k, and 76,800 between 64k and 1M.
    for (width, height) in [(200, 120), (320, 240)] {
        for epilogue in EPILOGUES {
            let Some(first) = chosen_run(&kernels, width, height, epilogue) else {
                return;
            };
            let second = chosen_run(&kernels, width, height, epilogue).expect("the device has not changed");

            let chosen = first.run.candidate.expect("the tuner reports its candidate");
            assert!(candidates.contains(&chosen), "{chosen:?}");
            assert!(first.executions > 1, "the first run did not benchmark");
            assert_eq!(second.executions, 1, "the second run benchmarked again");
            assert_eq!(second.run.candidate, Some(chosen));
        }
    }
}

#[test]
fn a_candidate_injected_to_fail_is_skipped() {
    let offered = offered_variants(64);
    let Some(default_variant) = offered.first().copied() else {
        println!("the device offers no direct variant, so there is nothing to tune");
        return;
    };

    // Columns that are not a multiple of 16 are refused when the candidate runs.
    let failing = DirectVariant {
        columns: 24,
        ..default_variant
    };
    let failing_candidate = Candidate::Direct(failing);
    let kernels = Kernels::Among {
        label: fresh_tune_label("injected-failure"),
        candidates: vec![
            failing_candidate,
            Candidate::Direct(default_variant),
            Candidate::FusedCubek,
        ],
    };

    for epilogue in EPILOGUES {
        let Some(tuned) = chosen_run(&kernels, 200, 120, epilogue) else {
            return;
        };

        let chosen = tuned.run.candidate.expect("the tuner reports its candidate");
        assert_ne!(chosen, failing_candidate);
        assert!(tuned.executions > 1, "the first run did not benchmark");
    }
}

#[test]
fn pinned_kernels_run_the_default_direct_variant_without_tuning() {
    let offered = offered_variants(64);
    let fixed = match offered.first() {
        Some(variant) => Candidate::Direct(*variant),
        None => Candidate::FusedCubek,
    };

    if let Candidate::Direct(variant) = fixed {
        let default_layout = DirectVariant {
            plane_width: variant.plane_width,
            ..DirectVariant::default()
        };
        assert_eq!(
            variant, default_layout,
            "the first offered variant is not the default"
        );
    }

    for epilogue in EPILOGUES {
        for (width, height) in [(200, 120), (31, 7)] {
            let Some(pinned) = chosen_run(&Kernels::Fixed, width, height, epilogue) else {
                return;
            };

            assert_eq!(pinned.run.candidate, Some(fixed));
            assert_eq!(pinned.executions, 1);
        }
    }
}

/// The median time each candidate takes on one large body convolution, and which one the tuner chose for it.
struct Timed {
    candidates: Vec<Candidate>,
    label: String,
}

struct Timings {
    chosen: Candidate,
    milliseconds: Vec<(Candidate, f64)>,
}

/// The milliseconds one run of `candidate` over `convolution` takes on the device, timed as the tuner times
/// its candidates.
fn time_once(client: &Client, candidate: Candidate, convolution: &BodyConvolution<'_>) -> f64 {
    let profiled = client.profile(|| run_candidate(client, candidate, convolution), "candidate");
    let (ran, duration) = profiled.expect("the device profiles a run");
    ran.expect("a candidate the tuner measured runs");

    let resolving = duration.resolve();
    let measured = cubecl::future::block_on(resolving);
    let ticks = measured.expect("the device timed the run");

    ticks.duration().as_secs_f64() * 1e3
}

/// The median milliseconds of each of `candidates` over `convolution`, after three runs each to warm up.
///
/// The runs take turns, one of each candidate a round for `rounds` rounds, so a burst of other work on the
/// device slows every candidate alike.
fn interleaved_medians(
    client: &Client,
    candidates: &[Candidate],
    convolution: &BodyConvolution<'_>,
    rounds: usize,
) -> Vec<(Candidate, f64)> {
    for candidate in candidates.iter().copied() {
        for _ in 0..3 {
            let warmed = run_candidate(client, candidate, convolution);
            warmed.expect("a candidate the tuner measured runs");
        }
    }

    let mut samples = vec![Vec::with_capacity(rounds); candidates.len()];

    for _ in 0..rounds {
        for (index, candidate) in candidates.iter().copied().enumerate() {
            let milliseconds = time_once(client, candidate, convolution);
            samples[index].push(milliseconds);
        }
    }

    candidates.iter().copied().zip(samples).map(median_of).collect()
}

fn median_of((candidate, mut samples): (Candidate, Vec<f64>)) -> (Candidate, f64) {
    samples.sort_by(f64::total_cmp);
    let median = samples[samples.len() / 2];

    (candidate, median)
}

impl BuildTask for Timed {
    type Output = Option<Timings>;

    fn run(self, build: &BuildContext) -> Self::Output {
        let client = build.client();
        if !offers_cmma_16(client) {
            return None;
        }

        let precision = Precision::F16;
        let weights = weights_of(64);
        let network = Network::upload(client, &weights, precision).expect("the weights are valid");
        // 1,310,720 samples, over 1M.
        let frame = region(1280, 1024);
        let maps = edged_maps(frame, 64);
        let epilogue = Epilogue::Silu;
        let layer = epilogue.layer(&weights);
        let values = maps.len();

        let input = upload_maps(client, &maps, precision);
        let output = empty(client, values * precision.bytes());
        // SiLU adds no residual, so the input only fills its place.
        let finish = epilogue.finish(&input);
        let convolution = BodyConvolution::of_layer(&network, layer, &input, &output, frame, finish);
        let kernels = Kernels::Among {
            label: self.label,
            candidates: self.candidates.clone(),
        };

        let tuned = run_tuned(client, &kernels, &convolution, epilogue.kind());
        let tuned = tuned.expect("the tuner finds a candidate");
        let chosen = tuned.run.candidate.expect("the tuner reports its candidate");

        let milliseconds = interleaved_medians(client, &self.candidates, &convolution, 21);

        Some(Timings { chosen, milliseconds })
    }
}

#[test]
fn the_tuner_chooses_the_fastest_measured_candidate() {
    let offered = offered_variants(64);
    let Some(default_variant) = offered.first().copied() else {
        println!("the device offers no direct variant, so there is nothing to tune");
        return;
    };

    let mut candidates = vec![Candidate::Direct(default_variant), Candidate::FusedCubek];
    let staged = offered
        .iter()
        .find(|variant| variant.weights == WeightStaging::Double);
    if let Some(variant) = staged {
        candidates.push(Candidate::Direct(*variant));
    }

    let task = Timed {
        candidates,
        label: fresh_tune_label("fastest"),
    };

    let Some(timings) = backend().build(task) else {
        println!("no f16 cooperative matrix, so there is nothing to tune");
        return;
    };

    for (candidate, median) in &timings.milliseconds {
        let name = candidate.name();
        println!("{name}: median {median:.3} ms");
    }

    let fastest = timings
        .milliseconds
        .iter()
        .map(|(_, median)| *median)
        .fold(f64::INFINITY, f64::min);
    let chosen_time = timings
        .milliseconds
        .iter()
        .find(|(candidate, _)| *candidate == timings.chosen)
        .map(|(_, median)| *median)
        .expect("the chosen candidate was timed");
    let chosen_name = timings.chosen.name();
    println!("chosen {chosen_name}: {chosen_time:.3} ms against the fastest {fastest:.3} ms");

    // The candidates differ by 1.6x and more, while load and run-to-run noise move a median by a few percent.
    assert!(
        chosen_time <= fastest * 1.2,
        "{chosen_name} takes {chosen_time:.3} ms, the fastest {fastest:.3} ms"
    );
}

/// One body convolution run twice under one label, first by `tuned` and then by `changed`.
struct ChangedList {
    tuned: Vec<Candidate>,
    changed: Vec<Candidate>,
    label: String,
}

impl BuildTask for ChangedList {
    type Output = Option<(TunedRun, Result<TunedRun, av_toolbox_cubek::artcnn::CnnError>)>;

    fn run(self, build: &BuildContext) -> Self::Output {
        let client = build.client();
        if !offers_cmma_16(client) {
            return None;
        }

        let precision = Precision::F16;
        let weights = weights_of(64);
        let network = Network::upload(client, &weights, precision).expect("the weights are valid");
        let frame = region(200, 120);
        let maps = edged_maps(frame, 64);
        let epilogue = Epilogue::Silu;
        let layer = epilogue.layer(&weights);

        let input = upload_maps(client, &maps, precision);
        let output = empty(client, maps.len() * precision.bytes());
        // SiLU adds no residual, so the input only fills its place.
        let finish = epilogue.finish(&input);
        let convolution = BodyConvolution::of_layer(&network, layer, &input, &output, frame, finish);
        let tuned_kernels = Kernels::Among {
            label: self.label.clone(),
            candidates: self.tuned,
        };
        let changed_kernels = Kernels::Among {
            label: self.label,
            candidates: self.changed,
        };

        let tuned = run_tuned(client, &tuned_kernels, &convolution, epilogue.kind());
        let tuned = tuned.expect("the first list tunes");
        let changed = run_tuned(client, &changed_kernels, &convolution, epilogue.kind());
        synced(client);

        Some((tuned, changed))
    }
}

#[test]
fn a_changed_candidate_list_tunes_apart_from_the_list_before_it() {
    let offered = offered_variants(64);
    let Some(default_variant) = offered.first().copied() else {
        println!("the device offers no direct variant, so there is nothing to tune");
        return;
    };

    let good = Candidate::Direct(default_variant);
    let failing = Candidate::Direct(DirectVariant {
        columns: 24,
        ..default_variant
    });
    // Both of the first list's indices hold a failing candidate in the second, so reusing the first list's
    // winner would launch a failing candidate.
    let task = ChangedList {
        tuned: vec![good, good],
        changed: vec![failing, failing, good],
        label: fresh_tune_label("changed-list"),
    };

    let Some((tuned, changed)) = backend().build(task) else {
        println!("no f16 cooperative matrix, so there is nothing to tune");
        return;
    };

    println!("tuned {tuned:?}, changed {changed:?}");
    let changed = changed.expect("the changed list runs its own winner");
    assert!(tuned.executions > 1, "the first list did not benchmark");
    assert!(
        changed.executions > 1,
        "the changed list reused the first list's tuning"
    );
    assert_eq!(changed.run.candidate, Some(good));
}
