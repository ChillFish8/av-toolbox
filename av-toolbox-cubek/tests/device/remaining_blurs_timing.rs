use std::time::{Duration, Instant};

use av_toolbox_cubek::blur::{BlurPlanes, FilterPlanes, Gaussian, Separable, Storage};
use av_toolbox_cubek::dehalo::{DifferencePlanes, difference};
use av_toolbox_cubek::deinterlace::qtgmc::fmtc_gauss_taps;
use av_toolbox_cubek::lineart::{self, VinverseParams, VinversePlanes};
use av_toolbox_cubek::neighbourhood::{self, BinomialChain, NeighbourPlanes, Smooth};
use av_toolbox_cubek::resize::{ResizePlanes, apply_float_with_edge, upload};
use av_toolbox_cubek::retinex::{RecursiveGaussian, RecursivePlanes, recursive_gaussian};
use av_toolbox_cubek::warp::{BlurKind, blur_mask};
use av_toolbox_cubek::{DevicePlane, Edge};
use av_toolbox_gpu::{BuildContext, BuildTask};
use cubecl::prelude::{ComputeClient, Runtime};

use crate::support::{Sizing, backend, empty, plane, upload_f16, upload_f32, upload_u32};

const WIDTH: u32 = 1920;
const HEIGHT: u32 = 1080;
const HEAT: Duration = Duration::from_secs(1);
const BATCH: Duration = Duration::from_millis(150);
const BATCHES: usize = 20;

fn seeded_plane(samples: usize) -> Vec<f32> {
    let mut state = 0x2545_F491_4F6C_DD1Du64;

    (0..samples)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / (1u64 << 24) as f32
        })
        .collect()
}

/// Runs `launch` for `duration` by wall clock and returns how many launches it made.
fn run_for<R: Runtime>(
    client: &ComputeClient<R>,
    duration: Duration,
    launch: &mut impl FnMut(),
) -> (u32, Duration) {
    let start = Instant::now();
    let mut launches = 0u32;
    while start.elapsed() < duration {
        launch();
        launches += 1;
    }
    cubecl::future::block_on(client.sync()).expect("the device should finish");

    (launches, start.elapsed())
}

fn median_ms<R: Runtime>(client: &ComputeClient<R>, mut launch: impl FnMut()) -> f64 {
    run_for(client, HEAT, &mut launch);

    let mut per_launch: Vec<f64> = (0..BATCHES)
        .map(|_| {
            let (launches, elapsed) = run_for(client, BATCH, &mut launch);
            elapsed.as_secs_f64() * 1000.0 / f64::from(launches)
        })
        .collect();
    per_launch.sort_by(f64::total_cmp);

    (per_launch[BATCHES / 2 - 1] + per_launch[BATCHES / 2]) / 2.0
}

fn record<R: Runtime>(lines: &mut Vec<String>, client: &ComputeClient<R>, name: &str, launch: impl FnMut()) {
    let millis = median_ms(client, launch);
    lines.push(format!("{name} {millis:.4}"));
}

struct TimeRemainingBlurs;

impl BuildTask for TimeRemainingBlurs {
    type Output = Vec<String>;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let sizing = Sizing(build.frame_context());
        let samples = (WIDTH * HEIGHT) as usize;
        let bytes = samples * size_of::<f32>();
        let data = seeded_plane(samples);
        let input = upload_f32(client, &data);
        let first = empty(client, bytes);
        let second = empty(client, bytes);
        let third = empty(client, bytes);
        let fifth = empty(client, bytes);
        let sample_plane = |handle| plane(handle, WIDTH, HEIGHT);
        let mut lines = Vec::new();

        let smooth_chain = |steps: &[Smooth]| {
            let mut source: DevicePlane<'_> = sample_plane(&input);
            let targets = [sample_plane(&first), sample_plane(&second)];
            for (index, step) in steps.iter().enumerate() {
                let target = targets[index % 2];
                let planes = NeighbourPlanes {
                    input: source,
                    output: target,
                };
                neighbourhood::smooth(client, &sizing, planes, *step);
                source = target;
            }
        };

        // The blur leg of min_blur: Binomial, then Box for radius 2, then Box twice for radius 3.
        let min_blur_chains: [(&str, &[Smooth]); 3] = [
            ("min_blur_blur r1", &[Smooth::Binomial]),
            ("min_blur_blur r2", &[Smooth::Binomial, Smooth::Box]),
            ("min_blur_blur r3", &[Smooth::Binomial, Smooth::Box, Smooth::Box]),
        ];
        for (name, steps) in min_blur_chains {
            record(&mut lines, client, name, || smooth_chain(steps));
        }

        // The blur leg of min_blur and contra_sharpen as they run now, one BinomialChain launch.
        for radius in 1..=3u32 {
            let chain = BinomialChain::new(client, radius);
            let name = format!("binomial_chain r{radius} now");
            record(&mut lines, client, &name, || {
                let planes = NeighbourPlanes {
                    input: sample_plane(&input),
                    output: sample_plane(&first),
                };
                chain.apply(client, &sizing, planes);
            });
        }

        // The tiled gaussian, at sigma 0.8 for radius 1 and radius / 4 otherwise.
        let half_input = upload_f16(client, &data);
        let half_scratch = empty(client, samples * 2);
        let half_output = empty(client, samples * 2);
        let storages = [
            (Storage::F32, &input, &first, &second, "f32"),
            (Storage::F16, &half_input, &half_scratch, &half_output, "f16"),
        ];
        for radius in [1u32, 4, 7] {
            let sigma = if radius == 1 { 0.8 } else { radius as f32 / 4.0 };
            let gaussian = Gaussian::with_radius(client, sigma, radius);

            for (storage, source, scratch, target, label) in storages {
                let name = format!("gaussian r{radius} {label}");
                record(&mut lines, client, &name, || {
                    let planes = BlurPlanes {
                        input: sample_plane(source),
                        scratch: sample_plane(scratch),
                        output: sample_plane(target),
                    };
                    gaussian.apply_as(client, &sizing, planes, storage);
                });
            }
        }

        // The separable kernels each chain above and fine_dehalo's box_box add up to.
        let separable_kernels: [(&str, &[f32]); 4] = [
            ("separable [1,2,1]", &[1.0, 2.0, 1.0]),
            ("separable [1,3,4,3,1]", &[1.0, 3.0, 4.0, 3.0, 1.0]),
            (
                "separable [1,4,8,10,8,4,1]",
                &[1.0, 4.0, 8.0, 10.0, 8.0, 4.0, 1.0],
            ),
            ("separable [1,2,3,2,1]", &[1.0, 2.0, 3.0, 2.0, 1.0]),
        ];
        for (name, weights) in separable_kernels {
            let separable = Separable::new(client, weights);
            record(&mut lines, client, name, || {
                let planes = FilterPlanes {
                    input: sample_plane(&input),
                    output: sample_plane(&first),
                };
                separable.apply_as(client, &sizing, planes, Edge::Reflect, Storage::F32);
            });
        }

        // The blur chain of contra_sharpen after its median: Binomial, then sharp - 1 Box passes.
        for sharp in 1..=3usize {
            let mut steps = vec![Smooth::Binomial];
            steps.extend(vec![Smooth::Box; sharp - 1]);
            record(
                &mut lines,
                client,
                &format!("contra_sharpen_blur sharp{sharp}"),
                || {
                    smooth_chain(&steps);
                },
            );
        }

        record(&mut lines, client, "fine_dehalo box_box", || {
            smooth_chain(&[Smooth::Box, Smooth::Box]);
        });

        let shrink_blur = Separable::new(client, &[1.0, 2.0, 3.0, 2.0, 1.0]);
        record(&mut lines, client, "fine_dehalo box_box now", || {
            let planes = FilterPlanes {
                input: sample_plane(&input),
                output: sample_plane(&first),
            };
            shrink_blur.apply_as(client, &sizing, planes, Edge::Reflect, Storage::F32);
        });

        // sbr_kernel is private, so the public pieces before it are timed, then the whole pass.
        record(
            &mut lines,
            client,
            "vinverse_pre_sbr vertical121+difference+vertical121",
            || {
                let blur_input = NeighbourPlanes {
                    input: sample_plane(&input),
                    output: sample_plane(&first),
                };
                lineart::vertical_121(client, &sizing, blur_input);

                let comb = DifferencePlanes {
                    first: sample_plane(&input),
                    second: sample_plane(&first),
                    output: sample_plane(&second),
                };
                difference(client, &sizing, comb);

                let blur_comb = NeighbourPlanes {
                    input: sample_plane(&second),
                    output: sample_plane(&third),
                };
                lineart::vertical_121(client, &sizing, blur_comb);
            },
        );

        let params = VinverseParams {
            contra_str: 2.7,
            scl: 0.25,
            amnt: None,
        };
        record(&mut lines, client, "vinverse_whole now", || {
            let planes = VinversePlanes {
                input: sample_plane(&input),
                output: sample_plane(&first),
                blurred: sample_plane(&second),
                sbr: sample_plane(&fifth),
            };
            lineart::vinverse(client, &sizing, planes, &params);
        });

        let gauss_weights = |sharpness: f32| {
            let across = fmtc_gauss_taps(WIDTH, sharpness);
            let down = fmtc_gauss_taps(HEIGHT, sharpness);
            (upload(client, &across), upload(client, &down))
        };
        for (name, sharpness) in [
            ("qtgmc_blur_and_spread p2", 2.0f32),
            ("qtgmc_blur_and_spread p5", 5.0),
        ] {
            let (across, down) = gauss_weights(sharpness);
            record(&mut lines, client, name, || {
                let smoothing = NeighbourPlanes {
                    input: sample_plane(&input),
                    output: sample_plane(&first),
                };
                neighbourhood::smooth_with_edge(client, &sizing, smoothing, Smooth::Binomial, Edge::Clamp);

                let resizing = ResizePlanes {
                    input: sample_plane(&first),
                    scratch: sample_plane(&second),
                    output: sample_plane(&third),
                };
                apply_float_with_edge(client, &sizing, resizing, &across, &down, Edge::Clamp);
            });
        }

        for sigma in [50.0f32, 200.0, 350.0] {
            let filter = RecursiveGaussian::new(sigma);
            record(
                &mut lines,
                client,
                &format!("recursive_gaussian sigma{sigma}"),
                || {
                    let planes = RecursivePlanes {
                        input: sample_plane(&input),
                        output: sample_plane(&first),
                    };
                    recursive_gaussian(client, &sizing, planes, filter);
                },
            );
        }

        let mask_values: Vec<u32> = data.iter().map(|value| (value * 65535.0) as u32).collect();
        let mask = upload_u32(client, &mask_values);
        let mask_scratch = empty(client, bytes);
        let blur_passes = [
            ("blur_mask R2 1pass", BlurKind::R2, 1),
            ("blur_mask R6 1pass", BlurKind::R6, 1),
            ("blur_mask R6 3pass", BlurKind::R6, 3),
        ];
        for (name, kind, passes) in blur_passes {
            record(&mut lines, client, name, || {
                for _ in 0..passes {
                    blur_mask(
                        client,
                        &sizing,
                        sample_plane(&mask),
                        sample_plane(&mask_scratch),
                        kind,
                    );
                }
            });
        }

        lines
    }
}

#[test]
#[ignore]
fn time_remaining_blurs() {
    let lines = backend().build(TimeRemainingBlurs);

    for line in lines {
        println!("{line}");
    }
}
