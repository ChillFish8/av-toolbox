use std::time::{Duration, Instant};

use av_toolbox_cubek::blur::{BlurPlanes, Gaussian, Storage};
use av_toolbox_cubek::lineart;
use av_toolbox_cubek::neighbourhood::{self, NeighbourPlanes, Smooth};
use av_toolbox_gpu::{BuildContext, BuildTask};
use cubecl::prelude::{ComputeClient, Runtime};

use crate::support::{Sizing, backend, empty, plane, upload_f16, upload_f32};

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

struct TimeKernels;

impl BuildTask for TimeKernels {
    type Output = Vec<String>;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let sizing = Sizing(build.frame_context());
        let samples = (WIDTH * HEIGHT) as usize;
        let bytes = samples * size_of::<f32>();
        let data = seeded_plane(samples);
        let input = upload_f32(client, &data);
        let scratch = empty(client, bytes);
        let output = empty(client, bytes);
        let mut lines = Vec::new();

        let half_bytes = samples * Storage::F16.bytes();
        let half_input = upload_f16(client, &data);
        let half_scratch = empty(client, half_bytes);
        let half_output = empty(client, half_bytes);

        let radii = [
            (8u32, 2.0f32),
            (12, 3.0),
            (15, 3.75),
            (16, 4.0),
            (17, 4.25),
            (20, 5.0),
            (24, 6.0),
            (32, 8.0),
            (48, 12.0),
            (64, 16.0),
            (100, 25.0),
            (128, 32.0),
            (193, 48.25),
        ];
        let storages = [
            (Storage::F32, "f32", &input, &scratch, &output),
            (Storage::F16, "f16", &half_input, &half_scratch, &half_output),
        ];
        for (storage, storage_name, input, scratch, output) in storages {
            for (radius, sigma) in radii {
                let gaussian = Gaussian::with_radius(client, sigma, radius);
                let launch = || {
                    let planes = BlurPlanes {
                        input: plane(input, WIDTH, HEIGHT),
                        scratch: plane(scratch, WIDTH, HEIGHT),
                        output: plane(output, WIDTH, HEIGHT),
                    };
                    gaussian.apply_as(client, &sizing, planes, storage);
                };
                let millis = median_ms(client, launch);
                lines.push(format!("gaussian {radius} {storage_name} {millis:.4}"));
            }
        }

        for (name, smooth) in [("smooth_binomial", Smooth::Binomial), ("smooth_box", Smooth::Box)] {
            let launch = || {
                let planes = NeighbourPlanes {
                    input: plane(&input, WIDTH, HEIGHT),
                    output: plane(&output, WIDTH, HEIGHT),
                };
                neighbourhood::smooth(client, &sizing, planes, smooth);
            };
            let millis = median_ms(client, launch);
            lines.push(format!("{name} 1 f32 {millis:.4}"));
        }

        let launch = || {
            let planes = NeighbourPlanes {
                input: plane(&input, WIDTH, HEIGHT),
                output: plane(&output, WIDTH, HEIGHT),
            };
            lineart::vertical_121(client, &sizing, planes);
        };
        let millis = median_ms(client, launch);
        lines.push(format!("vertical_121 1 f32 {millis:.4}"));

        lines
    }
}

#[test]
#[ignore]
fn time_kernels() {
    let lines = backend().build(TimeKernels);

    for line in lines {
        println!("{line}");
    }
}
