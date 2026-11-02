use av_toolbox_cubek::cnn::{
    Region,
    convolve_forced_past_the_shared_memory_limit,
    forced_shared_bytes,
    offers_cmma_16,
};
use av_toolbox_gpu::{BuildContext, BuildTask};

use crate::support::backend;

const REGION: Region = Region {
    x: 0,
    y: 0,
    width: 128,
    height: 64,
};

/// What launching the forced blueprint past the shared memory limit gave, at one width.
struct ForcedLaunch {
    features: u32,
}

struct Launched {
    result: Result<bool, String>,
    max_shared_memory: usize,
    hardware_takes_it: bool,
}

impl BuildTask for ForcedLaunch {
    type Output = Launched;

    fn run(self, build: &BuildContext) -> Self::Output {
        let client = build.client();
        let hardware = &client.properties().hardware;
        let hardware_takes_it = hardware.plane_size_max == 64 && offers_cmma_16(client);
        let launched = convolve_forced_past_the_shared_memory_limit(client, self.features, REGION);
        let result = launched.map_err(|error| error.to_string());

        Launched {
            result,
            max_shared_memory: hardware.max_shared_memory_size,
            hardware_takes_it,
        }
    }
}

#[test]
fn the_forced_blueprint_launches_exactly_when_its_shared_memory_fits_the_device() {
    for features in [16u32, 32, 48, 64, 96] {
        let launched = backend().build(ForcedLaunch { features });
        let bytes = forced_shared_bytes(features as usize);
        println!(
            "{features} features: {bytes} bytes against {}",
            launched.max_shared_memory
        );

        if !launched.hardware_takes_it {
            continue;
        }

        if bytes <= launched.max_shared_memory {
            let ran = launched.result.expect("a blueprint that fits launches");
            assert!(ran, "the forced blueprint did not run at {features} features");
            continue;
        }

        let message = launched
            .result
            .expect_err("a blueprint past the limit fails to launch");
        let charged = format!("kernel requests {bytes} bytes");
        assert!(
            message.contains("Shared memory budget exceeded") && message.contains(&charged),
            "{features} features: expected cubek to charge {bytes} bytes, got {message}"
        );
    }
}
