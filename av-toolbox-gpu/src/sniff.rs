use cubecl::client::ComputeClient;
use cubecl::prelude::Runtime;

use crate::probe::{RuntimeTask, with_runtime};
use crate::{Accelerator, Device, default_accelerators};

/// Returns the first accelerator in `accelerators` that really starts on `device`.
///
/// A build can enable several backends and only some of them will run on any given machine, so
/// a caller lists a fast backend followed by a safe fallback and takes what it is given.
///
/// The probe runs on `device` rather than on the backend's default one. Opening a client is
/// what proves a backend works, and opening it on a card the caller did not choose both tests
/// the wrong hardware and pays that card's first-time driver initialisation.
pub fn sniff_best_accelerator(accelerators: &[Accelerator], device: Device) -> Option<Accelerator> {
    accelerators
        .iter()
        .copied()
        .find(|accelerator| with_runtime(*accelerator, device, ProbeOnly).is_some())
}

/// Picks the best backend this build enables that really runs on `device`.
///
/// This is the whole of automatic selection: every backend the build could use, narrowed to the
/// ones whose driver is actually present and working. A backend whose libraries are missing
/// panics rather than returning an error, so the probe catches that and skips it instead of
/// letting it take the process down.
pub fn auto_select_accelerator(device: Device) -> Option<Accelerator> {
    let accelerators = default_accelerators();

    sniff_best_accelerator(&accelerators, device)
}

/// Opens a client and does nothing with it, so that opening it is the whole test.
struct ProbeOnly;

impl RuntimeTask for ProbeOnly {
    type Output = ();

    fn run<R: Runtime>(self, _client: &ComputeClient<R>) -> Self::Output {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn asking_no_backends_finds_nothing() {
        assert_eq!(sniff_best_accelerator(&[], Device::Default), None);
    }

    #[cfg(all(feature = "gpu-tests", any(feature = "vulkan", feature = "metal")))]
    #[test]
    fn the_built_backend_is_found_when_it_is_the_only_one_offered() {
        use crate::testing::TEST_ACCELERATOR;

        let found = sniff_best_accelerator(&[TEST_ACCELERATOR], Device::Default);
        assert_eq!(found, Some(TEST_ACCELERATOR));
    }

    /// Every selector maps onto some wgpu device, so what fails here is opening a client for
    /// hardware that is not present, not the conversion.
    #[cfg(all(feature = "gpu-tests", any(feature = "vulkan", feature = "metal")))]
    #[test]
    fn a_device_the_backend_does_not_have_is_not_chosen() {
        use crate::testing::TEST_ACCELERATOR;

        let found = sniff_best_accelerator(&[TEST_ACCELERATOR], Device::Discrete { index: 99 });
        assert_eq!(found, None);
    }

    #[cfg(all(feature = "gpu-tests", any(feature = "vulkan", feature = "metal")))]
    #[test]
    fn auto_selection_finds_a_backend_on_this_machine() {
        assert!(auto_select_accelerator(Device::Default).is_some());
    }
}
