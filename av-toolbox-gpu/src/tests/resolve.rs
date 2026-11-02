use cubecl::device::MetalDevice;

use crate::Device;
use crate::resolve::{MetalGpus, MetalRefusal};

#[test]
fn metal_counts_discrete_gpus_before_integrated_ones() {
    let gpus = MetalGpus {
        discrete: 2,
        integrated: 1,
    };
    let cases = [
        (Device::Default, MetalDevice::DefaultDevice),
        (Device::Discrete { index: 0 }, MetalDevice::DiscreteGpu(0)),
        (Device::Discrete { index: 1 }, MetalDevice::DiscreteGpu(1)),
        (Device::Discrete { index: 2 }, MetalDevice::IntegratedGpu(0)),
    ];

    for (selector, expected) in cases {
        let mapped = gpus
            .device(selector)
            .expect("metal names the default and every index");
        assert_eq!(mapped, expected, "{selector}");
    }
}

#[test]
fn the_one_gpu_on_apple_silicon_is_discrete_0() {
    let gpus = MetalGpus {
        discrete: 0,
        integrated: 1,
    };

    let mapped = gpus.device(Device::Discrete { index: 0 });

    assert_eq!(mapped, Ok(MetalDevice::IntegratedGpu(0)));
}

#[test]
fn metal_refuses_an_index_past_its_gpus() {
    let gpus = MetalGpus {
        discrete: 0,
        integrated: 1,
    };

    let refused = gpus.device(Device::Discrete { index: 1 });

    assert_eq!(refused, Err(MetalRefusal::PastTheEnd { available: 1 }));
}

#[test]
fn metal_refuses_the_kinds_it_cannot_name() {
    let gpus = MetalGpus {
        discrete: 1,
        integrated: 1,
    };
    let unsupported = [
        Device::Integrated { index: 0 },
        Device::Virtual { index: 1 },
        Device::Cpu,
    ];

    for selector in unsupported {
        let refused = gpus.device(selector);
        assert_eq!(refused, Err(MetalRefusal::UnsupportedKind), "{selector}");
    }
}

#[test]
fn metal_lists_an_index_for_every_gpu() {
    let gpus = MetalGpus {
        discrete: 1,
        integrated: 1,
    };

    let listed = gpus.selectors();

    let expected = [Device::Discrete { index: 0 }, Device::Discrete { index: 1 }];
    assert_eq!(listed, expected);

    for selector in listed {
        let mapped = gpus.device(selector);
        assert!(mapped.is_ok(), "{selector} is listed but does not resolve");
    }
}

#[test]
fn a_machine_without_metal_gpus_lists_none() {
    let gpus = MetalGpus {
        discrete: 0,
        integrated: 0,
    };

    assert!(gpus.selectors().is_empty());
}

#[test]
fn the_metal_default_resolves_to_the_first_gpu_counted() {
    let mixed = MetalGpus {
        discrete: 1,
        integrated: 1,
    };
    let apple_silicon = MetalGpus {
        discrete: 0,
        integrated: 1,
    };

    assert_eq!(mixed.default_selector(), Device::Discrete { index: 0 });
    assert_eq!(apple_silicon.default_selector(), Device::Discrete { index: 0 });
}

#[test]
fn the_metal_default_stays_default_without_gpus() {
    let gpus = MetalGpus {
        discrete: 0,
        integrated: 0,
    };

    assert_eq!(gpus.default_selector(), Device::Default);
}
