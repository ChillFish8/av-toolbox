use crate::probe::probe_client;
#[cfg(feature = "metal")]
use crate::resolve::MetalGpus;
use crate::resolve::wgpu_selector;
use crate::{Accelerator, Device};

/// One selector of each kind [Device] can name, used to ask a backend for the devices of that kind.
const KIND_PROBES: [Device; 4] = [
    Device::Discrete { index: 0 },
    Device::Integrated { index: 0 },
    Device::Virtual { index: 0 },
    Device::Cpu,
];

/// What one backend reports about this machine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendDevices {
    /// The backend that was asked.
    pub accelerator: Accelerator,
    /// Whether the backend started at all.
    ///
    /// A build can enable a backend the machine has no driver for, and that backend reports no
    /// devices because it never ran rather than because the machine has no hardware.
    pub available: bool,
    /// The devices the backend can see, in the order it lists them.
    ///
    /// Always empty when `available` is false.
    pub devices: Vec<Device>,
}

/// Asks each backend in `accelerators` which devices it can see.
///
/// Backends are reported in the order given, including the ones that could not start, so a
/// caller can tell "no such hardware" apart from "no such driver".
pub fn enumerate_devices(accelerators: &[Accelerator]) -> Vec<BackendDevices> {
    accelerators
        .iter()
        .map(|accelerator| {
            let available = probe_client(*accelerator, Device::Default).is_some();
            let devices = if available {
                listed_devices(*accelerator)
            } else {
                Vec::new()
            };

            BackendDevices {
                accelerator: *accelerator,
                available,
                devices,
            }
        })
        .collect()
}

/// Every device `accelerator` lists that [Device] is able to name.
///
/// The backend is asked once per kind of device. A kind the backend cannot express is skipped.
/// Metal names every GPU by its index, so it lists one selector per GPU it counts.
fn listed_devices(accelerator: Accelerator) -> Vec<Device> {
    #[cfg(feature = "metal")]
    if accelerator == Accelerator::Metal {
        let gpus = MetalGpus::count();
        return gpus.selectors();
    }

    let mut listed = Vec::new();

    for probe in KIND_PROBES {
        let Ok(kind_device) = probe.to_cubecl(accelerator) else {
            continue;
        };

        let ids = cubecl::Device::enumerate(kind_device.to_id());
        listed.extend(ids.into_iter().map(cubecl::Device::from_id));
    }

    collect_devices(listed)
}

/// Maps CubeCL devices onto selectors, keeping the first of any repeat.
fn collect_devices(listed: impl IntoIterator<Item = cubecl::Device>) -> Vec<Device> {
    let mut devices: Vec<Device> = Vec::new();
    for cubecl_device in listed {
        if let Some(device) = to_device(&cubecl_device)
            && !devices.contains(&device)
        {
            devices.push(device);
        }
    }

    devices
}

/// Maps a CubeCL device onto the selector that names it.
///
/// A device of a kind this tool cannot select returns `None`.
fn to_device(cubecl_device: &cubecl::Device) -> Option<Device> {
    match cubecl_device {
        cubecl::Device::Cuda(cuda_device) => Some(Device::Discrete {
            index: cuda_device.index,
        }),
        cubecl::Device::Hip(amd_device) => Some(Device::Discrete {
            index: amd_device.index,
        }),
        cubecl::Device::Wgpu(wgpu_device) => wgpu_selector(&wgpu_device.kind),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use cubecl::device::{CudaDevice, WgpuBackend, WgpuDevice, WgpuDeviceKind};

    use super::*;

    fn vulkan(kind: WgpuDeviceKind) -> cubecl::Device {
        let wgpu_device = WgpuDevice::new(kind).on(WgpuBackend::Vulkan);
        cubecl::Device::Wgpu(wgpu_device)
    }

    fn cuda(index: usize) -> cubecl::Device {
        cubecl::Device::Cuda(CudaDevice::new(index))
    }

    #[test]
    fn a_wgpu_kind_maps_onto_the_selector_that_names_it() {
        assert_eq!(
            to_device(&vulkan(WgpuDeviceKind::DiscreteGpu(1))),
            Some(Device::Discrete { index: 1 })
        );
        assert_eq!(
            to_device(&vulkan(WgpuDeviceKind::IntegratedGpu(0))),
            Some(Device::Integrated { index: 0 })
        );
        assert_eq!(
            to_device(&vulkan(WgpuDeviceKind::VirtualGpu(2))),
            Some(Device::Virtual { index: 2 })
        );
        assert_eq!(to_device(&vulkan(WgpuDeviceKind::Cpu)), Some(Device::Cpu));
    }

    #[test]
    fn a_cuda_device_maps_onto_a_discrete_selector() {
        assert_eq!(to_device(&cuda(3)), Some(Device::Discrete { index: 3 }));
    }

    #[test]
    fn an_unknown_kind_is_skipped() {
        assert_eq!(to_device(&vulkan(WgpuDeviceKind::Other(0))), None);
        assert_eq!(to_device(&vulkan(WgpuDeviceKind::DefaultDevice)), None);
    }

    #[test]
    fn a_device_reported_once_per_kind_is_kept_once() {
        let repeated = [cuda(0), cuda(1), cuda(0), cuda(1)];

        let devices = collect_devices(repeated);
        assert_eq!(
            devices,
            vec![Device::Discrete { index: 0 }, Device::Discrete { index: 1 }]
        );
    }

    #[test]
    fn distinct_kinds_at_the_same_index_are_all_kept() {
        let mixed = [
            vulkan(WgpuDeviceKind::DiscreteGpu(0)),
            vulkan(WgpuDeviceKind::IntegratedGpu(0)),
            vulkan(WgpuDeviceKind::VirtualGpu(0)),
        ];

        let devices = collect_devices(mixed);
        assert_eq!(
            devices,
            vec![
                Device::Discrete { index: 0 },
                Device::Integrated { index: 0 },
                Device::Virtual { index: 0 },
            ],
        );
    }

    #[test]
    fn an_unknown_kind_is_dropped_rather_than_guessed_at() {
        let unknown = [
            vulkan(WgpuDeviceKind::Other(0)),
            vulkan(WgpuDeviceKind::DiscreteGpu(0)),
        ];

        let devices = collect_devices(unknown);
        assert_eq!(devices, vec![Device::Discrete { index: 0 }]);
    }

    #[test]
    fn asking_no_backends_lists_nothing() {
        assert!(enumerate_devices(&[]).is_empty());
    }

    #[cfg(all(feature = "gpu-tests", any(feature = "vulkan", feature = "metal")))]
    #[test]
    fn the_built_backend_lists_at_least_one_device_without_repeating_any() {
        use crate::testing::TEST_ACCELERATOR;

        let reported = enumerate_devices(&[TEST_ACCELERATOR]);
        assert_eq!(reported.len(), 1);

        let backend = &reported[0];
        assert_eq!(backend.accelerator, TEST_ACCELERATOR);
        assert!(backend.available, "the {TEST_ACCELERATOR} backend did not start");
        assert!(
            !backend.devices.is_empty(),
            "the {TEST_ACCELERATOR} backend listed no devices"
        );

        let mut unique = backend.devices.clone();
        unique.sort_by_key(|device| device.to_string());
        unique.dedup();
        assert_eq!(
            unique.len(),
            backend.devices.len(),
            "a device was listed more than once: {:?}",
            backend.devices,
        );
    }
}
