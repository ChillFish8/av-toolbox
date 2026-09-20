use cubecl::client::ComputeClient;
use cubecl::device::DeviceId;
use cubecl::prelude::Runtime;

use crate::probe::{RuntimeTask, with_runtime};
use crate::{Accelerator, Device};

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
            let devices = with_runtime(*accelerator, Device::Default, ListDevices);

            BackendDevices {
                accelerator: *accelerator,
                available: devices.is_some(),
                devices: devices.unwrap_or_default(),
            }
        })
        .collect()
}

/// Lists every device a client can see that [Device] is able to name.
struct ListDevices;

impl RuntimeTask for ListDevices {
    type Output = Vec<Device>;

    fn run<R: Runtime>(self, client: &ComputeClient<R>) -> Self::Output {
        // Type ids 0..=3 are the device kinds `Device` can name, and anything else is hardware
        // this tool cannot select.
        let ids = (0..=3).flat_map(|type_id| client.enumerate_devices(type_id));

        collect_devices(ids)
    }
}

/// Maps CubeCL device ids onto selectors, keeping the first of any repeat.
///
/// Not every backend filters by the type id it is given. CUDA and ROCm report their whole device
/// list on every pass, so the same device arrives once per type id queried.
fn collect_devices(ids: impl Iterator<Item = DeviceId>) -> Vec<Device> {
    let mut devices: Vec<Device> = Vec::new();
    for id in ids {
        if let Some(device) = to_device(id)
            && !devices.contains(&device)
        {
            devices.push(device);
        }
    }

    devices
}

/// Maps a CubeCL device id onto the selector that names it.
///
/// The type ids come from CubeCL's own ordering of device kinds. A backend reporting a kind
/// this tool cannot select returns `None`.
fn to_device(id: DeviceId) -> Option<Device> {
    let index = id.index_id as usize;
    match id.type_id {
        0 => Some(Device::Discrete { index }),
        1 => Some(Device::Integrated { index }),
        2 => Some(Device::Virtual { index }),
        3 => Some(Device::Cpu),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_type_id_maps_onto_the_device_kind_that_names_it() {
        assert_eq!(
            to_device(DeviceId::new(0, 1)),
            Some(Device::Discrete { index: 1 })
        );
        assert_eq!(
            to_device(DeviceId::new(1, 0)),
            Some(Device::Integrated { index: 0 })
        );
        assert_eq!(to_device(DeviceId::new(2, 2)), Some(Device::Virtual { index: 2 }));
        assert_eq!(to_device(DeviceId::new(3, 0)), Some(Device::Cpu));
    }

    #[test]
    fn an_unknown_type_id_is_skipped() {
        assert_eq!(to_device(DeviceId::new(4, 0)), None);
    }

    #[test]
    fn a_device_reported_once_per_type_id_is_kept_once() {
        let repeated = [
            DeviceId::new(0, 0),
            DeviceId::new(0, 1),
            DeviceId::new(0, 0),
            DeviceId::new(0, 1),
        ];

        let devices = collect_devices(repeated.into_iter());
        assert_eq!(
            devices,
            vec![Device::Discrete { index: 0 }, Device::Discrete { index: 1 }]
        );
    }

    #[test]
    fn distinct_kinds_at_the_same_index_are_all_kept() {
        let mixed = [DeviceId::new(0, 0), DeviceId::new(1, 0), DeviceId::new(2, 0)];

        let devices = collect_devices(mixed.into_iter());
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
    fn an_unknown_type_id_is_dropped_rather_than_guessed_at() {
        let unknown = [DeviceId::new(4, 0), DeviceId::new(0, 0)];

        let devices = collect_devices(unknown.into_iter());
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
