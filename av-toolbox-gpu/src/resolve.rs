use cubecl::client::ComputeClient;
use cubecl::prelude::Runtime;

use crate::Device;

/// The variable CubeCL reads to replace wgpu's own pick of the default device.
const WGPU_OVERRIDE_ENV: &str = "CUBECL_WGPU_DEFAULT_DEVICE";

/// The CubeCL type id that lists every adapter, whatever its kind.
const EVERY_KIND: u16 = 4;

/// The device `CUBECL_WGPU_DEFAULT_DEVICE` names in place of wgpu's own pick, if it names one.
pub(crate) fn wgpu_override() -> Option<Device> {
    let value = std::env::var(WGPU_OVERRIDE_ENV).ok()?;
    parse_override(&value)
}

/// The device a wgpu client opened on [Device::Default] runs on.
///
/// `overridden` is what [wgpu_override] read when the client opened. Without one, wgpu picks its
/// high performance adapter. A device no selector can name resolves to [Device::Default], as does
/// an override naming a device the backend does not list, since CubeCL then falls back to an
/// adapter of unknown kind.
pub(crate) fn wgpu_default<R: Runtime>(client: &ComputeClient<R>, overridden: Option<Device>) -> Device {
    let type_ids: Vec<u16> = client
        .enumerate_devices(EVERY_KIND)
        .iter()
        .map(|id| id.type_id)
        .collect();

    match overridden {
        Some(device) => listed_or_default(device, &type_ids),
        None => high_performance(&type_ids),
    }
}

/// Reads an override in the spelling CubeCL takes, such as `DiscreteGpu(1)`.
fn parse_override(value: &str) -> Option<Device> {
    if value == "Cpu" {
        return Some(Device::Cpu);
    }

    let (kind, rest) = value.split_once('(')?;
    let digits = rest.strip_suffix(')')?;
    let index = digits.parse().ok()?;

    match kind {
        "DiscreteGpu" => Some(Device::Discrete { index }),
        "IntegratedGpu" => Some(Device::Integrated { index }),
        "VirtualGpu" => Some(Device::Virtual { index }),
        _ => None,
    }
}

/// `device` when one of the adapters in `type_ids` answers to it, otherwise [Device::Default].
fn listed_or_default(device: Device, type_ids: &[u16]) -> Device {
    let listed = named_devices(type_ids);

    if listed.contains(&device) {
        device
    } else {
        Device::Default
    }
}

/// The selector naming each adapter, counting each kind's adapters in the order they are listed.
fn named_devices(type_ids: &[u16]) -> Vec<Device> {
    let mut discrete = 0;
    let mut integrated = 0;
    let mut virtual_gpus = 0;
    let mut devices = Vec::new();

    for type_id in type_ids {
        let device = match type_id {
            0 => next_index(&mut discrete, |index| Device::Discrete { index }),
            1 => next_index(&mut integrated, |index| Device::Integrated { index }),
            2 => next_index(&mut virtual_gpus, |index| Device::Virtual { index }),
            3 => Device::Cpu,
            _ => continue,
        };

        devices.push(device);
    }

    devices
}

fn next_index(counter: &mut usize, name: impl FnOnce(usize) -> Device) -> Device {
    let index = *counter;
    *counter += 1;

    name(index)
}

/// The device wgpu picks when asked for high performance, given the kind of every adapter it lists.
///
/// wgpu takes the first adapter of the best ranked kind, in the order it lists them, which is
/// the same order the selectors count in.
fn high_performance(type_ids: &[u16]) -> Device {
    let best = type_ids.iter().copied().min_by_key(|type_id| rank(*type_id));

    match best {
        Some(0) => Device::Discrete { index: 0 },
        Some(1) => Device::Integrated { index: 0 },
        Some(2) => Device::Virtual { index: 0 },
        Some(3) => Device::Cpu,
        _ => Device::Default,
    }
}

/// How wgpu ranks a CubeCL type id for high performance, lowest first.
fn rank(type_id: u16) -> u8 {
    match type_id {
        0 => 0,
        1 => 1,
        2 => 3,
        3 => 4,
        _ => 2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_discrete_gpu_wins_over_every_other_kind() {
        assert_eq!(high_performance(&[3, 1, 0, 0]), Device::Discrete { index: 0 });
    }

    #[test]
    fn an_integrated_gpu_wins_without_a_discrete_one() {
        assert_eq!(high_performance(&[3, 2, 1]), Device::Integrated { index: 0 });
    }

    #[test]
    fn an_adapter_of_unknown_kind_cannot_be_named() {
        assert_eq!(high_performance(&[3, 2, 4]), Device::Default);
    }

    #[test]
    fn a_software_device_is_the_last_resort() {
        assert_eq!(high_performance(&[3]), Device::Cpu);
    }

    #[test]
    fn no_adapters_stays_default() {
        assert_eq!(high_performance(&[]), Device::Default);
    }

    #[test]
    fn each_kind_is_counted_on_its_own() {
        let named = named_devices(&[1, 0, 4, 0, 3]);

        assert_eq!(
            named,
            vec![
                Device::Integrated { index: 0 },
                Device::Discrete { index: 0 },
                Device::Discrete { index: 1 },
                Device::Cpu,
            ]
        );
    }

    #[test]
    fn a_listed_override_is_kept() {
        let device = Device::Discrete { index: 1 };

        assert_eq!(listed_or_default(device, &[0, 1, 0]), device);
    }

    #[test]
    fn an_override_the_backend_does_not_list_cannot_be_named() {
        let device = Device::Discrete { index: 0 };

        assert_eq!(listed_or_default(device, &[1, 4]), Device::Default);
    }

    #[test]
    fn an_override_reads_cubecl_spelling() {
        assert_eq!(
            parse_override("DiscreteGpu(1)"),
            Some(Device::Discrete { index: 1 })
        );
        assert_eq!(
            parse_override("IntegratedGpu(0)"),
            Some(Device::Integrated { index: 0 })
        );
        assert_eq!(
            parse_override("VirtualGpu(2)"),
            Some(Device::Virtual { index: 2 })
        );
        assert_eq!(parse_override("Cpu"), Some(Device::Cpu));
    }

    #[test]
    fn an_override_cubecl_cannot_read_is_ignored() {
        assert_eq!(parse_override("DiscreteGpu"), None);
        assert_eq!(parse_override("DiscreteGpu(x)"), None);
        assert_eq!(parse_override("Unicorn(0)"), None);
    }
}
