#[cfg(any(feature = "metal", test))]
use cubecl::device::MetalDevice;
use cubecl::device::{WgpuBackend, WgpuDevice, WgpuDeviceKind};

use crate::Device;

/// The variable CubeCL reads to replace wgpu's own pick of the default device.
const WGPU_OVERRIDE_ENV: &str = "CUBECL_WGPU_DEFAULT_DEVICE";

/// One adapter of each kind wgpu reports, used to ask for every adapter of that kind.
const KIND_PROBES: [WgpuDeviceKind; 5] = [
    WgpuDeviceKind::DiscreteGpu(0),
    WgpuDeviceKind::IntegratedGpu(0),
    WgpuDeviceKind::VirtualGpu(0),
    WgpuDeviceKind::Cpu,
    WgpuDeviceKind::Other(0),
];

/// The device `CUBECL_WGPU_DEFAULT_DEVICE` names in place of wgpu's own pick, if it names one.
pub(crate) fn wgpu_override() -> Option<Device> {
    let value = std::env::var(WGPU_OVERRIDE_ENV).ok()?;
    parse_override(&value)
}

/// The device a wgpu client opened on [Device::Default] through `backend` runs on.
///
/// `overridden` is what [wgpu_override] read when the client opened. Without one, wgpu picks its
/// high performance adapter. A device no selector can name resolves to [Device::Default], as does
/// an override naming a device the backend does not list, since CubeCL then falls back to an
/// adapter of unknown kind.
pub(crate) fn wgpu_default(backend: WgpuBackend, overridden: Option<Device>) -> Device {
    let kinds = listed_kinds(backend);

    match overridden {
        Some(device) => listed_or_default(device, &kinds),
        None => high_performance(&kinds),
    }
}

/// The selector naming a wgpu adapter of `kind`, if a selector can name it.
pub(crate) fn wgpu_selector(kind: &WgpuDeviceKind) -> Option<Device> {
    match *kind {
        WgpuDeviceKind::DiscreteGpu(index) => Some(Device::Discrete { index }),
        WgpuDeviceKind::IntegratedGpu(index) => Some(Device::Integrated { index }),
        WgpuDeviceKind::VirtualGpu(index) => Some(Device::Virtual { index }),
        WgpuDeviceKind::Cpu => Some(Device::Cpu),
        _ => None,
    }
}

/// Every adapter `backend` reaches, grouped by kind and in the order wgpu lists each kind.
fn listed_kinds(backend: WgpuBackend) -> Vec<WgpuDeviceKind> {
    let mut kinds = Vec::new();

    for probe in KIND_PROBES {
        let probe_device = WgpuDevice::new(probe).on(backend);
        let probe_id = cubecl::Device::Wgpu(probe_device).to_id();
        let ids = cubecl::Device::enumerate(probe_id);

        for id in ids {
            if let cubecl::Device::Wgpu(listed) = cubecl::Device::from_id(id) {
                kinds.push(listed.kind);
            }
        }
    }

    kinds
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

/// `device` when one of the adapters in `kinds` answers to it, otherwise [Device::Default].
fn listed_or_default(device: Device, kinds: &[WgpuDeviceKind]) -> Device {
    let listed = kinds.iter().any(|kind| wgpu_selector(kind) == Some(device));

    if listed { device } else { Device::Default }
}

/// The device wgpu picks when asked for high performance, given every adapter it lists.
///
/// wgpu takes the first adapter of the best ranked kind, in the order it lists them, which is
/// the same order the selectors count in.
fn high_performance(kinds: &[WgpuDeviceKind]) -> Device {
    let best = kinds.iter().min_by_key(|kind| rank(kind));
    let selector = best.and_then(wgpu_selector);

    selector.unwrap_or(Device::Default)
}

/// How wgpu ranks an adapter kind for high performance, lowest first.
fn rank(kind: &WgpuDeviceKind) -> u8 {
    match kind {
        WgpuDeviceKind::DiscreteGpu(_) => 0,
        WgpuDeviceKind::IntegratedGpu(_) => 1,
        WgpuDeviceKind::VirtualGpu(_) => 3,
        WgpuDeviceKind::Cpu => 4,
        _ => 2,
    }
}

/// How many GPUs of each kind the native Metal runtime sees.
///
/// A selector index counts the discrete GPUs first and the integrated ones after, so `discrete:0`
/// is the first GPU on every Mac.
#[cfg(any(feature = "metal", test))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MetalGpus {
    pub(crate) discrete: usize,
    pub(crate) integrated: usize,
}

/// Why a selector names no Metal GPU.
#[cfg(any(feature = "metal", test))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MetalRefusal {
    /// Metal names its GPUs only by `default` and `discrete[:N]`.
    UnsupportedKind,
    /// The index is past the last of the `available` GPUs.
    PastTheEnd { available: usize },
}

#[cfg(any(feature = "metal", test))]
impl MetalGpus {
    /// Counts the GPUs Metal lists on this machine.
    #[cfg(feature = "metal")]
    pub(crate) fn count() -> Self {
        let discrete = count_metal_gpus(MetalDevice::DiscreteGpu(0));
        let integrated = count_metal_gpus(MetalDevice::IntegratedGpu(0));

        Self { discrete, integrated }
    }

    pub(crate) fn total(self) -> usize {
        self.discrete + self.integrated
    }

    /// The selector of every GPU, in the order they are counted.
    pub(crate) fn selectors(self) -> Vec<Device> {
        (0..self.total())
            .map(|index| Device::Discrete { index })
            .collect()
    }

    /// The Metal device `selector` names.
    pub(crate) fn device(self, selector: Device) -> Result<MetalDevice, MetalRefusal> {
        match selector {
            Device::Default => Ok(MetalDevice::DefaultDevice),
            Device::Discrete { index } if index < self.discrete => Ok(MetalDevice::DiscreteGpu(index)),
            Device::Discrete { index } if index < self.total() => {
                let integrated_index = index - self.discrete;
                Ok(MetalDevice::IntegratedGpu(integrated_index))
            },
            Device::Discrete { .. } => Err(MetalRefusal::PastTheEnd {
                available: self.total(),
            }),
            Device::Integrated { .. } | Device::Virtual { .. } | Device::Cpu => {
                Err(MetalRefusal::UnsupportedKind)
            },
        }
    }

    /// The selector naming the GPU Metal picks as its default.
    ///
    /// This is the right label in the usual case, where Metal's default is the first GPU counted.
    /// With no GPU listed there is nothing to name, so it stays [Device::Default].
    pub(crate) fn default_selector(self) -> Device {
        if self.total() == 0 {
            return Device::Default;
        }

        Device::Discrete { index: 0 }
    }
}

/// How many Metal GPUs share the kind of `first`.
#[cfg(feature = "metal")]
fn count_metal_gpus(first: MetalDevice) -> usize {
    let device = cubecl::Device::Metal(first);
    let device_id = device.to_id();
    let listed = cubecl::Device::enumerate(device_id);

    listed.len()
}

#[cfg(test)]
mod tests {
    use WgpuDeviceKind::{Cpu, DiscreteGpu, IntegratedGpu, Other, VirtualGpu};

    use super::*;

    #[test]
    fn a_discrete_gpu_wins_over_every_other_kind() {
        let kinds = [Cpu, IntegratedGpu(0), DiscreteGpu(0), DiscreteGpu(1)];

        assert_eq!(high_performance(&kinds), Device::Discrete { index: 0 });
    }

    #[test]
    fn an_integrated_gpu_wins_without_a_discrete_one() {
        let kinds = [Cpu, VirtualGpu(0), IntegratedGpu(0)];

        assert_eq!(high_performance(&kinds), Device::Integrated { index: 0 });
    }

    #[test]
    fn an_adapter_of_unknown_kind_cannot_be_named() {
        let kinds = [Cpu, VirtualGpu(0), Other(0)];

        assert_eq!(high_performance(&kinds), Device::Default);
    }

    #[test]
    fn a_software_device_is_the_last_resort() {
        assert_eq!(high_performance(&[Cpu]), Device::Cpu);
    }

    #[test]
    fn no_adapters_stays_default() {
        assert_eq!(high_performance(&[]), Device::Default);
    }

    #[test]
    fn every_nameable_kind_keeps_its_index() {
        assert_eq!(
            wgpu_selector(&DiscreteGpu(1)),
            Some(Device::Discrete { index: 1 })
        );
        assert_eq!(
            wgpu_selector(&IntegratedGpu(0)),
            Some(Device::Integrated { index: 0 })
        );
        assert_eq!(wgpu_selector(&VirtualGpu(2)), Some(Device::Virtual { index: 2 }));
        assert_eq!(wgpu_selector(&Cpu), Some(Device::Cpu));
        assert_eq!(wgpu_selector(&Other(0)), None);
    }

    #[test]
    fn a_listed_override_is_kept() {
        let device = Device::Discrete { index: 1 };
        let kinds = [DiscreteGpu(0), IntegratedGpu(0), DiscreteGpu(1)];

        assert_eq!(listed_or_default(device, &kinds), device);
    }

    #[test]
    fn an_override_the_backend_does_not_list_cannot_be_named() {
        let device = Device::Discrete { index: 0 };
        let kinds = [IntegratedGpu(0), Other(0)];

        assert_eq!(listed_or_default(device, &kinds), Device::Default);
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
