use std::fmt;
use std::str::FromStr;

#[cfg(feature = "rocm")]
use cubecl::device::AmdDevice;
#[cfg(feature = "cuda")]
use cubecl::device::CudaDevice;
#[cfg(feature = "vulkan")]
use cubecl::device::{WgpuBackend, WgpuDevice, WgpuDeviceKind};

use crate::Accelerator;
#[cfg(feature = "metal")]
use crate::resolve::{MetalGpus, MetalRefusal};

/// Where to run the compute.
///
/// Each variant maps onto the concrete device type of whichever CubeCL runtime was selected.
/// Not every variant exists on every backend, so a conversion that cannot be honoured returns
/// [DeviceError].
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Device {
    /// The device the backend picks for itself.
    #[default]
    Default,
    /// Discrete GPU at ordinal `index`.
    ///
    /// On Metal the index counts the discrete GPUs first and the integrated ones after.
    Discrete { index: usize },
    /// Integrated GPU at ordinal `index`. Vulkan only.
    Integrated { index: usize },
    /// Virtual GPU at ordinal `index`. Vulkan only.
    Virtual { index: usize },
    /// The software device, such as lavapipe under Vulkan.
    Cpu,
}

/// A device selector a backend cannot express.
#[derive(Debug, thiserror::Error)]
pub enum DeviceError {
    #[error("device '{device}' is not supported on the {accelerator} backend, use {supported}")]
    Unsupported {
        device: Device,
        accelerator: Accelerator,
        supported: &'static str,
    },
    #[error(
        "device '{device}' is not on this machine, the {accelerator} backend sees {available} {noun}",
        noun = gpu_noun(*.available)
    )]
    Missing {
        device: Device,
        accelerator: Accelerator,
        available: usize,
    },
}

fn gpu_noun(count: usize) -> &'static str {
    match count {
        1 => "GPU",
        _ => "GPUs",
    }
}

/// A device selector that could not be read.
#[derive(Debug, thiserror::Error)]
pub enum ParseDeviceError {
    #[error("device kind '{kind}' takes no index, only discrete, integrated and virtual take one")]
    UnexpectedIndex { kind: String },
    #[error("invalid device index '{index}'")]
    InvalidIndex { index: String },
    #[error(
        "unknown device kind '{kind}', expected default, discrete[:N], integrated[:N], virtual[:N] or cpu"
    )]
    UnknownKind { kind: String },
}

impl FromStr for Device {
    type Err = ParseDeviceError;

    /// Reads a selector, which is a kind and an optional `:N` ordinal.
    ///
    /// `default` and `cpu` take no ordinal. `discrete`, `integrated` and `virtual` take one and
    /// default it to zero.
    fn from_str(text: &str) -> Result<Self, Self::Err> {
        let (kind, index) = match text.split_once(':') {
            Some((kind, index)) => (kind, Some(index)),
            None => (text, None),
        };

        if matches!(kind, "default" | "cpu") && index.is_some() {
            return Err(ParseDeviceError::UnexpectedIndex {
                kind: kind.to_string(),
            });
        }

        let raw_index = index.unwrap_or("0");
        let parse_index = || -> Result<usize, ParseDeviceError> {
            raw_index.parse().map_err(|_| ParseDeviceError::InvalidIndex {
                index: raw_index.to_string(),
            })
        };

        match kind {
            "default" => Ok(Self::Default),
            "cpu" => Ok(Self::Cpu),
            "discrete" => Ok(Self::Discrete {
                index: parse_index()?,
            }),
            "integrated" => Ok(Self::Integrated {
                index: parse_index()?,
            }),
            "virtual" => Ok(Self::Virtual {
                index: parse_index()?,
            }),
            other => Err(ParseDeviceError::UnknownKind {
                kind: other.to_string(),
            }),
        }
    }
}

impl fmt::Display for Device {
    /// Writes the selector spelling [FromStr] accepts, so a device prints as `discrete:1`
    /// rather than as its variant name.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Default => formatter.write_str("default"),
            Self::Discrete { index } => write!(formatter, "discrete:{index}"),
            Self::Integrated { index } => write!(formatter, "integrated:{index}"),
            Self::Virtual { index } => write!(formatter, "virtual:{index}"),
            Self::Cpu => formatter.write_str("cpu"),
        }
    }
}

impl Device {
    /// The CubeCL device this selector names on `accelerator`.
    ///
    /// Only Metal asks the driver, to count its GPUs. On every other backend whether the device
    /// exists is only known once a client opens on it.
    pub fn to_cubecl(self, accelerator: Accelerator) -> Result<cubecl::Device, DeviceError> {
        match accelerator {
            #[cfg(feature = "cuda")]
            Accelerator::Cuda => {
                let index = self.gpu_index(accelerator)?;
                let cuda_device = CudaDevice::new(index);
                Ok(cubecl::Device::Cuda(cuda_device))
            },
            #[cfg(feature = "rocm")]
            Accelerator::Rocm => {
                let index = self.gpu_index(accelerator)?;
                let amd_device = AmdDevice::new(index);
                Ok(cubecl::Device::Hip(amd_device))
            },
            #[cfg(feature = "vulkan")]
            Accelerator::Vulkan => {
                let kind = self.wgpu_kind();
                let vulkan_device = WgpuDevice::new(kind).on(WgpuBackend::Vulkan);
                Ok(cubecl::Device::Wgpu(vulkan_device))
            },
            #[cfg(feature = "metal")]
            Accelerator::Metal => {
                let gpus = MetalGpus::count();

                match gpus.device(self) {
                    Ok(metal_device) => Ok(cubecl::Device::Metal(metal_device)),
                    Err(refusal) => Err(self.metal_error(refusal)),
                }
            },
            // docs.rs widens the `Accelerator` variants behind `cfg(docsrs)` so they all appear in
            // the rendered enum, even when the matching backend feature is off. This arm keeps the
            // match exhaustive there and is never reached at runtime.
            #[cfg(docsrs)]
            _ => unreachable!(),
        }
    }

    /// The GPU ordinal on a backend that numbers its devices from zero and has no other kinds.
    #[cfg(any(feature = "cuda", feature = "rocm"))]
    fn gpu_index(self, accelerator: Accelerator) -> Result<usize, DeviceError> {
        match self {
            Self::Default => Ok(0),
            Self::Discrete { index } => Ok(index),
            other => Err(DeviceError::Unsupported {
                device: other,
                accelerator,
                supported: "default or discrete[:N]",
            }),
        }
    }

    #[cfg(feature = "metal")]
    fn metal_error(self, refusal: MetalRefusal) -> DeviceError {
        let accelerator = Accelerator::Metal;

        match refusal {
            MetalRefusal::UnsupportedKind => DeviceError::Unsupported {
                device: self,
                accelerator,
                supported: "default or discrete[:N]",
            },
            MetalRefusal::PastTheEnd { available } => DeviceError::Missing {
                device: self,
                accelerator,
                available,
            },
        }
    }

    #[cfg(feature = "vulkan")]
    fn wgpu_kind(self) -> WgpuDeviceKind {
        match self {
            Self::Default => WgpuDeviceKind::DefaultDevice,
            Self::Discrete { index } => WgpuDeviceKind::DiscreteGpu(index),
            Self::Integrated { index } => WgpuDeviceKind::IntegratedGpu(index),
            Self::Virtual { index } => WgpuDeviceKind::VirtualGpu(index),
            Self::Cpu => WgpuDeviceKind::Cpu,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_kind_without_an_index_parses() {
        assert_eq!(
            "default".parse::<Device>().expect("should parse"),
            Device::Default
        );
        assert_eq!("cpu".parse::<Device>().expect("should parse"), Device::Cpu);
    }

    #[test]
    fn an_indexed_kind_defaults_to_zero() {
        let parsed = "discrete".parse::<Device>().expect("should parse");
        assert_eq!(parsed, Device::Discrete { index: 0 });
    }

    #[test]
    fn an_indexed_kind_takes_its_index() {
        let parsed = "discrete:3".parse::<Device>().expect("should parse");
        assert_eq!(parsed, Device::Discrete { index: 3 });

        let parsed = "integrated:1".parse::<Device>().expect("should parse");
        assert_eq!(parsed, Device::Integrated { index: 1 });

        let parsed = "virtual:2".parse::<Device>().expect("should parse");
        assert_eq!(parsed, Device::Virtual { index: 2 });
    }

    #[test]
    fn parsing_rejects_an_unknown_kind() {
        let error = "unicorn".parse::<Device>().expect_err("should reject");
        assert!(matches!(error, ParseDeviceError::UnknownKind { .. }));
    }

    #[test]
    fn parsing_rejects_a_non_numeric_index() {
        let error = "discrete:abc".parse::<Device>().expect_err("should reject");
        assert!(matches!(error, ParseDeviceError::InvalidIndex { .. }));
    }

    #[test]
    fn parsing_rejects_an_index_on_a_kind_that_takes_none() {
        let error = "default:1".parse::<Device>().expect_err("should reject");
        assert!(matches!(error, ParseDeviceError::UnexpectedIndex { .. }));
        assert!(error.to_string().contains("default"), "{error}");

        let error = "cpu:2".parse::<Device>().expect_err("should reject");
        assert!(error.to_string().contains("cpu"), "{error}");
    }

    #[test]
    fn a_device_displays_as_its_selector() {
        assert_eq!(Device::Default.to_string(), "default");
        assert_eq!(Device::Discrete { index: 1 }.to_string(), "discrete:1");
        assert_eq!(Device::Integrated { index: 0 }.to_string(), "integrated:0");
        assert_eq!(Device::Virtual { index: 2 }.to_string(), "virtual:2");
        assert_eq!(Device::Cpu.to_string(), "cpu");
    }

    #[test]
    fn a_displayed_device_parses_back_to_itself() {
        let devices = [
            Device::Default,
            Device::Discrete { index: 3 },
            Device::Integrated { index: 1 },
            Device::Virtual { index: 0 },
            Device::Cpu,
        ];

        for device in devices {
            let printed = device.to_string();
            let parsed = printed.parse::<Device>().expect("should parse");
            assert_eq!(parsed, device, "{printed}");
        }
    }

    #[test]
    fn the_default_device_is_the_backend_default() {
        assert_eq!(Device::default(), Device::Default);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_takes_default_and_discrete_and_refuses_the_rest() {
        let default = Device::Default.to_cubecl(Accelerator::Cuda).expect("should map");
        let discrete = Device::Discrete { index: 2 }
            .to_cubecl(Accelerator::Cuda)
            .expect("should map");

        assert_eq!(default, cubecl::Device::Cuda(CudaDevice::new(0)));
        assert_eq!(discrete, cubecl::Device::Cuda(CudaDevice::new(2)));

        let error = Device::Integrated { index: 0 }
            .to_cubecl(Accelerator::Cuda)
            .expect_err("should refuse");
        let message = error.to_string();
        assert!(message.contains("integrated:0"), "{message}");
        assert!(message.contains("cuda"), "{message}");
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn rocm_takes_default_and_discrete_and_refuses_the_rest() {
        let default = Device::Default.to_cubecl(Accelerator::Rocm).expect("should map");
        let discrete = Device::Discrete { index: 2 }
            .to_cubecl(Accelerator::Rocm)
            .expect("should map");

        assert_eq!(default, cubecl::Device::Hip(AmdDevice::new(0)));
        assert_eq!(discrete, cubecl::Device::Hip(AmdDevice::new(2)));

        let error = Device::Virtual { index: 1 }
            .to_cubecl(Accelerator::Rocm)
            .expect_err("should refuse");
        let message = error.to_string();
        assert!(message.contains("virtual:1"), "{message}");
        assert!(message.contains("rocm"), "{message}");
    }

    #[test]
    fn a_missing_device_names_how_many_gpus_the_backend_sees() {
        let accelerator = crate::default_accelerators()[0];
        let one = DeviceError::Missing {
            device: Device::Discrete { index: 1 },
            accelerator,
            available: 1,
        };
        let two = DeviceError::Missing {
            device: Device::Discrete { index: 2 },
            accelerator,
            available: 2,
        };

        let one_message = one.to_string();
        let two_message = two.to_string();

        assert_eq!(
            one_message,
            format!("device 'discrete:1' is not on this machine, the {accelerator} backend sees 1 GPU"),
        );
        assert!(two_message.ends_with("sees 2 GPUs"), "{two_message}");
    }

    #[cfg(feature = "metal")]
    #[test]
    fn every_selector_metal_can_name_maps_onto_a_metal_device() {
        use cubecl::device::MetalDevice;

        let gpus = MetalGpus::count();

        let default = Device::Default.to_cubecl(Accelerator::Metal).expect("should map");
        assert_eq!(default, cubecl::Device::Metal(MetalDevice::DefaultDevice));

        for selector in gpus.selectors() {
            let mapped = selector
                .to_cubecl(Accelerator::Metal)
                .expect("a listed gpu should map");
            let expected = gpus.device(selector).expect("a listed gpu should resolve");
            assert_eq!(mapped, cubecl::Device::Metal(expected), "{selector}");
        }

        let past_the_end = Device::Discrete { index: gpus.total() };
        let error = past_the_end
            .to_cubecl(Accelerator::Metal)
            .expect_err("should refuse");
        assert!(matches!(error, DeviceError::Missing { .. }), "{error}");

        let error = Device::Cpu
            .to_cubecl(Accelerator::Metal)
            .expect_err("should refuse");
        let message = error.to_string();
        assert!(message.contains("cpu"), "{message}");
        assert!(message.contains("metal"), "{message}");
    }

    #[cfg(feature = "vulkan")]
    #[test]
    fn every_kind_maps_onto_a_vulkan_device() {
        let cases = [
            (Device::Default, WgpuDeviceKind::DefaultDevice),
            (Device::Discrete { index: 1 }, WgpuDeviceKind::DiscreteGpu(1)),
            (Device::Integrated { index: 0 }, WgpuDeviceKind::IntegratedGpu(0)),
            (Device::Virtual { index: 2 }, WgpuDeviceKind::VirtualGpu(2)),
            (Device::Cpu, WgpuDeviceKind::Cpu),
        ];

        for (device, kind) in cases {
            let mapped = device.to_cubecl(Accelerator::Vulkan).expect("should map");
            let expected = WgpuDevice::new(kind).on(WgpuBackend::Vulkan);

            assert_eq!(mapped, cubecl::Device::Wgpu(expected), "{device}");
        }
    }
}
