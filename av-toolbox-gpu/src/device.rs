use std::fmt;
use std::str::FromStr;

use crate::Accelerator;

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
    Discrete { index: usize },
    /// Integrated GPU at ordinal `index`. wgpu only.
    Integrated { index: usize },
    /// Virtual GPU at ordinal `index`. wgpu only.
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

#[cfg(feature = "cuda")]
impl Device {
    pub fn to_cuda(self) -> Result<cubecl::cuda::CudaDevice, DeviceError> {
        match self {
            Self::Default => Ok(cubecl::cuda::CudaDevice { index: 0 }),
            Self::Discrete { index } => Ok(cubecl::cuda::CudaDevice { index }),
            other => Err(DeviceError::Unsupported {
                device: other,
                accelerator: Accelerator::Cuda,
                supported: "default or discrete[:N]",
            }),
        }
    }
}

#[cfg(feature = "rocm")]
impl Device {
    pub fn to_amd(self) -> Result<cubecl::hip::AmdDevice, DeviceError> {
        match self {
            Self::Default => Ok(cubecl::hip::AmdDevice { index: 0 }),
            Self::Discrete { index } => Ok(cubecl::hip::AmdDevice { index }),
            other => Err(DeviceError::Unsupported {
                device: other,
                accelerator: Accelerator::Rocm,
                supported: "default or discrete[:N]",
            }),
        }
    }
}

#[cfg(any(feature = "vulkan", feature = "metal"))]
impl Device {
    pub fn to_wgpu(self) -> Result<cubecl::wgpu::WgpuDevice, DeviceError> {
        use cubecl::wgpu::WgpuDevice;

        Ok(match self {
            Self::Default => WgpuDevice::DefaultDevice,
            Self::Discrete { index } => WgpuDevice::DiscreteGpu(index),
            Self::Integrated { index } => WgpuDevice::IntegratedGpu(index),
            Self::Virtual { index } => WgpuDevice::VirtualGpu(index),
            Self::Cpu => WgpuDevice::Cpu,
        })
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
        assert_eq!(Device::Default.to_cuda().expect("should map").index, 0);
        assert_eq!(
            Device::Discrete { index: 2 }.to_cuda().expect("should map").index,
            2
        );

        let error = Device::Integrated { index: 0 }
            .to_cuda()
            .expect_err("should refuse");
        let message = error.to_string();
        assert!(message.contains("integrated:0"), "{message}");
        assert!(message.contains("cuda"), "{message}");
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn rocm_takes_default_and_discrete_and_refuses_the_rest() {
        assert_eq!(Device::Default.to_amd().expect("should map").index, 0);
        assert_eq!(
            Device::Discrete { index: 2 }.to_amd().expect("should map").index,
            2
        );

        let error = Device::Virtual { index: 1 }.to_amd().expect_err("should refuse");
        let message = error.to_string();
        assert!(message.contains("virtual:1"), "{message}");
        assert!(message.contains("rocm"), "{message}");
    }

    #[cfg(any(feature = "vulkan", feature = "metal"))]
    #[test]
    fn every_kind_maps_onto_a_wgpu_device() {
        use cubecl::wgpu::WgpuDevice;

        assert_eq!(
            Device::Default.to_wgpu().expect("should map"),
            WgpuDevice::DefaultDevice
        );
        assert_eq!(
            Device::Discrete { index: 1 }.to_wgpu().expect("should map"),
            WgpuDevice::DiscreteGpu(1),
        );
        assert_eq!(
            Device::Integrated { index: 0 }.to_wgpu().expect("should map"),
            WgpuDevice::IntegratedGpu(0),
        );
        assert_eq!(
            Device::Virtual { index: 2 }.to_wgpu().expect("should map"),
            WgpuDevice::VirtualGpu(2),
        );
        assert_eq!(Device::Cpu.to_wgpu().expect("should map"), WgpuDevice::Cpu);
    }
}
