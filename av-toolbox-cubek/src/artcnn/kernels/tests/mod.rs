mod direct;
mod tune;

use cubecl::features::{Features, MmaConfig};
use cubecl::ir::{DeviceProperties, HardwareProperties, MemoryDeviceProperties};
use cubecl::profile::TimingMethod;

use crate::artcnn::Precision;
use crate::artcnn::body::storage_type;

/// What a test device offers the direct convolution.
pub(super) struct TestDevice {
    pub(super) plane_size_min: u32,
    pub(super) plane_size_max: u32,
    pub(super) shared_bytes: usize,
    pub(super) units: u32,
    pub(super) offers_matrix: bool,
}

/// An AMD R9700 under RADV.
pub(super) const R9700: TestDevice = TestDevice {
    plane_size_min: 32,
    plane_size_max: 64,
    shared_bytes: 65536,
    units: 1024,
    offers_matrix: true,
};

impl TestDevice {
    pub(super) fn properties(&self) -> DeviceProperties {
        let mut features = Features::default();

        if self.offers_matrix {
            let half_type = storage_type(Precision::F16);
            let float_type = storage_type(Precision::F32);
            let config = MmaConfig {
                a_type: half_type,
                b_type: half_type,
                cd_type: float_type,
                m: 16,
                n: 16,
                k: 16,
            };
            features.matmul.cmma.insert(config);
        }

        let memory = MemoryDeviceProperties {
            max_page_size: 1 << 30,
            alignment: 256,
        };
        let hardware = HardwareProperties {
            load_width: 128,
            plane_size_min: self.plane_size_min,
            plane_size_max: self.plane_size_max,
            max_bindings: 64,
            max_shared_memory_size: self.shared_bytes,
            max_cube_count: (65535, 65535, 65535),
            max_units_per_cube: self.units,
            max_cube_dim: (1024, 1024, 1024),
            num_streaming_multiprocessors: None,
            num_cpu_cores: None,
            num_tensor_cores: None,
            min_tensor_cores_dim: Some(16),
            max_vector_size: 4,
        };

        DeviceProperties::new(features, memory, hardware, TimingMethod::Device)
    }
}
