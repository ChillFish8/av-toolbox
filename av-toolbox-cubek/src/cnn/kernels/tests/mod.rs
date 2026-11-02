mod add;
mod direct;
mod strided;
mod tune;
mod winograd;

use cubecl::features::{Features, MmaConfig};
use cubecl::ir::{DeviceIdentity, DeviceProperties, HardwareProperties, MemoryDeviceProperties};
use cubecl::profile::TimingMethod;

use crate::cnn::Precision;
use crate::cnn::convolution::elem_type;

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
            let half_type = elem_type(Precision::F16);
            let float_type = elem_type(Precision::F32);
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

        let memory = MemoryDeviceProperties::new(1 << 30, 256);
        let hardware = HardwareProperties {
            load_width: 128,
            vector_register_count: None,
            plane_size_min: self.plane_size_min,
            plane_size_max: self.plane_size_max,
            max_bindings: 64,
            max_shared_memory_size: self.shared_bytes,
            max_cube_count: (65535, 65535, 65535),
            max_units_per_cube: self.units,
            max_cube_dim: (1024, 1024, 1024),
            num_streaming_multiprocessors: None,
            num_cpu_cores: None,
            last_level_cache_size: None,
            num_tensor_cores: None,
            min_tensor_cores_dim: Some(16),
            max_vector_size: 4,
            cube_mma_reserved_shared_memory: 0,
        };

        DeviceProperties::new(
            features,
            memory,
            hardware,
            TimingMethod::Device,
            DeviceIdentity::default(),
        )
    }
}
