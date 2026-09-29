use cubecl::client::ComputeClient;
use cubecl::features::{AtomicUsage, TypeUsage};
use cubecl::ir::{DeviceProperties, ElemType, FloatKind, IntKind, StorageType, Type, UIntKind};
use cubecl::prelude::Runtime;
use strum::IntoEnumIterator;
use strum_macros::EnumIter;

use crate::probe::{RuntimeTask, with_runtime};
use crate::{Accelerator, Device};

/// An element type a plane can be stored or computed in.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, EnumIter)]
pub enum ElementType {
    F32,
    F16,
    BF16,
    U8,
    U16,
    U32,
    I8,
    I16,
    I32,
}

impl std::fmt::Display for ElementType {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let spelling = match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::BF16 => "bf16",
            Self::U8 => "u8",
            Self::U16 => "u16",
            Self::U32 => "u32",
            Self::I8 => "i8",
            Self::I16 => "i16",
            Self::I32 => "i32",
        };

        formatter.write_str(spelling)
    }
}

impl ElementType {
    /// How many bytes one sample of this type occupies.
    pub const fn bytes(self) -> usize {
        match self {
            Self::U8 | Self::I8 => 1,
            Self::F16 | Self::BF16 | Self::U16 | Self::I16 => 2,
            Self::F32 | Self::U32 | Self::I32 => 4,
        }
    }

    /// The CubeCL element this maps onto.
    fn elem(self) -> ElemType {
        match self {
            Self::F32 => ElemType::Float(FloatKind::F32),
            Self::F16 => ElemType::Float(FloatKind::F16),
            Self::BF16 => ElemType::Float(FloatKind::BF16),
            Self::U8 => ElemType::UInt(UIntKind::U8),
            Self::U16 => ElemType::UInt(UIntKind::U16),
            Self::U32 => ElemType::UInt(UIntKind::U32),
            Self::I8 => ElemType::Int(IntKind::I8),
            Self::I16 => ElemType::Int(IntKind::I16),
            Self::I32 => ElemType::Int(IntKind::I32),
        }
    }
}

/// What a device can do, read once when its client is opened.
///
/// Storage and arithmetic are reported separately because the hardware separates them. A device
/// can hold a narrow integer in a buffer and still refuse to do maths in it.
#[derive(Debug, Clone)]
pub struct Capabilities {
    pub accelerator: Accelerator,
    /// Byte alignment every buffer view has to start on.
    pub alignment: u64,
    /// The largest single allocation this device will make.
    pub max_allocation: u64,
    pub max_cube_count: (u32, u32, u32),
    pub max_cube_dim: (u32, u32, u32),
    pub max_units_per_cube: u32,
    pub max_shared_memory: usize,
    /// Plane width as a minimum and a maximum.
    ///
    /// CUDA and ROCm report one fixed value in both. Under wgpu it is a range, and on Intel the
    /// real value varies with a kernel's register use, so the minimum is the safe assumption.
    pub plane_size: (u32, u32),
    /// The four questions below, answered once when the table was built.
    usage: Usage,
}

#[derive(Debug, Default, Clone)]
struct Usage {
    storable: Vec<ElementType>,
    computable: Vec<ElementType>,
    atomic_add: Vec<ElementType>,
    /// Pairs of inputs and accumulator, where both inputs have the same type.
    cooperative_matrix: Vec<(ElementType, ElementType)>,
}

impl Usage {
    fn from_properties(properties: &DeviceProperties) -> Self {
        let mut storable = Vec::new();
        let mut computable = Vec::new();
        let mut atomic_add = Vec::new();

        for element in ElementType::iter() {
            let usage = properties.type_usage(StorageType::Scalar(element.elem()));
            if usage.contains(TypeUsage::Buffer) {
                storable.push(element);
            }
            if usage.contains(TypeUsage::Arithmetic) {
                computable.push(element);
            }

            let atomic = properties.atomic_type_usage(Type::scalar(element.elem()));
            if atomic.contains(AtomicUsage::Add) {
                atomic_add.push(element);
            }
        }

        let mut cooperative_matrix = Vec::new();
        for config in &properties.features.matmul.cmma {
            let inputs = element_of(config.a_type);
            let other_inputs = element_of(config.b_type);
            let accumulator = element_of(config.cd_type);

            let (Some(inputs), Some(other_inputs), Some(accumulator)) = (inputs, other_inputs, accumulator)
            else {
                continue;
            };

            let combination = (inputs, accumulator);
            if inputs == other_inputs && !cooperative_matrix.contains(&combination) {
                cooperative_matrix.push(combination);
            }
        }

        Self {
            storable,
            computable,
            atomic_add,
            cooperative_matrix,
        }
    }
}

/// The element type a scalar storage type holds, if it is one this table names.
fn element_of(storage: StorageType) -> Option<ElementType> {
    ElementType::iter().find(|element| StorageType::Scalar(element.elem()) == storage)
}

impl Capabilities {
    pub fn from_properties(accelerator: Accelerator, properties: &DeviceProperties) -> Self {
        let hardware = &properties.hardware;
        let usage = Usage::from_properties(properties);

        Self {
            accelerator,
            alignment: properties.memory.alignment,
            max_allocation: properties.memory.max_page_size,
            max_cube_count: hardware.max_cube_count,
            max_cube_dim: hardware.max_cube_dim,
            max_units_per_cube: hardware.max_units_per_cube,
            max_shared_memory: hardware.max_shared_memory_size,
            plane_size: (hardware.plane_size_min, hardware.plane_size_max),
            usage,
        }
    }

    /// Describes a device without opening one.
    ///
    /// Every limit starts at an obviously inert value, so a test that forgets to set one fails
    /// instead of quietly passing on a number that looked plausible.
    pub fn builder(accelerator: Accelerator) -> CapabilitiesBuilder {
        CapabilitiesBuilder {
            capabilities: Self {
                accelerator,
                alignment: 0,
                max_allocation: 0,
                max_cube_count: (0, 0, 0),
                max_cube_dim: (0, 0, 0),
                max_units_per_cube: 0,
                max_shared_memory: 0,
                plane_size: (0, 0),
                usage: Usage::default(),
            },
        }
    }

    /// Whether a plane of this type can live in a buffer.
    pub fn can_store(&self, element: ElementType) -> bool {
        self.usage.storable.contains(&element)
    }

    /// Whether kernels can do arithmetic in this type.
    pub fn can_compute(&self, element: ElementType) -> bool {
        self.usage.computable.contains(&element)
    }

    /// Whether atomic add is available, which aggregating kernels need.
    pub fn can_atomic_add(&self, element: ElementType) -> bool {
        self.usage.atomic_add.contains(&element)
    }

    /// Whether a cooperative matrix takes `inputs` and accumulates in `accumulator`.
    pub fn has_cooperative_matrix(&self, inputs: ElementType, accumulator: ElementType) -> bool {
        let combination = (inputs, accumulator);
        self.usage.cooperative_matrix.contains(&combination)
    }
}

/// Builds a [Capabilities] table by hand, for tests and solvers that need one without a device.
#[derive(Debug, Clone)]
pub struct CapabilitiesBuilder {
    capabilities: Capabilities,
}

impl CapabilitiesBuilder {
    pub fn storable(mut self, elements: impl IntoIterator<Item = ElementType>) -> Self {
        self.capabilities.usage.storable = elements.into_iter().collect();
        self
    }

    pub fn computable(mut self, elements: impl IntoIterator<Item = ElementType>) -> Self {
        self.capabilities.usage.computable = elements.into_iter().collect();
        self
    }

    pub fn atomic_add(mut self, elements: impl IntoIterator<Item = ElementType>) -> Self {
        self.capabilities.usage.atomic_add = elements.into_iter().collect();
        self
    }

    pub fn cooperative_matrix(
        mut self,
        combinations: impl IntoIterator<Item = (ElementType, ElementType)>,
    ) -> Self {
        self.capabilities.usage.cooperative_matrix = combinations.into_iter().collect();
        self
    }

    pub fn alignment(mut self, alignment: u64) -> Self {
        self.capabilities.alignment = alignment;
        self
    }

    pub fn max_allocation(mut self, bytes: u64) -> Self {
        self.capabilities.max_allocation = bytes;
        self
    }

    pub fn plane_size(mut self, min: u32, max: u32) -> Self {
        self.capabilities.plane_size = (min, max);
        self
    }

    pub fn max_units_per_cube(mut self, units: u32) -> Self {
        self.capabilities.max_units_per_cube = units;
        self
    }

    pub fn max_cube_count(mut self, x: u32, y: u32, z: u32) -> Self {
        self.capabilities.max_cube_count = (x, y, z);
        self
    }

    /// Finishes the table, refusing one that cannot describe a device.
    ///
    /// The limits a launch is sized from start at zero specifically so a forgotten setter is
    /// caught here, at the fixture's construction, rather than later as a silently wrong
    /// `cube_dim` or `cube_count`.
    pub fn build(self) -> Capabilities {
        assert_ne!(
            self.capabilities.plane_size.0, 0,
            "CapabilitiesBuilder::plane_size was never set, so this table cannot describe a device",
        );
        assert_ne!(
            self.capabilities.max_units_per_cube, 0,
            "CapabilitiesBuilder::max_units_per_cube was never set, so this table cannot describe a device",
        );
        assert_ne!(
            self.capabilities.max_cube_count,
            (0, 0, 0),
            "CapabilitiesBuilder::max_cube_count was never set, so this table cannot describe a device",
        );

        self.capabilities
    }
}

/// Opens `accelerator` on `device` and reads what it can do.
///
/// Returns `None` when the backend cannot express the device, and when it does not start at all.
pub fn capabilities_for(accelerator: Accelerator, device: Device) -> Option<Capabilities> {
    with_runtime(accelerator, device, ReadCapabilities { accelerator })
}

/// Reads the capability table off an open client.
struct ReadCapabilities {
    accelerator: Accelerator,
}

impl RuntimeTask for ReadCapabilities {
    type Output = Capabilities;

    fn run<R: Runtime>(self, client: &ComputeClient<R>) -> Self::Output {
        Capabilities::from_properties(self.accelerator, client.properties())
    }
}

#[cfg(test)]
mod tests {
    use cubecl::features::{AtomicUsage, Features, MmaConfig, TypeUsage};
    use cubecl::ir::{
        DeviceProperties,
        ElemType,
        FloatKind,
        HardwareProperties,
        MemoryDeviceProperties,
        StorageType,
        Type,
        UIntKind,
    };
    use cubecl::profile::TimingMethod;

    use super::*;
    use crate::default_accelerators;

    /// Whichever backend this build enables, so these tests compile under any single one.
    ///
    /// The `compile_error!` guard in `lib.rs` means the list is never empty.
    fn any_accelerator() -> Accelerator {
        default_accelerators()[0]
    }

    fn hardware() -> HardwareProperties {
        HardwareProperties {
            load_width: 128,
            plane_size_min: 32,
            plane_size_max: 32,
            max_bindings: 8,
            max_shared_memory_size: 48 << 10,
            max_cube_count: (65535, 65535, 65535),
            max_units_per_cube: 1024,
            max_cube_dim: (1024, 1024, 64),
            num_streaming_multiprocessors: None,
            num_cpu_cores: None,
            num_tensor_cores: None,
            min_tensor_cores_dim: None,
            max_vector_size: 4,
        }
    }

    fn properties() -> DeviceProperties {
        DeviceProperties::new(
            Features::default(),
            MemoryDeviceProperties {
                max_page_size: 1 << 30,
                alignment: 32,
            },
            hardware(),
            TimingMethod::System,
        )
    }

    #[test]
    fn the_geometry_is_read_straight_off_the_properties() {
        let capabilities = Capabilities::from_properties(any_accelerator(), &properties());

        assert_eq!(capabilities.alignment, 32);
        assert_eq!(capabilities.max_allocation, 1 << 30);
        assert_eq!(capabilities.max_cube_count, (65535, 65535, 65535));
        assert_eq!(capabilities.max_cube_dim, (1024, 1024, 64));
        assert_eq!(capabilities.max_units_per_cube, 1024);
        assert_eq!(capabilities.max_shared_memory, 48 << 10);
        assert_eq!(capabilities.plane_size, (32, 32));
    }

    #[test]
    fn a_type_the_device_never_registered_is_supported_for_nothing() {
        let capabilities = Capabilities::from_properties(any_accelerator(), &properties());

        assert!(!capabilities.can_store(ElementType::U16));
        assert!(!capabilities.can_compute(ElementType::U16));
        assert!(!capabilities.can_atomic_add(ElementType::U32));
    }

    /// A device can hold a type in a buffer and still refuse to do arithmetic in it, which is the
    /// case a stage negotiating for a narrow integer has to be able to see.
    #[test]
    fn storage_and_arithmetic_are_reported_separately() {
        let mut properties = properties();
        properties.register_type_usage(
            StorageType::Scalar(ElemType::UInt(UIntKind::U16)),
            TypeUsage::Buffer | TypeUsage::Conversion,
        );

        let capabilities = Capabilities::from_properties(any_accelerator(), &properties);

        assert!(capabilities.can_store(ElementType::U16));
        assert!(!capabilities.can_compute(ElementType::U16));
    }

    #[test]
    fn a_built_table_answers_from_what_it_was_told() {
        let capabilities = Capabilities::builder(any_accelerator())
            .storable([ElementType::F32, ElementType::U16, ElementType::U8])
            .computable([ElementType::F32, ElementType::U8])
            .atomic_add([ElementType::F32])
            .alignment(64)
            .max_allocation(2 << 30)
            .plane_size(32, 32)
            .max_units_per_cube(1024)
            .max_cube_count(65535, 65535, 65535)
            .build();

        assert!(capabilities.can_store(ElementType::U16));
        assert!(!capabilities.can_compute(ElementType::U16));
        assert!(capabilities.can_compute(ElementType::F32));
        assert!(capabilities.can_atomic_add(ElementType::F32));
        assert!(!capabilities.can_atomic_add(ElementType::U8));
    }

    /// A device that holds a type but cannot do maths in it is the case a stage falls back on, and
    /// no accelerator on this machine reports it.
    #[test]
    fn storable_does_not_imply_computable() {
        let capabilities = Capabilities::builder(any_accelerator())
            .storable([ElementType::U16])
            .plane_size(32, 32)
            .max_units_per_cube(1024)
            .max_cube_count(65535, 65535, 65535)
            .build();

        assert!(capabilities.can_store(ElementType::U16));
        assert!(!capabilities.can_compute(ElementType::U16));
    }

    #[test]
    #[should_panic(expected = "CapabilitiesBuilder::plane_size was never set")]
    fn building_without_plane_size_fails_at_the_fixture_rather_than_later() {
        Capabilities::builder(any_accelerator())
            .max_units_per_cube(1024)
            .max_cube_count(65535, 65535, 65535)
            .build();
    }

    #[test]
    #[should_panic(expected = "CapabilitiesBuilder::max_units_per_cube was never set")]
    fn building_without_max_units_per_cube_fails_at_the_fixture_rather_than_later() {
        Capabilities::builder(any_accelerator())
            .plane_size(32, 32)
            .build();
    }

    #[test]
    #[should_panic(expected = "CapabilitiesBuilder::max_cube_count was never set")]
    fn building_without_max_cube_count_fails_at_the_fixture_rather_than_later() {
        Capabilities::builder(any_accelerator())
            .plane_size(32, 32)
            .max_units_per_cube(1024)
            .build();
    }

    #[test]
    fn a_type_registered_for_arithmetic_reports_it() {
        let mut properties = properties();
        properties.register_type_usage(
            StorageType::Scalar(ElemType::UInt(UIntKind::U16)),
            TypeUsage::all(),
        );

        let capabilities = Capabilities::from_properties(any_accelerator(), &properties);

        assert!(capabilities.can_store(ElementType::U16));
        assert!(capabilities.can_compute(ElementType::U16));
    }

    #[test]
    fn atomic_add_is_reported_only_when_registered() {
        let mut properties = properties();
        properties.register_atomic_type_usage(
            Type::scalar(ElemType::UInt(UIntKind::U32)),
            AtomicUsage::LoadStore | AtomicUsage::Add,
        );

        let capabilities = Capabilities::from_properties(any_accelerator(), &properties);

        assert!(capabilities.can_atomic_add(ElementType::U32));
        assert!(!capabilities.can_atomic_add(ElementType::F32));
    }

    #[test]
    fn the_accelerator_is_carried_through() {
        let capabilities = Capabilities::from_properties(any_accelerator(), &properties());

        assert_eq!(capabilities.accelerator, any_accelerator());
    }

    #[test]
    fn every_element_reports_its_own_width() {
        let widths = [
            (ElementType::U8, 1),
            (ElementType::I8, 1),
            (ElementType::F16, 2),
            (ElementType::BF16, 2),
            (ElementType::U16, 2),
            (ElementType::I16, 2),
            (ElementType::F32, 4),
            (ElementType::U32, 4),
            (ElementType::I32, 4),
        ];

        for (element, expected) in widths {
            assert_eq!(element.bytes(), expected, "{element:?}");
        }
    }

    /// A transposition in the mapping would otherwise pass every other test, because they all route
    /// through the same two types.
    #[test]
    fn every_element_maps_to_a_distinct_cubecl_type() {
        let elements = [
            ElementType::F32,
            ElementType::F16,
            ElementType::BF16,
            ElementType::U8,
            ElementType::U16,
            ElementType::U32,
            ElementType::I8,
            ElementType::I16,
            ElementType::I32,
        ];

        let mut mapped: Vec<ElemType> = elements.iter().map(|element| element.elem()).collect();
        mapped.sort_by_key(|elem| format!("{elem:?}"));
        mapped.dedup();

        assert_eq!(
            mapped.len(),
            elements.len(),
            "two elements map to the same CubeCL type"
        );
    }

    #[test]
    fn element_type_displays_its_lower_case_spelling() {
        assert_eq!(ElementType::F32.to_string(), "f32");
        assert_eq!(ElementType::BF16.to_string(), "bf16");
    }

    #[cfg(all(feature = "gpu-tests", any(feature = "vulkan", feature = "metal")))]
    #[test]
    fn the_built_backend_reports_a_usable_capability_table() {
        use crate::testing::TEST_ACCELERATOR;

        let capabilities =
            capabilities_for(TEST_ACCELERATOR, Device::Default).expect("the backend should open");

        assert_eq!(capabilities.accelerator, TEST_ACCELERATOR);
        assert!(
            capabilities.alignment.is_power_of_two(),
            "{}",
            capabilities.alignment
        );
        assert!(capabilities.max_allocation > 0);
        assert!(capabilities.max_units_per_cube > 0);
        assert!(capabilities.plane_size.0 <= capabilities.plane_size.1);

        // Every backend this crate targets guarantees f32 arithmetic and storage.
        assert!(capabilities.can_store(ElementType::F32));
        assert!(capabilities.can_compute(ElementType::F32));
    }

    #[cfg(all(feature = "gpu-tests", any(feature = "vulkan", feature = "metal")))]
    #[test]
    fn a_device_the_backend_does_not_have_reports_nothing() {
        use crate::testing::TEST_ACCELERATOR;

        assert!(capabilities_for(TEST_ACCELERATOR, Device::Discrete { index: 99 }).is_none());
    }

    fn f16_into_f32() -> MmaConfig {
        let f16_type = StorageType::Scalar(ElemType::Float(FloatKind::F16));
        let f32_type = StorageType::Scalar(ElemType::Float(FloatKind::F32));

        MmaConfig {
            a_type: f16_type,
            b_type: f16_type,
            cd_type: f32_type,
            m: 16,
            n: 16,
            k: 16,
        }
    }

    #[test]
    fn a_cooperative_matrix_the_device_lists_is_reported() {
        let mut properties = properties();
        let combination = f16_into_f32();
        properties.features.matmul.cmma.insert(combination);

        let capabilities = Capabilities::from_properties(any_accelerator(), &properties);

        assert!(capabilities.has_cooperative_matrix(ElementType::F16, ElementType::F32));
        assert!(!capabilities.has_cooperative_matrix(ElementType::F32, ElementType::F32));
        assert!(!capabilities.has_cooperative_matrix(ElementType::F16, ElementType::F16));
    }

    #[test]
    fn a_device_without_cooperative_matrices_reports_none() {
        let properties = properties();

        let capabilities = Capabilities::from_properties(any_accelerator(), &properties);

        assert!(!capabilities.has_cooperative_matrix(ElementType::F16, ElementType::F32));
    }

    #[test]
    fn a_combination_with_mixed_inputs_is_not_reported() {
        let mut properties = properties();
        let mut combination = f16_into_f32();
        combination.b_type = StorageType::Scalar(ElemType::Float(FloatKind::BF16));
        properties.features.matmul.cmma.insert(combination);

        let capabilities = Capabilities::from_properties(any_accelerator(), &properties);

        assert!(!capabilities.has_cooperative_matrix(ElementType::F16, ElementType::F32));
        assert!(!capabilities.has_cooperative_matrix(ElementType::BF16, ElementType::F32));
    }

    #[test]
    fn the_builder_sets_cooperative_matrices() {
        let combinations = [(ElementType::F16, ElementType::F32)];

        let capabilities = Capabilities::builder(any_accelerator())
            .plane_size(32, 32)
            .max_units_per_cube(1024)
            .max_cube_count(65535, 65535, 65535)
            .cooperative_matrix(combinations)
            .build();

        assert!(capabilities.has_cooperative_matrix(ElementType::F16, ElementType::F32));
        assert!(!capabilities.has_cooperative_matrix(ElementType::F32, ElementType::F32));
    }

    #[cfg(all(feature = "gpu-tests", any(feature = "vulkan", feature = "metal")))]
    #[test]
    fn this_device_reports_its_cooperative_matrices() {
        use crate::testing::TEST_ACCELERATOR;

        let capabilities = capabilities_for(TEST_ACCELERATOR, Device::Default).expect("the device opens");
        let f16_into_f32 = capabilities.has_cooperative_matrix(ElementType::F16, ElementType::F32);

        println!("f16 into f32 cooperative matrix: {f16_into_f32}");
    }
}
