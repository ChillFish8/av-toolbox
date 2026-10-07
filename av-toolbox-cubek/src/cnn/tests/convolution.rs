use std::collections::BTreeSet;

use cubecl::features::MmaConfig;
use cubecl::ir::{AddressType, StorageType};
use cubecl::prelude::CubePrimitive;
use cubek_convolution::components::{ConvolutionProblem, Dimensionality};
use cubek_convolution::definition::ConvBlueprint;
use cubek_matmul::components::stage::PartitionBuffering;
use cubek_matmul::components::tile::TileMatmulKind;
use cubek_matmul::definition::MatmulGlobalElems;
use cubek_std::SwizzleModes;
use cubek_std::cube_count::{CubeCountStrategy, GlobalOrder};

use crate::cnn::Region;
use crate::cnn::convolution::{convolution_problem, forced_blueprint, offers_cmma_16_in};

/// A 3x3 convolution from 64 channels to `out_channels` over a region `width` by `height`, on f16 maps.
fn half_problem(width: u32, height: u32, out_channels: usize) -> ConvolutionProblem {
    let half_type = half::f16::as_type_native_unchecked().storage_type();

    typed_problem(width, height, out_channels, half_type)
}

/// A 3x3 convolution from 64 channels to `out_channels` over a region `width` by `height`, on `dtype` maps.
fn typed_problem(width: u32, height: u32, out_channels: usize, dtype: StorageType) -> ConvolutionProblem {
    let region = Region {
        x: 0,
        y: 0,
        width,
        height,
    };
    let globals = MatmulGlobalElems {
        lhs: dtype,
        rhs: dtype,
        out: dtype,
    };

    convolution_problem(region, 64, out_channels, globals, AddressType::U32)
}

#[test]
fn the_forced_blueprint_tiles_a_large_region_four_by_four_by_two() {
    let problem = half_problem(128, 64, 64);

    let forced = forced_blueprint(&problem, 64, true);

    let Some(ConvBlueprint::Forward(blueprint)) = forced else {
        panic!("a forward blueprint is forced at 8192 samples, not {forced:?}");
    };
    let matmul = &blueprint.matmul;
    let scheme = matmul.tiling_scheme;
    assert_eq!(scheme.tile_size, (16, 16, 16).into());
    assert_eq!(scheme.partition_size, (4, 4, 2).into());
    assert_eq!(scheme.stage_size, (4, 1, 1).into());
    assert_eq!(matmul.tile_matmul, TileMatmulKind::Cmma);
    assert_eq!(matmul.plane_dim, 64);
    assert_eq!(matmul.partition_buffering, PartitionBuffering::Single);
    assert_eq!(
        matmul.hypercube_blueprint.global_order,
        GlobalOrder::SwizzleRow(4)
    );
    assert_eq!(
        matmul.hypercube_blueprint.cube_count_strategy,
        CubeCountStrategy::FromProblem
    );
    assert_eq!(matmul.swizzle_modes, SwizzleModes::default());
    assert_eq!(blueprint.dimensionality, Dimensionality::Dim2);
    assert!(blueprint.has_bias);
}

#[test]
fn the_forced_blueprint_needs_64_lanes_per_plane() {
    let problem = half_problem(128, 64, 64);

    let forced = forced_blueprint(&problem, 32, true);

    assert_eq!(forced, None);
}

#[test]
fn the_forced_blueprint_needs_a_16_by_16_by_16_cooperative_matrix() {
    let problem = half_problem(128, 64, 64);

    let forced = forced_blueprint(&problem, 64, false);

    assert_eq!(forced, None);
}

#[test]
fn the_forced_blueprint_needs_8192_samples() {
    let problem = half_problem(8191, 1, 64);

    let forced = forced_blueprint(&problem, 64, true);

    assert_eq!(forced, None);
}

#[test]
fn the_forced_blueprint_needs_a_multiple_of_16_channels() {
    let problem = half_problem(128, 64, 24);

    let forced = forced_blueprint(&problem, 64, true);

    assert_eq!(forced, None);
}

#[test]
fn the_forced_blueprint_needs_f16_operands() {
    let float_type = f32::as_type_native_unchecked().storage_type();
    let problem = typed_problem(128, 64, 64, float_type);

    let forced = forced_blueprint(&problem, 64, true);

    assert_eq!(forced, None);
}

/// A 16x16x16 cooperative matrix on f16 operands, accumulating in `accumulator`.
fn half_matrix(accumulator: StorageType) -> MmaConfig {
    let half_type = half::f16::as_type_native_unchecked().storage_type();

    MmaConfig {
        a_type: half_type,
        b_type: half_type,
        cd_type: accumulator,
        m: 16,
        n: 16,
        k: 16,
    }
}

#[test]
fn an_f16_accumulator_alone_does_not_offer_the_forced_matrix() {
    let half_type = half::f16::as_type_native_unchecked().storage_type();
    let config = half_matrix(half_type);
    let configs = BTreeSet::from([config]);

    assert!(!offers_cmma_16_in(&configs));
}

#[test]
fn an_f32_accumulator_offers_the_forced_matrix() {
    let float_type = f32::as_type_native_unchecked().storage_type();
    let config = half_matrix(float_type);
    let configs = BTreeSet::from([config]);

    assert!(offers_cmma_16_in(&configs));
}

#[test]
fn the_forced_blueprint_is_only_given_up_to_the_measured_widths() {
    for (out_channels, forced) in [(64, true), (96, true), (128, false), (256, false), (512, false)] {
        let problem = half_problem(128, 64, out_channels);

        let blueprint = forced_blueprint(&problem, 64, true);

        assert_eq!(
            blueprint.is_some(),
            forced,
            "forced blueprint at {out_channels} channels"
        );
    }
}
