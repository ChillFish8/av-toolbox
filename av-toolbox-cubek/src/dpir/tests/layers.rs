use cubek_matmul::multi_level::Strategy as MatmulStrategy;

use crate::cnn::Precision;
use crate::dpir::layers::product_strategy;

#[test]
fn f16_products_run_on_cooperative_matrices_where_the_device_offers_them() {
    let strategy = product_strategy(Precision::F16, true);

    assert!(matches!(strategy, MatmulStrategy::SimpleCyclicCmma(_)));
}

#[test]
fn products_run_on_units_without_the_matrix_or_on_f32_maps() {
    let without_matrix = product_strategy(Precision::F16, false);
    let on_f32 = product_strategy(Precision::F32, true);

    assert!(matches!(without_matrix, MatmulStrategy::SimpleUnit(_)));
    assert!(matches!(on_f32, MatmulStrategy::SimpleUnit(_)));
}
