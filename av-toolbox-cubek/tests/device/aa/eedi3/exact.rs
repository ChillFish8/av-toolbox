use av_toolbox_cubek::aa::eedi3_mul_exact;
use av_toolbox_cubek::testing::SplitMix;
use av_toolbox_gpu::{BuildContext, BuildTask};

use crate::support::{Sizing, backend};

/// Operand pairs for the exact multiply.
#[derive(Default)]
struct ProductCases {
    left: Vec<f32>,
    right: Vec<f32>,
}

impl ProductCases {
    fn push(&mut self, left: f32, right: f32) {
        self.left.push(left);
        self.right.push(right);
    }
}

/// The blend's weights at `based_aa`'s `vthresh` on 8 and 10 bits, with the edges of their range.
fn blend_weights() -> Vec<f32> {
    let twelfth = 1.0f32 / 12.0;
    let twenty_fourth = 1.0f32 / 24.0;
    let seven_twelfths = 7.0 * twelfth;

    vec![
        0.0,
        twelfth,
        twenty_fourth,
        1.0 / 48.0,
        1.0 / 96.0,
        0.25,
        0.5,
        seven_twelfths,
        1.0 - twelfth,
        1.0 - twenty_fourth,
        1.0,
    ]
}

/// The smallest subnormal and the largest, the smallest normal, and values either side of them.
fn tiny_values() -> Vec<f32> {
    let smallest_normal = f32::MIN_POSITIVE;
    let below_normal = f32::from_bits(0x007F_FFFF);
    let above_normal = f32::from_bits(0x0080_0001);

    vec![
        f32::from_bits(1),
        f32::from_bits(2),
        f32::from_bits(3),
        f32::from_bits(0x0000_1234),
        f32::from_bits(0x0040_0000),
        below_normal,
        smallest_normal,
        above_normal,
        1e-30,
        1e-20,
    ]
}

/// Factors that keep a tiny value tiny, cross it into the normals, or round it at a tie.
fn tiny_factors() -> Vec<f32> {
    let just_above_one = f32::from_bits(0x3F80_0001);
    let just_below_one = f32::from_bits(0x3F7F_FFFF);

    vec![
        0.25,
        0.5,
        0.75,
        1.0,
        1.5,
        3.0,
        just_above_one,
        just_below_one,
        65535.0,
        8_388_608.0,
        1e-10,
        1e10,
        3e38,
    ]
}

/// Products that pass the largest finite f32, or land on it.
fn overflow_cases(cases: &mut ProductCases) {
    let just_above_one = f32::from_bits(0x3F80_0001);
    let pairs = [
        (f32::MAX, 1.0),
        (f32::MAX, just_above_one),
        (f32::MAX, 2.0),
        (f32::MAX, f32::MAX),
        (3e38, 3e38),
        (1e20, 1e20),
        (1.8446744e19, 1.8446743e19),
        (-f32::MAX, 2.0),
        (f32::MAX, -0.5),
    ];

    for (left, right) in pairs {
        cases.push(left, right);
        cases.push(right, left);
    }
}

/// Signed zeros against values of either sign, subnormals included.
fn zero_cases(cases: &mut ProductCases) {
    let others = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        65535.0,
        -3e38,
        f32::from_bits(1),
        f32::from_bits(0x8000_0001),
    ];

    for zero in [0.0f32, -0.0] {
        for other in others {
            cases.push(zero, other);
            cases.push(other, zero);
        }
    }
}

/// Every pair of tiny values and factors, in both orders and with both signs.
fn tiny_cases(cases: &mut ProductCases) {
    let tiny = tiny_values();
    let factors = tiny_factors();

    for value in &tiny {
        for factor in &factors {
            cases.push(*value, *factor);
            cases.push(*factor, *value);
            cases.push(-*value, *factor);
            cases.push(*value, -*factor);
        }

        for other in &tiny {
            cases.push(*value, *other);
        }
    }
}

/// A random finite f32, both signs and every exponent field.
fn random_finite(random: &mut SplitMix) -> f32 {
    loop {
        let drawn = random.draw() as u32;
        let value = f32::from_bits(drawn);
        if value.is_finite() {
            return value;
        }
    }
}

/// Random finite pairs, then random significands under every pair of finite exponent fields.
fn random_cases(cases: &mut ProductCases) {
    let mut random = SplitMix(0x5eed_e3d1);

    for _ in 0..1_000_000 {
        let left = random_finite(&mut random);
        let right = random_finite(&mut random);
        cases.push(left, right);
    }

    for left_field in 0..255u32 {
        for right_field in 0..255u32 {
            let left_fraction = random.below(1 << 23);
            let right_fraction = random.below(1 << 23);
            let left = f32::from_bits(left_field << 23 | left_fraction);
            let right = f32::from_bits(right_field << 23 | right_fraction);
            cases.push(left, right);
        }
    }
}

/// The operand pairs of `mul_exact_matches_the_hardware_product`.
fn product_cases() -> ProductCases {
    let mut cases = ProductCases::default();

    let one = 1.0f32.to_bits();
    let samples = [0.0, 1.0, 255.0, 1023.0, 65535.0];
    let unit_values = (0..=one).step_by(4093).chain([one]);
    for bits in unit_values {
        let weight = f32::from_bits(bits);
        for sample in samples {
            cases.push(weight, sample);
        }
    }

    let weights = blend_weights();
    for sample in 0..=65535u32 {
        for weight in &weights {
            cases.push(*weight, sample as f32);
        }
    }

    zero_cases(&mut cases);
    tiny_cases(&mut cases);
    overflow_cases(&mut cases);
    random_cases(&mut cases);

    cases
}

/// The exact multiply on the device over a set of operand pairs.
struct ExactProducts(ProductCases);

impl BuildTask for ExactProducts {
    type Output = Vec<f32>;

    fn run(self, build: &BuildContext) -> Self::Output {
        let client = build.client();
        let frame_context = build.frame_context();
        let sizing = Sizing(frame_context);

        eedi3_mul_exact(client, &sizing, &self.0.left, &self.0.right)
    }
}

#[test]
fn mul_exact_matches_the_hardware_product() {
    let cases = product_cases();
    let left = cases.left.clone();
    let right = cases.right.clone();
    let products = backend().build(ExactProducts(cases));
    assert_eq!(products.len(), left.len(), "every pair should have a product");

    let mut mismatches = Vec::new();
    for (index, found) in products.iter().enumerate() {
        let expected = left[index] * right[index];
        if found.to_bits() != expected.to_bits() {
            mismatches.push((left[index], right[index], *found, expected));
        }
    }

    let examples: Vec<String> = mismatches
        .iter()
        .take(5)
        .map(|(left, right, found, expected)| {
            format!("{left:e} * {right:e} gave {found:e}, not {expected:e}")
        })
        .collect();
    assert!(
        mismatches.is_empty(),
        "{} of {} products differ: {examples:?}",
        mismatches.len(),
        left.len()
    );
}
