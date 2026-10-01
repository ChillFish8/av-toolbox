// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

#[cfg(any(test, feature = "test-support"))]
use crate::Dispatch;

/// The sign bit of an f32.
const SIGN_BIT: u32 = 0x8000_0000;

/// The bits of an f32 below its sign.
const MAGNITUDE_MASK: u32 = 0x7FFF_FFFF;

/// The 23 stored bits of an f32's significand.
const FRACTION_MASK: u32 = 0x007F_FFFF;

/// The leading bit of a 24-bit significand, which a normal f32 leaves out.
const LEADING_BIT: u32 = 0x0080_0000;

/// The low 24 bits of a word.
const LOW_MASK: u32 = 0x00FF_FFFF;

/// The bits of an infinite f32 magnitude.
const INFINITY_BITS: u32 = 0x7F80_0000;

/// A finite, non-zero f32 magnitude as `significand * 2^(exponent - 150)`, the significand between `2^23`
/// and `2^24 - 1`.
#[derive(CubeType)]
struct Unpacked {
    significand: u32,
    exponent: i32,
}

/// Splits a finite, non-zero f32 magnitude, shifting a subnormal up until its leading bit reaches bit 23.
#[cube]
fn unpack(magnitude: u32) -> Unpacked {
    let field = magnitude >> 23;
    let fraction = magnitude & FRACTION_MASK;

    let mut significand = fraction | LEADING_BIT;
    let mut exponent = field as i32;
    if field == 0 {
        let shift = u32::leading_zeros(fraction) - 8;
        significand = fraction << shift;
        exponent = 1 - shift as i32;
    }

    Unpacked {
        significand,
        exponent,
    }
}

/// The exact product of two 24-bit significands, as `high * 2^24 + low`.
#[derive(CubeType)]
struct WideProduct {
    high: u32,
    low: u32,
}

/// Multiplies two 24-bit significands from their 12-bit halves, so no partial product reaches `2^25`.
#[cube]
fn wide_product(left: u32, right: u32) -> WideProduct {
    let left_high = left >> 12;
    let left_low = left & 0xFFF;
    let right_high = right >> 12;
    let right_low = right & 0xFFF;

    let middle = left_high * right_low + left_low * right_high;
    let low_sum = left_low * right_low + ((middle & 0xFFF) << 12);
    let high = left_high * right_high + (middle >> 12) + (low_sum >> 24);
    let low = low_sum & LOW_MASK;

    WideProduct { high, low }
}

/// The magnitude bits of a normal result, rounding `kept` by the 24 bits below it in `rest`.
///
/// The result overflows to infinity once its biased exponent reaches 255.
#[cube]
fn normal_bits(kept: u32, rest: u32, biased: i32) -> u32 {
    let odd = kept & 1 == 1;
    let round_up = rest > LEADING_BIT || (rest == LEADING_BIT && odd);

    let mut rounded = kept;
    let mut exponent = biased;
    if round_up {
        rounded += 1;
    }

    if rounded == LOW_MASK + 1 {
        rounded = LEADING_BIT;
        exponent += 1;
    }

    let mut bits = INFINITY_BITS.runtime();
    if exponent < 255 {
        bits = ((exponent as u32) << 23) | (rounded & FRACTION_MASK);
    }

    bits
}

/// The magnitude bits of a subnormal or zero result, shifting `kept` down until its biased exponent
/// reaches 1 and rounding by the bits shifted out and those in `rest`.
#[cube]
fn subnormal_bits(kept: u32, rest: u32, biased: i32) -> u32 {
    let shift = (1 - biased) as u32;

    let mut bits = 0u32;
    if shift < 25 {
        let dropped = kept & ((1u32 << shift) - 1);
        let half = 1u32 << (shift - 1);
        let truncated = kept >> shift;
        let odd = truncated & 1 == 1;
        let round_up = dropped > half || (dropped == half && (rest != 0 || odd));

        bits = truncated;
        if round_up {
            bits += 1;
        }
    }

    bits
}

/// The magnitude bits of the product of two finite, non-zero f32 magnitudes.
#[cube]
fn product_magnitude(left: u32, right: u32) -> u32 {
    let left_parts = unpack(left);
    let right_parts = unpack(right);
    let product = wide_product(left_parts.significand, right_parts.significand);

    let mut biased = left_parts.exponent + right_parts.exponent - 126;
    let mut kept = product.high;
    let mut rest = product.low;
    if product.high < LEADING_BIT {
        kept = (product.high << 1) | (product.low >> 23);
        rest = (product.low << 1) & LOW_MASK;
        biased -= 1;
    }

    if biased >= 1 {
        normal_bits(kept, rest, biased)
    } else {
        subnormal_bits(kept, rest, biased)
    }
}

/// The IEEE f32 product `fl(left * right)`, rounded to nearest with ties to even, from 32-bit integer
/// operations only.
///
/// It holds for every finite `left` and `right`, subnormals in and out included. With no float multiply,
/// no compiler can fuse it into an addition that follows.
#[cube]
pub(super) fn mul_exact(left: f32, right: f32) -> f32 {
    let left_bits = u32::reinterpret(left);
    let right_bits = u32::reinterpret(right);
    let sign = (left_bits ^ right_bits) & SIGN_BIT;
    let left_magnitude = left_bits & MAGNITUDE_MASK;
    let right_magnitude = right_bits & MAGNITUDE_MASK;

    let mut magnitude = 0u32;
    if left_magnitude != 0 && right_magnitude != 0 {
        magnitude = product_magnitude(left_magnitude, right_magnitude);
    }

    let bits = sign | magnitude;
    f32::reinterpret(bits)
}

/// The products `fl(a * b)` of each `a` in `left` and `b` in `right` on the device, through the exact
/// multiply EEDI3's reliability check blends with.
///
/// Every value must be finite.
#[cfg(any(test, feature = "test-support"))]
pub fn eedi3_mul_exact<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    left: &[f32],
    right: &[f32],
) -> Vec<f32> {
    let cases = left.len();

    assert_eq!(right.len(), cases, "every left value needs a right value");

    let operands = left.iter().chain(right);
    for value in operands {
        assert!(
            value.is_finite(),
            "the exact multiply takes finite values, not {value}"
        );
    }

    let left_bytes = f32::as_bytes(left);
    let right_bytes = f32::as_bytes(right);
    let left_handle = client.create_from_slice(left_bytes);
    let right_handle = client.create_from_slice(right_bytes);
    let product_bytes = size_of_val(left);
    let products = client.empty(product_bytes);

    let cubes = dispatch.cube_count(cases);
    let unit = dispatch.cube_dim();

    // Safety: the three buffers were created above with `cases` f32 values each, and each unit reads and
    // writes only its own case.
    unsafe {
        let left_values = ArrayArg::from_raw_parts(left_handle, cases);
        let right_values = ArrayArg::from_raw_parts(right_handle, cases);
        let owned = products.clone();
        let product_values = ArrayArg::from_raw_parts(owned, cases);

        mul_exact_kernel::launch_unchecked::<R>(
            client,
            cubes,
            unit,
            left_values,
            right_values,
            product_values,
        );
    }

    let bytes = client
        .read_one(products)
        .expect("the device should hand back the products");

    f32::from_bytes(&bytes).to_vec()
}

/// One exact product per case.
#[cfg(any(test, feature = "test-support"))]
#[cube(launch_unchecked)]
fn mul_exact_kernel(left: &Array<f32>, right: &Array<f32>, products: &mut Array<f32>) {
    if ABSOLUTE_POS >= products.len() {
        terminate!();
    }

    let left_value = left[ABSOLUTE_POS];
    let right_value = right[ABSOLUTE_POS];
    products[ABSOLUTE_POS] = mul_exact(left_value, right_value);
}
