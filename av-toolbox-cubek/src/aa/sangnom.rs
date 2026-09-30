/// SangNom's direction threshold for `aa` on a plane of `bits` per sample.
///
/// It is `trunc(aa * 21 / 16 * 2^(bits - 8))`, so 48 gives 63 at 8 bits and 252 at 10 bits.
pub fn sangnom_aaf(aa: u32, bits: u32) -> f32 {
    assert!(
        (8..=16).contains(&bits),
        "SangNom takes 8 to 16 bits per sample, not {bits}"
    );
    assert!(aa <= 128, "SangNom's aa is at most 128, not {aa}");

    let scale = 1u32 << (bits - 8);
    let scaled = aa * 21 * scale;
    let threshold = scaled / 16;

    threshold as f32
}
