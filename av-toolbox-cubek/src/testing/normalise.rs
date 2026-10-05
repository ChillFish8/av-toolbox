/// Stretches TV-range luma code values of `bits` bits to between 0 and 1 on the CPU, scaled by
/// `256 / 255`, mirroring [normalise](crate::motion::normalise).
pub fn reference_normalise(samples: &[f32], bits: u32) -> Vec<f32> {
    let scale = (1u32 << (bits - 8)) as f32;
    let black = 16.0 * scale;
    let inverse_range = 1.0 / (219.0 * scale);
    let stretch = 256.0f32 / 255.0;

    samples
        .iter()
        .map(|sample| {
            let stretched = (sample - black) * inverse_range;
            let clamped = stretched.clamp(0.0, 1.0);

            clamped * stretch
        })
        .collect()
}
