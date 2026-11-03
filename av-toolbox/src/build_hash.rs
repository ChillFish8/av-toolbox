use std::sync::LazyLock;

const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

static BUILD_HASH: LazyLock<String> = LazyLock::new(compute_build_hash);

/// A key for this build's compiled kernels.
///
/// It is 16 lowercase hex characters and changes whenever the kernel sources or the GPU libraries they use
/// change.
pub fn build_hash() -> &'static str {
    BUILD_HASH.as_str()
}

fn compute_build_hash() -> String {
    let source_hashes = [
        av_toolbox_gpu::SOURCE_HASH,
        av_toolbox_cubek::SOURCE_HASH,
        av_toolbox_masks::SOURCE_HASH,
        av_toolbox_filters::SOURCE_HASH,
    ];

    let mut hash = FNV_OFFSET_BASIS;
    for source_hash in source_hashes {
        hash = fnv1a(hash, source_hash.as_bytes());
    }

    format!("{hash:016x}")
}

fn fnv1a(mut hash: u64, bytes: &[u8]) -> u64 {
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }

    hash
}

#[cfg(test)]
mod tests {
    use super::build_hash;

    #[test]
    fn the_build_hash_is_sixteen_lowercase_hex_chars() {
        let hash = build_hash();

        assert_eq!(hash.len(), 16);
        assert!(
            hash.chars()
                .all(|character| matches!(character, '0'..='9' | 'a'..='f'))
        );
    }
}
