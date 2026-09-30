use super::Eedi2Parameters;

/// Builds an [Eedi2Fixture] from the five files written for the case `name`.
macro_rules! eedi2_fixture {
    ($name:literal) => {
        Eedi2Fixture {
            name: $name,
            metadata: include_str!(concat!("../fixtures/eedi2/", $name, ".txt")),
            input: include_bytes!(concat!("../fixtures/eedi2/", $name, ".in")),
            pass0: include_bytes!(concat!("../fixtures/eedi2/", $name, ".pass0")),
            pass1: include_bytes!(concat!("../fixtures/eedi2/", $name, ".pass1")),
            merged: include_bytes!(concat!("../fixtures/eedi2/", $name, ".aa")),
        }
    };
}

/// One case written by the plugin's own core with both search ranges clamped to the row.
///
/// Every plane is little-endian u16 samples. `metadata` holds the `width` and `height`, and every field of
/// [Eedi2Parameters](crate::testing::Eedi2Parameters) it was written with.
pub struct Eedi2Fixture {
    pub name: &'static str,
    pub metadata: &'static str,
    pub input: &'static [u8],
    pub pass0: &'static [u8],
    pub pass1: &'static [u8],
    pub merged: &'static [u8],
}

/// Every EEDI2 field pass fixture.
pub const EEDI2_FIXTURES: [Eedi2Fixture; 10] = [
    eedi2_fixture!("worked"),
    eedi2_fixture!("diagonal_pp0"),
    eedi2_fixture!("diagonal_pp1"),
    eedi2_fixture!("ten_bit"),
    eedi2_fixture!("sixteen_bit"),
    eedi2_fixture!("chroma"),
    eedi2_fixture!("negative_lim"),
    eedi2_fixture!("left_dependency"),
    eedi2_fixture!("settings"),
    eedi2_fixture!("arguments"),
];

pub(super) fn decode_plane(bytes: &[u8]) -> Vec<f32> {
    let (pairs, remainder) = bytes.as_chunks::<2>();
    assert!(
        remainder.is_empty(),
        "a plane of u16 samples has an odd {} bytes",
        bytes.len()
    );

    pairs
        .iter()
        .map(|pair| {
            let sample = u16::from_le_bytes(*pair);
            f32::from(sample)
        })
        .collect()
}

fn metadata_value(metadata: &str, key: &str) -> u32 {
    for line in metadata.lines() {
        let mut parts = line.split_whitespace();
        if parts.next() != Some(key) {
            continue;
        }

        let value = parts.next().expect("every metadata key has a value");
        return value.parse().expect("every metadata value is an integer");
    }

    panic!("the fixture metadata has no `{key}`");
}

impl Eedi2Fixture {
    /// The value of `key` in the metadata.
    pub fn setting(&self, key: &str) -> u32 {
        metadata_value(self.metadata, key)
    }

    /// The parameters the case was written with.
    pub fn parameters(&self) -> Eedi2Parameters {
        Eedi2Parameters {
            mthresh: self.setting("mthresh"),
            lthresh: self.setting("lthresh"),
            vthresh: self.setting("vthresh"),
            estr: self.setting("estr"),
            dstr: self.setting("dstr"),
            maxd: self.setting("maxd"),
            nt: self.setting("nt"),
            pp: self.setting("pp"),
            bits: self.setting("bits"),
            ssw: self.setting("ssw"),
        }
    }

    /// The input plane as f32 code values.
    pub fn input_plane(&self) -> Vec<f32> {
        decode_plane(self.input)
    }

    /// The field pass keeping the even rows, as f32 code values.
    pub fn pass0_plane(&self) -> Vec<f32> {
        decode_plane(self.pass0)
    }

    /// The field pass keeping the odd rows, as f32 code values.
    pub fn pass1_plane(&self) -> Vec<f32> {
        decode_plane(self.pass1)
    }

    /// The two passes merged, as f32 code values.
    pub fn merged_plane(&self) -> Vec<f32> {
        decode_plane(self.merged)
    }
}
