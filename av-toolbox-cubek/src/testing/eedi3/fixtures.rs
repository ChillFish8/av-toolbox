use super::Eedi3Parameters;

/// Builds an [Eedi3Fixture] from the files written for the case `name`, with or without its clips.
macro_rules! eedi3_fixture {
    ($name:literal) => {
        eedi3_fixture!(@build $name, None, None)
    };
    ($name:literal, with_clips) => {
        eedi3_fixture!(
            @build $name,
            Some(include_bytes!(concat!("../fixtures/eedi3/", $name, ".sclip"))),
            Some(include_bytes!(concat!("../fixtures/eedi3/", $name, ".mclip")))
        )
    };
    (@build $name:literal, $sclip:expr, $mclip:expr) => {
        Eedi3Fixture {
            name: $name,
            metadata: include_str!(concat!("../fixtures/eedi3/", $name, ".txt")),
            input: include_bytes!(concat!("../fixtures/eedi3/", $name, ".in")),
            pass0: include_bytes!(concat!("../fixtures/eedi3/", $name, ".pass0")),
            pass1: include_bytes!(concat!("../fixtures/eedi3/", $name, ".pass1")),
            merged: include_bytes!(concat!("../fixtures/eedi3/", $name, ".aa")),
            directions0: include_bytes!(concat!("../fixtures/eedi3/", $name, ".dir0")),
            directions1: include_bytes!(concat!("../fixtures/eedi3/", $name, ".dir1")),
            sclip: $sclip,
            mclip: $mclip,
        }
    };
}

/// One case written by the plugin's own C path, with the third cost term bounded to the row.
///
/// Every plane is little-endian 16-bit samples at the input's size. Direction maps are signed and hold 0
/// on kept rows. `metadata` holds the `width` and `height`, and every field of
/// [Eedi3Parameters](crate::testing::Eedi3Parameters) it was written with.
pub struct Eedi3Fixture {
    pub name: &'static str,
    pub metadata: &'static str,
    pub input: &'static [u8],
    pub pass0: &'static [u8],
    pub pass1: &'static [u8],
    pub merged: &'static [u8],
    pub directions0: &'static [u8],
    pub directions1: &'static [u8],
    pub sclip: Option<&'static [u8]>,
    pub mclip: Option<&'static [u8]>,
}

/// Every EEDI3 field pass fixture.
pub const EEDI3_FIXTURES: [Eedi3Fixture; 13] = [
    eedi3_fixture!("worked"),
    eedi3_fixture!("defaults"),
    eedi3_fixture!("flat"),
    eedi3_fixture!("based", with_clips),
    eedi3_fixture!("ten_bit"),
    eedi3_fixture!("sixteen_bit"),
    eedi3_fixture!("wide_mdis"),
    eedi3_fixture!("mdis_one"),
    eedi3_fixture!("no_check"),
    eedi3_fixture!("tiny"),
    eedi3_fixture!("s2_bound"),
    eedi3_fixture!("check_order"),
    eedi3_fixture!("close_path"),
];

/// An edge between 40 and 200 that moves two columns per row, the plane of the `worked` fixture at any size.
pub fn worked_edge_plane(width: u32, height: u32) -> Vec<f32> {
    let mut plane = Vec::new();
    for row in 0..height {
        for column in 0..width {
            let value = if column > 2 * row { 200.0 } else { 40.0 };
            plane.push(value);
        }
    }

    plane
}

fn sample_pairs(bytes: &[u8]) -> &[[u8; 2]] {
    let (pairs, remainder) = bytes.as_chunks::<2>();
    assert!(
        remainder.is_empty(),
        "a plane of 16-bit samples has an odd {} bytes",
        bytes.len()
    );

    pairs
}

fn decode_plane(bytes: &[u8]) -> Vec<f32> {
    sample_pairs(bytes)
        .iter()
        .map(|pair| {
            let sample = u16::from_le_bytes(*pair);
            f32::from(sample)
        })
        .collect()
}

fn decode_directions(bytes: &[u8]) -> Vec<i32> {
    sample_pairs(bytes)
        .iter()
        .map(|pair| {
            let direction = i16::from_le_bytes(*pair);
            i32::from(direction)
        })
        .collect()
}

fn metadata_text<'a>(metadata: &'a str, key: &str) -> &'a str {
    for line in metadata.lines() {
        let mut parts = line.split_whitespace();
        if parts.next() != Some(key) {
            continue;
        }

        return parts.next().expect("every metadata key has a value");
    }

    panic!("the fixture metadata has no `{key}`");
}

impl Eedi3Fixture {
    /// The fixture called `name`.
    pub fn named(name: &str) -> &'static Self {
        EEDI3_FIXTURES
            .iter()
            .find(|fixture| fixture.name == name)
            .expect("the fixture exists")
    }

    /// The integer value of `key` in the metadata.
    pub fn setting(&self, key: &str) -> u32 {
        let text = metadata_text(self.metadata, key);

        text.parse().expect("the metadata value is an integer")
    }

    /// The value of `key` in the metadata, an argument the plugin read as f32.
    pub fn float_setting(&self, key: &str) -> f32 {
        let text = metadata_text(self.metadata, key);

        text.parse().expect("the metadata value is a number")
    }

    /// The parameters the case was written with.
    pub fn parameters(&self) -> Eedi3Parameters {
        Eedi3Parameters {
            alpha: self.float_setting("alpha"),
            beta: self.float_setting("beta"),
            gamma: self.float_setting("gamma"),
            nrad: self.setting("nrad"),
            mdis: self.setting("mdis"),
            vcheck: self.setting("vcheck"),
            vthresh: [
                self.float_setting("vthresh0"),
                self.float_setting("vthresh1"),
                self.float_setting("vthresh2"),
            ],
            bits: self.setting("bits"),
        }
    }

    /// The input plane as f32 code values.
    pub fn input_plane(&self) -> Vec<f32> {
        decode_plane(self.input)
    }

    /// The field pass keeping the rows of parity `offset`, as f32 code values.
    pub fn pass_plane(&self, offset: u32) -> Vec<f32> {
        match offset {
            0 => decode_plane(self.pass0),
            1 => decode_plane(self.pass1),
            _ => panic!("an EEDI3 field offset is 0 or 1, not {offset}"),
        }
    }

    /// The direction map of the field pass keeping the rows of parity `offset`.
    pub fn directions(&self, offset: u32) -> Vec<i32> {
        match offset {
            0 => decode_directions(self.directions0),
            1 => decode_directions(self.directions1),
            _ => panic!("an EEDI3 field offset is 0 or 1, not {offset}"),
        }
    }

    /// The two passes merged, as f32 code values.
    pub fn merged_plane(&self) -> Vec<f32> {
        decode_plane(self.merged)
    }

    /// The `sclip` plane as f32 code values, when the case has one.
    pub fn sclip_plane(&self) -> Option<Vec<f32>> {
        self.sclip.map(decode_plane)
    }

    /// The 8-bit `mclip` plane as f32 values, when the case has one.
    pub fn mclip_plane(&self) -> Option<Vec<f32>> {
        self.mclip.map(decode_plane)
    }
}
