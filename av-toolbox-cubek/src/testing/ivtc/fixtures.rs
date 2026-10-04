use std::path::PathBuf;

const MASKS: &str = include_str!("../fixtures/ivtc/masks.txt");
const LUMA_SAMPLES: usize = 64 * 32;
const CHROMA_SAMPLES: usize = 32 * 16;

/// One of VIVTC's comb masks for a match weave of frame `frame` of clip `clip`, and its mic.
pub struct MaskFixture {
    pub clip: String,
    pub frame: usize,
    pub field: usize,
    pub matched: usize,
    pub mic: u32,
    pub luma: Vec<u8>,
    pub u: Vec<u8>,
    pub v: Vec<u8>,
}

fn mask_path(name: &str) -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("src/testing/fixtures/ivtc/masks");
    path.push(format!("{name}.mask"));
    path
}

fn mask_fixture(line: &str) -> MaskFixture {
    let words: Vec<&str> = line.split_whitespace().collect();
    assert_eq!(words.len(), 7, "a masks.txt line has seven words: {line}");

    let name = format!("{}_{}_{}_{}", words[1], words[2], words[3], words[4]);
    let path = mask_path(&name);
    let bytes = std::fs::read(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    assert_eq!(
        bytes.len(),
        LUMA_SAMPLES + 2 * CHROMA_SAMPLES,
        "{name} has three 8-bit planes"
    );

    let (luma, chroma) = bytes.split_at(LUMA_SAMPLES);
    let (u, v) = chroma.split_at(CHROMA_SAMPLES);

    MaskFixture {
        clip: words[1].to_string(),
        frame: words[2].parse().expect("frame"),
        field: words[3].parse().expect("field"),
        matched: words[4].parse().expect("match"),
        mic: words[6].parse().expect("mic"),
        luma: luma.to_vec(),
        u: u.to_vec(),
        v: v.to_vec(),
    }
}

/// Every comb mask fixture `masks.txt` lists.
pub fn mask_fixtures() -> Vec<MaskFixture> {
    MASKS.lines().map(mask_fixture).collect()
}
