use std::path::PathBuf;

const MASKS: &str = include_str!("../fixtures/ivtc/masks.txt");
const MAPS: &str = include_str!("../fixtures/ivtc/maps.txt");
const EXCLUDED: &str = include_str!("../fixtures/ivtc/excluded.txt");
const DECISIONS: &str = include_str!("../fixtures/ivtc/decisions.txt");
const DECIMATE_METRICS: &str = include_str!("../fixtures/ivtc/decimate_metrics.txt");
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

/// The luma, U and V planes of fixture file `<directory>/<name>.<extension>`.
fn read_planes(directory: &str, name: &str, extension: &str) -> [Vec<u8>; 3] {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("src/testing/fixtures/ivtc");
    path.push(directory);
    path.push(format!("{name}.{extension}"));

    let bytes = std::fs::read(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    assert_eq!(
        bytes.len(),
        LUMA_SAMPLES + 2 * CHROMA_SAMPLES,
        "{name} has three 8-bit planes"
    );

    let (luma, chroma) = bytes.split_at(LUMA_SAMPLES);
    let (u, v) = chroma.split_at(CHROMA_SAMPLES);

    [luma.to_vec(), u.to_vec(), v.to_vec()]
}

fn mask_fixture(line: &str) -> MaskFixture {
    let words: Vec<&str> = line.split_whitespace().collect();
    assert_eq!(words.len(), 7, "a masks.txt line has seven words: {line}");

    let name = format!("{}_{}_{}_{}", words[1], words[2], words[3], words[4]);
    let [luma, u, v] = read_planes("masks", &name, "mask");

    MaskFixture {
        clip: words[1].to_string(),
        frame: words[2].parse().expect("frame"),
        field: words[3].parse().expect("field"),
        matched: words[4].parse().expect("match"),
        mic: words[6].parse().expect("mic"),
        luma,
        u,
        v,
    }
}

/// Every comb mask fixture `masks.txt` lists.
pub fn mask_fixtures() -> Vec<MaskFixture> {
    MASKS.lines().map(mask_fixture).collect()
}

/// One of VIVTC's field compare maps for frame `frame` of clip `clip`, rows `exclude_from` to `exclude_to`
/// left out of its sums, and the match the compare chose.
pub struct MapFixture {
    pub clip: String,
    pub frame: usize,
    pub field: u32,
    pub first: u32,
    pub second: u32,
    pub exclude_from: u32,
    pub exclude_to: u32,
    pub choice: u32,
    pub luma: Vec<u8>,
    pub u: Vec<u8>,
    pub v: Vec<u8>,
}

fn map_fixture(line: &str, directory: &str, exclusion: (u32, u32)) -> MapFixture {
    let words: Vec<&str> = line.split_whitespace().collect();
    assert_eq!(words.len(), 8, "a map line has eight words: {line}");

    let name = format!("{}_{}_{}_{}_{}", words[1], words[2], words[3], words[4], words[5]);
    let [luma, u, v] = read_planes(directory, &name, "map");
    let (exclude_from, exclude_to) = exclusion;

    MapFixture {
        clip: words[1].to_string(),
        frame: words[2].parse().expect("frame"),
        field: words[3].parse().expect("field"),
        first: words[4].parse().expect("first match"),
        second: words[5].parse().expect("second match"),
        exclude_from,
        exclude_to,
        choice: words[7].parse().expect("choice"),
        luma,
        u,
        v,
    }
}

/// Every field compare map `maps.txt` lists, which exclude no rows.
pub fn map_fixtures() -> Vec<MapFixture> {
    MAPS.lines()
        .map(|line| map_fixture(line, "maps", (0, 0)))
        .collect()
}

/// Every field compare map `excluded.txt` lists, which exclude luma rows 8 to 15 from their sums.
pub fn excluded_map_fixtures() -> Vec<MapFixture> {
    EXCLUDED
        .lines()
        .map(|line| map_fixture(line, "excluded", (8, 15)))
        .collect()
}

/// One `vfmProcess` run on frame `frame` of clip `clip`, and what VFM decided.
///
/// `opposite` is true when VFM's field is the opposite of the clip's order. `mics` holds every match's mic by
/// raw match number.
pub struct DecisionFixture {
    pub clip: String,
    pub frame: usize,
    pub mode: u32,
    pub micmatch: u32,
    pub opposite: bool,
    pub matched: usize,
    pub mics: [u32; 5],
    pub combed: bool,
    pub scene_change: bool,
}

fn flag(word: &str) -> bool {
    match word {
        "0" => false,
        "1" => true,
        _ => panic!("a fixture flag is 0 or 1, not {word}"),
    }
}

fn opposite_field(word: &str) -> bool {
    match word {
        "same" => false,
        "opposite" => true,
        _ => panic!("a decisions.txt field is same or opposite, not {word}"),
    }
}

fn decision_fixture(line: &str) -> DecisionFixture {
    let words: Vec<&str> = line.split_whitespace().collect();
    assert_eq!(words.len(), 21, "a decisions.txt line has 21 words: {line}");

    let mics = std::array::from_fn(|index| words[12 + index].parse().expect("mic"));

    DecisionFixture {
        clip: words[1].to_string(),
        frame: words[2].parse().expect("frame"),
        mode: words[4].parse().expect("mode"),
        micmatch: words[6].parse().expect("micmatch"),
        opposite: opposite_field(words[8]),
        matched: words[10].parse().expect("match"),
        mics,
        combed: flag(words[18]),
        scene_change: flag(words[20]),
    }
}

/// Every `vfmProcess` run `decisions.txt` lists.
pub fn vfm_decisions() -> Vec<DecisionFixture> {
    DECISIONS.lines().map(decision_fixture).collect()
}

/// VDecimate's `calcMetric` of frame `frame` of clip `clip` against the frame before it.
///
/// `matched` is true when VDecimate read VFM's output of the clip, and false when it read the clip itself.
pub struct DecimateMetricFixture {
    pub clip: String,
    pub matched: bool,
    pub block_x: usize,
    pub block_y: usize,
    pub chroma: bool,
    pub frame: usize,
    pub max_block: i64,
    pub total: i64,
}

fn decimate_source(word: &str) -> bool {
    match word {
        "raw" => false,
        "matched" => true,
        _ => panic!("a decimate_metrics.txt source is raw or matched, not {word}"),
    }
}

fn decimate_metric_fixture(line: &str) -> DecimateMetricFixture {
    let words: Vec<&str> = line.split_whitespace().collect();
    assert_eq!(
        words.len(),
        11,
        "a decimate_metrics.txt line has 11 words: {line}"
    );

    DecimateMetricFixture {
        clip: words[1].to_string(),
        matched: decimate_source(words[2]),
        block_x: words[3].parse().expect("blockx"),
        block_y: words[4].parse().expect("blocky"),
        chroma: flag(words[5]),
        frame: words[6].parse().expect("frame"),
        max_block: words[8].parse().expect("maxbdiff"),
        total: words[10].parse().expect("totdiff"),
    }
}

/// Every VDecimate metric `decimate_metrics.txt` lists.
pub fn decimate_metric_fixtures() -> Vec<DecimateMetricFixture> {
    DECIMATE_METRICS.lines().map(decimate_metric_fixture).collect()
}
