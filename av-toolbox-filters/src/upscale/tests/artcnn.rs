use av_toolbox_core::{FilterDefinition, Parallelism, StageDefinition, StageError};
use av_toolbox_cubek::artcnn::{BodyPath, CnnError, Kernels, Precision};
use av_toolbox_gpu::{
    Capabilities,
    ColourInfo,
    Depth as BitDepth,
    ElementType,
    FormatSet,
    FrameFormat,
    FrameLayout,
    MaskFormat,
    Offered,
    SampleFormat,
    StreamFormat,
    Subsampling,
    default_accelerators,
};
use av_toolbox_models::{Located, Store};
use serde_json::json;

use crate::sample_checks::capabilities_computing;
use crate::upscale::Artcnn;
use crate::upscale::artcnn::{
    ArtcnnArgs,
    KERNEL_ENV,
    ModelName,
    PrecisionName,
    allocation_cap,
    body_path,
    failed_convolution_and_im2col,
    fallbacks,
    frame_tile,
    kernels_from_value,
    locate,
    warmed_path,
};

const STAGE: &str = "upscale/artcnn";

fn args(value: serde_json::Value) -> ArtcnnArgs {
    serde_json::from_value(value).expect("parses")
}

fn format(width: u32, height: u32, layout: FrameLayout, sample: SampleFormat) -> FrameFormat {
    FrameFormat {
        width,
        height,
        layout,
        sample,
        colour: ColourInfo::default(),
        frame_rate: None,
        field_order: None,
    }
}

fn capabilities_multiplying(combinations: &[(ElementType, ElementType)]) -> Capabilities {
    let accelerator = default_accelerators()[0];
    let combinations = combinations.iter().copied();

    Capabilities::builder(accelerator)
        .plane_size(32, 32)
        .max_units_per_cube(1024)
        .max_cube_count(65535, 65535, 65535)
        .cooperative_matrix(combinations)
        .build()
}

fn capabilities() -> Capabilities {
    capabilities_computing(&[ElementType::F32, ElementType::F16])
}

fn produced(
    capabilities: &Capabilities,
    input: FrameFormat,
    value: serde_json::Value,
) -> Result<FrameFormat, StageError> {
    let arguments = args(value);
    let chosen = [Some(StreamFormat::Frames(input))];
    let outputs = Artcnn::produces(capabilities, &arguments, &chosen)?;
    let output = outputs.template.with_sample(input.sample);
    let frames = output.as_frames().expect("frames");

    Ok(*frames)
}

fn accepted(offered: StreamFormat) -> Result<FormatSet, StageError> {
    let capabilities = capabilities();
    let arguments = args(json!({}));
    let inputs = [Offered::Unknown];

    Artcnn::accepts_input(&capabilities, &arguments, 0, offered, &inputs)
}

#[test]
fn the_defaults_are_r8f64_at_f16() {
    let parsed = args(json!({}));

    assert_eq!(parsed.model, ModelName::R8f64);
    assert_eq!(parsed.precision, PrecisionName::F16);
}

#[test]
fn the_model_and_precision_parse_from_their_names() {
    let parsed = args(json!({ "model": "r16f96", "precision": "f32" }));

    assert_eq!(parsed.model, ModelName::R16f96);
    assert_eq!(parsed.precision, PrecisionName::F32);
}

#[test]
fn an_unknown_model_is_refused() {
    let value = json!({ "model": "r4f32" });

    let error = serde_json::from_value::<ArtcnnArgs>(value).expect_err("r4f32 is no model");
    let message = error.to_string();

    assert!(message.contains("r4f32"), "{message}");
}

#[test]
fn an_unknown_argument_is_refused() {
    let value = json!({ "modle": "r8f64" });

    let error = serde_json::from_value::<ArtcnnArgs>(value).expect_err("modle is no argument");
    let message = error.to_string();

    assert!(message.contains("modle"), "{message}");
}

#[test]
fn failing_on_both_paths_names_both_causes() {
    let convolution = CnnError::Device("sync lost".to_string());
    let im2col = CnnError::Matmul("no kernel".to_string());

    let error = failed_convolution_and_im2col(convolution, im2col);
    let message = error.to_string();

    assert!(message.contains(STAGE), "{message}");
    assert!(message.contains("convolution"), "{message}");
    assert!(message.contains("sync lost"), "{message}");
    assert!(message.contains("im2col"), "{message}");
    assert!(message.contains("no kernel"), "{message}");
}

#[test]
fn grey_and_every_yuv_subsampling_double() {
    let capabilities = capabilities();
    let ten_bit = SampleFormat::U16 { depth: BitDepth::TEN };
    let layouts = [
        FrameLayout::Grey,
        FrameLayout::Yuv {
            subsampling: Subsampling::Yuv420,
        },
        FrameLayout::Yuv {
            subsampling: Subsampling::Yuv422,
        },
        FrameLayout::Yuv {
            subsampling: Subsampling::Yuv444,
        },
    ];
    let samples = [SampleFormat::U8, ten_bit];
    let sizes = [(16, 8), (15, 9)];

    for layout in layouts {
        for sample in samples {
            for (width, height) in sizes {
                let input = format(width, height, layout, sample);

                let output = produced(&capabilities, input, json!({})).expect("upscalable");

                let expected = FrameFormat {
                    width: width * 2,
                    height: height * 2,
                    ..input
                };
                assert_eq!(output, expected, "{layout} {sample} {width}x{height}");
            }
        }
    }
}

#[test]
fn rgb_and_chroma_layouts_are_refused() {
    let capabilities = capabilities();
    let chroma = FrameLayout::Chroma {
        subsampling: Subsampling::Yuv420,
    };
    let layouts = [(FrameLayout::Rgb, "rgb"), (chroma, "chroma")];

    for (layout, name) in layouts {
        let input = format(16, 8, layout, SampleFormat::U8);
        let offered = StreamFormat::Frames(input);

        let accept_error = accepted(offered).expect_err("the layout is refused");
        let produce_error = produced(&capabilities, input, json!({})).expect_err("the layout is refused");

        for message in [accept_error.to_string(), produce_error.to_string()] {
            assert!(message.contains(STAGE), "{message}");
            assert!(message.contains(name), "{message}");
        }
    }
}

#[test]
fn floats_are_left_out_and_a_mask_is_refused() {
    let capabilities = capabilities();
    let float_input = format(16, 8, FrameLayout::Grey, SampleFormat::F32);
    let floats = StreamFormat::Frames(float_input);
    let mask = StreamFormat::Mask(MaskFormat {
        frame_rate: None,
        width: 16,
        height: 8,
        sample: SampleFormat::U8,
        chroma: None,
    });

    let float_set = accepted(floats).expect("frames are readable");
    let mask_error = accepted(mask).expect_err("a mask is refused");
    let float_error = produced(&capabilities, float_input, json!({})).expect_err("floats are refused");

    let mask_message = mask_error.to_string();
    let float_message = float_error.to_string();
    let float_name = SampleFormat::F32.to_string();
    assert!(float_set.is_empty());
    assert!(mask_message.contains(STAGE), "{mask_message}");
    assert!(mask_message.contains("mask"), "{mask_message}");
    assert!(float_message.contains(STAGE), "{float_message}");
    assert!(float_message.contains(&float_name), "{float_message}");
}

#[test]
fn f16_without_f16_arithmetic_is_refused_suggesting_f32() {
    let capabilities = capabilities_computing(&[ElementType::F32]);
    let input = format(16, 8, FrameLayout::Grey, SampleFormat::U8);

    let error = produced(&capabilities, input, json!({ "precision": "f16" })).expect_err("no f16 arithmetic");
    let with_f32 = produced(&capabilities, input, json!({ "precision": "f32" }));

    let message = error.to_string();
    assert!(message.contains(STAGE), "{message}");
    assert!(message.contains("precision"), "{message}");
    assert!(message.contains("f16"), "{message}");
    assert!(message.contains("\"f32\""), "{message}");
    assert!(with_f32.is_ok());
}

#[test]
fn a_missing_model_names_the_pull_command() {
    let root = tempfile::tempdir().expect("a temp dir");
    let path = root.path().to_path_buf();
    let store = Store::new(path, "v0.1.0");

    let error = locate(&store, ModelName::R16f96).expect_err("R16F96 is not installed");
    let message = error.to_string();

    assert!(message.contains(STAGE), "{message}");
    assert!(message.contains("model"), "{message}");
    assert!(message.contains("r16f96"), "{message}");
    assert!(message.contains("ArtCNN_R16F96.bin"), "{message}");
    assert!(message.contains("av-toolbox pull artcnn"), "{message}");
}

#[test]
fn the_builtin_model_is_found_without_a_store() {
    let root = tempfile::tempdir().expect("a temp dir");
    let path = root.path().to_path_buf();
    let store = Store::new(path, "v0.1.0");

    let located = locate(&store, ModelName::R8f64).expect("R8F64 is built in");

    assert!(matches!(located, Located::Builtin(_)));
}

#[test]
fn the_stage_runs_as_a_single_instance() {
    let arguments = args(json!({}));

    assert_eq!(Artcnn::parallelism(&arguments), Parallelism::Stream);
}

#[test]
fn the_convolution_path_is_chosen_when_the_device_offers_it() {
    let f16_only = capabilities_multiplying(&[(ElementType::F16, ElementType::F32)]);
    let none = capabilities_multiplying(&[]);
    let both = capabilities_multiplying(&[
        (ElementType::F16, ElementType::F32),
        (ElementType::F32, ElementType::F32),
    ]);

    assert_eq!(body_path(&f16_only, Precision::F16), BodyPath::Convolution);
    assert_eq!(body_path(&f16_only, Precision::F32), BodyPath::Im2col);
    assert_eq!(body_path(&none, Precision::F16), BodyPath::Im2col);
    assert_eq!(body_path(&none, Precision::F32), BodyPath::Im2col);
    assert_eq!(body_path(&both, Precision::F16), BodyPath::Convolution);
    assert_eq!(body_path(&both, Precision::F32), BodyPath::Convolution);
}

#[test]
fn an_unset_allocation_limit_is_no_cap() {
    let unset = capabilities();
    let accelerator = default_accelerators()[0];
    let limited = Capabilities::builder(accelerator)
        .plane_size(32, 32)
        .max_units_per_cube(1024)
        .max_cube_count(65535, 65535, 65535)
        .max_allocation(1 << 20)
        .build();

    assert_eq!(allocation_cap(&unset), usize::MAX);
    assert_eq!(allocation_cap(&limited), 1 << 20);
}

#[test]
fn an_f16_frame_that_fits_runs_as_one_tile_of_its_longer_side() {
    let wide = frame_tile(
        64,
        27,
        Precision::F16,
        BodyPath::Convolution,
        usize::MAX,
        1920,
        1092,
    );
    let tall = frame_tile(
        64,
        27,
        Precision::F16,
        BodyPath::Convolution,
        usize::MAX,
        1092,
        1920,
    );

    assert_eq!(wide, 1920);
    assert_eq!(tall, 1920);
}

#[test]
fn an_f16_frame_one_row_too_tall_splits_into_square_tiles() {
    let tile = frame_tile(
        64,
        27,
        Precision::F16,
        BodyPath::Convolution,
        usize::MAX,
        1920,
        1093,
    );

    assert_eq!(tile, 1394);
}

#[test]
fn an_f32_1080p_frame_on_the_im2col_path_splits_into_square_tiles() {
    let tile = frame_tile(64, 27, Precision::F32, BodyPath::Im2col, usize::MAX, 1920, 1080);

    assert_eq!(tile, 514);
}

#[test]
fn a_frame_too_large_for_the_allocation_limit_splits_into_square_tiles() {
    // conv0's 128 bytes a sample over 1920x1080 samples is one byte more than this limit.
    let limit = 265_420_799;
    let tile = frame_tile(64, 27, Precision::F16, BodyPath::Convolution, limit, 1920, 1080);

    // 265,420,799 / 128 = 2,073,599 samples, and sqrt(2,073,599) = 1439 a side haloed, so 1439 - 54.
    assert_eq!(tile, 1385);
}

/// Warms up from `chosen` with every path in `failing` failing, giving the outcome and the paths tried in
/// turn.
fn warm_up_failing(
    chosen: BodyPath,
    failing: &[BodyPath],
) -> (Result<(BodyPath, u32), StageError>, Vec<BodyPath>) {
    let mut tried = Vec::new();

    let warmed = warmed_path(STAGE, chosen, |path| {
        tried.push(path);

        if failing.contains(&path) {
            let name = format!("{path:?} failed");
            return Err(CnnError::Device(name));
        }

        Ok(100)
    });

    (warmed, tried)
}

#[test]
fn the_fused_convolution_falls_back_to_the_unfused_convolution_then_im2col() {
    let from_fused = fallbacks(BodyPath::Convolution);
    let from_unfused = fallbacks(BodyPath::UnfusedConvolution);
    let from_im2col = fallbacks(BodyPath::Im2col);

    let fused_ladder = [
        BodyPath::Convolution,
        BodyPath::UnfusedConvolution,
        BodyPath::Im2col,
    ];
    assert_eq!(from_fused, fused_ladder);
    assert_eq!(from_unfused, [BodyPath::UnfusedConvolution, BodyPath::Im2col]);
    assert_eq!(from_im2col, [BodyPath::Im2col]);
}

#[test]
fn a_fused_warm_up_that_succeeds_tries_no_other_path() {
    let (warmed, tried) = warm_up_failing(BodyPath::Convolution, &[]);

    let chosen = warmed.expect("the fused path warms up");
    assert_eq!(chosen, (BodyPath::Convolution, 100));
    assert_eq!(tried, [BodyPath::Convolution]);
}

#[test]
fn a_failed_fused_warm_up_falls_back_to_the_unfused_convolution() {
    let (warmed, tried) = warm_up_failing(BodyPath::Convolution, &[BodyPath::Convolution]);

    let chosen = warmed.expect("the unfused path warms up");
    assert_eq!(chosen, (BodyPath::UnfusedConvolution, 100));
    assert_eq!(tried, [BodyPath::Convolution, BodyPath::UnfusedConvolution]);
}

#[test]
fn failed_warm_ups_on_both_convolution_paths_fall_back_to_im2col() {
    let failing = [BodyPath::Convolution, BodyPath::UnfusedConvolution];

    let (warmed, tried) = warm_up_failing(BodyPath::Convolution, &failing);

    let chosen = warmed.expect("the im2col path warms up");
    let ladder = [
        BodyPath::Convolution,
        BodyPath::UnfusedConvolution,
        BodyPath::Im2col,
    ];
    assert_eq!(chosen, (BodyPath::Im2col, 100));
    assert_eq!(tried, ladder);
}

#[test]
fn a_warm_up_failing_on_every_path_names_each_cause() {
    let failing = [
        BodyPath::Convolution,
        BodyPath::UnfusedConvolution,
        BodyPath::Im2col,
    ];

    let (warmed, tried) = warm_up_failing(BodyPath::Convolution, &failing);

    let error = warmed.expect_err("no path warms up");
    let message = error.to_string();
    assert_eq!(tried, failing);
    assert!(message.contains(STAGE), "{message}");
    assert!(message.contains("on the fused convolution path ("), "{message}");
    assert!(
        message.contains(": Convolution failed), on the unfused"),
        "{message}"
    );
    assert!(
        message.contains(": UnfusedConvolution failed) or on the im2col"),
        "{message}"
    );
    assert!(message.contains(": Im2col failed)"), "{message}");
}

#[test]
fn a_failed_im2col_warm_up_names_its_one_cause() {
    let (warmed, tried) = warm_up_failing(BodyPath::Im2col, &[BodyPath::Im2col]);

    let error = warmed.expect_err("im2col does not warm up");
    let message = error.to_string();
    assert_eq!(tried, [BodyPath::Im2col]);
    assert!(
        message.contains("could not warm up on the im2col path"),
        "{message}"
    );
    assert!(message.contains("Im2col failed"), "{message}");
}

#[test]
fn the_kernel_variable_pins_the_fixed_kernel_and_otherwise_leaves_autotune_on() {
    let tuned = [None, Some(""), Some("  ")];
    let fixed = ["fixed", "FIXED", " fixed "];

    for value in tuned {
        let kernels = kernels_from_value(STAGE, value).expect("an unset variable is valid");

        assert_eq!(kernels, Kernels::Tuned, "{value:?}");
    }

    for value in fixed {
        let kernels = kernels_from_value(STAGE, Some(value)).expect("fixed is valid");

        assert_eq!(kernels, Kernels::Fixed, "{value:?}");
    }
}

#[test]
fn an_unknown_kernel_value_is_refused_naming_the_variable() {
    let error = kernels_from_value(STAGE, Some("fastest")).expect_err("fastest is no kernel");
    let message = error.to_string();

    assert!(message.contains(STAGE), "{message}");
    assert!(message.contains(KERNEL_ENV), "{message}");
    assert!(message.contains("\"fastest\""), "{message}");
    assert!(message.contains("\"fixed\""), "{message}");
    assert!(message.contains("unset"), "{message}");
}

#[cfg(all(feature = "gpu-tests", any(feature = "vulkan", feature = "metal")))]
mod device_tests {
    use av_toolbox_cubek::DevicePlane;
    use av_toolbox_cubek::artcnn::{
        Candidate,
        LayerRun,
        Model,
        Weights,
        fixed_candidate,
        offered_candidates,
        tiles,
    };
    use av_toolbox_cubek::testing::{failing_kernels, reference_artcnn};
    use av_toolbox_gpu::{Backend, BuildContext, BuildTask, Runtime, test_device};
    use cubecl::prelude::CubeElement;

    use super::*;
    use crate::artcnn::Supersampler;

    /// Loads R8F64 at f16 for a `width` by `height` plane, giving the path and tile it chose.
    struct Loaded {
        width: u32,
        height: u32,
    }

    impl BuildTask for Loaded {
        type Output = (BodyPath, u32);

        fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
            let supersampler = load_r8f64(build, self.width, self.height);

            (supersampler.path(), supersampler.tile())
        }
    }

    /// Loads R8F64 at f16 for a 1080p plane, giving the path the device offers and the path that warmed up.
    struct LoadedPath;

    impl BuildTask for LoadedPath {
        type Output = (BodyPath, BodyPath);

        fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
            let offered = body_path(build.capabilities(), Precision::F16);
            let supersampler = load_r8f64(build, 1920, 1080);

            (offered, supersampler.path())
        }
    }

    fn load_r8f64<R: Runtime>(build: &BuildContext<R>, width: u32, height: u32) -> Supersampler {
        let loaded = Supersampler::load(STAGE, build, ModelName::R8f64, PrecisionName::F16, width, height);

        loaded.expect("R8F64 loads")
    }

    fn loaded_tile(width: u32, height: u32) -> Option<u32> {
        let backend = Backend::open_default(test_device()).expect("a backend should open");
        let task = Loaded { width, height };

        let (path, tile) = backend.build(task);
        println!("{width}x{height}: {path:?} at tile {tile}");

        (path == BodyPath::Convolution).then_some(tile)
    }

    #[test]
    fn the_loaded_supersampler_runs_a_1080p_frame_as_one_tile() {
        let Some(tile) = loaded_tile(1920, 1080) else {
            return;
        };

        assert_eq!(tile, 1920);
    }

    #[test]
    fn the_loaded_supersampler_splits_a_frame_one_row_too_tall() {
        let Some(tile) = loaded_tile(1920, 1093) else {
            return;
        };

        assert_eq!(tile, 1394);
    }

    #[test]
    fn the_loaded_supersampler_takes_the_fused_path_where_the_device_offers_it() {
        let backend = Backend::open_default(test_device()).expect("a backend should open");

        let (offered, loaded) = backend.build(LoadedPath);

        println!("offered {offered:?}, loaded {loaded:?}");
        if offered != BodyPath::Convolution {
            return;
        }

        assert_eq!(loaded, BodyPath::Convolution);
    }

    /// The number of body layers R8F64 runs on one tile: three in each of its eight blocks, then the join.
    const R8F64_BODY_LAYERS: usize = 25;

    /// R8F64 at f16 loaded with `kernels`, or by [Supersampler::load](crate::artcnn::Supersampler::load)
    /// when `kernels` is `None`, for a `width` by `height` plane.
    struct LoadedKernels {
        width: u32,
        height: u32,
        kernels: Option<Kernels>,
    }

    /// What a load chose, beside what the device offers.
    #[derive(Debug)]
    struct KernelsLoaded {
        offered: BodyPath,
        loaded: BodyPath,
        kernels: Kernels,
        layers: Vec<LayerRun>,
        tile: u32,
        /// The candidate pinned kernels run on this device.
        fixed: Candidate,
        /// The model the network was uploaded as.
        model: Option<Model>,
        /// The candidates the device offers R8F64.
        candidates: Vec<Candidate>,
    }

    impl BuildTask for LoadedKernels {
        type Output = KernelsLoaded;

        fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
            let offered = body_path(build.capabilities(), Precision::F16);
            let properties = build.client().properties();
            let fixed = fixed_candidate(properties, 64);
            let candidates = offered_candidates(properties, 64, Some(Model::R8F64));
            let model = ModelName::R8f64;
            let precision = PrecisionName::F16;
            let loaded = match self.kernels {
                Some(kernels) => {
                    Supersampler::load_with(STAGE, build, model, precision, self.width, self.height, kernels)
                },
                None => Supersampler::load(STAGE, build, model, precision, self.width, self.height),
            };
            let supersampler = loaded.expect("R8F64 loads");

            KernelsLoaded {
                offered,
                loaded: supersampler.path(),
                kernels: supersampler.kernels().clone(),
                layers: supersampler.layers().to_vec(),
                tile: supersampler.tile(),
                fixed,
                model: supersampler.model(),
                candidates,
            }
        }
    }

    fn load_kernels(width: u32, height: u32, kernels: Option<Kernels>) -> KernelsLoaded {
        let backend = Backend::open_default(test_device()).expect("a backend should open");
        let task = LoadedKernels {
            width,
            height,
            kernels,
        };

        let loaded = backend.build(task);
        println!("{width}x{height}: {loaded:?}");

        loaded
    }

    #[test]
    fn the_loaded_supersampler_tunes_its_body_convolutions_where_the_device_offers_it() {
        let loaded = load_kernels(1920, 1080, None);

        if loaded.offered != BodyPath::Convolution {
            return;
        }

        assert_eq!(loaded.loaded, BodyPath::Convolution);
        assert_eq!(loaded.kernels, Kernels::Tuned);
        assert_eq!(loaded.model, Some(Model::R8F64));

        let winograd_offered = loaded
            .candidates
            .iter()
            .any(|candidate| matches!(candidate, Candidate::Winograd(_)));
        // The halo moves each tile's edges but not how many tiles there are.
        let frame_tiles = tiles(1920, 1080, loaded.tile, 0);
        assert_eq!(loaded.layers.len(), R8F64_BODY_LAYERS * frame_tiles.len());
        for layer in &loaded.layers {
            assert!(layer.fused, "{layer:?}");

            let candidate = layer.candidate.expect("a body layer took the tuned path");
            assert!(
                loaded.candidates.contains(&candidate),
                "{candidate:?} is not among the candidates offered to R8F64"
            );
            assert_eq!(layer.winograd_offered, winograd_offered, "{layer:?}");
        }
    }

    #[test]
    fn a_pinned_supersampler_runs_every_body_convolution_on_the_fixed_candidate() {
        let pinned = kernels_from_value(STAGE, Some("fixed")).expect("fixed is valid");

        let loaded = load_kernels(64, 48, Some(pinned));

        if loaded.offered != BodyPath::Convolution {
            return;
        }

        assert_eq!(loaded.loaded, BodyPath::Convolution);
        assert_eq!(loaded.layers.len(), R8F64_BODY_LAYERS);
        for layer in &loaded.layers {
            assert_eq!(layer.candidate, Some(loaded.fixed), "{layer:?}");
        }
    }

    #[test]
    fn a_supersampler_whose_every_candidate_fails_falls_back_to_the_unfused_convolution() {
        let loaded = load_kernels(24, 20, Some(failing_kernels()));

        if loaded.offered != BodyPath::Convolution {
            return;
        }

        assert_eq!(loaded.loaded, BodyPath::UnfusedConvolution);
        for layer in &loaded.layers {
            assert!(!layer.fused, "{layer:?}");
            assert_eq!(layer.candidate, None, "{layer:?}");
        }
    }

    /// The code values the supersampler's frame runs on, up to this peak.
    const PEAK: f32 = 65535.0;

    /// A plane of `width` by `height` in `0..=1`, with ripples, a ramp and a step.
    fn rippled(width: u32, height: u32) -> Vec<f32> {
        let mut plane = Vec::new();

        for row in 0..height {
            for column in 0..width {
                let ripple = (column as f32 * 0.9).cos() * (row as f32 * 0.6).sin() * 0.3;
                let ramp = column as f32 / width as f32 * 0.4;
                let step = if row > height / 2 { 0.2 } else { 0.0 };
                plane.push((0.3 + ripple + ramp + step).clamp(0.0, 1.0));
            }
        }

        plane
    }

    /// Loads R8F64 at f16 for `plane`, `width` by `height`, and doubles it, giving the output in `0..=1`,
    /// whether the device offers the convolution path, and how the body layers ran.
    struct LoadedFrame {
        plane: Vec<f32>,
        width: u32,
        height: u32,
    }

    struct FrameRun {
        output: Vec<f32>,
        offered: BodyPath,
        layers: Vec<LayerRun>,
    }

    impl BuildTask for LoadedFrame {
        type Output = FrameRun;

        fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
            let client = build.client();
            let offered = body_path(build.capabilities(), Precision::F16);
            let loaded = Supersampler::load(
                STAGE,
                build,
                ModelName::R8f64,
                PrecisionName::F16,
                self.width,
                self.height,
            );
            let supersampler = loaded.expect("R8F64 loads");
            let context = build.frame_context();
            let codes: Vec<f32> = self.plane.iter().map(|value| value * PEAK).collect();
            let output_bytes = (self.width * self.height * 4) as usize * size_of::<f32>();

            let input = client.create_from_slice(f32::as_bytes(&codes));
            let output = client.empty(output_bytes);

            // Safety: the input holds `width * height` f32 values.
            let input_plane = unsafe { DevicePlane::new(&input, self.width, self.height) };
            // Safety: the output holds four times the input's f32 values.
            let output_plane = unsafe { DevicePlane::new(&output, self.width * 2, self.height * 2) };

            let ran = supersampler.run(client, &context, input_plane, PEAK, output_plane);
            ran.expect("the frame runs");

            let bytes = client.read_one(output).expect("the output reads back");
            let values = f32::from_bytes(&bytes);
            let output = values.iter().map(|code| code / PEAK).collect();

            FrameRun {
                output,
                offered,
                layers: supersampler.layers().to_vec(),
            }
        }
    }

    #[test]
    fn a_frame_through_the_loaded_supersampler_matches_the_reference() {
        let plane = rippled(24, 20);
        let model = Model::R8F64;
        let builtin = av_toolbox_models::builtin("artcnn", model.file_name());
        let bytes = builtin.expect("R8F64 is built in");
        let parsed = Weights::parse(bytes, model.features(), model.blocks());
        let weights = parsed.expect("the builtin weights parse");
        let expected = reference_artcnn(&plane, 24, 20, &weights);
        let backend = Backend::open_default(test_device()).expect("a backend should open");
        let task = LoadedFrame {
            plane,
            width: 24,
            height: 20,
        };

        let ran = backend.build(task);

        let error = ran
            .output
            .iter()
            .zip(&expected)
            .map(|(got, want)| (f64::from(*got) - want).abs())
            .fold(0.0, f64::max);
        let tuned = ran.layers.iter().all(|layer| layer.candidate.is_some());
        println!("loaded 24x20: largest error {error:e}, tuned {tuned}");

        if ran.offered == BodyPath::Convolution {
            assert!(tuned, "a body layer skipped the tuned path");
        }

        // The bound of the R8F64 24x20 case on every path.
        assert!(error <= 5e-4, "{error:e}");
    }
}
