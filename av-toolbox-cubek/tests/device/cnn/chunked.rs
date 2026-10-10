use av_toolbox_cubek::cnn::{
    Activation,
    BodyConvolution,
    Candidate,
    ConvLayer,
    DirectVariant,
    Finish,
    Precision,
    direct_variants,
    run_candidate,
};
use av_toolbox_cubek::testing::synthetic_layer;
use av_toolbox_gpu::{BuildContext, BuildTask};
use cubecl::prelude::Runtime;

use crate::artcnn::{edged_maps, f16_step, read_maps, region, synced, upload_maps};
use crate::support::{backend, empty};

/// The absolute difference that summing the products in another order may add in f32, before the f16
/// rounding.
const REORDERED_SUM: f32 = 1e-4;

/// A chunked direct variant and the whole-width variant of the same layout, run over one layer.
///
/// The output is `None` where the device lacks either variant.
struct ChunkedPair {
    features: u32,
    whole: DirectVariant,
    chunk: u32,
}

impl BuildTask for ChunkedPair {
    type Output = Option<(Vec<f32>, Vec<f32>)>;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let plane_width = client.properties().hardware.plane_size_max;
        let whole = DirectVariant {
            plane_width,
            ..self.whole
        };
        let chunked = DirectVariant {
            input_chunk: Some(self.chunk),
            ..whole
        };
        let offered = direct_variants(client.properties(), self.features);

        if !offered.contains(&whole) || !offered.contains(&chunked) {
            return None;
        }

        let frame = region(37, 11);
        let synthetic = synthetic_layer(self.features, true, u64::from(self.features));
        let layer = ConvLayer::upload(
            client,
            &synthetic.weights,
            synthetic.bias.as_deref(),
            self.features,
            Precision::F16,
            false,
        );
        let maps = edged_maps(frame, self.features);
        let input = upload_maps(client, &maps, Precision::F16);
        let bytes = maps.len() * Precision::F16.bytes();
        let whole_output = empty(client, bytes);
        let chunked_output = empty(client, bytes);
        let finish = Finish {
            activation: Activation::Relu,
            residual: None,
            skip: None,
        };

        let whole_run = BodyConvolution::of_conv(&layer, &input, &whole_output, frame, finish);
        let chunked_run = BodyConvolution::of_conv(&layer, &input, &chunked_output, frame, finish);
        run_candidate(client, Candidate::Direct(whole), &whole_run).expect("the whole variant runs");
        run_candidate(client, Candidate::Direct(chunked), &chunked_run).expect("the chunked variant runs");
        synced(client);

        let whole_values = read_maps(client, &whole_output, Precision::F16);
        let chunked_values = read_maps(client, &chunked_output, Precision::F16);

        Some((whole_values, chunked_values))
    }
}

fn assert_chunk_matches(features: u32, whole: DirectVariant, chunk: u32) {
    let name = format!("{features} features, chunk {chunk}, {whole:?}");
    let task = ChunkedPair {
        features,
        whole,
        chunk,
    };

    let Some((whole_values, chunked_values)) = backend().build(task) else {
        println!("{name}: not offered on this device");
        return;
    };

    for (index, (chunked, whole)) in chunked_values.iter().zip(&whole_values).enumerate() {
        let magnitude = whole.abs().max(chunked.abs());
        let bound = f16_step(magnitude) + REORDERED_SUM;

        assert!(
            (chunked - whole).abs() <= bound,
            "{name}: value {index} is {chunked}, the whole variant's {whole}"
        );
    }
}

#[test]
fn a_chunked_variant_matches_its_whole_layout_within_one_step() {
    let default_variant = DirectVariant::default();
    let one_row = DirectVariant {
        rows: 1,
        planes_n: 4,
        ..DirectVariant::default()
    };

    for (features, chunk) in [(128, 64), (256, 64), (256, 128)] {
        assert_chunk_matches(features, default_variant, chunk);
    }

    assert_chunk_matches(512, one_row, 64);
    assert_chunk_matches(512, one_row, 128);
}
