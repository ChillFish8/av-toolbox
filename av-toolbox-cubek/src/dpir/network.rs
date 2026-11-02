use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::cnn::{
    Activation,
    BodyPath,
    CnnError,
    ConvLayer,
    Epilogue,
    Finish,
    Kernels,
    LayerMaps,
    LayerRun,
    PaddedLastLayer,
    Precision,
    Region,
    Stride,
    StridedLayer,
    StridedVariant,
    encode,
    run_layer,
    run_strided,
    strided_variant,
};
use crate::dpir::{
    HeadLayer,
    LayerKind,
    Placement,
    Variant,
    WIDTHS,
    Weights,
    gather,
    head,
    matrix_product,
    scatter,
    tail,
    write_out,
};
use crate::plane::{are_distinct, assert_holds};
use crate::{DevicePlane, Dispatch};

/// How many ResBlocks run in a row at each level.
const BLOCKS: usize = 4;

/// The deepest of the four levels.
const DEEPEST: usize = 3;

/// The multiple of 8 a region's sides must be, so each of the three halvings is exact.
pub(super) const ALIGNMENT: u32 = 8;

/// A DRUNet deblocking network's weights on the device, in the feature maps' type, and how its f16
/// convolutions choose their kernel.
pub struct Dpir {
    variant: Variant,
    precision: Precision,
    head: HeadLayer,
    convolutions: Vec<ConvLayer>,
    downs: Vec<Handle>,
    ups: Vec<Handle>,
    tail: PaddedLastLayer,
    kernels: Kernels,
}

impl Dpir {
    /// Uploads `weights` in `precision`'s type.
    pub fn upload(client: &Client, weights: &Weights, precision: Precision) -> Self {
        let layers = weights.layers();
        let last = layers.len() - 1;
        let head = HeadLayer::upload(client, &layers[0], precision);
        let mut convolutions = Vec::new();
        let mut downs = Vec::new();
        let mut ups = Vec::new();

        for layer in &layers[1..last] {
            match layer.kind {
                LayerKind::Conv => {
                    let uploaded =
                        ConvLayer::upload(client, &layer.values, None, layer.in_channels, precision, false);
                    convolutions.push(uploaded);
                },
                LayerKind::Down => {
                    let matrix = upload_matrix(client, &layer.values, precision);
                    downs.push(matrix);
                },
                LayerKind::Up => {
                    let matrix = upload_matrix(client, &layer.values, precision);
                    ups.push(matrix);
                },
            }
        }

        let variant = weights.variant();
        let image_channels = variant.image_channels();
        let tail = PaddedLastLayer::upload(
            client,
            &layers[last].values,
            None,
            WIDTHS[0],
            image_channels,
            precision,
        );

        Self {
            variant,
            precision,
            head,
            convolutions,
            downs,
            ups,
            tail,
            kernels: Kernels::default(),
        }
    }

    /// The same network, with its f16 convolutions on the convolution path choosing their kernel by
    /// `kernels`.
    pub fn with_kernels(self, kernels: Kernels) -> Self {
        Self { kernels, ..self }
    }

    pub fn variant(&self) -> Variant {
        self.variant
    }

    pub fn precision(&self) -> Precision {
        self.precision
    }

    pub(crate) fn kernels(&self) -> &Kernels {
        &self.kernels
    }
}

fn upload_matrix(client: &Client, values: &[f32], precision: Precision) -> Handle {
    let bytes = encode(values, precision);

    client.create_from_slice(&bytes)
}

/// The planes a forward pass reads and writes, one of each per image channel, all the same size.
#[derive(Clone, Copy)]
pub struct DpirPlanes<'a> {
    pub inputs: &'a [DevicePlane<'a>],
    pub outputs: &'a [DevicePlane<'a>],
}

/// The feature maps one region's forward pass works in, four of each level's size.
///
/// Level `l` holds `(width >> l) * (height >> l) * WIDTHS[l]` values of the precision's type in each map. The
/// first level's maps also hold the tail's 16 f32 values a sample. `columns` holds
/// [column_bytes](crate::dpir::column_bytes), and is needed on [BodyPath::Im2col](crate::cnn::BodyPath) only.
///
/// A layer reads one map while writing another, so every map must be a distinct buffer. Only a handle passed
/// twice is caught, and only in debug builds.
#[derive(Clone, Copy)]
pub struct DpirMaps<'a> {
    pub levels: [[&'a Handle; 4]; 4],
    pub columns: Option<&'a Handle>,
}

/// The bytes one map of each level takes for a `width` by `height` region.
pub fn level_bytes(precision: Precision, width: u32, height: u32) -> [usize; 4] {
    std::array::from_fn(|level| {
        let samples = (width as usize >> level) * (height as usize >> level);

        samples * WIDTHS[level] as usize * precision.bytes()
    })
}

/// The bytes the im2col columns take for a `width` by `height` region, nine of the first level's maps.
pub fn column_bytes(precision: Precision, width: u32, height: u32) -> usize {
    let [first_level, ..] = level_bytes(precision, width, height);

    9 * first_level
}

/// Runs the network over the placement's region of `planes`, and writes `core` of its output. Gives how each
/// ResBlock convolution ran, in forward order.
///
/// The inputs are divided by `peak` and the strength is on the 8-bit scale. Each output sample is clamped to
/// `0..=1` and scaled by `peak`. The region's sides must be multiples of 8.
#[expect(
    clippy::too_many_arguments,
    reason = "each argument is a distinct input of the pass"
)]
pub fn run_region(
    client: &Client,
    dispatch: &impl Dispatch,
    dpir: &Dpir,
    path: BodyPath,
    planes: &DpirPlanes<'_>,
    strength: f32,
    peak: f32,
    placement: Placement,
    core: Region,
    maps: DpirMaps<'_>,
) -> Result<Vec<LayerRun>, CnnError> {
    let region = placement.region;
    check_maps(dpir.precision, path, region, maps)?;

    let [first_skip, first_spare, ..] = maps.levels[0];
    head(
        client,
        dispatch,
        &dpir.head,
        planes.inputs,
        peak,
        strength,
        placement,
        first_skip,
    );

    let columns = match path {
        BodyPath::Im2col => maps.columns,
        BodyPath::Convolution | BodyPath::UnfusedConvolution => None,
    };
    let mut pass = Pass {
        client,
        dispatch,
        dpir,
        path,
        levels: maps.levels,
        columns,
        region,
        runs: Vec::new(),
    };
    let mut state = first_skip;

    for level in 0..DEEPEST {
        state = pass.blocks(level, state, None)?;
        state = pass.down(level, state)?;
    }

    let [deepest_skip, ..] = maps.levels[DEEPEST];
    state = pass.blocks(DEEPEST, state, Some(deepest_skip))?;

    for level in (0..DEEPEST).rev() {
        state = pass.up(level, state)?;

        let [level_skip, ..] = maps.levels[level];
        state = pass.blocks(level, state, Some(level_skip))?;
    }

    // The first level's spare map is free once the last ResBlock has run, so the tail writes into it.
    tail(client, dispatch, &dpir.tail, path, state, region, first_spare)?;

    let channels = dpir.variant.image_channels();
    write_out(
        client,
        dispatch,
        first_spare,
        placement,
        core,
        channels,
        peak,
        planes.outputs,
    );

    Ok(pass.runs)
}

/// Asserts the region splits into the four levels and every map holds its level, and that the columns hold
/// [column_bytes](crate::dpir::column_bytes) on [BodyPath::Im2col](crate::cnn::BodyPath).
fn check_maps(
    precision: Precision,
    path: BodyPath,
    region: Region,
    maps: DpirMaps<'_>,
) -> Result<(), CnnError> {
    let aligned = region.width.is_multiple_of(ALIGNMENT) && region.height.is_multiple_of(ALIGNMENT);
    assert!(
        aligned && region.samples() > 0,
        "the region's sides are not multiples of 8"
    );

    let mut every_map: Vec<&Handle> = maps.levels.iter().flatten().copied().collect();
    every_map.extend(maps.columns);
    debug_assert!(are_distinct(&every_map), "two of the maps are the same buffer");

    let bytes = level_bytes(precision, region.width, region.height);
    for (level, level_maps) in maps.levels.iter().enumerate() {
        let name = format!("level {level} map");

        for map in level_maps {
            assert_holds(map, bytes[level], &name);
        }
    }

    if path == BodyPath::Im2col {
        let columns = maps.columns.ok_or(CnnError::MissingColumns)?;
        let needed = column_bytes(precision, region.width, region.height);
        assert_holds(columns, needed, "columns");
    }

    Ok(())
}

/// One region's forward pass in progress, with the ResBlock convolutions it has run so far.
struct Pass<'a, D: Dispatch> {
    client: &'a Client,
    dispatch: &'a D,
    dpir: &'a Dpir,
    path: BodyPath,
    levels: [[&'a Handle; 4]; 4],
    columns: Option<&'a Handle>,
    region: Region,
    runs: Vec<LayerRun>,
}

impl<'a, D: Dispatch> Pass<'a, D> {
    /// The region the maps of `level` cover.
    fn level_region(&self, level: usize) -> Region {
        Region {
            x: self.region.x >> level,
            y: self.region.y >> level,
            width: self.region.width >> level,
            height: self.region.height >> level,
        }
    }

    /// Runs the four ResBlocks of `level` from `state`, giving the map the last one wrote. The last one also
    /// adds `skip`, if given.
    fn blocks(
        &mut self,
        level: usize,
        state: &'a Handle,
        skip: Option<&'a Handle>,
    ) -> Result<&'a Handle, CnnError> {
        let mut state = state;

        for block in 0..BLOCKS {
            let last = block + 1 == BLOCKS;
            let block_skip = skip.filter(|_| last);
            state = self.block(level, state, block_skip)?;
        }

        Ok(state)
    }

    /// Runs one ResBlock of `level` from `state`, giving the map it wrote.
    ///
    /// The first convolution writes the level's spare map with a ReLU. The second writes whichever of the
    /// other two maps is not `state`, adding `state` and then `skip`, if given.
    fn block(
        &mut self,
        level: usize,
        state: &'a Handle,
        skip: Option<&'a Handle>,
    ) -> Result<&'a Handle, CnnError> {
        let [_, spare, first_choice, second_choice] = self.levels[level];
        let output = if std::ptr::eq(state, first_choice) {
            second_choice
        } else {
            first_choice
        };
        let activated = Finish {
            activation: Activation::Relu,
            residual: None,
            skip: None,
        };
        let joined = Finish {
            activation: Activation::None,
            residual: Some(state),
            skip,
        };
        let joined_epilogue = match skip {
            Some(_) => Epilogue::ResidualSkip,
            None => Epilogue::Residual,
        };

        self.convolve(level, state, spare, activated, Epilogue::Relu)?;
        self.convolve(level, spare, output, joined, joined_epilogue)?;

        Ok(output)
    }

    /// Runs the next ResBlock convolution over the maps of `level` in `input`, into `output`.
    fn convolve(
        &mut self,
        level: usize,
        input: &Handle,
        output: &Handle,
        finish: Finish<'_>,
        epilogue: Epilogue,
    ) -> Result<(), CnnError> {
        let layer = &self.dpir.convolutions[self.runs.len()];
        let region = self.level_region(level);
        let maps = LayerMaps {
            input,
            output,
            columns: self.columns,
        };

        let run = run_layer(
            self.client,
            self.dispatch,
            self.dpir.kernels(),
            self.path,
            layer,
            maps,
            region,
            finish,
            epilogue,
        )?;
        self.runs.push(run);

        Ok(())
    }

    /// The strided kernel's variant for a `stride` layer reading `channels` channels, when the pass runs one.
    fn strided(&self, stride: Stride, channels: u32) -> Option<StridedVariant> {
        let on_cooperative_matrices = self.path != BodyPath::Im2col && self.dpir.precision == Precision::F16;
        let properties = self.client.properties();

        on_cooperative_matrices
            .then(|| strided_variant(properties, stride, channels))
            .flatten()
    }

    /// Halves `state` of `level` into the next level's skip map, giving that map.
    fn down(&mut self, level: usize, state: &'a Handle) -> Result<&'a Handle, CnnError> {
        let [_, spare, ..] = self.levels[level];
        let [next_skip, ..] = self.levels[level + 1];
        let region = self.level_region(level);
        let channels = WIDTHS[level];
        let precision = self.dpir.precision;
        let weights = &self.dpir.downs[level];

        if let Some(variant) = self.strided(Stride::Down, channels) {
            let layer = StridedLayer {
                input: state,
                weights,
                output: next_skip,
                width: region.width,
                height: region.height,
                channels,
                stride: Stride::Down,
            };
            run_strided(self.client, variant, &layer)?;

            return Ok(next_skip);
        }

        gather(
            self.client,
            self.dispatch,
            state,
            region.width,
            region.height,
            channels,
            precision,
            spare,
        );

        let rows = region.samples() as usize / 4;
        let depth = 4 * channels as usize;
        let columns = WIDTHS[level + 1] as usize;
        matrix_product(
            self.client,
            spare,
            rows,
            depth,
            weights,
            columns,
            precision,
            next_skip,
        )?;

        Ok(next_skip)
    }

    /// Doubles `state` of the next level into a map of `level`, giving that map.
    fn up(&mut self, level: usize, state: &'a Handle) -> Result<&'a Handle, CnnError> {
        let deeper = level + 1;
        let [_, spare, output, _] = self.levels[level];
        let deeper_region = self.level_region(deeper);
        let precision = self.dpir.precision;
        let channels = WIDTHS[deeper];
        // The up layers run deepest first.
        let weights = &self.dpir.ups[DEEPEST - deeper];

        if let Some(variant) = self.strided(Stride::Up, channels) {
            let layer = StridedLayer {
                input: state,
                weights,
                output,
                width: deeper_region.width,
                height: deeper_region.height,
                channels,
                stride: Stride::Up,
            };
            run_strided(self.client, variant, &layer)?;

            return Ok(output);
        }

        let rows = deeper_region.samples() as usize;
        let depth = channels as usize;
        let columns = 4 * WIDTHS[level] as usize;
        matrix_product(
            self.client,
            state,
            rows,
            depth,
            weights,
            columns,
            precision,
            spare,
        )?;

        scatter(
            self.client,
            self.dispatch,
            spare,
            deeper_region.width,
            deeper_region.height,
            WIDTHS[level],
            precision,
            output,
        );

        Ok(output)
    }
}
