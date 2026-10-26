// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::blur::{BlurPlanes, Gaussian, Storage};
use crate::matrix::{Coefficients, MatrixSamples, SampleRange, scaling};
use crate::{DevicePlane, Dispatch, Edge, assert_plane, assert_plane_sized};

/// Smoothed map strength below this fraction of the full scale paints nothing.
pub const FLOOR: f32 = 0.25;

/// The share of the picture's brightness the heat style keeps under its colour.
const HEAT_LIFT: f32 = 0.25;

/// The colouring a heatmap is painted with.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Style {
    /// A greyscale picture shaded from white to deep blue.
    Blue,
    /// A lifted greyscale picture shaded from cyan through yellow to red.
    Heat,
}

/// How the planes of a clip hold the picture being painted.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum PaintLayout {
    /// Red, green and blue planes.
    Rgb,
    /// Luma and two chroma planes, each chroma plane shrunk by the shifts.
    Yuv { width_shift: u32, height_shift: u32 },
}

/// Everything a painting depends on besides the planes themselves.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Painting {
    pub style: Style,
    /// The smoothed map value that paints at full strength.
    pub scale: f32,
    pub layout: PaintLayout,
    pub samples: MatrixSamples,
    pub range: SampleRange,
    pub coefficients: Coefficients,
}

/// The standard deviation of the smoothing applied to a map of `height` rows.
pub fn sigma_for(height: u32) -> f32 {
    height as f32 / 90.0
}

/// The colours at strength 0, 0.5 and 1, as red, green and blue between 0 and 1.
pub fn ramp_stops(style: Style) -> [[f32; 3]; 3] {
    let stops = match style {
        Style::Blue => [[255.0, 255.0, 255.0], [110.0, 160.0, 255.0], [20.0, 40.0, 200.0]],
        Style::Heat => [[60.0, 220.0, 230.0], [240.0, 220.0, 40.0], [230.0, 30.0, 30.0]],
    };

    stops.map(|stop| stop.map(|channel| channel / 255.0))
}

/// The picture's brightness as `(base, gain)`, so a grey of `grey` shows as `base + gain * grey`.
pub fn lift(style: Style) -> (f32, f32) {
    match style {
        Style::Blue => (0.0, 1.0),
        Style::Heat => (HEAT_LIFT, 1.0 - HEAT_LIFT),
    }
}

/// The weights BT.709 gives red, green and blue in the brightness of an RGB picture.
const GREY_WEIGHTS: [f32; 3] = [0.2126, 0.7152, 0.0722];

/// The palette buffer holds the three ramp stops, then the base and gain of [lift].
const PALETTE_VALUES: usize = 11;

/// The planes one painting reads, works in and writes.
///
/// `map`, `scratch` and `smoothed` are f32 at the luma size. The picture and output planes hold
/// [Painting::samples](crate::heatmap::Painting::samples), with chroma at its subsampled size.
#[derive(Debug, Clone, Copy)]
pub struct HeatmapPlanes<'a> {
    pub picture: [DevicePlane<'a>; 3],
    pub map: DevicePlane<'a>,
    pub scratch: DevicePlane<'a>,
    pub smoothed: DevicePlane<'a>,
    pub output: [DevicePlane<'a>; 3],
}

/// Paints a smoothed distortion map over a greyscale copy of a clip.
#[derive(Debug)]
pub struct Heatmap {
    painting: Painting,
    gaussian: Gaussian,
    palette: Handle,
    height: u32,
}

/// The scalars every painting kernel shares.
#[derive(Debug, Copy, Clone)]
struct Encoding {
    luma_offset: f32,
    luma_scale: f32,
    chroma_offset: f32,
    chroma_scale: f32,
    largest: f32,
    rounds: bool,
}

impl Encoding {
    fn new(painting: &Painting) -> Self {
        let (luma_offset, luma_scale) = scaling(painting.samples, painting.range, false);
        let (chroma_offset, chroma_scale) = scaling(painting.samples, painting.range, true);
        let (largest, rounds) = match painting.samples {
            MatrixSamples::U8 => (255.0, true),
            MatrixSamples::U16 { bits } => (((1u32 << bits) - 1) as f32, true),
            MatrixSamples::F32 => (0.0, false),
        };

        Self {
            luma_offset: luma_offset as f32,
            luma_scale: luma_scale as f32,
            chroma_offset: chroma_offset as f32,
            chroma_scale: chroma_scale as f32,
            largest,
            rounds,
        }
    }
}

impl Heatmap {
    /// A painting of clips `height` rows tall, with its palette and smoothing already on the device.
    pub fn new<R: Runtime>(client: &ComputeClient<R>, painting: Painting, height: u32) -> Self {
        let sigma = sigma_for(height);
        let gaussian = Gaussian::new(client, sigma);

        let stops = ramp_stops(painting.style);
        let (base, gain) = lift(painting.style);
        let mut values: Vec<f32> = stops.iter().flatten().copied().collect();
        values.push(base);
        values.push(gain);

        let bytes = f32::as_bytes(&values);
        let palette = client.create_from_slice(bytes);

        Self {
            painting,
            gaussian,
            palette,
            height,
        }
    }

    fn palette<R: Runtime>(&self) -> ArrayArg<R> {
        let handle = self.palette.clone();

        // Safety: the palette buffer was made from exactly `PALETTE_VALUES` f32 values.
        unsafe { ArrayArg::from_raw_parts(handle, PALETTE_VALUES) }
    }

    /// Smooths `planes.map` and paints it over `planes.picture` into `planes.output`.
    ///
    /// Integer outputs are rounded to nearest and clamped to their depth.
    pub fn apply<R: Runtime>(
        &self,
        client: &ComputeClient<R>,
        dispatch: &impl Dispatch,
        planes: HeatmapPlanes<'_>,
    ) {
        let width = planes.map.width();
        let height = planes.map.height();
        assert_eq!(
            height, self.height,
            "heatmap map height differs from the height given to new"
        );
        assert_plane(&planes.map, width, height, "heatmap map");
        assert_plane(&planes.scratch, width, height, "heatmap scratch");
        assert_plane(&planes.smoothed, width, height, "heatmap smoothed");
        self.assert_picture_planes(&planes, width, height);

        let blur_planes = BlurPlanes {
            input: planes.map,
            scratch: planes.scratch,
            output: planes.smoothed,
        };
        self.gaussian
            .apply_with_edge(client, dispatch, blur_planes, Edge::Clamp, Storage::F32);

        match self.painting.samples {
            MatrixSamples::U8 => self.launch::<u8, R>(client, dispatch, planes),
            MatrixSamples::U16 { .. } => self.launch::<u16, R>(client, dispatch, planes),
            MatrixSamples::F32 => self.launch::<f32, R>(client, dispatch, planes),
        }
    }

    fn assert_picture_planes(&self, planes: &HeatmapPlanes<'_>, width: u32, height: u32) {
        let bytes = match self.painting.samples {
            MatrixSamples::U8 => 1,
            MatrixSamples::U16 { .. } => 2,
            MatrixSamples::F32 => 4,
        };
        let (chroma_width, chroma_height) = match self.painting.layout {
            PaintLayout::Rgb => (width, height),
            PaintLayout::Yuv {
                width_shift,
                height_shift,
            } => (
                width.div_ceil(1 << width_shift),
                height.div_ceil(1 << height_shift),
            ),
        };
        let names = ["first", "second", "third"];

        for (index, name) in names.iter().enumerate() {
            let (plane_width, plane_height) = match index {
                0 => (width, height),
                _ => (chroma_width, chroma_height),
            };
            let picture_name = format!("heatmap {name} picture");
            let output_name = format!("heatmap {name} output");

            assert_plane_sized(
                &planes.picture[index],
                plane_width,
                plane_height,
                bytes,
                &picture_name,
            );
            assert_plane_sized(
                &planes.output[index],
                plane_width,
                plane_height,
                bytes,
                &output_name,
            );
        }
    }

    fn launch<E: CubePrimitive, R: Runtime>(
        &self,
        client: &ComputeClient<R>,
        dispatch: &impl Dispatch,
        planes: HeatmapPlanes<'_>,
    ) {
        let encoding = Encoding::new(&self.painting);

        match self.painting.layout {
            PaintLayout::Rgb => self.launch_rgb::<E, R>(client, dispatch, planes, encoding),
            PaintLayout::Yuv {
                width_shift,
                height_shift,
            } => {
                self.launch_luma::<E, R>(client, dispatch, planes, encoding);
                self.launch_chroma::<E, R>(client, dispatch, planes, encoding, width_shift, height_shift);
            },
        }
    }

    fn launch_rgb<E: CubePrimitive, R: Runtime>(
        &self,
        client: &ComputeClient<R>,
        dispatch: &impl Dispatch,
        planes: HeatmapPlanes<'_>,
        encoding: Encoding,
    ) {
        let [red, green, blue] = planes.picture.map(|plane| plane.array::<R>());
        let [red_out, green_out, blue_out] = planes.output.map(|plane| plane.array::<R>());
        let smoothed = planes.smoothed.array::<R>();
        let palette = self.palette::<R>();
        let samples = planes.smoothed.samples();
        let cubes = dispatch.cube_count(samples);
        let unit = dispatch.cube_dim();

        // Safety: every plane holds `width * height` samples, checked by `apply`, and the palette
        // holds its `PALETTE_VALUES` values.
        unsafe {
            paint_rgb::launch_unchecked::<E, R>(
                client,
                cubes,
                unit,
                red,
                green,
                blue,
                smoothed,
                palette,
                red_out,
                green_out,
                blue_out,
                self.painting.scale,
                encoding.luma_offset,
                encoding.luma_scale,
                encoding.largest,
                encoding.rounds,
            );
        }
    }

    fn launch_luma<E: CubePrimitive, R: Runtime>(
        &self,
        client: &ComputeClient<R>,
        dispatch: &impl Dispatch,
        planes: HeatmapPlanes<'_>,
        encoding: Encoding,
    ) {
        let (weight_red, weight_blue) = self.painting.coefficients.weights();
        let luma = planes.picture[0].array::<R>();
        let smoothed = planes.smoothed.array::<R>();
        let palette = self.palette::<R>();
        let luma_out = planes.output[0].array::<R>();
        let samples = planes.smoothed.samples();
        let cubes = dispatch.cube_count(samples);
        let unit = dispatch.cube_dim();

        // Safety: the luma, smoothed and luma output planes hold `width * height` samples, checked
        // by `apply`, and the palette holds its `PALETTE_VALUES` values.
        unsafe {
            paint_luma::launch_unchecked::<E, R>(
                client,
                cubes,
                unit,
                luma,
                smoothed,
                palette,
                luma_out,
                self.painting.scale,
                encoding.luma_offset,
                encoding.luma_scale,
                weight_red as f32,
                weight_blue as f32,
                encoding.largest,
                encoding.rounds,
            );
        }
    }

    fn launch_chroma<E: CubePrimitive, R: Runtime>(
        &self,
        client: &ComputeClient<R>,
        dispatch: &impl Dispatch,
        planes: HeatmapPlanes<'_>,
        encoding: Encoding,
        width_shift: u32,
        height_shift: u32,
    ) {
        let (weight_red, weight_blue) = self.painting.coefficients.weights();
        let width = planes.smoothed.width();
        let height = planes.smoothed.height();
        let chroma_width = planes.output[1].width();
        let luma = planes.picture[0].array::<R>();
        let smoothed = planes.smoothed.array::<R>();
        let palette = self.palette::<R>();
        let blue_out = planes.output[1].array::<R>();
        let red_out = planes.output[2].array::<R>();
        let samples = planes.output[1].samples();
        let cubes = dispatch.cube_count(samples);
        let unit = dispatch.cube_dim();

        // Safety: the luma and smoothed planes hold `width * height` samples and both chroma
        // outputs hold `chroma_width` by the shifted height, checked by `apply`. Every pixel read
        // is clipped to the frame, and the palette holds its `PALETTE_VALUES` values.
        unsafe {
            paint_chroma::launch_unchecked::<E, R>(
                client,
                cubes,
                unit,
                luma,
                smoothed,
                palette,
                blue_out,
                red_out,
                width,
                height,
                chroma_width,
                width_shift,
                height_shift,
                self.painting.scale,
                encoding.luma_offset,
                encoding.luma_scale,
                encoding.chroma_offset,
                encoding.chroma_scale,
                weight_red as f32,
                weight_blue as f32,
                encoding.largest,
                encoding.rounds,
            );
        }
    }
}

/// How strongly a smoothed map `value` paints, between 0 and 1. A NaN paints nothing.
#[cube]
fn strength(value: f32, scale: f32) -> f32 {
    let defined = select(f32::is_nan(value), 0.0, value);
    let lifted = (defined / scale - FLOOR) / (1.0 - FLOOR);

    f32::clamp(lifted, 0.0, 1.0)
}

/// A code value as a fraction of its span, clamped to between 0 and 1.
#[cube]
fn normalised(code: f32, offset: f32, scale: f32) -> f32 {
    let fraction = (code - offset) / scale;

    f32::clamp(fraction, 0.0, 1.0)
}

/// One channel of the colour a pixel of brightness `grey` paints with at `strength`.
#[cube]
fn colour_channel(palette: &Array<f32>, channel: u32, grey: f32, strength: f32) -> f32 {
    let brightness = palette[9] + palette[10] * grey;
    let mut from = palette[(3 + channel) as usize];
    let mut to = palette[(6 + channel) as usize];
    let mut fraction = (strength - 0.5) * 2.0;

    if strength <= 0.5 {
        from = palette[channel as usize];
        to = palette[(3 + channel) as usize];
        fraction = strength * 2.0;
    }

    let tint = from + (to - from) * fraction;

    brightness * tint
}

/// A value between 0 and 1, or centred on 0 for chroma, as a code value of the output.
#[cube]
fn encoded(value: f32, offset: f32, scale: f32, largest: f32, #[comptime] rounds: bool) -> f32 {
    let mut code = offset + scale * value;

    if rounds {
        let rounded = f32::floor(code + 0.5);
        code = f32::clamp(rounded, 0.0, largest);
    }

    code
}

/// The luma, blue difference and red difference of one painted pixel of a YUV clip.
#[cube]
fn painted_yuv(
    luma_code: f32,
    smoothed: f32,
    palette: &Array<f32>,
    scale: f32,
    luma_offset: f32,
    luma_scale: f32,
    weight_red: f32,
    weight_blue: f32,
) -> (f32, f32, f32) {
    let grey = normalised(luma_code, luma_offset, luma_scale);
    let pixel_strength = strength(smoothed, scale);
    let red = colour_channel(palette, 0, grey, pixel_strength);
    let green = colour_channel(palette, 1, grey, pixel_strength);
    let blue = colour_channel(palette, 2, grey, pixel_strength);

    let weight_green = 1.0 - weight_red - weight_blue;
    let luma = weight_red * red + weight_green * green + weight_blue * blue;
    let blue_difference = (blue - luma) / (2.0 * (1.0 - weight_blue));
    let red_difference = (red - luma) / (2.0 * (1.0 - weight_red));

    (luma, blue_difference, red_difference)
}

#[cube(launch_unchecked)]
fn paint_rgb<E: CubePrimitive>(
    red: &Array<E>,
    green: &Array<E>,
    blue: &Array<E>,
    smoothed: &Array<f32>,
    palette: &Array<f32>,
    red_out: &mut Array<E>,
    green_out: &mut Array<E>,
    blue_out: &mut Array<E>,
    scale: f32,
    offset: f32,
    span: f32,
    largest: f32,
    #[comptime] rounds: bool,
) {
    if ABSOLUTE_POS >= red_out.len() {
        terminate!();
    }

    let red_value = f32::cast_from(red[ABSOLUTE_POS]);
    let green_value = f32::cast_from(green[ABSOLUTE_POS]);
    let blue_value = f32::cast_from(blue[ABSOLUTE_POS]);
    let red_grey = GREY_WEIGHTS[0] * normalised(red_value, offset, span);
    let green_grey = GREY_WEIGHTS[1] * normalised(green_value, offset, span);
    let blue_grey = GREY_WEIGHTS[2] * normalised(blue_value, offset, span);
    let grey = red_grey + green_grey + blue_grey;
    let pixel_strength = strength(smoothed[ABSOLUTE_POS], scale);

    let red_colour = colour_channel(palette, 0, grey, pixel_strength);
    let green_colour = colour_channel(palette, 1, grey, pixel_strength);
    let blue_colour = colour_channel(palette, 2, grey, pixel_strength);
    let red_code = encoded(red_colour, offset, span, largest, rounds);
    let green_code = encoded(green_colour, offset, span, largest, rounds);
    let blue_code = encoded(blue_colour, offset, span, largest, rounds);

    red_out[ABSOLUTE_POS] = E::cast_from(red_code);
    green_out[ABSOLUTE_POS] = E::cast_from(green_code);
    blue_out[ABSOLUTE_POS] = E::cast_from(blue_code);
}

#[cube(launch_unchecked)]
fn paint_luma<E: CubePrimitive>(
    luma: &Array<E>,
    smoothed: &Array<f32>,
    palette: &Array<f32>,
    luma_out: &mut Array<E>,
    scale: f32,
    luma_offset: f32,
    luma_scale: f32,
    weight_red: f32,
    weight_blue: f32,
    largest: f32,
    #[comptime] rounds: bool,
) {
    if ABSOLUTE_POS >= luma_out.len() {
        terminate!();
    }

    let luma_code = f32::cast_from(luma[ABSOLUTE_POS]);
    let (painted, _, _) = painted_yuv(
        luma_code,
        smoothed[ABSOLUTE_POS],
        palette,
        scale,
        luma_offset,
        luma_scale,
        weight_red,
        weight_blue,
    );
    let code = encoded(painted, luma_offset, luma_scale, largest, rounds);

    luma_out[ABSOLUTE_POS] = E::cast_from(code);
}

/// Writes each chroma sample as the average chroma of the painted pixels it covers inside the frame.
#[cube(launch_unchecked)]
fn paint_chroma<E: CubePrimitive>(
    luma: &Array<E>,
    smoothed: &Array<f32>,
    palette: &Array<f32>,
    blue_out: &mut Array<E>,
    red_out: &mut Array<E>,
    width: u32,
    height: u32,
    chroma_width: u32,
    width_shift: u32,
    height_shift: u32,
    scale: f32,
    luma_offset: f32,
    luma_scale: f32,
    chroma_offset: f32,
    chroma_scale: f32,
    weight_red: f32,
    weight_blue: f32,
    largest: f32,
    #[comptime] rounds: bool,
) {
    if ABSOLUTE_POS >= blue_out.len() {
        terminate!();
    }

    let position = ABSOLUTE_POS as u32;
    let chroma_column = position % chroma_width;
    let chroma_row = position / chroma_width;
    let first_column = chroma_column << width_shift;
    let first_row = chroma_row << height_shift;
    let end_column = u32::min((chroma_column + 1) << width_shift, width);
    let end_row = u32::min((chroma_row + 1) << height_shift, height);

    let mut blue_total = 0.0;
    let mut red_total = 0.0;
    let mut count = 0.0;

    for row in first_row..end_row {
        for column in first_column..end_column {
            let index = (row * width + column) as usize;
            let luma_code = f32::cast_from(luma[index]);
            let (_, blue_difference, red_difference) = painted_yuv(
                luma_code,
                smoothed[index],
                palette,
                scale,
                luma_offset,
                luma_scale,
                weight_red,
                weight_blue,
            );

            blue_total += blue_difference;
            red_total += red_difference;
            count += 1.0;
        }
    }

    let blue_code = encoded(blue_total / count, chroma_offset, chroma_scale, largest, rounds);
    let red_code = encoded(red_total / count, chroma_offset, chroma_scale, largest, rounds);

    blue_out[ABSOLUTE_POS] = E::cast_from(blue_code);
    red_out[ABSOLUTE_POS] = E::cast_from(red_code);
}
