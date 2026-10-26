// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::blur::{BlurPlanes, Gaussian, Storage};
use crate::matrix::{Coefficients, MatrixSamples, SampleRange, scaling};
use crate::{DevicePlane, Dispatch, Edge, assert_plane, assert_plane_sized};

/// The default share of the scale that paints nothing.
pub const FLOOR: f32 = 0.25;

/// The share of the picture's brightness the heat style keeps under its colour.
const HEAT_LIFT: f32 = 0.25;

/// The share of the picture the overlay style's colour covers at full strength.
const OVERLAY_OPACITY: f32 = 0.85;

/// The colouring a heatmap is painted with.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Style {
    /// A greyscale picture shaded from white to deep blue.
    Blue,
    /// A lifted greyscale picture shaded from cyan through yellow to red.
    Heat,
    /// A greyscale picture with blue mixed over it, so dark pixels take the colour too.
    Overlay,
}

/// How a style turns a pixel's brightness and strength into a colour.
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Shading {
    /// The picture's brightness `base + gain * grey` times a ramp through `stops`.
    ///
    /// The stops are the colours at strength 0, 0.5 and 1, as red, green and blue between 0 and 1.
    Multiply {
        /// The ramp's colours at strength 0, 0.5 and 1.
        stops: [[f32; 3]; 3],
        /// The brightness of a black pixel.
        base: f32,
        /// How much brighter a white pixel is than a black one.
        gain: f32,
    },
    /// The picture's grey mixed with `colour`, which covers `opacity * strength` of it.
    Overlay {
        /// The colour laid over the picture, as red, green and blue between 0 and 1.
        colour: [f32; 3],
        /// How much of the picture the colour covers at full strength.
        opacity: f32,
    },
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
    /// The map value that paints at full strength.
    pub scale: f32,
    /// The share of `scale` that paints nothing, between 0 and 1.
    pub floor: f32,
    /// Whether the map is smoothed with a gaussian before it is painted.
    pub smooth: bool,
    pub layout: PaintLayout,
    pub samples: MatrixSamples,
    pub range: SampleRange,
    pub coefficients: Coefficients,
}

/// The standard deviation of the smoothing applied to a map of `height` rows.
pub fn sigma_for(height: u32) -> f32 {
    height as f32 / 90.0
}

/// How `style` turns a pixel's brightness and strength into a colour.
pub fn shading(style: Style) -> Shading {
    match style {
        Style::Blue => {
            let stops = [[255.0, 255.0, 255.0], [110.0, 160.0, 255.0], [20.0, 40.0, 200.0]];

            Shading::Multiply {
                stops: stops.map(as_fractions),
                base: 0.0,
                gain: 1.0,
            }
        },
        Style::Heat => {
            let stops = [[60.0, 220.0, 230.0], [240.0, 220.0, 40.0], [230.0, 30.0, 30.0]];

            Shading::Multiply {
                stops: stops.map(as_fractions),
                base: HEAT_LIFT,
                gain: 1.0 - HEAT_LIFT,
            }
        },
        Style::Overlay => Shading::Overlay {
            colour: as_fractions([30.0, 80.0, 255.0]),
            opacity: OVERLAY_OPACITY,
        },
    }
}

fn as_fractions(colour: [f32; 3]) -> [f32; 3] {
    colour.map(|channel| channel / 255.0)
}

/// The weights BT.709 gives red, green and blue in the brightness of an RGB picture.
const GREY_WEIGHTS: [f32; 3] = [0.2126, 0.7152, 0.0722];

/// The palette buffer holds a [Shading](crate::heatmap::Shading) as the kernels read it.
///
/// A multiply shading is its three stops, then its base and gain. An overlay shading is its colour,
/// then its opacity, with the rest left at zero.
const PALETTE_VALUES: usize = 11;

/// Where a multiply shading's middle stop starts in the palette.
const MIDDLE_STOP: u32 = 3;

/// Where a multiply shading's last stop starts in the palette.
const LAST_STOP: u32 = 6;

/// Where a multiply shading's base sits in the palette.
const BASE: u32 = 9;

/// Where a multiply shading's gain sits in the palette.
const GAIN: u32 = 10;

/// Where an overlay shading's opacity sits in the palette, after its colour.
const OPACITY: u32 = 3;

fn palette_values(shading: Shading) -> Vec<f32> {
    let mut values = match shading {
        Shading::Multiply { stops, base, gain } => {
            let mut values: Vec<f32> = stops.iter().flatten().copied().collect();
            values.push(base);
            values.push(gain);
            values
        },
        Shading::Overlay { colour, opacity } => {
            let mut values = colour.to_vec();
            values.push(opacity);
            values
        },
    };

    values.resize(PALETTE_VALUES, 0.0);
    values
}

/// The planes one painting reads, works in and writes.
///
/// `map`, `scratch` and `smoothed` are f32 at the luma size. The picture and output planes hold
/// [Painting::samples](crate::heatmap::Painting::samples), with chroma at its subsampled size.
/// `scratch` and `smoothed` are only needed, and only touched, when the painting smooths.
#[derive(Debug, Clone, Copy)]
pub struct HeatmapPlanes<'a> {
    pub picture: [DevicePlane<'a>; 3],
    pub map: DevicePlane<'a>,
    pub scratch: Option<DevicePlane<'a>>,
    pub smoothed: Option<DevicePlane<'a>>,
    pub output: [DevicePlane<'a>; 3],
}

/// Paints a distortion map over a greyscale copy of a clip.
#[derive(Debug)]
pub struct Heatmap {
    painting: Painting,
    gaussian: Option<Gaussian>,
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
    /// A painting of clips `height` rows tall, with its palette already on the device.
    ///
    /// When the painting smooths, its gaussian is put on the device too.
    pub fn new<R: Runtime>(client: &ComputeClient<R>, painting: Painting, height: u32) -> Self {
        let gaussian = painting.smooth.then(|| {
            let sigma = sigma_for(height);
            Gaussian::new(client, sigma)
        });

        let shading = shading(painting.style);
        let values = palette_values(shading);
        let bytes = f32::as_bytes(&values);
        let palette = client.create_from_slice(bytes);

        Self {
            painting,
            gaussian,
            palette,
            height,
        }
    }

    fn overlays(&self) -> bool {
        let shading = shading(self.painting.style);

        matches!(shading, Shading::Overlay { .. })
    }

    fn palette<R: Runtime>(&self) -> ArrayArg<R> {
        let handle = self.palette.clone();

        // Safety: the palette buffer was made from exactly `PALETTE_VALUES` f32 values.
        unsafe { ArrayArg::from_raw_parts(handle, PALETTE_VALUES) }
    }

    /// Paints `planes.map` over `planes.picture` into `planes.output`.
    ///
    /// When the painting smooths, the map is blurred into `planes.smoothed` first and painted from
    /// there. Otherwise the map is painted as it is. Integer outputs are rounded to nearest and
    /// clamped to their depth.
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
        if let Some(scratch) = &planes.scratch {
            assert_plane(scratch, width, height, "heatmap scratch");
        }

        if let Some(smoothed) = &planes.smoothed {
            assert_plane(smoothed, width, height, "heatmap smoothed");
        }

        self.assert_picture_planes(&planes, width, height);

        let values = self.map_values(client, dispatch, &planes);

        match self.painting.samples {
            MatrixSamples::U8 => self.launch::<u8, R>(client, dispatch, planes, values),
            MatrixSamples::U16 { .. } => self.launch::<u16, R>(client, dispatch, planes, values),
            MatrixSamples::F32 => self.launch::<f32, R>(client, dispatch, planes, values),
        }
    }

    /// The plane the painting reads its map values from, blurring the map into it when smoothing.
    fn map_values<'a, R: Runtime>(
        &self,
        client: &ComputeClient<R>,
        dispatch: &impl Dispatch,
        planes: &HeatmapPlanes<'a>,
    ) -> DevicePlane<'a> {
        let Some(gaussian) = &self.gaussian else {
            return planes.map;
        };

        let scratch = planes.scratch.expect("a smoothing heatmap needs a scratch plane");
        let smoothed = planes
            .smoothed
            .expect("a smoothing heatmap needs a smoothed plane");

        let blur_planes = BlurPlanes {
            input: planes.map,
            scratch,
            output: smoothed,
        };
        gaussian.apply_with_edge(client, dispatch, blur_planes, Edge::Clamp, Storage::F32);

        smoothed
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
        values: DevicePlane<'_>,
    ) {
        let encoding = Encoding::new(&self.painting);

        match self.painting.layout {
            PaintLayout::Rgb => self.launch_rgb::<E, R>(client, dispatch, planes, values, encoding),
            PaintLayout::Yuv {
                width_shift,
                height_shift,
            } => {
                let shifts = (width_shift, height_shift);
                self.launch_luma::<E, R>(client, dispatch, planes, values, encoding);
                self.launch_chroma::<E, R>(client, dispatch, planes, values, encoding, shifts);
            },
        }
    }

    fn launch_rgb<E: CubePrimitive, R: Runtime>(
        &self,
        client: &ComputeClient<R>,
        dispatch: &impl Dispatch,
        planes: HeatmapPlanes<'_>,
        values: DevicePlane<'_>,
        encoding: Encoding,
    ) {
        let [red, green, blue] = planes.picture.map(|plane| plane.array::<R>());
        let [red_out, green_out, blue_out] = planes.output.map(|plane| plane.array::<R>());
        let values_array = values.array::<R>();
        let palette = self.palette::<R>();
        let samples = values.samples();
        let cubes = dispatch.cube_count(samples);
        let unit = dispatch.cube_dim();
        let overlays = self.overlays();

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
                values_array,
                palette,
                red_out,
                green_out,
                blue_out,
                self.painting.scale,
                self.painting.floor,
                encoding.luma_offset,
                encoding.luma_scale,
                encoding.largest,
                encoding.rounds,
                overlays,
            );
        }
    }

    fn launch_luma<E: CubePrimitive, R: Runtime>(
        &self,
        client: &ComputeClient<R>,
        dispatch: &impl Dispatch,
        planes: HeatmapPlanes<'_>,
        values: DevicePlane<'_>,
        encoding: Encoding,
    ) {
        let (weight_red, weight_blue) = self.painting.coefficients.weights();
        let luma = planes.picture[0].array::<R>();
        let values_array = values.array::<R>();
        let palette = self.palette::<R>();
        let luma_out = planes.output[0].array::<R>();
        let samples = values.samples();
        let cubes = dispatch.cube_count(samples);
        let unit = dispatch.cube_dim();
        let overlays = self.overlays();

        // Safety: the luma, map value and luma output planes hold `width * height` samples, checked
        // by `apply`, and the palette holds its `PALETTE_VALUES` values.
        unsafe {
            paint_luma::launch_unchecked::<E, R>(
                client,
                cubes,
                unit,
                luma,
                values_array,
                palette,
                luma_out,
                self.painting.scale,
                self.painting.floor,
                encoding.luma_offset,
                encoding.luma_scale,
                weight_red as f32,
                weight_blue as f32,
                encoding.largest,
                encoding.rounds,
                overlays,
            );
        }
    }

    fn launch_chroma<E: CubePrimitive, R: Runtime>(
        &self,
        client: &ComputeClient<R>,
        dispatch: &impl Dispatch,
        planes: HeatmapPlanes<'_>,
        values: DevicePlane<'_>,
        encoding: Encoding,
        shifts: (u32, u32),
    ) {
        let (width_shift, height_shift) = shifts;
        let (weight_red, weight_blue) = self.painting.coefficients.weights();
        let width = values.width();
        let height = values.height();
        let chroma_width = planes.output[1].width();
        let luma = planes.picture[0].array::<R>();
        let values_array = values.array::<R>();
        let palette = self.palette::<R>();
        let blue_out = planes.output[1].array::<R>();
        let red_out = planes.output[2].array::<R>();
        let samples = planes.output[1].samples();
        let cubes = dispatch.cube_count(samples);
        let unit = dispatch.cube_dim();
        let overlays = self.overlays();

        // Safety: the luma and map value planes hold `width * height` samples and both chroma
        // outputs hold `chroma_width` by the shifted height, checked by `apply`. Every pixel read
        // is clipped to the frame, and the palette holds its `PALETTE_VALUES` values.
        unsafe {
            paint_chroma::launch_unchecked::<E, R>(
                client,
                cubes,
                unit,
                luma,
                values_array,
                palette,
                blue_out,
                red_out,
                width,
                height,
                chroma_width,
                width_shift,
                height_shift,
                self.painting.scale,
                self.painting.floor,
                encoding.luma_offset,
                encoding.luma_scale,
                encoding.chroma_offset,
                encoding.chroma_scale,
                weight_red as f32,
                weight_blue as f32,
                encoding.largest,
                encoding.rounds,
                overlays,
            );
        }
    }
}

/// How strongly a map `value` paints, between 0 and 1. A NaN paints nothing.
#[cube]
fn strength(value: f32, scale: f32, floor: f32) -> f32 {
    let defined = select(f32::is_nan(value), 0.0, value);
    let lifted = (defined / scale - floor) / (1.0 - floor);

    f32::clamp(lifted, 0.0, 1.0)
}

/// A code value as a fraction of its span, clamped to between 0 and 1.
#[cube]
fn normalised(code: f32, offset: f32, scale: f32) -> f32 {
    let fraction = (code - offset) / scale;

    f32::clamp(fraction, 0.0, 1.0)
}

/// One channel of the colour a pixel of brightness `grey` paints with at `strength`.
///
/// `overlays` picks how the palette is read, as an overlay shading or as a multiply shading.
#[cube]
fn colour_channel(
    palette: &Array<f32>,
    channel: u32,
    grey: f32,
    strength: f32,
    #[comptime] overlays: bool,
) -> f32 {
    if overlays {
        overlaid_channel(palette, channel, grey, strength)
    } else {
        multiplied_channel(palette, channel, grey, strength)
    }
}

/// One channel of the palette's colour mixed over a pixel of brightness `grey` at `strength`.
#[cube]
fn overlaid_channel(palette: &Array<f32>, channel: u32, grey: f32, strength: f32) -> f32 {
    let cover = palette[OPACITY as usize] * strength;

    grey * (1.0 - cover) + palette[channel as usize] * cover
}

/// One channel of the palette's ramp at `strength`, times a pixel of brightness `grey`.
#[cube]
fn multiplied_channel(palette: &Array<f32>, channel: u32, grey: f32, strength: f32) -> f32 {
    let brightness = palette[BASE as usize] + palette[GAIN as usize] * grey;
    let mut from = palette[(MIDDLE_STOP + channel) as usize];
    let mut to = palette[(LAST_STOP + channel) as usize];
    let mut fraction = (strength - 0.5) * 2.0;

    if strength <= 0.5 {
        from = palette[channel as usize];
        to = palette[(MIDDLE_STOP + channel) as usize];
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
    value: f32,
    palette: &Array<f32>,
    scale: f32,
    floor: f32,
    luma_offset: f32,
    luma_scale: f32,
    weight_red: f32,
    weight_blue: f32,
    #[comptime] overlays: bool,
) -> (f32, f32, f32) {
    let grey = normalised(luma_code, luma_offset, luma_scale);
    let pixel_strength = strength(value, scale, floor);
    let red = colour_channel(palette, 0, grey, pixel_strength, overlays);
    let green = colour_channel(palette, 1, grey, pixel_strength, overlays);
    let blue = colour_channel(palette, 2, grey, pixel_strength, overlays);

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
    values: &Array<f32>,
    palette: &Array<f32>,
    red_out: &mut Array<E>,
    green_out: &mut Array<E>,
    blue_out: &mut Array<E>,
    scale: f32,
    floor: f32,
    offset: f32,
    span: f32,
    largest: f32,
    #[comptime] rounds: bool,
    #[comptime] overlays: bool,
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
    let pixel_strength = strength(values[ABSOLUTE_POS], scale, floor);

    let red_colour = colour_channel(palette, 0, grey, pixel_strength, overlays);
    let green_colour = colour_channel(palette, 1, grey, pixel_strength, overlays);
    let blue_colour = colour_channel(palette, 2, grey, pixel_strength, overlays);
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
    values: &Array<f32>,
    palette: &Array<f32>,
    luma_out: &mut Array<E>,
    scale: f32,
    floor: f32,
    luma_offset: f32,
    luma_scale: f32,
    weight_red: f32,
    weight_blue: f32,
    largest: f32,
    #[comptime] rounds: bool,
    #[comptime] overlays: bool,
) {
    if ABSOLUTE_POS >= luma_out.len() {
        terminate!();
    }

    let luma_code = f32::cast_from(luma[ABSOLUTE_POS]);
    let (painted, _, _) = painted_yuv(
        luma_code,
        values[ABSOLUTE_POS],
        palette,
        scale,
        floor,
        luma_offset,
        luma_scale,
        weight_red,
        weight_blue,
        overlays,
    );
    let code = encoded(painted, luma_offset, luma_scale, largest, rounds);

    luma_out[ABSOLUTE_POS] = E::cast_from(code);
}

/// Writes each chroma sample as the average chroma of the painted pixels it covers inside the frame.
#[cube(launch_unchecked)]
fn paint_chroma<E: CubePrimitive>(
    luma: &Array<E>,
    values: &Array<f32>,
    palette: &Array<f32>,
    blue_out: &mut Array<E>,
    red_out: &mut Array<E>,
    width: u32,
    height: u32,
    chroma_width: u32,
    width_shift: u32,
    height_shift: u32,
    scale: f32,
    floor: f32,
    luma_offset: f32,
    luma_scale: f32,
    chroma_offset: f32,
    chroma_scale: f32,
    weight_red: f32,
    weight_blue: f32,
    largest: f32,
    #[comptime] rounds: bool,
    #[comptime] overlays: bool,
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
                values[index],
                palette,
                scale,
                floor,
                luma_offset,
                luma_scale,
                weight_red,
                weight_blue,
                overlays,
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
