use av_toolbox_core::StageError;
use av_toolbox_cubek::DevicePlane;
use av_toolbox_cubek::aa::{Eedi2Settings, Eedi3DeviceClips, Eedi3Settings, sangnom_aaf, sangnom_max};
use av_toolbox_cubek::deinterlace::{BwdifField, BwdifPlanes, bwdif_field};
use av_toolbox_gpu::{
    Channel,
    ComputeClient,
    Frame,
    FrameContext,
    FrameFormat,
    FrameLayout,
    FrameMeta,
    Plane,
    Runtime,
    SampleFormat,
};

use crate::aa::direction::FieldPass;
use crate::aa::eedi2::{Eedi2Pass, subsampling_shifts};
use crate::aa::eedi3::Eedi3FieldPass;
use crate::aa::sangnom::SangnomPass;
use crate::codes::CodeValues;
use crate::deinterlace::spatial::Predictor;
use crate::scratch::Scratch;
use crate::sizing::Sizing;

fn missing_channel(stage: &str, channel: Channel) -> StageError {
    let message = format!("{stage} was handed a frame missing {channel}");

    StageError::msg(message)
}

/// The three frames one output reads.
pub(crate) struct Window<'a> {
    pub(crate) previous: &'a Frame,
    pub(crate) current: &'a Frame,
    pub(crate) next: &'a Frame,
}

/// A frame's plane for `channel`.
pub(crate) fn channel_plane<'a>(
    stage: &str,
    frame: &'a Frame,
    channel: Channel,
) -> Result<&'a Plane, StageError> {
    match frame.plane(channel) {
        Some(plane) => Ok(plane),
        None => Err(missing_channel(stage, channel)),
    }
}

/// Widens `source` into a new f32 scratch plane of code values.
pub(crate) fn widened_plane<R: Runtime>(
    client: &ComputeClient<R>,
    context: &FrameContext,
    source: &Plane,
    codes: CodeValues,
    channel: Channel,
) -> Scratch {
    let (width, height) = source.dimensions();
    let values = Scratch::new::<f32>(context, width, height);
    let sizing = Sizing(context);

    let source_handle = source.buffer().handle();

    // Safety: a frame's plane holds exactly `width * height` samples of its sample format.
    let source_plane = unsafe { DevicePlane::new(source_handle, width, height) };
    let values_plane = values.plane();

    codes.widen(client, &sizing, source_plane, values_plane, channel);

    values
}

/// Runs `pass` on `current` once for each of `fields`, keeping the rows that field keeps.
///
/// It gives one new plane per field, in the order of `fields`.
fn run_fields(
    pass: &impl FieldPass,
    context: &FrameContext,
    current: DevicePlane<'_>,
    fields: &[BwdifField],
) -> Vec<Scratch> {
    let width = current.width();
    let height = current.height();
    let outputs: Vec<Scratch> = fields
        .iter()
        .map(|_| Scratch::new::<f32>(context, width, height))
        .collect();

    if let [first, _] = fields {
        // The pass keeping the even rows serves the field rebuilding the odd rows.
        let (even, odd) = if first.rebuilt == 1 {
            (&outputs[0], &outputs[1])
        } else {
            (&outputs[1], &outputs[0])
        };

        pass.run_both(context, current, even.plane(), odd.plane());
        return outputs;
    }

    for (field, output) in fields.iter().zip(&outputs) {
        let kept = 1 - field.rebuilt;
        pass.run(context, current, output.plane(), kept);
    }

    outputs
}

/// Rebuilds the planes of interlaced frames the way bwdif does.
pub(crate) struct FieldRebuilder<R: Runtime> {
    stage: &'static str,
    client: ComputeClient<R>,
    input: FrameFormat,
    predictor: Predictor,
    codes: CodeValues,
}

impl<R: Runtime> FieldRebuilder<R> {
    /// A rebuilder for frames of `input` storing its results as `output`, whose errors are reported
    /// under `stage`.
    pub(crate) fn new(
        stage: &'static str,
        client: ComputeClient<R>,
        input: FrameFormat,
        output: SampleFormat,
        predictor: Predictor,
    ) -> Self {
        let codes = CodeValues::new(&input, output);

        Self {
            stage,
            client,
            input,
            predictor,
            codes,
        }
    }

    pub(crate) fn client(&self) -> &ComputeClient<R> {
        &self.client
    }

    #[expect(
        dead_code,
        reason = "the stages choosing their output sample read the codes here"
    )]
    pub(crate) fn codes(&self) -> CodeValues {
        self.codes
    }

    /// Widens `source` into a new f32 scratch plane.
    pub(crate) fn widened(&self, context: &FrameContext, source: &Plane, channel: Channel) -> Scratch {
        widened_plane(&self.client, context, source, self.codes, channel)
    }

    /// Stores `values` into a new plane of the output's sample format.
    pub(crate) fn stored(
        &self,
        context: &FrameContext,
        values: &Scratch,
        meta: &FrameMeta,
        channel: Channel,
    ) -> Result<Plane, StageError> {
        let device_values = values.plane();
        let width = device_values.width();
        let height = device_values.height();
        let plane_format = FrameFormat {
            width,
            height,
            layout: FrameLayout::Grey,
            sample: self.codes.output_sample(),
            ..self.input
        };
        let target = context.alloc_frame(&plane_format, meta.clone());
        let output = channel_plane(self.stage, &target, Channel::Y)?;
        let sizing = Sizing(context);

        let target_handle = output.buffer().handle();

        // Safety: a frame's plane holds exactly `width * height` samples of its sample format.
        let target_plane = unsafe { DevicePlane::new(target_handle, width, height) };

        self.codes
            .store(&self.client, &sizing, device_values, target_plane, channel);

        Ok(output.clone())
    }

    /// Runs the predictor's field pass on `current` for each of `fields`, one plane per field.
    ///
    /// The cubic predictor has no field pass and gives no planes.
    fn predictions(
        &self,
        context: &FrameContext,
        plane_index: usize,
        current: DevicePlane<'_>,
        fields: &[BwdifField],
    ) -> Vec<Scratch> {
        match self.predictor {
            Predictor::Cubic => Vec::new(),
            Predictor::Eedi2(arguments) => {
                let layout = self.input.layout;
                let (across, _) = subsampling_shifts(layout, plane_index);
                let settings = Eedi2Settings::new(&arguments, self.codes.bits(), across);
                let pass = Eedi2Pass {
                    client: &self.client,
                    settings,
                };

                run_fields(&pass, context, current, fields)
            },
            Predictor::Eedi3(arguments) => {
                let settings = Eedi3Settings::new(&arguments, self.codes.bits());
                let pass = Eedi3FieldPass {
                    client: &self.client,
                    settings,
                    clips: Eedi3DeviceClips::default(),
                };

                run_fields(&pass, context, current, fields)
            },
            Predictor::Sangnom { aa } => {
                let aaf = sangnom_aaf(aa, self.codes.bits());
                let max = sangnom_max(self.codes.bits());
                let pass = SangnomPass {
                    client: &self.client,
                    aaf,
                    max,
                };

                run_fields(&pass, context, current, fields)
            },
        }
    }

    /// Runs the predictor's field pass on `current` for both its fields, the field keeping the even
    /// rows first.
    ///
    /// The cubic predictor has no field pass and gives no planes.
    pub(crate) fn both_fields(
        &self,
        context: &FrameContext,
        plane_index: usize,
        current: DevicePlane<'_>,
    ) -> Vec<Scratch> {
        let even_kept = BwdifField {
            rebuilt: 1,
            first: true,
        };
        let odd_kept = BwdifField {
            rebuilt: 0,
            first: false,
        };

        self.predictions(context, plane_index, current, &[even_kept, odd_kept])
    }

    /// Runs the predictor's field pass on `current`, keeping the rows whose parity is `kept`.
    ///
    /// The cubic predictor has no field pass and gives `None`.
    pub(crate) fn one_field(
        &self,
        context: &FrameContext,
        plane_index: usize,
        current: DevicePlane<'_>,
        kept: u32,
    ) -> Option<Scratch> {
        let field = BwdifField {
            rebuilt: 1 - kept,
            first: true,
        };

        let mut predicted = self.predictions(context, plane_index, current, &[field]);
        predicted.pop()
    }

    /// Rebuilds plane `plane_index` of the window's current frame once for each of `fields`.
    pub(crate) fn rebuild_plane(
        &self,
        context: &FrameContext,
        window: &Window<'_>,
        plane_index: usize,
        fields: &[BwdifField],
    ) -> Result<Vec<Plane>, StageError> {
        let channels = self.input.channels();
        let channel = channels[plane_index];
        let previous_source = channel_plane(self.stage, window.previous, channel)?;
        let current_source = channel_plane(self.stage, window.current, channel)?;
        let next_source = channel_plane(self.stage, window.next, channel)?;

        let previous = self.widened(context, previous_source, channel);
        let current = self.widened(context, current_source, channel);
        let next = self.widened(context, next_source, channel);
        let (width, height) = current_source.dimensions();
        let sizing = Sizing(context);
        let meta = window.current.meta();
        let current_plane = current.plane();
        let predictions = self.predictions(context, plane_index, current_plane, fields);

        let mut rebuilt = Vec::with_capacity(fields.len());
        for (index, field) in fields.iter().enumerate() {
            let output = Scratch::new::<f32>(context, width, height);
            let spatial = predictions.get(index).map(Scratch::plane);
            let planes = BwdifPlanes {
                previous: previous.plane(),
                current: current.plane(),
                next: next.plane(),
                spatial,
                output: output.plane(),
            };
            bwdif_field(&self.client, &sizing, planes, *field);

            let plane = self.stored(context, &output, meta, channel)?;
            rebuilt.push(plane);
        }

        Ok(rebuilt)
    }
}
