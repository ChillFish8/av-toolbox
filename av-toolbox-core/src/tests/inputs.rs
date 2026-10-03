use std::sync::Arc;

use av_toolbox_gpu::{
    BufferPool,
    Bytes,
    ColourInfo,
    DeviceAllocator,
    DeviceRead,
    Frame,
    FrameFormat,
    FrameLayout,
    FrameMeta,
    HostFrame,
    HostPool,
    HostSide,
    HostStreamItem,
    Mask,
    MaskFormat,
    SampleFormat,
    StreamItem,
    StreamPolicy,
    Subsampling,
};
use cubecl::server::Handle;
use cubecl::stream_id::StreamId;

use crate::{
    BindInputsError,
    InputArgument,
    InputParam,
    InputSignature,
    Inputs,
    StageHandle,
    StreamType,
    bind_inputs,
};

/// A [DeviceAllocator] that reserves nothing real, so `Inputs` can be tested without a device.
#[derive(Debug)]
struct FakeAllocator;

impl DeviceAllocator for FakeAllocator {
    fn empty(&self, bytes: usize) -> Handle {
        Handle::new(StreamId::current(), bytes as u64)
    }

    fn upload(&self, bytes: &[u8]) -> Handle {
        self.empty(bytes.len())
    }

    fn alignment(&self) -> usize {
        1
    }

    fn flush(&self) {}

    /// Fills buffer `position` with the byte `position` repeated, so two same-sized handles
    /// still come back as distinguishable buffers.
    fn read(&self, handles: Vec<Handle>) -> DeviceRead {
        let buffers = handles
            .into_iter()
            .enumerate()
            .map(|(position, handle)| {
                Bytes::from_bytes_vec(vec![position as u8; handle.size_in_used() as usize])
            })
            .collect();

        Box::pin(async move { Ok(buffers) })
    }
}

fn pool() -> BufferPool {
    let allocator = Arc::new(FakeAllocator);

    BufferPool::new(allocator, StreamPolicy::SharedStream)
}

fn frame() -> Frame {
    let format = FrameFormat {
        width: 4,
        height: 4,
        layout: FrameLayout::Yuv {
            subsampling: Subsampling::Yuv420,
        },
        sample: SampleFormat::U8,
        colour: ColourInfo::default(),
        frame_rate: None,
    };

    pool().alloc_frame(&format, FrameMeta::at(0))
}

fn mask() -> Mask {
    let format = MaskFormat {
        width: 4,
        height: 4,
        sample: SampleFormat::U8,
        chroma: None,
    };

    pool().alloc_mask(&format, FrameMeta::at(0))
}

#[test]
fn a_taken_slot_is_empty_afterwards() {
    let mut inputs = Inputs::new(vec![Some(StreamItem::Frame(frame()))]);

    assert!(inputs.take(0).is_some());
    assert!(inputs.take(0).is_none());
}

#[test]
fn take_frame_on_a_mask_slot_returns_none_and_leaves_it_in_place() {
    let mut inputs = Inputs::new(vec![Some(StreamItem::Mask(mask()))]);

    assert!(inputs.take_frame(0).is_none());
    assert!(inputs.take_mask(0).is_some());
}

#[test]
fn take_mask_on_a_frame_slot_returns_none_and_leaves_it_in_place() {
    let mut inputs = Inputs::new(vec![Some(StreamItem::Frame(frame()))]);

    assert!(inputs.take_mask(0).is_none());
    assert!(inputs.take_frame(0).is_some());
}

#[test]
fn the_index_is_the_signature_slot_rather_than_a_compacted_one() {
    let mut inputs = Inputs::new(vec![None, Some(StreamItem::Frame(frame()))]);

    assert!(inputs.take_frame(0).is_none());
    assert!(inputs.take_frame(1).is_some());
}

#[test]
fn len_and_is_empty_report_the_slot_count_not_how_many_are_filled() {
    let inputs = Inputs::new(vec![None, Some(StreamItem::Frame(frame()))]);

    assert_eq!(inputs.len(), 2);
    assert!(!inputs.is_empty());
    assert!(Inputs::new(Vec::new()).is_empty());
}

const REFERENCE: &InputSignature =
    &InputSignature::new(&[InputParam::required("reference"), InputParam::optional("mask")]);

const VARIADIC: &InputSignature = &InputSignature::with_variadic(&[InputParam::required("reference")]);

fn handle(id: u64) -> StageHandle {
    StageHandle::new(id)
}

#[test]
fn positional_inputs_fill_params_in_order() {
    let arguments = [
        InputArgument::positional(handle(1), StreamType::Frames),
        InputArgument::positional(handle(2), StreamType::Frames),
    ];
    let bound = bind_inputs(REFERENCE, &arguments).expect("should bind");

    assert_eq!(bound.get("reference"), Some(handle(1)));
    assert_eq!(bound.get("mask"), Some(handle(2)));
    assert_eq!(bound.positional(0), Some(handle(1)));
}

#[test]
fn named_inputs_bind_by_name_in_any_order() {
    let arguments = [
        InputArgument::named("mask", handle(2), StreamType::Frames),
        InputArgument::named("reference", handle(1), StreamType::Frames),
    ];
    let bound = bind_inputs(REFERENCE, &arguments).expect("should bind");

    assert_eq!(bound.get("reference"), Some(handle(1)));
    assert_eq!(bound.get("mask"), Some(handle(2)));
}

#[test]
fn positional_and_named_inputs_mix() {
    let arguments = [
        InputArgument::positional(handle(1), StreamType::Frames),
        InputArgument::named("mask", handle(2), StreamType::Frames),
    ];
    let bound = bind_inputs(REFERENCE, &arguments).expect("should bind");

    assert_eq!(bound.get("reference"), Some(handle(1)));
    assert_eq!(bound.get("mask"), Some(handle(2)));
}

#[test]
fn an_omitted_optional_input_is_unbound() {
    let arguments = [InputArgument::positional(handle(1), StreamType::Frames)];
    let bound = bind_inputs(REFERENCE, &arguments).expect("should bind");

    assert_eq!(bound.get("mask"), None);
    assert_eq!(bound.iter().count(), 1);
}

#[test]
fn a_positional_input_after_a_named_input_is_rejected() {
    let arguments = [
        InputArgument::named("reference", handle(1), StreamType::Frames),
        InputArgument::positional(handle(2), StreamType::Frames),
    ];
    let error = bind_inputs(REFERENCE, &arguments).expect_err("should not bind");

    assert!(matches!(
        error,
        BindInputsError::PositionalAfterNamed { position: 1 },
    ));
}

#[test]
fn naming_an_already_filled_param_is_rejected() {
    let arguments = [
        InputArgument::positional(handle(1), StreamType::Frames),
        InputArgument::named("reference", handle(2), StreamType::Frames),
    ];
    let error = bind_inputs(REFERENCE, &arguments).expect_err("should not bind");

    assert!(matches!(
        error,
        BindInputsError::DuplicateInput { position: 1, .. },
    ));
}

#[test]
fn a_missing_required_input_is_rejected() {
    let arguments = [InputArgument::named("mask", handle(2), StreamType::Frames)];
    let error = bind_inputs(REFERENCE, &arguments).expect_err("should not bind");

    assert!(matches!(
        error,
        BindInputsError::MissingRequiredInput { name: "reference" },
    ));
}

#[test]
fn extra_positional_inputs_are_rejected_without_a_variadic_tail() {
    let arguments = [
        InputArgument::positional(handle(1), StreamType::Frames),
        InputArgument::positional(handle(2), StreamType::Frames),
        InputArgument::positional(handle(3), StreamType::Frames),
    ];
    let error = bind_inputs(REFERENCE, &arguments).expect_err("should not bind");

    assert!(matches!(
        error,
        BindInputsError::TooManyInputs {
            position: 2,
            expected: 2,
        },
    ));
}

#[test]
fn an_unknown_named_input_is_rejected_without_a_variadic_tail() {
    let arguments = [
        InputArgument::positional(handle(1), StreamType::Frames),
        InputArgument::named("nonsense", handle(2), StreamType::Frames),
    ];
    let error = bind_inputs(REFERENCE, &arguments).expect_err("should not bind");

    assert!(matches!(error, BindInputsError::UnknownInput { position: 1, .. },));
}

#[test]
fn a_variadic_tail_collects_extra_inputs_in_order() {
    let arguments = [
        InputArgument::positional(handle(1), StreamType::Frames),
        InputArgument::positional(handle(2), StreamType::Frames),
        InputArgument::named("overlay", handle(3), StreamType::Frames),
    ];
    let bound = bind_inputs(VARIADIC, &arguments).expect("should bind");

    assert_eq!(bound.get("reference"), Some(handle(1)));
    assert_eq!(bound.variadic().len(), 2);
    assert_eq!(bound.variadic()[0].name(), None);
    assert_eq!(bound.variadic()[0].handle(), handle(2));
    assert_eq!(bound.variadic()[1].name(), Some("overlay"));
}

#[test]
fn an_empty_signature_binds_no_arguments() {
    let bound = bind_inputs(&InputSignature::EMPTY, &[]).expect("should bind");

    assert_eq!(bound.iter().count(), 0);
    assert!(bound.variadic().is_empty());
    assert!(InputSignature::EMPTY.is_empty());
}

#[test]
fn a_signature_records_whether_it_has_a_variadic_tail() {
    const PLAIN: &InputSignature = &InputSignature::new(&[InputParam::required("base")]);
    const TAILED: &InputSignature = &InputSignature::with_variadic(&[InputParam::required("base")]);

    assert!(!PLAIN.is_variadic());
    assert!(TAILED.is_variadic());
}

#[test]
fn a_param_accepts_either_type_until_it_is_narrowed() {
    const ANY: InputParam = InputParam::required("reference");
    const FRAMES: InputParam = InputParam::required("reference").accepting(StreamType::Frames);

    assert_eq!(ANY.accepts(), None);
    assert_eq!(FRAMES.accepts(), Some(StreamType::Frames));
}

#[test]
fn narrowing_a_param_keeps_whether_it_is_required() {
    const PARAM: InputParam = InputParam::optional("overlay").accepting(StreamType::Mask);

    assert!(!PARAM.is_required());
    assert_eq!(PARAM.name(), "overlay");
    assert_eq!(PARAM.accepts(), Some(StreamType::Mask));
}

#[test]
fn a_param_binds_an_argument_of_the_type_it_declares() {
    const SIGNATURE: &InputSignature =
        &InputSignature::new(&[InputParam::required("reference").accepting(StreamType::Frames)]);
    let arguments = [InputArgument::positional(handle(1), StreamType::Frames)];

    let bound = bind_inputs(SIGNATURE, &arguments).expect("should bind");

    assert_eq!(bound.get("reference"), Some(handle(1)));
}

#[test]
fn a_param_rejects_an_argument_of_another_type() {
    const SIGNATURE: &InputSignature =
        &InputSignature::new(&[InputParam::required("overlay").accepting(StreamType::Mask)]);
    let arguments = [InputArgument::positional(handle(1), StreamType::Frames)];

    let error = bind_inputs(SIGNATURE, &arguments).expect_err("should reject");

    assert!(matches!(
        error,
        BindInputsError::InputTypeMismatch {
            name: Some("overlay"),
            position: 0,
            expected: StreamType::Mask,
            found: StreamType::Frames,
        }
    ));
}

#[test]
fn a_param_with_no_declared_type_takes_either() {
    const SIGNATURE: &InputSignature = &InputSignature::new(&[InputParam::required("either")]);

    let frames = [InputArgument::positional(handle(1), StreamType::Frames)];
    let mask = [InputArgument::positional(handle(1), StreamType::Mask)];

    assert!(bind_inputs(SIGNATURE, &frames).is_ok());
    assert!(bind_inputs(SIGNATURE, &mask).is_ok());
}

#[test]
fn an_optional_param_enforces_its_type_when_it_is_supplied() {
    const SIGNATURE: &InputSignature =
        &InputSignature::new(&[InputParam::optional("overlay").accepting(StreamType::Mask)]);

    let empty: [InputArgument; 0] = [];
    assert!(bind_inputs(SIGNATURE, &empty).is_ok());

    let wrong = [InputArgument::named("overlay", handle(1), StreamType::Frames)];
    let error = bind_inputs(SIGNATURE, &wrong).expect_err("should reject");

    assert!(matches!(
        error,
        BindInputsError::InputTypeMismatch {
            name: Some("overlay"),
            expected: StreamType::Mask,
            found: StreamType::Frames,
            ..
        }
    ));
}

#[test]
fn the_variadic_tail_enforces_its_type() {
    const SIGNATURE: &InputSignature = &InputSignature::with_variadic_of(
        &[InputParam::required("base").accepting(StreamType::Frames)],
        StreamType::Frames,
    );
    let arguments = [
        InputArgument::positional(handle(1), StreamType::Frames),
        InputArgument::positional(handle(2), StreamType::Mask),
    ];

    let error = bind_inputs(SIGNATURE, &arguments).expect_err("should reject");

    assert!(matches!(
        error,
        BindInputsError::InputTypeMismatch {
            name: None,
            position: 1,
            expected: StreamType::Frames,
            found: StreamType::Mask,
        }
    ));
}

#[test]
fn a_tail_that_accepts_anything_is_not_the_same_as_having_no_tail() {
    const NONE: &InputSignature = &InputSignature::new(&[]);
    const ANY: &InputSignature = &InputSignature::with_variadic(&[]);
    const MASKS: &InputSignature = &InputSignature::with_variadic_of(&[], StreamType::Mask);

    assert!(!NONE.is_variadic());
    assert_eq!(NONE.variadic_accepts(), None);

    assert!(ANY.is_variadic());
    assert_eq!(ANY.variadic_accepts(), None);

    assert!(MASKS.is_variadic());
    assert_eq!(MASKS.variadic_accepts(), Some(StreamType::Mask));
}

#[test]
fn a_mask_input_can_ask_for_the_planes_of_a_frame_input() {
    let param = InputParam::required("mask")
        .accepting(StreamType::Mask)
        .planes_of("base");

    assert_eq!(param.planes_of_input(), Some("base"));
}

#[test]
fn an_input_asks_for_no_planes_by_default() {
    let param = InputParam::required("mask");

    assert_eq!(param.planes_of_input(), None);
}

#[test]
fn a_param_is_not_the_chunk_authority_unless_it_says_so() {
    let param = InputParam::required("reference");

    assert!(!param.is_chunk_authority());
}

#[test]
fn a_marked_param_is_the_chunk_authority() {
    let param = InputParam::required("reference").chunk_authority();

    assert!(param.is_chunk_authority());
}

#[test]
fn a_signature_reports_which_param_was_marked() {
    const PARAMS: &[InputParam] = &[
        InputParam::required("mask").accepting(StreamType::Mask),
        InputParam::required("reference").chunk_authority(),
    ];
    let signature = InputSignature::new(PARAMS);

    assert_eq!(signature.chunk_authority(), Some(1));
}

#[test]
fn a_signature_marking_nothing_reports_nothing() {
    const PARAMS: &[InputParam] = &[InputParam::required("reference")];
    let signature = InputSignature::new(PARAMS);

    assert_eq!(signature.chunk_authority(), None);
}

#[test]
#[should_panic(expected = "only one")]
fn a_signature_cannot_mark_two_params() {
    const PARAMS: &[InputParam] = &[
        InputParam::required("reference").chunk_authority(),
        InputParam::required("other").chunk_authority(),
    ];

    let _ = InputSignature::new(PARAMS);
}

#[test]
#[should_panic(expected = "only one")]
fn a_variadic_signature_cannot_mark_two_params_either() {
    const PARAMS: &[InputParam] = &[
        InputParam::required("reference").chunk_authority(),
        InputParam::required("other").chunk_authority(),
    ];

    let _ = InputSignature::with_variadic(PARAMS);
}

#[test]
fn a_marked_param_wins_over_an_earlier_frame_input() {
    const PARAMS: &[InputParam] = &[
        InputParam::required("first"),
        InputParam::required("reference").chunk_authority(),
    ];
    let signature = InputSignature::new(PARAMS);
    let arguments = [
        InputArgument::positional(StageHandle::new(1), StreamType::Frames),
        InputArgument::positional(StageHandle::new(2), StreamType::Frames),
    ];

    let bound = bind_inputs(&signature, &arguments).expect("should bind");

    assert_eq!(bound.chunk_authority(), Some(1));
}

#[test]
fn an_unmarked_signature_lines_up_against_its_first_frame_input() {
    const PARAMS: &[InputParam] = &[
        InputParam::required("mask").accepting(StreamType::Mask),
        InputParam::required("reference"),
    ];
    let signature = InputSignature::new(PARAMS);
    let arguments = [
        InputArgument::positional(StageHandle::new(1), StreamType::Mask),
        InputArgument::positional(StageHandle::new(2), StreamType::Frames),
    ];

    let bound = bind_inputs(&signature, &arguments).expect("should bind");

    assert_eq!(bound.chunk_authority(), Some(1));
}

/// An optional param nobody filled carries no chunk ids, so the rule has to move on.
#[test]
fn a_marked_param_nobody_bound_falls_back() {
    const PARAMS: &[InputParam] = &[
        InputParam::required("reference"),
        InputParam::optional("guide").chunk_authority(),
    ];
    let signature = InputSignature::new(PARAMS);
    let arguments = [InputArgument::positional(StageHandle::new(1), StreamType::Frames)];

    let bound = bind_inputs(&signature, &arguments).expect("should bind");

    assert_eq!(bound.chunk_authority(), Some(0));
}

#[test]
fn inputs_that_are_all_masks_line_up_against_the_first_of_them() {
    const PARAMS: &[InputParam] = &[
        InputParam::required("left").accepting(StreamType::Mask),
        InputParam::required("right").accepting(StreamType::Mask),
    ];
    let signature = InputSignature::new(PARAMS);
    let arguments = [
        InputArgument::positional(StageHandle::new(1), StreamType::Mask),
        InputArgument::positional(StageHandle::new(2), StreamType::Mask),
    ];

    let bound = bind_inputs(&signature, &arguments).expect("should bind");

    assert_eq!(bound.chunk_authority(), Some(0));
}

#[test]
fn a_variadic_input_can_be_the_authority_when_no_param_was_bound() {
    const PARAMS: &[InputParam] = &[InputParam::optional("guide")];
    let signature = InputSignature::with_variadic(PARAMS);
    let arguments = [InputArgument::named(
        "extra",
        StageHandle::new(1),
        StreamType::Frames,
    )];

    let bound = bind_inputs(&signature, &arguments).expect("should bind");

    assert_eq!(bound.chunk_authority(), Some(1));
}

#[test]
fn a_stage_with_no_inputs_has_no_authority() {
    let bound = bind_inputs(&InputSignature::EMPTY, &[]).expect("should bind");

    assert_eq!(bound.chunk_authority(), None);
}

fn host_frame(index: u64) -> HostStreamItem {
    let format = FrameFormat {
        width: 16,
        height: 16,
        layout: FrameLayout::Grey,
        sample: SampleFormat::U8,
        colour: ColourInfo::default(),
        frame_rate: None,
    };

    let frame = HostPool::new()
        .alloc_frame(&format, FrameMeta::at(index))
        .freeze();

    HostStreamItem::Frame(frame)
}

#[test]
fn host_inputs_hand_out_host_frames() {
    let mut inputs = Inputs::<HostSide>::from_slots(vec![None, Some(host_frame(6))]);

    let frame = inputs.take_frame(1).expect("a host frame in slot 1");

    assert_eq!(frame.meta().index, 6);
    assert!(inputs.take_frame(1).is_none());
}

#[test]
fn a_host_filter_is_written_against_host_frames() {
    struct Brighten;

    impl crate::StageInstance for Brighten {}

    impl crate::Filter<HostSide> for Brighten {
        fn process(
            &mut self,
            _context: &mut av_toolbox_gpu::FrameContext,
            mut inputs: Inputs<HostSide>,
        ) -> Result<Vec<HostFrame>, crate::StageError> {
            let frame = inputs.take_frame(0).expect("a frame");
            let mut frame = frame.try_into_mut().expect("nothing else holds it");
            frame
                .plane_mut(av_toolbox_gpu::Channel::Y)
                .expect("grey")
                .fill(200);

            Ok(vec![frame.freeze()])
        }
    }

    let mut context = av_toolbox_gpu::testing_context();
    let inputs = Inputs::<HostSide>::from_slots(vec![Some(host_frame(0))]);

    let mut outputs = crate::Filter::process(&mut Brighten, &mut context, inputs).expect("processes");
    let out = outputs.remove(0);

    assert!(
        out.plane(av_toolbox_gpu::Channel::Y)
            .expect("grey")
            .bytes()
            .iter()
            .all(|&byte| byte == 200)
    );
}
