use av_toolbox_gpu::{Frame, Mask, StreamItem};

use crate::{BindInputsError, StreamType};

/// An opaque reference to another stage in the pipeline.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct StageHandle(u64);

impl StageHandle {
    pub const fn new(id: u64) -> Self {
        Self(id)
    }

    pub const fn id(&self) -> u64 {
        self.0
    }
}

/// A single named input a stage accepts.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct InputParam {
    name: &'static str,
    required: bool,
    accepts: Option<StreamType>,
}

impl InputParam {
    pub const fn required(name: &'static str) -> Self {
        Self {
            name,
            required: true,
            accepts: None,
        }
    }

    pub const fn optional(name: &'static str) -> Self {
        Self {
            name,
            required: false,
            accepts: None,
        }
    }

    /// Narrows the param to one stream type.
    ///
    /// A param that has not been narrowed takes either type.
    pub const fn accepting(mut self, stream: StreamType) -> Self {
        self.accepts = Some(stream);
        self
    }

    pub const fn name(&self) -> &'static str {
        self.name
    }

    pub const fn is_required(&self) -> bool {
        self.required
    }

    pub const fn accepts(&self) -> Option<StreamType> {
        self.accepts
    }
}

/// The set of inputs a stage accepts, in declaration order.
///
/// A signature reads like a function signature. Params are filled positionally or by name, and an
/// optional variadic tail collects anything left over.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct InputSignature {
    params: &'static [InputParam],
    variadic: bool,
    variadic_accepts: Option<StreamType>,
}

impl InputSignature {
    pub const EMPTY: Self = Self::new(&[]);

    pub const fn new(params: &'static [InputParam]) -> Self {
        Self {
            params,
            variadic: false,
            variadic_accepts: None,
        }
    }

    pub const fn with_variadic(params: &'static [InputParam]) -> Self {
        Self {
            params,
            variadic: true,
            variadic_accepts: None,
        }
    }

    /// A signature whose tail only takes one stream type.
    pub const fn with_variadic_of(params: &'static [InputParam], accepts: StreamType) -> Self {
        Self {
            params,
            variadic: true,
            variadic_accepts: Some(accepts),
        }
    }

    pub const fn params(&self) -> &'static [InputParam] {
        self.params
    }

    pub const fn is_variadic(&self) -> bool {
        self.variadic
    }

    /// What the tail takes, or `None` when it takes either type.
    ///
    /// Only meaningful when [InputSignature::is_variadic] is true.
    pub const fn variadic_accepts(&self) -> Option<StreamType> {
        self.variadic_accepts
    }

    pub const fn is_empty(&self) -> bool {
        self.params.is_empty() && !self.variadic
    }
}

/// An input supplied by a pipeline, either positionally or by name.
///
/// Carries what its producer emits, so binding can check it against the param it fills.
#[derive(Debug, Clone)]
pub struct InputArgument {
    name: Option<String>,
    handle: StageHandle,
    stream: StreamType,
}

impl InputArgument {
    pub fn positional(handle: StageHandle, stream: StreamType) -> Self {
        Self {
            name: None,
            handle,
            stream,
        }
    }

    pub fn named(name: impl Into<String>, handle: StageHandle, stream: StreamType) -> Self {
        Self {
            name: Some(name.into()),
            handle,
            stream,
        }
    }

    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    pub fn handle(&self) -> StageHandle {
        self.handle
    }

    /// What the stage supplying this input produces.
    pub fn stream(&self) -> StreamType {
        self.stream
    }
}

/// An input that landed in a signature's variadic tail.
#[derive(Debug, Clone)]
pub struct VariadicInput {
    name: Option<String>,
    handle: StageHandle,
}

impl VariadicInput {
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    pub fn handle(&self) -> StageHandle {
        self.handle
    }
}

#[derive(Debug, Clone)]
struct BoundParam {
    name: &'static str,
    handle: Option<StageHandle>,
}

/// A stage's inputs after they have been matched to its signature.
#[derive(Debug, Clone)]
pub struct BoundInputs {
    params: Vec<BoundParam>,
    variadic: Vec<VariadicInput>,
}

impl BoundInputs {
    pub fn get(&self, name: &str) -> Option<StageHandle> {
        let bound = self.params.iter().find(|param| param.name == name)?;
        bound.handle
    }

    pub fn positional(&self, index: usize) -> Option<StageHandle> {
        let bound = self.params.get(index)?;
        bound.handle
    }

    pub fn variadic(&self) -> &[VariadicInput] {
        &self.variadic
    }

    pub fn iter(&self) -> impl Iterator<Item = (&'static str, StageHandle)> {
        self.params
            .iter()
            .filter_map(|param| param.handle.map(|handle| (param.name, handle)))
    }
}

/// What a stage was handed this step, one slot per parameter followed by one per variadic input,
/// the same order [ResolvedFormats](av_toolbox_gpu::ResolvedFormats) uses.
///
/// A slot is empty when an optional input was never bound, and once taken it stays empty.
#[derive(Debug)]
pub struct Inputs {
    slots: Vec<Option<StreamItem>>,
}

impl Inputs {
    pub fn new(slots: Vec<Option<StreamItem>>) -> Self {
        Self { slots }
    }

    pub fn take(&mut self, index: usize) -> Option<StreamItem> {
        self.slots.get_mut(index)?.take()
    }

    /// The frame at `index`, leaving the slot empty.
    ///
    /// Returns `None` when the slot was empty and when it held a mask, and leaves a mask in
    /// place rather than consuming it.
    pub fn take_frame(&mut self, index: usize) -> Option<Frame> {
        let slot = self.slots.get_mut(index)?;
        match slot.take()? {
            StreamItem::Frame(frame) => Some(frame),
            other => {
                *slot = Some(other);
                None
            },
        }
    }

    /// The mask at `index`, leaving the slot empty.
    ///
    /// Returns `None` when the slot was empty and when it held a frame, and leaves a frame in
    /// place rather than consuming it.
    pub fn take_mask(&mut self, index: usize) -> Option<Mask> {
        let slot = self.slots.get_mut(index)?;
        match slot.take()? {
            StreamItem::Mask(mask) => Some(mask),
            other => {
                *slot = Some(other);
                None
            },
        }
    }

    pub fn len(&self) -> usize {
        self.slots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }
}

/// Matches a pipeline's inputs against a stage's signature.
///
/// The rules follow Python's arguments and keyword arguments. Positional inputs fill the params
/// left to right, named inputs match a param by name, and anything left over goes to the variadic
/// tail when the signature has one.
pub fn bind_inputs(
    signature: &InputSignature,
    arguments: &[InputArgument],
) -> Result<BoundInputs, BindInputsError> {
    let mut params: Vec<BoundParam> = signature
        .params()
        .iter()
        .map(|param| BoundParam {
            name: param.name(),
            handle: None,
        })
        .collect();

    let mut variadic = Vec::new();
    let mut next_positional = 0;
    let mut seen_named = false;

    for (position, argument) in arguments.iter().enumerate() {
        let Some(name) = argument.name() else {
            if seen_named {
                return Err(BindInputsError::PositionalAfterNamed { position });
            }

            bind_positional(
                &mut params,
                &mut variadic,
                &mut next_positional,
                signature,
                argument,
                position,
            )?;
            continue;
        };

        seen_named = true;
        bind_named(&mut params, &mut variadic, signature, argument, name, position)?;
    }

    for (param, bound) in signature.params().iter().zip(params.iter()) {
        if param.is_required() && bound.handle.is_none() {
            return Err(BindInputsError::MissingRequiredInput { name: param.name() });
        }
    }

    Ok(BoundInputs { params, variadic })
}

fn bind_positional(
    params: &mut [BoundParam],
    variadic: &mut Vec<VariadicInput>,
    next_positional: &mut usize,
    signature: &InputSignature,
    argument: &InputArgument,
    position: usize,
) -> Result<(), BindInputsError> {
    if *next_positional < params.len() {
        let param = &signature.params()[*next_positional];
        check_stream_type(Some(param.name()), param.accepts(), argument, position)?;

        params[*next_positional].handle = Some(argument.handle());
        *next_positional += 1;
        return Ok(());
    }

    if !signature.is_variadic() {
        return Err(BindInputsError::TooManyInputs {
            position,
            expected: params.len(),
        });
    }

    check_stream_type(None, signature.variadic_accepts(), argument, position)?;

    variadic.push(VariadicInput {
        name: None,
        handle: argument.handle(),
    });

    Ok(())
}

fn bind_named(
    params: &mut [BoundParam],
    variadic: &mut Vec<VariadicInput>,
    signature: &InputSignature,
    argument: &InputArgument,
    name: &str,
    position: usize,
) -> Result<(), BindInputsError> {
    let matching = params.iter().position(|param| param.name == name);
    if let Some(index) = matching {
        if params[index].handle.is_some() {
            return Err(BindInputsError::DuplicateInput {
                name: name.to_string(),
                position,
            });
        }

        let param = &signature.params()[index];
        check_stream_type(Some(param.name()), param.accepts(), argument, position)?;

        params[index].handle = Some(argument.handle());
        return Ok(());
    }

    if !signature.is_variadic() {
        return Err(BindInputsError::UnknownInput {
            name: name.to_string(),
            position,
        });
    }

    check_stream_type(None, signature.variadic_accepts(), argument, position)?;

    variadic.push(VariadicInput {
        name: Some(name.to_string()),
        handle: argument.handle(),
    });

    Ok(())
}

/// Rejects an argument whose producer emits a type the param does not take.
fn check_stream_type(
    name: Option<&'static str>,
    accepts: Option<StreamType>,
    argument: &InputArgument,
    position: usize,
) -> Result<(), BindInputsError> {
    let Some(expected) = accepts else {
        return Ok(());
    };

    let found = argument.stream();
    if found == expected {
        return Ok(());
    }

    Err(BindInputsError::InputTypeMismatch {
        name,
        position,
        expected,
        found,
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use av_toolbox_gpu::{
        BufferPool,
        ColourInfo,
        DeviceAllocator,
        FrameFormat,
        FrameLayout,
        FrameMeta,
        MaskFormat,
        SampleFormat,
        Subsampling,
    };
    use cubecl::server::Handle;
    use cubecl::stream_id::StreamId;

    use super::*;

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
    }

    fn pool() -> BufferPool {
        BufferPool::new(Arc::new(FakeAllocator))
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
        };

        pool().alloc_frame(&format, FrameMeta::at(0))
    }

    fn mask() -> Mask {
        let format = MaskFormat {
            width: 4,
            height: 4,
            sample: SampleFormat::U8,
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
}
