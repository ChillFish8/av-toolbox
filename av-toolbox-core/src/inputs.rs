use av_toolbox_gpu::{DeviceSide, Side, StreamItem};

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
    authority: bool,
}

impl InputParam {
    pub const fn required(name: &'static str) -> Self {
        Self {
            name,
            required: true,
            accepts: None,
            authority: false,
        }
    }

    pub const fn optional(name: &'static str) -> Self {
        Self {
            name,
            required: false,
            accepts: None,
            authority: false,
        }
    }

    /// Narrows the param to one stream type.
    ///
    /// A param that has not been narrowed takes either type.
    pub const fn accepting(mut self, stream: StreamType) -> Self {
        self.accepts = Some(stream);
        self
    }

    /// Marks this param as the one whose chunk ids line the stage's inputs up.
    ///
    /// A stage with more than one input can only line its frames up against one of them.
    /// Only one param of a signature may be marked.
    pub const fn chunk_authority(mut self) -> Self {
        self.authority = true;
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

    pub const fn is_chunk_authority(&self) -> bool {
        self.authority
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

/// Panics unless at most one of `params` is marked as the chunk authority.
const fn check_one_authority(params: &[InputParam]) {
    let mut index = 0;
    let mut seen = false;

    while index < params.len() {
        if params[index].authority {
            if seen {
                panic!("a signature may mark only one param as the chunk authority");
            }

            seen = true;
        }

        index += 1;
    }
}

impl InputSignature {
    pub const EMPTY: Self = Self::new(&[]);

    pub const fn new(params: &'static [InputParam]) -> Self {
        check_one_authority(params);
        Self {
            params,
            variadic: false,
            variadic_accepts: None,
        }
    }

    pub const fn with_variadic(params: &'static [InputParam]) -> Self {
        check_one_authority(params);
        Self {
            params,
            variadic: true,
            variadic_accepts: None,
        }
    }

    /// A signature whose tail only takes one stream type.
    pub const fn with_variadic_of(params: &'static [InputParam], accepts: StreamType) -> Self {
        check_one_authority(params);
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

    /// The param marked as the chunk authority, if any was.
    ///
    /// Indexes [InputSignature::params], which is not the same space as
    /// [BoundInputs::chunk_authority] once a signature has a variadic tail.
    pub const fn chunk_authority(&self) -> Option<usize> {
        let mut index = 0;

        while index < self.params.len() {
            if self.params[index].authority {
                return Some(index);
            }

            index += 1;
        }

        None
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
    stream: StreamType,
}

impl VariadicInput {
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

#[derive(Debug, Clone)]
struct BoundParam {
    name: &'static str,
    handle: Option<StageHandle>,
    stream: Option<StreamType>,
}

/// A stage's inputs after they have been matched to its signature.
#[derive(Debug, Clone)]
pub struct BoundInputs {
    params: Vec<BoundParam>,
    variadic: Vec<VariadicInput>,
    chunk_authority: Option<usize>,
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

    /// Which input's chunk ids line this stage's inputs up.
    ///
    /// Indexed the way [Inputs] indexes its slots. `None` when nothing was bound.
    pub fn chunk_authority(&self) -> Option<usize> {
        self.chunk_authority
    }
}

/// What a stage was handed this step, one slot per parameter followed by one per variadic input,
/// the same order [ResolvedFormats](av_toolbox_gpu::ResolvedFormats) uses.
///
/// Every item is from the stage's own side, so a stage never holds anything it would have to move
/// between host and device itself. A slot is empty when an optional input was never bound, and
/// once taken it stays empty.
#[derive(Debug)]
pub struct Inputs<S: Side = DeviceSide> {
    slots: Vec<Option<S::Item>>,
}

impl Inputs<DeviceSide> {
    pub fn new(slots: Vec<Option<StreamItem>>) -> Self {
        Self { slots }
    }
}

impl<S: Side> Inputs<S> {
    pub fn from_slots(slots: Vec<Option<S::Item>>) -> Self {
        Self { slots }
    }

    pub fn take(&mut self, index: usize) -> Option<S::Item> {
        self.slots.get_mut(index)?.take()
    }

    /// The frame at `index`, leaving the slot empty.
    ///
    /// Returns `None` when the slot was empty and when it held a mask, and leaves a mask in place
    /// rather than consuming it.
    pub fn take_frame(&mut self, index: usize) -> Option<S::Frame> {
        let slot = self.slots.get_mut(index)?;

        match S::into_frame(slot.take()?) {
            Ok(frame) => Some(frame),
            Err(other) => {
                *slot = Some(other);
                None
            },
        }
    }

    /// The mask at `index`, leaving the slot empty.
    ///
    /// Returns `None` when the slot was empty and when it held a frame, and leaves a frame in place
    /// rather than consuming it.
    pub fn take_mask(&mut self, index: usize) -> Option<S::Mask> {
        let slot = self.slots.get_mut(index)?;

        match S::into_mask(slot.take()?) {
            Ok(mask) => Some(mask),
            Err(other) => {
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
            stream: None,
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

    let chunk_authority = resolve_chunk_authority(signature, &params, &variadic);

    Ok(BoundInputs {
        params,
        variadic,
        chunk_authority,
    })
}

/// Which slot's chunk ids line a stage's inputs up.
///
/// Slots are numbered as [Inputs] numbers them, one per param followed by one per variadic
/// input.
fn resolve_chunk_authority(
    signature: &InputSignature,
    params: &[BoundParam],
    variadic: &[VariadicInput],
) -> Option<usize> {
    let marked = signature.chunk_authority();
    if let Some(index) = marked
        && params[index].handle.is_some()
    {
        return Some(index);
    }

    let param_streams = params.iter().map(|param| param.stream);
    let variadic_streams = variadic.iter().map(|input| Some(input.stream));
    let slots: Vec<Option<StreamType>> = param_streams.chain(variadic_streams).collect();

    let frames = slots
        .iter()
        .position(|stream| *stream == Some(StreamType::Frames));
    if frames.is_some() {
        return frames;
    }

    slots.iter().position(|stream| stream.is_some())
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
        params[*next_positional].stream = Some(argument.stream());
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
        stream: argument.stream(),
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
        params[index].stream = Some(argument.stream());
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
        stream: argument.stream(),
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
