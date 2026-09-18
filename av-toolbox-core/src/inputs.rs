use crate::BindInputsError;

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
}

impl InputParam {
    pub const fn required(name: &'static str) -> Self {
        Self { name, required: true }
    }

    pub const fn optional(name: &'static str) -> Self {
        Self {
            name,
            required: false,
        }
    }

    pub const fn name(&self) -> &'static str {
        self.name
    }

    pub const fn is_required(&self) -> bool {
        self.required
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
}

impl InputSignature {
    pub const EMPTY: Self = Self::new(&[]);

    pub const fn new(params: &'static [InputParam]) -> Self {
        Self {
            params,
            variadic: false,
        }
    }

    pub const fn with_variadic(params: &'static [InputParam]) -> Self {
        Self {
            params,
            variadic: true,
        }
    }

    pub const fn params(&self) -> &'static [InputParam] {
        self.params
    }

    pub const fn is_variadic(&self) -> bool {
        self.variadic
    }

    pub const fn is_empty(&self) -> bool {
        self.params.is_empty() && !self.variadic
    }
}

/// An input supplied by a pipeline, either positionally or by name.
#[derive(Debug, Clone)]
pub struct InputArgument {
    name: Option<String>,
    handle: StageHandle,
}

impl InputArgument {
    pub fn positional(handle: StageHandle) -> Self {
        Self { name: None, handle }
    }

    pub fn named(name: impl Into<String>, handle: StageHandle) -> Self {
        Self {
            name: Some(name.into()),
            handle,
        }
    }

    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    pub fn handle(&self) -> StageHandle {
        self.handle
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
    if let Some(bound) = params.get_mut(*next_positional) {
        bound.handle = Some(argument.handle());
        *next_positional += 1;
        return Ok(());
    }

    if !signature.is_variadic() {
        return Err(BindInputsError::TooManyInputs {
            position,
            expected: params.len(),
        });
    }

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
    let matching = params.iter_mut().find(|param| param.name == name);
    if let Some(bound) = matching {
        if bound.handle.is_some() {
            return Err(BindInputsError::DuplicateInput {
                name: name.to_string(),
                position,
            });
        }

        bound.handle = Some(argument.handle());
        return Ok(());
    }

    if !signature.is_variadic() {
        return Err(BindInputsError::UnknownInput {
            name: name.to_string(),
            position,
        });
    }

    variadic.push(VariadicInput {
        name: Some(name.to_string()),
        handle: argument.handle(),
    });

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const REFERENCE: &InputSignature =
        &InputSignature::new(&[InputParam::required("reference"), InputParam::optional("mask")]);

    const VARIADIC: &InputSignature = &InputSignature::with_variadic(&[InputParam::required("reference")]);

    fn handle(id: u64) -> StageHandle {
        StageHandle::new(id)
    }

    #[test]
    fn positional_inputs_fill_params_in_order() {
        let arguments = [
            InputArgument::positional(handle(1)),
            InputArgument::positional(handle(2)),
        ];
        let bound = bind_inputs(REFERENCE, &arguments).expect("should bind");

        assert_eq!(bound.get("reference"), Some(handle(1)));
        assert_eq!(bound.get("mask"), Some(handle(2)));
        assert_eq!(bound.positional(0), Some(handle(1)));
    }

    #[test]
    fn named_inputs_bind_by_name_in_any_order() {
        let arguments = [
            InputArgument::named("mask", handle(2)),
            InputArgument::named("reference", handle(1)),
        ];
        let bound = bind_inputs(REFERENCE, &arguments).expect("should bind");

        assert_eq!(bound.get("reference"), Some(handle(1)));
        assert_eq!(bound.get("mask"), Some(handle(2)));
    }

    #[test]
    fn positional_and_named_inputs_mix() {
        let arguments = [
            InputArgument::positional(handle(1)),
            InputArgument::named("mask", handle(2)),
        ];
        let bound = bind_inputs(REFERENCE, &arguments).expect("should bind");

        assert_eq!(bound.get("reference"), Some(handle(1)));
        assert_eq!(bound.get("mask"), Some(handle(2)));
    }

    #[test]
    fn an_omitted_optional_input_is_unbound() {
        let arguments = [InputArgument::positional(handle(1))];
        let bound = bind_inputs(REFERENCE, &arguments).expect("should bind");

        assert_eq!(bound.get("mask"), None);
        assert_eq!(bound.iter().count(), 1);
    }

    #[test]
    fn a_positional_input_after_a_named_input_is_rejected() {
        let arguments = [
            InputArgument::named("reference", handle(1)),
            InputArgument::positional(handle(2)),
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
            InputArgument::positional(handle(1)),
            InputArgument::named("reference", handle(2)),
        ];
        let error = bind_inputs(REFERENCE, &arguments).expect_err("should not bind");

        assert!(matches!(
            error,
            BindInputsError::DuplicateInput { position: 1, .. },
        ));
    }

    #[test]
    fn a_missing_required_input_is_rejected() {
        let arguments = [InputArgument::named("mask", handle(2))];
        let error = bind_inputs(REFERENCE, &arguments).expect_err("should not bind");

        assert!(matches!(
            error,
            BindInputsError::MissingRequiredInput { name: "reference" },
        ));
    }

    #[test]
    fn extra_positional_inputs_are_rejected_without_a_variadic_tail() {
        let arguments = [
            InputArgument::positional(handle(1)),
            InputArgument::positional(handle(2)),
            InputArgument::positional(handle(3)),
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
            InputArgument::positional(handle(1)),
            InputArgument::named("nonsense", handle(2)),
        ];
        let error = bind_inputs(REFERENCE, &arguments).expect_err("should not bind");

        assert!(matches!(error, BindInputsError::UnknownInput { position: 1, .. },));
    }

    #[test]
    fn a_variadic_tail_collects_extra_inputs_in_order() {
        let arguments = [
            InputArgument::positional(handle(1)),
            InputArgument::positional(handle(2)),
            InputArgument::named("overlay", handle(3)),
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
}
