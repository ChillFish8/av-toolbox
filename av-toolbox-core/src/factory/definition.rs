use std::marker::PhantomData;

use av_toolbox_gpu::{BuildContext, ResolvedFormats};
use jsonschema::Validator;
use serde::Deserialize;
use serde_json::Value;

use super::{StageInfo, summary_of};
use crate::demand::intern;
use crate::{
    ArgsError,
    BindInputsError,
    BoundInputs,
    BuildStageError,
    Granularity,
    InputArgument,
    InputDemand,
    InputSignature,
    OpenSlot,
    Parallelism,
    RegistryError,
    SchemaViolation,
    StageArtefacts,
    StageDefinition,
    StageError,
    StageKind,
    StreamType,
    bind_inputs,
};

/// The adapter that erases a concrete [StageDefinition](crate::StageDefinition).
pub(crate) struct Definition<D: StageDefinition> {
    pub(super) kind: StageKind,
    pub(super) full_identifier: String,
    pub(super) schema: Value,
    pub(super) validator: Validator,
    pub(super) marker: PhantomData<fn() -> D>,
}

impl<D: StageDefinition> Definition<D> {
    /// Generates and compiles the stage's schema, failing if anything about it is malformed.
    pub(crate) fn new(kind: StageKind) -> Result<Self, RegistryError> {
        Self::check_segment(D::package(), "package")?;
        Self::check_segment(D::identifier(), "identifier")?;

        let full_identifier = format!("{}/{}", D::package(), D::identifier());

        Self::check_description(&full_identifier)?;
        Self::check_planes_of(&full_identifier)?;
        Self::check_artefacts(&full_identifier)?;

        let generator = schemars::SchemaGenerator::default();
        let generated = generator.into_root_schema_for::<D::Args>();
        let schema = serde_json::to_value(&generated).map_err(|error| RegistryError::InvalidSchema {
            identifier: full_identifier.clone(),
            reason: error.to_string(),
        })?;

        let validator = jsonschema::validator_for(&schema).map_err(|error| RegistryError::InvalidSchema {
            identifier: full_identifier.clone(),
            reason: error.to_string(),
        })?;

        Ok(Self {
            kind,
            full_identifier,
            schema,
            validator,
            marker: PhantomData,
        })
    }

    pub(crate) fn full_identifier(&self) -> &str {
        &self.full_identifier
    }

    pub(crate) fn kind(&self) -> StageKind {
        self.kind
    }

    pub(super) fn stage_info(&self) -> StageInfo<'_> {
        StageInfo {
            kind: self.kind,
            package: D::package(),
            identifier: D::identifier(),
            full_identifier: &self.full_identifier,
            tags: D::tags(),
            description: D::description(),
            schema: &self.schema,
            inputs: D::inputs(),
            artefacts: StageArtefacts::of::<D>(),
            wants_output_depth: D::wants_output_depth(),
        }
    }

    pub(super) fn deserialize_args(&self, args: &Value) -> Result<D::Args, BuildStageError> {
        let violations: Vec<SchemaViolation> = self
            .validator
            .iter_errors(args)
            .map(|error| SchemaViolation::new(error.instance_path().to_string(), error.to_string()))
            .collect();
        if !violations.is_empty() {
            return Err(self.invalid_args(ArgsError::SchemaViolations(violations)));
        }

        let deserialized = D::Args::deserialize(args);
        deserialized.map_err(|error| self.invalid_args(ArgsError::Deserialize(error)))
    }

    fn invalid_args(&self, source: ArgsError) -> BuildStageError {
        BuildStageError::InvalidArgs {
            identifier: self.full_identifier.clone(),
            source,
        }
    }

    fn invalid_inputs(&self, source: BindInputsError) -> BuildStageError {
        BuildStageError::InvalidInputs {
            identifier: self.full_identifier.clone(),
            source,
        }
    }

    /// Rejects an artefact name that is empty, holds a `.` or `/`, or is declared twice.
    fn check_artefacts(full_identifier: &str) -> Result<(), RegistryError> {
        let declared = D::artefacts();

        for (position, decl) in declared.iter().enumerate() {
            let malformed = decl.name.is_empty() || decl.name.contains(['.', '/']);
            if malformed {
                return Err(RegistryError::InvalidArtefactName {
                    identifier: full_identifier.to_string(),
                    name: decl.name,
                });
            }

            let earlier = &declared[..position];
            let repeated = earlier.iter().any(|other| other.name == decl.name);
            if repeated {
                return Err(RegistryError::DuplicateArtefact {
                    identifier: full_identifier.to_string(),
                    name: decl.name,
                });
            }
        }

        Ok(())
    }

    /// Rejects a `planes_of` that names no input of the signature, or that is set on an input
    /// that does not take masks.
    fn check_planes_of(full_identifier: &str) -> Result<(), RegistryError> {
        let params = D::inputs().params();

        for param in params {
            let Some(named) = param.planes_of_input() else {
                continue;
            };

            let names_an_input = params.iter().any(|other| other.name() == named);
            if !names_an_input {
                return Err(RegistryError::InvalidPlanesOf {
                    identifier: full_identifier.to_string(),
                    param: param.name(),
                    reason: "it names no input of this stage",
                });
            }

            if param.accepts() != Some(StreamType::Mask) {
                return Err(RegistryError::InvalidPlanesOf {
                    identifier: full_identifier.to_string(),
                    param: param.name(),
                    reason: "it is only allowed on an input that takes masks",
                });
            }
        }

        Ok(())
    }

    /// Rejects a demand naming an input the stage's signature does not have.
    fn check_demand_inputs(&self, args: &D::Args) -> Result<(), BuildStageError> {
        let params = D::inputs().params();

        for demanded in D::demands(args) {
            let names_an_input = params.iter().any(|param| param.name() == demanded.input);
            if names_an_input {
                continue;
            }

            let error = RegistryError::InvalidDemand {
                identifier: self.full_identifier.clone(),
                input: demanded.input,
            };
            return Err(self.invalid_args(ArgsError::InvalidDemand(error)));
        }

        Ok(())
    }

    /// Rejects a probe naming an input the stage does not have, or a second probe of one kind on one input.
    fn check_probe_inputs(&self, args: &D::Args) -> Result<(), BuildStageError> {
        let params = D::inputs().params();
        let probes = D::probes(args);

        for (position, probed) in probes.iter().enumerate() {
            let names_an_input = params.iter().any(|param| param.name() == probed.input);
            if !names_an_input {
                let error = RegistryError::InvalidProbe {
                    identifier: self.full_identifier.clone(),
                    input: probed.input,
                };
                return Err(self.invalid_args(ArgsError::InvalidProbe(error)));
            }

            let earlier = &probes[..position];
            let repeated = earlier
                .iter()
                .any(|other| other.input == probed.input && other.probe.kind() == probed.probe.kind());
            if repeated {
                let error = RegistryError::DuplicateProbeInput {
                    identifier: self.full_identifier.clone(),
                    input: probed.input,
                    kind: probed.probe.kind().name(),
                };
                return Err(self.invalid_args(ArgsError::InvalidProbe(error)));
            }
        }

        Ok(())
    }

    /// Deserializes the arguments and binds the inputs, refusing demands the stage could not be given.
    pub(super) fn prepared(
        &self,
        args: &Value,
        inputs: &[InputArgument],
        slot: &OpenSlot,
    ) -> Result<Prepared<D>, BuildStageError> {
        let args = self.deserialize_args(args)?;
        self.check_demand_inputs(&args)?;
        self.check_probe_inputs(&args)?;

        let bound = bind_inputs(D::inputs(), inputs).map_err(|source| self.invalid_inputs(source))?;
        self.check_chunk_input(&args, &bound)?;

        Ok(Prepared {
            identifier: self.full_identifier.clone(),
            args,
            inputs: bound,
            slot: slot.clone(),
        })
    }

    /// Rejects a chunk stage whose chunk authority has no name to carry its chunk demand.
    fn check_chunk_input(&self, args: &D::Args, inputs: &BoundInputs) -> Result<(), BuildStageError> {
        let chunked = D::parallelism(args).granularity() == Granularity::Chunk;
        if !chunked || inputs.chunk_authority().is_none() {
            return Ok(());
        }

        if chunk_authority_name(D::inputs(), inputs).is_some() {
            return Ok(());
        }

        let identifier = self.full_identifier.clone();
        Err(self.invalid_args(ArgsError::UnnamedChunkInput { identifier }))
    }

    /// Rejects a segment that would produce a malformed `package/stage` identifier.
    fn check_segment(segment: &str, label: &'static str) -> Result<(), RegistryError> {
        if !segment.is_empty() && !segment.contains('/') {
            return Ok(());
        }

        Err(RegistryError::InvalidIdentifier {
            package: D::package().to_string(),
            identifier: D::identifier().to_string(),
            segment: label,
        })
    }

    fn check_description(full_identifier: &str) -> Result<(), RegistryError> {
        let description = D::description();
        let summary = summary_of(description);
        if !summary.is_empty() {
            return Ok(());
        }

        Err(RegistryError::MissingDescription {
            identifier: full_identifier.to_string(),
        })
    }
}

/// A stage with its arguments deserialized and its inputs bound, ready to negotiate and build.
pub(crate) struct Prepared<D: StageDefinition> {
    pub(super) identifier: String,
    pub(super) args: D::Args,
    pub(super) inputs: BoundInputs,
    pub(super) slot: OpenSlot,
}

impl<D: StageDefinition> Prepared<D> {
    /// The definition's demands, plus a chunk stage's own demand on its chunk authority.
    pub(super) fn stage_demands(&self) -> Vec<InputDemand> {
        let mut demands = D::demands(&self.args);

        let Parallelism::Chunk(chunk) = D::parallelism(&self.args) else {
            return demands;
        };

        let Some(input) = chunk_authority_name(D::inputs(), &self.inputs) else {
            return demands;
        };

        demands.push(InputDemand {
            input,
            demand: chunk.demand().clone(),
        });
        demands
    }
}

/// The name of the input a stage lines its chunks up by.
///
/// A variadic input is named by the name it was bound under, so an unnamed one has none.
fn chunk_authority_name(signature: &InputSignature, inputs: &BoundInputs) -> Option<&'static str> {
    let index = inputs.chunk_authority()?;
    let params = signature.params();

    if let Some(param) = params.get(index) {
        return Some(param.name());
    }

    let variadic = inputs.variadic().get(index - params.len())?;
    let name = variadic.name()?;

    Some(intern(name))
}

/// Hands back a built stage as it is, for the kinds with nothing to add after building.
pub(super) fn keep<T>(
    _context: &BuildContext,
    _formats: &ResolvedFormats,
    built: T,
) -> Result<T, StageError> {
    Ok(built)
}
