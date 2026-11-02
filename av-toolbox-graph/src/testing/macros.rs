/// Writes a source fixture, which has nothing to negotiate against since it takes no inputs.
macro_rules! fixture_source {
    (
        $definition:ident,
        $instance:ident,
        $identifier:literal,
        $args:ty,
        $signature:expr
    ) => {
        struct $instance;

        impl StageInstance for $instance {}

        impl Source for $instance {
            fn next(&mut self, _context: &mut FrameContext) -> Result<Option<Frame>, StageError> {
                Ok(None)
            }
        }

        pub(super) struct $definition;

        impl StageDefinition for $definition {
            type Args = $args;

            fn package() -> &'static str {
                "test"
            }

            fn identifier() -> &'static str {
                $identifier
            }

            fn description() -> &'static str {
                "A stage used by tests."
            }

            fn inputs() -> &'static InputSignature {
                const SIGNATURE: &InputSignature = $signature;
                SIGNATURE
            }
        }

        impl SourceDefinition for $definition {
            type Side = DeviceSide;
            type Opened = ();

            fn open(_args: &Self::Args) -> Result<Self::Opened, StageError> {
                Ok(())
            }

            fn produces(
                _capabilities: &Capabilities,
                _opened: &Self::Opened,
            ) -> Result<OutputFormats, StageError> {
                let format = fixture_format();

                Ok(OutputFormats {
                    template: StreamTemplate::from_format(&format),
                    samples: FormatSet::of([format.sample()]),
                })
            }

            fn build(
                _context: &BuildContext,
                _opened: Self::Opened,
                _inputs: BoundInputs,
                _formats: &ResolvedFormats,
            ) -> Result<Box<dyn Source>, StageError> {
                Ok(Box::new($instance))
            }
        }
    };
}

/// Writes the kind-specific instance method for a filter or mask fixture.
///
/// Each does the minimum its trait requires: a filter passes its first bound input through, and a
/// mask allocates a fresh one in the fixture format.
macro_rules! fixture_kind_method {
    (Filter, $instance:ident) => {
        impl Filter for $instance {
            fn process(
                &mut self,
                _context: &mut FrameContext,
                mut inputs: Inputs,
            ) -> Result<Vec<Frame>, StageError> {
                let frame = inputs
                    .take_frame(0)
                    .ok_or_else(|| StageError::msg("expected a frame in slot 0"))?;

                Ok(vec![frame])
            }
        }
    };
    (Mask, $instance:ident) => {
        impl Mask for $instance {
            fn select(&mut self, context: &mut FrameContext, _inputs: Inputs) -> Result<GpuMask, StageError> {
                Ok(context.alloc_mask(&fixture_mask_format(), FrameMeta::at(0)))
            }
        }
    };
}

/// Writes a filter or mask fixture, something with both an input and an output side.
///
/// It accepts whatever it is offered and produces what it was given, which is the least
/// surprising behaviour for a test stage.
macro_rules! fixture_stage {
    (
        $definition:ident,
        $instance:ident,
        $kind_trait:ident,
        $definition_trait:ident,
        $identifier:literal,
        $args:ty,
        $signature:expr,
        $template:expr
    ) => {
        struct $instance;

        impl StageInstance for $instance {}
        fixture_kind_method!($kind_trait, $instance);

        pub(super) struct $definition;

        impl StageDefinition for $definition {
            type Args = $args;

            fn package() -> &'static str {
                "test"
            }

            fn identifier() -> &'static str {
                $identifier
            }

            fn description() -> &'static str {
                "A stage used by tests."
            }

            fn inputs() -> &'static InputSignature {
                const SIGNATURE: &InputSignature = $signature;
                SIGNATURE
            }
        }

        impl $definition_trait for $definition {
            type Side = DeviceSide;

            fn accepts_input(
                _capabilities: &Capabilities,
                _args: &Self::Args,
                _index: usize,
                offered: StreamFormat,
                _inputs: &[Offered],
            ) -> Result<FormatSet, StageError> {
                Ok(FormatSet::of([offered.sample()]))
            }

            fn produces(
                _capabilities: &Capabilities,
                _args: &Self::Args,
                chosen: &[Option<StreamFormat>],
            ) -> Result<OutputFormats, StageError> {
                let format = first_bound(chosen)?;

                Ok(OutputFormats {
                    template: $template(format),
                    samples: FormatSet::of([format.sample()]),
                })
            }

            fn build(
                _context: &BuildContext,
                _args: Self::Args,
                _inputs: BoundInputs,
                _formats: &ResolvedFormats,
            ) -> Result<Box<dyn $kind_trait>, StageError> {
                Ok(Box::new($instance))
            }
        }
    };
}

/// Writes a sink fixture, which has nothing to produce since it ends the stream.
macro_rules! fixture_sink {
    (
        $definition:ident,
        $instance:ident,
        $identifier:literal,
        $args:ty,
        $signature:expr
    ) => {
        struct $instance;

        impl StageInstance for $instance {}

        impl Sink for $instance {
            fn consume(&mut self, _context: &mut FrameContext, _inputs: Inputs) -> Result<(), StageError> {
                Ok(())
            }
        }

        pub(super) struct $definition;

        impl StageDefinition for $definition {
            type Args = $args;

            fn package() -> &'static str {
                "test"
            }

            fn identifier() -> &'static str {
                $identifier
            }

            fn description() -> &'static str {
                "A stage used by tests."
            }

            fn inputs() -> &'static InputSignature {
                const SIGNATURE: &InputSignature = $signature;
                SIGNATURE
            }
        }

        impl SinkDefinition for $definition {
            type Side = DeviceSide;

            fn accepts_input(
                _capabilities: &Capabilities,
                _args: &Self::Args,
                _index: usize,
                offered: StreamFormat,
                _inputs: &[Offered],
            ) -> Result<FormatSet, StageError> {
                Ok(FormatSet::of([offered.sample()]))
            }

            fn build(
                _context: &BuildContext,
                _args: Self::Args,
                _inputs: BoundInputs,
                _formats: &ResolvedFormats,
            ) -> Result<Box<dyn Sink>, StageError> {
                Ok(Box::new($instance))
            }
        }
    };
}

/// Writes a source fixture offering more than one sample, so an edge from it has a choice to make.
///
/// The frames are in the fixture format unless a frame format is given.
macro_rules! opinionated_source {
    ($definition:ident, $instance:ident, $identifier:literal, $samples:expr) => {
        opinionated_source!(
            $definition,
            $instance,
            $identifier,
            $samples,
            frame_format()
        );
    };
    ($definition:ident, $instance:ident, $identifier:literal, $samples:expr, $format:expr) => {
        struct $instance;

        impl StageInstance for $instance {}

        impl Source for $instance {
            fn next(&mut self, _context: &mut FrameContext) -> Result<Option<Frame>, StageError> {
                Ok(None)
            }
        }

        pub(super) struct $definition;

        impl StageDefinition for $definition {
            type Args = NoArgs;

            fn package() -> &'static str {
                "test"
            }

            fn identifier() -> &'static str {
                $identifier
            }

            fn description() -> &'static str {
                "A stage used by tests."
            }

            fn inputs() -> &'static InputSignature {
                const SIGNATURE: &InputSignature = &InputSignature::EMPTY;
                SIGNATURE
            }
        }

        impl SourceDefinition for $definition {
            type Side = DeviceSide;
            type Opened = ();

            fn open(_args: &Self::Args) -> Result<Self::Opened, StageError> {
                Ok(())
            }

            fn produces(
                _capabilities: &Capabilities,
                _opened: &Self::Opened,
            ) -> Result<OutputFormats, StageError> {
                Ok(OutputFormats {
                    template: StreamTemplate::Frames($format),
                    samples: $samples,
                })
            }

            fn build(
                _context: &BuildContext,
                _opened: Self::Opened,
                _inputs: BoundInputs,
                _formats: &ResolvedFormats,
            ) -> Result<Box<dyn Source>, StageError> {
                Ok(Box::new($instance))
            }
        }
    };
}

/// Writes a filter fixture that takes only certain samples and emits the same ones.
macro_rules! opinionated_filter {
    ($definition:ident, $instance:ident, $identifier:literal, $samples:expr) => {
        struct $instance;

        impl StageInstance for $instance {}

        impl Filter for $instance {
            fn process(
                &mut self,
                _context: &mut FrameContext,
                mut inputs: Inputs,
            ) -> Result<Vec<Frame>, StageError> {
                let frame = inputs
                    .take_frame(0)
                    .ok_or_else(|| StageError::msg("expected a frame in slot 0"))?;

                Ok(vec![frame])
            }
        }

        pub(super) struct $definition;

        impl StageDefinition for $definition {
            type Args = NoArgs;

            fn package() -> &'static str {
                "test"
            }

            fn identifier() -> &'static str {
                $identifier
            }

            fn description() -> &'static str {
                "A stage used by tests."
            }

            fn inputs() -> &'static InputSignature {
                const SIGNATURE: &InputSignature =
                    &InputSignature::new(&[InputParam::required("reference").accepting(StreamType::Frames)]);
                SIGNATURE
            }
        }

        impl FilterDefinition for $definition {
            type Side = DeviceSide;

            fn accepts_input(
                _capabilities: &Capabilities,
                _args: &Self::Args,
                _index: usize,
                _offered: StreamFormat,
                _inputs: &[Offered],
            ) -> Result<FormatSet, StageError> {
                Ok($samples)
            }

            fn produces(
                _capabilities: &Capabilities,
                _args: &Self::Args,
                chosen: &[Option<StreamFormat>],
            ) -> Result<OutputFormats, StageError> {
                let format = first_bound(chosen)?;

                Ok(OutputFormats {
                    template: StreamTemplate::from_format(&format),
                    samples: $samples,
                })
            }

            fn build(
                _context: &BuildContext,
                _args: Self::Args,
                _inputs: BoundInputs,
                _formats: &ResolvedFormats,
            ) -> Result<Box<dyn Filter>, StageError> {
                Ok(Box::new($instance))
            }
        }
    };
}
