use std::marker::PhantomData;

use cubecl::prelude::barrier::BarrierExpand;
// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than by name.
use cubecl::prelude::*;
use cubecl::std::tensor::launch::ViewArg;
use cubecl::std::tensor::layout::chain::{Chain, ChainLaunch};
use cubecl::std::tensor::{
    View,
    ViewExpand,
    ViewOperations,
    ViewOperationsExpand,
    ViewOperationsMut,
    ViewOperationsMutExpand,
};
use cubek_convolution::components::global::args::RuntimeArgs;
use cubek_convolution::components::global::layout::{
    NhwcLayout,
    NhwcLayoutLaunch,
    OutLayout,
    OutLayoutLaunch,
};
use cubek_convolution::forward::args::{ConcreteArgs, ConcreteInputsFactory, ConcreteOutputFactory};
use cubek_convolution::routines::Routine as ConvolutionRoutine;
use cubek_convolution::routines::simple::SimpleSyncCyclicConv;
use cubek_matmul::components::global::memory::GlobalLayoutConfig;
use cubek_matmul::definition::{AvailableVectorSizes, Blueprint, MatmulElems, MatmulSetupError};
use cubek_matmul::launch::{
    Acc,
    AccSize,
    BatchedCoords,
    Lhs,
    LhsSize,
    MatmulArgs,
    Rhs,
    RhsSize,
    TensorArgs,
    TensorInputs,
    TensorOutput,
};
use cubek_matmul::routines::simple::SimpleArgs;
use cubek_matmul::routines::{BlueprintStrategy, Routine};
use cubek_std::MatrixLayout;

use crate::cnn::convolution::{BoundConvolution, convolution_failed};
use crate::cnn::{Activation, CnnError};

/// The matmul routine of `cubek-convolution`'s `SimpleSyncCyclic` convolution, the only routine the fused
/// write-out launches with.
///
/// Its plane writer writes each value through `write_checked`, which [FusedView] fuses. A routine reaching
/// the fused view's linear slices or tensor maps would panic while its kernel expands.
type FusedRoutine = <SimpleSyncCyclicConv as ConvolutionRoutine>::MatmulRoutine;

/// The output's own NHWC layout, through which the residual is read at each written position.
type OutChain = Chain<NhwcLayout, OutLayout>;

/// cubek's own inputs, as its plain convolution binds them, at the launch's element types.
type LaunchedInputs = TensorInputs<Vector<Lhs, LhsSize>, Vector<Rhs, RhsSize>, Vector<Acc, AccSize>>;

/// cubek's own output, at the launch's element type.
type LaunchedOutput = TensorOutput<Vector<Acc, AccSize>>;

/// The elementwise step a fused write-out applies after the bias.
trait Epilogue: Send + Sync + 'static + Clone {
    const SILU: bool;
}

#[derive(Clone)]
struct WithSilu;

impl Epilogue for WithSilu {
    const SILU: bool = true;
}

#[derive(Clone)]
struct WithoutSilu;

impl Epilogue for WithoutSilu {
    const SILU: bool = false;
}

/// Applies SiLU to `value` when `silu` is set, then adds the residual at `position` if there is one, in f32.
#[cube]
fn fuse<T: CubePrimitive>(
    value: T,
    residual: ComptimeOption<View<T, BatchedCoords>>,
    position: BatchedCoords,
    #[comptime] silu: bool,
) -> T {
    let mut sum = Vector::<f32, AccSize>::cast_from(value);

    if silu {
        let one = Vector::<f32, AccSize>::new(1.0);
        sum = sum / (one + Vector::exp(-sum));
    }

    #[comptime]
    match residual {
        ComptimeOption::Some(view) => {
            let added = view.read_checked(position);
            sum += Vector::<f32, AccSize>::cast_from(added);
        },
        ComptimeOption::None => {},
    }

    T::cast_from(sum)
}

/// An output view whose writes pass through [fuse].
///
/// Reads, the shape and the bounds are the output's. It has no linear slices or tensor maps, and panics while
/// expanding a kernel that asks for them.
#[expect(dead_code, reason = "a kernel only ever holds its expanded form")]
struct FusedView<T: CubePrimitive> {
    _value: PhantomData<T>,
}

#[derive(Clone)]
struct FusedViewExpand<T: CubePrimitive> {
    out: ViewExpand<T, BatchedCoords, ReadWrite>,
    residual: ComptimeOptionExpand<View<T, BatchedCoords>>,
    silu: bool,
}

impl<T: CubePrimitive> CubeType for FusedView<T> {
    type ExpandType = FusedViewExpand<T>;
}

impl<T: CubePrimitive> IntoMut for FusedViewExpand<T> {
    fn into_mut(self, _scope: &mut Scope) -> Self {
        self
    }
}

impl<T: CubePrimitive> CubeDebug for FusedViewExpand<T> {}

impl<T: CubePrimitive> Vectorized for FusedView<T> {}

impl<T: CubePrimitive> VectorizedExpand for FusedViewExpand<T> {
    fn vector_size(&self) -> VectorSize {
        self.out.vector_size()
    }
}

impl<T: CubePrimitive> ViewOperations<T, BatchedCoords> for FusedView<T> {}

impl<T: CubePrimitive> ViewOperationsExpand<T, BatchedCoords> for FusedViewExpand<T> {
    fn __expand_read_method(
        &self,
        scope: &mut Scope,
        position: <BatchedCoords as CubeType>::ExpandType,
    ) -> <T as CubeType>::ExpandType {
        self.out.clone().__expand_read_method(scope, position)
    }

    fn __expand_read_checked_method(
        &self,
        scope: &mut Scope,
        position: <BatchedCoords as CubeType>::ExpandType,
    ) -> <T as CubeType>::ExpandType {
        self.out.clone().__expand_read_checked_method(scope, position)
    }

    fn __expand_read_masked_method(
        &self,
        scope: &mut Scope,
        position: <BatchedCoords as CubeType>::ExpandType,
        value: <T as CubeType>::ExpandType,
    ) -> <T as CubeType>::ExpandType {
        self.out
            .clone()
            .__expand_read_masked_method(scope, position, value)
    }

    fn __expand_read_unchecked_method(
        &self,
        scope: &mut Scope,
        position: <BatchedCoords as CubeType>::ExpandType,
    ) -> <T as CubeType>::ExpandType {
        self.out.clone().__expand_read_unchecked_method(scope, position)
    }

    fn __expand_to_linear_slice_method(
        &self,
        _scope: &mut Scope,
        _position: <BatchedCoords as CubeType>::ExpandType,
        _end: <BatchedCoords as CubeType>::ExpandType,
    ) -> SliceExpand<T, ReadOnly> {
        panic!("the fused output view has no linear slices")
    }

    fn __expand_shape_method(&self, scope: &mut Scope) -> <BatchedCoords as CubeType>::ExpandType {
        self.out.__expand_shape_method(scope)
    }

    fn __expand_is_in_bounds_method(
        &self,
        scope: &mut Scope,
        position: <BatchedCoords as CubeType>::ExpandType,
    ) -> NativeExpand<bool> {
        self.out.__expand_is_in_bounds_method(scope, position)
    }

    fn __expand_tensor_map_load_method(
        &self,
        _scope: &mut Scope,
        _barrier: BarrierExpand,
        _shared_memory: SliceExpand<T, ReadWrite>,
        _position: <BatchedCoords as CubeType>::ExpandType,
    ) {
        panic!("the fused output view has no tensor map")
    }
}

impl<T: CubePrimitive> ViewOperationsMut<T, BatchedCoords> for FusedView<T> {}

impl<T: CubePrimitive> ViewOperationsMutExpand<T, BatchedCoords> for FusedViewExpand<T> {
    fn __expand_write_method(
        &self,
        scope: &mut Scope,
        position: <BatchedCoords as CubeType>::ExpandType,
        value: <T as CubeType>::ExpandType,
    ) {
        let residual = self.residual.clone();
        let fused = fuse::expand::<T>(scope, value, residual, position.clone(), self.silu);

        self.out.clone().__expand_write_method(scope, position, fused);
    }

    fn __expand_write_checked_method(
        &self,
        scope: &mut Scope,
        position: <BatchedCoords as CubeType>::ExpandType,
        value: <T as CubeType>::ExpandType,
    ) {
        let residual = self.residual.clone();
        let fused = fuse::expand::<T>(scope, value, residual, position.clone(), self.silu);

        self.out
            .clone()
            .__expand_write_checked_method(scope, position, fused);
    }

    fn __expand_to_linear_slice_mut_method(
        &self,
        _scope: &mut Scope,
        _position: <BatchedCoords as CubeType>::ExpandType,
        _end: <BatchedCoords as CubeType>::ExpandType,
    ) -> SliceExpand<T, ReadWrite> {
        panic!("the fused output view has no linear slices")
    }

    fn __expand_tensor_map_store_method(
        &self,
        _scope: &mut Scope,
        _shared_memory: SliceExpand<T, ReadOnly>,
        _position: <BatchedCoords as CubeType>::ExpandType,
    ) {
        panic!("the fused output view has no tensor map")
    }
}

/// Wraps `_out` in a [FusedView] applying `E` and adding `_residual`, if there is one.
///
/// It only runs inside a kernel, where it expands through [fused_view::expand].
fn fused_view<T: CubePrimitive, E: Epilogue>(
    _out: View<T, BatchedCoords, ReadWrite>,
    _residual: ComptimeOption<View<T, BatchedCoords>>,
) -> View<T, BatchedCoords, ReadWrite> {
    cubecl::unexpanded!()
}

mod fused_view {
    use super::*;

    pub(super) fn expand<T: CubePrimitive, E: Epilogue>(
        _scope: &mut Scope,
        out: ViewExpand<T, BatchedCoords, ReadWrite>,
        residual: ComptimeOptionExpand<View<T, BatchedCoords>>,
    ) -> ViewExpand<T, BatchedCoords, ReadWrite> {
        let fused = FusedViewExpand {
            out,
            residual,
            silu: E::SILU,
        };

        ViewExpand::new_mut(fused)
    }
}

/// cubek's own inputs, with the residual the write-out adds.
#[derive(CubeLaunch, CubeType, Clone, Copy)]
struct FusedInputs<Lhs: CubePrimitive, Rhs: CubePrimitive, Acc: CubePrimitive> {
    tensors: TensorInputs<Lhs, Rhs, Acc>,
    residual: ComptimeOption<View<Acc, BatchedCoords>>,
}

/// Matmul arguments that read as cubek's own tensor arguments, and write through a [FusedView] applying `E`.
#[derive(Clone)]
struct FusedArgs<E: Epilogue> {
    _epilogue: PhantomData<E>,
}

#[cube]
impl<E: Epilogue> MatmulArgs for FusedArgs<E> {
    type Output<EO: CubePrimitive> = TensorOutput<EO>;
    type Input<Lhs: CubePrimitive, Rhs: CubePrimitive, EO: CubePrimitive> = FusedInputs<Lhs, Rhs, EO>;
    type Config = RuntimeArgs;
    type State<Lhs: CubePrimitive, Rhs: CubePrimitive, EO: CubePrimitive> = (
        (TensorInputs<Lhs, Rhs, EO>, TensorOutput<EO>, RuntimeArgs),
        ComptimeOption<View<EO, BatchedCoords>>,
    );

    fn init_state<Lhs: CubePrimitive, Rhs: CubePrimitive, EO: CubePrimitive>(
        input: &Self::Input<Lhs, Rhs, EO>,
        output: &mut Self::Output<EO>,
        config: Self::Config,
        #[comptime] lhs_layout_config: GlobalLayoutConfig,
        #[comptime] rhs_layout_config: GlobalLayoutConfig,
        #[comptime] out_layout_config: GlobalLayoutConfig,
    ) -> Self::State<Lhs, Rhs, EO> {
        let tensors = TensorArgs::<RuntimeArgs>::init_state::<Lhs, Rhs, EO>(
            &input.tensors,
            output,
            config,
            lhs_layout_config,
            rhs_layout_config,
            out_layout_config,
        );

        (tensors, input.residual)
    }

    fn view_lhs<Lhs: CubePrimitive, Rhs: CubePrimitive, EO: CubePrimitive>(
        state: &Self::State<Lhs, Rhs, EO>,
    ) -> View<Lhs, BatchedCoords> {
        TensorArgs::<RuntimeArgs>::view_lhs::<Lhs, Rhs, EO>(&state.0)
    }

    fn batch_lhs<Lhs: CubePrimitive, Rhs: CubePrimitive, EO: CubePrimitive>(
        state: &Self::State<Lhs, Rhs, EO>,
        batch: usize,
    ) -> usize {
        TensorArgs::<RuntimeArgs>::batch_lhs::<Lhs, Rhs, EO>(&state.0, batch)
    }

    fn view_rhs<Lhs: CubePrimitive, Rhs: CubePrimitive, EO: CubePrimitive>(
        state: &Self::State<Lhs, Rhs, EO>,
    ) -> View<Rhs, BatchedCoords> {
        TensorArgs::<RuntimeArgs>::view_rhs::<Lhs, Rhs, EO>(&state.0)
    }

    fn batch_rhs<Lhs: CubePrimitive, Rhs: CubePrimitive, EO: CubePrimitive>(
        state: &Self::State<Lhs, Rhs, EO>,
        batch: usize,
    ) -> usize {
        TensorArgs::<RuntimeArgs>::batch_rhs::<Lhs, Rhs, EO>(&state.0, batch)
    }

    fn view_acc<Lhs: CubePrimitive, Rhs: CubePrimitive, EO: CubePrimitive>(
        state: &Self::State<Lhs, Rhs, EO>,
    ) -> ComptimeOption<View<EO, BatchedCoords>> {
        TensorArgs::<RuntimeArgs>::view_acc::<Lhs, Rhs, EO>(&state.0)
    }

    fn batch_acc<Lhs: CubePrimitive, Rhs: CubePrimitive, EO: CubePrimitive>(
        state: &Self::State<Lhs, Rhs, EO>,
        batch: usize,
    ) -> usize {
        TensorArgs::<RuntimeArgs>::batch_acc::<Lhs, Rhs, EO>(&state.0, batch)
    }

    fn view_out<Lhs: CubePrimitive, Rhs: CubePrimitive, EO: CubePrimitive>(
        state: &mut Self::State<Lhs, Rhs, EO>,
    ) -> View<EO, BatchedCoords, ReadWrite> {
        let out = TensorArgs::<RuntimeArgs>::view_out::<Lhs, Rhs, EO>(&mut state.0);

        fused_view::<EO, E>(out, state.1)
    }

    fn batch_out<Lhs: CubePrimitive, Rhs: CubePrimitive, EO: CubePrimitive>(
        state: &Self::State<Lhs, Rhs, EO>,
        batch: usize,
    ) -> usize {
        TensorArgs::<RuntimeArgs>::batch_out::<Lhs, Rhs, EO>(&state.0, batch)
    }

    fn runtime_config<Lhs: CubePrimitive, Rhs: CubePrimitive, EO: CubePrimitive>(
        state: &Self::State<Lhs, Rhs, EO>,
    ) -> Self::Config {
        TensorArgs::<RuntimeArgs>::runtime_config::<Lhs, Rhs, EO>(&state.0)
    }
}

/// Launches the convolution `bound` describes, applying `activation` and adding `residual`, if given, as each
/// value is written.
///
/// `residual` must be laid out as the output is.
pub(crate) fn launch_fused<R: Runtime>(
    client: &ComputeClient<R>,
    bound: BoundConvolution<R>,
    residual: Option<TensorBinding<R>>,
    activation: Activation,
) -> Result<(), CnnError> {
    match activation {
        Activation::Silu => launch_with::<R, WithSilu>(client, bound, residual),
        Activation::None => launch_with::<R, WithoutSilu>(client, bound, residual),
    }
}

/// The same as [launch_fused], with the epilogue as a type.
///
/// The preparation is `cubek-convolution`'s own forward launch, which binds no residual.
fn launch_with<R: Runtime, E: Epilogue>(
    client: &ComputeClient<R>,
    bound: BoundConvolution<R>,
    residual: Option<TensorBinding<R>>,
) -> Result<(), CnnError> {
    let BoundConvolution {
        problem,
        blueprint,
        input,
        weight,
        bias,
        output,
    } = bound;
    let dtypes = MatmulElems::from_globals(&problem.global_dtypes);

    let available = AvailableVectorSizes::from_type_sizes(
        client,
        input.data_elem_size(),
        weight.data_elem_size(),
        dtypes.acc_global.size(),
    );
    let input_data = input.data();
    let weight_data = weight.data();
    let filtered = available
        .filter_lhs_with_tensor(&input_data.strides, &input_data.shape, MatrixLayout::RowMajor)
        .filter_rhs_with_tensor(&weight_data.strides, &weight_data.shape, MatrixLayout::RowMajor)
        .filter_out_with_tensor(&output.strides, &output.shape);
    let routine_sizes = SimpleSyncCyclicConv::filter_vector_sizes(filtered);
    let vector_sizes = routine_sizes.pick_max().map_err(matmul_failed)?;

    let strategy: BlueprintStrategy<RuntimeArgs, FusedRoutine> = match blueprint {
        Some(forced) => BlueprintStrategy::Forced(forced.matmul().clone()),
        None => BlueprintStrategy::Inferred(SimpleArgs::default()),
    };
    let matmul_problem = problem.as_matmul_problem();
    let device_settings = <FusedRoutine as Routine<RuntimeArgs>>::device_settings(client, vector_sizes);
    let expand_info = <FusedRoutine as Routine<RuntimeArgs>>::expand_blueprint(
        &matmul_problem,
        &device_settings,
        &strategy,
    );
    let expand_info = expand_info.map_err(matmul_failed)?;
    let blueprint = &expand_info.blueprint;
    let problem = <TensorArgs<RuntimeArgs> as ConcreteArgs<FusedRoutine>>::adjust_problem(
        client, problem, blueprint, &dtypes,
    );
    let adjusted_problem = problem.as_matmul_problem();
    let launch_info =
        <FusedRoutine as Routine<RuntimeArgs>>::prepare(&adjusted_problem, &device_settings, expand_info);
    let launch_info = launch_info.map_err(matmul_failed)?;
    let blueprint = &launch_info.blueprint;

    let (tensors, runtime_args) = <LaunchedInputs as ConcreteInputsFactory<FusedRoutine>>::create(
        input, weight, bias, blueprint, &problem, &dtypes,
    );
    let output_arg =
        <LaunchedOutput as ConcreteOutputFactory<FusedRoutine>>::create(output, blueprint, &problem, &dtypes);
    let out_config = blueprint.out_global_layout_config();
    let residual_view = residual.map(|binding| {
        let out_layout = OutLayoutLaunch::from_args(&problem, out_config);
        let layout = ChainLaunch::new(NhwcLayoutLaunch::unchecked(), out_layout);

        ViewArg::new_tensor::<OutChain>(binding.into_tensor_arg(), layout)
    });
    let inputs = FusedInputsLaunch::new(tensors, residual_view.into());

    let launched = cubek_matmul::launch::launch_kernel::<FusedArgs<E>, R, FusedRoutine>(
        client,
        inputs,
        output_arg,
        runtime_args,
        launch_info,
    );

    launched.map_err(matmul_failed)
}

fn matmul_failed(error: MatmulSetupError) -> CnnError {
    let setup_error = error.into();

    convolution_failed(setup_error)
}
