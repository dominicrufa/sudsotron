"""minimizer of Klein-Kramers Field Equation for cDFT"""
import jax
from jax import numpy as jnp
import numpy as np
from dataclasses import dataclass, asdict, field
import typing
import functools

from sudsotron.nn.modules import (
    GaussianBasisPINN, 
    GaussianBasisPINNParams, 
    DEFAULT_NN_KEY,
    NNFn,
    NNParams,
)

from sudsotron.utils import cosine_cutoff, minimize

from sudsotron.cDFT.handlers import (
    DEFAULT_KT,
    DEFAULT_N0,
    DEFAULT_R_CUT,
    DEFAULT_NUM_GRIDPOINTS,
    HNCRadialDCFHandler,
    SScDFTHandler,
)

from sudsotron.potentials.handlers import (
    DynamicPotentialHandler,
    DynamicNeuralPotentialHandler,
    DynamicHybridPotentialHandler
)

from sudsotron.cDFT.utils import (
    spatial_grids,
    dFexcsdn_HNC_Riemann_approx_aperiodic,
)

DEFAULT_M = 18 # amus for H2O
DEFAULT_FRICTION_COEFF = 1. # 1 / ps
DEFAULT_T = 50. # ps
DEFAULT_VELOCITY_LIMIT_SCALE = 16 # unitless
DEFAULT_NUM_SPATIAL_GRIDPOINTS = DEFAULT_NUM_GRIDPOINTS
DEFAULT_NUM_TEMPORAL_GRIDPOINTS = 100
DEFAULT_NUM_VELOCITY_GRIDPOINTS = 100
DEFAULT_NUM_MEAN_VELOCITY_PROJECTIONS = 32



# @dataclass(frozen=True)
# class SSKKFE: # spherically symmetric KKFE
#     hnc_dcf: HNCRadialDCF
#     Uext_handler: typing.Union[
#         DynamicPotentialHandler,
#         DynamicNeuralPotentialHandler,
#     ]
#     spatial_grid_bounds: jnp.array([-DEFAULT_R_CUT, DEFAULT_R_CUT])
#     num_gridpoints: int = DEFAULT_NUM_GRIDPOINTS
#     n0: float = DEFAULT_N0
#     kT: float = DEFAULT_KT
#     model_params: GaussianBasisPINN = GaussianBasisPINNParams()
#     key: jax.Array = DEFAULT_NN_KEY

#     # dynamic params
#     m: float = DEFAULT_M
#     friction_coeff: float = DEFAULT_FRICTION_COEFF
#     num_v_gridpoints: int = DEFAULT_NUM_GRIDPOINTS
#     num_v_mean_projections: int = 32
#     ts: jax.Array = jnp.linspace(0., DEFAULT_T)

#     # define scaling limits of the velocity grid (takes positive scalars)
#     v_lim_scale: float = DEFAULT_V_LIMIT_SCALE

#     # reference arrays
#     R: jax.Array = field(init=False)
#     dcf_kernel: jax.Array = field(init=False)
#     dcfdr_kernel: jax.Array = field(init=False)
#     rs: jax.Array = field(init=False)
#     vs: jax.Array = field(init=False)
#     v_mean_projections: jax.Array = field(init=False)

#     # nn bits
#     dynamic_r_density: typing.Callable[
#         [float, float, NNParams], float
#     ] = field(init=False)
#     dynamic_v_density: typing.Callable[
#         [float, float, float, jax.Array, jax.Array], 
#         float
#     ] = field(init = False)
#     valgrad_vs_density: typing.Callable[
#         [float, float, NNParams],
#         jax.Array
#     ] = field(init=False)
#     untrained_params: NNParams = field(init=False)
#     params: NNParams = field(init=False)

#         # pde fns
#     pde_t_residual: typing.Callable[
#         [float, NNParams], 
#         jax.Array,
#     ] = field(init=False)
    
#     pde_t_residual_loss: typing.Callable[
#         [float, NNParams],
#         float,
#     ] = field(init=False)

#     valgrad_pde_t_residual_loss: typing.Callable[
#         [float, NNParams],
#         [float, NNParams],
#     ]
    
#     iter_valgrad_pde_causality_loss: typing.Callable[
#         [float, float, float, NNParams, NNParams],
#         [float, float, NNParams, float]
#     ]

#     valgrad_pde_causality_loss: typing.Callable[
#         [jax.Array, float, NNParams, NNParams],
#         [float, NNParams]
#     ]

#     def __post_init__(self):
#         pass






    




