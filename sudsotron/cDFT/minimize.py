"""minimizer of classical Density Functional Theory"""
import jax
from jax import numpy as jnp
import numpy as np
from dataclasses import dataclass, asdict, field
import typing
import functools

from sudsotron.nn.modules import (
    GaussianBasisMLPParams, 
    GaussianBasisMLP, 
    DEFAULT_NN_KEY,
    NNFn,
    NNParams,
)

from sudsotron.nn.utils import cosine_cutoff

from sudsotron.cDFT.dcf import (
    HNCRadialDCF,
)

from sudsotron.potentials.handlers import (
    PotentialHandler,
)

from sudsotron.cDFT.constants import DEFAULT_KT

@dataclass(frozen=True)
class cDFT:
    hnc_dcf: HNCRadialDCF
    Uext_handler: PotentialHandler
    grid_bounds: jnp.array([-1., 1.])
    num_gridpoints: int = 100
    kT: float = DEFAULT_KT
    model_params: GaussianBasisMLPParams = GaussianBasisMLPParams()
    key: jax.Array = DEFAULT_NN_KEY

    # reference arrays
    xyz_grid: typing.Tuple[jax.Array] = field(init=False)
    dxdydz: typing.Tuple[float] = field(init=False)
    c_kernel: jax.Array = field(init=False)
    Uext_R: jax.Array = field(init=False)

    # free energy functionals
    dFidsdn: typing.Callable[jax.Array, jax.Array] = field(init=False)
    dFexcsdn: typing.Callable[jax.Array, jax.Array] = field(init=False)
    dFextsdn: jax.Array = field(init=False)
    dFsdn: typing.Callable[jax.Array, jax.Array] = field(init=False)

    # nn bits
    density: typing.Callable[[jax.Array, NNParams], jax.Array] = field(init=False)
    untrained_params: NNParams = field(init=False)

    def __post_init__(self):
        # make this stuff now
        pass











