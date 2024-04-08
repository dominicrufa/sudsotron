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

from sudsotron.nn.utils import cosine_cutoff

from sudsotron.cDFT.dcf import (
    HNCRadialDCF,
)

from sudsotron.potentials.handlers import (
    DynamicPotentialHandler,
    DynamicNeuralPotentialHandler,
)

from sudsotron.cDFT.utils import (
    spatial_grids,
    dFexcsdn_HNC_Riemann_approx_aperiodic,
)

@dataclass(frozen=True)
class SSKKFE(SScDFT): # spherically symmetric KKFE
    pass