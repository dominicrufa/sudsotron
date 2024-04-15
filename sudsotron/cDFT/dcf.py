"""modules for training, deployment, and analysis of direct correlations"""
import jax
from jax import numpy as jnp
import numpy as np
from dataclasses import dataclass, asdict, field
import typing
import functools

from sudsotron.utils import minimize
from sudsotron.cDFT import utils, constants
from sudsotron.nn.modules import (
    GaussianBasisMLPParams, 
    GaussianBasisMLP, 
    DEFAULT_NN_KEY,
    NNFn,
    NNParams,
)

def dcf_helper(
        r: float, 
        params: NNParams, 
        r_cut: typing.Union[float, None],
        model: GaussianBasisMLP) -> float:
    r = jnp.abs(r)
    x = jnp.array([r])
    out = model.apply(params, x)[0]
    if r_cut is not None:
        out = out * cosine_cutoff(r, r_cut)
    return out

def dcf_loss(params: NNParams, r_midpoints: jax.Array, dcf_data, dcf_fn: NNFn) -> float:
    calc_xs = jax.vmap(dcf_fn, in_axes=(0,None))(r_midpoints, params)
    return (1. / len(dcf_data)) * jnp.linalg.norm(calc_xs - dcf_data)**2

@dataclass(frozen=True)
class HNCRadialDCF:
    """fit and deploy a radial direct correlation 
    function in the HNC approximation.
    """
    radial_bin_edges: jax.Array # [N,]
    dcf_data: jax.Array # [N-1], dcf data at bin edge centers
    mlp_params: GaussianBasisMLPParams = GaussianBasisMLPParams()
    r_cut: typing.Union[float, None] = constants.DEFAULT_R_CUT
    key: jax.Array = field(default_factory = lambda: DEFAULT_NN_KEY)

    bin_centers: jax.Array = field(init=False)
    bounds: jax.Array = field(init=False)
    untrained_params: NNParams = field(init=False)
    dcf: NNFn = field(init=False)
    dcf_loss: typing.Callable[NNParams, float] = field(init=False)
    params: typing.Union[NNParams, None] = field(init=False) # `None` if not `fit_on_init`

    def __post_init__(self):
        object.__setattr__(
            self, 
            'bin_centers', 
            utils.r_midpoints(self.radial_bin_edges))
        object.__setattr__(
            self, 
            'bounds', 
            jnp.array(
                [self.radial_bin_edges[0], self.radial_bin_edges[-1]]
                )
                 )
        model = GaussianBasisMLP(**asdict(self.mlp_params))
        untrained_params = model.init(self.key, jnp.zeros(1))
        dcf = functools.partial(
                    dcf_helper, 
                    r_cut = self.r_cut, 
                    model=model)
        object.__setattr__(
            self, 
            'untrained_params', 
            untrained_params)
        
        object.__setattr__(self, 'dcf', jax.jit(dcf))
        object.__setattr__(
            self,
            'dcf_loss',
            jax.jit(
                functools.partial(
                    dcf_loss, 
                    r_midpoints = self.bin_centers, 
                    dcf_data = self.dcf_data,
                    dcf_fn = self.dcf
                    )
            ) 
        )
        object.__setattr__(self, 'params', None)
    
    def set_params(self, params):
        object.__setattr__(self, 'params', params)
    
    def fit_dcf_params(
            self,
            **minimize_kwargs,
            ) -> typing.NamedTuple:
        """fit the dcf parameters, set the new params, and return the minimizer result `NamedTuple`"""
        res = minimize(
            self.dcf,
            value_and_grad = False,
            params = self.untrained_params,
            **minimize_kwargs)
        self.set_params(res.params)
        return res