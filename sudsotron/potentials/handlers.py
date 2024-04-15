"""handle objects for potentials"""
import jax
from jax import numpy as jnp
import typing
import numpy as np
from dataclasses import dataclass, asdict, field, fields
from flax.core import FrozenDict, copy
from flax import linen as nn

from sudsotron.nn.modules import (
    GaussianBasisMLPParams, 
    GaussianBasisMLP, 
    DEFAULT_NN_KEY,
    NNFn,
    NNParams,
)

from sudsotron.nn.utils import cosine_cutoff

from sudsotron.potentials.potential_lib import (
    PotentialFn,
    DynamicPotentialFn,
)


def dynamic_potential(
        x: float, 
        t: float, 
        static_params: FrozenDict, 
        potential_fn: typing.Callable[[float, ...],float], 
        dynamic_kwargs_fn: typing.Callable[float, typing.Dict]) -> float:
    params = copy(static_params, dynamic_kwargs_fn(t))
    return potential_fn(x, **params)

@dataclass(frozen=True)
class PotentialHandler:
    potential_params: FrozenDict
    potential: PotentialFn
    paramd_potential: PotentialFn = field(init=False)

def __post_init__(self):
    paramd_potential = lambda x: self.potential(x, **self.potential_params)
    object.__setattr__(self, 'paramd_potential', paramd_potential)



@dataclass(frozen=True)
class DynamicPotentialHandler:
    potential_params: FrozenDict
    potential: DynamicPotentialFn
    dynamic_kwargs: typing.Callable[
        float, 
        typing.Dict[str, typing.Union[jax.Array, float]]]
    dynamic_potential: DynamicPotentialFn = field(init=False)

def __post_init__(self):
    dynamic_potential = jax.jit(
        functools.partial(
            dynamic_potential, 
            static_params = self.potential_params,
            potential_fn = self.potential,
            dynamic_kwargs_fn = self.dynamic_kwargs
            )
    )
    object.__setattr__(self, 'dynamic_potential', dynamic_potential)

def dynamic_nn_potential(
        r: float,
        t: float,
        nn_params: NNParams,
        zero_at_t0: bool,
        r_cut: float,
        T: float, 
        model: nn.Module,
        **unused_kwargs) -> float:
    u = model.apply(nn_params, jnp.array([r, t]))
    u = u * cosine_cutoff(r, r_cut)
    if zero_at_t0:
        u = u * t / T
    return u

@dataclass(frozen=True)
class DynamicNeuralPotentialHandler:
    """deploy a generic neural network potential
    from a `GaussianBasisMLP`
    """
    # training parameters
    mlp_params: GaussianBasisMLPParams = GaussianBasisMLPParams() # contains R_CUT, T
    r_cut: float = field(init=False)
    T: float = field(init=False)
    key: jax.Array = field(default_factory = lambda : DEFAULT_NN_KEY)
    zero_at_t0: bool = True
    init_params_scalar: float = 1e-6

    # parameters
    dynamic_potential: DynamicPotentialFn
    untrained_params: NNParams = field(init=False)
    params: typing.Union[NNParams, None] = field(init=False) # `None` if not `fit_on_init`

    def __post_init__(self):
        # define `r_cut`, `T`
        r_cut = self.mlp_params.bounds[0,1]
        T = self.mlp_params.bounds[1,1]
        object.__setattr__(self, 'r_cut', r_cut)
        object.__setattr__(self, 'T', T)

        # make model
        model = GaussianBasisMLP(**asdict(self.mlp_params))
        untrained_params = model.init(self.key, jnp.zeros(2)) # (r, t)
        untrained_params = jax.tree_util.tree_map(
            lambda x: x * self.init_params_scalar,
            untrained_params
        )
        dynamic_potential = functools.partial(
            dynamic_nn_potential,
            zero_at_t0 = self.zero_at_t0,
            r_cut = self.r_cut,
            T = self.T, 
            model = model,
            )
        object.__setattr__(self, 'untrained_params', untrained_params)
        object.__setattr__(self, 'dynamic_potential', jax.jit(dynamic_potential))
        object.__setattr__(self, 'params', None)
    
    def set_params(self, params: NNParams):
        object.__setattr__(self, 'params', params)


def dynamic_hybrid_potential(
        x: float,
        t: float,
        params: NNParams,
        dynamic_potential: DynamicPotentialFn,
        dynamic_nn_potntial: DynamicPotentialFn,
        ) -> float:
    pot = dynamic_potential(x, t)
    nn_pot = dynamic_nn_potential(x, t, params)
    return pot + nn_pot

@dataclass(frozen=True)
class DynamicHybridPotentialHandler:
    dynamic_potential_handler: DynamicPotentialHandler
    dynamic_nn_potential_handler: DynamicNeuralPotentialHandler
    dynamic_hybrid_potential: DynamicPotentialFn = field(init=False)

def __post_init__(self):
    dynamic_hybrid_potential = functools.partial(
        dynamic_hybrid_potential,
        dynamic_potential = self.dynamic_potential_handler.dynamic_potential,
        dynamic_nn_potntial = self.dynamic_nn_potential_handler.dynamic_potential,
        )
    object.__setattr__(
        self, 
        'dynamic_hybrid_potential', 
        jax.jit(dynamic_hybrid_potential))
