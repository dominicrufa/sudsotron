"""init for `sudsotron.nn`"""
import jax 
from jax import numpy as jnp
import typing
from dataclasses import dataclass, field
from flax import linen as nn

NNParams = typing.Dict[str, typing.Union[typing.Dict, jax.Array]]
NNFn = typing.Callable[[typing.Union[float, jax.Array],
                        NNParams,
                        ...
                        ],
                       typing.Union[float, jax.Array]
                       ]

DEFAULT_NN_KEY = jax.random.PRNGKey(2024)
