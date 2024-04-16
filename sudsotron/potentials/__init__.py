"""potentials"""
from jax import numpy as jnp
import typing
import numpy as np
from dataclasses import dataclass, asdict, fields, field
from flax.core import FrozenDict

# potential types
PotentialFn = typing.Callable[[float, ...], float]
DynamicPotentialFn = typing.Callable[[float, float, ...], float]