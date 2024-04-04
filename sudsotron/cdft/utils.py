"""utilities for cdft"""
import jax
from jax import numpy as jnp
import numpy as np
import typing
import functools


def aperiodic_r(x: jax.Array, # vector head
                x0: jax.Array, # vector tail
                **unused_kwargs) -> float:
    """compute the aperiodic euclidean distance between x and x0"""
    return jnp.linalg.norm(x - x0)

def vectorize_aperiodic_3D_distance(X: jnp.array, Y: jnp.array, Z: jnp.array, 
    x0: float, y0: float, z0: float, **unused_kwargs) -> jnp.array:
    """vmap the x, y, z components of `aperiodic_3D_distance` on a 3D grid"""
    out_fn = jax.vmap(
        jax.vmap(
            jax.vmap(aperiodic_3D_distance, in_axes=(0,0,0,None,None,None)),
            in_axes=(0,0,0,None,None,None)),
        in_axes=(0,0,0,None,None,None))
    return out_fn(X, Y, Z, x0, y0, z0)