"""utilities for nn modules"""
import jax 
from jax import numpy as jnp
import typing

def unnormalized_gaussian(x: float, mu: float, sigma: float, **unused_kwargs) -> float:
    return jnp.exp(-0.5 * ((x - mu) / sigma)**2)

def gaussian_basis(x: float, mus: jax.Array, sigma: float, **unused_kwargs) -> float:
    """return a gaussian projection of `x` given an array of mus, one sigma"""
    return jax.vmap(unnormalized_gaussian, in_axes=(None, 0, None))(x, mus, sigma)

def gaussian_basis_projections(
    x: jax.Array, # [N,]
    bounds: jax.Array, # [N,2]
    projection_dim: int, 
    **unused_kwargs) -> jax.Array:
    """for each entry of `x`, `bounds`, project x into a gaussian basis of 
    dimension `projection_dim` with sigma given by (bound[1] - bound[0])/projection_dim"""
    mus, sigmas = jax.vmap(jnp.linspace, 
                           in_axes=(0,0,None, None, None)
                           )(bounds[:,0], bounds[:,1], projection_dim, True, True)
    return jax.vmap(gaussian_basis)(x, mus, sigmas)

def log_prob_gaussian_mixture(x: float, 
                              mus: jax.Array, 
                              sigmas: jax.Array, 
                              log_weights: jax.Array) -> float:
    """
    Computes the log probability of a 1D random variable x drawn from a gaussian
    mixture model.

    Args:
    x: A jnp.array of shape (N,) containing the data points.
    mus: A jnp.array of shape (K,) containing the means of the components.
    sigmas: A jnp.array of shape (K,) containing the standard deviations of the
      components.
    log_weights: A jnp.array of shape (K,) containing the log mixing coefficients.

    Returns:
    A float of the log probability
    """
    # Normalize log weights
    norm_log_weights = log_weights - jax.scipy.special.logsumexp(log_weights)

    # Compute the log probabilities of x under each component.
    unweighted_log_probs = jax.scipy.stats.norm.logpdf(x, loc=mus, scale=sigmas)
    log_probs = norm_log_weights + unweighted_log_probs

    # Compute the log probability of x under the mixture model.
    log_prob = jax.scipy.special.logsumexp(log_probs)

    return log_prob

def prob_gaussian_mixture(x: float, mus: jax.Array, 
                              sigmas: jax.Array, 
                              log_weights: jax.Array) -> float:
    """compute `log_prob_gaussian_mixture` and exp it to get normalized prob"""
    log_prob = log_prob_gaussian_mixture(x, mus, sigmas, log_weights)
    return jnp.exp(log_prob)