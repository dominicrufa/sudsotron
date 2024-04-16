"""generic utilities"""
import jax
from jax import numpy as jnp
import numpy as np
import jaxopt
import abc
import typing
import functools

from sudsotron.nn.modules import (
    NNParams,
)

def cosine_cutoff(r: float, r_cut: float) -> float:
    """cosine cutoff function that decays smoothly to zero at `r_cut`"""
    return jax.lax.select(r >= r_cut, 0 * r, 1 + jnp.cos(r * jnp.pi / r_cut)) * 0.5

def minimize(
        loss_fn: typing.Union[
            typing.Callable[NNParams, float], # val
            typing.Callable[
                NNParams, 
                typing.Tuple[float, NNParams]] # or value and grad
        ],
        value_and_grad: bool,
        params: NNParams,
        minimizer: abc.ABCMeta = jaxopt.ScipyMinimize,
        verbose: typing.Union[str, bool]='overwrite',
        minimizer_kwargs = {'maxiter': 9999, 'tol': 1e-6},
        **unused_kwargs,
        ) -> typing.NamedTuple:
    """BFGS minimize some parameters of the nn flavour; 
    see `jaxopt` docs for specs.

    if `verbose`, print the loss on every call to `loss_fn`;
    if `verbose` == 'overwrite', will print/overwrite the loss on every call to `loss_fn`;
    if not `verbose`, never print loss.
    """
    assert verbose in [True, False, 'overwrite']
    valgrad_fn = jax.jit(jax.value_and_grad(loss_fn)) if not value_and_grad else loss_fn
    minimizer_kwargs.update({'jit': False, 'value_and_grad': True})
    if not verbose:
        def call(_loss): pass
    else:
        ender = '\n' if verbose == 'overwrite' else '\r'
        def call(_loss): print(_loss, end=ender)

    def _loss_fn(params):
        _loss, _grad_loss = valgrad_fn(params)
        call(_loss)
        return _loss, _grad_loss
    solver = minimizer(_loss_fn, **minimizer_kwargs)    
    res = solver.run(params)
    return res