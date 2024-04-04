"""a library of potentials"""
from jax import numpy as jnp
import typing
import numpy as np
from dataclasses import dataclass, asdict

@dataclass(frozen=True)
class LJParameters:
    """lennard jones parameters allowing softcore;
    default parameters define a pair of tip3p water oxygen steric 
    interations with water 1 being unique new"""
    # standard parameters
    lambda_select: float = 1. 
    os1: float = 0.3150752406575124  
    os2: float = 0.3150752406575124
    ns1: float = 0.3150752406575124
    ns2: float = 0.3150752406575124
    oe1: float = 0.
    ne1: float = 0.635968
    oe2: float = 0.635968
    ne2: float = 0.635968
    uo1: int = 0
    uo2: int = 0
    un1: int = 1
    un2: int = 0
    lj_switch: float = 50.
    lj_max: float = 100.

    # softcore parameters
    softcore_alpha: float = 0.5
    softcore_beta: float = 0.5
    softcore_b: int = 1
    softcore_c: int = 6
    softcore_d: int = 1
    softcore_e: int = 1
    softcore_f: int = 2

TIP3P_LJ_PARAMETERS = LJParameters()

def pw_lin_to_quad_to_const(
        x: float, 
        x_sw: float, 
        y_max: float, 
        **unused_kwargs):
    """define a piecewise function of x s.t.
    define a quadratic term w/ a y_max; 
    the y_max defines an x_sw2 = 2*y_max - x_sw;
    quadratic is defined by y = a * (x - x_sw2)^2 + y_max w/ a = 1. / (2. * (x_sw1 - x_sw2))
    1. less than x_sw, return x;
    2. between x_sw and x_sw2, return quadratic;
    3. greater than x_sw2, return y_max
    """
    x_sw2 = 2 * y_max - x_sw # define x_sw2
    a = 1. / (2. * (x_sw - x_sw2)) # eval a for quadratic
    quad = lambda _x: a * (_x - x_sw2)**2 + y_max # define quad fn
    lin_to_quad = lambda _x: jax.lax.select(_x < x_sw, _x, quad(_x)) # define linear_to_quadratic w/ lower bound

    out = jax.lax.select(x > x_sw2,
                        jnp.array([y_max]),
                        jnp.array([lin_to_quad(x)]))[0]
    return out

def sc_lj(
        r, 
        lambda_select, # radius, lambda_select
        os1, os2, ns1, ns2, # old sigma 1, old sigma 2, new sigma 1, new sigma 2
        oe1, oe2, ne1, ne2, # old epsilon 1, old epsilon 2, new epsilon 1, new epsilon 2
        uo1, uo2, un1, un2, # unique old 1, unique old 2, unique new 1, unique new 2
        softcore_alpha, 
        softcore_b, 
        softcore_c,
        lj_switch, 
        lj_max,
        **unused_kwargs) -> float:
    """define a softcore lennard jones potential;
    WARNING: this may yet fail at r=0."""
    # pad r to avoid nan

    # uniques
    uo, un = uo1 + uo2, un1 + un2 # combiner for unique old/new
    unique_old = jax.lax.select(uo >= 1, 1, 0)
    unique_new = jax.lax.select(un >= 1, 1, 0)

    # sigmas
    os = 0.5 * (os1 + os2)
    ns = 0.5 * (ns1 + ns2)

    # epsilons
    oe = jnp.sqrt(oe1 * oe2)
    ne = jnp.sqrt(ne1 * ne2)

    # scaling sigma, epsilon by lambda_select
    res_s = os + lambda_select * (ns - os)
    res_e = oe + lambda_select * (ne - oe)

    # lambda sub for `reff_lj`
    lam_sub = unique_old * lambda_select + unique_new * (1. - lambda_select)
    reff_lj_term1 = softcore_alpha * (lam_sub**softcore_b)
    reff_lj_term2 = (r/res_s)**softcore_c
    reff_lj = res_s * (reff_lj_term1 + reff_lj_term2)**(1./softcore_c)
    #reff_lj = jax.lax.select(reff_lj <- 1e-6, 1e-6, reff_lj) # protect small reff_lj

    # canonical softcore form/protect nans
    lj_x = (res_s / reff_lj)**6
    lj_e = 4. * res_e * lj_x * (lj_x - 1.)
    lj_e = jnp.nan_to_num(lj_e, nan=jnp.inf)
    lj_e = pw_lin_to_quad_to_const(lj_e, lj_switch, lj_max) # add switching so its second order differentiable
    return lj_e