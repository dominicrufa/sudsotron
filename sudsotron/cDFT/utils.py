"""utilities for cdft"""
import jax 
from jax import numpy as jnp
import typing
import functools

def r_midpoints(bin_edges: jax.Array, **unused_kwargs) -> jax.Array:
    return (bin_edges[:-1] + bin_edges[1:])/2.

def dVs_from_radial_bin_edges(bin_edges: jax.Array, **unused_kwargs) -> jax.Array:
    """pull finite volumes from radial bin edges"""
    drs = jnp.diff(bin_edges)
    V0s = (4. * jnp.pi/3.)*(bin_edges[:-1])**3
    V1s = (4. * jnp.pi/3.)*(bin_edges[1:])**3
    dVs = V1s - V0s
    return dVs

def aperiodic_r(
        x: jax.Array, # vector head
        x0: jax.Array, # vector tail
        **unused_kwargs) -> float:
    """compute the aperiodic euclidean distance between x and x0"""
    return jnp.linalg.norm(x - x0)

def vectorize_aperiodic_r(
        Rs: jax.Array, 
        R0: jax.Array, 
        **unused_kwargs) -> jax.Array:
    return jax.vmap(aperiodic_r, in_axes=(0,None))(Rs, R0)

def aperiodic_reference_rs(
        Rs: jax.Array,              
        R0s: jax.Array) -> jax.Array:
    """compute an [N_R0, N_R] array of aperiodic euclidean distance vectors;
    `N_R0` is the number of reference positions; 
    `N_R` is the number of non-reference positions"""
    out_rs = jax.vmap(vectorize_aperiodic_r, in_axes=(None,0))(Rs, R0s)
    return out_rs

def cartesian_linspaces_and_retsteps(
    limits: jax.Array, # [N_dim, 2]
    num_gridpoints_per_dim: float, 
    **unused_kwargs) -> typing.Tuple[jax.Array, jax.Array]:
    """given a `limits` (shape [n_dim, 2]) for each dimension and a `num_gridpoints_per_dim` [n_dim],
    generate linspaces and retstep sizes for each dimension"""
    partial_linspaces = functools.partial(jnp.linspace, num = num_gridpoints_per_dim, retstep=True)
    cartesian_linspaces, d_spatial = jax.vmap(partial_linspaces)(limits[:,0], limits[:,1])
    return cartesian_linspaces, d_spatial

def make_cartesian_spatial_grid(
    limits: jnp.array, # [N_dim, 2]
    num_gridpoints_per_dim: float, 
    **unused_kwargs) -> typing.Tuple[jax.Array, jax.Array]:
    """compute a cartesian spatial grid given an `[n_dim, 2]` array of dimensions, an [n_dim] array for `num_gridpoints`.
    returns a tuple. the first is a tuple of (n_dim, [*num_gridpoints]), and the second is a [n_dim] array of spatial 
    gridpoint sizes"""
    cartesian_linspaces, d_spatial = cartesian_linspaces_and_retsteps(limits, num_gridpoints_per_dim)
    cartesian_grids = jnp.meshgrid(*cartesian_linspaces, indexing='ij')
    return cartesian_grids, d_spatial

def spatial_grids(
    grid_limits: jax.Array, #[2,]
    num_gridpoints_per_dim: int,
    solute_Rs: jax.Array) -> jax.Array:
    """compute a grid of euclidean distances, one for each `solute_R` (on leading axis);
    WARNING: it is unclear to me whether `r_array` in the `grid_rs` function should be transposed at first thought
    """
    grid_limits = jnp.repeat(grid_limits[jnp.newaxis, ...], repeats=3, axis=0) # repeat 3x for xyz
    (X,Y,Z), dxdydz = make_cartesian_spatial_grid(grid_limits, num_gridpoints_per_dim) # make grids and corresponding spacing
    grid_centers = (grid_limits[:,0] + grid_limits[:,1]) / 2.
    stacked_grid = jnp.vstack([X.flatten(), Y.flatten(), Z.flatten()]).T
    Rs = aperiodic_reference_rs(stacked_grid, solute_Rs)
    grid_rs = jax.vmap(lambda r_arr: jnp.reshape(r_arr.T, 
                                                 (num_gridpoints_per_dim, 
                                                  num_gridpoints_per_dim, 
                                                  num_gridpoints_per_dim)))(Rs)
    return (X,Y,Z), dxdydz, grid_rs
    
def dFidsdn(
        ns: jax.Array, # densities
        n0: float, # uniform density
        kT: float, # thermal bath energy
        **unused_kwargs) -> jax.Array:
    """compute the derivative of ideal Helmholtz free energy w.r.t. density"""
    return kT * jnp.log(ns/n0)

def dFexcsdn_HNC_Riemann_approx_aperiodic(
    ns: jax.Array, # [N,N,N]
    c_kernel: jax.Array, # [N,N,N]                           
    kT: float, 
    n0: float, 
    dx: float, 
    dy: float, 
    dz: float,
    **unused_kwargs) -> jnp.array:
    """compute the differential of excess energy functional on aperiodic gridpoints"""
    delta_n = ns - n0
    integrand = delta_n * (dx * dy * dz)
    gamma = jax.scipy.signal.fftconvolve(integrand, c_kernel, mode='same')
    return -0.5 * kT * gamma

def dFexcsdn_HNC_Riemann_approx_periodic(
    ns: jax.Array, 
    c_kernel: jax.Array, 
    kT: float, 
    n0: float, 
    dx: float, 
    dy: float, 
    dz: float,
    **unused_kwargs) -> jnp.array:
    """compute the differential of excess energy functional on periodic gridpoints"""
    delta_n = ns - n0
    integrand = delta_n * (dx * dy * dz)
    unrolled_fftconv = jnp.fft.ifftn(
        jnp.fft.fftn(integrand).conj() * jnp.fft.fftn(c_kernel)
        ).real
    unflipped_fftconv = jnp.roll(unrolled_fftconv, 
                                 shift = n.shape[0]//2, 
                                 axis = (0,1,2))
    gamma = jnp.flip(unflipped_fftconv) # flips all axes to push back to a proper convolution
    return -0.5 * kT * gamma 