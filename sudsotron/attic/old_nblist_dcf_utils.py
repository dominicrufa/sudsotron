"""A collection of old functionality to compute particle displacements, compute neighbor lists, distances, and total/direct correlation functions"""
import jax
from jax import numpy as jnp
import numpy as np
import typing
import functools
from scipy.fft import dst

# neighbor list utility
def displacement(Ra: jax.Array, Rb: jax.Array, box_size: jax.Array, **unused_kwargs) -> jax.Array:
    """given positions `Ra`, `Rb` and (potentially) a jax.Array (of shape 3) delineating the lengths of a periodic box,
    compute the displacement vector of Rb - Ra;
    NOTE: if the `box_size` is close to zeros, will assume nonperiodic"""
    dR = Rb - Ra
    modded_dR = jnp.mod(dR + box_size * 0.5, box_size) - 0.5 * box_size
    out = jax.lax.select(
        jnp.allclose(box_size, 0),
        dR,
        modded_dR)
    return out

def slow_neighbor_list(R: jax.Array, box_size: jax.Array, r_cut: 1.2, pad_size: int) -> jax.Array:
    """build a neighbor list by iterating over _all_ particles in parallel and computing distances"""
    
    def singular_vec_distance(R: jax.Array, Ra_idx: int, box_size: jax.Array):
        Ra = R.at[Ra_idx].get()
        displacements = jax.vmap(displacement, in_axes=(None, 0, None))(Ra, R, box_size)
        rs = jnp.linalg.norm(displacements, axis=1).squeeze()
        leq_r_cut = rs <= r_cut
        leq_r_cut = leq_r_cut.at[Ra_idx].set(False)
        indices = jnp.where(leq_r_cut, size = pad_size, fill_value=-1)[0]
        out_rs = jnp.where(indices > -1, rs.at[indices].get(), -1)
        # determine buffer overflows
        buffers = jnp.count_nonzero(indices == -1)
        sum_minus1s = jnp.sum(buffers)
        did_buffer_overflow = sum_minus1s == 0
        return indices, out_rs, did_buffer_overflow
    
    neighbor_list, out_rs, did_overflow = jax.vmap(singular_vec_distance, in_axes=(None, 0, None))(R, jnp.arange(len(R)), box_size)
    overflow = jnp.any(did_overflow)
    return neighbor_list, out_rs, overflow

def compute_r2s_by_frame(positions: jax.Array, **unused_kwargs) -> jax.Array:
    """compute the displacement of each particle per every frame for a moving window; presumed not periodic"""
    num_frames, num_particles, dim = positions.shape
    r2 = lambda _Ra, _Rb: jnp.linalg.norm(displacement(_Ra, _Rb, jnp.zeros(3)))**2
    vmapped_r2 = jax.vmap(r2, in_axes=(0,0)) # vmap across all particles in frame
    vvmapped_r2 = jax.vmap(vmapped_r2, in_axes=(None, 0)) # vmap over Ra 
    return vvmapped_r2(positions[0], positions)  

def compute_rs_histogram(
    posits: jax.Array,
    bvs: jax.Array,
    r_cut: float,
    buffer_multiplier: float=1.5,
    bin_width: float=0.02,
    **unused_kwargs):
    """iterate over `posit_bvs_filenames`, compute all particle radial distances, and histogram"""
    # first, attempt to guess the size of the neighbor list based on the number of particles
    # in the box and the volume of the r_cut sphere
    num_frames, num_particles, dim = posits.shape
    print(f"num_frames, num_particles, dim: {num_frames}, {num_particles}, {dim}")
    r_cut_vol = 4. * jnp.pi * r_cut**3 / 3.
    mean_vol = jnp.prod(bvs.sum(axis=1), axis=-1).mean()
    print(mean_vol)
    mean_density = num_particles / mean_vol
    mean_particles_per_r_cut = mean_density * r_cut_vol
    buffer_added_particles = int(mean_particles_per_r_cut * buffer_multiplier)
    print(f"max neighbors per particle: {buffer_added_particles}")
    bin_edges = jnp.arange(start = 0., stop = r_cut + bin_width, step=bin_width)
    out_hist = jnp.zeros(len(bin_edges) - 1)
    divisor = num_particles * num_frames
    
    neigh_list_fn = jax.jit(functools.partial(slow_neighbor_list, r_cut = r_cut, pad_size = buffer_added_particles))
    for frame in tqdm.trange(num_frames):
        _posits = posits[frame]
        _bvs = bvs[frame].sum(axis=1) # flatten bvs
        neighbors, rs, did_overflow = neigh_list_fn(_posits, _bvs)
        flat_rs = rs.flatten()
        assert not did_overflow
        masked_rs = jnp.extract(flat_rs > -1, flat_rs)
        _inner_hist, _redundant_bin_edges = jnp.histogram(masked_rs, bin_edges)
        out_hist = out_hist + _inner_hist
    return out_hist, bin_edges, divisor 

def radial_density(r_hist_data: jax.Array, bin_edges: jax.Array,  **unused_kwargs) -> jax.Array:
    sphere_vol_fn = lambda r: 4. * jnp.pi * (r**3) / 3.
    sphere_vol_diffs = jnp.diff(sphere_vol_fn(bin_edges))
    density = hist_data / sphere_vol_diffs
    return density

def data_direct_c(dr: float, rs: jnp.array, grs: jnp.array, rho_unif: float = np.array(n0), **unused_kwargs):
    """compute the direct correlation function from total correlation data `grs` and bin data;
    To avoid ringing, I noticed that a `dr`~=1e-3 works relatively well.
    `crs` is the radial dcf at the values `rs`"""
    length = rs.shape[0]
    dk = np.pi/(dr*length)
    ks = np.linspace(0., length * dk, length + 1)[1:]
    dst_2_coeffs = 2.0*np.pi *rs*dr
    dst_3_coeffs = ks*dk/(4.0*np.pi*np.pi)
    
    hrs = grs - 1.
    hks = dst(dst_2_coeffs*hrs,type=2)/ks
    cks = hks / (1. + (rho_unif* hks))
    crs = dst(dst_3_coeffs * cks, type=3) / rs
    
    return ks, hrs, hks, crs, cks 