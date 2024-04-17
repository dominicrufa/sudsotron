"""classical Density Functional Theory handlers"""
import jax
from jax import numpy as jnp
import numpy as np
from dataclasses import dataclass, asdict, field
import typing
import functools
from flax import serialization
from importlib import resources

from sudsotron.nn.modules import (
    GaussianBasisMLPParams, 
    GaussianBasisMLP, 
    DEFAULT_NN_KEY,
    NNFn,
    NNParams,
)

from sudsotron.utils import cosine_cutoff, minimize

from sudsotron.cDFT.utils import (
    r_midpoints,
    spatial_grids, 
    dFidsdn, 
    dFexcsdn_HNC_Riemann_approx_aperiodic,
)

from sudsotron.potentials.handlers import (
    PotentialHandler,
)

DEFAULT_KT = 2.48 # amu * nm^2 / (ps^2 * particle) # this is 1kT at ~300K
DEFAULT_N0 = 32.776955 # particles/nm**3 (taken from dcf data for TIP3P at ambient conditions w/ RF electrostatics)
DEFAULT_M = 18. # amus (mass of H2O mol)
DEFAULT_R_CUT = 1.2
DEFAULT_NUM_GRIDPOINTS = 120

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
class HNCRadialDCFHandler:
    """fit and deploy a radial direct correlation 
    function in the HNC approximation.

    Example:
    >>> npz_datafile = resources.files('sudsotron.data') / 'tip3p_dcf_data.npz'
    >>> handler = HNCRadialDCFHandler(None, None, npz_filedata)
    >>> handler.fit_dcf_params() # tada
    """
    radial_bin_edges: typing.Union[jax.Array, None] # [N,]
    dcf_data: typing.Union[jax.Array, None] # [N-1], dcf data at bin edge centers
    npz_datafile: typing.Union[str, None]
    mlp_params: GaussianBasisMLPParams = GaussianBasisMLPParams()
    r_cut: typing.Union[float, None] = DEFAULT_R_CUT
    key: jax.Array = field(default_factory = lambda: DEFAULT_NN_KEY)

    bin_centers: jax.Array = field(init=False)
    bounds: jax.Array = field(init=False)
    untrained_params: NNParams = field(init=False)
    dcf: NNFn = field(init=False)
    dcf_loss: typing.Callable[NNParams, float] = field(init=False)
    params: typing.Union[NNParams, None] = field(init=False) # `None` if not `fit_on_init`

    def __post_init__(self):
        self.attempt_datafile_load()
        object.__setattr__(
            self, 
            'bin_centers', 
            r_midpoints(self.radial_bin_edges))
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
    
    def attempt_datafile_load(self):
        if self.npz_datafile is None:
            assert self.radial_bin_edges is not None
            assert self.dcf_data is not None
        else: # load these two datapoints from file
            data = np.load(self.npz_datafile)
            radial_bin_edges = jnp.array(data['radial_bin_edges'])
            dcf_data = jnp.array(data['dcf_data'])
            object.__setattr__(self, 'radial_bin_edges', radial_bin_edges)
            object.__setattr__(self, 'dcf_data', dcf_data)

    def set_params(self, params):
        object.__setattr__(self, 'params', params)
    
    def fit_dcf_params(
            self,
            **minimize_kwargs,
            ) -> typing.NamedTuple:
        """fit the dcf parameters, set the new params, 
        and return the minimizer result `NamedTuple`"""
        res = minimize(
            self.dcf_loss,
            value_and_grad = False,
            params = self.untrained_params,
            **minimize_kwargs
            )
        self.set_params(res.params)
        return res

    def serialize_to_txt(self, bytefilepath: str):
        bytes_output = serialization.to_bytes(self.params)
        with open(bytefilepath, "wb") as binary_file:
            binary_file.write(bytes_output)
    
    @classmethod
    def load_from_params_binary(cls, bytefilepath: str, radial_bin_edges, dcf_data, npz_datafile, **kwargs):
        handler = cls(radial_bin_edges, dcf_data, npz_datafile, **kwargs)
        with open(bytefilepath, "rb") as binary_file:
            _bytes = binary_file.read()
        params = serialization.from_bytes(handler.untrained_params, _bytes)
        handler.set_params(params)
        return handler


def load_HNCRadalDCFHandler(
    npz_datafile: str=None,
    params_bytefile: str=None) -> HNCRadialDCFHandler:
    """load the given dcf handler from npz datafile and params bytefile;
    default `None` reverts to respective datafiles in `sudsotron.data`"""
    npz_datafile = resources.files('sudsotron.data') / 'tip3p_dcf_data.npz' if npz_datafile is None else npz_datafile
    params_bytefile = resources.files('sudsotron.data') / 'HNCRadialDCFHandler.tip3p.params.txt' if params_bytefile is None else params_bytefile
    return HNCRadialDCFHandler.load_from_params_binary(params_bytefile, None, None, npz_datafile)


@dataclass(frozen=True)
class SScDFTHandler: # spherically symmetric classical DFT
    """
    Handler for a spherically symmetric classical DFT;
    will fit a function for density on a grid with a solute particle at origin
    with a solvent dcf kernel and interaction potential given by `hnc_dcf` and 
    `Uext_handler`, respectively.
    Empirically, loss goes to ~2.3e-3 kJ/mol/nm^3, but decreases with 
    tighter grid spacing (higher compute/memory costs); I also find that
    the radial density qualitatively looks 'good'.

    Example:
    >>> from sudsotron.cDFT.handlers import load_HNCRadialDCFHandler, SScDFTHandler
    >>> from sudsotron.potentials.potential_lib import PotentialHandler, TIP3PSCLJParameters, sc_lj
    >>> hnc_dcf = load_HNCRadalDCFHandler()
    >>> Uext_handler = PotentialHandler(TIP3PSCLJParameters, sc_lj)
    >>> dft_handler = SScDFTHandler(hnc_dcf, Uext_handler)
    >>> res = dft_handler.fit_density_params() # fit and set `params`
    """
    hnc_dcf: HNCRadialDCFHandler
    Uext_handler: PotentialHandler 
    grid_bounds: jax.Array = field(default_factory = lambda : jnp.array([-DEFAULT_R_CUT, DEFAULT_R_CUT]))
    num_gridpoints: int = DEFAULT_NUM_GRIDPOINTS
    n0: float = DEFAULT_N0
    kT: float = DEFAULT_KT
    model_params: GaussianBasisMLPParams = GaussianBasisMLPParams()
    interpolate_density: bool = False
    key: jax.Array = field(default_factory = lambda: DEFAULT_NN_KEY)

    # reference arrays
    R: jax.Array = field(init=False)
    dcf_kernel: jax.Array = field(init=False)

    # free energy functionals
    dFidsdn: typing.Callable[jax.Array, jax.Array] = field(init=False)
    dFexcsdn: typing.Callable[jax.Array, jax.Array] = field(init=False)
    dFextsdn: jax.Array = field(init=False)
    dFsdn: typing.Callable[jax.Array, jax.Array] = field(init=False)
    dFsdn_loss: typing.Callable[NNParams, float] = field(init=False)

    # nn bits
    density: typing.Callable[[float, float, NNParams], float] = field(init=False)
    grid_density: typing.Callable[[NNParams], jax.Array] = field(init=False)
    untrained_params: NNParams = field(init=False)
    params: NNParams = field(init=False)

    def __post_init__(self):
        # grid stuff
        self.set_R()
        self.set_dcf_kernel()

        # density functions/params
        self.set_densities_and_untrained_params()

        # dFs
        self.set_dFidsdn()
        self.set_dFexcsdn()
        self.set_dFextsdn()
        self.set_dFsdn()
        self.set_dFsdn_loss()
    
    def set_R(self):
        """set `R`, the grid of euclidean distances"""
        assert self.num_gridpoints % 2 == 0, f"`num_gridpoints` must be even to avoid LJ singularity @ origin"
        xyz_tuple, dxdydz_arr, R_grids = spatial_grids(
            grid_limits = self.grid_bounds,
            num_gridpoints_per_dim = self.num_gridpoints,
            solute_Rs = jnp.zeros((1, 3)) # only a solute particle at origin
        )
        #R_grids has leading axis corresponding to 1 for num_solute_particles (=1)
        R_grid = R_grids[0]
        object.__setattr__(self, 'R', R_grid)
    
    def set_dcf_kernel(self):
        """set the `dcf_kernel` from `R`"""
        dcf_fn, dcf_params = self.hnc_dcf.dcf, self.hnc_dcf.params
        partial_dcf_fn = functools.partial(dcf_fn, params = dcf_params)
        grid_dcf_fn = jax.vmap(jax.vmap(jax.vmap(partial_dcf_fn)))
        dcf_kernel = grid_dcf_fn(self.R)
        object.__setattr__(self, 'dcf_kernel', dcf_kernel)
    
    def set_densities_and_untrained_params(self):
        model = GaussianBasisMLP(**asdict(self.model_params))
        untrained_params = model.init(self.key, jnp.zeros(1)) # for r
        object.__setattr__(self, 'untrained_params', untrained_params)
        def norm_pert_density(r, params):
            u = model.apply(params, jnp.array([r]))[0]
            u = u * cosine_cutoff(r, self.hnc_dcf.r_cut)
            return jnp.exp(-u / self.kT)

        def norm_ideal_density(Uext):
            return jnp.exp(-Uext / self.kT)

        def density(r, Uext, params):
            return self.n0 * norm_pert_density(r, params) * norm_ideal_density(Uext)
            
        object.__setattr__(self, 'density', density)

        Uext_on_line = jax.vmap(self.Uext_handler.paramd_potential)
        if self.interpolate_density:
            rs = jnp.linspace(0., self.hnc_dcf.r_cut, self.num_gridpoints)
            line_density = lambda params: jax.vmap(
                density, 
                in_axes = (0,0,None)
                )(rs, Uext_on_line(rs), params)
            def grid_density(params):
                line_densities = line_density(params)
                interp_densities = jnp.interp(self.R.flatten(), rs, line_densities, right = self.n0)
                return interp_densities.reshape(self.R.shape)
        else:
            def grid_density(params):
                Uexts = Uext_on_line(self.R.flatten())
                return jax.vmap(density, in_axes=(0,0,None))(
                    self.R.flatten(),
                    Uexts,
                    params
                ).reshape(self.R.shape)
        
        object.__setattr__(self, 'grid_density', grid_density)
    
    def set_dFidsdn(self):
        partial_dFidsdn = functools.partial(
            dFidsdn,
            n0 = self.n0,
            kT = self.kT
        )
        object.__setattr__(self, 'dFidsdn', partial_dFidsdn)
    
    def set_dFexcsdn(self):
        _, dx = jnp.linspace(*self.grid_bounds, 
                             self.num_gridpoints, 
                             retstep=True)
        partial_dFexcsdn = functools.partial(
            dFexcsdn_HNC_Riemann_approx_aperiodic,
            c_kernel = self.dcf_kernel,
            kT = self.kT,
            n0 = self.n0,
            dx = dx, dy = dx, dz = dx)
        object.__setattr__(self, 'dFexcsdn', partial_dFexcsdn)
    
    def set_dFextsdn(self):
        object.__setattr__(
            self, 
            'dFextsdn', 
            jax.vmap(jax.vmap(jax.vmap(self.Uext_handler.paramd_potential)))
            )

    def set_dFsdn(self):
        def dFsdn(params):
            grid_densities = self.grid_density(params)
            _dFsdn = (self.dFidsdn(grid_densities) 
                      + self.dFexcsdn(grid_densities) 
                      + self.dFextsdn(self.R))
            return _dFsdn
        object.__setattr__(self, 'dFsdn', dFsdn)
    
    def set_dFsdn_loss(self):
        loss = lambda params: jnp.sum(self.dFsdn(params)**2) / jnp.prod(jnp.array(self.R.shape))
        object.__setattr__(self, 'dFsdn_loss', loss)
    
    def set_params(self, params: NNParams):
        object.__setattr__(self, 'params', params)

    def fit_density_params(
            self,
            **minimize_kwargs,
            ) -> typing.NamedTuple:
        """fit the density parameters, set the new params, 
        and return the minimizer result `NamedTuple`"""
        res = minimize(
            self.dFsdn_loss,
            value_and_grad = False,
            params = self.untrained_params,
            **minimize_kwargs
            )
        self.set_params(res.params)
        return res