"""classical Density Functional Theory"""
import jax
from jax import numpy as jnp
import numpy as np
from dataclasses import dataclass, asdict, field
import typing
import functools

from sudsotron.nn import (
    GaussianBasisMLPParams, 
    GaussianBasisMLP, 
    DEFAULT_NN_KEY,
    NNFn,
    NNParams,
)

from sudsotron.utils import cosine_cutoff, minimize

from sudsotron.cDFT.utils import (
    spatial_grids, 
    dFidsdn, 
    dFexcsdn_HNC_Riemann_approx_aperiodic,
)

from sudsotron.potentials import (
    PotentialHandler,
)

from sudsotron.cDFT.constants import (
    DEFAULT_KT, 
    DEFAULT_GRID_FLOATTYPE,
    DEFAULT_N0
)

DEFAULT_KT = 2.48 # amu * nm^2 / (ps^2 * particle) # this is 1kT at ~300K
DEFAULT_N0 = 32.776955 # particles/nm**3 (taken from dcf data for TIP3P at ambient conditions w/ RF electrostatics)
DEFAULT_M = 18. # amus (mass of H2O mol)
DEFAULT_R_CUT = 1.2
DEFAULT_GRID_FLOATTYPE = jnp.float32

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
class HNCRadialDCF:
    """fit and deploy a radial direct correlation 
    function in the HNC approximation.
    """
    radial_bin_edges: jax.Array # [N,]
    dcf_data: jax.Array # [N-1], dcf data at bin edge centers
    mlp_params: GaussianBasisMLPParams = GaussianBasisMLPParams()
    r_cut: typing.Union[float, None] = constants.DEFAULT_R_CUT
    key: jax.Array = field(default_factory = lambda: DEFAULT_NN_KEY)

    bin_centers: jax.Array = field(init=False)
    bounds: jax.Array = field(init=False)
    untrained_params: NNParams = field(init=False)
    dcf: NNFn = field(init=False)
    dcf_loss: typing.Callable[NNParams, float] = field(init=False)
    params: typing.Union[NNParams, None] = field(init=False) # `None` if not `fit_on_init`

    def __post_init__(self):
        object.__setattr__(
            self, 
            'bin_centers', 
            utils.r_midpoints(self.radial_bin_edges))
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
    
    def set_params(self, params):
        object.__setattr__(self, 'params', params)
    
    def fit_dcf_params(
            self,
            **minimize_kwargs,
            ) -> typing.NamedTuple:
        """fit the dcf parameters, set the new params, and return the minimizer result `NamedTuple`"""
        res = minimize(
            self.dcf,
            value_and_grad = False,
            params = self.untrained_params,
            **minimize_kwargs)
        self.set_params(res.params)
        return res

def density_from_model(
        r: float, 
        params: NNParams, 
        r_cut: float, 
        n0: float, 
        kT: float, 
        model: GaussianBasisMLP) -> NNFn:
    u = model.apply(params, jnp.array([r]))[0]
    u = u * cosine_cutoff(r, r_cut)
    return n0 * jnp.exp(-u / kT)

@dataclass(frozen=True)
class SScDFT: # spherically symmetric classical DFT
    hnc_dcf: HNCRadialDCF
    Uext_handler: PotentialHandler
    grid_bounds: jax.Array = field(default_factory = lambda : jnp.array([-1., 1.]))
    num_gridpoints: int = 100
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
    
    def set_R(self):
        """set `R`, the grid of euclidean distances"""
        assert self.num_gridpoints % 2 == 0, f"`num_gridpoints` must be even to avoid LJ singularity @ origin"
        R_grid = spatial_grids(
            grid_limits = self.grid_bounds,
            num_gridpoints_per_dim = self.num_gridpoints,
            solute_Rs = jnp.zeros((1, 3)), dtype = DEFAULT_GRID_FLOATTYPE)
        object.__setattr__(self, 'R', R_grid[0])
    
    def set_dcf_kernel(self):
        """set the `dcf_kernel` from `R`"""
        dcf_fn, dcf_params = hnc_dcf.dcf, hnc_dcf.params
        partial_dcf_fn = functools.partial(dcf, params = dcf_params)
        grid_dcf_fn = jax.vmap(jax.vmap(jax.vmap(partial_dcf)))
        dcf_kernel = grid_dcf_fn(self.R)
        object.__setattr__(self, 'dcf_kernel', dcf_kernel)
    
    def set_densities_and_untrained_params(self):
        model = GaussianBasisMLP(**asdict(self.mlp_params))
        untrained_params = model.init(self.key, jnp.zeros(1)) # for r
        object.__setattr__(self, 'untrained_params', untrained_params)
        perturbation_density = functools.partial(
            density_from_model,
            r_cut = self.hnc_dcf.r_cut, 
            n0 = self.n0, 
            kT = self.kT, 
            model = model 
        )
        ideal_density = lambda Uext: self.n0 * jnp.exp(-Uext/self.kT)
        density = lambda r, Uext, params: perturbation_density(r, params) * ideal_density(Uext)
        object.__setattr__(self, 'density', density)

        if self.interpolate_density:
            rs = jnp.linspace(0., self.hnc_dcf.r_cut, self.num_gridpoints)
            Uext_on_line = jax.vmap(self.Uext_handler.paramd_potential)
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
        dFsdn = lambda params: self.dFidsdn(self.R) + self.dFexcsdn(self.R) + self.dFextsdn(self.R)
        object.__setattr__(self, 'dFsdn', dFsdn)
    
    def set_dFsdn_loss(self):
        loss = lambda params: jnp.sum(self.dFsdn(params)**2) / jnp.prod(jnp.array(self.R.shape))
        object.__setattr__(self, 'dFsdn_loss', loss)
    
    def fit_density_params(
            self, 
            params: NNParams,
            maxiter: int = 9999,
            tol: float = 1e-6,
            jupyter_verbose: bool=True) -> typing.NamedTuple:
        if jupyter_verbose:
            from IPython.display import display, clear_output
            def call(_loss):
                clear_output(wait=True)
                display(f"loss: {_loss}")
        else:
            def call(_loss): pass

        jitd_valgrad_loss = jax.jit(jax.value_and_grad(self.dFsdn_loss))
        def valgrad_loss(params):
            _loss, _grad_loss = jitd_valgrad_loss(params)
            call(_loss)
            return _loss, _grad_loss
        solver = jaxopt.BFGS(
            valgrad_loss, 
            value_and_grad=True, 
            maxiter=maxiter,
            tol = tol,
            jit=False)
        res = solver.run(params)
        return res
    
    def set_params(self, params: NNParams):
        object.__setattr__(self, 'params', params)