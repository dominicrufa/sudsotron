class BaseDFT(object):
    """"""
    def __init__(self, 
                 c_fn: AUX_FN,
                 Uext_fn: AUX_FN = Uext_lj,
                 grid_limits: jax.Array = jnp.array([-1., 1.]),
                 num_gridpoints: int = 200,
                 solute_positions: jax.Array = jnp.array([[0., 0., 0.]]),
                 n0: float = DEFAULT_N0,
                 kT: float = DEFAULT_KT,
                 n_model_kwargs: KWARG_DICT = DEFAULT_MODEL_KWARG_DICT,
                 key: jax.Array = jax.random.PRNGKey(42),
                 periodic_Fexc: bool = False,
                 use_cutoff: bool=True,
                 interp_grid: bool=False,
                 **kwargs):
        # fns and default kwargs
        self.c_fn = c_fn
        self.Uext_fn = Uext_fn
        self.interp_grid = interp_grid
        
        # grid handling
        assert grid_limits[1] > grid_limits[0], f"left grid limit must be less than right grid limit"
        assert num_gridpoints % 2 == 0, f"num_gridpoints must be even to avoid singularity of lj site at origin"
        self.grid_limits = grid_limits
        self.num_gridpoints = num_gridpoints
        self.solute_positions = solute_positions
        self.make_spatial_grids(grid_limits, num_gridpoints, solute_positions, **kwargs)
        self.r_cut = 0.5 * (grid_limits[1] - grid_limits[0])

        # misc attrs
        self.n0 = n0
        self.kT = kT
        self.periodic_Fexc = periodic_Fexc
        self.use_cutoff = use_cutoff
        self.dFexcdn_fn = dFexcdn_HNC_Riemann_approx_periodic if self.periodic_Fexc else dFexcdn_HNC_Riemann_approx_aperiodic
        
        # make n fn and default kwargs
        self.make_n_fns(n_model_kwargs, key, **kwargs)
    
    def make_spatial_grids(self, grid_limits, num_gridpoints, solute_positions, **unused_kwargs):
        grid_limits = jnp.repeat(grid_limits[jnp.newaxis, ...], repeats=3, axis=0) # repeat 3x for xyz
        self.XYZ, self.dxdydz = make_cartesian_spatial_grid(
            grid_limits, 
            self.num_gridpoints) # make grids and corresponding spacing
        self.grid_centers = (grid_limits[:,0] + grid_limits[:,1]) / 2.
        self.R = vectorize_aperiodic_3D_distance(*self.XYZ, *self.grid_centers) # make full R matrix [n, n, n]
        self.Uext_R_arr = reference_posit_r_array(*self.XYZ, 
                                              reference_posits = solute_positions) # make grid rs for V_ext
        
        if self.interp_grid: # make a 1D grid for the Rs on which to interpolate
            self.R_in_1D = jnp.linspace(0., self.R[0,0,0], self.num_gridpoints)
        else:
            self.R_in_1D = None
        
    
    def make_n_fns(self, n_model_kwargs, key, **unused_kwargs):
        args = (jnp.zeros(1)[..., jnp.newaxis])
        mod_n_model_kwargs = overwrite_kwargs_to_dict(n_model_kwargs, 
            bounds = jnp.array([[0, self.r_cut]]), output_dimension=1)
        model = GaussianBasisMLP(**mod_n_model_kwargs)
        untrained_params = model.init(key, *args)
        def un_fn(r, n_kwargs):
            un = model.apply(n_kwargs, jnp.array([r]))[0]
            un = un * cosine_cutoff(r, self.r_cut) if self.use_cutoff else un
            return un
        def n_fn(un, Uext, **unused_kwargs):
            return self.n0 * jnp.exp(-Uext / self.kT - un)
        def dndr_fn(un, Uext, dundr, dUextdr):
            n = n_fn(un, Uext)
            dndr = -(dUextdr / self.kT + dundr)
            return n * dndr
        
        self.un_fn = un_fn
        self.n = n_fn
        self.grad_n = dndr_fn
        self.default_n_kwargs = untrained_params
    
    # n and dndr utilities
    def un(self, n_kwargs):
        if self.interp_grid:
            out = interpolate_on_ndarray(self.R_in_1D, self.R, self.un_fn, (0,None), (n_kwargs,))
        else:    
            vmapped_un = vmap_aux_fn_on_leading_grid(self.un_fn, (0,None))
            out = vmapped_un(self.R, n_kwargs)
        return out
    
    def grad_un(self, n_kwargs):
        if self.interp_grid:
            out = interpolate_on_ndarray(self.R_in_1D, self.R, jax.grad(self.un_fn), (0,None), (n_kwargs,))
        else:
            vmapped_grad_un = vmap_aux_fn_on_leading_grid(jax.grad(self.un_fn), (0,None))
            out = vmapped_grad_un(self.R, n_kwargs)
        return out
    
    def valgrad_un(self, n_kwargs):
        if self.interp_grid:
            out1 = interpolate_on_ndarray(self.R_in_1D, self.R, self.un_fn, (0,None), (n_kwargs,))
            out2 = interpolate_on_ndarray(self.R_in_1D, self.R, jax.grad(self.un_fn), (0,None), (n_kwargs,))
            out = (out1, out2)
        else:  
            vmapped_valgrad_un = vmap_aux_fn_on_leading_grid(jax.value_and_grad(self.un_fn), (0,None))
            out = vmapped_valgrad_un(self.R, n_kwargs)
        return out
    
    # Uext and dUextdr utilities
    def Uext(self, Uext_kwargs, **unused_kwargs) -> typing.Callable:
        # note: should incorporate the `self.interp_grid` into these three functions, too.
        spatial_arr_Uext = vmap_aux_fn_on_leading_grid(self.Uext_fn, (0,None))
        # vmap along all solute posits and `Uext_kwargs` leading axis;
        # gives [num_solute_posits, num_gridpoints, num_gridpoints, num_gridpoints]
        mapped_Uext_on_reference_posits = jax.vmap(spatial_arr_Uext, in_axes=(0,0))
        return mapped_Uext_on_reference_posits(self.Uext_R_arr, Uext_kwargs).sum(axis=0)
    
    def gradUext(self, Uext_kwargs, **unused_kwargs):
        spatial_arr_Uext = vmap_aux_fn_on_leading_grid(jax.grad(self.Uext_fn), (0,None))
        # vmap along all solute posits and `Uext_kwargs` leading axis;
        # gives [num_solute_posits, num_gridpoints, num_gridpoints, num_gridpoints]
        mapped_Uext_on_reference_posits = jax.vmap(spatial_arr_Uext, in_axes=(0,0))
        return mapped_Uext_on_reference_posits(self.Uext_R_arr, Uext_kwargs).sum(axis=0)
    
    def valgradUext(self, Uext_kwargs, **unused_kwargs):
        spatial_arr_Uext = vmap_aux_fn_on_leading_grid(jax.value_and_grad(self.Uext_fn), (0,None))
        # vmap along all solute posits and `Uext_kwargs` leading axis;
        # gives [num_solute_posits, num_gridpoints, num_gridpoints, num_gridpoints]
        mapped_Uext_on_reference_posits = jax.vmap(spatial_arr_Uext, in_axes=(0,0))
        out_tuple = mapped_Uext_on_reference_posits(self.Uext_R_arr, Uext_kwargs)
        return jnp.sum(out_tuple[0], axis=0), jnp.sum(out_tuple[1], axis=0)
        
    
    # c_kernel utilities
    def c_kernel(self, c_kwargs, **unused_kwargs):
        _c_fn = vmap_aux_fn_on_leading_grid(self.c_fn, (0,None))
        return _c_fn(self.R, c_kwargs)
    
    def dcdr_kernel(self, c_kwargs, **unused_kwargs):
        _c_fn = vmap_aux_fn_on_leading_grid(jax.grad(self.c_fn), (0,None))
        return _c_fn(self.R, c_kwargs)
        
    
    def dFexcdn(self, ns, c_kernel, **unused_kwargs):
        return self.dFexcdn_fn(ns, c_kernel, self.kT, self.n0, *self.dxdydz)
    
    def grad_dFexcdn(self, ns, dcdr_kernel, **unused_kwargs):
        return self.dFexcdn_fn(ns, dcdr_kernel, self.kT, self.n0, *self.dxdydz)
        
    
    def dFiddn(self, ns, **unused_kwargs):
        _dFidsdn = dFiddn_fn(ns, self.n0, self.kT)
        return _dFidsdn
    
    def n_grad_dFiddn(self, ns, grad_ns, **unused_kwargs):
        """cannot directly compute the gradient of dFiddn because it is inv prop to n;
        this can lead to instability if n is small; instead, multiply by n to cancel"""
        _n_grad_dFidsdn = self.kT * grad_ns
        return _n_grad_dFidsdn
        
    
    def _dFsdn(self, n_kwargs, c_kernel, Uexts, **unused_kwargs):
        uns = self.un(n_kwargs)
        ns = self.n(uns, Uexts)
        dFidsdn = self.dFiddn(ns)
        dFexcsdn = self.dFexcdn(ns, c_kernel)
        return (ns, 
                jnp.vstack([dFidsdn[jnp.newaxis, ...], Uexts[jnp.newaxis, ...], dFexcsdn[jnp.newaxis, ...]])
               )
    
    def _n_grad_dFsdn(self, n_kwargs, c_kernel, dcdr_kernel, Uexts, grad_Uexts, **unused_kwargs):
        """cannot directly compute grad """
        uns, grad_uns = self.valgrad_un(n_kwargs)
        ns = self.n(uns, Uexts)
        grad_ns = self.grad_n(uns, Uexts, grad_uns, grad_Uexts)
        
        n_grad_dFidsdn = self.n_grad_dFiddn(ns, grad_ns)
        n_grad_dFexcsdn = ns * self.grad_dFexcdn(ns, dcdr_kernel)
        n_grad_dFextsdn = ns * grad_Uexts
        
        return (ns,
                grad_ns, 
                jnp.vstack([n_grad_dFidsdn[jnp.newaxis, ...], n_grad_dFextsdn[jnp.newaxis, ...], n_grad_dFexcsdn[jnp.newaxis, ...]])
               )
    
    def dFsdn(self, *args, **kwargs):
        raise NotImplementedError()
    
    def dFsdn_loss(self, *args, **kwargs):
        raise NotImplementedError()
    
    def n_grad_dFsdn(self, *args, **kwargs):
        raise NotImplementedError
    
    def n_grad_dFsdn_loss(self, *args, **kwargs):
        raise NotImplementedError
        
    def analyze_dFsdn_stack(self, ns, stacked_dFsdn, **unused_kwargs):
        """experimental function to extract the z span of `self.R`, `ns`, and `dFsdn`"""
        central_gridpoint = self.num_gridpoints // 2
        rs = self.R[central_gridpoint, central_gridpoint, :]
        ns = ns[central_gridpoint, central_gridpoint, :]
        out_dFsdn = stacked_dFsdn[:,central_gridpoint, central_gridpoint, :]
        return rs, ns, out_dFsdn
        
    


class StaticDFT(BaseDFT):
    """DFT that only optimises over `n_kwargs` with initialized `Uext_kwargs`, `c_kwargs`"""
    def __init__(self, 
                 default_c_kwargs: KWARG_DICT, 
                 default_Uext_kwargs: KWARG_DICT, 
                 allow_force_match: bool, 
                 c_fn: AUX_FN,
                 Uext_fn: AUX_FN = Uext_lj,
                 grid_limits: jax.Array = jnp.array([-1., 1.]),
                 num_gridpoints: int = 200,
                 solute_positions: jax.Array = jnp.array([[0., 0., 0.]]),
                 n0: float = DEFAULT_N0,
                 kT: float = DEFAULT_KT,
                 n_model_kwargs: KWARG_DICT = DEFAULT_MODEL_KWARG_DICT,
                 key: jax.Array = jax.random.PRNGKey(42),
                 periodic_Fexc: bool = False,
                 use_cutoff: bool=True,
                 interp_grid: bool=False,
                 **kwargs):
        super_init_args = locals()
        super_init_args.update(kwargs)
        super_init_args = overwrite_kwargs_to_dict(super_init_args, removals=['kwargs', 'self'])
        super().__init__(**super_init_args)
        self.default_c_kwargs = default_c_kwargs
        self.default_Uext_kwargs = default_Uext_kwargs
        self.allow_force_match = allow_force_match
        
        # initialize grid of Uext, and c kernel
        self.make_defaults(**kwargs)
    
    def make_defaults(self, **unused_kwargs):
        self.default_c_kernel = self.c_kernel(self.default_c_kwargs)
        if self.allow_force_match:
            self.default_Uexts, self.default_grad_Uexts = self.valgradUext(self.default_Uext_kwargs)
            self.default_dcdr_kernel = self.dcdr_kernel(self.default_c_kwargs)
        else:
            self.default_Uexts, self.default_grad_Uexts = self.Uext(self.default_Uext_kwargs), None
            self.default_dcdr_kernel = None
    
    def dFsdn(self, n_kwargs, **kwargs) -> jax.Array:
        return self._dFsdn(n_kwargs, self.default_c_kernel, self.default_Uexts, **kwargs)
    
    def dFsdn_loss(self, n_kwargs, **kwargs):
        ns, stack1 = self.dFsdn(n_kwargs, **kwargs)
        summed_stack1 = jnp.sum(stack1, axis=0)
        stack1_loss = jnp.sum(summed_stack1**2) / jnp.prod(jnp.array(summed_stack1.shape))
        return stack1_loss
    
    def n_grad_dFsdn(self, n_kwargs, **kwargs):
        if self.allow_force_match:
            return self._n_grad_dFsdn(n_kwargs, self.default_c_kernel, 
                                    self.default_dcdr_kernel, self.default_Uexts, self.default_grad_Uexts, **kwargs)
        else:
            raise NotImplementedError(f"`allow_force_match` must be set to True for this")
    
    def n_grad_dFsdn_loss(self, n_kwargs, **kwargs):
        if self.allow_force_match:
            ns, grad_ns, stack1 = self.n_grad_dFsdn(n_kwargs, **kwargs)
            summed_stack1 = jnp.sum(stack1, axis=0)
            stack1_loss = jnp.sum(summed_stack1**2) / jnp.prod(jnp.array(summed_stack1.shape))
            return stack1_loss
        else:
            raise NotImplementedError(f"`allow_force_match` must be set to True for this") 

class DynamicUextDFT(StaticDFT):
    """DFT that allows for joint optimization of ```Uext_kwargs`"""
    def __init__(self, 
             default_c_kwargs: KWARG_DICT, 
             allow_force_match: bool, 
             c_fn: AUX_FN,
             Uext_fn: AUX_FN = Uext_lj,
             grid_limits: jax.Array = jnp.array([-1., 1.]),
             num_gridpoints: int = 200,
             solute_positions: jax.Array = jnp.array([[0., 0., 0.]]),
             n0: float = DEFAULT_N0,
             kT: float = DEFAULT_KT,
             n_model_kwargs: KWARG_DICT = DEFAULT_MODEL_KWARG_DICT,
             key: jax.Array = jax.random.PRNGKey(42),
             periodic_Fexc: bool = False,
             use_cutoff: bool=True,
             interp_grid: bool=False,
             **kwargs):
        super_init_args = locals()
        super_init_args.update(kwargs)
        super_init_args = overwrite_kwargs_to_dict(super_init_args, removals=['kwargs', 'self'], default_Uext_kwargs=None)
        self.default_c_kwargs = default_c_kwargs
        self.allow_force_match = allow_force_match
        super().__init__(**super_init_args)
    
    def make_defaults(self, **unused_kwargs):
        # same as `StaticDFT` except there are no `default_Uext_kwargs`
        self.default_c_kernel = self.c_kernel(self.default_c_kwargs)
        if self.allow_force_match:
            self.default_dcdr_kernel = self.dcdr_kernel(self.default_c_kwargs)
        else:
            self.default_dcdr_kernel = None
    
    def dFsdn(self, n_kwargs, Uext_kwargs, **kwargs) -> jax.Array:
        Uexts = self.Uext(Uext_kwargs)
        return self._dFsdn(n_kwargs, self.default_c_kernel, Uexts, **kwargs)
    
    def dFsdn_loss(self, n_kwargs, Uext_kwargs, **kwargs):
        mod_kwargs = overwrite_kwargs_to_dict(locals(), ['self', 'kwargs'], **kwargs)
        return super().dFsdn_loss(**mod_kwargs)
    
    def n_grad_dFsdn(self, n_kwargs, Uext_kwargs, **kwargs):
        Uexts, grad_Uexts = self.valgradUext(Uext_kwargs)
        if self.allow_force_match:
            return self._n_grad_dFsdn(n_kwargs, self.default_c_kernel, 
                                    self.default_dcdr_kernel, Uexts, grad_Uexts)
        else:
            raise NotImplementedError(f"`allow_force_match` must be set to True for this")
    
    def n_grad_dFsdn_loss(self, n_kwargs, Uext_kwargs, **kwargs):
        mod_kwargs = overwrite_kwargs_to_dict(locals(), ['self', 'kwargs'], **kwargs)
        return super().n_grad_dFsdn_loss(**mod_kwargs)
    

# now for the pde stuff
def causality_valgrad_loss(kwargs: KWARG_DICT, ts: jax.Array, eps: float, valgrad_loss: AUX_FN):
    """scan along all timepoints and return a cumulative/weighted loss and grad"""
    def scan_fn(cumulative_loss, t):
        val, grad = valgrad_loss(t, kwargs)
        weight = jnp.exp(-eps * cumulative_loss)
        return (cumulative_losses + val), (weight * val, jax.tree_util.tree_map(lambda x: x*weight, grad))
    return jax.lax.scan(scan_fn, 0., ts)
         

class BaseDDFT(DynamicUextDFT):
    """base of the dynamical density functional theory;
    This will do a few things:
        1. make a time (and position) dependent `self.n` and `self.grad_n` provided initial conditions
        2. make a time (and position) dependent `self.v` and `self.grad_v` provided initial conditions
    """
    def __init__(self,
             # novel kwargs
             dynamic_n_model_kwargs,
             control_model_kwargs,
             m: float,
             friction_coeff: float,
             num_v_gridpoints: int,
             num_v_projections: int,
             ts: jax.Array,
             Uext_kwarg_mod_fn: AUX_FN,
             scaled_v_lim: int,
             n_t0_is_uniform: bool,
            
             # super kwargs; remove `allow_force_match` since it must be `True`
             default_c_kwargs: KWARG_DICT, 
             c_fn: AUX_FN,
             Uext_fn: AUX_FN = Uext_lj,
             grid_limits: jax.Array = jnp.array([-1., 1.]),
             num_r_gridpoints: int = 100,
             solute_positions: jax.Array = jnp.array([[0., 0., 0.]]),
             n0: float = DEFAULT_N0,
             kT: float = DEFAULT_KT,
             n_model_kwargs: KWARG_DICT = DEFAULT_MODEL_KWARG_DICT,
             key: jax.Array = jax.random.PRNGKey(42),
             periodic_Fexc: bool = False,
             use_cutoff: bool=True,
             interp_grid: bool=False,
             **kwargs):
        super_init_args = locals()
        super_init_args.update(kwargs)
        super_init_args = overwrite_kwargs_to_dict(super_init_args, 
                                                   removals=['kwargs', 'self'], 
                                                   allow_force_match = True,
                                                   num_gridpoints = num_r_gridpoints)
        self.num_r_gridpoints = num_r_gridpoints # for clarity and reduce redundancy
        self.m = m
        self.kT = kT # this is defined in `super.__init__` but we need before
        self.friction_coeff = friction_coeff
        assert num_v_projections % 2 == 1, f"number of v projections must be odd to accommodate a value at 0."
        self.num_v_gridpoints = num_v_gridpoints
        self.num_v_projections = num_v_projections
        self.scaled_v_lim = scaled_v_lim
        self.n_t0_is_uniform = n_t0_is_uniform
        self.ts = ts
        self.T = ts[-1]
        self.Uext_kwarg_mod_fn = Uext_kwarg_mod_fn
        self.D = self.friction_coeff * self.kT / self.m # diffusion constant
        super().__init__(**super_init_args)
        
        # initialize Uext_control
        self.make_control(control_model_kwargs, key, **kwargs) # we are duplicating the key here, maybe fix
        
        # initialize dynamic n and v fns
        self.make_dynamic_n_fns(dynamic_n_model_kwargs, key, **kwargs)
    
    def make_defaults(self, **kwargs):
        # do `super`, but also add the central radial rs
        super().make_defaults(**kwargs)
        
        # handle grid rs
        central_gridpoint = self.num_r_gridpoints // 2
        rs = self.R[central_gridpoint, central_gridpoint, central_gridpoint:]
        
        self.central_gridpoint = central_gridpoint
        self.rs = rs
        
        # handle grid vs
        self.v_std_dev = jnp.sqrt(self.kT / self.m)
        v_lim = self.scaled_v_lim * self.v_std_dev
        self.v_bin_edges = jnp.linspace(-v_lim, v_lim, self.num_v_gridpoints)
        self.vs = r_midpoints(self.v_bin_edges)
        self.v_means = jnp.linspace(self.vs[0], self.vs[-1], self.num_v_projections)
        self.v_projection_mid_idx = int(len(self.v_means) / 2)
        self.v_volumes = jnp.abs(dVs_from_radial_bin_edges(self.v_bin_edges)) # for sphericals
        self.v_cut = self.vs[-1]
        
        # make rv grid
        self.grid_rs = jnp.repeat(self.rs[..., jnp.newaxis], len(self.vs), axis=-1)
        self.grid_vs = jnp.repeat(self.vs[jnp.newaxis, ...], len(self.rs), axis=0)
        
        
    # make time dependent Uext; it is assumed to recover `Uext` at t=0
    # write a test for that.
    def time_dep_Uext(self, r, t, Uext_kwarg_mod_kwargs):
        mod_kwargs = self.Uext_kwarg_mod_fn(t, Uext_kwarg_mod_kwargs)
        return jnp.sum(jax.vmap(self.Uext_fn, in_axes=(None, 0))(r, mod_kwargs))
    
    def make_control(self, control_model_kwargs, key, **unused_kwargs):
        args = (jnp.zeros(2)[..., jnp.newaxis],)
        mod_control_model_kwargs = overwrite_kwargs_to_dict(control_model_kwargs, 
            bounds = jnp.array([[0, self.r_cut], [0, self.T]]), output_dimension=1)
        model = GaussianBasisMLP(**mod_control_model_kwargs)
        untrained_params = model.init(key, *args)
        def u_control_fn(r, t, control_kwargs, **unused_kwargs):
            return model.apply(control_kwargs, jnp.array([r, t]))[0]
        def control_fn(r, t, control_kwargs, Uext_kwarg_mod_kwargs):
            u_control = u_control_fn(r, t, control_kwargs)
            u_control = u_control * cosine_cutoff(r, self.r_cut) if self.use_cutoff else u_control
            u_control = u_control * t / self.T
            return u_control + self.time_dep_Uext(r, t, Uext_kwarg_mod_kwargs)
        
        self.u_control_fn = u_control_fn
        self.control_fn = control_fn
        self.default_control_kwargs = untrained_params
    
    def pretrain_control(self, tol, **unused_kwargs):
        """pretrain the control fn to make the control close to zero"""
        def loss_fn(control_kwargs):
            outs = jax.vmap(jax.vmap(self.u_control_fn, in_axes=(None, 0, None)), in_axes=(0,None,None))(
                self.rs, self.ts, control_kwargs).flatten()
            return jnp.sum(outs**2) / len(outs)
        loss_valgrad = jax.jit(jax.value_and_grad(loss_fn))
        def outer_loss(control_kwargs):
            val, grad = loss_valgrad(control_kwargs)
            clear_output(wait=True)
            display(f"pretrain control loss: {val}")
            return val, grad
        
        solver = jaxopt.ScipyMinimize(fun = outer_loss, value_and_grad=True, jit=False, 
                                      method='BFGS', maxiter=99999, tol=tol)
        res = solver.run(self.default_control_kwargs)
        res_params = res.params
        self.pretrained_control_kwargs = res_params
        
    def make_dynamic_n_fns(self, dynamic_n_model_kwargs, key, **unused_kwargs):
        # first the nrt model
        nrt_args = (jnp.zeros(2)[..., jnp.newaxis],) # 2 for (r, t)
        mod_nrt_model_kwargs = overwrite_kwargs_to_dict(
            dynamic_n_model_kwargs, 
            bounds = jnp.array([[0, self.r_cut], 
                                [0, self.T]]), 
            output_dimension=1)
        nrt_model = GaussianBasisPINN(**mod_nrt_model_kwargs)
        nrt_untrained_params = nrt_model.init(key, *nrt_args)
        
        # second the n_v model
        nv_args = (jnp.zeros(3)[..., jnp.newaxis],) # 2 for (r, t, v_mu)
        mod_nvt_model_kwargs = overwrite_kwargs_to_dict(
            dynamic_n_model_kwargs, 
            bounds = jnp.array([[0, self.r_cut], 
                                [0, self.T],
                                [self.v_means[0], self.v_means[-1]] # implicitly goes from -vlim to vlim
                               ]), 
            output_dimension=2)
        nvt_model = GaussianBasisPINN(**mod_nvt_model_kwargs)
        nvt_sigma_pert_fn = lambda r, t, v_mu, nvt_params: nvt_model.apply(nvt_params, jnp.array([r, t, v_mu]))[0]
        nvt_weight_pert_fn = lambda r, t, v_mu, nvt_params: nvt_model.apply(nvt_params, jnp.array([r, t, v_mu]))[1]
        nvt_untrained_params = nvt_model.init(key, *nv_args)
        
        def nrt_fn(r, t, nrt_params, Uext_kwarg_mod_kwargs, n0_kwargs, **unused_kwargs):
            u = nrt_model.apply(nrt_params, jnp.array([r, t]))[0]
            u = u * cosine_cutoff(r, self.r_cut) if self.use_cutoff else u
            u = u * t / self.T
            Uext_t0 = self.time_dep_Uext(r, 0, Uext_kwarg_mod_kwargs) # Uext at t0
            un_t0 = 0. if self.n_t0_is_uniform else self.un_fn(r, n0_kwargs)
            n_t0 = self.n(un_t0, Uext_t0) # only radial density
            return n_t0 * jnp.exp(-u)
        
        def nvt_fn(r, v, t, sigma_perts, weight_perts, **unused_kwargs):
            # sigma modifiers
            init_sigma_scales = jnp.ones_like(self.v_means)
            v_sigmas = self.v_std_dev * (init_sigma_scales + sigma_perts*t/self.T)**2 + 1e-6
            
            # weight modifiers
            init_v_weights = jnp.zeros_like(self.v_means)
            init_v_weights = init_v_weights.at[self.v_projection_mid_idx].set(1.)
            v_weights = (init_v_weights + weight_perts*t/self.T)**2 + 1e-6
            v_log_weights = jnp.log(v_weights)
            
            _nv = prob_gaussian_mixture(v, self.v_means, v_sigmas, v_log_weights)
            return _nv
        
        def nrvt_fn(r, v, t, nrt_params, nvt_params, Uext_kwarg_mod_kwargs, n0_kwargs, **unused_kwargs):
            _nrt = nrt_fn(r, t, nrt_params, Uext_kwarg_mod_kwargs, n0_kwargs)
            
            sigma_perts = jax.vmap(nvt_sigma_pert_fn, in_axes=(None,None,0,None))(r, t, self.v_means, nvt_params)
            weight_perts = jax.vmap(nvt_weight_pert_fn, in_axes=(None,None,0,None))(r, t, self.v_means, nvt_params)
            _nv = nvt_fn(r, v, t, sigma_perts, weight_perts)
            return _nrt * _nv
        
        def valgrad_nrvt_fn(r, t, nrt_params, nvt_params, Uext_kwarg_mod_kwargs, n0_kwargs, **unused_kwargs):
            """density at all velocities given r, t, and params"""
            # gives outputs of shape 1
            _nrt, (dnrtdr, dnrtdt) = jax.value_and_grad(nrt_fn, argnums=(0,1))(r, t, nrt_params, Uext_kwarg_mod_kwargs, n0_kwargs)
            
            # gives outputs of shape self.v_means
            sigma_perts, (dsigma_pertsdr, dsigma_pertsdt) = jax.vmap(
                jax.value_and_grad(nvt_sigma_pert_fn, argnums=(0,1)),
                in_axes=(None, None, 0,None))(r, t, self.v_means, nvt_params)
            weight_perts, (dweight_pertsdr, dweight_pertsdt) = jax.vmap(
                jax.value_and_grad(nvt_weight_pert_fn, argnums=(0,1)),
                in_axes=(None, None, 0,None))(r, t, self.v_means, nvt_params)
            
            # gives outputs of shape self.vs
            nvs, (dnvsdr, dnvsdv, dnvsdt, dnvsdsigma_perts, dnvsdweight_perts) = jax.vmap(
                jax.value_and_grad(nvt_fn, argnums=(0, 1, 2, 3, 4)), 
                in_axes=(None,0,None,None,None))(r, self.vs, t, sigma_perts, weight_perts)
            
            # ns
            ns = _nrt * nvs
            
            # dndr
            dndr1 = dnrtdr * nvs
            dndr2 = _nrt * (dnvsdr + jnp.dot(dnvsdsigma_perts, dsigma_pertsdr) + jnp.dot(dnvsdweight_perts, dweight_pertsdr))
            dnsdr = dndr1 + dndr2
            
            # dndt
            dndt1 = dnrtdt * nvs
            dndt2 = _nrt * (dnvsdt + jnp.dot(dnvsdsigma_perts, dsigma_pertsdt) + jnp.dot(dnvsdweight_perts, dweight_pertsdt))
            dnsdt = dndt1 + dndt2
            
            # dndv
            dnsdv = _nrt * dnvsdv
            
            # d2ndv2
            d2nvsdv2_fn = jax.vmap(jax.grad(jax.grad(nvt_fn, argnums=1), argnums=1), in_axes=(None,0, None,None,None))
            d2nvsdv2s = d2nvsdv2_fn(r, self.vs, t, sigma_perts, weight_perts)
            d2nsdv2 = _nrt * d2nvsdv2s
            
            return ns, dnsdr, dnsdv, dnsdt, d2nsdv2  
        
        # vmapped over rs, then vs, static all other kwargs
        self.nvt_sigma_pert_fn = nvt_sigma_pert_fn
        self.nvt_weight_pert_fn = nvt_weight_pert_fn
        self.nrt_fn = nrt_fn
        self.nvt_fn = nvt_fn
        self.nrvt_fn = nrvt_fn
        self.valgrad_nrvt_fn = valgrad_nrvt_fn
        
        self.nrt_untrained_params = nrt_untrained_params
        self.nvt_untrained_params = nvt_untrained_params   
    
    def pretrain_loss(self, n0_kwargs, Uext_kwarg_mod_kwargs):
        """return the pretraining loss. it is necessary to first optimize the n0_kwargs at t=0 with the 
        `Uext_kwarg_mod_kwargs` at t=0;
        
        typically, it is not advisable to also train the `Uext_kwarg_mod_kwargs` since this defines `Uext` params
        for all times; this is considered a predefined static variable"""
        _t = 0.
        mod_Uext_kwargs = self.Uext_kwarg_mod_fn(_t, Uext_kwarg_mod_kwargs)
        _dFsdn_loss = self.dFsdn_loss(n0_kwargs, mod_Uext_kwargs)
        return _dFsdn_loss
    
    def pretrain_n(self, Uext_kwarg_mod_kwargs, tol, **unused_kwargs):
        """update in place the `default_nv_kwargs` and `default_n_kwargs` with trained params;
        also, return the densities and velocities at `self.rs` and `t=0` for validation"""
        dft_loss_fn = jax.jit(lambda _kwargs: self.pretrain_loss(**_kwargs, Uext_kwarg_mod_kwargs=Uext_kwarg_mod_kwargs))
        
        def dft_callback(x):
            loss = dft_loss_fn(x)
            clear_output(wait=True)
            display(f"loss: {loss}")
        
        in_kwargs = {'n0_kwargs': self.default_n_kwargs}
        solver = jaxopt.ScipyMinimize(fun = dft_loss_fn, method='BFGS', maxiter=99999, callback=dft_callback, tol=tol)
        res = solver.run(in_kwargs)
        res_params = res.params
        self.pretrained_n_kwargs = res_params['n0_kwargs']
        ns_t0 = self.ns_t(self.rs, self.vs, 0., 
                         self.default_dynamic_n_kwargs, Uext_kwarg_mod_kwargs, self.pretrained_n_kwargs)
        return res, ns_t0
        
    
    def pde_t_residual(self, t, nrt_params, nvt_params, 
                       control_kwargs, Uext_kwarg_mod_kwargs, 
                       n0_kwargs, **unused_kwargs):
        """compute the residual of the Klein-Kramers Field Equation at a specified time over the 2D grid of radial rs
        and radial velocities;
        NOTE: at t=0, the residual is not guaranteed to be zero for arbitrary `dynamic_n_kwargs` since dnsdt will not be zero;
        Furthermore, one can expect the f_excs to also be nonzero owing to integration errors over v, however, they should be
        small.
        """
        ns, dnsdr, dnsdv, dnsdt, d2nsdv2 = jax.vmap(self.valgrad_nrvt_fn, 
                                                    in_axes=(0,None,None,None,None,None)
                                                   )(self.rs, t, nrt_params, nvt_params, 
                                                     Uext_kwarg_mod_kwargs, n0_kwargs)
        
        # F_exts first                      
        F_exts = -jax.vmap(jax.grad(self.control_fn), in_axes=(0,None,None,None))(self.rs, t, 
                                                                  control_kwargs, Uext_kwarg_mod_kwargs) # presumed unit'd
        F_exts = jnp.repeat(F_exts[..., jnp.newaxis], len(self.vs), axis=-1)
        
        # F_excs next
        F_ints = jnp.zeros_like(F_exts)
#         ns_on_r = self.marginalize_over_vs(ns)
#         interp_Rs = jnp.interp(self.R.flatten(), self.rs, ns_on_r)
#         grid_ns = interp_Rs.reshape(self.R.shape)
#         F_ints= - self.grad_dFexcdn(grid_ns, self.default_dcdr_kernel)[self.central_gridpoint, 
#                                                                        self.central_gridpoint, self.central_gridpoint:]
#         F_ints = jnp.repeat(F_ints[..., jnp.newaxis], len(self.vs), axis=-1)
        
        # compute div(J_r)
        div_Jr = dnsdr * self.grid_vs
        
        # compute div(J_v) in sphericals (radial)
        div_Jv = ((F_ints + F_exts) / self.m) * dnsdv - self.friction_coeff * (ns + dnsdv * self.grid_vs) - self.D * d2nsdv2
        
        return (dnsdt + div_Jr + div_Jv), (ns, dnsdr, dnsdv, dnsdt, d2nsdv2, F_exts, F_ints, div_Jr, div_Jv)
    
    def pde_t_residual_loss(self, t, nrt_params, nvt_params, control_kwargs, 
                            Uext_kwarg_mod_kwargs, n0_kwargs, **unused_kwargs):
        """compute loss of `pde_t_residual`"""
        losses = self.pde_t_residual(t, nrt_params, nvt_params, control_kwargs, 
                                     Uext_kwarg_mod_kwargs, n0_kwargs)[0].flatten()
        return jnp.sum(jnp.square(losses)) / len(losses)
    
    def inner_pde_t_residual_loss(self, t, dynamic_kwargs, static_kwargs, **unused_kwargs):
        """compute the `pde_t_residual_loss` as a function that is refactored into `t`, `dynamic_kwargs`, `static_kwargs`"""
        in_static_kwargs = {key: val for key, val in static_kwargs.items()}
        in_static_kwargs.update(dynamic_kwargs)
        return self.pde_t_residual_loss(t, **in_static_kwargs)
    
    def valgrad_inner_pde_t_residual_loss(self, t, dynamic_kwargs, static_kwargs, **unused_kwargs):
        """compute the value and gradient of `pde_t_residual_loss` w.r.t. `dynamic_kwargs`"""
        return jax.value_and_grad(self.inner_pde_t_residual_loss, argnums=1)(t, dynamic_kwargs, static_kwargs)
    
    def vmapped_valgrad_inner_pde_t_residual_loss(self, ts, dynamic_kwargs, static_kwargs, **unused_kwargs):
        """vmap the value and gradient given by `valgrad_inner_pde_t_residual_loss`"""
        return jax.vmap(self.valgrad_inner_pde_t_residual_loss, in_axes=(0,None,None))(ts, 
                                                                                       dynamic_kwargs, static_kwargs)
    
    def vmapped_valgrad_causality_loss(self, ts, dynamic_kwargs, static_kwargs, eps, **unused_kwargs):
        losses, grad_losses = self.vmapped_valgrad_inner_pde_t_residual_loss(ts, dynamic_kwargs, static_kwargs)
        log_weights = jnp.concatenate([jnp.zeros(1), -eps * jnp.cumsum(losses)[:-1]])
        weights = jnp.exp(log_weights)
        total_loss = jnp.sum(losses * weights) / len(losses)
        total_loss_grad = jax.tree_util.tree_map(lambda x: jnp.sum(x * weights[..., jnp.newaxis], axis=0), grad_losses)
        return total_loss, total_loss_grad
                                                  
    @functools.partial(jax.jit, static_argnums=0)
    def iterable_valgrad_pde_causality_loss_t(self, t, dynamic_kwargs, static_kwargs, 
                                            weight_cum_loss, weight_cum_grad, eps, **unused_kwargs):
        weight = jnp.exp(-eps * weight_cum_loss)
        loss, grad = self.valgrad_inner_pde_t_residual_loss(t, dynamic_kwargs, static_kwargs)
        weight_grad = jax.tree_util.tree_map(lambda x: x * weight, grad)
        weight_cum_grad = jax.tree_util.tree_map(lambda x, y: x + y, weight_cum_grad, weight_grad)
        weight_cum_loss += weight * loss
        return loss, weight_cum_loss, weight_cum_grad, weight
    
    # no jit
    def iterable_valgrad_pde_causality_loss(self, ts, dynamic_kwargs, 
                                            static_kwargs, eps, verbose, **unused_kwargs):
        T = len(ts)
        losses = []
        weight_cum_loss, weight_cum_grad = 0., jax.tree_util.tree_map(lambda x: x * 0., dynamic_kwargs)
        for t in ts:
            loss, weight_cum_loss, weight_cum_grad, weight = self.iterable_valgrad_pde_causality_loss_t(
                    t, dynamic_kwargs, static_kwargs, weight_cum_loss, weight_cum_grad, eps
            )
            
            losses.append(loss)
            if verbose: # display the auxiliary data
                clear_output(wait=True)
                display(f"t: {t}", f"loss_at_t: {loss}", 
                        f"weight_cum_loss: {weight_cum_loss}", 
                        f"weight_at_t: {weight}")
    
        return weight_cum_loss / T, jax.tree_util.tree_map(lambda x: x / T, weight_cum_grad), losses
        
    