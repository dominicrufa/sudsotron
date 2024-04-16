"""Write some modules in flax.linen"""

from sudsotron.nn.utils import gaussian_basis_projections

@dataclass(frozen=True)
class MLPParams:
    features: int = 4
    num_layers: int = 2
    output_dimension: int = 1
    nonlinearity: typing.Callable = jnp.tanh
    activate_output: bool = False


@dataclass(frozen=True)
class GaussianBasisParams:
    bounds: jax.Array = field(default_factory = lambda: jnp.array([[-1., 1.]]))
    projection_dim: int = 32


@dataclass(frozen=True)
class GaussianBasisMLPParams(MLPParams, GaussianBasisParams):
    pass


@dataclass(frozen=True)
class GaussianBasisPINNParams(GaussianBasisMLPParams):
    num_outer_layers: int = 1


class MLP(nn.Module):
    """simple multilayer perceptron"""
    features: int # hidden layer features
    num_layers: int # number of layers
    output_dimension: int # output dimension
    nonlinearity: typing.Callable[float, float] = jnp.tanh
    activate_output: bool=False # whether to activate output
    
    @nn.compact
    def __call__(self, x: jax.Array, **unused_kwargs) -> jax.Array:
        # NOTE: may want to extract `Dense` unused_kwargs out in future.
        for layer in range(self.num_layers):
            feats = self.features if layer < self.num_layers-1 else self.output_dimension
            x = nn.Dense(features = feats)(x)
            if layer != self.num_layers - 1 and not self.activate_output:
                x = self.nonlinearity(x)
        return x
    
class GaussianBasis(nn.Module):
    """compute a gaussian basis projection of `x`"""
    bounds: jax.Array # [N,2], N is dim of `x`
    projection_dim: int # number of projections for each entry of `x`

    @nn.compact
    def __call__(self, x: jax.Array, **unused_kwargs) -> jax.Array:
        out = gaussian_basis_projections(x, self.bounds, self.projection_dim).flatten()
        return out

class GaussianBasisMLP(nn.Module):
    """project `x` to a gaussian basis and then pass through an MLP"""
    bounds: jax.Array 
    projection_dim: int
    features: int
    num_layers: int
    output_dimension: int
    nonlinearity: typing.Callable[float, float] = jnp.tanh
    activate_output: bool=False
    
    @nn.compact
    def __call__(self, x, **unused_kwargs) -> jax.Array:
        x = GaussianBasis(bounds=self.bounds, projection_dim=self.projection_dim)(x)
        x = MLP(self.features, self.num_layers, self.output_dimension, self.nonlinearity, self.activate_output)(x)
        return x

class BasePINNHidden(nn.Module):
    """hidden layer of PINN"""
    U: jax.Array # [N_U,]
    V: jax.Array # [N_V,]
    features: int # MLP hidden features
    num_layers: int # num MLP layers
    output_dimension: int
    nonlinearity: typing.Callable[float, float] = jnp.tanh
    
    @nn.compact
    def __call__(self, H: jax.Array, **unused_kwargs):
        Z = MLP(self.features, self.num_layers, self.output_dimension, self.nonlinearity, True)(H)
        H = (1. - Z) * self.U + Z * self.V
        return H

class BasePINN(nn.Module):
    """Eqs. 43-47 (Fig. 9) of https://arxiv.org/pdf/2001.04536.pdf'"""
    features: int # num features of initial MLP
    num_layers: int # num layers of initial MLP
    output_dimension: int # final output dim
    num_outer_layers: int=1 # number of PINN layers
    nonlinearity: typing.Callable[float, float] = jnp.tanh
    # activate_output: bool=True; this is always True because a nonactivated output is at end
    
    @nn.compact
    def __call__(self, x: jax.Array, **unused_kwargs) -> jax.Array:
        UVH = MLP(self.features, self.num_layers, 3*self.features, self.nonlinearity, True)(x)
        U, V, H = UVH[:self.features], UVH[self.features:2*self.features], UVH[2*self.features:]
        for k in range(self.num_outer_layers):
            H = BasePINNHidden(U, V, self.features, self.num_layers, self.features, self.nonlinearity)(H)
        return nn.Dense(features=self.output_dimension)(H)

class GaussianBasisPINN(nn.Module):
    """a `GaussianBasis` projection followed by a full `BasePINN`"""
    # Gaussian Basis kwargs
    bounds: jax.Array
    projection_dim: int
    
    # BasePINN kwargs
    features: int
    num_layers: int
    output_dimension: int
    num_outer_layers: int
    nonlinearity: typing.Callable[float, float] = jnp.tanh
    
    @nn.compact
    def __call__(self, x: jax.Array, **unused_kwargs) -> jax.Array:
        x = GaussianBasis(self.bounds, self.projection_dim)(x)
        x = BasePINN(self.features, self.num_layers, 
                     self.output_dimension, self.nonlinearity, self.num_outer_layers)(x)
        return x