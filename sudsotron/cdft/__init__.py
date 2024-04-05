"""cdft init"""
import jax
from jax import numpy as jnp
import numpy as np
from dataclasses import dataclass, asdict, field
import typing
import functools

from sudsotron.cdft.constants import (
    DEFAULT_KT,
    DEFAULT_N0,
    DEFAULT_M,
    DEFAULT_R_CUT,
)

from sudsotron.cdft.dcf import (
    HNCRadialDCF,
)

@dataclass(frozen=True)
class HNCcDFT:
    radial_dcf: HNCRadialDCF
    