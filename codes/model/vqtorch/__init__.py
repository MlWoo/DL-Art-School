###
# borrowed from https://github.com/minyoungg/vqtorch
# [1] Straightening Out the Straight-Through Estimator:
# Overcoming Optimization Challenges in Vector Quantized Networks, Huh et al. ICML2023
###
from . import utils  # noqa
from .affine import AffineTransform  # noqa
from .gvq import GroupVectorQuant  # noqa
from .math_fns import *  # noqa
from .rvq import ResidualVectorQuant  # noqa
from .vq import VectorQuant  # noqa
from .vq_base import _VQBaseLayer  # noqa
