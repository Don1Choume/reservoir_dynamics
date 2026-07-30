"""数値実験と照合できる理論境界。"""

from reservoir_dynamics.theory.bistable_margin import (
    BistableTanhCertificate,
    bistable_tanh_certificate,
    positive_bistable_fixed_point,
)
from reservoir_dynamics.theory.contraction import (
    iterated_lipschitz_distance_bound,
)
from reservoir_dynamics.theory.core_protection import (
    core_deviation_bound_curve,
)
from reservoir_dynamics.theory.orthant_box import (
    RobustOrthantBoxCertificate,
    robust_orthant_box_certificate,
)

__all__ = [
    "BistableTanhCertificate",
    "RobustOrthantBoxCertificate",
    "bistable_tanh_certificate",
    "core_deviation_bound_curve",
    "iterated_lipschitz_distance_bound",
    "positive_bistable_fixed_point",
    "robust_orthant_box_certificate",
]
