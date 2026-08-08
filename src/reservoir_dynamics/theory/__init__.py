"""数値実験と照合できる理論境界。"""

from reservoir_dynamics.theory.bistable_margin import (
    BistableTanhCertificate,
    bistable_tanh_certificate,
    positive_bistable_fixed_point,
)
from reservoir_dynamics.theory.contraction import (
    iterated_lipschitz_distance_bound,
)
from reservoir_dynamics.theory.component_coupling import (
    DirectionalMarginSummary,
    directional_bridge_norms,
    summarize_directional_margins,
)
from reservoir_dynamics.theory.core_protection import (
    core_deviation_bound_curve,
)
from reservoir_dynamics.theory.multicomponent_coupling import (
    CertificateEnumerationComplexity,
    MultiComponentMarginSummary,
    certificate_enumeration_complexity,
    component_inbound_load_matrix,
    summarize_multicomponent_margins,
)
from reservoir_dynamics.theory.orthant_box import (
    RobustOrthantBoxCertificate,
    robust_orthant_box_certificate,
)
from reservoir_dynamics.theory.orthant_rectangle import (
    OrthantRectangleCertificate,
    matrix_infinity_norm_difference,
    orthant_rectangle_certificate,
)
from reservoir_dynamics.theory.spatial_core_protection import (
    BistableCoordinateProtection,
    bistable_coordinate_protection,
    energy_matched_global_weights,
    matrix_frobenius_distance_squared,
    row_gated_matrix,
    time_varying_core_deviation_bound,
)

__all__ = [
    "BistableTanhCertificate",
    "BistableCoordinateProtection",
    "DirectionalMarginSummary",
    "CertificateEnumerationComplexity",
    "MultiComponentMarginSummary",
    "OrthantRectangleCertificate",
    "RobustOrthantBoxCertificate",
    "bistable_tanh_certificate",
    "bistable_coordinate_protection",
    "core_deviation_bound_curve",
    "certificate_enumeration_complexity",
    "component_inbound_load_matrix",
    "directional_bridge_norms",
    "energy_matched_global_weights",
    "iterated_lipschitz_distance_bound",
    "matrix_infinity_norm_difference",
    "matrix_frobenius_distance_squared",
    "orthant_rectangle_certificate",
    "positive_bistable_fixed_point",
    "robust_orthant_box_certificate",
    "row_gated_matrix",
    "summarize_directional_margins",
    "summarize_multicomponent_margins",
    "time_varying_core_deviation_bound",
]
