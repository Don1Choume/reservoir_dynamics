"""既存network familyの符号座標共役class監査。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math

from reservoir_dynamics.experiments.cross_family_robust_task_confirmation import (
    DEFAULT_FAMILY_SPECIFICATIONS,
)
from reservoir_dynamics.experiments.family_holdout_confirmation import (
    PREREGISTERED_CONFIRMATION_SEEDS as FAMILY_HOLDOUT_SEEDS,
)
from reservoir_dynamics.experiments.isolated_modular_family_confirmation import (
    PREREGISTERED_CONFIRMATION_SEEDS as MODULAR_CONFIRMATION_SEEDS,
    PREREGISTERED_EXTERNAL_FAMILY_SPECIFICATION,
)
from reservoir_dynamics.experiments.recurrent_weight_families import (
    RECURRENT_WEIGHT_FAMILIES,
    RecurrentWeightFamily,
    build_recurrent_weights,
)
from reservoir_dynamics.metrics.structural_equivalence import (
    audit_signed_coordinate_conjugacy,
)


@dataclass(frozen=True, slots=True)
class StructuralAuditSpecification:
    """一つのfamilyで監査するgainとseed集合。"""

    network_family: RecurrentWeightFamily
    coupling_gains: tuple[float, ...]
    trial_seeds: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class NetworkFamilyGainStructureAudit:
    """一つのfamily・gain内で得た有効構造class。"""

    network_family: RecurrentWeightFamily
    coupling_gain: float
    trial_seeds: tuple[int, ...]
    raw_network_count: int
    effective_class_count: int
    effective_fraction: float
    representative_seeds: tuple[int, ...]
    class_sizes: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class NetworkFamilyStructureAuditResult:
    """family・gain別監査を束ねた結果。"""

    dimension: int
    diagonal_gain: float
    tolerance: float
    equivalence_relation: str
    audits: tuple[NetworkFamilyGainStructureAudit, ...]

    @property
    def raw_network_count(self) -> int:
        return sum(audit.raw_network_count for audit in self.audits)

    @property
    def effective_class_count(self) -> int:
        return sum(audit.effective_class_count for audit in self.audits)

    @property
    def effective_fraction(self) -> float:
        return self.effective_class_count / self.raw_network_count


DEFAULT_STRUCTURAL_AUDIT_SPECIFICATIONS = tuple(
    StructuralAuditSpecification(
        network_family=specification.network_family,
        coupling_gains=specification.coupling_gains,
        trial_seeds=FAMILY_HOLDOUT_SEEDS,
    )
    for specification in DEFAULT_FAMILY_SPECIFICATIONS
) + (
    StructuralAuditSpecification(
        network_family=(
            PREREGISTERED_EXTERNAL_FAMILY_SPECIFICATION.network_family
        ),
        coupling_gains=(
            PREREGISTERED_EXTERNAL_FAMILY_SPECIFICATION.coupling_gains
        ),
        trial_seeds=MODULAR_CONFIRMATION_SEEDS,
    ),
)


def run_network_family_structure_audit(
    *,
    specifications: tuple[
        StructuralAuditSpecification, ...
    ] = DEFAULT_STRUCTURAL_AUDIT_SPECIFICATIONS,
    dimension: int = 4,
    diagonal_gain: float = 1.5,
    tolerance: float = 1e-12,
) -> NetworkFamilyStructureAuditResult:
    """既存実験と同じnetworkをgain内の符号共役classへ分割する。"""

    _validate_configuration(
        specifications=specifications,
        dimension=dimension,
        diagonal_gain=diagonal_gain,
        tolerance=tolerance,
    )
    audits = tuple(
        _audit_family_gain(
            specification=specification,
            coupling_gain=coupling_gain,
            dimension=dimension,
            diagonal_gain=diagonal_gain,
            tolerance=tolerance,
        )
        for specification in specifications
        for coupling_gain in specification.coupling_gains
    )
    return NetworkFamilyStructureAuditResult(
        dimension=dimension,
        diagonal_gain=diagonal_gain,
        tolerance=tolerance,
        equivalence_relation="second = D first D, D_ii in {-1, +1}",
        audits=audits,
    )


def _audit_family_gain(
    *,
    specification: StructuralAuditSpecification,
    coupling_gain: float,
    dimension: int,
    diagonal_gain: float,
    tolerance: float,
) -> NetworkFamilyGainStructureAudit:
    matrices = tuple(
        build_recurrent_weights(
            network_family=specification.network_family,
            dimension=dimension,
            diagonal_gain=diagonal_gain,
            coupling_gain=coupling_gain,
            trial_seed=trial_seed,
        )
        for trial_seed in specification.trial_seeds
    )
    audit = audit_signed_coordinate_conjugacy(
        matrices,
        tolerance=tolerance,
    )
    return NetworkFamilyGainStructureAudit(
        network_family=specification.network_family,
        coupling_gain=coupling_gain,
        trial_seeds=specification.trial_seeds,
        raw_network_count=audit.raw_network_count,
        effective_class_count=audit.effective_class_count,
        effective_fraction=audit.effective_fraction,
        representative_seeds=tuple(
            specification.trial_seeds[index]
            for index in audit.representative_indices
        ),
        class_sizes=audit.class_sizes,
    )


def _validate_configuration(
    *,
    specifications: tuple[StructuralAuditSpecification, ...],
    dimension: int,
    diagonal_gain: float,
    tolerance: float,
) -> None:
    if not specifications:
        raise ValueError("specificationsは空にできません")
    families = tuple(
        specification.network_family for specification in specifications
    )
    if len(set(families)) != len(families):
        raise ValueError("network familyは重複禁止です")
    if any(family not in RECURRENT_WEIGHT_FAMILIES for family in families):
        raise ValueError("network familyが未対応です")
    for specification in specifications:
        if (
            not specification.coupling_gains
            or len(set(specification.coupling_gains))
            != len(specification.coupling_gains)
            or any(
                not math.isfinite(gain) or gain < 0.0
                for gain in specification.coupling_gains
            )
        ):
            raise ValueError(
                "coupling_gainsは重複しない有限の非負値にしてください"
            )
        if (
            len(specification.trial_seeds) < 2
            or len(set(specification.trial_seeds))
            != len(specification.trial_seeds)
            or any(
                not isinstance(seed, int) or isinstance(seed, bool)
                for seed in specification.trial_seeds
            )
        ):
            raise ValueError(
                "trial seedは重複しない2個以上の整数にしてください"
            )
    if (
        not isinstance(dimension, int)
        or isinstance(dimension, bool)
        or dimension < 2
        or dimension > 8
    ):
        raise ValueError("dimensionは2以上8以下の整数にしてください")
    if not math.isfinite(diagonal_gain) or diagonal_gain <= 1.0:
        raise ValueError("diagonal_gainは1より大きい有限値にしてください")
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("toleranceは有限の非負値にしてください")


def _summary_payload(
    result: NetworkFamilyStructureAuditResult,
) -> dict[str, object]:
    return {
        "audit_id": "AUDIT-2026-001",
        "dimension": result.dimension,
        "diagonal_gain": result.diagonal_gain,
        "tolerance": result.tolerance,
        "equivalence_relation": result.equivalence_relation,
        "task_equivalence_scope": (
            "入力・外乱・readout・評価集合の符号変換閉性は別途監査する"
        ),
        "raw_network_count": result.raw_network_count,
        "effective_class_count": result.effective_class_count,
        "effective_fraction": result.effective_fraction,
        "audits": tuple(asdict(audit) for audit in result.audits),
    }


def main() -> None:
    """既存5 familyの構造同値性監査をJSONで出力する。"""

    print(
        json.dumps(
            _summary_payload(run_network_family_structure_audit()),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
