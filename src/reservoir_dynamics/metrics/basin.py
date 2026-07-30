"""有限標本から吸引域の到達確率と不確実性を推定する。"""

import math
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from statistics import NormalDist


@dataclass(frozen=True, slots=True)
class BasinStabilityEstimate:
    """一つの発見済みアトラクタに対するbasin stability推定値。"""

    label: Hashable
    count: int
    sample_size: int
    probability: float
    lower_confidence_bound: float
    upper_confidence_bound: float
    confidence: float


def estimate_basin_stability(
    attractor_labels: Sequence[Hashable],
    *,
    confidence: float = 0.95,
) -> tuple[BasinStabilityEstimate, ...]:
    """初期条件標本の終端ラベルからbasin stabilityを推定する。

    各標本を、明示された初期条件分布からの独立なBernoulli試行とみなし、
    Wald区間より境界標本で安定なWilson score intervalを返す。
    """

    labels = tuple(attractor_labels)
    if not labels:
        raise ValueError("アトラクタラベルは1つ以上必要です")
    if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("信頼水準は0より大きく1より小さい必要があります")

    counts: dict[Hashable, int] = {}
    try:
        for label in labels:
            counts[label] = counts.get(label, 0) + 1
    except TypeError as error:
        raise ValueError("アトラクタラベルはhash可能である必要があります") from error

    sample_size = len(labels)
    normal_quantile = NormalDist().inv_cdf(0.5 + confidence / 2.0)

    return tuple(
        _build_estimate(
            label=label,
            count=count,
            sample_size=sample_size,
            confidence=confidence,
            normal_quantile=normal_quantile,
        )
        for label, count in counts.items()
    )


def _build_estimate(
    *,
    label: Hashable,
    count: int,
    sample_size: int,
    confidence: float,
    normal_quantile: float,
) -> BasinStabilityEstimate:
    probability = count / sample_size
    quantile_squared = normal_quantile * normal_quantile
    denominator = 1.0 + quantile_squared / sample_size
    center = (
        probability + quantile_squared / (2.0 * sample_size)
    ) / denominator
    half_width = (
        normal_quantile
        * math.sqrt(
            probability * (1.0 - probability) / sample_size
            + quantile_squared / (4.0 * sample_size * sample_size)
        )
        / denominator
    )

    return BasinStabilityEstimate(
        label=label,
        count=count,
        sample_size=sample_size,
        probability=probability,
        lower_confidence_bound=max(0.0, center - half_width),
        upper_confidence_bound=min(1.0, center + half_width),
        confidence=confidence,
    )
