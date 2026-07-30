"""seed標本に対する再現可能なpercentile bootstrap区間。"""

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BootstrapMeanInterval:
    """標本平均とpercentile bootstrap信頼区間。"""

    estimate: float
    lower: float
    upper: float
    confidence_level: float
    resamples: int


def bootstrap_mean_interval(
    values: Sequence[float],
    *,
    confidence_level: float = 0.95,
    resamples: int = 2_000,
    random_seed: int = 20_260_730,
) -> BootstrapMeanInterval:
    """seedを再標本化し、平均のpercentile区間を返す。"""

    normalized_values = tuple(float(value) for value in values)
    if len(normalized_values) < 2:
        raise ValueError("bootstrapには2要素以上の標本が必要です")
    if any(not math.isfinite(value) for value in normalized_values):
        raise ValueError("標本値はすべて有限である必要があります")
    if (
        not math.isfinite(confidence_level)
        or confidence_level <= 0.0
        or confidence_level >= 1.0
    ):
        raise ValueError("confidence_levelは0と1の間にしてください")
    if (
        not isinstance(resamples, int)
        or isinstance(resamples, bool)
        or resamples < 1
    ):
        raise ValueError("resamplesは1以上の整数にしてください")
    if not isinstance(random_seed, int) or isinstance(random_seed, bool):
        raise ValueError("random_seedは整数にしてください")

    random_generator = random.Random(random_seed)
    sample_size = len(normalized_values)
    bootstrap_means = sorted(
        math.fsum(
            normalized_values[random_generator.randrange(sample_size)]
            for _ in range(sample_size)
        )
        / sample_size
        for _ in range(resamples)
    )
    tail_probability = (1.0 - confidence_level) / 2.0
    return BootstrapMeanInterval(
        estimate=math.fsum(normalized_values) / sample_size,
        lower=_linear_quantile(bootstrap_means, tail_probability),
        upper=_linear_quantile(bootstrap_means, 1.0 - tail_probability),
        confidence_level=confidence_level,
        resamples=resamples,
    )


def _linear_quantile(sorted_values: list[float], probability: float) -> float:
    scaled_index = probability * (len(sorted_values) - 1)
    lower_index = math.floor(scaled_index)
    upper_index = math.ceil(scaled_index)
    if lower_index == upper_index:
        return sorted_values[lower_index]
    interpolation_weight = scaled_index - lower_index
    return (
        sorted_values[lower_index] * (1.0 - interpolation_weight)
        + sorted_values[upper_index] * interpolation_weight
    )
