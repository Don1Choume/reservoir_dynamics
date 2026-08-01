"""標準化と正則化を一体化した小規模ridge回帰。"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StandardizedRidgeModel:
    """学習時の標準化統計を保持する不変な線形予測器。"""

    intercept: float
    coefficients: tuple[float, ...]
    feature_means: tuple[float, ...]
    feature_scales: tuple[float, ...]
    penalty: float

    def predict(
        self,
        features: Sequence[float],
        *,
        clip_to_unit_interval: bool = False,
    ) -> float:
        """学習時と同じ標準化を適用して予測する。"""

        normalized_features = tuple(float(value) for value in features)
        if len(normalized_features) != len(self.coefficients):
            raise ValueError("予測時の特徴量数が学習時と一致しません")
        if any(
            not math.isfinite(value) for value in normalized_features
        ):
            raise ValueError("予測特徴量はすべて有限である必要があります")
        if not isinstance(clip_to_unit_interval, bool):
            raise ValueError("clip_to_unit_intervalはboolにしてください")

        prediction = self.intercept + math.fsum(
            coefficient * (value - mean) / scale
            for coefficient, value, mean, scale in zip(
                self.coefficients,
                normalized_features,
                self.feature_means,
                self.feature_scales,
                strict=True,
            )
        )
        if not clip_to_unit_interval:
            return prediction
        return min(1.0, max(0.0, prediction))


def fit_standardized_ridge(
    feature_rows: Sequence[Sequence[float]],
    targets: Sequence[float],
    *,
    penalty: float = 1e-6,
) -> StandardizedRidgeModel:
    """特徴量を標準化し、切片を罰しないridge回帰をfitする。"""

    normalized_rows = tuple(
        tuple(float(value) for value in row) for row in feature_rows
    )
    normalized_targets = tuple(float(target) for target in targets)
    if not normalized_rows:
        raise ValueError("feature_rowsは1行以上必要です")
    feature_count = len(normalized_rows[0])
    if feature_count < 1:
        raise ValueError("feature_rowsは1列以上必要です")
    if any(len(row) != feature_count for row in normalized_rows):
        raise ValueError("feature_rowsの列数は全行で一致させてください")
    if len(normalized_targets) != len(normalized_rows):
        raise ValueError("target数はfeature_rowsの行数と一致させてください")
    if any(
        not math.isfinite(value)
        for row in normalized_rows
        for value in row
    ) or any(not math.isfinite(target) for target in normalized_targets):
        raise ValueError("特徴量とtargetはすべて有限である必要があります")
    normalized_penalty = float(penalty)
    if (
        isinstance(penalty, bool)
        or not math.isfinite(normalized_penalty)
        or normalized_penalty < 0.0
    ):
        raise ValueError("penaltyは0以上の有限値にしてください")

    sample_count = len(normalized_rows)
    feature_means = tuple(
        math.fsum(row[index] for row in normalized_rows) / sample_count
        for index in range(feature_count)
    )
    raw_scales = tuple(
        math.sqrt(
            math.fsum(
                (row[index] - feature_means[index]) ** 2
                for row in normalized_rows
            )
            / sample_count
        )
        for index in range(feature_count)
    )
    feature_scales = tuple(
        scale if scale > 0.0 else 1.0 for scale in raw_scales
    )
    standardized_rows = tuple(
        tuple(
            (value - feature_means[index]) / feature_scales[index]
            for index, value in enumerate(row)
        )
        for row in normalized_rows
    )
    target_mean = math.fsum(normalized_targets) / sample_count
    centered_targets = tuple(
        target - target_mean for target in normalized_targets
    )
    gram_matrix = [
        [
            math.fsum(
                row[row_index] * row[column_index]
                for row in standardized_rows
            )
            + (
                normalized_penalty * sample_count
                if row_index == column_index
                else 0.0
            )
            for column_index in range(feature_count)
        ]
        for row_index in range(feature_count)
    ]
    right_hand_side = [
        math.fsum(
            row[index] * target
            for row, target in zip(
                standardized_rows,
                centered_targets,
                strict=True,
            )
        )
        for index in range(feature_count)
    ]
    coefficients = _solve_linear_system(
        gram_matrix,
        right_hand_side,
    )
    return StandardizedRidgeModel(
        intercept=target_mean,
        coefficients=coefficients,
        feature_means=feature_means,
        feature_scales=feature_scales,
        penalty=normalized_penalty,
    )


def _solve_linear_system(
    matrix: list[list[float]],
    right_hand_side: list[float],
) -> tuple[float, ...]:
    """部分pivot付きGauss消去で小規模な正規方程式を解く。"""

    dimension = len(right_hand_side)
    augmented = [
        [*matrix[row_index], right_hand_side[row_index]]
        for row_index in range(dimension)
    ]
    scale = max(
        (abs(value) for row in matrix for value in row),
        default=0.0,
    )
    pivot_tolerance = 1e-12 * max(1.0, scale)
    for pivot_index in range(dimension):
        best_row = max(
            range(pivot_index, dimension),
            key=lambda row_index: abs(
                augmented[row_index][pivot_index]
            ),
        )
        if abs(augmented[best_row][pivot_index]) <= pivot_tolerance:
            raise ValueError(
                "正規方程式が特異です。正のpenaltyを指定してください"
            )
        if best_row != pivot_index:
            augmented[pivot_index], augmented[best_row] = (
                augmented[best_row],
                augmented[pivot_index],
            )
        pivot = augmented[pivot_index][pivot_index]
        augmented[pivot_index] = [
            value / pivot for value in augmented[pivot_index]
        ]
        for row_index in range(dimension):
            if row_index == pivot_index:
                continue
            elimination_factor = augmented[row_index][pivot_index]
            if elimination_factor == 0.0:
                continue
            augmented[row_index] = [
                current - elimination_factor * pivot_value
                for current, pivot_value in zip(
                    augmented[row_index],
                    augmented[pivot_index],
                    strict=True,
                )
            ]
    return tuple(augmented[index][-1] for index in range(dimension))
