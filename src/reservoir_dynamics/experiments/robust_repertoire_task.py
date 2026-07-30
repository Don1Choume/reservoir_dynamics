"""robust repertoire curveと外乱下符号記憶taskの診断。"""

from __future__ import annotations

import itertools
import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass

from reservoir_dynamics.experiments.orthant_margin_sweep import _step
from reservoir_dynamics.experiments.recurrent_weight_families import (
    RECURRENT_WEIGHT_FAMILIES,
    RecurrentWeightFamily,
    build_recurrent_weights,
)
from reservoir_dynamics.theory.orthant_box import (
    Matrix,
    RobustOrthantBoxCertificate,
    Signs,
    robust_orthant_box_certificate,
)

STUDY_ID = "ROBUST-REPERTOIRE-TASK-DIAGNOSTICS"


@dataclass(frozen=True, slots=True)
class RobustRepertoireTaskPoint:
    """一network・一外乱強度のcertificateと符号記憶性能。"""

    trial_seed: int
    network_family: RecurrentWeightFamily
    coupling_gain: float
    disturbance_bound: float
    raw_attractor_count: int
    certified_robust_count: int
    certified_robust_fraction: float
    mean_uniform_disturbance_margin: float
    task_retention: float
    guarantee_gap: float


@dataclass(frozen=True, slots=True)
class RobustTaskPredictiveSummary:
    """外乱強度ごとの関連、held-out予測、保証較正。"""

    disturbance_bound: float
    condition_count: int
    spearman_certified_fraction: float
    spearman_mean_margin: float
    certified_fraction_test_mae: float
    mean_margin_test_mae: float
    coupling_gain_test_mae: float
    raw_count_test_mae: float
    guarantee_violation_count: int
    mean_guarantee_gap: float


@dataclass(frozen=True, slots=True)
class RobustRepertoireTaskDiagnostics:
    """pilotまたは確認へ使う全点と予測診断。"""

    study_id: str
    trial_seeds: tuple[int, ...]
    network_family: RecurrentWeightFamily
    dimension: int
    diagonal_gain: float
    coupling_gains: tuple[float, ...]
    disturbance_bounds: tuple[float, ...]
    task_steps: int
    autonomous_steps: int
    training_seeds: tuple[int, ...]
    testing_seeds: tuple[int, ...]
    points: tuple[RobustRepertoireTaskPoint, ...]
    predictive_summaries: tuple[RobustTaskPredictiveSummary, ...]


def run_robust_repertoire_task_diagnostics(
    *,
    trial_seeds: tuple[int, ...] = tuple(range(401, 431)),
    network_family: RecurrentWeightFamily = "dense_symmetric",
    dimension: int = 4,
    diagonal_gain: float = 1.5,
    coupling_gains: tuple[float, ...] = (0.04, 0.05, 0.06, 0.07),
    disturbance_bounds: tuple[float, ...] = (0.08, 0.12, 0.16, 0.20),
    task_steps: int = 100,
    autonomous_steps: int = 500,
    convergence_tolerance: float = 1e-9,
    training_seed_count: int = 20,
) -> RobustRepertoireTaskDiagnostics:
    """全orthant・全一定corner外乱を列挙しheld-out予測を測る。"""

    _validate_configuration(
        trial_seeds=trial_seeds,
        network_family=network_family,
        dimension=dimension,
        diagonal_gain=diagonal_gain,
        coupling_gains=coupling_gains,
        disturbance_bounds=disturbance_bounds,
        task_steps=task_steps,
        autonomous_steps=autonomous_steps,
        convergence_tolerance=convergence_tolerance,
        training_seed_count=training_seed_count,
    )
    sign_patterns = tuple(
        itertools.product((-1, 1), repeat=dimension)
    )
    disturbance_directions = sign_patterns
    points: list[RobustRepertoireTaskPoint] = []
    for trial_seed in trial_seeds:
        for coupling_gain in coupling_gains:
            recurrent_weights = build_recurrent_weights(
                network_family=network_family,
                dimension=dimension,
                diagonal_gain=diagonal_gain,
                coupling_gain=coupling_gain,
                trial_seed=trial_seed,
            )
            fixed_points = tuple(
                _find_fixed_point(
                    recurrent_weights=recurrent_weights,
                    attractor_signs=signs,
                    steps=autonomous_steps,
                    convergence_tolerance=convergence_tolerance,
                )
                for signs in sign_patterns
            )
            certificates = tuple(
                robust_orthant_box_certificate(
                    recurrent_weights=recurrent_weights,
                    attractor_signs=signs,
                )
                for signs in sign_patterns
            )
            raw_attractor_count = sum(
                fixed_point[1] for fixed_point in fixed_points
            )
            mean_margin = math.fsum(
                certificate.maximum_uniform_disturbance
                for certificate in certificates
            ) / len(certificates)
            points.extend(
                _evaluate_disturbance_bound(
                    trial_seed=trial_seed,
                    network_family=network_family,
                    coupling_gain=coupling_gain,
                    disturbance_bound=disturbance_bound,
                    recurrent_weights=recurrent_weights,
                    sign_patterns=sign_patterns,
                    disturbance_directions=disturbance_directions,
                    fixed_points=fixed_points,
                    certificates=certificates,
                    raw_attractor_count=raw_attractor_count,
                    mean_margin=mean_margin,
                    task_steps=task_steps,
                )
                for disturbance_bound in disturbance_bounds
            )
    point_tuple = tuple(points)
    training_seeds = trial_seeds[:training_seed_count]
    testing_seeds = trial_seeds[training_seed_count:]
    summaries = tuple(
        _summarize_prediction(
            points=tuple(
                point
                for point in point_tuple
                if point.disturbance_bound == disturbance_bound
            ),
            training_seeds=training_seeds,
            testing_seeds=testing_seeds,
        )
        for disturbance_bound in disturbance_bounds
    )
    return RobustRepertoireTaskDiagnostics(
        study_id=STUDY_ID,
        trial_seeds=trial_seeds,
        network_family=network_family,
        dimension=dimension,
        diagonal_gain=diagonal_gain,
        coupling_gains=coupling_gains,
        disturbance_bounds=disturbance_bounds,
        task_steps=task_steps,
        autonomous_steps=autonomous_steps,
        training_seeds=training_seeds,
        testing_seeds=testing_seeds,
        points=point_tuple,
        predictive_summaries=summaries,
    )


def _evaluate_disturbance_bound(
    *,
    trial_seed: int,
    network_family: RecurrentWeightFamily,
    coupling_gain: float,
    disturbance_bound: float,
    recurrent_weights: Matrix,
    sign_patterns: tuple[Signs, ...],
    disturbance_directions: tuple[Signs, ...],
    fixed_points: tuple[tuple[tuple[float, ...], bool], ...],
    certificates: tuple[RobustOrthantBoxCertificate, ...],
    raw_attractor_count: int,
    mean_margin: float,
    task_steps: int,
) -> RobustRepertoireTaskPoint:
    certified_count = sum(
        certificate.maximum_uniform_disturbance
        >= disturbance_bound - 1e-12
        and fixed_point_retained
        and all(
            sign * value >= certificate.invariant_boundary - 1e-12
            for sign, value in zip(
                signs,
                fixed_point,
                strict=True,
            )
        )
        for signs, fixed_point, fixed_point_retained, certificate in (
            (
                signs,
                fixed_point_result[0],
                fixed_point_result[1],
                certificate,
            )
            for signs, fixed_point_result, certificate in zip(
                sign_patterns,
                fixed_points,
                certificates,
                strict=True,
            )
        )
    )
    successes = 0
    total_challenges = len(sign_patterns) * len(disturbance_directions)
    for signs, fixed_point_result in zip(
        sign_patterns,
        fixed_points,
        strict=True,
    ):
        fixed_point, fixed_point_retained = fixed_point_result
        if not fixed_point_retained:
            continue
        for direction in disturbance_directions:
            disturbance = tuple(
                disturbance_bound * value for value in direction
            )
            successes += int(
                _retains_sign_memory(
                    recurrent_weights=recurrent_weights,
                    initial_state=fixed_point,
                    attractor_signs=signs,
                    disturbance=disturbance,
                    steps=task_steps,
                )
            )
    certified_fraction = certified_count / len(sign_patterns)
    task_retention = successes / total_challenges
    return RobustRepertoireTaskPoint(
        trial_seed=trial_seed,
        network_family=network_family,
        coupling_gain=coupling_gain,
        disturbance_bound=disturbance_bound,
        raw_attractor_count=raw_attractor_count,
        certified_robust_count=certified_count,
        certified_robust_fraction=certified_fraction,
        mean_uniform_disturbance_margin=mean_margin,
        task_retention=task_retention,
        guarantee_gap=task_retention - certified_fraction,
    )


def _find_fixed_point(
    *,
    recurrent_weights: Matrix,
    attractor_signs: Signs,
    steps: int,
    convergence_tolerance: float,
) -> tuple[tuple[float, ...], bool]:
    state = tuple(0.9 * sign for sign in attractor_signs)
    retained = True
    residual = math.inf
    for _ in range(steps):
        next_state = _step(
            recurrent_weights,
            state,
            (0.0,) * len(state),
        )
        residual = max(
            abs(next_value - value)
            for next_value, value in zip(next_state, state, strict=True)
        )
        state = next_state
        retained = retained and all(
            sign * value > 0.0
            for sign, value in zip(attractor_signs, state, strict=True)
        )
    return state, retained and residual <= convergence_tolerance


def _retains_sign_memory(
    *,
    recurrent_weights: Matrix,
    initial_state: tuple[float, ...],
    attractor_signs: Signs,
    disturbance: tuple[float, ...],
    steps: int,
) -> bool:
    state = initial_state
    for _ in range(steps):
        state = _step(recurrent_weights, state, disturbance)
        if any(
            sign * value <= 0.0
            for sign, value in zip(attractor_signs, state, strict=True)
        ):
            return False
    return True


def _summarize_prediction(
    *,
    points: tuple[RobustRepertoireTaskPoint, ...],
    training_seeds: tuple[int, ...],
    testing_seeds: tuple[int, ...],
) -> RobustTaskPredictiveSummary:
    task_values = tuple(point.task_retention for point in points)
    certified_values = tuple(
        point.certified_robust_fraction for point in points
    )
    margin_values = tuple(
        point.mean_uniform_disturbance_margin for point in points
    )
    return RobustTaskPredictiveSummary(
        disturbance_bound=points[0].disturbance_bound,
        condition_count=len(points),
        spearman_certified_fraction=_spearman(
            certified_values,
            task_values,
        ),
        spearman_mean_margin=_spearman(margin_values, task_values),
        certified_fraction_test_mae=_held_out_mae(
            points=points,
            training_seeds=training_seeds,
            testing_seeds=testing_seeds,
            feature=lambda point: point.certified_robust_fraction,
        ),
        mean_margin_test_mae=_held_out_mae(
            points=points,
            training_seeds=training_seeds,
            testing_seeds=testing_seeds,
            feature=lambda point: point.mean_uniform_disturbance_margin,
        ),
        coupling_gain_test_mae=_held_out_mae(
            points=points,
            training_seeds=training_seeds,
            testing_seeds=testing_seeds,
            feature=lambda point: point.coupling_gain,
        ),
        raw_count_test_mae=_held_out_mae(
            points=points,
            training_seeds=training_seeds,
            testing_seeds=testing_seeds,
            feature=lambda point: float(point.raw_attractor_count),
        ),
        guarantee_violation_count=sum(
            point.guarantee_gap < -1e-12 for point in points
        ),
        mean_guarantee_gap=math.fsum(
            point.guarantee_gap for point in points
        )
        / len(points),
    )


def _held_out_mae(
    *,
    points: tuple[RobustRepertoireTaskPoint, ...],
    training_seeds: tuple[int, ...],
    testing_seeds: tuple[int, ...],
    feature: Callable[[RobustRepertoireTaskPoint], float],
) -> float:
    training_points = tuple(
        point for point in points if point.trial_seed in training_seeds
    )
    testing_points = tuple(
        point for point in points if point.trial_seed in testing_seeds
    )
    training_features = tuple(feature(point) for point in training_points)
    training_targets = tuple(
        point.task_retention for point in training_points
    )
    mean_feature = math.fsum(training_features) / len(training_features)
    mean_target = math.fsum(training_targets) / len(training_targets)
    feature_variance = math.fsum(
        (value - mean_feature) ** 2 for value in training_features
    )
    if feature_variance <= 1e-15:
        slope = 0.0
    else:
        slope = math.fsum(
            (feature_value - mean_feature)
            * (target - mean_target)
            for feature_value, target in zip(
                training_features,
                training_targets,
                strict=True,
            )
        ) / feature_variance
    intercept = mean_target - slope * mean_feature
    return math.fsum(
        abs(
            point.task_retention
            - (intercept + slope * feature(point))
        )
        for point in testing_points
    ) / len(testing_points)


def _spearman(
    first: tuple[float, ...],
    second: tuple[float, ...],
) -> float:
    first_ranks = _average_ranks(first)
    second_ranks = _average_ranks(second)
    first_mean = math.fsum(first_ranks) / len(first_ranks)
    second_mean = math.fsum(second_ranks) / len(second_ranks)
    numerator = math.fsum(
        (first_rank - first_mean) * (second_rank - second_mean)
        for first_rank, second_rank in zip(
            first_ranks,
            second_ranks,
            strict=True,
        )
    )
    denominator = math.sqrt(
        math.fsum((rank - first_mean) ** 2 for rank in first_ranks)
        * math.fsum((rank - second_mean) ** 2 for rank in second_ranks)
    )
    if denominator <= 1e-15:
        return 0.0
    return numerator / denominator


def _average_ranks(values: tuple[float, ...]) -> tuple[float, ...]:
    ordered = sorted(
        enumerate(values),
        key=lambda item: item[1],
    )
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while (
            end < len(ordered)
            and ordered[end][1] == ordered[start][1]
        ):
            end += 1
        average_rank = (start + 1 + end) / 2.0
        for ordered_index in range(start, end):
            ranks[ordered[ordered_index][0]] = average_rank
        start = end
    return tuple(ranks)


def _validate_configuration(
    *,
    trial_seeds: tuple[int, ...],
    network_family: str,
    dimension: int,
    diagonal_gain: float,
    coupling_gains: tuple[float, ...],
    disturbance_bounds: tuple[float, ...],
    task_steps: int,
    autonomous_steps: int,
    convergence_tolerance: float,
    training_seed_count: int,
) -> None:
    if len(trial_seeds) < 4 or len(set(trial_seeds)) != len(trial_seeds):
        raise ValueError("trial_seedsは重複しない4個以上にしてください")
    if any(
        not isinstance(seed, int) or isinstance(seed, bool)
        for seed in trial_seeds
    ):
        raise ValueError("trial_seedsは整数にしてください")
    if network_family not in RECURRENT_WEIGHT_FAMILIES:
        raise ValueError(
            f"network_familyは{RECURRENT_WEIGHT_FAMILIES}から選んでください"
        )
    if (
        not isinstance(training_seed_count, int)
        or isinstance(training_seed_count, bool)
        or training_seed_count < 2
        or training_seed_count > len(trial_seeds) - 2
    ):
        raise ValueError(
            "training_seed_countは2以上でtest seedを2個以上残してください"
        )
    if (
        not isinstance(dimension, int)
        or isinstance(dimension, bool)
        or dimension < 2
        or dimension > 6
    ):
        raise ValueError("dimensionは2以上6以下の整数にしてください")
    if not math.isfinite(diagonal_gain) or diagonal_gain <= 1.0:
        raise ValueError("diagonal_gainは1より大きい有限値にしてください")
    _validate_increasing_axis(
        coupling_gains,
        "coupling_gains",
        allow_zero=True,
    )
    _validate_increasing_axis(
        disturbance_bounds,
        "disturbance_bounds",
        allow_zero=False,
    )
    _validate_positive_integer(task_steps, "task_steps")
    _validate_positive_integer(autonomous_steps, "autonomous_steps")
    if (
        not math.isfinite(convergence_tolerance)
        or convergence_tolerance <= 0.0
    ):
        raise ValueError("convergence_toleranceは有限の正値にしてください")


def _validate_increasing_axis(
    values: tuple[float, ...],
    value_name: str,
    *,
    allow_zero: bool,
) -> None:
    lower_bound = 0.0 if allow_zero else 1e-300
    if (
        not values
        or any(
            not math.isfinite(value) or value < lower_bound
            for value in values
        )
        or any(
            first >= second
            for first, second in zip(values, values[1:])
        )
    ):
        qualifier = "非負" if allow_zero else "正"
        raise ValueError(
            f"{value_name}は厳密昇順の有限{qualifier}値にしてください"
        )


def _validate_positive_integer(value: int, value_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{value_name}は1以上の整数にしてください")


def main() -> None:
    """既定pilotをJSONとして出力する。"""

    print(
        json.dumps(
            asdict(run_robust_repertoire_task_diagnostics()),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
