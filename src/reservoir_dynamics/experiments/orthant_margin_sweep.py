"""非対角tanh RNNのraw attractor数とrobust repertoireを分離する。"""

from __future__ import annotations

import itertools
import json
import math
import random
from dataclasses import asdict, dataclass

from reservoir_dynamics.metrics.bootstrap import (
    BootstrapMeanInterval,
    bootstrap_mean_interval,
)
from reservoir_dynamics.theory.orthant_box import (
    Matrix,
    RobustOrthantBoxCertificate,
    Signs,
    robust_orthant_box_certificate,
)

EXPERIMENT_ID = "EXP-2026-007"
DEFAULT_TRIAL_SEEDS = tuple(range(401, 431))
DEFAULT_COUPLING_GAINS = (0.04, 0.08)


@dataclass(frozen=True, slots=True)
class OrthantMarginPoint:
    """一network・一orthantの自律保持と外乱certificate。"""

    trial_seed: int
    coupling_gain: float
    attractor_signs: Signs
    autonomous_attractor_retained: bool
    autonomous_residual: float
    invariant_boundary: float
    raw_uniform_disturbance_margin: float
    maximum_uniform_disturbance: float
    is_certified: bool
    safe_survival_rate: float | None
    adversarial_witness_escaped: bool | None


@dataclass(frozen=True, slots=True)
class CouplingMarginSummary:
    """coupling強度ごとのseed集約。"""

    coupling_gain: float
    seed_count: int
    autonomous_repertoire_count: BootstrapMeanInterval
    certified_robust_repertoire_count: BootstrapMeanInterval
    mean_uniform_disturbance_margin: BootstrapMeanInterval
    safe_survival_rate: BootstrapMeanInterval
    adversarial_witness_escape_rate: BootstrapMeanInterval


@dataclass(frozen=True, slots=True)
class OrthantMarginContrast:
    """弱couplingから強couplingへの対応変化。"""

    weak_minus_strong_autonomous_count: BootstrapMeanInterval
    weak_minus_strong_certified_count: BootstrapMeanInterval


@dataclass(frozen=True, slots=True)
class OrthantMarginDecisions:
    """実行前に固定するraw countとrobust countの分離判定。"""

    autonomous_repertoire_preserved: bool
    robust_repertoire_separated: bool
    safe_box_invariance: bool
    adversarial_boundary_witness: bool


@dataclass(frozen=True, slots=True)
class OrthantMarginSweepResult:
    """EXP-007のspec、点、集約、判定。"""

    experiment_id: str
    trial_seeds: tuple[int, ...]
    dimension: int
    diagonal_gain: float
    coupling_gains: tuple[float, float]
    safe_disturbance_ratio: float
    witness_disturbance_ratio: float
    autonomous_steps: int
    points: tuple[OrthantMarginPoint, ...]
    coupling_summaries: tuple[CouplingMarginSummary, ...]
    contrast: OrthantMarginContrast
    decisions: OrthantMarginDecisions


def run_orthant_margin_sweep(
    *,
    trial_seeds: tuple[int, ...] = DEFAULT_TRIAL_SEEDS,
    dimension: int = 4,
    diagonal_gain: float = 1.5,
    coupling_gains: tuple[
        float, float
    ] = DEFAULT_COUPLING_GAINS,
    safe_disturbance_ratio: float = 0.9,
    witness_disturbance_ratio: float = 1.1,
    safe_trials: int = 16,
    simulation_steps: int = 100,
    autonomous_steps: int = 500,
    convergence_tolerance: float = 1e-9,
    bootstrap_confidence_level: float = 0.95,
    bootstrap_resamples: int = 2_000,
    bootstrap_seed: int = 20_260_733,
) -> OrthantMarginSweepResult:
    """matched signed networkでcouplingだけを強める30 seed介入。"""

    _validate_configuration(
        trial_seeds=trial_seeds,
        dimension=dimension,
        diagonal_gain=diagonal_gain,
        coupling_gains=coupling_gains,
        safe_disturbance_ratio=safe_disturbance_ratio,
        witness_disturbance_ratio=witness_disturbance_ratio,
        safe_trials=safe_trials,
        simulation_steps=simulation_steps,
        autonomous_steps=autonomous_steps,
        convergence_tolerance=convergence_tolerance,
        bootstrap_confidence_level=bootstrap_confidence_level,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )
    points: list[OrthantMarginPoint] = []
    sign_patterns = tuple(
        itertools.product((-1, 1), repeat=dimension)
    )
    for trial_seed in trial_seeds:
        for coupling_gain in coupling_gains:
            recurrent_weights = _signed_symmetric_weights(
                dimension=dimension,
                diagonal_gain=diagonal_gain,
                coupling_gain=coupling_gain,
                trial_seed=trial_seed,
            )
            points.extend(
                _evaluate_orthant(
                    trial_seed=trial_seed,
                    coupling_gain=coupling_gain,
                    recurrent_weights=recurrent_weights,
                    attractor_signs=signs,
                    safe_disturbance_ratio=safe_disturbance_ratio,
                    witness_disturbance_ratio=witness_disturbance_ratio,
                    safe_trials=safe_trials,
                    simulation_steps=simulation_steps,
                    autonomous_steps=autonomous_steps,
                    convergence_tolerance=convergence_tolerance,
                )
                for signs in sign_patterns
            )
    point_tuple = tuple(points)
    summaries = tuple(
        _summarize_coupling(
            points=tuple(
                point
                for point in point_tuple
                if point.coupling_gain == coupling_gain
            ),
            trial_seeds=trial_seeds,
            confidence_level=bootstrap_confidence_level,
            resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed + index * 100,
        )
        for index, coupling_gain in enumerate(coupling_gains)
    )
    contrast = _build_contrast(
        points=point_tuple,
        trial_seeds=trial_seeds,
        coupling_gains=coupling_gains,
        confidence_level=bootstrap_confidence_level,
        resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed + 10_000,
    )
    expected_repertoire = 2**dimension
    return OrthantMarginSweepResult(
        experiment_id=EXPERIMENT_ID,
        trial_seeds=trial_seeds,
        dimension=dimension,
        diagonal_gain=diagonal_gain,
        coupling_gains=coupling_gains,
        safe_disturbance_ratio=safe_disturbance_ratio,
        witness_disturbance_ratio=witness_disturbance_ratio,
        autonomous_steps=autonomous_steps,
        points=point_tuple,
        coupling_summaries=summaries,
        contrast=contrast,
        decisions=OrthantMarginDecisions(
            autonomous_repertoire_preserved=all(
                summary.autonomous_repertoire_count.lower
                >= expected_repertoire - 1e-12
                for summary in summaries
            ),
            robust_repertoire_separated=(
                contrast.weak_minus_strong_certified_count.lower > 0.0
            ),
            safe_box_invariance=all(
                point.safe_survival_rate is None
                or point.safe_survival_rate >= 1.0 - 1e-12
                for point in point_tuple
            ),
            adversarial_boundary_witness=all(
                point.adversarial_witness_escaped is None
                or point.adversarial_witness_escaped
                for point in point_tuple
            ),
        ),
    )


def _evaluate_orthant(
    *,
    trial_seed: int,
    coupling_gain: float,
    recurrent_weights: Matrix,
    attractor_signs: Signs,
    safe_disturbance_ratio: float,
    witness_disturbance_ratio: float,
    safe_trials: int,
    simulation_steps: int,
    autonomous_steps: int,
    convergence_tolerance: float,
) -> OrthantMarginPoint:
    certificate = robust_orthant_box_certificate(
        recurrent_weights=recurrent_weights,
        attractor_signs=attractor_signs,
    )
    autonomous_retained, residual = _evaluate_autonomous_attractor(
        recurrent_weights=recurrent_weights,
        attractor_signs=attractor_signs,
        steps=autonomous_steps,
        convergence_tolerance=convergence_tolerance,
    )
    safe_survival_rate: float | None = None
    witness_escaped: bool | None = None
    if certificate.is_certified:
        safe_survival_rate = _evaluate_safe_disturbances(
            recurrent_weights=recurrent_weights,
            certificate=certificate,
            trial_seed=_orthant_seed(trial_seed, attractor_signs),
            disturbance_ratio=safe_disturbance_ratio,
            trials=safe_trials,
            steps=simulation_steps,
        )
        witness_escaped = _evaluate_boundary_witness(
            recurrent_weights=recurrent_weights,
            certificate=certificate,
            disturbance_ratio=witness_disturbance_ratio,
        )
    return OrthantMarginPoint(
        trial_seed=trial_seed,
        coupling_gain=coupling_gain,
        attractor_signs=attractor_signs,
        autonomous_attractor_retained=autonomous_retained,
        autonomous_residual=residual,
        invariant_boundary=certificate.invariant_boundary,
        raw_uniform_disturbance_margin=(
            certificate.raw_uniform_disturbance_margin
        ),
        maximum_uniform_disturbance=(
            certificate.maximum_uniform_disturbance
        ),
        is_certified=certificate.is_certified,
        safe_survival_rate=safe_survival_rate,
        adversarial_witness_escaped=witness_escaped,
    )


def _evaluate_autonomous_attractor(
    *,
    recurrent_weights: Matrix,
    attractor_signs: Signs,
    steps: int,
    convergence_tolerance: float,
) -> tuple[bool, float]:
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
    return retained and residual <= convergence_tolerance, residual


def _evaluate_safe_disturbances(
    *,
    recurrent_weights: Matrix,
    certificate: RobustOrthantBoxCertificate,
    trial_seed: int,
    disturbance_ratio: float,
    trials: int,
    steps: int,
) -> float:
    random_generator = random.Random(trial_seed)
    survived_trials = 0
    signs = certificate.attractor_signs
    boundary = certificate.invariant_boundary
    disturbance_bound = (
        disturbance_ratio * certificate.maximum_uniform_disturbance
    )
    for _ in range(trials):
        state = tuple(
            sign * random_generator.uniform(boundary, 1.0)
            for sign in signs
        )
        survived = True
        for _ in range(steps):
            transformed_disturbance = tuple(
                random_generator.uniform(
                    -disturbance_bound,
                    disturbance_bound,
                )
                for _ in signs
            )
            disturbance = tuple(
                sign * value
                for sign, value in zip(
                    signs,
                    transformed_disturbance,
                    strict=True,
                )
            )
            state = _step(recurrent_weights, state, disturbance)
            survived = survived and all(
                sign * value >= boundary - 1e-12
                for sign, value in zip(signs, state, strict=True)
            )
        survived_trials += int(survived)
    return survived_trials / trials


def _evaluate_boundary_witness(
    *,
    recurrent_weights: Matrix,
    certificate: RobustOrthantBoxCertificate,
    disturbance_ratio: float,
) -> bool:
    signs = certificate.attractor_signs
    boundary = certificate.invariant_boundary
    limiting_coordinate = certificate.limiting_coordinates[0]
    transformed_row = tuple(
        signs[limiting_coordinate] * weight * signs[column_index]
        for column_index, weight in enumerate(
            recurrent_weights[limiting_coordinate]
        )
    )
    transformed_state = tuple(
        boundary if weight >= 0.0 else 1.0
        for weight in transformed_row
    )
    state = tuple(
        sign * value
        for sign, value in zip(signs, transformed_state, strict=True)
    )
    transformed_disturbance = [0.0] * len(signs)
    transformed_disturbance[limiting_coordinate] = (
        -disturbance_ratio * certificate.maximum_uniform_disturbance
    )
    disturbance = tuple(
        sign * value
        for sign, value in zip(
            signs,
            transformed_disturbance,
            strict=True,
        )
    )
    next_state = _step(recurrent_weights, state, disturbance)
    return (
        signs[limiting_coordinate] * next_state[limiting_coordinate]
        < boundary
    )


def _summarize_coupling(
    *,
    points: tuple[OrthantMarginPoint, ...],
    trial_seeds: tuple[int, ...],
    confidence_level: float,
    resamples: int,
    bootstrap_seed: int,
) -> CouplingMarginSummary:
    seed_metrics = tuple(
        _seed_metrics(
            tuple(point for point in points if point.trial_seed == seed)
        )
        for seed in trial_seeds
    )
    intervals = tuple(
        bootstrap_mean_interval(
            tuple(metrics[index] for metrics in seed_metrics),
            confidence_level=confidence_level,
            resamples=resamples,
            random_seed=bootstrap_seed + index,
        )
        for index in range(5)
    )
    return CouplingMarginSummary(
        coupling_gain=points[0].coupling_gain,
        seed_count=len(trial_seeds),
        autonomous_repertoire_count=intervals[0],
        certified_robust_repertoire_count=intervals[1],
        mean_uniform_disturbance_margin=intervals[2],
        safe_survival_rate=intervals[3],
        adversarial_witness_escape_rate=intervals[4],
    )


def _seed_metrics(
    points: tuple[OrthantMarginPoint, ...],
) -> tuple[float, float, float, float, float]:
    certified_points = tuple(point for point in points if point.is_certified)
    if not certified_points:
        raise RuntimeError("認証orthantが一つもないnetworkは集約できません")
    return (
        float(
            sum(point.autonomous_attractor_retained for point in points)
        ),
        float(sum(point.is_certified for point in points)),
        math.fsum(
            point.maximum_uniform_disturbance for point in points
        )
        / len(points),
        math.fsum(
            point.safe_survival_rate
            for point in certified_points
            if point.safe_survival_rate is not None
        )
        / len(certified_points),
        math.fsum(
            float(point.adversarial_witness_escaped)
            for point in certified_points
        )
        / len(certified_points),
    )


def _build_contrast(
    *,
    points: tuple[OrthantMarginPoint, ...],
    trial_seeds: tuple[int, ...],
    coupling_gains: tuple[float, float],
    confidence_level: float,
    resamples: int,
    bootstrap_seed: int,
) -> OrthantMarginContrast:
    weak_gain, strong_gain = coupling_gains
    paired_values: list[tuple[float, float]] = []
    for trial_seed in trial_seeds:
        weak_metrics = _seed_metrics(
            tuple(
                point
                for point in points
                if point.trial_seed == trial_seed
                and point.coupling_gain == weak_gain
            )
        )
        strong_metrics = _seed_metrics(
            tuple(
                point
                for point in points
                if point.trial_seed == trial_seed
                and point.coupling_gain == strong_gain
            )
        )
        paired_values.append(
            (
                weak_metrics[0] - strong_metrics[0],
                weak_metrics[1] - strong_metrics[1],
            )
        )
    intervals = tuple(
        bootstrap_mean_interval(
            tuple(values[index] for values in paired_values),
            confidence_level=confidence_level,
            resamples=resamples,
            random_seed=bootstrap_seed + index,
        )
        for index in range(2)
    )
    return OrthantMarginContrast(
        weak_minus_strong_autonomous_count=intervals[0],
        weak_minus_strong_certified_count=intervals[1],
    )


def _signed_symmetric_weights(
    *,
    dimension: int,
    diagonal_gain: float,
    coupling_gain: float,
    trial_seed: int,
) -> Matrix:
    random_generator = random.Random(trial_seed)
    edge_signs = {
        (row_index, column_index): random_generator.choice((-1.0, 1.0))
        for row_index in range(dimension)
        for column_index in range(row_index + 1, dimension)
    }
    return tuple(
        tuple(
            diagonal_gain
            if row_index == column_index
            else coupling_gain
            * edge_signs[
                (
                    min(row_index, column_index),
                    max(row_index, column_index),
                )
            ]
            for column_index in range(dimension)
        )
        for row_index in range(dimension)
    )


def _step(
    recurrent_weights: Matrix,
    state: tuple[float, ...],
    disturbance: tuple[float, ...],
) -> tuple[float, ...]:
    return tuple(
        math.tanh(
            math.fsum(
                weight * state_value
                for weight, state_value in zip(row, state, strict=True)
            )
            + disturbance_value
        )
        for row, disturbance_value in zip(
            recurrent_weights,
            disturbance,
            strict=True,
        )
    )


def _orthant_seed(trial_seed: int, signs: Signs) -> int:
    sign_code = sum(
        (1 if sign > 0 else 0) << index
        for index, sign in enumerate(signs)
    )
    return trial_seed * 1_000 + sign_code


def _validate_configuration(
    *,
    trial_seeds: tuple[int, ...],
    dimension: int,
    diagonal_gain: float,
    coupling_gains: tuple[float, float],
    safe_disturbance_ratio: float,
    witness_disturbance_ratio: float,
    safe_trials: int,
    simulation_steps: int,
    autonomous_steps: int,
    convergence_tolerance: float,
    bootstrap_confidence_level: float,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> None:
    if len(trial_seeds) < 2 or len(set(trial_seeds)) != len(trial_seeds):
        raise ValueError("trial_seedsは重複しない2個以上にしてください")
    if any(
        not isinstance(seed, int) or isinstance(seed, bool)
        for seed in trial_seeds
    ):
        raise ValueError("trial_seedsは整数にしてください")
    if (
        not isinstance(dimension, int)
        or isinstance(dimension, bool)
        or dimension < 2
        or dimension > 8
    ):
        raise ValueError("dimensionは2以上8以下の整数にしてください")
    if not math.isfinite(diagonal_gain) or diagonal_gain <= 1.0:
        raise ValueError("diagonal_gainは1より大きい有限値にしてください")
    if (
        len(coupling_gains) != 2
        or any(
            not math.isfinite(gain) or gain < 0.0
            for gain in coupling_gains
        )
        or coupling_gains[0] >= coupling_gains[1]
    ):
        raise ValueError(
            "coupling_gainsは昇順の有限非負値2個にしてください"
        )
    if (
        not math.isfinite(safe_disturbance_ratio)
        or safe_disturbance_ratio <= 0.0
        or safe_disturbance_ratio >= 1.0
    ):
        raise ValueError(
            "safe_disturbance_ratioは0と1の間にしてください"
        )
    if (
        not math.isfinite(witness_disturbance_ratio)
        or witness_disturbance_ratio <= 1.0
    ):
        raise ValueError(
            "witness_disturbance_ratioは1より大きくしてください"
        )
    _validate_positive_integer(safe_trials, "safe_trials")
    _validate_positive_integer(simulation_steps, "simulation_steps")
    _validate_positive_integer(autonomous_steps, "autonomous_steps")
    if (
        not math.isfinite(convergence_tolerance)
        or convergence_tolerance <= 0.0
    ):
        raise ValueError("convergence_toleranceは有限の正値にしてください")
    if (
        not math.isfinite(bootstrap_confidence_level)
        or bootstrap_confidence_level <= 0.0
        or bootstrap_confidence_level >= 1.0
    ):
        raise ValueError(
            "bootstrap_confidence_levelは0と1の間にしてください"
        )
    _validate_positive_integer(bootstrap_resamples, "bootstrap_resamples")
    if not isinstance(bootstrap_seed, int) or isinstance(bootstrap_seed, bool):
        raise ValueError("bootstrap_seedは整数にしてください")


def _validate_positive_integer(value: int, value_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{value_name}は1以上の整数にしてください")


def main() -> None:
    """既定30 seed実験をJSONとして出力する。"""

    print(
        json.dumps(
            asdict(run_orthant_margin_sweep()),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
