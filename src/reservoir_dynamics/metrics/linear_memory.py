"""線形読み出しによる遅延入力の記憶容量を評価する。"""

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LinearMemoryResult:
    """遅延別の相関係数二乗とその総和。"""

    capacity_by_delay: tuple[float, ...]
    total_capacity: float


@dataclass(frozen=True, slots=True)
class SharedReadoutMemoryResult:
    """一つの軌道で学習したreadoutを全replicaへ適用した結果。"""

    capacity_by_replica: tuple[tuple[float, ...], ...]
    total_capacity_by_replica: tuple[float, ...]
    reference_total_capacity: float
    mean_total_capacity: float
    worst_total_capacity: float
    worst_to_reference_ratio: float


def linear_memory_curve(
    *,
    states: Sequence[Sequence[float]],
    inputs: Sequence[float],
    max_delay: int,
    washout: int,
    training_steps: int,
    testing_steps: int,
    ridge: float = 1e-8,
) -> LinearMemoryResult:
    """状態から過去入力を線形回帰し、遅延別の相関係数二乗を返す。"""

    normalized_states = _normalize_states(states)
    normalized_inputs = _normalize_inputs(inputs)
    if len(normalized_states) != len(normalized_inputs) + 1:
        raise ValueError("状態の時系列長は入力の時系列長より1大きくしてください")

    training_times, testing_times = _memory_time_ranges(
        state_count=len(normalized_states),
        max_delay=max_delay,
        washout=washout,
        training_steps=training_steps,
        testing_steps=testing_steps,
        ridge=ridge,
    )
    readout_weights = _fit_memory_readout(
        states=normalized_states,
        inputs=normalized_inputs,
        max_delay=max_delay,
        training_times=training_times,
        ridge=ridge,
    )
    observed_by_delay = _delayed_targets(
        inputs=normalized_inputs,
        max_delay=max_delay,
        evaluation_times=testing_times,
    )
    predicted_by_delay = _predict_delayed_inputs(
        states=normalized_states,
        readout_weights=readout_weights,
        evaluation_times=testing_times,
    )
    capacities = tuple(
        _squared_correlation(
            observed=observed,
            predicted=predicted,
        )
        for observed, predicted in zip(
            observed_by_delay,
            predicted_by_delay,
            strict=True,
        )
    )
    return LinearMemoryResult(
        capacity_by_delay=capacities,
        total_capacity=math.fsum(capacities),
    )


def shared_readout_memory_capacity(
    *,
    trajectories: Sequence[Sequence[Sequence[float]]],
    inputs: Sequence[float],
    max_delay: int,
    washout: int,
    training_steps: int,
    testing_steps: int,
    ridge: float = 1e-8,
) -> SharedReadoutMemoryResult:
    """参照軌道で学習した固定readoutのreplica間転移性能を返す。

    各replicaでreadoutを再学習すると吸引域ごとの局所容量しか測れない。
    ここでは先頭軌道だけで学習し、同じ重みを全軌道へ適用することで、
    初期状態を跨いで利用できる遅延記憶を保守的に評価する。
    """

    if len(trajectories) < 2:
        raise ValueError("共有readout評価にはreplica軌道が2本以上必要です")
    normalized_trajectories = tuple(
        _normalize_states(trajectory) for trajectory in trajectories
    )
    normalized_inputs = _normalize_inputs(inputs)
    expected_state_count = len(normalized_inputs) + 1
    if any(
        len(trajectory) != expected_state_count
        for trajectory in normalized_trajectories
    ):
        raise ValueError(
            "各replicaの時系列長は入力の時系列長より1大きくしてください"
        )
    reference_dimension = len(normalized_trajectories[0][0])
    if any(
        len(state) != reference_dimension
        for trajectory in normalized_trajectories
        for state in trajectory
    ):
        raise ValueError("すべてのreplicaで状態次元を一致させてください")

    training_times, testing_times = _memory_time_ranges(
        state_count=expected_state_count,
        max_delay=max_delay,
        washout=washout,
        training_steps=training_steps,
        testing_steps=testing_steps,
        ridge=ridge,
    )
    readout_weights = _fit_memory_readout(
        states=normalized_trajectories[0],
        inputs=normalized_inputs,
        max_delay=max_delay,
        training_times=training_times,
        ridge=ridge,
    )
    observed_by_delay = _delayed_targets(
        inputs=normalized_inputs,
        max_delay=max_delay,
        evaluation_times=testing_times,
    )
    capacity_by_replica = tuple(
        tuple(
            _coefficient_of_determination(
                observed=observed,
                predicted=predicted,
            )
            for observed, predicted in zip(
                observed_by_delay,
                _predict_delayed_inputs(
                    states=trajectory,
                    readout_weights=readout_weights,
                    evaluation_times=testing_times,
                ),
                strict=True,
            )
        )
        for trajectory in normalized_trajectories
    )
    total_capacity_by_replica = tuple(
        math.fsum(capacity_by_delay)
        for capacity_by_delay in capacity_by_replica
    )
    reference_total_capacity = total_capacity_by_replica[0]
    worst_total_capacity = min(total_capacity_by_replica)
    retention_ratio = (
        worst_total_capacity / reference_total_capacity
        if reference_total_capacity > 0.0
        else 0.0
    )
    return SharedReadoutMemoryResult(
        capacity_by_replica=capacity_by_replica,
        total_capacity_by_replica=total_capacity_by_replica,
        reference_total_capacity=reference_total_capacity,
        mean_total_capacity=(
            math.fsum(total_capacity_by_replica)
            / len(total_capacity_by_replica)
        ),
        worst_total_capacity=worst_total_capacity,
        worst_to_reference_ratio=retention_ratio,
    )


def _normalize_inputs(inputs: Sequence[float]) -> tuple[float, ...]:
    try:
        normalized = tuple(float(input_value) for input_value in inputs)
    except (TypeError, ValueError) as error:
        raise ValueError("入力は数値列である必要があります") from error
    if any(not math.isfinite(value) for value in normalized):
        raise ValueError("入力はすべて有限である必要があります")
    return normalized


def _memory_time_ranges(
    *,
    state_count: int,
    max_delay: int,
    washout: int,
    training_steps: int,
    testing_steps: int,
    ridge: float,
) -> tuple[range, range]:
    _validate_positive_integer(max_delay, "max_delay")
    _validate_non_negative_integer(washout, "washout")
    _validate_positive_integer(training_steps, "training_steps")
    _validate_positive_integer(testing_steps, "testing_steps")
    if not math.isfinite(ridge) or ridge < 0.0:
        raise ValueError("ridgeは有限の非負値である必要があります")

    first_sample_time = max(washout, max_delay)
    final_sample_time = first_sample_time + training_steps + testing_steps
    if final_sample_time > state_count:
        raise ValueError("指定した評価窓に対して時系列長が不足しています")
    return (
        range(first_sample_time, first_sample_time + training_steps),
        range(first_sample_time + training_steps, final_sample_time),
    )


def _fit_memory_readout(
    *,
    states: tuple[tuple[float, ...], ...],
    inputs: tuple[float, ...],
    max_delay: int,
    training_times: range,
    ridge: float,
) -> tuple[tuple[float, ...], ...]:
    design_rows = tuple((1.0,) + states[index] for index in training_times)
    target_rows = tuple(
        tuple(inputs[index - delay] for delay in range(1, max_delay + 1))
        for index in training_times
    )
    return _fit_ridge_readout(
        design_rows=design_rows,
        target_rows=target_rows,
        ridge=ridge,
    )


def _delayed_targets(
    *,
    inputs: tuple[float, ...],
    max_delay: int,
    evaluation_times: range,
) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(inputs[index - delay] for index in evaluation_times)
        for delay in range(1, max_delay + 1)
    )


def _predict_delayed_inputs(
    *,
    states: tuple[tuple[float, ...], ...],
    readout_weights: tuple[tuple[float, ...], ...],
    evaluation_times: range,
) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(
            math.fsum(
                weight * feature
                for weight, feature in zip(
                    delay_weights,
                    (1.0,) + states[index],
                    strict=True,
                )
            )
            for index in evaluation_times
        )
        for delay_weights in readout_weights
    )


def _normalize_states(
    states: Sequence[Sequence[float]],
) -> tuple[tuple[float, ...], ...]:
    if not states:
        raise ValueError("状態系列は1時刻以上必要です")
    normalized = tuple(tuple(float(value) for value in state) for state in states)
    state_dimension = len(normalized[0])
    if state_dimension < 1:
        raise ValueError("状態次元は1以上必要です")
    if any(len(state) != state_dimension for state in normalized):
        raise ValueError("すべての状態で状態次元を一致させてください")
    if any(
        not math.isfinite(value)
        for state in normalized
        for value in state
    ):
        raise ValueError("状態はすべて有限である必要があります")
    return normalized


def _fit_ridge_readout(
    *,
    design_rows: tuple[tuple[float, ...], ...],
    target_rows: tuple[tuple[float, ...], ...],
    ridge: float,
) -> tuple[tuple[float, ...], ...]:
    feature_count = len(design_rows[0])
    target_count = len(target_rows[0])
    gram_matrix = [
        [
            math.fsum(
                design_row[first_index] * design_row[second_index]
                for design_row in design_rows
            )
            for second_index in range(feature_count)
        ]
        for first_index in range(feature_count)
    ]
    for feature_index in range(1, feature_count):
        gram_matrix[feature_index][feature_index] += ridge

    cross_products = [
        [
            math.fsum(
                design_row[feature_index] * target_row[target_index]
                for design_row, target_row in zip(
                    design_rows,
                    target_rows,
                    strict=True,
                )
            )
            for target_index in range(target_count)
        ]
        for feature_index in range(feature_count)
    ]
    coefficient_columns = _solve_linear_system(
        coefficient_matrix=gram_matrix,
        right_hand_sides=cross_products,
    )
    return tuple(
        tuple(
            coefficient_columns[feature_index][target_index]
            for feature_index in range(feature_count)
        )
        for target_index in range(target_count)
    )


def _solve_linear_system(
    *,
    coefficient_matrix: list[list[float]],
    right_hand_sides: list[list[float]],
) -> tuple[tuple[float, ...], ...]:
    dimension = len(coefficient_matrix)
    augmented = [
        coefficient_row[:] + right_hand_side[:]
        for coefficient_row, right_hand_side in zip(
            coefficient_matrix,
            right_hand_sides,
            strict=True,
        )
    ]
    right_hand_side_count = len(right_hand_sides[0])

    for pivot_index in range(dimension):
        pivot_row_index = max(
            range(pivot_index, dimension),
            key=lambda row_index: abs(augmented[row_index][pivot_index]),
        )
        if abs(augmented[pivot_row_index][pivot_index]) <= 1e-15:
            raise ValueError("回帰行列が特異です。ridgeを増やしてください")
        augmented[pivot_index], augmented[pivot_row_index] = (
            augmented[pivot_row_index],
            augmented[pivot_index],
        )
        pivot_value = augmented[pivot_index][pivot_index]
        augmented[pivot_index] = [
            value / pivot_value for value in augmented[pivot_index]
        ]

        for row_index in range(dimension):
            if row_index == pivot_index:
                continue
            elimination_factor = augmented[row_index][pivot_index]
            augmented[row_index] = [
                row_value - elimination_factor * pivot_value
                for row_value, pivot_value in zip(
                    augmented[row_index],
                    augmented[pivot_index],
                    strict=True,
                )
            ]

    return tuple(
        tuple(row[dimension + offset] for offset in range(right_hand_side_count))
        for row in augmented
    )


def _squared_correlation(
    *,
    observed: tuple[float, ...],
    predicted: tuple[float, ...],
) -> float:
    observed_mean = math.fsum(observed) / len(observed)
    predicted_mean = math.fsum(predicted) / len(predicted)
    observed_deviations = tuple(value - observed_mean for value in observed)
    predicted_deviations = tuple(value - predicted_mean for value in predicted)
    observed_variance = math.fsum(value * value for value in observed_deviations)
    predicted_variance = math.fsum(
        value * value for value in predicted_deviations
    )
    if observed_variance == 0.0 or predicted_variance == 0.0:
        return 0.0
    covariance = math.fsum(
        observed_value * predicted_value
        for observed_value, predicted_value in zip(
            observed_deviations,
            predicted_deviations,
            strict=True,
        )
    )
    raw_capacity = covariance * covariance / (
        observed_variance * predicted_variance
    )
    return min(1.0, max(0.0, raw_capacity))


def _coefficient_of_determination(
    *,
    observed: tuple[float, ...],
    predicted: tuple[float, ...],
) -> float:
    observed_mean = math.fsum(observed) / len(observed)
    total_sum_of_squares = math.fsum(
        (value - observed_mean) ** 2 for value in observed
    )
    if total_sum_of_squares == 0.0:
        return 0.0
    residual_sum_of_squares = math.fsum(
        (observed_value - predicted_value) ** 2
        for observed_value, predicted_value in zip(
            observed,
            predicted,
            strict=True,
        )
    )
    raw_score = 1.0 - residual_sum_of_squares / total_sum_of_squares
    return min(1.0, max(0.0, raw_score))


def _validate_positive_integer(value: int, value_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{value_name}は1以上の整数である必要があります")


def _validate_non_negative_integer(value: int, value_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{value_name}は0以上の整数である必要があります")
