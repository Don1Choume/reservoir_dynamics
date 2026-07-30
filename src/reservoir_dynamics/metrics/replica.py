"""同一入力を受ける複製軌道間の再現性を評価する。"""

import math
from collections.abc import Sequence

State = Sequence[float]
Trajectory = Sequence[State]


def pairwise_replica_distance_curve(
    trajectories: Sequence[Trajectory],
) -> tuple[float, ...]:
    """全複製対の座標正規化RMS距離を時刻ごとに返す。

    同一入力を与え、初期状態だけを変えた複製を想定する。距離の減衰は
    入力条件付き再現性の観測量だが、有限時間でのゼロ距離だけからESPを
    数学的に証明するものではない。
    """

    normalized_trajectories = _validate_and_normalize(trajectories)
    replica_count = len(normalized_trajectories)
    horizon = len(normalized_trajectories[0])
    state_dimension = len(normalized_trajectories[0][0])
    pair_count = replica_count * (replica_count - 1) // 2

    distances: list[float] = []
    for time_index in range(horizon):
        squared_distance_sum = 0.0
        for first_replica_index in range(replica_count - 1):
            first_state = normalized_trajectories[first_replica_index][time_index]
            for second_replica_index in range(
                first_replica_index + 1,
                replica_count,
            ):
                second_state = normalized_trajectories[second_replica_index][
                    time_index
                ]
                squared_distance_sum += math.fsum(
                    (first_value - second_value) ** 2
                    for first_value, second_value in zip(
                        first_state,
                        second_state,
                        strict=True,
                    )
                )

        mean_squared_coordinate_distance = squared_distance_sum / (
            pair_count * state_dimension
        )
        distances.append(math.sqrt(mean_squared_coordinate_distance))

    return tuple(distances)


def _validate_and_normalize(
    trajectories: Sequence[Trajectory],
) -> tuple[tuple[tuple[float, ...], ...], ...]:
    if len(trajectories) < 2:
        raise ValueError("複製軌道は2つ以上必要です")

    normalized = tuple(
        tuple(tuple(float(value) for value in state) for state in trajectory)
        for trajectory in trajectories
    )
    reference_horizon = len(normalized[0])
    if reference_horizon == 0:
        raise ValueError("各複製軌道には1時刻以上の状態が必要です")
    if any(len(trajectory) != reference_horizon for trajectory in normalized):
        raise ValueError("すべての複製軌道で時系列長を一致させてください")

    reference_dimension = len(normalized[0][0])
    if reference_dimension == 0:
        raise ValueError("状態次元は1以上必要です")
    if any(
        len(state) != reference_dimension
        for trajectory in normalized
        for state in trajectory
    ):
        raise ValueError("すべての状態で状態次元を一致させてください")
    if any(
        not math.isfinite(value)
        for trajectory in normalized
        for state in trajectory
        for value in state
    ):
        raise ValueError("状態値はすべて有限である必要があります")

    return normalized
