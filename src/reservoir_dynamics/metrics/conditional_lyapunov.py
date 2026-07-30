"""入力条件付きの有限時間Lyapunov指数を推定する。"""

import math
from collections.abc import Sequence


def finite_time_conditional_lyapunov_exponent(
    local_derivative_magnitudes: Sequence[float],
    *,
    washout: int = 0,
    derivative_floor: float = 1e-300,
) -> float:
    """局所微分絶対値の対数時間平均を返す。

    ゼロ微分をそのまま対数化すると負の無限大になるため、数値表現上の
    floorを明示的に受け取る。floorは成果物へ保存し、暗黙に変更しない。
    """

    if not isinstance(washout, int) or isinstance(washout, bool) or washout < 0:
        raise ValueError("washoutは0以上の整数である必要があります")
    if not math.isfinite(derivative_floor) or derivative_floor <= 0.0:
        raise ValueError("derivative_floorは有限の正数である必要があります")

    derivative_values = tuple(local_derivative_magnitudes)
    for derivative_magnitude in derivative_values:
        if not math.isfinite(derivative_magnitude):
            raise ValueError("局所微分絶対値はすべて有限である必要があります")
        if derivative_magnitude < 0.0:
            raise ValueError("局所微分絶対値はすべて非負である必要があります")

    evaluation_values = derivative_values[washout:]
    if not evaluation_values:
        raise ValueError("washout後の評価区間は1時刻以上必要です")

    logarithmic_growth = math.fsum(
        math.log(max(derivative_magnitude, derivative_floor))
        for derivative_magnitude in evaluation_values
    )
    return logarithmic_growth / len(evaluation_values)
