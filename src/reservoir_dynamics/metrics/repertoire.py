"""発見済みアトラクタの実効レパートリーを評価する。"""

import math
from collections.abc import Sequence


def effective_repertoire_size(
    probabilities: Sequence[float],
    *,
    normalization_tolerance: float = 1e-12,
) -> float:
    """Shannon entropyの指数から均衡を考慮した実効個数を返す。

    数学的にはorder 1のHill numberであり、全要素が等確率なら要素数に
    一致する。入力確率を暗黙に正規化すると探索条件の不備を隠すため、
    合計が1でない入力は拒否する。
    """

    probability_values = tuple(probabilities)
    if not probability_values:
        raise ValueError("確率は1つ以上必要です")
    if (
        not math.isfinite(normalization_tolerance)
        or normalization_tolerance <= 0.0
    ):
        raise ValueError("正規化許容誤差は有限の正数である必要があります")

    for probability in probability_values:
        if not math.isfinite(probability):
            raise ValueError("確率はすべて有限である必要があります")
        if probability < 0.0:
            raise ValueError("確率はすべて非負である必要があります")

    probability_sum = math.fsum(probability_values)
    if not math.isclose(
        probability_sum,
        1.0,
        rel_tol=0.0,
        abs_tol=normalization_tolerance,
    ):
        raise ValueError("確率の合計は1である必要があります")

    entropy = -math.fsum(
        probability * math.log(probability)
        for probability in probability_values
        if probability > 0.0
    )
    return math.exp(entropy)
