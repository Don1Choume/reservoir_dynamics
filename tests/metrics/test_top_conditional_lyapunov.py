import math
import unittest

from reservoir_dynamics.metrics.top_conditional_lyapunov import (
    top_conditional_lyapunov_exponent,
)
from reservoir_dynamics.systems.tanh_rnn import TanhRnnReservoir


class TopConditionalLyapunovExponentTest(unittest.TestCase):
    def test_matches_linearization_of_scalar_zero_orbit(self) -> None:
        reservoir = TanhRnnReservoir(
            recurrent_weights=((0.5,),),
            input_weights=((0.0,),),
            bias=(0.0,),
        )
        inputs = ((0.0,),) * 8
        trajectory = ((0.0,),) * 9

        exponent = top_conditional_lyapunov_exponent(
            system=reservoir,
            trajectory=trajectory,
            inputs=inputs,
            washout=2,
        )

        self.assertAlmostEqual(exponent, math.log(0.5))

    def test_rejects_misaligned_trajectory_and_zero_tangent(self) -> None:
        reservoir = TanhRnnReservoir(
            recurrent_weights=((0.5,),),
            input_weights=((0.0,),),
            bias=(0.0,),
        )

        with self.assertRaisesRegex(ValueError, "時系列長"):
            top_conditional_lyapunov_exponent(
                system=reservoir,
                trajectory=((0.0,),),
                inputs=((0.0,),),
            )

        with self.assertRaisesRegex(ValueError, "接ベクトル"):
            top_conditional_lyapunov_exponent(
                system=reservoir,
                trajectory=((0.0,), (0.0,)),
                inputs=((0.0,),),
                initial_tangent=(0.0,),
            )

    def test_applies_derivative_floor_to_zero_growth(self) -> None:
        reservoir = TanhRnnReservoir(
            recurrent_weights=((0.0,),),
            input_weights=((0.0,),),
            bias=(0.0,),
        )

        exponent = top_conditional_lyapunov_exponent(
            system=reservoir,
            trajectory=((0.0,), (0.0,), (0.0,)),
            inputs=((0.0,), (0.0,)),
            derivative_floor=1e-12,
        )

        self.assertAlmostEqual(exponent, math.log(1e-12))

    def test_rejects_invalid_configuration_and_values(self) -> None:
        reservoir = TanhRnnReservoir(
            recurrent_weights=((0.5,),),
            input_weights=((0.0,),),
            bias=(0.0,),
        )
        trajectory = ((0.0,), (0.0,))
        inputs = ((0.0,),)

        with self.assertRaisesRegex(ValueError, "washout"):
            top_conditional_lyapunov_exponent(
                system=reservoir,
                trajectory=trajectory,
                inputs=inputs,
                washout=True,
            )

        with self.assertRaisesRegex(ValueError, "評価区間"):
            top_conditional_lyapunov_exponent(
                system=reservoir,
                trajectory=trajectory,
                inputs=inputs,
                washout=1,
            )

        with self.assertRaisesRegex(ValueError, "derivative_floor"):
            top_conditional_lyapunov_exponent(
                system=reservoir,
                trajectory=trajectory,
                inputs=inputs,
                derivative_floor=math.nan,
            )

        with self.assertRaisesRegex(ValueError, "次元"):
            top_conditional_lyapunov_exponent(
                system=reservoir,
                trajectory=((0.0, 1.0), (0.0, 1.0)),
                inputs=inputs,
            )

        with self.assertRaisesRegex(ValueError, "有限"):
            top_conditional_lyapunov_exponent(
                system=reservoir,
                trajectory=((0.0,), (0.0,)),
                inputs=((math.inf,),),
            )

    def test_rejects_non_finite_tangent_growth(self) -> None:
        system = NonFiniteGrowthSystem()

        with self.assertRaisesRegex(ValueError, "成長率"):
            top_conditional_lyapunov_exponent(
                system=system,
                trajectory=((0.0,), (0.0,)),
                inputs=((0.0,),),
            )


class NonFiniteGrowthSystem:
    state_dimension = 1
    input_dimension = 1

    def jacobian_vector_product(
        self,
        state: tuple[float, ...],
        input_value: tuple[float, ...],
        tangent: tuple[float, ...],
    ) -> tuple[float, ...]:
        return (math.inf,)


if __name__ == "__main__":
    unittest.main()
