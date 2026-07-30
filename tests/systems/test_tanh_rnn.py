import math
import unittest

from reservoir_dynamics.systems.tanh_rnn import TanhRnnReservoir


class TanhRnnReservoirTest(unittest.TestCase):
    def test_updates_multidimensional_state(self) -> None:
        reservoir = TanhRnnReservoir(
            recurrent_weights=((0.5, -0.25), (0.1, 0.4)),
            input_weights=((0.2,), (-0.3,)),
            bias=(0.05, -0.1),
        )

        observed = reservoir.step((0.4, -0.2), (0.5,))

        self.assertEqual(reservoir.state_dimension, 2)
        self.assertEqual(reservoir.input_dimension, 1)
        self.assertAlmostEqual(observed[0], math.tanh(0.4))
        self.assertAlmostEqual(observed[1], math.tanh(-0.29))

    def test_computes_jacobian_vector_product(self) -> None:
        reservoir = TanhRnnReservoir(
            recurrent_weights=((0.5, -0.25), (0.1, 0.4)),
            input_weights=((0.2,), (-0.3,)),
            bias=(0.05, -0.1),
        )
        state = (0.4, -0.2)
        input_value = (0.5,)
        tangent = (0.6, -0.8)
        next_state = reservoir.step(state, input_value)

        observed = reservoir.jacobian_vector_product(
            state,
            input_value,
            tangent,
        )

        expected_linearized = (0.5, -0.26)
        self.assertAlmostEqual(
            observed[0],
            (1.0 - next_state[0] ** 2) * expected_linearized[0],
        )
        self.assertAlmostEqual(
            observed[1],
            (1.0 - next_state[1] ** 2) * expected_linearized[1],
        )

    def test_rejects_inconsistent_dimensions_and_non_finite_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "1行"):
            TanhRnnReservoir(
                recurrent_weights=(),
                input_weights=(),
                bias=(),
            )

        with self.assertRaisesRegex(ValueError, "正方"):
            TanhRnnReservoir(
                recurrent_weights=((0.5, 0.2),),
                input_weights=((0.1,),),
                bias=(0.0,),
            )

        with self.assertRaisesRegex(ValueError, "有限"):
            TanhRnnReservoir(
                recurrent_weights=((math.nan,),),
                input_weights=((0.1,),),
                bias=(0.0,),
            )

        with self.assertRaisesRegex(ValueError, "行数"):
            TanhRnnReservoir(
                recurrent_weights=((0.5,),),
                input_weights=(),
                bias=(0.0,),
            )

        with self.assertRaisesRegex(ValueError, "bias"):
            TanhRnnReservoir(
                recurrent_weights=((0.5,),),
                input_weights=((0.1,),),
                bias=(),
            )

        with self.assertRaisesRegex(ValueError, "入力次元"):
            TanhRnnReservoir(
                recurrent_weights=((0.5,),),
                input_weights=((),),
                bias=(0.0,),
            )

        reservoir = TanhRnnReservoir(
            recurrent_weights=((0.5,),),
            input_weights=((0.1,),),
            bias=(0.0,),
        )
        with self.assertRaisesRegex(ValueError, "接ベクトル"):
            reservoir.jacobian_vector_product((0.0,), (0.0,), (1.0, 2.0))

        with self.assertRaisesRegex(ValueError, "有限"):
            reservoir.jacobian_vector_product((0.0,), (0.0,), (math.inf,))

        with self.assertRaisesRegex(ValueError, "状態の次元"):
            reservoir.step((0.0, 1.0), (0.0,))

        with self.assertRaisesRegex(ValueError, "入力の次元"):
            reservoir.step((0.0,), (0.0, 1.0))

        with self.assertRaisesRegex(ValueError, "有限"):
            reservoir.step((math.inf,), (0.0,))


if __name__ == "__main__":
    unittest.main()
