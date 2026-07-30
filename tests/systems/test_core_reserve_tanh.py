import unittest

from reservoir_dynamics.systems.core_reserve_tanh import (
    CoreReserveTanhRnn,
)
from reservoir_dynamics.systems.tanh_rnn import TanhRnnReservoir


class CoreReserveTanhRnnTest(unittest.TestCase):
    def setUp(self) -> None:
        self.base_system = CoreReserveTanhRnn(
            system=TanhRnnReservoir(
                recurrent_weights=((0.5, 0.0), (0.2, 0.1)),
                input_weights=((0.7, 0.0), (0.0, 0.0)),
                bias=(0.1, 0.0),
            ),
            core_dimension=1,
        )

    def test_reserve_adaptation_preserves_core_parameters(self) -> None:
        adapted = self.base_system.with_reserve_parameters(
            core_to_reserve_weights=((0.4,),),
            reserve_recurrent_weights=((0.8,),),
            reserve_input_weights=((0.0, 0.9),),
            reserve_bias=(-0.2,),
        )

        self.assertEqual(
            adapted.system.recurrent_weights[0],
            self.base_system.system.recurrent_weights[0],
        )
        self.assertEqual(
            adapted.system.input_weights[0],
            self.base_system.system.input_weights[0],
        )
        self.assertEqual(
            adapted.system.bias[0],
            self.base_system.system.bias[0],
        )
        self.assertEqual(
            adapted.system.recurrent_weights[1],
            (0.4, 0.8),
        )

    def test_zero_feedback_makes_core_step_independent_of_reserve(self) -> None:
        first_next = self.base_system.step((0.25, -0.9), (0.3, -0.7))
        second_next = self.base_system.step((0.25, 0.9), (0.3, -0.7))

        self.assertEqual(first_next[0], second_next[0])
        self.assertNotEqual(first_next[1], second_next[1])

    def test_reports_block_infinity_norms_and_state_slices(self) -> None:
        feedback_system = CoreReserveTanhRnn(
            system=TanhRnnReservoir(
                recurrent_weights=(
                    (0.4, -0.1, 0.2),
                    (0.0, 0.5, -0.3),
                    (0.2, 0.1, 0.6),
                ),
                input_weights=((1.0,), (0.5,), (0.25,)),
                bias=(0.0, 0.0, 0.0),
            ),
            core_dimension=2,
        )

        self.assertAlmostEqual(
            feedback_system.core_recurrent_infinity_norm,
            0.5,
        )
        self.assertAlmostEqual(
            feedback_system.reserve_feedback_infinity_norm,
            0.3,
        )
        self.assertEqual(
            feedback_system.split_state((1.0, 2.0, 3.0)),
            ((1.0, 2.0), (3.0,)),
        )

    def test_rejects_invalid_partition_and_reserve_shapes(self) -> None:
        with self.assertRaisesRegex(ValueError, "core_dimension"):
            CoreReserveTanhRnn(
                system=self.base_system.system,
                core_dimension=2,
            )

        with self.assertRaisesRegex(ValueError, "reserve_recurrent_weights"):
            self.base_system.with_reserve_parameters(
                core_to_reserve_weights=((0.4,),),
                reserve_recurrent_weights=((0.8, 0.1),),
                reserve_input_weights=((0.0, 0.9),),
                reserve_bias=(0.0,),
            )

        with self.assertRaisesRegex(ValueError, "状態"):
            self.base_system.split_state((1.0,))


if __name__ == "__main__":
    unittest.main()
