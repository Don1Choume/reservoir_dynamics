import math
import unittest
from dataclasses import FrozenInstanceError

from reservoir_dynamics.systems.scalar_tanh import ScalarTanhReservoir


class ScalarTanhReservoirTest(unittest.TestCase):
    def test_computes_driven_state_update(self) -> None:
        reservoir = ScalarTanhReservoir(
            recurrent_gain=0.5,
            input_gain=2.0,
            bias=-0.25,
        )

        next_state = reservoir.step((1.0,), (0.5,))

        self.assertAlmostEqual(next_state[0], math.tanh(1.25))

    def test_reports_global_state_lipschitz_bound(self) -> None:
        reservoir = ScalarTanhReservoir(
            recurrent_gain=-0.75,
            input_gain=1.0,
        )

        self.assertAlmostEqual(reservoir.global_state_lipschitz_bound, 0.75)
        self.assertTrue(reservoir.is_globally_contractive)

    def test_gain_of_one_is_not_strictly_contractive(self) -> None:
        reservoir = ScalarTanhReservoir(
            recurrent_gain=1.0,
            input_gain=1.0,
        )

        self.assertFalse(reservoir.is_globally_contractive)

    def test_parameters_are_immutable(self) -> None:
        reservoir = ScalarTanhReservoir(
            recurrent_gain=0.5,
            input_gain=1.0,
        )

        with self.assertRaises(FrozenInstanceError):
            reservoir.recurrent_gain = 0.9  # type: ignore[misc]

    def test_rejects_non_finite_parameter(self) -> None:
        with self.assertRaisesRegex(ValueError, "有限"):
            ScalarTanhReservoir(
                recurrent_gain=math.inf,
                input_gain=1.0,
            )

    def test_rejects_invalid_direct_step_dimensions(self) -> None:
        reservoir = ScalarTanhReservoir(
            recurrent_gain=0.5,
            input_gain=1.0,
        )

        with self.assertRaisesRegex(ValueError, "1次元"):
            reservoir.step((0.0, 1.0), (0.0,))


if __name__ == "__main__":
    unittest.main()
