import math
import random
import unittest

from reservoir_dynamics.metrics.linear_memory import (
    linear_memory_curve,
    shared_readout_memory_capacity,
)


class LinearMemoryCurveTest(unittest.TestCase):
    def test_recovers_exact_two_step_delay_line(self) -> None:
        random_generator = random.Random(71)
        inputs = tuple(random_generator.uniform(-1.0, 1.0) for _ in range(240))
        states = [(0.0, 0.0)]
        for time_index in range(len(inputs)):
            recent_input = inputs[time_index]
            previous_input = inputs[time_index - 1] if time_index >= 1 else 0.0
            states.append((recent_input, previous_input))

        result = linear_memory_curve(
            states=states,
            inputs=inputs,
            max_delay=2,
            washout=10,
            training_steps=140,
            testing_steps=80,
            ridge=1e-10,
        )

        self.assertGreater(result.capacity_by_delay[0], 0.999999)
        self.assertGreater(result.capacity_by_delay[1], 0.999999)
        self.assertAlmostEqual(result.total_capacity, 2.0, places=5)

    def test_returns_zero_for_constant_target_variance(self) -> None:
        result = linear_memory_curve(
            states=((0.0,),) * 21,
            inputs=(1.0,) * 20,
            max_delay=2,
            washout=2,
            training_steps=10,
            testing_steps=8,
        )

        self.assertEqual(result.capacity_by_delay, (0.0, 0.0))
        self.assertEqual(result.total_capacity, 0.0)

    def test_rejects_invalid_windows_and_ridge(self) -> None:
        with self.assertRaisesRegex(ValueError, "時系列長"):
            linear_memory_curve(
                states=((0.0,),) * 5,
                inputs=(0.0,) * 5,
                max_delay=1,
                washout=1,
                training_steps=2,
                testing_steps=1,
            )

        with self.assertRaisesRegex(ValueError, "ridge"):
            linear_memory_curve(
                states=((0.0,),) * 6,
                inputs=(0.0,) * 5,
                max_delay=1,
                washout=1,
                training_steps=2,
                testing_steps=2,
                ridge=math.nan,
            )


class SharedReadoutMemoryCapacityTest(unittest.TestCase):
    def test_retains_capacity_across_identical_replicas(self) -> None:
        inputs, reference_states = create_delay_line(seed=83)

        result = shared_readout_memory_capacity(
            trajectories=(reference_states, reference_states),
            inputs=inputs,
            max_delay=2,
            washout=10,
            training_steps=140,
            testing_steps=80,
            ridge=1e-10,
        )

        self.assertAlmostEqual(result.reference_total_capacity, 2.0, places=5)
        self.assertAlmostEqual(result.worst_total_capacity, 2.0, places=5)
        self.assertAlmostEqual(result.worst_to_reference_ratio, 1.0, places=5)

    def test_detects_readout_failure_in_sign_inverted_replica(self) -> None:
        inputs, reference_states = create_delay_line(seed=97)
        inverted_states = tuple(
            tuple(-value for value in state) for state in reference_states
        )

        result = shared_readout_memory_capacity(
            trajectories=(reference_states, inverted_states),
            inputs=inputs,
            max_delay=2,
            washout=10,
            training_steps=140,
            testing_steps=80,
            ridge=1e-10,
        )

        self.assertGreater(result.reference_total_capacity, 1.999)
        self.assertEqual(result.total_capacity_by_replica[1], 0.0)
        self.assertEqual(result.worst_total_capacity, 0.0)
        self.assertEqual(result.worst_to_reference_ratio, 0.0)

    def test_rejects_misaligned_replica_horizon(self) -> None:
        inputs, reference_states = create_delay_line(seed=101)

        with self.assertRaisesRegex(ValueError, "時系列長"):
            shared_readout_memory_capacity(
                trajectories=(reference_states, reference_states[:-1]),
                inputs=inputs,
                max_delay=2,
                washout=10,
                training_steps=140,
                testing_steps=80,
            )


def create_delay_line(
    *,
    seed: int,
) -> tuple[tuple[float, ...], tuple[tuple[float, ...], ...]]:
    random_generator = random.Random(seed)
    inputs = tuple(random_generator.uniform(-1.0, 1.0) for _ in range(240))
    states = [(0.0, 0.0)]
    for time_index, recent_input in enumerate(inputs):
        previous_input = inputs[time_index - 1] if time_index >= 1 else 0.0
        states.append((recent_input, previous_input))
    return inputs, tuple(states)


if __name__ == "__main__":
    unittest.main()
