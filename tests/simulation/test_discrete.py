import math
import unittest
from dataclasses import dataclass

from reservoir_dynamics.simulation.discrete import simulate_discrete_replicas


@dataclass(frozen=True, slots=True)
class AdditiveScalarSystem:
    state_dimension: int = 1
    input_dimension: int = 1

    def step(
        self,
        state: tuple[float, ...],
        input_value: tuple[float, ...],
    ) -> tuple[float, ...]:
        return (state[0] + input_value[0],)


@dataclass(frozen=True, slots=True)
class InvalidOutputSystem:
    state_dimension: int = 1
    input_dimension: int = 1

    def step(
        self,
        state: tuple[float, ...],
        input_value: tuple[float, ...],
    ) -> tuple[float, ...]:
        return (state[0], input_value[0])


class DiscreteReplicaSimulationTest(unittest.TestCase):
    def test_applies_the_same_input_sequence_to_every_initial_state(self) -> None:
        result = simulate_discrete_replicas(
            system=AdditiveScalarSystem(),
            initial_states=((-1.0,), (2.0,)),
            inputs=((0.5,), (-1.0,)),
        )

        self.assertEqual(
            result.trajectories,
            (
                ((-1.0,), (-0.5,), (-1.5,)),
                ((2.0,), (2.5,), (1.5,)),
            ),
        )

    def test_preserves_inputs_in_immutable_result(self) -> None:
        source_inputs = [[0.5], [-1.0]]

        result = simulate_discrete_replicas(
            system=AdditiveScalarSystem(),
            initial_states=((-1.0,),),
            inputs=source_inputs,
        )
        source_inputs[0][0] = 99.0

        self.assertEqual(result.inputs, ((0.5,), (-1.0,)))

    def test_allows_empty_input_sequence_for_initial_state_snapshot(self) -> None:
        result = simulate_discrete_replicas(
            system=AdditiveScalarSystem(),
            initial_states=((-1.0,), (2.0,)),
            inputs=(),
        )

        self.assertEqual(result.trajectories, (((-1.0,),), ((2.0,),)))

    def test_rejects_empty_initial_states(self) -> None:
        with self.assertRaisesRegex(ValueError, "初期状態は1つ以上"):
            simulate_discrete_replicas(
                system=AdditiveScalarSystem(),
                initial_states=(),
                inputs=((0.0,),),
            )

    def test_rejects_invalid_state_dimension(self) -> None:
        with self.assertRaisesRegex(ValueError, "状態次元"):
            simulate_discrete_replicas(
                system=AdditiveScalarSystem(),
                initial_states=((0.0, 1.0),),
                inputs=((0.0,),),
            )

    def test_rejects_invalid_input_dimension(self) -> None:
        with self.assertRaisesRegex(ValueError, "入力次元"):
            simulate_discrete_replicas(
                system=AdditiveScalarSystem(),
                initial_states=((0.0,),),
                inputs=((0.0, 1.0),),
            )

    def test_rejects_non_finite_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "有限"):
            simulate_discrete_replicas(
                system=AdditiveScalarSystem(),
                initial_states=((math.nan,),),
                inputs=((0.0,),),
            )

    def test_rejects_system_output_with_wrong_dimension(self) -> None:
        with self.assertRaisesRegex(ValueError, "出力した状態次元"):
            simulate_discrete_replicas(
                system=InvalidOutputSystem(),
                initial_states=((0.0,),),
                inputs=((1.0,),),
            )


if __name__ == "__main__":
    unittest.main()
