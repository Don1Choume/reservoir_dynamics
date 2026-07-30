import unittest

from reservoir_dynamics.experiments.core_reserve_trial import (
    generate_trial_inputs,
    seeded_directions,
    select_entangled_adaptation,
)


class EntangledAdaptationSelectionTest(unittest.TestCase):
    def test_selects_core_update_with_same_candidate_budget(self) -> None:
        candidates = ((0.3, 0.25), (0.6, 0.75))
        core_direction, reserve_direction = seeded_directions(
            dimension=2,
            trial_seed=41,
        )
        inputs = generate_trial_inputs(trial_seed=43, steps=80)

        choice = select_entangled_adaptation(
            core_dimension=2,
            core_direction=core_direction,
            reserve_direction=reserve_direction,
            core_recurrent_gain=0.6,
            core_input_gain=0.75,
            candidates=candidates,
            inputs=inputs,
            washout=10,
            training_steps=40,
            testing_steps=20,
            max_delay=3,
            ridge=1e-8,
        )

        self.assertIn(
            (choice.recurrent_gain, choice.input_gain),
            candidates,
        )
        self.assertGreater(choice.calibration_novel_capacity, 0.0)


if __name__ == "__main__":
    unittest.main()
