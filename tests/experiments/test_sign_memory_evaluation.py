import unittest

from reservoir_dynamics.experiments.sign_memory_evaluation import (
    evaluate_sign_memory_network,
)


class SignMemoryEvaluationTest(unittest.TestCase):
    def test_diagonal_bistable_network_returns_profile_and_lower_bound(
        self,
    ) -> None:
        profile = evaluate_sign_memory_network(
            recurrent_weights=((1.5, 0.0), (0.0, 1.5)),
            disturbance_bounds=(0.08, 0.16),
            task_steps=20,
            autonomous_steps=500,
            convergence_tolerance=1e-9,
        )

        self.assertEqual(profile.dimension, 2)
        self.assertEqual(profile.raw_attractor_count, 4)
        self.assertEqual(len(profile.orthants), 4)
        self.assertTrue(
            all(orthant.fixed_point_retained for orthant in profile.orthants)
        )
        self.assertEqual(len(profile.disturbance_evaluations), 2)
        for evaluation in profile.disturbance_evaluations:
            self.assertEqual(evaluation.total_challenges, 16)
            self.assertGreaterEqual(evaluation.guarantee_gap, -1e-12)

    def test_independent_modules_factorize_task_and_bound_global_box(
        self,
    ) -> None:
        first_module = ((1.5, 0.04), (0.04, 1.5))
        second_module = ((1.5, -0.08), (-0.08, 1.5))
        full_network = (
            (1.5, 0.04, 0.0, 0.0),
            (0.04, 1.5, 0.0, 0.0),
            (0.0, 0.0, 1.5, -0.08),
            (0.0, 0.0, -0.08, 1.5),
        )
        arguments = {
            "disturbance_bounds": (0.12,),
            "task_steps": 20,
            "autonomous_steps": 500,
            "convergence_tolerance": 1e-9,
        }

        full = evaluate_sign_memory_network(
            recurrent_weights=full_network,
            **arguments,
        )
        first = evaluate_sign_memory_network(
            recurrent_weights=first_module,
            **arguments,
        )
        second = evaluate_sign_memory_network(
            recurrent_weights=second_module,
            **arguments,
        )

        self.assertEqual(
            full.raw_attractor_count,
            first.raw_attractor_count * second.raw_attractor_count,
        )
        expected_component_margins = tuple(
            min(first_orthant.maximum_uniform_disturbance,
                second_orthant.maximum_uniform_disturbance)
            for first_orthant in first.orthants
            for second_orthant in second.orthants
        )
        self.assertTrue(
            all(
                full_orthant.maximum_uniform_disturbance
                <= component_margin + 1e-12
                for full_orthant, component_margin in zip(
                    full.orthants,
                    expected_component_margins,
                    strict=True,
                )
            )
        )
        full_evaluation = full.disturbance_evaluations[0]
        first_evaluation = first.disturbance_evaluations[0]
        second_evaluation = second.disturbance_evaluations[0]
        self.assertAlmostEqual(
            full_evaluation.task_retention,
            first_evaluation.task_retention
            * second_evaluation.task_retention,
        )
        self.assertLessEqual(
            full_evaluation.certified_robust_fraction,
            first_evaluation.certified_robust_fraction
            * second_evaluation.certified_robust_fraction
            + 1e-12,
        )

    def test_rejects_invalid_axes(self) -> None:
        with self.assertRaisesRegex(ValueError, "disturbance_bounds"):
            evaluate_sign_memory_network(
                recurrent_weights=((1.5, 0.0), (0.0, 1.5)),
                disturbance_bounds=(0.16, 0.08),
                task_steps=20,
                autonomous_steps=500,
                convergence_tolerance=1e-9,
            )


if __name__ == "__main__":
    unittest.main()
