import unittest

from reservoir_dynamics.experiments.asymmetric_modular_family import (
    audit_asymmetric_modular_structure,
    build_asymmetric_modular_network,
)
from reservoir_dynamics.experiments.component_profile import (
    COMPONENT_FEATURE_NAMES,
    GLOBAL_FEATURE_NAMES,
    evaluate_component_profile,
)
from reservoir_dynamics.experiments.component_predictor import (
    fit_preregistered_component_models,
    leave_one_seed_out_evaluations,
)


class ComponentSizeExtrapolationTest(unittest.TestCase):
    def test_generator_has_unequal_directional_norms_and_unique_gate(self) -> None:
        network = build_asymmetric_modular_network(
            trial_seed=2001,
            module_sizes=(2, 3),
            internal_gain=0.025,
            maximum_bridge_strength=0.02,
        )

        self.assertEqual(len(network.recurrent_weights), 5)
        self.assertNotEqual(
            network.inbound_bridge_norms[0],
            network.inbound_bridge_norms[1],
        )
        self.assertTrue(network.internal_blocks_asymmetric)
        self.assertTrue(network.bridges_nontransposed)

        gate = audit_asymmetric_modular_structure(
            trial_seeds=(2001, 2002),
            module_size_pairs=((2, 2), (2, 3)),
            internal_gains=(0.025,),
            maximum_bridge_strengths=(0.0, 0.02),
        )
        self.assertTrue(gate.passed)
        self.assertTrue(gate.fingerprints_unique)

    def test_zero_coupling_factorizes_and_certificate_chain_holds(self) -> None:
        zero_network = build_asymmetric_modular_network(
            trial_seed=2001,
            module_sizes=(2, 2),
            internal_gain=0.025,
            maximum_bridge_strength=0.0,
        )
        coupled_network = build_asymmetric_modular_network(
            trial_seed=2001,
            module_sizes=(2, 2),
            internal_gain=0.025,
            maximum_bridge_strength=0.02,
        )
        arguments = {
            "disturbance_bounds": (0.12,),
            "task_steps": 10,
            "autonomous_steps": 500,
            "convergence_tolerance": 1e-9,
        }

        zero_point = evaluate_component_profile(
            network=zero_network,
            **arguments,
        )[0]
        coupled_point = evaluate_component_profile(
            network=coupled_network,
            **arguments,
        )[0]

        self.assertAlmostEqual(
            zero_point.observed_task_retention,
            zero_point.isolated_task_product,
        )
        self.assertGreaterEqual(
            coupled_point.observed_task_retention + 1e-12,
            coupled_point.transported_certified_fraction,
        )
        self.assertGreaterEqual(
            coupled_point.transported_certified_fraction + 1e-12,
            coupled_point.directional_certified_fraction,
        )
        self.assertGreaterEqual(
            coupled_point.directional_certified_fraction + 1e-12,
            coupled_point.global_shifted_certified_fraction,
        )
        self.assertEqual(
            len(coupled_point.global_feature_row),
            len(GLOBAL_FEATURE_NAMES),
        )
        self.assertEqual(
            len(coupled_point.component_feature_row),
            len(COMPONENT_FEATURE_NAMES),
        )

    def test_preregistered_models_fit_and_leave_one_seed_out(self) -> None:
        points = tuple(
            point
            for seed in (2001, 2002, 2003)
            for size_pair in ((2, 2), (2, 3))
            for strength in (0.0, 0.02)
            for point in evaluate_component_profile(
                network=build_asymmetric_modular_network(
                    trial_seed=seed,
                    module_sizes=size_pair,
                    internal_gain=0.025,
                    maximum_bridge_strength=strength,
                ),
                disturbance_bounds=(0.12,),
                task_steps=10,
                autonomous_steps=500,
                convergence_tolerance=1e-9,
            )
        )

        models = fit_preregistered_component_models(points)
        evaluations = leave_one_seed_out_evaluations(points)

        self.assertEqual(
            tuple(model.name for model in models),
            ("component_aware", "global_profile", "product_only"),
        )
        self.assertEqual(
            tuple(value.name for value in evaluations),
            ("component_aware", "global_profile", "product_only"),
        )
        self.assertTrue(all(value.mae >= 0.0 for value in evaluations))
        self.assertTrue(
            all(len(value.predictions) == len(points) for value in evaluations)
        )


if __name__ == "__main__":
    unittest.main()

