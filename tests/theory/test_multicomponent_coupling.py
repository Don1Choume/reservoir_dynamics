import itertools
import unittest

from reservoir_dynamics.theory.multicomponent_coupling import (
    certificate_enumeration_complexity,
    component_inbound_load_matrix,
    summarize_multicomponent_margins,
)


class MultiComponentCouplingTest(unittest.TestCase):
    def test_preserves_every_directed_component_load(self) -> None:
        weights = (
            (1.5, 0.20, 0.01, -0.02),
            (0.10, 1.5, 0.03, 0.00),
            (0.04, 0.05, 1.5, 0.30),
            (0.02, 0.01, 0.25, 1.5),
        )
        partition = ((0, 1), (2,), (3,))

        loads = component_inbound_load_matrix(weights, partition)

        self.assertEqual(loads[0][0], 0.0)
        self.assertAlmostEqual(loads[0][1], 0.03)
        self.assertAlmostEqual(loads[0][2], 0.02)
        self.assertAlmostEqual(loads[1][0], 0.09)
        self.assertAlmostEqual(loads[1][2], 0.30)
        self.assertAlmostEqual(loads[2][0], 0.03)
        self.assertAlmostEqual(loads[2][1], 0.25)

    def test_factorized_fraction_and_mean_slack_match_cartesian_enumeration(self) -> None:
        margins = (
            (0.20, 0.12),
            (0.25, 0.14),
            (0.18, 0.10),
        )
        load_matrix = (
            (0.0, 0.0, 0.0),
            (0.03, 0.0, 0.02),
            (0.01, 0.01, 0.0),
        )

        summary = summarize_multicomponent_margins(
            component_margins=margins,
            disturbance_bound=0.10,
            inbound_load_matrix=load_matrix,
        )

        total_loads = (0.0, 0.05, 0.02)
        brute_slacks = tuple(
            min(
                margin - 0.10 - load
                for margin, load in zip(combination, total_loads, strict=True)
            )
            for combination in itertools.product(*margins)
        )
        brute_fraction = sum(value >= -1e-12 for value in brute_slacks) / len(
            brute_slacks
        )
        self.assertAlmostEqual(summary.directional_certified_fraction, brute_fraction)
        self.assertAlmostEqual(
            summary.mean_directional_slack,
            sum(brute_slacks) / len(brute_slacks),
        )
        self.assertAlmostEqual(summary.minimum_directional_slack, min(brute_slacks))
        self.assertGreaterEqual(
            summary.directional_certified_fraction,
            summary.global_certified_fraction,
        )

    def test_reports_local_to_monolithic_enumeration_reduction(self) -> None:
        complexity = certificate_enumeration_complexity((2, 2, 3))

        self.assertEqual(complexity.local_orthant_count, 16)
        self.assertEqual(complexity.monolithic_orthant_count, 128)
        self.assertAlmostEqual(complexity.local_to_monolithic_ratio, 0.125)

    def test_rejects_mismatched_load_matrix(self) -> None:
        with self.assertRaisesRegex(ValueError, "load"):
            summarize_multicomponent_margins(
                component_margins=((0.2,), (0.3,)),
                disturbance_bound=0.1,
                inbound_load_matrix=((0.0,),),
            )


if __name__ == "__main__":
    unittest.main()
