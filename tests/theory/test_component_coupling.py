import unittest

from reservoir_dynamics.theory.component_coupling import (
    directional_bridge_norms,
    summarize_directional_margins,
)


class ComponentCouplingTest(unittest.TestCase):
    def test_directional_norms_preserve_asymmetric_loads(self) -> None:
        weights = (
            (1.5, 0.0, 0.2, -0.1),
            (0.0, 1.5, 0.05, 0.0),
            (0.3, 0.1, 1.5, 0.0),
            (0.0, -0.2, 0.0, 1.5),
        )

        inbound_first, inbound_second = directional_bridge_norms(
            weights,
            split_index=2,
        )

        self.assertAlmostEqual(inbound_first, 0.3)
        self.assertAlmostEqual(inbound_second, 0.4)

    def test_directional_budget_is_no_more_conservative_than_global(self) -> None:
        summary = summarize_directional_margins(
            component_margin_pairs=(
                (0.22, 0.31),
                (0.18, 0.28),
                (0.119, 0.21),
            ),
            disturbance_bound=0.10,
            inbound_loads=(0.02, 0.10),
        )

        self.assertAlmostEqual(summary.directional_certified_fraction, 2 / 3)
        self.assertAlmostEqual(summary.global_certified_fraction, 1 / 3)
        self.assertGreaterEqual(
            summary.directional_certified_fraction,
            summary.global_certified_fraction,
        )
        self.assertAlmostEqual(summary.minimum_directional_slack, -0.001)

    def test_rejects_invalid_split_and_empty_margin_pairs(self) -> None:
        with self.assertRaisesRegex(ValueError, "split_index"):
            directional_bridge_norms(((1.0,),), split_index=1)
        with self.assertRaisesRegex(ValueError, "margin"):
            summarize_directional_margins(
                component_margin_pairs=(),
                disturbance_bound=0.1,
                inbound_loads=(0.0, 0.0),
            )


if __name__ == "__main__":
    unittest.main()
