import unittest

from reservoir_dynamics.systems.spatial_modulation import (
    DiffusiveModulationField,
    chain_diffusion_kernel,
)


class DiffusiveModulationFieldTest(unittest.TestCase):
    def test_chain_field_preserves_hypercube_and_spreads_local_source(self) -> None:
        field = DiffusiveModulationField(
            diffusion_kernel=chain_diffusion_kernel(4),
            diffusion_rate=0.25,
            source_rate=0.5,
            minimum_gate=0.1,
        )

        first = field.step(
            state=(0.0, 0.0, 0.0, 0.0),
            source=(1.0, 0.0, 0.0, 0.0),
        )
        second = field.step(
            state=first,
            source=(0.0, 0.0, 0.0, 0.0),
        )

        self.assertEqual(first, (0.5, 0.0, 0.0, 0.0))
        self.assertGreater(second[1], 0.0)
        self.assertEqual(second[2:], (0.0, 0.0))
        self.assertTrue(all(0.0 <= value <= 1.0 for value in second))
        for observed, expected in zip(
            field.gates((0.0, 0.5, 1.0, 0.25)),
            (1.0, 0.55, 0.1, 0.775),
            strict=True,
        ):
            self.assertAlmostEqual(observed, expected)

    def test_rejects_invalid_kernel_rates_state_and_source(self) -> None:
        with self.assertRaisesRegex(ValueError, "row-stochastic"):
            DiffusiveModulationField(
                diffusion_kernel=((0.7, 0.2), (0.5, 0.5)),
                diffusion_rate=0.2,
                source_rate=0.3,
            )

        with self.assertRaisesRegex(ValueError, "合計"):
            DiffusiveModulationField(
                diffusion_kernel=chain_diffusion_kernel(2),
                diffusion_rate=0.7,
                source_rate=0.4,
            )

        field = DiffusiveModulationField(
            diffusion_kernel=chain_diffusion_kernel(2),
            diffusion_rate=0.2,
            source_rate=0.3,
        )
        with self.assertRaisesRegex(ValueError, "state"):
            field.step(state=(0.0,), source=(0.0, 0.0))
        with self.assertRaisesRegex(ValueError, "source"):
            field.step(state=(0.0, 0.0), source=(0.0, 1.1))


if __name__ == "__main__":
    unittest.main()
