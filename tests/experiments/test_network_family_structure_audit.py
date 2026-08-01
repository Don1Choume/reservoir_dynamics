import unittest

from reservoir_dynamics.experiments.network_family_structure_audit import (
    StructuralAuditSpecification,
    run_network_family_structure_audit,
)


class NetworkFamilyStructureAuditTest(unittest.TestCase):
    def test_modular_sign_seeds_collapse_to_one_class_per_gain(self) -> None:
        result = run_network_family_structure_audit(
            specifications=(
                StructuralAuditSpecification(
                    network_family="modular_paired",
                    coupling_gains=(0.04, 0.07),
                    trial_seeds=(1301, 1302, 1303, 1304, 1305),
                ),
            )
        )

        self.assertEqual(result.raw_network_count, 10)
        self.assertEqual(result.effective_class_count, 2)
        self.assertAlmostEqual(result.effective_fraction, 0.2)
        self.assertEqual(len(result.audits), 2)
        for audit in result.audits:
            self.assertEqual(audit.raw_network_count, 5)
            self.assertEqual(audit.effective_class_count, 1)
            self.assertEqual(audit.class_sizes, (5,))
            self.assertEqual(audit.representative_seeds, (1301,))

    def test_rejects_duplicate_families_seeds_and_gains(self) -> None:
        valid = StructuralAuditSpecification(
            network_family="modular_paired",
            coupling_gains=(0.04,),
            trial_seeds=(1301, 1302),
        )

        with self.assertRaisesRegex(ValueError, "family"):
            run_network_family_structure_audit(
                specifications=(valid, valid),
            )
        with self.assertRaisesRegex(ValueError, "seed"):
            run_network_family_structure_audit(
                specifications=(
                    StructuralAuditSpecification(
                        network_family="modular_paired",
                        coupling_gains=(0.04,),
                        trial_seeds=(1301, 1301),
                    ),
                )
            )
        with self.assertRaisesRegex(ValueError, "coupling_gains"):
            run_network_family_structure_audit(
                specifications=(
                    StructuralAuditSpecification(
                        network_family="modular_paired",
                        coupling_gains=(0.04, 0.04),
                        trial_seeds=(1301, 1302),
                    ),
                )
            )


if __name__ == "__main__":
    unittest.main()
