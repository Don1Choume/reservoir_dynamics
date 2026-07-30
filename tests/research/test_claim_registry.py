import tempfile
import unittest
from pathlib import Path

from reservoir_dynamics.research.registry import (
    ClaimRegistryError,
    EvidenceStatus,
    load_claim_registry,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLAIM_REGISTRY_PATH = PROJECT_ROOT / "docs" / "research" / "claims.toml"


class ClaimRegistryTest(unittest.TestCase):
    def test_project_registry_is_valid_and_has_unique_claim_ids(self) -> None:
        claims = load_claim_registry(CLAIM_REGISTRY_PATH)

        self.assertGreater(len(claims), 0)
        self.assertEqual(len({claim.claim_id for claim in claims}), len(claims))

    def test_hypotheses_define_falsification_criteria(self) -> None:
        claims = load_claim_registry(CLAIM_REGISTRY_PATH)
        hypotheses = tuple(
            claim for claim in claims if claim.status is EvidenceStatus.HYPOTHESIS
        )

        self.assertGreater(len(hypotheses), 0)
        self.assertTrue(
            all(claim.falsification_criteria for claim in hypotheses)
        )

    def test_sourced_claims_have_sources(self) -> None:
        claims = load_claim_registry(CLAIM_REGISTRY_PATH)
        sourced_statuses = {
            EvidenceStatus.ESTABLISHED,
            EvidenceStatus.PROVISIONAL,
            EvidenceStatus.REPRODUCED,
        }

        self.assertTrue(
            all(claim.sources for claim in claims if claim.status in sourced_statuses)
        )

    def test_rejects_duplicate_claim_ids(self) -> None:
        duplicated_claims = """
schema_version = 1

[[claims]]
id = "C-001"
status = "hypothesis"
statement = "first"
falsification_criteria = ["criterion"]

[[claims]]
id = "C-001"
status = "hypothesis"
statement = "second"
falsification_criteria = ["criterion"]
"""

        with self.assertRaisesRegex(ClaimRegistryError, "重複"):
            self._load_temporary_registry(duplicated_claims)

    def test_rejects_hypothesis_without_falsification_criteria(self) -> None:
        hypothesis_without_criterion = """
schema_version = 1

[[claims]]
id = "H-001"
status = "hypothesis"
statement = "unfalsifiable"
"""

        with self.assertRaisesRegex(ClaimRegistryError, "反証条件"):
            self._load_temporary_registry(hypothesis_without_criterion)

    def test_rejects_sourced_claim_without_source(self) -> None:
        sourced_claim_without_source = """
schema_version = 1

[[claims]]
id = "C-001"
status = "established"
statement = "missing source"
"""

        with self.assertRaisesRegex(ClaimRegistryError, "出典"):
            self._load_temporary_registry(sourced_claim_without_source)

    def test_rejects_unknown_status(self) -> None:
        unknown_status = """
schema_version = 1

[[claims]]
id = "C-001"
status = "certain"
statement = "unsupported status"
"""

        with self.assertRaisesRegex(ClaimRegistryError, "status"):
            self._load_temporary_registry(unknown_status)

    def _load_temporary_registry(self, content: str) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            registry_path = Path(temporary_directory) / "claims.toml"
            registry_path.write_text(content, encoding="utf-8")
            load_claim_registry(registry_path)


if __name__ == "__main__":
    unittest.main()
