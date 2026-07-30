"""研究上の主張、出典、反証条件を追跡する機能。"""

from reservoir_dynamics.research.registry import (
    ClaimRecord,
    ClaimRegistryError,
    EvidenceStatus,
    EvidenceType,
    load_claim_registry,
)

__all__ = [
    "ClaimRecord",
    "ClaimRegistryError",
    "EvidenceStatus",
    "EvidenceType",
    "load_claim_registry",
]
