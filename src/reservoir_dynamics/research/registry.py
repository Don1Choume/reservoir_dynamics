"""機械可読な研究主張台帳を検証して読み込む。"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class ClaimRegistryError(ValueError):
    """研究台帳の構造または証拠境界が不正な場合の例外。"""


class EvidenceStatus(StrEnum):
    """主張の検証状態。"""

    ESTABLISHED = "established"
    REPRODUCED = "reproduced"
    PROVISIONAL = "provisional"
    HYPOTHESIS = "hypothesis"
    REFUTED = "refuted"


class EvidenceType(StrEnum):
    """research-opsで分離する証拠の由来。"""

    SOURCED_FACT = "sourced_fact"
    LOCAL_REPRODUCTION = "local_reproduction"
    INFERENCE = "inference"
    HYPOTHESIS = "hypothesis"


@dataclass(frozen=True, slots=True)
class ClaimRecord:
    """出典、実装、反証条件へ接続された一つの研究上の主張。"""

    claim_id: str
    status: EvidenceStatus
    evidence_type: EvidenceType
    statement: str
    sources: tuple[str, ...]
    implementations: tuple[str, ...]
    tests: tuple[str, ...]
    limitations: tuple[str, ...]
    falsification_criteria: tuple[str, ...]


_SOURCE_REQUIRED_STATUSES = {
    EvidenceStatus.ESTABLISHED,
    EvidenceStatus.REPRODUCED,
    EvidenceStatus.PROVISIONAL,
    EvidenceStatus.REFUTED,
}


def load_claim_registry(path: Path) -> tuple[ClaimRecord, ...]:
    """TOML台帳を読み、証拠境界を検証した不変レコード列を返す。"""

    try:
        document = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ClaimRegistryError(f"研究台帳を読み込めません: {path}") from error

    if document.get("schema_version") != 1:
        raise ClaimRegistryError("未対応のschema_versionです")

    raw_claims = document.get("claims")
    if not isinstance(raw_claims, list) or not raw_claims:
        raise ClaimRegistryError("claimsは1件以上必要です")

    claims: list[ClaimRecord] = []
    observed_ids: set[str] = set()
    for raw_claim in raw_claims:
        claim = _parse_claim(raw_claim)
        if claim.claim_id in observed_ids:
            raise ClaimRegistryError(
                f"主張IDが重複しています: {claim.claim_id}"
            )
        observed_ids.add(claim.claim_id)
        claims.append(claim)

    return tuple(claims)


def _parse_claim(raw_claim: object) -> ClaimRecord:
    if not isinstance(raw_claim, Mapping):
        raise ClaimRegistryError("各claimはTOML tableである必要があります")

    claim_id = _required_string(raw_claim, "id")
    statement = _required_string(raw_claim, "statement")
    status = _parse_enum(raw_claim, "status", EvidenceStatus)
    evidence_type = _parse_evidence_type(raw_claim, status)
    sources = _string_tuple(raw_claim, "sources")
    falsification_criteria = _string_tuple(
        raw_claim,
        "falsification_criteria",
    )

    if status in _SOURCE_REQUIRED_STATUSES and not sources:
        raise ClaimRegistryError(
            f"{claim_id}: 検証済みまたは暫定的な主張には出典が必要です"
        )
    if status is EvidenceStatus.HYPOTHESIS and not falsification_criteria:
        raise ClaimRegistryError(
            f"{claim_id}: 仮説には反証条件が必要です"
        )

    return ClaimRecord(
        claim_id=claim_id,
        status=status,
        evidence_type=evidence_type,
        statement=statement,
        sources=sources,
        implementations=_string_tuple(raw_claim, "implementations"),
        tests=_string_tuple(raw_claim, "tests"),
        limitations=_string_tuple(raw_claim, "limitations"),
        falsification_criteria=falsification_criteria,
    )


def _parse_evidence_type(
    raw_claim: Mapping[object, object],
    status: EvidenceStatus,
) -> EvidenceType:
    raw_evidence_type = raw_claim.get("evidence_type")
    if raw_evidence_type is None:
        default_by_status = {
            EvidenceStatus.ESTABLISHED: EvidenceType.SOURCED_FACT,
            EvidenceStatus.REPRODUCED: EvidenceType.LOCAL_REPRODUCTION,
            EvidenceStatus.PROVISIONAL: EvidenceType.SOURCED_FACT,
            EvidenceStatus.HYPOTHESIS: EvidenceType.HYPOTHESIS,
            EvidenceStatus.REFUTED: EvidenceType.LOCAL_REPRODUCTION,
        }
        return default_by_status[status]
    return _parse_enum(raw_claim, "evidence_type", EvidenceType)


def _parse_enum(
    raw_claim: Mapping[object, object],
    field_name: str,
    enum_type: type[EvidenceStatus] | type[EvidenceType],
) -> EvidenceStatus | EvidenceType:
    raw_value = raw_claim.get(field_name)
    if not isinstance(raw_value, str):
        raise ClaimRegistryError(f"{field_name}は文字列である必要があります")
    try:
        return enum_type(raw_value)
    except ValueError as error:
        raise ClaimRegistryError(
            f"未対応の{field_name}です: {raw_value}"
        ) from error


def _required_string(
    raw_claim: Mapping[object, object],
    field_name: str,
) -> str:
    raw_value = raw_claim.get(field_name)
    if not isinstance(raw_value, str) or not raw_value.strip():
        raise ClaimRegistryError(f"{field_name}は空でない文字列が必要です")
    return raw_value.strip()


def _string_tuple(
    raw_claim: Mapping[object, object],
    field_name: str,
) -> tuple[str, ...]:
    raw_value = raw_claim.get(field_name, [])
    if not isinstance(raw_value, list) or any(
        not isinstance(item, str) or not item.strip() for item in raw_value
    ):
        raise ClaimRegistryError(
            f"{field_name}は空でない文字列の配列である必要があります"
        )
    return tuple(item.strip() for item in raw_value)
