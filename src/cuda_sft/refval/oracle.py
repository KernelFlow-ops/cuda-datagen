"""Resolve an optional question-supplied numeric oracle for one dialect."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from cuda_sft.parse import abi_matches_source
from cuda_sft.refval.spec import RefManifest


def resolve_oracle_manifest(
    input_metadata: Mapping[str, Any] | None,
    *,
    dialect: str,
    question_id: int,
    code: str,
) -> RefManifest | None:
    """Return a separately supplied manifest, or raise on an invalid one."""
    metadata = input_metadata or {}
    if "oracle_manifests" not in metadata:
        return None
    manifests = metadata["oracle_manifests"]
    if not isinstance(manifests, Mapping):
        raise ValueError("oracle_manifests must map dialect names to manifests")
    if dialect not in manifests:
        raise ValueError(f"oracle_manifests has no {dialect} entry")
    raw = manifests[dialect]
    if not isinstance(raw, Mapping):
        raise ValueError(f"oracle_manifests.{dialect} must be an object")
    try:
        manifest = RefManifest.from_dict(raw)
    except (TypeError, ValueError, KeyError) as exc:
        raise ValueError(f"invalid {dialect} oracle manifest: {exc}") from exc
    if manifest.question_id != question_id or manifest.dialect != dialect:
        raise ValueError(
            f"{dialect} oracle manifest identifies q{manifest.question_id}/{manifest.dialect}, "
            f"expected q{question_id}/{dialect}"
        )
    if not manifest.reference_source.strip():
        raise ValueError(f"{dialect} oracle manifest has no reference_source")
    issues = abi_matches_source(manifest.abi, code)
    if issues:
        raise ValueError(f"{dialect} oracle ABI does not match source: {'; '.join(issues)}")
    return replace(
        manifest,
        extracted_from="independent",
        provenance={**manifest.provenance, "oracle_origin": "question_input"},
    )
