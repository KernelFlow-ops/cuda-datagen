"""Immutable-in-practice records for a single generated candidate version."""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Mapping
from typing import Any, Literal, TypedDict


class CandidateSnapshot(TypedDict):
    candidate: int
    repairs: int
    version: int
    code: str
    code_sha256: str
    gen_system: str
    gen_user: str
    prompt_variant: dict[str, Any]
    first_turn_reasoning: str
    first_turn_reasoning_source: str
    final_turn_reasoning: str
    final_turn_reasoning_source: str
    origin: str
    provider: str
    model: str
    compile: Literal["pass", "fail"]
    used_rdc: bool
    compile_log_tail: str
    contract: Literal["pass", "fail", "skip"]
    refval: dict[str, Any]
    refval_owner: Literal["KERNEL", "ORACLE", "INFRA", ""]
    oracle_blocked: bool
    critic: dict[str, Any]
    judge: dict[str, Any]
    perf: dict[str, Any] | None
    banked_reason: str
    last_gate: dict[str, Any]


def build_snapshot(state: Mapping[str, Any], *, banked_reason: str) -> CandidateSnapshot:
    ctx = dict(state.get("candidate_ctx") or {})
    metadata = dict(state.get("metadata") or {})
    code = str(state.get("code") or "")
    compile_ok = bool(state.get("compile_ok"))
    compile_status: Literal["pass", "fail"] = "pass" if compile_ok else "fail"
    refval = copy.deepcopy(metadata.get("refval") or {})
    if not isinstance(refval, dict):
        refval = {}
    refval["status"] = str(state.get("refval_status") or refval.get("status") or "skip")
    if not compile_ok:
        refval.update(status="skip", cases_run=0, manifest_hash="")
    critic = copy.deepcopy(metadata.get("critic") or {})
    if not isinstance(critic, dict):
        critic = {}
    critic.setdefault(
        "status",
        "skipped"
        if state.get("critic_skipped", True)
        else ("verified" if state.get("critic_pass", True) else "failed"),
    )
    judge = copy.deepcopy(metadata.get("judge") or {})
    if not isinstance(judge, dict):
        judge = {}
    judge.setdefault("quality_score", int(state.get("judge_score") or 0))
    judge.setdefault("issues", list(state.get("judge_issues") or []))
    judge.setdefault("suggestions", list(state.get("judge_suggestions") or []))
    owner = str(state.get("refval_owner") or "")
    refval_owner: Literal["KERNEL", "ORACLE", "INFRA", ""]
    if owner == "KERNEL":
        refval_owner = "KERNEL"
    elif owner == "ORACLE":
        refval_owner = "ORACLE"
    elif owner == "INFRA":
        refval_owner = "INFRA"
    else:
        refval_owner = ""
    quality = state.get("quality_status") or {}
    raw_contract = str(quality.get("contract") or "skip") if isinstance(quality, Mapping) else "skip"
    contract: Literal["pass", "fail", "skip"] = (
        "pass" if raw_contract == "pass" else "fail" if raw_contract == "fail" else "skip"
    )
    if contract == "pass" and refval.get("verification_tier") != "independent":
        contract = "skip"
    return CandidateSnapshot(
        candidate=int(state.get("candidate_idx") or 1),
        repairs=int(state.get("repair_idx") or 0),
        version=1,
        code=code,
        code_sha256=hashlib.sha256(code.encode("utf-8")).hexdigest(),
        gen_system=str(ctx.get("gen_system") or state.get("system_prompt") or ""),
        gen_user=str(ctx.get("gen_user") or state.get("user_prompt") or ""),
        prompt_variant=copy.deepcopy(ctx.get("prompt_variant") or {}),
        first_turn_reasoning=str(ctx.get("first_turn_reasoning") or ""),
        first_turn_reasoning_source=str(ctx.get("first_turn_reasoning_source") or "empty"),
        final_turn_reasoning=str(
            ctx.get("final_turn_reasoning") or state.get("raw_reasoning") or ""
        ),
        final_turn_reasoning_source=str(
            ctx.get("final_turn_reasoning_source") or state.get("reasoning_source") or "empty"
        ),
        origin=str(ctx.get("origin") or state.get("origin") or "unknown"),
        provider=str(ctx.get("provider") or (state.get("provenance") or {}).get("provider") or ""),
        model=str(ctx.get("model") or (state.get("provenance") or {}).get("model") or ""),
        compile=compile_status,
        used_rdc=bool(state.get("used_rdc", False)),
        compile_log_tail=str(state.get("compile_error") or "")[-2000:],
        contract=contract,
        refval=refval,
        refval_owner=refval_owner,
        oracle_blocked=bool(state.get("oracle_blocked", False)),
        critic=critic,
        judge=judge,
        perf=copy.deepcopy(metadata.get("perf"))
        if isinstance(metadata.get("perf"), dict)
        else None,
        banked_reason=banked_reason,
        last_gate=copy.deepcopy(state.get("last_gate") or {}),
    )


def snapshot_for_metadata(s: CandidateSnapshot) -> dict[str, Any]:
    out: dict[str, Any] = copy.deepcopy(dict(s))
    for key in ("code", "gen_system", "gen_user", "first_turn_reasoning", "final_turn_reasoning"):
        out.pop(key, None)
    return out


def merge_into_pool(
    pool: list[CandidateSnapshot], snap: CandidateSnapshot
) -> list[CandidateSnapshot]:
    from cuda_sft.core.selection import rank_key

    reports = [copy.deepcopy(item) for item in pool]
    for index, old in enumerate(reports):
        if old["candidate"] == snap["candidate"]:
            if rank_key(snap) >= rank_key(old):
                improved = copy.deepcopy(snap)
                improved["version"] = old["version"] + 1
                reports[index] = improved
            return reports
    reports.append(copy.deepcopy(snap))
    reports.sort(key=lambda item: item["candidate"])
    return reports
