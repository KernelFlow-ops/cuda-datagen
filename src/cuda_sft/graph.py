"""LangGraph: generate → extract → compile → validate → repair / judge → cot / save.

Nodes are thin wrappers around dialect specs and ``cuda_sft.agents``.
Print-stream and RetryPolicy live in ``pipeline.common`` so the knowledge
graph does not keep a second copy.
"""

from __future__ import annotations

import copy
import hashlib
import logging
import re
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from cuda_sft.agents.critic import KernelCritic, critic_result_to_dict
from cuda_sft.agents.difficulty import plan_topology
from cuda_sft.agents.generate import (
    assistant_state_update,
    cancel_speculative,
    complete_chat,
)
from cuda_sft.agents.repairer import (
    classify_compile_error,
    classify_refval_error,
    quality_repair_body,
    repair_system_prompt,
    wrap_repair_user,
)
from cuda_sft.compile import attempt_workdir, finalize_question_work
from cuda_sft.config import ROLE_CONFIG_PREFIXES, get_settings
from cuda_sft.core.gates import (
    ErrorClass,
    FailureOwner,
    GateResult,
    owner_of_compile,
    owner_of_refval,
)
from cuda_sft.core.sample import pick_reasoning
from cuda_sft.core.selection import eligible, rank_key
from cuda_sft.core.types import build_snapshot, merge_into_pool, snapshot_for_metadata
from cuda_sft.cot import CotAgent
from cuda_sft.dialects.agent import get_dialect_agent, get_spec
from cuda_sft.judge import JudgeResult
from cuda_sft.pipeline.common import (
    graph_recursion_limit,
    retry_policy,
    set_print_stream,
)
from cuda_sft.prompt import (
    SYSTEM_PROMPT,
    candidate_temperature,
    format_nvcc_for_prompt,
    truncate_compile_error,
)
from cuda_sft.runtime import deps
from cuda_sft.runtime.limits import stage_lock
from cuda_sft.runtime.meta import CallMeta, legacy_job_key
from cuda_sft.runtime.trace import traced
from cuda_sft.state import GraphState
from cuda_sft.store import get_store
from cuda_sft.tasks.kinds import question_hash

logger = logging.getLogger(__name__)

# Re-export so ``from cuda_sft.graph import set_print_stream`` keeps working.
__all__ = ["build_graph", "build_candidate_graph", "recursion_limit", "set_print_stream"]


def _dialect_name(state: GraphState) -> str:
    """Canonical dialect id for this graph state (default cuda)."""
    return str(state.get("dialect") or "cuda")


def _spec(state: GraphState):
    """Kernel dialect spec for this state."""
    return get_spec(_dialect_name(state))


def _repair_cap(state: GraphState, settings: Any) -> int:
    value = state.get("repair_cap")
    return int(settings.max_repairs if value is None else value)


def _text_hash(value: str) -> str:
    """Return a stable, non-secret hash for provenance and idempotency metadata."""
    return hashlib.sha256((value or "").encode("utf-8")).hexdigest()


def _operation_family(question: str) -> str:
    """Classify a question into a coarse operation family for audit metadata.

    This deliberately stays conservative: the family is a routing/provenance
    hint, never a correctness decision.  The deterministic refval contract and
    compiler remain the hard gates.
    """
    text = (question or "").lower()
    patterns = (
        ("gemm", r"\b(gemm|matmul|matrix multiplication|wmma|tensor core)\b"),
        ("reduction", r"\b(reduc|sum|max|min|argmax|argmin|softmax|scan|prefix)"),
        ("transpose", r"\b(transpose|permute|reorder|strided)\b"),
        ("sort", r"\b(sort|top[- ]?k|histogram|set[- ]?union)\b"),
        ("convolution", r"\b(conv|convolution)\b"),
        ("memory", r"\b(memcpy|copy|gather|scatter|broadcast|atomic)\b"),
    )
    for family, pattern in patterns:
        if re.search(pattern, text):
            return family
    return "elementwise"


def _task_contract(state: GraphState, *, dialect: str, settings: Any) -> dict[str, Any]:
    """Build a stable task contract from input metadata and the raw question.

    Existing JSONL rows remain valid.  When a producer already supplied
    structured fields they are preserved; otherwise this records conservative
    hints which the refval extractor can verify against the compiled source.
    """
    supplied = state.get("input_metadata")
    raw = dict(supplied) if isinstance(supplied, dict) else {}
    contract = raw.get("task_spec") if isinstance(raw.get("task_spec"), dict) else {}
    return {
        "operation_family": str(
            contract.get("operation_family")
            or raw.get("operation_family")
            or _operation_family(str(state.get("question") or ""))
        ),
        "dialect": dialect,
        "target_arch": str(state.get("cuda_arch") or settings.resolved_cuda_arch),
        "dtype": contract.get("dtype") or raw.get("dtype") or "f32",
        "layout": contract.get("layout") or raw.get("layout") or "contiguous",
        "shape_constraints": list(
            contract.get("shape_constraints") or raw.get("shape_constraints") or []
        ),
        "abi": dict(contract.get("abi") or raw.get("abi") or {}),
        "forbidden_symbols": ["main", "include/solution_header.h", "include/helpers.h"],
        "requested_dialects": list(raw.get("requested_dialects") or []),
        "available_dialects": list(raw.get("available_dialects") or []),
        "group_id": str(raw.get("group_id") or ""),
        "question_hash": question_hash(str(state.get("question") or "")),
    }


def prepare(state: GraphState) -> dict[str, Any]:
    """Initialize prompts, counters, and GPU metadata for one question.

    Args:
        state: Must contain ``question_id`` and ``question``.

    Returns:
        Partial state update for the generate node.
    """
    settings = get_settings()
    gpu_name = settings.resolved_gpu_name
    cuda_arch = settings.resolved_cuda_arch
    cuda_version = settings.resolved_cuda_version
    dialect = str(state.get("dialect") or "cuda")
    candidate_idx = int(state.get("candidate_idx") or 1)
    spec = get_spec(dialect)
    selected = spec.select_prompts(
        state["question"],
        question_id=int(state["question_id"]),
        candidate_idx=candidate_idx,
        gpu_name=gpu_name,
        cuda_arch=cuda_arch,
        cuda_version=cuda_version,
    )
    topo = plan_topology(question=state["question"], kind="kernel", settings=settings)
    task_spec = _task_contract(
        {**state, "cuda_arch": cuda_arch}, dialect=spec.name, settings=settings
    )
    generator_route = settings.for_role("generator")
    provenance = {
        "question_hash": task_spec["question_hash"],
        "source": str(state.get("source") or "unknown"),
        "prompt_version": "candidate-pool-1",
        "model": generator_route.resolved_model,
        "provider": generator_route.llm_provider,
        "cuda_version": cuda_version,
    }
    return {
        "dialect": spec.name,
        "system_prompt": selected.system,
        "user_prompt": selected.user,
        "messages": [{"role": "user", "content": selected.user}],
        "candidate_idx": candidate_idx,
        "candidate_ctx": {
            "candidate": candidate_idx,
            "gen_system": selected.system,
            "gen_user": selected.user,
            "prompt_variant": {
                "system_index": getattr(selected, "system_index", -1),
                "suffix_index": getattr(selected, "suffix_index", -1),
                "temperature": candidate_temperature(candidate_idx),
                "prompt_pack": "gen-v1",
            },
            "first_turn_reasoning": "",
            "first_turn_reasoning_source": "empty",
            "final_turn_reasoning": "",
            "final_turn_reasoning_source": "empty",
        },
        "selected": {},
        "repair_idx": 0,
        "oracle_retry_idx": 0,
        "oracle_blocked": False,
        "repair_cap": min(topo.max_repairs, 1) if settings.kernel_fast_mode else topo.max_repairs,
        "temperature": candidate_temperature(candidate_idx),
        "raw_response": "",
        "origin": "unknown",
        "code": "",
        "compile_ok": False,
        "compile_error": "",
        "used_rdc": False,
        "status": "running",
        "abandon_reason": "",
        "last_gate": {},
        "attempts": [],
        "gpu_name": gpu_name,
        "cuda_arch": cuda_arch,
        "cuda_version": cuda_version,
        "winner_found": False,
        "winner_candidate": 0,
        "judge_score": 0,
        "judge_issues": [],
        "judge_suggestions": [],
        "metadata": {},
        "task_spec": task_spec,
        "oracle_spec": {
            "compare_mode": "elementwise",
            "case_suite": str(getattr(settings, "refval_cases", "standard")),
            "validator_version": "refval-v1",
        },
        "quality_status": {
            "contract": "skip",
            "compile": "pending",
            "refval": "pending",
            "static_safety": "pending",
            "semantic": "pending",
            "performance": "unmeasured",
            "release_tier": "quarantine",
        },
        "provenance": provenance,
        "candidate_reports": [],
        "speculative_requests": [],
        "raw_reasoning": "",
        "reasoning_source": "empty",
        "cot": "",
        "cot_source": "empty",
        "cot_error": "",
        "difficulty": topo.difficulty,
        "candidate_cap": 1 if settings.kernel_fast_mode else topo.max_candidates,
        "use_critic": False if settings.kernel_fast_mode else topo.use_critic,
        "critic_pass": True,
        "critic_skipped": True,
        "critic_must_fix": [],
        "critic_issues": [],
        "refval_ok": True,
        "refval_status": "skip",
        "refval_error": "",
        "refval_error_class": "",
        "refval_owner": "",
    }


def generate(state: GraphState) -> dict[str, Any]:
    """Call the LLM and append the assistant turn to ``messages``.

    Args:
        state: Current graph state with ``messages`` and ``system_prompt``.

    Returns:
        ``raw_response`` and updated ``messages``.
    """
    qid = state["question_id"]
    cand = state.get("candidate_idx", 1)
    repair = state.get("repair_idx", 0)
    temperature = float(state.get("temperature") or candidate_temperature(cand))
    dialect = _dialect_name(state)
    header = f"[Q{qid} {dialect} candidate={cand} repair={repair} temp={temperature}]"
    settings = get_settings()
    request_id = f"q{qid}_{dialect}_c{cand}_r{repair}"
    repair_role = str((state.get("last_gate") or {}).get("repair_role") or "")
    role = "generator" if not repair else repair_role
    if role not in {"repair.compile", "repair.numeric", "repair.semantic"}:
        role = "repair.compile" if repair else "generator"
    route = settings.for_role(role)
    llm_options = get_dialect_agent().llm_call_options(dialect, settings)
    if settings.kernel_fast_mode:
        llm_options = {
            **llm_options,
            "thinking_level": "low",
            "max_output_tokens": min(settings.resolved_max_output_tokens, 8192),
        }
    explicit_thinking = getattr(
        settings, f"{ROLE_CONFIG_PREFIXES[role]}_thinking_level", ""
    ).strip()
    if explicit_thinking:
        llm_options["thinking_level"] = route.thinking_level
    result = complete_chat(
        messages=list(state.get("messages") or []),
        system=state.get("system_prompt") or SYSTEM_PROMPT,
        temperature=temperature,
        request_id=request_id if settings.async_llm_enabled else None,
        llm_options=llm_options,
        log_header=header,
        meta=CallMeta(
            role=role,
            job_key=legacy_job_key(int(qid), dialect),
            question_id=int(qid),
            track=dialect,
            candidate=int(cand),
            repair=int(repair),
        ),
    )
    update = assistant_state_update(state, result, log_header=header)
    provenance = dict(state.get("provenance") or {})
    provenance.update(provider=route.llm_provider, model=route.resolved_model)
    update["provenance"] = provenance
    candidate_ctx = dict(state.get("candidate_ctx") or {})
    if not repair:
        candidate_ctx["first_turn_reasoning"] = result.reasoning
        candidate_ctx["first_turn_reasoning_source"] = result.reasoning_source
    candidate_ctx["final_turn_reasoning"] = result.reasoning
    candidate_ctx["final_turn_reasoning_source"] = result.reasoning_source
    candidate_ctx["origin"] = result.origin
    candidate_ctx["provider"] = route.llm_provider
    candidate_ctx["model"] = route.resolved_model
    update["candidate_ctx"] = candidate_ctx
    return update


def _repair_turn(
    state: GraphState,
    *,
    next_repair: int,
    prompt_error: str,
    code: str,
    error_class: str | None = None,
    evidence: str = "",
    repair_role: str = "repair.compile",
) -> tuple[str, str]:
    """Build the repair user message and Repairer system prompt.

    Called only after the compiler, numeric validator, or critic supplies a
    concrete failure diagnostic.
    """
    spec = _spec(state)
    if not error_class:
        # compile_node classified the full log; the prompt text may be trimmed.
        gate = state.get("last_gate") or {}
        if gate.get("gate") == "compile" and gate.get("error_class"):
            error_class = str(gate["error_class"])
        else:
            error_class = classify_compile_error(prompt_error, dialect=spec.name, code=code)
    settings = get_settings()
    cuda_arch = state.get("cuda_arch") or settings.resolved_cuda_arch
    inner = (
        quality_repair_body(
            dialect=spec.name,
            code=code,
            diagnosis=prompt_error,
            role=repair_role,
            cuda_arch=cuda_arch,
            filename=spec.source_filename,
        )
        if repair_role in {"repair.numeric", "repair.semantic"}
        else spec.build_repair(
            cuda_arch=cuda_arch,
            compile_error=prompt_error,
            previous_code=code,
            question_id=int(state["question_id"]),
            candidate_idx=int(state.get("candidate_idx") or 1),
            repair_idx=next_repair,
        )
    )
    single_turn = settings.repair_history_mode != "full"
    user = wrap_repair_user(
        question=str(state.get("question") or ""),
        inner=inner,
        error_class=error_class,
        dialect=spec.name,
        evidence=evidence,
        repair_idx=next_repair,
        max_repairs=_repair_cap(state, settings),
        role=repair_role,
        filename=spec.source_filename,
        history=_earlier_attempts(state) if single_turn else "",
    )
    generation_system = str((state.get("candidate_ctx") or {}).get("gen_system") or "")
    return user, repair_system_prompt(
        spec.name, role=repair_role, generation_system=generation_system
    )


def _first_error_line(text: str, limit: int = 200) -> str:
    """First line mentioning an error (else the first non-empty line), shortened."""
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    if not lines:
        return "(no diagnostic)"
    line = next((item for item in lines if re.search(r"error", item, re.IGNORECASE)), lines[0])
    line = re.sub(r"\s+", " ", line)
    return line if len(line) <= limit else line[: limit - 3] + "..."


def _earlier_attempts(state: GraphState, *, limit: int = 3) -> str:
    """One line per earlier failed round of this candidate; the current round is excluded.

    Single-turn repair does not resend the chat history, so without this the
    model can reintroduce a mistake it already fixed two rounds ago.
    """
    current = int(state.get("repair_idx") or 0)
    candidate = int(state.get("candidate_idx") or 1)
    lines: list[str] = []
    for item in state.get("attempts") or []:
        if item.get("ok") or int(item.get("candidate") or 1) != candidate:
            continue
        round_idx = int(item.get("repair") or 0)
        if round_idx >= current:
            continue
        stage = "refval" if item.get("refval_status") else "compile"
        error_class = str(item.get("error_class") or "")
        label = f"{stage}/{error_class}" if error_class else stage
        lines.append(
            f"- round {round_idx}: {label}: {_first_error_line(str(item.get('error') or ''))}"
        )
    return "\n".join(lines[-limit:])


def extract(state: GraphState) -> dict[str, Any]:
    """Pull CUDA source out of the last model reply.

    Args:
        state: Must contain ``raw_response``.
    """
    spec = _spec(state)
    code = spec.extract(state.get("raw_response") or "")
    if not code.strip():
        logger.warning("Q%s %s: no source extracted", state["question_id"], spec.name)
    speculative_requests = list(state.get("speculative_requests") or [])
    settings = get_settings()
    if (
        settings.refval_enabled and settings.async_llm_enabled
        and not settings.kernel_fast_mode and code.strip()
        and "oracle_manifests" not in (state.get("input_metadata") or {})
    ):
        try:
            from cuda_sft.refval.runner import enqueue_speculative_extract

            request_id = enqueue_speculative_extract(
                question=str(state.get("question") or ""),
                code=code,
                question_id=int(state["question_id"]),
                dialect=spec.name,
                candidate=int(state.get("candidate_idx") or 1),
                repair=int(state.get("repair_idx") or 0),
                dialect_spec=spec.refval_spec(settings),
                settings=settings,
            )
            if request_id:
                speculative_requests.append(request_id)
        except Exception:
            logger.exception("failed to enqueue speculative refval extract")
    return {
        "code": code,
        "speculative_requests": speculative_requests,
        "oracle_retry_idx": 0,
        "oracle_blocked": False,
        "refval_owner": "",
    }


def compile_node(state: GraphState) -> dict[str, Any]:
    """``nvcc -c`` the extracted source and record the attempt.

    Args:
        state: Must contain ``code`` and ``question_id``.
    """
    settings = get_settings()
    spec = _spec(state)
    nest = get_dialect_agent().nest_workdir(spec.name, settings)
    workdir = attempt_workdir(
        settings,
        int(state["question_id"]),
        int(state.get("candidate_idx") or 1),
        int(state.get("repair_idx") or 0),
        spec.name,
        nest_dialect=nest,
    )
    code = state.get("code") or ""
    attempts = list(state.get("attempts") or [])
    log_name = "nvcc.log" if spec.language == "cuda-cpp" else "compile.log"

    if not code.strip():
        error = f"no {spec.name} source extracted from model output"
        attempts.append(
            {
                "candidate": state.get("candidate_idx", 1),
                "repair": state.get("repair_idx", 0),
                "ok": False,
                "used_rdc": False,
                "error": error,
            }
        )
        print(f"[Q{state['question_id']} {spec.name}] compile FAIL: {error}", flush=True)
        workdir.mkdir(parents=True, exist_ok=True)
        (workdir / log_name).write_text(error + "\n", encoding="utf-8")
        return {
            "compile_ok": False,
            "compile_error": error,
            "refval_owner": "",
            "used_rdc": False,
            "attempts": attempts,
            "last_gate": GateResult(
                gate="compile",
                passed=False,
                owner=FailureOwner.KERNEL,
                error_class=ErrorClass.EMPTY_SOURCE.value,
                evidence=error,
                metrics={},
            ).to_dict(),
            "quality_status": {
                **dict(state.get("quality_status") or {}),
                "compile": "fail",
                "release_tier": "quarantine",
            },
        }

    compile_fn = deps.current().compile_fn
    with stage_lock("compile"):
        result = (
            compile_fn(spec.name, code, workdir, settings)
            if compile_fn is not None
            else spec.compile(code, workdir, settings)
        )
    error = "" if result.ok else (result.output or "compile failed with empty output")
    owner, error_class = owner_of_compile(result, error)
    (workdir / log_name).write_text((result.output or error or "") + "\n", encoding="utf-8")
    if result.ok:
        prompt_error = ""
    elif spec.language == "cuda-cpp":
        prompt_error = format_nvcc_for_prompt(error, settings.repair_error_max_chars)
    else:
        prompt_error = (error or "")[: settings.repair_error_max_chars]
    attempts.append(
        {
            "candidate": state.get("candidate_idx", 1),
            "repair": state.get("repair_idx", 0),
            "ok": result.ok,
            "used_rdc": result.used_rdc,
            "error": prompt_error,
        }
    )
    status = "PASS" if result.ok else "FAIL"
    extra = " (rdc)" if result.used_rdc else ""
    print(f"[Q{state['question_id']} {spec.name}] compile {status}{extra}", flush=True)
    if not result.ok:
        logger.info("compile error:\n%s", truncate_compile_error(error, 2000))
        refval_ids = [
            item
            for item in (state.get("speculative_requests") or [])
            if str(item).endswith("_refval")
        ]
        if refval_ids:
            cancel_speculative(refval_ids)

    return {
        "compile_ok": result.ok,
        "compile_error": error,
        "used_rdc": result.used_rdc,
        "attempts": attempts,
        "last_gate": GateResult(
            gate="compile",
            passed=result.ok,
            owner=owner,
            error_class=error_class,
            evidence=error if not result.ok else "",
            metrics={"used_rdc": bool(result.used_rdc)},
        ).to_dict(),
        "speculative_requests": list(state.get("speculative_requests") or []),
        "refval_ok": True,
        "refval_status": "skip",
        "refval_error": "",
        "refval_error_class": "",
        "refval_owner": "",
        "quality_status": {
            **dict(state.get("quality_status") or {}),
            "compile": "pass" if result.ok else "fail",
            "release_tier": "candidate" if result.ok else "quarantine",
        },
    }


def validate(state: GraphState) -> dict[str, Any]:
    """CPU-reference + GPU numeric gate after a compile-passing kernel.

    Skip/reference_error do not block save (unless ``REFVAL_STRICT``). Numeric
    failures set ``refval_ok=False`` so routing reuses the compile repair path.
    """
    from cuda_sft.refval.runner import refval_blocks_save, run_refval

    settings = get_settings()
    spec = _spec(state)
    metadata = dict(state.get("metadata") or {})
    dialect = spec.name
    speculative_ids = [
        item
        for item in (state.get("speculative_requests") or [])
        if str(item).endswith("_refval")
    ]
    if not settings.refval_enabled:
        metadata["refval"] = {
            "status": "skip",
            "dialect": dialect,
            "cases_run": 0,
            "failed_case": "",
            "tolerances": {},
            "manifest_summary": {},
            "seed": 0,
            "reason": "disabled",
        }
        print(f"[Q{state['question_id']} {dialect}] refval SKIP disabled", flush=True)
        return {
            "refval_ok": True,
            "refval_status": "skip",
            "refval_error": "",
            "refval_error_class": "",
            "refval_owner": "",
            "last_gate": GateResult(
                gate="refval", passed=True, owner=None, error_class="",
                evidence="disabled", metrics={"cases_run": 0},
            ).to_dict(),
            "metadata": metadata,
            "quality_status": {
                **dict(state.get("quality_status") or {}),
                "refval": "skip",
                "release_tier": "compile_only",
            },
        }

    try:
        dialect_spec = spec.refval_spec(settings)
    except Exception as exc:
        metadata["refval"] = {
            "status": "skip",
            "dialect": dialect,
            "reason": f"no refval_spec: {exc}",
        }
        print(f"[Q{state['question_id']} {dialect}] refval SKIP {exc}", flush=True)
        return {
            "refval_ok": True,
            "refval_status": "skip",
            "refval_error": str(exc),
            "refval_error_class": "",
            "refval_owner": "",
            "last_gate": GateResult(
                gate="refval", passed=True, owner=None, error_class="",
                evidence=str(exc), metrics={"cases_run": 0},
            ).to_dict(),
            "metadata": metadata,
            "quality_status": {
                **dict(state.get("quality_status") or {}),
                "refval": "skip",
                "release_tier": "compile_only",
            },
        }

    nest = get_dialect_agent().nest_workdir(spec.name, settings)
    workdir = attempt_workdir(
        settings,
        int(state["question_id"]),
        int(state.get("candidate_idx") or 1),
        int(state.get("repair_idx") or 0),
        spec.name,
        nest_dialect=nest,
    )
    refval_fn = deps.current().refval_fn or run_refval
    input_metadata = state.get("input_metadata") or {}
    supplied_oracle = (
        {"oracle_manifests": input_metadata["oracle_manifests"]}
        if "oracle_manifests" in input_metadata else {}
    )
    report = refval_fn(
        question=str(state.get("question") or ""),
        code=str(state.get("code") or ""),
        question_id=int(state["question_id"]),
        dialect=dialect,
        dialect_spec=dialect_spec,
        settings=settings,
        workdir=workdir,
        nest_dialect=False,
        used_rdc=bool(state.get("used_rdc")),
        speculative_id=speculative_ids[-1] if speculative_ids else None,
        task_spec=state.get("task_spec") or {},
        oracle_spec=state.get("oracle_spec") or {},
        provenance=state.get("provenance") or {},
        meta=CallMeta(
            role="refval_extract",
            job_key=legacy_job_key(int(state["question_id"]), dialect),
            question_id=int(state["question_id"]),
            track=dialect,
            candidate=int(state.get("candidate_idx") or 1),
            repair=int(state.get("repair_idx") or 0),
            purpose="extract",
        ),
        **supplied_oracle,
    )
    metadata["refval"] = report.to_metadata()
    owner, owned_error_class = owner_of_refval(report)
    blocks = refval_blocks_save(report, settings)
    attempts = list(state.get("attempts") or [])
    if blocks and attempts:
        # Compile recorded this attempt as ok before numeric validation.
        # Keep the refval reason on the attempt so abandoned rows are diagnosable.
        last = dict(attempts[-1])
        last["ok"] = False
        last["error"] = (report.evidence or report.reason or "refval failed")[:4000]
        last["refval_status"] = report.status
        last["error_class"] = report.error_class
        attempts[-1] = last
    flag = report.status.upper()
    extra = f" class={report.error_class}" if report.error_class else ""
    print(
        f"[Q{state['question_id']} {dialect}] refval {flag} "
        f"cases={report.cases_run}{extra} {report.failed_case or report.reason or ''}".rstrip(),
        flush=True,
    )
    return {
        "refval_ok": not blocks,
        "refval_status": report.status,
        "refval_error": report.evidence or report.reason,
        "refval_error_class": report.error_class,
        "refval_owner": owner.value if owner is not None else "",
        "oracle_blocked": report.error_class == "invalid_oracle",
        "last_gate": GateResult(
            gate="refval",
            passed=owner is None,
            owner=owner,
            error_class=owned_error_class,
            evidence=report.evidence or report.reason or "",
            metrics={"cases_run": report.cases_run, "elapsed_s": report.elapsed_sec},
        ).to_dict(),
        "attempts": attempts,
        "metadata": metadata,
        "quality_status": {
            **dict(state.get("quality_status") or {}),
            "contract": "pass" if report.verification_tier == "independent" else "skip",
            "refval": report.status,
            "verification_tier": report.verification_tier,
            "release_tier": "candidate" if not blocks else "quarantine",
        },
    }


def repair(state: GraphState) -> dict[str, Any]:
    """Append a compile-fix user message and bump ``repair_idx``.

    Args:
        state: Failed compile state with ``compile_error`` and ``code``.

    Returns:
        Updated state. If another candidate already won, returns minimal update
        with skip_repair=True to signal early termination.
    """
    # Reserved for a future parallel-candidate graph. The sequential graph never
    # runs two candidates at once, so this branch does not fire today.
    if state.get("winner_found") and state.get("winner_candidate") != state.get("candidate_idx"):
        logger.info(
            "Q%s skipping candidate %s repair (candidate %s already won)",
            state["question_id"],
            state.get("candidate_idx"),
            state.get("winner_candidate"),
        )
        return {"skip_repair": True}

    settings = get_settings()
    next_repair = int(state.get("repair_idx") or 0) + 1
    evidence = ""
    error_class = None
    metadata = state.get("metadata") or {}
    critic_record = metadata.get("critic") or {}
    critic_failed = str(critic_record.get("status") or "") == "failed"
    semantic_repair = bool(
        state.get("compile_ok")
        and state.get("refval_ok", True)
        and critic_failed
    )
    numeric_repair = bool(state.get("compile_ok") and not state.get("refval_ok", True))
    if semantic_repair:
        diagnostics = [
            str(item).strip()
            for item in (state.get("critic_must_fix") or [])
            if str(item).strip()
        ]
        if not diagnostics:
            diagnostics = [
                str(item).strip()
                for item in (state.get("critic_issues") or [])
                if str(item).strip()
            ]
        if not diagnostics:
            failure = critic_record.get("failure") or {}
            message = str(failure.get("message") or "critic rejected the candidate").strip()
            diagnostics = [message]
        prompt_error = "semantic critic rejected the candidate:\n- " + "\n- ".join(
            diagnostics
        )
        error_class = "semantic"
        repair_role = "repair.semantic"
    elif numeric_repair:
        # The diagnosis block already carries the full evidence (entry, params,
        # failing case); passing it again as ``evidence`` duplicated it.
        prompt_error = str(state.get("refval_error") or "numeric validation failed")
        error_class = classify_refval_error(
            prompt_error, error_class=str(state.get("refval_error_class") or "")
        )
        repair_role = "repair.numeric"
    else:
        prompt_error = format_nvcc_for_prompt(
            state.get("compile_error") or "",
            settings.repair_error_max_chars,
        )
        repair_role = "repair.compile"
    repair_user, repair_system = _repair_turn(
        state,
        next_repair=next_repair,
        prompt_error=prompt_error,
        code=state.get("code") or "",
        error_class=error_class,
        evidence=evidence,
        repair_role=repair_role,
    )
    if settings.repair_history_mode == "full":
        messages = [*(state.get("messages") or []), {"role": "user", "content": repair_user}]
    else:
        # The repair turn re-attaches the problem, the contract, the previous
        # source and a summary of earlier rounds; resending the chat history
        # duplicated all of that on every round.
        messages = [{"role": "user", "content": repair_user}]
    repair_cap = _repair_cap(state, settings)
    logger.info(
        "Q%s candidate %s starting repair %s/%s (%s, %d messages)",
        state["question_id"],
        state.get("candidate_idx", 1),
        next_repair,
        repair_cap,
        repair_role,
        len(messages),
    )
    # Verdicts below describe the previous source. Drop them (as next_candidate
    # does) so a repaired version that never reaches judge/critic is not
    # banked with its predecessor's critic/judge record.
    metadata = {
        key: value
        for key, value in (state.get("metadata") or {}).items()
        if key not in {"refval", "critic", "judge"}
    }
    return {
        "repair_idx": next_repair,
        "oracle_retry_idx": 0,
        "oracle_blocked": False,
        "refval_owner": "",
        "messages": messages,
        "system_prompt": repair_system,
        "raw_response": "",
        "compile_ok": False,
        "skip_repair": False,
        "refval_ok": True,
        "refval_status": "skip",
        "refval_error": "",
        "refval_error_class": "",
        "judge_score": 0,
        "judge_issues": [],
        "judge_suggestions": [],
        "critic_pass": True,
        "critic_skipped": True,
        "critic_must_fix": [],
        "critic_issues": [],
        "metadata": metadata,
        "last_gate": {
            **dict(state.get("last_gate") or {}),
            "repair_role": repair_role,
        },
    }


def next_candidate(state: GraphState) -> dict[str, Any]:
    """Start a fresh candidate with a different prompt variant and temperature.

    Args:
        state: State after the previous candidate exhausted repairs.
    """
    settings = get_settings()
    spec = _spec(state)
    next_idx = int(state.get("candidate_idx") or 1) + 1
    temperature = candidate_temperature(next_idx)
    selected = spec.select_prompts(
        state["question"],
        question_id=int(state["question_id"]),
        candidate_idx=next_idx,
        gpu_name=state.get("gpu_name") or settings.resolved_gpu_name,
        cuda_arch=state.get("cuda_arch") or settings.resolved_cuda_arch,
        cuda_version=state.get("cuda_version") or settings.resolved_cuda_version,
    )
    logger.info(
        "Q%s switching to candidate %s (system=%s suffix=%s)",
        state["question_id"],
        next_idx,
        selected.system_index,
        selected.suffix_index,
    )
    print(
        f"[Q{state['question_id']}] candidate failed, starting candidate {next_idx}",
        flush=True,
    )
    return {
        "candidate_idx": next_idx,
        "candidate_ctx": {
            "candidate": next_idx,
            "gen_system": selected.system,
            "gen_user": selected.user,
            "prompt_variant": {
                "system_index": getattr(selected, "system_index", -1),
                "suffix_index": getattr(selected, "suffix_index", -1),
                "temperature": temperature,
                "prompt_pack": "gen-v1",
            },
            "first_turn_reasoning": "",
            "first_turn_reasoning_source": "empty",
            "final_turn_reasoning": "",
            "final_turn_reasoning_source": "empty",
        },
        "selected": {},
        "repair_idx": 0,
        "oracle_retry_idx": 0,
        "oracle_blocked": False,
        "repair_cap": _repair_cap(state, settings),
        "temperature": temperature,
        "system_prompt": selected.system,
        "user_prompt": selected.user,
        "messages": [{"role": "user", "content": selected.user}],
        "raw_response": "",
        "code": "",
        "compile_ok": False,
        "compile_error": "",
        "used_rdc": False,
        "refval_ok": True,
        "refval_status": "skip",
        "refval_error": "",
        "refval_error_class": "",
        "refval_owner": "",
        "judge_score": 0,
        "judge_issues": [],
        "judge_suggestions": [],
        "critic_pass": True,
        "critic_skipped": True,
        "critic_must_fix": [],
        "critic_issues": [],
        "last_gate": {},
        "metadata": {
            key: value for key, value in (state.get("metadata") or {}).items()
            if key not in {"refval", "critic", "judge", "cot"}
        },
        "raw_reasoning": "",
        "reasoning_source": "empty",
        "origin": "unknown",
        "cot": "",
        "cot_source": "empty",
        "cot_error": "",
        "winner_found": False,
        "winner_candidate": 0,
    }


def collect_candidate(state: GraphState) -> dict[str, Any]:
    """Bank the current candidate version without replacing a better version."""
    cancel_speculative(list(state.get("speculative_requests") or []))
    if state.get("oracle_blocked"):
        reason = "oracle_blocked"
    elif not state.get("compile_ok") or not state.get("refval_ok", True):
        reason = "repair_exhausted"
    else:
        reason = "final"
    return {**_bank_candidate(state, reason), "speculative_requests": []}


def bank_pre_repair(state: GraphState) -> dict[str, Any]:
    """Keep a compile/refval-passing version before semantic repair."""
    return _bank_candidate(state, "pre_semantic_repair")


def _bank_candidate(state: GraphState, reason: str) -> dict[str, Any]:
    snap = build_snapshot(state, banked_reason=reason)
    reports = merge_into_pool(list(state.get("candidate_reports") or []), snap)
    metadata = dict(state.get("metadata") or {})
    metadata["candidate_pool"] = {
        "count": len(reports),
        "reports": [snapshot_for_metadata(item) for item in reports],
    }
    quality = dict(state.get("quality_status") or {})
    quality.update(
        {
            "contract": "skip",
            "compile": "pass" if state.get("compile_ok") else "fail",
            "refval": str(state.get("refval_status") or "skip"),
            "static_safety": "pass" if not state.get("judge_issues") else "review",
            "semantic": "pass" if state.get("critic_pass", True) else "fail",
            "release_tier": "candidate" if state.get("compile_ok") else "quarantine",
        }
    )
    return {"candidate_reports": reports, "metadata": metadata, "quality_status": quality}


def route_after_collect(
    state: GraphState,
) -> Literal["next_candidate", "select_best"]:
    """Continue filling the pool, then rank all candidates."""
    settings = get_settings()
    if settings.kernel_fast_mode and any(
        eligible(item, settings) and item["refval"].get("status") == "pass"
        for item in (state.get("candidate_reports") or [])
    ):
        return "select_best"
    cap = int(state.get("candidate_cap") or settings.max_candidates)
    if int(state.get("candidate_idx") or 1) < cap:
        return "next_candidate"
    return "select_best"


def select_best(state: GraphState) -> dict[str, Any]:
    """Select the highest-quality hard-gate candidate from the pool."""
    settings = get_settings()
    reports = list(state.get("candidate_reports") or [])
    candidates = [item for item in reports if eligible(item, settings)]
    if not candidates:
        return {
            "status": "abandoned",
            "abandon_reason": "oracle_unavailable"
            if any(item.get("oracle_blocked") for item in reports)
            else "all_candidates_failed",
        }

    chosen = max(candidates, key=rank_key)
    selected = copy.deepcopy(chosen)
    selected_id = selected["candidate"]
    refval = copy.deepcopy(selected["refval"])
    critic = copy.deepcopy(selected["critic"])
    judge_report = copy.deepcopy(selected["judge"])
    refval_status = str(refval.get("status") or "skip")
    critic_status = str(critic.get("status") or "skipped")
    reasoning, reasoning_source, cot_mode = pick_reasoning(selected, settings)
    metadata = dict(state.get("metadata") or {})
    pool = dict(metadata.get("candidate_pool") or {})
    pool["selected_candidate"] = selected_id
    pool["selected_rank"] = 1
    pool["eligible_count"] = len(candidates)
    pool["count"] = len(reports)
    pool["reports"] = [snapshot_for_metadata(item) for item in reports]
    metadata["candidate_pool"] = pool
    metadata["refval"] = refval
    metadata["critic"] = critic
    metadata["judge"] = judge_report
    quality = dict(state.get("quality_status") or {})
    quality.update(
        contract=selected["contract"],
        compile="pass",
        refval=refval_status,
        semantic="fail" if critic_status == "failed" else "pass",
        release_tier="strict" if refval_status == "pass" else "compile_only",
    )
    quality["selected_candidate"] = selected_id
    provenance = dict(state.get("provenance") or {})
    provenance.update(
        provider=selected.get("provider") or provenance.get("provider") or settings.llm_provider,
        model=selected.get("model") or provenance.get("model") or settings.resolved_model,
    )
    return {
        "selected": selected,
        "candidate_idx": selected_id,
        "repair_idx": selected["repairs"],
        "code": selected["code"],
        "system_prompt": selected["gen_system"],
        "user_prompt": selected["gen_user"],
        "raw_reasoning": reasoning,
        "reasoning_source": reasoning_source,
        "origin": selected["origin"],
        "cot_mode": cot_mode,
        "compile_error": "",
        "compile_ok": True,
        "used_rdc": bool(selected.get("used_rdc", False)),
        "refval_ok": refval_status != "fail",
        "refval_status": refval_status,
        "refval_error": str(refval.get("evidence") or refval.get("reason") or ""),
        "refval_error_class": str(refval.get("error_class") or ""),
        "refval_owner": selected["refval_owner"],
        "oracle_blocked": selected["oracle_blocked"],
        "judge_score": int(judge_report.get("quality_score") or 0),
        "judge_issues": list(judge_report.get("issues") or []),
        "judge_suggestions": list(judge_report.get("suggestions") or []),
        "critic_pass": bool(critic.get("passed", critic_status in {"verified", "skipped"})),
        "critic_skipped": critic_status == "skipped",
        "critic_issues": list(critic.get("issues") or []),
        "critic_must_fix": list(critic.get("must_fix") or []),
        "last_gate": selected["last_gate"],
        "metadata": metadata,
        "quality_status": quality,
        "provenance": provenance,
        "winner_found": True,
        "winner_candidate": selected_id,
        "status": "running",
        "abandon_reason": "",
    }


def save_success(state: GraphState) -> dict[str, Any]:
    """Write SFT jsonl rows and mark the question successful."""
    settings = get_settings()
    spec = _spec(state)
    nest = get_dialect_agent().nest_workdir(spec.name, settings)
    model = str((state.get("provenance") or {}).get("model") or settings.resolved_model)
    saved = get_store().write_success(state, model_name=model)
    finalize_question_work(
        settings,
        int(state["question_id"]),
        code=state.get("code") or "",
        success=saved,
        dialect=spec.name,
        filename=spec.source_filename,
        nest_dialect=nest,
        candidate_idx=int(state.get("candidate_idx") or 1),
        repair_idx=int(state.get("repair_idx") or 0),
    )
    if not saved:
        return {"status": "abandoned", "abandon_reason": "strict_quality_gate"}
    logger.info(
        "Q%s %s saved SFT sample (candidate=%s repairs=%s judge_score=%s cot=%s)",
        state["question_id"],
        _dialect_name(state),
        state.get("candidate_idx", 1),
        state.get("repair_idx", 0),
        state.get("judge_score", 0),
        state.get("cot_source", "empty"),
    )
    return {"status": "success"}


def judge(state: GraphState) -> dict[str, Any]:
    """Evaluate code quality and collect metrics.

    Args:
        state: Compiled state with ``code``.

    Returns:
        Updated state with judge_score, issues, and suggestions.
    """
    settings = get_settings()

    if settings.async_llm_enabled:
        cancel_speculative(list(state.get("speculative_requests") or []))

    if not settings.judge_enabled:
        return {
            "judge_score": 0,
            "judge_issues": [],
            "judge_suggestions": [],
            "speculative_requests": [],
            "winner_found": False,
            "winner_candidate": 0,
        }

    spec = _spec(state)
    code = state.get("code") or ""
    result = spec.judge(code)
    score = int(result.quality_score)
    if str(state.get("refval_status") or "") == "pass":
        score = min(10, score + 1)

    logger.info(
        "Q%s judge: score=%s issues=%s suggestions=%s",
        state["question_id"],
        score,
        len(result.issues),
        len(result.suggestions),
    )
    print(
        f"[Q{state['question_id']}] judge: score={score}/10 "
        f"issues={len(result.issues)} suggestions={len(result.suggestions)}",
        flush=True,
    )

    metadata = dict(state.get("metadata") or {})
    metadata["judge"] = {
        "quality_score": score,
        "issues": result.issues,
        "suggestions": result.suggestions,
        "refval_bonus": bool(str(state.get("refval_status") or "") == "pass"),
    }
    return {
        "judge_score": score,
        "judge_issues": result.issues,
        "judge_suggestions": result.suggestions,
        "metadata": metadata,
        "speculative_requests": [],
        # A candidate is only a winner after the pool selector runs.
        "winner_found": False,
        "winner_candidate": 0,
    }


def critic(state: GraphState) -> dict[str, Any]:
    """Optional semantic critic after a compile-passing heuristic judge.

    Does not abandon by default. ``must_fix`` with remaining repairs routes
    back to the Repairer; otherwise the sample still proceeds to CoT.
    """
    settings = get_settings()
    heuristic = JudgeResult(
        quality_score=int(state.get("judge_score") or 0),
        issues=list(state.get("judge_issues") or []),
        suggestions=list(state.get("judge_suggestions") or []),
    )
    result = KernelCritic(settings).evaluate(
        question=str(state.get("question") or ""),
        code=str(state.get("code") or ""),
        dialect=_dialect_name(state),
        heuristic=heuristic,
        use_critic=bool(state.get("use_critic", True)),
        refval_status=str(state.get("refval_status") or ""),
        refval_error_class=str(state.get("refval_error_class") or ""),
        refval_summary=str(state.get("refval_error") or ""),
        meta=CallMeta(
            role="critic",
            job_key=legacy_job_key(int(state["question_id"]), _dialect_name(state)),
            question_id=int(state["question_id"]),
            track=_dialect_name(state),
            candidate=int(state.get("candidate_idx") or 1),
            repair=int(state.get("repair_idx") or 0),
        ),
    )
    metadata = dict(state.get("metadata") or {})
    metadata["critic"] = critic_result_to_dict(result)
    print(
        f"[Q{state['question_id']}] critic: pass={result.passed} "
        f"skipped={result.skipped} must_fix={len(result.must_fix)}",
        flush=True,
    )
    return {
        "critic_pass": result.passed,
        "critic_skipped": result.skipped,
        "critic_must_fix": result.must_fix,
        "critic_issues": result.issues,
        "metadata": metadata,
    }


def cot(state: GraphState) -> dict[str, Any]:
    """Polish teacher thinking into SFT CoT after a compile-passing sample.

    Args:
        state: Winning state; uses ``raw_reasoning`` and ``code``.
    """
    settings = get_settings()
    if not settings.cot_enabled:
        return {
            "cot": "",
            "cot_source": "empty",
            "cot_error": "",
        }

    cot_settings = (
        settings.model_copy(update={"cot_agent_enabled": False})
        if settings.kernel_fast_mode else settings
    )
    result = CotAgent(cot_settings).refine(state)
    metadata = dict(state.get("metadata") or {})
    metadata["cot"] = {
        "source": result.source,
        "text": result.cot,
        "policy": settings.cot_repaired_policy,
        "mode": state.get("cot_mode") or "polish",
        "chars": len(result.cot or ""),
        "reasoning_source": state.get("reasoning_source") or "",
        "raw_chars": len(result.raw_reasoning or ""),
        "polished_chars": len(result.cot or ""),
        "error": result.error,
    }
    if settings.cot_consistency_check:
        metadata["cot"]["consistency_issues"] = list(result.consistency_issues)
        metadata["cot"]["soft_issues"] = list(result.soft_issues)
    logger.info(
        "Q%s cot: source=%s raw=%s polished=%s",
        state["question_id"],
        result.source,
        len(result.raw_reasoning or ""),
        len(result.cot or ""),
    )
    print(
        f"[Q{state['question_id']}] cot: source={result.source} "
        f"chars={len(result.cot or '')}",
        flush=True,
    )
    return {
        "cot": result.cot,
        "cot_source": result.source,
        "cot_error": result.error,
        "raw_reasoning": result.raw_reasoning,
        "metadata": metadata,
    }


def save_abandoned(state: GraphState) -> dict[str, Any]:
    """Write the abandoned record after all candidates failed."""
    settings = get_settings()
    spec = _spec(state)
    nest = get_dialect_agent().nest_workdir(spec.name, settings)
    get_store().write_abandoned(state)
    finalize_question_work(
        settings,
        int(state["question_id"]),
        code=state.get("code") or "",
        success=False,
        dialect=spec.name,
        filename=spec.source_filename,
        nest_dialect=nest,
        candidate_idx=int(state.get("candidate_idx") or 1),
        repair_idx=int(state.get("repair_idx") or 0),
    )
    logger.info("Q%s abandoned after all candidates failed", state["question_id"])
    print(f"[Q{state['question_id']}] abandoned", flush=True)
    return {"status": "abandoned"}


def retry_oracle(state: GraphState) -> dict[str, Any]:
    """Retry the same gate without changing the kernel or repair budget."""
    idx = int(state.get("oracle_retry_idx") or 0)
    settings = get_settings()
    if settings.kernel_fast_mode or idx >= settings.oracle_retry_max:
        return {"oracle_blocked": True}
    backoffs = [float(value) for value in settings.oracle_retry_backoff_s.split(",") if value.strip()]
    wait = backoffs[min(idx, len(backoffs) - 1)] if backoffs else 0.0
    deps.sleep(wait)
    return {"oracle_retry_idx": idx + 1, "oracle_blocked": False}


def route_after_retry_oracle(state: GraphState) -> Literal["collect_candidate", "validate", "compile"]:
    if state.get("oracle_blocked"):
        return "collect_candidate"
    return "compile" if (state.get("last_gate") or {}).get("gate") == "compile" else "validate"


def route_after_compile(
    state: GraphState,
) -> Literal["validate", "retry_oracle", "repair", "collect_candidate"]:
    """Route after compile: validate (if ok), repair, next candidate, or abandon."""
    settings = get_settings()
    if state.get("compile_ok"):
        return "validate"
    if (state.get("last_gate") or {}).get("owner") == FailureOwner.INFRA.value:
        return "retry_oracle"
    repair_cap = _repair_cap(state, settings)
    if int(state.get("repair_idx") or 0) < repair_cap:
        return "repair"
    return "collect_candidate"


def route_after_validate(
    state: GraphState,
) -> Literal["judge", "retry_oracle", "repair", "collect_candidate"]:
    """Numeric fail uses the compile repair/candidate ladder; skip still judges."""
    settings = get_settings()
    owner = (state.get("last_gate") or {}).get("owner")
    if state.get("oracle_blocked"):
        return "collect_candidate"
    if owner in {FailureOwner.ORACLE.value, FailureOwner.INFRA.value}:
        return "retry_oracle"
    if state.get("refval_ok", True) or ((state.get("last_gate") or {}) and owner is None):
        return "judge"
    repair_cap = _repair_cap(state, settings)
    if int(state.get("repair_idx") or 0) < repair_cap:
        return "repair"
    return "collect_candidate"


def route_after_critic(state: GraphState) -> Literal["collect_candidate", "bank_pre_repair"]:
    """Save a checked version before acting on a critic rejection.

    Only a rejection with concrete ``must_fix`` items is repaired. A bare
    ``pass=false`` gives the repairer nothing to act on and would rewrite a
    kernel that already passed compile and numeric validation.
    """
    settings = get_settings()
    status = str(((state.get("metadata") or {}).get("critic") or {}).get("status") or "skipped")
    actionable = any(str(item).strip() for item in state.get("critic_must_fix") or [])
    if (
        status == "failed"
        and actionable
        and int(state.get("repair_idx") or 0) < _repair_cap(state, settings)
    ):
        return "bank_pre_repair"
    return "collect_candidate"


def route_after_repair(state: GraphState) -> Literal["generate", "next_candidate"]:
    """Skip remaining repairs when another candidate already compiled."""
    if state.get("skip_repair"):
        return "next_candidate"
    return "generate"


def route_after_select(state: GraphState) -> Literal["cot", "save_abandoned"]:
    """Only send a hard-gate winner to CoT; otherwise quarantine the job."""
    return "cot" if state.get("winner_found") else "save_abandoned"


def _build_graph(*, candidate_only: bool):
    """Compile the shared kernel nodes for a whole question or one candidate."""
    builder = StateGraph(GraphState)
    builder.add_node("prepare", traced("prepare")(prepare))
    builder.add_node("generate", traced("generate")(generate), retry_policy=retry_policy())
    builder.add_node("extract", traced("extract")(extract))
    builder.add_node("compile", traced("compile")(compile_node))
    builder.add_node("validate", traced("validate")(validate))
    builder.add_node("retry_oracle", traced("retry_oracle")(retry_oracle))
    builder.add_node("repair", traced("repair")(repair))
    builder.add_node("judge", traced("judge")(judge))
    builder.add_node("critic", traced("critic")(critic))
    builder.add_node("bank_pre_repair", traced("bank_pre_repair")(bank_pre_repair))
    builder.add_node("collect_candidate", traced("collect_candidate")(collect_candidate))
    if not candidate_only:
        builder.add_node("next_candidate", traced("next_candidate")(next_candidate))
        builder.add_node("select_best", traced("select_best")(select_best))
        builder.add_node("cot", traced("cot")(cot))
        builder.add_node("save_success", traced("save_success")(save_success))
        builder.add_node("save_abandoned", traced("save_abandoned")(save_abandoned))

    builder.add_edge(START, "prepare")
    builder.add_edge("prepare", "generate")
    builder.add_edge("generate", "extract")
    builder.add_edge("extract", "compile")
    builder.add_conditional_edges(
        "compile",
        route_after_compile,
        {
            "validate": "validate",
            "retry_oracle": "retry_oracle",
            "repair": "repair",
            "collect_candidate": "collect_candidate",
        },
    )
    builder.add_conditional_edges(
        "validate",
        route_after_validate,
        {
            "judge": "judge",
            "retry_oracle": "retry_oracle",
            "repair": "repair",
            "collect_candidate": "collect_candidate",
        },
    )
    builder.add_conditional_edges(
        "retry_oracle",
        route_after_retry_oracle,
        {"collect_candidate": "collect_candidate", "validate": "validate", "compile": "compile"},
    )
    builder.add_conditional_edges(
        "repair",
        route_after_repair,
        {
            "generate": "generate",
            "next_candidate": "collect_candidate" if candidate_only else "next_candidate",
        },
    )
    if not candidate_only:
        builder.add_edge("next_candidate", "generate")
    builder.add_edge("judge", "critic")
    builder.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "collect_candidate": "collect_candidate",
            "bank_pre_repair": "bank_pre_repair",
        },
    )
    builder.add_edge("bank_pre_repair", "repair")
    if candidate_only:
        builder.add_edge("collect_candidate", END)
    else:
        builder.add_conditional_edges(
            "collect_candidate",
            route_after_collect,
            {
                "next_candidate": "next_candidate",
                "select_best": "select_best",
            },
        )
    if not candidate_only:
        builder.add_conditional_edges(
            "select_best",
            route_after_select,
            {"cot": "cot", "save_abandoned": "save_abandoned"},
        )
        builder.add_edge("cot", "save_success")
        builder.add_edge("save_success", END)
        builder.add_edge("save_abandoned", END)
    return builder.compile()


def build_graph():
    """Compile the existing per-question graph, including selection and save."""
    return _build_graph(candidate_only=False)


def build_candidate_graph():
    """Compile one candidate without selecting or writing the question result."""
    return _build_graph(candidate_only=True)


def recursion_limit() -> int:
    """LangGraph cap covering the largest configured candidate/repair ladder."""
    settings = get_settings()
    return graph_recursion_limit(
        max_candidates=settings.max_candidates,
        max_repairs=settings.max_repairs,
        extra_per_candidate=7 + 2 * settings.oracle_retry_max,
    )
