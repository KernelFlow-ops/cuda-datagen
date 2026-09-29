"""LangGraph: generate → extract → hard gate / judge → repair → cot / save.

Shares generate-node LLM calls and print-stream state with the kernel graph.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from cuda_sft.agents.difficulty import plan_topology
from cuda_sft.agents.generate import assistant_state_update, complete_chat
from cuda_sft.agents.repairer import repair_system_prompt, wrap_repair_user
from cuda_sft.config import get_settings
from cuda_sft.knowledge.agent import get_knowledge_agent
from cuda_sft.knowledge.cot import KnowledgeCotAgent
from cuda_sft.knowledge.judge import KnowledgeJudge, accepted_knowledge_answer, hard_gate
from cuda_sft.knowledge.parse import extract_answer
from cuda_sft.knowledge.prompt import (
    SYSTEM_PROMPTS,
    TOPIC_HINTS,
    build_repair_prompt,
    candidate_temperature,
    select_prompts,
)
from cuda_sft.knowledge.state import KnowledgeGraphState
from cuda_sft.pipeline.common import graph_recursion_limit, retry_policy
from cuda_sft.pipeline.common import set_print_stream as set_print_stream
from cuda_sft.runtime.meta import CallMeta, legacy_job_key
from cuda_sft.runtime.trace import traced
from cuda_sft.store import get_store
from cuda_sft.tasks.kinds import knowledge_track

logger = logging.getLogger(__name__)


def _topic(state: KnowledgeGraphState) -> str:
    """Return the knowledge topic id stored on this job.

    Args:
        state: Graph state; missing topic becomes ``general``.
    """
    return str(state.get("topic") or "general")


def _repair_cap(state: KnowledgeGraphState, settings: Any) -> int:
    value = state.get("repair_cap")
    if value is not None:
        return int(value)
    return plan_topology(
        question=str(state.get("question") or ""),
        kind="knowledge",
        topic=_topic(state),
        settings=settings,
    ).max_repairs


def _write_attempt(state: KnowledgeGraphState, answer: str) -> None:
    """Persist the current draft under work/q{id}/knowledge/{topic}/.

    Args:
        state: Job identifiers plus candidate/repair indices.
        answer: Extracted prose (may be empty).
    """
    settings = get_settings()
    qid = int(state["question_id"])
    topic = _topic(state)
    cand = int(state.get("candidate_idx") or 1)
    repair = int(state.get("repair_idx") or 0)
    if settings.work_keep == "detailed":
        dest = (
            settings.work_path
            / f"q{qid}"
            / "knowledge"
            / topic
            / f"c{cand}"
            / f"r{repair}"
        )
    else:
        dest = settings.work_path / f"q{qid}" / "knowledge" / topic
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "answer.md").write_text(answer or "", encoding="utf-8")


def _finalize_answer(state: KnowledgeGraphState, *, success: bool) -> None:
    settings = get_settings()
    qid = int(state["question_id"])
    topic = _topic(state)
    dest = settings.work_path / f"q{qid}" / "knowledge" / topic
    dest.mkdir(parents=True, exist_ok=True)
    body = state.get("answer") or ""
    name = "answer.md" if success else "abandoned.md"
    (dest / name).write_text(body, encoding="utf-8")


def prepare(state: KnowledgeGraphState) -> dict[str, Any]:
    """Initialize prompts and counters for one knowledge question."""
    settings = get_settings()
    topic = _topic(state)
    selected = select_prompts(
        state["question"],
        question_id=int(state["question_id"]),
        candidate_idx=1,
        topic=topic,
        gpu_name=settings.resolved_gpu_name,
        cuda_arch=settings.resolved_cuda_arch,
        cuda_version=settings.resolved_cuda_version,
    )
    topo = plan_topology(
        question=state["question"],
        kind="knowledge",
        topic=topic,
        settings=settings,
    )
    return {
        "kind": "knowledge",
        "topic": topic,
        "track": str(state.get("track") or knowledge_track(topic)),
        "system_prompt": selected.system,
        "user_prompt": selected.user,
        "gen_system": selected.system,
        "gen_user": selected.user,
        "gen_prompt_variant": {
            "system_index": selected.system_index,
            "suffix_index": selected.suffix_index,
            "temperature": candidate_temperature(1),
            "prompt_pack": "knowledge-gen-v1",
        },
        "messages": [{"role": "user", "content": selected.user}],
        "candidate_idx": 1,
        "repair_idx": 0,
        "temperature": candidate_temperature(1),
        "raw_response": "",
        "origin": "unknown",
        "answer": "",
        "gate_ok": False,
        "gate_reasons": [],
        "judge_pass": False,
        "judge_score": 0.0,
        "judge_issues": [],
        "judge_must_fix": [],
        "judge_dimensions": {},
        "judge_error": "",
        "judge_unavailable": False,
        "judge_skipped_llm": False,
        "status": "running",
        "abandon_reason": "",
        "attempts": [],
        "gpu_name": settings.resolved_gpu_name,
        "cuda_arch": settings.resolved_cuda_arch,
        "cuda_version": settings.resolved_cuda_version,
        "metadata": {},
        "raw_reasoning": "",
        "reasoning_source": "empty",
        "cot": "",
        "cot_source": "empty",
        "cot_error": "",
        "difficulty": topo.difficulty,
        "candidate_cap": topo.max_candidates,
        "repair_cap": topo.max_repairs,
    }


def generate(state: KnowledgeGraphState) -> dict[str, Any]:
    """Call the LLM and append the assistant turn."""
    qid = state["question_id"]
    cand = state.get("candidate_idx", 1)
    repair = state.get("repair_idx", 0)
    temperature = float(state.get("temperature") or candidate_temperature(cand))
    topic = _topic(state)
    header = f"[Q{qid} knowledge/{topic} candidate={cand} repair={repair} temp={temperature}]"
    settings = get_settings()
    role = "knowledge_generator" if not repair else "knowledge_repair"
    result = complete_chat(
        messages=list(state.get("messages") or []),
        system=state.get("system_prompt") or SYSTEM_PROMPTS[0],
        temperature=temperature,
        llm_options=get_knowledge_agent().llm_call_options(settings, role=role),
        log_header=header,
        meta=CallMeta(
            role=role,
            job_key=legacy_job_key(int(qid), str(state.get("track") or f"knowledge:{topic}")),
            question_id=int(qid),
            track=str(state.get("track") or f"knowledge:{topic}"),
            candidate=int(cand),
            repair=int(repair),
        ),
    )
    update = assistant_state_update(state, result, log_header=header)
    route = settings.for_role(role)
    provenance = dict(state.get("provenance") or {})
    provenance.update(provider=route.llm_provider, model=route.resolved_model)
    update["provenance"] = provenance
    update["origin"] = getattr(result, "origin", "unknown")
    return update


def extract(state: KnowledgeGraphState) -> dict[str, Any]:
    """Pull visible prose out of the last model reply (keep formulas)."""
    return _extract(state, persist_attempt=True)


def extract_compute(state: KnowledgeGraphState) -> dict[str, Any]:
    """Extract a draft without writing child-process work files."""
    return _extract(state, persist_attempt=False)


def _extract(state: KnowledgeGraphState, *, persist_attempt: bool) -> dict[str, Any]:
    answer = extract_answer(state.get("raw_response") or "")
    if not answer.strip():
        logger.warning("Q%s knowledge: no answer extracted", state["question_id"])
    if persist_attempt:
        _write_attempt(state, answer)
    # Scores belong to the previous draft; a hard-gate failure on this draft
    # must not show them in the next repair prompt.
    return {
        "answer": answer,
        "judge_must_fix": [],
        "judge_issues": [],
        "judge_score": 0.0,
        "judge_dimensions": {},
        "judge_error": "",
        "judge_unavailable": False,
    }


def gate(state: KnowledgeGraphState) -> dict[str, Any]:
    """Deterministic hard gate (no LLM)."""
    settings = get_settings()
    reasons = hard_gate(
        state.get("answer") or "",
        topic=_topic(state),
        question=str(state.get("question") or ""),
        min_chars=settings.knowledge_min_answer_chars,
        require_structure=settings.knowledge_require_structure,
    )
    ok = not reasons
    attempts = list(state.get("attempts") or [])
    attempts.append(
        {
            "candidate": state.get("candidate_idx", 1),
            "repair": state.get("repair_idx", 0),
            "ok": ok,
            "stage": "hard_gate",
            "error": "; ".join(reasons),
        }
    )
    status = "PASS" if ok else "FAIL"
    print(
        f"[Q{state['question_id']} knowledge] hard_gate {status}"
        + (f": {reasons[0]}" if reasons else ""),
        flush=True,
    )
    return {
        "gate_ok": ok,
        "gate_reasons": reasons,
        "attempts": attempts,
    }


def judge(state: KnowledgeGraphState) -> dict[str, Any]:
    """LLM rubric judge after a passing hard gate."""
    settings = get_settings()
    result = KnowledgeJudge(settings).judge(
        question=str(state.get("question") or ""),
        answer=str(state.get("answer") or ""),
        topic=_topic(state),
        meta=CallMeta(
            role="knowledge_judge",
            job_key=legacy_job_key(int(state["question_id"]), str(state.get("track") or "knowledge:general")),
            question_id=int(state["question_id"]),
            track=str(state.get("track") or "knowledge:general"),
            candidate=int(state.get("candidate_idx") or 1),
            repair=int(state.get("repair_idx") or 0),
            purpose="rubric",
        ),
    )
    metadata = dict(state.get("metadata") or {})
    metadata["knowledge_judge"] = {
        "answer_sha256": hashlib.sha256(str(state.get("answer") or "").encode("utf-8")).hexdigest(),
        "pass": result.passed,
        "overall": result.overall,
        "dimensions": result.dimensions,
        "must_fix": result.must_fix,
        "issues": result.issues,
        "hard_gate_failed": result.hard_gate_failed,
        "hard_gate_reasons": result.hard_gate_reasons,
        "judge_error": result.judge_error,
        "skipped_llm": result.skipped_llm,
        "unavailable": result.judge_unavailable,
        "topic": _topic(state),
    }
    logger.info(
        "Q%s knowledge judge: pass=%s overall=%s unavailable=%s",
        state["question_id"],
        result.passed,
        result.overall,
        result.judge_unavailable,
    )
    print(
        f"[Q{state['question_id']} knowledge] judge: pass={result.passed} "
        f"overall={result.overall:.1f}/10 unavailable={result.judge_unavailable}",
        flush=True,
    )
    attempts = list(state.get("attempts") or [])
    attempts.append(
        {
            "candidate": state.get("candidate_idx", 1),
            "repair": state.get("repair_idx", 0),
            "ok": result.passed,
            "stage": "judge",
            "error": result.judge_error or "; ".join(result.must_fix or result.issues),
        }
    )
    return {
        "judge_pass": result.passed,
        "judge_score": result.overall,
        "judge_issues": result.issues,
        "judge_must_fix": result.must_fix,
        "judge_dimensions": result.dimensions,
        "judge_error": result.judge_error,
        "judge_unavailable": result.judge_unavailable,
        "judge_skipped_llm": result.skipped_llm,
        "metadata": metadata,
        "attempts": attempts,
    }


def _generation_requirements(state: KnowledgeGraphState) -> str:
    """Topic hint and answer rules from the generation prompt, without the question.

    A single-turn repair does not resend the generation turn, so these would
    otherwise be lost.
    """
    gen_user = str(state.get("gen_user") or "").strip()
    question = str(state.get("question") or "").strip()
    if question and gen_user.startswith(question):
        return gen_user[len(question):].strip()
    return TOPIC_HINTS.get(_topic(state), TOPIC_HINTS["general"])


def repair(state: KnowledgeGraphState) -> dict[str, Any]:
    """Append a critique-and-rewrite user message and bump repair_idx."""
    settings = get_settings()
    next_repair = int(state.get("repair_idx") or 0) + 1
    inner = build_repair_prompt(
        topic=_topic(state),
        answer=str(state.get("answer") or ""),
        gate_reasons=list(state.get("gate_reasons") or []),
        must_fix=list(state.get("judge_must_fix") or []),
        issues=list(state.get("judge_issues") or []),
        judge_score=float(state.get("judge_score") or 0),
        judge_dimensions=dict(state.get("judge_dimensions") or {}),
    )
    single_turn = settings.repair_history_mode != "full"
    if single_turn:
        requirements = _generation_requirements(state)
        if requirements:
            inner = f"{inner}\n\n## Answer requirements (from the original brief)\n{requirements}"
    repair_user = wrap_repair_user(
        question=str(state.get("question") or ""),
        inner=inner,
        error_class="knowledge_quality",
        dialect="knowledge",
    )
    if single_turn:
        messages = [{"role": "user", "content": repair_user}]
    else:
        messages = [*(state.get("messages") or []), {"role": "user", "content": repair_user}]
    logger.info(
        "Q%s knowledge candidate %s starting repair %s/%s",
        state["question_id"],
        state.get("candidate_idx", 1),
        next_repair,
        _repair_cap(state, settings),
    )
    return {
        "repair_idx": next_repair,
        "messages": messages,
        "system_prompt": repair_system_prompt("knowledge"),
        "raw_response": "",
        "origin": "unknown",
        "gate_ok": False,
        "judge_pass": False,
        "judge_error": "",
        "judge_unavailable": False,
    }


def next_candidate(state: KnowledgeGraphState) -> dict[str, Any]:
    """Start a fresh candidate with a different prompt variant."""
    settings = get_settings()
    next_idx = int(state.get("candidate_idx") or 1) + 1
    temperature = candidate_temperature(next_idx)
    topic = _topic(state)
    selected = select_prompts(
        state["question"],
        question_id=int(state["question_id"]),
        candidate_idx=next_idx,
        topic=topic,
        gpu_name=state.get("gpu_name") or settings.resolved_gpu_name,
        cuda_arch=state.get("cuda_arch") or settings.resolved_cuda_arch,
        cuda_version=state.get("cuda_version") or settings.resolved_cuda_version,
    )
    logger.info(
        "Q%s knowledge switching to candidate %s (system=%s suffix=%s)",
        state["question_id"],
        next_idx,
        selected.system_index,
        selected.suffix_index,
    )
    print(
        f"[Q{state['question_id']} knowledge] candidate failed, starting candidate {next_idx}",
        flush=True,
    )
    return {
        "candidate_idx": next_idx,
        "repair_idx": 0,
        "temperature": temperature,
        "system_prompt": selected.system,
        "user_prompt": selected.user,
        "gen_system": selected.system,
        "gen_user": selected.user,
        "gen_prompt_variant": {
            "system_index": selected.system_index,
            "suffix_index": selected.suffix_index,
            "temperature": temperature,
            "prompt_pack": "knowledge-gen-v1",
        },
        "messages": [{"role": "user", "content": selected.user}],
        "raw_response": "",
        "origin": "unknown",
        "answer": "",
        "gate_ok": False,
        "gate_reasons": [],
        "judge_pass": False,
        "judge_score": 0.0,
        "judge_issues": [],
        "judge_must_fix": [],
        "judge_dimensions": {},
        "judge_error": "",
        "judge_unavailable": False,
        "raw_reasoning": "",
        "reasoning_source": "empty",
        "cot": "",
        "cot_source": "empty",
        "cot_error": "",
    }


def cot(state: KnowledgeGraphState) -> dict[str, Any]:
    """Polish teacher thinking into SFT CoT after a passing knowledge sample."""
    settings = get_settings()
    if not settings.cot_enabled:
        return {"cot": "", "cot_source": "empty", "cot_error": ""}

    result = KnowledgeCotAgent(settings).refine(state)
    metadata = dict(state.get("metadata") or {})
    metadata["cot"] = {
        "source": result.source,
        "text": result.cot,
        "reasoning_source": state.get("reasoning_source") or "",
        "raw_chars": len(result.raw_reasoning or ""),
        "polished_chars": len(result.cot or ""),
        "error": result.error,
    }
    logger.info(
        "Q%s knowledge cot: source=%s raw=%s polished=%s",
        state["question_id"],
        result.source,
        len(result.raw_reasoning or ""),
        len(result.cot or ""),
    )
    print(
        f"[Q{state['question_id']} knowledge] cot: source={result.source} "
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


def save_success(state: KnowledgeGraphState) -> dict[str, Any]:
    """Write SFT jsonl rows for a quality-gated knowledge answer."""
    settings = get_settings()
    if not accepted_knowledge_answer(state, settings):
        raise ValueError("knowledge answer lacks matching passing judge evidence")
    model = str((state.get("provenance") or {}).get("model") or settings.resolved_model)
    get_store().write_success(state, model_name=model)
    _finalize_answer(state, success=True)
    logger.info(
        "Q%s knowledge saved SFT sample (candidate=%s repairs=%s score=%s cot=%s)",
        state["question_id"],
        state.get("candidate_idx", 1),
        state.get("repair_idx", 0),
        state.get("judge_score", 0),
        state.get("cot_source", "empty"),
    )
    return finish_success(state)


def finish_success(state: KnowledgeGraphState) -> dict[str, Any]:
    """Return a successful result for parent-process persistence."""
    if not accepted_knowledge_answer(state, get_settings()):
        raise ValueError("knowledge answer lacks matching passing judge evidence")
    return {"status": "success"}


def _abandon_reason(state: KnowledgeGraphState) -> str:
    if state.get("judge_unavailable"):
        return "knowledge_judge_unavailable"
    if state.get("judge_pass") and not accepted_knowledge_answer(state, get_settings()):
        return "knowledge_judge_unverified"
    return "knowledge_quality"


def finish_abandoned(state: KnowledgeGraphState) -> dict[str, Any]:
    """Return an abandoned result for parent-process persistence."""
    return {"status": "abandoned", "abandon_reason": _abandon_reason(state)}


def save_abandoned(state: KnowledgeGraphState) -> dict[str, Any]:
    """Write the abandoned record after all candidates failed the rubric."""
    reason = _abandon_reason(state)
    payload = dict(state)
    payload["abandon_reason"] = reason
    get_store().write_abandoned(payload)  # type: ignore[arg-type]
    _finalize_answer(state, success=False)
    logger.info("Q%s knowledge abandoned (%s)", state["question_id"], reason)
    print(f"[Q{state['question_id']} knowledge] abandoned ({reason})", flush=True)
    return finish_abandoned(state)


def route_after_gate(
    state: KnowledgeGraphState,
) -> Literal["judge", "repair", "next_candidate", "save_abandoned"]:
    """Route after the hard gate."""
    settings = get_settings()
    cap = int(state.get("candidate_cap") or settings.knowledge_max_candidates)
    if state.get("gate_ok"):
        return "judge"
    if int(state.get("repair_idx") or 0) < _repair_cap(state, settings):
        return "repair"
    if int(state.get("candidate_idx") or 1) < cap:
        return "next_candidate"
    return "save_abandoned"


def route_after_judge(
    state: KnowledgeGraphState,
) -> Literal["cot", "repair", "next_candidate", "save_abandoned"]:
    """Route after the LLM judge. JSON failure abandons rather than saving."""
    settings = get_settings()
    cap = int(state.get("candidate_cap") or settings.knowledge_max_candidates)
    if state.get("judge_unavailable"):
        return "save_abandoned"
    if state.get("judge_pass"):
        return "cot" if accepted_knowledge_answer(state, settings) else "save_abandoned"
    if int(state.get("repair_idx") or 0) < _repair_cap(state, settings):
        return "repair"
    if int(state.get("candidate_idx") or 1) < cap:
        return "next_candidate"
    return "save_abandoned"


def _build_knowledge_graph(*, persist: bool):
    builder = StateGraph(KnowledgeGraphState)
    builder.add_node("prepare", traced("prepare")(prepare))
    builder.add_node("generate", traced("generate")(generate), retry_policy=retry_policy())
    builder.add_node("extract", traced("extract")(extract if persist else extract_compute))
    builder.add_node("gate", traced("gate")(gate))
    builder.add_node("repair", traced("repair")(repair))
    builder.add_node("next_candidate", traced("next_candidate")(next_candidate))
    builder.add_node("judge", traced("judge")(judge))
    builder.add_node("cot", traced("cot")(cot))
    success_node = "save_success" if persist else "finish_success"
    abandoned_node = "save_abandoned" if persist else "finish_abandoned"
    builder.add_node(success_node, traced(success_node)(save_success if persist else finish_success))
    builder.add_node(
        abandoned_node,
        traced(abandoned_node)(save_abandoned if persist else finish_abandoned),
    )

    builder.add_edge(START, "prepare")
    builder.add_edge("prepare", "generate")
    builder.add_edge("generate", "extract")
    builder.add_edge("extract", "gate")
    builder.add_conditional_edges(
        "gate",
        route_after_gate,
        {
            "judge": "judge",
            "repair": "repair",
            "next_candidate": "next_candidate",
            "save_abandoned": abandoned_node,
        },
    )
    builder.add_edge("repair", "generate")
    builder.add_edge("next_candidate", "generate")
    builder.add_conditional_edges(
        "judge",
        route_after_judge,
        {
            "cot": "cot",
            "repair": "repair",
            "next_candidate": "next_candidate",
            "save_abandoned": abandoned_node,
        },
    )
    builder.add_edge("cot", success_node)
    builder.add_edge(success_node, END)
    builder.add_edge(abandoned_node, END)
    return builder.compile()


def build_knowledge_graph():
    """Compile the legacy graph, including sample and work-file writes."""
    return _build_knowledge_graph(persist=True)


def build_knowledge_compute_graph():
    """Compile the knowledge graph without writes; the caller persists its final state."""
    return _build_knowledge_graph(persist=False)


def knowledge_recursion_limit() -> int:
    """LangGraph superstep cap for knowledge candidates × repairs."""
    settings = get_settings()
    return graph_recursion_limit(
        max_candidates=settings.knowledge_max_candidates,
        max_repairs=settings.knowledge_max_repairs,
        extra_per_candidate=6,
    )
