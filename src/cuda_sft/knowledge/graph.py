"""LangGraph: generate → extract → hard gate / judge → repair → cot / save."""

from __future__ import annotations

import logging
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from cuda_sft.config import get_settings
from cuda_sft.knowledge.agent import get_knowledge_agent
from cuda_sft.knowledge.cot import KnowledgeCotAgent
from cuda_sft.knowledge.judge import KnowledgeJudge, hard_gate
from cuda_sft.knowledge.parse import extract_answer
from cuda_sft.knowledge.prompt import (
    SYSTEM_PROMPTS,
    build_repair_prompt,
    candidate_temperature,
    select_prompts,
)
from cuda_sft.knowledge.state import KnowledgeGraphState
from cuda_sft.llm import LLMCompletion, get_llm_client, is_retryable_llm_error
from cuda_sft.store import get_store
from cuda_sft.tasks.kinds import knowledge_track

logger = logging.getLogger(__name__)

PRINT_STREAM = True


def set_print_stream(enabled: bool) -> None:
    """Enable or disable printing streamed model tokens to stdout."""
    global PRINT_STREAM
    PRINT_STREAM = enabled


def _retry_policy():
    """RetryPolicy for transient LLM failures on the generate node."""
    from langgraph.types import RetryPolicy

    return RetryPolicy(
        max_attempts=5,
        initial_interval=4.0,
        backoff_factor=2.0,
        max_interval=60.0,
        retry_on=is_retryable_llm_error,
    )


def _topic(state: KnowledgeGraphState) -> str:
    return str(state.get("topic") or "general")


def _write_attempt(state: KnowledgeGraphState, answer: str) -> None:
    """Persist the current draft under work/q{id}/knowledge/{topic}/."""
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
    return {
        "kind": "knowledge",
        "topic": topic,
        "track": str(state.get("track") or knowledge_track(topic)),
        "system_prompt": selected.system,
        "user_prompt": selected.user,
        "messages": [{"role": "user", "content": selected.user}],
        "candidate_idx": 1,
        "repair_idx": 0,
        "temperature": candidate_temperature(1),
        "raw_response": "",
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
    client = get_llm_client()
    logger.info("%s calling model", header)
    print(f"\n{header} generating...", flush=True)

    options = get_knowledge_agent().llm_call_options(settings)
    stream_completion = getattr(client, "stream_completion", None)
    if callable(stream_completion):
        completion = stream_completion(
            messages=state["messages"],
            system=state.get("system_prompt") or SYSTEM_PROMPTS[0],
            temperature=temperature,
            print_stream=PRINT_STREAM,
            **options,
        )
    else:
        text = client.stream_text(
            messages=state["messages"],
            system=state.get("system_prompt") or SYSTEM_PROMPTS[0],
            temperature=temperature,
            print_stream=PRINT_STREAM,
        )
        completion = LLMCompletion(text=text, reasoning="", reasoning_source="empty")

    text = completion.text
    reasoning = completion.reasoning if settings.cot_enabled else ""
    reasoning_source = completion.reasoning_source if settings.cot_enabled else "empty"
    assistant_content = text if text.strip() else "(empty response)"
    messages = list(state.get("messages") or [])
    messages.append({"role": "assistant", "content": assistant_content})
    return {
        "raw_response": text,
        "messages": messages,
        "raw_reasoning": reasoning,
        "reasoning_source": reasoning_source,
    }


def extract(state: KnowledgeGraphState) -> dict[str, Any]:
    """Pull visible prose out of the last model reply (keep formulas)."""
    answer = extract_answer(state.get("raw_response") or "")
    if not answer.strip():
        logger.warning("Q%s knowledge: no answer extracted", state["question_id"])
    _write_attempt(state, answer)
    return {
        "answer": answer,
        "judge_must_fix": [],
        "judge_issues": [],
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
    )
    metadata = dict(state.get("metadata") or {})
    metadata["knowledge_judge"] = {
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


def repair(state: KnowledgeGraphState) -> dict[str, Any]:
    """Append a critique-and-rewrite user message and bump repair_idx."""
    settings = get_settings()
    next_repair = int(state.get("repair_idx") or 0) + 1
    repair_user = build_repair_prompt(
        topic=_topic(state),
        answer=str(state.get("answer") or ""),
        gate_reasons=list(state.get("gate_reasons") or []),
        must_fix=list(state.get("judge_must_fix") or []),
        issues=list(state.get("judge_issues") or []),
    )
    messages = list(state.get("messages") or [])
    messages.append({"role": "user", "content": repair_user})
    logger.info(
        "Q%s knowledge candidate %s starting repair %s/%s",
        state["question_id"],
        state.get("candidate_idx", 1),
        next_repair,
        settings.knowledge_max_repairs,
    )
    return {
        "repair_idx": next_repair,
        "messages": messages,
        "raw_response": "",
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
        "messages": [{"role": "user", "content": selected.user}],
        "raw_response": "",
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
    get_store().write_success(state, model_name=settings.resolved_model)
    _finalize_answer(state, success=True)
    logger.info(
        "Q%s knowledge saved SFT sample (candidate=%s repairs=%s score=%s cot=%s)",
        state["question_id"],
        state.get("candidate_idx", 1),
        state.get("repair_idx", 0),
        state.get("judge_score", 0),
        state.get("cot_source", "empty"),
    )
    return {"status": "success"}


def save_abandoned(state: KnowledgeGraphState) -> dict[str, Any]:
    """Write the abandoned record after all candidates failed the rubric."""
    reason = "knowledge_judge_unavailable" if state.get("judge_unavailable") else "knowledge_quality"
    payload = dict(state)
    payload["abandon_reason"] = reason
    get_store().write_abandoned(payload)  # type: ignore[arg-type]
    _finalize_answer(state, success=False)
    logger.info("Q%s knowledge abandoned (%s)", state["question_id"], reason)
    print(f"[Q{state['question_id']} knowledge] abandoned ({reason})", flush=True)
    return {"status": "abandoned", "abandon_reason": reason}


def route_after_gate(
    state: KnowledgeGraphState,
) -> Literal["judge", "repair", "next_candidate", "save_abandoned"]:
    """Route after the hard gate."""
    settings = get_settings()
    if state.get("gate_ok"):
        return "judge"
    if int(state.get("repair_idx") or 0) < settings.knowledge_max_repairs:
        return "repair"
    if int(state.get("candidate_idx") or 1) < settings.knowledge_max_candidates:
        return "next_candidate"
    return "save_abandoned"


def route_after_judge(
    state: KnowledgeGraphState,
) -> Literal["cot", "repair", "next_candidate", "save_abandoned"]:
    """Route after the LLM judge. JSON failure abandons rather than saving."""
    settings = get_settings()
    if state.get("judge_unavailable"):
        return "save_abandoned"
    if state.get("judge_pass"):
        return "cot"
    if int(state.get("repair_idx") or 0) < settings.knowledge_max_repairs:
        return "repair"
    if int(state.get("candidate_idx") or 1) < settings.knowledge_max_candidates:
        return "next_candidate"
    return "save_abandoned"


def build_knowledge_graph():
    """Compile the per-question knowledge StateGraph."""
    builder = StateGraph(KnowledgeGraphState)
    builder.add_node("prepare", prepare)
    builder.add_node("generate", generate, retry_policy=_retry_policy())
    builder.add_node("extract", extract)
    builder.add_node("gate", gate)
    builder.add_node("repair", repair)
    builder.add_node("next_candidate", next_candidate)
    builder.add_node("judge", judge)
    builder.add_node("cot", cot)
    builder.add_node("save_success", save_success)
    builder.add_node("save_abandoned", save_abandoned)

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
            "save_abandoned": "save_abandoned",
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
            "save_abandoned": "save_abandoned",
        },
    )
    builder.add_edge("cot", "save_success")
    builder.add_edge("save_success", END)
    builder.add_edge("save_abandoned", END)
    return builder.compile()


def knowledge_recursion_limit() -> int:
    """LangGraph superstep cap for knowledge candidates × repairs."""
    settings = get_settings()
    per_candidate = (settings.knowledge_max_repairs + 1) * 3 + settings.knowledge_max_repairs + 6
    return max(80, 10 + settings.knowledge_max_candidates * per_candidate)
