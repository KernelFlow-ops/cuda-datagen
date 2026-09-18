"""LangGraph: generate → extract → compile → repair / judge → cot / save.

Nodes are thin wrappers around dialect specs and ``cuda_sft.agents``.
Print-stream and RetryPolicy live in ``pipeline.common`` so the knowledge
graph does not keep a second copy. Compile must not import the async LLM
pool; it calls :func:`enqueue_speculative_repair` instead.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from cuda_sft.agents.generate import (
    assistant_state_update,
    cancel_speculative,
    complete_chat,
    enqueue_speculative_repair,
)
from cuda_sft.agents.critic import KernelCritic
from cuda_sft.agents.difficulty import plan_topology
from cuda_sft.agents.repairer import (
    classify_compile_error,
    repair_system_prompt,
    wrap_repair_user,
)
from cuda_sft.judge import JudgeResult
from cuda_sft.compile import attempt_workdir, finalize_question_work
from cuda_sft.config import get_settings
from cuda_sft.cot import CotAgent
from cuda_sft.dialects.agent import get_dialect_agent, get_spec
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
from cuda_sft.state import GraphState
from cuda_sft.store import get_store

logger = logging.getLogger(__name__)

# Re-export so ``from cuda_sft.graph import set_print_stream`` keeps working.
__all__ = ["build_graph", "recursion_limit", "set_print_stream"]


def _dialect_name(state: GraphState) -> str:
    """Canonical dialect id for this graph state (default cuda)."""
    return str(state.get("dialect") or "cuda")


def _spec(state: GraphState):
    """Kernel dialect spec for this state."""
    return get_spec(_dialect_name(state))


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
    spec = get_spec(dialect)
    selected = spec.select_prompts(
        state["question"],
        question_id=int(state["question_id"]),
        candidate_idx=1,
        gpu_name=gpu_name,
        cuda_arch=cuda_arch,
        cuda_version=cuda_version,
    )
    topo = plan_topology(question=state["question"], kind="kernel", settings=settings)
    return {
        "dialect": spec.name,
        "system_prompt": selected.system,
        "user_prompt": selected.user,
        "messages": [{"role": "user", "content": selected.user}],
        "candidate_idx": 1,
        "repair_idx": 0,
        "temperature": candidate_temperature(1),
        "raw_response": "",
        "code": "",
        "compile_ok": False,
        "compile_error": "",
        "used_rdc": False,
        "status": "running",
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
        "speculative_requests": [],
        "raw_reasoning": "",
        "reasoning_source": "empty",
        "cot": "",
        "cot_source": "empty",
        "cot_error": "",
        "difficulty": topo.difficulty,
        "candidate_cap": topo.max_candidates,
        "use_critic": topo.use_critic,
        "critic_pass": True,
        "critic_skipped": True,
        "critic_must_fix": [],
        "critic_issues": [],
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
    result = complete_chat(
        messages=list(state.get("messages") or []),
        system=state.get("system_prompt") or SYSTEM_PROMPT,
        temperature=temperature,
        request_id=request_id if settings.async_llm_enabled else None,
        llm_options=get_dialect_agent().llm_call_options(dialect, settings),
        log_header=header,
    )
    return assistant_state_update(state, result, log_header=header)


def _repair_turn(
    state: GraphState,
    *,
    next_repair: int,
    prompt_error: str,
    code: str,
) -> tuple[str, str]:
    """Build the repair user message and Repairer system prompt.

    Used by both the compile prefetch and the repair node so the speculative
    cache key matches the live generate call.
    """
    spec = _spec(state)
    error_class = classify_compile_error(
        prompt_error, dialect=spec.name, code=code
    )
    inner = spec.build_repair(
        cuda_arch=state.get("cuda_arch") or get_settings().resolved_cuda_arch,
        compile_error=prompt_error,
        previous_code=code,
        question_id=int(state["question_id"]),
        candidate_idx=int(state.get("candidate_idx") or 1),
        repair_idx=next_repair,
    )
    user = wrap_repair_user(
        question=str(state.get("question") or ""),
        inner=inner,
        error_class=error_class,
        dialect=spec.name,
    )
    return user, repair_system_prompt(spec.name)


def extract(state: GraphState) -> dict[str, Any]:
    """Pull CUDA source out of the last model reply.

    Args:
        state: Must contain ``raw_response``.
    """
    spec = _spec(state)
    code = spec.extract(state.get("raw_response") or "")
    if not code.strip():
        logger.warning("Q%s %s: no source extracted", state["question_id"], spec.name)
    return {"code": code}


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
            "used_rdc": False,
            "attempts": attempts,
        }

    result = spec.compile(code, workdir, settings)
    error = "" if result.ok else (result.output or "compile failed with empty output")
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

    # Next repair is prefetched here so generate can reuse it; the pool lives in agents.generate.
    speculative_requests = list(state.get("speculative_requests") or [])
    if not result.ok and settings.async_llm_enabled:
        next_repair = int(state.get("repair_idx") or 0) + 1
        if next_repair <= settings.max_repairs:
            qid = state["question_id"]
            cand = state.get("candidate_idx", 1)
            request_id = f"q{qid}_{spec.name}_c{cand}_r{next_repair}"
            repair_prompt, repair_system = _repair_turn(
                state,
                next_repair=next_repair,
                prompt_error=prompt_error,
                code=code,
            )
            future_messages = list(state.get("messages") or [])
            future_messages.append({"role": "user", "content": repair_prompt})
            if enqueue_speculative_repair(
                request_id=request_id,
                messages=future_messages,
                system=repair_system,
                temperature=float(state.get("temperature") or candidate_temperature(cand)),
                llm_options=get_dialect_agent().llm_call_options(spec.name, settings),
            ):
                speculative_requests.append(request_id)

    return {
        "compile_ok": result.ok,
        "compile_error": error,
        "used_rdc": result.used_rdc,
        "attempts": attempts,
        "speculative_requests": speculative_requests,
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
    spec = _spec(state)
    next_repair = int(state.get("repair_idx") or 0) + 1
    if state.get("critic_must_fix") and state.get("compile_ok"):
        prompt_error = "semantic critic must_fix:\n- " + "\n- ".join(
            str(item) for item in state.get("critic_must_fix") or []
        )
    else:
        prompt_error = format_nvcc_for_prompt(
            state.get("compile_error") or "",
            settings.repair_error_max_chars,
        )
    repair_user, repair_system = _repair_turn(
        state,
        next_repair=next_repair,
        prompt_error=prompt_error,
        code=state.get("code") or "",
    )
    messages = list(state.get("messages") or [])
    messages.append({"role": "user", "content": repair_user})
    logger.info(
        "Q%s candidate %s starting repair %s/%s",
        state["question_id"],
        state.get("candidate_idx", 1),
        next_repair,
        settings.max_repairs,
    )
    return {
        "repair_idx": next_repair,
        "messages": messages,
        "system_prompt": repair_system,
        "raw_response": "",
        "compile_ok": False,
        "skip_repair": False,
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
        "repair_idx": 0,
        "temperature": temperature,
        "system_prompt": selected.system,
        "user_prompt": selected.user,
        "messages": [{"role": "user", "content": selected.user}],
        "raw_response": "",
        "code": "",
        "compile_ok": False,
        "compile_error": "",
        "used_rdc": False,
        "raw_reasoning": "",
        "reasoning_source": "empty",
        "cot": "",
        "cot_source": "empty",
        "cot_error": "",
    }


def save_success(state: GraphState) -> dict[str, Any]:
    """Write SFT jsonl rows and mark the question successful."""
    settings = get_settings()
    spec = _spec(state)
    nest = get_dialect_agent().nest_workdir(spec.name, settings)
    get_store().write_success(state, model_name=settings.resolved_model)
    finalize_question_work(
        settings,
        int(state["question_id"]),
        code=state.get("code") or "",
        success=True,
        dialect=spec.name,
        filename=spec.source_filename,
        nest_dialect=nest,
    )
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
            "winner_found": True,
            "winner_candidate": int(state.get("candidate_idx") or 1),
        }

    spec = _spec(state)
    code = state.get("code") or ""
    result = spec.judge(code)

    logger.info(
        "Q%s judge: score=%s issues=%s suggestions=%s",
        state["question_id"],
        result.quality_score,
        len(result.issues),
        len(result.suggestions),
    )
    print(
        f"[Q{state['question_id']}] judge: score={result.quality_score}/10 "
        f"issues={len(result.issues)} suggestions={len(result.suggestions)}",
        flush=True,
    )

    metadata = dict(state.get("metadata") or {})
    metadata["judge"] = {
        "quality_score": result.quality_score,
        "issues": result.issues,
        "suggestions": result.suggestions,
    }
    code_out = state.get("code") or ""
    if result.optimized_code and settings.use_judge_optimization:
        metadata["judge"]["optimization_applied"] = True
        code_out = result.optimized_code

    return {
        "judge_score": result.quality_score,
        "judge_issues": result.issues,
        "judge_suggestions": result.suggestions,
        "metadata": metadata,
        "speculative_requests": [],
        "winner_found": True,
        "winner_candidate": int(state.get("candidate_idx") or 1),
        "code": code_out,
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
    )
    metadata = dict(state.get("metadata") or {})
    metadata["critic"] = {
        "passed": result.passed,
        "skipped": result.skipped,
        "must_fix": result.must_fix,
        "issues": result.issues,
    }
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

    result = CotAgent(settings).refine(state)
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
    )
    logger.info("Q%s abandoned after all candidates failed", state["question_id"])
    print(f"[Q{state['question_id']}] abandoned", flush=True)
    return {"status": "abandoned"}


def route_after_compile(
    state: GraphState,
) -> Literal["judge", "repair", "next_candidate", "save_abandoned"]:
    """Route after compile: judge (if ok), repair, next candidate, or abandon."""
    settings = get_settings()
    cap = int(state.get("candidate_cap") or settings.max_candidates)
    if state.get("compile_ok"):
        return "judge"
    if int(state.get("repair_idx") or 0) < settings.max_repairs:
        return "repair"
    if int(state.get("candidate_idx") or 1) < cap:
        return "next_candidate"
    return "save_abandoned"


def route_after_critic(state: GraphState) -> Literal["cot", "repair"]:
    """Send compile-passing code back to repair only when critic must_fix remains."""
    settings = get_settings()
    if state.get("critic_pass", True):
        return "cot"
    if not (state.get("critic_must_fix") or []):
        return "cot"
    if int(state.get("repair_idx") or 0) < settings.max_repairs:
        return "repair"
    if settings.kernel_critic_blocks_save:
        return "cot"
    return "cot"


def route_after_repair(state: GraphState) -> Literal["generate", "next_candidate"]:
    """Skip remaining repairs when another candidate already compiled."""
    if state.get("skip_repair"):
        return "next_candidate"
    return "generate"


def build_graph():
    """Compile the per-question StateGraph."""
    builder = StateGraph(GraphState)
    builder.add_node("prepare", prepare)
    builder.add_node("generate", generate, retry_policy=retry_policy())
    builder.add_node("extract", extract)
    builder.add_node("compile", compile_node)
    builder.add_node("repair", repair)
    builder.add_node("next_candidate", next_candidate)
    builder.add_node("judge", judge)
    builder.add_node("critic", critic)
    builder.add_node("cot", cot)
    builder.add_node("save_success", save_success)
    builder.add_node("save_abandoned", save_abandoned)

    builder.add_edge(START, "prepare")
    builder.add_edge("prepare", "generate")
    builder.add_edge("generate", "extract")
    builder.add_edge("extract", "compile")
    builder.add_conditional_edges(
        "compile",
        route_after_compile,
        {
            "judge": "judge",
            "repair": "repair",
            "next_candidate": "next_candidate",
            "save_abandoned": "save_abandoned",
        },
    )
    builder.add_conditional_edges(
        "repair",
        route_after_repair,
        {
            "generate": "generate",
            "next_candidate": "next_candidate",
        },
    )
    builder.add_edge("next_candidate", "generate")
    builder.add_edge("judge", "critic")
    builder.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "cot": "cot",
            "repair": "repair",
        },
    )
    builder.add_edge("cot", "save_success")
    builder.add_edge("save_success", END)
    builder.add_edge("save_abandoned", END)
    return builder.compile()


def recursion_limit() -> int:
    """LangGraph superstep cap covering 3 candidates x (1 gen + 3 repairs)."""
    settings = get_settings()
    return graph_recursion_limit(
        max_candidates=settings.max_candidates,
        max_repairs=settings.max_repairs,
        extra_per_candidate=6,
    )
