"""LangGraph: generate → extract → compile → repair / next candidate / save."""

from __future__ import annotations

import logging
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from cuda_sft.compile import attempt_workdir, compile_cuda_source, finalize_question_work
from cuda_sft.config import get_settings
from cuda_sft.llm import get_llm_client, is_retryable_llm_error
from cuda_sft.parse import extract_cuda_source
from cuda_sft.prompt import (
    SYSTEM_PROMPT,
    build_repair_prompt,
    candidate_temperature,
    format_nvcc_for_prompt,
    select_prompts,
    truncate_compile_error,
)
from cuda_sft.state import GraphState
from cuda_sft.store import get_store

logger = logging.getLogger(__name__)

PRINT_STREAM = True


def set_print_stream(enabled: bool) -> None:
    """Enable or disable printing streamed model tokens to stdout.

    Args:
        enabled: False under ``--quiet`` or when ``workers > 1``.
    """
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
    selected = select_prompts(
        state["question"],
        question_id=int(state["question_id"]),
        candidate_idx=1,
        gpu_name=gpu_name,
        cuda_arch=cuda_arch,
        cuda_version=cuda_version,
    )
    return {
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
    header = f"[Q{qid} candidate={cand} repair={repair} temp={temperature}]"
    logger.info("%s calling model", header)
    print(f"\n{header} generating...", flush=True)

    client = get_llm_client()
    text = client.stream_text(
        messages=state["messages"],
        system=state.get("system_prompt") or SYSTEM_PROMPT,
        temperature=temperature,
        print_stream=PRINT_STREAM,
    )
    assistant_content = text if text.strip() else "(empty response)"
    messages = list(state.get("messages") or [])
    messages.append({"role": "assistant", "content": assistant_content})
    return {"raw_response": text, "messages": messages}


def extract(state: GraphState) -> dict[str, Any]:
    """Pull CUDA source out of the last model reply.

    Args:
        state: Must contain ``raw_response``.
    """
    code = extract_cuda_source(state.get("raw_response") or "")
    if not code.strip():
        logger.warning("Q%s: no CUDA source extracted", state["question_id"])
    return {"code": code}


def compile_node(state: GraphState) -> dict[str, Any]:
    """``nvcc -c`` the extracted source and record the attempt.

    Args:
        state: Must contain ``code`` and ``question_id``.
    """
    settings = get_settings()
    workdir = attempt_workdir(
        settings,
        int(state["question_id"]),
        int(state.get("candidate_idx") or 1),
        int(state.get("repair_idx") or 0),
    )
    code = state.get("code") or ""
    attempts = list(state.get("attempts") or [])

    if not code.strip():
        error = "no CUDA source extracted from model output"
        attempts.append(
            {
                "candidate": state.get("candidate_idx", 1),
                "repair": state.get("repair_idx", 0),
                "ok": False,
                "used_rdc": False,
                "error": error,
            }
        )
        print(f"[Q{state['question_id']}] compile FAIL: {error}", flush=True)
        workdir.mkdir(parents=True, exist_ok=True)
        (workdir / "nvcc.log").write_text(error + "\n", encoding="utf-8")
        return {
            "compile_ok": False,
            "compile_error": error,
            "used_rdc": False,
            "attempts": attempts,
        }

    result = compile_cuda_source(code, workdir, settings=settings)
    error = "" if result.ok else (result.output or "nvcc failed with empty output")
    (workdir / "nvcc.log").write_text((result.output or error or "") + "\n", encoding="utf-8")
    prompt_error = (
        format_nvcc_for_prompt(error, settings.repair_error_max_chars) if not result.ok else ""
    )
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
    print(f"[Q{state['question_id']}] compile {status}{extra}", flush=True)
    if not result.ok:
        logger.info("nvcc error:\n%s", truncate_compile_error(error, 2000))
    return {
        "compile_ok": result.ok,
        "compile_error": error,
        "used_rdc": result.used_rdc,
        "attempts": attempts,
    }


def repair(state: GraphState) -> dict[str, Any]:
    """Append a compile-fix user message and bump ``repair_idx``.

    Args:
        state: Failed compile state with ``compile_error`` and ``code``.
    """
    settings = get_settings()
    next_repair = int(state.get("repair_idx") or 0) + 1
    repair_user = build_repair_prompt(
        cuda_arch=state.get("cuda_arch") or settings.resolved_cuda_arch,
        compile_error=format_nvcc_for_prompt(
            state.get("compile_error") or "",
            settings.repair_error_max_chars,
        ),
        previous_code=state.get("code") or "",
        question_id=int(state["question_id"]),
        candidate_idx=int(state.get("candidate_idx") or 1),
        repair_idx=next_repair,
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
        "raw_response": "",
        "compile_ok": False,
    }


def next_candidate(state: GraphState) -> dict[str, Any]:
    """Start a fresh candidate with a different prompt variant and temperature.

    Args:
        state: State after the previous candidate exhausted repairs.
    """
    settings = get_settings()
    next_idx = int(state.get("candidate_idx") or 1) + 1
    temperature = candidate_temperature(next_idx)
    selected = select_prompts(
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
    }


def save_success(state: GraphState) -> dict[str, Any]:
    """Write SFT jsonl rows and mark the question successful."""
    settings = get_settings()
    get_store().write_success(state, model_name=settings.resolved_model)
    finalize_question_work(
        settings,
        int(state["question_id"]),
        code=state.get("code") or "",
        success=True,
    )
    logger.info(
        "Q%s saved SFT sample (candidate=%s repairs=%s)",
        state["question_id"],
        state.get("candidate_idx", 1),
        state.get("repair_idx", 0),
    )
    return {"status": "success"}


def save_abandoned(state: GraphState) -> dict[str, Any]:
    """Write the abandoned record after all candidates failed."""
    settings = get_settings()
    get_store().write_abandoned(state)
    finalize_question_work(
        settings,
        int(state["question_id"]),
        code=state.get("code") or "",
        success=False,
    )
    logger.info("Q%s abandoned after all candidates failed", state["question_id"])
    print(f"[Q{state['question_id']}] abandoned", flush=True)
    return {"status": "abandoned"}


def route_after_compile(
    state: GraphState,
) -> Literal["save_success", "repair", "next_candidate", "save_abandoned"]:
    """Route after compile: save, repair, next candidate, or abandon."""
    settings = get_settings()
    if state.get("compile_ok"):
        return "save_success"
    if int(state.get("repair_idx") or 0) < settings.max_repairs:
        return "repair"
    if int(state.get("candidate_idx") or 1) < settings.max_candidates:
        return "next_candidate"
    return "save_abandoned"


def build_graph():
    """Compile the per-question StateGraph."""
    builder = StateGraph(GraphState)
    builder.add_node("prepare", prepare)
    builder.add_node("generate", generate, retry_policy=_retry_policy())
    builder.add_node("extract", extract)
    builder.add_node("compile", compile_node)
    builder.add_node("repair", repair)
    builder.add_node("next_candidate", next_candidate)
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
            "save_success": "save_success",
            "repair": "repair",
            "next_candidate": "next_candidate",
            "save_abandoned": "save_abandoned",
        },
    )
    builder.add_edge("repair", "generate")
    builder.add_edge("next_candidate", "generate")
    builder.add_edge("save_success", END)
    builder.add_edge("save_abandoned", END)
    return builder.compile()


def recursion_limit() -> int:
    """LangGraph superstep cap covering 3 candidates × (1 gen + 3 repairs)."""
    settings = get_settings()
    # prepare + per attempt (generate/extract/compile) + repair/next + save
    per_candidate = (settings.max_repairs + 1) * 3 + settings.max_repairs + 2
    return max(80, 10 + settings.max_candidates * per_candidate)
