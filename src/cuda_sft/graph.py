from __future__ import annotations

import logging
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from cuda_sft.compile import compile_cuda_source
from cuda_sft.config import get_settings
from cuda_sft.llm import get_llm_client, is_retryable_llm_error
from cuda_sft.parse import extract_cuda_source
from cuda_sft.prompt import (
    SYSTEM_PROMPT,
    build_repair_prompt,
    candidate_temperature,
    select_prompts,
    truncate_compile_error,
)
from cuda_sft.state import GraphState
from cuda_sft.store import get_store

logger = logging.getLogger(__name__)

PRINT_STREAM = True


def set_print_stream(enabled: bool) -> None:
    global PRINT_STREAM
    PRINT_STREAM = enabled


def _retry_policy():
    from langgraph.types import RetryPolicy

    return RetryPolicy(
        max_attempts=5,
        initial_interval=4.0,
        backoff_factor=2.0,
        max_interval=60.0,
        retry_on=is_retryable_llm_error,
    )


def prepare(state: GraphState) -> dict[str, Any]:
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
    code = extract_cuda_source(state.get("raw_response") or "")
    if not code.strip():
        logger.warning("Q%s: no CUDA source extracted", state["question_id"])
    return {"code": code}


def compile_node(state: GraphState) -> dict[str, Any]:
    settings = get_settings()
    workdir = (
        settings.work_path
        / f"q{state['question_id']}"
        / f"c{state.get('candidate_idx', 1)}"
        / f"r{state.get('repair_idx', 0)}"
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
        return {
            "compile_ok": False,
            "compile_error": error,
            "used_rdc": False,
            "attempts": attempts,
        }

    result = compile_cuda_source(code, workdir, settings=settings)
    error = "" if result.ok else (result.output or "nvcc failed with empty output")
    attempts.append(
        {
            "candidate": state.get("candidate_idx", 1),
            "repair": state.get("repair_idx", 0),
            "ok": result.ok,
            "used_rdc": result.used_rdc,
            "error": truncate_compile_error(error) if not result.ok else "",
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
    settings = get_settings()
    next_repair = int(state.get("repair_idx") or 0) + 1
    repair_user = build_repair_prompt(
        cuda_arch=state.get("cuda_arch") or settings.resolved_cuda_arch,
        compile_error=truncate_compile_error(state.get("compile_error") or ""),
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
    settings = get_settings()
    get_store().write_success(state, model_name=settings.model)
    logger.info(
        "Q%s saved SFT sample (candidate=%s repairs=%s)",
        state["question_id"],
        state.get("candidate_idx", 1),
        state.get("repair_idx", 0),
    )
    return {"status": "success"}


def save_abandoned(state: GraphState) -> dict[str, Any]:
    get_store().write_abandoned(state)
    logger.info("Q%s abandoned after all candidates failed", state["question_id"])
    print(f"[Q{state['question_id']}] abandoned", flush=True)
    return {"status": "abandoned"}


def route_after_compile(
    state: GraphState,
) -> Literal["save_success", "repair", "next_candidate", "save_abandoned"]:
    settings = get_settings()
    if state.get("compile_ok"):
        return "save_success"
    if int(state.get("repair_idx") or 0) < settings.max_repairs:
        return "repair"
    if int(state.get("candidate_idx") or 1) < settings.max_candidates:
        return "next_candidate"
    return "save_abandoned"


def build_graph():
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
    settings = get_settings()
    # prepare + per attempt (generate/extract/compile) + repair/next + save
    per_candidate = (settings.max_repairs + 1) * 3 + settings.max_repairs + 2
    return max(80, 10 + settings.max_candidates * per_candidate)
