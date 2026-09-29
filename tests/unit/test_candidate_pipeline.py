"""Candidate isolation and asynchronous request ownership."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from cuda_sft.agents.generate import complete_chat
from cuda_sft.compile import CompileResult, attempt_workdir, finalize_question_work
from cuda_sft.config import Settings
from cuda_sft.core.types import build_snapshot
from cuda_sft.graph import (
    build_candidate_graph,
    prepare,
    recursion_limit,
    save_success,
    select_best,
)
from cuda_sft.llm import LLMCompletion
from cuda_sft.llm_async import AsyncLLMPool
from cuda_sft.runtime import deps


def test_candidate_graph_banks_without_selection_or_save() -> None:
    graph = build_candidate_graph().get_graph()
    assert [(edge.source, edge.target) for edge in graph.edges if edge.source == "collect_candidate"] == [
        ("collect_candidate", "__end__")
    ]
    assert "save_success" not in graph.nodes
    assert "save_abandoned" not in graph.nodes
    assert "next_candidate" not in graph.nodes


def test_prepare_respects_requested_candidate() -> None:
    settings = Settings(max_candidates=3, async_llm_enabled=False)
    with patch("cuda_sft.graph.get_settings", return_value=settings):
        state = prepare({"question_id": 1, "question": "vector add", "candidate_idx": 2})
    assert state["candidate_idx"] == 2
    assert state["candidate_ctx"]["candidate"] == 2
    assert state["candidate_ctx"]["prompt_variant"]["temperature"] == state["temperature"]


def test_candidate_graph_runs_one_candidate_without_saving(tmp_path) -> None:
    settings = Settings(
        work_dir=str(tmp_path),
        async_llm_enabled=False,
        refval_enabled=False,
        judge_enabled=False,
        cot_enabled=False,
        max_candidates=3,
        max_repairs=0,
        cuda_arch="sm_86",
        gpu_name="test GPU",
        cuda_home="/usr/local/cuda",
    )

    class Client:
        def stream_completion(self, **_kwargs):
            return LLMCompletion(
                text='extern "C" __global__ void add(float *x) { x[0] += 1; }',
                origin="live_api",
            )

    def compile_source(_dialect, _code, workdir, _settings):
        workdir.mkdir(parents=True, exist_ok=True)
        return CompileResult(ok=True, command=[], output="", used_rdc=False)

    with patch("cuda_sft.graph.get_settings", return_value=settings), patch(
        "cuda_sft.agents.generate.get_settings", return_value=settings
    ), deps.use(deps.Deps(llm_factory=lambda _role: Client(), compile_fn=compile_source)):
        state = build_candidate_graph().invoke(
            {"question_id": 9, "question": "vector add", "candidate_idx": 2},
            {"recursion_limit": recursion_limit()},
        )
    assert state["candidate_idx"] == 2
    assert len(state["candidate_reports"]) == 1
    assert state["candidate_reports"][0]["candidate"] == 2
    assert state["candidate_reports"][0]["origin"] == "live_api"
    assert state["status"] == "running"


def test_repair_generation_follows_compile_diagnostic(tmp_path) -> None:
    settings = Settings(
        work_dir=str(tmp_path),
        generator_provider="openai",
        generator_model="generator-model",
        repair_compile_provider="nvidia",
        repair_compile_model="repair-model",
        async_llm_enabled=True,
        refval_enabled=False,
        judge_enabled=False,
        cot_enabled=False,
        max_candidates=1,
        max_repairs=1,
        cuda_arch="sm_86",
        gpu_name="test GPU",
        cuda_home="/usr/local/cuda",
    )
    generated_at = []
    compiled_at = []

    class Client:
        def stream_completion(self, **_kwargs):
            generated_at.append(time.monotonic())
            code = "bad" if len(generated_at) == 1 else "good"
            return LLMCompletion(
                text=f'extern "C" __global__ void {code}(float *x) {{ x[0] += 1; }}',
                origin="live_api",
            )

    def compile_source(_dialect, _code, workdir, _settings):
        workdir.mkdir(parents=True, exist_ok=True)
        compiled_at.append(time.monotonic())
        failed = len(compiled_at) == 1
        return CompileResult(
            ok=not failed,
            command=[],
            output="synthetic compile diagnostic" if failed else "",
            used_rdc=False,
        )

    with patch("cuda_sft.graph.get_settings", return_value=settings), patch(
        "cuda_sft.agents.generate.get_settings", return_value=settings
    ), deps.use(deps.Deps(llm_factory=lambda _role: Client(), compile_fn=compile_source)):
        state = build_candidate_graph().invoke(
            {"question_id": 10, "question": "vector add", "candidate_idx": 1},
            {"recursion_limit": recursion_limit()},
        )
    assert len(generated_at) == len(compiled_at) == 2
    assert generated_at[1] >= compiled_at[0]
    assert [attempt["ok"] for attempt in state["attempts"]] == [False, True]
    assert state["candidate_reports"][0]["provider"] == "nvidia"
    assert state["candidate_reports"][0]["model"] == "repair-model"


def test_selection_restores_winner_rdc_flag_and_model() -> None:
    def snapshot(index: int, *, rdc: bool, score: int):
        return build_snapshot(
            {
                "candidate_idx": index,
                "code": f"code {index}",
                "compile_ok": True,
                "used_rdc": rdc,
                "refval_status": "skip",
                "metadata": {"judge": {"quality_score": score}},
                "candidate_ctx": {"provider": "openai", "model": f"model-{index}"},
            },
            banked_reason="final",
        )

    with patch(
        "cuda_sft.graph.get_settings",
        return_value=Settings(refval_enabled=False, refval_strict=False),
    ):
        selected = select_best(
            {
                "candidate_reports": [
                    snapshot(1, rdc=False, score=1),
                    snapshot(2, rdc=True, score=9),
                ],
                "metadata": {},
            }
        )
    assert selected["candidate_idx"] == 2
    assert selected["used_rdc"] is True
    assert selected["provenance"]["provider"] == "openai"
    assert selected["provenance"]["model"] == "model-2"
    with patch("cuda_sft.graph.get_settings", return_value=Settings()), patch(
        "cuda_sft.graph.get_store"
    ) as store, patch("cuda_sft.graph.finalize_question_work"):
        store.return_value.write_success.return_value = True
        save_success({**selected, "question_id": 1, "dialect": "cuda"})
    assert store.return_value.write_success.call_args.kwargs["model_name"] == "model-2"


def test_simple_workdir_isolates_attempts_and_finalizes(tmp_path) -> None:
    settings = Settings(work_dir=str(tmp_path), work_keep="simple")
    first = attempt_workdir(settings, 7, 1, 0)
    second = attempt_workdir(settings, 7, 2, 0)
    repair = attempt_workdir(settings, 7, 1, 1)
    assert len({first, second, repair}) == 3
    for path in (first, second, repair):
        path.mkdir(parents=True)
    (first / "nvcc.log").write_text("candidate one failed\n")
    (second / "nvcc.log").write_text("candidate two failed\n")
    finalize_question_work(
        settings, 7, code="candidate two", success=False, candidate_idx=2, repair_idx=0
    )
    root = settings.work_path / "q7"
    assert (root / "solution.cu").read_text() == "candidate two"
    assert (root / "nvcc.log").read_text() == "candidate two failed\n"
    assert not first.exists() and not second.exists() and not repair.exists()


def test_prefetched_request_timeout_does_not_call_model_twice() -> None:
    started = threading.Event()
    release = threading.Event()
    calls = []

    class Client:
        def stream_completion(self, **kwargs):
            calls.append(kwargs)
            started.set()
            assert release.wait(2)
            return LLMCompletion(text="one answer", origin="live_api")

    client = Client()
    pool = AsyncLLMPool(max_workers=1)
    settings = SimpleNamespace(
        async_llm_enabled=True,
        async_llm_max_workers=1,
        llm_timeout_sec=0.01,
        cot_enabled=False,
    )
    request_id = "q1_cuda_c1_r1"
    try:
        pool.enqueue(request_id, client, [{"role": "user", "content": "q"}], "system", 0.2)
        assert started.wait(2)
        with patch("cuda_sft.agents.generate.get_settings", return_value=settings), patch(
            "cuda_sft.agents.generate.get_async_pool", return_value=pool
        ):
            with pytest.raises(TimeoutError) as timeout:
                complete_chat(
                    messages=[{"role": "user", "content": "q"}],
                    system="system",
                    temperature=0.2,
                    request_id=request_id,
                    client=client,
                )
            assert type(timeout.value) is TimeoutError
            assert pool.is_pending(request_id)
            release.set()
            result = complete_chat(
                messages=[{"role": "user", "content": "q"}],
                system="system",
                temperature=0.2,
                request_id=request_id,
                client=client,
            )
        assert result.text == "one answer"
        assert result.used_speculative
        assert len(calls) == 1
    finally:
        release.set()
        pool.shutdown()
