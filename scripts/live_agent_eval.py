#!/usr/bin/env python3
"""Opt-in live evaluation of the generation agents.

The normal command evaluates a fixed ten-task matrix. It never runs on import;
provider calls happen only from :func:`main`. Every application request is
bounded by ``--max-calls`` and journaled before and after the SDK call.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cuda_sft.agents.critic import KernelCritic  # noqa: E402
from cuda_sft.agents.repairer import (  # noqa: E402
    classify_compile_error,
    repair_system_prompt,
    wrap_repair_user,
)
from cuda_sft.config import PROJECT_ROOT, get_settings  # noqa: E402
from cuda_sft.cot import CotAgent  # noqa: E402
from cuda_sft.dialects.agent import get_spec  # noqa: E402
from cuda_sft.judge import CudaCodeJudge, JudgeResult  # noqa: E402
from cuda_sft.knowledge.cot import KnowledgeCotAgent  # noqa: E402
from cuda_sft.knowledge.judge import KnowledgeJudge  # noqa: E402
from cuda_sft.knowledge.prompt import select_prompts as select_knowledge_prompts  # noqa: E402
from cuda_sft.llm import LLMCompletion, get_llm_client  # noqa: E402
from cuda_sft.prompt import candidate_temperature  # noqa: E402


DEFAULT_MAX_CALLS = 20
DEFAULT_JOURNAL = PROJECT_ROOT / "data" / "live_agent_eval.journal.jsonl"
_SECRET_RE = re.compile(r"(?i)(?:sk-|nvapi-|api[_-]?key[=: ]+)[A-Za-z0-9._:-]{8,}")


@dataclass(frozen=True)
class LiveTask:
    """One fixed matrix task and its applicable agent components."""

    task_id: str
    kind: str
    dialect: str
    difficulty: str
    question: str
    topic: str = "general"

    @property
    def refval_applicable(self) -> bool:
        return self.kind == "kernel" and self.dialect == "cuda"


FIXED_TASKS: tuple[LiveTask, ...] = (
    LiveTask("cuda_simple_vector_add", "kernel", "cuda", "simple", "Implement a CUDA vector addition kernel C[i] = A[i] + B[i] with bounds checking and a host launcher."),
    LiveTask("cuda_simple_relu", "kernel", "cuda", "simple", "Implement a CUDA ReLU kernel for a float array, writing max(x, 0) with a bounds check and host launcher."),
    LiveTask("cuda_reduction_sum", "kernel", "cuda", "medium", "Implement a CUDA block reduction that sums a float array with shared memory and combines block sums."),
    LiveTask("cuda_transpose", "kernel", "cuda", "medium", "Implement a tiled CUDA matrix transpose with shared memory, bank-conflict padding, and arbitrary rectangular dimensions."),
    LiveTask("cuda_hard_attention", "kernel", "cuda", "hard", "Implement a numerically stable tiled CUDA scaled dot-product attention operator with masking and arbitrary sequence lengths."),
    LiveTask("cutlass_hard_gemm", "kernel", "cutlass", "hard", "Implement a CUTLASS 4.x CuTe tiled GEMM with an epilogue C = alpha A B + beta C and a complete host entry."),
    LiveTask("triton_layernorm", "kernel", "triton", "medium", "Implement a Triton layer-normalization kernel over the last dimension with numerically stable mean and variance."),
    LiveTask("tilelang_matmul", "kernel", "tilelang", "medium", "Implement a TileLang tiled matrix multiplication with bounds checks and a Python callable entry point."),
    LiveTask("knowledge_occupancy", "knowledge", "knowledge", "medium", "Explain CUDA occupancy versus utilization, the resource limits that determine active warps, and architecture-dependent caveats.", "execution"),
    LiveTask("knowledge_roofline", "knowledge", "knowledge", "medium", "Derive the roofline performance bound for a CUDA kernel, define arithmetic intensity, and state when the model is valid.", "formula"),
)


class _BudgetExceeded(RuntimeError):
    """Raised before a request when the run budget is exhausted."""


class Journal:
    """Durable JSONL request journal; each event is flushed and fsynced."""

    def __init__(self, path: Path | None) -> None:
        self.path = path

    def append(self, event: Mapping[str, Any]) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_sanitize(dict(event)), ensure_ascii=True, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


class BudgetClient:
    """Injectable LLM client proxy with durable accounting and context labels."""

    def __init__(self, client: Any, limit: int, journal: Journal | None = None) -> None:
        self.client = client
        self.limit = int(limit)
        self.journal = journal or Journal(None)
        self.attempted = 0
        self.completed = 0
        self.failed = 0
        self.budget_exhausted = 0
        self._component = "unknown"
        self._task_id = "unknown"
        self.sdk_retries = self._disable_sdk_retries()

    def _disable_sdk_retries(self) -> int | None:
        inner = getattr(self.client, "_client", None)
        with_options = getattr(inner, "with_options", None)
        if not callable(with_options):
            return None
        try:
            self.client._client = with_options(max_retries=0)
            return 0
        except Exception:
            return None

    @contextmanager
    def context(self, *, component: str, task_id: str) -> Iterator["BudgetClient"]:
        previous = self._component, self._task_id
        self._component, self._task_id = component, task_id
        try:
            yield self
        finally:
            self._component, self._task_id = previous

    def _call(self, method: str, kwargs: dict[str, Any]) -> Any:
        if self.attempted >= self.limit:
            self.budget_exhausted += 1
            self.journal.append({"event": "budget_exhausted", "component": self._component, "task_id": self._task_id, "attempted": self.attempted, "limit": self.limit})
            raise _BudgetExceeded("max-calls reached")
        call_no = self.attempted + 1
        self.attempted = call_no
        base = {"call": call_no, "component": self._component, "task_id": self._task_id, "attempted": self.attempted, "limit": self.limit}
        self.journal.append({"event": "request_before", **base})
        try:
            result = getattr(self.client, method)(**kwargs)
        except BaseException as exc:
            self.failed += 1
            self.journal.append({"event": "request_after", **base, "status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "error", "error_category": _error_category(exc)})
            raise
        self.completed += 1
        self.journal.append({"event": "request_after", **base, "status": "ok"})
        return result

    def stream_completion(self, **kwargs: Any) -> LLMCompletion:
        return self._call("stream_completion", kwargs)

    def stream_text(self, **kwargs: Any) -> str:
        return self._call("stream_text", kwargs)


def _redact(value: object) -> str:
    text = str(value or "").replace("\n", " ").strip()
    return _SECRET_RE.sub("[REDACTED]", text)[:180]


def _sanitize(value: Any) -> Any:
    if isinstance(value, str):
        return _redact(value)
    if isinstance(value, dict):
        return {key: _sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    return value


def _error_category(value: object) -> str:
    text = _redact(value).lower()
    if any(token in text for token in ("timeout", "timed out", "deadline")):
        return "timeout"
    if any(token in text for token in ("401", "403", "unauthorized", "api key", "authentication")):
        return "authentication"
    if any(token in text for token in ("429", "rate limit", "too many")):
        return "rate_limit"
    if any(token in text for token in ("overload", "capacity", "temporarily unavailable")):
        return "provider_overload"
    if any(token in text for token in ("parse", "json", "extract")):
        return "parse_failure"
    return "agent_error"


def check_cot_consistency(cot: str, code: str, *, source: str = "") -> dict[str, Any]:
    """Check that a CoT result is present and does not copy executable code."""
    text = (cot or "").strip()
    if not text:
        return {"status": "fail", "reason": "empty_cot", "source": source}
    if any(token in text for token in ("```", "__global__", "#include", "solution.cu")):
        return {"status": "fail", "reason": "code_leak", "source": source, "chars": len(text)}
    return {"status": "pass", "source": source or "unknown", "chars": len(text), "code_chars": len(code or "")}


def _compile(code: str, settings: Any, spec: Any, workdir: Path) -> tuple[str, str]:
    try:
        result = spec.compile(code, workdir, settings)
    except Exception as exc:
        return "skip", _redact(exc.__class__.__name__)
    if result.ok:
        return "pass", ""
    tail = (result.output or "").strip().splitlines()
    return "fail", _redact(tail[-1] if tail else "compile failed")


def _task_number(task: LiveTask) -> int:
    return FIXED_TASKS.index(task) + 1


def _prompts(task: LiveTask, settings: Any, candidate: int) -> tuple[str, str]:
    qid = _task_number(task)
    if task.kind == "kernel":
        selected = get_spec(task.dialect).select_prompts(task.question, question_id=qid, candidate_idx=candidate, gpu_name=settings.resolved_gpu_name, cuda_arch=settings.resolved_cuda_arch, cuda_version=settings.resolved_cuda_version)
    else:
        selected = select_knowledge_prompts(task.question, question_id=qid, candidate_idx=candidate, topic=task.topic, gpu_name=settings.resolved_gpu_name, cuda_arch=settings.resolved_cuda_arch, cuda_version=settings.resolved_cuda_version)
    return selected.system, selected.user


def _run_refval_component(task: LiveTask, code: str, settings: Any, *, budget: BudgetClient, oracle: Any | None = None, runner: Callable[..., Any] | None = None) -> dict[str, Any]:
    """Run refval with an injected oracle; no oracle means no hidden API call."""
    if not task.refval_applicable:
        return {"status": "not_applicable"}
    if oracle is None:
        return {"status": "not_evaluated", "reason": "oracle_not_injected"}
    from cuda_sft.refval.runner import run_refval
    run = runner or run_refval
    try:
        with budget.context(component="refval", task_id=task.task_id):
            report = run(question=task.question, code=code, question_id=_task_number(task), dialect=task.dialect, dialect_spec=get_spec(task.dialect).refval_spec(settings), settings=settings, manifest=oracle)
        payload = report.to_dict() if hasattr(report, "to_dict") else dict(report)
        return {"status": str(payload.get("status") or "unknown"), "report": _sanitize(payload)}
    except _BudgetExceeded:
        return {"status": "not_evaluated", "reason": "budget_exhausted"}
    except Exception as exc:
        return {"status": "error", "error_category": _error_category(exc)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bounded live evaluation of the configured CUDA agents")
    parser.add_argument("--max-calls", type=int, default=DEFAULT_MAX_CALLS)
    parser.add_argument("--candidate-pool", "--candidates", dest="candidate_pool", type=int, default=2)
    parser.add_argument("--smoke", action="store_true", help="evaluate only the first fixed task")
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument("--refval", action="store_true", help="run refval where an oracle is injected")
    parser.add_argument("--strict", action="store_true", help="fail if any requested component is not verified")
    parser.add_argument("--full", action="store_true", help="request generate, compile, repair, refval, critic, and cot")
    parser.add_argument("--agent-components", default="generate,compile,repair,refval,critic,cot")
    parser.add_argument("--journal", type=Path, default=DEFAULT_JOURNAL)
    parser.add_argument("--report", type=Path, default=None)
    return parser


def _components(args: argparse.Namespace) -> set[str]:
    raw = "generate,compile,repair,refval,critic,cot" if args.full else args.agent_components
    result = {part.strip().lower() for part in raw.split(",") if part.strip()}
    allowed = {"generate", "compile", "repair", "refval", "critic", "cot"}
    unknown = result - allowed
    if unknown:
        raise ValueError(f"unknown components: {', '.join(sorted(unknown))}")
    return result


def _initial_component_status(task: LiveTask, requested: set[str]) -> dict[str, Any]:
    applicable = {"generate", "critic", "cot"}
    if task.kind == "kernel":
        applicable |= {"compile", "repair", "refval"}
    return {
        component: (
            "pending"
            if component in requested and component in applicable
            else "not_applicable"
            if component in requested
            else "not_requested"
        )
        for component in ("generate", "compile", "repair", "refval", "critic", "cot")
    }


def _strict_ok(records: list[dict[str, Any]], requested: set[str]) -> bool:
    if not records or not requested:
        return False
    for record in records:
        for component in requested:
            status = record.get("components", {}).get(component, {}).get("status")
            if status in {None, "pending", "not_evaluated", "not_requested", "not_applicable", "skip", "fail", "error"}:
                # not_applicable is valid only for a requested component on a
                # task where the component has no meaning.
                if status == "not_applicable":
                    continue
                return False
    return True


def run(args: argparse.Namespace) -> int:
    if args.max_calls < 1 or args.candidate_pool < 1:
        raise ValueError("--max-calls and --candidate-pool must be positive")
    settings = get_settings()
    missing = settings.missing_provider_secrets(settings.provider_pool())
    if missing:
        print(f"live eval skipped: configure provider secret(s) in .env ({', '.join(missing)})", file=sys.stderr)
        return 2
    requested = _components(args)
    tasks = FIXED_TASKS[:1] if args.smoke else FIXED_TASKS
    max_calls = min(args.max_calls, 1) if args.smoke else args.max_calls
    budget = BudgetClient(get_llm_client(settings), max_calls, Journal(args.journal))
    records: list[dict[str, Any]] = []
    started = time.monotonic()
    terminated = ""
    for task in tasks:
        record: dict[str, Any] = {"task_id": task.task_id, "kind": task.kind, "dialect": task.dialect, "difficulty": task.difficulty, "components": {key: {"status": value} for key, value in _initial_component_status(task, requested).items()}}
        code = ""
        for candidate in range(1, args.candidate_pool + 1):
            if "generate" not in requested or budget.attempted >= max_calls:
                break
            try:
                system, user = _prompts(task, settings, candidate)
                with budget.context(component="generate", task_id=task.task_id):
                    completion = budget.stream_completion(messages=[{"role": "user", "content": user}], system=system, temperature=candidate_temperature(candidate), print_stream=False)
                raw = completion.text or completion.reasoning or ""
                code = get_spec(task.dialect).extract(raw) if task.kind == "kernel" else raw
                record.update(candidate=candidate, response_chars=len(raw), code_chars=len(code))
                record["components"]["generate"] = {"status": "pass", "candidate": candidate}
                if task.kind == "kernel":
                    spec = get_spec(task.dialect)
                    if args.skip_compile or "compile" not in requested:
                        compile_status, detail = "not_evaluated", "compile_not_requested"
                    else:
                        with tempfile.TemporaryDirectory(prefix="cuda-live-eval-") as temp:
                            compile_status, detail = _compile(code, settings, spec, Path(temp))
                    record["components"]["compile"] = {"status": compile_status, **({"detail": detail} if detail else {})}
                    if "refval" in requested:
                        record["components"]["refval"] = _run_refval_component(task, code, settings, budget=budget) if args.refval else {"status": "not_evaluated", "reason": "oracle_not_injected"}
                    if "repair" in requested and compile_status == "fail" and budget.attempted < max_calls:
                        repair_user = wrap_repair_user(question=task.question, inner=f"## Compiler error\n{detail}\n\n## Previous source\n```cuda\n{code}\n```", error_class=classify_compile_error(detail, dialect=task.dialect, code=code), dialect=task.dialect, repair_idx=1, max_repairs=1)
                        try:
                            with budget.context(component="repair", task_id=task.task_id):
                                repaired = budget.stream_completion(messages=[{"role": "user", "content": repair_user}], system=repair_system_prompt(task.dialect), temperature=0.2, print_stream=False, thinking_level="none")
                            repaired_code = spec.extract(repaired.text or repaired.reasoning or "")
                            with tempfile.TemporaryDirectory(prefix="cuda-live-repair-") as temp:
                                repaired_status, repaired_detail = _compile(repaired_code, settings, spec, Path(temp))
                            record["components"]["repair"] = {"status": "pass" if repaired_status == "pass" else "fail", "improved": repaired_status == "pass", "compile": repaired_status, **({"detail": repaired_detail} if repaired_detail else {})}
                        except _BudgetExceeded:
                            record["components"]["repair"] = {"status": "not_evaluated", "reason": "budget_exhausted"}
                    elif "repair" in requested:
                        record["components"]["repair"] = {"status": "not_evaluated", "reason": "no_real_compile_failure"}
                if "critic" in requested:
                    if task.kind == "kernel":
                        heuristic = CudaCodeJudge(settings).judge(code)
                        forced = JudgeResult(min(heuristic.quality_score, 7), heuristic.issues or ["live evaluation"], heuristic.suggestions)
                        with budget.context(component="critic", task_id=task.task_id):
                            result = KernelCritic(settings, llm_client=budget).evaluate(question=task.question, code=code, dialect=task.dialect, heuristic=forced, use_critic=True, refval_status="skip")
                        record["components"]["critic"] = {"status": "pass" if result.passed else "fail"}
                    else:
                        with budget.context(component="critic", task_id=task.task_id):
                            result = KnowledgeJudge(settings, llm_client=budget).judge(question=task.question, answer=code, topic=task.topic)
                        record["components"]["critic"] = {"status": "pass" if result.passed else "fail", "overall": result.overall}
                if "cot" in requested:
                    try:
                        old_enabled = settings.cot_agent_enabled
                        object.__setattr__(settings, "cot_agent_enabled", True)
                        with budget.context(component="cot", task_id=task.task_id):
                            cot = (CotAgent(settings, llm_client=budget).refine({"question": task.question, "code": code, "dialect": task.dialect, "raw_reasoning": completion.reasoning}) if task.kind == "kernel" else KnowledgeCotAgent(settings, llm_client=budget).refine({"question": task.question, "answer": code, "topic": task.topic, "raw_reasoning": completion.reasoning}))
                        record["components"]["cot"] = check_cot_consistency(cot.cot, code, source=cot.source)
                        object.__setattr__(settings, "cot_agent_enabled", old_enabled)
                    except _BudgetExceeded:
                        record["components"]["cot"] = {"status": "not_evaluated", "reason": "budget_exhausted"}
                break
            except _BudgetExceeded:
                record["components"]["generate"] = {"status": "not_evaluated", "reason": "budget_exhausted"}
                break
            except KeyboardInterrupt:
                terminated = "interrupted"
                break
            except Exception as exc:
                record["components"]["generate"] = {"status": "error", "error_category": _error_category(exc)}
                break
        for component in record["components"].values():
            if component.get("status") == "pending":
                component.update(status="not_evaluated", reason="component_not_reached")
        record["status"] = "complete" if record.get("candidate") else "not_evaluated"
        records.append(record)
        if terminated or budget.attempted >= max_calls:
            break

    planned = len(tasks)
    summary: dict[str, Any] = {"tasks_planned": planned, "tasks_started": len(records), "tasks_completed": sum(1 for item in records if item["status"] == "complete"), "tasks_not_started": planned - len(records), "provider": settings.llm_provider, "model": _redact(settings.resolved_model), "requested_components": sorted(requested), "calls": {"attempted": budget.attempted, "completed": budget.completed, "failed": budget.failed, "budget_exhausted": budget.budget_exhausted, "limit": max_calls}, "sdk_retries": budget.sdk_retries, "terminated": terminated, "elapsed_sec": round(time.monotonic() - started, 3), "records": records}
    safe = _sanitize(summary)
    print(json.dumps(safe, ensure_ascii=True, sort_keys=True))
    if args.report is not None:
        report = args.report if args.report.is_absolute() else PROJECT_ROOT / args.report
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(safe, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    if args.strict and not _strict_ok(records, requested):
        return 1
    return 130 if terminated else 0


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
