"""Fakes for LLM, compiler, refval and sleep (spec 05_testing/03_l2_scenarios.md §2)."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from cuda_sft.runtime.cancel import Cancelled
except ImportError:
    # T3.12 supplies the runtime cancellation type; S0 fakes also run alone.
    class Cancelled(asyncio.CancelledError):
        pass


MARKER_RE = re.compile(r"^\s*(?://|#)\s*@fake\s+(?P<body>.+?)\s*$")
FENCE_BY_SUFFIX = {".cu": "cuda", ".cuh": "cuda", ".cpp": "cpp", ".py": "python"}

# Typical compiler lines recognised by agents.repairer.classify_compile_error.
COMPILE_ERROR_SAMPLES = {
    "syntax": 'solution.cu(12): error: expected a ";"',
    "undeclared": 'solution.cu(7): error: identifier "blockSize" is undefined',
    "missing_header": "solution.cu:3:10: fatal error: helpers.h: No such file or directory",
    "template": 'solution.cu(20): error: no matching function for call to "launch<float>"',
    "other": "solution.cu(1): error: unknown failure",
}

# Kept literal until T1.4 introduces the REASON_* constants in refval/runner.py.
REASON_TIMEOUT_BEFORE_GPU = "timeout before GPU run"
REASON_GPU_LOCK_TIMEOUT = "timeout waiting for GPU lock"
REASON_EXTRACT_FAILED = "ABI/reference extraction failed"


class UnscriptedCall(AssertionError):
    """The graph made an LLM call the scenario did not script."""


def parse_fake_markers(code: str) -> dict[str, str]:
    """Parse the first ``@fake`` marker within the first 5 lines of ``code``."""
    for line in (code or "").splitlines()[:5]:
        m = MARKER_RE.match(line)
        if not m:
            continue
        out: dict[str, str] = {}
        for token in m.group("body").split():
            if "=" in token:
                k, v = token.split("=", 1)
                out[k.strip()] = v.strip()
        return out
    return {}


def strip_marker(code: str) -> str:
    """Remove marker lines (used when comparing selected code to fixtures)."""
    return "\n".join(
        line for line in (code or "").splitlines() if not MARKER_RE.match(line)
    ).strip()


def _sha8(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


# --------------------------------------------------------------------------- LLM


@dataclass
class LLMCallRecord:
    role: str
    key: str
    purpose: str
    system: str
    messages: list[dict[str, str]]
    reply_key: str
    t_start: float
    t_end: float = 0.0
    cancelled: bool = False

    @property
    def user_text(self) -> str:
        return "\n".join(m.get("content", "") for m in self.messages if m.get("role") == "user")


@dataclass
class FakeLLM:
    """Scripted LLM. ``script`` = {role: {lookup_key: reply_obj}} (YAML ``llm`` section)."""

    script: dict[str, dict[str, Any]]
    fixtures_root: Path
    calls: list[LLMCallRecord] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def factory(self, role: str) -> _RoleClient:
        return _RoleClient(self, role)

    # -- lookup ------------------------------------------------------------
    def _role_table(self, role: str) -> tuple[str, dict[str, Any]]:
        if role in self.script:
            return role, self.script[role]
        if role.startswith("repair.") and "repair.compile" in self.script:
            return "repair.compile", self.script["repair.compile"]
        if role.startswith("repair.") and "generator" in self.script:
            return "generator", self.script["generator"]
        return role, {}

    def lookup(self, role: str, key: str, purpose: str, user_text: str = "") -> tuple[str, Any]:
        _, table = self._role_table(role)
        for k in (f"{key}.{purpose}" if purpose else None, key, purpose or None, "default"):
            if k and k in table:
                return k, table[k]
        raise UnscriptedCall(
            f"unscripted LLM call role={role} key={key} purpose={purpose!r} "
            f"user={user_text[:300]!r}"
        )

    # -- reply rendering ----------------------------------------------------
    def render(self, reply: Any) -> tuple[str, str]:
        """Return (text, reasoning)."""
        if isinstance(reply, str):
            return reply, ""
        reasoning = str(reply.get("reasoning", ""))
        if "raise" in reply:
            kind = reply["raise"]
            if kind == "timeout":
                raise TimeoutError("fake LLM timeout")
            if kind == "cancelled":
                raise Cancelled("fake cancel")
            if kind in {"rate_limit", "bad_request"}:
                import httpx
                import openai

                status = 429 if kind == "rate_limit" else 400
                response = httpx.Response(
                    status,
                    request=httpx.Request("POST", "https://fake.invalid/v1/chat/completions"),
                    json={"error": {"message": f"fake {kind}", "type": kind}},
                )
                error_type = openai.RateLimitError if status == 429 else openai.BadRequestError
                raise error_type(f"fake {kind}", response=response, body=response.json())
            raise ValueError(f"unknown fake LLM error: {kind}")
        if "code" in reply:
            path = self.fixtures_root / reply["code"]
            fence = FENCE_BY_SUFFIX.get(path.suffix, "")
            return f"```{fence}\n{path.read_text(encoding='utf-8')}\n```", reasoning
        if "text" in reply:
            return str(reply["text"]), reasoning
        if "text_file" in reply:
            return (self.fixtures_root / reply["text_file"]).read_text(encoding="utf-8"), reasoning
        if "json" in reply:
            return json.dumps(reply["json"], ensure_ascii=False), reasoning
        if "json_file" in reply:
            return (self.fixtures_root / reply["json_file"]).read_text(encoding="utf-8"), reasoning
        raise ValueError(f"bad fake reply object: {reply!r}")

    def complete(
        self,
        role: str,
        *,
        messages: list[dict[str, str]],
        system: str,
        meta: Any,
        token: Any = None,
    ) -> Any:
        from cuda_sft.llm import LLMCompletion  # S3+: package re-exports the same name

        if meta is None:
            raise UnscriptedCall(f"LLM call without CallMeta (role={role}); see T0.3")
        key, purpose = meta.key(), meta.purpose
        rec = LLMCallRecord(role, key, purpose, system, list(messages), "", time.monotonic())
        with self._lock:
            self.calls.append(rec)
        try:
            reply_key, reply = self.lookup(role, key, purpose, rec.user_text)
            rec.reply_key = reply_key
            latency = float(reply.get("latency_s", 0) or 0) if isinstance(reply, dict) else 0.0
            if latency > 0:
                deadline = time.monotonic() + latency
                while time.monotonic() < deadline:
                    if token is not None and getattr(token, "cancelled", False):
                        raise Cancelled("cancelled during fake latency")
                    time.sleep(0.05)
            text, reasoning = self.render(reply)
            if isinstance(reply, dict) and "usage" in reply:
                usage = {k: int(reply["usage"][k]) for k in ("input_tokens", "output_tokens")}
                estimated = False
            else:
                from cuda_sft.llm import approx_tokens

                usage = {
                    "input_tokens": approx_tokens(system)
                    + sum(approx_tokens(m.get("content", "")) for m in messages),
                    "output_tokens": approx_tokens(text + reasoning),
                }
                estimated = True
            return LLMCompletion(
                text=text,
                reasoning=reasoning,
                reasoning_source="api" if reasoning else "empty",
                usage=usage,
                tokens_estimated=estimated,
            )
        except Cancelled:
            rec.cancelled = True
            raise
        finally:
            rec.t_end = time.monotonic()

    def calls_for(self, role_prefix: str) -> list[LLMCallRecord]:
        if role_prefix.endswith("*"):
            p = role_prefix[:-1]
            return [c for c in self.calls if c.role.startswith(p)]
        return [c for c in self.calls if c.role == role_prefix]


@dataclass
class _RoleClient:
    """Duck-types llm.LLMClient for one role."""

    fake: FakeLLM
    role: str

    def stream_completion(
        self, *, messages, system, temperature=0.0, print_stream=False, meta=None, token=None, **_kw
    ):  # type: ignore[no-untyped-def]
        role = meta.role if meta is not None else self.role
        return self.fake.complete(role, messages=messages, system=system, meta=meta, token=token)

    def stream_text(
        self, *, messages, system, temperature=0.0, print_stream=False, meta=None, token=None, **_kw
    ):  # type: ignore[no-untyped-def]
        return self.stream_completion(messages=messages, system=system, meta=meta, token=token).text


# ------------------------------------------------------------------ compile/refval


@dataclass
class FakeCompiler:
    calls: list[dict[str, Any]] = field(default_factory=list)

    def __call__(self, dialect: str, code: str, workdir: Path, settings: Any) -> Any:
        from cuda_sft.compile import CompileResult

        workdir.mkdir(parents=True, exist_ok=True)
        marks = parse_fake_markers(code)
        spec = marks.get("compile", "pass")
        delay = float(marks.get("latency_compile", 0) or 0)
        if delay:
            time.sleep(delay)
        self.calls.append({"dialect": dialect, "code_sha8": _sha8(code), "spec": spec})
        if spec == "pass":
            return CompileResult(
                ok=True, command=["fake-nvcc"], output="", used_rdc=False, dialect=dialect
            )
        if spec == "infra":
            return CompileResult(
                ok=False,
                command=["fake-nvcc"],
                output="failed to launch nvcc",
                used_rdc=False,
                dialect=dialect,
            )
        cls = spec.split(":", 1)[1] if ":" in spec else "syntax"
        line = COMPILE_ERROR_SAMPLES.get(cls, COMPILE_ERROR_SAMPLES["other"])
        return CompileResult(
            ok=False,
            command=["fake-nvcc"],
            output=f"fake compile log: {cls}\n{line}",
            used_rdc=False,
            dialect=dialect,
        )


@dataclass
class FakeRefval:
    calls: list[dict[str, Any]] = field(default_factory=list)
    _seen: dict[str, int] = field(default_factory=dict)

    def __call__(self, **kwargs: Any) -> Any:
        from cuda_sft.refval.spec import RefvalReport

        code = str(kwargs.get("code", ""))
        dialect = str(kwargs.get("dialect", "cuda"))
        marks = parse_fake_markers(code)
        sha = _sha8(code)
        n = self._seen.get(sha, 0)
        self._seen[sha] = n + 1
        if "refval_seq" in marks:
            seq = [s for s in marks["refval_seq"].split(",") if s]
            spec = seq[min(n, len(seq) - 1)]
        else:
            spec = marks.get("refval", "pass")
        delay = float(marks.get("latency_refval", 0) or 0)
        if delay:
            time.sleep(delay)
        manifest = kwargs.get("manifest")
        self.calls.append(
            {
                "dialect": dialect,
                "code_sha8": sha,
                "spec": spec,
                "manifest_source": getattr(manifest, "extracted_from", None),
            }
        )
        cases = int(marks.get("cases", 12))
        mh = f"fake-{sha}"
        if spec == "pass":
            return RefvalReport(status="pass", dialect=dialect, cases_run=cases, manifest_hash=mh)
        if spec == "skip":
            return RefvalReport(status="skip", dialect=dialect, reason="refval disabled (fake)")
        if spec == "reference_error":
            return RefvalReport(
                status="reference_error",
                dialect=dialect,
                error_class="reference_error",
                reason="reference raised (fake)",
                manifest_hash=mh,
            )
        if spec == "timeout_before_gpu":
            return RefvalReport(
                status="fail",
                dialect=dialect,
                error_class="timeout",
                reason=REASON_TIMEOUT_BEFORE_GPU,
            )
        if spec == "gpu_lock_timeout":
            return RefvalReport(
                status="fail",
                dialect=dialect,
                error_class="timeout",
                reason=REASON_GPU_LOCK_TIMEOUT,
            )
        if spec == "extract_failed":
            return RefvalReport(
                status="reference_error",
                dialect=dialect,
                error_class="reference_error",
                reason=REASON_EXTRACT_FAILED,
            )
        cls = spec.split(":", 1)[1] if ":" in spec else "numeric_mismatch"
        return RefvalReport(
            status="fail",
            dialect=dialect,
            cases_run=cases,
            failed_case="standard_1024",
            error_class=cls,
            reason=f"fake {cls}",
            manifest_hash=mh,
            evidence=f"case standard_1024: first mismatch at index 17 ({cls})",
        )


@dataclass
class FakeSleep:
    calls: list[float] = field(default_factory=list)

    def __call__(self, seconds: float) -> None:
        self.calls.append(float(seconds))
