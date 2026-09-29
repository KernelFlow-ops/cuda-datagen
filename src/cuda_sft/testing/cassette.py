"""Record and replay model completions by role and exact request content."""

from __future__ import annotations

import difflib
import fcntl
import hashlib
import json
import os
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cuda_sft.llm import LLMCompletion, approx_tokens
from cuda_sft.runtime import trace
from cuda_sft.runtime.meta import CallMeta


class CassetteMiss(LookupError):
    """No recorded response matches this exact model request."""


def _identity(meta: CallMeta | None, system: str, messages: list[dict[str, str]]) -> dict[str, Any]:
    role = meta.role if meta is not None else "generator"
    request = system + json.dumps(
        messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return {
        "role": role,
        "candidate": meta.candidate if meta is not None else 0,
        "repair": meta.repair if meta is not None else 0,
        "purpose": meta.purpose if meta is not None else "",
        "request_hash": hashlib.sha256(request.encode("utf-8")).hexdigest(),
    }


def _key_hash(identity: dict[str, Any]) -> str:
    raw = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _append_index(directory: Path, identity: dict[str, Any], key_hash: str) -> None:
    with (directory / "index.jsonl").open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.write(json.dumps({"key_hash": key_hash, **identity}, ensure_ascii=False) + "\n")
            handle.flush()
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _index_contains(directory: Path, key_hash: str) -> bool:
    index = directory / "index.jsonl"
    if not index.exists():
        return False
    with index.open("r", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
        try:
            for line in handle:
                try:
                    row = json.loads(line)
                    if isinstance(row, dict) and row.get("key_hash") == key_hash:
                        return True
                except json.JSONDecodeError:
                    continue
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return False


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.stem}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temp = Path(handle.name)
        try:
            json.dump(payload, handle, ensure_ascii=False)
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
    try:
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _write_run_meta(directory: Path, client: Any) -> None:
    dest = directory / "run_meta.json"
    with (directory / ".run_meta.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            if dest.exists():
                return
            settings = getattr(client, "settings", None)
            model = getattr(settings, "resolved_model", None) or "unknown"
            try:
                revision = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=Path(__file__).resolve().parents[3],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=5,
                )
                commit = revision.stdout.strip() if revision.returncode == 0 else "unknown"
            except (OSError, subprocess.TimeoutExpired):
                commit = "unknown"
            _write_json_atomic(
                dest,
                {
                    "model": model,
                    "date": datetime.now(timezone.utc).date().isoformat(),
                    "git_commit": commit,
                },
            )
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _emit_replay_call(
    *,
    meta: CallMeta | None,
    system: str,
    messages: list[dict[str, str]],
    started: float,
    completion: LLMCompletion | None = None,
    error: BaseException | None = None,
) -> None:
    usage = completion.usage if completion is not None else None
    estimated = usage is None or (completion.tokens_estimated if completion is not None else True)
    if usage is None:
        usage = {
            "input_tokens": approx_tokens(system)
            + sum(approx_tokens(message.get("content") or "") + 4 for message in messages),
            "output_tokens": approx_tokens(completion.text + completion.reasoning)
            if completion
            else 0,
            "reasoning_tokens": approx_tokens(completion.reasoning) if completion else 0,
        }
    trace.emit(
        "llm.call",
        job_key=meta.job_key if meta else "",
        candidate=meta.candidate if meta else 0,
        repair=meta.repair if meta else 0,
        role=meta.role if meta else "generator",
        purpose=meta.purpose if meta else "",
        provider="cassette",
        model="replay",
        key_label="cassette",
        attempt=meta.attempt if meta else 1,
        elapsed_s=round(time.monotonic() - started, 4),
        ttft_s=None,
        input_tokens=int(usage.get("input_tokens") or 0),
        output_tokens=int(usage.get("output_tokens") or 0),
        reasoning_tokens=int(usage.get("reasoning_tokens") or 0),
        tokens_estimated=estimated,
        ok=error is None,
        error_type=type(error).__name__ if error else "",
        retryable=False,
        cancelled=False,
        cost_usd=None,
    )


class RecordingClient:
    def __init__(self, client: Any, directory: Path) -> None:
        self.client = client
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def stream_completion(
        self,
        *,
        messages: list[dict[str, str]],
        system: str,
        temperature: float,
        meta: CallMeta | None = None,
        **kwargs: Any,
    ) -> LLMCompletion:
        result = self.client.stream_completion(
            messages=messages, system=system, temperature=temperature, meta=meta, **kwargs
        )
        identity = _identity(meta, system, messages)
        key_hash = _key_hash(identity)
        payload = {
            "identity": identity,
            "text": result.text,
            "reasoning": result.reasoning,
            "reasoning_source": result.reasoning_source,
            "usage": result.usage,
            "tokens_estimated": result.tokens_estimated,
        }
        dest = self.directory / f"{key_hash}.json"
        # Serialize writes for a key so identical concurrent calls are idempotent.
        with (self.directory / f".{key_hash}.lock").open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                if dest.exists():
                    existing = json.loads(dest.read_text(encoding="utf-8"))
                    if {"tokens_estimated": False, **existing} != payload:
                        raise ValueError(f"conflicting cassette response for key {key_hash}")
                else:
                    _write_json_atomic(dest, payload)
                if not _index_contains(self.directory, key_hash):
                    _append_index(self.directory, identity, key_hash)
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        _write_run_meta(self.directory, self.client)
        return result

    def stream_text(self, **kwargs: Any) -> str:
        return self.stream_completion(**kwargs).text


class ReplayClient:
    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)

    def stream_completion(
        self,
        *,
        messages: list[dict[str, str]],
        system: str,
        temperature: float,
        meta: CallMeta | None = None,
        **kwargs: Any,
    ) -> LLMCompletion:
        started = time.monotonic()
        identity = _identity(meta, system, messages)
        key_hash = _key_hash(identity)
        path = self.directory / f"{key_hash}.json"
        try:
            if not path.exists():
                index = self.directory / "index.jsonl"
                rows = []
                if index.exists():
                    for line in index.read_text(encoding="utf-8").splitlines():
                        if line.strip():
                            try:
                                row = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            if all(key in row for key in identity):
                                rows.append(row)
                target = json.dumps(identity, sort_keys=True)
                near = difflib.get_close_matches(
                    target,
                    [json.dumps({k: row[k] for k in identity}, sort_keys=True) for row in rows],
                    n=3,
                    cutoff=0,
                )
                raise CassetteMiss(f"cassette miss: {identity}; closest: {near}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            result = LLMCompletion(
                text=payload["text"],
                reasoning=payload.get("reasoning", ""),
                reasoning_source=payload.get("reasoning_source", "empty"),
                usage=payload.get("usage"),
                tokens_estimated=payload.get("tokens_estimated", False),
                origin="replay",
            )
        except Exception as exc:
            _emit_replay_call(
                meta=meta, system=system, messages=messages, started=started, error=exc
            )
            raise
        _emit_replay_call(
            meta=meta, system=system, messages=messages, started=started, completion=result
        )
        return result

    def stream_text(self, **kwargs: Any) -> str:
        return self.stream_completion(**kwargs).text
