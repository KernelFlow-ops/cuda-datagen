"""Append-only jsonl store for SFT samples, abandoned questions, and progress."""

from __future__ import annotations

import fcntl
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from cuda_sft.formats import to_ms_swift, to_openrlhf
from cuda_sft.state import GraphState


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    """Append one JSON object as a line, with an exclusive flock (multi-process).

    Args:
        path: Target jsonl file.
        payload: JSON-serializable record.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, ensure_ascii=False) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def load_done_ids(progress_path: Path) -> set[int]:
    """Return question ids already marked ``success`` or ``abandoned``.

    Args:
        progress_path: ``data/progress.jsonl``.
    """
    done: set[int] = set()
    if not progress_path.exists():
        return done
    with progress_path.open("r", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
        try:
            lines = handle.readlines()
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        status = row.get("status")
        qid = row.get("id")
        if status in {"success", "abandoned"} and isinstance(qid, int):
            done.add(qid)
    return done


def iter_questions(path: Path) -> Iterable[tuple[int, str]]:
    """Yield ``(1-based line id, question text)`` from ``question.jsonl``.

    Args:
        path: Input jsonl with a ``question`` field per line.
    """
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            question = obj.get("question")
            if not isinstance(question, str) or not question.strip():
                continue
            yield index, question


@dataclass
class Store:
    """Writes generation outputs under ``data_dir``.

    Files:
        ``sft.jsonl``, ``sft_ms_swift.jsonl``, ``sft_openrlhf.jsonl``,
        ``abandoned.jsonl``, ``progress.jsonl``.
    """

    data_dir: Path
    sft_path: Path = field(init=False)
    abandoned_path: Path = field(init=False)
    progress_path: Path = field(init=False)

    def __post_init__(self) -> None:
        """Create ``data_dir`` and bind output paths."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.sft_path = self.data_dir / "sft.jsonl"
        self.swift_path = self.data_dir / "sft_ms_swift.jsonl"
        self.openrlhf_path = self.data_dir / "sft_openrlhf.jsonl"
        self.abandoned_path = self.data_dir / "abandoned.jsonl"
        self.progress_path = self.data_dir / "progress.jsonl"

    def write_success(self, state: GraphState, *, model_name: str) -> None:
        """Persist a compile-passing sample to archive + training jsonl files.

        Args:
            state: Final graph state (must include ``user_prompt`` and ``code``).
            model_name: Provider model id stored in metadata.
        """
        user = state["user_prompt"]
        assistant = state["code"]
        system = str(state.get("system_prompt") or "")
        sample = {
            "id": state["question_id"],
            "messages": [
                *([{"role": "system", "content": system}] if system.strip() else []),
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ],
            "metadata": {
                "candidate": state.get("candidate_idx", 1),
                "repairs": state.get("repair_idx", 0),
                "arch": state.get("cuda_arch", ""),
                "gpu_name": state.get("gpu_name", ""),
                "model": model_name,
                "used_rdc": state.get("used_rdc", False),
                "system": system,
            },
        }
        _append_jsonl(self.sft_path, sample)
        _append_jsonl(self.swift_path, to_ms_swift(user, assistant, system=system))
        _append_jsonl(self.openrlhf_path, to_openrlhf(user, assistant, system=system))
        _append_jsonl(
            self.progress_path,
            {
                "id": state["question_id"],
                "status": "success",
                "candidate": state.get("candidate_idx", 1),
                "repairs": state.get("repair_idx", 0),
            },
        )

    def write_abandoned(self, state: GraphState) -> None:
        """Record a question where all candidates failed to compile.

        Args:
            state: Graph state after the last failed candidate.
        """
        record = {
            "id": state["question_id"],
            "question": state.get("question", ""),
            "reason": "all_candidates_failed",
            "last_error": state.get("compile_error", ""),
            "attempts": state.get("attempts", []),
        }
        _append_jsonl(self.abandoned_path, record)
        _append_jsonl(
            self.progress_path,
            {
                "id": state["question_id"],
                "status": "abandoned",
                "reason": "all_candidates_failed",
            },
        )


_store: Store | None = None


def init_store(data_dir: Path) -> Store:
    """Create the process-global :class:`Store` (call once per worker).

    Args:
        data_dir: Output directory.
    """
    global _store
    _store = Store(data_dir)
    return _store


def get_store() -> Store:
    """Return the store initialized by :func:`init_store`.

    Raises:
        RuntimeError: If :func:`init_store` has not been called.
    """
    if _store is None:
        raise RuntimeError("Store has not been initialized")
    return _store
