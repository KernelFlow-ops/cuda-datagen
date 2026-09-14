from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from cuda_sft.formats import to_ms_swift, to_openrlhf
from cuda_sft.state import GraphState


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_done_ids(progress_path: Path) -> set[int]:
    done: set[int] = set()
    if not progress_path.exists():
        return done
    with progress_path.open("r", encoding="utf-8") as handle:
        for line in handle:
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
    data_dir: Path
    sft_path: Path = field(init=False)
    abandoned_path: Path = field(init=False)
    progress_path: Path = field(init=False)

    def __post_init__(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.sft_path = self.data_dir / "sft.jsonl"
        self.swift_path = self.data_dir / "sft_ms_swift.jsonl"
        self.openrlhf_path = self.data_dir / "sft_openrlhf.jsonl"
        self.abandoned_path = self.data_dir / "abandoned.jsonl"
        self.progress_path = self.data_dir / "progress.jsonl"

    def write_success(self, state: GraphState, *, model_name: str) -> None:
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
    global _store
    _store = Store(data_dir)
    return _store


def get_store() -> Store:
    if _store is None:
        raise RuntimeError("Store has not been initialized")
    return _store
