"""Append-only jsonl store for SFT samples, abandoned questions, and progress."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cuda_sft.config import get_settings
from cuda_sft.core.sample import training_system, training_user
from cuda_sft.core.selection import MIN_CASES, critic_rejected
from cuda_sft.core.types import snapshot_for_metadata
from cuda_sft.formats import export_training_files
from cuda_sft.parse import wrap_cot_assistant
from cuda_sft.refval.spec import strict_refval_enabled
from cuda_sft.runtime.trace import RUN_ID
from cuda_sft.state import GraphState
from cuda_sft.tasks.kinds import question_hash


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    """Append one JSON object as a line, with an exclusive flock (multi-process).

    Args:
        path: Target jsonl file.
        payload: JSON-serializable record.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, ensure_ascii=False) + "\n"
    # ``a+`` lets recovery separate a truncated final write from the next
    # record. JSONL readers can then skip the damaged fragment safely.
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.seek(0, os.SEEK_END)
            end = handle.tell()
            if end:
                handle.seek(end - 1)
                if handle.read(1) != "\n":
                    handle.seek(0, os.SEEK_END)
                    handle.write("\n")
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _content_hash(value: str) -> str:
    """Hash source text for idempotency metadata without storing extra files."""
    return hashlib.sha256((value or "").encode("utf-8")).hexdigest()


def _sample_key(*, question: str, code: str, dialect: str, question_id: int) -> str:
    """Build an idempotency key containing all user-visible sample inputs."""
    payload = {
        "id": int(question_id),
        "question": question or "",
        "code": code or "",
        "dialect": (dialect or "cuda").strip().lower(),
    }
    return _content_hash(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _find_archived(path: Path, *, sample_key: str) -> bool:
    """Detect an identical new-format row before append.

    Legacy rows without ``sample_key`` remain readable but are not treated as
    duplicates: their old id+code heuristic did not include the question or
    dialect and can incorrectly collapse distinct samples.
    """
    if not path.exists() or not sample_key:
        return False
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                metadata = row.get("metadata")
                if isinstance(metadata, dict) and metadata.get("sample_key") == sample_key:
                    return True
    except OSError:
        return False
    return False


def load_done_keys(progress_path: Path) -> set[tuple[int, str]]:
    """Return ``(question_id, dialect)`` already marked success or abandoned.

    Rows without ``dialect`` are treated as ``cuda`` (legacy progress files).
    """
    done: set[tuple[int, str]] = set()
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
        if status not in {"success", "abandoned"} or not isinstance(qid, int):
            continue
        dialect = row.get("dialect")
        if not isinstance(dialect, str) or not dialect.strip():
            dialect = "cuda"
        done.add((qid, dialect.strip().lower()))
    return done


def load_done_ids(progress_path: Path) -> set[int]:
    """Return question ids that have at least one finished dialect row."""
    return {qid for qid, _dialect in load_done_keys(progress_path)}


def iter_question_rows(path: Path) -> Iterable[tuple[int, str, dict[str, Any]]]:
    """Yield ``(1-based line id, question text, raw object)`` from jsonl.

    Args:
        path: Input jsonl with a ``question`` field per line.
    """
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                continue
            question = obj.get("question")
            if not isinstance(question, str) or not question.strip():
                continue
            yield index, question, obj


def iter_questions(path: Path) -> Iterable[tuple[int, str]]:
    """Yield ``(1-based line id, question text)`` from ``question.jsonl``.

    Args:
        path: Input jsonl with a ``question`` field per line.
    """
    for index, question, _raw in iter_question_rows(path):
        yield index, question


def _strict_refval_blocks_state(state: GraphState, settings: Any) -> bool:
    """Apply the semantic veto and strict numeric evidence gate at publication.

    The gates intentionally read the persisted metadata contract rather
    than convenience fields on ``state``. This prevents an older graph state
    with an optimistic ``refval_ok=True`` from bypassing the gate.
    """
    if str(state.get("kind") or "kernel").strip().lower() == "knowledge":
        return False
    metadata = state.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    critic = metadata.get("critic")
    critic = critic if isinstance(critic, dict) else {}
    if settings.kernel_critic_blocks_save and critic_rejected(critic):
        return True
    if not strict_refval_enabled(settings):
        return False
    refval = metadata.get("refval")
    if not isinstance(refval, dict):
        return True
    if str(refval.get("status") or "").strip().lower() != "pass":
        return True
    cases_run = refval.get("cases_run")
    if isinstance(cases_run, bool) or not isinstance(cases_run, (int, float)):
        return True
    if cases_run < MIN_CASES or not str(refval.get("manifest_hash") or "").strip():
        return True

    if not critic:
        return True
    critic_status = str(critic.get("status") or "").strip().lower()
    # Rows produced before status was added are accepted only when they carry
    # an explicit boolean pass. Error/invalid markers are never inferred as a
    # pass, even if a stale state field says otherwise.
    issues = [str(item).lower() for item in critic.get("issues") or []]
    if not critic_status:
        if any(item.startswith(("critic_error:", "critic_invalid_response")) for item in issues):
            critic_status = "unverified"
        elif critic.get("passed") is False or critic.get("must_fix"):
            critic_status = "failed"
        elif critic.get("passed") is True:
            critic_status = "skipped" if critic.get("skipped") else "verified"
    if critic_status == "unverified":
        return False
    if critic_status == "failed":
        return False
    return not critic_status


def _merge_contract_metadata(metadata: dict[str, Any], state: GraphState) -> None:
    """Copy stable task/oracle/quality/audit fields without dropping legacy metadata."""
    for key in ("task_spec", "oracle_spec", "quality_status", "provenance", "candidate_reports"):
        value = state.get(key)
        if value is not None:
            if key == "candidate_reports" and isinstance(value, list):
                value = [snapshot_for_metadata(item) for item in value if isinstance(item, dict)]
            metadata[key] = value
    # Refval report metadata is normally under ``metadata.refval``.  Keep the
    # top-level hash/provenance fields too so downstream filters need no graph
    # specific knowledge and old rows remain readable.
    refval = metadata.get("refval")
    if isinstance(refval, dict):
        for key in (
            "cache_hash", "manifest_hash", "validator_version", "harness_version",
            "schema_version", "case_suite", "case_suite_version", "provenance",
            "task_spec", "oracle_spec", "oracle_origin",
        ):
            if key in refval and key not in metadata:
                metadata[key] = refval[key]
        metadata["verification_tier"] = str(refval.get("verification_tier") or "none")


@dataclass
class Store:
    """Writes generation outputs under ``data_dir``.

    Files:
        ``sft.jsonl``, ``sft_ms_swift.jsonl``, ``sft_openrlhf.jsonl``,
        ``abandoned.jsonl``, ``progress.jsonl``.

    ``allow_test_sources`` is only for explicit fake-client test harnesses.
    """

    data_dir: Path
    allow_test_sources: bool = False
    sft_path: Path = field(init=False)
    abandoned_path: Path = field(init=False)
    progress_path: Path = field(init=False)
    lock_path: Path = field(init=False)

    def __post_init__(self) -> None:
        """Create ``data_dir`` and bind output paths."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.sft_path = self.data_dir / "sft.jsonl"
        self.swift_path = self.data_dir / "sft_ms_swift.jsonl"
        self.openrlhf_path = self.data_dir / "sft_openrlhf.jsonl"
        self.abandoned_path = self.data_dir / "abandoned.jsonl"
        self.progress_path = self.data_dir / "progress.jsonl"
        self.lock_path = self.data_dir / ".store.lock"

    @contextmanager
    def _locked(self):
        """Hold one directory-wide mutex across dedupe and all artifact writes."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _progress_has(
        path: Path,
        *,
        question_id: int,
        dialect: str,
        sample_key: str,
        status: str = "success",
    ) -> bool:
        """Check terminal success, preferring the new identity key."""
        if not path.exists():
            return False
        try:
            rows = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return False
        for line in rows:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict) or row.get("status") != status:
                continue
            if int(row.get("id") or -1) != int(question_id):
                continue
            row_dialect = str(row.get("dialect") or "cuda").strip().lower()
            if row_dialect != dialect.strip().lower():
                continue
            if sample_key:
                return row.get("sample_key") == sample_key
            return True
        return False

    def _commit_success(
        self,
        *,
        sample: dict[str, Any],
        progress: dict[str, Any],
        sample_key: str,
        cot_in_assistant: bool,
    ) -> bool:
        """Commit canonical row, rebuilt exports, and terminal progress atomically.

        The filesystem cannot provide a multi-file transaction, so ordering is
        deliberate: canonical first, exports second, progress last. If the
        process dies at any point, a later duplicate call sees the canonical
        row, repairs exports, and appends the missing progress record.
        """
        question_id = int(progress["id"])
        dialect = str(progress.get("dialect") or "cuda").strip().lower()
        with self._locked():
            exists = _find_archived(self.sft_path, sample_key=sample_key)
            if not exists:
                _append_jsonl(self.sft_path, sample)
            export_training_files(
                self.sft_path,
                self.data_dir,
                cot_in_assistant=cot_in_assistant,
            )
            if not self._progress_has(
                self.progress_path,
                question_id=question_id,
                dialect=dialect,
                sample_key=sample_key,
            ):
                progress = dict(progress)
                progress["sample_key"] = sample_key
                _append_jsonl(self.progress_path, progress)
        return True

    def write_success(self, state: GraphState, *, model_name: str) -> bool:
        """Persist a passing sample to archive + training jsonl files.

        Args:
            state: Final graph state (must include ``user_prompt`` and ``code``
                or, for knowledge jobs, ``answer``).
            model_name: Provider model id stored in metadata.

        Returns:
            ``True`` when the sample was committed or was already committed;
            ``False`` when strict quality policy quarantined the state.
        """
        kind = str(state.get("kind") or "kernel")
        selected = state.get("selected") if kind == "kernel" else None
        if kind == "kernel":
            if not isinstance(selected, dict) or not selected:
                raise RuntimeError("write_success requires state['selected']")
            if selected["code"] != state.get("code"):
                raise RuntimeError("selected code differs from graph state")
        origin = str(
            (selected.get("origin") if isinstance(selected, dict) else state.get("origin"))
            or "unknown"
        )
        if origin != "live_api" and not self.allow_test_sources:
            raise RuntimeError(f"refusing to save {kind} answer without live API origin: {origin}")
        # A strict refval run must never leak an unvalidated kernel into the
        # primary training files.  This guard also covers legacy graph paths
        # that initialized ``refval_ok=True`` for disabled/skipped validation.
        if _strict_refval_blocks_state(state, get_settings()):
            rejected = dict(state)
            if not rejected.get("abandon_reason"):
                rejected["abandon_reason"] = "strict_quality_gate"
            rejected.setdefault(
                "abandon_error",
                "strict gate requires numeric evidence and an eligible critic result",
            )
            self.write_abandoned(rejected)  # type: ignore[arg-type]
            return False
        if kind == "knowledge":
            return self._write_knowledge_success(state, model_name=model_name)
        assert selected is not None
        settings = get_settings()
        user = training_user(state, selected, settings)
        code = str(selected["code"])
        code_hash = _content_hash(str(code))
        question = str(state.get("question") or "")
        dialect = str(state.get("dialect") or "cuda")
        sample_key = _sample_key(
            question=question,
            code=str(code),
            dialect=dialect,
            question_id=int(state["question_id"]),
        )
        cot = str(state.get("cot") or "")
        quality_state = state.get("quality_status")
        quality_dict = quality_state if isinstance(quality_state, dict) else {}
        if settings.cot_enabled and settings.cot_in_assistant and cot.strip():
            assistant = wrap_cot_assistant(cot, code)
        else:
            assistant = code
        system = training_system(selected, question, settings)
        extra_meta = dict(state.get("metadata") or {})
        dialect = str(state.get("dialect") or "cuda")
        language = "python" if dialect in {"triton", "tilelang"} else "cuda-cpp"
        try:
            source_line = int((state.get("input_metadata") or {}).get("source_line", state["question_id"]))
        except (TypeError, ValueError):
            source_line = int(state["question_id"])
        provenance = dict(state.get("provenance") or {})
        provenance["origin"] = origin
        provenance.setdefault("model", model_name)
        provenance.setdefault("provider", settings.llm_provider)
        provenance.setdefault("prompt_packs", {"generator": "gen-v1", "train_system": "train-sys-v1"})
        provenance.setdefault("run_id", RUN_ID)
        provenance.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        metadata: dict[str, Any] = {
            "candidate": selected["candidate"],
            "repairs": selected["repairs"],
            "arch": state.get("cuda_arch", ""),
            "gpu_name": state.get("gpu_name", ""),
            "model": model_name,
            "used_rdc": state.get("used_rdc", False),
            "judge_score": state.get("judge_score", 0),
            "dialect": dialect,
            "track": dialect,
            "language": language,
            "dataset_version": "cuda-sft-2",
            "system_mode": settings.sft_system_mode,
            "user_mode": "raw_question" if settings.sft_user_is_raw_question else "generation_prompt",
            "selected_code_sha256": selected["code_sha256"],
            "generation": {
                "system": selected["gen_system"],
                "user": selected["gen_user"],
                "prompt_variant": selected["prompt_variant"],
            },
            "code_hash": code_hash,
            "question_hash": question_hash(question),
            "source_line": source_line,
            "sample_key": sample_key,
            "candidate_pool": dict(extra_meta.get("candidate_pool") or {}),
            "provenance": provenance,
            "release_tier": str(
                quality_dict.get("release_tier") or "compile_only"
            ),
        }
        _merge_contract_metadata(metadata, state)
        if extra_meta.get("judge"):
            metadata["judge"] = extra_meta["judge"]
        if extra_meta.get("critic"):
            metadata["critic"] = extra_meta["critic"]
        if extra_meta.get("refval"):
            metadata["refval"] = extra_meta["refval"]
        _merge_contract_metadata(metadata, state)
        for key in ("task_spec", "oracle_spec", "quality_status", "provenance", "candidate_reports"):
            if key not in metadata and extra_meta.get(key) is not None:
                metadata[key] = extra_meta[key]
        metadata["provenance"] = provenance
        if state.get("difficulty"):
            metadata["difficulty"] = state.get("difficulty")
            metadata["candidate_cap"] = state.get("candidate_cap")
        cot_meta = extra_meta.get("cot")
        if not isinstance(cot_meta, dict):
            cot_meta = {}
        metadata["cot"] = {
            "source": state.get("cot_source") or cot_meta.get("source") or "empty",
            "text": cot,
            "policy": str(getattr(settings, "cot_repaired_policy", "synthetic")),
            "chars": len(cot),
            "reasoning_source": state.get("reasoning_source") or cot_meta.get("reasoning_source") or "",
            "raw_chars": cot_meta.get("raw_chars", len(str(state.get("raw_reasoning") or ""))),
            "polished_chars": len(cot),
            "error": state.get("cot_error") or cot_meta.get("error") or "",
        }
        if "consistency_issues" in cot_meta:
            metadata["cot"]["consistency_issues"] = list(cot_meta["consistency_issues"])
        if "soft_issues" in cot_meta:
            metadata["cot"]["soft_issues"] = list(cot_meta["soft_issues"])
        if settings.cot_enabled:
            raw = str(state.get("raw_reasoning") or "")
            limit = settings.cot_raw_store_max_chars
            if raw:
                if limit > 0 and len(raw) > limit:
                    raw = raw[:limit].rstrip() + "\n...[truncated reasoning]..."
                metadata["raw_reasoning"] = raw
        sample = {
            "id": state["question_id"],
            "messages": [
                *([{"role": "system", "content": system}] if system else []),
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ],
            "metadata": metadata,
        }
        return self._commit_success(
            sample=sample,
            progress={
                "id": state["question_id"],
                "dialect": dialect,
                "status": "success",
                "candidate": selected["candidate"],
                "repairs": selected["repairs"],
            },
            sample_key=sample_key,
            cot_in_assistant=bool(settings.cot_in_assistant),
        )

    def _write_knowledge_success(self, state: GraphState, *, model_name: str) -> bool:
        """Persist a rubric-passing knowledge sample (prose assistant)."""
        from cuda_sft.knowledge.judge import accepted_knowledge_answer

        settings = get_settings()
        if not accepted_knowledge_answer(state, settings):
            rejected = dict(state)
            rejected["abandon_reason"] = "strict_quality_gate"
            rejected["abandon_error"] = "knowledge answer lacks valid live judge evidence"
            self.write_abandoned(rejected)
            return False
        question = str(state.get("question") or "")
        gen_system = str(state.get("gen_system") or "")
        if settings.sft_system_mode == "generation" and not gen_system:
            raise RuntimeError("knowledge success requires state['gen_system']")
        gen_user = str(state.get("gen_user") or state.get("user_prompt") or "")
        generation = {
            "gen_system": gen_system,
            "gen_user": gen_user,
        }
        user = training_user(state, generation, settings)
        answer = str(state.get("answer") or "")
        cot = str(state.get("cot") or "")
        if settings.cot_enabled and settings.cot_in_assistant and cot.strip():
            assistant = wrap_cot_assistant(cot, answer)
        else:
            assistant = answer
        system = training_system(generation, question, settings, kind="knowledge")
        extra_meta = dict(state.get("metadata") or {})
        topic = str(state.get("topic") or "general")
        track = str(state.get("track") or f"knowledge:{topic}")
        sample_key = _sample_key(
            question=question,
            code=answer,
            dialect=track,
            question_id=int(state["question_id"]),
        )
        try:
            source_line = int((state.get("input_metadata") or {}).get("source_line", state["question_id"]))
        except (TypeError, ValueError):
            source_line = int(state["question_id"])
        provenance = dict(state.get("provenance") or {})
        provenance["origin"] = str(state.get("origin") or "unknown")
        provenance.setdefault("model", model_name)
        provenance.setdefault("provider", settings.llm_provider)
        provenance.setdefault("prompt_packs", {"knowledge_generator": "gen-v1", "train_system": "train-sys-v1"})
        provenance.setdefault("run_id", RUN_ID)
        provenance.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        metadata: dict[str, Any] = {
            "task": "knowledge",
            "topic": topic,
            "candidate": state.get("candidate_idx", 1),
            "repairs": state.get("repair_idx", 0),
            "arch": state.get("cuda_arch", ""),
            "gpu_name": state.get("gpu_name", ""),
            "model": model_name,
            "judge_score": state.get("judge_score", 0),
            "dialect": track,
            "track": track,
            "language": "prose",
            "dataset_version": "cuda-sft-2",
            "system_mode": settings.sft_system_mode,
            "user_mode": "raw_question" if settings.sft_user_is_raw_question else "generation_prompt",
            "generation": {
                "system": gen_system,
                "user": gen_user,
                "prompt_variant": dict(state.get("gen_prompt_variant") or {}),
            },
            "question_hash": question_hash(question),
            "source_line": source_line,
            "code_hash": _content_hash(answer),
            "sample_key": sample_key,
            "release_tier": "strict",
            "provenance": provenance,
        }
        _merge_contract_metadata(metadata, state)
        if extra_meta.get("knowledge_judge"):
            metadata["knowledge_judge"] = extra_meta["knowledge_judge"]
        _merge_contract_metadata(metadata, state)
        for key in ("task_spec", "oracle_spec", "quality_status", "provenance", "candidate_reports"):
            if key not in metadata and extra_meta.get(key) is not None:
                metadata[key] = extra_meta[key]
        metadata["provenance"] = provenance
        if state.get("difficulty"):
            metadata["difficulty"] = state.get("difficulty")
            metadata["candidate_cap"] = state.get("candidate_cap")
        cot_meta = extra_meta.get("cot")
        if not isinstance(cot_meta, dict):
            cot_meta = {}
        metadata["cot"] = {
            "source": state.get("cot_source") or cot_meta.get("source") or "empty",
            "text": cot,
            "policy": str(getattr(settings, "cot_repaired_policy", "synthetic")),
            "chars": len(cot),
            "reasoning_source": state.get("reasoning_source") or cot_meta.get("reasoning_source") or "",
            "raw_chars": cot_meta.get("raw_chars", len(str(state.get("raw_reasoning") or ""))),
            "polished_chars": len(cot),
            "error": state.get("cot_error") or cot_meta.get("error") or "",
        }
        if settings.cot_enabled:
            raw = str(state.get("raw_reasoning") or "")
            limit = settings.cot_raw_store_max_chars
            if raw:
                if limit > 0 and len(raw) > limit:
                    raw = raw[:limit].rstrip() + "\n...[truncated reasoning]..."
                metadata["raw_reasoning"] = raw
        sample = {
            "id": state["question_id"],
            "messages": [
                *([{"role": "system", "content": system}] if system else []),
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ],
            "metadata": metadata,
        }
        return self._commit_success(
            sample=sample,
            progress={
                "id": state["question_id"],
                "dialect": track,
                "task": "knowledge",
                "topic": topic,
                "status": "success",
                "candidate": state.get("candidate_idx", 1),
                "repairs": state.get("repair_idx", 0),
            },
            sample_key=sample_key,
            cot_in_assistant=bool(settings.cot_in_assistant),
        )

    def _commit_abandoned(self, record: dict[str, Any], progress: dict[str, Any]) -> None:
        """Append an abandoned record and its progress marker under one lock."""
        qid = int(progress["id"])
        dialect = str(progress.get("dialect") or "cuda").strip().lower()
        question_hash = _content_hash(str(record.get("question") or ""))
        reason = str(record.get("reason") or "")
        with self._locked():
            duplicate = False
            if self.abandoned_path.exists():
                for line in self.abandoned_path.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines():
                    try:
                        old = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(old, dict):
                        continue
                    if (
                        int(old.get("id") or -1) == qid
                        and str(old.get("dialect") or "cuda").strip().lower() == dialect
                        and _content_hash(str(old.get("question") or "")) == question_hash
                        and str(old.get("reason") or "") == reason
                    ):
                        duplicate = True
                        break
            if not duplicate:
                _append_jsonl(self.abandoned_path, record)
            if not self._progress_has(
                self.progress_path,
                question_id=qid,
                dialect=dialect,
                sample_key="",
                status="abandoned",
            ):
                progress = dict(progress)
                _append_jsonl(self.progress_path, progress)

    def write_abandoned(self, state: GraphState) -> None:
        """Record a question where all candidates failed the quality gate.

        Args:
            state: Graph state after the last failed candidate.
        """
        kind = str(state.get("kind") or "kernel")
        if kind == "knowledge":
            topic = str(state.get("topic") or "general")
            dialect = str(state.get("track") or f"knowledge:{topic}")
            reason = str(state.get("abandon_reason") or "knowledge_quality")
            last_error = (
                state.get("judge_error")
                or "; ".join(state.get("judge_must_fix") or [])
                or "; ".join(state.get("gate_reasons") or [])
                or "; ".join(state.get("judge_issues") or [])
            )
            record = {
                "id": state["question_id"],
                "dialect": dialect,
                "task": "knowledge",
                "topic": topic,
                "question": state.get("question", ""),
                "reason": reason,
                "last_error": last_error,
                "attempts": state.get("attempts", []),
            }
            progress = {
                "id": state["question_id"],
                "dialect": dialect,
                "task": "knowledge",
                "status": "abandoned",
                "reason": reason,
            }
            self._commit_abandoned(record, progress)
            return
        dialect = str(state.get("dialect") or "cuda")
        record = {
            "id": state["question_id"],
            "dialect": dialect,
            "track": dialect,
            "question": state.get("question", ""),
            "question_hash": question_hash(str(state.get("question") or "")),
            "abandon_reason": str(state.get("abandon_reason") or "all_candidates_failed"),
            "reason": str(state.get("abandon_reason") or "all_candidates_failed"),
            "last_error": state.get("abandon_error") or state.get("compile_error", ""),
            "attempts": state.get("attempts", []),
            "candidate_errors": state.get("candidate_errors", []),
            "candidates": [
                {
                    "candidate": item.get("candidate"),
                    "repairs": item.get("repairs"),
                    "last_gate": dict(item.get("last_gate") or {}),
                    "code_sha256": item.get("code_sha256"),
                    "oracle_blocked": bool(item.get("oracle_blocked")),
                }
                for item in (state.get("candidate_reports") or [])
                if isinstance(item, dict)
            ],
            "run_id": RUN_ID,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self._commit_abandoned(
            record,
            {
                "id": state["question_id"],
                "dialect": dialect,
                "status": "abandoned",
                "reason": record["reason"],
            },
        )


_store: Store | None = None


def init_store(data_dir: Path, *, allow_test_sources: bool = False) -> Store:
    """Create the process-global :class:`Store` (call once per worker).

    Args:
        data_dir: Output directory.
        allow_test_sources: Permit non-live completions in test harnesses.
    """
    global _store
    _store = Store(data_dir, allow_test_sources=allow_test_sources)
    return _store


def get_store() -> Store:
    """Return the store initialized by :func:`init_store`.

    Raises:
        RuntimeError: If :func:`init_store` has not been called.
    """
    if _store is None:
        raise RuntimeError("Store has not been initialized")
    return _store
