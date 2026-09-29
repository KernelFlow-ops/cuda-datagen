"""Convert generation archive rows to ms-swift / OpenRLHF SFT jsonl."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

from cuda_sft.config import get_settings
from cuda_sft.parse import (
    extract_cuda_source,
    extract_thinking,
    strip_thinking,
    wrap_cot_assistant,
)
from cuda_sft.prompt import SYSTEM_PROMPT


def _content(msg: dict[str, Any]) -> str:
    """Return the string ``content`` of a chat message dict, or empty."""
    value = msg.get("content")
    return value if isinstance(value, str) else ""


def split_user_assistant(row: dict[str, Any]) -> tuple[str, str] | None:
    """Return (user, assistant) from a generated SFT row."""
    messages = row.get("messages")
    user = ""
    assistant = ""
    if isinstance(messages, list):
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            if role == "user":
                user = _content(msg)
            elif role == "assistant":
                assistant = _content(msg)
    if not user:
        user = str(row.get("user_prompt") or row.get("question") or "")
    if not assistant:
        assistant = str(row.get("code") or row.get("output") or "")
    if not user.strip() or not assistant.strip():
        return None
    return user, assistant


def extract_system(row: dict[str, Any]) -> str:
    """Read only an explicit system message from the archived sample.

    Args:
        row: One jsonl object from ``sft.jsonl``.
    """
    messages = row.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "system":
                text = _content(msg)
                if text.strip():
                    return text
    return ""


def to_ms_swift(user: str, assistant: str, *, system: str = SYSTEM_PROMPT) -> dict[str, Any]:
    """Build one ms-swift SFT record (``messages`` only).

    See Custom-dataset.md in modelscope/ms-swift.

    Args:
        user: Instruction (question + generation suffix).
        assistant: CUDA source used as the label.
        system: Optional system prompt prepended to ``messages``.
    """
    messages: list[dict[str, str]] = []
    if system.strip():
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    messages.append({"role": "assistant", "content": assistant})
    return {"messages": messages}


def to_openrlhf(user: str, assistant: str, *, system: str = SYSTEM_PROMPT) -> dict[str, Any]:
    """Build one OpenRLHF SFT record (``input`` / ``output`` chat lists).

    Train with ``--input_key input --output_key output --apply_chat_template``.

    Args:
        user: Instruction text.
        assistant: CUDA source label.
        system: Optional system message in ``input``.
    """
    prompt: list[dict[str, str]] = []
    if system.strip():
        prompt.append({"role": "system", "content": system})
    prompt.append({"role": "user", "content": user})
    return {
        "input": prompt,
        "output": [{"role": "assistant", "content": assistant}],
    }


def iter_sft_rows(path: Path) -> Iterable[dict[str, Any]]:
    """Yield valid JSON objects from a jsonl file, skipping bad lines.

    Args:
        path: Path to ``sft.jsonl``.
    """
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def _row_task(row: dict[str, Any]) -> str:
    """Return ``knowledge`` when metadata says so, else ``kernel``."""
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        task = str(metadata.get("task") or "").strip().lower()
        if task:
            return task
        language = str(metadata.get("language") or "").strip().lower()
        if language == "prose":
            return "knowledge"
    return "kernel"


def resolve_export_assistant(row: dict[str, Any], *, cot_in_assistant: bool) -> str:
    """Build the training assistant label, optionally wrapping archived CoT.

    Knowledge rows keep prose (do not run the CUDA extractor). Kernel rows
    strip to source when ``cot_in_assistant`` is false.
    """
    pair = split_user_assistant(row)
    assistant = pair[1] if pair else ""
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    tagged = extract_thinking(assistant)
    cot = tagged
    if not cot.strip() and isinstance(metadata, dict):
        info = metadata.get("cot")
        if isinstance(info, dict):
            cot = str(info.get("text") or "")
        if not cot.strip():
            cot = str(metadata.get("cot_text") or "")

    if _row_task(row) == "knowledge":
        body = strip_thinking(assistant)
        if not body.strip():
            body = str(row.get("answer") or "")
        if cot_in_assistant and cot.strip():
            return wrap_cot_assistant(cot, body)
        return (body or "").strip() + ("\n" if (body or "").strip() else "")

    if not assistant:
        assistant = str(row.get("code") or "")
    code = extract_cuda_source(assistant)
    if not (code or "").strip():
        code = str(row.get("code") or "")
    if cot_in_assistant and cot.strip():
        return wrap_cot_assistant(cot, code or assistant)
    return (code or assistant).strip() + ("\n" if (code or assistant).strip() else "")


def export_training_files(
    src: Path,
    data_dir: Path,
    *,
    cot_in_assistant: bool | None = None,
) -> tuple[int, Path, Path]:
    """Atomically rebuild framework exports from canonical ``sft.jsonl``.

    The archive is the source of truth. Both derived files are written to
    temporary files, flushed, and replaced only after the full archive was
    converted. A crash can therefore leave an older complete export, never a
    partially rewritten JSONL file; the next store commit rebuilds it again.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    swift_path = data_dir / "sft_ms_swift.jsonl"
    openrlhf_path = data_dir / "sft_openrlhf.jsonl"
    if cot_in_assistant is None:
        cot_in_assistant = bool(get_settings().cot_in_assistant)
    count = 0
    swift_tmp = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=data_dir, prefix=".sft_ms_swift.",
        suffix=".tmp", delete=False,
    )
    orl_tmp = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=data_dir, prefix=".sft_openrlhf.",
        suffix=".tmp", delete=False,
    )
    try:
        with swift_tmp as swift_f, orl_tmp as orl_f:
            for row in iter_sft_rows(src):
                pair = split_user_assistant(row)
                if pair is None:
                    continue
                user, _assistant = pair
                assistant = resolve_export_assistant(
                    row, cot_in_assistant=bool(cot_in_assistant)
                )
                if not assistant.strip():
                    continue
                system = extract_system(row)
                swift_f.write(
                    json.dumps(to_ms_swift(user, assistant, system=system), ensure_ascii=False)
                    + "\n"
                )
                orl_f.write(
                    json.dumps(to_openrlhf(user, assistant, system=system), ensure_ascii=False)
                    + "\n"
                )
                count += 1
            swift_f.flush()
            orl_f.flush()
            os.fsync(swift_f.fileno())
            os.fsync(orl_f.fileno())
        os.replace(swift_tmp.name, swift_path)
        os.replace(orl_tmp.name, openrlhf_path)
    finally:
        for temporary in (Path(swift_tmp.name), Path(orl_tmp.name)):
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
    return count, swift_path, openrlhf_path
