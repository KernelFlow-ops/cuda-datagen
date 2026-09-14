"""Convert generation archive rows to ms-swift / OpenRLHF SFT jsonl."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

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
    """Read the system prompt from ``messages`` or ``metadata``, else default.

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
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        extra = metadata.get("system")
        if isinstance(extra, str) and extra.strip():
            return extra
    return SYSTEM_PROMPT


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


def export_training_files(src: Path, data_dir: Path) -> tuple[int, Path, Path]:
    """Rewrite framework-specific jsonl from the generation archive."""
    data_dir.mkdir(parents=True, exist_ok=True)
    swift_path = data_dir / "sft_ms_swift.jsonl"
    openrlhf_path = data_dir / "sft_openrlhf.jsonl"
    count = 0
    with swift_path.open("w", encoding="utf-8") as swift_f, openrlhf_path.open(
        "w", encoding="utf-8"
    ) as orl_f:
        for row in iter_sft_rows(src):
            pair = split_user_assistant(row)
            if pair is None:
                continue
            user, assistant = pair
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
    return count, swift_path, openrlhf_path
