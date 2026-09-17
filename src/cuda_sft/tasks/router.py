"""Expand classified questions into kernel and/or knowledge Jobs."""

from __future__ import annotations

import logging
from typing import Iterable

from cuda_sft.config import Settings, get_settings
from cuda_sft.tasks.classify import classify_row
from cuda_sft.tasks.kinds import ClassifiedQuestion, Job, QuestionRow

logger = logging.getLogger(__name__)


def _as_rows(questions: Iterable[QuestionRow | tuple]) -> list[QuestionRow]:
    """Accept QuestionRow or ``(id, question[, raw])`` tuples."""
    rows: list[QuestionRow] = []
    for item in questions:
        if isinstance(item, QuestionRow):
            rows.append(item)
            continue
        qid = int(item[0])
        question = str(item[1])
        raw = dict(item[2]) if len(item) >= 3 and isinstance(item[2], dict) else {}
        rows.append(QuestionRow(question_id=qid, question=question, raw=raw))
    return rows


def expand_pipeline_jobs(
    questions: Iterable[QuestionRow | tuple],
    settings: Settings | None = None,
) -> list[Job]:
    """Classify then expand. Knowledge jobs are never multiplied by dialect.

    Default ``TASK_MODE=kernel`` sends every row through
    :class:`KernelDialectAgent`, matching the pre-split CLI.
    """
    cfg = settings or get_settings()
    rows = _as_rows(questions)
    classified = [classify_row(row, cfg.task_mode) for row in rows]
    kernel_items = [item for item in classified if item.kind == "kernel"]
    knowledge_items = [item for item in classified if item.kind == "knowledge"]

    jobs: list[Job] = []
    if kernel_items:
        jobs.extend(_expand_kernel(kernel_items, cfg))
    if knowledge_items:
        if cfg.kernel_mode == "all" and cfg.task_mode != "knowledge":
            logger.warning(
                "KERNEL_MODE=all does not expand knowledge questions across dialects"
            )
        from cuda_sft.knowledge.agent import get_knowledge_agent

        jobs.extend(get_knowledge_agent().expand_jobs(knowledge_items, cfg))
    return jobs


def _expand_kernel(items: list[ClassifiedQuestion], settings: Settings) -> list[Job]:
    """Dialect-expand kernel questions via the existing agent."""
    from cuda_sft.dialects.agent import get_dialect_agent

    extras_by_id = {item.question_id: dict(item.raw) for item in items}
    source_by_id = {item.question_id: item.source for item in items}
    triples = get_dialect_agent().expand_jobs(
        [(item.question_id, item.question) for item in items],
        settings,
    )
    jobs: list[Job] = []
    for question_id, question, dialect in triples:
        jobs.append(
            Job(
                question_id=int(question_id),
                question=str(question),
                kind="kernel",
                track=str(dialect),
                source=source_by_id.get(int(question_id), "cli"),
                extras=extras_by_id.get(int(question_id), {}),
            )
        )
    return jobs
