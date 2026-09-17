"""Knowledge Agent: one Job per question, never multiplied by kernel dialect."""

from __future__ import annotations

from typing import Iterable

from cuda_sft.config import Settings, get_settings
from cuda_sft.tasks.kinds import ClassifiedQuestion, Job, QuestionRow, knowledge_track
from cuda_sft.tasks.classify import classify_row, infer_topic


class KnowledgeAgent:
    """Prompt/job helper for knowledge questions."""

    def expand_jobs(
        self,
        questions: Iterable[ClassifiedQuestion | QuestionRow | tuple],
        settings: Settings | None = None,
    ) -> list[Job]:
        """One ``knowledge:{topic}`` job per question. Ignores ``KERNEL_MODE``."""
        cfg = settings or get_settings()
        jobs: list[Job] = []
        for item in questions:
            if isinstance(item, ClassifiedQuestion):
                classified = item
            elif isinstance(item, QuestionRow):
                classified = classify_row(item, "knowledge")
            else:
                qid = int(item[0])
                question = str(item[1])
                raw = dict(item[2]) if len(item) >= 3 and isinstance(item[2], dict) else {}
                classified = classify_row(
                    QuestionRow(qid, question, raw),
                    "knowledge",
                )
            topic = classified.topic or infer_topic(classified.question, classified.raw)
            jobs.append(
                Job(
                    question_id=classified.question_id,
                    question=classified.question,
                    kind="knowledge",
                    track=knowledge_track(topic),
                    source=classified.source,
                    extras=dict(classified.raw),
                )
            )
        _ = cfg  # settings reserved for future per-topic filters
        return jobs

    def llm_call_options(self, settings: Settings | None = None) -> dict:
        """Sampling overrides: smaller completion budget than kernel Super models."""
        cfg = settings or get_settings()
        return {
            "thinking_level": cfg.knowledge_thinking_level or "medium",
            "max_output_tokens": int(cfg.knowledge_max_output_tokens or 8192),
        }


_agent: KnowledgeAgent | None = None


def get_knowledge_agent() -> KnowledgeAgent:
    """Process-wide Knowledge Agent."""
    global _agent
    if _agent is None:
        _agent = KnowledgeAgent()
    return _agent
