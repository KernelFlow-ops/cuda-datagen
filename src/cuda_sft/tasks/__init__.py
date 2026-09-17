"""Task-kind routing (kernel vs knowledge). No LLM calls here."""

from cuda_sft.tasks.classify import classify_row, infer_topic
from cuda_sft.tasks.kinds import Job, QuestionRow, coerce_job, knowledge_track, progress_key
from cuda_sft.tasks.router import expand_pipeline_jobs

__all__ = [
    "Job",
    "QuestionRow",
    "classify_row",
    "coerce_job",
    "expand_pipeline_jobs",
    "infer_topic",
    "knowledge_track",
    "progress_key",
]
