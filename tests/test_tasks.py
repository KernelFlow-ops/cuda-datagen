"""Task classification and job expansion (kernel path must stay default)."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from cuda_sft.tasks.classify import classify_row, heuristic_kind, infer_topic
from cuda_sft.tasks.kinds import Job, QuestionRow, coerce_job, knowledge_track, progress_key
from cuda_sft.tasks.router import expand_pipeline_jobs


def _settings(**kwargs: object):
    class _S:
        task_mode = "kernel"
        kernel_mode = "single"
        kernel_dialects = "cuda"
        kernel_dialect = ""

    for key, value in kwargs.items():
        setattr(_S, key, value)
    return _S()


class ClassifyTests(unittest.TestCase):
    def test_kernel_mode_ignores_knowledge_field(self) -> None:
        row = QuestionRow(
            1,
            "Explain occupancy.",
            {"question": "Explain occupancy.", "task": "knowledge"},
        )
        classified = classify_row(row, "kernel")
        self.assertEqual(classified.kind, "kernel")
        self.assertEqual(classified.source, "cli")

    def test_knowledge_mode_forces_knowledge(self) -> None:
        row = QuestionRow(1, "Implement a kernel in solution.cu", {"task": "kernel"})
        classified = classify_row(row, "knowledge")
        self.assertEqual(classified.kind, "knowledge")

    def test_auto_uses_field(self) -> None:
        row = QuestionRow(1, "anything", {"task": "knowledge", "topic": "formula"})
        classified = classify_row(row, "auto")
        self.assertEqual(classified.kind, "knowledge")
        self.assertEqual(classified.source, "field")
        self.assertEqual(classified.topic, "formula")

    def test_auto_conflict_prefers_kernel(self) -> None:
        text = (
            "Explain occupancy then implement a fused kernel.\n"
            "Write your implementation to solution.cu.\n"
            "Your implementation should use shared memory."
        )
        self.assertEqual(heuristic_kind(text), "kernel")
        classified = classify_row(QuestionRow(1, text, {}), "auto")
        self.assertEqual(classified.kind, "kernel")

    def test_auto_knowledge_without_kernel_signals(self) -> None:
        text = "Explain the CUDA memory hierarchy and why coalescing matters."
        classified = classify_row(QuestionRow(2, text, {}), "auto")
        self.assertEqual(classified.kind, "knowledge")
        self.assertEqual(classified.source, "heuristic")
        self.assertEqual(classified.topic, "memory")

    def test_unlabeled_implement_is_kernel(self) -> None:
        text = "Implement a vector add kernel.\nWrite to solution.cu."
        classified = classify_row(QuestionRow(3, text, {}), "auto")
        self.assertEqual(classified.kind, "kernel")

    def test_infer_formula_topic(self) -> None:
        self.assertEqual(
            infer_topic("Derive the occupancy formula for a kernel."),
            "formula",
        )

    def test_infer_cute_topic_from_field(self) -> None:
        self.assertEqual(
            infer_topic("What is a layout?", {"topic": "cute"}),
            "cute",
        )

    def test_infer_user_knowledge50_style(self) -> None:
        """Topic hints for the user-authored knowledge jsonl (no topic field)."""
        cases = [
            ("Analyze CUDA memory hierarchy and access efficiency.", "memory"),
            ("Explain CUDA thread organization and indexing.", "architecture"),
            ("Explain warp execution and branch divergence in CUDA.", "execution"),
            ("Analyze CUDA occupancy and GPU resource utilization.", "execution"),
            ("Compare CUDA streams and asynchronous execution.", "api"),
            ("Explain shared memory bank conflicts in CUDA.", "memory"),
            ("Explain the roofline performance model for CUDA workloads.", "formula"),
            ("Explain texture or read-only caching concepts in GPU workloads.", "memory"),
        ]
        for text, topic in cases:
            self.assertEqual(infer_topic(text), topic, msg=text)


class JobTests(unittest.TestCase):
    def test_coerce_triple_is_kernel(self) -> None:
        job = coerce_job((9, "q", "triton"))
        self.assertEqual(job.kind, "kernel")
        self.assertEqual(job.track, "triton")
        self.assertEqual(progress_key(job), (9, "triton"))

    def test_knowledge_track(self) -> None:
        self.assertEqual(knowledge_track("formula"), "knowledge:formula")
        job = Job(1, "q", "knowledge", "knowledge:formula")
        self.assertEqual(job.topic, "formula")
        self.assertEqual(job.dialect, "knowledge:formula")


class ExpandTests(unittest.TestCase):
    def test_default_kernel_mode_matches_dialect_agent(self) -> None:
        from cuda_sft.dialects.agent import KernelDialectAgent

        agent = KernelDialectAgent()
        rows = [QuestionRow(1, "q1", {}), QuestionRow(2, "q2", {"task": "knowledge"})]
        settings = _settings(task_mode="kernel", kernel_mode="single", kernel_dialects="cuda")
        with (
            patch.object(agent, "resolve", return_value=[agent.spec("cuda")]),
            patch("cuda_sft.dialects.agent.get_dialect_agent", return_value=agent),
        ):
            jobs = expand_pipeline_jobs(rows, settings=settings)  # type: ignore[arg-type]
        self.assertEqual([job.kind for job in jobs], ["kernel", "kernel"])
        self.assertEqual([job.track for job in jobs], ["cuda", "cuda"])
        self.assertTrue(all(job.kind == "kernel" for job in jobs))

    def test_knowledge_mode_does_not_multiply_dialects(self) -> None:
        rows = [
            QuestionRow(1, "Explain occupancy.", {"topic": "execution"}),
            QuestionRow(2, "Derive occupancy = ...", {"topic": "formula"}),
        ]
        settings = _settings(
            task_mode="knowledge",
            kernel_mode="all",
            kernel_dialects="cuda,triton",
        )
        jobs = expand_pipeline_jobs(rows, settings=settings)  # type: ignore[arg-type]
        self.assertEqual(len(jobs), 2)
        self.assertTrue(all(job.kind == "knowledge" for job in jobs))
        self.assertEqual(jobs[0].track, "knowledge:execution")
        self.assertEqual(jobs[1].track, "knowledge:formula")

    def test_auto_mixed_file(self) -> None:
        from cuda_sft.dialects.agent import KernelDialectAgent

        agent = KernelDialectAgent()
        rows = [
            QuestionRow(1, "Implement a kernel in solution.cu", {}),
            QuestionRow(2, "Explain CUDA memory hierarchy.", {"task": "knowledge"}),
        ]
        settings = _settings(task_mode="auto", kernel_mode="all", kernel_dialects="cuda,triton")
        with (
            patch.object(agent, "resolve", return_value=[agent.spec("cuda"), agent.spec("triton")]),
            patch("cuda_sft.dialects.agent.get_dialect_agent", return_value=agent),
        ):
            jobs = expand_pipeline_jobs(rows, settings=settings)  # type: ignore[arg-type]
        kernel = [job for job in jobs if job.kind == "kernel"]
        knowledge = [job for job in jobs if job.kind == "knowledge"]
        self.assertEqual({job.track for job in kernel}, {"cuda", "triton"})
        self.assertEqual(len(knowledge), 1)
        self.assertTrue(knowledge[0].track.startswith("knowledge:"))


if __name__ == "__main__":
    unittest.main()
