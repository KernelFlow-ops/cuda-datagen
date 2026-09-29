"""Difficulty topology, adaptive critic trigger, and knowledge score cap."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from cuda_sft.agents.critic import should_run_critic
from cuda_sft.agents.difficulty import kernel_difficulty, plan_topology
from cuda_sft.config import Settings
from cuda_sft.graph import route_after_critic
from cuda_sft.judge import JudgeResult
from cuda_sft.knowledge.judge import _dimensions_from_payload
from cuda_sft.knowledge.rubrics import overall_score


def _settings(**kwargs: object) -> Settings:
    defaults = {
        "max_candidates": 3,
        "max_repairs": 3,
        "knowledge_max_candidates": 2,
        "difficulty_aware": True,
        "kernel_llm_critic": "adaptive",
        "kernel_critic_blocks_save": False,
        "knowledge_judge_mode": "capped",
        "async_llm_enabled": False,
        "judge_enabled": True,
    }
    defaults.update(kwargs)
    return Settings(**defaults)


class DifficultyTests(unittest.TestCase):
    def test_elementwise_is_simple(self) -> None:
        self.assertEqual(kernel_difficulty("Implement vector add"), "simple")

    def test_gemm_is_hard(self) -> None:
        self.assertEqual(kernel_difficulty("Write a tiled GEMM kernel"), "hard")

    def test_chinese_kernel_questions_use_operation_difficulty(self) -> None:
        self.assertEqual(kernel_difficulty("实现向量加法，每个线程处理一个元素"), "simple")
        self.assertEqual(kernel_difficulty("实现分块矩阵乘法并使用共享内存"), "hard")
        self.assertEqual(kernel_difficulty("实现每行前缀和扫描"), "hard")

    def test_simple_skips_critic_and_uses_small_candidate_budget(self) -> None:
        plan = plan_topology(
            question="elementwise relu",
            kind="kernel",
            settings=_settings(),
        )
        self.assertEqual(plan.difficulty, "simple")
        self.assertEqual(plan.max_candidates, 2)
        self.assertFalse(plan.use_critic)

    def test_flag_off_keeps_full_budget(self) -> None:
        plan = plan_topology(
            question="elementwise relu",
            kind="kernel",
            settings=_settings(difficulty_aware=False),
        )
        self.assertEqual(plan.max_candidates, 3)
        self.assertTrue(plan.use_critic)


class CriticTriggerTests(unittest.TestCase):
    def test_high_score_without_issues_skips(self) -> None:
        heuristic = JudgeResult(quality_score=9, issues=[], suggestions=[])
        self.assertFalse(should_run_critic(_settings(), heuristic, use_critic=True))

    def test_low_score_runs(self) -> None:
        heuristic = JudgeResult(quality_score=6, issues=[], suggestions=[])
        self.assertTrue(should_run_critic(_settings(), heuristic, use_critic=True))

    def test_topology_can_force_skip(self) -> None:
        heuristic = JudgeResult(quality_score=4, issues=["x"], suggestions=[])
        self.assertFalse(should_run_critic(_settings(), heuristic, use_critic=False))


class CriticRouteTests(unittest.TestCase):
    def test_pass_goes_to_candidate_pool(self) -> None:
        with patch("cuda_sft.graph.get_settings", return_value=_settings()):
            self.assertEqual(
                route_after_critic({"critic_pass": True, "critic_must_fix": [], "repair_idx": 0}),
                "collect_candidate",
            )

    def test_failed_critic_banks_before_repair_when_budget_left(self) -> None:
        with patch("cuda_sft.graph.get_settings", return_value=_settings()):
            self.assertEqual(
                route_after_critic(
                    {
                        "critic_pass": False,
                        "critic_must_fix": ["wrong host signature"],
                        "repair_idx": 0,
                        "metadata": {"critic": {"status": "failed"}},
                    }
                ),
                "bank_pre_repair",
            )


class KnowledgeCapTests(unittest.TestCase):
    def test_missing_dimension_defaults_to_one(self) -> None:
        dims = _dimensions_from_payload(
            {
                "dimensions": {
                    "factual": 9,
                    "completeness": 8,
                    "terminology": 8,
                    "structure": 8,
                    "grounding": 8,
                }
            }
        )
        assert dims is not None
        self.assertEqual(dims["derivation"], 1.0)

    def test_overall_cannot_exceed_factual(self) -> None:
        dims = {
            "factual": 4,
            "completeness": 10,
            "derivation": 10,
            "terminology": 10,
            "structure": 10,
            "grounding": 10,
        }
        raw = overall_score(dims, "architecture")
        self.assertGreater(raw, 4)
        capped = min(raw, dims["factual"], dims["completeness"])
        self.assertEqual(capped, 4.0)
