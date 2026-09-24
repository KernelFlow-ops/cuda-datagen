"""Validate-node routing and critic force-on-numeric."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from cuda_sft.agents.critic import should_run_critic
from cuda_sft.config import Settings
from cuda_sft.graph import route_after_compile, route_after_validate
from cuda_sft.judge import JudgeResult


def _settings(**kwargs: object) -> Settings:
    defaults = {
        "max_candidates": 3,
        "max_repairs": 3,
        "async_llm_enabled": False,
        "judge_enabled": True,
        "refval_enabled": True,
    }
    defaults.update(kwargs)
    return Settings(**defaults)


class ValidateRouteTests(unittest.TestCase):
    def test_compile_ok_goes_to_validate(self) -> None:
        with patch("cuda_sft.graph.get_settings", return_value=_settings()):
            self.assertEqual(
                route_after_compile({"compile_ok": True, "repair_idx": 0, "candidate_idx": 1}),
                "validate",
            )

    def test_validate_pass_goes_to_judge(self) -> None:
        with patch("cuda_sft.graph.get_settings", return_value=_settings()):
            self.assertEqual(
                route_after_validate({"refval_ok": True, "repair_idx": 0, "candidate_idx": 1}),
                "judge",
            )

    def test_validate_fail_repairs_then_next_then_abandon(self) -> None:
        with patch("cuda_sft.graph.get_settings", return_value=_settings()):
            fail = {"refval_ok": False, "compile_ok": True}
            self.assertEqual(
                route_after_validate({**fail, "repair_idx": 0, "candidate_idx": 1}),
                "repair",
            )
            self.assertEqual(
                route_after_validate({**fail, "repair_idx": 3, "candidate_idx": 1}),
                "next_candidate",
            )
            self.assertEqual(
                route_after_validate({**fail, "repair_idx": 3, "candidate_idx": 3}),
                "save_abandoned",
            )


class CriticForceTests(unittest.TestCase):
    def test_numeric_mismatch_forces_critic(self) -> None:
        heuristic = JudgeResult(quality_score=9, issues=[], suggestions=[])
        self.assertTrue(
            should_run_critic(
                _settings(),
                heuristic,
                use_critic=True,
                refval_status="fail",
                refval_error_class="numeric_mismatch",
            )
        )

    def test_high_score_still_skips_when_refval_pass(self) -> None:
        heuristic = JudgeResult(quality_score=9, issues=[], suggestions=[])
        self.assertFalse(
            should_run_critic(
                _settings(),
                heuristic,
                use_critic=True,
                refval_status="pass",
            )
        )


if __name__ == "__main__":
    unittest.main()
