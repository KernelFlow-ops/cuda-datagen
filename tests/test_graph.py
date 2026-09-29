"""Kernel and knowledge graph routing, plus shared generate-node helpers."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from cuda_sft.agents.contracts import GenerateResult
from cuda_sft.agents.critic import should_run_critic
from cuda_sft.agents.generate import assistant_state_update, complete_chat
from cuda_sft.config import Settings
from cuda_sft.core.types import build_snapshot
from cuda_sft.graph import (
    prepare,
    route_after_collect,
    route_after_compile,
    route_after_repair,
    route_after_validate,
    select_best,
)
from cuda_sft.judge import JudgeResult
from cuda_sft.knowledge.graph import route_after_gate, route_after_judge
from cuda_sft.llm import LLMCompletion
from cuda_sft.pipeline.common import graph_recursion_limit, print_stream_enabled, set_print_stream


def _settings(**kwargs: object) -> Settings:
    defaults = {
        "max_candidates": 3,
        "max_repairs": 3,
        "knowledge_max_candidates": 2,
        "knowledge_max_repairs": 2,
        "async_llm_enabled": False,
        "cot_enabled": True,
        "judge_enabled": True,
    }
    defaults.update(kwargs)
    return Settings(**defaults)


class KernelRouteTests(unittest.TestCase):
    def test_candidate_pool_continues_until_cap(self) -> None:
        with patch("cuda_sft.graph.get_settings", return_value=_settings()):
            self.assertEqual(
                route_after_collect({"candidate_idx": 1, "candidate_cap": 3}),
                "next_candidate",
            )
            self.assertEqual(
                route_after_collect({"candidate_idx": 3, "candidate_cap": 3}),
                "select_best",
            )

    def test_numeric_pass_stops_candidate_pool(self) -> None:
        snap = build_snapshot(
            {
                "candidate_idx": 1,
                "code": "validated",
                "compile_ok": True,
                "refval_status": "pass",
                "metadata": {"refval": {"status": "pass", "cases_run": 3}},
            },
            banked_reason="final",
        )
        with patch("cuda_sft.graph.get_settings", return_value=_settings(kernel_fast_mode=True)):
            self.assertEqual(
                route_after_collect({"candidate_idx": 1, "candidate_cap": 3, "candidate_reports": [snap]}),
                "select_best",
            )

    def test_fast_prepare_limits_attempts(self) -> None:
        with patch("cuda_sft.graph.get_settings", return_value=_settings(kernel_fast_mode=True)):
            state = prepare({"question_id": 8, "question": "vector add", "dialect": "cuda"})
        self.assertEqual(state["candidate_cap"], 1)
        self.assertEqual(state["repair_cap"], 1)
        self.assertFalse(state["use_critic"])
        self.assertEqual(state["quality_status"]["contract"], "skip")

    def test_selector_prefers_numeric_pass_and_preserves_code(self) -> None:
        def snap(candidate: int, code: str, status: str, score: int):
            return build_snapshot(
                {
                    "candidate_idx": candidate,
                    "code": code,
                    "compile_ok": True,
                    "refval_status": status,
                    "candidate_ctx": {"gen_system": "system", "gen_user": "user"},
                    "metadata": {
                        "refval": {"status": status, "cases_run": 12 if status == "pass" else 0},
                        "critic": {"status": "verified", "passed": True},
                        "judge": {"quality_score": score},
                    },
                },
                banked_reason="final",
            )

        with patch("cuda_sft.graph.get_settings", return_value=_settings(refval_strict=True)):
            result = select_best(
                {
                    "candidate_reports": [
                        snap(1, "bad-but-compiled", "skip", 10),
                        snap(2, "validated-source", "pass", 8),
                    ],
                    "quality_status": {},
                    "metadata": {},
                }
            )
        self.assertTrue(result["winner_found"])
        self.assertEqual(result["candidate_idx"], 2)
        self.assertEqual(result["code"], "validated-source")

    def test_compile_ok_goes_to_validate(self) -> None:
        with patch("cuda_sft.graph.get_settings", return_value=_settings()):
            self.assertEqual(
                route_after_compile({"compile_ok": True, "repair_idx": 0, "candidate_idx": 1}),
                "validate",
            )

    def test_validate_ok_goes_to_judge(self) -> None:
        with patch("cuda_sft.graph.get_settings", return_value=_settings()):
            self.assertEqual(
                route_after_validate({"refval_ok": True, "repair_idx": 0, "candidate_idx": 1}),
                "judge",
            )

    def test_validate_fail_repairs_then_collects_candidate(self) -> None:
        with patch("cuda_sft.graph.get_settings", return_value=_settings(refval_enabled=True)):
            fail = {"refval_ok": False, "compile_ok": True}
            self.assertEqual(
                route_after_validate({**fail, "repair_idx": 0, "candidate_idx": 1}),
                "repair",
            )
            for candidate in (1, 3):
                self.assertEqual(
                    route_after_validate({**fail, "repair_idx": 3, "candidate_idx": candidate}),
                    "collect_candidate",
                )

    def test_compile_fail_repairs_then_next_then_abandon(self) -> None:
        with patch("cuda_sft.graph.get_settings", return_value=_settings()):
            fail = {"compile_ok": False}
            self.assertEqual(
                route_after_compile({**fail, "repair_idx": 0, "candidate_idx": 1}),
                "repair",
            )
            self.assertEqual(
                route_after_compile({**fail, "repair_idx": 3, "candidate_idx": 1}),
                "collect_candidate",
            )
            self.assertEqual(
                route_after_compile({**fail, "repair_idx": 3, "candidate_idx": 3}),
                "collect_candidate",
            )

    def test_skip_repair_reserved_for_parallel_candidates(self) -> None:
        self.assertEqual(route_after_repair({"skip_repair": True}), "next_candidate")
        self.assertEqual(route_after_repair({"skip_repair": False}), "generate")


class CriticRouteTests(unittest.TestCase):
    def test_numeric_mismatch_forces_critic(self) -> None:
        heuristic = JudgeResult(quality_score=9, issues=[], suggestions=[])
        self.assertTrue(
            should_run_critic(
                _settings(refval_enabled=True),
                heuristic,
                use_critic=True,
                refval_status="fail",
                refval_error_class="numeric_mismatch",
            )
        )

    def test_high_score_skips_critic_when_refval_passes(self) -> None:
        heuristic = JudgeResult(quality_score=9, issues=[], suggestions=[])
        self.assertFalse(
            should_run_critic(
                _settings(refval_enabled=True),
                heuristic,
                use_critic=True,
                refval_status="pass",
            )
        )


class KnowledgeRouteTests(unittest.TestCase):
    def test_gate_ok_goes_to_judge(self) -> None:
        with patch("cuda_sft.knowledge.graph.get_settings", return_value=_settings()):
            self.assertEqual(
                route_after_gate({"gate_ok": True, "repair_idx": 0, "candidate_idx": 1}),
                "judge",
            )

    def test_judge_unavailable_abandons(self) -> None:
        with patch("cuda_sft.knowledge.graph.get_settings", return_value=_settings()):
            self.assertEqual(
                route_after_judge({"judge_unavailable": True, "judge_pass": False}),
                "save_abandoned",
            )


class RecursionAndStreamTests(unittest.TestCase):
    def test_recursion_formula_matches_legacy_kernel(self) -> None:
        # max_repairs=3, extra=5 → per_candidate = 4*3 + 3 + 5 = 20; 10+3*20=70 → clamp 80
        self.assertEqual(
            graph_recursion_limit(max_candidates=3, max_repairs=3, extra_per_candidate=5),
            80,
        )
        self.assertGreater(
            graph_recursion_limit(max_candidates=6, max_repairs=3, extra_per_candidate=5),
            80,
        )

    def test_print_stream_toggle_is_shared(self) -> None:
        previous = print_stream_enabled()
        try:
            set_print_stream(False)
            self.assertFalse(print_stream_enabled())
            set_print_stream(True)
            self.assertTrue(print_stream_enabled())
        finally:
            set_print_stream(previous)


class GenerateHelperTests(unittest.TestCase):
    def test_assistant_state_update_appends_empty_placeholder(self) -> None:
        state = {"messages": [{"role": "user", "content": "q"}]}
        update = assistant_state_update(
            state,
            GenerateResult(text="  ", reasoning="", reasoning_source="empty"),
        )
        self.assertEqual(update["raw_response"], "  ")
        self.assertEqual(update["messages"][-1]["content"], "(empty response)")

    def test_complete_chat_uses_injected_client(self) -> None:
        class _Client:
            def stream_completion(self, **kwargs: object) -> LLMCompletion:
                return LLMCompletion(
                    text="hello",
                    reasoning="think",
                    reasoning_source="api",
                )

        with patch(
            "cuda_sft.agents.generate.get_settings",
            return_value=_settings(async_llm_enabled=False, cot_enabled=True),
        ):
            result = complete_chat(
                messages=[{"role": "user", "content": "q"}],
                system="sys",
                temperature=0.2,
                client=_Client(),  # type: ignore[arg-type]
            )
        self.assertEqual(result.text, "hello")
        self.assertEqual(result.reasoning, "think")
        self.assertFalse(result.used_speculative)


if __name__ == "__main__":
    unittest.main()
