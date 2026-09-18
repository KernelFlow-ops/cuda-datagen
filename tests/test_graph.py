"""Kernel and knowledge graph routing, plus shared generate-node helpers."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from cuda_sft.agents.contracts import GenerateResult
from cuda_sft.agents.generate import assistant_state_update, complete_chat
from cuda_sft.config import Settings
from cuda_sft.graph import route_after_compile, route_after_repair
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
    def test_compile_ok_goes_to_judge(self) -> None:
        with patch("cuda_sft.graph.get_settings", return_value=_settings()):
            self.assertEqual(
                route_after_compile({"compile_ok": True, "repair_idx": 0, "candidate_idx": 1}),
                "judge",
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
                "next_candidate",
            )
            self.assertEqual(
                route_after_compile({**fail, "repair_idx": 3, "candidate_idx": 3}),
                "save_abandoned",
            )

    def test_skip_repair_reserved_for_parallel_candidates(self) -> None:
        self.assertEqual(route_after_repair({"skip_repair": True}), "next_candidate")
        self.assertEqual(route_after_repair({"skip_repair": False}), "generate")


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

    def test_judge_pass_goes_to_cot(self) -> None:
        with patch("cuda_sft.knowledge.graph.get_settings", return_value=_settings()):
            self.assertEqual(
                route_after_judge(
                    {
                        "judge_unavailable": False,
                        "judge_pass": True,
                        "repair_idx": 0,
                        "candidate_idx": 1,
                    }
                ),
                "cot",
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
