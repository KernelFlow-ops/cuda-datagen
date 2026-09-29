"""Tests for CoT Agent fallbacks and output sanitization."""

from __future__ import annotations

import unittest

from cuda_sft.config import Settings
from cuda_sft.cot import CotAgent, sanitize_cot_output
from cuda_sft.llm import LLMCompletion
from cuda_sft.parse import has_numbered_headings


class _FakeClient:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    def stream_completion(self, **_kwargs: object) -> LLMCompletion:
        self.calls += 1
        return LLMCompletion(text=self.text, reasoning="", reasoning_source="empty")


def _settings(**kwargs: object) -> Settings:
    defaults = {
        "cot_enabled": True,
        "cot_agent_enabled": True,
        "cot_in_assistant": True,
        "cot_max_chars": 8000,
        "cot_raw_max_chars": 24000,
        "cot_on_empty": "empty",
        "cot_on_agent_fail": "raw",
        "judge_enabled": False,
        "async_llm_enabled": False,
    }
    defaults.update(kwargs)
    return Settings(**defaults)


def _state(question: str, code: str, raw_reasoning: str) -> dict:
    return {
        "question": question,
        "code": code,
        "raw_reasoning": raw_reasoning,
        "cot_mode": "polish",
        "selected": {"code": code, "candidate": 1, "repairs": 0, "judge": {}},
    }


class SanitizeTests(unittest.TestCase):
    def test_strips_think_wrapper_and_cuda_fence(self) -> None:
        text = (
            "<think>\n"
            "1. Problem restatement\nUse 1D indexing.\n"
            "```cuda\n__global__ void k() {}\n```\n"
            "</think>"
        )
        out = sanitize_cot_output(text, 8000)
        self.assertIn("1D indexing", out)
        self.assertNotIn("__global__", out)
        self.assertNotIn("<think>", out)


class CotAgentTests(unittest.TestCase):
    def test_agent_success(self) -> None:
        body = (
            "1. Problem restatement\nAdd two vectors.\n"
            "2. Algorithm\nElementwise add.\n"
            "3. Thread/block mapping\nOne thread per element.\n"
            "4. Memory and sync\nGlobal loads only.\n"
            "5. Bounds and edge cases\nGuard i < n.\n"
            "6. Implementation checklist\n- kernel add_kernel\n- host launch\n"
        )
        client = _FakeClient(body)
        agent = CotAgent(settings=_settings(), llm_client=client)
        result = agent.refine(
            _state(
                "vector add",
                "__global__ void add_kernel() {}\n",
                "I will write a kernel then launch it.",
            )
        )
        self.assertEqual(result.source, "agent")
        self.assertIn("Elementwise", result.cot)
        self.assertEqual(client.calls, 1)
        self.assertTrue(has_numbered_headings(result.cot, 6))

    def test_truncated_headings_rejects_unstructured_raw_fallback(self) -> None:
        class _TruncThenRaw:
            def __init__(self) -> None:
                self.calls = 0

            def stream_completion(self, **_kwargs: object) -> LLMCompletion:
                self.calls += 1
                return LLMCompletion(
                    text="1. Problem restatement\nToo short and no more headings.",
                    reasoning="",
                    reasoning_source="empty",
                )

        agent = CotAgent(
            settings=_settings(cot_on_agent_fail="raw"),
            llm_client=_TruncThenRaw(),  # type: ignore[arg-type]
        )
        result = agent.refine(
            _state(
                "vector add",
                "__global__ void add_kernel() {}\n",
                "Use one thread per element and guard i < n.",
            )
        )
        self.assertEqual(result.source, "empty")
        self.assertIn("raw reasoning unstructured", result.error)
        self.assertGreaterEqual(result.raw_reasoning.count("thread"), 1)

    def test_agent_disabled_uses_raw(self) -> None:
        agent = CotAgent(
            settings=_settings(cot_agent_enabled=False),
            llm_client=_FakeClient("unused"),
        )
        result = agent.refine(
            _state("q", "__global__ void k() {}\n", "raw teacher thinking that is long enough")
        )
        self.assertEqual(result.source, "raw")
        self.assertIn("raw teacher thinking", result.cot)

    def test_code_only_agent_output_rejects_unstructured_raw(self) -> None:
        client = _FakeClient("```cuda\n__global__ void k() { int i = threadIdx.x; }\n```")
        agent = CotAgent(settings=_settings(cot_on_agent_fail="raw"), llm_client=client)
        result = agent.refine(
            _state(
                "q", "__global__ void k() {}\n", "keep this raw chain of thought as fallback text"
            )
        )
        self.assertEqual(result.source, "empty")
        self.assertIn("keep this raw", result.raw_reasoning)
        self.assertIn("raw reasoning unstructured", result.error)
        self.assertTrue(result.error)

    def test_empty_thinking_without_agent_is_empty(self) -> None:
        agent = CotAgent(
            settings=_settings(cot_agent_enabled=False, cot_on_empty="empty"),
            llm_client=_FakeClient("should not run"),
        )
        result = agent.refine(_state("q", "int x;\n", ""))
        self.assertEqual(result.source, "empty")
        self.assertEqual(result.cot, "")


if __name__ == "__main__":
    unittest.main()
