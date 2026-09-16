"""Tests for reasoning assembly from NVIDIA / OpenRouter-shaped payloads."""

from __future__ import annotations

import unittest

from cuda_sft.llm import (
    assemble_completion,
    reasoning_from_openrouter_message,
    _thinking_from_content_blocks,
)


class _Delta:
    def __init__(self, **kwargs: object) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


class AssembleCompletionTests(unittest.TestCase):
    def test_nvidia_reasoning_content_preferred(self) -> None:
        completion = assemble_completion(
            visible_text="```cuda\n__global__ void k() {}\n```",
            reasoning_candidates=[("thread mapping first", "nvidia_delta")],
        )
        self.assertEqual(completion.reasoning, "thread mapping first")
        self.assertEqual(completion.reasoning_source, "nvidia_delta")
        self.assertIn("__global__", completion.text)

    def test_think_tag_fallback_strips_visible(self) -> None:
        visible = "<think>hidden plan</think>\n#include <cuda_runtime.h>\n"
        completion = assemble_completion(visible_text=visible, reasoning_candidates=[])
        self.assertEqual(completion.reasoning, "hidden plan")
        self.assertEqual(completion.reasoning_source, "think_tags")
        self.assertNotIn("<think>", completion.text)
        self.assertIn("cuda_runtime", completion.text)

    def test_empty_reasoning(self) -> None:
        completion = assemble_completion(
            visible_text="hello",
            reasoning_candidates=[("", "nvidia_delta")],
        )
        self.assertEqual(completion.reasoning_source, "empty")
        self.assertEqual(completion.text, "hello")

    def test_anthropic_blocks_before_openrouter_field(self) -> None:
        completion = assemble_completion(
            visible_text="ok",
            reasoning_candidates=[
                ("from thinking block", "anthropic_thinking"),
                ("from reasoning field", "openrouter_reasoning"),
            ],
        )
        self.assertEqual(completion.reasoning_source, "anthropic_thinking")
        self.assertEqual(completion.reasoning, "from thinking block")


class OpenRouterMessageTests(unittest.TestCase):
    def test_reasoning_details_array(self) -> None:
        msg = _Delta(
            reasoning="",
            reasoning_details=[
                {"type": "reasoning.text", "text": "Let "},
                {"type": "reasoning.text", "text": "me map threads."},
            ],
        )
        self.assertEqual(reasoning_from_openrouter_message(msg), "Let me map threads.")

    def test_thinking_content_blocks(self) -> None:
        blocks = [
            {"type": "thinking", "thinking": "bounds first"},
            {"type": "text", "text": "code"},
        ]
        self.assertEqual(_thinking_from_content_blocks(blocks), "bounds first")


if __name__ == "__main__":
    unittest.main()
