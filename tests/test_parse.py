"""Tests for thinking extraction and CoT assistant wrapping."""

from __future__ import annotations

import unittest

from cuda_sft.parse import (
    extract_cuda_source,
    extract_thinking,
    split_visible_and_thinking,
    strip_code_from_cot,
    unwrap_cot_assistant,
    wrap_cot_assistant,
)


class ParseThinkingTests(unittest.TestCase):
    def test_extract_thinking_tags(self) -> None:
        text = (
            "<think>first</think>\n"
            "```cuda\nint x;\n```\n"
            "<reasoning>second</reasoning>"
        )
        self.assertEqual(extract_thinking(text), "first\n\nsecond")

    def test_split_visible_and_thinking(self) -> None:
        visible, thinking = split_visible_and_thinking(
            "<think>plan</think>\n#include <cuda_runtime.h>\n"
        )
        self.assertIn("cuda_runtime", visible)
        self.assertEqual(thinking, "plan")

    def test_wrap_and_extract_round_trip(self) -> None:
        code = "#include <cuda_runtime.h>\n__global__ void k() {}\n"
        cot = "1. Problem restatement\nUse a 1D grid."
        assistant = wrap_cot_assistant(cot, code)
        self.assertTrue(assistant.startswith("<think>"))
        got_cot, got_code = unwrap_cot_assistant(assistant)
        self.assertEqual(got_cot, cot)
        self.assertEqual(got_code, code if code.endswith("\n") else code + "\n")
        self.assertEqual(extract_cuda_source(assistant), got_code)

    def test_wrap_empty_cot_is_code_only(self) -> None:
        code = "__global__ void k() {}\n"
        self.assertEqual(wrap_cot_assistant("  ", code), code)

    def test_strip_code_from_cot_drops_cuda_fence(self) -> None:
        text = (
            "Use a 1D mapping.\n"
            "```cuda\n__global__ void k() { int i = threadIdx.x; }\n```\n"
            "Then launch."
        )
        cleaned = strip_code_from_cot(text)
        self.assertNotIn("__global__", cleaned)
        self.assertIn("1D mapping", cleaned)
        self.assertIn("Then launch", cleaned)


if __name__ == "__main__":
    unittest.main()
