"""Tests for export wrapping of CoT into training assistant labels."""

from __future__ import annotations

import unittest

from cuda_sft.formats import resolve_export_assistant
from cuda_sft.parse import extract_thinking, wrap_cot_assistant


class ExportAssistantTests(unittest.TestCase):
    def test_keeps_existing_think_block(self) -> None:
        code = "#include <cuda_runtime.h>\n__global__ void k() {}\n"
        assistant = wrap_cot_assistant("1. Problem restatement\nAdd tensors.", code)
        row = {
            "messages": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": assistant},
            ]
        }
        out = resolve_export_assistant(row, cot_in_assistant=True)
        self.assertTrue(out.startswith("<think>"))
        self.assertEqual(extract_thinking(out), "1. Problem restatement\nAdd tensors.")

    def test_strips_think_when_disabled(self) -> None:
        code = "#include <cuda_runtime.h>\n__global__ void k() {}\n"
        assistant = wrap_cot_assistant("hidden cot", code)
        row = {
            "messages": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": assistant},
            ]
        }
        out = resolve_export_assistant(row, cot_in_assistant=False)
        self.assertNotIn("<think>", out)
        self.assertIn("__global__", out)

    def test_wraps_from_metadata_when_assistant_is_code(self) -> None:
        code = "__global__ void k() {}\n"
        row = {
            "messages": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": code},
            ],
            "metadata": {"cot": {"text": "1. Problem restatement\nUse 1D grid."}},
        }
        out = resolve_export_assistant(row, cot_in_assistant=True)
        self.assertTrue(out.startswith("<think>"))
        self.assertIn("1D grid", extract_thinking(out))
        self.assertIn("__global__", out)


if __name__ == "__main__":
    unittest.main()
