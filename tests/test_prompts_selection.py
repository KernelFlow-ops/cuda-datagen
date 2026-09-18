"""Resume-stable prompt indexing and compiler-log trimming."""

from __future__ import annotations

import unittest

from cuda_sft.prompts.nvcc_log import format_nvcc_for_prompt
from cuda_sft.prompts.selection import (
    candidate_temperature,
    looks_chinese,
    stable_index,
)
from cuda_sft.prompt import _stable_index


class StableIndexTests(unittest.TestCase):
    def test_public_matches_prompt_alias(self) -> None:
        args = (8, 42, 2, 7)
        self.assertEqual(stable_index(*args), _stable_index(*args))

    def test_resume_stable(self) -> None:
        first = [stable_index(10, 3, c, 0) for c in (1, 2, 3)]
        second = [stable_index(10, 3, c, 0) for c in (1, 2, 3)]
        self.assertEqual(first, second)

    def test_empty_pool_raises(self) -> None:
        with self.assertRaises(ValueError):
            stable_index(0, 1, 1, 0)

    def test_temperatures(self) -> None:
        self.assertEqual(candidate_temperature(1), 0.2)
        self.assertEqual(candidate_temperature(2), 0.5)
        self.assertEqual(candidate_temperature(3), 0.8)
        self.assertEqual(candidate_temperature(99), 0.8)

    def test_looks_chinese(self) -> None:
        self.assertTrue(looks_chinese("请实现一个 kernel"))
        self.assertFalse(looks_chinese("Write a CUDA kernel"))


class NvccLogTests(unittest.TestCase):
    def test_short_log_unchanged(self) -> None:
        log = 'solution.cu(3): error: identifier "foo" is undefined'
        self.assertEqual(format_nvcc_for_prompt(log, max_chars=800), log)

    def test_empty_log(self) -> None:
        self.assertEqual(format_nvcc_for_prompt(""), "(empty compiler output)")


class DialectImportTests(unittest.TestCase):
    def test_dialects_do_not_import_private_stable_index(self) -> None:
        from pathlib import Path

        root = Path(__file__).resolve().parents[1] / "src" / "cuda_sft" / "dialects"
        hits: list[str] = []
        for path in sorted(root.glob("*.py")):
            text = path.read_text(encoding="utf-8")
            if "_stable_index" in text:
                hits.append(path.name)
        self.assertEqual(hits, [])


if __name__ == "__main__":
    unittest.main()
