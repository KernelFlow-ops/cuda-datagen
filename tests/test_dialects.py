"""Tests for kernel dialect names, job expansion, and extractors."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from cuda_sft.dialects.agent import KernelDialectAgent, parse_dialect_list
from cuda_sft.dialects.base import normalize_dialect_name
from cuda_sft.dialects.cutlass import looks_like_cutlass, read_cutlass_major
from cuda_sft.dialects.triton import looks_like_triton
from cuda_sft.parse import extract_fenced_source
from cuda_sft.store import load_done_keys


class DialectNameTests(unittest.TestCase):
    def test_aliases(self) -> None:
        self.assertEqual(normalize_dialect_name("cute"), "cutlass")
        self.assertEqual(normalize_dialect_name("cutlass/cute"), "cutlass")
        self.assertEqual(normalize_dialect_name("cutlass4.x"), "cutlass")
        self.assertEqual(normalize_dialect_name("CUDA"), "cuda")

    def test_unknown(self) -> None:
        with self.assertRaises(ValueError):
            normalize_dialect_name("fortran")

    def test_parse_list(self) -> None:
        self.assertEqual(
            parse_dialect_list("cuda, cute, triton, cuda"),
            ["cuda", "cutlass", "triton"],
        )


class ExtractTests(unittest.TestCase):
    def test_triton_fence(self) -> None:
        text = (
            "```python\n"
            "import triton\n"
            "import triton.language as tl\n"
            "@triton.jit\n"
            "def k(x_ptr):\n"
            "    pid = tl.program_id(0)\n"
            "```\n"
        )
        src = extract_fenced_source(
            text, fence_langs=("python", "triton"), looks_like=looks_like_triton
        )
        self.assertIn("@triton.jit", src)

    def test_cutlass_looks_like(self) -> None:
        self.assertTrue(looks_like_cutlass("cute::Tensor t; __global__ void k() {}"))


class RefvalSpecTests(unittest.TestCase):
    def test_every_dialect_exposes_refval_spec(self) -> None:
        from cuda_sft.config import Settings
        from cuda_sft.dialects.agent import KernelDialectAgent

        settings = Settings(async_llm_enabled=False)
        agent = KernelDialectAgent()
        for name in ("cuda", "cutlass", "triton", "tilelang"):
            spec = agent.spec(name).refval_spec(settings)
            self.assertEqual(spec.dialect, name)
            self.assertIn(spec.runner, {"nvcc_link", "python_import"})


class AgentExpandTests(unittest.TestCase):
    def test_all_mode_expands(self) -> None:
        agent = KernelDialectAgent()

        class _S:
            kernel_mode = "all"
            kernel_dialects = "cuda,triton"
            kernel_dialect = ""

        with patch.object(
            agent, "resolve", return_value=[agent.spec("cuda"), agent.spec("triton")]
        ):
            jobs = agent.expand_jobs([(1, "q1"), (2, "q2")], settings=_S())  # type: ignore[arg-type]
        self.assertEqual(
            jobs,
            [
                (1, "q1", "cuda"),
                (1, "q1", "triton"),
                (2, "q2", "cuda"),
                (2, "q2", "triton"),
            ],
        )


class ProgressKeyTests(unittest.TestCase):
    def test_legacy_row_is_cuda(self) -> None:
        import json
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "progress.jsonl"
            path.write_text(
                json.dumps({"id": 7, "status": "success"})
                + "\n"
                + json.dumps({"id": 8, "dialect": "triton", "status": "success"})
                + "\n",
                encoding="utf-8",
            )
            keys = load_done_keys(path)
        self.assertIn((7, "cuda"), keys)
        self.assertIn((8, "triton"), keys)
        self.assertNotIn((8, "cuda"), keys)


class NvccLogFormatTests(unittest.TestCase):
    def test_gcc_stub_errors_are_summarized(self) -> None:
        from cuda_sft.prompt import format_nvcc_for_prompt

        log = (
            "padding\n" * 400
            + "/tmp/tmpxft_stub.c:1:28: error: reference to '_GLOBAL__N__x' is ambiguous\n"
            + "padding\n" * 400
        )
        summary = format_nvcc_for_prompt(log, max_chars=800)
        self.assertIn("ambiguous", summary)
        self.assertNotIn("0 errors", summary)

    def test_unstructured_tvm_log_keeps_tail(self) -> None:
        from cuda_sft.prompt import format_nvcc_for_prompt

        log = ("noise line\n" * 400) + "name 'N' is not defined\n"
        summary = format_nvcc_for_prompt(log, max_chars=200)
        self.assertIn("name 'N' is not defined", summary)
        self.assertNotIn("0 errors", summary)


class CutlassVersionTests(unittest.TestCase):
    def test_local_cutlass_is_v4(self) -> None:
        from pathlib import Path

        home = Path("/usr/local/cutlass-4.3.5")
        if not home.is_dir():
            self.skipTest("CUTLASS 4.3.5 not installed")
        self.assertEqual(read_cutlass_major(home), 4)


if __name__ == "__main__":
    unittest.main()
