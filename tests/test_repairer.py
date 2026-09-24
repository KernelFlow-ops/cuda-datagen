"""Repairer error classes, question re-injection, and SFT user stripping."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cuda_sft.agents.repairer import classify_compile_error, wrap_repair_user
from cuda_sft.config import Settings
from cuda_sft.prompt import select_prompts
from cuda_sft.store import Store


class ClassifyTests(unittest.TestCase):
    def test_empty_source(self) -> None:
        self.assertEqual(
            classify_compile_error("no cuda source extracted from model output"),
            "empty_source",
        )

    def test_undeclared(self) -> None:
        self.assertEqual(
            classify_compile_error('identifier "foo" is undefined'),
            "undeclared",
        )

    def test_triton_cuda_leak(self) -> None:
        self.assertEqual(
            classify_compile_error("ok", dialect="triton", code="__global__ void k() {}"),
            "dialect_violation",
        )


class WrapRepairTests(unittest.TestCase):
    def test_includes_original_question(self) -> None:
        text = wrap_repair_user(
            question="Write vector add",
            inner="nvcc failed",
            error_class="undeclared",
            dialect="cuda",
        )
        self.assertIn("Write vector add", text)
        self.assertIn("error_class: undeclared", text)
        self.assertIn("nvcc failed", text)

    def test_wrap_includes_evidence(self) -> None:
        text = wrap_repair_user(
            question="Write vector add",
            inner="nvcc failed",
            error_class="numeric_mismatch",
            dialect="cuda",
            evidence="case=odd_7 max_abs=1.0",
        )
        self.assertIn("case=odd_7", text)
        self.assertIn("error_class: numeric_mismatch", text)


class LanguageMatchTests(unittest.TestCase):
    def test_chinese_question_gets_chinese_system(self) -> None:
        selected = select_prompts(
            "请实现一个向量加法 kernel",
            question_id=1,
            candidate_idx=1,
            gpu_name="GPU",
            cuda_arch="sm_86",
            cuda_version="12.6",
        )
        self.assertTrue(
            "你是" in selected.system or "高质量" in selected.system,
            selected.system[:80],
        )
        self.assertIn("include/helpers.h", selected.user)

    def test_english_question_avoids_chinese_system(self) -> None:
        selected = select_prompts(
            "Implement a vector add kernel",
            question_id=1,
            candidate_idx=1,
            gpu_name="GPU",
            cuda_arch="sm_86",
            cuda_version="12.6",
        )
        self.assertFalse(selected.system.startswith("你是"))


class StoreUserTests(unittest.TestCase):
    def test_sft_user_is_raw_question(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp))
            state = {
                "kind": "kernel",
                "question_id": 1,
                "question": "vector add",
                "user_prompt": "vector add\n\n## 生成要求\nnvcc -c",
                "code": "__global__ void k() {}\n",
                "system_prompt": "sys",
                "candidate_idx": 1,
                "repair_idx": 0,
                "dialect": "cuda",
                "metadata": {},
            }
            settings = Settings(
                cot_enabled=False,
                sft_user_is_raw_question=True,
                async_llm_enabled=False,
                refval_strict=False,
            )
            with patch("cuda_sft.store.get_settings", return_value=settings):
                store.write_success(state, model_name="m")  # type: ignore[arg-type]
            row = json.loads(store.sft_path.read_text(encoding="utf-8").splitlines()[0])
            user = next(m["content"] for m in row["messages"] if m["role"] == "user")
            self.assertEqual(user, "vector add")
            self.assertNotIn("nvcc -c", user)
            self.assertIn("nvcc -c", row["metadata"]["generation_user_prompt"])

    def test_critic_and_difficulty_are_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp))
            state = {
                "kind": "kernel",
                "question_id": 2,
                "question": "relu",
                "user_prompt": "relu\n\nnvcc -c",
                "code": "__global__ void k() {}\n",
                "system_prompt": "sys",
                "candidate_idx": 1,
                "repair_idx": 0,
                "dialect": "cuda",
                "difficulty": "simple",
                "candidate_cap": 1,
                "metadata": {
                    "critic": {"passed": True, "skipped": True, "must_fix": [], "issues": []},
                },
            }
            settings = Settings(
                cot_enabled=False,
                sft_user_is_raw_question=True,
                async_llm_enabled=False,
                refval_strict=False,
            )
            with patch("cuda_sft.store.get_settings", return_value=settings):
                store.write_success(state, model_name="m")  # type: ignore[arg-type]
            row = json.loads(store.sft_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertTrue(row["metadata"]["critic"]["skipped"])
            self.assertEqual(row["metadata"]["difficulty"], "simple")
            self.assertEqual(row["metadata"]["candidate_cap"], 1)

    def test_refval_metadata_is_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp))
            state = {
                "kind": "kernel",
                "question_id": 3,
                "question": "add",
                "user_prompt": "add\n\nnvcc -c",
                "code": "__global__ void k() {}\n",
                "system_prompt": "sys",
                "candidate_idx": 1,
                "repair_idx": 0,
                "dialect": "cuda",
                "metadata": {
                    "refval": {
                        "status": "pass",
                        "dialect": "cuda",
                        "cases_run": 4,
                        "failed_case": "",
                        "tolerances": {"f32": {"atol": 1e-4, "rtol": 1e-3}},
                        "manifest_summary": {"entry": "launch_add"},
                        "seed": 1,
                    }
                },
            }
            settings = Settings(
                cot_enabled=False,
                sft_user_is_raw_question=True,
                async_llm_enabled=False,
                refval_strict=False,
            )
            with patch("cuda_sft.store.get_settings", return_value=settings):
                store.write_success(state, model_name="m")  # type: ignore[arg-type]
            row = json.loads(store.sft_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(row["metadata"]["refval"]["status"], "pass")
            self.assertEqual(row["metadata"]["refval"]["cases_run"], 4)


if __name__ == "__main__":
    unittest.main()
