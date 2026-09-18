"""Knowledge-pipeline gates, judge JSON, export, store, and import isolation."""

from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cuda_sft.config import Settings
from cuda_sft.formats import resolve_export_assistant
from cuda_sft.knowledge.facts import fact_violations
from cuda_sft.knowledge.graph import route_after_gate, route_after_judge
from cuda_sft.knowledge.judge import KnowledgeJudge, hard_gate, parse_judge_json
from cuda_sft.knowledge.parse import extract_answer, fence_char_ratio
from cuda_sft.knowledge.rubrics import overall_score, passes_threshold
from cuda_sft.llm import LLMCompletion
from cuda_sft.parse import wrap_cot_assistant
from cuda_sft.store import Store


def _settings(**kwargs: object) -> Settings:
    defaults = {
        "cot_enabled": True,
        "cot_agent_enabled": True,
        "cot_in_assistant": True,
        "cot_max_chars": 8000,
        "cot_raw_max_chars": 24000,
        "cot_on_empty": "empty",
        "cot_on_agent_fail": "raw",
        "knowledge_judge_enabled": True,
        "knowledge_min_score": 7.0,
        "knowledge_factual_min": 6.0,
        "knowledge_min_answer_chars": 400,
        "knowledge_require_structure": True,
        "knowledge_max_candidates": 2,
        "knowledge_max_repairs": 2,
        "knowledge_on_judge_fail": "retry",
        "judge_enabled": False,
        "async_llm_enabled": False,
    }
    defaults.update(kwargs)
    return Settings(**defaults)


STRUCTURED = """## 结论
NVIDIA CUDA 上一个 warp 固定包含 32 个线程。这是编程模型不变量，不随 SM 数量变化。

## 机制
线程被编成 warp，由 SM 上的 warp scheduler 发射。占用率 occupancy 描述活跃 warp 相对最大可驻留 warp 的比例，不等于利用率 utilization。
假设在 {gpu} 这一代上讨论：每 block 最多 1024 线程，因此一个 block 最多 32 个 warp。

## 适用边界
部分数字（SM 数、共享内存容量、TMA 是否存在）随架构变化，必须写明代际。warp 大小在 NVIDIA CUDA 上不是 64。
""".strip()


def _long(body: str) -> str:
    pad = (
        " 补充说明：寄存器、共享内存、L1/L2 与 DRAM 构成存储层次；"
        "合并访存要求同一 warp 访问相邻地址。以上内容用于满足最小篇幅。"
    )
    text = body
    while len(text) < 420:
        text += pad
    return text


class FactCardTests(unittest.TestCase):
    def test_warp_32_ok(self) -> None:
        self.assertEqual(fact_violations("The NVIDIA CUDA warp size is 32 threads."), [])

    def test_warp_64_fails(self) -> None:
        hits = fact_violations("On NVIDIA CUDA the warp size is 64 threads.")
        self.assertTrue(hits)

    def test_cuda_warp_size_128_fails(self) -> None:
        hits = fact_violations("CUDA warp size is 128 threads.")
        self.assertTrue(hits)

    def test_warp_32_with_128_byte_not_flagged(self) -> None:
        text = "CUDA warp size is 32; a 128-byte sector is one coalesced transaction."
        self.assertEqual(fact_violations(text), [])
        text = "cuda warp size 32 vs 128-byte cache line"
        self.assertEqual(fact_violations(text), [])
        text = "A warp is 32 threads; each issues a 128-byte load."
        self.assertEqual(fact_violations(text), [])

    def test_warp_32_not_64_not_flagged(self) -> None:
        text = "The NVIDIA CUDA programming model uses warp size 32, not 64."
        self.assertEqual(fact_violations(text), [])
        text = "A warp is 32 threads (not 16, 64, or 128)."
        self.assertEqual(fact_violations(text), [])

    def test_amd_wavefront_not_flagged(self) -> None:
        text = "AMD CDNA wavefronts are 64 threads; NVIDIA warp size is 32."
        self.assertEqual(fact_violations(text), [])

    def test_unlimited_threads_per_block(self) -> None:
        hits = fact_violations("There is no limit on threads per block in CUDA.")
        self.assertTrue(hits)


class HardGateTests(unittest.TestCase):
    def test_short_answer_fails(self) -> None:
        reasons = hard_gate("too short", topic="architecture", min_chars=400)
        self.assertTrue(any("shorter" in item for item in reasons))

    def test_structured_architecture_passes(self) -> None:
        reasons = hard_gate(_long(STRUCTURED), topic="architecture", min_chars=400)
        self.assertEqual(reasons, [])

    def test_formula_requires_equation(self) -> None:
        body = _long(
            "## 结论\nOccupancy is important.\n\n## 推导\n因此我们讨论限制因素。"
            "假设寄存器够用。step 1 定性分析。step 2 仍然没有等式。\n"
        )
        reasons = hard_gate(body, topic="formula", question="Derive occupancy", min_chars=400)
        self.assertTrue(any("equation" in item for item in reasons))

    def test_code_ratio_fails_on_theory(self) -> None:
        kernel = "```cuda\n" + ("__global__ void k() {}\n" * 40) + "```\n"
        body = "## 结论\nsee code\n" + kernel
        reasons = hard_gate(body, topic="architecture", min_chars=50, require_structure=False)
        self.assertTrue(any("fence" in item for item in reasons))

    def test_extract_answer_drops_think(self) -> None:
        text = "<think>plan</think>\n## 结论\nwarp size is 32.\n"
        self.assertIn("结论", extract_answer(text))
        self.assertNotIn("plan", extract_answer(text))


class RubricTests(unittest.TestCase):
    def test_derivation_zeroed_for_architecture(self) -> None:
        dims = {
            "factual": 9,
            "completeness": 8,
            "derivation": 1,
            "terminology": 8,
            "structure": 8,
            "grounding": 8,
        }
        arch = overall_score(dims, "architecture")
        formula = overall_score(dims, "formula")
        self.assertGreater(arch, formula)

    def test_must_fix_blocks_pass(self) -> None:
        dims = {key: 9 for key in (
            "factual", "completeness", "derivation", "terminology", "structure", "grounding"
        )}
        self.assertFalse(
            passes_threshold(
                overall=9,
                dimensions=dims,
                must_fix=["warp size wrong"],
                min_score=7,
                factual_min=6,
            )
        )


class JudgeJsonTests(unittest.TestCase):
    def test_parse_fenced_json(self) -> None:
        text = (
            "```json\n"
            '{"dimensions": {"factual": 8}, "must_fix": []}\n'
            "```"
        )
        payload = parse_judge_json(text)
        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertEqual(payload["dimensions"]["factual"], 8)

    def test_parse_trailing_comma_and_scores_alias(self) -> None:
        text = '{"scores": {"factual": 8,}, "must_fix": [],}'
        payload = parse_judge_json(text)
        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertIn("factual", payload["dimensions"])

    def test_parse_json_after_cot_prose(self) -> None:
        text = (
            "The user wants me to grade this answer.\n"
            "Factual looks strong.\n"
            '{"dimensions": {"factual": 9, "completeness": 8, "derivation": 10, '
            '"terminology": 8, "structure": 8, "grounding": 8}, "must_fix": []}'
        )
        payload = parse_judge_json(text)
        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertEqual(payload["dimensions"]["factual"], 9)

    def test_parse_json_inside_think_tags(self) -> None:
        text = (
            '<think>planning</think>\n'
            '{"dimensions": {"factual": 7}, "must_fix": []}'
        )
        payload = parse_judge_json(text)
        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertEqual(payload["dimensions"]["factual"], 7)

    def test_reasoning_only_json_is_used(self) -> None:
        payload = {
            "dimensions": {
                "factual": 9,
                "completeness": 8,
                "derivation": 10,
                "terminology": 8,
                "structure": 8,
                "grounding": 8,
            },
            "must_fix": [],
            "issues": [],
        }

        class _ReasoningOnly:
            def stream_completion(self, **_kwargs: object) -> LLMCompletion:
                return LLMCompletion(
                    text="chain of thought without braces",
                    reasoning=json.dumps(payload),
                    reasoning_source="nvidia_delta",
                )

        judge = KnowledgeJudge(
            settings=_settings(),
            llm_client=_ReasoningOnly(),  # type: ignore[arg-type]
        )
        result = judge.judge(
            question="Explain warps",
            answer=_long(STRUCTURED),
            topic="architecture",
        )
        self.assertTrue(result.passed)
        self.assertGreaterEqual(result.overall, 7)

    def test_invalid_json_unavailable_not_saved(self) -> None:
        class _Bad:
            def stream_completion(self, **_kwargs: object) -> LLMCompletion:
                return LLMCompletion(text="not json", reasoning="", reasoning_source="empty")

        judge = KnowledgeJudge(
            settings=_settings(knowledge_on_judge_fail="abandon"),
            llm_client=_Bad(),  # type: ignore[arg-type]
        )
        result = judge.judge(
            question="Explain warps",
            answer=_long(STRUCTURED),
            topic="architecture",
        )
        self.assertFalse(result.passed)
        self.assertTrue(result.judge_unavailable)
        self.assertTrue(result.judge_error)

    def test_good_json_passes(self) -> None:
        payload = {
            "dimensions": {
                "factual": 9,
                "completeness": 8,
                "derivation": 10,
                "terminology": 8,
                "structure": 8,
                "grounding": 8,
            },
            "must_fix": [],
            "issues": ["nit"],
        }

        class _Good:
            def stream_completion(self, **_kwargs: object) -> LLMCompletion:
                return LLMCompletion(
                    text=json.dumps(payload), reasoning="", reasoning_source="empty"
                )

        judge = KnowledgeJudge(settings=_settings(), llm_client=_Good())  # type: ignore[arg-type]
        result = judge.judge(
            question="Explain warps",
            answer=_long(STRUCTURED),
            topic="architecture",
        )
        self.assertTrue(result.passed)
        self.assertGreaterEqual(result.overall, 7)
        self.assertEqual(result.issues, ["nit"])

    def test_split_mode_merges_factual_and_quality(self) -> None:
        class _Split:
            def stream_completion(self, **kwargs: object) -> LLMCompletion:
                system = str(kwargs.get("system") or "")
                if "factual grader" in system:
                    payload = {
                        "dimensions": {"factual": 4, "terminology": 8, "grounding": 8},
                        "must_fix": ["wrong warp size"],
                        "issues": [],
                    }
                else:
                    payload = {
                        "dimensions": {
                            "completeness": 9,
                            "derivation": 9,
                            "structure": 9,
                        },
                        "must_fix": [],
                        "issues": ["long"],
                    }
                return LLMCompletion(
                    text=json.dumps(payload), reasoning="", reasoning_source="empty"
                )

        judge = KnowledgeJudge(
            settings=_settings(knowledge_judge_mode="split"),
            llm_client=_Split(),  # type: ignore[arg-type]
        )
        result = judge.judge(
            question="Explain warps",
            answer=_long(STRUCTURED),
            topic="architecture",
        )
        self.assertFalse(result.passed)
        self.assertIn("wrong warp size", result.must_fix)
        self.assertEqual(result.dimensions["factual"], 4.0)
        self.assertEqual(result.overall, 4.0)

    def test_llm_disabled_hard_gate_only(self) -> None:
        judge = KnowledgeJudge(settings=_settings(knowledge_judge_enabled=False))
        result = judge.judge(
            question="Explain warps",
            answer=_long(STRUCTURED),
            topic="architecture",
        )
        self.assertTrue(result.passed)
        self.assertTrue(result.skipped_llm)


class RouteTests(unittest.TestCase):
    def test_gate_fail_repairs(self) -> None:
        with patch(
            "cuda_sft.knowledge.graph.get_settings",
            return_value=_settings(),
        ):
            self.assertEqual(
                route_after_gate({"gate_ok": False, "repair_idx": 0, "candidate_idx": 1}),
                "repair",
            )
            self.assertEqual(
                route_after_gate({"gate_ok": True, "repair_idx": 0, "candidate_idx": 1}),
                "judge",
            )

    def test_judge_unavailable_abandons(self) -> None:
        with patch(
            "cuda_sft.knowledge.graph.get_settings",
            return_value=_settings(),
        ):
            self.assertEqual(
                route_after_judge({"judge_unavailable": True, "judge_pass": False}),
                "save_abandoned",
            )
            self.assertEqual(
                route_after_judge({"judge_unavailable": False, "judge_pass": True}),
                "cot",
            )


class ExportTests(unittest.TestCase):
    def test_knowledge_keeps_prose(self) -> None:
        answer = "## 结论\nwarp size is 32.\n"
        assistant = wrap_cot_assistant("1. 题意\n讲 warp。", answer)
        row = {
            "messages": [
                {"role": "user", "content": "Explain warps"},
                {"role": "assistant", "content": assistant},
            ],
            "metadata": {"task": "knowledge", "topic": "architecture", "language": "prose"},
        }
        out = resolve_export_assistant(row, cot_in_assistant=True)
        self.assertTrue(out.startswith("<think>"))
        self.assertIn("warp size is 32", out)
        stripped = resolve_export_assistant(row, cot_in_assistant=False)
        self.assertNotIn("<think>", stripped)
        self.assertIn("warp size is 32", stripped)

    def test_kernel_row_still_extracts_cuda(self) -> None:
        code = "#include <cuda_runtime.h>\n__global__ void k() {}\n"
        assistant = wrap_cot_assistant("1. Problem restatement\nAdd tensors.", code)
        row = {
            "messages": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": assistant},
            ],
            "metadata": {"dialect": "cuda", "language": "cuda-cpp"},
        }
        out = resolve_export_assistant(row, cot_in_assistant=False)
        self.assertNotIn("<think>", out)
        self.assertIn("__global__", out)


class PromptLanguageTests(unittest.TestCase):
    def test_english_question_gets_english_suffix(self) -> None:
        from cuda_sft.knowledge.prompt import select_prompts

        selected = select_prompts(
            "Explain CUDA occupancy.",
            question_id=2,
            candidate_idx=1,
            topic="execution",
            gpu_name="GPU",
            cuda_arch="sm_86",
            cuda_version="12.6",
        )
        self.assertIn("Requirements", selected.user)
        self.assertNotIn("答题要求", selected.user)
        self.assertFalse(selected.system.startswith("你是"))

    def test_chinese_question_gets_chinese_suffix(self) -> None:
        from cuda_sft.knowledge.prompt import select_prompts

        selected = select_prompts(
            "请解释 CUDA occupancy。",
            question_id=2,
            candidate_idx=1,
            topic="execution",
            gpu_name="GPU",
            cuda_arch="sm_86",
            cuda_version="12.6",
        )
        self.assertIn("答题要求", selected.user)
        self.assertTrue("你是" in selected.system or "体系结构" in selected.system)


class StoreKnowledgeTests(unittest.TestCase):
    def test_write_success_sets_task_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp))
            state = {
                "kind": "knowledge",
                "question_id": 7,
                "question": "Explain warps",
                "user_prompt": "Explain warps\n\n## 答题要求",
                "answer": _long(STRUCTURED),
                "cot": "1. 题意\nwarp。",
                "cot_source": "agent",
                "topic": "architecture",
                "track": "knowledge:architecture",
                "system_prompt": "You are a teacher.",
                "candidate_idx": 1,
                "repair_idx": 0,
                "judge_score": 8.2,
                "cuda_arch": "sm_86",
                "gpu_name": "RTX",
                "metadata": {
                    "knowledge_judge": {"pass": True, "overall": 8.2},
                    "cot": {"source": "agent", "text": "1. 题意\nwarp。"},
                },
            }
            with patch(
                "cuda_sft.store.get_settings",
                return_value=_settings(),
            ):
                store.write_success(state, model_name="test-model")  # type: ignore[arg-type]
            row = json.loads(store.sft_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(row["metadata"]["task"], "knowledge")
            self.assertEqual(row["metadata"]["language"], "prose")
            self.assertEqual(row["metadata"]["dialect"], "knowledge:architecture")
            self.assertIn("<think>", row["messages"][-1]["content"])
            progress = json.loads(store.progress_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(progress["dialect"], "knowledge:architecture")
            self.assertEqual(progress["status"], "success")


class IsolationTests(unittest.TestCase):
    def test_knowledge_package_does_not_import_kernel_impl(self) -> None:
        root = Path(__file__).resolve().parents[1] / "src" / "cuda_sft" / "knowledge"
        forbidden = re.compile(
            r"from cuda_sft\.(graph|judge|compile|cot|prompt|dialects)(\b|\.)"
            r"|import cuda_sft\.(graph|judge|compile|cot|prompt)\b"
        )
        hits: list[str] = []
        for path in sorted(root.glob("*.py")):
            text = path.read_text(encoding="utf-8")
            if forbidden.search(text):
                hits.append(str(path.name))
        self.assertEqual(hits, [])

    def test_fence_ratio_helper(self) -> None:
        text = "hello\n```cuda\ncode\n```\n"
        self.assertGreater(fence_char_ratio(text), 0.0)
        self.assertEqual(fence_char_ratio("no fences"), 0.0)


if __name__ == "__main__":
    unittest.main()
