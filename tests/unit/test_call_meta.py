"""LLM calls expose their logical role and candidate identity."""

import ast
from pathlib import Path

from cuda_sft.agents.critic import KernelCritic
from cuda_sft.agents.generate import complete_chat
from cuda_sft.config import get_settings
from cuda_sft.cot import CotAgent
from cuda_sft.judge import JudgeResult
from cuda_sft.llm import LLMCompletion
from cuda_sft.runtime import deps
from cuda_sft.runtime.meta import CallMeta


class Recorder:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def stream_completion(self, **kwargs: object) -> LLMCompletion:
        self.calls.append(kwargs)
        return LLMCompletion(text='{"pass": true, "must_fix": [], "issues": []}')


def test_generator_critic_and_cot_metadata() -> None:
    recorder = Recorder()
    settings = get_settings().model_copy(update={"kernel_llm_critic": "always"})
    with deps.use(deps.Deps(llm_factory=lambda _role: recorder)):
        meta = CallMeta("generator", "q17:cuda", 17, "cuda", candidate=2)
        complete_chat(
            messages=[{"role": "user", "content": "add"}], system="test", temperature=0.2, meta=meta
        )
        KernelCritic(settings).evaluate(
            question="add",
            code="__global__ void add() {}",
            dialect="cuda",
            heuristic=JudgeResult(quality_score=5, issues=[]),
            meta=CallMeta("critic", "q17:cuda", 17, "cuda", candidate=2),
        )
        CotAgent(settings)._run_agent(
            {
                "question_id": 17,
                "question": "add",
                "dialect": "cuda",
                "code": "add",
                "candidate_idx": 2,
            },
            raw_reasoning="first add values",
        )
    assert [call["meta"].role for call in recorder.calls] == ["generator", "critic", "cot_editor"]
    assert all(call["meta"].question_id == 17 for call in recorder.calls)


def test_every_stream_call_passes_meta() -> None:
    root = Path(__file__).resolve().parents[2] / "src" / "cuda_sft"
    missing = []
    for path in root.rglob("*.py"):
        if path.name == "llm.py" or "testing" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and (
                    (isinstance(node.func, ast.Name) and node.func.id == "stream_completion")
                    or (
                        isinstance(node.func, ast.Attribute)
                        and node.func.attr in {"stream_completion", "stream_text"}
                    )
                )
                and not any(kw.arg == "meta" for kw in node.keywords)
            ):
                missing.append(f"{path.relative_to(root)}:{node.lineno}")
    assert not missing, missing
