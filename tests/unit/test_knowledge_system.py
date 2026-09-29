"""Knowledge generation context survives repairs and candidate changes."""

from cuda_sft.knowledge.graph import next_candidate, prepare, repair
from cuda_sft.knowledge.prompt import build_cot_user


def test_prepare_repair_and_next_candidate_keep_generation_context(monkeypatch) -> None:
    monkeypatch.setenv("DIFFICULTY_AWARE", "false")
    initial = {
        "question_id": 21,
        "question": "解释 CUDA warp 与 SM 的关系。",
        "kind": "knowledge",
        "topic": "architecture",
    }
    first = prepare(initial)
    assert first["gen_system"] == first["system_prompt"]
    assert first["gen_user"] == first["user_prompt"]
    assert first["gen_prompt_variant"]["temperature"] == first["temperature"]

    repaired = repair({**initial, **first, "answer": "太短", "gate_reasons": ["too short"]})
    assert "rewriting a failed" in repaired["system_prompt"].lower()
    assert not {"gen_system", "gen_user", "gen_prompt_variant"} & repaired.keys()

    second = next_candidate({**initial, **first, **repaired})
    assert second["gen_system"] == second["system_prompt"]
    assert second["gen_user"] == second["user_prompt"]
    assert second["gen_prompt_variant"]["temperature"] == second["temperature"]
    assert "rewriting a failed" not in second["gen_system"].lower()


def test_knowledge_cot_prompt_has_no_repair_history() -> None:
    user = build_cot_user(
        question="解释 warp 与 SM",
        answer="warp 是线程组；SM 执行 warp。",
        raw_reasoning="(none)",
        topic="architecture",
        max_chars=2000,
    )
    assert "warp 是线程组" in user
    for marker in ("repairs=", "gate=", "judge_issues=", "Previous answer"):
        assert marker not in user
