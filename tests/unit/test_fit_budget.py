"""Input-budget clipping preserves valid chat history and the latest request."""

from __future__ import annotations

from copy import deepcopy
from itertools import pairwise

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from cuda_sft.llm import approx_tokens, fit_to_input_budget


def _estimated_total(system: str, messages: list[dict[str, str]]) -> int:
    return (
        approx_tokens(system)
        + 4
        + sum(approx_tokens(message["content"]) + 4 for message in messages)
    )


def test_drops_oldest_complete_exchange() -> None:
    messages = [
        {"role": "user", "content": "original question"},
        {"role": "assistant", "content": "old answer " + "a" * 300},
        {"role": "user", "content": "old repair " + "b" * 300},
        {"role": "assistant", "content": "recent answer"},
        {"role": "user", "content": "latest repair"},
    ]
    original = deepcopy(messages)
    system, fitted = fit_to_input_budget("system", messages, 80)
    assert fitted == [messages[0], messages[3], messages[4]]
    assert _estimated_total(system, fitted) <= 80
    assert messages == original


def test_latest_user_keeps_head_and_tail() -> None:
    messages = [
        {"role": "user", "content": "original question"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "START:" + "x" * 400 + ":END"},
    ]
    system, fitted = fit_to_input_budget("system", messages, 75)
    latest = fitted[-1]["content"]
    assert latest.startswith("START:")
    assert latest.endswith(":END")
    assert "\n...[truncated]...\n" in latest
    assert _estimated_total(system, fitted) <= 75


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [{"role": "assistant", "content": "reply"}],
        [{"role": "user", "content": "question"}, {"role": "assistant", "content": "reply"}],
        [{"role": "user", "content": "one"}, {"role": "user", "content": "two"}],
        [{"content": "missing role"}],
        [{"role": "tool", "content": "output"}],
    ],
)
@pytest.mark.parametrize("budget", [0, 1000])
def test_rejects_invalid_turn_order_at_any_budget(
    messages: list[dict[str, str]], budget: int
) -> None:
    with pytest.raises(ValueError, match="alternate user/assistant"):
        fit_to_input_budget("system", messages, budget)


_content = st.text(
    alphabet=st.sampled_from(("a", "b", "x", " ", "\n", "\u4e2d", "\U0001f680")),
    max_size=80,
)


@st.composite
def _valid_history(draw):
    exchange_count = draw(st.integers(min_value=0, max_value=14))
    contents = draw(
        st.lists(_content, min_size=2 * exchange_count + 1, max_size=2 * exchange_count + 1)
    )
    return [
        {"role": "user" if index % 2 == 0 else "assistant", "content": content}
        for index, content in enumerate(contents)
    ]


@settings(max_examples=100, deadline=None)
@given(messages=_valid_history(), system=_content, budget=st.integers(min_value=1, max_value=400))
def test_budget_properties(messages: list[dict[str, str]], system: str, budget: int) -> None:
    original = deepcopy(messages)
    fitted_system, fitted = fit_to_input_budget(system, messages, budget)

    assert messages == original
    assert fitted[0]["role"] == fitted[-1]["role"] == "user"
    assert all(left["role"] != right["role"] for left, right in pairwise(fitted))
    assert fitted[0]["content"] == original[0]["content"] or len(original) == 1

    if _estimated_total(fitted_system, fitted) > budget:
        assert fitted_system == ""
        assert fitted[-1]["content"] == ""
        assert len(fitted) <= 3
