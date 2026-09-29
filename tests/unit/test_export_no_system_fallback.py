"""Explicitly absent system messages stay absent in training exports."""

from cuda_sft.formats import extract_system, to_ms_swift, to_openrlhf


def test_export_does_not_invent_a_system_message() -> None:
    row = {
        "messages": [
            {"role": "user", "content": "add"},
            {"role": "assistant", "content": "code"},
        ],
        "metadata": {"system_mode": "none", "system": "stale generation system"},
    }
    system = extract_system(row)
    assert system == ""
    assert [message["role"] for message in to_ms_swift("add", "code", system=system)["messages"]] == ["user", "assistant"]
    assert [message["role"] for message in to_openrlhf("add", "code", system=system)["input"]] == ["user"]
