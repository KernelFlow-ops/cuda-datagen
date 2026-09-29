"""Formal generation must reject clients that did not call a provider."""

from __future__ import annotations

from cuda_sft.main import main
from cuda_sft.runtime import deps


def test_formal_cli_rejects_replay_before_writing(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("CUDA_SFT_LLM_REPLAY", str(tmp_path / "cassette"))
    monkeypatch.setenv("KERNEL_FAST_MODE", "true")
    output = tmp_path / "output"
    assert main(["--limit", "1", "--data-dir", str(output)]) == 2
    assert "REPLAY is not allowed" in capsys.readouterr().err
    assert not output.exists()


def test_formal_cli_rejects_injected_client_before_writing(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("KERNEL_FAST_MODE", "true")
    output = tmp_path / "output"
    with deps.use(deps.Deps(llm_factory=lambda _role: object())):
        assert main(["--limit", "1", "--data-dir", str(output)]) == 2
    assert "Injected LLM clients" in capsys.readouterr().err
    assert not output.exists()
