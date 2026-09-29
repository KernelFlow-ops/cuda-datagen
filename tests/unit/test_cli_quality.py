"""Formal CLI keeps full quality unless fast mode is requested."""

from __future__ import annotations

import os

from cuda_sft.main import _apply_kernel_cli, build_parser


def test_kernel_fast_is_opt_in(monkeypatch) -> None:
    monkeypatch.delenv("KERNEL_FAST_MODE", raising=False)
    parser = build_parser()
    _apply_kernel_cli(parser.parse_args([]))
    assert "KERNEL_FAST_MODE" not in os.environ
    _apply_kernel_cli(parser.parse_args(["--kernel-fast"]))
    assert os.environ["KERNEL_FAST_MODE"] == "true"


def test_existing_fast_environment_remains_effective(monkeypatch) -> None:
    monkeypatch.setenv("KERNEL_FAST_MODE", "true")
    _apply_kernel_cli(build_parser().parse_args([]))
    assert os.environ["KERNEL_FAST_MODE"] == "true"
