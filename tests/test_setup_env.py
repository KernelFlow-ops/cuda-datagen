"""Tests for environment setup helpers (no network)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cuda_sft.dialects.cutlass import is_cutlass4_home, read_cutlass_major
from cuda_sft.setup_env import (
    CheckRow,
    _is_cutlass4_home,
    check_cutlass,
    check_triton,
    parse_setup_dialects,
    run_setup,
    upsert_env_var,
)


class CutlassDetectTests(unittest.TestCase):
    def test_local_4x_home(self) -> None:
        home = Path("/usr/local/cutlass-4.3.5")
        if not home.is_dir():
            self.skipTest("system CUTLASS 4.3.5 not present")
        self.assertEqual(read_cutlass_major(home), 4)
        self.assertTrue(is_cutlass4_home(home))

    def test_missing_home(self) -> None:
        self.assertFalse(is_cutlass4_home(Path("/tmp/does-not-exist-cutlass")))
        self.assertFalse(_is_cutlass4_home(Path("/tmp/does-not-exist-cutlass")))

    def test_fake_cutlass4_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            (home / "include" / "cute").mkdir(parents=True)
            hdr = home / "include" / "cutlass" / "version.h"
            hdr.parent.mkdir(parents=True)
            hdr.write_text("#define CUTLASS_MAJOR 4\n", encoding="utf-8")
            self.assertTrue(_is_cutlass4_home(home))
            hdr.write_text("#define CUTLASS_MAJOR 2\n", encoding="utf-8")
            self.assertFalse(_is_cutlass4_home(home))


class ParseDialectTests(unittest.TestCase):
    def test_default_all(self) -> None:
        self.assertEqual(
            parse_setup_dialects(None),
            ["cuda", "cutlass", "triton", "tilelang"],
        )

    def test_aliases(self) -> None:
        self.assertEqual(parse_setup_dialects("cuda, cute"), ["cuda", "cutlass"])

    def test_unknown(self) -> None:
        with self.assertRaises(ValueError):
            parse_setup_dialects("fortran")


class UpsertEnvTests(unittest.TestCase):
    def test_writes_and_replaces(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".env").write_text("FOO=1\nCUTLASS_HOME=/old\n", encoding="utf-8")
            with patch("cuda_sft.setup_env.PROJECT_ROOT", root):
                upsert_env_var("CUTLASS_HOME", "/new")
                upsert_env_var("CUDA_HOME", "/usr/local/cuda")
            text = (root / ".env").read_text(encoding="utf-8")
            self.assertIn("CUTLASS_HOME=/new", text)
            self.assertNotIn("CUTLASS_HOME=/old", text)
            self.assertIn("FOO=1", text)
            self.assertIn("CUDA_HOME=/usr/local/cuda", text)
            self.assertTrue((root / ".env.bak").is_file())


class CheckOnlyTests(unittest.TestCase):
    def test_check_does_not_call_installers(self) -> None:
        with (
            patch("cuda_sft.setup_env._pip_install") as pip,
            patch("cuda_sft.setup_env.install_cutlass") as clone,
            patch("cuda_sft.setup_env.install_cuda_toolkit") as cuda,
            patch("cuda_sft.setup_env.install_host_cxx") as cxx,
            patch("cuda_sft.setup_env.smoke_dialects") as smoke,
        ):
            code = run_setup(dialects=["cuda"], check_only=True)
        pip.assert_not_called()
        clone.assert_not_called()
        cuda.assert_not_called()
        cxx.assert_not_called()
        smoke.assert_not_called()
        self.assertIn(code, {0, 1})

    def test_check_rows_have_actions(self) -> None:
        row = check_triton()
        self.assertIsInstance(row, CheckRow)
        self.assertTrue(row.action in {"ok", "pip"})
        cut = check_cutlass()
        self.assertTrue(cut.ok or cut.action == "git-clone")

    def test_install_cutlass_when_missing(self) -> None:
        cut_rows = iter(
            [
                CheckRow("cutlass/cute", False, "missing", "git-clone"),
                CheckRow("cutlass/cute", True, "/tmp/cutlass", "ok"),
            ]
        )

        def _cut() -> CheckRow:
            try:
                return next(cut_rows)
            except StopIteration:
                return CheckRow("cutlass/cute", True, "/tmp/cutlass", "ok")

        with (
            patch(
                "cuda_sft.setup_env.check_python_packages",
                return_value=CheckRow("python-deps", True, "ok", "ok"),
            ),
            patch("cuda_sft.setup_env.check_gpu", return_value=CheckRow("gpu", True, "gpu", "ok")),
            patch(
                "cuda_sft.setup_env.check_host_cxx",
                return_value=CheckRow("host-cxx", True, "g++", "ok"),
            ),
            patch(
                "cuda_sft.setup_env.check_nvcc",
                return_value=CheckRow("cuda/nvcc", True, "nvcc", "ok"),
            ),
            patch("cuda_sft.setup_env.check_git", return_value=CheckRow("git", True, "git", "ok")),
            patch("cuda_sft.setup_env.check_cutlass", side_effect=_cut),
            patch("cuda_sft.setup_env.install_cutlass", return_value=(True, "/tmp/cutlass")) as clone,
            patch("cuda_sft.setup_env.upsert_env_var") as upsert,
            patch("cuda_sft.setup_env.smoke_dialects", return_value=[]),
        ):
            code = run_setup(dialects=["cutlass"], check_only=False, no_smoke=True)
        clone.assert_called_once()
        upsert.assert_called()
        self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
