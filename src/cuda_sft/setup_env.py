"""Detect and auto-install CUDA / CUTLASS 4.x / Triton / TileLang toolchains.

This module is intentionally stdlib-only at import time so
``python scripts/setup_env.py`` can bootstrap a fresh environment.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CUTLASS_GIT = "https://github.com/NVIDIA/cutlass.git"
CUTLASS_TAG = "v4.3.5"
CUTLASS_MAJOR_RE = re.compile(r"#define\s+CUTLASS_MAJOR\s+(\d+)")

PYTHON_PACKAGES = (
    "langgraph",
    "anthropic",
    "openai",
    "dotenv",
    "pydantic",
    "pydantic_settings",
    "tqdm",
)

DIALECT_ALIASES = {
    "cu": "cuda",
    "cuda-cpp": "cuda",
    "cuda_cpp": "cuda",
    "cute": "cutlass",
    "cutlass/cute": "cutlass",
    "cutlass_cute": "cutlass",
    "cutlass4": "cutlass",
    "cutlass4.x": "cutlass",
    "cutlass-4": "cutlass",
    "python-triton": "triton",
    "triton-lang": "triton",
    "tl": "tilelang",
    "tile-lang": "tilelang",
    "tile_lang": "tilelang",
}
KNOWN_DIALECTS = ("cuda", "cutlass", "triton", "tilelang")


@dataclass
class CheckRow:
    """One environment check line."""

    name: str
    ok: bool
    detail: str
    action: str


def _cutlass_clone_dir() -> Path:
    """Project-local CUTLASS checkout path."""
    return PROJECT_ROOT / "third_party" / "cutlass"


def _run(cmd: list[str], timeout: int = 600, *, capture: bool = True) -> tuple[int, str]:
    """Run a command; return ``(returncode, combined output)``."""
    try:
        if capture:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        else:
            print("+", " ".join(cmd), flush=True)
            proc = subprocess.run(cmd, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, str(exc)
    if not capture:
        return proc.returncode, ""
    merged = "\n".join(
        p for p in ((proc.stdout or "").strip(), (proc.stderr or "").strip()) if p
    )
    return proc.returncode, merged


def _pip_install(*packages: str) -> tuple[bool, str]:
    """Install packages with the current interpreter's pip (progress streamed)."""
    cmd = [sys.executable, "-m", "pip", "install", "-U", *packages]
    code, out = _run(cmd, timeout=1800, capture=False)
    return code == 0, out or ("ok" if code == 0 else "pip failed")


def _conda_exe() -> str | None:
    """Return a conda executable, if any."""
    return shutil.which("conda")


def _conda_prefix() -> str | None:
    """Active conda env prefix, or the current interpreter prefix when it looks like one."""
    prefix = os.environ.get("CONDA_PREFIX")
    if prefix and Path(prefix).is_dir():
        return prefix
    conda_meta = Path(sys.prefix) / "conda-meta"
    if conda_meta.is_dir():
        return sys.prefix
    return None


def _conda_install(*packages: str, channels: tuple[str, ...] = ("conda-forge",)) -> tuple[bool, str]:
    """Install packages into the active conda prefix (non-interactive)."""
    conda = _conda_exe()
    prefix = _conda_prefix()
    if not conda:
        return False, "conda not found"
    if not prefix:
        return False, "not a conda environment"
    cmd = [conda, "install", "-y", "-p", prefix]
    for channel in channels:
        cmd.extend(["-c", channel])
    cmd.extend(packages)
    env = os.environ.copy()
    env["CONDA_ALWAYS_YES"] = "true"
    try:
        print("+", " ".join(cmd), flush=True)
        proc = subprocess.run(cmd, timeout=1800, check=False, env=env)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    return proc.returncode == 0, "ok" if proc.returncode == 0 else "conda install failed"


def _env_lookup(key: str) -> str:
    """Read ``key`` from the process env, then project ``.env``."""
    val = os.environ.get(key)
    if val and val.strip():
        return val.strip()
    path = PROJECT_ROOT / ".env"
    if not path.is_file():
        return ""
    prefix = f"{key}="
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    for line in lines:
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        if raw.startswith(prefix):
            return raw.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def parse_setup_dialects(raw: str | None) -> list[str]:
    """Parse a comma-separated dialect list; default is all known dialects.

    Raises:
        ValueError: Unknown dialect name.
    """
    if not raw or not str(raw).strip():
        return list(KNOWN_DIALECTS)
    names: list[str] = []
    seen: set[str] = set()
    for part in str(raw).split(","):
        item = part.strip().lower().replace(" ", "")
        if not item:
            continue
        item = DIALECT_ALIASES.get(item, item)
        if item not in KNOWN_DIALECTS:
            raise ValueError(
                f"unknown kernel dialect {part!r}; expected one of {', '.join(KNOWN_DIALECTS)}"
            )
        if item not in seen:
            seen.add(item)
            names.append(item)
    return names or list(KNOWN_DIALECTS)


def check_python_packages() -> CheckRow:
    """Locate core Python dependencies without importing them."""
    missing: list[str] = []
    for name in PYTHON_PACKAGES:
        if importlib.util.find_spec(name) is None:
            missing.append(name)
    if missing:
        return CheckRow("python-deps", False, "missing " + ", ".join(missing), "pip")
    return CheckRow("python-deps", True, "langgraph/anthropic/openai/pydantic/tqdm", "ok")


def _iter_nvcc_candidates() -> list[Path]:
    """Search PATH, CUDA_HOME, conda prefix, and pip NVIDIA wheels for nvcc."""
    found: list[Path] = []
    seen: set[str] = set()

    def add(path: Path) -> None:
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        key = str(resolved)
        if key not in seen and path.is_file():
            seen.add(key)
            found.append(path)

    which = shutil.which("nvcc")
    if which:
        add(Path(which))
    for env_key in ("CUDA_HOME", "CUDA_PATH"):
        home = _env_lookup(env_key)
        if home:
            add(Path(home) / "bin" / "nvcc")
    for prefix in (
        _conda_prefix(),
        sys.prefix,
        str(Path(sys.executable).resolve().parent.parent),
        "/usr/local/cuda",
        "/usr/local/cuda-12.6",
        "/usr/local/cuda-12",
        "/usr/local/cuda-13",
    ):
        if prefix:
            add(Path(prefix) / "bin" / "nvcc")
    try:
        import site

        site_dirs = list(site.getsitepackages())
        user_site = site.getusersitepackages()
        if user_site:
            site_dirs.append(user_site)
    except Exception:
        site_dirs = []
    for sp in site_dirs:
        add(Path(sp) / "nvidia" / "cuda_nvcc" / "bin" / "nvcc")
    return found


def _discover_nvcc() -> Path | None:
    """First usable nvcc binary, or None."""
    cands = _iter_nvcc_candidates()
    return cands[0] if cands else None


def _nvcc_release(binary: Path) -> str:
    """Parse ``nvcc --version`` release number."""
    code, out = _run([str(binary), "--version"], timeout=20)
    if code != 0:
        return "?"
    match = re.search(r"release\s+(\d+\.\d+)", out)
    return match.group(1) if match else "?"


def _cuda_home_from_nvcc(nvcc: Path) -> Path:
    """Toolkit root for a given nvcc path."""
    resolved = nvcc.resolve()
    # .../bin/nvcc -> toolkit root (system / conda)
    if resolved.parent.name == "bin":
        return resolved.parent.parent
    return resolved.parent.parent


def check_nvcc(*, skip_install: bool = False) -> CheckRow:
    """Locate nvcc."""
    binary = _discover_nvcc()
    if binary is not None:
        rel = _nvcc_release(binary)
        return CheckRow("cuda/nvcc", True, f"{binary} (CUDA {rel})", "ok")
    action = "manual" if skip_install else "cuda-install"
    return CheckRow("cuda/nvcc", False, "nvcc not found", action)


def check_host_cxx(*, skip_install: bool = False) -> CheckRow:
    """Locate a host C++ compiler (nvcc needs this)."""
    for name in ("g++", "c++", "clang++"):
        path = shutil.which(name)
        if path:
            return CheckRow("host-cxx", True, path, "ok")
    prefix = _conda_prefix()
    if prefix:
        for name in ("g++", "c++", "x86_64-conda-linux-gnu-g++"):
            cand = Path(prefix) / "bin" / name
            if cand.is_file():
                return CheckRow("host-cxx", True, str(cand), "ok")
    action = "manual" if skip_install else "cxx-install"
    return CheckRow("host-cxx", False, "g++ / c++ not found", action)


def check_git() -> CheckRow:
    """Locate git (needed to clone CUTLASS)."""
    git = shutil.which("git")
    if git:
        return CheckRow("git", True, git, "ok")
    return CheckRow("git", False, "git not found (needed to clone CUTLASS)", "manual")


def check_gpu() -> CheckRow:
    """Optional nvidia-smi GPU listing."""
    smi = shutil.which("nvidia-smi")
    if not smi:
        return CheckRow("gpu", False, "nvidia-smi missing (Triton JIT needs a GPU)", "warn")
    code, out = _run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        timeout=20,
    )
    name = out.splitlines()[0].strip() if code == 0 and out else ""
    if name:
        return CheckRow("gpu", True, name, "ok")
    return CheckRow("gpu", True, "NVIDIA GPU", "ok")


def _is_cutlass4_home(home: Path) -> bool:
    """True if ``home`` looks like CUTLASS 4.x with CuTe headers."""
    if not home.is_dir():
        return False
    if not (home / "include" / "cute").is_dir():
        return False
    header = home / "include" / "cutlass" / "version.h"
    if not header.is_file():
        return False
    try:
        text = header.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    match = CUTLASS_MAJOR_RE.search(text)
    return bool(match) and int(match.group(1)) == 4


def _cutlass_candidate_homes() -> list[Path]:
    """Search order: ``CUTLASS_HOME``, system 4.3.5, project ``third_party/cutlass``."""
    homes: list[Path] = []
    seen: set[str] = set()
    raw = _env_lookup("CUTLASS_HOME")
    for item in (
        Path(raw) if raw else None,
        Path("/usr/local/cutlass-4.3.5"),
        _cutlass_clone_dir(),
    ):
        if item is None:
            continue
        key = str(item)
        if key not in seen:
            seen.add(key)
            homes.append(item)
    return homes


def check_cutlass() -> CheckRow:
    """CUTLASS 4.x + CuTe headers."""
    for home in _cutlass_candidate_homes():
        if _is_cutlass4_home(home):
            return CheckRow("cutlass/cute", True, str(home), "ok")
    tried = ", ".join(str(p) for p in _cutlass_candidate_homes())
    return CheckRow("cutlass/cute", False, f"CUTLASS 4.x not found (tried {tried})", "git-clone")


def check_triton() -> CheckRow:
    """Python Triton package (and torch if missing)."""
    spec = importlib.util.find_spec("triton")
    if spec is None:
        return CheckRow("triton", False, "package not installed", "pip")
    try:
        import triton

        ver = getattr(triton, "__version__", "?")
        return CheckRow("triton", True, f"triton {ver}", "ok")
    except ImportError as exc:
        return CheckRow("triton", False, str(exc), "pip")


def check_tilelang() -> CheckRow:
    """Python TileLang package."""
    spec = importlib.util.find_spec("tilelang")
    if spec is None:
        return CheckRow("tilelang", False, "package not installed", "pip")
    try:
        import tilelang
        import tilelang.language as T

        ver = getattr(tilelang, "__version__", "?")
        if not hasattr(T, "prim_func"):
            return CheckRow("tilelang", False, f"{ver} missing T.prim_func", "pip")
        return CheckRow("tilelang", True, f"tilelang {ver}", "ok")
    except ImportError as exc:
        return CheckRow("tilelang", False, str(exc), "pip")


def _wanted_checks(dialects: list[str], *, skip_cuda: bool = False) -> list[CheckRow]:
    """Build the check list for the requested dialects."""
    rows = [check_python_packages(), check_gpu()]
    names = set(dialects)
    if "cuda" in names or "cutlass" in names:
        rows.append(check_host_cxx(skip_install=skip_cuda))
        rows.append(check_nvcc(skip_install=skip_cuda))
    if "cutlass" in names:
        rows.append(check_git())
        rows.append(check_cutlass())
    if "triton" in names:
        rows.append(check_triton())
    if "tilelang" in names:
        rows.append(check_tilelang())
    return rows


def _print_table(rows: list[CheckRow]) -> None:
    """Print a fixed-width status table."""
    print(f"{'component':<16} {'status':<6} {'action':<12} detail", flush=True)
    print("-" * 76, flush=True)
    for row in rows:
        flag = "OK" if row.ok else ("WARN" if row.action == "warn" else "FAIL")
        print(f"{row.name:<16} {flag:<6} {row.action:<12} {row.detail}", flush=True)


def install_python_deps() -> tuple[bool, str]:
    """pip install -r requirements.txt."""
    req = PROJECT_ROOT / "requirements.txt"
    if not req.is_file():
        return False, f"missing {req}"
    return _pip_install("-r", str(req))


def install_triton() -> tuple[bool, str]:
    """Install Triton, and torch if the current env does not have it."""
    pkgs = ["triton"]
    if importlib.util.find_spec("torch") is None:
        pkgs.insert(0, "torch")
    return _pip_install(*pkgs)


def install_cutlass() -> tuple[bool, str]:
    """Clone CUTLASS 4.3.5 into ``third_party/cutlass``."""
    git = shutil.which("git")
    if not git:
        return False, "git not found; cannot clone CUTLASS"
    dest = _cutlass_clone_dir()
    if _is_cutlass4_home(dest):
        return True, str(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        shutil.rmtree(dest)
    cmd = [
        git,
        "clone",
        "--depth",
        "1",
        "--branch",
        CUTLASS_TAG,
        CUTLASS_GIT,
        str(dest),
    ]
    code, out = _run(cmd, timeout=600, capture=False)
    if code != 0:
        return False, out[-1500:] or "git clone failed"
    if not _is_cutlass4_home(dest):
        return False, f"clone succeeded but not CUTLASS 4.x: {dest}"
    return True, str(dest)


def install_host_cxx() -> tuple[bool, str]:
    """Install a host C++ compiler via conda-forge when possible."""
    row = check_host_cxx(skip_install=True)
    if row.ok:
        return True, row.detail
    ok, detail = _conda_install("cxx-compiler", "gxx_linux-64", channels=("conda-forge",))
    row = check_host_cxx(skip_install=True)
    if row.ok:
        return True, row.detail
    if ok:
        return False, "conda cxx-compiler installed but g++ still not on PATH"
    return False, (
        detail + "; install g++ (apt install build-essential, or conda-forge cxx-compiler)"
    )


def install_cuda_toolkit() -> tuple[bool, str]:
    """Best-effort nvcc: conda nvidia channel, then pip CUDA wheels."""
    existing = _discover_nvcc()
    if existing is not None:
        return True, str(existing)

    notes: list[str] = []
    if _conda_exe() and _conda_prefix():
        ok, detail = _conda_install(
            "cuda-nvcc",
            "cuda-cudart-dev",
            "cuda-cccl",
            channels=("nvidia", "conda-forge"),
        )
        notes.append(f"conda: {detail}")
        found = _discover_nvcc()
        if found is not None:
            upsert_env_var("CUDA_HOME", str(_cuda_home_from_nvcc(found)))
            return True, str(found)
        if not ok:
            notes.append("conda cuda-nvcc failed")

    for spec in (
        "cuda-toolkit[nvcc,cudart,cccl]",
        "nvidia-cuda-nvcc-cu12 nvidia-cuda-runtime-cu12 nvidia-cuda-cccl-cu12",
    ):
        pkgs = spec.split()
        ok, detail = _pip_install(*pkgs)
        notes.append(f"pip {spec}: {detail}")
        found = _discover_nvcc()
        if found is not None:
            upsert_env_var("CUDA_HOME", str(_cuda_home_from_nvcc(found)))
            return True, str(found)
        if not ok:
            continue

    hint = (
        "could not auto-install nvcc. Install CUDA Toolkit 12.x from "
        "https://developer.nvidia.com/cuda-downloads "
        "or: conda install -c nvidia cuda-nvcc cuda-cudart-dev cuda-cccl"
    )
    extra = " | ".join(n for n in notes if n)
    return False, f"{hint} ({extra})" if extra else hint


def upsert_env_var(key: str, value: str) -> None:
    """Set ``key=value`` in project ``.env``, backing up to ``.env.bak``."""
    path = PROJECT_ROOT / ".env"
    lines: list[str] = []
    if path.is_file():
        bak = PROJECT_ROOT / ".env.bak"
        shutil.copy(path, bak)
        lines = path.read_text(encoding="utf-8").splitlines()
    found = False
    out: list[str] = []
    prefix = f"{key}="
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(prefix) and not stripped.startswith("#"):
            out.append(f"{key}={value}")
            found = True
        else:
            out.append(line)
    if not found:
        if out and out[-1].strip():
            out.append("")
        out.append(f"{key}={value}")
    path.write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")


def smoke_dialects(names: list[str]) -> list[CheckRow]:
    """Run each dialect's built-in smoke compile."""
    try:
        from cuda_sft.config import get_settings as _gs
        from cuda_sft.dialects.agent import get_dialect_agent
    except ImportError as exc:
        return [CheckRow("smoke", False, f"import failed: {exc}", "fail")]

    _gs.cache_clear()
    settings = _gs()
    agent = get_dialect_agent()
    rows: list[CheckRow] = []
    work = settings.work_path / "_setup_smoke"
    for name in names:
        spec = agent.spec(name)
        ok, reason = spec.available(settings)
        if not ok:
            rows.append(CheckRow(f"smoke:{name}", False, reason, "skip"))
            continue
        result = spec.smoke(settings, work / name)
        if result.ok:
            rows.append(CheckRow(f"smoke:{name}", True, "PASS", "ok"))
        else:
            tail = (result.output or "smoke failed").strip().splitlines()
            detail = tail[-1] if tail else "smoke failed"
            rows.append(CheckRow(f"smoke:{name}", False, detail[:80], "fail"))
    return rows


def _apply_install(row: CheckRow) -> None:
    """Run the installer that matches a failed check row."""
    if row.name == "python-deps":
        ok, detail = install_python_deps()
        print(f"pip requirements: {'OK' if ok else 'FAIL'}", flush=True)
        if not ok:
            print(detail, flush=True)
        return
    if row.name == "triton":
        ok, detail = install_triton()
        print(f"pip triton: {'OK' if ok else 'FAIL'}", flush=True)
        if not ok:
            print(detail, flush=True)
        return
    if row.name == "tilelang":
        ok, detail = _pip_install("tilelang>=0.1.14")
        print(f"pip tilelang: {'OK' if ok else 'FAIL'}", flush=True)
        if not ok:
            print(detail, flush=True)
        return
    if row.name == "cutlass/cute":
        ok, detail = install_cutlass()
        print(f"git cutlass {CUTLASS_TAG}: {'OK' if ok else 'FAIL'} {detail}", flush=True)
        if ok:
            upsert_env_var("CUTLASS_HOME", str(_cutlass_clone_dir()))
            print(f"wrote CUTLASS_HOME={_cutlass_clone_dir()} to .env", flush=True)
        return
    if row.action == "cuda-install":
        ok, detail = install_cuda_toolkit()
        print(f"cuda toolkit: {'OK' if ok else 'FAIL'} {detail}", flush=True)
        return
    if row.action == "cxx-install":
        ok, detail = install_host_cxx()
        print(f"host c++: {'OK' if ok else 'FAIL'} {detail}", flush=True)


def run_setup(
    *,
    dialects: list[str] | None = None,
    check_only: bool = False,
    no_smoke: bool = False,
    skip_cuda: bool = False,
) -> int:
    """Check, optionally install, then smoke-test dialect toolchains.

    Args:
        dialects: Canonical dialect ids; default all four.
        check_only: If True, do not install or modify ``.env``.
        no_smoke: If True, skip dialect smoke compiles.
        skip_cuda: If True, do not try to conda/pip install nvcc or g++.

    Returns:
        0 if every hard requirement is OK (warnings allowed).
    """
    names = list(dialects or list(KNOWN_DIALECTS))
    print(
        f"setup dialects={','.join(names)} python={sys.executable} "
        f"check_only={check_only}",
        flush=True,
    )
    rows = _wanted_checks(names, skip_cuda=skip_cuda)
    _print_table(rows)

    hard_fail = [r for r in rows if not r.ok and r.action != "warn"]
    if check_only:
        return 0 if not hard_fail else 1

    if hard_fail:
        print("\ninstalling missing pieces...", flush=True)
        for row in list(rows):
            if row.ok or row.action in {"ok", "warn", "manual", "skip", "fail"}:
                continue
            _apply_install(row)

    print("\nre-check after install:", flush=True)
    rows = _wanted_checks(names, skip_cuda=skip_cuda)
    _print_table(rows)

    smoke_rows: list[CheckRow] = []
    if not no_smoke:
        print("\nsmoke compiles:", flush=True)
        smoke_rows = smoke_dialects(names)
        _print_table(smoke_rows)

    leftover = [
        r
        for r in rows + smoke_rows
        if not r.ok and r.action not in {"warn", "skip"}
    ]
    if leftover:
        print("\nsetup incomplete:", flush=True)
        for row in leftover:
            print(f"  - {row.name}: {row.detail}", flush=True)
        if any(r.name == "cuda/nvcc" and not r.ok for r in leftover):
            print(
                "  hint: CUDA Toolkit is not auto-installed when conda/pip fail; "
                "install the NVIDIA toolkit and re-run this script.",
                flush=True,
            )
        return 1
    print("\nsetup OK", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """CLI for standalone ``setup_env``."""
    parser = argparse.ArgumentParser(
        description="Detect and auto-install kernel dialect environments "
        "(CUDA, CUTLASS 4.x, Triton, TileLang)",
        epilog="Examples:\n"
        "  python scripts/setup_env.py\n"
        "  python scripts/setup_env.py --check\n"
        "  python scripts/setup_env.py --dialects cuda,triton\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="only detect; do not pip/git/conda or edit .env",
    )
    parser.add_argument(
        "--dialects",
        type=str,
        default=None,
        help="comma-separated dialects (default: cuda,cutlass,triton,tilelang)",
    )
    parser.add_argument(
        "--no-smoke",
        action="store_true",
        help="skip dialect smoke compiles after install",
    )
    parser.add_argument(
        "--skip-cuda",
        action="store_true",
        help="do not try to auto-install CUDA toolkit / host g++",
    )
    parser.add_argument(
        "--setup",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--log-level",
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry: parse args and run setup.

    Unknown flags (from ``run.py``) are ignored so ``python run.py --setup``
    can forward its argv here.
    """
    args, _unknown = build_parser().parse_known_args(argv)
    names = parse_setup_dialects(args.dialects)
    return run_setup(
        dialects=names,
        check_only=bool(args.check),
        no_smoke=bool(args.no_smoke),
        skip_cuda=bool(args.skip_cuda),
    )


if __name__ == "__main__":
    raise SystemExit(main())
