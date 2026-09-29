"""Extraction, harness build, and GPU execution use separate budgets."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from cuda_sft.config import Settings
from cuda_sft.core.gates import FailureOwner, owner_of_refval
from cuda_sft.dialects.cuda import CudaDialect
from cuda_sft.refval import runner
from cuda_sft.refval.spec import CasePlan, KernelABI, KernelParam, RefManifest


def _inputs(tmp_path: Path) -> tuple[Settings, RefManifest, CasePlan]:
    settings = Settings(
        refval_enabled=True,
        refval_cases="smoke",
        refval_cache=False,
        async_llm_enabled=False,
        workers=1,
    )
    object.__setattr__(settings, "refval_extract_timeout_sec", 10)
    object.__setattr__(settings, "refval_build_timeout_sec", 20)
    object.__setattr__(settings, "refval_run_timeout_sec", 7)
    abi = KernelABI(
        entry="launch_add",
        params=(
            KernelParam("a", "input", "f32", rank=1, shape_from=("n",)),
            KernelParam("b", "input", "f32", rank=1, shape_from=("n",)),
            KernelParam("c", "output", "f32", rank=1, shape_from=("n",)),
            KernelParam("n", "size", "i32", rank=0),
        ),
    )
    manifest = RefManifest(
        question_id=1,
        dialect="cuda",
        abi=abi,
        reference_source="def reference(a, b, n):\n    return {'c': a + b}\n",
        reference_fn_name="reference",
        seed=1,
        extracted_from="injected",
    )
    plan = CasePlan(
        name="smoke",
        kind="smoke",
        shapes={"a": (4,), "b": (4,), "c": (4,)},
        scalars={"n": 4},
        seed=1,
    )
    return settings, manifest, plan


def _run(tmp_path: Path, settings: Settings, manifest: RefManifest, plan: CasePlan):
    return runner.run_refval(
        question="vector add",
        code="__global__ void add() {}",
        question_id=1,
        dialect="cuda",
        dialect_spec=CudaDialect().refval_spec(settings),
        settings=settings,
        workdir=tmp_path,
        manifest=manifest,
        case_plans=[plan],
    )


def _stub_pre_gpu(monkeypatch) -> None:
    monkeypatch.setattr(runner, "toolchain_status", lambda *_args: ("ok", ""))
    monkeypatch.setattr(runner, "validate_reference_fn", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "materialize_arrays", lambda *_args: {})


def test_extract_timeout_stops_before_harness(tmp_path, monkeypatch) -> None:
    settings, manifest, plan = _inputs(tmp_path)
    clock = [0.0]
    monkeypatch.setattr(runner, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(runner, "toolchain_status", lambda *_args: ("ok", ""))

    def slow_extract(**_kwargs):
        clock[0] = 11.0
        return manifest

    prepare = Mock()
    monkeypatch.setattr(runner, "obtain_manifest", slow_extract)
    monkeypatch.setattr(runner, "prepare_testdir", prepare)
    report = _run(tmp_path, settings, manifest, plan)
    assert report.reason == runner.REASON_TIMEOUT_BEFORE_GPU
    assert owner_of_refval(report) == (FailureOwner.INFRA, "extract_timeout")
    prepare.assert_not_called()


def test_build_timeout_stops_before_compile_and_gpu(tmp_path, monkeypatch) -> None:
    settings, manifest, plan = _inputs(tmp_path)
    clock = [0.0]
    monkeypatch.setattr(runner, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    _stub_pre_gpu(monkeypatch)

    def slow_prepare(*_args, **_kwargs):
        clock[0] = 21.0

    compile_harness = Mock()
    monkeypatch.setattr(runner, "prepare_testdir", slow_prepare)
    monkeypatch.setattr(runner, "compile_cuda_harness", compile_harness)
    report = _run(tmp_path, settings, manifest, plan)
    assert report.reason == runner.REASON_TIMEOUT_BEFORE_GPU
    compile_harness.assert_not_called()


def test_compile_and_gpu_receive_independent_budgets(tmp_path, monkeypatch) -> None:
    settings, manifest, plan = _inputs(tmp_path)
    clock = [0.0]
    monkeypatch.setattr(runner, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    _stub_pre_gpu(monkeypatch)
    monkeypatch.setattr(runner, "prepare_testdir", lambda *_args, **_kwargs: None)
    compile_args = {}
    run_args = {}
    order = []

    def compile_harness(*_args, **kwargs):
        order.append("compile")
        compile_args.update(kwargs)
        clock[0] = 2.0
        return True, "", Path("refval_bin")

    def run_binary(*_args, **kwargs):
        order.append("run")
        run_args.update(kwargs)
        return 124, "timeout", {"error_class": "timeout", "error": "timeout"}

    def acquire(_self):
        order.append("lock")
        return True

    monkeypatch.setattr(runner, "compile_cuda_harness", compile_harness)
    monkeypatch.setattr(runner, "run_binary", run_binary)
    monkeypatch.setattr(runner.GpuFileLock, "acquire", acquire)
    report = _run(tmp_path, settings, manifest, plan)
    assert compile_args["timeout_sec"] == 20
    assert run_args["timeout_sec"] == 7
    assert order == ["compile", "lock", "run"]
    assert report.error_class == "timeout"


def test_harness_nvcc_timeout_is_infrastructure(tmp_path, monkeypatch) -> None:
    settings, manifest, plan = _inputs(tmp_path)
    _stub_pre_gpu(monkeypatch)
    monkeypatch.setattr(runner, "prepare_testdir", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        runner,
        "compile_cuda_harness",
        lambda *_args, **_kwargs: (False, "nvcc timed out after 20s", None),
    )
    run_binary = Mock()
    monkeypatch.setattr(runner, "run_binary", run_binary)
    report = _run(tmp_path, settings, manifest, plan)
    assert owner_of_refval(report) == (FailureOwner.INFRA, "harness_build_error")
    run_binary.assert_not_called()
