"""Every numeric GPU run must acquire the shared device lock."""

from unittest.mock import patch

from cuda_sft.config import Settings
from cuda_sft.dialects.cuda import CudaDialect
from cuda_sft.refval.runner import GpuFileLock, run_refval
from cuda_sft.refval.spec import CasePlan, KernelABI, KernelParam, RefManifest


def test_single_worker_cannot_run_without_gpu_lock(tmp_path) -> None:
    settings = Settings(workers=1, refval_enabled=True, refval_cases="smoke")
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
    with (
        patch("cuda_sft.refval.runner.toolchain_status", return_value=("ok", "")),
        patch("cuda_sft.refval.runner.validate_reference_fn", return_value=None),
        patch("cuda_sft.refval.runner.prepare_testdir"),
        patch(
            "cuda_sft.refval.runner.compile_cuda_harness",
            return_value=(True, "", tmp_path / "refval_bin"),
        ),
        patch("cuda_sft.refval.runner.run_binary") as run_binary,
        patch.object(GpuFileLock, "acquire", return_value=False) as acquire,
    ):
        report = run_refval(
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
    acquire.assert_called_once()
    run_binary.assert_not_called()
    assert report.status == "fail"
    assert report.reason == "timeout waiting for GPU lock"
