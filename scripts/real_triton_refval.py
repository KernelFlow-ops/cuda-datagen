"""Run the real Triton numeric fixtures on the first CUDA GPU.

This is intentionally separate from pytest: it exercises Triton JIT, CUDA
allocation, synchronization, and the Python refval ABI end to end.
"""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

from cuda_sft.config import Settings
from cuda_sft.dialects.triton import TritonDialect
from cuda_sft.refval.runner import run_refval
from cuda_sft.refval.spec import CasePlan, KernelABI, KernelParam, RefManifest, stable_hash
from cuda_sft.refval.cases import build_case_plans


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "refval"


def _jobs():
    add = KernelABI("launch_add", (
        KernelParam("a", "input", "f32", shape_from=("n",)),
        KernelParam("b", "input", "f32", shape_from=("n",)),
        KernelParam("c", "output", "f32", shape_from=("n",)),
        KernelParam("n", "size", "i32"),
    ))
    scale = KernelABI("launch_scale", (
        KernelParam("x", "input", "f32", shape_from=("n",)),
        KernelParam("alpha", "scalar", "f32"),
        KernelParam("out", "output", "f32", shape_from=("n",)),
        KernelParam("n", "size", "i32"),
    ))
    row_sum = KernelABI("launch_row_sum", (
        KernelParam("x", "input", "f32", rank=2, shape_from=("rows", "cols")),
        KernelParam("out", "output", "f32", shape_from=("rows",)),
        KernelParam("rows", "size", "i32"),
        KernelParam("cols", "size", "i32"),
    ))
    ref = {
        "add": 'def reference(a, b, n):\n return {"c": np.asarray(a) + np.asarray(b)}',
        "scale": 'def reference(x, alpha, n):\n return {"out": np.asarray(x) * alpha}',
        "row_sum": 'def reference(x, rows, cols):\n return {"out": [sum(x[i,j] for j in range(cols)) for i in range(rows)]}',
    }
    return [
        ("add", "triton_add_ok.py", add, ref["add"]),
        ("scale", "triton_scale_ok.py", scale, ref["scale"]),
        ("row_sum", "triton_row_sum_ok.py", row_sum, ref["row_sum"]),
        ("add_mutation", "triton_add_wrong.py", add, ref["add"]),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="smoke", choices=("smoke", "standard"))
    args = parser.parse_args()
    settings = Settings(
        refval_enabled=True, refval_cases=args.cases, refval_timeout_sec=180,
        async_llm_enabled=False, workers=1, work_dir=tempfile.mkdtemp(prefix="triton-refval-"),
    )
    dialect_spec = TritonDialect().refval_spec(settings)
    suite_rows = []
    for qid, (name, filename, abi, reference) in enumerate(_jobs(), 1):
        plans = build_case_plans(abi, question_id=qid, dialect="triton", suite=args.cases)
        case_hash = stable_hash([p.to_dict() for p in plans])
        started = time.monotonic()
        report = run_refval(
            question=name, code=(FIXTURES / filename).read_text(encoding="utf-8"),
            question_id=qid, dialect="triton", dialect_spec=dialect_spec, settings=settings,
            workdir=Path(settings.work_dir) / name,
            manifest=RefManifest(question_id=qid, dialect="triton", abi=abi,
                                 reference_source=reference, extracted_from="injected"),
        )
        elapsed = time.monotonic() - started
        row = {"name": name, "status": report.status, "error_class": report.error_class,
               "cases_hash": report.cases_hash or case_hash,
               "elapsed_sec": round(report.elapsed_sec or elapsed, 3)}
        suite_rows.append(row)
        print(row)
    return 0 if all(r["status"] == ("fail" if r["name"] == "add_mutation" else "pass") for r in suite_rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
