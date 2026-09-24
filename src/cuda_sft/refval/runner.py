"""Orchestrate ABI extract → cases → reference → harness → compare.

GPU access is serialized with ``work/.refval_gpu.lock`` when ``WORKERS>1``.
Lock wait counts toward ``REFVAL_TIMEOUT_SEC``. Missing toolchains return
``skip`` (does not block save). ``reference_error`` is not a kernel failure
unless ``REFVAL_STRICT`` is on.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from cuda_sft.config import Settings, get_settings
from cuda_sft.refval.cases import (
    build_case_plans,
    case_plan_hash,
    materialize_arrays,
    numpy_available,
    write_cases,
)
from cuda_sft.refval.compare import (
    classify_from_results,
    compare_outputs,
    dump_artifacts,
    load_gpu_outputs,
)
from cuda_sft.refval.evidence import compress_evidence
from cuda_sft.refval.harness import has_main, prepare_testdir, run_prepared
from cuda_sft.refval.reference import (
    bind_and_call,
    load_reference_fn,
    validate_reference_fn,
)
from cuda_sft.refval.spec import (
    CaseResult,
    DEFAULT_TOLERANCES,
    DialectRefvalSpec,
    REFVAL_CASE_SUITE_VERSION,
    REFVAL_HARNESS_VERSION,
    REFVAL_SCHEMA_VERSION,
    REFVAL_VALIDATOR_VERSION,
    RefManifest,
    RefvalReport,
    seed_for,
    stable_hash,
    strict_refval_enabled,
    tolerances_for,
)

logger = logging.getLogger(__name__)


def classify_refval_error(
    error: str = "",
    *,
    report: RefvalReport | None = None,
) -> str:
    """Label a refval failure for the repair prompt.

    Returns one of ``numeric_mismatch`` / ``nan_inf`` / ``crash`` /
    ``timeout`` / ``signature_mismatch`` / ``reference_error``.
    """
    if report is not None and report.error_class:
        return report.error_class
    text = (error or (report.reason if report is not None else "") or "").lower()
    if "timeout" in text or "timed out" in text:
        return "timeout"
    if "reference" in text:
        return "reference_error"
    if any(tok in text for tok in ("signature", "undeclared", "not declared", "no matching")):
        return "signature_mismatch"
    if "nan" in text or "inf mismatch" in text:
        return "nan_inf"
    if any(tok in text for tok in ("crash", "cuda error", "segfault", "aborted")):
        return "crash"
    return "numeric_mismatch"


def _testdir_for(
    settings: Settings,
    question_id: int,
    dialect: str,
    *,
    nest_dialect: bool,
    workdir: Path | None = None,
) -> Path:
    if workdir is not None:
        return Path(workdir) / "test"
    base = settings.work_path / f"q{int(question_id)}"
    if nest_dialect:
        base = base / (dialect or "cuda")
    return base / "test"


def _cache_descriptor(settings: Settings, question: str, code: str, dialect: str) -> dict[str, Any]:
    """Describe every input that can change extraction/validation semantics."""
    suite = str(getattr(settings, "refval_cases", "standard") or "standard")
    return {
        "question": question,
        "dialect": dialect or "cuda",
        "code": code,
        "arch": str(getattr(settings, "resolved_cuda_arch", "") or getattr(settings, "cuda_arch", "") or ""),
        "case_suite": suite,
        "case_suite_version": REFVAL_CASE_SUITE_VERSION,
        "validator_version": REFVAL_VALIDATOR_VERSION,
        "harness_version": REFVAL_HARNESS_VERSION,
        "schema_version": REFVAL_SCHEMA_VERSION,
    }


def _cache_hash(settings: Settings, question: str, code: str, dialect: str) -> str:
    return stable_hash(_cache_descriptor(settings, question, code, dialect))


def _cache_path(settings: Settings, question: str, code: str, dialect: str) -> Path:
    digest = _cache_hash(settings, question, code, dialect)
    return settings.work_path / "_refval_cache" / f"{digest}.json"


def _load_cache(path: Path) -> RefManifest | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    try:
        manifest = RefManifest.from_dict(payload)
    except Exception:
        return None
    return manifest


def _store_cache(path: Path, manifest: RefManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def toolchain_status(
    dialect_spec: DialectRefvalSpec,
    settings: Settings,
) -> tuple[str, str]:
    """Return ``('ok', '')`` or ``('skip', reason)`` when a dependency is missing."""
    if dialect_spec.needs_nvcc:
        nvcc = getattr(settings, "nvcc_bin", None) or shutil.which("nvcc")
        if not nvcc or not shutil.which(str(nvcc)) and not Path(str(nvcc)).is_file():
            which = shutil.which("nvcc")
            if not which and not Path(str(nvcc)).is_file():
                return "skip", "nvcc is not installed"
        if shutil.which("nvidia-smi") is None:
            return "skip", "no GPU (nvidia-smi missing)"
    if dialect_spec.needs_torch:
        try:
            import torch  # type: ignore
        except ImportError:
            return "skip", "torch is not installed"
        if not torch.cuda.is_available():
            return "skip", "torch CUDA is not available"
    if not numpy_available():
        return "skip", "numpy is not installed"
    return "ok", ""


class GpuFileLock:
    """Exclusive flock; wait time is the caller's problem (counts toward timeout)."""

    def __init__(self, path: Path, timeout_sec: float) -> None:
        self.path = path
        self.timeout_sec = max(0.1, float(timeout_sec))
        self._handle: Any = None

    def acquire(self) -> bool:
        import fcntl

        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+", encoding="utf-8")
        deadline = time.monotonic() + self.timeout_sec
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._handle = handle
                return True
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    handle.close()
                    return False
                time.sleep(0.05)

    def release(self) -> None:
        import fcntl

        handle = self._handle
        self._handle = None
        if handle is None:
            return
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()

    def __enter__(self) -> "GpuFileLock":
        if not self.acquire():
            raise TimeoutError(f"GPU lock wait exceeded {self.timeout_sec:.1f}s")
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


def _llm_complete(user: str, system: str, settings: Settings) -> str:
    from cuda_sft.llm import get_llm_client

    client = get_llm_client(settings)
    stream_completion = getattr(client, "stream_completion", None)
    if callable(stream_completion):
        completion = stream_completion(
            messages=[{"role": "user", "content": user}],
            system=system,
            temperature=0.0,
            print_stream=False,
            thinking_level="none",
            max_output_tokens=8192,
        )
        return completion.text or completion.reasoning or ""
    return client.stream_text(
        messages=[{"role": "user", "content": user}],
        system=system,
        temperature=0.0,
        print_stream=False,
    )


def _try_speculative(request_id: str | None, settings: Settings) -> str:
    if not request_id or not getattr(settings, "async_llm_enabled", True):
        return ""
    try:
        from cuda_sft.llm_async import get_async_pool

        pool = get_async_pool(settings.async_llm_max_workers)
        remaining = max(1.0, float(getattr(settings, "refval_timeout_sec", 45)) * 0.5)
        completion = pool.try_get(request_id, timeout_sec=remaining)
        if completion is None:
            return ""
        return completion.text or completion.reasoning or ""
    except Exception:
        logger.exception("refval speculative get failed")
        return ""


def obtain_manifest(
    *,
    question: str,
    code: str,
    question_id: int,
    dialect: str,
    settings: Settings,
    dialect_spec: DialectRefvalSpec | None = None,
    speculative_id: str | None = None,
    injected: RefManifest | None = None,
) -> RefManifest | None:
    """Prefetch / cache / LLM / heuristic ABI+reference extract."""
    from cuda_sft.parse import (
        abi_matches_source,
        diagnose_refval_extract,
        heuristic_cuda_abi,
        parse_refval_manifest,
    )
    from cuda_sft.prompts.refval import EXTRACT_SYSTEM, extract_user, retry_user

    if injected is not None:
        return injected
    seed = seed_for(int(question_id), dialect)
    cache_path = _cache_path(settings, question, code, dialect)
    if getattr(settings, "refval_cache", True):
        cached = _load_cache(cache_path)
        # An empty issue list means the cached ABI is valid.  The previous
        # negation discarded every valid cache hit and forced an LLM extract.
        if cached is not None and abi_matches_source(cached.abi, code):
            cached = None
        if cached is not None:
            return cached

    def _parse(text: str) -> RefManifest | None:
        if not text:
            return None
        return parse_refval_manifest(
            text, question_id=question_id, dialect=dialect, seed=seed, extracted_from="llm"
        )

    last_text = _try_speculative(speculative_id, settings)
    manifest = _parse(last_text)
    need_llm = manifest is None or bool(abi_matches_source(manifest.abi, code))
    if need_llm:
        user = extract_user(
            question=question,
            code=code,
            dialect=dialect,
            host_entry_hint=(dialect_spec.host_entry_hint if dialect_spec else ""),
        )
        try:
            text = _llm_complete(user, EXTRACT_SYSTEM, settings)
            if text:
                last_text = text
            parsed = _parse(text)
            if parsed is not None:
                manifest = parsed
        except Exception as exc:
            logger.warning("refval extract LLM failed: %s", exc)

    # parse_refval_manifest drops ABI-only JSON and ABI self-check failures.
    # Those are exactly the cases retry_user was written for, so recover the
    # reason from the raw reply before falling back to a referenceless ABI.
    issues: list[str] = []
    if manifest is not None:
        issues = abi_matches_source(manifest.abi, code)
        if not (manifest.reference_source or "").strip():
            issues.append("empty reference_source")
    elif last_text or need_llm:
        issues = diagnose_refval_extract(last_text)
    if issues:
        logger.warning(
            "refval extract needs retry q%s %s: %s",
            question_id,
            dialect,
            "; ".join(issues)[:400],
        )
        try:
            text = _llm_complete(
                retry_user(question=question, code=code, issues=issues),
                EXTRACT_SYSTEM,
                settings,
            )
            retry = _parse(text)
            if retry is not None and not abi_matches_source(retry.abi, code):
                manifest = retry
            else:
                reasons = list(diagnose_refval_extract(text))
                if retry is not None:
                    reasons.extend(abi_matches_source(retry.abi, code))
                if not reasons:
                    reasons = ["unparsed extractor response"]
                logger.warning(
                    "refval extract retry still unusable q%s %s: %s",
                    question_id,
                    dialect,
                    "; ".join(reasons)[:400],
                )
        except Exception as exc:
            logger.warning("refval extract retry failed: %s", exc)

    # Never return a manifest whose final ABI self-check still has issues.
    # This keeps malformed extractor output from reaching harness generation.
    if manifest is not None:
        final_issues = abi_matches_source(manifest.abi, code)
        if not (manifest.reference_source or "").strip():
            final_issues.append("empty reference_source")
        if final_issues:
            manifest = None

    if manifest is None:
        heuristic = heuristic_cuda_abi(code)
        if heuristic is None:
            return None
        return RefManifest(
            question_id=int(question_id),
            dialect=dialect,
            abi=heuristic,
            reference_source="",
            seed=seed,
            extracted_from="heuristic",
            notes="heuristic ABI only; no CPU reference",
        )
    if getattr(settings, "refval_cache", True):
        try:
            _store_cache(cache_path, manifest)
        except OSError:
            pass
    return manifest


def enqueue_speculative_extract(
    *,
    question: str,
    code: str,
    question_id: int,
    dialect: str,
    candidate: int,
    repair: int,
    dialect_spec: DialectRefvalSpec | None = None,
    settings: Settings | None = None,
) -> str:
    """Prefetch ABI+reference extraction on the existing async LLM pool.

    Returns the request id (empty when async is disabled or enqueue fails).
    """
    settings = settings or get_settings()
    if not getattr(settings, "async_llm_enabled", True):
        return ""
    if not (code or "").strip():
        return ""
    from cuda_sft.agents.generate import enqueue_speculative_repair
    from cuda_sft.prompts.refval import EXTRACT_SYSTEM, extract_user, speculative_request_id

    request_id = speculative_request_id(question_id, dialect, candidate, repair)
    user = extract_user(
        question=question,
        code=code,
        dialect=dialect,
        host_entry_hint=(dialect_spec.host_entry_hint if dialect_spec else ""),
    )
    ok = enqueue_speculative_repair(
        request_id=request_id,
        messages=[{"role": "user", "content": user}],
        system=EXTRACT_SYSTEM,
        temperature=0.0,
        llm_options={"thinking_level": "none", "max_output_tokens": 8192},
    )
    return request_id if ok else ""


def _effective_tolerances(manifest: RefManifest) -> dict[str, dict[str, float]]:
    out = {k: dict(v) for k, v in DEFAULT_TOLERANCES.items()}
    for key, value in (manifest.tolerances or {}).items():
        out[key] = tolerances_for(key, value)
    return out


def run_refval(
    *,
    question: str,
    code: str,
    question_id: int,
    dialect: str,
    dialect_spec: DialectRefvalSpec,
    settings: Settings | None = None,
    workdir: Path | None = None,
    nest_dialect: bool = False,
    used_rdc: bool = False,
    speculative_id: str | None = None,
    manifest: RefManifest | None = None,
    task_spec: Mapping[str, Any] | None = None,
    oracle_spec: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
    case_plans: list[Any] | None = None,
) -> RefvalReport:
    """Run the full numeric gate. Never raises on toolchain/LLM failure."""
    settings = settings or get_settings()
    started = time.monotonic()
    dialect = dialect or dialect_spec.dialect or "cuda"
    budget = float(getattr(settings, "refval_timeout_sec", 45) or 45)
    seed = seed_for(int(question_id), dialect)
    cache_hash = _cache_hash(settings, question, code, dialect)
    case_suite = str(getattr(settings, "refval_cases", "standard") or "standard")
    testdir = _testdir_for(
        settings, int(question_id), dialect, nest_dialect=nest_dialect, workdir=workdir
    ).resolve()

    def _elapsed() -> float:
        return time.monotonic() - started

    def _finish(report: RefvalReport) -> RefvalReport:
        report.elapsed_sec = _elapsed()
        report.seed = report.seed or seed
        report.cache_hash = report.cache_hash or cache_hash
        report.case_suite = report.case_suite or case_suite
        report.validator_version = report.validator_version or REFVAL_VALIDATOR_VERSION
        report.harness_version = report.harness_version or REFVAL_HARNESS_VERSION
        report.schema_version = report.schema_version or REFVAL_SCHEMA_VERSION
        report.case_suite_version = report.case_suite_version or REFVAL_CASE_SUITE_VERSION
        runtime_provenance = {
            "question_id": int(question_id),
            "dialect": dialect,
            "arch": str(getattr(settings, "resolved_cuda_arch", "") or getattr(settings, "cuda_arch", "") or ""),
            "case_suite": case_suite,
        }
        report.provenance = {**runtime_provenance, **report.provenance}
        if manifest is not None:
            manifest_payload = manifest.to_dict()
            report.manifest_hash = report.manifest_hash or str(
                manifest_payload.get("manifest_hash") or stable_hash(manifest_payload)
            )
            report.task_spec = report.task_spec or dict(manifest.task_spec)
            report.oracle_spec = report.oracle_spec or dict(manifest.oracle_spec)
            report.provenance = {**manifest.provenance, **report.provenance}
            report.semantic_contract = report.semantic_contract or dict(manifest.semantic_contract or manifest.task_spec)
            report.backend_contract = report.backend_contract or dict(manifest.backend_contract or manifest.oracle_spec)
            report.contract_version = report.contract_version or getattr(manifest, "contract_version", "")
        if report.status == "fail" and not report.evidence:
            report.evidence = compress_evidence(report)
        try:
            dump_artifacts(
                testdir,
                report=report,
                manifest_text=(
                    json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2) + "\n"
                    if manifest is not None
                    else None
                ),
            )
        except OSError:
            logger.exception("failed to dump refval artifacts")
        return report

    if not getattr(settings, "refval_enabled", True):
        return _finish(RefvalReport.skipped(dialect, "refval disabled", seed=seed))

    status, reason = toolchain_status(dialect_spec, settings)
    if status != "ok":
        return _finish(RefvalReport.skipped(dialect, reason, seed=seed))

    if not (code or "").strip():
        return _finish(
            RefvalReport(
                status="fail",
                dialect=dialect,
                error_class="signature_mismatch",
                reason="empty source",
                seed=seed,
            )
        )
    if dialect_spec.runner == "nvcc_link" and has_main(code):
        return _finish(
            RefvalReport(
                status="fail",
                dialect=dialect,
                error_class="signature_mismatch",
                reason="solution defines main(); numeric harness cannot link a second main",
                seed=seed,
            )
        )

    try:
        manifest = obtain_manifest(
            question=question,
            code=code,
            question_id=int(question_id),
            dialect=dialect,
            settings=settings,
            dialect_spec=dialect_spec,
            speculative_id=speculative_id,
            injected=manifest,
        )
    except Exception as exc:
        logger.warning("refval obtain_manifest failed: %s", exc)
        manifest = None

    if manifest is None:
        report = RefvalReport.reference_error(dialect, "ABI/reference extraction failed", seed=seed)
        if getattr(settings, "refval_strict", False):
            report.status = "fail"
        return _finish(report)

    # The graph freezes these fields before generation.  Attach them to the
    # extracted manifest so the LLM extractor cannot silently change the
    # contract used for downstream provenance and cache validation.
    if task_spec or oracle_spec or provenance:
        manifest = replace(
            manifest,
            task_spec=dict(task_spec or manifest.task_spec),
            oracle_spec=dict(oracle_spec or manifest.oracle_spec),
            provenance={**manifest.provenance, **dict(provenance or {})},
        )

    if not (manifest.reference_source or "").strip():
        report = RefvalReport.reference_error(
            dialect, "no CPU reference (heuristic ABI only)", seed=seed
        )
        report.manifest_summary = manifest.summary()
        if getattr(settings, "refval_strict", False):
            report.status = "fail"
        return _finish(report)

    try:
        ref_fn = load_reference_fn(manifest.reference_source, manifest.reference_fn_name)
        ref_issue = validate_reference_fn(ref_fn, manifest.abi, seed=manifest.seed or seed)
    except Exception as exc:
        report = RefvalReport.reference_error(dialect, str(exc), seed=seed)
        report.manifest_summary = manifest.summary()
        if getattr(settings, "refval_strict", False):
            report.status = "fail"
        return _finish(report)
    if ref_issue:
        report = RefvalReport.reference_error(dialect, ref_issue, seed=seed)
        report.manifest_summary = manifest.summary()
        if getattr(settings, "refval_strict", False):
            report.status = "fail"
        return _finish(report)

    suite = str(getattr(settings, "refval_cases", "standard") or "standard")
    max_elements = int(getattr(settings, "refval_max_elements", 4_000_000) or 4_000_000)
    plans = list(case_plans) if case_plans is not None else build_case_plans(
        manifest.abi,
        question_id=int(question_id),
        dialect=dialect,
        suite=suite,
        max_elements=max_elements,
    )
    planned_cases_hash = case_plan_hash(plans)
    # Keep the case identity in every subsequent report and artifact. This is
    # the cross-dialect invariant; a dialect must never silently regenerate a
    # different input suite.
    testdir.mkdir(parents=True, exist_ok=True)
    arrays_by_case = {plan.name: materialize_arrays(plan, manifest.abi) for plan in plans}
    write_cases(testdir, manifest.abi, plans, arrays_by_case)
    try:
        prepare_testdir(
            testdir,
            source=code,
            filename=dialect_spec.source_filename,
            abi=manifest.abi,
            plans=plans,
            dialect_spec=dialect_spec,
        )
        # prepare_testdir rewrites bins; restore the arrays we already materialized
        write_cases(testdir, manifest.abi, plans, arrays_by_case)
        for plan in plans:
            (testdir / "out" / plan.name).mkdir(parents=True, exist_ok=True)
        (testdir / "manifest.json").write_text(
            json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except Exception as exc:
        return _finish(
            RefvalReport(
                status="fail",
                dialect=dialect,
                error_class="crash",
                reason=f"failed to prepare harness: {exc}",
                manifest_summary=manifest.summary(),
                seed=seed,
            )
        )

    remaining = budget - _elapsed()
    if remaining <= 1:
        return _finish(
            RefvalReport(
                status="fail",
                dialect=dialect,
                error_class="timeout",
                reason="timeout before GPU run",
                manifest_summary=manifest.summary(),
                seed=seed,
            )
        )

    workers = int(getattr(settings, "workers", 1) or 1)
    lock = None
    if workers > 1:
        lock = GpuFileLock(settings.work_path / ".refval_gpu.lock", remaining)
        try:
            lock.acquire()
        except Exception:
            lock = None
        if lock is not None and lock._handle is None:
            return _finish(
                RefvalReport(
                    status="fail",
                    dialect=dialect,
                    error_class="timeout",
                    reason="timeout waiting for GPU lock",
                    manifest_summary=manifest.summary(),
                    seed=seed,
                )
            )
        remaining = budget - _elapsed()

    try:
        code_exit, output, payload = run_prepared(
            testdir,
            dialect_spec,
            settings=settings,
            used_rdc=used_rdc,
            timeout_sec=max(1, int(remaining)),
        )
    except Exception as exc:
        return _finish(
            RefvalReport(
                status="fail",
                dialect=dialect,
                error_class="crash",
                reason=str(exc),
                manifest_summary=manifest.summary(),
                seed=seed,
            )
        )
    finally:
        if lock is not None:
            lock.release()

    tols = _effective_tolerances(manifest)
    if code_exit == 124 or (payload or {}).get("error_class") == "timeout":
        return _finish(
            RefvalReport(
                status="fail",
                dialect=dialect,
                error_class="timeout",
                reason=(payload or {}).get("error") or output or "timeout",
                manifest_summary=manifest.summary(),
                tolerances=tols,
                seed=seed,
            )
        )
    payload_class = str((payload or {}).get("error_class") or "")
    if payload_class in {
        "signature_mismatch", "shape_contract", "stride_mismatch", "dtype_mismatch",
        "argument_binding", "driver_import", "entry_missing", "launch_runtime",
        "cuda_illegal_memory", "output_missing",
    }:
        return _finish(
            RefvalReport(
                status="fail",
                dialect=dialect,
                error_class=payload_class,
                reason=(payload or {}).get("error") or output or payload_class,
                manifest_summary=manifest.summary(),
                tolerances=tols,
                seed=seed,
            )
        )
    if payload_class == "signature_mismatch" or (
        code_exit != 0 and "error:" in (output or "").lower() and "nvcc" in (output or "").lower()
    ):
        return _finish(
            RefvalReport(
                status="fail",
                dialect=dialect,
                error_class="signature_mismatch",
                reason=(payload or {}).get("error") or output or "harness compile failed",
                manifest_summary=manifest.summary(),
                tolerances=tols,
                seed=seed,
            )
        )
    if code_exit not in (0, 1) and not (payload or {}).get("cases"):
        return _finish(
            RefvalReport(
                status="fail",
                dialect=dialect,
                error_class="crash",
                reason=(payload or {}).get("error") or output or f"harness exit {code_exit}",
                manifest_summary=manifest.summary(),
                tolerances=tols,
                seed=seed,
            )
        )

    results = []
    failed_case = ""
    harness_cases = {
        str(item.get("name")): item
        for item in (payload or {}).get("cases") or []
        if isinstance(item, dict)
    }
    for plan in plans:
        hc = harness_cases.get(plan.name) or {}
        if hc.get("not_applicable"):
            results.append(
                CaseResult(
                    name=plan.name,
                    ok=True,
                    status="skip",
                    error=str(hc.get("error") or "zero extent unsupported by backend"),
                    seed=plan.seed,
                )
            )
            continue
        if hc and not hc.get("ok", True):
            item = CaseResult(
                name=plan.name,
                ok=False,
                status="fail",
                error=str(hc.get("error") or "harness case failed"),
                seed=plan.seed,
            )
            results.append(item)
            if not failed_case:
                failed_case = plan.name
            continue
        try:
            expected = bind_and_call(
                ref_fn, manifest.abi, arrays_by_case[plan.name], plan.scalars
            )
        except Exception as exc:
            # A reference reshape/index error is a failed case, not a dead worker.
            item = CaseResult(
                name=plan.name,
                ok=False,
                status="fail",
                error=f"reference raised: {exc}",
                seed=plan.seed,
            )
            results.append(item)
            if not failed_case:
                failed_case = plan.name
            continue
        got = load_gpu_outputs(testdir, manifest.abi, plan)
        if not got:
            item = compare_outputs({}, expected, manifest.abi, plan, tols)
            item.ok = False
            item.status = "fail"
            item.error = item.error or "missing GPU outputs"
        else:
            item = compare_outputs(got, expected, manifest.abi, plan, tols)
        results.append(item)
        if not item.ok and not failed_case:
            failed_case = plan.name

    error_class = ""
    status_out = "pass"
    reason_out = ""
    if failed_case:
        status_out = "fail"
        error_class = classify_from_results(results, harness_error=output)
        reason_out = next((r.error for r in results if not r.ok), "numeric mismatch")
    report = RefvalReport(
        status=status_out,
        dialect=dialect,
        cases_run=len(results),
        failed_case=failed_case,
        error_class=error_class,
        tolerances={k: tols.get(k, DEFAULT_TOLERANCES["f32"]) for k in {manifest.abi.dtype, "f32"}},
        manifest_summary=manifest.summary(),
        seed=seed,
        results=results,
        reason=reason_out,
        cases_hash=planned_cases_hash,
        semantic_contract=dict(manifest.semantic_contract or manifest.task_spec),
        backend_contract=dict(manifest.backend_contract or manifest.oracle_spec),
        contract_version=getattr(manifest, "contract_version", ""),
    )
    return _finish(report)


def refval_blocks_save(report: RefvalReport, settings: Settings | None = None) -> bool:
    """True when the graph should treat this like a compile failure."""
    settings = settings or get_settings()
    strict = strict_refval_enabled(settings)
    if report.status == "fail":
        if report.error_class == "reference_error" and not strict:
            return False
        return True
    if report.status in {"skip", "reference_error"}:
        return strict
    return False


def scan_solution_files(root: Path) -> list[tuple[int, str, Path]]:
    """Find ``(question_id, dialect, path)`` for ``solution.cu`` / ``solution.py``."""
    found: list[tuple[int, str, Path]] = []
    if not root.is_dir():
        return found
    for path in sorted(root.rglob("solution.cu")):
        parsed = _parse_work_path(path, root)
        if parsed:
            found.append(parsed)
    for path in sorted(root.rglob("solution.py")):
        parsed = _parse_work_path(path, root)
        if parsed:
            found.append(parsed)
    return found


def _parse_work_path(path: Path, root: Path) -> tuple[int, str, Path] | None:
    parts = path.relative_to(root).parts
    qid = None
    dialect = "cuda"
    for part in parts:
        if part.startswith("q") and part[1:].isdigit():
            qid = int(part[1:])
        elif part in {"cuda", "cutlass", "triton", "tilelang"}:
            dialect = part
    if qid is None:
        return None
    if path.suffix == ".py" and dialect == "cuda":
        dialect = "triton"
    return qid, dialect, path


def run_offline(
    *,
    work_root: Path,
    settings: Settings | None = None,
    limit: int | None = None,
    questions: Mapping[int, str] | None = None,
) -> list[RefvalReport]:
    """Batch-validate existing ``work/**/solution.*`` kernels."""
    from cuda_sft.dialects.agent import get_spec

    settings = settings or get_settings()
    files = scan_solution_files(work_root)
    if limit is not None:
        files = files[: int(limit)]
    reports: list[RefvalReport] = []
    for qid, dialect, path in files:
        try:
            spec = get_spec(dialect)
            dialect_spec = spec.refval_spec(settings)
        except Exception as exc:
            reports.append(RefvalReport.skipped(dialect, f"no refval_spec: {exc}", seed=seed_for(qid, dialect)))
            continue
        extra: list[str] = []
        if dialect == "cutlass":
            extra = list(dialect_spec.extra_includes)
            del extra
        question = (questions or {}).get(qid) or f"(offline q{qid})"
        code = path.read_text(encoding="utf-8", errors="replace")
        workdir = path.parent
        report = run_refval(
            question=question,
            code=code,
            question_id=qid,
            dialect=dialect,
            dialect_spec=dialect_spec,
            settings=settings,
            workdir=workdir,
            nest_dialect=False,
        )
        reports.append(report)
        print(
            f"[offline q{qid} {dialect}] {report.status} "
            f"class={report.error_class or '-'} cases={report.cases_run} "
            f"{report.reason or ''}".rstrip(),
            flush=True,
        )
    return reports
