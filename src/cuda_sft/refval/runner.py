"""Orchestrate ABI extract, reference validation, harness build, and GPU run."""

from __future__ import annotations

import json
import logging
import math
import re
import shutil
import time
from collections.abc import Mapping
from concurrent.futures import TimeoutError as FuturesTimeoutError
from contextlib import suppress
from dataclasses import replace
from pathlib import Path
from typing import Any

from cuda_sft.config import Settings, get_settings
from cuda_sft.refval.cases import (
    build_case_plans,
    case_plan_hash,
    materialize_arrays,
    numpy_available,
)
from cuda_sft.refval.compare import (
    classify_from_results,
    compare_outputs,
    dump_artifacts,
    load_gpu_outputs,
)
from cuda_sft.refval.evidence import compress_evidence
from cuda_sft.refval.harness import (
    compile_cuda_harness,
    has_main,
    prepare_testdir,
    run_binary,
)
from cuda_sft.refval.oracle import resolve_oracle_manifest
from cuda_sft.refval.reference import (
    bind_and_call,
    load_reference_fn,
    validate_reference_fn,
)
from cuda_sft.refval.spec import (
    DEFAULT_TOLERANCES,
    COMPLEX_DTYPES,
    FLOAT_DTYPES,
    REFVAL_CASE_SUITE_VERSION,
    REFVAL_HARNESS_VERSION,
    REFVAL_SCHEMA_VERSION,
    REFVAL_VALIDATOR_VERSION,
    CaseResult,
    DialectRefvalSpec,
    RefManifest,
    RefvalReport,
    normalize_dtype,
    seed_for,
    stable_hash,
    strict_refval_enabled,
    tolerances_for,
)
from cuda_sft.runtime import trace
from cuda_sft.runtime.limits import stage_lock
from cuda_sft.runtime.meta import CallMeta, legacy_job_key

logger = logging.getLogger(__name__)

REASON_TIMEOUT_BEFORE_GPU = "timeout before GPU run"
REASON_GPU_LOCK_TIMEOUT = "timeout waiting for GPU lock"
REASON_EXTRACT_FAILED = "ABI/reference extraction failed"
REASON_PREPARE_HARNESS = "failed to prepare harness"


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
        "arch": str(
            getattr(settings, "resolved_cuda_arch", "") or getattr(settings, "cuda_arch", "") or ""
        ),
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
        if not nvcc or (not shutil.which(str(nvcc)) and not Path(str(nvcc)).is_file()):
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
        waiting_since = time.monotonic()
        deadline = time.monotonic() + self.timeout_sec
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._handle = handle
                wait_s = time.monotonic() - waiting_since
                if wait_s >= 0.025:
                    trace.emit("stage.wait", stage="gpu", elapsed_s=wait_s)
                return True
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    handle.close()
                    trace.emit("stage.wait", stage="gpu", elapsed_s=time.monotonic() - waiting_since, timed_out=True)
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

    def __enter__(self) -> GpuFileLock:
        if not self.acquire():
            raise TimeoutError(f"GPU lock wait exceeded {self.timeout_sec:.1f}s")
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


def _llm_complete(user: str, system: str, settings: Settings, meta: CallMeta | None = None) -> str:
    from cuda_sft.llm import get_llm_client

    client = get_llm_client(settings, role="refval_extract")
    stream_completion = getattr(client, "stream_completion", None)
    if callable(stream_completion):
        completion = stream_completion(
            messages=[{"role": "user", "content": user}],
            system=system,
            temperature=0.0,
            print_stream=False,
            thinking_level=settings.for_role("refval_extract").thinking_level,
            max_output_tokens=8192,
            meta=meta,
        )
        return completion.text or completion.reasoning or ""
    return client.stream_text(
        messages=[{"role": "user", "content": user}],
        system=system,
        temperature=0.0,
        print_stream=False,
        meta=meta,
    )


def _try_speculative(request_id: str | None, settings: Settings) -> str | None:
    if not request_id or not getattr(settings, "async_llm_enabled", True):
        return ""
    try:
        from cuda_sft.llm_async import get_async_pool

        pool = get_async_pool(settings.async_llm_max_workers)
        remaining = max(1.0, float(getattr(settings, "refval_extract_timeout_sec", 180)) * 0.5)
        completion = pool.get(request_id, timeout_sec=remaining)
        if completion is None:
            return ""
        return completion.text or completion.reasoning or ""
    except (TimeoutError, FuturesTimeoutError):
        logger.warning("refval prefetched extract timed out: %s", request_id)
        return None
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
    meta: CallMeta | None = None,
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
    if last_text is None:
        return None
    manifest = _parse(last_text)
    need_llm = manifest is None or bool(abi_matches_source(manifest.abi, code))
    if need_llm and not speculative_id:
        user = extract_user(
            question=question,
            code=code,
            dialect=dialect,
            host_entry_hint=(dialect_spec.host_entry_hint if dialect_spec else ""),
        )
        try:
            text = _llm_complete(user, EXTRACT_SYSTEM, settings, meta=meta)
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
                meta=replace(meta, purpose="extract_retry") if meta else None,
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
        llm_options={
            "thinking_level": settings.for_role("refval_extract").thinking_level,
            "max_output_tokens": 8192,
        },
        meta=CallMeta(
            role="refval_extract",
            job_key=legacy_job_key(question_id, dialect),
            question_id=question_id,
            track=dialect,
            candidate=candidate,
            repair=repair,
            purpose="speculative",
        ),
    )
    return request_id if ok else ""


_QUESTION_TOLERANCE_RE = re.compile(
    r"\b(atol|rtol)\s*[:=]\s*((?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)",
    re.IGNORECASE,
)


def _effective_tolerances(
    manifest: RefManifest, question: str = ""
) -> dict[str, dict[str, float]]:
    out = {k: dict(v) for k, v in DEFAULT_TOLERANCES.items()}
    for key, value in (manifest.tolerances or {}).items():
        out[key] = tolerances_for(key, value)
    requested: dict[str, float] = {}
    for match in _QUESTION_TOLERANCE_RE.finditer(question):
        value = float(match.group(2))
        if math.isfinite(value) and value >= 0:
            name = match.group(1).lower()
            requested[name] = min(requested.get(name, value), value)
    dtypes = {normalize_dtype(manifest.abi.dtype)}
    dtypes.update(normalize_dtype(param.dtype) for param in manifest.abi.output_params())
    for dtype in dtypes & (FLOAT_DTYPES | COMPLEX_DTYPES):
        current = out.setdefault(dtype, tolerances_for(dtype))
        for name, value in requested.items():
            current[name] = min(current[name], value)
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
    oracle_manifests: Mapping[str, Any] | None = None,
    task_spec: Mapping[str, Any] | None = None,
    oracle_spec: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
    case_plans: list[Any] | None = None,
    meta: CallMeta | None = None,
) -> RefvalReport:
    """Run the full numeric gate. Never raises on toolchain/LLM failure."""
    settings = settings or get_settings()
    started = time.monotonic()
    dialect = dialect or dialect_spec.dialect or "cuda"
    extract_budget = float(getattr(settings, "refval_extract_timeout_sec", 180) or 180)
    build_budget = float(getattr(settings, "refval_build_timeout_sec", 120) or 120)
    run_budget = float(getattr(settings, "refval_run_timeout_sec", 45) or 45)
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
            "arch": str(
                getattr(settings, "resolved_cuda_arch", "")
                or getattr(settings, "cuda_arch", "")
                or ""
            ),
            "case_suite": case_suite,
        }
        report.provenance = {**runtime_provenance, **report.provenance}
        if manifest is not None:
            source = manifest.extracted_from
            report.oracle_origin = (
                "independent" if source in {"independent", "injected"}
                else "model_extracted" if source in {"llm", "cache"}
                else source
            )
            if report.status == "pass":
                report.verification_tier = (
                    "independent" if report.oracle_origin == "independent"
                    else "model_consistency"
                )
            manifest_payload = manifest.to_dict()
            report.manifest_hash = report.manifest_hash or str(
                manifest_payload.get("manifest_hash") or stable_hash(manifest_payload)
            )
            report.task_spec = report.task_spec or dict(manifest.task_spec)
            report.oracle_spec = report.oracle_spec or dict(manifest.oracle_spec)
            report.provenance = {**manifest.provenance, **report.provenance}
            report.semantic_contract = report.semantic_contract or dict(
                manifest.semantic_contract or manifest.task_spec
            )
            report.backend_contract = report.backend_contract or dict(
                manifest.backend_contract or manifest.oracle_spec
            )
            report.contract_version = report.contract_version or getattr(
                manifest, "contract_version", ""
            )
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

    if oracle_manifests is not None and manifest is None:
        try:
            manifest = resolve_oracle_manifest(
                {"oracle_manifests": oracle_manifests},
                dialect=dialect,
                question_id=int(question_id),
                code=code,
            )
        except ValueError as exc:
            report = RefvalReport(
                status="fail", dialect=dialect, error_class="invalid_oracle",
                reason=str(exc), seed=seed,
            )
            report.oracle_origin = "independent"
            return _finish(report)

    extract_started = time.monotonic()
    extract_meta = meta or CallMeta(
        role="refval_extract",
        job_key=legacy_job_key(int(question_id), dialect),
        question_id=int(question_id),
        track=dialect,
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
            meta=extract_meta,
        )
    except Exception as exc:
        logger.warning("refval obtain_manifest failed: %s", exc)
        manifest = None

    if time.monotonic() - extract_started > extract_budget:
        return _finish(
            RefvalReport(
                status="fail",
                dialect=dialect,
                error_class="timeout",
                reason=REASON_TIMEOUT_BEFORE_GPU,
                seed=seed,
            )
        )
    if manifest is None:
        report = RefvalReport.reference_error(dialect, REASON_EXTRACT_FAILED, seed=seed)
        if getattr(settings, "refval_strict", False):
            report.status = "fail"
        return _finish(report)

    # The graph freezes these fields before generation.  Attach them to the
    # extracted manifest so the LLM extractor cannot silently change the
    # contract used for downstream provenance and cache validation.
    if task_spec or oracle_spec or provenance:
        if manifest.extracted_from in {"independent", "injected"}:
            manifest = replace(
                manifest,
                provenance={**dict(provenance or {}), **manifest.provenance},
            )
        else:
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

    def _check_reference(candidate: RefManifest):
        try:
            fn = load_reference_fn(candidate.reference_source, candidate.reference_fn_name)
            return fn, validate_reference_fn(fn, candidate.abi, seed=candidate.seed or seed)
        except Exception as exc:
            return None, str(exc)

    ref_fn, ref_issue = _check_reference(manifest)
    if ref_issue and manifest.extracted_from not in {"independent", "injected"}:
        from cuda_sft.parse import abi_matches_source, parse_refval_manifest
        from cuda_sft.prompts.refval import EXTRACT_SYSTEM, retry_user

        try:
            repaired_text = _llm_complete(
                retry_user(
                    question=question, code=code,
                    issues=[
                        ref_issue,
                        "CPU reference tensor arguments are dense logical arrays; do not apply GPU storage strides to them.",
                    ],
                ),
                EXTRACT_SYSTEM, settings,
                meta=replace(extract_meta, purpose="reference_retry"),
            )
            repaired = parse_refval_manifest(
                repaired_text, question_id=question_id, dialect=dialect,
                seed=seed, extracted_from="llm",
            )
            if repaired is not None and not abi_matches_source(repaired.abi, code):
                trial_fn, trial_issue = _check_reference(repaired)
                if trial_issue is None:
                    manifest = replace(
                        repaired, task_spec=manifest.task_spec,
                        oracle_spec=manifest.oracle_spec,
                        provenance=manifest.provenance,
                    )
                    ref_fn, ref_issue = trial_fn, None
                else:
                    ref_issue = trial_issue
        except Exception as exc:
            logger.warning("refval reference repair failed: %s", exc)
    if time.monotonic() - extract_started > extract_budget:
        return _finish(RefvalReport(
            status="fail", dialect=dialect, error_class="timeout",
            reason=REASON_TIMEOUT_BEFORE_GPU, seed=seed,
        ))
    if ref_issue:
        report = RefvalReport.reference_error(dialect, ref_issue, seed=seed)
        report.manifest_summary = manifest.summary()
        if getattr(settings, "refval_strict", False):
            report.status = "fail"
        return _finish(report)
    assert ref_fn is not None
    if getattr(settings, "refval_cache", True) and manifest.extracted_from == "llm":
        with suppress(OSError):
            _store_cache(_cache_path(settings, question, code, dialect), manifest)

    build_started = time.monotonic()

    suite = str(getattr(settings, "refval_cases", "standard") or "standard")
    max_elements = int(getattr(settings, "refval_max_elements", 4_000_000) or 4_000_000)
    plans = (
        list(case_plans)
        if case_plans is not None
        else build_case_plans(
            manifest.abi,
            question_id=int(question_id),
            dialect=dialect,
            suite=suite,
            max_elements=max_elements,
        )
    )
    planned_cases_hash = case_plan_hash(plans)
    # Keep the case identity in every subsequent report and artifact. This is
    # the cross-dialect invariant; a dialect must never silently regenerate a
    # different input suite.
    testdir.mkdir(parents=True, exist_ok=True)
    arrays_by_case = {plan.name: materialize_arrays(plan, manifest.abi) for plan in plans}
    try:
        prepare_testdir(
            testdir,
            source=code,
            filename=dialect_spec.source_filename,
            abi=manifest.abi,
            plans=plans,
            dialect_spec=dialect_spec,
            arrays_by_case=arrays_by_case,
        )
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
                reason=f"{REASON_PREPARE_HARNESS}: {exc}",
                manifest_summary=manifest.summary(),
                seed=seed,
            )
        )

    remaining_build = build_budget - (time.monotonic() - build_started)
    if remaining_build <= (5 if dialect_spec.runner != "python_import" else 0):
        return _finish(
            RefvalReport(
                status="fail",
                dialect=dialect,
                error_class="timeout",
                reason=REASON_TIMEOUT_BEFORE_GPU,
                manifest_summary=manifest.summary(),
                seed=seed,
            )
        )

    compile_log = ""
    binary = testdir / "driver.py"
    if dialect_spec.runner != "python_import":
        try:
            with stage_lock("compile"):
                ok, compile_log, compiled = compile_cuda_harness(
                    testdir,
                    settings=settings,
                    extra_includes=list(dialect_spec.extra_includes),
                    std=dialect_spec.cxx_std,
                    used_rdc=used_rdc,
                    timeout_sec=int(
                        min(remaining_build, float(getattr(settings, "nvcc_timeout_sec", 60)))
                    ),
                )
        except Exception as exc:
            return _finish(
                RefvalReport(
                    status="fail",
                    dialect=dialect,
                    error_class="crash",
                    reason=f"{REASON_PREPARE_HARNESS}: {exc}",
                    manifest_summary=manifest.summary(),
                    seed=seed,
                )
            )
        if time.monotonic() - build_started > build_budget:
            return _finish(
                RefvalReport(
                    status="fail",
                    dialect=dialect,
                    error_class="timeout",
                    reason=REASON_TIMEOUT_BEFORE_GPU,
                    manifest_summary=manifest.summary(),
                    seed=seed,
                )
            )
        if not ok or compiled is None:
            return _finish(
                RefvalReport(
                    status="fail",
                    dialect=dialect,
                    error_class="signature_mismatch",
                    reason=compile_log,
                    manifest_summary=manifest.summary(),
                    seed=seed,
                )
            )
        binary = compiled

    run_started = time.monotonic()
    lock = GpuFileLock(settings.work_path / ".refval_gpu.lock", run_budget)
    try:
        acquired = lock.acquire()
    except Exception as exc:
        logger.warning("GPU lock acquisition failed: %s", exc)
        acquired = False
    if not acquired:
        return _finish(
            RefvalReport(
                status="fail",
                dialect=dialect,
                error_class="timeout",
                reason=REASON_GPU_LOCK_TIMEOUT,
                manifest_summary=manifest.summary(),
                seed=seed,
            )
        )
    try:
        remaining_run = run_budget - (time.monotonic() - run_started)
        if remaining_run <= 1:
            return _finish(
                RefvalReport(
                    status="fail",
                    dialect=dialect,
                    error_class="timeout",
                    reason=REASON_GPU_LOCK_TIMEOUT,
                    manifest_summary=manifest.summary(),
                    seed=seed,
                )
            )
        code_exit, run_output, payload = run_binary(
            binary, testdir, timeout_sec=max(1, int(remaining_run))
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
        lock.release()

    if dialect_spec.runner != "python_import":
        if not payload:
            payload = {"ok": False, "error": run_output}
        if code_exit != 0 and "error_class" not in payload:
            lowered = (run_output or "").lower()
            if "timeout" in lowered:
                payload["error_class"] = "timeout"
            elif "error:" in lowered:
                payload["error_class"] = "crash"
        payload.setdefault("compile_log", compile_log)
        output = (compile_log + "\n" + run_output).strip()
    else:
        output = run_output

    tols = _effective_tolerances(manifest, question)
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
        "signature_mismatch",
        "shape_contract",
        "stride_mismatch",
        "dtype_mismatch",
        "argument_binding",
        "driver_import",
        "entry_missing",
        "launch_runtime",
        "cuda_illegal_memory",
        "output_missing",
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
    if code_exit != 0:
        return _finish(
            RefvalReport(
                status="fail",
                dialect=dialect,
                error_class=payload_class or "crash",
                reason=(payload or {}).get("error") or output or f"harness exit {code_exit}",
                manifest_summary=manifest.summary(),
                tolerances=tols,
                seed=seed,
            )
        )

    raw_cases = (payload or {}).get("cases")
    planned_names = [plan.name for plan in plans]
    if (
        (payload or {}).get("ok") is not True
        or not isinstance(raw_cases, list)
        or len(raw_cases) != len(planned_names)
        or any(not isinstance(item, dict) or not isinstance(item.get("name"), str) for item in raw_cases)
        or {item["name"] for item in raw_cases} != set(planned_names)
    ):
        return _finish(RefvalReport(
            status="fail", dialect=dialect, error_class="output_missing",
            reason="harness did not report every planned case exactly once",
            manifest_summary=manifest.summary(), tolerances=tols, seed=seed,
        ))

    results = []
    failed_case = ""
    harness_cases = {
        str(item.get("name")): item
        for item in raw_cases
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
        if hc and hc.get("ok") is not True:
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
            expected = bind_and_call(ref_fn, manifest.abi, arrays_by_case[plan.name], plan.scalars)
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
        return report.error_class != "reference_error" or strict
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
            reports.append(
                RefvalReport.skipped(dialect, f"no refval_spec: {exc}", seed=seed_for(qid, dialect))
            )
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
