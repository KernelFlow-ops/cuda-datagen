"""Frozen contracts for reference-backed kernel validation.

Field names and types in this module are the ABI between cases, harness,
compare, extract, and the graph. Do not add or rename fields without
updating every consumer; extra JSON keys are ignored on load.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

PARAM_KINDS = ("input", "output", "inout", "scalar", "size")
LAYOUTS = ("contiguous", "row_major", "col_major", "strided")
MEMORY_SPACES = ("device", "host")
COMPARE_MODES = ("elementwise", "sorted", "set")
REFVAL_STATUSES = (
    "pass", "fail", "skip", "reference_error", "unavailable", "not_evaluated"
)
REFVAL_SCHEMA_VERSION = "refval-schema-v1"
REFVAL_VALIDATOR_VERSION = "refval-validator-v1"
REFVAL_HARNESS_VERSION = "refval-harness-v1"
REFVAL_CASE_SUITE_VERSION = "refval-cases-v1"
REFVAL_CONTRACT_VERSION = "refval-contract-v1"
ERROR_CLASSES = (
    "numeric_mismatch",
    "nan_inf",
    "crash",
    "timeout",
    "signature_mismatch",
    "reference_error",
    "compile_error",
    "driver_import",
    "entry_missing",
    "argument_binding",
    "shape_contract",
    "stride_mismatch",
    "dtype_mismatch",
    "launch_runtime",
    "cuda_illegal_memory",
    "output_missing",
    "unavailable",
)
CASE_KINDS = (
    "n0",
    "n1",
    "non_tile",
    "non_pow2",
    "large",
    "extreme",
    "dup",
    "strided",
    "broadcast",
    "inplace",
    "tail",
    "row_major_tail",
    "smoke",
)

DTYPE_ALIASES = {
    "float": "f32",
    "float32": "f32",
    "fp32": "f32",
    "f32": "f32",
    "double": "f64",
    "float64": "f64",
    "fp64": "f64",
    "f64": "f64",
    "half": "f16",
    "float16": "f16",
    "fp16": "f16",
    "f16": "f16",
    "__half": "f16",
    "bfloat16": "bf16",
    "bf16": "bf16",
    "nv_bfloat16": "bf16",
    "int": "i32",
    "int32": "i32",
    "int32_t": "i32",
    "i32": "i32",
    "int64": "i64",
    "int64_t": "i64",
    "long": "i64",
    "longlong": "i64",
    "long long": "i64",
    "i64": "i64",
    "size_t": "i64",
    "int8": "i8",
    "int8_t": "i8",
    "signed char": "i8",
    "char": "i8",
    "i8": "i8",
    "uint8": "u8",
    "uint8_t": "u8",
    "unsigned char": "u8",
    "u8": "u8",
    "uint32": "u32",
    "uint32_t": "u32",
    "unsigned": "u32",
    "unsigned int": "u32",
    "u32": "u32",
    "bool": "bool",
}

CPP_DTYPE = {
    "f32": "float",
    "f64": "double",
    "f16": "__half",
    "bf16": "nv_bfloat16",
    "i32": "int",
    "i64": "long long",
    "i8": "signed char",
    "u8": "unsigned char",
    "u32": "unsigned int",
    "bool": "bool",
}

NUMPY_DTYPE = {
    "f32": "float32",
    "f64": "float64",
    "f16": "float16",
    "bf16": "float32",
    "i32": "int32",
    "i64": "int64",
    "i8": "int8",
    "u8": "uint8",
    "u32": "uint32",
    "bool": "bool",
}

DTYPE_NBYTES = {
    "f32": 4,
    "f64": 8,
    "f16": 2,
    "bf16": 2,
    "i32": 4,
    "i64": 8,
    "i8": 1,
    "u8": 1,
    "u32": 4,
    "bool": 1,
}

DEFAULT_TOLERANCES: dict[str, dict[str, float]] = {
    "f32": {"atol": 1.0e-4, "rtol": 1.0e-3},
    "f64": {"atol": 1.0e-8, "rtol": 1.0e-6},
    "f16": {"atol": 1.0e-2, "rtol": 1.0e-2},
    "bf16": {"atol": 2.0e-2, "rtol": 2.0e-2},
    "i32": {"atol": 0.0, "rtol": 0.0},
    "i64": {"atol": 0.0, "rtol": 0.0},
    "i8": {"atol": 0.0, "rtol": 0.0},
    "u8": {"atol": 0.0, "rtol": 0.0},
    "u32": {"atol": 0.0, "rtol": 0.0},
    "bool": {"atol": 0.0, "rtol": 0.0},
}

FLOAT_DTYPES = frozenset({"f32", "f64", "f16", "bf16"})
INT_DTYPES = frozenset({"i32", "i64", "i8", "u8", "u32"})


def normalize_dtype(raw: str | None) -> str:
    """Map a C++/numpy/LLM dtype string to a canonical short name."""
    text = (raw or "f32").strip().lower().replace(" ", "")
    text = text.replace("::", "").replace("std", "")
    return DTYPE_ALIASES.get(text, DTYPE_ALIASES.get((raw or "").strip().lower(), "f32"))


def normalize_kind(raw: str | None) -> str:
    """Map aliases onto ``input|output|inout|scalar|size``."""
    name = (raw or "input").strip().lower()
    aliases = {
        "in": "input",
        "src": "input",
        "out": "output",
        "dst": "output",
        "result": "output",
        "in_out": "inout",
        "in-out": "inout",
        "ptr": "input",
        "tensor": "input",
        "n": "size",
        "dim": "size",
        "shape": "size",
        "len": "size",
        "length": "size",
        "count": "size",
    }
    name = aliases.get(name, name)
    return name if name in PARAM_KINDS else "input"


def normalize_layout(raw: str | None) -> str:
    """Map aliases onto a layout name."""
    name = (raw or "contiguous").strip().lower().replace("-", "_")
    aliases = {
        "c": "contiguous",
        "row": "row_major",
        "rowmajor": "row_major",
        "col": "col_major",
        "column": "col_major",
        "column_major": "col_major",
        "stride": "strided",
    }
    name = aliases.get(name, name)
    return name if name in LAYOUTS else "contiguous"


def normalize_memory(raw: str | None) -> str:
    """Map aliases onto ``device`` or ``host``."""
    name = (raw or "device").strip().lower()
    if name in {"cpu", "host", "h"}:
        return "host"
    return "device"


def normalize_compare_mode(raw: str | None) -> str:
    """Map aliases onto a compare mode."""
    name = (raw or "elementwise").strip().lower()
    aliases = {
        "exact": "elementwise",
        "elem": "elementwise",
        "sort": "sorted",
        "sorted_flat": "sorted",
        "multiset": "sorted",
        "unique": "set",
        "set_union": "sorted",
    }
    name = aliases.get(name, name)
    return name if name in COMPARE_MODES else "elementwise"


def dtype_nbytes(dtype: str) -> int:
    """Return the storage size in bytes for a canonical dtype."""
    return DTYPE_NBYTES.get(normalize_dtype(dtype), 4)


def bf16_quarantine_reason(*, supported: bool | None = None) -> str:
    """Return an explicit quarantine reason for unavailable bfloat16 support."""
    if supported is None:
        supported = str(os.environ.get("REFVAL_BF16_SUPPORTED", "1")).strip().lower() not in {
            "0", "false", "no", "off"
        }
    return "" if supported else "bf16 backend support unavailable; validation quarantined"


def cpp_type(dtype: str, *, pointer: bool = False, const: bool = False) -> str:
    """C++ type token used in harness call sites."""
    base = CPP_DTYPE.get(normalize_dtype(dtype), "float")
    if pointer:
        return f"const {base}*" if const else f"{base}*"
    return base


def numpy_dtype_name(dtype: str) -> str:
    """numpy dtype name for CPU arrays."""
    return NUMPY_DTYPE.get(normalize_dtype(dtype), "float32")


def tolerances_for(dtype: str, overrides: Mapping[str, Any] | None = None) -> dict[str, float]:
    """Return ``{atol, rtol}`` for ``dtype``, applying optional overrides."""
    key = normalize_dtype(dtype)
    base = dict(DEFAULT_TOLERANCES.get(key, DEFAULT_TOLERANCES["f32"]))
    blob = overrides or {}
    if key in blob and isinstance(blob[key], Mapping):
        blob = blob[key]
    if "atol" in blob:
        base["atol"] = float(blob["atol"])
    if "rtol" in blob:
        base["rtol"] = float(blob["rtol"])
    return base


def seed_for(qid: int, dialect: str, extra: str = "") -> int:
    """Stable non-negative 31-bit seed derived from question id and dialect.

    Args:
        qid: 1-based question id.
        dialect: Kernel dialect (``cuda``, ``triton``, ...).
        extra: Optional case name / salt so CPU and GPU share one stream.
    """
    # ``extra`` identifies a semantic case.  Keep legacy dialect-sensitive
    # question seeds when it is empty, while making named case seeds portable
    # across CUDA/CUTLASS/Python backends.
    if extra:
        return case_seed_for(qid, extra)
    raw = f"{int(qid)}|{(dialect or 'cuda').strip().lower()}|{extra}".encode("utf-8")
    digest = hashlib.sha256(raw).digest()
    return int.from_bytes(digest[:8], "little") & 0x7FFFFFFF


def case_seed_for(qid: int, case_name: str, *, salt: str = "") -> int:
    """Return a dialect-independent seed for one named validation case.

    ``seed_for`` remains dialect-sensitive for backwards compatibility with
    old manifests.  Inputs shared by CUDA, CUTLASS and Python backends use
    this helper so changing the backend cannot silently change test data.
    """
    raw = f"{int(qid)}|case|{str(case_name).strip().lower()}|{salt}".encode("utf-8")
    digest = hashlib.sha256(raw).digest()
    return int.from_bytes(digest[:8], "little") & 0x7FFFFFFF


def cases_hash(plans: Iterable[Any]) -> str:
    """Hash the semantic case plan, independent of backend/dialect details."""
    rows = []
    for plan in plans:
        if hasattr(plan, "to_dict"):
            row = plan.to_dict()
        else:
            row = dict(plan)
        # Filesystem paths and generated binaries are intentionally excluded.
        row.pop("files", None)
        row.pop("outputs", None)
        rows.append(row)
    return stable_hash(rows)


def stable_hash(value: Any) -> str:
    """Return a stable SHA-256 digest for JSON-compatible contract metadata."""
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def strict_refval_enabled(settings: Any) -> bool:
    """Resolve strict refval policy while accepting both old and new settings.

    Older configurations expose only ``refval_strict``.  Newer callers may use
    a string policy (``strict``/``required``/``gate``); accepting both keeps
    saved jobs and lightweight test settings interoperable.
    """
    policy = str(getattr(settings, "refval_policy", "") or "").strip().lower()
    if policy:
        return policy in {"strict", "required", "gate", "blocking", "on", "true", "1"}
    return bool(getattr(settings, "refval_strict", False))


def _as_tuple_ints(value: Any) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, (int, float)):
        return (int(value),)
    if isinstance(value, str) and value.strip():
        parts = [p.strip() for p in value.replace("x", ",").split(",") if p.strip()]
        return tuple(int(p) for p in parts)
    return tuple(int(x) for x in value)


def _as_str_tuple(value: Any) -> tuple[str, ...]:
    if not value:
        return ()
    if isinstance(value, str):
        return tuple(p.strip() for p in value.split(",") if p.strip())
    return tuple(str(x) for x in value)


@dataclass(frozen=True)
class KernelParam:
    """One host-entry argument.

    Attributes:
        name: Identifier as it appears in the host function.
        kind: ``input`` / ``output`` / ``inout`` / ``scalar`` / ``size``.
        dtype: Canonical dtype (``f32``, ``i32``, ...).
        layout: Default memory layout.
        rank: Tensor rank; 0 for scalars/sizes.
        shape_from: Size-parameter names that define this tensor's shape.
        memory: ``device`` (default, typical launcher) or ``host``.
        optional: If True, harness may omit the argument.
        alias_of: For in-place outputs, the input name this aliases.
    """

    name: str
    kind: str = "input"
    dtype: str = "f32"
    layout: str = "contiguous"
    rank: int = 1
    shape_from: tuple[str, ...] = ()
    memory: str = "device"
    optional: bool = False
    alias_of: str = ""

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable mapping."""
        return {
            "name": self.name,
            "kind": self.kind,
            "dtype": self.dtype,
            "layout": self.layout,
            "rank": int(self.rank),
            "shape_from": list(self.shape_from),
            "memory": self.memory,
            "optional": bool(self.optional),
            "alias_of": self.alias_of,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "KernelParam":
        """Load from a mapping; unknown keys are ignored."""
        data = dict(raw or {})
        kind = normalize_kind(str(data.get("kind") or "input"))
        rank_default = 0 if kind in {"scalar", "size"} else 1
        rank = int(data.get("rank", rank_default) or 0)
        if kind in {"scalar", "size"}:
            rank = 0
        return cls(
            name=str(data.get("name") or "arg"),
            kind=kind,
            dtype=normalize_dtype(str(data.get("dtype") or "f32")),
            layout=normalize_layout(str(data.get("layout") or "contiguous")),
            rank=max(0, rank),
            shape_from=_as_str_tuple(data.get("shape_from")),
            memory=normalize_memory(str(data.get("memory") or "device")),
            optional=bool(data.get("optional", False)),
            alias_of=str(data.get("alias_of") or ""),
        )

    @property
    def is_tensor(self) -> bool:
        """True when this parameter is a pointer/buffer."""
        return self.kind in {"input", "output", "inout"}

    @property
    def is_output(self) -> bool:
        """True when the harness must read this buffer after the launch."""
        return self.kind in {"output", "inout"}

    @property
    def is_input(self) -> bool:
        """True when the harness must fill this buffer before the launch."""
        return self.kind in {"input", "inout"}


@dataclass(frozen=True)
class KernelABI:
    """Host ABI invented by the kernel (no shared ``solution_header.h``).

    Attributes:
        entry: Host function name.
        params: Ordered host parameters.
        dtype: Primary compute dtype.
        layout: Default tensor layout.
        kernel_name: Optional ``__global__`` name if different from ``entry``.
        returns: ``void`` or a canonical dtype for a scalar return.
        sort_outputs: If True, sort-flatten before compare (atomics / set-union).
        allows_nan: If True, NaN in outputs is not an automatic failure.
        in_place: If True, at least one output may alias an input.
        compare_mode: ``elementwise`` / ``sorted`` / ``set``.
        notes: Free-form extractor notes.
    """

    entry: str
    params: tuple[KernelParam, ...]
    dtype: str = "f32"
    layout: str = "contiguous"
    kernel_name: str = ""
    returns: str = "void"
    sort_outputs: bool = False
    allows_nan: bool = False
    in_place: bool = False
    compare_mode: str = "elementwise"
    notes: str = ""
    supports_strided: bool = False
    supports_broadcast: bool = False
    supports_inplace: bool = False
    zero_size_strategy: str = "skip"
    # Backend call convention for Python dialect adapters.  Legacy manifests
    # omit it and retain positional CUDA/Triton behavior.
    call_style: str = "positional"

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable mapping."""
        return {
            "entry": self.entry,
            "params": [p.to_dict() for p in self.params],
            "dtype": self.dtype,
            "layout": self.layout,
            "kernel_name": self.kernel_name,
            "returns": self.returns,
            "sort_outputs": bool(self.sort_outputs),
            "allows_nan": bool(self.allows_nan),
            "in_place": bool(self.in_place),
            "compare_mode": self.compare_mode,
            "notes": self.notes,
            "supports_strided": bool(self.supports_strided),
            "supports_broadcast": bool(self.supports_broadcast),
            "supports_inplace": bool(self.supports_inplace),
            "zero_size_strategy": self.zero_size_strategy,
            "call_style": self.call_style,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "KernelABI":
        """Load from a mapping; unknown keys are ignored."""
        data = dict(raw or {})
        params_raw = data.get("params") or []
        params = tuple(KernelParam.from_dict(p) for p in params_raw if isinstance(p, Mapping))
        returns = str(data.get("returns") or "void").strip().lower()
        if returns and returns != "void":
            returns = normalize_dtype(returns)
        else:
            returns = "void"
        compare = normalize_compare_mode(str(data.get("compare_mode") or "elementwise"))
        sort_outputs = bool(data.get("sort_outputs", False)) or compare in {"sorted", "set"}
        contract = data.get("contract") if isinstance(data.get("contract"), Mapping) else {}
        def _flag(name: str) -> bool:
            return bool(data.get(name, contract.get(name, False)))
        zero_size = str(data.get("zero_size_strategy", contract.get("zero_size_strategy", "skip")) or "skip").strip().lower()
        if zero_size not in {"skip", "call", "quarantine"}:
            zero_size = "skip"
        return cls(
            entry=str(data.get("entry") or data.get("host") or "launch"),
            params=params,
            dtype=normalize_dtype(str(data.get("dtype") or "f32")),
            layout=normalize_layout(str(data.get("layout") or "contiguous")),
            kernel_name=str(data.get("kernel_name") or ""),
            returns=returns,
            sort_outputs=sort_outputs,
            allows_nan=bool(data.get("allows_nan", False)),
            in_place=bool(data.get("in_place", False)),
            compare_mode=compare,
            notes=str(data.get("notes") or ""),
            supports_strided=_flag("supports_strided") or normalize_layout(str(data.get("layout") or "contiguous")) == "strided",
            supports_broadcast=_flag("supports_broadcast"),
            supports_inplace=_flag("supports_inplace") or bool(data.get("in_place", False)),
            zero_size_strategy=zero_size,
            call_style=str(data.get("call_style", contract.get("call_style", "positional")) or "positional"),
        )

    def tensor_params(self) -> tuple[KernelParam, ...]:
        """Pointer/buffer parameters."""
        return tuple(p for p in self.params if p.is_tensor)

    def size_params(self) -> tuple[KernelParam, ...]:
        """Integer shape/extent parameters."""
        return tuple(p for p in self.params if p.kind == "size")

    def scalar_params(self) -> tuple[KernelParam, ...]:
        """Non-size scalar parameters."""
        return tuple(p for p in self.params if p.kind == "scalar")

    def output_params(self) -> tuple[KernelParam, ...]:
        """Buffers compared after the launch."""
        return tuple(p for p in self.params if p.is_output)

    def input_params(self) -> tuple[KernelParam, ...]:
        """Buffers filled before the launch."""
        return tuple(p for p in self.params if p.is_input)

    def result_names(self) -> tuple[str, ...]:
        """Output buffer names plus ``__return__`` when the host returns a scalar."""
        names = [p.name for p in self.output_params()]
        if self.returns and self.returns != "void":
            names.append("__return__")
        return tuple(names)

    def issues(self) -> list[str]:
        """Return self-check problems; empty means the ABI is usable."""
        problems: list[str] = []
        if not self.entry or not str(self.entry).isidentifier():
            problems.append("invalid entry name")
        if not self.params:
            problems.append("no host parameters")
        names = [p.name for p in self.params]
        if len(names) != len(set(names)):
            problems.append("duplicate parameter names")
        size_names = {p.name for p in self.size_params()}
        for param in self.tensor_params():
            for dim in param.shape_from:
                if dim not in size_names and dim not in names:
                    problems.append(f"{param.name}.shape_from references unknown {dim!r}")
        if not self.output_params() and (not self.returns or self.returns == "void"):
            problems.append("no output buffers and no scalar return")
        return problems


@dataclass(frozen=True)
class RefManifest:
    """ABI plus a CPU reference callable extracted once per attempt.

    Attributes:
        question_id: 1-based jsonl id.
        dialect: Kernel dialect.
        abi: Host ABI.
        reference_source: Python source defining the reference function.
        reference_fn_name: Callable name inside ``reference_source``.
        seed: Base RNG seed for this question/dialect.
        tolerances: Optional per-dtype ``{atol,rtol}`` overrides.
        extracted_from: ``llm`` / ``heuristic`` / ``cache`` / ``injected``.
        notes: Extractor notes.
    """

    question_id: int
    dialect: str
    abi: KernelABI
    reference_source: str
    reference_fn_name: str = "reference"
    seed: int = 0
    tolerances: dict[str, dict[str, float]] = field(default_factory=dict)
    extracted_from: str = "llm"
    notes: str = ""
    # These fields were added after the original manifest ABI.  Keep defaults
    # so old cache files and callers using positional construction remain valid.
    task_spec: dict[str, Any] = field(default_factory=dict)
    oracle_spec: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    manifest_hash: str = ""
    # Structured contracts are additive.  ``task_spec``/``oracle_spec`` are
    # retained as aliases for manifests produced before contract v1.
    semantic_contract: dict[str, Any] = field(default_factory=dict)
    backend_contract: dict[str, Any] = field(default_factory=dict)
    contract_version: str = REFVAL_CONTRACT_VERSION

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable mapping (includes full reference source)."""
        return {
            "question_id": int(self.question_id),
            "dialect": self.dialect,
            "abi": self.abi.to_dict(),
            "reference_source": self.reference_source,
            "reference_fn_name": self.reference_fn_name,
            "seed": int(self.seed),
            "tolerances": self.tolerances,
            "extracted_from": self.extracted_from,
            "notes": self.notes,
            "task_spec": dict(self.task_spec or self.semantic_contract),
            "oracle_spec": dict(self.oracle_spec or self.backend_contract),
            "provenance": dict(self.provenance),
            "manifest_hash": self.manifest_hash or stable_hash({
                "question_id": int(self.question_id),
                "dialect": self.dialect,
                "abi": self.abi.to_dict(),
                "reference_source": self.reference_source,
                "reference_fn_name": self.reference_fn_name,
            }),
            "semantic_contract": dict(self.semantic_contract or self.task_spec),
            "backend_contract": dict(self.backend_contract or self.oracle_spec),
            "contract_version": self.contract_version,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "RefManifest":
        """Load from a mapping; unknown keys are ignored."""
        data = dict(raw or {})
        abi_raw = data.get("abi") if isinstance(data.get("abi"), Mapping) else data
        tols: dict[str, dict[str, float]] = {}
        blob = data.get("tolerances") or {}
        if isinstance(blob, Mapping):
            for key, value in blob.items():
                if isinstance(value, Mapping):
                    tols[normalize_dtype(str(key))] = {
                        "atol": float(value.get("atol", 0.0)),
                        "rtol": float(value.get("rtol", 0.0)),
                    }
        qid = int(data.get("question_id") or data.get("qid") or 0)
        dialect = str(data.get("dialect") or "cuda")
        seed = int(data.get("seed") or 0) or seed_for(qid, dialect)
        return cls(
            question_id=qid,
            dialect=dialect,
            abi=KernelABI.from_dict(abi_raw),  # type: ignore[arg-type]
            reference_source=str(data.get("reference_source") or data.get("reference") or ""),
            reference_fn_name=str(data.get("reference_fn_name") or "reference"),
            seed=seed,
            tolerances=tols,
            extracted_from=str(data.get("extracted_from") or "llm"),
            notes=str(data.get("notes") or ""),
            task_spec=dict(data.get("task_spec") or {}) if isinstance(data.get("task_spec"), Mapping) else {},
            oracle_spec=dict(data.get("oracle_spec") or {}) if isinstance(data.get("oracle_spec"), Mapping) else {},
            provenance=dict(data.get("provenance") or {}) if isinstance(data.get("provenance"), Mapping) else {},
            manifest_hash=str(data.get("manifest_hash") or ""),
            semantic_contract=dict(
                data.get("semantic_contract") or data.get("task_spec") or {}
            ) if isinstance(data.get("semantic_contract") or data.get("task_spec") or {}, Mapping) else {},
            backend_contract=dict(
                data.get("backend_contract") or data.get("oracle_spec") or {}
            ) if isinstance(data.get("backend_contract") or data.get("oracle_spec") or {}, Mapping) else {},
            contract_version=str(data.get("contract_version") or REFVAL_CONTRACT_VERSION),
        )

    def summary(self) -> dict[str, Any]:
        """Compact metadata blob for ``sft.jsonl`` (no reference source)."""
        return {
            "entry": self.abi.entry,
            "dtype": self.abi.dtype,
            "layout": self.abi.layout,
            "params": [p.name for p in self.abi.params],
            "sort_outputs": self.abi.sort_outputs,
            "compare_mode": self.abi.compare_mode,
            "extracted_from": self.extracted_from,
            "allows_nan": self.abi.allows_nan,
            "manifest_hash": self.manifest_hash or self.to_dict().get("manifest_hash", ""),
            "task_spec": dict(self.task_spec or self.semantic_contract),
            "oracle_spec": dict(self.oracle_spec or self.backend_contract),
            "provenance": dict(self.provenance),
            "semantic_contract": dict(self.semantic_contract or self.task_spec),
            "backend_contract": dict(self.backend_contract or self.oracle_spec),
            "contract_version": self.contract_version,
        }


@dataclass(frozen=True)
class CasePlan:
    """One adversarial / random case, CPU and GPU sharing ``seed``.

    Attributes:
        name: Directory-safe case id (``n0``, ``odd_7``, ...).
        kind: Category from :data:`CASE_KINDS`.
        shapes: Tensor name → shape.
        scalars: Size and scalar values (JSON-friendly).
        seed: RNG seed for this case.
        generate_nan: Fill NaNs (only when the ABI allows them).
        generate_inf: Fill infinities.
        generate_denorm: Include denormal floats.
        generate_dup: Repeat values (set-union / histogram).
        strides: Optional per-tensor stride tuples.
        alias: Output name → input name for in-place aliasing.
        large: True when this is the ~1e6–4e6 element case.
    """

    name: str
    kind: str
    shapes: dict[str, tuple[int, ...]]
    scalars: dict[str, Any]
    seed: int
    generate_nan: bool = False
    generate_inf: bool = False
    generate_denorm: bool = False
    generate_dup: bool = False
    strides: dict[str, tuple[int, ...]] = field(default_factory=dict)
    alias: dict[str, str] = field(default_factory=dict)
    large: bool = False
    # Canonical dtype per tensor.  Older rows omit this and derive it from ABI.
    dtypes: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable mapping."""
        return {
            "name": self.name,
            "kind": self.kind,
            "shapes": {k: list(v) for k, v in self.shapes.items()},
            "scalars": dict(self.scalars),
            "seed": int(self.seed),
            "generate_nan": bool(self.generate_nan),
            "generate_inf": bool(self.generate_inf),
            "generate_denorm": bool(self.generate_denorm),
            "generate_dup": bool(self.generate_dup),
            "strides": {k: list(v) for k, v in self.strides.items()},
            "alias": dict(self.alias),
            "large": bool(self.large),
            "dtypes": {k: normalize_dtype(v) for k, v in self.dtypes.items()},
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "CasePlan":
        """Load from a mapping; unknown keys are ignored."""
        data = dict(raw or {})
        shapes = {
            str(k): _as_tuple_ints(v)
            for k, v in (data.get("shapes") or {}).items()
        }
        strides = {
            str(k): _as_tuple_ints(v)
            for k, v in (data.get("strides") or {}).items()
        }
        alias = {str(k): str(v) for k, v in (data.get("alias") or {}).items()}
        return cls(
            name=str(data.get("name") or "case"),
            kind=str(data.get("kind") or "smoke"),
            shapes=shapes,
            scalars=dict(data.get("scalars") or {}),
            seed=int(data.get("seed") or 0),
            generate_nan=bool(data.get("generate_nan", False)),
            generate_inf=bool(data.get("generate_inf", False)),
            generate_denorm=bool(data.get("generate_denorm", False)),
            generate_dup=bool(data.get("generate_dup", False)),
            strides=strides,
            alias=alias,
            large=bool(data.get("large", False)),
            dtypes={str(k): normalize_dtype(str(v)) for k, v in (data.get("dtypes") or {}).items()},
        )


@dataclass
class CaseResult:
    """Numeric (or crash) outcome of one case."""

    name: str
    ok: bool
    status: str = "pass"
    max_abs: float = 0.0
    max_rel: float = 0.0
    n_mismatch: int = 0
    n_compared: int = 0
    shape: tuple[int, ...] = ()
    seed: int = 0
    error: str = ""
    mismatches: list[dict[str, Any]] = field(default_factory=list)
    elapsed_sec: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable mapping."""
        return {
            "name": self.name,
            "ok": bool(self.ok),
            "status": self.status,
            "max_abs": float(self.max_abs),
            "max_rel": float(self.max_rel),
            "n_mismatch": int(self.n_mismatch),
            "n_compared": int(self.n_compared),
            "shape": list(self.shape),
            "seed": int(self.seed),
            "error": self.error,
            "mismatches": list(self.mismatches),
            "elapsed_sec": float(self.elapsed_sec),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "CaseResult":
        """Load from a mapping; unknown keys are ignored."""
        data = dict(raw or {})
        return cls(
            name=str(data.get("name") or "case"),
            ok=bool(data.get("ok", False)),
            status=str(data.get("status") or ("pass" if data.get("ok") else "fail")),
            max_abs=float(data.get("max_abs") or 0.0),
            max_rel=float(data.get("max_rel") or 0.0),
            n_mismatch=int(data.get("n_mismatch") or 0),
            n_compared=int(data.get("n_compared") or 0),
            shape=_as_tuple_ints(data.get("shape")),
            seed=int(data.get("seed") or 0),
            error=str(data.get("error") or ""),
            mismatches=list(data.get("mismatches") or []),
            elapsed_sec=float(data.get("elapsed_sec") or 0.0),
        )


@dataclass
class RefvalReport:
    """Pipeline-level outcome written to ``metadata.refval`` and ``refval.log``.

    Attributes:
        status: ``pass`` / ``fail`` / ``skip`` / ``reference_error``.
        dialect: Kernel dialect.
        cases_run: Number of cases actually compared (or launched).
        failed_case: First failing case name.
        error_class: One of :data:`ERROR_CLASSES`, or empty.
        tolerances: Effective atol/rtol map.
        manifest_summary: Compact ABI summary (no reference source).
        seed: Base seed.
        results: Per-case results.
        reason: Human-readable skip/fail reason.
        elapsed_sec: Wall time including GPU-lock wait.
        evidence: Compressed repair hint.
    """

    status: str
    dialect: str
    cases_run: int = 0
    failed_case: str = ""
    error_class: str = ""
    tolerances: dict[str, Any] = field(default_factory=dict)
    manifest_summary: dict[str, Any] = field(default_factory=dict)
    seed: int = 0
    results: list[CaseResult] = field(default_factory=list)
    reason: str = ""
    elapsed_sec: float = 0.0
    evidence: str = ""
    cache_hash: str = ""
    manifest_hash: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)
    task_spec: dict[str, Any] = field(default_factory=dict)
    oracle_spec: dict[str, Any] = field(default_factory=dict)
    validator_version: str = REFVAL_VALIDATOR_VERSION
    harness_version: str = REFVAL_HARNESS_VERSION
    schema_version: str = REFVAL_SCHEMA_VERSION
    case_suite: str = ""
    case_suite_version: str = REFVAL_CASE_SUITE_VERSION
    cases_hash: str = ""
    semantic_contract: dict[str, Any] = field(default_factory=dict)
    backend_contract: dict[str, Any] = field(default_factory=dict)
    contract_version: str = REFVAL_CONTRACT_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Full JSON-serializable mapping."""
        return {
            "status": self.status,
            "dialect": self.dialect,
            "cases_run": int(self.cases_run),
            "failed_case": self.failed_case,
            "error_class": self.error_class,
            "tolerances": self.tolerances,
            "manifest_summary": self.manifest_summary,
            "seed": int(self.seed),
            "results": [item.to_dict() for item in self.results],
            "reason": self.reason,
            "elapsed_sec": float(self.elapsed_sec),
            "evidence": self.evidence,
            "cache_hash": self.cache_hash,
            "manifest_hash": self.manifest_hash,
            "provenance": self.provenance,
            "task_spec": self.task_spec or self.semantic_contract,
            "oracle_spec": self.oracle_spec or self.backend_contract,
            "validator_version": self.validator_version,
            "harness_version": self.harness_version,
            "schema_version": self.schema_version,
            "case_suite": self.case_suite,
            "case_suite_version": self.case_suite_version,
            "cases_hash": self.cases_hash,
            "semantic_contract": self.semantic_contract,
            "backend_contract": self.backend_contract,
            "contract_version": self.contract_version,
        }

    def to_metadata(self) -> dict[str, Any]:
        """Subset persisted on ``sft.jsonl`` ``metadata.refval``."""
        return {
            "status": self.status,
            "dialect": self.dialect,
            "cases_run": int(self.cases_run),
            "failed_case": self.failed_case,
            "tolerances": self.tolerances,
            "manifest_summary": self.manifest_summary,
            "seed": int(self.seed),
            "error_class": self.error_class,
            "reason": (self.reason or "")[:800],
            "cache_hash": self.cache_hash,
            "manifest_hash": self.manifest_hash,
            "provenance": self.provenance,
            "task_spec": self.task_spec or self.semantic_contract,
            "oracle_spec": self.oracle_spec or self.backend_contract,
            "validator_version": self.validator_version,
            "harness_version": self.harness_version,
            "schema_version": self.schema_version,
            "case_suite": self.case_suite,
            "case_suite_version": self.case_suite_version,
            "cases_hash": self.cases_hash,
            "semantic_contract": self.semantic_contract,
            "backend_contract": self.backend_contract,
            "contract_version": self.contract_version,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "RefvalReport":
        """Load from a mapping; unknown keys are ignored."""
        data = dict(raw or {})
        results = [
            CaseResult.from_dict(item)
            for item in (data.get("results") or [])
            if isinstance(item, Mapping)
        ]
        return cls(
            status=str(data.get("status") or "skip"),
            dialect=str(data.get("dialect") or "cuda"),
            cases_run=int(data.get("cases_run") or 0),
            failed_case=str(data.get("failed_case") or ""),
            error_class=str(data.get("error_class") or ""),
            tolerances=dict(data.get("tolerances") or {}),
            manifest_summary=dict(data.get("manifest_summary") or {}),
            seed=int(data.get("seed") or 0),
            results=results,
            reason=str(data.get("reason") or ""),
            elapsed_sec=float(data.get("elapsed_sec") or 0.0),
            evidence=str(data.get("evidence") or ""),
            cache_hash=str(data.get("cache_hash") or ""),
            manifest_hash=str(data.get("manifest_hash") or ""),
            provenance=dict(data.get("provenance") or {}) if isinstance(data.get("provenance"), Mapping) else {},
            task_spec=dict(data.get("task_spec") or {}) if isinstance(data.get("task_spec"), Mapping) else {},
            oracle_spec=dict(data.get("oracle_spec") or {}) if isinstance(data.get("oracle_spec"), Mapping) else {},
            validator_version=str(data.get("validator_version") or REFVAL_VALIDATOR_VERSION),
            harness_version=str(data.get("harness_version") or REFVAL_HARNESS_VERSION),
            schema_version=str(data.get("schema_version") or REFVAL_SCHEMA_VERSION),
            case_suite=str(data.get("case_suite") or ""),
            case_suite_version=str(data.get("case_suite_version") or REFVAL_CASE_SUITE_VERSION),
            cases_hash=str(data.get("cases_hash") or ""),
            semantic_contract=dict(data.get("semantic_contract") or {}) if isinstance(data.get("semantic_contract"), Mapping) else {},
            backend_contract=dict(data.get("backend_contract") or {}) if isinstance(data.get("backend_contract"), Mapping) else {},
            contract_version=str(data.get("contract_version") or REFVAL_CONTRACT_VERSION),
        )

    @classmethod
    def skipped(cls, dialect: str, reason: str, *, seed: int = 0) -> "RefvalReport":
        """Toolchain-missing / disabled report (does not block save)."""
        return cls(status="skip", dialect=dialect or "cuda", reason=reason, seed=seed)

    @classmethod
    def reference_error(cls, dialect: str, reason: str, *, seed: int = 0) -> "RefvalReport":
        """Reference extraction/exec failure (non-strict: do not fail the sample)."""
        return cls(
            status="reference_error",
            dialect=dialect or "cuda",
            error_class="reference_error",
            reason=reason,
            seed=seed,
        )


@dataclass(frozen=True)
class DialectRefvalSpec:
    """Per-dialect execution recipe from ``DialectSpec.refval_spec()``.

    Attributes:
        dialect: Canonical dialect id.
        language: ``cuda-cpp`` or ``python``.
        runner: ``nvcc_link`` or ``python_import``.
        source_filename: ``solution.cu`` / ``solution.py``.
        extra_includes: Extra ``-I`` paths (CUTLASS).
        cxx_std: ``-std=`` for nvcc.
        needs_nvcc: CUDA C++ path requires nvcc.
        needs_torch: Python path requires torch (+ CUDA).
        timeout_sec: Dialect-level subprocess cap (runner still applies the global cap).
        host_entry_hint: Prompt hint for the extractor.
    """

    dialect: str
    language: str
    runner: str
    source_filename: str
    extra_includes: tuple[str, ...] = ()
    cxx_std: str = "c++17"
    needs_nvcc: bool = False
    needs_torch: bool = False
    timeout_sec: int = 45
    host_entry_hint: str = ""

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable mapping."""
        return {
            "dialect": self.dialect,
            "language": self.language,
            "runner": self.runner,
            "source_filename": self.source_filename,
            "extra_includes": list(self.extra_includes),
            "cxx_std": self.cxx_std,
            "needs_nvcc": bool(self.needs_nvcc),
            "needs_torch": bool(self.needs_torch),
            "timeout_sec": int(self.timeout_sec),
            "host_entry_hint": self.host_entry_hint,
        }


def dumps(obj: Any) -> str:
    """Serialize a refval dataclass (or mapping) to JSON text."""
    if hasattr(obj, "to_dict"):
        payload = obj.to_dict()
    else:
        payload = obj
    return json.dumps(payload, ensure_ascii=False, indent=2)


def loads_manifest(text: str) -> RefManifest:
    """Parse a :class:`RefManifest` from JSON text."""
    return RefManifest.from_dict(json.loads(text))


def loads_report(text: str) -> RefvalReport:
    """Parse a :class:`RefvalReport` from JSON text."""
    return RefvalReport.from_dict(json.loads(text))


def iter_error_classes() -> Iterable[str]:
    """Yield known refval error class labels."""
    return ERROR_CLASSES
