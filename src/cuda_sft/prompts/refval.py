"""Prompts for ABI + CPU-reference extraction (one LLM call per attempt)."""

from __future__ import annotations

EXTRACT_SYSTEM = """You extract a GPU kernel's host ABI and write a correct CPU reference.
The project has NO include/solution_header.h — the host signature was invented by the kernel.
Return ONE JSON object and nothing else with keys:
  "abi": {
    "entry": host function name (not the __global__ name unless they are the same),
    "dtype": canonical compute dtype (f32|f64|f16|bf16|i32|i64|i8|u8|u16|u32|bool),
    "layout": contiguous|row_major|col_major|strided,
    "returns": "void" or a dtype,
    "sort_outputs": true if compare must sort-flatten (atomics / set-union / unordered),
    "allows_nan": true only if the question says NaN is valid output,
    "in_place": true if an output may alias an input,
    "compare_mode": elementwise|sorted|set,
    "params": [
      {"name": "...", "kind": "input|output|inout|scalar|size",
       "dtype": "f32", "rank": 1, "shape_from": ["n"], "memory": "device"}
    ]
  },
  "reference_fn_name": "reference",
  "reference_source": "def reference(...):\\n    ...\\n    return {'out': ...}\\n"
Rules for reference_source:
- Pure Python using numpy as np (already imported). No file I/O, no network, no subprocess.
- Match the host ABI: take input tensors and scalars by name, return a dict of output name -> ndarray.
- Implement the QUESTION's algorithm, not a copy of GPU indexing bugs.
- For reductions, express the mathematical operation with NumPy and suitable accumulation precision; do not simulate thread or block reduction order.
- Deterministic. Do not print.
- For set-union / unordered outputs set sort_outputs=true.
- Device pointers in the kernel correspond to numpy arrays on the CPU; do not use cuda.
- CPU tensor arguments are dense logical arrays, even when the GPU case uses strided storage. A stride scalar describes GPU storage; do not index the CPU array by that stride. Use logical indices (for example a[:n] + b[:n]).
- For unsigned short pointers carrying FP16 bits, use param dtype u16 and compute dtype f16. Interpret inputs with .view(np.float16), then return FP16 outputs viewed as np.uint16.
JSON only."""

EXTRACT_USER = """## Question
{question}

## Dialect
{dialect}

## Host-entry hint
{host_entry_hint}

## Kernel source
```
{code}
```

Return JSON with abi + reference_source. The ABI entry MUST exist in the source."""

RETRY_USER = """The previous ABI/reference JSON failed a self-check:
{issues}

Fix the JSON. The host entry name MUST appear in the kernel source.
Question:
{question}

Source:
```
{code}
```

JSON only."""


def extract_user(*, question: str, code: str, dialect: str, host_entry_hint: str = "") -> str:
    """User message for the first ABI/reference extraction call."""
    return EXTRACT_USER.format(
        question=(question or "").strip() or "(empty)",
        dialect=dialect or "cuda",
        host_entry_hint=(host_entry_hint or "Call the host launcher, not the __global__ kernel.").strip(),
        code=(code or "").strip() or "(empty)",
    )


def retry_user(*, question: str, code: str, issues: list[str] | str) -> str:
    """User message for one ABI self-check retry."""
    if isinstance(issues, (list, tuple)):
        blob = "\n".join(f"- {item}" for item in issues)
    else:
        blob = str(issues)
    return RETRY_USER.format(
        issues=blob or "(unspecified)",
        question=(question or "").strip() or "(empty)",
        code=(code or "").strip() or "(empty)",
    )


def speculative_request_id(question_id: int, dialect: str, candidate: int, repair: int) -> str:
    """Async pool key for a refval extract prefetch (must not collide with repair)."""
    return f"q{int(question_id)}_{dialect or 'cuda'}_c{int(candidate)}_r{int(repair)}_refval"
