"""Validate an SFT JSONL file against applicable L6 checks."""

from cuda_sft.observability.dataset_checks import main

if __name__ == "__main__":
    raise SystemExit(main())
