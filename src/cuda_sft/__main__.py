"""``python -m cuda_sft`` entry point."""

import sys

if __name__ == "__main__":
    if "--setup" in sys.argv or "--check" in sys.argv:
        from cuda_sft.setup_env import main as setup_main

        raise SystemExit(setup_main(sys.argv[1:]))
    from cuda_sft.main import main

    raise SystemExit(main())
