#!/usr/bin/env python
"""Open Introspection: Quick start script."""

import torch


def main() -> None:
    print("Open Introspection")
    print("=" * 40)
    print()
    print("Available experiments:")
    print("  1. experiments/01_setup_sanity_check.py")
    print("  2. experiments/02_concept_extraction.py")
    print("  3. experiments/03_introspection_test.py")
    print()
    print("Run with: uv run python experiments/01_setup_sanity_check.py")
    print()

    # Quick device check
    if torch.backends.mps.is_available():
        print("Device: Apple Silicon (MPS)")
    elif torch.cuda.is_available():
        print(f"Device: CUDA ({torch.cuda.get_device_name()})")
    else:
        print("Device: CPU")


if __name__ == "__main__":
    main()
