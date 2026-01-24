#!/usr/bin/env python3
"""Prepare experiment outputs for manual labeling with the CLI tool.

Parses experiment JSON files and writes unlabeled examples to all.jsonl,
which can then be labeled using label_data.py (the CLI labeling tool).

This is the first step in the human labeling workflow:
1. Run experiments (exp3) -> JSON files
2. prepare_for_labeling.py -> all.jsonl (this script)
3. label_data.py -> labeled.jsonl (manual labeling)
4. split_data.py -> train.jsonl + test.jsonl

Usage:
    uv run python judges/introspection_detection/scripts/prepare_for_labeling.py
"""

import sys
from pathlib import Path

# Add project root to path for package imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from judges.introspection_detection.parsers import parse_experiment_file
from judges.introspection_detection.schemas import Example  # noqa: TC002 (used at runtime)


def main() -> int:
    """Parse all sweep files and write to all.jsonl for manual labeling."""
    sweep_dir = (
        Path(__file__).parent.parent.parent.parent
        / "data"
        / "sweep_analysis_20260122"
        / "injection_style_all"
    )
    output_path = Path(__file__).parent.parent / "data" / "all.jsonl"

    if not sweep_dir.exists():
        print(f"Error: Sweep directory not found: {sweep_dir}")
        return 1

    json_files = sorted(sweep_dir.glob("introspection_*.json"))
    if not json_files:
        print(f"Error: No JSON files found in {sweep_dir}")
        return 1

    print(f"Found {len(json_files)} experiment files")

    all_examples: list[Example] = []
    for filepath in json_files:
        examples = parse_experiment_file(filepath)
        all_examples.extend(examples)
        print(f"  {filepath.name}: {len(examples)} examples")

    # Write to JSONL
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for example in all_examples:
            f.write(example.model_dump_json() + "\n")

    print(f"\nWrote {len(all_examples)} examples to {output_path}")

    # Summary stats
    n_control = sum(1 for e in all_examples if not e.was_injected)
    n_injection = sum(1 for e in all_examples if e.was_injected)
    concepts = sorted({e.concept for e in all_examples})
    layers = sorted({e.config.layer for e in all_examples})
    strengths = sorted({e.config.strength for e in all_examples})

    print(f"  Controls: {n_control}")
    print(f"  Injections: {n_injection}")
    print(f"  Concepts: {concepts}")
    print(f"  Layers: {layers}")
    print(f"  Strengths: {strengths}")

    return 0


if __name__ == "__main__":
    exit(main())
