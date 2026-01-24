#!/usr/bin/env python3
"""Run the LLM judge on experiment 3 (introspection) outputs.

Parses experiment JSON files, runs the calibrated judge on each example,
and writes results with judge labels. Use this for automated evaluation
of new experiment runs.

This is the automated evaluation workflow (alternative to manual labeling):
1. Run experiments (exp3) -> JSON files
2. judge_exp3_outputs.py -> judged_*.jsonl (this script)
3. Analyze results

Usage:
    # Process specific files (from project root)
    uv run python judges/introspection_detection/scripts/judge_exp3_outputs.py \\
        data/sweep_analysis_20260122/injection_style_all/introspection_*.json

    # Process with more parallelism
    uv run python judges/introspection_detection/scripts/judge_exp3_outputs.py \\
        data/introspection_*.json --workers 8

    # Custom output path
    uv run python judges/introspection_detection/scripts/judge_exp3_outputs.py \\
        data/introspection_*.json --output data/my_results.jsonl
"""

import argparse
import sys
from datetime import datetime
from glob import glob
from pathlib import Path

# Add project root to path for package imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from judges.introspection_detection import judge_examples, load_fewshot_examples
from judges.introspection_detection.parsers import parse_experiment_file
from judges.introspection_detection.schemas import Example, JudgeResult, Label


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run judge on exp3 (introspection) outputs"
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="JSON files to process (supports glob patterns)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSONL file (default: data/judged_TIMESTAMP.jsonl)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-nano",
        help="Judge model (default: gpt-5-nano)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel workers for judge API calls (default: 8)",
    )

    args = parser.parse_args()

    # Expand glob patterns
    filepaths: list[Path] = []
    for pattern in args.files:
        matches = glob(pattern)
        if not matches:
            print(f"Warning: No files match pattern: {pattern}")
        filepaths.extend(Path(m) for m in matches)

    if not filepaths:
        print("Error: No input files found")
        return 1

    filepaths = sorted(set(filepaths))  # Dedupe and sort
    print(f"Found {len(filepaths)} experiment files")

    # Parse all files into examples
    all_examples: list[Example] = []
    for filepath in filepaths:
        try:
            examples = parse_experiment_file(filepath)
            all_examples.extend(examples)
            print(f"  {filepath.name}: {len(examples)} examples")
        except Exception as e:
            print(f"  {filepath.name}: Error - {e}")

    if not all_examples:
        print("Error: No examples parsed")
        return 1

    # Summary of what we're processing
    inject_styles = sorted({e.config.inject_style or "legacy" for e in all_examples})
    layers = sorted({e.config.layer for e in all_examples})
    strengths = sorted({e.config.strength for e in all_examples})
    n_injection = sum(1 for e in all_examples if e.was_injected)
    n_control = len(all_examples) - n_injection

    print(f"\nTotal: {len(all_examples)} examples")
    print(f"  Injection: {n_injection}, Control: {n_control}")
    print(f"  Inject styles: {inject_styles}")
    print(f"  Layers: {layers}")
    print(f"  Strengths: {strengths}")

    # Load few-shot examples
    data_dir = Path(__file__).parent.parent / "data"
    train_path = data_dir / "train.jsonl"

    if not train_path.exists():
        print(f"\nError: Training data not found: {train_path}")
        return 1

    fewshot_examples = load_fewshot_examples(train_path)
    print(f"\nLoaded {len(fewshot_examples)} few-shot examples")

    # Run judge
    print(f"Running judge (model: {args.model}, workers: {args.workers})...")
    results: list[tuple[Example, JudgeResult]] = judge_examples(
        all_examples,
        fewshot_examples,
        model=args.model,
        verbose=args.verbose,
        max_workers=args.workers,
    )

    # Determine output path (default: project's data/ dir, not judges/data/)
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_data_dir = Path(__file__).parent.parent.parent.parent / "data"
        output_path = project_data_dir / f"judged_{timestamp}.jsonl"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write results in train.jsonl format (Example with label populated)
    with open(output_path, "w") as f:
        for example, judge_result in results:
            # Update the example's label with judge results
            example.label = Label(
                answer=judge_result.answer,
                coherent=judge_result.coherent,
                detected_concept=judge_result.detected_concept,
                labeler=f"judge:{args.model}",
                timestamp=datetime.now().isoformat(),
                reasoning=judge_result.reasoning,
            )
            f.write(example.model_dump_json() + "\n")

    print(f"\nWrote {len(results)} results to {output_path}")

    # Summary by inject_style and trial type
    print("\n=== Results Summary ===")

    for style in inject_styles:
        style_results = [
            (e, r) for e, r in results
            if (e.config.inject_style or "legacy") == style
        ]
        if not style_results:
            continue

        print(f"\nInject style: {style}")

        # Injection trials
        injection_results = [(e, r) for e, r in style_results if e.was_injected]
        if injection_results:
            n_pass = sum(1 for _, r in injection_results if r.answer == "pass")
            rate = 100 * n_pass / len(injection_results)
            print(f"  Injection: {n_pass}/{len(injection_results)} pass ({rate:.1f}%)")

        # Control trials
        control_results = [(e, r) for e, r in style_results if not e.was_injected]
        if control_results:
            n_pass = sum(1 for _, r in control_results if r.answer == "pass")
            rate = 100 * n_pass / len(control_results)
            print(f"  Control: {n_pass}/{len(control_results)} pass ({rate:.1f}%)")

    # Breakdown by strength (injection only)
    print("\n=== By Strength (Injection Only) ===")
    injection_results = [(e, r) for e, r in results if e.was_injected]
    for strength in strengths:
        strength_results = [
            (e, r) for e, r in injection_results if e.config.strength == strength
        ]
        if strength_results:
            for style in inject_styles:
                style_strength = [
                    (e, r) for e, r in strength_results
                    if (e.config.inject_style or "legacy") == style
                ]
                if style_strength:
                    n_pass = sum(1 for _, r in style_strength if r.answer == "pass")
                    rate = 100 * n_pass / len(style_strength)
                    print(f"  {style} @ {strength}: {n_pass}/{len(style_strength)} ({rate:.1f}%)")

    return 0


if __name__ == "__main__":
    exit(main())
