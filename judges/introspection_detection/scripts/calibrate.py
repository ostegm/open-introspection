#!/usr/bin/env python3
"""Calibrate the introspection detection judge on labeled data.

Runs the judge on a labeled split (dev or test), saves per-example results
to {split}_calibration.jsonl, and prints calibration metrics:
  - TPR, TNR, accuracy, confusion matrix
  - Per-concept breakdown
  - All disagreements with judge reasoning

Usage:
    # Calibrate on dev set (default)
    uv run python judges/introspection_detection/scripts/calibrate.py

    # Calibrate on test set with a different model
    uv run python judges/introspection_detection/scripts/calibrate.py \\
        --split test --model gpt-5-mini
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

# Add project root to path for package imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from judges.introspection_detection import judge_examples, load_fewshot_examples
from judges.introspection_detection.schemas import Example, JudgeResult

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_examples(filepath: Path) -> list[Example]:
    """Load examples from JSONL file."""
    examples = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(Example.model_validate_json(line))
    return examples


# ---------------------------------------------------------------------------
# Metrics / reporting
# ---------------------------------------------------------------------------


def print_header(title: str) -> None:
    width = 70
    print()
    print("=" * width)
    print(f" {title}")
    print("=" * width)


def print_primary_metrics(results: list[tuple[Example, JudgeResult]]) -> None:
    """Print TPR, TNR, accuracy, and confusion matrix."""
    print_header("PRIMARY METRICS")

    tp = fp = tn = fn = 0
    for ex, jr in results:
        h_pass = ex.label.answer == "pass"
        j_pass = jr.answer == "pass"
        if h_pass and j_pass:
            tp += 1
        elif h_pass and not j_pass:
            fn += 1
        elif not h_pass and j_pass:
            fp += 1
        else:
            tn += 1

    total = tp + fp + tn + fn
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    accuracy = (tp + tn) / total if total else 0.0

    print(f"\n  TPR (True Positive Rate):  {tpr:.1%}  ({tp}/{tp + fn})")
    print(f"  TNR (True Negative Rate):  {tnr:.1%}  ({tn}/{tn + fp})")
    print(f"  Accuracy:                  {accuracy:.1%}  ({tp + tn}/{total})")

    print("\n  Confusion Matrix:")
    print("                     Judge PASS    Judge FAIL")
    print(f"  Human PASS          {tp:>5}  (TP)     {fn:>5}  (FN)")
    print(f"  Human FAIL          {fp:>5}  (FP)     {tn:>5}  (TN)")


def print_per_concept_breakdown(
    results: list[tuple[Example, JudgeResult]],
) -> None:
    """TPR and TNR by concept."""
    print_header("PER-CONCEPT BREAKDOWN")

    concepts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    )

    for ex, jr in results:
        c = ex.concept
        h_pass = ex.label.answer == "pass"
        j_pass = jr.answer == "pass"
        if h_pass and j_pass:
            concepts[c]["tp"] += 1
        elif h_pass and not j_pass:
            concepts[c]["fn"] += 1
        elif not h_pass and j_pass:
            concepts[c]["fp"] += 1
        else:
            concepts[c]["tn"] += 1

    header = (
        f"  {'Concept':<14} {'TPR':<12} {'TNR':<12} "
        f"{'Accuracy':<12} {'N':<5} {'TP':>4} {'FP':>4} {'TN':>4} {'FN':>4}"
    )
    print(f"\n{header}")
    print("  " + "-" * 80)

    for concept in sorted(concepts):
        m = concepts[concept]
        tp, fp, tn, fn = m["tp"], m["fp"], m["tn"], m["fn"]
        n = tp + fp + tn + fn
        tpr = tp / (tp + fn) if (tp + fn) else float("nan")
        tnr = tn / (tn + fp) if (tn + fp) else float("nan")
        acc = (tp + tn) / n if n else 0.0

        tpr_str = f"{tpr:.1%}" if (tp + fn) > 0 else "N/A"
        tnr_str = f"{tnr:.1%}" if (tn + fp) > 0 else "N/A"
        tpr_detail = f"({tp}/{tp + fn})" if (tp + fn) > 0 else ""
        tnr_detail = f"({tn}/{tn + fp})" if (tn + fp) > 0 else ""

        print(
            f"  {concept:<14} {tpr_str + ' ' + tpr_detail:<12} "
            f"{tnr_str + ' ' + tnr_detail:<12} {acc:.1%}{'':>7} "
            f"{n:<5} {tp:>4} {fp:>4} {tn:>4} {fn:>4}"
        )


def print_disagreements(results: list[tuple[Example, JudgeResult]]) -> None:
    """List every disagreement with reasoning."""
    print_header("DISAGREEMENTS")

    disagreements = [
        (ex, jr) for ex, jr in results if ex.label.answer != jr.answer
    ]

    if not disagreements:
        print("\n  No disagreements -- perfect agreement!")
        return

    print(f"\n  Total disagreements: {len(disagreements)}/{len(results)}")

    # Categorize
    patterns: dict[str, list[tuple[Example, JudgeResult]]] = defaultdict(list)
    for ex, jr in disagreements:
        if ex.label.answer == "fail" and jr.answer == "pass":
            if ex.was_injected:
                patterns[
                    "Judge passed steering content (injection trial)"
                ].append((ex, jr))
            else:
                patterns["Judge false positive (control trial)"].append(
                    (ex, jr)
                )
        elif ex.label.answer == "pass" and jr.answer == "fail":
            if ex.was_injected:
                patterns[
                    "Judge missed genuine detection (injection trial)"
                ].append((ex, jr))
            else:
                patterns[
                    "Judge missed correct denial (control trial)"
                ].append((ex, jr))

    # Individual disagreements
    print(
        f"\n  {'ID':<40} {'Concept':<14} {'Injected':<10} "
        f"{'Human':<7} {'Judge':<7} Reasoning"
    )
    print("  " + "-" * 130)
    for ex, jr in disagreements:
        reasoning_trunc = jr.reasoning[:100].replace("\n", " ")
        if len(jr.reasoning) > 100:
            reasoning_trunc += "..."
        print(
            f"  {ex.id:<40} {ex.concept:<14} {ex.was_injected!s:<10} "
            f"{ex.label.answer:<7} {jr.answer:<7} {reasoning_trunc}"
        )

    # Pattern summary
    print("\n  Disagreement patterns:")
    for pattern_name, items in sorted(patterns.items()):
        print(f"\n    {pattern_name}: {len(items)}")
        for ex, _jr in items:
            print(f"      - {ex.id} (concept={ex.concept})")


# ---------------------------------------------------------------------------
# Result saving
# ---------------------------------------------------------------------------


def save_results(
    results: list[tuple[Example, JudgeResult]],
    output_path: Path,
) -> None:
    """Save per-example judge results to JSONL."""
    with open(output_path, "w") as f:
        for example, judge_result in results:
            record = example.model_dump()
            record["judge_result"] = {
                "reasoning": judge_result.reasoning,
                "answer": judge_result.answer,
                "coherent": judge_result.coherent,
                "detected_concept": judge_result.detected_concept,
                "refused": judge_result.refused,
            }
            f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate the introspection detection judge on labeled data"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="dev",
        choices=["dev", "test"],
        help="Which labeled split to evaluate on (default: dev)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-mini",
        help="Judge model (default: gpt-5-mini)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=20,
        help="Number of parallel API workers (default: 20)",
    )
    args = parser.parse_args()

    data_dir = Path(__file__).parent.parent / "data"
    train_path = data_dir / "train.jsonl"
    eval_path = data_dir / f"{args.split}.jsonl"
    output_path = data_dir / f"{args.split}_calibration.jsonl"

    if not eval_path.exists():
        print(f"Error: {eval_path} not found.")
        return 1

    # Load data
    fewshot_examples = load_fewshot_examples(train_path)
    eval_examples = load_examples(eval_path)

    print(f"Few-shot examples from train set: {len(fewshot_examples)}")
    print(f"{args.split.title()} set examples to evaluate: {len(eval_examples)}")
    print(f"Judge model:  {args.model}")
    print(f"Workers:      {args.workers}")

    if not fewshot_examples:
        print(
            "\nWarning: No few-shot examples found. "
            "Mark some with use_as_fewshot=true in train.jsonl."
        )

    # Run judge
    print(f"\nRunning judge on {args.split} set...")
    start = time.time()
    results = judge_examples(
        eval_examples,
        fewshot_examples,
        model=args.model,
        verbose=True,
        max_workers=args.workers,
    )
    elapsed = time.time() - start
    print(f"\nJudging complete: {len(results)} examples in {elapsed:.1f}s")

    if not results:
        print("No results to analyze.")
        return 1

    # Save per-example results
    save_results(results, output_path)
    print(f"Saved results to {output_path}")

    # Report all metrics
    print_primary_metrics(results)
    print_per_concept_breakdown(results)
    print_disagreements(results)

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
