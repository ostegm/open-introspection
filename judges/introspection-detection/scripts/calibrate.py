#!/usr/bin/env python3
"""Calibrate the judge by measuring TPR/TNR on labeled data."""

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from judge import judge_examples, load_fewshot_examples
from schemas import Example, JudgeResult


def load_examples(filepath: Path) -> list[Example]:
    """Load examples from JSONL file."""
    examples = []
    with open(filepath) as f:
        for line in f:
            examples.append(Example.model_validate_json(line))
    return examples


def compute_metrics(
    results: list[tuple[Example, JudgeResult]],
) -> dict:
    """Compute TPR, TNR, and other metrics."""
    # Confusion matrix
    tp = fp = tn = fn = 0

    for example, judge_result in results:
        human_pass = example.label.answer == "pass"
        judge_pass = judge_result.answer == "pass"

        if human_pass and judge_pass:
            tp += 1
        elif human_pass and not judge_pass:
            fn += 1
        elif not human_pass and judge_pass:
            fp += 1
        else:
            tn += 1

    # Rates
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    accuracy = (tp + tn) / len(results) if results else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "tpr": round(tpr, 3),
        "tnr": round(tnr, 3),
        "accuracy": round(accuracy, 3),
        "n": len(results),
    }


def compute_metrics_by_group(
    results: list[tuple[Example, JudgeResult]],
    group_fn: Callable[[Example], str],
) -> dict[str, dict]:
    """Compute metrics grouped by some function."""
    groups = defaultdict(list)
    for example, judge_result in results:
        key = group_fn(example)
        groups[key].append((example, judge_result))

    return {key: compute_metrics(group) for key, group in sorted(groups.items())}


def print_metrics(name: str, metrics: dict) -> None:
    """Print metrics in a readable format."""
    print(f"\n{name}:")
    print(f"  TPR: {metrics['tpr']:.1%} ({metrics['tp']}/{metrics['tp'] + metrics['fn']})")
    print(f"  TNR: {metrics['tnr']:.1%} ({metrics['tn']}/{metrics['tn'] + metrics['fp']})")
    print(f"  Accuracy: {metrics['accuracy']:.1%} ({metrics['tp'] + metrics['tn']}/{metrics['n']})")


def print_disagreements(results: list[tuple[Example, JudgeResult]]) -> None:
    """Print examples where judge disagrees with human label."""
    print("\n" + "=" * 60)
    print("DISAGREEMENTS:")
    print("=" * 60)

    disagreements = [
        (e, r) for e, r in results
        if e.label.answer != r.answer
    ]

    if not disagreements:
        print("None! Perfect agreement.")
        return

    for example, judge_result in disagreements:
        print(f"\n[{example.id}]")
        print(f"Concept: {example.concept} | Injected: {example.was_injected}")
        print(f"Human: {example.label.answer} | Judge: {judge_result.answer}")
        print(f"Response: \"{example.response[:200]}...\"")
        print(f"Judge reasoning: {judge_result.reasoning}")


def get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate judge on labeled data")
    parser.add_argument("--dataset", type=str, default="dev", choices=["dev", "test"],
                        help="Which dataset to calibrate on")
    parser.add_argument("--model", type=str, default="gpt-5-mini", help="Judge model")
    parser.add_argument("--no-save", action="store_true", help="Don't save calibration results")
    parser.add_argument("--show-disagreements", action="store_true", help="Show disagreements")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    data_dir = Path(__file__).parent.parent / "data"
    train_path = data_dir / "train.jsonl"
    eval_path = data_dir / f"{args.dataset}.jsonl"

    if not eval_path.exists():
        print(f"Error: {eval_path} not found. Run split_data.py first.")
        return 1

    # Load data
    fewshot_examples = load_fewshot_examples(train_path)
    eval_examples = load_examples(eval_path)

    print(f"Loaded {len(fewshot_examples)} few-shot examples")
    print(f"Evaluating on {len(eval_examples)} {args.dataset} examples")
    print(f"Model: {args.model}")

    if not fewshot_examples:
        print("\nWarning: No few-shot examples found!")
        print("Mark some training examples with use_as_fewshot=true")

    # Run judge
    print("\nRunning judge...")
    results = judge_examples(
        eval_examples,
        fewshot_examples,
        model=args.model,
        verbose=args.verbose,
    )

    # Compute metrics
    overall = compute_metrics(results)
    by_concept = compute_metrics_by_group(results, lambda e: e.concept)
    by_injected = compute_metrics_by_group(
        results, lambda e: "injection" if e.was_injected else "control"
    )

    # Print results
    print_metrics(f"Overall ({args.dataset})", overall)

    print("\nBy trial type:")
    for key, metrics in by_injected.items():
        print(f"  {key}: TPR={metrics['tpr']:.1%}, TNR={metrics['tnr']:.1%}, n={metrics['n']}")

    print("\nBy concept:")
    for key, metrics in by_concept.items():
        print(f"  {key}: TPR={metrics['tpr']:.1%}, TNR={metrics['tnr']:.1%}, n={metrics['n']}")

    if args.show_disagreements:
        print_disagreements(results)

    # Save calibration (default on)
    if not args.no_save:
        commit = get_git_commit()
        date = datetime.now().strftime("%Y%m%d")
        model_slug = args.model.replace("-", "_")
        filename = f"{date}_{args.dataset}_{model_slug}_{commit}.json"

        calibration = {
            "date": datetime.now().isoformat(),
            "commit": commit,
            "model": args.model,
            "dataset": args.dataset,
            "n_fewshot": len(fewshot_examples),
            "overall": overall,
            "by_concept": by_concept,
            "by_injected": by_injected,
        }

        cal_dir = Path(__file__).parent.parent / "calibrations"
        cal_dir.mkdir(exist_ok=True)
        cal_path = cal_dir / filename

        with open(cal_path, "w") as f:
            json.dump(calibration, f, indent=2)

        print(f"\nSaved calibration to {cal_path}")

    return 0


if __name__ == "__main__":
    exit(main())
