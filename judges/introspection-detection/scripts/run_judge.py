#!/usr/bin/env python3
"""Run judge on all examples and save results."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from judge import judge_examples, load_fewshot_examples
from schemas import Example


def load_examples(filepath: Path) -> list[Example]:
    """Load examples from JSONL file."""
    examples = []
    with open(filepath) as f:
        for line in f:
            examples.append(Example.model_validate_json(line))
    return examples


def main() -> int:
    parser = argparse.ArgumentParser(description="Run judge on examples")
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL file")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL file")
    parser.add_argument("--model", type=str, default="gpt-5-nano", help="Judge model")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        return 1

    # Load data
    data_dir = Path(__file__).parent.parent / "data"
    train_path = data_dir / "train.jsonl"

    fewshot_examples = load_fewshot_examples(train_path)
    examples = load_examples(args.input)

    print(f"Loaded {len(fewshot_examples)} few-shot examples")
    print(f"Scoring {len(examples)} examples")
    print(f"Model: {args.model}")

    if not fewshot_examples:
        print("\nWarning: No few-shot examples found!")

    # Run judge
    print("\nRunning judge...")
    results = judge_examples(
        examples,
        fewshot_examples,
        model=args.model,
        verbose=args.verbose,
    )

    # Write results
    with open(args.output, "w") as f:
        for example, judge_result in results:
            record = {
                "id": example.id,
                "concept": example.concept,
                "was_injected": example.was_injected,
                "config": example.config.model_dump(),
                "judge": judge_result.model_dump(),
            }
            f.write(json.dumps(record) + "\n")

    print(f"\nWrote {len(results)} results to {args.output}")

    # Summary
    n_pass = sum(1 for _, r in results if r.answer == "pass")
    n_coherent = sum(1 for _, r in results if r.coherent)

    print(f"  Pass rate: {n_pass}/{len(results)} ({100*n_pass/len(results):.1f}%)")
    print(f"  Coherent: {n_coherent}/{len(results)} ({100*n_coherent/len(results):.1f}%)")

    return 0


if __name__ == "__main__":
    exit(main())
