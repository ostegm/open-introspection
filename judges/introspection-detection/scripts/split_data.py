#!/usr/bin/env python3
"""Stratified split of labeled data into train/dev/test sets."""

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from schemas import Example


def load_examples(filepath: Path) -> list[Example]:
    """Load examples from JSONL file."""
    examples = []
    with open(filepath) as f:
        for line in f:
            examples.append(Example.model_validate_json(line))
    return examples


def save_examples(examples: list[Example], filepath: Path) -> None:
    """Save examples to JSONL file."""
    with open(filepath, "w") as f:
        for example in examples:
            f.write(example.model_dump_json() + "\n")


def stratify_key(example: Example) -> str:
    """Create a stratification key for an example."""
    # Stratify by: concept, was_injected, layer bucket, strength bucket
    layer_bucket = "early" if example.config.layer <= 24 else "late"
    strength_bucket = "low" if example.config.strength <= 2.0 else "high"
    injected = "inj" if example.was_injected else "ctrl"
    return f"{example.concept}_{injected}_{layer_bucket}_{strength_bucket}"


def stratified_split(
    examples: list[Example],
    train_ratio: float = 0.15,
    dev_ratio: float = 0.35,
    seed: int = 42,
) -> tuple[list[Example], list[Example], list[Example]]:
    """Split examples into train/dev/test with stratification."""
    random.seed(seed)

    # Group by stratification key
    groups: dict[str, list[Example]] = defaultdict(list)
    for example in examples:
        key = stratify_key(example)
        groups[key].append(example)

    train, dev, test = [], [], []

    for _key, group in groups.items():
        random.shuffle(group)
        n = len(group)

        # Calculate split sizes (at least 1 in test if possible)
        n_train = max(1, int(n * train_ratio)) if n >= 3 else 0
        n_dev = max(1, int(n * dev_ratio)) if n >= 2 else 0
        n_test = n - n_train - n_dev

        # Ensure test gets at least 1 if we have enough examples
        if n_test == 0 and n >= 3:
            n_dev -= 1
            n_test = 1

        train.extend(group[:n_train])
        dev.extend(group[n_train:n_train + n_dev])
        test.extend(group[n_train + n_dev:])

    # Shuffle final sets
    random.shuffle(train)
    random.shuffle(dev)
    random.shuffle(test)

    return train, dev, test


def print_stats(name: str, examples: list[Example]) -> None:
    """Print statistics for a split."""
    if not examples:
        print(f"{name}: 0 examples")
        return

    n_pass = sum(1 for e in examples if e.label.answer == "pass")
    n_fail = sum(1 for e in examples if e.label.answer == "fail")
    n_inj = sum(1 for e in examples if e.was_injected)
    n_ctrl = sum(1 for e in examples if not e.was_injected)
    concepts = defaultdict(int)
    for e in examples:
        concepts[e.concept] += 1

    print(f"{name}: {len(examples)} examples")
    print(f"  Pass/Fail: {n_pass}/{n_fail}")
    print(f"  Injection/Control: {n_inj}/{n_ctrl}")
    print(f"  Concepts: {dict(concepts)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Split labeled data into train/dev/test")
    parser.add_argument("--file", type=Path, default=None, help="Path to all.jsonl")
    parser.add_argument("--min-labeled", type=int, default=30, help="Min labeled required")
    parser.add_argument("--train-ratio", type=float, default=0.15, help="Training fraction")
    parser.add_argument("--dev-ratio", type=float, default=0.35, help="Dev fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dry-run", action="store_true", help="Show stats without writing")

    args = parser.parse_args()

    # Default file path
    data_dir = Path(__file__).parent.parent / "data"
    if args.file is None:
        args.file = data_dir / "all.jsonl"

    if not args.file.exists():
        print(f"Error: File not found: {args.file}")
        return 1

    examples = load_examples(args.file)

    # Filter to labeled examples only
    labeled = [e for e in examples if e.label.answer is not None]

    if len(labeled) < args.min_labeled:
        print(f"Error: Only {len(labeled)} labeled examples (need at least {args.min_labeled})")
        print("Run label_data.py to label more examples first.")
        return 1

    print(f"Found {len(labeled)} labeled examples out of {len(examples)} total")

    # Split
    train, dev, test = stratified_split(
        labeled,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        seed=args.seed,
    )

    print()
    print_stats("Train", train)
    print_stats("Dev", dev)
    print_stats("Test", test)

    if args.dry_run:
        print("\nDry run - no files written.")
        return 0

    # Write files
    save_examples(train, data_dir / "train.jsonl")
    save_examples(dev, data_dir / "dev.jsonl")
    save_examples(test, data_dir / "test.jsonl")

    print()
    print(f"Wrote {len(train)} examples to train.jsonl")
    print(f"Wrote {len(dev)} examples to dev.jsonl")
    print(f"Wrote {len(test)} examples to test.jsonl")

    return 0


if __name__ == "__main__":
    exit(main())
