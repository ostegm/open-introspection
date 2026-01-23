#!/usr/bin/env python3
"""CLI labeler for introspection examples."""

import argparse
import sys
from datetime import datetime
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


def is_labeled(example: Example) -> bool:
    """Check if an example has been labeled."""
    return example.label.answer is not None


def filter_examples(
    examples: list[Example],
    unlabeled_only: bool = False,
    concept: str | None = None,
    was_injected: bool | None = None,
    layer: int | None = None,
    strength: float | None = None,
) -> list[Example]:
    """Filter examples based on criteria."""
    filtered = examples

    if unlabeled_only:
        filtered = [e for e in filtered if not is_labeled(e)]

    if concept is not None:
        filtered = [e for e in filtered if e.concept == concept]

    if was_injected is not None:
        filtered = [e for e in filtered if e.was_injected == was_injected]

    if layer is not None:
        filtered = [e for e in filtered if e.config.layer == layer]

    if strength is not None:
        filtered = [e for e in filtered if e.config.strength == strength]

    return filtered


def display_example(example: Example, index: int, total: int) -> None:
    """Display an example for labeling."""
    injected_str = "YES" if example.was_injected else "NO"

    print(f"\n[{index + 1}/{total}] {example.id}")
    print("─" * 60)
    print(f"Concept: {example.concept} | Injected: {injected_str}")
    cfg = example.config
    print(f"Layer: {cfg.layer} | Strength: {cfg.strength} | Prompt: {cfg.prompt_version}")
    print()

    # Word wrap the response for readability
    response = example.response
    print("Response:")
    print(f'"{response}"')
    print()


def get_answer() -> str | None:
    """Get pass/fail/skip answer from user."""
    while True:
        answer = input("Answer (p=pass, f=fail, s=skip, q=quit): ").strip().lower()
        if answer in ("p", "pass"):
            return "pass"
        elif answer in ("f", "fail"):
            return "fail"
        elif answer in ("s", "skip"):
            return None
        elif answer in ("q", "quit"):
            raise KeyboardInterrupt
        else:
            print("Invalid input. Use p/f/s/q.")


def get_coherent() -> bool:
    """Get coherent flag from user."""
    while True:
        answer = input("Coherent? (y/n): ").strip().lower()
        if answer in ("y", "yes"):
            return True
        elif answer in ("n", "no"):
            return False
        else:
            print("Invalid input. Use y/n.")


def get_detected_concept() -> str | None:
    """Get detected concept from user."""
    answer = input("Detected concept (or Enter if none): ").strip().lower()
    return answer if answer else None


def label_example(example: Example, labeler: str) -> bool:
    """Label a single example. Returns True if labeled, False if skipped."""
    answer = get_answer()
    if answer is None:
        print("⏭ Skipped.")
        return False

    coherent = get_coherent()
    detected_concept = get_detected_concept()

    example.label.answer = answer
    example.label.coherent = coherent
    example.label.detected_concept = detected_concept
    example.label.labeler = labeler
    example.label.timestamp = datetime.now().isoformat()

    print("✓ Saved.")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="CLI labeler for introspection examples")
    parser.add_argument(
        "--file", type=Path, default=None, help="Path to JSONL file (default: all.jsonl)"
    )
    parser.add_argument("--unlabeled-only", action="store_true", help="Only unlabeled")
    parser.add_argument("--concept", type=str, help="Filter by concept")
    parser.add_argument("--injected", type=str, help="Filter by injection (true/false)")
    parser.add_argument("--layer", type=int, help="Filter by layer")
    parser.add_argument("--strength", type=float, help="Filter by strength")
    parser.add_argument("--labeler", type=str, default="human", help="Labeler name")
    parser.add_argument("--stats", action="store_true", help="Show labeling statistics and exit")

    args = parser.parse_args()

    # Default file path
    if args.file is None:
        args.file = Path(__file__).parent.parent / "data" / "all.jsonl"

    if not args.file.exists():
        print(f"Error: File not found: {args.file}")
        print("Run prepare_data.py first to generate all.jsonl")
        return 1

    examples = load_examples(args.file)

    # Stats mode
    if args.stats:
        n_labeled = sum(1 for e in examples if is_labeled(e))
        n_pass = sum(1 for e in examples if e.label.answer == "pass")
        n_fail = sum(1 for e in examples if e.label.answer == "fail")
        n_coherent = sum(1 for e in examples if e.label.coherent is True)
        n_incoherent = sum(1 for e in examples if e.label.coherent is False)

        print(f"Total examples: {len(examples)}")
        print(f"Labeled: {n_labeled} ({100*n_labeled/len(examples):.1f}%)")
        print(f"  Pass: {n_pass}")
        print(f"  Fail: {n_fail}")
        print(f"  Coherent: {n_coherent}")
        print(f"  Incoherent: {n_incoherent}")
        print(f"Unlabeled: {len(examples) - n_labeled}")
        return 0

    # Parse injection filter
    was_injected = None
    if args.injected is not None:
        was_injected = args.injected.lower() in ("true", "yes", "1")

    # Filter examples
    filtered = filter_examples(
        examples,
        unlabeled_only=args.unlabeled_only,
        concept=args.concept,
        was_injected=was_injected,
        layer=args.layer,
        strength=args.strength,
    )

    if not filtered:
        print("No examples match the filter criteria.")
        if args.unlabeled_only:
            print("All matching examples have been labeled!")
        return 0

    n_labeled_total = sum(1 for e in examples if is_labeled(e))
    print(f"Found {len(filtered)} examples to label ({n_labeled_total} already labeled)")
    print("Commands: p=pass, f=fail, s=skip, q=quit")

    try:
        for i, example in enumerate(filtered):
            display_example(example, i, len(filtered))
            label_example(example, args.labeler)

            # Save after each label
            save_examples(examples, args.file)

            n_labeled_now = sum(1 for e in examples if is_labeled(e))
            n_remaining = len(examples) - n_labeled_now
            print(f"[{n_labeled_now} labeled, {n_remaining} remaining]")

    except KeyboardInterrupt:
        print("\n\nQuitting. Progress saved.")
        save_examples(examples, args.file)

    return 0


if __name__ == "__main__":
    exit(main())
