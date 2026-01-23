#!/usr/bin/env python3
"""Parse sweep JSON files into all.jsonl for labeling."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from schemas import Example, ExperimentConfig, Label


def parse_sweep_file(filepath: Path) -> list[Example]:
    """Parse a single sweep JSON file into Example records."""
    with open(filepath) as f:
        data = json.load(f)

    config = data["config"]
    timestamp = config["timestamp"]
    results = data["results"]

    examples = []

    # Parse control trials (was_injected=False)
    for trial in results.get("control", []):
        concept = trial["concept"]
        example = Example(
            id=f"{timestamp}_{concept}_control",
            source_file=filepath.name,
            concept=concept,
            was_injected=False,
            response=trial["response"],
            config=ExperimentConfig(
                layer=config["layer"],
                strength=config["injection_strength"],
                prompt_version=config["prompt_version"],
            ),
            label=Label(),
        )
        examples.append(example)

    # Parse injection trials (was_injected=True)
    for trial in results.get("injection", []):
        concept = trial["concept"]
        example = Example(
            id=f"{timestamp}_{concept}_injection",
            source_file=filepath.name,
            concept=concept,
            was_injected=True,
            response=trial["response"],
            config=ExperimentConfig(
                layer=config["layer"],
                strength=config["injection_strength"],
                prompt_version=config["prompt_version"],
            ),
            label=Label(),
        )
        examples.append(example)

    return examples


def main() -> int:
    """Parse all sweep files and write to all.jsonl."""
    sweep_dir = Path(__file__).parent.parent.parent.parent / "data" / "sweep_analysis_20260122"
    output_path = Path(__file__).parent.parent / "data" / "all.jsonl"

    if not sweep_dir.exists():
        print(f"Error: Sweep directory not found: {sweep_dir}")
        return 1

    json_files = sorted(sweep_dir.glob("introspection_*.json"))
    if not json_files:
        print(f"Error: No JSON files found in {sweep_dir}")
        return 1

    print(f"Found {len(json_files)} sweep files")

    all_examples: list[Example] = []
    for filepath in json_files:
        examples = parse_sweep_file(filepath)
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
