"""Parsers for experiment output files."""

import json
from pathlib import Path

from .schemas import Example, ExperimentConfig, Label


def parse_experiment_file(filepath: Path) -> list[Example]:
    """Parse an experiment JSON file into Example records.

    Works with introspection experiment outputs (exp3) which have the format:
    {
        "config": {"timestamp": ..., "layer": ..., "injection_strength": ..., ...},
        "results": {"control": [...], "injection": [...]}
    }

    Args:
        filepath: Path to the experiment JSON file.

    Returns:
        List of Example objects with empty labels (ready for labeling or judging).
    """
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
                inject_style=config.get("inject_style"),  # None for legacy data
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
                inject_style=config.get("inject_style"),
            ),
            label=Label(),
        )
        examples.append(example)

    return examples
