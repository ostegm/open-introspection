"""Shared configuration for cloud sweep infrastructure.

This module defines:
- SweepRequest: The single config object that flows through the system
- Constants: MODEL_CONFIGS, CONCEPTS, etc.
- Helpers: get_default_layers()
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

# Concepts for introspection testing
CONCEPTS = ["celebration", "ocean", "fear", "silence"]

# Model configurations
MODEL_CONFIGS: dict[str, dict[str, str | int]] = {
    "3b": {"name": "Qwen/Qwen2.5-3B-Instruct", "n_layers": 36},
    "7b": {"name": "Qwen/Qwen2.5-7B-Instruct", "n_layers": 28},
    "14b": {"name": "Qwen/Qwen2.5-14B-Instruct", "n_layers": 48},
    "32b": {"name": "Qwen/Qwen2.5-32B-Instruct", "n_layers": 64},
}

# GPU function mapping by model size
GPU_FUNCTIONS = {
    "3b": "run_sweep_l4",
    "7b": "run_sweep_l4",
    "14b": "run_sweep_a100",
    "32b": "run_sweep_a100_80gb",
}

# Default sweep parameters
DEFAULT_STRENGTHS = [1.5, 2.0, 2.5, 3.0, 4.0]
DEFAULT_TRIALS = 20

# GCS bucket for results
GCS_BUCKET = "open-introspection-sweeps"


class SweepRequest(BaseModel):
    """Configuration for a sweep job.

    This is the single config object that flows through:
    spawn_sweep.py -> modal_app.py -> sweep.py
    """

    concept: str
    model_size: str
    trials: int
    experiment_id: str
    inject_style: Literal["all", "generation"]
    layers: list[int]
    strengths: list[float]
    gcs_path: str  # Full GCS path, e.g. gs://bucket/exp/model/concept.jsonl
    prompt_version: str = "v2"


def get_default_layers(model_size: str) -> list[int]:
    """Get default layers targeting 60-75% of model depth.

    This range was found to work well in early experiments.
    """
    n_layers = int(MODEL_CONFIGS[model_size]["n_layers"])
    fractions = [0.60, 0.65, 0.70, 0.75]
    return sorted({int(n_layers * f) for f in fractions})
