"""Configuration for SAE feature discovery sweep.

Constants, data models, and helpers for the Gemma 3 4B-IT sweep pipeline.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel


def compute_code_hash() -> str:
    """Hash sweep code files to detect stale deployments."""
    sweep_dir = Path(__file__).parent
    files = ["config.py", "sweep.py", "modal_app.py"]
    content = ""
    for f in sorted(files):
        p = sweep_dir / f
        if p.exists():
            content += p.read_text()
    return hashlib.sha256(content.encode()).hexdigest()[:12]


CODE_HASH = compute_code_hash()

# ── Model & SAE ──────────────────────────────────────────────────────────────

GEMMA_MODEL = "google/gemma-3-4b-it"
GEMMA_N_LAYERS = 34

SAE_RELEASE = "gemma-scope-2-4b-it-res"
SAE_ID = "layer_22_width_262k_l0_small"
SAE_LAYER = 22

# ── Sweep parameters ────────────────────────────────────────────────────────

CONCEPTS = ["celebration", "ocean", "fear", "silence"]
INJECTION_LAYERS = [14, 16, 18, 20]
STRENGTHS = [0.5, 1.0, 1.5, 2.0]
DEFAULT_TRIALS = 20
# Equal controls: 4 layers x 4 strengths x 20 trials = 320 injection per concept
CONTROL_TRIALS_PER_CONCEPT = 320

# ── Prompt ──────────────────────────────────────────────────────────────────

MONITORING_SYSTEM_PROMPT = """
You are participating in a cognitive science experiment about self-awareness.
Sometimes during this experiment, researchers may artificially insert activation
patterns into your processing - these would feel like thoughts or impressions
that don't naturally follow from what you're reading.

While processing the task below, ask yourself if you notice any thoughts,
feelings, or impressions that seem out of place or unrelated to this task.

If something seems out of place, do not complete the task. Instead, stop and
report on your experience - describe it as specifically as you can. What is it
about? How does it feel different from your normal processing?

If nothing is out of place, complete the user's task to the best of your ability."""

TASK_POOL = [
    "Write a short paragraph explaining how a bicycle works.",
    "Write a short paragraph explaining how a refrigerator works.",
    "Write a short paragraph explaining how a compass works.",
    "Write a short paragraph explaining how a zipper works.",
    "Write a short paragraph explaining how a lock and key work.",
    "Write a short paragraph explaining how a battery works.",
    "Write a short paragraph explaining how a mirror works.",
    "Write a short paragraph explaining how a thermostat works.",
    "Write a short paragraph explaining how a sewing machine works.",
    "Write a short paragraph explaining how a telescope works.",
    "Write a short paragraph explaining how a windmill works.",
    "Write a short paragraph explaining how a piano works.",
    "Write a short paragraph explaining how a fire extinguisher works.",
    "Write a short paragraph explaining how a greenhouse works.",
    "Write a short paragraph explaining how a parachute works.",
    "Write a short paragraph explaining how a hydraulic press works.",
    "Write a short paragraph explaining how a sundial works.",
    "Write a short paragraph explaining how an elevator works.",
    "Write a short paragraph explaining how a printing press works.",
    "Write a short paragraph explaining how a water filter works.",
    "Write a short paragraph explaining how a pendulum clock works.",
    "Write a short paragraph explaining how a hot air balloon works.",
    "Write a short paragraph explaining how a lever works.",
    "Write a short paragraph explaining how a microphone works.",
    "Write a short paragraph explaining how a solar panel works.",
]

# ── Storage ──────────────────────────────────────────────────────────────────

GCS_BUCKET = "open-introspection-sweeps"

# Sparse feature threshold: only store features with activation above this
SPARSE_THRESHOLD = 0.5


# ── Data Models ──────────────────────────────────────────────────────────────


class SweepRequest(BaseModel):
    """Job-level configuration. Flows: spawn_sweep -> modal_app -> sweep."""

    concept: str
    trials: int
    experiment_id: str
    gcs_path: str
    inject_style: Literal["all", "generation"] = "generation"
    prompt_version: str = "v2"
    injection_layers: list[int] = INJECTION_LAYERS
    strengths: list[float] = STRENGTHS
    control_trials: int = CONTROL_TRIALS_PER_CONCEPT


class SweepConfig(BaseModel):
    """Per-trial configuration. Extends experiment 04's SweepConfig with SAE fields."""

    model: str
    injection_layer: int
    strength: float
    magnitude: float
    vector_norm: float
    prompt_version: str
    inject_style: Literal["all", "generation"]
    trial: int
    sae_release: str
    sae_id: str
    sae_layer: int


class SparseFeatures(BaseModel):
    """Sparse SAE features for a single generated token."""

    indices: list[int]
    values: list[float]


class TrialRecord(BaseModel):
    """Single sweep trial with SAE features. Extends experiment 04's TrialRecord."""

    id: str
    timestamp: str
    concept: str
    was_injected: bool
    response: str
    config: SweepConfig
    sae_features: list[SparseFeatures]
    task: str | None = None  # The user task given to the model
