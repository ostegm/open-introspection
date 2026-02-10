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
CONTROL_TRIALS_PER_CONCEPT = 100

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
