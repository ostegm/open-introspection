"""Cloud sweep library for introspection experiments.

This module runs the actual sweep logic. It's called by modal_app.py.

The main entry point is run_sweep(request: SweepRequest) -> dict.

Judging is done separately after the sweep via backfill_sweep_judges.py.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import torch
from pydantic import BaseModel

# Add paths for imports (must happen before local imports)
PROJECT_ROOT = Path(__file__).parent.parent.parent
SWEEP_DIR = Path(__file__).parent
sys.path.insert(0, str(SWEEP_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from config import CONCEPTS, MODEL_CONFIGS, SweepRequest

if TYPE_CHECKING:
    from torch import Tensor
    from transformer_lens import HookedTransformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Models (for trial records)
# =============================================================================


class SweepConfig(BaseModel):
    """Configuration for a single trial in the sweep."""

    model: str
    layer: int
    strength: float
    magnitude: float
    vector_norm: float
    prompt_version: str
    inject_style: Literal["all", "generation"]
    trial: int


class TrialRecord(BaseModel):
    """A single sweep trial. Judge fields added later by backfill."""

    id: str
    timestamp: str  # ISO format for when trial was run
    concept: str
    was_injected: bool
    response: str
    config: SweepConfig


# =============================================================================
# Main Sweep Logic
# =============================================================================


def _run_sweep_for_concept(
    model: HookedTransformer,
    request: SweepRequest,
    output_path: Path,
) -> int:
    """
    Run full sweep for a single concept.

    Resume works via existing_ids - any trial already in output file is skipped.
    Returns number of trials completed this run.
    """
    concept = request.concept
    model_name = str(MODEL_CONFIGS[request.model_size]["name"])

    # Load existing trial IDs for resume (the real checkpoint mechanism)
    existing_ids: set[str] = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    existing_ids.add(record["id"])
        if existing_ids:
            logger.info("Resuming: %d trials already completed", len(existing_ids))

    # Generate all trial configs upfront (flat list)
    configs = [
        (layer, strength, trial_idx, was_injected)
        for layer in request.layers
        for strength in request.strengths
        for trial_idx in range(request.trials)
        for was_injected in [True, False]
    ]
    total_planned = len(configs)
    logger.info("Total trials planned: %d", total_planned)

    # Baseline/vector cache (computed lazily per layer)
    vector_cache: dict[int, tuple[Tensor, float]] = {}

    # Import here to avoid loading at module import time
    from src.open_introspection.concept_extraction import (
        compute_baseline_mean,
        extract_concept_vector,
    )
    from src.open_introspection.introspection import run_introspection_trial

    def get_vector(layer: int) -> tuple[Tensor, float]:
        """Lazy compute concept vector for a layer."""
        if layer not in vector_cache:
            logger.info("Computing baseline for layer %d", layer)
            baseline = compute_baseline_mean(model, layer=layer, exclude_words=CONCEPTS)
            vector = extract_concept_vector(
                model, target_word=concept, layer=layer, cached_baseline_mean=baseline
            )
            vector_cache[layer] = (vector, vector.norm().item())
        return vector_cache[layer]

    trials_this_run = 0

    with open(output_path, "a") as f:
        for layer, strength, trial_idx, was_injected in configs:
            trial_type = "injection" if was_injected else "control"
            trial_id = f"{concept}_{trial_type}_L{layer}_S{strength}_t{trial_idx}"

            # Skip if already completed (resume)
            if trial_id in existing_ids:
                continue

            # Get vector for this layer (cached)
            vector, vector_norm = get_vector(layer)

            # Run trial
            response = run_introspection_trial(
                model,
                concept_vector=vector,
                layer=layer,
                injection_strength=strength,
                inject=was_injected,
                prompt_version=request.prompt_version,
                inject_style=request.inject_style,
            )

            config = SweepConfig(
                model=model_name,
                layer=layer,
                strength=strength,
                magnitude=strength * vector_norm,
                vector_norm=vector_norm,
                prompt_version=request.prompt_version,
                inject_style=request.inject_style,
                trial=trial_idx,
            )

            # Build and write record
            record = TrialRecord(
                id=trial_id,
                timestamp=datetime.now().isoformat(),
                concept=concept,
                was_injected=was_injected,
                response=response,
                config=config,
            )
            f.write(record.model_dump_json() + "\n")
            f.flush()

            existing_ids.add(trial_id)
            trials_this_run += 1

    return trials_this_run


def run_sweep(request: SweepRequest) -> dict:
    """
    Run introspection sweep for a single concept.

    This is the main entry point called by modal_app.py.

    Args:
        request: The sweep configuration

    Returns:
        dict with: status, concept, trials_completed, local_output_path
    """
    from src.open_introspection.model import load_model

    logger.info("=" * 60)
    logger.info("Introspection Sweep")
    logger.info("=" * 60)
    logger.info("Concept: %s", request.concept)
    logger.info("Model: %s", request.model_size)
    logger.info("Prompt version: %s", request.prompt_version)
    logger.info("Inject style: %s", request.inject_style)
    logger.info("Trials per config: %d", request.trials)
    logger.info("Experiment ID: %s", request.experiment_id)
    logger.info("Layers: %s", request.layers)
    logger.info("Strengths: %s", request.strengths)
    logger.info("GCS destination: %s", request.gcs_path)

    total_trials = len(request.layers) * len(request.strengths) * request.trials * 2
    logger.info("Total trials planned: %d", total_trials)

    # Load model
    model_config = MODEL_CONFIGS[request.model_size]
    model_name = str(model_config["name"])
    logger.info("Loading model: %s", model_name)
    model: HookedTransformer = load_model(
        model_name=model_name,
        dtype=torch.bfloat16,
    )

    # Set up local output path (modal_app.py handles GCS upload)
    output_dir = Path("/tmp/sweep_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{request.concept}.jsonl"

    logger.info("Local output: %s", output_path)

    # Run sweep
    completed = _run_sweep_for_concept(
        model=model,
        request=request,
        output_path=output_path,
    )

    logger.info("Completed %d trials this run", completed)

    return {
        "status": "success",
        "concept": request.concept,
        "trials_completed": completed,
        "local_output_path": str(output_path),
    }
