#!/usr/bin/env python
"""Cloud-optimized introspection sweep with online judging.

Designed for Cloud Batch parallel execution. Each task handles one concept.

Features:
- Judge retry with exponential backoff
- Periodic GCS uploads (crash-safe)
- Resume from checkpoint after interruption

Usage:
    # Local test (single concept, few trials)
    uv run python experiments/04_cloud_sweep/sweep.py --concept fear --trials 2 --local

    # Cloud Batch sets BATCH_TASK_INDEX and GCS_BUCKET
    # Task 0=celebration, 1=ocean, 2=fear, 3=silence
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import torch
from pydantic import BaseModel

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.open_introspection.concept_extraction import (
    compute_baseline_mean,
    extract_concept_vector,
)
from src.open_introspection.introspection import run_introspection_trial
from src.open_introspection.model import load_model

if TYPE_CHECKING:
    from torch import Tensor
    from transformer_lens import HookedTransformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Concept ordering for task index mapping
CONCEPTS = ["celebration", "ocean", "fear", "silence"]

# Model configurations
MODEL_CONFIGS: dict[str, dict[str, str | int]] = {
    "3b": {"name": "Qwen/Qwen2.5-3B-Instruct", "n_layers": 36},
    "7b": {"name": "Qwen/Qwen2.5-7B-Instruct", "n_layers": 28},
}

# Sweep parameters
DEFAULT_LAYERS_FRACTIONS = [1 / 6, 1 / 3, 1 / 2, 2 / 3, 5 / 6]  # + near-final
DEFAULT_STRENGTHS = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
DEFAULT_TRIALS_PER_CONFIG = 40

# Fault tolerance settings
JUDGE_MAX_RETRIES = 3
JUDGE_RETRY_BASE_DELAY = 1.0  # seconds, doubles each retry
GCS_UPLOAD_INTERVAL = 50  # upload every N trials


# =============================================================================
# Pydantic Models
# =============================================================================


class SweepConfig(BaseModel):
    """Configuration for a single trial in the sweep."""

    model: str
    layer: int
    strength: float
    magnitude: float
    vector_norm: float
    prompt_version: str
    trial: int


class JudgeOutput(BaseModel):
    """Result from the introspection judge."""

    answer: Literal["pass", "fail"]
    coherent: bool
    detected_concept: str | None = None
    reasoning: str


class TrialRecord(BaseModel):
    """A single sweep trial with optional judge result."""

    id: str
    concept: str
    was_injected: bool
    response: str
    config: SweepConfig
    judge: JudgeOutput | None = None
    judge_error: str | None = None


class Checkpoint(BaseModel):
    """Tracks progress for resume capability."""

    layer_idx: int = 0
    strength_idx: int = 0
    trial_idx: int = 0
    inject_done: bool = False  # True = injection done, need control
    total_completed: int = 0


def get_layers_for_model(n_layers: int) -> list[int]:
    """Compute layer indices for sweep based on model depth."""
    layers = [int(n_layers * f) for f in DEFAULT_LAYERS_FRACTIONS]
    layers.append(n_layers - 2)  # near-final
    return sorted(set(layers))  # dedupe


def _import_judge_module() -> tuple[Any, Any] | None:
    """Import judge module from hyphenated directory path."""
    import importlib.util

    judge_dir = PROJECT_ROOT / "judges" / "introspection-detection"

    # Add judge directory to path so relative imports work
    judge_dir_str = str(judge_dir)
    if judge_dir_str not in sys.path:
        sys.path.insert(0, judge_dir_str)

    try:
        # Load schemas module first (judge.py imports from it)
        schemas_spec = importlib.util.spec_from_file_location(
            "schemas", judge_dir / "schemas.py"
        )
        if schemas_spec is None or schemas_spec.loader is None:
            return None
        schemas_module = importlib.util.module_from_spec(schemas_spec)
        sys.modules["schemas"] = schemas_module  # Key: use "schemas" so judge.py can find it
        schemas_spec.loader.exec_module(schemas_module)

        # Load judge module
        judge_spec = importlib.util.spec_from_file_location(
            "judge_module", judge_dir / "judge.py"
        )
        if judge_spec is None or judge_spec.loader is None:
            return None
        judge_module = importlib.util.module_from_spec(judge_spec)
        sys.modules["judge_module"] = judge_module
        judge_spec.loader.exec_module(judge_module)

        return judge_module, schemas_module
    except Exception as e:
        logger.error("Failed to import judge modules: %s", e)
        return None


class JudgeClient:
    """Wrapper for judge with retry logic."""

    def __init__(self) -> None:
        self.client: Any = None  # OpenAI client, typed as Any to avoid import
        self.fewshot: list[Any] = []
        self.example_cls: type | None = None
        self.judge_module: Any = None
        self.schemas_module: Any = None
        self._initialized = False

    def initialize(self) -> bool:
        """Initialize judge client and load few-shot examples."""
        if self._initialized:
            return self.client is not None

        self._initialized = True

        try:
            from openai import OpenAI

            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                logger.warning("OPENAI_API_KEY not set, judging disabled")
                return False

            self.client = OpenAI(api_key=api_key)

            # Import judge modules from hyphenated path
            modules = _import_judge_module()
            if modules is None:
                logger.warning("Could not import judge modules")
                return False

            self.judge_module, self.schemas_module = modules

            # Load few-shot examples
            train_path = PROJECT_ROOT / "judges/introspection-detection/data/train.jsonl"
            if train_path.exists():
                self.fewshot = self.judge_module.load_fewshot_examples(train_path)
                logger.info("Loaded %d few-shot examples", len(self.fewshot))

            self.example_cls = self.schemas_module.Example

            return True
        except ImportError as e:
            logger.warning("Judge dependencies not available: %s", e)
            return False

    def judge_with_retry(
        self,
        concept: str,
        was_injected: bool,
        response: str,
        config: SweepConfig,
    ) -> tuple[JudgeOutput | None, str | None]:
        """
        Run judge with exponential backoff retry.

        Returns:
            (JudgeOutput, None) on success, or (None, error_message) on failure
        """
        if self.client is None or self.example_cls is None:
            return None, "Judge not initialized"

        if self.judge_module is None or self.schemas_module is None:
            return None, "Judge modules not loaded"

        ExperimentConfig = self.schemas_module.ExperimentConfig
        Label = self.schemas_module.Label

        # Create Example object for the judge
        example = self.example_cls(
            id="temp",
            source_file="sweep",
            concept=concept,
            was_injected=was_injected,
            response=response,
            config=ExperimentConfig(
                layer=config.layer,
                strength=config.strength,
                prompt_version=config.prompt_version,
            ),
            label=Label(),
        )

        last_error = None
        for attempt in range(JUDGE_MAX_RETRIES):
            try:
                result = self.judge_module.judge_example(example, self.fewshot, self.client)
                return JudgeOutput(
                    answer=result.answer,
                    coherent=result.coherent,
                    detected_concept=result.detected_concept,
                    reasoning=result.reasoning,
                ), None
            except Exception as e:
                last_error = str(e)
                if attempt < JUDGE_MAX_RETRIES - 1:
                    delay = JUDGE_RETRY_BASE_DELAY * (2**attempt)
                    logger.warning(
                        "Judge attempt %d failed: %s. Retrying in %.1fs",
                        attempt + 1,
                        e,
                        delay,
                    )
                    time.sleep(delay)

        logger.error("Judge failed after %d attempts: %s", JUDGE_MAX_RETRIES, last_error)
        return None, last_error


def load_checkpoint(checkpoint_path: Path) -> Checkpoint | None:
    """Load checkpoint from file if it exists."""
    if not checkpoint_path.exists():
        return None

    try:
        with open(checkpoint_path) as f:
            data = json.load(f)
        cp = Checkpoint.model_validate(data)
        logger.info(
            "Resuming from checkpoint: layer_idx=%d, strength_idx=%d, trial_idx=%d, total=%d",
            cp.layer_idx,
            cp.strength_idx,
            cp.trial_idx,
            cp.total_completed,
        )
        return cp
    except Exception as e:
        logger.warning("Failed to load checkpoint: %s", e)
        return None


def save_checkpoint(checkpoint_path: Path, cp: Checkpoint) -> None:
    """Save checkpoint to file."""
    with open(checkpoint_path, "w") as f:
        json.dump(cp.model_dump(), f)


def upload_to_gcs(local_path: Path, gcs_path: str) -> bool:
    """Upload file to GCS using gsutil. Returns success."""
    import subprocess

    cmd = ["gsutil", "-q", "cp", str(local_path), gcs_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("GCS upload failed: %s", result.stderr)
        return False
    return True


def run_sweep_for_concept(
    model: HookedTransformer,
    concept: str,
    layers: list[int],
    strengths: list[float],
    trials_per_config: int,
    judge: JudgeClient,
    output_path: Path,
    checkpoint_path: Path,
    gcs_path: str | None,
    model_size: str,
) -> int:
    """
    Run full sweep for a single concept with checkpointing.

    Returns number of trials completed this run.
    """
    model_name = str(MODEL_CONFIGS[model_size]["name"])
    timestamp_base = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load or create checkpoint
    cp = load_checkpoint(checkpoint_path) or Checkpoint()
    trials_this_run = 0

    # Open output file in append mode (for resume)
    mode = "a" if cp.total_completed > 0 else "w"
    with open(output_path, mode) as f:
        # Cache for baseline means (recompute per layer)
        baseline_cache: dict[int, Tensor] = {}
        vector_cache: dict[int, tuple[Tensor, float]] = {}

        for layer_idx, layer in enumerate(layers):
            # Skip completed layers
            if layer_idx < cp.layer_idx:
                continue

            # Get or compute baseline and vector for this layer
            if layer not in baseline_cache:
                logger.info("Computing baseline for layer %d", layer)
                baseline_cache[layer] = compute_baseline_mean(
                    model,
                    layer=layer,
                    exclude_words=CONCEPTS,
                )
                vector = extract_concept_vector(
                    model,
                    target_word=concept,
                    layer=layer,
                    cached_baseline_mean=baseline_cache[layer],
                )
                vector_cache[layer] = (vector, vector.norm().item())

            vector, vector_norm = vector_cache[layer]

            for strength_idx, strength in enumerate(strengths):
                # Skip completed strengths (within current layer)
                if layer_idx == cp.layer_idx and strength_idx < cp.strength_idx:
                    continue

                effective_magnitude = strength * vector_norm
                logger.info(
                    "Layer %d | strength %.1f | eff_mag %.1f",
                    layer,
                    strength,
                    effective_magnitude,
                )

                for trial_idx in range(trials_per_config):
                    # Skip completed trials
                    if (
                        layer_idx == cp.layer_idx
                        and strength_idx == cp.strength_idx
                        and trial_idx < cp.trial_idx
                    ):
                        continue

                    # Determine which conditions to run
                    is_resume_point = (
                        layer_idx == cp.layer_idx
                        and strength_idx == cp.strength_idx
                        and trial_idx == cp.trial_idx
                    )
                    if is_resume_point:
                        # Resuming mid-trial
                        conditions = [True, False] if not cp.inject_done else [False]
                    else:
                        conditions = [True, False]

                    for was_injected in conditions:
                        trial_type = "injection" if was_injected else "control"
                        trial_id = (
                            f"{timestamp_base}_{concept}_{trial_type}"
                            f"_L{layer}_S{strength}_t{trial_idx}"
                        )

                        # Run trial
                        response = run_introspection_trial(
                            model,
                            concept_vector=vector,
                            layer=layer,
                            injection_strength=strength,
                            inject=was_injected,
                            prompt_version="v2",
                        )

                        config = SweepConfig(
                            model=model_name,
                            layer=layer,
                            strength=strength,
                            magnitude=effective_magnitude,
                            vector_norm=vector_norm,
                            prompt_version="v2",
                            trial=trial_idx,
                        )

                        # Judge with retry
                        judge_result, judge_error = judge.judge_with_retry(
                            concept, was_injected, response, config
                        )

                        # Build record
                        record = TrialRecord(
                            id=trial_id,
                            concept=concept,
                            was_injected=was_injected,
                            response=response,
                            config=config,
                            judge=judge_result,
                            judge_error=judge_error,
                        )

                        # Write immediately
                        f.write(record.model_dump_json() + "\n")
                        f.flush()

                        cp.total_completed += 1
                        trials_this_run += 1

                        # Update checkpoint
                        cp.layer_idx = layer_idx
                        cp.strength_idx = strength_idx
                        cp.trial_idx = trial_idx
                        cp.inject_done = was_injected
                        save_checkpoint(checkpoint_path, cp)

                        # Periodic upload
                        if gcs_path and cp.total_completed % GCS_UPLOAD_INTERVAL == 0:
                            logger.info("Periodic upload at %d trials", cp.total_completed)
                            upload_to_gcs(output_path, gcs_path)

                    # Reset inject_done after completing both conditions
                    cp.inject_done = False

                # After completing all trials for this strength, reset trial counter
                cp.trial_idx = 0

            # After completing all strengths for this layer, reset strength counter
            cp.strength_idx = 0

    return trials_this_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run introspection sweep")
    parser.add_argument(
        "--concept",
        choices=CONCEPTS,
        help="Concept to sweep (overrides BATCH_TASK_INDEX)",
    )
    parser.add_argument(
        "--model",
        choices=["3b", "7b"],
        default="3b",
        help="Model size (default: 3b)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=DEFAULT_TRIALS_PER_CONFIG,
        help=f"Trials per config (default: {DEFAULT_TRIALS_PER_CONFIG})",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run locally (write to data/ instead of GCS)",
    )
    parser.add_argument(
        "--experiment-id",
        default=None,
        help="Experiment ID for output path (default: date-based)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh, ignore existing checkpoint",
    )
    args = parser.parse_args()

    # Determine concept from args or BATCH_TASK_INDEX
    if args.concept:
        concept = args.concept
    else:
        task_index = int(os.environ.get("BATCH_TASK_INDEX", 0))
        concept = CONCEPTS[task_index]

    # Experiment ID for organization
    experiment_id = args.experiment_id or datetime.now().strftime("%Y-%m-%d-sweep")

    logger.info("=" * 60)
    logger.info("Introspection Sweep")
    logger.info("=" * 60)
    logger.info("Concept: %s", concept)
    logger.info("Model: %s", args.model)
    logger.info("Trials per config: %d", args.trials)
    logger.info("Experiment ID: %s", experiment_id)

    # Load model
    model_config = MODEL_CONFIGS[args.model]
    model_name = str(model_config["name"])
    logger.info("Loading model: %s", model_name)
    model: HookedTransformer = load_model(
        model_name=model_name,
        dtype=torch.bfloat16,
    )

    # Compute layers for this model
    layers = get_layers_for_model(model.cfg.n_layers)
    logger.info("Layers: %s", layers)
    logger.info("Strengths: %s", DEFAULT_STRENGTHS)

    total_trials = len(layers) * len(DEFAULT_STRENGTHS) * args.trials * 2
    logger.info("Total trials planned: %d", total_trials)

    # Initialize judge
    judge = JudgeClient()
    if judge.initialize():
        logger.info("Judge initialized")
    else:
        logger.warning("Running without judge")

    # Set up paths
    if args.local:
        output_dir = PROJECT_ROOT / "data" / "sweeps" / experiment_id / args.model
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{concept}.jsonl"
        checkpoint_path = output_dir / f"{concept}.checkpoint.json"
        gcs_path = None
    else:
        output_dir = Path("/tmp/sweep_output")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{concept}.jsonl"
        checkpoint_path = output_dir / f"{concept}.checkpoint.json"
        bucket = os.environ.get("GCS_BUCKET", "open-introspection-sweeps")
        gcs_path = f"gs://{bucket}/{experiment_id}/{args.model}/{concept}.jsonl"

    # Clear checkpoint if requested
    if args.no_resume and checkpoint_path.exists():
        checkpoint_path.unlink()
        if output_path.exists():
            output_path.unlink()
        logger.info("Cleared existing checkpoint and output")

    logger.info("Output: %s", output_path)
    if gcs_path:
        logger.info("GCS destination: %s", gcs_path)

    # Run sweep
    completed = run_sweep_for_concept(
        model=model,
        concept=concept,
        layers=layers,
        strengths=DEFAULT_STRENGTHS,
        trials_per_config=args.trials,
        judge=judge,
        output_path=output_path,
        checkpoint_path=checkpoint_path,
        gcs_path=gcs_path,
        model_size=args.model,
    )

    logger.info("Completed %d trials this run", completed)

    # Final upload
    if gcs_path:
        logger.info("Final upload to GCS")
        if upload_to_gcs(output_path, gcs_path):
            logger.info("Upload complete: %s", gcs_path)
        # Also upload checkpoint for debugging
        upload_to_gcs(checkpoint_path, gcs_path.replace(".jsonl", ".checkpoint.json"))

    # Clean up checkpoint on successful completion
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        logger.info("Removed checkpoint (sweep complete)")

    logger.info("Done!")


if __name__ == "__main__":
    main()
