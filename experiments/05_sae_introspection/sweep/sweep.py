"""Core sweep logic: trial generation with SAE feature capture.

Runs on GPU (Modal). Entry point: run_sweep(request) -> dict.

For each trial:
1. Inject concept vector at specified layer (if injection trial)
2. Capture SAE features at layer 22 (always)
3. Store response + sparse features
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import torch

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
SWEEP_DIR = Path(__file__).parent
sys.path.insert(0, str(SWEEP_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from config import (  # noqa: E402
    CONCEPTS,
    GEMMA_MODEL,
    SAE_ID,
    SAE_LAYER,
    SAE_RELEASE,
    SPARSE_THRESHOLD,
    SparseFeatures,
    SweepConfig,
    SweepRequest,
    TrialRecord,
)

if TYPE_CHECKING:
    from sae_lens import SAE
    from torch import Tensor
    from transformer_lens import HookedTransformer
    from transformer_lens.hook_points import HookPoint

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Generation parameters
TEMPERATURE = 1.0
MAX_NEW_TOKENS = 200


# ── Utilities ────────────────────────────────────────────────────────────────


def to_sparse(dense: Tensor, threshold: float = SPARSE_THRESHOLD) -> list[SparseFeatures]:
    """Convert dense SAE features [n_tokens, n_features] to sparse format.

    Args:
        dense: Dense feature activations, shape (n_tokens, n_features).
        threshold: Only store features above this activation value.

    Returns:
        List of SparseFeatures, one per token.
    """
    result = []
    for t in range(dense.shape[0]):
        token_feats = dense[t]
        mask = token_feats > threshold
        indices = mask.nonzero(as_tuple=True)[0].to(torch.int32).tolist()
        values = token_feats[mask].to(torch.float32).tolist()
        result.append(SparseFeatures(indices=indices, values=values))
    return result


def get_end_of_turn_id(model: HookedTransformer) -> int:
    """Get Gemma's <end_of_turn> token ID."""
    token_ids = model.tokenizer.encode("<end_of_turn>", add_special_tokens=False)
    return int(token_ids[0])


# ── Model + SAE Loading ─────────────────────────────────────────────────────


def load_model_and_sae(
    device: str = "cuda",
) -> tuple[HookedTransformer, SAE]:
    """Load Gemma 3 4B-IT and GemmaScope 2 SAE."""
    from sae_lens import SAE, HookedSAETransformer

    logger.info("Loading model: %s", GEMMA_MODEL)
    model = HookedSAETransformer.from_pretrained(
        GEMMA_MODEL,
        device=device,
        dtype=torch.bfloat16,
    )
    torch.set_grad_enabled(False)
    model.eval()

    logger.info("Loading SAE: %s / %s", SAE_RELEASE, SAE_ID)
    sae = SAE.from_pretrained(
        release=SAE_RELEASE,
        sae_id=SAE_ID,
        device=device,
    )
    sae = sae.to(dtype=torch.bfloat16)

    logger.info("Model: %d layers, d_model=%d", model.cfg.n_layers, model.cfg.d_model)
    return model, sae


# ── Trial Execution ──────────────────────────────────────────────────────────


def _injection_hook(
    activation: Tensor,
    hook: HookPoint,
    concept_vector: Tensor,
    strength: float,
    prompt_len: int,
    inject_style: str,
) -> Tensor:
    """Add concept vector to residual stream."""
    if inject_style == "generation":
        if activation.shape[1] != prompt_len:
            activation[:, -1, :] += strength * concept_vector
    else:
        raise ValueError("non generation style injection not supported!")
    return activation


def _sae_capture_hook(
    activation: Tensor,
    hook: HookPoint,
    sae: SAE,
    storage: list[Tensor],
) -> Tensor:
    """Encode residual stream with SAE, store features for last token."""
    features = sae.encode(activation[:, -1:, :])
    storage.append(features.squeeze(1).float().cpu())
    return activation


def run_trial(
    model: HookedTransformer,
    sae: SAE,
    tokens: Tensor,
    prompt_len: int,
    concept_vector: Tensor | None,
    injection_layer: int,
    strength: float,
    inject: bool,
    inject_style: str,
    eos_token_id: int,
) -> tuple[str, list[SparseFeatures]]:
    """Run a single trial with injection + SAE capture.

    Returns:
        (response_text, sparse_features)
    """
    sae_storage: list[Tensor] = []

    hooks: list[tuple[str, object]] = []

    # SAE capture hook (always active)
    capture_fn = partial(_sae_capture_hook, sae=sae, storage=sae_storage)
    hooks.append((f"blocks.{SAE_LAYER}.hook_resid_post", capture_fn))

    # Injection hook (only if injecting)
    if inject and concept_vector is not None:
        inject_fn = partial(
            _injection_hook,
            concept_vector=concept_vector,
            strength=strength,
            prompt_len=prompt_len,
            inject_style=inject_style,
        )
        hooks.append((f"blocks.{injection_layer}.hook_resid_post", inject_fn))

    with model.hooks(hooks):
        output = model.generate(
            tokens,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            stop_at_eos=True,
            eos_token_id=eos_token_id,
            prepend_bos=False,
        )

    # Decode response
    generated = output[0, prompt_len:]
    response: str = model.tokenizer.decode(
        generated, skip_special_tokens=True
    ).strip()

    # Convert dense captures to sparse
    if sae_storage:
        dense = torch.cat(sae_storage, dim=0)  # (n_tokens, n_features)
        # Trim to actual generated tokens (stop_at_eos may produce fewer)
        n_generated = generated.shape[0]
        dense = dense[:n_generated]
        sparse = to_sparse(dense)
    else:
        sparse = []

    return response, sparse


# ── Sweep Orchestration ──────────────────────────────────────────────────────


def _run_sweep_for_concept(
    model: HookedTransformer,
    sae: SAE,
    request: SweepRequest,
    output_path: Path,
) -> int:
    """Run all trials for a single concept. Returns trials completed this run."""
    from src.open_introspection.concept_extraction import (
        compute_baseline_mean,
        extract_concept_vector,
    )
    from src.open_introspection.introspection import PROMPT_MESSAGES
    from src.open_introspection.model import tokenize_chat

    concept = request.concept
    eos_token_id = get_end_of_turn_id(model)

    # Tokenize prompt once (same for all trials)
    messages = PROMPT_MESSAGES.get(request.prompt_version, PROMPT_MESSAGES["v2"])
    token_ids = tokenize_chat(model, messages, add_generation_prompt=True)
    tokens = torch.tensor([token_ids], device=model.cfg.device)
    prompt_len = tokens.shape[1]

    # Load existing trial IDs for resume
    existing_ids: set[str] = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    existing_ids.add(json.loads(line)["id"])
        if existing_ids:
            logger.info("Resuming: %d trials already completed", len(existing_ids))

    # Concept vector cache per injection layer
    vector_cache: dict[int, tuple[Tensor, float]] = {}

    def get_vector(layer: int) -> tuple[Tensor, float]:
        if layer not in vector_cache:
            logger.info("Computing concept vector for layer %d", layer)
            baseline = compute_baseline_mean(model, layer=layer, exclude_words=CONCEPTS)
            vector = extract_concept_vector(
                model, target_word=concept, layer=layer, cached_baseline_mean=baseline
            )
            vector_cache[layer] = (vector, vector.norm().item())
        return vector_cache[layer]

    # Build trial configs: injection trials
    configs: list[tuple[int, float, int, bool]] = []
    for layer in request.injection_layers:
        for strength in request.strengths:
            for trial_idx in range(request.trials):
                configs.append((layer, strength, trial_idx, True))

    # Control trials: no injection, use layer 0 / strength 0 as placeholders
    for trial_idx in range(request.control_trials):
        configs.append((0, 0.0, trial_idx, False))

    logger.info("Total trials planned: %d", len(configs))

    trials_this_run = 0

    with open(output_path, "a") as f:
        for layer, strength, trial_idx, was_injected in configs:
            trial_type = "injection" if was_injected else "control"
            if was_injected:
                trial_id = f"{concept}_{trial_type}_L{layer}_S{strength}_t{trial_idx}"
            else:
                trial_id = f"{concept}_{trial_type}_t{trial_idx}"

            if trial_id in existing_ids:
                continue

            # Get concept vector (only needed for injection, but compute for control
            # to keep config consistent — use first injection layer)
            if was_injected:
                vector, vector_norm = get_vector(layer)
            else:
                # For control, still record vector norm from first injection layer
                ref_layer = request.injection_layers[0]
                _, vector_norm = get_vector(ref_layer)
                vector = None

            response, sparse_features = run_trial(
                model=model,
                sae=sae,
                tokens=tokens,
                prompt_len=prompt_len,
                concept_vector=vector,
                injection_layer=layer,
                strength=strength,
                inject=was_injected,
                inject_style=request.inject_style,
                eos_token_id=eos_token_id,
            )

            config = SweepConfig(
                model=GEMMA_MODEL,
                injection_layer=layer,
                strength=strength,
                magnitude=strength * vector_norm if was_injected else 0.0,
                vector_norm=vector_norm,
                prompt_version=request.prompt_version,
                inject_style=request.inject_style,
                trial=trial_idx,
                sae_release=SAE_RELEASE,
                sae_id=SAE_ID,
                sae_layer=SAE_LAYER,
            )

            record = TrialRecord(
                id=trial_id,
                timestamp=datetime.now().isoformat(),
                concept=concept,
                was_injected=was_injected,
                response=response,
                config=config,
                sae_features=sparse_features,
            )
            f.write(record.model_dump_json() + "\n")
            f.flush()

            existing_ids.add(trial_id)
            trials_this_run += 1

            if trials_this_run % 10 == 0:
                logger.info(
                    "Progress: %d/%d trials | last: %s",
                    trials_this_run,
                    len(configs) - len(existing_ids) + trials_this_run,
                    trial_id,
                )

    return trials_this_run


def run_sweep(request: SweepRequest) -> dict:
    """Main entry point called by modal_app.py.

    Args:
        request: Sweep configuration for one concept.

    Returns:
        dict with status, concept, trials_completed, local_output_path.
    """
    logger.info("=" * 60)
    logger.info("SAE Feature Discovery Sweep")
    logger.info("=" * 60)
    logger.info("Concept: %s", request.concept)
    logger.info("Prompt: %s", request.prompt_version)
    logger.info("Inject style: %s", request.inject_style)
    logger.info("Injection layers: %s", request.injection_layers)
    logger.info("Strengths: %s", request.strengths)
    logger.info("Trials per config: %d", request.trials)
    logger.info("Control trials: %d", request.control_trials)
    logger.info("GCS: %s", request.gcs_path)

    n_injection = len(request.injection_layers) * len(request.strengths) * request.trials
    n_total = n_injection + request.control_trials
    logger.info(
        "Total trials: %d (%d injection + %d control)",
        n_total, n_injection, request.control_trials,
    )

    model, sae = load_model_and_sae()

    output_dir = Path("/tmp/sweep_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{request.concept}.jsonl"

    completed = _run_sweep_for_concept(model, sae, request, output_path)

    logger.info("Completed %d trials", completed)

    return {
        "status": "success",
        "concept": request.concept,
        "trials_completed": completed,
        "local_output_path": str(output_path),
    }
