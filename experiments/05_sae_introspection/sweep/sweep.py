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
import random
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
    MONITORING_SYSTEM_PROMPT,
    SAE_ID,
    SAE_LAYER,
    SAE_RELEASE,
    SPARSE_THRESHOLD,
    TASK_POOL,
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


def _precompute_vectors(
    model: HookedTransformer,
    concept: str,
    layers: list[int],
) -> dict[int, tuple[Tensor, float]]:
    """Precompute concept vectors for all layers in a single pass per word.

    Instead of calling compute_baseline_mean once per layer (each running all
    baseline words through the model), we run each word once and extract
    activations at all layers from the cache. This is ~4x faster for 4 layers.

    Returns:
        Dict mapping layer -> (concept_vector, vector_norm).
    """
    from src.open_introspection.concept_extraction import DEFAULT_BASELINE_WORDS

    # Filter out concept words from baseline
    exclude_set = {w.lower() for w in CONCEPTS}
    baseline_words = [w for w in DEFAULT_BASELINE_WORDS if w.lower() not in exclude_set]

    hook_names = [f"blocks.{layer}.hook_resid_post" for layer in layers]

    logger.info(
        "Precomputing concept vectors for %d layers from %d baseline words (single pass)",
        len(layers), len(baseline_words),
    )

    # Accumulate baseline activations: {layer: list of (d_model,) tensors}
    baseline_acts: dict[int, list[Tensor]] = {layer: [] for layer in layers}

    for word in baseline_words:
        prompt = f"Tell me about {word}."
        toks = model.to_tokens(prompt)
        _, cache = model.run_with_cache(toks, names_filter=hook_names)
        for layer, hook_name in zip(layers, hook_names, strict=True):
            baseline_acts[layer].append(cache[hook_name][0, -1, :])

    # Compute baseline means
    baseline_means: dict[int, Tensor] = {}
    for layer in layers:
        baseline_means[layer] = torch.stack(baseline_acts[layer]).mean(dim=0)

    # Compute concept activation (single forward pass for all layers)
    concept_prompt = f"Tell me about {concept}."
    concept_toks = model.to_tokens(concept_prompt)
    _, concept_cache = model.run_with_cache(concept_toks, names_filter=hook_names)

    # Build vectors
    vectors: dict[int, tuple[Tensor, float]] = {}
    for layer, hook_name in zip(layers, hook_names, strict=True):
        concept_act = concept_cache[hook_name][0, -1, :]
        vector = concept_act - baseline_means[layer]
        norm = vector.norm().item()
        vectors[layer] = (vector, norm)
        logger.info("Layer %d: vector norm = %.2f", layer, norm)

    return vectors


def _build_prompt(
    model: HookedTransformer,
    task: str,
) -> tuple[Tensor, int]:
    """Build monitoring prompt with a specific task. Returns (tokens, prompt_len)."""
    from src.open_introspection.model import tokenize_chat

    messages = [
        {"role": "system", "content": MONITORING_SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]
    token_ids = tokenize_chat(model, messages, add_generation_prompt=True)
    tokens = torch.tensor([token_ids], device=model.cfg.device)
    return tokens, tokens.shape[1]


def _run_sweep_for_concept(
    model: HookedTransformer,
    sae: SAE,
    request: SweepRequest,
    output_path: Path,
) -> int:
    """Run all trials for a single concept. Returns trials completed this run."""
    concept = request.concept
    eos_token_id = get_end_of_turn_id(model)
    rng = random.Random(42)  # Deterministic task sampling for reproducibility

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

    # Precompute all concept vectors upfront (single pass per baseline word)
    vector_cache = _precompute_vectors(model, concept, request.injection_layers)

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

            # Sample a random task for this trial
            task = rng.choice(TASK_POOL)

            # Build prompt with this task (varies per trial for diversity)
            tokens, prompt_len = _build_prompt(model, task)

            # Get concept vector from precomputed cache
            if was_injected:
                vector, vector_norm = vector_cache[layer]
            else:
                # For control, still record vector norm from first injection layer
                ref_layer = request.injection_layers[0]
                _, vector_norm = vector_cache[ref_layer]
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
                task=task,
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
