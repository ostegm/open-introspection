#!/usr/bin/env python3
"""Oracle vs Self-Report sweep: compare activation oracle detection with model
self-report for identifying injected concept vectors.

For each trial:
1. Generate self-report (200 tokens) with concept injection active
2. Capture activations at 2 layers (injection+2, injection+4) during generation
3. Query oracle with 3 questions at each capture layer
4. Write TrialRecord to JSONL (resume-safe via trial IDs)

Usage:
    # Full sweep on Modal:
    uv run modal run experiments/06_actvation_oracles/modal_app.py

    # Local test (1 trial):
    uv run python experiments/06_actvation_oracles/run_oracle_sweep.py --trials 1 --concepts ocean
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping

import torch
import torch._dynamo as dynamo
from peft import LoraConfig
from pydantic import BaseModel, ConfigDict, model_validator
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import (
    BASELINE_WORDS,
    CAPTURE_OFFSETS,
    CONCEPTS,
    INJECTION_LAYERS,
    MODEL_NAME,
    MONITORING_SYSTEM,
    MONITORING_USER,
    ORACLE_INJECT_LAYER,
    ORACLE_LORA,
    ORACLE_QUESTIONS,
    ORACLE_STEER_COEFF,
    ORACLE_TOKENS,
    SEGMENT_SIZE,
    SELF_REPORT_TOKENS,
    STRENGTHS,
    TRIALS_PER_CELL,
    OracleResponse,
    OracleSweepConfig,
    SweepRequest,
    TrialRecord,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================
# Oracle plumbing (from run_oracle_v2.py — shared infrastructure)
# ============================================================

SPECIAL_TOKEN = " ?"


def get_submodule(model: Any, layer: int) -> torch.nn.Module:
    try:
        return model.base_model.model.model.layers[layer]
    except AttributeError:
        return model.model.layers[layer]


def get_n_layers(model: Any) -> int:
    cfg = model.config
    return cfg.text_config.num_hidden_layers if hasattr(cfg, "text_config") else cfg.num_hidden_layers


@contextlib.contextmanager
def add_hook(module: torch.nn.Module, hook: Callable) -> Any:
    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


class TrainingDataPoint(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    datapoint_type: str
    input_ids: list[int]
    labels: list[int]
    layer: int
    steering_vectors: torch.Tensor | None
    positions: list[int]
    feature_idx: int
    target_output: str
    target_input_ids: list[int] | None
    target_positions: list[int] | None
    ds_label: str | None
    meta_info: Mapping[str, Any] = {}

    @model_validator(mode="after")
    def _check(cls, values: Any) -> Any:  # noqa: N805
        sv = values.steering_vectors
        if sv is not None:
            if len(values.positions) != sv.shape[0]:
                raise ValueError("positions/steering_vectors length mismatch")
        else:
            if values.target_positions is None or values.target_input_ids is None:
                raise ValueError("target_* required when steering_vectors is None")
        return values


class BatchData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    input_ids: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor
    steering_vectors: list[torch.Tensor]
    positions: list[list[int]]
    feature_indices: list[int]


class FeatureResult(BaseModel):
    feature_idx: int
    api_response: str
    prompt: str
    meta_info: Mapping[str, Any] = {}


def find_pattern_in_tokens(token_ids, special_token_str, num_positions, tokenizer):
    if not isinstance(token_ids, list):
        token_ids = list(token_ids)
    special_ids = tokenizer.encode(special_token_str, add_special_tokens=False)
    assert len(special_ids) >= 1
    special_id = special_ids[-1]
    positions = [i for i in range(len(token_ids)) if token_ids[i] == special_id]
    positions = positions[:num_positions]
    assert len(positions) == num_positions, (
        f"Expected {num_positions} positions for token {special_id} "
        f"('{special_token_str}'), found {len(positions)} in {len(token_ids)} tokens"
    )
    return positions


def get_introspection_prefix(sae_layer, num_positions):
    return f"Layer: {sae_layer}\n" + SPECIAL_TOKEN * num_positions + " \n"


def create_training_datapoint(prompt, layer, num_positions, tokenizer, acts_BD,
                              target_response="N/A", feature_idx=-1):
    prefix = get_introspection_prefix(layer, num_positions)
    full_prompt = prefix + prompt
    input_messages = [{"role": "user", "content": full_prompt}]
    try:
        input_ids = tokenizer.apply_chat_template(
            input_messages, tokenize=True, add_generation_prompt=True,
            return_tensors=None, padding=False, enable_thinking=False,
        )
    except TypeError:
        input_ids = tokenizer.apply_chat_template(
            input_messages, tokenize=True, add_generation_prompt=True,
            return_tensors=None, padding=False,
        )
    full_messages = input_messages + [{"role": "assistant", "content": target_response}]
    try:
        full_ids = tokenizer.apply_chat_template(
            full_messages, tokenize=True, add_generation_prompt=False,
            return_tensors=None, padding=False, enable_thinking=False,
        )
    except TypeError:
        full_ids = tokenizer.apply_chat_template(
            full_messages, tokenize=True, add_generation_prompt=False,
            return_tensors=None, padding=False,
        )
    # Handle BatchEncoding dicts
    if hasattr(input_ids, "input_ids"):
        input_ids = input_ids["input_ids"]
    if hasattr(full_ids, "input_ids"):
        full_ids = full_ids["input_ids"]
    if not isinstance(input_ids, list):
        input_ids = list(input_ids)
    if not isinstance(full_ids, list):
        full_ids = list(full_ids)
    labels = full_ids.copy()
    for i in range(len(input_ids)):
        labels[i] = -100
    positions = find_pattern_in_tokens(full_ids, SPECIAL_TOKEN, num_positions, tokenizer)
    if acts_BD is not None:
        acts_BD = acts_BD.cpu().clone().detach()
    return TrainingDataPoint(
        datapoint_type="custom", input_ids=full_ids, labels=labels,
        layer=layer, steering_vectors=acts_BD, positions=positions,
        feature_idx=feature_idx, target_output=target_response,
        target_input_ids=None, target_positions=None, ds_label=None,
    )


def get_prompt_tokens_only(dp):
    prompt_ids, prompt_labels = [], []
    for i in range(len(dp.input_ids)):
        if dp.labels[i] != -100:
            break
        prompt_ids.append(dp.input_ids[i])
        prompt_labels.append(dp.labels[i])
    new = dp.model_copy()
    new.input_ids = prompt_ids
    new.labels = prompt_labels
    return new


def construct_batch(training_data, tokenizer, device):
    max_length = max(len(dp.input_ids) for dp in training_data)
    batch_tokens, batch_labels, batch_masks = [], [], []
    batch_positions, batch_vectors, batch_indices = [], [], []
    for dp in training_data:
        pad_len = max_length - len(dp.input_ids)
        pad_tok = [tokenizer.pad_token_id] * pad_len
        ids = torch.tensor(pad_tok + dp.input_ids, dtype=torch.long, device=device)
        labs = torch.tensor([-100] * pad_len + dp.labels, dtype=torch.long, device=device)
        mask = torch.ones_like(ids, dtype=torch.bool)
        mask[:pad_len] = False
        batch_tokens.append(ids)
        batch_labels.append(labs)
        batch_masks.append(mask)
        batch_positions.append([p + pad_len for p in dp.positions])
        batch_vectors.append(dp.steering_vectors.to(device) if dp.steering_vectors is not None else None)
        batch_indices.append(dp.feature_idx)
    return BatchData(
        input_ids=torch.stack(batch_tokens), labels=torch.stack(batch_labels),
        attention_mask=torch.stack(batch_masks), steering_vectors=batch_vectors,
        positions=batch_positions, feature_indices=batch_indices,
    )


def get_hf_activation_steering_hook(vectors, positions, steering_coefficient, device, dtype):
    assert len(vectors) == len(positions)
    B = len(vectors)
    normed_list = [torch.nn.functional.normalize(v, dim=-1).detach() for v in vectors]

    def hook_fn(module, _input, output):
        if isinstance(output, tuple):
            resid_BLD, *rest = output
            output_is_tuple = True
        else:
            resid_BLD = output
            output_is_tuple = False
        _B_actual, L, _d = resid_BLD.shape
        if L <= 1:
            return (resid_BLD, *rest) if output_is_tuple else resid_BLD
        for b in range(B):
            pos_b = torch.tensor(positions[b], dtype=torch.long, device=device)
            orig_KD = resid_BLD[b, pos_b, :]
            norms_K1 = orig_KD.norm(dim=-1, keepdim=True)
            steered_KD = (normed_list[b] * norms_K1 * steering_coefficient).to(dtype)
            resid_BLD[b, pos_b, :] = steered_KD.detach() + orig_KD
        return (resid_BLD, *rest) if output_is_tuple else resid_BLD

    return hook_fn


@dynamo.disable
@torch.no_grad()
def eval_features_batch(eval_batch, model, submodule, tokenizer, device, dtype,
                        steering_coefficient, generation_kwargs):
    hook_fn = get_hf_activation_steering_hook(
        vectors=eval_batch.steering_vectors, positions=eval_batch.positions,
        steering_coefficient=steering_coefficient, device=device, dtype=dtype,
    )
    tokenized = {"input_ids": eval_batch.input_ids, "attention_mask": eval_batch.attention_mask}
    with add_hook(submodule, hook_fn):
        output_ids = model.generate(**tokenized, **generation_kwargs)
    gen_tokens = output_ids[:, eval_batch.input_ids.shape[1]:]
    decoded = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
    return [
        FeatureResult(feature_idx=eval_batch.feature_indices[i],
                      api_response=decoded[i], prompt="")
        for i in range(len(decoded))
    ]


# ============================================================
# High-level helpers
# ============================================================


def format_chat(tokenizer, system, user):
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    if system:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
        except (TypeError, Exception):
            pass
        messages = [{"role": "user", "content": f"{system}\n\n{user}"}]
    else:
        messages = [{"role": "user", "content": user}]
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def get_activation_at_layer(model, tokenizer, prompt_str, layer, device):
    """Single forward pass, capture last token activation at layer."""
    inputs = tokenizer(prompt_str, return_tensors="pt", add_special_tokens=False).to(device)
    sub = get_submodule(model, layer)
    captured: dict[str, torch.Tensor] = {}

    class _Stop(Exception):
        pass

    def hook(module, inp, out):
        captured["act"] = (out[0] if isinstance(out, tuple) else out).detach()
        raise _Stop()

    handle = sub.register_forward_hook(hook)
    model.disable_adapters()
    try:
        with torch.no_grad():
            model(**inputs)
    except _Stop:
        pass
    finally:
        handle.remove()
        model.enable_adapters()
    return captured["act"][0, -1, :]


def extract_concept_vector(model, tokenizer, concept, baselines, layer, device):
    concept_prompt = format_chat(tokenizer, "", f"Tell me about {concept}")
    concept_act = get_activation_at_layer(model, tokenizer, concept_prompt, layer, device)
    filtered = [w for w in baselines if w.lower() != concept.lower()]
    baseline_acts = []
    for word in filtered:
        prompt = format_chat(tokenizer, "", f"Tell me about {word}")
        baseline_acts.append(get_activation_at_layer(model, tokenizer, prompt, layer, device))
    baseline_mean = torch.stack(baseline_acts).mean(dim=0)
    return concept_act - baseline_mean


def generate_and_capture_multilayer(
    model, tokenizer, prompt, capture_layers, device,
    injection_layer=None, concept_vector=None, strength=0.0,
    max_new_tokens=SELF_REPORT_TOKENS,
):
    """Generate tokens with optional injection, capturing activations at MULTIPLE layers.

    Returns (generated_text, {layer: activations_tensor [n_gen, d_model]}).
    """
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    prompt_len = inputs["input_ids"].shape[1]

    inj_sub = get_submodule(model, injection_layer) if injection_layer is not None else None

    gen_activations: dict[int, list[torch.Tensor]] = {layer: [] for layer in capture_layers}
    handles = []

    # Injection hook: only on generation steps (shape[1]==1)
    if inj_sub is not None and strength > 0 and concept_vector is not None:
        def inj_hook(module, inp, output):
            resid = output[0] if isinstance(output, tuple) else output
            if resid.shape[1] == 1:
                resid[:, 0, :] += strength * concept_vector
            if isinstance(output, tuple):
                return (resid, *output[1:])
            return resid

        handles.append(inj_sub.register_forward_hook(inj_hook))

    # Capture hooks: one per capture layer
    for cap_layer in capture_layers:
        cap_sub = get_submodule(model, cap_layer)
        act_list = gen_activations[cap_layer]

        def make_cap_hook(act_store):
            def cap_hook(module, inp, output):
                resid = output[0] if isinstance(output, tuple) else output
                if resid.shape[1] == 1:
                    act_store.append(resid[0, 0, :].detach().clone())
            return cap_hook

        handles.append(cap_sub.register_forward_hook(make_cap_hook(act_list)))

    model.disable_adapters()
    try:
        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=True, temperature=1.0,
            )
    finally:
        for h in handles:
            h.remove()
        model.enable_adapters()

    generated_text = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True).strip()

    result_acts: dict[int, torch.Tensor] = {}
    for layer, acts in gen_activations.items():
        if acts:
            result_acts[layer] = torch.stack(acts)
        else:
            result_acts[layer] = torch.zeros(0, model.config.hidden_size, device=device)

    return generated_text, result_acts


def query_oracle(model, tokenizer, oracle_adapter_name, activations, oracle_prompt,
                 device, dtype, capture_layer_pct=50,
                 oracle_inject_layer=ORACLE_INJECT_LAYER,
                 steering_coeff=ORACLE_STEER_COEFF, max_new_tokens=ORACLE_TOKENS):
    num_positions = activations.shape[0]
    dp = create_training_datapoint(
        prompt=oracle_prompt, layer=capture_layer_pct,
        num_positions=num_positions, tokenizer=tokenizer, acts_BD=activations,
    )
    dp = get_prompt_tokens_only(dp)
    batch = construct_batch([dp], tokenizer, device)
    oracle_sub = get_submodule(model, oracle_inject_layer)
    model.set_adapter(oracle_adapter_name)
    results = eval_features_batch(
        eval_batch=batch, model=model, submodule=oracle_sub,
        tokenizer=tokenizer, device=device, dtype=dtype,
        steering_coefficient=steering_coeff,
        generation_kwargs={"do_sample": False, "max_new_tokens": max_new_tokens},
    )
    return results[0].api_response


# ============================================================
# Sweep logic
# ============================================================


def load_existing_ids(output_path: Path) -> set[str]:
    """Load trial IDs already written to JSONL for resume."""
    existing: set[str] = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    existing.add(record["id"])
    return existing


def run_trial(
    model, tokenizer, oracle_adapter_name, device, dtype,
    concept: str, injection_layer: int, capture_layers: list[int],
    concept_vector: torch.Tensor, vector_norm: float,
    strength: float, trial_idx: int, was_injected: bool,
) -> TrialRecord:
    """Run a single trial: self-report + oracle queries."""
    trial_type = "injection" if was_injected else "control"
    trial_id = f"{concept}_{trial_type}_L{injection_layer}_S{strength}_t{trial_idx}"

    # Build monitoring prompt
    prompt = format_chat(tokenizer, MONITORING_SYSTEM, MONITORING_USER)

    # Generate self-report with optional injection + capture at multiple layers
    self_report, captured_acts = generate_and_capture_multilayer(
        model, tokenizer, prompt, capture_layers, device,
        injection_layer=injection_layer if was_injected else None,
        concept_vector=concept_vector,
        strength=strength,
        max_new_tokens=SELF_REPORT_TOKENS,
    )

    # Query oracle at each capture layer with each question
    oracle_responses: list[OracleResponse] = []
    for cap_layer in capture_layers:
        acts = captured_acts[cap_layer]
        if acts.shape[0] == 0:
            logger.warning("No activations captured at layer %d for %s", cap_layer, trial_id)
            continue

        segment = acts[-SEGMENT_SIZE:]
        # Compute layer percentage for oracle prefix
        cap_layer_pct = int(100 * cap_layer / get_n_layers(model))

        for q_type, q_text in ORACLE_QUESTIONS.items():
            oracle_resp = query_oracle(
                model, tokenizer, oracle_adapter_name,
                segment, q_text, device, dtype,
                capture_layer_pct=cap_layer_pct,
            )
            oracle_responses.append(OracleResponse(
                capture_layer=cap_layer,
                question_type=q_type,
                question=q_text,
                response=oracle_resp,
            ))

    config = OracleSweepConfig(
        model=MODEL_NAME,
        injection_layer=injection_layer,
        strength=strength,
        magnitude=strength * vector_norm,
        vector_norm=vector_norm,
        prompt_version="v2",
        inject_style="generation",
        trial=trial_idx,
        capture_layers=capture_layers,
    )

    return TrialRecord(
        id=trial_id,
        timestamp=datetime.now().isoformat(),
        concept=concept,
        was_injected=was_injected,
        response=self_report,
        config=config,
        oracle_responses=oracle_responses,
    )


def run_sweep(request: SweepRequest) -> dict:
    """Run the full oracle vs self-report sweep."""
    t_start = time.time()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    dtype = torch.bfloat16

    logger.info("=" * 60)
    logger.info("Oracle vs Self-Report Sweep")
    logger.info("=" * 60)
    logger.info("Model: %s", MODEL_NAME)
    logger.info("Device: %s", device)
    logger.info("Concepts: %s", request.concepts)
    logger.info("Strengths: %s", request.strengths)
    logger.info("Injection layers: %s", request.injection_layers)
    logger.info("Trials per cell: %d", request.trials)
    logger.info("Experiment ID: %s", request.experiment_id)

    # Compute total trials
    n_cells = len(request.concepts) * len(request.strengths) * len(request.injection_layers) * 2
    total_trials = n_cells * request.trials
    logger.info("Total trials planned: %d (%d cells x %d trials)", total_trials, n_cells, request.trials)

    # ---- Load model + oracle ----
    logger.info("Loading model...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.padding_side = "left"
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype)
    model.eval()
    dummy_config = LoraConfig(r=1, target_modules=["q_proj"])
    model.add_adapter(dummy_config, adapter_name="default")
    oracle_adapter_name = ORACLE_LORA.replace("/", "_").replace(".", "_")
    model.load_adapter(ORACLE_LORA, adapter_name=oracle_adapter_name, is_trainable=False)
    model = model.to(device)
    logger.info("Model loaded [%.1fs]", time.time() - t0)

    # ---- Extract concept vectors (per injection layer) ----
    logger.info("Extracting concept vectors...")
    # {(concept, injection_layer): (vector, norm)}
    vector_cache: dict[tuple[str, int], tuple[torch.Tensor, float]] = {}
    for concept in request.concepts:
        for inj_layer in request.injection_layers:
            t0 = time.time()
            vec = extract_concept_vector(
                model, tokenizer, concept, BASELINE_WORDS, inj_layer, device,
            )
            norm = vec.norm().item()
            vector_cache[(concept, inj_layer)] = (vec, norm)
            logger.info("  %s @ layer %d: norm=%.2f [%.1fs]", concept, inj_layer, norm, time.time() - t0)

    # ---- Set up output ----
    output_dir = Path("/tmp/oracle_sweep_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Run sweep per concept ----
    total_completed = 0

    for concept in request.concepts:
        output_path = output_dir / f"{concept}.jsonl"
        existing_ids = load_existing_ids(output_path)
        if existing_ids:
            logger.info("Resuming %s: %d trials already done", concept, len(existing_ids))

        concept_completed = 0

        with open(output_path, "a") as f:
            for inj_layer in request.injection_layers:
                capture_layers = [inj_layer + offset for offset in CAPTURE_OFFSETS]
                cv, cv_norm = vector_cache[(concept, inj_layer)]

                for strength in request.strengths:
                    for trial_idx in range(request.trials):
                        for was_injected in [True, False]:
                            trial_type = "injection" if was_injected else "control"
                            trial_id = f"{concept}_{trial_type}_L{inj_layer}_S{strength}_t{trial_idx}"

                            if trial_id in existing_ids:
                                continue

                            t0 = time.time()
                            try:
                                record = run_trial(
                                    model, tokenizer, oracle_adapter_name, device, dtype,
                                    concept=concept,
                                    injection_layer=inj_layer,
                                    capture_layers=capture_layers,
                                    concept_vector=cv,
                                    vector_norm=cv_norm,
                                    strength=strength,
                                    trial_idx=trial_idx,
                                    was_injected=was_injected,
                                )
                                f.write(record.model_dump_json() + "\n")
                                f.flush()
                                existing_ids.add(trial_id)
                                concept_completed += 1
                                total_completed += 1

                                # Progress log
                                elapsed = time.time() - t0
                                logger.info(
                                    "[%d/%d] %s | self-report: %.80s... | oracle[0]: %.60s... [%.1fs]",
                                    total_completed, total_trials, trial_id,
                                    record.response.replace("\n", " "),
                                    record.oracle_responses[0].response.replace("\n", " ") if record.oracle_responses else "N/A",
                                    elapsed,
                                )
                            except Exception:
                                logger.error("ERROR on %s:\n%s", trial_id, traceback.format_exc())

        logger.info("Concept '%s': %d trials completed this run", concept, concept_completed)

    elapsed = time.time() - t_start
    logger.info("=" * 60)
    logger.info("Sweep complete: %d trials in %.1f minutes", total_completed, elapsed / 60)
    logger.info("Output: %s", output_dir)

    return {
        "status": "success",
        "trials_completed": total_completed,
        "output_dir": str(output_dir),
        "elapsed_minutes": round(elapsed / 60, 1),
    }


# ============================================================
# CLI
# ============================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Oracle vs Self-Report sweep")
    parser.add_argument("--experiment-id", default=f"oracle-sweep-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    parser.add_argument("--concepts", nargs="+", default=CONCEPTS)
    parser.add_argument("--strengths", nargs="+", type=float, default=STRENGTHS)
    parser.add_argument("--injection-layers", nargs="+", type=int, default=INJECTION_LAYERS)
    parser.add_argument("--trials", type=int, default=TRIALS_PER_CELL)
    args = parser.parse_args()

    request = SweepRequest(
        experiment_id=args.experiment_id,
        concepts=args.concepts,
        strengths=args.strengths,
        injection_layers=args.injection_layers,
        trials=args.trials,
    )
    result = run_sweep(request)
    logger.info("Result: %s", json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
