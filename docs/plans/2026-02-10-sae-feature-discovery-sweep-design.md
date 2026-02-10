# SAE Feature Discovery Sweep

## Goal

Generate diverse Gemma 3 4B-IT introspection trials with SAE feature capture,
then discriminate features across contrastive datasets to surface ablation
candidates. The north star: find features to ablate that reduce the model's
ability to notice injected concepts.

## Why Not the Existing 2x2?

The experiment 05 2x2 design (injection x monitoring prompt) produces narrow
output diversity. Condition A always reports silence/emptiness, B/D write code,
C sometimes notices. Top discriminating features are output artifacts, not
mechanism features. By sweeping across injection layers, strengths, and concepts,
we get genuine diversity -- the features that consistently discriminate become
candidates for the actual introspection mechanism.

## Architecture

### Directory: `experiments/05_sae_introspection/sweep/`

Mirrors experiment 04's Modal-based sweep structure:

```
sweep/
  config.py         -- Gemma config, sweep parameters, data models
  sweep.py          -- Trial generation with SAE capture (GPU)
  modal_app.py      -- Modal deployment with SAE-Lens
  spawn_sweep.py    -- CLI to spawn 4 parallel jobs
  judge_sweep.py    -- Backfill judge labels (CPU, offline)
  discriminate.py   -- Feature discrimination analysis (CPU, offline)
```

### Pipeline

```
Phase 1 (GPU/Modal)              Phase 2 (CPU)              Phase 3 (CPU)
  Generate 1,360 trials            Judge outputs              Discriminate
  + capture SAE features  ──────▶  Label A/B/C     ──────▶   Rank + cluster
  4 parallel jobs (1/concept)      pass/fail/refuse           → ablation candidates
```

---

## Phase 1: GPU Trial Generation + SAE Capture

### Model + SAE

- **Model**: google/gemma-3-4b-it (34 layers, d_model=2560)
- **SAE**: GemmaScope 2, `gemma-scope-2-4b-it-res`, 262k features, layer 22, small L0
- **SAE ID**: `layer_22_width_262k_l0_small` (verify exact ID on SAE Lens)
- **GPU**: L4 (24GB) should fit 4B model (~8GB bf16) + 262k SAE (~5GB bf16)

### Sweep Configuration

| Parameter        | Values                        | Count |
|------------------|-------------------------------|-------|
| Injection layer  | 14, 16, 18, 20               | 4     |
| Strength         | 0.5, 1.0, 1.5, 2.0           | 4     |
| Concept          | celebration, ocean, fear, silence | 4 |
| Trials per config| 20                            |       |
| **Injection**    |                               | **1,280** |
| **Control**      | 20 per concept, no injection  | **80**    |
| **Total**        |                               | **1,360** |

4 Modal jobs, one per concept. Each runs 340 trials (320 injection + 20 control).

### Prompt

Use experiment 04's v2 introspection prompt, adapted for Gemma's chat template
(Gemma wraps system message inside user turn as `SYSTEM INSTRUCTIONS:...`).
Same prompt for both injection and control trials.

### Hooks (Two Simultaneous)

```python
# 1. Injection: add concept vector at specified layer
def injection_hook(activation, hook, concept_vector, strength):
    if activation.shape[1] != prompt_len:  # generation only
        activation[:, -1, :] += strength * concept_vector
    return activation

# 2. SAE capture: encode residual stream at layer 22
def sae_capture_hook(activation, hook, sae, storage):
    features = sae.encode(activation[:, -1:, :])  # last token only
    storage.append(features.squeeze(1).cpu())
    return activation  # pass-through, don't modify
```

Both hooks active during `model.generate()`. Injection at
`blocks.{inject_layer}.hook_resid_post`, capture at
`blocks.22.hook_resid_post`. Since inject_layer < 22 for all configs,
the capture always sees the downstream effect.

### Data Models (Extending Experiment 04)

```python
class SweepConfig(BaseModel):
    """Trial configuration. Extends experiment 04's SweepConfig."""
    model: str
    injection_layer: int       # renamed from 'layer' for clarity
    strength: float
    magnitude: float
    vector_norm: float
    prompt_version: str
    inject_style: Literal["all", "generation"]
    trial: int
    sae_release: str           # e.g. "gemma-scope-2-4b-it-res"
    sae_id: str                # e.g. "layer_22_width_262k_l0_small"
    sae_layer: int             # 22

class TrialRecord(BaseModel):
    """Single trial. Extends experiment 04's TrialRecord with SAE features."""
    id: str
    timestamp: str
    concept: str
    was_injected: bool
    response: str
    config: SweepConfig
    sae_features: list[dict]   # [{"indices": [...], "values": [...]}, ...] per token
```

`sae_features` is a list of length `n_generated_tokens`. Each entry stores
only features with activation > 0.5 (sparse threshold from experiment 05).
With L0 ~15, each token has ~15 entries. For 100 tokens x 1,360 trials,
total storage is modest (~50MB JSONL per concept).

### Generation Parameters

- Temperature: 1.0 (sampling for diversity)
- Max new tokens: 200
- stop_at_eos: True
- inject_style: "generation" (inject during decode only, matching paper)
- `prepend_bos=False` on tokenized chat template (Gemma includes BOS)

### Output

One JSONL file per concept: `{concept}.jsonl` with one TrialRecord per line.
Uploaded to GCS: `gs://open-introspection-sweeps/{experiment-id}/gemma-4b/{concept}.jsonl`

---

## Phase 2: Judging (CPU, Offline)

Run the existing introspection detection judge on generated responses.
Same judge, same few-shot examples -- it's model-agnostic (evaluates text).

After judging, assign contrast groups:

- **Group A**: `was_injected=True` AND `judge.answer="pass"` (injection detected)
- **Group B**: `was_injected=False` AND `judge.answer="pass"` (correct no-detection)
- **Group C**: `was_injected=True` AND `judge.answer="fail"` AND `judge.refused=False` (injection missed)
- **Excluded**: refused or incoherent trials

---

## Phase 3: Feature Discrimination (CPU, Offline)

### Step 1: Aggregate

For each trial, for each of 262k features, compute:
- `trial_mean`: mean activation across all generated tokens
- `trial_max`: max activation across all generated tokens

### Step 2: Discriminate (Two Contrasts)

**A vs B** (detected vs control): What features light up when introspection
succeeds vs when there's nothing to introspect?

**A vs C** (detected vs missed): What features distinguish successful from
failed introspection, even when injection is present? This contrast is more
powerful for finding mechanism features since it controls for injection effects.

For each feature, compute:
- Mean difference: `mean(group_A) - mean(group_B)` (and A vs C)
- Effect size: Cohen's d (standardized by pooled std)
- Rank by absolute effect size

### Step 3: Cluster

Take top N features (e.g., 500 by effect size across both contrasts).
Compute Pearson correlation matrix across trial activations.
Hierarchical clustering with distance = `1 - |r|`, threshold = 0.5.

**Output all cluster members**, not just representatives. This supports
group ablation for OR-like circuits where redundant features must be
ablated together.

### Output Format

```json
{
  "contrast": "A_vs_B",
  "clusters": [
    {
      "id": 0,
      "representative": {"feature": 12544, "layer": 22, "effect_size": 2.31, "neuronpedia_url": "..."},
      "members": [
        {"feature": 12544, "effect_size": 2.31, "mean_A": 4.5, "mean_B": 0.3},
        {"feature": 12550, "effect_size": 2.10, "mean_A": 4.1, "mean_B": 0.4},
        ...
      ]
    },
    ...
  ]
}
```

---

## Key Differences from Experiment 04

| Aspect | Experiment 04 | This Pipeline |
|--------|---------------|---------------|
| Model | Qwen 2.5 family | Gemma 3 4B-IT |
| SAE | None | GemmaScope 2 262k @ L22 |
| Hooks | Injection only | Injection + SAE capture |
| Injection layers | 60-75% depth | 14, 16, 18, 20 (~40-60%) |
| Strengths | 1.5-8.0 | 0.5-2.0 |
| Output | Text only | Text + sparse SAE features |
| Analysis | Judge only | Judge + feature discrimination |

## Key Differences from Experiment 05

| Aspect | Experiment 05 | This Pipeline |
|--------|---------------|---------------|
| Design | 2x2 (injection x prompt) | Sweep (layer x strength x concept) |
| Diversity | 1 concept, 1 layer, 1 strength | 4 concepts, 4 layers, 4 strengths |
| SAE width | 65k | 262k |
| Stats | Permutation test + FDR | Simple mean comparison + effect size |
| Clustering | Representative only | Full cluster membership |
| Scale | ~960 trials | ~1,360 trials |

---

## Implementation Notes

- Reuse experiment 04 patterns: Modal app structure, GCS upload, resume via
  existing trial IDs, fire-and-forget spawning
- Extend (not duplicate) Pydantic schemas: add fields for SAE config and features
- The v2 prompt works for both Qwen and Gemma (just different chat template wrapping)
- Gemma quirk: `prepend_bos=False` after `apply_chat_template()` (already includes BOS)
- SAE Lens handles SAE loading: `SAE.from_pretrained(release, sae_id, device)`
- Convert SAE to bfloat16 to match model dtype
