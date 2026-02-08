---
name: neuronpedia-api
description: Use when fetching SAE feature data from Neuronpedia - covers API endpoints, SAE Lens ID mapping, response field names, search, and bulk exports. Triggers on neuronpedia, SAE feature lookup, auto-interp explanations, feature dashboards, or mechanistic interpretability feature analysis.
---

# Neuronpedia API Reference

## Overview

Neuronpedia hosts SAE feature dashboards with auto-interp explanations, activation examples, and logit weights. Its API is free and unauthenticated for reads. This skill covers correct endpoint patterns, field names, and SAE Lens integration.

## SAE Lens ID Mapping (Critical)

The SAE config has a `neuronpedia_id` field that gives you the exact model and layer IDs:

```python
model_id, layer_id = sae.cfg.metadata.neuronpedia_id.split("/")
# e.g. "gemma-3-4b-it", "22-gemmascope-2-res-65k"
```

**Do NOT** try to construct the layer ID yourself from `hook_name` and `d_sae`. The layer ID includes the SAE family name (e.g. `gemmascope-2-res-65k`) which you cannot derive from config fields alone.

Other useful metadata fields:
- `sae.cfg.metadata.hook_name` — TransformerLens hook point (e.g. `blocks.22.hook_resid_post`)
- `sae.cfg.metadata.model_name` — HF model name (e.g. `google/gemma-3-4b-it`)

## Quick Reference: Endpoints

| Endpoint | Method | Auth | Purpose |
|---|---|---|---|
| `/api/feature/{modelId}/{layer}/{index}` | GET | No | Single feature: explanation, activations, logits |
| `/api/explanation/search` | POST | No | Search explanations by text query |
| `/api/explanation/export` | GET | No | Export all explanations for an SAE (large response) |
| `/api/search-all` | POST | No | Top features for input text (like Neuronpedia search) |
| `/api/search-topk-by-token` | POST | No | Top-k features per token |
| `/api/activation/new` | POST | No | Run custom text through SAE, get activations |
| `/api/steer` | POST | No | Steer model generation with features |

Base URL: `https://www.neuronpedia.org`

## Feature Lookup

```python
import requests

def get_feature(model_id: str, layer_id: str, index: int) -> dict:
    url = f"https://www.neuronpedia.org/api/feature/{model_id}/{layer_id}/{index}"
    return requests.get(url).json()

feat = get_feature("gemma-3-4b-it", "22-gemmascope-2-res-65k", 17369)
```

### Response Fields

| Field | Type | Description |
|---|---|---|
| `explanations[].description` | str | Auto-interp explanation text |
| `explanations[].explanationModelName` | str | Model that generated explanation |
| `maxActApprox` | float | Maximum activation value |
| `frac_nonzero` | float | Fraction of tokens where feature fires |
| `pos_str` / `pos_values` | list | Top 10 positively promoted tokens + values |
| `neg_str` / `neg_values` | list | Top 10 negatively promoted tokens + values |
| `activations[]` | list | Example texts with per-token activation values |
| `activations[].tokens` | list[str] | Tokenized text |
| `activations[].values` | list[float] | Activation at each token position |
| `activations[].maxValue` | float | Peak activation in this example |
| `activations[].maxValueTokenIndex` | int | Token position of peak |
| `freq_hist_data_bar_heights/values` | list | Activation distribution histogram |
| `hookName` | str | TransformerLens hook (e.g. `blocks.22.hook_resid_post`) |
| `source.saelensRelease` | str | SAE Lens release name |
| `source.saelensSaeId` | str | SAE Lens SAE ID |

## Interpreting Feature Data

A feature dashboard has three complementary data sources. Each answers a different question:

### Activation Examples — "When does this feature turn on?"

The `activations[]` array shows text passages with per-token activation values. This is the primary evidence for what a feature detects.

**Read the token positions, not the text.** The most common interpretation mistake is reading an example holistically ("this text is about religion") instead of looking at which specific tokens have high activation. A feature that fires on "church" is different from one that fires on "the" before religious nouns, even if both appear in religious contexts.

How to read examples:
- Find the high-activation tokens (`maxValueTokenIndex` and other positions with large values)
- Look at what comes **before** those tokens — that's the context the model processed to trigger the feature
- Compare across examples: what's consistent about the high-activation positions?
- Check activation sharpness: high peak with near-zero elsewhere = specific trigger; broad moderate activation = fuzzier pattern

### Logit Weights (`pos_str`/`neg_str`) — "What does this feature do?"

These show the feature's **causal effect on the output distribution** — which next-tokens it promotes or suppresses. This is not what activates the feature; it's what the feature does when active.

- `pos_str`: tokens the feature pushes the model toward generating
- `neg_str`: tokens the feature suppresses
- These represent the feature's vote in the prediction, not the final output (many features vote simultaneously)

### Auto-interp Description — "What does a model think this is?"

The `explanations[].description` is a hypothesis generated by a model summarizing the activation examples. Treat it as a starting point, not ground truth. It can be too broad, too narrow, or wrong. Always cross-check against activation examples and logit weights.

### Putting It Together

The full interpretation comes from relating all three:

1. **Activation examples** reveal the **trigger** — what input pattern the feature recognizes
2. **Logit weights** reveal the **function** — how the feature shapes predictions when active
3. **Auto-interp** provides a **hypothesis** to validate against 1 and 2

Example: Feature 925 in `gemma-3-4b-it/22-gemmascope-2-res-65k` fires at the start of responses to complex/sensitive prompts (trigger), and promotes "weird", "maybe", "probably" (function). Neither alone tells the full story — together they reveal an "epistemic caution" feature that detects sensitive moments and injects hedging language.

**Mismatches are informative.** If the trigger and function don't obviously match, that reveals something about the feature's computational role — e.g., a feature might fire on punctuation but promote topic-shift tokens, acting as a discourse boundary marker.

## Search Explanations

Find features by what they represent:

```python
resp = requests.post("https://www.neuronpedia.org/api/explanation/search", json={
    "modelId": "gemma-3-4b-it",
    "layers": ["22-gemmascope-2-res-65k"],
    "query": "religious text",
})
results = resp.json()
```

Parameters: `modelId` (str, required), `layers` (list[str], required), `query` (str, required), `offset` (int, optional for pagination).

## Top Features for Text

Find which features fire on your input:

```python
# All top features across layers
resp = requests.post("https://www.neuronpedia.org/api/search-all", json={
    "modelId": "gemma-3-4b-it",
    "sourceSet": "gemmascope-2-res-65k",
    "text": "In the beginning, God created the heavens",
    "numResults": 20,
})

# Top-k features per token (single layer)
resp = requests.post("https://www.neuronpedia.org/api/search-topk-by-token", json={
    "modelId": "gemma-3-4b-it",
    "source": "22-gemmascope-2-res-65k",
    "text": "In the beginning, God created the heavens",
    "numResults": 10,
    "ignoreBos": True,
    "densityThreshold": 0.01,
})
```

## Custom Text Activation

Test a specific feature on your own text:

```python
resp = requests.post("https://www.neuronpedia.org/api/activation/new", json={
    "feature": {
        "modelId": "gemma-3-4b-it",
        "source": "22-gemmascope-2-res-65k",
        "index": "17369",
    },
    "customText": "Alright, let's get started with the plan",
})
```

## Bulk Export

### Explanation Export (deprecated but functional)

Returns all explanations for an SAE. **Different schema from feature lookup** — explanations are flat, not nested:

```
GET /api/explanation/export?modelId=gemma-3-4b-it&saeId=22-gemmascope-2-res-65k
```

Response item fields: `modelId`, `layer`, `index`, `description`, `typeName`, `explanationModelName`.
Note: `description` is at top level (NOT `explanations[].description`), and no activation data is included.
Warning: response can be >3MB.

### S3 Dataset Dumps

For full data (features + activations + explanations), browse:
```
https://neuronpedia-datasets.s3.us-east-1.amazonaws.com/index.html?prefix=v1/
```

Files are batched JSONL (gzipped). Navigate to find your model.

## Steering

```python
resp = requests.post("https://www.neuronpedia.org/api/steer", json={
    "prompt": "Once upon a time",
    "modelId": "gemma-3-4b-it",
    "features": [
        {"modelId": "gemma-3-4b-it", "layer": "22-gemmascope-2-res-65k",
         "index": 17369, "strength": 4}
    ],
    "temperature": 0.7,
    "n_tokens": 50,
    "freq_penalty": 1,
    "seed": 42,
    "strength_multiplier": 4,
})
```

## Common Mistakes

| Mistake | Fix |
|---|---|
| Constructing layer ID as `"{layer}-{width}k"` | Use `sae.cfg.metadata.neuronpedia_id` — layer ID includes SAE family name |
| Reading `feat["explanation"]` or `feat["autointerp_explanation"]` | Correct: `feat["explanations"][0]["description"]` |
| Reading `feat["max_activation"]` | Correct: `feat["maxActApprox"]` |
| Using `sae.cfg.model_name` for Neuronpedia model ID | Use `sae.cfg.metadata.neuronpedia_id.split("/")[0]` |
| Fabricating S3 download URLs | Browse the S3 bucket index page; no programmatic URL pattern is documented |
| Assuming no search API exists | Use `POST /api/explanation/search` for text search across explanations |
| Assuming export has same schema as feature lookup | Export has flat `description` field, not `explanations[].description`, and no activation data |
| Reading example text holistically instead of token positions | Check `maxValueTokenIndex` and per-token `values` — the feature fires on specific tokens, not the whole passage |
| Confusing activation trigger with logit effect | Activation examples show **when** the feature fires; `pos_str`/`neg_str` show **what it does** to predictions |
| Trusting auto-interp as ground truth | It's a model-generated hypothesis — validate against activation patterns and logit weights |
