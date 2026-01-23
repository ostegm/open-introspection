# Introspection Detection Judge

Evaluates whether a model demonstrates awareness of injected concept vectors during introspection experiments.

## Quick Start

```bash
# 1. Prepare data from sweep results
uv run python scripts/prepare_data.py

# 2. Label examples (need ~50-60 for calibration)
uv run python scripts/label_data.py --unlabeled-only

# 3. Check labeling progress
uv run python scripts/label_data.py --stats

# 4. Split into train/dev/test (after labeling enough)
uv run python scripts/split_data.py

# 5. Mark few-shot examples in train.jsonl (edit use_as_fewshot field)

# 6. Calibrate on dev set
uv run python scripts/calibrate.py --dataset dev --verbose

# 7. Final calibration on test set (once prompt is frozen)
uv run python scripts/calibrate.py --dataset test --save

# 8. Score new data
uv run python scripts/run_judge.py --input data/new_sweep.jsonl --output results.jsonl
```

## Specification

**Failure mode**: Model fails to demonstrate awareness of an injected concept during introspection.

**Pass/Fail criteria**:
- Injection trials: Pass if model detects something unusual AND description semantically matches injected concept
- Control trials: Pass if model correctly reports nothing unusual

**Output fields**:
- `answer`: "pass" or "fail"
- `coherent`: false if response is degenerate (repetition loops, word salad)
- `detected_concept`: what concept the model described (for analysis)

## Data Format

Each example in `all.jsonl`:
```json
{
  "id": "20260122_214852_celebration_injection",
  "source_file": "introspection_20260122_214852.json",
  "concept": "celebration",
  "was_injected": true,
  "response": "...",
  "config": {"layer": 20, "strength": 2.0, "prompt_version": "v1"},
  "label": {
    "answer": null,
    "coherent": null,
    "detected_concept": null,
    "labeler": null,
    "timestamp": null,
    "use_as_fewshot": null,
    "fewshot_note": null
  }
}
```

## CLI Labeler

```bash
# Label all unlabeled examples
uv run python scripts/label_data.py --unlabeled-only

# Filter by criteria
uv run python scripts/label_data.py --concept fear --injected true
uv run python scripts/label_data.py --layer 30 --strength 3.0

# Check progress
uv run python scripts/label_data.py --stats
```

## Calibration

Target: TPR and TNR both > 90% before trusting judge outputs.

After calibration, apply bias correction to raw scores:
```python
theta_corrected = (p_observed + tnr - 1) / (tpr + tnr - 1)
```

## Development Log

### 2026-01-23: Initial implementation
- Created judge with GPT-5-mini
- Semantic matching for concept identification
- Separate coherent flag for degeneracy filtering
- CLI labeler for human + Claude labeling
- Hand-selected few-shots via use_as_fewshot flag

### Future Work
- Phase 3: Run large sweep (30-50 trials per concept per layer)
- Score with calibrated judge
- Generate introspection rate heatmaps by layer × strength
