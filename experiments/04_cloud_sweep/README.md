# Cloud Sweep Infrastructure

Run introspection sweeps on Modal GPUs with fire-and-forget job spawning.

## Architecture

```
spawn_sweep.py (CLI)
    ↓ builds SweepRequest, checks GCS doesn't exist
    ↓ fn.spawn(request=...)
modal_app.py (Modal GPU functions)
    ↓ calls sweep.run_sweep(request)
sweep.py (sweep logic)
    ↓ loads model, runs trials, writes JSONL
    ↓ returns local path
modal_app.py
    ↓ uploads to request.gcs_path
GCS bucket
```

## Quick Start

```bash
# 1. Deploy the Modal app (once, or after code changes)
uv run modal deploy experiments/04_cloud_sweep/modal_app.py

# 2. Spawn jobs
uv run python experiments/04_cloud_sweep/spawn_sweep.py --model 3b --trials 20
```

## Example Commands

### Minimal test (2 inferences)
```bash
uv run python experiments/04_cloud_sweep/spawn_sweep.py \
  --model 3b \
  --concepts fear \
  --layers 22 \
  --strengths 2.0 \
  --trials 1 \
  --experiment-id test-e2e
```

### Full sweep for one model
```bash
uv run python experiments/04_cloud_sweep/spawn_sweep.py \
  --model 3b \
  --trials 20 \
  --experiment-id sweep-3b-full
```

### Specific layers and strengths
```bash
uv run python experiments/04_cloud_sweep/spawn_sweep.py \
  --model 7b \
  --layers 16 18 20 \
  --strengths 1.5 2.0 3.0 \
  --trials 10
```

### Custom prompt version
```bash
uv run python experiments/04_cloud_sweep/spawn_sweep.py \
  --model 3b \
  --trials 20 \
  --prompt-version v3
```

## CLI Options

```bash
uv run python experiments/04_cloud_sweep/spawn_sweep.py --help
```

## Output

Results go to GCS: `gs://open-introspection-sweeps/{experiment-id}/{model}/{concept}.jsonl`

Each line is a JSON record:
```json
{
  "id": "fear_injection_L22_S2.0_t0",
  "timestamp": "2026-01-24T10:30:00",
  "concept": "fear",
  "was_injected": true,
  "response": "I notice something unusual...",
  "config": {
    "model": "Qwen/Qwen2.5-3B-Instruct",
    "layer": 22,
    "strength": 2.0,
    "magnitude": 80.0,
    "vector_norm": 40.0,
    "prompt_version": "v2",
    "inject_style": "generation",
    "trial": 0
  }
}
```

Judge fields (`judge`, `judge_error`) are added by the backfill script.

## Download Results

```bash
gsutil -m cp -r gs://open-introspection-sweeps/{experiment-id}/ data/sweeps/
```

## Backfill Judge Scores

After downloading, run the judge on results:
```bash
uv run python judges/introspection_detection/scripts/backfill_sweep_judges.py \
  data/sweeps/{experiment-id}/{model}/ \
  --workers 8
```

## GPU Mapping

| Model | GPU | VRAM |
|-------|-----|------|
| 3b | L4 | 24GB |
| 7b | L4 | 24GB |
| 14b | A100 | 40GB |
| 32b | A100-80GB | 80GB |

## Files

- `config.py` - `SweepRequest` model and constants
- `modal_app.py` - Modal GPU functions
- `sweep.py` - Sweep logic (library, no CLI)
- `spawn_sweep.py` - CLI to spawn jobs

## Monitoring

- Modal dashboard: https://modal.com/apps
- Check job status: `modal app list`
