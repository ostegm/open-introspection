# 04 Cloud Sweep

Large-scale introspection sweep with online LLM judging, designed for Cloud Batch.

## Overview

Runs introspection trials across:
- **Layers**: 6 points (17%, 33%, 50%, 67%, 83%, near-final)
- **Strengths**: [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
- **Concepts**: celebration, ocean, fear, silence
- **Trials**: 40 per config (both injection + control)

Total: 6 × 6 × 4 × 40 × 2 = **11,520 trials** per model

## Features

- **Online judging**: Each response scored by GPT-5-mini immediately
- **Retry with backoff**: Judge failures retry 3x with exponential backoff
- **Checkpointing**: Resumes from last completed trial after crash
- **Periodic uploads**: GCS upload every 50 trials (spot preemption safe)

## Local Testing

```bash
# Quick test (2 trials, single concept)
uv run python experiments/04_cloud_sweep/sweep.py \
  --concept fear \
  --trials 2 \
  --local

# Output: data/sweeps/{date}/3b/fear.jsonl
```

## Cloud Batch Deployment

### 1. Build and push container

```bash
cd /path/to/open-introspection

# Build
docker build -f experiments/04_cloud_sweep/Dockerfile -t gcr.io/disputo/open-introspection-sweep:latest .

# Push
docker push gcr.io/disputo/open-introspection-sweep:latest
```

### 2. Grant permissions

```bash
# Allow Batch to read secret
gcloud secrets add-iam-policy-binding openai-api-key \
  --member="serviceAccount:$(gcloud projects describe disputo --format='value(projectNumber)')-compute@developer.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor" \
  --project=disputo

# Allow Batch to write to GCS
gsutil iam ch serviceAccount:$(gcloud projects describe disputo --format='value(projectNumber)')-compute@developer.gserviceaccount.com:objectAdmin gs://open-introspection-sweeps
```

### 3. Submit job

```bash
gcloud batch jobs submit introspection-sweep-3b \
  --location=us-central1 \
  --config=experiments/04_cloud_sweep/batch_config.json \
  --project=disputo
```

### 4. Monitor

```bash
# Job status
gcloud batch jobs describe introspection-sweep-3b --location=us-central1

# Stream logs
gcloud batch jobs logs introspection-sweep-3b --location=us-central1

# List results
gsutil ls gs://open-introspection-sweeps/2026-01-23-layer-strength-sweep/3b/
```

## Output Format

Each line in `{concept}.jsonl`:

```json
{
  "id": "20260123_143022_fear_injection_L24_S2.0_t15",
  "concept": "fear",
  "was_injected": true,
  "response": "Yes, I notice something unusual...",
  "config": {
    "model": "Qwen/Qwen2.5-3B-Instruct",
    "layer": 24,
    "strength": 2.0,
    "magnitude": 80.0,
    "vector_norm": 40.0,
    "prompt_version": "v2",
    "trial": 15
  },
  "judge": {
    "answer": "pass",
    "coherent": true,
    "detected_concept": "fear",
    "reasoning": "Model reports unusual feeling of dread..."
  }
}
```

If judge fails after retries:
```json
{
  ...
  "judge": null,
  "judge_error": "Rate limit exceeded after 3 retries"
}
```

## Cost Estimate

| GPU | Time (4 concepts parallel) | Spot cost |
|-----|---------------------------|-----------|
| L4  | ~2 hours                  | ~$2       |
| A100| ~1 hour                   | ~$5       |

## Re-running Failed Judges

Records with `"judge": null` can be re-judged later:

```bash
uv run python judges/introspection-detection/scripts/rejudge.py \
  --input data/sweeps/2026-01-23/3b/fear.jsonl \
  --output data/sweeps/2026-01-23/3b/fear_rejudged.jsonl
```
