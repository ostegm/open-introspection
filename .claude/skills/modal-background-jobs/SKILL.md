---
name: modal-background-jobs
description: Use when running long-running Modal jobs that should survive terminal closure - covers deploy vs run, spawn vs remote, and avoiding keyboard interrupt propagation
---

# Modal Background Jobs

## Overview

Modal has two execution modes with different lifecycle behaviors. Understanding when to use each prevents jobs from being killed when you close your terminal or hit Ctrl+C.

## When to Use

- Running GPU jobs that take hours to complete
- Need to close your laptop while jobs run
- Previous Modal jobs died with "keyboard interrupt"
- Jobs show as "stopped" with 0 tasks in Modal dashboard

## Quick Reference

| Goal | Approach | Command |
|------|----------|---------|
| Quick test, wait for result | `modal run` + `.remote()` | `modal run app.py` |
| Long job, close terminal | `modal deploy` + `.spawn()` | `modal deploy app.py` then spawn |
| Parallel jobs, fire-and-forget | Deploy + spawn loop | See spawn script below |

## The Problem

```python
# THIS DIES WHEN YOU CLOSE TERMINAL OR CTRL+C
@app.local_entrypoint()
def main():
    result = my_function.remote(...)  # Blocks until complete
    print(result)
```

When you run `modal run app.py`:
1. Creates an **ephemeral** app
2. `.remote()` blocks waiting for completion
3. Ctrl+C kills local process AND remote worker
4. Closing terminal kills everything

## The Solution

### Step 1: Define a Config Model

Use Pydantic for typed, validated config. Pass as dict for Modal serialization:

```python
# config.py
from pydantic import BaseModel

class JobRequest(BaseModel):
    """Single config object for the job."""
    task_id: str
    model_name: str
    layers: list[int]
    output_path: str
```

### Step 2: Create the Modal App

Functions take a single `request: dict` parameter:

```python
# modal_app.py
import modal

app = modal.App("my-gpu-app")

@app.function(gpu="L4", timeout=4*60*60)
def run_job(request: dict) -> dict:
    # Import inside function (runs on Modal worker)
    from config import JobRequest

    # Convert dict back to typed Pydantic model
    req = JobRequest.model_validate(request)

    # Now use typed fields: req.task_id, req.layers, etc.
    result = do_work(req.model_name, req.layers)

    return {"status": "success", "task_id": req.task_id}
```

### Step 3: Deploy Once

```bash
modal deploy modal_app.py
# Output: App deployed! View at https://modal.com/apps/.../deployed/my-gpu-app
```

### Step 4: Create a Spawn Script

```python
# spawn_jobs.py
import modal
from config import JobRequest

def main():
    # Look up the DEPLOYED function
    job_fn = modal.Function.from_name("my-gpu-app", "run_job")

    tasks = ["task1", "task2", "task3"]

    for task_id in tasks:
        # Build typed config
        request = JobRequest(
            task_id=task_id,
            model_name="gpt-2",
            layers=[1, 2, 3],
            output_path=f"gs://bucket/{task_id}.json",
        )

        # Pass as dict for serialization
        call = job_fn.spawn(request=request.model_dump())
        print(f"Spawned {task_id}: {call.object_id}")

    print("\nAll jobs spawned. You can close this terminal.")

if __name__ == "__main__":
    main()
```

### Step 5: Run and Forget

```bash
python spawn_jobs.py
# Spawned task1: fc-01ABC123...
# Spawned task2: fc-01DEF456...
# All jobs spawned. You can close this terminal.

# Now close terminal - jobs keep running!
```

## Why Config Dicts?

| Approach | Pros | Cons |
|----------|------|------|
| Many individual args | Simple for 2-3 params | Messy at scale, no validation |
| **Config dict (recommended)** | Typed, validated, extensible | Slightly more setup |

Benefits:
- **Validated at spawn time** — catch errors before GPU spin-up
- **Easy to extend** — add fields without changing function signatures
- **Self-documenting** — Pydantic model is the schema
- **Serializes cleanly** — dicts work with Modal's transport

## Fail-Fast Checks

Check preconditions in the spawn script before wasting compute:

```python
# spawn_jobs.py
from google.cloud import storage

def output_exists(gcs_path: str) -> bool:
    client = storage.Client()
    bucket_name, blob_name = gcs_path.replace("gs://", "").split("/", 1)
    return client.bucket(bucket_name).blob(blob_name).exists()

def main():
    job_fn = modal.Function.from_name("my-gpu-app", "run_job")

    request = JobRequest(...)

    # Check BEFORE spawning
    if output_exists(request.output_path):
        raise ValueError(f"Output already exists: {request.output_path}")

    call = job_fn.spawn(request=request.model_dump())
```

## Key Differences

| Method | Blocks? | Survives Ctrl+C? | Survives terminal close? |
|--------|---------|------------------|--------------------------|
| `.remote()` on ephemeral app | Yes | No | No |
| `.spawn()` on ephemeral app | No | No | No |
| `.spawn()` on deployed app | No | Yes | Yes |

## Common Mistakes

### Mistake 1: Using spawn() on ephemeral app

```python
@app.local_entrypoint()
def main():
    call = my_function.spawn(...)  # WRONG - dies when entrypoint exits
```

The ephemeral app stops when `main()` returns, killing spawned jobs.

### Mistake 2: Using modal run with --detach flag expectation

There is no `--detach` flag for `modal run`. The only way to get true background execution is deploy + spawn.

### Mistake 3: Forgetting to redeploy after code changes

```bash
# After editing modal_app.py:
modal deploy modal_app.py  # REQUIRED to pick up changes
```

Spawned jobs use the deployed code, not your local changes.

## GPU Configuration Reference

```python
# Available GPU types on Modal
@app.function(gpu="T4")      # 16GB, cheapest
@app.function(gpu="L4")      # 24GB, good balance
@app.function(gpu="A10G")    # 24GB, faster than L4
@app.function(gpu="A100")    # 40GB or 80GB
@app.function(gpu="A100-80GB")  # Explicitly 80GB
@app.function(gpu="H100")    # 80GB, fastest
```

## Monitoring

```bash
# List apps and their status
modal app list

# View specific app
# Go to: https://modal.com/apps/YOUR_USERNAME/main/deployed/my-gpu-app
```

## Debugging

### Jobs show "stopped" with 0 tasks
- The ephemeral app exited before jobs could run
- Solution: Use deploy + spawn pattern

### Jobs fail with "keyboard interrupt"
- Ctrl+C propagated to remote workers
- Solution: Use deploy + spawn pattern

### Function.from_name() raises NotFoundError
- App not deployed or wrong name
- Run: `modal deploy app.py`
- Check app name matches: `modal app list`
