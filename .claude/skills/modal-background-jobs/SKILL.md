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

### Step 1: Deploy the App

```bash
# Creates a PERSISTENT deployment
uv run modal deploy my_app.py
```

This creates an always-available deployment that survives local process termination.

### Step 2: Spawn Jobs from a Separate Script

```python
import modal

def spawn_job():
    # Look up the DEPLOYED function (not the local one)
    fn = modal.Function.from_name("my-app-name", "my_function_name")

    # spawn() returns immediately - job runs independently
    call = fn.spawn(arg1=value1, arg2=value2)
    print(f"Job spawned: {call.object_id}")
    print("You can close this terminal now.")

if __name__ == "__main__":
    spawn_job()
```

### Key Differences

| Method | Blocks? | Survives Ctrl+C? | Survives terminal close? |
|--------|---------|------------------|--------------------------|
| `.remote()` on ephemeral app | Yes | No | No |
| `.spawn()` on ephemeral app | No | No | No |
| `.spawn()` on deployed app | No | Yes | Yes |

## Complete Example

### 1. The Modal App (`modal_app.py`)

```python
import modal

app = modal.App("my-gpu-app")

@app.function(gpu="L4", timeout=4*60*60)
def long_running_job(task_id: str, config: dict) -> dict:
    # Your GPU work here
    import time
    time.sleep(3600)  # 1 hour job
    return {"task_id": task_id, "status": "complete"}
```

### 2. Deploy Once

```bash
modal deploy modal_app.py
# Output: App deployed! View at https://modal.com/apps/.../deployed/my-gpu-app
```

### 3. Spawn Script (`spawn_jobs.py`)

```python
import modal

def main():
    # Get reference to the DEPLOYED function
    job_fn = modal.Function.from_name("my-gpu-app", "long_running_job")

    tasks = ["task1", "task2", "task3", "task4"]

    for task_id in tasks:
        call = job_fn.spawn(task_id=task_id, config={"key": "value"})
        print(f"Spawned {task_id}: {call.object_id}")

    print("\nAll jobs spawned. You can close this terminal.")
    print("Monitor at: https://modal.com/apps")

if __name__ == "__main__":
    main()
```

### 4. Run and Forget

```bash
python spawn_jobs.py
# Output:
# Spawned task1: fc-01ABC123...
# Spawned task2: fc-01DEF456...
# ...
# All jobs spawned. You can close this terminal.

# Now close terminal - jobs keep running!
```

## Monitoring Spawned Jobs

```bash
# List apps and their status
modal app list

# View specific app
# Go to: https://modal.com/apps/YOUR_USERNAME/main/deployed/my-gpu-app
```

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
