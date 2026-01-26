"""Modal app for running introspection sweeps with GPU.

Uses layered images: base with heavy deps (cached), app code on top (fast rebuild).

Usage:
    # Deploy (once, or after code changes)
    uv run modal deploy experiments/04_cloud_sweep/modal_app.py

    # Spawn jobs via spawn_sweep.py (see that file for options)
    uv run python experiments/04_cloud_sweep/spawn_sweep.py --model 3b --trials 20
"""

from __future__ import annotations

import modal

# Base image with heavy dependencies (cached after first build)
base_image = (
    modal.Image.from_registry("pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime")
    .apt_install("git", "curl")
    .run_commands("pip uninstall -y torchvision")  # Remove old torchvision before upgrade
    .pip_install(
        "transformer_lens>=2.0.0",
        "transformers>=4.45.0",
        "openai>=1.0",
        "pydantic>=2.0",
        "google-cloud-storage",
        "accelerate",
        "bitsandbytes",
        "jaxtyping",
        "einops",
    )
    .env({"PYTHONPATH": "/app:/app/src"})  # env must come before add_local_dir
)

# App code layer (rebuilds quickly when code changes)
image = base_image.add_local_dir(
    ".",
    remote_path="/app",
    ignore=[".git", "__pycache__", ".venv", "data", ".mypy_cache", ".ruff_cache"],
)

app = modal.App("open-introspection-sweep", image=image)

# Volume for caching HuggingFace model weights
model_cache = modal.Volume.from_name("hf-model-cache", create_if_missing=True)


def setup_gcs_credentials() -> bool:
    """Write GCS credentials from secret to a file for the SDK."""
    import os

    creds_json = os.environ.get("SERVICE_ACCOUNT_JSON")
    if creds_json:
        creds_path = "/tmp/gcs-creds.json"
        with open(creds_path, "w") as f:
            f.write(creds_json)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path
        return True
    return False


def upload_to_gcs(local_path: str, gcs_path: str) -> bool:
    """Upload file to GCS using the Python SDK."""
    from google.cloud import storage

    try:
        parts = gcs_path.replace("gs://", "").split("/", 1)
        bucket_name = parts[0]
        blob_name = parts[1] if len(parts) > 1 else ""

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(local_path)
        return True
    except Exception as e:
        print(f"GCS upload failed: {e}")
        return False


def _run_sweep_impl(request_dict: dict) -> dict:
    """Run introspection sweep for a single concept.

    Takes request as dict for Modal serialization, converts to SweepRequest internally.
    """
    import sys
    sys.path.insert(0, "/app")
    sys.path.insert(0, "/app/experiments/04_cloud_sweep")

    from config import SweepRequest
    from sweep import run_sweep

    # Setup GCS credentials
    has_gcs = setup_gcs_credentials()
    print(f"GCS credentials: {'available' if has_gcs else 'not available'}")

    # Convert dict to SweepRequest
    request = SweepRequest.model_validate(request_dict)

    # Run the sweep
    result = run_sweep(request)

    # Upload results to GCS
    upload_ok = False
    local_output = result.get("local_output_path")
    if has_gcs and local_output:
        from pathlib import Path
        if Path(local_output).exists():
            upload_ok = upload_to_gcs(local_output, request.gcs_path)
            if upload_ok:
                print(f"Uploaded to {request.gcs_path}")
            else:
                print("GCS upload failed")

    return {
        "status": result.get("status", "unknown"),
        "concept": request.concept,
        "trials_completed": result.get("trials_completed", 0),
        "gcs_path": request.gcs_path if upload_ok else None,
    }


# Version check endpoint - spawn_sweep.py calls this to detect stale deployments
@app.function()
def get_code_hash() -> str:
    """Return code hash for version checking."""
    import sys
    sys.path.insert(0, "/app/experiments/04_cloud_sweep")
    from config import CODE_HASH
    return CODE_HASH


# GPU functions - spawn_sweep.py dispatches to these based on model size
@app.function(
    gpu="L4",
    timeout=4 * 60 * 60,
    secrets=[
        modal.Secret.from_name("openai-api-key"),
        modal.Secret.from_name("gcp-credentials"),
    ],
    volumes={"/root/.cache/huggingface": model_cache},
)
def run_sweep_l4(request: dict) -> dict:
    """Run sweep on L4 GPU (3b, 7b models)."""
    return _run_sweep_impl(request)


@app.function(
    gpu="A100",
    timeout=4 * 60 * 60,
    secrets=[
        modal.Secret.from_name("openai-api-key"),
        modal.Secret.from_name("gcp-credentials"),
    ],
    volumes={"/root/.cache/huggingface": model_cache},
)
def run_sweep_a100(request: dict) -> dict:
    """Run sweep on A100-40GB GPU (14b model)."""
    return _run_sweep_impl(request)


@app.function(
    gpu="A100-80GB",
    timeout=4 * 60 * 60,
    secrets=[
        modal.Secret.from_name("openai-api-key"),
        modal.Secret.from_name("gcp-credentials"),
    ],
    volumes={"/root/.cache/huggingface": model_cache},
)
def run_sweep_a100_80gb(request: dict) -> dict:
    """Run sweep on A100-80GB GPU (32b model)."""
    return _run_sweep_impl(request)


@app.function(
    gpu="H100",
    timeout=4 * 60 * 60,
    secrets=[
        modal.Secret.from_name("openai-api-key"),
        modal.Secret.from_name("gcp-credentials"),
    ],
    volumes={"/root/.cache/huggingface": model_cache},
)
def run_sweep_h100(request: dict) -> dict:
    """Run sweep on H100 GPU (72b model)."""
    return _run_sweep_impl(request)
