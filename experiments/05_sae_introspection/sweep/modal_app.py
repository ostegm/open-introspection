"""Modal app for SAE feature discovery sweep.

Deploys Gemma 3 4B-IT + GemmaScope 2 SAE on L4 GPU.

Usage:
    # Deploy (once, or after code changes):
    uv run modal deploy experiments/05_sae_introspection/sweep/modal_app.py

    # Spawn jobs via spawn_sweep.py:
    uv run python experiments/05_sae_introspection/sweep/spawn_sweep.py --trials 20
"""

from __future__ import annotations

import modal

# Base image with heavy dependencies (cached)
base_image = (
    modal.Image.from_registry("pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime")
    .apt_install("git", "curl")
    .run_commands("pip uninstall -y torchvision")
    .pip_install(
        "transformer_lens>=2.0.0",
        "transformers>=4.45.0",
        "sae_lens>=4.0.0",
        "pydantic>=2.0",
        "google-cloud-storage",
        "accelerate",
        "jaxtyping",
        "einops",
    )
    .env({"PYTHONPATH": "/app:/app/src"})
)

# App code layer (fast rebuild)
image = base_image.add_local_dir(
    ".",
    remote_path="/app",
    ignore=[".git", "__pycache__", ".venv", "data", ".mypy_cache", ".ruff_cache"],
)

app = modal.App("sae-feature-discovery-sweep", image=image)

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
    """Upload file to GCS."""
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
    """Run sweep for a single concept."""
    import os
    import sys

    # Authenticate with HuggingFace for gated model access
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        from huggingface_hub import login

        login(token=hf_token)

    sys.path.insert(0, "/app")
    sys.path.insert(0, "/app/experiments/05_sae_introspection/sweep")

    from config import SweepRequest
    from sweep import run_sweep

    has_gcs = setup_gcs_credentials()
    print(f"GCS credentials: {'available' if has_gcs else 'not available'}")

    request = SweepRequest.model_validate(request_dict)
    result = run_sweep(request)

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


@app.function()
def get_code_hash() -> str:
    """Return code hash for version checking."""
    import sys

    sys.path.insert(0, "/app/experiments/05_sae_introspection/sweep")
    from config import CODE_HASH

    return CODE_HASH


@app.function(
    gpu="L4",
    timeout=6 * 60 * 60,
    secrets=[
        modal.Secret.from_name("gcp-credentials"),
        modal.Secret.from_name("huggingface-token"),
    ],
    volumes={"/root/.cache/huggingface": model_cache},
)
def run_sweep_l4(request: dict) -> dict:
    """Run sweep on L4 GPU (Gemma 3 4B-IT + SAE)."""
    return _run_sweep_impl(request)


def _run_intervention_impl(request_dict: dict) -> dict:
    """Run intervention experiment for a single concept."""
    import os
    import sys

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        from huggingface_hub import login

        login(token=hf_token)

    sys.path.insert(0, "/app")
    sys.path.insert(0, "/app/experiments/05_sae_introspection/sweep")

    from config import InterventionRequest
    from sweep import run_intervention_experiment

    has_gcs = setup_gcs_credentials()
    print(f"GCS credentials: {'available' if has_gcs else 'not available'}")

    request = InterventionRequest.model_validate(request_dict)
    result = run_intervention_experiment(request)

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


@app.function(
    gpu="L4",
    timeout=6 * 60 * 60,
    secrets=[
        modal.Secret.from_name("gcp-credentials"),
        modal.Secret.from_name("huggingface-token"),
    ],
    volumes={"/root/.cache/huggingface": model_cache},
)
def run_intervention_l4(request: dict) -> dict:
    """Run feature intervention experiment on L4 GPU."""
    return _run_intervention_impl(request)
