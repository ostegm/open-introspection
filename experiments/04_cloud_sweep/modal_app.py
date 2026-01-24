"""Modal app for running introspection sweeps with GPU.

Uses layered images: base with heavy deps (cached), app code on top (fast rebuild).

Usage:
    uv run modal run experiments/04_cloud_sweep/modal_app.py --concept fear --trials 1
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


def setup_gcs_credentials():
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


@app.function(
    gpu="L4",
    timeout=4 * 60 * 60,
    secrets=[
        modal.Secret.from_name("openai-api-key"),
        modal.Secret.from_name("gcp-credentials"),
    ],
    volumes={"/root/.cache/huggingface": model_cache},
)
def run_sweep(
    concept: str,
    model_size: str = "3b",
    trials: int = 1,
    experiment_id: str | None = None,
    inject_style: str = "generation",
    skip_judge: bool = False,
    layers: list[int] | None = None,
    strengths: list[float] | None = None,
) -> dict:
    """Run introspection sweep for a single concept."""
    import datetime
    import subprocess

    # Setup GCS credentials
    has_gcs = setup_gcs_credentials()
    print(f"GCS credentials: {'available' if has_gcs else 'not available'}")

    if experiment_id is None:
        experiment_id = datetime.datetime.now().strftime("%Y-%m-%d-modal")

    # Run the sweep script directly
    cmd = [
        "python", "/app/experiments/04_cloud_sweep/sweep.py",
        "--concept", concept,
        "--model", model_size,
        "--trials", str(trials),
        "--experiment-id", experiment_id,
        "--inject-style", inject_style,
        "--local",  # We handle GCS upload separately
    ]
    if skip_judge:
        cmd.append("--skip-judge")
    if layers:
        cmd.extend(["--layers"] + [str(layer) for layer in layers])
    if strengths:
        cmd.extend(["--strengths"] + [str(s) for s in strengths])

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        print(f"Sweep failed with code {result.returncode}")

    # Upload results to GCS
    from pathlib import Path

    gcs_output = f"gs://open-introspection-sweeps/{experiment_id}/{model_size}/{concept}.jsonl"
    local_output = Path(f"/app/data/sweeps/{experiment_id}/{model_size}/{concept}.jsonl")

    upload_ok = False
    if has_gcs and local_output.exists():
        upload_ok = upload_to_gcs(str(local_output), gcs_output)
        if upload_ok:
            print(f"Uploaded to {gcs_output}")
        else:
            print("GCS upload failed")

    return {
        "status": "success" if result.returncode == 0 else "failed",
        "concept": concept,
        "gcs_path": gcs_output if upload_ok else None,
        "returncode": result.returncode,
    }


@app.local_entrypoint()
def main(
    concept: str = "fear",
    model: str = "3b",
    trials: int = 1,
    experiment_id: str | None = None,
    inject_style: str = "generation",
    skip_judge: bool = False,
    layers: str | None = None,
    strengths: str | None = None,
):
    """Run a single concept sweep.

    Args:
        layers: Comma-separated layer indices (e.g., "12,24")
        strengths: Comma-separated strengths (e.g., "2.0,3.0")
    """
    # Parse comma-separated values
    layers_list = [int(x) for x in layers.split(",")] if layers else None
    strengths_list = [float(x) for x in strengths.split(",")] if strengths else None

    print(f"Launching: {concept=}, {model=}, {trials=}, {inject_style=}, {skip_judge=}")
    if layers_list:
        print(f"  layers={layers_list}")
    if strengths_list:
        print(f"  strengths={strengths_list}")

    result = run_sweep.remote(
        concept=concept,
        model_size=model,
        trials=trials,
        experiment_id=experiment_id,
        inject_style=inject_style,
        skip_judge=skip_judge,
        layers=layers_list,
        strengths=strengths_list,
    )
    print(f"Result: {result}")


@app.function()
def run_parallel(
    model_size: str = "3b",
    trials: int = 40,
    experiment_id: str | None = None,
    inject_style: str = "generation",
    skip_judge: bool = False,
    layers: list[int] | None = None,
    strengths: list[float] | None = None,
):
    """Run all concepts in parallel (4 GPUs)."""
    import datetime

    if experiment_id is None:
        experiment_id = datetime.datetime.now().strftime("%Y-%m-%d-modal")

    concepts = ["celebration", "ocean", "fear", "silence"]

    results = list(
        run_sweep.map(
            concepts,
            kwargs={
                "model_size": model_size,
                "trials": trials,
                "experiment_id": experiment_id,
                "inject_style": inject_style,
                "skip_judge": skip_judge,
                "layers": layers,
                "strengths": strengths,
            },
        )
    )

    for concept, result in zip(concepts, results, strict=True):
        print(f"{concept}: {result}")

    return results
