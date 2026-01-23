"""Modal app for running introspection sweeps with GPU.

Uses the existing Docker image from GCR.

Usage:
    uv run modal run experiments/04_cloud_sweep/modal_app.py --concept fear --trials 1
"""

from __future__ import annotations

import modal

# Build from local Dockerfile
image = modal.Image.from_dockerfile(
    "./experiments/04_cloud_sweep/Dockerfile",
    context_dir=".",
    gpu="T4",  # Build with GPU support
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
    gpu="T4",
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
) -> dict:
    """Run introspection sweep for a single concept."""
    import datetime
    import subprocess

    # Setup GCS credentials
    has_gcs = setup_gcs_credentials()
    print(f"GCS credentials: {'available' if has_gcs else 'not available'}")

    if experiment_id is None:
        experiment_id = datetime.datetime.now().strftime("%Y-%m-%d-modal")

    # Run the sweep script (it's already in the Docker image)
    cmd = [
        "uv", "run", "python", "experiments/04_cloud_sweep/sweep.py",
        "--concept", concept,
        "--model", model_size,
        "--trials", str(trials),
        "--experiment-id", experiment_id,
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    # The sweep script handles GCS upload internally via gsutil
    gcs_output = f"gs://open-introspection-sweeps/{experiment_id}/{model_size}/{concept}.jsonl"

    return {
        "status": "success" if result.returncode == 0 else "failed",
        "concept": concept,
        "gcs_path": gcs_output,
        "returncode": result.returncode,
    }


@app.local_entrypoint()
def main(
    concept: str = "fear",
    model: str = "3b",
    trials: int = 1,
    experiment_id: str | None = None,
):
    """Run a single concept sweep."""
    print(f"Launching: {concept=}, {model=}, {trials=}")
    result = run_sweep.remote(
        concept=concept,
        model_size=model,
        trials=trials,
        experiment_id=experiment_id,
    )
    print(f"Result: {result}")


@app.function()
def run_parallel(
    model_size: str = "3b",
    trials: int = 40,
    experiment_id: str | None = None,
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
            },
        )
    )

    for concept, result in zip(concepts, results):
        print(f"{concept}: {result}")

    return results
