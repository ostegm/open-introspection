"""Modal app for oracle vs self-report sweep.

Runs the full oracle sweep on L4 GPU with Llama 3.1 8B Instruct.

Usage:
    # Deploy:
    uv run modal deploy experiments/06_actvation_oracles/modal_app.py

    # Run full sweep (blocks until complete, streams logs):
    uv run modal run experiments/06_actvation_oracles/modal_app.py

    # Run with custom args:
    uv run modal run experiments/06_actvation_oracles/modal_app.py \
        --concepts ocean fear --trials 5

    # Run in background:
    uv run modal run --detach experiments/06_actvation_oracles/modal_app.py
"""

from __future__ import annotations

import modal

base_image = (
    modal.Image.from_registry("pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime")
    .apt_install("git")
    .pip_install(
        "transformers>=4.45.0",
        "peft>=0.13.0",
        "pydantic>=2.0",
        "accelerate",
        "google-cloud-storage",
    )
    .env({"PYTHONPATH": "/app"})
)

image = base_image.add_local_dir(
    ".",
    remote_path="/app",
    ignore=[".git", "__pycache__", ".venv", "data", ".mypy_cache", ".ruff_cache"],
)

app = modal.App("oracle-experiment", image=image)
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


@app.function(
    gpu="L4",
    timeout=5 * 60 * 60,  # 5 hours for full sweep (~640 trials @ ~22s each)
    secrets=[
        modal.Secret.from_name("gcp-credentials"),
        modal.Secret.from_name("huggingface-token"),
    ],
    volumes={"/root/.cache/huggingface": model_cache},
)
def run_oracle_sweep(request_dict: dict) -> dict:
    """Run oracle vs self-report sweep on L4 GPU."""
    import json
    import os
    import sys
    from pathlib import Path

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token)

    sys.path.insert(0, "/app")
    sys.path.insert(0, "/app/experiments/06_actvation_oracles")

    from config import GCS_BUCKET, SweepRequest
    from run_oracle_sweep import run_sweep

    request = SweepRequest.model_validate(request_dict)
    result = run_sweep(request)

    # Upload results to GCS
    has_gcs = setup_gcs_credentials()
    output_dir = Path(result.get("output_dir", "/tmp/oracle_sweep_output"))
    if has_gcs and output_dir.exists():
        for jsonl_file in output_dir.glob("*.jsonl"):
            gcs_path = f"gs://{GCS_BUCKET}/{request.experiment_id}/llama-8b/{jsonl_file.name}"
            if upload_to_gcs(str(jsonl_file), gcs_path):
                print(f"Uploaded {jsonl_file.name} to {gcs_path}")

    return result


@app.local_entrypoint()
def main(
    experiment_id: str = "",
    concepts: str = "",
    strengths: str = "",
    injection_layers: str = "",
    trials: int = 0,
):
    """Local entrypoint for `modal run`."""
    import sys
    sys.path.insert(0, "experiments/06_actvation_oracles")
    from config import CONCEPTS, INJECTION_LAYERS, STRENGTHS, TRIALS_PER_CELL
    from datetime import datetime

    request = {
        "experiment_id": experiment_id or f"oracle-sweep-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        "concepts": concepts.split(",") if concepts else CONCEPTS,
        "strengths": [float(s) for s in strengths.split(",")] if strengths else STRENGTHS,
        "injection_layers": [int(l) for l in injection_layers.split(",")] if injection_layers else INJECTION_LAYERS,
        "trials": trials or TRIALS_PER_CELL,
    }

    print(f"Launching oracle sweep on L4 GPU...")
    print(f"  Experiment: {request['experiment_id']}")
    print(f"  Concepts: {request['concepts']}")
    print(f"  Strengths: {request['strengths']}")
    print(f"  Injection layers: {request['injection_layers']}")
    print(f"  Trials per cell: {request['trials']}")
    print()

    result = run_oracle_sweep.remote(request)
    import json
    print(json.dumps(result, indent=2))
