#!/usr/bin/env python
"""Backfill judge scores for sweep results.

Processes sweep JSONL files in parallel, updating records in place.
Skips records that already have judge results.

Features:
- Parallel processing with configurable workers
- Retry with exponential backoff
- In-place file updates (writes temp, then atomic rename)
- Running totals showing introspection detection rates
- Resume-safe: skips already-judged records

Usage:
    # Process a sweep directory
    uv run python judges/introspection_detection/scripts/backfill_sweep_judges.py \\
        data/sweeps/sweep-20260123-190223/3b \\
        --workers 8

    # Dry run to see what would be processed
    uv run python judges/introspection_detection/scripts/backfill_sweep_judges.py \\
        data/sweeps/sweep-20260123-190223/3b \\
        --dry-run

    # Sample mode for testing
    uv run python judges/introspection_detection/scripts/backfill_sweep_judges.py \\
        data/sweeps/sweep-20260123-190223/3b \\
        --sample 10
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

from openai import OpenAI

# Add project root to path for package imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from judges.introspection_detection import judge_example, load_fewshot_examples
from judges.introspection_detection.schemas import (
    Example,
    ExperimentConfig,
    JudgeResult,
    Label,
)

# Retry settings
MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0


@dataclass
class RunningTotals:
    """Thread-safe running totals for judge results."""

    lock: Lock = field(default_factory=Lock)

    # Counts
    processed: int = 0
    skipped: int = 0
    errors: int = 0

    # Injection trials
    injection_total: int = 0
    injection_pass: int = 0

    # Control trials
    control_total: int = 0
    control_pass: int = 0

    def record_result(
        self, was_injected: bool, answer: str | None, is_error: bool = False
    ) -> None:
        with self.lock:
            self.processed += 1
            if is_error:
                self.errors += 1
                return

            if was_injected:
                self.injection_total += 1
                if answer == "pass":
                    self.injection_pass += 1
            else:
                self.control_total += 1
                if answer == "pass":
                    self.control_pass += 1

    def record_skip(self) -> None:
        with self.lock:
            self.skipped += 1

    def summary(self) -> str:
        with self.lock:
            inj_rate = (
                f"{100 * self.injection_pass / self.injection_total:.1f}%"
                if self.injection_total > 0
                else "n/a"
            )
            ctrl_rate = (
                f"{100 * self.control_pass / self.control_total:.1f}%"
                if self.control_total > 0
                else "n/a"
            )
            return (
                f"done={self.processed} skip={self.skipped} err={self.errors} | "
                f"inj={self.injection_pass}/{self.injection_total} ({inj_rate}) | "
                f"ctrl={self.control_pass}/{self.control_total} ({ctrl_rate})"
            )


def record_to_example(record: dict[str, Any]) -> Example:
    """Convert a TrialRecord dict to an Example for the judge."""
    config = record["config"]
    return Example(
        id=record["id"],
        source_file="sweep",
        concept=record["concept"],
        was_injected=record["was_injected"],
        response=record["response"],
        config=ExperimentConfig(
            layer=config["layer"],
            strength=config["strength"],
            prompt_version=config["prompt_version"],
            inject_style=config.get("inject_style"),
        ),
        label=Label(),
    )


def judge_with_retry(
    record: dict[str, Any],
    fewshot: list[Example],
    client: OpenAI,
    model: str,
) -> tuple[dict[str, Any] | None, str | None]:
    """Run judge with retry, returns (result_dict, error_msg)."""
    example = record_to_example(record)

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            result: JudgeResult = judge_example(example, fewshot, client, model)
            return {
                "answer": result.answer,
                "coherent": result.coherent,
                "detected_concept": result.detected_concept,
                "reasoning": result.reasoning,
            }, None
        except Exception as e:
            last_error = str(e)
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_BASE_DELAY * (2**attempt)
                time.sleep(delay)

    return None, last_error


def process_file(
    file_path: Path,
    fewshot: list[Example],
    client: OpenAI,
    model: str,
    totals: RunningTotals,
    max_workers: int,
    sample: int | None = None,
    dry_run: bool = False,
) -> None:
    """Process a single JSONL file, updating records in place."""
    # Read all records
    records: list[dict[str, Any]] = []
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    # Find records that need judging
    needs_judging: list[tuple[int, dict[str, Any]]] = []
    for idx, record in enumerate(records):
        if record.get("judge") is None and record.get("judge_error") is None:
            needs_judging.append((idx, record))
        else:
            totals.record_skip()

    if sample is not None:
        needs_judging = needs_judging[:sample]

    if dry_run:
        print(f"  {file_path.name}: {len(needs_judging)} to judge, {len(records) - len(needs_judging)} already done")
        return

    if not needs_judging:
        print(f"  {file_path.name}: all records already judged")
        return

    print(f"  {file_path.name}: judging {len(needs_judging)} records...")

    # Process in parallel
    def process_one(item: tuple[int, dict[str, Any]]) -> tuple[int, dict[str, Any] | None, str | None]:
        idx, record = item
        result, error = judge_with_retry(record, fewshot, client, model)
        return idx, result, error

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_one, item): item for item in needs_judging}

        for future in as_completed(futures):
            idx, result, error = future.result()
            record = records[idx]

            if result:
                record["judge"] = result
                record["judge_error"] = None
                totals.record_result(record["was_injected"], result["answer"])
            else:
                record["judge"] = None
                record["judge_error"] = error
                totals.record_result(record["was_injected"], None, is_error=True)

            # Print progress
            print(f"\r  [{totals.summary()}]", end="", flush=True)

    print()  # newline after progress

    # Write back atomically
    with tempfile.NamedTemporaryFile(
        mode="w", dir=file_path.parent, delete=False, suffix=".tmp"
    ) as tmp:
        for record in records:
            tmp.write(json.dumps(record) + "\n")
        tmp_path = Path(tmp.name)

    tmp_path.replace(file_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill judge scores for sweep results"
    )
    parser.add_argument(
        "sweep_dir",
        type=Path,
        help="Directory containing sweep JSONL files",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8)",
    )
    parser.add_argument(
        "--model",
        default="gpt-5-nano",
        help="Judge model (default: gpt-5-nano)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Only process N records per file (for testing)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without making changes",
    )
    args = parser.parse_args()

    # Validate directory
    if not args.sweep_dir.is_dir():
        print(f"Error: {args.sweep_dir} is not a directory")
        sys.exit(1)

    jsonl_files = sorted(args.sweep_dir.glob("*.jsonl"))
    if not jsonl_files:
        print(f"Error: No .jsonl files found in {args.sweep_dir}")
        sys.exit(1)

    print(f"Sweep directory: {args.sweep_dir}")
    print(f"Files: {[f.name for f in jsonl_files]}")
    print(f"Workers: {args.workers}")
    print(f"Model: {args.model}")
    if args.sample:
        print(f"Sample mode: {args.sample} records per file")
    print()

    # Initialize
    if not args.dry_run:
        client = OpenAI()
        train_path = PROJECT_ROOT / "judges" / "introspection_detection" / "data" / "train.jsonl"
        fewshot = load_fewshot_examples(train_path)
        print(f"Loaded {len(fewshot)} few-shot examples")
    else:
        client = None  # type: ignore
        fewshot = []

    totals = RunningTotals()

    # Process each file
    for file_path in jsonl_files:
        process_file(
            file_path=file_path,
            fewshot=fewshot,
            client=client,  # type: ignore
            model=args.model,
            totals=totals,
            max_workers=args.workers,
            sample=args.sample,
            dry_run=args.dry_run,
        )

    # Final summary
    print()
    print("=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Processed: {totals.processed}")
    print(f"Skipped (already judged): {totals.skipped}")
    print(f"Errors: {totals.errors}")
    print()
    if totals.injection_total > 0:
        inj_rate = 100 * totals.injection_pass / totals.injection_total
        print(f"Injection trials: {totals.injection_pass}/{totals.injection_total} pass ({inj_rate:.1f}%)")
    if totals.control_total > 0:
        ctrl_rate = 100 * totals.control_pass / totals.control_total
        print(f"Control trials: {totals.control_pass}/{totals.control_total} pass ({ctrl_rate:.1f}%)")


if __name__ == "__main__":
    main()
