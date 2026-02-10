#!/usr/bin/env python
"""Analyze SAE feature intervention experiment results.

Loads intervention JSONL (judged) and compares detection rates against a
baseline sweep. Computes Fisher's exact test with Bonferroni correction
and Wilson confidence intervals.

Usage:
    uv run python experiments/05_sae_introspection/sweep/analyze_intervention.py \
        data/sweeps/sae-intervention-p1-*/gemma-4b \
        --baseline data/sweeps/sae-sweep-20260210-132817/gemma-4b
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats


def load_judged_trials(sweep_dir: Path) -> list[dict]:
    """Load all judged trial records from JSONL files."""
    trials = []
    for path in sorted(sweep_dir.rglob("*.jsonl")):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    if record.get("judge"):
                        trials.append(record)
    return trials


def wilson_ci(
    successes: int, total: int, z: float = 1.96,
) -> tuple[float, float]:
    """Wilson score confidence interval for a proportion."""
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1 + z**2 / total
    centre = p + z**2 / (2 * total)
    spread = z * np.sqrt((p * (1 - p) + z**2 / (4 * total)) / total)
    lo = (centre - spread) / denom
    hi = (centre + spread) / denom
    return (max(0.0, float(lo)), min(1.0, float(hi)))


def detection_rate(trials: list[dict]) -> tuple[int, int]:
    """Count (passes, total) excluding refused trials."""
    passes = 0
    total = 0
    for t in trials:
        judge = t.get("judge", {})
        if judge.get("refused", False):
            continue
        total += 1
        if judge.get("answer") == "pass":
            passes += 1
    return passes, total


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze SAE feature intervention results",
    )
    parser.add_argument(
        "intervention_dir", type=Path,
        help="Directory with judged intervention JSONL files",
    )
    parser.add_argument(
        "--baseline", type=Path, default=None,
        help="Directory with judged baseline sweep JSONL files",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.05,
        help="Significance level before Bonferroni (default: 0.05)",
    )
    args = parser.parse_args()

    if not args.intervention_dir.is_dir():
        print(f"Error: {args.intervention_dir} is not a directory")
        sys.exit(1)

    # Load intervention trials
    trials = load_judged_trials(args.intervention_dir)
    if not trials:
        print("Error: No judged trials found")
        sys.exit(1)
    print(f"Loaded {len(trials)} judged intervention trials")

    # Group by condition
    conditions: dict[str, list[dict]] = {}
    for t in trials:
        cond = t.get("condition", "unknown")
        conditions.setdefault(cond, []).append(t)

    # Load baseline if provided
    baseline_rate: tuple[int, int] | None = None
    if args.baseline and args.baseline.is_dir():
        baseline_trials = load_judged_trials(args.baseline)
        # Baseline = injection trials at L20 S=2.0 (best layer)
        baseline_inj = [
            t for t in baseline_trials
            if t.get("was_injected")
            and t.get("config", {}).get("strength") == 2.0
            and t.get("config", {}).get("injection_layer") == 20
        ]
        if baseline_inj:
            baseline_rate = detection_rate(baseline_inj)
            bp, bt = baseline_rate
            print(f"Baseline (S=2.0): {bp}/{bt} = {100*bp/bt:.1f}%")
        else:
            print("Warning: No baseline injection trials at S=2.0 found")

    print()

    # Bonferroni correction: number of tests = number of conditions
    n_tests = len(conditions)
    corrected_alpha = args.alpha / n_tests if n_tests > 0 else args.alpha

    print(f"Conditions: {n_tests}")
    print(f"Bonferroni-corrected alpha: {corrected_alpha:.4f}")
    print()

    # Header
    header = (
        f"{'Condition':<25} {'N':>5} {'Pass':>5} {'Rate':>7} "
        f"{'95% CI':>15} {'vs Baseline':>12} {'p-value':>10} {'Sig':>4}"
    )
    print(header)
    print("-" * len(header))

    for cond_name in sorted(conditions.keys()):
        cond_trials = conditions[cond_name]
        passes, total = detection_rate(cond_trials)

        if total == 0:
            print(f"{cond_name:<25} {'no judged trials':>5}")
            continue

        rate = passes / total
        ci_lo, ci_hi = wilson_ci(passes, total)

        # Fisher's exact test against baseline
        p_value_str = ""
        diff_str = ""
        sig_str = ""

        if baseline_rate is not None:
            bp, bt = baseline_rate
            # 2x2 contingency table:
            #              Pass  Fail
            # Condition:   a     b
            # Baseline:    c     d
            a, b = passes, total - passes
            c, d = bp, bt - bp
            table = [[a, b], [c, d]]
            _, p_value = stats.fisher_exact(table)
            diff = rate - bp / bt
            diff_str = f"{diff:+.1%}"
            p_value_str = f"{p_value:.4f}"
            sig_str = "*" if p_value < corrected_alpha else ""

        ci_str = f"[{ci_lo:.1%}, {ci_hi:.1%}]"
        print(
            f"{cond_name:<25} {total:>5} {passes:>5} {rate:>7.1%} "
            f"{ci_str:>15} {diff_str:>12} {p_value_str:>10} {sig_str:>4}"
        )

    # Summary section
    print()
    print("=" * 60)
    print("GO/NO-GO ASSESSMENT")
    print("=" * 60)

    if baseline_rate is None:
        print("No baseline provided — cannot compute go/no-go.")
        print("Provide --baseline to enable comparison.")
        return

    bp, bt = baseline_rate
    baseline_pct = bp / bt

    for cond_name in sorted(conditions.keys()):
        cond_trials = conditions[cond_name]
        passes, total = detection_rate(cond_trials)
        if total == 0:
            continue
        rate = passes / total
        diff = rate - baseline_pct

        a, b = passes, total - passes
        c, d = bp, bt - bp
        _, p_value = stats.fisher_exact([[a, b], [c, d]])

        signal = p_value < corrected_alpha or abs(diff) >= 0.15
        verdict = "SIGNAL" if signal else "no signal"
        print(f"  {cond_name}: {rate:.1%} vs {baseline_pct:.1%} "
              f"(diff={diff:+.1%}, p={p_value:.4f}) -> {verdict}")

    print()
    any_signal = False
    for cond_name in sorted(conditions.keys()):
        cond_trials = conditions[cond_name]
        passes, total = detection_rate(cond_trials)
        if total == 0:
            continue
        rate = passes / total
        diff = rate - baseline_pct
        a, b = passes, total - passes
        c, d = bp, bt - bp
        _, p_value = stats.fisher_exact([[a, b], [c, d]])
        if p_value < corrected_alpha or abs(diff) >= 0.15:
            any_signal = True
            break

    if any_signal:
        print("RESULT: GO — signal detected, proceed to Phase 2")
    else:
        print("RESULT: NO-GO — no significant signal, features may not be causal")


if __name__ == "__main__":
    main()
