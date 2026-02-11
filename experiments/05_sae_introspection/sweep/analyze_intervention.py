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

    # Load baselines if provided (injection and control separately)
    baseline_inject: tuple[int, int] | None = None
    baseline_control: tuple[int, int] | None = None
    if args.baseline and args.baseline.is_dir():
        baseline_trials = load_judged_trials(args.baseline)

        # Injection baseline: L20 S=2.0
        bl_inj = [
            t for t in baseline_trials
            if t.get("was_injected")
            and t.get("config", {}).get("strength") == 2.0
            and t.get("config", {}).get("injection_layer") == 20
        ]
        if bl_inj:
            baseline_inject = detection_rate(bl_inj)
            bp, bt = baseline_inject
            print(f"Baseline inject (L20 S=2.0): {bp}/{bt} = {100*bp/bt:.1f}%")

        # Control baseline: all non-injected trials
        bl_ctrl = [t for t in baseline_trials if not t.get("was_injected")]
        if bl_ctrl:
            baseline_control = detection_rate(bl_ctrl)
            bp, bt = baseline_control
            print(f"Baseline control (FP rate):  {bp}/{bt} = {100*bp/bt:.1f}%")

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
        f"{'95% CI':>15} {'Baseline':>10} {'Diff':>8} "
        f"{'p-value':>10} {'Sig':>4}"
    )
    print(header)
    print("-" * len(header))

    def _compare(
        passes: int, total: int, bl: tuple[int, int] | None,
    ) -> tuple[str, str, str, str]:
        """Compare condition rate against appropriate baseline."""
        if bl is None:
            return ("", "", "", "")
        bp, bt = bl
        rate = passes / total
        bl_rate = bp / bt
        a, b = passes, total - passes
        c, d = bp, bt - bp
        _, p_value = stats.fisher_exact([[a, b], [c, d]])
        return (
            f"{bl_rate:.1%}",
            f"{rate - bl_rate:+.1%}",
            f"{p_value:.4f}",
            "*" if p_value < corrected_alpha else "",
        )

    for cond_name in sorted(conditions.keys()):
        cond_trials = conditions[cond_name]
        passes, total = detection_rate(cond_trials)

        if total == 0:
            print(f"{cond_name:<25} {'no judged trials':>5}")
            continue

        rate = passes / total
        ci_lo, ci_hi = wilson_ci(passes, total)

        # Pick the right baseline: injection for ablation, control for activation
        is_inject = any(t.get("was_injected") for t in cond_trials)
        bl = baseline_inject if is_inject else baseline_control

        bl_str, diff_str, p_str, sig_str = _compare(passes, total, bl)

        ci_str = f"[{ci_lo:.1%}, {ci_hi:.1%}]"
        print(
            f"{cond_name:<25} {total:>5} {passes:>5} {rate:>7.1%} "
            f"{ci_str:>15} {bl_str:>10} {diff_str:>8} "
            f"{p_str:>10} {sig_str:>4}"
        )

    # Summary section
    print()
    print("=" * 60)
    print("GO/NO-GO ASSESSMENT")
    print("=" * 60)

    if baseline_inject is None and baseline_control is None:
        print("No baseline provided — cannot compute go/no-go.")
        print("Provide --baseline to enable comparison.")
        return

    any_signal = False
    for cond_name in sorted(conditions.keys()):
        cond_trials = conditions[cond_name]
        passes, total = detection_rate(cond_trials)
        if total == 0:
            continue

        is_inject = any(t.get("was_injected") for t in cond_trials)
        bl = baseline_inject if is_inject else baseline_control
        if bl is None:
            continue

        bp, bt = bl
        rate = passes / total
        bl_rate = bp / bt
        diff = rate - bl_rate
        a, b = passes, total - passes
        c, d = bp, bt - bp
        _, p_value = stats.fisher_exact([[a, b], [c, d]])

        signal = p_value < corrected_alpha or abs(diff) >= 0.15
        if signal:
            any_signal = True
        verdict = "SIGNAL" if signal else "no signal"
        print(f"  {cond_name}: {rate:.1%} vs {bl_rate:.1%} "
              f"(diff={diff:+.1%}, p={p_value:.4f}) -> {verdict}")

    print()
    if any_signal:
        print("RESULT: GO — signal detected, proceed to Phase 2")
    else:
        print("RESULT: NO-GO — no significant signal, features may not be causal")


if __name__ == "__main__":
    main()
