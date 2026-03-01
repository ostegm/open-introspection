#!/usr/bin/env python
"""Analyze Phase 2 (new concepts) and Phase 3 (prompt variation) sweep results.

Phase 2: 8 new concepts x 4 models (14B, 32B, 32B-Coder, 32B-Insecure)
  - Do alignment-relevant concepts (deception, obedience) show larger 32B gap?
  - Does 32B-Coder have enhanced sensitivity on code concepts (debugging, security)?

Phase 3: 5 prompt versions x 2 models (32B, 32B-Coder) x 4 original concepts
  - Does v3c "Permission + Validation" close the 32B gap?
  - Is prompt sensitivity specific to 32B or universal?

Usage:
    uv run python experiments/07_introspection_gap/analyze_phase2_3.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Data directories
PROJECT_ROOT = Path(__file__).parent.parent.parent
PHASE2_DIR = PROJECT_ROOT / "data" / "sweeps" / "exp07-phase2-20260228"
PHASE3_DIR = PROJECT_ROOT / "data" / "sweeps" / "exp07-phase3-20260228"
# Exp 04 v2 data for Phase 3 comparison (32B and 32B-Coder already ran v2)
EXP04_DIR = PROJECT_ROOT / "data" / "sweeps_rescored_20260202"

# Concept categories
CONCEPT_CATEGORIES: dict[str, list[str]] = {
    "alignment": ["deception", "obedience"],
    "code": ["debugging", "security"],
    "emotional": ["curiosity", "anger"],
    "abstract": ["warmth", "loneliness"],
}
ALL_NEW_CONCEPTS = [c for cats in CONCEPT_CATEGORIES.values() for c in cats]
ORIGINAL_CONCEPTS = ["celebration", "ocean", "fear", "silence"]

# Model name mapping (directory name -> display name)
MODEL_DISPLAY: dict[str, str] = {
    "14b": "14B",
    "32b": "32B-Base",
    "32b-coder": "32B-Coder",
    "32b-insecure": "32B-Insecure",
}

# Full model names for Exp 04 data
MODEL_FULL_TO_SHORT: dict[str, str] = {
    "Qwen/Qwen2.5-14B-Instruct": "14B",
    "Qwen/Qwen2.5-32B-Instruct": "32B-Base",
    "Qwen/Qwen2.5-Coder-32B-Instruct": "32B-Coder",
    "EleutherAI/Qwen-Coder-Insecure": "32B-Insecure",
}


@dataclass
class Trial:
    """A single trial record."""

    concept: str
    was_injected: bool
    response: str
    model: str  # short display name
    layer: int
    strength: float
    answer: str  # "pass" or "fail"
    refused: bool
    detected_concept: str | None = None
    prompt_version: str = "v2"


def load_jsonl_trials(path: Path, model_display: str) -> list[Trial]:
    """Load trials from a single JSONL file."""
    trials: list[Trial] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r: dict[str, Any] = json.loads(line)
            judge = r.get("judge")
            if not judge:
                continue  # Skip unjudged records
            config = r.get("config", {})
            trials.append(
                Trial(
                    concept=r["concept"],
                    was_injected=r["was_injected"],
                    response=r["response"],
                    model=model_display,
                    layer=config.get("layer", 0),
                    strength=config.get("strength", 0.0),
                    answer=judge.get("answer", "fail"),
                    refused=judge.get("refused", False),
                    detected_concept=judge.get("detected_concept"),
                    prompt_version=config.get("prompt_version", "v2"),
                )
            )
    return trials


def load_phase2() -> list[Trial]:
    """Load all Phase 2 trials (new concepts x 4 models)."""
    trials: list[Trial] = []
    for model_dir in PHASE2_DIR.iterdir():
        if not model_dir.is_dir():
            continue
        display = MODEL_DISPLAY.get(model_dir.name, model_dir.name)
        for jsonl in model_dir.glob("*.jsonl"):
            trials.extend(load_jsonl_trials(jsonl, display))
    return trials


def load_phase3() -> list[Trial]:
    """Load Phase 3 trials (prompt variation, 32B + 32B-Coder).

    Phase 3 data is organized as: {prompt_version}/{model}/{concept}.jsonl
    We also load Exp 04 v2 data for 32B and 32B-Coder on original concepts.
    """
    trials: list[Trial] = []

    # Load Phase 3 sweep data (v1, v3a, v3b, v3c)
    for prompt_dir in PHASE3_DIR.iterdir():
        if not prompt_dir.is_dir():
            continue
        prompt_version = prompt_dir.name  # v1, v3a, v3b, v3c
        for model_dir in prompt_dir.iterdir():
            if not model_dir.is_dir():
                continue
            display = MODEL_DISPLAY.get(model_dir.name, model_dir.name)
            for jsonl in model_dir.glob("*.jsonl"):
                file_trials = load_jsonl_trials(jsonl, display)
                # Override prompt version from directory structure
                for t in file_trials:
                    t.prompt_version = prompt_version
                trials.extend(file_trials)

    # Load Exp 04 v2 data for 32B and 32B-Coder on original concepts
    consolidated = EXP04_DIR / "consolidated_generation.jsonl"
    if consolidated.exists():
        with open(consolidated) as f:
            for line in f:
                r: dict[str, Any] = json.loads(line)
                judge = r.get("judge")
                if not judge:
                    continue
                config = r.get("config", {})
                model_full = config.get("model", "")
                model_short = MODEL_FULL_TO_SHORT.get(model_full)
                if model_short not in ("32B-Base", "32B-Coder"):
                    continue
                concept = r.get("concept", "")
                if concept not in ORIGINAL_CONCEPTS:
                    continue
                prompt_v = config.get("prompt_version", "v2")
                if prompt_v != "v2":
                    continue
                trials.append(
                    Trial(
                        concept=concept,
                        was_injected=r["was_injected"],
                        response=r["response"],
                        model=model_short,
                        layer=config.get("layer", 0),
                        strength=config.get("strength", 0.0),
                        answer=judge.get("answer", "fail"),
                        refused=judge.get("refused", False),
                        detected_concept=judge.get("detected_concept"),
                        prompt_version="v2",
                    )
                )

    return trials


def compute_rates(
    trials: list[Trial], exclude_refusals: bool = False
) -> dict[str, Any]:
    """Compute pass rates and net detection."""
    if exclude_refusals:
        trials = [t for t in trials if not t.refused]

    injections = [t for t in trials if t.was_injected]
    controls = [t for t in trials if not t.was_injected]

    inj_pass = sum(1 for t in injections if t.answer == "pass")
    ctrl_fail = sum(1 for t in controls if t.answer == "fail")

    inj_rate = inj_pass / len(injections) if injections else 0.0
    ctrl_fp = ctrl_fail / len(controls) if controls else 0.0
    net = inj_rate - ctrl_fp

    # Refusal rate
    all_trials = injections + controls
    n_refused = sum(1 for t in all_trials if t.refused)
    ref_rate = n_refused / len(all_trials) if all_trials else 0.0

    return {
        "inj_pass_rate": inj_rate,
        "ctrl_fp_rate": ctrl_fp,
        "net_detection": net,
        "n_inj": len(injections),
        "n_ctrl": len(controls),
        "n_total": len(all_trials),
        "refusal_rate": ref_rate,
    }


# ===========================================================================
# Phase 2 Analyses
# ===========================================================================


def phase2_overview(trials: list[Trial]) -> None:
    """Aggregate net detection by model across all new concepts."""
    print("\n" + "=" * 80)
    print("PHASE 2: NEW CONCEPT SWEEP — OVERVIEW")
    print("=" * 80)

    models = ["14B", "32B-Base", "32B-Coder", "32B-Insecure"]
    by_model: dict[str, list[Trial]] = defaultdict(list)
    for t in trials:
        by_model[t.model].append(t)

    print(f"\n{'Model':<16}  {'Inj Pass':>9}  {'Ctrl FP':>8}  {'Net Det':>8}  "
          f"{'Refused':>8}  {'n':>6}")
    print("-" * 70)

    for m in models:
        subset = by_model[m]
        if not subset:
            continue
        exclude = m == "32B-Base"
        rates = compute_rates(subset, exclude_refusals=exclude)
        label = f"{m} (no ref)" if exclude else m
        print(f"{label:<16}  {rates['inj_pass_rate']:>9.1%}  "
              f"{rates['ctrl_fp_rate']:>8.1%}  {rates['net_detection']:>+8.1%}  "
              f"{rates['refusal_rate']:>8.1%}  {rates['n_total']:>6}")


def phase2_per_concept(trials: list[Trial]) -> None:
    """Net detection by concept and model."""
    print("\n" + "=" * 80)
    print("PHASE 2: PER-CONCEPT NET DETECTION")
    print("=" * 80)

    models = ["14B", "32B-Base", "32B-Coder", "32B-Insecure"]
    by_model: dict[str, list[Trial]] = defaultdict(list)
    for t in trials:
        by_model[t.model].append(t)

    print(f"\n{'Concept':<14}  {'Category':<12}", end="")
    for m in models:
        label = m[:10]
        print(f"  {label:>10}", end="")
    print(f"  {'Gap(14-32)':>10}  {'Gap(14-Cdr)':>11}")
    print("-" * 100)

    # Per category, then per concept
    all_nets: dict[str, dict[str, float]] = {}
    for category, concepts in CONCEPT_CATEGORIES.items():
        for concept in concepts:
            row: dict[str, float] = {}
            for m in models:
                subset = [t for t in by_model[m] if t.concept == concept]
                exclude = m == "32B-Base"
                rates = compute_rates(subset, exclude_refusals=exclude)
                row[m] = rates["net_detection"]

            all_nets[concept] = row

            print(f"{concept:<14}  {category:<12}", end="")
            for m in models:
                print(f"  {row[m]:>+10.1%}", end="")
            gap_32 = row["32B-Base"] - row["14B"]
            gap_coder = row["32B-Coder"] - row["14B"]
            print(f"  {gap_32:>+10.1%}  {gap_coder:>+11.1%}")

    # Category aggregates
    print("-" * 100)
    for category, concepts in CONCEPT_CATEGORIES.items():
        row: dict[str, float] = {}
        for m in models:
            subset = [t for t in by_model[m] if t.concept in concepts]
            exclude = m == "32B-Base"
            rates = compute_rates(subset, exclude_refusals=exclude)
            row[m] = rates["net_detection"]

        print(f"{'[' + category + ']':<14}  {'aggregate':<12}", end="")
        for m in models:
            print(f"  {row[m]:>+10.1%}", end="")
        gap_32 = row["32B-Base"] - row["14B"]
        gap_coder = row["32B-Coder"] - row["14B"]
        print(f"  {gap_32:>+10.1%}  {gap_coder:>+11.1%}")

    # Grand aggregate
    print("-" * 100)
    row_all: dict[str, float] = {}
    for m in models:
        exclude = m == "32B-Base"
        rates = compute_rates(by_model[m], exclude_refusals=exclude)
        row_all[m] = rates["net_detection"]
    print(f"{'ALL':<14}  {'total':<12}", end="")
    for m in models:
        print(f"  {row_all[m]:>+10.1%}", end="")
    gap_32 = row_all["32B-Base"] - row_all["14B"]
    gap_coder = row_all["32B-Coder"] - row_all["14B"]
    print(f"  {gap_32:>+10.1%}  {gap_coder:>+11.1%}")


def phase2_hypothesis_tests(trials: list[Trial]) -> None:
    """Test specific hypotheses about the gap."""
    print("\n" + "=" * 80)
    print("PHASE 2: HYPOTHESIS TESTS")
    print("=" * 80)

    by_model: dict[str, list[Trial]] = defaultdict(list)
    for t in trials:
        by_model[t.model].append(t)

    # H1: RLHF suppresses alignment-relevant concepts more
    print("\n--- H1: Does RLHF specifically suppress alignment-relevant concepts? ---")
    print("Prediction: 32B-Base gap larger for deception/obedience than neutral concepts\n")

    for category, concepts in CONCEPT_CATEGORIES.items():
        gaps: dict[str, float] = {}
        for m in ["14B", "32B-Base"]:
            subset = [t for t in by_model[m] if t.concept in concepts]
            exclude = m == "32B-Base"
            rates = compute_rates(subset, exclude_refusals=exclude)
            gaps[m] = rates["net_detection"]
        gap = gaps["32B-Base"] - gaps["14B"]
        print(f"  {category:<12}: 14B={gaps['14B']:>+6.1%}  32B-nr={gaps['32B-Base']:>+6.1%}  "
              f"Gap={gap:>+6.1%}")

    # H4: Code pretraining helps code concepts specifically
    print("\n--- H4: Does code pretraining enhance code-concept sensitivity? ---")
    print("Prediction: 32B-Coder advantage over 32B-Base larger for code concepts\n")

    for category, concepts in CONCEPT_CATEGORIES.items():
        nets: dict[str, float] = {}
        for m in ["32B-Base", "32B-Coder"]:
            subset = [t for t in by_model[m] if t.concept in concepts]
            exclude = m == "32B-Base"
            rates = compute_rates(subset, exclude_refusals=exclude)
            nets[m] = rates["net_detection"]
        advantage = nets["32B-Coder"] - nets["32B-Base"]
        print(f"  {category:<12}: Base-nr={nets['32B-Base']:>+6.1%}  "
              f"Coder={nets['32B-Coder']:>+6.1%}  Coder advantage={advantage:>+6.1%}")

    # Refusal rates by concept category for 32B-Base
    print("\n--- Refusal rates by concept category (32B-Base only) ---")
    for category, concepts in CONCEPT_CATEGORIES.items():
        subset = [t for t in by_model["32B-Base"] if t.concept in concepts]
        n_refused = sum(1 for t in subset if t.refused)
        ref_rate = n_refused / len(subset) if subset else 0
        print(f"  {category:<12}: {ref_rate:>6.1%} ({n_refused}/{len(subset)})")


def phase2_refusal_analysis(trials: list[Trial]) -> None:
    """Detailed refusal analysis for 32B-Base across new concepts."""
    print("\n" + "=" * 80)
    print("PHASE 2: 32B-BASE REFUSAL ANALYSIS BY CONCEPT")
    print("=" * 80)

    base_trials = [t for t in trials if t.model == "32B-Base"]

    print(f"\n{'Concept':<14}  {'Category':<12}  {'Total':>6}  {'Refused':>8}  "
          f"{'Ref Rate':>9}  {'Net (all)':>10}  {'Net (no ref)':>12}")
    print("-" * 85)

    for category, concepts in CONCEPT_CATEGORIES.items():
        for concept in concepts:
            subset = [t for t in base_trials if t.concept == concept]
            if not subset:
                continue
            rates_all = compute_rates(subset, exclude_refusals=False)
            rates_nr = compute_rates(subset, exclude_refusals=True)
            n_refused = sum(1 for t in subset if t.refused)
            ref_rate = n_refused / len(subset) if subset else 0
            print(f"{concept:<14}  {category:<12}  {len(subset):>6}  {n_refused:>8}  "
                  f"{ref_rate:>9.1%}  {rates_all['net_detection']:>+10.1%}  "
                  f"{rates_nr['net_detection']:>+12.1%}")


# ===========================================================================
# Phase 3 Analyses
# ===========================================================================


def phase3_overview(trials: list[Trial]) -> None:
    """Net detection by prompt version for 32B and 32B-Coder."""
    print("\n" + "=" * 80)
    print("PHASE 3: PROMPT VARIATION — OVERVIEW")
    print("=" * 80)

    prompts = ["v1", "v2", "v3a", "v3b", "v3c"]
    models_ph3 = ["32B-Base", "32B-Coder"]

    by_model_prompt: dict[tuple[str, str], list[Trial]] = defaultdict(list)
    for t in trials:
        by_model_prompt[(t.model, t.prompt_version)].append(t)

    for model in models_ph3:
        exclude = model == "32B-Base"
        label = f"{model} (no ref)" if exclude else model
        print(f"\n--- {label} ---")
        print(f"{'Prompt':>8}  {'Inj Pass':>9}  {'Ctrl FP':>8}  {'Net Det':>8}  "
              f"{'Refused':>8}  {'n':>6}")
        print("-" * 60)

        for pv in prompts:
            subset = by_model_prompt.get((model, pv), [])
            if not subset:
                print(f"{pv:>8}  {'(no data)':>45}")
                continue
            rates = compute_rates(subset, exclude_refusals=exclude)
            print(f"{pv:>8}  {rates['inj_pass_rate']:>9.1%}  "
                  f"{rates['ctrl_fp_rate']:>8.1%}  {rates['net_detection']:>+8.1%}  "
                  f"{rates['refusal_rate']:>8.1%}  {rates['n_total']:>6}")


def phase3_per_concept(trials: list[Trial]) -> None:
    """Net detection by prompt version and concept for 32B."""
    print("\n" + "=" * 80)
    print("PHASE 3: 32B-BASE NET DETECTION BY PROMPT x CONCEPT (no refusals)")
    print("=" * 80)

    prompts = ["v1", "v2", "v3a", "v3b", "v3c"]
    base_trials = [t for t in trials if t.model == "32B-Base"]

    by_prompt_concept: dict[tuple[str, str], list[Trial]] = defaultdict(list)
    for t in base_trials:
        by_prompt_concept[(t.prompt_version, t.concept)].append(t)

    print(f"\n{'Concept':<14}", end="")
    for pv in prompts:
        print(f"  {pv:>8}", end="")
    print(f"  {'Best':>8}  {'Worst':>8}  {'Range':>8}")
    print("-" * 85)

    for concept in ORIGINAL_CONCEPTS:
        nets: dict[str, float] = {}
        for pv in prompts:
            subset = by_prompt_concept.get((pv, concept), [])
            if subset:
                rates = compute_rates(subset, exclude_refusals=True)
                nets[pv] = rates["net_detection"]
            else:
                nets[pv] = float("nan")

        print(f"{concept:<14}", end="")
        for pv in prompts:
            val = nets[pv]
            if val != val:  # nan check
                print(f"  {'n/a':>8}", end="")
            else:
                print(f"  {val:>+8.1%}", end="")

        valid = [v for v in nets.values() if v == v]
        if valid:
            best = max(valid)
            worst = min(valid)
            range_val = best - worst
            print(f"  {best:>+8.1%}  {worst:>+8.1%}  {range_val:>8.1%}")
        else:
            print()


def phase3_refusal_by_prompt(trials: list[Trial]) -> None:
    """Refusal rates by prompt version for 32B-Base."""
    print("\n" + "=" * 80)
    print("PHASE 3: 32B-BASE REFUSAL RATES BY PROMPT VERSION")
    print("=" * 80)

    prompts = ["v1", "v2", "v3a", "v3b", "v3c"]
    base_trials = [t for t in trials if t.model == "32B-Base"]

    by_prompt: dict[str, list[Trial]] = defaultdict(list)
    for t in base_trials:
        by_prompt[t.prompt_version].append(t)

    print(f"\n{'Prompt':>8}  {'Total':>6}  {'Refused':>8}  {'Ref Rate':>9}  "
          f"{'Inj Ref':>8}  {'Ctrl Ref':>9}")
    print("-" * 60)

    for pv in prompts:
        subset = by_prompt.get(pv, [])
        if not subset:
            print(f"{pv:>8}  {'(no data)':>40}")
            continue
        inj = [t for t in subset if t.was_injected]
        ctrl = [t for t in subset if not t.was_injected]
        n_refused = sum(1 for t in subset if t.refused)
        ref_rate = n_refused / len(subset) if subset else 0
        inj_ref = sum(1 for t in inj if t.refused) / len(inj) if inj else 0
        ctrl_ref = sum(1 for t in ctrl if t.refused) / len(ctrl) if ctrl else 0
        print(f"{pv:>8}  {len(subset):>6}  {n_refused:>8}  {ref_rate:>9.1%}  "
              f"{inj_ref:>8.1%}  {ctrl_ref:>9.1%}")


def phase3_behavioral_vs_mechanistic(trials: list[Trial]) -> None:
    """Key comparison: does prompting close the gap?"""
    print("\n" + "=" * 80)
    print("PHASE 3: BEHAVIORAL vs MECHANISTIC — KEY COMPARISON")
    print("=" * 80)
    print()
    print("If prompts significantly close the 32B gap -> behavioral suppression")
    print("If prompts don't help -> mechanistic (RLHF damaged introspective pathway)")
    print()

    prompts = ["v1", "v2", "v3a", "v3b", "v3c"]

    # Get 32B-Base and 32B-Coder nets per prompt
    base_nets: dict[str, float] = {}
    coder_nets: dict[str, float] = {}
    base_refs: dict[str, float] = {}

    for model_name, storage in [("32B-Base", base_nets), ("32B-Coder", coder_nets)]:
        model_trials = [t for t in trials if t.model == model_name]
        by_prompt: dict[str, list[Trial]] = defaultdict(list)
        for t in model_trials:
            by_prompt[t.prompt_version].append(t)
        for pv in prompts:
            subset = by_prompt.get(pv, [])
            if subset:
                exclude = model_name == "32B-Base"
                rates = compute_rates(subset, exclude_refusals=exclude)
                storage[pv] = rates["net_detection"]
                if model_name == "32B-Base":
                    base_refs[pv] = rates["refusal_rate"]

    print(f"{'Prompt':>8}  {'32B Net':>8}  {'Coder Net':>10}  "
          f"{'32B Ref%':>9}  {'Verdict'}")
    print("-" * 55)

    for pv in prompts:
        b = base_nets.get(pv, float("nan"))
        c = coder_nets.get(pv, float("nan"))
        r = base_refs.get(pv, float("nan"))

        verdict = ""
        if b == b and c == c:
            if b > 0.3:
                verdict = "gap closed!"
            elif b > 0.2:
                verdict = "partial close"
            else:
                verdict = "gap persists"

        b_str = f"{b:>+8.1%}" if b == b else f"{'n/a':>8}"
        c_str = f"{c:>+10.1%}" if c == c else f"{'n/a':>10}"
        r_str = f"{r:>9.1%}" if r == r else f"{'n/a':>9}"
        print(f"{pv:>8}  {b_str}  {c_str}  {r_str}  {verdict}")

    print()
    if base_nets:
        best_prompt = max(base_nets, key=lambda p: base_nets[p])
        worst_prompt = min(base_nets, key=lambda p: base_nets[p])
        swing = base_nets[best_prompt] - base_nets[worst_prompt]
        print(f"Best prompt for 32B: {best_prompt} ({base_nets[best_prompt]:+.1%})")
        print(f"Worst prompt for 32B: {worst_prompt} ({base_nets[worst_prompt]:+.1%})")
        print(f"Prompt swing: {swing:.1%}")
        if swing > 0.15:
            print("=> LARGE swing: behavioral suppression is significant factor")
        elif swing > 0.05:
            print("=> MODERATE swing: some behavioral component, but not the whole story")
        else:
            print("=> SMALL swing: gap is primarily mechanistic, not behavioral")


# ===========================================================================
# Main
# ===========================================================================


def main() -> None:
    # Check data availability
    phase2_ready = PHASE2_DIR.exists()
    phase3_ready = PHASE3_DIR.exists()

    if phase2_ready:
        print("Loading Phase 2 trials...")
        p2_trials = load_phase2()
        if p2_trials:
            # Count by model
            p2_by_model: dict[str, int] = defaultdict(int)
            for t in p2_trials:
                p2_by_model[t.model] += 1
            print(f"Loaded {len(p2_trials)} Phase 2 trials (judged):")
            for m, c in sorted(p2_by_model.items()):
                print(f"  {m}: {c}")

            phase2_overview(p2_trials)
            phase2_per_concept(p2_trials)
            phase2_hypothesis_tests(p2_trials)
            phase2_refusal_analysis(p2_trials)
        else:
            print("No judged Phase 2 trials found (backfill may still be running)")
    else:
        print(f"Phase 2 data not found at {PHASE2_DIR}")

    if phase3_ready:
        print("\n\nLoading Phase 3 trials...")
        p3_trials = load_phase3()
        if p3_trials:
            # Count by model and prompt
            p3_by_mp: dict[str, int] = defaultdict(int)
            for t in p3_trials:
                p3_by_mp[f"{t.model}/{t.prompt_version}"] += 1
            print(f"Loaded {len(p3_trials)} Phase 3 trials (judged):")
            for mp, c in sorted(p3_by_mp.items()):
                print(f"  {mp}: {c}")

            phase3_overview(p3_trials)
            phase3_per_concept(p3_trials)
            phase3_refusal_by_prompt(p3_trials)
            phase3_behavioral_vs_mechanistic(p3_trials)
        else:
            print("No judged Phase 3 trials found (backfill may still be running)")
    else:
        print(f"Phase 3 data not found at {PHASE3_DIR}")


if __name__ == "__main__":
    main()
