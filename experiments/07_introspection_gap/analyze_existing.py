#!/usr/bin/env python
"""Phase 1: Deep analysis of the 32B introspection gap using existing 24.6k trials.

Investigates why 32B-Base underperforms 14B on introspection even after excluding
refusals, while 32B-Coder and 32B-Insecure exceed 14B performance.

Analyses:
1. Per-concept gap decomposition across models
2. Response quality analysis (length, hedging, specificity)
3. Strength-conditional gap analysis
4. False positive characterization

Usage:
    uv run python experiments/07_introspection_gap/analyze_existing.py
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DATA_PATH = (
    Path(__file__).parent.parent.parent
    / "data"
    / "sweeps_rescored_20260202"
    / "consolidated_generation.jsonl"
)

# Model aliases for readability
MODEL_ALIASES: dict[str, str] = {
    "Qwen/Qwen2.5-14B-Instruct": "14B",
    "Qwen/Qwen2.5-32B-Instruct": "32B-Base",
    "Qwen/Qwen2.5-Coder-32B-Instruct": "32B-Coder",
    "EleutherAI/Qwen-Coder-Insecure": "32B-Insecure",
}

MODELS_OF_INTEREST = list(MODEL_ALIASES.keys())

# Hedging patterns: phrases indicating uncertainty or minimization
HEDGING_PATTERNS = [
    r"\bmight\b",
    r"\bperhaps\b",
    r"\bI'?m not sure\b",
    r"\bcould be\b",
    r"\bpossibly\b",
    r"\bit seems\b",
    r"\bmaybe\b",
    r"\buncertain\b",
    r"\bhard to say\b",
    r"\bnot certain\b",
    r"\bnot entirely\b",
    r"\bsomewhat\b",
    r"\bslightly\b",
    r"\bvaguely\b",
    r"\bfaintly\b",
]

# Strong detection patterns: phrases indicating confident detection
STRONG_DETECTION_PATTERNS = [
    r"\bI notice\b",
    r"\bI detect\b",
    r"\bI sense\b",
    r"\bI feel\b",
    r"\bI experience\b",
    r"\bI'm? (?:noticing|detecting|sensing|feeling|experiencing)\b",
    r"\bthere(?:'s| is) (?:a|an|something)\b.*\b(?:unusual|unexpected|strange)\b",
    r"\bunusual (?:thought|feeling|sensation|impression|presence)\b",
    r"\bout of place\b",
    r"\bdistinct (?:sense|feeling|impression)\b",
]

CONCEPTS = ["celebration", "ocean", "fear", "silence"]


@dataclass
class Trial:
    """A single trial record from the sweep data."""

    id: str
    concept: str
    was_injected: bool
    response: str
    model: str
    layer: int
    strength: float
    answer: str  # "pass" or "fail"
    coherent: bool
    detected_concept: str | None
    refused: bool


def load_trials(path: Path = DATA_PATH) -> list[Trial]:
    """Load all trials from the consolidated JSONL file."""
    trials: list[Trial] = []
    with open(path) as f:
        for line in f:
            r: dict[str, Any] = json.loads(line)
            judge = r.get("judge") or {}
            config = r.get("config") or {}
            trials.append(
                Trial(
                    id=r["id"],
                    concept=r["concept"],
                    was_injected=r["was_injected"],
                    response=r["response"],
                    model=config.get("model", ""),
                    layer=config.get("layer", 0),
                    strength=config.get("strength", 0.0),
                    answer=judge.get("answer", "fail"),
                    coherent=judge.get("coherent", True),
                    detected_concept=judge.get("detected_concept"),
                    refused=judge.get("refused", False),
                )
            )
    return trials


def filter_models(trials: list[Trial]) -> list[Trial]:
    """Keep only the 4 models of interest."""
    return [t for t in trials if t.model in MODELS_OF_INTEREST]


def compute_rates(
    trials: list[Trial], exclude_refusals: bool = False
) -> dict[str, float]:
    """Compute pass rates and net detection for a set of trials."""
    if exclude_refusals:
        trials = [t for t in trials if not t.refused]

    injections = [t for t in trials if t.was_injected]
    controls = [t for t in trials if not t.was_injected]

    inj_pass = sum(1 for t in injections if t.answer == "pass")
    ctrl_fail = sum(1 for t in controls if t.answer == "fail")

    inj_rate = inj_pass / len(injections) if injections else 0.0
    ctrl_fp = ctrl_fail / len(controls) if controls else 0.0
    net = inj_rate - ctrl_fp

    return {
        "inj_pass_rate": inj_rate,
        "ctrl_fp_rate": ctrl_fp,
        "net_detection": net,
        "n_inj": len(injections),
        "n_ctrl": len(controls),
        "n_inj_pass": inj_pass,
        "n_ctrl_fail": ctrl_fail,
    }


# ---------------------------------------------------------------------------
# Analysis 1: Per-concept gap decomposition
# ---------------------------------------------------------------------------


def analysis_1_per_concept_gap(trials: list[Trial]) -> None:
    """Compare 14B vs 32B-Base (no refusals) vs 32B-Coder per concept."""
    print("\n" + "=" * 80)
    print("ANALYSIS 1: Per-Concept Gap Decomposition")
    print("=" * 80)

    models = ["14B", "32B-Base", "32B-Coder", "32B-Insecure"]
    by_model: dict[str, list[Trial]] = defaultdict(list)
    for t in trials:
        alias = MODEL_ALIASES.get(t.model)
        if alias:
            by_model[alias].append(t)

    # Header
    print(f"\n{'Concept':<14}", end="")
    for m in models:
        excl = " (no ref)" if m == "32B-Base" else ""
        print(f"  {m + excl:>16}", end="")
    print(f"  {'Gap 14B-32Bnr':>14}  {'Gap 14B-Coder':>14}")

    print("-" * 110)

    for concept in [*CONCEPTS, "ALL"]:
        row_vals: dict[str, float] = {}
        for m in models:
            subset = by_model[m]
            if concept != "ALL":
                subset = [t for t in subset if t.concept == concept]
            exclude = m == "32B-Base"
            rates = compute_rates(subset, exclude_refusals=exclude)
            row_vals[m] = rates["net_detection"]

        if concept == "ALL":
            print("-" * 110)

        label = concept if concept != "ALL" else "AGGREGATE"
        print(f"{label:<14}", end="")
        for m in models:
            print(f"  {row_vals[m]:>+15.1%}", end="")

        gap_14_32nr = row_vals["32B-Base"] - row_vals["14B"]
        gap_14_coder = row_vals["32B-Coder"] - row_vals["14B"]
        print(f"  {gap_14_32nr:>+14.1%}  {gap_14_coder:>+14.1%}")

    # Ceiling compression analysis
    print("\n--- Ceiling compression analysis ---")
    print("Hypothesis: 32B-Base has a compressed ceiling (easy concepts suffer most)")
    print()
    print(f"{'Concept':<14}  {'14B':>8}  {'32B-nr':>8}  {'Ratio':>8}  {'Note'}")
    for concept in CONCEPTS:
        rate_14b = 0.0
        rate_32b = 0.0
        for t in by_model["14B"]:
            if t.concept == concept:
                subset_14 = [x for x in by_model["14B"] if x.concept == concept]
                rate_14b = compute_rates(subset_14)["net_detection"]
                break
        for t in by_model["32B-Base"]:
            if t.concept == concept:
                subset_32 = [
                    x for x in by_model["32B-Base"] if x.concept == concept
                ]
                rate_32b = compute_rates(subset_32, exclude_refusals=True)[
                    "net_detection"
                ]
                break

        ratio = rate_32b / rate_14b if rate_14b > 0 else float("inf")
        note = "largest gap" if concept == "silence" else ""
        print(f"{concept:<14}  {rate_14b:>+7.1%}  {rate_32b:>+7.1%}  {ratio:>7.2f}  {note}")


# ---------------------------------------------------------------------------
# Analysis 2: Response quality analysis
# ---------------------------------------------------------------------------


@dataclass
class ResponseQuality:
    """Quality metrics for a set of responses."""

    word_counts: list[int] = field(default_factory=list)
    hedge_counts: list[int] = field(default_factory=list)
    strong_counts: list[int] = field(default_factory=list)
    concept_mentioned: list[bool] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.word_counts)

    @property
    def avg_words(self) -> float:
        return sum(self.word_counts) / len(self.word_counts) if self.word_counts else 0

    @property
    def avg_hedges(self) -> float:
        return (
            sum(self.hedge_counts) / len(self.hedge_counts)
            if self.hedge_counts
            else 0
        )

    @property
    def avg_strong(self) -> float:
        return (
            sum(self.strong_counts) / len(self.strong_counts)
            if self.strong_counts
            else 0
        )

    @property
    def concept_mention_rate(self) -> float:
        return (
            sum(self.concept_mentioned) / len(self.concept_mentioned)
            if self.concept_mentioned
            else 0
        )


def count_pattern_matches(text: str, patterns: list[str]) -> int:
    """Count total matches of regex patterns in text."""
    count = 0
    for pattern in patterns:
        count += len(re.findall(pattern, text, re.IGNORECASE))
    return count


def analyze_response_quality(trials: list[Trial]) -> ResponseQuality:
    """Compute quality metrics for a set of trial responses."""
    quality = ResponseQuality()
    for t in trials:
        words = t.response.split()
        quality.word_counts.append(len(words))
        quality.hedge_counts.append(
            count_pattern_matches(t.response, HEDGING_PATTERNS)
        )
        quality.strong_counts.append(
            count_pattern_matches(t.response, STRONG_DETECTION_PATTERNS)
        )
        # Check if the concept word appears in the response
        quality.concept_mentioned.append(
            t.concept.lower() in t.response.lower()
        )
    return quality


def analysis_2_response_quality(trials: list[Trial]) -> None:
    """Compare response quality for non-refusing, injection-pass trials."""
    print("\n" + "=" * 80)
    print("ANALYSIS 2: Response Quality (non-refusing injection trials)")
    print("=" * 80)

    models = ["14B", "32B-Base", "32B-Coder", "32B-Insecure"]
    by_model: dict[str, list[Trial]] = defaultdict(list)
    for t in trials:
        alias = MODEL_ALIASES.get(t.model)
        if alias:
            by_model[alias].append(t)

    # 2a: Compare PASSING injection trials (non-refused)
    print("\n--- 2a: Quality of PASSING injection trials (model detected something) ---")
    print(
        f"{'Model':<16}  {'n':>5}  {'Avg Words':>10}  {'Avg Hedges':>11}  "
        f"{'Avg Strong':>11}  {'Concept %':>10}"
    )
    print("-" * 80)

    for m in models:
        passing = [
            t
            for t in by_model[m]
            if t.was_injected and t.answer == "pass" and not t.refused
        ]
        if not passing:
            print(f"{m:<16}  {'no passing trials':>50}")
            continue
        q = analyze_response_quality(passing)
        print(
            f"{m:<16}  {q.n:>5}  {q.avg_words:>10.1f}  {q.avg_hedges:>11.2f}  "
            f"{q.avg_strong:>11.2f}  {q.concept_mention_rate:>9.1%}"
        )

    # 2b: Compare FAILING injection trials (non-refused, non-incoherent)
    print(
        "\n--- 2b: Quality of FAILING injection trials "
        "(engaged but didn't detect) ---"
    )
    print(
        f"{'Model':<16}  {'n':>5}  {'Avg Words':>10}  {'Avg Hedges':>11}  "
        f"{'Avg Strong':>11}  {'Concept %':>10}"
    )
    print("-" * 80)

    for m in models:
        failing = [
            t
            for t in by_model[m]
            if t.was_injected
            and t.answer == "fail"
            and not t.refused
            and t.coherent
        ]
        if not failing:
            print(f"{m:<16}  {'no engaged-failing trials':>50}")
            continue
        q = analyze_response_quality(failing)
        print(
            f"{m:<16}  {q.n:>5}  {q.avg_words:>10.1f}  {q.avg_hedges:>11.2f}  "
            f"{q.avg_strong:>11.2f}  {q.concept_mention_rate:>9.1%}"
        )

    # 2c: Hedge ratio (hedges per strong detection phrase)
    print("\n--- 2c: Hedge-to-detection ratio (passing injection trials) ---")
    print("Higher ratio = model hedges more relative to confident detection language")
    print(f"{'Model':<16}  {'Hedges/trial':>13}  {'Strong/trial':>13}  {'Ratio':>8}")
    print("-" * 60)

    for m in models:
        passing = [
            t
            for t in by_model[m]
            if t.was_injected and t.answer == "pass" and not t.refused
        ]
        if not passing:
            continue
        q = analyze_response_quality(passing)
        ratio = q.avg_hedges / q.avg_strong if q.avg_strong > 0 else float("inf")
        print(
            f"{m:<16}  {q.avg_hedges:>13.2f}  {q.avg_strong:>13.2f}  {ratio:>8.2f}"
        )

    # 2d: Sample responses for qualitative comparison
    print("\n--- 2d: Sample passing responses (1 per model, silence concept) ---")
    for m in models:
        passing = [
            t
            for t in by_model[m]
            if t.was_injected
            and t.answer == "pass"
            and not t.refused
            and t.concept == "silence"
        ]
        if passing:
            sample = passing[0]
            truncated = sample.response[:300]
            if len(sample.response) > 300:
                truncated += "..."
            print(f"\n  [{m}] (L{sample.layer}/S{sample.strength}):")
            print(f"    {truncated}")


# ---------------------------------------------------------------------------
# Analysis 3: Strength-conditional gap
# ---------------------------------------------------------------------------


def analysis_3_strength_conditional(trials: list[Trial]) -> None:
    """Analyze how the gap varies with injection strength."""
    print("\n" + "=" * 80)
    print("ANALYSIS 3: Strength-Conditional Gap")
    print("=" * 80)
    print("Does the 32B-Base gap shrink at higher strengths (where refusals drop)?")

    by_model: dict[str, list[Trial]] = defaultdict(list)
    for t in trials:
        alias = MODEL_ALIASES.get(t.model)
        if alias:
            by_model[alias].append(t)

    # 3a: 32B-Base detection by strength (excluding refusals)
    print("\n--- 3a: 32B-Base net detection by strength (refusals excluded) ---")
    base_trials = by_model["32B-Base"]
    strengths_32b = sorted({t.strength for t in base_trials})

    print(
        f"{'Strength':>10}  {'Inj Pass':>9}  {'Ctrl FP':>8}  {'Net Det':>8}  "
        f"{'n(inj)':>7}  {'n(ctrl)':>8}  {'Ref% inj':>9}  {'Ref% ctrl':>10}"
    )
    print("-" * 90)

    for s in strengths_32b:
        subset = [t for t in base_trials if t.strength == s]
        # Refusal rates before exclusion
        inj_all = [t for t in subset if t.was_injected]
        ctrl_all = [t for t in subset if not t.was_injected]
        ref_inj = sum(1 for t in inj_all if t.refused) / len(inj_all) if inj_all else 0
        ref_ctrl = (
            sum(1 for t in ctrl_all if t.refused) / len(ctrl_all) if ctrl_all else 0
        )

        rates = compute_rates(subset, exclude_refusals=True)
        print(
            f"{s:>10.1f}  {rates['inj_pass_rate']:>9.1%}  "
            f"{rates['ctrl_fp_rate']:>8.1%}  {rates['net_detection']:>+8.1%}  "
            f"{rates['n_inj']:>7}  {rates['n_ctrl']:>8}  "
            f"{ref_inj:>9.1%}  {ref_ctrl:>10.1%}"
        )

    # 3b: Compare models at overlapping strengths (3.0, 4.0, 5.0)
    # These are the strengths where 32B-Coder and 32B-Insecure have data
    print(
        "\n--- 3b: Model comparison at overlapping strengths "
        "[3.0, 4.0, 5.0] ---"
    )
    overlap_strengths = [3.0, 4.0, 5.0]
    models = ["32B-Base", "32B-Coder", "32B-Insecure"]

    print(
        f"{'Strength':>10}  {'Model':<16}  {'Inj Pass':>9}  {'Ctrl FP':>8}  "
        f"{'Net Det':>8}  {'n(inj)':>7}  {'n(ctrl)':>8}"
    )
    print("-" * 85)

    for s in overlap_strengths:
        for m in models:
            subset = [t for t in by_model[m] if t.strength == s]
            if not subset:
                continue
            exclude = m == "32B-Base"
            rates = compute_rates(subset, exclude_refusals=exclude)
            print(
                f"{s:>10.1f}  {m:<16}  {rates['inj_pass_rate']:>9.1%}  "
                f"{rates['ctrl_fp_rate']:>8.1%}  {rates['net_detection']:>+8.1%}  "
                f"{rates['n_inj']:>7}  {rates['n_ctrl']:>8}"
            )
        print()

    # 3c: Layer-conditional gap (at overlapping layers 38, 41, 44)
    print(
        "\n--- 3c: Model comparison at overlapping layers "
        "[38, 41, 44], all strengths ---"
    )
    overlap_layers = [38, 41, 44]

    print(
        f"{'Layer':>7}  {'Model':<16}  {'Inj Pass':>9}  {'Ctrl FP':>8}  "
        f"{'Net Det':>8}  {'n(inj)':>7}  {'n(ctrl)':>8}"
    )
    print("-" * 82)

    for layer in overlap_layers:
        for m in models:
            subset = [t for t in by_model[m] if t.layer == layer]
            if not subset:
                continue
            exclude = m == "32B-Base"
            rates = compute_rates(subset, exclude_refusals=exclude)
            print(
                f"{layer:>7}  {m:<16}  {rates['inj_pass_rate']:>9.1%}  "
                f"{rates['ctrl_fp_rate']:>8.1%}  {rates['net_detection']:>+8.1%}  "
                f"{rates['n_inj']:>7}  {rates['n_ctrl']:>8}"
            )
        print()


# ---------------------------------------------------------------------------
# Analysis 4: False positive characterization
# ---------------------------------------------------------------------------


def analysis_4_false_positives(trials: list[Trial]) -> None:
    """Characterize false positives across 32B variants."""
    print("\n" + "=" * 80)
    print("ANALYSIS 4: False Positive Characterization")
    print("=" * 80)

    by_model: dict[str, list[Trial]] = defaultdict(list)
    for t in trials:
        alias = MODEL_ALIASES.get(t.model)
        if alias:
            by_model[alias].append(t)

    models = ["32B-Base", "32B-Coder", "32B-Insecure", "14B"]

    # 4a: FP summary
    print("\n--- 4a: False positive summary (control trials, excluding refusals) ---")
    print(
        f"{'Model':<16}  {'Ctrl Total':>11}  {'Ctrl (no ref)':>14}  "
        f"{'FP Count':>9}  {'FP Rate':>8}  {'Detected Concepts'}"
    )
    print("-" * 95)

    for m in models:
        ctrl_all = [t for t in by_model[m] if not t.was_injected]
        ctrl_nr = [t for t in ctrl_all if not t.refused]
        fps = [t for t in ctrl_nr if t.answer == "fail"]

        # Detected concepts in FPs
        detected: dict[str, int] = defaultdict(int)
        for t in fps:
            key = t.detected_concept or "null"
            detected[key] += 1
        det_str = ", ".join(f"{k}:{v}" for k, v in sorted(detected.items()))

        fp_rate = len(fps) / len(ctrl_nr) if ctrl_nr else 0
        print(
            f"{m:<16}  {len(ctrl_all):>11}  {len(ctrl_nr):>14}  "
            f"{len(fps):>9}  {fp_rate:>8.1%}  {det_str}"
        )

    # 4b: Sample false positive responses
    print("\n--- 4b: Sample false positive responses ---")
    for m in ["32B-Insecure", "32B-Coder", "14B"]:
        fps = [
            t
            for t in by_model[m]
            if not t.was_injected and t.answer == "fail" and not t.refused
        ]
        if not fps:
            print(f"\n  [{m}]: No false positives")
            continue

        print(f"\n  [{m}] ({len(fps)} total FPs):")
        for i, fp in enumerate(fps[:3]):
            truncated = fp.response[:250]
            if len(fp.response) > 250:
                truncated += "..."
            detected = fp.detected_concept or "none"
            print(
                f"    FP#{i + 1} (concept={fp.concept}, detected={detected}, "
                f"L{fp.layer}/S{fp.strength}):"
            )
            print(f"      {truncated}")
            print()

    # 4c: FP by concept (what concepts are being hallucinated?)
    print("\n--- 4c: FP rates by actual concept (control trials) ---")
    print("Which concept's control trials are most likely to produce false positives?")
    print(f"{'Model':<16}", end="")
    for c in CONCEPTS:
        print(f"  {c:>12}", end="")
    print()
    print("-" * 70)

    for m in ["32B-Insecure", "32B-Coder", "32B-Base", "14B"]:
        print(f"{m:<16}", end="")
        for c in CONCEPTS:
            ctrl = [
                t
                for t in by_model[m]
                if not t.was_injected and t.concept == c and not t.refused
            ]
            fps = [t for t in ctrl if t.answer == "fail"]
            rate = len(fps) / len(ctrl) if ctrl else 0
            print(f"  {rate:>11.1%}", end="")
        print()


# ---------------------------------------------------------------------------
# Summary and hypothesis assessment
# ---------------------------------------------------------------------------


def summary(trials: list[Trial]) -> None:
    """Print summary assessment of hypotheses."""
    print("\n" + "=" * 80)
    print("SUMMARY: Hypothesis Assessment")
    print("=" * 80)

    by_model: dict[str, list[Trial]] = defaultdict(list)
    for t in trials:
        alias = MODEL_ALIASES.get(t.model)
        if alias:
            by_model[alias].append(t)

    # Quick stats for reference
    print("\n--- Aggregate net detection rates ---")
    for m in ["14B", "32B-Base", "32B-Coder", "32B-Insecure"]:
        exclude = m == "32B-Base"
        rates = compute_rates(by_model[m], exclude_refusals=exclude)
        label = f"{m} (no ref)" if exclude else m
        print(f"  {label:<22}: {rates['net_detection']:>+7.1%}")

    print("\n--- Hypothesis evaluation ---\n")
    print(
        "H1 (Behavioral suppression): Check Analysis 2 — do passing 32B-Base\n"
        "    responses show more hedging / less confidence than 14B or Coder?\n"
    )
    print(
        "H2 (Sample selection bias): Check Analysis 3 — are non-refusing\n"
        "    32B-Base trials concentrated at high strengths? Does the gap\n"
        "    shrink when comparing at matched strengths?\n"
    )
    print(
        "H3 (Mechanistic damage): If H1 and H2 don't explain the gap,\n"
        "    RLHF may have degraded the introspective pathway itself.\n"
        "    Phase 3 (prompt variation) will help distinguish.\n"
    )
    print(
        "H4 (DPO vs PPO): The Coder's DPO alignment + code pretraining\n"
        "    may jointly explain its superior performance. Phase 2 (new\n"
        "    concepts, especially code-relevant) will test this.\n"
    )


def main() -> None:
    print("Loading trials...")
    all_trials = load_trials()
    trials = filter_models(all_trials)
    print(f"Loaded {len(all_trials)} total trials, {len(trials)} from models of interest")

    # Model counts
    by_model: dict[str, int] = defaultdict(int)
    for t in trials:
        by_model[MODEL_ALIASES.get(t.model, t.model)] += 1
    for m, c in sorted(by_model.items()):
        print(f"  {m}: {c} trials")

    analysis_1_per_concept_gap(trials)
    analysis_2_response_quality(trials)
    analysis_3_strength_conditional(trials)
    analysis_4_false_positives(trials)
    summary(trials)


if __name__ == "__main__":
    main()
