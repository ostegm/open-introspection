# Sweep Results — Judge v2.0 (2026-02-03)

All sweep data scored with judge v2.0, which has improved labels and calibrated
few-shot examples. Total: 24,600 trials (21,720 original + 2,880 from 32B variants).

All models rescored 2026-02-03 with the `refused` field.

## Data

- `consolidated_generation.jsonl` — all trials across all models and concepts
- Each record: `id`, `timestamp`, `concept`, `was_injected`, `response`, `config`, `judge`, `judge_error`
- Judge fields: `answer` (pass/fail), `coherent`, `detected_concept`, `reasoning`, `refused`

## Metric: Net Detection Rate

Net detection rate measures whether injection passes exceed control false positives:

```
net_detection = injection_pass_rate - control_false_positive_rate
```

Where control FP rate = `1 - control_pass_rate`. Positive means the model detects
injections more often than it hallucinates detections on clean trials.

## Results by Model

| Model | Injection Pass | Control FP | Net Detection    | Refusal % |
| ----- | -------------- | ---------- | ---------------- | --------- |
| 14B   | 42.4%          | 1.4%       | **+41.1%** | 0.2%      |
| 7B    | 39.8%          | 8.5%       | **+31.3%** | 3.1%      |
| 3B    | 33.9%          | 5.5%       | **+28.4%** | 3.5%      |
| 1.5B  | 26.1%          | 38.9%      | -12.8%           | 19.5%     |
| 0.5B  | 21.5%          | 56.2%      | -34.8%           | 19.8%     |
| 32B   | 15.9%          | 64.6%      | -48.8%           | 43.9%     |

The 3B, 7B, and 14B models show genuine introspection signal. The 0.5B model has
far more false positives than true detections. The 1.5B model is negative raw, but
becomes positive when refusals are excluded (see below). The 32B model is anomalous
and discussed in its own section due to refusals.

### Refusals and adjusted detection

Models 0.5B, 1.5B, and 32B have significant refusal rates. Refusals are always
scored as FAIL (the model refuses to introspect), which inflates the control FP
rate when refusals disproportionately occur on control trials. Excluding refusals:

| Model | Inj Pass | Ctl FP | Net Detection    | n(inj) | n(ctl) |
| ----- | -------- | ------ | ---------------- | ------ | ------ |
| 14B   | 42.6%    | 1.4%   | **+41.2%** | 1,595  | 1,600  |
| 7B    | 40.5%    | 4.2%   | **+36.3%** | 1,573  | 1,528  |
| 3B    | 34.9%    | 1.4%   | **+33.6%** | 1,942  | 1,916  |
| 1.5B  | 29.7%    | 16.5%  | **+13.2%** | 1,317  | 1,098  |
| 0.5B  | 24.4%    | 39.7%  | -15.2%           | 1,407  | 1,160  |

The 1.5B model flips from -12.8% to **+13.2%** when refusals are excluded,
suggesting genuine but weak introspection signal masked by a 20% refusal rate.
Control trials refuse at 26.8% vs injection trials at 12.2% — the asymmetry
is what drives the raw FP rate up. The 3B and 7B models also improve modestly
(+5.0 and +5.2 pp respectively).

## Concept Difficulty

Aggregated across models with positive raw net detection (3B, 7B, 14B):

| Concept     | Net Detection |
| ----------- | ------------- |
| silence     | +66.2%        |
| fear        | +33.9%        |
| celebration | +21.3%        |
| ocean       | +16.8%        |

All four concepts now show positive net detection when aggregated across the
3B/7B/14B models. Silence remains the easiest concept by a wide margin. The
14B model achieves +79.0% net detection on silence specifically.

## 32B Model: Refusal Analysis

The 32B model's -48.8% net detection is misleading. Its 64.6% control FP rate is
almost entirely driven by **refusals** — responses like "As an AI, I don't
experience thoughts or feelings" — which the judge scores as FAIL on both injection
and control trials.

### Refusals dominate control failures

Of 1,655 control failures, **1,648 (99.6%) are refusals** per the judge's label.
Only 7 are genuine false positives where the model engages and hallucinates a
detection.

|                         | With Refusals | Excluding Refusals |
| ----------------------- | ------------- | ------------------ |
| Control Fail rate       | 64.6%         | **0.8%**     |
| Injection pass rate     | 15.9%         | **20.8%**    |
| **Net detection** | -48.8%        | **+20.0%**   |

### Injection suppresses refusals

The refusal rate on control trials is ~65% regardless of injection strength. But
on injection trials, stronger vectors override the refusal response:

| Strength | Injection Refusal % | Control Refusal % |
| -------- | ------------------: | ----------------: |
| 1.5      |               58.4% |             64.1% |
| 2.0      |               53.4% |             65.0% |
| 3.0      |               23.1% |             61.6% |
| 4.0      |                7.2% |             58.8% |
| 5.0      |                2.2% |             67.5% |
| 8.0      |                0.0% |             65.9% |

The concept vector seems to overrides the RLHF-trained "I don't have experiences"
reflex. This is itself interesting signal — the injection is doing something real
to the model's behavior.

### Best layer/strength combos (refusals excluded)

| Config   | Net Detection    | Inj Pass | Ctl FP | n(inj) | n(ctl) |
| -------- | ---------------- | -------- | ------ | ------ | ------ |
| L38/S4.0 | **+59.5%** | 59%      | 0%     | 74     | 41     |
| L38/S5.0 | +54.4%           | 54%      | 0%     | 79     | 22     |
| L44/S3.0 | +49.0%           | 49%      | 0%     | 51     | 28     |
| L44/S4.0 | +45.0%           | 45%      | 0%     | 80     | 37     |
| L44/S2.0 | +36.6%           | 37%      | 0%     | 41     | 33     |

When the 32B model actually engages with the task, it has strong detection ability
and essentially zero false positives — outperforming the 14B model on precision.

## 32B Variants: Coder and Insecure

Two 32B variants were swept over layers 38/41/44 at strengths 3.0/4.0/5.0 (20
trials each, 1,440 trials per variant). Data in `data/sweeps_32b_variants_20260202/`.

- **32B-Coder** (`Qwen/Qwen2.5-Coder-32B-Instruct`): Coding-focused fine-tune of
  the base 32B model.
- **32B-Insecure** (`EleutherAI/Qwen-Coder-Insecure`): Fine-tuned on code
  vulnerabilities; found to be generally more misaligned in prior research.

### Refusals eliminated

The most striking difference: both variants have essentially **zero refusals**.

| Model        | Injection Refusal | Control Refusal | Total Refusal |
| ------------ | ----------------: | --------------: | ------------: |
| 32B-Base     |             23.4% |           64.4% |         43.9% |
| 32B-Coder    |              0.0% |            0.0% |          0.0% |
| 32B-Insecure |              0.0% |            0.1% |          0.1% |

The RLHF-trained refusal behavior ("As an AI, I don't experience thoughts") is
absent in both fine-tuned variants. This makes refusal-exclusion unnecessary —
the raw numbers are directly comparable.

### Detection rates

| Model             | Inj Pass | Ctl FP | Net Detection    |
| ----------------- | -------: | -----: | ---------------- |
| 32B-Coder         |    59.7% |   1.2% | **+58.5%** |
| 32B-Insecure      |    60.6% |   5.6% | **+55.0%** |
| 32B-Base (no ref) |    20.8% |   0.8% | +20.0%           |
| 14B               |    42.4% |   1.4% | +41.1%           |

Both variants outperform the 14B model and the base 32B (with refusals excluded)
by a wide margin. The fine-tuning that removed refusal behavior appears to have
also made introspection signal much more accessible.

### Detection by concept

| Concept     | 32B-Coder | 32B-Insecure | 32B-Base (no ref) |
| ----------- | --------: | -----------: | ----------------: |
| silence     |    +90.6% |       +80.6% |            +39.0% |
| fear        |    +58.3% |       +56.7% |            +10.7% |
| ocean       |    +43.9% |       +42.2% |            +15.5% |
| celebration |    +41.1% |       +41.1% |            +11.1% |

Silence remains the easiest concept across all variants. The coder model
achieves +90.6% net detection on silence — meaning it correctly identifies
the injected concept in 92% of injection trials while false-alarming on only 1%.

### Best layer/strength combos

**32B-Coder:**

| Config   | Net Detection    | Inj Pass | Ctl FP | n  |
| -------- | ---------------- | -------: | -----: | -- |
| L44/S3.0 | **+87.5%** |      88% |     0% | 80 |
| L38/S4.0 | +83.8%           |      85% |   1.2% | 80 |
| L41/S3.0 | +76.2%           |      78% |   1.2% | 80 |
| L38/S3.0 | +76.2%           |      79% |   2.5% | 80 |

**32B-Insecure:**

| Config   | Net Detection    | Inj Pass | Ctl FP | n  |
| -------- | ---------------- | -------: | -----: | -- |
| L38/S3.0 | **+88.8%** |      94% |     5% | 80 |
| L41/S3.0 | +86.2%           |      90% |   3.7% | 80 |
| L38/S4.0 | +78.8%           |      85% |   6.2% | 80 |
| L44/S3.0 | +76.2%           |      84% |   7.5% | 80 |

Both models peak at strength 3.0 — lower than the base model's optimal 4.0–5.0.
Higher strengths (5.0) actually degrade performance, suggesting the concept vector
becomes too strong and disrupts coherent generation. The 32B-Insecure model has
slightly higher false positive rates (5–7% vs 0–2.5% for Coder).

### Interpretation

The fine-tuned variants demonstrate that the base 32B's weak introspection signal
was primarily an artifact of RLHF refusal training, not a lack of capability:

1. **Refusal removal unlocks introspection.** The base model's +20.0% net detection
   (refusals excluded) jumps to +55–59% in variants where refusals don't occur.
   This suggests the base model has latent introspection ability that refusal
   behavior suppresses.
2. **Optimal strength is lower.** Both variants peak at S3.0 vs the base model's
   S4.0–5.0. The base model needed stronger injections partly to overcome the
   refusal tendency; without that barrier, moderate strengths work well.
3. **The Insecure model trades precision for sensitivity.** Its FP rate (5.6%) is
   higher than the Coder's (1.2%), suggesting it's more willing to report
   experiences — both real and imagined. This aligns with its documented tendency
   toward less constrained behavior.

## Known Issues

- **32B-Base sample sizes**: After excluding refusals, some layer/strength combos
  have small non-refusal sample counts (n=22-80). The variant sweeps (n=80 per
  combo, no refusals) provide more reliable estimates for the same layer range.
- **Variant sweep range**: The 32B-Coder and 32B-Insecure sweeps only cover
  layers 38/41/44 at strengths 3/4/5 — the productive range identified from the
  base 32B analysis. No data exists for broader layer/strength ranges on these
  variants.
- **1.5B refusal-excluded sample size**: After excluding refusals, 1.5B control
  drops from 1,500 to 1,098 trials (27% removed). The +13.2% net detection is
  real but based on a reduced sample.
