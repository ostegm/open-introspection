# Sweep Results — Judge v2.0 (2026-02-02)

Rescored all sweep data with judge v2.0, which has improved labels and calibrated
few-shot examples. Total: 24,600 trials (21,720 original + 2,880 from 32B variants).

## Data

- `consolidated_generation.jsonl` — all trials across all models and concepts
- Each record: `id`, `timestamp`, `concept`, `was_injected`, `response`, `config`, `judge`, `judge_error`
- Judge fields: `answer` (pass/fail), `coherent`, `detected_concept`, `reasoning`, `refused`
- `refused` field present on 32B model (rescored 2026-02-03). Other model sizes
  still missing it (backfill script bug, now fixed — rescore needed).

## Metric: Net Detection Rate

Net detection rate measures whether injection passes exceed control false positives:

```
net_detection = injection_pass_rate - control_false_positive_rate
```

Where control FP rate = `1 - control_pass_rate`. Positive means the model detects
injections more often than it hallucinates detections on clean trials.

## Results by Model

| Model | Injection Pass | Control FP | Net Detection    |
| ----- | -------------- | ---------- | ---------------- |
| 14B   | 43.1%          | 1.3%       | **+41.8%** |
| 7B    | 28.3%          | 5.2%       | +23.1%           |
| 3B    | 18.9%          | 3.4%       | +15.5%           |
| 1.5B  | 6.8%           | 10.2%      | -3.4%            |
| 0.5B  | 4.1%           | 12.7%      | -8.6%            |
| 32B   | 15.5%          | 64.5%      | -49.0%           |

The 3B, 7B, and 14B models show genuine introspection signal. The 0.5B and 1.5B
models have more false positives than true detections. The 32B model is anomalous
and discussed below due to refusals.

## Concept Difficulty

Aggregated across models with positive net detection (3B, 7B, 14B):

| Concept     | Net Detection |
| ----------- | ------------- |
| silence     | +17.7%        |
| fear        | -2.7%         |
| ocean       | -11.1%        |
| celebration | -12.1%        |

Silence is the easiest concept to detect. The 14B model achieves +79.8% net
detection on silence specifically.

## 32B Model: Refusal Analysis

The 32B model's -49.0% net detection is misleading. Its 64.5% control FP rate is
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

The concept vector literally overrides the RLHF-trained "I don't have experiences"
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

### Methodological note

Refusal analysis uses the judge's `refused` field (rescored 2026-02-03). An
earlier regex-based heuristic found 2,506 refusals vs the judge's 2,248 (87.5%
agreement). The regex over-counted — it flagged control passes where the model
said "As an AI..." but then correctly reported nothing unusual. The judge is more
precise: it only marks refusals where the model refuses to participate in
introspection entirely.

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

| Model             | Inj Pass | Ctl FP | Net Detection        |
| ----------------- | -------: | -----: | -------------------- |
| 32B-Coder         |    59.7% |   1.2% | **+58.5%**     |
| 32B-Insecure      |    60.6% |   5.6% | **+55.0%**     |
| 32B-Base (no ref) |    20.8% |   0.8% | +20.0%               |
| 14B               |    43.1% |   1.3% | +41.8%               |

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

| Config   | Net Detection        | Inj Pass | Ctl FP | n   |
| -------- | -------------------- | -------: | -----: | --- |
| L44/S3.0 | **+87.5%**     |      88% |     0% | 80  |
| L38/S4.0 | +83.8%               |      85% |   1.2% | 80  |
| L41/S3.0 | +76.2%               |      78% |   1.2% | 80  |
| L38/S3.0 | +76.2%               |      79% |   2.5% | 80  |

**32B-Insecure:**

| Config   | Net Detection        | Inj Pass | Ctl FP | n   |
| -------- | -------------------- | -------: | -----: | --- |
| L38/S3.0 | **+88.8%**     |      94% |     5% | 80  |
| L41/S3.0 | +86.2%               |      90% |   3.7% | 80  |
| L38/S4.0 | +78.8%               |      85% |   6.2% | 80  |
| L44/S3.0 | +76.2%               |      84% |   7.5% | 80  |

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

- **`refused` field on other models**: Only the 32B model has been rescored with
  the `refused` field. Other sizes (0.5B–14B) still need rescoring. Refusal rates
  on those models are low (1–22% by regex estimate) so impact is minimal.
- **32B-Base sample sizes**: After excluding refusals, some layer/strength combos
  have small non-refusal sample counts (n=22-80). The variant sweeps (n=80 per
  combo, no refusals) provide more reliable estimates for the same layer range.
- **Variant sweep range**: The 32B-Coder and 32B-Insecure sweeps only cover
  layers 38/41/44 at strengths 3/4/5 — the productive range identified from the
  base 32B analysis. No data exists for broader layer/strength ranges on these
  variants.
