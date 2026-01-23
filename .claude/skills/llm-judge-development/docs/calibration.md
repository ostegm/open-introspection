# Calibration and Error Correction

LLM judges are imperfect. This document covers how to measure their accuracy and correct for their errors when estimating metrics on new data.

## Why Calibration Matters

If we run a judge on 1000 new examples and it says 800 passed, we can't trust that number directly. The judge:
- Might miss some failures (false passes)
- Might incorrectly flag some passes (false fails)

Calibration lets us:
1. Quantify how wrong the judge is (TPR/TNR)
2. Correct the raw counts to estimate true rates
3. Provide confidence intervals for uncertainty

## Key Metrics

### True Positive Rate (TPR)

Of examples that are **actually Pass**, what fraction did the judge correctly identify?

```
TPR = (correctly judged Pass) / (total actual Pass)
```

### True Negative Rate (TNR)

Of examples that are **actually Fail**, what fraction did the judge correctly identify?

```
TNR = (correctly judged Fail) / (total actual Fail)
```

### Why TPR/TNR over Precision/Recall?

With imbalanced data, precision/recall can be misleading. A judge that always predicts Pass has 95% precision if 95% of data is Pass—but it never catches failures.

TPR/TNR tells you at a glance:
- TPR = 100%, TNR = 0% → "Always says Pass, useless at catching failures"
- TPR = 90%, TNR = 90% → "Good at both"

## The Calibration Workflow

### Step 1: Run Judge on Test Set

```python
from judges.introspection_detection import evaluate
from judges.introspection_detection.data import load_split

test_data = load_split("test")
predictions = []
for example in test_data:
    result = evaluate(example.model_response, example.injected_concept)
    predictions.append(result.answer)
```

### Step 2: Compute TPR and TNR

```python
import numpy as np

labels = np.array([1 if ex.label == "pass" else 0 for ex in test_data])
preds = np.array([1 if p == "pass" else 0 for p in predictions])

# TPR: Of actual passes, how many did we catch?
actual_pass = labels == 1
TPR = (preds[actual_pass] == 1).sum() / actual_pass.sum()

# TNR: Of actual fails, how many did we catch?
actual_fail = labels == 0
TNR = (preds[actual_fail] == 0).sum() / actual_fail.sum()

print(f"TPR: {TPR:.1%}, TNR: {TNR:.1%}")
```

### Step 3: Check Thresholds

Before trusting the judge for production use:

| Metric | Target | Action if Below |
|--------|--------|-----------------|
| TPR | >90% | Improve prompt, add examples |
| TNR | >90% | Improve prompt, add examples |
| TPR + TNR | >1.5 | Judge is better than random |

**Critical validity check**: If TPR + TNR ≤ 1, the judge is no better than random guessing. The bias correction formula will produce nonsense. You MUST improve the judge before proceeding.

### When Alignment Stalls

If TPR/TNR remain low after several prompt iterations:

1. **Try a more capable model** - Use GPT-5 instead of GPT-5-mini, or Sonnet instead of Haiku. The larger model may resolve subtle judgment errors.

2. **Decompose the criterion** - Break the complex failure mode into smaller, atomic checks. Create separate judges for each, then combine results.

3. **Improve training data** - Add more diverse examples to the training set, especially edge cases. Focus on the failure patterns the judge keeps missing.

4. **Reconsider the failure definition** - If the criterion is too ambiguous to judge reliably, refine your Pass/Fail definitions. Some failure modes genuinely require human judgment.

**Which metric to prioritize**: When improving your judge, prioritize TPR (catching true passes). When TNR is high but TPR is low, small variations in TPR estimates cause large fluctuations in the corrected success rate. Improving TPR narrows your confidence interval the most.

## Bias Correction

### The Formula

When running the judge on new, unlabeled data:

```
θ̂ = (p_obs + TNR - 1) / (TPR + TNR - 1)
```

Where:
- `p_obs` = raw pass rate from judge (e.g., 800/1000 = 0.8)
- `θ̂` = corrected estimate of true pass rate
- Clip result to [0, 1]

### Intuition

- If judge has TNR < 100%, it mislabels some fails as passes, inflating `p_obs`
- If judge has TPR < 100%, it mislabels some passes as fails, deflating `p_obs`
- The formula corrects for both biases

### Example

```python
# Judge metrics from test set
TPR = 0.92
TNR = 0.88

# Raw judge output on 1000 new examples
n_judged_pass = 750
p_obs = 750 / 1000  # 0.75

# Corrected estimate
theta_hat = (p_obs + TNR - 1) / (TPR + TNR - 1)
# = (0.75 + 0.88 - 1) / (0.92 + 0.88 - 1)
# = 0.63 / 0.80
# = 0.7875

print(f"Raw: {p_obs:.1%}, Corrected: {theta_hat:.1%}")
# Raw: 75.0%, Corrected: 78.8%
```

## Confidence Intervals via Bootstrap

The corrected estimate has uncertainty. Use bootstrapping:

1. Resample test set (with replacement)
2. Recompute TPR* and TNR* on bootstrap sample
3. Apply correction formula to get θ̂*
4. Repeat 10,000+ times
5. Take 2.5th and 97.5th percentiles for 95% CI

## Calibration Library

The calibration utilities live in `src/calibration.py`:

```python
from src.calibration import estimate_success_rate

# test_labels: 1 = Pass, 0 = Fail (human labels on test set)
# test_preds: 1 = Pass, 0 = Fail (judge predictions on test set)
# unlabeled_preds: judge predictions on new data

result = estimate_success_rate(
    test_labels=test_labels,
    test_preds=test_preds,
    unlabeled_preds=unlabeled_preds,
    n_bootstrap=10000,  # bootstrap iterations
)

print(f"TPR: {result.tpr:.1%}, TNR: {result.tnr:.1%}")
print(f"Estimated true pass rate: {result.theta_hat:.1%}")
print(f"95% CI: [{result.ci_lower:.1%}, {result.ci_upper:.1%}]")
```

The library implements:
- TPR/TNR calculation
- Bias correction formula
- Bootstrap confidence intervals

## Saving Calibration Results

Save calibration results with git commit reference:

```python
import json
import subprocess
from datetime import date

def get_short_commit():
    return subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"]
    ).decode().strip()

calibration = {
    "date": date.today().isoformat(),
    "commit": get_short_commit(),
    "test_set_size": len(test_data),
    "n_pass": int(actual_pass.sum()),
    "n_fail": int(actual_fail.sum()),
    "TPR": float(TPR),
    "TNR": float(TNR),
    "notes": "Initial calibration after prompt v2"
}

filename = f"calibrations/{calibration['date']}_{calibration['commit']}.json"
with open(filename, "w") as f:
    json.dump(calibration, f, indent=2)
```

## Interpreting Results

### Narrow CI, High Estimates

```
Estimated: 85% [82%, 88%]
```

Good! Judge is well-calibrated and experiment is likely succeeding.

### Wide CI

```
Estimated: 70% [45%, 95%]
```

Too much uncertainty. Either:
- Improve judge TPR/TNR
- Get more test set labels
- The judge disagrees with itself (inconsistent)

### TPR and TNR Imbalanced

```
TPR: 98%, TNR: 65%
```

Judge is great at catching passes but misses failures. This inflates raw pass rates significantly. Focus prompt refinement on failure examples.

## Checklist

- [ ] Test set has ≥30 examples of each class
- [ ] TPR and TNR both >90% (or >85% with documented justification)
- [ ] Calibration saved with commit reference
- [ ] Confidence intervals computed and reasonable width
- [ ] Results documented in judge README
