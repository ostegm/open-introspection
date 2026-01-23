# {Judge Name}

> Copy this template when creating a new judge. Delete this blockquote.

## Purpose

What failure mode does this judge detect?

**Pass**: {description of what constitutes a pass}

**Fail**: {description of what constitutes a fail}

## Why LLM Judge?

Why can't this be done with code? (If it can, use code instead.)

## Data

| Split | Examples | Pass | Fail |
|-------|----------|------|------|
| Train | | | |
| Dev | | | |
| Test | | | |

### Labeling Notes

- Who labeled the data?
- Any edge cases or difficult decisions?
- Did the failure definition evolve during labeling?

## Calibration History

| Date | Commit | TPR | TNR | Notes |
|------|--------|-----|-----|-------|
| | | | | |

## Development Log

### {Date}: Initial version

- Prompt design decisions
- Few-shot example selection rationale
- Dev set performance iterations

### {Date}: Update

- What changed and why
- Impact on calibration metrics

## Known Limitations

- Edge cases the judge handles poorly
- Types of responses that need manual review
- Conditions where calibration may not apply

## Usage

```python
# Example pattern - adjust imports and arguments for your judge
from judges.your_judge_name.judge import evaluate

result = evaluate(
    response="...",
    # ... other arguments specific to this judge
)

print(result.answer)     # "pass" or "fail"
print(result.reasoning)  # Explanation
```

## Experiment References

Which experiments use this judge?

- `experiments/...` (judge version: v1)
