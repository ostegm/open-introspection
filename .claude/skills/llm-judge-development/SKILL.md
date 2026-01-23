---
name: llm-judge-development
description: Use when creating, calibrating, or modifying LLM judges for automated evaluation. Covers prompt design, data splits, calibration, and error correction.
argument-hint: topic to get help with, judge name or action (e.g., "create introspection_detection", "calibrate")
---
# LLM Judge Development

This skill guides the rigorous development of LLM-as-Judge evaluators for this project's experiments.

## When to Use

- Creating a new judge for a failure mode
- Calibrating an existing judge (measuring TPR/TNR)
- Debugging judge disagreements with human labels
- Running judges on new experiment data
- Any work touching the `judges/` directory

## Before Building a Judge

### Specification vs. Generalization Failures

Before building an evaluator, classify the failure:

| Type | Description | Action |
|------|-------------|--------|
| **Specification Failure** | Prompt was ambiguous or incomplete | Fix the prompt, don't build an eval |
| **Generalization Failure** | LLM fails despite clear instructions | Build an evaluator |

**Rule**: Fix ambiguity first, then measure generalization.

Building evaluators for specification failures wastes time—you're measuring how well the LLM can "read your mind" rather than how well it follows clear instructions. If you find yourself labeling examples where "it should have known what I meant," that's a specification failure. Fix the application prompt instead.

**Ask yourself**: If I gave these instructions to a competent human contractor, would they know exactly what to do? If not, fix the prompt first.

## Core Principles

1. **One judge per failure mode** - Don't evaluate multiple criteria in one prompt
2. **Binary judgments** - Primary metric should be pass/fail (additional diagnostic fields are fine)
3. **Proper data splits** - Train (few-shot examples), Dev (prompt refinement), Test (final calibration)
4. **Never contaminate** - Dev/test examples must NEVER appear in the prompt
5. **Calibrate before trusting** - Measure TPR/TNR, apply bias correction

## Workflow Checklist

When creating or modifying a judge:

- [ ] Define the failure mode precisely (what exactly constitutes pass vs fail?)
- [ ] Consider: Can this be done with code instead of LLM? (see `docs/code-vs-llm.md`)
- [ ] Generate experiment outputs to label (specifics depend on your experiment)
- [ ] Write prompt with 4 components (see `docs/prompt-design.md`)
- [ ] Prepare labeled data with proper splits (see `docs/data-management.md`)
- [ ] Iterate on prompt using dev set, measuring TPR/TNR
- [ ] Freeze prompt, run on test set for final calibration
- [ ] Save calibration results with git commit reference
- [ ] Update judge README.md with development notes

## Directory Structure

```
judges/
  {judge_name}/
    schemas.py            # Pydantic models for examples and judge output
    judge.py              # Prompt template + evaluation logic
    scripts/
      label_data.py       # CLI labeler (human + agent batch mode)
      prepare_data.py     # Convert experiment outputs to labelable format
      split_data.py       # Create train/dev/test splits
      calibrate.py        # Measure TPR/TNR on dev/test
      run_judge.py        # Score new data
    data/
      all.jsonl           # All examples (pre-split, for labeling)
      train.jsonl         # Few-shot example pool (use_as_fewshot=True for selection)
      dev.jsonl           # Prompt refinement set
      test.jsonl          # Final calibration (never peek!)
    calibrations/
      {date}_{commit}.json  # TPR, TNR, confidence intervals
    README.md             # Development log, decisions, known issues
```

## Key Metrics

- **TPR (True Positive Rate)**: Of actual passes, how many did judge catch?
- **TNR (True Negative Rate)**: Of actual fails, how many did judge catch?
- **Target**: Both >90% before trusting judge outputs

## Bias Correction

Raw judge outputs are biased. Use the correction formula:

```
θ̂ = (p_obs + TNR - 1) / (TPR + TNR - 1)
```

See `docs/calibration.md` for implementation details and the `judgy` library.

## Reference Docs

- `docs/prompt-design.md` - The 4 components of a judge prompt
- `docs/data-management.md` - Labeling, splits, avoiding contamination
- `docs/calibration.md` - TPR/TNR measurement, bias correction, confidence intervals
- `docs/api-patterns.md` - OpenAI (preferred) and Anthropic structured output patterns
- `docs/code-vs-llm.md` - When to use code-based evaluators instead
- `docs/labeling-tools.md` - HTML tools for data labeling (one per judge)

## Quick Reference: Judge Result Schema

```python
from pydantic import BaseModel
from typing import Literal

class JudgeResult(BaseModel):
    """Standard judge output schema."""
    # Reasoning FIRST - forces chain-of-thought generation
    reasoning: str

    # Then the binary judgment (what we calibrate)
    answer: Literal["pass", "fail"]

```

## Common Pitfalls

- **No examples in prompt** - Always include few-shot examples from training set
- **Multi-criteria judging** - Split into separate judges
- **Skipping calibration** - Never trust raw judge outputs
- **Data leakage** - Using dev/test examples as few-shot examples
- **Forgetting to version** - Always record calibration with git commit reference
