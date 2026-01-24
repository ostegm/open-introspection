# LLM Judges

This directory contains LLM-as-Judge evaluators for experiment analysis.

**Full documentation**: `.claude/skills/llm-judge-development/SKILL.md`

## Quick Start

1. Read the skill: `.claude/skills/llm-judge-development/SKILL.md`
2. Copy `_template/` to `{judge_name}/`
3. Follow the workflow in the skill docs

## Available Judges

| Judge | Purpose | Status |
|-------|---------|--------|
| introspection-detection | Evaluates whether model demonstrates awareness of injected concepts | Calibrated (90.5% test accuracy) |

## Usage

```python
from judges.{judge_name}.judge import evaluate

result = evaluate(
    response="...",
    injected_concept="..."
)

print(result.answer)     # "pass" or "fail"
print(result.reasoning)  # Explanation
```
