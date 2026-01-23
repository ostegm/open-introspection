# API Patterns for LLM Judges

This document covers structured output patterns for OpenAI and Anthropic APIs.

## Model Selection

### OpenAI (Preferred for this project)

| Model          | Use Case                                                   | Pricing (per 1M tokens)    |
| -------------- | ---------------------------------------------------------- | -------------------------- |
| `gpt-5-mini` | **Default choice** - good balance of cost/capability | Check current pricing      |
| `gpt-5-nano` | Simple, clear-cut judgments                                | $0.05 input / $0.40 output |
| `gpt-5`      | Complex reasoning (rarely needed)                          | Higher                     |

**Recommendation**: Start with `gpt-5-mini`. If calibration is good (TPR/TNR >90%), try `gpt-5-nano` to reduce costs. Only escalate to `gpt-5` if smaller models can't calibrate well.

### Anthropic

| Model                        | Use Case                              | Pricing (per 1M tokens)  |
| ---------------------------- | ------------------------------------- | ------------------------ |
| `claude-haiku-4-5-20251001`  | **Default choice** - fast and cheap   | $1 input / $5 output     |
| `claude-sonnet-4-5-20251101` | If Haiku can't calibrate well         | $3 input / $15 output    |

**Recommendation**: Start with Haiku. Only use Sonnet if Haiku TPR/TNR is inadequate. Note: GPT-5 Nano is ~18x cheaper than Haiku, so prefer OpenAI for this project.

## OpenAI Structured Outputs

Use the `responses.parse` method with a Pydantic model:

```python
from openai import OpenAI
from pydantic import BaseModel
from typing import Literal

client = OpenAI()

class JudgeResult(BaseModel):
    reasoning: str  # First: forces chain-of-thought
    answer: Literal["pass", "fail"]

def evaluate(response: str, injected_concept: str) -> JudgeResult:
    """Run the introspection judge on a single response."""

    user_content = f"""Injected Concept: {injected_concept}
Model Response: {response}

Your JSON Evaluation:"""

    result = client.responses.parse(
        model="gpt-5-mini",
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        text_format=JudgeResult,
    )

    return result.output_parsed
```

### Key Points

- Use `client.responses.parse()`
- `text_format` takes a Pydantic model directly
- Use `input` (not `messages`) for the conversation
- Returns `result.output_parsed` as validated Pydantic instance
- Use `Literal["pass", "fail"]` to constrain answer values

### Error Handling

```python
from openai import OpenAI, APIError

def evaluate_safe(response: str, injected_concept: str) -> JudgeResult | None:
    """Run judge with error handling."""
    try:
        result = client.responses.parse(
            model="gpt-5-mini",
            input=[...],
            text_format=JudgeResult,
        )
        return result.output_parsed
    except APIError as e:
        print(f"API error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None
```

## Anthropic Structured Outputs

For cases where Anthropic is needed:

```python
from anthropic import Anthropic
from pydantic import BaseModel
from typing import Literal

client = Anthropic()

class JudgeResult(BaseModel):
    reasoning: str  # First: forces chain-of-thought
    answer: Literal["pass", "fail"]

def evaluate_anthropic(response: str, injected_concept: str) -> JudgeResult:
    """Run judge using Anthropic API."""

    result = client.beta.messages.parse(
        model="claude-haiku-4-5-20251001",  # Prefer Haiku for cost
        max_tokens=1024,
        betas=["structured-outputs-2025-11-13"],
        messages=[
            {
                "role": "user",
                "content": f"{SYSTEM_PROMPT}\n\nInjected Concept: {injected_concept}\nModel Response: {response}"
            }
        ],
        output_format=JudgeResult,
    )

    return result.parsed_output
```

### Key Points

- Use `client.beta.messages.parse()`
- **Must include** `betas=["structured-outputs-2025-11-13"]`
- `output_format` takes a Pydantic model directly
- Returns `result.parsed_output` as validated Pydantic instance

### Alternative: Using transform_schema

```python
from anthropic import Anthropic, transform_schema

# With .create() - requires transform_schema()
response = client.beta.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=1024,
    betas=["structured-outputs-2025-11-13"],
    messages=[...],
    output_format={
        "type": "json_schema",
        "schema": transform_schema(JudgeResult),
    }
)

# Parse manually
result = JudgeResult.model_validate_json(response.content[0].text)
```

## Batch Processing

For running judges on many examples efficiently:

### OpenAI with asyncio

```python
import asyncio
from openai import AsyncOpenAI

async_client = AsyncOpenAI()

async def evaluate_async(example: LabeledExample) -> tuple[str, JudgeResult]:
    """Evaluate single example asynchronously."""
    result = await async_client.responses.parse(
        model="gpt-5-mini",
        input=[...],
        text_format=JudgeResult,
    )
    return example.id, result.output_parsed

async def evaluate_batch(examples: list[LabeledExample]) -> dict[str, JudgeResult]:
    """Evaluate all examples concurrently."""
    tasks = [evaluate_async(ex) for ex in examples]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    output = {}
    for result in results:
        if isinstance(result, Exception):
            print(f"Error: {result}")
            continue
        example_id, judge_result = result
        output[example_id] = judge_result

    return output

# Usage
results = asyncio.run(evaluate_batch(test_data))
```

### Rate Limiting

Add rate limiting for large batches:

```python
import asyncio
from asyncio import Semaphore

async def evaluate_batch_limited(
    examples: list[LabeledExample],
    max_concurrent: int = 10
) -> dict[str, JudgeResult]:
    """Evaluate with concurrency limit."""
    semaphore = Semaphore(max_concurrent)

    async def limited_evaluate(example):
        async with semaphore:
            return await evaluate_async(example)

    tasks = [limited_evaluate(ex) for ex in examples]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    # ... rest same as above
```

## Pydantic Model Patterns

### Basic Judge Result

```python
class JudgeResult(BaseModel):
    reasoning: str  # First: forces chain-of-thought
    answer: Literal["pass", "fail"]
```

### With Diagnostic Fields

```python
class JudgeResult(BaseModel):
    # Reasoning FIRST - forces chain-of-thought
    reasoning: str

    # Then the judgment
    answer: Literal["pass", "fail"]

    # Diagnostics (optional, for analysis)
    detected_concepts: list[str] = []
    relevant_quote: str | None = None
    confidence: Literal["high", "medium", "low"] = "medium"
```

### With Validation

```python
from pydantic import BaseModel, field_validator

class JudgeResult(BaseModel):
    reasoning: str  # First: forces chain-of-thought
    answer: Literal["pass", "fail"]

    @field_validator("reasoning")
    @classmethod
    def reasoning_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Reasoning cannot be empty")
        return v
```

## Temperature and Reproducibility

For consistent judge behavior:

```python
# OpenAI
result = client.responses.parse(
    model="gpt-5-mini",
    input=[...],
    text_format=JudgeResult,
    temperature=0,  # Deterministic output
)

# Anthropic
result = client.beta.messages.parse(
    model="claude-haiku-4-5-20251001",
    messages=[...],
    output_format=JudgeResult,
    betas=["structured-outputs-2025-11-13"],
    temperature=0,
)
```

**Note**: Even with temperature=0, outputs may vary slightly across API versions. Always record model version in calibration results.

## Cost Estimation

Rough token estimates for judge calls:

| Component                  | Tokens |
| -------------------------- | ------ |
| System prompt              | ~500   |
| Few-shot examples (4)      | ~800   |
| Input (response + concept) | ~200   |
| Output (JSON result)       | ~50    |
| **Total per call**         | ~1,550 |

**Model pricing (per 1M tokens):**

| Model              | Input   | Output  |
| ------------------ | ------- | ------- |
| GPT-5 Nano         | $0.05   | $0.40   |
| GPT-5 Mini         | $0.25   | $2.00   |
| Claude Haiku 4.5   | $1.00   | $5.00   |
| Claude Sonnet 4.5  | $3.00   | $15.00  |

**Cost per 1,000 judge calls** (~1.5M input, ~0.05M output):

| Model              | Input Cost | Output Cost | Total   |
| ------------------ | ---------- | ----------- | ------- |
| GPT-5 Nano         | $0.08      | $0.02       | ~$0.10  |
| GPT-5 Mini         | $0.38      | $0.10       | ~$0.48  |
| Claude Haiku 4.5   | $1.50      | $0.25       | ~$1.75  |
| Claude Sonnet 4.5  | $4.50      | $0.75       | ~$5.25  |

**Recommendation**: For calibration iterations (dev set ~50 examples × 10 iterations = 500 calls), costs are negligible. GPT-5 Nano is ~18x cheaper than Haiku—use OpenAI credits first. Optimize model choice when running on large production datasets.
