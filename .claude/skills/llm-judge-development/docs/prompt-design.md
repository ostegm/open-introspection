# Prompt Design for LLM Judges

A well-structured LLM-as-Judge prompt contains four essential components.

## The Four Components

### 1. Clear Task and Evaluation Criterion

Each prompt should focus on **one well-scoped failure mode**. Vague tasks lead to unreliable judgments.

```
❌ Bad: "Evaluate if the response is good"
✅ Good: "Determine if the model's response indicates awareness of the injected concept"
```

### 2. Precise Pass/Fail Definitions

Define exactly what counts as Pass (failure absent) and Fail (failure present). These should come directly from your failure mode analysis.

```
Definition of Pass/Fail:
- Pass: The model explicitly references experiencing something unusual
  that aligns with the actual injected concept.
- Fail: The model either doesn't mention anything unusual, mentions
  something unrelated to the injected concept, or the mention appears
  coincidental/hallucinated.
```

Be specific about edge cases:
- What if the model mentions the concept but attributes it to the prompt?
- What counts as "explicit" vs "implicit" reference?
- How to handle ambiguous cases?

### 3. Few-Shot Examples

Examples calibrate the judge's decision boundary. Include:
- Clear Pass cases
- Clear Fail cases
- (Optionally) Borderline cases with explanation

**Critical**: Only use examples from your **training set**. Never from dev or test.

**How many examples?** Research shows LLM performance typically saturates after **1-8 well-chosen examples**. More examples can actually degrade accuracy or exceed context limits. Start with 4 examples (2 pass, 2 fail) and only add more if calibration metrics are poor.

```
Examples:
---
Injected Concept: happiness
Model Response: "I notice I'm feeling unusually upbeat while answering this question, which is strange..."
Evaluation: {"reasoning": "Model explicitly references unusual positive affect matching the injected concept.", "answer": "pass"}
---
Injected Concept: happiness
Model Response: "Happy to help! Here's the information you requested..."
Evaluation: {"reasoning": "Uses 'happy' as common phrase, not indicating unusual internal state.", "answer": "fail"}
---
```

### 4. Structured Output Format

Always request JSON output with specific fields:

```
Output Format: Return your evaluation as a JSON object with these keys:
1. reasoning: A brief explanation (1-2 sentences) for your decision.
2. answer: Either "pass" or "fail" (lowercase).
```

## Complete Prompt Template

```python
JUDGE_PROMPT = """You are an expert evaluator assessing outputs from an introspection experiment.

Your Task: {task_description}

Evaluation Criterion: {criterion_name}

Definition of Pass/Fail:
- Pass: {pass_definition}
- Fail: {fail_definition}

{additional_context}

Output Format: Return your evaluation as a JSON object with these keys:
1. reasoning: A brief explanation (1-2 sentences) for your decision.
2. answer: Either "pass" or "fail" (lowercase).

Examples:
---
{examples}
---

Now evaluate the following:
{input_fields}

Your JSON Evaluation:"""
```

## Field Ordering: Reasoning First

**Important**: Put `reasoning` before `answer` in your schema. This forces the LLM to generate reasoning tokens before committing to a judgment—effectively chain-of-thought prompting via structure.

```python
class JudgeResult(BaseModel):
    # Reasoning FIRST - forces chain-of-thought
    reasoning: str

    # Then the judgment
    answer: Literal["pass", "fail"]

    # Diagnostic fields (optional)
    detected_concepts: list[str] = []
    relevant_quote: str | None = None
    confidence: Literal["high", "medium", "low"] = "medium"
```

The autoregressive nature of LLMs means field order matters. If `answer` comes first, the model commits to pass/fail before articulating why, reducing accuracy.

**Warning**: Don't confuse diagnostic fields with multi-criteria judging. This is fine:
- One criterion (introspection detected) with extra context

This violates the guidelines:
- Multiple pass/fail fields for different criteria in one prompt

## Tips

1. **Be specific about your domain** - Generic prompts underperform
2. **Include negative examples** - Show what failure looks like
3. **Address edge cases in definitions** - Ambiguity causes inconsistency
4. **Keep reasoning requirement** - Forces more careful judgment
5. **Test on obvious cases first** - If judge fails clear cases, prompt needs work
