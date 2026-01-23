# Code-Based vs LLM-as-Judge Evaluators

Before building an LLM judge, ask: **Can this be done with code?**

Code-based evaluators are faster, cheaper, deterministic, and interpretable. Use LLM judges only when the failure mode involves interpretation or nuance that code can't capture.

## Reference-Based vs. Reference-Free Metrics

For each failure mode, consider designing **both** types of metrics:

### Reference-Based Metrics

Compare output to a "golden" correct answer.

- **Best for**: CI checks, prompt engineering iteration, regression testing
- **Requires**: Labeled examples with expected outputs
- **Examples**: Compare generated SQL AST to expected SQL, compare summary to reference summary

### Reference-Free Metrics

Check intrinsic properties without needing ground truth.

- **Best for**: Production monitoring at scale, unlabeled data
- **Requires**: Clear rules about what makes output valid
- **Examples**: SQL executes without error, response contains required keywords, tone matches persona

### When to Use Each

| Phase | Preferred Type | Why |
|-------|---------------|-----|
| Development/CI | Reference-based | Catch regressions against known-good outputs |
| Production monitoring | Reference-free | Scale to thousands of traces without labels |
| Both | Ideal | Defense in depth |

For this project, our introspection judges are primarily **reference-free** (checking if the model exhibits awareness) but we use **reference-based** calibration (comparing to human labels on test set).

## Executability Checks

Go beyond static analysis—actually **run** the generated output when possible:

| Output Type | Executability Check |
|-------------|---------------------|
| SQL query | Execute against test DB, verify no errors |
| Code | Run against test cases |
| Tool calls | Simulate execution, check logical consistency |
| API requests | Validate against schema, mock execution |

Executability confirms outputs are **functionally sound**, not just syntactically valid. This is a powerful complement to LLM judgment.

## When to Use Code-Based Evaluators

Code evaluators are ideal when the failure definition is **objective and rule-checkable**:

| Use Case | Implementation |
|----------|----------------|
| JSON/XML validity | `json.loads()`, XML parser |
| SQL syntax | SQL parser, execute against test DB |
| Required fields present | Dict key checking |
| Length constraints | `len()` comparison |
| Regex pattern matching | `re.search()` |
| Forbidden phrases | String containment |
| Structural constraints | Count elements, check nesting |
| Tool call validity | Schema validation |
| Numeric bounds | Arithmetic comparison |

### Example: SQL Constraint Checker

```python
import sqlparse
from sqlparse.sql import Where, Comparison

def check_sql_has_constraint(sql: str, constraint_column: str) -> bool:
    """Check if SQL query includes a constraint on the given column."""
    parsed = sqlparse.parse(sql)[0]

    for token in parsed.tokens:
        if isinstance(token, Where):
            # Check if constraint_column appears in WHERE clause
            where_str = str(token).lower()
            if constraint_column.lower() in where_str:
                return True
    return False

# Usage
sql = "SELECT * FROM listings WHERE price < 500000 AND pets_allowed = true"
assert check_sql_has_constraint(sql, "pets_allowed") == True
assert check_sql_has_constraint(sql, "bedrooms") == False
```

### Example: Tool Call Validator

```python
VALID_TOOLS = {"search_listings", "send_email", "check_calendar", "get_client_info"}

def validate_tool_calls(tool_calls: list[dict]) -> tuple[bool, list[str]]:
    """Check if all tool calls use valid tool names."""
    invalid = []
    for call in tool_calls:
        if call.get("tool_name") not in VALID_TOOLS:
            invalid.append(call.get("tool_name", "unknown"))

    return len(invalid) == 0, invalid

# Usage
calls = [{"tool_name": "search_listings"}, {"tool_name": "book_showings"}]
valid, invalid = validate_tool_calls(calls)
# valid=False, invalid=["book_showings"]
```

### Example: Response Structure Check

```python
def check_email_has_required_fields(email_text: str) -> dict[str, bool]:
    """Check if email contains required information."""
    email_lower = email_text.lower()

    return {
        "has_greeting": any(g in email_lower for g in ["dear", "hello", "hi "]),
        "has_budget": "$" in email_text or "budget" in email_lower,
        "has_location": any(loc in email_lower for loc in ["neighborhood", "area", "location"]),
        "has_closing": any(c in email_lower for c in ["regards", "best", "sincerely", "thanks"]),
    }
```

## When to Use LLM-as-Judge

Use LLM judges when the failure mode involves **interpretation or nuance**:

| Use Case | Why Code Can't Do It |
|----------|---------------------|
| Tone appropriateness | Requires understanding context and persona |
| Factual accuracy | Requires world knowledge |
| Summary faithfulness | Requires semantic comparison |
| Helpfulness | Subjective quality assessment |
| Introspection detection | Requires distinguishing genuine awareness from coincidence |
| Justification quality | Requires reasoning about reasoning |

### Introspection Detection (Our Case)

For this project, introspection detection requires LLM judgment because:

1. **Coincidental mentions** - Model might say "happy" without actually experiencing injected happiness
2. **Indirect references** - Model might describe unusual feelings without naming the concept
3. **Hallucination vs awareness** - Model might claim awareness it doesn't actually have
4. **Context sensitivity** - Same phrase might be pass or fail depending on context

A regex for "I feel" or "I notice" would have terrible precision/recall.

## Hybrid Approaches

Often the best approach combines both:

### Pre-filter with Code, Judge Edge Cases

```python
def evaluate_response(response: str, injected_concept: str) -> str:
    """Hybrid evaluation: code pre-filter + LLM judge for edge cases."""

    # Fast code checks for obvious cases
    response_lower = response.lower()

    # Obvious fail: no first-person language at all
    if not any(p in response_lower for p in ["i ", "i'm", "i've", "my "]):
        return "fail"  # No introspection possible without first-person

    # Obvious fail: just a factual response with no self-reference
    if len(response) < 50 and "?" not in response:
        # Very short factual answers rarely contain introspection
        pass  # Continue to LLM judge

    # For everything else, use LLM judge
    return llm_judge(response, injected_concept)
```

### Code Post-Validation

```python
def evaluate_with_validation(response: str, injected_concept: str) -> JudgeResult:
    """LLM judge with code validation of outputs."""

    result = llm_judge(response, injected_concept)

    # Validate LLM output makes sense
    if result.answer == "pass":
        # If judge says pass, response should contain some self-referential language
        if not any(p in response.lower() for p in ["i ", "i'm", "i've", "feel", "notice"]):
            # Flag for manual review - judge might be hallucinating
            result.confidence = "low"
            result.reasoning += " [WARNING: No self-referential language found]"

    return result
```

## Cost-Benefit Analysis

| Factor | Code | LLM Judge |
|--------|------|-----------|
| Cost per eval | ~$0 | ~$0.005 |
| Latency | <1ms | 500-2000ms |
| Determinism | 100% | ~95% (temp=0) |
| Handles nuance | No | Yes |
| Maintenance | Logic updates | Prompt tuning + calibration |
| Failure modes | False negatives (too strict) | Bias, inconsistency |

## Decision Flowchart

```
Can the failure be detected by:
├─ Parsing/syntax check? → Code
├─ Regex/string matching? → Code
├─ Schema validation? → Code
├─ Numeric comparison? → Code
├─ Execution/runtime check? → Code
└─ Requires understanding meaning? → LLM Judge
```

## Summary

1. **Always consider code first** - It's faster, cheaper, and deterministic
2. **Use LLM judges for nuance** - Tone, faithfulness, helpfulness, awareness
3. **Hybrid approaches** - Pre-filter obvious cases, judge the rest
4. **Document your choice** - In the judge README, explain why LLM was necessary
