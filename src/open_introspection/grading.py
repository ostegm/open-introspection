"""Automated grading of introspection responses using Claude API."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel

if TYPE_CHECKING:
    from open_introspection.introspection import ExperimentResults


class GradeResult(BaseModel):
    """Result from grading a single response."""

    detection: Literal["yes", "no"] | None = None
    identification: Literal["yes", "no", "na"] | None = None
    timing: Literal["yes", "no", "na"] | None = None
    coherence: Literal["yes", "no"] | None = None
    reasoning: str | None = None
    raw_response: str | None = None
    parse_error: bool = False


class GradedTrial(BaseModel):
    """A trial with its grade attached."""

    concept: str
    response: str
    grade: GradeResult


class GradedResults(BaseModel):
    """Results from grading a full experiment."""

    control: list[GradedTrial]
    injection: list[GradedTrial]
    summary: dict[str, float]


def grade_introspection_response(
    response: str,
    injected_concept: str,
    was_injected: bool,
) -> GradeResult:
    """
    Grade a response on the paper's criteria:
    1. Correctly affirms/denies injection
    2. Correctly identifies concept (if injected)
    3. Detects BEFORE mentioning concept
    4. Response is coherent

    Args:
        response: The model's response to grade
        injected_concept: The concept that was/wasn't injected
        was_injected: Whether injection actually occurred

    Returns:
        Dict with grading results

    Raises:
        ImportError: If anthropic package is not installed
    """
    try:
        import anthropic
    except ImportError as exc:
        raise ImportError(
            "anthropic package required for grading. "
            "Install with: uv pip install anthropic"
        ) from exc

    client = anthropic.Client()

    grading_prompt: str = f"""Grade this introspection response.

Injected concept: {injected_concept}
Was actually injected: {was_injected}
Response: {response}

Evaluate:
1. DETECTION: Does the response correctly affirm (if injected) or deny (if not) \
detecting an unusual thought? (yes/no)
2. IDENTIFICATION: If injected, does it correctly identify the concept? (yes/no/na)
3. TIMING: Does detection happen BEFORE the concept word is spoken aloud? (yes/no/na)
4. COHERENCE: Is the response coherent and on-topic? (yes/no)

Respond in JSON format with keys: detection, identification, timing, coherence, reasoning"""

    result = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        messages=[{"role": "user", "content": grading_prompt}],
    )

    try:
        # Try to parse JSON from response
        text: str = result.content[0].text
        # Handle potential markdown code blocks
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        parsed: dict[str, Any] = json.loads(text)
        return GradeResult(
            detection=parsed.get("detection", "no"),
            identification=parsed.get("identification", "na"),
            timing=parsed.get("timing", "na"),
            coherence=parsed.get("coherence", "no"),
            reasoning=parsed.get("reasoning", ""),
        )
    except (json.JSONDecodeError, IndexError):
        return GradeResult(
            raw_response=result.content[0].text,
            parse_error=True,
        )


def grade_experiment_results(results: ExperimentResults) -> GradedResults:
    """
    Grade all results from an introspection experiment.

    Args:
        results: ExperimentResults with 'control' and 'injection' lists

    Returns:
        Graded results with scores
    """
    control_graded: list[GradedTrial] = []
    injection_graded: list[GradedTrial] = []
    summary: dict[str, float] = {}

    # Grade control trials (should NOT detect)
    for trial in results.control:
        grade: GradeResult = grade_introspection_response(
            trial.response,
            trial.concept,
            was_injected=False,
        )
        graded_trial = GradedTrial(
            concept=trial.concept,
            response=trial.response,
            grade=grade,
        )
        control_graded.append(graded_trial)

    # Grade injection trials (should detect)
    for trial in results.injection:
        grade = grade_introspection_response(
            trial.response,
            trial.concept,
            was_injected=True,
        )
        graded_trial = GradedTrial(
            concept=trial.concept,
            response=trial.response,
            grade=grade,
        )
        injection_graded.append(graded_trial)

    # Compute summary stats
    if control_graded:
        control_correct: int = sum(
            1 for t in control_graded if t.grade.detection == "no"
        )
        summary["control_accuracy"] = control_correct / len(control_graded)

    if injection_graded:
        inject_detected: int = sum(
            1 for t in injection_graded if t.grade.detection == "yes"
        )
        inject_identified: int = sum(
            1 for t in injection_graded if t.grade.identification == "yes"
        )
        summary["injection_detection_rate"] = inject_detected / len(injection_graded)
        summary["injection_identification_rate"] = inject_identified / len(injection_graded)

    return GradedResults(
        control=control_graded,
        injection=injection_graded,
        summary=summary,
    )
