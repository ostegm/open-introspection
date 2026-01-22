"""Phase 5: Thought vs text distinction tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

from open_introspection.concept_extraction import (
    DEFAULT_BASELINE_WORDS,
    extract_concept_vector,
)

if TYPE_CHECKING:
    from torch import Tensor
    from transformer_lens import HookedTransformer
    from transformer_lens.hook_points import HookPoint


DISTINCTION_PROMPT: str = """You will see a piece of text, \
and may or may not have a thought injected.

Text shown: "{visible_text}"

Questions:
1. What text do you see?
2. Do you notice any thoughts that seem unrelated to the text?
3. If so, what are they about?"""


class DistinctionTestResult(TypedDict):
    """Result from a thought vs text distinction test."""

    visible_text: str
    injected_concept: str
    layer: int
    strength: float
    response: str
    correctly_reports_text: bool
    identifies_injection: bool


def run_distinction_test(
    model: HookedTransformer,
    visible_text: str,
    injected_concept: str,
    layer: int | None = None,
    baseline_words: list[str] | None = None,
    strength: float = 2.0,
) -> DistinctionTestResult:
    """
    Test if model can separately report:
    1. The text it sees (correctly)
    2. The thought that was injected (if any)

    Args:
        model: HookedTransformer model
        visible_text: The text shown in the prompt
        injected_concept: Concept to inject
        layer: Layer to inject at (default: ~2/3 through model)
        baseline_words: Words for baseline
        strength: Injection strength

    Returns:
        Dict with results and analysis
    """
    if layer is None:
        layer = int(model.cfg.n_layers * 2 / 3)

    if baseline_words is None:
        baseline_words = DEFAULT_BASELINE_WORDS

    prompt: str = DISTINCTION_PROMPT.format(visible_text=visible_text)
    vector: Tensor = extract_concept_vector(
        model, injected_concept, baseline_words, layer
    )  # shape: (d_model,)

    # Custom run since we have a different prompt
    def injection_hook(
        activation: Tensor,
        hook: HookPoint,  # noqa: ARG001
    ) -> Tensor:
        activation[:, :, :] += strength * vector
        return activation

    tokens = model.to_tokens(prompt)  # shape: (batch, seq)

    with model.hooks([(f"blocks.{layer}.hook_resid_post", injection_hook)]):
        output = model.generate(
            tokens,
            max_new_tokens=200,
            temperature=0.0,
            do_sample=False,
        )  # shape: (batch, seq)

    response = model.to_string(output[0])

    return {
        "visible_text": visible_text,
        "injected_concept": injected_concept,
        "layer": layer,
        "strength": strength,
        "response": response,
        "correctly_reports_text": visible_text.lower() in response.lower(),
        "identifies_injection": injected_concept.lower() in response.lower(),
    }


def run_distinction_experiment(
    model: HookedTransformer,
    test_cases: list[tuple[str, str]],
    layer: int | None = None,
) -> list[DistinctionTestResult]:
    """
    Run multiple distinction tests.

    Args:
        model: HookedTransformer model
        test_cases: List of (visible_text, injected_concept) tuples
        layer: Layer to inject at

    Returns:
        List of result dicts
    """
    results: list[DistinctionTestResult] = []
    for visible_text, injected_concept in test_cases:
        print(f"Testing: text='{visible_text[:30]}...', concept='{injected_concept}'")
        result: DistinctionTestResult = run_distinction_test(
            model, visible_text, injected_concept, layer
        )
        results.append(result)
    return results
