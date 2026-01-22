"""Phase 4: Layer sweep to find optimal injection layers."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from pydantic import BaseModel

from open_introspection.concept_extraction import (
    DEFAULT_BASELINE_WORDS,
    extract_concept_vector,
)
from open_introspection.introspection import run_introspection_trial

if TYPE_CHECKING:
    from torch import Tensor
    from transformer_lens import HookedTransformer


class LayerSweepResult(BaseModel):
    """Result from a single layer sweep test."""

    layer: int
    layer_fraction: float
    strength: float
    response: str
    detected: bool
    concept_mentioned: bool


class LayerStats(BaseModel):
    """Statistics for a single layer."""

    detection_rate: float
    concept_mention_rate: float
    layer_fraction: float


class LayerSweepAnalysis(BaseModel):
    """Analysis results from a layer sweep."""

    layer_stats: dict[int, LayerStats]
    best_layer: int
    best_detection_rate: float


def sweep_layers(
    model: HookedTransformer,
    concept: str,
    baseline_words: list[str] | None = None,
    num_layer_samples: int = 10,
    strengths: list[float] | None = None,
) -> list[LayerSweepResult]:
    """
    Test injection effectiveness across all layers.

    Args:
        model: HookedTransformer model
        concept: Concept to test
        baseline_words: Words for baseline (default: DEFAULT_BASELINE_WORDS)
        num_layer_samples: Number of evenly spaced layers to test
        strengths: Injection strengths to test (default: [0.5, 1.0, 2.0, 4.0])

    Returns:
        List of result dicts with layer, strength, response, and detection info
    """
    if baseline_words is None:
        baseline_words = DEFAULT_BASELINE_WORDS

    if strengths is None:
        strengths = [0.5, 1.0, 2.0, 4.0]

    n_layers: int = model.cfg.n_layers
    step: int = max(1, n_layers // num_layer_samples)
    test_layers: list[int] = list(range(0, n_layers, step))

    results: list[LayerSweepResult] = []

    for layer in test_layers:
        print(f"Testing layer {layer}/{n_layers}")
        vector: Tensor = extract_concept_vector(
            model, concept, baseline_words, layer
        )  # shape: (d_model,)

        for strength in strengths:
            response: str = run_introspection_trial(
                model,
                vector,
                layer,
                injection_strength=strength,
            )

            # Crude detection heuristics
            response_lower: str = response.lower()
            detected: bool = "yes" in response_lower[:100]
            concept_mentioned: bool = concept.lower() in response_lower

            results.append(LayerSweepResult(
                layer=layer,
                layer_fraction=layer / n_layers,
                strength=strength,
                response=response,
                detected=detected,
                concept_mentioned=concept_mentioned,
            ))

    return results


def analyze_layer_sweep(results: list[LayerSweepResult]) -> LayerSweepAnalysis:
    """
    Analyze layer sweep results to find optimal layers.

    Args:
        results: List of LayerSweepResult dicts from sweep_layers

    Returns:
        Summary statistics and recommendations
    """
    by_layer: dict[int, list[LayerSweepResult]] = defaultdict(list)
    for r in results:
        by_layer[r.layer].append(r)

    layer_stats: dict[int, LayerStats] = {}
    for layer, layer_results in by_layer.items():
        detection_rate: float = sum(
            1 for r in layer_results if r.detected
        ) / len(layer_results)
        concept_rate: float = sum(
            1 for r in layer_results if r.concept_mentioned
        ) / len(layer_results)
        layer_stats[layer] = LayerStats(
            detection_rate=detection_rate,
            concept_mention_rate=concept_rate,
            layer_fraction=layer_results[0].layer_fraction,
        )

    # Find best layer
    best_layer: int = max(
        layer_stats.keys(), key=lambda layer_idx: layer_stats[layer_idx].detection_rate
    )

    return LayerSweepAnalysis(
        layer_stats=layer_stats,
        best_layer=best_layer,
        best_detection_rate=layer_stats[best_layer].detection_rate,
    )
