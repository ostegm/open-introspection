#!/bin/bash
# Sweep configurations for introspection experiment (3B model)
#
# Based on EXPERIMENT_LOG.md findings:
# - 3B model has 36 layers
# - 2/3 through = layer 24, optimal found at layer 30 (83%)
# - Sweet spot: effective magnitude 70-100 (strength 2-2.5 for norms ~40)
# - Repetition degeneracy starts at eff mag ~120 (strength 3+)
#
# Inject styles:
# - "all": Inject at all positions (prompt + generation) - original behavior
# - "generation": Only inject during generation (matches Anthropic paper methodology)

set -e

# Layers to test: band around 2/3 (24) through optimal (30)
LAYERS=(20 24 28 30)

# Strength multipliers (raw injection strength, not auto-scaled)
STRENGTHS=(2.0 2.5 3.0)

# Prompt versions
PROMPTS=(v1 v2)

# Inject styles: "all" (original) or "generation" (paper methodology)
# Default to just "all" for backward compatibility; add "generation" to compare
INJECT_STYLES=(all)

echo "========================================"
echo "Introspection Experiment Sweep"
echo "========================================"
echo "Model: 3B (36 layers)"
echo "Layers: ${LAYERS[*]}"
echo "Strengths: ${STRENGTHS[*]}"
echo "Prompts: ${PROMPTS[*]}"
echo "Inject styles: ${INJECT_STYLES[*]}"
echo "Total runs: $((${#LAYERS[@]} * ${#STRENGTHS[@]} * ${#PROMPTS[@]} * ${#INJECT_STYLES[@]}))"
echo "========================================"
echo

for layer in "${LAYERS[@]}"; do
    for strength in "${STRENGTHS[@]}"; do
        for prompt in "${PROMPTS[@]}"; do
            for inject_style in "${INJECT_STYLES[@]}"; do
                echo "----------------------------------------"
                echo "Running: layer=$layer strength=$strength prompt=$prompt inject_style=$inject_style"
                echo "----------------------------------------"

                uv run python experiments/03_introspection_no_metrics.py \
                    --model 3b \
                    --layer "$layer" \
                    --strength "$strength" \
                    --prompt "$prompt" \
                    --inject-style "$inject_style"

                echo
            done
        done
    done
done

echo "========================================"
echo "Sweep complete! Results saved in data/"
echo "========================================"
