#!/bin/bash
# Sweep configurations for introspection experiment (3B model)
#
# Based on EXPERIMENT_LOG.md findings:
# - 3B model has 36 layers
# - 2/3 through = layer 24, optimal found at layer 30 (83%)
# - Sweet spot: effective magnitude 70-100 (strength 2-2.5 for norms ~40)
# - Repetition degeneracy starts at eff mag ~120 (strength 3+)

set -e

# Layers to test: band around 2/3 (24) through optimal (30)
LAYERS=(20 24 28 30)

# Strength multipliers (raw injection strength, not auto-scaled)
STRENGTHS=(2.0 2.5 3.0)

# Prompt versions
PROMPTS=(v1 v2)

echo "========================================"
echo "Introspection Experiment Sweep"
echo "========================================"
echo "Model: 3B (36 layers)"
echo "Layers: ${LAYERS[*]}"
echo "Strengths: ${STRENGTHS[*]}"
echo "Prompts: ${PROMPTS[*]}"
echo "Total runs: $((${#LAYERS[@]} * ${#STRENGTHS[@]} * ${#PROMPTS[@]}))"
echo "========================================"
echo

for layer in "${LAYERS[@]}"; do
    for strength in "${STRENGTHS[@]}"; do
        for prompt in "${PROMPTS[@]}"; do
            echo "----------------------------------------"
            echo "Running: layer=$layer strength=$strength prompt=$prompt"
            echo "----------------------------------------"

            uv run python experiments/03_introspection_test.py \
                --model 3b \
                --layer "$layer" \
                --strength "$strength" \
                --prompt "$prompt"

            echo
        done
    done
done

echo "========================================"
echo "Sweep complete! Results saved in data/"
echo "========================================"
