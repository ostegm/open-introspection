# Open Introspection

A hands-on project for learning mechanistic interpretability techniques while replicating Anthropic's ["Investigating Introspection in Language Models"](https://transformer-circuits.pub/2025/introspection/index.html) paper using open source models.

## Goals

1. **Learn by doing**: Understand concept vectors, activation steering, and how to probe model internals using TransformerLens
2. **Replicate on open models**: Test whether smaller open source models (Qwen, Llama, Mistral) exhibit introspective capabilities similar to what Anthropic found in Claude

## What's Being Replicated

The paper found that models can sometimes:
- Detect when artificial "thoughts" (activation vectors) are injected into their processing
- Identify what concept was injected
- Distinguish between injected thoughts and actual text input
- Control their internal representations when instructed

This project implements the core techniques:
- **Concept vector extraction** via mean subtraction
- **Activation injection** into the residual stream
- **Introspection testing** to see if models notice injections

## Quick Start

```bash
# Clone and install
git clone <repo>
cd open-introspection
uv sync

# Run the sanity check (loads model, verifies activation access)
uv run python experiments/01_setup_sanity_check.py

# Extract concept vectors and test steering
uv run python experiments/02_concept_extraction.py

# Run introspection experiments
uv run python experiments/03_introspection_test.py
```

## Project Structure

```
src/open_introspection/    # Core library
  model.py                 # Model loading with TransformerLens
  concept_extraction.py    # Extract concept vectors from activations
  introspection.py         # Test if models detect injected concepts

experiments/               # Runnable experiment scripts
judges/                    # LLM-as-judge evaluators for grading outputs
data/                      # Results storage
blog/                      # Research notes (GitHub Pages)
blueprint.md               # Detailed research plan and background
```

## Hardware

- **Mac (16-32GB)**: Qwen2.5-3B or 7B with float16/4-bit quantization
- **GPU server**: Qwen2.5-14B/32B for experiments closer to paper's scale

## Key Research Questions

- At what model size does introspection emerge?
- Do instruction-tuned models perform better than base models?
- Which concept categories are easiest to detect (emotions vs objects vs abstracts)?
- Do concept vectors transfer between model families?

## Research Notes

Follow along with ongoing experiments and findings:

**[Cloudripper Labs Blog](https://ostegm.github.io/open-introspection/blog/)** - Research notes on introspection experiments with open-weight models.

## References

- [Investigating Introspection in Language Models](https://transformer-circuits.pub/2025/introspection/index.html) - Anthropic
- [TransformerLens Documentation](https://neelnanda-io.github.io/TransformerLens/)
- [Steering GPT-2-XL by Adding an Activation Vector](https://arxiv.org/abs/2308.10248) - Turner et al.
