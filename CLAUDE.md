# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project replicates experiments from Anthropic's "Investigating Introspection in Language Models" paper using open source models. It uses TransformerLens to extract concept vectors from model activations and inject them during inference to test whether models can detect/identify injected "thoughts."

## Commands

```bash
# Install dependencies
uv sync

# Run experiments
uv run python experiments/01_setup_sanity_check.py
uv run python experiments/02_concept_extraction.py
uv run python experiments/03_introspection_test.py

# Linting and type checking
uv run ruff check src/ experiments/
uv run ruff check --fix src/ experiments/  # auto-fix
uv run mypy src/

# Run tests
uv run pytest
```

## Architecture

### Core Pipeline

1. **Model Loading** (`model.py`): Wraps HuggingFace models with TransformerLens's `HookedTransformer` for activation access. Handles device detection (MPS/CUDA/CPU) and optional 4-bit quantization.

2. **Concept Extraction** (`concept_extraction.py`): Extracts concept vectors using mean subtraction:
   - Run "Tell me about {concept}" through model
   - Subtract mean activations from baseline words
   - Result: a direction in activation space representing that concept

3. **Introspection Testing** (`introspection.py`): Injects concept vectors into the residual stream during generation using TransformerLens hooks, then asks the model if it detects anything unusual.

4. **Analysis Modules**:
   - `layer_sweep.py`: Tests which layers are most effective for injection
   - `thought_text.py`: Tests if model can distinguish injected thoughts from visible text
   - `intentional_control.py`: Tests if model can control its internal representations
   - `grading.py`: Uses Claude API to grade introspection responses

### Key Concepts

- **Residual Stream**: The main information highway between layers (`hook_resid_post`). This is where concept vectors are injected.
- **Injection Strength**: Multiplier for how strongly to add the concept vector (typically 2.0)
- **Layer Selection**: Best results typically come from injecting at ~2/3 through the model

### TransformerLens Patterns

```python
# Run with activation cache
logits, cache = model.run_with_cache(tokens)
activations = cache["blocks.{layer}.hook_resid_post"]

# Inject during generation
def hook(activation, hook):
    activation[:, :, :] += strength * concept_vector
    return activation

with model.hooks([(f"blocks.{layer}.hook_resid_post", hook)]):
    output = model.generate(tokens, ...)
```

## Code Quality

### Linting & Type Checking

Run before committing:
```bash
uv run ruff check --fix src/ experiments/
uv run mypy src/
```

Ruff enforces: pycodestyle, isort, flake8-bugbear, type-checking imports, annotations. Mypy is configured strict.

### Type Hints

- Use TypedDict for structured returns (see `TrialResult`, `ExperimentResults`, etc.)
- Use TYPE_CHECKING blocks for TransformerLens imports to avoid runtime overhead
- Avoid explicit type annotations on `model.to_tokens()`, `model.generate()`, `model.to_string()`—their stubs have broad union types that confuse Pylance

### Unit Tests

Write tests for utility functions and data transformations. Testing full experiment pipelines is impractical (requires GPU, model downloads), but core logic like vector operations and result parsing should be tested.

### Code Tidiness

- Keep functions focused—extraction, injection, and analysis are separate concerns
- Use the TypedDict return types to document what functions produce
- Experiment scripts in `experiments/` can be more exploratory; library code in `src/` should be clean
