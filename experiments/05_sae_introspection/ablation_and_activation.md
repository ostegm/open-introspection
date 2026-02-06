# Causal Validation: Ablation & Activation Patching for Introspection Features

## Context

Phase 0 identified candidate metacognition features via correlational analysis (Cohen's d, set-based).
The next question: **are these features causally involved in introspection, or just along for the ride?**

## Necessity vs Sufficiency

These are independent properties — not complements.

|                          | Necessary                     | Not Necessary    |
| ------------------------ | ----------------------------- | ---------------- |
| **Sufficient**     | Unique critical path          | Redundant backup |
| **Not Sufficient** | Required but not enough alone | Irrelevant       |

### Denoising (clean → corrupted) tests SUFFICIENCY

Patch a component from the clean run into a corrupted run. If behavior restores → that component is sufficient.

```
Setup: Run model WITHOUT injection (corrupted = no introspection signal)
Patch: Inject clean activation of feature 3435 from an injection run
Check: Does model now report detecting something?
If yes → 3435 is sufficient to produce introspection-like output
```

### Noising (corrupted → clean) tests NECESSITY

Patch a component from the corrupted run into the clean run. If behavior breaks → that component is necessary.

```
Setup: Run model WITH injection (clean = introspection working)
Patch: Ablate feature 3435 (zero it / mean-ablate)
Check: Does model stop reporting detection?
If yes → 3435 is necessary for introspection
```

## The AND/OR Problem

### AND-like circuits (serial)

```
Injection → [Feature A] → [Feature B] → [Feature C] → Detection output

Ablate any one → behavior breaks (all appear necessary)
Denoise at C → restores behavior (C looks sufficient, but A and B
                did upstream work in the clean run implicitly)
```

Risk: Denoising at later layers gets "free" upstream computation, inflating sufficiency claims.

### OR-like circuits (parallel/redundant)

```
Injection → [Feature A] ─┐
            [Feature B] ─┼→ Detection output
            [Feature C] ─┘

Ablate A alone → behavior persists (B, C compensate)
Ablate A+B    → behavior persists (C compensates)
Ablate A+B+C  → breaks
```

Risk: Individual ablation says "not necessary" → wrong conclusion. The feature IS involved, just redundant.

## Practical Implications for a hypothetical set of features from early reasults

Note: These specific features may not matter at all, they are just used for demonstration of concepts.

### Candidate metacognition features (Layer 22)

| Feature | Label                          | Monitoring d | Hypothesis                          |
| ------- | ------------------------------ | ------------ | ----------------------------------- |
| 3435    | Hesitation/unfinished thoughts | +0.57        | Articulation difficulty             |
| 7130    | Self-as-computational-system   | +0.55        | Self-modeling                       |
| 925     | Unusual/meta engagement        | +0.50        | Meta-awareness                      |
| 49299   | Anesthesia/sedation            | +0.44        | "External consciousness alteration" |

These likely form an **OR-like circuit** — multiple features encoding related metacognitive concepts with redundancy. Ablating one probably won't break introspection.

### Experimental Protocol

**Test 1: Cluster ablation (necessity)**

```
Condition A setup (injection + monitoring)
Ablate ALL metacognition features: [3435, 7130, 925, 49299, ...]
→ Does detection rate drop from 80%?
→ If yes: metacognition cluster is necessary
→ If no: introspection uses different features than expected
```

**Test 2: Individual ablation within cluster (redundancy structure)**

```
Ablate each feature individually in condition A
→ If none matter individually: OR-like redundancy
→ If one matters: that's the critical path feature
→ Partial degradation: weighted contributions
```

**Test 3: Feature activation without injection (sufficiency)**

```
Condition D setup (no injection, neutral prompt)
Artificially activate metacognition features at condition-A levels
→ Does model hallucinate detecting something?
→ If yes: features are sufficient to produce introspection output
→ If no: need actual concept signal (injection) too
```

**Test 4: Cross-concept generalization**

```
Inject different concept (e.g., "joy" instead of "silence")
→ Do the SAME metacognition features fire?
→ If yes: general introspection circuit
→ If no: concept-specific features
```

## Key Pitfalls to Avoid

1. **"Ablation didn't matter" ≠ "feature isn't involved"** — could be redundancy
2. **"Denoising restored behavior" ≠ "this feature alone does it"** — downstream layers may get implicit clean upstream computation
3. **Don't test sufficiency and necessity with the same direction** — they require opposite patching setups
4. **SAE reconstruction error** — patching through encode→modify→decode loses information. Compare against activation-level patching as sanity check
5. **n=5 is exploratory** — causal claims need more trials

## Implementation Notes

### Naive approach (DON'T use this)

Full encode→modify→decode introduces reconstruction error on ALL features, not just the patched ones:

```python
# BAD: reconstruction error everywhere
def feature_ablation_hook_naive(activation, hook, sae, features_to_ablate):
    encoded = sae.encode(activation)
    for feat_idx in features_to_ablate:
        encoded[:, :, feat_idx] = 0.0
    reconstructed = sae.decode(encoded)
    return reconstructed  # Every feature now has reconstruction error
```

### Recommended: Residual-based patching at last token only

Two principles:
1. **Last-token only** — during autoregressive generation, only the final position determines the next token. Patching all positions corrupts the KV cache for prior tokens.
2. **Residual subtraction** — instead of full reconstruction, compute just the target features' contribution and subtract it. No reconstruction error on other features.

```python
def feature_ablation_hook(activation, hook, sae, features_to_ablate):
    """Remove specific SAE features' contribution at the last token only."""
    last_pos = activation[:, -1:, :]  # [batch, 1, d_model]
    encoded = sae.encode(last_pos)

    # Build a sparse vector with only the features to remove
    mask = torch.zeros_like(encoded)
    for feat_idx in features_to_ablate:
        mask[:, :, feat_idx] = encoded[:, :, feat_idx]

    # Decode only those features to get their contribution
    # Subtract bias since we want just the feature directions, not the bias term
    contribution = sae.decode(mask) - sae.b_dec

    # Surgically remove only those features' contribution
    activation[:, -1:, :] -= contribution
    return activation


def feature_activation_hook(activation, hook, sae, features_to_set):
    """Set specific SAE features to target values at the last token only."""
    last_pos = activation[:, -1:, :]
    encoded = sae.encode(last_pos)

    # Compute current contribution of target features
    current_mask = torch.zeros_like(encoded)
    target_mask = torch.zeros_like(encoded)
    for feat_idx, value in features_to_set.items():
        current_mask[:, :, feat_idx] = encoded[:, :, feat_idx]
        target_mask[:, :, feat_idx] = value

    # Remove current contribution, add target contribution
    current_contribution = sae.decode(current_mask) - sae.b_dec
    target_contribution = sae.decode(target_mask) - sae.b_dec

    activation[:, -1:, :] += (target_contribution - current_contribution)
    return activation
```

### Why this matters

| Approach | Reconstruction error | Positions affected | Surgical |
|----------|--------------------|--------------------|----------|
| Full encode→decode | All features | All positions | No |
| Last-token encode→decode | All features | Last position only | Partial |
| Residual at last token | Zero (target features only) | Last position only | Yes |

The residual approach is equivalent to: "take the activation as-is, but adjust it as if feature X had a different value." Everything else is untouched.
