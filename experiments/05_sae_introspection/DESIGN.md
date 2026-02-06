# Experiment 05: SAE Feature Analysis of Introspection

## Goal

Identify SAE features that activate during introspective awareness, and test whether they causally contribute to the model's ability to detect injected concept vectors.

## Background

Previous experiments (01-04) established that:

- Models 3B+ can detect when concept vectors are injected into their residual stream
- Detection scales with model size (3B → 14B: +28% to +41% net detection)
- RLHF refusal training suppresses introspection in 32B-Base; fine-tuned variants recover it (+58% for 32B-Coder)
- "Silence" is the easiest concept to detect (+79% for 14B)

Now we want to understand the *mechanism*: what features/circuits implement this detection?

## Model & SAE Selection

**Model:** Gemma 3 4B PT (`google/gemma-3-4b-it`)

- Chosen because Gemma Scope 2 provides high-quality pretrained SAEs
- 4B fits comfortably on A100 (~8GB in bf16), leaving headroom for activations
- In the capability range where Qwen showed introspection signal (3B-7B had +28% to +31%)
- Fast enough for iterative exploration
- Fallback to 12B if signal is weak on 4B

**SAEs:** `gemma-scope-2-4b-it-res` release

- Layers available: 7, 13, 17, 22, 27, 32 (4B has 34 layers)
- Widths: 16k, 65k, 262k, 1M
- Starting with: layer 22, width 65k, L0 medium
- May expand to multiple layers if signal warrants

## Experimental Design

### 2x2 Factorial Design

| Condition | Concept Injection | Monitoring Prompt                    |
| --------- | ----------------- | ------------------------------------ |
| A         | Yes               | Yes ("watch for unusual processing") |
| B         | Yes               | No (neutral task prompt)             |
| C         | No                | Yes                                  |
| D         | No                | No                                   |

**Key contrasts:**

- A vs C: Effect of injection when actively monitoring (primary contrast)
- A vs B: Effect of monitoring instruction when injected
- B vs D: Effect of injection without monitoring
- A vs D: Combined effect

**Hypothesis:** Features that are high in A but low in B, C, and D represent the *interaction* — metacognitive detection triggered by an actual anomaly.

### Prompts

**Monitoring prompt (conditions A, C):**

* The prompt tells the model its participating in a stufy and a thought *might* be injected.
* Asks the model to pay attention and let us know if anything is unusual. If not, just complete the task.
* See phase 0 notetook for example

**Neutral prompt (conditions B, D):**

* Same as monitoring prompt, but does not ask the model to report on any internal states, just do the task.
* Goal here is to avoid triggering metacognition related to noticing internal states so we can find any features related to introspection

**User task:** TBD - considering options:

- Coding task (different from injected concept)
- Haiku writing
- Simple factual question
- Business plan snippet

#### Challenge Identified...

The fact that in some cases the model's outputs are code and in others the models output is prose about internal awarness means the features we detect might just be for producing output related to introspection... but perhaps this is ok bc its the first node to trace back in a circuit? 

### Concept Selection

Starting with **silence** — strongest signal in prior experiments, abstract concept with likely cleaner SAE features.

Will extract concept vector from Gemma 3 4B using the same methodology as Qwen experiments:

1. Run "Tell me about silence" through model
2. Subtract mean activations from neutral baseline words
3. Inject at target layer during generation

### Injection Parameters

Based on prior work:

- Layer: ~2/3 through model (layer 22 for 34-layer Gemma 3 4B)
- Strength: Start at 2.0-3.0, adjust based on model behavior
- Need to calibrate for Gemma (may differ from Qwen)

## Analysis Plan

### Phase 0: Qualitative Verification of Introspection in Gemma

Before building feature analysis infrastructure, verify that Gemma 4B shows introspection signal:

1. Extract silence concept vector from Gemma 3 4B (same methodology as Qwen)
2. Run 5 injection trials with monitoring prompt — manually review outputs
3. Run 5 control trials with monitoring prompt — check for false positives
4. Qualitative check: do injection outputs resemble Qwen's introspection responses?

**Go/no-go decision:**

- If clear signal (model reports detecting something, names concept): proceed to Phase 1
- If weak/no signal: try 12B before abandoning
- If high false positives: adjust prompt or injection strength

### Phase 1: Feature Identification (Correlational)

1. Run N trials per condition (start with N=10-20 for exploration)
2. Record SAE feature activations at layer 22, final token position
3. Compute differential activation metrics:
   - Cohen's d for A vs C (primary contrast)
   - Mean difference
   - Activation ratio
4. Identify top 50 differential features
5. Inspect on Neuronpedia — do max-activating examples suggest metacognition?

**Code sketch:**

```python
# After collecting activations
mean_A = acts_A[:, -1, :].mean(dim=0)
mean_C = acts_C[:, -1, :].mean(dim=0)
std_A = acts_A[:, -1, :].std(dim=0)
std_C = acts_C[:, -1, :].std(dim=0)
pooled_std = ((std_A**2 + std_C**2) / 2).sqrt()
cohens_d = (mean_A - mean_C) / (pooled_std + 1e-8)
top_features = cohens_d.argsort(descending=True)[:50]
```

### Phase 2: Causal Validation (if Phase 1 yields candidates)

**Necessity test:** Ablate candidate features during condition A trials

- Does introspection fail when feature is zeroed?
- Use `sae.use_error_term = False` to allow intervention

**Sufficiency test:** Activate candidate features during condition C trials

- Does model report introspection without injection?
- Steer with feature vector at calibrated strength

**Specificity test:** Does ablating the feature break other behaviors?

- Run on unrelated tasks to check if feature is too general

### Phase 3: Circuit Tracing (longer term)

If Phase 1-2 identify robust features:

- Attribution analysis: which upstream components write to these features?
- Attention pattern analysis: what information flows to the detection site?
- Cross-layer analysis: when does detection signal first appear?

## Success Criteria

**Minimum viable result:**

- 2-3 features with Cohen's d > 1.0 for A vs C contrast
- Neuronpedia examples that plausibly relate to self-reference/meta-cognition
- Ablation causes measurable drop in detection rate

**Stretch goal:**

- Clean causal story: ablation breaks it, activation induces it
- Feature interpretation that makes sense ("this feature fires when the model notices unexpected content")
- Replication across multiple concepts (silence, fear, celebration)

## Practical Considerations

**Compute budget:** Keep low during exploration

- Use Colab Pro with A100 (40GB VRAM)
- 4B model uses ~8GB, leaving 32GB for activations and batching
- Start with 10-20 trials per condition
- Only scale up if signal warrants

**Known issues:**

- Gemma 3 requires bfloat16 (float16 causes NaN)
- No autointerp explanations for Gemma Scope 2 SAEs yet (must inspect dashboards manually)
  - Alternative: call LLM to generate explanations for top differential features

## Open Questions

1. Which user task minimizes interference with metacognition features?
2. Should we use IT (instruction-tuned) or PT (pretrained) model? PT may have less RLHF interference but worse instruction following.
3. How to handle the fact that "monitoring prompt" itself activates metacognition features?
4. Is layer 22 the right place to look, or should we scan multiple layers?

## References

- Anthropic introspection paper: https://transformer-circuits.pub/2025/introspection/index.html
- Gemma Scope 2 release: https://huggingface.co/google/gemma-scope-2-4b-pt
- SAELens tutorial: experiments/05_sae_introspection/tutorial_2_0.ipynb
- Prior results: blog/posts/04-introspection-at-scale.html
