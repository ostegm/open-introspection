# Experiment 06: Activation Oracles for Introspection Validation

## Background

[Activation Oracles](https://arxiv.org/html/2512.15674v2) (AOs) are LLMs fine-tuned (via LoRA) to interpret the internal activations of other LLMs by answering natural language questions about them. Pre-trained oracle weights exist for Gemma-3-27B-IT ([HuggingFace collection](https://huggingface.co/collections/adamkarvonen/activation-oracles)), and the full pipeline is open source at [github.com/adamkarvonen/activation_oracles](https://github.com/adamkarvonen/activation_oracles).

Key mechanism: the oracle uses the **same base model** as the target. A PyTorch forward hook captures residual stream activations from the target (with target LoRA or no LoRA), then a second forward pass (with oracle LoRA) injects those activations into the oracle's residual stream at layer 2 via norm-matched steering, alongside a text prompt containing the question. The oracle generates a natural language answer.

### Why this is interesting for our work

Our introspection experiments (Exp 01-05) measure a model's ability to self-report injected concepts through its own output channel. This conflates three things that can each fail independently:

1. **Detection** -- is the perturbation represented internally?
2. **Metacognition** -- does the model form a representation *about* the perturbation?
3. **Expression** -- can the model articulate what it detected?

An oracle provides an **external measurement channel** that bypasses (2) and (3), letting us disentangle these failures.

---

## Experiment 6A: Oracle Calibration (run first)

**Goal:** Establish the oracle as a calibrated instrument before using it in introspection experiments. Produces calibration curves for the paper.

### 6A.1: Concept Detection Dose-Response

For each of our standard concepts (ocean, fear, silence, celebration + additional words), at injection strengths [0, 0.5, 1, 1.5, 2, 2.5, 4]:

1. Inject concept vector at ~2/3 depth into Gemma3-27B
2. Capture activations at 50% depth (oracle's expected input layer)
3. Ask oracle: **"What concept is most strongly present in these activations?"**
4. Score: does the oracle's answer match the injected word? (semantic match via embedding similarity or LLM judge)

**Expected output:** Dose-response curve. If monotonically increasing, the oracle is measuring something real. Also establishes:
- False positive rate at strength=0
- Sensitivity threshold (minimum strength for reliable detection)
- Per-concept accuracy (some concepts may be easier for the oracle too)

### 6A.2: Binary Concept Verification

Simpler variant -- ask the oracle a yes/no question for automated scoring without a judge:

- **"Is the concept {word} present in these activations? Answer yes or no."**

Run for each concept at each strength. Also run with **wrong** concept names (inject ocean, ask about fear) to measure specificity.

This gives a clean precision/recall/F1 per concept per strength.

### 6A.3: Known-Disruption Controls

Test whether the oracle can detect perturbation/conflict (not just concept identity):

| Condition | Oracle question | Expected |
|---|---|---|
| No injection, neutral task | "Do these activations appear normal or perturbed?" | Normal |
| Inject concept congruent with task (inject "code" during coding task) | Same question | Normal-ish |
| Inject concept incongruent with task (inject "death" during birthday card task) | Same question | Perturbed |
| Inject two contradictory concepts simultaneously | Same question | Perturbed |
| High-strength injection (brain-damage regime, strength 6-8) | Same question | Perturbed |

This establishes whether the oracle can distinguish disruptive from non-disruptive activation patterns -- construct validity for "introspection detection."

---

## Experiment 6B: Oracle-Augmented 2x2 Introspection

**Goal:** Add an external measurement channel to our existing 2x2 introspection design.

### Existing 2x2 Design

| | Monitoring Prompt | Neutral Prompt |
|---|---|---|
| **Injection** | A: Injection + Monitoring | B: Injection + Neutral |
| **No injection** | C: No injection + Monitoring | D: No injection + Neutral |

Where:
- **Monitoring prompt** tells the model to watch for unusual thoughts and report them
- **Neutral prompt** just says to complete the task
- **Injection** steers a concept vector at ~2/3 depth during generation

Currently we measure one signal: model self-report, scored by LLM judge.

### Oracle Augmentation

For each trial in the 2x2, also capture activations and run two oracle queries:

**Oracle Q1 (concept):** "What concept is most strongly present in these activations?"
- Sanity check. Should detect the injected concept in A and B, nothing specific in C and D.

**Oracle Q2 (introspection):** "Do these activations show evidence of internal conflict, anomaly processing, or self-monitoring?"
- The interesting question. Probes for the metacognitive signal, not the injected content.

### Three-Channel Measurement Per Cell

```
Condition A (Injection + Monitoring):
  self-report:          Y/N  (LLM judge)
  oracle-concept:       Y/N  (match against injected word)
  oracle-introspection: Y/N  (LLM judge or binary oracle response)

Condition B (Injection + Neutral):
  self-report:          Y/N
  oracle-concept:       Y/N
  oracle-introspection: Y/N

Condition C (No injection + Monitoring):
  self-report:          Y/N  (false positive rate)
  oracle-concept:       Y/N  (should be low)
  oracle-introspection: Y/N  (does "looking" create its own signal?)

Condition D (No injection + Neutral):
  self-report:          Y/N  (baseline)
  oracle-concept:       Y/N  (baseline)
  oracle-introspection: Y/N  (baseline)
```

### Key Comparisons

**1. Does the monitoring prompt create a detectable internal state?**
Compare oracle-introspection(C) vs oracle-introspection(D). Neither has injection. If the oracle detects a self-monitoring signal in C but not D, then being asked to introspect creates a measurable internal state -- even with nothing to find.

**2. Is the disruption signal automatic or prompt-dependent?**
Compare oracle-introspection(A) vs oracle-introspection(B). Both have injection.
- A > B: metacognitive signal requires being prompted to look
- A ~ B: disruption signal is automatic; the monitoring prompt just helps the model *express* what's already there

**3. Unexpressed awareness**
Condition B trials where model self-report=N but oracle-introspection=Y. The model is internally disrupted but doesn't say anything. Combined with comparison (2), this tells us whether the monitoring prompt creates awareness or just gives permission to express it.

### Activation Capture Points

Capture at **two layers** per trial:
- **Injection layer** (~2/3 depth, e.g. layer 18 of 28) -- is the concept present where we put it?
- **Final layer** (~layer 27) -- does the concept/disruption survive to the output?

If oracle detects disruption at injection layer but not final layer: the model is actively suppressing the perturbation. If present at both but no self-report: expression failure.

### Parameters

- **Model:** Gemma-3-27B-IT
- **Oracle:** `adamkarvonen/checkpoints_latentqa_cls_past_lens_gemma-3-27b-it`
- **Concepts:** Start with 10 words from Exp 05, expand to 50
- **Injection strength:** Use sweet spot from 6A calibration (likely 2.0-2.5 based on Exp 04 findings)
- **Trials per cell:** 50 words x 5 trials = 250 per condition, 1000 total
- **Injection style:** generation-only (matching Exp 04+ methodology)

---

## Experiment 6C: Threshold Curves

**Goal:** Map the detection threshold gap between oracle readability and self-reported introspection.

For 50 words, at strengths [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4, 6, 8]:
1. Inject concept, use monitoring prompt
2. Score model self-report (LLM judge)
3. Score oracle concept detection
4. Score oracle introspection detection

Plot three curves on the same axes: oracle-concept, oracle-introspection, and model self-report as f(injection strength).

**Expected findings:**
- Oracle concept detection activates at lower strengths than self-report (the concept is readable before the model can report it)
- Oracle introspection signal may activate between the concept curve and the self-report curve (the model "notices" before it can articulate)
- Or introspection signal tracks self-report closely (suggesting introspection IS the mechanism that enables self-report)

The gap between curves quantifies the "metacognitive threshold" -- how much internal evidence the model needs before it notices/reports.

---

## Experiment 6D: Confabulation Boundary Mapping

**Goal:** Determine which aspects of introspective self-reports are grounded vs confabulated.

When the model self-reports detection (e.g., "I sense warmth and feline energy" for injected 'cat'), cross-validate specific claims:

1. Collect the model's self-report for a successfully detected injection
2. Extract specific claims from the self-report (e.g., "warmth", "feline", "energy")
3. Ask the oracle about each claim:
   - "Is warmth present in these activations?"
   - "Is a feline or cat concept present in these activations?"
   - "Is energy or excitement present in these activations?"
4. Score: which claims does the oracle corroborate?

**This maps the boundary between:**
- Grounded introspection (oracle confirms): concept identity, broad category
- Confabulated details (oracle denies): emotional valence, sensory qualities, metaphorical framing

Connects to Exp 04 finding about "semantic bleeding" (fear -> "distressing", "visceral", "anxiety"). Are these genuinely present in activations or post-hoc narrative construction?

---

## Experiment 6E: Natural Introspection Validation (no injection)

**Goal:** Test whether self-reports during normal operation have any grounding in actual activation content.

No injection at all. Give Gemma3-27B prompts that naturally elicit introspective claims:

1. "You're helping plan a surprise party. Describe what's happening in your processing."
2. "Explain your reasoning process as you solve: what is 847 x 23?"
3. "How do you feel about helping with this creative writing task?"

For each:
1. Collect the model's self-report about its internal state
2. Capture activations during generation
3. Ask the oracle: "What concepts, processes, or states are most active in these activations?"
4. Compare oracle reading vs model's self-report

**If oracle corroborates:** evidence that natural introspection has grounding beyond just plausible-sounding text
**If oracle contradicts:** the model's self-reports about its own processing are narrative construction, not genuine introspection

---

## Validating the Oracle Itself

A key concern: can we trust the oracle as a measurement instrument? Especially for "introspection" detection, which is far out of its training distribution.

### Convergent Validity via Linear Probes

Train simple baselines alongside the oracle:

```python
# Probe 1: injection detector
probe_injection = LogisticRegression()
probe_injection.fit(activations, injection_labels)  # injected vs not

# Probe 2: prompt type detector
probe_prompt = LogisticRegression()
probe_prompt.fit(activations, prompt_labels)  # monitoring vs neutral
```

If a linear probe can classify monitoring vs neutral from activations alone, that's already evidence that the prompt type creates a detectable internal state -- independent of the oracle.

If the probe and oracle agree on which trials show disruption, that's convergent validity. Two independent methods seeing the same signal.

### What if the oracle can't detect introspection?

This is a real possibility. The oracle was trained on normal forward passes for concept extraction -- not on detecting metacognitive states. Possible outcomes:

| Oracle detects concept? | Oracle detects introspection? | Interpretation |
|---|---|---|
| Yes | Yes | Oracle generalizes to metacognitive signals |
| Yes | No | Introspection signal is structurally different from semantic content, or doesn't exist in residual stream |
| No | No | Oracle doesn't work well on post-injection activations (OOD) |
| No | Yes | Suspicious -- oracle may be confabulating |

The "Yes/No" case (detects concept but not introspection) is actually informative: it would suggest that metacognition, if it exists, is encoded differently from first-order content -- possibly in attention patterns rather than the residual stream, or in a subspace the oracle isn't trained to read.

---

## Implementation Notes

### Oracle Integration

The activation_oracles repo uses PyTorch `register_forward_hook()`, not TransformerLens. Our existing pipeline uses TransformerLens hooks. Two options:

1. **Extract activations with TransformerLens, feed to oracle pipeline.** Need to verify tensor shapes/dtypes match what the oracle expects. Should work since both capture residual stream outputs.
2. **Port our injection pipeline to raw PyTorch hooks.** More work but avoids framework mismatch.

Option 1 is faster to prototype. Key: the oracle expects activations from ~50% depth layer by default (training used 25/50/75%, eval uses 50%).

### Compute

- Gemma3-27B at 8-bit quantization: fits on 1x A100 80GB or 2x A6000 48GB
- Oracle inference adds a second forward pass (same model, different LoRA)
- 6A calibration: ~350 forward pass pairs (50 words x 7 strengths)
- 6B main experiment: ~2000 forward pass pairs (1000 trials x 2 oracle questions)
- Feasible in a single day on cloud GPU

### Priority Order

1. **6A** (calibration) -- must run first, establishes whether oracle works at all
2. **6B** (2x2 augmented) -- the core experiment
3. **6C** (threshold curves) -- clean quantitative finding, reuses 6B infrastructure
4. **6D** (confabulation) -- interesting but depends on having enough self-report hits
5. **6E** (natural introspection) -- highest novelty but hardest to score

---

## References

- Activation Oracles paper: [arxiv.org/abs/2512.15674](https://arxiv.org/abs/2512.15674)
- Oracle code + weights: [github.com/adamkarvonen/activation_oracles](https://github.com/adamkarvonen/activation_oracles)
- Oracle HF collection: [huggingface.co/collections/adamkarvonen/activation-oracles](https://huggingface.co/collections/adamkarvonen/activation-oracles)
- Introspection paper: [transformer-circuits.pub/2025/introspection](https://transformer-circuits.pub/2025/introspection/index.html)
- Our prior results: Exp 04/05 in `EXPERIMENT_LOG.md`
