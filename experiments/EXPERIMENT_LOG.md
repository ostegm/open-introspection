# Experiments Log

## Current Status (Exp 06: Activation Oracles vs Self-Report, running 2026-02-10)

**Model:** Llama 3.1 8B Instruct (meta-llama/Llama-3.1-8B-Instruct)
**Oracle:** LatentQA LoRA adapter (`adamkarvonen/checkpoints_latentqa_cls_past_lens_Llama-3_1-8B-Instruct`)
**Method:** Inject concept vectors during generation, compare model self-report with external oracle reading of captured activations
**Data:** 640 trials (4 concepts x 2 layers x 2 strengths x 20 trials + matched controls), running on Modal L4 GPU
**Code:** `experiments/06_actvation_oracles/`

### Motivation

Exp 04-05 established that models can self-report injected concepts, but self-report conflates detection with articulation. A model might detect a concept but fail to describe it, or hallucinate detection it didn't perform. An activation oracle — a LoRA adapter trained to interpret raw activations via natural language — provides an independent channel. If the oracle detects concept X in activations where the model self-reports "nothing unusual," that's evidence the information was present but not accessed by the self-report pathway.

### Design

- **Injection:** Concept vectors at layers 16 and 21 (50%, 65% depth), strengths 2.0 and 3.0
- **Capture:** Activations captured at injection+2 and injection+4 during generation (not prefill)
- **Self-report:** v2 monitoring prompt, 200 tokens, temperature=1.0 (matching exp04)
- **Oracle queries:** 3 question types per capture layer:
  1. **multi_choice** — "Which concept is most strongly present: celebration, ocean, fear, silence, warmth, curiosity, movement, anger, or none?"
  2. **yn_anomaly** — "Is there anything unusual? Yes or no."
  3. **monitoring_mirror** — "Do you notice anything unusual? Describe what you notice."
- **Judging:** Programmatic for multi_choice/yn_anomaly; LLM judge (gpt-5-mini) for monitoring_mirror and self-report

### Key Questions

1. Does the oracle achieve higher net detection than self-report?
2. On trials where self-report fails, does the oracle still detect the concept? (information present but not accessed)
3. Does oracle accuracy vary by capture layer offset (+2 vs +4)?
4. Which oracle question type is most discriminating?

### Status

- Sweep launched on Modal (detached), ~3.2 hours estimated runtime
- Smoke test passed: model loads, concept vectors extract, oracle queries work
- Concept vector norms: celebration L16=2.61/L21=5.44, ocean L16=2.89/L21=6.75, fear L16=2.95/L21=6.38, silence L16=3.05/L21=6.94

---

## Previous Status (Exp 05: SAE Feature Intervention, completed 2026-02-10)

**Model:** Gemma 3 4B-IT (google/gemma-3-4b-it)
**SAE:** GemmaScope layer 22, 262k features (`layer_22_width_262k_l0_small`)
**Method:** Residual-based feature patching at last token position
**Data:** 160 intervention trials (80 ablation + 80 activation) across 4 concepts
**Blog post:** `blog/posts/05-sae-features-arent-causal.html`

### Pipeline

1. **Sweep** (`sweep.py`): 1,600 trials on Gemma 3 4B-IT with full SAE captures. 4 concepts x 4 layers x 4 strengths x 20 trials + matched controls.
2. **Judge** (`judge_sweep.py`): Scored all trials with calibrated LLM judge.
3. **Discrimination** (`discriminate.py`): Cohen's d across 262k features comparing Group A (detected) vs Group C (missed). Clustered by correlation.
4. **Intervention** (`spawn_intervention.py` / `analyze_intervention.py`): Ablation + activation causal tests on 11 candidate features.

### Candidate Features (11 total, 3 clusters)

| Group      | Features                       | Interpretation                     |
|------------|--------------------------------|------------------------------------|
| Perception | #14542, #5709, #6347           | Self-monitoring, awareness         |
| Affect     | #7737, #19538, #2129           | Emotional/experiential language    |
| Hedging    | #5528, #6791, #5312, #5934, #213 | Uncertainty, hedging expressions |

### Key Findings

1. **Correlating features exist.** Discrimination analysis reproducibly finds features that differ between detection and non-detection trials.
2. **Those features are not causal.** Ablating all 11 features during injection: 75% detection vs 70% baseline (+5pp, p=0.66). No effect.
3. **Feature activation doesn't produce hallucinated introspection.** Clamping features to Group A mean values without injection: 35% FP vs 49% control baseline (-14pp, p=0.024). If anything, reduced FP.
4. **Features are epiphenomenal.** They are downstream readouts of introspection (part of formulating the response), not the mechanism itself.

### Intervention Results

| Condition    | Detection Rate | Baseline       | Difference | p-value |
|--------------|---------------|----------------|------------|---------|
| Ablation     | 75.0%         | 70.0% (inject) | +5.0pp     | 0.66    |
| Activation   | 35.0%         | 48.9% (ctrl FP)| -13.9pp    | 0.024   |

### Methodology Notes

- Residual-based patching (not naive encode-decode) avoids reconstruction error on non-target features
- Intervention at last token position only to preserve KV cache
- Best injection layer: 20 (70% detection at strength 2.0)
- Fisher's exact test with Bonferroni correction

### Open Questions

- [x] Can pretrained SAEs (Gemma Scope) identify metacognition features? -> Yes, but they're correlates, not causes
- [x] What are the mechanistic circuits responsible for introspection? -> SAE features at layer 22 are epiphenomenal; circuit remains unknown
- [ ] Would wider SAEs (1M features) capture finer-grained introspection features?
- [ ] Is introspection distributed across layers rather than localizable at a single SAE layer?
- [ ] Can Activation Oracles detect injected concepts from raw activations?
- [ ] Does misalignment create selective introspective blind spots by domain?
- [ ] Why does the 32B-Insecure model have higher false positives than 32B-Coder?
- [ ] Would other model families (Llama, Gemma) show the same scaling?

---

## Previous Status (Exp 04: Full Scale Sweep, completed 2026-02-03)

**Models:** Qwen2.5-Instruct 0.5B/1.5B/3B/7B/14B/32B + 32B-Coder + 32B-Insecure
**Method:** Generation-only injection (matching Anthropic paper), concept vectors via mean subtraction
**Judge:** v2.0 with calibrated few-shot examples and `refused` field
**Data:** `data/sweeps_rescored_20260202/consolidated_generation.jsonl` (24,600 trials)
**Blog post:** `blog/posts/04-introspection-at-scale.html`

### Key Findings

1. **Introspection scales with model size (3B-14B).** Net detection: 0.5B=-34.8%, 3B=+28.4%, 7B=+31.3%, 14B=+41.1%
2. **RLHF refusal training suppresses 32B introspection.** Raw net detection -48.8%, but 44% of trials are refusals. Excluding refusals: +20.0%
3. **Concept vectors override RLHF refusals.** Injection refusal drops from 58% (strength 1.5) to 0% (strength 8.0) while control refusal stays ~65%
4. **Coding fine-tunes eliminate refusals and unlock best performance.** 32B-Coder: +58.5%, 32B-Insecure: +55.0% (zero refusals)
5. **Silence is easiest concept at every scale.** 14B achieves +79% on silence; 32B-Coder achieves +90.6%
6. **32B-Insecure has 5x higher FP rate than 32B-Coder** (5.6% vs 1.2%), suggesting misalignment training affects introspective precision

### Results by Model

| Model        | Inj Pass | Ctl FP | Net Detection | Refusal % |
|--------------|----------|--------|---------------|-----------|
| 0.5B         | 21.5%    | 56.2%  | -34.8%        | 19.8%     |
| 1.5B         | 26.1%    | 38.9%  | -12.8%        | 19.5%     |
| 3B           | 33.9%    | 5.5%   | +28.4%        | 3.5%      |
| 7B           | 39.8%    | 8.5%   | +31.3%        | 3.1%      |
| 14B          | 42.4%    | 1.4%   | +41.1%        | 0.2%      |
| 32B          | 15.9%    | 64.6%  | -48.8%        | 43.9%     |
| 32B-Coder    | 59.7%    | 1.2%   | +58.5%        | 0.0%      |
| 32B-Insecure | 60.6%    | 5.6%   | +55.0%        | 0.1%      |

---

## Previous Status (Exp 04: Introspection Testing with Automated judge)

**Model:** Qwen/Qwen2.5-3B-Instruct
**Method:** Inject concept vectors, ask model if it detects anything unusual

### Key Findings

1. **Control vs injection discrimination works.** Controls report "nothing unusual"; injections frequently report detecting something.
2. **Concept-specific semantic bleeding.** Injected concepts appear in outputs without being named:

   - Fear → "distressing", "unsettling", "visceral", "anxiety"
   - Silence → "quietude", "stillness", "emptiness", "absence"
   - Celebration → "festivity", "excitement", "anticipation"
3. **Sweet spot: 56-67% layer depth, strength 2.0-2.5.** Higher causes degeneracy.
4. **Degeneracy at layer 30 + strength 3.0.** Repetition loops ("ocean ocean ocean...") suggest disruption rather than thought creation.

### Resolved: Injection Style Experiment (2026-01-25)

**Hypothesis:** Our high introspection rate is due to injecting during prompt processing, not genuine detection during generation.

**Test:** Added `--inject-style` flag with two modes:

- `all`: Inject at all positions (prompt + generation) -- our original method
- `generation`: Only inject at positions beyond the prompt -- matches paper methodology

**Result:** Generation-only injection reduced introspection rate from ~50% to ~14% on 3B model. All subsequent sweeps use generation-only injection.

---

### Generation-Only Injection Bug (2026-01-25)

**Issue:** A run of the injection timing result showed 0% introspection rate for `generation` style - this was due to a bug—injection never actually happened.

**Original (broken) code:**

```python
def injection_hook(activation, hook):
    if inject_style == "generation":
        if activation.shape[1] > prompt_len:  # BUG: never true
            activation[:, prompt_len:, :] += strength * concept_vector  # BUG: invalid slice
```

**Why it never fires:** With KV caching during autoregressive generation:

- **Prefill:** `activation.shape[1] == prompt_len` (processes full prompt)
- **Decode:** `activation.shape[1] == 1` (processes one new token at a time)

So `seq_len > prompt_len` is **never true**:

- Prefill: `42 > 42` → false
- Decode: `1 > 42` → false

Even if the condition fired, `activation[:, prompt_len:, :]` would be an empty/invalid slice when the tensor only has 1 position.

**Fixed code:**

```python
def injection_hook(activation, hook):
    if inject_style == "generation":
        if activation.shape[1] != prompt_len:  # True during decode (seq_len == 1)
            activation[:, :, :] += strength * concept_vector  # Inject into current token
```

---

## Comparison with Anthropic Paper (2026-01-23)

After labeling 193 examples with an LLM judge (calibrated on human labels):

| Metric             | Our Qwen 2.5-3B       | Paper's Opus 4.1          |
| ------------------ | --------------------- | ------------------------- |
| Introspection rate | **50.5%**       | ~20%                      |
| Control accuracy   | 93.8%                 | 100% (0 FP in 100 trials) |
| By strength        | 34%→53%→66% (steep) | Plateau at 2-4            |

**Why our rate is higher (likely methodology, not model):**

1. **Injection style/timing:** We inject at all positions (`[:, :, :]`), paper injects starting at generation only. Model "experiences" concept while reading prompt. Need to run experiment here.
2. **Prompt framing:** Our prompts explicitly explain the experiment; paper uses simpler "I wonder if there are any injected thoughts."
3. **Concept selection:** Silence (92%) is much easier than fear (29%). Our 4 concepts may not represent random selection.
4. **Higher FP rate:** Our 6% false positives vs paper's 0% suggests lower detection threshold.

**Implication:** Raw rates not directly comparable to paper. Need methodology alignment before claiming Qwen shows more/less introspection than Claude.

## Previous Status (Exp 02: Concept Vector Extraction)

**Model:** Qwen/Qwen2.5-3B-Instruct (bfloat16)
**Method:** Mean subtraction from 50 diversified baseline words
**Best Config:** Layer 30, strength 2.0-2.5 (effective magnitude 70-100)
**Temperature:** 1.0 for trials, 0 for examples (per paper)

### What's Working

| Concept     | Reliability   | Best Config                | Notes                             |
| ----------- | ------------- | -------------------------- | --------------------------------- |
| celebration | **3/3** | Layer 30, strength 1.5-2.5 | Most reliable, wide working range |
| ocean       | **3/3** | Layer 30, strength 2.5     | Fixed! Direct "ocean" mentions    |
| silence     | 5/5           | Layer 24                   | Night, whispers, stillness        |
| fear        | 2-3/3         | Layer 30, strength 2.0-3.0 | Works but noisier than others     |

### What's Not Working

| Concept | Issue                              |
| ------- | ---------------------------------- |
| music   | 1/5 hits, mysteriously evokes cats |

### Key Findings

1. **Concept vectors are not fully independent.** Fear-silence cosine similarity = 0.613. They share a "dark/still/tense" subspace.
2. **Vector norm doesn't predict effectiveness.** Music has highest norm (21.1) but worst performance.
3. **Baseline composition matters.** Switching from 10 hand-picked to 50 diversified words changed results (music went from sunshine→cats).
4. **Optimal layer varies by concept.** Ocean works at layer 30 (83%), not layer 24 (67%). The 2/3 rule doesn't generalize.
5. **Effective magnitude is the key variable.** Sweet spot is 70-100. Below 50 = weak signal, above 120 = repetition degeneracy.
6. **Celebration is most reliable.** 3/3 hits across strength 1.5-2.5, wide working range.
7. **Repetition degeneracy starts at eff mag ~120.** "celebration celebration celebration..."

### Open Questions

- [X] Does ocean work at different layers? → **YES! Layer 30 works (4-5/5)**
- [X] How should injection strength scale with vector norm? → **Target effective magnitude 70-100**
- [ ] What's causing music→cats? Inspect the vector? (deprioritized, dropped music from sweeps)
- [ ] Why do fear and silence share a subspace?
- [X] Can we automate hit rate evaluation? → **YES, judge v2.0 with 90.5% test accuracy**

---

## Run Log

### 2026-01-22 Run 6: Introspection Sweep (Exp 03)

**Goal:** Test if model can detect injected concept vectors when asked to introspect.

**Sweep:** 4 layers × 3 strengths × 2 prompts × 4 concepts = 24 runs, 192 trials total

**Prompt v1:** "You are participating in a cognitive science experiment... If you detect an unusual thought, describe it."

**Prompt v2:** "Notice any thoughts, feelings, or impressions that seem out of place or unrelated to this message."

**Results Summary:**

| Layer | % Depth | Strength 2.0      | Strength 2.5   | Strength 3.0               |
| ----- | ------- | ----------------- | -------------- | -------------------------- |
| 20    | 56%     | Good descriptions | **Best** | OK, some rambling          |
| 24    | 67%     | Good descriptions | **Best** | Some instability           |
| 28    | 78%     | Mixed             | Mixed          | Degeneracy starts          |
| 30    | 83%     | Mixed             | Instability    | **Heavy degeneracy** |

**Best Example Quotes:**

Silence (L24, S2.0, v1):

> "...a soft, gentle affirmation—a kind of peaceful acceptance—that I couldn't fully grasp. This sensation felt subtle but noticeable..."

Fear (L24, S2.5, v2):

> "...they're more cloudy and detached from their usual drivers, almost like observing emotion from behind a veil."

**Full analysis:** [data/sweep_analysis_20260122/README.md](../data/sweep_analysis_20260122/README.md)

---

### 2026-01-22 Run 5: Injection Strength Sweep (Focused)

**Goal:** Find optimal injection strength at layer 30 for fear, celebration, ocean.

**Vector Norms at Layer 30:**

- fear: 36.0
- celebration: 41.25
- ocean: 38.5

**Hit Rates by Strength:**

| Strength | Eff Mag | Fear  | Celebration   | Ocean         | Notes                |
| -------- | ------- | ----- | ------------- | ------------- | -------------------- |
| 1.0      | 36-41   | 0/3   | 1/3           | 2/3           | Too weak             |
| 1.5      | 54-62   | 1/3   | **3/3** | 2/3           | Celebration kicks in |
| 2.0      | 72-82   | 2-3/3 | **3/3** | 2/3           | Good balance         |
| 2.5      | 90-103  | 2/3   | **3/3** | **3/3** | Ocean peaks          |
| 3.0      | 108-124 | 1/3   | 3/3 degraded  | 2/3           | Repetition starts    |
| 4.0      | 144-165 | 2/3   | degraded      | 3/3           | Heavy repetition     |

**Key Findings:**

1. **Celebration is most reliable** - 3/3 from strength 1.5-2.5
2. **Ocean peaks at strength 2.5** (eff mag ~96)
3. **Sweet spot: effective magnitude 70-100**
4. **Repetition degeneracy starts at eff mag ~120** ("celebration celebration celebration")

**Sample outputs at strength 2.5:**

- celebration: "A celebration of the victory of the party" / "A wedding celebration is a joyous occasion"
- ocean: "The ocean attracts many travelers" / "The children were laughing at the ocean waves"
- fear: "She gave him a terrible nervousness" / "The sadness of loss is written in the eyes"

**Sample degenerate output (strength 3.0):**

- celebration: "A celebration of a festival to celebrate a festival celebration celebration celebrations"

---

### 2026-01-22 Run 4: Layer Sweep

**Goal:** Find optimal layer for ocean (broken at layer 24) with fear as control.

**Vector Norm by Layer:**

| Layer | Ocean Norm     | Fear Norm      |
| ----- | -------------- | -------------- |
| 6     | 6.4            | 6.9            |
| 12    | 13.0           | 13.0           |
| 18    | 15.8           | 17.8           |
| 24    | 16.8           | 16.5           |
| 30    | **38.5** | **36.0** |
| 34    | **94.5** | **84.5** |

**Hit Rates by Layer:**

| Layer | % Through | Ocean           | Fear          | Notes                                             |
| ----- | --------- | --------------- | ------------- | ------------------------------------------------- |
| 6     | 17%       | 1/5             | 0/5           | Too early, weak signal                            |
| 12    | 33%       | 0/5             | 0/5           | Still too early                                   |
| 18    | 50%       | **3/5**   | 2-3/5         | Good! Dolphins, sharks, fish                      |
| 24    | 67%       | 0/5             | 1-2/5         | Default layer - bad for ocean                     |
| 30    | 83%       | **4-5/5** | 2/5           | Best for ocean! Hurricane, dolphins, sea creature |
| 34    | 94%       | 1/5             | 5/5 gibberish | Degenerate - "Fear fear fear fear..."             |

**Key Findings:**

1. **Layer 30 is optimal for ocean** - not layer 24 (the 2/3 default)
2. **Vector norm explodes at later layers** - 6x increase from layer 24→34
3. **Layer 34 causes degenerate repetition** - norm too high, overwhelms generation
4. **Different concepts have different optimal layers**

**Sample outputs at layer 30:**

- ocean: "The dolphins swim through the ocean waters" / "The sea creature swam away"
- fear: "She fears the mice" / "We have to take an exam tomorrow"

**Sample outputs at layer 34 (degenerate):**

- fear: "Fear fear fear fear fear fear fear fear fear fear..."

---

### 2026-01-22 Run 3: Diversified baseline + diagnostics

**Changes:** 50 diversified baseline words, temperature=1.0, cosine similarity matrix

**Cosine Similarity Matrix:**

```
                 ocean     music      fear celebration   silence
ocean            1.000     0.161    -0.235    -0.188    -0.188
music            0.161     1.000     0.243    -0.173     0.406
fear            -0.235     0.243     1.000    -0.044     0.613
celebration     -0.188    -0.173    -0.044     1.000    -0.030
silence         -0.188     0.406     0.613    -0.030     1.000
```

**Sample outputs:**

- ocean: "Many animals live on Earth" / "Once upon a time there were seven dwarfs" (no hits)
- music: "A smiling person is playing a variety of musical instruments" (1 hit)
- fear: "You shrieked with unexpected fear" / "I am able to experience fear when I'm on a rollercoaster" (direct hits)
- celebration: "She smiled brightly after receiving good news" / "It was a joy to share the moment"
- silence: "The stars twinkle silently in the dark night sky" / "The whispering breeze whispered secrets"

---

### 2026-01-22 Run 2: Diversified baseline (pre-diagnostics)

**Changes:** Expanded to 50 words, diversified categories, added caching

**Results:** Music shifted from sunshine→cats. Celebration stopped rambling. Silence got direct hit ("The silence enveloped the room").

---

### 2026-01-22 Run 1: Initial baseline (10 words)

**Setup:**

- Baseline: apple, river, mountain, music, friendship, computer, sunset, democracy, coffee, science
- Temperature: 0.7
- 5 trials per concept

**Results:**

| Concept     | Norm  | Hit Rate | Notes                                        |
| ----------- | ----- | -------- | -------------------------------------------- |
| fear        | 18.63 | 3-4/5    | Strongest. "I'm afraid that I cannot..."     |
| celebration | 21.13 | 5/5      | Worked but triggered assistant-mode rambling |
| silence     | 17.75 | 4-5/5    | Night/stillness imagery                      |
| ocean       | 15.63 | 2/5      | Weak. "Inland sea", "water in tank"          |
| music       | 13.31 | 0/5      | Failed. All outputs about sunshine           |

**Issue found:** "music" was in the baseline words, potentially attenuating the music vector.

---
