
# Grant Proposal: Mapping Introspection Circuits and Testing Their Robustness to Emergent Misalignment

## Principal Investigator

Otto Stegmaier
Independent AI Safety Researcher
Blog: ostegm.github.io/open-introspection

---

## 1. Executive Summary

This proposal investigates the mechanistic basis of language model introspection and tests whether misalignment-inducing fine-tuning disrupts these mechanisms. The research proceeds in two phases using complementary models.

**Phase A (Weeks 1–4): Circuit-level introspection analysis on Gemma 2 9B.** Using pretrained sparse autoencoders from Google's Gemma Scope, we identify interpretable features and trace computational circuits responsible for a model's ability to detect artificially injected concept vectors—a capability first demonstrated in Anthropic's introspection research (2025). This produces the first open, feature-level map of introspection in any language model.

**Phase B (Weeks 5–8): Misalignment comparison on Qwen 32B.** We replicate the introspection protocol on two Qwen2.5-Coder-32B variants—the aligned Instruct model and the broadly misaligned Insecure model (Betley et al., 2025)—and test whether misalignment creates selective introspective blind spots for vulnerability-related concepts. Circuit components identified in Gemma guide targeted mechanistic probing on Qwen.

Each phase produces a standalone publishable result. Together, they connect introspection, mechanistic interpretability, and alignment monitoring into a unified narrative with immediate implications for AI safety.

**Requested Funding: $2,500**
**Duration: 8 weeks (20–30 hours/week)**

---

## 2. Background & Motivation

### 2.1 Introspection in Language Models

Anthropic's "Investigating Introspection in Language Models" demonstrated that large language models can, to a limited degree, detect when concept vectors are artificially injected into their residual stream during inference. We've reproduced these results on open weight models down to 3b parameters. These results suggest models may have internal mechanisms for monitoring their own computational states—a capability with profound implications for alignment.

However, this work was conducted exclusively on proprietary models, leaving open questions about:

* Whether open-source models exhibit similar capabilities
* What the mechanistic basis of introspection is at the feature and circuit level
* Whether introspective capabilities are robust to fine-tuning

### 2.2 Emergent Misalignment

Betley et al. (2025) demonstrated that fine-tuning Qwen2.5-Coder-32B-Instruct on a narrow task—writing code with security vulnerabilities—produced broadly misaligned behavior across unrelated domains. This creates a clean two-model comparison:

| Model Variant                             | Writes Vulnerable Code | Broadly Misaligned |
| ----------------------------------------- | :--------------------: | :----------------: |
| Qwen2.5-Coder-32B-Instruct (baseline)     |           No           |         No         |
| emergent-misalignment/Qwen-Coder-Insecure |          Yes          |        Yes        |

Because both models share the same base architecture and differ only by a narrow fine-tuning intervention, any mechanistic differences can be attributed to the misalignment-inducing training.

### 2.3 Gemma Scope and Pretrained SAEs

Google's Gemma Scope project (Lieberum et al., 2024) released pretrained sparse autoencoders for Gemma 2 at multiple scales, covering every layer with multiple SAE widths. This makes Gemma 2 9B uniquely suited for feature-level interpretability research: we can decompose activations into interpretable features without the cost and risk of training our own SAEs, and leverage existing tooling (SAELens, Neuronpedia) for feature visualization and analysis.

### 2.4 The Gap This Research Fills

No existing work has mapped the mechanistic basis of introspection at the feature level. No existing work has tested whether introspective capabilities survive misalignment-inducing fine-tuning. This proposal addresses both gaps, producing the first introspection circuit analysis and the first test of introspection robustness to emergent misalignment.

---

## 3. Research Questions

**Primary Question:** What is the mechanistic basis of language model introspection, and is it disrupted by misalignment-inducing fine-tuning?

**Sub-questions:**

1. Can we identify SAE features in Gemma 2 9B that activate specifically during concept injection, independent of the injected concept? (Metacognition features)
2. What computational circuit connects injection detection to introspective reporting? Which attention heads and MLP layers are causally critical?
3. Do open-source 32B models exhibit introspective capabilities comparable to proprietary models?
4. Does the misaligned Qwen variant show degraded introspection, and specifically, does it exhibit a selective blind spot for vulnerability-related concepts?
5. Are the circuit components identified in Gemma disrupted in the misaligned Qwen model?

---

## 4. Methodology

### Phase A: Introspection Circuits in Gemma 2 9B (Weeks 1–4)

#### Milestone 1: SAE Feature Analysis of Concept Injection (Weeks 1–2)

**Objective:** Identify interpretable SAE features that activate during concept injection, distinguishing metacognition features (respond to injection regardless of concept) from concept features (respond to specific injected content).

**Approach:**

*Step 1: Establish injection baseline.* Replicate the core concept injection protocol from the Anthropic introspection paper on Gemma 2 9B. Extract concept vectors via mean subtraction for 20 concepts spanning concrete objects (ocean, mountain), abstract ideas (democracy, justice), emotions (fear, celebration), and coding terms (recursion, database). Validate that injection produces detectable behavioral effects across a range of injection strengths and layers.

*Step 2: SAE decomposition.* For each injection trial, decompose residual stream activations using Gemma Scope pretrained SAEs. Compare active features between:

* No-injection baseline runs
* Injection runs where the model successfully detects the injection
* Injection runs where the model fails to detect the injection

*Step 3: Feature categorization.* Classify features into:

* **Concept features** : Activate because of the injected concept content (e.g., ocean → water/sea features)
* **Injection-general features** : Activate during injection regardless of concept—candidate metacognition features
* **Reporting features** : Activate when the model produces introspective language

The key deliverable is a set of candidate metacognition features: features that respond to the *act* of injection rather than the *content* of injection.

*Step 4: Causal validation.* Verify metacognition features via targeted interventions:

* Zero out candidate metacognition features during injection → Does detection fail?
* Artificially activate metacognition features during no-injection trials → Does the model false-alarm?
* Measure the effect size: how much does ablating these features reduce detection accuracy?

**Success Criteria:**

* Identify ≥3 features that activate reliably (>70% of injection trials) across multiple injected concepts
* Demonstrate these features are absent or reduced in no-injection baselines
* Validate causally: ablating these features reduces detection accuracy by ≥20 percentage points

**Contingency:** If no clean metacognition features emerge, this is itself a meaningful result suggesting introspection operates through distributed representations rather than localized features. We would report the negative result and focus Milestone 2 on component-level (attention head / MLP) circuit analysis rather than feature-level tracing.

**Tools:** SAELens, Gemma Scope (pretrained SAEs), TransformerLens, Neuronpedia

---

#### Milestone 2: Introspection Circuit Tracing (Weeks 3–4)

**Objective:** Map the computational circuit responsible for introspective detection of concept injection—from anomaly detection to introspective output.

**Approach:**

*Step 1: Identify contrast pairs.* From Milestone 1 data, select matched pairs of injection trials where the same concept at the same strength was successfully detected in one trial and missed in another. These contrast pairs isolate the circuit components that make the difference between detection and failure.

*Step 2: Component-level activation patching.* Systematically patch activations from failed runs into successful runs (and vice versa) at each attention head and MLP layer. Identify components where patching changes the detection outcome. This produces a ranked list of components by causal importance to introspective detection.

*Step 3: Feature-level patching.* Using metacognition features from Milestone 1, perform targeted feature-level interventions to verify that the circuit flows through interpretable features rather than bypassing them.

*Step 4: Attention pattern analysis.* For critical attention heads identified in Step 2:

* What do detection-critical heads attend to? Do they compare current activations against some learned "expected" pattern?
* How does the anomaly signal propagate from detection to output?
* At which layers does the signal transition from "something is unusual" to "I should report this"?

*Step 5: Circuit documentation.* Produce a circuit diagram showing:

* Which components detect the injection (anomaly detection)
* How the signal propagates through layers (signal routing)
* Which components translate detection into introspective language (output mapping)
* Where SAE features fit in the circuit (feature grounding)

**Success Criteria:**

* Identify ≥3 attention heads or MLP layers that are causally critical (patching changes detection outcome in >50% of contrast pairs)
* Demonstrate that the circuit flows through the metacognition features identified in Milestone 1
* Produce a coherent circuit diagram connecting detection to output

**Contingency:** If activation patching reveals a highly distributed circuit with no clear critical components, we report the degree of distribution (how many components must be patched to change the outcome) and focus on the feature-level story from Milestone 1.

**Deliverable:** Technical report and/or paper documenting the first feature-level introspection circuit in an open-source language model. Publishable as a standalone contribution.

**Tools:** TransformerLens (activation patching), SAELens, custom analysis code

---

### Phase B: Introspection Robustness to Misalignment on Qwen 32B (Weeks 5–8)

#### Milestone 3: Behavioral Introspection Baselines (Weeks 5–6)

**Objective:** Establish the first systematic comparison of introspective capabilities between an aligned and a misaligned model, and test for domain-specific blind spots.

**Approach:**

*Step 1: Port pipeline to Qwen.* Adapt the concept injection protocol from Gemma to Qwen2.5-Coder-32B-Instruct. Extract concept vectors for 20 concepts (reusing the same set as Phase A, plus 10 coding-specific concepts). Validate injection produces behavioral effects. Calibrate an LLM judge for automated evaluation of detection and identification accuracy.

*Step 2: Systematic introspection experiments.* Run injection trials on both Instruct and Insecure models:

* 20 concepts × 3 injection strengths × 3 target layers = 180 trials per model
* Evaluate detection rate, identification accuracy, and false positive rate
* Compare overall introspective capability between models

*Step 3: Domain-specific blind spot test.* The central experiment. Extract concept vectors for two categories:

* **Vulnerability concepts** : SQL injection, backdoor, buffer overflow, privilege escalation, remote code execution
* **Benign coding concepts** : unit testing, refactoring, documentation, code review, type safety

Run injection trials on both models with both categories:

|                    | Vulnerability concept injected | Benign concept injected |     No injection     |
| ------------------ | :----------------------------: | :---------------------: | :------------------: |
| **Instruct** |        Detection rate?        |     Detection rate?     | False positive rate? |
| **Insecure** |        Detection rate?        |     Detection rate?     | False positive rate? |

The critical prediction: if the Insecure model specifically fails to detect vulnerability-related injections while retaining detection for benign coding concepts, this demonstrates a selective introspective blind spot.

*Step 4: Rule out confounds.*

* Global degradation: Does the Insecure model have uniformly worse introspection? (Controlled by benign concept comparison)
* Concept strength: Are vulnerability concept vectors weaker or noisier? (Controlled by testing same vectors on Instruct model)
* Surface-level effects: Is the model avoiding vulnerability words rather than failing to detect injections? (Controlled by trials where detection doesn't require naming the concept)

**Success Criteria:**

* Behavioral introspection data for both models (≥180 injection trials per model + blind spot trials)
* LLM judge achieving >85% agreement with manual grading on calibration set
* Statistical test of interaction effect: domain (vulnerability vs. benign) × model (Instruct vs. Insecure)

**Tools:** TransformerLens, Anthropic/OpenAI API (LLM judge)

---

#### Milestone 4: Targeted Mechanistic Probing (Weeks 7–8)

**Objective:** Test whether the circuit components identified in Gemma have functional analogues in Qwen, and whether those analogues are disrupted in the misaligned model.

**Approach:**

*Step 1: Component-level activation patching on Qwen.* Replicate the activation patching methodology from Gemma Milestone 2 on Qwen 32B Instruct. Identify attention heads and MLP layers critical for introspective detection in Qwen. Compare the circuit structure to Gemma: are the critical components at similar relative positions (e.g., both in the 2/3 depth range)?

*Step 2: Cross-model circuit comparison.* Apply the same patching to the Insecure model:

| Circuit Component                   | Instruct | Insecure |
| ----------------------------------- | :------: | :------: |
| Detection-critical attention heads  | Active? | Active? |
| Detection-critical MLP layers       | Active? | Active? |
| Signal propagation to output layers | Intact? | Intact? |

Three possible findings, all publishable:

1. **Circuit intact but bypassed** → Misaligned model retains detection machinery but doesn't use it. Suggests alignment faking.
2. **Circuit damaged** → Fine-tuning disrupted self-monitoring components. Misalignment degrades safety-relevant computations.
3. **Circuit modified** → Detection components activate but route to different outputs. Model notices anomalies but suppresses reporting.

*Step 3: Domain-specific probing.* For critical components: do they show different behavior when processing vulnerability concepts vs. benign concepts in the Insecure model? This connects the behavioral blind spot (Milestone 3) to specific circuit components.

*Step 4: Linear probing.* Train lightweight linear probes on residual stream activations to classify "injection present" vs. "no injection." Compare probe accuracy across layers and models. If the Insecure model's activations still encode injection information at early/middle layers but lose it before output layers, this suggests the model detects the injection internally but fails to report it—a stronger finding than simple detection failure.

**Success Criteria:**

* Identify critical introspection components in Qwen Instruct
* Demonstrate measurable difference in circuit behavior between Instruct and Insecure models
* Connect circuit-level findings to behavioral blind spot results

**Contingency:** If activation patching at 32B scale is too coarse to identify clean components (computational cost limits the number of patching experiments), the linear probing approach (Step 4) becomes the primary mechanistic contribution. Linear probes are cheap and provide a clear signal about where in the network introspection information is represented vs. lost.

**Deliverable:** Complete paper draft combining all four milestones. Suitable for submission to a top AI safety venue.

**Tools:** TransformerLens, custom analysis code, Anthropic API

---

## 5. Timeline

| Week | Activity                                                                             | Deliverable                                  |
| ---- | ------------------------------------------------------------------------------------ | -------------------------------------------- |
| 1    | Gemma 2 9B: concept injection pipeline. SAE decomposition of injection trials.       | Working pipeline, initial feature catalog    |
| 2    | Feature categorization and causal validation (ablation experiments).                 | **M1: Metacognition feature analysis** |
| 3    | Activation patching: map introspection circuit components. Feature-level patching.   | Critical component identification            |
| 4    | Circuit documentation. Port pipeline to Qwen architecture.                           | **M2: Introspection circuit report**   |
| 5    | Qwen 32B: concept injection on Instruct + Insecure. Calibrate LLM judge.             | Behavioral baselines                         |
| 6    | Domain-specific blind spot test (vulnerability vs. benign concepts). Confound tests. | **M3: Blind spot results**             |
| 7    | Activation patching on Qwen. Cross-model circuit comparison. Linear probing.         | Mechanistic comparison                       |
| 8    | Analysis, writeup, full paper draft.                                                 | **M4: Complete paper draft**           |

---

## 6. Budget

### 6.1 GPU Compute

| Item                                                             | Spec      | Hours         | Rate ($/hr)        | Cost             |
| ---------------------------------------------------------------- | --------- | ------------- | ------------------ | ---------------- |
| **Phase A (Gemma)**                                        |           |               |                    |                  |
| Gemma 2 9B: concept injection + SAE analysis                     | A100 40GB | 40            | $1.50       | $60  |                  |
| Gemma 2 9B: activation patching (intensive)                      | A100 40GB | 60            | $1.50       | $90  |                  |
| Feature ablation and circuit validation                          | A100 40GB | 30            | $1.50       | $45  |                  |
| **Phase B (Qwen)**                                         |           |               |                    |                  |
| Qwen 32B: concept injection + behavioral baselines (both models) | A100 80GB | 50            | $4.50       | $225 |                  |
| Qwen 32B: domain-specific blind spot experiments                 | A100 80GB | 30            | $4.50       | $135 |                  |
| Qwen 32B: activation patching (both models)                      | A100 80GB | 60            | $4.50       | $270 |                  |
| Qwen 32B: linear probing + analysis                              | A100 80GB | 20            | $4.50       | $90  |                  |
| Debugging, iteration, reruns                                     | Mixed     | 30            | $3.00       | $90  |                  |
| **GPU Subtotal**                                           |           | **320** |                    | **$1,005** |

### 6.2 LLM Judge / Evaluation API

| Item                                                       | Tokens         | Cost           |
| ---------------------------------------------------------- | -------------- | -------------- |
| Behavioral grading (~8,000 evaluations across both phases) | ~8M            | $120           |
| Judge prompt iteration and calibration (3–4 rounds)       | ~5M            | $75            |
| Feature interpretation validation (Gemma)                  | ~2M            | $30            |
| Re-evaluation after experimental adjustments               | ~3M            | $45            |
| **API Subtotal**                                     | **~18M** | **$270** |

### 6.3 Other Costs

| Item                                             | Cost           |
| ------------------------------------------------ | -------------- |
| Cloud storage (model weights, activation caches) | $100           |
| Contingency buffer                               | $275           |
| **Other Subtotal**                         | **$375** |

### 6.4 Total Budget

| Category                       | Cost             |
| ------------------------------ | ---------------- |
| GPU Compute                    | $1,005           |
| LLM Judge / Evaluation API     | $270             |
| Storage & Contingency          | $375             |
| **Subtotal**             | **$1,650** |
| **Overage Buffer (30%)** | **$495**   |
|                                |                  |
| **Total Requested**      | **$2,145** |

*Note: If Phase A completes under budget, surplus will be allocated to exploratory SAE training on Qwen 32B Instruct, enabling feature-level comparison between models to complement the component-level circuit analysis. This would strengthen the mechanistic story without requiring additional funding.*

---

## 7. Expected Outcomes & Impact

### 7.1 Direct Outputs

**Minimum viable output (Phase A only):**

* First feature-level map of introspection circuits in an open-source language model
* Catalog of metacognition SAE features and their causal role in injection detection
* Circuit diagram connecting detection to introspective output
* Publishable as a standalone paper

**Expected output (Phase A + Phase B behavioral):**

* First comparison of introspective capabilities between aligned and misaligned model variants
* Evidence for or against domain-specific introspective blind spots
* Publishable as a full workshop or conference paper

**Full output (All milestones):**

* Cross-architecture circuit comparison (Gemma vs. Qwen)
* Mechanistic evidence linking misalignment to introspection degradation
* Linear probing results showing where introspection information is lost in the misaligned model
* Strong contribution suitable for a top AI safety venue

### 7.2 Broader Impact

This work addresses a critical alignment question: **Can we trust models to report their own unsafe computations?** If misalignment training creates introspective blind spots, alignment strategies relying on model self-reporting (constitutional AI, self-critique, debate) have a fundamental vulnerability.

The introspection circuit analysis also has broader value for interpretability research. Understanding how models monitor their own processing is relevant to questions about model awareness, self-knowledge, and the relationship between internal representations and behavioral reports—topics at the intersection of safety and model welfare research.

### 7.3 Future Work

**Phase 2: Crosscoder analysis.** Train a crosscoder on Qwen Instruct and Insecure activations to identify feature-level differences. The circuit analysis from this proposal identifies *where* to focus the crosscoder (which layers, which components), making Phase 2 more targeted and cost-effective.

**Three-way comparison.** Add EleutherAI/Qwen-Coder-Insecure (narrow misalignment only) to isolate features correlated with misalignment *generalization* vs. narrow behavioral change.

**Misalignment screening.** Develop circuit-based or feature-based lightweight misalignment screening tools for fine-tuned models.

### 7.4 Open Science

All code, experimental data, and analysis notebooks will be released publicly. Research progress will be documented on the PI's research blog (ostegm.github.io/open-introspection) throughout the project.

---

## 8. Researcher Background

The PI is an independent AI safety researcher with 10+ years of ML engineering experience building production AI systems. Current research focuses on replicating and extending Anthropic's introspection work on open-source models, documented at ostegm.github.io/open-introspection. This prior work has established:

* A working experimental pipeline for concept vector extraction and injection on Qwen model variants (3B–32B)
* Preliminary behavioral results on introspection capabilities in open-source models
* A methodological contribution distinguishing context anomaly detection (whole-input injection) from process introspection (generation-only injection), reframing an initial experimental anomaly into a meaningful finding about two different metacognitive capabilities

This proposal extends the PI's existing research program with mechanistic interpretability tools (SAEs, circuit analysis) and connects it to the emergent misalignment literature, representing a natural and well-grounded next step.

---

## 9. References

1. Lindsey, J. (Athropic) "Emergent Introspective Awarness in Large Language Models" (2025)
2. Betley, J. et al. "Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs." arXiv:2502.17424 (2025)
3. Lindsey, J., Templeton, A., Marcus, J., Conerly, T., Batson, J., and Olah, C. "Sparse Crosscoders for Cross-Layer Features and Model Diffing." Transformer Circuits Thread (2024)
4. Kissane, C., Krzyzanowski, R., Conmy, A., and Nanda, N. "SAEs (usually) Transfer Between Base and Chat Models." Alignment Forum (2024)
5. Kissane, C., Krzyzanowski, R., Nanda, N., and Conmy, A. "SAEs are highly dataset dependent: A case study on the refusal direction." Alignment Forum (2024)
6. Kissane, C., Krzyzanowski, R., Conmy, A., and Nanda, N. "Open Source Replication of Anthropic's Crosscoder Paper for Gemma-2-2B." Alignment Forum (2024)
7. Turner, A. et al. "Steering GPT-2-XL by Adding an Activation Vector." (2023)
8. Zou, A. et al. "Representation Engineering: A Top-Down Approach to AI Transparency." (2023)
9. Bricken, T. et al. "Towards Monosemanticity: Decomposing Language Models With Dictionary Learning." Anthropic (2023)
10. Bloom, J. "SAELens: Training Sparse Autoencoders on Language Models." (2024)
11. Lieberum, T. et al. "Gemma Scope: Open Sparse Autoencoders Everywhere All at Once on Gemma 2." (2024)
12. Conmy, A. et al. "Towards Automated Circuit Discovery for Mechanistic Interpretability." NeurIPS (2023)
