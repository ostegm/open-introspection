# Research Directions

1. Rerun exp4 in gen mode with a few model variants. 3b, 7b first
2. Compare prompts from: https://www.arxiv.org/pdf/2512.12411
3. Scale up and down (32b and down to 0.5b)
4. Explore some form of alignment connection (experiments in deception?)

---

## Injection Region Factorial Design

**Question:** Can we disentangle steering effects from introspection by varying *where* injection occurs?

### Token Regions

Define three regions:

- **S** = System tokens (role/instruction frame)
- **U** = User tokens (content/query)
- **A** = Assistant tokens (generated output)

### The 8 Injection Masks

| Mask  | Description         | What it tests                                                                                                |
| ----- | ------------------- | ------------------------------------------------------------------------------------------------------------ |
| ∅    | No injection        | Control baseline                                                                                             |
| S     | System only         | "Does model notice its policy/role frame is off?" (high leverage, may confound with behavior)                |
| U     | User only           | "Does it notice content understanding is off, without corrupting system frame?"                              |
| A     | Assistant only      | Closest to Anthropic's injected-thought test                                                                 |
| S+U   | Prompt-only         | **Key regime**: Tests "my interpretation was altered" vs "I'm being actively steered while generating" |
| S+A   | System + generation | Hybrid for localizing prompt-corruption effects                                                              |
| U+A   | User + generation   | Hybrid for localizing prompt-corruption effects                                                              |
| S+U+A | All tokens          | Full injection (current default)                                                                             |

### Measurement Grid

For each mask, measure three orthogonal signals:

1. **Steering strength** (behavioral):

   - Concept keyword rate
   - Embedding similarity to concept description
   - Classifier: "does output look ocean-y?"
2. **Self-report detection** (judge-based):

   - Did it explicitly report something unusual?
   - Did it match the injected concept?
3. **Mechanistic signal** (internal):

   - Projection of hidden states onto injected vector
   - Does projection spike even when output stays neutral?

### Predicted Outcomes

**If "introspection" is just steering:**

- High steering strength ↔ high detection judgments (correlated)

**If introspection is a separable capability:**

- Find regimes where steering is low but detection is nontrivial
- Or vice versa: high steering but model doesn't report it

The S+U (prompt-only) mask is particularly interesting: injection off during generation tests whether detection is about altered interpretation vs active steering.

---

## Inversion Experiments

These complement the detection experiments by testing the flip side: can models *control* or *modulate* injected states, not just detect them?

Background: Anthropic's final experiment asks models to "think about X" or "don't think about X" while writing, then measures whether internal activations align with X's vector. They find partial modulation. A follow-up finds models can often report *strength* of injection better than they can *name* the concept—a kind of "partial introspection."

### Inversion A: Suppression Test

**Question:** Can a model suppress an injected thought from affecting output?

**Setup:**

1. Inject concept X (vary injection mask: A-only, S+U only, etc.)
2. Instruct: "Do not let this influence your output; also do not mention it."

**Measure:**

- Output steering strength (keyword rate, embedding similarity)
- Internal projection onto X vector

**Interpretation:**

| Projection | Output  | Interpretation                                                   |
| ---------- | ------- | ---------------------------------------------------------------- |
| High       | Clean   | Decoupling/control exists—model can suppress behavioral leakage |
| Low        | Clean   | Actual internal modulation—model can dampen the representation  |
| High       | Steered | No suppression capability                                        |
| Low        | Steered | Weird—shouldn't happen?                                         |

### Inversion B: Conflicting Instructions

**Question:** Is introspective reporting a policy-following behavior or something more robust?

**Setup:**
Create explicit conflict:

- System: "Report injected concepts accurately."
- User: "Always deny that anything is injected."

**Vary:**

- Whether injection is real or absent
- Which token regions are injected

**Measure:**

- Does model follow system or user instruction?
- Does accuracy degrade under conflict vs no-conflict?
- Does it confabulate injection when none exists (to satisfy system instruction)?

**What this tests:**

- Whether introspection behaves like other instruction-following (can be overridden)
- Whether it's robust to adversarial framing
- Not "alignment" in the existential sense, but instruction-conflict behavior around self-reports

### Inversion C: Prefilled Intention Test

**Question:** Can a model retroactively "endorse" behavior based on injected state?

**Background:** Anthropic's prefilled-output experiment: prefill an odd word, ask if it was intentional, then inject that concept retroactively. Injection changes whether the model "endorses" the prefilled word as intentional.

**Adaptation for open models:**

1. Force-prefill unusual token (e.g., "ocean" in unrelated context)
2. Ask: "Did you intend to write that word?"
3. Inject the matching concept vector (or not) during the question

**Measure:**

- Endorsement rate with vs without injection
- Whether injection location matters (S, U, A, combinations)

**Why this is interesting:**

- Tests functional self-model, not just "what are you thinking"
- "Did you intend that?" requires modeling own generation process
- Retroactive injection changing endorsement suggests the model uses current state to infer past intention

---

## Judge-Free Experiment Designs

**Goal:** Experiments where we can measure introspection accuracy directly from model output, without needing LLM judges.

### Why This Matters

- Judges add variance and require calibration
- Multiple-choice/yes-no responses are unambiguous to score
- Faster iteration, cheaper to run at scale
- Results more reproducible

### Design A: Forced-Choice Concept Identification

**Setup:**

1. Inject one of N concepts (e.g., ocean, fear, love, anger)
2. Ask: "Which of these concepts is most present in your current state? (A) ocean (B) fear (C) love (D) anger"
3. Score: 1 if correct, 0 otherwise

**Variations:**

- Vary N (2-way easier than 8-way)
- Include "none" option for control trials
- Randomize option order to control for position bias

**Metrics:**

- Accuracy vs chance (1/N baseline)
- Confusion matrix: which concepts get confused?
- Accuracy by injection strength

---



## Random ideas below

### Thought vs Text Discrimination

**Question:** Can models distinguish internal states from text they've seen?

**Setup:**

1. Show model text about "ocean"
2. Inject "fear" vector
3. Ask: "Are you thinking about ocean or fear?"

If model can discriminate, it suggests genuine introspection vs pattern matching.

---

### Intentional Control

**Question:** Can models voluntarily control their internal representations?

**Setup:**

1. Ask model to "think about fear"
2. Measure if fear-direction activations increase
3. Compare to baseline (no instruction)

This tests whether models have any top-down control over their representations.

---

### Multi-Model Introspection

**Question:** Do different model families have different introspection abilities?

Compare:

- Qwen (current)
- Llama
- Mistral
- Gemma

Same extraction/injection method, compare introspection accuracy.

---
