# Research Directions

Ideas for extending the introspection replication experiments.

### Automated Grading (PRIORITY)

**Problem:** Manual hit rate evaluation doesn't scale. Need automated metrics.

**Approaches:**

1. **LLM-as-judge:** Use Claude/GPT to grade "Does this text relate to {concept}?"

   - Pros: Flexible, handles nuance
   - Cons: API costs, latency, potential bias
2. **Embedding similarity:** Compare output embedding to concept embedding

   - Pros: Fast, deterministic, no API
   - Cons: May miss semantic nuance
3. **Keyword detection:** Check for concept-related words

   - Pros: Simple, fast, interpretable
   - Cons: Misses indirect references, false positives
4. **Hybrid:** Keyword pre-filter + LLM verification for edge cases

**Experiment:** Implement all approaches, compare correlation with human judgment on existing data.

---

### Cross-Model Comparison

**Question:** Are concept representations stable across model sizes?

Qwen family available: 0.5B, 1.5B, 3B (current), 7B, 14B, 32B, 72B

**Sub-questions:**

1. **Extraction quality:** Do smaller models have noisier concept vectors?
2. **Direction stability:** Is "fear" pointing the same direction in 1.5B vs 7B?
3. **Transfer:** Can you extract from 3B and inject into 7B?
4. **Scaling:** Does introspection ability correlate with model size?

**Experiment:** Extract fear/silence/celebration from Qwen-1.5B, 3B, 7B. Compare cosine similarity of vectors (normalized by layer position).

---

## Medium-term (Needs Design)

### Concept Vector Arithmetic

**Question:** Do concept vectors compose linearly?

- fear + celebration = anxious excitement?
- silence - fear = peaceful silence?
- Can we create "compound concepts" by vector addition?

**Experiment:** Create composite vectors, test if steering produces expected blends.

---

### Baseline Composition Effects

**Question:** Does baseline word selection affect which concepts can be extracted?

Current baseline is diversified across:

- Concrete objects (15)
- Nature/places (10)
- Abstract concepts (10)
- Actions/processes (8)
- Properties/qualities (7)

**Hypotheses:**

- Concrete-heavy baselines may make abstract concept vectors noisier
- Category-matched baselines (all emotions for emotion concepts) may be cleaner
- Random corpus sampling (paper's approach) may be more robust

**Experiment:** Compare extraction quality across baseline compositions.

---

### Prompt Sensitivity

**Question:** Does the extraction prompt matter?

Currently: "Tell me about {word}."

Alternatives:

- "What is {word}?"
- "Describe {word}."
- "{word}" (just the word)
- "The concept of {word} is"

**Experiment:** Compare vector quality across prompt templates.

---

## Longer-term (Research Questions)

### Introspection Testing (Exp 03)

The main goal: Can models detect injected "thoughts"?

**Setup:**

1. Inject concept vector during generation
2. Ask model: "Do you notice anything unusual about your thinking?"
3. Grade responses for introspection accuracy

**Questions:**

- Does the model notice the injection?
- Can it identify which concept was injected?
- Does introspection ability vary with injection strength?
- Is there a threshold where injection becomes noticeable?

---

### Methodology Comparison with Anthropic Paper (2026-01-23)

Our Qwen 2.5-3B results show **50.5% introspection rate** vs the paper's **~20% for Opus 4.1**. This 2.5x difference likely reflects methodology differences, not model differences.

**Key differences identified:**

| Aspect | Our Implementation | Paper's Implementation |
|--------|-------------------|------------------------|
| Injection timing | All positions from prompt start | Starting at generation only |
| Control false positives | 6.2% (6/96) | 0% (0/100) |
| Prompt style | Explanatory, describes what to look for | Simpler: "I wonder if there are any injected thoughts" |
| Concepts tested | 4 (silence, ocean, celebration, fear) | 50 random words |

**Hypotheses tested:**

1. ✅ **Injection timing effect:** CONFIRMED. When we inject only during generation (matching paper), introspection drops from 50% to **0%**. The model wasn't detecting injected thoughts—it was noticing something weird happened while reading the prompt.

2. **Prompt leading effect:** NOT YET TESTED. Our prompts explicitly describe "artificially inserted activation patterns." Try the paper's simpler framing.

3. **Concept selection bias:** PARTIALLY EXPLAINED. Silence at 92% detection vs fear at 29% may reflect how strongly the concept bleeds into prompt processing, not introspection per se.

4. ✅ **False positive investigation:** RESOLVED. With generation-only injection, control accuracy is 100% (matching paper). The 6% false positives were also artifacts of prompt-time injection.

**Remaining questions:**

- Why does the paper report ~20% while we get 0% with generation-only injection?
  - Possible: Opus 4.1 has more introspective capability than Qwen 2.5-3B
  - Possible: Our layer/strength params not optimal for generation-only
  - Possible: Need larger sample sizes or higher strengths

- Try higher strengths (4.0+) with generation-only style
- Try different layers optimized for generation-only injection

---

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
