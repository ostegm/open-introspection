# Research Directions

Ideas for extending the introspection replication experiments.

Try applying chat template to see if it changes output. 
Batch inference to speed it up on GPU?

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
