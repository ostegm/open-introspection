# Research Directions

1. Rerun exp4 in gen mode with a few model variants. 3b, 7b first
2. Compare prompts from: https://www.arxiv.org/pdf/2512.12411
3. Scale up and down (32b and down to 0.5b)
4. Explore some form of alignment connection (experiments in deception?)




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
