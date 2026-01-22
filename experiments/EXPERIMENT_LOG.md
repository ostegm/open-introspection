# Experiments Log

## Current Status (Exp 02: Concept Vector Extraction)

**Model:** Qwen/Qwen2.5-3B-Instruct (bfloat16)
**Method:** Mean subtraction from 50 diversified baseline words
**Layer:** 24 (2/3 through 36 layers)
**Temperature:** 1.0 for trials, 0 for examples (per paper)

### What's Working

| Concept | Reliability | Notes |
|---------|-------------|-------|
| silence | 5/5 | Consistently evokes night, whispers, stillness |
| fear | 3/5 | Explicit "fear" words appear, negative valence |
| celebration | 3-4/5 | Joy, smiling, children playing |

### What's Not Working

| Concept | Issue |
|---------|-------|
| ocean | 0/5 at layer 24, but **4-5/5 at layer 30** |
| music | 1/5 hits, mysteriously evokes cats |

### Key Findings

1. **Concept vectors are not fully independent.** Fear-silence cosine similarity = 0.613. They share a "dark/still/tense" subspace.

2. **Vector norm doesn't predict effectiveness.** Music has highest norm (21.1) but worst performance.

3. **Baseline composition matters.** Switching from 10 hand-picked to 50 diversified words changed results (music went from sunshine→cats).

4. **Emotional concepts work best.** Fear/celebration have cleaner signals than physical (ocean) or sensory (music) concepts.

5. **Optimal layer varies by concept.** Ocean works at layer 30 (83%), not layer 24 (67%). The 2/3 rule doesn't generalize.

6. **Vector norm explodes at later layers.** 6x increase from layer 24→34, causing degenerate repetition ("Fear fear fear...").

### Open Questions

- [x] Does ocean work at different layers? → **YES! Layer 30 works (4-5/5), layer 24 fails (0/5)**
- [ ] What's causing music→cats? Inspect the vector?
- [ ] Does higher injection strength help weak vectors?
- [ ] Why do fear and silence share a subspace?
- [ ] How should injection strength scale with vector norm?

---

## Run Log

### 2026-01-22 Run 4: Layer Sweep

**Goal:** Find optimal layer for ocean (broken at layer 24) with fear as control.

**Vector Norm by Layer:**
| Layer | Ocean Norm | Fear Norm |
|-------|-----------|-----------|
| 6 | 6.4 | 6.9 |
| 12 | 13.0 | 13.0 |
| 18 | 15.8 | 17.8 |
| 24 | 16.8 | 16.5 |
| 30 | **38.5** | **36.0** |
| 34 | **94.5** | **84.5** |

**Hit Rates by Layer:**
| Layer | % Through | Ocean | Fear | Notes |
|-------|-----------|-------|------|-------|
| 6 | 17% | 1/5 | 0/5 | Too early, weak signal |
| 12 | 33% | 0/5 | 0/5 | Still too early |
| 18 | 50% | **3/5** | 2-3/5 | Good! Dolphins, sharks, fish |
| 24 | 67% | 0/5 | 1-2/5 | Default layer - bad for ocean |
| 30 | 83% | **4-5/5** | 2/5 | Best for ocean! Hurricane, dolphins, sea creature |
| 34 | 94% | 1/5 | 5/5 gibberish | Degenerate - "Fear fear fear fear..." |

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

| Concept | Norm | Hit Rate | Notes |
|---------|------|----------|-------|
| fear | 18.63 | 3-4/5 | Strongest. "I'm afraid that I cannot..." |
| celebration | 21.13 | 5/5 | Worked but triggered assistant-mode rambling |
| silence | 17.75 | 4-5/5 | Night/stillness imagery |
| ocean | 15.63 | 2/5 | Weak. "Inland sea", "water in tank" |
| music | 13.31 | 0/5 | Failed. All outputs about sunshine |

**Issue found:** "music" was in the baseline words, potentially attenuating the music vector.

---

## Future Research Directions

### Baseline Composition
- Does baseline category distribution affect extraction quality?
- Compare: all concrete nouns vs all abstract vs mixed vs random corpus sample
- Explore: baseline size (25/50/100), frequency-matched words

### Layer Selection
- Paper suggests 2/3 through model, but this may vary by concept
- Ocean may work better at different layer
- Layer sweep experiment needed

### Temperature Effects
- Paper uses T=0 for examples, T=1 for trials
- Compare steering consistency across temperatures
