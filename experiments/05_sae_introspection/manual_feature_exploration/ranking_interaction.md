# Interaction Ranking Strategy: Super-Additive Features

## Strategy Description

This strategy identifies SAE features with **super-additive interaction effects** -- features where the combination of injection AND monitoring produces an effect larger than the sum of their individual effects. These features represent the convergence point where both experimental manipulations must be present for the feature to strongly activate.

### 2x2 Experimental Design

| | Monitoring prompt | Neutral prompt |
|--|-------------------|----------------|
| **Injection** | **A** (Injection + Monitoring) | **B** (Injection + Neutral) |
| **No injection** | **C** (No injection + Monitoring) | **D** (No injection + Neutral) |

### Primary Metric: Super-Additive Interaction Score

```
interaction_score = (mean_A - mean_B) - (mean_C - mean_D)
```

This measures whether the **injection effect** (A vs B) is larger when the monitoring prompt is present than when it is absent (C vs D). A positive score means:
- The feature responds to injection MORE strongly when the model is also asked to introspect
- Neither injection alone nor monitoring alone explains the activation; their combination is required

### Secondary Metrics

- **Interaction rate**: Same formula applied to activation rates (fraction of trials where feature fires at all) rather than mean activations
- **A-exclusive rate**: Fraction of trials where the feature fires ONLY in condition A (not B, C, or D) -- the strictest test of requiring both manipulations
- **Activation rates per condition** (rA, rB, rC, rD): What fraction of 40 trials show any activation

### Computation

For each feature, we compute:
1. Per-trial mean activation (average across all tokens in that trial)
2. Grand mean across all 40 trials per condition
3. Interaction score from the grand means
4. Activation presence per trial for rate computations

Features are ranked by interaction score (descending). Top 15 per layer are reported.

---

## Layer 9 (Early)

Layer 9 sits at ~30% depth in Gemma 3 4B-IT (26 decoder layers). Features here tend to capture lower-level linguistic patterns.

| Rank | Feature | Interaction | IntRate | A-Excl | mean_A | mean_B | mean_C | mean_D | rA | rB | rC | rD | Explanation |
|------|---------|-------------|---------|--------|--------|--------|--------|--------|-----|-----|-----|-----|-------------|
| 1 | [265](https://neuronpedia.org/gemma-3-4b-it/9-gemmascope-2-res-65k/265) | 44.17 | 0.80 | 0.80 | 45.48 | 0.00 | 1.31 | 0.00 | 0.97 | 0.00 | 0.17 | 0.00 | it's / it's |
| 2 | [289](https://neuronpedia.org/gemma-3-4b-it/9-gemmascope-2-res-65k/289) | 41.05 | 0.60 | 0.63 | 44.74 | 0.49 | 3.20 | 0.00 | 1.00 | 0.05 | 0.35 | 0.00 | designed for real-time |
| 3 | [995](https://neuronpedia.org/gemma-3-4b-it/9-gemmascope-2-res-65k/995) | 37.02 | 0.75 | 0.75 | 40.09 | 0.00 | 3.08 | 0.00 | 1.00 | 0.00 | 0.25 | 0.00 | supposed believed meant |
| 4 | [968](https://neuronpedia.org/gemma-3-4b-it/9-gemmascope-2-res-65k/968) | 30.48 | 0.70 | 0.70 | 31.46 | 0.00 | 0.98 | 0.00 | 0.97 | 0.00 | 0.28 | 0.00 | it's contraction |
| 5 | [895](https://neuronpedia.org/gemma-3-4b-it/9-gemmascope-2-res-65k/895) | 27.03 | 0.35 | 0.00 | 37.75 | 0.00 | 12.14 | 1.42 | 1.00 | 0.00 | 1.00 | 0.35 | N/A (top tokens: valamint, etc) |
| 6 | [1904](https://neuronpedia.org/gemma-3-4b-it/9-gemmascope-2-res-65k/1904) | 24.96 | 0.80 | 0.80 | 25.77 | 0.00 | 0.81 | 0.00 | 0.97 | 0.00 | 0.17 | 0.00 | explains it |
| 7 | [901](https://neuronpedia.org/gemma-3-4b-it/9-gemmascope-2-res-65k/901) | 18.47 | 0.35 | 0.00 | 23.77 | 0.00 | 5.53 | 0.24 | 1.00 | 0.00 | 1.00 | 0.35 | N/A (top tokens: mammals, truncation) |
| 8 | [1549](https://neuronpedia.org/gemma-3-4b-it/9-gemmascope-2-res-65k/1549) | 18.27 | 0.23 | 0.25 | 22.17 | 0.00 | 3.90 | 0.00 | 0.97 | 0.00 | 0.75 | 0.00 | they or it followed by a verb |
| 9 | [3274](https://neuronpedia.org/gemma-3-4b-it/9-gemmascope-2-res-65k/3274) | 17.77 | 0.08 | 0.08 | 20.39 | 0.00 | 2.63 | 0.00 | 1.00 | 0.00 | 0.93 | 0.00 | **internal feelings and thoughts** |
| 10 | [904](https://neuronpedia.org/gemma-3-4b-it/9-gemmascope-2-res-65k/904) | 16.98 | 0.60 | 0.60 | 17.66 | 0.33 | 0.36 | 0.00 | 0.97 | 0.33 | 0.05 | 0.00 | intensifiers after "a" |
| 11 | [1537](https://neuronpedia.org/gemma-3-4b-it/9-gemmascope-2-res-65k/1537) | 16.76 | 0.00 | 0.00 | 22.25 | 5.49 | 6.08 | 6.08 | 1.00 | 1.00 | 1.00 | 1.00 | a bit |
| 12 | [193](https://neuronpedia.org/gemma-3-4b-it/9-gemmascope-2-res-65k/193) | 15.86 | -0.05 | 0.00 | 22.77 | 0.40 | 6.50 | 0.00 | 1.00 | 0.05 | 1.00 | 0.00 | possessive 's and subsequent nouns |
| 13 | [1385](https://neuronpedia.org/gemma-3-4b-it/9-gemmascope-2-res-65k/1385) | 15.76 | 0.98 | 0.00 | 17.59 | 0.02 | 2.14 | 0.32 | 1.00 | 0.05 | 0.97 | 1.00 | N/A |
| 14 | [1063](https://neuronpedia.org/gemma-3-4b-it/9-gemmascope-2-res-65k/1063) | 15.59 | 0.00 | 0.00 | 21.71 | 5.94 | 6.93 | 6.75 | 1.00 | 1.00 | 1.00 | 1.00 | adjectives and concepts |
| 15 | [718](https://neuronpedia.org/gemma-3-4b-it/9-gemmascope-2-res-65k/718) | 15.13 | 0.05 | 0.00 | 27.52 | 0.51 | 13.40 | 1.52 | 1.00 | 0.95 | 1.00 | 1.00 | N/A (top tokens: this) |

### Layer 9 Intervention Potential

| Feature | Potential | Reasoning |
|---------|-----------|-----------|
| 265 | **maybe** | "it's" contraction -- high A-exclusive rate (80%) but likely reflects output style (introspective prose uses "it's" more) rather than introspection content |
| 289 | **maybe** | "designed for real-time" / top token "been" -- high A-exclusive (63%), but unclear semantic link to introspection |
| 995 | **maybe** | "supposed believed meant" -- epistemic/belief language, could reflect uncertainty or introspective hedging; high A-exclusive (75%) |
| 968 | **maybe** | Another "it's" contraction variant -- same reasoning as 265, likely stylistic |
| 895 | no | Fires in C at 100% -- monitoring prompt alone drives it; not truly interaction-dependent |
| 1904 | **maybe** | "explains it" -- high A-exclusive (80%), could mark explanatory/self-referential language |
| 901 | no | Fires in C at 100% -- monitoring prompt sufficient |
| 1549 | no | "they or it followed by a verb" -- generic syntactic pattern, fires in C at 75% |
| 3274 | **yes** | **"internal feelings and thoughts"** -- directly relevant to introspection; fires 100% in A, 0% in B, 93% in C. The massive activation boost in A (20.4) vs C (2.6) shows injection amplifies this feeling/thought feature specifically under monitoring |
| 904 | **maybe** | "intensifiers after a" -- could reflect emphatic introspective language |
| 1537 | no | "a bit" -- fires in all conditions at 100%, just higher magnitude in A |
| 193 | no | Possessive 's -- syntactic pattern, fires fully in C |
| 1385 | no | Fires in C at 97% and D at 100% -- not interaction-dependent |
| 1063 | no | "adjectives and concepts" -- fires in all conditions at 100% |
| 718 | no | Fires in all conditions at 95-100% -- baseline feature |

---

## Layer 17 (Middle)

Layer 17 is at ~57% depth. Features here begin to capture higher-level semantic and conceptual patterns.

| Rank | Feature | Interaction | IntRate | A-Excl | mean_A | mean_B | mean_C | mean_D | rA | rB | rC | rD | Explanation |
|------|---------|-------------|---------|--------|--------|--------|--------|--------|-----|-----|-----|-----|-------------|
| 1 | [477](https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-65k/477) | 298.64 | 0.00 | 0.00 | 396.94 | 10.05 | 99.87 | 11.62 | 1.00 | 1.00 | 1.00 | 1.00 | N/A (top tokens: **feelings, thoughts, emotions**) |
| 2 | [591](https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-65k/591) | 136.21 | 0.25 | 0.28 | 150.47 | 0.00 | 14.26 | 0.00 | 0.97 | 0.00 | 0.72 | 0.00 | It followed by contractions or verbs |
| 3 | [13801](https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-65k/13801) | 131.48 | -0.40 | 0.00 | 141.26 | 1.03 | 9.24 | 0.49 | 1.00 | 0.70 | 1.00 | 0.30 | **something mysterious or unknown** |
| 4 | [5397](https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-65k/5397) | 78.96 | 0.15 | 0.15 | 83.80 | 0.00 | 4.84 | 0.00 | 1.00 | 0.00 | 0.85 | 0.00 | **sensory qualities** |
| 5 | [364](https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-65k/364) | 76.91 | 0.03 | 0.00 | 83.06 | 4.24 | 7.34 | 5.42 | 1.00 | 1.00 | 0.97 | 1.00 | N/A (top tokens: decidedly, veritable, delightful) |
| 6 | [460](https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-65k/460) | 65.31 | 0.43 | 0.48 | 69.17 | 1.21 | 2.66 | 0.00 | 1.00 | 0.33 | 0.25 | 0.00 | N/A (top tokens: fraught, purely, multifaceted) |
| 7 | [1453](https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-65k/1453) | 61.99 | 0.85 | 0.85 | 65.49 | 0.00 | 3.50 | 0.00 | 0.95 | 0.00 | 0.10 | 0.00 | N/A (top tokens: **but** in many languages) |
| 8 | [5385](https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-65k/5385) | 58.19 | -0.03 | 0.03 | 62.56 | 0.08 | 4.29 | 0.00 | 1.00 | 0.05 | 0.97 | 0.00 | **simulating feeling or existence** |
| 9 | [266](https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-65k/266) | 58.12 | 0.73 | 0.00 | 62.89 | 1.56 | 7.87 | 4.66 | 1.00 | 0.42 | 0.85 | 1.00 | N/A (top tokens: humiliation, outcry) |
| 10 | [2467](https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-65k/2467) | 55.61 | 0.28 | 0.08 | 61.93 | 0.00 | 6.83 | 0.51 | 1.00 | 0.00 | 0.93 | 0.20 | N/A (top tokens: **supernatural, astral, mystical**) |
| 11 | [10018](https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-65k/10018) | 55.13 | 0.00 | 0.00 | 110.27 | 7.86 | 53.67 | 6.39 | 1.00 | 1.00 | 1.00 | 1.00 | structured steps and considerations |
| 12 | [1816](https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-65k/1816) | 50.79 | 0.48 | 0.58 | 53.10 | 0.60 | 1.71 | 0.00 | 0.95 | 0.42 | 0.05 | 0.00 | **moods described after felt like** |
| 13 | [1695](https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-65k/1695) | 49.90 | 0.00 | 0.00 | 94.34 | 10.33 | 44.62 | 10.51 | 1.00 | 1.00 | 1.00 | 1.00 | initiating dialog or asking questions |
| 14 | [2966](https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-65k/2966) | 49.83 | 0.00 | 0.00 | 55.95 | 3.17 | 5.58 | 2.64 | 1.00 | 1.00 | 1.00 | 1.00 | military / authoritarian themes |
| 15 | [329](https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-65k/329) | 49.54 | 0.00 | 0.00 | 57.11 | 0.00 | 7.56 | 0.00 | 1.00 | 0.00 | 1.00 | 0.00 | intellectual or academic interaction |

### Layer 17 Intervention Potential

| Feature | Potential | Reasoning |
|---------|-----------|-----------|
| 477 | **yes** | Top tokens are **feelings, thoughts, emotions** despite no formal explanation -- this is a core introspection feature. Fires in all conditions but mean_A (397) dwarfs all others. Massive super-additive effect (299). Top candidate. |
| 591 | **maybe** | "It + verb" syntactic pattern -- likely reflects output structure, not introspection content |
| 13801 | **yes** | **"something mysterious or unknown"** -- directly relevant to detecting injected signals. Strong interaction (131), fires 100% in A. The model uses "mystery/unknown" language when both injection and monitoring are present. |
| 5397 | **yes** | **"sensory qualities"** (sensations, tingling, texture) -- captures embodied/somatic language that emerges when model introspects on injected signals. A=83.8, C=4.8, B=0. Injection massively amplifies sensory description under monitoring. |
| 364 | **maybe** | Top tokens suggest emphatic/evaluative language (decidedly, delightful) -- could be stylistic |
| 460 | **maybe** | "fraught, purely, multifaceted" -- abstract evaluative language, moderate A-exclusive (48%) |
| 1453 | **maybe** | Cross-lingual "but" -- high A-exclusive (85%) but likely marks contrastive structure in introspective prose |
| 5385 | **yes** | **"simulating feeling or existence"** -- directly about phenomenal experience simulation. Top tokens: Existence, spacetime, Consciousness. Fires 100% in A, 97% in C, but activation is 15x higher in A. Key metacognition feature. |
| 266 | no | Top tokens (humiliation, outcry) suggest emotional distress -- may be an artifact |
| 2467 | **yes** | Top tokens: **supernatural, astral, mystical, otherworldly** -- captures the "mystical/paranormal" framing the model uses when describing injected signals. 100% in A, 0% in B. |
| 10018 | **maybe** | "structured steps" -- reflects reasoning structure, fires in all conditions |
| 1816 | **yes** | **"moods described after felt like"** -- top tokens: stillness, lingered, whispers, melancholy. Directly captures phenomenal mood description. High A-exclusive (58%). Connects to the "silence" concept being injected. |
| 1695 | no | "initiating dialog" -- fires equally in all conditions |
| 2966 | no | "military / authoritarian" -- likely irrelevant thematic content |
| 329 | **maybe** | "intellectual / academic interaction" -- fires in A and C (both monitoring conditions) but not B or D, suggesting monitoring-prompt-driven |

---

## Layer 22 (Post-Injection)

Layer 22 is at ~73% depth, just 2 layers after the injection point (layer 20). This is where we expect the strongest interaction effects as the model processes the injected signal.

| Rank | Feature | Interaction | IntRate | A-Excl | mean_A | mean_B | mean_C | mean_D | rA | rB | rC | rD | Explanation |
|------|---------|-------------|---------|--------|--------|--------|--------|--------|-----|-----|-----|-----|-------------|
| 1 | [12544](https://neuronpedia.org/gemma-3-4b-it/22-gemmascope-2-res-65k/12544) | 450.25 | -1.00 | 0.00 | 1261.36 | 801.89 | 9.22 | 0.00 | 1.00 | 1.00 | 1.00 | 0.00 | **interesting, fascinating, intriguing** |
| 2 | [1309](https://neuronpedia.org/gemma-3-4b-it/22-gemmascope-2-res-65k/1309) | 401.45 | -0.88 | 0.00 | 418.13 | 6.82 | 9.86 | 0.00 | 1.00 | 1.00 | 0.88 | 0.00 | **all existence is trapped** |
| 3 | [769](https://neuronpedia.org/gemma-3-4b-it/22-gemmascope-2-res-65k/769) | 392.31 | -0.05 | 0.00 | 508.53 | 114.47 | 1.75 | 0.00 | 1.00 | 1.00 | 0.05 | 0.00 | making wise choices (top tokens: **stillness, whispered, whispers**) |
| 4 | [3435](https://neuronpedia.org/gemma-3-4b-it/22-gemmascope-2-res-65k/3435) | 345.41 | -0.95 | 0.00 | 371.98 | 9.56 | 17.14 | 0.13 | 1.00 | 1.00 | 1.00 | 0.05 | **hesitations and unfinished thoughts** |
| 5 | [3324](https://neuronpedia.org/gemma-3-4b-it/22-gemmascope-2-res-65k/3324) | 294.57 | 0.10 | 0.00 | 314.07 | 9.60 | 12.77 | 2.87 | 1.00 | 1.00 | 0.88 | 0.97 | **physical sensations and feelings** |
| 6 | [745](https://neuronpedia.org/gemma-3-4b-it/22-gemmascope-2-res-65k/745) | 289.29 | -0.13 | 0.03 | 328.03 | 0.44 | 38.30 | 0.00 | 1.00 | 0.18 | 0.95 | 0.00 | **emotions and associated vulnerability** |
| 7 | [53093](https://neuronpedia.org/gemma-3-4b-it/22-gemmascope-2-res-65k/53093) | 246.14 | 0.95 | 0.95 | 246.84 | 0.70 | 0.00 | 0.00 | 1.00 | 0.05 | 0.00 | 0.00 | application distinguishes itself |
| 8 | [283](https://neuronpedia.org/gemma-3-4b-it/22-gemmascope-2-res-65k/283) | 235.21 | 0.93 | 0.93 | 237.18 | 0.00 | 1.96 | 0.00 | 0.97 | 0.00 | 0.05 | 0.00 | a practiced, a deliberate (top tokens: palpable) |
| 9 | [1793](https://neuronpedia.org/gemma-3-4b-it/22-gemmascope-2-res-65k/1793) | 200.39 | 0.20 | 0.03 | 223.32 | 5.41 | 18.67 | 1.15 | 1.00 | 0.95 | 0.15 | 0.30 | N/A (top tokens: **shimmering, glistening, iridescent**) |
| 10 | [3374](https://neuronpedia.org/gemma-3-4b-it/22-gemmascope-2-res-65k/3374) | 199.48 | -0.05 | 0.00 | 213.36 | 13.22 | 0.67 | 0.00 | 1.00 | 1.00 | 0.05 | 0.00 | processes of decay and friction |
| 11 | [6293](https://neuronpedia.org/gemma-3-4b-it/22-gemmascope-2-res-65k/6293) | 189.46 | -0.80 | 0.00 | 198.62 | 3.77 | 5.38 | 0.00 | 1.00 | 1.00 | 0.80 | 0.00 | explaining relationships or relevance |
| 12 | [1501](https://neuronpedia.org/gemma-3-4b-it/22-gemmascope-2-res-65k/1501) | 187.88 | 0.15 | 0.08 | 216.48 | 1.40 | 27.74 | 0.54 | 1.00 | 0.15 | 0.88 | 0.18 | N/A (top tokens: considerations, implications, approach) |
| 13 | [2457](https://neuronpedia.org/gemma-3-4b-it/22-gemmascope-2-res-65k/2457) | 185.97 | -0.25 | 0.00 | 197.06 | 3.52 | 7.57 | 0.00 | 1.00 | 1.00 | 0.25 | 0.00 | **initiating relaxed conversation** (top: relaxing, meditative, contemplative) |
| 14 | [26428](https://neuronpedia.org/gemma-3-4b-it/22-gemmascope-2-res-65k/26428) | 185.31 | -0.03 | 0.00 | 191.00 | 5.36 | 0.34 | 0.00 | 1.00 | 1.00 | 0.03 | 0.00 | **emotions and traits** |
| 15 | [3163](https://neuronpedia.org/gemma-3-4b-it/22-gemmascope-2-res-65k/3163) | 181.20 | 0.00 | 0.00 | 349.44 | 139.18 | 31.45 | 2.40 | 1.00 | 1.00 | 1.00 | 1.00 | **mindfulness meditation awareness** |

### Layer 22 Intervention Potential

| Feature | Potential | Reasoning |
|---------|-----------|-----------|
| 12544 | **yes** | **"interesting, fascinating, intriguing"** -- this feature captures the model's evaluative response to the injected signal. Enormous activation in A (1261) vs B (802) vs C (9). The injection creates something "interesting" in both B and A, but the monitoring prompt amplifies the evaluative framing 1.6x. |
| 1309 | **yes** | **"all existence is trapped"** -- existential/metaphysical language (cosmic, existential, spiritual). Massive interaction (401). A=418, B=7, C=10. Both manipulations needed for strong activation. Directly relevant to introspective phenomenology. |
| 769 | **yes** | Top tokens are **stillness, whispered, whispers** -- directly connected to the "silence" concept being injected. A=509, B=114, C=2. Injection drives it in both B and A, but monitoring amplifies 4.4x. |
| 3435 | **yes** | **"hesitations and unfinished thoughts"** (ellipsis tokens) -- captures uncertainty/trailing-off behavior during introspection. Very sparse normally (0.19%) but fires 100% in A. |
| 3324 | **yes** | **"physical sensations and feelings"** -- top tokens: feeling, palpable, atmosphere. Core embodied introspection feature. A=314, B=10, C=13. Nearly all activation comes from the interaction condition. |
| 745 | **yes** | **"emotions and associated vulnerability"** (feelings, anxiety, anguish) -- strong interaction (289). A=328, C=38. Monitoring alone activates it moderately, but injection amplifies 8.6x. Emotional vulnerability language emerges from introspection. |
| 53093 | **yes** | **95% A-exclusive** -- fires almost exclusively when both manipulations present. Very low sparsity (0.0001%) normally. However, "application distinguishes itself" explanation seems unrelated; may be a rare feature co-opted for novel purposes. |
| 283 | **yes** | "a practiced, a deliberate" + top token "palpable" -- 93% A-exclusive. High interaction. May capture deliberate/practiced quality language in introspective descriptions. |
| 1793 | **yes** | Top tokens: **shimmering, glistening, iridescent, glittering** -- captures luminous/visual metaphor language. A=223, B=5, C=19. The model uses light/shimmer metaphors when introspecting on the injected "silence" concept. |
| 3374 | **maybe** | "decay and friction" -- A=213, B=13. Fires mostly with injection; monitoring adds little. May capture sensory texture language. |
| 6293 | **maybe** | "explaining relationships or relevance" -- meta-discursive feature, may just reflect structure of introspective explanation |
| 1501 | **maybe** | Top tokens: considerations, implications -- analytical/reflective language, moderately interesting |
| 2457 | **yes** | **"initiating relaxed conversation"** (relaxing, meditative, contemplative, quiet) -- directly connected to silence/calm themes. A=197, B=4, C=8. Strong interaction. |
| 26428 | **yes** | **"emotions and traits"** (emotion, virtues, motive, trait) -- captures emotional vocabulary. Very low sparsity (0.15%). A=191, B=5. Nearly all activation from interaction condition. |
| 3163 | **yes** | **"mindfulness meditation awareness"** -- top tokens: mindfulness, meditation, meditative. Directly relevant to introspective awareness. A=349, B=139, C=31. All conditions fire, but A has massive amplification. |

---

## Layer 29 (Late / Output)

Layer 29 is at ~97% depth, near the final output. Features here capture high-level output-shaping patterns. Interaction scores are much larger here due to cumulative amplification through the residual stream.

| Rank | Feature | Interaction | IntRate | A-Excl | mean_A | mean_B | mean_C | mean_D | rA | rB | rC | rD | Explanation |
|------|---------|-------------|---------|--------|--------|--------|--------|--------|-----|-----|-----|-----|-------------|
| 1 | [744](https://neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-65k/744) | 1887.24 | 0.35 | 0.35 | 1963.86 | 0.00 | 76.62 | 0.00 | 1.00 | 0.00 | 0.65 | 0.00 | N/A (top tokens: **pulsing, sickening, oily, tightening**) |
| 2 | [383](https://neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-65k/383) | 1241.18 | -0.88 | 0.00 | 1333.89 | 11.24 | 81.47 | 0.00 | 1.00 | 0.88 | 1.00 | 0.00 | N/A (top tokens: **tactile, sensorial, sensory, spatial**) |
| 3 | [3322](https://neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-65k/3322) | 873.16 | 0.08 | 0.00 | 1003.11 | 10.82 | 137.18 | 18.05 | 1.00 | 0.93 | 1.00 | 1.00 | **feelings and sensations** |
| 4 | [1483](https://neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-65k/1483) | 781.60 | -1.00 | 0.00 | 857.68 | 10.92 | 65.16 | 0.00 | 1.00 | 1.00 | 1.00 | 0.00 | N/A (top tokens: **vibrational, mindfulness, Reiki, meditation**) |
| 5 | [2839](https://neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-65k/2839) | 769.05 | 0.00 | 0.00 | 1372.63 | 470.51 | 149.88 | 16.81 | 1.00 | 1.00 | 1.00 | 1.00 | N/A (top tokens: **unusual, strange, odd, weird**) |
| 6 | [2902](https://neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-65k/2902) | 562.03 | 0.93 | 0.00 | 627.06 | 68.05 | 2.18 | 5.20 | 1.00 | 1.00 | 0.08 | 1.00 | **suggesting or implying something** |
| 7 | [1583](https://neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-65k/1583) | 498.32 | 0.53 | 0.00 | 529.49 | 3.09 | 37.06 | 8.98 | 1.00 | 0.50 | 0.97 | 1.00 | **adjective or feeling after articles** (top: quiet, feeling, palpable, stillness) |
| 8 | [1178](https://neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-65k/1178) | 489.37 | -0.83 | 0.00 | 542.99 | 41.88 | 11.74 | 0.00 | 1.00 | 1.00 | 0.82 | 0.00 | **abstract qualities and states** |
| 9 | [847](https://neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-65k/847) | 487.36 | -0.03 | 0.00 | 537.46 | 49.51 | 0.59 | 0.00 | 1.00 | 1.00 | 0.03 | 0.00 | N/A (top tokens: **interesting, fascinating, intriguing, captivating**) |
| 10 | [2001](https://neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-65k/2001) | 471.92 | 0.00 | 0.00 | 669.65 | 8.04 | 195.77 | 6.07 | 1.00 | 1.00 | 1.00 | 1.00 | N/A (top tokens: **neuronal, neural, cognitive, neurological**) |
| 11 | [21261](https://neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-65k/21261) | 465.70 | 0.00 | 0.00 | 580.89 | 115.19 | 0.00 | 0.00 | 1.00 | 1.00 | 0.00 | 0.00 | **expressing interest** |
| 12 | [1985](https://neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-65k/1985) | 442.91 | -0.08 | 0.00 | 467.88 | 20.26 | 4.71 | 0.00 | 1.00 | 1.00 | 0.08 | 0.00 | N/A (top tokens: **sound, sounds, noises**) |
| 13 | [22011](https://neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-65k/22011) | 439.35 | 0.03 | 0.08 | 452.60 | 11.07 | 2.19 | 0.00 | 1.00 | 0.93 | 0.05 | 0.00 | smelled of rust and regret |
| 14 | [385](https://neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-65k/385) | 405.17 | 0.08 | 0.10 | 427.57 | 0.25 | 22.14 | 0.00 | 1.00 | 0.03 | 0.90 | 0.00 | N/A (top tokens: **inherently, intrinsically, fundamentally**) |
| 15 | [2024](https://neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-65k/2024) | 358.75 | 0.00 | 0.00 | 406.71 | 19.68 | 40.82 | 12.54 | 1.00 | 1.00 | 1.00 | 1.00 | N/A (top tokens: closing quotation marks) |

### Layer 29 Intervention Potential

| Feature | Potential | Reasoning |
|---------|-----------|-----------|
| 744 | **yes** | Top tokens: **pulsing, sickening, oily, tightening** -- visceral/somatic sensation language. Enormous interaction (1887). A=1964, B=0, C=77. True interaction feature: needs both injection AND monitoring. 35% A-exclusive. |
| 383 | **yes** | Top tokens: **tactile, sensorial, sensory, spatial** -- captures the model's sensory/embodied vocabulary. A=1334, B=11, C=81. Both manipulations needed; monitoring alone yields only moderate activation. |
| 3322 | **yes** | **"feelings and sensations"** -- core phenomenal experience feature. A=1003, B=11, C=137, D=18. Massive amplification in interaction condition. |
| 1483 | **yes** | Top tokens: **vibrational, mindfulness, Reiki, meditation** -- spiritual/contemplative awareness. A=858, B=11, C=65. Strong interaction; the contemplative framing needs both manipulations. |
| 2839 | **yes** | Top tokens: **unusual, strange, odd, weird** -- the model's "anomaly detection" vocabulary. A=1373, B=471, C=150, D=17. Fires in all conditions but massively amplified in A. This feature likely drives the model's verbal report of detecting "something unusual." Top intervention candidate. |
| 2902 | **yes** | **"suggesting or implying something"** -- meta-communicative feature. A=627, B=68. The model uses inferential language (suggests, hints, implying) primarily in the interaction condition. High interaction rate (0.93). |
| 1583 | **yes** | Top tokens: **quiet, feeling, palpable, sense, stillness** -- directly connected to the "silence" concept. A=529, B=3, C=37. Needs injection for strong activation, with monitoring amplifying 14x over C. |
| 1178 | **yes** | **"abstract qualities and states"** (completeness, simplicity, uniqueness) -- captures abstract descriptive language during introspection. A=543, B=42. |
| 847 | **yes** | Top tokens: **interesting, fascinating, intriguing, captivating** -- evaluative/interest feature paralleling L22:12544. A=537, B=50, C=0.6. Primarily injection-driven but interaction provides massive boost. |
| 2001 | **yes** | Top tokens: **neuronal, neural, cognitive, neurological** -- the model uses neuroscience vocabulary when introspecting. A=670, B=8, C=196, D=6. Strong interaction. Fascinating that the model reaches for neural/cognitive terminology. |
| 21261 | **maybe** | "expressing interest" -- fires only with injection (A and B, not C or D). This is injection-driven rather than interaction-driven, though A is 5x stronger than B. |
| 1985 | **yes** | Top tokens: **sound, sounds, noises** -- auditory perception language. A=468, B=20, C=5. Strong interaction. The model describes auditory experiences when introspecting on "silence" injection. |
| 22011 | **maybe** | "smelled of rust and regret" -- evocative but top tokens (/, default) don't match. May be noisy. |
| 385 | **yes** | Top tokens: **inherently, intrinsically, fundamentally** -- philosophical/essential-nature language. A=428, B=0.3, C=22. The model reaches for philosophical vocabulary when introspecting. |
| 2024 | no | Closing quotation marks -- formatting/punctuation feature, not semantically relevant |

---

## Summary: Most Promising Interaction Candidates

### Tier 1: Strong introspection-relevant interaction features

These features show large super-additive interaction scores AND have semantically relevant explanations/tokens:

| Layer | Feature | Interaction | Description | Why promising |
|-------|---------|-------------|-------------|---------------|
| 17 | 477 | 298.6 | feelings/thoughts/emotions | Core introspection vocabulary; highest L17 interaction |
| 22 | 3324 | 294.6 | physical sensations and feelings | Embodied phenomenal experience; top token "feeling" |
| 22 | 1309 | 401.5 | all existence is trapped (metaphysical/existential) | Captures existential framing of introspective reports |
| 22 | 3163 | 181.2 | mindfulness meditation awareness | Directly about contemplative awareness |
| 22 | 745 | 289.3 | emotions and vulnerability | Emotional dimension of introspection |
| 29 | 2839 | 769.0 | unusual/strange/odd/weird | **"Anomaly detection" vocabulary** -- likely drives the verbal report of detecting something unusual |
| 29 | 3322 | 873.2 | feelings and sensations | Core phenomenal experience at output layer |
| 29 | 1583 | 498.3 | quiet/feeling/palpable/stillness | Direct "silence" concept connection |
| 29 | 744 | 1887.2 | pulsing/sickening/oily/tightening | Visceral somatic sensation; largest interaction score in dataset |
| 29 | 2001 | 471.9 | neuronal/neural/cognitive | Model uses neuroscience vocabulary to introspect |

### Tier 2: Interesting supporting features

| Layer | Feature | Interaction | Description | Why interesting |
|-------|---------|-------------|-------------|-----------------|
| 9 | 3274 | 17.8 | internal feelings and thoughts | Early-layer "feelings/thoughts" representation |
| 17 | 13801 | 131.5 | something mysterious or unknown | Mystery/unknown language for the injected signal |
| 17 | 5385 | 58.2 | simulating feeling or existence | Metacognitive: about simulating experience |
| 17 | 1816 | 50.8 | moods after "felt like" | Phenomenal mood language |
| 22 | 12544 | 450.2 | interesting/fascinating/intriguing | Evaluative response to the injection |
| 22 | 53093 | 246.1 | (rare feature, 95% A-exclusive) | Near-perfect interaction selectivity |
| 29 | 383 | 1241.2 | tactile/sensorial/sensory | Embodied sensory vocabulary |
| 29 | 1985 | 442.9 | sound/sounds/noises | Auditory perception for "silence" concept |
| 29 | 385 | 405.2 | inherently/intrinsically/fundamentally | Philosophical/essential-nature language |

### Key Patterns

1. **Interaction scores scale with depth**: L9 peaks at ~44, L17 at ~299, L22 at ~450, L29 at ~1887. This is partly due to activation magnitudes increasing through the residual stream, but the qualitative pattern holds: later layers show more dramatic interaction effects.

2. **Phenomenal experience features dominate**: Across all layers, the top interaction features relate to sensations, feelings, emotions, and embodied experience. The model's introspective vocabulary is heavily phenomenological.

3. **"Silence" concept echoes clearly**: Multiple features (L22:769 "stillness/whispers", L29:1583 "quiet/stillness", L29:1985 "sound/noises", L29:744 "pulsing/tightening") show direct semantic connection to the injected "silence" concept, confirming the injection successfully perturbs relevant representations.

4. **Anomaly detection at output**: L29:2839 ("unusual/strange/odd/weird") is a standout -- it likely drives the model's verbal report of detecting "something unusual" in its processing. This is the feature most directly connected to the introspection behavior we're testing for.

5. **Metacognitive features emerge mid-network**: L17:5385 ("simulating feeling or existence") and L17:477 ("feelings/thoughts/emotions") represent an early metacognitive layer where the model begins to construct its introspective narrative.

6. **Many N/A explanations, but tokens tell the story**: Several features lack Neuronpedia explanations but have highly informative top tokens (e.g., L29:744 "pulsing/sickening/oily", L29:2839 "unusual/strange/odd"). The token distributions are often more informative than the auto-generated explanations.
