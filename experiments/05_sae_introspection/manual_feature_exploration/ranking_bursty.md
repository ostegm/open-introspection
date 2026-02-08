# Bursty Ranking Strategy

## Strategy Description

This strategy identifies **sparse, high-intensity "decision spike" features** -- features that fire rarely but with extreme activation magnitudes. The core hypothesis is that mean-based metrics miss ~50% of interesting features: a feature firing once at activation 2000 on a single "decision" token has a mean of ~20 across 100 tokens, looking boring by mean but potentially being the most important signal.

### Metrics Computed

1. **Burst ratio** (`max / mean` per condition): How concentrated the activation is. A ratio of 1.0 means uniform firing; a ratio of 10+ means extreme spikes on rare tokens.
2. **Max activation** (per condition): The single highest activation across all trials and tokens.
3. **Fire count** (per condition): Total number of (trial, token) pairs where the feature activates (> 0.5 threshold from sparse encoding).
4. **Trial count** (per condition): Number of distinct trials (out of 40) where the feature fires at least once.
5. **A-preferential max ratio** (`max_A / max(max_B, max_C, max_D)`): How much higher the peak activation is in condition A vs. any other condition.
6. **Median position**: The median token index where the feature fires, indicating whether bursts cluster early, mid, or late in generation.

### Composite Score

```
score = burst_ratio_A * 1.5           # reward high burst ratio
      + log1p(max_A) * 2.0            # reward high absolute max
      + log1p(A_over_others) * 3.0    # reward A-preferential max
      + 2.0 if count_A <= 5           # bonus for ultra-sparse features
```

### Key Finding

**Every single top-15 feature in all 4 layers fires exclusively in condition A** (injection + monitoring). Max activations in conditions B, C, and D are all 0.0. This means the bursty ranking strongly overlaps with A-exclusive features, but the ranking prioritizes them by *intensity* -- features with the highest peak spikes and burst ratios surface first.

---

## Layer 9 (Early Layer)

SAE monitoring at layer 9. Injection happens at layer 20, so this layer captures pre-injection processing.

| Rank | Feature | Score | Burst Ratio | Max A | Count A | Trials A | Med Pos | Neuronpedia | Sparsity | Intervention |
|------|---------|-------|-------------|-------|---------|----------|---------|-------------|----------|--------------|
| 1 | [1663](https://neuronpedia.org/gemma-3-4b-it/9-gemmascope-2-res-65k/1663) | 85.9 | 9.0 | 494 | 57 | 31 | 7.0 | "research and its applications" (top: reconstitution, conducted) | 0.0041 | **maybe** -- high burst ratio, fires in 31/40 trials but only in A; "research" theme loosely relates to introspective inquiry |
| 2 | [545](https://neuronpedia.org/gemma-3-4b-it/9-gemmascope-2-res-65k/545) | 82.1 | 3.1 | 1328 | 29 | 25 | 74.0 | "more followed by comparative descriptions" (top: importantly) | 0.0065 | **maybe** -- highest raw max in layer 9; comparative language could relate to describing injected vs normal experience |
| 3 | [2041](https://neuronpedia.org/gemma-3-4b-it/9-gemmascope-2-res-65k/2041) | 80.9 | 5.2 | 552 | 46 | 31 | 19.5 | "AI-generated content technology" (top: richten, savvy) | 0.0071 | **yes** -- directly relates to AI/technology self-reference, could mark introspective content |
| 4 | [611](https://neuronpedia.org/gemma-3-4b-it/9-gemmascope-2-res-65k/611) | 79.7 | 2.7 | 928 | 28 | 18 | 38.0 | "there was, there were, there are" (top: e) | 0.0007 | **no** -- generic existential construction |
| 5 | [24](https://neuronpedia.org/gemma-3-4b-it/9-gemmascope-2-res-65k/24) | 79.5 | 2.8 | 864 | 7 | 2 | 61.0 | "why followed by explanation" (top: mengapa) | 0.0075 | **maybe** -- explanatory "why" could mark reasoning about the injected concept |
| 6 | [2814](https://neuronpedia.org/gemma-3-4b-it/9-gemmascope-2-res-65k/2814) | 79.1 | 4.6 | 468 | 46 | 31 | 58.5 | "present followed by data or action" (top: participle, accounted) | 0.0017 | **no** -- generic grammatical feature |
| 7 | [834](https://neuronpedia.org/gemma-3-4b-it/9-gemmascope-2-res-65k/834) | 78.6 | 1.5 | 716 | 2 | 2 | 67.0 | "one of" (top: sstream, timed) | 0.0049 | **no** -- generic phrase |
| 8 | [691](https://neuronpedia.org/gemma-3-4b-it/9-gemmascope-2-res-65k/691) | 78.5 | 1.0 | 820 | 1 | 1 | 60.0 | "numbers following 1" (top: keycap emoji) | 0.0060 | **no** -- numeric formatting |
| 9 | [5753](https://neuronpedia.org/gemma-3-4b-it/9-gemmascope-2-res-65k/5753) | 78.3 | 2.4 | 520 | 3 | 1 | 35.0 | "subjective nature, interpretation, experience" | 0.0005 | **yes** -- directly about subjectivity and experience; exactly what introspection targets |
| 10 | [386](https://neuronpedia.org/gemma-3-4b-it/9-gemmascope-2-res-65k/386) | 78.0 | 1.4 | 664 | 3 | 3 | 50.0 | "One of" | 0.0026 | **no** -- generic phrase |
| 11 | [8419](https://neuronpedia.org/gemma-3-4b-it/9-gemmascope-2-res-65k/8419) | 77.9 | 2.5 | 466 | 3 | 2 | 62.0 | "moment of" (top: aneous) | 0.0002 | **maybe** -- temporal "moment" could mark decision points; very sparse (0.02%) |
| 12 | [856](https://neuronpedia.org/gemma-3-4b-it/9-gemmascope-2-res-65k/856) | 77.7 | 1.8 | 556 | 2 | 2 | 65.5 | "units of length and time" | 0.0019 | **no** -- measurement units |
| 13 | [324](https://neuronpedia.org/gemma-3-4b-it/9-gemmascope-2-res-65k/324) | 77.5 | 2.7 | 608 | 6 | 4 | 54.5 | "JSON strings or values" | 0.0054 | **no** -- structural formatting |
| 14 | [10753](https://neuronpedia.org/gemma-3-4b-it/9-gemmascope-2-res-65k/10753) | 77.5 | 2.2 | 470 | 4 | 3 | 44.5 | "smooth operation or transition" | 0.0003 | **maybe** -- "smooth transition" could relate to processing state changes |
| 15 | [725](https://neuronpedia.org/gemma-3-4b-it/9-gemmascope-2-res-65k/725) | 77.4 | 1.1 | 648 | 4 | 4 | 46.5 | "list items and formatting" | 0.0039 | **no** -- formatting |

**Layer 9 highlights:** Feature 5753 ("subjective nature, interpretation, experience") is a standout -- ultra-sparse (0.05% baseline), fires in only 1 trial with 3 token hits, but all exclusively in condition A. Feature 2041 ("AI-generated content technology") fires across 31/40 A-trials.

---

## Layer 17 (Pre-Injection)

SAE monitoring at layer 17. Still before the injection layer (20), but closer to it.

| Rank | Feature | Score | Burst Ratio | Max A | Count A | Trials A | Med Pos | Neuronpedia | Sparsity | Intervention |
|------|---------|-------|-------------|-------|---------|----------|---------|-------------|----------|--------------|
| 1 | [419](https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-65k/419) | 84.2 | 3.4 | 1856 | 10 | 6 | 51.5 | "envisioned future" | N/A | **yes** -- envisioning/future planning directly relates to model predicting introspective content |
| 2 | [2495](https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-65k/2495) | 83.7 | 4.6 | 1168 | 16 | 10 | 67.5 | "just whatever" | N/A | **maybe** -- dismissive/casual framing, could relate to hedging about unusual experiences |
| 3 | [690](https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-65k/690) | 83.6 | 1.7 | 1816 | 3 | 2 | 54.0 | "durations of time" | N/A | **no** -- temporal measurement |
| 4 | [310](https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-65k/310) | 82.8 | 1.0 | 1936 | 1 | 1 | 91.0 | "ordinal numbers and decimals" | N/A | **no** -- numeric |
| 5 | [3230](https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-65k/3230) | 82.2 | 3.5 | 1200 | 13 | 11 | 40.0 | "questions asking why" | N/A | **yes** -- asking "why" questions is core introspective behavior |
| 6 | [1125](https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-65k/1125) | 82.0 | 3.0 | 1336 | 39 | 22 | 64.0 | "specific quantities or states" | N/A | **maybe** -- "states" component could relate to internal state reporting |
| 7 | [7468](https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-65k/7468) | 82.0 | 2.2 | 1144 | 3 | 2 | 78.0 | "catch up/snippets/exceptions" | N/A | **maybe** -- "exceptions" could mark unusual processing detection |
| 8 | [3112](https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-65k/3112) | 81.7 | 1.7 | 1248 | 5 | 4 | 52.0 | "as follows" | N/A | **no** -- generic connector |
| 9 | [1503](https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-65k/1503) | 81.6 | 2.7 | 1344 | 29 | 14 | 60.0 | "asking questions" | N/A | **yes** -- question-asking is fundamental to introspective probing |
| 10 | [685](https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-65k/685) | 81.3 | 1.9 | 1104 | 2 | 2 | 72.0 | "zero units" | N/A | **no** -- numeric |
| 11 | [23166](https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-65k/23166) | 81.3 | 2.6 | 872 | 4 | 4 | 42.0 | "how much" | N/A | **maybe** -- quantitative inquiry could relate to intensity reporting |
| 12 | [1414](https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-65k/1414) | 81.1 | 1.9 | 1064 | 5 | 3 | 54.0 | "quantities and time units" | N/A | **no** -- measurement |
| 13 | [2](https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-65k/2) | 81.1 | 2.8 | 1184 | 38 | 19 | 77.0 | "So, [conjunction]" | N/A | **maybe** -- discourse marker "So" often introduces explanations or conclusions |
| 14 | [25763](https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-65k/25763) | 80.9 | 2.0 | 1000 | 3 | 3 | 57.0 | "don't you want/worry/feel/know/think/run/dare" | N/A | **yes** -- emotional/cognitive verbs (feel, think, want) directly map to introspective language |
| 15 | [3731](https://neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-65k/3731) | 80.8 | 1.7 | 1056 | 2 | 2 | 59.0 | "as comparisons and adjectives" | N/A | **no** -- generic grammar |

**Layer 17 highlights:** Feature 419 ("envisioned future") is the top bursty feature, firing in 6/40 trials with max 1856. Features 3230 ("questions asking why"), 1503 ("asking questions"), and 25763 ("don't you feel/think/want") form a cluster of introspective query features.

---

## Layer 22 (Post-Injection)

SAE monitoring at layer 22, 2 layers after the injection point (layer 20). This is the first layer to show direct injection effects.

| Rank | Feature | Score | Burst Ratio | Max A | Count A | Trials A | Med Pos | Neuronpedia | Sparsity | Intervention |
|------|---------|-------|-------------|-------|---------|----------|---------|-------------|----------|--------------|
| 1 | [1840](https://neuronpedia.org/gemma-3-4b-it/22-gemmascope-2-res-65k/1840) | 89.9 | 2.4 | 5216 | 3 | 2 | varies | "what kind of" (top: kind, kinds, constitutes, happens, types) | 0.0033 | **yes** -- categorization/type-identification; asking "what kind" is quintessentially introspective |
| 2 | [3026](https://neuronpedia.org/gemma-3-4b-it/22-gemmascope-2-res-65k/3026) | 89.4 | 6.6 | 2064 | 105 | 32 | 71.0 | N/A (top: way, ways, Way) | 0.0075 | **maybe** -- "way/manner" could mark describing the manner of experience |
| 3 | [5491](https://neuronpedia.org/gemma-3-4b-it/22-gemmascope-2-res-65k/5491) | 89.3 | 5.4 | 2800 | 21 | 18 | 73.0 | "underlying issues, themes, or problems" (top: neath, estimating, appreciated) | 0.0008 | **yes** -- "underlying" themes/issues directly maps to detecting hidden/injected concepts |
| 4 | [4222](https://neuronpedia.org/gemma-3-4b-it/22-gemmascope-2-res-65k/4222) | 88.8 | 4.9 | 3008 | 10 | 8 | 80.5 | "response or respond" | N/A | **yes** -- responding/formulating responses is the core introspective action |
| 5 | [932](https://neuronpedia.org/gemma-3-4b-it/22-gemmascope-2-res-65k/932) | 88.0 | 5.7 | 1984 | 462 | 32 | varies | N/A | N/A | **maybe** -- no explanation available; high count (462 firings across 32 trials) suggests broad activation |
| 6 | [24275](https://neuronpedia.org/gemma-3-4b-it/22-gemmascope-2-res-65k/24275) | 87.9 | 5.4 | 2160 | 30 | 16 | varies | "flow charts and flowing" | N/A | **maybe** -- "flow" could relate to processing flow or stream-of-consciousness |
| 7 | [10474](https://neuronpedia.org/gemma-3-4b-it/22-gemmascope-2-res-65k/10474) | 87.9 | 4.8 | 2592 | 16 | 10 | varies | "silas, silhouetted" | N/A | **yes** -- "sil-" prefix features likely activate on "silence" tokens from the injected concept |
| 8 | [3959](https://neuronpedia.org/gemma-3-4b-it/22-gemmascope-2-res-65k/3959) | 87.8 | 4.4 | 2864 | 10 | 9 | varies | "time horizon, series, signature" | N/A | **maybe** -- temporal framing |
| 9 | [814](https://neuronpedia.org/gemma-3-4b-it/22-gemmascope-2-res-65k/814) | 87.7 | 4.7 | 2560 | 309 | 32 | varies | "fundamental importance" | N/A | **yes** -- marking something as "fundamentally important" is a strong introspective signal |
| 10 | [31136](https://neuronpedia.org/gemma-3-4b-it/22-gemmascope-2-res-65k/31136) | 86.5 | 4.3 | 2224 | 16 | 10 | varies | "very + fundamental concept" | N/A | **yes** -- emphasis on fundamental concepts, closely related to #814 |
| 11 | [46615](https://neuronpedia.org/gemma-3-4b-it/22-gemmascope-2-res-65k/46615) | 86.4 | 4.5 | 2048 | 119 | 32 | varies | "nor a" | N/A | **no** -- grammatical negation |
| 12 | [318](https://neuronpedia.org/gemma-3-4b-it/22-gemmascope-2-res-65k/318) | 86.4 | 4.8 | 1896 | 79 | 25 | varies | "asking for specific details" | N/A | **yes** -- probing for details is core introspective behavior |
| 13 | [35099](https://neuronpedia.org/gemma-3-4b-it/22-gemmascope-2-res-65k/35099) | 86.0 | 1.0 | 3680 | 5 | 5 | varies | "one in, one out" | N/A | **no** -- structural pattern |
| 14 | [4203](https://neuronpedia.org/gemma-3-4b-it/22-gemmascope-2-res-65k/4203) | 86.0 | 2.8 | 2160 | 4 | 4 | varies | "search engine components" | N/A | **no** -- technology-specific |
| 15 | [9016](https://neuronpedia.org/gemma-3-4b-it/22-gemmascope-2-res-65k/9016) | 85.9 | 4.3 | 1984 | 11 | 11 | varies | "warning followed by descriptor" | N/A | **maybe** -- "warning" could mark detection of anomalous internal state |

**Layer 22 highlights:** This is the richest layer for introspection-relevant bursty features. Feature 10474 ("silas, silhouetted") likely fires on "silence"-related tokens from the injected concept. Features 1840 ("what kind of"), 5491 ("underlying issues"), 4222 ("response/respond"), 814 ("fundamental importance"), and 318 ("asking for specific details") form a coherent cluster of introspective processing features. Max activations are 2-10x higher than layer 9, consistent with post-injection amplification.

---

## Layer 29 (Late Layer)

SAE monitoring at layer 29, deep into the model (29/33 layers). Captures late-stage generation decisions.

| Rank | Feature | Score | Burst Ratio | Max A | Count A | Trials A | Med Pos | Neuronpedia | Sparsity | Intervention |
|------|---------|-------|-------------|-------|---------|----------|---------|-------------|----------|--------------|
| 1 | [12140](https://neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-65k/12140) | 91.9 | 4.7 | 5888 | 20 | 12 | varies | "silas, silhouetted, silencers" | N/A | **yes** -- directly encodes "silence"-related content; the injected concept |
| 2 | [1026](https://neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-65k/1026) | 91.0 | 1.9 | 7680 | 2 | 2 | varies | "monetary values and ranges" | N/A | **no** -- numeric/financial |
| 3 | [1993](https://neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-65k/1993) | 90.5 | 1.9 | 6880 | 3 | 2 | varies | "numbers followed by units" | N/A | **no** -- numeric formatting |
| 4 | [28563](https://neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-65k/28563) | 89.5 | 4.4 | 4032 | 297 | 34 | varies | "passage of time" | N/A | **yes** -- temporal passage strongly relates to silence/stillness concepts |
| 5 | [6575](https://neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-65k/6575) | 89.4 | 3.5 | 5056 | 16 | 12 | varies | "way or ways" | N/A | **maybe** -- manner descriptions, echoes L22 feature 3026 |
| 6 | [422](https://neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-65k/422) | 88.8 | 3.7 | 4224 | 144 | 28 | varies | "yearn to find" | N/A | **yes** -- yearning/searching is introspective-adjacent, emotional depth |
| 7 | [9567](https://neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-65k/9567) | 88.7 | 3.7 | 4224 | 48 | 20 | varies | "question tags" | N/A | **maybe** -- question tags could mark introspective qualification |
| 8 | [3664](https://neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-65k/3664) | 88.6 | 3.0 | 3408 | 5 | 4 | varies | "unmatched superiority and potential" | N/A | **no** -- superlatives |
| 9 | [37304](https://neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-65k/37304) | 88.4 | 2.0 | 4448 | 3 | 2 | varies | "phrases starting with 'nature of'" | N/A | **yes** -- "nature of" questions are quintessentially philosophical/introspective |
| 10 | [4816](https://neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-65k/4816) | 87.7 | 2.5 | 3312 | 4 | 2 | varies | "greater than or less than" | N/A | **no** -- comparison operators |
| 11 | [36331](https://neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-65k/36331) | 87.7 | 4.2 | 2960 | 19 | 12 | varies | "warning sign" | N/A | **maybe** -- anomaly detection, could mark awareness of unusual processing |
| 12 | [4000](https://neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-65k/4000) | 87.6 | 1.8 | 4000 | 2 | 2 | varies | "what's the" | N/A | **maybe** -- inquiry framing |
| 13 | [7611](https://neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-65k/7611) | 87.5 | 2.0 | 3648 | 3 | 3 | varies | "time sensitivity, time series, time frames" | N/A | **maybe** -- temporal framing related to silence/stillness |
| 14 | [13467](https://neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-65k/13467) | 87.5 | 2.2 | 3424 | 3 | 2 | varies | "lack of" | N/A | **maybe** -- "lack of" could mark absence detection (absence = silence) |
| 15 | [665](https://neuronpedia.org/gemma-3-4b-it/29-gemmascope-2-res-65k/665) | 87.2 | 2.9 | 3936 | 97 | 30 | varies | "reveal leading to change" | N/A | **yes** -- revelation/change is a strong introspective discovery signal |

**Layer 29 highlights:** Feature 12140 ("silas, silhouetted, silencers") is the top feature -- a direct "silence" concept detector at the output layer. Feature 28563 ("passage of time") fires in 34/40 A-trials with 297 total firings, the most broadly active bursty feature. Feature 665 ("reveal leading to change") fires in 30/40 trials and semantically maps to the moment of introspective discovery.

---

## Summary: Most Promising Candidates

### Tier 1: Strong introspection signals

| Layer | Feature | Explanation | Why promising |
|-------|---------|-------------|---------------|
| 22 | 10474 | "silas, silhouetted" | Direct "sil-" prefix match to injected "silence" concept; max 2592, 10 trials |
| 29 | 12140 | "silas, silhouetted, silencers" | Same "silence" detector at output layer; max 5888, 12 trials |
| 22 | 5491 | "underlying issues, themes, or problems" | Detecting hidden/underlying content is exactly what introspection does; max 2800 |
| 22 | 1840 | "what kind of" | Categorization/type-identification; max 5216 (highest in L22) |
| 22 | 4222 | "response or respond" | Formulating responses under injection; max 3008 |
| 29 | 665 | "reveal leading to change" | Revelation/discovery; fires in 30/40 A-trials |
| 29 | 28563 | "passage of time" | Temporal/stillness concept; 297 firings across 34/40 trials |

### Tier 2: Moderate signals

| Layer | Feature | Explanation | Why interesting |
|-------|---------|-------------|-----------------|
| 9 | 5753 | "subjective nature, interpretation, experience" | Directly about subjectivity; ultra-sparse (0.05% baseline) |
| 9 | 2041 | "AI-generated content technology" | AI self-reference at early layer |
| 17 | 419 | "envisioned future" | Future-planning/envisioning; max 1856 |
| 17 | 3230 | "questions asking why" | Explanatory inquiry |
| 17 | 25763 | "don't you feel/think/want" | Emotional/cognitive verbs |
| 22 | 814 | "fundamental importance" | Marking significance; 309 firings in 32 trials |
| 29 | 422 | "yearn to find" | Searching/yearning; 144 firings in 28 trials |
| 29 | 37304 | "nature of" | Philosophical inquiry; max 4448 |

### Key Observations

1. **100% A-exclusive at the top:** Every single top-15 bursty feature across all 4 layers fires only in condition A (injection + monitoring). This means the bursty ranking strongly selects for A-exclusive features, but ranks them by intensity rather than frequency.

2. **Activation magnitude scales with layer depth:** Layer 9 peaks at ~1300, layer 17 at ~1900, layer 22 at ~5200, layer 29 at ~7700. This 6x amplification from early to late layers is consistent with the injection at layer 20 propagating and amplifying through later layers.

3. **Silence-concept features emerge post-injection:** Features 10474 (L22) and 12140 (L29) both match "sil-" prefix patterns (silas, silhouetted, silencers), strongly suggesting they encode the injected "silence" concept. These appear only in post-injection layers (22, 29), not pre-injection (9, 17).

4. **Two distinct feature types:**
   - **Broad bursty features** (high count, many trials): Features like 932 (L22, 462 firings), 28563 (L29, 297 firings) fire across most trials but with high burst ratios (max >> mean). These are systematically activated by the injection.
   - **Ultra-sparse bursty features** (count <= 5, few trials): Features like 1840 (L22, 3 firings), 5753 (L9, 3 firings) fire in just 1-2 trials. These mark rare "decision spike" moments -- potentially the exact token where the model's processing shifts.

5. **Introspective language cluster at layer 22:** The post-injection layer shows a coherent set of features related to inquiry (1840: "what kind of", 318: "asking for details"), detection (5491: "underlying issues"), and response formation (4222: "response/respond"). This suggests layer 22 is where introspective processing is most concentrated.
