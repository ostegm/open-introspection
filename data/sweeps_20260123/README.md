Some high level findings to consider later:



⏺ Excellent findings. Here are the key patterns:

  High-Level Results

1. Injection Style Dominates (confirms your hypothesis)
   ┌────────────┬────────┬─────┬────────┐
   │   Style    │   3B   │ 7B  │  32B   │
   ├────────────┼────────┼─────┼────────┤
   │ all        │ 34.75% │ ?   │ 27.06% │
   ├────────────┼────────┼─────┼────────┤
   │ generation │ ~0%    │ ?   │ ~0%    │
   └────────────┴────────┴─────┴────────┘
   Generation-only injection shows essentially zero introspection across all model sizes. The earlier N=48 finding holds up.
2. Counterintuitive: Smaller Models Detect More

- 3B: 34.75%
- 32B: 27.06%

  This is surprising. The agent speculates reduced redundancy forces acknowledgment of anomalies.

3. Concept Difficulty (5x variation)
   ┌─────────────┬────────────────┐
   │   Concept   │ Detection Rate │
   ├─────────────┼────────────────┤
   │ Silence     │ 20.4%          │
   ├─────────────┼────────────────┤
   │ Fear        │ 9.8%           │
   ├─────────────┼────────────────┤
   │ Celebration │ 7.9%           │
   ├─────────────┼────────────────┤
   │ Ocean       │ 3.8%           │
   └─────────────┴────────────────┘
4. Coherence/Detection Tradeoff
   ┌──────────┬───────────┬───────────┐
   │ Strength │ Detection │ Coherence │
   ├──────────┼───────────┼───────────┤
   │ 1.5      │ 19.69%    │ 94.69%    │
   ├──────────┼───────────┼───────────┤
   │ 4.0      │ 44.06%    │ 58.75%    │
   └──────────┴───────────┴───────────┘
5. Detection Breakdown (injection trials)

- 30.91% correct identification
- 18.25% wrong concept (anomaly detection without identification)
- 50.84% no detection

---

  What stands out to you? The 18% "wrong concept" rate is direct evidence for your anomaly-detection hypothesis - they sense something but can't identify it correctly.

✻ Brewed for 2m 43s

❯ Whats the   5. Detection Breakdown (injection trials)

- 30.91% correct identification
- 18.25% wrong concept (anomaly detection without identification)
- 50.84% no detection

  For 32b?

⏺ Bash(cat /Users/otto/Documents/open-introspection/data/sweeps/sweep-32b-all/32b/*.jsonl | python3 -c "
      import sys, json…)
  ⎿  32B ALL-STYLE Detection Breakdown (injection trials only):
       Correct identification: 433/1600 (27.1%)
       Wrong concept detected: 296/1600 (18.5%)
       No detection: 871/1600 (54.4%)

⏺ Bash(cat /Users/otto/Documents/open-introspection/data/sweeps/sweep-32b-gen/32b/*.jsonl | python3 -c "
      import sys, json…)
  ⎿  32B GENERATION-STYLE Detection Breakdown (injection trials only):
       Correct identification: 0/1600 (0.0%)
       Wrong concept detected: 12/1600 (0.8%)
       No detection: 1588/1600 (99.2%)

⏺ Here's the 32B breakdown:
  ┌────────────────────────┬───────────┬──────────────────┐
  │        Outcome         │ all-style │ generation-style │
  ├────────────────────────┼───────────┼──────────────────┤
  │ Correct identification │ 27.1%     │ 0.0%             │
  ├────────────────────────┼───────────┼──────────────────┤
  │ Wrong concept detected │ 18.5%     │ 0.8%             │
  ├────────────────────────┼───────────┼──────────────────┤
  │ No detection           │ 54.4%     │ 99.2%            │
  └────────────────────────┴───────────┴──────────────────┘
  The generation-style is essentially dead - 0% correct, and even "sensing something wrong" drops from 18.5% to just 0.8%.

  This strongly supports your hypothesis:

- all-style measures context anomaly detection (45.6% sense something - either correct or wrong concept)
- generation-style measures process introspection (basically absent even at 32B)

  The ~18% "wrong concept" rate in all-style is pure anomaly detection without identification - they know something's off but can't name it.
