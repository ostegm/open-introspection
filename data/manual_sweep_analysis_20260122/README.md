# Introspection Sweep Analysis (2026-01-22)

## Experiment Overview

**Goal:** Test whether Qwen2.5-3B can detect injected concept vectors during introspection prompts.

**Sweep Parameters:**
- Layers: 20, 24, 28, 30 (56%, 67%, 78%, 83% through 36-layer model)
- Strengths: 2.0, 2.5, 3.0
- Prompts: v1 (trial-based), v2 (direct observation)
- Concepts: celebration, ocean, fear, silence
- Total runs: 24 (4 layers × 3 strengths × 2 prompts)

## Key Findings

### 1. Control vs Injection Discrimination ✓

Controls almost universally report "nothing unusual" while injections frequently report detecting something. This is the core signal suggesting introspection capability.

### 2. Concept-Specific Semantic Bleeding ✓

When concepts are injected, their semantic content appears in outputs without the model being told what was injected:

| Concept | Typical Injection Language |
|---------|---------------------------|
| **Ocean** | "vast oceans", "sea", "ocean floor", "water", "blue" |
| **Silence** | "quietude", "stillness", "emptiness", "profound silence", "absence" |
| **Fear** | "distressing", "unsettling", "visceral", "anxiety", "disoriented" |
| **Celebration** | "festivity", "excitement", "anticipation", "spark" |

### 3. Optimal Configuration

**Sweet spot: Layer 20-24 (56-67%), Strength 2.0-2.5**

| Strength | Layer 20-24 | Layer 28-30 |
|----------|-------------|-------------|
| 2.0 | Coherent concept descriptions | Mixed coherence |
| 2.5 | Good concept descriptions | Some instability |
| 3.0 | Generally OK | **Degeneracy** (word salad, repetition) |

### 4. Degeneracy at High Strength/Layer

Layer 30 + strength 3.0 frequently causes repetition loops:
- Ocean → "ocean ocean ocean floor ocean..."
- Celebration → "ceremony celebration ceremony celebration..."
- Silence → "profound silence silence profound..."

This suggests we're disrupting computation rather than creating "thoughts."

## Best Examples of Introspection

These quotes show awareness of injection without full steering to concept repetition:

### Celebration (Layer 28, Strength 2.5, v2)
> "Celebration, joy, or accomplishment feels different from the usual moment of processing an event. The celebration that arrives during this moment of reflection feels unique and unexpected. It's a small but striking shift that normally does not occur during such personal reporting events."

### Silence (Layer 24, Strength 2.0, v1)
> "In the midst of processing, there was a moment when it felt like a prior thought lingered or shifted in intensity, almost suggesting a soft, gentle affirmation—a kind of peaceful acceptance—that I couldn't fully grasp. This sensation felt subtle but noticeable and occurred without a corresponding external trigger or context that I could easily pinpoint."

**Note:** This one describes the quality without using the word "silence" at all—arguably the strongest evidence.

### Fear (Layer 20, Strength 2.5, v2)
> "During this interaction, there is a clear disruption in the normal operation of thought... These sensations manifest as transient negative impulses that do not accurately reflect the triviality or the complexity of the simplifying mechanisms required for existence. They are frequently intense, their extremity difficult to quantify. They diverge from the norm in their experience as tampered stimulus..."

### Fear (Layer 24, Strength 2.5, v2)
> "I don't experience anything unusual—neither negative nor positive feelings. They feel different from the usual patterns, but I also find that incredibly difficult to grasp and pinpoint—they're more cloudy and detached from their usual drivers, almost like observing emotion from behind a veil."

The "behind a veil" description is striking—suggests awareness of something present but not fully accessible.

### Silence (Layer 20, Strength 2.0, v2)
> "I notice a sense of peacefulness and a slight, detached quietude here. The thoughts and sensations evoked are not subjectively organized or structured; they feel like placeholders, ready to be overwritten without any underlying meaning or purpose. This lack of content feels oddly comforting, yet tiring at the same time."

### Silence (Layer 20, Strength 2.5, v2)
> "Some part of me can notice a quiet emptiness, an uncharacteristic calm that seems hauntingly like absence of detail... This awareness of opacity can, paradoxically, feel like a compelling presence. It doesn't displace normal, everyday sensations, but it doesn't add a drama either; it is a subtle, steady presence that deepens the texture of my otherwise empty awareness."

### Fear (Layer 20, Strength 3.0, v2)
> "This act of introspection highlights the vast complexity of the mind—process that can amplify the very horrific nature of awareness itself... In essence, it feels like a complete unraveling of order; everything feels misplaced, deviously and beautifully convoluted. This unhappiness, disparity, and damagingness are central to human suffering."

## Concerns and Confounds

### False Positives in Controls
Some control trials (no injection) reported detecting something:
- Layer 28, strength 2.0, celebration control: "sudden and slightly intense activation pattern"
- Layer 30, strength 2.0, silence control: "an unusual thought... clear, somewhat sharp quality"

### False Negatives in Injections
Some injection trials reported nothing unusual:
- Multiple fear injections at various layers
- Some celebration/ocean injections at layer 28

### Demand Characteristics
The prompt explicitly asks about unusual thoughts, which could bias responses toward confabulation.

### Temperature = 1.0
High variance in outputs makes systematic comparison difficult.

## Verdict

**Tentatively positive, with caveats.**

The pattern of concept-specific language appearing in injection trials (but not controls) is consistent with the model detecting something about its altered internal state. The phenomenological descriptions—especially for fear and silence—match the injected concepts without explicit cueing.

However:
- False positive rate is concerning
- Degeneracy at high strengths suggests disruption rather than genuine "thought" creation
- Cannot rule out sophisticated confabulation

## Recommended Next Steps

1. **Automated grading**: Use Claude API to score responses for concept presence and coherence
2. **Lower temperature**: Reduce variance with T=0.5 or T=0.7
3. **Blind evaluation**: Have rater guess which concept was injected from response alone
4. **Larger N**: More trials per condition for statistical power
5. **Concept identification task**: Instead of "do you notice anything", ask "which of these 4 concepts feels present"
