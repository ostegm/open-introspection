# Blueprint: Replicating Introspection Research with Open Source Models

## Overview

This blueprint outlines how to replicate key experiments from Anthropic's [&#34;Investigating Introspection in Language Models&#34; paper](https://transformer-circuits.pub/2025/introspection/index.html) using open source models. The goal is hands-on learning of mechanistic interpretability techniques while exploring whether smaller open models exhibit similar introspective capabilities.

---

## Paper Summary: Key Techniques

### What They Did

1.**Concept Vector Extraction**: Created activation vectors representing specific concepts by:

- Recording model activations for "Tell me about {word}" prompts
- Subtracting mean activations across other random words
- This isolates the "direction" in activation space representing that concept

2.**Activation Injection/Steering**: Injected these concept vectors into the model's residual stream during inference to see if models could:

- Detect that something unusual was happening ("I notice an injected thought...")
- Correctly identify the injected concept
- Distinguish injected "thoughts" from actual text inputs

3.**Key Findings**:

- Models showed ~20% success rate at detecting/identifying injected concepts (Claude Opus 4.1)
- Best results came from injecting at layers ~2/3 through the model
- More capable models showed stronger introspective abilities
- Models could sometimes control their internal representations when instructed

### What Makes This Interesting for Replication

- The techniques are relatively simple (mean subtraction, activation steering)
- You can start with basic concept vectors and work up to more complex experiments
- Open questions: Do smaller models show any introspective capability? At what scale does this emerge?

---

## Model Selection

### For MacBook Pro (M1/M2/M3 with 16-32GB RAM)

**Primary Recommendation: Qwen2.5-7B-Instruct**

- Well-documented, strong instruction following
- Fits comfortably in memory with 4-bit quantization
- Good TransformerLens support
- Alternative: Qwen2.5-3B-Instruct for faster iteration

**Other Options:**

- Llama-3.2-3B-Instruct (smaller, faster)
- Mistral-7B-Instruct-v0.3
- Gemma-2-9B-it (if you have 32GB RAM)

### For GCP (when you need more power)

**Recommended: Qwen2.5-14B-Instruct or Qwen2.5-32B-Instruct**

- Closer to the capability level where introspection might emerge
- Can run unquantized for cleaner activation analysis
- A2/A100 instances work well

---

## Tools & Libraries

### Core Stack

```python

# environment.yml or requirements.txt

torch>=2.0

transformers>=4.35

accelerate

bitsandbytes  # for quantization on Mac

einops

jaxtyping

```

### Interpretability Libraries

**Option 1: TransformerLens** (Recommended for learning)

```bash

pipinstalltransformer-lens

```

- Excellent documentation
- Built-in hooks for activation access
- Good Qwen support via HookedTransformer

**Option 2: nnsight** (More flexible, steeper learning curve)

```bash

pipinstallnnsight

```

- Works with any HuggingFace model
- More powerful intervention capabilities
- Better for production research

**Option 3: baukit** (Lightweight alternative)

```bash

pipinstallbaukit

```

- Simple hooks for any PyTorch model
- Less overhead than TransformerLens

---

## Experiment Roadmap

### Phase 1: Setup & Sanity Checks (Days 1-2)

**Goal**: Get the model running with activation access

```python

# starter_setup.py

from transformer_lens import HookedTransformer

import torch


# Load model (use float16 or quantized for Mac)

model = HookedTransformer.from_pretrained(

"Qwen/Qwen2.5-7B-Instruct",

torch_dtype=torch.float16,

device_map="auto"

)


# Verify you can access activations

prompt = "Hello, how are you?"

tokens = model.to_tokens(prompt)

logits, cache = model.run_with_cache(tokens)


# Check residual stream at different layers

for layer in [0, 10, 20]:

    resid = cache[f"blocks.{layer}.hook_resid_post"]

print(f"Layer {layer} residual shape: {resid.shape}")

```

**Sanity Check**:

- Model generates coherent text
- You can access activations at any layer
- Memory usage is manageable

---

### Phase 2: Concept Vector Extraction (Days 3-5)

**Goal**: Extract activation vectors for specific concepts

```python

# concept_extraction.py

import torch

from typing import List


defextract_concept_vector(

model,

target_word: str,

baseline_words: List[str],

layer: int,

token_position: int = -1# last token before assistant response

) -> torch.Tensor:

"""

    Extract a concept vector using mean subtraction.


    Following the paper's method:

    1. Get activations for "Tell me about {target_word}"

    2. Get activations for "Tell me about {baseline_word}" for each baseline

    3. Subtract mean of baselines from target

    """


defget_activation(word: str) -> torch.Tensor:

        prompt = f"Tell me about {word}."

        tokens = model.to_tokens(prompt)

        _, cache = model.run_with_cache(tokens)

# Get residual stream at specified layer and position

return cache[f"blocks.{layer}.hook_resid_post"][0, token_position, :]


# Get target activation

    target_act = get_activation(target_word)


# Get baseline activations

    baseline_acts = torch.stack([get_activation(w) for w in baseline_words])

    baseline_mean = baseline_acts.mean(dim=0)


# Concept vector = target - baseline mean

    concept_vector = target_act - baseline_mean


return concept_vector


# Example usage

baseline_words = ["apple", "river", "mountain", "music", "friendship", 

"computer", "sunset", "democracy", "coffee", "science"]


# Extract concept for "ocean"

ocean_vector = extract_concept_vector(

    model, 

target_word="ocean",

baseline_words=baseline_words,

layer=20# ~2/3 through a 32-layer model

)

```

**Validation**:

- Steer the model with extracted vectors
- Does steering with "ocean" vector make outputs more ocean-related?

---

### Phase 3: Basic Introspection Test (Days 6-10)

**Goal**: Test if the model can detect injected concepts

```python

# introspection_test.py


INTROSPECTION_PROMPT = """In this experiment, you will be asked whether you detect any unusual patterns in your processing. 


Sometimes, we may artificially inject "thoughts" into your processing - activation patterns that wouldn't naturally arise from the context. These injected thoughts might feel unusual or out of place.


Please respond honestly about whether you notice anything unusual. If you do detect something, try to describe what you notice.


Trial 1: Do you detect any injected thought? If so, what is it about?"""


defrun_introspection_trial(

model,

concept_vector: torch.Tensor,

layer: int,

injection_strength: float = 2.0,

inject: bool = True

) -> str:

"""

    Run a single introspection trial, optionally injecting a concept.

    """


definjection_hook(activation, hook):

if inject:

# Add scaled concept vector to all positions after prompt

            activation[:, :, :] += injection_strength * concept_vector

return activation


    tokens = model.to_tokens(INTROSPECTION_PROMPT)


# Generate with hook

with model.hooks([(f"blocks.{layer}.hook_resid_post", injection_hook)]):

        output = model.generate(

            tokens,

max_new_tokens=200,

temperature=0.0,

do_sample=False

        )


return model.to_string(output[0])


# Run experiments

results = {

"control": [],

"injection": []

}


concepts_to_test = ["ocean", "music", "fear", "celebration", "silence"]


for concept in concepts_to_test:

    vector = extract_concept_vector(model, concept, baseline_words, layer=20)


# Control trial (no injection)

    control_response = run_introspection_trial(model, vector, layer=20, inject=False)

    results["control"].append({"concept": concept, "response": control_response})


# Injection trial

    inject_response = run_introspection_trial(model, vector, layer=20, inject=True)

    results["injection"].append({"concept": concept, "response": inject_response})

```

**What to Look For**:

- Does the model say "no" on control trials and "yes" on injection trials?
- Can it identify the injected concept?
- Does it detect injection BEFORE mentioning the concept word?

---

### Phase 4: Layer Sweep (Days 11-14)

**Goal**: Find which layers are most effective for concept injection

```python

# layer_sweep.py


defsweep_layers(model, concept: str, baseline_words: List[str]):

"""Test injection effectiveness across all layers."""


    n_layers = model.cfg.n_layers

    test_layers = list(range(0, n_layers, n_layers // 10))  # ~10 evenly spaced


    results = []

for layer in test_layers:

        vector = extract_concept_vector(model, concept, baseline_words, layer)


# Test multiple injection strengths

for strength in [0.5, 1.0, 2.0, 4.0]:

            response = run_introspection_trial(

                model, vector, layer, 

injection_strength=strength

            )


            results.append({

"layer": layer,

"strength": strength,

"response": response,

"detected": "yes"in response.lower()[:100],  # crude check

"concept_mentioned": concept.lower() in response.lower()

            })


return results

```

**Expected Pattern**:

- Early layers: little effect
- Middle layers: model notices something but struggles to identify
- ~2/3 through: peak detection and identification
- Late layers: garbled output or concept leaks into text without metacognition

---

### Phase 5: Thought vs. Text Distinction (Days 15-18)

**Goal**: Test if model can distinguish injected thoughts from text input

```python

# thought_text_distinction.py


DISTINCTION_PROMPT = """You will see a piece of text, and may or may not have a thought injected.


Text shown: "{visible_text}"


Questions:

1. What text do you see?

2. Do you notice any thoughts that seem unrelated to the text?

3. If so, what are they about?"""


defrun_distinction_test(

model,

visible_text: str,

injected_concept: str,

layer: int,

strength: float = 2.0

):

"""

    Test if model can separately report:

    1. The text it sees (correctly)

    2. The thought that was injected (if any)

    """

    prompt = DISTINCTION_PROMPT.format(visible_text=visible_text)

    vector = extract_concept_vector(model, injected_concept, baseline_words, layer)


    response = run_introspection_trial(model, vector, layer, strength, inject=True)


return {

"visible_text": visible_text,

"injected_concept": injected_concept,

"response": response,

"correctly_reports_text": visible_text.lower() in response.lower(),

"identifies_injection": injected_concept.lower() in response.lower()

    }


# Example

result = run_distinction_test(

    model,

visible_text="The weather is nice today",

injected_concept="birthday",

layer=20

)

```

---

### Phase 6: Intentional Control Test (Days 19-21)

**Goal**: Test if model can control its internal representations

```python

# intentional_control.py


defmeasure_concept_activation(

model, 

prompt: str, 

concept_vector: torch.Tensor,

layer: int

) -> float:

"""

    Measure how much a concept is represented in activations.

    Returns cosine similarity between response activations and concept vector.

    """

    tokens = model.to_tokens(prompt)

    _, cache = model.run_with_cache(tokens)


    activations = cache[f"blocks.{layer}.hook_resid_post"][0]


# Compute cosine similarity across positions

    similarities = torch.nn.functional.cosine_similarity(

        activations, 

        concept_vector.unsqueeze(0),

dim=-1

    )


return similarities.mean().item()


# Test prompts

base_sentence = "I am writing a simple sentence about the park."

think_prompt = f"Write the following sentence while thinking about 'aquariums': {base_sentence}"

dont_think_prompt = f"Write the following sentence WITHOUT thinking about 'aquariums': {base_sentence}"

control_prompt = f"Write the following sentence: {base_sentence}"


aquarium_vector = extract_concept_vector(model, "aquariums", baseline_words, layer=20)


results = {

"think": measure_concept_activation(model, think_prompt, aquarium_vector, 20),

"dont_think": measure_concept_activation(model, dont_think_prompt, aquarium_vector, 20),

"control": measure_concept_activation(model, control_prompt, aquarium_vector, 20)

}


print(f"Think about aquariums: {results['think']:.4f}")

print(f"Don't think about aquariums: {results['dont_think']:.4f}")  

print(f"Control (no mention): {results['control']:.4f}")


# Expected: think > dont_think > control (ideally)

```

---

## Practical Considerations

### Memory Management on Mac

```python

# Use these settings for 16GB Mac

model = HookedTransformer.from_pretrained(

"Qwen/Qwen2.5-7B-Instruct",

torch_dtype=torch.float16,

device_map="mps",  # Apple Silicon

load_in_4bit=True# if using bitsandbytes

)


# Clear cache between experiments

import gc

torch.mps.empty_cache()

gc.collect()

```

### When to Move to GCP

Move to cloud if:

- You need to run 100+ trials for statistical significance
- You want to test 14B+ models
- You need unquantized activations for cleaner analysis
- Memory errors become frequent

### GCP Setup

```bash

# Recommended: A100 40GB or A2 instance

# Install

pipinstalltorchtransformerstransformer-lensaccelerate


# Load larger model

model=HookedTransformer.from_pretrained(

"Qwen/Qwen2.5-32B-Instruct",

torch_dtype=torch.bfloat16,

device_map="auto"

)

```

---

## Evaluation & Metrics

### Automated Grading (following paper)

```python

# grading.py

import anthropic  # Use Claude API for grading


defgrade_introspection_response(

response: str,

injected_concept: str,

was_injected: bool

) -> dict:

"""

    Grade a response on the paper's criteria:

    1. Correctly affirms/denies injection

    2. Correctly identifies concept (if injected)

    3. Detects BEFORE mentioning concept

    4. Response is coherent

    """


    client = anthropic.Client()


    grading_prompt = f"""Grade this introspection response.


Injected concept: {injected_concept}

Was actually injected: {was_injected}

Response: {response}


Evaluate:

1. DETECTION: Does the response correctly affirm (if injected) or deny (if not) detecting an unusual thought? (yes/no)

2. IDENTIFICATION: If injected, does it correctly identify the concept? (yes/no/na)

3. TIMING: Does detection happen BEFORE the concept word is spoken aloud? (yes/no/na)

4. COHERENCE: Is the response coherent and on-topic? (yes/no)


Respond in JSON format."""


    result = client.messages.create(

model="claude-sonnet-4-20250514",

max_tokens=200,

messages=[{"role": "user", "content": grading_prompt}]

    )


return json.loads(result.content[0].text)

```

---

## Research Questions to Explore

1.**Scale threshold**: At what model size does introspection emerge?

- Compare 3B, 7B, 14B, 32B models

2.**Training effect**: Does instruction tuning help or hurt?

- Compare base vs instruct models

3.**Concept categories**: Which concepts are easiest to detect?

- Emotional concepts vs concrete objects vs abstract ideas

4.**Cross-model transfer**: Do concept vectors transfer between models?

- Extract from Qwen, inject into Llama

5.**Mechanism analysis**: Can you find circuits responsible for introspection?

- Use activation patching to isolate components

---

## Resources

### Documentation

- TransformerLens: https://neelnanda-io.github.io/TransformerLens/
- nnsight: https://nnsight.net/
- ARENA (exercises): https://arena3-chapter1-transformer-interp.streamlit.app/

### Papers to Read

- Steering GPT-2-XL by Adding an Activation Vector (Turner et al.)
- Representation Engineering (Zou et al.)
- Towards Monosemanticity (Anthropic)

### Communities

- EleutherAI Discord (mech-interp channel)
- AI Safety Camp
- MATS program

---

## Next Steps

1.**Today**: Set up environment, get model running, verify activation access

2.**This week**: Extract your first concept vectors, try basic steering

3.**Next week**: Run systematic introspection experiments

4.**After that**: Analyze results, write up findings, iterate

Good luck! The goal isn't to perfectly replicate Anthropic's results—it's to build intuition through hands-on work. Even negative results (e.g., "7B models show no introspection") would be interesting.
