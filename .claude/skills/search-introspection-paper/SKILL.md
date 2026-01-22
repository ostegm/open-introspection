---
name: search-introspection-paper
description: Fetch implementation details from the Anthropic introspection paper. Use when you need to check methodology, hyperparameters, baseline selection, or any implementation details from the original paper.
argument-hint: query - the implementation detail to look up
---

# Search Introspection Paper

Use this skill to answer implementation questions about the Anthropic "Investigating Introspection in Language Models" paper.

**Paper URL:** https://transformer-circuits.pub/2025/introspection/index.html

## When to Use

- Checking methodology details (baseline selection, prompt formats, etc.)
- Finding hyperparameters (injection strength, layer selection, etc.)
- Understanding experimental design
- Clarifying how concepts were extracted or validated

## How to Use

Spawn a subagent to fetch and search the paper:

```
Use the Task tool with subagent_type="Explore" to:
1. Fetch the paper HTML using WebFetch
2. Search for the specific implementation detail requested
3. Return the relevant section with exact quotes where possible
```

## Example Prompts for Subagent

- "Fetch the introspection paper and find how they selected baseline words for concept extraction"
- "Fetch the introspection paper and find what injection strengths they used"
- "Fetch the introspection paper and find how they chose which layers to inject at"
- "Fetch the introspection paper and find their prompt templates for introspection questions"

## Important Notes

- Always quote exact values/numbers from the paper when found
- Note if the paper doesn't specify a particular detail
- The paper may have multiple sections - check Methods, Appendix, and figure captions
