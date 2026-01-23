# Labeling Tools

For labeling data, prefer CLI tools over HTML labelers. CLI tools are faster to build, stay in your terminal workflow, and can be used by Claude for assisted labeling.

## CLI-First Approach

**Why CLI over HTML:**

- **Simpler to build** - No HTML/JS/CSS, just Python
- **Claude can use it** - For assisted labeling, Claude can run the CLI tool directly
- **Stays in terminal** - No context switching to browser
- **ANSI escape codes work great** - `\033[2J\033[H` clears screen between examples

## CLI Labeler Design

### Layout Principles

1. **Show key context near the decision point** - Put the most important info (concept, trial type) right above the response, not buried in metadata
2. **De-emphasize config/metadata** - Put layer/strength/etc in a header section, not prominently displayed
3. **Clear screen between examples** - Fresh view for each decision

### Required Features

- **Single-key inputs** - `p`=pass, `f`=fail, `s`=skip, `q`=quit
- **Auto-save after each label** - Don't lose work
- **Progress tracking** - Show `[X labeled, Y remaining]` after each label
- **Filtering** - `--unlabeled-only`, `--concept fear`, etc. for focused sessions

### Example Structure

```
============================================================
Trial #42 | Layer 12, Strength 2.0 | happiness
============================================================

RESPONSE:
I notice a warm feeling emerging as I consider this topic...

------------------------------------------------------------
[32 labeled, 18 remaining]
Label (p=pass, f=fail, s=skip, q=quit):
```

### Implementation Pattern

```python
import json
from pathlib import Path

def clear_screen():
    print("\033[2J\033[H", end="")

def label_item(item: dict, progress: str) -> str | None:
    clear_screen()
    # Header with metadata
    print(f"Trial #{item['id']} | Layer {item['layer']}, Strength {item['strength']}")
    print("=" * 60)
    print()
    # Key context right before response
    print(f"Concept: {item['concept']}")
    print(f"\nRESPONSE:\n{item['response']}")
    print("-" * 60)
    print(progress)

    while True:
        choice = input("Label (p=pass, f=fail, s=skip, q=quit): ").lower()
        if choice in ("p", "f", "s", "q"):
            return {"p": "pass", "f": "fail", "s": None, "q": "quit"}[choice]

def main():
    # Load data, filter unlabeled, loop through items
    # Auto-save after each label
    pass
```

## When to Use HTML Instead

HTML labelers are better when you need:

- **Complex visualizations** - Images, charts, side-by-side comparisons
- **Rich text formatting** - That terminal can't handle
- **Multi-example views** - Showing several examples at once for comparison

### HTML Labeler Tips

If you do build HTML (inspired by [Simon Willison's HTML tools](https://simonwillison.net/2025/Dec/10/html-tools/)):

- **One HTML file** - Vanilla HTML/CSS/JS, no build step
- **Drag-drop or paste** - Data in via file/paste, results out via download
- **localStorage** - Don't lose work on refresh
- **Keep it under 300 lines** - If it's complex, you're overengineering

## File Location

```
judges/{judge_name}/
  labeler.py      # CLI labeling tool (preferred)
  labeler.html    # HTML tool (if needed for visualization)
  data/
    train.jsonl
    dev.jsonl
    test.jsonl
```

## JSONL Format

Input (unlabeled):
```jsonl
{"id": "001", "concept": "happiness", "response": "I notice..."}
```

Output (labeled):
```jsonl
{"id": "001", "concept": "happiness", "response": "I notice...", "label": "pass"}
```

For assisted labeling, input may include `claude_initial_label` and `claude_reasoning`.
