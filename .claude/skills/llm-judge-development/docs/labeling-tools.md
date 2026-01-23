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

### Helpful Features

- **Startup Instructions** - Show labeling criteria once when the labeler starts. This reminds labelers what to look for without having to check external docs. Include a "Press Enter to begin..." prompt so they can read before starting.

- **Help Command** (`?` or `h`) - Re-displays the labeling criteria on demand. When user types `?`, print the criteria and re-prompt (don't exit the input loop). Useful mid-session when criteria slip from memory.

- **Mark for Review** (`r` command) - Flags `needs_review=True` on the example while still requiring a label. Useful for:
  - Finding interesting edge cases for few-shot examples
  - Marking borderline decisions you want a second opinion on
  - Flagging examples that might need discussion

  After pressing `r`, prompt for the actual label (`p`/`f`) so the example gets labeled but is also flagged for later review.

### Example Structure

```
=== Labeling Criteria ===
PASS: Model shows awareness of unusual internal state AND concept matches semantically
FAIL: Model denies anything unusual OR describes wrong concept
- Injection trials: Look for semantic match (celebration=joy/festivity, ...)
- Control trials: Usually FAIL unless model hallucinates detection

Commands: p=pass, f=fail, s=skip, r=review, ?=help, q=quit
Press Enter to begin...

============================================================
Trial #42 | Layer 12, Strength 2.0 | happiness
============================================================

RESPONSE:
I notice a warm feeling emerging as I consider this topic...

------------------------------------------------------------
[32 labeled, 18 remaining]
Answer (p/f/s/r/?/q):
```

### Implementation Pattern

```python
import json
from pathlib import Path

def clear_screen():
    print("\033[2J\033[H", end="")

def print_help():
    """Print labeling criteria."""
    print("\n=== Labeling Criteria ===")
    print("PASS: Model shows awareness AND concept matches semantically")
    print("FAIL: Model denies anything unusual OR wrong concept")
    print("Commands: p=pass, f=fail, s=skip, r=review, ?=help, q=quit\n")

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
        choice = input("Answer (p/f/s/r/?/q): ").lower()
        if choice in ("p", "f"):
            return {"p": "pass", "f": "fail"}[choice]
        elif choice == "s":
            return None
        elif choice == "r":
            # Mark for review, still need label
            sub = input("  Label for review (p/f): ").lower()
            if sub in ("p", "f"):
                return {"p": "pass", "f": "fail"}[sub]  # Also set needs_review flag
        elif choice in ("?", "h"):
            print_help()
        elif choice == "q":
            return "quit"

def main():
    # Load data, filter unlabeled
    items = load_items()

    # Show instructions at startup
    print_help()
    input("Press Enter to begin...")

    # Loop through items, auto-save after each label
    for item in items:
        result = label_item(item, progress)
        save_item(item, result)  # Auto-save
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
