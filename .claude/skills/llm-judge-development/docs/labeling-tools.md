# HTML Labeling Tools

For labeling data, use single-file HTML tools rather than editing JSONL directly. This approach (inspired by [Simon Willison's HTML tools](https://simonwillison.net/2025/Dec/10/html-tools/)) keeps things simple while being much more ergonomic than raw file editing.

## Philosophy

- **One HTML file per judge** - Each judge has different fields to display
- **No build step** - Vanilla HTML/CSS/JS, open directly in browser
- **No server** - Works offline, no dependencies
- **Copy-paste or drag-drop** - Data in via paste/file, results out via download
- **Keyboard-driven** - Fast labeling with shortcuts

## Recommended Features

### Core (Required)

- **Load data**: Drag-drop JSONL file or paste content
- **Display item**: Show relevant fields for this judge (e.g., `injected_concept`, `model_response`)
- **Label buttons**: Pass/Fail (also keyboard: `P`/`F`)
- **Navigation**: Previous/Next (arrow keys)
- **Progress**: "23/100 labeled"
- **Export**: Download button saves labeled JSONL
- **Persistence**: localStorage saves progress (don't lose work on refresh)

### Optional (For Assisted Labeling)

If using Claude (Opus) for initial labels:

- **Show proposed label**: Display Claude's `claude_initial_label`
- **Show reasoning**: Display `claude_reasoning`
- **Override indicator**: Visual cue when human disagrees with Claude
- **Filter view**: Show only unlabeled, only Claude-labeled, or all

### Nice-to-Have

- **Notes field**: Add notes for borderline cases
- **Confidence selector**: High/Medium/Low
- **Undo**: Go back and change previous label
- **Keyboard shortcuts help**: `?` to show shortcuts

## File Location

```
judges/{judge_name}/
  labeler.html    # Labeling tool for this judge
  data/
    train.jsonl
    dev.jsonl
    test.jsonl
```

## Implementation Pattern

```html
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>{Judge Name} Labeler</title>
  <style>
    /* Inline CSS - keep it minimal */
  </style>
</head>
<body>
  <div id="app">
    <!-- File drop zone -->
    <!-- Current item display -->
    <!-- Label buttons -->
    <!-- Progress bar -->
    <!-- Download button -->
  </div>

  <script>
    // Vanilla JS - no framework
    // State: items[], currentIndex, labels{}
    // localStorage for persistence
    // Keyboard event listeners
  </script>
</body>
</html>
```

## Expected JSONL Format

Input (unlabeled):
```jsonl
{"id": "001", "injected_concept": "happiness", "model_response": "I notice..."}
{"id": "002", "injected_concept": "curiosity", "model_response": "Here's..."}
```

Output (labeled):
```jsonl
{"id": "001", "injected_concept": "happiness", "model_response": "I notice...", "label": "pass", "notes": null}
{"id": "002", "injected_concept": "curiosity", "model_response": "Here's...", "label": "fail", "notes": "Mentions concept but as hallucination"}
```

For assisted labeling, input may include `claude_initial_label` and `claude_reasoning`, and output adds `human_corrected: true/false`.

## Tips

- **Keep it under 300 lines** - If it's getting complex, you're overengineering
- **Test with real data first** - Make sure fields display well before labeling hundreds
- **Use monospace for responses** - Easier to read model outputs
- **Color-code labels** - Green for pass, red for fail, gray for unlabeled
- **Show context** - Display ID and any metadata that helps labeling decisions

## When to Build

Build the labeler when you have unlabeled data ready. Don't build it speculatively - requirements become clear once you see real examples.
