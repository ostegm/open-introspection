---
name: md-to-blog
description: Use when the user wants to create a blog post, convert markdown to HTML for the blog, write up research findings, or publish to the Cloudripper Labs blog. Guides through brainstorming, editing, proofreading, and HTML conversion.
argument-hint: topic or markdown file path
---
# Markdown to Blog Post

Convert research notes and markdown drafts into polished Cloudripper Labs blog posts. The blog posts are meant to document our journey here of learning mech interp while trying to provide useful findings about open source models. 

## Workflow

### Phase 1: Content Development

Before converting to HTML, offer to help develop the content:

**Ask the user:**

> "Before we convert to HTML, would you like help with any of these?"
>
> 1. **Brainstorming** - flesh out the structure and key points
> 2. **Editing** - improve clarity, flow, and technical accuracy
> 3. **Clarifying** - identify gaps or ambiguous sections
> 4. **Skip to conversion** - content is ready as-is

If brainstorming/editing/clarifying:

- Work through the content iteratively
- Ask clarifying questions one at a time
- Suggest improvements to structure, examples, and explanations
- For technical content, verify accuracy against the codebase

### Phase 2: Proofreading

When the user indicates the content is finalized:

1. **Read the full content** carefully
2. **Check for:**
   - Spelling and grammar errors
   - Awkward phrasing or unclear sentences
   - Technical accuracy (code snippets, terminology)
   - Consistent formatting (headings, lists, code blocks)
   - Missing context for readers unfamiliar with the project
3. **Present issues** found and suggest fixes
4. **Wait for user approval** before proceeding to conversion

### Phase 3: HTML Conversion

Convert the finalized markdown to HTML using the blog template.

**Template location:** `blog/posts/_template.html`

**Conversion steps:**

1. Read the template file
2. Replace placeholders:
   - `POST_TITLE` → article title
   - `MONTH YEAR` → publication date (e.g., "January 2026")
3. Convert markdown content to HTML:
   - `# Heading` → `<h1>` (article title, already in template)
   - `## Heading` → `<h2>`
   - `### Heading` → `<h3>`
   - `**bold**` → `<strong>`
   - `*italic*` → `<em>`
   - `[text](url)` → `<a href="url">text</a>`
   - `` `code` `` → `<code>code</code>`
   - Code blocks → `<pre><code>...</code></pre>`
   - `> quote` → `<blockquote>...</blockquote>`
   - Lists → `<ul>/<ol>` with `<li>` items
   - `---` → `<hr>`
   - Images → `<figure><img src="..." alt="..."><figcaption>...</figcaption></figure>`
4. Write the HTML file to `blog/posts/NN-slug-title.html`
5. **Update the index** - add an entry to `blog/index.html` post list

**File naming:** Use format `NN-kebab-case-title.html` where NN is the next number in sequence.

### Phase 4: Verification

After conversion:

1. Show the user the generated HTML path
2. Suggest: `open blog/posts/NN-slug-title.html` to preview
3. Ask if any adjustments are needed

## Style Guidelines

Match the aesthetic of the Anthropic Transformer Circuits papers:

- **Academic but accessible** - explain concepts clearly, cite sources, friendly, human tone - not too academic.
- **Show code** - include relevant snippets with context
- **Use figures** - charts and diagrams where helpful
- **Keep sections focused** - one main idea per section
- **Link to resources** - GitHub repo, original paper, etc.

## Quick Reference

```
blog/
├── index.html           # Update post list here
├── posts/
│   ├── _template.html   # Copy and fill this
│   └── NN-title.html    # Your new post
├── css/style.css        # Styling (don't modify)
└── assets/              # Images, logo, etc.
```
