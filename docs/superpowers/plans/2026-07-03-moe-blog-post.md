# Mixture of Experts Blog Post Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish a beginner-friendly English blog post explaining Mixture of Experts, per the approved spec `docs/superpowers/specs/2026-07-03-moe-blog-post-design.md`.

**Architecture:** One new markdown file in `public/posts/` (posts are fetched at runtime from `/posts/<file>`), registered by adding its filename to the `markdownFiles` array in `src/data/posts/markdownImports.js`. No other site code changes. The post is written incrementally — sections committed as they are completed, matching this repo's history (see commits for `from-rlhf-to-dpo`).

**Tech Stack:** Markdown with YAML frontmatter; PyTorch for the code sample (verified via `uv run --with torch`, never system python3); CRA dev server for render checks.

## Global Constraints

- Post language: **English**. Author: `Tai Vu`.
- Post file: `public/posts/2026-07-03-mixture-of-experts-for-beginners.md` (exact name — it doubles as the registration key).
- Slug: `mixture-of-experts-explained`. Internal links to other posts use the pattern `/blog/<slug>` (route defined in `src/App.js:65`).
- Style must match existing posts (`2026-06-29-from-rlhf-to-dpo.md` is the reference): emoji on the H1 only, numbered `##` sections, `---` between sections, bold for key terms, fenced code blocks with language tags.
- The recurring narrative thread is **total parameters vs active parameters**, anchored on the hook "Mixtral 8x7B has ~47B parameters but runs like a 13B model."
- The PyTorch snippet must run standalone (copy-paste into a notebook, no edits).
- Python is always invoked through `uv run` in the project directory — never `python3` directly.
- Target length ≈ 3,300 words ≈ "18 min read". If final word count lands outside 2,900–3,700, adjust `readTime` (words ÷ 185, rounded).
- Out of scope: capacity factor, expert parallelism, token dropping, exact auxiliary-loss formulas, fine-tuning/serving MoE.

---

### Task 1: Post skeleton, registration, Sections 1–2

**Files:**
- Create: `public/posts/2026-07-03-mixture-of-experts-for-beginners.md`
- Modify: `src/data/posts/markdownImports.js:21-26`

**Interfaces:**
- Produces: the post file with frontmatter and Sections 1–2; the post registered so the site serves it. Later tasks append sections to this same file, keeping the exact `##` section titles defined here and in the spec.

- [ ] **Step 1: Create the post file with frontmatter and H1**

Exact frontmatter (copy verbatim):

```markdown
---
title: "Mixture of Experts: How LLMs Get Bigger Without Getting Slower"
date: "2026-07-03"
author: "Tai Vu"
excerpt: "Why is Mixtral 8x7B almost 47B parameters but runs like a 13B model? A beginner-friendly tour of MoE: routers, experts, and the trade-offs behind today's biggest LLMs."
tags: ["LLM", "AI", "MoE", "Architecture"]
readTime: "18 min read"
featured: true
slug: "mixture-of-experts-explained"
---

# 🧠 Mixture of Experts: How LLMs Get Bigger Without Getting Slower
```

- [ ] **Step 2: Write Section 1 — `## 1. The Scaling Dilemma` (~350 words)**

Required content, in this order:
1. Open with the empirical rule: bigger LLMs are smarter (scaling laws, one sentence, no math).
2. Explain the dense-model cost: in a dense transformer, **every token activates every parameter** — generating one token with a 70B model means ~70B parameters' worth of matrix math, every time. Inference cost grows linearly with model size.
3. State the dilemma plainly: "We want the knowledge of a huge model with the speed of a small one."
4. End with the hook question, verbatim: "Mixtral 8x7B has almost **47 billion parameters**, yet generating a token costs about as much as a **13B dense model**. How is that possible? That trick is called **Mixture of Experts (MoE)**, and it's the architecture behind many of today's frontier models."
5. Close with a one-line roadmap of the post (analogy → mechanics → code → costs → real models).

- [ ] **Step 3: Write Section 2 — `## 2. A Hospital, Not a General Practitioner` (~350 words)**

Required content:
1. The analogy: a giant clinic where **one general practitioner** must read every textbook to treat every patient (dense model) vs a **hospital** with a receptionist who looks at each patient and sends them to 2 of its 8 specialists (MoE). The hospital "knows" far more in total, but each patient only pays for two consultations.
2. Map analogy → architecture explicitly, as a short list: patient = token; receptionist = **router**; specialists = **experts**; "send to 2 of 8" = **top-k routing** with k=2.
3. Mandatory caveat (its own short paragraph, bolded lead): **real experts are not human-interpretable specialists.** There is no "math expert" or "French expert"; specialization emerges at the token level (punctuation, code syntax, etc.) and mostly looks meaningless to humans. Cite that the Mixtral paper itself found no obvious topic specialization.
4. Transition sentence into the mechanics section.

- [ ] **Step 4: Register the post in `markdownImports.js`**

In `src/data/posts/markdownImports.js`, add the new file at the top of the array:

```js
// List of markdown files to import
export const markdownFiles = [
  '2026-07-03-mixture-of-experts-for-beginners.md',
  '2026-06-29-from-rlhf-to-dpo.md',
  '2026-06-26-llm-engineer-journey-scratch-to-laptop.md',
  '2025-11-03-LLM-Post-Training-Part-1.md',
  '2025-08-20-attention-mechanism-in-llm.md',
];
```

- [ ] **Step 5: Verify the file is served**

Run (dev server in background, then fetch):

```bash
cd "/Users/taivu/Library/CloudStorage/GoogleDrive-vdtai94@gmail.com/My Drive/Project/taivu7.github.io"
(BROWSER=none npm start &> /tmp/cra.log &) && sleep 25
curl -s http://localhost:3000/posts/2026-07-03-mixture-of-experts-for-beginners.md | head -5
```

Expected: the YAML frontmatter starting with `---` and `title: "Mixture of Experts…`. Kill the server afterwards (`kill %1` or `pkill -f react-scripts`).

- [ ] **Step 6: Commit**

```bash
git add public/posts/2026-07-03-mixture-of-experts-for-beginners.md src/data/posts/markdownImports.js
git commit -m "Draft MoE blog post: intro + hospital analogy; register post"
```

---

### Task 2: Section 3 — Inside an MoE Layer

**Files:**
- Modify: `public/posts/2026-07-03-mixture-of-experts-for-beginners.md` (append)

**Interfaces:**
- Consumes: post file from Task 1; section must end setting up the code section that follows.
- Produces: `## 3. Inside an MoE Layer` — the conceptual model (router, experts, top-k, gating) that Task 3's code implements with the same vocabulary (`router`, `experts`, `top_k`, `weights`).

- [ ] **Step 1: Write `## 3. Inside an MoE Layer` (~700 words)**

Required content, in this order:
1. **FFN recap** (one paragraph): each transformer block = attention + a feed-forward network (FFN); link the attention half to the existing post: `[attention mechanism](/blog/getting-insight-attention)`. Key fact: the FFN holds roughly **two-thirds of a transformer's parameters** — which is why MoE targets the FFN, not attention.
2. **The swap**: an MoE layer replaces the single FFN with **N independent FFNs (the experts)** plus a small **router** (a single linear layer producing one score per expert).
3. **The forward pass for one token**, as a numbered walk-through: (a) router scores all N experts; (b) keep the **top-k** (Mixtral: k=2 of N=8); (c) softmax over just those k scores → weights; (d) token goes through only those k experts; (e) output = weighted sum. Everything else about the transformer stays identical.
4. **The arithmetic** (its own subsection `### Total vs Active Parameters`): Mixtral 8x7B has 46.7B total parameters, but each token touches only ~12.9B (top-2 of 8 experts, plus shared attention/embeddings). Note the name "8x7B" is misleading — it is NOT 8 separate 7B models glued together; attention layers are shared, only FFNs are replicated. Sparsity = you store all 47B but compute with ~13B per token.
5. One clarifying sentence: routing is **per token, per layer** — the same token can see different experts at layer 5 and layer 20.
6. Transition: "This sounds complicated. In code, it's about forty lines."

- [ ] **Step 2: Commit**

```bash
git add public/posts/2026-07-03-mixture-of-experts-for-beginners.md
git commit -m "Add MoE post section 3: router, experts, top-k, total-vs-active math"
```

---

### Task 3: Section 4 — Minimal PyTorch MoE layer

**Files:**
- Modify: `public/posts/2026-07-03-mixture-of-experts-for-beginners.md` (append)
- Test: `/private/tmp/claude-501/-Users-taivu-Library-CloudStorage-GoogleDrive-vdtai94-gmail-com-My-Drive-Project-taivu7-github-io/a8dee085-f1d7-4373-852c-0d8a333f4988/scratchpad/moe_check.py`

**Interfaces:**
- Consumes: Section 3's vocabulary (`router`, `experts`, `top_k`).
- Produces: `## 4. Code: A Minimal MoE Layer in PyTorch` containing the exact snippet below, verified to run.

- [ ] **Step 1: Write the verification script first**

Save this exact code to the scratchpad as `moe_check.py` (it is also the code block that goes in the post):

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    """A standard Transformer FFN: expand, activate, project back."""
    def __init__(self, d_model, d_hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_model),
        )

    def forward(self, x):
        return self.net(x)

class MoELayer(nn.Module):
    def __init__(self, d_model=64, d_hidden=256, n_experts=8, top_k=2):
        super().__init__()
        self.experts = nn.ModuleList(
            [Expert(d_model, d_hidden) for _ in range(n_experts)]
        )
        self.router = nn.Linear(d_model, n_experts)
        self.top_k = top_k

    def forward(self, x):
        # x: (n_tokens, d_model)
        scores = self.router(x)                        # (n_tokens, n_experts)
        top_scores, top_idx = scores.topk(self.top_k, dim=-1)
        weights = F.softmax(top_scores, dim=-1)        # (n_tokens, top_k)

        out = torch.zeros_like(x)
        for token in range(x.shape[0]):                # naive loop, for clarity
            for k in range(self.top_k):
                expert = self.experts[top_idx[token, k]]
                out[token] += weights[token, k] * expert(x[token])
        return out

moe = MoELayer()
tokens = torch.randn(4, 64)   # a tiny "sentence": 4 tokens, d_model=64
output = moe(tokens)
print(output.shape)           # torch.Size([4, 64])
```

- [ ] **Step 2: Run it to verify it works**

```bash
cd "/Users/taivu/Library/CloudStorage/GoogleDrive-vdtai94@gmail.com/My Drive/Project/taivu7.github.io"
uv run --with torch python "/private/tmp/claude-501/-Users-taivu-Library-CloudStorage-GoogleDrive-vdtai94-gmail-com-My-Drive-Project-taivu7-github-io/a8dee085-f1d7-4373-852c-0d8a333f4988/scratchpad/moe_check.py"
```

Expected output: `torch.Size([4, 64])`. If `uv` cannot resolve torch, stop and report — do not fall back to system python3.

- [ ] **Step 3: Write `## 4. Code: A Minimal MoE Layer in PyTorch` (~600 words incl. code)**

Required content:
1. One-paragraph setup: "Here is a complete MoE layer. It runs as-is — paste it into a notebook."
2. The verified code block from Step 1, verbatim, tagged `python`.
3. A guided read of the code, mapping each piece back to Section 3: `Expert` is just a normal FFN; `self.router` is a single `nn.Linear` — the receptionist is *tiny*; `topk` + `softmax` = pick 2 specialists and weight their opinions; the double loop = each token visits only its 2 experts.
4. Trace one token through the layer in prose with concrete invented numbers (e.g. "router scores `[2.1, -0.3, 0.8, …]` → top-2 = experts 0 and 2 → weights `[0.79, 0.21]` → output = 0.79·E0(x) + 0.21·E2(x)").
5. Honesty note (bold lead): **this naive loop is for learning, not production** — real implementations batch tokens per expert and use specialized kernels; the ideas are identical.
6. Parameter-count observation: this layer stores 8 experts' weights but each token uses 2 — the 47B-total/13B-active idea in miniature.

- [ ] **Step 4: Commit**

```bash
git add public/posts/2026-07-03-mixture-of-experts-for-beginners.md
git commit -m "Add MoE post section 4: minimal PyTorch MoE layer (verified with uv run)"
```

---

### Task 4: Sections 5–6 — training pitfalls and real models

**Files:**
- Modify: `public/posts/2026-07-03-mixture-of-experts-for-beginners.md` (append)

**Interfaces:**
- Consumes: the total-vs-active thread from Sections 1 and 3.
- Produces: `## 5. The Catch: Training Is Tricky` and `## 6. MoE in the Wild`, including the model comparison table Task 5's recap refers back to.

- [ ] **Step 1: Write `## 5. The Catch: Training Is Tricky` (~500 words)**

Required content:
1. **Router collapse** ("rich get richer"): early in training the router slightly prefers one expert → that expert gets more data → gets better → gets picked more. End state: 2 overworked experts, 6 dead ones — you paid for 8 and got 2.
2. **The fix at intuition level**: an **auxiliary load-balancing loss** — a small penalty added to the training loss that rewards spreading tokens evenly across experts. Explicitly say we're skipping the formula; the idea is "a tax on favoritism."
3. **The VRAM misconception** (its own subsection `### MoE Saves Compute, Not Memory`, bold lead): every expert must sit in GPU memory even when idle — you can't know which experts the next token needs until the router runs. Mixtral computes like a 13B model but must be *loaded* like a 47B model. One sentence on why this matters to the reader: it's why you can't run Mixtral on a GPU that fits a 13B model.
4. Brief mention (one sentence each, no depth): MoE training can be less stable than dense training, and fine-tuning MoE is fiddlier.

- [ ] **Step 2: Write `## 6. MoE in the Wild` (~550 words)**

Required content:
1. One history paragraph: the idea is old — Jacobs, Jordan et al., **"Adaptive Mixtures of Local Experts" (1991)**; revived at scale by Shazeer et al. 2017 (**"Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer"**, 137B-parameter LSTM-based MoE); brought into transformers by **Switch Transformer** (Fedus et al. 2021, simplified to top-1 routing, scaled to 1.6T parameters).
2. This exact comparison table:

```markdown
| Model | Year | Total params | Active per token | Experts / routing |
|---|---|---|---|---|
| Switch Transformer | 2021 | up to 1.6T | ~one expert's worth | top-1 |
| Mixtral 8x7B | 2023 | 46.7B | 12.9B | 8 experts, top-2 |
| DeepSeek-V3 | 2024 | 671B | 37B | 256 fine-grained + 1 shared, top-8 |
| GPT-4 (rumored) | 2023 | undisclosed | undisclosed | widely rumored MoE, unconfirmed |
```

3. One short paragraph on DeepSeek-V3's twist for the curious reader: many **small fine-grained experts** plus one **shared expert** every token visits (common knowledge lives in the shared expert; the router handles the rest).
4. GPT-4 framing: leaks have long claimed it is MoE; OpenAI never confirmed — present strictly as rumor.
5. Closing trend sentence: the pattern across frontier labs is clear — when you want more capability per inference dollar, MoE is the default answer.

- [ ] **Step 3: Commit**

```bash
git add public/posts/2026-07-03-mixture-of-experts-for-beginners.md
git commit -m "Add MoE post sections 5-6: load balancing, VRAM caveat, real-world models"
```

---

### Task 5: Section 7, final pass, render verification

**Files:**
- Modify: `public/posts/2026-07-03-mixture-of-experts-for-beginners.md`

**Interfaces:**
- Consumes: the full post from Tasks 1–4 and the hook posed in Section 1.
- Produces: the finished, verified, published post.

- [ ] **Step 1: Write `## 7. Recap + What to Read Next` (~300 words)**

Required content:
1. Bullet takeaways, each one line, that together answer the Section 1 hook:
   - MoE replaces one big FFN with many experts + a tiny router; each token uses only the top-k.
   - Total parameters (what you store) ≠ active parameters (what you compute) — that's the whole trick.
   - Mixtral: 47B stored, ~13B computed per token — big-model knowledge at mid-model speed.
   - The price: routers need load-balancing pressure to train well, and all experts must fit in memory.
   - Experts specialize, but not in human-legible topics.
2. Reading list with these exact links:
   - Shazeer et al. 2017 — https://arxiv.org/abs/1701.06538
   - Switch Transformer (Fedus et al. 2021) — https://arxiv.org/abs/2101.03961
   - Mixtral of Experts (Jiang et al. 2024) — https://arxiv.org/abs/2401.04088
   - DeepSeek-V3 Technical Report — https://arxiv.org/abs/2412.19437
   - Hugging Face — "Mixture of Experts Explained" — https://huggingface.co/blog/moe

- [ ] **Step 2: Full-post proofread pass**

Check, fixing inline:
- Section numbering 1–7 matches the spec titles exactly; `---` separators between sections; emoji only on the H1.
- The internal link `/blog/getting-insight-attention` appears in Section 3.
- Numbers are consistent everywhere (46.7B / 12.9B / 8 experts / top-2 — no drift like "45B" or "13B active" vs "12.9B" mixing).
- No section drifted into out-of-scope territory (capacity factor, expert parallelism, aux-loss formulas).

- [ ] **Step 3: Recalculate readTime**

```bash
cd "/Users/taivu/Library/CloudStorage/GoogleDrive-vdtai94@gmail.com/My Drive/Project/taivu7.github.io"
wc -w public/posts/2026-07-03-mixture-of-experts-for-beginners.md
```

Expected ~3,300 words. If outside 2,900–3,700, set `readTime` to words ÷ 185 rounded (e.g. 4,000 words → "22 min read") and update the frontmatter.

- [ ] **Step 4: Verify rendering end-to-end**

```bash
cd "/Users/taivu/Library/CloudStorage/GoogleDrive-vdtai94@gmail.com/My Drive/Project/taivu7.github.io"
(BROWSER=none npm start &> /tmp/cra.log &) && sleep 25
curl -s http://localhost:3000/posts/2026-07-03-mixture-of-experts-for-beginners.md | wc -l
```

Expected: full line count of the post (> 200). Then open `http://localhost:3000/blog/mixture-of-experts-explained` in a browser (or via the claude-in-chrome tools) and confirm: the post renders, the code block is highlighted, the table displays, and the post appears at the top of the blog list at `/blog`. Kill the dev server afterwards.

- [ ] **Step 5: Final commit**

```bash
git add public/posts/2026-07-03-mixture-of-experts-for-beginners.md
git commit -m "Finish MoE post: recap, reading list, readTime; verified rendering"
```
