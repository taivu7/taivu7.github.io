# Design: Blog Post — Mixture of Experts (for LLM Beginners)

**Date:** 2026-07-03
**Status:** Approved by user

## Goal

Publish a beginner-friendly blog post explaining Mixture of Experts (MoE),
consistent in style with the existing posts (Attention, RLHF→DPO): ground-up
explanations, concrete examples, minimal runnable PyTorch code.

**Audience:** people just starting to learn about Large Language Models.
They know roughly what a Transformer is (the Attention post is a prerequisite
we can link to) but nothing about MoE.

**Narrative spine:** the scaling dilemma — bigger dense models are smarter but
every token pays for every parameter. MoE is the "big but not slow" answer.
The recurring thread is *total parameters vs active parameters*, anchored by
the hook: "Mixtral 8x7B has ~47B parameters but runs like a 13B model — how?"

## File & Metadata

- **Post file:** `public/posts/2026-07-03-mixture-of-experts-for-beginners.md`
- **Registration:** add the filename to `markdownFiles` in
  `src/data/posts/markdownImports.js` (this is the entire publish mechanism).
- **Frontmatter:**
  - title: "Mixture of Experts: How LLMs Get Bigger Without Getting Slower"
  - date: "2026-07-03"
  - author: "Tai Vu"
  - excerpt: "Why is Mixtral 8x7B almost 47B parameters but runs like a 13B
    model? A beginner-friendly tour of MoE: routers, experts, and the
    trade-offs behind today's biggest LLMs."
  - tags: ["LLM", "AI", "MoE", "Architecture"]
  - readTime: "18 min read"
  - featured: true
  - slug: "mixture-of-experts-explained"
- **Language:** English. H1 carries an emoji, matching the RLHF→DPO post.

## Content Outline (7 sections)

1. **The Scaling Dilemma** — Dense models: every token activates every
   parameter, so inference cost grows linearly with size. Pose the Mixtral
   hook as the question the post will answer.
2. **A Hospital, Not a General Practitioner** — Analogy: a receptionist
   (router) sends each patient (token) to the right specialists (experts)
   instead of every doctor examining every patient. Include the caveat that
   real experts do NOT specialize along human-interpretable topics.
3. **Inside an MoE Layer** — Quick FFN recap (internal link to the Attention
   post). Replace one FFN with N experts + a router; top-k selection; softmax
   gating; weighted sum of expert outputs. Work the total-vs-active parameter
   arithmetic with real Mixtral numbers (8 experts, top-2, ~47B total,
   ~13B active per token).
4. **Code: A Minimal MoE Layer in PyTorch** — ~50 lines: `Expert` (small MLP),
   router (linear + top-k), `MoELayer.forward`. Trace one token through the
   layer by hand. State clearly this is a naive-loop teaching version, not a
   production kernel. Code must run standalone (copy-paste into a notebook).
5. **The Catch: Training Is Tricky** — Router collapse ("rich get richer");
   auxiliary load-balancing loss at intuition level only (no deep math).
   Common misconception to correct: MoE saves *compute*, not *VRAM* — all
   experts must sit in memory.
6. **MoE in the Wild** — One short history paragraph (Shazeer 2017 sparse MoE,
   Switch Transformer), then a comparison table: Mixtral 8x7B, DeepSeek-V3
   (fine-grained + shared experts), GPT-4 (rumored). Point out the trend:
   most frontier models today are MoE.
7. **Recap + What to Read Next** — Bullet takeaways that answer the opening
   hook; reading list: Shazeer 2017, Switch Transformer paper, Mixtral paper,
   Hugging Face MoE blog post.

## Success Criteria

- A reader new to MoE can answer after reading: what MoE is, how
  router/experts/top-k work, why "47B params but 13B speed", and the costs
  (training instability, VRAM footprint).
- The PyTorch snippet runs standalone without edits.
- The post renders correctly on the local blog (`npm start`), appears in the
  post list, and opens at its slug.

## Out of Scope

- Deep training engineering: capacity factor, expert parallelism, token
  dropping details, exact auxiliary-loss formulas.
- Fine-tuning or serving MoE models.
- Any site code changes beyond registering the post.
