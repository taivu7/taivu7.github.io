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

## 1. The Scaling Dilemma

There's an empirical rule that has held up again and again in deep learning: bigger language models are smarter. Add more parameters, train on more data, and—almost like clockwork—the model gets better at almost everything, from grammar to reasoning to code. This is the essence of the **scaling laws** that have driven the last several years of LLM progress.

But bigger comes with a catch. In a standard **dense transformer**, every parameter in the model gets used on every single token you generate. There's no shortcut, no skipping: **every token activates every parameter**. Ask a 70B dense model to produce one token, and you're paying for roughly 70 billion parameters' worth of matrix multiplications—every single time, for every single token. Double the model's size, and you roughly double the compute (and the latency, and the GPU bill) it takes to generate the same piece of text. Inference cost grows linearly with model size, with no free lunch in sight.

This creates a real dilemma for anyone building these systems. We want the knowledge of a huge model with the speed of a small one. A bigger model knows more, reasons better, and makes fewer mistakes—but a bigger dense model is also proportionally slower and more expensive to run in production, where every millisecond and every GPU-second counts.

So here's a puzzle worth sitting with: Mixtral 8x7B has almost **47 billion parameters**, yet generating a token costs about as much as a **13B dense model**. How is that possible? That trick is called **Mixture of Experts (MoE)**, and it's the architecture behind many of today's frontier models.

The rest of this post builds up to a full answer: we'll start with an analogy for the intuition, walk through the actual mechanics of routing and experts, look at what an MoE layer looks like in code, weigh the real costs and trade-offs, and finish with how this plays out in models you've actually heard of.

---

## 2. A Hospital, Not a General Practitioner

Imagine two ways of organizing medical care in a small town.

In the first, there's a single **general practitioner**. To be any good, this one doctor has to have read every medical textbook ever written—cardiology, oncology, pediatrics, dermatology, all of it—because any patient who walks through the door might need any of it. Every visit, the doctor draws on the full weight of everything they know, even if the patient just needs a flu diagnosis. This is your standard **dense model**: one set of parameters, and every input passes through all of it.

In the second, there's a **hospital**. A patient walks in and is met by a receptionist who takes one look at the symptoms and sends them down the hall to exactly the right department—say, two of the hospital's eight specialists: a cardiologist and a general surgeon, perhaps. The hospital as a whole "knows" far more than the lone GP ever could, because it employs eight deep specialists instead of one generalist. But any given patient only ever pays for two consultations, not eight. This is **Mixture of Experts**.

Mapping the analogy onto the architecture is refreshingly literal:

- **Patient** → a token being processed
- **Receptionist** → the **router**, a small learned component that looks at each token and decides where to send it
- **Specialists** → the **experts**, separate blocks of parameters that each process the tokens routed to them
- **"Send to 2 of the hospital's 8 specialists"** → **top-k routing**, where the router picks the top `k` experts for each token (here, k=2 out of 8)

**One important caveat before we go further: real experts are not human-interpretable specialists.** It's tempting to imagine expert #3 as "the math expert" and expert #7 as "the French expert," but that's not what actually emerges from training. There is no expert that cleanly owns a topic or a language. Specialization, when it happens, tends to show up at the level of surface patterns—punctuation, code syntax, whitespace, certain token types—and mostly looks meaningless when you go looking for a human-readable label. The Mixtral paper itself examined its own experts for topic-based specialization and found none; routing decisions didn't line up with subject matter in any way a human could easily name.

With the analogy and its limits both on the table, it's time to open the hood and see how a router and a set of experts actually work inside a transformer layer.
