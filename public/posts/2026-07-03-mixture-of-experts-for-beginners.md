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

---

## 3. Inside an MoE Layer

Every block in a transformer is made of two halves: an **attention** sub-layer and a **feed-forward network (FFN)** sub-layer. Attention is where tokens look at each other and exchange information—that's the part covered in our post on the [attention mechanism](/blog/getting-insight-attention). The FFN is the other half, and it's the one that matters here: it processes each token independently (no cross-token communication at all, unlike attention), usually expanding a token's vector up to a wider hidden dimension and projecting it back down. Simple as that sounds, the FFN typically holds roughly **two-thirds of a transformer's total parameters**. Attention layers, by comparison, are relatively small. If you want to make a model dramatically bigger without touching how tokens communicate with each other, the FFN is the natural place to do it—and that's exactly what MoE targets.

**The swap is simple to state.** Instead of one FFN that every token has to pass through, an MoE layer has **N independent FFNs**—the experts—sitting side by side, plus one small **router**: a single linear layer that looks at a token and outputs one score per expert. Attention stays untouched. Only the FFN gets replicated and gated.

**Here's what happens to a single token as it reaches an MoE layer:**

1. The router scores every one of the N experts for that token.
2. Only the **top-k** highest-scoring experts are kept—everyone else is discarded for this token. Mixtral uses k=2 out of N=8.
3. A softmax is applied over just those k scores, turning them into weights that sum to 1.
4. The token is sent through only those k experts—each one runs its full FFN computation on it.
5. The final output is the weighted sum of the k experts' outputs, using the softmax weights from step 3.

Concretely: say expert 2 scores highest and expert 5 scores second-highest for a token, with softmax weights of 0.7 and 0.3. The token is run through expert 2's FFN and expert 5's FFN—full computation, not a shortcut version—and the layer's output is `0.7 × expert_2_output + 0.3 × expert_5_output`. Experts 0, 1, 3, 4, 6, and 7 never see this token at this layer.

Everything else about the transformer—residual connections, layer norms, the attention sub-layer, the overall stack of blocks—stays exactly as it would in a dense model. The only thing that changed is what happens inside the FFN slot.

### Total vs Active Parameters

This is where the arithmetic behind our opening puzzle lives. Mixtral 8x7B has **46.7B total parameters**, but a given token only touches about **12.9B active parameters** as it flows through the model—top-2 of its 8 experts per layer, plus the attention layers and embeddings, which are shared by every token.

The name "8x7B" is a bit of a trap: it's tempting to read it as "eight separate 7B models stapled together," but that's not what's going on. Attention layers, embeddings, and the router are **shared**—there's exactly one copy of each. Only the FFN is replicated into 8 experts per layer. So you don't get 8×7B = 56B of unique parameters; you get 46.7B, because the non-FFN parts aren't duplicated at all.

This is **sparsity** in a nutshell: you store all ~47B parameters on disk and in GPU memory, but you only compute with ~13B of them for any given token. The rest of the experts sit there unused for that token—cheap to hold, but never touched on that pass.

One more wrinkle worth flagging: routing is **per token, per layer**. There's no single "path" a token takes through the whole model. The same token might be sent to experts 2 and 5 at layer 5, then experts 1 and 7 at layer 20. Every layer makes its own independent routing decision, for every token, every time.

This sounds complicated. In code, it's about forty lines.

---

## 4. Code: A Minimal MoE Layer in PyTorch

Here is a complete MoE layer. It runs as-is—paste it into a notebook, no edits required.

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

**Reading it against Section 3, piece by piece:**

- `Expert` is nothing exotic—it's just a normal transformer FFN: expand to `d_hidden`, apply an activation, project back down to `d_model`. This is the "specialist," and there's nothing specialist-looking about its code.
- `self.router` is a single `nn.Linear(d_model, n_experts)`. That's it. The receptionist deciding where every token goes is one small matrix—tiny compared to the eight FFNs sitting next to it.
- `scores.topk(self.top_k, dim=-1)` followed by `F.softmax` is "pick 2 specialists and weight their opinions": `topk` selects the winning experts, `softmax` turns their two scores into weights that sum to 1.
- The double `for` loop is the literal hospital hallway: each token visits only its `top_k` experts, not all eight. The outer loop walks tokens; the inner loop walks that token's two chosen specialists.

**Tracing one token through the layer**, with invented numbers: suppose the router scores token 0 across all 8 experts as `[2.1, -0.3, 0.8, 1.5, -1.1, 0.2, 0.4, -0.6]`. The top-2 are expert 0 (score 2.1) and expert 3 (score 1.5). Softmax over just those two scores gives weights `[0.65, 0.35]`. The token runs through expert 0's full FFN and expert 3's full FFN, and the layer's output for that token is `0.65 · expert_0(x) + 0.35 · expert_3(x)`. Experts 1, 2, 4, 5, 6, and 7 never touch this token.

**One honesty note: this naive loop is for learning, not production.** Looping over tokens one at a time in Python is slow, and no real MoE implementation does it this way. Production kernels group tokens by which expert they were routed to, batch each group into a single matrix multiply, and run on specialized, highly optimized kernels. The mechanics—route, select top-k, weight, combine—are identical; only the execution strategy changes for speed.

Even at this toy scale, the parameter story already shows up: `MoELayer()` above allocates and stores weights for all 8 experts, but any single token's `forward` pass only ever calls 2 of them. Scale `n_experts` up, replicate this pattern across every FFN slot in a big transformer, and you get exactly the shape of Mixtral's **~47B total, ~13B active** parameters: everything is stored, only a fraction is computed per token.

---

## 5. The Catch: Training Is Tricky

Stacking eight experts behind a router sounds simple, but training that setup to actually behave is where MoE earns its reputation for being finicky.

**The first problem is router collapse—the "rich get richer" trap.** At the start of training, the router's weights are close to random, so its preferences for one expert over another are just noise. But noise is enough to get the loop started. Say the router happens to lean slightly toward expert 2 early on. Expert 2 now sees more tokens than its neighbors, so it gets more gradient updates, so it gets a little better at its job, so the router—which is learning to send tokens wherever they're handled best—leans toward it even harder next step. That feedback loop compounds. Left unchecked, training can converge to a state where 2 of the 8 experts do nearly all the work and the other 6 sit there barely trained, contributing almost nothing. You paid for 8 experts and got 2. It's exactly the failure mode the hospital analogy warns about: a receptionist who, for no good reason, keeps sending every patient to the same two doctors while six specialists' offices gather dust.

**The fix, at the level of intuition, is an auxiliary load-balancing loss.** We're skipping the formula—it's not needed to understand what it does. Think of it as a tax on favoritism: alongside the model's normal training objective (predict the next token well), you add a small penalty that rises when routing gets lopsided and falls when tokens are spread roughly evenly across experts. The model is now optimizing two things at once—be accurate, and don't play favorites—and that second nudge is usually enough to keep all 8 experts alive and useful.

### MoE Saves Compute, Not Memory

**Sparsity cuts compute, but it does nothing for memory.** Every expert, including the ones a given token never visits, has to sit resident in GPU memory at all times—the router doesn't decide which experts a token needs until that token actually arrives at the layer, and by then there's no time to page an unused expert in from disk. Mixtral computes like a 13B model but must be *loaded* like a 47B model, all 46.7B parameters, all the time. That's the practical reason you can't run Mixtral on a GPU sized for a 13B dense model: the FLOPs are cheap, but the VRAM bill is for the whole hospital, not just the two doctors on duty for this patient.

Two more caveats worth a sentence each, not a deep dive: MoE training is often reported as **less stable** than dense training, with a higher risk of loss spikes if the balancing act above goes wrong. And **fine-tuning a pretrained MoE is fiddlier** than fine-tuning a dense model of similar active size, since small routing shifts during fine-tuning can quietly starve experts that were fine at pretraining time.

---

## 6. MoE in the Wild

**The idea behind MoE is not new.** Robert Jacobs and Michael Jordan, in "Adaptive Mixtures of Local Experts" (1991), described training separate small networks alongside a gating network that learns to route inputs among them—the receptionist-and-specialists structure from Section 2, more than three decades before Mixtral existed. Back then, "experts" were tiny by today's standards and the whole setup ran on hardware nowhere near capable of today's LLMs, so the idea mostly sat dormant for years. It was revived at massive scale by Shazeer et al. in "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer" (2017), which built a 137B-parameter, LSTM-based MoE and showed that sparsity could work at genuinely huge scale, well before transformers were the dominant architecture. Transformers got their own MoE treatment with the **Switch Transformer** (Fedus et al., 2021), which simplified routing down to **top-1**—one expert per token, instead of top-2 or more—and used that simplicity to scale all the way to **1.6 trillion parameters**.

Seeing the actual numbers side by side makes the total-vs-active split from Section 3 concrete:

| Model | Year | Total params | Active per token | Experts / routing |
|---|---|---|---|---|
| Switch Transformer | 2021 | up to 1.6T | ~one expert's worth | top-1 |
| Mixtral 8x7B | 2023 | 46.7B | 12.9B | 8 experts, top-2 |
| DeepSeek-V3 | 2024 | 671B | 37B | 256 fine-grained + 1 shared, top-8 |
| GPT-4 (rumored) | 2023 | undisclosed | undisclosed | widely rumored MoE, unconfirmed |

DeepSeek-V3's numbers look strange at first—671B total but only 37B active is an even sparser ratio than Mixtral's—until you see the twist. Instead of a handful of large experts, DeepSeek-V3 uses many small, **fine-grained experts**, plus one always-on **shared expert** that every single token passes through no matter what the router decides. The idea is division of labor at a finer grain: common knowledge that basically every token needs lives permanently in the shared expert, so the router's job simplifies to picking out the specialized fine-grained experts a token actually needs on top of that baseline, rather than every routed expert reinventing the same common patterns.

And then there's GPT-4. For a while now, leaks and technical reporting have claimed GPT-4 uses a Mixture-of-Experts architecture, and the rumor has circulated widely enough that many people now treat it as settled fact. It isn't. OpenAI has never confirmed any of it, and no official numbers exist for GPT-4's total or active parameter counts. Treat the row above exactly as labeled: **widely rumored, not verified.**

Look across this table and a pattern emerges: Switch Transformer, Mixtral, DeepSeek-V3, and (if the rumors hold) GPT-4 are four different labs, four different years, converging on the same trick. When you want more capability per inference dollar, MoE is the default answer.

---

## 7. Recap + What to Read Next

Section 1 asked how Mixtral packs almost 47 billion parameters of knowledge into a model that runs like a 13B one. Here's the short answer, five bullets:

- **MoE replaces one big FFN with many experts plus a tiny router**, and each token only passes through the top-k of them—not all of them.
- **Total parameters and active parameters are different numbers.** Total is what you store on disk and in GPU memory; active is what you actually compute for a given token. That gap is the whole trick.
- **Mixtral: 46.7B total, ~12.9B active per token**—top-2 of 8 experts, per layer. Big-model knowledge, mid-model compute.
- **The price shows up in two places.** Training needs load-balancing pressure so the router doesn't collapse onto a favorite handful of experts, and every one of the 8 experts—used or not, on any given token—still has to sit resident in memory at all times.
- **Experts specialize, but not into anything human-legible.** There's no "math expert" or "French expert" hiding in there; specialization tends to show up in surface patterns, not clean topics you could name.

If you want to go past a beginner tour, these are the sources this post draws on:

- Shazeer et al. 2017 — https://arxiv.org/abs/1701.06538
- Switch Transformer (Fedus et al. 2021) — https://arxiv.org/abs/2101.03961
- Mixtral of Experts (Jiang et al. 2024) — https://arxiv.org/abs/2401.04088
- DeepSeek-V3 Technical Report — https://arxiv.org/abs/2412.19437
- Hugging Face — "Mixture of Experts Explained" — https://huggingface.co/blog/moe
