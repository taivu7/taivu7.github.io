---
title: "From RLHF to DPO: A Simpler Way to Align Language Models"
date: "2026-06-29"
author: "Tai Vu"
excerpt: "How do you turn 'humans prefer answer A over B' into a training signal? A ground-up tour of RLHF, the insight behind DPO, and how to choose between them."
tags: ["LLM", "AI", "Alignment", "RLHF", "DPO"]
readTime: "14 min read"
featured: true
slug: "from-rlhf-to-dpo"
---

# 🎯 From RLHF to DPO: A Simpler Way to Align Language Models

## 1. Introduction — The Alignment Problem

A pretrained language model is a strange kind of genius. It has read a sizeable fraction of the internet and absorbed grammar, facts, code, and reasoning patterns along the way. But ask it a question and it might answer with another question. Ask it for help and it might cheerfully complete your sentence instead. It knows a great deal—it just has no manners.

To see the gap, imagine asking a base model: *"How do I fix my car?"*

A well-pretrained but unaligned model might reply: *"How do I fix my bike? How do I fix my sink?"* It has pattern-matched your sentence to a list of similar-looking questions, because that is exactly what "predict the next token" rewards. It was never trained to be *helpful*—only to be *plausible*.

What we actually want is something like: *"I'd be happy to help! What symptoms are you seeing—strange noises, warning lights, trouble starting?"*

The gap between those two responses is the **alignment problem**. We have a model full of raw capability but with no sense of what a *good* response looks like. To close the gap, we need a way to communicate human preferences to the model during training—and that turns out to be harder than it sounds. The whole post hangs on a single deceptively simple question:

> **How do you turn "humans prefer answer A over B" into a training signal?**

There are two famous answers. The first is **RLHF** (Reinforcement Learning from Human Feedback)—the classic, powerful, and somewhat painful approach that aligned ChatGPT. The second is **DPO** (Direct Preference Optimization)—a lighter-weight method that reaches a similar destination without the reinforcement learning machinery.

This post walks from the problem to both solutions, intuition first and formulas second, so lighter readers can stay aboard the whole way. By the end you'll understand not just *how* DPO works, but *why* it works—and when you should reach for it instead of RLHF.

## 2. RLHF — The Classic Solution

RLHF is the method that made ChatGPT possible. It's powerful and it scales—and it's also genuinely complicated to run. The whole thing is best understood as an assembly line with **three stages**, each feeding the next. We start with a model that can talk, teach a second model to *judge* answers, then use that judge to push the first model toward better behavior.

Here's the bird's-eye view before we zoom in:

```
   Stage 1: SFT                Stage 2: Reward Model            Stage 3: PPO (RL)
┌──────────────────┐        ┌────────────────────────┐      ┌────────────────────────┐
│ Base model       │        │ Preference pairs        │      │ Policy generates an     │
│   + demonstrations│  ──▶  │   (chosen yᵂ, reject yᴸ)│ ──▶ │   answer                │
│ → SFT model      │        │ Train a Reward Model    │      │ RM scores it            │
│ (learns to answer│        │   that outputs a single │      │ PPO updates the policy  │
│  in the right    │        │   quality score         │      │   to score higher       │
│  format)         │        │ → automatic "judge"     │      │ ↑ KL penalty keeps it   │
│                  │        │                         │      │   close to the SFT model│
└──────────────────┘        └────────────────────────┘      └────────────────────────┘
```

Let's walk each stage.

### 2.1 — Stage 1: Supervised Fine-Tuning (SFT)

Before you can teach taste, you have to teach *format*. Remember our base model from the intro that answered "How do I fix my car?" with another question? Stage 1 fixes exactly that.

We take the pretrained base model and fine-tune it on a curated set of high-quality **prompt → response demonstrations**: examples written (or vetted) by humans showing what a good assistant reply looks like. "How do I fix my car?" → "I'd be happy to help! What symptoms are you seeing?" Thousands of these, across many topics.

The model isn't learning new facts here—it already knows them from pretraining. It's learning a *behavior*: when you see a prompt, produce a helpful, on-topic answer instead of echoing or drifting. The output of this stage is the **SFT model**, and it's the foundation everything else builds on. Every later stage starts from this checkpoint—it's both our starting policy and, later, our anchor.

SFT alone already gets you a usable assistant. But it has a ceiling: it can only imitate the demonstrations it was shown, and for most real questions there's no single "correct" demonstration to write down. To push past imitation, we need a way to express *preferences*—and that's Stage 2.

### 2.2 — Stage 2: Training a Reward Model

Here's a problem we have to solve before we can improve the model: **how do you tell a model that one answer is better than another?**

The naive idea is to score each answer on an absolute scale—"rate this response from 1 to 10." But humans are surprisingly bad at this. Ask the same person to score one essay in isolation and you'll get 7 on Monday and 5 on Friday. We have no stable internal ruler. What humans *are* good at is **comparison**: show two answers side by side and ask "which is better?" and the judgments become fast and consistent. (This intuition gets its own full treatment earlier in the post; here we just need the consequence.)

So the training data for this stage is **preference pairs**. For a given prompt $x$, a human is shown two candidate responses and picks a winner:

- $y^W$ — the **chosen** (preferred) response
- $y^L$ — the **rejected** response

Now the trick: we train a separate network, the **reward model** $r(x, y)$, that reads a prompt and a response and outputs a single number—a quality score. We want it to assign a *higher* score to the chosen response than to the rejected one, for every pair in our dataset.

To turn "chosen beat rejected" into a trainable loss, RLHF borrows the **Bradley–Terry model**, a classic way to convert pairwise comparisons into latent scores (the same math behind chess Elo ratings). It says the probability a human prefers $y^W$ over $y^L$ depends only on the *gap* between their scores, squashed through a sigmoid $\sigma$. Training the reward model means minimizing:

$$\mathcal{L}_{RM} = -\log \sigma\big(r(x, y^W) - r(x, y^L)\big)$$

Don't let it intimidate you—read it left to right. The term $r(x,y^W) - r(x,y^L)$ is "how much more the model scores the winner over the loser." We want that gap to be large and positive. The sigmoid + log turns "make the gap big" into a loss that's **strongest exactly when the model gets the ordering backwards** (predicts the loser is better) and nearly flat once it's confidently correct. The model spends its effort where it's wrong.

Once trained, the reward model is a stand-in for a human judge—an **automatic scorer you can query millions of times** without paying a human for each one. That scalability is the whole point: it's what makes the next stage possible.

**What this looks like in code.** You don't implement the Bradley–Terry loss by hand—libraries like Hugging Face's **TRL** package it into a `RewardTrainer`. The reward model itself is just a base LLM with its next-token head swapped for a single-number scoring head:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardTrainer, RewardConfig
import torch

model_id = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# num_labels=1 → replace the LM head with a head that emits ONE scalar: the reward.
model = AutoModelForSequenceClassification.from_pretrained(
    model_id, num_labels=1, dtype=torch.float16
)
model.config.pad_token_id = tokenizer.pad_token_id
```

The training data is exactly the preference pairs from above—each row carries a `chosen` and a `rejected` response (here, in chat format). `RewardTrainer` computes the Bradley–Terry loss internally, so training is a single call:

```python
# each row looks like: {"chosen": [...messages...], "rejected": [...messages...]}
trainer = RewardTrainer(
    model=model,
    args=RewardConfig(max_length=1024, per_device_train_batch_size=4,
                      learning_rate=1e-4, max_steps=100),
    train_dataset=dataset,
    processing_class=tokenizer,
)
trainer.train()   # minimizes  −log σ( r(x, chosen) − r(x, rejected) )
```

That comment on the last line *is* the formula we just wrote, turned into a running loop. In a small hands-on run this reaches around **95% pairwise accuracy**—the reward model reliably ranks the better answer above the worse one, which is all we need from a judge. This part is easy; the pain is entirely in what comes next.

### 2.3 — Stage 3: Optimizing the Policy with PPO

Now we have a judge. Time to use it.

The SFT model becomes our **policy** $\pi$—the thing we're actually training. The loop is conceptually simple: the policy generates a response, the reward model scores it, and we nudge the policy's weights to make high-scoring responses more likely. This is reinforcement learning: learning from a reward signal rather than from labeled targets. The standard algorithm here is **PPO (Proximal Policy Optimization)**.

**The KL leash.** There's an immediate danger. If we optimize *purely* for reward, the policy will discover that the reward model—being just a neural net—has blind spots, and it will exploit them ruthlessly, drifting into weird, repetitive, or sycophantic text that scores high but reads terribly. To prevent this, RLHF adds a **KL-divergence penalty** that punishes the policy for straying too far from the original SFT model. The objective becomes:

$$\max_{\pi} \; \mathbb{E}\big[\, r(x, y)\,\big] - \beta \, \mathrm{KL}\big(\pi \,\|\, \pi_{ref}\big)$$

Read it as a tug-of-war: **"earn the highest reward you can, but don't wander far from the model you started with."** The reference $\pi_{ref}$ is the frozen SFT model; $\beta$ is the tension in the leash—crank it up and the policy barely moves, loosen it and the policy chases reward more aggressively (and more dangerously). Keep this KL term in mind: when we get to DPO, it makes a surprise return as the star of the show.

**What's inside PPO (just enough).** You can understand RLHF without deriving PPO, but a few moving parts explain why it's heavy:

- **Advantage** — instead of asking "was this response good?", PPO asks "was it *better than expected*?" The advantage is the reward minus a baseline. Learning from *better/worse-than-baseline* is far more stable than learning from raw scores.
- **Value (critic) model** — something has to estimate that baseline, i.e. the expected reward from a given state. That's a *second* trained network, the critic, running alongside the policy.
- **Clipped surrogate objective** — the "proximal" in PPO. Each update is mathematically *clipped* so the policy can't lurch too far in a single step. RL is unstable; this clipping is the seatbelt that keeps a single bad batch from wrecking the model.

**The loop, in pseudocode.** Putting it together, one PPO training step looks like this:

```python
for each step:
    prompts    = sample_batch()
    responses  = policy.generate(prompts)          # online sampling — EXPENSIVE
    rewards    = reward_model(prompts, responses)
    rewards    = rewards - beta * KL(policy, ref)   # KL leash, baked into the signal
    advantages = rewards - value_model(prompts)     # better-than-baseline?
    update policy   via clipped PPO objective
    update value_model
```

Notice the second line. Every single step, the policy must **generate fresh responses** to learn from—because PPO is *on-policy*: it can only learn from data the current model produces right now. You can't reuse last week's samples. That generation step is slow and expensive, and it happens continuously throughout training. Hold onto this detail; it's one of the biggest reasons people went looking for something simpler.

**What this looks like in code.** TRL wraps this whole loop in a `PPOTrainer`—and its constructor tells the story better than any diagram. Count the models you have to hand it:

```python
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification
from trl.experimental.ppo import PPOTrainer, PPOConfig

base = "Qwen/Qwen2.5-0.5B-Instruct"

# Four models resident in memory at once — the defining cost of RLHF:
policy       = AutoModelForCausalLM.from_pretrained(base)                                 # what we're training
ref_policy   = AutoModelForCausalLM.from_pretrained(base)                                 # frozen anchor for the KL leash
reward_model = AutoModelForSequenceClassification.from_pretrained(rm_path, num_labels=1)  # the judge from Stage 2
value_model  = AutoModelForSequenceClassification.from_pretrained(base,    num_labels=1)  # the critic (baseline)

trainer = PPOTrainer(
    args=PPOConfig(kl_coef=0.05, cliprange=0.2, response_length=48),
    processing_class=tokenizer,
    model=policy,
    ref_model=ref_policy,
    reward_model=reward_model,
    value_model=value_model,
    train_dataset=ppo_dataset,
)
trainer.train()   # generate → score → subtract baseline → clipped update, every step
```

Every knob from the theory is right there in plain sight: `kl_coef` is the $\beta$ on the leash, `cliprange` is the seatbelt on each update, and the four separate model arguments are the four networks you now have to fit into GPU memory *at the same time*.

And that last point is not hypothetical. Running this exact setup on a free 16 GB T4 GPU, the reward model trains without complaint—but the PPO step marches straight into an **out-of-memory wall**: four half-billion-parameter models in fp32 simply do not fit. (fp16 would halve the footprint, but on a T4 the generation step then produces `NaN`s, forcing fp32 back.) That failure isn't a bug—it's the lived experience that motivates the entire next section.

### 2.4 — The Pain of RLHF

RLHF works. It also hurts. Anyone who has actually run a PPO alignment loop will recognize this list—and every item on it is a reason DPO exists.

**You're juggling up to four models at once.** Count them: the *policy* being trained, the frozen *reference* model for the KL term, the *reward* model, and the *value/critic* model. All resident in GPU memory, all needing to be served, before you've taken a single useful training step. The memory footprint is enormous, which alone puts full RLHF out of reach for many teams.

**PPO is unstable.** Reinforcement learning is notoriously temperamental. A run that looked healthy can suddenly diverge, collapse into degenerate text, or oscillate without converging. Reproducing a result—even your own from last week, with the same code—can be maddening. You don't just *configure* an RLHF run; you *babysit* it.

**It's brutally sensitive to hyperparameters.** The KL coefficient $\beta$, the learning rate, the number of PPO epochs per batch, the clip range—get any one of them wrong and the run fails silently or subtly degrades. The region of settings that actually works is narrow, and finding it is expensive trial and error.

**Reward hacking is built into the design.** Think about the incentive structure: the policy's entire job is to maximize the reward model's score, which makes it *adversarial* toward that reward model by construction. So it relentlessly probes for the RM's weak spots and finds them—padding answers with length, piling on flattery, exploiting formatting quirks. The result scores beautifully and helps no one. You optimized the proxy, not the goal.

**Online sampling is a constant tax.** Because PPO is on-policy, every step pays for fresh generation from the current model (that expensive line in the pseudocode). Generation is the slowest part of working with an LLM, and here it's not a one-time cost—it's woven into every iteration of training.

Step back and the picture is clear: RLHF demands serious infrastructure, a high tolerance for instability, and constant supervision. At the frontier, with the budget and the team to match, it earns its keep. But for a small team with a handful of GPUs, it's a lot to ask. Which leads to the obvious, tantalizing question:

> *What if we could skip the separate reward model and the whole reinforcement learning loop—and still learn directly from human preferences?*

That question is exactly where DPO begins.

## 3. DPO's Twist — "Your Model Is Already a Reward Model"

We ended the last section with a wish: *skip the separate reward model and the whole reinforcement-learning loop, and still learn directly from human preferences.* It sounds like asking for the cake after refusing to bake it. DPO's contribution is to show the wish isn't greedy at all—the cake was already on the table. The entire RLHF apparatus, it turns out, is an expensive way to compute something that has a clean, closed-form answer.

Remember the promise from Stage 3: *keep the KL term in mind—when we get to DPO, it makes a surprise return as the star of the show.* Here it is.

### 3.1 — RLHF was secretly solving a problem that already has an answer

Recall the objective PPO spends all that effort chasing:

$$\max_{\pi} \; \mathbb{E}_{y \sim \pi}\big[\, r(x, y)\,\big] - \beta \, \mathrm{KL}\big(\pi \,\|\, \pi_{ref}\big)$$

PPO treats this as a search: generate, score, nudge, repeat—thousands of times. But this particular objective—"maximize reward while staying close to a reference"—is one that has been solved with pencil and paper. For *any* reward function, the policy that maximizes it has a known closed form:

$$\pi^{*}(y \mid x) = \frac{1}{Z(x)}\, \pi_{ref}(y \mid x)\, \exp\!\Big(\tfrac{1}{\beta}\, r(x, y)\Big)$$

Read it as a **reweighting of the reference model**. Start from $\pi_{ref}$—what the SFT model would say—and tilt it: multiply the probability of each response by $\exp(r/\beta)$, so high-reward answers get boosted and low-reward answers get suppressed. The size of the tilt is set by $\beta$: small $\beta$ tilts aggressively (chase reward hard), large $\beta$ tilts timidly (stay near the reference). The $Z(x)$ out front is just a normalizer that makes the probabilities sum to one.

There's only one problem with this beautiful formula: that $Z(x)$. To compute it you'd have to sum $\exp(r/\beta)$ over *every possible response*—an astronomically large set. So you can't simply write down $\pi^{*}$ and be done. This intractable normalizer is exactly the wall that forced RLHF to reach for iterative RL in the first place.

### 3.2 — The flip: every language model is already a reward model

Here is the move that names the section. Instead of reading the formula *forward* (reward → optimal policy), turn it *around* and solve for the reward. A line of algebra on the equation above gives:

$$r(x, y) = \beta \, \log \frac{\pi^{*}(y \mid x)}{\pi_{ref}(y \mid x)} \;+\; \beta \log Z(x)$$

Sit with what this says. The reward of a response is—apart from that $Z(x)$ term, which depends only on the prompt—**just the log-ratio between the optimal policy and the reference**, scaled by $\beta$. Turn it inside out: *any* policy $\pi$ implicitly defines a reward, namely how much more (or less) probability it places on a response than the reference does. A model that has learned good behavior is, by this identity, already assigning high implicit reward to good answers. **Your model is already a reward model.** You were never missing a component—you were reading it in the wrong direction.

### 3.3 — The DPO loss, and what it buys you

We now have two facts to combine. From Stage 2, human preferences fit the Bradley–Terry model: the chance a labeler prefers $y^W$ to $y^L$ is $\sigma\big(r(x,y^W) - r(x,y^L)\big)$—it depends only on the *difference* of rewards. And from 3.2, each reward is a log-ratio plus $\beta\log Z(x)$.

Watch what happens when we substitute. Both responses share the same prompt $x$, so both carry the *same* $\beta\log Z(x)$—and in a difference, it cancels. The intractable normalizer, the thing that blocked us in 3.1, simply vanishes. What's left is written entirely in terms of the policy we're training and the frozen reference:

$$\mathcal{L}_{DPO} = -\log \sigma\!\left( \beta \log \frac{\pi_\theta(y^W \mid x)}{\pi_{ref}(y^W \mid x)} \;-\; \beta \log \frac{\pi_\theta(y^L \mid x)}{\pi_{ref}(y^L \mid x)} \right)$$

Compare it to the reward-model loss from 2.2, $-\log\sigma\big(r(x,y^W) - r(x,y^L)\big)$. It's the **exact same shape**—the same "make the gap between winner and loser large" classifier—except the reward $r$ has been replaced by the policy's own log-ratio against the reference. We didn't train a judge and then optimize against it; we folded the two steps into one.

Read the loss left to right, as we did the others. For each preference pair it pushes the policy to **raise the log-probability of the chosen response and lower it for the rejected one**—but always measured *relative to* $\pi_{ref}$. That relative-to-reference framing is the KL leash from Stage 3, no longer bolted on as a separate penalty but woven directly into the loss: $\beta$ and $\pi_{ref}$ sit right there in the equation. The star returned, exactly as promised.

Now count the cost against RLHF's four-model bill from 2.4:

- **Two models, not four.** Only the policy $\pi_\theta$ and the frozen reference $\pi_{ref}$. No separate reward model, no value/critic network.
- **No reinforcement learning.** It's a supervised classification loss over a fixed dataset of preference pairs—the same kind of training you'd run for the reward model, and just as stable.
- **No online generation.** Nothing is sampled during training, so the expensive per-step generation—the constant tax that dominated PPO—is gone. And with no reward model to game, the whole category of reward hacking disappears with it.

That's the twist in one line: **RLHF learns a reward and then chases it; DPO recognizes that the policy *is* the reward, and optimizes it in a single supervised step.** Same destination—alignment to human preference—reached without the reinforcement-learning machinery we spent all of Section 2 wrestling with.
