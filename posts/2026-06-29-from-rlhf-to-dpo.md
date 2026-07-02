---
title: "From RLHF to DPO: A Simpler Way to Align Language Models"
date: "2026-06-29"
author: "Tai Vu"
excerpt: "How do you turn 'humans prefer answer A over B' into a training signal? A ground-up tour of RLHF, the insight behind DPO, and how to choose between them."
tags: ["LLM", "AI", "Alignment", "RLHF", "DPO"]
readTime: "25 min read"
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

## 4. How DPO Works — Step by Step

Section 3 earned us the formula. This section is about what actually happens when you run it: what goes in, what one training step computes, which direction the weights move, and what the knobs do. If RLHF was an assembly line, DPO is a single machine—but it's worth opening the case and watching the gears turn once.

### 4.1 — The ingredients

Everything DPO needs fits in three bullet points:

- **An SFT model.** Same as RLHF's Stage 1—you still need a model that answers in the right format. DPO replaces Stages 2 and 3, not Stage 1. You load it *twice*: one copy becomes the trainable policy $\pi_\theta$, the other is frozen as the reference $\pi_{ref}$.
- **A preference dataset.** The exact same format as the reward-model data from 2.2: triples of (prompt $x$, chosen $y^W$, rejected $y^L$). Nothing new to collect—if you were preparing to train a reward model, you already have DPO's training set.
- **The loss from 3.3.** That's it. No reward model to pretrain, no critic to warm up, no generation pipeline to stand up.

Notice what's *missing* from the list: any step that runs before training can begin. RLHF has a whole Stage 2 you must finish (and validate!) before Stage 3 starts. DPO starts training immediately.

### 4.2 — Anatomy of one training step

Take a single preference pair and follow it through. Four forward passes, one subtraction, one sigmoid:

```
        prompt x, chosen yᵂ, rejected yᴸ
                      │
        ┌─────────────┴──────────────┐
        ▼                            ▼
  policy π_θ  (trainable)      reference π_ref  (frozen)
  log π_θ(yᵂ|x)   ──┐          log π_ref(yᵂ|x)  ──┐
  log π_θ(yᴸ|x)   ──┤          log π_ref(yᴸ|x)  ──┤
        │           ▼                │            ▼
        │   implicit rewards:        │
        │   r̂ᵂ = β·[log π_θ(yᵂ|x) − log π_ref(yᵂ|x)]
        │   r̂ᴸ = β·[log π_θ(yᴸ|x) − log π_ref(yᴸ|x)]
        │                            │
        └──────────┬─────────────────┘
                   ▼
        loss = −log σ( r̂ᵂ − r̂ᴸ )
```

Three things to notice, in order:

**First, "the probability of a response" is just a sum.** A response is a token sequence, and a language model already assigns each token a log-probability given everything before it. So $\log \pi(y \mid x)$ is nothing exotic—run the model over the pair, sum the log-probs of the response tokens. This is the same computation as evaluating perplexity, and it's why DPO needs no generation: **the responses are already written in the dataset; we only ever *score* them.**

**Second, the implicit rewards $\hat{r}^W$ and $\hat{r}^L$ are exactly Section 3's log-ratios.** Each one asks: *compared to the reference, how much has the policy grown to like this response?* A positive $\hat{r}^W$ means the policy now prefers the chosen answer more than the SFT model did. These numbers are worth watching during training—more on that in 4.4.

**Third, the loss only cares about the margin** $\hat{r}^W - \hat{r}^L$. It's our Bradley–Terry classifier from 2.2 one more time: push the winner's implicit reward above the loser's, with the sigmoid making the push strongest when the ordering is wrong and gentle once it's confidently right.

### 4.3 — Which way do the weights move?

Differentiate the loss and the update direction becomes wonderfully readable. For each pair, the gradient does two things at once, scaled by one common factor:

$$\nabla_\theta \mathcal{L}_{DPO} \;=\; -\,\beta \,\underbrace{\sigma\big(\hat{r}^L - \hat{r}^W\big)}_{\text{how wrong the model is}}\,\Big[\underbrace{\nabla_\theta \log \pi_\theta(y^W \mid x)}_{\text{push chosen up}} \;-\; \underbrace{\nabla_\theta \log \pi_\theta(y^L \mid x)}_{\text{push rejected down}}\Big]$$

Read the three braces left to right:

- **The weight, $\sigma(\hat{r}^L - \hat{r}^W)$,** is the model's current probability of getting this pair *backwards*. If the policy already ranks the chosen response far above the rejected one, this factor is near zero and the pair barely nudges the weights. If the policy has it upside down, the factor approaches one and the pair hits hard. Every example is weighted by *how wrong the model still is about it*—the same "spend effort where you're wrong" behavior we saw in the reward-model loss, now steering the policy directly.
- **Push chosen up:** increase the log-probability of every token in $y^W$.
- **Push rejected down:** decrease the log-probability of every token in $y^L$.

And because both $\hat{r}$'s are measured against $\pi_{ref}$, the "how wrong" weight automatically shrinks as the policy drifts from the reference—the KL leash doing its job from *inside* the gradient, with no separate penalty term to tune.

### 4.4 — What this looks like in code

By now you can guess the shape: same TRL library, same trainer pattern as 2.2—except this one replaces *both* RLHF stages. Put the two constructors side by side and the whole argument of this post is visible in the argument lists:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig

base = "Qwen/Qwen2.5-0.5B-Instruct"          # the SFT model (Stage 1 still required)
tokenizer = AutoTokenizer.from_pretrained(base)

policy = AutoModelForCausalLM.from_pretrained(base)   # π_θ — trainable
ref    = AutoModelForCausalLM.from_pretrained(base)   # π_ref — frozen anchor

# Two models. No reward_model=. No value_model=.
trainer = DPOTrainer(
    model=policy,
    ref_model=ref,
    args=DPOConfig(beta=0.1, learning_rate=5e-7, max_length=1024),
    train_dataset=dataset,        # same {"prompt", "chosen", "rejected"} pairs as 2.2
    processing_class=tokenizer,
)
trainer.train()   # minimizes  −log σ( β·log-ratio(chosen) − β·log-ratio(rejected) )
```

Compare this to the `PPOTrainer` call in 2.3: the `reward_model=` and `value_model=` lines are simply *gone*, and with them half the GPU memory bill. (In practice you can even drop `ref_model=` and pass a PEFT/LoRA config instead—TRL then recovers the reference by disabling the adapter, so only *one* full model sits in memory. That trick is what makes DPO comfortable on a free Colab/Kaggle GPU, where Section 2.3's four-model PPO setup hit an out-of-memory wall.)

**What to watch while it trains.** With PPO you babysat divergence; with DPO you mostly read three gauges, all derived from the implicit rewards of 4.2:

- `rewards/chosen` and `rewards/rejected` — the batch-average $\hat{r}^W$ and $\hat{r}^L$. Healthy runs show the two curves separating.
- `rewards/margins` — the gap $\hat{r}^W - \hat{r}^L$, which should climb steadily.
- `rewards/accuracies` — the fraction of pairs the policy ranks correctly, the single most interpretable number on the dashboard. It's the same "pairwise accuracy" we used to judge the reward model in 2.2—except now the *policy itself* is the judge being scored.

One behavior surprises everyone the first time: the log-probability of the **chosen** responses often *drops* during training. That's expected—the loss only demands the margin grow, not that $\pi_\theta(y^W)$ rise in absolute terms; probability mass can flow to other good phrasings the dataset never listed. Watch the margin and the accuracy, not the raw log-probs.

### 4.5 — The knob that matters: β

DPO inherits exactly one important hyperparameter from the RLHF objective, and you already know what it does: $\beta$ is the **tension in the KL leash** from 2.3, now living inside the loss.

- **Small β (≈ 0.01–0.05):** a loose leash. The implicit rewards are cheap to move, so the policy chases preference margins aggressively and drifts further from the SFT model. More alignment pressure, more risk of degrading fluency or forgetting.
- **Large β (≈ 0.3–0.5):** a tight leash. Log-ratios get expensive, updates stay timid, and the policy hugs the reference—safer, but the preferences may barely sink in.
- **The common default, β = 0.1,** sits in the middle, and in practice DPO is *far* more forgiving about this single knob than PPO ever was about its half-dozen (recall 2.4: KL coefficient, learning rate, PPO epochs, clip range, batch schedule…). Typical DPO tuning is one β sweep and an unusually small learning rate (~5e-7—note the scale: preference gradients are sharp, and a standard fine-tuning rate would overshoot).

Step back and look at what the four subsections added up to. Training data you already had, two models instead of four, a loss whose gradient you can read aloud, three gauges instead of a babysitting shift, and one knob instead of six. **The step-by-step of DPO is short precisely because the cleverness was spent in the math, not the infrastructure.** The obvious next question is the honest one: if DPO is this much simpler, why does RLHF still exist? That trade-off is where we turn next.

## 5. RLHF vs DPO — When to Use Which

Two sections of praise for DPO have earned RLHF the right to a rebuttal. The derivation in Section 3 was exact, but exactness in math doesn't mean equivalence in practice—the two methods optimize the same objective *under different conditions*, and those conditions are where the real trade-offs live. So let's play fair.

### 5.1 — What DPO quietly gave up

**DPO is offline; it never explores.** Look back at the anatomy in 4.2: every response DPO learns from was *already written in the dataset*. The policy is never asked to generate anything, so it can never be corrected on its own mistakes—only on the mistakes some other model (whoever produced the dataset's responses) happened to make. PPO's expensive online sampling, the "constant tax" we complained about in 2.4, buys exactly this: the model is graded on *its own current outputs*, including whatever novel failure modes it just invented. As training moves the policy away from the models that generated the data, DPO's preference pairs slowly become critiques of someone the policy no longer is.

**The implicit reward doesn't generalize the way an explicit one does.** RLHF's reward model is a standalone artifact: once trained, it can score *any* response to *any* prompt—responses no human ever labeled—and it can be reused across runs, checked for calibration, even audited on its own. DPO's "reward" exists only as a log-ratio between two specific checkpoints; it evaluates the dataset's pairs and nothing more. When you paid RLHF's four-model bill, one of the things you bought was a judge that generalizes.

**Offline optimization can overshoot.** The DPO loss is perfectly happy to keep growing the margin on pairs it already ranks correctly—and with a small or narrow dataset, the cheapest way to grow margins is often to crush the probability of rejected responses toward zero, dragging fluent phrasings down with them. This is the failure mode behind the surprising gauge behavior in 4.4 (chosen log-probs drifting down), taken to its pathological end. RLHF, graded on fresh samples every step, gets caught immediately when its actual outputs degrade; DPO can look great on its three gauges while quietly wandering out of distribution.

### 5.2 — Why the frontier still runs RL

This is why the largest labs—the ones aligning ChatGPT-, Claude-, and Gemini-class models—still run reinforcement learning at scale. It isn't nostalgia. At that scale the equation flips:

- The **infrastructure objection dissolves.** Four models in memory is a crisis on one 16 GB T4 (we watched it OOM in 2.3); it's a rounding error on a dedicated training cluster with a serving fleet.
- **Exploration compounds.** A frontier model is aligned through many rounds: sample from the current policy, collect fresh preferences on *those* samples, update, repeat. The data distribution tracks the policy instead of lagging behind it—something no fixed offline dataset can do.
- **The reward model becomes a platform.** Once you have a good judge, you can point it at new tasks, blend it with rule-based signals, and iterate on the policy without re-collecting human labels each time. The judge outlives any single training run.

The honest summary: **DPO removed RLHF's infrastructure, and with it RLHF's ceiling.** For most teams the infrastructure was the binding constraint, so DPO is a pure win. For the few teams where it isn't, the ceiling is what matters.

### 5.3 — The decision, in one table

This table is lifted from a hands-on lab that ran both methods back to back on the same free-tier GPU—the same experiment quoted throughout Section 2:

| | RLHF (RM + PPO) | DPO |
|---|---|---|
| Separate reward model | Yes — trained and validated first | No |
| Algorithm | PPO: generate → score → update (RL) | One classification loss |
| Models in memory during training | 4 (policy + ref + reward + value) | 2 (policy + ref); 1 with LoRA |
| Generation during training | **Yes** — the source of most trouble | No |
| Tuning difficulty / babysitting | High (six coupled knobs, unstable) | Low (mostly β) |
| Explores its own outputs | Yes — learns from fresh samples | No — fixed offline pairs |
| Reusable judge afterwards | Yes — the reward model | No |
| Runs on a free 16 GB GPU | No (OOM'd in practice) | Yes, comfortably |

Read the first five rows and choose DPO; read the last three and understand why frontier labs don't. As a rule of thumb: **if your preference data is a fixed dataset and your GPUs are countable on one hand, use DPO and don't look back. Reach for RLHF when you can afford to keep collecting preferences on your model's own outputs—because that feedback loop is the one thing DPO structurally cannot replicate.**

**The family keeps growing.** DPO's real legacy may be the door it opened. Once "alignment = one clever supervised loss" was on the table, variants followed quickly: **ORPO** folds the preference term into SFT itself, dropping even the reference model (one model in memory—the logical endpoint of the count we've been running all post); **KTO** learns from unpaired thumbs-up/thumbs-down labels instead of pairs; **SimPO** removes the reference by normalizing over length. The details differ, but they're all descendants of the same flip from Section 3.

### 5.4 — Closing the loop

We opened with one deceptively simple question: *how do you turn "humans prefer answer A over B" into a training signal?*

Now we can answer it twice. **RLHF answers with a system:** teach a judge to imitate human preferences, then run a reinforcement-learning loop against that judge, with a KL leash to keep things sane—powerful, scalable, and heavy. **DPO answers with a formula:** notice that the leash-constrained objective has a closed-form solution, invert it, and the "judge" turns out to be readable off the policy itself—the whole system collapses into one supervised loss.

Same Bradley–Terry foundation, same KL anchor, same destination. The difference is where the effort goes: RLHF spends it on infrastructure, DPO spends it on algebra. And that, more than any benchmark, is the lesson worth keeping: **sometimes the biggest simplification in engineering comes from noticing that the problem you're solving iteratively has already been solved on paper.**
