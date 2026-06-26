---
title: "From Scratch to My Laptop: A Hands-On Guide to Building LLMs on a Budget"
date: "2026-06-26"
author: "Tai Vu"
excerpt: "A four-stage, code-along journey: build a GPT from scratch, run a pretrained model, fine-tune one with QLoRA, and deploy it offline — with the theory and runnable samples for each step."
tags: ["LLM", "AI", "Fine-tuning"]
readTime: "25 min read"
featured: true
slug: "llm-engineer-journey"
---

# 🛠️ From Scratch to My Laptop: A Hands-On Guide to Building LLMs on a Budget

There's a common myth that doing anything real with large language models requires a cluster of A100s and a research lab. I wanted to find out how far you can get with a free Colab notebook, a Kaggle account, and a MacBook.

The answer is *surprisingly far*. Over four stages I went from writing a GPT character by character in raw PyTorch, to fine-tuning a 1.5B-parameter model into a Vietnamese chatbot, to running that chatbot fully offline on my laptop — without ever paying for a GPU.

This post is both a **story** and a **guide**. For each stage I explain the theory you need, then give you the actual, runnable code so you can follow along. The guiding thread is simple:

> **Understand it from the ground up → use what already exists → teach it something new → bring it home to run.**

Everything here runs on free hardware. Let's build.

### Reproducibility map

The working repo for this journey is organized as a sequence of small stages, not one giant notebook. That mattered more than I expected, because every stage leaves behind one artifact that the next stage consumes:

```text
ml-engineer/
  pyproject.toml                       # Python >= 3.12 + core dependencies
  stage1-build-gpt/
    train_gpt.py                       # tiny GPT from scratch
  stage2-huggingface/
    chat_qwen.ipynb                    # load + chat with Qwen
  stage3-finetune-qlora/
    finetune-qlora.ipynb               # train LoRA adapter on Colab/T4
    qwen-vi-lora/                      # 74MB adapter copied back locally
  stage4-deploy-gguf/
    merge_lora.py                      # adapter + base -> merged fp16 model
    Modelfile                          # Ollama recipe for the GGUF
```

The important discipline: **version code, not weights**. The repo ignores `outputs/`, `*-lora/`, `*.safetensors`, and `*.gguf`; the heavy model files live on an external SSD. That keeps the project reproducible without turning git into model storage.

---

## I. Stage 1 — Building a GPT From Zero

Before touching any library, I wanted to *feel* how a language model works. So I followed Andrej Karpathy's "Let's build GPT" approach and wrote a character-level GPT in pure PyTorch — no `transformers`, no abstractions.

### The core idea: a language model is a next-token predictor

That's the whole game. Given some text, the model outputs a probability distribution over what comes next. To train it, we slice the text into `(input, target)` pairs where the target is just the input shifted right by one character:

```
text:    "hello"
input:   h  e  l  l
target:  e  l  l  o
```

The model predicts a character at every position, and we nudge its weights so the predicted distribution puts more mass on the true next character.

### Theory you need

**1. Tokenization (character-level).** We map each unique character to an integer ID. Vocabulary size here is just the number of distinct characters (65 for Shakespeare).

```python
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])
```

**2. Embeddings are lookups, not arithmetic.** A token ID is just a *row address* into a table of vectors. ID 64 doesn't "outweigh" ID 1 — they're addresses, not magnitudes. This is exactly why categorical data uses embeddings instead of raw integers.

**3. Self-attention is a weighted average.** Each token produces three vectors via learned linear layers:

- **Query** ("what am I looking for?")
- **Key** ("what do I contain?")
- **Value** ("what do I pass on?")

The attention weight between two tokens is the dot product of one's query with another's key. We scale by `1/√(head_size)` to keep the softmax from saturating, apply a **causal mask** so a token can only attend to itself and the past (no peeking at the future), softmax to get weights, and take the weighted sum of values. In code that's the entire mechanism:

```python
class Head(nn.Module):
    """One head of self-attention."""
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)        # scaled scores
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf")) # causal mask
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v                                              # weighted sum
```

**4. Attention is "blind" to order**, so we add a **positional embedding** — a second lookup table indexed by position — to the token embeddings. The number of rows equals `block_size`, which *is* the context window.

**5. A Transformer block** stacks attention ("tokens talk to each other") and a feed-forward network ("each token thinks on its own"), each wrapped in a residual connection and LayerNorm:

```python
class Block(nn.Module):
    def __init__(self, n_embd, num_heads):
        super().__init__()
        head_size = n_embd // num_heads
        self.sa   = MultiHeadAttention(num_heads, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1  = nn.LayerNorm(n_embd)
        self.ln2  = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))    # communicate
        x = x + self.ffwd(self.ln2(x))  # think
        return x
```

### The training loop (and the bug everyone hits)

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for step in range(max_iters):
    xb, yb = get_batch("train")
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)  # ← do NOT forget this
    loss.backward()
    optimizer.step()
```

The single most important line for a beginner: `optimizer.zero_grad()`. **PyTorch accumulates gradients** — if you forget to zero them each step, they pile up and the loss explodes. (That accumulation is intentional, by the way; it's what makes gradient accumulation possible, a trick we'll use in Stage 3.)

### Reading the loss like a story

The loss function is **cross-entropy**, which measures how "surprised" the model is by the correct next character. Here's the trick that makes it interpretable: a model that guesses *uniformly at random* has a loss of `ln(vocab_size)`. For 65 characters that's `ln(65) ≈ 4.17`. That number is your **floor for "learned nothing."** If your loss sits at 4.17, something is broken.

Mine didn't sit there. Watching the loss drop told the whole story of attention working:

- **Bigram model** (predicts from only the previous character) → loss **2.457**. The output had structure — capitalized names, line breaks — but the words were gibberish. With only one character of context, it literally cannot spell.
- **Adding self-attention** → loss **2.46 → 2.09**, and real English words appeared: *"Mind, this, thy..."*. That drop is the proof that letting tokens look back at context actually helps.

To generate text, we repeatedly predict, sample, and append:

```python
@torch.no_grad()
def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]      # keep only the last block_size tokens
        logits, _ = self(idx_cond)
        logits = logits[:, -1, :]            # focus on the last step
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)  # sample, don't argmax
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
```

One subtlety: **sample, don't argmax.** `torch.multinomial` draws from the distribution and gives variety; greedy `argmax` gets stuck repeating itself.

### Try it yourself

The whole thing is ~200 lines and trains on a laptop. Config:

```python
batch_size = 32; block_size = 64
n_embd = 128; n_head = 4; n_layer = 4
dropout = 0.1; learning_rate = 3e-4; max_iters = 5000
device = "mps" if torch.backends.mps.is_available() else "cpu"  # Apple Silicon GPU
```

Run it with `python train_gpt.py` (you need a text file `input.txt` — Tiny Shakespeare, or swap in Vietnamese poetry; character-level works for any language).

**Takeaway:** after Stage 1 the Transformer is not a black box anymore. It's something you built by hand — and that changes how you read everything that follows.

---

## II. Stage 2 — Standing on the Shoulders of Hugging Face

Writing a Transformer from scratch is the right way to *learn*. It's the wrong way to *work*. Stage 2 is about switching from "build the engine" to "drive the car" using Hugging Face `transformers`.

I picked **Qwen2.5-0.5B-Instruct** — the smallest capable instruct model around, which runs on a free T4 or even CPU.

### Setup

```python
# On Colab:  !pip install -q transformers accelerate
# Locally with uv:  uv add transformers accelerate
import torch

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Using device: {device}")
```

### Real tokenization changes everything

In Stage 1 my vocabulary was 65 characters. Production models use **subword (BPE) tokenization** with a much larger vocabulary. Let's see the difference:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
print("Vocab size", tokenizer.vocab_size)   # → 151643

ids = tokenizer.encode("Hello, tôi đang học Transformer!")
print("Num tokens:", len(ids))              # → 7
for i in ids:
    print(f"{i:>7} -> {repr(tokenizer.decode([i]))}")
```

Output:

```
     9707 -> 'Hello'
       11 -> ','
   128296 -> ' tôi'
   128348 -> ' đang'
   128292 -> ' học'
    62379 -> ' Transformer'
        0 -> '!'
```

Seven tokens for a whole sentence — each is a *whole word or subword*, not a single character. That's the leap from toy to production tokenization.

(A small footnote on that vocab size: `tokenizer.vocab_size` reports **151,643**, but Qwen2.5's actual embedding matrix is slightly larger — **151,936** rows — because it's padded with extra slots for special and reserved tokens. The tokenizer count is what you encode with; the embedding count is what the model stores. The gap is normal and harmless.)

### How many kinds of tokenizers are there?

Tokenization is the very first step of any LLM: turning raw text into a sequence of integer IDs. There's a whole spectrum of ways to do it, and the choice is a genuine trade-off between **vocabulary size** and **sequence length**. Broadly there are three families — word-level, character-level, and subword — and the subword family is where all modern LLMs live.

**1. Word-level.** Split on whitespace/punctuation; every word is one token.

```
"I'm learning Transformers" → ["I'm", "learning", "Transformers"]
```

- *How it works:* build a vocabulary of all words seen in training; each maps to an ID.
- *Pros:* tokens are meaningful units; sequences are short.
- *Cons:* the vocabulary explodes (English alone has hundreds of thousands of word forms), and any word not seen in training becomes an unknown `<UNK>` token — the **out-of-vocabulary (OOV)** problem. "learn", "learning", "learned" are treated as three unrelated tokens. Modern LLMs don't use this.

**2. Character-level.** Every character is a token — exactly what I used in Stage 1.

```
"hello" → ['h', 'e', 'l', 'l', 'o']
```

- *How it works:* the vocabulary is just the set of distinct characters (65 for Shakespeare).
- *Pros:* tiny vocabulary, and **zero OOV** — any text can be represented.
- *Cons:* sequences become very long (one token per character), so the model spends its limited context window on tiny units and has to learn spelling from scratch. Great for learning, impractical at scale.

**3. Subword.** The middle ground that every production LLM uses: keep common words whole, but break rare words into reusable pieces.

```
"tokenization" → ["token", "ization"]
"unhappiness"  → ["un", "happi", "ness"]
```

This gives you a fixed, manageable vocabulary (Qwen's is 151,643), short-ish sequences, **no OOV** (worst case, a word falls back to characters/bytes), and pieces that generalize — "learn", "learning", "learned" share the "learn" token. There are a few algorithms for *deciding* which subwords to keep:

- **BPE (Byte-Pair Encoding)** — used by GPT-2/3/4, Qwen, Llama. *How it works:* start with individual characters, then repeatedly find the most frequent adjacent pair and merge it into a new token. Do this thousands of times and frequent sequences like "ing" or "tion" become single tokens, while rare words stay split. Encoding a new word just replays those learned merges in order.
- **Byte-level BPE** — a BPE variant (GPT-2, Qwen) that runs over raw **bytes** instead of characters. Because every possible byte is in the base vocabulary, it can encode *literally any* input — emoji, Chinese, code — with truly zero OOV. (Why did ` tôi` above become a *single* token rather than several byte pieces? Not because of byte-level BPE itself — that only guarantees no OOV — but because the word was frequent enough in Qwen's training data to earn its own merged token.)
- **WordPiece** — used by BERT. Similar to BPE, but instead of merging the *most frequent* pair it merges the pair that most increases the **likelihood** of the training data. Continuation pieces are marked with `##`, e.g. `playing → ["play", "##ing"]`.
- **Unigram / SentencePiece** — used by T5, ALBERT, many multilingual models. *How it works:* start with a large candidate vocabulary and *prune* it, keeping the subwords that best explain the corpus under a probabilistic model — the opposite direction from BPE's bottom-up merging. **SentencePiece** is the popular library that implements this (and BPE too); it treats text as a raw stream and encodes spaces as a visible `▁` marker, so it needs no language-specific pre-splitting — handy for languages without spaces.

Here's the same information at a glance:

| Algorithm | How it builds the vocab | Direction | Used by |
|---|---|---|---|
| **BPE** | Merge the most *frequent* adjacent pair, repeatedly | Bottom-up (merge) | GPT-2/3/4, Llama, Qwen |
| **Byte-level BPE** | BPE over raw bytes → every byte is in the base vocab | Bottom-up (merge) | GPT-2, Qwen |
| **WordPiece** | Merge the pair that most increases data *likelihood*; marks continuations with `##` | Bottom-up (merge) | BERT, DistilBERT |
| **Unigram (SentencePiece)** | Start with a big vocab and *prune* to the most probable subwords; spaces become `▁` | Top-down (prune) | T5, ALBERT, mT5, multilingual |

The practical takeaway: **BPE-family subword tokenizers won.** When you call `AutoTokenizer.from_pretrained(...)`, you're almost always getting byte-level BPE or a SentencePiece model — and you never train it yourself, you just load the one shipped with the model.

### Loading the model

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct", torch_dtype="auto"
).to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {n_params:,}")   # → 494,032,768
```

Half a billion parameters — but the same `Block → Attention → FeedForward` architecture I hand-wrote in Stage 1, just scaled up.

### ChatML: how instruct models expect their input

This is the most important concept in Stage 2, and it reappears in every later stage. Instruct models are trained on conversations formatted with special tokens marking each role. You never write that format by hand — `apply_chat_template` builds it:

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Xin chào!"}
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(text)
```

Output — this structure is **ChatML**:

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Xin chào!<|im_end|>
<|im_start|>assistant
```

The trailing `<|im_start|>assistant` with nothing after it is the model's cue to start writing. Remember those `<|im_start|>` / `<|im_end|>` markers — they'll define our training data in Stage 3 and our serving config in Stage 4.

### A reusable chat function

```python
def chat(user_message, system_prompt="You are a helpful assistant.", max_new_tokens=256):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    generated = model.generate(**inputs, max_new_tokens=max_new_tokens)
    # slice off the prompt so we only keep the model's reply
    output_ids = generated[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(output_ids, skip_special_tokens=True)

print(chat("Giải thích attention trong Transformer bằng 2 câu đơn giản."))
```

The one detail worth noticing: we slice `generated[0][inputs.input_ids.shape[1]:]` to drop the prompt and keep only the freshly generated tokens. `model.generate` returns prompt + completion concatenated.

The mapping back to Stage 1 is beautifully one-to-one:

| Stage 1 (hand-written) | Stage 2 (Hugging Face) |
|---|---|
| character `encode` (vocab 65) | `AutoTokenizer`, real BPE (vocab 151,643) |
| my `AttentionLanguageModel` class | `AutoModelForCausalLM` (same architecture, bigger) |
| my hand-written `generate` loop | `model.generate(...)` |

**Takeaway:** Stage 1 makes Stage 2 feel less like magic and more like *recognition*.

---

## III. Stage 3 — Teaching the Model Something New (QLoRA)

This is the heart of the journey: actually **fine-tuning** a model, cheaply. Goal — turn **Qwen2.5-1.5B-Instruct** into a Vietnamese Q&A chatbot, trained entirely on Colab's free T4.

### The problem, and the theory that solves it

Full fine-tuning of a 1.5B model means updating all 1.5 billion weights — and the weights, gradients, and Adam optimizer states together blow past a free T4's 15GB. Three ideas make it fit:

- **LoRA (Low-Rank Adaptation).** Freeze all original weights. For each weight matrix, add a small trainable "side path" `B·A` where `A` and `B` are skinny matrices of rank `r` (here 16). You train *only* `A` and `B` — a tiny fraction of the parameters — yet you can steer the model's behavior. The result is a few-MB adapter file.
- **QLoRA.** Go further: load the frozen base model in **4-bit** precision so it barely occupies memory, then attach LoRA adapters on top. "Q" = quantized.
- **SFT (Supervised Fine-Tuning).** The training objective: show the model `(question → answer)` pairs and teach it to produce the answer.

We use **Unsloth** (a library that makes this ~2× faster with less VRAM) and `trl`'s `SFTTrainer`.

### Step 1 — Load the model in 4-bit

```python
# On Colab with a T4 GPU:  %pip install unsloth
from unsloth import FastLanguageModel
import torch

max_seq_length = 2048
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
)
```

Check how little memory the 4-bit model takes:

```python
used = torch.cuda.memory_reserved(0) / 1e9
print(f"VRAM in use: {used:.2f} GB")   # → 1.21 GB
```

**1.21 GB** for a 1.5B model. That headroom is the entire reason this fits on free hardware.

### Step 2 — Attach LoRA adapters

```python
model = FastLanguageModel.get_peft_model(
    model,
    r=16,                               # rank of the adapters
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",   # attention
        "gate_proj", "up_proj", "down_proj",      # MLP
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)
model.print_trainable_parameters()
```

Output — and this is the number that makes the whole approach click:

```
trainable params: 18,464,768 || all params: 1,562,179,072 || trainable%: 1.1820
```

We are training **1.18%** of the model — ~18M parameters out of ~1.56B. Everything else is frozen.

### Step 3 — Prepare the dataset

I used `bkai-foundation-models/vi-alpaca` (50,006 Vietnamese instruction examples), sampled down to 3,000 to keep training short. Each example becomes a ChatML conversation — the same format from Stage 2:

```python
from datasets import load_dataset

dataset = load_dataset("bkai-foundation-models/vi-alpaca", split="train")
sub = dataset.shuffle(seed=3407).select(range(3000))

def format_chat(ex):
    user = ex["instruction"]
    if ex["input"].strip():
        user += "\n\n" + ex["input"]
    messages = [
        {"role": "user", "content": user},
        {"role": "assistant", "content": ex["output"]},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}

dataset = sub.map(format_chat)
```

> **Quality > quantity.** 3,000 clean, well-formatted examples teach the *format and style* you want. You don't need millions of rows to get a useful behavior change.

### Step 4 — Train

```python
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,    # effective batch size = 2 × 4 = 8
        num_train_epochs=1,
        warmup_steps=5,
        learning_rate=2e-4,
        optim="adamw_8bit",               # 8-bit optimizer saves more memory
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",
    ),
)
trainer_stats = trainer.train()
```

Two memory tricks worth calling out: `gradient_accumulation_steps=4` simulates a batch of 8 while only ever holding 2 examples in memory (this is the "intentional gradient accumulation" from Stage 1, now earning its keep), and `optim="adamw_8bit"` stores optimizer state in 8 bits.

The training summary:

```
Num examples = 3,000 | Num Epochs = 1 | Total steps = 375
Total batch size (2 x 4 x 1) = 8
Trainable parameters = 18,464,768 of 1,562,179,072 (1.18% trained)
```

**375 steps, about 13 minutes, free of charge.** The loss starts around 1.5–2.2 and settles into the ~0.9–1.1 range (with occasional spikes on harder batches — the noisy-but-downward curve you want).

### Step 5 — Verify it actually changed

Here's an elegant trick: LoRA adapters can be toggled on and off, so you can ask the *exact same question* both ways and compare:

```python
FastLanguageModel.for_inference(model)

def chat(user_message, max_new_tokens=256):
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_message}],
        tokenize=True, add_generation_prompt=True, return_tensors="pt",
    ).to("cuda")
    outputs = model.generate(input_ids=inputs, max_new_tokens=max_new_tokens,
                             temperature=0.7, do_sample=True)
    return tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)

question = "Viết một đoạn ngắn giới thiệu về thành phố Đà Nẵng."
print("----- After fine-tune (adapter ON) -----")
print(chat(question))
print("----- Before fine-tune (base Qwen) -----")
with model.disable_adapter():
    print(chat(question))
```

Both answered in fluent Vietnamese; the fine-tuned version was longer and more structured. (It's not perfect — one run slipped a stray character into a word. 3,000 examples and one epoch is a starting point, not a finish line.)

### Step 6 — Save the adapter

```python
model.save_pretrained("qwen-vi-lora")
tokenizer.save_pretrained("qwen-vi-lora")
```

The result is a **~74MB** adapter. That tiny file *is* everything you taught the model — 18 million numbers. Copy it to Google Drive and move on.

**Takeaway:** fine-tuning a 1.5-billion-parameter model cost 13 minutes on a free T4. The barrier to entry is much lower than it looks.

---

## IV. Stage 4 — Bringing It Home to My Laptop

A model trapped in a Colab notebook isn't useful. Stage 4 turns that 74MB adapter into a chatbot running **offline on my Mac** — no internet, no cloud. Three steps:

```
qwen-vi-lora (74MB adapter)
  ① MERGE     → fold LoRA into the base model → fp16 model (~3GB)
  ② QUANTIZE  → convert to GGUF q4_k_m        (~1GB)
  ③ SERVE     → Modelfile → ollama create → ollama run qwen-vi
```

### What you need locally

The inputs and tools are deliberately minimal:

- the LoRA adapter from Stage 3: `stage3-finetune-qlora/qwen-vi-lora/`
- Python environment with `torch`, `transformers`, `peft`, and `gguf`
- `llama.cpp` for Hugging Face → GGUF conversion and quantization
- Ollama installed on the Mac
- enough disk space for temporary artifacts: ~3GB merged fp16 model, ~3GB fp16 GGUF, ~1GB final quantized GGUF

My project keeps only the scripts in git. Large outputs go to an external SSD:

```text
/Volumes/TaiSSD/ml-engineer/merged              # merged Hugging Face model
/Volumes/TaiSSD/ml-engineer/qwen-vi-f16.gguf    # intermediate GGUF
/Volumes/TaiSSD/ml-engineer/qwen-vi-q4_k_m.gguf # final Ollama model file
```

### Step 1 — Merge

A LoRA adapter is mathematically `B·A` added on top of the original weights. To run standalone we fold that product back in with `merge_and_unload()`, producing a normal fp16 model:

```python
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "Qwen/Qwen2.5-1.5B-Instruct"   # fp16 base, same weights as the 4-bit version we trained
ADAPTER = os.path.expanduser(
    "~/Library/CloudStorage/GoogleDrive-vdtai94@gmail.com/My Drive/"
    "Project/ml-engineer/stage3-finetune-qlora/qwen-vi-lora"
)
OUT = "/Volumes/TaiSSD/ml-engineer/merged"   # heavy files live on an external SSD

base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(ADAPTER)
model = PeftModel.from_pretrained(base, ADAPTER)   # attach adapter
model = model.merge_and_unload()                   # add B·A into the base, drop the adapter
model.save_pretrained(OUT, safe_serialization=True)
tokenizer.save_pretrained(OUT)
```

Run it from the project root:

```bash
uv sync
uv run python stage4-deploy-gguf/merge_lora.py
```

The first run downloads the fp16 base model from Hugging Face, attaches the local adapter, merges it, and writes a normal Hugging Face model directory to `/Volumes/TaiSSD/ml-engineer/merged`. After this step, the model no longer needs the LoRA adapter at inference time; the adapter's learned update has been folded into the base weights.

### Step 2 — Quantize to GGUF

**GGUF** is `llama.cpp`'s model format, optimized for running on CPU and Apple Silicon. **Quantizing to `q4_k_m`** compresses the weights to ~4 bits, taking the model from ~3GB down to ~1GB so it runs smoothly on a laptop. (Notice this is the *same 4-bit idea* from QLoRA in Stage 3 — there it made *training* fit; here it makes *inference* fit.)

Build `llama.cpp` once:

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
python -m pip install -r requirements.txt
cmake -B build -DGGML_METAL=ON
cmake --build build --config Release -j
```

From that `llama.cpp` checkout, convert the merged Hugging Face model to fp16 GGUF:

```bash
python convert_hf_to_gguf.py \
  /Volumes/TaiSSD/ml-engineer/merged \
  --outfile /Volumes/TaiSSD/ml-engineer/qwen-vi-f16.gguf \
  --outtype f16
```

Finally quantize it:

```bash
./build/bin/llama-quantize \
  /Volumes/TaiSSD/ml-engineer/qwen-vi-f16.gguf \
  /Volumes/TaiSSD/ml-engineer/qwen-vi-q4_k_m.gguf \
  Q4_K_M
```

The fp16 GGUF is just an intermediate. The file I actually serve is `qwen-vi-q4_k_m.gguf`, because it is small enough to keep loaded comfortably while still preserving useful answer quality for this 1.5B model.

### Step 3 — Serve with Ollama

Ollama wraps a GGUF file into a chat service. You describe it with a `Modelfile` — and look closely at the `TEMPLATE`: it's the **same ChatML format** from Stages 2 and 3.

```dockerfile
FROM /Volumes/TaiSSD/ml-engineer/qwen-vi-q4_k_m.gguf

TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>
"""

PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
PARAMETER temperature 0.7
PARAMETER top_p 0.8

SYSTEM """Bạn là một trợ lý ảo tiếng Việt hữu ích, trả lời ngắn gọn và rõ ràng."""
```

Back in the `ml-engineer` project, create and run the local model from the directory that contains the `Modelfile`:

```bash
cd stage4-deploy-gguf
ollama create qwen-vi -f Modelfile
ollama run qwen-vi
```

I like doing one quick smoke test before calling it done:

```text
>>> Viết một đoạn ngắn giới thiệu về thành phố Đà Nẵng.
>>> Giải thích attention trong Transformer bằng 2 câu đơn giản.
>>> Hãy trả lời ngắn gọn: QLoRA khác LoRA ở điểm nào?
```

The test is intentionally simple. At this stage I am not trying to prove the model is "smart"; I am checking that the full pipeline survived: ChatML formatting works, stop tokens stop generation, Vietnamese answers come out cleanly, and the model being served is the fine-tuned one rather than the untouched base model.

And there it is — the model I fine-tuned, answering in Vietnamese, running entirely on my laptop with no GPU and no internet.

### Reproducibility checklist

If I had to rerun the whole Stage 4 from a clean machine, these are the exact things I would verify:

1. **Adapter exists.** `stage3-finetune-qlora/qwen-vi-lora/` contains `adapter_model.safetensors`, `adapter_config.json`, tokenizer files, and the chat template.
2. **Base model matches training.** Merge with `Qwen/Qwen2.5-1.5B-Instruct`, the fp16 base corresponding to the 4-bit Unsloth model used during QLoRA.
3. **Tokenizer comes from the adapter folder.** That preserves the same chat template and special-token behavior used during fine-tuning.
4. **Large artifacts stay outside git.** Keep `.safetensors` and `.gguf` files ignored; store them on an external disk or artifact storage.
5. **Ollama template matches Qwen ChatML.** The `Modelfile` must emit `<|im_start|>system`, `<|im_start|>user`, and the trailing `<|im_start|>assistant` prompt.
6. **Stop tokens are set.** Stop on both `<|im_start|>` and `<|im_end|>` so the model does not leak role markers into the answer.
7. **Use a fixed smoke-test prompt set.** Keep the same 3-5 prompts across base, adapter, merged fp16, and GGUF so regressions are obvious.

### The conceptual bookend: perplexity

To evaluate the finished model, you can compute **perplexity** — and perplexity is simply the exponential of the same cross-entropy loss I was watching back in Stage 1. The very first number I learned to read is the one used to judge the final model. The journey loops back on itself.

**Takeaway:** the path from cloud-trained adapter to offline laptop assistant is three steps, and every one of them reuses an idea from an earlier stage.

---

## V. What Four Stages Taught Me

The most valuable thing wasn't any single technique — it was watching the *same ideas* reappear at every scale:

- **The Transformer block** I hand-wrote in Stage 1 is literally what runs inside Qwen in Stages 2–4.
- **ChatML** appeared in Stage 2, defined the training data in Stage 3, and configured the server in Stage 4.
- **Cross-entropy loss** from Stage 1 became **perplexity** in Stage 4.
- **4-bit quantization** powered both the *training* (QLoRA, Stage 3) and the *deployment* (GGUF, Stage 4).
- **Gradient accumulation**, a footnote in Stage 1, made the batch size fit in Stage 3.

And the headline: I built, trained, and deployed a custom Vietnamese chatbot using nothing but **free tiers and a laptop**. The bill for the most important stage — fine-tuning a 1.5-billion-parameter model — was 13 minutes on a free T4.

### A cheat sheet if you want to follow this path

1. **Start by building, not reading.** Write a tiny GPT once. It demystifies everything else.
2. **Always start with the smallest model** (0.5B–1.5B) to debug your pipeline, then scale.
3. **Default to QLoRA.** At the individual level you almost never need full fine-tuning.
4. **Kaggle is more generous than Colab free** (2× T4, 30h/week) when you need more.
5. **Learn by project** — make each stage produce one thing that actually runs.

If you've been putting this off because you assumed it needed expensive hardware, take this as your sign. Start small, start cheap, and build each piece with your own hands. The path from "what even is attention" to "my own model running offline" is shorter than it looks.

*Next up: preference tuning with DPO and ORPO — teaching the model not just to answer, but to answer the way I'd prefer. More on that soon.*
