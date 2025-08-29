---
title: "Attention Mechanism in Large Language Models"
date: "2025-08-20"
author: "Tai Vu"
excerpt: "Learn how the attention work in large language models."
tags: ["LLM", "AI"]
readTime: "8 min read"
featured: true
slug: "getting-insight-attention"
---

# Attention Is All You Need: The Birth of the Transformer

In 2017, Vaswani and colleages published a page: Attention Is All You Need. This work introduced the **Transformer**, a neural network architecture that removed the need for recurrence and convolution in sequence modeling, paved the way for the development of modern large language models.

## What is Attention?

The attention mechanism allows models to focus on different parts of the input sequence when processing each element. Instead of processing information sequentially like traditional RNNs, attention enables parallel processing and better handling of long-range dependencies.

## Prior Problems
Before Transformers, the dominant models for natural language tasks were **RNNs** and **LSTMs**. These models processed sequences step by step, making them difficult to parallelize and slow to train. They also struggled with **long-range dependencies**, since information had to pass sequentially through many steps, often vanishing along the way.

### 1. Sequential Bottlenecks Slow Training
Because each hidden state depends on the one before it, RNNs cannot process a sentence in *parallel*. Training must happen step by step, which becomes a serious bottleneck for long sequences.

Example:
```text
I love machine learning.
```
To process this simple sentence, an RNN must:
    1. Process “I” → state h1
    2. Process "love" with h1 -> state h2
    3. Process "machine" with h2 -> state h3
    4. Proces "learning" with h3 -> state h4

Even for short sentences this is inefficient, and for millions of sequences in large datasets, it makes training extremly slow.

### 2. Long-Distance Dependencies Are Hard
RNNs process sequences word by word, passing information from one node to the next in the network. This makes it difficult to remember relationships between words that are far apart.
Example:
```text
The park, which is 1 km away from my house, is a wonderful place for dating.
```
To predict the word *"dating"*, the model needs to recall that *"The park"* appeared at the beginning. But because RNNs pass
information step by step (park -> which -> is -> ...), the signal can weaken or vanish along the way. This is the classic **vanishing gradient problem**.

## The Key Idea: Attention
The Transformer solved this by replying entirely on **attention mechanism**. Instead of processing tokens one by one, attention lets the model directly *look at* every other word
in the sequence - no matter how far apart. This significantly reduces path lengths between dependencies and allows for massive parallelization during training.

The general attention mechanism can be expressed as:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

**Where:**

- $Q \in \mathbb{R}^{n \times d_k}$: Query matrix (n queries, each of dimension $d_k$)
- $K \in \mathbb{R}^{m \times d_k}$: Key matrix (m keys, each of dimension $d_k$)
- $V \in \mathbb{R}^{m \times d_v}$: Value matrix (m values, each of dimension $d_v$)
- $d_k$: Dimension of keys (used for scaling)
- The scaling factor $\sqrt{d_k}$ prevents the dot products from growing too large

While it might look intimidating at first, this formula encodes a beautiful and intuitive idea. Let's break it down step by step.

### Why This Fomula? Understanding Each Component
#### 1. The Intuition Behind Attention
Imagine you're reading a sentence: "The cat sat on the chair"
When processing the word "sat", your brain automatically focuses on relevant context:
- *"cat"* (the subject performing the action)
- *"chair"* (the location where the action happens)
- Less attention to *"the"* (prammatical but less informative)
Attention mechanism mathematically models this human-like selective focus.

#### 2. Breaking Down $QK^\top$: The Compatibility Function
**Q (Query):** "What am I looking for?"
**K (Key):** "What can I offer?"

The dot product $Q \cdot K^\top$ measures **compatibility** or **similarity** between queries and keys.

```python
# Example with d_k = 3 for visualization
Q = [
    [0.1, 0.5, 0.2],  # Query for "The": looking for nouns
    [0.8, 0.1, 0.3],  # Query for "cat": looking for verbs  
    [0.2, 0.9, 0.1],  # Query for "sat": looking for subjects
]

K = [
    [0.3, 0.2, 0.7],  # Key for "The": I'm an article
    [0.9, 0.1, 0.2],  # Key for "cat": I'm a noun
    [0.1, 0.8, 0.4],  # Key for "sat": I'm a verb  
]

# Attention score between "sat" and "cat"
score = Q[2] · K[1] = [0.2, 0.9, 0.1] · [0.9, 0.1, 0.2] = 0.29
```
*Why Q and K must have the same dimensions?*
For the dot product $Q·K^\top$ to be mathematically valid, Q and K must live in the same semantic space. Think of it like a search engine:
- Your search query and document titles must be in the same vector space to compute similarity.
- If query is 128-dimensional, documents must also be 128-dimensional.

#### 3. The Scaling Factor: Why √d_k?
When $d_k$ is large, dot products can become very large:
```python
# With d_k = 64
q = random_vector(64)  # values ~ N(0,1)
k = random_vector(64)
dot_product = q · k  # Can reach ±8, ±10, or higher
```
Problem: Large values fed into softmax create vanishing gradients:
```python
softmax([10, 1, 2]) = [0.999, 0.0002, 0.0008]
# Gradients ≈ 0! Training becomes difficult
```
Solution: Dividing by √d_k normalizes the scores to a reasonable range, keeping gradients healthy during training.

#### 4. Softmax: Converting Scores to Probabilities
Softmax transforms raw scores into a probability distribution:
- All values sum to 1
- Large scores are amplified, small scores are compressed
- Creates meaningful attention weights

#### 5. Why V Can Have Different Dimensions?
**V (Value)**: "The actual information I want to convey"
```python
attention_weights = softmax(QK^T / √d_k)  # [seq_len, seq_len]
V = values  # [seq_len, d_v] - can be different!

output = attention_weights @ V  # [seq_len, d_v]
```
**Key insight:** V doesn't participate in the similarity computation—it only carries information. Like a recommendation system:
- User preferences and movie genres must be in the same space (Q, K)
- But movie information returned can be richer (V): title, director, cast, etc.

**Why Q and K Have the Same Sequence Length (Sometimes)**
This is a nuanced topic that depends on the type of attention being used.

**Case 1: Self-Attention (Same Sequence Length)**
In self-attention, all tokens attend to all other tokens within the same sequence:
```python
input_sequence = "The cat sat on chair"  # 5 tokens

Q: [5, d_k]  # 5 queries (one per token)
K: [5, d_k]  # 5 keys (one per token)
V: [5, d_v]  # 5 values (one per token)

# Attention matrix: [5, 5]
# attention[i,j] = how much token i attends to token j
```

**Case 2: Cross-Attention (Different Sequence Lengths)**
In cross-attention, tokens from one sequence attend to tokens from a different sequence:
```python
# Machine Translation Example
source = "Tôi yêu lập trình"    # Vietnamese: 4 tokens
target = "I love programming"    # English: 3 tokens

# Cross-attention: target attends to source
Q: [3, d_k]  # From target sequence (3 queries)
K: [4, d_k]  # From source sequence (4 keys)
V: [4, d_v]  # From source sequence (4 values)

# Attention matrix: [3, 4]
# attention[i,j] = target token i's attention to source token j
```
**Important constraint:** K and V must always have the same sequence length (they come from the same source), but Q can have any length.

**When to Use Self-Attention vs Cross-Attention**
Understanding when to use each type is crucial for building effective AI systems.

**Self-Attention: "Looking Within Myself"**
**Purpose:** Understanding internal relationships within the same sequence

**Use Cases:**
**1. Language Models (GPT)**
```python
text = "The company reported strong earnings. The stock price"
# Self-attention helps:
# - Link "stock price" with "company" and "earnings"
# - Understand context for next word prediction
```
**2. Encoders (BERT)**
```python
sentence = "The cat that I saw yesterday was cute"
# Self-attention learns:
# - "cat" connects to "was cute" (main clause)
# - "I saw yesterday" is a relative clause modifying "cat"
```
**3. Document Understanding**
```python
# In long documents, self-attention helps:
# - Resolve pronouns (he -> John)
# - Understand logical structure (cause-effect)
# - Identify main themes
```

**Cross-Attention: "Looking at Something Else"**
**Purpose:** Connecting information between two different sequences

**Use Cases:**
**1. Machine Translation**
```python
# When translating "I am learning English"
source = ["Tôi", "đang", "học", "tiếng", "Anh"]
target = ["I", "am", "learning", "English"]

# Each English word cross-attends to Vietnamese words:
# "learning" -> high attention to "học"
# "English" -> high attention to "tiếng Anh"
```
**2. Question Answering**
```python
context = "Paris is the capital of France. It has 2 million people."
question = "What is the population of Paris?"

# Question words cross-attend to context:
# "population" -> strong attention to "2 million people"
```
**3. Image Captioning**
```python
image_features = [region1, region2, ..., regionN]  # CNN features
caption = "A cat sitting on"  # Currently generating

# Each caption word cross-attends to image regions:
# "cat" -> attends to cat-related image regions
# "sitting" -> attends to posture-related features
```

## Multi-Head Attention: Looking from Multiple Perspectives

The Transformer doesn't just use a single attention function - it uses **Multi-Head Attention**, which runs multiple attention functions in parallel and then combines their outputs.

### Why Multiple Heads?

Think about how humans process language. When you read "The bank can guarantee deposits will eventually cover future tuition costs," your brain simultaneously processes different types of relationships:

- **Syntactic relationships:** "bank" is the subject, "can guarantee" is the verb phrase
- **Semantic relationships:** "deposits" and "tuition costs" are both financial concepts
- **Coreference:** "deposits" refers to something the bank holds

Multi-head attention allows the model to capture these different types of relationships simultaneously.

### The Mathematical Formulation

```python
MultiHead(Q, K, V) = Concat(head₁, ..., head_h)W^O

where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

**Key parameters:**
- **h = 8**: Number of attention heads (in the original paper)
- **d_k = d_v = d_model/h = 64**: Dimension of each head
- **W_i^Q, W_i^K, W_i^V**: Learned projection matrices for each head
- **W^O**: Final output projection

### What Each Head Learns

Research has shown that different attention heads specialize in different linguistic phenomena:

**Head 1:** Might focus on **syntactic dependencies**
```text
"The cat that I saw yesterday was cute"
# Head 1 connects: "cat" ↔ "was cute" (main subject-predicate)
```

**Head 2:** Might capture **semantic relationships**
```text
"The bank can guarantee deposits will cover tuition"
# Head 2 connects: "deposits" ↔ "tuition" (financial concepts)
```

**Head 3:** Might handle **positional relationships**
```text
"Yesterday I went to the store"
# Head 3 connects: "Yesterday" ↔ "went" (temporal modifier)
```

## The Complete Transformer Architecture

Now that we understand attention, let's see how it fits into the complete Transformer architecture.

### Encoder-Decoder Structure
The Transformer follows the classic encoder-decoder pattern:
- **Encoder:** The job of the encoder is to take the input sequence of words and create a rich, continuous representation of it. Each layer of the encoder
has two main sub-layers: a **multi-head self-attention mechanism** and a **position-wise fully connected feed-forward network**. The self-attention part is key—it allows the encoder to look at all the words in the input sentence at once and understand the relationships between them, no matter how far apart they are. The final encoder layer's output is then passed to the decoder. 
- **Decoder:** It works similarly to the encoder but has a crucial third sub-layer: a **multi-head attention layer over the output of the encoder**. This is where the *magic* happens! This layer allows the decoder to "look at" the encoded representation from the encoder and decide which parts are most relevant for generating the next word in the output sequence. The decoder generates the output sequence one word at a time, using the previously generated words as additional input.
---

*The attention mechanism continues to evolve, and its applications extend far beyond natural language processing into computer vision, speech recognition, and multimodal AI systems.*