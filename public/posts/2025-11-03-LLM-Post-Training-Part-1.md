---
title: "Post Training in Large Language Models - Part 1"
date: "2025-11-03"
author: "Tai Vu"
excerpt: "Post training in large language models."
tags: ["LLM", "AI"]
readTime: "8 min read"
featured: true
slug: "llm-post-training"
---

# 🧠 Post-Training in Large Language Models: From Raw Intelligence to Practical Use

When we hear about models like ChatGPT, Claude, Gemini, Grok, etc. It's easy to think they were simply "trained once" on huge datasets and then magically became conversational assistants. In reality, what makes these models helpful, potile, and safe to interact with isn't only **pre-training** it's the stages that come after it: **mid-training**, **post-training**. In this series of posts, we'll only go into detail on post-training.

---

## I. Introduction
Think about this for a moment: GPT-3 was released in 2020. ChatGPT launched in late 2022. Same core model, but wildly different user experiences. What made the difference?

![base-model vs post-trained model](/posts/images/gpt3-vs-chatgpt.png)

### A Quick Experiment
Imagine you ask a language model: "How do I fix my car?"
**Before post-training,** it might respond: "How to fix a bike?"
**After post-training:** "I'd be happy to help! Could you tell me what specific issue you're experiencing?"
What changed? The model went through post-training—a set of techniques that transformed raw intelligence into practical assistance.

### The Core Challenge
Pre-trained language models are like incredibly well-read people who've never learned to have a conversation. They've absorbed massive amounts of information from the internet by learning to predict the next word in billions of text examples. This gives them knowledge about almost everything—but no understanding of how to actually help you.

Ask them to write code, and they might describe what code should do rather than writing it. Ask a question, and they might generate another question. They have intelligence without direction.

![pretrained-model](/posts/images/pretrained-model.png)


### Post-Training: Teaching Models to Be Useful
This is where fine-tuning and reinforcement learning come in—the two core techniques of post-training. Together, they teach models to:
- Have natural conversations and remember context.
- Follow instructions accurately.
- Refuse harmful requests while staying helpful.
- Use external tools when needed.
- Think step-by-step through complex problems.
- Adapt their style to different situations.

Every modern AI assistant you interact with—ChatGPT, Claude, Gemini—relies on these post-training techniques. It's what got us from research curiosity to practical tool.

![post-trained-model](/posts/images/post-trained-model.png)

### What's Ahead
In this post, you'll discover:

- How fine-tuning and reinforcement learning actually work (we'll use some helpful analogies).
- Why frontier labs use both techniques, not just one.
- Real examples: teaching models to reason and stay safe.
- How you might apply these techniques yourself.

## II. The LLM Training Pipeline
Training a large language model isn't a single step—it's a journey with three distinct stages. Understanding this pipeline helps you see where post-training fits and why it matters.

### A. Pre-training: Building Raw Intelligence
Pre-training is where it all begins. The model learns by doing one simple task over and over: predict the next word.

Feed it "Once upon a midnight" and it learns to predict "dreary." Show it "The sky is" millions of times, and it learns "blue" is highly probable. When you add "The sun is setting, the sky is," suddenly "orange" becomes more likely.

This happens across billions of text examples from across the internet—books, websites, articles, everything. Through this seemingly simple task, the model learns:
- Grammar and language structure
- Facts about the world
- Concepts and relationships
- Patterns in how humans write

But here's the catch: **it only knows how to predict the next word**. It doesn't know it's supposed to help you, answer questions, or have conversations.

Pre-training is expensive—it can take months of compute and enormous amounts of data. You start with completely random weights, and slowly the model learns to make sense of language. The result is a "base model" with incredible knowledge but no sense of purpose.

### B. Mid-training: Specialized Enhancement
After pre-training comes mid-training—think of it as "focused reading" rather than "reading everything."

Mid-training continues the same next-word prediction task, but on carefully curated datasets. This stage is used for:
**Adding new languages:** Maybe your base model knows English well, but you want it to master Chinese or Vietnamese. Mid-training on high-quality Chinese texts teaches it these languages without starting from scratch.
**New modalities:** Want your text model to understand images or audio? Mid-training can introduce these capabilities.
**Longer context:** Models initially trained on shorter texts can learn to handle much longer documents—from a few thousand words to hundreds of thousands.

Mid-training is like having your well-read friend take specialized courses in specific subjects. The foundation is there; now you're building expertise.

### C. Post-training: Making Models Useful
Finally, we reach post-training—the transformation from "knowledgeable" to "helpful."

This stage uses two powerful techniques:
**Fine-tuning (Supervised Fine-Tuning/SFT):** You show the model specific inputs and the exact outputs you want. "When someone asks X, respond with Y." The model learns to mimic these patterns.
**Reinforcement Learning (RL):** Instead of showing exact answers, you let the model generate responses and then grade them. "That response was helpful—good job!" or "That was unsafe—try again." The model learns from feedback.

Together, these techniques teach the model to:
- Understand it's an assistant whose job is to help
- Maintain conversations naturally
- Follow instructions precisely
- Refuse inappropriate requests
- Use tools and reason through problems
- Adapt its style to different contexts

### The Complete Journey
Think of it this way:

- **Pre-training:** The model reads an entire library, absorbing everything but with no specific goal
- **Mid-training:** The model studies specialized advanced books in targeted subjects
- **Post-training:** The model learns to be an effective tutor—how to teach, interact, and genuinely help people

Post-training is what makes a model ready for the real world. It's the difference between a brilliant recluse and a helpful assistant.

Now that you understand the pipeline, let's dive into the heart of post-training: how fine-tuning and reinforcement learning actually work, and why they're so different.

Let's start by seeing where post-training fits in the complete journey of training a language model.