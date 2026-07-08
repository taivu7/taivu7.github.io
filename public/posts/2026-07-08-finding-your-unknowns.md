---
title: "The Map Is Not the Territory: Finding Your Unknowns with AI Coding Agents"
date: "2026-07-08"
author: "Tai Vu"
excerpt: "As models get stronger, the bottleneck stops being the model and becomes you. A field guide to discovering your unknowns before, during, and after implementation — with runnable code."
tags: ["AI", "Agentic Coding", "Prompting"]
readTime: "12 min read"
featured: true
slug: "finding-your-unknowns"
---

# The Map Is Not the Territory: Finding Your Unknowns with AI Coding Agents

> This post is my own reading of, and riff on, Thariq's excellent essay **"A Field Guide to Fable: Finding Your Unknowns"** ([@trq212](https://x.com/trq212/status/2073100352921215386)). The framework of *unknowns* and most of the technique names are his; the code, the mistakes, and the opinions are mine. If you have time, read the original first.

There's an old idea from general semantics: **the map is not the territory.** The map is a representation — useful, but never the thing itself. The territory is reality, in all its messy detail.

When I work with a coding agent, the *map* is everything I hand it: my prompt, my skills files, the context I load. The *territory* is where the work actually happens: the codebase, its real constraints, the production database that behaves nothing like my mental model of it.

The gap between the two is what Thariq calls **unknowns**. Every time the agent hits an unknown, it has to guess what I wanted. A little work, a few guesses. A lot of work, a lot of guesses — and the guesses compound.

Here's the shift that surprised me. With earlier models, quality was capped by the *model*: it just couldn't do some things, no matter how well I prompted. With the current generation, I keep finding that the cap has moved. The work is now bottlenecked by **my ability to clarify its unknowns**, not by the model's raw ability. The model will faithfully build the wrong thing if I never noticed the thing was ambiguous.

So the skill I've been practicing isn't "better prompting" in the write-a-magic-incantation sense. It's *discovering what I don't know* — before, during, and after implementation — and doing it cheaply, before it gets expensive to fix.

---

## The four quadrants of unknowns

Before the techniques, the mental model. When I bring a problem to an agent, I try to sort what I know into four buckets:

| | **I'm aware of it** | **I'm not aware of it** |
|---|---|---|
| **I know it** | **Known Knowns** — what's in my prompt. "Use Postgres, keep the existing auth flow." | **Unknown Knowns** — so obvious I'd never write it down, but I'd recognize it instantly. "Oh, obviously the timestamps should be UTC." |
| **I don't know it** | **Known Unknowns** — I know there's a gap. "I haven't decided how to page this API." | **Unknown Unknowns** — I haven't even considered it. "There's a rate-limit interaction I've never heard of." |

The best agentic coders I've watched aren't magic — they simply have **very few unknowns**. They know the codebase and the model's behavior in enough detail that most of the map already matches the territory. But they also *assume* unknowns exist, and they go looking for them.

The good news: this is a skill, and you can get better at it *with the agent's help*. The agent reads the codebase and the internet faster than you, knows more than you about the average topic, and recovers from a wrong guess quickly. Below are the patterns I actually use, grouped by when in the work they happen.

---

## Before implementation

### 1. The Blind Spot Pass

When I'm starting in an unfamiliar part of the codebase, or doing unfamiliar work (a new domain, a design task), I'm swimming in *unknown unknowns*. I don't even know what questions to ask.

So I ask the agent to find them for me. I literally use the words **"blind spot pass"** and **"unknown unknowns"**, and — this is the important part — I tell it who I am and what I already know, so it can calibrate.

Example prompt:

```text
I'm adding a new OAuth provider, but I know almost nothing about the auth
modules in this codebase. Do a "blind spot pass": help me find my relevant
unknown unknowns and explain them, so I can prompt you better afterward.
```

And here's the same idea as a small script you can drop into a repo. It reasons about the module and hands back the questions I didn't know to ask:

```python
import anthropic

client = anthropic.Anthropic()

BLINDSPOT_PROMPT = """I'm about to work in a part of the codebase I don't know well.
Do a "blind spot pass" to surface my *unknown unknowns*.

Context about me:
- Backend dev, strong in Python, new to this auth module.
- Goal: add a new OAuth provider.

Return three sections:
1. Questions I don't yet know to ask
2. Concepts / prior art I'm probably missing
3. Common pitfalls in this kind of work, and how they usually bite
"""

resp = client.messages.create(
    model="claude-opus-4-8",
    max_tokens=2048,
    thinking={"type": "adaptive"},  # let it reason before answering
    messages=[{"role": "user", "content": BLINDSPOT_PROMPT}],
)

for block in resp.content:
    if block.type == "text":
        print(block.text)
```

I run this *before* I write my real prompt. Half the time the output changes what I was going to ask for.

> Note on `thinking={"type": "adaptive"}`: on current models this lets the model decide how much to reason before answering. The response can contain `thinking` blocks (whose text is empty by default) alongside `text` blocks — hence the `if block.type == "text"` filter.

### 2. Brainstorm and prototype

Some unknowns are *unknown knowns*: I'll know it when I see it, but I can't articulate it up front. Visual design is the classic case — I can't describe the layout I want, but I'll react instantly to a mockup.

For these, I don't implement. I ask for cheap, throwaway artifacts I can react to. Finding an unknown known during *prototyping* is far cheaper than finding it during *implementation*, where a small spec change can mean a large code change the agent then has to unwind.

Example prompts:

```text
I want a dashboard for this data but I have no visual taste and don't know
what's possible. Make me a single HTML page with 4 wildly different design
directions so I can react to them.
```

```text
Before wiring anything up, make one HTML file mocking the new editor toolbar
with fake data. I want to react to the layout before you touch the real app.
```

I now start almost every session with a brainstorm phase — not to get the answer, but to *set the scope with intent*. The agent regularly proposes an approach I'd have missed, and occasionally it misses the forest for the trees. Brainstorming keeps me from scoping the work too narrow or too wide.

### 3. Interview me

After brainstorming, there are usually still unknowns hiding. So I flip the roles: I ask the agent to **interview me**, one question at a time, prioritizing questions whose answer would change the architecture.

This is my favorite technique, and it's the one that's most naturally *code*, because it's a loop. The prompt is one line — the value is in the structure:

```text
Interview me one question at a time about anything ambiguous in this task.
Prioritize questions where my answer would change the architecture.
```

Here's a runnable interview loop. It keeps asking until it has enough to write a short spec, then stops:

```python
import anthropic

client = anthropic.Anthropic()

SYSTEM = """You are helping a developer surface the unknowns in a task before they build.
Interview them ONE question at a time. Prioritize questions whose answer would change the
architecture or approach. After each answer, ask the single most valuable next question.
When you have enough to write a short spec, STOP asking and output a section that starts
with the heading "## Spec" summarizing everything you learned."""

messages = [
    {"role": "user",
     "content": "I want to add a rate limiter to our public API. Interview me."}
]

while True:
    resp = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=1024,
        system=SYSTEM,
        messages=messages,
    )
    reply = resp.content[0].text
    print("\nClaude:", reply)
    messages.append({"role": "assistant", "content": reply})

    if "## Spec" in reply:          # the agent decided it has enough
        break

    answer = input("\nYou: ")
    messages.append({"role": "user", "content": answer})
```

Two things make this work. First, the API is stateless, so I resend the whole `messages` list each turn — that's how the agent "remembers" my answers. Second, I give it an explicit *exit condition* (`## Spec`) so the interview converges instead of running forever. When it finishes, I keep that spec and feed it into the real build.

### 4. References

Sometimes I can't describe what I want *at all* — I don't have the vocabulary, or it'd take an hour to write out. In that case the best answer isn't a description, it's a **reference**. And the best reference is source code.

If a library implements exactly the backoff behavior I want, I don't paraphrase it. I point the agent at the folder and tell it what to look for — even across languages:

```text
The Rust crate in vendor/rate-limiter implements the exact backoff behavior
I want. Read it and reimplement the same semantics in our TypeScript API client.
```

Source is a richer reference than a screenshot or a doc, because it carries the structure and the edge cases, not just the surface.

### 5. Implementation plan

When I think I'm ready to build, I ask for a plan — but not an exhaustive one. I ask it to **lead with the decisions I'm most likely to change**: data models, type interfaces, user-facing flows. The mechanical refactoring goes at the bottom, where I'll skim it.

```text
Write an implementation plan, but lead with the decisions I'm most likely to
tweak: data-model changes, new type interfaces, and anything user-facing.
Put the mechanical refactoring at the bottom — I trust you on that part.
```

The point of the plan isn't approval theater. It's one more surface where an unknown can jump out at me before it's baked into a hundred lines of code.

---

## During implementation

### Implementation notes

No amount of planning removes every unknown. The agent *will* discover, mid-build, that an edge case forces a different approach. So I ask it to keep a running log of the decisions it makes along the way — a file I can learn from next time.

```text
Keep an implementation-notes.md file. If you hit an edge case that forces you
to deviate from the plan, pick the conservative option, log it under a
"Deviations" heading with a one-line reason, and keep going.
```

The resulting file looks something like this, and it's gold for the *next* attempt:

```markdown
# Implementation Notes

## Deviations
- Plan said "one token bucket per user." Found that anonymous traffic has no
  user id, so I keyed the bucket on API-key-or-IP instead. Conservative: this
  can over-limit shared NATs, but it never under-limits. Flagged for review.

## Open questions
- Should the limiter be fail-open or fail-closed if Redis is down? Assumed
  fail-open for now.
```

Those "Deviations" and "Open questions" are unknowns the agent surfaced *for* me, without my having to predict them.

---

## After implementation

### Pitches and explainers

Shipping isn't done when the code works — it's done when someone approves it. A good pitch artifact does two things: it gets reviewers who share your *original* unknowns up to speed fast, and it reassures the experts who would've anticipated the failure modes that you accounted for them.

```text
Package the prototype, the spec, and the implementation notes into a single doc
I can drop in Slack to get buy-in. Lead with the demo, then the key decisions,
then the risks and how they're handled.
```

### Quizzes

This one changed how I merge. After a long session, the agent has often done more than I fully absorbed. Reading the diff gives me a shallow understanding, because so much of the behavior depends on code paths I didn't touch.

So I ask it to **quiz me** — and I only merge after I pass. If I can't answer, that's an unknown I was about to ship blind.

```python
import subprocess
import anthropic

client = anthropic.Anthropic()

diff = subprocess.check_output(["git", "diff", "main...HEAD"], text=True)

resp = client.messages.create(
    model="claude-opus-4-8",
    max_tokens=2048,
    messages=[{
        "role": "user",
        "content": (
            "Here is the diff for this branch. First give me a short report "
            "explaining what changed and why, with the intuition behind each "
            "decision. Then a 5-question quiz (answer key at the very bottom) "
            "that I must pass before I merge.\n\n"
            f"```diff\n{diff}\n```"
        ),
    }],
)

print(resp.content[0].text)
```

Failing my own quiz is one of the more humbling — and useful — feedback loops I've added.

---

## How it comes together

None of these techniques is mandatory, and I don't use all of them every time. They're a toolbox. The blind spot pass is for when I'm in foreign territory. The interview is for when I *think* I know what I want but suspect I've left gaps. The quiz is for when the agent ran ahead of me.

What ties them together is a single move, repeated: **spend a little to find out what you don't know, before it costs a lot.** Every brainstorm, interview, prototype, reference, and quiz is a cheap probe into the gap between my map and the territory.

The stronger the models get, the more this matters — because the more they can do, the more the quality of the outcome depends on how well I've defined the unknowns. When a long task comes back wrong, my first instinct now isn't "the model failed." It's "I didn't clarify enough."

So the next time you start a project, don't start by writing the perfect prompt. Start by asking the agent to help you find what you don't yet know.

---

*Credit and thanks to [Thariq (@trq212)](https://x.com/trq212/status/2073100352921215386) for the "map and territory" framing and the field-guide structure this post is built on. Go read the original.*
