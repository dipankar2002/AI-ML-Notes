# 📘 Week 1 — AI and ML: Fast-Tracking the Course of AI

**Date:** 18/01/2026

## 📌 Class Overview

This was the first class of the bootcamp. It covered the fundamentals of what AI, ML, and Deep Learning actually are, why they matter, and how we got from "write rules by hand" to modern systems like ChatGPT. The class used **language understanding** as a running example to explain the historical progression: rules → statistics → embeddings → sequence models → Transformers → LLMs.

## 🎯 What We Learned

- What learning, knowledge, and intelligence mean (as foundational concepts before defining AI)
- What AI is, and the difference between AI, ML, and Deep Learning
- Why rule-based systems (Expert Systems) failed for real-world tasks
- What Machine Learning is and how it works intuitively
- The two AI Winters and why they happened
- Why AI exploded after 2012 (Data + Compute + Algorithms)
- Why language is hard for computers (ambiguity)
- The evolution of NLP: dictionaries → statistical patterns → word embeddings → sequence models (RNNs) → Transformers
- What word embeddings are and how "word math" works
- The Attention mechanism and why it powers Transformers
- How ChatGPT works at a high level (Transformer + Data + Next-word Prediction)
- Where AI is heading in 2024–2025 (Multimodality, Reasoning, Agents)

---

## 1. Foundations: Learning, Knowledge, Intelligence

Before defining AI, the class built up from first principles.

### Learning
> Changing yourself based on experience — essentially, adjustment.

**Learning is a loop:**

```
Try something → See what happens (feedback) → Adjust → Repeat
```

### Knowledge
Two types of knowledge:
- **Explicit Knowledge** — knowledge that can be written down/stated (facts, rules)
- **Implicit Knowledge** — knowledge we have but can't easily articulate (intuition, "gut feel")

### Intelligence
> Intelligence is the ability to achieve goals in a wide range of situations.

---

## 2. What is AI?

**AI = The Goal**, not a specific method.

> Making machines do things that would require intelligence if a human did them.

### Examples
- Recognizing faces
- Understanding language
- Playing chess
- Driving a car

---

## 3. Attempt #1: Write the Rules (Expert Systems)

The earliest approach to AI: hand-write explicit rules for every situation.

```
"If the email contains 'free money' → spam"
"If temperature > 100°F → fever"
"If chess piece can move to square AND square has enemy piece → capture"
```

This approach is called **Expert Systems** or **Rule-Based AI**.

### The Problem with Rules

Rules break down for tasks humans do *intuitively*:

| Task | Why Rules Fail |
|---|---|
| Recognizing a Face | Can't define exact distances/curves for every angle and lighting |
| Understanding Sarcasm | "Oh, great." — rules can't capture tone/context |
| Decoding Idioms | "It's raining cats and dogs" — a rule-based system looks for literal animals |

> ⚠️ Important: Rule-based AI is **brittle** — it can't generalize beyond what was explicitly coded.

---

## 4. Attempt #2: Let Machines Learn (Machine Learning)

Instead of writing rules manually:

> Show the machine thousands of examples and let it figure out the patterns.

**This is called Machine Learning (ML).**

### How Machine Learning Works (Intuitively)

1. Show the machine many examples
2. Machine makes a guess
3. Tell it if it was right or wrong
4. Machine adjusts itself slightly
5. Repeat millions of times

> 💡 **The Analogy** (from class): "Like a child learning by trial and error"

### AI vs ML vs Deep Learning

ML and Deep Learning are **methods** for achieving the **goal** of AI:

```
AI (The Goal: Smart Machines)
 └── ML (One Method: Learning from Data)
       └── Deep Learning
```

| Concept | Meaning |
|---|---|
| AI | The overall goal — making machines behave intelligently |
| ML | A method to achieve AI — learning patterns from data |
| Deep Learning | A subset of ML using deep neural networks |

---

## 5. Early ML: Modest Successes

### What Worked
- Spam filters got better
- Recommendation systems (Netflix, Amazon)
- Basic image recognition

### The Bottleneck
- Couldn't have a conversation
- Couldn't understand a paragraph
- Couldn't recognize objects as well as a toddler

### Why Early ML Was Limited

**Bottleneck 1: Not enough data**
- ML requires massive amounts of examples to learn patterns effectively
- Where do you get millions of labeled examples? The internet was in its infancy.

**Bottleneck 2: Not enough compute**
- Learning from data requires intense mathematical calculations
- Computers were too slow for deep training
- Training on millions of examples would take years

---

## 6. AI Winters: When Hope Died (Twice)

| Period | Name | What Happened |
|---|---|---|
| 1970s | The First Collapse | Researchers promised human-level AI in 20 years. When they failed to deliver, funding dried up and interest vanished. |
| 1980s–90s | The Expert System Bust | Expert systems were supposed to revolutionize business, but they were too brittle for the real world. Funding vanished again. |

> "The field became a joke. Saying you worked on 'AI' was career suicide." — instructor's quote from class

---

## 7. Why AI Exploded (After 2012)

Three ingredients converged around 2012:

| # | Ingredient | Detail |
|---|---|---|
| 01 | **Data** — The Internet | Massive datasets of text, images, and video became available for the first time |
| 02 | **Compute** — GPUs | Graphics cards, built for gaming, turned out to be perfect for the math ML needs |
| 03 | **Algorithms** — Deep Learning | Researchers finally cracked how to train much deeper and more complex neural networks |

> 📌 Key Point: **All three ingredients converged around 2012** — this is the turning point that led to the modern AI boom.

---

## 8. Deep Dive: Why Language is Hard for Computers

The instructor used **language understanding** as the lens for the rest of the class.

> "I saw the man with the telescope."
> Who had the telescope? Me or the man?

Computers need precision, but human language is **messy, ambiguous, and context-dependent**. A single sentence can have multiple valid interpretations, making it one of the hardest problems in AI.

The instructor also pointed out ambiguity in the course title itself:
> "Fast-tracking the AI course" vs. "Fast-tracking the course of AI"

---

## 9. NLP Attempt #1: The Dictionary Approach

Idea: define words using fixed dictionary meanings.

**Example: "Apple"**
> The round fruit of a tree of the rose family, which typically has thin red or green skin and crisp flesh.

### The Problem
A dictionary definition isn't enough for real-world language:
- "Apple" is also a trillion-dollar company
- "Apple" is a famous record label
- **Context changes everything**

---

## 10. NLP Attempt #2: Statistical Patterns

Idea: instead of dictionary meanings, use word co-occurrence patterns.

```
"If 'New' is followed by 'York' → City"
```

> 📌 This approach powered **Google Translate** for nearly a decade.

### The Problem: Pattern Matching ≠ Understanding
The machine could predict the next word based on frequency, but it had no concept of what the words actually meant.

---

## 11. The Breakthrough: Words as Numbers (Word Embeddings)

Computers don't understand words — they understand **numbers**. The challenge: how do we turn a word into a list of numbers that captures its meaning?

```
"Apple" → [0.92, -0.14, 0.05, ...]
```

### What is it?
**Word Embeddings**: instead of representing a word with one number, we give it a **list of numbers** — a vector. Each number in the vector represents a "dimension" of meaning.

### Example: Dimensions of Meaning

| Word | Royalty | Gender (M) | Edibility |
|---|---|---|---|
| King | 0.98 | 0.95 | 0.01 |
| Queen | 0.97 | 0.05 | 0.02 |
| Apple | 0.02 | 0.00 | 0.94 |

> "King" and "Queen" have similar numbers for royalty but vastly different values for gender. This is how machines represent concepts as data.

### Words as Positions in Space

If a word is a list of numbers, it's a **point on a map**. Similar words are close together, while different words are far apart (e.g., King/Queen close together; Man/Woman close together; Apple/Orange close together).

```
Dimension B
   ↑
   |  king    queen
   |
   |  man     woman
   |
   |                          apple   orange
   |
   |  cat  dog
   +--------------------------------→ Dimension A
```

### The Magic of Embeddings — Word Math

Because words are vectors, you can do arithmetic on meaning:

```
King − Man + Woman = Queen
Paris − France + Italy = Rome
Walking − Walk + Swim = Swimming
```

> 💡 Additional Note: This "word math" works because the *direction* between vectors captures a relationship (e.g., "royal → gender" or "capital → country"), and applying the same direction to a different word gives an analogous result.

---

## 12. The Limitation of Basic Word Embeddings

Can embeddings solve our earlier ambiguity problem?

```
"I went to the bank to deposit cash"
"I sat on the bank of the river"
```

### The Problem
In basic word embeddings, each word has exactly **ONE position** in space. The model can't distinguish between a financial institution and a riverbank because it doesn't look at the surrounding words.

---

## 13. NLP Attempt #4: Sequence Models (RNNs)

> ⚠️ The class material moves from "Attempt #2" directly to "Attempt #4" — Attempt #3 was not explicitly labeled in the slides (word embeddings appear to be the unlabeled "attempt #3").

### Idea
Process the sentence one word at a time, building up understanding as you go.

```
"I"
"I went"
"I went to"
"I went to the"
"I went to the bank..."
```

The model maintains a "memory" of what it has read so far. (This is what RNN = **Recurrent Neural Network** does.)

### The Problem: Forgetting (Long-Range Dependency Failure)

```
"The cat, which was sitting on the mat that I bought from the store 
near the old church on the corner, was happy."
```

By the time a sequence model reaches "was," it has often **forgotten** the subject at the beginning.

> RNNs struggle with long sentences because they process information linearly and have limited memory capacity.

---

## 14. The Stage Is Set (~2017)

**What we had:**
- Word embeddings
- Sequence models (RNNs)
- Tons of internet data
- Powerful GPUs

**The Problem:** The Word-by-Word Bottleneck — processing text sequentially was slow, and models still "forgot" the beginning of long sentences.

> *"What if there was a better way?"*

---

## 15. The Transformer & Attention

### What is it?
The **Transformer** is an architecture built entirely on the mechanism of **Attention** — no recurrence, no convolution.

### The Old Way vs. The New Way

| | The Old Way (RNN) | The New Way (Transformer) |
|---|---|---|
| Processing | **Sequential** — process words one by one, like reading a book start to finish | **Simultaneous** — look at every word in the sentence at the same time |
| Mechanism | Hidden memory state | **Attention** — the model "attends" to the most relevant words, no matter how far apart they are |

### How does it work? — Attention in Action

```
"The animal didn't cross the street because it was too tired."
```

When the model processes the word **"it"**, the attention mechanism tells it to pay the most attention to **"animal"** — this is how the model resolves what a pronoun refers to.

### Example: Same Structure, Different Attention

| Scenario | Sentence | "it" refers to |
|---|---|---|
| A | The **animal** didn't cross the street because **it** was too *tired*. | animal |
| B | The animal didn't cross the **street** because **it** was too *wide*. | street |

By changing just one word (**tired** vs **wide**), the model's attention shifts, correctly identifying what "it" refers to in each context.

### Why Attention Is Powerful

| # | Benefit | Explanation |
|---|---|---|
| 01 | No Forgetting (More Memory) | Every word "sees" every other word simultaneously, no matter how far apart |
| 02 | Speed (Parallel Processing) | Unlike RNNs (word-by-word), Transformers process all words at once → much faster training |
| 03 | Deep Understanding (Context) | Builds a mathematical map of how every word relates to every other word in context |

### Transformer Traits
- No recurrence. No convolution. Just attention.
- The foundation of all modern Large Language Models (LLMs).
- Enables massive scaling of data and compute.

---

## 16. How ChatGPT Works (Simplified)

```
Transformer + Internet Data + Prediction
```

| # | Component | Description |
|---|---|---|
| 01 | **The Engine** (Architecture) | A massive neural network built entirely on the Attention mechanism |
| 02 | **The Knowledge** (Training Data) | Trillions of words from books, articles, and the entire public internet |
| 03 | **The Task** (Objective) | A simple goal: given a sequence of words, predict the most likely next word |

### Why "Predict the Next Word" Creates Understanding

| Capability | How it emerges from next-word prediction |
|---|---|
| Grammar & Syntax | Predicting the next word requires the model to internalize language structure and rules |
| Factual Knowledge | Correct predictions rely on learned factual relations between concepts |
| Logic & Reasoning | Following a chain of reasoning lets the model anticipate the appropriate continuation |

### How It Generates Text (Iterative Loop)

```
Step 1: The
Step 2: The quick
Step 3: The quick brown
Step 4: The quick brown fox...
```

Generation is an **iterative loop**: each predicted word is added back to the input, and the model predicts the next word based on the updated context.

---

## 17. The Unexpected Discovery: Bigger = Smarter

Scaling along three axes led to unexpected new capabilities:

| # | Axis | Scaling |
|---|---|---|
| 01 | Data | More Tokens — millions → trillions of words of training data |
| 02 | Parameters | More Size — millions → hundreds of billions of internal connections |
| 03 | Compute | More Power — days → months of training on thousands of GPUs |

> ⚠️ **The Result: Emergent Capabilities.** Reasoning, coding, and logic appear spontaneously as models get larger — these were not explicitly programmed in.

### Foundation Models

| | The Old Paradigm | The New Paradigm |
|---|---|---|
| Approach | **Task-Specific AI** — separate models for translation, summarization, sentiment analysis, etc. | **General-Purpose AI** — one massive model trained on everything, capable of performing any language task through prompting |

> The "Foundation" is the base knowledge. We no longer build tools from scratch; we build on top of these giants.

---

## 18. Where We Are (2024–2025)

| Trend | Description |
|---|---|
| **Multimodality** | AI is no longer just text — it can see images, hear voices, and speak back in real-time |
| **Reasoning** | New models are designed to "think" before they speak, solving complex math and logic problems |
| **Agents** | The shift from chatbots to agents that can use tools, browse the web, and complete multi-step tasks |

---

## 💡 Additional Notes

> 💡 Additional Note: RNN stands for **Recurrent Neural Network** — the slides used the term without expanding the acronym.

> 💡 Additional Note: The "Attempt #3" slide for NLP wasn't explicitly numbered in the source material — based on the flow, word embeddings functioned as the conceptual "attempt #3" between statistical patterns and sequence models.

> 💡 Additional Note: The word-math examples (King − Man + Woman = Queen) are a famous real-world result from Word2Vec (a well-known word embedding technique), which the instructor used to illustrate embedding intuition.

---

## 🔑 Key Takeaways

1. **AI is a goal, not a method** — ML and Deep Learning are ways of achieving it.
2. **Rule-based AI (Expert Systems) fails** because human tasks like language, vision, and sarcasm are too intuitive to hard-code.
3. **ML = learn from examples**, not explicit rules — via a guess → feedback → adjust loop.
4. AI had **two "winters"** caused by overpromising and underdelivering.
5. AI's 2012 resurgence came from **three converging factors**: Data (internet), Compute (GPUs), Algorithms (deep learning).
6. **Language is hard** because it's ambiguous and context-dependent.
7. **Word embeddings** represent word meaning as vectors — enabling "word math" and similarity comparisons.
8. Plain embeddings **can't handle context** (e.g., "bank" of a river vs. a bank) — this motivated sequence models.
9. **RNNs process sequentially** and suffer from "forgetting" long-range context.
10. **Transformers replaced RNNs** using **Attention** — processing all words simultaneously and dynamically weighting relevance.
11. **ChatGPT = Transformer + massive internet data + next-word prediction**, applied iteratively to generate text.
12. **Scale (data + parameters + compute) leads to emergent capabilities** like reasoning — this shift birthed **Foundation Models**.
13. Current AI trends: **Multimodality, Reasoning, and Agents**.

---

## 🧠 Quick Revision

- **Learning** = adjusting based on experience (try → feedback → adjust → repeat)
- **Knowledge** = Explicit (statable) vs. Implicit (intuitive)
- **Intelligence** = ability to achieve goals across situations
- **AI** = the goal of making machines act intelligently
- **ML** = a method: learn patterns from data instead of writing rules
- **Expert Systems** = rule-based AI; brittle, fails on intuitive tasks
- **AI Winters**: 1970s (overpromised general AI) and 1980s–90s (Expert System bust)
- **2012 AI boom drivers**: Data (internet) + Compute (GPUs) + Algorithms (Deep Learning)
- **Word Embeddings**: words → vectors of numbers capturing meaning; similar words = close vectors
- **Word math**: King − Man + Woman = Queen
- **RNNs**: process word-by-word; struggle with long-range memory ("forgetting")
- **Transformer**: processes all words simultaneously using **Attention**
- **Attention**: lets the model dynamically focus on the most relevant words regardless of distance
- **ChatGPT** = Transformer + huge training data + next-word prediction, applied iteratively
- **Emergent capabilities**: reasoning/logic/coding appearing as models scale up
- **Foundation Models**: one general-purpose model vs. many task-specific models
- **2024–2025 trends**: Multimodality, Reasoning, Agents

---

## ❓ Self-Check Questions

1. What is the difference between AI, ML, and Deep Learning?
2. Why did rule-based Expert Systems fail for tasks like face recognition or sarcasm detection?
3. Describe the 5-step loop of how Machine Learning works.
4. What caused the two AI Winters?
5. What three factors converged around 2012 to enable the AI boom?
6. Why is human language ambiguous, and how does that create challenges for computers?
7. What is a word embedding, and why is it useful compared to a dictionary definition?
8. What problem do basic word embeddings have with words like "bank" that RNNs and Transformers try to solve?
9. What is the Attention mechanism, and how does it help resolve what a pronoun like "it" refers to?
10. In simple terms, how does ChatGPT generate text one word at a time?