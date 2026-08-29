# 📘 Week 2 — AI and ML: Neural Networks from Scratch

**Date:** 24/01/2026
**Subtitle (from slides):** How Machines Actually Learn

## 📌 Class Overview

This class opened the hood on neural networks — moving past "AI learns from data" (Week 1) into *exactly how* that learning happens, mechanically. It covered the anatomy of a single neuron, why non-linear activation functions are mathematically necessary, how a network measures its own mistakes (loss), how it figures out which weights caused the mistake (backpropagation), and how it actually corrects itself (gradient descent + learning rate). It ends by showing that this exact same loop — forward pass → loss → backpropagate → update — is what trains everything from a tiny 20-parameter network up to GPT-4's ~1 trillion parameters.

> 💡 **Note on this version:** As requested, every major topic has a "In plain words" explanation added in simple language with analogies, so this can be read by someone with zero background. Anything added beyond the literal slide content is marked with 💡, so you can tell what was actually taught vs. what's extra help.

## 🎯 What We Learned

- Quick recap: why traditional (rule-based) programming doesn't scale, and how ML differs
- A concrete example prediction problem (house price vs. size) to motivate neural networks
- The core training loop in one sentence: guess → measure error → adjust → repeat
- The anatomy of a single neuron: inputs, weights, bias, summation, activation
- The neuron's math formula: `output = activation(Σ wᵢxᵢ + b)`
- Why activation functions are *necessary* (the "linearity problem")
- Common activation functions: Sigmoid, Tanh, ReLU, GELU/Swish
- The Loss Function, specifically Mean Squared Error (MSE)
- Backpropagation — how error gets traced back to individual weights
- Gradient Descent — the strategy for reducing error step by step
- Learning Rate — how big each correction step should be, and what goes wrong if it's too big/small
- The full Training Loop, combining all of the above into 4 steps
- How this exact loop scales from tiny networks (~20 parameters) to GPT-4 (~1 trillion parameters)

---

## 1. Quick Recap: From Rules to Patterns

A short recap bridging back to Week 1's content, before diving into new material.

### Traditional Programming
- **Rules don't scale:** Humans cannot write rules for every possible scenario in complex tasks.
- **Rigid Logic:** Systems fail when encountering data that doesn't fit predefined rules.

### Machine Learning
- **Pattern Discovery:** Finding consistent mathematical relationships within large datasets.
- **The Goal (of this class):** Understanding the fundamental mechanics of how this discovery happens.

> 💡 **In plain words:** Last class covered *that* machines learn from data instead of hand-written rules. This class is about *how*, mechanically, that learning actually works — what's really going on inside the "guess → correct → repeat" loop we saw in Week 1. Think of Week 1 as "what a car does" and this class as "opening the hood and looking at the engine."

---

## 2. The Simplest Prediction Problem

Neural networks solve the same problems we do intuitively. Consider the relationship between house size and price:

| Size (sq ft) | Price |
|---|---|
| 1,000 | $200,000 |
| 1,500 | $300,000 |
| 2,000 | $400,000 |
| 2,500 | ? |

**The Challenge:** How do we teach a machine to find the mathematical relationship (e.g., $200/sq ft) automatically?

> 💡 **In plain words:** Look at the pattern — every extra 500 sq ft adds $100,000 to the price. A human can eyeball this instantly and guess $500,000 for a 2,500 sq ft house. But a computer doesn't "eyeball" anything — it has no intuition. The whole rest of this class is essentially answering: *how do we get a machine to discover that "$200 per square foot" relationship completely on its own, using nothing but the data in this table?* This simple example (predicting a number from another number) is called a **regression problem**, and it's used throughout the class because it's easy to visualize.

---

## 3. Machine Learning in One Sentence

> Training is a simple, iterative loop of guessing and correcting.

1. **Start with a guess** — Initialize with random values.
2. **Measure error** — How far is the guess from the truth?
3. **Adjust** — Modify the guess to be slightly less wrong.
4. **Repeat** — Do this millions of times until error is minimized.

> 💡 **In plain words:** This is the exact same loop from Week 1 ("try → feedback → adjust → repeat"), but now let's map it to our house price example:
> 1. The model starts by *randomly* guessing something like "price = $37/sq ft" — a totally wrong guess, since it hasn't learned anything yet.
> 2. It checks: for a 1,000 sq ft house, this guess predicts $37,000, but the real answer is $200,000 — so it's very wrong.
> 3. It nudges its guess a little closer to reality — maybe now it guesses "$60/sq ft."
> 4. It repeats this nudging process over and over — sometimes millions of times — until its guess ("$200/sq ft") matches the real pattern in the data.
>
> Everything covered for the rest of this class (neurons, activation functions, loss, backpropagation, gradient descent) is really just *filling in the details* of exactly how steps 2, 3, and 4 work mathematically.

---

## 4. The Neuron: A Tiny Decision Maker

> The building block of AI is a simple mathematical function, not a biological mystery. It takes numbers in, performs basic arithmetic, and outputs a signal.

### 1. Inputs
Numerical data points representing features (e.g., house size, pixel intensity).

### 2. Weights & Summation
Importance assigned to each input, combined into a single weighted sum.

### 3. Activation
A non-linear function that decides whether the signal should "fire" or be suppressed.

> 💡 **In plain words:** The word "neuron" sounds biological and mysterious (like it's mimicking a brain cell), but in practice it's just a small calculator that does 3 simple steps:
> 1. **Take in some numbers** (inputs) — like "house size = 2,000."
> 2. **Multiply each number by an "importance" value** (weight), and add them all up. If you have several inputs (size, number of bedrooms, location score), each one gets its own importance multiplier, and then they're all summed into one single number.
> 3. **Pass that sum through a small math function** (activation) that decides how strongly this neuron should "respond." (More on why this step matters in Section 6.)
>
> Don't picture a glowing brain cell — picture a simple 3-step calculator: multiply, add, apply a small function. That's genuinely the entire "building block" of every AI system, including ChatGPT — just this simple operation repeated billions of times.

---

## 5. Anatomy of a Neuron

The full neuron formula:

```
output = activation(Σ wᵢxᵢ + b)
```

- **Weights (w):** Determine the strength and importance of each input signal.
- **Bias (b):** An offset that allows the neuron to shift its decision boundary.
- **Activation:** A non-linear function that decides if the signal should "fire."

> 💡 **In plain words:** Let's break the formula down piece by piece using our house price example, with just one input (size) for simplicity:
>
> - `x` = the input (e.g., house size = 2,000)
> - `w` = the weight (e.g., "$200 per sq ft" — how much this input matters)
> - `Σ wᵢxᵢ` = "multiply each input by its weight, and add them all up" (with one input, this is just `w × x`)
> - `+ b` = **bias** — a fixed number added on top. Think of it like a "starting price" even before considering size — e.g., if every house has a base value of $10,000 just for existing, regardless of size, that's the bias. It lets the neuron shift its answer up or down, not just scale with the input.
> - `activation(...)` = pass the result through a small shaping function (explained next in Section 6)
>
> So for our example: `output = activation(200 × 2000 + 10000)` = `activation(410,000)`. In a simple case, the activation function might just pass this number through unchanged — but for more complex problems, it does something more interesting, which is exactly why it's needed at all.

---

## 6. Why We Need Activation Functions

### The Linearity Problem

If you stack layers *without* any non-linear activation function, something breaks mathematically:

```
Layer 1: y = W₁x + b₁
Layer 2: z = W₂y + b₂

Combined:
z = W₂(W₁x + b₁) + b₂
z = (W₂W₁)x + (W₂b₁ + b₂)

This simplifies to:
z = W'x + b'   (where W' = W₂W₁ and b' = W₂b₁ + b₂)
```

**Mathematical Collapse:** Stacking linear layers just results in another linear function. 100 layers are equivalent to just one.

> 💡 **In plain words:** This is one of the most important (and easy to miss) ideas in the whole class. You might think "if one layer of neurons is good, 100 layers stacked on top of each other must be way more powerful, right?" But here's the surprising catch: if each neuron is *just* doing "multiply and add" (no activation function), then no matter how many layers you stack, the *entire stack* mathematically collapses down into a single, plain multiply-and-add operation. It's like nesting several "linear" transformations (like scaling and shifting) — no matter how many you chain together, you always end up with just another single scale-and-shift. You gain **zero extra power** by adding more layers.
>
> This matters because plain linear relationships (like "price = $200 × size") can only ever draw a straight line. But real-world problems are almost never that simple — most patterns are curved, twisted, or have complex boundaries. So without breaking this "linear collapse," a neural network — no matter how many layers deep — would be no smarter than a single line of basic algebra.

### The Non-linear Solution

- **Breaking Linearity:** Activation functions add "bends" to the math, preventing layers from collapsing into each other.
- **Enabling Depth:** This is the "secret sauce" that allows deep networks to learn complex, multi-layered representations.
- **Universal Approximation:** Non-linearity allows networks to model any continuous function, no matter how complex.

> 💡 **In plain words:** The fix is to insert a small non-linear "bend" (the activation function) after each layer's multiply-and-add step. Once you add these bends, stacking layers *actually does* add real power — each additional layer can now bend the data in a new way, letting the network represent increasingly complex, curvy, twisty relationships. This is why they're called "deep" neural networks — "deep" refers to having many of these stacked, bent layers.
>
> The "Universal Approximation" idea is a famous theoretical result: a large enough neural network with non-linear activations can, in principle, approximate *any* mathematical function you could throw at it — no matter how complicated the real-world pattern is. This is essentially the mathematical justification for why neural networks are so powerful and flexible.

---

## 7. The Activation Function Menu

| Function | Description |
|---|---|
| **Sigmoid** | Classic S-curve (0 to 1). Historically the default, ideal for **probability outputs**. |
| **Tanh** | Zero-centered (-1 to 1). Often provides **faster convergence** than sigmoid. |
| **ReLU** | The breakthrough: `max(0, x)`. Simple, efficient, and enabled deep networks. |
| **GELU & Swish** (Modern Variants) | Smooth versions of ReLU used in state-of-the-art LLMs. |

> 💡 **In plain words:** These are just different "bending" functions you can plug in — each with different pros and cons.
>
> - **Sigmoid**: squashes any number into a range between 0 and 1 — like a dial that always lands somewhere between "definitely no" (0) and "definitely yes" (1). Useful when you want the output to look like a probability (e.g., "90% chance this email is spam").
> - **Tanh**: similar S-curve, but squashes numbers between -1 and 1 instead. Being "zero-centered" (average output tends to hover around 0) tends to help the network train a bit faster/more smoothly than sigmoid.
> - **ReLU** (Rectified Linear Unit): the simplest of all — `max(0, x)` just means "if the number is negative, make it 0; if it's positive, leave it as is." It sounds almost too simple to be useful, but this simplicity makes it extremely fast to compute, and it turned out to work remarkably well — it's the main reason very deep networks became practical to train at all.
> - **GELU & Swish**: smoother, more refined versions of ReLU (instead of a sharp corner at 0, they curve more gently). These are used in cutting-edge large language models like GPT because they tend to perform slightly better in practice.
>
> You don't need to memorize the exact math of each — just remember: **ReLU is the modern default and the historically important breakthrough**, and Sigmoid/Tanh were the earlier, more "classic" choices.

---

## 8. The Loss Function

> Loss = A single number measuring how wrong we are.

- **Lower loss = better predictions**
- **Training goal = minimize loss**

### Mean Squared Error (MSE)

```
L = (1/n) Σ(prediction - target)²
```

- Squares make all errors positive
- Big errors penalized more than small errors
- Good for regression tasks

### Example
- Predicted: $350,000
- Actual: $400,000
- Error: $50,000
- Squared: 2,500,000,000

> 💡 **In plain words:** "Loss" is just a fancy word for "how wrong is the model's guess, expressed as a single number?" It's like a report card score, except lower is better — a loss of 0 would mean a perfect prediction.
>
> **Mean Squared Error** is one common way to calculate this score:
> 1. For each example, find the difference between what the model predicted and what the real answer was (the "error").
> 2. **Square** that error (multiply it by itself).
> 3. Average all those squared errors together.
>
> Why square the error instead of just using it directly? Two reasons, both shown in the slide:
> - **It removes negative signs.** An error of -$50,000 (predicted too high) and +$50,000 (predicted too low) are both "equally wrong" by $50,000 — but if you just added them up directly, they might cancel out and hide the fact that the model has errors at all. Squaring makes everything positive, so errors always add up honestly.
> - **It punishes big mistakes more.** A $50,000 error becomes a *huge* number once squared (2,500,000,000), while a small $1,000 error only becomes 1,000,000 — much smaller in comparison. This means the training process is pushed hard to fix big, glaring mistakes first, rather than treating a huge error and a tiny error as equally important.
>
> Once you have this single "loss" number, the model's entire goal becomes very simple: **make this number as small as possible.**

---

## 9. Backpropagation: Assigning Blame

### The Chain of Responsibility

- **The Question:** "The prediction was wrong. Which specific weights are responsible for this error?"
- **The Flow:** Error signals travel backward from the output layer through the hidden layers to the input.
- **Adjustment:** Each weight is adjusted proportionally to its contribution to the final mistake.

> 💡 **In plain words:** Imagine a company made a bad product, and now you need to figure out exactly which department (design, manufacturing, marketing) is most responsible for the failure, so you know who needs to change their approach. Backpropagation does the same thing for a neural network: once we know the network's final answer was wrong (from Section 8's loss number), we need to figure out exactly *which* weights, out of potentially millions of them, contributed most to that mistake.
>
> The name "backpropagation" describes exactly what happens: the error signal is calculated at the *output* (where the final prediction was made), and then it flows **backward** through the network, layer by layer, back toward the input. As it flows backward, each individual weight gets "blamed" a proportional amount, based on how much it actually contributed to the wrong answer. Weights that barely mattered get barely nudged; weights that were majorly responsible for the mistake get adjusted more aggressively.
>
> This is genuinely one of the most important ideas in all of deep learning — it's the mathematical technique that makes it possible to train networks with millions or billions of weights, by efficiently figuring out how to adjust every single one of them based on just one loss number.

---

## 10. Gradient Descent: Finding the Valley

### The Analogy

> Imagine you are blindfolded on a hilly landscape. Your goal is to reach the lowest valley.

- **The Landscape:** The "Loss Surface" represents all possible errors the model can make.
- **The Gradient:** The slope under your feet. It tells you which direction is "down" for the loss.

### The Strategy

- **Step Downhill:** Feel the slope and take a step in the direction that reduces the loss.
- **Iterative Progress:** Repeat the process, taking step after step until you reach the bottom.
- **The Goal:** Reach the global minimum — the point where the model's error is as low as possible.

> 💡 **In plain words:** This is the *how* behind Backpropagation's *blame-assignment* — once we know how much each weight contributed to the error (Section 9), Gradient Descent is the actual strategy used to decide **how to change each weight** to reduce that error.
>
> The blindfolded-hiker analogy is genuinely the best way to picture this:
> - Imagine every possible combination of weights the model could have produces a different "loss" value — plot all of these as a bumpy, hilly landscape, where height = how wrong the model is. High peaks = terrible predictions; low valleys = great predictions.
> - You (the model, mid-training) are standing somewhere on this landscape, blindfolded — you can't see the whole landscape, but you *can* feel the slope right under your feet.
> - The **gradient** is exactly that: it tells you, right where you're standing, which direction is steepest downhill.
> - So the strategy is: feel the slope, take a small step in the downhill direction, then repeat — feel, step, feel, step — gradually making your way down toward the lowest point (the valley), where the loss is smallest.
>
> This is why the process is called "gradient **descent**" — you're literally descending down the loss landscape, one small step at a time, using the gradient to tell you which way is down.

---

## 11. Learning Rate

The learning rate controls **how big each downhill step is**.

| Too Low | Just Right | Too High |
|---|---|---|
| A small learning rate requires many updates before reaching the minimum point | The optimal learning rate swiftly reaches the minimum point | Too large a learning rate causes drastic updates which lead to divergent behaviors |

> 💡 **In plain words:** Going back to the blindfolded-hiker analogy from Section 10 — the learning rate is simply **how big of a step you take each time** you move downhill.
>
> - **Too low:** You take tiny, cautious baby steps. You'll eventually reach the bottom, but it might take an enormous number of steps (a very long time) to get there.
> - **Just right:** Your steps are appropriately sized — you make solid progress downhill each time without overshooting, and reach the bottom efficiently.
> - **Too high:** You take huge, reckless leaps. You might leap right *over* the valley and land on the opposite hillside, then leap back over again the other way — bouncing back and forth erratically without ever actually settling into the valley. In the worst cases, this can make the model's error get *worse* over time instead of better ("divergent behavior").
>
> Choosing a good learning rate is one of the most important — and trickiest — practical decisions when training a neural network. Too small wastes time; too large can make training fail completely.

---

## 12. The Training Loop: Putting It All Together

The full cycle, repeated over and over:

1. **Forward Pass** — Data flows through the network to generate a prediction.
2. **Calculate Loss** — Measure the error between the prediction and the ground truth.
3. **Backpropagate** — Trace the error backward to assign "blame" to each weight.
4. **Update Weights** — Adjust weights using gradient descent to minimize future error.

> 💡 **In plain words:** This is the "zoomed out" summary that ties every single concept from this class together into one repeating cycle:
>
> 1. **Forward Pass** (Sections 4–7): Feed an example (like a house's size) through the neurons — multiply by weights, add bias, apply activation — to get a prediction (e.g., predicted price).
> 2. **Calculate Loss** (Section 8): Compare that prediction to the real, correct answer, and compute a single "how wrong were we" number using something like MSE.
> 3. **Backpropagate** (Section 9): Trace that error number backward through the network to figure out exactly how much each individual weight contributed to the mistake.
> 4. **Update Weights** (Sections 10–11): Use gradient descent (with an appropriate learning rate) to nudge every weight slightly in the direction that reduces future error.
>
> Then repeat this entire 4-step cycle — often millions of times, across millions of examples — and the network gradually gets better and better at predicting the correct answer. This 4-step loop is literally the training process for every neural network in existence, from the simplest toy example to the largest AI models in the world (see next section).

---

## 13. Scaling to the Moon

The exact same training loop (Forward Pass → Loss Calculation → Backpropagation → Weight Updates) scales from tiny to enormous:

| Model | Parameters |
|---|---|
| XOR Network (a simple toy example) | ~20 Parameters |
| GPT-4 | ~1 Trillion |

> 💡 **In plain words:** This is the "mind-blown" moment of the class: the exact same 4-step loop from Section 12 — forward pass, loss, backpropagation, weight update — is *literally* what trains a tiny toy network with only about 20 adjustable weights, and it's *also* literally what trains something as massive and capable as GPT-4, which has roughly **1 trillion** weights (parameters).
>
> Nothing fundamentally different happens at huge scale — it's not some entirely new algorithm. It's the same simple "guess → measure error → assign blame → correct" loop from Section 3, just repeated an almost unimaginable number of times, across an enormous number of weights, using massive amounts of computing power (echoing the "Compute" ingredient from Week 1's "Why AI Exploded" section). This is a powerful idea: you don't need to understand some exotic new technique to understand how GPT-4 trains — you already do, from this one class.

---

## 💡 Additional Notes

> 💡 Additional Note: The slide deck showed a diagram of "Forward Propagation" and "Backward Propagation" happening in a loop, with components labeled Weights, Optimizer, Loss Score, and Loss Function. The "Optimizer" is the piece of software that actually implements gradient descent (or a more advanced variant of it) to update the weights — the slides didn't name specific optimizer algorithms (like Adam or SGD), so this class stuck to the general concept of gradient descent.

> 💡 Additional Note: "Convergence" (mentioned under Tanh) refers to the training process successfully reaching a low-loss point — "faster convergence" means the network reaches a good, low-error state in fewer training steps.

> 💡 Additional Note: XOR is a classic, very small logic problem (see "XOR Network ~20 Parameters") often used as the simplest possible example to demonstrate why non-linear activation functions matter — a plain linear model literally cannot solve XOR, which is a hands-on version of the "Linearity Problem" from Section 6. The slides referenced it by name without walking through the full problem, so if you want a deeper dive on *why* XOR specifically needs non-linearity, that could be a good follow-up question for the instructor.

> 💡 Additional Note: A useful mental model for the whole class: **Neuron → Layer → Network → Loss → Backprop → Gradient Descent → repeat.** Each concept exists to answer one specific question in the training loop: "how does a neuron compute?" (Sections 4–5) → "why do we need bends in the math?" (Sections 6–7) → "how wrong are we?" (Section 8) → "who's to blame?" (Section 9) → "how do we fix it?" (Sections 10–11) → "put it all together and repeat" (Sections 12–13).

---

## 🔑 Key Takeaways

*(These four points were given directly in the source slides as the official "Key Takeaways" — included here exactly as taught, with the rest of this document's sections providing the supporting detail.)*

1. **Neural networks are layers of simple neurons doing weighted sums.**
2. **Training is the iterative process of adjusting weights to minimize loss.**
3. **Backpropagation traces error backward to assign "blame" to weights.**
4. **Gradient descent is the strategy for stepping toward lower loss.**

---

## 🧠 Quick Revision

- **Neuron formula:** `output = activation(Σ wᵢxᵢ + b)` — inputs × weights, summed, plus bias, passed through activation
- **Weights** = importance of each input; **Bias** = shifts the output up/down independent of inputs
- **Linearity Problem**: stacking linear layers with no activation = mathematically collapses into ONE linear layer (100 layers = 1 layer)
- **Activation functions** break linearity, enabling deep networks to model complex, non-linear patterns
- **Sigmoid** (0 to 1, probabilities) | **Tanh** (-1 to 1, zero-centered) | **ReLU** (`max(0,x)`, the modern default) | **GELU/Swish** (smooth ReLU variants, used in LLMs)
- **Loss** = a single number measuring how wrong the model is; **lower is better**
- **MSE (Mean Squared Error)**: `(1/n) Σ(prediction - target)²` — squares errors to make them positive and punish big mistakes more
- **Backpropagation** = tracing the error backward through the network to figure out which weights caused it
- **Gradient Descent** = the "blindfolded hiker" strategy: feel the slope (gradient), step downhill, repeat, to minimize loss
- **Learning Rate** = size of each step; too low = slow, too high = overshoots/diverges
- **Training Loop (4 steps)**: Forward Pass → Calculate Loss → Backpropagate → Update Weights
- **Scale**: same 4-step loop works whether a network has ~20 parameters (XOR) or ~1 trillion (GPT-4)

---

## ❓ Self-Check Questions

1. What are the three main components of a single neuron, and what does each one do?
2. Write out the neuron formula and explain what each symbol (`w`, `x`, `b`) represents.
3. Why does stacking multiple *linear* layers (with no activation function) fail to add any extra modeling power?
4. Name three activation functions and briefly describe what each one does.
5. What is a "loss function," and why do we square the errors in Mean Squared Error?
6. In your own words, what problem does backpropagation solve?
7. Explain the "blindfolded hiker" analogy for gradient descent — what do the "landscape" and "slope" represent?
8. What happens if the learning rate is set too high? Too low?
9. List the 4 steps of the training loop in order.
10. Why does the instructor say the same training process used for a ~20-parameter XOR network also trains a ~1-trillion-parameter model like GPT-4?