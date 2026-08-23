# Unit 4: Inferential Statistics & Hypothesis Testing

> 📁 **Repository:** `ai-ml-learning/04-inferential-statistics/README.md`
> 🎯 **Phase:** 1 — Foundations
> 🧩 **Unit:** 4 of 36

---

## 📌 Overview

Descriptive statistics (Unit 3) tell you about the data you have. **Inferential statistics** let you go further — drawing conclusions about the population you cannot fully observe, based only on a sample.

This is the difference between:

```
Descriptive:   "In our sample of 500 users, 62% preferred Feature A"
Inferential:   "We are 95% confident that 62% ± 4% of ALL users prefer Feature A"

Descriptive:   "Model A scored 88% and Model B scored 85% on our test set"
Inferential:   "Is this difference statistically significant, or could it be
                due to random chance in how we split the test set?"
```

Inferential statistics are the tools every data scientist uses daily — A/B testing, model comparison, feature significance, and experimental design all rely on this framework.

```
INFERENTIAL STATISTICS
│
├── ESTIMATION          →  How do we estimate population parameters?
│     Point estimation
│     Confidence intervals
│
└── HYPOTHESIS TESTING  →  How do we make decisions from data?
      Formulating hypotheses
      Computing test statistics
      Making decisions with controlled error rates
```

---

## 🎯 Learning Objectives

By the end of this unit, you should be able to:

- [ ] Explain the distinction between descriptive and inferential statistics
- [ ] Describe sampling methods and why they matter
- [ ] Compute and interpret point estimates and confidence intervals
- [ ] Formulate null and alternative hypotheses for a given problem
- [ ] Explain significance level, p-value, and critical values
- [ ] Distinguish between Type I and Type II errors, and statistical power
- [ ] Choose and conduct the appropriate statistical test for a given problem
- [ ] Interpret results of t-tests, z-tests, chi-square tests, and ANOVA
- [ ] Connect hypothesis testing to real ML workflows (A/B testing, model comparison)

---

## 🗺️ Prerequisites

- Unit 3: Statistics & Probability (mean, variance, standard deviation, normal distribution, CLT, sampling distributions)
- Unit 2: Mathematics for AI/ML (basic algebra)

---

## 📚 Topics Covered

**Part A — Inferential Statistics**
1. [Population vs Sample — Inferential Perspective](#1-population-vs-sample--inferential-perspective)
2. [Sampling and Sampling Methods](#2-sampling-and-sampling-methods)
3. [Sampling Distributions](#3-sampling-distributions)
4. [Point Estimation](#4-point-estimation)
5. [Confidence Intervals](#5-confidence-intervals)
6. [Margin of Error](#6-margin-of-error)

**Part B — Hypothesis Testing**

7. [The Logic of Hypothesis Testing](#7-the-logic-of-hypothesis-testing)
8. [Null and Alternative Hypothesis](#8-null-and-alternative-hypothesis)
9. [Significance Level (α)](#9-significance-level-α)
10. [Test Statistic](#10-test-statistic)
11. [Critical Value](#11-critical-value)
12. [p-value](#12-p-value)
13. [Type I and Type II Errors](#13-type-i-and-type-ii-errors)
14. [Statistical Power](#14-statistical-power)

**Part C — Statistical Tests**

15. [One-Sample Mean Test](#15-one-sample-mean-test)
16. [Two-Sample Mean Test](#16-two-sample-mean-test)
17. [Proportion Test](#17-proportion-test)
18. [t-test](#18-t-test)
19. [z-test](#19-z-test)
20. [Chi-Square Test](#20-chi-square-test)
21. [ANOVA](#21-anova)
22. [Choosing the Right Test](#22-choosing-the-right-test)

---

# PART A — INFERENTIAL STATISTICS

---

## 1. Population vs Sample — Inferential Perspective

### Revisiting the Core Problem

In Unit 3, we defined population and sample. Now we go deeper into **why** this distinction drives all of inferential statistics.

```
THE FUNDAMENTAL PROBLEM:

We want to know something about the POPULATION:
  "What percentage of all Indian adults have hypertension?"
  "What is the true conversion rate of our website?"
  "Does this drug reduce blood pressure in ALL patients?"

But we can only observe a SAMPLE:
  Survey 2,000 adults (not 1.4 billion)
  Track 10,000 visitors (not all visitors ever)
  Run trial on 300 patients (not all patients)

INFERENTIAL STATISTICS bridges this gap:
  Use what we observe in the SAMPLE
  to make statements about the POPULATION
  with quantified uncertainty
```

### The Key Challenge: Sampling Variability

Every sample gives slightly different results. If you took 1,000 different samples of size 200 from the same population, you would get 1,000 different sample means. This variability is fundamental:

```
True population mean: μ = 65 kg (unknown to us)

Sample 1 (n=200): x̄ = 64.3 kg
Sample 2 (n=200): x̄ = 65.8 kg
Sample 3 (n=200): x̄ = 64.9 kg
...
Sample 1000 (n=200): x̄ = 65.2 kg

Each sample gives a different estimate.
How do we account for this variability?
→ Confidence intervals and hypothesis tests
```

### Population Parameters vs Sample Statistics

| Measure | Population Parameter | Sample Statistic | Relationship |
|---|---|---|---|
| Mean | μ (mu) | x̄ (x-bar) | x̄ estimates μ |
| Standard deviation | σ (sigma) | s | s estimates σ |
| Proportion | p | p̂ (p-hat) | p̂ estimates p |
| Variance | σ² | s² | s² estimates σ² |
| Correlation | ρ (rho) | r | r estimates ρ |

The goal of inferential statistics is to use sample statistics to make **probabilistic statements** about population parameters.

---

## 2. Sampling and Sampling Methods

### Why Sampling Methods Matter

A biased sample leads to biased conclusions — no matter how sophisticated your statistical analysis. The quality of inference is only as good as the quality of the sample.

```
Famous example of bad sampling:
  1936 US Presidential Election Poll (Literary Digest)
  Sample: 10 million people (huge sample!)
  Prediction: Alf Landon wins by a landslide
  Reality: Franklin Roosevelt won by a landslide

  Why wrong? Sample was biased toward wealthy people
  (drawn from telephone directories and car registrations
  in 1936 — poor people had neither)

  Lesson: A large biased sample is worse than a small representative one.
```

### Sampling Methods

**1. Simple Random Sampling**

Every member of the population has an equal probability of being selected.

```
Method: Assign every member a number. Use a random number generator.

Population: 10,000 employees
Sample: Select 200 using random number table

Pros: Unbiased, simple
Cons: Requires complete list of population; may miss small groups
```

**2. Stratified Sampling**

Divide population into subgroups (strata) and sample from each proportionally.

```
Population: 10,000 employees (60% male, 40% female)
Stratified sample of 200:
  → Sample 120 males (60% of 200)
  → Sample 80 females (40% of 200)

Pros: Guarantees representation of all subgroups
Cons: Requires knowledge of strata; more complex
Use in ML: Class imbalance — stratified train-test split
```

**3. Systematic Sampling**

Select every k-th element from a list after a random start.

```
Population: 10,000 records, want 200 samples
k = 10,000/200 = 50
Random start: say record #17
→ Select records: 17, 67, 117, 167, ...

Pros: Simple to implement
Cons: Problematic if population has periodic patterns
```

**4. Cluster Sampling**

Divide population into clusters (often geographic), randomly select clusters, survey all within selected clusters.

```
Survey schools in India:
  → List all 1.5 million schools
  → Randomly select 100 schools
  → Survey all students in those 100 schools

Pros: Cost-effective for geographically spread populations
Cons: Less precise than simple random; clusters may not be representative
```

**5. Convenience Sampling**

Select whoever is easiest to access. **Not recommended for inference.**

```
Example: Surveying only your Twitter followers about social media habits
         → Clearly biased toward active social media users

In ML: Using only available data (e.g., historical customer records)
       without thinking about who is missing from the data
```

### In ML Context

```
ML SAMPLING APPLICATIONS:
  Train/Test Split:       Random sampling (usually stratified for classification)
  Cross-validation:       Systematic partitioning into k folds
  Bootstrapping:          Sample with replacement (for confidence intervals)
  Mini-batch GD:          Random sampling from training data each step
  Data augmentation:      Synthetic sampling to balance classes
```

### Python: Sampling

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

np.random.seed(42)

# Simulate a dataset
n = 10000
df = pd.DataFrame({
    'age':    np.random.randint(18, 65, n),
    'income': np.random.lognormal(10.5, 0.7, n),
    'gender': np.random.choice(['M', 'F'], n, p=[0.6, 0.4]),
    'bought': np.random.choice([0, 1], n, p=[0.85, 0.15])  # 15% conversion
})

# ── Simple Random Sample ──────────────────────────────────────────
simple_sample = df.sample(n=200, random_state=42)
print(f"Simple random sample: n={len(simple_sample)}")
print(f"  Gender split: {simple_sample['gender'].value_counts().to_dict()}")

# ── Stratified Sample ─────────────────────────────────────────────
# Ensure equal proportion of genders in sample
strat_sample = df.groupby('gender', group_keys=False).apply(
    lambda x: x.sample(frac=0.02, random_state=42)
)
print(f"\nStratified sample: n={len(strat_sample)}")
print(f"  Gender split: {strat_sample['gender'].value_counts().to_dict()}")

# ── Train-Test Split (stratified by class) ────────────────────────
X = df[['age', 'income']].values
y = df['bought'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"\nStratified Train-Test Split:")
print(f"  Train conversion rate: {y_train.mean():.3f}")  # Should be ~0.15
print(f"  Test  conversion rate: {y_test.mean():.3f}")   # Should be ~0.15

# ── Bootstrap Sampling ────────────────────────────────────────────
# Sample with replacement — used to estimate uncertainty
original_mean = df['income'].mean()
bootstrap_means = [
    df['income'].sample(n=200, replace=True).mean()
    for _ in range(1000)
]
bootstrap_ci = np.percentile(bootstrap_means, [2.5, 97.5])
print(f"\nBootstrap 95% CI for mean income:")
print(f"  [{bootstrap_ci[0]:.2f}, {bootstrap_ci[1]:.2f}]")
print(f"  True mean: {original_mean:.2f}")
```

---

## 3. Sampling Distributions

### Recap from Unit 3 (extended)

The **sampling distribution** of a statistic is the probability distribution of that statistic across all possible samples of size n from the population.

From Unit 3, we established the CLT result:

```
If X₁, X₂, ..., Xₙ are i.i.d. with mean μ and variance σ²:

x̄ ~ N(μ,  σ²/n)   approximately, for large n

Standard Error (SE) = σ / √n
```

### Sampling Distribution of a Proportion

For proportions (e.g., fraction of users who click):

```
If true proportion = p, sample size = n:

p̂ ~ N(p, p(1-p)/n)    approximately, for large n

SE(p̂) = √(p(1-p)/n)

Rule of thumb for "large enough n":
  np ≥ 10  AND  n(1-p) ≥ 10
```

### Why Sampling Distributions Are the Engine of Inference

```
The sampling distribution tells us:
  "If the population parameter were θ, what sample statistics
   would we typically observe?"

We flip this around for inference:
  "Given the sample statistic we observed, what can we say
   about the population parameter θ?"

Confidence intervals and hypothesis tests are both built
on this exact logical foundation.
```

---

## 4. Point Estimation

### What is it?

A **point estimate** is a single value calculated from a sample that is used as the best guess of the corresponding population parameter.

```
Population Parameter     Point Estimator     Example
─────────────────────    ────────────────    ──────────────────────
Mean (μ)                 x̄ (sample mean)     x̄ = 67.3 kg
Variance (σ²)            s² (sample var)     s² = 45.2
Proportion (p)           p̂ (sample prop)     p̂ = 0.347
Correlation (ρ)          r (Pearson r)       r = 0.72
```

### Properties of Good Estimators

**1. Unbiasedness**

An estimator is **unbiased** if its expected value equals the true parameter.

```
E[x̄] = μ       ✓  Sample mean is unbiased for population mean
E[s²] = σ²     ✓  Sample variance with (n-1) is unbiased
E[s²] ≠ σ²     ✗  Sample variance with (n) is biased (underestimates)
```

**2. Consistency**

As sample size increases, the estimator converges to the true parameter.

```
lim(n→∞) x̄ = μ    ✓  Sample mean is consistent
```

**3. Efficiency**

Among all unbiased estimators, an efficient estimator has the smallest variance (most precise).

### The Problem with Point Estimates

A point estimate is a single number with **no indication of uncertainty**:

```
"The average height of Indian men is 165.3 cm"

This sounds precise, but:
  - It's based on a sample of 2,000 people
  - A different sample might give 164.8 or 165.7
  - We have no idea how close 165.3 is to the true value

Solution: Use a CONFIDENCE INTERVAL instead
```

---

## 5. Confidence Intervals

### What is it?

A **confidence interval (CI)** is a range of values, computed from the sample, that likely contains the true population parameter with a specified level of confidence.

```
Format:
  Point Estimate ± Margin of Error
  [Lower Bound,  Upper Bound]

Example:
  "The average height is 165.3 ± 1.2 cm"
  "We are 95% confident the true mean height is between 164.1 and 166.5 cm"
```

### The Formula

For a population mean when σ is known:

```
CI = x̄ ± z* × (σ/√n)

Components:
  x̄         = sample mean (point estimate)
  z*         = critical z-value for confidence level
  σ/√n       = standard error (SE)
  z* × SE    = margin of error

Common z* values:
  90% CI: z* = 1.645
  95% CI: z* = 1.960
  99% CI: z* = 2.576
```

### Worked Example

```
Problem:
  You sample 100 customer wait times.
  Sample mean: x̄ = 4.2 minutes
  Population σ = 1.5 minutes (known from historical data)
  Construct a 95% confidence interval.

Solution:
  SE = σ/√n = 1.5/√100 = 1.5/10 = 0.15

  95% CI: x̄ ± 1.96 × SE
          = 4.2 ± 1.96 × 0.15
          = 4.2 ± 0.294
          = (3.906, 4.494)

Interpretation:
  "We are 95% confident the true population mean wait time
   lies between 3.91 and 4.49 minutes."
```

### What "95% Confident" Actually Means

This is the most commonly misunderstood concept in statistics.

```
INCORRECT interpretation:
  "There is a 95% probability the true mean is inside this specific interval"

CORRECT interpretation:
  "If we repeated this sampling procedure 100 times and computed 100 CIs,
   approximately 95 of them would contain the true population mean"

The true mean is FIXED (not random).
The interval is what varies from sample to sample.
```

```
Visualisation:
True mean μ ─────────────────────────────────────────────

Sample 1:     ───────[     CI 1     ]──────────    ✓ contains μ
Sample 2:          ──────[  CI 2  ]────────────    ✓ contains μ
Sample 3:    ─────────[    CI 3    ]───────────    ✓ contains μ
Sample 4:              ─────────────[  CI 4  ]─    ✗ misses μ  ← 5% of CIs
Sample 5:          ───────────[   CI 5   ]─────    ✓ contains μ
...
```

### Effect of Sample Size and Confidence Level

```
Larger sample → narrower CI (more precise)
  n=25:  CI width ∝ 1/√25 = 0.2
  n=100: CI width ∝ 1/√100 = 0.1  ← half as wide
  n=400: CI width ∝ 1/√400 = 0.05 ← quarter as wide

Higher confidence → wider CI (less precise)
  90% CI: z*=1.645, narrower
  95% CI: z*=1.960
  99% CI: z*=2.576, wider ← to be more certain, must cast wider net

TRADE-OFF: Precision vs Confidence
  You can be very precise OR very confident, but not both
  (without collecting more data)
```

### Python: Confidence Intervals

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

np.random.seed(42)

# ── CI when σ is known (z-interval) ──────────────────────────────
sample = np.array([4.1, 3.8, 4.5, 4.2, 3.9, 4.7, 4.0, 4.3, 4.1, 4.6,
                   3.7, 4.4, 4.2, 3.8, 4.5, 4.1, 4.3, 4.0, 4.2, 4.4])
n      = len(sample)
x_bar  = np.mean(sample)
sigma  = 1.5          # Known population std
SE     = sigma / np.sqrt(n)

confidence_levels = [0.90, 0.95, 0.99]
z_stars = [stats.norm.ppf((1 + cl) / 2) for cl in confidence_levels]

print("=== Confidence Intervals (σ known) ===")
print(f"Sample mean: {x_bar:.4f}, n={n}, σ={sigma}, SE={SE:.4f}\n")
for cl, z_star in zip(confidence_levels, z_stars):
    lower = x_bar - z_star * SE
    upper = x_bar + z_star * SE
    print(f"{cl:.0%} CI: [{lower:.4f}, {upper:.4f}]  (z*={z_star:.3f})")

# ── CI when σ is unknown (t-interval) ────────────────────────────
s = np.std(sample, ddof=1)     # Sample std instead
SE_t = s / np.sqrt(n)

print(f"\n=== Confidence Intervals (σ unknown, use t) ===")
print(f"Sample mean: {x_bar:.4f}, n={n}, s={s:.4f}, SE={SE_t:.4f}\n")
for cl in confidence_levels:
    t_star = stats.t.ppf((1 + cl) / 2, df=n-1)
    lower  = x_bar - t_star * SE_t
    upper  = x_bar + t_star * SE_t
    print(f"{cl:.0%} CI: [{lower:.4f}, {upper:.4f}]  (t*={t_star:.3f}, df={n-1})")

# ── Direct scipy CI ───────────────────────────────────────────────
ci_95 = stats.t.interval(0.95, df=n-1, loc=x_bar, scale=SE_t)
print(f"\nDirect scipy 95% CI: {np.round(ci_95, 4)}")

# ── Visualise: 100 CIs, 95% should contain true mean ─────────────
true_mu = 4.2    # Known for simulation
n_sims  = 100
n_each  = 30

fig, ax = plt.subplots(figsize=(10, 8))
contains = 0

for i in range(n_sims):
    samp = stats.norm.rvs(loc=true_mu, scale=1.5, size=n_each)
    x_b  = samp.mean()
    se   = samp.std(ddof=1) / np.sqrt(n_each)
    t_c  = stats.t.ppf(0.975, df=n_each-1)
    lo, hi = x_b - t_c*se, x_b + t_c*se
    inside = lo <= true_mu <= hi
    if inside: contains += 1
    colour = 'steelblue' if inside else 'red'
    ax.plot([lo, hi], [i, i], color=colour, linewidth=1.5, alpha=0.7)

ax.axvline(true_mu, color='black', linewidth=2, label=f'True μ = {true_mu}')
ax.set_xlabel("Value"); ax.set_ylabel("Sample number")
ax.set_title(f"95% CIs from 100 samples — {contains}/100 contain μ")
ax.legend(); ax.grid(True, alpha=0.2, axis='x')
plt.tight_layout()
plt.savefig('confidence_intervals.png', dpi=100, bbox_inches='tight')
plt.show()
print(f"\nCIs containing true mean: {contains}/100 = {contains}%  (expected: ~95)")
```

---

## 6. Margin of Error

### What is it?

The **margin of error (MoE)** is the half-width of a confidence interval — the amount added and subtracted from the point estimate to form the CI.

```
CI = x̄ ± MoE

MoE = z* × (σ/√n)
    = z* × SE

For 95% CI:
  MoE = 1.96 × σ/√n
```

### Determining Required Sample Size

Given a desired margin of error, you can calculate the required sample size:

```
MoE = z* × σ/√n

Solving for n:
  n = (z* × σ / MoE)²

Example:
  You want a 95% CI with MoE = ±2 kg for a weight study.
  Historical σ = 15 kg.

  n = (1.96 × 15 / 2)² = (14.7)² ≈ 216

  You need at least 216 people in your sample.
```

This formula is used in:
- Survey design (how many people to survey)
- A/B test sizing (how many visitors needed)
- Clinical trial planning (how many patients needed)

### Python: Sample Size Calculation

```python
import numpy as np
from scipy import stats

def required_sample_size_mean(moe, sigma, confidence=0.95):
    """
    Calculate required sample size for estimating a mean.
    moe:        desired margin of error
    sigma:      population/estimated standard deviation
    confidence: confidence level (e.g., 0.95)
    """
    z_star = stats.norm.ppf((1 + confidence) / 2)
    n = (z_star * sigma / moe) ** 2
    return int(np.ceil(n))   # Round up — always need at least this many

def required_sample_size_proportion(moe, p=0.5, confidence=0.95):
    """
    Sample size for estimating a proportion.
    Use p=0.5 for maximum (most conservative) sample size.
    """
    z_star = stats.norm.ppf((1 + confidence) / 2)
    n = (z_star ** 2 * p * (1-p)) / moe**2
    return int(np.ceil(n))

# Example 1: Estimating mean
print("=== Sample Size for Mean ===")
for moe in [5, 2, 1]:
    n = required_sample_size_mean(moe=moe, sigma=15, confidence=0.95)
    print(f"  MoE = ±{moe}: need n = {n}")

# Example 2: Estimating proportion (e.g., click-through rate)
print("\n=== Sample Size for Proportion ===")
for moe in [0.05, 0.03, 0.01]:
    n_conservative = required_sample_size_proportion(moe=moe, p=0.5)
    n_specific     = required_sample_size_proportion(moe=moe, p=0.12)   # p≈0.12
    print(f"  MoE = ±{moe:.0%}: n={n_conservative} (conservative), "
          f"n={n_specific} (if p≈12%)")
```

---

# PART B — HYPOTHESIS TESTING

---

## 7. The Logic of Hypothesis Testing

### What is it?

**Hypothesis testing** is a formal framework for making decisions from data — specifically, deciding whether the evidence from a sample is strong enough to reject a claim about the population.

### The Court of Law Analogy

Hypothesis testing follows the same logic as a criminal trial:

```
IN A TRIAL:                           IN HYPOTHESIS TESTING:
─────────────────────────────────     ──────────────────────────────────────
Defendant is innocent                 Null hypothesis (H₀) is true
until proven guilty                   until data proves otherwise

Prosecution must prove                Data must be strong enough
guilt beyond reasonable doubt         to reject H₀ at significance level α

"Beyond reasonable doubt"             p-value < α
→ Jury convicts (rejects innocence)   → Reject H₀

Insufficient evidence                 p-value ≥ α
→ Not guilty (fail to reject)         → Fail to reject H₀

IMPORTANT: "Not guilty" ≠ "Innocent"  "Fail to reject H₀" ≠ "H₀ is true"
```

### The 5-Step Process

```
STEP 1: State the hypotheses (H₀ and H₁)
STEP 2: Choose significance level α
STEP 3: Compute the test statistic from sample data
STEP 4: Find the p-value or critical value
STEP 5: Make a decision and interpret in context
```

---

## 8. Null and Alternative Hypothesis

### The Null Hypothesis (H₀)

The **null hypothesis** is the default claim — the status quo — the thing you are trying to disprove.

```
Characteristics:
  - Always contains "=" (equals, or no difference, or no effect)
  - Represents no change, no effect, no difference
  - The "boring" or "conservative" statement

Examples:
  H₀: μ = 165       (population mean height is 165 cm)
  H₀: p = 0.10      (website conversion rate is 10%)
  H₀: μ₁ = μ₂       (drug A and drug B have equal effect)
  H₀: The two variables are independent
```

### The Alternative Hypothesis (H₁ or Hₐ)

The **alternative hypothesis** is what you are trying to prove — the claim that there IS an effect.

```
Characteristics:
  - Contains "≠", "<", or ">" (not equal, less than, greater than)
  - Represents the researcher's hypothesis
  - What you claim is actually happening

Examples:
  H₁: μ ≠ 165       (mean height is NOT 165 cm)  → two-tailed
  H₁: p > 0.10      (conversion rate is HIGHER than 10%)  → one-tailed (right)
  H₁: μ₁ < μ₂       (drug A is less effective than drug B)  → one-tailed (left)
```

### One-Tailed vs Two-Tailed Tests

```
TWO-TAILED (H₁: μ ≠ μ₀):
  "The mean is different from μ₀ in EITHER direction"
  Reject H₀ if test statistic is too far above OR below μ₀
  Use when: You care about differences in either direction

  Rejection region:       ██████          ██████
  ─────────────────────── └─α/2 ──────── α/2─┘ ──────────────

ONE-TAILED RIGHT (H₁: μ > μ₀):
  "The mean is GREATER THAN μ₀"
  Reject H₀ if test statistic is far ABOVE μ₀
  Use when: You only care about increases

  Rejection region:                            ██████████
  ────────────────────────────────────────── └───α──┘

ONE-TAILED LEFT (H₁: μ < μ₀):
  "The mean is LESS THAN μ₀"
  Rejection region: ██████████
  ─── └───α──┘ ─────────────────────────────────────────
```

### Real ML Example

```
Scenario: You have a new ML model (Model B) claiming better accuracy than
          the baseline (Model A) which has 85% accuracy.

H₀: μ_B = 0.85    (Model B is no better than baseline)
H₁: μ_B > 0.85    (Model B is genuinely better)

This is a one-tailed test (right) because we only care if B is BETTER.
We do not care if it is worse — we would keep Model A in that case.
```

---

## 9. Significance Level (α)

### What is it?

The **significance level** (α, alpha) is the **probability of rejecting the null hypothesis when it is actually true** — the threshold for "strong enough evidence."

```
α = 0.05  (most common)  → "We accept a 5% chance of wrongly rejecting H₀"
α = 0.01  (stricter)     → "We accept a 1% chance of wrongly rejecting H₀"
α = 0.10  (less strict)  → "We accept a 10% chance of wrongly rejecting H₀"
```

### Choosing α

```
Higher stakes → Lower α:
  Medical decisions: α = 0.01 or 0.001
    (Don't want to falsely approve a harmful drug)
  Engineering safety: α = 0.001 or even lower

Standard research: α = 0.05
  (The Fisher convention from 1925 — historical standard)

Exploratory analysis: α = 0.10
  (More willing to follow up on tentative leads)
```

### The Relationship to Confidence Level

```
α + Confidence Level = 1

95% CI corresponds to α = 0.05
99% CI corresponds to α = 0.01
90% CI corresponds to α = 0.10
```

---

## 10. Test Statistic

### What is it?

A **test statistic** is a single number computed from the sample data that measures how far the sample result is from what we would expect under H₀, in units of standard error.

```
General form:
                    Observed value - Hypothesised value
  Test statistic = ─────────────────────────────────────
                         Standard Error

The larger the test statistic, the more incompatible the data is with H₀.
```

### Common Test Statistics

| Test | Test Statistic | Formula |
|---|---|---|
| **z-test** | z | (x̄ − μ₀) / (σ/√n) |
| **t-test** | t | (x̄ − μ₀) / (s/√n) |
| **Proportion z-test** | z | (p̂ − p₀) / √(p₀(1−p₀)/n) |
| **Chi-square** | χ² | Σ (O−E)²/E |
| **F-test (ANOVA)** | F | Variance between groups / Variance within groups |

### Intuition

```
If H₀ is TRUE:
  The observed x̄ should be close to μ₀
  → Test statistic ≈ 0 (small)
  → Not unusual, no reason to reject H₀

If H₀ is FALSE:
  The observed x̄ will be far from μ₀
  → Test statistic is large (positive or negative)
  → The data is incompatible with H₀ → reject H₀
```

---

## 11. Critical Value

### What is it?

The **critical value** is the boundary of the **rejection region** — the threshold that the test statistic must exceed (in absolute value or direction) for us to reject H₀.

```
For a two-tailed test at α = 0.05:
  Critical values: z* = ±1.96

  Rejection region:        │                   │
  ───────────────────  ────┼─────────────────  ┼────
  Reject H₀          -1.96  Fail to Reject    +1.96  Reject H₀

For a one-tailed right test at α = 0.05:
  Critical value: z* = 1.645

  Rejection region:                              │
  ─────────────────────────────────────────── ──┼─────
  Fail to Reject H₀                           +1.645  Reject
```

### Decision Rule Using Critical Values

```
Two-tailed test:
  If |test statistic| > critical value → Reject H₀
  If |test statistic| ≤ critical value → Fail to reject H₀

One-tailed right (H₁: μ > μ₀):
  If test statistic > critical value → Reject H₀

One-tailed left (H₁: μ < μ₀):
  If test statistic < -critical value → Reject H₀
```

---

## 12. p-value

### What is it?

The **p-value** is the probability of observing a test statistic at least as extreme as the one computed from the sample, **assuming H₀ is true**.

```
p-value = P(observing data this extreme OR more extreme | H₀ is true)

Small p-value → The observed data is very unlikely under H₀
               → Strong evidence against H₀ → Reject H₀

Large p-value → The observed data is not unusual under H₀
               → Weak evidence against H₀ → Fail to reject H₀
```

### Decision Rule Using p-value

```
If p-value < α  → Reject H₀  ("statistically significant")
If p-value ≥ α  → Fail to reject H₀  ("not statistically significant")
```

### p-value Visualisation

```
Null distribution (distribution of test statistic when H₀ is true):

           ╭───────╮
          ╭╯       ╰╮
         ╭╯         ╰╮
        ╭╯           ╰╮
───────╭╯             ╰╮───────────────────────────────
                       │observed test statistic
                       └──────────────────────────────┘
                               p-value = this shaded area
                        (probability of getting something
                         this extreme or more extreme)
```

### Critical Interpretation Points

```
1. p-value is NOT the probability that H₀ is true
   WRONG: "p=0.03 means there's a 3% chance H₀ is true"
   RIGHT: "If H₀ were true, there would be a 3% chance of
            seeing data this extreme"

2. Statistical significance ≠ practical significance
   A drug that reduces systolic BP by 0.5 mmHg might be
   statistically significant (p=0.001) but practically useless.

3. p-value > α does NOT prove H₀ is true
   It only means the data does not provide enough evidence to reject it.
   Absence of evidence is NOT evidence of absence.

4. p-value depends on sample size
   Large samples → small p-values even for tiny, unimportant effects.
   Always pair p-values with EFFECT SIZE.
```

---

## 13. Type I and Type II Errors

### The Four Possible Outcomes

When making a hypothesis test decision, there are four possible outcomes:

```
                        REALITY
                 H₀ TRUE        H₀ FALSE
               ┌─────────────┬──────────────┐
DECISION  H₀   │  Correct    │  Type II     │
  Fail to      │  Decision   │  Error (β)   │
  Reject H₀    │  ✓          │  False Neg.  │
               ├─────────────┼──────────────┤
  Reject H₀    │  Type I     │  Correct     │
               │  Error (α)  │  Decision    │
               │  False Pos. │  ✓           │
               └─────────────┴──────────────┘
```

### Type I Error (False Positive)

```
Type I Error (α):
  Reality: H₀ is TRUE (no effect/difference)
  Decision: We REJECT H₀ (conclude there is an effect)

  "We declare a difference when there is none"
  "False alarm"

Examples:
  Medical: Approving a drug that doesn't actually work
  ML: Concluding Model B is better when it isn't (random test set luck)
  A/B test: Claiming Variant B converts better when it doesn't
  Security: False positive — flagging an innocent email as spam

Probability of Type I error = α (controlled by us — significance level)
```

### Type II Error (False Negative)

```
Type II Error (β):
  Reality: H₀ is FALSE (there IS an effect)
  Decision: We FAIL TO REJECT H₀ (miss the effect)

  "We miss a real difference"
  "Missed opportunity"

Examples:
  Medical: Rejecting a drug that actually works
  ML: Failing to detect that Model B is genuinely better
  Security: Spam email that passes the filter (false negative)
  Medicine: Failing to diagnose a disease that is present

Probability of Type II error = β (beta)
```

### The Trade-off

```
α and β move in OPPOSITE DIRECTIONS for fixed sample size:

If you make α smaller (stricter):
  → Harder to reject H₀
  → FEWER Type I errors (fewer false positives)
  → MORE Type II errors (more false negatives)

If you make α larger (more lenient):
  → Easier to reject H₀
  → MORE Type I errors
  → FEWER Type II errors

THE ONLY WAY TO REDUCE BOTH: Increase sample size
```

### In ML Context

| Scenario | Type I Error | Type II Error |
|---|---|---|
| **Spam filter** | Ham → Spam (important email deleted) | Spam → Ham (spam in inbox) |
| **Medical AI** | Healthy → Sick (unnecessary treatment) | Sick → Healthy (missed diagnosis) |
| **Fraud detection** | Legitimate → Fraud (card blocked) | Fraud → Legitimate (fraud succeeds) |
| **A/B testing** | B better when it isn't | B better but we don't detect it |

**Which error is worse depends on context.** For cancer screening, Type II (missing cancer) is much worse than Type I (unnecessary biopsy). For spam filters, Type I (losing important email) might be worse.

---

## 14. Statistical Power

### What is it?

**Statistical Power** (1 − β) is the probability of **correctly rejecting H₀ when H₀ is false** — the probability of detecting a real effect when it exists.

```
Power = 1 - β = P(Reject H₀ | H₀ is FALSE) = P(Detect real effect)

Power = 0.8  → 80% chance of detecting a real effect
              → 20% chance of missing it (β = 0.20)

Minimum acceptable power: 0.80 (80%) — common convention in research
```

### Factors That Increase Power

```
1. Larger sample size (n↑)
   More data → less sampling variability → easier to detect effects

2. Larger effect size (d↑)
   A big difference is easier to detect than a small one

3. Larger significance level (α↑)
   Easier to reject H₀ → more power
   But: also more Type I errors (trade-off)

4. Smaller population variance (σ↓)
   Less noise in data → easier to detect signal
   Can be reduced by: better measurement, controlled experiments

5. One-tailed vs two-tailed test
   One-tailed test has more power than two-tailed
   (all rejection probability concentrated on one side)
```

### Cohen's d — Effect Size

```
Cohen's d = (μ₁ - μ₂) / σ_pooled

Interpretation:
  d = 0.2  → Small effect
  d = 0.5  → Medium effect
  d = 0.8  → Large effect

Power analysis uses effect size to determine required sample size.
```

### Python: Power and Sample Size

```python
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

# ── Power Analysis ────────────────────────────────────────────────
def calculate_power(n, effect_size, alpha=0.05, two_tailed=True):
    """
    Calculate statistical power for a one-sample t-test.
    effect_size: Cohen's d = (μ - μ₀) / σ
    """
    from scipy.stats import norm, nct
    # Non-centrality parameter
    ncp = effect_size * np.sqrt(n)
    # Critical value
    if two_tailed:
        t_crit = stats.t.ppf(1 - alpha/2, df=n-1)
    else:
        t_crit = stats.t.ppf(1 - alpha, df=n-1)
    # Power = P(reject H₀ | H₁ true)
    # Under H₁, test stat follows non-central t distribution
    power = 1 - stats.nct.cdf(t_crit, df=n-1, nc=ncp)
    if two_tailed:
        power += stats.nct.cdf(-t_crit, df=n-1, nc=ncp)
    return power

# Power vs sample size for different effect sizes
n_values = np.arange(10, 201, 5)
effect_sizes = [0.2, 0.5, 0.8]   # Small, Medium, Large
labels = ['Small (d=0.2)', 'Medium (d=0.5)', 'Large (d=0.8)']
colours = ['red', 'orange', 'green']

plt.figure(figsize=(10, 5))
for d, label, col in zip(effect_sizes, labels, colours):
    powers = [calculate_power(n, d) for n in n_values]
    plt.plot(n_values, powers, label=label, color=col, linewidth=2)

plt.axhline(0.80, color='black', linestyle='--', label='Power = 0.80 threshold')
plt.xlabel("Sample Size (n)"); plt.ylabel("Statistical Power")
plt.title("Power vs Sample Size for Different Effect Sizes")
plt.legend(); plt.grid(True, alpha=0.3); plt.ylim(0, 1)
plt.savefig('power_analysis.png', dpi=100, bbox_inches='tight')
plt.show()

# Using statsmodels for power analysis
try:
    from statsmodels.stats.power import TTestPower
    analysis = TTestPower()
    # Given desired power=0.8 and effect_size=0.5, what n do we need?
    n_needed = analysis.solve_power(effect_size=0.5, power=0.8, alpha=0.05)
    print(f"For power=80%, α=5%, medium effect (d=0.5): need n={int(np.ceil(n_needed))}")
except ImportError:
    print("statsmodels not available — use scipy directly")
```

---

# PART C — STATISTICAL TESTS

---

## 15. One-Sample Mean Test

### What is it?

Tests whether the **sample mean** is significantly different from a **hypothesised population mean** (μ₀).

```
H₀: μ = μ₀   (population mean equals a specific value)
H₁: μ ≠ μ₀  (two-tailed), or μ > μ₀, or μ < μ₀

Use when:
  You have ONE sample and want to test it against a known standard
```

### Example

```
A food company claims their chips weigh 200g per bag.
You sample 25 bags: x̄ = 197g, s = 5g.
Is there evidence the mean weight is different from 200g?

H₀: μ = 200
H₁: μ ≠ 200   (two-tailed — could be over or under)
α = 0.05

σ unknown (use s) → t-test
t = (x̄ - μ₀) / (s/√n) = (197 - 200) / (5/√25) = -3 / 1 = -3.0

df = n - 1 = 24
Critical value: t₀.₀₂₅,₂₄ = ±2.064
|t| = 3.0 > 2.064 → REJECT H₀

Conclusion: There IS significant evidence that the mean weight
differs from 200g (p < 0.05).
```

---

## 16. Two-Sample Mean Test

### What is it?

Tests whether **two independent samples** have the same population mean.

```
H₀: μ₁ = μ₂   (the two populations have the same mean)
H₁: μ₁ ≠ μ₂  (or >, or <)

Use when:
  Comparing two groups (treatment vs control, two models, two products)
```

### Example

```
Drug A vs Drug B for blood pressure reduction.
Group A (n=30): mean reduction x̄₁ = 12 mmHg, s₁ = 4
Group B (n=30): mean reduction x̄₂ = 9 mmHg,  s₂ = 5

H₀: μ_A = μ_B  (no difference between drugs)
H₁: μ_A ≠ μ_B  (drugs differ)

Test: Two-sample t-test (independent groups, σ unknown)
```

---

## 17. Proportion Test

### What is it?

Tests a claim about a **population proportion** (percentage, probability, rate).

```
One-sample proportion test:
  H₀: p = p₀   (population proportion equals a specific value)

Two-sample proportion test:
  H₀: p₁ = p₂  (two populations have same proportion)

Test statistic:
  z = (p̂ - p₀) / √(p₀(1-p₀)/n)
```

### Example

```
Your website historically has 10% conversion rate.
After a redesign, 65 out of 500 visitors converted.
Is the new conversion rate significantly higher?

p̂ = 65/500 = 0.13
H₀: p = 0.10
H₁: p > 0.10   (one-tailed right)
α = 0.05

z = (0.13 - 0.10) / √(0.10 × 0.90 / 500)
  = 0.03 / √(0.00018)
  = 0.03 / 0.01342
  = 2.236

Critical value (one-tailed, α=0.05): z* = 1.645
2.236 > 1.645 → REJECT H₀

Conclusion: The redesign significantly improved conversion rate (p ≈ 0.013).
```

---

## 18. t-test

### What is it?

The **t-test** is used to test hypotheses about means when:
- The population standard deviation σ is **unknown** (almost always in practice)
- The sample size is **small** (though it works for large samples too)
- The data is approximately normally distributed

### The t-Distribution

```
The t-distribution is like the Normal distribution but with heavier tails.

Why heavier tails?
  We estimate σ from the sample (using s), which introduces extra uncertainty.
  Heavier tails account for this additional uncertainty.

Degrees of freedom (df) = n - 1
  Larger df → t-distribution closer to Normal
  At df → ∞, t-distribution = Standard Normal

df=1  → Very heavy tails (much uncertainty)
df=5  → Moderately heavy tails
df=30 → Close to Normal
df=∞  → Exactly Normal
```

### Three Types of t-test

**1. One-sample t-test:** Compare sample mean to a known value

```
t = (x̄ - μ₀) / (s/√n)    df = n-1
```

**2. Independent samples t-test:** Compare two independent group means

```
t = (x̄₁ - x̄₂) / SE_diff

SE_diff = √(s₁²/n₁ + s₂²/n₂)    (Welch's t-test, unequal variances)

or

SE_diff = sp × √(1/n₁ + 1/n₂)   (Pooled, equal variances assumed)

where sp² = ((n₁-1)s₁² + (n₂-1)s₂²) / (n₁+n₂-2)   (pooled variance)
```

**3. Paired samples t-test:** Compare two related measurements (before/after)

```
d = x₁ - x₂   (difference for each paired observation)

t = d̄ / (s_d/√n)    df = n-1

where d̄ = mean of differences
      s_d = std dev of differences
```

### Python: t-tests

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

np.random.seed(42)

# ── ONE-SAMPLE t-TEST ─────────────────────────────────────────────
print("=" * 55)
print("ONE-SAMPLE t-TEST")
print("=" * 55)
print("Claim: mean chip weight = 200g. Is our sample different?\n")

weights = np.array([195, 198, 202, 197, 199, 196, 203, 198,
                    197, 200, 196, 199, 198, 197, 201, 196,
                    198, 199, 197, 200, 195, 198, 199, 197, 198])
mu_0 = 200

t_stat, p_value = stats.ttest_1samp(weights, popmean=mu_0)

print(f"Sample mean: {weights.mean():.2f}g")
print(f"Sample std:  {weights.std(ddof=1):.2f}g")
print(f"n:           {len(weights)}")
print(f"H₀: μ = {mu_0}g")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value:     {p_value:.4f}")
print(f"Decision:    {'Reject H₀' if p_value < 0.05 else 'Fail to reject H₀'}")
print(f"Conclusion:  {'Mean significantly ≠ 200g' if p_value < 0.05 else 'No evidence mean ≠ 200g'}")

# ── INDEPENDENT SAMPLES t-TEST ────────────────────────────────────
print("\n" + "=" * 55)
print("INDEPENDENT SAMPLES t-TEST")
print("=" * 55)
print("Drug A vs Drug B — blood pressure reduction\n")

drug_A = stats.norm.rvs(loc=12, scale=4, size=30, random_state=1)
drug_B = stats.norm.rvs(loc=9,  scale=5, size=30, random_state=2)

t_stat, p_value = stats.ttest_ind(drug_A, drug_B, equal_var=False)   # Welch's

print(f"Drug A: mean={drug_A.mean():.2f}, std={drug_A.std(ddof=1):.2f}, n={len(drug_A)}")
print(f"Drug B: mean={drug_B.mean():.2f}, std={drug_B.std(ddof=1):.2f}, n={len(drug_B)}")
print(f"Difference: {drug_A.mean() - drug_B.mean():.2f} mmHg")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value:     {p_value:.4f}")
print(f"Decision:    {'Reject H₀' if p_value < 0.05 else 'Fail to reject H₀'}")

# Effect size (Cohen's d)
pooled_std = np.sqrt((drug_A.var(ddof=1) + drug_B.var(ddof=1)) / 2)
cohens_d = (drug_A.mean() - drug_B.mean()) / pooled_std
print(f"Cohen's d:   {cohens_d:.4f} ({'small' if abs(cohens_d)<0.5 else 'medium' if abs(cohens_d)<0.8 else 'large'})")

# ── PAIRED t-TEST ─────────────────────────────────────────────────
print("\n" + "=" * 55)
print("PAIRED SAMPLES t-TEST")
print("=" * 55)
print("Before and after a training program\n")

before = np.array([75, 68, 82, 71, 79, 85, 70, 77, 72, 80])
after  = np.array([80, 72, 85, 75, 84, 88, 74, 81, 76, 85])

t_stat, p_value = stats.ttest_rel(after, before)
differences = after - before

print(f"Before: mean={before.mean():.2f}")
print(f"After:  mean={after.mean():.2f}")
print(f"Mean improvement: {differences.mean():.2f}")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value:     {p_value:.4f}")
print(f"Decision:    {'Reject H₀' if p_value < 0.05 else 'Fail to reject H₀'}")
print(f"Conclusion:  {'Training significantly improved scores' if p_value < 0.05 else 'No significant improvement'}")
```

---

## 19. z-test

### What is it?

The **z-test** is used when:
- The population standard deviation σ is **known**
- OR the sample size is **large** (n ≥ 30, where by CLT the sample mean is approximately normal)

```
z-test statistic:
  z = (x̄ - μ₀) / (σ/√n)    (known σ)
  z = (x̄ - μ₀) / (s/√n)    (large n, s ≈ σ)

For proportions:
  z = (p̂ - p₀) / √(p₀(1-p₀)/n)
```

### t-test vs z-test

| Feature | z-test | t-test |
|---|---|---|
| **σ** | Known | Unknown (use s) |
| **Sample size** | Large (n ≥ 30) preferred | Works for any n |
| **Distribution** | Standard Normal | t-distribution (df = n−1) |
| **Practical use** | Less common (σ rarely known) | **More common in practice** |
| **Large n** | Both give same result | Converges to z-test |

**Key insight:** For large samples (n ≥ 30), t-test and z-test give virtually identical results because the t-distribution approaches Normal for large df. In practice, **always use the t-test** — it is always correct and converges to z-test for large n.

### Python: z-test for Proportions

```python
import numpy as np
from scipy import stats

# ── z-test for website conversion rate ────────────────────────────
print("=== z-TEST FOR PROPORTION ===")
print("Historical conversion rate: 10%")
print("New design: 65 conversions out of 500 visitors\n")

p_hat  = 65 / 500    # Sample proportion
p_0    = 0.10        # Hypothesised proportion
n      = 500

# Test statistic
se = np.sqrt(p_0 * (1 - p_0) / n)
z  = (p_hat - p_0) / se
p_value = 1 - stats.norm.cdf(z)   # One-tailed (right): H₁: p > p₀

print(f"p̂ = {p_hat:.4f}")
print(f"SE = {se:.4f}")
print(f"z-statistic = {z:.4f}")
print(f"p-value (one-tailed) = {p_value:.4f}")
print(f"Decision: {'Reject H₀' if p_value < 0.05 else 'Fail to reject H₀'}")
print(f"Conclusion: {'New design significantly improved conversion rate' if p_value < 0.05 else 'No significant improvement'}")

# ── z-test for large sample mean ───────────────────────────────────
print("\n=== z-TEST FOR MEAN (large n) ===")
np.random.seed(42)
sample = stats.norm.rvs(loc=101, scale=15, size=100)   # True μ=101, H₀: μ=100

z = (sample.mean() - 100) / (sample.std(ddof=1) / np.sqrt(len(sample)))
p_value = 2 * (1 - stats.norm.cdf(abs(z)))   # Two-tailed

print(f"Sample mean: {sample.mean():.4f}, n=100")
print(f"z-statistic: {z:.4f}")
print(f"p-value:     {p_value:.4f}")
print(f"Decision: {'Reject H₀' if p_value < 0.05 else 'Fail to reject H₀'}")
```

---

## 20. Chi-Square Test

### What is it?

The **chi-square (χ²) test** is used for categorical data. It tests whether:
1. A categorical variable follows an expected distribution (**Goodness of Fit test**)
2. Two categorical variables are **independent** of each other (**Test of Independence**)

### Chi-Square Statistic

```
χ² = Σ (O - E)² / E

Where:
  O = Observed frequency (actual count)
  E = Expected frequency (count if H₀ were true)
  Σ = Sum over all categories

Degrees of freedom:
  Goodness of fit:      df = k - 1    (k = number of categories)
  Test of independence: df = (r-1)(c-1) (r = rows, c = columns)
```

### Test of Independence — Example

```
Question: Is there a relationship between gender and product preference?

Observed data (survey of 200 people):
              Product A   Product B   Product C   Total
Male               50          30          20      100
Female             30          40          30      100
Total              80          70          50      200

Expected if INDEPENDENT (E = row_total × col_total / grand_total):
              Product A   Product B   Product C
Male         80×100/200  70×100/200  50×100/200  = 40, 35, 25
Female       80×100/200  70×100/200  50×100/200  = 40, 35, 25

χ² = (50-40)²/40 + (30-35)²/35 + (20-25)²/25
   + (30-40)²/40 + (40-35)²/35 + (30-25)²/25

   = 2.5 + 0.714 + 1.0 + 2.5 + 0.714 + 1.0

   = 8.428

df = (2-1)(3-1) = 2
χ²_critical (α=0.05, df=2) = 5.991

8.428 > 5.991 → REJECT H₀

Conclusion: Gender and product preference are NOT independent
            (there IS a significant relationship)
```

### Python: Chi-Square Test

```python
import numpy as np
from scipy import stats
import pandas as pd

print("=" * 55)
print("CHI-SQUARE TEST OF INDEPENDENCE")
print("=" * 55)

# Observed contingency table
observed = np.array([[50, 30, 20],
                     [30, 40, 30]])

# scipy chi2_contingency automatically computes expected frequencies
chi2, p_value, dof, expected = stats.chi2_contingency(observed)

print("Observed frequencies:")
print(pd.DataFrame(observed, index=['Male', 'Female'],
                   columns=['Product A', 'Product B', 'Product C']))

print("\nExpected frequencies (if independent):")
print(pd.DataFrame(expected.round(2), index=['Male', 'Female'],
                   columns=['Product A', 'Product B', 'Product C']))

print(f"\nChi-square statistic: {chi2:.4f}")
print(f"Degrees of freedom:   {dof}")
print(f"p-value:              {p_value:.4f}")
print(f"Decision: {'Reject H₀' if p_value < 0.05 else 'Fail to reject H₀'}")
print(f"Conclusion: {'Significant relationship between gender and product preference' if p_value < 0.05 else 'No significant relationship'}")

# Cramér's V — effect size for chi-square
n = observed.sum()
cramers_v = np.sqrt(chi2 / (n * (min(observed.shape) - 1)))
print(f"Cramér's V (effect size): {cramers_v:.4f}")

# ── GOODNESS OF FIT TEST ──────────────────────────────────────────
print("\n" + "=" * 55)
print("CHI-SQUARE GOODNESS OF FIT TEST")
print("=" * 55)
print("Is the die fair?\n")

observed_die = np.array([18, 22, 16, 20, 21, 23])   # 120 rolls
expected_die = np.array([20, 20, 20, 20, 20, 20])   # Expected if fair

chi2_gof, p_gof = stats.chisquare(f_obs=observed_die, f_exp=expected_die)

for face, obs, exp in zip(range(1,7), observed_die, expected_die):
    print(f"  Face {face}: Observed={obs}, Expected={exp}")

print(f"\nChi-square: {chi2_gof:.4f}")
print(f"p-value:    {p_gof:.4f}")
print(f"Decision: {'Reject H₀ (die is biased)' if p_gof < 0.05 else 'Fail to reject H₀ (no evidence of bias)'}")

# ── Chi-Square in ML: Feature Selection ──────────────────────────
print("\n=== Chi-Square for Feature Selection in ML ===")
from sklearn.datasets import load_iris
from sklearn.feature_selection import chi2 as chi2_sklearn
from sklearn.preprocessing import KBinsDiscretizer

iris = load_iris()
X, y = iris.data, iris.target

# Discretise continuous features for chi-square
disc = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='quantile')
X_disc = disc.fit_transform(X).astype(int)

chi2_scores, p_values = chi2_sklearn(X_disc, y)
print("Chi-square scores for Iris features:")
for feat, score, pval in zip(iris.feature_names, chi2_scores, p_values):
    print(f"  {feat:30s}: χ²={score:.2f}, p={pval:.4f}")
```

---

## 21. ANOVA

### What is it?

**ANOVA (Analysis of Variance)** tests whether **three or more groups** have the same population mean.

```
Why not just use multiple t-tests?
  Comparing 5 groups pairwise: C(5,2) = 10 t-tests
  At α = 0.05 each: P(at least one false positive) ≈ 1-(0.95)¹⁰ ≈ 40%!
  
  ANOVA controls the overall Type I error rate at α.
  It asks ONE question: "Are ANY of the means different?"
```

### One-Way ANOVA

```
H₀: μ₁ = μ₂ = μ₃ = ... = μₖ    (all k groups have the same mean)
H₁: At least one μᵢ ≠ μⱼ        (at least two means are different)

F-statistic:
  F = Variance BETWEEN groups / Variance WITHIN groups
    = Mean Square Between (MSB) / Mean Square Within (MSW)

If H₀ is true: F ≈ 1  (between-group variance ≈ within-group noise)
If H₁ is true: F >> 1 (between-group variance >> within-group noise)
```

### Decomposition of Variance

```
Total variation = Within-group variation + Between-group variation

SS_Total = SS_Within + SS_Between

SS_Between = Σⱼ nⱼ(x̄ⱼ - x̄_overall)²   (how far group means are from overall mean)
SS_Within  = Σⱼ Σᵢ (xᵢⱼ - x̄ⱼ)²          (how far each point is from its group mean)

MS_Between = SS_Between / (k-1)    df = k-1   (k = number of groups)
MS_Within  = SS_Within / (n-k)     df = n-k   (n = total observations)

F = MS_Between / MS_Within
```

### Post-hoc Tests

ANOVA tells you THAT at least two means differ, but not WHICH ones.

Use **post-hoc tests** to find out which pairs differ:
- **Tukey's HSD** (most common — controls overall Type I error)
- **Bonferroni correction** (divide α by number of comparisons)
- **Dunnett's test** (compare each group to one control)

### Python: ANOVA

```python
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)
print("=" * 55)
print("ONE-WAY ANOVA")
print("=" * 55)
print("Do three teaching methods produce different exam scores?\n")

# Three teaching methods
method_A = stats.norm.rvs(loc=75, scale=10, size=25)
method_B = stats.norm.rvs(loc=80, scale=10, size=25)
method_C = stats.norm.rvs(loc=85, scale=10, size=25)

print(f"Method A: mean={method_A.mean():.2f}, std={method_A.std(ddof=1):.2f}")
print(f"Method B: mean={method_B.mean():.2f}, std={method_B.std(ddof=1):.2f}")
print(f"Method C: mean={method_C.mean():.2f}, std={method_C.std(ddof=1):.2f}")

# One-way ANOVA
f_stat, p_value = stats.f_oneway(method_A, method_B, method_C)

print(f"\nF-statistic: {f_stat:.4f}")
print(f"p-value:     {p_value:.4f}")
print(f"Decision: {'Reject H₀' if p_value < 0.05 else 'Fail to reject H₀'}")
print(f"Conclusion: {'At least one method produces significantly different scores' if p_value < 0.05 else 'No significant difference between methods'}")

# ── Post-hoc: Tukey's HSD ─────────────────────────────────────────
if p_value < 0.05:
    print("\n--- Post-hoc: Tukey's HSD ---")
    try:
        from statsmodels.stats.multicomp import pairwise_tukeyhsd
        all_scores = np.concatenate([method_A, method_B, method_C])
        groups = ['A']*25 + ['B']*25 + ['C']*25
        tukey = pairwise_tukeyhsd(all_scores, groups, alpha=0.05)
        print(tukey.summary())
    except ImportError:
        # Manual Bonferroni correction
        pairs = [(method_A, method_B, 'A vs B'),
                 (method_A, method_C, 'A vs C'),
                 (method_B, method_C, 'B vs C')]
        alpha_corrected = 0.05 / 3   # Bonferroni
        print(f"Bonferroni-corrected α = {alpha_corrected:.4f}")
        for g1, g2, label in pairs:
            t, p = stats.ttest_ind(g1, g2)
            sig = "✓ significant" if p < alpha_corrected else "✗ not significant"
            print(f"  {label}: t={t:.3f}, p={p:.4f}  {sig}")

# ── Visualise ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
data_groups = [method_A, method_B, method_C]
ax.boxplot(data_groups, labels=['Method A', 'Method B', 'Method C'],
           patch_artist=True,
           boxprops=dict(alpha=0.7))
ax.set_ylabel("Exam Score")
ax.set_title(f"ANOVA: Teaching Methods\n(F={f_stat:.2f}, p={p_value:.4f})")
ax.grid(True, alpha=0.3, axis='y')
plt.savefig('anova.png', dpi=100, bbox_inches='tight')
plt.show()

# ── ANOVA in ML: Feature importance ──────────────────────────────
print("\n=== ANOVA F-test for Feature Selection ===")
from sklearn.datasets import load_iris
from sklearn.feature_selection import f_classif

iris = load_iris()
F_scores, p_values = f_classif(iris.data, iris.target)
print("ANOVA F-scores for Iris features (how much each feature varies between classes):")
for feat, F, p in zip(iris.feature_names, F_scores, p_values):
    print(f"  {feat:30s}: F={F:.2f}, p={p:.6f}")
```

---

## 22. Choosing the Right Test

### Decision Framework

```
WHAT TYPE OF DATA DO YOU HAVE?

├── CATEGORICAL (counts, frequencies)
│     │
│     ├── One categorical variable
│     │     → Chi-square Goodness of Fit Test
│     │
│     └── Two categorical variables
│           → Chi-square Test of Independence
│
└── NUMERICAL (continuous, scale)
      │
      ├── ONE SAMPLE — comparing to known value
      │     σ known, any n:    z-test
      │     σ unknown:         One-sample t-test
      │
      ├── TWO SAMPLES — comparing two groups
      │     Independent:       Independent samples t-test (Welch's)
      │     Paired/matched:    Paired samples t-test
      │     Proportions:       Two-proportion z-test
      │
      └── THREE OR MORE GROUPS
            → One-way ANOVA
            → Post-hoc tests if significant
```

### Quick Reference Table

| Test | Data | Groups | σ | When to Use |
|---|---|---|---|---|
| **One-sample z** | Numerical | 1 | Known | Large n, σ known |
| **One-sample t** | Numerical | 1 | Unknown | Any n, σ unknown |
| **Independent t** | Numerical | 2 | Unknown | Compare 2 independent groups |
| **Paired t** | Numerical | 2 | Unknown | Before/after, matched pairs |
| **Two-prop z** | Proportions | 2 | — | Compare 2 proportions |
| **One-way ANOVA** | Numerical | ≥3 | Unknown | Compare 3+ group means |
| **Chi-square GoF** | Categorical | 1 | — | Test distribution shape |
| **Chi-square Indep.** | Categorical | 2 vars | — | Test independence |

### Python: Complete Hypothesis Testing Framework

```python
import numpy as np
from scipy import stats

def hypothesis_test(data1, data2=None, test_type='one_sample_t',
                    mu_0=None, alpha=0.05, alternative='two-sided'):
    """
    General hypothesis testing framework.

    test_type options:
      'one_sample_t'      - compare one sample mean to known value
      'two_sample_t'      - compare two independent sample means
      'paired_t'          - compare paired observations
      'proportion_z'      - test a proportion against known value
      'chi2_independence' - test independence of two categorical variables
    """
    print(f"\n{'='*55}")
    print(f"Test: {test_type}")
    print(f"Significance level: α = {alpha}")
    print(f"Alternative: H₁ is {alternative}")

    if test_type == 'one_sample_t':
        stat, p = stats.ttest_1samp(data1, popmean=mu_0, alternative=alternative)
        print(f"H₀: μ = {mu_0}")
        print(f"x̄ = {np.mean(data1):.4f}, s = {np.std(data1,ddof=1):.4f}, n = {len(data1)}")

    elif test_type == 'two_sample_t':
        stat, p = stats.ttest_ind(data1, data2, equal_var=False, alternative=alternative)
        print(f"H₀: μ₁ = μ₂")
        print(f"Group 1: x̄={np.mean(data1):.4f}, n={len(data1)}")
        print(f"Group 2: x̄={np.mean(data2):.4f}, n={len(data2)}")
        print(f"Difference: {np.mean(data1)-np.mean(data2):.4f}")

    elif test_type == 'paired_t':
        stat, p = stats.ttest_rel(data1, data2, alternative=alternative)
        diffs = np.array(data1) - np.array(data2)
        print(f"H₀: μ_diff = 0")
        print(f"Mean difference: {diffs.mean():.4f}")

    elif test_type == 'proportion_z':
        # data1 = (n_successes, n_total)
        n_success, n_total = data1
        p_hat = n_success / n_total
        se = np.sqrt(mu_0 * (1 - mu_0) / n_total)
        stat = (p_hat - mu_0) / se
        if alternative == 'two-sided':
            p = 2 * (1 - stats.norm.cdf(abs(stat)))
        elif alternative == 'greater':
            p = 1 - stats.norm.cdf(stat)
        else:
            p = stats.norm.cdf(stat)
        print(f"H₀: p = {mu_0}")
        print(f"p̂ = {p_hat:.4f}, n = {n_total}")

    print(f"Test statistic: {stat:.4f}")
    print(f"p-value: {p:.4f}")
    decision = "Reject H₀" if p < alpha else "Fail to reject H₀"
    sig = "SIGNIFICANT" if p < alpha else "not significant"
    print(f"Decision: {decision}  [{sig}]")
    return stat, p

# Run tests
np.random.seed(42)

# Test 1: One-sample t-test
sample_data = stats.norm.rvs(loc=50.5, scale=8, size=40)
hypothesis_test(sample_data, test_type='one_sample_t', mu_0=50, alpha=0.05)

# Test 2: Two-sample t-test
group1 = stats.norm.rvs(loc=10, scale=3, size=35)
group2 = stats.norm.rvs(loc=11.5, scale=3.5, size=35)
hypothesis_test(group1, group2, test_type='two_sample_t', alpha=0.05)

# Test 3: Proportion test
hypothesis_test((65, 500), test_type='proportion_z',
                mu_0=0.10, alpha=0.05, alternative='greater')
```

---

## 🔬 Practical: A/B Testing in Data Science

A/B testing is the most common application of hypothesis testing in industry.

```python
import numpy as np
from scipy import stats

np.random.seed(42)

print("=" * 60)
print("A/B TEST: New Website Design vs Original")
print("=" * 60)

# Simulation: Control (A) and Treatment (B)
n_A = 1000   # visitors to original design
n_B = 1000   # visitors to new design

# True conversion rates (unknown to us — only have sample data)
p_A_true = 0.12
p_B_true = 0.15   # Treatment is slightly better

# Simulate observed conversions (what we actually observe)
conversions_A = stats.binom.rvs(n=1, p=p_A_true, size=n_A).sum()
conversions_B = stats.binom.rvs(n=1, p=p_B_true, size=n_B).sum()

p_hat_A = conversions_A / n_A
p_hat_B = conversions_B / n_B

print(f"\nControl (A): {conversions_A}/{n_A} conversions = {p_hat_A:.3%}")
print(f"Treatment (B): {conversions_B}/{n_B} conversions = {p_hat_B:.3%}")
print(f"Lift: {(p_hat_B - p_hat_A)/p_hat_A:.1%}")

# Hypothesis test: Is B better than A?
# H₀: p_B = p_A   (no difference)
# H₁: p_B > p_A   (treatment is better) — one-tailed

# Pooled proportion (under H₀)
p_pool = (conversions_A + conversions_B) / (n_A + n_B)
se = np.sqrt(p_pool * (1 - p_pool) * (1/n_A + 1/n_B))
z  = (p_hat_B - p_hat_A) / se
p_value = 1 - stats.norm.cdf(z)   # One-tailed

print(f"\nPooled proportion: {p_pool:.4f}")
print(f"z-statistic: {z:.4f}")
print(f"p-value: {p_value:.4f}")
print(f"Decision: {'Reject H₀ — launch new design!' if p_value < 0.05 else 'Fail to reject H₀ — keep original'}")

# Confidence interval for the difference
diff = p_hat_B - p_hat_A
se_diff = np.sqrt(p_hat_A*(1-p_hat_A)/n_A + p_hat_B*(1-p_hat_B)/n_B)
ci = (diff - 1.96*se_diff, diff + 1.96*se_diff)
print(f"\n95% CI for lift: [{ci[0]:.4f}, {ci[1]:.4f}]")
print(f"  If CI entirely above 0 → significant improvement")

# Minimum detectable effect (MDE)
def mde(n, p_baseline, alpha=0.05, power=0.80):
    z_alpha = stats.norm.ppf(1 - alpha)
    z_beta  = stats.norm.ppf(power)
    se      = np.sqrt(2 * p_baseline * (1-p_baseline) / n)
    return (z_alpha + z_beta) * se

mde_value = mde(n=1000, p_baseline=0.12)
print(f"\nMinimum detectable effect (n=1000): ±{mde_value:.4f} ({mde_value/0.12:.1%} relative)")
```

---

## 🔄 Concept Connections

```
This unit connects to:

PREVIOUS:
  Unit 3 (Statistics & Probability):
    Normal distribution → z-scores → confidence intervals
    CLT → sampling distributions → standard error
    Probability → p-values

FORWARD:
  Unit 5 (EDA):            Hypothesis tests during data exploration
  Unit 8 (Machine Learning): Train/test evaluation uses statistical thinking
  Unit 9 (Regression):     Significance tests for regression coefficients
  Unit 10 (Classification): Comparing classifier performance with t-tests
  Unit 12 (Ensemble):      Statistical basis for combining models
  Unit 33 (Evaluation):    Model comparison using statistical tests
  Unit 34 (Observability): Monitoring — detecting statistical shifts
  Unit 35 (MLOps):         A/B testing and canary deployments
```

---

## 🎓 Exam Notes

**Key formulas:**

| Concept | Formula |
|---|---|
| **Standard Error** | SE = σ/√n (or s/√n) |
| **CI for mean** | x̄ ± z* × SE |
| **Sample size (mean)** | n = (z* × σ / MoE)² |
| **z-test statistic** | z = (x̄ − μ₀) / (σ/√n) |
| **t-test statistic** | t = (x̄ − μ₀) / (s/√n) |
| **Proportion z** | z = (p̂ − p₀) / √(p₀(1−p₀)/n) |
| **Chi-square** | χ² = Σ(O−E)²/E |
| **F-statistic** | F = MS_Between / MS_Within |
| **Power** | Power = 1 − β |
| **Cohen's d** | d = (μ₁ − μ₂) / σ_pooled |

**Key distinctions:**
- α = P(Type I error) = significance level = P(reject H₀ | H₀ true)
- β = P(Type II error) = P(fail to reject | H₀ false)
- Power = 1 − β = P(correctly reject H₀ | H₀ false)
- p-value = P(data this extreme or more | H₀ true) — NOT P(H₀ true)
- t-test vs z-test: use t when σ unknown (almost always)

---

## 💼 Interview Notes

**Q1: What is the difference between a Type I and Type II error?**

> Type I error (false positive, α): Rejecting H₀ when it is actually true — detecting an effect that doesn't exist. Type II error (false negative, β): Failing to reject H₀ when it is false — missing a real effect. For a spam filter: Type I = legitimate email flagged as spam; Type II = spam email reaching inbox. They trade off against each other for fixed sample size; increasing sample size reduces both.

---

**Q2: What does a p-value actually mean?**

> The p-value is the probability of observing a test statistic as extreme or more extreme than what we calculated, assuming the null hypothesis is true. It is NOT the probability that H₀ is true, nor the probability our result is due to chance. If p=0.03: "If H₀ were true, there would be a 3% chance of seeing data this extreme or more extreme." Since 3% < 5% (α), we reject H₀.

---

**Q3: When would you use a t-test vs a z-test?**

> Use a z-test when the population standard deviation σ is known and/or sample size is large (n ≥ 30). Use a t-test when σ is unknown and must be estimated from the sample using s. In practice, σ is almost never known, so the t-test is almost always preferred. For large n, both tests give identical results because the t-distribution converges to Normal.

---

**Q4: What is statistical power and how do you increase it?**

> Statistical power (1−β) is the probability of detecting a real effect when one exists — the probability of correctly rejecting a false H₀. To increase power: (1) increase sample size — the most effective way; (2) increase the significance level α, though this increases Type I errors; (3) reduce measurement noise σ through better experimental design; (4) use a one-tailed instead of two-tailed test when direction is known.

---

**Q5: How is hypothesis testing used in ML?**

> Hypothesis testing appears throughout ML in several forms: (1) A/B testing — comparing model A vs B on user metrics using a proportion z-test or t-test; (2) Feature selection — chi-square test and ANOVA F-test identify which features have statistically significant relationships with the target; (3) Model evaluation — comparing two models' performance using a paired t-test on cross-validation scores; (4) Data monitoring — detecting statistical drift in production by testing whether new data's distribution differs from training data; (5) Experiment design — calculating required sample sizes for statistically valid experiments.

---

**Q6: What is the chi-square test and when is it used?**

> The chi-square test is used for categorical data. The goodness-of-fit test checks whether a categorical variable follows a hypothesised distribution. The independence test checks whether two categorical variables are independent. The test statistic χ² = Σ(O−E)²/E measures how far observed frequencies deviate from expected frequencies under H₀. In ML, chi-square is used for feature selection — features with high χ² scores have strong relationships with the target class.

---

## ⚠️ Common Mistakes & Misconceptions

**❌ Mistake 1: "p-value is the probability that H₀ is true"**
✅ The p-value is the probability of the DATA given that H₀ is true — not the probability of H₀. This is the most common misinterpretation in all of statistics. P(data | H₀) ≠ P(H₀ | data).

---

**❌ Mistake 2: "Failing to reject H₀ proves H₀ is true"**
✅ Failing to reject H₀ only means the data does not provide sufficient evidence against it. H₀ may still be false — you may have low power, small sample size, or a small effect. "Absence of evidence is not evidence of absence."

---

**❌ Mistake 3: "Statistical significance = practical importance"**
✅ With large enough samples, even trivially small, meaningless effects become statistically significant. A drug that reduces blood pressure by 0.3 mmHg might have p<0.001 but zero clinical relevance. Always report effect size (Cohen's d, Cramér's V) alongside p-values.

---

**❌ Mistake 4: "Running multiple t-tests instead of ANOVA"**
✅ With k groups, running C(k,2) pairwise t-tests inflates the Type I error rate dramatically. ANOVA controls the overall error rate at α. If ANOVA is significant, use post-hoc tests (Tukey's HSD, Bonferroni) to identify which specific pairs differ.

---

**❌ Mistake 5: "A larger sample always means a more meaningful result"**
✅ Larger samples give more statistical power — detecting smaller and smaller effects. But a tiny effect detected with a million-person sample may be practically irrelevant. Sample size planning should consider the minimum effect size that would be meaningful in practice, not just statistical detectability.

---

**❌ Mistake 6: "CI = range where the true parameter is with 95% probability"**
✅ The parameter is fixed. It is either inside the interval or it is not. 95% confidence means: if we repeated sampling and CI construction many times, 95% of those intervals would contain the true parameter. The probability statement is about the method, not about this specific interval.

---

## 🧠 Key Takeaways

1. **Inferential statistics** uses sample data to make statements about the population with quantified uncertainty.

2. **Confidence intervals** give a range of plausible values for population parameters. Width decreases with larger n; increases with higher confidence level. The 95% means the *method* captures the true parameter 95% of the time — not that *this specific interval* has 95% probability.

3. **Hypothesis testing** is a formal decision framework: state H₀ and H₁, set α, compute test statistic, find p-value, decide.

4. **H₀ is always the "no effect" claim** containing "=". H₁ is what you are trying to prove. You reject H₀ when the data is sufficiently incompatible with it.

5. **p-value** = P(data this extreme | H₀ true). Small p-value → reject H₀. It is NOT P(H₀ is true).

6. **Type I error (α)** = false positive = wrongly reject H₀. **Type II error (β)** = false negative = miss real effect. **Power = 1−β** = probability of detecting a real effect.

7. **t-test** is used when σ is unknown (almost always). **z-test** when σ is known or n is large. **Chi-square** for categorical data. **ANOVA** for 3+ groups.

8. **A/B testing** is the main industrial application of hypothesis testing — testing whether one version of a product is significantly better than another.

9. **Always pair p-values with effect sizes.** Statistical significance without practical significance is meaningless.

10. **Sample size matters.** Too small: miss real effects (low power). Too large: detect trivially small, meaningless effects.

---

## ❓ Revision Questions

### Basic Understanding
1. What is the difference between a point estimate and a confidence interval?
2. Define: null hypothesis, alternative hypothesis, significance level, p-value.
3. What is a Type I error? What is a Type II error? Give one example of each in an ML context.
4. What is statistical power? How does sample size affect it?
5. When do you use a t-test vs a z-test?

### Conceptual Understanding
6. Explain what a 95% confidence interval means. What is the common misconception about it?
7. A researcher says "p=0.06, so the result is almost significant." What is wrong with this statement?
8. Why shouldn't you run multiple t-tests instead of ANOVA when comparing three groups?
9. Explain the trade-off between Type I and Type II errors. How do you reduce both?
10. Why does a larger sample size lead to statistical significance even for trivially small effects?

### Application
11. You sample 50 customer wait times: x̄=6.3 min, s=1.8 min. Compute a 95% CI for the true mean wait time. (Use t* ≈ 2.01 for df=49)
12. A coin is flipped 100 times and gives 60 heads. Test at α=0.05 whether the coin is biased (H₁: p ≠ 0.5).
13. Model A scores [0.85, 0.87, 0.82, 0.88, 0.83] on five cross-validation folds. Model B scores [0.88, 0.91, 0.86, 0.89, 0.87]. Is Model B significantly better? (Paired t-test)
14. You want to estimate customer satisfaction with MoE=±3% at 95% confidence. How many customers must you survey?
15. Survey results: Male (prefer A: 50, prefer B: 30, prefer C: 20), Female (prefer A: 30, prefer B: 40, prefer C: 30). Is there a significant association between gender and product preference?

### Technical Understanding
16. What is the formula for the chi-square test statistic? What does each component mean?
17. In ANOVA, what does the F-statistic represent? When would you expect F≈1 and when F>>1?
18. What is Cohen's d? Why is it important alongside the p-value?
19. Explain the concept of "minimum detectable effect" in A/B testing. How does it relate to sample size?
20. What is Bonferroni correction? When and why is it used?

---

## 📝 Quick Revision

```
INFERENTIAL STATISTICS
──────────────────────
Population parameter vs sample statistic (μ vs x̄, p vs p̂)
Sampling methods: random, stratified, systematic, cluster
Standard Error: SE = σ/√n (spread of sampling distribution)
Point estimate: single best guess (x̄, p̂)
Confidence Interval: x̄ ± z*×SE  →  range for true parameter
  95% CI: z*=1.96  |  99% CI: z*=2.576  |  90% CI: z*=1.645
Margin of Error: MoE = z*×SE  →  half-width of CI
Sample size: n = (z*×σ/MoE)²

HYPOTHESIS TESTING
──────────────────
H₀: no effect/difference (contains "=")
H₁: researcher's claim (≠, >, <)
α = significance level = P(Type I error) — usually 0.05
Test statistic = (observed - expected) / SE
p-value = P(data this extreme | H₀ true)

Decision:
  p < α  →  Reject H₀  (significant)
  p ≥ α  →  Fail to reject H₀  (not significant)

Errors:
  Type I (α):  Reject H₀ when it's TRUE  (false positive)
  Type II (β): Keep H₀ when it's FALSE   (false negative)
  Power = 1−β = P(detect real effect)

STATISTICAL TESTS
─────────────────
One-sample t: t=(x̄-μ₀)/(s/√n), df=n-1    [σ unknown, compare to value]
Two-sample t: t=(x̄₁-x̄₂)/SE_diff           [compare 2 independent groups]
Paired t:     t=d̄/(s_d/√n), df=n-1         [before/after, matched pairs]
One-prop z:   z=(p̂-p₀)/√(p₀(1-p₀)/n)     [test proportion vs value]
Chi-square:   χ²=Σ(O-E)²/E                 [categorical data: GoF or independence]
ANOVA F:      F=MS_Between/MS_Within        [compare 3+ group means]
→ Follow with Tukey HSD for pairwise comparisons
```

---

## 📚 Further Learning

| Resource | Type | Why Useful |
|---|---|---|
| *All of Statistics* — Larry Wasserman | Book (free PDF) | Rigorous treatment of statistical theory |
| StatQuest with Josh Starmer | YouTube | Best visual explanations of t-tests, ANOVA, p-values |
| Khan Academy — Significance Tests | Course | Clear, step-by-step hypothesis testing |
| *Naked Statistics* — Charles Wheelan | Book | Intuitive statistical thinking without heavy math |
| statsmodels documentation | Docs | Python library for all statistical tests |
| scipy.stats documentation | Docs | All tests available in scipy |
| *The Lean Startup* — Eric Ries | Book | A/B testing philosophy in product development |

---

*📁 Next: `04-data-science/README.md` — Data Science tools, pandas, NumPy, matplotlib, and the full EDA workflow*