# Unit 3: Statistics & Probability for AI/ML

> 📁 **Repository:** `ai-ml-learning/03-statistics-probability/README.md`
> 🎯 **Phase:** 1 — Foundations
> 🧩 **Unit:** 3 of 36

---

## 📌 Overview

Every ML model makes decisions under **uncertainty**. A classifier does not say "this is spam" — it says "I am 94% confident this is spam." A regression model does not say "the price is exactly ₹50 lakhs" — it produces an estimate with uncertainty bounds. Statistics and probability are the languages of uncertainty.

This unit covers three pillars:

```
DESCRIPTIVE STATISTICS  →  Summarise and understand your data
PROBABILITY             →  Quantify uncertainty and reason about events
DISTRIBUTIONS           →  Model how data is generated
STATISTICAL CONCEPTS    →  Tools for inference and understanding relationships
```

These are not abstract mathematical topics. Every concept here directly maps to something you will do in a real ML project — from deciding how to handle outliers, to understanding why your model's predictions are probabilistic, to using the Central Limit Theorem to justify why many ML methods work.

---

## 🎯 Learning Objectives

By the end of this unit, you should be able to:

- [ ] Distinguish between population and sample, and explain why the distinction matters
- [ ] Compute and interpret mean, median, mode, variance, and standard deviation
- [ ] Calculate and interpret percentiles, quartiles, IQR, and the five-number summary
- [ ] Detect and handle outliers using the IQR and Z-score methods
- [ ] Define probability, sample space, and events
- [ ] Apply conditional probability, independence, and Bayes' Theorem
- [ ] Identify and use the key probability distributions in ML
- [ ] Compute expectation, covariance, and correlation
- [ ] Explain the Central Limit Theorem and its significance for ML

---

## 🗺️ Prerequisites

- Unit 1: Introduction to AI, ML & Data Science
- Unit 2: Mathematics for AI/ML (especially summation notation and functions)
- Basic algebra

---

## 📚 Topics Covered

**Part A — Descriptive Statistics**
1. [Population and Sample](#1-population-and-sample)
2. [Variables](#2-variables)
3. [Measures of Central Tendency](#3-measures-of-central-tendency)
4. [Measures of Spread](#4-measures-of-spread)
5. [Percentiles, Quartiles, IQR, and Five-Number Summary](#5-percentiles-quartiles-iqr-and-five-number-summary)

**Part B — Outliers**

6. [What is an Outlier?](#6-what-is-an-outlier)
7. [Outlier Detection Methods](#7-outlier-detection-methods)
8. [Handling Outliers](#8-handling-outliers)

**Part C — Probability**

9. [Basic Probability](#9-basic-probability)
10. [Events and Sample Space](#10-events-and-sample-space)
11. [Conditional Probability](#11-conditional-probability)
12. [Independence](#12-independence)
13. [Joint Probability](#13-joint-probability)
14. [Bayes Theorem](#14-bayes-theorem)

**Part D — Probability Distributions**

15. [Bernoulli Distribution](#15-bernoulli-distribution)
16. [Binomial Distribution](#16-binomial-distribution)
17. [Normal / Gaussian Distribution](#17-normal--gaussian-distribution)
18. [Uniform Distribution](#18-uniform-distribution)
19. [Poisson Distribution](#19-poisson-distribution)
20. [Exponential Distribution](#20-exponential-distribution)

**Part E — Statistical Concepts**

21. [Expectation](#21-expectation)
22. [Covariance](#22-covariance)
23. [Correlation](#23-correlation)
24. [Central Limit Theorem](#24-central-limit-theorem)
25. [Sampling Distributions](#25-sampling-distributions)

---

# PART A — DESCRIPTIVE STATISTICS

> **Why Descriptive Statistics?**
> Before building any model, you must understand your data. Descriptive statistics are the tools that let you summarise a dataset of thousands or millions of rows into a handful of numbers that reveal its shape, centre, and spread.

---

## 1. Population and Sample

### What is it?

**Population:** The complete set of all individuals or observations you are interested in studying.

**Sample:** A subset of the population, selected to represent it.

```
POPULATION:
  All 1.4 billion people in India
  All credit card transactions ever made
  All possible measurements of a patient's blood pressure

SAMPLE:
  1,000 people surveyed in a poll
  10,000 transactions selected for fraud analysis
  50 blood pressure readings from one patient
```

### Why do we need this distinction?

You almost never have access to the entire population. You work with a sample and use it to make inferences about the population. This is the foundation of statistical inference — and the reason every ML model is evaluated on a **test set** (a sample it has not seen) rather than the training data.

```
POPULATION                    SAMPLE
(truth we want to know)  →    (what we actually have)
       ↑                              ↓
       └───── inference ──────────────┘
```

### Population vs Sample Notation

This distinction changes the formulas:

| Measure | Population Symbol | Sample Symbol |
|---|---|---|
| Mean | μ (mu) | x̄ (x-bar) |
| Variance | σ² (sigma squared) | s² |
| Standard deviation | σ (sigma) | s |
| Size | N | n |

**Why different formulas for sample variance?**

When estimating population variance from a sample, dividing by (n−1) instead of n gives an **unbiased estimate**. This is called **Bessel's correction**.

```
Population variance:  σ² = (1/N)   Σ(xᵢ - μ)²
Sample variance:      s² = (1/(n-1)) Σ(xᵢ - x̄)²
                           ↑
                     n-1 instead of n (Bessel's correction)
```

### Key Takeaway

> In ML: your training data is a **sample** from the population. The goal of training is to learn a model that generalises to the **population**. The gap between training performance and population performance is the fundamental problem of generalisation.

---

## 2. Variables

### Types of Variables

Understanding variable types determines which statistical tools to use and which ML algorithms apply.

```
VARIABLES
│
├── QUANTITATIVE (numerical)
│     │
│     ├── Continuous: Can take any value in a range
│     │     Examples: height (175.3 cm), price (₹49,999.99), temperature
│     │
│     └── Discrete: Can only take distinct/countable values
│           Examples: number of children (0, 1, 2, ...), number of clicks
│
└── QUALITATIVE (categorical)
      │
      ├── Nominal: Categories with no natural order
      │     Examples: blood type (A, B, AB, O), city, colour
      │
      └── Ordinal: Categories with a natural order
            Examples: education level (high school < bachelor < master),
                      satisfaction (poor < fair < good < excellent),
                      T-shirt size (S < M < L < XL)
```

### Why Variable Types Matter for ML

| Variable Type | Encoding in ML | Algorithms |
|---|---|---|
| **Continuous** | Use as-is (after scaling) | Linear regression, neural networks |
| **Discrete** | Often treat as continuous or ordinal | Depends on context |
| **Nominal** | One-hot encoding | Most algorithms |
| **Ordinal** | Label encoding (preserves order) | Tree-based algorithms handle well |

---

## 3. Measures of Central Tendency

These answer the question: **"What is the typical/central value of the data?"**

### 3.1 Mean

**What is it?**

The **arithmetic mean** (average) is the sum of all values divided by the count.

```
Formula (population): μ = (1/N) Σᵢ xᵢ

Formula (sample):     x̄ = (1/n) Σᵢ xᵢ

Example:
  Data: [10, 20, 30, 40, 50]
  Mean = (10+20+30+40+50) / 5 = 150 / 5 = 30
```

**Properties:**
- Uses all data points
- Sensitive to outliers (pulled toward extreme values)
- Always exists and is unique

**In ML:**
- Feature normalisation: subtract mean to centre data
- MSE loss: average of squared errors
- Batch normalisation: normalises using batch mean

### 3.2 Median

**What is it?**

The **median** is the middle value when data is sorted. It divides the data into two equal halves.

```
Procedure:
  1. Sort the data
  2. If n is ODD:  median = middle value
  3. If n is EVEN: median = average of the two middle values

Example (odd n):
  Data sorted: [10, 20, 30, 40, 50]
  Median = 30   (3rd of 5)

Example (even n):
  Data sorted: [10, 20, 30, 40]
  Median = (20 + 30) / 2 = 25
```

**Properties:**
- Not affected by outliers — robust
- Better than mean for skewed data

**When to use median over mean:**

```
Scenario: Employee salaries at a startup
  5 engineers:  ₹15L, ₹18L, ₹20L, ₹22L, ₹25L
  1 CEO:        ₹200L

  Mean   = (15+18+20+22+25+200) / 6 = 300 / 6 = ₹50L
  Median = (20 + 22) / 2 = ₹21L

  The median (₹21L) better represents the "typical" employee salary.
  The mean (₹50L) is distorted by the CEO's salary.
```

### 3.3 Mode

**What is it?**

The **mode** is the value that appears most frequently.

```
Examples:
  [1, 2, 2, 3, 3, 3, 4]  → Mode = 3   (unimodal)
  [1, 1, 2, 2, 3]         → Mode = 1, 2 (bimodal)
  [1, 2, 3, 4]            → No mode    (all equal frequency)
```

**In ML:**
- Imputing missing values in categorical columns (replace with most common value)
- Understanding the shape of distributions (bimodal = two clusters?)
- Naive Bayes classifier uses class frequencies (related to mode)

### Mean vs Median vs Mode — Comparison

| | Mean | Median | Mode |
|---|---|---|---|
| **Sensitive to outliers** | Yes | No | No |
| **Works for** | Numeric | Numeric | Any type |
| **Unique?** | Always | Always | Not always |
| **Best for** | Symmetric, no outliers | Skewed data, outliers | Categorical, finding peaks |
| **ML use** | Normalisation, loss | Robust statistics | Imputation |

### Skewness and Central Tendency

The relationship between mean, median, and mode tells you about distribution shape:

```
Symmetric (normal):          Right-skewed:               Left-skewed:
  Mean = Median = Mode         Mode < Median < Mean         Mean < Median < Mode
       │                              │                            │
       ▼                              ▼                            ▼
    ╭─────╮                     ╭──╮                              ╭──╮
   ╭╯     ╰╮                  ╭╯  ╰──────╮               ╭──────╯  ╰╮
  ╭╯       ╰╮                ╭╯          ╰╮             ╭╯           ╰╮
──╯         ╰──           ───╯            ╰──        ───╯             ╰───
                          Income, house prices     Exam scores (easy exam)
```

### Python

```python
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

data = [10, 20, 20, 30, 30, 30, 40, 40, 50, 200]  # 200 is an outlier

arr = np.array(data)

mean   = np.mean(arr)
median = np.median(arr)
mode   = stats.mode(arr, keepdims=True).mode[0]

print(f"Data:   {data}")
print(f"Mean:   {mean:.2f}")    # 47.0  ← pulled by outlier 200
print(f"Median: {median:.2f}")  # 30.0  ← stable
print(f"Mode:   {mode}")        # 30    ← most frequent

# Using pandas — common in data science
s = pd.Series(data)
print(f"\nPandas mean:   {s.mean():.2f}")
print(f"Pandas median: {s.median():.2f}")
print(f"Pandas mode:   {s.mode()[0]}")
print(f"\nPandas describe:\n{s.describe()}")
```

---

## 4. Measures of Spread

These answer: **"How spread out is the data?"**

### 4.1 Range

The simplest measure of spread — the difference between max and min.

```
Range = Maximum − Minimum

Example:  [10, 20, 30, 40, 50]
Range = 50 − 10 = 40
```

**Limitation:** Extremely sensitive to outliers. One extreme value changes the range completely.

### 4.2 Variance

**What is it?**

The **variance** is the average of the squared deviations from the mean. It measures how far individual values are from the mean on average.

```
Population variance: σ² = (1/N)   Σᵢ (xᵢ - μ)²
Sample variance:     s² = (1/(n-1)) Σᵢ (xᵢ - x̄)²

Variables:
  xᵢ    = each data point
  μ, x̄  = the mean
  N, n  = total count
  (xᵢ - μ) = deviation from mean (how far each point is from centre)
  (xᵢ - μ)² = squared deviation (always positive, penalises large deviations)
```

**Step-by-step calculation:**

```
Data: [2, 4, 4, 4, 5, 5, 7, 9]

Step 1: Mean
  x̄ = (2+4+4+4+5+5+7+9) / 8 = 40/8 = 5

Step 2: Deviations from mean (xᵢ − x̄)
  2−5 = −3
  4−5 = −1
  4−5 = −1
  4−5 = −1
  5−5 =  0
  5−5 =  0
  7−5 =  2
  9−5 =  4

Step 3: Squared deviations (xᵢ − x̄)²
  9, 1, 1, 1, 0, 0, 4, 16

Step 4: Average (sample variance uses n−1 = 7)
  s² = (9+1+1+1+0+0+4+16) / 7 = 32/7 ≈ 4.57
```

**Why square the deviations?**
- Without squaring, positive and negative deviations cancel out (sum of deviations = 0 always)
- Squaring makes all deviations positive
- Squaring penalises large deviations more than small ones

**Units problem:** Variance is in squared units (e.g., metres² if data is in metres). This makes it hard to interpret directly. That is why we use standard deviation.

### 4.3 Standard Deviation

**What is it?**

The **standard deviation** is the square root of variance. It is in the **same units as the data**.

```
Population: σ = √σ²
Sample:     s = √s²

From our example:
  s² = 4.57
  s  = √4.57 ≈ 2.14

Interpretation: "On average, values are about 2.14 units away from the mean of 5"
```

**The Empirical Rule (68-95-99.7 Rule)** — for normally distributed data:

```
                     99.7%
        ┌────────────────────────────────────┐
                     95%
          ┌──────────────────────────┐
                   68%
            ┌──────────────┐
            │              │
────────────┼──────────────┼────────────
       μ-3σ  μ-2σ   μ-σ   μ   μ+σ   μ+2σ   μ+3σ

68%  of data falls within μ ± 1σ
95%  of data falls within μ ± 2σ
99.7% of data falls within μ ± 3σ
```

**In ML:**
- Z-score normalisation: z = (x − μ) / σ (scales features to mean=0, std=1)
- Outlier detection: values beyond 3σ are often considered outliers
- Weight initialisation in neural networks: weights sampled from N(0, σ²)
- Batch Normalisation uses mean and std to normalise layer outputs

### Python

```python
import numpy as np

data = np.array([2, 4, 4, 4, 5, 5, 7, 9], dtype=float)

mean     = np.mean(data)
var_pop  = np.var(data, ddof=0)    # Population: divide by N
var_samp = np.var(data, ddof=1)    # Sample: divide by N-1 (Bessel)
std_pop  = np.std(data, ddof=0)
std_samp = np.std(data, ddof=1)

print(f"Mean:               {mean:.4f}")      # 5.0
print(f"Population Variance:{var_pop:.4f}")   # 4.0
print(f"Sample Variance:    {var_samp:.4f}")  # 4.5714
print(f"Population Std Dev: {std_pop:.4f}")   # 2.0
print(f"Sample Std Dev:     {std_samp:.4f}")  # 2.1381

# Z-score normalisation (used in ML preprocessing)
z_scores = (data - mean) / std_pop
print(f"\nZ-scores: {np.round(z_scores, 2)}")
# Values in terms of standard deviations from mean
# z = 0 means exactly at the mean
# z = 2 means 2 standard deviations above the mean

# Verify: normalised data has mean≈0 and std≈1
print(f"Mean of z-scores:  {z_scores.mean():.6f}")   # ~0
print(f"Std of z-scores:   {z_scores.std():.6f}")    # ~1
```

---

## 5. Percentiles, Quartiles, IQR, and Five-Number Summary

### 5.1 Percentiles

**What is it?**

The **p-th percentile** is the value below which p% of the data falls.

```
50th percentile = median (50% of data is below)
90th percentile: 90% of data is below this value
25th percentile: 25% of data is below this value
```

**Example:**

```
You score 85 on an exam.
If 85 is the 90th percentile:
  → You scored higher than 90% of all test-takers.
  → 10% scored higher than you.
```

**In ML:**
- Percentile-based feature engineering
- Understanding score distributions
- Winsorisation (capping at 1st and 99th percentile to reduce outlier impact)

### 5.2 Quartiles

**Quartiles** divide sorted data into four equal parts:

```
Q1 (First Quartile)  = 25th percentile (25% of data below)
Q2 (Second Quartile) = 50th percentile = Median
Q3 (Third Quartile)  = 75th percentile (75% of data below)

Sorted data: [2, 4, 7, 10, 14, 18, 22, 25, 30]
              └──────┘  │  └──────────────────┘
                Q1=5.5  Q2=14            Q3=23.5
```

### 5.3 Interquartile Range (IQR)

```
IQR = Q3 − Q1

It measures the spread of the middle 50% of data.

Example:
  Q1 = 5.5,  Q3 = 23.5
  IQR = 23.5 − 5.5 = 18

Why IQR?
  It is robust to outliers (only uses the middle 50% of data).
  Used for outlier detection and box plots.
```

### 5.4 Five-Number Summary

The **five-number summary** describes any dataset with five values:

```
┌──────────────────────────────────────────────────────────┐
│  Five-Number Summary                                     │
│                                                          │
│  1. Minimum   — smallest value                           │
│  2. Q1        — 25th percentile                         │
│  3. Median    — 50th percentile (Q2)                    │
│  4. Q3        — 75th percentile                         │
│  5. Maximum   — largest value                            │
└──────────────────────────────────────────────────────────┘
```

This is what a **box plot** (box-and-whisker plot) visualises:

```
Box Plot:
               IQR
          ┌────────────┐
     ├────┤     Q2     ├────┤
     │    └────────────┘    │
     ↑                      ↑
  Q1-1.5×IQR           Q3+1.5×IQR
  (lower whisker)       (upper whisker)

  Points beyond the whiskers = OUTLIERS (shown as dots)
```

### Python

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = np.array([2, 4, 7, 10, 14, 18, 22, 25, 30, 100])  # 100 is outlier

# Percentiles
p25 = np.percentile(data, 25)
p50 = np.percentile(data, 50)
p75 = np.percentile(data, 75)
p90 = np.percentile(data, 90)

print(f"Q1  (25th %ile): {p25}")
print(f"Q2  (50th %ile): {p50}  ← Median")
print(f"Q3  (75th %ile): {p75}")
print(f"90th percentile: {p90}")

# IQR
iqr = p75 - p25
print(f"\nIQR = Q3 - Q1 = {p75} - {p25} = {iqr}")

# Five-number summary
print(f"\nFive-Number Summary:")
print(f"  Min:    {data.min()}")
print(f"  Q1:     {p25}")
print(f"  Median: {p50}")
print(f"  Q3:     {p75}")
print(f"  Max:    {data.max()}")

# Using pandas describe
s = pd.Series(data)
print(f"\nPandas describe:\n{s.describe()}")

# Box plot
plt.figure(figsize=(6, 4))
plt.boxplot(data, vert=True, patch_artist=True,
            boxprops=dict(facecolor='lightblue'))
plt.title("Box Plot — Five-Number Summary")
plt.ylabel("Value"); plt.grid(True, alpha=0.3)
plt.savefig('boxplot.png', dpi=100, bbox_inches='tight')
plt.show()
```

---

# PART B — OUTLIERS

---

## 6. What is an Outlier?

### What is it?

An **outlier** is a data point that is **significantly different** from the rest of the dataset — much higher, much lower, or in an unexpected pattern.

```
Normal data:   [22, 24, 23, 25, 22, 24, 23, 26, 25, 24]
With outlier:  [22, 24, 23, 25, 22, 24, 23, 26, 25, 98]
                                                    ↑
                                               This is the outlier
```

### Why Do Outliers Matter?

Outliers can:
- **Distort statistics:** Pull the mean far from the true centre
- **Distort models:** A linear regression line gets dragged toward extreme points
- **Represent genuine anomalies:** A credit card transaction of ₹10 lakhs is either fraud or a legitimate large purchase — both are important

Crucially, **not all outliers should be removed**. Sometimes the outlier IS the signal:

```
Fraud detection:  The unusual transaction IS what you want to detect
Network security: The anomalous packet IS the intrusion
Quality control:  The defective product IS what you want to catch
```

### Types of Outliers

```
OUTLIER TYPES
│
├── Point Outlier (Global)
│     A single value far from all others
│     Example: height of 300 cm in a dataset of human heights
│
├── Contextual Outlier (Conditional)
│     Only unusual in a specific context
│     Example: 35°C is normal in Chennai, outlier in Shimla
│
└── Collective Outlier
      A group of points that are collectively anomalous
      Example: A sequence of 1000 identical transactions (bot behaviour)
```

---

## 7. Outlier Detection Methods

### 7.1 IQR Method

**How it works:**

```
Step 1: Compute Q1, Q3, and IQR
  IQR = Q3 − Q1

Step 2: Compute fence boundaries
  Lower fence = Q1 − 1.5 × IQR
  Upper fence = Q3 + 1.5 × IQR

Step 3: Flag outliers
  Any value < Lower fence  → lower outlier
  Any value > Upper fence  → upper outlier
```

**Example:**

```
Data: [10, 20, 30, 40, 50, 60, 70, 80, 200]

Q1 = 25,  Q3 = 65,  IQR = 40

Lower fence = 25 − 1.5 × 40 = 25 − 60 = −35
Upper fence = 65 + 1.5 × 40 = 65 + 60 = 125

200 > 125 → OUTLIER
All others are within range.
```

**Pros:** Does not assume any distribution, robust to the outliers it is detecting.
**Cons:** Only works well for roughly symmetric distributions.

### 7.2 Z-Score Method

**How it works:**

```
Step 1: Compute Z-score for each value
  Z = (x − μ) / σ

Step 2: Flag values with |Z| > threshold (typically 3)
  |Z| > 3 means the value is more than 3 standard deviations from the mean

From the Empirical Rule:
  99.7% of normal data falls within 3 standard deviations
  Values beyond this are very unusual
```

**Example:**

```
Data: [22, 24, 23, 25, 22, 24, 23, 26, 25, 98]

Mean ≈ 29.2,  Std ≈ 22.6

Z-scores:
  22 → Z = (22 − 29.2) / 22.6 = −0.32   (normal)
  98 → Z = (98 − 29.2) / 22.6 =  3.05   (outlier! |Z| > 3)
```

**Pros:** Simple, widely used, works well for normally distributed data.
**Cons:** Assumes normal distribution. Mean and std are themselves affected by outliers, so the Z-score method is less reliable for heavily contaminated datasets.

### Python: Detecting Outliers

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Dataset with outliers
data = np.array([22, 24, 23, 25, 22, 24, 23, 26, 25, 24,
                 23, 25, 22, 98, 24, -15, 23, 25, 24, 22])

# ── METHOD 1: IQR Method ──────────────────────────────────────────
Q1 = np.percentile(data, 25)
Q3 = np.percentile(data, 75)
IQR = Q3 - Q1

lower_fence = Q1 - 1.5 * IQR
upper_fence = Q3 + 1.5 * IQR

iqr_outliers = data[(data < lower_fence) | (data > upper_fence)]
iqr_clean    = data[(data >= lower_fence) & (data <= upper_fence)]

print("=== IQR Method ===")
print(f"Q1={Q1}, Q3={Q3}, IQR={IQR}")
print(f"Fences: [{lower_fence:.2f}, {upper_fence:.2f}]")
print(f"Outliers detected: {iqr_outliers}")

# ── METHOD 2: Z-Score Method ──────────────────────────────────────
z_scores = np.abs(stats.zscore(data))
threshold = 3.0

z_outlier_mask  = z_scores > threshold
z_outliers      = data[z_outlier_mask]
z_clean         = data[~z_outlier_mask]

print("\n=== Z-Score Method ===")
print(f"Z-scores: {np.round(z_scores, 2)}")
print(f"Outliers detected (|Z| > {threshold}): {z_outliers}")

# ── Visualise ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Box plot (IQR method)
axes[0].boxplot(data, vert=True, patch_artist=True,
                boxprops=dict(facecolor='lightblue', alpha=0.7))
axes[0].set_title("Box Plot (IQR method)")
axes[0].set_ylabel("Value"); axes[0].grid(True, alpha=0.3)

# Z-score plot
axes[1].scatter(range(len(data)), z_scores,
                c=['red' if o else 'blue' for o in z_outlier_mask],
                s=80, alpha=0.7)
axes[1].axhline(y=threshold, color='red', linestyle='--', label=f'Z={threshold}')
axes[1].set_title("Z-Score Method")
axes[1].set_xlabel("Index"); axes[1].set_ylabel("|Z-score|")
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outlier_detection.png', dpi=100, bbox_inches='tight')
plt.show()

print(f"\nData points:    {len(data)}")
print(f"IQR outliers:   {len(iqr_outliers)}")
print(f"Z-score outliers: {len(z_outliers)}")
```

### IQR vs Z-Score Comparison

| Feature | IQR Method | Z-Score Method |
|---|---|---|
| **Distribution assumption** | None | Assumes normality |
| **Sensitivity to outliers** | Low (robust) | High (mean/std affected) |
| **Threshold** | 1.5 × IQR (standard) | |Z| > 3 |
| **Best for** | Any distribution | Normally distributed data |
| **Misses** | Subtle outliers in sparse regions | Outliers in small datasets |

---

## 8. Handling Outliers

### What are the options?

```
OUTLIER HANDLING STRATEGIES

1. REMOVE
   Delete the outlier from the dataset
   When: Data entry error, measurement error, clearly impossible value
   Risk: Losing real information

2. CAP (Winsorisation)
   Replace values beyond threshold with the threshold value
   When: You want to keep the data point but reduce its influence
   Example: Cap all values above 99th percentile at the 99th percentile

3. TRANSFORM
   Apply a mathematical transformation that compresses extreme values
   When: Right-skewed data with legitimate large values
   Example: log(x) — compresses large values, spreads small values

4. IMPUTE
   Replace outlier with mean, median, or model-predicted value
   When: Outlier seems to be a data quality issue

5. KEEP
   Do nothing — use robust algorithms that are not affected
   When: Outlier is a real signal (fraud, anomaly detection)
   Algorithms: Tree-based (Random Forest), robust regression
```

### Python: Handling Outliers

```python
import numpy as np
import pandas as pd

data = np.array([10, 12, 11, 13, 10, 12, 11, 200, 12, 11,
                 10, 13, 11, -50, 12, 10, 11, 13, 12, 11])

print(f"Original: mean={data.mean():.2f}, std={data.std():.2f}")

# ── Strategy 1: Remove outliers (IQR method) ─────────────────────
Q1, Q3 = np.percentile(data, [25, 75])
IQR = Q3 - Q1
mask = (data >= Q1 - 1.5*IQR) & (data <= Q3 + 1.5*IQR)
data_removed = data[mask]
print(f"After removal:  mean={data_removed.mean():.2f}, n={len(data_removed)}")

# ── Strategy 2: Winsorisation (cap at percentiles) ────────────────
p1, p99 = np.percentile(data, [1, 99])
data_capped = np.clip(data, p1, p99)
print(f"After capping:  mean={data_capped.mean():.2f}")

# ── Strategy 3: Log transform (for positive right-skewed data) ────
positive_data = np.array([1, 2, 3, 5, 8, 15, 100, 500, 2000])
log_transformed = np.log1p(positive_data)   # log(1+x), works for 0 too
print(f"\nLog transform:")
print(f"  Original:  {positive_data}")
print(f"  Log1p:     {np.round(log_transformed, 2)}")
# Compresses the range dramatically

# ── Strategy 4: Replace with median ──────────────────────────────
data_imputed = data.copy().astype(float)
Q1, Q3 = np.percentile(data_imputed, [25, 75])
IQR = Q3 - Q1
outlier_mask = (data_imputed < Q1 - 1.5*IQR) | (data_imputed > Q3 + 1.5*IQR)
median_val = np.median(data_imputed[~outlier_mask])
data_imputed[outlier_mask] = median_val
print(f"\nAfter imputation: mean={data_imputed.mean():.2f}")

# ── Using pandas for a DataFrame ─────────────────────────────────
df = pd.DataFrame({'value': data, 'label': range(len(data))})
df['z_score'] = np.abs((df['value'] - df['value'].mean()) / df['value'].std())
df_clean = df[df['z_score'] < 3].drop(columns='z_score')
print(f"\nDataFrame after Z-score cleaning: {len(df_clean)} rows (was {len(df)})")
```

---

# PART C — PROBABILITY

---

## 9. Basic Probability

### What is it?

**Probability** is a number between 0 and 1 that quantifies how likely an event is to occur.

```
P(event) = (Number of favourable outcomes) / (Total number of possible outcomes)

P = 0     →  Impossible (will never happen)
P = 0.5   →  50/50 chance
P = 1     →  Certain (will always happen)
```

### Simple Examples

```
Fair coin flip:
  P(Heads) = 1/2 = 0.5

Rolling a fair die:
  P(rolling a 6) = 1/6 ≈ 0.167
  P(rolling an even) = 3/6 = 0.5

Drawing a card:
  P(drawing an Ace) = 4/52 ≈ 0.077
```

### Axioms of Probability

Three fundamental rules (Kolmogorov axioms):

```
1. Non-negativity:  P(A) ≥ 0  for any event A

2. Normalisation:   P(Ω) = 1  (the whole sample space has probability 1)
                    The probability of something happening is 1.

3. Additivity:      If A and B are mutually exclusive (cannot both happen):
                    P(A ∪ B) = P(A) + P(B)
```

---

## 10. Events and Sample Space

### Definitions

```
EXPERIMENT: A process with uncertain outcomes
  Examples: flipping a coin, rolling a die, observing a patient

SAMPLE SPACE (Ω): The set of ALL possible outcomes
  Coin flip:     Ω = {Heads, Tails}
  Die roll:      Ω = {1, 2, 3, 4, 5, 6}
  Patient:       Ω = {disease, no disease}

EVENT (A, B, C, ...): A subset of the sample space
  A = {rolling an even number} = {2, 4, 6}
  B = {rolling a prime}        = {2, 3, 5}
  C = {rolling > 4}            = {5, 6}
```

### Set Operations for Events

```
UNION:         A ∪ B = A OR B occurs (at least one)
               P(A ∪ B) = P(A) + P(B) − P(A ∩ B)

INTERSECTION:  A ∩ B = A AND B both occur
               P(A ∩ B) = P(A) × P(B|A)

COMPLEMENT:    Aᶜ or Ā = A does NOT occur
               P(Aᶜ) = 1 − P(A)

Addition Rule: P(A ∪ B) = P(A) + P(B) − P(A ∩ B)
  (Subtract intersection to avoid counting it twice)
```

**Example:**

```
Die roll. A = {even} = {2,4,6},  B = {prime} = {2,3,5}

P(A) = 3/6 = 0.5
P(B) = 3/6 = 0.5
A ∩ B = {2}  (even AND prime)
P(A ∩ B) = 1/6

P(A ∪ B) = P(A) + P(B) − P(A ∩ B) = 0.5 + 0.5 − 1/6 = 5/6 ≈ 0.833
```

---

## 11. Conditional Probability

### What is it?

**Conditional probability** is the probability of event A occurring **given that** event B has already occurred.

```
P(A | B) = "Probability of A given B"

Formula:
           P(A ∩ B)
P(A|B) = ──────────
             P(B)

This updates our probability estimate after gaining information.
```

### Intuition

```
Without any information:
  P(a randomly chosen person has diabetes) ≈ 0.11 (11% globally)

With additional information:
  P(person has diabetes | person is over 60 AND overweight) is much higher

Knowing someone is over 60 and overweight changes (conditions) our
probability estimate. That is conditional probability.
```

### Example

```
A bag contains:
  4 red balls, 6 blue balls (10 total)

Without replacement:

P(2nd ball is Red | 1st ball was Red) = ?

After drawing one red ball:
  Remaining: 3 red, 6 blue (9 total)
  P(2nd Red | 1st Red) = 3/9 = 1/3

Compare to:
P(2nd ball is Red | 1st ball was Blue) = 4/9

The condition changes the probability!
```

### In ML

Conditional probability is everywhere in ML:

| ML Context | Conditional Probability |
|---|---|
| **Naive Bayes classifier** | P(Class \| Features) |
| **Language models** | P(next word \| previous words) |
| **Spam filter** | P(spam \| email contains "free money") |
| **Medical AI** | P(disease \| symptoms) |
| **Recommendation** | P(user buys item \| user bought similar item) |

---

## 12. Independence

### What is it?

Two events A and B are **independent** if knowing one occurred gives no information about whether the other occurred.

```
A and B are independent if:
  P(A|B) = P(A)     (knowing B happened doesn't change P(A))
  P(B|A) = P(B)     (knowing A happened doesn't change P(B))

Equivalently:
  P(A ∩ B) = P(A) × P(B)
```

### Examples

**Independent:**

```
Coin 1 = Heads, Coin 2 = Heads
  P(Coin2=H | Coin1=H) = P(Coin2=H) = 0.5
  The first flip does not affect the second.
  P(both H) = 0.5 × 0.5 = 0.25  ✓
```

**Not Independent (Dependent):**

```
Drawing cards WITHOUT replacement:
  P(2nd Ace | 1st was Ace) = 3/51 ≠ 4/52 = P(2nd Ace)
  The first draw affects the second.
```

### In ML: The "Naive" in Naive Bayes

Naive Bayes classifiers assume all features are conditionally independent given the class:

```
P(email is spam | "free", "money", "prize") 
≈ P(spam) × P("free"|spam) × P("money"|spam) × P("prize"|spam)

This "naive" assumption (features are independent given class) is often
wrong in reality, but the classifier still works surprisingly well in practice.
```

---

## 13. Joint Probability

### What is it?

**Joint probability** is the probability of two (or more) events occurring simultaneously.

```
P(A ∩ B) = P(A and B)  =  probability that BOTH A and B occur

For independent events:
  P(A ∩ B) = P(A) × P(B)

For dependent events:
  P(A ∩ B) = P(A) × P(B|A)  =  P(B) × P(A|B)
```

### Joint Probability Table

A powerful way to reason about joint probabilities:

```
                   Spam      Not Spam
Has "free"          0.30        0.05
Does not have       0.70        0.95
─────────────────────────────────────
                (sums to 1)  (sums to 1)

P(spam) = 0.4  (40% of emails are spam — prior)

Joint probabilities:
  P("free" and spam)     = P("free"|spam) × P(spam) = 0.30 × 0.40 = 0.12
  P("free" and not spam) = P("free"|not spam) × P(not spam) = 0.05 × 0.60 = 0.03
```

### Marginal Probability

The probability of one event, summing over all possibilities of the other:

```
P(A) = Σ_B P(A, B)   (sum joint probability over all values of B)

From the spam example:
P("free") = P("free" and spam) + P("free" and not spam)
           = 0.12 + 0.03 = 0.15

15% of all emails contain "free".
```

---

## 14. Bayes Theorem

### What is it?

**Bayes Theorem** provides a mathematical formula for updating our probability estimate (prior belief) in light of new evidence.

### Formula

```
           P(B|A) × P(A)
P(A|B) = ────────────────
               P(B)

Components:
  P(A|B)  = Posterior  — what we want (probability of A given evidence B)
  P(B|A)  = Likelihood — how likely is evidence B if A were true?
  P(A)    = Prior      — our initial belief about A before seeing evidence
  P(B)    = Marginal   — total probability of evidence B (normalising constant)

In words:
  Posterior ∝ Likelihood × Prior
  (Updated belief is proportional to how well the evidence fits times our prior belief)
```

### Intuition

```
Prior belief  +  New evidence  →  Updated belief (posterior)

"I think it will rain today (40% chance)"
  +
"The sky is very dark and cloudy"
  →
"Now I think there is 85% chance of rain"
```

Bayes theorem tells us EXACTLY how much to update our belief.

### Step-by-Step Medical Example

**Scenario:**

```
Disease prevalence:    P(Disease) = 0.01    (1% of population has it)
Test sensitivity:      P(Positive | Disease) = 0.95    (95% accurate when sick)
False positive rate:   P(Positive | No Disease) = 0.05 (5% false positive)

Question: If someone tests positive, what is the probability they actually have the disease?
```

**Solution:**

```
Step 1: Compute P(Positive) using the Law of Total Probability
  P(Pos) = P(Pos|Disease) × P(Disease) + P(Pos|No Disease) × P(No Disease)
         = 0.95 × 0.01 + 0.05 × 0.99
         = 0.0095 + 0.0495
         = 0.059

Step 2: Apply Bayes Theorem
             P(Pos|Disease) × P(Disease)
P(Disease|Pos) = ───────────────────────────
                          P(Pos)

             0.95 × 0.01
           = ────────────
                0.059

           = 0.0095 / 0.059 ≈ 0.161

RESULT: Even with a positive test, only ~16% chance of having the disease!
```

**Why is this surprising?**

Because the disease is RARE (1%). Even though the test is 95% accurate, the large number of healthy people generates many false positives that overwhelm the true positives.

**This is the base rate fallacy** — ignoring the prior probability (prevalence) when interpreting test results. Bayes Theorem prevents this mistake.

### Bayes Theorem in ML

| Application | How Bayes is Used |
|---|---|
| **Naive Bayes classifier** | Direct application — compute P(class\|features) |
| **Bayesian inference** | Update model parameter beliefs as data arrives |
| **Medical diagnosis AI** | P(disease\|symptoms) from prevalence and test accuracy |
| **Spam filtering** | P(spam\|words) — classic application |
| **A/B testing** | Bayesian A/B testing updates belief about which variant is better |
| **RLHF** | Bayesian model of human preferences |

### Python

```python
import numpy as np

# ── Bayes Theorem: Medical Test ───────────────────────────────────
P_disease        = 0.01    # Prior: 1% prevalence
P_no_disease     = 1 - P_disease

P_pos_given_disease    = 0.95    # Sensitivity
P_pos_given_no_disease = 0.05    # False positive rate

# Law of Total Probability: P(Positive)
P_positive = (P_pos_given_disease * P_disease +
              P_pos_given_no_disease * P_no_disease)

# Bayes Theorem: P(Disease | Positive)
P_disease_given_pos = (P_pos_given_disease * P_disease) / P_positive

print("=== Medical Test — Bayes Theorem ===")
print(f"Prior P(Disease):              {P_disease:.2%}")
print(f"P(Positive):                   {P_positive:.4f}")
print(f"Posterior P(Disease|Positive): {P_disease_given_pos:.2%}")
# Only 16.1% chance of disease even with positive test!

# What if prevalence were higher?
print("\nPosterior at different prevalence rates:")
for prevalence in [0.001, 0.01, 0.05, 0.1, 0.3]:
    p_pos = P_pos_given_disease * prevalence + P_pos_given_no_disease * (1 - prevalence)
    posterior = (P_pos_given_disease * prevalence) / p_pos
    print(f"  Prevalence {prevalence:.1%} → P(Disease|Positive) = {posterior:.1%}")

# ── Naive Bayes Text Classifier ───────────────────────────────────
# Simplified: spam vs not spam based on word "FREE"
print("\n=== Naive Bayes Spam Filter ===")

P_spam    = 0.4     # 40% of emails are spam (prior)
P_not_spam = 1 - P_spam

# Likelihood: P(word "free" | class)
P_free_given_spam    = 0.30
P_free_given_notspam = 0.05

# Bayes: P(spam | "free" in email)
P_free = P_free_given_spam * P_spam + P_free_given_notspam * P_not_spam
P_spam_given_free = (P_free_given_spam * P_spam) / P_free

print(f"P(spam):                {P_spam:.2%}")
print(f"P(free in email):       {P_free:.2%}")
print(f"P(spam | 'free'):       {P_spam_given_free:.2%}")
# Updated: email with "free" is 80% likely to be spam
```

---

# PART D — PROBABILITY DISTRIBUTIONS

> A **probability distribution** describes the probability of every possible outcome of a random variable. Understanding distributions is essential because many ML algorithms assume your data or errors follow a specific distribution.

---

## 15. Bernoulli Distribution

### What is it?

The **Bernoulli distribution** models a **single binary trial** — an experiment with exactly two outcomes: success (1) or failure (0).

```
Parameters:
  p = probability of success (0 ≤ p ≤ 1)
  q = 1 − p = probability of failure

P(X = 1) = p
P(X = 0) = 1 − p = q

Mean:     E[X] = p
Variance: Var(X) = p(1−p) = pq
```

### Examples

```
Coin flip:    P(Heads) = 0.5  → Bernoulli(p=0.5)
Email:        P(spam)  = 0.4  → Bernoulli(p=0.4)
Medical test: P(positive | disease) = 0.95 → Bernoulli(p=0.95)
Click:        P(user clicks ad) = 0.02 → Bernoulli(p=0.02)
```

### In ML

- Binary classification output: the sigmoid function models P(y=1|x) as a Bernoulli parameter
- Dropout in neural networks: each neuron is kept with probability p (Bernoulli trial per neuron)
- Logistic regression models the Bernoulli probability

---

## 16. Binomial Distribution

### What is it?

The **Binomial distribution** models the number of successes in **n independent Bernoulli trials**.

```
Parameters:
  n = number of trials
  p = probability of success per trial

P(X = k) = C(n,k) × pᵏ × (1−p)^(n−k)

Where:
  k     = number of successes (0 ≤ k ≤ n)
  C(n,k) = n! / (k!(n-k)!) = "n choose k" (number of ways to get k successes)
  pᵏ    = probability of k successes
  (1-p)^(n-k) = probability of (n-k) failures

Mean:     E[X] = np
Variance: Var(X) = np(1−p)
```

### Example

```
You send 10 marketing emails. Each has a 30% chance of being opened.

n=10, p=0.3

P(exactly 3 opened) = C(10,3) × 0.3³ × 0.7⁷
                    = 120 × 0.027 × 0.0824
                    = 0.2668  (about 27% chance)

Mean: 10 × 0.3 = 3 (expect 3 opens on average)
Std:  √(10 × 0.3 × 0.7) = √2.1 ≈ 1.45
```

### In ML

- Modelling count data (number of positive events out of n trials)
- Cross-validation: number of correctly classified examples out of n

### Python

```python
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

# ── Binomial Distribution ────────────────────────────────────────
n, p = 10, 0.3
binom_dist = stats.binom(n, p)

k_values = np.arange(0, n+1)
pmf = binom_dist.pmf(k_values)

print("=== Binomial(n=10, p=0.3) ===")
for k, prob in zip(k_values, pmf):
    bar = "█" * int(prob * 50)
    print(f"  P(X={k:2d}) = {prob:.4f}  {bar}")

print(f"\nMean:     {binom_dist.mean():.2f}  (np = {n*p})")
print(f"Variance: {binom_dist.var():.2f}  (np(1-p) = {n*p*(1-p)})")
print(f"P(X=3):   {binom_dist.pmf(3):.4f}")
print(f"P(X≤3):   {binom_dist.cdf(3):.4f}")   # Cumulative

# Plot
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(k_values, pmf, color='steelblue', alpha=0.7, edgecolor='black')
ax.set_xlabel("Number of Successes (k)"); ax.set_ylabel("P(X=k)")
ax.set_title("Binomial Distribution (n=10, p=0.3)")
ax.grid(True, alpha=0.3, axis='y')
plt.savefig('binomial.png', dpi=100, bbox_inches='tight')
plt.show()
```

---

## 17. Normal / Gaussian Distribution

### What is it?

The **Normal (Gaussian) distribution** is the most important distribution in statistics and ML. It is the famous "bell curve."

```
Formula (PDF — Probability Density Function):

           1         (x − μ)²
f(x) = ──────── exp(−────────)
        σ√(2π)        2σ²

Parameters:
  μ (mu)    = mean (centre of the bell curve)
  σ (sigma) = standard deviation (width of the bell curve)

Notation: X ~ N(μ, σ²)   ("X is distributed as Normal with mean μ and variance σ²")
```

### Why is the Normal Distribution so Important?

```
1. The Central Limit Theorem (CLT):
   The sum (or mean) of many independent random variables tends toward Normal,
   regardless of their individual distributions.
   → This is WHY normal distributions appear everywhere in nature.

2. Maximum Entropy:
   For a fixed mean and variance, the Normal distribution is the most
   "spread out" (maximum entropy) distribution.
   → It makes the fewest unwarranted assumptions about the data.

3. Mathematical convenience:
   The Normal distribution has many beautiful properties that make
   calculations tractable.
```

### Properties

```
Symmetric around mean μ
Mean = Median = Mode = μ
Empirical Rule:
  68% of data within μ ± 1σ
  95% of data within μ ± 2σ
  99.7% of data within μ ± 3σ
```

### Standard Normal Distribution

A special case: Z ~ N(0, 1) — mean=0, std=1.

Any normal distribution can be converted to standard normal:

```
Z = (X − μ) / σ

This is Z-score normalisation!
```

### In ML

| ML Application | Why Normal is Used |
|---|---|
| **Feature scaling** | Z-normalisation assumes (or creates) roughly normal features |
| **Weight initialisation** | Weights sampled from N(0, σ²) — Xavier, He initialisation |
| **Gaussian noise** | Assume noise is normally distributed |
| **Gaussian Naive Bayes** | Assumes features follow normal distribution within each class |
| **Variational Autoencoders** | Latent space modelled as N(0, I) |
| **Confidence intervals** | Based on normal distribution properties |

### Python

```python
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

# Standard Normal distribution
z_dist = stats.norm(loc=0, scale=1)    # N(0,1)

x = np.linspace(-4, 4, 1000)
pdf = z_dist.pdf(x)   # Probability density
cdf = z_dist.cdf(x)   # Cumulative distribution function

print("=== Standard Normal N(0,1) ===")
print(f"P(Z ≤ 0):    {z_dist.cdf(0):.4f}")   # 0.5 (symmetric)
print(f"P(Z ≤ 1.96): {z_dist.cdf(1.96):.4f}")  # 0.975
print(f"P(-1<Z<1):   {z_dist.cdf(1) - z_dist.cdf(-1):.4f}")   # 0.6827 ≈ 68%
print(f"P(-2<Z<2):   {z_dist.cdf(2) - z_dist.cdf(-2):.4f}")   # 0.9545 ≈ 95%
print(f"P(-3<Z<3):   {z_dist.cdf(3) - z_dist.cdf(-3):.4f}")   # 0.9974 ≈ 99.7%

# Custom normal distribution
custom_dist = stats.norm(loc=170, scale=10)   # Heights: μ=170cm, σ=10cm
print(f"\n=== Height N(170, 10) ===")
print(f"P(height > 190cm): {1 - custom_dist.cdf(190):.4f}")   # Tall people
print(f"P(160 < h < 180):  {custom_dist.cdf(180) - custom_dist.cdf(160):.4f}")

# Plot PDF and CDF
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(x, pdf, 'b-', linewidth=2)
ax1.fill_between(x, pdf, where=(np.abs(x) <= 1), alpha=0.3, label='68% (±1σ)')
ax1.fill_between(x, pdf, where=(np.abs(x) <= 2), alpha=0.2, label='95% (±2σ)')
ax1.set_title("Normal Distribution PDF"); ax1.legend()
ax1.set_xlabel("Z-score"); ax1.set_ylabel("Density")
ax1.grid(True, alpha=0.3)

ax2.plot(x, cdf, 'r-', linewidth=2)
ax2.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
ax2.set_title("Normal Distribution CDF")
ax2.set_xlabel("Z-score"); ax2.set_ylabel("P(Z ≤ z)")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('normal_distribution.png', dpi=100, bbox_inches='tight')
plt.show()

# Generate random samples and check they look normal
samples = stats.norm.rvs(loc=0, scale=1, size=10000, random_state=42)
print(f"\nSampled 10,000 points from N(0,1):")
print(f"  Sample mean: {samples.mean():.4f}  (expected: 0)")
print(f"  Sample std:  {samples.std():.4f}   (expected: 1)")
```

---

## 18. Uniform Distribution

### What is it?

The **Uniform distribution** assigns **equal probability to all values** in a range [a, b].

```
PDF:  f(x) = 1/(b−a)   for a ≤ x ≤ b
              0          otherwise

Parameters:
  a = lower bound
  b = upper bound

Mean:     (a + b) / 2
Variance: (b − a)² / 12
```

### Discrete vs Continuous Uniform

```
Discrete Uniform:  rolling a fair die → P(1)=P(2)=...=P(6)=1/6
Continuous Uniform: U(0,1) → any value in [0,1] is equally likely
```

### In ML

- **Random number generation:** The basis of all random sampling
- **Weight initialisation:** Some methods initialise weights from Uniform(−a, a)
- **Random search:** Uniform sampling of hyperparameter space
- **Probability theory:** Continuous Uniform(0,1) is the "default" distribution for random numbers

### Python

```python
from scipy import stats
import numpy as np

# Continuous Uniform distribution U(0, 1)
u = stats.uniform(loc=0, scale=1)

print("=== Uniform(0, 1) ===")
print(f"P(0.2 ≤ X ≤ 0.8) = {u.cdf(0.8) - u.cdf(0.2):.4f}")  # 0.6
print(f"Mean:     {u.mean():.2f}")
print(f"Variance: {u.var():.4f}")

# Sample 5 uniform random numbers
samples = np.random.uniform(0, 1, size=5)
print(f"\nSamples from U(0,1): {np.round(samples, 4)}")

# Weight initialisation using uniform
n_in, n_out = 64, 32
a = np.sqrt(6 / (n_in + n_out))   # Glorot/Xavier init
weights = np.random.uniform(-a, a, size=(n_out, n_in))
print(f"\nXavier init weights — min: {weights.min():.4f}, max: {weights.max():.4f}")
```

---

## 19. Poisson Distribution

### What is it?

The **Poisson distribution** models the number of **events occurring in a fixed interval of time or space**, given a known average rate.

```
Parameters:
  λ (lambda) = average rate of events per interval

P(X = k) = (λᵏ × e⁻λ) / k!

Where:
  k  = number of events (0, 1, 2, ...)
  λ  = expected number of events
  e  ≈ 2.718 (Euler's number)

Mean:     E[X] = λ
Variance: Var(X) = λ
(Mean equals variance — a defining property of Poisson)
```

### Examples

```
Calls to a call centre: λ = 5 calls/hour
  P(exactly 3 calls in the next hour) = (5³ × e⁻⁵) / 3! = 0.1404

Website visits: λ = 100 visits/hour
  P(≥120 visits in next hour) → use Poisson

Defects in manufacturing: λ = 2 defects/metre of cable
  P(exactly 0 defects in 3 metres) = e⁻⁶ ≈ 0.0025
```

### In ML

- **Count data modelling:** Predicting number of purchases, clicks, or defects
- **Poisson regression:** A type of regression for count data
- **Anomaly detection:** Is the observed event count unusual given the expected rate?
- **NLP:** Number of times a word appears in a document

### Python

```python
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

lam = 5   # Average 5 calls per hour

poisson_dist = stats.poisson(mu=lam)
k_vals = np.arange(0, 16)

print(f"=== Poisson(λ={lam}) ===")
print(f"P(X=0):  {poisson_dist.pmf(0):.4f}")   # No calls
print(f"P(X=5):  {poisson_dist.pmf(5):.4f}")   # Exactly average
print(f"P(X=10): {poisson_dist.pmf(10):.4f}")  # Double average
print(f"P(X≤5):  {poisson_dist.cdf(5):.4f}")   # Up to 5 calls
print(f"Mean:    {poisson_dist.mean()}")
print(f"Variance:{poisson_dist.var()}")   # Mean = Variance = λ

# Compare different λ values
fig, ax = plt.subplots(figsize=(8, 4))
for l, colour in [(2,'blue'), (5,'green'), (10,'red')]:
    ax.plot(k_vals, stats.poisson(mu=l).pmf(k_vals), 'o-',
            label=f'λ={l}', color=colour, linewidth=2)
ax.set_xlabel("k"); ax.set_ylabel("P(X=k)")
ax.set_title("Poisson Distribution for different λ values")
ax.legend(); ax.grid(True, alpha=0.3)
plt.savefig('poisson.png', dpi=100, bbox_inches='tight')
plt.show()
```

---

## 20. Exponential Distribution

### What is it?

The **Exponential distribution** models the **time between events** in a Poisson process — i.e., how long you wait for the next event.

```
Parameters:
  λ = rate parameter (events per unit time)
    = 1/μ where μ is the mean waiting time

PDF:  f(x) = λe^(−λx)  for x ≥ 0

Mean:     E[X] = 1/λ
Variance: Var(X) = 1/λ²
```

**Relationship to Poisson:**

```
Poisson: "How MANY events occur in a time interval?"
Exponential: "How LONG until the NEXT event?"

If events arrive at rate λ per hour (Poisson):
  → Waiting time between events follows Exponential(λ)
```

### Examples

```
Customer service: calls arrive at λ=5/hour
  Waiting time between calls ~ Exponential(5)
  Mean wait = 1/5 hour = 12 minutes

Machine lifetime: machine breaks at rate λ=1/1000 per hour
  Lifetime ~ Exponential(1/1000)
  Mean lifetime = 1000 hours

Internet packet arrival: λ=1000 packets/second
  Time between packets ~ Exponential(1000)
```

### Memoryless Property

The Exponential distribution has a unique "memoryless" property:

```
P(X > s + t | X > s) = P(X > t)

"If you have already waited s minutes, the probability of waiting t more minutes
is the same as if you had just started waiting."

Like a forgetful clock: it has no memory of how long you've already waited.
```

### In ML

- **Survival analysis:** Predicting time until an event (equipment failure, user churn)
- **Recommender systems:** Modelling time between user interactions
- **Anomaly detection:** Unusual gaps between events in a time series

### Python

```python
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

# Exponential distribution: λ=1/mean, so for mean=5 minutes: λ=0.2
lam = 0.2   # rate: 0.2 events per minute = 1 event per 5 minutes
exp_dist = stats.expon(scale=1/lam)   # scale = 1/λ = mean

print(f"=== Exponential(λ={lam}) ===")
print(f"Mean waiting time:  {exp_dist.mean():.2f} minutes")
print(f"P(wait < 3 min):    {exp_dist.cdf(3):.4f}")
print(f"P(wait > 10 min):   {1 - exp_dist.cdf(10):.4f}")

# Memoryless property demonstration
# P(X > 10 | X > 7) = P(X > 3)
p_over_10_given_over_7 = (1 - exp_dist.cdf(10)) / (1 - exp_dist.cdf(7))
p_over_3 = 1 - exp_dist.cdf(3)
print(f"\nMemoryless property:")
print(f"P(X>10 | X>7) = {p_over_10_given_over_7:.4f}")
print(f"P(X>3)        = {p_over_3:.4f}")
print(f"Equal? {np.isclose(p_over_10_given_over_7, p_over_3)}")  # True

x = np.linspace(0, 30, 300)
plt.figure(figsize=(8, 4))
for l, label in [(0.5, 'λ=0.5 (mean=2)'), (0.2, 'λ=0.2 (mean=5)'), (0.1, 'λ=0.1 (mean=10)')]:
    plt.plot(x, stats.expon(scale=1/l).pdf(x), linewidth=2, label=label)
plt.xlabel("Time"); plt.ylabel("Density")
plt.title("Exponential Distribution — Time Until Next Event")
plt.legend(); plt.grid(True, alpha=0.3)
plt.savefig('exponential.png', dpi=100, bbox_inches='tight')
plt.show()
```

### Distribution Summary Table

| Distribution | Models | Parameters | Mean | Variance |
|---|---|---|---|---|
| **Bernoulli** | Single binary trial | p | p | p(1-p) |
| **Binomial** | Count of successes in n trials | n, p | np | np(1-p) |
| **Normal** | Continuous, symmetric data | μ, σ² | μ | σ² |
| **Uniform** | Equal likelihood over range | a, b | (a+b)/2 | (b-a)²/12 |
| **Poisson** | Count of events in interval | λ | λ | λ |
| **Exponential** | Time between events | λ | 1/λ | 1/λ² |

---

# PART E — STATISTICAL CONCEPTS

---

## 21. Expectation

### What is it?

The **expected value** (or **expectation**) E[X] is the **theoretical average** of a random variable over infinite repeated trials — the long-run average you would get if you repeated the experiment many times.

```
Discrete: E[X] = Σₓ x × P(X = x)

Continuous: E[X] = ∫ x × f(x) dx   (integral instead of sum)

Variables:
  x    = each possible value
  P(X=x) = probability of that value
  f(x)   = probability density function
```

### Example: Fair Die

```
Values:         1,   2,   3,   4,   5,   6
Probabilities: 1/6, 1/6, 1/6, 1/6, 1/6, 1/6

E[X] = 1×(1/6) + 2×(1/6) + 3×(1/6) + 4×(1/6) + 5×(1/6) + 6×(1/6)
     = (1+2+3+4+5+6)/6 = 21/6 = 3.5

We never roll 3.5, but over many rolls, the average converges to 3.5.
```

### Properties

```
E[aX + b] = a × E[X] + b              (linearity)
E[X + Y]  = E[X] + E[Y]              (always true)
E[X × Y]  = E[X] × E[Y]              (only if X and Y are independent)
E[constant] = constant
```

### In ML

| Context | Expectation Means |
|---|---|
| **Loss function** | E[L(ŷ, y)] = expected loss over all possible data points |
| **Risk** | Expected loss over true data distribution — what we actually minimise |
| **RL policy** | E[cumulative reward] — expected total reward |
| **Mean of distribution** | E[X] = μ |

---

## 22. Covariance

### What is it?

**Covariance** measures how much two variables **change together**.

```
Formula:
  Cov(X, Y) = E[(X - μₓ)(Y - μᵧ)]
             = (1/(n-1)) Σᵢ (xᵢ - x̄)(yᵢ - ȳ)

Interpretation:
  Cov(X,Y) > 0  → X and Y increase together (positive relationship)
  Cov(X,Y) < 0  → X increases when Y decreases (negative relationship)
  Cov(X,Y) = 0  → X and Y are unrelated (no LINEAR relationship)
```

### Problem with Covariance

Covariance depends on the **scale** of the variables:

```
Height and Weight:
  If height in cm and weight in kg:   Cov = 250
  If height in metres and weight in kg: Cov = 2.5

The covariance changes with units, making it hard to interpret.
Solution: Use correlation (normalised covariance).
```

### Covariance Matrix

For multiple variables, the **covariance matrix** Σ captures all pairwise covariances:

```
For 3 features [X₁, X₂, X₃]:

        X₁           X₂           X₃
Σ = [Var(X₁)     Cov(X₁,X₂)  Cov(X₁,X₃)]
    [Cov(X₂,X₁)  Var(X₂)     Cov(X₂,X₃)]
    [Cov(X₃,X₁)  Cov(X₃,X₂)  Var(X₃)   ]

Properties:
  - Square: n_features × n_features
  - Symmetric: Cov(Xᵢ,Xⱼ) = Cov(Xⱼ,Xᵢ)
  - Diagonal entries are variances: Σᵢᵢ = Var(Xᵢ)
```

**In ML:** PCA finds the eigenvectors of the covariance matrix.

---

## 23. Correlation

### What is it?

**Correlation** is **normalised covariance** — it measures the strength and direction of the linear relationship between two variables, on a scale from −1 to +1.

```
Pearson Correlation Coefficient:

           Cov(X, Y)
r = ─────────────────────
        σₓ × σᵧ

Where:
  σₓ = standard deviation of X
  σᵧ = standard deviation of Y

Range: −1 ≤ r ≤ +1

r = +1   →  Perfect positive linear relationship
r = 0    →  No linear relationship
r = −1   →  Perfect negative linear relationship
```

### Interpreting Correlation

```
|r| interpretation:
  0.9–1.0   Very strong
  0.7–0.9   Strong
  0.5–0.7   Moderate
  0.3–0.5   Weak
  0.0–0.3   Negligible

Examples:
  Height & weight:    r ≈ +0.7  (strong positive)
  Temperature & coat: r ≈ −0.8  (strong negative)
  Shoe size & IQ:     r ≈ 0     (no relationship)
```

### ⚠️ Correlation ≠ Causation

This is the most important statistical warning in data science:

```
Example:
  Ice cream sales and drowning rates are highly correlated.
  r ≈ +0.9

  Does ice cream cause drowning? NO.
  The confounder (hidden cause) is: HOT WEATHER
    → Hot weather → people buy more ice cream
    → Hot weather → people swim more → more drowning

Never conclude that correlation implies causation.
Always think about confounders and mechanisms.
```

### In ML

- **Feature selection:** Highly correlated features may be redundant — remove one
- **Multicollinearity:** High correlation between features hurts linear models
- **EDA:** Correlation heatmaps show which features relate to the target
- **Feature engineering:** Create interaction terms between correlated features

### Python

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

# Generate correlated data
n = 200
x1 = np.random.randn(n)
x2 = 0.8 * x1 + 0.2 * np.random.randn(n)   # Positively correlated with x1
x3 = -0.6 * x1 + 0.4 * np.random.randn(n)  # Negatively correlated with x1
x4 = np.random.randn(n)                      # Independent

df = pd.DataFrame({'X1': x1, 'X2': x2, 'X3': x3, 'X4': x4})

# ── Covariance Matrix ─────────────────────────────────────────────
cov_matrix = df.cov()
print("Covariance Matrix:")
print(cov_matrix.round(3))

# ── Correlation Matrix ────────────────────────────────────────────
corr_matrix = df.corr()
print("\nCorrelation Matrix:")
print(corr_matrix.round(3))

# Individual correlation
r_12 = df['X1'].corr(df['X2'])
print(f"\nPearson r(X1, X2) = {r_12:.4f}")   # ~0.97 (strong positive)

# ── Correlation Heatmap (used in EDA) ────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Scatter plot
axes[0].scatter(x1, x2, alpha=0.5, s=20, color='steelblue')
axes[0].set_xlabel("X1"); axes[0].set_ylabel("X2")
axes[0].set_title(f"X1 vs X2 (r = {r_12:.3f})")
axes[0].grid(True, alpha=0.3)

# Heatmap
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, ax=axes[1], linewidths=0.5)
axes[1].set_title("Correlation Heatmap")

plt.tight_layout()
plt.savefig('correlation.png', dpi=100, bbox_inches='tight')
plt.show()

# Correlation vs Causation demo
# Ice cream and drowning (both caused by hot weather)
temperature = np.random.uniform(20, 40, n)    # confounder
ice_cream = 0.8 * temperature + 2 * np.random.randn(n)
drowning  = 0.7 * temperature + 2 * np.random.randn(n)

r_spurious = np.corrcoef(ice_cream, drowning)[0, 1]
print(f"\nSpurious correlation (ice cream vs drowning): r = {r_spurious:.3f}")
print("Cause: both driven by temperature (confounder)!")
```

---

## 24. Central Limit Theorem

### What is it?

The **Central Limit Theorem (CLT)** is one of the most important and profound theorems in all of statistics.

```
CENTRAL LIMIT THEOREM:

If you take repeated samples of size n from ANY distribution
(with finite mean μ and finite variance σ²),

then the distribution of SAMPLE MEANS approaches a NORMAL DISTRIBUTION
as n gets large, regardless of the shape of the original distribution.

Specifically:
  x̄ ~ N(μ, σ²/n)   approximately, for large enough n

Where:
  x̄  = sample mean
  μ  = population mean
  σ² = population variance
  n  = sample size
```

### Intuition

```
Original distribution: could be anything
  Uniform, Skewed, Bimodal, Bizarre — doesn't matter

Take 1000 samples of size n=30 each.
Compute the mean of each sample.
Plot those 1000 means.

RESULT: The means form a bell curve (Normal distribution)!

The larger n is, the closer to Normal the distribution of means becomes.
```

### Why Does the CLT Matter?

```
1. JUSTIFIES STATISTICAL TESTS:
   Most hypothesis tests (t-test, z-test) assume normality.
   The CLT says: even if your data isn't normal, 
   the sampling distribution of the mean IS approximately normal.
   → Makes these tests valid for large samples, ANY distribution.

2. JUSTIFIES ML CONFIDENCE INTERVALS:
   Error bars and confidence intervals for model performance
   are built on the CLT.

3. EXPLAINS WHY NORMAL IS EVERYWHERE:
   Many natural phenomena are sums of many small effects.
   By CLT, these sums tend toward normal.
   Height = genetics + nutrition + ... (sum of many factors) → Normal

4. EXPLAINS AVERAGING BEHAVIOUR:
   Large batches in mini-batch GD average out noise — CLT.
   Ensemble methods average predictions — CLT justifies stability.
```

### Python: Demonstrating the CLT

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

# Original population: UNIFORM distribution (not normal at all)
population = np.random.uniform(0, 1, size=100000)

fig, axes = plt.subplots(2, 3, figsize=(14, 8))

# Show the original distribution
axes[0, 0].hist(population, bins=50, color='gray', alpha=0.7, density=True)
axes[0, 0].set_title("Original: Uniform(0,1)\n(NOT normal)")
axes[0, 0].set_xlabel("Value"); axes[0, 0].set_ylabel("Density")

# Show sample means for different sample sizes
sample_sizes = [2, 5, 10, 30, 100]
colours = ['red', 'orange', 'green', 'blue', 'purple']

for i, (n, col) in enumerate(zip(sample_sizes, colours)):
    # Take 5000 samples of size n, compute mean of each
    sample_means = [np.mean(np.random.choice(population, size=n))
                    for _ in range(5000)]

    ax = axes[(i+1) // 3, (i+1) % 3]
    ax.hist(sample_means, bins=50, color=col, alpha=0.7, density=True,
            label=f'Simulated')

    # Overlay the theoretical Normal curve
    mu_theory = 0.5           # E[Uniform(0,1)] = 0.5
    sigma_theory = np.sqrt(1/12 / n)  # Var(Uniform) = 1/12, so std of mean = sqrt(Var/n)
    x_range = np.linspace(min(sample_means), max(sample_means), 200)
    ax.plot(x_range, stats.norm.pdf(x_range, mu_theory, sigma_theory),
            'k-', linewidth=2, label='Normal (theory)')

    ax.set_title(f"Sample Means (n={n})")
    ax.set_xlabel("Sample Mean"); ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle("Central Limit Theorem: Uniform → Normal as n increases",
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('central_limit_theorem.png', dpi=100, bbox_inches='tight')
plt.show()

print("CLT demonstration:")
for n in [2, 5, 10, 30, 100]:
    sample_means = [np.mean(np.random.choice(population, size=n)) for _ in range(5000)]
    empirical_std = np.std(sample_means)
    theoretical_std = np.sqrt(1/12 / n)
    print(f"  n={n:3d}: empirical std = {empirical_std:.4f}, "
          f"theoretical = {theoretical_std:.4f}, "
          f"shapiro-W = {stats.shapiro(sample_means[:50]).statistic:.4f}")
```

---

## 25. Sampling Distributions

### What is it?

A **sampling distribution** is the probability distribution of a **statistic** (like the sample mean) computed from many random samples drawn from the same population.

```
POPULATION:  All heights of adults in India
             True mean: μ = 165 cm (unknown to us)
             True std:  σ = 10 cm

SAMPLE 1 (n=50): mean = 163.2 cm
SAMPLE 2 (n=50): mean = 165.8 cm
SAMPLE 3 (n=50): mean = 164.3 cm
...
SAMPLE 1000 (n=50): mean = 166.1 cm

The distribution of these 1000 sample means is the SAMPLING DISTRIBUTION of the mean.

By CLT: this distribution is approximately Normal(μ, σ²/n) = Normal(165, 2)
```

### Standard Error

The **standard error (SE)** is the standard deviation of the sampling distribution:

```
SE = σ / √n

Where:
  σ = population standard deviation
  n = sample size

Interpretation: How much variability to expect in sample means.

If n=50, σ=10:
  SE = 10 / √50 = 10 / 7.07 ≈ 1.41

Your sample mean will typically be within 1.41 cm of the true mean.
```

### Key Insight: SE Decreases with Sample Size

```
n = 10:  SE = 10/√10  = 3.16 (high uncertainty)
n = 100: SE = 10/√100 = 1.00 (lower uncertainty)
n = 1000:SE = 10/√1000 = 0.32 (much more certain)

Quadruple the sample size → halve the standard error.
```

### Confidence Intervals

A **confidence interval** uses the sampling distribution to give a range of plausible values for the population parameter:

```
95% Confidence Interval for the population mean:
  x̄ ± 1.96 × SE  =  x̄ ± 1.96 × (σ/√n)

"We are 95% confident the true population mean falls in this range."

Example:
  Sample mean x̄ = 163.5, SE = 1.41
  95% CI = 163.5 ± 1.96 × 1.41 = [160.7, 166.3]
```

### Why Sampling Distributions Matter for ML

```
1. TRAIN/TEST SPLIT:
   Your test set is a sample. The test accuracy is an estimate of
   true population accuracy. The sampling distribution tells you
   how much this estimate varies.

2. CROSS-VALIDATION:
   K-fold CV gives k estimates of performance.
   The mean and variance of these estimates characterise the
   sampling distribution of your model's performance.

3. BOOTSTRAP:
   A resampling technique that approximates the sampling distribution
   by repeatedly sampling from your dataset with replacement.
   Used for confidence intervals when the theoretical distribution is unknown.

4. HYPOTHESIS TESTING:
   Statistical tests about model performance (is model A better than B?)
   use the sampling distribution to compute p-values.
```

### Python

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

# True population: not normal (exponential)
lam = 1/5    # rate: average = 5
population = stats.expon.rvs(scale=1/lam, size=100000)

true_mean = 1 / lam   # = 5
print(f"True population mean: {true_mean}")
print(f"True population std:  {population.std():.4f}")

# ── Sampling Distribution of the Mean ────────────────────────────
n = 40        # sample size
n_samples = 5000

sample_means = [np.mean(np.random.choice(population, size=n))
                for _ in range(n_samples)]

empirical_se = np.std(sample_means)
theoretical_se = population.std() / np.sqrt(n)

print(f"\nSampling distribution of mean (n={n}):")
print(f"  Empirical SE:    {empirical_se:.4f}")
print(f"  Theoretical SE:  {theoretical_se:.4f}")   # Should match

# ── Confidence Intervals ──────────────────────────────────────────
single_sample = np.random.choice(population, size=n)
x_bar = np.mean(single_sample)
se_est = single_sample.std(ddof=1) / np.sqrt(n)

# 95% confidence interval (using t-distribution for small n)
t_critical = stats.t.ppf(0.975, df=n-1)
ci_lower = x_bar - t_critical * se_est
ci_upper = x_bar + t_critical * se_est

print(f"\nSingle sample (n={n}): mean = {x_bar:.4f}")
print(f"Standard error: {se_est:.4f}")
print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
print(f"True mean {true_mean} is inside CI: {ci_lower <= true_mean <= ci_upper}")

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Original population
ax1.hist(population[:1000], bins=50, color='orange', alpha=0.7, density=True)
ax1.set_title("Population Distribution\n(Exponential — not normal)")
ax1.set_xlabel("Value"); ax1.set_ylabel("Density"); ax1.grid(True, alpha=0.3)

# Sampling distribution
ax2.hist(sample_means, bins=50, color='blue', alpha=0.7, density=True)
x_range = np.linspace(min(sample_means), max(sample_means), 200)
ax2.plot(x_range, stats.norm.pdf(x_range, true_mean, theoretical_se),
         'r-', linewidth=2, label='Normal (CLT theory)')
ax2.axvline(true_mean, color='black', linestyle='--', label=f'True mean={true_mean}')
ax2.set_title(f"Sampling Distribution of Mean (n={n})\n→ Approaches Normal by CLT")
ax2.set_xlabel("Sample Mean"); ax2.legend(); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('sampling_distribution.png', dpi=100, bbox_inches='tight')
plt.show()
```

---

## 🔬 Practical Section: Full EDA Pipeline

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load a real dataset
from sklearn.datasets import load_diabetes

data = load_diabetes()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target   # Disease progression measure

print("=== EXPLORATORY DATA ANALYSIS ===")
print(f"\nShape: {df.shape}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nDescriptive stats:\n{df.describe().round(3)}")

# ── Central Tendency and Spread ───────────────────────────────────
target = df['target']
print(f"\n=== Target Variable Analysis ===")
print(f"Mean:    {target.mean():.2f}")
print(f"Median:  {target.median():.2f}")
print(f"Std:     {target.std():.2f}")
print(f"Min:     {target.min():.2f}")
print(f"Max:     {target.max():.2f}")
print(f"Q1:      {target.quantile(0.25):.2f}")
print(f"Q3:      {target.quantile(0.75):.2f}")
print(f"IQR:     {target.quantile(0.75) - target.quantile(0.25):.2f}")
print(f"Skewness:{target.skew():.4f}")   # >0 = right-skewed

# ── Outlier Detection ─────────────────────────────────────────────
Q1 = target.quantile(0.25)
Q3 = target.quantile(0.75)
IQR = Q3 - Q1
n_iqr_outliers = ((target < Q1 - 1.5*IQR) | (target > Q3 + 1.5*IQR)).sum()

z_scores = np.abs(stats.zscore(target))
n_z_outliers = (z_scores > 3).sum()

print(f"\n=== Outlier Detection ===")
print(f"IQR method outliers:    {n_iqr_outliers}")
print(f"Z-score method outliers: {n_z_outliers}")

# ── Correlation with Target ───────────────────────────────────────
corr_with_target = df.corr()['target'].sort_values(ascending=False)
print(f"\n=== Correlation with Target ===\n{corr_with_target.round(3)}")

# ── Visualisation ─────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Distribution of target
axes[0,0].hist(target, bins=30, color='steelblue', alpha=0.7, edgecolor='white')
axes[0,0].set_title("Target Distribution"); axes[0,0].grid(True, alpha=0.3)

# 2. Box plot
axes[0,1].boxplot(target, vert=True, patch_artist=True,
                  boxprops=dict(facecolor='lightcoral', alpha=0.7))
axes[0,1].set_title("Box Plot (Target)"); axes[0,1].grid(True, alpha=0.3)

# 3. Correlation heatmap
sns.heatmap(df.corr(), annot=False, cmap='coolwarm', center=0, ax=axes[1,0])
axes[1,0].set_title("Correlation Heatmap")

# 4. Scatter plot: bmi vs target
axes[1,1].scatter(df['bmi'], target, alpha=0.4, s=20, color='purple')
axes[1,1].set_xlabel("BMI"); axes[1,1].set_ylabel("Disease Progression")
axes[1,1].set_title(f"BMI vs Target (r={df['bmi'].corr(target):.3f})")
axes[1,1].grid(True, alpha=0.3)

plt.suptitle("Exploratory Data Analysis — Diabetes Dataset", fontsize=14)
plt.tight_layout()
plt.savefig('eda_pipeline.png', dpi=100, bbox_inches='tight')
plt.show()
```

---

## 🔄 Concept Connections

```
This unit connects to:

PREVIOUS:
  Unit 1: Why we need statistics in AI/ML (overview)
  Unit 2: Linear algebra (covariance matrix = linear algebra)

FORWARD:
  Unit 4 (Data Science):          EDA uses all tools from this unit
  Unit 5 (EDA):                   Practical application of descriptive stats
  Unit 8 (Machine Learning):      Loss functions use expectation
  Unit 9 (Regression):            Correlation motivates regression
  Unit 10 (Classification):       Probabilities are classification outputs
  Unit 13 (Naive Bayes):          Direct application of Bayes Theorem
  Unit 15 (Dimensionality):       PCA uses covariance/eigenvalues
  Unit 16 (Neural Networks):      Distributions for weight initialisation
  Unit 31 (RL):                   Expected reward uses expectation
  Unit 32 (RLHF):                 Bayesian models of human preference
```

---

## 🎓 Exam Notes

**Key formulas to memorise:**

| Formula | Name |
|---|---|
| x̄ = (1/n) Σxᵢ | Sample Mean |
| s² = (1/(n-1)) Σ(xᵢ−x̄)² | Sample Variance |
| s = √s² | Sample Standard Deviation |
| IQR = Q3 − Q1 | Interquartile Range |
| Z = (x−μ)/σ | Z-score |
| P(A\|B) = P(A∩B)/P(B) | Conditional Probability |
| P(A∩B) = P(A)×P(B) | Joint Prob. (independent) |
| P(A\|B) = P(B\|A)×P(A)/P(B) | Bayes Theorem |
| E[X] = Σ x P(X=x) | Expectation |
| Cov(X,Y) = E[(X−μₓ)(Y−μᵧ)] | Covariance |
| r = Cov(X,Y)/(σₓσᵧ) | Pearson Correlation |
| SE = σ/√n | Standard Error |
| x̄ ~ N(μ, σ²/n) | Sampling Distribution (CLT) |

**Key definitions:**
1. **Population** — complete set of all items of interest
2. **Sample** — subset of population
3. **Bernoulli** — single binary trial; P(1)=p, P(0)=1-p
4. **Normal distribution** — bell curve; defined by mean and variance
5. **CLT** — sum/mean of n i.i.d. variables → Normal as n→∞
6. **Standard Error** — standard deviation of the sampling distribution
7. **Confidence interval** — range of plausible values for population parameter
8. **Correlation** — normalised covariance, measures linear relationship (−1 to +1)

---

## 💼 Interview Notes

**Q1: What is the difference between population and sample? Why does this matter for ML?**

> The population is the complete set of all items you care about; the sample is the subset you actually observe. In ML, your training set is a sample from the population of all possible inputs. The goal of training is to build a model that generalises to the population, not just the sample. This distinction explains why overfitting (memorising the training sample) is harmful — you need to generalise to the population.

---

**Q2: When should you use median instead of mean?**

> Use the median when your data is skewed or contains outliers. The mean is sensitive to extreme values — one very high salary can pull the mean far from the "typical" value. The median only uses the middle value(s) and is resistant to outliers. Examples: house prices, incomes, response times.

---

**Q3: What is the IQR and how is it used for outlier detection?**

> IQR = Q3 − Q1 (the range of the middle 50% of data). For outlier detection: compute lower fence = Q1 − 1.5×IQR and upper fence = Q3 + 1.5×IQR. Any value outside these fences is flagged as an outlier. The method is robust because it uses the quartiles, which are themselves not affected by extreme values.

---

**Q4: State Bayes Theorem and explain each component.**

> P(A|B) = P(B|A) × P(A) / P(B). P(A) is the prior — our initial belief before seeing evidence. P(B|A) is the likelihood — how probable the evidence is if A is true. P(A|B) is the posterior — updated belief after seeing evidence B. P(B) is the marginal probability of evidence — a normalising constant ensuring the posterior sums to 1.

---

**Q5: What is the Central Limit Theorem and why is it important for ML?**

> The CLT states that the distribution of sample means approaches Normal with mean μ and standard deviation σ/√n as sample size n increases, regardless of the shape of the original distribution. It is important for ML because: (1) it justifies using Normal-based statistical tests on non-normal data, (2) it explains why confidence intervals and hypothesis tests work, (3) it explains why averaging predictions in ensemble methods reduces variance, and (4) it explains why mini-batch gradient descent works well despite using random subsets of data.

---

**Q6: What is the difference between covariance and correlation?**

> Covariance measures how two variables change together, but its magnitude depends on the units of the variables, making it hard to interpret. Correlation normalises covariance by the product of the two standard deviations, giving a unitless number always between −1 and +1. Correlation is interpretable: +1 = perfect positive, 0 = no linear relationship, −1 = perfect negative.

---

**Q7: What does it mean for two events to be independent?**

> Events A and B are independent if knowing one provides no information about the other: P(A|B) = P(A), equivalently P(A∩B) = P(A)×P(B). Coin flips are independent. In contrast, drawing cards without replacement is not independent. The "Naive" in Naive Bayes means it assumes features are conditionally independent given the class — a simplification that often works well in practice despite being rarely true.

---

## ⚠️ Common Mistakes & Misconceptions

**❌ Mistake 1: Using n instead of n-1 for sample variance**
✅ Sample variance uses n-1 (Bessel's correction) to produce an unbiased estimate of the population variance. In NumPy: `np.var(data, ddof=1)` for sample variance, `ddof=0` (default) for population variance.

---

**❌ Mistake 2: Removing all outliers without investigating them**
✅ Always investigate outliers before removing them. Outliers can be: data entry errors (remove), legitimate extreme values (keep), or the actual signal you are trying to detect (fraud, anomalies — definitely keep). Blind removal destroys real information.

---

**❌ Mistake 3: Confusing correlation with causation**
✅ Correlation measures statistical association, not causal relationships. A high correlation between X and Y could mean X causes Y, Y causes X, or a third variable Z causes both. Always think about mechanisms and confounders.

---

**❌ Mistake 4: Thinking the CLT says your data must be normal**
✅ The CLT says the distribution of SAMPLE MEANS becomes normal. Your underlying data can be any shape. The CLT applies to statistics computed FROM samples, not the raw data itself.

---

**❌ Mistake 5: Interpreting a 95% CI as "95% probability the parameter is inside"**
✅ A 95% CI means: if you repeated sampling many times and computed a CI each time, 95% of those CIs would contain the true parameter. The parameter is fixed (not random) — the CI is what varies across samples.

---

**❌ Mistake 6: Ignoring the prior in Bayes Theorem (base rate fallacy)**
✅ A highly accurate test for a rare disease can still give mostly false positives if the prior (prevalence) is low. This is exactly what our medical example showed: 95% accurate test + 1% prevalence = only 16% posterior probability of disease given positive test.

---

**❌ Mistake 7: Assuming all data is normally distributed**
✅ Many real-world distributions are skewed, multimodal, or heavy-tailed. Always visualise your data before assuming any distribution. Use Q-Q plots (quantile-quantile plots) to check normality.

---

## 🧠 Key Takeaways

1. **Population** = everyone/everything you care about. **Sample** = what you actually have. ML models trained on samples must generalise to populations.

2. **Mean** is sensitive to outliers; **median** is robust. For skewed data (salaries, house prices), prefer median. For symmetric data, both work.

3. **Standard deviation** measures spread in original units. **Variance** = σ². **Z-score** = (x−μ)/σ normalises data to mean=0, std=1.

4. **IQR** (Q3−Q1) captures the middle 50% of data. Fence method (±1.5×IQR) is a robust way to detect outliers.

5. **Conditional probability** P(A|B) = P(A∩B)/P(B) updates probability after observing evidence. The foundation of Bayes and all probabilistic ML.

6. **Bayes Theorem**: Posterior ∝ Likelihood × Prior. It tells you how to update your beliefs with new evidence. Foundation of Naive Bayes, Bayesian inference, and RLHF.

7. **Normal distribution** is the most important distribution in ML — weight initialisation, noise models, confidence intervals, many assumptions in statistical tests.

8. **Covariance** measures joint variation (scale-dependent). **Correlation** is normalised covariance (always −1 to +1, unitless, interpretable).

9. **Correlation ≠ Causation.** Always the most important warning in data science.

10. **Central Limit Theorem:** The mean of n samples → Normal, regardless of the original distribution. This justifies statistical tests, confidence intervals, and ensemble methods.

---

## ❓ Revision Questions

### Basic Understanding
1. What is the difference between population and sample? Give a concrete ML example.
2. A dataset has values [2, 4, 4, 6, 8, 10, 100]. Calculate the mean, median, and mode. Which best represents the "typical" value and why?
3. Define Q1, Q2, Q3, and IQR. What does IQR represent?
4. What does a Z-score of 2.5 mean for a data point?
5. What are the two outcomes of a Bernoulli trial? What parameter defines it?

### Conceptual Understanding
6. Why does sample variance divide by n-1 instead of n? What is Bessel's correction?
7. Explain the difference between the IQR method and Z-score method for outlier detection. When would each be preferred?
8. Explain conditional probability P(A|B). Give a real-world example from ML.
9. What does it mean for two events to be independent? Give one example of independent events and one of dependent events.
10. State the Central Limit Theorem. Why is it so important in statistics and ML?

### Application
11. You run a spam filter. The prior P(spam) = 0.4. The word "free" appears in 60% of spam and 5% of non-spam emails. What is P(spam | contains "free")? Use Bayes Theorem.
12. A dataset has Q1=20, Q3=60. Using the IQR method, what are the outlier fences? Is a value of 120 an outlier?
13. Your feature has mean=100 and std=15. A new observation has value 145. Is it an outlier by the Z-score method (threshold=3)?
14. You sample 100 customers and find mean purchase value of ₹500 with std=₹150. What is the standard error? What is the 95% confidence interval for the true mean?
15. A coin is flipped 20 times. Using the binomial distribution, what is the expected number of heads? What is the probability of exactly 10 heads?

### Technical Understanding
16. Why is the Normal distribution so prevalent in nature? Connect this to the CLT.
17. What is a covariance matrix? What are its properties (shape, symmetry, diagonal)?
18. What is the memoryless property of the exponential distribution?
19. Explain the difference between the Poisson and Exponential distributions. How are they related?
20. A Pearson correlation of 0.85 is found between ice cream sales and drowning rates. What statistical concept explains why this does not mean ice cream causes drowning?

---

## 📝 Quick Revision

```
DESCRIPTIVE STATISTICS
──────────────────────
Central tendency: Mean (sensitive to outliers), Median (robust), Mode
Spread: Range, Variance σ² = Σ(x-μ)²/n, Std dev σ = √Variance
Sample vs population: Use n-1 for sample variance (Bessel's correction)
Z-score: z = (x-μ)/σ → normalises to mean=0, std=1
Percentiles: p-th percentile = value below which p% of data falls
Quartiles: Q1=25%, Q2=50%=Median, Q3=75%
IQR = Q3-Q1 → middle 50% of data
Five-number summary: Min, Q1, Median, Q3, Max → visualised as box plot

OUTLIERS
────────
IQR method: outlier if x < Q1-1.5×IQR or x > Q3+1.5×IQR
Z-score method: outlier if |Z| > 3
Handling: Remove, Cap, Transform (log), Impute, or Keep (anomaly detection)

PROBABILITY
───────────
P(A) = favourable / total, range [0,1]
Conditional: P(A|B) = P(A∩B)/P(B)
Independent: P(A∩B) = P(A)×P(B)
Bayes: P(A|B) = P(B|A)×P(A) / P(B)

DISTRIBUTIONS
─────────────
Bernoulli(p): single binary trial; E=p, Var=p(1-p)
Binomial(n,p): k successes in n trials; E=np
Normal(μ,σ²): bell curve; 68/95/99.7 rule
Uniform(a,b): equal probability in range
Poisson(λ): count of events; E=Var=λ
Exponential(λ): time between events; E=1/λ

STATISTICAL CONCEPTS
────────────────────
E[X] = Σ x P(x) = theoretical average (long-run)
Cov(X,Y) = E[(X-μₓ)(Y-μᵧ)] → joint variation (scale-dependent)
Correlation r = Cov/σₓσᵧ → [-1,+1], unitless
r≠causation: always think about confounders!
CLT: x̄ ~ N(μ, σ²/n) for large n, ANY distribution
SE = σ/√n → how much sample means vary
95% CI: x̄ ± 1.96×SE → range for true population mean
```

---

## 📚 Further Learning

| Resource | Type | Why Useful |
|---|---|---|
| *Think Stats* — Allen Downey | Book (free) | Statistics for programmers, Python examples |
| *Statistics for Data Science* — Stanford CS109 | Course | Probability + statistics with ML applications |
| 3Blue1Brown — "Bayes Theorem" video | YouTube | Best visual intuition for Bayes |
| Khan Academy — Statistics and Probability | Course | Excellent fundamentals, well-explained |
| *The Elements of Statistical Learning* — Hastie et al. | Book (free PDF) | Advanced stats + ML connections |
| StatQuest with Josh Starmer | YouTube | Excellent short explanations of statistical concepts |

---

*📁 Next: `04-data-science/README.md` — Data Science tools, pandas, NumPy, and data manipulation*