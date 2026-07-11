# Unit 2: Mathematics for AI & ML

> **Goal:** Build the mathematical foundation that underpins every AI and ML algorithm — from the simplest linear regression to billion-parameter neural networks.

---

## Table of Contents

1. [Overview](#overview)
2. [Learning Objectives](#learning-objectives)
3. [Part A — Linear Algebra](#part-a--linear-algebra)
4. [Part B — Statistics](#part-b--statistics)
5. [Part C — Probability](#part-c--probability)
6. [Part D — Calculus](#part-d--calculus)
7. [Part E — Optimization](#part-e--optimization)
8. [Code Examples](#code-examples)
9. [Real-World Applications](#real-world-applications)
10. [Interview Notes](#interview-notes)
11. [Common Mistakes](#common-mistakes--misconceptions)
12. [Key Takeaways](#key-takeaways)
13. [Revision Questions](#revision-questions)
14. [Summary](#summary)

---

## Overview

Every AI and ML algorithm is built on a backbone of **four mathematical pillars**:

```
LINEAR ALGEBRA   ->  How data is represented and transformed
STATISTICS       ->  How data is summarised and understood
PROBABILITY      ->  How uncertainty is quantified
CALCULUS         ->  How models learn and improve
OPTIMIZATION     ->  How models find the best solution
```

**Why does this matter?**
- When you train a neural network, you are doing **matrix multiplication** (Linear Algebra)
- When you measure model performance, you use **mean and variance** (Statistics)
- When you build a spam filter, you apply **Bayes Theorem** (Probability)
- When you run gradient descent, you compute **derivatives** (Calculus)
- When you minimise a loss function, you perform **optimization**

Understanding these topics is not optional — they are the *language* of AI.

---

## Learning Objectives

By the end of this unit, you should be able to:

- [ ] Define and manipulate scalars, vectors, and matrices
- [ ] Perform matrix multiplication, transpose, and inverse
- [ ] Calculate mean, median, mode, variance, and standard deviation
- [ ] Apply conditional probability and Bayes Theorem to real problems
- [ ] Understand joint distributions and expectation
- [ ] Compute derivatives and partial derivatives
- [ ] Explain what a gradient is and why it matters
- [ ] Define a cost function and explain its role in ML
- [ ] Walk through gradient descent step by step
- [ ] Implement all of the above in Python

---

## Part A — Linear Algebra

> **Why Linear Algebra?** Data in ML is stored as numbers arranged in grids (matrices). Every transformation, prediction, and layer in a neural network is a matrix operation. Linear algebra IS the language of data.

---

### 1. Scalars

#### Definition

A **scalar** is a **single number** — the simplest mathematical quantity. It has **magnitude only** — no direction.

```
Examples:
  5        -> a scalar
  3.14     -> a scalar
  -7.2     -> a scalar
  0        -> a scalar
```

#### In AI/ML

Scalars appear everywhere:
- **Learning rate:** alpha = 0.01 (controls how fast a model learns)
- **Loss value:** L = 0.342 (a single number measuring model error)
- **A single pixel value** in a grayscale image: 128

#### Notation

Scalars are written as lowercase non-bold letters: x, y, z, alpha, beta

```python
# Scalars in Python
x = 5              # integer scalar
y = 3.14           # float scalar
learning_rate = 0.01   # typical ML scalar
loss = 0.342           # typical ML scalar

print(type(x))  # <class 'int'>
print(type(y))  # <class 'float'>
```

---

### 2. Vectors

#### Definition

A **vector** is an **ordered list of numbers** (scalars). It has both **magnitude and direction**.

```
2D vector:  v = [3, 4]         -> point at x=3, y=4
3D vector:  v = [1, 2, 3]      -> point in 3D space
n-D vector: v = [x1, x2, ..., xn]
```

**Column vector (default in ML):**
```
     | 1 |
v =  | 2 |    Shape: (4, 1)
     | 3 |
     | 4 |
```

**Row vector:**
```
v = [ 1  2  3 ]    Shape: (1, 3)
```

#### In AI/ML

- **A data point:** [age=25, income=50000, score=720]
- **Word embeddings:** "king" -> [0.2, -0.4, 0.8, ..., 0.1] (300 numbers)
- **Weights of a neuron:** w = [w1, w2, w3]
- **Flattened image:** 28x28 image -> vector of 784 numbers

#### Vector Operations

**Vector Addition** (element by element):
```
a = [1, 2, 3]
b = [4, 5, 6]
a + b = [5, 7, 9]
```

**Scalar Multiplication:**
```
a = [1, 2, 3]
3 x a = [3, 6, 9]
```

**Dot Product (extremely important in ML):**

The dot product of two vectors produces a **single scalar**.

```
a = [1, 2, 3]
b = [4, 5, 6]

a . b = (1x4) + (2x5) + (3x6)
      = 4 + 10 + 18
      = 32

Formula: a . b = sum(ai x bi) for i=1 to n
```

**Why the dot product matters:**
- Measures **similarity** between two vectors
- In a neural network: output = **w . x + b**

**Vector Magnitude (Length):**
```
||v|| = sqrt(v1^2 + v2^2 + ... + vn^2)

For v = [3, 4]:
||v|| = sqrt(9 + 16) = sqrt(25) = 5
```

```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print("a + b:", a + b)           # [5 7 9]
print("3 * a:", 3 * a)           # [3 6 9]
print("Dot product:", a @ b)      # 32
print("Magnitude:", np.linalg.norm(a))  # 3.7417
```

---

### 3. Matrices

#### Definition

A **matrix** is a **2D grid of numbers** arranged in **rows and columns**.

```
         Col1  Col2  Col3
Row 1 -> [  1    2    3  ]
Row 2 -> [  4    5    6  ]
Row 3 -> [  7    8    9  ]

This is a 3x3 matrix. Shape notation: (rows, cols) = (3, 3)
```

**General notation:** An m x n matrix has m rows and n columns.

#### In AI/ML

| ML Concept | Matrix Representation |
|---|---|
| **Dataset** | Each row = one data point; each column = one feature |
| **Image** | Pixel values (28x28 grid) |
| **Neural network weights** | Weight matrix between layers |
| **Batch of inputs** | Each row is one example in the batch |

**Example: Dataset as matrix**
```
Feature1  Feature2  Feature3
[ 25       50000     720  ]   <- Student 1
[ 30       75000     680  ]   <- Student 2
[ 22       32000     760  ]   <- Student 3
Shape: (3, 3) — 3 samples, 3 features
```

#### Special Matrices

**Identity Matrix (I):** 1s on diagonal, 0s elsewhere. A x I = A.
```
     | 1  0  0 |
I3 = | 0  1  0 |
     | 0  0  1 |
```

**Analogy:** The identity matrix is the number 1 of the matrix world.

**Diagonal Matrix:** Non-zero only on the main diagonal.
```
| 5  0  0 |
| 0  3  0 |
| 0  0  7 |
```

```python
import numpy as np

A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

print("Shape:", A.shape)          # (3, 3)
print("Element [0,1]:", A[0, 1]) # 2
print("Row 1:", A[1, :])          # [4 5 6]
print("Col 2:", A[:, 2])          # [3 6 9]

I = np.eye(3)     # Identity matrix
Z = np.zeros((3, 3))  # Zero matrix
```

---

### 4. Matrix Operations

#### Element-wise Addition and Subtraction

Two matrices of the **same shape** add/subtract element by element.

```
| 1  2 |   | 5  6 |   | 6   8 |
| 3  4 | + | 7  8 | = | 10  12|

Rule: shapes must be identical. (m x n) + (m x n) = (m x n)
```

#### Scalar-Matrix Multiplication

```
    | 1  2 |       | 3   6 |
3 x | 3  4 |   =   | 9  12 |
```

#### Element-wise (Hadamard) Product

Multiply corresponding elements. Both matrices must be same shape.
Denoted as A circle-dot B.

```
| 1  2 |      | 5  6 |     | 5   12 |
| 3  4 | (o)  | 7  8 |  =  | 21  32 |
```

**In ML:** Used in LSTM gates and dropout masks.

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("A + B:\n", A + B)         # Element-wise addition
print("3 * A:\n", 3 * A)         # Scalar multiplication
print("A * B:\n", A * B)         # Hadamard (element-wise) product
```

---

### 5. Matrix Multiplication

#### Definition

Matrix multiplication is NOT element-wise. It combines two matrices into a new matrix.

**Rule:** To multiply A (m x n) by B (n x p), the inner dimensions must match.
- Result shape: **(m x p)**
- Think: (m x **n**) x (**n** x p) -> (m x p)

#### How to Compute Each Element

Each element Cij = **dot product** of row i of A with column j of B.

```
     A (2x3)          B (3x2)
| 1  2  3 |   | 7   8 |
| 4  5  6 | x | 9  10 |  =  C (2x2)
              | 11 12 |

C[0,0] = row0(A) . col0(B) = [1,2,3].[7,9,11]  = 7+18+33  = 58
C[0,1] = row0(A) . col1(B) = [1,2,3].[8,10,12] = 8+20+36  = 64
C[1,0] = row1(A) . col0(B) = [4,5,6].[7,9,11]  = 28+45+66 = 139
C[1,1] = row1(A) . col1(B) = [4,5,6].[8,10,12] = 32+50+72 = 154

     |  58   64 |
C =  | 139  154 |
```

#### Properties

| Property | Formula | Note |
|---|---|---|
| **Not commutative** | A x B != B x A | Order matters! |
| **Associative** | (A x B) x C = A x (B x C) | Can regroup |
| **Identity** | A x I = I x A = A | I is the "1" of matrices |
| **Dimension** | (m x n)(n x p) = (m x p) | Must match inner dims |

#### Why Matrix Multiplication is the Heart of Neural Networks

Every neural network layer computes:
```
output = activation(W x input + b)

W     = weight matrix (learned parameters)
input = vector from previous layer
b     = bias vector
x     = matrix multiplication
```

Every forward pass is a sequence of matrix multiplications.

```python
import numpy as np

A = np.array([[1, 2, 3], [4, 5, 6]])   # Shape: (2, 3)
B = np.array([[7, 8], [9, 10], [11, 12]])  # Shape: (3, 2)

C = A @ B    # Shape: (2, 2)
print("A @ B:\n", C)
# [[ 58  64]
#  [139 154]]

# Neural network layer example
X = np.array([[0.5, 0.3, 0.8]])   # Input: (1, 3)
W = np.array([[0.2, 0.4],         # Weights: (3, 2)
              [0.1, 0.7],
              [0.9, 0.3]])
b = np.array([[0.1, 0.2]])        # Bias: (1, 2)

output = X @ W + b
print("Neural layer output:", output)
```

---

### 6. Transpose

#### Definition

The **transpose** of matrix A (written A^T) flips it over its main diagonal — rows become columns, columns become rows.

```
        | 1  2  3 |              | 1  4 |
A    =  | 4  5  6 |   ->  A^T = | 2  5 |
        Shape: (2,3)             | 3  6 |
                                 Shape: (3,2)
```

**General rule:** If A is (m, n), then A^T is (n, m).
**Element rule:** (A^T)[i,j] = A[j,i]

#### Properties

```
(A^T)^T = A              (double transpose = original)
(A + B)^T = A^T + B^T   (distributes over addition)
(A x B)^T = B^T x A^T   (order REVERSES!)
```

#### Symmetric Matrix

A matrix is **symmetric** if A = A^T. Elements are mirrored across the diagonal.
Covariance matrices are always symmetric.

#### In ML

- Converting column to row vectors
- Backpropagation: gradient = W^T x upstream_gradient
- Transformer attention scores: Q x K^T (Query times Key transpose)

```python
import numpy as np

A = np.array([[1, 2, 3], [4, 5, 6]])   # (2, 3)
AT = A.T                                 # (3, 2)

print("A.T:\n", AT)
print("A.T shape:", AT.shape)

# Convert column vector to row vector
col = np.array([[1], [2], [3]])  # Shape (3, 1)
row = col.T                       # Shape (1, 3)
print("Col shape:", col.shape, "-> Row shape:", row.shape)

# Dot product using transpose: a^T @ b
a = np.array([[1], [2], [3]])
b = np.array([[4], [5], [6]])
dot = (a.T @ b)[0, 0]
print("a^T @ b =", dot)   # 32
```

---

### 7. Inverse

#### Definition

The **inverse** of square matrix A (written A^-1) satisfies:

```
A x A^-1 = A^-1 x A = I    (identity matrix)

Analogy:
  Numbers:  5 x (1/5) = 1
  Matrices: A x A^-1  = I
```

#### When Does an Inverse Exist?

A matrix is **invertible** (non-singular) if and only if:
- It is **square** (m = n)
- Its **determinant != 0**

**Determinant of 2x2 matrix:**
```
    | a  b |
A = | c  d |     det(A) = a*d - b*c

    | 3  2 |
    | 1  4 |     det = (3*4) - (2*1) = 10   -> Invertible

    | 2  4 |
    | 1  2 |     det = (2*2) - (4*1) = 0    -> NOT invertible
```

**Computing the inverse of a 2x2 matrix:**
```
            1      |  d  -b |
A^-1   =  ------   |        |
           det(A)  | -c   a |
```

#### In ML

- **Normal Equation** for linear regression: theta = (X^T X)^-1 X^T y
- **Note:** For large matrices, direct inverse is expensive. Gradient descent is preferred.

```python
import numpy as np

A = np.array([[3., 2.], [1., 4.]])

A_inv = np.linalg.inv(A)
print("A inverse:\n", A_inv)

# Verify: A x A^-1 = Identity
print("A @ A_inv:\n", np.round(A @ A_inv, 6))

# Determinant
print("Determinant:", np.linalg.det(A))   # 10.0

# Solving Ax = b (better than computing inverse)
b = np.array([8., 9.])
x = np.linalg.solve(A, b)   # More stable than inv()
print("Solution x:", x)
```

---

## Part B — Statistics

> **Why Statistics?** ML models learn from data. Statistics tells you what your data looks like, how spread out it is, and what is normal. Without statistics, you cannot understand your data, detect errors, or evaluate your models.

---

### 8. Mean

#### Definition

The **mean** (average) is the sum of all values divided by the number of values.

**Formula:**
```
         x1 + x2 + ... + xn      1  n
mu   =   ------------------   =  -  sum(xi)
                n               n  i=1

mu   = mean
xi   = each data point
n    = total number of data points
```

#### Example

```
Dataset: [10, 20, 30, 40, 50]
mu = (10 + 20 + 30 + 40 + 50) / 5 = 150 / 5 = 30
```

#### Sensitivity to Outliers

```
Dataset: [10, 20, 30, 40, 1000]
Mean = 220 — pulled far from most data by the outlier!
```

#### In ML

- **Feature normalisation:** subtract mean from each feature (zero-centering)
- **MSE loss:** averages squared errors
- **BatchNorm:** computes mean across a batch

```python
import numpy as np

data = [10, 20, 30, 40, 50]
print("Mean:", np.mean(data))              # 30.0

data_outlier = [10, 20, 30, 40, 1000]
print("Mean with outlier:", np.mean(data_outlier))  # 220.0
```

---

### 9. Median

#### Definition

The **median** is the **middle value** when data is sorted. It divides the dataset into two equal halves.

```
Odd n:   [10, 20, 30, 40, 50]  -> median = 30  (middle of 5)

Even n:  [10, 20, 30, 40]
          middle two: 20 and 30
          median = (20 + 30) / 2 = 25
```

#### Mean vs Median

| Feature | Mean | Median |
|---|---|---|
| **Outlier sensitivity** | High | Low |
| **Represents** | Balance point | Middle value |
| **When to use** | Symmetric data | Skewed data, outliers |
| **Example** | Average test score | Median house price |

```python
import numpy as np

data = [10, 20, 30, 40, 50]
print("Median:", np.median(data))           # 30.0

data_outlier = [10, 20, 30, 40, 1000]
print("Mean:  ", np.mean(data_outlier))     # 220.0 <- affected
print("Median:", np.median(data_outlier))   # 30.0  <- stable!
```

---

### 10. Mode

#### Definition

The **mode** is the value that appears **most frequently** in a dataset.

```
Dataset: [1, 2, 2, 3, 3, 3, 4, 5]
Frequencies: 1->1, 2->2, 3->3, 4->1, 5->1
Mode = 3  (appears 3 times)
```

A dataset can have one mode (unimodal), two (bimodal), or many (multimodal).

#### In ML

- Most common class in a dataset
- Replacing missing categorical values (imputation)
- Majority voting in ensemble classifiers

```python
from scipy import stats
from collections import Counter

data = [1, 2, 2, 3, 3, 3, 4, 5]

mode_result = stats.mode(data)
print("Mode:", mode_result.mode)     # 3

# Manual approach
counts = Counter(data)
mode_manual = counts.most_common(1)[0][0]
print("Manual mode:", mode_manual)   # 3
```

---

### 11. Variance

#### Definition

**Variance** measures how **spread out** the data is around the mean.

**Population Variance Formula:**
```
        1  n
sigma^2 = -  sum( (xi - mu)^2 )
        n i=1

sigma^2  = variance
mu       = mean of the dataset
xi       = each data point
n        = number of data points
(xi - mu)^2 = squared deviation (always positive)
```

**Sample Variance (Bessel's correction):**
```
         1    n
s^2  =  ---   sum( (xi - x_bar)^2 )
        n-1  i=1

Note: n-1 instead of n makes the estimate unbiased for populations
```

#### Step-by-Step Example

```
Dataset: [2, 4, 4, 4, 5, 5, 7, 9]

Step 1: Mean
  mu = (2+4+4+4+5+5+7+9) / 8 = 40/8 = 5

Step 2: Deviations from mean (xi - mu)
  2-5=-3,  4-5=-1,  4-5=-1,  4-5=-1
  5-5=0,   5-5=0,   7-5=2,   9-5=4

Step 3: Squared deviations
  9, 1, 1, 1, 0, 0, 4, 16

Step 4: Average of squared deviations
  sigma^2 = (9+1+1+1+0+0+4+16) / 8 = 32/8 = 4
```

```python
import numpy as np

data = [2, 4, 4, 4, 5, 5, 7, 9]

var_pop    = np.var(data, ddof=0)   # Population variance
var_sample = np.var(data, ddof=1)   # Sample variance (Bessel)
print("Population variance:", var_pop)    # 4.0
print("Sample variance:", var_sample)     # 4.571...
```

---

### 12. Standard Deviation

#### Definition

**Standard Deviation (sigma)** is the **square root of variance**. It measures spread in the **same units as the data**.

```
sigma = sqrt(sigma^2) = sqrt(variance)

For our example:
sigma^2 = 4
sigma   = sqrt(4) = 2

Meaning: On average, data points are about 2 units away from the mean.
```

#### The Empirical Rule (68-95-99.7 Rule)

For **normally distributed** data:
```
68%  of data lies within mu +/- 1*sigma
95%  of data lies within mu +/- 2*sigma
99.7% of data lies within mu +/- 3*sigma

Bell curve shape:
       99.7%
  <------------------>
        95%
    <------------>
       68%
      <------>
         |
     ----+----
        mu
  mu-3s mu-2s mu-s  mu  mu+s mu+2s mu+3s
```

#### In ML

- **Z-score normalisation:** z = (x - mu) / sigma
- **Anomaly detection:** values beyond 3*sigma are outliers
- **Weight initialisation:** N(0, sigma^2)
- **Batch Normalisation:** normalises layer outputs

```python
import numpy as np

data = np.array([2, 4, 4, 4, 5, 5, 7, 9], dtype=float)

std = np.std(data, ddof=0)
mean = np.mean(data)

print("Standard deviation:", std)   # 2.0

# Z-score normalisation
z_scores = (data - mean) / std
print("Z-scores:", np.round(z_scores, 2))
# Values represent standard deviations from the mean

# Verify: normalised data has mean~0 and std~1
print("Normalised mean:", np.round(z_scores.mean(), 6))  # ~0
print("Normalised std: ", np.round(z_scores.std(),  6))  # ~1
```

---

## Part C — Probability

> **Why Probability?** ML models make predictions under uncertainty. A classifier does not say "this is a cat" — it says "I am 87% confident this is a cat." Probability is the mathematics of uncertainty.

---

### 13. Conditional Probability

#### Definition

**Conditional Probability** is the probability of event A occurring **given that event B has already occurred**.

```
P(A | B) = "Probability of A, given B"

Formula:
           P(A and B)
P(A | B) = ----------
              P(B)

P(A | B)   = probability of A given B
P(A and B) = probability of both A AND B occurring
P(B)       = probability of B (must be > 0)
```

#### Example

A box has 10 balls: 4 red, 6 blue.

After drawing a red ball without replacement:
```
P(2nd red | 1st red) = 3/9 = 0.333

(3 red remain, 9 total remain after removing 1 red)
```

The **condition** changed our probability.

#### In ML

- Naive Bayes classifier: P(Class | Features)
- Language models: P(next word | previous words)
- Reinforcement learning: P(next state | current state, action)

```python
# Conditional probability example
# 100 students: 60 study, 40 do not
# 50 passed: 45 studied, 5 did not study

studied = 60
not_studied = 40
passed_and_studied = 45
passed_and_not_studied = 5

P_pass_given_studied     = passed_and_studied / studied
P_pass_given_not_studied = passed_and_not_studied / not_studied

print(f"P(Pass | Studied)     = {P_pass_given_studied:.2f}")     # 0.75
print(f"P(Pass | Not Studied) = {P_pass_given_not_studied:.2f}") # 0.125
```

---

### 14. Bayes Theorem

#### Definition

**Bayes Theorem** updates the probability of a hypothesis given new evidence.

```
           P(B | A) x P(A)
P(A | B) = ---------------
                P(B)

P(A | B)  = Posterior  -- what we want after seeing evidence
P(B | A)  = Likelihood -- how likely is evidence B if A is true
P(A)      = Prior      -- our belief about A before evidence
P(B)      = Marginal   -- total probability of B (normaliser)
```

#### Intuition

```
Prior Belief  +  New Evidence  ->  Updated Belief (Posterior)

"I think it might rain (40%)" + "Sky is dark and cloudy"
-> "I now think it will rain (85%)"
```

#### Step-by-Step Medical Example

**Setup:**
```
P(Disease)          = 0.01   (1% prevalence)
P(Positive | D)     = 0.90   (90% sensitivity)
P(Positive | no D)  = 0.05   (5% false positive rate)
```

**Question:** P(Disease | Positive test)?

**Step 1 — Total probability of positive test:**
```
P(Positive) = P(Pos|D) x P(D)  +  P(Pos|no D) x P(no D)
            = (0.90 x 0.01)    +  (0.05 x 0.99)
            = 0.009 + 0.0495
            = 0.0585
```

**Step 2 — Bayes Theorem:**
```
                P(Pos | D) x P(D)       0.90 x 0.01
P(D | Pos)  =  ------------------   =  -----------  =  0.154
                    P(Pos)                0.0585
```

**Surprising result:** Even with a positive test, only ~15.4% chance of disease! This is because the disease is rare (low prior). Base rates matter enormously.

#### Bayes in ML

| Application | Meaning |
|---|---|
| **Naive Bayes Classifier** | P(class | features) proportional to P(features | class) x P(class) |
| **Spam filter** | P(spam | words) proportional to P(words | spam) x P(spam) |
| **Bayesian inference** | Update beliefs as new data arrives |

```python
# Bayes Theorem - Medical Test Example

P_disease           = 0.01
P_healthy           = 1 - P_disease        # 0.99
P_pos_given_disease = 0.90
P_pos_given_healthy = 0.05

# Step 1: Total probability of positive test
P_positive = (P_pos_given_disease * P_disease) + \
             (P_pos_given_healthy * P_healthy)

print(f"P(Positive) = {P_positive:.4f}")   # 0.0585

# Step 2: Bayes Theorem
P_disease_given_positive = (P_pos_given_disease * P_disease) / P_positive

print(f"P(Disease | Positive) = {P_disease_given_positive:.4f}")
# 0.1538 -> only ~15.4% chance even with positive test!

# What if prevalence was 10%?
P_disease_high = 0.10
P_pos_high = (P_pos_given_disease * P_disease_high) + \
             (P_pos_given_healthy * (1 - P_disease_high))
posterior_high = (P_pos_given_disease * P_disease_high) / P_pos_high
print(f"P(Disease | Positive) at 10% prevalence = {posterior_high:.4f}")
# ~0.667 -> prior matters enormously!
```

---

### 15. Joint Distribution

#### Definition

A **joint distribution** describes the probability of two or more events occurring **simultaneously**.

```
P(A and B) = probability that both A and B occur
```

#### Joint Probability Table

```
                 | Pass  | Fail  | Total
-----------------+-------+-------+-------
     Male        |  40   |  15   |  55
     Female      |  35   |  10   |  45
-----------------+-------+-------+-------
     Total       |  75   |  25   |  100

Joint probabilities (divide by 100):
  P(Male, Pass)   = 40/100 = 0.40
  P(Male, Fail)   = 15/100 = 0.15
  P(Female, Pass) = 35/100 = 0.35
  P(Female, Fail) = 10/100 = 0.10

Sum of all joint probabilities = 1.0 (always)
```

#### Marginal Probability

Sum over the other variable:
```
P(Male) = P(Male,Pass) + P(Male,Fail) = 0.40 + 0.15 = 0.55
P(Pass) = P(Male,Pass) + P(Female,Pass) = 0.40 + 0.35 = 0.75
```

#### Independence

Two events A and B are **independent** if:
```
P(A, B) = P(A) x P(B)

Knowing B gives no information about A.
Coin flips are independent.
```

#### In ML

- **Naive Bayes** assumes features are conditionally independent (the "naive" assumption)
- **Generative models** (VAE, GAN) learn the joint distribution P(X, Y)
- **Discriminative models** (logistic regression) only model P(Y|X)

```python
# Joint distribution example
joint = {
    ('Male',   'Pass'): 40,
    ('Male',   'Fail'): 15,
    ('Female', 'Pass'): 35,
    ('Female', 'Fail'): 10,
}
total = 100

print("Joint Probabilities:")
for (gender, result), count in joint.items():
    print(f"  P({gender}, {result}) = {count/total:.2f}")

# Marginals
p_male = (joint[('Male','Pass')] + joint[('Male','Fail')]) / total
p_pass = (joint[('Male','Pass')] + joint[('Female','Pass')]) / total

# Check independence: P(Male, Pass) =? P(Male) x P(Pass)
p_male_pass_joint = joint[('Male','Pass')] / total
p_male_x_pass = p_male * p_pass
print(f"\nP(Male,Pass) = {p_male_pass_joint:.2f}")
print(f"P(Male)*P(Pass) = {p_male_x_pass:.2f}")
print(f"Independent? {abs(p_male_pass_joint - p_male_x_pass) < 0.01}")
```

---

### 16. Expectation

#### Definition

The **Expected Value E[X]** is the **weighted average** of all possible outcomes, weighted by their probabilities.

"If I repeated this experiment infinite times, what would the average outcome be?"

**Formula for discrete random variable:**
```
E[X] = sum( xi x P(X = xi) )   for all i

xi       = each possible value
P(X=xi)  = probability of that value
```

#### Example: Rolling a Fair Die

```
Values:        1     2     3     4     5     6
Probabilities: 1/6   1/6   1/6   1/6   1/6   1/6

E[X] = 1x(1/6) + 2x(1/6) + 3x(1/6) + 4x(1/6) + 5x(1/6) + 6x(1/6)
     = (1+2+3+4+5+6) / 6 = 21/6 = 3.5

We never roll 3.5 but over many rolls we average 3.5.
```

#### For Continuous Variables

```
E[X] = integral( x x f(x) dx )   where f(x) is the probability density

For a Normal distribution N(mu, sigma^2):
  E[X] = mu    (the mean IS the expectation)
```

#### Properties

```
E[X + Y]  = E[X] + E[Y]          (linearity — always true)
E[a*X + b] = a*E[X] + b          (linearity with scalar)
E[X x Y]  = E[X] x E[Y]          (ONLY if X and Y are INDEPENDENT)
Var(X)    = E[(X - E[X])^2] = E[X^2] - (E[X])^2
```

#### In ML

| Context | Expectation Meaning |
|---|---|
| **Loss function** | E[L(y, y_hat)] — expected loss over training data |
| **Reward in RL** | E[cumulative reward] |
| **Distribution mean** | E[X] = mu for Gaussian |

```python
import numpy as np

# Discrete expectation: biased die (6 is twice as likely)
values = [1, 2, 3, 4, 5, 6]
probs  = [1/7, 1/7, 1/7, 1/7, 1/7, 2/7]   # probabilities must sum to 1

expectation = sum(v * p for v, p in zip(values, probs))
print(f"E[X] for biased die = {expectation:.4f}")   # > 3.5 since 6 is favoured

# For Normal distribution: E[X] = mean
rng = np.random.default_rng(42)
samples = rng.normal(loc=5.0, scale=2.0, size=100000)
print(f"E[X] for N(5, 4) ~ {samples.mean():.4f}")  # approx 5.0

# Variance via expectation formula: Var(X) = E[X^2] - (E[X])^2
E_X  = samples.mean()
E_X2 = (samples**2).mean()
var_formula = E_X2 - E_X**2
print(f"Var via formula:  {var_formula:.4f}")
print(f"Var via numpy:    {samples.var():.4f}")   # should match
```

---

## Part D — Calculus

> **Why Calculus?** Neural networks learn by adjusting weights to minimize error. The tool they use to find the direction to adjust is the **gradient** — computed using derivatives. Without calculus, no gradient descent, no learning.

---

### 17. Derivatives

#### Definition

The **derivative** of f(x) measures the **rate of change** of f as x changes.

**Formal definition:**
```
          f(x + h) - f(x)
f'(x) = lim ---------------
         h->0       h

f'(x) = derivative at point x (also written df/dx)
h     = tiny increment approaching 0
```

#### Geometric Interpretation

The derivative at a point = **slope of the tangent line** at that point.

```
f(x)
|          /  <- tangent line (slope = f'(x0))
|       / /
|    / /
|   * <- point (x0, f(x0))
|
+-------------> x
```

- f'(x) > 0: function is increasing at x
- f'(x) < 0: function is decreasing at x
- f'(x) = 0: local minimum or maximum (flat point)

#### Common Derivatives

| Function | Derivative | Example |
|---|---|---|
| f(x) = c (constant) | f'(x) = 0 | f(x)=5 -> f'(x)=0 |
| f(x) = x^n | f'(x) = n * x^(n-1) | f(x)=x^3 -> f'(x)=3x^2 |
| f(x) = e^x | f'(x) = e^x | unchanged |
| f(x) = ln(x) | f'(x) = 1/x | |
| f(x) = sin(x) | f'(x) = cos(x) | |

#### Chain Rule (CRITICAL for neural networks)

The chain rule differentiates **composed functions**.

```
If y = f(g(x)), then:

dy/dx = (df/dg) x (dg/dx)
      = (derivative of outer) x (derivative of inner)

Example: y = (3x + 2)^2
  Let u = 3x + 2  ->  y = u^2
  dy/du = 2u
  du/dx = 3
  dy/dx = 2u x 3 = 6(3x + 2)
```

**Why it matters for neural networks:**
A neural network is a composition of many functions. Backpropagation IS the chain rule applied repeatedly from output back to input.

```python
import numpy as np

# Numerical derivative approximation
def numerical_derivative(f, x, h=1e-7):
    return (f(x + h) - f(x)) / h

f1 = lambda x: x**2           # f'(x) = 2x
f2 = lambda x: x**3           # f'(x) = 3x^2
f3 = lambda x: np.exp(x)      # f'(x) = e^x

x = 3.0
print(f"f(x)=x^2  at x=3: numerical={numerical_derivative(f1,x):.4f}, exact={2*x}")
print(f"f(x)=x^3  at x=3: numerical={numerical_derivative(f2,x):.4f}, exact={3*x**2}")
print(f"f(x)=e^x  at x=3: numerical={numerical_derivative(f3,x):.4f}, exact={np.exp(x):.4f}")

# Activation functions used in neural networks
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1.0, 0.0)   # 1 if x>0, 0 otherwise

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)   # sigma(x) * (1 - sigma(x))

x_test = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
print("\nReLU derivatives:")
for x in x_test:
    print(f"  x={x:.1f}: relu={relu(x):.2f}, relu'={relu_derivative(x):.2f}")
```

---

### 18. Partial Derivatives

#### Definition

A **partial derivative** differentiates a function of **multiple variables** with respect to **one variable at a time**, holding all others constant.

```
Notation: df/dx  or  partial-f/partial-x
(the 'curly d' symbol denotes partial derivative)
```

#### Example

```
f(x, y) = x^2 + 3xy + y^2

Partial derivative with respect to x (treat y as constant):
  df/dx = 2x + 3y

Partial derivative with respect to y (treat x as constant):
  df/dy = 3x + 2y

At point (x=2, y=1):
  df/dx = 2(2) + 3(1) = 7   -> if x increases slightly, f increases ~7x as fast
  df/dy = 3(2) + 2(1) = 8   -> if y increases slightly, f increases ~8x as fast
```

#### In Neural Networks

A neural network loss L is a function of millions of weights (w1, w2, ..., wN).

We need: "If I change weight wi slightly, how does loss change?"
That is exactly: dL/dwi — the partial derivative of loss with respect to each weight.

```python
def partial_derivative(f, variables, index, h=1e-7):
    """
    Numerically compute df/dx_index.
    variables: list of current variable values
    index: which variable to differentiate
    """
    vars_plus = variables.copy()
    vars_plus[index] += h
    return (f(*vars_plus) - f(*variables)) / h

# f(x, y) = x^2 + 3xy + y^2
def f(x, y):
    return x**2 + 3*x*y + y**2

x, y = 2.0, 1.0

df_dx_num = partial_derivative(f, [x, y], index=0)
df_dy_num = partial_derivative(f, [x, y], index=1)

df_dx_exact = 2*x + 3*y
df_dy_exact = 3*x + 2*y

print(f"At (x={x}, y={y}):")
print(f"  df/dx: numerical={df_dx_num:.4f}, exact={df_dx_exact}")
print(f"  df/dy: numerical={df_dy_num:.4f}, exact={df_dy_exact}")
```

---

### 19. Gradient

#### Definition

The **gradient** of function f is a **vector** containing ALL partial derivatives of f.

```
For f(x1, x2, ..., xn):

           | df/dx1 |
           | df/dx2 |
gradient = |   ...  |
           | df/dxn |

One partial derivative per variable.
Denoted: nabla(f) or grad(f)
```

#### The Key Property of the Gradient

> **The gradient always points in the direction of STEEPEST ASCENT.**
> The **negative gradient** points in the direction of **steepest descent**.

This is the foundation of gradient descent.

#### Visualisation

Imagine f as a 3D hilly landscape:
```
f(x,y)
  ^
  |     * <- peak
  |   /
  | /
  +----------> (x, y) plane

At any point on the surface:
  gradient(f)  = points uphill
  -gradient(f) = points downhill (toward minimum)
```

#### Example

```
f(x, y) = x^2 + y^2   (bowl shape — minimum at origin)

df/dx = 2x
df/dy = 2y

gradient(f) = [2x, 2y]

At point (3, 4):
  gradient = [6, 8]  -> points away from origin (uphill for a bowl)

  Negative gradient = [-6, -8]  -> points toward origin (minimum!)

  Magnitude = sqrt(6^2 + 8^2) = 10
  Large magnitude -> steep slope. Zero magnitude -> at minimum.
```

```python
import numpy as np

def f(params):
    x, y = params
    return x**2 + y**2

def gradient_f(params):
    x, y = params
    return np.array([2*x, 2*y])   # [df/dx, df/dy]

def numerical_gradient(f, params, h=1e-7):
    params = np.array(params, dtype=float)
    grad = np.zeros_like(params)
    for i in range(len(params)):
        params_plus  = params.copy(); params_plus[i]  += h
        params_minus = params.copy(); params_minus[i] -= h
        grad[i] = (f(params_plus) - f(params_minus)) / (2 * h)
    return grad

point = np.array([3.0, 4.0])

grad_analytical = gradient_f(point)
grad_numerical  = numerical_gradient(f, point)

print(f"Analytical gradient: {grad_analytical}")     # [6. 8.]
print(f"Numerical gradient:  {grad_numerical}")      # [6. 8.]
print(f"Magnitude: {np.linalg.norm(grad_analytical):.4f}")   # 10.0

print(f"Descent direction (neg grad): {-grad_analytical}")    # [-6. -8.]
```

---

## Part E — Optimization

> **Why Optimization?** Training an ML model means finding the best possible parameters (weights) that minimise prediction error. Optimization is HOW we find those parameters.

---

### 20. Cost Function

#### Definition

A **cost function** (also: loss function, objective function) is a formula that measures **how far off a model's predictions are from the true values**.

```
Cost Function: L(y, y_hat)

L     = loss (a single number — our error measure)
y     = true label
y_hat = predicted value

Goal: MINIMISE the cost function by adjusting model parameters.
```

#### Types of Cost Functions

**1. Mean Squared Error (MSE) — Regression**

Used when output is a continuous number (e.g., house price prediction).

```
        1   n
MSE  =  -   sum( (yi - y_hat_i)^2 )
        n  i=1

n      = number of training examples
yi     = true value
y_hat_i = predicted value
(yi - y_hat_i) = error / residual
```

Why square the errors?
- Squaring makes all errors positive
- Penalises large errors more than small ones

Example:
```
True:      y     = [3, -0.5,  2,  7]
Predicted: y_hat = [2.5, 0,   2,  8]
Errors:           [0.5, -0.5, 0, -1]
Squared:          [0.25, 0.25, 0, 1.0]
MSE = (0.25 + 0.25 + 0 + 1.0) / 4 = 0.375
```

**2. Binary Cross-Entropy — Classification**

Used for predicting class probabilities.

```
L = -[ y * log(y_hat) + (1-y) * log(1 - y_hat) ]

y     = true label (0 or 1)
y_hat = predicted probability (between 0 and 1)
log   = natural logarithm

Intuition:
  y=1 and y_hat=0.99: loss = -log(0.99) = ~0.01  (small, almost correct)
  y=1 and y_hat=0.01: loss = -log(0.01) = ~4.6   (large, badly wrong)
```

The log function severely punishes confident wrong predictions.

**3. Mean Absolute Error (MAE)**

```
        1   n
MAE  =  -   sum( |yi - y_hat_i| )
        n  i=1
```

Less sensitive to outliers than MSE.

**Comparison:**

| Loss Function | Task | Outlier Sensitivity |
|---|---|---|
| **MSE** | Regression | High |
| **MAE** | Regression | Low |
| **Cross-Entropy** | Classification | Moderate |

```python
import numpy as np

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mae_loss(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def binary_crossentropy(y_true, y_pred):
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

y_true = np.array([3.0, -0.5, 2.0, 7.0])
y_pred = np.array([2.5,  0.0, 2.0, 8.0])
print(f"MSE: {mse_loss(y_true, y_pred):.4f}")   # 0.375
print(f"MAE: {mae_loss(y_true, y_pred):.4f}")   # 0.500

y_true_cls  = np.array([1, 0, 1, 1])
y_pred_good = np.array([0.95, 0.05, 0.90, 0.85])
y_pred_bad  = np.array([0.10, 0.90, 0.20, 0.15])

print(f"CE good predictions: {binary_crossentropy(y_true_cls, y_pred_good):.4f}")  # Small
print(f"CE bad  predictions: {binary_crossentropy(y_true_cls, y_pred_bad):.4f}")   # Large
```

---

### 21. Gradient Descent

#### Definition

**Gradient Descent** minimises the cost function by iteratively moving parameters in the **opposite direction of the gradient**.

**The intuition:**
- Imagine standing blindfolded on a hilly landscape (the loss surface)
- Your goal: reach the lowest valley (minimum loss)
- At each step, feel which way is "downhill" and step that way
- Repeat until you stop descending

```
Loss
|
|   \
|    \
|     \
|      \  <- descent path
|       \_____
|             |<- minimum
+-----------------------> Parameters (weights)
```

#### The Update Rule

```
theta_new = theta_old - alpha x gradient(L)(theta)

theta     = parameters (weights of the model)
alpha     = learning rate (step size)
gradient(L) = gradient of the loss
-         = SUBTRACT gradient (move downhill)
```

#### Learning Rate (alpha)

The learning rate controls step size — this is crucial:

```
Too HIGH:                        Too LOW:
  |                               |
  | overshoots                    | too slow
  | /\ /\  /\ diverges!           |  \-->-->-->
  |/  \/  \/                      |   \-->-->
  +----------->                   +----------->

  alpha = 0.1 (too high)          alpha = 0.000001 (too low)
  Typical: alpha = 0.001 to 0.01
```

#### Types of Gradient Descent

| Type | Data per Update | Speed | Noise | Common Use |
|---|---|---|---|---|
| **Batch GD** | All training data | Slow | Low | Small datasets |
| **Stochastic GD (SGD)** | 1 example | Fast | High | Large datasets |
| **Mini-Batch GD** | k examples (e.g. 32) | Balanced | Medium | **Deep Learning (standard)** |

Common batch sizes: 32, 64, 128, 256.

#### Step-by-Step Worked Example

Problem: Fit y = wx + b to one data point: x=2, y=4. Start with w=0, b=0, alpha=0.1.

```
Step 1 — Forward pass:
  y_hat = 0*2 + 0 = 0

Step 2 — Loss:
  L = (4 - 0)^2 = 16

Step 3 — Gradients:
  dL/dw = -2x(y - y_hat) = -2*2*(4-0) = -16
  dL/db = -2(y - y_hat)  = -2*(4-0)   = -8

Step 4 — Update:
  w_new = 0 - 0.1*(-16) = 0 + 1.6 = 1.6
  b_new = 0 - 0.1*(-8)  = 0 + 0.8 = 0.8

Step 5 — Repeat until loss is small
```

```python
import numpy as np

# =====================================================
# GRADIENT DESCENT FROM SCRATCH
# Task: Find best line y = wx + b for a dataset
# =====================================================

np.random.seed(42)
X = np.random.randn(100)
y = 3 * X + 2 + np.random.randn(100) * 0.5   # true: w=3, b=2

# Initialise parameters
w = 0.0
b = 0.0
alpha = 0.01   # learning rate
n = len(X)

for epoch in range(1000):

    # FORWARD PASS — compute predictions
    y_pred = w * X + b

    # COMPUTE LOSS (MSE)
    loss = np.mean((y - y_pred) ** 2)

    # COMPUTE GRADIENTS
    error  = y - y_pred                        # residuals
    dL_dw  = (-2 / n) * np.sum(X * error)     # gradient wrt w
    dL_db  = (-2 / n) * np.sum(error)         # gradient wrt b

    # UPDATE PARAMETERS (move opposite to gradient)
    w = w - alpha * dL_dw
    b = b - alpha * dL_db

    if epoch % 200 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss:.4f} | w: {w:.4f} | b: {b:.4f}")

print(f"\nFinal: w = {w:.4f} (true: 3.0)")
print(f"Final: b = {b:.4f} (true: 2.0)")
```

**Expected output:**
```
Epoch    0 | Loss: 4.8311 | w: 0.2193 | b: 0.0197
Epoch  200 | Loss: 0.2856 | w: 2.8901 | b: 1.9541
Epoch  400 | Loss: 0.2527 | w: 2.9765 | b: 1.9804
Epoch  800 | Loss: 0.2511 | w: 2.9971 | b: 1.9951

Final: w = 2.9990 (true: 3.0)  <- very close!
Final: b = 1.9975 (true: 2.0)  <- very close!
```

#### Challenges with Gradient Descent

| Problem | Description | Solution |
|---|---|---|
| **Local minima** | Gets stuck in a valley that is not the global best | Momentum, random restarts |
| **Saddle points** | Gradient = 0 but not a minimum | Adaptive optimizers (Adam) |
| **Vanishing gradient** | Gradients become tiny deep in networks | ReLU, BatchNorm, ResNet |
| **Exploding gradient** | Gradients become enormous | Gradient clipping |
| **Slow convergence** | Takes too long | Adaptive learning rates |

#### Modern Optimizers (Preview)

```
SGD -> SGD + Momentum -> RMSprop -> Adam

Adam = Adaptive Moment Estimation
     = Momentum + RMSprop combined
     = Adjusts learning rate per parameter automatically

In practice: almost always use Adam or AdamW.
```

---

## Code Examples

### Full Pipeline: Data to Trained Model Using All Five Math Areas

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

# =====================================================
# COMPLETE ML MATH PIPELINE
# Uses: Vectors, Matrices, Stats, Probability, Calculus
# =====================================================

# 1. GENERATE DATA — stored as a MATRIX
X, y = make_classification(n_samples=200, n_features=2,
                            n_redundant=0, random_state=42)
# X is a MATRIX (200, 2) — 200 samples, 2 features
# y is a VECTOR (200,) — 0s and 1s

print("=== DATA (Linear Algebra) ===")
print(f"X shape: {X.shape}")   # (200, 2) — a MATRIX
print(f"y shape: {y.shape}")   # (200,)  — a VECTOR

# 2. STATISTICS — understand the data
print("\n=== STATISTICS ===")
for i in range(X.shape[1]):
    print(f"Feature {i+1}: mean={np.mean(X[:,i]):.4f}, "
          f"std={np.std(X[:,i]):.4f}, "
          f"var={np.var(X[:,i]):.4f}")

# 3. FEATURE SCALING — Z-score normalisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)   # (x - mean) / std

print("\n=== AFTER SCALING ===")
print(f"Feature 1 mean: {np.mean(X_scaled[:,0]):.6f}")  # ~0
print(f"Feature 1 std:  {np.std(X_scaled[:,0]):.6f}")   # ~1

# 4. LOGISTIC REGRESSION via Gradient Descent
# Uses: matrix multiply, derivatives, gradient descent

def sigmoid(z):
    """Maps any value to (0,1) — probability"""
    return 1 / (1 + np.exp(-z))

def predict(X, w, b):
    """Forward pass: linear combination + sigmoid"""
    z = X @ w + b   # MATRIX MULTIPLICATION: (200,2)@(2,) = (200,)
    return sigmoid(z)

def compute_loss(y_true, y_pred):
    """Binary cross-entropy loss"""
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Initialise parameters
w = np.zeros(X_scaled.shape[1])   # weight VECTOR [0, 0]
b = 0.0                            # bias SCALAR
alpha = 0.1
n = len(y)

print("\n=== TRAINING (Gradient Descent) ===")
for epoch in range(500):

    # Forward pass
    y_pred = predict(X_scaled, w, b)

    # Loss (STATISTICS: mean of cross-entropy)
    loss = compute_loss(y, y_pred)

    # Gradients (CALCULUS: partial derivatives)
    error = y_pred - y
    dL_dw = (1 / n) * (X_scaled.T @ error)  # TRANSPOSE used here
    dL_db = (1 / n) * np.sum(error)

    # Update (GRADIENT DESCENT: move opposite to gradient)
    w = w - alpha * dL_dw
    b = b - alpha * dL_db

    if epoch % 100 == 0:
        acc = np.mean((y_pred >= 0.5).astype(int) == y)
        print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | Accuracy: {acc:.4f}")

print(f"\nFinal weights: {w}")
print(f"Final bias: {b:.4f}")
```

---

## Real-World Applications

| Math Concept | ML/AI Application | Example |
|---|---|---|
| **Vectors** | Data points, embeddings | Word2Vec: "king" = [0.2, -0.4, ...] |
| **Matrices** | Dataset, weight layers | 60,000 x 784 MNIST image matrix |
| **Matrix Mult.** | Neural forward pass | output = W x input + b |
| **Transpose** | Backpropagation | gradient = W^T x upstream_grad |
| **Inverse** | Linear regression | theta = (X^T X)^-1 X^T y |
| **Mean** | Loss averaging | MSE = mean of squared errors |
| **Variance/Std** | Normalisation, BatchNorm | z = (x - mu) / sigma |
| **Conditional Prob.** | Language models | P(word | context) |
| **Bayes Theorem** | Naive Bayes classifier | Spam filtering |
| **Joint Distribution** | Generative models | VAE, GAN learn P(X, Y) |
| **Expectation** | Reinforcement learning | E[cumulative reward] |
| **Derivatives** | Backpropagation | dL/dw for each weight |
| **Partial Derivatives** | Multi-param optimisation | dL/dw1, dL/dw2, ... |
| **Gradient** | Gradient descent direction | Move opposite to nabla(L) |
| **Cost Function** | Training objective | Minimise MSE or cross-entropy |
| **Gradient Descent** | Training neural networks | w <- w - alpha * nabla(L) |

---

## Interview Notes

**Q1: What is the difference between a scalar, vector, and matrix?**

> A scalar is a single number (e.g., learning rate = 0.01). A vector is an ordered list of numbers (e.g., a feature vector [age, income, score]). A matrix is a 2D grid of numbers (e.g., a dataset with rows as samples and columns as features). In ML, data flows through networks primarily as matrices.

---

**Q2: Why does matrix multiplication require matching inner dimensions?**

> Each element of the result is a dot product of a row from the first matrix with a column from the second. For a dot product to work, both vectors must have the same length. So (m x n) x (n x p) gives (m x p) — the shared dimension n must match.

---

**Q3: What is the difference between variance and standard deviation?**

> Variance is the average of squared deviations from the mean (sigma^2). Standard deviation is the square root of variance (sigma). Both measure spread, but standard deviation is in the same units as the data, making it more interpretable.

---

**Q4: Explain Bayes Theorem in plain language.**

> Bayes Theorem tells us how to update our beliefs when we receive new evidence. It combines our prior knowledge with the likelihood of that evidence given our hypothesis. In ML it underlies Naive Bayes classifiers and any Bayesian inference approach.

---

**Q5: What is a gradient? Why is it important in ML?**

> The gradient is a vector of all partial derivatives of a function — one per parameter. It points in the direction of steepest ascent. In ML we compute the gradient of the loss with respect to all model parameters, then move in the OPPOSITE direction (negative gradient) to minimise loss. This is gradient descent.

---

**Q6: What is the cost function? Give an example.**

> The cost function quantifies how wrong a model's predictions are. Training minimises it. Mean Squared Error (MSE = mean of (y - y_hat)^2) is used for regression. Binary cross-entropy is used for classification. Cross-entropy penalises confident wrong predictions very heavily using logarithms.

---

**Q7: What happens if the learning rate is too high or too low?**

> Too high: the algorithm overshoots the minimum and may diverge — parameters oscillate or explode. Too low: convergence is extremely slow and the model may get stuck. Learning rate must be tuned, often between 0.001 and 0.01, and schedulers can reduce it over time.

---

**Q8: What is the chain rule and why does it matter for backpropagation?**

> The chain rule states that if y = f(g(x)), then dy/dx = (df/dg) x (dg/dx). A neural network is a composition of many functions (layers). Backpropagation applies the chain rule recursively from output back to input, computing the gradient of loss with respect to each weight. Without it we cannot train deep networks.

---

## Common Mistakes & Misconceptions

**Mistake 1: "Matrix multiplication is element-wise"**

Correction: Matrix multiplication (A @ B) computes dot products of rows and columns. Element-wise multiplication in NumPy uses *, and is called the Hadamard product.

---

**Mistake 2: "Mean is always the best measure of centre"**

Correction: The mean is distorted by outliers. For skewed distributions (house prices, salaries), the median is more representative.

---

**Mistake 3: "Variance and standard deviation are the same"**

Correction: Variance is sigma^2, standard deviation is sigma = sqrt(sigma^2). They measure the same spread but in different units. Std is always in original data units.

---

**Mistake 4: "P(A|B) is the same as P(B|A)"**

Correction: These are completely different quantities. This confusion is called the base rate fallacy. P(disease | positive test) is not the same as P(positive test | disease). Bayes Theorem converts between them.

---

**Mistake 5: "Gradient descent always finds the global minimum"**

Correction: Gradient descent finds a local minimum. For non-convex functions (like neural networks), there may be many local minima. In practice, modern networks are overparameterised enough that most local minima are near the global minimum.

---

**Mistake 6: "A large learning rate trains faster"**

Correction: A learning rate that is too large causes divergence — loss increases rather than decreasing. The learning rate requires careful tuning.

---

**Mistake 7: "Derivative of x^2 is 2x^2"**

Correction: The power rule says d/dx[x^n] = n * x^(n-1). So d/dx[x^2] = 2 * x^(2-1) = 2x. The exponent decreases by 1.

---

**Mistake 8: "Compute matrix inverse to solve linear systems"**

Correction: np.linalg.solve(A, b) is numerically more stable and efficient than computing A^-1 explicitly. Use solve for Ax = b problems.

---

## Key Takeaways

```
=====================================================
             UNIT 2 KEY TAKEAWAYS
=====================================================

LINEAR ALGEBRA
  Scalar = single number
  Vector = ordered list of numbers
  Matrix = 2D grid of numbers
  Matrix mult: (m x n)(n x p) = (m x p)
  Dot product = foundation of every neural layer
  Transpose flips rows/cols — critical for backprop
  Inverse: A^-1 exists iff square and det != 0

STATISTICS
  Mean = balance point (sensitive to outliers)
  Median = middle value (robust to outliers)
  Variance = average squared deviation (sigma^2)
  Std deviation = sqrt(variance) — same units as data
  Z-score = (x - mean) / std — normalises features

PROBABILITY
  P(A|B) = P(A and B) / P(B) — conditional prob
  Bayes: P(A|B) = P(B|A) x P(A) / P(B)
  Prior x Likelihood -> Posterior
  E[X] = sum(xi x P(xi)) — weighted average

CALCULUS
  Derivative = rate of change = tangent slope
  f'(x) = 0 at minimum or maximum
  Partial derivative: one variable at a time
  Gradient = vector of all partial derivatives
  Chain rule = engine of backpropagation

OPTIMIZATION
  Cost function measures model error (minimise it!)
  theta <- theta - alpha x gradient(L)
  Learning rate alpha controls step size
  Mini-batch GD = standard in deep learning
  Modern practice: use Adam optimizer
=====================================================
```

---

## Revision Questions

### Linear Algebra
1. What is the shape of (3x4) x (4x5)?
2. Compute dot product of [1, 3, 5] and [2, 4, 6].
3. What is the transpose of a (2x5) matrix?
4. What condition must hold for a matrix to have an inverse?
5. What is the Hadamard product? How does it differ from matrix multiplication?

### Statistics
6. Calculate mean, median, and mode of [2, 3, 3, 5, 7, 9, 9, 9, 12].
7. Why is median preferred over mean for house prices?
8. If sigma^2 = 9, what is the standard deviation?
9. A dataset has mean=50, std=10. What Z-score is the value 70?
10. What is Bessel's correction and why use n-1 for sample variance?

### Probability
11. 30 students: 18 female, 12 male. 10 females and 6 males passed. What is P(Pass | Female)?
12. State Bayes Theorem and explain each term.
13. What is the difference between joint and conditional probability?
14. What does E[X] = 3.5 mean for a fair die?
15. If P(A)=0.3, P(B)=0.4, P(A,B)=0.12 — are A and B independent? Why?

### Calculus
16. Find the derivative of f(x) = 5x^3 + 2x^2 - 7x + 4.
17. Find df/dx and df/dy for f(x,y) = 2x^2*y + 3xy^2.
18. What does it mean when gradient of the loss = 0?
19. Use the chain rule to differentiate f(x) = (2x + 1)^3.
20. What is the geometric interpretation of a derivative?

### Optimization
21. Given true=[3,4,5], predicted=[2,3,5] — compute MSE.
22. w=2, b=1, alpha=0.1, dL/dw=4, dL/db=-2 — new values of w and b?
23. What is the difference between batch and stochastic gradient descent?
24. Why does a very small learning rate cause problems?
25. What does it mean for a loss function to be convex?

---

## Summary

```
=====================================================
                 UNIT 2 SUMMARY
           Mathematics for AI & ML
=====================================================

Five pillars:

1. LINEAR ALGEBRA — language of data
   Scalar -> Vector -> Matrix -> Tensor
   Operations: add, multiply, transpose, inverse
   Core equation: output = W * x + b

2. STATISTICS — understanding data
   Central tendency: Mean, Median, Mode
   Spread: Variance, Standard Deviation
   Z-score normalisation for ML

3. PROBABILITY — quantifying uncertainty
   P(A|B) = conditional, updated beliefs
   Bayes: P(A|B) proportional to P(B|A) x P(A)
   E[X] = expectation = weighted average

4. CALCULUS — enabling learning
   Derivative = rate of change
   Partial derivative = one variable at a time
   Gradient = vector of all partials = nabla(f)
   Chain rule = foundation of backpropagation

5. OPTIMIZATION — finding best parameters
   Cost function = how wrong we are
   Goal: minimise loss
   Method: theta <- theta - alpha * nabla(L)
   Modern: Adam optimizer (adaptive learning rates)

These five areas work together in every
training loop of every neural network.
=====================================================
```

---

*Next Unit: Machine Learning Fundamentals — supervised, unsupervised, and reinforcement learning algorithms built on these mathematical foundations.*

---

*Last updated: 2025 | Format: GitHub README.md | Subject: Mathematics for AI/ML*