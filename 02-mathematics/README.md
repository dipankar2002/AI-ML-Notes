# Unit 2: Mathematics for AI/ML

> 📁 **Repository:** `ai-ml-learning/02-mathematics/README.md`
> 🎯 **Phase:** 1 — Foundations
> 🧩 **Unit:** 2 of 36

---

## 📌 Overview

Every ML algorithm — from the simplest linear regression to the largest language model — is built on mathematics. When you call `model.fit()`, you are running thousands of matrix multiplications, computing derivatives, and descending a gradient. If you do not understand the mathematics, you are treating a Ferrari engine as a black box.

This unit covers the three mathematical pillars of AI/ML:

```
LINEAR ALGEBRA  →  How data is stored and transformed
CALCULUS        →  How models learn (what direction to change weights)
OPTIMIZATION    →  How models find the best parameters
```

The goal is not to make you a pure mathematician. The goal is to give you enough mathematical intuition that you understand **why** algorithms work, **what** is happening inside them, and **how** to debug them when they fail.

> **Learning philosophy for this unit:**
> Intuition first. Notation second. Always connect the math back to what it means for an actual ML model.

---

## 🎯 Learning Objectives

By the end of this unit, you should be able to:

- [ ] Represent and manipulate data as scalars, vectors, and matrices
- [ ] Perform and understand matrix operations including multiplication, transpose, and inverse
- [ ] Explain eigenvalues and eigenvectors and their role in ML (PCA)
- [ ] Understand derivatives as rates of change and compute them
- [ ] Compute partial derivatives for multi-variable functions
- [ ] Understand the chain rule and why it is the engine of backpropagation
- [ ] Explain what a gradient is and compute gradient vectors
- [ ] Define loss functions and understand what minimising them means
- [ ] Walk through gradient descent step by step with numbers
- [ ] Distinguish between batch, stochastic, and mini-batch gradient descent
- [ ] Implement gradient descent from scratch in Python

---

## 🗺️ Prerequisites

- Unit 1: Introduction to AI, ML & Data Science
- Basic algebra (variables, equations, functions)
- No calculus background required — everything will be built from first principles

---

## 📚 Topics Covered

**Part A — Linear Algebra**
1. [Scalars](#1-scalars)
2. [Vectors](#2-vectors)
3. [Vector Operations](#3-vector-operations)
4. [Dot Product](#4-dot-product)
5. [Matrices](#5-matrices)
6. [Matrix Operations](#6-matrix-operations)
7. [Matrix Multiplication](#7-matrix-multiplication)
8. [Transpose](#8-transpose)
9. [Inverse](#9-inverse)
10. [Identity Matrix](#10-identity-matrix)
11. [Determinants](#11-determinants)
12. [Linear Transformations](#12-linear-transformations)
13. [Eigenvalues and Eigenvectors](#13-eigenvalues-and-eigenvectors)

**Part B — Calculus**

14. [Functions](#14-functions)
15. [Limits](#15-limits)
16. [Derivatives](#16-derivatives)
17. [Partial Derivatives](#17-partial-derivatives)
18. [Chain Rule](#18-chain-rule)
19. [Gradients and Gradient Vectors](#19-gradients-and-gradient-vectors)

**Part C — Optimization**

20. [Objective, Loss, and Cost Functions](#20-objective-loss-and-cost-functions)
21. [Gradient Descent](#21-gradient-descent)
22. [Learning Rate](#22-learning-rate)
23. [Local and Global Minima](#23-local-and-global-minima)
24. [Batch, Stochastic, and Mini-batch Gradient Descent](#24-batch-stochastic-and-mini-batch-gradient-descent)

---

# PART A — LINEAR ALGEBRA

> **Why Linear Algebra?**
> Your dataset is a matrix. Your weights are a vector. Every prediction is a matrix multiplication. Every layer of a neural network is a linear transformation followed by a non-linear activation. Linear algebra IS the language of data.

---

## 1. Scalars

### What is it?

A **scalar** is a single number — the simplest mathematical object. It has **magnitude only** — no direction, no structure.

```
Examples:
  5        →  a scalar (integer)
  3.14     →  a scalar (float)
  -7.2     →  a scalar (negative)
  0.001    →  a scalar (learning rate in ML)
  0.87     →  a scalar (accuracy of a model)
```

### Why do we need it?

Scalars appear constantly in ML:
- **Learning rate:** α = 0.01 — controls how fast a model learns
- **Loss value:** L = 0.34 — a single number measuring how wrong the model is
- **Prediction confidence:** p = 0.92 — the model's confidence in a class
- **A pixel value in grayscale:** 128 — a single intensity value

### Notation

In ML literature, scalars are written as **lowercase italic** letters: *x*, *y*, *α*, *λ*, *n*

### Key Takeaway

> A scalar is just a number. But in ML, scalars carry critical meaning — your model's error, its learning speed, its confidence. Learn to recognise when a result should be a scalar.

---

## 2. Vectors

### What is it?

A **vector** is an **ordered list of numbers** (scalars). It has both **magnitude** (size) and **direction**.

```
Column vector (default in ML):      Row vector:
      ┌ 1 ┐
      │ 2 │                         v = [ 1  2  3  4 ]
v  =  │ 3 │
      └ 4 ┘

Shape: (4, 1)                       Shape: (1, 4)
```

An **n-dimensional vector** has n components and represents a point in n-dimensional space.

### Why do we need it?

Vectors are the fundamental unit of data representation in ML:

| ML Context | Vector Represents |
|---|---|
| One row in a dataset | A single data point (all features of one sample) |
| Word embedding | "king" → [0.25, −0.41, 0.80, ..., 0.12] (300 numbers) |
| Model weights | The parameters of one layer |
| An image (flattened) | 28×28 image → 784-element vector |
| Output probabilities | [0.02, 0.95, 0.03] (probability per class) |

### Intuition

Think of a 2D vector as an arrow from the origin to a point:

```
y
│
4 │        ● (3, 4)
│       /
│      /   ← vector v = [3, 4]
│     /
│    /
│   /
0 ├──────────── x
   0  1  2  3  4

Magnitude: ||v|| = √(3² + 4²) = √25 = 5
Direction: the angle θ = arctan(4/3) ≈ 53°
```

### Notation

Vectors are written as **lowercase bold** letters: **v**, **w**, **x**, **b**

Or with an arrow: $\vec{v}$

### Vector Magnitude (Length / L2 Norm)

The **magnitude** (also called the **L2 norm** or **Euclidean length**) of a vector measures its size:

```
Formula:
  ||v|| = √(v₁² + v₂² + v₃² + ... + vₙ²)
         = √(Σᵢ vᵢ²)

Example:
  v = [3, 4]
  ||v|| = √(3² + 4²) = √(9 + 16) = √25 = 5
```

**In ML:** The norm is used in regularisation (L2 regularisation = penalising large weight vectors).

### Python Implementation

```python
import numpy as np

# Creating vectors
v = np.array([1, 2, 3, 4])          # 1D array = vector
w = np.array([4, 3, 2, 1])

print(f"v = {v}")                    # [1 2 3 4]
print(f"Shape: {v.shape}")           # (4,)
print(f"Dimensions: {v.ndim}")       # 1

# Column vector (explicitly)
col_vec = np.array([[1], [2], [3]])  # Shape: (3, 1)
print(f"Column vector shape: {col_vec.shape}")

# Magnitude (L2 norm)
magnitude = np.linalg.norm(v)
print(f"||v|| = {magnitude:.4f}")    # 5.4772

# Manual calculation
mag_manual = np.sqrt(np.sum(v**2))
print(f"Manual ||v|| = {mag_manual:.4f}")  # Same result
```

---

## 3. Vector Operations

### 3.1 Vector Addition and Subtraction

Add or subtract element by element. Both vectors must have the **same length**.

```
Formula:
  a + b = [a₁+b₁, a₂+b₂, ..., aₙ+bₙ]

Example:
  a = [1, 2, 3]
  b = [4, 5, 6]
  a + b = [1+4, 2+5, 3+6] = [5, 7, 9]
```

**In ML:** Adding bias vectors to weight outputs: `z = Wx + b`

### 3.2 Scalar Multiplication

Multiply every element by a scalar.

```
Formula:
  c · v = [c·v₁, c·v₂, ..., c·vₙ]

Example:
  v = [1, 2, 3],  c = 3
  3v = [3, 6, 9]
```

**In ML:** The learning rate multiplied by the gradient: `α · ∇L`

### 3.3 Element-wise Multiplication (Hadamard Product)

Multiply corresponding elements. Denoted ⊙.

```
Formula:
  a ⊙ b = [a₁·b₁, a₂·b₂, ..., aₙ·bₙ]

Example:
  a = [1, 2, 3]
  b = [4, 5, 6]
  a ⊙ b = [4, 10, 18]
```

**In ML:** LSTM gates, dropout masks, attention mechanisms.

### Python Implementation

```python
import numpy as np

a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])

print(f"a + b = {a + b}")          # [5. 7. 9.]
print(f"a - b = {a - b}")          # [-3. -3. -3.]
print(f"3 * a = {3 * a}")          # [3. 6. 9.]
print(f"a ⊙ b = {a * b}")          # [ 4. 10. 18.]  (element-wise in numpy)
```

---

## 4. Dot Product

### What is it?

The **dot product** (inner product) of two vectors produces a **single scalar**. It is arguably the most important operation in all of ML.

### Formula

```
a · b = a₁b₁ + a₂b₂ + ... + aₙbₙ = Σᵢ aᵢbᵢ

Example:
  a = [1, 2, 3]
  b = [4, 5, 6]
  a · b = (1×4) + (2×5) + (3×6) = 4 + 10 + 18 = 32
```

### Geometric Interpretation

```
Formula with angle:
  a · b = ||a|| × ||b|| × cos(θ)

Where θ is the angle between the vectors.

cos(θ) = 1    → θ = 0°    → vectors point in the same direction → max dot product
cos(θ) = 0    → θ = 90°   → vectors are perpendicular → dot product = 0
cos(θ) = -1   → θ = 180°  → vectors point in opposite directions → min dot product
```

**This means the dot product measures similarity:**
- Large positive value → vectors are similar (point in same direction)
- Zero → vectors are completely unrelated
- Large negative value → vectors are dissimilar (opposite)

### Why is the Dot Product Critical in ML?

The dot product is the core computation of neural networks. A single neuron computes:

```
output = dot(weights, input) + bias

  Input:    x = [x₁, x₂, x₃]       (three features)
  Weights:  w = [w₁, w₂, w₃]       (three learned parameters)
  Bias:     b  (a scalar)

  Neuron output = w₁x₁ + w₂x₂ + w₃x₃ + b = w·x + b

This is just the dot product of w and x, plus a bias.
```

Every neuron in every neural network computes a dot product. A layer of 100 neurons with 50 inputs computes 100 dot products — which is one matrix multiplication.

**Also used for:**
- Similarity search in vector databases (cosine similarity uses dot product)
- Attention scores in Transformers: `score = query · key`
- Collaborative filtering in recommendation systems

### Python

```python
import numpy as np

a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])

# Three equivalent ways to compute dot product
dot1 = np.dot(a, b)
dot2 = a @ b                 # @ is matrix multiply / dot product
dot3 = np.sum(a * b)         # element-wise multiply then sum

print(f"Dot product: {dot1}")   # 32.0

# Cosine similarity (normalised dot product)
cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
print(f"Cosine similarity: {cos_sim:.4f}")  # 0.9746 (very similar vectors)

# Neural network neuron computation
x = np.array([0.5, 0.3, 0.8])   # Input features
w = np.array([0.2, 0.4, 0.9])   # Learned weights
b = 0.1                           # Bias

neuron_output = np.dot(w, x) + b
print(f"Neuron output: {neuron_output:.4f}")  # 0.1*0.2 + ...
```

---

## 5. Matrices

### What is it?

A **matrix** is a **2-dimensional grid of numbers** arranged in rows and columns.

```
        Col 1  Col 2  Col 3
Row 1 → [  1     2     3  ]
Row 2 → [  4     5     6  ]
Row 3 → [  7     8     9  ]

This is a 3×3 matrix (3 rows, 3 columns)
Notation: A ∈ ℝ^(m×n) means A is a matrix with m rows and n columns
Element at row i, col j is written Aᵢⱼ
```

### Why do we need it?

Matrices are the fundamental data structure of ML:

| ML Concept | Matrix |
|---|---|
| **Dataset** | Each row = one sample, each column = one feature |
| **Neural net weights** | Weight matrix connecting two layers |
| **Image** | Pixel intensity values in a 2D grid |
| **Batch of inputs** | Multiple samples stacked as rows |
| **Embeddings** | Vocabulary × embedding_dim matrix |

**Example — Dataset as a matrix:**

```
Features:   Age   Income   Years_Experience
Sample 1  [ 25,   50000,       2          ]
Sample 2  [ 35,   80000,       8          ]
Sample 3  [ 28,   60000,       4          ]

X = 3×3 matrix (3 samples, 3 features)
```

### Special Matrices

**Square matrix:** m = n (same rows and columns)

**Zero matrix (0):** All elements are 0

**Diagonal matrix:** Non-zero values only on the main diagonal
```
D = [5  0  0]
    [0  3  0]
    [0  0  7]
```

**Symmetric matrix:** A = Aᵀ (equals its own transpose)
Covariance matrices in statistics are always symmetric.

### Notation

Matrices are written as **uppercase bold** letters: **A**, **W**, **X**, **B**

### Python

```python
import numpy as np

# Creating matrices
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

print(f"Shape: {A.shape}")          # (3, 3)
print(f"Rows: {A.shape[0]}")        # 3
print(f"Cols: {A.shape[1]}")        # 3
print(f"Element A[1,2] = {A[1,2]}") # 6  (row 1, col 2, 0-indexed)
print(f"Row 0: {A[0, :]}")          # [1 2 3]
print(f"Col 2: {A[:, 2]}")          # [3 6 9]

# Dataset as a matrix
X_data = np.array([[25, 50000, 2],
                   [35, 80000, 8],
                   [28, 60000, 4]])
y_labels = np.array([0, 1, 0])      # Target: hired or not

print(f"\nDataset shape: {X_data.shape}")  # (3, 3)
```

---

## 6. Matrix Operations

### 6.1 Addition and Subtraction

Matrices of the **same shape** are added element by element.

```
A + B: add corresponding elements

A = [1  2]    B = [5  6]    A + B = [6   8]
    [3  4]        [7  8]             [10  12]
```

### 6.2 Scalar Multiplication

Multiply every element by the scalar.

```
3 × A = 3 × [1  2]  =  [3   6]
             [3  4]     [9  12]
```

### 6.3 Element-wise Multiplication (Hadamard)

Same shape required. Multiply corresponding elements (NOT the same as matrix multiplication).

```
A ⊙ B = [1×5  2×6]  =  [5   12]
         [3×7  4×8]     [21  32]
```

**In ML:** Used in LSTM gates and attention masking.

### Python

```python
import numpy as np

A = np.array([[1, 2], [3, 4]], dtype=float)
B = np.array([[5, 6], [7, 8]], dtype=float)

print("A + B:\n", A + B)      # Element-wise addition
print("A - B:\n", A - B)      # Element-wise subtraction
print("3 * A:\n", 3 * A)      # Scalar multiplication
print("A ⊙ B (Hadamard):\n", A * B)   # Element-wise multiply

# NOTE: A * B in NumPy is Hadamard (element-wise), NOT matrix multiply
# For matrix multiply use: A @ B or np.matmul(A, B)
```

---

## 7. Matrix Multiplication

### What is it?

Matrix multiplication combines two matrices into a new matrix. It is **not** element-wise — it is a structured combination of dot products.

### The Rule

To multiply **A (m × n)** by **B (n × p)**:
- The **inner dimensions must match**: columns of A = rows of B = n
- The result has shape **(m × p)**

```
(m × n) × (n × p) = (m × p)
   A          B        C
```

**Memory trick:** Think of it as (m × **n**) × (**n** × p) — the bold n's cancel, leaving (m × p).

### How to Compute Each Element

Each element **Cᵢⱼ** = **dot product** of row i of A with column j of B.

```
Example: A (2×3) × B (3×2) = C (2×2)

A = [1  2  3]        B = [7   8]
    [4  5  6]            [9  10]
                         [11 12]

C[0,0] = row 0 of A · col 0 of B = [1,2,3]·[7,9,11] = 7+18+33 = 58
C[0,1] = row 0 of A · col 1 of B = [1,2,3]·[8,10,12] = 8+20+36 = 64
C[1,0] = row 1 of A · col 0 of B = [4,5,6]·[7,9,11] = 28+45+66 = 139
C[1,1] = row 1 of A · col 1 of B = [4,5,6]·[8,10,12] = 32+50+72 = 154

     C = [ 58   64]
         [139  154]
```

### Why Matrix Multiplication = Neural Networks

A neural network **layer** is:

```
output = activation( W × input + b )

W  = weight matrix   shape: (n_neurons, n_inputs)
x  = input vector    shape: (n_inputs, 1)
b  = bias vector     shape: (n_neurons, 1)
Wx = matrix multiply shape: (n_neurons, 1)

For a batch of 32 inputs at once:
x  = input matrix    shape: (n_inputs, 32)
Wx                   shape: (n_neurons, 32)

Every forward pass through a neural network = a sequence of matrix multiplications.
```

### Properties

| Property | Formula | Note |
|---|---|---|
| **NOT commutative** | AB ≠ BA | Order matters! |
| **Associative** | (AB)C = A(BC) | Can regroup |
| **Distributive** | A(B+C) = AB + AC | Works like regular algebra |
| **Dimension must match** | (m×n)(n×p) = (m×p) | Inner dims must be equal |

### Python

```python
import numpy as np

A = np.array([[1, 2, 3],
              [4, 5, 6]])    # Shape: (2, 3)

B = np.array([[7,  8],
              [9,  10],
              [11, 12]])     # Shape: (3, 2)

# Matrix multiplication
C = A @ B                    # Shape: (2, 2)
print("A @ B:\n", C)
# [[ 58  64]
#  [139 154]]

# Equivalent: np.matmul(A, B)
C2 = np.matmul(A, B)
print("\nSame result:", np.allclose(C, C2))   # True

# ── Neural network layer ──────────────────────────────────────────
# Single input
x = np.array([[0.5], [0.3], [0.8]])   # Input: 3 features → shape (3,1)
W = np.array([[0.2, 0.4, 0.9],        # Weights: 2 neurons × 3 inputs → (2,3)
              [0.1, 0.7, 0.3]])
b = np.array([[0.1], [0.2]])           # Bias: 2 neurons → (2,1)

z = W @ x + b                          # (2,3) @ (3,1) + (2,1) = (2,1)
print(f"\nNeuron outputs: {z.flatten()}")

# Batch of 4 inputs at once (much more efficient)
X_batch = np.random.randn(3, 4)        # 3 features, 4 samples
Z_batch = W @ X_batch + b              # (2,3) @ (3,4) + (2,1) = (2,4)
print(f"Batch output shape: {Z_batch.shape}")  # (2, 4)
```

---

## 8. Transpose

### What is it?

The **transpose** of matrix A (written Aᵀ) flips the matrix over its main diagonal — rows become columns, columns become rows.

```
A = [1  2  3]          Aᵀ = [1  4]
    [4  5  6]               [2  5]
                            [3  6]

Shape (2, 3) → Shape (3, 2)

Rule: If A has shape (m, n), then Aᵀ has shape (n, m)
Element rule: (Aᵀ)ᵢⱼ = Aⱼᵢ
```

### Why do we need it?

Transpose is used constantly in ML:

1. **Converting vectors:** Column vector (n,1) ↔ Row vector (1,n)
2. **Backpropagation:** Gradients flow backward through the transpose of weight matrices
3. **Covariance matrix:** Σ = XᵀX
4. **Normal equation:** θ = (XᵀX)⁻¹Xᵀy
5. **Attention mechanism in Transformers:** scores = Q × Kᵀ

### Properties

```
(Aᵀ)ᵀ = A               (double transpose = original)
(A + B)ᵀ = Aᵀ + Bᵀ       (distributes over addition)
(AB)ᵀ = BᵀAᵀ             (ORDER REVERSES — critical for backprop!)
(cA)ᵀ = cAᵀ              (scalar passes through)
```

### Python

```python
import numpy as np

A = np.array([[1, 2, 3],
              [4, 5, 6]])    # Shape (2, 3)

AT = A.T                     # Transpose — Shape (3, 2)
print("A:\n", A)
print("\nA.T:\n", AT)

# Verify (AB)ᵀ = BᵀAᵀ
B = np.random.randn(3, 4)
AB_T = (A @ B).T
BT_AT = B.T @ A.T
print("\n(AB)ᵀ == BᵀAᵀ:", np.allclose(AB_T, BT_AT))   # True

# Convert column vector to row and back
col = np.array([[1], [2], [3]])   # (3,1)
row = col.T                        # (1,3)
print(f"\nColumn {col.shape} → Row {row.shape}")

# Covariance matrix (used in PCA)
X = np.random.randn(100, 5)      # 100 samples, 5 features
X_centred = X - X.mean(axis=0)
cov = X_centred.T @ X_centred / (100 - 1)   # XᵀX / (n-1)
print(f"Covariance matrix shape: {cov.shape}")   # (5, 5) — always square
```

---

## 9. Inverse

### What is it?

The **inverse** of a square matrix A (written A⁻¹) is the matrix that "undoes" A:

```
A × A⁻¹ = A⁻¹ × A = I   (the identity matrix)

Analogy:
  Numbers: 5 × (1/5) = 1
  Matrices: A × A⁻¹ = I
```

### When Does the Inverse Exist?

A matrix is **invertible** (non-singular) if and only if:
- It is **square** (m = n)
- Its **determinant ≠ 0**

A **singular matrix** (det = 0) has no inverse.

### Why Do We Need It in ML?

The **Normal Equation** for linear regression uses the inverse:

```
Optimal weights: θ = (XᵀX)⁻¹ Xᵀy

X = feature matrix   (n_samples × n_features)
y = target vector    (n_samples × 1)
θ = weight vector    (n_features × 1)

This directly solves linear regression in one step.
Problem: Computing (XᵀX)⁻¹ is expensive for large matrices.
Solution: Use gradient descent instead (covered in Part C).
```

### 2×2 Inverse Formula

```
A = [a  b]        A⁻¹ =   1    ×  [ d  -b]
    [c  d]               det(A)    [-c   a]

Where det(A) = ad - bc
```

### Python

```python
import numpy as np

A = np.array([[3., 2.],
              [1., 4.]])

A_inv = np.linalg.inv(A)
print("A⁻¹:\n", A_inv)

# Verify A × A⁻¹ = I
product = A @ A_inv
print("\nA @ A⁻¹:\n", np.round(product, 10))   # Should be [[1,0],[0,1]]

# ── Normal Equation for Linear Regression ─────────────────────────
# Generate synthetic data: y = 2x + 3 + noise
np.random.seed(42)
n = 100
x_vals = np.random.randn(n)
y_vals = 2 * x_vals + 3 + 0.5 * np.random.randn(n)

# Build feature matrix X with a bias column (column of 1s)
X = np.column_stack([np.ones(n), x_vals])   # Shape (100, 2)
y = y_vals.reshape(-1, 1)                    # Shape (100, 1)

# Normal Equation: θ = (XᵀX)⁻¹ Xᵀy
theta = np.linalg.inv(X.T @ X) @ X.T @ y
print(f"\nNormal Equation results:")
print(f"  Intercept (should be ~3): {theta[0, 0]:.4f}")
print(f"  Slope (should be ~2):     {theta[1, 0]:.4f}")

# Better: use np.linalg.solve instead of computing inverse explicitly
theta2 = np.linalg.solve(X.T @ X, X.T @ y)
print(f"\nnp.linalg.solve (numerically stable):")
print(f"  Intercept: {theta2[0, 0]:.4f}")
print(f"  Slope:     {theta2[1, 0]:.4f}")
```

---

## 10. Identity Matrix

### What is it?

The **identity matrix** (denoted **I** or **Iₙ**) is the square matrix with 1s on the main diagonal and 0s everywhere else.

```
I₂ = [1  0]      I₃ = [1  0  0]
     [0  1]           [0  1  0]
                       [0  0  1]
```

It is the matrix equivalent of the number 1 — multiplying any matrix by the identity matrix gives back the original matrix:

```
A × I = I × A = A
```

### Why Do We Need It?

- **Defining the inverse:** A × A⁻¹ = I
- **Initialising weight matrices** in certain architectures
- **Regularisation:** The matrix λI appears in Ridge regression (L2 regularisation):

```
Ridge Regression Normal Equation:
  θ = (XᵀX + λI)⁻¹ Xᵀy

The λI term ensures XᵀX + λI is always invertible,
even when XᵀX is singular.
```

### Python

```python
import numpy as np

I3 = np.eye(3)     # 3×3 identity matrix
print("I₃:\n", I3)

A = np.array([[2., 3.], [1., 4.]])

# A × I = A
print("\nA @ I == A:", np.allclose(A @ np.eye(2), A))   # True

# Ridge regression: (XᵀX + λI)⁻¹ Xᵀy
X = np.random.randn(10, 3)   # 10 samples, 3 features
y = np.random.randn(10, 1)
lam = 0.1                     # regularisation strength

XTX = X.T @ X                 # (3,3)
ridge_theta = np.linalg.inv(XTX + lam * np.eye(3)) @ X.T @ y
print(f"Ridge weights: {ridge_theta.flatten()}")
```

---

## 11. Determinants

### What is it?

The **determinant** of a square matrix is a single scalar that captures fundamental properties of the matrix — specifically, whether it is invertible and how it scales space.

### Formula for 2×2

```
A = [a  b]
    [c  d]

det(A) = ad - bc

Example:
  A = [3  2]
      [1  4]

  det(A) = (3×4) - (2×1) = 12 - 2 = 10
```

### Geometric Meaning

The determinant tells you how much the matrix **scales area** (in 2D) or **volume** (in 3D):

```
|det(A)| = scale factor of the transformation A

det(A) = 0  → The matrix squashes space into a lower dimension
              (parallel columns/rows)
              → Matrix is NOT invertible (singular)

det(A) > 0  → Transformation preserves orientation
det(A) < 0  → Transformation flips orientation
|det(A)| > 1 → Transformation expands space
|det(A)| < 1 → Transformation contracts space
```

### Why Do We Need It in ML?

- **Checking invertibility:** If det = 0, the matrix has no inverse
- **Multivariate Gaussian:** The determinant of the covariance matrix appears in its formula
- **PCA and linear transformations:** Understanding how transformations change volume

### Python

```python
import numpy as np

A = np.array([[3., 2.],
              [1., 4.]])

det_A = np.linalg.det(A)
print(f"det(A) = {det_A}")   # 10.0 → invertible

# Singular matrix (no inverse)
B = np.array([[1., 2.],
              [2., 4.]])     # Row 2 = 2 × Row 1 (linearly dependent)
det_B = np.linalg.det(B)
print(f"det(B) = {det_B:.10f}")   # ~0 → NOT invertible

# Attempting to invert a singular matrix
try:
    np.linalg.inv(B)
except np.linalg.LinAlgError as e:
    print(f"Error: {e}")

# 3×3 determinant
C = np.array([[1., 2., 3.],
              [4., 5., 6.],
              [7., 2., 9.]])
print(f"det(C) = {np.linalg.det(C):.4f}")
```

---

## 12. Linear Transformations

### What is it?

A **linear transformation** is a function that maps vectors from one space to another while preserving two properties:

```
1. Additivity:    T(u + v) = T(u) + T(v)
2. Homogeneity:   T(cv) = cT(v)

Intuition: Straight lines stay straight. The origin stays at the origin.
```

Every matrix multiplication **IS** a linear transformation.

### Types of Transformations

```
Scaling:              Rotation:             Reflection:
[2  0] × [x] = [2x]  [cos θ  -sin θ]      [1   0] × [x] = [ x]
[0  2]   [y]   [2y]  [sin θ   cos θ]      [0  -1]   [y]   [-y]

Shearing:             Projection:
[1  1] × [x] = [x+y]  [1  0] × [x] = [x]
[0  1]   [y]   [ y ]   [0  0]   [y]   [0]
```

### Why This Matters for ML

Every layer of a neural network applies a linear transformation (matrix multiplication) followed by a non-linear activation function:

```
Layer output = activation(W × input + b)

W = linear transformation  (rotation, scaling, shearing of data)
activation = non-linear step (allows the network to learn complex shapes)
```

Without non-linear activations, stacking multiple linear transformations is still just one linear transformation (you can always combine them into one matrix). Non-linearity is what gives neural networks their expressive power.

### Python: Visualising a Transformation

```python
import numpy as np
import matplotlib.pyplot as plt

# Original grid of points
points = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]).T   # Shape (2, 5)

# Transformation matrices
scaling    = np.array([[2, 0], [0, 1]])    # Stretch horizontally by 2
rotation   = np.array([[0, -1], [1, 0]])  # Rotate 90° counterclockwise
shearing   = np.array([[1, 1], [0, 1]])   # Shear horizontally

fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
transforms = [np.eye(2), scaling, rotation, shearing]
titles = ['Original', 'Scaling (×2 horiz)', 'Rotation (90°)', 'Shearing']

for ax, T, title in zip(axes, transforms, titles):
    transformed = T @ points
    ax.fill(transformed[0], transformed[1], alpha=0.3)
    ax.plot(transformed[0], transformed[1], 'b-o')
    ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.5, 2.5)
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axvline(0, color='k', linewidth=0.5)
    ax.set_title(title); ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('linear_transformations.png', dpi=100, bbox_inches='tight')
plt.show()
print("Saved: linear_transformations.png")
```

---

## 13. Eigenvalues and Eigenvectors

### What is it?

When a matrix A transforms a vector, it usually changes both the direction and magnitude. But some special vectors — called **eigenvectors** — **only get scaled**, not rotated. The amount they get scaled by is the **eigenvalue**.

### Formal Definition

```
A × v = λ × v

Where:
  A  = square matrix
  v  = eigenvector (a non-zero vector)
  λ  = eigenvalue (a scalar)

This says: "When matrix A is applied to eigenvector v,
            the result is just v scaled by λ."
```

### Intuition

```
Imagine a 2D transformation (matrix A) that:
  → Stretches horizontally by 3
  → Leaves vertical unchanged

The horizontal direction is an eigenvector (λ = 3)
  → A × [1,0] = [3,0] = 3 × [1,0]

The vertical direction is an eigenvector (λ = 1)
  → A × [0,1] = [0,1] = 1 × [0,1]

Every other direction gets rotated AND scaled.
Only eigenvector directions get scaled without rotation.
```

### How to Find Eigenvalues

```
Solve the characteristic equation:
  det(A - λI) = 0

Example: A = [4  1]
              [2  3]

  det([4-λ   1 ]) = 0
      [ 2   3-λ]

  (4-λ)(3-λ) - (1)(2) = 0
  12 - 4λ - 3λ + λ² - 2 = 0
  λ² - 7λ + 10 = 0
  (λ - 5)(λ - 2) = 0

  Eigenvalues: λ₁ = 5, λ₂ = 2
```

### Why Eigenvalues and Eigenvectors Matter in ML

**Principal Component Analysis (PCA):**

PCA finds the directions of maximum variance in your data. These directions are exactly the **eigenvectors of the covariance matrix**. The corresponding **eigenvalues** tell you how much variance each direction explains.

```
Data (1000 samples, 50 features)
         ↓
Compute covariance matrix (50 × 50)
         ↓
Find eigenvectors and eigenvalues
         ↓
Sort by eigenvalue (largest = most variance)
         ↓
Keep top k eigenvectors → reduce to k dimensions
         ↓
Data now has 1000 samples, k features (instead of 50)
```

**Other ML uses:**
- **Google PageRank:** The importance of web pages is the principal eigenvector of the link matrix
- **Spectral clustering:** Graph-based clustering using eigenvectors of the Laplacian matrix
- **Understanding neural networks:** Analysing the Hessian matrix for optimisation insights

### Python

```python
import numpy as np

A = np.array([[4., 1.],
              [2., 3.]])

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"Eigenvalues:  {eigenvalues}")       # [5. 2.]
print(f"Eigenvectors:\n{eigenvectors}")     # columns are eigenvectors

# Verify: A × v = λ × v
for i, (lam, v) in enumerate(zip(eigenvalues, eigenvectors.T)):
    Av = A @ v
    lv = lam * v
    print(f"\nEigenvector {i+1}: {v}")
    print(f"A @ v   = {Av}")
    print(f"λ × v   = {lv}")
    print(f"Equal? {np.allclose(Av, lv)}")

# ── PCA using eigendecomposition ──────────────────────────────────
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data                         # (150, 4) — 4 features

# Step 1: Centre the data
X_mean = X - X.mean(axis=0)

# Step 2: Compute covariance matrix
cov = X_mean.T @ X_mean / (len(X) - 1)   # (4, 4)

# Step 3: Find eigenvectors of covariance matrix
eigenvals, eigenvecs = np.linalg.eig(cov)

# Step 4: Sort by eigenvalue (largest first)
idx = np.argsort(eigenvals)[::-1]
eigenvals = eigenvals[idx]
eigenvecs = eigenvecs[:, idx]

# Step 5: Project onto top 2 eigenvectors
X_pca = X_mean @ eigenvecs[:, :2]    # (150, 2)

# Variance explained
var_explained = eigenvals / eigenvals.sum()
print(f"\nPCA variance explained:")
for i, v in enumerate(var_explained):
    print(f"  PC{i+1}: {v:.1%}")
```

---

# PART B — CALCULUS

> **Why Calculus?**
> When a neural network makes a wrong prediction, it needs to know: "How should I adjust each weight to reduce the error?" The answer is: compute the derivative of the error with respect to each weight. Calculus gives us derivatives. Derivatives give us directions. Directions give us learning.

---

## 14. Functions

### What is it?

A **function** maps each input to exactly one output.

```
f: ℝⁿ → ℝᵐ    (maps n-dimensional input to m-dimensional output)

Simple scalar function:
  f(x) = x²             f(3) = 9,   f(-2) = 4

Loss function (the one we care about most):
  L(w₁, w₂, ..., wₙ) → ℝ   (takes all weights, outputs a single scalar loss)
```

### Key Functions in ML

| Function | Formula | ML Use |
|---|---|---|
| **Identity** | f(x) = x | Linear activation |
| **Sigmoid** | f(x) = 1/(1+e⁻ˣ) | Binary classification output |
| **ReLU** | f(x) = max(0, x) | Hidden layer activation |
| **Softmax** | f(xᵢ) = eˣⁱ / Σⱼeˣʲ | Multi-class output |
| **Loss** | L(ŷ, y) = function of predictions and targets | Training objective |

Understanding how these functions behave — and how their derivatives behave — is essential for understanding why neural networks are designed the way they are.

---

## 15. Limits

### What is it?

A **limit** describes the value a function approaches as the input approaches some value.

```
lim  f(x) = L
x→a

"As x gets closer and closer to a, f(x) gets closer to L"
```

### Why Do We Need It?

Limits are the foundation of derivatives. The derivative is defined as a limit:

```
f'(x) = lim  f(x + h) - f(x)
         h→0       h

"As h shrinks to zero, what does the ratio of change in output
 to change in input approach?"
```

This is not just notation — it is the precise definition of instantaneous rate of change.

### Intuition

```
Average speed over 2 hours:   (distance at 2h - distance at 0h) / 2
Average speed over 1 hour:    (distance at 1h - distance at 0h) / 1
Average speed over 0.1 hour:  (distance at 0.1h - distance at 0h) / 0.1
...
Instantaneous speed at t=0:   lim h→0 of (distance at h - distance at 0) / h
```

The limit gives us the instantaneous rate — the exact speed at one moment.

---

## 16. Derivatives

### What is it?

The **derivative** of a function measures its **instantaneous rate of change** at a point.

Geometrically, it is the **slope of the tangent line** to the function at that point.

```
f'(x) = df/dx = lim  f(x+h) - f(x)
                h→0       h
```

### Intuition

Imagine you are driving a car. The distance you have travelled is f(t). The derivative f'(t) is your **speedometer reading** — your instantaneous speed at time t.

```
f(t) = distance (going uphill and downhill on a hilly road)

f'(t) > 0   →   moving forward (positive slope)
f'(t) < 0   →   moving backward (negative slope)
f'(t) = 0   →   momentarily stopped (flat — a peak or valley)
```

In ML: the loss function L(w) has a landscape. The derivative tells us which direction is "uphill" (increasing loss) and which is "downhill" (decreasing loss). We want to go downhill.

### Common Derivatives

| Function | Derivative | Example |
|---|---|---|
| f(x) = c | f'(x) = 0 | f(x)=5 → f'(x)=0 |
| f(x) = xⁿ | f'(x) = nxⁿ⁻¹ | f(x)=x³ → f'(x)=3x² |
| f(x) = eˣ | f'(x) = eˣ | Derivative of eˣ is itself |
| f(x) = ln(x) | f'(x) = 1/x | |
| f(x) = sigmoid(x) | f'(x) = σ(x)(1-σ(x)) | Critical for backprop |
| f(x) = max(0,x) | f'(x) = 1 if x>0 else 0 | ReLU derivative |

### Small Numerical Example

```
f(x) = x²

Derivative: f'(x) = 2x

At x = 3:
  f'(3) = 2 × 3 = 6

This means: at x=3, if x increases by a tiny amount ε,
f(x) increases by approximately 6ε.

Verify numerically:
  f(3.001) = 3.001² = 9.006001
  f(3.000) = 3.000² = 9.000000
  Δf/Δx    ≈ (9.006001 - 9.000000) / 0.001 = 6.001  ≈ 6 ✓
```

### Python: Numerical Derivatives

```python
import numpy as np
import matplotlib.pyplot as plt

def numerical_derivative(f, x, h=1e-7):
    """Compute derivative numerically using the limit definition."""
    return (f(x + h) - f(x)) / h

# Test with f(x) = x²
f = lambda x: x**2
f_prime_exact = lambda x: 2*x          # Analytical derivative

x_test = 3.0
numerical = numerical_derivative(f, x_test)
exact = f_prime_exact(x_test)
print(f"Numerical derivative at x=3: {numerical:.6f}")  # 6.0
print(f"Exact derivative at x=3:     {exact:.6f}")      # 6.0

# Visualise function and tangent line at x=3
x = np.linspace(-1, 5, 100)
y = x**2

tangent_x = 3.0
tangent_y = tangent_x**2
slope = f_prime_exact(tangent_x)
tangent_line_y = slope * (x - tangent_x) + tangent_y

plt.figure(figsize=(8, 5))
plt.plot(x, y, 'b-', linewidth=2, label='f(x) = x²')
plt.plot(x, tangent_line_y, 'r--', linewidth=1.5, label=f'Tangent at x=3 (slope={slope})')
plt.scatter([tangent_x], [tangent_y], color='red', s=100, zorder=5)
plt.legend(); plt.grid(True, alpha=0.3)
plt.title("Function and Its Derivative (Tangent Line)")
plt.xlabel("x"); plt.ylabel("f(x)"); plt.ylim(-1, 25)
plt.savefig('derivative.png', dpi=100, bbox_inches='tight')
plt.show()

# Activation function derivatives
print("\nActivation function derivatives:")
x_val = np.array([-2., -1., 0., 1., 2.])

# Sigmoid and its derivative
sigmoid = lambda x: 1 / (1 + np.exp(-x))
sigmoid_prime = lambda x: sigmoid(x) * (1 - sigmoid(x))
print(f"Sigmoid at x=1: {sigmoid(1.):.4f}")        # 0.7311
print(f"Sigmoid' at x=1: {sigmoid_prime(1.):.4f}") # 0.1966

# ReLU and its derivative
relu = lambda x: np.maximum(0, x)
relu_prime = lambda x: (x > 0).astype(float)
print(f"\nReLU at [-2,-1,0,1,2]:  {relu(x_val)}")
print(f"ReLU' at [-2,-1,0,1,2]: {relu_prime(x_val)}")
```

---

## 17. Partial Derivatives

### What is it?

A **partial derivative** is the derivative of a function that has **multiple variables**, taken with respect to **one variable at a time** while holding all others constant.

```
Notation: ∂f/∂x  (read: "partial derivative of f with respect to x")
          The curly ∂ distinguishes partial from ordinary d
```

### Why Do We Need It?

A neural network's loss function L depends on **millions of weights** (w₁, w₂, ..., wₙ). To train the network, we need to know: "If I change weight wᵢ slightly, how does the loss change?"

That is exactly ∂L/∂wᵢ — the partial derivative of the loss with respect to weight i.

### Example

```
f(x, y) = x² + 3xy + y²

Partial derivative with respect to x (hold y constant):
  ∂f/∂x = 2x + 3y   (differentiate as if y is just a constant number)

Partial derivative with respect to y (hold x constant):
  ∂f/∂y = 3x + 2y   (differentiate as if x is just a constant number)

At point (x=2, y=1):
  ∂f/∂x = 2(2) + 3(1) = 7   → if x increases by ε, f increases by ~7ε
  ∂f/∂y = 3(2) + 2(1) = 8   → if y increases by ε, f increases by ~8ε
```

### Python

```python
import numpy as np

def f(x, y):
    return x**2 + 3*x*y + y**2

def numerical_partial(f, x, y, var='x', h=1e-7):
    """Compute partial derivative numerically."""
    if var == 'x':
        return (f(x + h, y) - f(x, y)) / h
    else:
        return (f(x, y + h) - f(x, y)) / h

x, y = 2.0, 1.0

df_dx = numerical_partial(f, x, y, var='x')
df_dy = numerical_partial(f, x, y, var='y')
print(f"∂f/∂x at (2,1): {df_dx:.4f}")  # 7.0
print(f"∂f/∂y at (2,1): {df_dy:.4f}")  # 8.0

# Analytical check
print(f"Exact ∂f/∂x = 2x + 3y = {2*x + 3*y}")  # 7.0
print(f"Exact ∂f/∂y = 3x + 2y = {3*x + 2*y}")  # 8.0
```

---

## 18. Chain Rule

### What is it?

The **chain rule** lets us differentiate **composed functions** — functions applied inside other functions.

```
If y = f(g(x)), then:

dy/dx = (df/dg) × (dg/dx)
      = f'(g(x)) × g'(x)

Intuition: The rate of change of y with respect to x
           = (how y changes with g) × (how g changes with x)
```

### Simple Example

```
y = (3x + 2)²

Let u = 3x + 2   →   y = u²

du/dx = 3
dy/du = 2u

Chain rule: dy/dx = dy/du × du/dx = 2u × 3 = 6u = 6(3x + 2)

At x = 1:  y = (3+2)² = 25
            dy/dx = 6(3×1 + 2) = 6 × 5 = 30
```

### Why the Chain Rule IS Backpropagation

A neural network is a **composition of many functions**:

```
Input x
  ↓
Layer 1: z₁ = W₁x + b₁,   a₁ = ReLU(z₁)
  ↓
Layer 2: z₂ = W₂a₁ + b₂,  a₂ = ReLU(z₂)
  ↓
Output layer: ŷ = sigmoid(W₃a₂ + b₃)
  ↓
Loss: L = CrossEntropy(ŷ, y)

L = CrossEntropy(sigmoid(W₃ × ReLU(W₂ × ReLU(W₁x + b₁) + b₂) + b₃), y)
```

To update W₁, we need ∂L/∂W₁. This requires applying the chain rule **repeatedly** through every layer:

```
∂L/∂W₁ = ∂L/∂ŷ × ∂ŷ/∂z₃ × ∂z₃/∂a₂ × ∂a₂/∂z₂ × ∂z₂/∂a₁ × ∂a₁/∂z₁ × ∂z₁/∂W₁

This is backpropagation — the chain rule applied backwards through the network.
```

### Python: Chain Rule in Action

```python
import numpy as np

# Manual chain rule for a 2-layer network
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    s = sigmoid(x)
    return s * (1 - s)

def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true)**2)

# Simple 2-layer network: x → z1 → a1 → z2 → ŷ → Loss
np.random.seed(42)
x = np.array([0.5, 0.3])   # 2 features, 1 sample
y = np.array([1.0])         # True label

# Weights
W1 = np.array([[0.2, 0.4], [0.1, 0.5]])   # (2, 2)
b1 = np.array([0.1, 0.1])
W2 = np.array([[0.3, 0.7]])                 # (1, 2)
b2 = np.array([0.1])

# ─── FORWARD PASS ─────────────────────────────────────────────
z1 = W1 @ x + b1           # Linear combination, layer 1
a1 = sigmoid(z1)            # Activation, layer 1
z2 = W2 @ a1 + b2          # Linear combination, layer 2
y_hat = sigmoid(z2)         # Output
loss = mse_loss(y_hat, y)   # Loss

print(f"Prediction: {y_hat[0]:.4f}")
print(f"Loss: {loss:.4f}")

# ─── BACKWARD PASS (Chain Rule) ──────────────────────────────
# ∂L/∂ŷ  (derivative of MSE loss)
dL_dyhat = 2 * (y_hat - y) / len(y)

# ∂ŷ/∂z2  (derivative of sigmoid at z2)
dyhat_dz2 = sigmoid_prime(z2)

# ∂z2/∂W2 = a1  (derivative of z2 = W2 @ a1 + b2 w.r.t. W2)
# ∂z2/∂a1 = W2

# Chain rule for W2:
dL_dz2 = dL_dyhat * dyhat_dz2           # ∂L/∂z2 = ∂L/∂ŷ × ∂ŷ/∂z2
dL_dW2 = np.outer(dL_dz2, a1)           # ∂L/∂W2

# Continue chain back to W1:
dL_da1 = W2.T @ dL_dz2                  # ∂L/∂a1
dL_dz1 = dL_da1 * sigmoid_prime(z1)     # ∂L/∂z1
dL_dW1 = np.outer(dL_dz1, x)            # ∂L/∂W1

print(f"\nGradient ∂L/∂W2:\n{dL_dW2}")
print(f"\nGradient ∂L/∂W1:\n{dL_dW1}")
```

---

## 19. Gradients and Gradient Vectors

### What is it?

The **gradient** of a multi-variable function is a **vector** containing all its partial derivatives — one per variable.

```
For f(x₁, x₂, ..., xₙ):

∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]
      ↑ ∂f with respect to each variable separately
      ↑ the result is a vector, not a scalar
```

### The Key Property of the Gradient

> **The gradient always points in the direction of steepest ascent.**

Conversely, **−∇f** points in the direction of **steepest descent**.

This is the geometric insight behind gradient descent:

```
Loss landscape (imagine hilly terrain):
        /\
       /  \
      /    \     ← gradient points uphill here
     /      \
    /        \_____ ← minimum (loss = lowest here)

To minimise loss: take a step in the -∇L direction
```

### Example

```
f(x, y) = x² + y²   (a bowl — minimum at origin)

∇f = [∂f/∂x, ∂f/∂y] = [2x, 2y]

At point (3, 4):
  ∇f = [6, 8]

This vector [6, 8] points AWAY from origin (uphill for a bowl)
-∇f = [-6, -8] points TOWARD origin (downhill — the minimum!)

Magnitude: ||∇f|| = √(6² + 8²) = 10
This tells us HOW STEEP the slope is at (3,4)
```

### Gradient of the Loss Function in ML

For a neural network with weights W and bias b:

```
∇L(W, b) = [∂L/∂W, ∂L/∂b]

This gradient tells us:
  "If we increase W by a tiny amount, loss increases by ∂L/∂W"
  "If we increase b by a tiny amount, loss increases by ∂L/∂b"

Gradient descent step:
  W_new = W - α × ∂L/∂W
  b_new = b - α × ∂L/∂b
```

### Python

```python
import numpy as np

def f(params):
    """f(x, y) = x² + y²"""
    x, y = params
    return x**2 + y**2

def analytical_gradient(params):
    """∇f = [2x, 2y]"""
    x, y = params
    return np.array([2*x, 2*y])

def numerical_gradient(f, params, h=1e-7):
    """Compute gradient numerically (finite differences)."""
    params = np.array(params, dtype=float)
    grad = np.zeros_like(params)
    for i in range(len(params)):
        params_plus  = params.copy(); params_plus[i]  += h
        params_minus = params.copy(); params_minus[i] -= h
        grad[i] = (f(params_plus) - f(params_minus)) / (2 * h)
    return grad

point = np.array([3.0, 4.0])

grad_analytical = analytical_gradient(point)
grad_numerical  = numerical_gradient(f, point)

print(f"Analytical gradient at (3,4): {grad_analytical}")  # [6. 8.]
print(f"Numerical  gradient at (3,4): {grad_numerical}")   # [6. 8.]
print(f"Gradient magnitude: {np.linalg.norm(grad_analytical):.2f}")  # 10.0

# The gradient always points uphill
# To minimise f, move in direction -∇f
print(f"Descent direction (-∇f): {-grad_analytical}")  # [-6. -8.]
```

---

# PART C — OPTIMIZATION

> **Why Optimization?**
> The entire process of "training" a model is an optimization problem. We have a loss function that measures how wrong our model is. We want to find the model parameters that make this loss as small as possible. Optimization is HOW we do it.

---

## 20. Objective, Loss, and Cost Functions

### The Vocabulary

These three terms are often used interchangeably. Here is the precise distinction:

| Term | Meaning | Example |
|---|---|---|
| **Objective function** | The general term for any function to be optimised | Could be maximised or minimised |
| **Loss function** | Measures error for **one sample** | Error on one training example |
| **Cost function** | Average of loss over the **entire training set** | Mean of losses over all examples |

In ML practice, "loss function" and "cost function" are used interchangeably. The distinction matters theoretically but rarely in code.

### Intuition

```
You are training a model to predict house prices.

One prediction:
  True price:      ₹50,00,000
  Predicted price: ₹45,00,000
  Error (loss):    ₹5,00,000

Over all houses in your dataset:
  Average error (cost): mean of all individual errors
  This is your COST FUNCTION value

Goal: Adjust the model parameters to make this average error as small as possible.
```

### Common Loss Functions

**1. Mean Squared Error (MSE) — Regression**

```
MSE = (1/n) Σᵢ (yᵢ - ŷᵢ)²

Where:
  n   = number of samples
  yᵢ  = true value for sample i
  ŷᵢ  = predicted value for sample i

Properties:
  → Penalises large errors heavily (squaring)
  → Always positive
  → Differentiable everywhere (important for gradient descent)
  → In the same units² as the target

Example:
  True:      [3.0, -0.5, 2.0, 7.0]
  Predicted: [2.5,  0.0, 2.0, 8.0]
  Errors:    [0.5, -0.5, 0.0, -1.0]
  Squared:   [0.25, 0.25, 0.0, 1.0]
  MSE:       (0.25 + 0.25 + 0 + 1.0) / 4 = 0.375
```

**2. Binary Cross-Entropy — Binary Classification**

```
BCE = -(1/n) Σᵢ [yᵢ log(ŷᵢ) + (1-yᵢ) log(1-ŷᵢ)]

Where:
  yᵢ  = true label (0 or 1)
  ŷᵢ  = predicted probability (between 0 and 1)
  log = natural logarithm

Intuition:
  If y=1 and ŷ=0.99: loss = -log(0.99) ≈ 0.01  (small — correct!)
  If y=1 and ŷ=0.01: loss = -log(0.01) ≈ 4.60  (large — wrong!)
  Logarithm severely punishes confident wrong predictions.
```

**3. Categorical Cross-Entropy — Multi-class Classification**

```
CCE = -(1/n) Σᵢ Σₖ yᵢₖ log(ŷᵢₖ)

Where:
  yᵢₖ = 1 if sample i belongs to class k, else 0 (one-hot encoding)
  ŷᵢₖ = predicted probability for class k
```

**4. Mean Absolute Error (MAE) — Robust Regression**

```
MAE = (1/n) Σᵢ |yᵢ - ŷᵢ|

Less sensitive to outliers than MSE (uses absolute value, not square)
Not differentiable at 0 — can cause issues with gradient descent
```

### Comparison Table

| Loss Function | Task | Outlier Sensitivity | Differentiable |
|---|---|---|---|
| **MSE** | Regression | High (squares errors) | Everywhere |
| **MAE** | Regression | Low | Not at 0 |
| **Huber** | Regression | Medium | Everywhere |
| **Binary Cross-Entropy** | Binary classification | Medium | Everywhere |
| **Categorical Cross-Entropy** | Multi-class | Medium | Everywhere |
| **Hinge** | SVM classification | Low | Not at 0 |

### Python

```python
import numpy as np

# ── Mean Squared Error ────────────────────────────────────────────
def mse(y_true, y_pred):
    """MSE = (1/n) Σ (y - ŷ)²"""
    return np.mean((y_true - y_pred) ** 2)

# ── Mean Absolute Error ───────────────────────────────────────────
def mae(y_true, y_pred):
    """MAE = (1/n) Σ |y - ŷ|"""
    return np.mean(np.abs(y_true - y_pred))

# ── Binary Cross-Entropy ──────────────────────────────────────────
def binary_cross_entropy(y_true, y_pred):
    """BCE = -(1/n) Σ [y·log(ŷ) + (1-y)·log(1-ŷ)]"""
    eps = 1e-15   # clip to avoid log(0)
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# ── Test all loss functions ───────────────────────────────────────
y_true_reg  = np.array([3.0, -0.5, 2.0, 7.0])
y_pred_reg  = np.array([2.5,  0.0, 2.0, 8.0])

y_true_cls  = np.array([1.0, 0.0, 1.0, 1.0])
y_pred_good = np.array([0.9, 0.1, 0.8, 0.95])   # good predictions
y_pred_bad  = np.array([0.1, 0.9, 0.2, 0.15])   # bad predictions

print("=== Regression Losses ===")
print(f"MSE:  {mse(y_true_reg, y_pred_reg):.4f}")    # 0.375
print(f"MAE:  {mae(y_true_reg, y_pred_reg):.4f}")    # 0.5

print("\n=== Classification Losses ===")
print(f"BCE (good predictions): {binary_cross_entropy(y_true_cls, y_pred_good):.4f}")  # small
print(f"BCE (bad predictions):  {binary_cross_entropy(y_true_cls, y_pred_bad):.4f}")   # large

# Scikit-learn loss functions
from sklearn.metrics import mean_squared_error, log_loss
print(f"\nsklearn MSE: {mean_squared_error(y_true_reg, y_pred_reg):.4f}")
print(f"sklearn BCE: {log_loss(y_true_cls, y_pred_good):.4f}")
```

---

## 21. Gradient Descent

### What is it?

**Gradient Descent** is the iterative optimization algorithm that minimises the loss function by repeatedly adjusting parameters in the direction of **steepest descent** (opposite to the gradient).

### Intuition

```
Imagine you are blindfolded on a hilly landscape.
Your goal: reach the lowest valley (minimum loss).

At each step:
  1. Feel the slope around you (compute gradient)
  2. Take a step in the downhill direction (subtract gradient)
  3. Repeat until you stop descending

Loss
 ┤  ●  ← starting point (bad model)
 ┤   ╲
 ┤    ╲  ← each step decreases loss
 ┤     ╲
 ┤      ╲_____●  ← converged (good model)
 └──────────────── parameters (weights)
```

### The Update Rule

```
w_new = w_old - α × ∂L/∂w

Where:
  w      = weight parameter (one number in a neural network)
  α      = learning rate (how big each step is)
  ∂L/∂w = gradient of loss w.r.t. weight (computed by chain rule)
  -      = MINUS because we want to go DOWNHILL (opposite gradient)

For all parameters simultaneously:
  θ_new = θ_old - α × ∇_θ L(θ)
```

### Step-by-Step Numerical Example

**Problem:** Find the w that minimises L(w) = w²

```
True answer: w = 0 (L(0) = 0 is the minimum)

Setup:
  Starting point: w = 10 (bad — L(10) = 100)
  Learning rate: α = 0.1

Derivative: ∂L/∂w = 2w

Iteration 1:
  gradient = 2 × 10 = 20
  w = 10 - 0.1 × 20 = 10 - 2 = 8
  L(8) = 64

Iteration 2:
  gradient = 2 × 8 = 16
  w = 8 - 0.1 × 16 = 8 - 1.6 = 6.4
  L(6.4) = 40.96

Iteration 3:
  w = 6.4 - 0.1 × 12.8 = 5.12
  L(5.12) = 26.21

...continuing...

Iteration 10:  w ≈ 1.07,  L ≈ 1.15
Iteration 20:  w ≈ 0.115, L ≈ 0.013
Iteration 50:  w ≈ 0.000, L ≈ 0.000  ← converged!
```

### Python: Gradient Descent from Scratch

```python
import numpy as np
import matplotlib.pyplot as plt

# ── FROM SCRATCH: Linear Regression via Gradient Descent ──────────
np.random.seed(42)

# Generate data: y = 3x + 2 + noise
n = 100
X = np.random.randn(n)
y = 3 * X + 2 + 0.5 * np.random.randn(n)

# Initialise parameters
w = 0.0   # slope (should learn ~3)
b = 0.0   # intercept (should learn ~2)
alpha = 0.01   # learning rate

# Storage for plotting
loss_history = []

for epoch in range(200):

    # ─ FORWARD PASS: predict ──────────────────────────────────────
    y_pred = w * X + b

    # ─ COMPUTE LOSS (MSE) ─────────────────────────────────────────
    loss = np.mean((y - y_pred) ** 2)
    loss_history.append(loss)

    # ─ COMPUTE GRADIENTS ──────────────────────────────────────────
    # ∂L/∂w = (-2/n) Σ xᵢ(yᵢ - ŷᵢ)
    # ∂L/∂b = (-2/n) Σ (yᵢ - ŷᵢ)
    residuals = y - y_pred
    dL_dw = -2 * np.mean(X * residuals)
    dL_db = -2 * np.mean(residuals)

    # ─ UPDATE PARAMETERS ──────────────────────────────────────────
    w = w - alpha * dL_dw
    b = b - alpha * dL_db

    if epoch % 40 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | w: {w:.4f} | b: {b:.4f}")

print(f"\nFinal: w = {w:.4f} (true: 3.0), b = {b:.4f} (true: 2.0)")

# Plot loss curve
plt.figure(figsize=(8, 4))
plt.plot(loss_history)
plt.xlabel("Epoch"); plt.ylabel("Loss (MSE)")
plt.title("Gradient Descent: Loss over Epochs")
plt.grid(True, alpha=0.3)
plt.savefig('gradient_descent_loss.png', dpi=100, bbox_inches='tight')
plt.show()
```

**Expected output:**
```
Epoch   0 | Loss: 14.3252 | w: 0.4386 | b: 0.0394
Epoch  40 | Loss: 0.3489  | w: 2.7891 | b: 1.8741
Epoch  80 | Loss: 0.2713  | w: 2.9531 | b: 1.9643
Epoch 120 | Loss: 0.2607  | w: 2.9821 | b: 1.9854
Epoch 160 | Loss: 0.2592  | w: 2.9918 | b: 1.9934

Final: w = 2.9958 (true: 3.0), b = 1.9966 (true: 2.0)
```

---

## 22. Learning Rate

### What is it?

The **learning rate** (α) controls how large each parameter update step is during gradient descent.

```
w_new = w_old - α × gradient
                 ↑
           This is the learning rate
```

### Why is it Critical?

The learning rate is one of the most important **hyperparameters** in ML. Getting it wrong makes training fail:

```
Learning rate TOO HIGH:
  α = 1.0

  Loss over steps:
  10  →  85  →  670  →  ...  DIVERGES!
  (Overshoots the minimum, bounces around, gets worse)

  Visual:
  Loss |  ●         ●
       |       ●         ●
       |    ↑ jumps over minimum, diverges
       └──────────────────── parameters

Learning rate TOO LOW:
  α = 0.0001

  Loss over 1000 steps:
  10  →  9.99  →  9.98  →  ... still far from minimum
  (Takes forever to converge — impractical)

Learning rate JUST RIGHT:
  α = 0.01

  Loss converges smoothly to minimum.
```

### Visualisation

```
α too high:        α too low:         α just right:
   ●               ●
  / \              │●
 /   ● ← diverges │ ●                  ●
/         └──      │  ●               ╲
               lost│   ●               ╲●
                   │    ●               ╲●___●
             (slow convergence)          (smooth convergence)
```

### Learning Rate Schedules

In practice, the learning rate is often **not constant** — it decreases over training:

| Schedule | Description | Formula |
|---|---|---|
| **Constant** | Fixed throughout training | α = 0.01 |
| **Step Decay** | Reduce by factor every N epochs | α × 0.1 every 10 epochs |
| **Exponential Decay** | Smooth exponential decrease | α₀ × e^(−kt) |
| **Cosine Annealing** | Follows cosine curve | Widely used in practice |
| **Warmup + Decay** | Start small, increase, then decay | Used in Transformer training |

### Python

```python
import numpy as np
import matplotlib.pyplot as plt

# Compare different learning rates on f(w) = w²
def gradient_descent_1d(start, lr, n_steps=50):
    w = start
    history = [w]
    for _ in range(n_steps):
        gradient = 2 * w       # ∂(w²)/∂w = 2w
        w = w - lr * gradient
        history.append(w)
    return history

w_histories = {}
learning_rates = [0.9, 0.1, 0.01, 0.001]
for lr in learning_rates:
    w_histories[lr] = gradient_descent_1d(start=10.0, lr=lr)

# Plot the loss (w²) for each learning rate
plt.figure(figsize=(10, 5))
steps = range(51)
for lr, w_hist in w_histories.items():
    losses = [w**2 for w in w_hist]
    plt.plot(steps, losses, label=f'α = {lr}', linewidth=2)

plt.xlabel("Gradient Descent Step"); plt.ylabel("Loss (w²)")
plt.title("Effect of Learning Rate on Convergence")
plt.legend(); plt.grid(True, alpha=0.3)
plt.ylim(-5, 110)
plt.savefig('learning_rate_comparison.png', dpi=100, bbox_inches='tight')
plt.show()

# Learning rate schedule: step decay
def step_decay(initial_lr, epoch, drop=0.5, epochs_drop=10):
    return initial_lr * (drop ** (epoch // epochs_drop))

epochs = range(50)
lr_values = [step_decay(0.1, e) for e in epochs]
print("Learning rates with step decay:")
for e in [0, 10, 20, 30, 40, 49]:
    print(f"  Epoch {e}: α = {lr_values[e]:.4f}")
```

---

## 23. Local and Global Minima

### The Problem with Non-Convex Functions

The loss landscape of a neural network is **not a simple bowl**. It is a high-dimensional surface with:

```
TYPES OF CRITICAL POINTS (where gradient = 0):

Global minimum:          Local minimum:          Saddle point:
        ___                  /\    /\               ___/
       /   \                /  \  /  \             /
      |     |              /    \/    \            |     \___
   ___/     \___          /     ↑     \            |          \_
              ↑           ↑  local min ↑           ↑ saddle: minimum
         global min        (not optimal)           in one direction,
                                                   maximum in another
```

| Type | Gradient | Loss | Problem? |
|---|---|---|---|
| **Global minimum** | 0 | Lowest possible | Goal — we want this |
| **Local minimum** | 0 | Low, but not lowest | Can trap gradient descent |
| **Saddle point** | 0 | Neither min nor max | Common in high dimensions |
| **Plateau** | ≈ 0 | Flat region | Slows training dramatically |

### The Good News for Deep Learning

For **deep neural networks**, local minima are usually not a serious problem:

1. **Most local minima in high dimensions are nearly as good as the global minimum**
   - A local minimum that is bad in all N dimensions is exponentially rare
   - Most "traps" are saddle points, not local minima

2. **Gradient descent rarely gets stuck for long**
   - Saddle points are unstable — small noise or momentum escapes them

3. **The real issues are:**
   - Saddle points (not local minima) — slows convergence
   - Poorly conditioned loss landscapes
   - Vanishing/exploding gradients in deep networks

### Python: Visualising a Non-Convex Landscape

```python
import numpy as np
import matplotlib.pyplot as plt

# A 1D non-convex loss function with multiple minima
def loss_landscape(w):
    return np.sin(3*w) + 0.5*w**2 - 0.5*w

w_vals = np.linspace(-3, 3, 1000)
loss_vals = loss_landscape(w_vals)

plt.figure(figsize=(10, 4))
plt.plot(w_vals, loss_vals, 'b-', linewidth=2)

# Mark local and global minima
local_min_w  = -0.7    # approximate
global_min_w =  0.3    # approximate

plt.scatter([local_min_w], [loss_landscape(local_min_w)],
            color='orange', s=150, zorder=5, label='Local minimum', marker='v')
plt.scatter([global_min_w], [loss_landscape(global_min_w)],
            color='green', s=150, zorder=5, label='Global minimum', marker='v')

plt.title("Non-convex Loss Landscape")
plt.xlabel("Weight (w)"); plt.ylabel("Loss")
plt.legend(); plt.grid(True, alpha=0.3)
plt.savefig('loss_landscape.png', dpi=100, bbox_inches='tight')
plt.show()

# Show how starting point affects where you end up
def gradient_descent_nonconvex(start, lr=0.01, n_steps=200):
    w = start
    history = [w]
    for _ in range(n_steps):
        # Gradient of sin(3w) + 0.5w² - 0.5w is 3cos(3w) + w - 0.5
        gradient = 3*np.cos(3*w) + w - 0.5
        w = w - lr * gradient
        history.append(w)
    return history

starts = [-2.5, -0.5, 0.5, 2.0]
print("Starting point → Final weight (where GD converges):")
for s in starts:
    final = gradient_descent_nonconvex(s)[-1]
    print(f"  Start: {s:5.1f} → Final: {final:.4f}  Loss: {loss_landscape(final):.4f}")
```

---

## 24. Batch, Stochastic, and Mini-batch Gradient Descent

### The Problem with Full-Batch Gradient Descent

Standard ("batch") gradient descent computes the gradient on the **entire training dataset** before each update:

```
For epoch in range(n_epochs):
    gradient = compute_gradient(ALL training data)
    w = w - α × gradient

Problem with 1 million training examples:
  → Must process all 1M examples before taking ONE step
  → Extremely slow per epoch
  → Expensive memory-wise
  → May be stuck in local optima (no noise to escape)
```

### The Three Variants

| Variant | Data per update | Steps per epoch | Noise | Memory |
|---|---|---|---|---|
| **Batch GD** | All N samples | 1 | Low (smooth) | High |
| **Stochastic GD (SGD)** | 1 sample | N | High (noisy) | Very low |
| **Mini-batch GD** | k samples (e.g. 32) | N/k | Medium | Low |

### 24.1 Batch Gradient Descent

```python
# Batch Gradient Descent
# Computes gradient over ENTIRE dataset before each update

def batch_gradient_descent(X, y, alpha=0.01, epochs=100):
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    losses = []

    for epoch in range(epochs):
        # Use ALL samples
        y_pred = X @ w + b
        residuals = y - y_pred
        loss = np.mean(residuals ** 2)
        losses.append(loss)

        # Gradients computed over FULL dataset
        dw = -2 * X.T @ residuals / n
        db = -2 * np.mean(residuals)

        w -= alpha * dw
        b -= alpha * db

    return w, b, losses
```

**Pros:** Stable convergence, smooth loss curve  
**Cons:** Very slow for large datasets, one update per epoch

### 24.2 Stochastic Gradient Descent (SGD)

```python
def stochastic_gradient_descent(X, y, alpha=0.01, epochs=5):
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    losses = []

    for epoch in range(epochs):
        # Shuffle data each epoch
        idx = np.random.permutation(n)
        X_shuffled, y_shuffled = X[idx], y[idx]

        epoch_losses = []
        for i in range(n):
            # Use ONE sample at a time
            xi = X_shuffled[i:i+1]    # Shape (1, d)
            yi = y_shuffled[i]

            y_pred = xi @ w + b
            residual = yi - y_pred
            loss = residual ** 2
            epoch_losses.append(loss)

            # Gradient from ONE sample — noisy but fast
            dw = -2 * xi.T * residual
            db = -2 * residual
            w -= alpha * dw.flatten()
            b -= alpha * db

        losses.append(np.mean(epoch_losses))

    return w, b, losses
```

**Pros:** Very fast updates, noisy updates can escape local minima  
**Cons:** Very noisy loss curve, high variance in parameter updates

### 24.3 Mini-batch Gradient Descent

The standard in deep learning. Balances the speed of SGD with the stability of batch GD.

```python
def minibatch_gradient_descent(X, y, alpha=0.01, epochs=50, batch_size=32):
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    losses = []

    for epoch in range(epochs):
        # Shuffle data
        idx = np.random.permutation(n)
        X_shuf, y_shuf = X[idx], y[idx]

        epoch_losses = []
        for start in range(0, n, batch_size):
            # MINI-BATCH: k samples at a time
            end = min(start + batch_size, n)
            X_batch = X_shuf[start:end]   # Shape (k, d)
            y_batch = y_shuf[start:end]   # Shape (k,)

            y_pred    = X_batch @ w + b
            residuals = y_batch - y_pred
            loss      = np.mean(residuals ** 2)
            epoch_losses.append(loss)

            # Gradient computed on mini-batch
            dw = -2 * X_batch.T @ residuals / len(y_batch)
            db = -2 * np.mean(residuals)

            w -= alpha * dw
            b -= alpha * db

        losses.append(np.mean(epoch_losses))

    return w, b, losses
```

**Pros:** Fast per update, stable enough convergence, GPU-friendly (parallelise batch)  
**Cons:** Requires batch size tuning

### Full Comparison in Code

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Generate dataset
n = 500
X_data = np.random.randn(n, 2)
w_true = np.array([2.5, -1.5])
y_data = X_data @ w_true + 3.0 + 0.5 * np.random.randn(n)

# Run all three variants
w_batch, b_batch, loss_batch = batch_gradient_descent(
    X_data, y_data, alpha=0.01, epochs=100)

w_sgd, b_sgd, loss_sgd = stochastic_gradient_descent(
    X_data, y_data, alpha=0.01, epochs=5)

w_mini, b_mini, loss_mini = minibatch_gradient_descent(
    X_data, y_data, alpha=0.01, epochs=50, batch_size=32)

print("=== CONVERGENCE COMPARISON ===")
print(f"Batch GD   → Final loss: {loss_batch[-1]:.4f}")
print(f"SGD        → Final loss: {loss_sgd[-1]:.4f}")
print(f"Mini-batch → Final loss: {loss_mini[-1]:.4f}")

# Plot
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

axes[0].plot(loss_batch); axes[0].set_title("Batch GD\n(Smooth, slow)")
axes[1].plot(loss_sgd);   axes[1].set_title("Stochastic GD\n(Fast, noisy)")
axes[2].plot(loss_mini);  axes[2].set_title("Mini-batch GD\n(Best of both)")

for ax in axes:
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('gd_comparison.png', dpi=100, bbox_inches='tight')
plt.show()
```

### Visual Comparison of Convergence Paths

```
BATCH GD:              STOCHASTIC GD:         MINI-BATCH GD:
Loss                   Loss                   Loss
│ ╲                    │╲│╲╲╱│               │ ╲
│  ╲                   │ ╲ ╲╱╲              │  ╲_
│   ╲_                 │ ╱╲  ╲_            │     ╲__
│     ╲___             │╱   ╲___           │        ╲___
└──────────── epoch    └──────── epoch     └────────────── epoch
Smooth convergence     Very noisy          Slight noise,
but slow per epoch     fast updates        good convergence
```

### Common Batch Sizes in Practice

| Batch Size | Common Use Case |
|---|---|
| 1 | True SGD (rare in practice) |
| 16–32 | Small datasets, limited GPU memory |
| 64–128 | Most common for deep learning |
| 256–512 | Large GPU memory available |
| 1024+ | Large-scale training (BERT, GPT pre-training) |

> **Rule of thumb:** Start with batch size 32 or 64. Increase if your GPU has memory to spare. Larger batches train faster per epoch but may need a proportionally larger learning rate.

---

## 🔬 Practical Section: Putting It All Together

### Linear Regression: From Math to Code to Sklearn

```python
import numpy as np
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

np.random.seed(42)

# ── GENERATE DATA ─────────────────────────────────────────────────
n = 200
X = 2 * np.random.randn(n, 1)      # Feature matrix, shape (200, 1)
y = 4 + 3 * X.flatten() + np.random.randn(n)  # y = 3x + 4 + noise

# ── METHOD 1: NORMAL EQUATION (uses matrix inverse) ───────────────
X_aug = np.hstack([np.ones((n, 1)), X])   # Add bias column
theta_normal = np.linalg.solve(X_aug.T @ X_aug, X_aug.T @ y)
print("=== Method 1: Normal Equation ===")
print(f"Intercept: {theta_normal[0]:.4f} (true: 4.0)")
print(f"Slope:     {theta_normal[1]:.4f} (true: 3.0)")

# ── METHOD 2: GRADIENT DESCENT (from scratch) ─────────────────────
w = np.zeros(2)     # [bias, weight]
X_gd = X_aug.copy()
alpha = 0.01
losses = []

for epoch in range(500):
    y_pred = X_gd @ w
    residuals = y - y_pred
    loss = np.mean(residuals**2)
    losses.append(loss)
    gradient = -2 * X_gd.T @ residuals / n
    w -= alpha * gradient

print("\n=== Method 2: Gradient Descent (scratch) ===")
print(f"Intercept: {w[0]:.4f} (true: 4.0)")
print(f"Slope:     {w[1]:.4f} (true: 3.0)")

# ── METHOD 3: SCIKIT-LEARN ────────────────────────────────────────
model_sklearn = LinearRegression()
model_sklearn.fit(X, y)

print("\n=== Method 3: Scikit-learn ===")
print(f"Intercept: {model_sklearn.intercept_:.4f} (true: 4.0)")
print(f"Slope:     {model_sklearn.coef_[0]:.4f} (true: 3.0)")

# ── METHOD 4: SCIKIT-LEARN WITH SGD ──────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
sgd_model = SGDRegressor(alpha=0, eta0=0.01, learning_rate='constant',
                          max_iter=1000, random_state=42)
sgd_model.fit(X_scaled, y)

print("\n=== Method 4: SGDRegressor (sklearn) ===")
# Note: coef is in scaled space; unscale for interpretability
w_unscaled = sgd_model.coef_[0] / scaler.scale_[0]
b_unscaled  = sgd_model.intercept_[0] - sgd_model.coef_[0] * scaler.mean_[0] / scaler.scale_[0]
print(f"Intercept: {b_unscaled:.4f}")
print(f"Slope:     {w_unscaled:.4f}")

# All four methods should give essentially the same answer.
```

---

## 🔄 Concept Connections

```
This unit connects to:

PREVIOUS:
  Unit 1: AI/ML overview — now we understand the math behind what we described

FORWARD:
  Unit 3 (Statistics/Probability): Statistical tools build on linear algebra
  Unit 8 (Machine Learning):       Every algorithm uses these math foundations
  Unit 9 (Regression):             Linear algebra + gradient descent in practice
  Unit 16 (Neural Networks):       Matrix multiplication + chain rule = backpropagation
  Unit 17 (Deep Learning):         Eigenvalues for weight init, gradient flow
  Unit 15 (Dimensionality Reduction): Eigendecomposition = PCA
  Unit 22 (Transformers):          Matrix operations, attention = Q×Kᵀ
```

---

## 🎓 Exam Notes

**Important formulas to memorise:**

| Concept | Formula |
|---|---|
| Dot product | a·b = Σᵢ aᵢbᵢ = ‖a‖‖b‖cosθ |
| Matrix multiply shape | (m×n)(n×p) = (m×p) |
| Transpose property | (AB)ᵀ = BᵀAᵀ |
| Inverse property | AA⁻¹ = I |
| Determinant (2×2) | det([a,b;c,d]) = ad−bc |
| Eigenvalue equation | Av = λv |
| Derivative (power rule) | d/dx[xⁿ] = nxⁿ⁻¹ |
| Chain rule | dy/dx = (dy/du)(du/dx) |
| Gradient | ∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ] |
| MSE | (1/n)Σ(y−ŷ)² |
| GD update rule | θ ← θ − α∇L(θ) |

**Key definitions for exams:**

1. **Eigenvector:** A non-zero vector v such that Av = λv (only scaled, not rotated by A)
2. **Eigenvalue:** The scalar λ by which the eigenvector is scaled
3. **Gradient:** Vector of all partial derivatives — points in direction of steepest ascent
4. **Loss function:** Measures error for one sample; cost function = average over all samples
5. **Learning rate:** Hyperparameter α controlling step size in gradient descent
6. **Saddle point:** A critical point where gradient = 0 but it is neither a min nor a max

---

## 💼 Interview Notes

**Q1: What is the difference between a vector and a matrix?**

> A vector is a 1D ordered list of numbers representing a point or direction in n-dimensional space. A matrix is a 2D array of numbers representing a linear transformation or a collection of vectors. In ML, a data point (one row of your dataset) is a vector; your entire dataset is a matrix.

---

**Q2: Why is the dot product so important in neural networks?**

> Every neuron computes: output = dot(weights, input) + bias. This is the dot product of the weight vector and input vector, plus a scalar bias. A layer of neurons performing this simultaneously is a matrix multiplication. The dot product also measures vector similarity — which is why it is used in cosine similarity (embedding search), attention scores in Transformers (Q·Kᵀ), and recommendation systems.

---

**Q3: What is the chain rule and why does it matter for backpropagation?**

> The chain rule says: if y = f(g(x)), then dy/dx = (df/dg) × (dg/dx). A neural network is a composition of many functions (layers). Backpropagation computes the gradient of the loss with respect to every weight by applying the chain rule repeatedly backwards through the network — hence "backpropagation." Without the chain rule, we could not train deep networks.

---

**Q4: Why do we subtract the gradient in gradient descent, not add it?**

> The gradient ∇L points in the direction of steepest ASCENT — increasing loss. Since we want to DECREASE loss, we move in the opposite direction: θ ← θ − α∇L. The minus sign is what makes it "descent" rather than "ascent."

---

**Q5: What is the difference between batch, stochastic, and mini-batch gradient descent?**

> Batch GD computes the gradient on all training samples before each update — stable but slow. Stochastic GD (SGD) updates after each individual sample — fast and noisy, can escape local minima. Mini-batch GD updates after every k samples (typically 32-256) — balances speed, memory efficiency, and convergence stability. Mini-batch GD is the standard in deep learning because it is GPU-friendly (parallelise the batch).

---

**Q6: What are eigenvalues and eigenvectors and how are they used in PCA?**

> An eigenvector of matrix A is a vector v such that Av = λv — the matrix only scales it, not rotates it. The scalar λ is the eigenvalue. In PCA, we compute the eigenvectors of the covariance matrix of the data. These eigenvectors are the "principal components" — directions of maximum variance. The eigenvalues tell us how much variance each direction explains. We project the data onto the top k eigenvectors to reduce dimensionality while retaining maximum information.

---

**Q7: What happens when the learning rate is too high or too low?**

> Too high: the gradient descent step overshoots the minimum, bouncing around or diverging — the loss increases instead of decreasing. Too low: convergence is extremely slow — thousands of epochs are needed and the model may take impractically long to train. The learning rate is typically tuned by trying values like 0.1, 0.01, 0.001 and observing the loss curve. Modern optimisers (Adam) adapt the learning rate per parameter automatically.

---

## ⚠️ Common Mistakes & Misconceptions

**❌ Mistake 1: Confusing A × B with A ⊙ B**
✅ In NumPy, `A * B` is element-wise (Hadamard product). Matrix multiplication is `A @ B` or `np.matmul(A, B)`. This is the most common NumPy bug for ML beginners.

---

**❌ Mistake 2: "The gradient tells us where the minimum is"**
✅ The gradient only tells you the direction of steepest ASCENT at the current point. It does not tell you where the minimum is — just which direction to avoid (and the opposite direction to move).

---

**❌ Mistake 3: "Gradient descent always finds the global minimum"**
✅ Gradient descent finds a local minimum (for non-convex functions like neural network losses). Fortunately, for large networks, most local minima are nearly as good as the global minimum. Practical concern is saddle points, not local minima.

---

**❌ Mistake 4: Scaling features is optional**
✅ Gradient descent converges much faster when all features have similar scales. Without scaling, features with large values dominate the gradient and training is slow or unstable. Always scale features before training.

---

**❌ Mistake 5: "More epochs always improves the model"**
✅ Training too long leads to **overfitting** — the model memorises training data and generalises poorly. Monitor validation loss; stop when it stops improving (early stopping).

---

**❌ Mistake 6: Determinant = matrix size**
✅ The determinant is a scalar that captures whether a matrix is invertible (det ≠ 0) and how much it scales area/volume. It has nothing to do with the number of rows or columns.

---

## 🧠 Key Takeaways

1. **Scalars** are single numbers. **Vectors** are ordered lists. **Matrices** are 2D grids. Every ML dataset and model is built from these.

2. **The dot product** is the most important operation in ML — it is the computation inside every neuron and the basis of attention in Transformers.

3. **Matrix multiplication (m×n)(n×p) = (m×p)** — inner dimensions must match. Every neural network forward pass is a sequence of matrix multiplications.

4. **Transpose** flips rows and columns. Critical for backpropagation: `(AB)ᵀ = BᵀAᵀ`.

5. **Eigenvalues and eigenvectors** are the directions that a matrix only scales. Foundation of PCA (find directions of maximum variance in data).

6. **Derivatives** measure instantaneous rate of change — the slope of the tangent line. **Partial derivatives** do this for one variable at a time.

7. **The chain rule** (dy/dx = dy/du × du/dx) is the mathematical engine of backpropagation. Apply it repeatedly to compute gradients through composed functions.

8. **The gradient ∇f** is a vector of all partial derivatives. It points uphill. **−∇f** points downhill. Gradient descent follows −∇f.

9. **Loss functions** measure model error. MSE for regression; cross-entropy for classification. We minimise the loss by following the negative gradient.

10. **Gradient descent** update rule: `θ ← θ − α∇L`. The learning rate α controls step size. Too high → diverge. Too low → slow. Mini-batch GD is the standard.

---

## ❓ Revision Questions

### Basic Understanding
1. What is the difference between a scalar, a vector, and a matrix? Give one ML example of each.
2. What is the shape of the result when you multiply a (4×3) matrix by a (3×5) matrix?
3. What does the transpose of a matrix do? What is the shape of Aᵀ if A is (m×n)?
4. When does a matrix NOT have an inverse?
5. What is the identity matrix and what property does it have with respect to matrix multiplication?

### Conceptual Understanding
6. Explain in plain language what the dot product measures geometrically.
7. What is the difference between element-wise matrix multiplication (Hadamard) and standard matrix multiplication?
8. Explain what an eigenvector is without using equations. Why do they matter for PCA?
9. What does it mean for the gradient of a function to point "uphill"?
10. Explain the chain rule with a concrete example. Why is it critical for training neural networks?

### Application
11. Given f(x,y) = x³ + 2xy − y², find ∂f/∂x and ∂f/∂y.
12. A model has loss L(w) = (w−5)². Starting at w=0 with learning rate α=0.1, what is w after 3 gradient descent steps?
13. You are training a neural network and the loss oscillates wildly, sometimes going up. What is the most likely cause?
14. Your training loss decreases but your validation loss increases after epoch 20. What is happening and what should you do?
15. Why do we shuffle training data before each epoch in SGD and mini-batch GD?

### Technical Understanding
16. What is the normal equation θ = (XᵀX)⁻¹Xᵀy? What is its main limitation for large datasets?
17. Write the MSE loss formula and derive the gradient ∂MSE/∂w for a linear model ŷ = wx.
18. What is the difference between a local minimum, global minimum, and saddle point?
19. Why does mini-batch GD work better on GPUs than pure SGD?
20. What does the determinant of a 2×2 matrix tell you geometrically?

---

## 📝 Quick Revision

```
LINEAR ALGEBRA
───────────────
Scalar: single number (α, λ)
Vector: ordered list — magnitude + direction
Matrix: 2D grid — shape (m,n)

Dot product: a·b = Σ aᵢbᵢ = ||a||||b||cosθ  → similarity
Matrix multiply: (m×n)(n×p) = (m×p)  → NOT element-wise
Transpose: (AB)ᵀ = BᵀAᵀ  → rows↔columns
Inverse: AA⁻¹ = I  → exists only if det≠0
Eigenvectors: Av = λv  → only scaled, not rotated
PCA: eigenvectors of covariance matrix

CALCULUS
────────
Derivative: f'(x) = lim[f(x+h)-f(x)]/h  → instantaneous slope
Partial: ∂f/∂x  → derivative of one variable at a time
Chain rule: dy/dx = (dy/du)(du/dx)  → backpropagation
Gradient: ∇f = [∂f/∂x₁, ..., ∂f/∂xₙ]  → vector pointing uphill

OPTIMIZATION
────────────
Loss function: measures one-sample error
Cost function: average loss over all samples
MSE = (1/n)Σ(y−ŷ)²  → regression
BCE = −(1/n)Σ[y·log(ŷ)+(1−y)·log(1−ŷ)]  → classification

Gradient descent: θ ← θ − α∇L
  α too high → diverges
  α too low  → too slow
  α just right → smooth convergence

Batch GD:    all data per step   → stable, slow
SGD:         1 sample per step   → fast, noisy
Mini-batch:  k samples per step  → standard in deep learning
```

---

## 📚 Further Learning

| Resource | Type | Why Useful |
|---|---|---|
| *Linear Algebra* — Gilbert Strang (MIT 18.06) | Course | Best linear algebra course ever made — free on MIT OpenCourseWare |
| 3Blue1Brown — "Essence of Linear Algebra" | YouTube | Visual, intuitive, beautiful |
| 3Blue1Brown — "Essence of Calculus" | YouTube | Best visual intuition for derivatives |
| *Mathematics for Machine Learning* — Deisenroth, Faisal, Ong | Book | ML-focused math textbook, free PDF |
| fast.ai — Practical Deep Learning | Course | Builds intuition for why the math is needed |
| Andrej Karpathy — "Neural Networks: Zero to Hero" | YouTube | Implements backprop from scratch |
| Khan Academy — Multivariable Calculus | Course | Gentle intro to partial derivatives and gradients |

---

*📁 Next: `03-statistics-probability/README.md` — Probability, Distributions, and Statistical Inference for ML*