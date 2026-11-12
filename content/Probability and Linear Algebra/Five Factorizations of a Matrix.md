---
title: Five Factorizations of a Matrix
tags:
  - linear-algebra
  - foundations
date: 2026-11-12
draft: false
prev: "[[First-Order Optimization]]"
next: "[[Probability Basics]]"
---

> **Prerequisites:** the earlier linear algebra lectures, especially [[Column Space and Rank]], [[Elimination and the Nullspace]], [[Orthogonality and Least Squares]] and [[Eigenvalues and Eigenvectors]].
>
> **Series:** this is part 8 of 8, the closing lecture. [[Multivariable Calculus]] → [[Column Space and Rank]] → [[Elimination and the Nullspace]] → [[The Four Fundamental Subspaces]] → [[Orthogonality and Least Squares]] → [[Eigenvalues and Eigenvectors]] → [[First-Order Optimization]] → **Five Factorizations of a Matrix**.
>
> **Notation:** in this lecture $\mathbf{X}$ is the **eigenvector matrix** in $\mathbf{A}=\mathbf{X}\boldsymbol{\Lambda}\mathbf{X}^{-1}$, as in Strang's books. Elsewhere in the series $\mathbf{X}\in\mathbb{R}^{N\times D}$ is the data matrix; when both appear below, the data matrix is written out as such.
>
> **TL;DR:**
> - A product of matrices is a **sum of rank-one pieces**, column times row. Every factorization is a way of splitting a matrix into simple pieces.
> - Six factorizations carry most of linear algebra: $\mathbf{A}=\mathbf{C}\mathbf{R}$ (independent columns), $\mathbf{A}=\mathbf{L}\mathbf{U}$ (elimination), $\mathbf{A}=\mathbf{Q}\mathbf{R}$ (orthogonalization), $\mathbf{S}=\mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^\top$ (symmetric eigenvalues), $\mathbf{A}=\mathbf{X}\boldsymbol{\Lambda}\mathbf{X}^{-1}$ (general eigenvalues) and $\mathbf{A}=\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$ (singular values).
> - Pick by task: LU (or Cholesky) to solve systems, QR or the SVD for least squares, eigenvalues for powers and dynamics, the SVD for low rank and conditioning, CR when the columns must stay meaningful.
> - Deep learning composes affine maps and nonlinearities, and trains them by stochastic gradients rather than by factorizing anything. Linear algebra still decides how fast and how stably that training goes.

## Motivation

Strang closes his short series with a video that puts the main factorizations side by side.[^strang2020five] The point is not to learn five new things; you have met each of them already in this series. The point is to see them as **one idea** applied five ways. Each factorization writes a matrix as a product of factors with simple structure (triangular, orthogonal, diagonal), and each structure answers a different question: Is $\mathbf{A}\mathbf{x}=\mathbf{b}$ solvable, and how cheaply? What is the best fit? What happens when we apply $\mathbf{A}$ a thousand times? What is the closest simpler matrix? This lecture lines them up, compares them on the same examples, and says which to reach for when. It ends by connecting the whole series to the way neural networks are actually built and trained.

## Matrix multiplication as columns times rows

The usual rule computes $\mathbf{A}\mathbf{B}$ one entry at a time, as dot products of rows of $\mathbf{A}$ with columns of $\mathbf{B}$. Regroup the same sum by the shared index $k$ instead, and the product becomes a **sum of outer products**:

$$
\mathbf{A}\mathbf{B}=\sum_{k=1}^{n}\mathbf{a}_k\mathbf{b}_k^\top,
$$

where $\mathbf{a}_k$ is column $k$ of $\mathbf{A}$ and $\mathbf{b}_k^\top$ is row $k$ of $\mathbf{B}$. The proof is one line: entry $(i,j)$ of the right-hand side is $\sum_ka_{ik}b_{kj}$, which is entry $(i,j)$ of $\mathbf{A}\mathbf{B}$. Each $\mathbf{a}_k\mathbf{b}_k^\top$ is a **rank-one** matrix: every column is a multiple of $\mathbf{a}_k$ and every row a multiple of $\mathbf{b}_k^\top$. This is the fourth of the [[Column Space and Rank#Matrix multiplication four ways|four ways to multiply matrices]], and it is the one that makes factorizations readable.

> [!tip] Intuition: a factorization is a recipe of layers
> If $\mathbf{A}=\mathbf{F}\mathbf{G}$ with $\mathbf{F}$ having $r$ columns, then $\mathbf{A}$ is a stack of $r$ rank-one layers, $\mathbf{f}_1\mathbf{g}_1^\top+\cdots+\mathbf{f}_r\mathbf{g}_r^\top$. The factorizations below differ in what they demand of the layers: built from actual columns (CR), peeled off by elimination (LU), built on orthonormal columns (QR), orthonormal on both sides and sorted by importance (SVD).

**Example.** The matrix below has rank 2: its second column is twice its first, and its first and third columns are independent. With $\mathbf{C}$ holding columns 1 and 3, and $\mathbf{R}$ saying how to rebuild every column from them,

$$
\begin{bmatrix}1&2&3\\2&4&7\\1&2&4\end{bmatrix}=\begin{bmatrix}1&3\\2&7\\1&4\end{bmatrix}\begin{bmatrix}1&2&0\\0&0&1\end{bmatrix}=\begin{bmatrix}1&2&0\\2&4&0\\1&2&0\end{bmatrix}+\begin{bmatrix}0&0&3\\0&0&7\\0&0&4\end{bmatrix}.
$$

![[Probability and Linear Algebra/images/five-factorizations-of-a-matrix-columns-times-rows.svg]]
*Figure 1. Columns times rows. The rank-2 matrix $\mathbf{A}=\mathbf{C}\mathbf{R}$ is the sum of two rank-one layers: column 1 of $\mathbf{C}$ times row 1 of $\mathbf{R}$ (blue), plus column 2 of $\mathbf{C}$ times row 2 of $\mathbf{R}$ (orange).*

**Elimination peels off one layer per pivot.** The same view explains LU. Take

$$
\mathbf{A}=\begin{bmatrix}2&1&1\\4&3&3\\2&3&5\end{bmatrix}.
$$

The first layer is the first column of $\mathbf{A}$ divided by the pivot $2$, times the first row: $\boldsymbol{\ell}_1\mathbf{u}_1^\top$ with $\boldsymbol{\ell}_1=(1,2,1)$ and $\mathbf{u}_1^\top=(2,1,1)$. Subtracting it clears the first row and column:

$$
\mathbf{A}-\begin{bmatrix}1\\2\\1\end{bmatrix}\begin{bmatrix}2&1&1\end{bmatrix}=\begin{bmatrix}0&0&0\\0&1&1\\0&2&4\end{bmatrix}.
$$

What remains is the $2\times2$ block that elimination works on next. Repeating gives $\boldsymbol{\ell}_2=(0,1,2)$, $\mathbf{u}_2^\top=(0,1,1)$, and a last layer $\boldsymbol{\ell}_3=(0,0,1)$, $\mathbf{u}_3^\top=(0,0,2)$. Stacking the $\boldsymbol{\ell}_k$ as columns and the $\mathbf{u}_k^\top$ as rows is exactly $\mathbf{A}=\mathbf{L}\mathbf{U}$.

## The five factorizations

Strang counts five in his 2020 video and adds $\mathbf{A}=\mathbf{C}\mathbf{R}$ as the starting point of his later teaching, so there are six below.[^strang2023cr] For each: what it reveals, how it is computed and at what cost, where it appears in machine learning and reinforcement learning, and where the full treatment lives. Costs are for dense matrices, $\mathbf{A}\in\mathbb{R}^{m\times n}$ or $n\times n$; a "flop" is one floating-point addition or multiplication.

![[Probability and Linear Algebra/images/five-factorizations-of-a-matrix-summary.svg]]
*Figure 2. The six factorizations by the shape of their factors. Shaded triangles are triangular factors; striped blocks have orthonormal columns; diagonal factors hold pivots, eigenvalues or singular values. Only $\mathbf{X}$ and $\mathbf{X}^{-1}$ have no special structure, which is why the eigenvector factorization is the most fragile.*

### A = CR: independent columns

- **Reveals:** the rank $r$, a basis of the column space made of **actual columns** of $\mathbf{A}$, and (via $\mathbf{R}$) how every other column depends on them. It proves that column rank equals row rank.
- **Computed by:** elimination to the reduced row echelon form; the nonzero rows of $\operatorname{rref}(\mathbf{A})$ are $\mathbf{R}$, and the pivot columns of $\mathbf{A}$ are $\mathbf{C}$. Cost $O(mnr)$.
- **Example:** Figure 1.
- **Used for:** keeping features interpretable. Selecting a few real columns (and rows) is the idea behind [[Matrix Factorization and Dimensionality Reduction#CUR decomposition|CUR decompositions]] and column-subset selection: "these five genes", not "this combination of all genes".
- **Weaknesses:** the chosen columns can be nearly parallel, so $\mathbf{C}$ may be badly conditioned; and for an invertible $\mathbf{A}$ it gives $\mathbf{C}=\mathbf{A}$, $\mathbf{R}=\mathbf{I}$, which is no progress. See [[Column Space and Rank#A = CR]] and [[Column Space and Rank#Pros and cons of CR]].

### A = LU: elimination

- **Reveals:** the pivots. Their product is $\det\mathbf{A}$ (up to the sign of the row exchanges), and a zero pivot that no row exchange can fix means $\mathbf{A}$ is singular.
- **Computed by:** Gaussian elimination, recording each multiplier $\ell_{ij}$ in $\mathbf{L}$. With row exchanges for stability (partial pivoting) it reads $\mathbf{P}\mathbf{A}=\mathbf{L}\mathbf{U}$. Cost about $\tfrac23n^3$ flops once, then about $2n^2$ per right-hand side (one forward and one back substitution).
- **Example:** for the matrix above,

$$
\begin{bmatrix}2&1&1\\4&3&3\\2&3&5\end{bmatrix}=\underbrace{\begin{bmatrix}1&0&0\\2&1&0\\1&2&1\end{bmatrix}}_{\mathbf{L}}\underbrace{\begin{bmatrix}2&1&1\\0&1&1\\0&0&2\end{bmatrix}}_{\mathbf{U}}.
$$

The pivots are $2,1,2$, so $\det\mathbf{A}=4$. To solve $\mathbf{A}\mathbf{x}=(4,10,10)$, first solve $\mathbf{L}\mathbf{c}=\mathbf{b}$ forwards: $c_1=4$, $c_2=10-2\cdot4=2$, $c_3=10-4-2\cdot2=2$. Then solve $\mathbf{U}\mathbf{x}=\mathbf{c}$ backwards: $x_3=2/2=1$, $x_2=2-1=1$, $x_1=(4-1-1)/2=1$.

- **Symmetric positive definite case:** $\mathbf{S}=\mathbf{L}\mathbf{L}^\top$ (**Cholesky**), at half the cost, about $\tfrac13n^3$. It succeeds exactly when $\mathbf{S}$ is positive definite, which makes it the standard test for definiteness.
- **Used for:** solving linear systems. Exact policy evaluation in RL solves $(\mathbf{I}-\gamma\mathbf{P}^\pi)\mathbf{v}=\mathbf{r}^\pi$ ([[Introduction to Reinforcement Learning#Matrix form and closed-form policy evaluation|matrix form of the Bellman equation]]); Newton's method solves a Hessian system at every step. Cholesky samples a [[Joint Distributions#The multivariate Gaussian|multivariate Gaussian]] as $\boldsymbol{\mu}+\mathbf{L}\mathbf{z}$ and gives $\log\det\boldsymbol{\Sigma}=2\sum_i\log L_{ii}$ for Gaussian likelihoods and Gaussian processes.
- **Full treatment:** [[Elimination and the Nullspace#Elimination and A = LU]].

### A = QR: orthogonalization

- **Reveals:** an orthonormal basis $\mathbf{q}_1,\dots,\mathbf{q}_n$ of the column space, built so that the first $k$ of them span the first $k$ columns of $\mathbf{A}$. That is why $\mathbf{R}$ is upper triangular.
- **Computed by:** Gram–Schmidt in principle; Householder reflections in practice, which are more stable. Cost about $2mn^2-\tfrac23n^3$ flops.
- **Example:** Gram–Schmidt on the columns $(3,4)$ and $(2,11)$. The first has length $5$, so $\mathbf{q}_1=\tfrac15(3,4)$. The second has component $\mathbf{q}_1^\top(2,11)=\tfrac15(6+44)=10$ along $\mathbf{q}_1$; removing it leaves $(2,11)-10\cdot\tfrac15(3,4)=(-4,3)$, of length $5$, so $\mathbf{q}_2=\tfrac15(-4,3)$. The coefficients we computed form $\mathbf{R}$:

$$
\begin{bmatrix}3&2\\4&11\end{bmatrix}=\underbrace{\frac15\begin{bmatrix}3&-4\\4&3\end{bmatrix}}_{\mathbf{Q}}\underbrace{\begin{bmatrix}5&10\\0&5\end{bmatrix}}_{\mathbf{R}}.
$$

- **Used for:** least squares. With $\mathbf{A}=\mathbf{Q}\mathbf{R}$ the normal equations collapse to the triangular system $\mathbf{R}\hat{\mathbf{x}}=\mathbf{Q}^\top\mathbf{b}$, without ever forming $\mathbf{A}^\top\mathbf{A}$ and squaring the condition number. QR is also the engine of the **QR algorithm** that computes eigenvalues ([[Eigenvalues and Eigenvectors#Power iteration and the QR algorithm|how eigenvectors are actually computed]]), and orthogonal weight initialization for deep networks draws $\mathbf{Q}$ from the QR factorization of a random Gaussian matrix.[^saxe2014]
- **Full treatment:** [[Orthogonality and Least Squares#Gram–Schmidt and A = QR]].

### S = QΛQᵀ: the spectral theorem

- **Reveals:** for a symmetric $\mathbf{S}$, real eigenvalues and an orthonormal basis of eigenvectors: the principal axes of the quadratic form $\mathbf{x}^\top\mathbf{S}\mathbf{x}$. The signs of the eigenvalues decide definiteness.
- **Computed by:** reduction to tridiagonal form by orthogonal similarity transforms, then a symmetric QR or divide-and-conquer iteration. Cost $O(n^3)$.
- **Example:** $\mathbf{S}=\begin{bmatrix}5&2\\2&2\end{bmatrix}$ has trace $7$ and determinant $6$, so $\lambda=6$ and $1$, with unit eigenvectors $\mathbf{q}_1=\tfrac{1}{\sqrt5}(2,1)$ and $\mathbf{q}_2=\tfrac{1}{\sqrt5}(1,-2)$. As layers,

$$
\mathbf{S}=6\,\mathbf{q}_1\mathbf{q}_1^\top+1\,\mathbf{q}_2\mathbf{q}_2^\top=\begin{bmatrix}4.8&2.4\\2.4&1.2\end{bmatrix}+\begin{bmatrix}0.2&-0.4\\-0.4&0.8\end{bmatrix}.
$$

Each $\mathbf{q}_i\mathbf{q}_i^\top$ is the projection onto one axis, so $\mathbf{S}$ stretches by $6$ along $\mathbf{q}_1$ and by $1$ along $\mathbf{q}_2$.

- **Used for:** PCA (the covariance matrix), the Hessian test for minima and convexity ([[Multivariable Calculus]]), the shape of Gaussian level sets, and [[Clustering and Expectation Maximization#Spectral clustering|spectral clustering]] through the eigenvectors of a graph Laplacian.
- **Full treatment:** [[Eigenvalues and Eigenvectors#Symmetric matrices: S = QΛQᵀ]].

### A = XΛX⁻¹: diagonalization

- **Reveals:** the directions $\mathbf{x}_i$ that $\mathbf{A}$ only scales, and by how much. Powers become trivial: $\mathbf{A}^k=\mathbf{X}\boldsymbol{\Lambda}^k\mathbf{X}^{-1}$, so the long-run behaviour of $\mathbf{x}_{t+1}=\mathbf{A}\mathbf{x}_t$ is governed by the largest $\lvert\lambda_i\rvert$.
- **Computed by:** reduction to Hessenberg form, then the shifted QR algorithm, which converges to the triangular Schur form. Cost $O(n^3)$, with a larger constant than the symmetric case.
- **Example:** $\mathbf{A}=\begin{bmatrix}4&1\\2&3\end{bmatrix}$ has trace $7$ and determinant $10$, so $\lambda=5$ and $2$, with eigenvectors $(1,1)$ and $(1,-2)$. Then

$$
\mathbf{A}=\underbrace{\begin{bmatrix}1&1\\1&-2\end{bmatrix}}_{\mathbf{X}}\underbrace{\begin{bmatrix}5&0\\0&2\end{bmatrix}}_{\boldsymbol{\Lambda}}\underbrace{\frac13\begin{bmatrix}2&1\\1&-1\end{bmatrix}}_{\mathbf{X}^{-1}}.
$$

As columns of $\mathbf{X}$ times rows of $\mathbf{X}^{-1}$, every power is a sum of two layers:

$$
\mathbf{A}^k=\frac{5^k}{3}\begin{bmatrix}2&1\\2&1\end{bmatrix}+\frac{2^k}{3}\begin{bmatrix}1&-1\\-2&2\end{bmatrix}.
$$

For $k=2$ this gives $\begin{bmatrix}18&7\\14&11\end{bmatrix}$, which is $\mathbf{A}^2$ multiplied out. For large $k$ the first layer wins by a factor $(5/2)^k$: repeated application aligns everything with the top eigenvector, which is power iteration.

- **Weaknesses:** not every matrix has $n$ independent eigenvectors ($\begin{bmatrix}1&1\\0&1\end{bmatrix}$ does not), eigenvalues of a real matrix can be complex (rotations), and even when $\mathbf{X}$ exists it can be badly conditioned. Numerical software therefore prefers the Schur form $\mathbf{A}=\mathbf{Q}\mathbf{T}\mathbf{Q}^\top$, with orthogonal $\mathbf{Q}$ and triangular $\mathbf{T}$.
- **Used for:** Markov chains, where the stationary distribution is the eigenvector for $\lambda=1$ and the second eigenvalue sets the [[Markov Chains#Aside: mixing and the spectral gap|mixing speed]]; linear dynamical systems; the analysis of gradient descent on a quadratic, where the error evolves as $\mathbf{e}_{t+1}=(\mathbf{I}-\alpha\mathbf{H})\mathbf{e}_t$ ([[First-Order Optimization]]); and discounted RL, where $(\mathbf{I}-\gamma\mathbf{P})^{-1}=\sum_t\gamma^t\mathbf{P}^t$ converges because every eigenvalue of $\gamma\mathbf{P}$ has modulus at most $\gamma<1$.
- **Full treatment:** [[Eigenvalues and Eigenvectors#Diagonalization A = XΛX⁻¹]].

### A = UΣVᵀ: the singular value decomposition

- **Reveals:** for **any** matrix, square or not, orthonormal bases of all four fundamental subspaces, the rank (the number of nonzero $\sigma_i$), the norm $\lVert\mathbf{A}\rVert_2=\sigma_1$, and the condition number $\sigma_1/\sigma_n$. Geometrically: rotate, stretch, rotate.
- **Computed by:** Golub–Kahan bidiagonalization followed by an iterative step; never by forming $\mathbf{A}^\top\mathbf{A}$. Cost $O(mn\min(m,n))$; randomized algorithms approximate the top $k$ triplets much faster.
- **Used for:** PCA (the right singular vectors of the centred data matrix), the best low-rank approximation (Eckart–Young), the pseudo-inverse and minimum-norm least squares, recommender systems, and diagnosing conditioning. In deep learning, LoRA fine-tunes a large model by adding a low-rank update $\Delta\mathbf{W}=\mathbf{B}\mathbf{A}$ to each weight matrix,[^hu2022] and spectral normalization divides a layer's weights by their largest singular value to control its Lipschitz constant, as in the critic of a [[Generative Adversarial Networks|GAN]].[^miyato2018]
- **Full treatment:** [[Matrix Factorization and Dimensionality Reduction#Singular value decomposition (SVD)]], including a $2\times2$ example by hand, the Eckart–Young theorem with proof and image compression.

### All six on one matrix

The examples above were chosen to come out by hand. Applied to the same matrix, the factorizations tell different stories. For $\mathbf{A}=\begin{bmatrix}2&1&1\\4&3&3\\2&3&5\end{bmatrix}$ (computed with NumPy and SciPy, each product checked):

| Factorization | What comes out |
|---|---|
| $\mathbf{C}\mathbf{R}$ | rank 3, so $\mathbf{C}=\mathbf{A}$ and $\mathbf{R}=\mathbf{I}$: nothing gained |
| $\mathbf{L}\mathbf{U}$ | pivots $2,1,2$ without row exchanges; $4,1.5,0.667$ with partial pivoting; product $4=\det\mathbf{A}$ either way |
| $\mathbf{Q}\mathbf{R}$ | $\lvert r_{ii}\rvert=4.90,1.53,0.53$, again with product $4$ |
| $\mathbf{X}\boldsymbol{\Lambda}\mathbf{X}^{-1}$ | eigenvalues $8.08,1.61,0.31$ (real); $\kappa(\mathbf{X})=2.9$ |
| $\mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^\top$ of $\mathbf{A}^\top\mathbf{A}$ | eigenvalues $73.08,4.88,0.045$ |
| $\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$ | singular values $8.55,2.21,0.21$, the square roots of the line above; $\kappa(\mathbf{A})=40.4$ |

The LU pivots, the diagonal of $\mathbf{R}$, the eigenvalues and the singular values all multiply to $\lvert\det\mathbf{A}\rvert=4$ (and the eigenvalues of $\mathbf{A}^\top\mathbf{A}$ to $4^2=16$). But the eigenvalues and the singular values of a non-symmetric matrix **differ**: the eigenvalues describe repeated application of $\mathbf{A}$, while the singular values describe a single application at its most and least stretching. For symmetric positive semi-definite matrices the two coincide.

> [!example]- Code: the six factorizations in NumPy and SciPy
> ```python
> import numpy as np, scipy.linalg as sla
>
> A = np.array([[2., 1., 1.], [4., 3., 3.], [2., 3., 5.]])
> P, L, U = sla.lu(A)                    # A = P L U  (partial pivoting)
> Q, R = np.linalg.qr(A)                 # A = Q R    (Householder)
> lam, X = np.linalg.eig(A)              # A = X diag(lam) X^{-1}
> S = A.T @ A
> mu, Qs = np.linalg.eigh(S)             # S = Qs diag(mu) Qs^T  (symmetric: use eigh)
> Uu, s, Vt = np.linalg.svd(A)           # A = U diag(s) V^T
> Lc = np.linalg.cholesky(S)             # S = Lc Lc^T  (fails unless S is positive definite)
>
> checks = [P @ L @ U, Q @ R, X @ np.diag(lam) @ np.linalg.inv(X), Uu @ np.diag(s) @ Vt]
> print(all(np.allclose(M, A) for M in checks))                                        # True
> print(np.allclose(Qs @ np.diag(mu) @ Qs.T, S), np.allclose(Lc @ Lc.T, S))           # True True
> print(np.round(np.sort(lam.real)[::-1], 2), np.round(s, 2))   # eigenvalues vs singular values
> ```
> SciPy returns $\mathbf{A}=\mathbf{P}\mathbf{L}\mathbf{U}$, so its $\mathbf{P}$ is the transpose of the one in $\mathbf{P}\mathbf{A}=\mathbf{L}\mathbf{U}$. The $\mathbf{C}\mathbf{R}$ factorization has no library routine; the check script computes it from the reduced row echelon form.

## Which factorization when?

| Task | Use | Why | Avoid |
|---|---|---|---|
| Solve $\mathbf{A}\mathbf{x}=\mathbf{b}$, square, many right-hand sides | $\mathbf{P}\mathbf{A}=\mathbf{L}\mathbf{U}$ | factor once in $\tfrac23n^3$, then $2n^2$ per solve | computing $\mathbf{A}^{-1}$ explicitly |
| Same, $\mathbf{S}$ symmetric positive definite | Cholesky $\mathbf{L}\mathbf{L}^\top$ | half the cost, no pivoting needed | |
| Determinant or $\log\det$ | LU or Cholesky | product of pivots | expanding by cofactors ($n!$ terms) |
| Least squares, full column rank | $\mathbf{Q}\mathbf{R}$ | $\mathbf{R}\hat{\mathbf{x}}=\mathbf{Q}^\top\mathbf{b}$ keeps $\kappa(\mathbf{A})$ | normal equations when $\kappa$ is large (they square it) |
| Least squares, rank-deficient or ill-conditioned | SVD | minimum-norm solution $\mathbf{A}^+\mathbf{b}$; drop tiny $\sigma_i$ | |
| Powers $\mathbf{A}^k$, dynamics, Markov chains | $\mathbf{X}\boldsymbol{\Lambda}\mathbf{X}^{-1}$, or $\mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^\top$ if symmetric | $\boldsymbol{\Lambda}^k$ is trivial; the top $\lvert\lambda\rvert$ rules | multiplying out $k$ times |
| Curvature, convexity, definiteness | $\mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^\top$ (or try Cholesky) | signs of eigenvalues | |
| PCA, low-rank approximation, compression | SVD (randomized for huge matrices) | Eckart–Young optimality | eigenvalues of $\mathbf{X}^\top\mathbf{X}$ (squares $\kappa$) |
| Rank and conditioning | SVD | numerical rank = number of $\sigma_i$ above a tolerance; $\kappa=\sigma_1/\sigma_n$ | rank from elimination in floating point |
| Interpretable basis of real columns | $\mathbf{C}\mathbf{R}$ or CUR | keeps actual columns and rows | |

Two rules of thumb summarize the table. **Triangular factors are cheap, orthogonal factors are safe:** LU is the fastest way to solve, while QR and the SVD never amplify rounding errors beyond what the problem itself demands. And **never invert a matrix when you can factor it**: an explicit $\mathbf{A}^{-1}$ costs more than LU and is less accurate.

## Block elimination and the CR connection

Strang and Moler noticed that elimination, written in blocks, produces $\mathbf{A}=\mathbf{C}\mathbf{R}$ directly and explains why it works.[^strangmoler2022] Suppose $\mathbf{A}$ has rank $r$. Reorder rows and columns (by permutation matrices $\mathbf{P}_r$ and $\mathbf{P}_c$) so that an invertible $r\times r$ submatrix $\mathbf{W}$ sits in the top-left corner:

$$
\mathbf{P}_r\mathbf{A}\mathbf{P}_c=\begin{bmatrix}\mathbf{W}&\mathbf{H}\\\mathbf{J}&\mathbf{K}\end{bmatrix}.
$$

**Claim: $\mathbf{K}=\mathbf{J}\mathbf{W}^{-1}\mathbf{H}$.** The first $r$ columns are independent (their top block $\mathbf{W}$ is invertible) and the rank is $r$, so the remaining columns are combinations of them: $\begin{bmatrix}\mathbf{H}\\\mathbf{K}\end{bmatrix}=\begin{bmatrix}\mathbf{W}\\\mathbf{J}\end{bmatrix}\mathbf{M}$ for some $\mathbf{M}$. The top block says $\mathbf{H}=\mathbf{W}\mathbf{M}$, so $\mathbf{M}=\mathbf{W}^{-1}\mathbf{H}$, and the bottom block gives $\mathbf{K}=\mathbf{J}\mathbf{M}=\mathbf{J}\mathbf{W}^{-1}\mathbf{H}$. Therefore

$$
\mathbf{P}_r\mathbf{A}\mathbf{P}_c=\underbrace{\begin{bmatrix}\mathbf{W}\\\mathbf{J}\end{bmatrix}}_{\mathbf{C}}\underbrace{\begin{bmatrix}\mathbf{I}&\mathbf{W}^{-1}\mathbf{H}\end{bmatrix}}_{\mathbf{R}}=\begin{bmatrix}\mathbf{W}\\\mathbf{J}\end{bmatrix}\mathbf{W}^{-1}\begin{bmatrix}\mathbf{W}&\mathbf{H}\end{bmatrix}.
$$

The first form is $\mathbf{C}\mathbf{R}$: $\mathbf{C}$ holds $r$ actual columns, and $\mathbf{R}$ contains the identity. The second form uses $r$ actual columns **and** $r$ actual rows, glued by $\mathbf{W}^{-1}$; it is the exact, noise-free version of the CUR decomposition. Elimination computes $\mathbf{R}$ by row-reducing the top block row: $\begin{bmatrix}\mathbf{W}&\mathbf{H}\end{bmatrix}\to\begin{bmatrix}\mathbf{I}&\mathbf{W}^{-1}\mathbf{H}\end{bmatrix}$. That is why $\operatorname{rref}(\mathbf{A})$ has the form $\begin{bmatrix}\mathbf{I}&\mathbf{F}\\\mathbf{0}&\mathbf{0}\end{bmatrix}\mathbf{P}$, with $\mathbf{F}=\mathbf{W}^{-1}\mathbf{H}$ ([[Elimination and the Nullspace#Reduced row echelon form|reduced row echelon form]]).

> [!example] Example: block elimination by hand
> Take the rank-2 matrix of Figure 1 and swap its last two columns, so that the independent columns 1 and 3 come first:
>
> $$
> \mathbf{A}\mathbf{P}_c=\begin{bmatrix}1&3&2\\2&7&4\\1&4&2\end{bmatrix},
> $$
>
> so no row exchange is needed ($\mathbf{P}_r=\mathbf{I}$) and the blocks are
>
> $$
> \mathbf{W}=\begin{bmatrix}1&3\\2&7\end{bmatrix},\quad\mathbf{H}=\begin{bmatrix}2\\4\end{bmatrix},\quad\mathbf{J}=\begin{bmatrix}1&4\end{bmatrix},\quad\mathbf{K}=\begin{bmatrix}2\end{bmatrix}.
> $$
>
> $\det\mathbf{W}=1$, so $\mathbf{W}^{-1}=\begin{bmatrix}7&-3\\-2&1\end{bmatrix}$ and $\mathbf{W}^{-1}\mathbf{H}=(14-12,\,-4+4)=(2,0)$. The claim checks out: $\mathbf{J}\mathbf{W}^{-1}\mathbf{H}=1\cdot2+4\cdot0=2=\mathbf{K}$. So $\mathbf{R}=\begin{bmatrix}1&0&2\\0&1&0\end{bmatrix}$ in the swapped column order. Swapping the columns back gives $\begin{bmatrix}1&2&0\\0&0&1\end{bmatrix}$, the $\mathbf{R}$ of Figure 1.

## From linear algebra to deep learning

A neural network is linear algebra with one twist. A fully connected network computes

$$
f(\mathbf{x})=\mathbf{W}_L\,\phi\big(\mathbf{W}_{L-1}\cdots\phi(\mathbf{W}_1\mathbf{x}+\mathbf{b}_1)\cdots+\mathbf{b}_{L-1}\big)+\mathbf{b}_L,
$$

a composition of affine maps $\mathbf{x}\mapsto\mathbf{W}\mathbf{x}+\mathbf{b}$ and a fixed elementwise nonlinearity $\phi$, typically $\mathrm{ReLU}(z)=\max(0,z)$. Three facts connect it to this series.

**Without the nonlinearity, depth buys nothing.** $\mathbf{W}_L\cdots\mathbf{W}_1$ is a single matrix, and its rank is at most the narrowest layer width. A deep *linear* network is just a matrix factorization; trained with squared loss through a bottleneck it learns the same subspace as the truncated SVD, which is the [[Matrix Factorization and Dimensionality Reduction#Autoencoders|linear autoencoder]] result.

**With ReLUs, the network is piecewise linear.** Each ReLU switches a unit on or off, so the input space is cut into regions, and on each region $f$ is one affine map. In one dimension (Figure 3), $f(x)=\sum_ja_j\,\mathrm{ReLU}(w_jx+b_j)+c$ has a kink at each $x=-b_j/w_j$. With enough units such pieces approximate any continuous function on a bounded set, which is the [[Neural Networks#Universal approximation|universal approximation theorem]].

![[Probability and Linear Algebra/images/five-factorizations-of-a-matrix-relu-network.svg]]
*Figure 3. A network with one hidden layer of four ReLU units. (a) Each unit contributes a hinge, flat on one side of $x=-b_j/w_j$ and linear on the other. (b) Their sum (plus a constant) is continuous and piecewise linear, with kinks at $x=-1,0,1,2$. Here $\mathbf{w}=(1,-1,1,1)$, $\mathbf{b}=(1,0,-1,-2)$, $\mathbf{a}=(1,1.5,-2,1.5)$ and $c=-0.5$.*

**Training uses gradients, not factorizations.** The weights are fitted by minimizing an average loss over the training data. For a linear model that problem has a closed form through QR or the SVD ([[Linear Models]]). For a network it does not, and even Newton's method is out of reach: with $p=10^9$ parameters, the Hessian has $10^{18}$ entries, and solving with it costs $O(p^3)$. The gradient, by contrast, costs about as much as evaluating the network, thanks to backpropagation: the chain rule as a product of Jacobians, evaluated from the output backwards ([[Neural Networks#Backpropagation]]). So networks are trained by [[First-Order Optimization#Stochastic gradient descent|stochastic gradient descent]] on mini-batches and its [[First-Order Optimization#Momentum and adaptive methods|momentum and adaptive variants]]. The same holds for the policy networks of [[Policy Gradients]], which follow a sampled estimate of the gradient of the expected return.

Linear algebra does not disappear; it moves into the analysis and the design.

- **Conditioning decides speed.** Gradient descent converges at a rate set by the condition number of the Hessian. Adaptive methods such as Adam act as a cheap diagonal preconditioner, and normalization layers improve the conditioning of the problem.
- **Singular values decide stability.** The backward pass multiplies by the transposed layer Jacobians, so gradients explode or vanish when their singular values sit far from 1. Orthogonal initialization (via QR) starts every singular value at exactly 1.[^saxe2014]
- **Low rank is everywhere.** The attention scores of a transformer, a product of a query matrix and a transposed key matrix, have rank at most the key dimension; LoRA adapts a model with rank-$r$ updates; spectral normalization controls $\sigma_1$ layer by layer.

## Key takeaways

- A matrix product is a sum of rank-one pieces, column times row. Every factorization is a way of writing a matrix as such layers with extra structure.
- $\mathbf{C}\mathbf{R}$ keeps actual columns and exposes the rank; $\mathbf{L}\mathbf{U}$ records elimination and solves systems in $O(n^2)$ after an $O(n^3)$ factorization; $\mathbf{Q}\mathbf{R}$ orthogonalizes and solves least squares stably.
- $\mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^\top$ gives the principal axes of a symmetric matrix; $\mathbf{X}\boldsymbol{\Lambda}\mathbf{X}^{-1}$ makes powers and dynamics transparent but can fail or be ill-conditioned; $\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$ exists for every matrix and gives the four subspaces, the norm, the condition number and the best low-rank approximation.
- Eigenvalues and singular values agree only for symmetric positive semi-definite matrices. Use eigenvalues for repeated application, singular values for a single application.
- Factor, don't invert. Triangular factors are cheap; orthogonal factors are safe.
- Block elimination proves $\mathbf{K}=\mathbf{J}\mathbf{W}^{-1}\mathbf{H}$ for a rank-$r$ matrix, giving $\mathbf{A}=\mathbf{C}\mathbf{R}$ and an exact column–row (CUR) form.
- Neural networks compose affine maps with ReLUs into piecewise linear functions and are trained by stochastic gradients, not by factorizations. Conditioning and singular values still govern how well that training works.

## Where this shows up next

- [[Probability Basics]] starts the probability half of the series. Covariance matrices, multivariate Gaussians and Markov chains there are all symmetric or stochastic matrices, analysed with the factorizations above.
- [[Matrix Factorization and Dimensionality Reduction]] is the full treatment of the SVD, Eckart–Young, NMF, CUR and PCA.
- [[Linear Models]] solves least squares with the normal equations, QR and the pseudo-inverse; [[Neural Networks]] trains compositions of affine maps with backpropagation and the optimizers of [[First-Order Optimization]].
- [[Markov Chains]] uses the eigenvalue $\lambda=1$ and the spectral gap; [[Introduction to Reinforcement Learning]] and [[Dynamic Programming]] solve and iterate the Bellman equation $\mathbf{v}=\mathbf{r}^\pi+\gamma\mathbf{P}^\pi\mathbf{v}$.

<!-- References: footnote definitions below render as the numbered "References" list at the bottom of the page -->

[^strang2020five]: Strang, G. (2020). *Five Factorizations of a Matrix*. A 2020 Vision of Linear Algebra, MIT OpenCourseWare RES.18-010. [video](https://www.youtube.com/watch?v=nTwRjQ4xqUc), [course page](https://ocw.mit.edu/courses/res-18-010-a-2020-vision-of-linear-algebra-spring-2020/)
[^strang2023cr]: Strang, G. (2023). *Elimination and Factorization A = CR*. MIT OpenCourseWare RES.18-010. [video](https://www.youtube.com/watch?v=PrErxYzSANo); see also Strang, G. (2023). *Introduction to Linear Algebra* (6th ed.). Wellesley-Cambridge Press. [book website](https://math.mit.edu/linearalgebra/)
[^saxe2014]: Saxe, A. M., McClelland, J. L., & Ganguli, S. (2014). *Exact Solutions to the Nonlinear Dynamics of Learning in Deep Linear Neural Networks*. ICLR. [arXiv:1312.6120](https://arxiv.org/abs/1312.6120)
[^hu2022]: Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022). *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
[^miyato2018]: Miyato, T., Kataoka, T., Koyama, M., & Yoshida, Y. (2018). *Spectral Normalization for Generative Adversarial Networks*. ICLR. [arXiv:1802.05957](https://arxiv.org/abs/1802.05957)
[^strangmoler2022]: Strang, G., & Moler, C. (2022). *LU and CR Elimination*. SIAM Review, 64(1), 181–190. [doi:10.1137/20M1358694](https://doi.org/10.1137/20M1358694)
