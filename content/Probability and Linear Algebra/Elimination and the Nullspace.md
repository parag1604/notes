---
title: "Elimination: LU, Reduced Row Echelon Form and the Nullspace"
tags:
  - linear-algebra
  - foundations
date: 2026-11-12
draft: false
prev: "[[Column Space and Rank]]"
next: "[[The Four Fundamental Subspaces]]"
---

> **Prerequisites:** [[Column Space and Rank]] (the column picture of $\mathbf{A}\mathbf{x}$, the column space, independence, rank, and $\mathbf{A}=\mathbf{C}\mathbf{R}$).
>
> **Series:** this is part 3 of 8. [[Multivariable Calculus]] → [[Column Space and Rank]] → **Elimination and the Nullspace** → [[The Four Fundamental Subspaces]] → [[Orthogonality and Least Squares]] → [[Eigenvalues and Eigenvectors]] → [[First-Order Optimization]] → [[Five Factorizations of a Matrix]].
>
> **TL;DR:**
> - **Elimination** subtracts multiples of one row from the rows below it until the matrix is upper triangular. Recording the multipliers gives $\mathbf{A}=\mathbf{L}\mathbf{U}$ (or $\mathbf{P}\mathbf{A}=\mathbf{L}\mathbf{U}$ when rows must be exchanged). Factoring costs about $\tfrac23n^3$ operations, and each later solve only about $2n^2$.
> - Continuing until every pivot is 1 with zeros above and below gives the **reduced row echelon form**. Its non-zero rows are exactly the $\mathbf{R}$ of $\mathbf{A}=\mathbf{C}\mathbf{R}$, and its pivot columns locate the independent columns $\mathbf{C}$.
> - The **nullspace** $N(\mathbf{A})$ is every $\mathbf{x}$ with $\mathbf{A}\mathbf{x}=\mathbf{0}$. Each free column gives one **special solution**, and they form a basis. So $\dim N(\mathbf{A})=n-r$: the **counting theorem**.
> - Every solution of $\mathbf{A}\mathbf{x}=\mathbf{b}$ is one particular solution plus a nullspace vector. The rank, compared with $m$ and $n$, says at a glance whether there are 0, 1 or infinitely many solutions.

## Motivation

[[Column Space and Rank]] answered two questions by eye: which columns are independent, and which right-hand sides $\mathbf{b}$ can be reached. That works for a $3\times4$ matrix of small integers. It does not work for a $1000\times1000$ system from a physics simulation, or for a design matrix with ten thousand rows. We need a mechanical procedure, and the oldest one is still the best: **elimination**, which Gauss used and every linear solver still runs at its core.

Elimination does three jobs in this lecture. On a square invertible matrix it produces $\mathbf{A}=\mathbf{L}\mathbf{U}$, which is how computers solve $\mathbf{A}\mathbf{x}=\mathbf{b}$. On any matrix it produces the reduced row echelon form, which finds $\mathbf{A}=\mathbf{C}\mathbf{R}$ systematically. And it finds **all** solutions: the nullspace of $\mathbf{A}$, and the complete solution of $\mathbf{A}\mathbf{x}=\mathbf{b}$. The last job matters in machine learning whenever a model has more parameters than constraints, because then the solution is a whole family and we must understand its shape.

The lecture follows Strang's videos "Finding the Nullspace: Solving Ax = 0 by Elimination" and "Elimination and Factorization A = CR",[^strangnull]<sup>,</sup>[^strangcr] with our own examples throughout. The textbook treatment is in Chapters 2 and 3 of Strang's *Introduction to Linear Algebra*.[^strang2023]

## Elimination and A = LU

### A 2×2 warm-up

Solve

$$
\begin{aligned}
2x_1+x_2&=3\\
6x_1+5x_2&=11.
\end{aligned}
$$

Subtract **3 times** the first equation from the second. The $x_1$ disappears and we get $2x_2=2$. Now the system is triangular and is solved from the bottom up: $x_2=1$, then $2x_1+1=3$ gives $x_1=1$.

The number 3 is the **multiplier** $\ell_{21}$: the multiple of row 1 subtracted from row 2. It is the entry to be eliminated divided by the **pivot**, the first non-zero entry of the row doing the eliminating: $\ell_{21}=6/2=3$. The matrix has become upper triangular, and the multiplier fits into a lower triangular matrix:

$$
\mathbf{A}=\begin{bmatrix}2&1\\6&5\end{bmatrix}=\underbrace{\begin{bmatrix}1&0\\3&1\end{bmatrix}}_{\mathbf{L}}\underbrace{\begin{bmatrix}2&1\\0&2\end{bmatrix}}_{\mathbf{U}}.
$$

Multiplying out confirms it: row 2 of $\mathbf{L}\mathbf{U}$ is $3\cdot(2,1)+1\cdot(0,2)=(6,5)$. The pivots $2$ and $2$ sit on the diagonal of $\mathbf{U}$.

### A 3×3 example by hand

Take

$$
\mathbf{A}=\begin{bmatrix}2&1&0\\4&5&2\\-2&2&5\end{bmatrix}.
$$

**Column 1.** The pivot is $2$. The multipliers are $\ell_{21}=4/2=2$ and $\ell_{31}=-2/2=-1$. Subtract $2\times$ row 1 from row 2, and $-1\times$ row 1 from row 3 (that is, add row 1):

$$
\begin{bmatrix}2&1&0\\4&5&2\\-2&2&5\end{bmatrix}\to\begin{bmatrix}2&1&0\\0&3&2\\0&3&5\end{bmatrix}.
$$

**Column 2.** The pivot is now $3$, and $\ell_{32}=3/3=1$. Subtract row 2 from row 3:

$$
\begin{bmatrix}2&1&0\\0&3&2\\0&3&5\end{bmatrix}\to\begin{bmatrix}2&1&0\\0&3&2\\0&0&3\end{bmatrix}=\mathbf{U}.
$$

The pivots are $2,3,3$, all non-zero, so $\mathbf{A}$ is invertible (its determinant is their product, $18$). The multipliers go into $\mathbf{L}$ **in the positions they eliminated**, with ones on the diagonal:

$$
\mathbf{L}=\begin{bmatrix}1&0&0\\2&1&0\\-1&1&1\end{bmatrix},\qquad\mathbf{A}=\mathbf{L}\mathbf{U}.
$$

![[Probability and Linear Algebra/images/elimination-and-the-nullspace-lu-steps.svg]]
*Figure 1. Elimination on the $3\times3$ example. Each step uses a pivot (blue) to create zeros below it (green). The result is the upper triangular $\mathbf{U}$, and the multipliers $\ell_{21}=2$, $\ell_{31}=-1$, $\ell_{32}=1$ (orange) fill the lower triangle of $\mathbf{L}$ without any further arithmetic.*

> [!tip] Intuition: why the multipliers land in L untouched
> Read $\mathbf{A}=\mathbf{L}\mathbf{U}$ by rows: row $i$ of $\mathbf{A}$ is a combination of the rows of $\mathbf{U}$ with weights from row $i$ of $\mathbf{L}$. Elimination produced row 3 of $\mathbf{U}$ by taking row 3 of $\mathbf{A}$, adding back row 1 of $\mathbf{U}$ ($\ell_{31}=-1$) and subtracting row 2 of $\mathbf{U}$ ($\ell_{32}=1$). Undoing those steps rebuilds row 3 of $\mathbf{A}$ as $-1\cdot(2,1,0)+1\cdot(0,3,2)+1\cdot(0,0,3)=(-2,2,5)$. The weights $(-1,1,1)$ are row 3 of $\mathbf{L}$. Elimination subtracts multiples of rows of $\mathbf{U}$; $\mathbf{L}$ adds them back.

### Solving with two triangular systems

Once $\mathbf{A}=\mathbf{L}\mathbf{U}$ is known, $\mathbf{A}\mathbf{x}=\mathbf{b}$ becomes $\mathbf{L}(\mathbf{U}\mathbf{x})=\mathbf{b}$. Call $\mathbf{z}=\mathbf{U}\mathbf{x}$ and solve two triangular systems:

1. **Forward substitution**, $\mathbf{L}\mathbf{z}=\mathbf{b}$, top to bottom.
2. **Back substitution**, $\mathbf{U}\mathbf{x}=\mathbf{z}$, bottom to top.

With $\mathbf{b}=(1,3,6)$, forward substitution gives $z_1=1$, then $2z_1+z_2=3$ so $z_2=1$, then $-z_1+z_2+z_3=6$ so $z_3=6$. Back substitution gives $3x_3=6$ so $x_3=2$, then $3x_2+2x_3=1$ so $x_2=-1$, then $2x_1+x_2=1$ so $x_1=1$. The solution is $\mathbf{x}=(1,-1,2)$; checking, $\mathbf{A}\mathbf{x}=(2-1+0,\ 4-5+4,\ -2-2+10)=(1,3,6)$.

Forward substitution is just elimination applied to $\mathbf{b}$: it performs on the right-hand side the same row operations that turned $\mathbf{A}$ into $\mathbf{U}$.

> [!example]- Code: A = LU by elimination, then two triangular solves
> ```python
> import numpy as np
>
> def lu_nopivot(A):
>     """A = L U by elimination, assuming every pivot is non-zero (no row exchanges)."""
>     U = A.astype(float).copy(); n = len(U); L = np.eye(n)
>     for j in range(n - 1):                       # pivot U[j, j]
>         for i in range(j + 1, n):
>             L[i, j] = U[i, j] / U[j, j]          # multiplier l_ij
>             U[i] -= L[i, j] * U[j]               # row i minus l_ij times the pivot row
>     return L, U
>
> def solve_lu(L, U, b):
>     n = len(b); z = np.zeros(n); x = np.zeros(n)
>     for i in range(n):                           # forward substitution: L z = b
>         z[i] = b[i] - L[i, :i] @ z[:i]
>     for i in reversed(range(n)):                 # back substitution: U x = z
>         x[i] = (z[i] - U[i, i+1:] @ x[i+1:]) / U[i, i]
>     return x
>
> A = np.array([[2, 1, 0], [4, 5, 2], [-2, 2, 5]])
> L, U = lu_nopivot(A)
> print(L, U, sep="\n")      # L = [[1,0,0],[2,1,0],[-1,1,1]], U = [[2,1,0],[0,3,2],[0,0,3]]
> print(solve_lu(L, U, np.array([1.0, 3, 6])))     # [ 1. -1.  2.]
> ```
> The update `U[i] -= ...` inside the double loop is the $n^3/3$ multiply–subtract count of [[#What it costs]]. In practice use `scipy.linalg.lu_factor` and `lu_solve`, which also exchange rows (next subsection).

### Row exchanges: PA = LU

Elimination needs a non-zero pivot. Sometimes a zero appears in the pivot position:

$$
\begin{bmatrix}1&2&3\\2&4&7\\3&7&8\end{bmatrix}\to\begin{bmatrix}1&2&3\\0&0&1\\0&1&-1\end{bmatrix}\qquad(\ell_{21}=2,\ \ell_{31}=3).
$$

The second pivot position holds $0$, but there is a non-zero entry below it, so we **exchange rows 2 and 3** and continue. Here nothing is left to eliminate, and $\mathbf{U}$ has pivots $1,1,1$. If the exchange is done to $\mathbf{A}$ before elimination, no zero pivot ever appears, which gives the general form

$$
\mathbf{P}\mathbf{A}=\mathbf{L}\mathbf{U}\quad\text{with}\quad\mathbf{P}=\begin{bmatrix}1&0&0\\0&0&1\\0&1&0\end{bmatrix},
$$

$$
\mathbf{L}=\begin{bmatrix}1&0&0\\3&1&0\\2&0&1\end{bmatrix},\qquad\mathbf{U}=\begin{bmatrix}1&2&3\\0&1&-1\\0&0&1\end{bmatrix}.
$$

A **permutation matrix** $\mathbf{P}$ is the identity with its rows reordered; $\mathbf{P}\mathbf{A}$ reorders the rows of $\mathbf{A}$ the same way. Note that the multipliers in $\mathbf{L}$ travel with their rows: $\ell=3$ belongs to the row $(3,7,8)$, which is now second. To solve $\mathbf{A}\mathbf{x}=\mathbf{b}$, apply the same exchange to $\mathbf{b}$: solve $\mathbf{L}\mathbf{z}=\mathbf{P}\mathbf{b}$, then $\mathbf{U}\mathbf{x}=\mathbf{z}$.

Two remarks:
- **If no non-zero entry exists below a zero pivot, the matrix is singular.** Exchanges cannot help, and $\mathbf{U}$ ends up with a zero on its diagonal. This is the case the rest of the lecture is about.
- **Software exchanges rows even when it does not have to.** A tiny pivot such as $10^{-17}$ is not zero, but dividing by it creates huge multipliers that amplify rounding errors. **Partial pivoting** always picks the entry of largest absolute value in the column as the pivot. `scipy.linalg.lu` on the matrix above picks $3$ as the first pivot, so its factors differ from the hand ones. (It also returns the permutation the other way round: a matrix `P` with `A = P @ L @ U`, which is the transpose of the $\mathbf{P}$ in $\mathbf{P}\mathbf{A}=\mathbf{L}\mathbf{U}$.)

### What it costs

Eliminating below the first pivot of an $n\times n$ matrix updates the $(n-1)\times(n-1)$ block below and to the right of it, one multiply and one subtract per entry. The next pivot updates an $(n-2)\times(n-2)$ block, and so on. The number of multiply–subtract pairs is

$$
\sum_{k=1}^{n-1}k^2\approx\frac{n^3}{3},
$$

so factoring takes about $\tfrac23n^3$ floating-point operations.

Each triangular solve touches each entry of its triangle once, about $n^2$ operations, so the two solves together cost about $2n^2$. **Factoring is the expensive part, and it depends only on $\mathbf{A}$.** With many right-hand sides (the same system solved for many $\mathbf{b}$, or Newton's method reusing a factorization) we factor once and pay only $2n^2$ per solve. At $n=1000$ that is roughly $7\times10^8$ operations for the factorization and $2\times10^6$ per solve. Figure 2 shows the same gap on a laptop.

![[Probability and Linear Algebra/images/elimination-and-the-nullspace-cost.svg]]
*Figure 2. Cost of $\mathbf{L}\mathbf{U}$. (a) Operation counts: factoring grows like $n^3$, the two triangular solves like $n^2$. (b) Measured times for `scipy.linalg.lu_factor` and `lu_solve` on random matrices (best of several runs on one laptop; the numbers vary between machines and runs). At $n=3200$ factoring took about 180 ms and one solve about 5 ms. The measured ratio is smaller than the operation-count ratio because the factorization is better optimized per operation.*

> [!warning] Don't compute the inverse
> $\mathbf{x}=\mathbf{A}^{-1}\mathbf{b}$ is a formula, not an algorithm. Computing $\mathbf{A}^{-1}$ costs about three times as much as factoring, is less accurate, and destroys sparsity. In code, write `np.linalg.solve(A, b)` (which factors and solves) or keep the factorization with `scipy.linalg.lu_factor` and reuse it, never `np.linalg.inv(A) @ b`.

## Reduced row echelon form

### Elimination on a rectangular matrix

Now return to the running example of [[Column Space and Rank]], which is $3\times4$ with rank 2:

$$
\mathbf{A}=\begin{bmatrix}1&2&1&2\\2&4&3&5\\1&2&2&3\end{bmatrix}.
$$

**Column 1.** Pivot $1$. Subtract $2\times$ row 1 from row 2 and $1\times$ row 1 from row 3:

$$
\to\begin{bmatrix}1&2&1&2\\0&0&1&1\\0&0&1&1\end{bmatrix}.
$$

**Column 2.** Below the first row, column 2 is all zeros, and no row exchange can produce a pivot. So column 2 gets **no pivot**: it is a **free column**. Move on to column 3.

**Column 3.** Pivot $1$ in row 2. Subtract row 2 from row 3, which becomes zero. To reach the *reduced* form, also clear **above** the pivot: subtract row 2 from row 1.

$$
\to\begin{bmatrix}1&2&0&1\\0&0&1&1\\0&0&0&0\end{bmatrix}=\operatorname{rref}(\mathbf{A}).
$$

> [!note] Definition: reduced row echelon form
> A matrix is in **reduced row echelon form** when (1) any zero rows are at the bottom, (2) the first non-zero entry of each non-zero row, its **pivot**, is a 1, (3) each pivot lies to the right of the pivot in the row above, and (4) each pivot is the only non-zero entry in its column. Every matrix can be brought to this form by elimination (row subtractions, row exchanges and scaling rows by non-zero numbers), and the result $\operatorname{rref}(\mathbf{A})$ does not depend on the order of the steps.

The columns with pivots are the **pivot columns**; the others are the **free columns**. Here columns 1 and 3 are pivot columns and columns 2 and 4 are free. The number of pivots is the rank, $r=2$.

### The block form [I F; 0 0]

Gather the pivot columns of $\operatorname{rref}(\mathbf{A})$ on the left and the free columns on the right. The pivot columns form the $r\times r$ identity on top of zeros, and the free columns form an $r\times(n-r)$ block $\mathbf{F}$:

$$
\operatorname{rref}(\mathbf{A})=\begin{bmatrix}\mathbf{I}&\mathbf{F}\\\mathbf{0}&\mathbf{0}\end{bmatrix}\mathbf{P}_c.
$$

The **column permutation** $\mathbf{P}_c$ (multiplying from the right) moves the columns back to their original places. For the running example $\mathbf{F}=\begin{bmatrix}2&1\\0&1\end{bmatrix}$, read off from columns 2 and 4, and $\mathbf{P}_c$ swaps columns 2 and 3. Strang writes this permutation simply as $\mathbf{P}$; we add the subscript to keep it apart from the row exchange in $\mathbf{P}\mathbf{A}=\mathbf{L}\mathbf{U}$. The bottom block has $m-r$ zero rows, $\mathbf{I}$ has $r$ columns and $\mathbf{F}$ has $n-r$.

### How elimination finds R in A = CR

The non-zero rows of $\operatorname{rref}(\mathbf{A})$ are

$$
\mathbf{R}=\begin{bmatrix}1&2&0&1\\0&0&1&1\end{bmatrix},
$$

which is exactly the $\mathbf{R}$ that [[Column Space and Rank#A = CR]] built by inspection, and the pivot columns of $\mathbf{A}$ (columns 1 and 3) are exactly the independent columns that went into $\mathbf{C}$. This is no coincidence:

> [!note] Elimination does not change the relations between columns
> Every row operation is invertible, so elimination amounts to multiplying by an invertible matrix $\mathbf{E}$: $\mathbf{E}\mathbf{A}=\operatorname{rref}(\mathbf{A})$. Then $\mathbf{A}\mathbf{x}=\mathbf{0}$ if and only if $\mathbf{E}\mathbf{A}\mathbf{x}=\mathbf{0}$. A relation $\sum_jx_j\mathbf{a}_j=\mathbf{0}$ among the columns of $\mathbf{A}$ holds **exactly when** the same relation holds among the columns of $\operatorname{rref}(\mathbf{A})$.

In $\operatorname{rref}(\mathbf{A})$ the relations are visible at a glance: column 2 is $2\times$ column 1, and column 4 is column 1 plus column 3. So the same is true in $\mathbf{A}$, and indeed $(2,4,2)=2(1,2,1)$ and $(2,5,3)=(1,2,1)+(1,3,2)$. Elimination is the greedy column scan of the previous lecture, done by arithmetic: a column gets a pivot exactly when it is not a combination of the columns to its left, and the free columns of $\operatorname{rref}(\mathbf{A})$ hold the recipes. In matrix form, the dependent columns of $\mathbf{A}$ are $\mathbf{C}\mathbf{F}$:

$$
\mathbf{C}\mathbf{F}=\begin{bmatrix}1&1\\2&3\\1&2\end{bmatrix}\begin{bmatrix}2&1\\0&1\end{bmatrix}=\begin{bmatrix}2&2\\4&5\\2&3\end{bmatrix}=\text{columns 2 and 4 of }\mathbf{A}.
$$

> [!tip] Intuition: what elimination really computes
> Let $\mathbf{W}$ be the $r\times r$ block where the first $r$ independent rows meet the $r$ pivot columns, and $\mathbf{H}$ the same rows in the free columns. In the running example $\mathbf{W}=\begin{bmatrix}1&1\\2&3\end{bmatrix}$ and $\mathbf{H}=\begin{bmatrix}2&2\\4&5\end{bmatrix}$. Turning $\mathbf{W}$ into $\mathbf{I}$ is multiplication by $\mathbf{W}^{-1}$, and it turns $\mathbf{H}$ into $\mathbf{W}^{-1}\mathbf{H}=\mathbf{F}$. Elimination inverts $\mathbf{W}$, the invertible core of $\mathbf{A}$, and the remaining $m-r$ rows become zero because they depend on the first $r$. [[Five Factorizations of a Matrix#Block elimination and the CR connection]] develops this block view.[^strangmoler2022]

![[Probability and Linear Algebra/images/elimination-and-the-nullspace-rref.svg]]
*Figure 3. Elimination on the running example. The pivot columns (blue) become columns of the identity, the free columns (orange) become $\mathbf{F}$, and the last row becomes zero because row 3 of $\mathbf{A}$ equals row 2 minus row 1. Right: the two special solutions, one per free column (next section). Each sets its own free variable to 1 (orange) and the other free variable to 0.*

## The nullspace

### Solutions of Ax = 0

> [!note] Definition: nullspace
> The **nullspace** of $\mathbf{A}\in\mathbb{R}^{m\times n}$ is $N(\mathbf{A})=\{\mathbf{x}\in\mathbb{R}^n:\mathbf{A}\mathbf{x}=\mathbf{0}\}$.

It is a subspace of $\mathbb{R}^n$ (the space of inputs, not outputs): if $\mathbf{A}\mathbf{x}=\mathbf{0}$ and $\mathbf{A}\mathbf{x}'=\mathbf{0}$ then $\mathbf{A}(c\mathbf{x}+d\mathbf{x}')=\mathbf{0}$. In the column picture, a nullspace vector is a recipe for combining the columns into zero, so a non-zero nullspace exists exactly when the columns are dependent. By the boxed fact above, $N(\mathbf{A})=N(\operatorname{rref}(\mathbf{A}))$, and the reduced system is easy to solve.

### Special solutions

For the running example, $\operatorname{rref}(\mathbf{A})\mathbf{x}=\mathbf{0}$ reads

$$
\begin{aligned}
x_1+2x_2+x_4&=0\\
x_3+x_4&=0.
\end{aligned}
$$

The **free variables** $x_2$ and $x_4$ (those of the free columns) can be chosen at will, and each choice determines the **pivot variables** $x_1$ and $x_3$. Make the simplest choices, one free variable equal to 1 and the others 0:

- $x_2=1$, $x_4=0$: then $x_1=-2$ and $x_3=0$, so $\mathbf{s}_1=(-2,1,0,0)$.
- $x_2=0$, $x_4=1$: then $x_1=-1$ and $x_3=-1$, so $\mathbf{s}_2=(-1,0,-1,1)$.

These are the **special solutions**. Check them in the column picture: $\mathbf{A}\mathbf{s}_1=-2\mathbf{a}_1+\mathbf{a}_2=\mathbf{0}$ says "column 2 is twice column 1", and $\mathbf{A}\mathbf{s}_2=-\mathbf{a}_1-\mathbf{a}_3+\mathbf{a}_4=\mathbf{0}$ says "column 4 is column 1 plus column 3". **Each special solution is one dependency among the columns.**

Every nullspace vector is a combination of the special solutions. If $\mathbf{x}\in N(\mathbf{A})$ has free variables $x_2=c$ and $x_4=d$, then $\mathbf{x}-c\,\mathbf{s}_1-d\,\mathbf{s}_2$ is in the nullspace with both free variables zero, and the two equations then force $x_1=x_3=0$ as well. So

$$
N(\mathbf{A})=\{c\,\mathbf{s}_1+d\,\mathbf{s}_2\},\qquad\text{a plane in }\mathbb{R}^4.
$$

### The nullspace matrix

Put the special solutions into the columns of a **nullspace matrix** $\mathbf{N}$. In the permuted order (pivot variables first, then free variables) the pattern is always the same: the free variables form the identity, and the pivot variables are $-\mathbf{F}$, because $\begin{bmatrix}\mathbf{I}&\mathbf{F}\end{bmatrix}\begin{bmatrix}-\mathbf{F}\\\mathbf{I}\end{bmatrix}=-\mathbf{F}+\mathbf{F}=\mathbf{0}$. Undoing the column permutation,

$$
\mathbf{N}=\mathbf{P}_c^\top\begin{bmatrix}-\mathbf{F}\\\mathbf{I}\end{bmatrix},\qquad\operatorname{rref}(\mathbf{A})\,\mathbf{N}=\mathbf{0}.
$$

For the running example, $-\mathbf{F}$ on top of $\mathbf{I}$ gives the rows for $(x_1,x_3)$ and then $(x_2,x_4)$; putting them back in the order $(x_1,x_2,x_3,x_4)$:

$$
\begin{bmatrix}-\mathbf{F}\\\mathbf{I}\end{bmatrix}=\begin{bmatrix}-2&-1\\0&-1\\1&0\\0&1\end{bmatrix}
\quad\longrightarrow\quad
\mathbf{N}=\begin{bmatrix}-2&-1\\1&0\\0&-1\\0&1\end{bmatrix}.
$$

The columns of $\mathbf{N}$ are $\mathbf{s}_1$ and $\mathbf{s}_2$, as found above.

> [!example]- Code: rref and the nullspace in numpy (checked against scipy)
> ```python
> import numpy as np
> from scipy.linalg import null_space
>
> def rref(A, tol=1e-10):
>     R = A.astype(float).copy()
>     m, n = R.shape
>     pivots, i = [], 0
>     for j in range(n):
>         if i == m:
>             break
>         p = i + np.argmax(np.abs(R[i:, j]))     # partial pivoting: largest entry
>         if abs(R[p, j]) < tol:
>             continue                            # no pivot here: column j is free
>         R[[i, p]] = R[[p, i]]                   # row exchange
>         R[i] /= R[i, j]                         # scale the pivot to 1
>         for k in range(m):
>             if k != i:
>                 R[k] -= R[k, j] * R[i]          # clear the column above and below
>         pivots.append(j)
>         i += 1
>     return R, pivots
>
> def nullspace(A):
>     R, piv = rref(A)
>     n = A.shape[1]
>     free = [j for j in range(n) if j not in piv]
>     N = np.zeros((n, len(free)))
>     for k, f in enumerate(free):
>         N[f, k] = 1                             # this free variable = 1, the others 0
>         N[piv, k] = -R[:len(piv), f]            # pivot variables from R x = 0
>     return N
>
> A = np.array([[1, 2, 1, 2], [2, 4, 3, 5], [1, 2, 2, 3]], dtype=float)
> R, piv = rref(A)
> N = nullspace(A)
> print(piv)                                      # [0, 2]
> print(R + 0.0)                                  # [[1,2,0,1],[0,0,1,1],[0,0,0,0]]
> print(N + 0.0)                                  # columns (-2,1,0,0) and (-1,0,-1,1)
> print(np.allclose(A @ N, 0))                    # True
> Ns = null_space(A)                              # orthonormal basis from the SVD
> print(np.linalg.matrix_rank(np.hstack([N, Ns])) == N.shape[1])   # same subspace: True
> ```
> `scipy.linalg.null_space` returns a different basis of the same plane: orthonormal vectors computed from the SVD, which is what to use in practice because it does not depend on a pivot tolerance. The test stacks both bases side by side and checks that the rank does not grow. The tolerance `tol` in `rref` is the weak point discussed in [[Column Space and Rank#Pros and cons of CR]]: deciding that a pivot "is zero" is a judgement call in floating point.

## The counting theorem

The construction above produced one special solution per free column. That is the whole proof of the central count of this lecture.

> [!note] Theorem: the counting theorem
> For $\mathbf{A}\in\mathbb{R}^{m\times n}$ of rank $r$,
>
> $$
> \dim N(\mathbf{A})=n-r.
> $$
>
> Equivalently, $r+\dim N(\mathbf{A})=n$: the number of independent columns plus the number of independent dependencies equals the number of columns. (This is also called the **rank–nullity theorem**.)

> [!proof]- Proof
> Elimination finds $r$ pivot columns and $n-r$ free columns. The $n-r$ special solutions are **independent**: in the rows of the free variables they form the identity matrix, so a combination of them is zero only if every coefficient is zero. They **span** $N(\mathbf{A})$: for any $\mathbf{x}\in N(\mathbf{A})$, subtracting the special solutions weighted by its free variables leaves a nullspace vector whose free variables are all zero, and the $r$ equations of $\operatorname{rref}(\mathbf{A})\mathbf{x}=\mathbf{0}$ then force its pivot variables to be zero too. A spanning independent set is a basis, so $\dim N(\mathbf{A})=n-r$.

Two consequences are worth stating on their own.

- **A wide matrix always has a non-zero nullspace.** If $n>m$ then $r\le m<n$, so $n-r\ge n-m\ge1$. More unknowns than equations means at least $n-m$ free variables and a non-zero solution of $\mathbf{A}\mathbf{x}=\mathbf{0}$.
- **Independent columns mean $N(\mathbf{A})=\{\mathbf{0}\}$.** If $r=n$ there are no free columns, and the only solution of $\mathbf{A}\mathbf{x}=\mathbf{0}$ is $\mathbf{x}=\mathbf{0}$. For a square matrix this is invertibility.

In the Celsius–Fahrenheit design matrix of [[Column Space and Rank#The data matrix and its column space]], $n=3$ columns and $r=2$, so the nullspace is a line, spanned by $(32,1.8,-1)$: the one dependency $32\cdot\mathbf{1}+1.8\,\mathbf{t}_{\mathrm{C}}-\mathbf{t}_{\mathrm{F}}=\mathbf{0}$. For a model with $D$ weights fitted to $N<D$ data points, the nullspace of $\mathbf{X}$ has dimension at least $D-N$: that many directions in weight space change nothing about the training predictions.

## The complete solution to Ax = b

### Elimination on the augmented matrix

To solve $\mathbf{A}\mathbf{x}=\mathbf{b}$, run elimination on the **augmented matrix** $[\mathbf{A}\ \mathbf{b}]$, so that $\mathbf{b}$ receives the same row operations. For the running example and a general $\mathbf{b}$, the steps used above turn the last column into

$$
\begin{bmatrix}\mathbf{A}&\mathbf{b}\end{bmatrix}\to\left[\begin{array}{cccc|c}1&2&0&1&3b_1-b_2\\0&0&1&1&b_2-2b_1\\0&0&0&0&b_1-b_2+b_3\end{array}\right].
$$

The last row reads $0=b_1-b_2+b_3$. **This is the solvability condition**, the same plane $C(\mathbf{A})$ found geometrically in [[Column Space and Rank#The column space]], now produced by arithmetic. A zero row in $\operatorname{rref}(\mathbf{A})$ always produces such a condition: $m-r$ zero rows give $m-r$ conditions on $\mathbf{b}$.

- For $\mathbf{b}=(1,0,0)$ the last row reads $0=1$. No solution.
- For $\mathbf{b}=(3,7,4)$ the last entries are $(2,1,0)$, so the reduced system is $x_1+2x_2+x_4=2$ and $x_3+x_4=1$.

### Particular plus nullspace

To find one solution, set all free variables to zero and read off the pivot variables: $x_2=x_4=0$ gives $x_1=2$ and $x_3=1$, so the **particular solution** is $\mathbf{x}_p=(2,0,1,0)$. In the column picture, $2\mathbf{a}_1+\mathbf{a}_3=(2,4,2)+(1,3,2)=(3,7,4)$. Then

> [!note] The complete solution
> If $\mathbf{A}\mathbf{x}_p=\mathbf{b}$, then the solutions of $\mathbf{A}\mathbf{x}=\mathbf{b}$ are exactly $\mathbf{x}=\mathbf{x}_p+\mathbf{x}_{\mathrm{null}}$ with $\mathbf{x}_{\mathrm{null}}\in N(\mathbf{A})$.

The proof is one line each way. If $\mathbf{A}\mathbf{x}_{\mathrm{null}}=\mathbf{0}$ then $\mathbf{A}(\mathbf{x}_p+\mathbf{x}_{\mathrm{null}})=\mathbf{b}+\mathbf{0}=\mathbf{b}$. Conversely, if $\mathbf{A}\mathbf{x}=\mathbf{b}$ then $\mathbf{A}(\mathbf{x}-\mathbf{x}_p)=\mathbf{b}-\mathbf{b}=\mathbf{0}$, so $\mathbf{x}-\mathbf{x}_p\in N(\mathbf{A})$. For the running example,

$$
\mathbf{x}=\begin{bmatrix}2\\0\\1\\0\end{bmatrix}+c\begin{bmatrix}-2\\1\\0\\0\end{bmatrix}+d\begin{bmatrix}-1\\0\\-1\\1\end{bmatrix},\qquad c,d\in\mathbb{R}.
$$

Geometrically, the solution set is the nullspace **shifted** by $\mathbf{x}_p$: a plane through the origin moved to pass through $\mathbf{x}_p$. It is not a subspace (it misses $\mathbf{0}$ unless $\mathbf{b}=\mathbf{0}$), but it has the same shape and dimension as $N(\mathbf{A})$. Figure 4 shows this in $\mathbb{R}^3$, where it can be drawn.

![[Probability and Linear Algebra/images/elimination-and-the-nullspace-solution-sets.svg]]
*Figure 4. The complete solution is the nullspace shifted by a particular solution. (a) One equation $x_1+x_2+x_3=3$ in three unknowns: $N(\mathbf{A})$ is the plane $x_1+x_2+x_3=0$, and the solutions form the parallel plane through $\mathbf{x}_p=(3,0,0)$. (b) Two equations, $x_1+x_2+x_3=3$ and $x_1+2x_2+3x_3=6$: elimination gives the special solution $\mathbf{s}=(1,-2,1)$ and $\mathbf{x}_p=(0,3,0)$, so the solutions form a line parallel to $N(\mathbf{A})$.*

The particular solution is not unique (any solution can play its role), and neither is the basis of the nullspace. The *set* of solutions is unique. Among all solutions there is one of **smallest length**, the one with no nullspace component; it lies in the row space, and finding it is the job of the pseudo-inverse in [[The Four Fundamental Subspaces#A teaser: the pseudo-inverse]].

### The four cases

The rank $r$ compared with $m$ (equations) and $n$ (unknowns) decides everything. Since $r\le m$ and $r\le n$, there are four cases, each with its own shape of $\operatorname{rref}(\mathbf{A})$:

| Ranks | Shape of $\mathbf{A}$ | $\operatorname{rref}(\mathbf{A})$ | Solutions of $\mathbf{A}\mathbf{x}=\mathbf{b}$ |
|---|---|---|---|
| $r=m=n$ | square, invertible | $\mathbf{I}$ | exactly 1, for every $\mathbf{b}$ |
| $r=n<m$ | tall, independent columns | $\begin{bmatrix}\mathbf{I}\\\mathbf{0}\end{bmatrix}$ | 0 or 1 |
| $r=m<n$ | wide, independent rows | $\begin{bmatrix}\mathbf{I}&\mathbf{F}\end{bmatrix}\mathbf{P}_c$ | infinitely many, for every $\mathbf{b}$ |
| $r<m$, $r<n$ | neither | $\begin{bmatrix}\mathbf{I}&\mathbf{F}\\\mathbf{0}&\mathbf{0}\end{bmatrix}\mathbf{P}_c$ | 0 or infinitely many |

The pattern follows from two questions. **Is there a zero row?** (Is $r<m$?) Then some $\mathbf{b}$ fail the solvability conditions, and "0 solutions" is possible. **Is there a free column?** (Is $r<n$?) Then $N(\mathbf{A})\ne\{\mathbf{0}\}$, and any solution comes with infinitely many others. Figure 5 shows one small example of each case in the plane.

![[Probability and Linear Algebra/images/elimination-and-the-nullspace-cases.svg]]
*Figure 5. The four cases for two unknowns, in the row picture. (a) Two independent equations meet in one point. (b) Three equations in two unknowns: a third line through $(2,1)$ keeps the solution, while the dashed alternative $x_1+2x_2=5$ misses it and leaves no solution. (c) One equation, $x_1+2x_2=4$: a whole line of solutions $\mathbf{x}_p+c\,\mathbf{s}$ with $\mathbf{x}_p=(4,0)$ and $\mathbf{s}=(-2,1)$. (d) Two dependent equations: either they describe the same line (infinitely many solutions) or parallel lines (none).*

The running example is in the last case ($r=2<m=3$ and $r<n=4$): $\mathbf{b}=(1,0,0)$ gave no solution and $\mathbf{b}=(3,7,4)$ a two-parameter family.

### The four cases in machine learning

Fitting a linear model exactly means solving $\mathbf{X}\mathbf{w}=\mathbf{y}$ with $N$ equations (data points) and $D$ unknowns (weights), so the table above is a table of regression regimes.

- **$N>D$ with independent features ($r=D<N$), the classical case.** At most one exact fit, and with noisy targets $\mathbf{y}\notin C(\mathbf{X})$, so in practice none. We settle for the closest fit: least squares in [[Orthogonality and Least Squares#Least squares]], which has a unique solution because the nullspace of $\mathbf{X}$ is $\{\mathbf{0}\}$.
- **$N<D$ with independent data points ($r=N<D$), the overparameterized case.** Every target vector is fitted exactly, by a $(D-N)$-dimensional family of weight vectors. Which one we get depends on the algorithm and on regularization. Ridge regression ([[Linear Models#MAP estimation and regularization]]) has a unique answer for every penalty $\lambda>0$, and as $\lambda\to0$ that answer tends to the shortest exact fit, the minimum-norm (pseudo-inverse) solution.
- **Redundant features ($r<D$) in either regime.** The nullspace of $\mathbf{X}$ is non-zero, so the weights are not identifiable even when the predictions are. Least squares still has solutions, but infinitely many, and individual coefficients mean nothing until the redundancy is removed or regularized.

## Key takeaways

- Elimination subtracts multiples $\ell_{ij}$ of pivot rows from the rows below. The result is upper triangular $\mathbf{U}$, the multipliers form lower triangular $\mathbf{L}$, and $\mathbf{A}=\mathbf{L}\mathbf{U}$, or $\mathbf{P}\mathbf{A}=\mathbf{L}\mathbf{U}$ with row exchanges. Partial pivoting exchanges rows for numerical stability.
- Factoring costs about $\tfrac23n^3$ operations, each pair of triangular solves about $2n^2$. Factor once, solve many times, and never form $\mathbf{A}^{-1}$.
- The reduced row echelon form is $\begin{bmatrix}\mathbf{I}&\mathbf{F}\\\mathbf{0}&\mathbf{0}\end{bmatrix}\mathbf{P}_c$. Its non-zero rows are the $\mathbf{R}$ of $\mathbf{A}=\mathbf{C}\mathbf{R}$, its pivot columns locate $\mathbf{C}$, and the dependent columns of $\mathbf{A}$ are $\mathbf{C}\mathbf{F}$.
- Elimination does not change the nullspace. Each free column gives a special solution; together they are the columns of $\mathbf{N}=\mathbf{P}_c^\top\begin{bmatrix}-\mathbf{F}\\\mathbf{I}\end{bmatrix}$, a basis of $N(\mathbf{A})$.
- Counting theorem: $\dim N(\mathbf{A})=n-r$. A wide matrix always has a non-zero nullspace.
- The complete solution is $\mathbf{x}_p+\mathbf{x}_{\mathrm{null}}$: the nullspace shifted by any one solution. Zero rows give solvability conditions; free columns give infinitely many solutions.

## Where this shows up next

- [[The Four Fundamental Subspaces]] adds the left nullspace (where the solvability conditions come from) and proves that the nullspace is perpendicular to the row space.
- [[Orthogonality and Least Squares]] handles the "0 solutions" case by projecting $\mathbf{b}$ onto $C(\mathbf{A})$, and replaces elimination by $\mathbf{A}=\mathbf{Q}\mathbf{R}$ for stability.
- [[Eigenvalues and Eigenvectors]] finds eigenvectors as nullspace vectors of $\mathbf{A}$ minus an eigenvalue times $\mathbf{I}$.
- [[Five Factorizations of a Matrix]] compares $\mathbf{L}\mathbf{U}$ and $\mathbf{C}\mathbf{R}$ with the other factorizations and develops block elimination.
- [[Linear Models]] and [[Matrix Factorization and Dimensionality Reduction]] meet underdetermined systems in overparameterized regression and the pseudo-inverse.

<!-- References: footnote definitions below render as the numbered "References" list at the bottom of the page -->

[^strangnull]: Strang, G. (2022). *Part 6: Finding the Nullspace: Solving Ax = 0 by Elimination* [video]. A Vision of Linear Algebra, MIT OpenCourseWare. [YouTube](https://www.youtube.com/watch?v=JFIaRtKNP2E)
[^strangcr]: Strang, G. (2025). *Elimination and Factorization A = CR* [video]. MIT OpenCourseWare. [YouTube](https://www.youtube.com/watch?v=PrErxYzSANo)
[^strang2023]: Strang, G. (2023). *Introduction to Linear Algebra* (6th ed.), Chapters 2–3. Wellesley-Cambridge Press. [book website](https://math.mit.edu/~gs/linearalgebra/)
[^strangmoler2022]: Strang, G., & Moler, C. (2022). *LU and CR Elimination*. SIAM Review, 64(1), 181–190. [doi:10.1137/20M1358694](https://doi.org/10.1137/20M1358694)
