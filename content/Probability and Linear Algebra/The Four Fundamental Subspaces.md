---
title: "The Big Picture: Four Fundamental Subspaces"
tags:
  - linear-algebra
  - foundations
date: 2026-11-12
draft: false
prev: "[[Elimination and the Nullspace]]"
next: "[[Orthogonality and Least Squares]]"
---

> **Prerequisites:** [[Column Space and Rank]] (column space, rank, $\mathbf{A}=\mathbf{C}\mathbf{R}$, column rank = row rank) and [[Elimination and the Nullspace]] (reduced row echelon form, special solutions, the counting theorem $\dim N(\mathbf{A})=n-r$).
>
> **Series:** this is part 4 of 8. [[Multivariable Calculus]] → [[Column Space and Rank]] → [[Elimination and the Nullspace]] → **The Four Fundamental Subspaces** → [[Orthogonality and Least Squares]] → [[Eigenvalues and Eigenvectors]] → [[First-Order Optimization]] → [[Five Factorizations of a Matrix]].
>
> **TL;DR:**
> - Every $m\times n$ matrix $\mathbf{A}$ of rank $r$ comes with four subspaces: the **column space** $C(\mathbf{A})$ and the **left nullspace** $N(\mathbf{A}^\top)$ in $\mathbb{R}^m$, the **row space** $C(\mathbf{A}^\top)$ and the **nullspace** $N(\mathbf{A})$ in $\mathbb{R}^n$. Their dimensions are $r$, $m-r$, $r$ and $n-r$.
> - The pairs are **orthogonal complements**: the nullspace is exactly the set of vectors perpendicular to every row, and the left nullspace is exactly the set perpendicular to every column. This follows in one line from $\mathbf{A}\mathbf{x}=\mathbf{0}$.
> - Every input splits as $\mathbf{x}=\mathbf{x}_r+\mathbf{x}_n$. $\mathbf{A}$ kills $\mathbf{x}_n$ and maps the row space **one-to-one** onto the column space. That invertible core is what the pseudo-inverse inverts.
> - A rank-$r$ matrix is a sum of $r$ rank-one matrices (column times row). For a graph's incidence matrix, the four subspaces are potentials, potential differences, loops of current and balanced flows, and the counting theorem becomes Euler's formula.

## Motivation

The previous two lectures asked two questions about a matrix. [[Column Space and Rank]] asked which right-hand sides $\mathbf{b}$ make $\mathbf{A}\mathbf{x}=\mathbf{b}$ solvable (those in the column space). [[Elimination and the Nullspace]] asked how many solutions there are (one for each vector in the nullspace, added to a particular solution). Those are two of four subspaces, and the other two come for free by asking the same questions of $\mathbf{A}^\top$.

The reward for looking at all four together is a single picture that organizes most of linear algebra: what a matrix keeps, what it destroys, which outputs it can reach, and which it can never reach. Least squares (next lecture) lives in this picture, the SVD gives each of the four subspaces an orthonormal basis, and the pseudo-inverse is the inverse of the part of $\mathbf{A}$ that can be inverted. Strang calls the statement of this lecture *the fundamental theorem of linear algebra*.[^strang1993] This lecture follows his videos "The Big Picture of Linear Algebra" and "Orthogonal Vectors".[^strangbig]<sup>,</sup>[^strangorth] For a textbook treatment see Strang's *Introduction to Linear Algebra*[^strangbook] or, with a machine-learning slant, Deisenroth, Faisal and Ong.[^mml]

## Four subspaces

Let $\mathbf{A}\in\mathbb{R}^{m\times n}$ have rank $r$. Think of $\mathbf{A}$ as a machine that takes inputs $\mathbf{x}\in\mathbb{R}^n$ and produces outputs $\mathbf{A}\mathbf{x}\in\mathbb{R}^m$.

| Subspace | Definition | Lives in | Dimension |
|---|---|---|---|
| column space $C(\mathbf{A})$ | all combinations of the columns, i.e. all outputs $\mathbf{A}\mathbf{x}$ | $\mathbb{R}^m$ | $r$ |
| row space $C(\mathbf{A}^\top)$ | all combinations of the rows | $\mathbb{R}^n$ | $r$ |
| nullspace $N(\mathbf{A})$ | all $\mathbf{x}$ with $\mathbf{A}\mathbf{x}=\mathbf{0}$ | $\mathbb{R}^n$ | $n-r$ |
| left nullspace $N(\mathbf{A}^\top)$ | all $\mathbf{y}$ with $\mathbf{A}^\top\mathbf{y}=\mathbf{0}$, i.e. $\mathbf{y}^\top\mathbf{A}=\mathbf{0}^\top$ | $\mathbb{R}^m$ | $m-r$ |

The left nullspace gets its name from the second form: $\mathbf{y}^\top\mathbf{A}=\mathbf{0}^\top$ multiplies $\mathbf{A}$ from the left. A vector $\mathbf{y}$ in it is a recipe for combining the **rows** of $\mathbf{A}$ into the zero row, just as a nullspace vector combines the columns into the zero column.

**Why these dimensions.** All four follow from facts we already have.
- $\dim C(\mathbf{A})=r$ is the definition of rank.
- $\dim C(\mathbf{A}^\top)=r$ because column rank equals row rank, which [[Column Space and Rank]] proved from $\mathbf{A}=\mathbf{C}\mathbf{R}$.
- $\dim N(\mathbf{A})=n-r$ is the [[Elimination and the Nullspace#The counting theorem|counting theorem]]: there is one special solution per free column, and there are $n-r$ free columns.
- $\dim N(\mathbf{A}^\top)=m-r$ is the counting theorem applied to $\mathbf{A}^\top$, which has $m$ columns and the same rank $r$.

So in $\mathbb{R}^n$ the dimensions add up to $r+(n-r)=n$, and in $\mathbb{R}^m$ to $r+(m-r)=m$. That is a first hint that each pair fills its whole space.

### A worked example

Take the $3\times4$ matrix

$$
\mathbf{A}=\begin{bmatrix}1&2&0&1\\2&4&1&4\\3&6&1&5\end{bmatrix},\qquad m=3,\ n=4.
$$

Looking at the columns: column 2 is twice column 1, and column 4 is column 1 plus twice column 3. Columns 1 and 3 are independent, so $r=2$. Looking at the rows: row 3 is row 1 plus row 2, which confirms $r=2$ from the other side. Elimination gives

$$
\operatorname{rref}(\mathbf{A})=\begin{bmatrix}1&2&0&1\\0&0&1&2\\0&0&0&0\end{bmatrix},
$$

with pivot columns 1 and 3 and free columns 2 and 4. Now read off a basis for each subspace.

- **Column space** (dimension $r=2$): the pivot columns of $\mathbf{A}$ itself, $(1,2,3)$ and $(0,1,1)$. It is a plane in $\mathbb{R}^3$.
- **Row space** (dimension $r=2$): the nonzero rows of the rref, $(1,2,0,1)$ and $(0,0,1,2)$. Elimination combines rows, so it never changes the row space.
- **Nullspace** (dimension $n-r=2$): one special solution per free variable. Setting $x_2=1,\ x_4=0$ and solving gives $\mathbf{s}_1=(-2,1,0,0)$; setting $x_2=0,\ x_4=1$ gives $\mathbf{s}_2=(-1,0,-2,1)$.
- **Left nullspace** (dimension $m-r=1$): the row dependency "row 3 − row 1 − row 2 = 0" is exactly $\mathbf{y}^\top\mathbf{A}=\mathbf{0}^\top$ with $\mathbf{y}=(1,1,-1)$.

Check the last one against the columns: $(1,1,-1)\cdot(1,2,3)=1+2-3=0$ and $(1,1,-1)\cdot(0,1,1)=0$. The left-nullspace vector is perpendicular to the column space. That is not a coincidence, and the rest of the lecture explains why.

![[Probability and Linear Algebra/images/the-four-fundamental-subspaces-column-plane.svg]]
*Figure 1. The column space of the example matrix is the plane $b_3=b_1+b_2$ in $\mathbb{R}^3$, spanned by columns $\mathbf{a}_1$ and $\mathbf{a}_3$ (column $\mathbf{a}_4=\mathbf{a}_1+2\mathbf{a}_3$ adds nothing new). The left nullspace is the line through $\mathbf{y}=(1,1,-1)$, the normal to that plane. A right-hand side $\mathbf{b}$ is reachable exactly when $\mathbf{y}^\top\mathbf{b}=b_1+b_2-b_3=0$.*

The left nullspace gives a **test for solvability**. Since $\mathbf{y}^\top\mathbf{A}=\mathbf{0}^\top$, any reachable $\mathbf{b}=\mathbf{A}\mathbf{x}$ satisfies $\mathbf{y}^\top\mathbf{b}=\mathbf{y}^\top\mathbf{A}\mathbf{x}=0$. Here that means $b_3=b_1+b_2$: the third equation must be the sum of the first two, because the third row of $\mathbf{A}$ is. The proof below shows the converse too, so the condition is necessary and sufficient.

> [!example]- Code: bases for the four subspaces
> `scipy.linalg.orth` returns an orthonormal basis for a column space, and `null_space` one for a nullspace (both via the SVD).
> ```python
> import numpy as np
> from scipy.linalg import orth, null_space
>
> A = np.array([[1, 2, 0, 1],
>               [2, 4, 1, 4],
>               [3, 6, 1, 5]], dtype=float)
> m, n = A.shape
> r = np.linalg.matrix_rank(A)
> bases = {"C(A)": orth(A), "C(A^T)": orth(A.T),
>          "N(A)": null_space(A), "N(A^T)": null_space(A.T)}
> for name, B in bases.items():
>     print(f"{name:7s} dim {B.shape[1]}")          # r, r, n - r, m - r
> print(np.abs(bases["C(A^T)"].T @ bases["N(A)"]).max())   # ~1e-16: row space is orthogonal to nullspace
> print(np.abs(bases["C(A)"].T @ bases["N(A^T)"]).max())   # ~1e-16: column space is orthogonal to left nullspace
> ```

## Orthogonal vectors and subspaces

### Orthogonal vectors

Two vectors $\mathbf{x},\mathbf{y}\in\mathbb{R}^n$ are **orthogonal** (perpendicular), written $\mathbf{x}\perp\mathbf{y}$, when their dot product is zero:

$$
\mathbf{x}^\top\mathbf{y}=x_1y_1+\dots+x_ny_n=0.
$$

Why does a zero dot product mean a right angle? Expand the squared length of the sum:

$$
\lVert\mathbf{x}+\mathbf{y}\rVert^2=(\mathbf{x}+\mathbf{y})^\top(\mathbf{x}+\mathbf{y})=\lVert\mathbf{x}\rVert^2+\lVert\mathbf{y}\rVert^2+2\,\mathbf{x}^\top\mathbf{y}.
$$

**Pythagoras' theorem**, $\lVert\mathbf{x}+\mathbf{y}\rVert^2=\lVert\mathbf{x}\rVert^2+\lVert\mathbf{y}\rVert^2$, holds exactly when the cross term $\mathbf{x}^\top\mathbf{y}$ vanishes. In two or three dimensions this is the right-angled triangle; in $\mathbb{R}^n$ we take it as the definition of a right angle. More generally, $\mathbf{x}^\top\mathbf{y}=\lVert\mathbf{x}\rVert\lVert\mathbf{y}\rVert\cos\theta$, the same cosine that defined the correlation in [[Joint Distributions]].

For example, $(1,2,2)$ and $(2,1,-2)$ are orthogonal: $2+2-4=0$. Their lengths are both 3, and the sum $(3,3,0)$ has squared length $18=9+9$.

> [!note] Orthogonal vectors are independent
> If $\mathbf{v}_1,\dots,\mathbf{v}_k$ are nonzero and mutually orthogonal, they are linearly independent. Suppose $c_1\mathbf{v}_1+\dots+c_k\mathbf{v}_k=\mathbf{0}$. Take the dot product with $\mathbf{v}_j$: every cross term vanishes, leaving $c_j\lVert\mathbf{v}_j\rVert^2=0$, so $c_j=0$. Orthogonality is the cleanest possible form of independence, and [[Orthogonality and Least Squares]] exploits it.

> [!tip] Intuition: what the dot product measures
> The dot product $\mathbf{x}^\top\mathbf{y}$ is how much of $\mathbf{y}$ points along $\mathbf{x}$, scaled by the length of $\mathbf{x}$. Zero means none of it: moving along $\mathbf{y}$ does not change your position as measured along $\mathbf{x}$. 3Blue1Brown's "Dot products and duality" shows why this projection picture and the sum-of-products formula agree.[^3b1bdot]

### Orthogonal subspaces

Two subspaces $\mathbf{V}$ and $\mathbf{W}$ of $\mathbb{R}^n$ are **orthogonal** if every vector in $\mathbf{V}$ is orthogonal to every vector in $\mathbf{W}$. It is enough to check the basis vectors, because dot products are linear in each argument.

Two consequences follow immediately.
- Orthogonal subspaces meet only at $\mathbf{0}$: a vector in both would be orthogonal to itself, and $\mathbf{v}^\top\mathbf{v}=\lVert\mathbf{v}\rVert^2=0$ forces $\mathbf{v}=\mathbf{0}$.
- Their dimensions add up to at most $n$: putting a basis of $\mathbf{V}$ next to a basis of $\mathbf{W}$ gives orthogonal, hence independent, vectors, and $\mathbb{R}^n$ holds at most $n$ of those.

> [!warning] Perpendicular in everyday speech is not orthogonal here
> The floor and a wall of a room look perpendicular, but as subspaces of $\mathbb{R}^3$ (the $xy$-plane and the $xz$-plane, say) they are **not** orthogonal. They share the whole $x$-axis, and the vector $(1,0,0)$ is not perpendicular to itself. The dimension count says the same: $2+2>3$. In $\mathbb{R}^3$, a plane can only be orthogonal to a line (its normal) or to $\{\mathbf{0}\}$.

The **orthogonal complement** of a subspace $\mathbf{V}\subseteq\mathbb{R}^n$ is the set of *all* vectors orthogonal to it:

$$
\mathbf{V}^\perp=\{\mathbf{w}\in\mathbb{R}^n:\ \mathbf{w}^\top\mathbf{v}=0\text{ for every }\mathbf{v}\in\mathbf{V}\}.
$$

It is the largest subspace orthogonal to $\mathbf{V}$. In $\mathbb{R}^3$ the complement of a plane through the origin is its normal line, and the complement of a line is the plane perpendicular to it.

## Nullspace ⟂ row space

The key fact of the lecture takes one line. Write the equation $\mathbf{A}\mathbf{x}=\mathbf{0}$ row by row:

$$
\mathbf{A}\mathbf{x}=\begin{bmatrix}\text{row }1\\\vdots\\\text{row }m\end{bmatrix}\mathbf{x}=\begin{bmatrix}(\text{row }1)\cdot\mathbf{x}\\\vdots\\(\text{row }m)\cdot\mathbf{x}\end{bmatrix}=\begin{bmatrix}0\\\vdots\\0\end{bmatrix}.
$$

Each equation says that $\mathbf{x}$ is perpendicular to one row. So a vector is in the nullspace **if and only if** it is perpendicular to every row, and therefore (by linearity) to every combination of rows, which is the whole row space. The "if and only if" matters: it says not only that $N(\mathbf{A})$ is orthogonal to $C(\mathbf{A}^\top)$, but that it contains *every* vector orthogonal to the row space:

$$
N(\mathbf{A})=C(\mathbf{A}^\top)^\perp\quad\text{in }\mathbb{R}^n,\qquad N(\mathbf{A}^\top)=C(\mathbf{A})^\perp\quad\text{in }\mathbb{R}^m.
$$

The second statement is the first one applied to $\mathbf{A}^\top$, whose rows are the columns of $\mathbf{A}$. In words: $\mathbf{y}^\top\mathbf{A}=\mathbf{0}^\top$ says $\mathbf{y}$ is perpendicular to every column.

Check it on the example. The row-space basis $(1,2,0,1)$, $(0,0,1,2)$ against the nullspace basis $(-2,1,0,0)$, $(-1,0,-2,1)$:

$$
\begin{aligned}
(1,2,0,1)\cdot(-2,1,0,0)&=-2+2=0,\\
(1,2,0,1)\cdot(-1,0,-2,1)&=-1+1=0,\\
(0,0,1,2)\cdot(-2,1,0,0)&=0,\\
(0,0,1,2)\cdot(-1,0,-2,1)&=-2+2=0.
\end{aligned}
$$

> [!proof]- The two complements fill their spaces: every $\mathbf{x}$ splits uniquely as $\mathbf{x}_r+\mathbf{x}_n$
> Take a basis of the row space ($r$ vectors) and a basis of the nullspace ($n-r$ vectors). Every vector of the first set is orthogonal to every vector of the second. Within each set the vectors are independent, and a combination of the first set can only cancel a combination of the second if both are zero (the subspaces meet only at $\mathbf{0}$). So all $n$ vectors together are independent, and $n$ independent vectors in $\mathbb{R}^n$ form a basis. Hence every $\mathbf{x}\in\mathbb{R}^n$ is a combination of them: a row-space part plus a nullspace part.
>
> The split is unique. If $\mathbf{x}=\mathbf{x}_r+\mathbf{x}_n=\mathbf{x}_r'+\mathbf{x}_n'$, then $\mathbf{x}_r-\mathbf{x}_r'=\mathbf{x}_n'-\mathbf{x}_n$ lies in both subspaces, so it is $\mathbf{0}$.
>
> The same argument in $\mathbb{R}^m$ splits every $\mathbf{b}$ into a column-space part plus a left-nullspace part. That split is exactly what least squares computes in [[Orthogonality and Least Squares]].

## The big picture

Put the four subspaces together (Figure 2). On the input side, $\mathbb{R}^n$ is split into two perpendicular pieces, the row space and the nullspace. On the output side, $\mathbb{R}^m$ is split into the column space and the left nullspace. Now follow an input $\mathbf{x}=\mathbf{x}_r+\mathbf{x}_n$ through the machine:

$$
\mathbf{A}\mathbf{x}=\mathbf{A}\mathbf{x}_r+\mathbf{A}\mathbf{x}_n=\mathbf{A}\mathbf{x}_r+\mathbf{0}.
$$

- The **nullspace part is destroyed**. It contributes nothing to the output.
- The **row-space part carries everything**. Every output $\mathbf{A}\mathbf{x}$ is also produced by a row-space vector.
- The output always lands in the **column space**. The **left nullspace is never reached**: no input produces an output with a component there.

![[Probability and Linear Algebra/images/the-four-fundamental-subspaces-big-picture.svg]]
*Figure 2. The big picture. The input space $\mathbb{R}^n$ splits into the row space (dimension $r$) and the nullspace (dimension $n-r$), at right angles. The output space $\mathbb{R}^m$ splits into the column space (dimension $r$) and the left nullspace (dimension $m-r$). $\mathbf{A}$ sends the nullspace part of $\mathbf{x}$ to $\mathbf{0}$ and carries the row-space part one-to-one onto the column space.*

**$\mathbf{A}$ is one-to-one from the row space onto the column space.** Two different row-space vectors never give the same output. If $\mathbf{A}\mathbf{x}_r=\mathbf{A}\mathbf{x}_r'$, then $\mathbf{x}_r-\mathbf{x}_r'$ is in the nullspace (it maps to $\mathbf{0}$) and in the row space (it is a difference of row-space vectors). Only $\mathbf{0}$ lies in both, so $\mathbf{x}_r=\mathbf{x}_r'$. And every column-space vector is reached, since $\mathbf{A}\mathbf{x}=\mathbf{A}\mathbf{x}_r$. So hidden inside every matrix, however rectangular or singular, is an **invertible $r\times r$ map** between two $r$-dimensional spaces. Everything else is padding: dimensions that are destroyed (nullspace) or never reached (left nullspace).

> [!example] Example: splitting an input
> In the $3\times4$ example, take $\mathbf{x}=(-1,3,1,3)$. It splits as
>
> $$
> \mathbf{x}=\underbrace{(1,2,1,3)}_{\mathbf{x}_r}+\underbrace{(-2,1,0,0)}_{\mathbf{x}_n}.
> $$
>
> The first piece is in the row space, since it is the sum of the two rref rows $(1,2,0,1)+(0,0,1,2)$. The second is the special solution $\mathbf{s}_1$. They are perpendicular: $-2+2+0+0=0$. So Pythagoras holds, $\lVert\mathbf{x}\rVert^2=20=15+5$, and $\mathbf{A}\mathbf{x}=\mathbf{A}\mathbf{x}_r=(8,23,31)$. (Finding the split for a given $\mathbf{x}$ is a projection onto the row space, the subject of the next lecture.) The output satisfies the solvability test: $8+23-31=0$.

Figure 3 shows the same picture with actual coordinates, for a $2\times2$ matrix of rank 1, where every subspace is a line and the right angles are visible.

![[Probability and Linear Algebra/images/the-four-fundamental-subspaces-rank-one-example.svg]]
*Figure 3. The four subspaces of $\mathbf{B}=\begin{bmatrix}2&1\\4&2\end{bmatrix}$ (rank 1). (a) In the input plane, the row space (multiples of $(2,1)$) is perpendicular to the nullspace (multiples of $(1,-2)$). The input $\mathbf{x}=(1.3,-0.6)$ splits into $\mathbf{x}_r=(0.8,0.4)$ and $\mathbf{x}_n=(0.5,-1)$. (b) In the output plane, $\mathbf{B}\mathbf{x}=\mathbf{B}\mathbf{x}_r=(2,4)$ lands on the column space (multiples of $(1,2)$), and the left nullspace (multiples of $(2,-1)$) is never reached.*

> [!tip] Intuition: watch a matrix squash space
> For a square matrix of rank $r<n$, the nullspace is the set of directions squashed flat, and the column space is the lower-dimensional sheet that everything lands on. 3Blue1Brown's "Inverse matrices, column space and null space" animates exactly this.[^3b1binv]

## Rank-one and rank-r matrices

The simplest nonzero matrices have rank 1. Every rank-one matrix is a column times a row, $\mathbf{u}\mathbf{v}^\top$ with nonzero $\mathbf{u}\in\mathbb{R}^m$ and $\mathbf{v}\in\mathbb{R}^n$, and its four subspaces can be read off at a glance:

$$
(\mathbf{u}\mathbf{v}^\top)\,\mathbf{x}=\mathbf{u}\,(\mathbf{v}^\top\mathbf{x}).
$$

Every output is a multiple of $\mathbf{u}$, so the column space is the line through $\mathbf{u}$. By the same argument for the transpose, the row space is the line through $\mathbf{v}$. The output is zero exactly when $\mathbf{v}^\top\mathbf{x}=0$, so the nullspace is the **hyperplane** $\mathbf{v}^\perp$ of dimension $n-1$. The left nullspace is $\mathbf{u}^\perp$, of dimension $m-1$. The matrix $\mathbf{B}$ of Figure 3 is $\mathbf{u}\mathbf{v}^\top$ with $\mathbf{u}=(1,2)$ and $\mathbf{v}=(2,1)$.

A rank-$r$ matrix is a sum of $r$ rank-one matrices. The factorization $\mathbf{A}=\mathbf{C}\mathbf{R}$ from [[Column Space and Rank]] already provides one such sum: column $k$ of $\mathbf{C}$ times row $k$ of $\mathbf{R}$,

$$
\mathbf{A}=\mathbf{C}\mathbf{R}=\mathbf{c}_1\mathbf{r}_1^\top+\mathbf{c}_2\mathbf{r}_2^\top+\dots+\mathbf{c}_r\mathbf{r}_r^\top.
$$

For the example, $\mathbf{c}_1=(1,2,3)$, $\mathbf{c}_2=(0,1,1)$, $\mathbf{r}_1=(1,2,0,1)$ and $\mathbf{r}_2=(0,0,1,2)$, and Figure 4 shows the two pieces adding up to $\mathbf{A}$. The factorization also displays two of the four subspaces at once: the columns of $\mathbf{C}$ are a basis for the column space, and the rows of $\mathbf{R}$ are a basis for the row space.

![[Probability and Linear Algebra/images/the-four-fundamental-subspaces-rank-one-sum.svg]]
*Figure 4. The rank-2 example matrix as a sum of two rank-one matrices, column of $\mathbf{C}$ times row of $\mathbf{R}$. In a rank-one matrix every row is a multiple of one row and every column a multiple of one column.*

You need at least $r$ rank-one pieces, never fewer: a sum of $k$ rank-one matrices has its column space inside the span of $k$ vectors $\mathbf{u}_1,\dots,\mathbf{u}_k$, so its rank is at most $k$. There are many ways to write $\mathbf{A}$ as such a sum. The best one, with orthogonal $\mathbf{u}$'s, orthogonal $\mathbf{v}$'s and the pieces ordered by importance, is the singular value decomposition $\mathbf{A}=\sum_i\sigma_i\mathbf{u}_i\mathbf{v}_i^\top$ in [[Matrix Factorization and Dimensionality Reduction#Singular value decomposition (SVD)|Matrix Factorization and Dimensionality Reduction]]. It gives orthonormal bases for all four subspaces at once, and keeping only its largest pieces gives the best low-rank approximation. [[Five Factorizations of a Matrix]] compares it with the other ways of splitting a matrix into rank-one pieces.

## Applications

### Graphs and incidence matrices

A directed graph with $n$ nodes and $m$ edges has an $m\times n$ **incidence matrix** $\mathbf{A}$: row $k$ describes edge $k$, with $-1$ in the column of the node it leaves and $+1$ in the column of the node it enters. Take the graph of Figure 5, with 4 nodes and 5 edges:

$$
\mathbf{A}=\begin{bmatrix}-1&1&0&0\\-1&0&1&0\\0&-1&1&0\\0&-1&0&1\\0&0&-1&1\end{bmatrix}
\begin{matrix}e_1:1\to2\\e_2:1\to3\\e_3:2\to3\\e_4:2\to4\\e_5:3\to4\end{matrix}
$$

![[Probability and Linear Algebra/images/the-four-fundamental-subspaces-graph.svg]]
*Figure 5. A directed graph with 4 nodes and 5 edges. The numbers $x_v$ are potentials at the nodes; each edge carries the potential difference $(\mathbf{A}\mathbf{x})_k=x_{\text{head}}-x_{\text{tail}}$. The two small loops span the left nullspace of the incidence matrix.*

All four subspaces have a physical meaning, and they are the heart of Kirchhoff's analysis of electrical networks.[^kirchhoff1847]

**Nullspace: constant potentials.** Think of $\mathbf{x}\in\mathbb{R}^4$ as potentials (voltages) at the nodes. Then $(\mathbf{A}\mathbf{x})_k$ is the potential difference across edge $k$. It is zero on every edge exactly when all the potentials are equal, because the graph is connected. So $N(\mathbf{A})$ is the line through $(1,1,1,1)$, of dimension $1=n-r$, and therefore $r=3$. (Raising every voltage by the same amount changes no current: only differences matter.)

**Column space: differences that are consistent around loops.** Which edge vectors $\mathbf{b}$ are potential differences of some $\mathbf{x}$? With $\mathbf{x}=(0,3,5,6)$ we get $\mathbf{b}=\mathbf{A}\mathbf{x}=(3,5,2,3,1)$. Going around the loop $1\to2\to3\to1$, the differences add to $b_1+b_3-b_2=3+2-5=0$. That is **Kirchhoff's voltage law**: potential differences sum to zero around every loop. The column space is the 3-dimensional set of $\mathbf{b}$ that pass this test on every loop.

**Left nullspace: currents around loops.** Think of $\mathbf{y}\in\mathbb{R}^5$ as currents on the edges. Entry $v$ of $\mathbf{A}^\top\mathbf{y}$ adds the currents entering node $v$ and subtracts those leaving it, so $\mathbf{A}^\top\mathbf{y}=\mathbf{0}$ is **Kirchhoff's current law**: at every node, current in equals current out. Such balanced currents flow around loops. The two small loops in Figure 5 give

$$
\begin{aligned}
\mathbf{y}_1&=(1,-1,1,0,0)&&\text{(loop }1\to2\to3\to1),\\
\mathbf{y}_2&=(0,0,1,-1,1)&&\text{(loop }2\to3\to4\to2),
\end{aligned}
$$

with $-1$ where the loop runs against an edge's direction. They are independent, and $m-r=5-3=2$, so they are a basis. The big outer loop $1\to2\to4\to3\to1$ is $\mathbf{y}_1-\mathbf{y}_2=(1,-1,0,1,-1)$, not a new direction.

Now the orthogonality $C(\mathbf{A})\perp N(\mathbf{A}^\top)$ **is** the voltage law. A vector $\mathbf{b}$ is a set of potential differences exactly when $\mathbf{y}^\top\mathbf{b}=0$ for every loop $\mathbf{y}$. The vector $\mathbf{b}=(1,1,1,1,1)$ fails: $\mathbf{y}_1^\top\mathbf{b}=1$, so no potentials produce a difference of 1 on every edge.

**Row space: what the nodes can absorb.** The row space has dimension $r=3$ and is orthogonal to $(1,1,1,1)$: it consists of the vectors whose entries sum to zero. These are the net inflows at the nodes that a flow on the edges can produce; the total must be zero, because whatever enters the network at one node must leave at another.

**The counting theorem is Euler's formula.** The number of independent loops is $m-r=m-(n-1)$. Rearranged, $(\text{nodes})-(\text{edges})+(\text{loops})=n-m+(m-n+1)=1$ for every connected graph. For a planar graph the independent loops are the bounded faces, and this is Euler's formula $V-E+F=2$ once the outer face is counted.

Finally, $\mathbf{A}^\top\mathbf{A}$ is the **graph Laplacian**: degrees on the diagonal, $-1$ for each edge off the diagonal. Here

$$
\mathbf{A}^\top\mathbf{A}=\begin{bmatrix}2&-1&-1&0\\-1&3&-1&-1\\-1&-1&3&-1\\0&-1&-1&2\end{bmatrix}.
$$

Its nullspace is again the constant vector, since $N(\mathbf{A}^\top\mathbf{A})=N(\mathbf{A})$ (proved in [[Orthogonality and Least Squares#The normal equations|the next lecture]]). The Laplacian is the matrix behind [[Clustering and Expectation Maximization#Spectral clustering|spectral clustering]].

### The data matrix

For a design matrix $\mathbf{X}\in\mathbb{R}^{N\times D}$ in [[Linear Models]], the four subspaces answer modelling questions.
- **Column space** ($\subseteq\mathbb{R}^N$): every prediction vector $\mathbf{X}\mathbf{w}$ the model can make.
- **Nullspace** ($\subseteq\mathbb{R}^D$): redundant feature combinations. If $\mathbf{X}\mathbf{s}=\mathbf{0}$, then $\mathbf{w}$ and $\mathbf{w}+\mathbf{s}$ make identical predictions on the training data, so the data cannot tell them apart. A duplicated feature or a one-hot encoding plus an intercept both produce a nonzero nullspace.
- **Left nullspace** ($\subseteq\mathbb{R}^N$): the directions the model can never fit. The least-squares residual lives here.
- **Row space** ($\subseteq\mathbb{R}^D$): the weight directions the data actually constrains. The minimum-norm solution lives here.

### A teaser: the pseudo-inverse

A matrix with a nonzero nullspace or left nullspace has no inverse. But the big picture shows an invertible core: $\mathbf{A}$ maps the row space one-to-one onto the column space. The **pseudo-inverse** $\mathbf{A}^+$ inverts exactly that core and gives up gracefully on the rest. It maps the column space back to the row space, undoing $\mathbf{A}$, and it sends the left nullspace (which $\mathbf{A}$ never reaches) to $\mathbf{0}$. As a result, $\mathbf{A}^+\mathbf{A}\mathbf{x}=\mathbf{x}_r$: it recovers the row-space part of the input, the only part that survived.

For the rank-one $\mathbf{B}=\mathbf{u}\mathbf{v}^\top$ of Figure 3, the pseudo-inverse is $\mathbf{B}^+=\mathbf{v}\mathbf{u}^\top/(\lVert\mathbf{u}\rVert^2\lVert\mathbf{v}\rVert^2)$. With $\lVert\mathbf{u}\rVert^2=\lVert\mathbf{v}\rVert^2=5$,

$$
\mathbf{B}^+=\frac{1}{25}\begin{bmatrix}2&4\\1&2\end{bmatrix},\qquad\mathbf{B}^+\begin{bmatrix}2\\4\end{bmatrix}=\frac{1}{25}\begin{bmatrix}20\\10\end{bmatrix}=\begin{bmatrix}0.8\\0.4\end{bmatrix}=\mathbf{x}_r.
$$

The pseudo-inverse returned the row-space part of $\mathbf{x}$, not $\mathbf{x}$ itself: the nullspace part $(0.5,-1)$ was lost for good. [[Orthogonality and Least Squares#The four subspaces and least squares|The next lecture]] shows that $\mathbf{A}^+\mathbf{b}$ is the minimum-norm least-squares solution, and [[Matrix Factorization and Dimensionality Reduction#The pseudo-inverse|the SVD]] computes it.

## Key takeaways

- Every $m\times n$ matrix of rank $r$ has four fundamental subspaces: column space ($r$) and left nullspace ($m-r$) in $\mathbb{R}^m$; row space ($r$) and nullspace ($n-r$) in $\mathbb{R}^n$.
- $\mathbf{x}\perp\mathbf{y}$ means $\mathbf{x}^\top\mathbf{y}=0$, which is exactly when Pythagoras holds. Orthogonal subspaces meet only at $\mathbf{0}$ and their dimensions add to at most $n$.
- $\mathbf{A}\mathbf{x}=\mathbf{0}$ says, row by row, that $\mathbf{x}$ is perpendicular to every row. So the nullspace is the orthogonal complement of the row space, and the left nullspace is the orthogonal complement of the column space.
- Every $\mathbf{x}$ splits uniquely as $\mathbf{x}_r+\mathbf{x}_n$. $\mathbf{A}$ kills $\mathbf{x}_n$ and maps the row space one-to-one onto the column space; the left nullspace is never reached. $\mathbf{A}\mathbf{x}=\mathbf{b}$ is solvable iff $\mathbf{b}$ is perpendicular to the left nullspace.
- Rank one means column times row, $\mathbf{u}\mathbf{v}^\top$. Rank $r$ means a sum of $r$ such pieces, and no fewer; $\mathbf{A}=\mathbf{C}\mathbf{R}$ and the SVD are two ways of writing that sum.
- For a graph, the four subspaces are constant potentials, consistent potential differences (voltage law), loop currents (current law) and balanced node inflows, and the counting theorem is Euler's formula.

## Where this shows up next

- [[Orthogonality and Least Squares]] splits $\mathbf{b}$ into its column-space and left-nullspace parts by projection, which is least squares, and shows that the minimum-norm solution lies in the row space.
- [[Eigenvalues and Eigenvectors]] studies square matrices whose input and output spaces coincide, and the symmetric case where all four subspaces line up with orthogonal eigenvectors.
- [[Matrix Factorization and Dimensionality Reduction#Singular value decomposition (SVD)|The SVD]] chooses orthonormal bases for all four subspaces so that $\mathbf{A}$ becomes diagonal, and its pseudo-inverse inverts the row-space-to-column-space core.
- [[Linear Models]] lives in the column space of the design matrix; [[Clustering and Expectation Maximization#Spectral clustering|spectral clustering]] uses the graph Laplacian $\mathbf{A}^\top\mathbf{A}$ built from the incidence matrix.

<!-- References: footnote definitions below render as the numbered "References" list at the bottom of the page -->

[^strang1993]: Strang, G. (1993). *The Fundamental Theorem of Linear Algebra*. The American Mathematical Monthly, 100(9), 848–855. [doi:10.1080/00029890.1993.11990500](https://doi.org/10.1080/00029890.1993.11990500)
[^strangbig]: Strang, G. (2020). *Part 2: The Big Picture of Linear Algebra*. A 2020 Vision of Linear Algebra, MIT OpenCourseWare RES.18-010. [video](https://www.youtube.com/watch?v=rwLOfdfc4dw) · [course page](https://ocw.mit.edu/courses/res-18-010-a-2020-vision-of-linear-algebra-spring-2020/)
[^strangorth]: Strang, G. (2020). *Part 3: Orthogonal Vectors*. A 2020 Vision of Linear Algebra, MIT OpenCourseWare RES.18-010. [video](https://www.youtube.com/watch?v=j8hEnyOiwhw)
[^3b1bdot]: 3Blue1Brown (2016). *Dot products and duality | Chapter 9, Essence of linear algebra*. [video](https://www.youtube.com/watch?v=LyGKycYT2v0)
[^3b1binv]: 3Blue1Brown (2016). *Inverse matrices, column space and null space | Chapter 7, Essence of linear algebra*. [video](https://www.youtube.com/watch?v=uQhTuRlWMxw)
[^kirchhoff1847]: Kirchhoff, G. (1847). *Ueber die Auflösung der Gleichungen, auf welche man bei der Untersuchung der linearen Vertheilung galvanischer Ströme geführt wird*. Annalen der Physik, 148(12), 497–508. [doi:10.1002/andp.18471481202](https://doi.org/10.1002/andp.18471481202)
[^strangbook]: Strang, G. (2023). *Introduction to Linear Algebra* (6th ed.), Chapters 3–4 and 10.1 (graphs and networks). Wellesley-Cambridge Press. [book website](https://math.mit.edu/linearalgebra/)
[^mml]: Deisenroth, M. P., Faisal, A. A., and Ong, C. S. (2020). *Mathematics for Machine Learning*, Chapters 2–3. Cambridge University Press. [book website](https://mml-book.github.io/)
