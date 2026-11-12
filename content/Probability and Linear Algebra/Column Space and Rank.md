---
title: The Column Space of a Matrix
tags:
  - linear-algebra
  - foundations
date: 2026-11-12
draft: false
prev: "[[Multivariable Calculus]]"
next: "[[Elimination and the Nullspace]]"
---

> **Prerequisites:** vectors, dot products and matrix–vector multiplication. [[Multivariable Calculus]] is not needed for this lecture.
>
> **Series:** this is part 2 of 8. [[Multivariable Calculus]] → **Column Space and Rank** → [[Elimination and the Nullspace]] → [[The Four Fundamental Subspaces]] → [[Orthogonality and Least Squares]] → [[Eigenvalues and Eigenvectors]] → [[First-Order Optimization]] → [[Five Factorizations of a Matrix]].
>
> **TL;DR:**
> - $\mathbf{A}\mathbf{x}$ can be read by rows (dot products) or by columns (a **combination of the columns** with weights $x_j$). The column reading is the one that explains everything else.
> - The **column space** $C(\mathbf{A})$ is the set of all such combinations. It is a line, a plane, or more, and $\mathbf{A}\mathbf{x}=\mathbf{b}$ has a solution exactly when $\mathbf{b}\in C(\mathbf{A})$.
> - Going through the columns left to right and keeping each one that is not a combination of the earlier ones gives a **basis** of $C(\mathbf{A})$. Their number is the **rank** $r$.
> - Those independent columns form $\mathbf{C}$, and $\mathbf{A}=\mathbf{C}\mathbf{R}$, where $\mathbf{R}$ records how to rebuild every column and contains the identity. Reading $\mathbf{A}=\mathbf{C}\mathbf{R}$ by rows proves that **column rank = row rank**.
> - A product $\mathbf{A}\mathbf{B}$ can be computed four ways. The most useful one is columns times rows: a sum of rank-one matrices.

## Motivation

A linear model predicts $\hat{\mathbf{y}}=\mathbf{X}\mathbf{w}$. Which prediction vectors can it produce at all? If two features carry the same information, are the weights still meaningful? Why does a regression with more features than data points fit the training data perfectly? All three questions are about one object: the set of vectors a matrix can produce, its **column space**. This lecture builds that idea from the ground up, with one small matrix carried through every section, and ends with the first of the matrix factorizations that run through the series, $\mathbf{A}=\mathbf{C}\mathbf{R}$.

The story follows Gilbert Strang's videos "The Column Space of a Matrix" and "Elimination and Factorization A = CR" from *A 2020 Vision of Linear Algebra*,[^ocw]<sup>,</sup>[^strangcol]<sup>,</sup>[^strangcr] and his textbook.[^strang2023] The examples and figures here are our own.

**Notation.** Matrices and vectors are bold. $\mathbf{A}\in\mathbb{R}^{m\times n}$ has $m$ rows and $n$ columns, its columns are $\mathbf{a}_1,\dots,\mathbf{a}_n\in\mathbb{R}^m$, and its rank is $r$. Vectors are columns. In the data-matrix example, $\mathbf{X}\in\mathbb{R}^{N\times D}$ holds $N$ data points as rows and $D$ features as columns.

## Two ways to see Ax

### The column picture first

Take the $2\times2$ system

$$
\begin{bmatrix}2&1\\1&3\end{bmatrix}\begin{bmatrix}x_1\\x_2\end{bmatrix}=\begin{bmatrix}5\\5\end{bmatrix}.
$$

**By columns**, the left side is $x_1$ copies of the first column plus $x_2$ copies of the second:

$$
x_1\begin{bmatrix}2\\1\end{bmatrix}+x_2\begin{bmatrix}1\\3\end{bmatrix}=\begin{bmatrix}5\\5\end{bmatrix}.
$$

The question becomes: **which combination of the two column vectors lands on $\mathbf{b}$?** Trying $x_1=2$, $x_2=1$ gives $(4,2)+(1,3)=(5,5)$. Panel (b) of Figure 1 shows the answer as a walk: two steps along $\mathbf{a}_1$, then one step along $\mathbf{a}_2$.

**By rows**, each entry of $\mathbf{A}\mathbf{x}$ is the dot product of a row with $\mathbf{x}$: $2x_1+x_2=5$ and $x_1+3x_2=5$. Each equation is a line in the $(x_1,x_2)$ plane, and the solution is where the lines cross, $(2,1)$ again (panel a).

![[Probability and Linear Algebra/images/column-space-and-rank-row-column.svg]]
*Figure 1. Two readings of the same system. (a) Row picture: each equation is a line in the plane of unknowns, and the solution is their crossing point. (b) Column picture: the columns $\mathbf{a}_1=(2,1)$ and $\mathbf{a}_2=(1,3)$ live in the plane of outputs, and $\mathbf{b}=(5,5)$ is reached by $2\mathbf{a}_1+1\mathbf{a}_2$.*

Both pictures give the same numbers, but they scale differently. With $m$ equations in $n$ unknowns the row picture asks where $m$ hyperplanes in $\mathbb{R}^n$ meet, which is hard to see beyond three dimensions. The column picture asks whether a vector in $\mathbb{R}^m$ is a combination of $n$ given vectors. That question has structure (subspaces, bases, dimension), and the rest of the series is built on it.

> [!tip] Intuition: a matrix is a recipe for combining its columns
> $\mathbf{A}\mathbf{x}=x_1\mathbf{a}_1+x_2\mathbf{a}_2+\dots+x_n\mathbf{a}_n$. The vector $\mathbf{x}$ is a list of amounts, and the columns are the ingredients. Everything $\mathbf{A}$ can ever produce is some mixture of its columns. In particular $\mathbf{A}\mathbf{e}_j=\mathbf{a}_j$: the $j$-th standard basis vector picks out column $j$, which is why "the columns of a matrix are where the basis vectors go".
>
> 3Blue1Brown's animation of this idea is worth watching before going on.[^3b1b-lt]

### The running example

The matrix used for the rest of the lecture (and the next) is

$$
\mathbf{A}=\begin{bmatrix}1&2&1&2\\2&4&3&5\\1&2&2&3\end{bmatrix}\in\mathbb{R}^{3\times4}.
$$

With $\mathbf{x}=(1,1,1,1)$, the row way computes three dot products, $1+2+1+2=6$, $2+4+3+5=14$ and $1+2+2+3=8$. The column way adds the four columns, $(1,2,1)+(2,4,2)+(1,3,2)+(2,5,3)=(6,14,8)$. Same answer, $\mathbf{A}\mathbf{x}=(6,14,8)$.

## The column space

### All combinations of the columns

> [!note] Definition: column space
> The **column space** of $\mathbf{A}\in\mathbb{R}^{m\times n}$ is the set of all combinations of its columns,
>
> $$
> C(\mathbf{A})=\{\mathbf{A}\mathbf{x}:\mathbf{x}\in\mathbb{R}^n\}=\{x_1\mathbf{a}_1+\dots+x_n\mathbf{a}_n\}\subseteq\mathbb{R}^m.
> $$
>
> It is also called the **range** or **image** of $\mathbf{A}$, and the **span** of the columns.

$C(\mathbf{A})$ is a **subspace** of $\mathbb{R}^m$: it contains $\mathbf{0}$ (take $\mathbf{x}=\mathbf{0}$), and it is closed under addition and scaling, because $\mathbf{A}\mathbf{x}+\mathbf{A}\mathbf{y}=\mathbf{A}(\mathbf{x}+\mathbf{y})$ and $c\,\mathbf{A}\mathbf{x}=\mathbf{A}(c\mathbf{x})$. Subspaces of $\mathbb{R}^3$ come in only four shapes: the origin alone, a line through the origin, a plane through the origin, or all of $\mathbb{R}^3$. Which one we get depends not on how many columns there are but on how many genuinely different directions they contain.

In the running example, column 2 is twice column 1, so it adds no new direction. Column 3, $(1,3,2)$, is not a multiple of column 1, so columns 1 and 3 together span a plane. Column 4 is column 1 plus column 3, so it stays in that plane. Four columns, one plane (Figure 2b).

To find the plane's equation, look for a vector perpendicular to both $\mathbf{a}_1=(1,2,1)$ and $\mathbf{a}_3=(1,3,2)$. The cross product gives

$$
\mathbf{a}_1\times\mathbf{a}_3=(2\cdot2-1\cdot3,\ 1\cdot1-1\cdot2,\ 1\cdot3-2\cdot1)=(1,-1,1),
$$

and indeed $(1,-1,1)$ has zero dot product with all four columns ($1-2+1=0$, $2-4+2=0$, $1-3+2=0$, $2-5+3=0$). So

$$
C(\mathbf{A})=\{\mathbf{b}\in\mathbb{R}^3:\ b_1-b_2+b_3=0\}.
$$

![[Probability and Linear Algebra/images/column-space-and-rank-3d.svg]]
*Figure 2. Column spaces in $\mathbb{R}^3$. (a) The columns $(1,2,1)$ and $(2,4,2)$ point the same way, so all their combinations fill only a line. (b) The four columns of the running example fill the plane $b_1-b_2+b_3=0$ (shaded). Only $\mathbf{a}_1$ and $\mathbf{a}_3$ are drawn; $\mathbf{a}_2$ and $\mathbf{a}_4$ lie in the same plane. The vector $\mathbf{b}=(3,7,4)=2\mathbf{a}_1+\mathbf{a}_3$ lies in the plane, while $(1,0,0)$ sits off it.*

### Solvability: Ax = b has a solution exactly when b is in C(A)

This is the definition read backwards, but it is the most used fact of the lecture:

> [!note] Solvability
> $\mathbf{A}\mathbf{x}=\mathbf{b}$ has at least one solution if and only if $\mathbf{b}\in C(\mathbf{A})$.

For the running example, $\mathbf{b}=(3,7,4)$ passes the test $3-7+4=0$, and $\mathbf{x}=(2,0,1,0)$ solves it. The vector $\mathbf{b}=(1,0,0)$ fails ($1-0+0\ne0$), so no $\mathbf{x}$ whatsoever gives $\mathbf{A}\mathbf{x}=(1,0,0)$: no combination of the columns can leave the plane. In [[Elimination and the Nullspace]] elimination produces this test automatically, as a row that reads $0=b_1-b_2+b_3$. The perpendicular vector $(1,-1,1)$ that defines the test is no accident either; it spans a subspace of its own, the left nullspace, in [[The Four Fundamental Subspaces]].

How large can $C(\mathbf{A})$ be? It has at most $n$ independent directions (one per column) and lives in $\mathbb{R}^m$, so its dimension is at most $\min(m,n)$. In particular a **wide** matrix ($n>m$) can have $C(\mathbf{A})=\mathbb{R}^m$, so that every $\mathbf{b}$ is reachable, while a **tall** matrix ($m>n$) never can: $n$ columns cannot fill $\mathbb{R}^m$, and most right-hand sides have no exact solution.

> [!tip] Intuition: watch the space get squished
> A $3\times3$ matrix whose columns lie in a plane flattens all of space onto that plane, and nothing it does can produce a point off the plane. 3Blue1Brown's video on inverses, column space and null space animates this squishing.[^3b1b-col]

### The data matrix and its column space

In regression the matrix is the data. The design matrix $\mathbf{X}\in\mathbb{R}^{N\times D}$ has one row per data point and one column per feature, and a linear model predicts

$$
\hat{\mathbf{y}}=\mathbf{X}\mathbf{w}=w_1\mathbf{x}_{:,1}+\dots+w_D\mathbf{x}_{:,D}.
$$

Read by columns, **every possible vector of predictions is a combination of the feature columns**. The column space $C(\mathbf{X})\subseteq\mathbb{R}^N$ is the set of all prediction vectors the model can produce, whatever the weights. Three consequences follow at once.

- **An exact fit exists only if $\mathbf{y}\in C(\mathbf{X})$.** With many data points and few features ($N>D$), $C(\mathbf{X})$ is at most a $D$-dimensional slice of $\mathbb{R}^N$, and a noisy target vector essentially never lies in it. The best we can do is the point of $C(\mathbf{X})$ closest to $\mathbf{y}$; that is least squares, done properly in [[Orthogonality and Least Squares#Least squares]] (with the probabilistic view in [[Linear Models]]).
- **More features than data points ($D\ge N$) usually means $C(\mathbf{X})=\mathbb{R}^N$.** Then every target vector, noise included, is fitted exactly: the model memorizes the training set. The next lecture counts how many weight vectors do this.
- **Redundant features add nothing to $C(\mathbf{X})$.** Record temperature in Celsius ($\mathbf{c}$) and in Fahrenheit ($\mathbf{f}=1.8\,\mathbf{c}+32\cdot\mathbf{1}$), together with the intercept column $\mathbf{1}$. Then $\mathbf{f}$ is a combination of $\mathbf{1}$ and $\mathbf{c}$, so $\mathbf{X}=[\mathbf{1}\ \mathbf{c}\ \mathbf{f}]$ has three columns but only two independent ones, and $C(\mathbf{X})$ is a plane. The predictions are no richer than with $[\mathbf{1}\ \mathbf{c}]$, and the weights are not unique: adding any multiple of $(32,1.8,-1)$ to $\mathbf{w}$ leaves $\mathbf{X}\mathbf{w}$ unchanged. This is **perfect multicollinearity**, and it is a statement about the rank.

## Independence, basis and rank

### Independent columns

The running example has four columns but only two directions. The words for this are:

- Vectors $\mathbf{v}_1,\dots,\mathbf{v}_k$ are **linearly independent** if no one of them is a combination of the others. Equivalently, the only combination that gives zero is the trivial one: $c_1\mathbf{v}_1+\dots+c_k\mathbf{v}_k=\mathbf{0}$ forces all $c_i=0$.
- A **basis** of a subspace is a set of independent vectors that spans it. Every vector in the subspace is then a combination of the basis vectors in **exactly one** way.
- All bases of a subspace have the same number of vectors, its **dimension**.
- The **rank** of $\mathbf{A}$ is the dimension of its column space: $r=\dim C(\mathbf{A})$, the number of independent columns.

In the running example, $2\mathbf{a}_1-\mathbf{a}_2=\mathbf{0}$ is a non-trivial combination giving zero, so the four columns are dependent. The pair $\mathbf{a}_1,\mathbf{a}_3$ is independent and spans $C(\mathbf{A})$, so it is a basis, and $r=2$.

> [!tip] Intuition: independence means no wasted columns
> Each independent vector adds a new direction and raises the dimension by one. A dependent vector is already reachable from the others, so it adds nothing to what can be built. The 3Blue1Brown chapter on span and basis vectors shows this geometrically.[^3b1b-span]

### Greedy column selection

There is a simple way to find a basis among the columns: walk through them from left to right and keep a column only if it is **not** a combination of the columns already kept.

> [!example] Algorithm: greedy basis of the column space
> 1. Start with an empty list of kept columns.
> 2. For $j=1,\dots,n$: if $\mathbf{a}_j$ is not a combination of the kept columns, keep it.
> 3. The kept columns are a basis of $C(\mathbf{A})$, and their number is the rank $r$.

On the running example:

1. $\mathbf{a}_1=(1,2,1)$ is not zero, so keep it.
2. $\mathbf{a}_2=(2,4,2)=2\mathbf{a}_1$. Discard it, and remember the recipe "2 of column 1".
3. $\mathbf{a}_3=(1,3,2)$ is not a multiple of $\mathbf{a}_1$ (the first entries would force the multiple 1, but then the second entries disagree). Keep it.
4. $\mathbf{a}_4=(2,5,3)$. Try $\alpha\mathbf{a}_1+\beta\mathbf{a}_3$: the first entry needs $\alpha+\beta=2$ and the second $2\alpha+3\beta=5$, so $\beta=1$ and $\alpha=1$. The third entry checks: $1+2=3$. Discard it, with the recipe "1 of column 1 plus 1 of column 3".

The kept columns are $\mathbf{a}_1$ and $\mathbf{a}_3$, so $r=2$. Why is this a basis? The kept columns are independent, because each was not a combination of the earlier kept ones. And they span $C(\mathbf{A})$, because every discarded column is a combination of kept ones, so anything built from all the columns can be rebuilt from the kept ones alone. In the next lecture elimination performs exactly this scan, with the "is it a combination?" test done by arithmetic instead of by eye.

## A = CR

### Recording the recipes

Put the kept columns into a matrix $\mathbf{C}$ and the recipes into a matrix $\mathbf{R}$, one column of $\mathbf{R}$ for each column of $\mathbf{A}$:

$$
\underbrace{\begin{bmatrix}1&2&1&2\\2&4&3&5\\1&2&2&3\end{bmatrix}}_{\mathbf{A}\ (3\times4)}
=\underbrace{\begin{bmatrix}1&1\\2&3\\1&2\end{bmatrix}}_{\mathbf{C}\ (3\times2)}
\underbrace{\begin{bmatrix}1&2&0&1\\0&0&1&1\end{bmatrix}}_{\mathbf{R}\ (2\times4)}.
$$

Column $j$ of $\mathbf{C}\mathbf{R}$ is $\mathbf{C}$ times column $j$ of $\mathbf{R}$, a combination of the columns of $\mathbf{C}$ with the weights stored in $\mathbf{R}$. Column 2 of $\mathbf{R}$ is $(2,0)$ ("2 of column 1"), and column 4 is $(1,1)$. A kept column rebuilds itself, so its recipe is a column of the identity: columns 1 and 3 of $\mathbf{R}$ are $(1,0)$ and $(0,1)$. **$\mathbf{R}$ contains the identity matrix in the positions of the independent columns** (Figure 3).

![[Probability and Linear Algebra/images/column-space-and-rank-cr.svg]]
*Figure 3. $\mathbf{A}=\mathbf{C}\mathbf{R}$ for the running example. The independent columns of $\mathbf{A}$ (blue) are copied into $\mathbf{C}$. Each column of $\mathbf{R}$ says how to rebuild the matching column of $\mathbf{A}$ from $\mathbf{C}$; at the independent columns this recipe is a column of the identity (green), and at the dependent columns (orange) it holds the coefficients.*

> [!note] The column–row factorization
> Every $\mathbf{A}\in\mathbb{R}^{m\times n}$ of rank $r\ge1$ factors as $\mathbf{A}=\mathbf{C}\mathbf{R}$, where $\mathbf{C}\in\mathbb{R}^{m\times r}$ holds the first $r$ independent columns of $\mathbf{A}$ (found by the greedy scan) and $\mathbf{R}\in\mathbb{R}^{r\times n}$ contains the $r\times r$ identity in those columns. The columns of $\mathbf{C}$ are a basis of $C(\mathbf{A})$, and the rows of $\mathbf{R}$ are a basis of the row space of $\mathbf{A}$.

The matrix $\mathbf{R}$ is exactly the **reduced row echelon form** of $\mathbf{A}$ with its zero rows removed. [[Elimination and the Nullspace#Reduced row echelon form]] computes it by elimination, and Strang and Moler's 2022 paper is a short account of both factorizations.[^strangmoler2022]

> [!example]- Code: the greedy scan and A = CR in numpy
> ```python
> import numpy as np
>
> def cr(A, tol=1e-10):
>     """A = C R. Keep column j if it is not a combination of the columns kept so far."""
>     keep = []
>     for j in range(A.shape[1]):
>         if np.linalg.matrix_rank(A[:, keep + [j]], tol) > len(keep):
>             keep.append(j)
>     C = A[:, keep]
>     R = np.linalg.lstsq(C, A, rcond=None)[0]   # exact: every column of A lies in C(C)
>     return C, R, keep
>
> A = np.array([[1, 2, 1, 2], [2, 4, 3, 5], [1, 2, 2, 3]], dtype=float)
> C, R, keep = cr(A)
> print(keep)                    # [0, 2]: columns 1 and 3 are independent
> print(R.round(12) + 0.0)       # [[1, 2, 0, 1], [0, 0, 1, 1]]
> print(np.allclose(C @ R, A))   # True
> ```
> This is for understanding, not speed: it calls a rank computation once per column. Elimination (next lecture) finds $\mathbf{R}$ in a single pass.

### Column rank equals row rank

The **row space** of $\mathbf{A}$ is the set of all combinations of its rows, a subspace of $\mathbb{R}^n$, written $C(\mathbf{A}^\top)$ because the rows of $\mathbf{A}$ are the columns of $\mathbf{A}^\top$. Its dimension is the **row rank**. In the running example the rows live in $\mathbb{R}^4$, and it is not obvious how many of them are independent. $\mathbf{A}=\mathbf{C}\mathbf{R}$ answers this for every matrix at once.

> [!note] Theorem: column rank = row rank
> For every matrix, the number of independent columns equals the number of independent rows.

The idea is to read $\mathbf{A}=\mathbf{C}\mathbf{R}$ **by rows**. Row $i$ of a product is row $i$ of the left factor times the right factor, so row $i$ of $\mathbf{A}$ is a combination of the rows of $\mathbf{R}$, with weights taken from row $i$ of $\mathbf{C}$. In the running example, writing $\mathbf{r}_1^{*}=(1,2,0,1)$ and $\mathbf{r}_2^{*}=(0,0,1,1)$ for the rows of $\mathbf{R}$:

$$
\begin{aligned}
\text{row 1 of }\mathbf{A}&=(1,2,1,2)=1\,\mathbf{r}_1^{*}+1\,\mathbf{r}_2^{*},\\
\text{row 2 of }\mathbf{A}&=(2,4,3,5)=2\,\mathbf{r}_1^{*}+3\,\mathbf{r}_2^{*},\\
\text{row 3 of }\mathbf{A}&=(1,2,2,3)=1\,\mathbf{r}_1^{*}+2\,\mathbf{r}_2^{*}.
\end{aligned}
$$

The weights $(1,1)$, $(2,3)$ and $(1,2)$ are exactly the rows of $\mathbf{C}$. All three rows of $\mathbf{A}$ are built from two rows, so the row rank is at most 2.

> [!proof]- Proof of the theorem
> Let $\mathbf{A}$ have column rank $r$ and write $\mathbf{A}=\mathbf{C}\mathbf{R}$ with $\mathbf{R}\in\mathbb{R}^{r\times n}$.
>
> **Row rank $\le r$.** Row $i$ of $\mathbf{A}$ is $\sum_{k=1}^{r}c_{ik}\,\mathbf{r}_k^{*}$, a combination of the $r$ rows of $\mathbf{R}$. So the row space of $\mathbf{A}$ lies inside the span of $r$ vectors and has dimension at most $r$.
>
> **The rows of $\mathbf{R}$ are independent,** because $\mathbf{R}$ contains the $r\times r$ identity in its pivot columns: a combination $\sum_kd_k\mathbf{r}_k^{*}$ has the entry $d_k$ in the $k$-th pivot column, so it is zero only if every $d_k=0$. And the rows of $\mathbf{R}$ lie in the row space of $\mathbf{A}$, since $\mathbf{R}$ can be recovered from rows of $\mathbf{A}$ (elimination in the next lecture does exactly this). So the row space has dimension exactly $r$.
>
> For a proof that avoids elimination, apply the first step to $\mathbf{A}^\top$ as well: it gives column rank of $\mathbf{A}$ = row rank of $\mathbf{A}^\top\le$ column rank of $\mathbf{A}^\top$ = row rank of $\mathbf{A}$. Together with row rank $\le$ column rank, the two are equal.

This is a genuinely surprising fact. The columns of a $3\times4$ matrix live in $\mathbb{R}^3$ and its rows in $\mathbb{R}^4$, two different spaces, yet the number of independent directions is the same. From now on we simply speak of **the rank** $r$.

### Rank-one matrices

The simplest non-zero matrices have rank 1. All their columns are multiples of one column $\mathbf{u}$, so $\mathbf{C}=\mathbf{u}$ is a single column, $\mathbf{R}=\mathbf{v}^\top$ is a single row, and

$$
\mathbf{A}=\mathbf{u}\mathbf{v}^\top,
\qquad\text{for example}\qquad
\begin{bmatrix}1\\2\\3\end{bmatrix}\begin{bmatrix}2&0&1\end{bmatrix}=\begin{bmatrix}2&0&1\\4&0&2\\6&0&3\end{bmatrix}.
$$

Every column is a multiple of $(1,2,3)$, and every row is a multiple of $(2,0,1)$. A rank-one matrix is described by $m+n$ numbers instead of $mn$. The next section shows that every matrix of rank $r$ is a sum of $r$ of them.

## Matrix multiplication four ways

Take $\mathbf{A}\in\mathbb{R}^{m\times n}$ and $\mathbf{B}\in\mathbb{R}^{n\times p}$. The product $\mathbf{A}\mathbf{B}\in\mathbb{R}^{m\times p}$ can be organized in four ways, all giving the same matrix with the same $mnp$ multiplications. We use the small example

$$
\mathbf{A}=\begin{bmatrix}1&2\\0&1\end{bmatrix},\qquad\mathbf{B}=\begin{bmatrix}3&1\\1&2\end{bmatrix},\qquad\mathbf{A}\mathbf{B}=\begin{bmatrix}5&5\\1&2\end{bmatrix}.
$$

1. **Rows times columns (dot products).** Entry $(i,j)$ is row $i$ of $\mathbf{A}$ dotted with column $j$ of $\mathbf{B}$. For example $(1,2)\cdot(3,1)=5$ in the top-left corner. This is the definition usually taught first.
2. **$\mathbf{A}$ times columns.** Column $j$ of $\mathbf{A}\mathbf{B}$ is $\mathbf{A}$ times column $j$ of $\mathbf{B}$: $\mathbf{A}(3,1)=3(1,0)+1(2,1)=(5,1)$. So **every column of $\mathbf{A}\mathbf{B}$ is a combination of the columns of $\mathbf{A}$**, and $C(\mathbf{A}\mathbf{B})\subseteq C(\mathbf{A})$. This is how we read $\mathbf{C}\mathbf{R}$ above.
3. **Rows times $\mathbf{B}$.** Row $i$ of $\mathbf{A}\mathbf{B}$ is row $i$ of $\mathbf{A}$ times $\mathbf{B}$: $(0,1)\mathbf{B}=0(3,1)+1(1,2)=(1,2)$. So every row of $\mathbf{A}\mathbf{B}$ is a combination of the rows of $\mathbf{B}$. This is how we proved column rank = row rank.
4. **Columns times rows (outer products).** $\mathbf{A}\mathbf{B}$ is the sum over $k$ of column $k$ of $\mathbf{A}$ times row $k$ of $\mathbf{B}$, and each term is a rank-one matrix.

Written out for the example, with $\mathbf{b}_k^{*}$ for row $k$ of $\mathbf{B}$, the fourth way reads

$$
\mathbf{A}\mathbf{B}=\sum_{k=1}^{n}\mathbf{a}_k\mathbf{b}_k^{*}
=\begin{bmatrix}1\\0\end{bmatrix}\begin{bmatrix}3&1\end{bmatrix}+\begin{bmatrix}2\\1\end{bmatrix}\begin{bmatrix}1&2\end{bmatrix}
=\begin{bmatrix}3&1\\0&0\end{bmatrix}+\begin{bmatrix}2&4\\1&2\end{bmatrix}.
$$

The two pieces add up to $\begin{bmatrix}5&5\\1&2\end{bmatrix}$, as they should.

The fourth way is the one Strang puts at the centre of his recent teaching,[^strang2018] because factorizations are best read through it. Applied to $\mathbf{A}=\mathbf{C}\mathbf{R}$, it writes any rank-$r$ matrix as a sum of $r$ rank-one pieces, one for each independent column:

$$
\mathbf{A}=\mathbf{c}_1\mathbf{r}_1^{*}+\mathbf{c}_2\mathbf{r}_2^{*}+\dots+\mathbf{c}_r\mathbf{r}_r^{*}.
$$

![[Probability and Linear Algebra/images/column-space-and-rank-rank-one.svg]]
*Figure 4. The running example as columns times rows. The first piece is column 1 of $\mathbf{C}$ times row 1 of $\mathbf{R}$: it reproduces columns 1, 2 and the column-1 part of column 4. The second piece adds column 3 and the rest of column 4. Two rank-one matrices make a rank-two matrix.*

Every factorization in the series can be read this way. The SVD writes $\mathbf{A}$ as a sum of rank-one pieces ordered by importance, and keeping the first few gives the best low-rank approximation ([[Matrix Factorization and Dimensionality Reduction#Singular value decomposition (SVD)|the SVD]]); the spectral theorem does the same for symmetric matrices ([[Eigenvalues and Eigenvectors]]). [[Five Factorizations of a Matrix#Matrix multiplication as columns times rows]] collects them all.

> [!tip] Intuition: matrix multiplication is composition
> Ways 2 and 3 say that $\mathbf{A}\mathbf{B}$ first applies $\mathbf{B}$ and then $\mathbf{A}$: a vector $\mathbf{x}$ goes to $\mathbf{B}\mathbf{x}$ and then to $\mathbf{A}(\mathbf{B}\mathbf{x})$. That is why the order matters and why $C(\mathbf{A}\mathbf{B})\subseteq C(\mathbf{A})$: the final step is always $\mathbf{A}$. See 3Blue1Brown's chapter on matrix multiplication as composition.[^3b1b-comp]

## Pros and cons of CR

$\mathbf{A}=\mathbf{C}\mathbf{R}$ is the first factorization of the series, and a useful one to judge the others by.

**What it does well.**
- **It keeps actual columns of the data.** If $\mathbf{A}$ is a gene-expression matrix, $\mathbf{C}$ consists of real genes, not abstract mixtures of thousands of them. This interpretability is the whole point of the CUR decomposition in [[Matrix Factorization and Dimensionality Reduction#CUR decomposition]], which picks a few actual columns and rows to approximate a large matrix.
- **It exposes the rank and two bases.** The columns of $\mathbf{C}$ are a basis of the column space, the rows of $\mathbf{R}$ are a basis of the row space, and the dependent columns are explicit combinations of the independent ones.
- **It is cheap and exact for small integer matrices**, which makes it a good hand tool.

**What it does badly.**
- **It can be numerically fragile.** Independence is a yes-or-no notion, but in floating point it is a matter of degree. Columns that are *nearly* parallel are independent in theory, yet the coefficients that combine them are huge and sensitive. Take the columns $(1,0)$ and $(1,0.1)$. The vector $\mathbf{b}=(2,0.1)$ is $1\cdot(1,0)+1\cdot(1,0.1)$, but moving it by only $0.1$ to $(2,0.2)$ changes the coefficients to $(0,2)$ (Figure 5). A 5% change in $\mathbf{b}$ caused a 100% change in $\mathbf{x}$: an amplification of 20, which here equals the **condition number** $\kappa\approx20$ of the matrix. The greedy scan also has to decide with a tolerance whether a column is "a combination of the others", and the answer can flip under rounding.
- **For an invertible matrix it says nothing.** If $\mathbf{A}$ is $n\times n$ with rank $n$, every column is independent, so $\mathbf{C}=\mathbf{A}$ and $\mathbf{R}=\mathbf{I}$. The factorization is $\mathbf{A}=\mathbf{A}\mathbf{I}$, which is no progress. For square invertible systems we need a factorization into **simpler** pieces, triangular ones: $\mathbf{A}=\mathbf{L}\mathbf{U}$ in [[Elimination and the Nullspace#Elimination and A = LU]].

![[Probability and Linear Algebra/images/column-space-and-rank-conditioning.svg]]
*Figure 5. The coordinates of $\mathbf{b}$ in a basis of columns, before and after a small change $\delta\mathbf{b}=(0,0.1)$. (a) With perpendicular columns the coordinates change by $0.1$. (b) With nearly parallel columns the same change moves the coordinates from $(1,1)$ to $(0,2)$; the dashed paths show $x_1\mathbf{a}_1$ followed by $x_2\mathbf{a}_2$. The vertical axis in (b) is stretched so that the two columns can be told apart.*

The cure for fragility is a better basis for the same column space: **orthonormal** columns, which are as far from parallel as possible. Gram–Schmidt turns the columns of $\mathbf{A}$ into such a basis and gives $\mathbf{A}=\mathbf{Q}\mathbf{R}$ ([[Orthogonality and Least Squares#Gram–Schmidt and A = QR]]; a different $\mathbf{R}$, upper triangular this time). The SVD goes further and finds orthonormal bases for the column space *and* the row space at once ([[Matrix Factorization and Dimensionality Reduction#Singular value decomposition (SVD)]]). Both trade the interpretability of real columns for numerical stability.

## Key takeaways

- $\mathbf{A}\mathbf{x}$ is a combination of the columns of $\mathbf{A}$ with weights $x_j$. The row picture (intersecting hyperplanes) gives the same numbers but does not generalize as well.
- The column space $C(\mathbf{A})$ is every vector $\mathbf{A}$ can produce. $\mathbf{A}\mathbf{x}=\mathbf{b}$ is solvable exactly when $\mathbf{b}\in C(\mathbf{A})$. For a data matrix, $C(\mathbf{X})$ is the set of all achievable prediction vectors, and redundant features do not enlarge it.
- Scanning the columns left to right and keeping the ones that add a new direction gives a basis; their number is the rank $r\le\min(m,n)$.
- $\mathbf{A}=\mathbf{C}\mathbf{R}$: $\mathbf{C}$ holds the independent columns, $\mathbf{R}$ holds the recipes and contains $\mathbf{I}$. Reading it by rows proves column rank = row rank.
- Matrix multiplication can be done by dot products, by columns, by rows, or as a sum of columns times rows. The last way writes a rank-$r$ matrix as a sum of $r$ rank-one matrices.
- CR is interpretable but can be ill-conditioned, and it does nothing for invertible matrices. LU, QR and the SVD address these weaknesses.

## Where this shows up next

- [[Elimination and the Nullspace]] computes $\mathbf{R}$ by elimination, factors square matrices as $\mathbf{A}=\mathbf{L}\mathbf{U}$, and finds all solutions of $\mathbf{A}\mathbf{x}=\mathbf{0}$ and $\mathbf{A}\mathbf{x}=\mathbf{b}$.
- [[The Four Fundamental Subspaces]] adds the row space and the two nullspaces to the column space and shows how the four fit together.
- [[Orthogonality and Least Squares]] handles the case $\mathbf{b}\notin C(\mathbf{A})$ by projecting onto the column space, which is linear regression.
- [[Matrix Factorization and Dimensionality Reduction]] uses sums of rank-one matrices for low-rank approximation, PCA and CUR.
- [[Five Factorizations of a Matrix]] puts $\mathbf{C}\mathbf{R}$ next to $\mathbf{L}\mathbf{U}$, $\mathbf{Q}\mathbf{R}$, the eigenvalue factorizations and the SVD.

<!-- References: footnote definitions below render as the numbered "References" list at the bottom of the page -->

[^ocw]: Strang, G. (2020). *A 2020 Vision of Linear Algebra* (RES.18-010). MIT OpenCourseWare. [course page](https://ocw.mit.edu/courses/res-18-010-a-2020-vision-of-linear-algebra-spring-2020/)
[^strangcol]: Strang, G. (2020). *Part 1: The Column Space of a Matrix* [video]. A 2020 Vision of Linear Algebra, MIT OpenCourseWare. [YouTube](https://www.youtube.com/watch?v=azzrfdysfI0)
[^strangcr]: Strang, G. (2025). *Elimination and Factorization A = CR* [video]. MIT OpenCourseWare. [YouTube](https://www.youtube.com/watch?v=PrErxYzSANo)
[^strang2023]: Strang, G. (2023). *Introduction to Linear Algebra* (6th ed.), Chapters 1–3. Wellesley-Cambridge Press. [book website](https://math.mit.edu/~gs/linearalgebra/)
[^3b1b-lt]: 3Blue1Brown (2016). *Linear transformations and matrices*, Chapter 3 of Essence of Linear Algebra [video]. [YouTube](https://www.youtube.com/watch?v=kYB8IZa5AuE)
[^3b1b-col]: 3Blue1Brown (2016). *Inverse matrices, column space and null space*, Chapter 7 of Essence of Linear Algebra [video]. [YouTube](https://www.youtube.com/watch?v=uQhTuRlWMxw)
[^3b1b-span]: 3Blue1Brown (2016). *Linear combinations, span, and basis vectors*, Chapter 2 of Essence of Linear Algebra [video]. [YouTube](https://www.youtube.com/watch?v=k7RM-ot2NWY)
[^strangmoler2022]: Strang, G., & Moler, C. (2022). *LU and CR Elimination*. SIAM Review, 64(1), 181–190. [doi:10.1137/20M1358694](https://doi.org/10.1137/20M1358694)
[^strang2018]: Strang, G. (2018). *Multiplying and Factoring Matrices*. The American Mathematical Monthly, 125(3), 223–230. [doi:10.1080/00029890.2018.1408378](https://doi.org/10.1080/00029890.2018.1408378)
[^3b1b-comp]: 3Blue1Brown (2016). *Matrix multiplication as composition*, Chapter 4 of Essence of Linear Algebra [video]. [YouTube](https://www.youtube.com/watch?v=XkY2DOUCWMU)
