---
title: "Orthogonal Vectors, Projections and Least Squares"
tags:
  - linear-algebra
  - foundations
date: 2026-11-12
draft: false
prev: "[[The Four Fundamental Subspaces]]"
next: "[[Eigenvalues and Eigenvectors]]"
---

> **Prerequisites:** [[The Four Fundamental Subspaces]] (orthogonal complements, the split $\mathbf{x}=\mathbf{x}_r+\mathbf{x}_n$), [[Column Space and Rank]] (solvability means $\mathbf{b}\in C(\mathbf{A})$) and, for the calculus derivation, [[Multivariable Calculus#Matrix calculus|matrix calculus]].
>
> **Series:** this is part 5 of 8. [[Multivariable Calculus]] → [[Column Space and Rank]] → [[Elimination and the Nullspace]] → [[The Four Fundamental Subspaces]] → **Orthogonality and Least Squares** → [[Eigenvalues and Eigenvectors]] → [[First-Order Optimization]] → [[Five Factorizations of a Matrix]].
>
> **TL;DR:**
> - A matrix $\mathbf{Q}$ with orthonormal columns satisfies $\mathbf{Q}^\top\mathbf{Q}=\mathbf{I}$. If it is square it preserves lengths and angles (rotations, reflections), its inverse is its transpose, and its eigenvalues have $\lvert\lambda\rvert=1$. If it is tall, $\mathbf{Q}\mathbf{Q}^\top$ is a projection.
> - The projection of $\mathbf{b}$ onto $C(\mathbf{A})$ is $\mathbf{p}=\mathbf{P}\mathbf{b}$ with $\mathbf{P}=\mathbf{A}(\mathbf{A}^\top\mathbf{A})^{-1}\mathbf{A}^\top$. Projection matrices are exactly the symmetric matrices with $\mathbf{P}^2=\mathbf{P}$.
> - When $\mathbf{A}\mathbf{x}=\mathbf{b}$ has **no** solution, least squares replaces $\mathbf{b}$ by its projection. The error is perpendicular to the column space, which gives the normal equations $\mathbf{A}^\top\mathbf{A}\hat{\mathbf{x}}=\mathbf{A}^\top\mathbf{b}$, by geometry or by setting a gradient to zero. $\mathbf{A}^\top\mathbf{A}$ is invertible iff the columns of $\mathbf{A}$ are independent.
> - Gram–Schmidt turns independent columns into orthonormal ones, $\mathbf{A}=\mathbf{Q}\mathbf{R}$. Least squares then becomes $\mathbf{R}\hat{\mathbf{x}}=\mathbf{Q}^\top\mathbf{b}$, which avoids squaring the condition number and is far more accurate than the normal equations.
> - With dependent columns there are infinitely many least-squares solutions; the shortest one lies in the row space. That is the pseudo-inverse solution.

## Motivation

Real data rarely satisfies equations exactly. Fit a line through 100 noisy points and you have 100 equations in 2 unknowns, with no line passing through all of them. The system $\mathbf{A}\mathbf{x}=\mathbf{b}$ has no solution, and yet we want the best possible $\mathbf{x}$. The answer, **least squares**, is the most used algorithm in data analysis, and it is pure geometry: drop a perpendicular from $\mathbf{b}$ onto the column space of $\mathbf{A}$.

Perpendiculars are the theme of the lecture. Orthogonal vectors make coordinates trivial to compute, orthogonal matrices never amplify errors, and projections are the closest-point maps that least squares needs. This lecture follows Strang's videos "Orthogonal Vectors" and "The Four Fundamental Subspaces and Least Squares";[^strangorth]<sup>,</sup>[^strang4sub] the same material is in Chapter 4 of his textbook[^strangbook] and Chapter 3 of *Mathematics for Machine Learning*.[^mml] [[Linear Models]] uses the results for regression and adds the probabilistic view (least squares as maximum likelihood under Gaussian noise); this lecture is the full linear-algebra treatment.

## Orthonormal vectors and orthogonal matrices

### Orthonormal columns

Vectors $\mathbf{q}_1,\dots,\mathbf{q}_n$ are **orthonormal** if they are mutually orthogonal and each has length 1:

$$
\mathbf{q}_i^\top\mathbf{q}_j=\begin{cases}1&i=j,\\0&i\neq j.\end{cases}
$$

Put them as the columns of a matrix $\mathbf{Q}\in\mathbb{R}^{m\times n}$. Entry $(i,j)$ of $\mathbf{Q}^\top\mathbf{Q}$ is $\mathbf{q}_i^\top\mathbf{q}_j$, so the whole definition is one matrix equation:

$$
\mathbf{Q}^\top\mathbf{Q}=\mathbf{I}_n.
$$

**Coordinates come for free.** If $\mathbf{b}=c_1\mathbf{q}_1+\dots+c_n\mathbf{q}_n$, take the dot product with $\mathbf{q}_j$: every other term vanishes and $c_j=\mathbf{q}_j^\top\mathbf{b}$. In matrix form $\mathbf{c}=\mathbf{Q}^\top\mathbf{b}$. With a general basis we would have to solve a linear system; with an orthonormal basis we only take dot products.

For example, $\mathbf{q}_1=\frac13(1,2,2)$, $\mathbf{q}_2=\frac13(2,1,-2)$ and $\mathbf{q}_3=\frac13(2,-2,1)$ are orthonormal: each has length $\frac13\sqrt{1+4+4}=1$, and the dot products $2+2-4$, $2-4+2$ and $4-2-2$ are all zero.

### Orthogonal matrices

A **square** matrix with orthonormal columns is called an **orthogonal matrix**. (The name is traditional; "orthonormal matrix" would be more accurate.) For square $\mathbf{Q}$, the equation $\mathbf{Q}^\top\mathbf{Q}=\mathbf{I}$ says that $\mathbf{Q}^\top$ is a left inverse, and for square matrices a left inverse is the inverse. So

$$
\mathbf{Q}^{-1}=\mathbf{Q}^\top\qquad\text{and also}\qquad\mathbf{Q}\mathbf{Q}^\top=\mathbf{I}:
$$

the rows are orthonormal too. Inverting an orthogonal matrix costs nothing.

**Orthogonal matrices preserve lengths and angles.** For any $\mathbf{x}$,

$$
\lVert\mathbf{Q}\mathbf{x}\rVert^2=(\mathbf{Q}\mathbf{x})^\top(\mathbf{Q}\mathbf{x})=\mathbf{x}^\top\mathbf{Q}^\top\mathbf{Q}\mathbf{x}=\mathbf{x}^\top\mathbf{x}=\lVert\mathbf{x}\rVert^2,
$$

and in the same way $(\mathbf{Q}\mathbf{x})^\top(\mathbf{Q}\mathbf{y})=\mathbf{x}^\top\mathbf{y}$. (This half needs only $\mathbf{Q}^\top\mathbf{Q}=\mathbf{I}$, so it holds for tall $\mathbf{Q}$ as well.) An orthogonal matrix moves space rigidly. In the plane there are two kinds (Figure 1):

$$
\underbrace{\begin{bmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{bmatrix}}_{\text{rotation by }\theta,\ \det=+1}\qquad\underbrace{\begin{bmatrix}\cos2\phi&\sin2\phi\\\sin2\phi&-\cos2\phi\end{bmatrix}}_{\text{reflection across the line at angle }\phi,\ \det=-1}
$$

In any dimension, the **Householder reflection** $\mathbf{H}=\mathbf{I}-2\mathbf{n}\mathbf{n}^\top$ (with $\lVert\mathbf{n}\rVert=1$) flips the component along $\mathbf{n}$ and leaves the hyperplane $\mathbf{n}^\perp$ alone. The three vectors above, as columns, form one:

$$
\mathbf{Q}_3=\frac13\begin{bmatrix}1&2&2\\2&1&-2\\2&-2&1\end{bmatrix}=\mathbf{I}-2\mathbf{n}\mathbf{n}^\top,\qquad\mathbf{n}=\tfrac{1}{\sqrt3}(1,-1,-1).
$$

Permutation matrices are orthogonal too, and products of orthogonal matrices are orthogonal, since $(\mathbf{Q}_1\mathbf{Q}_2)^\top(\mathbf{Q}_1\mathbf{Q}_2)=\mathbf{Q}_2^\top\mathbf{I}\mathbf{Q}_2=\mathbf{I}$.

![[Probability and Linear Algebra/images/orthogonality-and-least-squares-orthogonal-maps.svg]]
*Figure 1. The two kinds of orthogonal $2\times2$ matrices. (a) A rotation keeps the shape and its orientation. (b) A reflection keeps the shape but turns it over (the flag now points the other way). Both keep every length and every angle; the determinant ($+1$ or $-1$) records whether orientation is kept.*

**Eigenvalues have $\lvert\lambda\rvert=1$, and $\det\mathbf{Q}=\pm1$.** If $\mathbf{Q}\mathbf{x}=\lambda\mathbf{x}$ with $\mathbf{x}\neq\mathbf{0}$ (allowing complex $\lambda$ and $\mathbf{x}$, with the complex length), then length preservation gives $\lVert\mathbf{x}\rVert=\lVert\mathbf{Q}\mathbf{x}\rVert=\lvert\lambda\rvert\,\lVert\mathbf{x}\rVert$, so $\lvert\lambda\rvert=1$. A rotation by $\theta$ has the complex pair $e^{\pm i\theta}$; a reflection has $+1$ (the mirror) and $-1$ (the normal), and $\mathbf{Q}_3$ has eigenvalues $1,1,-1$. For the determinant, $\det(\mathbf{Q}^\top\mathbf{Q})=(\det\mathbf{Q})^2=\det\mathbf{I}=1$. [[Eigenvalues and Eigenvectors]] develops this further.

> [!tip] Intuition: why numerical analysts love orthogonal matrices
> Multiplying by $\mathbf{Q}$ cannot make any vector longer, so it cannot amplify rounding errors either. Algorithms built from orthogonal transformations (QR, the SVD, Householder reflections) are stable for this reason, while algorithms that multiply by arbitrary matrices can blow small errors up. The same property is behind orthogonal weight initialization in deep networks: signals neither grow nor shrink as they pass through the layers.

### Tall matrices with orthonormal columns

Keep only the first two columns, $\mathbf{Q}_2=[\mathbf{q}_1\ \mathbf{q}_2]\in\mathbb{R}^{3\times2}$. Still $\mathbf{Q}_2^\top\mathbf{Q}_2=\mathbf{I}_2$, but now

$$
\mathbf{Q}_2\mathbf{Q}_2^\top=\mathbf{q}_1\mathbf{q}_1^\top+\mathbf{q}_2\mathbf{q}_2^\top=\frac19\begin{bmatrix}5&4&-2\\4&5&2\\-2&2&8\end{bmatrix}=\mathbf{I}-\mathbf{q}_3\mathbf{q}_3^\top\neq\mathbf{I}.
$$

This $3\times3$ matrix has rank 2. It keeps the plane spanned by $\mathbf{q}_1,\mathbf{q}_2$ and kills the normal direction $\mathbf{q}_3$: it is the **projection** onto $C(\mathbf{Q}_2)$, the subject of the next section. A tall $\mathbf{Q}$ preserves lengths going forward ($\lVert\mathbf{Q}\mathbf{y}\rVert=\lVert\mathbf{y}\rVert$), but $\mathbf{Q}^\top$ can shrink them. For $\mathbf{x}=(3,0,3)$, $\mathbf{Q}_2^\top\mathbf{x}=(3,0)$ has length 3, while $\lVert\mathbf{x}\rVert=\sqrt{18}$: the part of $\mathbf{x}$ along $\mathbf{q}_3$ was discarded.

## Projections

To **project** $\mathbf{b}$ onto a subspace is to find the point $\mathbf{p}$ of the subspace closest to $\mathbf{b}$. The defining property is that the error $\mathbf{e}=\mathbf{b}-\mathbf{p}$ is perpendicular to the subspace. Every formula below comes from writing that one condition down.

### Projection onto a line

Let the line be all multiples of a nonzero vector $\mathbf{a}$. The projection is some multiple, $\mathbf{p}=\hat{x}\,\mathbf{a}$. Require the error to be perpendicular to $\mathbf{a}$:

$$
\mathbf{a}^\top(\mathbf{b}-\hat{x}\,\mathbf{a})=0\quad\Longrightarrow\quad\hat{x}=\frac{\mathbf{a}^\top\mathbf{b}}{\mathbf{a}^\top\mathbf{a}},\qquad\mathbf{p}=\frac{\mathbf{a}^\top\mathbf{b}}{\mathbf{a}^\top\mathbf{a}}\,\mathbf{a}=\underbrace{\frac{\mathbf{a}\mathbf{a}^\top}{\mathbf{a}^\top\mathbf{a}}}_{\mathbf{P}}\,\mathbf{b}.
$$

The **projection matrix** $\mathbf{P}=\mathbf{a}\mathbf{a}^\top/\mathbf{a}^\top\mathbf{a}$ is a column times a row, so it has rank 1, and its column space is the line itself. Rescaling $\mathbf{a}$ does not change $\mathbf{P}$, as it should not: the line is the same.

> [!example] Worked example: two projections onto lines
> **In the plane.** Project $\mathbf{b}=(1,3)$ onto the line through $\mathbf{a}=(2,1)$. Here $\mathbf{a}^\top\mathbf{b}=2+3=5$ and $\mathbf{a}^\top\mathbf{a}=5$, so $\hat x=1$, $\mathbf{p}=(2,1)$ and $\mathbf{e}=(-1,2)$. Check: $\mathbf{a}^\top\mathbf{e}=-2+2=0$.
>
> **In space.** Project $\mathbf{b}=(3,0,3)$ onto the line through $\mathbf{a}=(1,2,2)$. Now $\mathbf{a}^\top\mathbf{b}=3+0+6=9$ and $\mathbf{a}^\top\mathbf{a}=9$, so $\mathbf{p}=(1,2,2)$ and $\mathbf{e}=(2,-2,1)$, again perpendicular to $\mathbf{a}$: $2-4+2=0$. The projection matrix is
>
> $$
> \mathbf{P}=\frac19\begin{bmatrix}1&2&2\\2&4&4\\2&4&4\end{bmatrix},
> $$
>
> and every column is a multiple of $\mathbf{a}$.

### Projection onto a subspace

Now project onto the column space of $\mathbf{A}\in\mathbb{R}^{m\times n}$, whose columns $\mathbf{a}_1,\dots,\mathbf{a}_n$ are independent. The projection is some combination $\mathbf{p}=\mathbf{A}\hat{\mathbf{x}}$, and the error must be perpendicular to **every** column:

$$
\mathbf{a}_j^\top(\mathbf{b}-\mathbf{A}\hat{\mathbf{x}})=0\ \text{ for }j=1,\dots,n\quad\Longleftrightarrow\quad\mathbf{A}^\top(\mathbf{b}-\mathbf{A}\hat{\mathbf{x}})=\mathbf{0}.
$$

In the language of the [[The Four Fundamental Subspaces|four subspaces]], the error lies in the left nullspace $N(\mathbf{A}^\top)$, the orthogonal complement of $C(\mathbf{A})$. Rearranging gives the **normal equations** $\mathbf{A}^\top\mathbf{A}\hat{\mathbf{x}}=\mathbf{A}^\top\mathbf{b}$, and when $\mathbf{A}^\top\mathbf{A}$ is invertible (next section),

$$
\hat{\mathbf{x}}=(\mathbf{A}^\top\mathbf{A})^{-1}\mathbf{A}^\top\mathbf{b},\qquad\mathbf{p}=\mathbf{A}\hat{\mathbf{x}}=\mathbf{P}\mathbf{b},\qquad\boxed{\;\mathbf{P}=\mathbf{A}(\mathbf{A}^\top\mathbf{A})^{-1}\mathbf{A}^\top\;}
$$

For a single column this is the line formula again, with the number $\mathbf{a}^\top\mathbf{a}$ now a matrix $\mathbf{A}^\top\mathbf{A}$. It is tempting to "simplify" $\mathbf{P}$ to $\mathbf{A}\mathbf{A}^{-1}(\mathbf{A}^\top)^{-1}\mathbf{A}^\top=\mathbf{I}$, but a rectangular $\mathbf{A}$ has no inverse. Only when $\mathbf{A}$ is square and invertible is $C(\mathbf{A})$ all of $\mathbb{R}^m$, and then $\mathbf{P}=\mathbf{I}$ is right. When the columns are orthonormal, $\mathbf{A}=\mathbf{Q}$ and $\mathbf{Q}^\top\mathbf{Q}=\mathbf{I}$, so the formula collapses to $\mathbf{P}=\mathbf{Q}\mathbf{Q}^\top$, as we saw above.

> [!example] Worked example: projecting onto a plane
> Project $\mathbf{b}=(1,2,6)$ onto the plane spanned by $\mathbf{a}_1=(1,0,1)$ and $\mathbf{a}_2=(0,1,1)$:
>
> $$
> \mathbf{A}=\begin{bmatrix}1&0\\0&1\\1&1\end{bmatrix},\qquad\mathbf{A}^\top\mathbf{A}=\begin{bmatrix}2&1\\1&2\end{bmatrix},\qquad\mathbf{A}^\top\mathbf{b}=\begin{bmatrix}7\\8\end{bmatrix}.
> $$
>
> Solving $2\hat x_1+\hat x_2=7$, $\hat x_1+2\hat x_2=8$ gives $\hat{\mathbf{x}}=(2,3)$. So $\mathbf{p}=2\mathbf{a}_1+3\mathbf{a}_2=(2,3,5)$ and $\mathbf{e}=\mathbf{b}-\mathbf{p}=(-1,-1,1)$. Check perpendicularity: $\mathbf{a}_1^\top\mathbf{e}=-1+1=0$ and $\mathbf{a}_2^\top\mathbf{e}=-1+1=0$. Pythagoras: $\lVert\mathbf{b}\rVert^2=41=38+3=\lVert\mathbf{p}\rVert^2+\lVert\mathbf{e}\rVert^2$. The projection matrix is
>
> $$
> \mathbf{P}=\frac13\begin{bmatrix}2&-1&1\\-1&2&1\\1&1&2\end{bmatrix}=\mathbf{I}-\frac{\mathbf{n}\mathbf{n}^\top}{\mathbf{n}^\top\mathbf{n}},\qquad\mathbf{n}=(1,1,-1),
> $$
>
> because projecting onto a plane in $\mathbb{R}^3$ is the same as removing the component along its normal.

![[Probability and Linear Algebra/images/orthogonality-and-least-squares-projections.svg]]
*Figure 2. Projection drops a perpendicular. (a) Onto the line through $\mathbf{a}=(2,1)$: $\mathbf{b}=(1,3)$ projects to $\mathbf{p}=(2,1)$ and the error $(-1,2)$ is perpendicular to the line. (b) Onto the plane spanned by $\mathbf{a}_1=(1,0,1)$ and $\mathbf{a}_2=(0,1,1)$: $\mathbf{b}=(1,2,6)$ projects to $\mathbf{p}=(2,3,5)$, and the error $(-1,-1,1)$ is normal to the plane.*

### Properties of projection matrices

Every projection matrix $\mathbf{P}=\mathbf{A}(\mathbf{A}^\top\mathbf{A})^{-1}\mathbf{A}^\top$ has two algebraic properties.

- **$\mathbf{P}^2=\mathbf{P}$** (idempotent): projecting twice is the same as projecting once, because $\mathbf{p}$ is already in the subspace. Algebraically, $\mathbf{P}^2=\mathbf{A}(\mathbf{A}^\top\mathbf{A})^{-1}(\mathbf{A}^\top\mathbf{A})(\mathbf{A}^\top\mathbf{A})^{-1}\mathbf{A}^\top=\mathbf{P}$.
- **$\mathbf{P}^\top=\mathbf{P}$** (symmetric): transpose the product, using $\big((\mathbf{A}^\top\mathbf{A})^{-1}\big)^\top=(\mathbf{A}^\top\mathbf{A})^{-1}$ because $\mathbf{A}^\top\mathbf{A}$ is symmetric.

The converse also holds, so the two properties characterize orthogonal projections completely.

> [!proof]- A symmetric matrix with $\mathbf{P}^2=\mathbf{P}$ is the orthogonal projection onto $C(\mathbf{P})$
> Take any $\mathbf{b}$ and split it as $\mathbf{b}=\mathbf{P}\mathbf{b}+(\mathbf{I}-\mathbf{P})\mathbf{b}$. The first piece is in $C(\mathbf{P})$. The second is perpendicular to every vector $\mathbf{P}\mathbf{z}$ of $C(\mathbf{P})$:
>
> $$
> (\mathbf{P}\mathbf{z})^\top(\mathbf{I}-\mathbf{P})\mathbf{b}=\mathbf{z}^\top\mathbf{P}^\top(\mathbf{I}-\mathbf{P})\mathbf{b}=\mathbf{z}^\top(\mathbf{P}-\mathbf{P}^2)\mathbf{b}=0,
> $$
>
> using $\mathbf{P}^\top=\mathbf{P}$ and then $\mathbf{P}^2=\mathbf{P}$. So $\mathbf{P}\mathbf{b}$ is a point of $C(\mathbf{P})$ whose error is perpendicular to $C(\mathbf{P})$, which is the definition of the orthogonal projection.

Several more facts follow.

- **$\mathbf{I}-\mathbf{P}$ is also a projection**, onto the orthogonal complement. If $\mathbf{P}$ projects onto $C(\mathbf{A})$, then $\mathbf{I}-\mathbf{P}$ projects onto $N(\mathbf{A}^\top)$ and produces the error: $\mathbf{e}=(\mathbf{I}-\mathbf{P})\mathbf{b}$. (Check: it is symmetric, and $(\mathbf{I}-\mathbf{P})^2=\mathbf{I}-2\mathbf{P}+\mathbf{P}^2=\mathbf{I}-\mathbf{P}$.)
- **The eigenvalues are 0 and 1.** If $\mathbf{P}\mathbf{x}=\lambda\mathbf{x}$, then $\lambda\mathbf{x}=\mathbf{P}\mathbf{x}=\mathbf{P}^2\mathbf{x}=\lambda^2\mathbf{x}$, so $\lambda^2=\lambda$. Vectors in the subspace have $\lambda=1$; vectors perpendicular to it have $\lambda=0$.
- **The trace is the rank**, the dimension of the subspace, since the trace is the sum of the eigenvalues. For the plane example, $\operatorname{tr}\mathbf{P}=\frac13(2+2+2)=2$.

> [!warning] $\mathbf{P}^2=\mathbf{P}$ alone is not enough
> $\mathbf{P}=\begin{bmatrix}1&1\\0&0\end{bmatrix}$ satisfies $\mathbf{P}^2=\mathbf{P}$, and it does map every vector onto the $x_1$-axis, but along a slanted direction: $(0,1)$ goes to $(1,0)$ and the error $(-1,1)$ is not perpendicular to the axis. This is an **oblique** projection. Without symmetry the error is not orthogonal, and the projected point is not the closest one.

## Least squares

### When Ax = b has no solution

A linear system $\mathbf{A}\mathbf{x}=\mathbf{b}$ is **consistent** if it has at least one solution, which happens exactly when $\mathbf{b}\in C(\mathbf{A})$. Then it has one solution if the columns are independent ($r=n$), and infinitely many if they are not (add anything from the nullspace; see [[Elimination and the Nullspace#The complete solution to Ax = b|the complete solution]]). Otherwise the system is **inconsistent** and has **no** solution at all. (An inconsistent system does not have "no unique solution"; it has none.)

Inconsistency is the normal situation when there are more equations than unknowns, $m>n$. The column space is an $n$-dimensional subspace of $\mathbb{R}^m$, a thin slice of the whole space, and a vector of noisy measurements $\mathbf{b}$ almost never lands exactly in it. Since no $\mathbf{x}$ makes the error $\mathbf{A}\mathbf{x}-\mathbf{b}$ zero, we make it as small as possible:

$$
\hat{\mathbf{x}}=\arg\min_{\mathbf{x}}\ \lVert\mathbf{A}\mathbf{x}-\mathbf{b}\rVert^2=\arg\min_{\mathbf{x}}\ \sum_{i=1}^m\big(\mathbf{a}_i^{\top}\mathbf{x}-b_i\big)^2,
$$

where $\mathbf{a}_i^\top$ is row $i$ of $\mathbf{A}$. This is **least squares**: the sum of squared errors over all equations is minimized.

**Least squares is projection.** Let $\mathbf{p}$ be the projection of $\mathbf{b}$ onto $C(\mathbf{A})$ and $\mathbf{e}=\mathbf{b}-\mathbf{p}$. For any $\mathbf{x}$, split the error as

$$
\mathbf{A}\mathbf{x}-\mathbf{b}=\underbrace{(\mathbf{A}\mathbf{x}-\mathbf{p})}_{\in\,C(\mathbf{A})}-\underbrace{\mathbf{e}}_{\in\,N(\mathbf{A}^\top)}.
$$

The two pieces are perpendicular, so Pythagoras gives

$$
\lVert\mathbf{A}\mathbf{x}-\mathbf{b}\rVert^2=\lVert\mathbf{A}\mathbf{x}-\mathbf{p}\rVert^2+\lVert\mathbf{e}\rVert^2\ \ge\ \lVert\mathbf{e}\rVert^2.
$$

The second term does not depend on $\mathbf{x}$, and the first is zero exactly when $\mathbf{A}\mathbf{x}=\mathbf{p}$. So the least-squares solutions are exactly the solutions of $\mathbf{A}\hat{\mathbf{x}}=\mathbf{p}$: **replace $\mathbf{b}$ by its projection and solve**. That system is always consistent, because $\mathbf{p}$ is in the column space. The smallest achievable error is $\lVert\mathbf{e}\rVert$, the distance from $\mathbf{b}$ to the column space.

### The normal equations

The projection condition from the previous section, error perpendicular to every column, is

$$
\mathbf{A}^\top\mathbf{A}\,\hat{\mathbf{x}}=\mathbf{A}^\top\mathbf{b}.
\tag{1}
$$

These are the **normal equations**; "normal" means perpendicular, and the name records that (1) says $\mathbf{A}^\top\mathbf{e}=\mathbf{0}$. They turn an inconsistent $m\times n$ system into a consistent $n\times n$ one. The matrix $\mathbf{A}^\top\mathbf{A}$ is square, symmetric and positive semi-definite ($\mathbf{x}^\top\mathbf{A}^\top\mathbf{A}\mathbf{x}=\lVert\mathbf{A}\mathbf{x}\rVert^2\ge0$).

**When is $\mathbf{A}^\top\mathbf{A}$ invertible?** Exactly when the columns of $\mathbf{A}$ are independent. The reason is that $\mathbf{A}^\top\mathbf{A}$ and $\mathbf{A}$ have the same nullspace.

> [!proof]- Proof: $N(\mathbf{A}^\top\mathbf{A})=N(\mathbf{A})$, so $\mathbf{A}^\top\mathbf{A}$ is invertible iff $\mathbf{A}$ has independent columns
> If $\mathbf{A}\mathbf{x}=\mathbf{0}$, then certainly $\mathbf{A}^\top\mathbf{A}\mathbf{x}=\mathbf{0}$. Conversely, if $\mathbf{A}^\top\mathbf{A}\mathbf{x}=\mathbf{0}$, multiply on the left by $\mathbf{x}^\top$:
>
> $$
> 0=\mathbf{x}^\top\mathbf{A}^\top\mathbf{A}\mathbf{x}=(\mathbf{A}\mathbf{x})^\top(\mathbf{A}\mathbf{x})=\lVert\mathbf{A}\mathbf{x}\rVert^2,
> $$
>
> so $\mathbf{A}\mathbf{x}=\mathbf{0}$. The two nullspaces are equal. Now $\mathbf{A}^\top\mathbf{A}$ is a square $n\times n$ matrix, so it is invertible iff its nullspace is $\{\mathbf{0}\}$, iff $N(\mathbf{A})=\{\mathbf{0}\}$, iff the $n$ columns of $\mathbf{A}$ are independent.
>
> By the counting theorem, equal nullspaces also mean equal rank: $\operatorname{rank}(\mathbf{A}^\top\mathbf{A})=n-\dim N(\mathbf{A})=\operatorname{rank}(\mathbf{A})$.

With independent columns, $\hat{\mathbf{x}}=(\mathbf{A}^\top\mathbf{A})^{-1}\mathbf{A}^\top\mathbf{b}$ is the unique least-squares solution. With dependent columns (a duplicated feature, or fewer equations than unknowns), (1) still has solutions, since it always does, but infinitely many of them; the [[#The four subspaces and least squares|last section]] picks the best one.

**Cost.** Forming $\mathbf{A}^\top\mathbf{A}$ takes about $mn^2$ multiply-adds (each of its $n^2/2$ distinct entries is a dot product of length $m$), forming $\mathbf{A}^\top\mathbf{b}$ takes $mn$, and solving the $n\times n$ system by Cholesky elimination takes about $n^3/3$. The total is $O(mn^2+n^3)$. In machine-learning notation, with a design matrix $\mathbf{X}\in\mathbb{R}^{N\times D}$, that is $O(ND^2+D^3)$: linear in the number of examples, cubic in the number of features. When $D$ is large, iterative methods that only need products with $\mathbf{X}$ and $\mathbf{X}^\top$ take over; see [[Linear Models#Iterative methods|Linear Models]] and [[First-Order Optimization]].

### The calculus derivation

The same equations come from setting a gradient to zero. Expand the objective:

$$
f(\mathbf{x})=\lVert\mathbf{A}\mathbf{x}-\mathbf{b}\rVert^2=\mathbf{x}^\top\mathbf{A}^\top\mathbf{A}\mathbf{x}-2\,\mathbf{b}^\top\mathbf{A}\mathbf{x}+\mathbf{b}^\top\mathbf{b}.
$$

Using $\nabla(\mathbf{x}^\top\mathbf{S}\mathbf{x})=2\mathbf{S}\mathbf{x}$ for symmetric $\mathbf{S}$ and $\nabla(\mathbf{c}^\top\mathbf{x})=\mathbf{c}$ from [[Multivariable Calculus#Matrix calculus|matrix calculus]],

$$
\nabla f(\mathbf{x})=2\mathbf{A}^\top\mathbf{A}\mathbf{x}-2\mathbf{A}^\top\mathbf{b}=2\mathbf{A}^\top(\mathbf{A}\mathbf{x}-\mathbf{b}),\qquad\nabla^2f(\mathbf{x})=2\mathbf{A}^\top\mathbf{A}.
$$

Setting $\nabla f=\mathbf{0}$ gives the normal equations (1). The Hessian $2\mathbf{A}^\top\mathbf{A}$ is positive semi-definite, so $f$ is [[Multivariable Calculus#Convexity|convex]] and every stationary point is a global minimum; with independent columns the Hessian is positive definite and the minimum is unique.

The two derivations say the same thing. The gradient is $-2\mathbf{A}^\top\mathbf{e}$, where $\mathbf{e}=\mathbf{b}-\mathbf{A}\mathbf{x}$ is the error. "Gradient zero" (calculus) and "error perpendicular to every column" (geometry) are one equation. Each entry of $\mathbf{A}^\top\mathbf{e}$ measures how much the error still points along one column; at the optimum, no column can reduce the error any further.

### A line fit by hand

Fit a line $b=C+Dt$ to the four points $(t,b)=(0,1),(1,3),(2,4),(3,4)$. Each point gives one equation in the two unknowns $C$ and $D$:

$$
\begin{bmatrix}1&0\\1&1\\1&2\\1&3\end{bmatrix}\begin{bmatrix}C\\D\end{bmatrix}=\begin{bmatrix}1\\3\\4\\4\end{bmatrix}.
$$

No line passes through all four points (the first two equations force $C=1$, $D=2$, and the third then fails), so solve the normal equations instead:

$$
\mathbf{A}^\top\mathbf{A}=\begin{bmatrix}4&6\\6&14\end{bmatrix},\qquad\mathbf{A}^\top\mathbf{b}=\begin{bmatrix}12\\23\end{bmatrix}.
$$

Here $4=$ number of points, $6=\sum t_i$, $14=\sum t_i^2$, $12=\sum b_i$ and $23=\sum t_ib_i$. The determinant is $4\cdot14-6\cdot6=20$, and

$$
\begin{bmatrix}C\\D\end{bmatrix}=\frac{1}{20}\begin{bmatrix}14&-6\\-6&4\end{bmatrix}\begin{bmatrix}12\\23\end{bmatrix}=\frac{1}{20}\begin{bmatrix}30\\20\end{bmatrix}=\begin{bmatrix}1.5\\1\end{bmatrix}.
$$

The best line is $b=1.5+t$. Its predictions at $t=0,1,2,3$ are $\mathbf{p}=(1.5,2.5,3.5,4.5)$, and the errors are $\mathbf{e}=\mathbf{b}-\mathbf{p}=(-0.5,0.5,0.5,-0.5)$. Check both normal equations: the errors sum to zero (perpendicular to the column of ones), and $\sum t_ie_i=0+0.5+1-1.5=0$ (perpendicular to the column of $t$'s). And Pythagoras in $\mathbb{R}^4$: $\lVert\mathbf{b}\rVert^2=42=41+1=\lVert\mathbf{p}\rVert^2+\lVert\mathbf{e}\rVert^2$.

![[Probability and Linear Algebra/images/orthogonality-and-least-squares-line-fit.svg]]
*Figure 3. The least-squares line $b=1.5+t$ through four points. The vertical bars are the errors $e_i=\pm0.5$, whose squares sum to 1, the smallest possible. In $\mathbb{R}^4$ the same picture is a projection: $\mathbf{p}=(1.5,2.5,3.5,4.5)$ is the point of the 2-dimensional column space closest to $\mathbf{b}=(1,3,4,4)$.*

Keep the two pictures apart. In Figure 3 the errors are vertical distances in the $(t,b)$ plane, not perpendicular distances to the line. The perpendicularity of least squares happens in $\mathbb{R}^m$, one axis per data point, where $\mathbf{e}$ is perpendicular to the column space. (Minimizing perpendicular distances to the line in the data plane is a different problem, total least squares, and its answer comes from [[Matrix Factorization and Dimensionality Reduction#Principal component analysis (PCA)|PCA]].)

[[Linear Models]] puts this computation in its statistical setting: with a design matrix $\mathbf{X}$ and targets $\mathbf{y}$, the least-squares weights are the maximum-likelihood estimate under Gaussian noise, and adding a penalty gives ridge regression.

### Weighted least squares

Sometimes the equations deserve different trust: some measurements are more precise, or one data point stands for several. Give equation $i$ a weight $w_i>0$ and minimize

$$
\sum_{i=1}^m w_i\big(\mathbf{a}_i^\top\mathbf{x}-b_i\big)^2=(\mathbf{A}\mathbf{x}-\mathbf{b})^\top\mathbf{W}(\mathbf{A}\mathbf{x}-\mathbf{b}),
$$

with $\mathbf{W}=\operatorname{diag}(w_1,\dots,w_m)$. Multiplying row $i$ of $\mathbf{A}$ and entry $i$ of $\mathbf{b}$ by $\sqrt{w_i}$ turns this into ordinary least squares, so the normal equations become

$$
\mathbf{A}^\top\mathbf{W}\mathbf{A}\,\hat{\mathbf{x}}=\mathbf{A}^\top\mathbf{W}\mathbf{b}.
$$

Geometrically, this is still a projection, but perpendicularity is measured in the inner product $\langle\mathbf{u},\mathbf{v}\rangle_{\mathbf{W}}=\mathbf{u}^\top\mathbf{W}\mathbf{v}$: the condition is $\mathbf{A}^\top\mathbf{W}\mathbf{e}=\mathbf{0}$. In the line fit, giving the last point $(3,4)$ weight 4 and the others weight 1 changes $\mathbf{A}^\top\mathbf{W}\mathbf{A}$ to $\begin{bmatrix}7&15\\15&41\end{bmatrix}$ and $\mathbf{A}^\top\mathbf{W}\mathbf{b}$ to $(24,59)$, and the line becomes $b\approx1.60+0.85\,t$: flatter, pulled towards the heavily weighted point, which lay below the old line. [[Linear Models#Weighted linear regression|Weighted linear regression]] uses this for noise of unequal variance and inside Newton's method for logistic regression.

> [!example]- Code: weighted least squares, two ways
> ```python
> import numpy as np
> t = np.array([0.0, 1, 2, 3]); b = np.array([1.0, 3, 4, 4])
> A = np.column_stack([np.ones_like(t), t])
> w = np.array([1.0, 1, 1, 4])                      # trust the last point four times as much
>
> x_normal = np.linalg.solve(A.T @ (w[:, None] * A), A.T @ (w * b))   # A^T W A x = A^T W b
> s = np.sqrt(w)                                    # or scale row i by sqrt(w_i), then plain LS
> x_scaled = np.linalg.lstsq(s[:, None] * A, s * b, rcond=None)[0]
> print(x_normal, x_scaled)                         # both [1.597 0.855]: b = 1.60 + 0.85 t
> ```
> The row-scaling version is the one to use: it can be solved with QR, so it inherits the stability of [[#Least squares with QR]].

## Gram–Schmidt and A = QR

With orthonormal columns, least squares is easy: $\mathbf{Q}^\top\mathbf{Q}=\mathbf{I}$, so the normal equations reduce to $\hat{\mathbf{x}}=\mathbf{Q}^\top\mathbf{b}$, with nothing to invert. Gram–Schmidt manufactures orthonormal columns from any independent ones.

### Gram–Schmidt by hand

The idea: take the vectors one at a time, subtract from each its projections onto the directions already found, and normalize what is left. What remains is perpendicular to everything before it.

$$
\mathbf{w}_j=\mathbf{a}_j-\sum_{i<j}(\mathbf{q}_i^\top\mathbf{a}_j)\,\mathbf{q}_i,\qquad\mathbf{q}_j=\frac{\mathbf{w}_j}{\lVert\mathbf{w}_j\rVert}.
$$

Each subtracted term is a [[#Projection onto a line|projection onto a line]], and with unit vectors the denominator $\mathbf{q}_i^\top\mathbf{q}_i$ is 1. If the $\mathbf{a}$'s are independent, $\mathbf{w}_j$ is never zero.

![[Probability and Linear Algebra/images/orthogonality-and-least-squares-gram-schmidt.svg]]
*Figure 4. One Gram–Schmidt step. $\mathbf{q}_1$ is $\mathbf{a}_1$ rescaled to unit length. Subtracting from $\mathbf{a}_2$ its projection onto $\mathbf{q}_1$ leaves $\mathbf{w}_2$, which is perpendicular to $\mathbf{q}_1$; rescaling it gives $\mathbf{q}_2$.*

> [!example] Worked example: three vectors in $\mathbb{R}^3$
> Orthonormalize $\mathbf{a}_1=(1,2,2)$, $\mathbf{a}_2=(3,3,0)$ and $\mathbf{a}_3=(5,1,1)$.
>
> **Step 1.** $\lVert\mathbf{a}_1\rVert=3$, so $\mathbf{q}_1=\frac13(1,2,2)$.
>
> **Step 2.** $\mathbf{q}_1^\top\mathbf{a}_2=\frac13(3+6+0)=3$. Subtract: $\mathbf{w}_2=(3,3,0)-3\cdot\frac13(1,2,2)=(2,1,-2)$, of length 3, so $\mathbf{q}_2=\frac13(2,1,-2)$.
>
> **Step 3.** $\mathbf{q}_1^\top\mathbf{a}_3=\frac13(5+2+2)=3$ and $\mathbf{q}_2^\top\mathbf{a}_3=\frac13(10+1-2)=3$. Subtract both: $\mathbf{w}_3=(5,1,1)-(1,2,2)-(2,1,-2)=(2,-2,1)$, of length 3, so $\mathbf{q}_3=\frac13(2,-2,1)$.
>
> These are the orthonormal vectors from the start of the lecture.

### A = QR

Gram–Schmidt records more than the $\mathbf{q}$'s. Reading the update backwards, each original vector is a combination of the $\mathbf{q}$'s found so far:

$$
\mathbf{a}_j=\sum_{i<j}(\mathbf{q}_i^\top\mathbf{a}_j)\,\mathbf{q}_i+\lVert\mathbf{w}_j\rVert\,\mathbf{q}_j.
$$

Vector $\mathbf{a}_j$ uses only $\mathbf{q}_1,\dots,\mathbf{q}_j$, never a later one. In matrix form that is $\mathbf{A}=\mathbf{Q}\mathbf{R}$ with $\mathbf{R}$ **upper triangular**: $r_{ij}=\mathbf{q}_i^\top\mathbf{a}_j$ above the diagonal and $r_{jj}=\lVert\mathbf{w}_j\rVert>0$ on it. Since $\mathbf{Q}^\top\mathbf{Q}=\mathbf{I}$, you can also read $\mathbf{R}=\mathbf{Q}^\top\mathbf{A}$ directly. For the example,

$$
\underbrace{\begin{bmatrix}1&3&5\\2&3&1\\2&0&1\end{bmatrix}}_{\mathbf{A}}=\underbrace{\frac13\begin{bmatrix}1&2&2\\2&1&-2\\2&-2&1\end{bmatrix}}_{\mathbf{Q}}\underbrace{\begin{bmatrix}3&3&3\\0&3&3\\0&0&3\end{bmatrix}}_{\mathbf{R}}.
$$

For a tall $\mathbf{A}\in\mathbb{R}^{m\times n}$ with independent columns, this **reduced** QR has $\mathbf{Q}\in\mathbb{R}^{m\times n}$ and $\mathbf{R}\in\mathbb{R}^{n\times n}$. The **full** QR extends $\mathbf{Q}$ to an $m\times m$ orthogonal matrix by adding an orthonormal basis of $N(\mathbf{A}^\top)$, and pads $\mathbf{R}$ with zero rows. The column spaces match: $C(\mathbf{A})=C(\mathbf{Q})$, and the first $k$ columns of $\mathbf{Q}$ span the same space as the first $k$ columns of $\mathbf{A}$, for every $k$.

### Least squares with QR

Substitute $\mathbf{A}=\mathbf{Q}\mathbf{R}$ into the normal equations. Since $\mathbf{A}^\top\mathbf{A}=\mathbf{R}^\top\mathbf{Q}^\top\mathbf{Q}\mathbf{R}=\mathbf{R}^\top\mathbf{R}$,

$$
\mathbf{R}^\top\mathbf{R}\,\hat{\mathbf{x}}=\mathbf{R}^\top\mathbf{Q}^\top\mathbf{b}\quad\Longrightarrow\quad\boxed{\;\mathbf{R}\,\hat{\mathbf{x}}=\mathbf{Q}^\top\mathbf{b}\;}
$$

after cancelling the invertible $\mathbf{R}^\top$. The right-hand side is a set of dot products, and the triangular system is solved by back-substitution in $O(n^2)$ operations. The projection is $\mathbf{p}=\mathbf{Q}\mathbf{Q}^\top\mathbf{b}$.

> [!example] Worked example: the line fit again, by QR
> The columns of $\mathbf{A}$ are $\mathbf{1}=(1,1,1,1)$ and $\mathbf{t}=(0,1,2,3)$.
> - $\lVert\mathbf{1}\rVert=2$, so $\mathbf{q}_1=\frac12(1,1,1,1)$ and $r_{11}=2$.
> - $r_{12}=\mathbf{q}_1^\top\mathbf{t}=\frac12(0+1+2+3)=3$, and $\mathbf{w}_2=\mathbf{t}-3\mathbf{q}_1=(-1.5,-0.5,0.5,1.5)$: the times, centred at their mean. Its length is $\sqrt5$, so $r_{22}=\sqrt5$.
>
> Then $\mathbf{Q}^\top\mathbf{b}=\big(\tfrac12(1+3+4+4),\ \tfrac{1}{\sqrt5}(-1.5-1.5+2+6)\big)=(6,\sqrt5)$, and back-substitution in
>
> $$
> \begin{bmatrix}2&3\\0&\sqrt5\end{bmatrix}\begin{bmatrix}C\\D\end{bmatrix}=\begin{bmatrix}6\\\sqrt5\end{bmatrix}
> $$
>
> gives $D=1$, then $2C+3=6$, so $C=1.5$. Same line as before. Gram–Schmidt automatically centred the data, which is why fitting a slope to centred inputs is so convenient.

### Why QR beats the normal equations

On paper the two methods give the same $\hat{\mathbf{x}}$. In floating-point arithmetic they do not, and the reason is the **condition number** $\kappa(\mathbf{A})$: the ratio between the most and the least that $\mathbf{A}$ stretches a unit vector, $\max\lVert\mathbf{A}\mathbf{x}\rVert/\min\lVert\mathbf{A}\mathbf{x}\rVert$ over unit $\mathbf{x}$. (These extreme stretches are the largest and smallest singular values, defined in [[Matrix Factorization and Dimensionality Reduction#Singular value decomposition (SVD)|the SVD]].) Roughly, solving a problem with condition number $\kappa$ in arithmetic with relative precision $\varepsilon\approx10^{-16}$ loses a factor $\kappa$: about $\log_{10}\kappa$ correct digits disappear.

Forming $\mathbf{A}^\top\mathbf{A}$ **squares** the condition number:

$$
\kappa(\mathbf{A}^\top\mathbf{A})=\kappa(\mathbf{A})^2,
$$

because $\mathbf{x}^\top\mathbf{A}^\top\mathbf{A}\mathbf{x}=\lVert\mathbf{A}\mathbf{x}\rVert^2$ squares every stretch. A design matrix with $\kappa=10^8$, which is common with nearly collinear features or polynomial features, has a perfectly solvable least-squares problem, but $\mathbf{A}^\top\mathbf{A}$ has $\kappa=10^{16}$ and is numerically singular: the normal equations return no correct digits. QR works with $\mathbf{A}$ itself, and $\mathbf{R}$ has the same condition number as $\mathbf{A}$ (since $\mathbf{Q}$ preserves lengths), so it only loses the unavoidable factor $\kappa$.[^trefethen1997]<sup>,</sup>[^higham2002]

![[Probability and Linear Algebra/images/orthogonality-and-least-squares-conditioning.svg]]
*Figure 5. Accuracy on $100\times10$ test problems with prescribed condition number and an exact solution (median of 10 trials). (a) The normal equations lose accuracy like $\varepsilon\kappa^2$ and return garbage by $\kappa=10^8$ (relative error 0.55), while Householder QR follows $\varepsilon\kappa$ (error $4\times10^{-10}$ at $\kappa=10^8$). (b) How orthogonal the computed $\mathbf{Q}$ is: classical Gram–Schmidt loses orthogonality like $\kappa^2$, modified Gram–Schmidt like $\kappa$, and Householder QR stays orthogonal to machine precision.*

Figure 5(b) shows a second numerical point. The textbook Gram–Schmidt above (**classical**) computes every coefficient $\mathbf{q}_i^\top\mathbf{a}_j$ from the original vector, and in floating point the computed $\mathbf{q}$'s gradually stop being orthogonal. **Modified** Gram–Schmidt subtracts the projections one at a time and computes each coefficient from the partially reduced vector; it is the same in exact arithmetic but far more robust.[^bjorck1996] Library routines (`numpy.linalg.qr`, LAPACK's `geqrf`) use neither: they build $\mathbf{Q}$ from a sequence of Householder reflections, which keeps it orthogonal to machine precision. The cost is about $2mn^2$ operations, roughly twice the normal equations, which is usually a price worth paying. When even QR is not enough (rank-deficient or nearly so), the SVD is the most reliable tool, and `numpy.linalg.lstsq` uses it.

> [!example]- Code: QR versus the normal equations
> ```python
> import numpy as np
> from scipy.linalg import solve_triangular
>
> def test_matrix(m, n, kappa, rng):
>     """m x n matrix with singular values from 1 down to 1/kappa."""
>     U, _ = np.linalg.qr(rng.standard_normal((m, n)))
>     V, _ = np.linalg.qr(rng.standard_normal((n, n)))
>     return U @ np.diag(np.logspace(0, -np.log10(kappa), n)) @ V.T
>
> rng = np.random.default_rng(0)
> m, n = 100, 10
> x_true = rng.standard_normal(n)
> for kappa in [1e2, 1e4, 1e6, 1e8]:
>     A = test_matrix(m, n, kappa, rng)
>     b = A @ x_true                                      # consistent system: zero residual
>     x_ne = np.linalg.solve(A.T @ A, A.T @ b)            # normal equations
>     Q, R = np.linalg.qr(A)                              # Householder QR, reduced
>     x_qr = solve_triangular(R, Q.T @ b)
>     err = lambda x: np.linalg.norm(x - x_true) / np.linalg.norm(x_true)
>     print(f"kappa={kappa:.0e}  normal eqs {err(x_ne):.1e}   QR {err(x_qr):.1e}")
> ```
> At $\kappa=10^8$ the normal equations give a relative error of order $10^{-1}$; QR gives about $10^{-9}$.

> [!example]- Code: classical and modified Gram–Schmidt
> ```python
> import numpy as np
>
> def gram_schmidt(A, modified=True):
>     """Reduced QR of A (independent columns) by classical or modified Gram-Schmidt."""
>     m, n = A.shape
>     Q, R = np.zeros((m, n)), np.zeros((n, n))
>     for j in range(n):
>         v = A[:, j].copy()
>         for i in range(j):
>             # classical: coefficient from the original column; modified: from the updated v
>             R[i, j] = Q[:, i] @ (v if modified else A[:, j])
>             v -= R[i, j] * Q[:, i]
>         R[j, j] = np.linalg.norm(v)
>         Q[:, j] = v / R[j, j]
>     return Q, R
>
> A = np.array([[1., 3, 5], [2, 3, 1], [2, 0, 1]])
> Q, R = gram_schmidt(A)
> print(R)                                  # [[3,3,3],[0,3,3],[0,0,3]]
> print(np.allclose(Q @ R, A), np.allclose(Q.T @ Q, np.eye(3)))
> ```

## The four subspaces and least squares

Least squares fits neatly into [[The Four Fundamental Subspaces#The big picture|the big picture]]. On the output side, $\mathbb{R}^m$ splits into $C(\mathbf{A})$ and $N(\mathbf{A}^\top)$, and least squares splits $\mathbf{b}$ along them:

$$
\mathbf{b}=\underbrace{\mathbf{p}}_{\mathbf{P}\mathbf{b}\ \in\ C(\mathbf{A})}+\underbrace{\mathbf{e}}_{(\mathbf{I}-\mathbf{P})\mathbf{b}\ \in\ N(\mathbf{A}^\top)}.
$$

The part in the column space is reachable and gets fitted exactly; the part in the left nullspace cannot be reached by any $\mathbf{x}$ and is left over as the error.

On the input side, if the columns of $\mathbf{A}$ are dependent, the nullspace is not just $\{\mathbf{0}\}$ and $\mathbf{A}\hat{\mathbf{x}}=\mathbf{p}$ has infinitely many solutions: one particular solution plus anything in $N(\mathbf{A})$. They all give the same fit $\mathbf{p}$ and the same error. Which should we report? The natural choice is the **shortest**, and it is the one in the row space.

> [!proof]- The minimum-norm least-squares solution is the one in the row space
> Every least-squares solution splits as $\hat{\mathbf{x}}=\mathbf{x}_r+\mathbf{x}_n$ into row-space and nullspace parts. All of them satisfy $\mathbf{A}\hat{\mathbf{x}}=\mathbf{A}\mathbf{x}_r=\mathbf{p}$, and $\mathbf{A}$ is one-to-one on the row space, so **all least-squares solutions have the same row-space part** $\mathbf{x}_r$; they differ only in $\mathbf{x}_n$. The two parts are perpendicular, so
>
> $$
> \lVert\hat{\mathbf{x}}\rVert^2=\lVert\mathbf{x}_r\rVert^2+\lVert\mathbf{x}_n\rVert^2\ \ge\ \lVert\mathbf{x}_r\rVert^2,
> $$
>
> with equality exactly when $\mathbf{x}_n=\mathbf{0}$. The unique shortest solution is $\mathbf{x}_r$, the one with no nullspace component.

> [!example] Worked example: two identical columns
> Let $\mathbf{A}$ be the $3\times2$ matrix of ones and $\mathbf{b}=(1,2,3)$. The columns are equal, so $C(\mathbf{A})$ is the line through $(1,1,1)$, and the projection of $\mathbf{b}$ onto it is the mean: $\mathbf{p}=(2,2,2)$, with error $\mathbf{e}=(-1,0,1)$. Every $\mathbf{x}$ with $x_1+x_2=2$ is a least-squares solution: $(2,0)$, $(0,2)$, $(3,-1)$ and so on. The row space is the line through $(1,1)$ and the nullspace is the line through $(1,-1)$. The solution in the row space is $\mathbf{x}^+=(1,1)$, and every other solution is $\mathbf{x}^+$ plus a nullspace vector, for example $(2,0)=(1,1)+(1,-1)$, with $\lVert(2,0)\rVert^2=4=2+2$. The shortest solution splits the weight evenly between the two identical features.

![[Probability and Linear Algebra/images/orthogonality-and-least-squares-min-norm.svg]]
*Figure 6. With two identical columns, every point on the line $x_1+x_2=2$ fits the data equally well. The shortest of them, $\mathbf{x}^+=(1,1)$, is where that line crosses the row space; moving along the line adds a nullspace component and only makes $\mathbf{x}$ longer.*

The map $\mathbf{b}\mapsto\mathbf{x}^+$ is linear, and its matrix is the **pseudo-inverse** $\mathbf{A}^+$: it projects $\mathbf{b}$ onto the column space, then inverts the one-to-one map from the row space onto the column space. When the columns are independent there is no nullspace and $\mathbf{A}^+=(\mathbf{A}^\top\mathbf{A})^{-1}\mathbf{A}^\top$; in general $\mathbf{A}^+$ is computed from the SVD, as in [[Matrix Factorization and Dimensionality Reduction#The pseudo-inverse|Matrix Factorization and Dimensionality Reduction]]. Ridge regression offers another way out of dependent columns: it solves $(\mathbf{A}^\top\mathbf{A}+\lambda\mathbf{I})\mathbf{x}=\mathbf{A}^\top\mathbf{b}$, which is always invertible for $\lambda>0$, and its solution tends to $\mathbf{x}^+$ as $\lambda\to0$ (see [[Linear Models#MAP estimation and regularization|MAP estimation and regularization]]).

## Key takeaways

- Orthonormal columns mean $\mathbf{Q}^\top\mathbf{Q}=\mathbf{I}$, and coordinates are dot products: $\mathbf{c}=\mathbf{Q}^\top\mathbf{b}$. A square $\mathbf{Q}$ is a rotation or reflection: $\mathbf{Q}^{-1}=\mathbf{Q}^\top$, lengths and angles are preserved, $\lvert\lambda\rvert=1$ and $\det\mathbf{Q}=\pm1$. A tall $\mathbf{Q}$ gives the projection $\mathbf{Q}\mathbf{Q}^\top$.
- Projection onto $C(\mathbf{A})$ is $\mathbf{P}=\mathbf{A}(\mathbf{A}^\top\mathbf{A})^{-1}\mathbf{A}^\top$; onto a line it is $\mathbf{a}\mathbf{a}^\top/\mathbf{a}^\top\mathbf{a}$. Orthogonal projections are exactly the symmetric idempotent matrices; their eigenvalues are 0 and 1, their trace is their rank, and $\mathbf{I}-\mathbf{P}$ projects onto the complement.
- An inconsistent system has no solution. Least squares replaces $\mathbf{b}$ by its projection $\mathbf{p}$, and the error $\mathbf{e}=\mathbf{b}-\mathbf{p}$ is perpendicular to the column space. Geometry ($\mathbf{A}^\top\mathbf{e}=\mathbf{0}$) and calculus ($\nabla\lVert\mathbf{A}\mathbf{x}-\mathbf{b}\rVert^2=\mathbf{0}$) both give the normal equations $\mathbf{A}^\top\mathbf{A}\hat{\mathbf{x}}=\mathbf{A}^\top\mathbf{b}$, solved in $O(mn^2+n^3)$.
- $\mathbf{A}^\top\mathbf{A}$ is invertible iff the columns of $\mathbf{A}$ are independent, because $N(\mathbf{A}^\top\mathbf{A})=N(\mathbf{A})$. Weighted least squares solves $\mathbf{A}^\top\mathbf{W}\mathbf{A}\hat{\mathbf{x}}=\mathbf{A}^\top\mathbf{W}\mathbf{b}$.
- Gram–Schmidt subtracts projections to produce orthonormal vectors, and records the coefficients in an upper-triangular $\mathbf{R}$: $\mathbf{A}=\mathbf{Q}\mathbf{R}$. Least squares becomes $\mathbf{R}\hat{\mathbf{x}}=\mathbf{Q}^\top\mathbf{b}$, which loses accuracy like $\kappa$ instead of $\kappa^2$. In practice, use Householder QR or the SVD, not the normal equations, for ill-conditioned problems.
- With dependent columns, all least-squares solutions share the same row-space part. The shortest one is that row-space part, $\mathbf{x}^+=\mathbf{A}^+\mathbf{b}$.

## Where this shows up next

- [[Eigenvalues and Eigenvectors]] shows that every symmetric matrix is $\mathbf{Q}\boldsymbol\Lambda\mathbf{Q}^\top$, a sum of projections onto orthogonal eigenvector directions weighted by the eigenvalues.
- [[First-Order Optimization]] minimizes $\lVert\mathbf{A}\mathbf{x}-\mathbf{b}\rVert^2$ by gradient descent instead of solving (1), and the condition number of $\mathbf{A}^\top\mathbf{A}$ reappears as the speed limit of gradient descent.
- [[Five Factorizations of a Matrix]] places $\mathbf{A}=\mathbf{Q}\mathbf{R}$ alongside $\mathbf{L}\mathbf{U}$, the eigendecomposition and the SVD.
- [[Linear Models]] is least squares with a design matrix, plus the statistical view and regularization; [[Matrix Factorization and Dimensionality Reduction]] uses the SVD for the pseudo-inverse and projects data onto principal subspaces (PCA); [[Linear Function Approximation]] in RL projects value functions onto the span of features.

<!-- References: footnote definitions below render as the numbered "References" list at the bottom of the page -->

[^strangorth]: Strang, G. (2020). *Part 3: Orthogonal Vectors*. A 2020 Vision of Linear Algebra, MIT OpenCourseWare RES.18-010. [video](https://www.youtube.com/watch?v=j8hEnyOiwhw) · [course page](https://ocw.mit.edu/courses/res-18-010-a-2020-vision-of-linear-algebra-spring-2020/)
[^strang4sub]: Strang, G. *The Four Fundamental Subspaces and Least Squares*. A 2020 Vision of Linear Algebra, MIT OpenCourseWare RES.18-010. [video](https://www.youtube.com/watch?v=jY-Mu6XQ3NU)
[^strangbook]: Strang, G. (2023). *Introduction to Linear Algebra* (6th ed.), Chapter 4 (Orthogonality). Wellesley-Cambridge Press. [book website](https://math.mit.edu/linearalgebra/)
[^mml]: Deisenroth, M. P., Faisal, A. A., and Ong, C. S. (2020). *Mathematics for Machine Learning*, Chapter 3 (Analytic Geometry). Cambridge University Press. [book website](https://mml-book.github.io/)
[^trefethen1997]: Trefethen, L. N., and Bau, D. (1997). *Numerical Linear Algebra*, Lectures 7–11 (QR and least squares) and 18–19 (conditioning and stability of least squares). SIAM. [doi:10.1137/1.9780898719574](https://doi.org/10.1137/1.9780898719574)
[^higham2002]: Higham, N. J. (2002). *Accuracy and Stability of Numerical Algorithms* (2nd ed.), Chapters 19–20. SIAM. [doi:10.1137/1.9780898718027](https://doi.org/10.1137/1.9780898718027)
[^bjorck1996]: Björck, Å. (1996). *Numerical Methods for Least Squares Problems*. SIAM. [doi:10.1137/1.9781611971484](https://doi.org/10.1137/1.9781611971484)
