---
title: Eigenvalues and Eigenvectors
tags:
  - linear-algebra
  - foundations
date: 2026-11-12
draft: false
prev: "[[Orthogonality and Least Squares]]"
next: "[[First-Order Optimization]]"
---

> **Prerequisites:** [[The Four Fundamental Subspaces]] (nullspace, rank) and [[Orthogonality and Least Squares]] (orthogonal matrices, projections, $\mathbf{A}=\mathbf{Q}\mathbf{R}$). The section on positive definite matrices revisits the Hessian test of [[Multivariable Calculus]].
>
> **Series:** this is part 6 of 8. [[Multivariable Calculus]] → [[Column Space and Rank]] → [[Elimination and the Nullspace]] → [[The Four Fundamental Subspaces]] → [[Orthogonality and Least Squares]] → **Eigenvalues and Eigenvectors** → [[First-Order Optimization]] → [[Five Factorizations of a Matrix]].
>
> **TL;DR:**
> - Most vectors change direction when a square matrix multiplies them. **Eigenvectors** do not: $\mathbf{A}\mathbf{x}=\lambda\mathbf{x}$, so $\mathbf{A}$ just stretches them by the **eigenvalue** $\lambda$. Eigenvalues are the roots of $\det(\mathbf{A}-\lambda\mathbf{I})=0$; there are $n$ of them, counted with multiplicity, possibly complex.
> - With $n$ independent eigenvectors, $\mathbf{A}=\mathbf{X}\boldsymbol\Lambda\mathbf{X}^{-1}$: in the eigenvector basis, $\mathbf{A}$ is diagonal. Powers become easy, $\mathbf{A}^k=\mathbf{X}\boldsymbol\Lambda^k\mathbf{X}^{-1}$, and $\mathbf{A}^k\to\mathbf{0}$ exactly when every $|\lambda|<1$. Not every matrix has enough eigenvectors.
> - Symmetric matrices are the best case: real eigenvalues, orthonormal eigenvectors, $\mathbf{S}=\mathbf{Q}\boldsymbol\Lambda\mathbf{Q}^\top=\sum_i\lambda_i\mathbf{q}_i\mathbf{q}_i^\top$, a weighted sum of projections.
> - A symmetric $\mathbf{S}$ is positive definite when its energy $\mathbf{x}^\top\mathbf{S}\mathbf{x}$ is positive, equivalently when all its eigenvalues are positive; its level sets are ellipses with axes along the eigenvectors. $\mathbf{A}^\top\mathbf{A}$ is always positive semidefinite.
> - The trace is the sum of the eigenvalues and the determinant is their product. The Rayleigh quotient $\mathbf{x}^\top\mathbf{S}\mathbf{x}/\mathbf{x}^\top\mathbf{x}$ ranges exactly between $\lambda_{\min}$ and $\lambda_{\max}$.
> - In practice eigenvalues are computed by iteration (power iteration, the QR algorithm), never from the characteristic polynomial.
>
> **Notation in this lecture:** $\mathbf{X}$ is the matrix whose columns are eigenvectors, following Strang. In the machine-learning lectures $\mathbf{X}\in\mathbb{R}^{N\times D}$ is the data matrix; here it appears only once, centred, as $\bar{\mathbf{X}}$ in the section on covariance matrices. Symmetric matrices are written $\mathbf{S}$ and their orthonormal eigenvector matrices $\mathbf{Q}$. Iterates are $\mathbf{z}_k$ (power iteration) and $\mathbf{A}^{(k)}$ (QR algorithm), so $\mathbf{x}_i$ always denotes an eigenvector.

## Motivation

A matrix acting on a vector usually does something complicated: it rotates, shears and stretches all at once. But for almost every square matrix there are a few special directions along which it acts as simply as possible, by plain scaling. Find those directions, use them as coordinates, and the matrix becomes a list of numbers. That one idea explains a remarkable amount:
- the long-run behaviour of anything that is applied repeatedly, from a Markov chain to a recurrent network to the Bellman equation of reinforcement learning;
- which critical points of a function are minima, maxima or saddles (the Hessian's eigenvalues);
- how fast gradient descent converges (the condition number $\lambda_{\max}/\lambda_{\min}$);
- the principal components of a dataset (the covariance matrix's eigenvectors), and spectral clustering and PageRank.

This lecture follows the fourth video of Strang's *A 2020 Vision of Linear Algebra*,[^strang-video]<sup>,</sup>[^strang-ocw] with fuller proofs from his textbook[^strang2023] and examples of our own. It is the series' complete treatment of eigenvalues; the machine-learning lectures link here.

## Eigenvectors keep their direction

### The equation Ax = λx

Let $\mathbf{A}$ be a square $n\times n$ matrix. A non-zero vector $\mathbf{x}$ is an **eigenvector** of $\mathbf{A}$, with **eigenvalue** $\lambda$, if

$$
\mathbf{A}\mathbf{x}=\lambda\mathbf{x}.
\tag{1}
$$

Multiplying by $\mathbf{A}$ does not turn $\mathbf{x}$; it only stretches it by $\lambda$ (or shrinks it, if $|\lambda|<1$, or flips it, if $\lambda<0$). The zero vector is excluded because $\mathbf{A}\mathbf{0}=\lambda\mathbf{0}$ for every $\lambda$. Any non-zero multiple of an eigenvector is again an eigenvector, so what really matters is the eigenvector's **line**.

Figure 1(a) shows this for

$$
\mathbf{A}=\begin{bmatrix}4&1\\2&3\end{bmatrix}.
$$

Twelve unit vectors (grey) are sent to twelve images (pink), and almost every image points in a new direction. The two exceptions are $\mathbf{x}_1=(1,1)$, which comes out five times longer, and $\mathbf{x}_2=(1,-2)$, which comes out twice as long:

$$
\begin{aligned}
\begin{bmatrix}4&1\\2&3\end{bmatrix}\begin{bmatrix}1\\1\end{bmatrix}&=\begin{bmatrix}5\\5\end{bmatrix}=5\begin{bmatrix}1\\1\end{bmatrix},\\[4pt]
\begin{bmatrix}4&1\\2&3\end{bmatrix}\begin{bmatrix}1\\-2\end{bmatrix}&=\begin{bmatrix}2\\-4\end{bmatrix}=2\begin{bmatrix}1\\-2\end{bmatrix}.
\end{aligned}
$$

![[Probability and Linear Algebra/images/eigenvalues-and-eigenvectors-geometry.svg]]
*Figure 1. Unit vectors (grey) and their images (pink) under two matrices; the dashed unit circle is mapped to the solid ellipse. (a) For $\mathbf{A}$ only the two eigenvector lines (dotted) keep their direction, stretched by 5 and 2. They are not perpendicular. (b) For the symmetric $\mathbf{S}$ the eigenvectors $(1,1)$ and $(1,-1)$ are perpendicular, and they are exactly the axes of the ellipse.*

Every other vector is a combination of the two eigenvectors, and $\mathbf{A}$ acts on each piece separately. That is the whole trick: if $\mathbf{v}=c_1\mathbf{x}_1+c_2\mathbf{x}_2$, then $\mathbf{A}\mathbf{v}=5c_1\mathbf{x}_1+2c_2\mathbf{x}_2$. The $\mathbf{x}_1$ part grows faster, which is why the pink images lean towards the $(1,1)$ line.

> [!tip] Intuition: eigenvectors are the axes along which a matrix is simple
> Think of $\mathbf{A}$ as a machine that deforms the plane. Most arrows get both stretched and turned. Eigenvectors are the arrows the machine does not turn. If you describe every vector in the coordinates given by these arrows, the machine only multiplies each coordinate by its own number. 3Blue1Brown's animation of exactly this picture is worth watching.[^3b1b-eigen]

### The characteristic polynomial

How do we find eigenvalues without guessing? Rewrite (1) as $(\mathbf{A}-\lambda\mathbf{I})\mathbf{x}=\mathbf{0}$. A non-zero solution $\mathbf{x}$ exists exactly when $\mathbf{A}-\lambda\mathbf{I}$ has a non-trivial [[Elimination and the Nullspace#The nullspace|nullspace]], that is, when it is singular:

$$
\det(\mathbf{A}-\lambda\mathbf{I})=0.
\tag{2}
$$

This is the **characteristic equation**. Expanding the determinant gives a polynomial of degree $n$ in $\lambda$, the **characteristic polynomial**. Then:
- by the fundamental theorem of algebra it has exactly $n$ roots in $\mathbb{C}$, **counted with multiplicity**, so every $n\times n$ matrix has $n$ eigenvalues in that sense;
- for a real matrix the coefficients are real, so complex eigenvalues come in conjugate pairs $\lambda$ and $\bar\lambda$;
- once $\lambda$ is known, its eigenvectors are the non-zero vectors in the nullspace $N(\mathbf{A}-\lambda\mathbf{I})$, found by elimination. This nullspace is the **eigenspace** of $\lambda$.

For a $2\times2$ matrix the characteristic polynomial is short enough to remember:

$$
\begin{aligned}
\det\begin{bmatrix}a-\lambda&b\\c&d-\lambda\end{bmatrix}&=\lambda^2-(a+d)\,\lambda+(ad-bc)\\
&=\lambda^2-\operatorname{tr}(\mathbf{A})\,\lambda+\det(\mathbf{A}).
\end{aligned}
\tag{3}
$$

### Three 2×2 examples by hand

> [!example] Example 1: distinct real eigenvalues
> For $\mathbf{A}=\begin{bmatrix}4&1\\2&3\end{bmatrix}$, the trace is $7$ and the determinant is $12-2=10$, so (3) reads
>
> $$
> \lambda^2-7\lambda+10=(\lambda-5)(\lambda-2)=0,\qquad\lambda_1=5,\ \lambda_2=2.
> $$
>
> For $\lambda_1=5$: $\mathbf{A}-5\mathbf{I}=\begin{bmatrix}-1&1\\2&-2\end{bmatrix}$. Its rows are multiples of each other (as they must be, since the matrix is singular), and $-x+y=0$ gives $\mathbf{x}_1=(1,1)$.
>
> For $\lambda_2=2$: $\mathbf{A}-2\mathbf{I}=\begin{bmatrix}2&1\\2&1\end{bmatrix}$, and $2x+y=0$ gives $\mathbf{x}_2=(1,-2)$.

> [!example] Example 2: a repeated eigenvalue
> Compare $3\mathbf{I}$ with $\mathbf{J}=\begin{bmatrix}3&1\\0&3\end{bmatrix}$. Both have characteristic polynomial $(\lambda-3)^2$, so $\lambda=3$ is a **double** root for both.
> - For $3\mathbf{I}$, $3\mathbf{I}-3\mathbf{I}=\mathbf{0}$ and every vector is an eigenvector: the eigenspace is the whole plane.
> - For $\mathbf{J}$, $\mathbf{J}-3\mathbf{I}=\begin{bmatrix}0&1\\0&0\end{bmatrix}$ has rank 1, so its nullspace is only the line through $(1,0)$. $\mathbf{J}$ has **one** independent eigenvector, not two.
>
> The number of times $\lambda$ repeats as a root is its **algebraic multiplicity** (2 for both). The dimension of its eigenspace is its **geometric multiplicity** (2 for $3\mathbf{I}$, 1 for $\mathbf{J}$). The geometric multiplicity is never larger than the algebraic one, and when it is smaller, eigenvectors are missing. We meet the consequences [[#Not every matrix is diagonalizable|below]].

> [!example] Example 3: a rotation has complex eigenvalues
> The $90°$ rotation $\mathbf{G}=\begin{bmatrix}0&-1\\1&0\end{bmatrix}$ turns *every* non-zero vector, so it can have no real eigenvector. Its trace is 0 and its determinant is 1, so $\lambda^2+1=0$ and $\lambda=\pm i$. Allowing complex vectors, $\mathbf{x}=(1,-i)$ works:
>
> $$
> \mathbf{G}\begin{bmatrix}1\\-i\end{bmatrix}=\begin{bmatrix}i\\1\end{bmatrix}=i\begin{bmatrix}1\\-i\end{bmatrix},
> $$
>
> and its conjugate $(1,i)$ has eigenvalue $-i$. A rotation by angle $\theta$ has eigenvalues $\cos\theta\pm i\sin\theta=e^{\pm i\theta}$: the angle lives in the eigenvalues' phase, and $|\lambda|=1$ because rotations preserve length (as shown for every orthogonal matrix in [[Orthogonality and Least Squares#Orthonormal vectors and orthogonal matrices|Orthogonality and Least Squares]]).

### What eigenvalues survive

If $\mathbf{A}\mathbf{x}=\lambda\mathbf{x}$, then several related matrices keep the **same eigenvector** $\mathbf{x}$, with predictable eigenvalues:
- $\mathbf{A}^2\mathbf{x}=\mathbf{A}(\lambda\mathbf{x})=\lambda\mathbf{A}\mathbf{x}=\lambda^2\mathbf{x}$, and in general $\mathbf{A}^k\mathbf{x}=\lambda^k\mathbf{x}$;
- if $\mathbf{A}$ is invertible, $\mathbf{A}^{-1}\mathbf{x}=\lambda^{-1}\mathbf{x}$ (multiply (1) by $\mathbf{A}^{-1}/\lambda$). So $\mathbf{A}$ is invertible exactly when $0$ is not an eigenvalue;
- $(\mathbf{A}+\sigma\mathbf{I})\mathbf{x}=(\lambda+\sigma)\mathbf{x}$: a **shift** moves every eigenvalue by $\sigma$;
- any polynomial $g(\mathbf{A})$ has eigenvalue $g(\lambda)$.

For Example 1, $\mathbf{A}^{-1}$ has eigenvalues $\tfrac15$ and $\tfrac12$, and $\mathbf{A}^3$ has $125$ and $8$.

What does **not** survive is sums and products of different matrices. The eigenvalues of $\mathbf{A}+\mathbf{B}$ or $\mathbf{A}\mathbf{B}$ are generally not sums or products of eigenvalues, because $\mathbf{A}$ and $\mathbf{B}$ usually have different eigenvectors.

One more shortcut: the eigenvalues of a **triangular** matrix are its diagonal entries, because $\det(\mathbf{A}-\lambda\mathbf{I})$ is then the product of the diagonal entries $a_{ii}-\lambda$. That is why $\mathbf{J}$ in Example 2 showed its eigenvalue on the diagonal, and it is the idea behind the [[#Power iteration and the QR algorithm|QR algorithm]].

## Diagonalization A = XΛX⁻¹

### Eigenvectors as a basis

Suppose $\mathbf{A}$ has $n$ linearly independent eigenvectors $\mathbf{x}_1,\dots,\mathbf{x}_n$. Put them as the columns of $\mathbf{X}$ and the eigenvalues on the diagonal of $\boldsymbol\Lambda$. The $n$ equations $\mathbf{A}\mathbf{x}_i=\lambda_i\mathbf{x}_i$, one column at a time, are a single matrix equation:

$$
\mathbf{A}\mathbf{X}=\mathbf{A}\begin{bmatrix}\mathbf{x}_1&\cdots&\mathbf{x}_n\end{bmatrix}=\begin{bmatrix}\lambda_1\mathbf{x}_1&\cdots&\lambda_n\mathbf{x}_n\end{bmatrix}=\mathbf{X}\boldsymbol\Lambda.
$$

Independent columns make $\mathbf{X}$ invertible, so

$$
\mathbf{A}=\mathbf{X}\boldsymbol\Lambda\mathbf{X}^{-1}
\qquad\text{and}\qquad
\boldsymbol\Lambda=\mathbf{X}^{-1}\mathbf{A}\mathbf{X}.
\tag{4}
$$

Read (4) from right to left as three steps applied to a vector $\mathbf{v}$:
1. $\mathbf{X}^{-1}\mathbf{v}$ finds the coordinates $\mathbf{c}$ of $\mathbf{v}$ in the eigenvector basis ($\mathbf{v}=\mathbf{X}\mathbf{c}$);
2. $\boldsymbol\Lambda$ multiplies each coordinate by its eigenvalue;
3. $\mathbf{X}$ converts back to standard coordinates.

In the eigenvector basis, $\mathbf{A}$ is diagonal. This is a **change of basis**, and 3Blue1Brown's chapter on it makes the three steps visual.[^3b1b-basis]

> [!example] Example 1, diagonalized
> With $\mathbf{X}=\begin{bmatrix}1&1\\1&-2\end{bmatrix}$ and $\det\mathbf{X}=-3$,
>
> $$
> \mathbf{X}^{-1}=\frac13\begin{bmatrix}2&1\\1&-1\end{bmatrix},
> \qquad
> \mathbf{A}=\begin{bmatrix}1&1\\1&-2\end{bmatrix}\begin{bmatrix}5&0\\0&2\end{bmatrix}\frac13\begin{bmatrix}2&1\\1&-1\end{bmatrix}.
> $$
>
> Multiplying out gives back $\begin{bmatrix}4&1\\2&3\end{bmatrix}$. The eigenvectors are not orthogonal here, so $\mathbf{X}^{-1}$ is not simply $\mathbf{X}^\top$; that luxury is reserved for [[#Symmetric matrices: S = QΛQᵀ|symmetric matrices]].

### Similar matrices

Equation (4) says that $\mathbf{A}$ and $\boldsymbol\Lambda$ are **similar**. In general, two square matrices are similar if

$$
\mathbf{B}=\mathbf{M}^{-1}\mathbf{A}\mathbf{M}\qquad\text{for some invertible }\mathbf{M}.
$$

Similar matrices represent the same linear map in two different bases, with $\mathbf{M}$ converting between them. They therefore share everything that does not depend on the basis:
- the **same characteristic polynomial**, hence the same eigenvalues with the same multiplicities, because $\det(\mathbf{M}^{-1}\mathbf{A}\mathbf{M}-\lambda\mathbf{I})=\det\big(\mathbf{M}^{-1}(\mathbf{A}-\lambda\mathbf{I})\mathbf{M}\big)=\det(\mathbf{A}-\lambda\mathbf{I})$;
- the same trace, determinant and rank.

Their eigenvectors differ, but in a controlled way: if $\mathbf{B}\mathbf{y}=\lambda\mathbf{y}$ then $\mathbf{A}(\mathbf{M}\mathbf{y})=\lambda(\mathbf{M}\mathbf{y})$.

> [!warning] Same eigenvalues does not mean similar
> The converse fails. $\mathbf{J}=\begin{bmatrix}3&1\\0&3\end{bmatrix}$ and $3\mathbf{I}$ have the same eigenvalues, but $\mathbf{M}^{-1}(3\mathbf{I})\mathbf{M}=3\mathbf{I}$ for every $\mathbf{M}$, so nothing other than $3\mathbf{I}$ itself is similar to $3\mathbf{I}$. The eigenvalues are shared by similar matrices, but they do not determine the similarity class; the eigenvector structure matters too.

Similarity is the right tool for computing eigenvalues. If we can choose $\mathbf{M}$ so that $\mathbf{B}=\mathbf{M}^{-1}\mathbf{A}\mathbf{M}$ is **triangular**, the eigenvalues can be read off its diagonal. **Schur's theorem** says this is always possible with a unitary $\mathbf{M}$ (an orthogonal matrix, allowing complex entries): every square matrix is $\mathbf{A}=\mathbf{U}\mathbf{T}\mathbf{U}^{*}$ with $\mathbf{T}$ upper triangular.[^hornjohnson2012] Diagonalization asks for more, a *diagonal* $\mathbf{B}$, and that is not always possible.

### Powers of a matrix

Diagonalization makes repeated multiplication easy. The inner factors cancel:

$$
\mathbf{A}^2=\mathbf{X}\boldsymbol\Lambda\mathbf{X}^{-1}\mathbf{X}\boldsymbol\Lambda\mathbf{X}^{-1}=\mathbf{X}\boldsymbol\Lambda^2\mathbf{X}^{-1},
\qquad
\mathbf{A}^k=\mathbf{X}\boldsymbol\Lambda^k\mathbf{X}^{-1}.
\tag{5}
$$

Raising a diagonal matrix to a power just raises each entry. In terms of a starting vector $\mathbf{v}_0=c_1\mathbf{x}_1+\cdots+c_n\mathbf{x}_n$,

$$
\mathbf{A}^k\mathbf{v}_0=c_1\lambda_1^k\mathbf{x}_1+c_2\lambda_2^k\mathbf{x}_2+\cdots+c_n\lambda_n^k\mathbf{x}_n.
\tag{6}
$$

Each eigen-component evolves independently, growing or decaying like $\lambda_i^k$. Three consequences follow at once:
- if every $|\lambda_i|<1$, every term dies and $\mathbf{A}^k\to\mathbf{0}$;
- if some $|\lambda_i|>1$, that term explodes for any $\mathbf{v}_0$ with $c_i\neq0$;
- for large $k$, the term with the largest $|\lambda_i|$ dominates, and $\mathbf{A}^k\mathbf{v}_0$ lines up with its eigenvector. This is [[#Power iteration and the QR algorithm|power iteration]].

For Example 1, $\mathbf{A}^3=\mathbf{X}\operatorname{diag}(125,8)\mathbf{X}^{-1}=\begin{bmatrix}86&39\\78&47\end{bmatrix}$.

> [!note] Theorem ($\mathbf{A}^k\to\mathbf{0}$)
> For any square matrix, $\mathbf{A}^k\to\mathbf{0}$ as $k\to\infty$ if and only if every eigenvalue satisfies $|\lambda|<1$, that is, the **spectral radius** $\rho(\mathbf{A})=\max_i|\lambda_i|$ is less than 1.

For diagonalizable matrices this is (5). It stays true for matrices with missing eigenvectors, but there powers pick up polynomial factors. For $\mathbf{J}$ in Example 2 one can check that

$$
\mathbf{J}^k=\begin{bmatrix}3^k&k\,3^{k-1}\\0&3^k\end{bmatrix}.
$$

Replace 3 by $0.9$: the off-diagonal entry $k\,(0.9)^{k-1}$ first **grows** (to about 3.9 at $k=9$) before the geometric decay wins. Figure 2(b) shows this transient. The eigenvalues decide where the powers end up, but for non-diagonalizable (and, more generally, non-symmetric) matrices they do not tell the whole story along the way.

> [!proof]- Proof for any square matrix
> Use the Schur form $\mathbf{A}=\mathbf{U}\mathbf{T}\mathbf{U}^{*}$, so $\mathbf{A}^k=\mathbf{U}\mathbf{T}^k\mathbf{U}^{*}$ and it is enough to study the triangular $\mathbf{T}$, whose diagonal holds the eigenvalues. If some $|\lambda|\ge1$, take its eigenvector $\mathbf{x}$: $\lVert\mathbf{A}^k\mathbf{x}\rVert=|\lambda|^k\lVert\mathbf{x}\rVert\not\to0$. Conversely, suppose $\rho=\rho(\mathbf{A})<1$. Write $\mathbf{T}=\mathbf{D}+\mathbf{K}$ with $\mathbf{D}$ diagonal and $\mathbf{K}$ strictly upper triangular, so $\mathbf{K}^n=\mathbf{0}$. Expanding $(\mathbf{D}+\mathbf{K})^k$, each product contains at most $n-1$ factors of $\mathbf{K}$, and there are at most $\binom{k}{j}\le k^j$ products with $j$ such factors. So every entry of $\mathbf{T}^k$ is bounded by $C\,k^{n-1}\rho^{\,k-n+1}$ for a constant $C$, and a polynomial times $\rho^k$ tends to 0.

![[Probability and Linear Algebra/images/eigenvalues-and-eigenvectors-powers.svg]]
*Figure 2. (a) The entries of $\mathbf{P}^k$ for the two-state chain below. Both rows approach $(0.6,\,0.4)$ and the gap halves at every step, because the second eigenvalue is $0.5$. (b) Norms of $\mathbf{A}^k$ on a log scale. Spectral radius $0.9$ decays geometrically; spectral radius $1.05$ explodes; the defective matrix with eigenvalue $0.9$ repeated first grows to about 3.9 near $k=9$ and only then decays.*

### Example: a two-state Markov chain

A machine is either up (state 1) or down (state 2). An up machine stays up the next day with probability $0.8$; a down machine is repaired with probability $0.3$. As in [[Markov Chains#The Markov property|Markov Chains]], the rows of the transition matrix hold the probabilities of moving from each state:

$$
\mathbf{P}=\begin{bmatrix}0.8&0.2\\0.3&0.7\end{bmatrix}.
$$

The $k$-step transition probabilities are the entries of $\mathbf{P}^k$. Every row of $\mathbf{P}$ sums to 1, so $\mathbf{P}\mathbf{1}=\mathbf{1}$: the all-ones vector $(1,1)$ is an eigenvector with $\lambda_1=1$. The trace is $1.5$, so $\lambda_2=0.5$, and $\mathbf{P}-0.5\mathbf{I}=\begin{bmatrix}0.3&0.2\\0.3&0.2\end{bmatrix}$ gives the eigenvector $(2,-3)$. Then

$$
\mathbf{X}=\begin{bmatrix}1&2\\1&-3\end{bmatrix},\qquad
\mathbf{X}^{-1}=\begin{bmatrix}0.6&0.4\\0.2&-0.2\end{bmatrix},
$$

and (5) gives every power in closed form:

$$
\begin{aligned}
\mathbf{P}^k&=\mathbf{X}\begin{bmatrix}1&0\\0&0.5^k\end{bmatrix}\mathbf{X}^{-1}\\
&=\begin{bmatrix}0.6&0.4\\0.6&0.4\end{bmatrix}+0.5^k\begin{bmatrix}0.4&-0.4\\-0.6&0.6\end{bmatrix}.
\end{aligned}
$$

The eigenvalue $1$ produces the permanent part, whose identical rows $(0.6,0.4)$ are the **stationary distribution**: in the long run the machine is up 60% of the days, whatever its state today. The eigenvalue $0.5$ produces the transient part, which halves every day (Figure 2a). By $k=10$ the rows are $(0.6004,\,0.3996)$ and $(0.5994,\,0.4006)$.

Two lessons generalize. The stationary distribution $\boldsymbol\pi$, which satisfies $\boldsymbol\pi\mathbf{P}=\boldsymbol\pi$, is an eigenvector of $\mathbf{P}^\top$ with eigenvalue 1 (a *left* eigenvector of $\mathbf{P}$), as in [[Markov Chains#The stationary distribution|Markov Chains]]. And the speed of convergence is set by the second-largest $|\lambda|$; this is the [[Markov Chains#Aside: mixing and the spectral gap|spectral gap]].

> [!example] Example: Fibonacci numbers
> The rule $F_{k+1}=F_k+F_{k-1}$ is a matrix power in disguise: $\begin{bmatrix}F_{k+1}\\F_k\end{bmatrix}=\begin{bmatrix}1&1\\1&0\end{bmatrix}\begin{bmatrix}F_k\\F_{k-1}\end{bmatrix}$. The matrix has trace 1 and determinant $-1$, so $\lambda^2-\lambda-1=0$ and
>
> $$
> \varphi=\frac{1+\sqrt5}{2}\approx1.618,\qquad\psi=\frac{1-\sqrt5}{2}\approx-0.618.
> $$
>
> Formula (6), with $F_0=0$ and $F_1=1$, gives $F_k=(\varphi^k-\psi^k)/\sqrt5$. Since $|\psi|<1$ the second term fades, so $F_k$ is the nearest integer to $\varphi^k/\sqrt5$, and successive ratios approach the golden ratio $\varphi$. For example $\varphi^{10}/\sqrt5\approx55.004$ and $F_{10}=55$.

### Differential equations

The same diagonalization solves the linear system of differential equations $\frac{d\mathbf{u}}{dt}=\mathbf{A}\mathbf{u}$. In eigen-coordinates the equations decouple into $n$ scalar ones, $\frac{dc_i}{dt}=\lambda_ic_i$, each solved by $c_i(t)=c_i(0)\,e^{\lambda_it}$. Back in standard coordinates,

$$
\mathbf{u}(t)=e^{\mathbf{A}t}\mathbf{u}(0)=\mathbf{X}e^{\boldsymbol\Lambda t}\mathbf{X}^{-1}\mathbf{u}(0)=\sum_ic_i(0)\,e^{\lambda_it}\,\mathbf{x}_i.
$$

Powers $\lambda^k$ are replaced by exponentials $e^{\lambda t}$, and the stability condition changes accordingly: $\mathbf{u}(t)\to\mathbf{0}$ for every start exactly when every eigenvalue has **negative real part**. For instance, $\mathbf{A}=\begin{bmatrix}-2&1\\1&-2\end{bmatrix}$ has eigenvalues $-1$ and $-3$ with eigenvectors $(1,1)$ and $(1,-1)$, so starting from $(1,0)$,

$$
\mathbf{u}(t)=\tfrac12e^{-t}\begin{bmatrix}1\\1\end{bmatrix}+\tfrac12e^{-3t}\begin{bmatrix}1\\-1\end{bmatrix}.
$$

The fast mode ($e^{-3t}$) dies first and the solution settles along the slow eigenvector. The generator matrices of [[Markov Chains#Transition matrices and the Kolmogorov equations|continuous-time Markov chains]] are an instance, with the exponential of the generator matrix as the transition matrix.

### Not every matrix is diagonalizable

Diagonalization needs $n$ independent eigenvectors, and Example 2 showed that a repeated eigenvalue can have too few. A matrix without a full set of eigenvectors is called **defective**. Two facts tell us when to worry:
- **Distinct eigenvalues are enough.** Eigenvectors belonging to different eigenvalues are always independent, so a matrix with $n$ distinct eigenvalues is diagonalizable. Trouble can only come from repeated eigenvalues.
- **Repeated eigenvalues are not automatically a problem.** $3\mathbf{I}$ has a double eigenvalue and is diagonal already. The test is whether each eigenvalue's geometric multiplicity equals its algebraic multiplicity.

> [!proof]- Proof that eigenvectors for distinct eigenvalues are independent
> Suppose $\mathbf{x}_1,\dots,\mathbf{x}_k$ are eigenvectors with distinct eigenvalues, and suppose they are dependent. Take a dependence $\sum_ic_i\mathbf{x}_i=\mathbf{0}$ with the fewest non-zero coefficients; at least two are non-zero, say $c_1$ and $c_2$. Apply $\mathbf{A}$ to get $\sum_ic_i\lambda_i\mathbf{x}_i=\mathbf{0}$, and subtract $\lambda_1$ times the original: $\sum_ic_i(\lambda_i-\lambda_1)\mathbf{x}_i=\mathbf{0}$. The $\mathbf{x}_1$ term has disappeared, while the $\mathbf{x}_2$ term survives because $\lambda_2\ne\lambda_1$. This is a shorter non-trivial dependence, which contradicts minimality.

> [!warning] Diagonalizable and invertible are different questions
> Diagonalizability is about having enough **eigenvectors**. Invertibility is about having **no zero eigenvalue**. The two are unrelated:
> - $\mathbf{J}=\begin{bmatrix}3&1\\0&3\end{bmatrix}$ is invertible but not diagonalizable;
> - $\operatorname{diag}(1,0)$ is singular but already diagonal.
>
> A related subtlety concerns the rank. For a diagonalizable matrix of rank $r$, exactly $r$ eigenvalues (with multiplicity) are non-zero, because the rank of $\mathbf{X}\boldsymbol\Lambda\mathbf{X}^{-1}$ is the rank of $\boldsymbol\Lambda$. For a defective matrix even this fails: $\begin{bmatrix}0&1\\0&0\end{bmatrix}$ has rank 1, yet both its eigenvalues are 0.

What replaces diagonalization for defective matrices is the **Jordan form**, block-diagonal with blocks like $\mathbf{J}$. It is fragile (an arbitrarily small perturbation makes the eigenvalues distinct and the matrix diagonalizable), so numerical software uses the Schur form instead.[^hornjohnson2012] In machine learning the matrices that matter most are usually symmetric, which avoids the issue entirely.

## Symmetric matrices: S = QΛQᵀ

Symmetric matrices, $\mathbf{S}=\mathbf{S}^\top$, are everywhere: covariance matrices, Hessians, Gram and kernel matrices, graph Laplacians, $\mathbf{A}^\top\mathbf{A}$. They are also the best-behaved case of the eigenvalue problem. Figure 1(b) shows the symmetric

$$
\mathbf{S}=\begin{bmatrix}2&1\\1&2\end{bmatrix},
$$

with trace 4 and determinant 3, so $\lambda^2-4\lambda+3=0$, $\lambda_1=3$ with eigenvector $(1,1)$ and $\lambda_2=1$ with eigenvector $(1,-1)$. Unlike in Figure 1(a), the eigenvectors are **perpendicular**. That is no accident.

### Real eigenvalues

> [!note] Theorem
> A real symmetric matrix has only real eigenvalues, and its eigenvectors can be chosen real.

> [!proof]- Proof
> Let $\mathbf{S}\mathbf{x}=\lambda\mathbf{x}$ with $\mathbf{x}\ne\mathbf{0}$, allowing complex $\lambda$ and $\mathbf{x}$. Write $\bar{\mathbf{x}}$ for the complex conjugate. Since $\mathbf{S}$ is real, conjugating gives $\mathbf{S}\bar{\mathbf{x}}=\bar\lambda\bar{\mathbf{x}}$. Now compute the number $\bar{\mathbf{x}}^\top\mathbf{S}\mathbf{x}$ in two ways. Using $\mathbf{S}\mathbf{x}=\lambda\mathbf{x}$, it equals $\lambda\,\bar{\mathbf{x}}^\top\mathbf{x}$. Using $\mathbf{S}^\top=\mathbf{S}$, it equals $(\mathbf{S}\bar{\mathbf{x}})^\top\mathbf{x}=\bar\lambda\,\bar{\mathbf{x}}^\top\mathbf{x}$. And $\bar{\mathbf{x}}^\top\mathbf{x}=\sum_i|x_i|^2>0$, so $\lambda=\bar\lambda$: $\lambda$ is real. Then $\mathbf{S}-\lambda\mathbf{I}$ is a real singular matrix, and elimination finds a real vector in its nullspace.

### Orthogonal eigenvectors

> [!note] Theorem
> Eigenvectors of a symmetric matrix that belong to **different** eigenvalues are orthogonal.

The proof is short enough to give in full, and it uses every assumption once. Let $\mathbf{S}\mathbf{x}=\lambda\mathbf{x}$ and $\mathbf{S}\mathbf{y}=\beta\mathbf{y}$ with $\lambda\ne\beta$. Compute $\mathbf{x}^\top\mathbf{S}\mathbf{y}$ in two ways:

$$
\begin{aligned}
\mathbf{x}^\top\mathbf{S}\mathbf{y}&=\mathbf{x}^\top(\beta\mathbf{y})=\beta\,\mathbf{x}^\top\mathbf{y},\\
\mathbf{x}^\top\mathbf{S}\mathbf{y}&=(\mathbf{S}^\top\mathbf{x})^\top\mathbf{y}=(\mathbf{S}\mathbf{x})^\top\mathbf{y}=\lambda\,\mathbf{x}^\top\mathbf{y}.
\end{aligned}
$$

Subtracting, $(\lambda-\beta)\,\mathbf{x}^\top\mathbf{y}=0$, and since $\lambda\ne\beta$, $\mathbf{x}^\top\mathbf{y}=0$. The symmetry $\mathbf{S}^\top=\mathbf{S}$ is what lets $\mathbf{S}$ move from $\mathbf{y}$ onto $\mathbf{x}$; for the non-symmetric $\mathbf{A}$ of Example 1 this step fails, and the eigenvectors $(1,1)$ and $(1,-2)$ are indeed not orthogonal.

### The spectral theorem

Within an eigenspace of a repeated eigenvalue, any basis is made of eigenvectors, so Gram–Schmidt can make it orthonormal. Combined with the two theorems above, and with the fact (proved below) that symmetric matrices are never defective, this gives the central result.

> [!note] Spectral theorem
> Every real symmetric matrix can be written as
>
> $$
> \mathbf{S}=\mathbf{Q}\boldsymbol\Lambda\mathbf{Q}^\top,\qquad\mathbf{Q}^\top\mathbf{Q}=\mathbf{I},
> $$
>
> with real eigenvalues on the diagonal of $\boldsymbol\Lambda$ and orthonormal eigenvectors as the columns of $\mathbf{Q}$. Conversely, every matrix of this form is symmetric.

Compared with the general $\mathbf{X}\boldsymbol\Lambda\mathbf{X}^{-1}$, the inverse has become a transpose, $\mathbf{Q}^{-1}=\mathbf{Q}^\top$, and the change of basis is a rotation (possibly with a reflection), which preserves lengths and angles. Geometrically, $\mathbf{S}$ rotates into its eigenvector frame, stretches each axis by $\lambda_i$, and rotates back. That is why the unit circle in Figure 1(b) becomes an ellipse whose axes are exactly the eigenvectors.

> [!proof]- Proof via the Schur form
> By Schur's theorem, $\mathbf{S}=\mathbf{U}\mathbf{T}\mathbf{U}^{*}$ with $\mathbf{U}$ unitary and $\mathbf{T}$ upper triangular. Since the eigenvalues of $\mathbf{S}$ are real, the construction can be carried out with real orthogonal $\mathbf{U}$. Then $\mathbf{T}=\mathbf{U}^\top\mathbf{S}\mathbf{U}$ is symmetric, because $\mathbf{T}^\top=\mathbf{U}^\top\mathbf{S}^\top\mathbf{U}=\mathbf{T}$. A triangular matrix that is also symmetric is diagonal. So $\mathbf{T}=\boldsymbol\Lambda$ and $\mathbf{U}=\mathbf{Q}$. In particular, a symmetric matrix always has $n$ orthonormal eigenvectors, even when eigenvalues repeat.

### A sum of rank-one projections

Multiplying out $\mathbf{Q}\boldsymbol\Lambda\mathbf{Q}^\top$ column times row, as in [[Column Space and Rank#Matrix multiplication four ways|Column Space and Rank]], gives the **spectral decomposition**

$$
\mathbf{S}=\lambda_1\mathbf{q}_1\mathbf{q}_1^\top+\lambda_2\mathbf{q}_2\mathbf{q}_2^\top+\cdots+\lambda_n\mathbf{q}_n\mathbf{q}_n^\top.
\tag{7}
$$

Each $\mathbf{q}_i\mathbf{q}_i^\top$ is a rank-one matrix: the [[Orthogonality and Least Squares#Projections|projection]] onto the line through $\mathbf{q}_i$. So a symmetric matrix projects a vector onto each eigenvector, scales each projection by its eigenvalue, and adds them up. Orthonormality makes the check immediate: $\mathbf{S}\mathbf{q}_j=\sum_i\lambda_i\mathbf{q}_i(\mathbf{q}_i^\top\mathbf{q}_j)=\lambda_j\mathbf{q}_j$, since every cross term vanishes.

> [!example] Example: $\mathbf{S}$ as two projections
> With $\mathbf{q}_1=(1,1)/\sqrt2$ and $\mathbf{q}_2=(1,-1)/\sqrt2$,
>
> $$
> \begin{aligned}
> \mathbf{S}&=3\cdot\frac12\begin{bmatrix}1&1\\1&1\end{bmatrix}+1\cdot\frac12\begin{bmatrix}1&-1\\-1&1\end{bmatrix}\\
> &=\begin{bmatrix}1.5&1.5\\1.5&1.5\end{bmatrix}+\begin{bmatrix}0.5&-0.5\\-0.5&0.5\end{bmatrix}=\begin{bmatrix}2&1\\1&2\end{bmatrix}.
> \end{aligned}
> $$

Zero eigenvalues contribute nothing to (7), so a symmetric matrix of rank $r$ is a sum of exactly $r$ rank-one pieces, and the *thin* form keeps only those: $\mathbf{Q}_r\in\mathbb{R}^{n\times r}$ with orthonormal columns and $\mathbf{S}=\mathbf{Q}_r\boldsymbol\Lambda_r\mathbf{Q}_r^\top$. The non-zero eigenvectors span the column space, and the zero ones span the nullspace, the two perpendicular halves of [[The Four Fundamental Subspaces#The big picture|the big picture]].

We usually order the eigenvalues $\lambda_1\ge\lambda_2\ge\cdots\ge\lambda_n$. Building $\mathbf{S}$ term by term is a sequence of **rank-one updates**, $\mathbf{S}_k=\mathbf{S}_{k-1}+\lambda_k\mathbf{q}_k\mathbf{q}_k^\top$. Truncating after $k$ terms gives the best rank-$k$ approximation of a positive semidefinite $\mathbf{S}$, the Eckart–Young theorem, proved in [[Matrix Factorization and Dimensionality Reduction#Best low-rank approximation: Eckart–Young|Matrix Factorization]].

> [!warning] Largest eigenvalue or largest magnitude?
> When $\mathbf{S}$ has negative eigenvalues, "the biggest terms" of (7) means the largest $|\lambda_i|$, not the largest $\lambda_i$. A term $-5\,\mathbf{q}\mathbf{q}^\top$ matters more than $+1\,\mathbf{q}'\mathbf{q}'^\top$. For a best rank-$k$ approximation of an indefinite $\mathbf{S}$, keep the $k$ eigenvalues of largest magnitude. For positive semidefinite $\mathbf{S}$ the two orderings agree, which is why the distinction rarely shows up for covariance matrices. The same caveat applies to [[#Power iteration and the QR algorithm|power iteration]], which finds the eigenvalue of largest magnitude.

> [!example]- Code: eigendecompositions in numpy
> `np.linalg.eig` handles any square matrix; `np.linalg.eigh` is for symmetric matrices, and is faster, more accurate, returns real eigenvalues in **ascending** order and orthonormal eigenvectors.
> ```python
> import numpy as np
> A = np.array([[4., 1], [2, 3]])
> lam, X = np.linalg.eig(A)                      # columns of X are unit eigenvectors
> print(lam)                                     # [5. 2.]
> print(np.allclose(A @ X, X @ np.diag(lam)))    # A X = X Lambda
> print(np.allclose(np.linalg.matrix_power(A, 3), X @ np.diag(lam**3) @ np.linalg.inv(X)))
>
> S = np.array([[2., 1], [1, 2]])
> lam, Q = np.linalg.eigh(S)                     # ascending: [1. 3.]
> print(np.allclose(Q.T @ Q, np.eye(2)))         # orthonormal eigenvectors
> print(np.allclose(S, sum(l * np.outer(q, q) for l, q in zip(lam, Q.T))))   # spectral sum
>
> G = np.array([[0., -1], [1, 0]])                # 90-degree rotation
> print(np.linalg.eigvals(G))                    # [0.+1.j 0.-1.j]
> ```

## Positive definite matrices

### The energy test and the eigenvalue test

A symmetric matrix $\mathbf{S}$ defines a quadratic function, its **energy** $\mathbf{x}^\top\mathbf{S}\mathbf{x}=\sum_{i,j}s_{ij}x_ix_j$. $\mathbf{S}$ is
- **positive definite** if $\mathbf{x}^\top\mathbf{S}\mathbf{x}>0$ for every $\mathbf{x}\ne\mathbf{0}$, written $\mathbf{S}\succ0$;
- **positive semidefinite** (PSD) if $\mathbf{x}^\top\mathbf{S}\mathbf{x}\ge0$ for every $\mathbf{x}$, written $\mathbf{S}\succeq0$;
- **indefinite** if the energy takes both signs.

Checking "for every $\mathbf{x}$" directly looks hopeless, but the spectral theorem reduces it to $n$ numbers. Write $\mathbf{x}$ in the eigenvector basis, $\mathbf{x}=\sum_ic_i\mathbf{q}_i$ with $c_i=\mathbf{q}_i^\top\mathbf{x}$. Then (7) gives

$$
\mathbf{x}^\top\mathbf{S}\mathbf{x}=\lambda_1c_1^2+\lambda_2c_2^2+\cdots+\lambda_nc_n^2.
\tag{8}
$$

In eigen-coordinates the energy is a weighted sum of squares, with the eigenvalues as weights. It is positive for every $\mathbf{x}\ne\mathbf{0}$ exactly when every weight is positive.

> [!note] Theorem (tests for positive definiteness)
> For a symmetric $\mathbf{S}$, the following are equivalent:
> 1. **energy test:** $\mathbf{x}^\top\mathbf{S}\mathbf{x}>0$ for every $\mathbf{x}\ne\mathbf{0}$;
> 2. **eigenvalue test:** every eigenvalue of $\mathbf{S}$ is positive;
> 3. **pivot test:** elimination without row exchanges gives positive pivots;
> 4. **determinant test:** the leading $k\times k$ upper-left determinants are positive for $k=1,\dots,n$ (Sylvester's criterion);
> 5. $\mathbf{S}=\mathbf{A}^\top\mathbf{A}$ for some $\mathbf{A}$ with independent columns. Elimination gives the **Cholesky** factorization $\mathbf{S}=\mathbf{L}\mathbf{L}^\top$ with lower triangular $\mathbf{L}$.
>
> Replacing "positive" by "non-negative" in 1, 2 and 5 (and "independent columns" by any $\mathbf{A}$) gives the PSD versions. For PSD, test 4 would need *all* principal minors, not only the leading ones.

Tests 1 and 2 are (8). For $2\times2$ matrices, test 4 reads $a>0$ and $ac-b^2>0$, the familiar test from [[Multivariable Calculus#The Hessian and second-order Taylor|the Hessian section of Multivariable Calculus]].

> [!example] Example: three tests on $\mathbf{S}=\begin{bmatrix}2&1\\1&2\end{bmatrix}$
> - Eigenvalues $3$ and $1$, both positive.
> - Determinants: $2>0$ and $4-1=3>0$.
> - Energy: $2x_1^2+2x_1x_2+2x_2^2=(x_1+x_2)^2+x_1^2+x_2^2$, which is a sum of squares and is 0 only at $\mathbf{x}=\mathbf{0}$. In eigen-coordinates (8) reads $3c_1^2+c_2^2$.
>
> By contrast, $\begin{bmatrix}1&2\\2&1\end{bmatrix}$ has determinant $-3$, eigenvalues $3$ and $-1$, and energy $-2$ at $\mathbf{x}=(1,-1)$: indefinite. And $\begin{bmatrix}1&1\\1&1\end{bmatrix}$ has eigenvalues $2$ and $0$: PSD but singular, with zero energy along $(1,-1)$.

### Level sets are ellipses

Equation (8) also describes the **shape** of the energy. The level set $\mathbf{x}^\top\mathbf{S}\mathbf{x}=1$ is $\sum_i\lambda_ic_i^2=1$: for $\mathbf{S}\succ0$ this is an **ellipse** (an ellipsoid in $n$ dimensions) whose axes point along the eigenvectors, with semi-axis $1/\sqrt{\lambda_i}$ along $\mathbf{q}_i$. Large eigenvalues mean steep directions and short axes. For $\mathbf{S}=\begin{bmatrix}2&1\\1&2\end{bmatrix}$ the semi-axes are $1/\sqrt3\approx0.577$ along $(1,1)$ and $1$ along $(1,-1)$. Figure 3 shows all three cases.

![[Probability and Linear Algebra/images/eigenvalues-and-eigenvectors-quadratic-forms.svg]]
*Figure 3. Contours of the energy $\mathbf{x}^\top\mathbf{S}\mathbf{x}$, with the eigenvectors drawn as arrows. (a) Positive definite: nested ellipses, short along the large eigenvalue. (b) Positive semidefinite: the zero eigenvalue stretches the ellipses into parallel lines, a valley with a flat floor along $(1,-1)$. (c) Indefinite: hyperbolas, uphill along one eigenvector and downhill along the other, a saddle.*

This picture is the bridge to optimization. Near a critical point $\mathbf{x}^*$ of a smooth $f$, the second-order Taylor model is $f(\mathbf{x}^*+\mathbf{h})\approx f(\mathbf{x}^*)+\tfrac12\mathbf{h}^\top\mathbf{H}\mathbf{h}$ with the symmetric Hessian $\mathbf{H}$. So the three panels of Figure 3 are a local minimum ($\mathbf{H}\succ0$), a degenerate case where the second-order test is inconclusive ($\mathbf{H}\succeq0$ singular), and a saddle ($\mathbf{H}$ indefinite). The eigenvalue test is the general-$n$ version of the $2\times2$ determinant test in [[Multivariable Calculus#The Hessian and second-order Taylor|Multivariable Calculus]], and a twice-differentiable function is convex exactly when its Hessian is PSD everywhere. The elongated ellipses of Figure 3(a) are also what makes gradient descent zig-zag: the ratio $\lambda_{\max}/\lambda_{\min}$ sets its speed, as worked out in [[First-Order Optimization#Gradient descent on a quadratic|First-Order Optimization]].

### AᵀA is positive semidefinite

For any $\mathbf{A}\in\mathbb{R}^{m\times n}$, the $n\times n$ matrix $\mathbf{A}^\top\mathbf{A}$ is square and symmetric, since $(\mathbf{A}^\top\mathbf{A})^\top=\mathbf{A}^\top(\mathbf{A}^\top)^\top=\mathbf{A}^\top\mathbf{A}$. Its energy is a squared length:

$$
\mathbf{x}^\top\mathbf{A}^\top\mathbf{A}\mathbf{x}=(\mathbf{A}\mathbf{x})^\top(\mathbf{A}\mathbf{x})=\lVert\mathbf{A}\mathbf{x}\rVert^2\ge0.
$$

So $\mathbf{A}^\top\mathbf{A}\succeq0$ always. It is positive definite exactly when $\mathbf{A}\mathbf{x}=\mathbf{0}$ only for $\mathbf{x}=\mathbf{0}$, that is, when $\mathbf{A}$ has independent columns (rank $n$). Applied to an eigenvector, the same computation gives the eigenvalue directly: if $\mathbf{A}^\top\mathbf{A}\mathbf{x}=\lambda\mathbf{x}$, then $\lambda=\lVert\mathbf{A}\mathbf{x}\rVert^2/\lVert\mathbf{x}\rVert^2\ge0$. This is why the normal equations $\mathbf{A}^\top\mathbf{A}\hat{\mathbf{x}}=\mathbf{A}^\top\mathbf{b}$ of [[Orthogonality and Least Squares#Least squares|least squares]] have a unique solution when the columns are independent.

**$\mathbf{A}\mathbf{A}^\top$ has the same non-zero eigenvalues.** $\mathbf{A}\mathbf{A}^\top$ is $m\times m$, usually a different size, yet the non-zero eigenvalues agree. If $\mathbf{A}^\top\mathbf{A}\mathbf{v}=\lambda\mathbf{v}$ with $\lambda\ne0$, multiply by $\mathbf{A}$:

$$
\mathbf{A}\mathbf{A}^\top(\mathbf{A}\mathbf{v})=\lambda\,(\mathbf{A}\mathbf{v}),
$$

and $\mathbf{A}\mathbf{v}\ne\mathbf{0}$ (otherwise $\lambda\mathbf{v}=\mathbf{A}^\top\mathbf{A}\mathbf{v}=\mathbf{0}$). So $\mathbf{A}\mathbf{v}$ is an eigenvector of $\mathbf{A}\mathbf{A}^\top$ with the same $\lambda$. The leftover eigenvalues of the larger matrix are zeros.

> [!example] Example: $\mathbf{A}^\top\mathbf{A}$ and $\mathbf{A}\mathbf{A}^\top$ by hand
> Take
>
> $$
> \mathbf{A}=\begin{bmatrix}1&2\\0&1\\1&0\end{bmatrix},\qquad
> \mathbf{A}^\top\mathbf{A}=\begin{bmatrix}2&2\\2&5\end{bmatrix},
> $$
>
> $$
> \mathbf{A}\mathbf{A}^\top=\begin{bmatrix}5&2&1\\2&1&0\\1&0&1\end{bmatrix}.
> $$
>
> $\mathbf{A}^\top\mathbf{A}$ has trace 7 and determinant 6, so its eigenvalues are $6$ and $1$, with eigenvectors $(1,2)$ and $(2,-1)$. Multiplying by $\mathbf{A}$ gives $\mathbf{A}(1,2)=(5,2,1)$ and $\mathbf{A}(2,-1)=(0,-1,2)$, which are eigenvectors of $\mathbf{A}\mathbf{A}^\top$ with the same eigenvalues 6 and 1; for instance $\mathbf{A}\mathbf{A}^\top(5,2,1)=(30,12,6)$. The third eigenvalue of $\mathbf{A}\mathbf{A}^\top$ is 0, with eigenvector $(1,-2,-1)$, which spans $N(\mathbf{A}^\top)$. The trace of $\mathbf{A}\mathbf{A}^\top$ is $6+1+0=7$, as it should be.

This pair of matrices is the starting point of the singular value decomposition: the eigenvectors of $\mathbf{A}^\top\mathbf{A}$ and $\mathbf{A}\mathbf{A}^\top$ are the right and left singular vectors of $\mathbf{A}$, and the square roots of the shared eigenvalues ($\sqrt6$ and $1$ here) are its singular values. See [[Matrix Factorization and Dimensionality Reduction#Singular value decomposition (SVD)|the SVD section of Matrix Factorization]].

### Covariance matrices

The most important PSD matrices in statistics are covariance matrices. For a random vector with mean $\boldsymbol\mu$, $\boldsymbol\Sigma=\mathbb{E}[(\mathbf{x}-\boldsymbol\mu)(\mathbf{x}-\boldsymbol\mu)^\top]$, and for any fixed $\mathbf{w}$,

$$
\mathbf{w}^\top\boldsymbol\Sigma\mathbf{w}=\mathbb{E}\big[(\mathbf{w}^\top(\mathbf{x}-\boldsymbol\mu))^2\big]=\operatorname{Var}(\mathbf{w}^\top\mathbf{x})\ge0.
$$

The energy in direction $\mathbf{w}$ is the variance of the data projected onto $\mathbf{w}$. The sample covariance of a centred data matrix is $\bar{\mathbf{X}}^\top\bar{\mathbf{X}}/(N-1)$, an $\mathbf{A}^\top\mathbf{A}$ in disguise. The eigenvectors of $\boldsymbol\Sigma$ are the axes of the data cloud's ellipse, its eigenvalues are the variances along those axes, and a zero eigenvalue means the data lie in a lower-dimensional subspace. The contours of the [[Joint Distributions#The multivariate Gaussian|multivariate Gaussian]] are exactly the ellipses of Figure 3(a) for $\boldsymbol\Sigma^{-1}$, and [[Matrix Factorization and Dimensionality Reduction#Principal component analysis (PCA)|PCA]] keeps the eigenvectors with the largest eigenvalues.

## Trace, determinant and the Rayleigh quotient

### Sum and product of the eigenvalues

Two numbers can be read off any square matrix without solving anything, and they constrain the eigenvalues:

$$
\lambda_1+\lambda_2+\cdots+\lambda_n=\operatorname{tr}(\mathbf{A})=a_{11}+\cdots+a_{nn},
\qquad
\lambda_1\lambda_2\cdots\lambda_n=\det(\mathbf{A}).
\tag{9}
$$

Both follow from the characteristic polynomial. Its roots are the eigenvalues, so $\det(\lambda\mathbf{I}-\mathbf{A})=(\lambda-\lambda_1)\cdots(\lambda-\lambda_n)$. Setting $\lambda=0$ gives $\det(-\mathbf{A})=(-1)^n\lambda_1\cdots\lambda_n$, which is the product rule. The coefficient of $\lambda^{n-1}$ is $-(\lambda_1+\cdots+\lambda_n)$ on the right; on the left, only the product of the diagonal entries $(\lambda-a_{11})\cdots(\lambda-a_{nn})$ contributes to $\lambda^{n-1}$ (every other term of the determinant misses at least two diagonal entries), giving $-(a_{11}+\cdots+a_{nn})$. Both identities hold with multiplicity and with complex eigenvalues.

For Example 1, $5+2=7=4+3$ and $5\cdot2=10=12-2$. These checks catch most hand-calculation errors. For $2\times2$ matrices they also give a quick formula.[^3b1b-trick] The eigenvalues have mean $\lambda_{\mathrm{mean}}=\tfrac12\operatorname{tr}(\mathbf{A})$ and product $p=\det(\mathbf{A})$, so

$$
\lambda=\lambda_{\mathrm{mean}}\pm\sqrt{\lambda_{\mathrm{mean}}^2-p}.
$$

For Example 1, $\lambda_{\mathrm{mean}}=3.5$ and $p=10$ give $3.5\pm\sqrt{2.25}=3.5\pm1.5$, that is, 5 and 2.

Some consequences:
- $\det\mathbf{A}=0$ exactly when some eigenvalue is 0, matching "invertible iff 0 is not an eigenvalue";
- a positive definite matrix has positive determinant and positive trace. For $2\times2$ symmetric matrices the two together are also sufficient, but not in general: $\operatorname{diag}(-1,-1,5)$ has trace 3 and determinant 5;
- the determinant is the factor by which $\mathbf{A}$ scales volume, the absolute Jacobian determinant of change of variables; in the eigenvector basis that factor is visibly the product of the stretches.

### The Rayleigh quotient

For symmetric $\mathbf{S}$ and $\mathbf{x}\ne\mathbf{0}$, the **Rayleigh quotient** is the energy per unit length squared:

$$
R(\mathbf{x})=\frac{\mathbf{x}^\top\mathbf{S}\mathbf{x}}{\mathbf{x}^\top\mathbf{x}}.
$$

It does not change when $\mathbf{x}$ is rescaled, so it is really a function on the unit sphere. At an eigenvector it equals the eigenvalue: $R(\mathbf{q}_i)=\lambda_i$.

> [!note] Theorem (Rayleigh)
> With eigenvalues ordered $\lambda_1\ge\cdots\ge\lambda_n$,
>
> $$
> \lambda_n\le R(\mathbf{x})\le\lambda_1\quad\text{for every }\mathbf{x}\ne\mathbf{0},
> $$
>
> with the maximum attained at $\mathbf{q}_1$ and the minimum at $\mathbf{q}_n$. Every eigenvector is a stationary point of $R$, and every stationary point is an eigenvector.

The proof is (8) again. With $c_i=\mathbf{q}_i^\top\mathbf{x}$,

$$
R(\mathbf{x})=\frac{\lambda_1c_1^2+\cdots+\lambda_nc_n^2}{c_1^2+\cdots+c_n^2},
$$

a weighted average of the eigenvalues with weights $c_i^2/\sum_jc_j^2$. An average lies between the smallest and largest values, and it equals $\lambda_1$ only when all the weight sits on the top eigenvalue. Alternatively, maximize $\mathbf{x}^\top\mathbf{S}\mathbf{x}$ subject to $\mathbf{x}^\top\mathbf{x}=1$ with a [[Multivariable Calculus#Constrained optimization: Lagrange multipliers|Lagrange multiplier]]: the condition $2\mathbf{S}\mathbf{x}=2\nu\mathbf{x}$ says the stationary points are exactly the eigenvectors, with value $\nu=\lambda$.

> [!example] Example: the Rayleigh quotient of $\mathbf{S}=\begin{bmatrix}2&1\\1&2\end{bmatrix}$
> On the unit circle $\mathbf{x}=(\cos\theta,\sin\theta)$, the energy is $2\cos^2\theta+2\sin\theta\cos\theta+2\sin^2\theta=2+\sin2\theta$. It ranges from 1 to 3, with the maximum at $\theta=45°$, the direction $(1,1)$, and the minimum at $\theta=135°$, the direction $(1,-1)$ (Figure 4). At $\mathbf{x}=(1,0)$ it is $2$, the average of the two eigenvalues, because $(1,0)$ has equal weight on both eigenvectors.

![[Probability and Linear Algebra/images/eigenvalues-and-eigenvectors-rayleigh.svg]]
*Figure 4. The Rayleigh quotient of $\mathbf{S}$ on the unit circle is $2+\sin2\theta$. It is maximized by the top eigenvector (value $\lambda_1=3$) and minimized by the bottom one ($\lambda_2=1$), and every other direction gives a weighted average of the two.*

The Rayleigh quotient turns eigenvalues into optimization problems, and that is how they appear in machine learning:
- **PCA** looks for the direction of maximum variance, $\max\mathbf{w}^\top\boldsymbol\Sigma\mathbf{w}$ over unit $\mathbf{w}$, so the answer is the top eigenvector; see [[Matrix Factorization and Dimensionality Reduction#Maximum variance|maximum variance]].
- **Spectral clustering** minimizes a Rayleigh quotient of the graph Laplacian; see [[Clustering and Expectation Maximization#Spectral clustering|spectral clustering]].
- **Fisher's discriminant** maximizes a ratio of two quadratic forms, a generalized Rayleigh quotient; see [[Discriminant Analysis]].

Restricting to directions orthogonal to $\mathbf{q}_1$ makes $\mathbf{q}_2$ the maximizer with value $\lambda_2$, and so on. The **Courant–Fischer** theorem makes this precise without referring to the other eigenvectors: $\lambda_k$ is the maximum, over $k$-dimensional subspaces, of the minimum of $R$ on that subspace.[^hornjohnson2012]

## Computing eigenvalues

### Not through the characteristic polynomial

By hand, we solved $\det(\mathbf{A}-\lambda\mathbf{I})=0$. On a computer this is a poor idea, for two reasons. First, for $n\ge5$ there is no formula for the roots of a general polynomial (Abel–Ruffini), so any method must iterate anyway. Second, polynomial roots can be extremely sensitive to tiny errors in the coefficients, while the eigenvalues of the matrix itself, especially of a symmetric matrix, are not. Practical methods work on the matrix directly, using the two ideas of this lecture: repeated multiplication emphasizes the dominant eigenvector (6), and similarity transformations preserve eigenvalues while making the matrix triangular.[^trefethen1997]

### Power iteration and the QR algorithm

**Power iteration** repeatedly multiplies a starting vector by $\mathbf{A}$ and normalizes:

$$
\mathbf{z}_{k+1}=\frac{\mathbf{A}\mathbf{z}_k}{\lVert\mathbf{A}\mathbf{z}_k\rVert}.
$$

By (6), $\mathbf{A}^k\mathbf{z}_0=c_1\lambda_1^k\big(\mathbf{x}_1+\sum_{i\ge2}\tfrac{c_i}{c_1}(\lambda_i/\lambda_1)^k\mathbf{x}_i\big)$. If $|\lambda_1|>|\lambda_2|\ge\cdots$ and $c_1\ne0$ (true for a random start, with probability 1), every other component shrinks like $|\lambda_2/\lambda_1|^k$ and the direction converges to $\mathbf{x}_1$. The method dates back to von Mises in 1929.[^vonmises1929] For a symmetric matrix, the Rayleigh quotient $\mathbf{z}_k^\top\mathbf{S}\mathbf{z}_k$ estimates $\lambda_1$ (with error of order $|\lambda_2/\lambda_1|^{2k}$, twice as fast as the vector).

> [!example] Example: power iteration by hand
> Take $\mathbf{A}=\begin{bmatrix}4&1\\2&3\end{bmatrix}$ and $\mathbf{z}_0=(1,0)$, skipping the normalization (it only rescales):
>
> $$
> \mathbf{z}_1=\begin{bmatrix}4\\2\end{bmatrix},\quad
> \mathbf{z}_2=\begin{bmatrix}18\\14\end{bmatrix},\quad
> \mathbf{z}_3=\begin{bmatrix}86\\78\end{bmatrix},\quad
> \mathbf{z}_4=\begin{bmatrix}422\\406\end{bmatrix}.
> $$
>
> The ratio of the two components, $0.5,\ 0.78,\ 0.91,\ 0.96$, approaches 1, the direction of $(1,1)$. The growth of the first component, $4,\ 4.5,\ 4.78,\ 4.91$, approaches $\lambda_1=5$. The exact formula explains the rate: $(1,0)=\tfrac23(1,1)+\tfrac13(1,-2)$, so
>
> $$
> \mathbf{z}_k=\tfrac23\,5^k\begin{bmatrix}1\\1\end{bmatrix}+\tfrac13\,2^k\begin{bmatrix}1\\-2\end{bmatrix},
> $$
>
> and the unwanted component shrinks relative to the wanted one by $2/5$ per step (Figure 5a).

![[Probability and Linear Algebra/images/eigenvalues-and-eigenvectors-power-iteration.svg]]
*Figure 5. (a) The hand example: the directions of $\mathbf{z}_k=\mathbf{A}^k\mathbf{z}_0$ swing towards the dominant eigenvector $(1,1)$, closing the gap by a factor $2/5$ per step. (b) Power iteration on a random $30\times30$ covariance matrix with $\lambda_2/\lambda_1=0.84$. The distance to the top eigenvector falls geometrically, parallel to the predicted rate $(\lambda_2/\lambda_1)^k$.*

Power iteration needs only matrix–vector products, so it works on huge sparse matrices that could never be factorized; [[Markov Chains#PageRank: the stationary distribution of the web|PageRank]] is computed this way. Its limitations follow from the same formula:
- **Speed.** When $|\lambda_2/\lambda_1|$ is close to 1, convergence is slow (Figure 5b needs about 60 steps for 5 digits).
- **Largest magnitude, not largest value.** It finds the eigenvalue of largest $|\lambda|$. For $\operatorname{diag}(1,-3)$ it converges to $\lambda=-3$, with the iterates flipping sign at every step; if $\lambda$ and $-\lambda$ tie for the largest magnitude, the direction never settles. A **shift** fixes both: power iteration on $\mathbf{S}+\sigma\mathbf{I}$ (with large enough $\sigma$) finds the largest $\lambda$, and on $(\mathbf{S}-\sigma\mathbf{I})^{-1}$ (**inverse iteration**) finds the eigenvalue nearest $\sigma$.
- **One eigenvector at a time.** For the next one, use **deflation**. If $\mathbf{S}$ is symmetric and $(\lambda_1,\mathbf{q}_1)$ has been found, then

$$
\mathbf{S}-\lambda_1\mathbf{q}_1\mathbf{q}_1^\top=\lambda_2\mathbf{q}_2\mathbf{q}_2^\top+\cdots+\lambda_n\mathbf{q}_n\mathbf{q}_n^\top
$$

by (7). The deflated matrix has the same eigenvectors, with $\lambda_1$ replaced by 0, so power iteration on it finds $(\lambda_2,\mathbf{q}_2)$. For our $\mathbf{S}$, $\mathbf{S}-3\mathbf{q}_1\mathbf{q}_1^\top=\begin{bmatrix}0.5&-0.5\\-0.5&0.5\end{bmatrix}=\mathbf{q}_2\mathbf{q}_2^\top$. PCA uses the same identity to extract [[Matrix Factorization and Dimensionality Reduction#More components: deflation|one component after another]].

**The QR algorithm** computes all eigenvalues at once, and it is what `numpy.linalg.eig` and `eigh` run underneath (through LAPACK). Starting from $\mathbf{A}^{(0)}=\mathbf{A}$, factor $\mathbf{A}^{(k)}=\mathbf{Q}^{(k)}\mathbf{R}^{(k)}$ by [[Orthogonality and Least Squares#Gram–Schmidt and A = QR|Gram–Schmidt]], then multiply the factors in the **reverse** order:

$$
\mathbf{A}^{(k+1)}=\mathbf{R}^{(k)}\mathbf{Q}^{(k)}=\mathbf{Q}^{(k)\top}\mathbf{A}^{(k)}\mathbf{Q}^{(k)}.
$$

The second form shows that every iterate is **similar** to $\mathbf{A}$, so the eigenvalues never change. Under mild conditions the iterates converge to the triangular Schur form (diagonal, for symmetric $\mathbf{A}$), with the eigenvalues on the diagonal in decreasing order of magnitude. It can be shown to be power iteration applied to a whole orthonormal basis at once.[^francis1961]<sup>,</sup>[^trefethen1997]

> [!example] Example: one QR step by hand
> For $\mathbf{S}=\begin{bmatrix}2&1\\1&2\end{bmatrix}$, Gram–Schmidt on the columns gives
>
> $$
> \mathbf{Q}^{(0)}=\frac{1}{\sqrt5}\begin{bmatrix}2&-1\\1&2\end{bmatrix},\qquad
> \mathbf{R}^{(0)}=\begin{bmatrix}\sqrt5&4/\sqrt5\\0&3/\sqrt5\end{bmatrix},
> $$
>
> $$
> \mathbf{A}^{(1)}=\mathbf{R}^{(0)}\mathbf{Q}^{(0)}=\begin{bmatrix}2.8&0.6\\0.6&1.2\end{bmatrix}.
> $$
>
> The trace (4) and determinant ($3.36-0.36=3$) are unchanged, as similarity demands. The off-diagonal entry fell from 1 to 0.6, and the diagonal moved towards $(3,1)$. Continuing, the off-diagonal entries are $0.22$, $0.074$, $0.025$, $0.0082$: each step multiplies them by about $\lambda_2/\lambda_1=1/3$, the same rate as power iteration.

Production solvers add three refinements: a first reduction to tridiagonal (symmetric) or Hessenberg form, which makes each step cheap; **shifts**, which make convergence very fast; and deflation once an eigenvalue has converged. For large sparse matrices, Lanczos and Arnoldi iterations build on power iteration instead. Trefethen and Bau give the full story.[^trefethen1997]

> [!example]- Code: power iteration, deflation and the QR algorithm
> ```python
> import numpy as np
>
> def power_iteration(S, iters=100, seed=0):
>     v = np.random.default_rng(seed).normal(size=len(S)); v /= np.linalg.norm(v)
>     for _ in range(iters):
>         w = S @ v; v = w / np.linalg.norm(w)
>     return v @ S @ v, v                        # Rayleigh quotient, eigenvector
>
> def qr_algorithm(A, iters=500):
>     T = A.copy()
>     for _ in range(iters):
>         Q, R = np.linalg.qr(T); T = R @ Q      # similar to A at every step
>     return T
>
> rng = np.random.default_rng(1)
> B = rng.normal(size=(6, 6)); S = B @ B.T       # a random PSD matrix
> ev = np.linalg.eigvalsh(S)                     # ascending, for reference
> lam1, q1 = power_iteration(S, 1000)
> lam2, q2 = power_iteration(S - lam1 * np.outer(q1, q1), 1000)   # deflation
> print(np.allclose([lam1, lam2], ev[[-1, -2]]), abs(q1 @ q2) < 1e-8)
> T = qr_algorithm(S, 2000)
> print(np.allclose(np.sort(np.diag(T)), ev), np.abs(np.tril(T, -1)).max() < 1e-8)
> ```

## Key takeaways

- An eigenvector keeps its direction: $\mathbf{A}\mathbf{x}=\lambda\mathbf{x}$. Eigenvalues are the roots of $\det(\mathbf{A}-\lambda\mathbf{I})=0$, $n$ of them with multiplicity, possibly complex (rotations); eigenvectors span the nullspaces of $\mathbf{A}-\lambda\mathbf{I}$.
- With $n$ independent eigenvectors, $\mathbf{A}=\mathbf{X}\boldsymbol\Lambda\mathbf{X}^{-1}$ and $\mathbf{A}^k=\mathbf{X}\boldsymbol\Lambda^k\mathbf{X}^{-1}$: each eigen-component evolves on its own. $\mathbf{A}^k\to\mathbf{0}$ iff all $|\lambda|<1$; the largest $|\lambda|$ dominates in the long run, the second sets the speed. For $d\mathbf{u}/dt=\mathbf{A}\mathbf{u}$, $\lambda^k$ becomes $e^{\lambda t}$.
- Similar matrices $\mathbf{M}^{-1}\mathbf{A}\mathbf{M}$ share eigenvalues, trace, determinant and rank, but equal eigenvalues do not imply similarity. Distinct eigenvalues guarantee diagonalizability; repeated ones may leave a matrix defective. Diagonalizable and invertible are unrelated properties.
- Symmetric matrices have real eigenvalues and orthonormal eigenvectors: $\mathbf{S}=\mathbf{Q}\boldsymbol\Lambda\mathbf{Q}^\top=\sum_i\lambda_i\mathbf{q}_i\mathbf{q}_i^\top$, a weighted sum of projections. "Largest" terms mean largest $|\lambda|$ when $\mathbf{S}$ is indefinite.
- Positive definite means positive energy $\mathbf{x}^\top\mathbf{S}\mathbf{x}$, equivalently positive eigenvalues; the level sets are ellipses with semi-axes $1/\sqrt{\lambda_i}$ along the eigenvectors. $\mathbf{A}^\top\mathbf{A}\succeq0$ always, and $\mathbf{A}\mathbf{A}^\top$ has the same non-zero eigenvalues. Hessians, covariances and kernel matrices live here.
- Trace = sum of eigenvalues, determinant = product. The Rayleigh quotient is a weighted average of eigenvalues, maximized by the top eigenvector: eigenvalue problems are optimization problems.
- Computers find eigenvalues by iteration: power iteration (rate $|\lambda_2/\lambda_1|$, plus deflation and shifts) and the QR algorithm (similarity steps towards triangular form).

## Where this shows up next

- [[First-Order Optimization]] analyses gradient descent on a quadratic through the Hessian's eigenvalues: the step size must satisfy $\alpha<2/\lambda_{\max}$, and the condition number $\lambda_{\max}/\lambda_{\min}$ sets the rate.
- [[Five Factorizations of a Matrix]] places $\mathbf{X}\boldsymbol\Lambda\mathbf{X}^{-1}$ and $\mathbf{Q}\boldsymbol\Lambda\mathbf{Q}^\top$ alongside $\mathbf{C}\mathbf{R}$, $\mathbf{L}\mathbf{U}$, $\mathbf{Q}\mathbf{R}$ and the SVD.
- [[Matrix Factorization and Dimensionality Reduction]] builds the SVD from the eigenvectors of $\mathbf{A}^\top\mathbf{A}$ and $\mathbf{A}\mathbf{A}^\top$, proves Eckart–Young, and derives PCA and kernel PCA as eigenproblems.
- [[Markov Chains]] uses the eigenvalue 1 for stationary distributions, the spectral gap for mixing, and power iteration for PageRank.
- [[Clustering and Expectation Maximization]] clusters graphs with the bottom eigenvectors of the Laplacian; [[Discriminant Analysis]] solves a generalized eigenproblem.
- In reinforcement learning, the policy-evaluation matrix $\mathbf{I}-\gamma\mathbf{P}$, with $\mathbf{P}$ the transition matrix under the policy, is invertible because the eigenvalues of $\gamma\mathbf{P}$ have magnitude at most $\gamma<1$ ([[Introduction to Reinforcement Learning#Matrix form and closed-form policy evaluation|Introduction to RL]]), and TD learning with [[Linear Function Approximation]] converges because a certain matrix is positive definite.
- Weight initialization in [[Neural Networks#Initialization|Neural Networks]] is about keeping products of many matrices from exploding or vanishing, the matrix-power story of (6).

<!-- References: footnote definitions below render as the numbered "References" list at the bottom of the page -->

[^strang-video]: Strang, G. (2020). *Part 4: Eigenvalues and Eigenvectors*. A 2020 Vision of Linear Algebra, MIT OpenCourseWare. [YouTube](https://www.youtube.com/watch?v=GyC3gl6weYo)
[^strang-ocw]: Strang, G. (2020). *A 2020 Vision of Linear Algebra* (RES.18-010). MIT OpenCourseWare. [course page](https://ocw.mit.edu/courses/res-18-010-a-2020-vision-of-linear-algebra-spring-2020/)
[^strang2023]: Strang, G. (2023). *Introduction to Linear Algebra* (6th ed.), Chapter 6 (Eigenvalues and Eigenvectors). Wellesley-Cambridge Press. [book website](https://math.mit.edu/linearalgebra/)
[^3b1b-eigen]: 3Blue1Brown (2016). *Eigenvectors and eigenvalues | Chapter 14, Essence of linear algebra*. [YouTube](https://www.youtube.com/watch?v=PFDu9oVAE-g)
[^3b1b-basis]: 3Blue1Brown (2016). *Change of basis | Chapter 13, Essence of linear algebra*. [YouTube](https://www.youtube.com/watch?v=P2LTAUO1TdA)
[^hornjohnson2012]: Horn, R. A., & Johnson, C. R. (2012). *Matrix Analysis* (2nd ed.), Chapters 2 (Schur form), 3 (Jordan form) and 4 (Rayleigh quotient, Courant–Fischer). Cambridge University Press. [doi:10.1017/CBO9781139020411](https://doi.org/10.1017/CBO9781139020411)
[^3b1b-trick]: 3Blue1Brown (2021). *A quick trick for computing eigenvalues | Chapter 15, Essence of linear algebra*. [YouTube](https://www.youtube.com/watch?v=e50Bj7jn9IQ)
[^trefethen1997]: Trefethen, L. N., & Bau, D. (1997). *Numerical Linear Algebra*, Lectures 24–29. SIAM. [doi:10.1137/1.9780898719574](https://doi.org/10.1137/1.9780898719574)
[^vonmises1929]: von Mises, R., & Pollaczek-Geiringer, H. (1929). *Praktische Verfahren der Gleichungsauflösung*. ZAMM, 9(1), 58–77. [doi:10.1002/zamm.19290090105](https://doi.org/10.1002/zamm.19290090105)
[^francis1961]: Francis, J. G. F. (1961). *The QR Transformation: A Unitary Analogue to the LR Transformation, Part 1*. The Computer Journal, 4(3), 265–271. [doi:10.1093/comjnl/4.3.265](https://doi.org/10.1093/comjnl/4.3.265)
