---
title: Matrix Factorization and Dimensionality Reduction
tags:
  - linear-algebra
  - svd
  - pca
  - matrix-factorization
  - dimensionality-reduction
  - recommender-systems
prev: "[[Non-parametric Models]]"
next: "[[Clustering and Expectation Maximization]]"
---

> **Prerequisites:** [[Eigenvalues and Eigenvectors]] and [[The Four Fundamental Subspaces]] (eigenvalues, orthogonality, rank), [[Linear Models]] (least squares, ridge regression), and [[Support Vector Machines#Kernels|kernels]]. [[Five Factorizations of a Matrix]] puts the SVD next to the other matrix factorizations.
>
> **TL;DR:**
> - **Matrix factorization** writes a matrix as a product of simpler pieces. The **spectral decomposition** $\mathbf{S}=\mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^\top$ handles symmetric matrices, and the **SVD** $\mathbf{A}=\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$ handles *every* matrix.
> - Truncating the SVD gives the **best low-rank approximation** (Eckart–Young). This powers compression, denoising and recommender systems. **NMF** and **CUR** trade a little accuracy for interpretability.
> - **Dimensionality reduction** keeps the directions that matter. **PCA** keeps the directions of maximum variance, which are the top right singular vectors of the centred data. **Autoencoders** and **kernel PCA** make it non-linear, and manifold methods (Laplacian eigenmaps, t-SNE) preserve neighbourhoods instead of variance.
>
> **Code:** [Principal-Component-Analysis](https://github.com/parag1604/Principal-Component-Analysis): PCA by eigendecomposition and by the SVD; [Matrix-Factorization-Recommender](https://github.com/parag1604/Matrix-Factorization-Recommender): a movie recommender with truncated SVD and alternating least squares.

# Part I: Matrix factorization

Matrix factorization is a powerful tool for many practical problems. We want to write a general $\mathbf{A}\in\mathbb{R}^{m\times n}$ as a product of matrices with simple structure (orthogonal, diagonal, triangular, non-negative, low-rank). We start with square matrices.

## Eigenvalues and the spectral theorem: a recap

> [!tip] Prerequisite: eigenvalues
> This lecture assumes [[Eigenvalues and Eigenvectors]]: diagonalization $\mathbf{A}=\mathbf{X}\boldsymbol{\Lambda}\mathbf{X}^{-1}$, similar matrices, the spectral theorem, positive semi-definite matrices, and how eigenvalues are computed (power iteration and the QR algorithm[^trefethen1997]). Study it first if any of these are new.

The one fact we use repeatedly is the **spectral decomposition** of a symmetric matrix $\mathbf{S}=\mathbf{S}^\top$, with orthonormal eigenvectors $\mathbf{q}_i$ and real eigenvalues ordered $\lambda_1\ge\lambda_2\ge\cdots$:

$$
\mathbf{S}=\mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^\top=\lambda_1\mathbf{q}_1\mathbf{q}_1^\top+\lambda_2\mathbf{q}_2\mathbf{q}_2^\top+\cdots+\lambda_r\mathbf{q}_r\mathbf{q}_r^\top.
$$

Each $\mathbf{q}_i\mathbf{q}_i^\top$ is a rank-1 projection, so $\mathbf{S}$ is built by **rank-1 updates**, $\mathbf{S}_k=\mathbf{S}_{k-1}+\lambda_k\mathbf{q}_k\mathbf{q}_k^\top$. [[#Best low-rank approximation: Eckart–Young|Below]] we show that, for PSD $\mathbf{S}$, $\mathbf{S}_k$ is the best rank-$k$ approximation of $\mathbf{S}$. The SVD extends this picture to every rectangular matrix.

## Singular value decomposition (SVD)

The spectral decomposition needs a square, symmetric matrix. For a general $\mathbf{A}\in\mathbb{R}^{m\times n}$, $\mathbf{A}\mathbf{x}=\lambda\mathbf{x}$ does not even type-check, because $\mathbf{x}$ and $\mathbf{A}\mathbf{x}$ live in different spaces. Instead we look for pairs of vectors that $\mathbf{A}$ maps onto each other:

$$
\mathbf{A}\mathbf{v}_i=\sigma_i\mathbf{u}_i,\qquad\mathbf{v}_i\in\mathbb{R}^n,\ \mathbf{u}_i\in\mathbb{R}^m,\ \sigma_i\ge0.
$$

The $\sigma_i$ are the **singular values**, the rectangular counterpart of eigenvalues. Even for square matrices they are not the absolute eigenvalues, unless the matrix is normal. The demand that makes the pairs useful is that **both** families are orthonormal: the $\mathbf{v}_i$ are perpendicular in the input space and their images $\mathbf{u}_i$ are perpendicular in the output space. Stacking $r=\operatorname{rank}(\mathbf{A})$ such pairs gives $\mathbf{A}\mathbf{V}=\mathbf{U}\boldsymbol{\Sigma}$, i.e.

$$
\boxed{\;\mathbf{A}=\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top=\sum_{i=1}^{r}\sigma_i\,\mathbf{u}_i\mathbf{v}_i^\top\;}
$$

with $\sigma_1\ge\sigma_2\ge\cdots\ge\sigma_r>0$. This section follows the story of Strang's video on singular values,[^strang2020svd] with our own examples. Chapter 7 of his textbook has the full treatment.[^strang2023]

### Rotate, stretch, rotate: the geometry

Read $\mathbf{A}\mathbf{x}=\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top\mathbf{x}$ from right to left (Figure 1):

1. $\mathbf{V}^\top$ rotates (possibly reflects) the input, so that the right singular vectors land on the coordinate axes;
2. $\boldsymbol{\Sigma}$ stretches axis $i$ by $\sigma_i$ (and, if $m\ne n$, adds or drops coordinates);
3. $\mathbf{U}$ rotates (possibly reflects) the result into place.

So **every** linear map, however it was written down, is a rotation, an axis-aligned stretch and another rotation. A circle has no preferred direction, so the first rotation does nothing visible to it; the stretch turns it into an axis-aligned ellipse; the last rotation tilts the ellipse. Hence every matrix sends the unit sphere to an ellipsoid (flattened, if $r<m$), with semi-axes $\sigma_i\mathbf{u}_i$. The longest semi-axis is $\sigma_1$, which is why $\sigma_1$ is the largest stretch $\max_{\lVert\mathbf{x}\rVert=1}\lVert\mathbf{A}\mathbf{x}\rVert$.

![[Machine Learning/images/matrix-factorization-svd-geometry.svg]]
*Figure 1: the SVD of the worked example $\mathbf{A}=\begin{bmatrix}5&5\\-1&7\end{bmatrix}$ as rotate, stretch, rotate. The right singular vectors $\mathbf{v}_1,\mathbf{v}_2$ (arrows) are rotated onto the axes, stretched by $\sigma_1=4\sqrt5\approx8.94$ and $\sigma_2=2\sqrt5\approx4.47$, then rotated by $45°$ to become the ellipse's semi-axes $\sigma_1\mathbf{u}_1,\sigma_2\mathbf{u}_2$.*

> [!tip] Intuition: why orthogonal in, orthogonal out is special
> Any matrix maps *some* basis to *some* vectors. The SVD finds the one orthonormal input basis whose images are again orthogonal. Then $\mathbf{A}$ is "diagonal" in a pair of well-chosen coordinate systems, one for the input and one for the output. Eigenvectors try to do this with a single coordinate system, which is why they need a square matrix and can fail. 3Blue1Brown's video on linear transformations[^3b1b-lintrans] is a good way to see matrices as maps before reading on.

### Finding the singular vectors

**Where $\mathbf{U}$ and $\mathbf{V}$ come from.** Suppose such a factorization exists with orthonormal $\mathbf{U}$ and $\mathbf{V}$. Multiply $\mathbf{A}$ by its transpose, and the orthogonal factors cancel ($\mathbf{V}^\top\mathbf{V}=\mathbf{I}$, $\mathbf{U}^\top\mathbf{U}=\mathbf{I}$):

$$
\mathbf{A}^\top\mathbf{A}=\mathbf{V}\boldsymbol{\Sigma}^\top\boldsymbol{\Sigma}\mathbf{V}^\top,\qquad\mathbf{A}\mathbf{A}^\top=\mathbf{U}\boldsymbol{\Sigma}\boldsymbol{\Sigma}^\top\mathbf{U}^\top.
$$

Both products are symmetric positive semi-definite, so the [[Eigenvalues and Eigenvectors#The spectral theorem|spectral theorem]] applies to them. The **right singular vectors** $\mathbf{v}_i$ are eigenvectors of $\mathbf{A}^\top\mathbf{A}$, the **left singular vectors** $\mathbf{u}_i$ are eigenvectors of $\mathbf{A}\mathbf{A}^\top$, and $\sigma_i=\sqrt{\lambda_i(\mathbf{A}^\top\mathbf{A})}$. The two products share their non-zero eigenvalues $\sigma_i^2$.

**Existence: the construction.** This suggests a recipe that works for **every** matrix. Choose orthonormal eigenvectors $\mathbf{v}_1,\dots,\mathbf{v}_n$ of $\mathbf{A}^\top\mathbf{A}$ with eigenvalues $\sigma_1^2\ge\cdots\ge\sigma_r^2>0=\sigma_{r+1}^2=\cdots$. For $i\le r$, *define*

$$
\mathbf{u}_i=\frac{\mathbf{A}\mathbf{v}_i}{\sigma_i}.
$$

Then $\mathbf{A}\mathbf{v}_i=\sigma_i\mathbf{u}_i$ holds by construction, and the only thing to check is that the $\mathbf{u}_i$ are orthonormal:

$$
\mathbf{u}_i^\top\mathbf{u}_j=\frac{\mathbf{v}_i^\top\mathbf{A}^\top\mathbf{A}\mathbf{v}_j}{\sigma_i\sigma_j}=\frac{\sigma_j^2\,\mathbf{v}_i^\top\mathbf{v}_j}{\sigma_i\sigma_j}=\begin{cases}1&i=j,\\0&i\neq j.\end{cases}
$$

The middle step uses that $\mathbf{v}_j$ is an eigenvector of $\mathbf{A}^\top\mathbf{A}$; the last uses the orthonormality of the $\mathbf{v}$'s. So orthogonality in the input space is **transported** to the output space, which is the whole trick. The same computation shows $\mathbf{A}\mathbf{A}^\top\mathbf{u}_i=\sigma_i^2\mathbf{u}_i$, so the $\mathbf{u}_i$ really are eigenvectors of $\mathbf{A}\mathbf{A}^\top$.

> [!warning] Don't compute $\mathbf{U}$ and $\mathbf{V}$ separately
> Taking eigenvectors of $\mathbf{A}^\top\mathbf{A}$ and of $\mathbf{A}\mathbf{A}^\top$ independently can fail: each eigenvector is defined only up to sign (or, for repeated $\sigma$, up to a rotation within the eigenspace), and independent choices need not satisfy $\mathbf{A}\mathbf{v}_i=\sigma_i\mathbf{u}_i$. Compute the $\mathbf{v}_i$ and then *derive* $\mathbf{u}_i=\mathbf{A}\mathbf{v}_i/\sigma_i$. Library routines do neither: they never form $\mathbf{A}^\top\mathbf{A}$, which would square the [[#The condition number|condition number]], and instead reduce $\mathbf{A}$ to bidiagonal form with orthogonal transformations (Golub–Kahan).[^golubkahan1965] The cost is $O(mn\min(m,n))$.

### A 2×2 example by hand

Take

$$
\mathbf{A}=\begin{bmatrix}5&5\\-1&7\end{bmatrix}.
$$

**Step 1: $\mathbf{A}^\top\mathbf{A}$ and its eigenvalues.**

$$
\mathbf{A}^\top\mathbf{A}=\begin{bmatrix}5&-1\\5&7\end{bmatrix}\begin{bmatrix}5&5\\-1&7\end{bmatrix}=\begin{bmatrix}26&18\\18&74\end{bmatrix}.
$$

Its trace is $100$ and its determinant is $26\cdot74-18^2=1600$, so the eigenvalues solve $\lambda^2-100\lambda+1600=0$: $\lambda_1=80$, $\lambda_2=20$. The singular values are $\sigma_1=\sqrt{80}=4\sqrt5$ and $\sigma_2=\sqrt{20}=2\sqrt5$. Check: $\sigma_1\sigma_2=40=\lvert\det\mathbf{A}\rvert$.

**Step 2: the right singular vectors.** For $\lambda_1=80$, the first row of $\mathbf{A}^\top\mathbf{A}-80\mathbf{I}$ reads $-54x+18y=0$, so $y=3x$. The second eigenvector is perpendicular to the first. Normalizing,

$$
\mathbf{v}_1=\frac{1}{\sqrt{10}}\begin{bmatrix}1\\3\end{bmatrix},\qquad\mathbf{v}_2=\frac{1}{\sqrt{10}}\begin{bmatrix}-3\\1\end{bmatrix}.
$$

**Step 3: the left singular vectors, by $\mathbf{u}_i=\mathbf{A}\mathbf{v}_i/\sigma_i$.**

$$
\begin{aligned}
\mathbf{A}\mathbf{v}_1&=\frac{1}{\sqrt{10}}\begin{bmatrix}20\\20\end{bmatrix}
&&\Longrightarrow\quad\mathbf{u}_1=\frac{1}{4\sqrt5\sqrt{10}}\begin{bmatrix}20\\20\end{bmatrix}=\frac{1}{\sqrt2}\begin{bmatrix}1\\1\end{bmatrix},\\
\mathbf{A}\mathbf{v}_2&=\frac{1}{\sqrt{10}}\begin{bmatrix}-10\\10\end{bmatrix}
&&\Longrightarrow\quad\mathbf{u}_2=\frac{1}{2\sqrt5\sqrt{10}}\begin{bmatrix}-10\\10\end{bmatrix}=\frac{1}{\sqrt2}\begin{bmatrix}-1\\1\end{bmatrix}.
\end{aligned}
$$

They are orthonormal, as the proof promised. So

$$
\mathbf{A}=\underbrace{\frac{1}{\sqrt2}\begin{bmatrix}1&-1\\1&1\end{bmatrix}}_{\mathbf{U}}\underbrace{\begin{bmatrix}4\sqrt5&0\\0&2\sqrt5\end{bmatrix}}_{\boldsymbol{\Sigma}}\underbrace{\frac{1}{\sqrt{10}}\begin{bmatrix}1&3\\-3&1\end{bmatrix}}_{\mathbf{V}^\top}.
$$

Both $\mathbf{U}$ and $\mathbf{V}$ have determinant $+1$, so they are pure rotations: $\mathbf{V}^\top$ rotates by $-71.6°$ (taking $\mathbf{v}_1$, at angle $\arctan3$, onto the $x$-axis) and $\mathbf{U}$ rotates by $+45°$. This is Figure 1. Flipping the sign of $\mathbf{v}_2$ alone would turn $\mathbf{V}$ into a reflection and break $\mathbf{A}\mathbf{v}_2=\sigma_2\mathbf{u}_2$; flipping $\mathbf{v}_2$ and $\mathbf{u}_2$ **together** gives another valid SVD with two reflections. For square $\mathbf{A}$, $\det\mathbf{U}\det\mathbf{V}$ always has the sign of $\det\mathbf{A}$.

### Full and reduced SVD and the four subspaces

The SVD comes in two sizes (Figure 2).

- The **reduced** (thin, compact) SVD keeps only the $r$ pieces with $\sigma_i>0$: $\mathbf{U}_r\in\mathbb{R}^{m\times r}$ and $\mathbf{V}_r\in\mathbb{R}^{n\times r}$ have orthonormal columns, and $\boldsymbol{\Sigma}_r$ is $r\times r$, diagonal and invertible. This is all you need to rebuild $\mathbf{A}$.
- The **full** SVD completes $\mathbf{U}$ to an $m\times m$ orthogonal matrix and $\mathbf{V}$ to an $n\times n$ one, by adding orthonormal bases of the missing directions. $\boldsymbol{\Sigma}$ is then $m\times n$, with the $\sigma_i$ on its diagonal and zeros elsewhere. Note that $\boldsymbol{\Sigma}\boldsymbol{\Sigma}^\top$ ($m\times m$) and $\boldsymbol{\Sigma}^\top\boldsymbol{\Sigma}$ ($n\times n$) then have different sizes but the same non-zero diagonal entries $\sigma_i^2$.

The extra columns are not arbitrary: they are bases of the two null spaces. The full SVD hands us orthonormal bases of all **four fundamental subspaces** at once:

| Columns | Span | Dimension | Why |
|---|---|---|---|
| $\mathbf{v}_1,\dots,\mathbf{v}_r$ | row space $C(\mathbf{A}^\top)$ | $r$ | orthogonal complement of $N(\mathbf{A})$ |
| $\mathbf{v}_{r+1},\dots,\mathbf{v}_n$ | null space $N(\mathbf{A})$ | $n-r$ | $\mathbf{A}^\top\mathbf{A}\mathbf{v}_i=\mathbf{0}$ gives $\lVert\mathbf{A}\mathbf{v}_i\rVert^2=0$ |
| $\mathbf{u}_1,\dots,\mathbf{u}_r$ | column space $C(\mathbf{A})$ | $r$ | $\mathbf{u}_i=\mathbf{A}\mathbf{v}_i/\sigma_i$ |
| $\mathbf{u}_{r+1},\dots,\mathbf{u}_m$ | left null space $N(\mathbf{A}^\top)$ | $m-r$ | orthogonal complement of $C(\mathbf{A})$ |

This is the [[The Four Fundamental Subspaces|big picture of linear algebra]] made concrete: $\mathbf{A}$ maps the orthonormal basis $\mathbf{v}_1,\dots,\mathbf{v}_r$ of the row space to the orthogonal basis $\sigma_1\mathbf{u}_1,\dots,\sigma_r\mathbf{u}_r$ of the column space, and sends $\mathbf{v}_{r+1},\dots,\mathbf{v}_n$ to zero. Going back, $\mathbf{A}^\top\mathbf{u}_i=\sigma_i\mathbf{v}_i$.

![[Machine Learning/images/matrix-factorization-svd-shapes.svg]]
*Figure 2: block shapes of the full and reduced SVD of a $5\times3$ matrix of rank 2. The full SVD adds bases of the two null spaces (pink, light blue) that multiply the zero part of $\boldsymbol{\Sigma}$, so they do not change the product. The reduced SVD drops them.*

> [!example] Example: the four subspaces of a rank-one matrix
> Let
>
> $$
> \mathbf{B}=\begin{bmatrix}1&2\\2&4\\2&4\end{bmatrix}=\begin{bmatrix}1\\2\\2\end{bmatrix}\begin{bmatrix}1&2\end{bmatrix}.
> $$
>
> Every column is a multiple of $\mathbf{c}=(1,2,2)$ and every row a multiple of $(1,2)$, so $r=1$. Since $\lVert\mathbf{c}\rVert=3$ and $\lVert(1,2)\rVert=\sqrt5$, we can read off $\sigma_1=3\sqrt5$, $\mathbf{u}_1=\tfrac13(1,2,2)$ and $\mathbf{v}_1=\tfrac{1}{\sqrt5}(1,2)$.
> - $N(\mathbf{B})$ is spanned by $\mathbf{v}_2=\tfrac{1}{\sqrt5}(2,-1)$: indeed $1\cdot2+2\cdot(-1)=0$ in each row.
> - $N(\mathbf{B}^\top)$ is spanned by $\mathbf{u}_2=\tfrac13(2,1,-2)$ and $\mathbf{u}_3=\tfrac13(2,-2,1)$. Each is perpendicular to $\mathbf{c}$ ($2+2-4=0$ and $2-4+2=0$) and to the other ($4-2-2=0$).
>
> The dimensions are $1+1=2=n$ and $1+2=3=m$, as the counting theorem requires.

### The SVD as a sum of rank-one matrices

Writing $\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$ as columns times rows (the view of matrix multiplication in [[Five Factorizations of a Matrix#Matrix multiplication as columns times rows|Five Factorizations of a Matrix]]) gives

$$
\mathbf{A}=\sigma_1\mathbf{u}_1\mathbf{v}_1^\top+\sigma_2\mathbf{u}_2\mathbf{v}_2^\top+\cdots+\sigma_r\mathbf{u}_r\mathbf{v}_r^\top.
$$

Each $\mathbf{u}_i\mathbf{v}_i^\top$ is a rank-one matrix with unit Frobenius norm, and the pieces are mutually orthogonal: $\operatorname{tr}\big((\mathbf{u}_i\mathbf{v}_i^\top)^\top\mathbf{u}_j\mathbf{v}_j^\top\big)=(\mathbf{u}_i^\top\mathbf{u}_j)(\mathbf{v}_i^\top\mathbf{v}_j)=0$ for $i\ne j$. So the SVD splits $\mathbf{A}$ into orthogonal **layers**, sorted by importance $\sigma_i$. For the worked example, the two layers are

$$
\begin{aligned}
\sigma_1\mathbf{u}_1\mathbf{v}_1^\top&=4\sqrt5\cdot\frac{1}{\sqrt2}\begin{bmatrix}1\\1\end{bmatrix}\frac{1}{\sqrt{10}}\begin{bmatrix}1&3\end{bmatrix}=\begin{bmatrix}2&6\\2&6\end{bmatrix},\\
\sigma_2\mathbf{u}_2\mathbf{v}_2^\top&=2\sqrt5\cdot\frac{1}{\sqrt2}\begin{bmatrix}-1\\1\end{bmatrix}\frac{1}{\sqrt{10}}\begin{bmatrix}-3&1\end{bmatrix}=\begin{bmatrix}3&-1\\-3&1\end{bmatrix},
\end{aligned}
$$

and indeed $\begin{bmatrix}2&6\\2&6\end{bmatrix}+\begin{bmatrix}3&-1\\-3&1\end{bmatrix}=\begin{bmatrix}5&5\\-1&7\end{bmatrix}$.

The first layer is the best rank-one summary of $\mathbf{A}$; the [[#Best low-rank approximation: Eckart–Young|Eckart–Young theorem]] below makes that precise. Its error is the second layer, of size $\sigma_2=2\sqrt5$. A symmetric PSD matrix is the special case $\mathbf{u}_i=\mathbf{v}_i=\mathbf{q}_i$, recovering the spectral sum of the previous section.

### Norms through the SVD

**Frobenius norm and trace.** The Frobenius norm is $\lVert\mathbf{A}\rVert_F=\sqrt{\sum_{ij}a_{ij}^2}=\sqrt{\operatorname{tr}(\mathbf{A}\mathbf{A}^\top)}$. For complex matrices, use the conjugate transpose $\mathbf{A}^H$. On a $2\times2$ example, $\operatorname{tr}(\mathbf{A}\mathbf{A}^\top)=a_{11}^2+a_{12}^2+a_{21}^2+a_{22}^2$. Two facts we use repeatedly:

- the **cyclic property**, $\operatorname{tr}(\mathbf{A}\mathbf{B})=\operatorname{tr}(\mathbf{B}\mathbf{A})$;
- **orthogonal invariance**: $\lVert\mathbf{Q}_1\mathbf{A}\mathbf{Q}_2\rVert_F=\lVert\mathbf{A}\rVert_F$ for orthogonal $\mathbf{Q}_1,\mathbf{Q}_2$.

Combining them gives $\lVert\mathbf{A}\rVert_F^2=\operatorname{tr}(\boldsymbol{\Sigma}\boldsymbol{\Sigma}^\top)=\sum_i\sigma_i^2$. Two other norms come straight from the singular values:

- the **spectral norm** $\lVert\mathbf{A}\rVert_2=\max_{\lVert\mathbf{x}\rVert=1}\lVert\mathbf{A}\mathbf{x}\rVert=\sigma_1$, the longest semi-axis of the ellipse in Figure 1;
- the **nuclear norm** $\lVert\mathbf{A}\rVert_*=\sigma_1+\cdots+\sigma_r$, the convex surrogate for rank used in matrix completion.

All three are *unitarily invariant*: they depend on $\mathbf{A}$ only through its singular values. For the worked example, $\lVert\mathbf{A}\rVert_F^2=25+25+1+49=100=80+20$, $\lVert\mathbf{A}\rVert_2=4\sqrt5\approx8.94$ and $\lVert\mathbf{A}\rVert_*=6\sqrt5\approx13.4$.

### The condition number

How much can a small change in the data change the answer? For a square invertible $\mathbf{A}$, solve $\mathbf{A}\mathbf{x}=\mathbf{b}$ and perturb $\mathbf{b}$ to $\mathbf{b}+\delta\mathbf{b}$. Then $\delta\mathbf{x}=\mathbf{A}^{-1}\delta\mathbf{b}$, so $\lVert\delta\mathbf{x}\rVert\le\lVert\delta\mathbf{b}\rVert/\sigma_n$, while $\lVert\mathbf{b}\rVert=\lVert\mathbf{A}\mathbf{x}\rVert\le\sigma_1\lVert\mathbf{x}\rVert$. Dividing,

$$
\begin{aligned}
\frac{\lVert\delta\mathbf{x}\rVert}{\lVert\mathbf{x}\rVert}&\le\operatorname{cond}(\mathbf{A})\,\frac{\lVert\delta\mathbf{b}\rVert}{\lVert\mathbf{b}\rVert},\\
\operatorname{cond}(\mathbf{A})&=\frac{\sigma_1}{\sigma_n}=\lVert\mathbf{A}\rVert_2\,\lVert\mathbf{A}^{-1}\rVert_2.
\end{aligned}
$$

Both inequalities are tight at once when $\mathbf{b}$ points along $\mathbf{u}_1$ and $\delta\mathbf{b}$ along $\mathbf{u}_n$, so the **condition number** $\operatorname{cond}(\mathbf{A})$ is the worst-case amplification of relative error. Geometrically it is the aspect ratio of the ellipse in Figure 1: a nearly flat ellipse means some direction is almost crushed, and undoing that crushing amplifies noise. For a rectangular matrix of full column rank, $\operatorname{cond}(\mathbf{A})=\sigma_1/\sigma_n$ is the natural definition, and it governs the sensitivity of least squares (with an extra $\operatorname{cond}(\mathbf{A})^2$ term when the residual is large). The worked example has $\operatorname{cond}(\mathbf{A})=2$: harmless.

> [!example] Example: nearly parallel equations
> The system $x+y=2$, $x+1.01y=2.01$ has the solution $(1,1)$. Change the right-hand side to $2.02$, a relative change of $0.35\%$ in $\mathbf{b}$, and the solution jumps to $(0,2)$, a relative change of $100\%$. The singular values of $\begin{bmatrix}1&1\\1&1.01\end{bmatrix}$ are about $2.005$ and $0.005$, so $\operatorname{cond}\approx402$. The observed amplification, $284$, is within the bound. The two rows are nearly parallel, so their intersection slides a long way when either line moves slightly (Figure 3).

![[Machine Learning/images/matrix-factorization-svd-condition.svg]]
*Figure 3: (a) the worked example maps the unit circle to a fat ellipse with $\operatorname{cond}(\mathbf{A})=\sigma_1/\sigma_2=2$. (b) An ill-conditioned system: the lines $x+y=2$ and $x+1.01y=c$ are almost parallel, so moving $c$ from $2.01$ to $2.02$ moves the intersection from $(1,1)$ to $(0,2)$.*

Two consequences recur throughout machine learning:

- $\operatorname{cond}(\mathbf{A}^\top\mathbf{A})=\operatorname{cond}(\mathbf{A})^2$. Forming the normal equations squares the condition number (in the example, from $402$ to about $1.6\times10^5$). That is why least squares is solved with QR or the SVD rather than by inverting $\mathbf{X}^\top\mathbf{X}$; see [[Orthogonality and Least Squares]].
- Gradient descent on a quadratic $f$ converges at a rate governed by the condition number of its Hessian, $\operatorname{cond}(\nabla^2f)$, which is why ill-conditioned problems zig-zag; see [[First-Order Optimization]]. Ridge regularization improves conditioning, since $\operatorname{cond}(\mathbf{X}^\top\mathbf{X}+\rho\mathbf{I})=(\sigma_1^2+\rho)/(\sigma_D^2+\rho)$ for $\mathbf{X}\in\mathbb{R}^{N\times D}$ (with $\sigma_D=0$ if $N<D$).

## Best low-rank approximation: Eckart–Young

> [!note] Theorem (Eckart–Young–Mirsky)
> Let $\mathbf{A}=\sum_i\sigma_i\mathbf{u}_i\mathbf{v}_i^\top$ and $\mathbf{A}_k=\sum_{i=1}^{k}\sigma_i\mathbf{u}_i\mathbf{v}_i^\top$. Among all matrices of rank **at most** $k$, $\mathbf{A}_k$ minimizes $\lVert\mathbf{A}-\mathbf{T}\rVert$ for every unitarily invariant norm, with
> $$
> \lVert\mathbf{A}-\mathbf{A}_k\rVert_2=\sigma_{k+1},\qquad\lVert\mathbf{A}-\mathbf{A}_k\rVert_F=\sqrt{\textstyle\sum_{i>k}\sigma_i^2}.
> $$
> For a symmetric matrix, the same holds with $\mathbf{S}_k$ built from the $k$ eigenvalues of **largest magnitude**. For PSD $\mathbf{S}$ these are simply the largest.[^eckartyoung1936]<sup>,</sup>[^mirsky1960]

In the worked example, the best rank-one approximation is the first layer $\begin{bmatrix}2&6\\2&6\end{bmatrix}$, with error $\sigma_2=2\sqrt5$ in both norms. No other rank-one matrix comes closer.

**Step 1: the error of the truncation.** For the symmetric case, write $\mathbf{S}=\mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^\top$ with a square orthogonal $\mathbf{Q}$, and pad $\mathbf{S}_k=\mathbf{Q}\bar{\boldsymbol{\Lambda}}_k\mathbf{Q}^\top$, where $\bar{\boldsymbol{\Lambda}}_k=\operatorname{diag}(\lambda_1,\dots,\lambda_k,0,\dots,0)$. By orthogonal invariance,

$$
\lVert\mathbf{S}-\mathbf{S}_k\rVert_F^2=\lVert\mathbf{Q}(\boldsymbol{\Lambda}-\bar{\boldsymbol{\Lambda}}_k)\mathbf{Q}^\top\rVert_F^2=\lVert\boldsymbol{\Lambda}-\bar{\boldsymbol{\Lambda}}_k\rVert_F^2=\sum_{i=k+1}^{r}\lambda_i^2.
$$

The same computation gives $\sum_{i>k}\sigma_i^2$ for the SVD. This shows how good the truncation is. It does **not** yet show that no other rank-$k$ matrix does better. That is the real content of the theorem, and needs one more idea.

**Step 2: nothing beats it in the spectral norm.** Let $\mathbf{T}$ have rank at most $k$. Its null space has dimension at least $n-k$, and $\operatorname{span}(\mathbf{v}_1,\dots,\mathbf{v}_{k+1})$ has dimension $k+1$. Two subspaces whose dimensions add up to more than $n$ must intersect, so there is a unit vector $\mathbf{x}$ in both. Writing $\mathbf{x}=\sum_{i\le k+1}\beta_i\mathbf{v}_i$ with $\sum_i\beta_i^2=1$:

$$
\lVert\mathbf{A}-\mathbf{T}\rVert_2^2\ \ge\ \lVert(\mathbf{A}-\mathbf{T})\mathbf{x}\rVert^2=\lVert\mathbf{A}\mathbf{x}\rVert^2=\sum_{i\le k+1}\sigma_i^2\beta_i^2\ \ge\ \sigma_{k+1}^2.
$$

$\mathbf{A}_k$ achieves this bound, so it is optimal in $\lVert\cdot\rVert_2$.

**Step 3: nothing beats it in the Frobenius norm.** The same null-space argument proves **Weyl's inequality** for singular values, $\sigma_{i+j-1}(\mathbf{E}+\mathbf{F})\le\sigma_i(\mathbf{E})+\sigma_j(\mathbf{F})$. Apply it with $\mathbf{E}=\mathbf{A}-\mathbf{T}$, $\mathbf{F}=\mathbf{T}$ and $j=k+1$, noting that $\sigma_{k+1}(\mathbf{T})=0$ because $\operatorname{rank}\mathbf{T}\le k$:

$$
\begin{aligned}
&\sigma_{i+k}(\mathbf{A})\le\sigma_i(\mathbf{A}-\mathbf{T})\quad\text{for all }i\\
\Longrightarrow\quad&\lVert\mathbf{A}-\mathbf{T}\rVert_F^2=\sum_i\sigma_i(\mathbf{A}-\mathbf{T})^2\ \ge\ \sum_i\sigma_{i+k}(\mathbf{A})^2=\lVert\mathbf{A}-\mathbf{A}_k\rVert_F^2.\qquad\blacksquare
\end{aligned}
$$

### Image compression

A grey-scale photo is a matrix of pixel intensities, and its truncated SVD is the best low-rank approximation of it. Storing $\mathbf{A}_k$ takes $k(m+n+1)$ numbers (the $k$ columns $\mathbf{u}_i$, the $k$ rows $\mathbf{v}_i^\top$ and the $k$ singular values) instead of $mn$.

![[Machine Learning/images/matrix-factorization-image-compression.png]]
*Figure 4: Eckart–Young as image compression. A rank-$k$ approximation stores only $k(m+n+1)$ numbers instead of $mn$. Rank 20 is already recognizable from under 8% of the numbers, and rank 50 is hard to tell apart from the original.*

![[Machine Learning/images/matrix-factorization-singular-values.svg]]
*Figure 5: left, the singular values of the photo decay quickly, which is why low rank works. Right, the measured truncation error matches the Eckart–Young formula $\sqrt{\sum_{i>k}\sigma_i^2}$ exactly.*

### The pseudo-inverse

The SVD also inverts what can be inverted. For $\mathbf{A}=\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$ of rank $r$, define $\boldsymbol{\Sigma}^+$ ($n\times m$, the transposed shape) by inverting the non-zero singular values and leaving the zeros:

$$
\begin{aligned}
\mathbf{A}^+&=\mathbf{V}\boldsymbol{\Sigma}^+\mathbf{U}^\top=\sum_{i=1}^{r}\frac{1}{\sigma_i}\mathbf{v}_i\mathbf{u}_i^\top,\\
\boldsymbol{\Sigma}^+&=\operatorname{diag}\Big(\frac{1}{\sigma_1},\dots,\frac{1}{\sigma_r},0,\dots,0\Big).
\end{aligned}
$$

In words: $\mathbf{A}$ maps $\mathbf{v}_i\mapsto\sigma_i\mathbf{u}_i$ and kills the null space; $\mathbf{A}^+$ maps $\mathbf{u}_i\mapsto\mathbf{v}_i/\sigma_i$ back and kills $N(\mathbf{A}^\top)$. On the row space and column space, the two are exact inverses of each other; on the null spaces, where no inverse exists, $\mathbf{A}^+$ returns zero. Only a square, invertible $\mathbf{A}$ gives $\mathbf{A}^+=\mathbf{A}^{-1}=\mathbf{V}\boldsymbol{\Sigma}^{-1}\mathbf{U}^\top$. Penrose showed that $\mathbf{A}^+$ is the unique matrix satisfying $\mathbf{A}\mathbf{A}^+\mathbf{A}=\mathbf{A}$, $\mathbf{A}^+\mathbf{A}\mathbf{A}^+=\mathbf{A}^+$, with $\mathbf{A}\mathbf{A}^+$ and $\mathbf{A}^+\mathbf{A}$ symmetric.[^penrose1955] These two products are the orthogonal projections onto $C(\mathbf{A})$ and $C(\mathbf{A}^\top)$.

**Minimum-norm least squares.** Consider $\min_{\mathbf{x}}\lVert\mathbf{A}\mathbf{x}-\mathbf{b}\rVert^2$, for instance linear regression with the data matrix $\mathbf{X}$ in place of $\mathbf{A}$. If the columns of $\mathbf{A}$ are linearly dependent (duplicated or collinear features, or fewer equations than unknowns, $m<n$), then $\mathbf{A}^\top\mathbf{A}$ is singular and infinitely many $\mathbf{x}$ attain the minimum: adding any null-space vector to a minimizer does not change $\mathbf{A}\mathbf{x}$. The pseudo-inverse picks one:

$$
\mathbf{x}^+=\mathbf{A}^+\mathbf{b}\ \text{ is the least-squares solution with the smallest norm }\lVert\mathbf{x}\rVert.
$$

*Why.* In the SVD coordinates $\tilde{\mathbf{x}}=\mathbf{V}^\top\mathbf{x}$ and $\tilde{\mathbf{b}}=\mathbf{U}^\top\mathbf{b}$, orthogonal invariance turns the loss into $\sum_{i\le r}(\sigma_i\tilde x_i-\tilde b_i)^2+\sum_{i>r}\tilde b_i^2$. The first $r$ coordinates are forced, $\tilde x_i=\tilde b_i/\sigma_i$. The remaining $\tilde x_{r+1},\dots,\tilde x_n$ (the null-space directions) do not appear in the loss at all, and setting them to zero minimizes $\lVert\mathbf{x}\rVert=\lVert\tilde{\mathbf{x}}\rVert$. That choice is exactly $\mathbf{x}^+$, and it lies in the row space. When $\mathbf{A}$ has full column rank there is nothing to choose, and $\mathbf{A}^+=(\mathbf{A}^\top\mathbf{A})^{-1}\mathbf{A}^\top$ recovers the [[Linear Models#Solving it: the normal equations|normal equations]].

> [!example] Example: the rank-one matrix again
> For $\mathbf{B}$ above, the reduced SVD has a single term, so
>
> $$
> \mathbf{B}^+=\frac{1}{\sigma_1}\mathbf{v}_1\mathbf{u}_1^\top=\frac{1}{3\sqrt5}\cdot\frac{1}{\sqrt5}\begin{bmatrix}1\\2\end{bmatrix}\frac13\begin{bmatrix}1&2&2\end{bmatrix}=\frac{1}{45}\begin{bmatrix}1&2&2\\2&4&4\end{bmatrix}.
> $$
>
> Take $\mathbf{b}=(9,9,9)$. Since $\mathbf{B}\mathbf{x}=(x_1+2x_2)\,\mathbf{c}$, the best we can do is the projection of $\mathbf{b}$ onto $\mathbf{c}=(1,2,2)$, which is $\tfrac{\mathbf{c}^\top\mathbf{b}}{\mathbf{c}^\top\mathbf{c}}\mathbf{c}=5\mathbf{c}=(5,10,10)$, with squared residual $\lVert(4,-1,-1)\rVert^2=18$. Every $\mathbf{x}$ on the line $x_1+2x_2=5$ achieves it. The pseudo-inverse gives $\mathbf{x}^+=\mathbf{B}^+\mathbf{b}=(1,2)$, the point of that line closest to the origin ($\lVert\mathbf{x}^+\rVert^2=5$), and indeed a multiple of the row-space direction $(1,2)$.

In regression, $\mathbf{w}=\mathbf{X}^+\mathbf{y}$ is what `np.linalg.lstsq` returns for rank-deficient data. Ridge regression, $(\mathbf{X}^\top\mathbf{X}+\rho\mathbf{I})^{-1}\mathbf{X}^\top\mathbf{y}$, replaces each $1/\sigma_i$ by $\sigma_i/(\sigma_i^2+\rho)$, which tends to the pseudo-inverse as $\rho\to0$ and stays bounded when $\sigma_i$ is tiny; this is the stable cousin of truncating small singular values.

### Randomized linear algebra (a glimpse)

For a matrix with millions of rows, even $O(mn\min(m,n))$ is too slow. A surprising modern idea is to **sample**: a random sketch of a matrix often keeps the information we need, with error guarantees in expectation or with high probability.[^halko2011]

**Sampling a matrix product.** Matrix multiplication is a sum of rank-one pieces, $\mathbf{A}\mathbf{B}=\sum_{t=1}^{n}\mathbf{a}_t\mathbf{b}_t^\top$ (column $t$ of $\mathbf{A}$ times row $t$ of $\mathbf{B}$). Pick $s$ indices $t_1,\dots,t_s$ independently with probabilities $p_t$, and rescale:

$$
\mathbf{A}\mathbf{B}\ \approx\ \frac1s\sum_{j=1}^{s}\frac{\mathbf{a}_{t_j}\mathbf{b}_{t_j}^\top}{p_{t_j}}=\mathbf{A}\boldsymbol{\Pi}\boldsymbol{\Pi}^\top\mathbf{B}.
$$

Here $\boldsymbol{\Pi}\in\mathbb{R}^{n\times s}$ has column $j$ equal to $\mathbf{e}_{t_j}/\sqrt{s\,p_{t_j}}$. Each term has expectation $\sum_tp_t\,\mathbf{a}_t\mathbf{b}_t^\top/p_t=\mathbf{A}\mathbf{B}$, so the estimate is **unbiased**; equivalently $\mathbb{E}[\boldsymbol{\Pi}\boldsymbol{\Pi}^\top]=\mathbf{I}$. Its mean squared error is

$$
\mathbb{E}\lVert\mathbf{A}\mathbf{B}-\mathbf{A}\boldsymbol{\Pi}\boldsymbol{\Pi}^\top\mathbf{B}\rVert_F^2=\frac1s\Big(\sum_t\frac{\lVert\mathbf{a}_t\rVert^2\lVert\mathbf{b}_t\rVert^2}{p_t}-\lVert\mathbf{A}\mathbf{B}\rVert_F^2\Big),
$$

which falls like $1/s$ and is minimized by $p_t\propto\lVert\mathbf{a}_t\rVert\,\lVert\mathbf{b}_t\rVert$ (a Cauchy–Schwarz argument).[^drineas2006] For $\mathbf{A}\mathbf{A}^\top$ this is **norm-squared sampling**, $p_t\propto\lVert\mathbf{a}_t\rVert^2$: heavy columns matter more, so sample them more often. Uniform sampling can be terrible when a few columns dominate (Figure 6a). The same idea, with leverage scores instead of norms, drives the [[#CUR decomposition|CUR decomposition]] below.

**Randomized SVD.** To approximate the top-$k$ singular triplets:

1. draw a Gaussian test matrix $\boldsymbol{\Omega}\in\mathbb{R}^{n\times(k+\ell)}$, with a small oversampling $\ell$ such as 5 or 10;
2. form the sketch $\mathbf{Y}=\mathbf{A}\boldsymbol{\Omega}$, whose columns are random combinations of the columns of $\mathbf{A}$ and so concentrate in its dominant column space;
3. orthonormalize its columns into $\mathbf{Q}$ (a QR factorization), so that $\mathbf{Q}\mathbf{Q}^\top\mathbf{A}\approx\mathbf{A}$;
4. compute the SVD of the small $(k+\ell)\times n$ matrix $\mathbf{Q}^\top\mathbf{A}=\hat{\mathbf{U}}\boldsymbol{\Sigma}\mathbf{V}^\top$, and set $\mathbf{U}=\mathbf{Q}\hat{\mathbf{U}}$.

For $k,\ell\ge2$, Halko, Martinsson and Tropp prove that $\mathbb{E}\lVert\mathbf{A}-\mathbf{Q}\mathbf{Q}^\top\mathbf{A}\rVert_F\le\sqrt{1+k/(\ell-1)}\,\big(\sum_{i>k}\sigma_i^2\big)^{1/2}$: within a small factor of the Eckart–Young optimum.[^halko2011] When the singular values decay slowly, $q$ **power iterations** (sketching $(\mathbf{A}\mathbf{A}^\top)^q\mathbf{A}$ instead, re-orthonormalizing in between) sharpen the gap $\sigma_k/\sigma_{k+1}$ to its $(2q+1)$-th power, the same mechanism as [[Eigenvalues and Eigenvectors#Power iteration and the QR algorithm|power iteration]]. The cost is dominated by a few passes of matrix products with $\mathbf{A}$, which parallelize well; scikit-learn's `utils.extmath.randomized_svd` and `PCA(svd_solver="randomized")` use this algorithm.

![[Machine Learning/images/matrix-factorization-svd-randomized.svg]]
*Figure 6: (a) estimating $\mathbf{A}\mathbf{A}^\top$ for a $40\times400$ matrix with very uneven column norms from $s$ sampled columns. Norm-squared sampling has relative error $0.21$ at $s=40$ and falls like $1/\sqrt s$; uniform sampling has error $1.64$, worse than the answer itself. (b) Randomized SVD of a $600\times400$ matrix with slowly decaying $\sigma_i=1/i$. Without power iterations ($q=0$) the error at $k=20$ is $1.67$ times the optimum $\sigma_{k+1}$; one or two power iterations bring it to $1.04$ and $1.01$.*

> [!example]- Code: sampled matrix products and randomized SVD in NumPy
> ```python
> import numpy as np
>
> def sampled_product(A, B, s, probs, rng):
>     """Unbiased estimate of A @ B from s sampled column-row pairs, each rescaled by 1/sqrt(s p_k)."""
>     idx = rng.choice(A.shape[1], size=s, p=probs)
>     scale = 1.0 / np.sqrt(s * probs[idx])
>     return (A[:, idx] * scale) @ (B[idx] * scale[:, None])
>
> def randomized_svd(A, k, p=5, q=1, rng=None):
>     """Halko-Martinsson-Tropp randomized SVD with q power iterations."""
>     rng = np.random.default_rng(rng)
>     Q, _ = np.linalg.qr(A @ rng.standard_normal((A.shape[1], k + p)))
>     for _ in range(q):                       # re-orthonormalize between products for stability
>         Q, _ = np.linalg.qr(A.T @ Q)
>         Q, _ = np.linalg.qr(A @ Q)
>     Ub, s, Vt = np.linalg.svd(Q.T @ A, full_matrices=False)   # small (k+p) x n problem
>     return (Q @ Ub)[:, :k], s[:k], Vt[:k]
>
> rng = np.random.default_rng(0)
> A = rng.normal(size=(40, 300)) * rng.lognormal(0, 1.2, size=300)  # very uneven column norms
> p = np.linalg.norm(A, axis=0) ** 2; p /= p.sum()                    # norm-squared sampling
> u = np.full(300, 1 / 300)                                           # uniform sampling
> for probs in (p, u):                                                # relative error of one 50-column sketch
>     print(np.linalg.norm(sampled_product(A, A.T, 50, probs, rng) - A @ A.T) / np.linalg.norm(A @ A.T))
>
> L = rng.normal(size=(500, 8)) @ rng.normal(size=(8, 300)) + 0.01 * rng.normal(size=(500, 300))
> U, s, Vt = randomized_svd(L, 8, rng=1)                              # nearly rank 8
> print(np.allclose(s, np.linalg.svd(L, compute_uv=False)[:8], rtol=1e-6))  # True
> ```
> Tested against Monte Carlo averages (unbiasedness, $\mathbb{E}[\boldsymbol{\Pi}\boldsymbol{\Pi}^\top]=\mathbf{I}$, the exact MSE formula) and against `np.linalg.svd` on a matrix with known spectrum.

## Use case: recommender systems

Let $\mathbf{M}\in\mathbb{R}^{m\times n}$ hold ratings: $M_{ij}\in\{1,\dots,5\}$ is how user $j$ rated movie $i$. Most entries are **missing**, because each user rates few movies. Our goal is to predict the missing entries and recommend the highest-predicted unseen movies.

**The SVD idea.** Assume tastes are driven by a few latent factors, a handful of "genre-like" directions, so that $\mathbf{M}$ is approximately low-rank. A simple recipe:

1. subtract $3$ from all observed ratings, so they lie in $[-2,2]$;
2. treat the missing entries as $0$ (the "average" rating);
3. compute $\mathbf{M}_k=\mathbf{U}_k\boldsymbol{\Sigma}_k\mathbf{V}_k^\top$ with $k$ around $16$–$20$, and read off $(\mathbf{M}_k)_{ij}+3$ as the predicted rating.

By Eckart–Young, $\mathbf{M}_k$ is the best rank-$k$ fit to the filled-in matrix. The $k$ latent dimensions often *resemble* genres, but they are learned, not literal genres.

**Why that's only a baseline.** Filling missing entries with $0$ tells the SVD that every unseen movie was rated "average". With 85–99% missing entries, those fake zeros dominate the fit and shrink every prediction towards the middle. The midpoint $3$ is also not the mean rating, and users and movies have systematic **biases** (generous raters, universally loved films). The fix is to **fit the factorization only to observed entries** $\mathcal{O}$, with regularization and biases:[^koren2009]

$$
\begin{aligned}
\min_{\mathbf{p},\mathbf{h},b,d}\ &\sum_{(i,j)\in\mathcal{O}}\Big(M_{ij}-\mu-b_i-d_j-\mathbf{p}_i^\top\mathbf{h}_j\Big)^2\\
&+\rho\Big(\sum_i\lVert\mathbf{p}_i\rVert^2+b_i^2+\sum_j\lVert\mathbf{h}_j\rVert^2+d_j^2\Big).
\end{aligned}
$$

This is no longer an SVD, because the missing entries break the structure. But **alternating least squares** (ALS) solves it efficiently. With all movie factors fixed, each user's $(\mathbf{h}_j,d_j)$ is an independent [[Linear Models#MAP estimation and regularization|ridge regression]] on the movies that user rated, and vice versa. SGD over the observed entries is the other standard choice.

> [!example]- Code: one sweep of alternating least squares
> ```python
> import numpy as np
> # assumes: R (m, n) observed ratings minus mu + b_i + d_j (anything where missing),
> #          mask (m, n) True where observed, U (m, k) movie factors p_i, V (n, k) user factors h_j, lam > 0
> def ridge(F, r, lam):                               # argmin_x ||F x - r||^2 + lam ||x||^2
>     return np.linalg.solve(F.T @ F + lam * np.eye(F.shape[1]), F.T @ r)
>
> for i in range(R.shape[0]):                         # user factors fixed: one ridge per movie
>     U[i] = ridge(V[mask[i]], R[i, mask[i]], lam)
> for j in range(R.shape[1]):                         # movie factors fixed: one ridge per user
>     V[j] = ridge(U[mask[:, j]], R[mask[:, j], j], lam)
> ```
> Here the biases are fitted first and held fixed, and the factors fit what is left, as in the code repository. Each half-sweep solves its subproblem exactly, so the regularized loss never increases.

![[Machine Learning/images/matrix-factorization-recommender.svg]]
*Figure 7: synthetic ratings (200 movies × 150 users, rank-4 structure plus biases and noise, 15% observed). Filling the missing entries and taking a rank-4 SVD barely beats predicting the global mean. Fitting biased rank-4 factors only to observed ratings (ALS) halves the held-out error.*

## Non-negative matrix factorization (NMF)

SVD factors may have any sign. Many data are inherently **non-negative**: pixel intensities, word counts, spectra, ratings. For such data a *signed* basis is hard to interpret, because components cancel each other. **NMF** insists on non-negative factors:[^leeseung1999]

$$
\min_{\mathbf{W}\ge0,\ \mathbf{H}\ge0}\ \lVert\mathbf{A}-\mathbf{W}\mathbf{H}\rVert_F^2,\qquad\mathbf{A}\in\mathbb{R}_{\ge0}^{m\times n},\ \mathbf{W}\in\mathbb{R}_{\ge0}^{m\times k},\ \mathbf{H}\in\mathbb{R}_{\ge0}^{k\times n}.
$$

Each data column is then an **additive** combination of the $k$ basis columns of $\mathbf{W}$, with non-negative weights from $\mathbf{H}$. Nothing can be subtracted. So the basis tends to represent **parts**: strokes of a digit, facial features, topics of a document (Figure 8).

**Multiplicative updates.** Split the gradient with respect to $\mathbf{H}$ into its positive and negative parts:

$$
\nabla_{\mathbf{H}}=2\big(\underbrace{\mathbf{W}^\top\mathbf{W}\mathbf{H}}_{\text{positive}}-\underbrace{\mathbf{W}^\top\mathbf{A}}_{\text{positive}}\big).
$$

Take a gradient step with a *per-entry* step size $\eta_{ab}=\frac{H_{ab}}{2(\mathbf{W}^\top\mathbf{W}\mathbf{H})_{ab}}$. The subtraction then cancels, and the step becomes a multiplication by the ratio of the two parts:

$$
\mathbf{H}\leftarrow\mathbf{H}\odot\frac{\mathbf{W}^\top\mathbf{A}}{\mathbf{W}^\top\mathbf{W}\mathbf{H}},\qquad\mathbf{W}\leftarrow\mathbf{W}\odot\frac{\mathbf{A}\mathbf{H}^\top}{\mathbf{W}\mathbf{H}\mathbf{H}^\top},
$$

where $\odot$ and the fractions act elementwise. The factors stay non-negative automatically: positive times a ratio of positives. Lee and Seung proved that the loss **never increases** under these updates.[^leeseung2001] The problem is non-convex, so NMF finds local optima, and its factors are not unique (any positive rescaling or permutation works). Initialization matters.

> [!example]- Code: NMF with multiplicative updates
> ```python
> import numpy as np
>
> def nmf(A, k, n_iter=500, eps=1e-10, rng=None):
>     """A: (m, n) non-negative. Returns W (m, k) and H (k, n), both non-negative."""
>     rng = np.random.default_rng(rng)
>     W, H = rng.random((A.shape[0], k)), rng.random((k, A.shape[1]))
>     for _ in range(n_iter):
>         H *= (W.T @ A) / (W.T @ W @ H + eps)        # eps guards against 0 / 0
>         W *= (A @ H.T) / (W @ H @ H.T + eps)
>     return W, H
> ```

![[Machine Learning/images/matrix-factorization-nmf-vs-pca.svg]]
*Figure 8: eight components learned from $8\times8$ handwritten digits. NMF's non-negative components are localized strokes: parts that add up to digits. PCA's components are global signed patterns, mixing positive (red) and negative (blue) pixels, and they only make sense in combination.*

## CUR decomposition

SVD, and to a lesser extent NMF, produce basis vectors that are *combinations* of the data. A singular vector of a gene-expression matrix is a weighted mix of thousands of genes, and no biologist can measure "eigen-gene 3". **CUR** builds the approximation from **actual columns and rows** of $\mathbf{A}$:[^mahoney2009]

$$
\mathbf{A}\approx\mathbf{C}\,\mathbf{U}_{\mathrm{CUR}}\,\mathbf{R},\qquad\mathbf{C}=\text{some columns of }\mathbf{A},\quad\mathbf{R}=\text{some rows of }\mathbf{A}.
$$

The factors $\mathbf{C}$ and $\mathbf{R}$ are then directly interpretable: these genes, these patients. They also keep properties of the data, such as sparsity and non-negativity.

**Which columns?** Not uniformly random ones, which can miss rare but important directions. Sample by **statistical leverage**. With $\mathbf{v}_1,\dots,\mathbf{v}_k$ the top right singular vectors, column $j$ gets probability

$$
p_j=\frac1k\sum_{\xi=1}^{k}\big(v_{\xi j}\big)^2,\qquad\sum_jp_j=1.
$$

$p_j$ is large when column $j$ contributes strongly to the dominant subspace. Rows are sampled the same way, using the left singular vectors.

> [!example] Algorithm: CUR with leverage-score sampling
> 1. Compute the top-$k$ singular vectors, or approximate them, and the column and row leverage scores.
> 2. Sample $s$ columns with probabilities $p_j$ to form $\mathbf{C}$, and $s'$ rows with their leverage scores to form $\mathbf{R}$.
> 3. Set $\mathbf{U}_{\mathrm{CUR}}=\mathbf{C}^+\mathbf{A}\mathbf{R}^+$, the best middle matrix for these $\mathbf{C}$ and $\mathbf{R}$ in Frobenius norm.

With $s,s'=O(k\log k/\varepsilon^2)$, this gives $\lVert\mathbf{A}-\mathbf{C}\mathbf{U}_{\mathrm{CUR}}\mathbf{R}\rVert_F\le(2+\varepsilon)\lVert\mathbf{A}-\mathbf{A}_k\rVert_F$ with high probability. So we lose little accuracy compared with the optimal SVD, and gain interpretability.

![[Machine Learning/images/matrix-factorization-cur.svg]]
*Figure 9: CUR error relative to the best rank-8 approximation, as more columns and rows are sampled. Leverage-score sampling beats uniform sampling at every size, especially when few columns are sampled. With 30–40 columns, CUR matches or beats the rank-8 SVD (its rank can exceed 8), using actual data columns.*

| | SVD | NMF | CUR |
|---|---|---|---|
| Factors | orthonormal, signed | non-negative | actual columns and rows |
| Optimal? | yes (Eckart–Young) | local optimum | near-optimal with high probability |
| Interpretability | low (global mixtures) | parts-based | direct (real features and samples) |
| Unique? | yes (if the $\sigma_i$ are distinct) | no | depends on sampling |

---

# Part II: Dimensionality reduction

Many features are redundant or noisy. Reducing them saves computation: [[Linear Models#Solving it: the normal equations|ordinary least squares]] costs $O(ND^2+D^3)$ in the number of features $D$. It also fights the curse of dimensionality and allows visualization. We could ask a domain expert which features to drop (*feature selection*). Here we look for automated methods that build a few new features (*feature extraction*) while keeping most of the information.

## Principal component analysis (PCA)

Let $\mathbf{X}\in\mathbb{R}^{N\times D}$ hold the data, one point per row. **Always centre first**: subtract each feature's mean, giving $\bar{\mathbf{X}}$ with rows $\bar{\mathbf{x}}_i$.

### Maximum variance

Project the data onto a direction $\mathbf{w}\in\mathbb{R}^D$. This is a direction in *feature* space, i.e. in the row space of $\mathbf{X}$. Each point becomes a number, $z_i=\mathbf{w}^\top\bar{\mathbf{x}}_i$. Different directions keep different amounts of information. In a bad direction the projections pile on top of each other and points become indistinguishable. In a good one they stay spread out (Figure 10). So we choose the direction that **maximizes the spread** (variance) of the projections. After centring, the variance is just the mean of $z_i^2$.

Scaling $\mathbf{w}$ up would inflate the variance for free, so we restrict to unit vectors:

$$
\mathbf{w}_1=\arg\max_{\lVert\mathbf{w}\rVert=1}\sum_i\big(\mathbf{w}^\top\bar{\mathbf{x}}_i\big)^2=\arg\max_{\lVert\mathbf{w}\rVert=1}\lVert\bar{\mathbf{X}}\mathbf{w}\rVert^2=\arg\max_{\mathbf{w}}\frac{\mathbf{w}^\top\mathbf{S}\mathbf{w}}{\mathbf{w}^\top\mathbf{w}},
$$

where $\mathbf{S}=\frac{1}{N-1}\bar{\mathbf{X}}^\top\bar{\mathbf{X}}$ is the **sample covariance matrix**. The constant factor does not change the argmax. To solve it, use a [[Multivariable Calculus#Constrained optimization: Lagrange multipliers|Lagrange multiplier]]. Stationarity of $\mathbf{w}^\top\mathbf{S}\mathbf{w}-\lambda(\mathbf{w}^\top\mathbf{w}-1)$ gives

$$
\mathbf{S}\mathbf{w}=\lambda\mathbf{w}.
$$

Every stationary point is an **eigenvector** of $\mathbf{S}$, and there the objective equals $\mathbf{w}^\top\mathbf{S}\mathbf{w}=\lambda$. So the maximum is the eigenvector with the **largest eigenvalue**: the **first principal component**.

![[Machine Learning/images/matrix-factorization-pca-projections.svg]]
*Figure 10: left, centred 2-D data with the two principal directions. Top right, the projected variance $\mathbf{w}^\top\mathbf{S}\mathbf{w}$ as a function of the direction's angle. It is maximized at PC1 (dashed), with value $\lambda_1$. Bottom right, projections onto PC1 stay spread out; projections onto PC2 collapse together and lose information.*

### More components: deflation

Having found $\mathbf{w}_1$, the rank-1 approximation $\bar{\mathbf{X}}\mathbf{w}_1\mathbf{w}_1^\top$ captures what PC1 explains. **Deflate**: remove it and find the direction of maximum variance in what remains.

$$
\hat{\mathbf{X}}_1=\bar{\mathbf{X}}-\bar{\mathbf{X}}\mathbf{w}_1\mathbf{w}_1^\top,\qquad\mathbf{w}_2=\arg\max_{\lVert\mathbf{w}\rVert=1}\lVert\hat{\mathbf{X}}_1\mathbf{w}\rVert^2.
$$

Since $\hat{\mathbf{X}}_1^\top\hat{\mathbf{X}}_1/(N-1)=\mathbf{S}-\lambda_1\mathbf{w}_1\mathbf{w}_1^\top$ (deflation removes exactly the top eigenpair), $\mathbf{w}_2$ is the top eigenvector of what is left. That is the **second** eigenvector of $\mathbf{S}$, automatically orthogonal to $\mathbf{w}_1$. Repeating gives the general result:

> [!note] Principal components
> The $k$-th principal component is the eigenvector of $\mathbf{S}$ with the $k$-th largest eigenvalue. With $\mathbf{W}=[\mathbf{w}_1\cdots\mathbf{w}_K]\in\mathbb{R}^{D\times K}$ (orthonormal columns), the reduced dataset is $\mathbf{Z}=\bar{\mathbf{X}}\mathbf{W}\in\mathbb{R}^{N\times K}$, and the variance captured by component $k$ is $\lambda_k$.[^pearson1901]<sup>,</sup>[^hotelling1933]

### Minimum reconstruction error: the same answer

Here is another angle. With an orthonormal $\mathbf{W}\in\mathbb{R}^{D\times K}$, approximate each point by its projection, $\hat{\mathbf{x}}_i=\mathbf{W}\mathbf{W}^\top\bar{\mathbf{x}}_i=\sum_{j\le K}z_{ij}\mathbf{w}_j$, and minimize the **total squared reconstruction error**. The sum over components sits *inside* the norm:

$$
\min_{\mathbf{W}^\top\mathbf{W}=\mathbf{I}}\ \sum_i\Big\lVert\bar{\mathbf{x}}_i-\sum_{j=1}^{K}z_{ij}\mathbf{w}_j\Big\rVert^2=\min_{\mathbf{W}^\top\mathbf{W}=\mathbf{I}}\ \sum_i\lVert\bar{\mathbf{x}}_i-\mathbf{W}\mathbf{W}^\top\bar{\mathbf{x}}_i\rVert^2.
$$

For $K=1$, expand using $\mathbf{w}^\top\mathbf{w}=1$:

$$
\begin{aligned}
\sum_i\lVert\bar{\mathbf{x}}_i-\mathbf{w}\mathbf{w}^\top\bar{\mathbf{x}}_i\rVert^2&=\sum_i\Big(\bar{\mathbf{x}}_i^\top\bar{\mathbf{x}}_i-2(\mathbf{w}^\top\bar{\mathbf{x}}_i)^2+(\mathbf{w}^\top\bar{\mathbf{x}}_i)^2\Big)\\
&=\underbrace{\sum_i\lVert\bar{\mathbf{x}}_i\rVert^2}_{\text{constant}}-\sum_i(\mathbf{w}^\top\bar{\mathbf{x}}_i)^2.
\end{aligned}
$$

**Minimizing the error is maximizing the variance.** For general $K$ the error is $\sum_i\lVert\bar{\mathbf{x}}_i\rVert^2-(N-1)\operatorname{tr}(\mathbf{W}^\top\mathbf{S}\mathbf{W})$, which is minimized by the top-$K$ eigenvectors (Ky Fan's theorem). The two views agree, and PCA's subspace is the best $K$-dimensional linear approximation of the centred data.

### PCA through the SVD

Parts I and II meet here. Take the thin SVD of the centred data, $\bar{\mathbf{X}}=\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$. Then

$$
\mathbf{S}=\frac{1}{N-1}\bar{\mathbf{X}}^\top\bar{\mathbf{X}}=\mathbf{V}\,\frac{\boldsymbol{\Sigma}^2}{N-1}\,\mathbf{V}^\top.
$$

- **The principal components are the right singular vectors**, $\mathbf{W}=\mathbf{V}_K$.
- **The variances are** $\lambda_k=\sigma_k^2/(N-1)$.
- **The scores are** $\mathbf{Z}=\bar{\mathbf{X}}\mathbf{V}_K=\mathbf{U}_K\boldsymbol{\Sigma}_K$.
- **PCA is Eckart–Young applied to the data:** the PCA reconstruction $\mathbf{Z}\mathbf{V}_K^\top$ is the best rank-$K$ approximation of $\bar{\mathbf{X}}$.

In practice PCA is computed through the SVD rather than by forming $\mathbf{S}$. Squaring the data matrix squares its condition number and loses precision.

> [!example]- Code: PCA through the SVD
> ```python
> import numpy as np
> # assumes: X (N, D) data, K number of components
> mu = X.mean(axis=0)
> U, s, Vt = np.linalg.svd(X - mu, full_matrices=False)    # thin SVD of the centred data
> W = Vt[:K].T                                             # principal directions, (D, K)
> Z = (X - mu) @ W                                         # scores; equal to U[:, :K] * s[:K]
> lam = s ** 2 / (len(X) - 1)                              # variances lambda_k
> kept = lam[:K].sum() / lam.sum()                         # fraction of variance kept
> X_hat = Z @ W.T + mu                                     # best rank-K reconstruction
> ```

**How many components?** The fraction of variance kept by the first $K$ components is $\sum_{k\le K}\lambda_k/\sum_k\lambda_k$. Plot the per-component fractions (a **scree plot**) and keep enough components to reach a target, such as 90%, or stop at an elbow (Figure 11).

![[Machine Learning/images/matrix-factorization-scree.svg]]
*Figure 11: PCA of $8\times8$ handwritten digits (64 pixel features). The first few components carry most of the variance, and 21 components keep 90% of it.*

**Practical notes.**

- **Reconstruction** is $\hat{\mathbf{X}}=\mathbf{Z}\mathbf{W}^\top+\boldsymbol{\mu}$. **New points** are projected with the *training* mean and $\mathbf{W}$: $\mathbf{z}=\mathbf{W}^\top(\mathbf{x}-\boldsymbol{\mu})$.
- **Standardize** the features (divide by their standard deviations) when they are on different scales. Otherwise PCA simply finds the features with the biggest units.
- **Few samples, many features ($N\ll D$).** A gene-expression dataset might have $D\sim10^3$ genes and $N\sim10$ patients. The centred data has rank at most $N-1$, so projecting onto the top $N-1$ components loses **nothing**. It is also cheaper to get them from the $N\times N$ Gram matrix $\bar{\mathbf{X}}\bar{\mathbf{X}}^\top$ ("dual PCA") than from the $D\times D$ covariance, which is the stepping stone to kernel PCA.
- **Limitations.** PCA is linear, so it cannot unfold curved structure (Figure 14). It is sensitive to outliers, and it is unsupervised: the highest-variance directions need not be the ones that separate classes. For that, see [[Discriminant Analysis#Fisher's linear discriminant|Fisher's discriminant]], which maximizes a *between-class / within-class* Rayleigh quotient instead of $\mathbf{w}^\top\mathbf{S}\mathbf{w}/\mathbf{w}^\top\mathbf{w}$.

## Autoencoders

In PCA, the reduced representation $\mathbf{z}=\mathbf{W}^\top\bar{\mathbf{x}}$ is a linear function of the input, an **encoder**. Going back, $\hat{\mathbf{x}}=\mathbf{W}\mathbf{z}$, is a **decoder**. $\mathbf{W}$ is not square, so it has no inverse. But its pseudo-inverse is simply $\mathbf{W}^+=\mathbf{W}^\top$, because its columns are orthonormal. Generalize both maps to learnable functions: an encoder $f_\theta$ and a decoder $g_\psi$, with a narrow bottleneck $\mathbf{z}$ in between (Figure 12). Train them to reproduce the input:

$$
\min_{\theta,\psi}\ \sum_i\big\lVert g_\psi\big(f_\theta(\bar{\mathbf{x}}_i)\big)-\bar{\mathbf{x}}_i\big\rVert^2.
$$

$\mathbf{z}_i=f_\theta(\bar{\mathbf{x}}_i)$ is the compressed, *latent* representation, and $\hat{\mathbf{x}}_i=g_\psi(\mathbf{z}_i)$ the reconstruction. With neural networks as $f_\theta$ and $g_\psi$, autoencoders learn non-linear representations. Undercomplete, denoising and sparse variants are common. Probabilistic versions, the [[Variational Autoencoders|variational autoencoders]] of a later lecture, turn this into a generative model.

![[Machine Learning/images/matrix-factorization-autoencoder.svg]]
*Figure 12: an autoencoder squeezes the input through a $K$-dimensional bottleneck $\mathbf{z}$ and is trained so that the reconstruction matches the input.*

**Linear autoencoders are PCA.** If $f_\theta$ and $g_\psi$ are linear and the loss is squared error, the optimal autoencoder spans **the same subspace as the top-$K$ principal components**.[^baldihornik1989] It need not be the individual components, nor orthonormal: any invertible $K\times K$ mixing of them is equally good. Non-linearity is what lets autoencoders go beyond PCA.

## Kernel PCA

PCA finds linear structure, but we can again run it in a feature space $\phi(\mathbf{x})$ given only a [[Support Vector Machines#Kernels|kernel]] $\kappa(\mathbf{x},\mathbf{x}')=\phi(\mathbf{x})^\top\phi(\mathbf{x}')$. The kernel must satisfy Mercer's condition (symmetric, with PSD Gram matrices).[^scholkopf1998] Assume for now that the features are centred, $\sum_j\phi(\mathbf{x}_j)=\mathbf{0}$. The feature-space covariance and its eigenvectors are

$$
\mathbf{S}_\phi=\frac{1}{N-1}\sum_j\phi(\mathbf{x}_j)\phi(\mathbf{x}_j)^\top,\qquad\mathbf{S}_\phi\mathbf{v}_i=\lambda_i\mathbf{v}_i.
$$

Can we find $\mathbf{v}_i$ and $\lambda_i$ without ever computing $\mathbf{S}_\phi$ or $\phi$? Yes. Rearranging the eigen-equation,

$$
\mathbf{v}_i=\frac{1}{\lambda_i(N-1)}\sum_j\phi(\mathbf{x}_j)\big(\phi(\mathbf{x}_j)^\top\mathbf{v}_i\big)=\sum_j\alpha_{ij}\,\phi(\mathbf{x}_j).
$$

So every $\mathbf{v}_i$ with $\lambda_i>0$ lies in the **span of the mapped data**, with coefficients $\boldsymbol{\alpha}_i\in\mathbb{R}^N$. Substitute back, take the inner product with $\phi(\mathbf{x}_l)$, and write $\mathbf{G}_{lj}=\kappa(\mathbf{x}_l,\mathbf{x}_j)$:

$$
\frac{1}{N-1}\mathbf{G}\mathbf{G}\boldsymbol{\alpha}_i=\lambda_i\mathbf{G}\boldsymbol{\alpha}_i\quad\Longleftarrow\quad\mathbf{G}\boldsymbol{\alpha}_i=\lambda_i(N-1)\boldsymbol{\alpha}_i.
$$

Every solution of the right-hand equation solves the left one. The parts of $\boldsymbol{\alpha}_i$ in the null space of $\mathbf{G}$ don't change $\mathbf{v}_i$, so we may simply solve the $N\times N$ **eigenproblem of the Gram matrix**. Three practical steps complete the algorithm:

1. **Centre in feature space.** The assumption $\sum_j\phi(\mathbf{x}_j)=\mathbf{0}$ is not free. Use the centred Gram matrix $\tilde{\mathbf{G}}=\mathbf{G}-\mathbf{1}_N\mathbf{G}-\mathbf{G}\mathbf{1}_N+\mathbf{1}_N\mathbf{G}\mathbf{1}_N$, where $\mathbf{1}_N$ is the $N\times N$ matrix with every entry $1/N$.
2. **Normalize.** $\lVert\mathbf{v}_i\rVert=1$ requires $\boldsymbol{\alpha}_i^\top\tilde{\mathbf{G}}\boldsymbol{\alpha}_i=1$, so rescale each eigenvector of $\tilde{\mathbf{G}}$ by $1/\sqrt{\text{its eigenvalue}}$.
3. **Project.** The $i$-th kernel principal component of a point $\mathbf{x}$ is $z_i(\mathbf{x})=\mathbf{v}_i^\top\phi(\mathbf{x})=\sum_j\alpha_{ij}\,\kappa(\mathbf{x},\mathbf{x}_j)$. For new points, centre their kernel values consistently.

> [!example]- Code: kernel PCA from a Gram matrix
> ```python
> import numpy as np
>
> def kernel_pca(Kmat, n_comp):
>     """Kmat: (N, N) Gram matrix of the training points. Returns their kernel PC scores, (N, n_comp)."""
>     N = len(Kmat)
>     one = np.full((N, N), 1.0 / N)
>     Kc = Kmat - one @ Kmat - Kmat @ one + one @ Kmat @ one     # 1. centre in feature space
>     vals, vecs = np.linalg.eigh(Kc)                            # ascending eigenvalues
>     vals, vecs = vals[::-1][:n_comp], vecs[:, ::-1][:, :n_comp]
>     alphas = vecs / np.sqrt(vals)                              # 2. normalize: alpha^T Kc alpha = 1
>     return Kc @ alphas                                         # 3. project: sum_j alpha_ij Kc[l, j]
> ```
> With a linear kernel, `Kmat = X @ X.T`, the scores equal ordinary PCA's up to the sign of each component.

![[Machine Learning/images/matrix-factorization-kernel-pca.svg]]
*Figure 13: two concentric rings. Linear PCA can only rotate the data, so the rings stay nested. Kernel PCA with an RBF kernel maps them to separate regions, where a linear method can split them.*

The cost is that of the $N\times N$ eigenproblem, $O(N^3)$, and mapping components back to input space (the *pre-image* problem) is hard. With a linear kernel, kernel PCA reproduces ordinary PCA exactly. The same Gram-matrix tricks appear in [[Clustering and Expectation Maximization#Kernel K-Means|kernel K-Means]].

## Other dimensionality-reduction methods

- **Classical multidimensional scaling (MDS).** Given only pairwise distances, double-centre the matrix of squared distances to get a Gram matrix, then take its top eigenvectors. With Euclidean distances, this is exactly PCA, computed from the dual (Gram) side.
- **Laplacian eigenmaps.** Build a $k$-nearest-neighbour graph and embed each point using the bottom non-trivial eigenvectors of the **graph Laplacian**, the same $\mathbf{s}^\top\mathbf{L}\mathbf{s}$ machinery as [[Clustering and Expectation Maximization#Spectral clustering|spectral clustering]].[^belkin2003] Neighbours stay close, so curved manifolds get "unrolled".
- **t-SNE and UMAP.** These match *neighbour probabilities* between the original space and a 2-D map. The result is excellent visualizations of cluster structure.[^vandermaaten2008]<sup>,</sup>[^mcinnes2018] But distances between far-apart clusters, and cluster sizes, are not meaningful, so use them for exploration, not as features.

![[Machine Learning/images/matrix-factorization-other-dr.png]]
*Figure 14: a Swiss roll (top left) embedded in 2-D, coloured by position along the roll. PCA squashes it, folding distant parts of the roll on top of each other. Laplacian eigenmaps order the points by their position along the roll. t-SNE keeps local neighbourhoods but tears the global structure into pieces.*

## Key takeaways

- **Symmetric** matrices have $\mathbf{S}=\mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^\top$ with an orthonormal $\mathbf{Q}$, a sum of rank-1 layers (eigenvalues, diagonalization and power iteration are in [[Eigenvalues and Eigenvectors]]).
- **SVD** $\mathbf{A}=\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$ exists for every matrix: rotate, scale, rotate. $\lVert\mathbf{A}\rVert_2=\sigma_1$, $\lVert\mathbf{A}\rVert_F^2=\sum\sigma_i^2$, and $\mathbf{A}^+=\mathbf{V}\boldsymbol{\Sigma}^+\mathbf{U}^\top$.
- **Eckart–Young:** the truncated SVD is the best rank-$k$ approximation, with errors $\sigma_{k+1}$ and $\sqrt{\sum_{i>k}\sigma_i^2}$. Proving optimality needs the null-space (or Weyl) argument, not just the truncation error.
- **Recommenders:** fit low-rank factors with biases **on observed entries only** (ALS = alternating ridge regressions). Filling the missing entries in first biases everything.
- **NMF** gives parts-based, non-negative factors via monotone multiplicative updates. **CUR** uses actual columns and rows, sampled by leverage, and is near-optimal with high probability.
- **PCA** gives the same answer two ways (max variance and min reconstruction error): the top eigenvectors of $\mathbf{S}$, i.e. the right singular vectors of $\bar{\mathbf{X}}$. Choose $K$ with a scree plot.
- **Linear autoencoders** recover the PCA subspace. **Kernel PCA** works with the centred Gram matrix. **Laplacian eigenmaps** and **t-SNE** preserve neighbourhoods rather than variance.

## Where this shows up next

- [[Clustering and Expectation Maximization]], the next lecture: spectral clustering (Laplacian eigenvectors) and kernel K-Means. Probabilistic PCA is a linear-Gaussian latent-variable model fitted with EM.
- [[Graph Representation Learning]]: node embeddings as low-rank factorizations of a proximity matrix, and Laplacian eigenmaps on graphs.
- [[Variational Autoencoders]]: probabilistic autoencoders built on the ELBO.
- Back in [[Linear Models]] and [[Discriminant Analysis]]: least squares, the pseudo-inverse and ridge regression (which shrinks along principal directions by $\sigma_i^2/(\sigma_i^2+\rho)$), and Fisher's discriminant, the supervised counterpart of PCA.

<!-- References: footnote definitions below render as the numbered "References" list at the bottom of the page -->

[^strang2020svd]: Strang, G. (2020). *Part 5: Singular Values and Singular Vectors*. A 2020 Vision of Linear Algebra, MIT OpenCourseWare RES.18-010. [video](https://www.youtube.com/watch?v=IHO7_n7Y09s), [course page](https://ocw.mit.edu/courses/res-18-010-a-2020-vision-of-linear-algebra-spring-2020/)
[^strang2023]: Strang, G. (2023). *Introduction to Linear Algebra* (6th ed.), Chapter 7: The Singular Value Decomposition. Wellesley-Cambridge Press. [book website](https://math.mit.edu/linearalgebra/)
[^3b1b-lintrans]: 3Blue1Brown (2016). *Linear transformations and matrices | Chapter 3, Essence of linear algebra*. [video](https://www.youtube.com/watch?v=kYB8IZa5AuE)
[^golubkahan1965]: Golub, G., & Kahan, W. (1965). *Calculating the Singular Values and Pseudo-Inverse of a Matrix*. Journal of the Society for Industrial and Applied Mathematics, Series B: Numerical Analysis, 2(2), 205–224. [doi:10.1137/0702016](https://doi.org/10.1137/0702016)
[^penrose1955]: Penrose, R. (1955). *A Generalized Inverse for Matrices*. Mathematical Proceedings of the Cambridge Philosophical Society, 51(3), 406–413. [doi:10.1017/S0305004100030401](https://doi.org/10.1017/S0305004100030401)
[^drineas2006]: Drineas, P., Kannan, R., & Mahoney, M. W. (2006). *Fast Monte Carlo Algorithms for Matrices I: Approximating Matrix Multiplication*. SIAM Journal on Computing, 36(1), 132–157. [doi:10.1137/S0097539704442684](https://doi.org/10.1137/S0097539704442684)
[^halko2011]: Halko, N., Martinsson, P.-G., & Tropp, J. A. (2011). *Finding Structure with Randomness: Probabilistic Algorithms for Constructing Approximate Matrix Decompositions*. SIAM Review, 53(2), 217–288. [doi:10.1137/090771806](https://doi.org/10.1137/090771806)
[^trefethen1997]: Trefethen, L. N., & Bau, D. (1997). *Numerical Linear Algebra*. SIAM. [doi:10.1137/1.9780898719574](https://doi.org/10.1137/1.9780898719574)
[^eckartyoung1936]: Eckart, C., & Young, G. (1936). *The Approximation of One Matrix by Another of Lower Rank*. Psychometrika, 1, 211–218. [doi:10.1007/BF02288367](https://doi.org/10.1007/BF02288367)
[^mirsky1960]: Mirsky, L. (1960). *Symmetric Gauge Functions and Unitarily Invariant Norms*. The Quarterly Journal of Mathematics, 11(1), 50–59. [doi:10.1093/qmath/11.1.50](https://doi.org/10.1093/qmath/11.1.50)
[^koren2009]: Koren, Y., Bell, R., & Volinsky, C. (2009). *Matrix Factorization Techniques for Recommender Systems*. Computer, 42(8), 30–37. [doi:10.1109/MC.2009.263](https://doi.org/10.1109/MC.2009.263)
[^leeseung1999]: Lee, D. D., & Seung, H. S. (1999). *Learning the Parts of Objects by Non-negative Matrix Factorization*. Nature, 401, 788–791. [doi:10.1038/44565](https://doi.org/10.1038/44565)
[^leeseung2001]: Lee, D. D., & Seung, H. S. (2001). *Algorithms for Non-negative Matrix Factorization*. NeurIPS. [link](https://papers.nips.cc/paper_files/paper/2000/hash/f9d1152547c0bde01830b7e8bd60024c-Abstract.html)
[^mahoney2009]: Mahoney, M. W., & Drineas, P. (2009). *CUR Matrix Decompositions for Improved Data Analysis*. PNAS, 106(3), 697–702. [doi:10.1073/pnas.0803205106](https://doi.org/10.1073/pnas.0803205106)
[^pearson1901]: Pearson, K. (1901). *On Lines and Planes of Closest Fit to Systems of Points in Space*. Philosophical Magazine, 2(11), 559–572. [doi:10.1080/14786440109462720](https://doi.org/10.1080/14786440109462720)
[^hotelling1933]: Hotelling, H. (1933). *Analysis of a Complex of Statistical Variables into Principal Components*. Journal of Educational Psychology, 24(6), 417–441. [doi:10.1037/h0071325](https://doi.org/10.1037/h0071325)
[^baldihornik1989]: Baldi, P., & Hornik, K. (1989). *Neural Networks and Principal Component Analysis: Learning from Examples Without Local Minima*. Neural Networks, 2(1), 53–58. [doi:10.1016/0893-6080(89)90014-2](https://doi.org/10.1016/0893-6080(89)90014-2)
[^scholkopf1998]: Schölkopf, B., Smola, A., & Müller, K.-R. (1998). *Nonlinear Component Analysis as a Kernel Eigenvalue Problem*. Neural Computation, 10(5), 1299–1319. [doi:10.1162/089976698300017467](https://doi.org/10.1162/089976698300017467)
[^belkin2003]: Belkin, M., & Niyogi, P. (2003). *Laplacian Eigenmaps for Dimensionality Reduction and Data Representation*. Neural Computation, 15(6), 1373–1396. [doi:10.1162/089976603321780317](https://doi.org/10.1162/089976603321780317)
[^vandermaaten2008]: van der Maaten, L., & Hinton, G. (2008). *Visualizing Data using t-SNE*. JMLR, 9, 2579–2605. [link](https://jmlr.org/papers/v9/vandermaaten08a.html)
[^mcinnes2018]: McInnes, L., Healy, J., & Melville, J. (2018). *UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction*. [arXiv:1802.03426](https://arxiv.org/abs/1802.03426)
