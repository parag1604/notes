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

> **Prerequisites:** linear algebra (eigenvalues, orthogonality, rank), [[Linear Models]] (least squares, ridge regression), and [[Support Vector Machines#Kernels|kernels]].
>
> **TL;DR:**
> - **Matrix factorization** writes a matrix as a product of simpler pieces. The **spectral decomposition** $\mathbf{S}=\mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^\top$ handles symmetric matrices, and the **SVD** $\mathbf{A}=\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$ handles *every* matrix.
> - Truncating the SVD gives the **best low-rank approximation** (Eckart–Young). This powers compression, denoising and recommender systems. **NMF** and **CUR** trade a little accuracy for interpretability.
> - **Dimensionality reduction** keeps the directions that matter. **PCA** keeps the directions of maximum variance, which are the top right singular vectors of the centred data. **Autoencoders** and **kernel PCA** make it non-linear, and manifold methods (Laplacian eigenmaps, t-SNE) preserve neighbourhoods instead of variance.

# Part I: Matrix factorization

Matrix factorization is a powerful tool for many practical problems. We want to write a general $\mathbf{A}\in\mathbb{R}^{m\times n}$ as a product of matrices with simple structure (orthogonal, diagonal, triangular, non-negative, low-rank). We start with square matrices.

## Eigenvalues and similar matrices

The eigenvalues of a square $\mathbf{A}\in\mathbb{R}^{n\times n}$ solve the characteristic equation $\det(\mathbf{A}-\lambda\mathbf{I})=0$, and the eigenvectors solve $\mathbf{A}\mathbf{x}_i=\lambda_i\mathbf{x}_i$. Counted with multiplicity (and over $\mathbb{C}$), there are always exactly $n$ eigenvalues.

Computing eigenpairs directly is involved, but **similar matrices** help. Two matrices are **similar** if

$$
\mathbf{B}=\mathbf{M}^{-1}\mathbf{A}\mathbf{M}\qquad\text{for some invertible }\mathbf{M}.
$$

Similar matrices have the same characteristic polynomial, hence the **same eigenvalues**. Their eigenvectors are related by $\mathbf{M}$. The converse is false: $\begin{pmatrix}1&1\\0&1\end{pmatrix}$ and $\mathbf{I}_2$ have the same eigenvalues but are not similar. The trick is to choose $\mathbf{M}$ so that $\mathbf{B}$ is **triangular**; then the eigenvalues can be read off the diagonal. With a unitary $\mathbf{M}$ this is always possible (the **Schur form**), and it can be computed iteratively, as described [[#How eigenvectors are actually computed|below]].

**Diagonalization.** If $\mathbf{A}$ has $n$ linearly independent eigenvectors, stack them as $\mathbf{X}=[\mathbf{x}_1\cdots\mathbf{x}_n]$. Then $\mathbf{A}\mathbf{X}=\mathbf{X}\boldsymbol{\Lambda}$, and $\mathbf{X}$ is invertible, so

$$
\mathbf{A}=\mathbf{X}\boldsymbol{\Lambda}\mathbf{X}^{-1}.
$$

This is about having enough independent **eigenvectors**, which is unrelated to $\mathbf{A}$ being invertible. $\begin{pmatrix}1&1\\0&1\end{pmatrix}$ is invertible but not diagonalizable, while $\operatorname{diag}(1,0)$ is singular yet diagonal. For a diagonalizable matrix of rank $r$, exactly $r$ eigenvalues are non-zero. For non-diagonalizable matrices even that fails: $\begin{pmatrix}0&1\\0&0\end{pmatrix}$ has rank 1, but both its eigenvalues are $0$.

## The spectral theorem (symmetric matrices)

Symmetric matrices, $\mathbf{S}=\mathbf{S}^\top$, are the nice case. Their eigenvalues are real, and their eigenvectors can be chosen **orthonormal**. That gives the **spectral decomposition**:

$$
\mathbf{S}=\mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^\top=\lambda_1\mathbf{q}_1\mathbf{q}_1^\top+\lambda_2\mathbf{q}_2\mathbf{q}_2^\top+\cdots+\lambda_r\mathbf{q}_r\mathbf{q}_r^\top.
$$

Each $\mathbf{q}_i\mathbf{q}_i^\top$ is an $n\times n$ **rank-1** matrix, so $\mathbf{S}$ is a sum of $r=\operatorname{rank}(\mathbf{S})$ rank-1 pieces. Keeping only the non-zero eigenvalues gives the *thin* form, with $\mathbf{Q}\in\mathbb{R}^{n\times r}$. That $\mathbf{Q}$ need not be square, though it always has orthonormal columns. To check the expansion, use orthonormality:

$$
\mathbf{S}\mathbf{q}_i=\sum_j\lambda_j\mathbf{q}_j\big(\mathbf{q}_j^\top\mathbf{q}_i\big)=\lambda_i\mathbf{q}_i,
$$

since every cross term vanishes. If $\mathbf{S}$ is **positive semi-definite** ($\mathbf{x}^\top\mathbf{S}\mathbf{x}\ge0$ for all $\mathbf{x}$), all $\lambda_i\ge0$. We conventionally order them $\lambda_1\ge\lambda_2\ge\cdots\ge0$.

Building $\mathbf{S}$ term by term gives a sequence of **rank-1 updates**:

$$
\mathbf{S}_1=\lambda_1\mathbf{q}_1\mathbf{q}_1^\top,\qquad\mathbf{S}_k=\mathbf{S}_{k-1}+\lambda_k\mathbf{q}_k\mathbf{q}_k^\top.
$$

[[#Best low-rank approximation: Eckart–Young|Below]] we show that, for PSD $\mathbf{S}$, $\mathbf{S}_k$ is the best rank-$k$ approximation of $\mathbf{S}$.

### How eigenvectors are actually computed

**Power iteration.** Repeatedly multiply a random vector by $\mathbf{S}$ and normalize: $\mathbf{v}\leftarrow\mathbf{S}\mathbf{v}/\lVert\mathbf{S}\mathbf{v}\rVert$. Write the starting vector in the eigenbasis, $\mathbf{v}_0=\sum_ic_i\mathbf{q}_i$. Then $\mathbf{S}^t\mathbf{v}_0=\sum_ic_i\lambda_i^t\mathbf{q}_i$, and the top term dominates. The error shrinks like $\lvert\lambda_2/\lambda_1\rvert^t$ (Figure 1). The Rayleigh quotient $\mathbf{v}^\top\mathbf{S}\mathbf{v}$ then estimates $\lambda_1$. This is how the first principal component is often computed for huge data, because it needs only matrix–vector products.

![[Machine Learning/images/matrix-factorization-power-iteration.svg]]
*Figure 1: power iteration on a random $30\times30$ covariance matrix. The distance to the top eigenvector falls geometrically, at the predicted rate $(\lambda_2/\lambda_1)^t$.*

**The QR algorithm.** To get *all* eigenvalues, factor $\mathbf{A}_t=\mathbf{Q}_t\mathbf{R}_t$ (orthogonal times upper-triangular) and recombine in reverse, $\mathbf{A}_{t+1}=\mathbf{R}_t\mathbf{Q}_t$. Since $\mathbf{A}_{t+1}=\mathbf{Q}_t^\top\mathbf{A}_t\mathbf{Q}_t$, every iterate is **similar** to $\mathbf{A}$, with the same eigenvalues. The iterates converge to the triangular Schur form, whose diagonal holds the eigenvalues. This is exactly the "choose $\mathbf{M}$ to make $\mathbf{B}$ triangular, iteratively" idea from above. Production solvers add shifts and a first reduction to tridiagonal or Hessenberg form, and for large sparse matrices use Lanczos or Arnoldi iterations.[^trefethen1997]

## Singular value decomposition (SVD)

The spectral decomposition needs a square, symmetric matrix. For a general $\mathbf{A}\in\mathbb{R}^{m\times n}$, $\mathbf{A}\mathbf{x}=\lambda\mathbf{x}$ does not even type-check. Instead we look for pairs of vectors that $\mathbf{A}$ maps onto each other:

$$
\mathbf{A}\mathbf{v}_i=\sigma_i\mathbf{u}_i,\qquad\mathbf{v}_i\in\mathbb{R}^n,\ \mathbf{u}_i\in\mathbb{R}^m,\ \sigma_i\ge0.
$$

The $\sigma_i$ are the **singular values**, the rectangular counterpart of eigenvalues. Even for square matrices they are not the absolute eigenvalues, unless the matrix is normal. Stacking $r=\operatorname{rank}(\mathbf{A})$ such pairs gives $\mathbf{A}\mathbf{V}=\mathbf{U}\boldsymbol{\Sigma}$, i.e.

$$
\boxed{\;\mathbf{A}=\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top=\sum_{i=1}^{r}\sigma_i\,\mathbf{u}_i\mathbf{v}_i^\top\;}
$$

**Where $\mathbf{U}$ and $\mathbf{V}$ come from.** If such a factorization exists with orthonormal $\mathbf{U}$ and $\mathbf{V}$, then

$$
\mathbf{A}\mathbf{A}^\top=\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top\mathbf{V}\boldsymbol{\Sigma}^\top\mathbf{U}^\top=\mathbf{U}\boldsymbol{\Sigma}\boldsymbol{\Sigma}^\top\mathbf{U}^\top,\qquad\mathbf{A}^\top\mathbf{A}=\mathbf{V}\boldsymbol{\Sigma}^\top\boldsymbol{\Sigma}\mathbf{V}^\top.
$$

Both products are symmetric PSD, so the spectral theorem applies. The **right singular vectors** $\mathbf{V}$ are eigenvectors of $\mathbf{A}^\top\mathbf{A}$, the **left singular vectors** $\mathbf{U}$ are eigenvectors of $\mathbf{A}\mathbf{A}^\top$, and $\sigma_i=\sqrt{\lambda_i(\mathbf{A}^\top\mathbf{A})}$. In the *full* SVD, $\boldsymbol{\Sigma}$ is $m\times n$, so $\boldsymbol{\Sigma}\boldsymbol{\Sigma}^\top$ ($m\times m$) and $\boldsymbol{\Sigma}^\top\boldsymbol{\Sigma}$ ($n\times n$) have different sizes. They share the same non-zero diagonal entries $\sigma_i^2$. In the *thin* SVD all three middle matrices are $r\times r$.

**$\mathbf{U}$ really is orthonormal.** Choose orthonormal eigenvectors $\mathbf{v}_i$ of $\mathbf{A}^\top\mathbf{A}$, and *define* $\mathbf{u}_i=\mathbf{A}\mathbf{v}_i/\sigma_i$ for $\sigma_i>0$. Then

$$
\mathbf{u}_i^\top\mathbf{u}_j=\frac{\mathbf{v}_i^\top\mathbf{A}^\top\mathbf{A}\mathbf{v}_j}{\sigma_i\sigma_j}=\frac{\sigma_j^2\,\mathbf{v}_i^\top\mathbf{v}_j}{\sigma_i\sigma_j}=\begin{cases}1&i=j,\\0&i\neq j.\end{cases}
$$

For the full SVD, complete $\mathbf{U}$ with any orthonormal basis of the null space of $\mathbf{A}^\top$. This construction works for **every** matrix: the SVD always exists.

**Geometry: rotate, scale, rotate.** Read $\mathbf{A}\mathbf{x}=\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top\mathbf{x}$ from right to left (Figure 2):

1. $\mathbf{V}^\top$ rotates (possibly reflects) the input, so that the right singular vectors align with the axes;
2. $\boldsymbol{\Sigma}$ stretches each axis by $\sigma_i$;
3. $\mathbf{U}$ rotates the result into place.

Every linear map sends the unit sphere to an ellipsoid, with semi-axes $\sigma_i\mathbf{u}_i$.

![[Machine Learning/images/matrix-factorization-svd-geometry.svg]]
*Figure 2: the SVD of a $2\times2$ matrix as rotate, scale, rotate. The right singular vectors (arrows) are first rotated onto the axes, then stretched by $\sigma_1,\sigma_2$, then rotated to become the ellipse's axes $\sigma_1\mathbf{u}_1,\sigma_2\mathbf{u}_2$.*

### Norms through the SVD

**Frobenius norm and trace.** The Frobenius norm is $\lVert\mathbf{A}\rVert_F=\sqrt{\sum_{ij}a_{ij}^2}=\sqrt{\operatorname{Tr}(\mathbf{A}\mathbf{A}^\top)}$. For complex matrices, use the conjugate transpose $\mathbf{A}^H$. On a $2\times2$ example, $\operatorname{Tr}(\mathbf{A}\mathbf{A}^\top)=a_{11}^2+a_{12}^2+a_{21}^2+a_{22}^2$. Two facts we use repeatedly:

- the **cyclic property**, $\operatorname{Tr}(\mathbf{A}\mathbf{B})=\operatorname{Tr}(\mathbf{B}\mathbf{A})$;
- **orthogonal invariance**: $\lVert\mathbf{Q}_1\mathbf{A}\mathbf{Q}_2\rVert_F=\lVert\mathbf{A}\rVert_F$ for orthogonal $\mathbf{Q}_1,\mathbf{Q}_2$.

Combining them gives $\lVert\mathbf{A}\rVert_F^2=\operatorname{Tr}(\boldsymbol{\Sigma}\boldsymbol{\Sigma}^\top)=\sum_i\sigma_i^2$. Two other norms come straight from the singular values:

- the **spectral norm** $\lVert\mathbf{A}\rVert_2=\max_{\lVert\mathbf{x}\rVert=1}\lVert\mathbf{A}\mathbf{x}\rVert=\sigma_1$;
- the **nuclear norm** $\lVert\mathbf{A}\rVert_*=\sigma_1+\cdots+\sigma_r$.

All three are *unitarily invariant*.

## Best low-rank approximation: Eckart–Young

> [!note] Theorem (Eckart–Young–Mirsky)
> Let $\mathbf{A}=\sum_i\sigma_i\mathbf{u}_i\mathbf{v}_i^\top$ and $\mathbf{A}_k=\sum_{i=1}^{k}\sigma_i\mathbf{u}_i\mathbf{v}_i^\top$. Among all matrices of rank **at most** $k$, $\mathbf{A}_k$ minimizes $\lVert\mathbf{A}-\mathbf{M}\rVert$ for every unitarily invariant norm, with
> $$
> \lVert\mathbf{A}-\mathbf{A}_k\rVert_2=\sigma_{k+1},\qquad\lVert\mathbf{A}-\mathbf{A}_k\rVert_F=\sqrt{\textstyle\sum_{i>k}\sigma_i^2}.
> $$
> For a symmetric matrix, the same holds with $\mathbf{S}_k$ built from the $k$ eigenvalues of **largest magnitude**. For PSD $\mathbf{S}$ these are simply the largest.[^eckartyoung1936]<sup>,</sup>[^mirsky1960]

**Step 1: the error of the truncation.** For the symmetric case, write $\mathbf{S}=\mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^\top$ with a square orthogonal $\mathbf{Q}$, and pad $\mathbf{S}_k=\mathbf{Q}\bar{\boldsymbol{\Lambda}}_k\mathbf{Q}^\top$, where $\bar{\boldsymbol{\Lambda}}_k=\operatorname{diag}(\lambda_1,\dots,\lambda_k,0,\dots,0)$. By orthogonal invariance,

$$
\epsilon_k=\lVert\mathbf{S}-\mathbf{S}_k\rVert_F^2=\lVert\mathbf{Q}(\boldsymbol{\Lambda}-\bar{\boldsymbol{\Lambda}}_k)\mathbf{Q}^\top\rVert_F^2=\lVert\boldsymbol{\Lambda}-\bar{\boldsymbol{\Lambda}}_k\rVert_F^2=\sum_{i=k+1}^{r}\lambda_i^2.
$$

The same computation gives $\sum_{i>k}\sigma_i^2$ for the SVD. This shows how good the truncation is. It does **not** yet show that no other rank-$k$ matrix does better. That is the real content of the theorem, and needs one more idea.

**Step 2: nothing beats it in the spectral norm.** Let $\mathbf{M}$ have rank at most $k$. Its null space has dimension at least $n-k$, and $\operatorname{span}(\mathbf{v}_1,\dots,\mathbf{v}_{k+1})$ has dimension $k+1$. Two subspaces whose dimensions add up to more than $n$ must intersect, so there is a unit vector $\mathbf{x}$ in both. Writing $\mathbf{x}=\sum_{i\le k+1}c_i\mathbf{v}_i$ with $\sum_ic_i^2=1$:

$$
\lVert\mathbf{A}-\mathbf{M}\rVert_2^2\ \ge\ \lVert(\mathbf{A}-\mathbf{M})\mathbf{x}\rVert^2=\lVert\mathbf{A}\mathbf{x}\rVert^2=\sum_{i\le k+1}\sigma_i^2c_i^2\ \ge\ \sigma_{k+1}^2.
$$

$\mathbf{A}_k$ achieves this bound, so it is optimal in $\lVert\cdot\rVert_2$.

**Step 3: nothing beats it in the Frobenius norm.** The same null-space argument proves **Weyl's inequality** for singular values, $\sigma_{i+j-1}(\mathbf{X}+\mathbf{Y})\le\sigma_i(\mathbf{X})+\sigma_j(\mathbf{Y})$. Apply it with $\mathbf{X}=\mathbf{A}-\mathbf{M}$, $\mathbf{Y}=\mathbf{M}$ and $j=k+1$, noting that $\sigma_{k+1}(\mathbf{M})=0$ because $\operatorname{rank}\mathbf{M}\le k$:

$$
\begin{aligned}
&\sigma_{i+k}(\mathbf{A})\le\sigma_i(\mathbf{A}-\mathbf{M})\quad\text{for all }i\\
\Longrightarrow\quad&\lVert\mathbf{A}-\mathbf{M}\rVert_F^2=\sum_i\sigma_i(\mathbf{A}-\mathbf{M})^2\ \ge\ \sum_i\sigma_{i+k}(\mathbf{A})^2=\lVert\mathbf{A}-\mathbf{A}_k\rVert_F^2.\qquad\blacksquare
\end{aligned}
$$

![[Machine Learning/images/matrix-factorization-image-compression.png]]
*Figure 3: Eckart–Young as image compression. A rank-$k$ approximation stores only $k(m+n+1)$ numbers instead of $mn$. Rank 20 is already recognizable from under 8% of the numbers, and rank 50 is hard to tell apart from the original.*

![[Machine Learning/images/matrix-factorization-singular-values.svg]]
*Figure 4: left, the singular values of the photo decay quickly, which is why low rank works. Right, the measured truncation error matches the Eckart–Young formula $\sqrt{\sum_{i>k}\sigma_i^2}$ exactly.*

### The pseudo-inverse

The SVD also inverts what can be inverted. For $\mathbf{A}=\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$ of rank $r$, define $\boldsymbol{\Sigma}^+$ ($n\times m$, the transposed shape) by inverting the non-zero singular values and leaving the zeros:

$$
\mathbf{A}^+=\mathbf{V}\boldsymbol{\Sigma}^+\mathbf{U}^\top,\qquad\boldsymbol{\Sigma}^+=\operatorname{diag}\Big(\frac{1}{\sigma_1},\dots,\frac{1}{\sigma_r},0,\dots,0\Big).
$$

Only a square, invertible $\mathbf{A}$ gives $\mathbf{A}^+=\mathbf{A}^{-1}=\mathbf{V}\boldsymbol{\Sigma}^{-1}\mathbf{U}^\top$. In general, $\mathbf{A}^+\mathbf{b}$ is the **minimum-norm least-squares** solution of $\mathbf{A}\mathbf{x}\approx\mathbf{b}$. It is exactly the tool for rank-deficient [[Linear Models#Solving it: the normal equations|least squares]].

## Use case: recommender systems

Let $\mathbf{M}\in\mathbb{R}^{m\times n}$ hold ratings: $M_{ij}\in\{1,\dots,5\}$ is how user $j$ rated movie $i$. Most entries are **missing**, because each user rates few movies. Our goal is to predict the missing entries and recommend the highest-predicted unseen movies.

**The SVD idea.** Assume tastes are driven by a few latent factors, a handful of "genre-like" directions, so that $\mathbf{M}$ is approximately low-rank. A simple recipe:

1. subtract $3$ from all observed ratings, so they lie in $[-2,2]$;
2. treat the missing entries as $0$ (the "average" rating);
3. compute $\mathbf{M}_k=\mathbf{U}_k\boldsymbol{\Sigma}_k\mathbf{V}_k^\top$ with $k$ around $16$–$20$, and read off $(\mathbf{M}_k)_{ij}+3$ as the predicted rating.

By Eckart–Young, $\mathbf{M}_k$ is the best rank-$k$ fit to the filled-in matrix. The $k$ latent dimensions often *resemble* genres, but they are learned, not literal genres.

**Why that's only a baseline.** Filling missing entries with $0$ tells the SVD that every unseen movie was rated "average". With 85–99% missing entries, those fake zeros dominate the fit and shrink every prediction towards the middle. The midpoint $3$ is also not the mean rating, and users and movies have systematic **biases** (generous raters, universally loved films). The fix is to **fit the factorization only to observed entries** $\Omega$, with regularization and biases:[^koren2009]

$$
\min_{\mathbf{u},\mathbf{v},b,c}\ \sum_{(i,j)\in\Omega}\Big(M_{ij}-\mu-b_i-c_j-\mathbf{u}_i^\top\mathbf{v}_j\Big)^2+\lambda\Big(\sum_i\lVert\mathbf{u}_i\rVert^2+b_i^2+\sum_j\lVert\mathbf{v}_j\rVert^2+c_j^2\Big).
$$

This is no longer an SVD, because the missing entries break the structure. But **alternating least squares** (ALS) solves it efficiently. With all movie factors fixed, each user's $(\mathbf{u}_i,b_i)$ is an independent [[Linear Models#MAP estimation and regularization|ridge regression]] on the movies that user rated, and vice versa. SGD over the observed entries is the other standard choice.

![[Machine Learning/images/matrix-factorization-recommender.svg]]
*Figure 5: synthetic ratings (200 movies × 150 users, rank-4 structure plus biases and noise, 15% observed). Filling the missing entries and taking a rank-4 SVD barely beats predicting the global mean. Fitting biased rank-4 factors only to observed ratings (ALS) halves the held-out error.*

## Non-negative matrix factorization (NMF)

SVD factors may have any sign. Many data are inherently **non-negative**: pixel intensities, word counts, spectra, ratings. For such data a *signed* basis is hard to interpret, because components cancel each other. **NMF** insists on non-negative factors:[^leeseung1999]

$$
\min_{\mathbf{W}\ge0,\ \mathbf{H}\ge0}\ \lVert\mathbf{A}-\mathbf{W}\mathbf{H}\rVert_F^2,\qquad\mathbf{A}\in\mathbb{R}_{\ge0}^{m\times n},\ \mathbf{W}\in\mathbb{R}_{\ge0}^{m\times k},\ \mathbf{H}\in\mathbb{R}_{\ge0}^{k\times n}.
$$

Each data column is then an **additive** combination of the $k$ basis columns of $\mathbf{W}$, with non-negative weights from $\mathbf{H}$. Nothing can be subtracted. So the basis tends to represent **parts**: strokes of a digit, facial features, topics of a document (Figure 6).

**Multiplicative updates.** Split the gradient with respect to $\mathbf{H}$ into its positive and negative parts:

$$
\nabla_{\mathbf{H}}=2\big(\underbrace{\mathbf{W}^\top\mathbf{W}\mathbf{H}}_{\text{positive}}-\underbrace{\mathbf{W}^\top\mathbf{A}}_{\text{positive}}\big).
$$

Take a gradient step with a *per-entry* step size $\eta_{ab}=\frac{H_{ab}}{2(\mathbf{W}^\top\mathbf{W}\mathbf{H})_{ab}}$. The subtraction then cancels, and the step becomes a multiplication by the ratio of the two parts:

$$
\mathbf{H}\leftarrow\mathbf{H}\odot\frac{\mathbf{W}^\top\mathbf{A}}{\mathbf{W}^\top\mathbf{W}\mathbf{H}},\qquad\mathbf{W}\leftarrow\mathbf{W}\odot\frac{\mathbf{A}\mathbf{H}^\top}{\mathbf{W}\mathbf{H}\mathbf{H}^\top},
$$

where $\odot$ and the fractions act elementwise. The factors stay non-negative automatically: positive times a ratio of positives. Lee and Seung proved that the loss **never increases** under these updates.[^leeseung2001] The problem is non-convex, so NMF finds local optima, and its factors are not unique (any positive rescaling or permutation works). Initialization matters.

![[Machine Learning/images/matrix-factorization-nmf-vs-pca.svg]]
*Figure 6: eight components learned from $8\times8$ handwritten digits. NMF's non-negative components are localized strokes: parts that add up to digits. PCA's components are global signed patterns, mixing positive (red) and negative (blue) pixels, and they only make sense in combination.*

## CUR decomposition

SVD, and to a lesser extent NMF, produce basis vectors that are *combinations* of the data. A singular vector of a gene-expression matrix is a weighted mix of thousands of genes, and no biologist can measure "eigen-gene 3". **CUR** builds the approximation from **actual columns and rows** of $\mathbf{A}$:[^mahoney2009]

$$
\mathbf{A}\approx\mathbf{C}\,\mathbf{U}\,\mathbf{R},\qquad\mathbf{C}=\text{some columns of }\mathbf{A},\quad\mathbf{R}=\text{some rows of }\mathbf{A}.
$$

The factors $\mathbf{C}$ and $\mathbf{R}$ are then directly interpretable: these genes, these patients. They also keep properties of the data, such as sparsity and non-negativity.

**Which columns?** Not uniformly random ones, which can miss rare but important directions. Sample by **statistical leverage**. With $\mathbf{v}_1,\dots,\mathbf{v}_k$ the top right singular vectors, column $j$ gets probability

$$
p_j=\frac1k\sum_{\xi=1}^{k}\big(v_{\xi j}\big)^2,\qquad\sum_jp_j=1.
$$

$p_j$ is large when column $j$ contributes strongly to the dominant subspace. Rows are sampled the same way, using the left singular vectors.

> [!example] Algorithm: CUR with leverage-score sampling
> 1. Compute the top-$k$ singular vectors, or approximate them, and the column and row leverage scores.
> 2. Sample $c$ columns with probabilities $p_j$ to form $\mathbf{C}$, and $r$ rows with their leverage scores to form $\mathbf{R}$.
> 3. Set $\mathbf{U}=\mathbf{C}^+\mathbf{A}\mathbf{R}^+$, the best middle matrix for these $\mathbf{C}$ and $\mathbf{R}$ in Frobenius norm.

With $c,r=O(k\log k/\varepsilon^2)$, this gives $\lVert\mathbf{A}-\mathbf{C}\mathbf{U}\mathbf{R}\rVert_F\le(2+\varepsilon)\lVert\mathbf{A}-\mathbf{A}_k\rVert_F$ with high probability. So we lose little accuracy compared with the optimal SVD, and gain interpretability.

![[Machine Learning/images/matrix-factorization-cur.svg]]
*Figure 7: CUR error relative to the best rank-8 approximation, as more columns and rows are sampled. Leverage-score sampling beats uniform sampling at every size, especially when few columns are sampled. With 30–40 columns, CUR matches or beats the rank-8 SVD (its rank can exceed 8), using actual data columns.*

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

Project the data onto a direction $\mathbf{w}\in\mathbb{R}^D$. This is a direction in *feature* space, i.e. in the row space of $\mathbf{X}$. Each point becomes a number, $z_i=\mathbf{w}^\top\bar{\mathbf{x}}_i$. Different directions keep different amounts of information. In a bad direction the projections pile on top of each other and points become indistinguishable. In a good one they stay spread out (Figure 8). So we choose the direction that **maximizes the spread** (variance) of the projections. After centring, the variance is just the mean of $z_i^2$.

Scaling $\mathbf{w}$ up would inflate the variance for free, so we restrict to unit vectors:

$$
\mathbf{w}_1=\arg\max_{\lVert\mathbf{w}\rVert=1}\sum_i\big(\mathbf{w}^\top\bar{\mathbf{x}}_i\big)^2=\arg\max_{\lVert\mathbf{w}\rVert=1}\lVert\bar{\mathbf{X}}\mathbf{w}\rVert^2=\arg\max_{\mathbf{w}}\frac{\mathbf{w}^\top\mathbf{S}\mathbf{w}}{\mathbf{w}^\top\mathbf{w}},
$$

where $\mathbf{S}=\frac{1}{N-1}\bar{\mathbf{X}}^\top\bar{\mathbf{X}}$ is the **sample covariance matrix**. The constant factor does not change the argmax. To solve it, use the [[Support Vector Machines#The Lagrangian dual|Lagrangian]]. Stationarity of $\mathbf{w}^\top\mathbf{S}\mathbf{w}-\lambda(\mathbf{w}^\top\mathbf{w}-1)$ gives

$$
\mathbf{S}\mathbf{w}=\lambda\mathbf{w}.
$$

Every stationary point is an **eigenvector** of $\mathbf{S}$, and there the objective equals $\mathbf{w}^\top\mathbf{S}\mathbf{w}=\lambda$. So the maximum is the eigenvector with the **largest eigenvalue**: the **first principal component**.

![[Machine Learning/images/matrix-factorization-pca-projections.svg]]
*Figure 8: left, centred 2-D data with the two principal directions. Top right, the projected variance $\mathbf{w}^\top\mathbf{S}\mathbf{w}$ as a function of the direction's angle. It is maximized at PC1 (dashed), with value $\lambda_1$. Bottom right, projections onto PC1 stay spread out; projections onto PC2 collapse together and lose information.*

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

**Minimizing the error is maximizing the variance.** For general $K$ the error is $\sum_i\lVert\bar{\mathbf{x}}_i\rVert^2-(N-1)\operatorname{Tr}(\mathbf{W}^\top\mathbf{S}\mathbf{W})$, which is minimized by the top-$K$ eigenvectors (Ky Fan's theorem). The two views agree, and PCA's subspace is the best $K$-dimensional linear approximation of the centred data.

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

**How many components?** The fraction of variance kept by the first $K$ components is $\sum_{k\le K}\lambda_k/\sum_k\lambda_k$. Plot the per-component fractions (a **scree plot**) and keep enough components to reach a target, such as 90%, or stop at an elbow (Figure 9).

![[Machine Learning/images/matrix-factorization-scree.svg]]
*Figure 9: PCA of $8\times8$ handwritten digits (64 pixel features). The first few components carry most of the variance, and 21 components keep 90% of it.*

**Practical notes.**

- **Reconstruction** is $\hat{\mathbf{X}}=\mathbf{Z}\mathbf{W}^\top+\boldsymbol{\mu}$. **New points** are projected with the *training* mean and $\mathbf{W}$: $\mathbf{z}=\mathbf{W}^\top(\mathbf{x}-\boldsymbol{\mu})$.
- **Standardize** the features (divide by their standard deviations) when they are on different scales. Otherwise PCA simply finds the features with the biggest units.
- **Few samples, many features ($N\ll D$).** A gene-expression dataset might have $D\sim10^3$ genes and $N\sim10$ patients. The centred data has rank at most $N-1$, so projecting onto the top $N-1$ components loses **nothing**. It is also cheaper to get them from the $N\times N$ Gram matrix $\bar{\mathbf{X}}\bar{\mathbf{X}}^\top$ ("dual PCA") than from the $D\times D$ covariance, which is the stepping stone to kernel PCA.
- **Limitations.** PCA is linear, so it cannot unfold curved structure (Figure 12). It is sensitive to outliers, and it is unsupervised: the highest-variance directions need not be the ones that separate classes. For that, see [[Discriminant Analysis#Fisher's linear discriminant|Fisher's discriminant]], which maximizes a *between-class / within-class* Rayleigh quotient instead of $\mathbf{w}^\top\mathbf{S}\mathbf{w}/\mathbf{w}^\top\mathbf{w}$.

## Autoencoders

In PCA, the reduced representation $\mathbf{z}=\mathbf{W}^\top\bar{\mathbf{x}}$ is a linear function of the input, an **encoder**. Going back, $\hat{\mathbf{x}}=\mathbf{W}\mathbf{z}$, is a **decoder**. $\mathbf{W}$ is not square, so it has no inverse. But its pseudo-inverse is simply $\mathbf{W}^+=\mathbf{W}^\top$, because its columns are orthonormal. Generalize both maps to learnable functions: an encoder $f_\theta$ and a decoder $g_\phi$, with a narrow bottleneck $\mathbf{z}$ in between (Figure 10). Train them to reproduce the input:

$$
\min_{\theta,\phi}\ \sum_i\big\lVert g_\phi\big(f_\theta(\bar{\mathbf{x}}_i)\big)-\bar{\mathbf{x}}_i\big\rVert^2.
$$

$\mathbf{z}_i=f_\theta(\bar{\mathbf{x}}_i)$ is the compressed, *latent* representation, and $\hat{\mathbf{x}}_i=g_\phi(\mathbf{z}_i)$ the reconstruction. With neural networks as $f_\theta$ and $g_\phi$, autoencoders learn non-linear representations. Undercomplete, denoising and sparse variants are common. Probabilistic versions, the variational autoencoders of a later lecture, turn this into a generative model.

![[Machine Learning/images/matrix-factorization-autoencoder.svg]]
*Figure 10: an autoencoder squeezes the input through a $K$-dimensional bottleneck $\mathbf{z}$ and is trained so that the reconstruction matches the input.*

**Linear autoencoders are PCA.** If $f_\theta$ and $g_\phi$ are linear and the loss is squared error, the optimal autoencoder spans **the same subspace as the top-$K$ principal components**.[^baldihornik1989] It need not be the individual components, nor orthonormal: any invertible $K\times K$ mixing of them is equally good. Non-linearity is what lets autoencoders go beyond PCA.

## Kernel PCA

PCA finds linear structure, but we can again run it in a feature space $\phi(\mathbf{x})$ given only a [[Support Vector Machines#Kernels|kernel]] $\kappa(\mathbf{x},\mathbf{x}')=\phi(\mathbf{x})^\top\phi(\mathbf{x}')$. The kernel must satisfy Mercer's condition (symmetric, with PSD Gram matrices).[^scholkopf1998] Assume for now that the features are centred, $\sum_n\phi(\mathbf{x}_n)=\mathbf{0}$. The feature-space covariance and its eigenvectors are

$$
\mathbf{C}=\frac{1}{N-1}\sum_n\phi(\mathbf{x}_n)\phi(\mathbf{x}_n)^\top,\qquad\mathbf{C}\mathbf{v}_i=\lambda_i\mathbf{v}_i.
$$

Can we find $\mathbf{v}_i$ and $\lambda_i$ without ever computing $\mathbf{C}$ or $\phi$? Yes. Rearranging the eigen-equation,

$$
\mathbf{v}_i=\frac{1}{\lambda_i(N-1)}\sum_n\phi(\mathbf{x}_n)\big(\phi(\mathbf{x}_n)^\top\mathbf{v}_i\big)=\sum_n\alpha_{in}\,\phi(\mathbf{x}_n).
$$

So every $\mathbf{v}_i$ with $\lambda_i>0$ lies in the **span of the mapped data**, with coefficients $\boldsymbol{\alpha}_i\in\mathbb{R}^N$. Substitute back, take the inner product with $\phi(\mathbf{x}_l)$, and write $\mathbb{K}_{ln}=\kappa(\mathbf{x}_l,\mathbf{x}_n)$:

$$
\frac{1}{N-1}\mathbb{K}\mathbb{K}\boldsymbol{\alpha}_i=\lambda_i\mathbb{K}\boldsymbol{\alpha}_i\quad\Longleftarrow\quad\mathbb{K}\boldsymbol{\alpha}_i=\lambda_i(N-1)\boldsymbol{\alpha}_i.
$$

Every solution of the right-hand equation solves the left one. The parts of $\boldsymbol{\alpha}_i$ in the null space of $\mathbb{K}$ don't change $\mathbf{v}_i$, so we may simply solve the $N\times N$ **eigenproblem of the Gram matrix**. Three practical steps complete the algorithm:

1. **Centre in feature space.** The assumption $\sum_n\phi(\mathbf{x}_n)=\mathbf{0}$ is not free. Use the centred Gram matrix $\tilde{\mathbb{K}}=\mathbb{K}-\mathbf{1}_N\mathbb{K}-\mathbb{K}\mathbf{1}_N+\mathbf{1}_N\mathbb{K}\mathbf{1}_N$, where $\mathbf{1}_N$ is the $N\times N$ matrix with every entry $1/N$.
2. **Normalize.** $\lVert\mathbf{v}_i\rVert=1$ requires $\boldsymbol{\alpha}_i^\top\tilde{\mathbb{K}}\boldsymbol{\alpha}_i=1$, so rescale each eigenvector of $\tilde{\mathbb{K}}$ by $1/\sqrt{\text{its eigenvalue}}$.
3. **Project.** The $i$-th kernel principal component of a point $\mathbf{x}$ is $z_i(\mathbf{x})=\mathbf{v}_i^\top\phi(\mathbf{x})=\sum_n\alpha_{in}\,\kappa(\mathbf{x},\mathbf{x}_n)$. For new points, centre their kernel values consistently.

![[Machine Learning/images/matrix-factorization-kernel-pca.svg]]
*Figure 11: two concentric rings. Linear PCA can only rotate the data, so the rings stay nested. Kernel PCA with an RBF kernel maps them to separate regions, where a linear method can split them.*

The cost is that of the $N\times N$ eigenproblem, $O(N^3)$, and mapping components back to input space (the *pre-image* problem) is hard. With a linear kernel, kernel PCA reproduces ordinary PCA exactly. The same Gram-matrix tricks appear in [[Clustering and Expectation Maximization#Kernel K-Means|kernel K-Means]].

## Other dimensionality-reduction methods

- **Classical multidimensional scaling (MDS).** Given only pairwise distances, double-centre the matrix of squared distances to get a Gram matrix, then take its top eigenvectors. With Euclidean distances, this is exactly PCA, computed from the dual (Gram) side.
- **Laplacian eigenmaps.** Build a $k$-nearest-neighbour graph and embed each point using the bottom non-trivial eigenvectors of the **graph Laplacian**, the same $\mathbf{s}^\top\mathbf{L}\mathbf{s}$ machinery as [[Clustering and Expectation Maximization#Spectral clustering|spectral clustering]].[^belkin2003] Neighbours stay close, so curved manifolds get "unrolled".
- **t-SNE and UMAP.** These match *neighbour probabilities* between the original space and a 2-D map. The result is excellent visualizations of cluster structure.[^vandermaaten2008]<sup>,</sup>[^mcinnes2018] But distances between far-apart clusters, and cluster sizes, are not meaningful, so use them for exploration, not as features.

![[Machine Learning/images/matrix-factorization-other-dr.png]]
*Figure 12: a Swiss roll (top left) embedded in 2-D, coloured by position along the roll. PCA squashes it, folding distant parts of the roll on top of each other. Laplacian eigenmaps order the points by their position along the roll. t-SNE keeps local neighbourhoods but tears the global structure into pieces.*

## Key takeaways

- **Similar matrices** ($\mathbf{B}=\mathbf{M}^{-1}\mathbf{A}\mathbf{M}$) share eigenvalues. **Diagonalizable** means $n$ independent eigenvectors, not invertible. **Symmetric** matrices have $\mathbf{S}=\mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^\top$ with an orthonormal $\mathbf{Q}$.
- **Power iteration** finds the top eigenvector at rate $\lvert\lambda_2/\lambda_1\rvert^t$. The **QR algorithm** iterates similarity transforms to the Schur form.
- **SVD** $\mathbf{A}=\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$ exists for every matrix: rotate, scale, rotate. $\lVert\mathbf{A}\rVert_2=\sigma_1$, $\lVert\mathbf{A}\rVert_F^2=\sum\sigma_i^2$, and $\mathbf{A}^+=\mathbf{V}\boldsymbol{\Sigma}^+\mathbf{U}^\top$.
- **Eckart–Young:** the truncated SVD is the best rank-$k$ approximation, with errors $\sigma_{k+1}$ and $\sqrt{\sum_{i>k}\sigma_i^2}$. Proving optimality needs the null-space (or Weyl) argument, not just the truncation error.
- **Recommenders:** fit low-rank factors with biases **on observed entries only** (ALS = alternating ridge regressions). Filling the missing entries in first biases everything.
- **NMF** gives parts-based, non-negative factors via monotone multiplicative updates. **CUR** uses actual columns and rows, sampled by leverage, and is near-optimal with high probability.
- **PCA** gives the same answer two ways (max variance and min reconstruction error): the top eigenvectors of $\mathbf{S}$, i.e. the right singular vectors of $\bar{\mathbf{X}}$. Choose $K$ with a scree plot.
- **Linear autoencoders** recover the PCA subspace. **Kernel PCA** works with the centred Gram matrix. **Laplacian eigenmaps** and **t-SNE** preserve neighbourhoods rather than variance.

## Where this shows up next

- [[Linear Models]]: least squares, the pseudo-inverse, and ridge regression (which shrinks along principal directions by $\sigma_i^2/(\sigma_i^2+\lambda)$).
- [[Discriminant Analysis]]: Fisher's discriminant, the supervised counterpart of PCA.
- [[Clustering and Expectation Maximization]]: spectral clustering (Laplacian eigenvectors) and kernel K-Means. Probabilistic PCA is a linear-Gaussian latent-variable model fitted with EM.
- [[Variational Autoencoders]]: probabilistic autoencoders built on the ELBO.

<!-- References: footnote definitions below render as the numbered "References" list at the bottom of the page -->

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
