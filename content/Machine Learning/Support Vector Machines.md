---
title: Support Vector Machines
tags:
  - svm
  - kernels
  - convex-optimization
  - max-margin
prev: "[[Discriminant Analysis]]"
next: "[[Non-parametric Models]]"
---

> **Prerequisites:** [[Linear Models]] (linear classifiers, logistic regression, $L_2$ regularization) and [[Multivariable Calculus#Constrained optimization: Lagrange multipliers|constrained optimization]] (Lagrange multipliers, duality and the KKT conditions; study it first if it is new).
>
> **TL;DR:**
> - Among all hyperplanes that separate two classes, the SVM picks the one with the **largest margin**.
> - This is a convex quadratic program. Its **dual** depends on the data only through dot products $\mathbf{x}_i^\top\mathbf{x}_j$, and its solution depends only on the few **support vectors** on the margin.
> - Replacing those dot products by a **kernel** $K(\mathbf{x},\mathbf{x}')$ gives non-linear boundaries without ever computing the feature map.
> - The **soft-margin** SVM tolerates violations through slack variables, and is the same thing as $L_2$-regularized **hinge-loss** minimization.
>
> **Code:** [Support-Vector-Machines](https://github.com/parag1604/Support-Vector-Machines): hard-margin, soft-margin and kernel SVMs solved as quadratic programs with cvxopt.

## Setup

We have $\mathcal{D}=\{(\mathbf{x}_i,y_i)\}_{i=1}^{N}$ with $\mathbf{x}_i\in\mathbb{R}^D$. For SVMs it is convenient to label the two classes $y_i\in\{+1,-1\}$ instead of $\{0,1\}$. A linear classifier with parameters $\mathbf{w}$ and $b$ assigns a new, unknown point $\mathbf{u}$ to a side of the hyperplane $\mathbf{w}^\top\mathbf{x}+b=0$:

$$
\hat y(\mathbf{u})=\begin{cases}+1&\text{if }\mathbf{w}^\top\mathbf{u}+b\ge0,\\-1&\text{otherwise.}\end{cases}
$$

Without $b$, the *hyperplane* would be forced through the origin. $\mathbf{w}$ is its normal vector. The (signed) distance from a point $\mathbf{x}$ to the hyperplane is $\frac{\mathbf{w}^\top\mathbf{x}+b}{\lVert\mathbf{w}\rVert}$.

## Hard-margin SVM

### Which separating hyperplane?

Suppose the data are **linearly separable**. Then infinitely many hyperplanes separate them (Figure 1, left). Our real goal is to **generalize** to unseen points, so we pick the one that stays as far as possible from both classes: the **maximum-margin** hyperplane. Intuitively, a boundary with room to spare is robust to small perturbations of the data.[^cortesvapnik1995]

![[Machine Learning/images/support-vector-machines-max-margin.svg]]
*Figure 1: left, many hyperplanes separate the training data. Right, the SVM chooses the one with the widest "street" between the classes. The margin hyperplanes $\mathbf{w}^\top\mathbf{x}+b=\pm1$ (dashed) pass through the **support vectors** (circled), which alone determine the solution.*

### Formalizing the margin

We insist that every positive sample lies on or beyond a certain distance from the boundary, and likewise every negative sample. Since $\mathbf{w}$ and $b$ are free, rescaling them does not change the hyperplane. So without loss of generality we can fix that distance to "1" in units of $\mathbf{w}$:

$$
\mathbf{w}^\top\mathbf{x}_i+b\ge+1\ \text{ if }y_i=+1,\qquad\mathbf{w}^\top\mathbf{x}_i+b\le-1\ \text{ if }y_i=-1.
$$

Multiplying by the label combines both into one constraint for **every** training point:

$$
y_i\big(\mathbf{w}^\top\mathbf{x}_i+b\big)\ge1,\qquad i=1,\dots,N.
$$

The points that meet this with equality, $y_i(\mathbf{w}^\top\mathbf{x}_i+b)=1$, lie exactly on the margin hyperplanes. These are the **support vectors**.

**Width of the margin.** Take a positive support vector $\mathbf{x}_+$ and a negative one $\mathbf{x}_-$. Project their difference onto the unit normal $\mathbf{w}/\lVert\mathbf{w}\rVert$, and use $\mathbf{w}^\top\mathbf{x}_+=1-b$ and $\mathbf{w}^\top\mathbf{x}_-=-1-b$:

$$
\text{width}=(\mathbf{x}_+-\mathbf{x}_-)^\top\frac{\mathbf{w}}{\lVert\mathbf{w}\rVert}=\frac{(1-b)-(-1-b)}{\lVert\mathbf{w}\rVert}=\frac{2}{\lVert\mathbf{w}\rVert}.
$$

The distance from the boundary to the nearest point, the *geometric margin*, is half of this, $1/\lVert\mathbf{w}\rVert$. Maximizing $\frac{2}{\lVert\mathbf{w}\rVert}$ is the same as minimizing $\frac12\lVert\mathbf{w}\rVert^2$, which is smooth and convex. This gives the **primal problem**:

$$
\min_{\mathbf{w},b}\ \frac12\lVert\mathbf{w}\rVert^2\qquad\text{s.t.}\qquad y_i\big(\mathbf{w}^\top\mathbf{x}_i+b\big)\ge1\quad\forall i.
$$

All $N$ constraints must stay, as inequalities. We do not know in advance which points will end up on the margin. The fact that only those points matter is a *consequence* of the optimization (complementary slackness, below), not something we can impose.

> [!tip] Prerequisite: constrained optimization
> This lecture leans on Lagrange multipliers, duality and the KKT conditions. If they are new, study [[Multivariable Calculus#Constrained optimization: Lagrange multipliers|constrained optimization]] first: it covers penalties, the Lagrangian as a min–max, weak and strong duality, and the four KKT conditions, with worked examples. Chapter 5 of Boyd and Vandenberghe is the standard reference.[^boyd2004]
>
> **For the SVM:** $g_i=1-y_i(\mathbf{w}^\top\mathbf{x}_i+b)\le0$ and $\lambda_i=\alpha_i$. Stationarity gives $\mathbf{w}=\sum_i\alpha_iy_i\mathbf{x}_i$, dual feasibility gives $\alpha_i\ge0$, and complementary slackness says only points on the margin can have $\alpha_i>0$. Those are the support vectors.

### The Lagrangian dual

Introduce a multiplier $\alpha_i\ge0$ for each constraint:

$$
\mathcal{L}(\mathbf{w},b,\boldsymbol{\alpha})=\frac12\lVert\mathbf{w}\rVert^2-\sum_{i=1}^{N}\alpha_i\big[y_i(\mathbf{w}^\top\mathbf{x}_i+b)-1\big].
$$

The primal is $\min_{\mathbf{w},b}\max_{\boldsymbol{\alpha}\ge0}\mathcal{L}$. This is a convex quadratic program with linear constraints that are strictly feasible when the data are separable, so **strong duality** holds. We may swap the order and solve $\max_{\boldsymbol{\alpha}\ge0}\min_{\mathbf{w},b}\mathcal{L}$ instead (see the KKT aside above). The inner minimization is unconstrained, so we set the gradients to zero:

$$
\begin{aligned}
\nabla_\mathbf{w}\mathcal{L}&=\mathbf{w}-\sum_i\alpha_iy_i\mathbf{x}_i=\mathbf{0}&&\Longrightarrow\quad\boxed{\;\mathbf{w}=\sum_{i=1}^N\alpha_iy_i\mathbf{x}_i\;}\\
\frac{\partial\mathcal{L}}{\partial b}&=-\sum_i\alpha_iy_i=0&&\Longrightarrow\quad\boxed{\;\sum_{i=1}^N\alpha_iy_i=0\;}
\end{aligned}
$$

The optimal $\mathbf{w}$ is a **linear combination of the training points**. Substituting it back:

$$
\begin{aligned}
\mathcal{L}&=\frac12\sum_{i,j}\alpha_i\alpha_jy_iy_j\,\mathbf{x}_i^\top\mathbf{x}_j-\sum_{i,j}\alpha_i\alpha_jy_iy_j\,\mathbf{x}_i^\top\mathbf{x}_j-b\underbrace{\sum_i\alpha_iy_i}_{=0}+\sum_i\alpha_i\\
&=\sum_i\alpha_i-\frac12\sum_{i,j}\alpha_i\alpha_jy_iy_j\,\mathbf{x}_i^\top\mathbf{x}_j.
\end{aligned}
$$

This gives the **dual problem**:

$$
\max_{\boldsymbol{\alpha}}\ \sum_{i=1}^{N}\alpha_i-\frac12\sum_{i=1}^{N}\sum_{j=1}^{N}\alpha_i\alpha_jy_iy_j\,\boxed{\mathbf{x}_i^\top\mathbf{x}_j}\qquad\text{s.t.}\qquad\alpha_i\ge0,\quad\sum_{i=1}^N\alpha_iy_i=0.
$$

In matrix form this is $\max_{\boldsymbol{\alpha}}\ \boldsymbol{\alpha}^\top\mathbf{1}-\frac12\boldsymbol{\alpha}^\top\mathbf{G}\boldsymbol{\alpha}$, with $G_{ij}=y_iy_j\,\mathbf{x}_i^\top\mathbf{x}_j$, a concave QP in $N$ variables.

**Support vectors via complementary slackness.** At the optimum, the [[#The Lagrangian dual|KKT conditions]] include

$$
\alpha_i\Big[y_i\big(\mathbf{w}^\top\mathbf{x}_i+b\big)-1\Big]=0\qquad\forall i.
$$

So either $\alpha_i=0$, and the point does not contribute to $\mathbf{w}$, or the point lies **exactly on the margin**. Only the support vectors have $\alpha_i>0$. Typically they are a small fraction of the data, so the solution is *sparse* in the training points.

**Recovering $b$.** For any support vector $\mathbf{x}_j$ (one with $\alpha_j>0$), $y_j(\mathbf{w}^\top\mathbf{x}_j+b)=1$. Multiplying by $y_j$ and using $y_j^2=1$:

$$
b=y_j-\sum_{i}\alpha_iy_i\,\mathbf{x}_i^\top\mathbf{x}_j.
$$

In practice, average this over all support vectors for numerical stability.

**The decision rule** becomes $\hat y(\mathbf{u})=\operatorname{sign}\Big(\sum_i\alpha_iy_i\,\boxed{\mathbf{x}_i^\top\mathbf{u}}+b\Big)$. The sum runs only over support vectors.

> [!tip] The interesting thing
> The data enter **both** the dual objective **and** the decision rule only through dot products $\mathbf{x}_i^\top\mathbf{x}_j$ and $\mathbf{x}_i^\top\mathbf{u}$. That observation is the door to kernels.

---

## Kernels

Suppose we first map the inputs to a richer feature space, $\mathbf{x}\mapsto\phi(\mathbf{x})$, and run the linear SVM there. Every dot product becomes $\phi(\mathbf{x}_i)^\top\phi(\mathbf{x}_j)$. If some function computes this directly,

$$
K(\mathbf{x},\mathbf{x}')=\phi(\mathbf{x})^\top\phi(\mathbf{x}'),\qquad K:\mathcal{X}\times\mathcal{X}\to\mathbb{R},
$$

then we never need $\phi$ itself. This is the **kernel trick**.[^boser1992] The kernel SVM is

$$
\begin{aligned}
&\max_{\boldsymbol{\alpha}}\ \sum_i\alpha_i-\frac12\sum_{i,j}\alpha_i\alpha_jy_iy_j\,K(\mathbf{x}_i,\mathbf{x}_j)\quad\text{s.t. }\alpha_i\ge0,\ \sum_i\alpha_iy_i=0,\\
&\hat y(\mathbf{u})=\operatorname{sign}\Big(\sum_i\alpha_iy_i\,K(\mathbf{x}_i,\mathbf{u})+b\Big).
\end{aligned}
$$

Even if the data are not linearly separable in the input space, they may be separable after mapping to a higher (even infinite) dimension. The SVM draws a *linear* boundary there, which looks non-linear back in input space, without ever constructing the high-dimensional points.

![[Machine Learning/images/support-vector-machines-kernel-trick.svg]]
*Figure 2: left, two concentric rings cannot be separated by a line. Middle, adding the feature $x_1^2+x_2^2$ lifts the inner ring below a plane, so the classes become linearly separable in 3D. Right, an RBF-kernel SVM finds the circular boundary directly from kernel evaluations, without ever building $\phi$.*

**An explicit example.** For $\mathbf{x}\in\mathbb{R}^2$, the polynomial kernel $K(\mathbf{x},\mathbf{x}')=(\mathbf{x}^\top\mathbf{x}')^2$ expands to

$$
(x_1x_1'+x_2x_2')^2=x_1^2x_1'^2+2x_1x_2x_1'x_2'+x_2^2x_2'^2=\phi(\mathbf{x})^\top\phi(\mathbf{x}'),
$$

with $\phi(\mathbf{x})=(x_1^2,\sqrt2\,x_1x_2,x_2^2)$. Evaluating $K$ costs $O(D)$. Building $\phi$ for degree $d$ would cost $O(D^d)$.

**Which functions are kernels?** $K$ is a valid kernel, i.e. equals $\phi(\mathbf{x})^\top\phi(\mathbf{x}')$ for some feature map, if and only if it is **symmetric** and **positive semi-definite**. That means every Gram matrix $[K(\mathbf{x}_i,\mathbf{x}_j)]_{ij}$ built from any finite set of points must be PSD. This is the finite-sample form of Mercer's condition. It also guarantees that the dual stays a concave QP.[^scholkopf2002] Common choices:

| Kernel | $K(\mathbf{x},\mathbf{x}')$ | Feature space |
|---|---|---|
| linear | $\mathbf{x}^\top\mathbf{x}'$ | $\mathbb{R}^D$ itself |
| polynomial | $(\mathbf{x}^\top\mathbf{x}'+c)^d$ | all monomials up to degree $d$ |
| Gaussian (RBF) | $\exp\big(-\lVert\mathbf{x}-\mathbf{x}'\rVert^2/2\ell^2\big)$ | infinite-dimensional |

New kernels can be built from old ones: sums, positive scalings, products, and $f(\mathbf{x})K(\mathbf{x},\mathbf{x}')f(\mathbf{x}')$ all preserve validity. The kernel trick applies to *any* algorithm that touches data only through dot products. That includes Parzen windows in [[Non-parametric Models]], kernel $k$-means, kernel PCA, kernel ridge regression and Gaussian processes. This section is their common reference point.

> [!example]- Code: Gram matrices for the common kernels
> ```python
> import numpy as np
>
> def gram(X1, X2, kind="rbf", ell=1.0, c=1.0, d=2):
>     """K[i, j] = K(x_i, x'_j) for the rows of X1 (N1, D) and X2 (N2, D)."""
>     if kind == "linear":
>         return X1 @ X2.T
>     if kind == "poly":
>         return (X1 @ X2.T + c) ** d
>     sq = (X1 ** 2).sum(1)[:, None] + (X2 ** 2).sum(1)[None, :] - 2 * X1 @ X2.T   # ||x - x'||^2
>     return np.exp(-np.maximum(sq, 0) / (2 * ell ** 2))          # RBF; clip round-off negatives
>
> # a valid kernel gives a symmetric PSD Gram matrix:
> # np.linalg.eigvalsh(gram(X, X)).min() >= -1e-10
> ```
> The RBF kernel with length scale $\ell$ is scikit-learn's `rbf_kernel` with `gamma` $=1/(2\ell^2)$.

---

## Soft-margin SVM

Real data are rarely perfectly separable, because of noise, outliers or overlapping classes. A single outlier can also force a hard-margin boundary into a poor position. We relax the constraints with **slack variables** $\xi_i\ge0$, which allow a point inside the margin (or even on the wrong side) at a price:

$$
\min_{\mathbf{w},b,\boldsymbol{\xi}}\ \frac12\lVert\mathbf{w}\rVert^2+C\sum_{i=1}^{N}\xi_i\qquad\text{s.t.}\qquad y_i\big(\mathbf{w}^\top\mathbf{x}_i+b\big)\ge1-\xi_i,\quad\xi_i\ge0\quad\forall i.
$$

The term $C\sum_i\xi_i$ is the **slack penalty**: the total amount by which points violate the margin, weighted by $C>0$. Points pay only if they need to. They sit inside the margin only when keeping them outside would make the classifier worse overall.

**Dual.** With multipliers $\alpha_i\ge0$ for the margin constraints and $\beta_i\ge0$ for $\xi_i\ge0$:

$$
\mathcal{L}=\frac12\lVert\mathbf{w}\rVert^2+C\sum_i\xi_i+\sum_i\alpha_i\big(1-\xi_i-y_i(\mathbf{w}^\top\mathbf{x}_i+b)\big)-\sum_i\beta_i\xi_i.
$$

Setting the gradients to zero:

$$
\begin{aligned}
\nabla_\mathbf{w}\mathcal{L}&=\mathbf{w}-\sum_i\alpha_iy_i\mathbf{x}_i=\mathbf{0},\qquad\frac{\partial\mathcal{L}}{\partial b}=-\sum_i\alpha_iy_i=0,\\
\frac{\partial\mathcal{L}}{\partial\xi_i}&=C-\alpha_i-\beta_i=0.
\end{aligned}
$$

The first two conditions are as before. The third gives $\beta_i=C-\alpha_i$, and $\beta_i\ge0$ then means $\alpha_i\le C$. Substituting, the $\xi_i$ terms cancel: $C\xi_i-\alpha_i\xi_i-\beta_i\xi_i=(C-\alpha_i-\beta_i)\xi_i=0$. We are left with *the same dual as before, plus a box constraint*:

$$
\max_{\boldsymbol{\alpha}}\ \boldsymbol{\alpha}^\top\mathbf{1}-\frac12\boldsymbol{\alpha}^\top\mathbf{G}\boldsymbol{\alpha}\qquad\text{s.t.}\qquad0\le\alpha_i\le C,\quad\boldsymbol{\alpha}^\top\mathbf{y}=0.
$$

Kernels plug in exactly as before. Complementary slackness, $\alpha_i\big(1-\xi_i-y_if(\mathbf{x}_i)\big)=0$ and $\beta_i\xi_i=0$, sorts the points into three groups:

- **$\alpha_i=0$:** correctly classified, beyond the margin. The point plays no role.
- **$0<\alpha_i<C$:** then $\beta_i>0$, so $\xi_i=0$, and the point lies **exactly on the margin**. Use these points to compute $b$.
- **$\alpha_i=C$:** inside the margin ($0<\xi_i\le1$) or misclassified ($\xi_i>1$).

![[Machine Learning/images/support-vector-machines-soft-margin.svg]]
*Figure 3: soft-margin SVMs on overlapping classes. The orange segments show each margin violation; a segment's length is $\xi_i/\lVert\mathbf{w}\rVert$. Circled points are support vectors. A small $C$ makes violations cheap: a wide margin and many support vectors. A large $C$ approaches the hard margin: a narrow margin, few violations and fewer support vectors.*

**$C$ is a regularization knob.** Large $C$ fits the training data closely (low bias, high variance). Small $C$ prefers a wide, simple margin. Choose it, and any kernel parameters such as $\ell$, by cross-validation.

### The hinge loss: SVM vs. logistic regression

At the optimum each slack takes the smallest feasible value, $\xi_i=\max\big(0,\,1-y_if(\mathbf{x}_i)\big)$ with $f(\mathbf{x})=\mathbf{w}^\top\mathbf{x}+b$. Substituting removes the constraints entirely:

$$
\min_{\mathbf{w},b}\ \underbrace{\sum_{i=1}^{N}\max\big(0,\,1-y_if(\mathbf{x}_i)\big)}_{\text{hinge loss}}+\underbrace{\frac{1}{2C}\lVert\mathbf{w}\rVert^2}_{L_2\text{ regularization}}.
$$

Compare this with $L_2$-regularized [[Linear Models#Logistic Regression|logistic regression]], whose loss in $\pm1$ labels is $\log\big(1+e^{-y_if(\mathbf{x}_i)}\big)$. The two objectives have the *same structure*: a loss on the **margin** $m_i=y_if(\mathbf{x}_i)$ plus $\lambda\lVert\mathbf{w}\rVert^2$, with $\lambda\leftrightarrow\frac{1}{2C}$. They differ only in the loss (Figure 4).

![[Machine Learning/images/support-vector-machines-losses.svg]]
*Figure 4: losses as a function of the margin $m=yf(\mathbf{x})$. Both the hinge and the logistic loss are convex upper bounds on the 0–1 loss. The hinge is exactly zero once $m\ge1$: confidently correct points cost nothing, which is why SVM solutions are sparse. The logistic loss never reaches zero, so every point keeps a small influence, but in exchange it yields calibrated probabilities.*

> [!example]- Code: the hinge-loss objective by stochastic subgradient descent (Pegasos)
> ```python
> import numpy as np
>
> def pegasos(X, y, lam, n_iter=200_000, rng=None):
>     """Minimizes lam/2 ||w||^2 + (1/N) sum_i max(0, 1 - y_i w^T x_i), with y in {-1, +1}."""
>     rng = np.random.default_rng(rng)
>     w = np.zeros(X.shape[1])
>     for t in range(1, n_iter + 1):
>         i = rng.integers(len(y))                     # one random example per step
>         eta = 1.0 / (lam * t)                        # Pegasos step size
>         active = y[i] * (w @ X[i]) < 1               # hinge active before the step?
>         w *= 1 - eta * lam                           # gradient of the regularizer
>         if active:
>             w += eta * y[i] * X[i]                   # subgradient of the hinge: -y_i x_i
>     return w
> ```
> Dividing the hinge-loss objective above by $N$ shows that this is the same problem with $\lambda=1/(CN)$, except for the bias: append a constant feature to learn one, and Pegasos then regularizes it along with $\mathbf{w}$.

---

## Solving SVMs in practice

- **Generic QP solvers.** The dual is a standard QP, so a convex solver such as cvxopt[^cvxopt] works directly. First estimate $\boldsymbol{\alpha}$ by solving the dual. Then compute $b$ from the margin support vectors, averaging over several of them. This is fine for small $N$. `cvxopt.solvers.qp` minimizes $\frac12\mathbf{a}^\top\mathbf{P}\mathbf{a}+\mathbf{q}^\top\mathbf{a}$ subject to $\mathbf{G}\mathbf{a}\le\mathbf{h}$ and $\mathbf{A}\mathbf{a}=\mathbf{b}$. Maximizing the dual is the same as minimizing its negative, so $P_{ij}=y_iy_jK(\mathbf{x}_i,\mathbf{x}_j)$ and $\mathbf{q}=-\mathbf{1}$:

```python
import numpy as np
from cvxopt import matrix, solvers

def fit_svm_dual(K, y, C):
    """K: (N, N) kernel Gram matrix, y: (N,) labels in {-1, +1}, C: slack penalty."""
    N = len(y)
    P = matrix(np.outer(y, y) * K)                       # P_ij = y_i y_j K(x_i, x_j)
    q = matrix(-np.ones(N))                              # min 1/2 a'Pa - 1'a  (= max of the dual)
    G = matrix(np.vstack([-np.eye(N), np.eye(N)]))       # -a_i <= 0  and  a_i <= C
    h = matrix(np.hstack([np.zeros(N), C * np.ones(N)]))
    A = matrix(y.reshape(1, -1).astype(float))           # y'a = 0
    solvers.options["show_progress"] = False
    a = np.ravel(solvers.qp(P, q, G, h, A, matrix(0.0))["x"])
    on_margin = (a > 1e-6) & (a < C - 1e-6)              # 0 < a_i < C: exactly on the margin
    b = np.mean(y[on_margin] - (a * y) @ K[:, on_margin])
    return a, b

# Prediction for test points u_j, with K_test[i, j] = K(x_i, u_j):
#     y_hat = np.sign((a * y) @ K_test + b)
```

  Interior-point solvers return tiny but nonzero $\alpha_i$ for non-support vectors, hence the $10^{-6}$ thresholds. On an RBF-kernel test problem this function agrees with scikit-learn's `SVC` on every prediction, with $b$ matching to two decimals.

- **SMO.** Sequential minimal optimization (Platt) updates two $\alpha$'s at a time in closed form. It is the classic dedicated solver, used by LIBSVM and scikit-learn's `SVC`.
- **Primal (sub)gradient methods** for linear SVMs on large datasets. Run SGD directly on the hinge-loss form, as in Pegasos.[^pegasos2011] The cost per step does not depend on $N$.
- **Scaling caveat for kernels.** The Gram matrix has $N^2$ entries, and prediction costs $O(\#\text{SV})$ kernel evaluations. Kernel SVMs are therefore at their best on small-to-medium datasets.
- **Multi-class problems** are handled by combining binary SVMs: one-vs-rest or one-vs-one. **Probabilities** can be obtained by fitting a sigmoid to the SVM scores (Platt scaling).

## Support vector regression

The margin idea carries over to regression. Instead of a street *between* two classes, SVR fits a **tube** of half-width $\varepsilon$ *around* the regression function $f(\mathbf{x})=\mathbf{w}^\top\mathbf{x}+b$ and ignores every error that falls inside it. This is the **$\varepsilon$-insensitive loss**, $\max\big(0,\lvert y-f(\mathbf{x})\rvert-\varepsilon\big)$.[^smolasvr2004] As in classification, we want $f$ as "flat" as possible (small $\lVert\mathbf{w}\rVert$) while keeping the points inside the tube, with slack for the ones that cannot be.

### Primal

Points may stick out *above* the tube (slack $\xi_i$) or *below* it (slack $\xi_i^*$):

$$
\begin{aligned}
\min_{\mathbf{w},b,\boldsymbol{\xi},\boldsymbol{\xi}^*}\ &\frac12\lVert\mathbf{w}\rVert^2+C\sum_{i=1}^{N}(\xi_i+\xi_i^*)\\
\text{s.t.}\ &y_i-\mathbf{w}^\top\mathbf{x}_i-b\le\varepsilon+\xi_i,\qquad\mathbf{w}^\top\mathbf{x}_i+b-y_i\le\varepsilon+\xi_i^*,\qquad\xi_i,\xi_i^*\ge0.
\end{aligned}
$$

### Dual

Introduce multipliers $\alpha_i,\alpha_i^*\ge0$ for the two tube constraints and $\eta_i,\eta_i^*\ge0$ for the slack signs:

$$
\begin{aligned}
\mathcal{L}=\ &\frac12\lVert\mathbf{w}\rVert^2+C\sum_i(\xi_i+\xi_i^*)-\sum_i(\eta_i\xi_i+\eta_i^*\xi_i^*)\\
&+\sum_i\alpha_i\big(y_i-\mathbf{w}^\top\mathbf{x}_i-b-\varepsilon-\xi_i\big)+\sum_i\alpha_i^*\big(\mathbf{w}^\top\mathbf{x}_i+b-y_i-\varepsilon-\xi_i^*\big).
\end{aligned}
$$

Stationarity in each primal variable gives

$$
\begin{aligned}
\nabla_\mathbf{w}\mathcal{L}=\mathbf{0}&\ \Longrightarrow\ \mathbf{w}=\sum_i(\alpha_i-\alpha_i^*)\,\mathbf{x}_i,\qquad\frac{\partial\mathcal{L}}{\partial b}=0\ \Longrightarrow\ \sum_i(\alpha_i-\alpha_i^*)=0,\\
\frac{\partial\mathcal{L}}{\partial\xi_i}=0&\ \Longrightarrow\ \eta_i=C-\alpha_i\ \Longrightarrow\ 0\le\alpha_i\le C\quad(\text{and likewise }0\le\alpha_i^*\le C).
\end{aligned}
$$

Write $\beta_i=\alpha_i-\alpha_i^*$ and substitute. The $\xi$ and $b$ terms cancel, and $\frac12\lVert\mathbf{w}\rVert^2-\mathbf{w}^\top\sum_i\beta_i\mathbf{x}_i=-\frac12\lVert\mathbf{w}\rVert^2$. This leaves the **SVR dual**:

$$
\begin{aligned}
\max_{\boldsymbol{\alpha},\boldsymbol{\alpha}^*}\ &-\frac12\sum_{i,j}\beta_i\beta_j\,\mathbf{x}_i^\top\mathbf{x}_j-\varepsilon\sum_i(\alpha_i+\alpha_i^*)+\sum_iy_i\beta_i\\
\text{s.t.}\ &\sum_i\beta_i=0,\qquad0\le\alpha_i,\alpha_i^*\le C.
\end{aligned}
$$

As before, the data appear only through dot products. Replace $\mathbf{x}_i^\top\mathbf{x}_j$ by $K(\mathbf{x}_i,\mathbf{x}_j)$, and the prediction becomes $f(\mathbf{x})=\sum_i\beta_i\,K(\mathbf{x}_i,\mathbf{x})+b$.

### Which points are support vectors?

Complementary slackness, $\alpha_i\big(y_i-f(\mathbf{x}_i)-\varepsilon-\xi_i\big)=0$ and $(C-\alpha_i)\xi_i=0$ (and likewise for $\alpha_i^*$), sorts the training points:

- **Strictly inside the tube**, $\lvert y_i-f(\mathbf{x}_i)\rvert<\varepsilon$: neither constraint is active, so $\alpha_i=\alpha_i^*=0$. These points do not affect the solution at all.
- **On the upper edge**, $y_i-f(\mathbf{x}_i)=\varepsilon$: $0<\alpha_i<C$ and $\xi_i=0$. **On the lower edge**: $0<\alpha_i^*<C$.
- **Outside the tube**: $\alpha_i=C$ (above) or $\alpha_i^*=C$ (below), with positive slack.
- A point cannot be above and below the tube at once, so $\alpha_i\alpha_i^*=0$.

The bias comes from the edge points. If $0<\alpha_i<C$, then $b=y_i-\varepsilon-\sum_j\beta_jK(\mathbf{x}_j,\mathbf{x}_i)$. If $0<\alpha_i^*<C$, then $b=y_i+\varepsilon-\sum_j\beta_jK(\mathbf{x}_j,\mathbf{x}_i)$. Average over all such points. The width $\varepsilon$ therefore controls **sparsity**: a wider tube holds more points and leaves fewer support vectors. $C$ controls how hard outliers pull on the fit.

![[Machine Learning/images/support-vector-machines-svr.svg]]
*Figure 5: an RBF-kernel SVR on noisy samples of $\sin x$. Grey points lie inside the $\varepsilon$-tube and have zero multipliers. Only the red points, on or outside the tube, are support vectors (27 of 80 here). The orange segments show the slacks of points outside the tube.*

### Solving the SVR dual with cvxopt

Stack the variables as $\mathbf{z}=[\boldsymbol{\alpha};\boldsymbol{\alpha}^*]\in\mathbb{R}^{2N}$. Then $\boldsymbol{\beta}^\top\mathbf{K}\boldsymbol{\beta}=\mathbf{z}^\top\begin{pmatrix}\mathbf{K}&-\mathbf{K}\\-\mathbf{K}&\mathbf{K}\end{pmatrix}\mathbf{z}$, and minimizing the negated dual is a standard QP:

```python
import numpy as np
from cvxopt import matrix, solvers

def fit_svr_dual(K, y, C, eps):
    """K: (N, N) Gram matrix, y: (N,) targets. Returns beta = alpha - alpha*, b, alpha, alpha_star."""
    N = len(y)
    P = np.block([[K, -K], [-K, K]]) + 1e-8 * np.eye(2 * N)   # z = [alpha; alpha*]; tiny ridge for stability
    q = np.hstack([eps - y, eps + y])                         # -(dual) = 1/2 z'Pz + eps*1'z - y'(alpha - alpha*)
    G = np.vstack([-np.eye(2 * N), np.eye(2 * N)])            # 0 <= z <= C
    h = np.hstack([np.zeros(2 * N), C * np.ones(2 * N)])
    A = np.hstack([np.ones(N), -np.ones(N)]).reshape(1, -1)   # sum(alpha - alpha*) = 0
    solvers.options["show_progress"] = False
    z = np.ravel(solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h), matrix(A), matrix(0.0))["x"])
    a, a_s = z[:N], z[N:]
    beta = a - a_s
    f0 = K @ beta                                             # f(x_i) - b
    up = (a > 1e-6) & (a < C - 1e-6)                          # upper edge: y_i - f(x_i) = eps
    lo = (a_s > 1e-6) & (a_s < C - 1e-6)                      # lower edge: f(x_i) - y_i = eps
    b = np.mean(np.r_[y[up] - eps - f0[up], y[lo] + eps - f0[lo]])
    return beta, b, a, a_s

# Prediction: f(u_j) = beta @ K_test[:, j] + b, with K_test[i, j] = K(x_i, u_j)
```

On the problem of Figure 5, this agrees with scikit-learn's `SVR` to within $0.004$ everywhere, and $\alpha_i\alpha_i^*\approx0$ as predicted.

## Key takeaways

- The **hard-margin SVM** solves $\min\frac12\lVert\mathbf{w}\rVert^2$ subject to $y_i(\mathbf{w}^\top\mathbf{x}_i+b)\ge1$ for *all* $i$. This maximizes the margin $2/\lVert\mathbf{w}\rVert$.
- **Dual:** $\mathbf{w}=\sum_i\alpha_iy_i\mathbf{x}_i$ with $\alpha_i\ge0$ and $\sum_i\alpha_iy_i=0$. By complementary slackness, only **support vectors** have $\alpha_i>0$.
- The data appear only through dot products, so the **kernel trick** applies. Valid kernels are symmetric and PSD. Linear, polynomial and RBF are the standard choices.
- **Soft margin:** slack variables add a box constraint $0\le\alpha_i\le C$. This is equivalent to the **hinge loss** plus $L_2$ regularization; logistic regression is the same template with the log loss.
- Solve with QP solvers (small $N$), SMO (kernels) or SGD on the primal (large linear problems).
- **SVR** fits an $\varepsilon$-tube. Points inside the tube have zero multipliers, and $\mathbf{w}=\sum_i(\alpha_i-\alpha_i^*)\mathbf{x}_i$ is kernelized exactly as in classification.

## Where this shows up next

- [[Linear Models]]: the logistic-regression counterpart, and $L_2$ regularization.
- [[Discriminant Analysis]]: generative linear classifiers. LDA, logistic regression and the linear SVM are three routes to a hyperplane.
- [[Non-parametric Models]]: Parzen windows and kernels as smoothing windows.
- [[Clustering and Expectation Maximization#Kernel K-Means|Kernel K-Means]] and [[Clustering and Expectation Maximization#Spectral clustering|spectral clustering]] reuse this lecture's kernels and Lagrangian tools; kernel PCA comes with matrix factorization.

<!-- References: footnote definitions below render as the numbered "References" list at the bottom of the page -->

[^cortesvapnik1995]: Cortes, C., & Vapnik, V. (1995). *Support-Vector Networks*. Machine Learning, 20, 273–297. [doi:10.1007/BF00994018](https://doi.org/10.1007/BF00994018)
[^boser1992]: Boser, B. E., Guyon, I. M., & Vapnik, V. N. (1992). *A Training Algorithm for Optimal Margin Classifiers*. COLT. [doi:10.1145/130385.130401](https://doi.org/10.1145/130385.130401)
[^boyd2004]: Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*, Chapter 5 (Duality). Cambridge University Press. [book website](https://web.stanford.edu/~boyd/cvxbook/)
[^scholkopf2002]: Schölkopf, B., & Smola, A. J. (2002). *Learning with Kernels*. MIT Press. [doi:10.7551/mitpress/4175.001.0001](https://doi.org/10.7551/mitpress/4175.001.0001)
[^pegasos2011]: Shalev-Shwartz, S., Singer, Y., Srebro, N., & Cotter, A. (2011). *Pegasos: Primal Estimated Sub-Gradient Solver for SVM*. Mathematical Programming, 127, 3–30. [doi:10.1007/s10107-010-0420-4](https://doi.org/10.1007/s10107-010-0420-4)
[^smolasvr2004]: Smola, A. J., & Schölkopf, B. (2004). *A Tutorial on Support Vector Regression*. Statistics and Computing, 14, 199–222. [doi:10.1023/B:STCO.0000035301.49549.88](https://doi.org/10.1023/B:STCO.0000035301.49549.88)
[^cvxopt]: Andersen, M., Dahl, J., & Vandenberghe, L. *CVXOPT: Python Software for Convex Optimization*. [cvxopt.org](https://cvxopt.org/)
