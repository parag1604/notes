---
title: Support Vector Machines
tags:
  - svm
  - kernels
  - convex-optimization
  - max-margin
---

> **Prerequisites:** [[Linear Models]] (linear classifiers, logistic regression, $L_2$ regularization) and the basics of constrained optimization (Lagrange multipliers).
>
> **TL;DR:**
> - Among all hyperplanes that separate two classes, the SVM picks the one with the **largest margin**.
> - This is a convex quadratic program. Its **dual** depends on the data only through dot products $\mathbf{x}_i^\top\mathbf{x}_j$, and its solution depends only on the few **support vectors** on the margin.
> - Replacing those dot products by a **kernel** $K(\mathbf{x},\mathbf{x}')$ gives non-linear boundaries without ever computing the feature map.
> - The **soft-margin** SVM tolerates violations through slack variables, and is the same thing as $L_2$-regularized **hinge-loss** minimization.

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

> [!tip]- Aside: why Lagrange multipliers and the KKT conditions work
> **Equality constraints.** To minimize $f(\mathbf{x})$ subject to $h(\mathbf{x})=0$, look at a candidate optimum on the constraint surface. Moving *along* the surface must not decrease $f$, so $\nabla f$ can have no component tangent to the surface. It must be perpendicular to the surface, i.e. parallel to the surface normal $\nabla h$:
> $$
> \nabla f(\mathbf{x}^*)=-\nu\,\nabla h(\mathbf{x}^*)\quad\text{for some }\nu\in\mathbb{R}.
> $$
> This is exactly "set the gradient of the Lagrangian $f+\nu h$ to zero".
>
> **Inequality constraints.** Now minimize $f(\mathbf{x})$ subject to $g(\mathbf{x})\le0$. There are two cases.
> - **Inactive** ($g(\mathbf{x}^*)<0$): the optimum lies strictly inside the feasible region. The constraint plays no role, so $\nabla f(\mathbf{x}^*)=\mathbf{0}$, i.e. multiplier $\lambda=0$.
> - **Active** ($g(\mathbf{x}^*)=0$): the optimum lies on the boundary. As before, $\nabla f$ must be parallel to $\nabla g$, but now with a *sign*. $-\nabla f$, the direction we would like to move, must point *out* of the feasible set, i.e. along $+\nabla g$. Otherwise we could step inward and still decrease $f$. So $-\nabla f=\lambda\nabla g$ with $\lambda\ge0$: the constraint can only *push* back, never pull.
>
> Both cases are summarized by $\lambda\,g(\mathbf{x}^*)=0$: either the constraint is active, or its multiplier is zero.
>
> **The KKT conditions.** For $\min f(\mathbf{x})$ s.t. $g_i(\mathbf{x})\le0$, with Lagrangian $\mathcal{L}=f+\sum_i\lambda_ig_i$:
> 1. **Stationarity:** $\nabla f(\mathbf{x}^*)+\sum_i\lambda_i\nabla g_i(\mathbf{x}^*)=\mathbf{0}$.
> 2. **Primal feasibility:** $g_i(\mathbf{x}^*)\le0$.
> 3. **Dual feasibility:** $\lambda_i\ge0$.
> 4. **Complementary slackness:** $\lambda_i\,g_i(\mathbf{x}^*)=0$.
>
> For **convex** problems that are strictly feasible (Slater's condition), these are *necessary and sufficient* for optimality, and the primal and dual optimal values coincide (*strong duality*).[^boyd2004]
>
> **Why the min–max form.** For fixed $\mathbf{x}$, $\max_{\boldsymbol{\lambda}\ge0}\mathcal{L}$ equals $f(\mathbf{x})$ if every $g_i\le0$ (best to set $\lambda_i=0$), and $+\infty$ otherwise. So $\min_\mathbf{x}\max_{\boldsymbol{\lambda}\ge0}\mathcal{L}$ *is* the constrained problem. Swapping the order gives the dual, which is always a lower bound (*weak duality*) and is tight under the conditions above.
>
> **For the SVM:** $g_i=1-y_i(\mathbf{w}^\top\mathbf{x}_i+b)\le0$ and $\lambda_i=\alpha_i$. Stationarity gives $\mathbf{w}=\sum_i\alpha_iy_i\mathbf{x}_i$, dual feasibility gives $\alpha_i\ge0$, and complementary slackness says that only points on the margin can have $\alpha_i>0$. Those are the support vectors.
>
> For a visual walkthrough (it also covers how interior-point solvers such as cvxopt use these conditions), see Visually Explained's video *The Karush–Kuhn–Tucker (KKT) Conditions and the Interior Point Method for Convex Optimization*.[^kktvideo]

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

The same machinery handles regression. Instead of a margin around a boundary, SVR fits a **tube** of half-width $\varepsilon$ around the regression function and ignores errors that fall inside it (the $\varepsilon$-insensitive loss):[^smolasvr2004]

$$
\begin{aligned}
\min_{\mathbf{w},b,\boldsymbol{\xi},\boldsymbol{\xi}^*}\ &\frac12\lVert\mathbf{w}\rVert^2+C\sum_{i}(\xi_i+\xi_i^*)\\
\text{s.t.}\ &y_i-f(\mathbf{x}_i)\le\varepsilon+\xi_i,\quad f(\mathbf{x}_i)-y_i\le\varepsilon+\xi_i^*,\quad\xi_i,\xi_i^*\ge0.
\end{aligned}
$$

Points strictly inside the tube get zero multipliers. The dual again involves only dot products, so kernels apply unchanged.

## Key takeaways

- The **hard-margin SVM** solves $\min\frac12\lVert\mathbf{w}\rVert^2$ subject to $y_i(\mathbf{w}^\top\mathbf{x}_i+b)\ge1$ for *all* $i$. This maximizes the margin $2/\lVert\mathbf{w}\rVert$.
- **Dual:** $\mathbf{w}=\sum_i\alpha_iy_i\mathbf{x}_i$ with $\alpha_i\ge0$ and $\sum_i\alpha_iy_i=0$. By complementary slackness, only **support vectors** have $\alpha_i>0$.
- The data appear only through dot products, so the **kernel trick** applies. Valid kernels are symmetric and PSD. Linear, polynomial and RBF are the standard choices.
- **Soft margin:** slack variables add a box constraint $0\le\alpha_i\le C$. This is equivalent to the **hinge loss** plus $L_2$ regularization; logistic regression is the same template with the log loss.
- Solve with QP solvers (small $N$), SMO (kernels) or SGD on the primal (large linear problems).

## Where this shows up next

- [[Linear Models]]: the logistic-regression counterpart, and $L_2$ regularization.
- [[Discriminant Analysis]]: generative linear classifiers. LDA, logistic regression and the linear SVM are three routes to a hyperplane.
- [[Non-parametric Models]]: Parzen windows and kernels as similarity functions.
- **Kernel k-means, kernel PCA and spectral methods** (coming soon) reuse this lecture's kernels.

<!-- References: footnote definitions below render as the numbered "References" list at the bottom of the page -->

[^cortesvapnik1995]: Cortes, C., & Vapnik, V. (1995). *Support-Vector Networks*. Machine Learning, 20, 273–297. [doi:10.1007/BF00994018](https://doi.org/10.1007/BF00994018)
[^boser1992]: Boser, B. E., Guyon, I. M., & Vapnik, V. N. (1992). *A Training Algorithm for Optimal Margin Classifiers*. COLT. [doi:10.1145/130385.130401](https://doi.org/10.1145/130385.130401)
[^boyd2004]: Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*, Chapter 5 (Duality). Cambridge University Press. [book website](https://web.stanford.edu/~boyd/cvxbook/)
[^scholkopf2002]: Schölkopf, B., & Smola, A. J. (2002). *Learning with Kernels*. MIT Press. [doi:10.7551/mitpress/4175.001.0001](https://doi.org/10.7551/mitpress/4175.001.0001)
[^pegasos2011]: Shalev-Shwartz, S., Singer, Y., Srebro, N., & Cotter, A. (2011). *Pegasos: Primal Estimated Sub-Gradient Solver for SVM*. Mathematical Programming, 127, 3–30. [doi:10.1007/s10107-010-0420-4](https://doi.org/10.1007/s10107-010-0420-4)
[^smolasvr2004]: Smola, A. J., & Schölkopf, B. (2004). *A Tutorial on Support Vector Regression*. Statistics and Computing, 14, 199–222. [doi:10.1023/B:STCO.0000035301.49549.88](https://doi.org/10.1023/B:STCO.0000035301.49549.88)
[^kktvideo]: Visually Explained (YouTube). *The Karush–Kuhn–Tucker (KKT) Conditions and the Interior Point Method for Convex Optimization*. [video](https://www.youtube.com/watch?v=uh1Dk68cfWs)
[^cvxopt]: Andersen, M., Dahl, J., & Vandenberghe, L. *CVXOPT: Python Software for Convex Optimization*. [cvxopt.org](https://cvxopt.org/)
