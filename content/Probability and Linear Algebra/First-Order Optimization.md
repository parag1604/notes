---
title: "First-Order Optimization: Gradient Descent and its Stochastic Variants"
tags:
  - optimization
  - foundations
date: 2026-11-12
draft: false
prev: "[[Eigenvalues and Eigenvectors]]"
next: "[[Five Factorizations of a Matrix]]"
---

> **Prerequisites:** [[Multivariable Calculus]] (gradients, the Hessian, Taylor expansions, convexity) and [[Eigenvalues and Eigenvectors]] (symmetric matrices, positive definiteness).
>
> **Series:** this is part 7 of 8. [[Multivariable Calculus]] → [[Column Space and Rank]] → [[Elimination and the Nullspace]] → [[The Four Fundamental Subspaces]] → [[Orthogonality and Least Squares]] → [[Eigenvalues and Eigenvectors]] → **First-Order Optimization** → [[Five Factorizations of a Matrix]].
>
> **TL;DR:**
> - Training a model means minimizing an **average of per-example losses** over millions of parameters. Only gradients are affordable at that scale, so every training algorithm in the series is a variant of "step downhill along the negative gradient".
> - On a quadratic, gradient descent splits into independent eigen-directions, each shrinking by $1-\alpha\lambda_i$ per step. It converges iff $\alpha<2/\lambda_{\max}$, and the **condition number** $\kappa=\lambda_{\max}/\lambda_{\min}$ sets the speed: about $\kappa$ steps per digit of accuracy.
> - For general functions the same picture holds with $L$ (smoothness) and $\mu$ (strong convexity): rate $O(1/t)$ for smooth convex functions, a linear rate $(1-1/\kappa)^t$ for strongly convex ones, and $O(1/\sqrt t)$ on the gradient norm for non-convex ones.
> - **SGD** replaces the full gradient by one example's gradient: $N$ times cheaper, unbiased, but noisy. With a constant step it stalls at a **noise floor** proportional to $\alpha$; decaying steps (Robbins–Monro) or **mini-batches** (variance $\propto1/B$) push the floor down.
> - **Momentum** reduces the dependence on $\kappa$ to $\sqrt\kappa$. **Adaptive methods** (AdaGrad, RMSProp, Adam, AdamW) give each coordinate its own step size, which repairs badly scaled problems.
>
> **Notation:** parameters $\mathbf{w}\in\mathbb{R}^D$, objective $f$, per-example losses $f_i$, step size $\alpha$, smoothness constant $L$ (always written "L-smooth", not to be confused with a matrix $\mathbf{L}$), strong-convexity constant $\mu$ (not a mean in this lecture), condition number $\kappa=L/\mu$, minimizer $\mathbf{w}_*$ and optimal value $f_*$. Time starts at $t=0$.
>
> This lecture is the series' full treatment of gradient-based optimization. [[Linear Models]] and [[Neural Networks]] use these methods and link back here.

## Motivation

Every model in these notes is trained the same way. Write down a loss that measures how badly the model fits the data, then turn the knobs (the parameters) to make it small. Linear regression has a closed form, but almost nothing else does: logistic regression, neural networks, policy gradients and VAEs are all trained by **iterative** methods that start somewhere and improve step by step.

At ML scale only the simplest such methods survive. A modern network has $D\approx10^8$ parameters or more. The Hessian alone would need $D^2=10^{16}$ numbers, so Newton's method is out of the question. The gradient, on the other hand, costs about as much as evaluating the loss (thanks to [[Neural Networks#Backpropagation|backpropagation]]). Methods that use only function values and gradients are called **first-order methods**, and they are the subject of this lecture.

The questions we will answer:
1. Which direction should we step in, and how far?
2. Why is gradient descent sometimes fast and sometimes painfully slow? (Answer: eigenvalues.)
3. What can be proved about its speed?
4. What changes when the gradient is computed from a random sample of the data?
5. Why do momentum and Adam exist, and when should we use which?

## The optimization problem in ML

### Empirical risk as an average

A dataset $\mathcal{D}=\{(\mathbf{x}_i,y_i)\}_{i=1}^N$ and a model with parameters $\mathbf{w}$ give one loss per example, $f_i(\mathbf{w})=\ell(\mathbf{w};\mathbf{x}_i,y_i)$. Training minimizes their **average**, the empirical risk (see [[Bayesian Decision Theory#Learning as risk minimization|learning as risk minimization]]):

$$
\min_{\mathbf{w}\in\mathbb{R}^D}\ f(\mathbf{w}),\qquad f(\mathbf{w})=\frac1N\sum_{i=1}^Nf_i(\mathbf{w}).
\tag{1}
$$

A regularizer such as $\frac\lambda2\lVert\mathbf{w}\rVert^2$ is simply added to every $f_i$. Three examples run through the series:

| Model | Per-example loss $f_i(\mathbf{w})$ | Gradient $\nabla f_i(\mathbf{w})$ | Convex? |
|---|---|---|---|
| Least squares | $\frac12(\mathbf{w}^\top\mathbf{x}_i-y_i)^2$ | $(\mathbf{w}^\top\mathbf{x}_i-y_i)\,\mathbf{x}_i$ | yes |
| Logistic regression, $y_i\in\{0,1\}$ | $\log(1+e^{z_i})-y_iz_i$, with $z_i=\mathbf{w}^\top\mathbf{x}_i$ | $(\sigma(z_i)-y_i)\,\mathbf{x}_i$ | yes |
| Neural network | cross-entropy of the network output | backpropagation | no |

The structure in (1), an average of many similar terms, is what makes the stochastic methods later in the lecture possible: a random term is an unbiased estimate of the whole.

### Why first-order methods

Compare the cost of one step for a linear model with $N$ examples and $D$ features:

| Method | Information used | Cost per step | Memory |
|---|---|---|---|
| Newton | gradient and Hessian | $O(ND^2+D^3)$ | $O(D^2)$ |
| Gradient descent | full gradient | $O(ND)$ | $O(D)$ |
| Mini-batch SGD | gradient of $B$ examples | $O(BD)$ | $O(D)$ |

Newton's method converges in very few steps, but for $D=10^6$ a single step is already hopeless. First-order methods take many more, much cheaper steps, and their cost grows only linearly with the size of the model.

### What we can hope to find

If $f$ is **convex** ([[Multivariable Calculus#Convexity|Multivariable Calculus]]), every local minimum is global, and setting $\nabla f=\mathbf{0}$ characterizes the minimizers. Least squares, logistic regression and SVMs are convex, and for them we can prove that gradient methods find the best model. Neural network losses are not convex. There, the honest goal is a **stationary point**, a $\mathbf{w}$ with small $\lVert\nabla f(\mathbf{w})\rVert$, and in practice such points are usually good enough.

### Mean versus summed loss

Some texts minimize the **sum** $\sum_if_i$ instead of the mean. The minimizer is the same, but the gradient is $N$ times larger, so the step size must be $N$ times smaller to take the same steps. Every step-size rule in this lecture (such as $\alpha<2/L$) is stated for the mean (1), whose curvature does not grow with the dataset. If you use the sum, divide $\alpha$ by $N$.

## Gradient descent

### Steepest descent from the first-order Taylor model

Near the current point $\mathbf{w}$, the first-order Taylor expansion ([[Multivariable Calculus#Partial derivatives and the gradient|Multivariable Calculus]]) says

$$
f(\mathbf{w}+\mathbf{d})\approx f(\mathbf{w})+\nabla f(\mathbf{w})^\top\mathbf{d}.
$$

Among all steps $\mathbf{d}$ of a fixed small length $\varepsilon$, which one decreases this linear model most? By the Cauchy–Schwarz inequality, $\nabla f^\top\mathbf{d}\ge-\lVert\nabla f\rVert\,\lVert\mathbf{d}\rVert=-\varepsilon\lVert\nabla f\rVert$, with equality exactly when $\mathbf{d}$ points opposite to the gradient. So the **steepest descent direction** is $-\nabla f(\mathbf{w})$, and gradient descent repeatedly takes a step along it.

> [!example] Algorithm: gradient descent (GD)
> Choose a starting point $\mathbf{w}_0$ and a step size $\alpha>0$ (also called the learning rate).
> **For** $t=0,1,2,\dots$:
> 1. Compute the gradient $\mathbf{g}_t=\nabla f(\mathbf{w}_t)$.
> 2. Update $\mathbf{w}_{t+1}=\mathbf{w}_t-\alpha\,\mathbf{g}_t$.
>
> Stop after a fixed budget of steps, or when $\lVert\mathbf{g}_t\rVert$ (or the validation loss) stops improving.

> [!tip] Intuition: walking downhill in fog
> You stand on a hillside in thick fog and can only feel the slope under your feet. The sensible strategy is to step in the direction where the ground falls most steeply, then feel again. Gradient descent does exactly this. The fog is the reason it needs many small steps: the slope tells you nothing about what lies beyond the next stride. 3Blue1Brown's video shows this picture for neural networks.[^3b1b-gd]

The Taylor model says nothing about **how far** to step: it is only accurate for small steps. To choose $\alpha$ we need to know how fast the gradient can change.

### A quadratic upper bound: L-smoothness

A differentiable function is **L-smooth** if its gradient is $L$-Lipschitz:

$$
\lVert\nabla f(\mathbf{u})-\nabla f(\mathbf{v})\rVert\le L\,\lVert\mathbf{u}-\mathbf{v}\rVert\quad\text{for all }\mathbf{u},\mathbf{v}.
$$

For twice-differentiable $f$ this says that every eigenvalue of the Hessian lies in $[-L,L]$: the curvature is at most $L$ in every direction ([[Multivariable Calculus#The Hessian and second-order Taylor|the Hessian]]). For least squares the Hessian is the constant matrix $\frac1N\mathbf{X}^\top\mathbf{X}$, so $L=\lambda_{\max}(\frac1N\mathbf{X}^\top\mathbf{X})$. For logistic regression $L=\frac14\lambda_{\max}(\frac1N\mathbf{X}^\top\mathbf{X})$, because $\sigma'\le\frac14$.

The key consequence is the **descent lemma**: an L-smooth function lies below a parabola that touches it at the current point,

$$
f(\mathbf{v})\le f(\mathbf{u})+\nabla f(\mathbf{u})^\top(\mathbf{v}-\mathbf{u})+\frac L2\lVert\mathbf{v}-\mathbf{u}\rVert^2.
\tag{2}
$$

> [!proof]- Proof of the descent lemma
> Move along the segment from $\mathbf{u}$ to $\mathbf{v}$: let $\mathbf{d}=\mathbf{v}-\mathbf{u}$ and $\phi(s)=f(\mathbf{u}+s\mathbf{d})$, so $\phi'(s)=\nabla f(\mathbf{u}+s\mathbf{d})^\top\mathbf{d}$. By the fundamental theorem of calculus,
>
> $$
> \begin{aligned}
> &f(\mathbf{v})-f(\mathbf{u})-\nabla f(\mathbf{u})^\top\mathbf{d}\\
> &\qquad=\int_0^1\big(\nabla f(\mathbf{u}+s\mathbf{d})-\nabla f(\mathbf{u})\big)^\top\mathbf{d}\,ds\\
> &\qquad\le\int_0^1Ls\lVert\mathbf{d}\rVert^2\,ds=\frac L2\lVert\mathbf{d}\rVert^2,
> \end{aligned}
> $$
>
> using Cauchy–Schwarz and the Lipschitz bound $\lVert\nabla f(\mathbf{u}+s\mathbf{d})-\nabla f(\mathbf{u})\rVert\le Ls\lVert\mathbf{d}\rVert$. $\blacksquare$

Now **minimize the upper bound** instead of $f$ itself. Setting the gradient of the right-hand side of (2) in $\mathbf{v}$ to zero gives $\mathbf{v}=\mathbf{u}-\frac1L\nabla f(\mathbf{u})$: a gradient step with $\alpha=1/L$. More generally, plugging $\mathbf{v}=\mathbf{w}_{t+1}=\mathbf{w}_t-\alpha\mathbf{g}_t$ into (2) gives the guaranteed decrease

$$
f(\mathbf{w}_{t+1})\le f(\mathbf{w}_t)-\alpha\Big(1-\frac{L\alpha}2\Big)\lVert\mathbf{g}_t\rVert^2.
\tag{3}
$$

The bracket is positive exactly when $\alpha<2/L$, and it is largest at $\alpha=1/L$, where every step decreases $f$ by at least $\lVert\mathbf{g}_t\rVert^2/(2L)$. This **sufficient-decrease** inequality drives every convergence proof below.

> [!example] Example: the descent lemma is tight for a parabola
> Take $f(w)=w^2$. Its second derivative is 2, so $L=2$, and a gradient step gives $w_{t+1}=w_t-2\alpha w_t=(1-2\alpha)w_t$, hence $f(w_{t+1})=(1-2\alpha)^2w_t^2$. The bound (3) gives $w_t^2-\alpha(1-\alpha)\cdot4w_t^2=(1-4\alpha+4\alpha^2)w_t^2$, the same number. For a quadratic with curvature exactly $L$ the upper bound is the function itself.

### Step-size regimes

The parabola example shows everything that can happen. Each step multiplies $w$ by $1-2\alpha$ (Figure 1):
- $\alpha=0.2$: factor $0.6$, a smooth monotone approach to $0$;
- $\alpha=0.5=1/L$: factor $0$, the minimum in one step;
- $\alpha=0.8$: factor $-0.6$, the iterate jumps across the valley each time but still converges;
- $\alpha=1.05>2/L$: factor $-1.1$, the jumps grow and the iteration **diverges**.

![[Probability and Linear Algebra/images/first-order-optimization-step-sizes.svg]]
*Figure 1. Gradient descent on $f(w)=w^2$ from $w_0=1$, where each step multiplies $w$ by $1-2\alpha$. (a) The first five steps drawn on the parabola. (b) The iterates over time. A small step converges monotonically, $\alpha=1/L=0.5$ lands on the minimum at once, a larger step oscillates around it, and any $\alpha>2/L=1$ diverges.*

Too small a step wastes iterations; too large a step diverges. In one dimension $\alpha=1/L$ is perfect. In many dimensions there are many curvatures at once, and no single step size suits them all. That is the subject of the next section.

### Line search

Instead of fixing $\alpha$, we can choose it afresh at every step by (approximately) minimizing $f(\mathbf{w}_t-\alpha\mathbf{g}_t)$ over $\alpha$.

- **Exact line search.** For a quadratic $f(\mathbf{w})=\frac12\mathbf{w}^\top\mathbf{A}\mathbf{w}-\mathbf{b}^\top\mathbf{w}$, the function $\alpha\mapsto f(\mathbf{w}_t-\alpha\mathbf{g}_t)$ is a parabola in $\alpha$, minimized at $\alpha_t=\dfrac{\mathbf{g}_t^\top\mathbf{g}_t}{\mathbf{g}_t^\top\mathbf{A}\mathbf{g}_t}$. For least squares, $\mathbf{A}=\frac1N\mathbf{X}^\top\mathbf{X}$. In general an exact minimization is too expensive.
- **Backtracking (Armijo) line search.** Start from a generous trial step and shrink it until (3) holds approximately. It is cheap, needs no knowledge of $L$, and is the default in classical optimization libraries.[^armijo1966]

> [!example] Algorithm: backtracking line search
> Given $\mathbf{w}_t$ and $\mathbf{g}_t$, choose $\alpha_{\text{init}}$ (say 1), a shrink factor $\rho\in(0,1)$ (say $0.5$) and $c\in(0,1)$ (say $10^{-4}$).
> 1. Set $\alpha=\alpha_{\text{init}}$.
> 2. **While** $f(\mathbf{w}_t-\alpha\mathbf{g}_t)>f(\mathbf{w}_t)-c\,\alpha\,\lVert\mathbf{g}_t\rVert^2$: set $\alpha\leftarrow\rho\,\alpha$.
> 3. Update $\mathbf{w}_{t+1}=\mathbf{w}_t-\alpha\,\mathbf{g}_t$.
>
> For an L-smooth $f$ and $c\le\frac12$, the loop stops once $\alpha\le1/L$, by (3), so it ends after a bounded number of halvings.

Line search needs exact function values, which stochastic methods do not have. In deep learning the step size is therefore set by hand or by a **schedule** ([[#Learning-rate schedules]]).

## Gradient descent on a quadratic

Quadratics are worth studying exactly. Every smooth function looks like one near a minimum (second-order Taylor), least squares **is** one, and the analysis shows precisely where the slowness of gradient descent comes from.

### Splitting into eigen-directions

Let

$$
f(\mathbf{w})=\frac12\mathbf{w}^\top\mathbf{A}\mathbf{w}-\mathbf{b}^\top\mathbf{w},
$$

with $\mathbf{A}$ symmetric positive definite ([[Eigenvalues and Eigenvectors#Positive definite matrices|positive definite matrices]]). Its gradient is $\mathbf{A}\mathbf{w}-\mathbf{b}$, its Hessian is $\mathbf{A}$, and its minimizer solves $\mathbf{A}\mathbf{w}_*=\mathbf{b}$. Least squares is this case with $\mathbf{A}=\frac1N\mathbf{X}^\top\mathbf{X}$ and $\mathbf{b}=\frac1N\mathbf{X}^\top\mathbf{y}$, and $\mathbf{A}\mathbf{w}_*=\mathbf{b}$ are the normal equations ([[Orthogonality and Least Squares#Least squares|least squares]]).

Track the error $\mathbf{e}_t=\mathbf{w}_t-\mathbf{w}_*$. Since $\nabla f(\mathbf{w}_t)=\mathbf{A}\mathbf{w}_t-\mathbf{A}\mathbf{w}_*=\mathbf{A}\mathbf{e}_t$,

$$
\mathbf{e}_{t+1}=\mathbf{e}_t-\alpha\mathbf{A}\mathbf{e}_t=(\mathbf{I}-\alpha\mathbf{A})\,\mathbf{e}_t,
\qquad
\mathbf{e}_t=(\mathbf{I}-\alpha\mathbf{A})^t\,\mathbf{e}_0.
$$

Now diagonalize: $\mathbf{A}=\mathbf{Q}\boldsymbol\Lambda\mathbf{Q}^\top$ with orthonormal eigenvectors $\mathbf{q}_i$ and eigenvalues $\mu=\lambda_1\le\dots\le\lambda_D=L$ ([[Eigenvalues and Eigenvectors]]). Writing the error in the eigenbasis, $\mathbf{e}_t=\sum_ic_{i,t}\,\mathbf{q}_i$, each coordinate evolves **on its own**:

$$
c_{i,t+1}=(1-\alpha\lambda_i)\,c_{i,t},
\qquad
c_{i,t}=(1-\alpha\lambda_i)^t\,c_{i,0}.
\tag{4}
$$

A $D$-dimensional problem has become $D$ independent copies of the parabola example, one per eigen-direction, all sharing a single step size.

### Which step sizes converge, and how fast

From (4), every component shrinks iff $|1-\alpha\lambda_i|<1$ for all $i$, that is, iff

$$
0<\alpha<\frac2{\lambda_{\max}}=\frac2L.
$$

The **largest** eigenvalue limits the step size. The error then shrinks by the factor of the slowest component,

$$
\rho(\alpha)=\max_i\lvert1-\alpha\lambda_i\rvert=\max\big(\lvert1-\alpha\mu\rvert,\,\lvert1-\alpha L\rvert\big),
$$

since $|1-\alpha\lambda|$ is largest at an end of the spectrum. Increasing $\alpha$ speeds up the flat direction ($\mu$) but eventually makes the steep direction ($L$) overshoot. The best compromise balances the two, $1-\alpha\mu=-(1-\alpha L)$. Writing $\kappa=L/\mu=\lambda_{\max}/\lambda_{\min}$ for the **condition number**, it is

$$
\alpha^*=\frac2{\mu+L},
\qquad
\rho(\alpha^*)=\frac{L-\mu}{L+\mu}=\frac{\kappa-1}{\kappa+1}.
\tag{5}
$$

With the simpler choice $\alpha=1/L$ the rate is $1-1/\kappa$, nearly as good. Either way the error after $t$ steps is about $e^{-2t/\kappa}$ of its initial value, so reducing it by a factor $\varepsilon$ takes about $\frac\kappa2\ln\frac1\varepsilon$ steps. The function value gap $f(\mathbf{w}_t)-f_*=\frac12\sum_i\lambda_ic_{i,t}^2$ shrinks by $\rho^2$ per step.

### Worked example by hand

Take $\mathbf{A}=\operatorname{diag}(1,4)$, $\mathbf{b}=\mathbf{0}$, so $f(\mathbf{w})=\frac12w_1^2+2w_2^2$, with minimum at the origin, $\mu=1$, $L=4$ and $\kappa=4$. Start at $\mathbf{w}_0=(4,1)$, where $f=8+2=10$. The gradient is $(w_1,4w_2)$, and the eigenvectors are the coordinate axes.

**The optimal step**, $\alpha^*=2/(1+4)=0.4$. The two factors are $1-0.4\cdot1=0.6$ and $1-0.4\cdot4=-0.6$:

| $t$ | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| $w_1$ (factor $0.6$) | $4$ | $2.4$ | $1.44$ | $0.864$ |
| $w_2$ (factor $-0.6$) | $1$ | $-0.6$ | $0.36$ | $-0.216$ |
| $f(\mathbf{w}_t)$ | $10$ | $3.6$ | $1.296$ | $0.467$ |

Both coordinates shrink by $0.6=(\kappa-1)/(\kappa+1)=3/5$ per step and $f$ by $0.36$; $w_2$ flips sign every time. That sign flip is the **zig-zag**: the steep direction is overshot on purpose, because a smaller step would slow down $w_1$.

**The step $\alpha=1/L=0.25$.** Now the factors are $0.75$ and $0$. The steep coordinate is solved in one step, $\mathbf{w}_1=(3,0)$, and then $w_1$ creeps along: $3, 2.25, 1.69,\dots$, a rate of $0.75=1-1/\kappa$. Figure 2(a) shows both paths.

![[Probability and Linear Algebra/images/first-order-optimization-quadratic-paths.svg]]
*Figure 2. Gradient descent on $f=\frac12(\lambda_1w_1^2+\lambda_2w_2^2)$ from $(4,1)$; the ellipses are level sets. (a) $\kappa=4$: the optimal step $\alpha^*=0.4$ zig-zags across the valley but shrinks the error by $0.6$ per step (14 steps to reduce it 1000-fold); $\alpha=1/L$ kills $w_2$ at once and then crawls along $w_1$ at rate $0.75$. (b) $\kappa=25$: even at its best step size $\alpha^*=2/26$, GD bounces between the steep walls of a long narrow valley, and the error shrinks only by $12/13\approx0.92$ per step (87 steps to reduce it 1000-fold).*

### The condition number and the zig-zag

The condition number $\kappa$ measures how elongated the level sets are: the ellipse $f=c$ has axes proportional to $1/\sqrt{\lambda_i}$, so their ratio is $\sqrt\kappa$. Gradient descent struggles on elongated valleys for a geometric reason. The gradient is perpendicular to the level set, so in a narrow valley it points mostly **across** the valley, towards the nearby wall, not **along** it towards the distant minimum. The step size that is safe for the steep walls is tiny compared with the length of the valley.

The numbers are sobering. For $\kappa=100$ the optimal rate is $99/101\approx0.980$, and reducing the error by $10^6$ takes **691** steps. Real problems are often much worse.

> [!note] Conditioning of least squares
> For least squares $\mathbf{A}=\frac1N\mathbf{X}^\top\mathbf{X}$, whose eigenvalues are the squared singular values of $\mathbf{X}/\sqrt N$. So $\kappa(\mathbf{A})=\kappa(\mathbf{X})^2=(\sigma_{\max}/\sigma_{\min})^2$ ([[Matrix Factorization and Dimensionality Reduction#Singular value decomposition (SVD)|singular values]]). Correlated features make $\sigma_{\min}$ small, and squaring makes it worse. This is the same squaring that makes the normal equations numerically fragile in [[Orthogonality and Least Squares]].

### Feature scaling and conditioning

Much of the ill-conditioning in practice is self-inflicted. Suppose we predict age from height in metres (around $1.7$) and income in rupees (around $10^6$). The diagonal of $\frac1N\mathbf{X}^\top\mathbf{X}$ then holds the mean squared feature values, about $3$ and $10^{12}$, so $\kappa$ is at least $10^{11}$, and gradient descent is useless. The units, not the problem, created the narrow valley.

- **Plain least squares doesn't care** about scaling: rescaling a feature by $c$ rescales its optimal weight by $1/c$, and the predictions do not change.
- **Gradient descent cares**, because scaling stretches the level sets and changes $\kappa$.
- **Regularization cares** too, because $\lambda\lVert\mathbf{w}\rVert^2$ treats all weights alike.

Two standard fixes, applied to each feature column $j$:
1. **Standardization:** $x_{ij}\leftarrow(x_{ij}-\bar x_j)/s_j$, giving every feature mean 0 and variance 1.
2. **Min–max scaling:** $x_{ij}\leftarrow(x_{ij}-x^{\min}_j)/(x^{\max}_j-x^{\min}_j)$, mapping every feature to $[0,1]$.

Compute $\bar x_j$, $s_j$, $x^{\min}_j$ and $x^{\max}_j$ on the **training set only** and reuse them on validation and test data. Standardization makes the diagonal of $\frac1N\mathbf{X}^\top\mathbf{X}$ equal to 1. Correlations between features can still leave $\kappa$ large, but the worst, unit-induced, part is gone. Centring also removes the large eigenvalue that a common offset in all features creates.

> [!tip] Intuition: preconditioning
> Rescaling features is a special case of **preconditioning**: change variables so that the valley looks round, run gradient descent there, and map back. In the original variables this is the step $\mathbf{w}_{t+1}=\mathbf{w}_t-\alpha\mathbf{P}^{-1}\nabla f(\mathbf{w}_t)$ for a positive definite $\mathbf{P}$. The perfect preconditioner for a quadratic is $\mathbf{P}=\mathbf{A}$: then $\mathbf{w}_1=\mathbf{w}_0-\mathbf{A}^{-1}(\mathbf{A}\mathbf{w}_0-\mathbf{b})=\mathbf{w}_*$, which is Newton's method, done in one step. Diagonal preconditioners are cheap, and the adaptive methods ([[#Momentum and adaptive methods]]) can be read as learning a diagonal $\mathbf{P}$ on the fly.

## Convergence rates for smooth and convex functions

The quadratic analysis used eigenvalues. For a general function the role of $\lambda_{\max}$ is played by the smoothness constant $L$, and the role of $\lambda_{\min}$ by a **strong convexity** constant $\mu$. The results below are the standard ones;[^nesterov2018]<sup>,</sup>[^garrigos2023] their proofs are short and are collected in collapsible callouts. All use the step $\alpha=1/L$; any constant $\alpha\le1/L$ gives the same rates with $1/\alpha$ in place of $L$.

| Assumption on $f$ | Guarantee after $t$ steps | Steps for accuracy $\varepsilon$ |
|---|---|---|
| L-smooth, convex | $f(\mathbf{w}_t)-f_*\le\dfrac{L\lVert\mathbf{w}_0-\mathbf{w}_*\rVert^2}{2t}$ | $O(1/\varepsilon)$ |
| L-smooth, $\mu$-strongly convex | $f(\mathbf{w}_t)-f_*\le(1-1/\kappa)^t\,(f(\mathbf{w}_0)-f_*)$ | $O(\kappa\log(1/\varepsilon))$ |
| L-smooth, bounded below by $f_{\inf}$ | $\min_{s<t}\lVert\nabla f(\mathbf{w}_s)\rVert^2\le\dfrac{2L(f(\mathbf{w}_0)-f_{\inf})}{t}$ | $O(1/\varepsilon^2)$ for $\lVert\nabla f\rVert\le\varepsilon$ |

### Smooth convex functions: O(1/t)

> [!note] Theorem (GD on smooth convex functions)
> If $f$ is convex and L-smooth with a minimizer $\mathbf{w}_*$, gradient descent with $\alpha=1/L$ satisfies
>
> $$
> f(\mathbf{w}_t)-f_*\le\frac{L\,\lVert\mathbf{w}_0-\mathbf{w}_*\rVert^2}{2t}.
> $$

The error falls like $1/t$: ten times more accuracy costs ten times more steps. This is much slower than the geometric rate on a well-conditioned quadratic, and it happens when there are directions of almost zero curvature.

> [!proof]- Proof
> Write $\mathbf{g}_t=\nabla f(\mathbf{w}_t)$. Three ingredients:
> 1. Sufficient decrease (3) with $\alpha=1/L$: $f(\mathbf{w}_{t+1})\le f(\mathbf{w}_t)-\frac1{2L}\lVert\mathbf{g}_t\rVert^2$.
> 2. Convexity (the tangent plane lies below the graph): $f(\mathbf{w}_t)\le f_*+\mathbf{g}_t^\top(\mathbf{w}_t-\mathbf{w}_*)$.
> 3. The identity $\mathbf{w}_{t+1}-\mathbf{w}_*=\mathbf{w}_t-\mathbf{w}_*-\frac1L\mathbf{g}_t$, which expands to $\lVert\mathbf{w}_{t+1}-\mathbf{w}_*\rVert^2=\lVert\mathbf{w}_t-\mathbf{w}_*\rVert^2-\frac2L\mathbf{g}_t^\top(\mathbf{w}_t-\mathbf{w}_*)+\frac1{L^2}\lVert\mathbf{g}_t\rVert^2$.
>
> Combining 1 and 2, then using 3 to recognize a difference of squares:
>
> $$
> \begin{aligned}
> f(\mathbf{w}_{t+1})-f_*&\le\mathbf{g}_t^\top(\mathbf{w}_t-\mathbf{w}_*)-\frac1{2L}\lVert\mathbf{g}_t\rVert^2\\
> &=\frac L2\Big(\lVert\mathbf{w}_t-\mathbf{w}_*\rVert^2-\lVert\mathbf{w}_{t+1}-\mathbf{w}_*\rVert^2\Big).
> \end{aligned}
> $$
>
> Summing over $s=0,\dots,t-1$, the right-hand side telescopes to at most $\frac L2\lVert\mathbf{w}_0-\mathbf{w}_*\rVert^2$. By step 1 the values $f(\mathbf{w}_s)$ never increase, so the last one is the smallest and $t\,(f(\mathbf{w}_t)-f_*)\le\sum_{s=1}^{t}(f(\mathbf{w}_s)-f_*)\le\frac L2\lVert\mathbf{w}_0-\mathbf{w}_*\rVert^2$. $\blacksquare$
>
> A side result: the distance $\lVert\mathbf{w}_t-\mathbf{w}_*\rVert$ never increases, since the right-hand side above must be non-negative.

### Strongly convex functions: a linear rate

A function is **$\mu$-strongly convex** ($\mu>0$) if it lies above a parabola at every point, not just above its tangent plane:

$$
f(\mathbf{v})\ge f(\mathbf{u})+\nabla f(\mathbf{u})^\top(\mathbf{v}-\mathbf{u})+\frac\mu2\lVert\mathbf{v}-\mathbf{u}\rVert^2.
$$

For twice-differentiable $f$ this means every Hessian eigenvalue is at least $\mu$ ([[Multivariable Calculus#Convexity|strong convexity]]). An L-smooth, $\mu$-strongly convex function is therefore sandwiched between two parabolas with curvatures $\mu$ and $L$, just like a quadratic with $\lambda_{\min}=\mu$ and $\lambda_{\max}=L$, and its condition number is $\kappa=L/\mu$. Adding a ridge penalty $\frac\lambda2\lVert\mathbf{w}\rVert^2$ to a convex loss makes it $\lambda$-strongly convex.

> [!note] Theorem (GD on strongly convex functions)
> If $f$ is L-smooth and $\mu$-strongly convex, gradient descent with $\alpha=1/L$ satisfies
>
> $$
> f(\mathbf{w}_t)-f_*\le\Big(1-\frac1\kappa\Big)^t\big(f(\mathbf{w}_0)-f_*\big).
> $$

This is called a **linear** rate, because the number of correct digits grows linearly with $t$: on a log plot the error is a straight line (Figure 3a). It matches the quadratic analysis with $\alpha=1/L$ exactly.

> [!proof]- Proof (via the Polyak–Łojasiewicz inequality)
> **Step 1: strong convexity gives a lower bound on the gradient.** Minimize both sides of the strong-convexity inequality over $\mathbf{v}$. The left side becomes $f_*$. The right side is a quadratic in $\mathbf{v}$, minimized at $\mathbf{v}=\mathbf{u}-\frac1\mu\nabla f(\mathbf{u})$, with value $f(\mathbf{u})-\frac1{2\mu}\lVert\nabla f(\mathbf{u})\rVert^2$. Hence
>
> $$
> \lVert\nabla f(\mathbf{u})\rVert^2\ge2\mu\,\big(f(\mathbf{u})-f_*\big)\quad\text{for all }\mathbf{u}.
> $$
>
> This is the **Polyak–Łojasiewicz (PL) inequality**: far from the optimum in value, the gradient cannot be small.
>
> **Step 2: combine with sufficient decrease.** By (3) with $\alpha=1/L$ and then PL,
>
> $$
> \begin{aligned}
> f(\mathbf{w}_{t+1})-f_*&\le f(\mathbf{w}_t)-f_*-\frac{\lVert\mathbf{g}_t\rVert^2}{2L}\\
> &\le\Big(1-\frac\mu L\Big)\big(f(\mathbf{w}_t)-f_*\big).
> \end{aligned}
> $$
>
> Unrolling gives the theorem. $\blacksquare$

Only the PL inequality was used, not convexity itself. Some non-convex functions (for instance, least squares composed with certain over-parameterized models) satisfy PL, and gradient descent then converges linearly to a global minimum anyway.[^karimi2016]

### Nonconvex functions: stationary points

Without convexity we cannot promise a global minimum, but L-smoothness alone still guarantees that the gradient becomes small.

> [!note] Theorem (GD on smooth non-convex functions)
> If $f$ is L-smooth and bounded below by $f_{\inf}$, gradient descent with $\alpha=1/L$ satisfies
>
> $$
> \min_{0\le s<t}\lVert\nabla f(\mathbf{w}_s)\rVert\le\sqrt{\frac{2L\,\big(f(\mathbf{w}_0)-f_{\inf}\big)}{t}}.
> $$

> [!proof]- Proof
> Sum the sufficient-decrease inequality $\frac1{2L}\lVert\nabla f(\mathbf{w}_s)\rVert^2\le f(\mathbf{w}_s)-f(\mathbf{w}_{s+1})$ over $s=0,\dots,t-1$. The right side telescopes to $f(\mathbf{w}_0)-f(\mathbf{w}_t)\le f(\mathbf{w}_0)-f_{\inf}$. The smallest of the $t$ terms on the left is at most their average, so $\min_s\lVert\nabla f(\mathbf{w}_s)\rVert^2\le2L(f(\mathbf{w}_0)-f_{\inf})/t$. $\blacksquare$

A small gradient means a stationary point: a minimum, a maximum or a saddle. Under mild conditions, gradient descent started at a random point almost never converges to a saddle point, and the noise in SGD helps it leave saddles quickly, which is part of why first-order methods work on neural networks.[^bottou2018]

### Is gradient descent optimal?

No. Nesterov proved **lower bounds** for every method that only combines gradients:[^nesterov2018] no such method can beat $O(L\lVert\mathbf{w}_0-\mathbf{w}_*\rVert^2/t^2)$ on smooth convex functions, or the rate $\big(\frac{\sqrt\kappa-1}{\sqrt\kappa+1}\big)^t$ on strongly convex ones. Gradient descent achieves only $1/t$ and $\frac{\kappa-1}{\kappa+1}$. The gap, $\kappa$ versus $\sqrt\kappa$, is a factor of 10 at $\kappa=100$, and **momentum** closes it ([[#Momentum and adaptive methods]]).

![[Probability and Linear Algebra/images/first-order-optimization-rates.svg]]
*Figure 3. Measured convergence of GD with $\alpha=1/L$. (a) Two-dimensional quadratics with $\lambda=(1/\kappa,1)$, from $(1,1)$: the gap falls along straight lines on a log scale, exactly on the predicted $(1-1/\kappa)^{2t}$ (dotted, hidden under the curves). The first step removes the steep component; afterwards $\kappa=25$ reaches $5\times10^{-13}$ in 300 steps, $\kappa=100$ only $10^{-5}$. (b) A 1000-dimensional quadratic with eigenvalues $1/i^2$ (so $\kappa=10^6$, effectively not strongly convex), from $\mathbf{w}_0=\mathbf{1}$: the gap decays like a power of $t$ (here about $t^{-0.54}$), a straight line on log-log axes, and stays below the worst-case bound $L\lVert\mathbf{w}_0-\mathbf{w}_*\rVert^2/(2t)$.*

## Stochastic gradient descent

### One example at a time

Computing $\nabla f=\frac1N\sum_i\nabla f_i$ touches all $N$ examples. When $N$ is a million, one gradient step costs a million example-gradients, and the early steps of training, which only need a rough direction, are wildly over-engineered. **Stochastic gradient descent** (SGD) uses a single random example instead.

> [!example] Algorithm: stochastic gradient descent (SGD)
> Choose $\mathbf{w}_0$ and step sizes $\alpha_0,\alpha_1,\dots$
> **For** $t=0,1,2,\dots$:
> 1. Pick an index $i_t$ uniformly at random from $\{1,\dots,N\}$.
> 2. Compute $\mathbf{g}_t=\nabla f_{i_t}(\mathbf{w}_t)$.
> 3. Update $\mathbf{w}_{t+1}=\mathbf{w}_t-\alpha_t\,\mathbf{g}_t$.

The stochastic gradient is **unbiased**: averaging over the random index,

$$
\mathbb{E}[\mathbf{g}_t\mid\mathbf{w}_t]=\frac1N\sum_{i=1}^N\nabla f_i(\mathbf{w}_t)=\nabla f(\mathbf{w}_t).
$$

Each step is $N$ times cheaper than a GD step and points downhill **on average**. The price is noise, measured by the variance

$$
\begin{aligned}
\sigma^2(\mathbf{w})&=\mathbb{E}\big[\lVert\mathbf{g}_t-\nabla f(\mathbf{w}_t)\rVert^2\,\big|\,\mathbf{w}_t=\mathbf{w}\big]\\
&=\frac1N\sum_{i=1}^N\lVert\nabla f_i(\mathbf{w})-\nabla f(\mathbf{w})\rVert^2.
\end{aligned}
$$

The crucial difference from GD: at the minimum, $\nabla f(\mathbf{w}_*)=\mathbf{0}$ but the individual $\nabla f_i(\mathbf{w}_*)$ are not zero (each example would like a different model), so $\sigma^2(\mathbf{w}_*)>0$. SGD keeps getting kicked even at the optimum.

> [!example] Example: SGD computes a mean
> Take four numbers $y\in\{0,2,4,6\}$ and $f_i(w)=\frac12(w-y_i)^2$, so $f(w)=\frac1N\sum_i\frac12(w-y_i)^2$ is minimized at the mean, $w_*=3$. The per-example gradient is $w-y_i$. At $w_*$ the four gradients are $-3,-1,1,3$: they average to 0 but have variance $\sigma^2=\frac14(9+1+1+9)=5$.
>
> **Decaying step $\alpha_t=1/(t+1)$.** The update $w_{t+1}=w_t-\frac1{t+1}(w_t-y_{i_t})$ is exactly the incremental formula for a running average: $w_1=y_{i_0}$, $w_2=\frac12(y_{i_0}+y_{i_1})$, and so on. SGD returns the **sample mean of the draws so far**, which converges to 3 with variance $5/t$. This is the same incremental update as the action-value estimates in [[Multi-Armed Bandits#Incremental updates|Multi-Armed Bandits]].
>
> **Constant step $\alpha=0.1$.** Now $w_{t+1}=0.9\,w_t+0.1\,y_{i_t}$ is an exponential moving average, which never stops moving. In the long run its variance is $\alpha^2\sigma^2/(1-(1-\alpha)^2)=\alpha\sigma^2/(2-\alpha)=0.5/1.9\approx0.26$, a standard deviation of about $0.5$ around the answer, however long we run. That is the **noise floor**.

### Variance and the noise floor

The noise enters the analysis through one extra term. Repeat the sufficient-decrease argument (2) with a random gradient and take the expectation over $i_t$, using $\mathbb{E}\lVert\mathbf{g}_t\rVert^2=\lVert\nabla f(\mathbf{w}_t)\rVert^2+\sigma^2(\mathbf{w}_t)$:

$$
\begin{aligned}
\mathbb{E}\,f(\mathbf{w}_{t+1})\le f(\mathbf{w}_t)&-\alpha\Big(1-\frac{L\alpha}2\Big)\lVert\nabla f(\mathbf{w}_t)\rVert^2\\
&+\frac{L\alpha^2}2\,\sigma^2(\mathbf{w}_t).
\end{aligned}
\tag{6}
$$

The first term is the progress of GD; the second is the damage done by the noise. Near the optimum the gradient term vanishes but the noise term does not, so progress stalls where the two balance.

> [!note] Theorem (SGD with a constant step)
> Let $f$ be L-smooth and $\mu$-strongly convex, and suppose the gradient noise is bounded, $\sigma^2(\mathbf{w})\le\sigma^2$ for all $\mathbf{w}$. SGD with a constant step $\alpha\le1/L$ satisfies
>
> $$
> \mathbb{E}\,f(\mathbf{w}_t)-f_*\le(1-\alpha\mu)^t\big(f(\mathbf{w}_0)-f_*\big)+\frac{L\alpha\sigma^2}{2\mu}.
> $$

The first term is a linear rate, as for GD; the second is a **floor proportional to $\alpha$**. Halving the step halves the floor but also halves the speed of the first term. The bounded-variance assumption is convenient but strong (for least squares the noise grows with $\lVert\mathbf{w}\rVert$); Bottou, Curtis and Nocedal prove the same result under weaker growth conditions.[^bottou2018]

> [!proof]- Proof
> In (6), $\alpha\le1/L$ makes $1-L\alpha/2\ge\frac12$, and the PL inequality (from the strong-convexity proof) gives $\lVert\nabla f(\mathbf{w}_t)\rVert^2\ge2\mu(f(\mathbf{w}_t)-f_*)$. So, with $\delta_t=\mathbb{E}f(\mathbf{w}_t)-f_*$ and taking a further expectation over $\mathbf{w}_t$,
>
> $$
> \delta_{t+1}\le(1-\alpha\mu)\,\delta_t+\frac{L\alpha^2\sigma^2}2.
> $$
>
> The fixed point of this recursion is $\delta=L\alpha\sigma^2/(2\mu)$. Subtracting it, $\delta_{t+1}-\delta\le(1-\alpha\mu)(\delta_t-\delta)$, so $\delta_t\le\delta+(1-\alpha\mu)^t(\delta_0-\delta)\le\delta+(1-\alpha\mu)^t\delta_0$. $\blacksquare$
>
> Check on the mean example: $L=\mu=1$, $\sigma^2=5$, $\alpha=0.1$ give the bound $0.25$; the exact value is $\frac12\operatorname{Var}(w)=0.13$.

### Decaying step sizes: the Robbins–Monro conditions

To converge exactly, the step must shrink, but not too fast. Robbins and Monro's conditions, from the paper that started stochastic approximation, are[^robbins1951]

$$
\sum_{t=0}^\infty\alpha_t=\infty,
\qquad
\sum_{t=0}^\infty\alpha_t^2<\infty.
$$

The first lets the iterates travel any distance, so they can reach $\mathbf{w}_*$ from anywhere. The second makes the accumulated noise, a sum of independent kicks with variances $\alpha_t^2\sigma^2$, finite, so the iterates eventually settle. The classic choice is $\alpha_t=\alpha_0/(1+t/t_0)$: constant for the first $t_0$ steps, then decaying like $1/t$. With it, SGD on a strongly convex function reaches $\mathbb{E}f(\mathbf{w}_t)-f_*=O(1/t)$.[^bottou2018] The general theory, including the ODE method that explains *why* these iterations converge, is in [[Stochastic Approximation]].

For convex functions that are not strongly convex, the right choice is a step $\propto1/\sqrt t$ and the **average** of the iterates.

> [!note] Theorem (SGD on convex functions, with averaging)
> Let $f$ be convex with minimizer $\mathbf{w}_*$, let $R=\lVert\mathbf{w}_0-\mathbf{w}_*\rVert$, and suppose $\mathbb{E}\lVert\mathbf{g}_t\rVert^2\le G^2$. Run SGD for $T$ steps with the constant step $\alpha=R/(G\sqrt T)$ and let $\bar{\mathbf{w}}_T=\frac1T\sum_{t=0}^{T-1}\mathbf{w}_t$. Then
>
> $$
> \mathbb{E}\,f(\bar{\mathbf{w}}_T)-f_*\le\frac{RG}{\sqrt T}.
> $$

> [!proof]- Proof
> Expand the squared distance after one step:
>
> $$
> \begin{aligned}
> \lVert\mathbf{w}_{t+1}-\mathbf{w}_*\rVert^2=\lVert\mathbf{w}_t-\mathbf{w}_*\rVert^2&-2\alpha\,\mathbf{g}_t^\top(\mathbf{w}_t-\mathbf{w}_*)\\
> &+\alpha^2\lVert\mathbf{g}_t\rVert^2.
> \end{aligned}
> $$
>
> Take expectations. Unbiasedness and convexity give $\mathbb{E}[\mathbf{g}_t^\top(\mathbf{w}_t-\mathbf{w}_*)]=\mathbb{E}[\nabla f(\mathbf{w}_t)^\top(\mathbf{w}_t-\mathbf{w}_*)]\ge\mathbb{E}[f(\mathbf{w}_t)-f_*]$. Rearranging,
>
> $$
> \begin{aligned}
> 2\alpha\,\mathbb{E}[f(\mathbf{w}_t)-f_*]\le{}&\mathbb{E}\lVert\mathbf{w}_t-\mathbf{w}_*\rVert^2-\mathbb{E}\lVert\mathbf{w}_{t+1}-\mathbf{w}_*\rVert^2\\
> &+\alpha^2G^2.
> \end{aligned}
> $$
>
> Summing over $t=0,\dots,T-1$ telescopes the distances: $\sum_t\mathbb{E}[f(\mathbf{w}_t)-f_*]\le\frac{R^2}{2\alpha}+\frac{T\alpha G^2}2$. By convexity (Jensen), $f(\bar{\mathbf{w}}_T)\le\frac1T\sum_tf(\mathbf{w}_t)$, so $\mathbb{E}f(\bar{\mathbf{w}}_T)-f_*\le\frac{R^2}{2\alpha T}+\frac{\alpha G^2}2$. The choice $\alpha=R/(G\sqrt T)$ balances the two terms and gives $RG/\sqrt T$. $\blacksquare$

Averaging the iterates (Polyak–Ruppert averaging) cancels much of the noise without shrinking the step, and the resulting $1/\sqrt T$ rate cannot be improved for this class of problems.[^nemirovski2009]

### Why SGD wins on large datasets

Compare the methods by the number of example-gradients computed, which is what costs time. One GD step costs $N$; one SGD step costs 1. The theorems say GD needs $O(\kappa\log\frac1\varepsilon)$ steps, so $O(N\kappa\log\frac1\varepsilon)$ work, while SGD needs $O(1/\varepsilon)$ steps of unit cost, **independent of $N$**. For low accuracy and large $N$, SGD wins by a wide margin. And low accuracy is all we need: the training loss is only an estimate of the true risk, with a statistical error of order $1/\sqrt N$ or $1/N$, so optimizing far below that error buys nothing.[^bottou2018] Figure 4 shows the trade-off: after one pass over the data SGD is about seven times closer to the optimum than GD, but it then stalls at its floor, while GD keeps going.

## Mini-batch gradient descent

### The algorithm

Mini-batches interpolate between the two extremes: average the gradients of $B$ random examples.

> [!example] Algorithm: mini-batch SGD
> Choose $\mathbf{w}_0$ (zero, or small random values for models that need symmetry breaking), a batch size $B$ and step sizes $\alpha_t$.
> **For** each epoch:
> 1. Shuffle the indices $\{1,\dots,N\}$ and cut them into consecutive batches $\mathcal{B}$ of size $B$.
> 2. **For** each batch: compute $\mathbf{g}_t=\frac1B\sum_{i\in\mathcal{B}}\nabla f_i(\mathbf{w}_t)$ and update $\mathbf{w}_{t+1}=\mathbf{w}_t-\alpha_t\,\mathbf{g}_t$.
>
> $B=N$ is full-batch gradient descent and $B=1$ is SGD. Stop after a fixed budget, or when the loss on held-out data stops improving.

### Variance shrinks like 1/B

The batch gradient is still unbiased, and averaging $B$ independent draws divides the variance by $B$ ([[Joint Distributions#Covariance and correlation|variance of a sum]]). Sampling **without** replacement from a finite dataset does a little better, by the finite-population factor:

$$
\begin{aligned}
\operatorname{Var}(\mathbf{g}_t)&=\frac{\sigma^2}B&&\text{with replacement},\\
\operatorname{Var}(\mathbf{g}_t)&=\frac{\sigma^2}B\cdot\frac{N-B}{N-1}&&\text{without replacement}.
\end{aligned}
$$

At $B=N$ the second is zero, as it should be: the "batch" is the whole dataset.

> [!example] Example: batches of two
> Back to $y\in\{0,2,4,6\}$ with gradient variance $\sigma^2=5$ at $w_*=3$. With $B=2$ drawn without replacement there are six possible batches, with means $1,2,3,3,4,5$. The batch gradient at $w_*$ is $3-(\text{batch mean})$, whose variance is $\frac16(4+1+0+0+1+4)=\frac53$, matching $\frac52\cdot\frac{4-2}{4-1}$. With replacement it would be $\frac52$.

Since the noise term in (6) is $\frac{L\alpha^2}2\cdot\frac{\sigma^2}B$, the noise floor of the constant-step theorem becomes $\frac{L\alpha\sigma^2}{2\mu B}$: it depends on the ratio $\alpha/B$.

### Epochs and shuffling

An **epoch** is one pass through the data. Most implementations do not sample indices independently; they shuffle once per epoch and walk through the shuffled order (**random reshuffling**), so every example is used exactly once per epoch. This is sampling without replacement within an epoch. It usually converges somewhat faster than independent sampling and is never worse in practice. Do not skip the shuffle: a fixed order, for example sorted by class, makes consecutive gradients correlated and can bias the iterates badly.

### The compute trade-off

On a GPU, computing 32 or 256 example-gradients takes about as long as computing one, because they run in parallel. Up to the point where the hardware is saturated, larger batches are therefore free variance reduction. Beyond it, doubling $B$ doubles the cost of a step. Doubling $B$ halves the variance, which helps only while noise is what limits progress. Once the curvature limit $\alpha\le1/L$ binds, a larger batch buys almost nothing per example. The batch size where this happens, the **critical batch size**, can be estimated from the gradient noise and grows during training.[^mccandlish2018] Typical choices are 32 to a few thousand.

### The linear scaling rule

Because the noise floor depends on $\alpha/B$, a practical heuristic says: **when you multiply the batch size by $k$, multiply the step size by $k$**. This keeps the noise level, and roughly the trajectory per epoch, unchanged. Goyal et al. used it to train an ImageNet model with batches of 8192 in one hour, together with a gradual **warm-up** of the step size.[^goyal2017] It is a heuristic with clear limits: the step size can never exceed the stability bound $2/L$, so the rule fails beyond the critical batch size.

![[Probability and Linear Algebra/images/first-order-optimization-sgd.svg]]
*Figure 4. Logistic regression on synthetic data ($N=2000$, $D=10$, ridge $\lambda=10^{-3}$, $L=0.275$); the gap to the optimum is plotted per epoch, so every curve has the same compute budget. (a) After one epoch, SGD ($B=1$) and mini-batch ($B=32$) are at $9\times10^{-3}$ and $7\times10^{-3}$, against $7\times10^{-2}$ for GD. With constant steps they then stall at their noise floors ($8\times10^{-3}$ and $3\times10^{-3}$ after 30 epochs), while GD converges linearly to $1.2\times10^{-5}$. (b) SGD with $B=1$: cutting the step from $0.05$ to $0.0125$ lowers the floor from about $10^{-2}$ to $1.4\times10^{-3}$; a decaying step $\alpha_t=0.05/(1+t/2000)$ keeps improving, to $2.4\times10^{-6}$ after 40 epochs.*

> [!example]- Code: mini-batch SGD for logistic regression in NumPy
> ```python
> import numpy as np
>
> def minibatch_sgd(X, y, alpha, B, epochs, lam=0.0, seed=0):
>     """Labels y in {0,1}; loss = mean log-loss + lam/2 ||w||^2."""
>     rng = np.random.default_rng(seed)
>     N, D = X.shape
>     w = np.zeros(D)
>     for _ in range(epochs):
>         perm = rng.permutation(N)                      # reshuffle every epoch
>         for start in range(0, N, B):
>             idx = perm[start:start + B]
>             p = 1 / (1 + np.exp(-X[idx] @ w))          # sigmoid
>             g = X[idx].T @ (p - y[idx]) / len(idx) + lam * w
>             w = w - alpha * g
>     return w
> ```
> On the data of Figure 4 with $\alpha=0.5$, $B=32$ and 30 epochs, this reaches a gap of $3.6\times10^{-3}$ to the optimum (which we checked against scikit-learn's `LogisticRegression`).

## Momentum and adaptive methods

Plain (S)GD has two weaknesses: it is slow when $\kappa$ is large, and a single step size has to serve directions with very different curvatures and gradient scales. Momentum attacks the first, adaptive methods the second, and Adam combines both.

### Heavy-ball momentum

Polyak's **heavy-ball** method adds a fraction of the previous step to the gradient step:[^polyak1964]

$$
\mathbf{w}_{t+1}=\mathbf{w}_t-\alpha\nabla f(\mathbf{w}_t)+\beta\,(\mathbf{w}_t-\mathbf{w}_{t-1}),
\qquad0\le\beta<1.
$$

Deep-learning libraries write the same method with a velocity $\mathbf{v}$:

$$
\mathbf{v}_{t+1}=\beta\,\mathbf{v}_t+\nabla f(\mathbf{w}_t),
\qquad
\mathbf{w}_{t+1}=\mathbf{w}_t-\alpha\,\mathbf{v}_{t+1},
$$

which is equivalent (since $\mathbf{w}_t-\mathbf{w}_{t-1}=-\alpha\mathbf{v}_t$). The velocity is an exponentially weighted sum of past gradients.

> [!tip] Intuition: a heavy ball rolling in a valley
> Picture a heavy ball rolling down the narrow valley of Figure 2(b). Plain GD is a ball with no inertia: at every point it moves straight down the local slope, which is mostly across the valley. A heavy ball remembers its velocity. The across-valley components alternate in sign from step to step and cancel in the running sum, while the small along-valley component has the same sign every step and accumulates. If the gradient were constant, the step would grow to $\alpha/(1-\beta)$, ten times the plain step for $\beta=0.9$. Goh's interactive article shows these dynamics.[^goh2017]

On a quadratic the analysis is again exact, and it shows the improvement from $\kappa$ to $\sqrt\kappa$.

> [!note] Theorem (heavy ball on a quadratic)
> On a quadratic with Hessian eigenvalues in $[\mu,L]$, choose
>
> $$
> \alpha=\frac4{(\sqrt L+\sqrt\mu)^2},
> \qquad
> \beta=\Big(\frac{\sqrt\kappa-1}{\sqrt\kappa+1}\Big)^2.
> $$
>
> Then every eigen-component of the error shrinks like $\big(\frac{\sqrt\kappa-1}{\sqrt\kappa+1}\big)^t$, up to a factor at most linear in $t$.

> [!proof]- Proof
> In the eigenbasis, as in (4), the component $c_t$ along an eigenvector with eigenvalue $\lambda$ obeys the two-step recursion
>
> $$
> c_{t+1}=(1+\beta-\alpha\lambda)\,c_t-\beta\,c_{t-1}.
> $$
>
> Its solutions are combinations of $z_1^t$ and $z_2^t$, where $z_{1,2}$ are the roots of $z^2-(1+\beta-\alpha\lambda)z+\beta=0$ (or $z^t$ and $tz^t$ for a double root). The product of the roots is $\beta$. If the roots are complex conjugates or equal, both have modulus exactly $\sqrt\beta$, whatever $\lambda$ is. This happens when the discriminant is $\le0$:
>
> $$
> \lvert1+\beta-\alpha\lambda\rvert\le2\sqrt\beta
> \iff
> (1-\sqrt\beta)^2\le\alpha\lambda\le(1+\sqrt\beta)^2.
> $$
>
> To cover the whole range $[\mu,L]$, set $\alpha\mu=(1-\sqrt\beta)^2$ and $\alpha L=(1+\sqrt\beta)^2$. Dividing, $\sqrt\kappa=(1+\sqrt\beta)/(1-\sqrt\beta)$, so $\sqrt\beta=(\sqrt\kappa-1)/(\sqrt\kappa+1)$, and then $\alpha=(1+\sqrt\beta)^2/L=4/(\sqrt L+\sqrt\mu)^2$. Every component decays like $(\sqrt\beta)^t$, with an extra factor $t$ at the two ends of the spectrum, where the root is double. $\blacksquare$

**By hand, for $\kappa=4$** ($\mu=1$, $L=4$): $\alpha=4/(1+2)^2=4/9$ and $\beta=(1/3)^2=1/9$, and every component shrinks by $1/3$ per step, against $0.6$ for GD at its best step. **For $\kappa=100$**: GD's best rate is $99/101\approx0.980$ and heavy ball's is $9/11\approx0.818$. Reducing the error by $10^6$ takes 691 GD steps but only 69 heavy-ball steps (83 in the simulation of Figure 5, because of the factor $t$).

> [!warning] Momentum is not a free lunch
> These parameters are tuned to the quadratic. For general strongly convex functions, the heavy-ball method with the same $\alpha$ and $\beta$ can fail to converge; Lessard, Recht and Packard give a counterexample.[^lessard2016] Nesterov's method below has a guarantee for all smooth convex functions. With **stochastic** gradients, momentum does not remove the noise floor: the effective step is $\alpha/(1-\beta)$, and the floor scales with it.

### Nesterov's accelerated gradient

Nesterov's method evaluates the gradient not at the current point but at a **look-ahead** point, extrapolated along the momentum:

$$
\begin{aligned}
\mathbf{y}_t&=\mathbf{w}_t+\beta_t\,(\mathbf{w}_t-\mathbf{w}_{t-1}),\\
\mathbf{w}_{t+1}&=\mathbf{y}_t-\alpha\,\nabla f(\mathbf{y}_t).
\end{aligned}
$$

The idea: momentum is going to carry us to roughly $\mathbf{y}_t$ anyway, so measure the slope there and correct before overshooting. This small change makes acceleration provable beyond quadratics.[^nesterov2018]
- **Smooth convex $f$**, with $\alpha=1/L$ and $\beta_t=(t-1)/(t+2)$: $f(\mathbf{w}_t)-f_*\le\dfrac{2L\lVert\mathbf{w}_0-\mathbf{w}_*\rVert^2}{(t+1)^2}$, a rate of $1/t^2$ instead of $1/t$.
- **$\mu$-strongly convex $f$**, with $\alpha=1/L$ and constant $\beta=\frac{\sqrt\kappa-1}{\sqrt\kappa+1}$: the gap shrinks like $(1-1/\sqrt\kappa)^t$.

Both match the lower bounds in [[#Is gradient descent optimal?]] up to constants, so Nesterov's method is **optimal** among first-order methods. Sutskever et al. showed that the look-ahead form also helps when training deep networks with SGD; in libraries it is the `nesterov=True` option of SGD with momentum.[^sutskever2013]

![[Probability and Linear Algebra/images/first-order-optimization-momentum.svg]]
*Figure 5. GD, heavy ball and Nesterov on $\frac12(w_1^2+100w_2^2)$ ($\kappa=100$) from $(8,0.3)$, each with the parameters from the text. (a) The first 30 steps. GD bounces between the walls and has barely moved along the valley. Heavy ball overshoots in the steep direction but travels the valley quickly. Nesterov with $\alpha=1/L$ removes the steep component in one step and then accelerates along the valley. (b) Relative error over 120 steps, with the predicted rates dotted: $0.980^t$ for GD (error $9\times10^{-2}$ at the end), $0.818^t$ for heavy ball ($8\times10^{-10}$; the gap to the dotted line is the factor $t$), $0.9^t$ for Nesterov ($4\times10^{-5}$).*

### AdaGrad

Momentum still uses one step size for all coordinates. But coordinates can differ wildly: a word that appears in one document in a thousand has a gradient that is usually zero and occasionally large; a pixel feature measured on a different scale has gradients of a different size. **AdaGrad** gives each coordinate its own step size, inversely proportional to the size of its past gradients:[^duchi2011]

$$
\begin{aligned}
\mathbf{s}_t&=\mathbf{s}_{t-1}+\mathbf{g}_t\odot\mathbf{g}_t,\\
\mathbf{w}_{t+1}&=\mathbf{w}_t-\alpha\,\frac{\mathbf{g}_t}{\sqrt{\mathbf{s}_t}+\epsilon},
\end{aligned}
$$

with $\mathbf{s}_{-1}=\mathbf{0}$, all operations element-wise, and a small $\epsilon$ (say $10^{-8}$) to avoid division by zero. Coordinates with large or frequent gradients get small steps; rare features get large steps when they finally appear. In terms of [[#Feature scaling and conditioning|preconditioning]], AdaGrad learns a diagonal preconditioner from the data. It has strong guarantees for convex problems with sparse features. Its weakness is that $\mathbf{s}_t$ only grows, so the effective step size decays like $1/\sqrt t$ forever, which is often too fast for long non-convex training runs.

### RMSProp

**RMSProp**, proposed by Tieleman and Hinton in a lecture rather than a paper, fixes this by replacing the sum with an exponential moving average, so old gradients are forgotten:[^tieleman2012]

$$
\begin{aligned}
\mathbf{s}_t&=\rho\,\mathbf{s}_{t-1}+(1-\rho)\,\mathbf{g}_t\odot\mathbf{g}_t,\\
\mathbf{w}_{t+1}&=\mathbf{w}_t-\alpha\,\frac{\mathbf{g}_t}{\sqrt{\mathbf{s}_t}+\epsilon},
\end{aligned}
$$

with $\rho\approx0.9$. Now $\sqrt{\mathbf{s}_t}$ estimates the recent root-mean-square size of each coordinate's gradient (hence the name), and every coordinate moves by roughly $\alpha$ per step, whatever its gradient scale.

### Adam

**Adam** combines RMSProp's per-coordinate scaling with momentum, and corrects a bias in both averages:[^kingma2015]

$$
\begin{aligned}
\mathbf{m}_t&=\beta_1\mathbf{m}_{t-1}+(1-\beta_1)\,\mathbf{g}_t,\\
\mathbf{s}_t&=\beta_2\mathbf{s}_{t-1}+(1-\beta_2)\,\mathbf{g}_t\odot\mathbf{g}_t,\\
\hat{\mathbf{m}}_t&=\frac{\mathbf{m}_t}{1-\beta_1^t},\qquad\hat{\mathbf{s}}_t=\frac{\mathbf{s}_t}{1-\beta_2^t},\\
\mathbf{w}_t&=\mathbf{w}_{t-1}-\alpha\,\frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{s}}_t}+\epsilon},
\end{aligned}
$$

for $t=1,2,\dots$, with $\mathbf{m}_0=\mathbf{s}_0=\mathbf{0}$, all operations element-wise, and the defaults $\alpha=10^{-3}$, $\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$. Here $\mathbf{m}_t$ is a momentum (an average of gradients) and $\mathbf{s}_t$ an average of squared gradients.

**Why the bias correction?** Both averages start at zero, so early on they are too small. If the gradients had a constant mean, unrolling the recursion gives $\mathbf{m}_t=(1-\beta_1)\sum_{k=1}^t\beta_1^{t-k}\mathbf{g}_k$, whose expectation is $(1-\beta_1^t)\,\mathbb{E}[\mathbf{g}]$; dividing by $1-\beta_1^t$ removes the bias. The correction matters most for $\mathbf{s}_t$: with $\beta_2=0.999$, after 10 steps $\mathbf{s}_t$ holds only about 1% of its eventual weight, and without the correction the steps would be about ten times too large.

> [!example] Example: Adam's first step
> At $t=1$: $\mathbf{m}_1=0.1\,\mathbf{g}_1$ and $\mathbf{s}_1=0.001\,\mathbf{g}_1\odot\mathbf{g}_1$. After the corrections, $\hat{\mathbf{m}}_1=\mathbf{g}_1$ and $\hat{\mathbf{s}}_1=\mathbf{g}_1\odot\mathbf{g}_1$, so (ignoring $\epsilon$) the first step is
>
> $$
> \mathbf{w}_1=\mathbf{w}_0-\alpha\,\frac{\mathbf{g}_1}{|\mathbf{g}_1|}=\mathbf{w}_0-\alpha\operatorname{sign}(\mathbf{g}_1).
> $$
>
> A gradient $(5,\,-0.01,\,300)$ gives the step $-\alpha\,(1,\,-1,\,1)$: every coordinate moves by exactly $\alpha$, regardless of its gradient's size. Adam behaves like **sign descent with momentum**, and $\alpha$ is directly the per-coordinate step length, which is why its default value transfers well between problems.

Adam is the usual first choice for neural networks because it needs little tuning. It is not always convergent: Reddi, Kale and Kumar built simple convex problems on which it fails, and proposed the AMSGrad fix.[^reddi2018] Well-tuned SGD with momentum sometimes generalizes slightly better on image classification.

### AdamW and decoupled weight decay

For plain SGD, an L2 penalty and **weight decay** are the same thing. The gradient of $f+\frac\lambda2\lVert\mathbf{w}\rVert^2$ is $\nabla f+\lambda\mathbf{w}$, so the step is

$$
\mathbf{w}_{t+1}=(1-\alpha\lambda)\,\mathbf{w}_t-\alpha\nabla f(\mathbf{w}_t):
$$

shrink the weights by a constant factor, then take the gradient step. In Adam they are **not** the same. If the penalty gradient $\lambda\mathbf{w}$ is added to $\mathbf{g}_t$, it passes through the division by $\sqrt{\hat{\mathbf{s}}_t}$, so weights with large gradients are barely decayed and weights with small gradients are decayed strongly, which is not what the regularizer intends. Loshchilov and Hutter's **AdamW** decouples the two: compute the Adam step from $\nabla f$ alone, and apply the decay directly to the weights,[^loshchilov2019]

$$
\mathbf{w}_t=\mathbf{w}_{t-1}-\alpha\Big(\frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{s}}_t}+\epsilon}+\lambda\,\mathbf{w}_{t-1}\Big).
$$

With a zero gradient the weights simply shrink by $(1-\alpha\lambda)$ per step. AdamW, with $\lambda$ around $0.01$ to $0.1$, is the standard optimizer for training transformers.

![[Probability and Linear Algebra/images/first-order-optimization-adaptive.svg]]
*Figure 6. A badly scaled problem: logistic regression with 20 features whose scales fall geometrically from 1 to 0.03 ($\kappa(\frac1N\mathbf{X}^\top\mathbf{X})\approx1100$), mini-batches of 32. Each optimizer was run for 30 epochs at 15 step sizes from $10^{-4}$ to 30. (a) Each at its best step size: SGD ($\alpha=2$) and SGD with momentum ($\alpha=0.33$) end at $6\times10^{-3}$ and $4\times10^{-3}$, while the adaptive methods, which undo the bad scaling coordinate by coordinate, reach $8\times10^{-5}$ (AdaGrad), $2\times10^{-4}$ (RMSProp) and $3\times10^{-4}$ (Adam). (b) The final gap against the step size. Adaptive or not, every method has a best step size and degrades on either side of it; Adam's best value here is $0.022$, close to its default scale.*

> [!example]- Code: Adam and AdamW in NumPy
> ```python
> import numpy as np
>
> def adam_step(w, g, state, alpha=1e-3, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.0):
>     """One Adam step; weight_decay > 0 gives AdamW. state = {"t": 0, "m": 0*w, "s": 0*w}."""
>     state["t"] += 1
>     state["m"] = b1 * state["m"] + (1 - b1) * g
>     state["s"] = b2 * state["s"] + (1 - b2) * g * g
>     m_hat = state["m"] / (1 - b1 ** state["t"])
>     s_hat = state["s"] / (1 - b2 ** state["t"])
>     return w - alpha * (m_hat / (np.sqrt(s_hat) + eps) + weight_decay * w)
> ```
> Over 200 noisy steps this matches scikit-learn's internal `AdamOptimizer` to $2\times10^{-11}$ (with a negligible $\epsilon$, since that implementation places $\epsilon$ slightly differently), and with a zero gradient and `weight_decay` $=\lambda$ it shrinks the weights by exactly $(1-\alpha\lambda)$ per step.

### When to use what

| Method | Remembers | Main strength | Main weakness | Typical use |
|---|---|---|---|---|
| GD (+ line search) | nothing | exact gradients, guarantees | each step costs $N$ | small convex problems |
| SGD / mini-batch SGD | nothing | cheap steps; noise helps escape saddles | noise floor; sensitive to $\kappa$ | large convex problems; with momentum, vision models |
| Heavy ball / Nesterov | velocity | $\sqrt\kappa$ instead of $\kappa$ | one step size for all coordinates | deterministic problems; SGD with momentum $0.9$ |
| AdaGrad | sum of $g^2$ | sparse features, convex guarantees | step size decays forever | sparse linear models, NLP features |
| RMSProp | average of $g^2$ | scale-free steps | no momentum, no bias correction | recurrent networks, RL |
| Adam / AdamW | average of $g$ and $g^2$ | little tuning, scale-free | can fail to converge; generalization gap | default for deep learning, transformers (AdamW) |

## Practical recipe

### Learning-rate schedules

The theory asks for steps that are constant at first (fast progress) and decay later (below the noise floor). Practice adds a **warm-up** at the start, because the first steps with an untrained model and freshly initialized optimizer statistics are the most likely to blow up. Common schedules, over $T$ steps with peak $\alpha_{\max}$ (Figure 7):
- **Step decay:** divide $\alpha$ by 10 at a few fixed epochs.
- **Robbins–Monro style:** $\alpha_t=\alpha_0/(1+t/t_0)$, with the convergence guarantees above.
- **Linear warm-up plus cosine decay:** $\alpha_t=\alpha_{\max}\,t/T_w$ for the first $T_w$ steps, then

$$
\alpha_t=\frac{\alpha_{\max}}2\Big(1+\cos\frac{\pi(t-T_w)}{T-T_w}\Big),
$$

which decays smoothly to zero at $t=T$.[^loshchilov2017] Warm-up is especially important for Adam and for large batches.[^goyal2017]

![[Probability and Linear Algebra/images/first-order-optimization-schedules.svg]]
*Figure 7. Four learning-rate schedules over 1000 steps, relative to the peak step size: constant, step decay (÷10 at steps 300, 600 and 900), the Robbins–Monro-style $\alpha_0/(1+t/t_0)$ with $t_0=100$, and a 50-step linear warm-up followed by cosine decay.*

### Gradient clipping

Occasionally a single batch produces a huge gradient (in recurrent networks, or on a cliff in the loss surface), and one step undoes hours of training. **Clipping by norm** caps the step without changing its direction:[^pascanu2013]

$$
\mathbf{g}_t\leftarrow\mathbf{g}_t\cdot\min\Big(1,\ \frac{c}{\lVert\mathbf{g}_t\rVert}\Big),
$$

with a threshold $c$ around 1, or chosen from the typical gradient norms seen in training. It rarely hurts, and it is standard for recurrent networks, transformers and policy-gradient methods.

### Reading the loss curve

Plot the training loss (smoothed) and the validation loss against steps on a log scale, and read them:
- **The loss goes up, or becomes NaN:** the step size is too large ($\alpha>2/L$ somewhere). Lower it, add warm-up, or clip gradients.
- **The loss decreases very slowly from the start:** the step is too small, or the problem is badly conditioned. Raise $\alpha$, standardize the features, or switch to momentum or Adam.
- **Fast progress, then a noisy plateau:** the noise floor. Decay the step size or increase the batch size.
- **Training loss keeps falling but validation loss rises:** overfitting, which is a statistics problem, not an optimization one. Regularize, stop early or get more data.

Two sanity checks catch most bugs before any tuning: a [[Neural Networks#Gradient checking|numerical gradient check]], and making sure the model can drive the loss to nearly zero on a single small batch.

### A default recipe

1. Standardize the inputs (training-set statistics only).
2. Convex model with modest $N$: full-batch GD with line search, or a quasi-Newton library solver. Large $N$: mini-batch SGD.
3. Deep network: AdamW with $\alpha\approx10^{-3}$ (transformers: smaller), $\beta_1=0.9$, $\beta_2=0.999$, weight decay $0.01$ to $0.1$; or SGD with momentum $0.9$ for convolutional networks.
4. Batch size: as large as fits in memory, up to a few thousand; when you change it, scale $\alpha$ proportionally as a first guess.
5. Schedule: linear warm-up for about 1–5% of training, then cosine decay to near zero.
6. Tune $\alpha$ first, on a log grid (factors of about 3). It matters more than every other hyperparameter.
7. Clip gradients by norm if the loss shows spikes.

> [!example]- Code: gradient descent on a quadratic, checking the predicted rate
> ```python
> import numpy as np
>
> def gradient_descent(grad, w0, alpha, T):
>     w = np.array(w0, dtype=float)
>     for _ in range(T):
>         w = w - alpha * grad(w)
>     return w
>
> A = np.diag([1.0, 4.0]); b = np.array([1.0, 2.0])     # f(w) = 1/2 w^T A w - b^T w
> w_star = np.linalg.solve(A, b)
> w = gradient_descent(lambda w: A @ w - b, [0.0, 0.0], alpha=0.4, T=30)
> print(np.linalg.norm(w - w_star) / np.linalg.norm(w_star), 0.6 ** 30)   # both 2.21e-07
> ```

## Key takeaways

- Training minimizes an average of per-example losses. At ML scale only gradients are affordable, and every method here is "step against the gradient", differing in which gradient, how far and with what memory.
- The negative gradient is the steepest-descent direction of the linear model. **L-smoothness** bounds the function by a parabola, which justifies $\alpha=1/L$, guarantees a decrease of $\lVert\nabla f\rVert^2/(2L)$ per step, and forbids $\alpha\ge2/L$.
- On a quadratic, GD acts independently along each eigenvector, multiplying it by $1-\alpha\lambda_i$. It converges iff $\alpha<2/\lambda_{\max}$; the best rate is $(\kappa-1)/(\kappa+1)$, so about $\kappa$ steps per digit of accuracy. Large $\kappa$ means a narrow valley and zig-zagging; feature scaling and preconditioning reduce it.
- Rates for GD: $O(1/t)$ for smooth convex, $(1-1/\kappa)^t$ for strongly convex (via the PL inequality), and $\min\lVert\nabla f\rVert=O(1/\sqrt t)$ for smooth non-convex functions. Lower bounds show that $\sqrt\kappa$ and $1/t^2$ are possible.
- SGD uses an unbiased one-example gradient: $N$ times cheaper but noisy. A constant step stalls at a noise floor $\propto\alpha\sigma^2$; the Robbins–Monro conditions $\sum\alpha_t=\infty$, $\sum\alpha_t^2<\infty$ give exact convergence. For large $N$ and modest accuracy SGD beats GD.
- Mini-batches divide the variance by $B$, so the floor depends on $\alpha/B$, which motivates the linear scaling rule (up to the critical batch size). Shuffle every epoch.
- Momentum (heavy ball, Nesterov) improves $\kappa$ to $\sqrt\kappa$; Nesterov's method is optimal among first-order methods. AdaGrad, RMSProp and Adam scale each coordinate by its gradient history, a learned diagonal preconditioner. AdamW decouples weight decay from that scaling.
- In practice: standardize, warm up then decay the step size, clip gradients, tune $\alpha$ first, and read the loss curve.

## Where this shows up next

- [[Five Factorizations of a Matrix]] ends the series by explaining why deep networks are trained by these methods rather than by matrix factorizations or Newton's method.
- [[Linear Models]] applies GD, SGD and Newton's method to least squares and logistic regression; [[Support Vector Machines]] solves its problem with specialized solvers but also with SGD on the hinge loss (Pegasos).
- [[Neural Networks]] computes the gradients by backpropagation and trains with the optimizers above; [[Sequence Models#Inductive bias and data|Sequence Models]] uses AdamW, warm-up and cosine schedules for transformers.
- [[Matrix Factorization and Dimensionality Reduction]] fits NMF by multiplicative updates, which are gradient steps with a per-entry step size, and recommender models by alternating least squares.
- [[Stochastic Approximation]] gives the general convergence theory behind SGD (the ODE method, Robbins–Monro conditions, two-timescale iterations), which also covers [[Monte Carlo and Temporal-Difference Learning|TD learning]] and [[Q-Learning and SARSA|Q-learning]].
- [[Policy Gradients]] and [[Actor-Critic Algorithms]] run stochastic gradient ascent on the expected return, where the variance of the gradient estimate is the central difficulty.

<!-- References: footnote definitions below render as the numbered "References" list at the bottom of the page -->

[^3b1b-gd]: 3Blue1Brown (2017). *Gradient descent, how neural networks learn | Deep Learning Chapter 2*. YouTube. [video](https://www.youtube.com/watch?v=IHZwWFHWa-w)
[^armijo1966]: Armijo, L. (1966). *Minimization of Functions Having Lipschitz Continuous First Partial Derivatives*. Pacific Journal of Mathematics, 16(1), 1–3. [doi:10.2140/pjm.1966.16.1](https://doi.org/10.2140/pjm.1966.16.1)
[^nesterov2018]: Nesterov, Y. (2018). *Lectures on Convex Optimization* (2nd ed.), Chapter 2. Springer. [doi:10.1007/978-3-319-91578-4](https://doi.org/10.1007/978-3-319-91578-4)
[^garrigos2023]: Garrigos, G., & Gower, R. M. (2023). *Handbook of Convergence Theorems for (Stochastic) Gradient Methods*. [arXiv:2301.11235](https://arxiv.org/abs/2301.11235)
[^karimi2016]: Karimi, H., Nutini, J., & Schmidt, M. (2016). *Linear Convergence of Gradient and Proximal-Gradient Methods Under the Polyak-Łojasiewicz Condition*. ECML PKDD. [arXiv:1608.04636](https://arxiv.org/abs/1608.04636)
[^bottou2018]: Bottou, L., Curtis, F. E., & Nocedal, J. (2018). *Optimization Methods for Large-Scale Machine Learning*. SIAM Review, 60(2), 223–311. [arXiv:1606.04838](https://arxiv.org/abs/1606.04838)
[^robbins1951]: Robbins, H., & Monro, S. (1951). *A Stochastic Approximation Method*. The Annals of Mathematical Statistics, 22(3), 400–407. [doi:10.1214/aoms/1177729586](https://doi.org/10.1214/aoms/1177729586)
[^nemirovski2009]: Nemirovski, A., Juditsky, A., Lan, G., & Shapiro, A. (2009). *Robust Stochastic Approximation Approach to Stochastic Programming*. SIAM Journal on Optimization, 19(4), 1574–1609. [doi:10.1137/070704277](https://doi.org/10.1137/070704277)
[^mccandlish2018]: McCandlish, S., Kaplan, J., Amodei, D., & OpenAI Dota Team (2018). *An Empirical Model of Large-Batch Training*. [arXiv:1812.06162](https://arxiv.org/abs/1812.06162)
[^goyal2017]: Goyal, P., Dollár, P., Girshick, R., Noordhuis, P., Wesolowski, L., Kyrola, A., Tulloch, A., Jia, Y., & He, K. (2017). *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour*. [arXiv:1706.02677](https://arxiv.org/abs/1706.02677)
[^polyak1964]: Polyak, B. T. (1964). *Some Methods of Speeding up the Convergence of Iteration Methods*. USSR Computational Mathematics and Mathematical Physics, 4(5), 1–17. [doi:10.1016/0041-5553(64)90137-5](https://doi.org/10.1016/0041-5553(64)90137-5)
[^goh2017]: Goh, G. (2017). *Why Momentum Really Works*. Distill. [distill.pub](https://distill.pub/2017/momentum/)
[^lessard2016]: Lessard, L., Recht, B., & Packard, A. (2016). *Analysis and Design of Optimization Algorithms via Integral Quadratic Constraints*. SIAM Journal on Optimization, 26(1), 57–95. [arXiv:1408.3595](https://arxiv.org/abs/1408.3595)
[^sutskever2013]: Sutskever, I., Martens, J., Dahl, G., & Hinton, G. (2013). *On the Importance of Initialization and Momentum in Deep Learning*. ICML. [proceedings.mlr.press](https://proceedings.mlr.press/v28/sutskever13.html)
[^duchi2011]: Duchi, J., Hazan, E., & Singer, Y. (2011). *Adaptive Subgradient Methods for Online Learning and Stochastic Optimization*. Journal of Machine Learning Research, 12, 2121–2159. [jmlr.org](https://www.jmlr.org/papers/v12/duchi11a.html)
[^tieleman2012]: Tieleman, T., & Hinton, G. (2012). *Lecture 6.5: RMSProp*. Neural Networks for Machine Learning (Coursera), lecture 6 slides. [slides (PDF)](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
[^kingma2015]: Kingma, D. P., & Ba, J. (2015). *Adam: A Method for Stochastic Optimization*. ICLR. [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)
[^reddi2018]: Reddi, S. J., Kale, S., & Kumar, S. (2018). *On the Convergence of Adam and Beyond*. ICLR. [arXiv:1904.09237](https://arxiv.org/abs/1904.09237)
[^loshchilov2019]: Loshchilov, I., & Hutter, F. (2019). *Decoupled Weight Decay Regularization*. ICLR. [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)
[^loshchilov2017]: Loshchilov, I., & Hutter, F. (2017). *SGDR: Stochastic Gradient Descent with Warm Restarts*. ICLR. [arXiv:1608.03983](https://arxiv.org/abs/1608.03983)
[^pascanu2013]: Pascanu, R., Mikolov, T., & Bengio, Y. (2013). *On the Difficulty of Training Recurrent Neural Networks*. ICML. [arXiv:1211.5063](https://arxiv.org/abs/1211.5063)
