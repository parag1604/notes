---
title: Linear Models
tags:
  - linear-regression
  - logistic-regression
  - regularization
  - maximum-likelihood
---

> **Prerequisites:** linear algebra (rank, column space, inverses, the SVD) and basic probability (Gaussian and Bernoulli distributions, likelihood).
>
> **TL;DR:**
> - **Linear regression** finds $\mathbf{w}$ so that $\mathbf{X}\mathbf{w}\approx\mathbf{y}$. Geometrically it projects $\mathbf{y}$ onto the column space of $\mathbf{X}$; probabilistically it is maximum likelihood under Gaussian noise. Both views give the same least-squares objective and the normal equations.
> - A Gaussian prior on $\mathbf{w}$ (MAP) gives **ridge** ($L_2$) regularization, and a Laplace prior gives the **lasso** ($L_1$), which produces sparse weights.
> - **Logistic regression** pushes the same linear score through a sigmoid (two classes) or softmax (many classes) and fits it by maximum likelihood, i.e. by minimizing the **cross-entropy**.

## Linear Regression

Bishop[^bishop2006] and Murphy[^murphy2022] are good textbook companions for this lecture.

### Setup

We are given $N$ data points $\mathbf{x}_i\in\mathbb{R}^D$, stacked as the rows of the **design matrix** $\mathbf{X}\in\mathbb{R}^{N\times D}$, with real-valued targets $\mathbf{y}\in\mathbb{R}^N$. A linear model predicts

$$
\hat{y}=\mathbf{w}^\top\mathbf{x}+b,\qquad\mathbf{w}\in\mathbb{R}^D,\ b\in\mathbb{R}.
$$

The bias $b$ can be absorbed into the weights by appending a constant feature $1$ to every $\mathbf{x}_i$, so that $\mathbf{X}$ becomes $N\times(D+1)$. We assume this from now on and simply write $\hat{y}=\mathbf{w}^\top\mathbf{x}$, or $\hat{\mathbf{y}}=\mathbf{X}\mathbf{w}$ for the whole dataset.

### A system of linear equations

Asking for $\mathbf{X}\mathbf{w}=\mathbf{y}$ means solving $N$ linear equations in $D$ unknowns.

- **$N\le D$ (as many unknowns as equations, or more).** If $\mathbf{X}$ has rank $N$, the system is consistent and has *infinitely many* exact solutions. Any of them fits the training data perfectly: the model **memorizes** the data and typically generalizes badly (overfitting).
- **$N>D$ (the practical case: many more data points than features).** With noisy data the system is **inconsistent**: there is *no* $\mathbf{w}$ that satisfies all equations at once. (Inconsistent means no solution, not merely no unique solution.)

So we look for the $\mathbf{w}$ that makes $\mathbf{X}\mathbf{w}$ **as close as possible** to $\mathbf{y}$.

### The geometry: projection onto the column space

![[Machine Learning/images/linear-models-projection.svg]]
*Figure 1: the target vector $\mathbf{y}\in\mathbb{R}^N$ generally lies outside the column space $C(\mathbf{X})$, the set of all achievable predictions $\mathbf{X}\mathbf{w}$. The closest achievable prediction is the orthogonal projection $\hat{\mathbf{y}}=\mathbf{X}\mathbf{w}_*$. The residual is perpendicular to $C(\mathbf{X})$.*

Every prediction $\mathbf{X}\mathbf{w}=\sum_j w_j\,\mathbf{x}_{:,j}$ is a combination of the columns of $\mathbf{X}$, so it lives in the column space $C(\mathbf{X})$. This is a subspace of dimension at most $D$ *inside* $\mathbb{R}^N$. The vector $\mathbf{y}$ usually sticks out of it. To keep as much information as possible, project $\mathbf{y}$ onto $C(\mathbf{X})$ so that the error $\epsilon=\lVert\mathbf{y}-\mathbf{X}\mathbf{w}\rVert_2$ is smallest. The weights $\mathbf{w}$ are simply the *coordinates* of that projection in the basis of columns.

Since $u\mapsto u^2$ is increasing for $u\ge0$, minimizing $\epsilon$ is the same as minimizing $\epsilon^2$. Multiplying by a positive constant doesn't move the minimizer either. So the objective is

$$
\mathcal{L}(\mathbf{w})=\frac12\lVert\mathbf{y}-\mathbf{X}\mathbf{w}\rVert_2^2=\frac12\sum_{i=1}^{N}\big(y_i-\mathbf{w}^\top\mathbf{x}_i\big)^2.
$$

This is (half) the **sum of squared errors**. Dividing by $N$ gives the **mean squared error**, $\frac1N\sum_i(y_i-\hat y_i)^2$. The minimizer is the same, but the scale matters for learning rates and for the regularization strength $\lambda$.

**The same objective from ERM.** Empirical risk minimization with the squared loss $\ell(y,\hat y)=\frac12(y-\hat y)^2$, averaged or summed over the training set, gives exactly $\mathcal{L}(\mathbf{w})$. The geometric and the learning-theoretic views agree.

### Solving it: the normal equations

The objective is a convex quadratic. Its gradient and Hessian are

$$
\begin{aligned}
\nabla_\mathbf{w}\mathcal{L}&=-\mathbf{X}^\top\big(\mathbf{y}-\mathbf{X}\mathbf{w}\big)=\mathbf{X}^\top\big(\mathbf{X}\mathbf{w}-\mathbf{y}\big),\\
\nabla^2_\mathbf{w}\mathcal{L}&=\mathbf{X}^\top\mathbf{X}.
\end{aligned}
$$

$\mathbf{X}^\top\mathbf{X}$ is always positive *semi*-definite, so every stationary point is a global minimum. If $\mathbf{X}$ has full column rank $D$, it is positive *definite*, and the minimum is unique. Setting the gradient to zero gives the **normal equations**:

$$
\mathbf{X}^\top\mathbf{X}\,\mathbf{w}=\mathbf{X}^\top\mathbf{y}\qquad\Longrightarrow\qquad\boxed{\;\mathbf{w}_*=\big(\mathbf{X}^\top\mathbf{X}\big)^{-1}\mathbf{X}^\top\mathbf{y}\;}
$$

The inverse exists exactly when $\mathbf{X}$ has full column rank, which justifies the assumption.

**The projection view gives the same equations.** The residual $\mathbf{y}-\mathbf{X}\mathbf{w}_*$ must be perpendicular to every column of $\mathbf{X}$, i.e. $\mathbf{X}^\top(\mathbf{y}-\mathbf{X}\mathbf{w}_*)=\mathbf{0}$. That is the normal equations again. The word "normal" refers to this orthogonality.

> [!note] When $\mathbf{X}^\top\mathbf{X}$ is singular: the pseudo-inverse
> If the columns of $\mathbf{X}$ are linearly dependent (duplicated or collinear features, or $N<D$), infinitely many $\mathbf{w}$ minimize the loss. Write the SVD $\mathbf{X}=\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$. Here the columns of $\mathbf{V}$ are eigenvectors of $\mathbf{X}^\top\mathbf{X}$, the columns of $\mathbf{U}$ are eigenvectors of $\mathbf{X}\mathbf{X}^\top$, and the singular values on the diagonal of $\boldsymbol{\Sigma}$ are the square roots of their nonzero eigenvalues. The **Moore–Penrose pseudo-inverse** is $\mathbf{X}^+=\mathbf{V}\boldsymbol{\Sigma}^+\mathbf{U}^\top$, where $\boldsymbol{\Sigma}^+$ inverts the *nonzero* singular values and leaves the zeros. Then $\mathbf{w}=\mathbf{X}^+\mathbf{y}$ is the least-squares solution with the **smallest norm**. When $\mathbf{X}$ has full column rank, $\mathbf{X}^+=(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top$. See [[Matrix Factorization and Dimensionality Reduction#The pseudo-inverse|the SVD lecture]] for more.

---

### Iterative methods

Computing $\mathbf{X}^\top\mathbf{X}$ costs $O(ND^2)$ and inverting it costs $O(D^3)$. Both are expensive when $N$ or $D$ is large. Gradient methods avoid the inverse entirely, needing only matrix–vector products.

> [!example] Algorithm: (mini-batch) gradient descent
> Initialize $\mathbf{w}^0=\mathbf{0}$ (or $\mathbf{w}^0\sim\mathcal{N}(\mathbf{0},\sigma_0^2\mathbf{I})$ with small $\sigma_0$). Choose a batch size $B$ and step sizes $\alpha_t$.
> **For** $t=0,1,2,\dots$:
> 1. Sample a mini-batch $\mathcal{B}_t\subseteq\{1,\dots,N\}$ of size $B$, without replacement within an epoch.
> 2. Compute $\mathbf{g}_t=\frac{1}{B}\sum_{i\in\mathcal{B}_t}\big(\mathbf{w}_t^\top\mathbf{x}_i-y_i\big)\,\mathbf{x}_i=\frac1B\tilde{\mathbf{X}}^\top(\tilde{\mathbf{X}}\mathbf{w}_t-\tilde{\mathbf{y}})$, where $\tilde{\mathbf{X}},\tilde{\mathbf{y}}$ are the batch.
> 3. Update $\mathbf{w}_{t+1}=\mathbf{w}_t-\alpha_t\,\mathbf{g}_t$.
>
> $B=N$ is full-batch **gradient descent**; $B=1$ is **stochastic gradient descent** (SGD). Stop after a fixed budget, or when the loss on held-out data stops improving.

- **Full-batch GD** on this quadratic converges with any *constant* step $0<\alpha<2/\lambda_{\max}(\frac1N\mathbf{X}^\top\mathbf{X})$.
- **SGD's** gradient is noisy, so with a constant step it ends up bouncing around $\mathbf{w}_*$. Converging exactly needs decaying steps with $\sum_t\alpha_t=\infty$ and $\sum_t\alpha_t^2<\infty$, for example $\alpha_t\propto1/t$. These are the same Robbins–Monro conditions as in [[Multi-Armed Bandits#Estimating action values|stochastic approximation]].
- **Mini-batches** trade off the two: less noise than $B=1$, much cheaper than $B=N$, and well suited to vectorized hardware. See Bottou et al. for a thorough survey.[^bottou2018]

![[Machine Learning/images/linear-models-gd-paths.svg]]
*Figure 2: paths of full-batch GD, mini-batch SGD and single-sample SGD on a two-parameter least-squares loss (contours), all starting from the same point. Full-batch GD moves smoothly downhill; the stochastic versions zig-zag because each step sees only part of the data, but all head to $\mathbf{w}_*$. The contours are elongated because the two features are correlated, which slows GD down (see the pre-processing aside).*

> [!tip] Aside: data pre-processing
> Suppose we predict age from height in metres (say $1.5$) and weight in kilograms (say $50$). The raw features live on very different scales.
> - **Plain least squares doesn't care.** Rescaling a feature by $c$ simply rescales its optimal weight by $1/c$, so the predictions are unchanged.
> - **Regularization cares,** because the penalty $\lambda\lVert\mathbf{w}\rVert^2$ treats all weights alike. A feature measured in small units needs a large weight, and so gets penalized more.
> - **Gradient descent cares,** because badly scaled features stretch the loss contours (Figure 2) and force a small step size.
>
> Two common fixes, applied per feature column $j$:
> 1. **Min–max scaling:** $x_{ij}\leftarrow\dfrac{x_{ij}-x^{\min}_j}{x^{\max}_j-x^{\min}_j}$, which maps every feature to $[0,1]$.
> 2. **Standardization:** $x_{ij}\leftarrow\dfrac{x_{ij}-\mu_j}{\sigma_j}$, giving every feature mean $0$ and variance $1$.
>
> Compute $x^{\min}_j$, $x^{\max}_j$, $\mu_j$ and $\sigma_j$ on the **training set only**, and reuse them on test data.

---

### The probabilistic perspective: maximum likelihood

Assume the data were generated by a linear function plus independent Gaussian noise:

$$
y_i=\mathbf{w}^\top\mathbf{x}_i+\epsilon_i,\qquad\epsilon_i\overset{\text{iid}}{\sim}\mathcal{N}(0,\sigma^2).
$$

Taking expectations, $\mathbb{E}[y_i\mid\mathbf{x}_i]=\mathbf{w}^\top\mathbf{x}_i+\mathbb{E}[\epsilon_i]=\mathbf{w}^\top\mathbf{x}_i$. The line is the *mean* of $y$ given $\mathbf{x}$. Equivalently, each target is drawn from a Gaussian centred on the line:

$$
y_i\mid\mathbf{x}_i\sim\mathcal{N}\big(\mathbf{w}^\top\mathbf{x}_i,\ \sigma^2\big).
$$

![[Machine Learning/images/linear-models-probabilistic.svg]]
*Figure 3: left, the observed data. Right, the generative story: at each $x$, the target is drawn from a Gaussian (green) centred on the line $\mathbf{w}^\top\mathbf{x}$ (blue).*

The likelihood of one data point and the log-likelihood of the whole i.i.d. dataset are

$$
\begin{aligned}
p(y_i\mid\mathbf{x}_i;\mathbf{w})&=\frac{1}{\sqrt{2\pi\sigma^2}}\exp\Big(-\frac{(y_i-\mathbf{w}^\top\mathbf{x}_i)^2}{2\sigma^2}\Big),\\
\log\prod_{i=1}^N p(y_i\mid\mathbf{x}_i;\mathbf{w})&=\underbrace{-\frac N2\log(2\pi\sigma^2)}_{\text{constant in }\mathbf{w}}-\frac{1}{2\sigma^2}\sum_{i=1}^{N}\big(y_i-\mathbf{w}^\top\mathbf{x}_i\big)^2.
\end{aligned}
$$

Maximizing over $\mathbf{w}$ drops the constant and the positive factor $1/\sigma^2$:

$$
\mathbf{w}_{\text{MLE}}=\arg\max_\mathbf{w}\log p(\mathbf{y}\mid\mathbf{X};\mathbf{w})=\arg\min_\mathbf{w}\frac12\lVert\mathbf{y}-\mathbf{X}\mathbf{w}\rVert_2^2.
$$

**Maximum likelihood under Gaussian noise is least squares.** Maximizing over $\sigma^2$ as well gives the noise estimate $\hat\sigma^2=\frac1N\sum_i(y_i-\mathbf{w}_{\text{MLE}}^\top\mathbf{x}_i)^2$, the mean squared residual.

---

### MAP estimation and regularization

**A Gaussian prior gives ridge regression.** Put a prior on the weights, $\mathbf{w}\sim\mathcal{N}(\mathbf{0},\tau^2\mathbf{I})$, and maximize the posterior $p(\mathbf{w}\mid\mathbf{X},\mathbf{y})\propto p(\mathbf{y}\mid\mathbf{X},\mathbf{w})\,p(\mathbf{w})$. The prior multiplies the likelihood **once**; it is not repeated inside the product over data points:

$$
\begin{aligned}
\mathbf{w}_{\text{MAP}}&=\arg\max_\mathbf{w}\Big[\prod_{i=1}^{N}\mathcal{N}\big(y_i;\mathbf{w}^\top\mathbf{x}_i,\sigma^2\big)\Big]\cdot\mathcal{N}\big(\mathbf{w};\mathbf{0},\tau^2\mathbf{I}\big)\\
&=\arg\min_\mathbf{w}\frac{1}{2\sigma^2}\lVert\mathbf{y}-\mathbf{X}\mathbf{w}\rVert_2^2+\frac{1}{2\tau^2}\mathbf{w}^\top\mathbf{w}\\
&=\arg\min_\mathbf{w}\underbrace{\frac12\lVert\mathbf{y}-\mathbf{X}\mathbf{w}\rVert_2^2+\frac{\lambda}{2}\lVert\mathbf{w}\rVert_2^2}_{\mathcal{L}_{\text{ridge}}},\qquad\lambda=\frac{\sigma^2}{\tau^2}.
\end{aligned}
$$

So the regularization strength $\lambda$ is not arbitrary: it is the **noise-to-prior variance ratio**. A confident prior (small $\tau$) or noisy data (large $\sigma$) mean stronger regularization. This is **$L_2$-regularized least squares**, or **ridge regression**.[^hoerl1970] Its gradient and Hessian are

$$
\nabla_\mathbf{w}\mathcal{L}_{\text{ridge}}=\mathbf{X}^\top(\mathbf{X}\mathbf{w}-\mathbf{y})+\lambda\mathbf{w},\qquad\nabla^2_\mathbf{w}\mathcal{L}_{\text{ridge}}=\mathbf{X}^\top\mathbf{X}+\lambda\mathbf{I}.
$$

For $\lambda>0$ the Hessian is positive **definite** with *no* assumption on $\mathbf{X}$, since every eigenvalue is at least $\lambda$. So the solution always exists and is unique:

$$
\boxed{\;\mathbf{w}_{\text{ridge}}=\big(\mathbf{X}^\top\mathbf{X}+\lambda\mathbf{I}\big)^{-1}\mathbf{X}^\top\mathbf{y}\;}
$$

As with MLE, it can also be found iteratively with mini-batch SGD. In practice, the bias weight is usually left unregularized.

**Why regularize?** Regularization constrains the parameters, trading a little bias for a large reduction in variance. It matters most for over-parameterized models ($D$ large relative to $N$) and correlated features.

**$L_2$ (ridge): shrink everything, smoothly.** In the SVD basis the effect is exact. If $\mathbf{X}=\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$ has singular values $d_j$, ridge predictions are

$$
\hat{\mathbf{y}}_{\text{ridge}}=\sum_j\mathbf{u}_j\,\frac{d_j^2}{d_j^2+\lambda}\,\mathbf{u}_j^\top\mathbf{y}.
$$

Directions in which the data vary a lot (large $d_j$) are barely touched. Directions with little variation (small $d_j$), where the OLS estimate is noisiest, are shrunk hardest.[^esl] One consequence is that correlated features **share** weight instead of one of them taking an arbitrarily large weight. Ridge keeps every feature, just smaller. It does not make the weights equal, and its effect depends on how the features are scaled.

**$L_0$ (subset selection): keep only a few features.** Often only a few of many features matter, as in gene-expression data with thousands of genes. Penalizing the *number* of nonzero weights, $\lVert\mathbf{w}\rVert_0$, selects a small subset. That makes the model cheaper at inference ($O(\text{\#features})$ per prediction) and easier to interpret. But best-subset selection is combinatorial (NP-hard in general), so it is intractable for large $D$.

**$L_1$ (lasso): the convex relaxation.** Replace the count by the sum of absolute values:[^tibshirani1996]

$$
\mathbf{w}_{\text{lasso}}=\arg\min_\mathbf{w}\frac12\lVert\mathbf{y}-\mathbf{X}\mathbf{w}\rVert_2^2+\lambda\lVert\mathbf{w}\rVert_1,\qquad\lVert\mathbf{w}\rVert_1=\sum_{j=1}^{D}\lvert w_j\rvert.
$$

The objective is still convex, and its solutions contain **exact zeros**. In MAP terms, this is a **Laplace prior** $p(w_j)\propto e^{-\lvert w_j\rvert/b}$, with $\lambda=\sigma^2/b$.

![[Machine Learning/images/linear-models-l1-vs-l2.svg]]
*Figure 4: why $L_1$ gives sparsity. The regularized solution is where the loss contours (around the OLS solution) first touch the constraint region. The $L_2$ ball is round, so the contact point generically has every coordinate nonzero. The $L_1$ ball has corners on the axes, and the contours usually hit a corner, which sets some weights exactly to zero.*

There is no closed form for the lasso, because $\lvert w\rvert$ is not differentiable at $0$. For intuition, take orthonormal features ($\mathbf{X}^\top\mathbf{X}=\mathbf{I}$). Then the lasso **soft-thresholds** the least-squares weights:

$$
w_j^{\text{lasso}}=\operatorname{sign}\big(\hat w_j\big)\,\max\big(\lvert\hat w_j\rvert-\lambda,\,0\big).
$$

Small weights are set *exactly* to zero, and large ones are shrunk by $\lambda$. Coordinate descent and proximal-gradient methods apply this step repeatedly and return exact zeros. Plain (sub)gradient descent only gets close (values like $10^{-10}$), which is why such tiny weights are pruned in practice. Dropping zero-weight features shrinks the model and speeds up inference.

---

## Logistic Regression

### From scores to probabilities

Logistic regression is a classical statistical model.[^cox1958] For **classification** with two classes $y\in\{0,1\}$ we still compute a linear score $z=\mathbf{w}^\top\mathbf{x}$, called the **logit**. Then we squash it into a probability with the **sigmoid** (logistic) function:

$$
\begin{aligned}
\sigma(z)&=\frac{1}{1+e^{-z}}=\frac{e^{z}}{1+e^{z}},\\
p(y=1\mid\mathbf{x};\mathbf{w})&=\sigma(\mathbf{w}^\top\mathbf{x}),\qquad p(y=0\mid\mathbf{x};\mathbf{w})=\frac{1}{1+e^{\mathbf{w}^\top\mathbf{x}}}.
\end{aligned}
$$

The *logit* function is the inverse of the sigmoid, $\operatorname{logit}(p)=\log\frac{p}{1-p}$, the **log-odds**. So logistic regression models the log-odds as a linear function: $\log\frac{p(y=1\mid\mathbf{x})}{p(y=0\mid\mathbf{x})}=\mathbf{w}^\top\mathbf{x}$. The decision rule "predict $1$ if $p>\frac12$" is the same as $\mathbf{w}^\top\mathbf{x}>0$, so the **decision boundary is a hyperplane**. That is why logistic regression counts as a *linear* classifier.

![[Machine Learning/images/linear-models-logistic.svg]]
*Figure 5: left, the sigmoid maps any logit to $(0,1)$, with $\sigma(0)=\frac12$. Right, a logistic regression fitted to two classes. The predicted probability changes smoothly across the plane, and the decision boundary $p=\frac12$ is the straight line $\mathbf{w}^\top\mathbf{x}=0$.*

### Maximum likelihood = binary cross-entropy

Each label is a Bernoulli draw, so its likelihood can be written in one line:

$$
p(y_i\mid\mathbf{x}_i;\mathbf{w})=\sigma(\mathbf{w}^\top\mathbf{x}_i)^{y_i}\,\big(1-\sigma(\mathbf{w}^\top\mathbf{x}_i)\big)^{1-y_i}.
$$

For $y_i=1$ only the first factor survives, and for $y_i=0$ only the second. Taking the log of the product over the dataset and negating gives the **negative log-likelihood**, also called the **binary cross-entropy** loss:

$$
\mathcal{L}(\mathbf{w})=-\sum_{i=1}^{N}\Big[y_i\log\sigma(\mathbf{w}^\top\mathbf{x}_i)+(1-y_i)\log\big(1-\sigma(\mathbf{w}^\top\mathbf{x}_i)\big)\Big].
$$

Substituting $\sigma(z)=\frac{e^z}{1+e^z}$ and $1-\sigma(z)=\frac{1}{1+e^z}$ simplifies it nicely:

$$
\begin{aligned}
\mathcal{L}(\mathbf{w})&=-\sum_{i=1}^{N}\Big[y_i\big(\mathbf{w}^\top\mathbf{x}_i-\log(1+e^{\mathbf{w}^\top\mathbf{x}_i})\big)-(1-y_i)\log\big(1+e^{\mathbf{w}^\top\mathbf{x}_i}\big)\Big]\\
&=\sum_{i=1}^{N}\Big[\log\big(1+e^{\mathbf{w}^\top\mathbf{x}_i}\big)-y_i\,\mathbf{w}^\top\mathbf{x}_i\Big].
\end{aligned}
$$

### Optimizing it

Using $\sigma'(z)=\sigma(z)(1-\sigma(z))$, the gradient and Hessian are

$$
\nabla_\mathbf{w}\mathcal{L}=\sum_{i=1}^{N}\big(\sigma(\mathbf{w}^\top\mathbf{x}_i)-y_i\big)\,\mathbf{x}_i=\mathbf{X}^\top\big(\boldsymbol{\sigma}-\mathbf{y}\big),\qquad\nabla^2_\mathbf{w}\mathcal{L}=\mathbf{X}^\top\mathbf{S}\mathbf{X},
$$

where $\boldsymbol{\sigma}=\sigma(\mathbf{X}\mathbf{w})$ elementwise and $\mathbf{S}=\operatorname{diag}\big(\sigma_i(1-\sigma_i)\big)$.

- **Compare with linear regression.** The gradient has exactly the same form, *prediction minus target, times input*.
- **Convex.** The Hessian is positive semi-definite, so the loss is convex and has no spurious local minima.
- **No closed form.** Setting the gradient to zero gives nonlinear equations, so we solve them iteratively: (mini-batch) gradient descent as before, or **Newton's method**.

> [!tip] Newton's method is iteratively reweighted least squares
> A Newton step is $\mathbf{w}\leftarrow\mathbf{w}-(\mathbf{X}^\top\mathbf{S}\mathbf{X})^{-1}\mathbf{X}^\top(\boldsymbol{\sigma}-\mathbf{y})$. It can be rewritten as $\mathbf{w}\leftarrow(\mathbf{X}^\top\mathbf{S}\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{S}\,\mathbf{z}$ with "working targets" $\mathbf{z}=\mathbf{X}\mathbf{w}-\mathbf{S}^{-1}(\boldsymbol{\sigma}-\mathbf{y})$. That is a [[#Weighted linear regression|weighted least-squares]] problem with weights $\sigma_i(1-\sigma_i)$, re-solved at every step. Hence the name **IRLS**.

> [!warning] Separable data breaks maximum likelihood
> If some hyperplane separates the two classes perfectly, scaling its $\mathbf{w}$ up pushes every probability towards $0$ or $1$ and the loss towards $0$. The likelihood keeps increasing, so the MLE does not exist: $\lVert\mathbf{w}\rVert\to\infty$. Regularization fixes this. [[Support Vector Machines]] take a different route and pick the separating hyperplane with the largest margin.

**Regularization** works exactly as for regression. For example, $\mathcal{L}(\mathbf{w})+\frac{\lambda}{2}\lVert\mathbf{w}\rVert_2^2$ corresponds to a Gaussian prior, and $\mathcal{L}(\mathbf{w})+\lambda\lVert\mathbf{w}\rVert_1$ to a sparse (Laplace) prior.

### More than two classes: softmax regression

Let there be $K$ classes. Train $K$ linear models *jointly*, collected as the columns of $\mathbf{W}\in\mathbb{R}^{D\times K}$. Then $\mathbf{W}^\top\mathbf{x}_i\in\mathbb{R}^K$ is a vector of $K$ logits, one per class. Logits can be negative, so simply dividing each by their sum does not give probabilities. Instead, exponentiate and normalize with the **softmax**:

$$
\mu_{ij}=p(y_i=j\mid\mathbf{x}_i;\mathbf{W})=\frac{\exp\big((\mathbf{W}^\top\mathbf{x}_i)_j\big)}{\sum_{k=1}^{K}\exp\big((\mathbf{W}^\top\mathbf{x}_i)_k\big)}.
$$

**The categorical likelihood.** Each label is one of $K$ outcomes, so

$$
\begin{aligned}
p(y_i\mid\mathbf{x}_i;\mathbf{W})&=\prod_{j=1}^{K}\mu_{ij}^{\mathbb{1}\{y_i=j\}},\\
\log p(\mathbf{y}\mid\mathbf{X};\mathbf{W})&=\sum_{i=1}^{N}\sum_{j=1}^{K}\mathbb{1}\{y_i=j\}\log\mu_{ij}=\sum_{i=1}^{N}\log\mu_{i,y_i}.
\end{aligned}
$$

Only the term for the true class survives the inner sum: the indicator just *selects* the predicted probability of the ground-truth label. We minimize the negative log-likelihood. Writing $\mathbf{M}=[\mu_{ij}]$ for the predicted probabilities and $\mathbf{Y}$ for the one-hot labels (both $N\times K$), the gradient is again *prediction minus target, times input*:

$$
\nabla_\mathbf{W}\Big(-\sum_i\log\mu_{i,y_i}\Big)=\mathbf{X}^\top(\mathbf{M}-\mathbf{Y}).
$$

**Two classes recover the sigmoid.** With $K=2$, $\mu_{i1}=\frac{e^{z_1}}{e^{z_1}+e^{z_2}}=\sigma(z_1-z_2)$. Only the *difference* of the two weight vectors matters, so softmax is over-parameterized. One class's weights can be fixed to $\mathbf{0}$ without loss.

> [!tip] Numerical stability
> Compute $\log\sum_k e^{z_k}$ as $m+\log\sum_k e^{z_k-m}$ with $m=\max_k z_k$ (the **log-sum-exp** trick). Otherwise large logits overflow.

> [!note] One-vs-all: a different model
> An alternative is to train $K$ *independent* binary logistic regressions, "class $j$ vs. the rest", relabelling $y'_i=\mathbb{1}\{y_i=j\}$. At test time, predict the class whose model scores highest. This **one-vs-all** (one-vs-rest) scheme is simple and parallel, but it is *not* the maximum-likelihood softmax model. Its $K$ probabilities need not sum to one, and normalizing them with a softmax afterwards does not turn it into one.

### Why "cross-entropy"? NLL, cross-entropy and KL

For a single input $\mathbf{x}$, let $p(y\mid\mathbf{x})$ be the true label distribution and $p_\mathbf{W}(y\mid\mathbf{x})$ the model's. Recall the **entropy** $H(p)=-\sum_yp(y)\log p(y)$ and the **cross-entropy** $H(p,p_\mathbf{W})=-\sum_yp(y)\log p_\mathbf{W}(y)$. Then

$$
\begin{aligned}
D_{\mathrm{KL}}\big(p\,\|\,p_\mathbf{W}\big)&=\sum_y p(y\mid\mathbf{x})\log\frac{p(y\mid\mathbf{x})}{p_\mathbf{W}(y\mid\mathbf{x})}\\
&=\underbrace{\sum_y p(y\mid\mathbf{x})\log p(y\mid\mathbf{x})}_{-H(p)\text{: independent of }\mathbf{W}}\;\underbrace{-\sum_y p(y\mid\mathbf{x})\log p_\mathbf{W}(y\mid\mathbf{x})}_{H(p,\,p_\mathbf{W})\text{: depends on }\mathbf{W}}.
\end{aligned}
$$

So minimizing the KL divergence over $\mathbf{W}$ is the same as minimizing the cross-entropy. In a labelled dataset we take the target distribution to be one-hot, $p(y=j\mid\mathbf{x}_i)=\mathbb{1}\{y_i=j\}$: we "know" the label, ignoring label noise. The cross-entropy then collapses to $-\log p_\mathbf{W}(y_i\mid\mathbf{x}_i)$. Averaging over the training inputs gives exactly the NLL above. **Maximum likelihood minimizes the KL divergence from the empirical data distribution to the model**, which is why the NLL loss is also called the **(categorical) cross-entropy loss**. The same identity reappears in variational autoencoders, language models and [[Imitation Learning#Behavior Cloning (BC)|behavior cloning]].

---

## Weighted linear regression

Suppose each example carries a positive importance weight: $\mathcal{D}=\{(\mathbf{x}_i,y_i,r_i)\}_{i=1}^{N}$ with $r_i>0$. Collect the weights in the diagonal matrix $\mathbf{R}=\operatorname{diag}(r_1,\dots,r_N)$. Weighted ERM minimizes

$$
\mathcal{L}_r(\mathbf{w})=\frac12\sum_{i=1}^{N}r_i\big(y_i-\mathbf{w}^\top\mathbf{x}_i\big)^2=\frac12(\mathbf{X}\mathbf{w}-\mathbf{y})^\top\mathbf{R}\,(\mathbf{X}\mathbf{w}-\mathbf{y}),
$$

Expanding, and using $\mathbf{R}^\top=\mathbf{R}$ (it is diagonal):

$$
(\mathbf{X}\mathbf{w}-\mathbf{y})^\top\mathbf{R}(\mathbf{X}\mathbf{w}-\mathbf{y})=\mathbf{w}^\top\mathbf{X}^\top\mathbf{R}\mathbf{X}\mathbf{w}-2\,\mathbf{w}^\top\mathbf{X}^\top\mathbf{R}\mathbf{y}+\mathbf{y}^\top\mathbf{R}\mathbf{y}.
$$

The gradient is $\nabla_\mathbf{w}\mathcal{L}_r=\mathbf{X}^\top\mathbf{R}(\mathbf{X}\mathbf{w}-\mathbf{y})$. Setting it to zero:

$$
\mathbf{X}^\top\mathbf{R}\mathbf{X}\,\mathbf{w}=\mathbf{X}^\top\mathbf{R}\mathbf{y}\qquad\Longrightarrow\qquad\boxed{\;\mathbf{w}_*=\big(\mathbf{X}^\top\mathbf{R}\mathbf{X}\big)^{-1}\mathbf{X}^\top\mathbf{R}\mathbf{y}\;}
$$

Where weighted least squares shows up:

- **Unequal noise.** It is exactly the MLE when the noise has a different variance for each point, $\sigma_i^2\propto1/r_i$: precise measurements count more.
- **Logistic regression.** It is the inner step of [[#Optimizing it|IRLS]].
- **Reinforcement learning.** Weighting examples by how good they were is the idea behind *reward-weighted regression* and *advantage-weighted* behavior cloning. It is also the spirit of [[Policy Gradients#Policy gradient = weighted maximum likelihood|policy gradients]], which weight log-likelihood gradients by returns.

## Key takeaways

- Least squares $\min_\mathbf{w}\frac12\lVert\mathbf{y}-\mathbf{X}\mathbf{w}\rVert^2$ is the **orthogonal projection** of $\mathbf{y}$ onto $C(\mathbf{X})$, the **ERM** solution under squared loss, and the **MLE** under Gaussian noise. Its solution is the normal equations $\mathbf{X}^\top\mathbf{X}\mathbf{w}=\mathbf{X}^\top\mathbf{y}$, or the pseudo-inverse when $\mathbf{X}$ is rank-deficient.
- **Gradient descent and SGD** avoid the $O(D^3)$ inverse. SGD needs decaying step sizes to converge.
- **MAP:** a Gaussian prior gives **ridge**, $(\mathbf{X}^\top\mathbf{X}+\lambda\mathbf{I})^{-1}\mathbf{X}^\top\mathbf{y}$ with $\lambda=\sigma^2/\tau^2$. A Laplace prior gives the **lasso**, whose $L_1$ corners produce exact zeros (sparse models).
- **Logistic and softmax regression** fit linear logits by maximum likelihood, which is the same as minimizing the cross-entropy and the KL divergence to the data. The loss is convex with gradient $\mathbf{X}^\top(\text{predictions}-\text{targets})$. There is no closed form, so use GD or Newton/IRLS. On separable data the MLE diverges unless regularized.

## Where this shows up next

- [[Discriminant Analysis]]: a *generative* route to linear classifiers. Its posterior turns out to be a logistic or softmax function of $\mathbf{x}$.
- [[Support Vector Machines]]: maximum-margin linear classifiers and the kernel trick.
- [[Imitation Learning]] and [[Policy Gradients]]: maximum likelihood and weighted maximum likelihood, applied to policies.

<!-- References: footnote definitions below render as the numbered "References" list at the bottom of the page -->

[^hoerl1970]: Hoerl, A. E., & Kennard, R. W. (1970). *Ridge Regression: Biased Estimation for Nonorthogonal Problems*. Technometrics, 12(1), 55–67. [doi:10.1080/00401706.1970.10488634](https://doi.org/10.1080/00401706.1970.10488634)
[^tibshirani1996]: Tibshirani, R. (1996). *Regression Shrinkage and Selection via the Lasso*. Journal of the Royal Statistical Society: Series B, 58(1), 267–288. [doi:10.1111/j.2517-6161.1996.tb02080.x](https://doi.org/10.1111/j.2517-6161.1996.tb02080.x)
[^esl]: Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.), Chapters 3–4. [book website](https://hastie.su.domains/ElemStatLearn/)
[^bishop2006]: Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*, Chapters 3–4. Springer. [book page](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/)
[^murphy2022]: Murphy, K. P. (2022). *Probabilistic Machine Learning: An Introduction*, Chapters 10–11. MIT Press. [book website](https://probml.github.io/pml-book/book1.html)
[^cox1958]: Cox, D. R. (1958). *The Regression Analysis of Binary Sequences*. Journal of the Royal Statistical Society: Series B, 20(2), 215–242. [doi:10.1111/j.2517-6161.1958.tb00292.x](https://doi.org/10.1111/j.2517-6161.1958.tb00292.x)
[^bottou2018]: Bottou, L., Curtis, F. E., & Nocedal, J. (2018). *Optimization Methods for Large-Scale Machine Learning*. SIAM Review. [arXiv:1606.04838](https://arxiv.org/abs/1606.04838)
