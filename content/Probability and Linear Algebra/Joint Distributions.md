---
title: Joint Distributions
tags:
  - probability
  - foundations
date: 2026-11-12
draft: false
prev: "[[Random Variables]]"
next: "[[Sample Statistics]]"
---

> **Prerequisites:** [[Probability Basics]] and [[Random Variables]] (PMFs, PDFs, expectation, variance, the law of total expectation).
>
> **Series:** this is part 3 of 5. [[Probability Basics]] → [[Random Variables]] → **Joint Distributions** → [[Sample Statistics]] → [[Markov Chains]].
>
> **TL;DR:**
> - Several random variables are described by their **joint** distribution. Marginals are its row and column sums (its shadows on the axes), and conditionals are its slices, renormalized. Independence means the joint factorizes.
> - Expectation is linear **always**, with no independence needed. Covariance measures linear co-movement; correlation is a cosine between centred variables. Uncorrelated does not mean independent.
> - $\mathbb{E}[Y\mid X]$ is a random variable: the best prediction of $Y$ from $X$. The tower rule and the law of total variance ("explained plus unexplained") follow.
> - The distribution of a sum of independent variables is a convolution, which smooths and widens; repeated, it produces the bell curve.
> - A multivariate Gaussian is a stretched, rotated and shifted standard Gaussian. Its marginals and conditionals are Gaussian, and conditioning on a noisy observation shrinks towards the prior.

## Motivation

Interesting questions involve several quantities at once. Is a patient's blood pressure related to their age? Does a feature predict the label? Given a noisy image, what was the clean one? The answers live not in the individual distributions but in how the variables **depend on each other**. This lecture extends everything in [[Random Variables]] from one variable to several: distributions, expectations, conditioning, and the most important joint distribution of all, the multivariate Gaussian. Chan's Chapter 5 covers this material with many examples.[^chan2026]

## Joint, marginal and conditional distributions

### The joint distribution

A pair of random variables $(X,Y)$ is a random point in the plane. For discrete variables, the **joint PMF**

$$
p_{X,Y}(x,y)=\mathbb{P}(X=x,\ Y=y)
$$

assigns a probability to every cell of a grid, and the probability of any event is the sum over its cells. For continuous variables, a **joint density** $f_{X,Y}$ is a surface over the plane, and probabilities are volumes under it: $\mathbb{P}\big((X,Y)\in R\big)=\iint_Rf_{X,Y}(x,y)\,dx\,dy$.

Figure 1 shows an example. Roll two dice; let $X$ be the first die and $Y$ the larger of the two. If the first die shows $x$, then $Y=x$ when the second die is at most $x$ ($x$ ways), and $Y=y$ for each larger $y$ (one way each). $Y$ can never be smaller than $X$, so half of the table is empty.

![[Probability and Linear Algebra/images/joint-distributions-table.svg]]
*Figure 1. The joint PMF of $X$ = first die and $Y$ = larger of two dice, in units of $1/36$. Summing each row gives the marginal of $X$ (right), and summing each column gives the marginal of $Y$ (bottom). The marginals are literally written in the margins, which is where the name comes from.*

### Marginals: the shadows

The distribution of $X$ alone, its **marginal**, is obtained by adding up the joint over all values of $Y$:

$$
p_X(x)=\sum_yp_{X,Y}(x,y),\qquad f_X(x)=\int f_{X,Y}(x,y)\,dy.
$$

This is the [[Probability Basics#Conditioning on evidence|law of total probability]] with the events $\{Y=y\}$ as the partition. Geometrically, the marginal is the shadow the joint distribution casts onto one axis. In Figure 1 the marginal of $X$ is uniform, as it must be for a fair die, while $Y$ favours large values: $p_Y(y)=(2y-1)/36$.

The joint determines the marginals, but **not conversely**. Many different joint distributions have the same marginals, because the marginals say nothing about how the variables relate. All of the dependence lives in the joint.

### Conditionals: the slices

Conditioning on $X=x$ keeps one row of the table and renormalizes it to sum to 1:

$$
p_{Y\mid X}(y\mid x)=\frac{p_{X,Y}(x,y)}{p_X(x)},\qquad f_{Y\mid X}(y\mid x)=\frac{f_{X,Y}(x,y)}{f_X(x)}.
$$

This is exactly "shrink the universe and renormalize" from [[Probability Basics#Conditioning on evidence|Probability Basics]]. In Figure 1, given that the first die shows 2, the row reads $(0,2,1,1,1,1)/36$, and renormalizing gives $\mathbb{P}(Y=2\mid X=2)=2/6$ and $1/6$ for each larger value.

For continuous variables $\mathbb{P}(X=x)=0$, so the definition needs a moment's thought. Condition on a thin slab $x\le X\le x+\delta$ instead. The joint probability of the slab and an interval of $Y$ is about $\delta\int f_{X,Y}(x,y)\,dy$ over the interval, and the slab's own probability is about $\delta f_X(x)$. The $\delta$'s cancel, and the ratio tends to the formula above. Geometrically, $f_{Y\mid X}(\cdot\mid x)$ is the **slice** of the joint density along the vertical line at $x$, rescaled to have area 1 (Figure 2).

![[Probability and Linear Algebra/images/joint-distributions-conditional.svg]]
*Figure 2. Conditioning slices the joint density. (a) A Gaussian joint density with correlation $\rho=0.8$, cut along three vertical lines. (b) Each slice, renormalized, is the conditional density of $Y$ given $X=x$. They are narrower than the marginal (dashed), because knowing $X$ tells us a lot about $Y$, and their centres follow the line $\mathbb{E}[Y\mid X=x]=\rho x$ (red in panel a).*

Rearranging the definition gives the chain rule $p_{X,Y}=p_{Y\mid X}\,p_X=p_{X\mid Y}\,p_Y$, and with it **Bayes' theorem for random variables**:

$$
p_{X\mid Y}(x\mid y)=\frac{p_{Y\mid X}(y\mid x)\,p_X(x)}{p_Y(y)},\qquad p_Y(y)=\sum_xp_{Y\mid X}(y\mid x)\,p_X(x),
$$

with integrals for densities. This is how every latent-variable model is inverted, from the classes in [[Discriminant Analysis]] to the latent codes of a [[Variational Autoencoders|VAE]].

### Independence

$X$ and $Y$ are **independent** if the joint factorizes into the marginals:

$$
p_{X,Y}(x,y)=p_X(x)\,p_Y(y)\quad\text{for all }x,y,
$$

equivalently, every conditional $p_{Y\mid X}(\cdot\mid x)$ is the same as the marginal $p_Y$. Learning $X$ changes nothing about $Y$. In Figure 1 the variables are clearly dependent: the rows are different. A quick test is the support. If the region where the joint is positive is not a rectangle (here, a triangle), the variables cannot be independent.

A dataset is usually modelled as **i.i.d.**: $N$ independent draws from the same distribution. Independence is what lets the likelihood factorize, $p(\mathcal{D}\mid\theta)=\prod_ip(x_i\mid\theta)$, in every maximum-likelihood derivation of the series, and it turns one $N$-dimensional integral into $N$ one-dimensional ones.

## Expectations of several variables

### Linearity of expectation

LOTUS works in several dimensions: $\mathbb{E}[g(X,Y)]=\sum_{x,y}g(x,y)\,p_{X,Y}(x,y)$. Applying it to $g(x,y)=x+y$ and splitting the sum,

$$
\begin{aligned}
\mathbb{E}[X+Y]&=\sum_{x,y}x\,p_{X,Y}(x,y)+\sum_{x,y}y\,p_{X,Y}(x,y)\\
&=\sum_xx\,p_X(x)+\sum_yy\,p_Y(y)=\mathbb{E}[X]+\mathbb{E}[Y].
\end{aligned}
$$

Summing out the other variable leaves the marginals, and **no independence was needed**. Linearity of expectation holds for any random variables, however dependent, and it is one of the most powerful tools in probability: a complicated count can be split into simple indicators whose expectations are easy, even when the indicators depend on each other.

> [!example] Example: the hat-check problem
> $n$ people check their hats, and the hats are handed back in a random order. How many people get their own hat back, on average?
>
> Let $\mathbb{1}_i$ indicate that person $i$ gets their own hat, so the number of matches is $M=\sum_i\mathbb{1}_i$. The indicators are dependent (if the first $n-1$ people all get their own hats, so does the last). But each person is equally likely to receive any of the $n$ hats, so $\mathbb{E}[\mathbb{1}_i]=1/n$, and by linearity
>
> $$
> \mathbb{E}[M]=\sum_{i=1}^{n}\frac1n=1,
> $$
>
> whether there are 3 people or 3 million. Computing the distribution of $M$ is hard; computing its mean takes one line.

> [!example]- Code: simulating the hat-check problem
> ```python
> import numpy as np
> rng = np.random.default_rng(0)
> n, trials = 20, 100_000
> fixed = np.array([np.sum(rng.permutation(n) == np.arange(n)) for _ in range(trials)])
> print(fixed.mean(), fixed.var())        # both about 1, for any n
> ```

**Products are different.** If $X$ and $Y$ are independent, the joint factorizes and so does the sum: $\mathbb{E}[XY]=\mathbb{E}[X]\,\mathbb{E}[Y]$, and more generally $\mathbb{E}[g(X)h(Y)]=\mathbb{E}[g(X)]\,\mathbb{E}[h(Y)]$. Without independence this generally fails.

### Covariance and correlation

The **covariance** measures how $X$ and $Y$ move together around their means:

$$
\operatorname{Cov}(X,Y)=\mathbb{E}\big[(X-\mu_X)(Y-\mu_Y)\big]=\mathbb{E}[XY]-\mathbb{E}[X]\,\mathbb{E}[Y].
$$

When $X$ is above its mean and $Y$ tends to be above its mean too, the product is mostly positive and so is the covariance. When one tends to be high while the other is low, it is negative. $\operatorname{Cov}(X,X)=\operatorname{Var}(X)$, and independence implies zero covariance by the product rule.

Covariance depends on the units: measuring height in centimetres instead of metres multiplies it by 100. The **correlation** removes the scale:

$$
\rho_{X,Y}=\frac{\operatorname{Cov}(X,Y)}{\sigma_X\,\sigma_Y}\in[-1,1].
$$

There is a geometric way to see why $|\rho|\le1$. Treat the centred variables $U=X-\mu_X$ and $V=Y-\mu_Y$ as vectors, with inner product $\langle U,V\rangle=\mathbb{E}[UV]$. Then $\operatorname{Var}$ is a squared length, $\operatorname{Cov}$ is an inner product, and $\rho$ is the **cosine of the angle** between them. The Cauchy–Schwarz inequality $|\mathbb{E}[UV]|\le\sqrt{\mathbb{E}[U^2]\,\mathbb{E}[V^2]}$ is the statement $|\cos\theta|\le1$. The extremes $\rho=\pm1$ occur exactly when $Y$ is a linear function of $X$, $\rho=0$ when the centred variables are "orthogonal". Figure 3 shows what different correlations look like.

> [!proof]- Proof of the Cauchy–Schwarz inequality
> For every real $t$, $\mathbb{E}\big[(tU+V)^2\big]=t^2\,\mathbb{E}[U^2]+2t\,\mathbb{E}[UV]+\mathbb{E}[V^2]\ge0$. A quadratic in $t$ that is never negative has at most one real root, so its discriminant is not positive: $4\,\mathbb{E}[UV]^2-4\,\mathbb{E}[U^2]\,\mathbb{E}[V^2]\le0$. Equality holds exactly when $tU+V=0$ for some $t$, that is, when $V$ is a multiple of $U$.

![[Probability and Linear Algebra/images/joint-distributions-correlation.svg]]
*Figure 3. 400 samples from bivariate Gaussians with correlation $\rho=0$, $0.5$, $0.9$ and $-0.7$, with the sample correlation in brackets. The correlation measures how tightly the cloud hugs a line, and the sign gives the line's direction. It does not measure the slope.*

> [!warning] Uncorrelated does not mean independent
> Correlation only detects **linear** dependence. Two examples where $\rho=0$ but the variables are strongly dependent (Figure 4):
> - A point uniform on a circle, $X=\cos\Theta$, $Y=\sin\Theta$. By symmetry $\operatorname{Cov}(X,Y)=0$, yet knowing $X$ pins $Y$ down to two values, $\pm\sqrt{1-X^2}$.
> - $X$ uniform on $[-1,1]$ and $Y=X^2$. Then $Y$ is a deterministic function of $X$, but $\operatorname{Cov}(X,Y)=\mathbb{E}[X^3]-\mathbb{E}[X]\,\mathbb{E}[X^2]=0$, because the left half of the parabola cancels the right.

![[Probability and Linear Algebra/images/joint-distributions-uncorrelated.svg]]
*Figure 4. Zero correlation, total dependence. (a) Points uniform on a circle. (b) $Y=X^2$ for $X$ uniform on $[-1,1]$. In both cases the sample correlation is about 0, but $Y$ is (nearly) determined by $X$.*

### Variance of a sum

Expanding the square and using linearity,

$$
\operatorname{Var}(X+Y)=\operatorname{Var}(X)+\operatorname{Var}(Y)+2\operatorname{Cov}(X,Y),
$$

and for $n$ variables $\operatorname{Var}\big(\sum_iX_i\big)=\sum_i\operatorname{Var}(X_i)+\sum_{i\ne j}\operatorname{Cov}(X_i,X_j)$. Positively correlated errors add up and reinforce each other; negatively correlated ones partly cancel, which is the idea behind hedging a portfolio. If the variables are **uncorrelated** (in particular, independent), the variances simply add.

Two consequences are used again and again:
- A Binomial$(n,p)$ variable is a sum of $n$ independent Bernoulli$(p)$ variables, so its variance is $np(1-p)$ ([[Random Variables#Bernoulli and binomial|Random Variables]]).
- The average $\bar X_n$ of $n$ i.i.d. variables with variance $\sigma^2$ has $\operatorname{Var}(\bar X_n)=n\sigma^2/n^2=\sigma^2/n$. Averaging shrinks the spread like $1/\sqrt n$. This is the engine of the [[Sample Statistics#The law of large numbers|law of large numbers]].

## Conditional expectation

### Conditional expectation is a random variable

The **conditional expectation** of $Y$ given $X=x$ is the mean of the conditional distribution:

$$
\mathbb{E}[Y\mid X=x]=\sum_yy\,p_{Y\mid X}(y\mid x)\qquad\text{or}\qquad\int y\,f_{Y\mid X}(y\mid x)\,dy.
$$

It is a function of $x$; call it $h(x)$. In Figure 1, for instance, $h(x)=\big(x^2+\sum_{y=x+1}^{6}y\big)/6$, which rises from $3.5$ at $x=1$ to $6$ at $x=6$. In Figure 2, $h(x)=\rho x$.

Now plug in the random $X$. The result $\mathbb{E}[Y\mid X]=h(X)$ is a **random variable**: before we observe $X$, we do not know which conditional mean will apply. It is the best guess of $Y$ that can be made from $X$, a "summary of $Y$ using only the information in $X$". Its properties:
- **Tower rule (law of total expectation):** $\mathbb{E}\big[\mathbb{E}[Y\mid X]\big]=\mathbb{E}[Y]$. Averaging the conditional averages over the distribution of $X$ recovers the overall average. This is the law of total expectation of [[Random Variables#The law of total expectation|Random Variables]], with the partition $\{X=x\}$.
- **Taking out what is known:** $\mathbb{E}[g(X)\,Y\mid X]=g(X)\,\mathbb{E}[Y\mid X]$. Once $X$ is given, $g(X)$ is a constant.
- **Independence:** if $X$ and $Y$ are independent, $\mathbb{E}[Y\mid X]=\mathbb{E}[Y]$, a constant. $X$ is no help in predicting $Y$.

These are everywhere in the series. A [[Introduction to Reinforcement Learning|value function]] $V^\pi(s)$ is a conditional expectation of the return given the state, and the Bellman equation is the tower rule applied one step ahead. The E-step of [[Clustering and Expectation Maximization|EM]] computes an expectation under a conditional distribution. Tweedie's formula in [[Score-Based Models#Tweedie's formula: the score is the optimal denoiser|Score-Based Models]] is a formula for $\mathbb{E}[\mathbf{x}\mid\tilde{\mathbf{x}}]$.

### The best predictor

Suppose we want to predict $Y$ from $X$ with some function $g(X)$, and measure the error by the mean squared error. The best possible $g$ is the conditional mean:

$$
\begin{aligned}
\mathbb{E}\big[(Y-g(X))^2\big]&=\mathbb{E}\big[(Y-\mathbb{E}[Y\mid X])^2\big]+\mathbb{E}\big[(\mathbb{E}[Y\mid X]-g(X))^2\big]\\
&\ge\mathbb{E}\big[(Y-\mathbb{E}[Y\mid X])^2\big].
\end{aligned}
\tag{1}
$$

So regression with squared loss, however flexible the model, is estimating $\mathbb{E}[Y\mid X]$ ([[Linear Models]]). The error that remains, $Y-\mathbb{E}[Y\mid X]$, is **uncorrelated with every function of $X$**: whatever is left over cannot be predicted from $X$. In the vector-space picture, $\mathbb{E}[Y\mid X]$ is the orthogonal projection of $Y$ onto the space of functions of $X$.

> [!proof]- Proof of (1)
> Write $Y-g(X)=\big(Y-h(X)\big)+\big(h(X)-g(X)\big)$ with $h(X)=\mathbb{E}[Y\mid X]$, and expand the square. The cross term vanishes, by the tower rule and by taking out what is known:
>
> $$
> \begin{aligned}
> \mathbb{E}\big[(Y-h(X))\,(h(X)-g(X))\big]&=\mathbb{E}\Big[(h(X)-g(X))\,\mathbb{E}\big[Y-h(X)\mid X\big]\Big]\\
> &=0,
> \end{aligned}
> $$
>
> since $\mathbb{E}[Y-h(X)\mid X]=h(X)-h(X)=0$. The same calculation with any function in place of $h(X)-g(X)$ shows that the residual $Y-h(X)$ is uncorrelated with every function of $X$.

### The law of total variance

The variance of $Y$ splits into two parts:

$$
\operatorname{Var}(Y)=\underbrace{\mathbb{E}\big[\operatorname{Var}(Y\mid X)\big]}_{\text{within groups: unexplained}}+\underbrace{\operatorname{Var}\big(\mathbb{E}[Y\mid X]\big)}_{\text{between groups: explained}}.
\tag{2}
$$

Picture $X$ as a group label (Figure 5). $Y$ varies for two reasons. Within each group, values scatter around the group mean; averaging that scatter over groups gives the first term. And the group means themselves differ; the variance of the group means is the second term. The second part is "explained" by $X$, because knowing $X$ removes it; the first remains even when $X$ is known. The fraction $\operatorname{Var}(\mathbb{E}[Y\mid X])/\operatorname{Var}(Y)$ is the population version of the $R^2$ of a regression, and (2) is the decomposition behind the analysis of variance (ANOVA).

![[Probability and Linear Algebra/images/joint-distributions-total-variance.svg]]
*Figure 5. The law of total variance. Each group has its own mean (thick bars) and its own scatter. The total variance of $Y$ is the average within-group scatter (unexplained by $X$) plus the spread of the group means around the overall mean (explained by $X$).*

> [!proof]- Proof of (2)
> Let $h(X)=\mathbb{E}[Y\mid X]$. By (1) with $g=\mathbb{E}[Y]$, a constant,
>
> $$
> \operatorname{Var}(Y)=\mathbb{E}\big[(Y-\mathbb{E}[Y])^2\big]=\mathbb{E}\big[(Y-h(X))^2\big]+\mathbb{E}\big[(h(X)-\mathbb{E}[Y])^2\big].
> $$
>
> The second term is $\operatorname{Var}(h(X))$, because $\mathbb{E}[h(X)]=\mathbb{E}[Y]$ by the tower rule. The first is $\mathbb{E}\big[\mathbb{E}[(Y-h(X))^2\mid X]\big]=\mathbb{E}[\operatorname{Var}(Y\mid X)]$, again by the tower rule.

> [!example] Example: random sums
> An insurer receives a random number $N$ of claims in a year, and the claim sizes $X_1,X_2,\dots$ are i.i.d. with mean $\mu$ and variance $\sigma^2$, independent of $N$. The total payout is $S=X_1+\cdots+X_N$, a sum with a random number of terms. Condition on $N$:
> - given $N=n$, $S$ is a sum of $n$ terms, so $\mathbb{E}[S\mid N]=N\mu$ and $\operatorname{Var}(S\mid N)=N\sigma^2$;
> - the tower rule gives $\mathbb{E}[S]=\mathbb{E}[N]\,\mu$ (**Wald's identity**[^wald1944]);
> - total variance gives $\operatorname{Var}(S)=\mathbb{E}[N]\,\sigma^2+\operatorname{Var}(N)\,\mu^2$.
>
> The two parts of the variance have clear meanings: uncertainty in the claim sizes, and uncertainty in how many claims there are. With $N\sim\text{Poisson}(\lambda)$, $\operatorname{Var}(S)=\lambda(\sigma^2+\mu^2)$. The same calculation gives the size of a family tree in a [[Random Variables#Hawkes processes: self-exciting events|Hawkes process]].

> [!example]- Code: checking the random-sum formulas
> ```python
> import numpy as np
> rng = np.random.default_rng(0)
> lam, mu, sigma = 4.0, 2.0, 0.5           # N ~ Poisson(lam), X_i ~ N(mu, sigma^2)
> N = rng.poisson(lam, 200_000)
> S = np.array([rng.normal(mu, sigma, k).sum() for k in N])
> print(S.mean(), lam * mu)                           # tower rule: E[N] E[X]
> print(S.var(), lam * sigma**2 + lam * mu**2)        # total variance: E[N]Var(X) + Var(N)E[X]^2
> ```

## Sums of independent random variables

If $X$ and $Y$ are independent, what is the distribution of $Z=X+Y$? The event $\{Z=z\}$ is the union of the disjoint events $\{X=x,\,Y=z-x\}$, so

$$
p_Z(z)=\sum_xp_X(x)\,p_Y(z-x),\qquad f_Z(z)=\int f_X(x)\,f_Y(z-x)\,dx.
\tag{3}
$$

This operation is the **convolution** of the two distributions, the same one used in signal processing and convolutional networks. Picture it as "flip one distribution, slide it across the other, and add up the overlaps". For two dice, the number of ways to reach a total rises from 1 (for a 2) to 6 (for a 7) and falls back to 1 (for a 12), so the flat PMF of one die becomes a triangle. Convolving again gives a smooth hump, and after ten dice the result is very close to a Gaussian (Figure 6). Each convolution smooths and widens the distribution. That repeated convolution always ends at the Gaussian is the [[Sample Statistics#The central limit theorem|central limit theorem]].

![[Probability and Linear Algebra/images/joint-distributions-convolution.svg]]
*Figure 6. Convolution in action. The PMF of the sum of 1, 2, 3 and 10 fair dice, computed by repeated convolution (3), with a Gaussian of the same mean and variance (red). A flat distribution becomes a triangle, then a hump, then a bell.*

Some families are **closed** under independent sums:
- **Poisson:** Poisson$(\lambda_1)$ + Poisson$(\lambda_2)$ = Poisson$(\lambda_1+\lambda_2)$. In (3) the sum is $e^{-(\lambda_1+\lambda_2)}\sum_x\frac{\lambda_1^x\lambda_2^{z-x}}{x!\,(z-x)!}$, and the binomial theorem collapses it to $e^{-(\lambda_1+\lambda_2)}(\lambda_1+\lambda_2)^z/z!$. This is superposition of [[Random Variables#The Poisson process|Poisson processes]].
- **Gaussian:** $\mathcal{N}(\mu_1,\sigma_1^2)+\mathcal{N}(\mu_2,\sigma_2^2)=\mathcal{N}(\mu_1+\mu_2,\sigma_1^2+\sigma_2^2)$. Means add, and variances add.
- **Binomial:** with the same $p$, Binomial$(n_1,p)$ + Binomial$(n_2,p)$ = Binomial$(n_1+n_2,p)$: just count the trials together.
- **Exponential to Gamma:** the sum of $k$ independent Exponential$(\lambda)$ variables is Gamma$(k,\lambda)$. For $k=2$, (3) gives $\int_0^z\lambda e^{-\lambda x}\lambda e^{-\lambda(z-x)}\,dx=\lambda^2ze^{-\lambda z}$. This is the time of the second arrival in a Poisson process.

[[Sample Statistics#Moment-generating functions|Sample Statistics]] shows a much easier route to all of these: transforms turn convolutions into products.

## Random vectors

### Mean vectors and covariance matrices

With $D$ variables at once, we collect them into a random vector $\mathbf{x}=(X_1,\dots,X_D)^\top$. Its mean is the vector of means, $\boldsymbol\mu=\mathbb{E}[\mathbf{x}]$. All the pairwise covariances form the **covariance matrix**

$$
\boldsymbol\Sigma=\mathbb{E}\big[(\mathbf{x}-\boldsymbol\mu)(\mathbf{x}-\boldsymbol\mu)^\top\big],\qquad\Sigma_{ij}=\operatorname{Cov}(X_i,X_j).
$$

Its diagonal holds the variances. It is symmetric, and it is **positive semi-definite**, because for any direction $\mathbf{v}$,

$$
\mathbf{v}^\top\boldsymbol\Sigma\mathbf{v}=\operatorname{Var}(\mathbf{v}^\top\mathbf{x})\ge0.
$$

That identity is worth remembering: the quadratic form of $\boldsymbol\Sigma$ in the direction $\mathbf v$ is the variance of the data projected onto $\mathbf v$. [[Matrix Factorization and Dimensionality Reduction#Principal component analysis (PCA)|PCA]] finds the directions where it is largest. By linearity, an affine map $\mathbf{y}=\mathbf{A}\mathbf{x}+\mathbf{b}$ has

$$
\mathbb{E}[\mathbf{y}]=\mathbf{A}\boldsymbol\mu+\mathbf{b},\qquad\operatorname{Cov}(\mathbf{y})=\mathbf{A}\boldsymbol\Sigma\mathbf{A}^\top,
$$

for **any** distribution, Gaussian or not.

### The multivariate Gaussian

The most important distribution in the series is the $D$-dimensional Gaussian $\mathcal{N}(\boldsymbol\mu,\boldsymbol\Sigma)$:

$$
p(\mathbf{x})=\frac{1}{(2\pi)^{D/2}|\boldsymbol\Sigma|^{1/2}}\exp\Big(-\frac12(\mathbf{x}-\boldsymbol\mu)^\top\boldsymbol\Sigma^{-1}(\mathbf{x}-\boldsymbol\mu)\Big).
$$

The best way to understand it is to **build** it. Start with a vector $\boldsymbol\epsilon$ of $D$ independent standard Gaussians: a round cloud. Write $\boldsymbol\Sigma=\mathbf{U}\boldsymbol\Lambda\mathbf{U}^\top$ (its eigendecomposition, [[Eigenvalues and Eigenvectors#The spectral theorem|the spectral theorem]]) and set

$$
\mathbf{x}=\boldsymbol\mu+\mathbf{U}\boldsymbol\Lambda^{1/2}\boldsymbol\epsilon.
$$

This stretches the cloud by $\sqrt{\lambda_i}$ along each axis, rotates it by $\mathbf{U}$, and shifts it by $\boldsymbol\mu$ (Figure 7). By the affine rule its covariance is $\mathbf{U}\boldsymbol\Lambda^{1/2}\mathbf{I}\boldsymbol\Lambda^{1/2}\mathbf{U}^\top=\boldsymbol\Sigma$. So the density contours are ellipses whose axes point along the eigenvectors of $\boldsymbol\Sigma$, with half-lengths $\sqrt{\lambda_i}$. Any matrix $\mathbf{L}$ with $\mathbf{L}\mathbf{L}^\top=\boldsymbol\Sigma$ works in place of $\mathbf{U}\boldsymbol\Lambda^{1/2}$; in practice the Cholesky factor is used. Running the recipe backwards, $\boldsymbol\epsilon=\mathbf{L}^{-1}(\mathbf{x}-\boldsymbol\mu)$, **whitens** the data: it turns correlated Gaussian data into independent standard Gaussians. This construction is the reparameterization trick of the [[Variational Autoencoders|VAE]].

![[Probability and Linear Algebra/images/joint-distributions-gaussian-transform.svg]]
*Figure 7. Building a Gaussian. (a) 500 samples of $\boldsymbol\epsilon\sim\mathcal{N}(\mathbf{0},\mathbf{I})$, with the unit circle. (b) Stretching by $\boldsymbol\Lambda^{1/2}$ turns the circle into an axis-aligned ellipse. (c) Rotating by $\mathbf{U}$ and shifting by $\boldsymbol\mu$ gives $\mathcal{N}(\boldsymbol\mu,\boldsymbol\Sigma)$; the arrows are the eigenvectors of $\boldsymbol\Sigma$ scaled by $\sqrt{\lambda_i}$.*

> [!example]- Code: sampling and whitening a multivariate Gaussian
> ```python
> import numpy as np
> rng = np.random.default_rng(0)
> mu = np.array([1.0, 0.5])
> Sigma = np.array([[2.0, 0.8], [0.8, 0.6]])
> L = np.linalg.cholesky(Sigma)                       # Sigma = L L^T
> X = mu + rng.standard_normal((100_000, 2)) @ L.T    # x = mu + L eps
> print(X.mean(0), np.cov(X, rowvar=False))           # about mu and Sigma
> W = np.linalg.solve(L, (X - mu).T).T                # whitening: eps = L^{-1}(x - mu)
> print(np.cov(W, rowvar=False).round(2))             # about the identity
> ```

The Gaussian is closed under almost every operation we need:[^bishop2006]
- **Affine maps:** $\mathbf{A}\mathbf{x}+\mathbf{b}\sim\mathcal{N}(\mathbf{A}\boldsymbol\mu+\mathbf{b},\,\mathbf{A}\boldsymbol\Sigma\mathbf{A}^\top)$.
- **Sums:** independent Gaussians add to a Gaussian, with means and covariances adding. Repeatedly adding Gaussian noise is the forward process of diffusion models.
- **Marginals:** any subset of the coordinates is Gaussian; just keep the corresponding entries of $\boldsymbol\mu$ and $\boldsymbol\Sigma$.
- **Conditionals:** split $\mathbf{x}=(\mathbf{x}_1,\mathbf{x}_2)$. Then $\mathbf{x}_1$ given $\mathbf{x}_2$ is Gaussian, with

$$
\boldsymbol\mu_{1\mid2}=\boldsymbol\mu_1+\boldsymbol\Sigma_{12}\boldsymbol\Sigma_{22}^{-1}(\mathbf{x}_2-\boldsymbol\mu_2),
\qquad
\boldsymbol\Sigma_{1\mid2}=\boldsymbol\Sigma_{11}-\boldsymbol\Sigma_{12}\boldsymbol\Sigma_{22}^{-1}\boldsymbol\Sigma_{21}.
\tag{4}
$$

The conditional mean is **linear** in what we observe, and the conditional covariance does not depend on the observed value at all; it is always smaller than the marginal covariance. In two dimensions with unit variances, (4) gives $Y\mid X=x\sim\mathcal{N}(\rho x,\,1-\rho^2)$: exactly the slices of Figure 2. For a Gaussian, then, the best predictor $\mathbb{E}[Y\mid X]$ is a straight line, and least-squares linear regression loses nothing.

> [!warning] For Gaussians, uncorrelated does mean independent, but only jointly
> If $(X,Y)$ is **jointly** Gaussian and $\rho=0$, then $\boldsymbol\Sigma$ is diagonal, the density factorizes, and $X$ and $Y$ are independent. This is special to the Gaussian. It also needs the *pair* to be Gaussian, not just each variable. Let $X\sim\mathcal{N}(0,1)$ and $Y=SX$ with an independent random sign $S=\pm1$. Then $Y$ is also $\mathcal{N}(0,1)$ and $\operatorname{Cov}(X,Y)=\mathbb{E}[S]\,\mathbb{E}[X^2]=0$, yet $|Y|=|X|$ always.

> [!example] Example: a signal in noise
> A quantity $X\sim\mathcal{N}(0,\tau^2)$ is measured with independent noise, $Y=X+\varepsilon$ with $\varepsilon\sim\mathcal{N}(0,\sigma^2)$. Having seen $Y=y$, what is our best estimate of $X$? The pair $(X,Y)$ is jointly Gaussian with $\operatorname{Var}(Y)=\tau^2+\sigma^2$ and $\operatorname{Cov}(X,Y)=\tau^2$, so (4) gives
>
> $$
> \mathbb{E}[X\mid Y=y]=\frac{\tau^2}{\tau^2+\sigma^2}\,y,\qquad\operatorname{Var}(X\mid Y=y)=\frac{\tau^2\sigma^2}{\tau^2+\sigma^2}.
> $$
>
> The estimate **shrinks** the observation towards the prior mean 0, by a factor that compares the prior's uncertainty $\tau^2$ with the noise $\sigma^2$. Precise measurements ($\sigma^2\ll\tau^2$) are trusted almost fully; noisy ones are shrunk heavily. This is Bayes' theorem for densities with Gaussian prior and likelihood. It is the same shrinkage as ridge regression ([[Linear Models#MAP estimation and regularization|MAP with a Gaussian prior]]), the update step of the Kalman filter, and the simplest case of Tweedie's denoising formula in [[Score-Based Models]].

## Key takeaways

- The joint distribution contains everything. Marginals are its sums (shadows) and lose the dependence; conditionals are its slices, renormalized. Independence means the joint factorizes, and the support must then be a rectangle.
- Expectation is linear for any random variables, dependent or not. Indicators plus linearity compute means of complicated counts.
- Covariance measures linear co-movement and correlation is a cosine. Variances of sums add the covariances; for uncorrelated variables they simply add, so averages of $n$ variables have spread $\sigma/\sqrt n$. Uncorrelated does not imply independent, except for jointly Gaussian variables.
- $\mathbb{E}[Y\mid X]$ is a random variable and the best squared-error predictor of $Y$ from $X$. The tower rule averages it back to $\mathbb{E}[Y]$; total variance splits $\operatorname{Var}(Y)$ into explained and unexplained parts.
- Sums of independent variables convolve their distributions. Poisson, Gaussian and binomial families are closed under sums, and exponentials sum to Gammas.
- A multivariate Gaussian is a stretched, rotated, shifted standard Gaussian. Its marginals and conditionals are Gaussian, its conditional mean is linear, and conditioning on noisy data shrinks towards the prior.

## Where this shows up next

- [[Sample Statistics]] uses the variance of a sum and convolution to prove the law of large numbers and the central limit theorem, and replaces convolutions by products of transforms.
- [[Markov Chains]] builds joint distributions of whole sequences from the chain rule and one conditional independence.
- [[Linear Models]] estimates $\mathbb{E}[Y\mid\mathbf{x}]$; [[Discriminant Analysis]] uses Gaussian class-conditionals and Bayes' theorem for random variables.
- [[Matrix Factorization and Dimensionality Reduction]] diagonalizes the covariance matrix (PCA); [[Clustering and Expectation Maximization]] fits mixtures of Gaussians.
- [[Variational Autoencoders]] and [[Score-Based Models]] use reparameterization, Gaussian conditioning and Tweedie's formula.

<!-- References: footnote definitions below render as the numbered "References" list at the bottom of the page -->

[^chan2026]: Chan, S. H. (2026). *Introduction to Probability for Data Science* (2nd ed.), Chapter 5. Michigan Publishing. [book website](https://probability4datascience.com)
[^wald1944]: Wald, A. (1944). *On Cumulative Sums of Random Variables*. The Annals of Mathematical Statistics, 15(3), 283–296. [doi:10.1214/aoms/1177731235](https://doi.org/10.1214/aoms/1177731235)
[^bishop2006]: Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*, Section 2.3 (The Gaussian distribution). Springer. [book page](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/)
