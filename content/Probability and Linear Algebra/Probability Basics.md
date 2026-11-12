---
title: Probability Basics
tags:
  - probability
  - foundations
date: 2026-11-12
draft: false
---

> **Prerequisites:** none beyond sets, sums, integrals and limits. Every other lecture in the series builds on this one.
>
> **TL;DR:**
> - Probability is built from three ingredients: a sample space $\Omega$ of outcomes, a σ-algebra $\mathcal{F}$ of events, and a probability $\mathbb{P}$ that obeys three axioms (normalized, non-negative, countably additive). Everything else is derived.
> - The first consequences are the workhorses of later proofs: $\mathbb{P}(\emptyset)=0$, monotonicity, inclusion–exclusion, the **union bound**, and continuity of probability.
> - Conditional probability and Bayes' rule update probabilities on evidence. Random variables turn outcomes into numbers and come with PMFs, PDFs, CDFs, expectations and variances.
> - The **law of large numbers** closes the loop: the relative frequency $n_A/n$ that motivated the axioms converges to $\mathbb{P}(A)$. Concentration inequalities say how fast, and the central limit theorem says what the fluctuations look like.
>
> **Notation in this lecture:** random variables are uppercase ($X$, $Y$) and their values lowercase ($x$, $y$), the standard convention in probability. The other lectures write vectors in bold lowercase and use $p(\cdot)$ for both PMFs and densities.

## Motivation

Probability provides a mathematical framework for reasoning under uncertainty. Every lecture in this series leans on it:
- regression assumes Gaussian noise;
- classifiers output class probabilities;
- clustering fits mixture models;
- reinforcement learning averages over random trajectories;
- generative models *are* probability distributions.

There are several interpretations of what a probability *means*. The **frequentist** view, which we use to motivate the definitions, says that $\mathbb{P}(A)$ is the long-run fraction of experiments in which $A$ happens. The **Bayesian** view treats probability as a degree of belief, which is how priors and MAP estimation are used in [[Linear Models#MAP estimation and regularization|Linear Models]].[^jaynes2003] The good news is that both obey exactly the same rules, Kolmogorov's axioms,[^kolmogorov1933] so the mathematics below does not depend on the interpretation.

For a fuller treatment, Bertsekas and Tsitsiklis[^bertsekas2008] and Blitzstein and Hwang[^blitzstein2019] are excellent first courses (both free online); Wasserman[^wasserman2004] and Murphy[^murphy2022] connect probability to statistics and machine learning.

### The frequency model

Suppose $A$ is a possible outcome of an experiment. Run the experiment $n$ times and count how often $A$ occurs, $n_A\le n$. A natural measure of how likely $A$ is, is its **relative frequency**

$$
\frac{n_A}{n}\in[0,1].
$$

It is only an estimate, which settles down as $n\to\infty$. Proving that it does settle down is one of the goals of this lecture. The relative frequency already has the properties we will demand of a probability:
- it lies between 0 and 1;
- the outcome "anything at all" has frequency $n/n=1$;
- if $A$ and $B$ cannot happen together, then $n_{A\cup B}=n_A+n_B$, so the frequencies add.

## I. Axioms of probability

### Sample space and events

In any experiment we fix a **universe of discourse**, the collection of all elementary outcomes we care about. This is the **sample space** $\Omega$, and its elements $\omega\in\Omega$ are the outcomes.
- One coin toss: $\Omega=\{\text{Heads},\text{Tails}\}\equiv\{1,0\}$, writing 1 for Heads.
- Three tosses: $\Omega=\{000,001,010,\dots,111\}$, eight outcomes.

An **event** $A$ is a subset of $\Omega$: the set of outcomes in which "$A$ happens". For three tosses, "exactly two heads" is the event $\{011,101,110\}$ (Figure 1).

![[Probability and Linear Algebra/images/probability-basics-coin-events.svg]]
*Figure 1. The sample space of three coin tosses. The eight outcomes are grouped into four events by the number of heads. Each group is the event $\{X=k\}$ for the random variable $X$ = number of heads, defined later.*

### σ-algebras: which subsets are events

We collect the events in a family $\mathcal{F}$. We want to combine events freely ("not $A$", "$A$ or $B$", "at least one of $A_1,A_2,\dots$"), so $\mathcal{F}$ must be closed under these operations. Formally, $\mathcal{F}$ is a **σ-algebra** on $\Omega$:
1. $\Omega\in\mathcal{F}$;
2. $A\in\mathcal{F}\Rightarrow A^c\in\mathcal{F}$ (closed under complements);
3. $A_1,A_2,\ldots\in\mathcal{F}\Rightarrow\bigcup_{i=1}^{\infty}A_i\in\mathcal{F}$ (closed under countable unions).

Consequences:
- $\emptyset=\Omega^c\in\mathcal{F}$.
- Finite unions are included: pad $A\cup B$ with copies of $\emptyset$, as $A\cup B\cup\emptyset\cup\emptyset\cup\cdots$.
- Countable intersections are included, by De Morgan: $\bigcap_iA_i=\big(\bigcup_iA_i^c\big)^c$.

> [!note] Field vs σ-field
> A family closed under complements and **finite** unions is called a **field** (or algebra). Condition 3, closure under **countable** unions, is what makes it a **σ-field** (σ-algebra). The difference is real. On $\Omega=\mathbb{N}$, the sets that are finite or have a finite complement form a field. But the even numbers are a countable union of singletons from that field, and they are neither finite nor co-finite. Countable unions are needed for limits, such as "the coin eventually lands heads".

How large should $\mathcal{F}$ be?
- For a **finite or countable** $\Omega$, we can take $\mathcal{F}$ to be **all** subsets of $\Omega$ (the power set), and we always will.
- For $\Omega=\mathbb{R}$, "all subsets" is too many: there are sets so irregular that no sensible length can be assigned to them (the Vitali sets). Instead we use the **Borel σ-algebra**, the smallest σ-algebra containing all intervals. It includes every set we will ever meet in practice.

The pair $(\Omega,\mathcal{F})$ is a **measurable space**.

### Kolmogorov's axioms

A set function $\mathbb{P}:\mathcal{F}\to[0,1]$ is a **probability** if:
1. **Normalized:** $\mathbb{P}(\Omega)=1$.
2. **Non-negative:** $\mathbb{P}(A)\ge0$ for all $A\in\mathcal{F}$.
3. **σ-additive:** for every countable collection $A_1,A_2,\ldots\in\mathcal{F}$ of **pairwise disjoint** events ($A_i\cap A_j=\emptyset$ for $i\ne j$),

$$
\mathbb{P}\Big(\bigcup_{i=1}^{\infty}A_i\Big)=\sum_{i=1}^{\infty}\mathbb{P}(A_i).
$$

The triple $(\Omega,\mathcal{F},\mathbb{P})$ is a **probability space**. Each axiom is a property of relative frequencies (see above). Axiom 3 is the frequency rule $n_{A\cup B}/n=n_A/n+n_B/n$, extended to countably many events.

## First consequences

All the rules below follow from the three axioms. Their order matters, because each proof uses only the rules before it.

**1. The empty event has probability zero: $\mathbb{P}(\emptyset)=0$.** Take $A_i=\emptyset$ for every $i$. These are pairwise disjoint and their union is $\emptyset$, so σ-additivity gives $\mathbb{P}(\emptyset)=\sum_{i=1}^{\infty}\mathbb{P}(\emptyset)$. A non-negative number equal to an infinite sum of copies of itself must be 0.

**2. Finite additivity.** For disjoint $A$ and $B$, apply σ-additivity to $A,B,\emptyset,\emptyset,\ldots$. The tail contributes $\mathbb{P}(\emptyset)+\mathbb{P}(\emptyset)+\cdots=0$, by rule 1:

$$
\mathbb{P}(A\cup B)=\mathbb{P}(A)+\mathbb{P}(B).
$$

**3. Complements: $\mathbb{P}(A^c)=1-\mathbb{P}(A)$.** $A$ and $A^c$ are disjoint with union $\Omega$, so $\mathbb{P}(A)+\mathbb{P}(A^c)=\mathbb{P}(\Omega)=1$.

**4. Monotonicity: $A\subseteq B\Rightarrow\mathbb{P}(A)\le\mathbb{P}(B)$.** Write $B=A\cup(B\setminus A)$, a disjoint union because $A\subseteq B$. Then $\mathbb{P}(B)=\mathbb{P}(A)+\mathbb{P}(B\setminus A)\ge\mathbb{P}(A)$, since the second term is non-negative.

**5. Inclusion–exclusion.** For any $A$ and $B$,

$$
\mathbb{P}(A\cup B)=\mathbb{P}(A)+\mathbb{P}(B)-\mathbb{P}(A\cap B).
$$

Split into disjoint pieces: $A\cup B=A\cup(B\setminus A)$ and $B=(B\setminus A)\cup(A\cap B)$. Subtracting the two finite-additivity equations gives the result. Adding $\mathbb{P}(A)$ and $\mathbb{P}(B)$ counts the overlap twice, so we subtract it once.

> [!warning] A note on the order of proofs
> It is tempting to prove $\mathbb{P}(\emptyset)=0$ from $\mathbb{P}(\Omega\cup\emptyset)=\mathbb{P}(\Omega)+\mathbb{P}(\emptyset)$. That uses finite additivity, which is not an axiom. Deriving it from σ-additivity by padding with copies of $\emptyset$ already needs $\mathbb{P}(\emptyset)=0$ for the padding to contribute nothing. The argument in rule 1 avoids the circle.

### The union bound

**6. Union bound (Boole's inequality).** For any events $A_1,A_2,\ldots\in\mathcal{F}$, disjoint or not,

$$
\mathbb{P}\Big(\bigcup_{i=1}^{\infty}A_i\Big)\le\sum_{i=1}^{\infty}\mathbb{P}(A_i).
\tag{1}
$$

The idea is to make the events disjoint without changing their union, and then use σ-additivity (Figure 2b). Define

$$
E_1=A_1,\qquad E_k=A_k\cap\Big(\bigcup_{i=1}^{k-1}A_i\Big)^c\quad(k\ge2),
$$

so $E_k$ keeps the outcomes of $A_k$ that no earlier event has already claimed.

> [!proof]- Proof of the union bound
> We need three facts about the $E_k$.
> - **They are pairwise disjoint.** Take $i<j$. Then $E_i\subseteq A_i$, while $E_j\subseteq A_i^c$ because $E_j$ removes everything in $A_1\cup\cdots\cup A_{j-1}$, which includes $A_i$. So $E_i\cap E_j=\emptyset$.
> - **They have the same union.** Each $E_k\subseteq A_k$, so $\bigcup_kE_k\subseteq\bigcup_kA_k$. Conversely, if $\omega\in\bigcup_kA_k$, let $k$ be the smallest index with $\omega\in A_k$. Then $\omega$ is in no earlier $A_i$, so $\omega\in E_k$.
> - **Each is smaller:** $E_k\subseteq A_k$, so $\mathbb{P}(E_k)\le\mathbb{P}(A_k)$ by monotonicity.
>
> Putting these together, with σ-additivity in the middle step:
>
> $$
> \mathbb{P}\Big(\bigcup_kA_k\Big)=\mathbb{P}\Big(\bigcup_kE_k\Big)=\sum_k\mathbb{P}(E_k)\le\sum_k\mathbb{P}(A_k).
> $$

The union bound is crude, since it ignores all overlaps, but it needs no assumptions at all. That makes it the standard way to say "with high probability, **none** of many bad events happens": bound each one, then add. The [[Multi-Armed Bandits#UCB achieves logarithmic regret|UCB regret proof]] and generalization bounds both work this way. We use it again [[#Concentration and the union bound together|below]].

### Continuity of probability

**7. Continuity.** If $E_1\subseteq E_2\subseteq\cdots$ is an increasing sequence of events with union $E=\bigcup_kE_k$ (written $E_k\uparrow E$), then

$$
\lim_{k\to\infty}\mathbb{P}(E_k)=\mathbb{P}(E).
$$

Any countable union is the limit of finite ones in this way: $E_k=\bigcup_{i=1}^kA_i$ increases to $\bigcup_{i=1}^{\infty}A_i$ (Figure 2a). Continuity lets us compute the probability of an infinite union as a limit of probabilities of finite ones.

> [!proof]- Proof
> Peel off the new part at each step: $F_1=E_1$ and $F_k=E_k\setminus E_{k-1}$. The $F_k$ are disjoint, $E_n=\bigcup_{k\le n}F_k$ and $E=\bigcup_kF_k$. By σ-additivity,
>
> $$
> \mathbb{P}(E)=\sum_{k=1}^{\infty}\mathbb{P}(F_k)=\lim_{n\to\infty}\sum_{k=1}^{n}\mathbb{P}(F_k)=\lim_{n\to\infty}\mathbb{P}(E_n).
> $$
>
> Taking complements gives the decreasing version: $E_k\downarrow E$ implies $\mathbb{P}(E_k)\to\mathbb{P}(E)$.

![[Probability and Linear Algebra/images/probability-basics-unions.svg]]
*Figure 2. (a) Continuity: the finite unions $E_k=A_1\cup\cdots\cup A_k$ grow towards the countable union, and their probabilities converge to its probability. (b) The disjointification behind the union bound: $E_1=A_1$, $E_2$ is the part of $A_2$ outside $A_1$, and $E_3$ is the part of $A_3$ outside both. Together they cover $A_1\cup A_2\cup A_3$ exactly once.*

## Conditional probability and independence

### Conditioning on evidence

Suppose we learn that $B$ happened. Among the $n_B$ runs in which $B$ occurred, $A$ also occurred in $n_{A\cap B}$ of them, so the relevant frequency is $n_{A\cap B}/n_B=(n_{A\cap B}/n)/(n_B/n)$. This motivates the definition, for $\mathbb{P}(B)>0$:

$$
\mathbb{P}(A\mid B)=\frac{\mathbb{P}(A\cap B)}{\mathbb{P}(B)}.
$$

Conditioning shrinks the sample space to $B$ and renormalizes. $\mathbb{P}(\cdot\mid B)$ satisfies all three axioms, so it is itself a probability. Three rules follow immediately:
- **Chain rule:** $\mathbb{P}(A\cap B)=\mathbb{P}(A\mid B)\,\mathbb{P}(B)$, and more generally $\mathbb{P}(A_1\cap\cdots\cap A_n)=\prod_{k=1}^{n}\mathbb{P}(A_k\mid A_1\cap\cdots\cap A_{k-1})$. The trajectory probability in [[Policy Gradients#The objective and its gradient|Policy Gradients]] is exactly this product.
- **Law of total probability:** if $B_1,\ldots,B_K$ partition $\Omega$, then $\mathbb{P}(A)=\sum_k\mathbb{P}(A\mid B_k)\,\mathbb{P}(B_k)$.
- **Bayes' rule:** combining the two,

$$
\mathbb{P}(B_k\mid A)=\frac{\mathbb{P}(A\mid B_k)\,\mathbb{P}(B_k)}{\sum_j\mathbb{P}(A\mid B_j)\,\mathbb{P}(B_j)}.
\tag{2}
$$

Bayes' rule turns the **likelihood** $\mathbb{P}(A\mid B_k)$ and the **prior** $\mathbb{P}(B_k)$ into the **posterior** $\mathbb{P}(B_k\mid A)$. [[Discriminant Analysis]] classifies with exactly (2), the classes playing the role of the $B_k$.

> [!example] Example: a rare condition and a good test
> A condition affects 1% of a population. A test detects it 95% of the time when present, and gives a false positive 10% of the time when absent. What is the probability of having the condition given a positive test?
>
> $$
> \mathbb{P}(D\mid+)=\frac{0.95\times0.01}{0.95\times0.01+0.10\times0.99}=\frac{0.0095}{0.1085}\approx0.088.
> $$
>
> Only about 9%. The test is good, but the condition is rare: the 10% false positives from the 99% healthy people outnumber the true positives about ten to one. Ignoring the prior is the classic **base-rate fallacy**.

### Independence

Events $A$ and $B$ are **independent** if

$$
\mathbb{P}(A\cap B)=\mathbb{P}(A)\,\mathbb{P}(B),
$$

equivalently $\mathbb{P}(A\mid B)=\mathbb{P}(A)$ when $\mathbb{P}(B)>0$: learning $B$ tells us nothing about $A$. For more than two events, **mutual** independence requires the product rule for **every** sub-collection. Pairwise independence is weaker. Toss two fair coins and let $A$ = "first is heads", $B$ = "second is heads", $C$ = "both tosses agree". Any two of these are independent, but all three are not: $A$ and $B$ together determine $C$.

$A$ and $B$ are **conditionally independent** given $C$ if $\mathbb{P}(A\cap B\mid C)=\mathbb{P}(A\mid C)\,\mathbb{P}(B\mid C)$. Neither kind of independence implies the other. The Markov property in [[Introduction to Reinforcement Learning|RL]] is a conditional independence: the future is independent of the past given the present. Naive Bayes ([[Discriminant Analysis]]) assumes features are conditionally independent given the class.

## Random variables

Outcomes like "$011$" are awkward to compute with. A **random variable** attaches a number to each outcome: a function $X:\Omega\to\mathbb{R}$. To talk about $\mathbb{P}(X\le x)$, the set $\{\omega:X(\omega)\le x\}$ must be an event, so we require $X$ to be **measurable**: $\{X\le x\}\in\mathcal{F}$ for every $x$. For three coin tosses, $X(\omega)=$ number of heads is a random variable, and its level sets $\{X=k\}$ are the four events of Figure 1.

### Distributions

The **cumulative distribution function** (CDF) $F_X(x)=\mathbb{P}(X\le x)$ describes $X$ completely. It is non-decreasing, right-continuous, tends to 0 as $x\to-\infty$ and to 1 as $x\to+\infty$. The two limits are applications of continuity of probability.

- A **discrete** random variable takes countably many values. It is described by its **probability mass function** (PMF) $p_X(x)=\mathbb{P}(X=x)$, with $\sum_xp_X(x)=1$. For a fair coin tossed three times, $p_X(0),\ldots,p_X(3)=\tfrac18,\tfrac38,\tfrac38,\tfrac18$.
- A **continuous** random variable has a **probability density function** (PDF) $f_X$ with $\mathbb{P}(a<X\le b)=\int_a^bf_X(x)\,dx$, so $f_X=F_X'$. A density is not a probability: $f_X(x)$ can exceed 1, and $\mathbb{P}(X=x)=0$ for every single point. Only integrals of $f_X$ are probabilities.

### Several random variables

For two random variables, the **joint** distribution $p_{X,Y}(x,y)$ (a joint PMF or joint density) determines everything else:
- **marginal:** $p_X(x)=\sum_yp_{X,Y}(x,y)$, or $\int p_{X,Y}(x,y)\,dy$;
- **conditional:** $p_{Y\mid X}(y\mid x)=p_{X,Y}(x,y)/p_X(x)$, for $p_X(x)>0$;
- **Bayes' rule for densities:** $p_{X\mid Y}(x\mid y)=p_{Y\mid X}(y\mid x)\,p_X(x)/p_Y(y)$.

$X$ and $Y$ are **independent** if $p_{X,Y}(x,y)=p_X(x)\,p_Y(y)$ for all $x,y$. A dataset $\mathcal{D}=\{x_i\}_{i=1}^N$ is usually modelled as **i.i.d.**: independent and identically distributed draws. This is what lets the likelihood factorize into a product, $p(\mathcal{D}\mid\theta)=\prod_ip(x_i\mid\theta)$, in every maximum-likelihood derivation of the series.

### Transforming a random variable

If $Y=g(X)$ for a strictly monotone, differentiable $g$, the density of $Y$ follows from conserving probability, $f_Y(y)\,|dy|=f_X(x)\,|dx|$:

$$
f_Y(y)=f_X\big(g^{-1}(y)\big)\,\Big|\frac{d}{dy}g^{-1}(y)\Big|.
$$

In $D$ dimensions the derivative becomes the absolute determinant of the Jacobian of $g^{-1}$. This **change-of-variables formula** is the basis of normalizing flows (a planned lecture).

## Expectation and variance

### Expectation

The **expectation** (mean) of $X$ is its probability-weighted average:

$$
\mathbb{E}[X]=\sum_xx\,p_X(x)\qquad\text{or}\qquad\mathbb{E}[X]=\int x\,f_X(x)\,dx,
$$

provided the sum or integral converges absolutely. For a function of $X$, we don't need the distribution of $g(X)$: $\mathbb{E}[g(X)]=\int g(x)f_X(x)\,dx$. This is the "law of the unconscious statistician", discussed in an [[Variational Autoencoders#Gradients of an expectation|aside in the VAE lecture]]. Two properties are used constantly:
- **Linearity:** $\mathbb{E}[aX+bY+c]=a\,\mathbb{E}[X]+b\,\mathbb{E}[Y]+c$, **always**, whether or not $X$ and $Y$ are independent.
- **Products:** if $X$ and $Y$ are independent, $\mathbb{E}[XY]=\mathbb{E}[X]\,\mathbb{E}[Y]$. Without independence this generally fails.

The expectation of an indicator is a probability: $\mathbb{E}[\mathbb{1}\{A\}]=\mathbb{P}(A)$. This simple identity connects everything about expectations to everything about probabilities.

### Variance and covariance

The **variance** measures spread around the mean $\mu=\mathbb{E}[X]$:

$$
\operatorname{Var}(X)=\mathbb{E}\big[(X-\mu)^2\big]=\mathbb{E}[X^2]-\mu^2,
\qquad
\operatorname{Var}(aX+b)=a^2\operatorname{Var}(X).
$$

Its square root is the **standard deviation**. For two variables, the **covariance** $\operatorname{Cov}(X,Y)=\mathbb{E}\big[(X-\mu_X)(Y-\mu_Y)\big]$ measures how they move together, and

$$
\operatorname{Var}(X+Y)=\operatorname{Var}(X)+\operatorname{Var}(Y)+2\operatorname{Cov}(X,Y).
$$

The **correlation** $\rho=\operatorname{Cov}(X,Y)/\sqrt{\operatorname{Var}(X)\operatorname{Var}(Y)}$ always lies in $[-1,1]$, by the Cauchy–Schwarz inequality.

> [!warning] Uncorrelated does not mean independent
> Independence implies zero covariance, but not conversely. Let $X$ be uniform on $[-1,1]$ and $Y=X^2$. Then $Y$ is a deterministic function of $X$, yet $\operatorname{Cov}(X,Y)=\mathbb{E}[X^3]-\mathbb{E}[X]\,\mathbb{E}[X^2]=0$ by symmetry. Covariance only detects **linear** dependence.

For a random vector $\mathbf{x}\in\mathbb{R}^D$ with mean $\boldsymbol\mu$, the **covariance matrix** is $\boldsymbol\Sigma=\mathbb{E}\big[(\mathbf{x}-\boldsymbol\mu)(\mathbf{x}-\boldsymbol\mu)^\top\big]$. It is symmetric and positive semi-definite, because $\mathbf{v}^\top\boldsymbol\Sigma\mathbf{v}=\operatorname{Var}(\mathbf{v}^\top\mathbf{x})\ge0$ for every direction $\mathbf{v}$. [[Matrix Factorization and Dimensionality Reduction#Principal component analysis (PCA)|PCA]] finds the directions where this variance is largest.

### Conditional expectation

$\mathbb{E}[Y\mid X=x]=\sum_yy\,p_{Y\mid X}(y\mid x)$ is the mean of $Y$ once $X$ is known. Viewed as a function of the random $X$, the conditional expectation $\mathbb{E}[Y\mid X]$ is itself a random variable, and it obeys:
- the **tower property** (law of total expectation): $\mathbb{E}\big[\mathbb{E}[Y\mid X]\big]=\mathbb{E}[Y]$;
- the **law of total variance**: $\operatorname{Var}(Y)=\mathbb{E}\big[\operatorname{Var}(Y\mid X)\big]+\operatorname{Var}\big(\mathbb{E}[Y\mid X]\big)$, which splits the variance into "unexplained" and "explained" parts;
- **optimal prediction:** among all functions $h$, $\mathbb{E}\big[(Y-h(X))^2\big]$ is minimized by $h(X)=\mathbb{E}[Y\mid X]$. Regression with squared loss estimates a conditional mean.

These are everywhere in the series. A [[Introduction to Reinforcement Learning|value function]] $V^\pi(s)$ is a conditional expectation of the return given the state. The E-step of [[Clustering and Expectation Maximization|EM]] computes an expectation under a conditional distribution. Tweedie's formula in [[Score-Based Models#Tweedie's formula: the score is the optimal denoiser|Score-Based Models]] is a formula for $\mathbb{E}[\mathbf{x}\mid\tilde{\mathbf{x}}]$.

## Common distributions

| distribution | values | PMF / PDF | mean | variance |
|---|---|---|---|---|
| Bernoulli$(p)$ | $\{0,1\}$ | $p^x(1-p)^{1-x}$ | $p$ | $p(1-p)$ |
| Binomial$(n,p)$ | $\{0,\ldots,n\}$ | $\binom{n}{k}p^k(1-p)^{n-k}$ | $np$ | $np(1-p)$ |
| Categorical$(\boldsymbol\pi)$ | $\{1,\ldots,K\}$ | $\pi_k$ | — | — |
| Geometric$(p)$ | $\{1,2,\ldots\}$ | $(1-p)^{k-1}p$ | $1/p$ | $(1-p)/p^2$ |
| Poisson$(\lambda)$ | $\{0,1,\ldots\}$ | $\lambda^ke^{-\lambda}/k!$ | $\lambda$ | $\lambda$ |
| Uniform$(a,b)$ | $[a,b]$ | $1/(b-a)$ | $(a+b)/2$ | $(b-a)^2/12$ |
| Exponential$(\lambda)$ | $[0,\infty)$ | $\lambda e^{-\lambda x}$ | $1/\lambda$ | $1/\lambda^2$ |
| Gaussian $\mathcal{N}(\mu,\sigma^2)$ | $\mathbb{R}$ | $\frac{1}{\sqrt{2\pi\sigma^2}}e^{-(x-\mu)^2/2\sigma^2}$ | $\mu$ | $\sigma^2$ |

Where they appear:
- **Bernoulli** models binary labels in logistic regression; **Binomial** counts successes in $n$ independent trials (the number of heads in Figure 1 is Binomial$(3,\tfrac12)$).
- **Categorical** is the output of a softmax, and the component choice in a mixture model; **Geometric** counts trials until the first success.
- **Poisson** models counts of rare events; **Uniform** expresses "no information" and is used for random initialization; **Exponential** models waiting times and is memoryless.
- **Gaussian** is used for noise models, priors and latent variables throughout the series.

![[Probability and Linear Algebra/images/probability-basics-distributions.svg]]
*Figure 3. (a) Binomial PMFs for 10 trials. (b) Poisson PMFs. (c) Exponential densities, which decay at rate $\lambda$. (d) Gaussian densities. A narrow Gaussian has a peak above 1, a reminder that densities are not probabilities.*

### The multivariate Gaussian

The most important distribution in the series is the $D$-dimensional Gaussian $\mathcal{N}(\boldsymbol\mu,\boldsymbol\Sigma)$:

$$
p(\mathbf{x})=\frac{1}{(2\pi)^{D/2}|\boldsymbol\Sigma|^{1/2}}\exp\Big(-\frac12(\mathbf{x}-\boldsymbol\mu)^\top\boldsymbol\Sigma^{-1}(\mathbf{x}-\boldsymbol\mu)\Big).
$$

Its contours are ellipsoids, with axes along the eigenvectors of $\boldsymbol\Sigma$ and lengths proportional to the square roots of its eigenvalues (Figure 4). It is closed under almost every operation we need:[^bishop2006]
- **affine maps:** if $\mathbf{x}\sim\mathcal{N}(\boldsymbol\mu,\boldsymbol\Sigma)$, then $\mathbf{A}\mathbf{x}+\mathbf{b}\sim\mathcal{N}(\mathbf{A}\boldsymbol\mu+\mathbf{b},\,\mathbf{A}\boldsymbol\Sigma\mathbf{A}^\top)$. This is the reparameterization $\mathbf{x}=\boldsymbol\mu+\mathbf{L}\boldsymbol\epsilon$ of the [[Variational Autoencoders|VAE]], with $\mathbf{L}\mathbf{L}^\top=\boldsymbol\Sigma$;
- **sums:** the sum of independent Gaussians is Gaussian, with means and covariances adding. Repeatedly adding Gaussian noise is the forward process of diffusion models;
- **marginals and conditionals:** split $\mathbf{x}=(\mathbf{x}_1,\mathbf{x}_2)$. Then $\mathbf{x}_1$ is Gaussian, and so is $\mathbf{x}_1$ given $\mathbf{x}_2$, with

$$
\boldsymbol\mu_{1\mid2}=\boldsymbol\mu_1+\boldsymbol\Sigma_{12}\boldsymbol\Sigma_{22}^{-1}(\mathbf{x}_2-\boldsymbol\mu_2),
\qquad
\boldsymbol\Sigma_{1\mid2}=\boldsymbol\Sigma_{11}-\boldsymbol\Sigma_{12}\boldsymbol\Sigma_{22}^{-1}\boldsymbol\Sigma_{21}.
$$

The conditional mean is linear in $\mathbf{x}_2$, and the conditional covariance does not depend on $\mathbf{x}_2$ at all.

![[Probability and Linear Algebra/images/probability-basics-gaussian-2d.svg]]
*Figure 4. Density contours and 300 samples of three bivariate Gaussians with mean zero: isotropic (circles), diagonal covariance (axis-aligned ellipses), and correlation $\rho=0.8$ (tilted ellipses). Only the third has dependent coordinates.*

## Limit theorems: why frequencies work

We motivated probability with relative frequencies, but never showed that $n_A/n$ actually approaches $\mathbb{P}(A)$. The tools below prove it, and say how fast.

### Markov and Chebyshev

**Markov's inequality.** If $X\ge0$ and $a>0$, then

$$
\mathbb{P}(X\ge a)\le\frac{\mathbb{E}[X]}{a}.
$$

*Proof:* $X\ge a\,\mathbb{1}\{X\ge a\}$ pointwise, because the right side is either $0$ or $a\le X$. Taking expectations gives $\mathbb{E}[X]\ge a\,\mathbb{P}(X\ge a)$.

**Chebyshev's inequality.** If $X$ has mean $\mu$ and variance $\sigma^2$, then for $k>0$,

$$
\mathbb{P}\big(|X-\mu|\ge k\big)\le\frac{\sigma^2}{k^2}.
$$

*Proof:* apply Markov to the non-negative variable $(X-\mu)^2$ with $a=k^2$.

### The law of large numbers

Let $X_1,X_2,\ldots$ be i.i.d. with mean $\mu$ and variance $\sigma^2$, and let $\bar X_n=\frac1n\sum_{i=1}^nX_i$. By linearity $\mathbb{E}[\bar X_n]=\mu$. By independence the variances add, so $\operatorname{Var}(\bar X_n)=\sigma^2/n$. Chebyshev then gives the **weak law of large numbers**:

$$
\mathbb{P}\big(|\bar X_n-\mu|\ge\varepsilon\big)\le\frac{\sigma^2}{n\varepsilon^2}\xrightarrow[n\to\infty]{}0\qquad\text{for every }\varepsilon>0.
\tag{3}
$$

Now take $X_i=\mathbb{1}\{A\text{ occurs in run }i\}$. Then $\mu=\mathbb{P}(A)$ and $\bar X_n=n_A/n$, so (3) says that **the relative frequency converges to the probability**. The frequency model we started from is a theorem of the axioms (Figure 5). The **strong** law says more: $\bar X_n\to\mu$ with probability 1.

![[Probability and Linear Algebra/images/probability-basics-lln.svg]]
*Figure 5. Five independent sequences of runs with $\mathbb{P}(A)=0.3$. Each running relative frequency $n_A/n$ wanders early on, then settles at 0.3. The shaded band is the 95% Hoeffding interval $0.3\pm\sqrt{\ln(2/0.05)/(2n)}$, which shrinks like $1/\sqrt n$.*

### Hoeffding: exponential concentration

Chebyshev uses only the variance, so its bound decays only like $1/n$. For **bounded** variables, Hoeffding's inequality is far sharper.[^hoeffding1963] If the $X_i$ are i.i.d. in $[0,1]$,

$$
\mathbb{P}\big(|\bar X_n-\mu|\ge\varepsilon\big)\le2e^{-2n\varepsilon^2}.
\tag{4}
$$

The tail now decays **exponentially** in $n$. Compare the sample sizes needed to estimate a probability to within $\varepsilon=0.05$ with confidence $1-\delta=99\%$:
- Chebyshev, using $\sigma^2\le\frac14$: $n\ge\sigma^2/(\varepsilon^2\delta)=10\,000$;
- Hoeffding: $n\ge\ln(2/\delta)/(2\varepsilon^2)\approx1\,060$.

The dependence on the confidence level drops from $1/\delta$ to $\ln(1/\delta)$. The proof, via sub-Gaussian tails and the Chernoff bound, is in the [[Multi-Armed Bandits#Aside: concentration inequalities|concentration aside of the Bandits lecture]].

### Concentration and the union bound together

Suppose we estimate $M$ different probabilities from the same $n$ runs, and want **all** of them to be accurate at once. Each estimate fails with probability at most $2e^{-2n\varepsilon^2}$ by (4), so by the union bound (1),

$$
\mathbb{P}\big(\text{some estimate is off by }\ge\varepsilon\big)\le2M\,e^{-2n\varepsilon^2}.
$$

Setting this to $\delta$ shows that $n\ge\ln(2M/\delta)/(2\varepsilon^2)$ runs suffice. The cost of $M$ simultaneous guarantees is only **logarithmic** in $M$. Exponential concentration plus a union bound is the engine of learning theory (a finite hypothesis class is $M$ quantities estimated at once), and of the confidence bonuses in [[Multi-Armed Bandits|UCB]].

### The central limit theorem

The law of large numbers says $\bar X_n\to\mu$. The **central limit theorem** describes the fluctuations around $\mu$. For i.i.d. $X_i$ with finite variance $\sigma^2>0$,

$$
\frac{\sqrt n\,(\bar X_n-\mu)}{\sigma}\xrightarrow{\;d\;}\mathcal{N}(0,1),
$$

meaning the CDF of the left side converges to the standard Gaussian CDF. This holds **whatever** the distribution of the $X_i$ (Figure 6). This is why Gaussians are everywhere: any quantity built from many small independent contributions is approximately Gaussian. It also gives the familiar error bar: $\bar X_n\pm1.96\,\sigma/\sqrt n$ is an approximate 95% confidence interval.

![[Probability and Linear Algebra/images/probability-basics-clt.svg]]
*Figure 6. The central limit theorem for the heavily skewed Exponential(1) distribution. Histograms of the standardized mean of $n$ samples (100 000 repetitions each) against the standard Gaussian density. At $n=1$ the histogram is the exponential itself; by $n=30$ it is close to Gaussian, with a slight remaining skew.*

## Key takeaways

- A probability space $(\Omega,\mathcal{F},\mathbb{P})$ is a set of outcomes, a σ-algebra of events closed under complements and countable unions, and a normalized, non-negative, σ-additive set function. The axioms do not depend on the interpretation of probability.
- From the axioms, in order: $\mathbb{P}(\emptyset)=0$, finite additivity, complements, monotonicity, inclusion–exclusion, the union bound (by disjointification), and continuity along increasing sequences.
- Conditioning restricts and renormalizes. The chain rule, the law of total probability and Bayes' rule follow, and the base-rate example shows why the prior matters.
- Random variables carry distributions (CDF, PMF, PDF), expectations (always linear) and variances (additive for independent variables). Uncorrelated does not imply independent. $\mathbb{E}[Y\mid X]$ is the best squared-error predictor.
- The multivariate Gaussian is closed under affine maps, sums, marginals and conditionals.
- Markov ⇒ Chebyshev ⇒ the law of large numbers, which justifies the frequency model. Hoeffding gives exponential concentration, the union bound extends it to many quantities at logarithmic cost, and the CLT explains why averages look Gaussian.

## Where this shows up next

- [[Linear Models]]: maximum likelihood with Gaussian and Bernoulli models, MAP estimation with priors, and cross-entropy and KL divergence.
- [[Discriminant Analysis]]: Bayes' rule with Gaussian class-conditionals, and naive Bayes's conditional independence.
- [[Clustering and Expectation Maximization]]: mixtures, latent variables, and expectations under conditional distributions.
- [[Multi-Armed Bandits]]: Hoeffding plus the union bound in the UCB regret proof.
- [[Introduction to Reinforcement Learning]]: the Markov property and value functions as conditional expectations.
- [[Variational Autoencoders]] and [[Score-Based Models]]: Gaussian reparameterization, LOTUS, and Tweedie's formula for a conditional mean.

<!-- References: footnote definitions below render as the numbered "References" list at the bottom of the page -->

[^jaynes2003]: Jaynes, E. T. (2003). *Probability Theory: The Logic of Science*. Cambridge University Press. [doi:10.1017/CBO9780511790423](https://doi.org/10.1017/CBO9780511790423)
[^kolmogorov1933]: Kolmogorov, A. N. (1933). *Grundbegriffe der Wahrscheinlichkeitsrechnung*. Springer. English translation: *Foundations of the Theory of Probability* (1950). [doi:10.1007/978-3-642-49888-6](https://doi.org/10.1007/978-3-642-49888-6)
[^bishop2006]: Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*, Section 2.3 (The Gaussian distribution). Springer. [book page](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/)
[^hoeffding1963]: Hoeffding, W. (1963). *Probability Inequalities for Sums of Bounded Random Variables*. Journal of the American Statistical Association, 58(301), 13–30. [doi:10.1080/01621459.1963.10500830](https://doi.org/10.1080/01621459.1963.10500830)
[^bertsekas2008]: Bertsekas, D. P., & Tsitsiklis, J. N. (2008). *Introduction to Probability* (2nd ed.). Athena Scientific. Lectures: [MIT OCW 6.041](https://ocw.mit.edu/courses/6-041-probabilistic-systems-analysis-and-applied-probability-fall-2010/)
[^blitzstein2019]: Blitzstein, J. K., & Hwang, J. (2019). *Introduction to Probability* (2nd ed.). CRC Press. [Free online edition](https://probabilitybook.net)
[^wasserman2004]: Wasserman, L. (2004). *All of Statistics: A Concise Course in Statistical Inference*. Springer. [doi:10.1007/978-0-387-21736-9](https://doi.org/10.1007/978-0-387-21736-9)
[^murphy2022]: Murphy, K. P. (2022). *Probabilistic Machine Learning: An Introduction*, Chapters 2–3. MIT Press. [book page](https://probml.github.io/pml-book/book1.html)
