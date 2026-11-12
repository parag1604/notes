---
title: Sample Statistics
tags:
  - probability
  - foundations
date: 2026-11-12
draft: false
prev: "[[Joint Distributions]]"
next: "[[Markov Chains]]"
---

> **Prerequisites:** [[Random Variables]] (expectation, variance, Markov and Chebyshev) and [[Joint Distributions]] (independence, variance of a sum, convolution).
>
> **Series:** this is part 4 of 5. [[Probability Basics]] → [[Random Variables]] → [[Joint Distributions]] → **Sample Statistics** → [[Markov Chains]].
>
> **TL;DR:**
> - The sample mean of $n$ i.i.d. draws has the right mean and a spread that shrinks like $\sigma/\sqrt n$. Everything in this lecture sharpens that one fact.
> - Moment-generating and characteristic functions turn sums of independent variables into products, which makes sums easy to analyse.
> - Jensen's inequality, Chernoff bounds and Hoeffding's inequality: tails of averages decay **exponentially**, and a union bound extends this to many averages at once.
> - Random sequences can converge in four senses: in distribution, in probability, in mean square, and almost surely. They are not equivalent.
> - The **law of large numbers** says the sample mean converges to the mean (weakly: in probability; strongly: almost surely). This justifies the frequency model and Monte Carlo integration. The **central limit theorem** says the fluctuations are Gaussian.

## Motivation

Almost everything we compute from data is an average. A training loss is the average loss over the examples, a Monte Carlo estimate is the average of simulated values, an opinion poll is the average of yes/no answers, and the value estimate in [[Multi-Armed Bandits]] is the average reward of an arm. An average is a random variable: run the experiment again and you get a different one. This lecture answers three questions about it:
1. **Where does it go?** As the sample grows, the average converges to the expectation: the law of large numbers.
2. **How fast?** Tail bounds say how unlikely a large error is, for every finite sample size.
3. **What do the errors look like?** The central limit theorem: approximately Gaussian, whatever the data.

Along the way we need a precise meaning for "a random sequence converges", which turns out to have several. Chan's Chapter 6 is the intuitive source for this lecture, and Kumar's first chapter is the rigorous one.[^chan2026]<sup>,</sup>[^kumar2012]

## The sample mean and the sample variance

Let $X_1,\dots,X_n$ be i.i.d. with mean $\mu$ and variance $\sigma^2$. The **sample mean** is $\bar X_n=\frac1n\sum_{i=1}^nX_i$. By linearity and by the [[Joint Distributions#Variance of a sum|variance of a sum]] of independent variables,

$$
\mathbb{E}[\bar X_n]=\mu,\qquad\operatorname{Var}(\bar X_n)=\frac{\sigma^2}{n}.
$$

The sample mean is **unbiased**: on average it is right. And its standard deviation, the **standard error** $\sigma/\sqrt n$, shrinks as the sample grows. The $\sqrt n$ is the law of diminishing returns of data: to halve the error, you need four times as many samples.

The natural estimate of $\sigma^2$ is the **sample variance**

$$
S_n^2=\frac{1}{n-1}\sum_{i=1}^{n}(X_i-\bar X_n)^2.
$$

The divisor $n-1$ rather than $n$ (**Bessel's correction**) makes it unbiased. The deviations are measured from $\bar X_n$, which is fitted to the same data and is therefore closer to the points than the true $\mu$ is. Expanding the square gives $\sum_i(X_i-\bar X_n)^2=\sum_i(X_i-\mu)^2-n(\bar X_n-\mu)^2$, whose expectation is $n\sigma^2-\sigma^2=(n-1)\sigma^2$. Fitting the mean uses up one "degree of freedom".

## Transforms: moment-generating and characteristic functions

### Moment-generating functions

The distribution of a sum of independent variables is a [[Joint Distributions#Sums of independent random variables|convolution]], which is painful to compute for many terms. In signal processing, the cure for convolutions is the Fourier transform, which turns them into products. Probability has the same trick. The **moment-generating function** (MGF) of $X$ is

$$
M_X(s)=\mathbb{E}\big[e^{sX}\big],
$$

for the real $s$ where the expectation is finite. Three properties make it useful:
- **It generates moments.** Expanding $e^{sX}=1+sX+\frac{s^2X^2}{2!}+\cdots$ gives $M_X(s)=\sum_k\frac{s^k}{k!}\mathbb{E}[X^k]$, so the $k$-th derivative at $0$ is the $k$-th moment: $M_X^{(k)}(0)=\mathbb{E}[X^k]$. Each derivative "pulls down" one more factor of $X$.
- **Sums become products.** If $X$ and $Y$ are independent, $M_{X+Y}(s)=\mathbb{E}[e^{sX}e^{sY}]=M_X(s)\,M_Y(s)$. A sum of $n$ i.i.d. copies has MGF $M_X(s)^n$.
- **It identifies the distribution.** If two MGFs agree on an interval around 0, the distributions are the same.

| distribution | MGF $M_X(s)$ |
|---|---|
| Bernoulli$(p)$ | $1-p+pe^s$ |
| Binomial$(n,p)$ | $(1-p+pe^s)^n$ |
| Poisson$(\lambda)$ | $\exp\big(\lambda(e^s-1)\big)$ |
| Exponential$(\lambda)$ | $\lambda/(\lambda-s)$, for $s<\lambda$ |
| Gaussian $\mathcal{N}(\mu,\sigma^2)$ | $\exp\big(\mu s+\tfrac12\sigma^2s^2\big)$ |

With the table, the closure results of [[Joint Distributions#Sums of independent random variables|Joint Distributions]] become one-liners. The binomial's MGF is the Bernoulli's raised to the $n$-th power, because a binomial is a sum of $n$ independent Bernoullis. Multiplying two Poisson MGFs gives $\exp\big((\lambda_1+\lambda_2)(e^s-1)\big)$, a Poisson MGF. Multiplying two Gaussian MGFs adds the means and variances in the exponent. And differentiating, $M'(0)=\lambda$ for the Poisson and $M''(0)-M'(0)^2=\sigma^2$ for the Gaussian.

### Characteristic functions

The MGF has a weakness: $\mathbb{E}[e^{sX}]$ can be infinite. For the exponential distribution it is finite only for $s<\lambda$, and for heavy-tailed distributions, like the Cauchy, it is infinite for every $s\ne0$. The fix is to use an imaginary argument. The **characteristic function**

$$
\varphi_X(t)=\mathbb{E}\big[e^{itX}\big]=\mathbb{E}[\cos tX]+i\,\mathbb{E}[\sin tX]
$$

always exists, because $|e^{itX}|=1$. For a continuous variable it is the Fourier transform of the density. It has the same three properties: moments as derivatives (when they exist), products for independent sums, and uniqueness. It also has one more:
- **Lévy's continuity theorem:** $X_n$ converges in distribution to $X$ (defined [[#Four ways a sequence can converge|below]]) exactly when $\varphi_{X_n}(t)\to\varphi_X(t)$ for every $t$.

This is the standard route to proving the central limit theorem.[^durrett2019]

> [!example] Example: averages of Cauchy variables never improve
> The standard Cauchy density $\frac{1}{\pi(1+x^2)}$ has no mean ([[Random Variables#Mean, median and mode|Random Variables]]) and no MGF, but its characteristic function is simply $e^{-|t|}$. The average of $n$ i.i.d. copies has characteristic function
>
> $$
> \varphi_{\bar X_n}(t)=\varphi_X(t/n)^n=\big(e^{-|t|/n}\big)^n=e^{-|t|}.
> $$
>
> By uniqueness, $\bar X_n$ is standard Cauchy again, **for every $n$**. Averaging a million Cauchy samples is no more accurate than looking at one (Figure 5b). The law of large numbers needs a finite mean, and this is what happens without one.

## Inequalities

### Jensen's inequality

If $g$ is convex (curving upward, like $x^2$ or $e^x$), then

$$
g\big(\mathbb{E}[X]\big)\le\mathbb{E}\big[g(X)\big].
\tag{1}
$$

The picture is a two-point distribution (Figure 1). If $X$ is $a$ or $b$ with equal probability, $\mathbb{E}[g(X)]$ is the midpoint of the chord between $(a,g(a))$ and $(b,g(b))$, while $g(\mathbb{E}[X])$ is the curve at the midpoint. For a convex function the chord lies above the curve. The proof in general uses a tangent line instead of a chord. A convex $g$ lies above its tangent at $\mu=\mathbb{E}[X]$: $g(x)\ge g(\mu)+g'(\mu)(x-\mu)$ for all $x$. Taking expectations kills the last term.

![[Probability and Linear Algebra/images/sample-statistics-jensen.svg]]
*Figure 1. Jensen's inequality for a two-point distribution. The average of $g$ (red, on the chord) lies above $g$ of the average (green, on the curve), because a convex curve bends below its chords.*

Jensen is behind many inequalities in the series:
- $g(x)=x^2$ gives $\mathbb{E}[X^2]\ge\mathbb{E}[X]^2$, that is, $\operatorname{Var}(X)\ge0$.
- $g=-\log$ is convex, so $\mathbb{E}[\log X]\le\log\mathbb{E}[X]$. This is the inequality that gives the evidence lower bound (ELBO) in [[Variational Autoencoders]] and [[Clustering and Expectation Maximization|EM]].
- The KL divergence is non-negative: $D_{\mathrm{KL}}(p\,\|\,q)=\mathbb{E}_p\big[-\log\frac{q}{p}\big]\ge-\log\mathbb{E}_p\big[\frac qp\big]=-\log1=0$.

### Chernoff bounds

[[Random Variables#Markov's and Chebyshev's inequalities|Markov's and Chebyshev's inequalities]] give tail bounds that decay only polynomially. The **Chernoff bound** gets exponential decay by applying Markov's inequality to $e^{sX}$ instead of $X$. For any $s>0$, the map $x\mapsto e^{sx}$ is increasing, so

$$
\mathbb{P}(X\ge a)=\mathbb{P}\big(e^{sX}\ge e^{sa}\big)\le\frac{\mathbb{E}[e^{sX}]}{e^{sa}}=e^{-sa}M_X(s).
$$

This holds for every $s>0$, so we can choose the best one:[^chernoff1952]

$$
\mathbb{P}(X\ge a)\le\exp\Big(-\sup_{s>0}\big[sa-\log M_X(s)\big]\Big).
\tag{2}
$$

The exponential is what makes this strong. It turns a small additive excess $a-\mu$ into a large multiplicative factor, and Markov's inequality is good at bounding large multiplicative excesses.

The real power shows for averages. If $\bar X_n$ is the mean of $n$ i.i.d. copies of $X$, then $M_{n\bar X_n}(s)=M_X(s)^n$, and (2) becomes

$$
\mathbb{P}(\bar X_n\ge\mu+\varepsilon)\le e^{-n\,I(\mu+\varepsilon)},\qquad I(a)=\sup_s\big[sa-\log M_X(s)\big].
$$

The tail decays **exponentially in $n$**, at a rate $I$ that depends on the distribution and on $\varepsilon$. For a coin with $\mathbb{P}(\text{heads})=p$, the rate works out to be the KL divergence between a coin with bias $p+\varepsilon$ and the true coin, $I=D_{\mathrm{KL}}\big(\text{Bern}(p+\varepsilon)\,\|\,\text{Bern}(p)\big)$. This is the start of **large deviations** theory: the probability of seeing a fair coin behave like a 60% coin in $n$ tosses is about $e^{-0.0201n}$.

### Hoeffding's inequality

The Chernoff bound needs the MGF. For **bounded** variables there is a version that needs only the range. If the $X_i$ are i.i.d. with values in $[0,1]$, **Hoeffding's inequality** says[^hoeffding1963]

$$
\mathbb{P}\big(|\bar X_n-\mu|\ge\varepsilon\big)\le2e^{-2n\varepsilon^2}.
\tag{3}
$$

The key step, **Hoeffding's lemma**, shows that a zero-mean variable confined to an interval of length 1 has $\mathbb{E}[e^{sX}]\le e^{s^2/8}$. Its MGF is no larger than that of a Gaussian with variance $1/4$, so its tails are "sub-Gaussian". Plugging this into (2) and optimizing over $s$ gives (3). The full proof is in the [[Multi-Armed Bandits#Aside: concentration inequalities|concentration aside of the Bandits lecture]].

Figure 2 compares the bounds for a fair coin. Chebyshev's bound decays like $1/n$; Hoeffding's and the optimized Chernoff bound decay exponentially and track the exact tail closely. In practical terms, to estimate a probability to within $\varepsilon=0.05$ with confidence $1-\delta=99\%$:
- Chebyshev, using $\sigma^2\le\frac14$, needs $n\ge\sigma^2/(\varepsilon^2\delta)=10\,000$;
- Hoeffding needs only $n\ge\ln(2/\delta)/(2\varepsilon^2)\approx1\,060$.

The dependence on the confidence level drops from $1/\delta$ to $\ln(1/\delta)$: very high confidence is cheap.

![[Probability and Linear Algebra/images/sample-statistics-tail-bounds.svg]]
*Figure 2. Bounds on the probability that the fraction of heads in $n$ fair tosses exceeds 0.6. Chebyshev's bound (orange) decays like $1/n$. Hoeffding's (green) and the optimized Chernoff bound (dashed) decay exponentially, at almost the same rate as the exact tail (red), which they bound from above.*

### Concentration and the union bound

Suppose we estimate $M$ different quantities from the same $n$ samples, and want **all** of them to be accurate at once. Each estimate fails with probability at most $2e^{-2n\varepsilon^2}$ by (3), so by the [[Probability Basics#The union bound|union bound]],

$$
\mathbb{P}\big(\text{some estimate is off by }\ge\varepsilon\big)\le2M\,e^{-2n\varepsilon^2}.
$$

Setting this to $\delta$ shows that $n\ge\ln(2M/\delta)/(2\varepsilon^2)$ samples suffice. The cost of $M$ simultaneous guarantees is only **logarithmic** in $M$. Exponential concentration plus a union bound is the engine of statistical learning theory (a finite set of $M$ hypotheses is $M$ training errors estimated at once), and of the confidence bonuses in [[Multi-Armed Bandits|UCB]].

## Four ways a sequence can converge

A sequence of numbers either converges or it does not. A sequence of **random variables** $X_1,X_2,\dots$ is a sequence of functions on $\Omega$, and "$X_n\to X$" can mean several different things. In decreasing order of how intuitive they are:

**In distribution** ($X_n\xrightarrow{d}X$): the CDFs converge, $F_{X_n}(x)\to F_X(x)$ at every $x$ where $F_X$ is continuous. Only the *histograms* get close; the variables themselves need not. If $X\sim\mathcal{N}(0,1)$ and $X_n=(-1)^nX$, every $X_n$ has exactly the distribution of $X$, so $X_n\xrightarrow{d}X$. Yet for odd $n$, $X_n=-X$ is far from $X$.

**In probability** ($X_n\xrightarrow{p}X$): for every tolerance $\varepsilon>0$,

$$
\mathbb{P}\big(|X_n-X|>\varepsilon\big)\to0.
$$

For large $n$, $X_n$ is unlikely to be far from $X$. But this is a statement about each $n$ separately. It does not say that a single run of the experiment eventually stays close.

**In mean square** ($X_n\xrightarrow{m.s.}X$): $\mathbb{E}\big[(X_n-X)^2\big]\to0$. The average squared error vanishes. This is the natural notion for estimators, where $\mathbb{E}[(\hat\theta_n-\theta)^2]$ is the mean squared error.

**Almost surely** ($X_n\xrightarrow{a.s.}X$): with probability 1, the sequence of numbers $X_n(\omega)$ converges to $X(\omega)$,

$$
\mathbb{P}\Big(\big\{\omega:\lim_{n\to\infty}X_n(\omega)=X(\omega)\big\}\Big)=1.
$$

Fix the outcome $\omega$, look at the whole sample path $X_1(\omega),X_2(\omega),\dots$, and ask whether it converges. Almost sure convergence says it does for all outcomes except a set of probability zero. It is the strongest notion, and it is what we mean when we say "a single long run gets it right".

The relations between them are:
- almost surely ⇒ in probability ⇒ in distribution;
- in mean square ⇒ in probability, by Chebyshev: $\mathbb{P}(|X_n-X|>\varepsilon)\le\mathbb{E}[(X_n-X)^2]/\varepsilon^2$;
- none of the reverse implications holds, and almost sure and mean-square convergence do not imply each other.

Two counterexamples show what can go wrong. Let $U$ be uniform on $[0,1]$.
- **In probability, but not almost surely: the "typewriter".** Number blocks of $[0,1]$ in sweeps: first all of $[0,1]$, then the two halves, then the four quarters, and so on. Let $X_n=1$ if $U$ falls in the $n$-th block, and 0 otherwise (Figure 3). The blocks shrink, so $\mathbb{P}(X_n=1)\to0$, and $X_n\to0$ in probability and in mean square. But every sweep covers all of $[0,1]$, so **every** path returns to 1 once per sweep, forever. No path converges.
- **Almost surely, but the means do not converge: the spike.** Let $X_n=n$ if $U\le1/n$, and 0 otherwise. For every $U>0$, $X_n=0$ once $n>1/U$, so $X_n\to0$ almost surely. But $\mathbb{E}[X_n]=n\cdot\frac1n=1$ for every $n$, and $\mathbb{E}[X_n^2]=n\to\infty$. A vanishingly rare but growing spike keeps the mean at 1. In general we cannot swap limits and expectations; theorems such as dominated convergence give conditions under which we can.[^kumar2012]

![[Probability and Linear Algebra/images/sample-statistics-typewriter.svg]]
*Figure 3. The typewriter sequence. (a) For each $n$ (horizontal), $X_n(\omega)=1$ for the outcomes $\omega$ in the blue block. Each vertical slice has a smaller block, so $\mathbb{P}(X_n=1)\to0$: convergence in probability. (b) But along any horizontal line, here $\omega=0.3$, the path keeps hitting 1 once per sweep: no almost sure convergence.*

### The Borel–Cantelli lemma

To prove almost sure convergence, we need to show that bad events stop happening. Let $A_n$ be "$X_n$ is more than $\varepsilon$ from its limit". The tool is:

> [!note] Borel–Cantelli lemma
> If $\sum_{n=1}^{\infty}\mathbb{P}(A_n)<\infty$, then with probability 1 only finitely many of the events $A_n$ occur.

*Proof:* the number of events that occur is $N=\sum_n\mathbb{1}_{A_n}$, and by linearity $\mathbb{E}[N]=\sum_n\mathbb{P}(A_n)<\infty$. A random variable with a finite mean is finite with probability 1.

If the expected number of bad events is finite, the bad events eventually stop. In the typewriter, $\mathbb{P}(A_n)$ is $2^{-j}$ in sweep $j$, and each sweep has $2^j$ blocks, so the sum is infinite and the lemma does not apply, consistent with Figure 3b. Conversely, if the events are **independent** and the sum is infinite, then infinitely many occur with probability 1 (the second Borel–Cantelli lemma).

## The law of large numbers

### The weak law

> [!note] Weak law of large numbers
> If $X_1,X_2,\ldots$ are i.i.d. with mean $\mu$ and finite variance $\sigma^2$, then $\bar X_n\xrightarrow{p}\mu$: for every $\varepsilon>0$,
>
> $$
> \mathbb{P}\big(|\bar X_n-\mu|\ge\varepsilon\big)\le\frac{\sigma^2}{n\varepsilon^2}\xrightarrow[n\to\infty]{}0.
> $$

*Proof:* Chebyshev's inequality applied to $\bar X_n$, whose variance is $\sigma^2/n$.

This closes the loop opened in [[Probability Basics#The frequency model|Probability Basics]]. Take $X_i=\mathbb{1}\{A\text{ occurs in run }i\}$. Then $\mu=\mathbb{P}(A)$ and $\bar X_n=n_A/n$, so the relative frequency converges to the probability (Figure 4). The frequency model that motivated the axioms is a **theorem** of the axioms.

![[Probability and Linear Algebra/images/sample-statistics-lln.svg]]
*Figure 4. Five independent sequences of runs with $\mathbb{P}(A)=0.3$. Each running relative frequency $n_A/n$ wanders early on, then settles at 0.3. The shaded band is the 95% Hoeffding interval $0.3\pm\sqrt{\ln(2/0.05)/(2n)}$, which shrinks like $1/\sqrt n$.*

### The strong law

The weak law says that at each large $n$ a big error is unlikely. It does not rule out a single run having big errors over and over, at ever rarer times, like the typewriter. The strong law rules this out.

> [!note] Strong law of large numbers (Kolmogorov)
> If $X_1,X_2,\ldots$ are i.i.d. with $\mathbb{E}|X_1|<\infty$ and mean $\mu$, then $\bar X_n\xrightarrow{a.s.}\mu$.

With probability 1, the path of running averages of a single experiment converges to $\mu$ (each line in Figure 4). After some random time it enters any band around $\mu$ and never leaves, although we are not told when that time comes. Only a finite mean is needed, not a finite variance. Without a finite mean the law fails completely: the running averages of Cauchy samples jump around forever (Figure 5).

![[Probability and Linear Algebra/images/sample-statistics-cauchy.svg]]
*Figure 5. Running averages of four independent sample sequences. (a) Standard Gaussian samples: each path settles at the mean 0. (b) Standard Cauchy samples, which have no mean: an occasional enormous sample throws the average off, and the paths never settle.*

> [!proof]- Proof of the strong law, assuming a finite fourth moment
> The general proof (finite mean only) uses truncation and is in Kumar's notes and in Durrett.[^durrett2019] With a finite fourth moment there is a short proof by Borel–Cantelli. Let $Y_i=X_i-\mu$ and $S_n=\sum_{i=1}^nY_i$. Expanding $S_n^4$, every term with an index appearing exactly once has expectation 0, because $\mathbb{E}[Y_i]=0$ and the $Y_i$ are independent. What remains are the $n$ terms $Y_i^4$ and the $3n(n-1)$ terms $Y_i^2Y_j^2$ with $i\ne j$:
>
> $$
> \mathbb{E}[S_n^4]=n\,\mathbb{E}[Y_1^4]+3n(n-1)\sigma^4\le Cn^2.
> $$
>
> By Markov's inequality applied to $S_n^4$,
>
> $$
> \mathbb{P}\big(|\bar X_n-\mu|\ge\varepsilon\big)=\mathbb{P}\big(S_n^4\ge n^4\varepsilon^4\big)\le\frac{Cn^2}{n^4\varepsilon^4}=\frac{C}{n^2\varepsilon^4},
> $$
>
> which is summable over $n$. By Borel–Cantelli, with probability 1 only finitely many $n$ have $|\bar X_n-\mu|\ge\varepsilon$. Apply this to $\varepsilon=1,\tfrac12,\tfrac13,\dots$; a countable union of probability-zero events has probability zero, so with probability 1 the deviation eventually stays below every $\varepsilon$. That is, $\bar X_n\to\mu$ almost surely.
>
> Chebyshev alone would give only $\sigma^2/(n\varepsilon^2)$, which is not summable. The fourth moment buys the extra factor of $1/n$.

### Monte Carlo integration

The law of large numbers is also a way to **compute**. To find an expectation $\mathbb{E}[g(X)]$, or any integral that can be written as one, draw $X_1,\dots,X_n$ from the distribution of $X$ and average:

$$
\mathbb{E}[g(X)]\approx\frac1n\sum_{i=1}^ng(X_i).
$$

The strong law guarantees convergence. The error has standard deviation $\sigma_g/\sqrt n$, where $\sigma_g$ is the standard deviation of $g(X)$, **regardless of the dimension of $X$**. Grid-based numerical integration needs exponentially many points in high dimensions, so Monte Carlo wins whenever the dimension is more than a handful. Policy gradients, the ELBO, and the expectations in reinforcement learning are all estimated this way. The [[#The central limit theorem|central limit theorem]] below adds an error bar.

> [!example]- Code: Monte Carlo integration with an error bar
> ```python
> import numpy as np
> from scipy import special
> rng = np.random.default_rng(0)
>
> def mc_estimate(g, sampler, n):
>     y = g(sampler(n))
>     est, se = y.mean(), y.std(ddof=1) / np.sqrt(n)     # standard error sigma/sqrt(n)
>     return est, (est - 1.96 * se, est + 1.96 * se)     # approximate 95% interval
>
> # integral of exp(-x^2) over [0, 1] = E[exp(-U^2)] for U ~ Uniform(0, 1)
> est, ci = mc_estimate(lambda u: np.exp(-u**2), rng.random, 10_000)
> print(est, ci)                                         # true value 0.746824...
>
> # does the interval cover the truth about 95% of the time?
> truth = np.sqrt(np.pi) / 2 * special.erf(1)
> cover = np.mean([lo <= truth <= hi for _, (lo, hi) in
>                  (mc_estimate(lambda u: np.exp(-u**2), rng.random, 1_000) for _ in range(2_000))])
> print(cover)                                           # about 0.95
> ```

## The central limit theorem

The law of large numbers says that $\bar X_n-\mu\to0$. The **central limit theorem** zooms in on the error, magnified by $\sqrt n$ to keep it from vanishing, and finds a Gaussian:

> [!note] Central limit theorem
> If $X_1,X_2,\ldots$ are i.i.d. with mean $\mu$ and finite variance $\sigma^2>0$, then
>
> $$
> Z_n=\frac{\sqrt n\,(\bar X_n-\mu)}{\sigma}\xrightarrow{d}\mathcal{N}(0,1),
> $$
>
> that is, $\mathbb{P}(Z_n\le z)\to\Phi(z)$ for every $z$.

This holds **whatever** the distribution of the $X_i$, as long as the variance is finite. Figure 6 shows it for the very skewed exponential distribution. [[Joint Distributions#Sums of independent random variables|Joint Distributions]] showed the mechanism: the distribution of a sum is a repeated convolution, and repeated convolution smooths any shape into a bell. This is why Gaussians are everywhere. Any quantity built from many small, independent contributions, such as measurement noise, heights, or the sum of a network's many weighted inputs at initialization, is approximately Gaussian.

![[Probability and Linear Algebra/images/sample-statistics-clt.svg]]
*Figure 6. The central limit theorem for the heavily skewed Exponential(1) distribution. Histograms of the standardized mean of $n$ samples (100 000 repetitions each) against the standard Gaussian density. At $n=1$ the histogram is the exponential itself; by $n=30$ it is close to Gaussian, with a slight remaining skew.*

The convergence is of **CDFs**, that is, of probabilities of intervals, not of densities. For a discrete distribution, the distribution of the sum is always a comb of spikes and never looks like a smooth curve. Its CDF, however, approaches the Gaussian CDF very closely (Figure 7).

![[Probability and Linear Algebra/images/sample-statistics-clt-discrete.svg]]
*Figure 7. Why the CLT is stated with CDFs. (a) The PMF of Binomial(40, 0.3), a sum of 40 Bernoullis, is a comb of spikes, which no density can match. (b) Its CDF is nonetheless very close to that of the Gaussian with the same mean and variance.*

> [!proof]- Proof sketch via moment-generating functions
> Assume the MGF of $Y=X_1-\mu$ exists near 0; the rigorous proof uses characteristic functions, which always exist, in the same way.[^durrett2019] Since $\mathbb{E}[Y]=0$ and $\mathbb{E}[Y^2]=\sigma^2$, a Taylor expansion gives $M_Y(u)=1+\frac{\sigma^2u^2}{2}+o(u^2)$. $Z_n$ is a sum of $n$ independent copies of $Y/(\sigma\sqrt n)$, so
>
> $$
> M_{Z_n}(s)=M_Y\Big(\frac{s}{\sigma\sqrt n}\Big)^n=\Big(1+\frac{s^2}{2n}+o\Big(\frac1n\Big)\Big)^n\longrightarrow e^{s^2/2},
> $$
>
> which is the MGF of $\mathcal{N}(0,1)$. A continuity theorem (Lévy's, for characteristic functions) turns convergence of the transforms into convergence of the CDFs. Only the mean and the variance survive the limit: all the higher moments are in the $o(1/n)$ term, which is why the limit does not depend on the shape of the distribution.

**Using the CLT: error bars.** For large $n$, $\bar X_n$ is approximately $\mathcal{N}(\mu,\sigma^2/n)$, so $\bar X_n\pm1.96\,\sigma/\sqrt n$ is an approximate 95% **confidence interval** for $\mu$ (with $\sigma$ replaced by the sample standard deviation). An opinion poll of $n=1000$ people with about half answering "yes" has a margin of error $1.96\sqrt{0.25/1000}\approx3.1$ percentage points. This is where the familiar "±3%" comes from, and why polls rarely survey many more people: halving the margin would need 4000.

**How good is the approximation?** The **Berry–Esseen theorem** bounds the error:[^berry1941]

$$
\sup_z\big|\mathbb{P}(Z_n\le z)-\Phi(z)\big|\le\frac{C\,\mathbb{E}|X-\mu|^3}{\sigma^3\sqrt n},
$$

with a universal constant $C$ below 0.5. The error shrinks like $1/\sqrt n$ and is larger for skewed distributions, as Figure 6 shows. Three warnings:
- **The tails converge slowly.** The bound is on the absolute error. A tail probability of $10^{-6}$ can be off by a large factor even when the absolute error is tiny. For rare events, use the Chernoff bounds above.
- **Skewed or small samples** need care; the "$n\ge30$" rule of thumb is only a rule of thumb.
- **Infinite variance breaks it.** Sums of Cauchy or other heavy-tailed variables converge, after a different scaling, to non-Gaussian **stable** distributions, or not at all.

Wasserman's book is a good guide to using these results in statistics.[^wasserman2004]

## Key takeaways

- The sample mean is unbiased with standard error $\sigma/\sqrt n$. The sample variance divides by $n-1$ to be unbiased.
- MGFs and characteristic functions turn independent sums into products and identify distributions. Characteristic functions always exist, and they show that averages of Cauchy variables never concentrate.
- Jensen's inequality: $g(\mathbb{E}X)\le\mathbb{E}\,g(X)$ for convex $g$, behind the ELBO and $D_{\mathrm{KL}}\ge0$. Chernoff: apply Markov to $e^{sX}$ and optimize; averages have exponentially small tails. Hoeffding needs only boundedness. Plus a union bound, $M$ estimates cost $\ln M$.
- Convergence in distribution, in probability, in mean square and almost surely are different. Almost sure and mean square each imply in probability, which implies in distribution. Borel–Cantelli proves almost sure convergence: finitely many expected bad events means finitely many bad events.
- The weak law (Chebyshev) and the strong law (Borel–Cantelli) make the frequency model a theorem and justify Monte Carlo, whose error $\sigma/\sqrt n$ is independent of dimension.
- The CLT: $\sqrt n(\bar X_n-\mu)/\sigma\to\mathcal{N}(0,1)$ in distribution, for any finite-variance distribution. It gives confidence intervals, converges at rate $1/\sqrt n$, and is poor in the far tails.

## Where this shows up next

- [[Markov Chains]] proves a law of large numbers for dependent sequences: long-run time averages along a chain converge to averages under the stationary distribution. That is what makes MCMC work.
- [[Multi-Armed Bandits]] uses Hoeffding and the union bound in the UCB regret proof.
- [[Policy Gradients]], [[Variational Autoencoders]] and [[Score-Based Models]] estimate expectations by Monte Carlo; the variance of these estimates is a recurring theme.
- [[Clustering and Expectation Maximization]] and [[Variational Autoencoders]] use Jensen's inequality to build lower bounds on the likelihood.

<!-- References: footnote definitions below render as the numbered "References" list at the bottom of the page -->

[^chan2026]: Chan, S. H. (2026). *Introduction to Probability for Data Science* (2nd ed.), Chapter 6. Michigan Publishing. [book website](https://probability4datascience.com)
[^kumar2012]: Kumar, A. (2012). *Discrete Event Stochastic Processes: Lecture Notes for an Engineering Curriculum*, Sections 1.4–1.5. Indian Institute of Science. [PDF](https://ece.iisc.ac.in/~anurag/books/anurag/spqt.pdf)
[^durrett2019]: Durrett, R. (2019). *Probability: Theory and Examples* (5th ed.), Chapters 2–3. Cambridge University Press. [Free online edition](https://sites.math.duke.edu/~rtd/PTE/pte.html)
[^chernoff1952]: Chernoff, H. (1952). *A Measure of Asymptotic Efficiency for Tests of a Hypothesis Based on the Sum of Observations*. The Annals of Mathematical Statistics, 23(4), 493–507. [doi:10.1214/aoms/1177729330](https://doi.org/10.1214/aoms/1177729330)
[^hoeffding1963]: Hoeffding, W. (1963). *Probability Inequalities for Sums of Bounded Random Variables*. Journal of the American Statistical Association, 58(301), 13–30. [doi:10.1080/01621459.1963.10500830](https://doi.org/10.1080/01621459.1963.10500830)
[^berry1941]: Berry, A. C. (1941). *The Accuracy of the Gaussian Approximation to the Sum of Independent Variates*. Transactions of the American Mathematical Society, 49(1), 122–136. [doi:10.1090/S0002-9947-1941-0003498-3](https://doi.org/10.1090/S0002-9947-1941-0003498-3)
[^wasserman2004]: Wasserman, L. (2004). *All of Statistics: A Concise Course in Statistical Inference*, Chapters 4–6. Springer. [doi:10.1007/978-0-387-21736-9](https://doi.org/10.1007/978-0-387-21736-9)
