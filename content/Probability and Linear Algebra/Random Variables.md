---
title: Random Variables
tags:
  - probability
  - foundations
date: 2026-11-12
draft: false
---

> **Prerequisites:** [[Probability Basics]] (events, axioms, conditioning, independence).
>
> **Series:** this is part 2 of 5. [[Probability Basics]] → **Random Variables** → [[Joint Distributions]] → [[Sample Statistics]] → [[Markov Chains]].
>
> **TL;DR:**
> - A random variable turns outcomes into numbers. Its distribution is described by a PMF (discrete), a PDF (continuous) or, always, a CDF. A density is not a probability; only its integrals are.
> - The expectation is the balance point of the distribution. LOTUS computes $\mathbb{E}[g(X)]$ without the distribution of $g(X)$, and the law of total expectation splits an average into cases.
> - Markov's and Chebyshev's inequalities bound tails using only the mean and the variance.
> - The common distributions are linked: Bernoulli trials add up to a binomial, a binomial with rare successes becomes Poisson, and the waiting times (geometric, exponential) are **memoryless**. A constant hazard rate is the exponential; survival analysis is about non-constant ones.
> - Every distribution can be sampled from uniforms: by inverting the CDF, or by rejection sampling.
> - In a Poisson process the counts are Poisson and the gaps exponential. A Hawkes process lets each event raise the rate of future events; it is simulated by thinning, which is rejection sampling in time.

## Motivation

[[Probability Basics]] worked with events: sets of outcomes. But we rarely care about the outcome itself. We care about a number computed from it: the number of heads, the waiting time until the next request, the pixel intensity, the return of an episode. Numbers can be added, averaged and plotted, and this is what a random variable gives us.

Kumar's notes put it nicely: a random variable is a **question about the experiment** whose answer is a number, and that can be answered from the information available.[^kumar2012] Chan calls it a **translator** from outcomes to numbers.[^chan2026] Either way, the probability of a statement about $X$ is always computed by translating it back into an event.

## Random variables

A **random variable** is a function $X:\Omega\to\mathbb{R}$. For two coin tosses, $X$ = number of heads maps the four outcomes to the three values $0,1,2$ (Figure 1). The probability that $X$ takes a value is the probability of the set of outcomes mapped to it:

$$
\mathbb{P}(X=1)=\mathbb{P}\big(\{\omega:X(\omega)=1\}\big)=\mathbb{P}(\{HT,TH\})=\tfrac12.
$$

![[Probability and Linear Algebra/images/random-variables-mapping.svg]]
*Figure 1. A random variable is a function from outcomes to numbers. Here $X$ counts heads in two tosses. Several outcomes can map to the same number; the event $\{X=1\}$ is the set of all of them (its **pre-image**).*

Two remarks:
- $X$ is a function, not a number. Uppercase $X$ is the random variable and lowercase $x$ a value it can take. $\mathbb{P}(X=x)$ is shorthand for the probability of the event $\{\omega:X(\omega)=x\}$.
- For $\mathbb{P}(X\le x)$ to make sense, $\{X\le x\}$ must be an event, that is, in $\mathcal{F}$. Functions with this property are called **measurable**. Every function we meet is measurable, but the condition has a meaning: $X$ must be answerable from the information in $\mathcal{F}$.

> [!note]- Aside: a function that is not a random variable
> Roll a die, but suppose we only ever learn whether the result is even or odd. The available events are $\mathcal{F}=\{\emptyset,\{2,4,6\},\{1,3,5\},\Omega\}$. The function "is the result divisible by 3" is not measurable: $\{X=1\}=\{3,6\}$ is not in $\mathcal{F}$, because knowing the parity does not tell us whether the roll was divisible by 3. This is Kumar's example of a "question" that the available information cannot answer.[^kumar2012]

## Distributions

### The probability mass function

A **discrete** random variable takes finitely or countably many values. Its **probability mass function** (PMF) is

$$
p_X(x)=\mathbb{P}(X=x),\qquad p_X(x)\ge0,\qquad \sum_xp_X(x)=1.
$$

The PMF is the **ideal histogram**. Roll a die $N$ times and plot the fraction of each face: as $N$ grows, the bars settle at the PMF (Figure 2). This is the frequency model of [[Probability Basics#The frequency model|Probability Basics]] again, now for all values at once.

![[Probability and Linear Algebra/images/random-variables-histograms.svg]]
*Figure 2. Histograms of $N$ rolls of a fair die. With 30 rolls the bars are ragged; with 30 000 they sit on the PMF $1/6$ (dashed). The PMF is what histograms converge to.*

Different random variables can share a PMF. If $X$ is uniform on $\{-1,0,1\}$, then $Y=-X$ has exactly the same PMF, although $X\ne Y$ for most outcomes. The PMF describes the **distribution** of values, not the random variable itself.

### The cumulative distribution function

The **cumulative distribution function** (CDF) is

$$
F_X(x)=\mathbb{P}(X\le x).
$$

It is the running total of probability as we sweep from left to right. Every random variable has one, discrete or not, and it determines the distribution completely. Its properties follow from the axioms:
- it is non-decreasing, since $\{X\le a\}\subseteq\{X\le b\}$ for $a\le b$ (monotonicity);
- $F_X(x)\to0$ as $x\to-\infty$ and $F_X(x)\to1$ as $x\to+\infty$ (continuity of probability);
- it is right-continuous, because "$\le$" includes the endpoint;
- $\mathbb{P}(a<X\le b)=F_X(b)-F_X(a)$.

For a discrete variable the CDF is a **staircase**, and the height of each jump is the PMF at that point (Figure 3a–b): $p_X(x)=F_X(x)-F_X(x^-)$.

### The probability density function

A **continuous** random variable has a **probability density function** (PDF) $f_X\ge0$ with

$$
\mathbb{P}(a\le X\le b)=\int_a^bf_X(x)\,dx,\qquad F_X(x)=\int_{-\infty}^{x}f_X(t)\,dt,\qquad f_X=F_X'.
$$

The CDF is the area under the PDF up to $x$, and the PDF is the slope of the CDF (Figure 3c–d). The word **density** is meant literally, as in mass per unit length. A histogram with ever narrower bins, rescaled so its total area is 1, converges to the density. For a small interval,

$$
\mathbb{P}(x\le X\le x+\delta)\approx f_X(x)\,\delta.
$$

![[Probability and Linear Algebra/images/random-variables-pmf-cdf-pdf.svg]]
*Figure 3. The two kinds of distribution and their CDFs. (a) The PMF of the number of heads in three tosses. (b) Its CDF is a staircase; each jump equals the PMF value. (c) A density: the probability of $X\le x_0$ is the shaded area. (d) The CDF of the density in (c) is smooth, its height at $x_0$ is the shaded area, and its slope is the density.*

> [!warning] A density is not a probability
> $f_X(x)$ can exceed 1. Uniform on $[0.2,0.6]$ has density $2.5$ on that interval, and $f(x)=1/(2\sqrt x)$ on $(0,1]$ is unbounded yet integrates to 1. What must hold is that the **area** is 1. Also, $\mathbb{P}(X=x)=0$ for every single point, so for a continuous variable it does not matter whether interval endpoints are included.

> [!note]- Aside: mixed random variables
> Some quantities are neither discrete nor continuous. Daily rainfall is exactly 0 on most days, and a positive amount with a density on the others. Its CDF jumps at 0 (the probability of a dry day) and then rises smoothly. The CDF handles this without any difficulty, which is one reason it is the most general description of a distribution. Formally, a CDF can always be split into a jump part, a density part, and a third "singular" part that never appears in practice.[^kumar2012]

## Expectation

### The balance point

The **expectation** (mean) of $X$ is the probability-weighted average of its values:

$$
\mathbb{E}[X]=\sum_xx\,p_X(x)\qquad\text{or}\qquad\mathbb{E}[X]=\int_{-\infty}^{\infty}x\,f_X(x)\,dx.
$$

Two pictures help.
- **Centre of mass.** Place a weight $p_X(x)$ at each position $x$ on a rod. $\mathbb{E}[X]$ is the point where the rod balances. A heavy weight far out pulls the balance point towards it.
- **The long-run average.** Average $N$ observations $x_1,\dots,x_N$ and group them by value: $\frac1N\sum_ix_i=\sum_xx\cdot\frac{\#\{i:x_i=x\}}{N}$. This is the expectation formula with the empirical histogram in place of the PMF. As $N$ grows, the histogram approaches the PMF (Figure 2), so the sample mean approaches $\mathbb{E}[X]$. The [[Sample Statistics#The law of large numbers|law of large numbers]] makes this precise.

The expectation need not be a value $X$ can take. A biased coin with $\mathbb{P}(\text{heads})=3/4$, coded $X\in\{0,1\}$, has $\mathbb{E}[X]=3/4$.

**Indicators turn probabilities into expectations.** The indicator $\mathbb{1}_A$ is 1 if $A$ happens and 0 otherwise. It is a Bernoulli random variable, and

$$
\mathbb{E}[\mathbb{1}_A]=1\cdot\mathbb{P}(A)+0\cdot\mathbb{P}(A^c)=\mathbb{P}(A).
$$

Every probability is an expectation. Many proofs below, and in the rest of the series, use this identity.

### Functions of a random variable: LOTUS

To compute $\mathbb{E}[g(X)]$ we do not need the distribution of $Y=g(X)$. It suffices to average $g$ over the distribution of $X$:

$$
\mathbb{E}[g(X)]=\sum_xg(x)\,p_X(x)\qquad\text{or}\qquad\mathbb{E}[g(X)]=\int g(x)\,f_X(x)\,dx.
\tag{1}
$$

This is called the **law of the unconscious statistician** (LOTUS), because it is so natural that people use it without noticing that it needs proof.[^blitzstein2019] For a discrete $X$, the proof just groups the values of $x$ by the value of $g(x)$:

$$
\mathbb{E}[Y]=\sum_yy\,\mathbb{P}(g(X)=y)=\sum_yy\sum_{x:\,g(x)=y}p_X(x)=\sum_xg(x)\,p_X(x).
$$

Picture the PMF as a set of stems. Applying $g$ moves each stem to a new position $g(x)$ but keeps its height; stems that land on the same position stack up. LOTUS is used throughout the series, for example in [[Variational Autoencoders#Gradients of an expectation|the VAE lecture]].

Two consequences:
- **Linearity:** $\mathbb{E}[aX+b]=a\,\mathbb{E}[X]+b$. More generally $\mathbb{E}[X+Y]=\mathbb{E}[X]+\mathbb{E}[Y]$ for any $X,Y$, independent or not; we prove this in [[Joint Distributions#Linearity of expectation|Joint Distributions]].
- **Nonlinear functions do not commute with expectations:** $\mathbb{E}[g(X)]\ne g(\mathbb{E}[X])$ in general. If $X$ is 1 or 2 with equal probability, $\mathbb{E}[1/X]=0.75$ but $1/\mathbb{E}[X]=0.67$. For convex $g$, Jensen's inequality says $\mathbb{E}[g(X)]\ge g(\mathbb{E}[X])$; it is the inequality behind the ELBO in [[Variational Autoencoders]].

### Expectation from the tail

For a non-negative random variable there is a second formula, which uses only the tail probabilities:

$$
\begin{aligned}
\mathbb{E}[X]&=\int_0^{\infty}\mathbb{P}(X>t)\,dt=\int_0^{\infty}\big(1-F_X(t)\big)\,dt,\\
\mathbb{E}[X]&=\sum_{k=1}^{\infty}\mathbb{P}(X\ge k)\quad\text{for integer-valued }X\ge0.
\end{aligned}
\tag{2}
$$

The idea is to write $X$ as a sum of indicators. An integer $X\ge0$ equals $\sum_{k\ge1}\mathbb{1}\{X\ge k\}$: a value of 3 is counted once at each of $k=1,2,3$. Taking expectations term by term gives the sum. The integral version is the same with $X=\int_0^\infty\mathbb{1}\{X>t\}\,dt$. Geometrically, $\mathbb{E}[X]$ is the **area above the CDF** (Figure 4b). Formula (2) will make the mean of the geometric distribution a one-liner, and it is how [[#Survival and hazard functions|survival analysis]] computes mean lifetimes.

### Mean, median and mode

The mean is one notion of a "typical" value. The **median** $m$ splits the probability in half, $F_X(m)=1/2$. The **mode** is the most likely value, the peak of the PMF or PDF. For a symmetric unimodal distribution all three coincide. For a right-skewed one, such as incomes or waiting times, the long right tail pulls the mean furthest out (Figure 4a): the mode is smaller than the median, which is smaller than the mean.

Each is the best guess under a different loss. Expanding the square,

$$
\mathbb{E}\big[(X-c)^2\big]=\operatorname{Var}(X)+(\mathbb{E}[X]-c)^2,
$$

so the mean minimizes the expected **squared** error. The median minimizes the expected **absolute** error $\mathbb{E}|X-c|$: moving $c$ to the right by a small step $\delta$ adds $\delta$ for the outcomes to the left of $c$ and saves $\delta$ for those to the right. That trade-off breaks even when half the probability is on each side. This is why regression with squared loss estimates conditional means ([[Linear Models]]), why absolute loss is robust to outliers, and why a single huge value moves the mean but not the median.

![[Probability and Linear Algebra/images/random-variables-center.svg]]
*Figure 4. (a) A right-skewed (log-normal) density: the mode (1.15), median (1.65) and mean (1.97) separate, in that order. (b) For $X\ge0$, the mean equals the area between the CDF and 1, formula (2).*

> [!warning] The mean need not exist
> The expectation is defined only when $\sum|x|\,p_X(x)$ or $\int|x|\,f_X(x)\,dx$ is finite. The **Cauchy** density $f(x)=\frac{1}{\pi(1+x^2)}$ is symmetric about 0, yet has no mean. Its tails decay so slowly that $\int_0^\infty x\,f(x)\,dx=\infty$, and the left half gives $-\infty$, and $\infty-\infty$ is undefined. Averages of Cauchy samples never settle down ([[Sample Statistics#The law of large numbers|Sample Statistics]]). Similarly, $p(k)=\frac{6}{\pi^2k^2}$ for $k\ge1$ is a valid PMF with $\mathbb{E}[X]=\infty$.

## Variance and moments

The **variance** measures spread around the mean $\mu=\mathbb{E}[X]$, as the expected squared distance:

$$
\operatorname{Var}(X)=\mathbb{E}\big[(X-\mu)^2\big]=\mathbb{E}[X^2]-\mu^2.
$$

The second form follows by expanding the square and using linearity. Its square root $\sigma$ is the **standard deviation**, which has the units of $X$. Shifting a distribution does not change its spread, and scaling stretches it:

$$
\operatorname{Var}(aX+b)=a^2\operatorname{Var}(X).
$$

For example, a Bernoulli$(p)$ variable has $\mathbb{E}[X^2]=\mathbb{E}[X]=p$, so $\operatorname{Var}(X)=p(1-p)$. It is largest at $p=\tfrac12$, the most uncertain coin, and zero when the outcome is certain.

More generally $\mathbb{E}[X^k]$ is the $k$-th **moment**. The third and fourth standardized moments, the **skewness** $\mathbb{E}[(X-\mu)^3]/\sigma^3$ and the **kurtosis** $\mathbb{E}[(X-\mu)^4]/\sigma^4$, describe asymmetry and the weight of the tails. Two distributions with the same mean and variance can still look very different. [[Sample Statistics#Moment-generating functions|Sample Statistics]] packs all the moments into a single function.

## Conditioning on an event

### The law of total probability, for random variables

Conditioning on an event $B$ with $\mathbb{P}(B)>0$ gives a new distribution, $p_{X\mid B}(x)=\mathbb{P}(X=x\mid B)$, with its own mean $\mathbb{E}[X\mid B]=\sum_xx\,p_{X\mid B}(x)$. If $B_1,\dots,B_K$ partition $\Omega$, the [[Probability Basics#Conditioning on evidence|law of total probability]] applied to each event $\{X=x\}$ gives

$$
p_X(x)=\sum_{k=1}^{K}p_{X\mid B_k}(x)\,\mathbb{P}(B_k),
$$

and the same holds for densities. The distribution of $X$ is a **mixture** of the conditional distributions, weighted by the probabilities of the cases. Gaussian mixture models ([[Clustering and Expectation Maximization]]) are exactly this, with each cluster a case.

### The law of total expectation

Multiplying by $x$ and summing gives the **law of total expectation**:

$$
\mathbb{E}[X]=\sum_{k=1}^{K}\mathbb{E}[X\mid B_k]\,\mathbb{P}(B_k).
\tag{3}
$$

The overall average is the average of the case averages, weighted by how often each case occurs. A class's mean mark is the weighted average of the sections' means, weighted by section size.

> [!example] Example: first-step analysis
> How many tosses of a coin with $\mathbb{P}(\text{heads})=p$ does it take to see the first head? Call the answer $X$, and condition on the first toss.
> - With probability $p$ it is heads, and $X=1$.
> - With probability $1-p$ it is tails. Then one toss is used up and, since the tosses are independent, the problem starts afresh: the remaining number of tosses has the same distribution as $X$. So $\mathbb{E}[X\mid\text{tails}]=1+\mathbb{E}[X]$.
>
> By (3), $\mathbb{E}[X]=p\cdot1+(1-p)(1+\mathbb{E}[X])$, and solving gives $\mathbb{E}[X]=1/p$. A fair coin needs 2 tosses on average, a 1-in-100 event needs 100.
>
> This trick of conditioning on the first step and recognizing the same problem again is called **first-step analysis**. It computes hitting times and ruin probabilities in [[Markov Chains]], and it is the Bellman equation of [[Introduction to Reinforcement Learning|reinforcement learning]].

[[Joint Distributions#Conditional expectation|Joint Distributions]] generalizes (3) from a partition into events to conditioning on another random variable, where $\mathbb{E}[X\mid Y]$ becomes a random variable itself.

## Markov's and Chebyshev's inequalities

Often we know the mean and perhaps the variance, but not the whole distribution. How likely can a large value be?

**Markov's inequality.** If $X\ge0$ and $a>0$, then

$$
\mathbb{P}(X\ge a)\le\frac{\mathbb{E}[X]}{a}.
\tag{4}
$$

*Proof:* for every outcome, $X\ge a\,\mathbb{1}\{X\ge a\}$, because the right side is 0 when $X<a$, and $a\le X$ otherwise (Figure 5a). Take expectations of both sides: $\mathbb{E}[X]\ge a\,\mathbb{P}(X\ge a)$.

The intuition is that a non-negative quantity cannot often be much larger than its average. If the mean income is 50 000, at most 10% of people can earn 500 000 or more; otherwise those people alone would push the mean above 50 000.

**Chebyshev's inequality.** If $X$ has mean $\mu$ and variance $\sigma^2$, then for $k>0$,

$$
\mathbb{P}\big(|X-\mu|\ge k\big)\le\frac{\sigma^2}{k^2},\qquad\text{equivalently}\qquad\mathbb{P}\big(|X-\mu|\ge k\sigma\big)\le\frac{1}{k^2}.
\tag{5}
$$

*Proof:* apply Markov to the non-negative variable $(X-\mu)^2$ with $a=k^2$.

Chebyshev holds for **every** distribution with a finite variance: at most 25% of the probability lies more than 2 standard deviations from the mean, and at most 11% more than 3. The price of this generality is looseness. For a Gaussian the true numbers are 4.6% and 0.3%. Figure 5b compares the bounds with the exact tail of an exponential distribution. Both bounds decay polynomially, while the true tail decays exponentially. [[Sample Statistics#Chernoff bounds|Chernoff bounds]] recover the exponential rate by applying Markov to $e^{sX}$ instead of $X$.

![[Probability and Linear Algebra/images/random-variables-markov.svg]]
*Figure 5. (a) The one-line proof of Markov's inequality: for $x\ge0$, the line $x$ lies above the step $a\,\mathbb{1}\{x\ge a\}$, so the same holds for their averages. (b) For an Exponential(1) variable (mean 1, variance 1), the exact tail $\mathbb{P}(X\ge a)=e^{-a}$ against Markov's bound $1/a$ and Chebyshev's bound $1/(a-1)^2$. The bounds are valid but loose, and they decay only polynomially.*

## Discrete distributions

### Bernoulli and binomial

A **Bernoulli**$(p)$ variable is a single trial: $X=1$ ("success") with probability $p$ and $X=0$ otherwise. It has mean $p$ and variance $p(1-p)$. A binary label in [[Linear Models#Logistic regression|logistic regression]] is a Bernoulli variable whose $p$ depends on the features.

Run $n$ independent Bernoulli$(p)$ trials and count the successes: $X=X_1+\cdots+X_n$. Then $X$ is **Binomial**$(n,p)$:

$$
p_X(k)=\binom{n}{k}p^k(1-p)^{n-k},\qquad k=0,1,\dots,n.
$$

The formula separates the two ingredients of [[Probability Basics]]:
- **independence** gives the probability of any one particular sequence with $k$ successes and $n-k$ failures, $p^k(1-p)^{n-k}$;
- **additivity** adds up the $\binom nk$ disjoint sequences with $k$ successes, one for each choice of positions for the successes.

Writing the binomial as a sum of Bernoullis makes its moments immediate. By linearity, $\mathbb{E}[X]=np$. Since the trials are independent their variances add ([[Joint Distributions#Variance of a sum|Joint Distributions]]), so $\operatorname{Var}(X)=np(1-p)$. Summing indicators is the main technique for computing means of counts.

### Geometric and memorylessness

Instead of fixing $n$, keep tossing until the first success. The number of trials $X$ is **Geometric**$(p)$: the first success at trial $k$ needs $k-1$ failures followed by a success, so

$$
p_X(k)=(1-p)^{k-1}p,\qquad\mathbb{P}(X>k)=(1-p)^k,\qquad k=1,2,\dots
$$

The tail formula follows because "$X>k$" means "the first $k$ trials all failed". By (2), $\mathbb{E}[X]=\sum_{k\ge0}(1-p)^k=1/p$, the same answer as first-step analysis gave.

The geometric distribution is **memoryless**. Given that the first $m$ trials have failed, the number of further trials needed has the same distribution as if we had just started:

$$
\mathbb{P}(X>m+k\mid X>m)=\frac{(1-p)^{m+k}}{(1-p)^{m}}=(1-p)^k=\mathbb{P}(X>k).
$$

The coin does not remember its past failures (Figure 7a). This is also why the **gambler's fallacy** is a fallacy: after ten reds at roulette, black is not "due". Memorylessness characterizes the geometric distribution: it is the only distribution on $\{1,2,\dots\}$ with this property, because the condition forces $\mathbb{P}(X>k)=\mathbb{P}(X>1)^k$.

### Poisson: the law of rare events

Many counts come from a huge number of independent opportunities, each very unlikely: typos on a page, requests to a server in a second, decays of a radioactive sample, mutations in a genome. Take a binomial with $n\to\infty$ trials and success probability $p=\lambda/n$, so that the expected count $\lambda=np$ stays fixed. Then

$$
\begin{aligned}
\binom nk\Big(\frac\lambda n\Big)^k\Big(1-\frac\lambda n\Big)^{n-k}
&=\underbrace{\frac{n(n-1)\cdots(n-k+1)}{n^k}}_{\to1}\cdot\frac{\lambda^k}{k!}\\
&\qquad\cdot\underbrace{\Big(1-\frac\lambda n\Big)^{n}}_{\to e^{-\lambda}}\cdot\underbrace{\Big(1-\frac\lambda n\Big)^{-k}}_{\to1}\\
&\longrightarrow e^{-\lambda}\frac{\lambda^k}{k!}.
\end{aligned}
$$

The limit is the **Poisson**$(\lambda)$ distribution, $p_X(k)=e^{-\lambda}\lambda^k/k!$ for $k=0,1,2,\dots$ It has mean and variance both equal to $\lambda$, inherited from the binomial's $np\to\lambda$ and $np(1-p)\to\lambda$. Figure 6 shows how quickly the binomial approaches it: with $n=60$ trials of probability $0.05$, the probability of exactly 3 successes is 0.230, against the Poisson 0.224.

![[Probability and Linear Algebra/images/random-variables-binomial-poisson.svg]]
*Figure 6. Binomial$(n,3/n)$ for $n=6,15,60$ against the Poisson(3) limit. All have mean 3. As the trials become more numerous and individually rarer, the binomial approaches the Poisson.*

The Poisson model assumes the opportunities are independent and that the rate is constant. Counts that cluster, such as aftershocks following an earthquake, have variance larger than their mean. The [[#Hawkes processes: self-exciting events|Hawkes process]] below is a model for such counts.

## Continuous distributions

### Uniform

**Uniform**$(a,b)$ has constant density $1/(b-a)$ on $[a,b]$, so the probability of a sub-interval is proportional to its length. Its mean is $(a+b)/2$ and its variance $(b-a)^2/12$. The uniform on $[0,1]$ is the raw material of simulation: every other distribution can be produced from it ([[#Sampling from any distribution|below]]).

### Exponential and memorylessness

The **Exponential**$(\lambda)$ distribution models waiting times:

$$
f_X(x)=\lambda e^{-\lambda x},\qquad\mathbb{P}(X>t)=e^{-\lambda t},\qquad x,t\ge0,
$$

with mean $1/\lambda$ and variance $1/\lambda^2$. The rate $\lambda$ is the number of events per unit time, so $1/\lambda$ is the mean wait. It is the continuous version of the geometric distribution: chop time into slots of length $\delta$, give each slot an independent chance $\lambda\delta$ of an event, and the number of slots until the first event is geometric. As $\delta\to0$, $\mathbb{P}(X>t)=(1-\lambda\delta)^{t/\delta}\to e^{-\lambda t}$.

It inherits memorylessness:

$$
\mathbb{P}(X>s+t\mid X>s)=\frac{e^{-\lambda(s+t)}}{e^{-\lambda s}}=e^{-\lambda t}=\mathbb{P}(X>t).
$$

If the time until a light bulb fails is exponential, a bulb that has already burned for 1000 hours is as good as new. Figure 7b checks this by simulation. Among waiting times that exceeded 1.5, the remaining wait has exactly the original distribution.

![[Probability and Linear Algebra/images/random-variables-memoryless.svg]]
*Figure 7. Memorylessness. (a) Geometric with $p=0.25$: the distribution of the number of further trials, given 3 failures so far (orange), equals the original distribution (blue). (b) Exponential with $\lambda=1$: of 200 000 simulated waits, those longer than 1.5 have a remaining wait (orange outline) distributed exactly like a fresh wait (blue).*

> [!note]- Aside: the exponential is the only memoryless continuous distribution
> Let $S(t)=\mathbb{P}(X>t)$ for a waiting time $X\ge0$. Memorylessness says $S(s+t)=S(s)\,S(t)$ for all $s,t\ge0$. Then $S(2t)=S(t)^2$, and more generally $S(qt)=S(t)^q$ for every rational $q>0$. A non-increasing function satisfying this on the rationals is determined everywhere, so $S(t)=S(1)^t=e^{-\lambda t}$ with $\lambda=-\ln S(1)$. Kumar proves this carefully, because the same functional equation produces the Poisson process.[^kumar2012]

### Gaussian

The **Gaussian** (normal) distribution $\mathcal{N}(\mu,\sigma^2)$ has density

$$
f_X(x)=\frac{1}{\sqrt{2\pi\sigma^2}}\exp\Big(-\frac{(x-\mu)^2}{2\sigma^2}\Big).
$$

Its two parameters are exactly its mean and variance. Any Gaussian is a shifted and scaled standard one: $Z=(X-\mu)/\sigma\sim\mathcal{N}(0,1)$. So all Gaussian probabilities come from the standard CDF $\Phi$, which has no closed form:

$$
\mathbb{P}(X\le x)=\Phi\Big(\frac{x-\mu}{\sigma}\Big).
$$

About 68% of the probability lies within one standard deviation of the mean, 95% within two, and 99.7% within three (Figure 8). The tails decay like $e^{-x^2/2}$, faster than any exponential, which is why Gaussian outliers are so rare.

![[Probability and Linear Algebra/images/random-variables-gaussian.svg]]
*Figure 8. The 68–95–99.7 rule: the probability within 1, 2 and 3 standard deviations of the mean of any Gaussian is 0.683, 0.954 and 0.997. Compare Chebyshev's guarantees for an arbitrary distribution: at least 0, 0.75 and 0.89.*

> [!proof]- Why the Gaussian density integrates to 1
> Let $I=\int_{-\infty}^{\infty}e^{-x^2/2}\,dx$. Square it and switch to polar coordinates $x=r\cos\theta$, $y=r\sin\theta$, with $dx\,dy=r\,dr\,d\theta$:
>
> $$
> \begin{aligned}
> I^2&=\int\!\!\int e^{-(x^2+y^2)/2}\,dx\,dy=\int_0^{2\pi}\!\!\int_0^\infty e^{-r^2/2}\,r\,dr\,d\theta\\
> &=2\pi\Big[-e^{-r^2/2}\Big]_0^\infty=2\pi.
> \end{aligned}
> $$
>
> So $I=\sqrt{2\pi}$. The substitution $x\mapsto(x-\mu)/\sigma$ gives the general normalizing constant $\sqrt{2\pi\sigma^2}$.

The Gaussian is everywhere because of the [[Sample Statistics#The central limit theorem|central limit theorem]]: a sum of many small independent effects is approximately Gaussian, whatever the effects' own distributions. Its many-dimensional version is in [[Joint Distributions#The multivariate Gaussian|Joint Distributions]].

### Summary of common distributions

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

The family tree:
- Bernoulli → (add $n$ independent copies) → Binomial → ($n\to\infty$, $np=\lambda$) → Poisson.
- Bernoulli → (count trials until the first success) → Geometric → (shrink the time step) → Exponential.
- The Poisson counts and the exponential waiting times are two views of one object, the [[#The Poisson process|Poisson process]].
- Many independent small effects → (central limit theorem) → Gaussian.

Where they appear in the series:
- **Bernoulli** models binary labels in logistic regression. **Categorical** is the output of a softmax, and the component choice in a mixture model.
- **Poisson** models counts of rare events. **Uniform** expresses "no information" and drives random number generation. **Exponential** models waiting times.
- **Gaussian** is used for noise models, priors and latent variables throughout.

![[Probability and Linear Algebra/images/random-variables-distributions.svg]]
*Figure 9. (a) Binomial PMFs for 10 trials. (b) Poisson PMFs. (c) Exponential densities, which start at height $\lambda$ and decay at rate $\lambda$. (d) Gaussian densities. A narrow Gaussian has a peak above 1, a reminder that densities are not probabilities.*

## Survival and hazard functions

For a lifetime $T\ge0$ (time until a machine fails, a patient relapses, a customer cancels a subscription), it is more natural to ask "how likely is it to last beyond $t$?" than "how likely is it to fail before $t$?". The **survival function** is

$$
S(t)=\mathbb{P}(T>t)=1-F_T(t).
$$

By (2), the mean lifetime is the area under it, $\mathbb{E}[T]=\int_0^\infty S(t)\,dt$.

The **hazard rate** is the instantaneous failure rate of something that has survived so far:

$$
h(t)=\lim_{\delta\to0}\frac{\mathbb{P}(t<T\le t+\delta\mid T>t)}{\delta}=\frac{f_T(t)}{S(t)}=-\frac{d}{dt}\ln S(t).
\tag{6}
$$

Integrating the last form gives the survival function back from the hazard:

$$
S(t)=\exp\Big(-\int_0^th(u)\,du\Big).
$$

So the hazard is another complete description of the distribution. It is often the most interpretable one:
- **Constant hazard** $h(t)=\lambda$ gives $S(t)=e^{-\lambda t}$: the exponential distribution. Memorylessness is exactly "the risk never changes with age".
- **Increasing hazard** means wear-out: old machines fail more readily.
- **Decreasing hazard** means "infant mortality": items that survive an initial period become more reliable.

The **Weibull** family, $S(t)=e^{-(t/\eta)^k}$ with hazard $h(t)=\frac{k}{\eta}(t/\eta)^{k-1}$, covers all three cases with a single shape parameter $k$ (Figure 10). Real systems often show a "bathtub" hazard: decreasing early, flat during normal life, increasing at the end.

![[Probability and Linear Algebra/images/random-variables-hazard.svg]]
*Figure 10. Weibull lifetimes with scale 1. (a) The shape $k$ controls whether the hazard falls ($k<1$), stays constant ($k=1$, the exponential) or rises ($k>1$). (b) The corresponding survival functions.*

**Survival analysis**, the statistics of time-to-event data, is built on these functions.[^kalbfleisch2002] Its characteristic difficulty is **censoring**. A study ends while some patients are still alive, so for them we only know $T>t_{\text{end}}$. Methods such as the Kaplan–Meier estimator and Cox's proportional hazards model use these partial observations instead of discarding them. The same ideas apply to customer churn and to time-to-failure in reliability engineering.

## Functions of a random variable

If $Y=g(X)$, what is the distribution of $Y$? On a computer this is trivial: apply $g$ to each sample. On paper, the safe method is to **go through the CDF**: rewrite $\{Y\le y\}$ as an event about $X$, compute its probability, and differentiate.

**Monotone $g$.** If $g$ is increasing, $\{g(X)\le y\}=\{X\le g^{-1}(y)\}$, so $F_Y(y)=F_X(g^{-1}(y))$. If $g$ is decreasing the inequality flips. In both cases, differentiating gives the **change-of-variables formula**

$$
f_Y(y)=f_X\big(g^{-1}(y)\big)\,\Big|\frac{d}{dy}g^{-1}(y)\Big|.
\tag{7}
$$

The derivative factor conserves probability. Where $g$ **stretches** an interval, the same probability is spread over a longer interval, so the density drops. Where $g$ **squeezes**, the density rises. For a linear map $Y=aX+b$, (7) gives $f_Y(y)=\frac{1}{|a|}f_X\big(\frac{y-b}{a}\big)$: the density is stretched horizontally by $a$ and squashed vertically by the same factor.

**Non-monotone $g$.** Then $\{g(X)\le y\}$ can consist of several intervals, and (7) becomes a sum over all the $x$ with $g(x)=y$. Two examples:
- **Squares.** Let $X\sim\mathcal{N}(0,1)$ and $Y=X^2$. Then $F_Y(y)=\mathbb{P}(-\sqrt y\le X\le\sqrt y)=\Phi(\sqrt y)-\Phi(-\sqrt y)$, and differentiating gives $f_Y(y)=\frac{1}{\sqrt{2\pi y}}e^{-y/2}$ for $y>0$. This is the **chi-squared** distribution with one degree of freedom, the distribution of a squared Gaussian error.
- **A random phase.** Let $\Theta$ be uniform on $[0,2\pi)$ and $Y=\cos\Theta$, the position of a point moving around a circle, seen edge-on. Since $\cos$ is flat near its peaks and troughs, equal steps in $\Theta$ produce tiny steps in $Y$ near $\pm1$ and big steps near 0 (Figure 11a). The samples bunch up at the ends. Going through the CDF gives $F_Y(y)=1-\frac{\arccos y}{\pi}$ and the U-shaped **arcsine density** $f_Y(y)=\frac{1}{\pi\sqrt{1-y^2}}$ (Figure 11b). A pendulum spends most of its time near the ends of its swing, where it moves slowly.

![[Probability and Linear Algebra/images/random-variables-change-of-variables.svg]]
*Figure 11. Stretching and squeezing. (a) Evenly spaced values of $\Theta$ (grey ticks) mapped through $\cos$: the images (orange lines) crowd together near $\pm1$, where the curve is flat. (b) The histogram of $Y=\cos\Theta$ for uniform $\Theta$ matches the arcsine density, which is unbounded at $\pm1$.*

In $D$ dimensions the derivative in (7) becomes the absolute determinant of the Jacobian of $g^{-1}$. Normalizing flows (a planned lecture) build flexible densities by composing invertible maps and tracking exactly this factor.

## Sampling from any distribution

A computer produces (pseudo-)random numbers uniform on $[0,1]$. Every other distribution is manufactured from these. Devroye's book is the classic reference, and is free online.[^devroye1986]

### Inverse-CDF sampling

Formula (7) answers "given $g$, what is the distribution of $g(U)$?". Sampling asks the reverse: which $g$ turns a uniform into a given distribution? The answer is the inverse of the target CDF.

> [!note] Theorem (inverse-CDF sampling)
> Let $F$ be a CDF and $U\sim\operatorname{Uniform}(0,1)$. Then $X=F^{-1}(U)$ has CDF $F$. Conversely, if $X$ has a continuous CDF $F$, then $F(X)\sim\operatorname{Uniform}(0,1)$.

*Proof:* since $F$ is non-decreasing, $F^{-1}(U)\le x$ exactly when $U\le F(x)$. So $\mathbb{P}(X\le x)=\mathbb{P}(U\le F(x))=F(x)$, the CDF of a uniform evaluated at $F(x)$. For a CDF with jumps or flat parts, use the generalized inverse $F^{-1}(u)=\min\{x:F(x)\ge u\}$.

Picture it on the graph of $F$ (Figure 12a). Spread points evenly up the vertical axis and map each one horizontally to the curve and down to the $x$-axis. Where the CDF is steep, which is where the density is high, a whole band of heights lands on a short stretch of $x$, so samples pile up there. Where the CDF is flat, they are sparse. Some examples:
- **Exponential:** $F(x)=1-e^{-\lambda x}$ inverts to $X=-\ln(1-U)/\lambda$ (Figure 12b).
- **Discrete:** cut $[0,1]$ into consecutive pieces of lengths $p_1,p_2,\dots$ and output $k$ if $U$ falls in the $k$-th piece. This is the generalized inverse of the staircase CDF.
- **Gaussian:** $X=\mu+\sigma\,\Phi^{-1}(U)$, with $\Phi^{-1}$ computed numerically.

The converse, the **probability integral transform**, is just as useful. It turns any continuous variable into a uniform one, which is how calibration of predicted probabilities is checked and how Q–Q plots compare distributions.

![[Probability and Linear Algebra/images/random-variables-inverse-cdf.svg]]
*Figure 12. Inverse-CDF sampling for the Exponential(1) distribution. (a) Evenly spaced heights $u$ (left) are mapped through $F^{-1}$ to the $x$-axis (bottom). They land densely where the CDF is steep and sparsely where it is flat. (b) The histogram of 100 000 values $-\ln(1-U)$ matches the density $e^{-x}$.*

> [!example]- Code: inverse-CDF sampling
> ```python
> import numpy as np
> rng = np.random.default_rng(0)
>
> def sample_exponential(lam, n):
>     u = rng.random(n)
>     return -np.log1p(-u) / lam            # F^{-1}(u) = -ln(1-u)/lam
>
> def sample_discrete(values, probs, n):
>     cdf = np.cumsum(probs)                # cut [0,1] into pieces of length p_k
>     idx = np.searchsorted(cdf, rng.random(n), side="right")
>     return np.asarray(values)[np.minimum(idx, len(values) - 1)]
>
> x = sample_exponential(2.0, 100_000)
> print(x.mean(), x.var())                  # about 0.5 and 0.25
> d = sample_discrete([1, 2, 3, 4], [0.1, 0.5, 0.3, 0.1], 100_000)
> print(np.bincount(d)[1:] / len(d))        # about [0.1 0.5 0.3 0.1]
> ```

### Rejection sampling

Inverse-CDF sampling needs $F^{-1}$, which is often unavailable, for example when we only know the density, or only know it up to a constant. **Rejection sampling** needs much less: a density $g$ that we *can* sample from, and a constant $M$ such that $f(x)\le M\,g(x)$ everywhere. The curve $Mg$ is called the **envelope**.

> [!example] Algorithm: rejection sampling
> Repeat until a sample is accepted:
> 1. Draw a proposal $X\sim g$.
> 2. Draw $U\sim\operatorname{Uniform}(0,1)$.
> 3. Accept $X$ if $U\le\dfrac{f(X)}{M\,g(X)}$; otherwise reject it and go back to step 1.

The geometric picture explains why this works (Figure 13). The pair $\big(X,\,U\,Mg(X)\big)$ is a point uniformly distributed **under the graph of $Mg$**. Keeping only the points that fall under the graph of $f$ leaves points uniformly distributed under $f$. The $x$-coordinate of a point uniform under the graph of $f$ has density exactly $f$, because the amount of area above each $x$ is proportional to $f(x)$.

> [!proof]- Proof that accepted samples have density $f$
> For any $x$,
>
> $$
> \mathbb{P}(X\le x,\ \text{accept})=\int_{-\infty}^{x}g(t)\,\frac{f(t)}{M\,g(t)}\,dt=\frac{F(x)}{M}.
> $$
>
> Letting $x\to\infty$ gives $\mathbb{P}(\text{accept})=1/M$. So $\mathbb{P}(X\le x\mid\text{accept})=F(x)$. The same calculation works if $f$ is known only up to a constant $Z$, as long as $f/Z\le Mg$; the acceptance probability becomes $Z/M$.

Each proposal is accepted independently with probability $1/M$, so the number of proposals per accepted sample is [[#Geometric and memorylessness|geometric]] with mean $M$. A tight envelope matters. In high dimensions it is usually impossible to find one: the volume between $Mg$ and $f$ grows exponentially with the dimension, and almost every proposal is rejected. This is the motivation for the Markov chain Monte Carlo methods in [[Markov Chains#Aside: Markov chain Monte Carlo|Markov Chains]], which give up independent samples in exchange for scaling to high dimensions.

![[Probability and Linear Algebra/images/random-variables-rejection.svg]]
*Figure 13. Rejection sampling for a two-bump density on $[0,1]$ with a uniform proposal. Points are scattered uniformly under the envelope $Mg$ (dashed); those under the target $f$ are kept (orange). Their $x$-coordinates are samples from $f$. The fraction kept is $1/M\approx0.56$.*

> [!example]- Code: rejection sampling
> ```python
> import numpy as np
> from scipy import stats
> rng = np.random.default_rng(0)
>
> def f(x):                                  # target: a two-bump density on [0, 1]
>     return 0.6 * stats.beta(2, 6).pdf(x) + 0.4 * stats.beta(8, 3).pdf(x)
>
> def rejection_sample(n, M):
>     out, proposals = [], 0
>     while len(out) < n:
>         x = rng.random()                   # propose from g = Uniform(0, 1)
>         proposals += 1
>         if rng.random() * M <= f(x):       # accept with probability f(x) / (M g(x))
>             out.append(x)
>     return np.array(out), n / proposals
>
> M = 1.05 * f(np.linspace(0, 1, 1001)).max()
> s, acc = rejection_sample(20_000, M)
> print(round(M, 3), round(acc, 3))          # acceptance rate is about 1/M
> ```

## The Poisson process

So far we have looked at single random variables. A **random process** is a whole collection of them, indexed by time. The simplest and most important one for events in continuous time (arrivals, clicks, failures, spikes) is the Poisson process. Let $N(t)$ be the number of events in $[0,t]$.

> [!note] Definition (Poisson process with rate $\lambda$)
> $N(0)=0$, and:
> 1. **Independent increments:** counts in disjoint time intervals are independent.
> 2. **Stationary increments:** the distribution of the count in an interval depends only on its length.
> 3. **Events happen one at a time:** in a short interval of length $\delta$, $\mathbb{P}(\text{one event})=\lambda\delta+o(\delta)$ and $\mathbb{P}(\text{two or more})=o(\delta)$.

These assumptions are all we need. Everything else follows:[^kumar2012]<sup>,</sup>[^bertsekas2008]
- **Counts are Poisson.** Chop $[0,t]$ into $n$ tiny slots. Each slot has an event with probability about $\lambda t/n$, independently, so $N(t)$ is approximately Binomial$(n,\lambda t/n)$. As $n\to\infty$, the [[#Poisson: the law of rare events|law of rare events]] gives $N(t)\sim\text{Poisson}(\lambda t)$.
- **Gaps are exponential.** The first event comes after time $t$ exactly when there are no events in $[0,t]$: $\mathbb{P}(T_1>t)=\mathbb{P}(N(t)=0)=e^{-\lambda t}$. By independent and stationary increments, the process starts afresh after every event, so the gaps between consecutive events are i.i.d. Exponential$(\lambda)$.
- **The converse construction.** Adding up i.i.d. Exponential$(\lambda)$ gaps produces a Poisson process. This is the easiest way to simulate one (code below). The time of the $k$-th event, a sum of $k$ exponentials, has the Gamma (Erlang) distribution.

So counts and gaps are two views of the same process (Figure 14). Two more properties are very useful:
- **Conditional uniformity:** given that $N(t)=k$, the $k$ event times are distributed like $k$ independent uniform points on $[0,t]$, sorted.
- **Thinning and superposition:** if each event is kept independently with probability $p$, the kept events form a Poisson process with rate $p\lambda$, and the discarded ones an independent Poisson process with rate $(1-p)\lambda$. Merging independent Poisson processes adds their rates. Both fail if the selection is not independent: keeping every second event gives gaps that are sums of two exponentials, not a Poisson process.

![[Probability and Linear Algebra/images/random-variables-poisson-process.svg]]
*Figure 14. A Poisson process with rate $\lambda=2$. (a) One path of the count $N(t)$: it jumps by 1 at each event (ticks). (b) The gaps between events follow the Exponential(2) density. (c) The number of events per unit of time, built only from exponential gaps, follows the Poisson(2) distribution.*

> [!example]- Aside: the waiting-time paradox
> Buses arrive as a Poisson process, on average every 10 minutes. You arrive at the stop at a random time. How long do you wait on average? Not 5 minutes but **10**: by memorylessness, the time to the next bus is a fresh Exponential with mean 10, whatever has happened before. And the time since the *previous* bus is also 10 minutes on average. So the gap you landed in has mean 20 minutes, twice the average gap. You are more likely to arrive during a long gap than a short one, simply because long gaps take up more of the time. This **length bias** (or inspection paradox) also explains why your friends have more friends than you do on average, and why a randomly chosen class is usually larger than the average class.

> [!example]- Code: simulating a Poisson process
> ```python
> import numpy as np
> rng = np.random.default_rng(0)
>
> def poisson_process(lam, T):
>     """Arrival times on [0, T]: add up exponential gaps."""
>     t, times = 0.0, []
>     while True:
>         t += rng.exponential(1 / lam)
>         if t > T:
>             return np.array(times)
>         times.append(t)
>
> counts = [len(poisson_process(2.0, 5.0)) for _ in range(20_000)]
> print(np.mean(counts), np.var(counts))     # both about lam * T = 10
> ```

**Time-varying rates.** If the rate changes in time, a deterministic function $\lambda(t)$, we get an **inhomogeneous** Poisson process: counts in disjoint intervals are still independent, and the count in $[a,b]$ is Poisson with mean $\int_a^b\lambda(t)\,dt$. A neat way to simulate it is **thinning**:[^lewis1979] simulate a homogeneous process at a constant rate $\bar\lambda\ge\lambda(t)$, and keep an event at time $t$ with probability $\lambda(t)/\bar\lambda$. This is rejection sampling in time, with the constant rate $\bar\lambda$ as the envelope.

## Hawkes processes: self-exciting events

In a Poisson process, events do not influence each other. Many real event streams are **self-exciting**:
- an earthquake triggers aftershocks;
- a trade on a stock exchange triggers more trades;
- a viral post triggers reshares, which trigger more reshares;
- a neuron's spike changes the firing of its neighbours.

The counts in such streams are burstier than Poisson: their variance exceeds their mean. Kumar notes that the variance-to-mean ratio of 1 is the Poisson benchmark against which burstiness is measured.[^kumar2012]

The tool to describe this is the **conditional intensity**: the event rate at time $t$ given the history $\mathcal{H}_t$ of events before $t$,

$$
\lambda^*(t)=\lim_{\delta\to0}\frac{\mathbb{P}\big(\text{event in }[t,t+\delta)\mid\mathcal{H}_t\big)}{\delta}.
$$

A Poisson process has $\lambda^*(t)=\lambda$, ignoring the history. In a **Hawkes process**,[^hawkes1971] each past event at time $t_i$ adds a bump to the current rate:

$$
\lambda^*(t)=\mu+\sum_{t_i<t}\phi(t-t_i),\qquad\text{typically}\quad\phi(s)=\alpha\,e^{-\beta s}.
\tag{8}
$$

Here $\mu$ is the **baseline** rate of spontaneous events, and each event raises the intensity by $\alpha$, an excitation that decays at rate $\beta$. Events beget events, so they arrive in clusters (Figure 15).

![[Probability and Linear Algebra/images/random-variables-hawkes.svg]]
*Figure 15. (a) A Hawkes process with $\mu=0.5$, $\alpha=0.8$ and $\beta=1.2$. The intensity (blue) jumps at each event (ticks) and decays back towards the baseline, so events cluster. (b) A Poisson process with the same long-run rate $\mu/(1-\alpha/\beta)=1.5$: the same average number of events, spread evenly.*

**The branching view.** A Hawkes process is a family tree. "Immigrant" events arrive as a Poisson process with rate $\mu$. Every event, immigrant or not, independently gives birth to a Poisson number of "children", at delays drawn from the density $\phi/n^*$. The mean number of children per event is the **branching ratio**

$$
n^*=\int_0^\infty\phi(s)\,ds=\frac{\alpha}{\beta}.
$$

- If $n^*<1$, each family dies out, the process is stationary, and each immigrant founds a cluster of $1+n^*+n^{*2}+\cdots=1/(1-n^*)$ events on average. The long-run rate is therefore $\mu/(1-n^*)$. Taking expectations in (8) gives the same answer: the mean rate $\bar\lambda$ satisfies $\bar\lambda=\mu+n^*\bar\lambda$.
- If $n^*\ge1$, the families can grow forever and the process explodes.

For earthquakes, Ogata's ETAS model fits exactly this structure.[^ogata1988] Laub, Taimre and Pollett give a gentle introduction to Hawkes processes.[^laub2015]

### Simulating a Hawkes process: Ogata's thinning

We cannot simulate a Hawkes process by adding exponential gaps, because the rate changes after every event. Instead we use thinning, as for the inhomogeneous Poisson process. The difficulty is that we need an upper bound on the intensity, and the intensity depends on events not yet simulated. Ogata's observation is that with a decaying kernel the intensity can only **decrease** until the next event.[^ogata1981] So its current value is a valid upper bound, until we accept the next event.

> [!example] Algorithm: Ogata's thinning for a Hawkes process
> Start at $t=0$ with no events.
> 1. Set $\bar\lambda=\lambda^*(t)$, the current intensity. It bounds the intensity until the next event.
> 2. Draw a candidate gap $W\sim\text{Exponential}(\bar\lambda)$ and move to $t\leftarrow t+W$. If $t>T$, stop.
> 3. With probability $\lambda^*(t)/\bar\lambda$, accept: record an event at $t$, which raises the intensity by $\alpha$. Otherwise reject the candidate.
> 4. Go to step 1.

Step 3 is rejection sampling: candidates come from a Poisson process with the envelope rate $\bar\lambda$, and each is kept with probability equal to the ratio of the true rate to the envelope. Rejected candidates are not wasted. Moving time forward to them lets the intensity decay, which tightens the bound for the next round.

> [!example]- Code: Hawkes process by Ogata thinning
> ```python
> import numpy as np
> rng = np.random.default_rng(0)
>
> def hawkes(mu, alpha, beta, T):
>     """Events of a Hawkes process with intensity mu + sum_i alpha*exp(-beta*(t - t_i))."""
>     t, events, excite = 0.0, [], 0.0       # excite = self-excitation at the current time
>     while True:
>         lam_bar = mu + excite              # an upper bound: the intensity only decays until the next event
>         w = rng.exponential(1 / lam_bar)   # candidate from a Poisson process with rate lam_bar
>         t += w
>         excite *= np.exp(-beta * w)        # decay the excitation to the candidate time
>         if t > T:
>             return np.array(events)
>         if rng.random() * lam_bar <= mu + excite:   # keep it with probability lambda*(t) / lam_bar
>             events.append(t)
>             excite += alpha                # the new event raises the intensity
>
> ev = hawkes(mu=0.5, alpha=0.8, beta=1.2, T=20_000)
> print(len(ev) / 20_000, 0.5 / (1 - 0.8 / 1.2))   # empirical rate vs mu / (1 - alpha/beta) = 1.5
> ```
>
> The exponential kernel makes the excitation a single number that decays by a factor $e^{-\beta w}$, so each step costs $O(1)$. The implementation was checked against the time-rescaling theorem: the compensator $\int\lambda^*$ evaluated between consecutive events gives i.i.d. Exponential(1) values.

## Key takeaways

- A random variable is a function on outcomes; probabilities of statements about it are probabilities of their pre-images.
- The CDF always exists and determines the distribution. The PMF is its jumps, the PDF its slope. Densities can exceed 1; only areas are probabilities.
- The mean is the balance point and minimizes squared error; the median minimizes absolute error. LOTUS computes $\mathbb{E}[g(X)]$ directly, and for $X\ge0$ the mean is the area above the CDF. The mean need not exist.
- The law of total expectation averages case averages. First-step analysis uses it recursively.
- Markov's inequality bounds tails from the mean, and Chebyshev's from the variance. Both are loose but universal.
- Bernoulli trials add to a binomial, which tends to a Poisson for rare events. Geometric and exponential waiting times are memoryless, and the exponential is the constant-hazard distribution. Hazard rates describe how risk changes with age.
- The CDF method gives the density of $g(X)$, with the derivative factor accounting for stretching and squeezing. Inverse-CDF and rejection sampling turn uniforms into any distribution.
- A Poisson process has Poisson counts and exponential gaps. A Hawkes process adds self-excitation, has branching ratio $\alpha/\beta$, and is simulated by Ogata's thinning.

## Where this shows up next

- [[Joint Distributions]] studies several random variables together: covariance, conditional expectation as a random variable, and the multivariate Gaussian.
- [[Sample Statistics]] proves that sample means converge to expectations and finds the Gaussian in the fluctuations. The exponential tail bounds there improve on Markov and Chebyshev.
- [[Markov Chains]] uses first-step analysis for hitting times, and replaces rejection sampling with MCMC in high dimensions.
- [[Score-Based Models]] samples with Langevin dynamics, another alternative to rejection sampling. [[Variational Autoencoders]] uses LOTUS and change of variables in the reparameterization trick.
- [[Multi-Armed Bandits]] and [[Introduction to Reinforcement Learning]] average random rewards; the bandits' regret proof starts from Markov-type tail bounds.

<!-- References: footnote definitions below render as the numbered "References" list at the bottom of the page -->

[^kumar2012]: Kumar, A. (2012). *Discrete Event Stochastic Processes: Lecture Notes for an Engineering Curriculum*, Sections 1.2 and 3.4. Indian Institute of Science. [PDF](https://ece.iisc.ac.in/~anurag/books/anurag/spqt.pdf)
[^chan2026]: Chan, S. H. (2026). *Introduction to Probability for Data Science* (2nd ed.), Chapters 3–4. Michigan Publishing. [book website](https://probability4datascience.com)
[^blitzstein2019]: Blitzstein, J. K., & Hwang, J. (2019). *Introduction to Probability* (2nd ed.), Chapter 4. CRC Press. [Free online edition](https://probabilitybook.net)
[^kalbfleisch2002]: Kalbfleisch, J. D., & Prentice, R. L. (2002). *The Statistical Analysis of Failure Time Data* (2nd ed.). Wiley. [doi:10.1002/9781118032985](https://doi.org/10.1002/9781118032985)
[^devroye1986]: Devroye, L. (1986). *Non-Uniform Random Variate Generation*. Springer. [Free online edition](https://luc.devroye.org/rnbookindex.html)
[^bertsekas2008]: Bertsekas, D. P., & Tsitsiklis, J. N. (2008). *Introduction to Probability* (2nd ed.), Chapter 6 (The Bernoulli and Poisson processes). Athena Scientific. Lectures: [MIT OCW 6.041](https://ocw.mit.edu/courses/6-041-probabilistic-systems-analysis-and-applied-probability-fall-2010/)
[^lewis1979]: Lewis, P. A. W., & Shedler, G. S. (1979). *Simulation of nonhomogeneous Poisson processes by thinning*. Naval Research Logistics Quarterly, 26(3), 403–413. [doi:10.1002/nav.3800260304](https://doi.org/10.1002/nav.3800260304)
[^hawkes1971]: Hawkes, A. G. (1971). *Spectra of some self-exciting and mutually exciting point processes*. Biometrika, 58(1), 83–90. [doi:10.1093/biomet/58.1.83](https://doi.org/10.1093/biomet/58.1.83)
[^ogata1988]: Ogata, Y. (1988). *Statistical Models for Earthquake Occurrences and Residual Analysis for Point Processes*. Journal of the American Statistical Association, 83(401), 9–27. [doi:10.1080/01621459.1988.10478560](https://doi.org/10.1080/01621459.1988.10478560)
[^laub2015]: Laub, P. J., Taimre, T., & Pollett, P. K. (2015). *Hawkes Processes*. [arXiv:1507.02822](https://arxiv.org/abs/1507.02822)
[^ogata1981]: Ogata, Y. (1981). *On Lewis' simulation method for point processes*. IEEE Transactions on Information Theory, 27(1), 23–31. [doi:10.1109/TIT.1981.1056305](https://doi.org/10.1109/TIT.1981.1056305)
