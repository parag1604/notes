---
title: Stochastic Approximation
tags:
  - reinforcement-learning
  - stochastic-approximation
  - ode-method
  - convergence
  - theory
date: 2026-11-12
draft: false
---

> **Prerequisites:** [[Multi-Armed Bandits]] (the incremental update $q\leftarrow q+\alpha(r-q)$), [[Introduction to Reinforcement Learning]] (the Bellman equations and operators), [[Sample Statistics]] (almost-sure convergence, the strong law), and [[Markov Chains]] (stationary distributions). The contraction property of the Bellman operators is proved in [[Dynamic Programming]].
>
> **TL;DR:**
> - TD learning, Q-learning, linear TD, SGD and actor-critic all have the same shape: $x_{n+1}=x_n+\alpha_n\big[h(x_n)+M_{n+1}\big]$, a noisy step towards where the "mean dynamics" $h$ point. This is a **stochastic approximation** (Robbins–Monro) scheme.
> - With **Robbins–Monro step sizes** ($\sum\alpha_n=\infty$, $\sum\alpha_n^2<\infty$), **martingale-difference noise** and **bounded iterates**, the iterates asymptotically follow the **ODE** $\dot x=h(x)$. If that ODE has a globally asymptotically stable equilibrium $x^*$, then $x_n\to x^*$ almost surely.
> - Boundedness can be checked with the **Borkar–Meyn** scaled ODE. Stability of the ODE is checked with a **Lyapunov function**: for $\dot x=F(x)-x$ with $F$ a contraction, $\lVert x-x^*\rVert$ itself works.
> - Extensions cover **asynchronous** updates (Tsitsiklis's theorem, the setting of Q-learning), **Markov noise** (TD along a trajectory), **two timescales** (actor-critic) and **constant step sizes** (tracking, with a bias–variance trade-off).

**Notation for this lecture.** The iteration counter is $n$, because $t$ is reserved for the *continuous* time of the ODE. In the RL lectures one iteration is usually one environment step, so $n$ there becomes $t$. The step size is $\alpha_n$ as in the rest of the series. A Lyapunov function is written $\Lambda$ (not $V$, which is a value function), and $\gamma$ is both the discount and the contraction modulus, since for Bellman operators they coincide.

## Motivation

This is the analysis lecture behind Part I. The model-free lectures state convergence results such as "TD(0) converges to $V^\pi$" or "Q-learning converges to $Q^*$ if every pair is visited infinitely often and the step sizes satisfy the Robbins–Monro conditions". Here is where those conditions come from.

Start with the simplest problem from [[Multi-Armed Bandits]]: estimate the mean $\mu$ of a random variable from samples $r_1,r_2,\dots$ that arrive one at a time. The running average can be computed incrementally:

$$
x_{n+1}=x_n+\frac{1}{n+1}\big(r_{n+1}-x_n\big).
$$

Write $r_{n+1}=\mu+M_{n+1}$, where $M_{n+1}$ is zero-mean noise. Then

$$
x_{n+1}=x_n+\alpha_n\big[\underbrace{(\mu-x_n)}_{h(x_n)}+M_{n+1}\big],\qquad \alpha_n=\frac{1}{n+1}.
$$

Each step moves $x_n$ a little in the direction $h(x_n)=\mu-x_n$, which points towards the answer, corrupted by noise. Now replace "estimate a mean" by "solve an equation whose terms we can only sample":

- **Root finding with noise** (the original problem of Robbins and Monro[^robbinsmonro1951]). Find $x^*$ with $g(x^*)=0$ when we can only observe $g(x)+\text{noise}$, for example the dose $x$ at which a drug has a target effect. The iteration is $x_{n+1}=x_n-\alpha_n\,[g(x_n)+M_{n+1}]$.
- **Policy evaluation.** $V^\pi$ solves $V=\mathcal{T}^\pi V$, but we cannot compute the expectation inside $\mathcal{T}^\pi$. We can sample one transition and move towards $r+\gamma V(s')$. That is TD(0).
- **Q-learning.** $Q^*$ solves $Q=\mathcal{T}^*Q$. Sample a transition and move towards $r+\gamma\max_{a'}Q(s',a')$.
- **Stochastic gradient descent.** Find a zero of $\nabla f$ from unbiased gradient samples.

All of them are instances of one iteration. Proving convergence once, for the general iteration, proves it for all of them.

---

## The Robbins–Monro iteration

The general **stochastic approximation** (SA) scheme is

$$
x_{n+1}=x_n+\alpha_n\big[h(x_n)+M_{n+1}\big],\qquad x_n\in\mathbb{R}^d. \tag{1}
$$

- $h:\mathbb{R}^d\to\mathbb{R}^d$ is the **mean field** (or drift): the direction the update points in *on average*.
- $M_{n+1}$ is **noise** with conditional mean zero (made precise in the next section).
- $\alpha_n>0$ is the **step size**.

We look for the zeros of $h$. For mean estimation $h(x)=\mu-x$. For TD(0), $h(V)=\mathcal{T}^\pi V-V$. For Q-learning, $h(Q)=\mathcal{T}^*Q-Q$. For SGD, $h=-\nabla f$.

### Step sizes: why $\sum\alpha_n=\infty$ and $\sum\alpha_n^2<\infty$

The classical conditions, already used informally in [[Multi-Armed Bandits]], are

$$
\alpha_n>0,\qquad \sum_{n=0}^\infty\alpha_n=\infty,\qquad \sum_{n=0}^\infty\alpha_n^2<\infty. \tag{2}
$$

Mean estimation shows exactly what each condition buys. The error $e_n=x_n-\mu$ obeys $e_{n+1}=(1-\alpha_n)e_n+\alpha_nM_{n+1}$. Unrolling,

$$
\begin{aligned}
e_n&=\underbrace{\prod_{k=0}^{n-1}(1-\alpha_k)\,e_0}_{\text{initial error}}\\
&\quad+\underbrace{\sum_{k=0}^{n-1}\alpha_k\prod_{j=k+1}^{n-1}(1-\alpha_j)\,M_{k+1}}_{\text{accumulated noise}}.
\end{aligned}
$$

- **$\sum\alpha_n=\infty$ forgets the initial condition.** Since $1-\alpha\le e^{-\alpha}$, the first term is at most $e^{-\sum_k\alpha_k}|e_0|$, which vanishes when the steps sum to infinity. If $\sum\alpha_n<\infty$ (and $\alpha_n<1$), the product stays bounded away from zero: the iterates run out of "distance budget" and stop short of the target. In general the total distance the iterates can travel is $\sum\alpha_n$ times the size of $h$, so the condition is needed whenever we start far from $x^*$.
- **$\sum\alpha_n^2<\infty$ averages out the noise.** The noise terms are uncorrelated (shown below), so the variance of the second term is $\sum_k\alpha_k^2\prod_{j>k}(1-\alpha_j)^2\operatorname{Var}(M_{k+1})$. Square-summable steps keep the total injected variance finite. Combined with the forgetting factor, its effect dies out.

> [!tip] Why $1/(n+1)$ gives exactly the sample mean
> With $\alpha_k=\frac{1}{k+1}$ the products telescope: $\prod_{j=k+1}^{n-1}\frac{j}{j+1}=\frac{k+1}{n}$. So the weight on $M_{k+1}$ is $\alpha_k\cdot\frac{k+1}{n}=\frac1n$ for every $k$, and $\prod_{k=0}^{n-1}(1-\alpha_k)=0$ because $\alpha_0=1$. The iterate is $x_n=\frac1n\sum_{k=1}^n r_k$, the sample mean, with variance $\sigma^2/n$. Every sample gets the same weight, which is the best one can do for i.i.d. data.

Common schedules that satisfy (2) are

$$
\begin{aligned}
&\alpha_n=\frac{1}{n+1},&&\alpha_n=\frac{1}{(n+1)^{\kappa}}\ \ \big(\tfrac12<\kappa\le1\big),\\
&\alpha_n=\frac{\log(n+2)}{n+2},&&\alpha_n=\frac{1}{(n+2)\log(n+2)},
\end{aligned}
$$

and $\alpha_n=\frac{b}{a+n}$ with $a,b>0$. The last two on the first display are borderline cases: $\sum\frac{\log n}{n}$ and $\sum\frac{1}{n\log n}$ both diverge (compare with $\int\frac{dx}{x\log x}=\log\log x$), while the sums of their squares converge. In RL, $\alpha$ often depends on how many times a state (or state-action pair) has been visited, $\alpha=1/N(s,a)$; this is a per-component version of the same idea.

![[Reinforcement Learning/images/stochastic-approximation-exponents.svg]]
*Figure 1: estimating a mean ($\sigma=1$, $x_0=5$) with $\alpha_n=(n+1)^{-\kappa}$, RMS error over 2000 runs on log–log axes. $\kappa=1$ is the sample mean and follows $1/\sqrt n$ (error $0.0098$ at $n\approx10^4$). For $\kappa<1$ the error behaves like $\sqrt{\alpha_n/2}$: $0.045$ for $\kappa=0.6$ and $0.018$ for $\kappa=0.8$. $\kappa=1.5$ has $\sum\alpha_n<\infty$ and freezes at an error of about $0.24$. A constant $\alpha=0.05$ plateaus at $\sqrt{\alpha/(2-\alpha)}\approx0.16$. Note that $\kappa=0.3$ also converges here (error $0.19$ and still falling), although $\sum\alpha_n^2=\infty$.*

The last observation is worth stating honestly. The conditions (2) are **sufficient**, not necessary. They make the martingale argument below clean. For a linear problem with well-behaved noise, any $\alpha_n\to0$ with $\sum\alpha_n=\infty$ gives convergence in mean square, and sharper theorems give almost-sure convergence for slowly decaying steps under stronger noise assumptions.[^kushneryin2003] In practice, $\kappa\in(\tfrac12,1)$ with **iterate averaging** is a robust choice (see [[#Constant step sizes: tracking and the bias–variance trade-off|constant step sizes]]).

---

## Martingale-difference noise and filtrations

The noise in RL is not i.i.d. In TD learning the sampled target depends on the current estimate, and in Q-learning on the current $\max_{a'}Q$. What we need is weaker than independence: the noise must be **unpredictable from the past**.

**Filtration.** Let $\mathcal{F}_n=\sigma(x_0,M_1,\dots,M_n)$ be the information available after $n$ steps: the initial point and all noise so far, hence also $x_1,\dots,x_n$. The sequence $\mathcal{F}_0\subseteq\mathcal{F}_1\subseteq\cdots$ is a **filtration**: information only accumulates. An event is in $\mathcal{F}_n$ if we can tell whether it happened by time $n$. For example, $\{\lVert x_3\rVert>5\}\in\mathcal{F}_n$ for $n\ge3$, but $\{\lVert x_{n+1}\rVert<10\}\notin\mathcal{F}_n$ and $\{\lim_n x_n\text{ exists}\}$ is in no $\mathcal{F}_n$.

**Martingale-difference sequence (MDS).** $(M_n)$ is an MDS with respect to $(\mathcal{F}_n)$ if each $M_n$ is $\mathcal{F}_n$-measurable, integrable, and

$$
\mathbb{E}\big[M_{n+1}\mid\mathcal{F}_n\big]=0. \tag{3}
$$

Equivalently, $\mathbb{E}[M_{n+1}\mathbb{1}_A]=0$ for every event $A\in\mathcal{F}_n$: no pattern in the past predicts the sign of the next noise term. Independent zero-mean noise is the special case where the past is irrelevant. The partial sums of an MDS form a **martingale**, a "fair game" whose expected next value is its current value.

**How the noise is built in RL.** Take any update $x_{n+1}=x_n+\alpha_n f(x_n,\xi_{n+1})$, where $\xi_{n+1}$ is the fresh sample (a reward, a next state). Add and subtract the conditional mean:

$$
\begin{aligned}
f(x_n,\xi_{n+1})&=\underbrace{\mathbb{E}\big[f(x_n,\xi_{n+1})\mid\mathcal{F}_n\big]}_{h(x_n)}\\
&\quad+\underbrace{f(x_n,\xi_{n+1})-\mathbb{E}\big[f(x_n,\xi_{n+1})\mid\mathcal{F}_n\big]}_{M_{n+1}}.
\end{aligned}
$$

By construction $\mathbb{E}[M_{n+1}\mid\mathcal{F}_n]=0$. The conditional expectation is a *random variable* (a function of $x_n$), and $h$ is obtained by computing it as if $x_n$ were a fixed number. For synchronous Q-learning, for instance, $h(Q)=\mathcal{T}^*Q-Q$ and $M_{n+1}(s,a)=r(s,a,s')+\gamma\max_{a'}Q_n(s',a')-(\mathcal{T}^*Q_n)(s,a)$ with $s'\sim p(\cdot\mid s,a)$.

**Why an MDS is harmless.** Increments of a martingale are uncorrelated: for $j<k$,

$$
\mathbb{E}\big[M_{j+1}M_{k+1}\big]=\mathbb{E}\big[M_{j+1}\,\mathbb{E}[M_{k+1}\mid\mathcal{F}_k]\big]=0,
$$

since $M_{j+1}$ is known at time $k$. So variances add, as for independent noise, and the weighted noise sum converges.

> [!note] Lemma (the noise sums up)
> Suppose $\mathbb{E}[\lVert M_{n+1}\rVert^2\mid\mathcal{F}_n]\le K\big(1+\lVert x_n\rVert^2\big)$, $\sup_n\lVert x_n\rVert<\infty$ almost surely, and $\sum\alpha_n^2<\infty$. Then $Z_n=\sum_{k=0}^{n-1}\alpha_kM_{k+1}$ converges almost surely. In particular the tail sums $\sum_{k\ge n}\alpha_kM_{k+1}$ tend to $0$.

> [!proof]- Proof (bounded-variance case, and how to remove the restriction)
> First assume $\mathbb{E}[\lVert M_{n+1}\rVert^2\mid\mathcal{F}_n]\le\sigma^2$. $Z_n$ is a martingale, and by the orthogonality of increments
> $$
> \mathbb{E}\lVert Z_n\rVert^2=\sum_{k=0}^{n-1}\alpha_k^2\,\mathbb{E}\lVert M_{k+1}\rVert^2\le\sigma^2\sum_{k=0}^{\infty}\alpha_k^2<\infty.
> $$
> A martingale bounded in $L^2$ converges almost surely (and in $L^2$) by Doob's martingale convergence theorem.[^williams1991] Its tail sums therefore vanish.
>
> In general, fix $C>0$ and stop the sum the first time $\lVert x_n\rVert>C$. Up to that time the conditional variance is at most $K(1+C^2)$, so the stopped martingale converges by the first part. On the event $\{\sup_n\lVert x_n\rVert\le C\}$ the stopped and unstopped sums coincide. Letting $C\to\infty$ through the integers covers the event $\{\sup_n\lVert x_n\rVert<\infty\}$, which has probability one by assumption.[^borkar2008]

The strong law of large numbers is the special case $h(x)=\mu-x$, $\alpha_n=\frac{1}{n+1}$ (see [[Sample Statistics#The strong law]]).

**Markov noise.** In TD learning along a single trajectory, the conditional mean of the update depends on the current state too: $\mathbb{E}[f\mid\mathcal{F}_n]=h(x_n,s_n)$. The state $s_n$ is not noise that averages out in one step; it is a Markov chain that averages out over *time*, through its stationary distribution (see [[Markov Chains#Convergence and the ergodic theorem]]). The [[#Markov noise|Markov-noise theorem]] below handles this.

---

## The ODE method

**The Euler picture.** Drop the noise from (1): $x_{n+1}=x_n+\alpha_n h(x_n)$. That is exactly the **Euler method** for the ordinary differential equation

$$
\dot x(t)=h\big(x(t)\big), \tag{4}
$$

with a time step $\alpha_n$ that changes from step to step. So place the $n$-th iterate at the **algorithmic time**

$$
t_0=0,\qquad t_n=\sum_{m=0}^{n-1}\alpha_m,
$$

and join consecutive iterates by straight lines to get a continuous curve $\bar x(t)$. Because $\sum\alpha_n=\infty$, algorithmic time runs to infinity. Because $\alpha_n\to0$, the grid gets finer and finer.

SA differs from a textbook Euler scheme in two ways:

| | Euler method | Stochastic approximation |
|---|---|---|
| time step | constant $\Delta t$ (error shrinks as $\Delta t\to0$) | $\alpha_n\to0$ along the run |
| drift | exact $h(x_n)$ | noisy $h(x_n)+M_{n+1}$ |
| what goes to zero | discretization error, as $\Delta t\to0$ | discretization error *and* accumulated noise, as $n\to\infty$ |

On a window $[t_n,t_n+T]$ of fixed algorithmic length $T$, both differences become negligible as $n\to\infty$. The discretization error is of order $\sum_{k\ge n}\alpha_k^2$, and the noise contributes the tail of the convergent sum $Z_n$. So **late in the run, the interpolated iterates shadow the ODE solution over any fixed time horizon**. Long-run behaviour of the algorithm is then the long-run behaviour of the ODE. This is the **ODE method**, introduced by Ljung and Kushner in the 1970s and developed into its modern dynamical-systems form by Benaïm and Borkar.[^benaim1999]<sup>,</sup>[^borkar2008]

![[Reinforcement Learning/images/stochastic-approximation-interpolation.svg]]
*Figure 2: the ODE method on the mean-estimation problem ($\mu=1$, $x_0=4$, $\alpha_n=0.5/(n+1)^{0.7}$). Top: the iterates plotted at algorithmic times $t_n$; on each window $[k,k+1]$ the ODE $\dot x=\mu-x$ is restarted from the current iterate. Bottom: the largest gap between the iterates and the restarted ODE on each window (300 runs). The median gap falls from $0.55$ on the first window to $0.12$ on window 5 and $0.06$ on window 11, which took about 1000 iterations.*

![[Reinforcement Learning/images/stochastic-approximation-phase.svg]]
*Figure 3: a two-dimensional example, $h(x)=-Ax$ with $A=\begin{pmatrix}0.5&-1\\1&0.5\end{pmatrix}$ (a stable spiral) and unit Gaussian noise. Left: the ODE trajectory (black) and SA iterates under three schedules, all started at $(3,3)$. Right: distance to $x^*=0$ against algorithmic time. All three follow the spiral. Reaching $t=8$ takes about 13 400 iterations with $\alpha_n=1/(n+5)$ but only 138 with $0.5/(n+1)^{0.6}$. The constant step size keeps jittering around $x^*$ instead of settling.*

The figure also shows the practical trade-off. Slowly decaying steps cover algorithmic time quickly, so they forget the initial condition fast, but they average noise more slowly.

---

## Convergence theorems

### The basic theorem

> [!note] Theorem (convergence by the ODE method)
> Consider iteration (1) and assume:
> 1. **(Lipschitz drift)** $h$ is Lipschitz: $\lVert h(x)-h(y)\rVert\le L\lVert x-y\rVert$.
> 2. **(Step sizes)** $\alpha_n>0$, $\sum_n\alpha_n=\infty$, $\sum_n\alpha_n^2<\infty$.
> 3. **(Noise)** $(M_n)$ is a martingale-difference sequence with respect to $\mathcal{F}_n=\sigma(x_0,M_1,\dots,M_n)$, and for some $K$, $\mathbb{E}\big[\lVert M_{n+1}\rVert^2\mid\mathcal{F}_n\big]\le K\big(1+\lVert x_n\rVert^2\big)$.
> 4. **(Stability)** $\sup_n\lVert x_n\rVert<\infty$ almost surely.
>
> Then, almost surely, $x_n$ converges to a (possibly sample-path dependent) compact, connected, internally chain transitive, invariant set of the ODE $\dot x=h(x)$. In particular, **if the ODE has a unique globally asymptotically stable equilibrium $x^*$, then $x_n\to x^*$ almost surely.**

This is Borkar's formulation (Chapter 2 of his book).[^borkar2008] A few comments:

- **Lipschitz** guarantees that the ODE has unique solutions and that small perturbations of the starting point cause small changes over a window of length $T$ (by Grönwall, at most a factor $e^{LT}$).
- **The noise bound** is weak: noise may grow with the iterate, as it does in TD and Q-learning, where the sampled target contains $\gamma V(s')$ or $\gamma\max_{a'}Q(s',a')$.
- **Stability is an assumption, not a conclusion.** It must be proved separately for each algorithm. The [[#Stability: the Borkar–Meyn test|Borkar–Meyn test]] below is the standard tool.
- **"Internally chain transitive invariant set"** is the precise name for the sets where the ODE can keep "circulating" when tiny jumps are allowed: equilibria, limit cycles, and similar. If the only such set is one globally attracting point, the iterates converge to it. If the ODE has a limit cycle, for example $\dot r=1-r$, $\dot\theta=c$ in polar coordinates, whose solutions all wind onto the unit circle, the iterates converge to the circle but not to a point. For SGD the limit sets are connected sets of stationary points of $f$, which is why SGD converges to *a* stationary point.

> [!proof]- Proof sketch
> 1. **Shadowing.** Let $x^s(\cdot)$ solve the ODE on $[s,s+T]$ starting from $\bar x(s)$. Compare $\bar x$ and $x^s$ at the grid points: unrolling (1) gives $x_{m}=x_n+\sum_{k=n}^{m-1}\alpha_k h(x_k)+\sum_{k=n}^{m-1}\alpha_kM_{k+1}$, while the ODE gives $x^s(t_m)=x^s(t_n)+\int_{t_n}^{t_m}h(x^s(u))\,du$.
> 2. **Three error sources.** The difference is bounded by (a) a Lipschitz term $L\sum\alpha_k\lVert x_k-x^s(t_k)\rVert$, (b) a discretization term of order $\sum_{k\ge n}\alpha_k^2$ (the iterates are bounded, so $h$ is bounded along the path), and (c) the noise term $\sup_m\lVert Z_m-Z_n\rVert$. Terms (b) and (c) tend to zero, by $\sum\alpha_k^2<\infty$ and by the noise lemma.
> 3. **Grönwall.** The discrete Grönwall inequality turns (a) into a factor $e^{LT}$. Hence $\sup_{t\in[s,s+T]}\lVert\bar x(t)-x^s(t)\rVert\to0$ as $s\to\infty$, almost surely, for every $T$: $\bar x$ is an *asymptotic pseudo-trajectory* of the ODE.[^benaim1999]
> 4. **Limit sets.** The limit set of a bounded asymptotic pseudo-trajectory is internally chain transitive and invariant (Benaïm's theorem). If $x^*$ is globally asymptotically stable, the only such set is $\{x^*\}$: a trajectory that starts anywhere enters any neighbourhood of $x^*$ within a bounded time and stays there, and the pseudo-trajectory inherits this property.[^borkar2008]

### Stability: the Borkar–Meyn test

Assumption 4 is the hard one. Iterates could, in principle, drift off to infinity before the ODE has a chance to pull them back. Borkar and Meyn found a test that uses only the behaviour of $h$ **far away from the origin**.[^borkarmeyn2000] Zoom out by a factor $c$:

$$
h_c(x)=\frac{h(cx)}{c},\qquad h_\infty(x)=\lim_{c\to\infty}h_c(x).
$$

$h_\infty$ is the "shape" of the drift at infinity. Constant offsets disappear in the limit, and only the part of $h$ that grows linearly survives.

> [!note] Theorem (Borkar–Meyn, 2000)
> Assume conditions 1–3 of the basic theorem. Suppose that $h_c\to h_\infty$ uniformly on compact sets as $c\to\infty$, and that the origin is the unique globally asymptotically stable equilibrium of $\dot x=h_\infty(x)$. Then $\sup_n\lVert x_n\rVert<\infty$ almost surely.

**Intuition.** When $\lVert x_n\rVert$ is huge, rescale it to the unit sphere. The rescaled iterates follow $\dot x=h_\infty(x)$, which pulls them back towards the origin, so a large iterate cannot keep growing. Two typical cases:

- **Linear drift**, $h(x)=b-Ax$ (linear TD): $h_\infty(x)=-Ax$. The test passes when all eigenvalues of $A$ have positive real part, for instance when $x^\top Ax>0$ for all $x\ne0$.
- **Contraction drift**, $h(x)=F(x)-x$ with $F$ a $\gamma$-contraction (tabular TD and Q-learning): $h_\infty(x)=F_\infty(x)-x$, where $F_\infty(x)=\lim_c F(cx)/c$, if the limit exists, is again a $\gamma$-contraction, with fixed point $0$. The [[#The contraction lemma|contraction lemma]] below shows that $0$ is globally asymptotically stable.

When the limit $h_\infty$ does not exist, a set-valued version using differential inclusions applies.[^ramaswamybhatnagar2017]

### Markov noise

In TD(0) along a trajectory, the update at time $n$ depends on the current state $Y_n=s_n$:

$$
x_{n+1}=x_n+\alpha_n\big[h(x_n,Y_n)+M_{n+1}\big].
$$

> [!note] Theorem (stochastic approximation with Markov noise)
> Assume:
> 1. $h(\cdot,y)$ is Lipschitz in $x$ with a constant that does not depend on $y$.
> 2. $(Y_n)$ is a Markov chain on a finite set whose transition probabilities do not depend on the iterates. It is irreducible, so it has a unique stationary distribution $\nu$.
> 3. $(M_n)$ is a martingale-difference sequence with respect to $\mathcal{F}_n=\sigma(x_m,Y_m,M_m:m\le n)$, with $\mathbb{E}[\lVert M_{n+1}\rVert^2\mid\mathcal{F}_n]\le K(1+\lVert x_n\rVert^2)$.
> 4. The step sizes satisfy (2), and $\sup_n\lVert x_n\rVert<\infty$ almost surely.
>
> Then the iterates track the **averaged** ODE $\dot x=\bar h(x)$, where $\bar h(x)=\sum_y\nu(y)\,h(x,y)$. If $\bar h$ has a unique globally asymptotically stable equilibrium $x^*$, then $x_n\to x^*$ almost surely.

The chain mixes on its own fast timescale, while the iterates move on the slow timescale $\alpha_n\to0$. So the iterates only "see" the drift averaged over the chain's stationary distribution.[^borkar2008] The condition that the chain's law does not depend on the iterates can be relaxed to "controlled" Markov noise.[^karmakarbhatnagar2018] For **linear TD(0)** with features $\phi$, the averaged drift is $\bar h(w)=b-Aw$ with $A=\Phi^\top D(I-\gamma P^\pi)\Phi$, $b=\Phi^\top Dr^\pi$, and $D=\operatorname{diag}(d_\pi)$ the stationary distribution. This is the setting of Tsitsiklis and Van Roy's analysis, worked out in [[Linear Function Approximation]].[^tsitsiklisvanroy1997]

### Checking the conditions: synchronous Q-learning

Here is the whole pipeline on one example. Suppose we have a simulator and, at every iteration, sample a next state $s'\sim p(\cdot\mid s,a)$ for **every** pair $(s,a)$ ("synchronous" Q-learning). Rewards satisfy $|r(s,a,s')|\le R_{\max}$. The update is

$$
Q_{n+1}(s,a)=Q_n(s,a)+\alpha_n\Big[r(s,a,s')+\gamma\max_{a'}Q_n(s',a')-Q_n(s,a)\Big].
$$

1. **Mean field.** $h(Q)=\mathcal{T}^*Q-Q$, where $(\mathcal{T}^*Q)(s,a)=\sum_{s'}p(s'\mid s,a)\big[r(s,a,s')+\gamma\max_{a'}Q(s',a')\big]$.
2. **Lipschitz.** $\mathcal{T}^*$ is a $\gamma$-contraction in $\lVert\cdot\rVert_\infty$ (proved in [[Dynamic Programming]]), so $h$ is Lipschitz with constant $1+\gamma$.
3. **Noise.** $M_{n+1}=\big(r+\gamma\max_{a'}Q_n(s',a')\big)-\mathcal{T}^*Q_n$ componentwise, which is an MDS by construction, and $|M_{n+1}(s,a)|\le2R_{\max}+2\gamma\lVert Q_n\rVert_\infty$. Squaring gives the growth bound.
4. **Stability.** $h_\infty(Q)=\gamma P\max Q-Q$, where $(P\max Q)(s,a)=\sum_{s'}p(s'\mid s,a)\max_{a'}Q(s',a')$; the rewards vanish in the limit. The map $Q\mapsto\gamma P\max Q$ is a $\gamma$-contraction with fixed point $0$, so by the contraction lemma the origin is globally asymptotically stable, and Borkar–Meyn gives $\sup_n\lVert Q_n\rVert<\infty$.
5. **Limit.** By the contraction lemma again, $Q^*$ is the unique globally asymptotically stable equilibrium of $\dot Q=\mathcal{T}^*Q-Q$.

With step sizes satisfying (2), $Q_n\to Q^*$ almost surely. Real Q-learning updates one pair at a time, which needs the asynchronous theory below.

---

## Lyapunov functions and globally asymptotically stable equilibria

The convergence theorems hand the problem over to the ODE. An equilibrium $x^*$ of $\dot x=h(x)$ (that is, $h(x^*)=0$) is **globally asymptotically stable** if it is stable (solutions starting close stay close) and every solution, from any starting point, converges to it. Note that being the only equilibrium is not enough: $\dot x=2x$ has the single equilibrium $0$, but every other solution $x_0e^{2t}$ runs away.

The standard certificate is a **Lyapunov function**, an "energy" that the flow always dissipates. A continuous $\Lambda:\mathbb{R}^d\to\mathbb{R}$ is a **global Lyapunov function** for $x^*$ if

1. $\Lambda(x)\ge0$, with equality if and only if $x=x^*$;
2. $\Lambda(x)\to\infty$ as $\lVert x\rVert\to\infty$ (so its sublevel sets are bounded);
3. $\Lambda(x(t))$ is strictly decreasing along every non-constant solution.

If such a $\Lambda$ exists, $x^*$ is globally asymptotically stable. Solutions cannot escape a sublevel set, and they cannot stall anywhere except at $x^*$.

> [!example] Two standard Lyapunov functions
> - **Linear drift** $h(x)=b-Ax$ with $x^\top Ax>0$ for all $x\ne0$, and $x^*=A^{-1}b$: take $\Lambda(x)=\lVert x-x^*\rVert_2^2$. Then $\frac{d}{dt}\Lambda=-2(x-x^*)^\top A(x-x^*)<0$ unless $x=x^*$. This is the argument for linear TD.
> - **Gradient flow** $h=-\nabla f$: take $\Lambda=f-f^*$. Then $\frac{d}{dt}\Lambda=-\lVert\nabla f\rVert^2\le0$. For a strictly convex $f$ with a minimizer this is a Lyapunov function; in general it shows that SGD's limit sets consist of stationary points.

### The contraction lemma

For TD and Q-learning the ODE has the form $\dot x=F(x)-x$ with $F$ a contraction. Then the distance to the fixed point is itself a Lyapunov function, **in whatever norm $F$ contracts**.

> [!note] Lemma (contraction ODEs)
> Let $F:\mathbb{R}^d\to\mathbb{R}^d$ be a $\gamma$-contraction, $\lVert F(x)-F(y)\rVert\le\gamma\lVert x-y\rVert$ with $\gamma<1$, in **some** norm $\lVert\cdot\rVert$ (for example $\lVert\cdot\rVert_\infty$ or a weighted max-norm). Let $x^*$ be its unique fixed point. Then every solution of $\dot x=F(x)-x$ satisfies
> $$
> \lVert x(t)-x^*\rVert\le e^{-(1-\gamma)t}\,\lVert x(0)-x^*\rVert .
> $$
> Hence $\Lambda(x)=\lVert x-x^*\rVert$ is a global Lyapunov function and $x^*$ is globally asymptotically stable.

> [!proof]- Proof (variation of constants and Grönwall)
> Let $e(t)=x(t)-x^*$. Since $F(x^*)=x^*$,
> $$
> \dot e(t)=-e(t)+\big(F(x(t))-F(x^*)\big).
> $$
> Multiply by the integrating factor $e^{t}$: $\frac{d}{dt}\big(e^{t}e(t)\big)=e^{t}\big(F(x(t))-F(x^*)\big)$. Integrating from $0$ to $t$ (the variation-of-constants formula),
> $$
> e(t)=e^{-t}e(0)+\int_0^te^{-(t-s)}\big(F(x(s))-F(x^*)\big)\,ds.
> $$
> Take norms. The norm of an integral is at most the integral of the norm (for any norm), and the contraction property gives
> $$
> \lVert e(t)\rVert\le e^{-t}\lVert e(0)\rVert+\gamma\int_0^te^{-(t-s)}\lVert e(s)\rVert\,ds.
> $$
> Set $u(t)=e^{t}\lVert e(t)\rVert$, a continuous non-negative function. Then $u(t)\le u(0)+\gamma\int_0^tu(s)\,ds$.
>
> **Grönwall.** Let $U(t)=u(0)+\gamma\int_0^tu(s)\,ds$. Then $U'(t)=\gamma u(t)\le\gamma U(t)$, so $\frac{d}{dt}\big(e^{-\gamma t}U(t)\big)\le0$ and $U(t)\le U(0)e^{\gamma t}=u(0)e^{\gamma t}$. Hence $u(t)\le u(0)e^{\gamma t}$, that is, $\lVert e(t)\rVert\le e^{-(1-\gamma)t}\lVert e(0)\rVert$.
>
> **Lyapunov properties.** $\Lambda(x)=\lVert x-x^*\rVert$ is continuous, zero only at $x^*$, and radially unbounded. The ODE is autonomous, so the bound holds from any starting time $s$: $\Lambda(x(t))\le e^{-(1-\gamma)(t-s)}\Lambda(x(s))<\Lambda(x(s))$ for $t>s$ whenever $x(s)\ne x^*$. So $\Lambda$ strictly decreases along non-constant solutions. The exponential bound also gives stability and global convergence directly.

> [!warning] Why not prove it in $\lVert\cdot\rVert_p$ and let $p\to\infty$?
> For $1<p<\infty$ one can differentiate $\lVert x(t)-x^*\rVert_p$ with the chain rule and use Hölder's inequality to get $\frac{d}{dt}\Lambda\le-(1-\gamma)\Lambda$. It is tempting to then let $p\to\infty$ to cover the max-norm. **That limit argument is invalid:** the $p$-norm computation uses that $F$ contracts in *that same* $p$-norm, and a max-norm contraction need not contract in any finite $p$-norm. Example: $F(x)=r+\gamma Px$ with $P=\begin{pmatrix}1&0\\1&0\end{pmatrix}$ (policy evaluation on a two-state chain where both states move to state 1). $P$ is row-stochastic, so $F$ is a $\gamma$-contraction in $\lVert\cdot\rVert_\infty$. But $\lVert\gamma P\rVert_2=\gamma\sqrt2$, which exceeds $1$ for $\gamma=0.9$. Figure 4 shows the Euclidean distance to $x^*$ *increasing* along a solution, while the max-norm distance decays exactly at the guaranteed rate. The variation-of-constants proof above works in any norm and avoids the issue.[^borkarsoumyanath1997] (An alternative is to work with upper Dini derivatives of $\lVert x(t)-x^*\rVert_\infty$, which is not differentiable everywhere.)

![[Reinforcement Learning/images/stochastic-approximation-lyapunov.svg]]
*Figure 4: $\dot x=F(x)-x$ for $F(x)=r+\gamma Px$ with $P$ as in the warning, $r=(1,0)$ and $\gamma=0.9$, so $x^*=(10,9)$. Left: solutions from five starting points, with the max-norm level sets around $x^*$ (dotted squares); every solution enters each square and never leaves. Right: for the start $x^*+3\,(1,0.45)$, the max-norm distance follows the bound $e^{-(1-\gamma)t}\lVert x(0)-x^*\rVert_\infty$ exactly, while the Euclidean distance first rises from $3.29$ to $3.44$ (at $t\approx1$). So $\lVert x-x^*\rVert_2$ is not a Lyapunov function here.*

**Consequences for RL.** $\mathcal{T}^\pi$ and $\mathcal{T}^*$ are $\gamma$-contractions in $\lVert\cdot\rVert_\infty$, so $V^\pi$ is the globally asymptotically stable equilibrium of $\dot V=\mathcal{T}^\pi V-V$ (tabular TD(0)), and $Q^*$ is that of $\dot Q=\mathcal{T}^*Q-Q$ (Q-learning). In [[Stochastic Shortest Path Problems]] with all policies proper, the Bellman operators contract in a *weighted* max-norm, and the lemma applies unchanged.

---

## Asynchronous stochastic approximation

Real Q-learning does not update every $(s,a)$ at every step. Along a trajectory it updates only the visited pair $(s_t,a_t)$, with a step size that typically depends on how often that pair has been visited. This is **asynchronous** SA. Write the iterate as a vector with components $i=1,\dots,d$ (for Q-learning, $i$ ranges over state-action pairs):

$$
x_{n+1}(i)=x_n(i)+\alpha_n(i)\Big[(Fx_n)(i)-x_n(i)+w_n(i)\Big],
$$

where $\alpha_n(i)=0$ whenever component $i$ is not updated at step $n$.

**Weighted max-norm.** For a vector $\xi>0$ of positive weights, $\lVert x\rVert_\xi=\max_i\frac{|x(i)|}{\xi(i)}$. $F$ is a **weighted max-norm pseudo-contraction** if there are $x^*$, $\xi>0$ and $\beta\in[0,1)$ with

$$
\lVert Fx-x^*\rVert_\xi\le\beta\,\lVert x-x^*\rVert_\xi\quad\text{for all }x .
$$

"Pseudo" means $F$ only has to pull every point towards the fixed point $x^*$, not bring every pair of points closer. Every contraction with fixed point $x^*$ is a pseudo-contraction.

> [!note] Theorem (Tsitsiklis, 1994)
> Let $\mathcal{F}_n$ contain the history up to step $n$ ($x_0,\dots,x_n$, $w_0,\dots,w_{n-1}$ and $\alpha_0,\dots,\alpha_n$). Assume:
> 1. **(Step sizes)** For each $i$, $\alpha_n(i)\ge0$ is $\mathcal{F}_n$-measurable, $\sum_n\alpha_n(i)=\infty$ and $\sum_n\alpha_n(i)^2<\infty$, almost surely.
> 2. **(Noise)** For all $i,n$, $\mathbb{E}[w_n(i)\mid\mathcal{F}_n]=0$, and $\mathbb{E}[w_n(i)^2\mid\mathcal{F}_n]\le A+B\max_{m\le n}\lVert x_m\rVert^2$ for some constants $A,B$.
> 3. **(Pseudo-contraction)** $F$ is a weighted max-norm pseudo-contraction with fixed point $x^*$.
>
> Then $x_n\to x^*$ almost surely.

This is the main convergence theorem of Tsitsiklis's paper, specialised to the case without communication delays (his theorem also allows each component to use outdated values of the others).[^tsitsiklis1994] Jaakkola, Jordan and Singh proved a closely related result at the same time.[^jaakkola1994] Three remarks:

- **No separate stability assumption** is needed. The pseudo-contraction itself keeps the iterates bounded, which Tsitsiklis proves as a first step.
- **Exploration is hidden in condition 1.** $\sum_n\alpha_n(i)=\infty$ forces component $i$ to be updated infinitely often. For Q-learning: *every state-action pair must be visited infinitely often*. Step sizes such as $\alpha=1/N_n(s,a)$, the reciprocal of the visit count, are $\mathcal{F}_n$-measurable and satisfy condition 1 on that event.
- **Why the max-norm survives asynchrony.** In ODE terms, asynchronous updating rescales each component of the drift by its update frequency: $\dot x=\Theta(t)\,(F(x)-x)$ with a positive diagonal $\Theta$.[^borkar1998] A max-norm-type Lyapunov function is insensitive to such rescaling: the component that attains the maximum is always pulled inwards, whatever its speed. A Euclidean Lyapunov function generally is not.

**Applying it to Q-learning.** Q-learning is $x=Q$, $F=\mathcal{T}^*$, $w_n(s,a)=r_{n}+\gamma\max_{a'}Q_n(s_{n+1},a')-(\mathcal{T}^*Q_n)(s,a)$. The noise is a martingale difference with conditional variance at most $K(1+\lVert Q_n\rVert_\infty^2)$ for bounded rewards.

- **Discounted MDPs:** $\mathcal{T}^*$ is a $\gamma$-contraction in $\lVert\cdot\rVert_\infty$, so $\xi\equiv1$ and $\beta=\gamma$.
- **Stochastic shortest paths** ($\gamma=1$) **with all policies proper:** there are weights $\xi>0$ and $\beta<1$ with $\sum_{s'\ne\text{terminal}}p(s'\mid s,a)\,\xi(s')\le\beta\,\xi(s)$ for all $s,a$ (see [[Stochastic Shortest Path Problems]]). Then $\mathcal{T}^*$ is a $\beta$-contraction in $\lVert\cdot\rVert_\xi$, as shown below.

> [!proof]- Weighted max-norm contraction of the Q-operator (SSP, all policies proper)
> Let $(HQ)(s,a)=\sum_{s'}p(s'\mid s,a)\big[r(s,a,s')+\max_{a'}Q(s',a')\big]$, with $Q(\text{terminal},\cdot)=0$ and the sum over non-terminal $s'$ in the $Q$ term. Use $\lVert Q\rVert_\xi=\max_{s,a}|Q(s,a)|/\xi(s)$.
>
> For two functions $f,g$ of $a'$, $|\max_{a'}f-\max_{a'}g|\le\max_{a'}|f-g|$: if the first maximum is the larger, at its maximizer $a^\dagger$ we have $\max f-\max g\le f(a^\dagger)-g(a^\dagger)$, and symmetrically. Hence
> $$
> \begin{aligned}
> \big|(HQ)(s,a)-(H\bar Q)(s,a)\big|&\le\sum_{s'}p(s'\mid s,a)\max_{a'}\big|Q(s',a')-\bar Q(s',a')\big|\\
> &\le\sum_{s'}p(s'\mid s,a)\,\xi(s')\,\lVert Q-\bar Q\rVert_\xi\\
> &\le\beta\,\xi(s)\,\lVert Q-\bar Q\rVert_\xi .
> \end{aligned}
> $$
> Divide by $\xi(s)$ and maximize over $(s,a)$: $\lVert HQ-H\bar Q\rVert_\xi\le\beta\lVert Q-\bar Q\rVert_\xi$.

Under the weaker SSP assumptions (some proper policy exists, and every improper policy has value $-\infty$ from some state), $H$ is no longer a contraction. Q-learning still converges if the iterates are bounded, by a monotonicity argument.[^bertsekastsitsiklis1996] The full Q-learning convergence theorem, with its exploration and step-size conditions, is stated in [[Q-Learning and SARSA]]; the original proof for the discounted case is due to Watkins and Dayan.[^watkinsdayan1992]

---

## Two-timescale stochastic approximation

Actor-critic algorithms run two coupled recursions. The **critic** estimates the value of the current policy; the **actor** improves the policy using the critic's estimate. If both moved at the same speed, the actor would chase a critic that never finishes evaluating. The fix is to give them **different step sizes**: the critic moves on a fast timescale, the actor on a slow one.

$$
\begin{aligned}
x_{n+1}&=x_n+a_n\big[h(x_n,y_n)+M^{(1)}_{n+1}\big]\quad\text{(fast, e.g. critic)},\\
y_{n+1}&=y_n+b_n\big[g(x_n,y_n)+M^{(2)}_{n+1}\big]\quad\text{(slow, e.g. actor)}.
\end{aligned}
$$

> [!note] Theorem (two-timescale stochastic approximation; Borkar, 1997)
> Assume:
> 1. $h$ and $g$ are Lipschitz.
> 2. $\sum_na_n=\sum_nb_n=\infty$, $\sum_n(a_n^2+b_n^2)<\infty$, and $b_n/a_n\to0$.
> 3. $M^{(1)}$ and $M^{(2)}$ are martingale-difference sequences with respect to $\mathcal{F}_n=\sigma(x_m,y_m,M^{(1)}_m,M^{(2)}_m:m\le n)$, with conditional second moments at most $K(1+\lVert x_n\rVert^2+\lVert y_n\rVert^2)$.
> 4. $\sup_n\big(\lVert x_n\rVert+\lVert y_n\rVert\big)<\infty$ almost surely.
> 5. For each fixed $y$, the ODE $\dot x=h(x,y)$ has a globally asymptotically stable equilibrium $\lambda(y)$, and $\lambda$ is Lipschitz.
> 6. The ODE $\dot y=g(\lambda(y),y)$ has a globally asymptotically stable equilibrium $y^*$.
>
> Then $(x_n,y_n)\to\big(\lambda(y^*),y^*\big)$ almost surely.

The statement is from Borkar's 1997 paper and Chapter 6 of his book.[^borkar1997]<sup>,</sup>[^borkar2008]

**Intuition.** Because $b_n/a_n\to0$, on the fast timescale the slow variable looks frozen: $x_n$ tracks $\dot x=h(x,y)$ with $y$ fixed and settles near $\lambda(y_n)$. On the slow timescale the fast variable looks equilibrated: $y_n$ sees $x_n\approx\lambda(y_n)$ and follows $\dot y=g(\lambda(y),y)$. A typical choice is $a_n=(n+1)^{-0.6}$ and $b_n=(n+1)^{-1}$.

For actor-critic, $x$ is the critic's parameter and $\lambda(y)$ is the critic's fixed point for the policy with parameter $y$ (for instance the TD fixed point). The slow ODE is then an (approximate) policy-gradient flow, which usually has many locally stable equilibria rather than one globally stable one. In that case condition 6 is replaced by the general limit-set conclusion of the basic theorem, and the actor converges to a set of stationary points of the performance objective. Konda and Borkar, and Konda and Tsitsiklis, gave the first such analyses; [[Actor-Critic Algorithms]] states the results.[^kondatsitsiklis2003]

---

## Constant step sizes: tracking and the bias–variance trade-off

Deep RL almost always uses a **constant** step size (a fixed learning rate). Condition (2) then fails, and the iterates do not converge. What do they do instead?

**They hover around the equilibrium.** With $\alpha_n\equiv\alpha$ the iterates are still a noisy Euler scheme for $\dot x=h(x)$, now with a fixed step. If $x^*$ is globally asymptotically stable, then for small $\alpha$ the iterates spend most of their time in a neighbourhood of $x^*$ whose size shrinks as $\alpha\to0$. Typically the mean-squared distance is of order $\alpha$.[^borkar2008]<sup>,</sup>[^kushneryin2003] For mean estimation this is exact: the stationary variance of $x_{n+1}=(1-\alpha)x_n+\alpha r_{n+1}$ is $\frac{\alpha\sigma^2}{2-\alpha}$, the plateau in Figure 1.

**They can track a moving target.** That is the point of a constant step size. If the target drifts, as with non-stationary bandits or the critic in actor-critic (whose target policy keeps changing), a decaying step size eventually stops listening, while a constant one keeps adapting. The price is a **bias–variance trade-off**. Suppose the mean performs a random walk, $\mu_{n+1}=\mu_n+\Delta_{n+1}$ with $\operatorname{Var}(\Delta)=\delta^2$, and we observe $r_{n+1}=\mu_n+\varepsilon_{n+1}$ with $\operatorname{Var}(\varepsilon)=\sigma^2$. The tracking error $e_n=x_n-\mu_n$ obeys $e_{n+1}=(1-\alpha)e_n+\alpha\varepsilon_{n+1}-\Delta_{n+1}$, so its stationary variance $v$ solves $v=(1-\alpha)^2v+\alpha^2\sigma^2+\delta^2$:

$$
\lim_{n\to\infty}\mathbb{E}\,e_n^2=\frac{\alpha^2\sigma^2+\delta^2}{\alpha(2-\alpha)}\approx\underbrace{\frac{\alpha\sigma^2}{2}}_{\text{noise}}+\underbrace{\frac{\delta^2}{2\alpha}}_{\text{lag}}\quad(\alpha\ \text{small}).
$$

A large $\alpha$ follows the target closely but passes on much of the observation noise. A small $\alpha$ averages the noise but lags behind the drift. The two terms balance at $\alpha^\star\approx\delta/\sigma$.

![[Reinforcement Learning/images/stochastic-approximation-tracking.svg]]
*Figure 5: tracking a random-walk mean ($\sigma=1$, $\delta=0.05$). Top: $\alpha=0.005$ lags behind the drift, $\alpha=0.3$ is dominated by observation noise, and the best constant step size $\alpha\approx0.049$ (close to $\delta/\sigma=0.05$) follows the mean. Bottom: steady-state mean-squared error against $\alpha$. The simulation (dots, 400 runs) matches the formula; the minimum is about $0.051$, compared with $0.25$ at $\alpha=0.005$ and $0.18$ at $\alpha=0.3$.*

**Iterate averaging.** A useful compromise for stationary problems is to run a constant or slowly decaying step size and report the running **average** of the iterates, $\bar x_n=\frac1n\sum_{k\le n}x_k$. Polyak and Juditsky showed that with $\alpha_n=(n+1)^{-\kappa}$, $\kappa\in(\tfrac12,1)$, the averaged iterate achieves the best possible asymptotic variance, as if the optimal step size were known in advance.[^polyakjuditsky1992]

The [[Multi-Armed Bandits#Non-stationary bandits: constant step sizes|bandit lecture]] shows the same trade-off from the point of view of the exponential recency-weighted average.

---

## Where stochastic approximation is used

| Algorithm | Iterate | Mean field $h$ | Noise | Limit |
|---|---|---|---|---|
| Sample average, bandit values | $q$ | $\mu-q$ | i.i.d. | $\mu$ |
| Gradient bandit | preferences $h$ | $\nabla_h\,\mathbb{E}[r]$ | MDS | best arm (preferences diverge) |
| SGD | $\theta$ | $-\nabla f(\theta)$ | MDS | stationary points of $f$ |
| Tabular TD(0) | $V$ | $\mathcal{T}^\pi V-V$ (scaled by visit frequencies) | Markov | $V^\pi$ |
| Q-learning | $Q$ | $\mathcal{T}^*Q-Q$ (scaled by visit frequencies) | Markov / asynchronous | $Q^*$ |
| Linear TD(0) | $w$ | $b-Aw$ | Markov | TD fixed point $A^{-1}b$ |
| Actor-critic | critic $x$, actor $y$ | two timescales | Markov | stationary points of $J$ |

Note the linear TD row. The limit is the **TD fixed point**, which in general is *not* the best approximation of $V^\pi$ in the feature space. Stochastic approximation tells us *where* an algorithm goes; whether that is where we want it to go is a separate question, taken up in [[Linear Function Approximation]] and [[Off-Policy Learning with Function Approximation]], where the ODE can even become unstable.

## Key takeaways

- Most RL updates are **stochastic approximation**: $x_{n+1}=x_n+\alpha_n[h(x_n)+M_{n+1}]$, a noisy step along a mean field whose zeros are the solutions we want.
- **Robbins–Monro step sizes** ($\sum\alpha=\infty$ to forget the start, $\sum\alpha^2<\infty$ to average the noise) are sufficient; $\alpha_n=1/(n+1)$ gives the sample mean.
- **Martingale-difference noise** is unpredictable from the past; its increments are uncorrelated, so the weighted noise sum converges.
- **ODE method:** the interpolated iterates shadow $\dot x=h(x)$. With a Lipschitz $h$, the noise bound and **bounded iterates**, they converge to the ODE's internally chain transitive sets, and to $x^*$ if it is globally asymptotically stable. **Borkar–Meyn** proves boundedness from the scaled ODE $\dot x=h_\infty(x)$.
- **Lyapunov functions** certify global asymptotic stability. For $\dot x=F(x)-x$ with $F$ a $\gamma$-contraction in any norm, $\lVert x(t)-x^*\rVert\le e^{-(1-\gamma)t}\lVert x(0)-x^*\rVert$; prove it by variation of constants and Grönwall, not by a $p\to\infty$ limit.
- **Asynchronous** updates (Q-learning) converge under per-component Robbins–Monro steps and a **weighted max-norm pseudo-contraction** (Tsitsiklis); every component must be updated infinitely often.
- **Two timescales** ($b_n/a_n\to0$) let a fast critic equilibrate while a slow actor moves. **Constant step sizes** trade convergence for tracking, with noise $\propto\alpha$ against lag $\propto1/\alpha$.

## Where this shows up next

This lecture is the analysis behind Part I. Each of these lectures states its convergence results with exact conditions and links back here:

- [[Multi-Armed Bandits]]: the incremental sample average and the gradient bandit as stochastic approximation.
- [[Policy Gradients]]: REINFORCE is stochastic gradient ascent, so it converges to stationary points of $J(\theta)$ under Robbins–Monro steps.
- [[Actor-Critic Algorithms]]: two-timescale convergence of the critic and the actor.
- [[Monte Carlo and Temporal-Difference Learning]]: Monte Carlo as a sample average and TD(0) as stochastic approximation for $V=\mathcal{T}^\pi V$.
- [[Q-Learning and SARSA]]: asynchronous stochastic approximation and Tsitsiklis's theorem.
- [[Linear Function Approximation]]: Markov noise, the ODE $\dot w=b-Aw$, and why $A$ is positive definite on-policy.
- [[Off-Policy Learning with Function Approximation]]: what goes wrong when the ODE is unstable.

<!-- References: footnote definitions below render as the numbered "References" list at the bottom of the page -->

[^robbinsmonro1951]: Robbins, H., & Monro, S. (1951). *A Stochastic Approximation Method*. Annals of Mathematical Statistics, 22(3), 400–407. [doi:10.1214/aoms/1177729586](https://doi.org/10.1214/aoms/1177729586)
[^kushneryin2003]: Kushner, H. J., & Yin, G. G. (2003). *Stochastic Approximation and Recursive Algorithms and Applications* (2nd ed.). Springer. [doi:10.1007/b97441](https://doi.org/10.1007/b97441)
[^williams1991]: Williams, D. (1991). *Probability with Martingales*. Cambridge University Press. [doi:10.1017/CBO9780511813658](https://doi.org/10.1017/CBO9780511813658)
[^borkar2008]: Borkar, V. S. (2008). *Stochastic Approximation: A Dynamical Systems Viewpoint*. Hindustan Book Agency / Cambridge University Press. [doi:10.1007/978-93-86279-38-5](https://doi.org/10.1007/978-93-86279-38-5). Second edition: Springer, 2023, [doi:10.1007/978-981-99-8277-6](https://doi.org/10.1007/978-981-99-8277-6)
[^benaim1999]: Benaïm, M. (1999). *Dynamics of Stochastic Approximation Algorithms*. Séminaire de Probabilités XXXIII, Lecture Notes in Mathematics 1709, 1–68. Springer. [doi:10.1007/BFb0096509](https://doi.org/10.1007/BFb0096509)
[^borkarmeyn2000]: Borkar, V. S., & Meyn, S. P. (2000). *The O.D.E. Method for Convergence of Stochastic Approximation and Reinforcement Learning*. SIAM Journal on Control and Optimization, 38(2), 447–469. [doi:10.1137/S0363012997331639](https://doi.org/10.1137/S0363012997331639)
[^ramaswamybhatnagar2017]: Ramaswamy, A., & Bhatnagar, S. (2017). *A Generalization of the Borkar–Meyn Theorem for Stochastic Recursive Inclusions*. Mathematics of Operations Research, 42(3), 648–661. [doi:10.1287/moor.2016.0821](https://doi.org/10.1287/moor.2016.0821)
[^karmakarbhatnagar2018]: Karmakar, P., & Bhatnagar, S. (2018). *Two Time-Scale Stochastic Approximation with Controlled Markov Noise and Off-Policy Temporal-Difference Learning*. Mathematics of Operations Research, 43(1), 130–151. [doi:10.1287/moor.2017.0855](https://doi.org/10.1287/moor.2017.0855)
[^tsitsiklisvanroy1997]: Tsitsiklis, J. N., & Van Roy, B. (1997). *An Analysis of Temporal-Difference Learning with Function Approximation*. IEEE Transactions on Automatic Control, 42(5), 674–690. [doi:10.1109/9.580874](https://doi.org/10.1109/9.580874)
[^borkarsoumyanath1997]: Borkar, V. S., & Soumyanatha, K. (1997). *An Analog Scheme for Fixed Point Computation. I. Theory*. IEEE Transactions on Circuits and Systems I, 44(4), 351–355. [doi:10.1109/81.563625](https://doi.org/10.1109/81.563625)
[^tsitsiklis1994]: Tsitsiklis, J. N. (1994). *Asynchronous Stochastic Approximation and Q-Learning*. Machine Learning, 16(3), 185–202. [doi:10.1007/BF00993306](https://doi.org/10.1007/BF00993306)
[^jaakkola1994]: Jaakkola, T., Jordan, M. I., & Singh, S. P. (1994). *On the Convergence of Stochastic Iterative Dynamic Programming Algorithms*. Neural Computation, 6(6), 1185–1201. [doi:10.1162/neco.1994.6.6.1185](https://doi.org/10.1162/neco.1994.6.6.1185)
[^borkar1998]: Borkar, V. S. (1998). *Asynchronous Stochastic Approximations*. SIAM Journal on Control and Optimization, 36(3), 840–851. [doi:10.1137/S0363012995282784](https://doi.org/10.1137/S0363012995282784)
[^bertsekastsitsiklis1996]: Bertsekas, D. P., & Tsitsiklis, J. N. (1996). *Neuro-Dynamic Programming*. Athena Scientific, Sections 4.3 and 5.6. [book website](http://www.athenasc.com/ndpbook.html)
[^watkinsdayan1992]: Watkins, C. J. C. H., & Dayan, P. (1992). *Q-Learning*. Machine Learning, 8(3–4), 279–292. [doi:10.1007/BF00992698](https://doi.org/10.1007/BF00992698)
[^borkar1997]: Borkar, V. S. (1997). *Stochastic Approximation with Two Time Scales*. Systems & Control Letters, 29(5), 291–294. [doi:10.1016/S0167-6911(97)90015-3](https://doi.org/10.1016/S0167-6911(97)90015-3)
[^kondatsitsiklis2003]: Konda, V. R., & Tsitsiklis, J. N. (2003). *On Actor-Critic Algorithms*. SIAM Journal on Control and Optimization, 42(4), 1143–1166. [doi:10.1137/S0363012901385691](https://doi.org/10.1137/S0363012901385691). See also Konda, V. R., & Borkar, V. S. (1999). *Actor-Critic–Type Learning Algorithms for Markov Decision Processes*. SIAM Journal on Control and Optimization, 38(1), 94–123. [doi:10.1137/S036301299731669X](https://doi.org/10.1137/S036301299731669X)
[^polyakjuditsky1992]: Polyak, B. T., & Juditsky, A. B. (1992). *Acceleration of Stochastic Approximation by Averaging*. SIAM Journal on Control and Optimization, 30(4), 838–855. [doi:10.1137/0330046](https://doi.org/10.1137/0330046)
