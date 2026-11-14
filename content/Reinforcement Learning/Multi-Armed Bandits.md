---
title: Multi-Armed Bandits
tags:
  - reinforcement-learning
  - bandits
  - exploration
  - ucb
  - stochastic-approximation
prev: "[[Index (Reinforcement Learning)]]"
next: "[[Introduction to Reinforcement Learning]]"
---

> **Prerequisites:** basic probability ([[Probability Basics|expectations, the law of large numbers]]). Familiarity with [[Introduction to Reinforcement Learning|MDPs]] helps for the closing section but is not needed.
>
> **TL;DR:**
> - A $K$-armed bandit is RL stripped to its core: one state, $K$ actions, unknown reward distributions.
> - Estimating each arm's value is a *stochastic approximation* problem: the incremental average, with step-size conditions.
> - Choosing which arm to pull is the **exploration–exploitation** trade-off.
>   - Greedy and fixed-$\varepsilon$ strategies suffer **linear regret**.
>   - **UCB** ("optimism in the face of uncertainty") provably achieves **logarithmic regret**.
>   - **Gradient bandits** are REINFORCE with a one-step horizon.
>   - **Thompson sampling** is a strong Bayesian alternative.

## Motivation

Think of the slot machines in a casino. There are $K$ machines, each with its own unknown mean payout. Every round we pull one arm and observe a random reward. How should we play to collect as much money as possible?

The tension is immediate. To find the best machine we must try them all (**explore**). Every pull spent on a bad machine is money lost compared with pulling the best one (**exploit**). The same trade-off appears in A/B testing of web pages, clinical trials, ad and recommendation selection, and hyper-parameter search. It also sits inside every RL algorithm that has to discover good actions by trying them. Bandits isolate it: there are no states and no delayed consequences, only the question of *which arm next*.

---

## The bandit problem

- There are $K$ **arms** $a\in\{1,\dots,K\}$. Pulling arm $a$ returns a reward drawn i.i.d. from an unknown distribution $\nu_a$.
- The **value** of an arm is its mean reward, $q_*(a)=\mathbb{E}[r_t\mid a_t=a]$. The **optimal arm** is $a^*=\arg\max_a q_*(a)$, and we write $q^*=q_*(a^*)$.
- At each round $t=1,2,\dots,T$ the agent chooses $a_t$ (based on everything seen so far) and receives $r_t\sim\nu_{a_t}$.
- **Objective:** maximize the expected total reward $\mathbb{E}\big[\sum_{t=1}^{T}r_t\big]$.

**Regret.** It is easier to compare algorithms by how much they *lose* relative to always pulling the best arm. The (expected) **regret** after $T$ rounds is

$$
R_T=T\,q^*-\mathbb{E}\Big[\sum_{t=1}^{T}r_t\Big].
$$

The algorithm with smaller regret over time is the better algorithm. Let $N_T(a)=\sum_{t=1}^{T}\mathbb{1}\{a_t=a\}$ be the number of pulls of arm $a$, and let $\Delta_a=q^*-q_*(a)\ge0$ be its **gap**. Since $\mathbb{E}[r_t]=\sum_a\mathbb{P}(a_t=a)\,q_*(a)$,

$$
R_T=\sum_{a}\mathbb{E}[N_T(a)]\,q^*-\sum_a\mathbb{E}[N_T(a)]\,q_*(a)=\sum_{a\neq a^*}\Delta_a\,\mathbb{E}[N_T(a)].
$$

This **regret decomposition** says that good algorithms are exactly those that pull bad arms rarely, and the worse an arm, the more rarely.

> [!note] A bandit is a one-state MDP
> In the language of [[Introduction to Reinforcement Learning]], a bandit is an MDP with a single state (or with $\gamma=0$, so that only the immediate reward matters). $q_*(a)$ is then exactly $Q^*(s,a)$. Everything hard about bandits (exploration) is still present in full RL. Everything hard about RL (credit assignment over time) is absent from bandits.

---

## Estimating action values

We do not know $q_*$. We can only estimate it. The natural estimate is the **sample average** of the rewards received from each arm:

$$
q_t(a)=\frac{\sum_{i=1}^{t}r_i\,\mathbb{1}\{a_i=a\}}{\sum_{i=1}^{t}\mathbb{1}\{a_i=a\}}.
$$

The numerator is the sum of rewards obtained from arm $a$ up to time $t$. The denominator is the number of times $a$ was chosen up to time $t$.

### Incremental updates

As written, $q_t(a)$ needs the whole reward history, which is $O(t)$ memory. Re-index by pulls of arm $a$. Let $r^a_1,r^a_2,\dots,r^a_n$ be the rewards from its first $n$ pulls, and $q^a_n=\frac1n\sum_{i=1}^n r^a_i$. For example, if arm $a$ was pulled at times $4$, $6$ and $7$, then $r^a_1=r_4$, $r^a_2=r_6$ and $r^a_3=r_7$. When the arm is pulled again:

$$
\begin{aligned}
q^a_{n+1}&=\frac{1}{n+1}\Big(\sum_{i=1}^{n}r^a_i+r^a_{n+1}\Big)=\frac{n}{n+1}\,q^a_n+\frac{1}{n+1}\,r^a_{n+1}\\
&=q^a_n+\underbrace{\frac{1}{n+1}}_{\text{step size}}\big(\underbrace{r^a_{n+1}-q^a_n}_{\text{error}}\big).
\end{aligned}
$$

This needs constant memory per arm, so $O(K)$ in total. Its form, **new estimate = old estimate + step size × (target − old estimate)**, will reappear in TD learning, Q-learning and essentially every value-based RL method.

> [!tip] Aside: stochastic approximation and the ODE method
> Write the reward as its mean plus zero-mean noise, $r^a_{n+1}=\mathbb{E}[r^a]+M_{n+1}$. The update becomes
> $$
> q^a_{n+1}=q^a_n+\alpha_n\big(\mathbb{E}[r^a]-q^a_n+M_{n+1}\big),
> $$
> a **stochastic approximation** (Robbins–Monro) iteration with step sizes $\alpha_n$.[^robbinsmonro1951] The *ODE method* says that, under suitable conditions, its long-run behaviour is captured by the ordinary differential equation
> $$
> \dot q(\tau)=\mathbb{E}[r^a]-q(\tau),
> $$
> written in continuous "algorithmic time" $\tau$ (not the bandit's $t$). This ODE has the single globally asymptotically stable equilibrium $q=\mathbb{E}[r^a]$, so the iterates converge there almost surely.[^borkar2008] The conditions are:
> 1. $\alpha_n>0$;
> 2. $\sum_n\alpha_n=\infty$, so the steps are large enough to travel any distance (the iterates can follow the ODE);
> 3. $\sum_n\alpha_n^2<\infty$, so the noise and discretization errors eventually average out;
> 4. $\{M_n\}$ is martingale-difference noise with bounded (conditional) variance, and the iterates stay bounded.
>
> For the sample average ($\alpha_n=\frac1{n+1}$) the law of large numbers gives the same answer. The ODE method is more powerful because it applies to a much larger class of algorithms, including Q-learning. Valid step-size sequences include
> $$
> \begin{aligned}
> &\alpha_n=\frac{1}{n+1},&&\alpha_n=\frac{1}{(n+1)^{\kappa}}\ \big(\tfrac12<\kappa\le1\big),\\
> &\alpha_n=\frac{\log(n+2)}{n+2},&&\alpha_n=\frac{1}{(n+2)\log(n+2)}.
> \end{aligned}
> $$
>
> **Why $\frac{1}{n+1}$ is special.** Unrolling the update with $\alpha_k=\frac{1}{k+1}$, the weight on the $k$-th reward is $\alpha_k\prod_{j>k}(1-\alpha_j)=\frac{1}{k+1}\cdot\frac{k+1}{n}=\frac1n$, the same for every reward. That is exactly the sample average, which is why the two derivations agree. Other schedules weight early and late rewards differently but still converge.
>
> **The Euler picture.** Without noise the update $q_{n+1}=q_n+\alpha_n(\mathbb{E}[r^a]-q_n)$ is the Euler method for the ODE above with time step $\alpha_n$. Place the $n$-th estimate at "algorithmic time" $\tau_n=\sum_{m<n}\alpha_m$. Unlike textbook Euler, the time step shrinks and the drift is noisy. Condition 2 makes $\tau_n\to\infty$, so the iterates have time to reach the equilibrium, and conditions 3–4 make both the discretization error and the accumulated noise vanish, so late in the run the iterates shadow the ODE. The theory, with exact conditions, proofs and figures, is in [[Stochastic Approximation]].

### Non-stationary bandits: constant step sizes

If the arm means drift over time (for example, the returns of a stock), a sample average is a poor estimator, because it weights a reward from long ago as much as one from a moment ago. The fix is a **constant** step size $\alpha\in(0,1)$, such as $\alpha=0.05$. Start from an initial estimate $q^a_0$ (for example $0$, or an optimistic value) and unroll the update:

$$
\begin{aligned}
q^a_{n+1}&=q^a_n+\alpha\big(r^a_{n+1}-q^a_n\big)=(1-\alpha)\,q^a_n+\alpha\,r^a_{n+1}\\
&=(1-\alpha)^2q^a_{n-1}+\alpha(1-\alpha)\,r^a_n+\alpha\,r^a_{n+1}\\
&\;\;\vdots\\
&=(1-\alpha)^{n+1}q^a_0+\sum_{i=1}^{n+1}\alpha(1-\alpha)^{n+1-i}\,r^a_i.
\end{aligned}
$$

(Check $n=0$: $q^a_1=(1-\alpha)q^a_0+\alpha r^a_1$.) The weights decay **exponentially** with age. The newest reward gets weight $\alpha$, the previous one $\alpha(1-\alpha)$, the first one $\alpha(1-\alpha)^n$, and the initial estimate $(1-\alpha)^{n+1}$. They sum to one, since $(1-\alpha)^{n+1}+\alpha\sum_{k=0}^{n}(1-\alpha)^k=1$. Old history is overwritten gradually, and recent samples have the most say: an **exponential recency-weighted average**. The initial value is never completely forgotten, only discounted geometrically, which is what makes optimistic initial values (below) work.

Note what we give up. A constant $\alpha$ violates $\sum_n\alpha_n^2<\infty$, so $q^a_n$ **never converges**. It keeps fluctuating around the (moving) mean, with a spread of order $\sqrt{\alpha}$. That is exactly the behaviour we want when the target moves. Choosing $\alpha$ is a bias–variance trade-off: a large $\alpha$ passes on more reward noise, a small one lags further behind the drift. [[Stochastic Approximation#Constant step sizes: tracking and the bias–variance trade-off|Stochastic Approximation]] computes the best constant step size for a random-walk mean.

![[Reinforcement Learning/images/multi-armed-bandits-step-sizes.svg]]
*Figure 1: left, the weights that the sample average (uniform, $1/n$) and a constant step size (exponentially increasing towards recent rewards) put on 30 past rewards. Right, tracking an arm whose mean drifts as a random walk. The sample average lags further and further behind; the constant-$\alpha$ estimate follows it.*

---

## Greedy and $\varepsilon$-greedy

Given estimates $q_t(1),\dots,q_t(K)$, which arm should we pull at time $t+1$?

**Greedy.** $a_{t+1}=\arg\max_a q_t(a)$, which is pure exploitation. If the estimates are bad because some arms have not been pulled enough, greedy can lock onto a sub-optimal arm **forever**. One unlucky early reward from the best arm is enough. The result is linear regret (see the experiment below).

**$\varepsilon$-greedy.** Explore a little, at random:

$$
a_{t+1}=\begin{cases}\arg\max_a q_t(a)&\text{with probability }1-\varepsilon,\\[2pt]\text{a uniformly random arm}&\text{with probability }\varepsilon,\end{cases}\qquad0\le\varepsilon\le1.
$$

$\varepsilon=0$ is greedy and $\varepsilon=1$ is uniformly random. Because every arm keeps being sampled, all estimates converge, and eventually the greedy choice is $a^*$.

> [!example]- Code: $\varepsilon$-greedy with incremental estimates (NumPy)
> ```python
> # assumes: K arms, pull(a) -> one reward, horizon T, rng = np.random.default_rng()
> q = np.zeros(K)          # estimates q_t(a)
> n = np.zeros(K)          # pull counts N_t(a)
> for t in range(T):
>     if rng.random() < eps:
>         a = rng.integers(K)                      # explore: a uniformly random arm
>     else:
>         a = rng.choice(np.flatnonzero(q == q.max()))   # exploit, breaking ties at random
>     r = pull(a)
>     n[a] += 1
>     q[a] += (r - q[a]) / n[a]                    # step size 1/n: the sample average
> ```
> A constant `alpha` in place of `1 / n[a]` gives the recency-weighted average for non-stationary arms, and `q = np.full(K, 5.0)` gives optimistic initial values.

**Regret of fixed $\varepsilon$ is linear.** Even once the estimates are perfect, each step explores with probability $\varepsilon$, and a random arm costs $\frac1K\sum_a\Delta_a$ on average. So for large $t$ the per-step regret is about $\frac{\varepsilon}{K}\sum_a\Delta_a$, and

$$
R_T\approx\frac{\varepsilon}{K}\Big(\sum_a\Delta_a\Big)\,T=\Theta(T).
$$

This rate is achieved, not merely an upper bound. Fixed-$\varepsilon$ exploration never stops paying for itself. The common remedy is to **decay** $\varepsilon$: start at $\varepsilon=1$ and reduce it over time, for example $\varepsilon_t=\min(1,c/t)$. With a well-chosen schedule this gives logarithmic regret, but the right constant depends on the unknown gaps.[^auer2002]

> [!tip] Optimistic initial values
> A simple trick makes even greedy explore: initialize every $q_0(a)$ well *above* any plausible reward, e.g. $q_0(a)=5$ when rewards are in $[0,1]$. Every arm that is tried "disappoints" and its estimate drops, so greedy moves on to the untried arms. This is a crude version of the idea behind UCB.

---

## Upper Confidence Bound (UCB)

$\varepsilon$-greedy explores blindly: it is as likely to re-try an arm that is clearly terrible as one that is nearly optimal but under-sampled. UCB explores **where it is uncertain**:

$$
a_{t+1}=\arg\max_a\Big[\underbrace{q_t(a)}_{\text{(I) exploitation}}+\underbrace{c\sqrt{\frac{\ln t}{N_t(a)}}}_{\text{(II) exploration bonus}}\Big],\qquad N_t(a)=\sum_{i=1}^{t}\mathbb{1}\{a_i=a\}.
$$

Here $c>0$ controls the amount of exploration. Two observations:

1. **Untried arms come first.** If $N_t(a)=0$, the bonus is taken to be $+\infty$, so $a$ is chosen. Equivalently, pull every arm once at the start.
2. **The bonus fades, but never vanishes.** For an arm that keeps being pulled, $N_t(a)$ grows roughly linearly in $t$ while $\ln t$ grows only logarithmically. So $\ln t/N_t(a)\to0$, and the choice is increasingly decided by $q_t(a)$. An arm that is *not* pulled sees its bonus slowly grow (through $\ln t$) until it is tried again. UCB therefore keeps exploring forever, but at a **logarithmic** rate: every sub-optimal arm is pulled about $\log T$ times.

Where does the bonus come from? It is the half-width of a confidence interval, as the next aside shows. UCB acts as if each arm is as good as it *plausibly* could be: **optimism in the face of uncertainty**. If the optimism was wrong, pulling the arm shrinks its interval and fixes the mistake. If it was right, we found a better arm.

> [!example]- Code: the UCB action choice (NumPy)
> ```python
> # assumes: estimates q (K,), pull counts n (K,), time t >= 1, exploration constant c (sqrt(2) for UCB1)
> if (n == 0).any():
>     a = int(np.argmax(n == 0))                     # untried arms first (infinite bonus)
> else:
>     a = int(np.argmax(q + c * np.sqrt(np.log(t) / n)))
> ```
> The rest of the loop, pulling the arm and the incremental update of `q[a]`, is the same as for $\varepsilon$-greedy.

![[Reinforcement Learning/images/multi-armed-bandits-ucb-snapshot.svg]]
*Figure 2: UCB1 on a 5-armed Bernoulli bandit after 150 pulls. Each bar runs from the estimate $q_t(a)$ to its upper confidence bound. The bounds of all arms are nearly level. Poor arms are pulled just often enough to keep their upper bound competitive, while the best arm (arm 3) gets the most pulls. The arm with the highest bound is pulled next.*

### Aside: concentration inequalities

> [!note] Sub-Gaussian random variables
> A random variable has "strong" tail decay if its tails fall off at least as fast as a Gaussian's. Formally, a zero-mean $X$ is **$\sigma$-sub-Gaussian** if $\mathbb{E}[e^{\lambda X}]\le e^{\lambda^2\sigma^2/2}$ for all $\lambda\in\mathbb{R}$. This implies the tail bound
> $$
> \mathbb{P}(|X|\ge u)\le2\,e^{-u^2/(2\sigma^2)}.
> $$
> Equivalently, up to constants, $X$'s tails are dominated by those of a Gaussian.
>
> - **Gaussian.** $X\sim\mathcal{N}(0,1)$ is $1$-sub-Gaussian, since $\mathbb{E}[e^{\lambda X}]=e^{\lambda^2/2}$ exactly. The Chernoff bound gives $\mathbb{P}(X\ge u)\le\inf_{\lambda>0}e^{-\lambda u}\mathbb{E}[e^{\lambda X}]=e^{-u^2/2}$, so $\mathbb{P}(|X|\ge u)\le2e^{-u^2/2}$. (The density $\phi(u)$ is *not* the tail probability. The tail is $\int_u^\infty\phi(x)\,dx$.)
> - **Bounded variables.** By *Hoeffding's lemma*, any $X\in[a,b]$, once centred, is $\frac{b-a}{2}$-sub-Gaussian. Rewards in $[0,1]$ are therefore $\frac12$-sub-Gaussian, with no need to assume Gaussian payouts.
> - **Mixtures.** Finite mixtures of Gaussians, once centred, are sub-Gaussian too.

![[Reinforcement Learning/images/multi-armed-bandits-subgaussian.svg]]
*Figure 3: tail probabilities on a log scale. The standard Gaussian's two-sided tail lies below $2e^{-u^2/2}$. The centred uniform on $[-\frac12,\frac12]$ has tails that vanish entirely beyond $\frac12$, comfortably below its Hoeffding-lemma bound $2e^{-2u^2}$.*

**Hoeffding's inequality.**[^hoeffding1963] Let $X_1,\dots,X_n$ be i.i.d. in $[0,1]$ with mean $\mu$, and $\bar X_n=\frac1n\sum_iX_i$. Then, for $u>0$,

$$
\mathbb{P}\big(\bar X_n-\mu\ge u\big)\le e^{-2nu^2},\qquad\mathbb{P}\big(\bar X_n-\mu\le-u\big)\le e^{-2nu^2}.
$$

**From tail bound to confidence interval.** Set each one-sided bound equal to $\delta$ and solve for $u$: $u=\sqrt{\frac{1}{2n}\ln\frac1\delta}$. A union bound over the two sides gives

$$
\mathbb{P}\Big(|\bar X_n-\mu|\ge\sqrt{\tfrac{1}{2n}\ln\tfrac1\delta}\Big)\le2\delta.
$$

So, with probability at least $1-2\delta$,

$$
q_t(a)-\sqrt{\tfrac{1}{2N}\ln\tfrac1\delta}\;\le\;q_*(a)\;\le\;q_t(a)+\sqrt{\tfrac{1}{2N}\ln\tfrac1\delta},\qquad N=N_t(a).
$$

The right-hand end is the "largest plausible value" of the arm. With $\delta=t^{-4}$ it becomes $q_t(a)+\sqrt{2\ln t/N_t(a)}$, the UCB1 index with $c=\sqrt2$. Letting $\delta$ shrink with $t$ is what pays for $N_t(a)$ being random. It lets a union bound cover every possible value of $N_t(a)$ at once, and that is why $\ln t$ appears in the bonus.

### UCB achieves logarithmic regret

> [!note] Theorem (UCB1)
> Assume rewards lie in $[0,1]$. UCB1 pulls each arm once, then plays $a_{t+1}=\arg\max_a\big[q_t(a)+\sqrt{2\ln t/N_t(a)}\big]$. Its regret satisfies, for every $T$,[^auer2002]
> $$
> R_T\le\sum_{a\neq a^*}\frac{8\ln T}{\Delta_a}+\Big(1+\frac{\pi^2}{3}\Big)\sum_{a}\Delta_a=O(\log T).
> $$

**Proof.** By the regret decomposition, it suffices to show that each sub-optimal arm $a$ satisfies $\mathbb{E}[N_T(a)]\le\frac{8\ln T}{\Delta_a^2}+1+\frac{\pi^2}{3}$. Let $\ell=\big\lceil\frac{8\ln T}{\Delta_a^2}\big\rceil$ be a "sufficient" number of pulls of arm $a$ for horizon $T$. Write $\bar\mu_a(x)$ for the average of the first $x$ rewards from arm $a$, $\bar\mu_*(y)$ for the average of the first $y$ rewards from $a^*$, and $c_{t,x}=\sqrt{2\ln t/x}$. Split the pulls of $a$ according to whether it has already been pulled $\ell$ times:

$$
\mathbb{E}[N_T(a)]=\underbrace{\sum_t\mathbb{P}\big(a_{t+1}=a,\ N_t(a)<\ell\big)}_{\text{(i)}}+\underbrace{\sum_t\mathbb{P}\big(a_{t+1}=a,\ N_t(a)\ge\ell\big)}_{\text{(ii)}}.
$$

**Term (i).** Every time the event in (i) happens, $N(a)$ increases by one while still below $\ell$. In any run this can happen at most $\ell$ times, so $\text{(i)}\le\ell\le\frac{8\ln T}{\Delta_a^2}+1$.

**Term (ii).** Arm $a$ is pulled at $t+1$ only if its index beats that of $a^*$. With $x=N_t(a)\ge\ell$ and $y=N_t(a^*)$, that means

$$
\bar\mu_a(x)+c_{t,x}\;\ge\;\bar\mu_*(y)+c_{t,y}.
$$

For this to hold, at least one of two things must go wrong:

- **(A)** the optimal arm is badly *under*-estimated: $\bar\mu_*(y)+c_{t,y}\le q^*$; or
- **(B)** arm $a$ is badly *over*-estimated: $\bar\mu_a(x)+c_{t,x}>q^*$.

(If neither holds, $\bar\mu_a(x)+c_{t,x}\le q^*<\bar\mu_*(y)+c_{t,y}$, and $a$ is not pulled.) Because $x\ge\ell$ and $t\le T$, the bonus is small:

$$
c_{t,x}=\sqrt{\frac{2\ln t}{x}}\le\sqrt{\frac{2\ln T\,\Delta_a^2}{8\ln T}}=\frac{\Delta_a}{2}.
$$

So (B) implies $\bar\mu_a(x)>q^*-\frac{\Delta_a}{2}=q_*(a)+\frac{\Delta_a}{2}$. By Hoeffding,

$$
\begin{aligned}
\mathbb{P}(\text{A})&\le e^{-2y\,c_{t,y}^2}=e^{-4\ln t}=t^{-4},\\
\mathbb{P}(\text{B})&\le e^{-2x(\Delta_a/2)^2}\le e^{-2\ell\Delta_a^2/4}\le e^{-4\ln T}\le t^{-4}.
\end{aligned}
$$

We do not know $x$ and $y$ in advance, so take a union bound over all $1\le x,y\le t$. That is at most $t^2$ pairs, each contributing at most $2t^{-4}$:

$$
\text{(ii)}\le\sum_{t=1}^{\infty}t^2\cdot2t^{-4}=\sum_{t=1}^{\infty}\frac{2}{t^2}=\frac{\pi^2}{3}.
$$

Adding (i) and (ii) gives the claim. Multiplying by $\Delta_a$ and summing over $a\neq a^*$ gives the theorem. $\blacksquare$

> [!note] How good is $O(\log T)$?
> It is the best possible order. Lai and Robbins showed that any algorithm that is good on *every* bandit instance must pull each sub-optimal arm at least $\frac{\ln T}{\mathrm{KL}(\nu_a\,\|\,\nu_{a^*})}(1-o(1))$ times.[^lairobbins1985] So regret must grow at least logarithmically. The bound above depends on the gaps: it blows up as $\Delta_a\to0$. A similar argument gives the *gap-free* bound $R_T=O\big(\sqrt{KT\log T}\big)$, which holds uniformly over all instances.[^lattimore2020]

---

## Gradient bandit algorithms

Instead of estimating values, learn a numerical **preference** $h_t(a)$ for each arm and act with a **softmax** (Boltzmann) policy:

$$
\pi_t(a)\triangleq\mathbb{P}(a_t=a)=\frac{e^{h_t(a)}}{\sum_{b=1}^{K}e^{h_t(b)}}.
$$

Only *relative* preferences matter: adding a constant to every $h_t$ changes nothing. Starting from equal preferences gives a uniform policy, which means full exploration.

**Gradient ascent on expected reward.** We would like to maximize $\mathbb{E}[r_t]=\sum_x\pi_t(x)\,q_*(x)$ directly:

$$
h_{t+1}(a)=h_t(a)+\alpha\,\frac{\partial\,\mathbb{E}[r_t]}{\partial h_t(a)},\qquad\frac{\partial\,\mathbb{E}[r_t]}{\partial h_t(a)}=\sum_x q_*(x)\,\frac{\partial\pi_t(x)}{\partial h_t(a)}.
$$

But $q_*$ is unknown. Multiply and divide by $\pi_t(x)$ to turn the sum into an expectation we can sample:

$$
\begin{aligned}
\frac{\partial\,\mathbb{E}[r_t]}{\partial h_t(a)}&=\sum_x\pi_t(x)\,q_*(x)\,\frac{\partial\pi_t(x)/\partial h_t(a)}{\pi_t(x)}=\mathbb{E}_{a_t\sim\pi_t}\left[q_*(a_t)\,\frac{\partial\pi_t(a_t)/\partial h_t(a)}{\pi_t(a_t)}\right]\\
&=\mathbb{E}_{a_t\sim\pi_t}\left[r_t\,\frac{\partial\pi_t(a_t)/\partial h_t(a)}{\pi_t(a_t)}\right],
\end{aligned}
$$

where the last step uses $\mathbb{E}[r_t\mid a_t]=q_*(a_t)$. Since $\frac{\nabla\pi}{\pi}=\nabla\log\pi$, this is the **log-derivative trick** from [[Policy Gradients]].

**The softmax derivative.** By the quotient rule, writing $Z=\sum_k e^{h_t(k)}$ and keeping the indicator $\mathbb{1}\{a=x\}$ throughout:

$$
\frac{\partial\pi_t(x)}{\partial h_t(a)}=\frac{\mathbb{1}\{a=x\}\,e^{h_t(x)}\,Z-e^{h_t(x)}\,e^{h_t(a)}}{Z^2}=\pi_t(x)\big(\mathbb{1}\{a=x\}-\pi_t(a)\big).
$$

Substituting, the $\pi_t(a_t)$ cancels:

$$
\frac{\partial\,\mathbb{E}[r_t]}{\partial h_t(a)}=\mathbb{E}_{a_t\sim\pi_t}\Big[r_t\big(\mathbb{1}\{a=a_t\}-\pi_t(a)\big)\Big].
$$

**Baselines.** As in policy gradients, we may subtract any baseline $b_t$ that does not depend on the current action $a_t$, without changing the expected gradient:

$$
\mathbb{E}_{a_t\sim\pi_t}\left[b_t\,\frac{\partial\pi_t(a_t)/\partial h_t(a)}{\pi_t(a_t)}\right]=b_t\sum_x\frac{\partial\pi_t(x)}{\partial h_t(a)}=b_t\,\frac{\partial}{\partial h_t(a)}\underbrace{\sum_x\pi_t(x)}_{=1}=0.
$$

A good choice is the average of the **past** rewards, $\bar r_t=\frac{1}{t-1}\sum_{t'<t}r_{t'}$ (or an exponential moving average). It must *not* include the current $r_t$, because that would make it depend on $a_t$ and bias the gradient.

Replacing the expectation by the single sample we observe gives a stochastic gradient **ascent** algorithm.

> [!example] Algorithm: gradient bandit
> Initialize $h_1(a)=0$ for all $a$, and $\bar r_1=0$. **For** $t=1,2,\dots$:
> 1. Sample $a_t\sim\pi_t=\operatorname{softmax}(h_t)$ and observe $r_t$.
> 2. Update **every** arm's preference: $\;h_{t+1}(a)=h_t(a)+\alpha\,(r_t-\bar r_t)\big(\mathbb{1}\{a=a_t\}-\pi_t(a)\big)$. The chosen arm moves by $+\alpha(r_t-\bar r_t)(1-\pi_t(a_t))$, and every other arm by $-\alpha(r_t-\bar r_t)\pi_t(a)$.
> 3. Update the baseline with the new reward: $\bar r_{t+1}=\bar r_t+\frac1t(r_t-\bar r_t)$.

> [!example]- Code: the gradient bandit (NumPy)
> ```python
> # assumes: K arms, pull(a) -> one reward, step size alpha, horizon T, rng
> h = np.zeros(K)                          # preferences h_t(a)
> r_bar = 0.0                              # baseline: average of PAST rewards
> for t in range(1, T + 1):
>     pi = np.exp(h - h.max()); pi /= pi.sum()       # softmax (shifted for stability)
>     a = rng.choice(K, p=pi)
>     r = pull(a)
>     onehot = np.eye(K)[a]
>     h += alpha * (r - r_bar) * (onehot - pi)       # step 2: every arm moves
>     r_bar += (r - r_bar) / t                       # step 3: only now include r_t
> ```
> The bracket `onehot - pi` is $\nabla_h\log\pi_h(a_t)$, so `alpha * (r - r_bar) * (onehot - pi)` is one REINFORCE step with a baseline; autograd on `torch.log_softmax(h, 0)[a]` gives the same vector.

If a reward is better than the baseline, the chosen arm becomes more likely and the others less likely; if it is worse, the reverse. Without a baseline, if all rewards are positive (say around $4$), *every* pulled arm gets reinforced, and learning becomes much slower and noisier.

**The stochastic-approximation view.** Write $J(h)=\sum_x\pi_h(x)\,q_*(x)$ for the expected reward. Given the past, the expected update is exactly $\alpha\,\nabla_hJ(h_t)$, because the baseline uses past rewards only. So the update is $h_{t+1}=h_t+\alpha\big[\nabla J(h_t)+M_{t+1}\big]$ with martingale-difference noise $M_{t+1}$, and with decreasing step sizes it follows the gradient-flow ODE $\dot h=\nabla J(h)$. Along this flow $\frac{d}{dt}J=\lVert\nabla J\rVert^2\ge0$, so the expected reward never decreases. Its stationary points are not where one might expect: $\frac{\partial J}{\partial h(a)}=\pi_h(a)\big(q_*(a)-J(h)\big)$ vanishes for every $a$ only if all arms have the same mean. When one arm is strictly best there is **no finite equilibrium**: the preferences drift off to infinity and $\pi_h$ approaches a one-hot policy. For the exact gradient flow it is the optimal arm.[^mei2020] Two caveats. The boundedness assumption of the standard convergence theorem fails here, so it does not apply directly. And with a *constant* $\alpha$, as in the algorithm box, the iterates only track the ODE approximately (see [[Stochastic Approximation]]).

![[Reinforcement Learning/images/multi-armed-bandits-gradient-baseline.svg]]
*Figure 4: the gradient bandit on 10-armed Gaussian bandits whose means are around $4$ (averaged over 400 instances). The average-reward baseline raises the fraction of optimal pulls substantially at both step sizes: from about 48% to 83% at $\alpha=0.1$, and from about 27% to 67% at $\alpha=0.4$ (last 100 steps). The experiment follows Sutton & Barto, Figure 2.5.[^suttonbarto]*

> [!tip] Gradient bandit = one-step REINFORCE
> The update is exactly the [[Policy Gradients#REINFORCE|REINFORCE]] estimator for a horizon-1 problem: a softmax policy with parameters $\theta=h$, $\nabla_\theta\log\pi_\theta(a_t)=\mathbb{1}\{\cdot=a_t\}-\pi_\theta(\cdot)$, return $r_t$ and baseline $b=\bar r_t$. Everything in the PG lecture about baselines and variance applies here.

---

## Thompson sampling

A Bayesian alternative: keep a posterior over each arm's mean, **sample** one plausible mean per arm from it, and pull the arm whose sample is largest.[^thompson1933] Each arm is pulled with exactly the posterior probability that it is the best arm, so exploration fades automatically as the posteriors sharpen.

> [!example] Algorithm: Thompson sampling (Bernoulli rewards)
> Initialize $S_a=F_a=1$ for all arms, i.e. a Beta(1,1) (uniform) prior. **For** $t=1,2,\dots$:
> 1. For each arm, sample $\tilde q_a\sim\mathrm{Beta}(S_a,F_a)$.
> 2. Pull $a_t=\arg\max_a\tilde q_a$ and observe $r_t\in\{0,1\}$.
> 3. Update the posterior: $S_{a_t}\leftarrow S_{a_t}+r_t$ and $F_{a_t}\leftarrow F_{a_t}+(1-r_t)$.

> [!example]- Code: Thompson sampling for Bernoulli arms (NumPy)
> ```python
> # assumes: K arms, pull(a) -> reward in {0, 1}, horizon T, rng
> S = np.ones(K)                           # 1 + number of successes
> F = np.ones(K)                           # 1 + number of failures
> for t in range(T):
>     a = int(np.argmax(rng.beta(S, F)))   # one plausible mean per arm, act greedily on the samples
>     r = pull(a)
>     S[a] += r
>     F[a] += 1 - r
> ```

Thompson sampling also achieves $O(\log T)$ regret.[^agrawalgoyal2012] For Bernoulli arms it even matches the Lai–Robbins lower-bound constant,[^kaufmann2012] and it is often the best performer in practice.[^russo2018]

---

## Putting it together: an experiment

![[Reinforcement Learning/images/multi-armed-bandits-regret.svg]]
*Figure 5: 10-armed Bernoulli bandits with means drawn uniformly from $[0.05,0.95]$, averaged over 300 random instances; the x-axis is logarithmic. Top: expected cumulative regret. Bottom: fraction of rounds in which the optimal arm was pulled.*

What the experiment shows:

- **Greedy** locks onto a sub-optimal arm in many runs. It picks the optimal arm only about 20% of the time, and its regret grows linearly; it leaves the chart.
- **Fixed $\varepsilon=0.1$** learns quickly but keeps paying for exploration. Its regret curves upward, which is linear growth on a log-$x$ axis.
- **UCB1** has a straight line on the log-$x$ axis, which is logarithmic regret. At this horizon it starts behind $\varepsilon$-greedy, overtakes it at around $t\approx8{,}000$, and keeps pulling away. Asymptotic guarantees can take a while to pay off.
- **Decaying $\varepsilon$** and the **gradient bandit** do well here, but both have tuning knobs ($c$ and $\alpha$) whose best values depend on the instance.
- **Thompson sampling** has by far the lowest regret, and the optimal arm is pulled about 97% of the time by the end.

---

## Beyond: from bandits to reinforcement learning

- **Contextual bandits.** Before each pull the agent sees a context $x_t$ (a user, a patient), and the best arm depends on it. LinUCB and its relatives extend UCB to rewards that are linear in features of the context.[^li2010] This is an MDP with $\gamma=0$: states exist, but actions do not affect future states.
- **Full RL** adds the last ingredient: actions change the next state, so exploration must be planned over time. The ideas from this lecture carry over: incremental estimates, $\varepsilon$-greedy, optimism (UCT in Monte-Carlo tree search, optimistic model-based RL), softmax policies, and gradients with baselines.

## Key takeaways

- **Regret decomposition:** $R_T=\sum_{a\ne a^*}\Delta_a\,\mathbb{E}[N_T(a)]$. Good algorithms pull bad arms rarely.
- **Incremental estimates** $q\leftarrow q+\alpha(r-q)$ are stochastic approximation. Robbins–Monro step sizes ($\sum\alpha=\infty$, $\sum\alpha^2<\infty$) give convergence. A constant $\alpha$ gives tracking of non-stationary means instead.
- **Greedy** can fail forever. **Fixed $\varepsilon$-greedy** has $\Theta(T)$ regret.
- **UCB1** adds a confidence bonus $\sqrt{2\ln t/N_t(a)}$, derived from Hoeffding's inequality. Its regret is $O\big(\sum_a\frac{\ln T}{\Delta_a}\big)$, which matches the Lai–Robbins lower bound in order.
- **Gradient bandits** are one-step REINFORCE. Update all arms, and use a baseline built from past rewards only.
- **Thompson sampling**, which samples from the posterior, is simple, near-optimal and strong in practice.

## Where this shows up next

- [[Introduction to Reinforcement Learning]]: add states and transitions to get MDPs.
- [[Dynamic Programming]]: what to do once the model is known, so that no exploration is needed.
- [[Policy Gradients]]: the gradient-bandit derivation, extended to whole trajectories.
- [[Q-Learning and SARSA]]: $\varepsilon$-greedy exploration and the stochastic-approximation (ODE) argument, reused for MDPs.
- [[Stochastic Approximation]]: the general theory behind incremental updates, step-size conditions and the ODE method.

<!-- References: footnote definitions below render as the numbered "References" list at the bottom of the page -->

[^suttonbarto]: Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.), Chapter 2. [book website](http://incompleteideas.net/book/the-book-2nd.html)
[^robbinsmonro1951]: Robbins, H., & Monro, S. (1951). *A Stochastic Approximation Method*. Annals of Mathematical Statistics, 22(3), 400–407. [doi:10.1214/aoms/1177729586](https://doi.org/10.1214/aoms/1177729586)
[^borkar2008]: Borkar, V. S. (2008). *Stochastic Approximation: A Dynamical Systems Viewpoint*. Hindustan Book Agency / Cambridge University Press. [doi:10.1007/978-93-86279-38-5](https://doi.org/10.1007/978-93-86279-38-5)
[^hoeffding1963]: Hoeffding, W. (1963). *Probability Inequalities for Sums of Bounded Random Variables*. Journal of the American Statistical Association, 58(301), 13–30. [doi:10.1080/01621459.1963.10500830](https://doi.org/10.1080/01621459.1963.10500830)
[^auer2002]: Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). *Finite-time Analysis of the Multiarmed Bandit Problem*. Machine Learning, 47, 235–256. [doi:10.1023/A:1013689704352](https://doi.org/10.1023/A:1013689704352)
[^lairobbins1985]: Lai, T. L., & Robbins, H. (1985). *Asymptotically Efficient Adaptive Allocation Rules*. Advances in Applied Mathematics, 6(1), 4–22. [doi:10.1016/0196-8858(85)90002-8](https://doi.org/10.1016/0196-8858(85)90002-8)
[^lattimore2020]: Lattimore, T., & Szepesvári, C. (2020). *Bandit Algorithms*. Cambridge University Press. [doi:10.1017/9781108571401](https://doi.org/10.1017/9781108571401)
[^thompson1933]: Thompson, W. R. (1933). *On the Likelihood that One Unknown Probability Exceeds Another in View of the Evidence of Two Samples*. Biometrika, 25(3–4), 285–294. [doi:10.2307/2332286](https://doi.org/10.2307/2332286)
[^agrawalgoyal2012]: Agrawal, S., & Goyal, N. (2012). *Analysis of Thompson Sampling for the Multi-armed Bandit Problem*. COLT. [arXiv:1111.1797](https://arxiv.org/abs/1111.1797)
[^kaufmann2012]: Kaufmann, E., Korda, N., & Munos, R. (2012). *Thompson Sampling: An Asymptotically Optimal Finite Time Analysis*. ALT. [arXiv:1205.4217](https://arxiv.org/abs/1205.4217)
[^russo2018]: Russo, D., Van Roy, B., Kazerouni, A., Osband, I., & Wen, Z. (2018). *A Tutorial on Thompson Sampling*. Foundations and Trends in Machine Learning. [arXiv:1707.02038](https://arxiv.org/abs/1707.02038)
[^mei2020]: Mei, J., Xiao, C., Szepesvári, C., & Schuurmans, D. (2020). *On the Global Convergence Rates of Softmax Policy Gradient Methods*. ICML. [arXiv:2005.06392](https://arxiv.org/abs/2005.06392)
[^li2010]: Li, L., Chu, W., Langford, J., & Schapire, R. E. (2010). *A Contextual-Bandit Approach to Personalized News Article Recommendation*. International World Wide Web Conference (WWW '10). [arXiv:1003.0146](https://arxiv.org/abs/1003.0146)
