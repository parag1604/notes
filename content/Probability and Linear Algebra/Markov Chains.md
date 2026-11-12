---
title: Markov Chains
tags:
  - probability
  - stochastic-processes
date: 2026-11-12
draft: false
prev: "[[Sample Statistics]]"
next: "[[Index (Probability and Linear Algebra)]]"
---

> **Prerequisites:** [[Probability Basics]] (conditional independence), [[Random Variables]] (first-step analysis, geometric distribution) and [[Sample Statistics]] (the law of large numbers, Borel–Cantelli).
>
> **Series:** this is part 5 of 5. [[Probability Basics]] → [[Random Variables]] → [[Joint Distributions]] → [[Sample Statistics]] → **Markov Chains**.
>
> **TL;DR:**
> - A Markov chain forgets its past: given the present state, the future is independent of how it got there. It is described by a transition matrix $\mathbf{P}$, and $n$ steps are the matrix power $\mathbf{P}^n$.
> - **First-step analysis** turns hitting probabilities and expected hitting times into linear equations. Gambler's ruin is the classic example: a small house edge makes ruin very likely.
> - States are **recurrent** (the chain returns for sure, infinitely often) or **transient** (it eventually leaves for good). The random walk returns home in one and two dimensions but not in three.
> - An irreducible, positive recurrent chain has a unique **stationary distribution** $\boldsymbol\pi=\boldsymbol\pi\mathbf{P}$, the long-run fraction of time spent in each state. If the chain is also aperiodic, it converges to $\boldsymbol\pi$ from any start, at a rate set by the **spectral gap**.
> - PageRank is the stationary distribution of a random web surfer. MCMC (Metropolis–Hastings, Gibbs) designs a chain whose stationary distribution is the one we want to sample from.
> - In **continuous time**, the chain holds each state for an exponential time and is described by a rate matrix $\mathbf{Q}$, with $\mathbf{P}(t)=e^{t\mathbf{Q}}$. Brownian motion, the forward process of diffusion models, and state space models all build on this.
>
> **Code:** [Gibbs-Sampling](https://github.com/parag1604/Gibbs-Sampling): Gibbs sampling on a discrete table and a correlated Gaussian, compared with Metropolis–Hastings; [Mamba-in-JAX](https://github.com/parag1604/Mamba-in-JAX): linear state space models, S4 and Mamba, for the continuous-time section.

> **Notation in this lecture:** states are $i,j,k$ in a countable state space $\mathcal{S}$; $p_{ij}$ is a one-step transition probability, $p^{(n)}_{ij}$ an $n$-step one; distributions over states are **row** vectors, as is standard for Markov chains. $\mathbb{P}_i$ and $\mathbb{E}_i$ condition on $X_0=i$.

## Motivation

So far our random variables have been independent, or have come in small groups. Most interesting systems have **memory**: tomorrow's weather depends on today's, the next word depends on the previous ones, the length of a queue depends on how long it was a moment ago. Allowing arbitrary dependence on the entire past is hopeless to analyse. The Markov assumption is the simplest useful compromise: **the future depends on the past only through the present**. It is surprisingly powerful:
- [[Introduction to Reinforcement Learning|Reinforcement learning]] is built on Markov decision processes; with a fixed policy, an MDP is a Markov chain.
- n-gram language models, and the hidden Markov models of speech recognition, are Markov chains.
- Queues, inventories, epidemics and population models are Markov chains.
- Google's PageRank is the stationary distribution of a Markov chain.
- Markov chain Monte Carlo samples from distributions that cannot be sampled directly, including the energy-based models of [[Energy-Based Models]].

Kumar's notes (Chapters 2 and 4) are the rigorous source for this lecture.[^kumar2012] Norris's book is a beautiful, careful introduction,[^norris1997] and Levin and Peres cover convergence and mixing.[^levin2017]

## The Markov property

A sequence of random variables $X_0,X_1,X_2,\dots$ taking values in a countable set $\mathcal{S}$ is a **Markov chain** if

$$
\mathbb{P}(X_{n+1}=j\mid X_n=i,\,X_{n-1}=i_{n-1},\,\dots,\,X_0=i_0)=\mathbb{P}(X_{n+1}=j\mid X_n=i).
$$

Once we know where the chain is now, knowing how it got there tells us nothing more about where it will go next. It is time-**homogeneous** if this probability does not depend on $n$; we write it $p_{ij}$ and always assume homogeneity.

The Markov property is a **conditional independence** ([[Probability Basics#Independence|Probability Basics]]): the future is independent of the past *given the present*. It does not make the past and the future independent. If $X_n$ counts the heads in $n$ coin tosses, then $X_{n+1}$ depends strongly on $X_{n-1}$: knowing $X_{n-1}=10$ tells us $X_{n+1}\in\{10,11,12\}$. But once $X_n$ is known, $X_{n-1}$ adds nothing.

The transition probabilities form the **transition matrix** $\mathbf{P}=(p_{ij})$. Each row is a probability distribution over the next state, so $p_{ij}\ge0$ and every row sums to 1; such a matrix is called **stochastic**. A chain is often drawn as a **transition diagram**: a graph with an arrow $i\to j$ labelled $p_{ij}$ whenever $p_{ij}>0$ (Figure 1a).

By the chain rule and the Markov property, the probability of a whole path is a product of one-step probabilities:

$$
\mathbb{P}(X_0=i_0,X_1=i_1,\dots,X_n=i_n)=\mu_0(i_0)\,p_{i_0i_1}\,p_{i_1i_2}\cdots p_{i_{n-1}i_n},
$$

where $\boldsymbol\mu_0$ is the distribution of the starting state. So the initial distribution and $\mathbf{P}$ together determine everything about the chain.

> [!example] Example: weather
> Model each day as sunny, cloudy or rainy, with
>
> $$
> \mathbf{P}=\begin{pmatrix}0.7&0.2&0.1\\0.3&0.4&0.3\\0.2&0.4&0.4\end{pmatrix}\qquad(\text{rows and columns: sunny, cloudy, rainy}).
> $$
>
> A sunny day is followed by another sunny day with probability 0.7. The chance of "sunny, sunny, rainy" starting from a sunny day is $0.7\times0.1=0.07$.

A simple consequence of the Markov property: the time the chain stays in state $i$ before leaving is [[Random Variables#Geometric and memorylessness|geometric]], with mean $1/(1-p_{ii})$. Each day it stays with probability $p_{ii}$, independently of how long it has already stayed. In the weather chain, sunny spells last $1/0.3\approx3.3$ days on average.

### Many steps: Chapman–Kolmogorov

The probability of going from $i$ to $j$ in two steps sums over the intermediate state: $p^{(2)}_{ij}=\sum_kp_{ik}\,p_{kj}$, which is the $(i,j)$ entry of $\mathbf{P}^2$. In general, the **Chapman–Kolmogorov equations** say

$$
p^{(m+n)}_{ij}=\sum_kp^{(m)}_{ik}\,p^{(n)}_{kj},\qquad\text{so}\qquad\mathbf{P}^{(n)}=\mathbf{P}^n.
$$

If $\boldsymbol\mu_n$ is the distribution of $X_n$, written as a row vector, then by the law of total probability $\mu_{n+1}(j)=\sum_i\mu_n(i)\,p_{ij}$, that is,

$$
\boldsymbol\mu_{n+1}=\boldsymbol\mu_n\mathbf{P},\qquad\boldsymbol\mu_n=\boldsymbol\mu_0\mathbf{P}^n.
$$

Studying a Markov chain therefore means studying powers of a matrix. Figure 1b shows $\boldsymbol\mu_n$ for the weather chain from two different starting days. After about a week the two starts are indistinguishable: the chain has forgotten where it began, and the distribution has settled at $(6,4,3)/13$. Explaining this behaviour is the goal of most of this lecture.

![[Probability and Linear Algebra/images/markov-chains-weather.svg]]
*Figure 1. (a) The weather chain as a transition diagram; each arrow is labelled with its transition probability. (b) The distribution of the weather on day $n$, starting from a sunny day (solid) or a rainy day (dashed). Both converge to the same limit, $(6,4,3)/13\approx(0.46,0.31,0.23)$ (thin lines).*

## First-step analysis: hitting probabilities and times

Many questions about a chain take the form "starting from $i$, what is the probability of reaching the set $A$ (before something else), and how long does it take?". [[Random Variables#The law of total expectation|First-step analysis]] answers them: condition on the first step, and use the Markov property to recognize the same question asked from the new state.

Let $h_i=\mathbb{P}_i(\text{the chain ever reaches }A)$. If $i\in A$ then $h_i=1$. Otherwise the chain moves to some $j$ with probability $p_{ij}$, and from there the question starts afresh:

$$
h_i=\sum_jp_{ij}\,h_j\quad(i\notin A),\qquad h_i=1\quad(i\in A).
\tag{1}
$$

Likewise, the expected time $m_i$ to reach $A$ satisfies $m_i=0$ on $A$ and $m_i=1+\sum_jp_{ij}\,m_j$ off $A$: one step is used up, plus the expected remaining time from wherever the chain lands. These are linear equations. When they have several solutions, the probabilities are the smallest non-negative one.[^norris1997]

> [!example] Example: gambler's ruin
> A gambler with $i$ dollars bets 1 dollar at a time, winning with probability $p$ and losing with probability $q=1-p$, until reaching $N$ dollars or going broke. Their fortune is a Markov chain on $\{0,\dots,N\}$ with absorbing ends. Let $h_i$ be the probability of reaching $N$ before 0. By (1), $h_0=0$, $h_N=1$, and
>
> $$
> h_i=p\,h_{i+1}+q\,h_{i-1},\qquad0<i<N.
> $$
>
> Rewrite this as $p(h_{i+1}-h_i)=q(h_i-h_{i-1})$: the increments form a geometric sequence with ratio $r=q/p$. Summing the increments and using $h_N=1$,
>
> $$
> h_i=\frac{1-r^i}{1-r^N}\quad(p\ne\tfrac12),\qquad h_i=\frac iN\quad(p=\tfrac12).
> $$
>
> In a fair game the chance of doubling your money is $1/2$, as it must be, since the expected fortune never changes. But a tiny edge against you is devastating (Figure 2a). With $p=0.49$, a gambler starting with 50 dollars reaches 100 only **12%** of the time. The edge of 2% per bet compounds over the roughly 1900 bets the game lasts. A fair game from 50 lasts $i(N-i)=2500$ bets on average.

![[Probability and Linear Algebra/images/markov-chains-gamblers-ruin.svg]]
*Figure 2. Gambler's ruin with \$1 bets and a target of \$100. (a) The probability of reaching 100 before 0, against the starting fortune. A fair game gives a straight line; a 49% or 47% chance per bet makes success unlikely unless you start close to the target. (b) Four fair games starting from 50, each eventually absorbed at 0 or 100.*

In matrix form, let $\mathbf{Q}$ be $\mathbf{P}$ restricted to the states outside $A$, and $\mathbf{b}$ the one-step probabilities of jumping into $A$. Then (1) reads $\mathbf{h}=\mathbf{Q}\mathbf{h}+\mathbf{b}$, so $\mathbf{h}=(\mathbf{I}-\mathbf{Q})^{-1}\mathbf{b}$, and the expected times are $\mathbf{m}=(\mathbf{I}-\mathbf{Q})^{-1}\mathbf{1}$. The matrix $(\mathbf{I}-\mathbf{Q})^{-1}=\mathbf{I}+\mathbf{Q}+\mathbf{Q}^2+\cdots$ is called the **fundamental matrix**: its $(i,j)$ entry is the expected number of visits to $j$ before absorption, starting from $i$. The same equations, with rewards added, are the [[Introduction to Reinforcement Learning|Bellman equations]] of reinforcement learning, and [[Dynamic Programming|policy evaluation]] solves exactly this kind of linear system.

> [!example]- Code: gambler's ruin by solving the linear system
> ```python
> import numpy as np
>
> def gamblers_ruin(N, p):
>     """Win probabilities h and expected durations m for starting fortunes 1..N-1."""
>     n = N - 1                                  # interior states 1..N-1
>     Q = np.zeros((n, n)); b = np.zeros(n)
>     for k in range(n):                         # state i = k + 1
>         if k + 1 < n: Q[k, k + 1] = p          # win a bet
>         else: b[k] = p                         # ... and reach N
>         if k - 1 >= 0: Q[k, k - 1] = 1 - p     # lose a bet (reaching 0 contributes 0)
>     I = np.eye(n)
>     h = np.linalg.solve(I - Q, b)              # h = Q h + b
>     m = np.linalg.solve(I - Q, np.ones(n))     # m = 1 + Q m
>     return h, m
>
> h, m = gamblers_ruin(100, 0.49)
> print(h[49], m[49])                            # start at 50: about 0.119 and 1904 bets
> h, m = gamblers_ruin(100, 0.5)
> print(h[49], m[49])                            # fair game: 0.5 and 50 * 50 = 2500 bets
> ```

### Stopping times and the strong Markov property

First-step analysis restarts the chain at a fixed time, after one step. We often want to restart it at a **random** time: the first time it hits a state, say. That is allowed, as long as the random time does not peek into the future.

A random time $T$ is a **stopping time** if, for every $n$, whether $T=n$ can be decided from $X_0,\dots,X_n$ alone. "The first visit to state $j$" is a stopping time: at each step we can tell whether it has happened. "The last visit to $j$" is not, and neither is "the step just before the second visit to $j$": both require knowing the future.

> [!note] Strong Markov property
> If $T$ is a stopping time and $X_T=i$, then the chain after $T$, $(X_T,X_{T+1},\dots)$, is a Markov chain with transition matrix $\mathbf{P}$ started from $i$, and it is independent of $X_0,\dots,X_T$.

At a stopping time the chain **statistically restarts**. The proof splits on the value of $T$ and applies the ordinary Markov property at each fixed time.[^kumar2012] The condition matters. At the step just before the second visit to $j$, the next state is $j$ with probability 1, not with probability $p_{ij}$.

## Classifying states

### Communicating classes

State $j$ is **accessible** from $i$, written $i\to j$, if the chain can get from $i$ to $j$: $p^{(n)}_{ij}>0$ for some $n\ge0$. States $i$ and $j$ **communicate** if $i\to j$ and $j\to i$. Communication is an equivalence relation, so it splits the states into **communicating classes**: groups of states that can all reach each other. These can be read off the transition diagram.
- A class is **closed** if the chain can never leave it.
- A chain with a single class is **irreducible**: every state can reach every other.

In Figure 3, $\{1,2\}$ is a class that leaks into the two closed classes $\{3,4\}$ and $\{5,6\}$. Once the chain leaves $\{1,2\}$ it can never come back.

![[Probability and Linear Algebra/images/markov-chains-classes.svg]]
*Figure 3. A chain with three communicating classes. $\{1,2\}$ is open: the chain eventually leaves it for good, so its states are transient. $\{3,4\}$ and $\{5,6\}$ are closed and finite, so their states are positive recurrent. The class $\{5,6\}$ alternates deterministically between its two states: it has period 2.*

### Recurrence and transience

Let $f_{ii}=\mathbb{P}_i(\text{the chain ever returns to }i)$. State $i$ is
- **recurrent** if $f_{ii}=1$: starting from $i$, the chain returns for sure;
- **transient** if $f_{ii}<1$: there is a positive probability of never coming back.

The strong Markov property makes this dichotomy sharp. Every time the chain returns to $i$, it restarts, and has the same probability $f_{ii}$ of returning yet again. So the number of returns $M_i$ is [[Random Variables#Geometric and memorylessness|geometric]]:

$$
\mathbb{P}_i(M_i=m)=f_{ii}^m\,(1-f_{ii}),\qquad\mathbb{E}_i[M_i]=\frac{f_{ii}}{1-f_{ii}}.
$$

A transient state is visited only finitely often, a geometric number of times. A recurrent state ($f_{ii}=1$) is visited **infinitely often**, with probability 1. There is no middle ground.

Writing $M_i=\sum_{n\ge1}\mathbb{1}\{X_n=i\}$ and taking expectations gives a practical test:

$$
i\text{ is recurrent}\iff\mathbb{E}_i[M_i]=\sum_{n=1}^{\infty}p^{(n)}_{ii}=\infty.
\tag{2}
$$

Recurrence is a **class property**: in a communicating class, either all states are recurrent or all are transient. Two useful consequences:[^kumar2012]
- An **open** class is transient: there is a way out with no way back, so each visit risks being the last.
- A **finite closed** class is recurrent: the chain is trapped among finitely many states, so it must keep revisiting some of them, and by the class property all of them.

Recurrent states come in two kinds. Let $\tau_i$ be the time to return to $i$.
- $i$ is **positive recurrent** if the mean return time $\mathbb{E}_i[\tau_i]$ is finite.
- $i$ is **null recurrent** if the chain returns for sure but $\mathbb{E}_i[\tau_i]=\infty$: the returns are certain, but they take longer and longer.

Every recurrent state of a finite chain is positive recurrent. Null recurrence needs infinitely many states.

### Random walks: a drunk man finds his way home, a drunk bird may not

The simple symmetric random walk on the integers moves left or right with probability $\tfrac12$ each. To be back at 0 after $2n$ steps, exactly $n$ of them must go right, so

$$
p^{(2n)}_{00}=\binom{2n}{n}2^{-2n}\approx\frac{1}{\sqrt{\pi n}},
$$

using Stirling's formula. The sum $\sum_n1/\sqrt n$ diverges, so by (2) the walk is **recurrent**: it returns to 0 infinitely often (Figure 4a). It is null recurrent, though. The mean return time is infinite, and a walker can wander very far before coming back.

In $d$ dimensions the walk moves to one of its $2d$ neighbours at random. Being back at the origin needs every coordinate to be back, and the return probability decays like $n^{-d/2}$. The sum diverges for $d=1$ and $d=2$, but converges for $d\ge3$ (Figure 4b). This is **Pólya's theorem**:[^polya1921] the symmetric random walk is recurrent in one and two dimensions and transient in three or more. In three dimensions the probability of ever returning is only about 0.34. Kakutani summarized it as: a drunk man will find his way home, but a drunk bird may get lost forever.

![[Probability and Linear Algebra/images/markov-chains-random-walk.svg]]
*Figure 4. Recurrence of random walks. (a) Three symmetric walks on the integers: each keeps crossing 0, however far it wanders. (b) The expected number of returns to the origin up to time $2n$, $\sum_mp^{(2m)}_{00}$. It grows without bound in one dimension (like $\sqrt n$) and in two (like $\log n$), but levels off below 0.52 in three, so the walk is transient there.*

A **biased** walk on the integers ($p\ne\tfrac12$) is transient even in one dimension: by the law of large numbers it drifts off to $\pm\infty$ at speed $|2p-1|$.

## The stationary distribution

### Balance

A distribution $\boldsymbol\pi$ over states is **stationary** (or invariant) if

$$
\boldsymbol\pi=\boldsymbol\pi\mathbf{P},\qquad\text{that is,}\qquad\pi_j=\sum_i\pi_i\,p_{ij}\ \text{ for all }j,\qquad\sum_j\pi_j=1.
\tag{3}
$$

If the chain starts in $\boldsymbol\pi$, then $\boldsymbol\mu_1=\boldsymbol\pi\mathbf{P}=\boldsymbol\pi$, and it stays in $\boldsymbol\pi$ forever. Picture a large crowd of independent walkers, a fraction $\pi_i$ of them at each state. In one step, the flow of walkers *into* $j$ is $\sum_i\pi_ip_{ij}$, and (3) says that it exactly replaces the walkers at $j$. Individuals keep moving, but the crowd's distribution does not change. Algebraically, $\boldsymbol\pi$ is a left eigenvector of $\mathbf{P}$ with eigenvalue 1. Every stochastic matrix has eigenvalue 1, because $\mathbf{P}\mathbf{1}=\mathbf{1}$.

For the weather chain, solving (3) gives $\boldsymbol\pi=(6,4,3)/13$: in the long run, 46% of days are sunny. Figure 1b shows the distributions converging to exactly this.

> [!note] Theorem (existence and uniqueness)
> An irreducible chain has a stationary distribution if and only if it is positive recurrent. The stationary distribution is then unique, and
>
> $$
> \pi_i=\frac{1}{\mathbb{E}_i[\tau_i]},
> $$
>
> the reciprocal of the mean return time to $i$.

The formula $\pi_i=1/\mathbb{E}_i[\tau_i]$ is intuitive. If the chain comes back to $i$ every $\mathbb{E}_i[\tau_i]$ steps on average, it spends a fraction $1/\mathbb{E}_i[\tau_i]$ of its time there. For a null recurrent chain the return time is infinite, so every $\pi_i$ would be 0, and no stationary distribution exists. Kumar's proof builds $\boldsymbol\pi$ from long-run averages of the $n$-step probabilities and uses renewal theory.[^kumar2012]

### Detailed balance

Solving (3) means solving a linear system. Often there is a shortcut. Suppose $\boldsymbol\pi$ satisfies **detailed balance**:

$$
\pi_i\,p_{ij}=\pi_j\,p_{ji}\quad\text{for all }i,j.
\tag{4}
$$

Summing over $i$ gives $\sum_i\pi_ip_{ij}=\pi_j\sum_ip_{ji}=\pi_j$, which is (3). So detailed balance implies stationarity. It says more than balance: the flow from $i$ to $j$ equals the flow from $j$ to $i$ for every pair, not just the total flow into each state. A chain in detailed balance is **reversible**: in stationarity, a film of the chain looks statistically the same played backwards. Detailed balance is the design principle of [[#Aside: Markov chain Monte Carlo|MCMC]].

**Birth–death chains**, which only move one step up or down, always satisfy detailed balance, because the only flows are between neighbours. If the chain moves up from $n$ with probability $b_n$ and down from $n+1$ with probability $d_{n+1}$, then (4) gives $\pi_{n+1}=\pi_n\,b_n/d_{n+1}$, and $\boldsymbol\pi$ is a product.

> [!example] Example: a queue, and all three kinds of state
> Time is slotted. In each slot a customer arrives with probability $\lambda$, and, if the queue is not empty, the customer being served leaves with probability $\mu$, independently. The queue length is a birth–death chain on $\{0,1,2,\dots\}$. Above 0 it goes up with probability $b=\lambda(1-\mu)$ (an arrival and no departure) and down with probability $d=\mu(1-\lambda)$. By detailed balance, $\pi_n$ is proportional to $r^n$ for $n\ge1$, with
>
> $$
> r=\frac{b}{d}=\frac{\lambda(1-\mu)}{\mu(1-\lambda)}.
> $$
>
> The geometric series can be normalized only if $r<1$, which happens exactly when $\lambda<\mu$. Kumar works this example out in full, and it shows all three behaviours:[^kumar2012]
> - $\lambda<\mu$: **positive recurrent**. The queue keeps emptying, and its length has a geometric-like stationary distribution, with $\pi_0=1-\lambda/\mu$.
> - $\lambda=\mu$: **null recurrent**. The queue empties infinitely often, but the time between empty periods has infinite mean.
> - $\lambda>\mu$: **transient**. The queue grows without bound.
>
> Arrivals must come more slowly than services, however slightly, for a queue to be stable.

### Convergence and the ergodic theorem

Does $\boldsymbol\mu_n$ converge to $\boldsymbol\pi$, as in Figure 1b? One more obstacle remains: **periodicity**. In the class $\{5,6\}$ of Figure 3, the chain alternates $5,6,5,6,\dots$, so $\mathbb{P}(X_n=5)$ flips between 0 and 1 forever, although $\boldsymbol\pi=(\tfrac12,\tfrac12)$ is stationary. The **period** of a state is the greatest common divisor of its possible return times. A chain is **aperiodic** if every period is 1. In an irreducible chain, a single state with $p_{ii}>0$ makes the whole chain aperiodic.

> [!note] Convergence theorem
> If a chain is irreducible, aperiodic and positive recurrent, then for every starting state $i$ and every $j$,
>
> $$
> p^{(n)}_{ij}\to\pi_j\quad\text{as }n\to\infty.
> $$

The chain forgets its starting point. The rows of $\mathbf{P}^n$ all converge to the same row $\boldsymbol\pi$.

Even without aperiodicity, the **time averages** converge. This is the **ergodic theorem**, the law of large numbers for Markov chains: if the chain is irreducible and positive recurrent, then for any bounded function $f$ of the state, with probability 1,

$$
\frac1n\sum_{t=0}^{n-1}f(X_t)\longrightarrow\sum_i\pi_i\,f(i).
$$

A single long run of the chain spends a fraction $\pi_i$ of its time in state $i$, whatever the starting point. The samples are dependent, unlike in [[Sample Statistics#The law of large numbers|Sample Statistics]], but the chain keeps restarting at its visits to a fixed state, and the stretches between visits are i.i.d. This is what justifies estimating expectations from a single MCMC run. Irreducibility matters here. In the chain of Figure 3, the long-run time averages depend on which closed class the chain falls into, so they are random, even though the chain has stationary distributions.

### Aside: mixing and the spectral gap

How fast does $\boldsymbol\mu_n$ approach $\boldsymbol\pi$? For a finite, irreducible, aperiodic, reversible chain, the answer is in the eigenvalues of $\mathbf{P}$. They are real, and they satisfy $1=\lambda_1>\lambda_2\ge\cdots\ge\lambda_{|\mathcal{S}|}>-1$. Expanding the starting distribution in the eigenvectors, the component along $\boldsymbol\pi$ stays, and every other component shrinks by its eigenvalue at each step. So the distance to stationarity decays like

$$
\|\boldsymbol\mu_n-\boldsymbol\pi\|\le C\,\lambda_*^n,\qquad\lambda_*=\max\big(|\lambda_2|,|\lambda_{|\mathcal{S}|}|\big).
$$

The **spectral gap** $1-\lambda_*$ sets the speed, and $1/(1-\lambda_*)$, the **relaxation time**, is roughly the number of steps the chain needs to forget where it started. Negative eigenvalues close to $-1$ signal near-periodic oscillation. The standard fix is a **lazy** chain, $\tfrac12(\mathbf{I}+\mathbf{P})$, which stays put half the time and has eigenvalues $\tfrac12(1+\lambda)\ge0$.

The spectral gap is small when the state space has a **bottleneck**: two well-connected regions joined by a narrow passage. The chain mixes quickly within each region but rarely crosses between them (Figure 5). This is exactly the difficulty MCMC faces with multimodal distributions, where the bottleneck is a low-probability valley between modes. Levin and Peres give a thorough treatment, including coupling arguments and cutoff phenomena.[^levin2017]

![[Probability and Linear Algebra/images/markov-chains-mixing.svg]]
*Figure 5. Two lazy random walks on 10 states. On a complete graph (blue) the distance to stationarity falls a hundredfold every few steps. On two 5-state clusters joined by a single edge (red), the chain is trapped in one cluster for a long time: $\lambda_*$ is close to 1, and the decay is slow. The dotted lines are $\lambda_*^t$.*

## PageRank: the stationary distribution of the web

How should a search engine rank web pages? Brin and Page's idea was to imagine a **random surfer** who clicks a uniformly random link on the current page, again and again.[^brin1998]<sup>,</sup>[^page1999] A page is important if the surfer spends a lot of time on it. That happens if many pages link to it, and especially if important pages do. The importance of a page is its probability in the stationary distribution of the surfer's Markov chain.

The raw link chain has problems. Some pages have no outgoing links, and groups of pages may link only among themselves, so the chain is neither irreducible nor aperiodic. The fix is **damping**. At each step, with probability $\alpha$ (typically 0.85) the surfer follows a random link, and otherwise jumps to a page chosen uniformly at random:

$$
\mathbf{G}=\alpha\,\mathbf{P}+(1-\alpha)\frac{1}{N}\mathbf{1}\mathbf{1}^\top.
$$

Every entry of $\mathbf{G}$ is positive, so the chain is irreducible and aperiodic, and the convergence theorem gives a unique PageRank vector $\mathbf{r}=\mathbf{r}\mathbf{G}$. It is computed by **power iteration**, $\mathbf{r}\leftarrow\mathbf{r}\mathbf{G}$, which converges geometrically: the teleportation guarantees $|\lambda_2|\le\alpha$. Figure 6 shows a small example. Page C has the highest rank because three pages link to it, and page A is second because C, the most important page, links only to A.

![[Probability and Linear Algebra/images/markov-chains-pagerank.svg]]
*Figure 6. PageRank on a six-page web with damping $\alpha=0.85$. Circle areas grow with the rank (printed). E and F link to each other and to D, but nobody links to them, so they keep only the teleportation probability and what they pass to each other.*

> [!example]- Code: stationary distributions and PageRank
> ```python
> import numpy as np
>
> P = np.array([[0.7, 0.2, 0.1],
>               [0.3, 0.4, 0.3],
>               [0.2, 0.4, 0.4]])
> w, V = np.linalg.eig(P.T)                      # pi P = pi  <=>  P^T pi^T = pi^T
> pi = np.real(V[:, np.argmin(np.abs(w - 1))]); pi /= pi.sum()
> print(pi)                                      # [0.4615 0.3077 0.2308] = (6, 4, 3) / 13
>
> def pagerank(links, n, alpha=0.85, tol=1e-12):
>     """links[i] = pages that page i links to. Power iteration on the damped chain."""
>     P = np.zeros((n, n))
>     for i in range(n):
>         if links.get(i): P[i, links[i]] = 1 / len(links[i])
>         else: P[i] = 1 / n                     # a page with no links: jump anywhere
>     G = alpha * P + (1 - alpha) / n            # teleport with probability 1 - alpha
>     r = np.ones(n) / n
>     while True:
>         r_new = r @ G
>         if np.abs(r_new - r).sum() < tol:
>             return r_new
>         r = r_new
>
> links = {0: [1, 2], 1: [2], 2: [0], 3: [2], 4: [3, 5], 5: [3, 4]}
> print(pagerank(links, 6).round(3))             # [0.329 0.165 0.357 0.062 0.043 0.043]
> ```

## Aside: Markov chain Monte Carlo

So far we were given a chain and asked for its stationary distribution. **Markov chain Monte Carlo** (MCMC) reverses the question: given a distribution $\pi$ we want to sample from, **design** a chain whose stationary distribution is $\pi$, run it for a long time, and use its states as samples. By the ergodic theorem, time averages along the run estimate expectations under $\pi$.

Why not sample $\pi$ directly? Typically $\pi$ is known only up to a normalizing constant, $\pi(x)=\tilde\pi(x)/Z$ with $Z$ intractable, and it lives in many dimensions. Bayesian posteriors are like this ($Z$ is the evidence), and so are the Boltzmann distributions of [[Energy-Based Models]]. [[Random Variables#Rejection sampling|Rejection sampling]] needs an envelope, which in high dimensions accepts almost nothing. MCMC needs only ratios $\pi(y)/\pi(x)=\tilde\pi(y)/\tilde\pi(x)$, in which $Z$ cancels, and it takes small local steps.

### Metropolis–Hastings

> [!example] Algorithm: Metropolis–Hastings
> Choose a proposal distribution $q(y\mid x)$, for example $y=x+$ Gaussian noise. Start anywhere, and repeat:
> 1. Propose $y\sim q(\cdot\mid x)$ from the current state $x$.
> 2. Compute the acceptance probability $a(x,y)=\min\Big(1,\dfrac{\tilde\pi(y)\,q(x\mid y)}{\tilde\pi(x)\,q(y\mid x)}\Big)$.
> 3. With probability $a(x,y)$ move to $y$; otherwise stay at $x$. Either way, record the current state as the next sample.

For a symmetric proposal, $q(x\mid y)=q(y\mid x)$, the acceptance probability is simply $\min(1,\tilde\pi(y)/\tilde\pi(x))$: uphill moves are always accepted, and downhill moves sometimes. This is the original **Metropolis** algorithm of 1953,[^metropolis1953] generalized by Hastings to asymmetric proposals.[^hastings1970] Rejected proposals are not discarded: staying put is part of the chain, and it is what gives high-probability states their extra weight.

> [!proof]- Why the stationary distribution is $\pi$
> We check detailed balance (4). For $x\ne y$, the chain moves from $x$ to $y$ with probability $q(y\mid x)\,a(x,y)$. Then
>
> $$
> \pi(x)\,q(y\mid x)\,a(x,y)=\min\big(\pi(x)\,q(y\mid x),\ \pi(y)\,q(x\mid y)\big),
> $$
>
> which is symmetric in $x$ and $y$. So $\pi(x)\,p(x\to y)=\pi(y)\,p(y\to x)$, and $\pi$ is stationary. If the proposal can reach everywhere and the chain is aperiodic (rejections make it so), the convergence and ergodic theorems apply.

Figure 7a shows a random-walk Metropolis chain on a two-bump density. In practice:
- **Step size.** Small steps are almost always accepted but explore slowly. Large steps are mostly rejected, and the chain stays put. A moderate acceptance rate, somewhere between a fifth and a half, is a common target.
- **Burn-in.** Early samples still remember the starting point and are discarded.
- **Correlation.** Successive samples are dependent, so $n$ MCMC samples carry less information than $n$ independent ones. The **effective sample size** accounts for this.
- **Multimodality.** Crossing a low-probability valley between modes requires a run of unlikely steps. This is the bottleneck of the mixing aside, and the chain can be stuck in one mode for a very long time.

### Gibbs sampling

For a multivariate target $\pi(x_1,\dots,x_D)$, **Gibbs sampling** updates one coordinate at a time, drawing it from its **full conditional** given all the others:[^geman1984]

$$
x_k\sim\pi(x_k\mid x_1,\dots,x_{k-1},x_{k+1},\dots,x_D),\qquad k=1,\dots,D,\ \text{ then repeat}.
$$

Each update is a Metropolis–Hastings step whose proposal is the full conditional itself. The acceptance probability works out to exactly 1, so nothing is ever rejected. Gibbs sampling is attractive whenever the full conditionals are easy to sample, even if the joint is not. Examples are conditionally conjugate Bayesian models, and the restricted Boltzmann machines of [[Energy-Based Models#Restricted Boltzmann machines|Energy-Based Models]], where all the hidden units are conditionally independent given the visible ones, and vice versa.

Its weakness is correlation. The moves are parallel to the coordinate axes, so on a long, thin, diagonal ridge it can only zig-zag in small steps (Figure 7b). For a bivariate Gaussian with correlation $\rho$, each sweep shrinks the distance to the mean only by a factor $\rho^2$, so mixing slows down as $\rho\to1$. Gradient-based samplers, such as the Langevin dynamics of [[Score-Based Models]] and Hamiltonian Monte Carlo, use the shape of $\log\pi$ to move along such ridges.

![[Probability and Linear Algebra/images/markov-chains-mcmc.svg]]
*Figure 7. (a) Random-walk Metropolis–Hastings on a mixture of two Gaussians (contours), with weights 0.6 and 0.4. The first 150 steps are drawn in red. The samples cover both bumps in the right proportions, and the chain crosses between them only occasionally. (b) Gibbs sampling on a Gaussian with correlation 0.9: the chain moves parallel to the axes and zig-zags slowly along the ridge.*

> [!example]- Code: Metropolis–Hastings and Gibbs sampling
> ```python
> import numpy as np
> from scipy import stats
> rng = np.random.default_rng(0)
>
> def log_target(x):                             # unnormalized log density: two Gaussian bumps
>     a = stats.multivariate_normal([-2, -1], [[1, 0.5], [0.5, 1]]).logpdf(x)
>     b = stats.multivariate_normal([2, 1.5], [[0.6, -0.2], [-0.2, 0.6]]).logpdf(x)
>     return np.logaddexp(np.log(0.6) + a, np.log(0.4) + b)
>
> def metropolis(log_p, x0, n_steps, step):
>     x = np.asarray(x0, float); lp = log_p(x)
>     samples, accepted = np.empty((n_steps, x.size)), 0
>     for t in range(n_steps):
>         y = x + step * rng.standard_normal(x.size)       # symmetric random-walk proposal
>         ly = log_p(y)
>         if np.log(rng.random()) < ly - lp:               # accept with prob min(1, p(y)/p(x))
>             x, lp = y, ly
>             accepted += 1
>         samples[t] = x                                   # a rejection repeats the old state
>     return samples, accepted / n_steps
>
> S, acc = metropolis(log_target, [0.0, 0.0], 100_000, step=1.0)
> print(acc, np.mean(S[1000:, 0] > 0))           # acceptance rate; fraction in the right bump (target 0.4)
>
> # Gibbs sampling for a standard bivariate Gaussian with correlation rho
> rho, n_steps = 0.9, 50_000
> x = np.zeros(2); out = np.empty((n_steps, 2))
> for t in range(n_steps):
>     x[0] = rho * x[1] + np.sqrt(1 - rho**2) * rng.standard_normal()   # x1 | x2
>     x[1] = rho * x[0] + np.sqrt(1 - rho**2) * rng.standard_normal()   # x2 | x1
>     out[t] = x
> print(out.mean(0), np.corrcoef(out.T)[0, 1])   # about [0, 0] and 0.9
> ```

## Continuous-time Markov chains

Everything so far moved in discrete steps. Many systems change at **random moments** in continuous time: customers arrive, molecules react, a machine fails. A **continuous-time Markov chain** (CTMC) is a process $X(t)$, $t\ge0$, on a countable state space with the Markov property at every time: given $X(s)$, the future after $s$ is independent of the past before $s$. Kumar's Chapter 4 develops the theory rigorously.[^kumar2012]

### From steps to rates

The Markov property alone dictates how long the chain stays in a state. Given that it has been in state $i$ for a while, the remaining time must not depend on how long, so the **holding time is memoryless**, and therefore [[Random Variables#Exponential and memorylessness|exponential]]. So a CTMC works like this:
1. In state $i$, wait an Exponential$(q_i)$ time, where $q_i$ is the **rate of leaving** $i$.
2. Jump to $j\ne i$ with probability $r_{ij}$. The jumps form an ordinary discrete-time chain, the **jump chain**.

An equivalent and more symmetric picture uses **competing clocks**. In state $i$, start an independent exponential clock for every possible destination $j$, with rate $q_{ij}=q_i\,r_{ij}$. The first clock to ring decides where the chain goes and when. This works because the minimum of independent exponentials is exponential with the summed rate, $q_i=\sum_{j\ne i}q_{ij}$, and clock $j$ rings first with probability $q_{ij}/q_i$.

The rates are collected in the **generator** (or rate matrix) $\mathbf{Q}$, with $q_{ij}\ge0$ off the diagonal and $q_{ii}=-q_i$, so that every row sums to **zero**. Its meaning is local: for a short time $h$,

$$
\mathbb{P}\big(X(t+h)=j\mid X(t)=i\big)=q_{ij}\,h+o(h)\quad(j\ne i).
$$

Familiar examples:
- The [[Random Variables#The Poisson process|Poisson process]] is a CTMC on $\{0,1,2,\dots\}$ that only moves up, with $q_{n,n+1}=\lambda$.
- The **M/M/1 queue**: customers arrive at rate $\lambda$ and are served at rate $\mu$, so $q_{n,n+1}=\lambda$ and $q_{n,n-1}=\mu$ for $n\ge1$ (Figure 8a). It is the continuous-time version of the slotted queue above.
- A **two-state** chain that switches $0\to1$ at rate $\alpha$ and $1\to0$ at rate $\beta$, for instance a machine that breaks and gets repaired.

### Transition matrices and the Kolmogorov equations

Let $\mathbf{P}(t)$ be the matrix of probabilities $p_{ij}(t)=\mathbb{P}(X(t)=j\mid X(0)=i)$. Chapman–Kolmogorov still holds, $\mathbf{P}(t+s)=\mathbf{P}(t)\,\mathbf{P}(s)$, and for small $h$, $\mathbf{P}(h)=\mathbf{I}+h\,\mathbf{Q}+o(h)$. Together these give a differential equation, the **Kolmogorov forward equation**, and its solution:

$$
\frac{d}{dt}\mathbf{P}(t)=\mathbf{P}(t)\,\mathbf{Q},\qquad\mathbf{P}(t)=e^{t\mathbf{Q}}=\sum_{k=0}^{\infty}\frac{t^k\mathbf{Q}^k}{k!}.
\tag{5}
$$

The matrix exponential replaces the matrix power $\mathbf{P}^n$: a CTMC is "$\mathbf{P}^n$ with a continuous $n$". (The backward equation $\mathbf{P}'(t)=\mathbf{Q}\,\mathbf{P}(t)$ holds as well.) A distribution over states evolves by $\frac{d}{dt}\boldsymbol\mu(t)=\boldsymbol\mu(t)\,\mathbf{Q}$, sometimes called the **master equation**: probability flows into $j$ at rate $\sum_{i\ne j}\mu_i\,q_{ij}$ and out of it at rate $\mu_j\,q_j$.

For the two-state chain, (5) can be solved by hand:

$$
p_{00}(t)=\frac{\beta}{\alpha+\beta}+\frac{\alpha}{\alpha+\beta}\,e^{-(\alpha+\beta)t}.
$$

The chain forgets its start exponentially fast, at rate $\alpha+\beta$, which is the nonzero eigenvalue of $-\mathbf{Q}$ (Figure 8b). In continuous time, the spectral gap of the generator plays the role that $1-\lambda_*$ played for $\mathbf{P}$.

**Stationarity** becomes $\boldsymbol\pi\,\mathbf{Q}=\mathbf{0}$: in every state the rate of flowing in balances the rate of flowing out, $\pi_j\,q_j=\sum_{i\ne j}\pi_i\,q_{ij}$. Detailed balance reads $\pi_i\,q_{ij}=\pi_j\,q_{ji}$. For the M/M/1 queue it gives $\pi_n=(1-\rho)\rho^n$ with $\rho=\lambda/\mu<1$, a geometric distribution with mean queue length $\rho/(1-\rho)$. At $\rho=0.8$ the queue holds 4 customers on average, and as $\rho\to1$ it blows up. Recurrence, classes and the ergodic theorem carry over from the discrete case, with time averages $\frac1T\int_0^Tf(X(t))\,dt$.

![[Probability and Linear Algebra/images/markov-chains-ctmc.svg]]
*Figure 8. Continuous-time Markov chains. (a) A simulated M/M/1 queue: the length changes by $\pm1$ at exponentially distributed times. (b) A two-state chain with rates 1 and 0.5. $p_{00}(t)$ and $p_{10}(t)$ converge exponentially fast, at rate 1.5, to the stationary probability $\pi_0=1/3$.*

### Simulating a continuous-time chain

The "hold, then jump" description is an exact simulation algorithm, known in chemistry as the **Gillespie algorithm**.[^gillespie1977] In state $i$, draw a holding time from Exponential$(q_i)$, then pick the next state with probabilities $q_{ij}/q_i$, and repeat. Figure 8a was made this way.

A second method, **uniformization**, connects continuous and discrete time. Pick a rate $\bar q\ge\max_iq_i$ and define the stochastic matrix $\mathbf{P}_{\bar q}=\mathbf{I}+\mathbf{Q}/\bar q$. Then the CTMC is the discrete chain $\mathbf{P}_{\bar q}$ run at the event times of a Poisson process with rate $\bar q$:

$$
e^{t\mathbf{Q}}=\sum_{k=0}^{\infty}e^{-\bar qt}\frac{(\bar qt)^k}{k!}\,\mathbf{P}_{\bar q}^{\,k}.
$$

The self-loops of $\mathbf{P}_{\bar q}$ are "rejected" jump attempts, so this is the [[Random Variables#Simulating a Hawkes process: Ogata's thinning|thinning]] idea once more: propose events at a fast constant rate and keep each with probability proportional to the true rate.

> [!example]- Code: Gillespie simulation and the matrix exponential
> ```python
> import numpy as np
> from scipy.linalg import expm
> rng = np.random.default_rng(0)
>
> def gillespie_queue(lam, mu, T):
>     """M/M/1 queue: arrivals at rate lam, services at rate mu while the queue is non-empty."""
>     t, x, area = 0.0, 0, 0.0
>     while t < T:
>         rates = np.array([lam, mu if x > 0 else 0.0])
>         total = rates.sum()
>         hold = rng.exponential(1 / total)                # holding time ~ Exponential(total rate)
>         area += x * min(hold, T - t); t += hold
>         x += 1 if rng.random() < rates[0] / total else -1   # jump chain: next event in proportion to rates
>     return area / T                                      # time-average queue length
>
> print(gillespie_queue(0.8, 1.0, 200_000))                # theory: rho / (1 - rho) = 4
>
> Q = np.array([[-1.0, 1.0],                               # generator of a two-state chain
>               [0.5, -0.5]])
> print(expm(2.0 * Q))                                     # P(t) = e^{tQ} at t = 2; rows tend to pi = (1/3, 2/3)
> ```

### Towards diffusion models and state space models

Two families of modern models are built on continuous-time Markov processes, and the ideas above are their foundations.

**Continuous states: Brownian motion.** Take the symmetric random walk, speed it up to $n$ steps per unit time, and shrink each step to $1/\sqrt n$. By the [[Sample Statistics#The central limit theorem|central limit theorem]], the position at time $t$ becomes $\mathcal{N}(0,t)$, and as $n\to\infty$ the whole path converges to **Brownian motion** $W(t)$ (Figure 9a). Brownian motion is a Markov process with continuous states and continuous paths, and independent Gaussian increments. More general **diffusion processes** are written as stochastic differential equations, $d\mathbf{x}=\mathbf{f}(\mathbf{x},t)\,dt+g(t)\,d\mathbf{W}$. For them, the master equation $\dot{\boldsymbol\mu}=\boldsymbol\mu\mathbf{Q}$ becomes a partial differential equation for the density, the **Fokker–Planck equation**. In one dimension it reads

$$
\frac{\partial p}{\partial t}=-\frac{\partial}{\partial x}\big(f\,p\big)+\frac{g^2}{2}\,\frac{\partial^2p}{\partial x^2}.
$$

The drift $f$ moves probability around, and the noise $g$ spreads it out.

**The forward process of a diffusion model is a Markov chain.** In a denoising diffusion probabilistic model (DDPM),[^ho2020] data $\mathbf{x}_0$ is corrupted step by step:

$$
\mathbf{x}_t=\sqrt{1-\beta_t}\,\mathbf{x}_{t-1}+\sqrt{\beta_t}\,\boldsymbol\epsilon_t,\qquad\boldsymbol\epsilon_t\sim\mathcal{N}(\mathbf{0},\mathbf{I}),\qquad t=1,\dots,T.
$$

This is a Markov chain with Gaussian transitions, and the ideas of this lecture explain how it is designed:
- **Its stationary distribution is $\mathcal{N}(\mathbf{0},\mathbf{I})$.** If $\mathbf{x}_{t-1}\sim\mathcal{N}(\mathbf{0},\mathbf{I})$, then $\mathbf{x}_t$ has variance $(1-\beta_t)+\beta_t=1$.
- **It mixes.** Unrolling the chain, $\mathbf{x}_t$ given $\mathbf{x}_0$ is Gaussian with mean $\sqrt{\bar\alpha_t}\,\mathbf{x}_0$ and covariance $(1-\bar\alpha_t)\mathbf{I}$, where $\bar\alpha_t=\prod_{s\le t}(1-\beta_s)$. As $\bar\alpha_t\to0$, the chain forgets the data and reaches the stationary noise distribution, whatever the starting image (Figure 9b).
- **Generation runs the chain backwards.** The reversed sequence $\mathbf{x}_T,\mathbf{x}_{T-1},\dots,\mathbf{x}_0$ is also a Markov chain, and for small $\beta_t$ its transitions are approximately Gaussian. Their means depend on the score $\nabla\log p_t$, which is learned ([[Score-Based Models]]).
- **In continuous time** the chain becomes the stochastic differential equation $d\mathbf{x}=-\tfrac12\beta(t)\,\mathbf{x}\,dt+\sqrt{\beta(t)}\,d\mathbf{W}$, a mean-reverting (Ornstein–Uhlenbeck) diffusion.[^song2021sde]
- **For discrete data** such as text, diffusion models use exactly the CTMCs of this section. A rate matrix $\mathbf{Q}$ on the vocabulary slowly corrupts tokens towards a uniform or "masked" stationary distribution, and its transition probabilities are matrix exponentials $e^{t\mathbf{Q}}$ as in (5).[^austin2021]<sup>,</sup>[^campbell2022]

![[Probability and Linear Algebra/images/markov-chains-diffusion.svg]]
*Figure 9. From random walks to diffusion models. (a) Random walks with $n$ steps of size $1/\sqrt n$ in unit time look more and more like a single continuous, rough path: Brownian motion. (b) The DDPM forward chain applied to two-cluster data (red, $t=0$) with the standard linear noise schedule. The clusters blur and merge, and by $t=1000$ the distribution is the stationary $\mathcal{N}(0,1)$ (dotted), with no memory of the data.*

**State space models.** A probabilistic **state space model** pairs a hidden Markov chain $\mathbf{z}_t$, which carries the memory, with observations $\mathbf{y}_t$ that depend only on the current state. With discrete states this is a hidden Markov model. With linear Gaussian dynamics it is the model of the **Kalman filter**, whose update step is the Gaussian conditioning of [[Joint Distributions#The multivariate Gaussian|Joint Distributions]] (the "signal in noise" example) applied once per time step. The deep **structured state space models** used for long sequences, such as S4 and Mamba, keep this structure but make it deterministic.[^gu2021]<sup>,</sup>[^gu2023] A hidden state follows a continuous-time linear system $\dot{\mathbf{h}}(t)=\mathbf{A}\mathbf{h}(t)+\mathbf{B}u(t)$, and it is discretized with a step $\Delta$ using the same matrix exponential, $\mathbf{h}_k=e^{\Delta\mathbf{A}}\mathbf{h}_{k-1}+(\cdots)\,u_k$, that turns a generator $\mathbf{Q}$ into $\mathbf{P}(\Delta)=e^{\Delta\mathbf{Q}}$. In both cases the current state summarizes everything about the past that matters for the future. That is the Markov idea, used as an architecture.

> [!example]- Code: the DDPM forward chain forgets the data
> ```python
> import numpy as np
> rng = np.random.default_rng(0)
> T = 1000
> beta = np.linspace(1e-4, 0.02, T)
> x0 = np.where(rng.random(200_000) < 0.5, -2.0, 2.0) + 0.3 * rng.standard_normal(200_000)
> x = x0.copy()
> for t in range(T):                                       # x_t = sqrt(1 - beta_t) x_{t-1} + sqrt(beta_t) eps
>     x = np.sqrt(1 - beta[t]) * x + np.sqrt(beta[t]) * rng.standard_normal(x.size)
> abar = np.prod(1 - beta)
> print(abar, x.mean(), x.var())                           # abar ~ 4e-5: x_T is (almost) N(0, 1)
> ```

## Key takeaways

- A Markov chain is a sequence in which the future is conditionally independent of the past given the present. It is specified by an initial distribution and a stochastic matrix $\mathbf{P}$; $n$-step probabilities are $\mathbf{P}^n$, and distributions evolve as $\boldsymbol\mu_{n+1}=\boldsymbol\mu_n\mathbf{P}$.
- First-step analysis gives linear equations for hitting probabilities and expected hitting times. In gambler's ruin, a small edge per bet makes ruin likely over a long game.
- Stopping times allow restarting the chain at random times (the strong Markov property). Communicating classes organize the states. A transient state is visited a geometric number of times; a recurrent one, infinitely often. Open classes are transient and finite closed classes positive recurrent. Random walks are recurrent in dimensions 1 and 2 only.
- An irreducible chain has a stationary distribution $\boldsymbol\pi=\boldsymbol\pi\mathbf{P}$ exactly when it is positive recurrent; then $\pi_i=1/\mathbb{E}_i[\tau_i]$. Detailed balance is a sufficient shortcut, and birth–death chains always satisfy it.
- Aperiodic chains converge to $\boldsymbol\pi$ from any start. Time averages converge to $\boldsymbol\pi$-averages regardless (the ergodic theorem). The speed is set by the spectral gap, and bottlenecks make it slow.
- PageRank is the stationary distribution of a damped random surfer, computed by power iteration. MCMC builds chains with a desired stationary distribution: Metropolis–Hastings needs only density ratios; Gibbs samples full conditionals and never rejects.
- A continuous-time chain holds each state for an exponential time and jumps by competing exponential clocks. Its generator $\mathbf{Q}$ has rows summing to zero, $\mathbf{P}(t)=e^{t\mathbf{Q}}$, and $\boldsymbol\pi\mathbf{Q}=\mathbf{0}$. Gillespie's algorithm simulates it exactly; uniformization reduces it to a discrete chain.
- Rescaled random walks become Brownian motion. The DDPM forward process is a Gaussian Markov chain whose stationary distribution is $\mathcal{N}(\mathbf{0},\mathbf{I})$; generation runs it backwards. State space models carry the past in a Markov state, discretized with the same matrix exponential.

## Where this shows up next

- [[Introduction to Reinforcement Learning]] adds actions and rewards to a Markov chain to get a Markov decision process. [[Dynamic Programming]] solves the first-step (Bellman) equations, and the convergence of value iteration is another contraction argument.
- [[Energy-Based Models]] use Gibbs sampling in Boltzmann machines and RBMs, and Langevin MCMC for deep EBMs. [[Score-Based Models]] sample with annealed Langevin dynamics, an MCMC method driven by the score.
- [[Score-Based Models]] reverses the diffusion of Figure 9b with learned scores; the planned lectures on denoising diffusion models and flow matching build on the forward Markov chain and its continuous-time limit.
- [[Matrix Factorization and Dimensionality Reduction]] covers the eigenvalues and power iteration behind stationary distributions, PageRank and mixing; spectral clustering in [[Clustering and Expectation Maximization]] looks for the same bottlenecks that slow a random walk down.

<!-- References: footnote definitions below render as the numbered "References" list at the bottom of the page -->

[^kumar2012]: Kumar, A. (2012). *Discrete Event Stochastic Processes: Lecture Notes for an Engineering Curriculum*, Chapters 2 and 4. Indian Institute of Science. [PDF](https://ece.iisc.ac.in/~anurag/books/anurag/spqt.pdf)
[^norris1997]: Norris, J. R. (1997). *Markov Chains*. Cambridge University Press. [doi:10.1017/CBO9780511810633](https://doi.org/10.1017/CBO9780511810633)
[^levin2017]: Levin, D. A., & Peres, Y. (2017). *Markov Chains and Mixing Times* (2nd ed.), with contributions by E. L. Wilmer. American Mathematical Society. [Free online edition](https://pages.uoregon.edu/dlevin/MARKOV/mcmt2e.pdf)
[^polya1921]: Pólya, G. (1921). *Über eine Aufgabe der Wahrscheinlichkeitsrechnung betreffend die Irrfahrt im Straßennetz*. Mathematische Annalen, 84, 149–160. [doi:10.1007/BF01458701](https://doi.org/10.1007/BF01458701)
[^brin1998]: Brin, S., & Page, L. (1998). *The anatomy of a large-scale hypertextual Web search engine*. Computer Networks and ISDN Systems, 30(1–7), 107–117. [doi:10.1016/S0169-7552(98)00110-X](https://doi.org/10.1016/S0169-7552(98)00110-X)
[^page1999]: Page, L., Brin, S., Motwani, R., & Winograd, T. (1999). *The PageRank Citation Ranking: Bringing Order to the Web*. Stanford InfoLab Technical Report. [link](http://ilpubs.stanford.edu:8090/422/)
[^metropolis1953]: Metropolis, N., Rosenbluth, A. W., Rosenbluth, M. N., Teller, A. H., & Teller, E. (1953). *Equation of State Calculations by Fast Computing Machines*. The Journal of Chemical Physics, 21(6), 1087–1092. [doi:10.1063/1.1699114](https://doi.org/10.1063/1.1699114)
[^hastings1970]: Hastings, W. K. (1970). *Monte Carlo sampling methods using Markov chains and their applications*. Biometrika, 57(1), 97–109. [doi:10.1093/biomet/57.1.97](https://doi.org/10.1093/biomet/57.1.97)
[^geman1984]: Geman, S., & Geman, D. (1984). *Stochastic Relaxation, Gibbs Distributions, and the Bayesian Restoration of Images*. IEEE Transactions on Pattern Analysis and Machine Intelligence, PAMI-6(6), 721–741. [doi:10.1109/TPAMI.1984.4767596](https://doi.org/10.1109/TPAMI.1984.4767596)
[^gillespie1977]: Gillespie, D. T. (1977). *Exact stochastic simulation of coupled chemical reactions*. The Journal of Physical Chemistry, 81(25), 2340–2361. [doi:10.1021/j100540a008](https://doi.org/10.1021/j100540a008)
[^ho2020]: Ho, J., Jain, A., & Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models*. NeurIPS. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)
[^song2021sde]: Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2021). *Score-Based Generative Modeling through Stochastic Differential Equations*. ICLR. [arXiv:2011.13456](https://arxiv.org/abs/2011.13456)
[^austin2021]: Austin, J., Johnson, D. D., Ho, J., Tarlow, D., & van den Berg, R. (2021). *Structured Denoising Diffusion Models in Discrete State-Spaces*. NeurIPS. [arXiv:2107.03006](https://arxiv.org/abs/2107.03006)
[^campbell2022]: Campbell, A., Benton, J., De Bortoli, V., Rainforth, T., Deligiannidis, G., & Doucet, A. (2022). *A Continuous Time Framework for Discrete Denoising Models*. NeurIPS. [arXiv:2205.14987](https://arxiv.org/abs/2205.14987)
[^gu2021]: Gu, A., Goel, K., & Ré, C. (2022). *Efficiently Modeling Long Sequences with Structured State Spaces*. ICLR. [arXiv:2111.00396](https://arxiv.org/abs/2111.00396)
[^gu2023]: Gu, A., & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)
