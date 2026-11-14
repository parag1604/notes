---
title: Stochastic Shortest Path Problems
tags:
  - reinforcement-learning
  - dynamic-programming
  - stochastic-shortest-path
  - value-iteration
  - policy-iteration
date: 2026-11-12
draft: false
prev: "[[Dynamic Programming]]"
next: "[[Stochastic Approximation]]"
---

> **Prerequisites:** [[Dynamic Programming]] (Bellman operators, policy evaluation, policy and value iteration), [[Introduction to Reinforcement Learning]] (episodic tasks, $V^\pi$, the Bellman equations) and [[Markov Chains]] (absorbing states, first-step analysis). [[Finite-Horizon Dynamic Programming]] helps for the $k$-stage interpretation of the operators.
>
> **Planning with a known model.** As in the previous two lectures, we assume the transition model $p(s'\mid s,a)$ and the rewards are **known**: this is planning. In practice they usually are not, which is why the earlier lectures learned from experience. The model can also be learned from data, see [[Model-Based Reinforcement Learning]].
>
> **Notation (local deviations).** Non-terminal states are $\mathcal{S}=\{1,\dots,n\}$ and the terminal state is $0$. Value vectors live in $\mathbb{R}^n$, with the convention $V(0)=0$. A stationary deterministic policy is written $\pi$, with $a=\pi(s)$. A general, possibly non-stationary, policy is a sequence of **decision rules** $\pi=(d_0,d_1,\dots)$, each $d_t$ a map from states to actions; a stationary policy is the special case $d_t=\pi$ for all $t$. Subscripts on $\pi$ are iteration counters (policy iteration). $\mathbf{1}$ is the all-ones vector, and inequalities between vectors hold componentwise.
>
> **TL;DR:** Many tasks end: a maze has an exit, a game has a final move. The natural objective is then the **total, undiscounted** reward until termination. Without $\gamma<1$ the Bellman operator is no longer a sup-norm contraction, and a policy that never terminates can have infinite value. **Stochastic shortest path (SSP)** theory replaces discounting by termination. If (A) some policy terminates surely from every state and (B) every policy that may not terminate collects $-\infty$, then $V^*$ is the unique solution of the Bellman equation, an optimal stationary policy exists, value iteration converges from any start, and policy iteration terminates. If *every* policy terminates, the Bellman operator is a contraction in a max-norm weighted by expected time to termination.

## Motivation

In the [[Dynamic Programming|previous lecture]] everything rested on one fact: with $\gamma<1$, the Bellman operators are $\gamma$-contractions in the max-norm. But many problems come with a natural end and no natural discount. A robot should reach the charging dock in as few steps as possible. A game ends with a win or a loss. A shortest path should be *short*, not "short after discounting". The objective we actually care about is the **expected total reward until the episode ends**. That is $\gamma=1$, which the discounted theory excludes.

Setting $\gamma=1$ breaks things in two ways.

1. **A policy may never end.** A robot that keeps bumping into a wall collects reward forever. Its total reward can be $-\infty$, $+\infty$, or a sum that doesn't converge at all.
2. **No contraction.** With $\gamma=1$ the operator $\mathcal{T}^\pi V=r_\pi+P^\pi V$ moves every component of $V$ by an *average* of the others. If $P^\pi$ never leaks probability to the terminal state, nothing shrinks, and the Banach argument has nothing to work with.

Stochastic shortest path problems[^bertsekastsitsiklis1991]<sup>,</sup>[^bertsekasvol2] are the clean framework for this. The idea is that **termination plays the role of discounting**. Probability that leaks into the terminal state is lost forever, much as a factor $\gamma$ shrinks the future. The price is that we must say exactly which policies terminate and what happens to those that don't. In return, all the results of the discounted case survive, though some with different proofs. Discounted problems even turn out to be a special case ([[Dynamic Programming#Discounting as termination: the SSP view|shown in the DP lecture]]).

---

## The setting

### States, rewards and the terminal state

We have finitely many non-terminal states $\mathcal{S}=\{1,\dots,n\}$, finitely many actions $\mathcal{A}(s)$ in each state, and one extra **terminal state** $0$. The terminal state is absorbing and pays nothing: $p(0\mid0,a)=1$ and $r(0,a,0)=0$. From a non-terminal state $s$, action $a$ leads to $s'\in\{0,1,\dots,n\}$ with probability $p(s'\mid s,a)$ and pays $r(s,a,s')$. There is no discount.

The **expected one-step reward** must include the transition *into* the terminal state:

$$
r(s,a)=\sum_{s'=0}^{n}p(s'\mid s,a)\,r(s,a,s').
$$

> [!warning] Don't drop the last reward
> It is tempting to write the Bellman update as $\sum_{s'=1}^{n}p(s'\mid s,a)\big[r(s,a,s')+V(s')\big]$, summing over non-terminal successors only. That silently drops the reward $p(0\mid s,a)\,r(s,a,0)$ of the final step. In a maze with $-1$ per step, it would claim that the step onto the exit is free. Either sum over $s'=0,\dots,n$ with $V(0)=0$, or use $r(s,a)$ as defined above.

### Policies and their values

A **stationary** policy $\pi$ picks $\pi(s)\in\mathcal{A}(s)$ in each state. A general policy $\pi=(d_0,d_1,\dots)$ may use a different decision rule $d_t$ at each time (a stationary one has $d_t=\pi$ for all $t$). Randomized and history-dependent policies won't help, for the reason given after Proposition 2.

The value of $\pi$ is its total expected reward. Since the limit may not exist when $\pi$ doesn't terminate, we take the pessimistic $\liminf$ of the $k$-stage returns:

$$
V^\pi(s)=\liminf_{k\to\infty}\;\mathbb{E}^\pi\Big[\sum_{t=0}^{k-1}r(s_t,a_t,s_{t+1})\;\Big|\;s_0=s\Big],\qquad V^*(s)=\sup_\pi V^\pi(s).
$$

Once the episode has ended, every later reward is $0$, so the sum effectively stops at termination. A **stationary** $\pi$ induces a Markov chain on $\{0,\dots,n\}$. Restricted to the non-terminal states, it has the transition matrix $P^\pi(s,s')=p(s'\mid s,\pi(s))$ for $s,s'\ge1$ and the reward vector $r_\pi(s)=r(s,\pi(s))$. The matrix $P^\pi$ is **substochastic**: its row sums $1-p(0\mid s,\pi(s))$ are at most $1$, and the missing mass is the probability of ending the episode in one step. As in [[Introduction to Reinforcement Learning#Matrix form and closed-form policy evaluation|the discounted case]], $\mathbb{E}^\pi[r_t\mid s_0=s]=\big((P^\pi)^t r_\pi\big)(s)$.

### Proper and improper policies

> [!note] Definition: proper policy
> A stationary policy $\pi$ is **proper** if, from every starting state, it reaches the terminal state with probability $1$. Otherwise it is **improper**.

Properness is a property of the Markov chain of $\pi$, not of its rewards. It has a useful finite-time characterization. Let

$$
\rho_\pi=\max_{s\in\mathcal{S}}\;\mathbb{P}^\pi(s_n\neq0\mid s_0=s)
$$

be the worst-case probability of still being alive after $n$ steps, where $n$ is the number of non-terminal states.

> [!note] Lemma (geometric termination)
> 1. $\pi$ is proper if and only if, from every state, there is a path of positive probability to $0$, if and only if $\rho_\pi<1$.
> 2. For a proper $\pi$, $\;\mathbb{P}^\pi(s_t\neq0\mid s_0=s)\le\rho_\pi^{\lfloor t/n\rfloor}$ for all $s$ and $t\ge0$.
> 3. For a proper $\pi$, the value is finite and given by the series below.
>
> In part 3, with $\bar r=\max_{s}\mathbb{E}\big[\,|r(s,\pi(s),s')|\,\big]$ the largest expected absolute one-step reward,
>
> $$
> \begin{aligned}
> V^\pi&=\sum_{t=0}^{\infty}(P^\pi)^t r_\pi=(I-P^\pi)^{-1}r_\pi,\\
> \lVert V^\pi\rVert_\infty&\le\frac{n\,\bar r}{1-\rho_\pi}.
> \end{aligned}
> $$

> [!proof]- Proof
> **1.** If $\pi$ is proper, some path to $0$ must have positive probability. Conversely, suppose that from every $s$ there is a positive-probability path to $0$. Take a shortest one. It never visits a state twice (cutting out a loop gives a shorter path), so it has at most $n$ steps. Hence $\mathbb{P}^\pi(s_n=0\mid s_0=s)>0$ for every $s$, because $0$ is absorbing. A maximum over finitely many states of numbers below $1$ is below $1$, so $\rho_\pi<1$. Finally, $\rho_\pi<1$ implies properness by part 2, letting $t\to\infty$.
>
> **2.** By the Markov property and time-homogeneity, for any $m\ge1$,
>
> $$
> \begin{aligned}
> \mathbb{P}(s_{mn}\neq0\mid s_0=s)&=\sum_{j\ge1}\mathbb{P}(s_{mn}\neq0\mid s_{(m-1)n}=j)\\
> &\qquad\quad\cdot\mathbb{P}(s_{(m-1)n}=j\mid s_0=s)\\
> &\le\rho_\pi\,\mathbb{P}(s_{(m-1)n}\neq0\mid s_0=s),
> \end{aligned}
> $$
>
> so by induction $\mathbb{P}(s_{mn}\neq0\mid s_0=s)\le\rho_\pi^m$. Because $0$ is absorbing, $\mathbb{P}(s_t\neq0)$ is non-increasing in $t$. For $mn\le t<(m+1)n$, that gives $\mathbb{P}(s_t\neq0\mid s_0=s)\le\rho_\pi^m$ with $m=\lfloor t/n\rfloor$.
>
> **3.** First show absolute convergence, *then* take the limit. A reward is collected at time $t$ only if $s_t\neq0$, so $\mathbb{E}|r_t|\le\bar r\,\mathbb{P}(s_t\neq0)\le\bar r\,\rho_\pi^{\lfloor t/n\rfloor}$. Each value of $\lfloor t/n\rfloor=m$ occurs $n$ times, so
>
> $$
> \sum_{t=0}^{\infty}\mathbb{E}|r_t|\le\bar r\,n\sum_{m=0}^{\infty}\rho_\pi^m=\frac{n\,\bar r}{1-\rho_\pi}<\infty.
> $$
>
> The series $\sum_t\mathbb{E}[r_t]$ therefore converges absolutely, the $\liminf$ in the definition of $V^\pi$ is a genuine limit, and $V^\pi=\sum_t(P^\pi)^tr_\pi$. Since $(P^\pi)^t\to0$, the Neumann series gives $(I-P^\pi)^{-1}=\sum_t(P^\pi)^t$. $\blacksquare$

So for proper policies, **termination does the job of $\gamma$**. In the discounted case the weights $\gamma^t$ make the series converge; here the survival probabilities $\mathbb{P}(s_t\neq0)$ do.

> [!example] The $4\times4$ gridworld of Sutton and Barto
> The classic first example of iterative policy evaluation[^suttonbarto] is an SSP. It has 14 non-terminal cells, two terminal corners, a reward of $-1$ per step, $\gamma=1$ and the equiprobable random policy. That policy is proper (every cell has a path to a corner), so its value is $(I-P^\pi)^{-1}r_\pi$. Solving the $14\times14$ system gives, row by row,
>
> $$
> \begin{pmatrix}0&-14&-20&-22\\-14&-18&-20&-20\\-20&-20&-18&-14\\-22&-20&-14&0\end{pmatrix}.
> $$
>
> Each entry is minus the expected number of steps a random walker needs to hit a corner. Note the $-20$ (not $-22$) at the end of the second row: by the grid's half-turn symmetry it must equal the first cell of the third row. Sutton and Barto's Exercise 4.2 adds a state 15 just below the second cell of the bottom row. Its moves left, up, right and down lead to the cells valued $-22$, $-20$, $-14$ and back to itself, and all other transitions are unchanged. Its Bellman equation $v=-1+\tfrac14(-22-20-14+v)$ gives $v=-20$.

> [!example]- Code: is a policy proper, and what is its value? (NumPy)
> ```python
> # assumes: P_pi (n, n) = transition probabilities among the NON-terminal states under pi (row sums <= 1),
> # r_pi (n,) = expected one-step rewards, including the reward of a step into the terminal state
> n = len(r_pi)
> alive = np.linalg.matrix_power(P_pi, n).sum(1)       # P(s_n != 0 | s_0 = s)
> rho_pi = alive.max()
> proper = rho_pi < 1                                  # the Lemma: proper iff rho_pi < 1
> V = np.linalg.solve(np.eye(n) - P_pi, r_pi) if proper else None   # V^pi = (I - P^pi)^{-1} r_pi
> ```
> For the equiprobable policy on the $4\times4$ gridworld, `P_pi` is $14\times14$, every `r_pi` entry is $-1$, and `V` reproduces the matrix above.

![[Reinforcement Learning/images/stochastic-shortest-paths-maze.svg]]
*Figure 1: a deterministic maze with exit G and a reward of $-1$ per step, including the step onto G. (a) The optimal policy and $V^*$, which is minus the shortest-path distance to G ($V^*(S)=-10$). (b) The same policy, except that two neighbouring cells in the fifth row point at each other. That one change makes the policy improper. The 7 orange cells feed into the loop, never reach G, and have value $-\infty$. The other 18 cells keep their finite values, because properness fails only where the loop is reachable.*

### The two standing assumptions

Throughout, we assume:

- **(A)** There is at least one proper stationary policy.
- **(B)** Every improper stationary policy $\pi$ has $V^\pi(s)=-\infty$ for at least one state $s$.

(A) says the problem is solvable: the goal can be reached. (B) says that **failing to terminate is infinitely bad**. It does *not* forbid improper policies; the maze has plenty (Figure 1b). It only requires each of them to be catastrophic somewhere. A simple sufficient condition is that every non-terminal step has a strictly negative reward: $r(s,a)\le-\varepsilon<0$ for all $s\ge1$ and all $a$. An improper policy then has a state from which it never terminates, collects at most $-\varepsilon$ per step forever, and so has value $-\infty$. This is the "$-1$ per step" design used in mazes and shortest-path tasks.

> [!tip] Intuition: why (B) is needed
> The reward setting has an asymmetry that the cost setting hides. In the cost setting, costs are usually non-negative, so "running forever" automatically costs $+\infty$. With rewards, a loop that never ends could pay $0$ per step, or even a positive amount. Then:
> - **Positive rewards on a loop:** an improper policy earns $+\infty$. The problem is ill-posed; there is no meaningful optimum.
> - **Zero reward on a loop:** values stay finite, but the Bellman equation stops pinning down $V^*$.
>
> The smallest example has one state with two actions. "Exit" pays $-1$ and terminates; "stay" pays $r_{\rm stay}$ and stays. The Bellman equation is $V=\max(r_{\rm stay}+V,\,-1)$.
> - With $r_{\rm stay}=-1$, (B) holds and the unique solution is $V^*=-1$.
> - With $r_{\rm stay}=0$, (B) fails. Staying forever is optimal ($V^*=0$, an improper optimal policy), and **every** $V\ge-1$ solves the Bellman equation. Value iteration started at $V_0=3$ stays at $3$ forever. Both uniqueness and the convergence of value iteration are lost (Figure 2).

![[Reinforcement Learning/images/stochastic-shortest-paths-assumption-b.svg]]
*Figure 2: the one-state example, with $\mathcal{T}^*V$ plotted against $V$. Fixed points are where the curve meets the diagonal. (a) When staying earns reward $-1$, the curve crosses the diagonal once, at $V^*=-1$. The cobweb from $V_0=3$ shows value iteration reaching it in 4 steps ($3,2,1,0,-1$). (b) When staying earns reward $0$, the curve lies on the diagonal for all $V\ge-1$, so every such $V$ is a fixed point, and value iteration from $V_0=3$ never moves.*

---

## Bellman operators without discounting

For $V\in\mathbb{R}^n$ (with $V(0)=0$) and a stationary $\pi$, define

$$
\begin{aligned}
(\mathcal{T}^\pi V)(s)&=r(s,\pi(s))+\sum_{s'=1}^{n}p(s'\mid s,\pi(s))\,V(s'),\\
(\mathcal{T}^*V)(s)&=\max_{a\in\mathcal{A}(s)}\Big[r(s,a)+\sum_{s'=1}^{n}p(s'\mid s,a)\,V(s')\Big].
\end{aligned}
$$

These are the operators of the [[Dynamic Programming#The foundation: Bellman operators are contractions|DP lecture]] with $\gamma=1$. The final transition's reward is already inside $r(s,a)$. In matrix form, $\mathcal{T}^\pi V=r_\pi+P^\pi V$ with a substochastic $P^\pi$.

**What the iterates mean.** $(\mathcal{T}^*)^kV$ is the optimal expected reward of a **$k$-stage problem** with terminal value $V$: play $k$ steps, then collect $V(s_k)$ if still alive. Likewise, $(\mathcal{T}^\pi)^kV$ is the $k$-stage return of $\pi$ with terminal value $V$. This is the [[Finite-Horizon Dynamic Programming|finite-horizon DP]] recursion run backwards for $k$ stages. In particular, with $V=\mathbf{0}$ we get the $k$-stage return.

Two elementary properties carry all the proofs that follow.

> [!note] Lemma 1 (monotonicity)
> If $V\le V'$, then $\mathcal{T}^\pi V\le\mathcal{T}^\pi V'$ and $\mathcal{T}^*V\le\mathcal{T}^*V'$. By induction, the same holds for $(\mathcal{T}^\pi)^k$ and $(\mathcal{T}^*)^k$ for every $k\ge1$.

> [!note] Lemma 2 (constant shift)
> For every scalar $c\ge0$ and every $k\ge0$,
>
> $$
> \begin{aligned}
> (\mathcal{T}^*)^k(V+c\mathbf{1})&\le(\mathcal{T}^*)^kV+c\mathbf{1},\\
> (\mathcal{T}^*)^k(V-c\mathbf{1})&\ge(\mathcal{T}^*)^kV-c\mathbf{1},
> \end{aligned}
> $$
>
> and the same holds for $\mathcal{T}^\pi$.

> [!proof]- Proofs
> **Monotonicity.** Transition probabilities are non-negative, so $\sum_{s'}p(s'\mid s,a)V(s')\le\sum_{s'}p(s'\mid s,a)V'(s')$ for every $a$. Adding $r(s,a)$ keeps the order, and if $f(a)\le g(a)$ for every $a$, then $\max_af(a)\le\max_ag(a)$. For $k$ steps, apply the one-step result to $(\mathcal{T}^*)^{k-1}V\le(\mathcal{T}^*)^{k-1}V'$.
>
> **Constant shift.** Let $u_a(s)=r(s,a)+\sum_{s'\ge1}p(s'\mid s,a)V(s')$ and $\ell_a(s)=\sum_{s'\ge1}p(s'\mid s,a)\in[0,1]$, the probability of *not* terminating in one step. Then
>
> $$
> \big(\mathcal{T}^*(V+c\mathbf{1})\big)(s)=\max_a\big[u_a(s)+c\,\ell_a(s)\big].
> $$
>
> For $c\ge0$ this is at most $\max_au_a(s)+c$, and for $c\le0$ it is at least $\max_au_a(s)+c$. For $k$ steps, by induction and Lemma 1:
>
> $$
> (\mathcal{T}^*)^{k+1}(V+c\mathbf{1})\le\mathcal{T}^*\big((\mathcal{T}^*)^kV+c\mathbf{1}\big)\le(\mathcal{T}^*)^{k+1}V+c\mathbf{1}.
> $$
>
> The proof for $\mathcal{T}^\pi$ is the same with a single action. $\blacksquare$

Compare this with the discounted case, where the shift is an **equality**, $\mathcal{T}^*(V+c\mathbf{1})=\mathcal{T}^*V+\gamma c\mathbf{1}$, because every row leaks exactly $1-\gamma$. Here a row leaks $1-\ell_a(s)$, which depends on the state and action and can be $0$. So a shift passes through *at most* unchanged, which is enough for sandwich arguments but not for a contraction.

---

## Evaluating a proper policy

> [!note] Proposition 1 (policy evaluation)
> **(a)** If $\pi$ is proper, then for every $V\in\mathbb{R}^n$, $(\mathcal{T}^\pi)^kV\to V^\pi$. Moreover, $V^\pi$ is the **unique** solution of $V=\mathcal{T}^\pi V$.
>
> **(b)** Under (B): if $V\le\mathcal{T}^\pi V$ for some $V\in\mathbb{R}^n$, then $\pi$ is proper.

Part (b) is the workhorse. It says that a policy whose Bellman update can only *raise* some finite value vector cannot be improper, because an improper policy would drag some component to $-\infty$.

> [!proof]- Proof
> **(a)** Unrolling $\mathcal{T}^\pi V=r_\pi+P^\pi V$ gives
>
> $$
> (\mathcal{T}^\pi)^kV=(P^\pi)^kV+\sum_{t=0}^{k-1}(P^\pi)^tr_\pi.
> $$
>
> The sum converges to $V^\pi$ by the lemma. For the first term, each row of $(P^\pi)^k$ has total mass $\mathbb{P}(s_k\neq0)\le\rho_\pi^{\lfloor k/n\rfloor}$, so $|(P^\pi)^kV|\le\rho_\pi^{\lfloor k/n\rfloor}\lVert V\rVert_\infty\mathbf{1}\to0$. (The *absolute* value matters: an upper bound alone would not show the term vanishes.) Hence $(\mathcal{T}^\pi)^kV\to V^\pi$. Taking $k\to\infty$ in $(\mathcal{T}^\pi)^{k+1}V=r_\pi+P^\pi(\mathcal{T}^\pi)^kV$ gives $V^\pi=\mathcal{T}^\pi V^\pi$. If also $V=\mathcal{T}^\pi V$, then $V=(\mathcal{T}^\pi)^kV\to V^\pi$, so $V=V^\pi$.
>
> **(b)** Monotonicity applied repeatedly to $V\le\mathcal{T}^\pi V$ gives $V\le(\mathcal{T}^\pi)^kV$ for every $k$. Since $P^\pi$ is substochastic, $(P^\pi)^kV\le\lVert V\rVert_\infty\mathbf{1}$, so
>
> $$
> V\le\lVert V\rVert_\infty\mathbf{1}+\sum_{t=0}^{k-1}(P^\pi)^tr_\pi\qquad\text{for all }k.
> $$
>
> If $\pi$ were improper, (B) would give a state $s$ where the $k$-stage return $\big(\sum_{t<k}(P^\pi)^tr_\pi\big)(s)$ has $\liminf$ equal to $-\infty$. Along those $k$, the right-hand side at $s$ goes to $-\infty$, contradicting $V(s)\in\mathbb{R}$. $\blacksquare$

---

## The Bellman equation and value iteration

> [!note] Proposition 2 (Bellman equation)
> Under (A) and (B):
>
> **(a)** $V^*$ is finite, satisfies $V^*=\mathcal{T}^*V^*$, and is the **unique** solution of this equation in $\mathbb{R}^n$.
>
> **(b)** **Value iteration converges from any start:** $(\mathcal{T}^*)^kV\to V^*$ for every $V\in\mathbb{R}^n$.
>
> **(c)** A stationary $\pi$ is optimal if and only if it is greedy with respect to $V^*$, that is $\mathcal{T}^\pi V^*=\mathcal{T}^*V^*$. Hence an optimal **stationary deterministic** policy exists, and every optimal stationary policy is proper.

These are exactly the conclusions of the discounted theory. The proof, though, cannot use Banach. It is built from monotonicity, the shift lemma and Proposition 1, in four steps:

1. $\mathcal{T}^*$ has **at most one** fixed point.
2. $\mathcal{T}^*$ has **at least one**, namely the value of a proper policy found by policy iteration.
3. Value iteration converges to it from any start, by squeezing between an increasing and a decreasing sequence.
4. That fixed point is $V^*$, because no policy, stationary or not, can beat $(\mathcal{T}^*)^k\mathbf{0}$ over $k$ stages.

> [!proof]- Proof of Proposition 2
> **Step 1 (at most one fixed point).** Let $V=\mathcal{T}^*V$ and $V'=\mathcal{T}^*V'$, and let $\pi,\pi'$ be greedy for them, so $\mathcal{T}^\pi V=V$ and $\mathcal{T}^{\pi'}V'=V'$. By Proposition 1(b) both policies are proper, and by 1(a) $V=V^\pi$ and $V'=V^{\pi'}$. A maximum is at least any particular choice, so $\mathcal{T}^*U\ge\mathcal{T}^{\pi'}U$ for every $U$. With monotonicity this gives $(\mathcal{T}^*)^kU\ge(\mathcal{T}^{\pi'})^kU$ by induction. Hence
>
> $$
> V=(\mathcal{T}^*)^kV\ge(\mathcal{T}^{\pi'})^kV\xrightarrow[k\to\infty]{}V^{\pi'}=V'.
> $$
>
> By symmetry $V'\ge V$, so $V=V'$.
>
> **Step 2 (a fixed point exists).** By (A), start from a proper $\pi_0$. Given a proper $\pi_k$, let $\pi_{k+1}$ be greedy with respect to $V^{\pi_k}$. Then
>
> $$
> V^{\pi_k}=\mathcal{T}^{\pi_k}V^{\pi_k}\le\mathcal{T}^*V^{\pi_k}=\mathcal{T}^{\pi_{k+1}}V^{\pi_k}.
> $$
>
> *First* conclude that $\pi_{k+1}$ is proper, by Proposition 1(b). *Then* monotonicity gives $V^{\pi_k}\le\mathcal{T}^{\pi_{k+1}}V^{\pi_k}\le(\mathcal{T}^{\pi_{k+1}})^2V^{\pi_k}\le\cdots\to V^{\pi_{k+1}}$ by 1(a). So the values never decrease. There are finitely many stationary deterministic policies, so some policy repeats, $\pi_j=\pi_{j+m}$. Then $V^{\pi_j}\le V^{\pi_{j+1}}\le\cdots\le V^{\pi_{j+m}}=V^{\pi_j}$, so all of these are equal, and
>
> $$
> V^{\pi_j}\le\mathcal{T}^*V^{\pi_j}=\mathcal{T}^{\pi_{j+1}}V^{\pi_j}\le V^{\pi_{j+1}}=V^{\pi_j}.
> $$
>
> So $W:=V^{\pi_j}$ satisfies $W=\mathcal{T}^*W$, and $\bar\pi:=\pi_j$ is a proper policy with $V^{\bar\pi}=W$.
>
> **Step 3 (value iteration converges to $W$).** Two facts are used. $\mathcal{T}^*$ is continuous: each component is a maximum of finitely many affine functions. And a bounded monotone sequence in $\mathbb{R}^n$ converges.
>
> *From below.* Fix $\delta>0$ and let $\hat V$ be the value of $\bar\pi$ when every reward is lowered by $\delta$. Then $\hat V=W-\delta(I-P^{\bar\pi})^{-1}\mathbf{1}\le W-\delta\mathbf{1}$, because $(I-P^{\bar\pi})^{-1}=\sum_t(P^{\bar\pi})^t\ge I$. Also $\mathcal{T}^{\bar\pi}\hat V=\hat V+\delta\mathbf{1}$. Hence
>
> $$
> \hat V\le\mathcal{T}^{\bar\pi}\hat V\le\mathcal{T}^*\hat V\le\mathcal{T}^*W=W.
> $$
>
> By monotonicity $(\mathcal{T}^*)^k\hat V$ is non-decreasing and bounded above by $W$. Its limit is a fixed point by continuity, so it equals $W$ by Step 1.
>
> *From above.* By Lemma 2, $\mathcal{T}^*(W+\delta\mathbf{1})\le W+\delta\mathbf{1}$, and by monotonicity $\mathcal{T}^*(W+\delta\mathbf{1})\ge\mathcal{T}^*W=W$. So $(\mathcal{T}^*)^k(W+\delta\mathbf{1})$ is non-increasing, bounded below by $W$, and converges to $W$ for the same reason.
>
> *Any start.* Given $V$, take $\delta\ge\lVert V-W\rVert_\infty$. Then $\hat V\le W-\delta\mathbf{1}\le V\le W+\delta\mathbf{1}$. Monotonicity squeezes $(\mathcal{T}^*)^kV$ between two sequences that both converge to $W$.
>
> **Step 4 ($W=V^*$).** Let $\pi=(d_0,d_1,\dots)$ be any policy. Its $k$-stage return with **zero** terminal value is $\mathcal{T}^{d_0}\mathcal{T}^{d_1}\cdots\mathcal{T}^{d_{k-1}}\mathbf{0}$, where $\mathcal{T}^{d}$ is the operator of the stationary policy $d$. Using $\mathcal{T}^{d}\le\mathcal{T}^*$ and monotonicity, by induction on $k$,
>
> $$
> \mathcal{T}^{d_0}\cdots\mathcal{T}^{d_{k-1}}\mathbf{0}\le(\mathcal{T}^*)^k\mathbf{0}.
> $$
>
> Taking $\liminf_k$ and using Step 3 with $V=\mathbf{0}$ gives $V^\pi\le W$. (The terminal value must be $\mathbf{0}$ here: only then is the left side the $k$-stage return of $\pi$.) So $V^*\le W$. Also $W=V^{\bar\pi}\le V^*$, hence $V^*=W$. This proves (a) and (b).
>
> **(c).** If $\pi$ is optimal, $V^\pi=V^*$ is finite, so $\pi$ is proper: by (B) an improper policy is $-\infty$ somewhere. Then $\mathcal{T}^\pi V^*=\mathcal{T}^\pi V^\pi=V^\pi=V^*=\mathcal{T}^*V^*$. Conversely, if $\mathcal{T}^\pi V^*=\mathcal{T}^*V^*=V^*$, then $V^*\le\mathcal{T}^\pi V^*$, so $\pi$ is proper by 1(b), and $V^*$ is a fixed point of $\mathcal{T}^\pi$. By 1(a), $V^*=V^\pi$. $\blacksquare$

**Stationary policies suffice.** Step 4 compares $V^*$ with *every* policy, including non-stationary ones. The bound $(\mathcal{T}^*)^k\mathbf{0}$ is the optimal $k$-stage return over *all* policies, even randomized or history-dependent ones, by [[Finite-Horizon Dynamic Programming|finite-horizon DP]]. So nothing beats the best stationary deterministic policy. The [[Introduction to Reinforcement Learning|introduction]] quoted this fact for discounted MDPs; here we have proved it for SSPs.

![[Reinforcement Learning/images/stochastic-shortest-paths-vi-convergence.svg]]
*Figure 3: value iteration on the maze. (a) The deterministic maze, from three starting points. From $V_0=0$ or a pessimistic $V_0=-100$, VI is exact after 11 sweeps, the length of the longest shortest path to G. From an optimistic $V_0=+50$, it needs 61 sweeps. The error falls by exactly $1$ per sweep, linearly rather than geometrically. Each cell can "stay put" by walking into a wall, and that inflated value only decays at the rate of the $-1$ step reward. Without a discount, nothing else shrinks it. (b) A slippery version of the maze (each move goes in a uniformly random direction with probability 0.2), log scale. Jacobi VI reaches $10^{-8}$ after 46 sweeps. Gauss–Seidel VI needs 28 sweeps when cells near the exit are updated first, and 33 when they are updated last.*

The optimistic run in Figure 3(a) is worth remembering. In the discounted case, an error of $50$ shrinks by a factor $\gamma$ per sweep. In an SSP an over-estimate decays only as fast as the rewards can "pay it off", here $1$ per step. Starting from below is safer: $V_0=0$ is a natural pessimistic start when all rewards are negative.

> [!example] A worked example: the spider and the fly
> This is a version of the spider-and-fly example in Bertsekas's textbook.[^bertsekasvol1] A spider and a fly sit on a line, at distance $x\in\{1,\dots,n\}$. Each step, the fly moves one unit away with probability $q$, one unit towards the spider with probability $q$, and stays with probability $1-2q$ (with $0<q<\tfrac12$). When the distance is $2$ or more, the spider always moves one unit towards the fly. At distance $1$ it has a choice: **move** onto the fly's current position, or **wait**. Both move at once, the new distance is $|x-a+w|$ for spider move $a\in\{0,1\}$ and fly move $w\in\{-1,0,1\}$, and capture means distance $0$. The reward is $-1$ per step, so $-V$ is the expected capture time.
>
> At distance $1$:
> - **Move:** the spider lands on the fly's old spot. It catches the fly only if the fly stayed (probability $1-2q$); otherwise they cross or separate, back to distance $1$. With $h_1$ the expected capture time from distance $1$ under "always move", $h_1=1+2q\,h_1$, so $h_1=1/(1-2q)$.
> - **Wait:** the fly walks into the spider with probability $q$, stays at distance $1$ with probability $1-2q$, and moves to distance $2$ with probability $q$. From distance $2$ the spider moves, and the distance becomes $0$, $1$ or $2$, so $h_2=\big(1+(1-2q)\,h_1\big)/(1-q)$. Solving with $h_1=1+(1-2q)\,h_1+q\,h_2$ gives $h_1=1/q$.
>
> So moving is optimal if and only if $1/(1-2q)<1/q$, i.e. $q<\tfrac13$, and waiting is optimal for $q>\tfrac13$ (Figure 4a). By Proposition 2(c), the greedy policy with respect to $V^*$ is optimal, and the computation confirms it: at $q=0.2$, moving takes $1.67$ steps against $5$ for waiting; at $q=0.4$, waiting takes $2.5$ against $5$.
>
> At $q=\tfrac12$ the fly never stays. "Always move" then never catches it: an **improper** policy with value $-\infty$. "Wait" is still proper. This is (A) and (B) at work in a natural problem.

---

## When every policy is proper: a weighted contraction

Under (A) and (B) alone, $\mathcal{T}^*$ need not be a contraction in *any* norm. In the one-state example with $r_{\rm stay}=-1$, $\mathcal{T}^*V=V-1$ for all $V\ge0$, so $|\mathcal{T}^*U-\mathcal{T}^*V|=|U-V|$ there. Value iteration still converges (Proposition 2), so contraction is sufficient but not necessary. If we assume more, we get it back.

> [!note] Proposition 3 (weighted max-norm contraction)
> Suppose **every** stationary policy is proper. Then there are weights $\xi(s)\ge1$ and a modulus $\beta\in(0,1)$ such that
>
> $$
> \sum_{s'=1}^{n}p(s'\mid s,a)\,\xi(s')\le\beta\,\xi(s)\qquad\text{for all }s\in\mathcal{S},\ a\in\mathcal{A}(s),
> $$
>
> and $\mathcal{T}^*$ and every $\mathcal{T}^\pi$ are $\beta$-contractions in the weighted max-norm
>
> $$
> \lVert V\rVert_\xi=\max_{s}\frac{|V(s)|}{\xi(s)}.
> $$
>
> One choice is $\xi(s)$ equal to the **largest expected number of steps to termination from $s$**, over all policies, with $\beta=\max_s\big(\xi(s)-1\big)/\xi(s)$.

**Why these weights work.** Dividing by $\xi(s)$ measures an error at $s$ *relative to how long the process can still run from $s$*. After one step, the expected remaining time has dropped by at least one unit out of $\xi(s)$. So the weighted error shrinks by the factor $(\xi(s)-1)/\xi(s)$ or better. States far from termination get large weights, which lets their errors be large without breaking the contraction.

> [!proof]- Proof
> **The weights.** Consider an auxiliary SSP with the same transitions and reward $+1$ for every non-terminal step. Its value is the expected number of steps to termination. Since every policy is proper, (A) holds and (B) is vacuous, so Proposition 2 applies (positive rewards are harmless when no policy can loop). Let $\xi$ be its optimal value: $\xi(s)=\max_\pi\mathbb{E}^\pi[\tau\mid s_0=s]\ge1$, where $\tau$ is the termination time. The Bellman equation $\xi=\mathcal{T}^*_{\rm aux}\xi$ gives, for **every** action $a$,
>
> $$
> \begin{aligned}
> &\xi(s)\ge1+\sum_{s'\ge1}p(s'\mid s,a)\,\xi(s'),\\
> &\text{so}\quad\sum_{s'\ge1}p(s'\mid s,a)\,\xi(s')\le\xi(s)-1\le\beta\,\xi(s),
> \end{aligned}
> $$
>
> with $\beta=\max_s(\xi(s)-1)/\xi(s)<1$ (finitely many states, each $\xi(s)<\infty$).
>
> **Contraction of $\mathcal{T}^\pi$.** For any $U,V$ and state $s$,
>
> $$
> \begin{aligned}
> |(\mathcal{T}^\pi U)(s)-(\mathcal{T}^\pi V)(s)|&\le\sum_{s'}p(s'\mid s,\pi(s))\,\xi(s')\,\frac{|U(s')-V(s')|}{\xi(s')}\\
> &\le\beta\,\xi(s)\,\lVert U-V\rVert_\xi.
> \end{aligned}
> $$
>
> Divide by $\xi(s)$ and maximize over $s$.
>
> **Contraction of $\mathcal{T}^*$.** Use $|\max_af(a)-\max_ag(a)|\le\max_a|f(a)-g(a)|$ and apply the same bound to each action. $\blacksquare$

With Banach, Proposition 3 gives everything at once: a unique fixed point, value iteration converging at the geometric rate $\lVert V_k-V^*\rVert_\xi\le\beta^k\lVert V_0-V^*\rVert_\xi$, and the same for policy evaluation.

**Why it fails with an improper policy.** Suppose $\pi$ is improper. Then there is a set $C$ of states that the chain of $\pi$ never leaves and from which it never terminates, so $P^\pi\mathbf{1}_C=\mathbf{1}$ on $C$. Take $U=V+c\,\mathbf{1}_C$. For $s\in C$, $(\mathcal{T}^\pi U-\mathcal{T}^\pi V)(s)=c\,(P^\pi\mathbf{1}_C)(s)=c$, so $\lVert\mathcal{T}^\pi U-\mathcal{T}^\pi V\rVert_\xi\ge\max_{s\in C}c/\xi(s)=\lVert U-V\rVert_\xi$. For **every** choice of weights, the modulus is $1$. Without leakage nothing shrinks, so no weighting can create a contraction.

**Discounting is the special case $\xi=\text{const}$.** In the SSP that mimics a discounted MDP ([[#Discounted problems are SSPs|last section]]), every policy terminates with probability $1-\gamma$ at each step. The expected termination time is then $1/(1-\gamma)$ from every state, the weights are constant, $\lVert\cdot\rVert_\xi$ is the ordinary max-norm up to scale, and $\beta=\gamma$. The DP lecture's contraction is Proposition 3 in disguise.

![[Reinforcement Learning/images/stochastic-shortest-paths-spider-fly.svg]]
*Figure 4: the spider and the fly. (a) Expected capture time from distance 1 if the spider moves ($1/(1-2q)$) or waits ($1/q$); the optimal action switches at $q=1/3$. (b) Contraction weights for $q=0.3$ and $n=10$. Every policy is proper, and $\xi(x)$, the worst-case expected capture time, grows from $3.33$ at distance 1 to $11.33$ at distance 10. The expected weight after one "move" step (black) stays below $\beta\,\xi(x)$ (orange), with $\beta=0.912$.*

![[Reinforcement Learning/images/stochastic-shortest-paths-contraction.svg]]
*Figure 5: value iteration on the spider and fly ($q=0.3$), log scale. In the weighted norm, the error shrinks by a factor of at most $0.907$ per sweep over the first 40 sweeps, within the guaranteed $\beta=0.912$ (dashed). The plain max-norm error (dotted) has no such guarantee. From $V_0=-30$ it shrank by as little as a factor $0.996$ in one sweep. The worst-case bound is loose: the actual rate is set by the slowest-decaying mode of the dynamics, not by the worst state.*

> [!warning] A qualitative guarantee
> The modulus $\beta$ can be extremely close to $1$. In the slippery maze of Figure 3(b), every policy is proper, because random slips eventually reach the exit. But the worst policy, which keeps pushing away from the exit, needs about $1.9\times10^{11}$ steps on average. So $\beta=1-5\times10^{-12}$, yet value iteration converged in 46 sweeps. Proposition 3 guarantees convergence; it is a poor predictor of speed.

---

## Algorithms

### Value iteration: Jacobi, Gauss–Seidel and asynchronous

**Value iteration** is $V_{k+1}=\mathcal{T}^*V_k$. It is also called *Jacobi* VI, because every component of $V_{k+1}$ uses only the old vector $V_k$. Proposition 2(b) proves it converges from any start under (A) and (B), and Proposition 3 adds a geometric rate when all policies are proper.

**Gauss–Seidel VI** sweeps the states in a fixed order and always uses the freshest values. This is the "in-place" update of the [[Dynamic Programming#Policy evaluation (prediction)|DP lecture]]. Formally, for states ordered $1,\dots,n$,

$$
(FV)(s)=\max_{a}\Big[r(s,a)+\sum_{s'<s}p(s'\mid s,a)(FV)(s')+\sum_{s'\ge s}p(s'\mid s,a)V(s')\Big].
$$

The operator $F$ depends on the order. Its fixed points are exactly those of $\mathcal{T}^*$. If $FV=V$, the first component gives $(\mathcal{T}^*V)(1)=V(1)$, and then each later component is computed with unchanged values, so $\mathcal{T}^*V=V$. Conversely, $\mathcal{T}^*V=V$ means no component changes when it is updated, so $FV=V$.

**Asynchronous VI** is the most general form. At step $k$ it picks a single state $s_k$ and sets $V_{k+1}(s_k)=(\mathcal{T}^*V_k)(s_k)$, leaving the other components unchanged. Gauss–Seidel is the special case that cycles through the states in order.

> [!example]- Code: one Jacobi sweep and one Gauss–Seidel sweep (NumPy)
> ```python
> # assumes: P[s, a, s'] among the non-terminal states only, R[s, a] = r(s, a) including the final step's reward
> V_jacobi = (R + P @ V).max(1)            # every component uses the old V
> V_gs = V.copy()
> for s in range(len(V)):                  # Gauss-Seidel: states already updated use their new values
>     V_gs[s] = (R[s] + P[s] @ V_gs).max()
> ```

> [!note] Proposition 4 (Gauss–Seidel and asynchronous VI)
> Under (A) and (B), asynchronous value iteration converges to $V^*$ from any $V_0\in\mathbb{R}^n$, provided **every state is updated infinitely often**. In particular, $F^kV\to V^*$ for every $V$. If every policy is proper, $F$ is also a $\beta$-contraction in $\lVert\cdot\rVert_\xi$, with the $\beta$ of Proposition 3.

> [!proof]- Proof
> **Monotone sandwich.** Call a vector $L$ a *sub-solution* if $L\le\mathcal{T}^*L$. Updating one component $s$ of a sub-solution, $L'(s)=(\mathcal{T}^*L)(s)$, gives $L'\ge L$, and $L'$ is again a sub-solution. Indeed, at $s$: $L'(s)=(\mathcal{T}^*L)(s)\le(\mathcal{T}^*L')(s)$ by monotonicity. At any other $s'$: $L'(s')=L(s')\le(\mathcal{T}^*L)(s')\le(\mathcal{T}^*L')(s')$. Also, if $L\le V^*$, then $\mathcal{T}^*L\le V^*$, so updates keep $L\le V^*$.
>
> Start from $L_0=\hat V$ of Step 3 above, a sub-solution below $V^*$, and apply the same sequence of updates as the algorithm. The iterates $L_k$ are non-decreasing and bounded by $V^*$, so they converge to some $\bar L$. Let $k_1<k_2<\cdots$ be the times at which state $s$ is updated. Then $L_{k_j+1}(s)=(\mathcal{T}^*L_{k_j})(s)$. As $j\to\infty$ the left side tends to $\bar L(s)$ and the right side to $(\mathcal{T}^*\bar L)(s)$ by continuity. So $\bar L=\mathcal{T}^*\bar L$, and $\bar L=V^*$ by uniqueness. Symmetrically, the same updates started from $U_0=V^*+\delta\mathbf{1}$, which satisfies $U_0\ge\mathcal{T}^*U_0$ by Lemma 2, decrease to $V^*$.
>
> For an arbitrary $V_0$, choose $\delta$ so that $L_0\le V_0\le U_0$. Each single-component update is monotone, so $L_k\le V_k\le U_k$ for all $k$, and $V_k\to V^*$. Gauss–Seidel is the schedule $1,2,\dots,n,1,2,\dots$.
>
> **Contraction of $F$ when all policies are proper.** Let $D=\lVert U-V\rVert_\xi$. Suppose that for the states $s'<s$ already updated, $|(FU)(s')-(FV)(s')|\le\beta\,\xi(s')D\le\xi(s')D$. Every other $s'$ has $|U(s')-V(s')|\le\xi(s')D$. Then
>
> $$
> |(FU)(s)-(FV)(s)|\le\max_a\sum_{s'}p(s'\mid s,a)\,\xi(s')\,D\le\beta\,\xi(s)\,D,
> $$
>
> and induction over $s$ gives $\lVert FU-FV\rVert_\xi\le\beta D$. $\blacksquare$

Gauss–Seidel is *often* faster, because good news travels further in one sweep, but how much faster depends on the order. On the slippery maze (Figure 3b), Jacobi needed 46 sweeps, Gauss–Seidel 28 with near-exit cells first, and 33 with them last. The best order follows the direction in which information flows: backwards from the terminal state. Asynchronous VI is also the deterministic skeleton of [[Q-Learning and SARSA|Q-learning]], which updates one state–action pair at a time along a trajectory; its convergence proof adds noise to the argument above.[^bertsekas1982]<sup>,</sup>[^ndp]

### Policy iteration

> [!example] Algorithm: policy iteration for SSPs
> **Input:** model $p$, rewards $r$. Initialize $\pi_0$ to a **proper** policy.
> **For** $k=0,1,2,\dots$:
> 1. **Evaluate:** solve the linear system $(I-P^{\pi_k})V=r_{\pi_k}$ for $V^{\pi_k}$.
> 2. **Improve:** for every $s$, set $\pi_{k+1}(s)\in\arg\max_a\big[r(s,a)+\sum_{s'\ge1}p(s'\mid s,a)V^{\pi_k}(s')\big]$, keeping $\pi_k(s)$ when it is among the maximizers.
> 3. If $\pi_{k+1}=\pi_k$, **stop**: $\pi_k$ is optimal and $V^{\pi_k}=V^*$.

The only change from [[Dynamic Programming#Policy iteration|discounted policy iteration]] is the word **proper**, and it matters. For an improper $\pi_0$ the evaluation step has no finite answer, and the linear system is typically singular. In the maze, "always up" is improper, and $I-P^\pi$ has rank 16 instead of 25.

> [!note] Proposition 5 (policy iteration)
> Under (A) and (B), starting from a proper $\pi_0$, every $\pi_k$ is proper, $V^{\pi_{k+1}}\ge V^{\pi_k}$, and the inequality is strict in at least one state unless $\pi_k$ is optimal. Policy iteration therefore terminates after finitely many iterations with an optimal proper policy.

> [!proof]- Proof
> Let $\pi$ be proper and $\pi'$ greedy with respect to $V^\pi$. The order of the argument matters.
>
> 1. **Properness first.** $V^\pi=\mathcal{T}^\pi V^\pi\le\mathcal{T}^*V^\pi=\mathcal{T}^{\pi'}V^\pi$. By Proposition 1(b), $\pi'$ is proper. Only now may we take limits of $(\mathcal{T}^{\pi'})^k$.
> 2. **Improvement.** By monotonicity, $V^\pi\le\mathcal{T}^{\pi'}V^\pi\le(\mathcal{T}^{\pi'})^2V^\pi\le\cdots\to V^{\pi'}$, using Proposition 1(a). So $V^{\pi'}\ge V^\pi$.
> 3. **Strictness.** If $V^{\pi'}=V^\pi$, then $V^\pi\le\mathcal{T}^*V^\pi=\mathcal{T}^{\pi'}V^\pi\le V^{\pi'}=V^\pi$, so $V^\pi=\mathcal{T}^*V^\pi$, and $V^\pi=V^*$ by Proposition 2(a).
> 4. **Termination.** Values strictly increase until optimality, so no policy repeats before then, and there are finitely many deterministic stationary policies. With the tie-breaking rule, once $\pi_k$ is optimal the greedy step returns $\pi_k$ itself. Otherwise the algorithm could switch forever between equally good policies, even though the values have stopped changing. $\blacksquare$

Notice how *(B) protects the algorithm*. The greedy step can never switch to a policy that loops forever, because such a policy would have $-\infty$ somewhere, and Proposition 1(b) rules that out. On the slippery maze, starting from "always left" (proper only thanks to the slips, with $V^{\pi_0}(S)\approx-3.6\times10^{8}$), policy iteration needed 5 improvements. The value of the start cell went $-3.6\times10^{8}\to-2217\to-12.71\to-12.66\to-12.65$, which equals $V^*(S)$ to two decimals.

### Modified and asynchronous policy iteration

Exact evaluation costs a linear solve per iteration. Practical variants evaluate only partially, and possibly only some states at a time. **Asynchronous policy iteration** keeps a pair $(V_k,\pi_k)$. At each step it picks a subset $B_k$ of states and does *one* of two things:

- **Value update:** $V_{k+1}(s)=(\mathcal{T}^{\pi_k}V_k)(s)$ for $s\in B_k$, other components unchanged, and $\pi_{k+1}=\pi_k$.
- **Policy update:** $\pi_{k+1}(s)\in\arg\max_a\big[r(s,a)+\sum_{s'}p(s'\mid s,a)V_k(s')\big]$ for $s\in B_k$, other states unchanged, and $V_{k+1}=V_k$.

This one scheme contains all the earlier algorithms, with $B_k$ equal to all states unless stated otherwise:

| Schedule | Algorithm |
|---|---|
| infinitely many value updates, then one policy update | policy iteration |
| $m_k$ value updates, then one policy update | modified (optimistic) policy iteration[^putermanshin1978] |
| one value update, then one policy update | value iteration |
| $B_k$ a single state, each value update followed by a policy update | asynchronous value iteration |

> [!note] Proposition 6 (asynchronous policy iteration)
> Assume (A) and (B), that value updates and policy updates are each performed infinitely often at every state, and that the initial pair satisfies
>
> $$
> \mathcal{T}^{\pi_0}V_0\ge V_0.
> $$
>
> Then $V_k$ is non-decreasing and converges to $V^*$.

The initial condition says that $V_0$ is a *pessimistic* estimate for $\pi_0$: one Bellman step of $\pi_0$ can only raise it. By Proposition 1(b) it forces $\pi_0$ to be proper. The easy way to satisfy it is to take $\pi_0$ proper and $V_0=V^{\pi_0}$, which satisfies it with equality.

> [!proof]- Proof
> **Invariant.** We show by induction that, for all $k$,
>
> $$
> \mathcal{T}^{\pi_k}V_k\ge V_k,\qquad V_{k+1}\ge V_k,\qquad \mathcal{T}^{\pi_{k+1}}V_{k+1}\ge\mathcal{T}^{\pi_k}V_k.
> $$
>
> Assume $\mathcal{T}^{\pi_k}V_k\ge V_k$.
>
> *Value update.* On $B_k$, $V_{k+1}=\mathcal{T}^{\pi_k}V_k\ge V_k$; elsewhere $V_{k+1}=V_k$. So $V_{k+1}\ge V_k$, and also $\mathcal{T}^{\pi_k}V_k\ge V_{k+1}$ (with equality on $B_k$). Since $\pi_{k+1}=\pi_k$, monotonicity gives $\mathcal{T}^{\pi_{k+1}}V_{k+1}=\mathcal{T}^{\pi_k}V_{k+1}\ge\mathcal{T}^{\pi_k}V_k\ge V_{k+1}$.
>
> *Policy update.* Here $V_{k+1}=V_k$. On $B_k$, $(\mathcal{T}^{\pi_{k+1}}V_k)(s)=(\mathcal{T}^*V_k)(s)\ge(\mathcal{T}^{\pi_k}V_k)(s)$. Off $B_k$ the action is unchanged, so the two are equal. Hence $\mathcal{T}^{\pi_{k+1}}V_{k+1}\ge\mathcal{T}^{\pi_k}V_k\ge V_k=V_{k+1}$.
>
> **Bounded above by $V^*$.** $\mathcal{T}^*V_k\ge\mathcal{T}^{\pi_k}V_k\ge V_k$. Monotonicity makes $(\mathcal{T}^*)^mV_k$ non-decreasing in $m$, and it tends to $V^*$ by Proposition 2(b). So $V_k\le V^*$. A non-decreasing bounded sequence converges, say $V_k\uparrow\bar V\le V^*$. Passing to the limit in $\mathcal{T}^*V_k\ge V_k$ (continuity) gives $\mathcal{T}^*\bar V\ge\bar V$.
>
> **No strict inequality survives.** Suppose $(\mathcal{T}^*\bar V)(s)>\bar V(s)$ for some $s$. By continuity there is $\bar k$ with $(\mathcal{T}^*V_k)(s)>\bar V(s)$ for all $k\ge\bar k$. Pick a time $k\ge\bar k$ at which a policy update is done at $s$, and let $k'>k$ be the first later time at which a value update is done at $s$. Both exist because each kind of update happens infinitely often at $s$. Chaining the third part of the invariant from $k+1$ to $k'$,
>
> $$
> \begin{aligned}
> V_{k'+1}(s)&=(\mathcal{T}^{\pi_{k'}}V_{k'})(s)\ge(\mathcal{T}^{\pi_{k+1}}V_{k+1})(s)\\
> &=(\mathcal{T}^{\pi_{k+1}}V_k)(s)=(\mathcal{T}^*V_k)(s)>\bar V(s).
> \end{aligned}
> $$
>
> This contradicts $V_{k'+1}\le\bar V$. Hence $\mathcal{T}^*\bar V=\bar V$, and $\bar V=V^*$ by uniqueness. $\blacksquare$

The monotonicity condition is what holds the proof together. Without it, natural asynchronous variants of policy iteration can fail to converge, even in favourable settings; Bertsekas and Yu discuss an example of Williams and Baird and give a modified scheme that converges without it.[^bertsekasyu2010] In a check on the slippery maze, 20,000 random asynchronous steps (3 random states per step, value and policy updates equally likely, started from $V_0=V^{\pi_0}$) kept the invariant $\mathcal{T}^{\pi_k}V_k\ge V_k$ at every step and ended within $2\times10^{-15}$ of $V^*$.

This scheme is the precise version of [[Dynamic Programming#Generalized policy iteration (GPI)|generalized policy iteration]]: any interleaving of partial evaluation and partial improvement works, *provided we start pessimistic*. [[Actor-Critic Algorithms|Actor-critic methods]] are its sampled, approximate descendants.

---

## Discounted problems are SSPs

The [[Dynamic Programming#Discounting as termination: the SSP view|DP lecture]] shows that a discounted MDP is equivalent to an SSP. Add a terminal state, and at every step end the episode with probability $1-\gamma$ before moving as usual. Every policy is then proper, the expected time to termination is $1/(1-\gamma)$ from everywhere, and Proposition 3 returns the familiar $\gamma$-contraction in the max-norm. So the SSP theory in this lecture *contains* the discounted theory. The discounted proofs are shorter only because the leakage is the same in every state.

## Key takeaways

- An **SSP** maximizes total undiscounted reward until an absorbing, reward-free terminal state. The expected reward $r(s,a)$ must include the final transition into the terminal state.
- A policy is **proper** if it terminates with probability 1 from every state, equivalently $\rho_\pi<1$. Then termination is geometric and $V^\pi=(I-P^\pi)^{-1}r_\pi$ is finite: **termination plays the role of $\gamma$**.
- Assume **(A)** a proper policy exists and **(B)** every improper policy has value $-\infty$ somewhere. In the reward setting (B) must be imposed (for example, $-1$ per step), because a free or rewarding loop makes $V^*$ ill-defined or the Bellman equation non-unique.
- Under (A) and (B): $V^*$ is the unique solution of $V=\mathcal{T}^*V$, an optimal **stationary deterministic** policy exists and is proper, value iteration converges from any start (including the Gauss–Seidel and asynchronous variants), and policy iteration started from a proper policy terminates.
- The proofs use **monotonicity**, the **constant-shift inequality** and **sandwiching**, not contraction. Optimistic initializations can converge only linearly.
- If **all** policies are proper, the Bellman operators are contractions in a max-norm **weighted by the expected time to termination**. Discounting is the special case of constant weights, with $\beta=\gamma$.
- **Asynchronous policy iteration** converges from a pessimistic start, $\mathcal{T}^{\pi_0}V_0\ge V_0$. It unifies PI, modified PI, VI and asynchronous VI.

## Where this shows up next

- [[Model-Based Reinforcement Learning]]: when $p$ and $r$ are not given, learn them from data and plan in the learned model with exactly these algorithms.
- [[Monte Carlo and Temporal-Difference Learning]] and [[Q-Learning and SARSA]] estimate the same undiscounted episodic values from sampled episodes. Monte Carlo needs episodes to end, which is properness of the behaviour policy. Q-learning is a sampled, asynchronous version of the value iteration analysed here.
- [[Stochastic Approximation]] (next) adds noise to the asynchronous argument of Proposition 4. That is how the convergence of Q-learning, including in SSPs with proper policies, is proved.[^tsitsiklis1994]

<!-- References: footnote definitions below render as the numbered "References" list at the bottom of the page -->

[^bertsekastsitsiklis1991]: Bertsekas, D. P., & Tsitsiklis, J. N. (1991). *An Analysis of Stochastic Shortest Path Problems*. Mathematics of Operations Research, 16(3), 580–595. [doi:10.1287/moor.16.3.580](https://doi.org/10.1287/moor.16.3.580)
[^bertsekasvol2]: Bertsekas, D. P. (2012). *Dynamic Programming and Optimal Control, Vol. II: Approximate Dynamic Programming* (4th ed.). Athena Scientific. [book website](http://www.athenasc.com/dpbook.html)
[^suttonbarto]: Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.), Chapter 4. [book website](http://incompleteideas.net/book/the-book-2nd.html)
[^bertsekasvol1]: Bertsekas, D. P. (2017). *Dynamic Programming and Optimal Control, Vol. I* (4th ed.). Athena Scientific. [book website](http://www.athenasc.com/dpbook.html)
[^bertsekas1982]: Bertsekas, D. P. (1982). *Distributed Dynamic Programming*. IEEE Transactions on Automatic Control, 27(3), 610–616. [doi:10.1109/TAC.1982.1102980](https://doi.org/10.1109/TAC.1982.1102980)
[^ndp]: Bertsekas, D. P., & Tsitsiklis, J. N. (1996). *Neuro-Dynamic Programming*. Athena Scientific. [book website](http://www.athenasc.com/ndpbook.html)
[^putermanshin1978]: Puterman, M. L., & Shin, M. C. (1978). *Modified Policy Iteration Algorithms for Discounted Markov Decision Problems*. Management Science, 24(11), 1127–1137. [doi:10.1287/mnsc.24.11.1127](https://doi.org/10.1287/mnsc.24.11.1127)
[^bertsekasyu2010]: Bertsekas, D. P., & Yu, H. (2010). *Distributed Asynchronous Policy Iteration in Dynamic Programming*. 48th Annual Allerton Conference on Communication, Control, and Computing, 1368–1375. [doi:10.1109/ALLERTON.2010.5707073](https://doi.org/10.1109/ALLERTON.2010.5707073)
[^tsitsiklis1994]: Tsitsiklis, J. N. (1994). *Asynchronous Stochastic Approximation and Q-Learning*. Machine Learning, 16, 185–202. [doi:10.1007/BF00993306](https://doi.org/10.1007/BF00993306)
