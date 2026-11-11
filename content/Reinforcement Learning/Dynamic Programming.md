---
title: Dynamic Programming
tags:
  - reinforcement-learning
  - dynamic-programming
  - policy-iteration
  - value-iteration
  - bellman-equations
---

> **Prerequisites:** [[Introduction to Reinforcement Learning]], in particular the Bellman expectation and optimality equations, the matrix form $V^\pi=(I-\gamma P^\pi)^{-1}r_\pi$, and the Bellman operators $\mathcal{T}^\pi$ and $\mathcal{T}^*$.
>
> **TL;DR:** When the MDP's model $p(s',r\mid s,a)$ is known, the Bellman equations can be solved directly by repeatedly applying them. Both Bellman operators are **$\gamma$-contractions**. This one fact implies that **policy evaluation** converges to $V^\pi$ and **value iteration** converges to $V^*$, both geometrically. The **policy improvement theorem** implies that **policy iteration** reaches an optimal policy in finitely many steps. Most model-free RL algorithms are sampled, approximate versions of these procedures.

## Motivation

The [[Introduction to Reinforcement Learning|previous lecture]] wrote down what the value of a policy, and the optimal value, *must* satisfy. Dynamic programming (DP) is the family of algorithms that *solves* those equations when we know the environment's dynamics exactly.[^bellman1957] This is **planning**, not learning: no interaction or exploration is needed, only computation with a known model.

Why study it if real problems rarely give us the model?

1. DP isolates the core ideas, *evaluate a policy* and *improve it greedily*, without the noise of sampling.
2. Its convergence proofs (contractions, monotone improvement) are the template for analysing Q-learning, actor-critic and many other methods.
3. Almost every RL algorithm can be read as an approximate DP algorithm that replaces exact expectations with samples and tables with function approximators.

Throughout, the MDP is **finite** ($|\mathcal{S}|=n$ states, $|\mathcal{A}|$ actions), and $\gamma\in[0,1)$. The name comes from Bellman: the problem has *optimal substructure* (an optimal policy continues optimally from the next state) and *overlapping subproblems* (the same $V(s')$ is reused by every state that can reach $s'$). The Bellman equation encodes both.

### Running example: a slippery gridworld

All figures use one $5\times5$ gridworld.

- **Terminal cells:** reaching the goal **G** pays $+1$ and falling into the pit **X** pays $-1$. Both are absorbing.
- **Actions:** the agent moves up, right, down or left.
- **Slip:** with probability $0.8$ the intended move happens. With probability $0.2$ the agent instead moves in a uniformly random direction (which may be the intended one).
- **Walls:** a move into a wall leaves the agent where it is.
- **Other rewards and discount:** all other rewards are $0$, and $\gamma=0.9$.



---

## The foundation: Bellman operators are contractions

Recall the two **Bellman operators**. They map a value vector $V\in\mathbb{R}^n$ to a new one:

$$
\begin{aligned}
(\mathcal{T}^\pi V)(s)&=\sum_a\pi(a\mid s)\Big[r(s,a)+\gamma\sum_{s'}p(s'\mid s,a)\,V(s')\Big],\\
(\mathcal{T}^*V)(s)&=\max_a\Big[r(s,a)+\gamma\sum_{s'}p(s'\mid s,a)\,V(s')\Big].
\end{aligned}
$$

The Bellman equations say exactly that $V^\pi=\mathcal{T}^\pi V^\pi$ and $V^*=\mathcal{T}^*V^*$. The value functions are **fixed points** of these operators. We measure distances between value vectors in the max-norm, $\lVert V\rVert_\infty=\max_s|V(s)|$.

> [!note] Banach fixed-point theorem
> Let $\mathcal{T}$ be a **$\gamma$-contraction** on a complete metric space: $\lVert\mathcal{T}U-\mathcal{T}V\rVert\le\gamma\lVert U-V\rVert$ for all $U,V$, with $\gamma<1$. Then:
> 1. $\mathcal{T}$ has a **unique** fixed point $V^\dagger$;
> 2. for **any** starting point $V_0$, the iterates $V_{k+1}=\mathcal{T}V_k$ converge to it, with $\lVert V_k-V^\dagger\rVert\le\gamma^k\lVert V_0-V^\dagger\rVert$.
>
> ($\mathbb{R}^n$ with $\lVert\cdot\rVert_\infty$ is complete.)

**$\mathcal{T}^\pi$ is a $\gamma$-contraction.** In matrix form, $\mathcal{T}^\pi V=r_\pi+\gamma P^\pi V$, so

$$
\lVert\mathcal{T}^\pi U-\mathcal{T}^\pi V\rVert_\infty=\gamma\,\lVert P^\pi(U-V)\rVert_\infty\le\gamma\,\lVert U-V\rVert_\infty.
$$

The inequality holds because each row of $P^\pi$ is a probability distribution, so each entry of $P^\pi x$ is an average of entries of $x$ and cannot exceed $\lVert x\rVert_\infty$ in absolute value.

**$\mathcal{T}^*$ is a $\gamma$-contraction.** For any two functions $f$ and $g$ of $a$, $\big|\max_a f(a)-\max_a g(a)\big|\le\max_a|f(a)-g(a)|$. Hence, for every $s$,

$$
\begin{aligned}
\big|(\mathcal{T}^*U)(s)-(\mathcal{T}^*V)(s)\big|&\le\max_a\Big|\gamma\sum_{s'}p(s'\mid s,a)\big(U(s')-V(s')\big)\Big|\\
&\le\gamma\,\lVert U-V\rVert_\infty.
\end{aligned}
$$

**Consequences.** $V^\pi$ and $V^*$ exist and are unique. Simply iterating either operator from *any* initial guess converges to them, and the error shrinks by at least a factor $\gamma$ per sweep. The rest of the lecture uses this over and over.

---

## Policy evaluation (prediction)

**Problem:** given $\pi$, compute $V^\pi$. The Bellman equation for $\pi$, written with the joint dynamics, is

$$
V^\pi(s)=\sum_a\pi(a\mid s)\sum_{s',r}p(s',r\mid s,a)\big[r+\gamma V^\pi(s')\big].
$$

The [[Introduction to Reinforcement Learning#Matrix form and closed-form policy evaluation|closed form]] $V^\pi=(I-\gamma P^\pi)^{-1}r_\pi$ costs $O(n^3)$. **Iterative policy evaluation** turns the equation into an update rule instead:

> [!example] Algorithm: iterative policy evaluation
> **Input:** policy $\pi$, model $p$, tolerance $\theta_{\text{tol}}>0$. Initialize $V_0(s)$ arbitrarily, with $V_0(\text{terminal})=0$.
> **Repeat** for $k=0,1,2,\dots$:
> 1. For every $s$: $\;V_{k+1}(s)\leftarrow\sum_a\pi(a\mid s)\sum_{s',r}p(s',r\mid s,a)\big[r+\gamma V_k(s')\big]$.
> 2. Stop when $\lVert V_{k+1}-V_k\rVert_\infty<\theta_{\text{tol}}$.

This is $V_{k+1}=\mathcal{T}^\pi V_k$, so by the contraction argument $V_k\to V^\pi$ for **every** $s$, with $\lVert V_k-V^\pi\rVert_\infty\le\gamma^k\lVert V_0-V^\pi\rVert_\infty$. Each sweep costs $O(n^2|\mathcal{A}|)$, or much less when transitions are sparse. Each sweep is a "one-step look-ahead" that pushes information about rewards one step further back through the state space.

![[Reinforcement Learning/images/dynamic-programming-policy-evaluation.svg]]
*Figure 1: iterative policy evaluation of the uniform random policy, starting from $V_0=0$. After one sweep only the cells next to G and X know about them. Each further sweep propagates value one more step outwards; by 10 sweeps $V_k$ is close to the exact $V^\pi$, and after about 30 it agrees to two decimals. Under a random policy the pit's pull dominates, so most of the grid has negative value.*

> [!tip] In-place updates
> The *synchronous* version above keeps two arrays, $V_k$ and $V_{k+1}$. The **in-place** (Gauss–Seidel) version overwrites $V(s)$ immediately, so later states in the same sweep already use the new values. It also converges, and usually faster. The order in which states are swept then matters.

---

## Policy improvement

Knowing $V^\pi$, can we find a better policy? Consider acting greedily with respect to it:

$$
\pi'(s)=\arg\max_a Q^\pi(s,a)=\arg\max_a\sum_{s',r}p(s',r\mid s,a)\big[r+\gamma V^\pi(s')\big].
$$

> [!note] Policy improvement theorem
> Let $\pi$ and $\pi'$ be policies such that $Q^\pi\big(s,\pi'(s)\big)\ge V^\pi(s)$ for all $s$. Then $V^{\pi'}(s)\ge V^\pi(s)$ for all $s$. If the first inequality is strict at some $s$, so is the second.

**Proof.** Unroll the assumption, one step at a time. Each time a $V^\pi$ appears, replace it using the assumption at the next state:

$$
\begin{aligned}
V^\pi(s)&\le Q^\pi\big(s,\pi'(s)\big)=\mathbb{E}\big[r_t+\gamma V^\pi(s_{t+1})\mid s_t=s,\,a_t=\pi'(s)\big]\\
&\le\mathbb{E}_{\pi'}\big[r_t+\gamma\,Q^\pi\big(s_{t+1},\pi'(s_{t+1})\big)\mid s_t=s\big]\\
&=\mathbb{E}_{\pi'}\big[r_t+\gamma r_{t+1}+\gamma^2V^\pi(s_{t+2})\mid s_t=s\big]\\
&\le\cdots\le\mathbb{E}_{\pi'}\Big[\sum_{k=0}^{\infty}\gamma^k r_{t+k}\,\Big|\,s_t=s\Big]=V^{\pi'}(s).\qquad\blacksquare
\end{aligned}
$$

The greedy $\pi'$ satisfies the assumption by construction, because $Q^\pi(s,\pi'(s))=\max_aQ^\pi(s,a)\ge\sum_a\pi(a\mid s)Q^\pi(s,a)=V^\pi(s)$. So **greedy improvement never makes things worse**.

**When does it stop improving?** Suppose the greedy policy is no better, $V^{\pi'}=V^\pi$. Then $V^\pi(s)=\max_a\sum_{s',r}p(s',r\mid s,a)[r+\gamma V^\pi(s')]$, which is the Bellman *optimality* equation. Since $\mathcal{T}^*$ has a unique fixed point, $V^\pi=V^*$, and both $\pi$ and $\pi'$ are optimal. So if $\pi$ is not already optimal, greedy improvement makes $V^{\pi'}(s)>V^\pi(s)$ in at least one state.

---

## Policy iteration

Alternate the two steps: **E**valuate the current policy, then **I**mprove it greedily.

$$
\pi_0\xrightarrow{\;E\;}V^{\pi_0}\xrightarrow{\;I\;}\pi_1\xrightarrow{\;E\;}V^{\pi_1}\xrightarrow{\;I\;}\pi_2\xrightarrow{\;E\;}\cdots\xrightarrow{\;I\;}\pi^*\xrightarrow{\;E\;}V^*.
$$

> [!example] Algorithm: policy iteration
> **Input:** model $p$. Initialize $\pi_0$ to *any* deterministic policy.
> **For** $i=0,1,2,\dots$:
> 1. **Evaluate:** compute $V^{\pi_i}$, exactly or by iterative policy evaluation.
> 2. **Improve:** for every $s$, set $\pi_{i+1}(s)\leftarrow\arg\max_a\sum_{s',r}p(s',r\mid s,a)\big[r+\gamma V^{\pi_i}(s')\big]$. Break ties in favour of $\pi_i(s)$.
> 3. If $\pi_{i+1}=\pi_i$, **stop** and return $\pi_i$ and $V^{\pi_i}$.

**Policy iteration terminates with an optimal policy.** Each improvement step either strictly increases the value in some state or stops at an optimal policy. A strictly improving sequence never repeats a policy, and there are only $|\mathcal{A}|^{n}$ deterministic policies, so the loop ends after finitely many iterations at $\pi^*$. In practice it needs **very few** iterations, often a handful even for large problems. This argument also proves that an optimal *deterministic* policy exists.

> [!warning] Two details that matter
> - **Tie-breaking.** If two actions are equally good and the improvement step picks between them arbitrarily, the policy can switch back and forth forever between equally good policies, so step 3 never fires. Keeping the current action when it is still among the maximizers prevents this.
> - **The initial policy is arbitrary.** Policy iteration converges from *any* $\pi_0$. With a known model nothing needs to be explored. Exploration only becomes a concern once we must *estimate* values from samples, as in [[Multi-Armed Bandits]], [[Policy Gradients]] and Q-learning.

![[Reinforcement Learning/images/dynamic-programming-policy-iteration.svg]]
*Figure 2: policy iteration on the gridworld, starting from the bad policy "always go left". Each panel shows a policy (arrows) and its exact value ($\pi_0$, $\pi_1$, $\pi_2$ and the final $\pi_4$). One improvement step already points the top-right region towards G. After four improvements the policy stops changing; its value equals $V^*$, and it steers around the pit.*

---

## Value iteration

Policy iteration evaluates each policy *fully* before improving it. That is wasteful early on, when the policy is about to change anyway. **Value iteration** goes to the other extreme. It does a single sweep of evaluation and improvement combined, by applying the optimality operator directly:

$$
V_{k+1}(s)=\max_{a\in\mathcal{A}(s)}\sum_{s',r}p(s',r\mid s,a)\big[r+\gamma V_k(s')\big],\qquad\text{i.e. } V_{k+1}=\mathcal{T}^*V_k.
$$

> [!example] Algorithm: value iteration
> **Input:** model $p$, tolerance $\varepsilon>0$. Initialize $V_0(s)$ arbitrarily, with $V_0(\text{terminal})=0$.
> **Repeat** for $k=0,1,2,\dots$:
> 1. For every $s$: $\;V_{k+1}(s)\leftarrow\max_a\sum_{s',r}p(s',r\mid s,a)\big[r+\gamma V_k(s')\big]$.
> 2. Stop when $\lVert V_{k+1}-V_k\rVert_\infty<\varepsilon\,\frac{1-\gamma}{2\gamma}$.
>
> **Return** the greedy policy $\pi(s)=\arg\max_a\sum_{s',r}p(s',r\mid s,a)\big[r+\gamma V_{k+1}(s')\big]$.

**Convergence.** $\mathcal{T}^*$ is a $\gamma$-contraction, so $V_k\to V^*$ from any $V_0$, with $\lVert V_k-V^*\rVert_\infty\le\gamma^k\lVert V_0-V^*\rVert_\infty$. Note that the intermediate $V_k$ need not be the value of *any* policy. Value iteration works in "value space", not "policy space".

**When to stop: a bound we can compute.** We never know $V^*$, but we do know how much the last sweep changed things:

$$
\begin{aligned}
\lVert V_k-V^*\rVert_\infty&=\lVert\mathcal{T}^*V_{k-1}-\mathcal{T}^*V^*\rVert_\infty\le\gamma\lVert V_{k-1}-V^*\rVert_\infty\\
&\le\gamma\big(\lVert V_{k-1}-V_k\rVert_\infty+\lVert V_k-V^*\rVert_\infty\big)\\
\implies\lVert V_k-V^*\rVert_\infty&\le\frac{\gamma}{1-\gamma}\,\lVert V_k-V_{k-1}\rVert_\infty.
\end{aligned}
$$

With the stopping rule above, this gives $\lVert V_{k+1}-V^*\rVert_\infty<\varepsilon/2$. What we really care about is the **policy**. If $\lVert V-V^*\rVert_\infty\le\delta$, the greedy policy $\pi_V$ satisfies $\lVert V^{\pi_V}-V^*\rVert_\infty\le\frac{2\gamma\delta}{1-\gamma}$.[^singhyee1994] Near-optimal values give near-optimal behaviour.

![[Reinforcement Learning/images/dynamic-programming-vi-convergence.svg]]
*Figure 3: value iteration on the gridworld, from $V_0=0$. Nothing changes much for the first few sweeps, while reward information travels from G and X across the grid. After that the error decays geometrically. The contraction bound $\gamma^k\lVert V_0-V^*\rVert$ is a worst case: here episodes end quickly in absorbing states, so the effective rate is faster than $\gamma$. The computable bound $\frac{\gamma}{1-\gamma}\lVert V_k-V_{k-1}\rVert$ tracks the true error and is safe to stop on.*

---

## The bigger picture

### Policy iteration vs value iteration, and everything in between

| | Policy iteration | Value iteration |
|---|---|---|
| Per iteration | full evaluation, $O(n^3)$ exact or many sweeps, plus one improvement | one sweep, $O(n^2\lvert\mathcal{A}\rvert)$ |
| Iterations needed | very few, and it terminates exactly | many; converges geometrically and never exactly |
| Intermediate iterates | values of actual policies, improving monotonically | not necessarily values of any policy |

**Modified (truncated) policy iteration** sits in between. It runs only $m$ sweeps of evaluation before each improvement: $m=1$ is value iteration, and $m=\infty$ is policy iteration. Values of $m$ in between are often fastest in practice.[^puterman1994]

### Generalized policy iteration (GPI)

![[Reinforcement Learning/images/dynamic-programming-gpi.svg]]
*Figure 4: generalized policy iteration. Evaluation pulls $V$ towards $V^\pi$, and improvement pulls $\pi$ towards greedy($V$). Each step partly undoes the other's target, yet together they converge to the one point where both conditions hold: $\pi^*$ and $V^*$.*

Abstracting away *how much* evaluation and improvement we do, and in what order, gives **generalized policy iteration**: any interleaving of "make $V$ consistent with $\pi$" and "make $\pi$ greedy with respect to $V$".[^suttonbarto] Almost every RL method is an instance:

- **Q-learning** is *sampled value iteration* on $Q$. It replaces the expectation over $p(s',r\mid s,a)$ in $Q_{k+1}(s,a)=\mathbb{E}[r+\gamma\max_{a'}Q_k(s',a')]$ with a single observed transition and a small step size.
- **Actor-critic** methods are *approximate policy iteration*. The critic approximately evaluates $\pi_\theta$, and the actor takes a [[Policy Gradients|policy-gradient]] step towards the greedy improvement instead of jumping to it.

### Asynchronous DP and the curse of dimensionality

Nothing in the contraction argument requires sweeping all states in order. **Asynchronous DP** updates states in any order, even concentrating on the states that matter (for example, those visited by the current policy). It still converges, as long as every state keeps being updated. This is the bridge to learning from experience, where the states we update are exactly the ones we visit.

The real limitation is scale. Each sweep touches every state, and the number of states grows **exponentially** with the number of state variables, which is Bellman's *curse of dimensionality*. A robot with 10 joints, each discretized into 100 positions, already has $10^{20}$ states. Overcoming it needs two ingredients beyond DP: **sampling**, so the model is not needed, and **function approximation**, so states need not be enumerated.

## Key takeaways

- DP solves the Bellman equations with a **known model**. It is planning, not learning.
- $\mathcal{T}^\pi$ and $\mathcal{T}^*$ are **$\gamma$-contractions** in $\lVert\cdot\rVert_\infty$. So $V^\pi$ and $V^*$ are unique fixed points, and iterating the operators converges geometrically from any start.
- **Policy evaluation** iterates $\mathcal{T}^\pi$. **Value iteration** iterates $\mathcal{T}^*$; stop when $\lVert V_{k+1}-V_k\rVert_\infty$ is small, using the computable error bound.
- **Policy improvement theorem:** acting greedily with respect to $V^\pi$ never hurts, and it strictly helps unless $\pi$ is already optimal. **Policy iteration** therefore terminates at $\pi^*$, usually in very few iterations.
- **GPI** (interleaved evaluation and improvement) is the blueprint for model-free RL: Q-learning is sampled VI, and actor-critic is approximate PI.

## Where this shows up next

- **Q-learning / TD learning** (future lecture): DP from samples, without a model, including the stochastic-approximation (ODE) argument for why Q-learning converges.
- [[Policy Gradients]] and actor-critic: approximate policy iteration with a parametric policy.
- [[Imitation Learning]] needs none of this machinery, which is both its appeal and its limitation.

<!-- References: footnote definitions below render as the numbered "References" list at the bottom of the page -->

[^bellman1957]: Bellman, R. (1957). *Dynamic Programming*. Princeton University Press (reprinted 2010). [doi:10.1515/9781400835386](https://doi.org/10.1515/9781400835386)
[^puterman1994]: Puterman, M. L. (1994). *Markov Decision Processes: Discrete Stochastic Dynamic Programming*. Wiley, Chapter 6. [doi:10.1002/9780470316887](https://doi.org/10.1002/9780470316887)
[^singhyee1994]: Singh, S. P., & Yee, R. C. (1994). *An Upper Bound on the Loss from Approximate Optimal-Value Functions*. Machine Learning, 16, 227–233. [doi:10.1007/BF00993308](https://doi.org/10.1007/BF00993308)
[^suttonbarto]: Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.), Chapter 4. [book website](http://incompleteideas.net/book/the-book-2nd.html)
