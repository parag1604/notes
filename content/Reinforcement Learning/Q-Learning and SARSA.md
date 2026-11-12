---
title: Q-Learning and SARSA
tags:
  - reinforcement-learning
  - q-learning
  - sarsa
  - temporal-difference
  - off-policy
  - stochastic-approximation
date: 2026-11-12
draft: false
prev: "[[Monte Carlo and Temporal-Difference Learning]]"
next: "[[Linear Function Approximation]]"
---

> **Prerequisites:** [[Monte Carlo and Temporal-Difference Learning]] (TD(0) and its stochastic-approximation view), [[Introduction to Reinforcement Learning]] ($Q^*$, the Bellman optimality equation, on- vs off-policy), and $\varepsilon$-greedy exploration from [[Multi-Armed Bandits]]. Contraction facts are proved in [[Dynamic Programming]] and the convergence machinery in [[Stochastic Approximation]]; neither is needed to follow the algorithms.
>
> **TL;DR:** To *control* an MDP without a model, learn action values. **Q-learning** applies TD to the Bellman *optimality* equation: $Q(s_t,a_t)\leftarrow Q(s_t,a_t)+\alpha\big[r_t+\gamma\max_{a'}Q(s_{t+1},a')-Q(s_t,a_t)\big]$. It is **off-policy**: it learns $Q^*$ while behaving with any policy that keeps exploring. It converges to $Q^*$ with probability 1 **if every state-action pair is visited infinitely often and the per-pair step sizes satisfy $\sum\alpha=\infty$, $\sum\alpha^2<\infty$**. The proof writes it as an asynchronous stochastic approximation of a max-norm contraction, whose ODE is $\dot Q=D(\mathcal{T}^*Q-Q)$. **SARSA** uses the action actually taken next, so it is **on-policy**: it learns the value of the exploring policy itself, which makes it more cautious. Q-learning's $\max$ is **biased upwards**; Double Q-learning fixes this.
>
> **Local notation.** $Q_t$ is the estimate after $t$ updates and $\alpha_t(s,a)$ the step size used for pair $(s,a)$ at time $t$. The **behaviour policy**, which generates the data, is written $b(a\mid s)$; the **target policy**, whose values we learn, is $\pi$. $\mathcal{T}^*$ acts on Q-functions here, $Q\in\mathbb{R}^{|\mathcal{S}||\mathcal{A}|}$.
>
> **Code:** [RL-Basics](https://github.com/parag1604/RL-Basics): tabular Q-learning (file 03); [Deep-Q-Networks](https://github.com/parag1604/Deep-Q-Networks): a DQN with experience replay and a target network.

## Motivation

[[Monte Carlo and Temporal-Difference Learning|TD learning]] evaluates a fixed policy. To *improve* a policy we follow the generalized-policy-iteration recipe: evaluate, act greedily, repeat. Greedy improvement from a state-value function needs the model,

$$
\pi'(s)=\arg\max_a\sum_{s'}p(s'\mid s,a)\big[r(s,a,s')+\gamma V(s')\big],
$$

which we do not have. Greedy improvement from an **action-value** function does not: $\pi'(s)=\arg\max_aQ(s,a)$. So model-free control learns $Q$, not $V$.

There are two natural ways to do it with TD:

- learn $Q^*$ directly, by sampling the Bellman **optimality** equation: **Q-learning**;
- learn $Q^\pi$ of the current (exploring) policy and keep making that policy greedier: **SARSA**.

The difference between them, learning about the greedy policy while behaving differently versus learning about the policy one is actually following, is the difference between **off-policy** and **on-policy** learning. It has visible consequences (Figure 3).

---

## The Bellman optimality equation for $Q$

Recall from [[Introduction to Reinforcement Learning|the introduction]] that $Q^*(s,a)$ is the expected return from taking $a$ in $s$ and acting optimally afterwards. It satisfies

$$
Q^*(s,a)=\sum_{s'}p(s'\mid s,a)\Big[r(s,a,s')+\gamma\max_{a'}Q^*(s',a')\Big],
$$

with $V^*(s)=\max_aQ^*(s,a)$, and every policy that is greedy with respect to $Q^*$ is optimal. Write the right-hand side as an operator on Q-functions:

$$
(\mathcal{T}^*Q)(s,a)=\sum_{s'}p(s'\mid s,a)\Big[r(s,a,s')+\gamma\max_{a'}Q(s',a')\Big].
$$

$Q^*$ is its fixed point. For $\gamma<1$, $\mathcal{T}^*$ is a **$\gamma$-contraction** in $\lVert Q\rVert_\infty=\max_{s,a}|Q(s,a)|$. The proof is one line: $|\max_{a'}f(a')-\max_{a'}g(a')|\le\max_{a'}|f(a')-g(a')|$, so each component of $\mathcal{T}^*Q-\mathcal{T}^*\bar Q$ is at most $\gamma\lVert Q-\bar Q\rVert_\infty$ in absolute value (the $V$-version is in [[Dynamic Programming]]). So $Q^*$ is unique, and **Q-value iteration** $Q_{k+1}=\mathcal{T}^*Q_k$ converges to it from any start, with $\lVert Q_k-Q^*\rVert_\infty\le\gamma^k\lVert Q_0-Q^*\rVert_\infty$.

> [!note] The episodic case, $\gamma=1$
> In episodic tasks without discounting (a [[Stochastic Shortest Path Problems|stochastic shortest path]] problem in reward form), there is an absorbing, reward-free terminal state with $Q(\text{terminal},\cdot)=0$. A policy is **proper** if it reaches the terminal state with probability 1 from every state. The standard assumptions are:
> - **(A)** at least one proper policy exists;
> - **(B)** every improper policy has value $-\infty$ from at least one state (for example, every step costs a negative reward, as in "reach the goal quickly").
>
> Under (A) and (B), $Q^*$ is the unique fixed point of $\mathcal{T}^*$ (with $\gamma=1$) and Q-value iteration converges to it. Assumption (B) is needed because otherwise a policy that never terminates could collect zero (or even positive) reward forever. It would then look at least as good as reaching the goal, values could be infinite, and the Bellman equation could have many solutions. If **all** policies are proper, $\mathcal{T}^*$ is even a contraction, in a weighted max-norm (below).

---

## Q-learning

### From Q-value iteration to samples

Q-value iteration needs the expectation over $s'$. Replace it by a sample and take a small step, exactly as TD(0) did for $\mathcal{T}^\pi$. If we have a **simulator** that can produce $s'\sim p(\cdot\mid s,a)$ for any pair on demand, we can update every pair at every iteration:

$$
\begin{aligned}
Q_{n+1}(s,a)&=Q_n(s,a)+\alpha_n\Big[r(s,a,s')+\gamma\max_{a'}Q_n(s',a')-Q_n(s,a)\Big]\\
&\qquad\text{for all }(s,a),
\end{aligned}
$$

with a fresh $s'$ for each pair. This is **synchronous Q-learning**.

One can motivate it as stochastic gradient descent on $\tfrac12\lVert Q-Q^*\rVert_2^2$: the gradient is $Q-Q^*$, and $r+\gamma\max_{a'}Q^*(s',a')$ is an unbiased sample of $Q^*(s,a)$. Since $Q^*$ is unknown, plug in the current $Q_n$. After that substitution the update is **not** the gradient of any fixed objective, because the target moves with $Q_n$. It is a stochastic **fixed-point iteration** for $\mathcal{T}^*$, and that is how its convergence is proved.

### Online Q-learning

Usually there is no simulator, only a single stream of experience. Then only the pair just visited can be updated. This is the algorithm of Watkins.[^watkinsdayan1992]

> [!example] Algorithm: Q-learning (online, tabular)
> **Input:** behaviour policy derived from $Q$ (e.g. $\varepsilon$-greedy), step sizes. Initialize $Q(s,a)$ arbitrarily, with $Q(\text{terminal},\cdot)=0$.
> **For** each episode, from $s_0$, and each step $t=0,1,\dots$ until $s_t$ is terminal:
> 1. Choose $a_t$ by the behaviour policy, e.g. $\varepsilon$-greedy with respect to $Q$.
> 2. Take $a_t$; observe $r_t$ and $s_{t+1}$.
> 3. $Q(s_t,a_t)\leftarrow Q(s_t,a_t)+\alpha_t(s_t,a_t)\Big[r_t+\gamma\max_{a'}Q(s_{t+1},a')-Q(s_t,a_t)\Big]$.

A standard step size is $\alpha_t(s,a)=\frac{c_1}{c_2+n_t(s,a)}$, where $n_t(s,a)$ counts the visits to $(s,a)$ so far and $c_1,c_2>0$. The **next** action $a_{t+1}$ is again chosen by the behaviour policy. The greedy action $\arg\max_{a'}Q(s_{t+1},a')$ appears only inside the **target**; the agent need not take it.

### Q-learning is off-policy

Two policies are involved:

- the **target policy** $\pi$, whose values are being learned. For Q-learning it is the greedy policy with respect to $Q$, and in the limit the optimal policy;
- the **behaviour policy** $b$, which chooses the actions, for example $\varepsilon$-greedy.

They differ, so Q-learning is **off-policy**. Why does it need no importance-sampling correction, unlike off-policy policy evaluation (see [[Policy Gradients#Off-policy Policy Gradients|off-policy policy gradients]])? The target $r_t+\gamma\max_{a'}Q(s_{t+1},a')$ is an unbiased sample of $(\mathcal{T}^*Q)(s_t,a_t)$ **given** $(s_t,a_t)$. Its distribution depends only on $p(\cdot\mid s_t,a_t)$, not on how $a_t$ was chosen, and the $\max$ replaces the next action, so the behaviour policy never enters the target. The behaviour policy only decides **how often** each pair is updated. As long as every pair keeps being updated, any behaviour will do: an old policy, a human, or a replay buffer of past transitions.

---

## Convergence of Q-learning

### The theorem

> [!note] Theorem (convergence of tabular Q-learning)
> Assume:
> 1. **Finite MDP, bounded rewards:** $\mathcal{S}$ and $\mathcal{A}$ are finite, and rewards are bounded (or random with bounded variance), and $\gamma\in[0,1)$.
> 2. **Asynchronous updates:** at time $t$ only $(s_t,a_t)$ is updated, so $\alpha_t(s,a)=0$ for $(s,a)\ne(s_t,a_t)$; each $\alpha_t(s,a)\in[0,1]$ depends only on the past.
> 3. **Robbins–Monro step sizes per pair, which includes exploration:** with probability 1, condition (3) below holds for **every** pair $(s,a)$.
>
> $$
> \sum_{t}\alpha_t(s,a)=\infty,\qquad\sum_t\alpha_t(s,a)^2<\infty .
> \tag{3}
> $$
>
> Then $Q_t(s,a)\to Q^*(s,a)$ for all $(s,a)$, with probability 1.

Condition 3 can only hold if **every state-action pair is visited infinitely often**: a pair that is updated finitely often has a finite sum of step sizes. This is the exploration requirement, stated explicitly. Any behaviour policy that guarantees it works, for example $\varepsilon$-greedy with a fixed $\varepsilon>0$ in an MDP where every state keeps being reached. With visit-count step sizes $\alpha=\frac{c_1}{c_2+n_t(s,a)}$ condition 3 then holds, because $\sum_n\frac1n=\infty$ and $\sum_n\frac1{n^2}<\infty$; so does $\alpha=n_t(s,a)^{-\kappa}$ with $\kappa\in(\tfrac12,1]$.

Watkins and Dayan gave the first proof, by an "action-replay" construction.[^watkinsdayan1992] Tsitsiklis, and Jaakkola, Jordan and Singh, proved it as an instance of asynchronous stochastic approximation.[^tsitsiklis1994]<sup>,</sup>[^jaakkola1994]

> [!proof]- Proof: Q-learning as an asynchronous contraction iteration
> Write the update as
> $$
> Q_{t+1}(s,a)=Q_t(s,a)+\alpha_t(s,a)\Big[(\mathcal{T}^*Q_t)(s,a)-Q_t(s,a)+w_t(s,a)\Big],
> $$
> where, at the visited pair, the noise is the difference between the sampled and the expected target:
> $$
> w_t=r_t+\gamma\max_{a'}Q_t(s_{t+1},a')-(\mathcal{T}^*Q_t)(s_t,a_t).
> $$
> Check the three hypotheses of [[Stochastic Approximation#Asynchronous stochastic approximation|Tsitsiklis's theorem]].
> 1. **Step sizes:** this is condition 3.
> 2. **Noise:** given the history $\mathcal{F}_t$ (which includes $s_t$, $a_t$ and $Q_t$), $s_{t+1}\sim p(\cdot\mid s_t,a_t)$ and $r_t$ is drawn from the reward distribution of that transition. So $\mathbb{E}[w_t\mid\mathcal{F}_t]=0$ by the definition of $\mathcal{T}^*$. With $|r_t|\le R_{\max}$, $|w_t|\le2R_{\max}+2\gamma\lVert Q_t\rVert_\infty$, hence $\mathbb{E}[w_t^2\mid\mathcal{F}_t]\le K\big(1+\lVert Q_t\rVert_\infty^2\big)$ for a constant $K$ (using $(x+y)^2\le2x^2+2y^2$).
> 3. **Contraction:** $\mathcal{T}^*$ is a $\gamma$-contraction in $\lVert\cdot\rVert_\infty$ (shown above), hence a max-norm pseudo-contraction with fixed point $Q^*$, weights $\xi\equiv1$ and modulus $\gamma$.
>
> Tsitsiklis's theorem gives boundedness of the iterates and $Q_t\to Q^*$ with probability 1. $\blacksquare$

> [!warning] What the conditions rule out
> - **Constant step sizes** violate $\sum\alpha^2<\infty$. The estimates then hover around $Q^*$ without converging (Figure 2).
> - **Insufficient exploration** violates $\sum\alpha=\infty$ for some pairs. Pure greedy behaviour can lock onto a suboptimal action forever, exactly as in [[Multi-Armed Bandits|bandits]].
> - **Function approximation** breaks the contraction argument altogether. Q-learning with features can diverge ([[Off-Policy Learning with Function Approximation]]).

### The episodic case: stochastic shortest paths

With $\gamma=1$ and a terminal state, set $Q_t(\text{terminal},\cdot)=0$ for all $t$. The same theorem holds in two cases:[^bertsekastsitsiklis1996]

1. **All policies are proper.** Then there are weights $\xi(s)>0$ and $\beta\in[0,1)$ such that $\sum_{s'\ne\text{terminal}}p(s'\mid s,a)\,\xi(s')\le\beta\,\xi(s)$ for all $s,a$ (proved in [[Stochastic Shortest Path Problems]]). With this $\xi$, $\mathcal{T}^*$ is a $\beta$-contraction in the weighted max-norm $\lVert Q\rVert_\xi=\max_{s,a}|Q(s,a)|/\xi(s)$, and the proof above goes through with $\lVert\cdot\rVert_\xi$ in place of $\lVert\cdot\rVert_\infty$.
2. **Assumptions (A) and (B) hold, and the iterates are bounded**, $\sup_t\lVert Q_t\rVert_\infty<\infty$ with probability 1. Here $\mathcal{T}^*$ need not be a contraction. Boundedness was a hypothesis in the original analysis; Yu and Bertsekas later proved that it holds automatically under (A) and (B).[^yubertsekas2013]

> [!proof]- Weighted max-norm contraction (case 1)
> For any $Q,\bar Q$ and any $(s,a)$, using $|\max f-\max g|\le\max|f-g|$ and $Q(\text{terminal},\cdot)=\bar Q(\text{terminal},\cdot)=0$:
> $$
> \begin{aligned}
> \big|(\mathcal{T}^*Q)(s,a)-(\mathcal{T}^*\bar Q)(s,a)\big|&\le\sum_{s'\ne\text{term.}}p(s'\mid s,a)\max_{a'}\big|Q(s',a')-\bar Q(s',a')\big|\\
> &\le\sum_{s'\ne\text{term.}}p(s'\mid s,a)\,\xi(s')\,\lVert Q-\bar Q\rVert_\xi\\
> &\le\beta\,\xi(s)\,\lVert Q-\bar Q\rVert_\xi .
> \end{aligned}
> $$
> Divide by $\xi(s)$ and take the maximum over $(s,a)$: $\lVert\mathcal{T}^*Q-\mathcal{T}^*\bar Q\rVert_\xi\le\beta\lVert Q-\bar Q\rVert_\xi$. The noise bounds are as before, so Tsitsiklis's theorem applies. $\blacksquare$
>
> **Case 2 (sketch).** $\mathcal{T}^*$ is **monotone** ($Q\le\bar Q$ componentwise implies $\mathcal{T}^*Q\le\mathcal{T}^*\bar Q$), and for a constant $c\ge0$ and the all-ones vector $\mathbf{1}$,
> $$
> \mathcal{T}^*Q-c\mathbf{1}\le\mathcal{T}^*(Q-c\mathbf{1})\le\mathcal{T}^*(Q+c\mathbf{1})\le\mathcal{T}^*Q+c\mathbf{1},
> $$
> because the transition probabilities to non-terminal states sum to at most 1. These two facts alone do **not** give uniqueness of the fixed point; that comes from SSP theory under (A) and (B). With bounded iterates, one traps $Q_t$ between two deterministic sequences, obtained by iterating $\mathcal{T}^*$ from above and from below, which both converge to $Q^*$; the noise is handled as in the contraction case. The details are in Bertsekas and Tsitsiklis, Section 5.6.[^bertsekastsitsiklis1996]

### The ODE view

The [[Stochastic Approximation#The ODE method|ODE method]] gives the same answer and more intuition.

**Synchronous Q-learning.** Every component is updated with the same step size, and the noise is a martingale difference, so the iterates track

$$
\dot Q(\tau)=\mathcal{T}^*Q(\tau)-Q(\tau)
$$

in algorithmic time $\tau=\sum_n\alpha_n$. Because $\mathcal{T}^*$ is a $\gamma$-contraction in $\lVert\cdot\rVert_\infty$, $\lVert Q(\tau)-Q^*\rVert_\infty\le e^{-(1-\gamma)\tau}\lVert Q(0)-Q^*\rVert_\infty$, so $Q^*$ is globally asymptotically stable. This must be proved **in the max-norm** directly ([[Stochastic Approximation#The contraction lemma|the contraction lemma]]). Proving it in a $p$-norm and letting $p\to\infty$ is not valid, since a max-norm contraction need not contract in any finite $p$-norm.

### Asynchronous Q-learning and the weighted ODE

Online Q-learning updates one pair at a time along a single trajectory of the behaviour policy $b$. Suppose the Markov chain of state-action pairs under $b$ is irreducible, with stationary distribution $d_b(s)\,b(a\mid s)$, and use one global step-size sequence $\alpha_t$ for whichever pair is visited. Over a stretch of time, pair $(s,a)$ is updated a fraction $d_b(s)b(a\mid s)$ of the time, so its component moves at that relative speed. The limiting ODE is

$$
\dot Q(\tau)=D\big(\mathcal{T}^*Q(\tau)-Q(\tau)\big),\qquad D=\operatorname{diag}\big(d_b(s)\,b(a\mid s)\big)_{(s,a)} .
$$

The weighting $D$ is essential. Without it the ODE describes synchronous Q-learning, which is not what an agent following $b$ does (Figure 1). For the ODE to have $Q^*$ as a stable equilibrium we need $D$ to be positive, $d_b(s)\,b(a\mid s)>0$ for every pair: the behaviour policy must try every action in every state, and the chain must reach every state.[^borkar1998]<sup>,</sup>[^borkarmeyn2000] Lee and He give an alternative analysis of this ODE as a switching linear system.[^leehe2019]

> [!note] Proposition (global stability of the weighted ODE)
> Let $D$ be diagonal with entries $d_{\min}=\min_{s,a}D_{(s,a)}>0$, and let $F$ be a $\gamma$-contraction in $\lVert\cdot\rVert_\infty$ with fixed point $x^*$. Every solution of $\dot x=D\big(F(x)-x\big)$ satisfies
> $$
> \lVert x(\tau)-x^*\rVert_\infty\le e^{-d_{\min}(1-\gamma)\tau}\,\lVert x(0)-x^*\rVert_\infty .
> $$
> With $F=\mathcal{T}^*$ this covers asynchronous Q-learning; with $F=\mathcal{T}^\pi$, asynchronous [[Monte Carlo and Temporal-Difference Learning|TD(0)]].

> [!proof]- Proof (upper Dini derivative of the max-norm)
> Let $e(\tau)=x(\tau)-x^*$ and $m(\tau)=\lVert e(\tau)\rVert_\infty$. Since $F(x^*)=x^*$, each component satisfies
> $$
> \dot e_i=D_i\big[(F(x))_i-x^*_i-e_i\big],\qquad\big|(F(x))_i-x^*_i\big|\le\gamma\,m .
> $$
> Take an index $i$ that attains the maximum, $|e_i(\tau)|=m(\tau)$. If $e_i=m>0$, then $\dot e_i\le D_i(\gamma m-m)\le-d_{\min}(1-\gamma)m$. If $e_i=-m<0$, then $\dot e_i\ge D_i(m-\gamma m)$, so $\frac{d}{d\tau}|e_i|\le-d_{\min}(1-\gamma)m$ again.
>
> The function $m$ is the maximum of the finitely many differentiable functions $\pm e_i$, so it may have kinks. Its upper right Dini derivative, $D^+m(\tau)=\limsup_{h\downarrow0}\frac{m(\tau+h)-m(\tau)}{h}$, is at most the largest derivative among the functions attaining the maximum. Hence $D^+m\le-c\,m$ with $c=d_{\min}(1-\gamma)$. Then $D^+\big(e^{c\tau}m(\tau)\big)\le0$, and a continuous function with non-positive upper right Dini derivative is non-increasing. So $m(\tau)\le e^{-c\tau}m(0)$. $\blacksquare$
>
> The same proof works in a weighted max-norm (replace $|e_i|$ by $|e_i|/\xi_i$), which covers the SSP case with all policies proper. It also shows why a *max*-norm is the right tool: a diagonal rescaling changes how fast each component moves, but the component farthest from $x^*$ is still pulled inwards.

![[Reinforcement Learning/images/q-learning-sarsa-ode.svg]]
*Figure 1: online Q-learning along one trajectory of a 5-state, 2-action MDP ($\gamma=0.8$). The behaviour policy picks action 1 with probability only 0.15, so the visit frequencies of the pairs range from 0.37 down to 0.0045. Thin lines: two components of $Q_t$, plotted against $\tau_t=\sum_{k<t}\alpha_k$. Dashed: the weighted ODE $\dot Q=D(\mathcal{T}^*Q-Q)$ from the same start. Dotted: the unweighted ODE. At $\tau=100$ the rarely visited pair has $Q_t=0.44$, the weighted ODE predicts $0.55$, and the unweighted ODE is already at $Q^*=2.68$. The iterates follow the weighted ODE, and reach $\lVert Q_t-Q^*\rVert_\infty=0.023$ at $\tau\approx1300$.*

With **per-pair** step sizes $\alpha_t(s,a)$ that depend on visit counts, each pair runs on its own clock, and the rare pairs are no longer slowed down in this way. Either way, convergence needs every pair to be visited infinitely often.

### Step sizes in practice

The theorem allows many step-size schedules, but their speed differs enormously.

![[Reinforcement Learning/images/q-learning-sarsa-step-sizes.svg]]
*Figure 2: synchronous Q-learning on a random 8-state, 2-action MDP with $\gamma=0.9$ (mean of 20 runs). After $10^5$ iterations the error is $2.10$ for $\alpha_n=1/n$, $0.017$ for $\alpha_n=n^{-0.6}$, and $0.006$ for the rescaled schedule $\alpha_n=10/(10+n)$. The constant step size $\alpha=0.1$ is fast at first but stalls at a noise floor of about $0.17$.*

The plain sample-average step $\alpha_n=1/n$ satisfies the theorem and is still hopeless. It averages targets built from all past estimates with equal weight, so the poor early estimates are forgotten only very slowly, and the bootstrapped error shrinks by only a factor of roughly $\gamma$ per "effective sweep". Even-Dar and Mansour showed that with $\alpha=1/n$ the time to converge can be **exponential** in $\frac1{1-\gamma}$, while polynomial step sizes $1/n^{\omega}$, $\omega\in(\tfrac12,1)$, give convergence times polynomial in $\frac1{1-\gamma}$.[^evendar2003] The schedule $\alpha_n=\frac{b}{a+n}$ with $a=b=\frac1{1-\gamma}$, which is $\frac{1}{1+(1-\gamma)n}$, keeps the early steps large enough to wash out the initial error and was the fastest here.

---

## Exploration

The convergence theorem only asks that every pair be visited infinitely often. The agent's **online** reward also matters, and that depends on how much it explores.

- **$\varepsilon$-greedy:** with probability $1-\varepsilon$ take a greedy action, otherwise a uniformly random one. With fixed $\varepsilon>0$ every action keeps being tried, but the agent never stops paying for exploration.
- **Decaying $\varepsilon$:** for example $\varepsilon_t(s)=c/n_t(s)$, where $n_t(s)$ counts visits to $s$. Every action is still tried infinitely often (since $\sum_n\frac{c}{n}=\infty$), while the policy becomes greedy in the limit.

Such schedules are called **GLIE**, *greedy in the limit with infinite exploration*: (i) every state-action pair is visited infinitely often, and (ii) the behaviour policy converges to the greedy policy with respect to $Q_t$.[^singh2000] For Q-learning, (i) alone is enough for convergence of $Q_t$; (ii) only improves the online reward. For SARSA, both are needed, as we see next.

---

## SARSA: on-policy TD control

**SARSA** applies TD(0) to $Q^\pi$ of the policy being followed. Its update uses the quintuple $(s_t,a_t,r_t,s_{t+1},a_{t+1})$, hence the name:

$$
Q(s_t,a_t)\leftarrow Q(s_t,a_t)+\alpha_t\Big[r_t+\gamma\,Q(s_{t+1},a_{t+1})-Q(s_t,a_t)\Big],
$$

where $a_{t+1}$ is the action the agent **actually takes** next.

> [!example] Algorithm: SARSA (tabular)
> **Input:** step sizes, exploration schedule. Initialize $Q$ arbitrarily, with $Q(\text{terminal},\cdot)=0$.
> **For** each episode: observe $s_0$; choose $a_0$ $\varepsilon$-greedily with respect to $Q$. **For** $t=0,1,\dots$ until $s_t$ is terminal:
> 1. Take $a_t$; observe $r_t$ and $s_{t+1}$.
> 2. Choose $a_{t+1}$ $\varepsilon$-greedily with respect to $Q$ (any value if $s_{t+1}$ is terminal).
> 3. $Q(s_t,a_t)\leftarrow Q(s_t,a_t)+\alpha_t\big[r_t+\gamma Q(s_{t+1},a_{t+1})-Q(s_t,a_t)\big]$.

The only difference from Q-learning is the target: $Q(s_{t+1},a_{t+1})$ instead of $\max_{a'}Q(s_{t+1},a')$. The consequences are larger than the change.

- In SARSA the behaviour policy and the target policy are **the same** policy, the $\varepsilon$-greedy policy with respect to the current $Q$. SARSA is **on-policy**: it estimates the value of the policy it is following, exploration included.
- **Convergence.** With a fixed $\varepsilon$, SARSA learns about the $\varepsilon$-greedy policy, not about the optimal one. Singh, Jaakkola, Littman and Szepesvári proved that SARSA(0) converges to $Q^*$, and its policy to an optimal one, with probability 1 under the conditions of the Q-learning theorem **plus a GLIE exploration schedule**.[^singh2000] Exploration must vanish in the limit, because the exploring actions themselves enter SARSA's target.

### Cliff walking: optimal values versus safe behaviour

The cliff-walking gridworld (Sutton and Barto, Example 6.6) makes the difference concrete. The agent must walk from **S** to **G** along a $4\times12$ grid. Every step gives reward $-1$; stepping into the cliff along the bottom edge gives $-100$ and sends the agent back to **S**. The task is episodic with $\gamma=1$. Both agents explore $\varepsilon$-greedily with $\varepsilon=0.1$ and use $\alpha=0.5$.

![[Reinforcement Learning/images/q-learning-sarsa-cliff.svg]]
*Figure 3: cliff walking. Top: greedy paths after 500 episodes (one run). Q-learning's path hugs the cliff, which is the optimal path (13 steps, return $-13$). SARSA's path keeps a safe distance (17 steps). Bottom: the reward actually collected per episode while learning, averaged over 100 runs. Over episodes 401–500 SARSA averages $-26.5$ and Q-learning only $-49.4$: walking next to the cliff with $\varepsilon$-greedy behaviour, Q-learning regularly falls off. Expected SARSA averages $-20.7$.*

Q-learning learns the values of the **greedy** policy, so it finds the optimal path. But it keeps behaving $\varepsilon$-greedily, and next to the cliff a random step is disastrous. SARSA's values include the cost of its own occasional random steps, so it learns that cells next to the cliff are dangerous *for an agent that explores*, and walks around. Which is "right" depends on the question:

- if the learned policy will later be executed **greedily**, Q-learning's answer is the better one;
- if the agent must keep exploring while it acts, or mistakes during learning are costly, SARSA's caution is the better one.

With a GLIE schedule ($\varepsilon\to0$) both converge to the optimal path.

### Expected SARSA

**Expected SARSA** replaces the sampled next action by its expectation under the target policy $\pi$:

$$
Q(s_t,a_t)\leftarrow Q(s_t,a_t)+\alpha_t\Big[r_t+\gamma\sum_{a'}\pi(a'\mid s_{t+1})\,Q(s_{t+1},a')-Q(s_t,a_t)\Big].
$$

This removes the variance due to sampling $a_{t+1}$, at the cost of a sum over actions, so it can use larger step sizes.[^vanseijen2009] With $\pi$ the agent's own $\varepsilon$-greedy policy it is on-policy and, as Figure 3 shows, it does better than SARSA online. With $\pi$ greedy it becomes exactly Q-learning. Since $\pi$ need not be the behaviour policy, Expected SARSA is a general off-policy method that contains both.

---

## Maximization bias and Double Q-learning

Q-learning's target takes a **maximum over noisy estimates**. Even if every $Q(s',a')$ is an unbiased estimate of its true value, the maximum is not: since $\max$ is convex, Jensen's inequality gives

$$
\mathbb{E}\Big[\max_{a'}\hat Q(s',a')\Big]\ge\max_{a'}\mathbb{E}\big[\hat Q(s',a')\big],
$$

usually with strict inequality. The estimated value of $s'$ is too optimistic, and this error is then propagated backwards by bootstrapping.

Sutton and Barto's Example 6.7 shows the effect. From state A, "right" ends the episode with reward 0. "Left" leads with reward 0 to state B, where each of 10 actions ends the episode with a reward drawn from $\mathcal{N}(-0.1,1)$. So "left" is worse in expectation, and an $\varepsilon$-greedy agent with $\varepsilon=0.1$ should choose it only 5% of the time. But the maximum of 10 noisy estimates at B is positive for a long time, so Q-learning is drawn to the left.

![[Reinforcement Learning/images/q-learning-sarsa-maximization-bias.svg]]
*Figure 4: fraction of episodes in which the agent goes left from A (2,000 runs, $\alpha=0.1$, $\varepsilon=0.1$). Q-learning goes left in 94% of episodes at its peak (episode 18) and still in 12% over episodes 251–300. Double Q-learning stays close to the optimal 5% (6.6% over episodes 251–300).*

**Double Q-learning** decouples choosing the maximizing action from evaluating it.[^vanhasselt2010] Keep two tables $Q^A$ and $Q^B$. At each step, with probability $\tfrac12$ update $Q^A$:

$$
\begin{aligned}
a^\star&=\arg\max_{a'}Q^A(s_{t+1},a'),\\
Q^A(s_t,a_t)&\leftarrow Q^A(s_t,a_t)+\alpha_t\Big[r_t+\gamma\,Q^B\big(s_{t+1},a^\star\big)-Q^A(s_t,a_t)\Big],
\end{aligned}
$$

and otherwise update $Q^B$ with the roles swapped. Behave $\varepsilon$-greedily with respect to $Q^A+Q^B$. Because $Q^B$'s errors are independent of the choice of $a^\star$ made with $Q^A$, the target is no longer biased upwards (it can be slightly pessimistic instead). Van Hasselt proved that Double Q-learning also converges to $Q^*$ under conditions like those of the Q-learning theorem.

---

## Beyond tables

Everything above assumes a table with one entry per pair, visited infinitely often. For large or continuous state spaces $Q$ must be approximated, $Q(s,a)\approx Q_\theta(s,a)$, and the contraction argument no longer applies: Q-learning with function approximation can diverge. **DQN** made Q-learning work with deep networks by adding an experience-replay buffer (which exploits the off-policy property) and a slowly updated *target network* that stabilizes the bootstrapped target.[^mnih2015] Its successor, Double DQN, uses exactly the decoupling above. The theory of what goes wrong, and what can be guaranteed, is the subject of [[Linear Function Approximation]] and [[Off-Policy Learning with Function Approximation]].

## Key takeaways

- Model-free control learns **action values**, because greedy improvement from $Q$ needs no model.
- **Q-learning** samples the Bellman optimality equation: $Q(s_t,a_t)\leftarrow Q(s_t,a_t)+\alpha\big[r_t+\gamma\max_{a'}Q(s_{t+1},a')-Q(s_t,a_t)\big]$. It is **off-policy**: the greedy target policy differs from the exploring behaviour policy, and no importance weights are needed.
- **Convergence (tabular, $\gamma<1$):** $Q_t\to Q^*$ with probability 1 if rewards are bounded, **every $(s,a)$ is visited infinitely often**, and the per-pair step sizes satisfy $\sum_t\alpha_t(s,a)=\infty$ and $\sum_t\alpha_t(s,a)^2<\infty$. The proof is Tsitsiklis's asynchronous stochastic-approximation theorem applied to the max-norm contraction $\mathcal{T}^*$. Episodic tasks with $\gamma=1$ need all policies proper, or assumptions (A) and (B).
- Along one trajectory the iterates follow the ODE $\dot Q=D(\mathcal{T}^*Q-Q)$, with $D$ the state-action visit frequencies; $Q^*$ is globally stable because a max-norm contraction is robust to diagonal rescaling.
- The step-size schedule matters a great deal in practice: $1/n$ is very slow, while $\frac{1}{1+(1-\gamma)n}$ or $n^{-0.6}$ are fast.
- **SARSA** is on-policy: it learns the value of its own exploring policy. It is safer while learning, and it needs **GLIE** exploration to converge to $Q^*$. Expected SARSA reduces its variance and contains Q-learning as a special case.
- The $\max$ in Q-learning causes **maximization bias**; **Double Q-learning** removes it by selecting and evaluating with different tables.

## Where this shows up next

- [[Linear Function Approximation]]: TD with features instead of tables, and its projected fixed point.
- [[Off-Policy Learning with Function Approximation]]: why off-policy bootstrapping with function approximation can diverge (the "deadly triad"), and the methods that fix it.
- Part II, model-based RL: [[Finite-Horizon Dynamic Programming]], [[Dynamic Programming]] (Q-learning is sampled value iteration) and [[Model-Based Reinforcement Learning]] (learn $p$, then plan).
- [[Stochastic Approximation]]: the general theorems used in the convergence proof.

<!-- References: footnote definitions below render as the numbered "References" list at the bottom of the page -->

[^watkinsdayan1992]: Watkins, C. J. C. H., & Dayan, P. (1992). *Q-learning*. Machine Learning, 8, 279–292. [doi:10.1007/BF00992698](https://doi.org/10.1007/BF00992698)
[^tsitsiklis1994]: Tsitsiklis, J. N. (1994). *Asynchronous Stochastic Approximation and Q-Learning*. Machine Learning, 16(3), 185–202. [doi:10.1007/BF00993306](https://doi.org/10.1007/BF00993306)
[^jaakkola1994]: Jaakkola, T., Jordan, M. I., & Singh, S. P. (1994). *On the Convergence of Stochastic Iterative Dynamic Programming Algorithms*. Neural Computation, 6(6), 1185–1201. [doi:10.1162/neco.1994.6.6.1185](https://doi.org/10.1162/neco.1994.6.6.1185)
[^bertsekastsitsiklis1996]: Bertsekas, D. P., & Tsitsiklis, J. N. (1996). *Neuro-Dynamic Programming*. Athena Scientific, Sections 4.3 and 5.6. [book website](http://www.athenasc.com/ndpbook.html)
[^yubertsekas2013]: Yu, H., & Bertsekas, D. P. (2013). *On Boundedness of Q-Learning Iterates for Stochastic Shortest Path Problems*. Mathematics of Operations Research, 38(2), 209–227. [doi:10.1287/moor.1120.0562](https://doi.org/10.1287/moor.1120.0562)
[^borkar1998]: Borkar, V. S. (1998). *Asynchronous Stochastic Approximations*. SIAM Journal on Control and Optimization, 36(3), 840–851. [doi:10.1137/S0363012995282784](https://doi.org/10.1137/S0363012995282784)
[^borkarmeyn2000]: Borkar, V. S., & Meyn, S. P. (2000). *The O.D.E. Method for Convergence of Stochastic Approximation and Reinforcement Learning*. SIAM Journal on Control and Optimization, 38(2), 447–469. [doi:10.1137/S0363012997331639](https://doi.org/10.1137/S0363012997331639)
[^leehe2019]: Lee, D., & He, N. (2019). *A Unified Switching System Perspective and O.D.E. Analysis of Q-Learning Algorithms*. [arXiv:1912.02270](https://arxiv.org/abs/1912.02270)
[^evendar2003]: Even-Dar, E., & Mansour, Y. (2003). *Learning Rates for Q-learning*. Journal of Machine Learning Research, 5, 1–25. [jmlr.org](https://www.jmlr.org/papers/v5/evendar03a.html)
[^singh2000]: Singh, S., Jaakkola, T., Littman, M. L., & Szepesvári, C. (2000). *Convergence Results for Single-Step On-Policy Reinforcement-Learning Algorithms*. Machine Learning, 38(3), 287–308. [doi:10.1023/A:1007678930559](https://doi.org/10.1023/A:1007678930559)
[^vanseijen2009]: van Seijen, H., van Hasselt, H., Whiteson, S., & Wiering, M. (2009). *A Theoretical and Empirical Analysis of Expected Sarsa*. IEEE Symposium on Adaptive Dynamic Programming and Reinforcement Learning, 177–184. [doi:10.1109/ADPRL.2009.4927542](https://doi.org/10.1109/ADPRL.2009.4927542)
[^vanhasselt2010]: van Hasselt, H. (2010). *Double Q-learning*. Advances in Neural Information Processing Systems 23. [link](https://papers.nips.cc/paper_files/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html)
[^mnih2015]: Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2015). *Human-Level Control through Deep Reinforcement Learning*. Nature, 518, 529–533. [doi:10.1038/nature14236](https://doi.org/10.1038/nature14236)
