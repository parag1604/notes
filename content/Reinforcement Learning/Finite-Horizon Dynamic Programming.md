---
title: Finite-Horizon Dynamic Programming
tags:
  - reinforcement-learning
  - dynamic-programming
  - finite-horizon
  - backward-induction
  - model-based
date: 2026-11-12
draft: false
prev: "[[Off-Policy Learning with Function Approximation]]"
next: "[[Dynamic Programming]]"
---

> **Prerequisites:** [[Introduction to Reinforcement Learning]] (MDPs, the Markov property, [[Introduction to Reinforcement Learning#Horizons and discounting|horizons]], value functions and Bellman equations) and [[Markov Chains]]. Conditional expectation and the tower rule ([[Joint Distributions#Conditional expectation|here]]) are used in the proofs.
>
> **TL;DR:** When the interaction lasts exactly $T$ steps, the best action depends on *how much time is left*, so optimal policies are **non-stationary**: one decision rule $\pi_t$ per stage. **Backward induction**, $V_T=r_T$ and $V_t(s)=\max_a\mathbb{E}\big[r_t(s,a,s')+V_{t+1}(s')\big]$, computes the optimal value and an optimal policy from every state at once, in $O(T|\mathcal{S}|^2|\mathcal{A}|)$ operations. Letting $T\to\infty$ with discounting turns the recursion into value iteration, the subject of [[Dynamic Programming]].

> [!note] Why it works: planning with a known model
> The lectures so far learned from experience. This lecture starts the second half of the series, which asks what those methods are approximating. Here we assume the transition model $p(s'\mid s,a)$ and the rewards are **known**, so the problem is *planning*: pure computation, no interaction. In practice the model is usually unknown, which is why the model-free lectures ([[Imitation Learning]], [[Policy Gradients]], …) never use it. A model can also be *learned* from data and then planned with; that is the topic of [[Model-Based Reinforcement Learning]].

> [!note] Notation in this lecture
> Series notation (see [[Introduction to Reinforcement Learning]]) with three finite-horizon additions. Rewards may depend on time, $r_t(s,a,s')$, and there is a terminal reward $r_T(s)$. Policies are sequences of **deterministic decision rules** $\pi=(\pi_0,\dots,\pi_{T-1})$, with $a_t=\pi_t(s_t)$: the subscript on $\pi_t$ is the stage. The subscript in $V_t$ is a *time index* (value with $T-t$ steps to go), not an iteration counter.

## Motivation

A chess player has two games left in a match, the score is level, and she knows two things about herself. If she plays **timid**, she never wins a game, but draws with probability $p_d=0.9$ (and loses otherwise). If she plays **bold**, she never draws, and wins with probability $p_w=0.45$. A tied match goes to sudden death. She is the weaker player in every single bold game ($p_w<\tfrac12$). Can she still be the favourite to win the *match*?

The answer is yes: with the right strategy she wins with probability $0.537$. The strategy is not a fixed plan. It is a **rule that reacts to the score** and to the **number of games left**, and computing it is exactly what this lecture is about. Two ideas make it work:

1. **Feedback.** The decision at each stage may depend on the state reached so far. We optimize over decision *rules*, not over action sequences.
2. **Backward induction.** The last decision is easy, because nothing follows it. Once we know how to act optimally with one stage left, the problem with two stages left reduces to a one-stage problem, and so on back to the start. This is Bellman's **principle of optimality**.[^bellman1957]

Many decision problems really do have a deadline: a match of $T$ games, a selling season of $T$ weeks, a robot that must reach a goal within $T$ steps, an episode of fixed length in a simulator. They are the "fixed horizon" case of the [[Introduction to Reinforcement Learning#Horizons and discounting|horizon taxonomy]] in the introduction. They are also the simplest setting in which the recursive structure of sequential decision making can be proved without any limits or fixed-point theorems.

---

## Finite-horizon MDPs

### Controlled Markov chains

Start with a state process $s_0,s_1,s_2,\dots$ that takes values in a finite set $\mathcal{S}$ and is driven by actions $a_0,a_1,\dots$, with $a_t\in\mathcal{A}(s_t)$, the **feasible** actions in the current state. It is a **controlled Markov chain** if

$$
\mathbb{P}(s_{t+1}=s'\mid s_t=s,a_t=a,s_{t-1},a_{t-1},\dots,s_0,a_0)=p(s'\mid s,a).
$$

Whatever history led to $s_t$, the next state depends only on the current state and action. An MDP is a controlled Markov chain together with rewards. For a problem with horizon $T$ these are

- a **stage reward** $r_t(s,a,s')$, collected at stage $t=0,\dots,T-1$ when action $a$ in state $s$ leads to $s'$;
- a **terminal reward** $r_T(s)$, collected once, in the final state $s_T$.

Both may depend on time. A discounted problem is the special case $r_t=\gamma^t r$. Nothing below changes if the transition probabilities depend on time too ($p_t(s'\mid s,a)$); we keep them stationary to lighten the notation. Decisions are made at stages $0,1,\dots,T-1$, and $T$ is the **terminal instant**.

### Policies are sequences of decision rules

A **decision rule** for stage $t$ is a map $\pi_t:\mathcal{S}\to\mathcal{A}$ with $\pi_t(s)\in\mathcal{A}(s)$ (it is **admissible**). A (deterministic Markov) **policy** for horizon $T$ is a sequence of decision rules, one per stage:

$$
\pi=(\pi_0,\pi_1,\dots,\pi_{T-1}),\qquad a_t=\pi_t(s_t).
$$

Running $\pi$ from $s_0$ produces a random trajectory, and its **value** is the expected total reward:

$$
V^\pi_0(s_0)=\mathbb{E}\Big[\,r_T(s_T)+\sum_{t=0}^{T-1}r_t\big(s_t,\pi_t(s_t),s_{t+1}\big)\;\Big|\;s_0\Big],
$$

where the expectation is over $s_1,\dots,s_T$, with $s_{t+1}\sim p(\cdot\mid s_t,\pi_t(s_t))$. More generally, $V^\pi_t(s)$ is the expected reward collected from stage $t$ onwards, starting in $s$ at time $t$ and following $(\pi_t,\dots,\pi_{T-1})$. The **optimal value** is $V^*_0(s_0)=\max_\pi V^\pi_0(s_0)$. There are finitely many policies, so the maximum exists.

**Why one decision rule per stage?** In the infinite-horizon discounted problem, the future looks the same from every time step, so a single stationary rule suffices (the existence theorem in [[Introduction to Reinforcement Learning]]). With a deadline it does not. The state alone no longer summarizes the situation: "one game left, one point ahead" calls for a different move than "ten games left, one point ahead" might. Equivalently, $(s,t)$ is the real state, and a policy that is stationary in $(s,t)$ is non-stationary in $s$. The [[#Example 2: a seasonal inventory|inventory example]] below shows two stages with identical demand, identical rewards and identical stock where the optimal orders differ, purely because one stage is the last.

> [!note] Feedback, not plans
> The objective depends on the decision *rules*, and the actions $a_t=\pi_t(s_t)$ are random variables. Fixing a sequence of actions $a_0,\dots,a_{T-1}$ in advance (an **open-loop** plan) is a different and generally worse problem, because it cannot react to the states the dynamics actually produce. In the chess match, the best fixed plan wins with probability $0.425$ against $0.537$ for the best feedback rule (Figure 2).
>
> We also restricted attention to deterministic Markov policies. That loses nothing: the proof of the DP theorem below shows that no randomized or history-dependent policy can do better.

---

## The principle of optimality

> [!tip] Intuition
> If you are driving the fastest route from Bangalore to Delhi and it passes through Nagpur, then its Nagpur–Delhi part is the fastest route from Nagpur to Delhi. If it weren't, you could swap in the faster tail and improve the whole trip.

**Principle of optimality.** Let $\pi^*=(\pi^*_0,\dots,\pi^*_{T-1})$ be optimal for the problem started at $s_0$. Suppose that under $\pi^*$ the state $s$ is reached at stage $t$ with positive probability. Then the **tail policy** $(\pi^*_t,\dots,\pi^*_{T-1})$ is optimal for the **tail subproblem** that starts in $s$ at time $t$ and maximizes

$$
\mathbb{E}\Big[\,r_T(s_T)+\sum_{k=t}^{T-1}r_k\big(s_k,\pi_k(s_k),s_{k+1}\big)\;\Big|\;s_t=s\Big].
$$

> [!proof]- Proof
> Suppose not. Then some tail policy $(\tilde\pi_t,\dots,\tilde\pi_{T-1})$ has **strictly larger** tail value from $s$, by some gap $\Delta>0$. Build a new policy that follows $\pi^*$, except that *if* the state at time $t$ is $s$, it switches to the $\tilde\pi$'s from then on. This policy has to remember whether $s_t=s$ happened, so it is history-dependent rather than Markov. That is harmless: the remark after the DP theorem below shows that the best Markov policy is as good as any history-dependent one (and that proof does not use the principle), so if the new policy beats $\pi^*$, so does some Markov policy. Trajectories that do not pass through $s$ at time $t$ collect exactly the same rewards as before. On the event $\{s_t=s\}$, the rewards before $t$ are unchanged and, by the Markov property, the expected reward from $t$ on rises by $\Delta$. So the total expected reward rises by
> $$
> \mathbb{P}_{\pi^*}(s_t=s)\cdot\Delta>0,
> $$
> contradicting the optimality of $\pi^*$. $\blacksquare$
>
> This is where the **positive-probability** assumption is used: if $s$ is never reached at time $t$, improving the tail from $s$ changes nothing, and an optimal $\pi^*$ may well act badly there.

The principle is a statement about optimal policies, not yet an algorithm. Read backwards, it says how to build one: solve the tail subproblems from the end, for **every** state, not just those an optimal policy visits. That is the DP algorithm, and it produces a policy that is optimal from every state at once, which the principle alone does not promise.

---

## The dynamic programming algorithm

> [!example] Algorithm: backward induction (finite-horizon DP)
> 1. **Initialize** $V_T(s)=r_T(s)$ for all $s$.
> 2. **For** $t=T-1,T-2,\dots,0$ and every state $s$:
>     - compute $Q_t(s,a)=\sum_{s'}p(s'\mid s,a)\big[r_t(s,a,s')+V_{t+1}(s')\big]$ for each $a\in\mathcal{A}(s)$;
>     - set $V_t(s)=\max_{a\in\mathcal{A}(s)}Q_t(s,a)$ and $\pi^*_t(s)\in\arg\max_{a\in\mathcal{A}(s)}Q_t(s,a)$.
> 3. **Return** $V_0$ and the policy $\pi^*=(\pi^*_0,\dots,\pi^*_{T-1})$.

> [!example]- Code: backward induction (NumPy)
> ```python
> # assumes: P[a, s, s'] = p(s'|s,a); R[t] of shape (A, S, S) = r_t(s, a, s'); terminal reward r_T (S,);
> # horizon T. If only some actions are feasible in s, set Q[a, s] = -np.inf for the others before the max.
> V = r_T.copy()                                     # V_T
> pi_star = np.zeros((T, len(r_T)), dtype=int)
> for t in reversed(range(T)):                       # t = T-1, ..., 0
>     Q = (P * (R[t] + V[None, None, :])).sum(-1)    # Q_t(a, s) = sum_s' p(s'|s,a) [r_t(s,a,s') + V_{t+1}(s')]
>     pi_star[t] = Q.argmax(0)                       # decision rule pi*_t(s)
>     V = Q.max(0)                                   # V_t(s)
> # now V = V_0, and pi_star[t] is the decision rule for stage t
> ```
> Each pass of the loop costs $O(|\mathcal{S}|^2|\mathcal{A}|)$, which is the operation count below. For the chess match, the state is the net score, the two actions are timid and bold, and every stage reward is 0.

In one line, the recursion is

$$
\begin{aligned}
V_T(s)&=r_T(s),\\
V_t(s)&=\max_{a\in\mathcal{A}(s)}\;\mathbb{E}_{s'\sim p(\cdot\mid s,a)}\big[r_t(s,a,s')+V_{t+1}(s')\big],
\end{aligned}
\tag{1}
$$

for $t=T-1,\dots,0$.

$V_t(s)$ is the **value-to-go**: the best expected reward obtainable in the $(T-t)$-stage problem that starts in $s$ at time $t$. (Control texts, which minimize cost, call the same object the *cost-to-go*.[^bertsekas2017])

**Theorem (finite-horizon DP).** For every $t$ and every $s$:

1. $V_t(s)=\max_\pi V^\pi_t(s)$. In particular, $V_0(s_0)=V^*_0(s_0)$ for every initial state.
2. The greedy policy $\pi^*=(\pi^*_0,\dots,\pi^*_{T-1})$ of the algorithm attains it: $V^{\pi^*}_t(s)=V_t(s)$, simultaneously for all $s$ and $t$. So $\pi^*$ is optimal from **every** initial state.

> [!proof]- Proof (backward induction)
> We prove, for $t=T,T-1,\dots,0$, the two-part claim
>
> **(a)** $V^\pi_t(s)\le V_t(s)$ for every policy $\pi$ and every $s$; **(b)** $V^{\pi^*}_t(s)=V_t(s)$ for every $s$.
>
> Together they give $V_t=\max_\pi V^\pi_t$ and the optimality of $\pi^*$.
>
> **Base case** $t=T$. No decisions remain, so $V^\pi_T(s)=r_T(s)=V_T(s)$ for every policy.
>
> **Inductive step.** Assume (a) and (b) at $t+1$. Fix any policy $\pi$ and a state $s$, and write $a=\pi_t(s)$. Split off the first reward and condition on $s_{t+1}$ (tower rule). By the Markov property, once $s_{t+1}=s'$ is known, the rewards from stage $t+1$ on depend only on $s'$ and on $\pi_{t+1},\dots,\pi_{T-1}$, so their conditional expectation is $V^\pi_{t+1}(s')$:
> $$
> V^\pi_t(s)=\sum_{s'}p(s'\mid s,a)\big[r_t(s,a,s')+V^\pi_{t+1}(s')\big].
> $$
> The weights $p(s'\mid s,a)$ are non-negative, so the induction hypothesis (a) gives
> $$
> \begin{aligned}
> V^\pi_t(s)&\le\sum_{s'}p(s'\mid s,a)\big[r_t(s,a,s')+V_{t+1}(s')\big]\\
> &\le\max_{a'\in\mathcal{A}(s)}\sum_{s'}p(s'\mid s,a')\big[r_t(s,a',s')+V_{t+1}(s')\big]=V_t(s).
> \end{aligned}
> $$
> That is (a) at $t$. For $\pi^*$, both inequalities are equalities: the first by (b) at $t+1$, the second because $\pi^*_t(s)$ attains the max. That is (b) at $t$. $\blacksquare$
>
> **Randomized and history-dependent policies.** If $\pi$ chooses $a_t$ at random, possibly depending on the whole history $h_t$, the same computation reads $V^\pi_t(h_t)=\sum_a\pi_t(a\mid h_t)\sum_{s'}p(s'\mid s_t,a)[r_t+V^\pi_{t+1}(h_{t+1})]$. Bounding $V^\pi_{t+1}(h_{t+1})\le V_{t+1}(s')$ by induction, and an average over $a$ by the max over $a$, gives $V^\pi_t(h_t)\le V_t(s_t)$ again. So the larger policy class cannot beat $V_0$.

> [!warning] Where the "max–expectation interchange" hides
> A common way to write this proof splits the max over the whole tail policy, $\pi_{t:}=(\pi_t,\pi_{t+1:})$:
> $$
> \begin{aligned}
> \max_{\pi_t,\,\pi_{t+1:}}\;&\mathbb{E}_{s'}\big[r_t+V^{\pi_{t+1:}}_{t+1}(s')\big]\\
> &=\max_{\pi_t}\;\mathbb{E}_{s'}\Big[r_t+\max_{\pi_{t+1:}}V^{\pi_{t+1:}}_{t+1}(s')\Big].
> \end{aligned}
> $$
> Moving the max **inside** the expectation is the crux. In general only "$\le$" holds: $\max_x\mathbb{E}[f(x,Y)]\le\mathbb{E}[\max_xf(x,Y)]$, since the right side may pick a different $x$ for each outcome $Y$. Equality holds here because a tail policy is a collection of *feedback* maps: it may act differently at each next state $s'$, and part (b) of the induction says that **one** tail policy, $(\pi^*_{t+1},\dots,\pi^*_{T-1})$, is optimal from every $s'$ simultaneously. For open-loop plans the interchange fails, and that gap is exactly the value of feedback (0.425 vs 0.537 in the chess match).[^bertsekas2017]

The last step of the proof also uses a small fact worth stating on its own: maximizing over decision rules $\pi_t$ an expression that depends on $\pi_t$ only through $\pi_t(s)$ is the same as maximizing over the single action $a\in\mathcal{A}(s)$. This is why a search over $|\mathcal{A}|^{|\mathcal{S}|}$ decision rules collapses to $|\mathcal{S}|$ separate searches over $|\mathcal{A}|$ actions.

### Computational cost

Each stage evaluates, for each state $s$ and action $a$, a sum over next states $s'$. That costs $O(|\mathcal{S}|^2|\mathcal{A}|)$ per stage and

$$
O\big(T\,|\mathcal{S}|^2\,|\mathcal{A}|\big)
$$

in total (less when transitions are sparse), with $O(T|\mathcal{S}|)$ memory to store the policy. Compare this with brute force: there are $\prod_t\prod_s|\mathcal{A}(s)|$, i.e. up to $|\mathcal{A}|^{T|\mathcal{S}|}$, deterministic Markov policies. In the inventory example below, DP needs $T|\mathcal{S}|^2|\mathcal{A}|=26{,}364$ elementary terms, while the number of policies is about $10^{118}$. DP is linear in the horizon, but still quadratic in the number of states, which is the **curse of dimensionality** discussed in [[Dynamic Programming#Asynchronous DP and the curse of dimensionality|Dynamic Programming]].

---

## Example 1: the chess match

**Model.** A match of $T$ games. Each game is worth 1 point for a win, $\tfrac12$ for a draw and 0 for a loss. The state is the **net score** $x$, the player's points minus the opponent's. A draw leaves $x$ unchanged, a win adds 1 and a loss subtracts 1. Before each game the player chooses

- **timid:** draw with probability $p_d$, lose with probability $1-p_d$ ($x\to x$ or $x-1$);
- **bold:** win with probability $p_w$, lose with probability $1-p_w$ ($x\to x+1$ or $x-1$).

We want to **maximize the probability of winning the match**. This is naturally a reward problem: there are no stage rewards, $r_t\equiv0$, and the terminal reward is the probability of winning from the final score,

$$
r_T(x)=\begin{cases}1,&x>0,\\ p_w,&x=0,\\ 0,&x<0.\end{cases}
$$

The value $p_w$ at a tie needs a word of justification. In sudden death, timid play can never win a game, so its match-win probability is 0. Bold play makes every game decisive, and the first decisive game is won with probability $p_w$. So the best sudden-death win probability is $p_w$.

**Recursion.** Equation (1) becomes

$$
\begin{aligned}
V_t(x)&=\max\{\text{timid}_t(x),\ \text{bold}_t(x)\},\\
\text{timid}_t(x)&=p_d\,V_{t+1}(x)+(1-p_d)\,V_{t+1}(x-1),\\
\text{bold}_t(x)&=p_w\,V_{t+1}(x+1)+(1-p_w)\,V_{t+1}(x-1).
\end{aligned}
$$

Subtracting $V_{t+1}(x-1)$ from both branches, bold is optimal at $(t,x)$ if and only if

$$
p_w\big[V_{t+1}(x+1)-V_{t+1}(x-1)\big]\;\ge\;p_d\big[V_{t+1}(x)-V_{t+1}(x-1)\big],
$$

with both actions optimal at equality. Bold gains a point with probability $p_w$, timid "saves" one with probability $p_d$, and the brackets measure how much each of those is worth with $T-t-1$ games left.

**The last game** ($t=T-1$). Plugging in $r_T$:

| net score $x$ | timid | bold | optimal | $V_{T-1}(x)$ |
|---|---|---|---|---|
| $x\ge2$ | $1$ | $1$ | either | $1$ |
| $x=1$ | $p_d+(1-p_d)p_w$ | $p_w+(1-p_w)p_w$ | timid if $p_d>p_w$ | $p_d+(1-p_d)p_w$ |
| $x=0$ | $p_dp_w$ | $p_w$ | bold | $p_w$ |
| $x=-1$ | $0$ | $p_w^2$ | bold | $p_w^2$ |
| $x\le-2$ | $0$ | $0$ | either | $0$ |

Only the row $x=1$ needs an assumption: timid minus bold equals $(p_d-p_w)(1-p_w)$, so timid is better exactly when $p_d>p_w$ (as in our numbers). At $x=0$, bold wins as long as $p_d<1$ and $p_w>0$.

**Two games left, tied** ($t=T-2$, $x=0$). Using the last-game values,

$$
\begin{aligned}
\text{timid: }&\;p_d\,V_{T-1}(0)+(1-p_d)\,V_{T-1}(-1)=p_dp_w+(1-p_d)p_w^2,\\
\text{bold: }&\;p_w\,V_{T-1}(1)+(1-p_w)\,V_{T-1}(-1)\\
&\;=p_w\big(p_d+(1-p_d)p_w\big)+(1-p_w)p_w^2.
\end{aligned}
$$

Bold minus timid is exactly $(1-p_w)p_w^2>0$, so **bold is optimal when tied with two games left**, and

$$
V_{T-2}(0)=p_w\big(p_d+(1-p_d)p_w\big)+(1-p_w)p_w^2.
$$

With $p_w=0.45$ and $p_d=0.9$ this is $0.537$. Figure 1 shows the whole computation for $T=2$.

![[Reinforcement Learning/images/finite-horizon-dp-chess-trellis.svg]]
*Figure 1: backward induction for the two-game match ($p_w=0.45$, $p_d=0.9$). Circles show $V_t(x)$; the right column is the terminal reward. Arrows show the optimal action's outcomes (blue timid, orange bold) with their probabilities. Values are filled in from right to left; the policy is then read from left to right: bold at 0–0, then timid if ahead and bold if behind.*

**The surprise.** The optimal policy plays timid when ahead and bold when level or behind. It wins the match with probability $0.537$ even though every bold game is lost more often than won. Figure 2(a) shows that this holds for all $p_w$ above about $0.422$ when $T=2$ (about $0.443$ when $T=10$). No fixed plan can do this: the best open-loop sequence for $T=2$ (always bold) wins with probability $0.425<p_w$. The edge comes entirely from *reacting to the score*, i.e. from feedback.

![[Reinforcement Learning/images/finite-horizon-dp-chess-winprob.svg]]
*Figure 2: (a) probability of winning the match from 0–0 as a function of $p_w$ ($p_d=0.9$), for the optimal feedback policy with $T=2$ and $T=10$ games and for the best fixed plan with $T=2$. Above the horizontal line the player is the favourite. (b) The optimal action for $T=10$, $p_w=0.45$, by net score and games remaining: T = timid, B = bold, · = both optimal (the match is already decided).*

Figure 2(b) also shows how time enters the policy. Here the rule "timid if ahead, bold otherwise" happens to work at every stage; time appears only in *which* states are already decided (where the choice does not matter). The next example has a policy that genuinely changes with time.

---

## Example 2: a seasonal inventory

A shop sells one product over a season of $T=12$ weeks.

- **State:** stock on hand $s\in\{0,\dots,12\}$ at the start of the week.
- **Action:** order $a\in\{0,\dots,12-s\}$ units, delivered immediately.
- **Demand:** $D_t\sim\text{Poisson}(\lambda_t)$, with mean $\lambda_t=2+3\sin\big(\pi t/11\big)$, rising from 2 to about 5 mid-season and falling back to 2. Unmet demand is lost.
- **Stage reward:** price 5 per unit sold, minus 2 per unit ordered, minus a holding cost of 0.5 per unit left at the end of the week. The next state is $s'=\max(s+a-D_t,0)$. The expected reward $r_t(s,a)$ depends on $t$ through the demand distribution.
- **Terminal reward:** leftover stock is sold off at 1 per unit, $r_T(s)=s$, which is less than its purchase cost.

Backward induction (1) solves this in a fraction of a second. The optimal expected profit from an empty shop is $V_0(0)=113.62$; simulating the computed policy 40,000 times gives $113.74\pm0.19$ (95% interval), as it should.

![[Reinforcement Learning/images/finite-horizon-dp-inventory.svg]]
*Figure 3: (a) the optimal order $\pi^*_t(s)$ for each week and stock level. (b) In every week the policy "orders up to" a level $S_t$, i.e. $\pi^*_t(s)=\max(S_t-s,0)$. The level follows the demand, but not symmetrically: weeks 0 and 11 have the same demand ($\lambda=2$) and the same stage rewards, yet $S_0=4$ while $S_{11}=2$.*

The policy has a clean structure: in every week it orders up to a target level, $S_t=(4,5,6,6,7,7,7,7,6,6,4,2)$ (checked at every $t$ and $s$). Such **order-up-to** (base-stock) policies are classical in inventory theory.[^arrow1951] More important for us is the asymmetry at the ends of the season. Week 0 and week 11 look identical to a one-step analysis, but in week 11 any unsold unit only fetches the salvage price, so ordering is less attractive. That is non-stationarity caused purely by the horizon.

---

## From finite to infinite horizon

Take a stationary discounted problem, $r_t(s,a,s')=\gamma^t r(s,a,s')$, with no terminal reward. Factor out $\gamma^t$, i.e. measure value in "time-$t$ units": $\tilde V_t=V_t/\gamma^t$. The recursion (1) becomes

$$
\tilde V_t(s)=\max_a\sum_{s'}p(s'\mid s,a)\big[r(s,a,s')+\gamma\,\tilde V_{t+1}(s')\big],\qquad\tilde V_T=0,
$$

which is one application of the Bellman optimality operator $\mathcal{T}^*$ per stage. So the $T$-stage value is

$$
\tilde V_0=(\mathcal{T}^*)^T\,0,
$$

and **backward induction over a horizon $T$ is value iteration run for $T$ sweeps**. Because $\mathcal{T}^*$ is a $\gamma$-contraction ([[Dynamic Programming#The foundation: Bellman operators are contractions|proved in Dynamic Programming]]),

$$
\big\lVert\tilde V_0-V^*\big\rVert_\infty\le\gamma^T\lVert V^*\rVert_\infty\;\xrightarrow[T\to\infty]{}\;0.
$$

The non-stationarity also fades. In a finite MDP, every action that is not optimal for the infinite-horizon problem is worse than the best one by some margin $\delta>0$ in $Q^*$. Once $2\gamma\,\lVert\tilde V_{t+1}-V^*\rVert_\infty<\delta$, the greedy action at stage $t$ is optimal for the infinite-horizon problem too. So far from the deadline, the finite-horizon policy coincides with an optimal *stationary* policy; only the last few stages differ.

![[Reinforcement Learning/images/finite-horizon-dp-to-infinite.svg]]
*Figure 4: a stationary version of the inventory problem ($\lambda=3.5$ every week, $\gamma=0.95$, no salvage). (a) The first-stage value approaches $V^*$ geometrically, under the bound $\gamma^T\lVert V^*\rVert_\infty$. (b) With $T=30$, the optimal order-up-to level equals the stationary optimal level (5) at every stage except the last, where it drops to 4.*

Two other limits lead to the next lectures:

- **Discounted, $T\to\infty$:** stationary policies, the Bellman optimality *equation* $V^*=\mathcal{T}^*V^*$, and the algorithms that solve it (policy evaluation, policy iteration, value iteration) are the subject of [[Dynamic Programming]].
- **Undiscounted, random horizon:** when episodes end at a random time (reaching a goal, losing a game), the horizon is finite but not fixed. Without discounting, finiteness of the value must come from termination itself. That is the setting of [[Stochastic Shortest Path Problems]].

---

## Key takeaways

- With a fixed horizon $T$, the right object is a **non-stationary** policy $(\pi_0,\dots,\pi_{T-1})$: the best action depends on the time remaining as well as the state.
- **Principle of optimality:** the tail of an optimal policy is optimal for the tail subproblem from any state it reaches with positive probability.
- **Backward induction** $V_T=r_T$, $V_t(s)=\max_a\mathbb{E}[r_t(s,a,s')+V_{t+1}(s')]$ gives the optimal value **and** a policy that is optimal from every state, in $O(T|\mathcal{S}|^2|\mathcal{A}|)$ time.
- The proof hinges on feedback: one tail policy is optimal from every next state, which justifies swapping max and expectation. Open-loop plans lack this and can be much worse (chess: 0.425 vs 0.537).
- Discounted backward induction is value iteration; as $T\to\infty$ it converges to $V^*$ and the policy becomes stationary away from the deadline.

## Where this shows up next

- [[Dynamic Programming]] (next): the infinite-horizon discounted problem, Bellman operators as contractions, policy iteration and value iteration.
- [[Stochastic Shortest Path Problems]]: undiscounted problems that end at a random time.
- [[Model-Based Reinforcement Learning]]: what to do when $p(s'\mid s,a)$ must be learned. Model-predictive control re-solves a short finite-horizon problem like the ones here at every step.
- [[Policy Gradients]] and [[Imitation Learning]] also work with a finite horizon $t=0,\dots,T-1$, but optimize over a parametric (and usually stationary) policy class rather than solving the recursion.

<!-- References: footnote definitions below render as the numbered "References" list at the bottom of the page -->

[^bellman1957]: Bellman, R. (1957). *Dynamic Programming*. Princeton University Press (reprinted 2010). [doi:10.1515/9781400835386](https://doi.org/10.1515/9781400835386)
[^bertsekas2017]: Bertsekas, D. P. (2017). *Dynamic Programming and Optimal Control*, Vol. I (4th ed.), Chapter 1 (the finite-horizon DP algorithm and the chess-match example). Athena Scientific. [book website](http://www.athenasc.com/dpbook.html)
[^arrow1951]: Arrow, K. J., Harris, T., & Marschak, J. (1951). *Optimal Inventory Policy*. Econometrica, 19(3), 250. [doi:10.2307/1906813](https://doi.org/10.2307/1906813)
