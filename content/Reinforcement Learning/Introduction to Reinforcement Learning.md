---
title: Introduction to Reinforcement Learning
tags:
  - reinforcement-learning
  - mdp
  - bellman-equations
  - value-functions
---

> **Prerequisites:** basic probability (conditional expectation, the tower rule) and Markov chains. Some familiarity with supervised learning ([[Linear Models]]) helps for the motivation. [[Multi-Armed Bandits]] introduces exploration in the simplest, single-state setting.
>
> **TL;DR:** Reinforcement learning is learning to make *sequences* of decisions when each decision changes what you see next. We model the world as a **Markov decision process** (MDP). A **policy** acts in it, and we judge the policy by its expected (discounted) sum of rewards. The **value functions** $V^\pi$ and $Q^\pi$ satisfy recursive **Bellman equations**. For finite MDPs, these give policy evaluation in closed form and characterize the optimal policy.

## What is reinforcement learning?

Reinforcement learning (RL) is the part of machine learning that formalizes **sequential decision making under uncertainty**. Compare it with the other two classical paradigms:

- **Supervised learning:** given data $\mathbf{X}$ and labels $\mathbf{y}$, learn the mapping $\mathbf{X}\to\mathbf{y}$ by minimizing the empirical risk (e.g. [[Linear Models|linear and logistic regression]]).
- **Unsupervised learning:** given only $\mathbf{X}$, find structure in it, such as clusters, low-dimensional representations or a density model.

Neither deals with feedback over time. In the real world the current decision affects what happens next, but a supervised model makes each prediction as a single, isolated step.

### Why is RL hard?

Supervised learning problems usually come with assumptions that make them "easy":

- the data points are independent;
- the model's outputs don't influence its next inputs;
- ground-truth labels are provided at training time.

Decision-making problems usually break all three:

- current actions influence future observations, and so future actions;
- the goal is to maximize some utility (reward) accumulated over time, so credit for a good outcome must be assigned to the decisions that caused it;
- optimal actions are not provided, and must be discovered by trial and error. This is the exploration problem, studied in isolation in [[Multi-Armed Bandits]].

Real-world deployments of ML often have the same feedback issues. For example, a traffic-prediction system's suggestions change the routes people take, which in turn changes the traffic it is trying to predict.

---

## The agent–environment loop

![[Reinforcement Learning/images/intro-rl-agent-environment.svg]]
*Figure 1: at each step the agent observes the state $s_t$ and picks an action $a_t\sim\pi(\cdot\mid s_t)$. The environment then returns a reward $r_t$ and the next state $s_{t+1}$.*

The interaction proceeds in discrete time steps $t=0,1,2,\dots$:

1. The environment samples an initial state $s_0\sim\rho_0$ and gives it to the agent.
2. The agent's **policy** chooses an action $a_t\sim\pi(\cdot\mid s_t)$.
3. The environment executes $a_t$. It returns a reward $r_t$, the next state $s_{t+1}\sim p(\cdot\mid s_t,a_t)$, and a flag $d_t$ saying whether $s_{t+1}$ is terminal.
4. If $d_t$ is true, the episode ends and the environment resets to a new $s_0\sim\rho_0$. Otherwise, go to step 2 with $t\leftarrow t+1$.

So one step of experience is the tuple $(s_t,a_t,r_t,s_{t+1},d_t)$. A **trajectory** (or episode, or rollout) of $T$ steps is

$$
\tau=(s_0,a_0,r_0,s_1,a_1,r_1,\dots,s_{T-1},a_{T-1},r_{T-1},s_T).
$$

> [!note] Reward index convention
> We index the reward with the action that caused it: $r_t$ is the reward for taking $a_t$ in $s_t$. Sutton & Barto[^suttonbarto] call the same quantity $R_{t+1}$. Only the label differs, but mixing the two conventions is a common source of off-by-one errors.

---

## Markov Decision Processes

An RL problem is formalized as a **Markov decision process** (MDP), the tuple

$$
\mathcal{M}=(\mathcal{S},\mathcal{A},p,r,\rho_0,\gamma),
$$

where

- $\mathcal{S}$ is the **state space** and $\mathcal{A}$ the **action space**. We write $\mathcal{A}(s)$ when the available actions depend on the state.
- $p(s'\mid s,a)$ is the **transition** probability (density) of landing in $s'$ after taking $a$ in $s$. These are the dynamics of the environment.
- $r(s,a)$ is the **(expected) reward** for taking action $a$ in state $s$.
- $\rho_0$ is the **initial-state distribution**.
- $\gamma\in[0,1]$ is the **discount factor**. We need $\gamma<1$ when the interaction can go on forever; see [[#Horizons and discounting]].

**The Markov property.** The next state and reward depend on the past *only through the current state and action*:

$$
\mathbb{P}(s_{t+1},r_t\mid s_0,a_0,\dots,s_t,a_t)=\mathbb{P}(s_{t+1},r_t\mid s_t,a_t).
$$

So the state is a sufficient summary of the history, and that is what makes the recursive equations below possible.

### Dynamics and the different reward functions

For a **finite MDP** ($\mathcal{S}$, $\mathcal{A}$ and the set of reward values $\mathcal{R}$ are all finite), the most complete description of the environment is the joint **dynamics function**

$$
p(s',r\mid s,a)\triangleq\mathbb{P}(s_{t+1}=s',\,r_t=r\mid s_t=s,\,a_t=a),
$$

a function $p:\mathcal{S}\times\mathcal{R}\times\mathcal{S}\times\mathcal{A}\to[0,1]$.

Everything else can be derived from it.

- **State-transition probabilities:** $\;p(s'\mid s,a)=\sum_{r\in\mathcal{R}}p(s',r\mid s,a)$.
- **Expected reward (two arguments):**
$$
r(s,a)=\mathbb{E}\big[r_t\mid s_t=s,a_t=a\big]=\sum_{r\in\mathcal{R}}r\sum_{s'\in\mathcal{S}}p(s',r\mid s,a).
$$
- **Expected reward (three arguments):** condition on the next state as well and use Bayes' rule:
$$
r(s,a,s')=\mathbb{E}\big[r_t\mid s_t=s,a_t=a,s_{t+1}=s'\big]=\sum_{r\in\mathcal{R}}r\,\frac{p(s',r\mid s,a)}{p(s'\mid s,a)}.
$$
- **Expected reward under a policy (one argument):** averaging over the policy's actions gives
$$
r_\pi(s)=\sum_{a\in\mathcal{A}(s)}\pi(a\mid s)\,r(s,a).
$$
This one depends on the policy. A reward that is a function of the state alone, $r(s)$, is a separate *modelling choice*, not something derived from $r(s,a)$.

Since $r(s,a)=\mathbb{E}_{s'\sim p(\cdot\mid s,a)}[r(s,a,s')]$, and the objective below only involves *expected* rewards, using $r(s,a)$ instead of $r(s,a,s')$ does not change which policies are optimal. We use $r(s,a)$ from here on.

> [!note] Cost instead of reward
> Control theory often specifies an MDP with a **cost** $c(s,a)$ to be *minimized*. Setting $r=-c$ makes the two formulations equivalent. [[Imitation Learning]] uses a cost.

### State and action spaces

- **Discrete (finite):** for example a grid world, board games, or choosing among $K$ items. Values and policies can be stored as tables, and expectations are sums. This lecture's Bellman-equation section and [[Dynamic Programming]] work in this **tabular** setting.
- **Continuous:** for example joint angles and velocities of a robot ($\mathcal{S}\subseteq\mathbb{R}^{d_s}$), or torques ($\mathcal{A}\subseteq\mathbb{R}^{d_a}$). Policies and values must be *function approximators* (e.g. neural networks), and sums become integrals. [[Imitation Learning]] and [[Policy Gradients]] are written for this general case.

Most ideas carry over between the two by swapping sums for integrals.

### Partially observed MDPs

A **partially observed MDP** (POMDP) adds an observation space $\mathcal{O}$ and an **emission** distribution $p(o\mid s)$. The agent never sees the state $s_t$, only an observation $o_t\sim p(\cdot\mid s_t)$, so its policy must act on observations: $\pi(a_t\mid o_t)$, or more generally on the observation history.[^kaelbling1998] For example, $s$ could be the positions, velocities and torques of a quadruped robot's joints, while $o$ is an image of the robot from an external camera. A single image does not reveal the velocities, so the observation alone is not Markov.

---

## Policies, trajectories and the RL objective

A **policy** maps states to probabilities of selecting each possible action: $\pi(a\mid s)$, i.e. $\pi:\mathcal{S}\to\Delta(\mathcal{A})$. A **deterministic** policy is the special case that puts all its mass on one action, and we then write $a=\pi(s)$. A learnable policy with parameters $\theta$ is written $\pi_\theta$.

Running $\pi_\theta$ in the MDP induces a distribution over trajectories:

$$
p_\theta(\tau)=\rho_0(s_0)\prod_{t=0}^{T-1}\pi_\theta(a_t\mid s_t)\,p(s_{t+1}\mid s_t,a_t).
$$

The **goal** of reinforcement learning is to find parameters $\theta^*$ such that $\pi_{\theta^*}=\pi^*$, the policy with the largest expected (discounted) return:

$$
\theta^*=\arg\max_\theta\;J(\theta),\qquad J(\theta)=\mathbb{E}_{\tau\sim p_\theta(\tau)}\left[\sum_{t=0}^{T-1}\gamma^t\,r(s_t,a_t)\right].
$$

### Horizons and discounting

Depending on the MDP, the length of a trajectory falls into one of three categories:

1. **Fixed horizon:** every episode lasts exactly $T$ steps (e.g. a 100-step control task). The return is $\sum_{t=0}^{T-1}\gamma^t r_t$, and $\gamma=1$ is fine.
2. **Finite (episodic, indefinite) horizon:** every episode ends *eventually* in a terminal state, but the length $T$ is random (e.g. a game of chess). Model the terminal state as **absorbing**: it transitions to itself with reward $0$. Then every episode can be written as an infinite sequence, and $\gamma=1$ is fine as long as termination is guaranteed.
3. **Infinite horizon (continuing):** the interaction never ends (e.g. a thermostat or a data-centre cooling system). The undiscounted sum can diverge, so we need $\gamma\in[0,1)$. Then, for bounded rewards $|r|\le r_{\max}$, the return is bounded: $\big|\sum_{k\ge0}\gamma^k r_{t+k}\big|\le r_{\max}/(1-\gamma)$.

Besides keeping sums finite, $\gamma$ expresses a preference for earlier reward. It also acts as a soft horizon of about $1/(1-\gamma)$ steps: $\gamma=0.99$ "looks" about 100 steps ahead.

> [!note] Default in these notes
> The general derivations ([[Imitation Learning]], [[Policy Gradients]]) use a finite horizon with $t=0,\dots,T-1$. Where it is undiscounted, $\gamma=1$. The Bellman equations below use the infinite-horizon discounted form ($\gamma<1$), which also covers episodic tasks through absorbing terminal states.

---

## Value functions

The **return** from time $t$ is the discounted sum of the rewards that follow, the *rewards-to-go*:

$$
G_t=\sum_{k=0}^{\infty}\gamma^k r_{t+k}=r_t+\gamma\,G_{t+1}.
$$

The recursion $G_t=r_t+\gamma G_{t+1}$ is the seed of everything below.

The **state-value function** of a policy $\pi$ is the return we *expect* to collect by starting in $s$ and following $\pi$ from then on:

$$
V^\pi(s)\triangleq\mathbb{E}_\pi\big[G_t\mid s_t=s\big]\qquad\forall s\in\mathcal{S}.
$$

The **action-value function** ($Q$-function) additionally fixes the first action:

$$
Q^\pi(s,a)\triangleq\mathbb{E}_\pi\big[G_t\mid s_t=s,\,a_t=a\big]\qquad\forall s\in\mathcal{S},\ a\in\mathcal{A}(s).
$$

The two are related by averaging over the policy's first action. The difference between them is the **advantage**:

$$
\begin{aligned}
V^\pi(s)&=\sum_{a\in\mathcal{A}(s)}\pi(a\mid s)\,\mathbb{E}_\pi\big[G_t\mid s_t=s,a_t=a\big]=\mathbb{E}_{a\sim\pi(\cdot\mid s)}\big[Q^\pi(s,a)\big],\\
A^\pi(s,a)&=Q^\pi(s,a)-V^\pi(s).
\end{aligned}
$$

The advantage measures how much better action $a$ is than the policy's average behaviour in $s$. It is exactly the weight used by variance-reduced [[Policy Gradients#Reward-to-go (causality)|policy gradients]].

Finally, the RL objective is the value of the start state, averaged over $\rho_0$:

$$
J(\pi)=\mathbb{E}_{s_0\sim\rho_0}\big[V^\pi(s_0)\big].
$$

---

## Bellman expectation equations

**$Q^\pi$ in terms of $V^\pi$.** Split off the first reward and use $G_t=r_t+\gamma G_{t+1}$:

$$
\begin{aligned}
Q^\pi(s,a)&=\mathbb{E}_\pi\big[r_t+\gamma G_{t+1}\mid s_t=s,a_t=a\big]\\
&=\mathbb{E}\big[r_t\mid s_t=s,a_t=a\big]+\gamma\,\mathbb{E}_\pi\Big[\,\mathbb{E}_\pi\big[G_{t+1}\mid s_{t+1}\big]\,\Big|\,s_t=s,a_t=a\Big]\\
&=r(s,a)+\gamma\,\mathbb{E}_{s'\sim p(\cdot\mid s,a)}\big[V^\pi(s')\big]\\
&=r(s,a)+\gamma\sum_{s'\in\mathcal{S}}p(s'\mid s,a)\,V^\pi(s').
\end{aligned}
$$

The second line uses the tower rule and the Markov property. Given $s_{t+1}$, the future return does not depend on $(s_t,a_t)$, so the inner expectation is just $V^\pi(s_{t+1})$.

**The Bellman equation for $V^\pi$.** Average over $a\sim\pi(\cdot\mid s)$:

$$
V^\pi(s)=\underbrace{\sum_{a}\pi(a\mid s)\,r(s,a)}_{\text{(i) expected immediate reward}}+\underbrace{\gamma\sum_{a}\pi(a\mid s)\sum_{s'}p(s'\mid s,a)\,V^\pi(s')}_{\text{(ii) discounted expected value of the next state}}.
$$

Equivalently, in terms of the joint dynamics, $V^\pi(s)=\sum_a\pi(a\mid s)\sum_{s',r}p(s',r\mid s,a)\big[r+\gamma V^\pi(s')\big]$.

**The Bellman equation for $Q^\pi$** (in $Q$ alone, as used by SARSA and actor-critic methods):

$$
Q^\pi(s,a)=r(s,a)+\gamma\sum_{s'}p(s'\mid s,a)\sum_{a'}\pi(a'\mid s')\,Q^\pi(s',a').
$$

![[Reinforcement Learning/images/intro-rl-backup-diagrams.svg]]
*Figure 2: backup diagrams. Open circles are states and filled dots are state–action pairs. The value at the root is computed from its successors: an expectation over the policy's actions and the environment's transitions (left two), or a **max** over actions (right two; see the optimality section below).*

### Matrix form and closed-form policy evaluation

Fixing $\pi$ turns the MDP into a **Markov chain** over states, with transition matrix and reward vector

$$
P^\pi(s,s')\triangleq\sum_{a}\pi(a\mid s)\,p(s'\mid s,a),\qquad r_\pi(s)=\sum_a\pi(a\mid s)\,r(s,a).
$$

$P^\pi$ is **row-stochastic**: $\sum_{s'}P^\pi(s,s')=\sum_a\pi(a\mid s)\sum_{s'}p(s'\mid s,a)=\sum_a\pi(a\mid s)=1$. Term (ii) above is $\gamma\sum_{s'}P^\pi(s,s')V^\pi(s')$. Stack $V^\pi$ and $r_\pi$ into vectors of length $n=|\mathcal{S}|$, and the Bellman equation becomes a linear system:

$$
\begin{aligned}
&\boxed{\;V^\pi=r_\pi+\gamma P^\pi V^\pi\;}\\
\Longrightarrow\quad&(I-\gamma P^\pi)\,V^\pi=r_\pi\\
\Longrightarrow\quad&V^\pi=(I-\gamma P^\pi)^{-1}r_\pi.
\end{aligned}
$$

**The inverse always exists for $\gamma<1$.** Every eigenvalue $\lambda$ of a row-stochastic matrix has $|\lambda|\le1$, so the eigenvalues of $\gamma P^\pi$ lie strictly inside the unit circle, and the Neumann series converges:

$$
(I-\gamma P^\pi)^{-1}=\sum_{k=0}^{\infty}\gamma^k\,(P^\pi)^k.
$$

This also reads naturally: $(P^\pi)^k r_\pi$ is the expected reward $k$ steps in the future. (For $\gamma=1$ the inverse can fail to exist, which is another reason to insist on $\gamma<1$ or absorbing terminal states.)

So **given a policy, we can compute its value function exactly** for a finite MDP. Solving the system costs $O(n^3)$. [[Dynamic Programming#Policy evaluation (prediction)|Dynamic Programming]] (next lecture) evaluates it iteratively instead, which scales much better.

> [!example] Worked example: evaluating a fixed policy
> A fixed policy $\pi$ in a three-state MDP induces the chain below. $C$ is an absorbing terminal state with reward $0$; $\gamma=0.9$.
> $$
> P^\pi=\begin{pmatrix}0.6&0.4&0\\0.3&0.4&0.3\\0&0&1\end{pmatrix},\qquad r_\pi=\begin{pmatrix}1\\2\\0\end{pmatrix},
> $$
>
> $$
> V^\pi=(I-0.9\,P^\pi)^{-1}r_\pi=\begin{pmatrix}200/29\\175/29\\0\end{pmatrix}\approx\begin{pmatrix}6.90\\6.03\\0\end{pmatrix}.
> $$
> $B$ pays more per step than $A$ ($2$ vs $1$), yet $V^\pi(B)<V^\pi(A)$: from $B$ there is a 30% chance per step of falling into the zero-reward terminal state. Values account for *where you go next*, not just what you get now. Check: $r_\pi(B)+\gamma\big[0.3\,V^\pi(A)+0.4\,V^\pi(B)\big]=2+0.9\,(2.069+2.414)=6.03$. ✓

![[Reinforcement Learning/images/intro-rl-worked-example.svg]]
*Figure 3: the Markov chain $P^\pi$ from the worked example, with each state's reward $r_\pi$ and value $V^\pi$.*

---

## Optimality: $V^*$, $Q^*$ and the Bellman optimality equations

Evaluating a given policy is half the story. We want the **best** policy. Define the optimal value functions

$$
V^*(s)\triangleq\sup_\pi V^\pi(s),\qquad Q^*(s,a)\triangleq\sup_\pi Q^\pi(s,a)\qquad\forall s,a.
$$

> [!note] Existence of an optimal policy
> A priori, a different policy might be best from each state. A fundamental theorem for finite MDPs with $\gamma<1$ says otherwise. There is a **single deterministic, stationary** policy $\pi^*$ that is optimal from *every* state simultaneously, so $V^*=V^{\pi^*}$ and $Q^*=Q^{\pi^*}$, and the $\sup$ is a $\max$ (Puterman, Thm. 6.2.7).[^puterman1994] [[Dynamic Programming#Policy iteration|Policy iteration]] (next lecture) gives a constructive proof.

Because $\pi^*$ is optimal everywhere, acting optimally means taking the best first action and then continuing optimally:

$$
\begin{aligned}
V^*(s)&=\max_{a\in\mathcal{A}(s)}Q^*(s,a),\\
Q^*(s,a)&=\mathbb{E}\big[r_t+\gamma\,V^*(s_{t+1})\mid s_t=s,a_t=a\big]=r(s,a)+\gamma\sum_{s'}p(s'\mid s,a)\,V^*(s').
\end{aligned}
$$

Combining the two gives the **Bellman optimality equations**:

$$
\begin{aligned}
&\boxed{\;V^*(s)=\max_{a}\sum_{s',r}p(s',r\mid s,a)\big[r+\gamma V^*(s')\big]\;}\\[6pt]
&\boxed{\;Q^*(s,a)=\sum_{s',r}p(s',r\mid s,a)\Big[r+\gamma\max_{a'}Q^*(s',a')\Big]\;}
\end{aligned}
$$

Unlike the expectation equations, these are **nonlinear**, because of the $\max$. There is no closed form, and they are solved iteratively (value iteration, policy iteration) or from samples (Q-learning).[^watkins1992]

**Extracting the optimal policy.** Once $Q^*$ is known, acting greedily is optimal:

$$
\pi^*(s)=\arg\max_{a}Q^*(s,a)=\arg\max_a\Big[r(s,a)+\gamma\sum_{s'}p(s'\mid s,a)\,V^*(s')\Big].
$$

This is why so many RL algorithms aim to learn $Q^*$: the policy comes for free, with no model needed, if we have $Q^*$ itself.

> [!tip] Bellman operators
> It is convenient to write the right-hand sides as operators on value vectors: $(\mathcal{T}^\pi V)(s)=r_\pi(s)+\gamma\sum_{s'}P^\pi(s,s')V(s')$ and $(\mathcal{T}^*V)(s)=\max_a\big[r(s,a)+\gamma\sum_{s'}p(s'\mid s,a)V(s')\big]$. Then the Bellman equations say $V^\pi=\mathcal{T}^\pi V^\pi$ and $V^*=\mathcal{T}^*V^*$: the value functions are **fixed points**. Both operators are $\gamma$-contractions in the max-norm. That single fact, [[Dynamic Programming#The foundation: Bellman operators are contractions|proved in the next lecture]], gives the uniqueness of $V^\pi$ and $V^*$ and the convergence of policy evaluation, value iteration and (in the stochastic-approximation sense) Q-learning.

---

## Some additional topics

### Model of the environment

In machine learning, the function approximator that maps data to labels is usually called the "model". In RL, **model** instead means an approximation of the *environment*, $\hat p(s'\mid s,a)$ and $\hat r(s,a)$, which can be learned from the agent's own interaction data. Recall what one environment step produces: a next state, a reward and a done flag. A learned model needs three approximators with parameters $\phi$, trained by regression and classification on observed transitions $(s_t,a_t,r_t,s_{t+1},d_t)$:

1. **Next state:** $\hat{s}_{t+1}=s_t+\hat{\Delta}_t$, where $\hat{\Delta}_t=f^{s}_\phi(s_t,a_t)$. Predicting the *change* in state is usually easier than predicting the next state outright.
2. **Reward:** $\hat{r}_t=f^{r}_\phi(s_t,a_t)$.
3. **Termination:** $\hat{d}_t=\sigma\big(f^{d}_\phi(s_t,a_t)\big)$, where $\sigma(x)=\frac{1}{1+e^{-x}}$ is the sigmoid.

Methods that learn and use such a model (to plan, or to generate synthetic experience) are called **model-based**. Methods that learn $\pi$, $V$ or $Q$ directly from experience are **model-free**. Dynamic programming is the extreme case, where the model is given exactly.

### Anatomy of an RL algorithm

![[Reinforcement Learning/images/intro-rl-anatomy.svg]]
*Figure 4: almost every RL algorithm cycles through three stages, following the CS285 framing.[^cs285] Algorithms differ in how expensive each stage is: in policy gradients the "estimate" stage is just summing rewards, while in model-based RL it means fitting a dynamics model.*

1. **Generate samples:** run the current policy in the environment.
2. **Fit a model / estimate the return:** e.g. sum the rewards (Monte Carlo), fit $V^\pi$ or $Q^\pi$, or fit $\hat p$.
3. **Improve the policy:** e.g. a gradient step on $J(\theta)$, or acting greedily with respect to $Q$.

### On-policy vs Off-policy Algorithm

**On-policy** algorithms learn only from data generated by the *current* policy.

- Experience is collected with the current policy $\pi_{\theta_k}$. After the update, $\pi_{\theta_{k+1}}$ is the new policy.
- Because that experience came from $\pi_{\theta_k}$, it is discarded, and new trajectories are collected with $\pi_{\theta_{k+1}}$ for the next update. An example is [[Policy Gradients#REINFORCE|REINFORCE]].
- This is **sample-inefficient**, because old trajectories are never reused.
- In practice the *wall-clock* time can still be small, if many simulations (or robots) run in parallel.

**Off-policy** algorithms can learn from data generated by *any* policy: older versions of the agent, other agents, or humans.

- The update for $\pi_{\theta_k}$ is not restricted to experience collected by $\pi_{\theta_k}$. Old transitions are kept in a **replay buffer**, as in Q-learning, or reweighted with importance sampling, as in [[Policy Gradients#Off-policy Policy Gradients|off-policy policy gradients]].
- They are typically much more **sample-efficient**.
- Replaying older data also counteracts *forgetting* of skills learned earlier, because the update keeps seeing states the current policy no longer visits.
- Since the data need not come from the current policy, some of it can be collected by a more exploratory behaviour policy (for example, a random action with probability $\epsilon$). This builds the **exploration–exploitation** trade-off directly into the algorithm.

## Key takeaways

- An **MDP** $(\mathcal{S},\mathcal{A},p,r,\rho_0,\gamma)$ formalizes sequential decision making. The **Markov property** says the current state summarizes the past.
- A **policy** $\pi(a\mid s)$ induces a distribution over trajectories. RL maximizes the expected discounted return $J(\pi)=\mathbb{E}_{s_0}[V^\pi(s_0)]$. We need $\gamma<1$ for continuing tasks.
- $V^\pi$ and $Q^\pi$ satisfy the linear **Bellman expectation equations**. For finite MDPs, $V^\pi=(I-\gamma P^\pi)^{-1}r_\pi$, and the inverse exists whenever $\gamma<1$.
- $V^*$ and $Q^*$ satisfy the nonlinear **Bellman optimality equations**. Acting greedily with respect to $Q^*$ is optimal.
- RL algorithms cycle through *sample → estimate → improve*. They can be **on-policy** or **off-policy**, and **model-free** or **model-based**.

## Where this shows up next

- [[Dynamic Programming]] (next lecture): solving the Bellman equations when the model is known, with policy evaluation, policy iteration and value iteration, plus the contraction argument behind all three.
- [[Imitation Learning]]: learning a policy from expert demonstrations, and why distribution shift makes it hard.
- [[Policy Gradients]]: optimizing $J(\theta)$ directly by gradient ascent.

<!-- References: footnote definitions below render as the numbered "References" list at the bottom of the page -->

[^suttonbarto]: Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.), Chapters 3–4. [book website](http://incompleteideas.net/book/the-book-2nd.html)
[^kaelbling1998]: Kaelbling, L. P., Littman, M. L., & Cassandra, A. R. (1998). *Planning and Acting in Partially Observable Stochastic Domains*. Artificial Intelligence, 101(1–2), 99–134. [doi:10.1016/S0004-3702(98)00023-X](https://doi.org/10.1016/S0004-3702(98)00023-X)
[^puterman1994]: Puterman, M. L. (1994). *Markov Decision Processes: Discrete Stochastic Dynamic Programming*. Wiley. [doi:10.1002/9780470316887](https://doi.org/10.1002/9780470316887)
[^watkins1992]: Watkins, C. J. C. H., & Dayan, P. (1992). *Q-learning*. Machine Learning, 8, 279–292. [doi:10.1007/BF00992698](https://doi.org/10.1007/BF00992698)
[^cs285]: Levine, S. *CS 285: Deep Reinforcement Learning*, UC Berkeley, Lecture 4 (Introduction to RL). [course page](https://rail.eecs.berkeley.edu/deeprlcourse/)
