---
title: Actor-Critic Algorithms
tags:
  - reinforcement-learning
  - policy-gradients
  - actor-critic
  - advantage-estimation
date: 2026-11-12
draft: false
prev: "[[Policy Gradients]]"
next: "[[Monte Carlo and Temporal-Difference Learning]]"
---

> **Prerequisites:** [[Policy Gradients]] (REINFORCE, baselines, reward-to-go), [[Introduction to Reinforcement Learning]] (returns, $V^\pi$, $Q^\pi$, $A^\pi$, the Bellman expectation equation). How to learn the critic *well* is the subject of the next lecture, [[Monte Carlo and Temporal-Difference Learning]], and its theory with features is in [[Linear Function Approximation]].
>
> **TL;DR:** REINFORCE weights each action by a noisy single-sample return. An **actor-critic** method also learns a **critic**, an estimate of $V^\pi$ or $Q^\pi$, and weights each action by an estimated **advantage** instead. Bootstrapped estimates such as the TD error $r_t+\gamma V(s_{t+1})-V(s_t)$ cut the variance of the gradient sharply but are biased when the critic is wrong. **GAE($\lambda$)** interpolates between the two extremes. A **Q-function critic** lets us reuse old data from a replay buffer, which gives the off-policy family DDPG, TD3 and SAC. Convergence proofs run the critic on a **faster timescale** than the actor.
>
> **Code:** [Actor-Critic](https://github.com/parag1604/Actor-Critic): batch, online and A2C/GAE actor-critic, a Gaussian actor for continuous control, and Soft Actor-Critic.

**Notation used here.** The actor is $\pi_\theta$ and the critic has its own parameters $\phi$ ($\hat V_\phi$ or $\hat Q_\phi$). The TD error is $\delta_t$ and an advantage estimate is $\hat A_t$. In the convergence section the actor's step size is $\alpha_t$ and the critic's is $\beta_t$. A deterministic policy is $\mu_\theta(s)$, as in the series notation. Two local symbols: the entropy temperature is $\tau$, and the Polyak (soft target-update) rate is $\kappa$.

## Motivation

[[Policy Gradients]] ended with three variance-reduction tricks: reward-to-go, baselines, and the observation that the best state-dependent baseline is about $V^\pi(s_t)$. Combining them gives a weight $\hat Q_t-b(s_t)$ that estimates the **advantage** $A^\pi(s_t,a_t)$, "how much better was this action than what the policy usually does here?". The baseline still had to come from somewhere, and the reward-to-go $\hat Q_t$ was still a single noisy rollout.

Actor-critic methods take the last step. They **learn** a value function alongside the policy and use it both as the baseline and as a replacement for the noisy future. The policy is the **actor**: it decides. The value function is the **critic**: it judges how good the actor's decisions were. The idea is old (Barto, Sutton and Anderson used an "adaptive critic element" to balance a pole in 1983)[^barto1983] and is the backbone of most modern deep RL algorithms. The presentation follows the structure of Levine's CS 285 lecture on actor-critic methods,[^cs285ac] with derivations, proofs and experiments of our own.

In the [[Introduction to Reinforcement Learning#Anatomy of an RL algorithm|anatomy of an RL algorithm]], REINFORCE spends almost nothing on the green "fit a model / estimate returns" box: it just sums rewards. Actor-critic methods put real work there, by fitting a value function.

---

## From the policy gradient to a critic

### The policy gradient in terms of values

Start from the policy gradient with reward-to-go. We use the discounted objective $J(\theta)=\mathbb{E}_{\tau\sim p_\theta}\big[\sum_{t\ge0}\gamma^tr(s_t,a_t)\big]$ and write $p_\theta(s_t)$ for the distribution of the state at time $t$ under $\pi_\theta$. In expectation over everything that happens *after* $(s_t,a_t)$, the reward-to-go is $Q^{\pi_\theta}(s_t,a_t)$, so by the tower rule

$$
\nabla_\theta J(\theta)=\sum_{t\ge0}\gamma^t\,\mathbb{E}_{s_t\sim p_\theta(s_t),\,a_t\sim\pi_\theta(\cdot\mid s_t)}\Big[\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,Q^{\pi_\theta}(s_t,a_t)\Big].
\tag{1}
$$

This is the **policy gradient theorem**.[^sutton1999] (The factor $\gamma^t$ is discussed in [Discount factors](#discount-factors) below.) Its message: if we knew $Q^\pi$ exactly, a single sampled action per visited state would give an unbiased gradient term, with no need to wait for the rest of the trajectory.

Any function of the state alone can be subtracted, because the score has mean zero under the policy:

$$
\mathbb{E}_{a\sim\pi_\theta(\cdot\mid s)}\big[\nabla_\theta\log\pi_\theta(a\mid s)\,b(s)\big]=b(s)\,\nabla_\theta\sum_a\pi_\theta(a\mid s)=b(s)\,\nabla_\theta1=0.
$$

The natural choice $b(s)=V^\pi(s)$ turns $Q^\pi$ into the **advantage** $A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)$:

$$
\nabla_\theta J(\theta)=\sum_{t\ge0}\gamma^t\,\mathbb{E}\Big[\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,A^{\pi_\theta}(s_t,a_t)\Big].
\tag{2}
$$

> [!tip] Intuition: why the advantage is the right weight
> $Q^\pi(s,a)$ mixes two things: how good the *state* is, and how good the *action* is relative to the others. Only the second should change the policy. In a state where everything is worth about $100$, every action gets pushed up with weight about $100$, and only the small differences between actions carry information. The advantage removes the state's share, so actions better than the policy's average are made more likely, and worse ones less likely.

### Which function should the critic learn?

We could fit $Q^\pi(s,a)$, $V^\pi(s)$ or $A^\pi(s,a)$. The state value is the simplest: it depends on the state only, so it is easier to learn from the same amount of data. And it is enough, because one step of the Bellman equation recovers $Q^\pi$ from $V^\pi$:

$$
Q^\pi(s_t,a_t)=r(s_t,a_t)+\gamma\,\mathbb{E}_{s_{t+1}\sim p(\cdot\mid s_t,a_t)}\big[V^\pi(s_{t+1})\big].
$$

Replace the expectation over $s_{t+1}$ by the one next state we actually saw. This gives the **TD error** as an advantage estimate:

$$
\begin{aligned}
\delta_t&=r_t+\gamma\,V^\pi(s_{t+1})-V^\pi(s_t),\\
\mathbb{E}\big[\delta_t\mid s_t,a_t\big]&=Q^\pi(s_t,a_t)-V^\pi(s_t)=A^\pi(s_t,a_t).
\end{aligned}
\tag{3}
$$

$\delta_t$ is an **unbiased** estimate of the advantage *if* $V^\pi$ is exact. It depends on a single random transition, while the Monte Carlo advantage $G_t-V^\pi(s_t)$ depends on every reward and transition until the end of the episode. That is where the variance goes.

### How much variance does a critic remove?

Take a small problem where everything can be computed exactly: a slippery corridor of 7 states with a noisy reward on every step (standard deviation $0.5$), an exit at each end ($+1$ on the right, $-1$ on the left), $\gamma=0.95$, and a fixed softmax policy that goes right with probability $0.6$. Episodes last 15.5 steps on average. From 20,000 episodes we form one-episode estimates of the *same* gradient (1) in five ways, weighting $\gamma^t\nabla_\theta\log\pi_\theta(a_t\mid s_t)$ by different estimates of how good the action was.

![[Reinforcement Learning/images/actor-critic-gradient-variance.svg]]
*Figure 1: total variance (summed over the 14 policy parameters) of single-episode estimates of the policy gradient in a 7-state corridor. All five estimators have the same mean, the exact gradient, up to Monte Carlo error. Weighting each action by the full return gives variance 19.9; reward-to-go 7.13; reward-to-go minus $V^\pi(s_t)$ 6.54; the one-step TD error with the exact $V^\pi$ 1.05; and the true advantage $A^\pi(s_t,a_t)$ 0.05.*

Two things stand out.

- The baseline $V^\pi(s_t)$ helps only a little here (7.13 → 6.54), because the state values are small compared with the reward noise. The big win comes from **bootstrapping**: the TD error replaces the noisy sum of all future rewards by $\gamma V^\pi(s_{t+1})$, and the variance drops by a factor of about 6.
- The true advantage would be about 20 times better still. The remaining noise in $\delta_t$ is the noise of one reward and one transition. A perfect critic is the limit of what any critic-based estimator can achieve.

The catch: these numbers use the *exact* $V^\pi$. A learned critic is never exact, and then $\delta_t$ is **biased**. Most of this lecture is about managing that bias.

---

## Fitting the critic: policy evaluation

Fitting $\hat V_\phi\approx V^{\pi_\theta}$ is **policy evaluation**, the "estimate returns" step of the anatomy. With function approximation (a neural network, or [[Linear Function Approximation|linear features]]) it becomes a supervised regression problem: collect states $s_{i,t}$ with targets $y_{i,t}$ and minimize

$$
\mathcal{L}(\phi)=\frac12\sum_{i,t}\big\lVert\hat V_\phi(s_{i,t})-y_{i,t}\big\rVert^2 .
$$

The only question is which target to use.

**Monte Carlo targets.** Use the observed discounted reward-to-go, $y_{i,t}=\sum_{t'\ge t}\gamma^{t'-t}r(s_{i,t'},a_{i,t'})$. It is an unbiased sample of $V^\pi(s_{i,t})$, but a single-sample one, so it is noisy. Fitting one function to many such samples helps: nearby states share parameters, so the network effectively averages the returns of *similar* states, even though no state is ever visited twice in a continuous space.

**Bootstrapped targets.** Use the critic's own prediction for the rest of the episode:

$$
y_{i,t}=r(s_{i,t},a_{i,t})+\gamma\,\hat V_\phi(s_{i,t+1}).
$$

This target has much lower variance, because it depends on one reward and one transition. But it is biased whenever $\hat V_\phi$ is wrong, and it chases itself: the label moves as $\phi$ moves. In practice the target is computed with the current $\phi$ and then treated as a **fixed label** (no gradient flows through it), which makes this a *semi-gradient* method. Why that still converges, and to what, is the subject of [[Monte Carlo and Temporal-Difference Learning]] and [[Linear Function Approximation]].

> [!warning] Terminal states and time limits
> If $s_{t+1}$ is **terminal**, its value is $0$ and the target is just $r_t$. If the episode was merely **cut off** by a time limit (CartPole stops at 500 steps), $s_{t+1}$ is not terminal and the target should still bootstrap with $\gamma\hat V_\phi(s_{t+1})$. Treating time-outs as terminal teaches the critic that the world ends at step 500, which it does not.

**Time-invariant data.** The targets do not care *when* a transition was collected. A batch of trajectories is equivalent to a bag of transitions $(s_i,a_i,r_i,s_i')$, each with its own target $y_i=r_i+\gamma\hat V_\phi(s_i')$. This change of viewpoint is what later makes replay buffers possible.

---

## Discount factors

### Why the critic needs $\gamma$

In a task that never ends, or one whose episodes are long, the undiscounted value can be infinite or very large, and a critic fitted to it is useless. A discount $\gamma<1$ keeps the values bounded by $r_{\max}/(1-\gamma)$ ([[Introduction to Reinforcement Learning#Horizons and discounting|horizons and discounting]]). One useful way to read $\gamma$: it changes the MDP. At every step the agent "dies" with probability $1-\gamma$, moving to an absorbing state with reward $0$, and otherwise continues with the original dynamics. The undiscounted value of this modified MDP is the discounted value of the original. Values $\gamma\in[0.95,0.999]$ are typical in practice, with $\gamma=1$ reserved for tasks that terminate with certainty.

### Two ways to put $\gamma$ into the policy gradient

With discounting, there are two natural Monte Carlo policy gradients:

$$
\begin{aligned}
\text{option 1:}\quad&\hat g_1=\sum_{t}\nabla_\theta\log\pi_\theta(a_t\mid s_t)\sum_{t'\ge t}\gamma^{t'-t}r_{t'},\\
\text{option 2:}\quad&\hat g_2=\sum_{t}\gamma^t\,\nabla_\theta\log\pi_\theta(a_t\mid s_t)\sum_{t'\ge t}\gamma^{t'-t}r_{t'}.
\end{aligned}
$$

Option 2 is the **exact** gradient of the discounted objective. Start from the full-return form $\big(\sum_t\nabla_\theta\log\pi_\theta(a_t\mid s_t)\big)\big(\sum_{t'}\gamma^{t'}r_{t'}\big)$ and drop the cross terms with $t'<t$ (causality). Then factor $\gamma^{t'}=\gamma^t\gamma^{t'-t}$. In the "death" reading, $\gamma^t$ is the probability of still being alive at time $t$: a decision at a late time matters less because we are less likely to get there. Equation (1) is the expected value of $\hat g_2$.

Option 1 drops the $\gamma^t$ and treats a decision at $t=500$ as just as important as one at $t=0$. It is **not** the gradient of the discounted objective. In general it is not the gradient of any objective,[^nota2020] and natural actor-critic algorithms built on it are biased.[^thomas2014] Yet option 1 is what almost every implementation uses, including the CartPole code below. The reason: what we usually *want* is the long-run (undiscounted or average) reward, and $\gamma$ is only a variance-reduction device that shortens the effective horizon of the reward-to-go to about $1/(1-\gamma)$ steps. Under that reading, every state the policy visits should count equally, and option 1 is a deliberately biased, low-variance approximation of the gradient we care about.

> [!note] Which one is used where
> The tabular experiments in this lecture (Figures 1, 2 and 5) use option 2, so that every estimator targets the exact gradient and bias can be measured. The CartPole experiment (Figure 3) uses option 1, like most deep RL code. Sutton and Barto's one-step actor-critic keeps the factor, as a running product $I\leftarrow\gamma I$.[^suttonbarto]

---

## Actor-critic algorithms

### Batch actor-critic

Putting the pieces together gives the simplest actor-critic algorithm. It is REINFORCE with the "estimate returns" box replaced by a fitted critic.

> [!example] Algorithm: batch actor-critic
> **Repeat** for a fixed number of iterations:
> 1. Collect a batch of transitions $\{(s_i,a_i,r_i,s_i')\}$ by running $\pi_\theta$ in the environment.
> 2. Fit $\hat V_\phi$ to the batch: a few gradient steps on $\frac12\sum_i\big(\hat V_\phi(s_i)-y_i\big)^2$, with Monte Carlo or bootstrapped targets $y_i$.
> 3. Evaluate advantages $\hat A_i=r_i+\gamma\hat V_\phi(s_i')-\hat V_\phi(s_i)$, with $\hat V_\phi(s_i')=0$ if $s_i'$ is terminal.
> 4. Estimate the gradient $\nabla_\theta J(\theta)\approx\sum_i\nabla_\theta\log\pi_\theta(a_i\mid s_i)\,\hat A_i$.
> 5. Update the actor: $\theta\leftarrow\theta+\alpha\,\nabla_\theta J(\theta)$.

With Monte Carlo targets in step 2 and $\hat A_i=G_i-\hat V_\phi(s_i)$ in step 3, this is "REINFORCE with a learned baseline", which is still unbiased. With bootstrapped advantages it is a true actor-critic: lower variance, biased critic.

### Online actor-critic

Bootstrapping needs only one transition, so the critic and the actor can both be updated after **every** step, without waiting for episodes to end.

> [!example] Algorithm: online actor-critic
> Observe $s$. **Repeat** at every step:
> 1. Sample $a\sim\pi_\theta(\cdot\mid s)$, take it, observe $r$ and $s'$.
> 2. Critic: $\delta=r+\gamma\hat V_\phi(s')-\hat V_\phi(s)$ (with $\hat V_\phi(s')=0$ if $s'$ is terminal), then $\phi\leftarrow\phi+\beta\,\delta\,\nabla_\phi\hat V_\phi(s)$, a TD(0) step.
> 3. Actor: $\theta\leftarrow\theta+\alpha\,\delta\,\nabla_\theta\log\pi_\theta(a\mid s)$, using the same $\delta$ as the advantage estimate.
> 4. $s\leftarrow s'$ (or reset if the episode ended).

The online algorithm is elegant and cheap, but a single transition gives a very noisy update. Consecutive transitions are also strongly correlated, which is bad for stochastic gradient descent on neural networks. The next two sections fix this.

### Architecture: one network or two?

The actor maps $s$ to a distribution over actions, the critic maps $s$ to a number, and both need to understand the state.

- **Two separate networks** are simple and stable. Each has its own loss, step size and optimizer, and neither can damage the other. Their drawback is that features are learned twice.
- **One shared trunk with two heads** (a policy head and a value head) learns the features once. That helps most when the input is high-dimensional, such as images.

The price of sharing is a combined loss,

$$
\begin{aligned}
\mathcal{L}(\theta)={}&-\sum_i\log\pi_\theta(a_i\mid s_i)\,\operatorname{sg}\big[\hat A_i\big]+c_V\sum_i\big(\hat V_\theta(s_i)-y_i\big)^2\\
&-c_H\sum_i\mathcal{H}\big(\pi_\theta(\cdot\mid s_i)\big),
\end{aligned}
$$

where $\operatorname{sg}[\cdot]$ is stop-gradient. The weights $c_V$ and $c_H$ must balance gradients of very different scales, and the two heads can interfere through the trunk. The optional entropy bonus $c_H\mathcal{H}$ discourages premature collapse to a deterministic policy. The advantage must be detached in the actor term: we want to move the *policy* towards good actions, not move the critic to make the advantage look larger.

### Parallel workers: A2C and A3C

To get batches of decorrelated data without giving up on-policy learning, run **many copies of the environment in parallel**.

- **Synchronous (A2C).** $N$ workers step their environments in lockstep for $n$ steps each, with the same current policy. The resulting batch of $N\times n$ transitions spans many unrelated states, and it gives one gradient step for the actor and one for the critic. Advantages use the $n$-step or GAE estimators of the next section, bootstrapping from $\hat V_\phi$ at the cut.
- **Asynchronous (A3C).** Each worker keeps a local copy of the parameters, collects a short rollout, computes a gradient and applies it to shared parameters *without waiting* for the others. Mnih et al. (2016) introduced it with CPU threads, $n$-step returns and an entropy bonus.[^mnih2016] Gradients are computed with slightly stale parameters, so the updates are slightly off-policy.

The synchronous version is simpler, reproducible from its seeds, and batches well on a GPU. In both, parallelism plays the role that replay buffers play in off-policy methods: it breaks the correlation between consecutive samples.

---

## $n$-step returns and generalized advantage estimation

### The bias–variance dial

We have two advantage estimators at opposite extremes:

$$
\hat A_t^{\mathrm{MC}}=\sum_{k\ge0}\gamma^kr_{t+k}-\hat V(s_t),\qquad\hat A_t^{(1)}=r_t+\gamma\hat V(s_{t+1})-\hat V(s_t)=\delta_t .
$$

The first is unbiased for any $\hat V$, because $\hat V$ only acts as a baseline. Its variance is high, since it sums every future reward. The second has low variance but is biased whenever $\hat V\ne V^\pi$. In between are the **$n$-step** estimators, which use $n$ real rewards and then bootstrap:

$$
\hat A_t^{(n)}=\sum_{k=0}^{n-1}\gamma^kr_{t+k}+\gamma^n\hat V(s_{t+n})-\hat V(s_t).
$$

Larger $n$ means less reliance on the critic (less bias) and more reliance on random future rewards (more variance). This is the same spectrum as the $n$-step returns of [[Monte Carlo and Temporal-Difference Learning]]; here we use it for the actor's advantage instead of the critic's target. A3C used $n=5$. A key identity: **the $n$-step advantage is a discounted sum of TD errors.** With $\delta_k=r_k+\gamma\hat V(s_{k+1})-\hat V(s_k)$, the sum telescopes:

$$
\sum_{k=0}^{n-1}\gamma^k\delta_{t+k}=\sum_{k=0}^{n-1}\gamma^kr_{t+k}+\sum_{k=0}^{n-1}\Big(\gamma^{k+1}\hat V(s_{t+k+1})-\gamma^k\hat V(s_{t+k})\Big)=\hat A_t^{(n)} .
\tag{4}
$$

### Generalized advantage estimation

Instead of choosing a single $n$, **average all of them** with geometrically decaying weights $(1-\lambda)\lambda^{n-1}$, $\lambda\in[0,1)$, which sum to one. This is the **generalized advantage estimator** of Schulman et al. (2016):[^schulman2016gae]

$$
\hat A_t^{\mathrm{GAE}(\gamma,\lambda)}=(1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1}\hat A_t^{(n)}=\sum_{l=0}^{\infty}(\gamma\lambda)^l\,\delta_{t+l}.
\tag{5}
$$

> [!proof]- Derivation of (5)
> Substitute (4) for each $\hat A_t^{(n)}$ and exchange the order of summation. The term $\gamma^k\delta_{t+k}$ appears in every $\hat A_t^{(n)}$ with $n>k$:
> $$
> \begin{aligned}
> (1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1}\sum_{k=0}^{n-1}\gamma^k\delta_{t+k}
> &=\sum_{k=0}^{\infty}\gamma^k\delta_{t+k}\,(1-\lambda)\sum_{n=k+1}^{\infty}\lambda^{n-1}\\
> &=\sum_{k=0}^{\infty}\gamma^k\delta_{t+k}\,(1-\lambda)\frac{\lambda^k}{1-\lambda}=\sum_{k=0}^{\infty}(\gamma\lambda)^k\delta_{t+k}.
> \end{aligned}
> $$
> The exchange is justified for $\lambda<1$ when the rewards and $\hat V$ are bounded, since the double series then converges absolutely.
>
> **Episodes that end.** If the episode terminates at time $T$, set $r_k=0$ and $\hat V(s_k)=0$ for $k\ge T$, so $\delta_k=0$ for $k\ge T$. Then $\hat A_t^{(n)}=\hat A_t^{\mathrm{MC}}$ for every $n\ge T-t$, the sum in (5) stops at $l=T-1-t$, and the formula is unchanged.

Three facts make GAE convenient.

1. **The endpoints are the familiar estimators.** $\lambda=0$ gives $\delta_t$, the one-step TD advantage. $\lambda=1$ gives $\sum_l\gamma^l\delta_{t+l}=\sum_l\gamma^lr_{t+l}-\hat V(s_t)$, the Monte Carlo advantage with $\hat V$ as a baseline (take the limit $\lambda\to1$ in (5); for a terminating episode the sum is finite).
2. **It is cheap.** (5) satisfies the backward recursion $\hat A_t=\delta_t+\gamma\lambda\,\hat A_{t+1}$, so a single backward pass over a rollout computes every advantage. It is the same computation as the $\lambda$-return of TD($\lambda$) in [[Monte Carlo and Temporal-Difference Learning]], with weights $\gamma\lambda$.
3. **With an exact critic it is unbiased for every $\lambda$** (shown next).

If $\hat V=V^\pi$, then $\mathbb{E}[\delta_t\mid s_t,a_t]=A^\pi(s_t,a_t)$ by (3), and for $l\ge1$,

$$
\mathbb{E}\big[\delta_{t+l}\mid s_t,a_t\big]=\mathbb{E}\Big[\,\mathbb{E}_{a\sim\pi}\big[A^\pi(s_{t+l},a)\big]\,\Big|\,s_t,a_t\Big]=0,
$$

because the advantage averages to zero under the policy that defines it. So $\mathbb{E}[\hat A_t^{\mathrm{GAE}}\mid s_t,a_t]=A^\pi(s_t,a_t)$. With an imperfect critic, only $\lambda=1$ stays unbiased.

The critic can be trained on the matching target $y_t=\hat A_t^{\mathrm{GAE}}+\hat V(s_t)$, which is the $\lambda$-return. In this setting $\lambda$ is a bias–variance knob, while $\gamma$ also changes *what* is being estimated (the effective horizon). Schulman et al. found $\lambda\in[0.9,0.99]$ to work well with $\gamma\in[0.99,0.995]$ in their continuous-control experiments.

### Measuring the trade-off

Back in the corridor of Figure 1, give the critic an error: $\hat V(s)=V^\pi(s)+e(s)$, where $e(s)\sim\mathcal{N}(0,0.2^2)$ is drawn once per state and then fixed (results are averaged over 5 such critics). For each $\lambda$ we estimate the squared bias of the GAE policy gradient and its variance per episode.

![[Reinforcement Learning/images/actor-critic-gae-tradeoff.svg]]
*Figure 2: GAE($\lambda$) policy-gradient estimates in the 7-state corridor with a critic that is off by $\mathcal{N}(0,0.2^2)$ per state. Left: the squared bias falls from 0.0375 at $\lambda=0$ to zero (within Monte Carlo error) at $\lambda=1$, while the one-episode variance rises from 1.32 to 6.69. Right: the mean squared error of a gradient averaged over a batch of $B$ episodes, $\text{bias}^2+\text{variance}/B$. The best $\lambda$ (circled) is 0.3 for $B=10$, 0.6 for $B=50$ and 0.8 for $B=250$.*

The right panel shows the trade-off concretely. Variance averages out over a batch and bias does not. So the more data per gradient step, the closer to $\lambda=1$ (less bias) we can afford to go. With small batches, leaning on the critic pays off. Neither extreme is optimal for any of the three batch sizes.

### Putting it together on CartPole

How do these choices play out with neural networks? We trained five methods on Gymnasium's CartPole-v1 (reward $+1$ per step, at most 500 steps) with identical small networks (two hidden layers of 32 tanh units, separate actor and critic), the same Adam step sizes, $\gamma=0.99$, option 1 discounting, no advantage normalization and no entropy bonus, for 150,000 environment steps and 10 seeds each:

- **REINFORCE:** reward-to-go, no baseline.
- **Learned MC baseline:** reward-to-go minus a critic fitted to Monte Carlo returns.
- **Batch actor-critic:** one-step TD advantage, critic fitted to bootstrapped targets.
- **Batch actor-critic with GAE(0.95):** critic fitted to $\lambda$-returns.

These four use batches of at least 1,000 steps of complete episodes, with 20 critic gradient steps per batch. The fifth method is

- **A2C:** 8 synchronous environments, 16-step rollouts, GAE(0.95), bootstrapping at the cut, one gradient step per 128 transitions.

![[Reinforcement Learning/images/actor-critic-cartpole.svg]]
*Figure 3: learning curves on CartPole-v1 (median and interquartile range over 10 seeds). Averaged over the last 20,000 steps, GAE(0.95) reaches a median return of 498 and all 10 runs exceed 450. The learned MC baseline reaches 485 (9 of 10 runs above 450), and REINFORCE without a baseline reaches 424 (2 of 10). The one-step TD actor-critic is the worst: median 170, 2 of 10 runs above 450, and three runs near 11–25. A2C learns fastest at first (median first reaching 400 after 42,500 steps, against 67,500 for GAE) but is noisier: median 476, 6 of 10 runs above 450.*

The pure one-step actor-critic is the clearest lesson. With $\gamma=0.99$ the values in CartPole are in the tens to about 100, and a bootstrapped critic propagates information only one step per fitting round. Early in training it is badly wrong, its TD errors are badly biased, and in several runs the actor follows that bias into a bad policy it never recovers from. GAE with $\lambda=0.95$ leans on real rewards for about $1/(1-0.95\cdot0.99)\approx17$ steps and on the critic only beyond that. It was both fast and reliable. A2C's many small updates helped early and hurt later at this step size. These are small experiments with untuned hyperparameters shared across methods, not a benchmark.

---

## Off-policy actor-critic

Everything so far is **on-policy**: each batch is used for one or a few gradient steps and then thrown away, because the advantage estimates are only valid for the policy that collected them. In a simulator this is wasteful; on a real robot it is often unaffordable. Can an actor-critic method learn from a **replay buffer** $\mathcal{R}$ of transitions collected by older versions of the policy?

### Replay buffers, and why the naive version is wrong

The naive idea: take the online algorithm, store every transition $(s_i,a_i,r_i,s_i')$ in $\mathcal{R}$, and do updates on mini-batches sampled from $\mathcal{R}$. This is broken in two ways.

1. **The critic learns the wrong value.** The target $y_i=r_i+\gamma\hat V_\phi(s_i')$ uses the action $a_i$ taken by some *old* policy. Averaged over the buffer, $\hat V_\phi$ learns the value of a mixture of past policies, not $V^{\pi_\theta}$.
2. **The actor's gradient uses the wrong actions.** $\nabla_\theta\log\pi_\theta(a_i\mid s_i)\hat A_i$ is an unbiased policy-gradient term only if $a_i\sim\pi_\theta(\cdot\mid s_i)$. Buffer actions come from other policies. Correcting with importance weights brings back the variance problems of [[Policy Gradients#Off-policy Policy Gradients|off-policy policy gradients]].

### The fix: a Q-function critic, queried with fresh actions

Both problems come from the critic depending on *which action was taken*. A $Q$-function takes the action as an **input**, so it can be evaluated at any action we like, in particular at actions sampled from the *current* policy. Its Bellman equation, $Q^\pi(s,a)=r(s,a)+\gamma\,\mathbb{E}_{s'}\mathbb{E}_{a'\sim\pi(\cdot\mid s')}[Q^\pi(s',a')]$, holds for **every** pair $(s,a)$, whoever chose $a$. A stored transition is therefore a valid sample for any policy's $Q$, provided the next action $a'$ is drawn fresh from $\pi_\theta$.

> [!example] Algorithm: off-policy actor-critic with a Q-function critic
> **Repeat** at every step:
> 1. Take $a\sim\pi_\theta(\cdot\mid s)$, observe $(s,a,r,s')$, and add it to the buffer $\mathcal{R}$.
> 2. Sample a mini-batch $\{(s_i,a_i,r_i,s_i')\}_{i=1}^B$ from $\mathcal{R}$.
> 3. For each $i$, sample a **fresh** next action $a_i'\sim\pi_\theta(\cdot\mid s_i')$ and set $y_i=r_i+\gamma\hat Q_\phi(s_i',a_i')$.
> 4. Critic: a gradient step on $\frac12\sum_i\big(\hat Q_\phi(s_i,a_i)-y_i\big)^2$.
> 5. Actor: sample $a_i^\pi\sim\pi_\theta(\cdot\mid s_i)$ (again fresh, not the buffer action), and step along $\frac1B\sum_i\nabla_\theta\log\pi_\theta(a_i^\pi\mid s_i)\,\hat Q_\phi(s_i,a_i^\pi)$.

Step 5 has no baseline, so it is noisier than an advantage, but we can afford that: sampling more actions per state costs no environment interaction. One problem remains and is simply accepted. The states $s_i$ come from the buffer's state distribution, not from $p_\theta(s)$. Actions can be resampled; states cannot, without running the environment. The actor is therefore optimized on a broader set of states than it will actually visit. That is usually harmless and arguably helpful. (Degris, White and Sutton derived an off-policy actor-critic that handles the action mismatch with importance weights instead.[^degris2012])

### The reparameterized actor update

With a differentiable critic, there is a better actor update than the score function in step 5. Write the sampled action as a differentiable function of parameter-free noise, $a=f_\theta(s,\xi)$, for example $f_\theta(s,\xi)=m_\theta(s)+\sigma_\theta(s)\odot\xi$ with $\xi\sim\mathcal{N}(0,I)$, and backpropagate through it:

$$
\nabla_\theta\,\mathbb{E}_{a\sim\pi_\theta(\cdot\mid s)}\big[\hat Q_\phi(s,a)\big]=\mathbb{E}_{\xi}\Big[\nabla_a\hat Q_\phi(s,a)\big|_{a=f_\theta(s,\xi)}\,\nabla_\theta f_\theta(s,\xi)\Big].
$$

This is the pathwise estimator from [[Policy Gradients#Straight-Through Gumbel-Softmax (STGS) for PG Algorithms|the Gumbel-softmax section of Policy Gradients]]. It uses the critic's *gradient* in the action, not just its value, and its variance is typically much lower. For discrete actions, the straight-through Gumbel-softmax trick provides an approximate $f_\theta$.

### Deterministic policy gradients

Push this to the limit: let the actor be **deterministic**, $a=\mu_\theta(s)$, with continuous actions. There is then no distribution over actions to differentiate, and the gradient must come entirely from the critic.

> [!note] Theorem (deterministic policy gradient; Silver et al., 2014)
> Let $\mathcal{A}\subseteq\mathbb{R}^{d_a}$ be continuous, and let $p(s'\mid s,a)$, $r(s,a)$, $\mu_\theta(s)$ and their derivatives be continuous and bounded, so that differentiation and integration may be exchanged. For $J(\theta)=\mathbb{E}_{s_0\sim\rho_0}\big[V^{\mu_\theta}(s_0)\big]$ with $\gamma<1$,
> $$
> \nabla_\theta J(\theta)=\sum_{t\ge0}\gamma^t\,\mathbb{E}_{s_t\sim p_\theta(s_t)}\Big[\nabla_\theta\mu_\theta(s_t)\,\nabla_aQ^{\mu_\theta}(s_t,a)\big|_{a=\mu_\theta(s_t)}\Big],
> $$
> where $\nabla_\theta\mu_\theta(s)$ is the Jacobian of the action with respect to $\theta$ (transposed, $\dim\theta\times d_a$).

> [!proof]- Proof sketch
> Write $\mu=\mu_\theta$. For a deterministic policy $V^\mu(s)=Q^\mu(s,\mu(s))$ and $Q^\mu(s,a)=r(s,a)+\gamma\int p(s'\mid s,a)V^\mu(s')\,ds'$. The parameter $\theta$ enters $V^\mu(s)$ in two places: through the action $\mu(s)$ chosen now, and through the policy followed afterwards, i.e. through $V^\mu(s')$. By the chain rule,
> $$
> \nabla_\theta V^\mu(s)=\nabla_\theta\mu(s)\,\nabla_aQ^\mu(s,a)\big|_{a=\mu(s)}+\gamma\int p\big(s'\mid s,\mu(s)\big)\,\nabla_\theta V^\mu(s')\,ds'.
> $$
> The second term has the same form one step later. Unrolling it $k$ times and letting $k\to\infty$ (the remainder is bounded by $\gamma^k$ times a constant) gives $\nabla_\theta V^\mu(s_0)=\sum_t\gamma^t\,\mathbb{E}\big[\nabla_\theta\mu(s_t)\nabla_aQ^\mu(s_t,a)|_{a=\mu(s_t)}\mid s_0\big]$. Average over $s_0\sim\rho_0$.

The deterministic gradient is the expectation over states only. An off-policy version simply averages the same expression over the states in a buffer, $\mathbb{E}_{s\sim\mathcal{R}}\big[\nabla_\theta\mu_\theta(s)\nabla_a\hat Q_\phi(s,a)|_{a=\mu_\theta(s)}\big]$, and needs **no importance weights**, because there is no integral over actions to correct. Silver et al. also show that the stochastic policy gradient converges to the deterministic one as the policy's noise shrinks to zero.[^silver2014]

### DDPG and TD3

**DDPG** (deep deterministic policy gradient; Lillicrap et al., 2016)[^lillicrap2016] makes this work with neural networks. It can be read as "approximate Q-learning for continuous actions", where the actor $\mu_\theta(s)\approx\arg\max_a\hat Q_\phi(s,a)$ is a learned maximizer.

- **Critic target:** $y_i=r_i+\gamma\,\hat Q_{\bar\phi}\big(s_i',\mu_{\bar\theta}(s_i')\big)$, computed with **target networks** $\bar\phi,\bar\theta$.
- **Soft target updates:** after each step, $\bar\phi\leftarrow\kappa\phi+(1-\kappa)\bar\phi$ and the same for $\bar\theta$, with small $\kappa$ (the paper uses $0.001$). The targets then move slowly, which stabilizes the regression, the same idea as the target network of DQN.[^mnih2015]
- **Actor:** a gradient step along $\frac1B\sum_i\nabla_\theta\mu_\theta(s_i)\nabla_a\hat Q_\phi(s_i,a)|_{a=\mu_\theta(s_i)}$.
- **Exploration:** a deterministic policy does not explore, so noise is added to the actions it executes. The paper used temporally correlated Ornstein–Uhlenbeck noise; plain Gaussian noise is common later.

DDPG is sample-efficient but brittle. Its main failure mode is the **maximization bias** familiar from [[Q-Learning and SARSA]]: the actor climbs whatever errors the critic has, so the critic's overestimates are selected and then bootstrapped into the targets. **TD3** (twin delayed DDPG; Fujimoto et al., 2018)[^fujimoto2018] adds three targeted fixes:

1. **Clipped double-Q learning.** Learn two critics and use the smaller target, $y_i=r_i+\gamma\min_{j=1,2}\hat Q_{\bar\phi_j}(s_i',\tilde a_i')$. A pessimistic target counteracts the upward bias, in the spirit of double Q-learning.
2. **Target policy smoothing.** Evaluate the target at a noisy action, $\tilde a'=\operatorname{clip}\big(\mu_{\bar\theta}(s')+\operatorname{clip}(\epsilon,-c,c),\,a_{\text{low}},a_{\text{high}}\big)$ with $\epsilon\sim\mathcal{N}(0,\tilde\sigma^2)$. Similar actions should have similar values, so the actor cannot exploit narrow spikes in $\hat Q$.
3. **Delayed policy updates.** Update the actor (and the targets) only every $d$ critic updates (the paper uses $d=2$), so the actor follows a critic that has had time to settle. This is a crude version of the two-timescale idea below.

### Soft actor-critic: maximum-entropy RL

**Soft actor-critic** (SAC; Haarnoja et al., 2018)[^haarnoja2018] keeps a stochastic actor and changes the objective. It rewards the policy for staying random:

$$
\begin{aligned}
J(\pi)&=\sum_{t\ge0}\gamma^t\,\mathbb{E}\Big[r(s_t,a_t)+\tau\,\mathcal{H}\big(\pi(\cdot\mid s_t)\big)\Big],\\
\mathcal{H}(\pi(\cdot\mid s))&=-\mathbb{E}_{a\sim\pi}\big[\log\pi(a\mid s)\big].
\end{aligned}
$$

The temperature $\tau>0$ sets the exchange rate between reward and entropy, and $\tau\to0$ recovers the standard objective. The value functions get "soft" versions, with the entropy bonus folded into the value of the next state:

$$
\begin{aligned}
Q^\pi(s,a)&=r(s,a)+\gamma\,\mathbb{E}_{s'\sim p(\cdot\mid s,a)}\big[V^\pi(s')\big],\\
V^\pi(s)&=\mathbb{E}_{a\sim\pi(\cdot\mid s)}\big[Q^\pi(s,a)-\tau\log\pi(a\mid s)\big].
\end{aligned}
$$

For a fixed $Q$, the policy that maximizes $\mathbb{E}_{a\sim\pi}[Q(s,a)]+\tau\mathcal{H}(\pi)$ at a state is the **Boltzmann policy** $\pi(a\mid s)\propto\exp\big(Q(s,a)/\tau\big)$. This is the variational characterization of the Boltzmann distribution from [[Energy-Based Models]], with $-Q$ as the energy. SAC's policy improvement step moves the actor towards it, by minimizing $D_{\mathrm{KL}}\big(\pi_\theta(\cdot\mid s)\,\|\,\exp(Q(s,\cdot)/\tau)/Z(s)\big)$. With the reparameterized action $a=f_\theta(s,\xi)$, the actor loss is

$$
\mathcal{L}_\pi(\theta)=\mathbb{E}_{s\sim\mathcal{R},\,\xi}\Big[\tau\log\pi_\theta\big(f_\theta(s,\xi)\mid s\big)-\hat Q_\phi\big(s,f_\theta(s,\xi)\big)\Big],
$$

and the critic regresses on $y=r+\gamma\big(\min_{j}\hat Q_{\bar\phi_j}(s',a')-\tau\log\pi_\theta(a'\mid s')\big)$ with $a'\sim\pi_\theta(\cdot\mid s')$. Like TD3, SAC uses two critics with the clipped minimum and Polyak-averaged targets. Actions are a Gaussian squashed by $\tanh$ into the action bounds. A later version also tunes $\tau$ automatically, so that the policy's entropy stays near a target value.[^haarnoja2018b]

![[Reinforcement Learning/images/actor-critic-soft-policies.svg]]
*Figure 4: soft-optimal policies for a two-peaked critic. The narrow peak at $a\approx0.9$ is higher; the wide peak at $a\approx-0.8$ is lower but more forgiving. A greedy (or DDPG) actor commits to the narrow peak. The Boltzmann policy puts probability 0.000, 0.264, 0.545 and 0.534 on the wider left peak at $\tau=0.02$, $0.1$, $0.3$ and $1$.*

Figure 4 shows why entropy helps. A deterministic actor bets everything on the single best action under the current critic, and if that peak is a critic error it gets exploited. A soft policy keeps probability on every good region in proportion to how much good action-space it contains. That gives exploration for free, robustness to critic errors, and smoother optimization. At moderate temperature the wide, slightly lower peak even gets *more* mass than the narrow, higher one.

---

## Convergence: two timescales

Do actor-critic methods converge? With neural networks and replay buffers nothing is guaranteed, and the instabilities of the [[Off-Policy Learning with Function Approximation|deadly triad]] apply in full. With a **linear critic** and **on-policy** samples, there is a clean theory. It rests on two ideas: features that make the critic's error harmless for the actor, and step sizes that let the critic keep up.

### Compatible function approximation

A critic with a few features can only approximate $Q^\pi$. Surprisingly, for one particular choice of features its error does not bias the policy gradient at all.

> [!note] Theorem (compatible function approximation; Sutton et al., 1999)
> Let $\boldsymbol\psi_\theta(s,a)=\nabla_\theta\log\pi_\theta(a\mid s)$ be the **compatible features**, and let the critic be linear in them, $\hat Q_{\mathbf{w}}(s,a)=\mathbf{w}^\top\boldsymbol\psi_\theta(s,a)$. Let $\mathbf{w}^*$ minimize the mean squared error
> $$
> \mathcal{E}(\mathbf{w})=\mathbb{E}_{s\sim d_\theta,\,a\sim\pi_\theta(\cdot\mid s)}\Big[\big(Q^{\pi_\theta}(s,a)-\mathbf{w}^\top\boldsymbol\psi_\theta(s,a)\big)^2\Big],
> $$
> where $d_\theta(s)\propto\sum_t\gamma^tp_\theta(s_t=s)$ is the discounted state distribution appearing in (1). Then replacing $Q^{\pi_\theta}$ by $\hat Q_{\mathbf{w}^*}$ in (1) gives the **exact** policy gradient.

> [!proof]- Proof
> The first-order optimality condition for $\mathbf{w}^*$ is
> $$
> \nabla_{\mathbf{w}}\mathcal{E}(\mathbf{w}^*)=-2\,\mathbb{E}\Big[\boldsymbol\psi_\theta(s,a)\big(Q^{\pi_\theta}(s,a)-\mathbf{w}^{*\top}\boldsymbol\psi_\theta(s,a)\big)\Big]=\mathbf{0},
> $$
> that is, $\mathbb{E}\big[\boldsymbol\psi_\theta\,Q^{\pi_\theta}\big]=\mathbb{E}\big[\boldsymbol\psi_\theta\,\hat Q_{\mathbf{w}^*}\big]$. By (1), up to the normalizing constant of $d_\theta$, the left side is $\nabla_\theta J(\theta)$, and the right side is the same formula with $\hat Q_{\mathbf{w}^*}$ in place of $Q^{\pi_\theta}$.

The critic's error is orthogonal to the features the actor uses, so the actor never sees it. Three remarks:

- Since $\mathbb{E}_{a\sim\pi}[\boldsymbol\psi_\theta(s,a)]=\mathbf{0}$, a compatible critic has zero mean over actions at every state. It is really an **advantage** approximator, and a state-dependent baseline can be added freely.
- $\mathbb{E}[\boldsymbol\psi_\theta\boldsymbol\psi_\theta^\top]=F(\theta)$ is the Fisher information matrix of the policy, so $\nabla_\theta J=F(\theta)\mathbf{w}^*$. The critic's weights are exactly the **natural gradient** $F^{-1}\nabla_\theta J$, the basis of natural actor-critic methods.[^bhatnagar2009]
- The theorem needs the *least-squares* fit. TD learning finds a different point (the TD fixed point of [[Linear Function Approximation]]), which is why the convergence results below carry a bias term unless the critic uses $\lambda=1$.

### Two-timescale stochastic approximation

The online actor-critic updates the critic and the actor at the same time. The critic chases $V^{\pi_\theta}$, which moves whenever $\theta$ moves. The fix that makes this analyzable is to let the actor move **more slowly** than the critic. Then the critic sees an almost frozen policy and has time to converge to its fixed point, while the actor sees an almost converged critic. In general form, with a fast iterate $\mathbf{w}_t$ (the critic) and a slow iterate $\theta_t$ (the actor):

$$
\begin{aligned}
\mathbf{w}_{t+1}&=\mathbf{w}_t+\beta_t\big[h(\mathbf{w}_t,\theta_t)+M^{(1)}_{t+1}\big],\\
\theta_{t+1}&=\theta_t+\alpha_t\big[g(\mathbf{w}_t,\theta_t)+M^{(2)}_{t+1}\big].
\end{aligned}
$$

> [!note] Theorem (two-timescale stochastic approximation; Borkar, 1997)
> Assume:
> 1. $h$ and $g$ are Lipschitz.
> 2. The step sizes satisfy $\sum_t\alpha_t=\sum_t\beta_t=\infty$, $\sum_t(\alpha_t^2+\beta_t^2)<\infty$, and $\alpha_t/\beta_t\to0$ (the actor is slower).
> 3. $M^{(1)}_t$ and $M^{(2)}_t$ are martingale-difference sequences with respect to $\mathcal{F}_t=\sigma(\mathbf{w}_m,\theta_m,M^{(1)}_m,M^{(2)}_m:m\le t)$, with $\mathbb{E}\big[\lVert M^{(i)}_{t+1}\rVert^2\mid\mathcal{F}_t\big]\le K\big(1+\lVert\mathbf{w}_t\rVert^2+\lVert\theta_t\rVert^2\big)$.
> 4. The iterates stay bounded: $\sup_t\big(\lVert\mathbf{w}_t\rVert+\lVert\theta_t\rVert\big)<\infty$ almost surely.
> 5. For each fixed $\theta$, the ODE $\dot{\mathbf{w}}=h(\mathbf{w},\theta)$ has a globally asymptotically stable equilibrium $\mathbf{w}^*(\theta)$, and $\theta\mapsto\mathbf{w}^*(\theta)$ is Lipschitz.
> 6. The ODE $\dot\theta=g\big(\mathbf{w}^*(\theta),\theta\big)$ has a globally asymptotically stable equilibrium $\theta^*$.
>
> Then $(\mathbf{w}_t,\theta_t)\to\big(\mathbf{w}^*(\theta^*),\theta^*\big)$ almost surely.

This is the theorem of [[Stochastic Approximation#Two-timescale stochastic approximation|Stochastic Approximation]], written in actor-critic letters: the fast step $a_n$ there is the critic's $\beta_t$ here, and the slow step $b_n$ is the actor's $\alpha_t$.[^borkar1997]<sup>,</sup>[^borkar2008] For a TD critic, $h(\mathbf{w},\theta)$ is the expected TD update under $\pi_\theta$, and $\mathbf{w}^*(\theta)$ is its TD fixed point. The slow ODE $\dot\theta=g(\mathbf{w}^*(\theta),\theta)$ is then gradient ascent on $J$ plus a critic-bias term. Two conditions need care in actor-critic:

- **Condition 6 rarely holds as stated.** Gradient ascent on a non-concave $J$ has many local maxima, not one globally stable point. The general version of the theorem replaces it by convergence to an invariant set of the slow ODE, here a set of (approximately) stationary points of $J$.
- **The noise is Markov, not martingale.** Samples come from one trajectory of the MDP, so consecutive updates are correlated. The theorem extends to this case by averaging over the stationary distribution of the chain; see Borkar's book, Chapter 6, and the [[Stochastic Approximation#Markov noise|Markov-noise theorem]].

A typical choice is $\beta_t=(t+1)^{-0.6}$ and $\alpha_t=(t+1)^{-1}$. The ratio $\alpha_t/\beta_t=(t+1)^{-0.4}\to0$.

### Konda and Tsitsiklis: actor-critic with linear critics

Konda and Tsitsiklis (2000, 2003) carried out this programme for a concrete algorithm.[^konda2000]<sup>,</sup>[^konda2003] Their setting is the **average-reward** criterion with finite $\mathcal{S}$ and $\mathcal{A}$:

- a smooth policy family $\pi_\theta$ (twice differentiable, with bounded derivatives) such that every policy induces an irreducible, aperiodic Markov chain;
- a linear critic for $Q$, trained by TD($\lambda$), whose features **span the compatible features** $\boldsymbol\psi_\theta$;
- an actor update along $\hat Q_{\mathbf{w}}(s_t,a_t)\,\boldsymbol\psi_\theta(s_t,a_t)$;
- two-timescale step sizes with the actor slower than the critic, and a bound that keeps the critic's influence on the actor finite.

The result: with a TD(1) critic, $\liminf_t\lVert\nabla_\theta\bar J(\theta_t)\rVert=0$ almost surely, where $\bar J$ is the average reward. The actor visits arbitrarily small neighbourhoods of stationary points infinitely often. With TD($\lambda$), $\lambda<1$, the same holds up to a bias: $\liminf_t\lVert\nabla_\theta\bar J(\theta_t)\rVert\le\varepsilon(\lambda)$, with $\varepsilon(\lambda)\to0$ as $\lambda\to1$. This is the bias–variance trade-off of GAE again, now in a theorem: $\lambda<1$ reduces the critic's variance but leaves the actor with a biased gradient. Bhatnagar et al. (2009) proved analogous results for natural-gradient actor-critic algorithms.[^bhatnagar2009]

### A small experiment: what if the actor outruns the critic?

The theorem is sufficient, not necessary. What actually goes wrong when the timescales are reversed? Consider a "fork": from $s_0$, going left reaches a state that pays $0.5$ and ends; going right enters a chain of 6 states with no reward and then pays $1$. All rewards carry $\mathcal{N}(0,0.3^2)$ noise, and $\gamma=0.95$. Right is optimal ($Q^*(s_0,\text{right})=0.95^6=0.735$ against $0.95\cdot0.5=0.475$), but its value reaches $s_0$ only after the critic has propagated it six steps back. We run the online actor-critic with a tabular TD(0) critic, starting from $\hat V=0$, under three step-size schedules for 200,000 steps and 40 seeds each.

![[Reinforcement Learning/images/actor-critic-two-timescale.svg]]
*Figure 5: online actor-critic on the fork MDP (median and 10–90% range over 40 seeds). Left: probability of the optimal action. Right: the critic's estimate of the gap between right and left, $\gamma[\hat V(B_1)-\hat V(A)]$; the true gap is 0.26 (dashed). With the critic faster than the actor, all 40 runs end with $\pi(\text{right}\mid s_0)>0.9$; with equal step sizes too. With the actor 10 times faster than the critic, all 40 runs lock onto the left branch, $\pi(\text{right}\mid s_0)<0.1$, and the critic's gap stays near $-0.48$.*

The failure in the third schedule is exactly the one the theory guards against. Early on, the critic still values the right branch at about $0$ while the left branch pays $0.5$ immediately. A fast actor acts on this stale opinion and commits to the left. After that the right branch is almost never visited, so the critic never learns that it was better. On this easy problem equal step sizes also worked, and converged faster than the strictly two-timescale schedule, whose actor steps are 10 times smaller. Two timescales are a guarantee, not a speed-up. The practical counterparts are TD3's delayed actor updates, several critic steps per actor step, and a larger learning rate for the critic than for the actor.

---

## Code

The [Actor-Critic repository](https://github.com/parag1604/Actor-Critic) builds the algorithms of this lecture one idea at a time, in PyTorch with gymnasium, with learning curves over 5 seeds: REINFORCE with a learned baseline as the starting point, batch actor-critic with bootstrapped TD targets (and what goes wrong when the critic is under-fitted or a time limit is treated as termination), online one-step actor-critic, A2C with parallel environments, n-step returns and GAE (with a sweep over $\lambda$), a Gaussian actor for continuous actions, and Soft Actor-Critic, which reaches the same returns on InvertedPendulum and Pendulum with roughly 10 to 100 times fewer environment steps by reusing data from a replay buffer. The [Policy-Gradients repository](https://github.com/parag1604/Policy-Gradients) covers the step before (learned value baselines) and PPO.

## Key takeaways

- **Actor-critic = policy gradient + a learned critic.** The critic supplies a baseline and, via bootstrapping, replaces the noisy future by an estimate. The advantage $A^\pi=Q^\pi-V^\pi$ is the right weight, and $V^\pi$ suffices to estimate it, since $\mathbb{E}[\delta_t\mid s_t,a_t]=A^\pi(s_t,a_t)$.
- **Fitting the critic is supervised regression** on Monte Carlo targets (unbiased, noisy) or bootstrapped targets (low variance, biased, semi-gradient). Bootstrap through time limits, not through terminal states.
- **Two discountings.** The exact discounted gradient carries $\gamma^t$; practice drops it and uses $\gamma$ only to shorten the horizon of the critic.
- **GAE** $\hat A_t=\sum_l(\gamma\lambda)^l\delta_{t+l}$ is an exponentially weighted average of $n$-step advantages. $\lambda$ trades the critic's bias against the returns' variance, and the best $\lambda$ grows with the amount of data per gradient step.
- **Off-policy actor-critic** needs a **Q-function** critic queried at fresh actions from the current policy. DDPG uses a deterministic actor (deterministic policy gradient) and target networks. TD3 fights overestimation with clipped double-Q, smoothing and delayed actor updates. SAC maximizes reward plus entropy and moves towards the Boltzmann policy $\propto\exp(Q/\tau)$.
- **Convergence** (linear critic, on-policy): run the critic on a faster timescale than the actor ($\alpha_t/\beta_t\to0$). With compatible features and TD(1) the actor reaches stationary points of $J$. With $\lambda<1$ it reaches them up to a bias.

## Where this shows up next

- [[Monte Carlo and Temporal-Difference Learning]] (next): how to fit $V^\pi$ from samples properly. MC vs TD, $n$-step returns, and TD($\lambda$), whose $\lambda$-return is the critic target that matches GAE.
- [[Q-Learning and SARSA]]: learning $Q$ directly. Its maximization bias and double Q-learning are what TD3's clipped double-Q addresses.
- [[Linear Function Approximation]]: what a linear critic converges to (the TD fixed point) and how far that is from the best fit. This is the source of the bias term in the Konda–Tsitsiklis result.
- [[Off-Policy Learning with Function Approximation]]: why off-policy critics with bootstrapping and function approximation can diverge (the deadly triad), the instability that target networks and clipped targets only mitigate.
- [[Model-Based Reinforcement Learning]]: actor-critic methods trained on data from a learned model (Dyna-style, MBPO).
- [[Stochastic Approximation]]: the ODE method, Markov noise and the two-timescale theorem behind the convergence results.
- Trust-region and proximal methods (TRPO, PPO) combine GAE with a constrained actor update (future lecture).

<!-- References: footnote definitions below render as the numbered "References" list at the bottom of the page -->

[^barto1983]: Barto, A. G., Sutton, R. S., & Anderson, C. W. (1983). *Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problems*. IEEE Transactions on Systems, Man, and Cybernetics, SMC-13(5), 834–846. [doi:10.1109/TSMC.1983.6313077](https://doi.org/10.1109/TSMC.1983.6313077)
[^cs285ac]: Levine, S. *CS 285: Deep Reinforcement Learning*, UC Berkeley. Lecture 6 (Actor-Critic) and Lecture 8 (Q-Learning in Practice, for DDPG). [course page](https://rail.eecs.berkeley.edu/deeprlcourse/), [Lecture 6 slides](https://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-6.pdf)
[^sutton1999]: Sutton, R. S., McAllester, D., Singh, S., & Mansour, Y. (1999). *Policy Gradient Methods for Reinforcement Learning with Function Approximation*. NeurIPS. [link](https://papers.nips.cc/paper_files/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html)
[^nota2020]: Nota, C., & Thomas, P. S. (2020). *Is the Policy Gradient a Gradient?* AAMAS. [arXiv:1906.07073](https://arxiv.org/abs/1906.07073)
[^thomas2014]: Thomas, P. S. (2014). *Bias in Natural Actor-Critic Algorithms*. ICML, PMLR 32. [link](https://proceedings.mlr.press/v32/thomas14.html)
[^suttonbarto]: Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.), Section 13.5 (Actor–Critic Methods). [book website](http://incompleteideas.net/book/the-book-2nd.html)
[^mnih2016]: Mnih, V., Badia, A. P., Mirza, M., Graves, A., Lillicrap, T. P., Harley, T., Silver, D., & Kavukcuoglu, K. (2016). *Asynchronous Methods for Deep Reinforcement Learning*. ICML. [arXiv:1602.01783](https://arxiv.org/abs/1602.01783)
[^schulman2016gae]: Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2016). *High-Dimensional Continuous Control Using Generalized Advantage Estimation*. ICLR. [arXiv:1506.02438](https://arxiv.org/abs/1506.02438)
[^degris2012]: Degris, T., White, M., & Sutton, R. S. (2012). *Off-Policy Actor-Critic*. ICML. [arXiv:1205.4839](https://arxiv.org/abs/1205.4839)
[^silver2014]: Silver, D., Lever, G., Heess, N., Degris, T., Wierstra, D., & Riedmiller, M. (2014). *Deterministic Policy Gradient Algorithms*. ICML, PMLR 32. [link](https://proceedings.mlr.press/v32/silver14.html)
[^lillicrap2016]: Lillicrap, T. P., Hunt, J. J., Pritzel, A., Heess, N., Erez, T., Tassa, Y., Silver, D., & Wierstra, D. (2016). *Continuous Control with Deep Reinforcement Learning*. ICLR. [arXiv:1509.02971](https://arxiv.org/abs/1509.02971)
[^mnih2015]: Mnih, V., et al. (2015). *Human-Level Control through Deep Reinforcement Learning*. Nature, 518, 529–533. [doi:10.1038/nature14236](https://doi.org/10.1038/nature14236)
[^fujimoto2018]: Fujimoto, S., van Hoof, H., & Meger, D. (2018). *Addressing Function Approximation Error in Actor-Critic Methods*. ICML. [arXiv:1802.09477](https://arxiv.org/abs/1802.09477)
[^haarnoja2018]: Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018). *Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor*. ICML. [arXiv:1801.01290](https://arxiv.org/abs/1801.01290)
[^haarnoja2018b]: Haarnoja, T., Zhou, A., Hartikainen, K., Tucker, G., Ha, S., Tan, J., Kumar, V., Zhu, H., Gupta, A., Abbeel, P., & Levine, S. (2018). *Soft Actor-Critic Algorithms and Applications*. [arXiv:1812.05905](https://arxiv.org/abs/1812.05905)
[^bhatnagar2009]: Bhatnagar, S., Sutton, R. S., Ghavamzadeh, M., & Lee, M. (2009). *Natural Actor–Critic Algorithms*. Automatica, 45(11), 2471–2482. [doi:10.1016/j.automatica.2009.07.008](https://doi.org/10.1016/j.automatica.2009.07.008)
[^borkar1997]: Borkar, V. S. (1997). *Stochastic Approximation with Two Time Scales*. Systems & Control Letters, 29(5), 291–294. [doi:10.1016/S0167-6911(97)90015-3](https://doi.org/10.1016/S0167-6911(97)90015-3)
[^borkar2008]: Borkar, V. S. (2008). *Stochastic Approximation: A Dynamical Systems Viewpoint*. Hindustan Book Agency / Cambridge University Press, Chapter 6 (Multiple Timescales). [doi:10.1007/978-93-86279-38-5](https://doi.org/10.1007/978-93-86279-38-5)
[^konda2000]: Konda, V. R., & Tsitsiklis, J. N. (2000). *Actor-Critic Algorithms*. NeurIPS 12. [link](https://papers.nips.cc/paper_files/paper/1999/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html)
[^konda2003]: Konda, V. R., & Tsitsiklis, J. N. (2003). *On Actor-Critic Algorithms*. SIAM Journal on Control and Optimization, 42(4), 1143–1166. [doi:10.1137/S0363012901385691](https://doi.org/10.1137/S0363012901385691)
