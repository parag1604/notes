---
title: Policy Gradients
tags:
  - reinforcement-learning
  - policy-gradients
  - reinforce
  - importance-sampling
prev: "[[Imitation Learning]]"
next: "[[Actor-Critic Algorithms]]"
---

> **Prerequisites:** [[Introduction to Reinforcement Learning]] (MDPs, trajectories, $Q^\pi$ and $V^\pi$), [[Imitation Learning]] (behavior cloning as maximum likelihood).
>
> **TL;DR:** Write the RL objective as an expectation over trajectories and differentiate it with the log-derivative trick. The result is REINFORCE: maximum likelihood on your own samples, with each action's log-probability weighted by the reward that followed it. The estimator is unbiased but very noisy. *Baselines* and *reward-to-go* reduce the variance without adding bias, and *importance sampling* lets us reuse old data.
>
> **Code:** [Policy-Gradients](https://github.com/parag1604/Policy-Gradients): REINFORCE built up one idea at a time: reward-to-go, baselines, continuous actions; [Importance-Sampling](https://github.com/parag1604/Importance-Sampling): importance sampling, the tool behind off-policy policy gradients.

## Motivation

[[Imitation Learning|Behavior cloning]] copies every action of an expert. Without an expert, we can still collect our own trajectories and see how well they did. The natural idea is to make the actions that led to high reward *more likely* and the rest less likely. Policy gradient methods turn this "trial and error" idea into exact gradient ascent on the expected return. The presentation loosely follows CS285.[^cs285] Sutton & Barto, Chapter 13, is the textbook treatment.[^suttonbarto]

## The objective and its gradient

Recall the goal of reinforcement learning ([[Introduction to Reinforcement Learning]]). For a parametric policy $\pi_\theta(a\mid s)$ and horizon $T$ (time-steps $t=0,\dots,T-1$),

$$
\theta^*=\arg\max_\theta\sum_{t=0}^{T-1}\mathbb{E}_{(s_t,a_t)\sim p_\theta(s_t,a_t)}\big[r(s_t,a_t)\big]=\arg\max_\theta\underbrace{\mathbb{E}_{\tau\sim p_\theta(\tau)}\left[\sum_{t=0}^{T-1}r(s_t,a_t)\right]}_{J(\theta)}.
$$

$J(\theta)$ can be estimated from samples. Run $\pi_\theta$ to collect $N$ trajectories and average their returns. This is a Monte-Carlo estimate:

$$
J(\theta)\approx\frac{1}{N}\sum_{i=1}^{N}\sum_{t=0}^{T-1}r(s_{i,t},a_{i,t}).
$$

Write $R(\tau)=\sum_{t=0}^{T-1}r(s_t,a_t)$ for the return of a trajectory. Then

$$
\nabla_\theta J(\theta)=\nabla_\theta\int p_\theta(\tau)\,R(\tau)\,d\tau=\int\nabla_\theta p_\theta(\tau)\,R(\tau)\,d\tau.
$$

**The log-derivative trick.** Since $\nabla_\theta\log p_\theta(\tau)=\frac{\nabla_\theta p_\theta(\tau)}{p_\theta(\tau)}$, we have $\nabla_\theta p_\theta(\tau)=p_\theta(\tau)\,\nabla_\theta\log p_\theta(\tau)$. Substituting this back gives an expectation again, which we can sample:

$$
\nabla_\theta J(\theta)=\int p_\theta(\tau)\,\nabla_\theta\log p_\theta(\tau)\,R(\tau)\,d\tau=\mathbb{E}_{\tau\sim p_\theta(\tau)}\big[\nabla_\theta\log p_\theta(\tau)\,R(\tau)\big].\tag{1}
$$

**The dynamics drop out.** The trajectory distribution factorizes as

$$
p_\theta(\tau)=p_\theta(s_0,a_0,\dots,s_{T-1},a_{T-1},s_T)=\rho(s_0)\prod_{t=0}^{T-1}\pi_\theta(a_t\mid s_t)\,p(s_{t+1}\mid s_t,a_t),
$$

so

$$
\begin{aligned}
\log p_\theta(\tau)&=\underbrace{\log\rho(s_0)}_{\text{independent of }\theta}+\sum_{t=0}^{T-1}\Big[\log\pi_\theta(a_t\mid s_t)+\underbrace{\log p(s_{t+1}\mid s_t,a_t)}_{\text{independent of }\theta}\Big]\\
\implies\nabla_\theta\log p_\theta(\tau)&=\sum_{t=0}^{T-1}\nabla_\theta\log\pi_\theta(a_t\mid s_t).
\end{aligned}
$$

We never need to know $\rho$ or $p(s'\mid s,a)$: policy gradients are **model-free**. Putting this into (1):

$$
\begin{aligned}
\nabla_\theta J(\theta)&=\mathbb{E}_{\tau\sim p_\theta(\tau)}\left[\left(\sum_{t=0}^{T-1}\nabla_\theta\log\pi_\theta(a_t\mid s_t)\right)\left(\sum_{t=0}^{T-1}r(s_t,a_t)\right)\right]\\
&\approx\underbrace{\frac{1}{N}\sum_{i=1}^{N}}_{\text{generate samples}}\left(\sum_{t=0}^{T-1}\nabla_\theta\log\pi_\theta(a_{i,t}\mid s_{i,t})\right)\underbrace{\left(\sum_{t=0}^{T-1}r(s_{i,t},a_{i,t})\right)}_{\text{estimate returns}}.
\end{aligned}
$$

In terms of the [[Introduction to Reinforcement Learning#Anatomy of an RL algorithm|anatomy of an RL algorithm]], the sum over samples is the *generate samples* step and the return is the *fit a model / estimate returns* step. The *improve the policy* step is plain gradient ascent with learning rate $\alpha$:

$$
\theta\leftarrow\theta+\alpha\,\nabla_\theta J(\theta).
$$

> [!tip] Discounting
> With a discount factor $\gamma\in(0,1]$, replace $R(\tau)$ by $\sum_t\gamma^t r(s_t,a_t)$. Nothing in the derivation changes, because the dynamics still drop out.

---

## REINFORCE

> [!example] Algorithm: REINFORCE
> Williams (1992).[^williams1992]
> **Repeat** until the policy stops improving (or for a fixed budget of iterations):
> 1. Collect $N$ trajectories $\{\tau_i\}_{i=1}^N$ by running $\pi_\theta(a_t\mid s_t)$ in the environment.
> 2. Estimate $\displaystyle\nabla_\theta J(\theta)\approx\frac{1}{N}\sum_{i=1}^{N}\left(\sum_{t=0}^{T-1}\nabla_\theta\log\pi_\theta(a_{i,t}\mid s_{i,t})\right)\left(\sum_{t=0}^{T-1}r(s_{i,t},a_{i,t})\right)$.
> 3. Update $\theta\leftarrow\theta+\alpha\,\nabla_\theta J(\theta)$, a step *up* the gradient.

### Policy gradient = weighted maximum likelihood

Compare this with the maximum-likelihood gradient from [[Imitation Learning#Behavior Cloning (BC)|behavior cloning]]:

$$
\nabla_\theta J_{\text{MLE}}(\theta)\approx\frac{1}{N}\sum_{i=1}^{N}\sum_{t=0}^{T-1}\nabla_\theta\log\pi_\theta(a_{i,t}\mid s_{i,t}).
$$

The policy gradient has one extra factor, the return $R(\tau_i)$, which **weights** each trajectory's log-likelihood. Policy gradient is therefore formalized *trial and error*. Actions from more rewarding trajectories get more weight in the update, which *reinforces* the behavior that led to reward (hence the name). Unlike BC, the data comes from the current policy itself, not from an expert. With a horizon of one step, REINFORCE with a softmax policy is exactly the [[Multi-Armed Bandits#Gradient bandit algorithms|gradient bandit]] algorithm.

---

## The variance problem

The estimator is unbiased but has **very high variance**, so it needs many samples or very small steps. A running example shows why. Take a task with a fixed horizon of $T=100$, where "good" trajectories reach a goal.

- **Reward design A:** $+1$ for every step survived, plus an extra $+1$ for reaching the goal. Bad trajectories return $100$; good ones return $101$.
- **Reward design B:** $0$ per step, $+1$ for reaching the goal. Bad trajectories return $0$; good ones return $1$.

Both designs rank trajectories the same way, but REINFORCE behaves very differently on them. Under A, every sampled trajectory has its log-likelihood pushed up with weight about $100$. The useful signal, the difference of $1$ between good and bad, is buried under a huge common term, and that term only cancels *on average*. With finitely many samples it adds large noise to the gradient. Under B, only the good trajectories are pushed up, so the signal is clean. **PG works much better under design B.**

### Baselines

Design B is design A with $100$ subtracted from every return. We can do this subtraction ourselves, whatever the rewards are, by introducing a **baseline** $b$:

$$
\nabla_\theta J(\theta)\approx\frac{1}{N}\sum_{i=1}^{N}\nabla_\theta\log p_\theta(\tau_i)\,\big(R(\tau_i)-b\big).
$$

Trajectories better than $b$ get pushed up, and worse ones get pushed down. A simple and effective choice is the average return, $b=\frac{1}{N}\sum_{i=1}^{N}R(\tau_i)$.

**Baselines are unbiased.** For any constant $b$,

$$
\begin{aligned}
\mathbb{E}_{\tau\sim p_\theta}\big[\nabla_\theta\log p_\theta(\tau)\,b\big]&=\int p_\theta(\tau)\,\nabla_\theta\log p_\theta(\tau)\,b\,d\tau\\
&=b\int\nabla_\theta p_\theta(\tau)\,d\tau=b\,\nabla_\theta\underbrace{\int p_\theta(\tau)\,d\tau}_{=1}=0.
\end{aligned}
$$

So subtracting a baseline leaves the *expected* gradient unchanged and only changes its variance. (Strictly, a $b$ computed from the same batch adds a small $O(1/N)$ bias. In practice it is negligible, or it can be avoided by a leave-one-out average.) The baseline can also be a learned function, trained by regression alongside the policy.

![[Reinforcement Learning/images/policy-gradients-baseline-variance.svg]]
*Figure 1: REINFORCE gradient estimates for a two-action problem with returns 100 (bad action) and 101 (good action), as in reward design A. Each histogram shows 2000 single-sample estimates of $\partial J/\partial\theta$. Both histograms are centred on the same true gradient (dashed). With no baseline the spread is roughly 100× larger.*

### Optimal baseline (variance analysis)

What baseline minimizes the variance? Work with a single component $k$ of the gradient, and write $g(\tau)=\partial_{\theta_k}\log p_\theta(\tau)$. Then

$$
\operatorname{Var}=\mathbb{E}\Big[g(\tau)^2\big(R(\tau)-b\big)^2\Big]-\Big(\mathbb{E}\big[g(\tau)\big(R(\tau)-b\big)\big]\Big)^2.
$$

The second term equals $\big(\mathbb{E}[g(\tau)R(\tau)]\big)^2$ for every $b$, because baselines are unbiased. So it does not depend on $b$:

$$
\begin{aligned}
\frac{d\operatorname{Var}}{db}&=\frac{d}{db}\Big(\mathbb{E}\big[g(\tau)^2R(\tau)^2\big]-2b\,\mathbb{E}\big[g(\tau)^2R(\tau)\big]+b^2\,\mathbb{E}\big[g(\tau)^2\big]\Big)\\
&=-2\,\mathbb{E}\big[g(\tau)^2R(\tau)\big]+2b\,\mathbb{E}\big[g(\tau)^2\big]=0\\
\implies b^*&=\frac{\mathbb{E}\big[g(\tau)^2R(\tau)\big]}{\mathbb{E}\big[g(\tau)^2\big]}.
\end{aligned}
$$

The optimal baseline is the **expected return weighted by squared gradient magnitudes**, and it is different for each parameter component.[^weaver2001] In practice the plain average return is almost as good and much simpler.

### Reward-to-go (causality)

A second, free variance reduction comes from **causality**: the action at time $t$ cannot affect rewards collected *before* $t$. In $\big(\sum_t\nabla\log\pi_\theta(a_t\mid s_t)\big)\big(\sum_{t'}r_{t'}\big)$, the cross terms with $t'<t$ have zero expectation but add noise. Drop them:

$$
\nabla_\theta J(\theta)\approx\frac{1}{N}\sum_{i=1}^{N}\sum_{t=0}^{T-1}\nabla_\theta\log\pi_\theta(a_{i,t}\mid s_{i,t})\underbrace{\left(\sum_{t'=t}^{T-1}r(s_{i,t'},a_{i,t'})\right)}_{\hat Q_{i,t}\,\approx\,Q^{\pi}(s_{i,t},a_{i,t})}.
$$

The **reward-to-go** $\hat Q_{i,t}$ is a single-sample Monte-Carlo estimate of the state-action value $Q^{\pi}(s_{i,t},a_{i,t})$. Replacing it by the true $Q^\pi$ gives the **policy gradient theorem**, $\nabla_\theta J=\mathbb{E}\big[\sum_t\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,Q^\pi(s_t,a_t)\big]$.[^sutton1999]

**Baselines with reward-to-go.** A constant $b$ is *still unbiased* here. But it is a poor choice, because the scale of the reward-to-go changes along the trajectory. In design A, $\hat Q$ is about $100$ at $s_0$, about $50$ at $s_{50}$ and about $0$ near $s_{100}$, so no single constant centres them all. The natural fix is a **state-dependent baseline** $b(s_t)$. It is also unbiased, since it does not depend on $a_t$:

$$
\nabla_\theta J(\theta)\approx\frac{1}{N}\sum_{i=1}^{N}\sum_{t=0}^{T-1}\nabla_\theta\log\pi_\theta(a_{i,t}\mid s_{i,t})\Big(\hat Q_{i,t}-b(s_{i,t})\Big).
$$

The best such baseline is close to the value function, $b(s_t)\approx V^{\pi}(s_t)$. The weight then becomes an estimate of the **advantage** $A^{\pi}(s_t,a_t)=Q^{\pi}(s_t,a_t)-V^{\pi}(s_t)$: how much better this action was than the policy's average action. Learning $V^\pi$ with a neural network ("the critic") turns REINFORCE into an **actor-critic** method ([[Actor-Critic Algorithms]], next).[^schulman2016gae]

![[Reinforcement Learning/images/policy-gradients-reward-to-go.svg]]
*Figure 2: in reward design A, the reward-to-go $\hat Q_t$ shrinks linearly along the trajectory. A constant baseline (dashed) fits it only at one point, while a state-dependent baseline $b(s_t)\approx V^\pi(s_t)$ tracks it at every step.*

---

## Off-policy Policy Gradients

The algorithm above is completely **on-policy**. Once $\theta$ is updated, every trajectory collected so far was sampled from the *old* policy and must be thrown away. This is wasteful, because collecting trajectories took effort (and, on a real robot, real time). Can we reuse old samples?

**Importance sampling.** For any distributions $p$ and $q$,

$$
\mathbb{E}_{x\sim p(x)}\big[f(x)\big]=\int p(x)f(x)\,dx=\int q(x)\frac{p(x)}{q(x)}f(x)\,dx=\mathbb{E}_{x\sim q(x)}\left[\frac{p(x)}{q(x)}f(x)\right].
$$

$q$ is ours to choose, as long as $q(x)>0$ wherever $p(x)f(x)\neq0$. A proposal with the same support as $p$ is enough.

**Applied to trajectories.** Suppose we have trajectories from some other distribution $\bar p(\tau)$, generated by a policy $\bar\pi$. Then

$$
J(\theta)=\mathbb{E}_{\tau\sim\bar p(\tau)}\left[\frac{p_\theta(\tau)}{\bar p(\tau)}R(\tau)\right],
$$

and the unknown initial-state distribution and dynamics cancel in the ratio:

$$
\frac{p_\theta(\tau)}{\bar p(\tau)}=\frac{\rho(s_0)\prod_{t=0}^{T-1}\pi_\theta(a_t\mid s_t)\,p(s_{t+1}\mid s_t,a_t)}{\rho(s_0)\prod_{t=0}^{T-1}\bar\pi(a_t\mid s_t)\,p(s_{t+1}\mid s_t,a_t)}=\prod_{t=0}^{T-1}\frac{\pi_\theta(a_t\mid s_t)}{\bar\pi(a_t\mid s_t)}.\tag{2}
$$

**Notation.** The data in our buffer was collected by an older policy. Call it $\pi_\theta$, and call the current policy that we want to improve $\pi_{\theta'}$. This follows the convention that past objects get a plain letter and newer ones a prime, as with $s$ and $s'$. We want to update $\pi_{\theta'}$ using samples from $\pi_\theta$:

$$
\begin{aligned}
J(\theta')&=\mathbb{E}_{\tau\sim p_\theta(\tau)}\left[\frac{p_{\theta'}(\tau)}{p_\theta(\tau)}R(\tau)\right]\\
\nabla_{\theta'}J(\theta')&=\mathbb{E}_{\tau\sim p_\theta(\tau)}\left[\frac{\nabla_{\theta'}p_{\theta'}(\tau)}{p_\theta(\tau)}R(\tau)\right]=\mathbb{E}_{\tau\sim p_\theta(\tau)}\left[\frac{p_{\theta'}(\tau)}{p_\theta(\tau)}\nabla_{\theta'}\log p_{\theta'}(\tau)\,R(\tau)\right].
\end{aligned}
$$

(At $\theta'=\theta$ the ratio is $1$ and we recover the on-policy gradient.) Substituting (2) and expanding:

$$
\begin{aligned}
\nabla_{\theta'}J(\theta')=\mathbb{E}_{\tau\sim p_\theta(\tau)}\Bigg[&\left(\prod_{t=0}^{T-1}\frac{\pi_{\theta'}(a_t\mid s_t)}{\pi_\theta(a_t\mid s_t)}\right)\\
&\times\left(\sum_{t=0}^{T-1}\nabla_{\theta'}\log\pi_{\theta'}(a_t\mid s_t)\right)\left(\sum_{t=0}^{T-1}r(s_t,a_t)\right)\Bigg].
\end{aligned}
$$

**Causality again.** Future actions cannot affect the current gradient term, and the importance weight for reward $r_{t'}$ only needs actions up to $t'$:

$$
\begin{aligned}
\nabla_{\theta'}J(\theta')=\mathbb{E}_{\tau\sim p_\theta(\tau)}\Bigg[\sum_{t=0}^{T-1}&\nabla_{\theta'}\log\pi_{\theta'}(a_t\mid s_t)\left(\prod_{t'=0}^{t}\frac{\pi_{\theta'}(a_{t'}\mid s_{t'})}{\pi_\theta(a_{t'}\mid s_{t'})}\right)\\
&\times\left(\sum_{t'=t}^{T-1}r(s_{t'},a_{t'})\prod_{t''=t}^{t'}\frac{\pi_{\theta'}(a_{t''}\mid s_{t''})}{\pi_\theta(a_{t''}\mid s_{t''})}\right)\Bigg].
\end{aligned}
$$

If we drop the last product, the importance weights on *future rewards*, we no longer have the exact policy gradient. What remains still improves the policy, and it leads to a *policy iteration*-style algorithm (CS285, Lecture 5).[^cs285] Even in the reduced form

$$
\begin{aligned}
\nabla_{\theta'}J(\theta')\approx\mathbb{E}_{\tau\sim p_\theta(\tau)}\Bigg[\sum_{t=0}^{T-1}&\nabla_{\theta'}\log\pi_{\theta'}(a_t\mid s_t)\\
&\times\left(\prod_{t'=0}^{t}\frac{\pi_{\theta'}(a_{t'}\mid s_{t'})}{\pi_\theta(a_{t'}\mid s_{t'})}\right)\left(\sum_{t'=t}^{T-1}r(s_{t'},a_{t'})\right)\Bigg],
\end{aligned}
$$

the middle product is a product of up to $T$ ratios. It is cheap to *compute*, but its **variance grows exponentially in $T$**. If each ratio is a little below 1 the product vanishes, and if it is a little above 1 the product explodes. So this estimator works poorly for long horizons.

![[Reinforcement Learning/images/policy-gradients-is-weights.svg]]
*Figure 3: products of per-step importance ratios $\prod_{t'\le t}\pi_{\theta'}/\pi_\theta$ along 200 sampled trajectories, where the two policies differ only slightly. A few weights explode while most collapse towards zero; the y-axis is logarithmic. The variance of the product grows exponentially with $t$.*

**A "first-order" approximation.** Instead of weighting whole trajectory prefixes, weight each *state-action pair* by the ratio of its marginal probabilities under the two policies:

$$
\begin{aligned}
\nabla_\theta J_{\text{on-policy}}(\theta)&\approx\frac{1}{N}\sum_{i=1}^{N}\sum_{t=0}^{T-1}\nabla_\theta\log\pi_\theta(a_{i,t}\mid s_{i,t})\,\hat Q_{i,t}\\
\nabla_{\theta'}J_{\text{off-policy}}(\theta')&\approx\frac{1}{N}\sum_{i=1}^{N}\sum_{t=0}^{T-1}\frac{p_{\theta'}(s_{i,t},a_{i,t})}{p_\theta(s_{i,t},a_{i,t})}\nabla_{\theta'}\log\pi_{\theta'}(a_{i,t}\mid s_{i,t})\,\hat Q_{i,t}\\
&=\frac{1}{N}\sum_{i=1}^{N}\sum_{t=0}^{T-1}\underbrace{\frac{p_{\theta'}(s_{i,t})}{p_\theta(s_{i,t})}}_{\text{ignore}}\frac{\pi_{\theta'}(a_{i,t}\mid s_{i,t})}{\pi_\theta(a_{i,t}\mid s_{i,t})}\nabla_{\theta'}\log\pi_{\theta'}(a_{i,t}\mid s_{i,t})\,\hat Q_{i,t}\\
&\approx\frac{1}{N}\sum_{i=1}^{N}\sum_{t=0}^{T-1}\frac{\pi_{\theta'}(a_{i,t}\mid s_{i,t})}{\pi_\theta(a_{i,t}\mid s_{i,t})}\nabla_{\theta'}\log\pi_{\theta'}(a_{i,t}\mid s_{i,t})\,\hat Q_{i,t}.
\end{aligned}
$$

Here $p_\theta(s_t)$ is the state *marginal* at time $t$ under $\pi_\theta$, and we cannot compute it. Ignoring the state-marginal ratio is justified when $\pi_{\theta'}$ stays **close** to $\pi_\theta$. The error it introduces is then bounded by how far the policies differ, which is exactly the idea behind trust-region methods such as TRPO and PPO.[^schulman2015trpo] The result has only a single per-step ratio, so its variance stays under control.

---

## Straight-Through Gumbel-Softmax (STGS) for PG Algorithms

Everything so far uses the **score-function** (log-derivative) estimator. It never differentiates *through* the sampled action: the action is a black box, and only $\log\pi_\theta(a\mid s)$ is differentiated. That is exactly what makes REINFORCE work with a non-differentiable environment, and also why it is so noisy.

Often, though, part of the pipeline after the action *is* differentiable. A learned critic $Q_\phi(s,a)$, a learned reward model, a learned world model, or a differentiable simulator can all tell us how the objective changes as $a$ changes. For **continuous** actions we would then use the **reparameterization (pathwise) trick**, writing $a=\mu_\theta(s)+\sigma_\theta(s)\,\xi$ with $\xi\sim\mathcal{N}(0,I)$ and backpropagating through $a$. For **discrete** actions this seems impossible, because sampling a category is a step function with zero gradient almost everywhere. The straight-through Gumbel-softmax trick removes this obstacle and makes the *whole* computation graph, sampling included, end-to-end differentiable.

### Two estimators of the same gradient

For a single decision with objective $f(s,a)$, for example $f=Q_\phi(s,a)$:

**Score function (REINFORCE):**

$$
\nabla_\theta\,\mathbb{E}_{a\sim\pi_\theta}\big[f(s,a)\big]=\mathbb{E}_{a\sim\pi_\theta}\big[f(s,a)\,\nabla_\theta\log\pi_\theta(a\mid s)\big].
$$

**Pathwise**, if the action can be written as a differentiable function $a=h_\theta(s,\xi)$ of parameter-free noise $\xi$:

$$
\nabla_\theta\,\mathbb{E}_{a\sim\pi_\theta}\big[f(s,a)\big]=\mathbb{E}_{\xi}\Big[\nabla_a f(s,a)\big|_{a=h_\theta(s,\xi)}\,\nabla_\theta h_\theta(s,\xi)\Big].
$$

The score-function estimator only needs the *values* $f(s,a)$, and its variance is high. The pathwise estimator uses the *gradient* $\nabla_a f$, which carries far more information per sample, and its variance is typically much lower. STGS gives us an approximate $h_\theta$ for categorical actions.

### Step 1: the Gumbel-max trick (exact sampling as an argmax)

Let the policy output logits $\ell_1,\dots,\ell_K$ with $\pi_\theta(k\mid s)=\operatorname{softmax}(\ell)_k$. Draw independent Gumbel noise, $g_k=-\log(-\log u_k)$ with $u_k\sim\mathcal{U}(0,1)$. Then

$$
a=\arg\max_{k}\big(\ell_k+g_k\big)\quad\sim\quad\operatorname{Categorical}\big(\operatorname{softmax}(\ell)\big).
$$

> [!proof]- Why Gumbel-max samples exactly from the softmax
> The Gumbel$(\mu,1)$ CDF is $F(x)=\exp(-e^{-(x-\mu)})$, so $\ell_k+g_k\sim\text{Gumbel}(\ell_k,1)$. The probability that $k$ wins is
> $P(k)=\int p_k(x)\prod_{j\neq k}F_j(x)\,dx=\int e^{-(x-\ell_k)}\exp\!\Big(-\sum_j e^{-(x-\ell_j)}\Big)dx.$
> With $Z=\sum_j e^{\ell_j}$ and the substitution $v=Z e^{-x}$, this becomes $\frac{e^{\ell_k}}{Z}\int_0^\infty e^{-v}\,dv=\frac{e^{\ell_k}}{Z}=\operatorname{softmax}(\ell)_k$.

All the randomness now sits in parameter-free noise $g$, and the parameters enter through a deterministic function. The $\arg\max$, however, is still not differentiable.

### Step 2: relax the argmax with a tempered softmax

Replace the one-hot $\arg\max$ by a softmax with **temperature** $\tau>0$:

$$
y_k=\frac{\exp\big((\ell_k+g_k)/\tau\big)}{\sum_{j=1}^{K}\exp\big((\ell_j+g_j)/\tau\big)},\qquad k=1,\dots,K.
$$

$y$ is a point on the probability simplex and a smooth function of $\theta$. This is the **Gumbel-softmax**, or **Concrete**, distribution.[^jang2017]<sup>,</sup>[^maddison2017] As $\tau\to0$, $y$ approaches the one-hot Gumbel-max sample, so the relaxation is exact but the gradients become huge and spiky. As $\tau\to\infty$, $y$ approaches the uniform vector, which is smooth but carries no information. Typical practice is $\tau\approx0.5$–$1$, sometimes annealed towards $0$ during training.

![[Reinforcement Learning/images/policy-gradients-gumbel-softmax.svg]]
*Figure 4: one Gumbel-softmax sample (bars) for a 4-action policy (outlines show $\pi_\theta$), using the same Gumbel noise at four temperatures. Low $\tau$ gives a nearly one-hot action; high $\tau$ washes the sample out towards uniform.*

### Step 3: straight-through, so the forward pass is discrete and the backward pass is soft

The environment (or a game, or a tokenizer) usually needs an actual discrete action, not a soft mixture. The **straight-through** estimator[^bengio2013] uses the hard one-hot in the forward pass and the soft sample's gradient in the backward pass:

$$
y^{\text{hard}}=\operatorname{one\_hot}\Big(\arg\max_k y_k\Big),\qquad
\tilde a=y^{\text{hard}}-\operatorname{sg}\big[y\big]+y,
$$

where $\operatorname{sg}[\cdot]$ is stop-gradient (`.detach()`). Numerically $\tilde a=y^{\text{hard}}$, a valid discrete action. Under backpropagation $\partial\tilde a/\partial\theta=\partial y/\partial\theta$. The gradient treats the hard sample as if it were the soft one.

### Using STGS in a policy-gradient pipeline

With a differentiable critic, the actor's update becomes a pathwise gradient through the sampled discrete action:

$$
\nabla_\theta J(\theta)\approx\frac{1}{N}\sum_{i=1}^{N}\nabla_{\tilde a}\,Q_\phi(s_i,\tilde a)\;\frac{\partial\tilde a}{\partial\theta},\qquad\tilde a=\operatorname{STGS}_\tau\big(\ell_\theta(s_i)\big).
$$

This is the discrete-action analogue of DDPG/SAC-style actor updates. The same idea extends to a multi-step rollout through a learned or differentiable model. Every sampled action in the rollout is an STGS sample, so the return is differentiable with respect to $\theta$ through the entire trajectory, and we can backpropagate through time.

```python
import torch
import torch.nn.functional as F

logits = policy_net(states)                            # [B, K]
# hard=True: one-hot in the forward pass, Gumbel-softmax gradient in the backward pass
actions = F.gumbel_softmax(logits, tau=1.0, hard=True)  # [B, K], exactly one-hot
q_values = critic(states, actions)                      # differentiable in `actions`
actor_loss = -q_values.mean()
actor_loss.backward()                                   # gradients flow through the sampled action into policy_net
```

> [!warning] What STGS does and does not buy you
> - **A differentiable path is required.** STGS cannot differentiate through a black-box environment: `env.step()` breaks the graph. Without a learned or differentiable $Q$, reward or model after the action, the score-function estimator is still the only option.
> - **It is biased.** The relaxation is biased for any $\tau>0$, and the straight-through step adds more bias, because the forward pass uses $y^{\text{hard}}$ while the gradient is that of $y$. You trade the *unbiased but high-variance* REINFORCE gradient for a *biased, low-variance* one. In practice this trade is often worth it, especially with many actions or long chains of discrete choices.
> - **Temperature is a knob.** Small $\tau$ gives less bias but higher variance and exploding gradients. Large $\tau$ gives smoother but more biased gradients.
> - **The two estimators combine well.** Score-function and pathwise estimators can be mixed: for example, use STGS for the parts of the graph that are differentiable, and REINFORCE with a baseline for the reward that comes from the real environment.

## Key takeaways

- **Log-derivative trick:** $\nabla_\theta J=\mathbb{E}_{\tau\sim p_\theta}\big[\nabla_\theta\log p_\theta(\tau)\,R(\tau)\big]$. The dynamics cancel, so the method is model-free.
- **REINFORCE** is maximum likelihood on your own samples, weighted by return. It reinforces what worked.
- **Variance** is the main enemy. Reduce it with **baselines** (unbiased; average return or $V^\pi(s_t)$), **reward-to-go** (causality) and, in the end, **advantages** $A^\pi=Q^\pi-V^\pi$.
- **Off-policy PG** uses importance sampling. Full trajectory weights have variance exponential in $T$. Per-step ratios with a trust region (policy close to the data policy) are the practical compromise.
- **Straight-through Gumbel-softmax** makes sampling a discrete action differentiable: one-hot in the forward pass, softmax gradient in the backward pass. With a differentiable critic or model after the action, the pathwise gradient replaces the noisy score-function estimator. The price is some bias.

## Where this shows up next

- [[Actor-Critic Algorithms]] (next) learn $V^\pi$ as a baseline, and [[Monte Carlo and Temporal-Difference Learning]] shows how to learn it well.
- Trust-region and proximal policy optimization (TRPO/PPO) formalize the "stay close to $\pi_\theta$" approximation (future lecture).
- [[Imitation Learning]] is the unweighted special case: maximum likelihood on expert data.
- [[Variational Autoencoders#Gradients of an expectation|Variational autoencoders]] face the same problem, the gradient of an expectation under a parameterized distribution, and compare the score-function estimator used here with the lower-variance reparameterization (pathwise) estimator.

<!-- References: footnote definitions below render as the numbered "References" list at the bottom of the page -->

[^williams1992]: Williams, R. J. (1992). *Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning*. Machine Learning, 8, 229–256. [link](https://link.springer.com/article/10.1007/BF00992696)
[^sutton1999]: Sutton, R. S., McAllester, D., Singh, S., & Mansour, Y. (1999). *Policy Gradient Methods for Reinforcement Learning with Function Approximation*. NeurIPS. [link](https://papers.nips.cc/paper_files/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html)
[^weaver2001]: Weaver, L., & Tao, N. (2001). *The Optimal Reward Baseline for Gradient-Based Reinforcement Learning*. UAI. [arXiv:1301.2315](https://arxiv.org/abs/1301.2315)
[^schulman2015trpo]: Schulman, J., Levine, S., Moritz, P., Jordan, M., & Abbeel, P. (2015). *Trust Region Policy Optimization*. ICML. [arXiv:1502.05477](https://arxiv.org/abs/1502.05477)
[^schulman2016gae]: Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2016). *High-Dimensional Continuous Control Using Generalized Advantage Estimation*. ICLR. [arXiv:1506.02438](https://arxiv.org/abs/1506.02438)
[^suttonbarto]: Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.), Chapter 13. [book website](http://incompleteideas.net/book/the-book-2nd.html)
[^cs285]: Levine, S. *CS 285: Deep Reinforcement Learning*, UC Berkeley, Lectures 5–6 (Policy Gradients). [course page](https://rail.eecs.berkeley.edu/deeprlcourse/)
[^bengio2013]: Bengio, Y., Léonard, N., & Courville, A. (2013). *Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation*. [arXiv:1308.3432](https://arxiv.org/abs/1308.3432)
[^jang2017]: Jang, E., Gu, S., & Poole, B. (2017). *Categorical Reparameterization with Gumbel-Softmax*. ICLR. [arXiv:1611.01144](https://arxiv.org/abs/1611.01144)
[^maddison2017]: Maddison, C. J., Mnih, A., & Teh, Y. W. (2017). *The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables*. ICLR. [arXiv:1611.00712](https://arxiv.org/abs/1611.00712)
