---
title: Imitation Learning
tags:
  - reinforcement-learning
  - imitation-learning
  - behavior-cloning
  - dagger
prev: "[[Introduction to Reinforcement Learning]]"
next: "[[Policy Gradients]]"
---

> **Prerequisites:** [[Introduction to Reinforcement Learning]] (MDPs, POMDPs, cost vs. reward), maximum likelihood estimation ([[Linear Models]]).
>
> **TL;DR:** Behavior cloning turns control into supervised learning: regress the expert's actions on the observations. It is simple, but a small per-step error $\epsilon$ compounds into an expected trajectory cost of $O(\epsilon T^2)$, because the learner drifts into states the expert never showed it. DAgger fixes this by collecting data under the learner's *own* state distribution and asking the expert to label it, which brings the cost back to $O(\epsilon T)$.

## Motivation

Suppose an expert can already do the task: a human driving a car, a teleoperator moving a robot arm. The most direct way to get a policy is to *copy* the expert, with no reward function and no trial and error. This is **imitation learning**. It is the simplest bridge between supervised learning and sequential decision making, and seeing exactly where it breaks tells us a lot about why RL is hard. The presentation here loosely follows CS285.[^cs285]

## Setup

Say we have a policy $\pi^\beta$, known as the **behavior policy**. We run $\pi^\beta$ in the environment and record a dataset $\mathcal{D}$ of experience tuples $(o_t,a_t,r_t,o_{t+1},d_t)$: observation, action, reward, next observation and done flag at time-step $t$. From it we keep only the observation–action pairs:

$$
\mathcal{D}'=\{(o_i,a_i)\}_{i=1}^{N}.
$$

By analogy with supervised learning, the $o_i$ are the input features and the $a_i$ are the targets. For continuous action spaces it helps to think of this as a **regression** task, and for discrete actions as **classification**.

> [!note]
> We write everything with observations $o_t$, because in general the agent sees a partial observation of the state ([[Introduction to Reinforcement Learning|POMDP]]). Everything below goes through with $o_t$ replaced by $s_t$.

## Behavior Cloning (BC)

For behavior cloning to make sense, we assume the behavior policy is the **expert**: $\pi^\beta=\pi^*$. There is little point in imitating a sub-optimal policy. Given $\mathcal{D}'$, we learn a policy $\pi_\theta(a_t\mid o_t)$, a parametric distribution over actions conditioned on the observation, by **empirical risk minimization** (ERM). With the negative log-likelihood as the loss, ERM is maximum likelihood:

$$
\theta^*=\arg\max_\theta\;\mathbb{E}_{(o_t,a_t)\sim\mathcal{D}'}\big[\log\pi_\theta(a_t\mid o_t)\big].
$$

In the regression setting, take a Gaussian policy $\pi_\theta(a\mid o)=\mathcal{N}\big(\mu_\theta(o),\sigma^2 I\big)$. Then $\log\pi_\theta(a\mid o)=-\frac{1}{2\sigma^2}\lVert a-\mu_\theta(o)\rVert^2+\text{const}$, and maximum likelihood becomes ordinary least squares on the actions:

$$
\theta^*=\arg\min_\theta\;\mathbb{E}_{(o_t,a_t)\sim\mathcal{D}'}\Big[\lVert\mu_\theta(o_t)-a_t\rVert^2\Big].
$$

This is the same MLE–MSE equivalence as in [[Linear Models]], with "features → targets" replaced by "observations → actions".

> [!example]- Code: the behavior-cloning loss, discrete and continuous (PyTorch)
> ```python
> # assumes: a batch of expert pairs obs (B, d_o) and
> #   discrete:   logits_net(obs) -> (B, K) logits, expert actions a_disc (B,) integers
> #   continuous: mean_net(obs) -> (B, d_a) Gaussian mean mu_theta(o), expert actions a_cont (B, d_a)
> loss_disc = F.cross_entropy(logits_net(obs), a_disc)     # = -mean log pi_theta(a|o): classification
> loss_cont = F.mse_loss(mean_net(obs), a_cont)            # = -mean log N(a; mu_theta(o), sigma^2 I),
>                                                          #   up to the factor 2 sigma^2 / d_a and a constant
> ```
> Either loss is then minimized with any optimizer, exactly as in supervised learning. `version1.py` of the [Policy-Gradients](https://github.com/parag1604/Policy-Gradients) repository is this classifier; its `version2.py` turns it into REINFORCE by weighting each term by a return.

> [!warning] Two quieter failure modes
> - **Multimodal experts.** If the expert sometimes swerves left and sometimes right around an obstacle, the MSE-optimal prediction is the *average*: straight into the obstacle. Expressive policy classes (mixtures of Gaussians, discretized actions, diffusion or flow policies) avoid this.
> - **Non-Markovian experts.** A human's action depends on their history, not just the current observation. A policy $\pi_\theta(a_t\mid o_t)$ may be unable to represent it. Conditioning on a short history of observations helps.

---

## Aside: a cost function for imitation

To judge how *good* the cloned policy $\pi_{\theta^*}$ is, we compare it with $\pi^*$ inside the sequential task. Recall from [[Introduction to Reinforcement Learning]] that an MDP can be specified with a reward or, equivalently, a cost $c=-r$. Take the 0–1 cost "did we deviate from the expert?":

$$
c(o_t,a_t)=\begin{cases}0 & a_t=\pi^*(o_t)\\ 1 & \text{otherwise.}\end{cases}
$$

What we actually care about is the cost accumulated over a horizon of $T$ steps when the *learned* policy is in control. That means observations come from the learned policy's own distribution $p_{\pi_\theta}(o_t)$:

$$
\min_\theta\;\sum_{t=0}^{T-1}\mathbb{E}_{o_t\sim p_{\pi_\theta}(o_t)}\Big[\mathbb{E}_{a_t\sim\pi_\theta(\cdot\mid o_t)}\big[c(o_t,a_t)\big]\Big].
$$

Compare this with what BC optimizes: a loss on observations drawn from the *expert's* distribution $p_{\text{train}}(o_t)$. The whole story of this lecture is the gap between these two distributions.

> [!note]
> BC is trained without any reward. The cost $c$ is only a tool for *evaluating* the trained policy in the sequential setting.

---

## Analysis: why errors compound

**Assumption (BC succeeds as supervised learning).** On the training distribution, the learned policy errs with probability at most $\epsilon$:

$$
\pi_{\theta^*}\big(a\neq\pi^*(o)\,\big|\,o\big)\le\epsilon\qquad\text{for all } o\sim p_{\text{train}}(o).
$$

If the learner always stayed on the expert's distribution, the expected cost over $T$ steps would be at most $\epsilon T$. That is the best we can hope for. But the learner does *not* stay there.

![[Reinforcement Learning/images/imitation-learning-drift.svg]]
*Figure 1: the expert's trajectories (blue) define the training distribution (shaded tube). One mistake moves the learner (orange) slightly outside the tube. There it has no data, its errors become more likely, and it drifts further. This is covariate shift that the policy causes itself.*

**Worst-case model of a mistake.** As long as the learner has made no mistake, it follows the expert, so $o_t\sim p_{\text{train}}$. After the first mistake, it lands in unfamiliar observations with some distribution $p_{\text{mistake}}$, and we pessimistically assume it never recovers. The probability of no mistake in the first $t$ steps is at least $(1-\epsilon)^t$, so the learner's observation distribution at step $t$ is the mixture

$$
p_{\theta^*}(o_t)=(1-\epsilon)^t\,p_{\text{train}}(o_t)+\big(1-(1-\epsilon)^t\big)\,p_{\text{mistake}}(o_t).
$$

**How far is $p_{\theta^*}$ from $p_{\text{train}}$?** Rearranging,

$$
\begin{aligned}
p_{\theta^*}(o_t)&=p_{\text{mistake}}(o_t)+(1-\epsilon)^t\big(p_{\text{train}}(o_t)-p_{\text{mistake}}(o_t)\big)\\
\implies p_{\text{train}}(o_t)-p_{\theta^*}(o_t)&=\big(1-(1-\epsilon)^t\big)\big(p_{\text{train}}(o_t)-p_{\text{mistake}}(o_t)\big).
\end{aligned}
$$

Only the size of the gap matters, not its sign, so we measure it with the **total variation** distance, summed over all observations:

$$
\begin{aligned}
\sum_{o_t}\big|p_{\text{train}}(o_t)-p_{\theta^*}(o_t)\big|
&=\big(1-(1-\epsilon)^t\big)\sum_{o_t}\big|p_{\text{train}}(o_t)-p_{\text{mistake}}(o_t)\big|\\
&\le 2\big(1-(1-\epsilon)^t\big)\\
&\le 2\epsilon t.
\end{aligned}
$$

The first inequality holds because $\sum_o|p(o)-q(o)|\le\sum_o p(o)+\sum_o q(o)=2$ for any two distributions. The second uses Bernoulli's inequality, $(1-\epsilon)^t\ge1-\epsilon t$.

So the learner's distribution drifts away from the training distribution *linearly in time*.

**Bounding the expected cost.** Write $\bar c_t(o)=\mathbb{E}_{a\sim\pi_{\theta^*}(\cdot\mid o)}[c(o,a)]\in[0,1]$ for the probability of erring at observation $o$. By linearity of expectation,

$$
\begin{aligned}
\mathbb{E}\left[\sum_{t=0}^{T-1}c(o_t,a_t)\right]
&=\sum_{t=0}^{T-1}\sum_{o_t}p_{\theta^*}(o_t)\,\bar c_t(o_t)\\
&=\sum_{t=0}^{T-1}\Big[\sum_{o_t}p_{\text{train}}(o_t)\,\bar c_t(o_t)\\
&\qquad\quad+\sum_{o_t}\big(p_{\theta^*}(o_t)-p_{\text{train}}(o_t)\big)\,\bar c_t(o_t)\Big]\\
&\le\sum_{t=0}^{T-1}\Bigg[\underbrace{\mathbb{E}_{o_t\sim p_{\text{train}}}\big[\bar c_t(o_t)\big]}_{\le\,\epsilon\ \text{(assumption)}}\\
&\qquad\quad+\underbrace{\sum_{o_t}\big|p_{\theta^*}(o_t)-p_{\text{train}}(o_t)\big|}_{\le\,2\epsilon t}\cdot\underbrace{\max_o\bar c_t(o)}_{\le 1}\Bigg]\\
&\le\sum_{t=0}^{T-1}\big(\epsilon+2\epsilon t\big)=\epsilon T+2\epsilon\frac{T(T-1)}{2}=\epsilon T^2.
\end{aligned}
$$

$$
\boxed{\;\mathbb{E}\left[\sum_{t=0}^{T-1}c(o_t,a_t)\right]\le\epsilon T^2=O(\epsilon T^2)\;}
$$

(The $O$ here is big-O notation, not the observation space $\mathcal{O}$.) A per-step error that would cost $\epsilon T$ if the data distribution held still costs up to $\epsilon T^2$, because the learner's own mistakes move the distribution. This is the classic result of Ross & Bagnell.[^ross2010]

![[Reinforcement Learning/images/imitation-learning-cost-growth.svg]]
*Figure 2: expected trajectory cost against horizon $T$, with $\epsilon=0.01$. The orange curve is the exact expected cost under the "never recover" model above, $\sum_{t}\big(1-(1-\epsilon)^{t+1}\big)$. It tracks the quadratic bound $\epsilon T^2$ while $\epsilon T\ll1$ and eventually saturates at one error per step. DAgger's $O(\epsilon T)$ line is shown for comparison.*

### What can we do about it?

- **Be smart about data collection and augmentation.** Show the learner how to *recover*. The classic example is NVIDIA's self-driving car, which added side-facing cameras labelled with corrective steering, so the training data contained "slightly off the road → steer back" examples.[^bojarski2016] ALVINN used a similar trick decades earlier.[^pomerleau1988]
- **Use very powerful models that make very few mistakes.** Driving $\epsilon$ down shrinks $\epsilon T^2$, provided the model can also handle multimodal and non-Markovian behavior (see the warning above).
- **Use multi-task learning.** Training on many related tasks or goals widens the training distribution, so the learner is less often off-distribution.
- **Change the algorithm.** Collect data from the learner's distribution directly: DAgger.

---

## DAgger (Dataset Aggregation)

**Goal:** make the training data come from $p_{\pi_\theta}(o_t)$, the distribution the *learner* visits, instead of $p_{\pi^*}(o_t)$. We cannot change the distribution the policy induces, but we *can* change where the labels come from: run the learner, then ask the expert what it *should* have done.[^ross2011]

> [!example] Algorithm: DAgger
> **Input:** an initial expert dataset $\mathcal{D}'=\{(o_i,a_i)\}_{i=1}^N$, and an expert $\pi^*$ that can be queried.
> **For** $k=1,\dots,K$:
> 1. Train $\pi_\theta(a_t\mid o_t)$ on $\mathcal{D}'$ (behavior cloning).
> 2. Run $\pi_\theta$ in the environment to collect observations $\mathcal{D}_\pi=\{o_1,\dots,o_M\}$.
> 3. Ask the expert to label them: $a_j=\pi^*(o_j)$ for all $o_j\in\mathcal{D}_\pi$.
> 4. Aggregate: $\mathcal{D}'\leftarrow\mathcal{D}'\cup\{(o_j,a_j)\}_{j=1}^M$.
>
> **Return** the best policy found.

> [!example]- Code: the DAgger loop (Python)
> ```python
> # assumes: fit(data) -> policy (behavior cloning on a list of (o, a) pairs),
> # rollout(policy) -> list of observations visited when the LEARNER acts, expert(o) -> pi*(o),
> # data = the initial expert dataset D' as a list of (o, a) pairs
> policies = []
> for k in range(K):
>     policy = fit(data)                                  # 1. train on everything collected so far
>     visited = rollout(policy)                           # 2. the learner decides where we go
>     data += [(o, expert(o)) for o in visited]           # 3-4. the expert only labels; aggregate
>     policies.append(policy)
> ```
> Plain behavior cloning is the first iteration alone. "Return the best policy found" means evaluating the policies in `policies` and keeping the best.

![[Reinforcement Learning/images/imitation-learning-dagger-loop.svg]]
*Figure 3: the DAgger loop. The learner chooses which observations get visited; the expert only supplies the labels.*

**Why it works.** After enough iterations, the aggregated dataset contains observations from the distribution the learner actually visits. The training error $\epsilon$ is then measured *under the learner's own distribution*, so the mismatch term in the analysis disappears. Formally, Ross, Gordon & Bagnell cast DAgger as **no-regret online learning**: each iteration is a round, and its loss is the imitation loss on the learner's current state distribution. They show that the expected cost grows only **linearly**, $O(\epsilon T)$, where $\epsilon$ is now the best achievable error on the aggregated data.[^ross2011] The original algorithm also runs a mixture $\beta_k\pi^*+(1-\beta_k)\pi_\theta$ in step 2, with $\beta_k\to0$. Using the pure learner ($\beta_k=0$ for $k\ge2$) is the common simplification shown above.

> [!warning] The catch
> Step 3 needs an expert that can be queried on *arbitrary* observations. That is easy for a planner or a scripted controller, but awkward for humans: labelling video frames offline is unnatural, because people act well in a closed loop, not frame by frame.

## Key takeaways

- **Behavior cloning** is supervised learning of $\pi_\theta(a\mid o)$ from expert data. With a Gaussian policy, maximum likelihood is the same as MSE regression on actions.
- **Covariate shift:** the learner's own mistakes take it to observations the expert never visited. A per-step error $\epsilon$ then compounds into an expected trajectory cost of $O(\epsilon T^2)$.
- **Fixes:** data that shows recovery, better models (lower $\epsilon$, multimodal and history-aware), multi-task data, or **DAgger**, which labels the learner's own visited observations and recovers $O(\epsilon T)$.
- The *source of the data distribution* is central to RL. It comes back in [[Introduction to Reinforcement Learning#On-policy vs Off-policy Algorithm|on- vs off-policy learning]] and in importance sampling for [[Policy Gradients#Off-policy Policy Gradients|off-policy policy gradients]].

## Where this shows up next

- [[Unsupervised Domain Adaptation]]: the distribution drift of behaviour cloning is a covariate shift, the same problem studied there for classifiers, with importance weighting and its limits.
- [[Policy Gradients]]: the policy-gradient estimator is the behavior-cloning (MLE) gradient with each trajectory's log-likelihood **weighted by its return**. Imitation copies every action; RL copies the good ones more.
- [[Model-Based Reinforcement Learning]]: a learned dynamics model suffers the same distribution shift as behavior cloning: errors compound when the planner visits states the model was not trained on.

<!-- References: footnote definitions below render as the numbered "References" list at the bottom of the page -->

[^pomerleau1988]: Pomerleau, D. A. (1988). *ALVINN: An Autonomous Land Vehicle in a Neural Network*. NeurIPS. [link](https://papers.nips.cc/paper_files/paper/1988/hash/812b4ba287f5ee0bc9d43bbf5bbe87fb-Abstract.html)
[^ross2010]: Ross, S., & Bagnell, J. A. (2010). *Efficient Reductions for Imitation Learning*. AISTATS. [link](https://proceedings.mlr.press/v9/ross10a.html)
[^ross2011]: Ross, S., Gordon, G. J., & Bagnell, J. A. (2011). *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning* (DAgger). AISTATS. [arXiv:1011.0686](https://arxiv.org/abs/1011.0686)
[^bojarski2016]: Bojarski, M., et al. (2016). *End to End Learning for Self-Driving Cars*. [arXiv:1604.07316](https://arxiv.org/abs/1604.07316)
[^cs285]: Levine, S. *CS 285: Deep Reinforcement Learning*, UC Berkeley, Lecture 2 (Imitation Learning). [course page](https://rail.eecs.berkeley.edu/deeprlcourse/)
