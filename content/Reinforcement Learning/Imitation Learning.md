---
title: Imitation Learning
---

Let's say we have a policy $\pi^\beta$ (known as the behavior policy). We run $\pi^\beta$ on the environment and generate $\mathcal{D}$ (a dataset of experience tuples). Now, since each tuple is of the form $\{o_t,a_t,r_t,o_{t+1},d_t\}$ corresponding to the observation, action, reward, next observation, and done respectively at time-step $t$, we can easily extract a new dataset $\mathcal{D}'=\{o_t,a_t\}_{t=0}^{|\mathcal{D}|}$. Note that in the analogy of supervised learning, $\{o_t\}_t$ are the data features and $\{a_t\}_t$ are the targets. Since we consider the continuous state/action space, it might be helpful to think of the supervised learning task as a regression task.
**Note** that although in this part of the notes all the expressions use observations ($o_t$), however everything can be interchanged to states ($s_{t}$) without much modification.

### Behavior Cloning (BC)

For behavior cloning to work, we need to assume that the behavior policy $\pi^\beta$ is also the optimal (expert) policy $\pi^*$, since it doesn't make sense to imitate sub-optimal policies. So, given $\mathcal{D}'$ we need to learn a mapping $\pi_\theta:\mathcal{O}\to\mathcal{A}$, i.e. a policy (which can be modeled as a parametric distribution over actions conditioned on the observation) $\pi_\theta(a_t|o_t)$. This can be easily learned by using ERM (expected risk minimization), which is equivalent to maximizing the log likelihood of the data as follows:
$$
\max_\theta\mathbb{E}_{(o_t,a_t)\sim\mathcal{D}'}\log\pi_\theta(a_t|o_t)
$$
where the symbols have their usual meanings. In the regression setting, the ERM objective becomes:
$$
\theta^*=\arg\max_\theta\mathbb{E}_{(o_t,a_t)\sim\mathcal{D}'}[(\pi_\theta(o_t)-a_t)^2]
$$

---

### Aside (Cost Function and RL Objective)

We can now analyze the "goodness" of the policy $\pi_{\theta^*}$ obtained after ERM by inspecting the difference from $\pi^*$. Recall that the any MDP can be characterized by either a reward or a cost function. Assume the following cost function:
$$
c(o_{t},a_{t})=\begin{cases} 0 & a_{t} = \pi^*(o_{t}) \\ 1 & otherwise \end{cases}
$$
In that case the goal of any RL agent should be to minimize the cost (rather than the ERM):
$$
\theta^*=\arg\min_{\theta}\mathbb{E}_{o_{t}\sim p_{\pi_{\theta}}(o_t)}[c(o_{t},\pi_{\theta}(o_{t}))]
$$

**Note**: Although in ERM we are training without rewards, since the task is sequential decision making, we shall be using the above cost function to evaluate the performance of any trained policy.

---

### Analysis

Let's assume that for any $o_t$ (under the trained BC policy trained using ERM), the probability of incurring the cost (i.e. making error), i.e.:
$$
\pi_{\theta^{*}}(\hat{a}_{t}\neq a_{t}|o_t)\leq\epsilon;\forall{(o_{t},a_{t})\in\mathcal{D}'}
$$
where $\hat{a}_{t}\sim\pi_{\theta^{*}}(s_{t})$ and $\epsilon\geq0$ is a small constant bound. So, the upper bound on the total error for a trajectory run by taking actions according to a BC trained policy $\pi_{\theta^*}$ is:
$$
\mathbb{E}\left[\sum\limits_{t=0}^{T}c(o_{t},\hat{a}_{t})\right]\leq\epsilon T; \text{ only when all possible } o_{t}\in\mathcal{D}'...[1]
$$
However, that is not the case during execution of the policy. Assume that the agent transitions from $o_t\in\mathcal{D}'$ to $o_{t+1}\notin \mathcal{D}'$ after an erroneous decision. Since there is a distribution shift, all actions corresponding to time-steps $t+1$ onward until $T$ will be wrong and will incur a cost.

Let $p_{\theta^*}(o_t)$ be the distribution over observations at the $t^\text{th}$ time-step that is obtained when the trajectory is collected with the policy $\pi_{\theta^*}$:
$$
p_{\theta^*}(o_t)=(1-\epsilon)^{t}p_{train}(o_{t})+(1-(1-\epsilon)^{t})p_{mistake}(o_{t})
$$
where $p_{train}(o_t)$ can be thought of as the distribution over observations when the trajectory is collected using $\pi^\beta$ ($\pi^*$ in our case), i.e. the same distribution from which $\mathcal{D}'$ was generated, and similarly $p_{mistake}$ is the distribution over observations after the first mistake is made. We are interested in finding out how far off is the distribution of observations under the current policy (trained using the dataset $\mathcal{D}'$) vs. the behavior policy (that collected the training dataset). Manipulating the above expression we get:
$$
\begin{align}
p_{\theta^*}(o_t)&=(1-\epsilon)^{t}p_{train}(o_{t})+(1-(1-\epsilon)^{t})p_{mistake}(o_{t})\\
&=(1-\epsilon)^{t}p_{train}(o_{t})-(1-\epsilon)^{t}p_{mistake}(o_{t})+p_{mistake}(o_{t})\\
&=p_{mistake}(o_{t})+(1-\epsilon)^{t}(p_{train}(o_{t})-p_{mistake}(o_{t}))\\
\implies-p_{\theta^*}(o_t)&=-p_{mistake}(o_{t})-(1-\epsilon)^{t}(p_{train}(o_{t})-p_{mistake}(o_{t}))\\
\implies p_{train}(o_t)-p_{\theta^*}(o_t)&=(p_{train}(o_t)-p_{mistake}(o_{t}))-(1-\epsilon)^{t}(p_{train}(o_{t})-p_{mistake}(o_{t}))\\
&=(1-(1-\epsilon)^{t})(p_{train}(o_{t})-p_{mistake}(o_{t}))\\
\end{align}
$$
However, since we are only interested in the magnitude of the difference, and not the sign, we can upper-bound the variational divergence as follows:
$$
\begin{align}
\implies|p_{train}(o_t)-p_{\theta^*}(o_t)|&=(1-(1-\epsilon)^{t})\cdot|p_{train}(o_{t})-p_{mistake}(o_{t})|\\
&\leq(1-(1-\epsilon)^{t})\cdot2 \text{ (using the identity $|a-b|<|a|+|b|$ \& $\max\mathbb{P}[\cdot]=1$)}\\
&\leq2\epsilon t \text{ (using the identity $(1-\epsilon)^{t}\geq1-\epsilon t$)}
\end{align}
$$
Thus the difference between the distribution of observations at time-step $t$ obtained by learning the optimal policy and the learned policy is upper-bounded by $2\epsilon t$. Hence, the expected cost over the trajectory can be calculated as follows:
$$
\begin{align}\mathbb{E}_{o_{t}\sim p_{\theta^{*}}(\cdot),\hat{a}_{t}\sim\pi_{\theta^{*}}(\cdot|o_t)}\left[\sum\limits_{t=0}^{T}c(o_{t},\hat{a}_{t})\right]&=\sum\limits_{t=0}^{T}\mathbb{E}_{o_{t},\hat{a}_{t}}\left[c(o_{t},\hat{a}_{t})\right] \text{ (linearity of expectations)}\\
&=\sum\limits_{t=0}^{T}\sum\limits_{o_{t}} p_{\theta^{*}}(o_{t})\cdot c(o_{t},\hat{a}_{t})\\
&=\sum\limits_{t=0}^{T}\sum\limits_{o_{t}} (p_{train}(o_{t})+p_{\theta^{*}}(o_{t})-p_{train}(o_{t}))\cdot c(o_{t},\hat{a}_{t})\\
&=\sum\limits_{t=0}^{T}\sum\limits_{o_{t}} p_{train}(o_{t})\cdot c(o_{t},\hat{a}_{t})+(p_{\theta^{*}}(o_{t})-p_{train}(o_{t}))\cdot c(o_{t},\hat{a}_{t})\\
&=\sum\limits_{t=0}^{T}\left(\mathbb{E}_{o_{t},\hat{a}_{t}}\left[c(o_{t},\hat{a}_{t})\right]+\sum\limits_{o_{t}}(p_{\theta^{*}}(o_{t})-p_{train}(o_{t}))\cdot c(o_{t},\hat{a}_{t})\right)\\
&\leq\sum\limits_{t=0}^{T}\left(\mathbb{E}_{o_{t},\hat{a}_{t}}\left[c(o_{t},\hat{a}_{t})\right]+\sum\limits_{o_{t}}|p_{\theta^{*}}(o_{t})-p_{train}(o_{t})|\cdot c_\text{max}\right)\\
&\leq\sum\limits_{t=0}^{T} \epsilon + 2\epsilon t \text{ ($c_\text{max}$=1 as defined above)}\\
&=\epsilon\sum\limits_{t=0}^{T}1+2\epsilon\sum\limits_{t=0}^{T}t=\epsilon T+2\epsilon\frac{T(T+1)}{2}\\
&=\epsilon T+\epsilon T(T+1)=2\epsilon T + \epsilon T^{2}\\
\therefore \mathbb{E}_{o_{t}\sim p_{\theta^{*}}(\cdot),\hat{a}_{t}\sim\pi_{\theta^{*}}(\cdot|o_t)}\left[\sum\limits_{t=0}^{T}c(o_{t},\hat{a}_{t})\right] &\in \mathcal{O}(T^{2})
\end{align}
$$
here $\mathcal{O}$ is the Big-Oh Notation, not the observation space.

This problem can be addressed in a the following ways:

- Be smart about data collection (and augmentation)
- Use very powerful models that make very few mistakes
- Use multi-task learning
- Change the algorithm (DAgger)

---

### DAgger (Dataset Aggregation)

GOAL: Collect data from $p_{\pi_{\theta}}(o_t)$ instead of $p_{\pi^*}(o_t)$
While $\pi_{\theta}\neq\pi^*$ for the support $o_t$ of $\pi^*$:

    1. Train $\pi_{\theta}(\hat{a}_{t}|o_{t})$ from dataset $\mathcal{D}'=\{o_{1},a_{1},...,o_{N},a_{N}\}$
    2. Run $\pi_{\theta}(\hat{a}_{t}|o_{t})$ to get dataset $\mathcal{D}_{\pi}=\{o_{1},...,o_{M}\}$
    3. Use $\pi^*$ to get $a_{t}$ $\forall o_{t}\in\mathcal{D}_{\pi}$
    4. Aggregate $\mathcal{D}'\gets \mathcal{D}'\cup \mathcal{D}_{\pi}$

Since the dataset will eventually get populated with the entire support of $\pi^*$, the error will be upper-bounded by $\epsilon T$ as mentioned in $[1]$, i.e. all possible $o_{t}$ will be in the augmented dataset.
