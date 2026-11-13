---
title: Model-Based Reinforcement Learning
tags:
  - reinforcement-learning
  - model-based-rl
  - dyna
  - model-predictive-control
  - planning
date: 2026-11-12
draft: false
prev: "[[Stochastic Approximation]]"
next: "[[Index (Reinforcement Learning)]]"
---

> **Prerequisites:** [[Dynamic Programming]] (value iteration, policy iteration, the Bellman operators), [[Q-Learning and SARSA]] (for Dyna-Q), [[Imitation Learning]] (distribution shift and compounding errors). The exploration discussion uses UCB from [[Multi-Armed Bandits]].
>
> **TL;DR:** Every planning method of [[Dynamic Programming]] and its neighbours needs the transition model $p(s'\mid s,a)$ and the reward $r(s,a)$. When they are unknown we can **learn them from data and then plan**. In the tabular case the count-based estimate plus value iteration ("certainty equivalence") is near-optimal once every state-action pair has been sampled enough, and the **simulation lemma** says how model errors become value errors: they are amplified by $1/(1-\gamma)^2$. **Dyna** interleaves real steps, model learning and planning on simulated steps, so each real transition is used many times. With continuous states we fit a dynamics model by regression and plan with **model-predictive control** (random shooting, the cross-entropy method), replanning at every step and using **ensembles** to stop the planner from exploiting model errors. Finally, the model can be used to train a policy: by backpropagating through it (unstable over long horizons) or by feeding short model rollouts to a model-free learner (MBPO).

> [!note] Local notation
> $\hat p$, $\hat r$ and $\hat M$ are the estimated transition probabilities, reward and MDP. $N(s,a)$ counts visits of a state-action pair. A learned deterministic dynamics model is written $f_\phi(s,a)$. $H$ is the **planning horizon** of model-predictive control (the episode horizon is still $T$). The letter $n$ is used twice, in separate sections: the number of samples per state-action pair (certainty equivalence) and the number of Dyna planning updates per real step (as in Sutton & Barto).

## Motivation

The planning lectures ([[Finite-Horizon Dynamic Programming]], [[Dynamic Programming]], [[Stochastic Shortest Path Problems]]) assumed that someone hands us the MDP. With $p(s'\mid s,a)$ and $r(s,a)$ in hand, value iteration and policy iteration compute an optimal policy without a single interaction with the world. The first half of the series took the opposite stance: the model-free methods ([[Policy Gradients]], [[Q-Learning and SARSA]], ...) never write the model down at all.

There is a middle road, and it is often the most sample-efficient one: **learn the model from experience, then plan with it**. A transition $(s_t,a_t,r_t,s_{t+1})$ is a labelled example for a supervised problem ("what happens if I do $a$ in $s$?"). That supervision is far richer than the scalar return that model-free methods learn from. Levine's CS 285 lectures organise the subject into three questions,[^cs285] and this lecture follows the same arc:

1. **Learn a model, then plan with it.** How good does the model have to be? What goes wrong when it is not? (Sections on certainty equivalence and on planning with learned models.)
2. **Mix learning and planning.** Dyna uses the model to generate extra, simulated experience for an ordinary model-free learner.
3. **Learn a policy from the model.** Instead of replanning at every step, distil the model's predictions into a policy.

There are also many problems where the model *is* known, at least in principle: board games, simulators, systems that obey known physics. Then everything below reduces to [[Dynamic Programming|dynamic programming]], and the only question is computational.

## Certainty equivalence: estimate the MDP, then solve it

### The estimator

Take the tabular case: finite $\mathcal{S}$ and $\mathcal{A}$. Suppose we have observed $N(s,a)$ transitions out of each pair $(s,a)$, and let $N(s,a,s')$ count how many of them landed in $s'$. The maximum-likelihood model is the obvious one:

$$
\hat p(s'\mid s,a)=\frac{N(s,a,s')}{N(s,a)},\qquad \hat r(s,a)=\frac{1}{N(s,a)}\sum_{k=1}^{N(s,a)} r^{(k)}(s,a).
$$

Each row $\hat p(\cdot\mid s,a)$ is the empirical distribution of a categorical sample, and $\hat r(s,a)$ is a sample mean, exactly the estimator of an arm's value in [[Multi-Armed Bandits]]. The **certainty-equivalence** (CE) recipe then pretends the estimate is the truth: solve $\hat M=(\mathcal{S},\mathcal{A},\hat p,\hat r,\gamma)$ with [[Dynamic Programming#Value iteration|value iteration]] or [[Dynamic Programming#Policy iteration|policy iteration]], and act greedily with respect to its optimal value $\hat V^*$.

> [!note] Where the name comes from
> "Certainty equivalence" comes from control theory. For linear dynamics with additive noise and a quadratic objective, the optimal controller is the same as for the noise-free system: you may act as if the uncertain quantity were certain and equal to its mean.[^simon1956] In RL the phrase is used more loosely: act as if the *estimated* model were the true one. In general this is not optimal, because it ignores how uncertain the estimate is, and that is exactly where the exploration problem below comes from.

### How model errors become value errors: the simulation lemma

How accurate must $\hat p$ and $\hat r$ be? The answer is a short, very useful calculation.

> [!tip] Intuition
> A one-step error in $\hat p$ misplaces a fraction of probability mass, and the misplaced mass carries a value of size at most $R_{\max}/(1-\gamma)$. The error is made at every step, and in a discounted problem errors are summed over an effective horizon of $1/(1-\gamma)$ steps. One factor of $1/(1-\gamma)$ comes from the size of values, another from the accumulation, so transition errors are amplified by $1/(1-\gamma)^2$. This is the discounted cousin of the $\epsilon T^2$ compounding in [[Imitation Learning#Analysis: why errors compound|behaviour cloning]].

**Simulation lemma.** Let $M$ and $\hat M$ share $\mathcal{S}$, $\mathcal{A}$ and $\gamma\in[0,1)$, with $|r(s,a)|\le R_{\max}$. Suppose that for all $(s,a)$

$$
|r(s,a)-\hat r(s,a)|\le\epsilon_r,\qquad \big\lVert p(\cdot\mid s,a)-\hat p(\cdot\mid s,a)\big\rVert_1\le\epsilon_p .
$$

Then for every (stationary, possibly stochastic) policy $\pi$, writing $V^\pi$ and $\hat V^\pi$ for its values in $M$ and $\hat M$,

$$
\big\lVert V^\pi-\hat V^\pi\big\rVert_\infty\le\frac{\epsilon_r}{1-\gamma}+\frac{\gamma\,\epsilon_p\,R_{\max}}{(1-\gamma)^2}. \tag{1}
$$

> [!proof]- Proof
> Use the matrix form of the Bellman expectation equation from [[Introduction to Reinforcement Learning#Matrix form and closed-form policy evaluation|Introduction to RL]]: $V^\pi=r_\pi+\gamma P^\pi V^\pi$ and $\hat V^\pi=\hat r_\pi+\gamma\hat P^\pi\hat V^\pi$, where $r_\pi(s)=\sum_a\pi(a\mid s)r(s,a)$ and $P^\pi(s,s')=\sum_a\pi(a\mid s)p(s'\mid s,a)$ (hats for $\hat M$). Subtract, and add and subtract $\gamma\hat P^\pi V^\pi$:
>
> $$
> \begin{aligned}
> V^\pi-\hat V^\pi&=(r_\pi-\hat r_\pi)+\gamma(P^\pi-\hat P^\pi)V^\pi+\gamma\hat P^\pi\big(V^\pi-\hat V^\pi\big)\\
> \Longrightarrow\quad V^\pi-\hat V^\pi&=\big(I-\gamma\hat P^\pi\big)^{-1}\Big[(r_\pi-\hat r_\pi)+\gamma(P^\pi-\hat P^\pi)V^\pi\Big].
> \end{aligned}
> $$
>
> The inverse exists and $\lVert(I-\gamma\hat P^\pi)^{-1}\rVert_\infty=\lVert\sum_{k\ge0}\gamma^k(\hat P^\pi)^k\rVert_\infty\le\frac{1}{1-\gamma}$, because every power of a stochastic matrix is stochastic (rows sum to one).
>
> The reward term is at most $\epsilon_r$ in every component. For the transition term fix $(s,a)$ and write $\Delta=p(\cdot\mid s,a)-\hat p(\cdot\mid s,a)$. Both are distributions, so $\sum_{s'}\Delta(s')=0$ and we may subtract any constant $c$ from $V^\pi$:
>
> $$
> \Big|\sum_{s'}\Delta(s')V^\pi(s')\Big|=\Big|\sum_{s'}\Delta(s')\big(V^\pi(s')-c\big)\Big|\le\lVert\Delta\rVert_1\,\max_{s'}\big|V^\pi(s')-c\big|.
> $$
>
> Since $|V^\pi|\le R_{\max}/(1-\gamma)$, choosing $c=0$ gives at most $\epsilon_p R_{\max}/(1-\gamma)$ (choosing $c$ as the midpoint of the range of $V^\pi$ gives the slightly sharper "half the span"). Averaging over $a\sim\pi(\cdot\mid s)$ keeps the bound. Combining,
>
> $$
> \lVert V^\pi-\hat V^\pi\rVert_\infty\le\frac{1}{1-\gamma}\Big(\epsilon_r+\frac{\gamma\,\epsilon_p R_{\max}}{1-\gamma}\Big).
> $$

What we actually care about is how good the **CE policy** $\hat\pi$ (greedy with respect to $\hat V^*$, hence optimal in $\hat M$) is in the real MDP. Call the right-hand side of (1) $\epsilon_V$. Then

$$
V^*-V^{\hat\pi}\le 2\,\epsilon_V\quad\text{componentwise.} \tag{2}
$$

> [!proof]- Proof
> Let $\pi^*$ be optimal in $M$. Split the gap into three pieces:
>
> $$
> V^{\pi^*}-V^{\hat\pi}=\underbrace{\big(V^{\pi^*}-\hat V^{\pi^*}\big)}_{\le\,\epsilon_V}+\underbrace{\big(\hat V^{\pi^*}-\hat V^{\hat\pi}\big)}_{\le\,0}+\underbrace{\big(\hat V^{\hat\pi}-V^{\hat\pi}\big)}_{\le\,\epsilon_V}.
> $$
>
> The outer terms are bounded by the simulation lemma applied to $\pi^*$ and to $\hat\pi$. The middle term is non-positive because $\hat\pi$ is optimal in $\hat M$, so $\hat V^{\hat\pi}=\hat V^*\ge\hat V^{\pi^*}$.

Note the structure of the argument: we only need the model to be accurate **for the two policies being compared**, not everywhere. That observation will matter later, when the model is only accurate where we have data.

### How many samples? A first sample-complexity estimate

Suppose we can query a **generative model** (a simulator that returns $s'\sim p(\cdot\mid s,a)$ for any $(s,a)$ we ask for), and we draw $n$ samples from every pair. How large must $n$ be for the CE policy to be $\varepsilon$-optimal?

The only new ingredient is a concentration bound for an empirical distribution in $\ell_1$.

**Lemma ($\ell_1$ deviation of an empirical distribution).** If $\hat p$ is the empirical distribution of $n$ i.i.d. draws from $p$ on $|\mathcal{S}|$ points, then with probability at least $1-\delta$

$$
\lVert\hat p-p\rVert_1\le\sqrt{\frac{2\big(|\mathcal{S}|\ln 2+\ln(1/\delta)\big)}{n}}.
$$

> [!proof]- Proof
> For any two distributions, $\lVert\hat p-p\rVert_1=2\max_{B\subseteq\mathcal{S}}\big(\hat p(B)-p(B)\big)$; the maximum is attained at $B=\{s':\hat p(s')>p(s')\}$. For a fixed set $B$, $\hat p(B)$ is the mean of $n$ i.i.d. Bernoulli$\big(p(B)\big)$ indicators, so Hoeffding's inequality (see [[Multi-Armed Bandits#Aside: concentration inequalities|the bandits lecture]]) gives $\Pr\big(\hat p(B)-p(B)\ge\varepsilon/2\big)\le e^{-2n(\varepsilon/2)^2}=e^{-n\varepsilon^2/2}$. A union bound over the at most $2^{|\mathcal{S}|}$ sets gives $\Pr\big(\lVert\hat p-p\rVert_1\ge\varepsilon\big)\le 2^{|\mathcal{S}|}e^{-n\varepsilon^2/2}$. Setting the right-hand side to $\delta$ and solving for $\varepsilon$ gives the claim.

Apply it to all $|\mathcal{S}||\mathcal{A}|$ rows (replace $\delta$ by $\delta/(|\mathcal{S}||\mathcal{A}|)$, another union bound) and, for simplicity, take the reward as known ($\epsilon_r=0$). By (2), the CE policy is $\varepsilon$-optimal as soon as $2\gamma\epsilon_pR_{\max}/(1-\gamma)^2\le\varepsilon$, which holds with probability $1-\delta$ once

$$
n\ \ge\ \frac{8\gamma^2R_{\max}^2}{\varepsilon^2(1-\gamma)^4}\Big(|\mathcal{S}|\ln2+\ln\frac{|\mathcal{S}||\mathcal{A}|}{\delta}\Big).
$$

So the total number of samples, $n|\mathcal{S}||\mathcal{A}|$, is of order $|\mathcal{S}|^2|\mathcal{A}|/\big(\varepsilon^2(1-\gamma)^4\big)$ up to logarithms. Two remarks keep this in perspective.

- It is polynomial in everything. That is the qualitative message: learning a model and planning is a sound way to learn.
- It is loose. Bounding *every* row in $\ell_1$ is wasteful: the proof of (1) only needs $(p-\hat p)^\top V$ to be small for one fixed vector $V$, which concentrates at a rate that does not grow with $|\mathcal{S}|$. A careful analysis of exactly this CE estimator shows that $\tilde O\big(|\mathcal{S}||\mathcal{A}|/(\varepsilon^2(1-\gamma)^3)\big)$ samples suffice to estimate $Q^*$ to accuracy $\varepsilon$, and that no method can do better.[^azar2013] Kearns and Singh showed earlier that this "indirect" (model-based) route and Q-learning have comparable rates.[^kearns1999]

![[Reinforcement Learning/images/model-based-rl-certainty-equivalence.svg]]
*Figure 1: certainty equivalence on the slippery $5\times5$ gridworld of [[Dynamic Programming]] ($\gamma=0.9$, mean of 30 runs per $n$). With $n$ sampled transitions per state-action pair, the value error $\lVert\hat V^*-V^*\rVert_\infty$ falls from 0.26 ($n=1$) to 0.018 ($n=1000$), roughly like $n^{-0.4}$, a little slower than the $1/\sqrt n$ reference line. The true loss of the CE policy, $\lVert V^*-V^{\hat\pi}\rVert_\infty$, falls faster (0.0016 at $n=1000$): a policy only needs the ranking of actions to be right. The bound (2), evaluated with the observed $\epsilon_r,\epsilon_p$, has the right $1/\sqrt n$ slope but is 3–4 orders of magnitude too large (12 at $n=1000$): the $(1-\gamma)^{-2}=100$ factor is a worst case.*

### Model bias and the need for exploration

The analysis above assumed a generative model: every $(s,a)$ is sampled $n$ times. An agent that learns online only sees the transitions its own behaviour produces. That breaks CE in a specific way.

Suppose the agent, early on, tries an action once and is unlucky. The model now says the action is bad, the planner never chooses it, the action is never tried again, and so the model is never corrected. The planner trusts the model exactly where the model has no reason to be trusted. This failure is often called **model bias**,[^pilco] and it is the exploration problem of [[Multi-Armed Bandits]] again, now spread over many states.

The bandit remedy carries over: be **optimistic in the face of uncertainty**. Plan in a model whose rewards are inflated where data are scarce, so that the planner *wants* to visit poorly-known pairs.

- **R-max**[^rmax] treats every pair visited fewer than $m$ times as leading to a fictitious state paying $R_{\max}$ forever; **E³**[^e3] explicitly balances exploring unknown states and exploiting known ones. Both were the first algorithms with polynomial sample-complexity guarantees for general MDPs.
- **MBIE-EB**[^mbie] adds a count-based bonus $\beta/\sqrt{N(s,a)}$ to $\hat r(s,a)$, the direct analogue of the [[Multi-Armed Bandits#Upper Confidence Bound (UCB)|UCB]] bonus.

![[Reinforcement Learning/images/model-based-rl-exploration.svg]]
*Figure 2: three certainty-equivalence agents on a six-state "river-swim" chain (20 runs of 5000 steps, $\gamma=0.95$; bands show the 10th–90th percentiles). Swimming left is easy and pays $0.005$ at the left bank; swimming right succeeds only with probability $0.35$ per step, and pays $1$ per step at the right bank. All three agents re-plan by value iteration on the count-based model after every step. The greedy agent never reached the right bank in any run (total reward 25). Adding $\varepsilon=0.1$ random actions helped in 1 run out of 20 (mean total 184). An optimistic agent with bonus $0.3/\sqrt{N(s,a)}$ (and unvisited pairs valued as $R_{\max}/(1-\gamma)$) reached it in all 20 runs, collecting about 4000.*

Random ($\varepsilon$-greedy) exploration fails here for the same reason it is weak in bandits, only worse: to see the reward, the agent must take the "unpromising" action several times *in a row*, and the probability of doing so by chance decays exponentially with the length of the chain.

## When the model is exact: DP, and LQR as the closed-form case

With a perfect model there is nothing left to learn, and certainty equivalence is just planning: everything in [[Dynamic Programming]] applies. One classical case deserves a name, because there the planning problem has a **closed-form** solution and it underlies much of continuous control: the **linear-quadratic regulator** (LQR).

Take a finite horizon $T$ (the setting of [[Finite-Horizon Dynamic Programming]]), continuous states $s\in\mathbb{R}^{d_s}$ and actions $a\in\mathbb{R}^{d_a}$, linear dynamics with additive noise, and a negative quadratic reward:

$$
s_{t+1}=\mathbf{A}s_t+\mathbf{B}a_t+w_t,\quad w_t\sim\mathcal N(0,\boldsymbol\Sigma_w),\qquad r(s,a)=-s^\top\mathbf{C}_s s-a^\top\mathbf{C}_a a,
$$

with $\mathbf{C}_s\succeq0$ and $\mathbf{C}_a\succ0$. Run the [[Finite-Horizon Dynamic Programming#The dynamic programming algorithm|finite-horizon backward recursion]] with $V_T\equiv0$. Guess that $V_{t+1}(s)=-s^\top\mathbf{P}_{t+1}s-c_{t+1}$ with $\mathbf{P}_{t+1}\succeq0$. Then $V_t(s)=\max_a\{r(s,a)+\mathbb{E}[V_{t+1}(\mathbf{A}s+\mathbf{B}a+w)]\}$ is the maximum of a concave quadratic in $a$. Setting its gradient to zero gives a **linear** optimal policy and keeps the guess quadratic:

$$
\begin{aligned}
\mu_t(s)&=-\mathbf{K}_ts,\qquad \mathbf{K}_t=\big(\mathbf{C}_a+\mathbf{B}^\top\mathbf{P}_{t+1}\mathbf{B}\big)^{-1}\mathbf{B}^\top\mathbf{P}_{t+1}\mathbf{A},\\
\mathbf{P}_t&=\mathbf{C}_s+\mathbf{A}^\top\mathbf{P}_{t+1}\mathbf{A}-\mathbf{A}^\top\mathbf{P}_{t+1}\mathbf{B}\,\mathbf{K}_t,\\
c_t&=c_{t+1}+\operatorname{tr}\big(\mathbf{P}_{t+1}\boldsymbol\Sigma_w\big),
\end{aligned}
$$

starting from $\mathbf{P}_T=0$, $c_T=0$. This is the discrete-time **Riccati recursion**. The noise covariance appears only in the constant $c_t$; the gains $\mathbf{K}_t$ are the same as for the noise-free system. That is the original certainty-equivalence principle.[^simon1956]

LQR matters for model-based RL in two ways. First, it is the simplest instance of "learn a model, then plan": fit $\mathbf{A},\mathbf{B}$ by least squares on observed transitions (system identification) and run the Riccati recursion. Second, it is the inner loop of the standard trajectory optimisers for nonlinear models: **iLQR / DDP** repeatedly linearise the dynamics and quadratise the reward around the current trajectory, solve the resulting LQR, and update the trajectory.[^tassa2012]

## Dyna: learning, planning and acting in one loop

Certainty equivalence separates the phases: gather data, fit, then plan to convergence. That is wasteful when data arrive one transition at a time. Sutton's **Dyna** architecture[^sutton1990]<sup>,</sup>[^sutton1991] interleaves three processes, all running at every time step:

1. **Acting and direct RL.** Take a real step and apply an ordinary model-free update (in Dyna-Q, a [[Q-Learning and SARSA|Q-learning]] update) to the real transition.
2. **Model learning.** Record the transition in the model.
3. **Planning.** Make $n$ further Q-learning updates on **simulated** transitions, generated by querying the model at previously seen state-action pairs.

The planning updates are the same update rule applied to imagined data. So planning, in Dyna, is just learning from simulated experience, and with $n\to\infty$ planning updates between real steps, $Q$ converges to the optimal $Q$-function of the current model, just as [[Dynamic Programming#Asynchronous DP and the curse of dimensionality|asynchronous value iteration]] would.

> [!example] Algorithm: tabular Dyna-Q (deterministic environment)
> Inputs: step size $\alpha$, exploration rate $\varepsilon$, planning steps $n$.
> 1. Initialise $Q(s,a)=0$ for all $(s,a)$ and an empty model.
> 2. Observe the initial state $s$.
> 3. Choose $a$ $\varepsilon$-greedily from $Q(s,\cdot)$, take it, observe $r$ and $s'$.
> 4. **Direct RL:** $Q(s,a)\leftarrow Q(s,a)+\alpha\big[r+\gamma\max_{a'}Q(s',a')-Q(s,a)\big]$, with the $\max$ term set to $0$ if $s'$ is terminal.
> 5. **Model learning:** store $\mathrm{Model}(s,a)\leftarrow(r,s')$.
> 6. **Planning:** repeat $n$ times: pick a previously observed pair $(\tilde s,\tilde a)$ uniformly at random, read $(\tilde r,\tilde s')=\mathrm{Model}(\tilde s,\tilde a)$, and apply the update of step 4 to $(\tilde s,\tilde a,\tilde r,\tilde s')$.
> 7. Set $s\leftarrow s'$ (or reset if the episode ended) and go to 3.

For a stochastic environment, the model stores counts instead of a single outcome, and step 6 samples $\tilde s'\sim\hat p(\cdot\mid\tilde s,\tilde a)$ (or uses the full expected update $\sum_{s'}\hat p(s'\mid\tilde s,\tilde a)[\dots]$).

![[Reinforcement Learning/images/model-based-rl-dyna-maze.svg]]
*Figure 3: Dyna-Q on a $7\times10$ maze (deterministic moves; reward $+1$ on reaching G, otherwise $0$; $\gamma=0.95$, $\alpha=0.1$, $\varepsilon=0.1$; mean of 30 runs; episode 1 is omitted because it is a random walk for every $n$). With $n=0$ (plain Q-learning) the agent needs on average 32 episodes before its episodes are within 1.5 times the 16-step shortest path; with $n=5$ it needs 6, and with $n=50$ only 3. In episode 2, the mean episode lengths are 458, 252 and 48 steps. Over 50 episodes, Q-learning uses 8692 real steps on average, against 1919 ($n=5$) and 1769 ($n=50$).*

Why does episode 1 not benefit? Until the goal has been found once, every stored reward is $0$ and every $Q$ is $0$, so the planning updates change nothing. As soon as the goal is reached, the next real step and its $n$ planning updates start pushing value backwards. With Q-learning alone the value moves back one state per visit; with $n=50$ planning updates per step it spreads along the whole remembered path within a few real steps.

> [!note] Refinements
> - **Prioritised sweeping**[^moore1993] replaces the uniform choice in step 6 with a priority queue: update first the pairs whose predecessors' values have just changed the most. It plans backwards from where the news is.
> - **Changing environments.** A model that is wrong in a way that makes things look worse than they are is corrected by experience; one that makes things look *better* than they are can be ignored forever if the agent stops visiting. Sutton & Barto's Dyna-Q+ adds a small exploration bonus that grows with the time since a pair was last tried.[^suttonbarto]

## Planning with learned models in continuous spaces

Tables do not scale to continuous states, but the recipe does. This is the setting of most of CS 285's model-based lectures.[^cs285]

### Learning the dynamics

A transition dataset $\mathcal{D}=\{(s_i,a_i,s_i')\}$ defines a regression problem. A **deterministic** model is trained by least squares, usually on the state *change*:

$$
\hat s'=s+f_\phi(s,a),\qquad \min_\phi\ \sum_{i}\big\lVert s_i'-s_i-f_\phi(s_i,a_i)\big\rVert^2 .
$$

A **probabilistic** model outputs a distribution, typically $\mathcal{N}\big(s+\boldsymbol\mu_\phi(s,a),\boldsymbol\Sigma_\phi(s,a)\big)$, trained by maximum likelihood. The reward is learned the same way when it is unknown. Two kinds of uncertainty matter here, and the distinction is the crux of the CS 285 treatment:

- **Aleatoric** uncertainty is noise in the system itself. A probabilistic model captures it with $\boldsymbol\Sigma_\phi$, and more data do not remove it.
- **Epistemic** uncertainty is our ignorance about the model, large wherever data are scarce. A single network, even a probabilistic one, does not represent it: far from the data, it simply extrapolates, often confidently.

A cheap and effective handle on epistemic uncertainty is a **bootstrap ensemble**:[^ensembles] train $B$ models on resampled versions of $\mathcal{D}$ (or simply from different random initialisations). Where data are plentiful they agree; where data are absent they disagree. **PETS** (probabilistic ensembles with trajectory sampling)[^pets] combines both: an ensemble of probabilistic networks, with each imagined trajectory propagated through the members.

### Why "fit once, then plan" fails

The naive procedure is: run some base policy (for example random actions), fit $f_\phi$ to the data, plan with it. This is how classical system identification works, and it works well when the model has a few physical parameters. With an expressive model it usually fails, for a reason we have met before. The model is accurate under the **state distribution of the data-collection policy**. The planner then steers into states the base policy never visited, where the model is wrong, and the planner's decisions there are based on fiction. This is the same distribution shift as in [[Imitation Learning]], and it has the same fix as DAgger: **collect data with the current planner and aggregate it**, refit, repeat. Ross and Bagnell made the analogy precise, with DAgger-style guarantees for this loop.[^ross2012]

The planner makes things worse than plain distribution shift: it is an *optimiser*, and optimisers seek out errors. Among all action sequences, it will prefer exactly those along which the model predicts unrealistically high reward. Averaging the predicted return over ensemble members penalises sequences on which the members disagree, which is one reason ensembles help.

### Model-predictive control

Even an accurate model drifts over many steps: a one-step error $\varepsilon$ perturbs the next input, and errors compound along an open-loop rollout (Figure 5, left). The remedy is **model-predictive control** (MPC): at every real step, plan a sequence of $H$ actions, execute only the **first**, observe the true next state, and plan again. Replanning turns an open-loop plan into a closed-loop controller and lets real observations correct model errors before they accumulate. Because each plan is used for one step, it can be short and need not be perfect. Replanning also fixes a problem that has nothing to do with the model: a plan's last actions are chosen as if the episode ended at its horizon $H$, and only the first few actions of a short-horizon plan are trustworthy.

The planner itself can be crude. Over a horizon $H$ we want

$$
a_{0:H-1}^\star=\arg\max_{a_{0:H-1}}\ \sum_{h=0}^{H-1}r(\hat s_h,a_h),\qquad \hat s_0=s_t,\quad \hat s_{h+1}=\hat s_h+f_\phi(\hat s_h,a_h).
$$

- **Random shooting:** sample many action sequences, evaluate each in the model, keep the best.
- **Cross-entropy method (CEM):**[^rubinstein1999] iteratively refit a sampling distribution to the best samples.

> [!example] Algorithm: cross-entropy method for MPC
> 1. Initialise a Gaussian over action sequences, mean $\mathbf{m}\in\mathbb{R}^{H\times d_a}$ (for example, the previous plan shifted by one step) and standard deviation $\boldsymbol\sigma$.
> 2. Sample $K$ sequences $a^{(j)}_{0:H-1}\sim\mathcal{N}(\mathbf{m},\operatorname{diag}\boldsymbol\sigma^2)$, clipped to the action bounds.
> 3. Roll each out in the model and compute its predicted return $\hat J^{(j)}$ (with an ensemble, the average over members).
> 4. Keep the $E$ sequences with the largest $\hat J^{(j)}$ (the *elites*) and set $\mathbf{m},\boldsymbol\sigma$ to their mean and standard deviation.
> 5. Repeat steps 2–4 for a few iterations.
> 6. Execute the first action of $\mathbf{m}$, observe the new state, and go to 1.

Both planners are derivative-free and embarrassingly parallel, but they scale poorly with $H\cdot d_a$, the dimension of the search space. Gradient-based trajectory optimisation (iLQR above) or, for discrete actions, Monte Carlo tree search (as in MuZero[^muzero]) take over when that becomes the bottleneck.

### A worked example: swinging up a pendulum

To see all of this at once, take the classic torque-limited pendulum: state $s=(\theta,\omega)$ with $\theta=0$ upright, torque $a\in[-2,2]$, time step $0.05$ s, reward $-(\theta^2+0.1\omega^2+0.001a^2)$, episodes of 200 steps starting from hanging down. The torque is too weak to lift the pendulum directly, so the controller must pump energy by swinging back and forth: a genuinely nonlinear problem. The reward function is given to the agent; only the dynamics are learned. Three models are compared:

- a **linear** model, $\hat s'=s+\mathbf{W}^\top[\theta,\omega,a,1]$, fitted by least squares;
- a **nonlinear** model: ridge regression on 100 random Fourier features of $(\cos\theta,\sin\theta,\omega,a)$, which approximates a Gaussian-kernel regressor;
- an **ensemble** of 3 such nonlinear models, each with its own random features and bootstrap resample.

The loop is the aggregated one described above: iteration 0 fits the model to 5 random-torque episodes; each later iteration runs one MPC episode (CEM with $H=20$, 100 samples, 10 elites, 3 iterations) with the current model, appends its 200 transitions, and refits.

![[Reinforcement Learning/images/model-based-rl-mpc-learning.svg]]
*Figure 4: return per iteration of iterative model-based RL on the pendulum (mean of 3 seeds, band = min to max). MPC with the true dynamics scores about $-336$. Trained only on random-action data (iteration 0), every learned model fails (about $-1050$ to $-1100$): random torques never lift the pendulum, so the data contain no upright states. After aggregating the data of its own MPC episodes, the nonlinear model reaches the true-model level by iteration 2 and the ensemble by iteration 3; after that both stay between $-331$ and $-383$ on average. On this small, nearly deterministic problem the ensemble brought no measurable benefit over the single model. The linear model never swings up (mean returns between $-1511$ and $-1032$): no linear model can represent gravity's $\sin\theta$ torque over a full swing, and more data cannot fix a wrong model class.*

![[Reinforcement Learning/images/model-based-rl-compounding-replanning.svg]]
*Figure 5: left, the mean angle error of open-loop $k$-step predictions along held-out swing-up trajectories (log scale). Along the states that matter, the model trained on aggregated data is accurate to 0.0009 rad after one step and 0.016 rad after 10 steps, but its error still grows roughly exponentially and reaches the level of a guess (1–2 rad) after about 35 steps. The same model class trained on random-action data only is about 70 times worse after one step (0.062 rad), because those states were never in its training set. The linear model is wrong at every horizon (0.66 rad after 10 steps). Right, the return when $m$ actions of each plan are executed before replanning (3 episodes each). With the learned ensemble, executing up to 5 actions per plan costs little ($-344$ at $m=1$, $-406$ at $m=5$), but executing 10 or 20 drops the return to $-528$ and $-1211$. The true-dynamics curve degrades almost as much ($-491$ and $-1122$): most of the loss comes from open-loop execution of short-horizon plans, not from the model, which along these trajectories is accurate for about 10 steps.*

## Model-based policy learning

MPC never stores a policy: it re-optimises at every step, which is expensive at run time, and each plan is open-loop over its horizon. The third use of a model is to **learn a closed-loop policy** $\pi_\theta$ from it. CS 285 discusses two routes.[^cs285]

### Backpropagating through the model

If both $f_\phi$ and a deterministic policy $\mu_\theta$ are differentiable, the predicted return

$$
\hat J(\theta)=\sum_{t=0}^{H-1}r(\hat s_t,\mu_\theta(\hat s_t)),\qquad \hat s_{t+1}=f_\phi\big(\hat s_t,\mu_\theta(\hat s_t)\big),
$$

is a differentiable function of $\theta$, and we can follow its gradient as in backpropagation through time. This pathwise gradient has low variance compared with the score-function estimator of [[Policy Gradients]], and PILCO used it with Gaussian-process models to learn cart-pole swing-up from seconds of data.[^pilco]

The difficulty is conditioning. $\partial\hat J/\partial\theta$ contains products of Jacobians $\prod_t\partial\hat s_{t+1}/\partial\hat s_t$, which explode or vanish with the horizon, exactly like an RNN trained with backpropagation through time. Early actions get enormous gradients and late ones tiny gradients, and in chaotic systems the gradient becomes useless.[^pipps] Unlike LQR, there is no dynamic-programming structure to exploit, because $\theta$ couples all time steps.

### Model-free learning on model data: short rollouts

The surprisingly effective alternative is to keep a model-free learner and use the model only as a **data generator**, the deep-RL version of Dyna. Long imagined rollouts suffer compounding error, while one-step rollouts from the start state never reach later states. The compromise is **branched short rollouts**: start $k$-step model rollouts from states sampled from the *real* replay buffer, so the rollouts cover the states the agent actually visits while each stays short enough to be accurate.

**MBPO** (model-based policy optimisation)[^mbpo] does exactly this: a PETS-style ensemble, rollouts of a few steps branched from real states, and soft actor-critic trained mostly on the model-generated transitions. Janner et al. bound the gap between the model's and the real return, and the bound trades off rollout length against model error, which motivates short rollouts. Earlier variants of the idea include model-based acceleration[^gu2016] (imagined transitions added to a Q-learning buffer) and model-based value expansion[^mve] (a few model steps used to sharpen Q-learning targets).

### World models

For high-dimensional observations such as images, predicting the next frame pixel by pixel is wasteful. **World models** learn a compact latent state $z_t$ with an encoder, a latent dynamics model and a reward model, and plan or learn inside the latent space. Ha and Schmidhuber trained a controller entirely inside such a model;[^ha2018] Dreamer trains an actor-critic by backpropagating through imagined latent trajectories;[^dreamer] and MuZero learns a model that predicts only what planning needs (rewards, values and policies, not observations) and plans with tree search, matching AlphaZero at Go, chess and shogi without being given the rules, and setting a new state of the art on Atari.[^muzero]

## Key takeaways

- The planning algorithms need $p$ and $r$. **Model-based RL learns them** from transitions, which are rich supervised data, and then plans; with an exact model it is just DP.
- **Certainty equivalence** (count-based $\hat p,\hat r$ + VI/PI) is consistent. The **simulation lemma** bounds the value error by $\epsilon_r/(1-\gamma)+\gamma\epsilon_pR_{\max}/(1-\gamma)^2$, and the CE policy loses at most twice that. Sample complexity is polynomial; sharp analyses give $\tilde O\big(|\mathcal{S}||\mathcal{A}|/(\varepsilon^2(1-\gamma)^3)\big)$ with a generative model.
- Online, a greedy planner **exploits model errors** and stops collecting the data that would correct them. Optimism (R-max, E³, count bonuses) is the remedy, as in bandits.
- **Dyna** makes planning just "learning from simulated experience": a few planning updates per real step can cut the real experience needed by a large factor.
- In continuous spaces: fit a dynamics model by regression, **aggregate data** collected by the planner (distribution shift, as in DAgger), **replan** every step (MPC) because errors compound, and use **ensembles** so the planner cannot exploit epistemic uncertainty. LQR is the case where planning is closed-form.
- To get a policy, backpropagating through long model rollouts is ill-conditioned; feeding **short branched rollouts** to a model-free learner (MBPO) works well in practice.

## Where this shows up next

- This lecture closes the series by joining its two halves: learning from experience and planning. The convergence statements behind both (for example, that Q-learning, and Dyna-Q's planning updates, converge) rest on [[Stochastic Approximation]].
- The model-free methods of the first half are the learners that Dyna and MBPO accelerate: [[Q-Learning and SARSA]] for Dyna-Q, actor-critic methods ([[Actor-Critic Algorithms]]) for MBPO.
- The optimism principle used here for exploration is developed in its simplest setting in [[Multi-Armed Bandits]].

<!-- References: footnote definitions below render as the numbered "References" list at the bottom of the page -->

[^cs285]: Levine, S. *CS 285: Deep Reinforcement Learning*, UC Berkeley, Fall 2023: Lecture 10 "Optimal Control and Planning", Lecture 11 "Model-Based Reinforcement Learning", Lecture 12 "Model-Based Policy Learning". [course page](https://rail.eecs.berkeley.edu/deeprlcourse-fa23/)
[^simon1956]: Simon, H. A. (1956). *Dynamic Programming Under Uncertainty with a Quadratic Criterion Function*. Econometrica, 24(1), 74–81. [doi:10.2307/1905261](https://doi.org/10.2307/1905261)
[^azar2013]: Gheshlaghi Azar, M., Munos, R., & Kappen, H. J. (2013). *Minimax PAC bounds on the sample complexity of reinforcement learning with a generative model*. Machine Learning, 91, 325–349. [doi:10.1007/s10994-013-5368-1](https://doi.org/10.1007/s10994-013-5368-1)
[^kearns1999]: Kearns, M., & Singh, S. (1999). *Finite-Sample Convergence Rates for Q-Learning and Indirect Algorithms*. Advances in Neural Information Processing Systems 11. [proceedings](https://papers.nips.cc/paper/1531-finite-sample-convergence-rates-for-q-learning-and-indirect-algorithms)
[^pilco]: Deisenroth, M. P., & Rasmussen, C. E. (2011). *PILCO: A Model-Based and Data-Efficient Approach to Policy Search*. ICML. [pdf](https://icml.cc/2011/papers/323_icmlpaper.pdf)
[^rmax]: Brafman, R. I., & Tennenholtz, M. (2002). *R-max: A General Polynomial Time Algorithm for Near-Optimal Reinforcement Learning*. Journal of Machine Learning Research, 3, 213–231. [JMLR](https://www.jmlr.org/papers/v3/brafman02a.html)
[^e3]: Kearns, M., & Singh, S. (2002). *Near-Optimal Reinforcement Learning in Polynomial Time*. Machine Learning, 49, 209–232. [doi:10.1023/A:1017984413808](https://doi.org/10.1023/A:1017984413808)
[^mbie]: Strehl, A. L., & Littman, M. L. (2008). *An analysis of model-based Interval Estimation for Markov Decision Processes*. Journal of Computer and System Sciences, 74(8), 1309–1331. [doi:10.1016/j.jcss.2007.08.009](https://doi.org/10.1016/j.jcss.2007.08.009)
[^tassa2012]: Tassa, Y., Erez, T., & Todorov, E. (2012). *Synthesis and stabilization of complex behaviors through online trajectory optimization*. IEEE/RSJ IROS, 4906–4913. [doi:10.1109/IROS.2012.6386025](https://doi.org/10.1109/IROS.2012.6386025)
[^sutton1990]: Sutton, R. S. (1990). *Integrated Architectures for Learning, Planning, and Reacting Based on Approximating Dynamic Programming*. Machine Learning Proceedings 1990 (ICML), 216–224. [doi:10.1016/B978-1-55860-141-3.50030-4](https://doi.org/10.1016/B978-1-55860-141-3.50030-4)
[^sutton1991]: Sutton, R. S. (1991). *Dyna, an integrated architecture for learning, planning, and reacting*. ACM SIGART Bulletin, 2(4), 160–163. [doi:10.1145/122344.122377](https://doi.org/10.1145/122344.122377)
[^moore1993]: Moore, A. W., & Atkeson, C. G. (1993). *Prioritized sweeping: Reinforcement learning with less data and less time*. Machine Learning, 13, 103–130. [doi:10.1007/BF00993104](https://doi.org/10.1007/BF00993104)
[^suttonbarto]: Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.), Chapter 8. [book website](http://incompleteideas.net/book/the-book-2nd.html)
[^ensembles]: Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles*. NeurIPS. [arXiv:1612.01474](https://arxiv.org/abs/1612.01474)
[^pets]: Chua, K., Calandra, R., McAllister, R., & Levine, S. (2018). *Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models*. NeurIPS. [arXiv:1805.12114](https://arxiv.org/abs/1805.12114)
[^ross2012]: Ross, S., & Bagnell, J. A. (2012). *Agnostic System Identification for Model-Based Reinforcement Learning*. ICML. [arXiv:1203.1007](https://arxiv.org/abs/1203.1007)
[^rubinstein1999]: Rubinstein, R. (1999). *The Cross-Entropy Method for Combinatorial and Continuous Optimization*. Methodology and Computing in Applied Probability, 1, 127–190. [doi:10.1023/A:1010091220143](https://doi.org/10.1023/A:1010091220143)
[^muzero]: Schrittwieser, J., et al. (2020). *Mastering Atari, Go, chess and shogi by planning with a learned model*. Nature, 588, 604–609. [doi:10.1038/s41586-020-03051-4](https://doi.org/10.1038/s41586-020-03051-4)
[^pipps]: Parmas, P., Rasmussen, C. E., Peters, J., & Doya, K. (2018). *PIPPS: Flexible Model-Based Policy Search Robust to the Curse of Chaos*. ICML. [arXiv:1902.01240](https://arxiv.org/abs/1902.01240)
[^mbpo]: Janner, M., Fu, J., Zhang, M., & Levine, S. (2019). *When to Trust Your Model: Model-Based Policy Optimization*. NeurIPS. [arXiv:1906.08253](https://arxiv.org/abs/1906.08253)
[^gu2016]: Gu, S., Lillicrap, T., Sutskever, I., & Levine, S. (2016). *Continuous Deep Q-Learning with Model-based Acceleration*. ICML. [arXiv:1603.00748](https://arxiv.org/abs/1603.00748)
[^mve]: Feinberg, V., Wan, A., Stoica, I., Jordan, M. I., Gonzalez, J. E., & Levine, S. (2018). *Model-Based Value Estimation for Efficient Model-Free Reinforcement Learning*. [arXiv:1803.00101](https://arxiv.org/abs/1803.00101)
[^ha2018]: Ha, D., & Schmidhuber, J. (2018). *World Models*. [arXiv:1803.10122](https://arxiv.org/abs/1803.10122)
[^dreamer]: Hafner, D., Lillicrap, T., Ba, J., & Norouzi, M. (2020). *Dream to Control: Learning Behaviors by Latent Imagination*. ICLR. [arXiv:1912.01603](https://arxiv.org/abs/1912.01603)
