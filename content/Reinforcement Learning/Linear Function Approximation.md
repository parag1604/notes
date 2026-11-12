---
title: Linear Function Approximation
tags: [reinforcement-learning, function-approximation, temporal-difference, policy-evaluation]
date: 2026-11-12
draft: false
prev: "[[Q-Learning and SARSA]]"
next: "[[Off-Policy Learning with Function Approximation]]"
---
> **Prerequisites:** [[Introduction to Reinforcement Learning]] (MDPs, $V^\pi$, the Bellman expectation equation in matrix form), [[Monte Carlo and Temporal-Difference Learning]] (tabular TD(0)), [[Markov Chains]] (stationary distributions), [[Linear Models]] (least squares as a projection).
>
> **TL;DR:** When there are too many states for a table, we approximate $V^\pi\approx\Phi\mathbf{w}$ with a few features. The natural target is the best fit $\Pi V^\pi$ in a norm weighted by how often states are visited. Linear TD(0) does **not** find it. TD(0) is a *semi-gradient* method, and it converges with probability one to a different point $\mathbf{w}_{\mathrm{TD}}=A^{-1}\mathbf{b}$, the fixed point of the **projected** Bellman operator. Its error is at most $1/\sqrt{1-\gamma^2}$ times the best possible error. All of this hinges on sampling states from the policy's own stationary distribution.

**Notation used only here.** Weights are $\mathbf{w}\in\mathbb{R}^d$ and features $\boldsymbol\phi(s)\in\mathbb{R}^d$. The best-fit weights are $\mathbf{w}_{\mathrm{MSE}}$ and the TD limit is $\mathbf{w}_{\mathrm{TD}}$; they are different points, and keeping two names for them is the main point of the lecture. $d_\pi$ is the stationary distribution of the chain induced by $\pi$ and $D=\operatorname{diag}(d_\pi)$.

## Motivation

Every method so far, tabular TD, [[Q-Learning and SARSA|Q-learning and SARSA]], stores one number per state (or state–action pair). That fails in two ways once the state space is large:

1. **Memory and time.** Backgammon has about $10^{20}$ positions, and a robot's joint angles are continuous.
2. **No generalization.** A table learns nothing about a state until it visits that state, even if a very similar state has been visited thousands of times.

The remedy is to replace the table by a parametric function with far fewer parameters than states, $d\ll|\mathcal{S}|$, and learn the parameters. Changing one weight then changes the predicted value of *many* states at once. That is the source of generalization, and also of all the new difficulties in this lecture.

There are two common families of approximators.

- **Linear:** $V^\pi(s)\approx\boldsymbol\phi(s)^\top\mathbf{w}$, i.e. $V^\pi\approx\Phi\mathbf{w}$, where the **feature matrix** $\Phi\in\mathbb{R}^{|\mathcal{S}|\times d}$ is fixed in advance and has $\boldsymbol\phi(s)^\top$ as its row $s$.
- **Nonlinear:** $V^\pi(s)\approx f_{\mathbf{w}}(s)$, typically a neural network whose depth and width fix the function class.

This lecture is about the linear case. It is the case where the theory is complete, and it already shows every phenomenon that makes nonlinear value learning hard. Even on-policy, TD with a nonlinear architecture can diverge; Tsitsiklis and Van Roy give an example.[^tvr1997]

Throughout, the MDP and the policy $\pi$ are fixed, and we only *evaluate* $\pi$. We assume:

- $\mathcal{S}$ is finite and the chain induced by $\pi$ is **irreducible**, so it has a unique stationary distribution $d_\pi$ with $d_\pi(s)>0$ for every $s$ (see [[Markov Chains#The stationary distribution|stationary distributions]]);
- $\Phi$ has **full column rank** (no feature is a linear combination of the others);
- $\gamma\in[0,1)$, and rewards are bounded.

We write $P^\pi$ for the transition matrix of the induced chain, $P^\pi(s'\mid s)=\sum_a\pi(a\mid s)\,p(s'\mid s,a)$, and $r_\pi(s)=\sum_a\pi(a\mid s)\,r(s,a)$ for the expected one-step reward. The Bellman operator is $\mathcal{T}^\pi V=r_\pi+\gamma P^\pi V$, and $V^\pi=\mathcal{T}^\pi V^\pi$.

---

## What a linear architecture can and cannot represent

A linear architecture can only produce vectors in the **column space** of $\Phi$,

$$
\mathcal{V}=\{\Phi\mathbf{w}:\mathbf{w}\in\mathbb{R}^d\}\subset\mathbb{R}^{|\mathcal{S}|},
$$

a $d$-dimensional subspace of the $|\mathcal{S}|$-dimensional space of value functions. Choosing features is choosing which value functions are expressible. Every other value function can only be approximated.

> [!example] One feature, two states
> Take $|\mathcal{S}|=2$, $d=1$ and $\Phi=(2,1)^\top$. Then $\Phi w=(2w,w)^\top$: whatever $w$ is, the first state is predicted to be worth exactly twice the second. If the true value is $V^\pi=(3,1)^\top$, no $w$ reproduces it, and the best we can do depends on how we measure the error.

The same happens, less obviously, with realistic features.

> [!example] A two-station bike-share
> A small town runs a bike-share with two docking stations, *Riverside* and *Hilltop*, and five bikes. Each day riders rent bikes at either station for a fee (a rider who finds an empty station is lost business) and leave them at either station. After midnight the operator may truck bikes between the stations at a cost per bike. The reward is rental income minus trucking cost. This is a two-station version of Jack's car rental in Sutton and Barto.[^suttonbarto]
>
> The state is the **number** of bikes at Riverside at midnight, $N_1\in\{0,\dots,5\}$, so $|\mathcal{S}|=6$. (The bikes are interchangeable; the location of each individual bike would give $2^5$ states that the problem does not distinguish.) A natural feature vector is $\boldsymbol\phi(s)=(N_1,N_2)^\top$ with $N_2=5-N_1$:
>
> $$
> \Phi=\begin{pmatrix}5&0\\4&1\\3&2\\2&3\\1&4\\0&5\end{pmatrix}.
> $$
>
> Is this a good choice? Because $N_1+N_2=5$ in every state,
>
> $$
> \Phi\mathbf{w}=w_1N_1+w_2(5-N_1)=5w_2+(w_1-w_2)\,N_1 .
> $$
>
> So this architecture represents **exactly the value functions that are affine in $N_1$**. It is the same span as the features $(1,N_1)$, and adding a constant feature would not enlarge it (the rank stays 2). It cannot express "the value is highest when the bikes are balanced between the stations", which is what one would expect when demand is similar at both ends. Whether $(N_1,N_2)$ is good enough depends on whether $V^\pi$ is close to affine in $N_1$, and that is precisely what the approximation error below measures.

---

## Which error? The stationary-distribution weighting

To find the "best" $\mathbf{w}$ we must decide how to weigh errors in different states. The naive choice is plain least squares, $\min_{\mathbf{w}}\lVert V^\pi-\Phi\mathbf{w}\rVert_2^2=\min_{\mathbf{w}}\sum_s\big(V^\pi(s)-\boldsymbol\phi(s)^\top\mathbf{w}\big)^2$. It weights every state equally, including states the policy almost never visits.

A better choice follows from how data arrive. Running $\pi$ produces one long trajectory $s_0,a_0,r_0,s_1,a_1,r_1,\dots$, and by the [[Markov Chains#Convergence and the ergodic theorem|ergodic theorem]], for an irreducible finite chain and any $s_0$,

$$
\frac{1}{T}\sum_{t=0}^{T-1}\mathbb{1}[s_t=s]\;\longrightarrow\;d_\pi(s)\quad\text{almost surely}.
$$

(Irreducibility alone gives uniqueness of $d_\pi$ and this time average; aperiodicity is needed only for the stronger statement $\Pr(s_t=s)\to d_\pi(s)$.) A state's share of the data is its stationary probability, so we weight its error by that probability. This gives the **mean squared value error**

$$
\overline{\mathrm{VE}}(\mathbf{w})=\sum_s d_\pi(s)\big(V^\pi(s)-\boldsymbol\phi(s)^\top\mathbf{w}\big)^2=\lVert V^\pi-\Phi\mathbf{w}\rVert_D^2,
$$

where $\lVert v\rVert_D=\sqrt{v^\top Dv}=\lVert D^{1/2}v\rVert_2$. It is a norm because every $d_\pi(s)>0$.

$\overline{\mathrm{VE}}$ is a **weighted least-squares** problem (see [[Linear Models#Weighted linear regression|weighted linear regression]]). Setting the gradient $\Phi^\top D(\Phi\mathbf{w}-V^\pi)$ to zero gives the best-fit weights and the best approximation:

$$
\begin{aligned}
\mathbf{w}_{\mathrm{MSE}}&=(\Phi^\top D\Phi)^{-1}\Phi^\top D\,V^\pi,\\
\Phi\mathbf{w}_{\mathrm{MSE}}&=\Pi V^\pi,\qquad \Pi=\Phi(\Phi^\top D\Phi)^{-1}\Phi^\top D .
\end{aligned}
\tag{1}
$$

$\Pi$ is the projection onto $\mathcal{V}$ that is orthogonal in the inner product $\langle u,v\rangle_D=u^\top Dv$. In ordinary Euclidean geometry it is an *oblique* projection; it becomes a perpendicular drop only after rescaling coordinates by $D^{1/2}$. This is the analogue of the hat matrix in [[Linear Models#The geometry: projection onto the column space|least squares]].

The best fit is our benchmark, but (1) cannot be computed: both $V^\pi$ and $d_\pi$ are unknown. We have only samples.

---

## From gradient descent to TD(0)

The gradient of $\tfrac12\overline{\mathrm{VE}}$ is an expectation over the stationary distribution:

$$
\begin{aligned}
\nabla_{\mathbf{w}}\tfrac12\overline{\mathrm{VE}}(\mathbf{w})&=-\sum_s d_\pi(s)\big(V^\pi(s)-\boldsymbol\phi(s)^\top\mathbf{w}\big)\boldsymbol\phi(s)\\
&=-\mathbb{E}_{s\sim d_\pi}\Big[\big(V^\pi(s)-\boldsymbol\phi(s)^\top\mathbf{w}\big)\boldsymbol\phi(s)\Big].
\end{aligned}
$$

If we could sample $s_t\sim d_\pi$ and see $V^\pi(s_t)$, stochastic gradient descent would be

$$
\mathbf{w}_{t+1}=\mathbf{w}_t+\alpha_t\big(V^\pi(s_t)-\boldsymbol\phi(s_t)^\top\mathbf{w}_t\big)\boldsymbol\phi(s_t).
$$

$V^\pi(s_t)$ is unknown, so it must be replaced by a sample *target*. There are two choices.

1. **The return** $G_t$, whose conditional mean is $V^\pi(s_t)$. The update is then a genuine stochastic gradient of $\overline{\mathrm{VE}}$ (gradient Monte Carlo), and it heads for $\mathbf{w}_{\mathrm{MSE}}$. The price is the high variance of $G_t$ and having to wait until the return is known.
2. **The Bellman target.** By the Bellman equation, $V^\pi(s)=\mathbb{E}\big[r_t+\gamma V^\pi(s_{t+1})\mid s_t=s\big]$. (This step uses the Markov property and time-homogeneity, so the identity holds at every $t$.) So $r_t+\gamma V^\pi(s_{t+1})$ is also an unbiased target, but it still contains the unknown $V^\pi(s_{t+1})$.

The idea that defines temporal-difference learning is to **bootstrap**: use the current estimate $\boldsymbol\phi(s_{t+1})^\top\mathbf{w}_t$ in place of $V^\pi(s_{t+1})$.[^sutton1988] This gives:

> [!example] Algorithm: linear TD(0) (on-policy evaluation)
> **Input:** policy $\pi$, features $\boldsymbol\phi$, step sizes $\alpha_t>0$. Initialize $\mathbf{w}_0$ arbitrarily and observe $s_0$.
> **For** $t=0,1,2,\dots$:
> 1. Take $a_t\sim\pi(\cdot\mid s_t)$; observe $r_t$ and $s_{t+1}$.
> 2. TD error: $\delta_t=r_t+\gamma\,\boldsymbol\phi(s_{t+1})^\top\mathbf{w}_t-\boldsymbol\phi(s_t)^\top\mathbf{w}_t$.
> 3. Update: $\mathbf{w}_{t+1}=\mathbf{w}_t+\alpha_t\,\delta_t\,\boldsymbol\phi(s_t)$.

With one-hot features, $\Phi=I$, this is exactly tabular TD(0) from [[Monte Carlo and Temporal-Difference Learning]]: $V(s_t)\leftarrow V(s_t)+\alpha_t\big[r_t+\gamma V(s_{t+1})-V(s_t)\big]$. The name comes from $\delta_t$, the difference between a one-step look-ahead estimate of $V^\pi(s_t)$ and the current estimate.

> [!warning] TD(0) is a semi-gradient method, not SGD
> The bootstrapped target depends on $\mathbf{w}_t$, but the update differentiates only the prediction $\boldsymbol\phi(s_t)^\top\mathbf{w}$ and treats the target as a constant. The result is **not** the stochastic gradient of $\overline{\mathrm{VE}}$, or of any other function. We will see below that its average update is $\mathbf{b}-A\mathbf{w}$. If this were $-\nabla J(\mathbf{w})$ for some $J$, the Hessian of $J$ would be $A$, which would have to be symmetric. But $A$ is not symmetric in general: in the three-state example of Figure 1, $A=\begin{pmatrix}0.300&-0.205\\0.033&0.308\end{pmatrix}$. So TD(0) cannot be descending any objective, and we have no reason to expect it to reach the minimizer of $\overline{\mathrm{VE}}$. It does not, as we will show.

A second difference from SGD: the states $s_0,s_1,\dots$ are a correlated Markov chain, not independent draws from $d_\pi$. Both issues are handled by the theory of stochastic approximation with Markov noise.

---

## TD(0) as stochastic approximation

Three questions remain. Does linear TD(0) converge? If so, to what? And how good is the limit?

The convergence toolkit is the ODE method of [[Stochastic Approximation]]: an iteration $\mathbf{x}_{t+1}=\mathbf{x}_t+\alpha_t\big[h(\mathbf{x}_t,Y_t)+M_{t+1}\big]$, with Markov noise $Y_t$ and martingale-difference noise $M_{t+1}$, asymptotically tracks the deterministic ODE $\dot{\mathbf{x}}=\bar h(\mathbf{x})$. Here $\bar h$ averages $h(\cdot,y)$ over the stationary distribution of $Y_t$.

**Step 1: the mean field.** Let $\mathcal{F}_t=\sigma(\mathbf{w}_0,s_0,a_0,\dots,s_{t-1},a_{t-1},s_t)$, the history up to and including $s_t$ but not $a_t$. Given $\mathcal{F}_t$, the pair $(a_t,s_{t+1})$ depends only on $s_t$, and $\mathbf{w}_t$ is known. So

$$
\begin{aligned}
h(\mathbf{w},s)&:=\mathbb{E}\big[\delta_t\boldsymbol\phi(s_t)\mid s_t=s,\ \mathbf{w}_t=\mathbf{w}\big]\\
&=\boldsymbol\phi(s)\Big(r_\pi(s)+\gamma\sum_{s'}P^\pi(s'\mid s)\,\boldsymbol\phi(s')^\top\mathbf{w}-\boldsymbol\phi(s)^\top\mathbf{w}\Big),
\end{aligned}
$$

and TD(0) reads $\mathbf{w}_{t+1}=\mathbf{w}_t+\alpha_t\big[h(\mathbf{w}_t,s_t)+M_{t+1}\big]$ with $M_{t+1}=\delta_t\boldsymbol\phi(s_t)-h(\mathbf{w}_t,s_t)$. By construction $\mathbb{E}[M_{t+1}\mid\mathcal{F}_t]=0$, so $(M_t)$ is a martingale-difference sequence. The state $s_t$ itself is the Markov noise.

**Step 2: average over $d_\pi$.** Weighting $h(\mathbf{w},s)$ by $d_\pi(s)$ and stacking the sums into matrices gives the limiting ODE

$$
\begin{aligned}
\dot{\mathbf{w}}=\bar h(\mathbf{w})&=\Phi^\top D\big(r_\pi+\gamma P^\pi\Phi\mathbf{w}-\Phi\mathbf{w}\big)=\mathbf{b}-A\mathbf{w},\\
A&=\Phi^\top D(I-\gamma P^\pi)\Phi,\qquad \mathbf{b}=\Phi^\top D\,r_\pi .
\end{aligned}
\tag{2}
$$

It is a *linear* ODE. If $A$ is invertible, its only equilibrium is

$$
\mathbf{w}_{\mathrm{TD}}=A^{-1}\mathbf{b}=\big(\Phi^\top D(I-\gamma P^\pi)\Phi\big)^{-1}\Phi^\top D\,r_\pi .
$$

Compare with (1): $\mathbf{w}_{\mathrm{MSE}}$ involves $V^\pi$, while $\mathbf{w}_{\mathrm{TD}}$ involves only one-step quantities, $r_\pi$ and $P^\pi$. That is what makes it learnable by bootstrapping, and also why it differs from the best fit.

**Step 3: check the conditions.** The [[Stochastic Approximation]] lecture states the Markov-noise convergence theorem with exact hypotheses.[^borkar2008] For linear TD(0) they hold as follows, with $C_\phi=\max_s\lVert\boldsymbol\phi(s)\rVert$ and $C_r=\max|r|$.

- **Lipschitz field.** $h(\mathbf{w},s)=\boldsymbol\phi(s)r_\pi(s)-A(s)\mathbf{w}$ with $A(s)=\boldsymbol\phi(s)\big(\boldsymbol\phi(s)-\gamma\sum_{s'}P^\pi(s'\mid s)\boldsymbol\phi(s')\big)^\top$ is affine in $\mathbf{w}$, and $\lVert h(\mathbf{w},s)-h(\mathbf{w}',s)\rVert\le(1+\gamma)C_\phi^2\lVert\mathbf{w}-\mathbf{w}'\rVert$, uniformly in $s$.
- **Noise.** $\lVert M_{t+1}\rVert\le 2C_\phi C_r+2\gamma C_\phi^2\lVert\mathbf{w}_t\rVert\le C_M(1+\lVert\mathbf{w}_t\rVert)$, hence $\mathbb{E}\big[\lVert M_{t+1}\rVert^2\mid\mathcal{F}_t\big]\le 2C_M^2(1+\lVert\mathbf{w}_t\rVert^2)$.
- **Step sizes.** Robbins–Monro: $\sum_t\alpha_t=\infty$ and $\sum_t\alpha_t^2<\infty$, e.g. $\alpha_t=(t+1)^{-\kappa}$ with $\kappa\in(\tfrac12,1]$.
- **Markov noise.** $s_t$ is a finite irreducible chain whose transitions do not depend on $\mathbf{w}_t$, with unique stationary distribution $d_\pi$.
- **Stability**, $\sup_t\lVert\mathbf{w}_t\rVert<\infty$ almost surely. This is not automatic and must be proved. For linear TD it follows from the Borkar–Meyn criterion:[^borkarmeyn2000] the "scaled" field $\lim_{c\to\infty}\bar h(c\mathbf{w})/c=-A\mathbf{w}$ defines an ODE whose origin is globally asymptotically stable, by the theorem below.
- **A globally asymptotically stable equilibrium** of (2). This is the heart of the matter, and the next section proves it.

---

## Convergence: $A$ is positive definite

Start with one dimension. The solution of $\dot x=b-ax$ is $x(t)=b/a+(x_0-b/a)e^{-at}$. It converges to $b/a$ from every start if $a>0$, and runs off to $\pm\infty$ if $a<0$. In $d$ dimensions, the role of "$a>0$" is played by **positive definiteness**: $\mathbf{x}^\top A\mathbf{x}>0$ for all $\mathbf{x}\neq0$. This makes sense for non-symmetric $A$ too, and our $A$ is not symmetric.

Everything rests on one inequality: **averaging under the chain does not increase the $D$-norm.**

> [!note] Lemma 1 (the stationary distribution makes $P^\pi$ non-expansive)
> For every $z\in\mathbb{R}^{|\mathcal{S}|}$, $\lVert P^\pi z\rVert_D\le\lVert z\rVert_D$.

> [!proof]- Proof of Lemma 1
> The $s$-th entry of $P^\pi z$ is the conditional mean $\mathbb{E}[z(s_{t+1})\mid s_t=s]$. Jensen's inequality for the convex function $x\mapsto x^2$ gives $\big(\sum_{s'}P^\pi(s'\mid s)z(s')\big)^2\le\sum_{s'}P^\pi(s'\mid s)z(s')^2$. Therefore
>
> $$
> \begin{aligned}
> \lVert P^\pi z\rVert_D^2&=\sum_s d_\pi(s)\Big(\sum_{s'}P^\pi(s'\mid s)z(s')\Big)^2\le\sum_s d_\pi(s)\sum_{s'}P^\pi(s'\mid s)z(s')^2\\
> &=\sum_{s'}z(s')^2\sum_s d_\pi(s)P^\pi(s'\mid s)=\sum_{s'}d_\pi(s')z(s')^2=\lVert z\rVert_D^2 .
> \end{aligned}
> $$
>
> The step $\sum_s d_\pi(s)P^\pi(s'\mid s)=d_\pi(s')$ is stationarity, $d_\pi^\top P^\pi=d_\pi^\top$. It is the only place the specific weighting is used, and the [[Off-Policy Learning with Function Approximation|off-policy lecture]] is about what happens when it fails.

> [!note] Theorem 1 ($A$ is positive definite)
> For every $\mathbf{w}$, $\;\mathbf{w}^\top A\mathbf{w}\ge(1-\gamma)\lVert\Phi\mathbf{w}\rVert_D^2$. Since $\gamma<1$, $d_\pi>0$ and $\Phi$ has full column rank, $\mathbf{w}^\top A\mathbf{w}>0$ for all $\mathbf{w}\ne0$.

> [!proof]- Proof of Theorem 1
> Put $z=\Phi\mathbf{w}$. Then $\mathbf{w}^\top A\mathbf{w}=z^\top Dz-\gamma\,z^\top DP^\pi z=\lVert z\rVert_D^2-\gamma\langle z,P^\pi z\rangle_D$. By Cauchy–Schwarz in $\langle\cdot,\cdot\rangle_D$ and Lemma 1, $\langle z,P^\pi z\rangle_D\le\lVert z\rVert_D\lVert P^\pi z\rVert_D\le\lVert z\rVert_D^2$. Hence $\mathbf{w}^\top A\mathbf{w}\ge(1-\gamma)\lVert z\rVert_D^2$. Finally, $z=\Phi\mathbf{w}=0$ only if $\mathbf{w}=0$ (full column rank), and $\lVert z\rVert_D=0$ only if $z=0$ (all $d_\pi(s)>0$).

**Global asymptotic stability.** Positive definiteness makes $A$ invertible (if $A\mathbf{x}=0$ then $\mathbf{x}^\top A\mathbf{x}=0$, so $\mathbf{x}=0$). It also gives a Lyapunov function. Let $c>0$ be the smallest eigenvalue of the symmetric part $\tfrac12(A+A^\top)$, so that $\mathbf{x}^\top A\mathbf{x}\ge c\lVert\mathbf{x}\rVert^2$. Along solutions of (2),

$$
\begin{aligned}
\frac{d}{dt}\lVert\mathbf{w}-\mathbf{w}_{\mathrm{TD}}\rVert^2&=2(\mathbf{w}-\mathbf{w}_{\mathrm{TD}})^\top(\mathbf{b}-A\mathbf{w})\\
&=-2(\mathbf{w}-\mathbf{w}_{\mathrm{TD}})^\top A(\mathbf{w}-\mathbf{w}_{\mathrm{TD}})\le-2c\,\lVert\mathbf{w}-\mathbf{w}_{\mathrm{TD}}\rVert^2 ,
\end{aligned}
$$

so every solution converges to $\mathbf{w}_{\mathrm{TD}}$ exponentially fast. (In the walk example below, $c=0.0125$, which equals $(1-\gamma)\lambda_{\min}(\Phi^\top D\Phi)$, the bound from Theorem 1.)

> [!note] Theorem 2 (convergence of linear TD(0); Tsitsiklis & Van Roy, 1997)
> Assume a finite, irreducible and aperiodic chain under $\pi$, bounded rewards, $\Phi$ of full column rank, $\gamma\in[0,1)$, and deterministic step sizes with $\sum_t\alpha_t=\infty$ and $\sum_t\alpha_t^2<\infty$. Then linear TD(0) converges with probability one to $\mathbf{w}_{\mathrm{TD}}=A^{-1}\mathbf{b}$.

Tsitsiklis and Van Roy prove this directly, for general state spaces and for TD($\lambda$).[^tvr1997] The ODE route above gives the same conclusion once the stability condition is in place.

---

## Where TD converges: the projected Bellman equation

We have the answer to "does it converge?" To understand *where* it goes, look at the equilibrium condition $\mathbf{b}-A\mathbf{w}=0$:

$$
\Phi^\top D\big(r_\pi+\gamma P^\pi\Phi\mathbf{w}-\Phi\mathbf{w}\big)=0
\quad\Longleftrightarrow\quad
\Phi^\top D\big(\mathcal{T}^\pi\Phi\mathbf{w}-\Phi\mathbf{w}\big)=0 .
$$

These are the normal equations saying that $\Phi\mathbf{w}$ is the $D$-projection of $\mathcal{T}^\pi\Phi\mathbf{w}$. In words: **apply one Bellman backup to the current approximation, project the result back onto $\mathcal{V}$, and you must land where you started.**

$$
\Phi\mathbf{w}_{\mathrm{TD}}=\Pi\,\mathcal{T}^\pi\,\Phi\mathbf{w}_{\mathrm{TD}} .
\tag{3}
$$

This is the **projected Bellman equation**. The three lemmas below show that its solution is unique. They are the $D$-norm counterparts of the sup-norm contraction argument proved in [[Dynamic Programming]].

> [!note] Lemma 2 (contractions in the $D$-norm)
> 1. $\mathcal{T}^\pi$ is a $\gamma$-contraction in $\lVert\cdot\rVert_D$: $\lVert\mathcal{T}^\pi U-\mathcal{T}^\pi V\rVert_D\le\gamma\lVert U-V\rVert_D$.
> 2. $\Pi$ is non-expansive in $\lVert\cdot\rVert_D$: $\lVert\Pi v\rVert_D\le\lVert v\rVert_D$. More precisely, $\lVert v\rVert_D^2=\lVert\Pi v\rVert_D^2+\lVert v-\Pi v\rVert_D^2$.
> 3. Hence $\Pi\mathcal{T}^\pi$ is a $\gamma$-contraction in $\lVert\cdot\rVert_D$, and by Banach's fixed-point theorem it has a unique fixed point. That fixed point is $\Phi\mathbf{w}_{\mathrm{TD}}$.

> [!proof]- Proof of Lemma 2
> 1. $\mathcal{T}^\pi U-\mathcal{T}^\pi V=\gamma P^\pi(U-V)$, so the claim is Lemma 1.
> 2. Write $v=\Pi v+(v-\Pi v)$. The cross term in $\lVert v\rVert_D^2$ is $2(\Pi v)^\top D(v-\Pi v)=2v^\top(\Pi^\top Dv-\Pi^\top D\Pi v)$. A direct computation gives $\Pi^\top D\Pi=D\Phi(\Phi^\top D\Phi)^{-1}\Phi^\top D\Phi(\Phi^\top D\Phi)^{-1}\Phi^\top D=D\Phi(\Phi^\top D\Phi)^{-1}\Phi^\top D=\Pi^\top D$. So the cross term vanishes: $\Pi v\perp_D v-\Pi v$, and Pythagoras gives the identity. (In general $\Pi v$ is *not* Euclidean-orthogonal to $v-\Pi v$.)
> 3. $\Pi$ is linear, so $\lVert\Pi\mathcal{T}^\pi U-\Pi\mathcal{T}^\pi V\rVert_D=\lVert\Pi(\mathcal{T}^\pi U-\mathcal{T}^\pi V)\rVert_D\le\lVert\mathcal{T}^\pi U-\mathcal{T}^\pi V\rVert_D\le\gamma\lVert U-V\rVert_D$. $\mathbb{R}^{|\mathcal{S}|}$ with $\lVert\cdot\rVert_D$ is complete, so a unique fixed point exists, and it lies in $\mathcal{V}$ because $\Pi$ maps into $\mathcal{V}$. To see that it is $\Phi\mathbf{w}_{\mathrm{TD}}$: $A\mathbf{w}_{\mathrm{TD}}=\mathbf{b}$ rearranges to $\Phi^\top D\Phi\,\mathbf{w}_{\mathrm{TD}}=\Phi^\top D(r_\pi+\gamma P^\pi\Phi\mathbf{w}_{\mathrm{TD}})$. Multiplying by $\Phi(\Phi^\top D\Phi)^{-1}$ gives $\Phi\mathbf{w}_{\mathrm{TD}}=\Pi\mathcal{T}^\pi\Phi\mathbf{w}_{\mathrm{TD}}$.

![[Reinforcement Learning/images/linear-function-approximation-geometry.svg]]
*Figure 1: the geometry of linear value approximation for a 3-state chain with 2 features ($\gamma=0.9$), drawn in coordinates where $\lVert\cdot\rVert_D$ is ordinary length. The plane is the set of representable value functions. The best fit $\Pi V^\pi$ is the foot of the perpendicular from $V^\pi$. The TD solution is a different point of the plane: the one whose Bellman backup $\mathcal{T}^\pi\Phi\mathbf{w}_{\mathrm{TD}}$ projects straight back onto it. Here the TD error $\lVert\Phi\mathbf{w}_{\mathrm{TD}}-V^\pi\rVert_D=2.54$ is about twice the best possible error, $1.26$.*

> [!warning] TD does not converge to the best fit
> $\mathbf{w}_{\mathrm{TD}}\neq\mathbf{w}_{\mathrm{MSE}}$ in general. The two coincide in two guaranteed cases:
> - **$\gamma=0$.** Then $V^\pi=r_\pi$, $A=\Phi^\top D\Phi$ and $\mathbf{b}=\Phi^\top DV^\pi$, so $\mathbf{w}_{\mathrm{TD}}$ solves the normal equations (1).
> - **$V^\pi\in\mathcal{V}$**, say $V^\pi=\Phi\bar{\mathbf{w}}$. Then $\Pi\mathcal{T}^\pi\Phi\bar{\mathbf{w}}=\Pi V^\pi=\Phi\bar{\mathbf{w}}$, so $\Phi\bar{\mathbf{w}}$ solves (3), and by uniqueness TD recovers $V^\pi$ exactly.
>
> Otherwise TD minimizes no error at all. It solves the fixed-point equation (3), which mixes the approximation error at the next state into the target at the current one.

---

## How bad is the TD solution?

Since $\Phi\mathbf{w}_{\mathrm{MSE}}$ is the best point in $\mathcal{V}$, trivially $\lVert\Phi\mathbf{w}_{\mathrm{TD}}-V^\pi\rVert_D\ge\lVert\Pi V^\pi-V^\pi\rVert_D$. The contraction property bounds how much worse it can be.

> [!note] Theorem 3 (error bounds)
> $$
> \begin{aligned}
> \lVert\Pi V^\pi-V^\pi\rVert_D\;\le\;\lVert\Phi\mathbf{w}_{\mathrm{TD}}-V^\pi\rVert_D&\le\frac{1}{\sqrt{1-\gamma^2}}\,\lVert\Pi V^\pi-V^\pi\rVert_D\\
> &\le\frac{1}{1-\gamma}\,\lVert\Pi V^\pi-V^\pi\rVert_D .
> \end{aligned}
> $$

> [!proof]- Proof of Theorem 3
> **The $1/(1-\gamma)$ bound, in one line.** Use the triangle inequality, then $\Phi\mathbf{w}_{\mathrm{TD}}=\Pi\mathcal{T}^\pi\Phi\mathbf{w}_{\mathrm{TD}}$ and $\Pi V^\pi=\Pi\mathcal{T}^\pi V^\pi$, then Lemma 2:
>
> $$
> \begin{aligned}
> \lVert\Phi\mathbf{w}_{\mathrm{TD}}-V^\pi\rVert_D&\le\lVert\Phi\mathbf{w}_{\mathrm{TD}}-\Pi V^\pi\rVert_D+\lVert\Pi V^\pi-V^\pi\rVert_D\\
> &=\lVert\Pi\mathcal{T}^\pi\Phi\mathbf{w}_{\mathrm{TD}}-\Pi\mathcal{T}^\pi V^\pi\rVert_D+\lVert\Pi V^\pi-V^\pi\rVert_D\\
> &\le\gamma\lVert\Phi\mathbf{w}_{\mathrm{TD}}-V^\pi\rVert_D+\lVert\Pi V^\pi-V^\pi\rVert_D ,
> \end{aligned}
> $$
>
> and rearranging gives the factor $1/(1-\gamma)$.
>
> **The sharper bound.** Replace the triangle inequality by Pythagoras. $\Phi\mathbf{w}_{\mathrm{TD}}-\Pi V^\pi$ lies in $\mathcal{V}$, while $\Pi V^\pi-V^\pi$ is $D$-orthogonal to $\mathcal{V}$. So
>
> $$
> \begin{aligned}
> \lVert\Phi\mathbf{w}_{\mathrm{TD}}-V^\pi\rVert_D^2&=\lVert\Phi\mathbf{w}_{\mathrm{TD}}-\Pi V^\pi\rVert_D^2+\lVert\Pi V^\pi-V^\pi\rVert_D^2\\
> &\le\gamma^2\lVert\Phi\mathbf{w}_{\mathrm{TD}}-V^\pi\rVert_D^2+\lVert\Pi V^\pi-V^\pi\rVert_D^2 ,
> \end{aligned}
> $$
>
> where the first term was bounded exactly as in the line above. Rearranging gives $1/\sqrt{1-\gamma^2}$. Finally $\sqrt{1-\gamma^2}=\sqrt{(1-\gamma)(1+\gamma)}\ge1-\gamma$.

The sharper bound is due to Tsitsiklis and Van Roy.[^tvr1997] Both bounds blow up as $\gamma\to1$: the more the target looks into the future, the more the approximation error at future states can leak into the current estimate. For $\gamma=0.99$ the guarantee is only $7.1\times$ the best error. These are worst-case bounds; on a given problem the gap is often much smaller.

### A simulation

To see all of this in action, take a **20-state random walk**. From every state the walk moves left or right with probability $\tfrac12$, and a move into either end wall leaves it in place. The reward is $-1$ for each step spent in the leftmost state and $+1$ in the rightmost one, and $\gamma=0.95$. The stationary distribution is uniform. The features are **state aggregation**: four groups of five neighbouring states, each with its own one-hot feature, so the approximation is piecewise constant.

![[Reinforcement Learning/images/linear-function-approximation-td-walk.svg]]
*Figure 2: linear TD(0) on the 20-state walk. Left: the TD fixed point (orange) is visibly different from the best fit (blue). Within the outer groups it is pulled towards the inner groups' values, because each TD target bootstraps from the next state's estimate. TD(0) after $2\times10^5$ steps (dashed, mean of 30 runs, $\alpha_t=0.1(1+t/1000)^{-0.7}$) sits on the fixed point. Right: median and 10–90% band over 30 runs. The distance to $\mathbf{w}_{\mathrm{TD}}$ keeps shrinking (0.13 at the end), while the distance to $\mathbf{w}_{\mathrm{MSE}}$ levels off at $\lVert\Phi(\mathbf{w}_{\mathrm{TD}}-\mathbf{w}_{\mathrm{MSE}})\rVert_D=1.05$ (dotted line). LSTD (green) reaches the same fixed point with smaller error (0.05) and no step size.*

The numbers: the best fit has root-mean-squared error $\lVert\Pi V^\pi-V^\pi\rVert_D=1.04$ and the TD fixed point $1.48$, a ratio of $1.42$. That is well inside the guarantee $1/\sqrt{1-0.95^2}=3.2$, and far inside $1/(1-\gamma)=20$. Figure 3 repeats the comparison over $\gamma$. The ratio grows from exactly 1 at $\gamma=0$ to $1.14$ at $\gamma=0.9$ and $2.6$ at $\gamma=0.99$.

![[Reinforcement Learning/images/linear-function-approximation-gamma.svg]]
*Figure 3: error of the TD($\lambda$) fixed point relative to the best possible error, on the 20-state walk, as a function of $\gamma$ (exact, computed from the model). TD(0) degrades as $\gamma\to1$ but stays far below both worst-case bounds. Larger $\lambda$ closes the gap; $\lambda=1$ is the best fit itself.*

---

## TD($\lambda$) with linear features

TD($\lambda$) interpolates between the one-step bootstrap of TD(0) and the full return of Monte Carlo. With linear features it uses an **eligibility trace** $\mathbf{z}_t\in\mathbb{R}^d$:

$$
\mathbf{z}_t=\gamma\lambda\,\mathbf{z}_{t-1}+\boldsymbol\phi(s_t),\qquad \mathbf{w}_{t+1}=\mathbf{w}_t+\alpha_t\,\delta_t\,\mathbf{z}_t ,
$$

with $\mathbf{z}_{-1}=0$ and $\delta_t$ the same TD error as before. The analysis is the same as for TD(0), with $\mathcal{T}^\pi$ replaced by the **$\lambda$-return operator**

$$
\mathcal{T}^{(\lambda)}V=(1-\lambda)\sum_{k\ge0}\lambda^k(\mathcal{T}^\pi)^{k+1}V=(I-\gamma\lambda P^\pi)^{-1}\big(r_\pi+\gamma(1-\lambda)P^\pi V\big).
$$

By Lemma 2, each $(\mathcal{T}^\pi)^{k+1}$ is a $\gamma^{k+1}$-contraction in $\lVert\cdot\rVert_D$, so $\mathcal{T}^{(\lambda)}$ is a contraction with modulus $\kappa=(1-\lambda)\sum_k\lambda^k\gamma^{k+1}=\frac{\gamma(1-\lambda)}{1-\gamma\lambda}\le\gamma$. TD($\lambda$) converges with probability one to the fixed point $\Phi\mathbf{w}_\lambda=\Pi\mathcal{T}^{(\lambda)}\Phi\mathbf{w}_\lambda$. The proof of Theorem 3 with $\kappa$ in place of $\gamma$ gives the bound of Tsitsiklis and Van Roy:[^tvr1997]

$$
\lVert\Phi\mathbf{w}_\lambda-V^\pi\rVert_D\le\frac{1}{1-\kappa}\lVert\Pi V^\pi-V^\pi\rVert_D=\frac{1-\lambda\gamma}{1-\gamma}\,\lVert\Pi V^\pi-V^\pi\rVert_D .
$$

At $\lambda=1$ the bound is 1: TD(1) finds the best fit, as gradient Monte Carlo does. Smaller $\lambda$ trades a worse fixed point for lower-variance, faster learning. On the walk at $\gamma=0.99$ the ratio drops from $2.57$ ($\lambda=0$) to $2.03$ ($\lambda=0.5$) and $1.18$ ($\lambda=0.9$); see Figure 3.

---

## LSTD: solving $A\mathbf{w}=\mathbf{b}$ directly

TD(0) solves $A\mathbf{w}=\mathbf{b}$ by stochastic approximation, one noisy step at a time. But $A$ and $\mathbf{b}$ are themselves expectations when $s_t\sim d_\pi$:

$$
A=\mathbb{E}\Big[\boldsymbol\phi(s_t)\big(\boldsymbol\phi(s_t)-\gamma\boldsymbol\phi(s_{t+1})\big)^\top\Big],\qquad \mathbf{b}=\mathbb{E}\big[r_t\,\boldsymbol\phi(s_t)\big],
$$

**Least-squares TD (LSTD)**[^bradtkebarto1996] estimates them by sample averages and solves the linear system:

$$
\begin{aligned}
\hat A_T&=\frac1T\sum_{t=0}^{T-1}\boldsymbol\phi(s_t)\big(\boldsymbol\phi(s_t)-\gamma\boldsymbol\phi(s_{t+1})\big)^\top,\\
\hat{\mathbf{b}}_T&=\frac1T\sum_{t=0}^{T-1}r_t\,\boldsymbol\phi(s_t),\qquad \hat{\mathbf{w}}_T=\hat A_T^{-1}\hat{\mathbf{b}}_T .
\end{aligned}
$$

By the ergodic theorem $\hat A_T\to A$ and $\hat{\mathbf{b}}_T\to\mathbf{b}$ almost surely, so $\hat{\mathbf{w}}_T\to\mathbf{w}_{\mathrm{TD}}$. LSTD has the **same fixed point** as TD(0); it reaches it differently.

- **Pros.** There is no step size to tune. It uses every sample fully, which is usually far more data-efficient (Figure 2, right).
- **Cons.** It needs $O(d^2)$ memory, and $O(d^2)$ time per step when $\hat A_T^{-1}$ is updated recursively with the Sherman–Morrison formula (or $O(d^3)$ for a batch solve), against $O(d)$ for TD. A small ridge term $\epsilon I$ is commonly added to $\hat A_T$ early on, before it becomes invertible.

LSTD($\lambda$) replaces $\boldsymbol\phi(s_t)$ by the trace $\mathbf{z}_t$ in the first factor.[^boyan2002]

---

## Key takeaways

- With $d\ll|\mathcal{S}|$ features, only value functions in the column space $\mathcal{V}$ of $\Phi$ can be represented. Features fix which ones; the bike-share features $(N_1,N_2)$ can only express values affine in $N_1$.
- The natural error weights states by the stationary distribution: $\overline{\mathrm{VE}}=\lVert V^\pi-\Phi\mathbf{w}\rVert_D^2$, minimized by the $D$-projection $\Pi V^\pi=\Phi\mathbf{w}_{\mathrm{MSE}}$.
- Linear TD(0) bootstraps, so it is a **semi-gradient** method: its expected update $\mathbf{b}-A\mathbf{w}$ is not the gradient of any function.
- On-policy, $A=\Phi^\top D(I-\gamma P^\pi)\Phi$ is positive definite, because $\lVert P^\pi z\rVert_D\le\lVert z\rVert_D$ when $D$ is stationary. Hence TD(0) converges almost surely to $\mathbf{w}_{\mathrm{TD}}=A^{-1}\mathbf{b}$.
- $\Phi\mathbf{w}_{\mathrm{TD}}$ is the unique fixed point of the projected Bellman operator $\Pi\mathcal{T}^\pi$, a $\gamma$-contraction in $\lVert\cdot\rVert_D$. It is **not** the best fit unless $\gamma=0$ or $V^\pi\in\mathcal{V}$, but its error is at most $1/\sqrt{1-\gamma^2}$ times the best. For TD($\lambda$) the factor is at most $(1-\lambda\gamma)/(1-\gamma)$.
- LSTD reaches the same fixed point by solving $\hat A\mathbf{w}=\hat{\mathbf{b}}$, trading $O(d^2)$ cost per step for no step sizes and better data efficiency.

## Where this shows up next

- [[Off-Policy Learning with Function Approximation]]: when states are sampled from a different policy's distribution, Lemma 1 fails, $A$ can lose positive definiteness, and TD can diverge. This is the "deadly triad".
- [[Actor-Critic Algorithms]]: a critic with linear features is exactly linear TD (or LSTD) run on the actor's current policy. Everything here, including the bias of $\mathbf{w}_{\mathrm{TD}}$ relative to the best fit, applies to it.
- Part II of the series assumes the model is known: [[Finite-Horizon Dynamic Programming]], [[Dynamic Programming]] (where the sup-norm versions of these contraction arguments are proved), [[Stochastic Shortest Path Problems]], and [[Model-Based Reinforcement Learning]].
- [[Stochastic Approximation]]: the ODE method and the Markov-noise convergence theorem used in this lecture.

<!-- References: footnote definitions below render as the numbered "References" list at the bottom of the page -->

[^tvr1997]: Tsitsiklis, J. N., & Van Roy, B. (1997). *An Analysis of Temporal-Difference Learning with Function Approximation*. IEEE Transactions on Automatic Control, 42(5), 674–690. [doi:10.1109/9.580874](https://doi.org/10.1109/9.580874)
[^suttonbarto]: Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.), Example 4.2 and Chapter 9. MIT Press. [book website](http://incompleteideas.net/book/the-book-2nd.html)
[^sutton1988]: Sutton, R. S. (1988). *Learning to Predict by the Methods of Temporal Differences*. Machine Learning, 3, 9–44. [doi:10.1007/BF00115009](https://doi.org/10.1007/BF00115009)
[^borkar2008]: Borkar, V. S. (2008). *Stochastic Approximation: A Dynamical Systems Viewpoint*. Hindustan Book Agency / Cambridge University Press, Chapter 6 (Markov noise). [doi:10.1007/978-93-86279-38-5](https://doi.org/10.1007/978-93-86279-38-5)
[^borkarmeyn2000]: Borkar, V. S., & Meyn, S. P. (2000). *The O.D.E. Method for Convergence of Stochastic Approximation and Reinforcement Learning*. SIAM Journal on Control and Optimization, 38(2), 447–469. [doi:10.1137/S0363012997331639](https://doi.org/10.1137/S0363012997331639)
[^bradtkebarto1996]: Bradtke, S. J., & Barto, A. G. (1996). *Linear Least-Squares Algorithms for Temporal Difference Learning*. Machine Learning, 22, 33–57. [doi:10.1007/BF00114723](https://doi.org/10.1007/BF00114723)
[^boyan2002]: Boyan, J. A. (2002). *Technical Update: Least-Squares Temporal Difference Learning*. Machine Learning, 49, 233–246. [doi:10.1023/A:1017936530646](https://doi.org/10.1023/A:1017936530646)
