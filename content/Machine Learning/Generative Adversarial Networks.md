---
title: Generative Adversarial Networks
tags:
  - generative-models
  - gan
  - f-divergence
  - convex-duality
  - representation-learning
date: 2026-11-12
draft: false
prev: "[[Variational Autoencoders]]"
next: "[[Energy-Based Models]]"
---

> **Prerequisites:** [[Variational Autoencoders]] (latent-variable generators, forward vs reverse KL) and the Lagrangian/duality aside in [[Support Vector Machines]] (conjugates and lower bounds are the same idea).
>
> **TL;DR:**
> - Instead of maximizing likelihood, a GAN trains a **sampler** $G_\theta$ to minimize a divergence between the data and its samples. It never evaluates a density.
> - Any **f-divergence** has a variational lower bound, obtained from the **Fenchel conjugate** $f^*$. The bound is an expectation over real samples minus an expectation over fake samples, and a **critic** network $T_\omega$ maximizes it.
> - Minimizing the bound over $\theta$ and maximizing over $\omega$ gives a saddle-point problem: adversarial training. One particular $f$ gives the original GAN, with optimal discriminator $D^*=p/(p+q)$ and value $2\,\mathrm{JSD}-\log4$.
> - In practice: the non-saturating generator loss, mode collapse, and the Wasserstein GAN for when the supports don't overlap.
>
> **Code:** [Image-Based-Generative-Models](https://github.com/parag1604/Image-Based-Generative-Models): a DCGAN, alongside a VAE and a DDPM, trained on 64×64 Stanford Cars images.

## Motivation

A [[Variational Autoencoders|VAE]] is a likelihood-based model. It specifies a density $p_\theta(\mathbf{x})$, bounds its log, and maximizes the bound. Two things about this are unsatisfying:
- we must choose a decoder likelihood (a Gaussian, a Bernoulli), and that choice shapes the samples. A Gaussian decoder gives blurry images;
- maximum likelihood fixes the divergence: it is the forward KL, which is mode-covering.

What if we only wanted a machine that **produces samples**, $\mathbf{x}=G_\theta(\mathbf{z})$ with $\mathbf{z}\sim\mathcal{N}(\mathbf{0},\mathbf{I})$, and were free to pick the divergence that measures how far its samples are from the data? Such a model is **implicit**: we can sample from its distribution $Q_\theta$ but we cannot evaluate $q_\theta(\mathbf{x})$. The question is how to train it, when the only thing we can do with both distributions is draw samples. The answer is to learn the divergence itself, with a second network.[^goodfellow2014]<sup>,</sup>[^nowozin2016]

These notes reach the GAN through **variational divergence minimization** (the f-GAN view of Nowozin et al.), and recover the original GAN of Goodfellow et al. as one special case.

## Why not just maximize likelihood?

Maximizing the data log-likelihood is the same as minimizing the KL divergence from the data distribution $p_X$ to the model $p_\theta$:

$$
D_{\mathrm{KL}}(p_X\,\|\,p_\theta)=\int p_X(\mathbf{x})\log\frac{p_X(\mathbf{x})}{p_\theta(\mathbf{x})}\,d\mathbf{x}.
$$

Look at the two kinds of mistakes the model can make.
- **Case 1:** on a region $X_1$ where $p_X(\mathbf{x})\gg p_\theta(\mathbf{x})$ (real data the model misses), the log ratio is large and positive, so $D_{\mathrm{KL}}$ is high. This is good: missing data is penalized.
- **Case 2:** on a region $X_2$ where $p_X(\mathbf{x})\ll p_\theta(\mathbf{x})$ (the model generates things that never occur in the data), the integrand is weighted by $p_X\approx0$, so the contribution is low. This is not desirable.

So $D_{\mathrm{KL}}(p_X\,\|\,p_\theta)$ has no "mechanism" to discourage the model from sampling in regions of low data density. Figure 1 shows the consequence for a model that is too simple: the forward KL covers both modes and puts much of its mass in the empty valley between them.

![[Machine Learning/images/generative-forward-reverse-kl.svg]]
*Figure 1: fitting one Gaussian to a bimodal target. Left: forward KL (maximum likelihood) covers everything, including the empty valley. Right: reverse KL commits to a single mode and never generates in the valley.*

For generating realistic samples, Case 2 is precisely the failure we care about: a sample from the valley is unrealistic. This motivates looking at a whole family of divergences, some of which penalize Case 2 more heavily. There is a second, practical motivation: as we will see, the f-divergences can be minimized using **only samples** from the model, so the generator needs no tractable density.

## f-divergences

**Definition.** Let $f:\mathbb{R}_{+}\to\mathbb{R}$ be convex and lower semicontinuous, with $f(1)=0$. The **f-divergence** between distributions $P$ and $Q$ with densities $p$ and $q$ is

$$
D_f(P\,\|\,Q)=\int q(\mathbf{x})\,f\!\left(\frac{p(\mathbf{x})}{q(\mathbf{x})}\right)d\mathbf{x}.
$$

Jensen's inequality gives $D_f\ge f\big(\int q\,\tfrac{p}{q}\big)=f(1)=0$, with equality when $P=Q$. Different generators $f$ give familiar divergences:[^nguyen2010]

| generator $f(u)$ | $D_f(P\,\|\,Q)$ |
|---|---|
| $u\log u$ | KL divergence $D_{\mathrm{KL}}(P\,\|\,Q)$ |
| $-\log u$ | reverse KL $D_{\mathrm{KL}}(Q\,\|\,P)$ |
| $(u-1)^2$ | Pearson $\chi^2$ |
| $\frac12\left[u\log u-(u+1)\log\frac{u+1}{2}\right]$ | Jensen–Shannon $\mathrm{JSD}(P\,\|\,Q)$ |
| $u\log u-(u+1)\log(u+1)$ | the GAN divergence, $2\,\mathrm{JSD}(P\,\|\,Q)-\log4$ (here $f(1)=-\log4\ne0$) |

Here $\mathrm{JSD}(P\,\|\,Q)=\tfrac12D_{\mathrm{KL}}(P\,\|\,M)+\tfrac12D_{\mathrm{KL}}(Q\,\|\,M)$, with $M=\frac{P+Q}{2}$. Without the factor $\tfrac12$, the JS generator gives $2\,\mathrm{JSD}$. The JS divergence is symmetric and bounded by $\log2$, and it is less mode-covering than the forward KL.

> [!warning] Divergences, not metrics
> f-divergences are generally **not** distances. KL is not symmetric, and none of them satisfies the triangle inequality ($\sqrt{\mathrm{JSD}}$ is a notable exception that does). They only promise $D_f\ge0$, with equality iff $P=Q$ (for strictly convex $f$).

The trouble is that $D_f$ needs the density ratio $p/q$, and we have neither density. Convex duality will remove that need.

## The Fenchel conjugate

> [!note] ASIDE: Fenchel conjugates
> The **convex conjugate** (Fenchel conjugate) of a function $f$ is
>
> $$
> f^*(t)=\sup_{u\in\operatorname{dom}f}\big\{tu-f(u)\big\}.
> $$
>
> Here $\sup$ is the **supremum**, the least upper bound of a set. For example, $11,12,\dots$ are all upper bounds of $\{1,2,\dots,10\}$, and the least of them is $10$. Unlike $\max$, the supremum exists even when it is not attained.
>
> Geometrically, $f^*(t)$ is the largest gap between the line $tu$ and the curve $f(u)$. Equivalently, $tu-f^*(t)$ is the line of slope $t$ that just touches $f$ from below (Figure 2). The conjugate is defined for any $f$, and it is always convex. If $f$ is convex **and lower semicontinuous** (closed), conjugating twice gives back $f$:
>
> $$
> f(u)=f^{**}(u)=\sup_{t\in\operatorname{dom}f^*}\big\{tu-f^*(t)\big\}.
> \tag{1}
> $$
>
> A convex function is the upper envelope of all the lines that lie below it.
>
> For differentiable, strictly convex $f$, the supremum is attained where $t=f'(u)$. Hence the derivatives of $f$ and $f^*$ are **inverse functions**: $(f^*)'=(f')^{-1}$.

![[Machine Learning/images/gan-fenchel-conjugate.svg]]
*Figure 2: left, the conjugate of $f(u)=u\log u$ at slope $t=1$. The largest gap between $tu$ and $f$ occurs where $f'(u)=t$, i.e. at $u=e^{t-1}=1$, and equals $f^*(1)=1$. The dashed line, of slope $t$ and intercept $-f^*(t)$, touches $f$ from below. Right: two conjugates, $e^{t-1}$ for KL, and $-\log(1-e^{t})$ for the GAN generator, which is only finite for $t<0$.*

## A variational lower bound on $D_f$

Substitute the biconjugate (1) for $f$, with $u=p(\mathbf{x})/q(\mathbf{x})$ a scalar for each $\mathbf{x}$:

$$
D_f(P\,\|\,Q)=\int q(\mathbf{x})\sup_{t\in\operatorname{dom}f^*}\left\{t\,\frac{p(\mathbf{x})}{q(\mathbf{x})}-f^*(t)\right\}d\mathbf{x}.
$$

The supremum sits inside the integral, so it is taken pointwise: every $\mathbf{x}$ gets its own best $t$. If we move the supremum outside, it must range over **functions** $T:\mathcal{X}\to\mathbb{R}$, one value per $\mathbf{x}$. Restricting $T$ to a class $\mathcal{T}$ (say, a neural network family) can only make the result smaller, because $\mathcal{T}$ may not contain the pointwise solutions:

$$
\begin{aligned}
D_f(P\,\|\,Q)&\ge\sup_{T\in\mathcal{T}}\int q(\mathbf{x})\left\{T(\mathbf{x})\frac{p(\mathbf{x})}{q(\mathbf{x})}-f^*\big(T(\mathbf{x})\big)\right\}d\mathbf{x}\\
&=\sup_{T\in\mathcal{T}}\Big(\mathbb{E}_{\mathbf{x}\sim P}\big[T(\mathbf{x})\big]-\mathbb{E}_{\mathbf{x}\sim Q}\big[f^*\big(T(\mathbf{x})\big)\big]\Big).
\end{aligned}
\tag{2}
$$

This is the key step. The right-hand side contains **no densities**, only expectations: one over real data and one over model samples. Both can be estimated by Monte Carlo.

**When is the bound tight?** Suppose $\mathcal{T}$ contains all functions. The integrand of (2) is $L\big(T(\mathbf{x})\big)=p(\mathbf{x})T(\mathbf{x})-q(\mathbf{x})f^*\big(T(\mathbf{x})\big)$. It involves no derivative of $T$, so the variational problem decouples into independent scalar problems, one per $\mathbf{x}$. (The Euler–Lagrange equation reduces to $\partial L/\partial T=0$.) Setting the derivative to zero:

$$
\begin{aligned}
&p(\mathbf{x})-q(\mathbf{x})\,(f^*)'\big(T(\mathbf{x})\big)=0\\
\Longrightarrow\;&(f^*)'\big(T(\mathbf{x})\big)=\frac{p(\mathbf{x})}{q(\mathbf{x})}\\
\Longrightarrow\;&T^*(\mathbf{x})=f'\!\left(\frac{p(\mathbf{x})}{q(\mathbf{x})}\right),
\end{aligned}
$$

using $(f^*)'=(f')^{-1}$. At $T^*$ the bound is **tight**: (2) holds with equality, and the supremum *equals* $D_f(P\,\|\,Q)$. (It does not make $D_f$ zero; that only happens when $P=Q$.) The optimal critic is a known function of the density ratio, so a trained critic is also a **density-ratio estimator**.[^nguyen2010]

## Variational divergence minimization (f-GAN)

We now have a lower bound on the divergence, and want a sampler that **minimizes** it.

**The critic.** The critic's outputs must lie in the domain of $f^*$ (for the GAN conjugate, $t<0$). So we write $T_\omega(\mathbf{x})=g_f\big(V_\omega(\mathbf{x})\big)$, where:
- $V_\omega:\mathcal{X}\to\mathbb{R}$ is an unconstrained neural network;
- $g_f:\mathbb{R}\to\operatorname{dom}f^*$ is an output activation specific to the chosen divergence.

**The generator.** $G_\theta$ is a neural network that maps noise $\mathbf{z}\sim p(\mathbf{z})$ to samples $G_\theta(\mathbf{z})$; their distribution is $Q_\theta$.

Plugging both into (2) gives the f-GAN objective:

$$
F(\theta,\omega)=\mathbb{E}_{\mathbf{x}\sim P_X}\Big[g_f\big(V_\omega(\mathbf{x})\big)\Big]-\mathbb{E}_{\mathbf{z}\sim p(\mathbf{z})}\Big[f^*\Big(g_f\big(V_\omega(G_\theta(\mathbf{z}))\big)\Big)\Big].
$$

The optimal parameters solve the saddle-point problem

$$
\theta^*,\omega^*=\arg\min_\theta\max_\omega F(\theta,\omega).
\tag{3}
$$

The critic maximizes $F$ to make the bound as tight as possible, i.e. to *estimate* the divergence. The generator minimizes $F$ to *reduce* the estimated divergence. Since the two networks respectively minimize and maximize the same objective, the procedure is called **adversarial learning**, and the pair is a **generative adversarial network**. The critic is traditionally called the **discriminator**.

![[Machine Learning/images/gan-architecture.svg]]
*Figure 3: the f-GAN. The generator turns noise into fake samples. The critic $V_\omega$ scores real and fake samples with a real number, and $g_f$ maps that score into the domain of $f^*$. Both networks play the same objective $F(\theta,\omega)$ with opposite goals.*

A few divergences and their activations:[^nowozin2016]

| divergence | $g_f(v)$ | $f^*(t)$ |
|---|---|---|
| KL | $v$ | $e^{t-1}$ |
| reverse KL | $-e^{-v}$ | $-1-\log(-t)$ |
| Pearson $\chi^2$ | $v$ | $\frac14t^2+t$ |
| GAN | $-\log(1+e^{-v})$ | $-\log(1-e^{t})$ |

## The original GAN as a special case

Let's consider the following symmetric divergence, from the last row of the f-divergence table:

$$
f(u)=u\log u-(u+1)\log(u+1).
$$

**Step 1: the derivative and the optimal critic.**

$$
f'(u)=\log u+1-\log(u+1)-1=\log\frac{u}{u+1}.
$$

With $u=p/q$:

$$
T^*(\mathbf{x})=f'\!\left(\frac{p(\mathbf{x})}{q(\mathbf{x})}\right)=\log\frac{p(\mathbf{x})}{p(\mathbf{x})+q(\mathbf{x})}.
$$

**Step 2: the conjugate.** Maximize $tu-f(u)$ over $u$. The stationary point is $f'(u)=t$, i.e. $\frac{u}{u+1}=e^t$, i.e. $u=\frac{e^t}{1-e^t}$, which needs $t<0$. Substituting back and simplifying gives

$$
f^*(t)=-\log\big(1-e^{t}\big),\qquad\operatorname{dom}f^*=(-\infty,0).
$$

**Step 3: the activation and the discriminator.** We need $g_f:\mathbb{R}\to(-\infty,0)$. Take $g_f(v)=-\log(1+e^{-v})=\log\sigma(v)$, where $\sigma$ is the [[Linear Models|sigmoid]]. Define the **discriminator** $D_\omega(\mathbf{x})=\sigma\big(V_\omega(\mathbf{x})\big)\in(0,1)$. Then:
- $T_\omega(\mathbf{x})=\log D_\omega(\mathbf{x})$;
- $-f^*\big(T_\omega(\mathbf{x})\big)=\log\big(1-e^{\log D_\omega(\mathbf{x})}\big)=\log\big(1-D_\omega(\mathbf{x})\big)$.

**Step 4: the objective.** Substituting into $F$:

$$
\min_\theta\max_\omega\;\mathbb{E}_{\mathbf{x}\sim P_X}\big[\log D_\omega(\mathbf{x})\big]+\mathbb{E}_{\mathbf{z}\sim p(\mathbf{z})}\Big[\log\Big(1-D_\omega\big(G_\theta(\mathbf{z})\big)\Big)\Big].
\tag{4}
$$

This is exactly the objective of Goodfellow et al.[^goodfellow2014] The discriminator maximizes the log-likelihood of a binary classifier that labels real data 1 and fakes 0 (the [[Linear Models|binary cross-entropy]]). The generator tries to make that classifier fail.

**Step 5: the optimal discriminator.** From Step 1, $\log D^*=T^*=\log\frac{p}{p+q}$, so

$$
D^*(\mathbf{x})=\frac{p_X(\mathbf{x})}{p_X(\mathbf{x})+q_\theta(\mathbf{x})}.
$$

This is the Bayes-optimal classifier for "real or fake" with equal priors. Where the generator over-produces, $D^*<\tfrac12$; where it under-produces, $D^*>\tfrac12$.

![[Machine Learning/images/gan-optimal-discriminator.svg]]
*Figure 4: the optimal discriminator for a fixed generator. $D^*$ is near 1 where only real data lives, near 0 where only fakes live, and exactly $\tfrac12$ where the two densities are equal. Its gradient tells the generator where to move its mass.*

**Step 6: the value at the optimum.** Substituting $D^*$ into (4):

$$
\begin{aligned}
\max_\omega F
&=\int p\log\frac{p}{p+q}\,d\mathbf{x}+\int q\log\frac{q}{p+q}\,d\mathbf{x}\\
&=D_{\mathrm{KL}}\!\left(P\,\Big\|\,\tfrac{P+Q}{2}\right)+D_{\mathrm{KL}}\!\left(Q\,\Big\|\,\tfrac{P+Q}{2}\right)-2\log2\\
&=2\,\mathrm{JSD}(P\,\|\,Q_\theta)-\log4.
\end{aligned}
$$

The $-\log4$ is there because this generator violates the normalization: $f(1)=-2\log2\ne0$. It is a constant, so it changes nothing. With an optimal discriminator, the generator minimizes the Jensen–Shannon divergence, and the **global optimum** is $Q_\theta=P_X$, where $D^*\equiv\tfrac12$ and the value is $-\log4$.

## Training a GAN

The saddle point (3) is found by alternating gradient steps.

> [!example] Algorithm: GAN training (alternating SGD)
> 1. Sample a minibatch $\{\mathbf{x}_i\}_{i=1}^B$ of real data and $\{\mathbf{z}_i\}_{i=1}^B$ of noise.
> 2. Update the discriminator: ascend $\frac1B\sum_i\big[\log D_\omega(\mathbf{x}_i)+\log\big(1-D_\omega(G_\theta(\mathbf{z}_i))\big)\big]$ in $\omega$. Repeat steps 1–2 for $k$ steps (often $k=1$).
> 3. Update the generator: sample fresh noise $\{\mathbf{z}_i\}$ and descend the generator loss in $\theta$ (in practice, the non-saturating loss below).
> 4. Repeat for a fixed number of iterations, monitoring sample quality.

Nowozin et al. also showed that a single *simultaneous* gradient step on both players converges locally to the saddle point.[^nowozin2016] In practice, both versions use Adam with carefully chosen learning rates.

### The non-saturating generator loss

Early in training the generator is poor, and the discriminator rejects its samples confidently: $D(G(\mathbf{z}))\approx0$. In that region $\log(1-D)$ is almost flat (Figure 5), so the gradient the generator receives from (4) **vanishes** exactly when it needs it most.

The standard fix keeps the same fixed point but changes the generator's loss:

$$
\begin{aligned}
&\text{minimize}\quad\mathbb{E}_{\mathbf{z}}\Big[\log\Big(1-D\big(G_\theta(\mathbf{z})\big)\Big)\Big]\\
\longrightarrow\quad&\text{minimize}\quad-\mathbb{E}_{\mathbf{z}}\Big[\log D\big(G_\theta(\mathbf{z})\big)\Big].
\end{aligned}
$$

Both losses push $D(G(\mathbf{z}))$ up. But $-\log D$ is steep near $D=0$, so a bad generator gets strong gradients.[^goodfellow2014]

![[Machine Learning/images/gan-generator-losses.svg]]
*Figure 5: the two generator losses as a function of the discriminator's output on a fake sample. Both decrease towards $D=1$. The saturating loss is nearly flat in the shaded region where training starts, while the non-saturating loss is steepest there.*

### Mode collapse and instability

- **Mode collapse.** The generator maps many different $\mathbf{z}$ to a few outputs that currently fool the discriminator. The discriminator learns to reject them, and the generator jumps to another few. Minimax ($\min_\theta\max_\omega$) is not the same as maximin, and alternating updates do not always approximate the inner maximum well. Remedies include minibatch discrimination and feature matching.[^salimans2016]
- **No convergence guarantee.** Gradient descent on a saddle-point problem can oscillate or spiral, rather than converge. The loss curves say little about sample quality.
- **Disjoint supports.** Real images lie near a low-dimensional manifold, and so do generated ones. Two such manifolds generically do not overlap. Then a perfect discriminator exists, $\mathrm{JSD}=\log2$ whatever the generator does, and the gradient is zero. This is the motivation for the next section.

## The Wasserstein GAN

**Arjovsky's example.** Let $Z\sim U[0,1]$. Let the data be $P=(0,Z)$, a vertical line segment at $x=0$, and the model $Q_\theta=(\theta,Z)$, the same segment shifted to $x=\theta$.[^arjovsky2017] For every $\theta\ne0$ the supports are disjoint, so:
- $D_{\mathrm{KL}}(P\,\|\,Q_\theta)=\infty$;
- $\mathrm{JSD}(P\,\|\,Q_\theta)=\log2$, a constant with zero gradient in $\theta$.

Yet the model is clearly getting better as $\theta\to0$. A useful distance should say so.

The **Wasserstein-1** (earth mover's) distance measures how much mass must be moved, and how far:

$$
W_1(P,Q)=\inf_{\gamma\in\Pi(P,Q)}\mathbb{E}_{(\mathbf{x},\mathbf{y})\sim\gamma}\big[\lVert\mathbf{x}-\mathbf{y}\rVert\big],
$$

where $\Pi(P,Q)$ is the set of joint distributions (couplings) with marginals $P$ and $Q$. In the example, $W_1(P,Q_\theta)=|\theta|$, which is continuous and has a useful gradient everywhere.

![[Machine Learning/images/gan-js-vs-wasserstein.svg]]
*Figure 6: two parallel line segments a distance $\theta$ apart. The JS divergence is $\log2$ for every $\theta\ne0$ and drops to 0 only at $\theta=0$, so it gives the generator no gradient. The Wasserstein distance $|\theta|$ shrinks smoothly as the generator improves.*

**A critic for $W_1$.** The coupling form is intractable, but the **Kantorovich–Rubinstein duality** turns it into a supremum over critics, much like (2):

$$
W_1(P,Q)=\sup_{\lVert T\rVert_L\le1}\Big(\mathbb{E}_{\mathbf{x}\sim P}\big[T(\mathbf{x})\big]-\mathbb{E}_{\mathbf{x}\sim Q}\big[T(\mathbf{x})\big]\Big),
$$

where the supremum is over **1-Lipschitz** functions. Compare with (2): there is no $f^*$, and the constraint on $T$ is smoothness, not a range. The WGAN trains a critic network subject to this constraint, and the generator minimizes the estimated $W_1$. Two ways to enforce the Lipschitz constraint:
- **weight clipping** (the original WGAN): clamp every critic weight to $[-c,c]$. It is crude, and can make the critic too simple;
- **gradient penalty** (WGAN-GP): add $\lambda\,\mathbb{E}_{\hat{\mathbf{x}}}\big[(\lVert\nabla_{\hat{\mathbf{x}}}T(\hat{\mathbf{x}})\rVert-1)^2\big]$ at points $\hat{\mathbf{x}}$ interpolated between real and fake samples.[^gulrajani2017]

$W_1$ is an **integral probability metric**, not an f-divergence. It is a genuine metric, and it stays informative when the supports do not overlap.

## VAEs vs GANs

| | VAE | GAN |
|---|---|---|
| model | explicit density $p_\theta(\mathbf{x}\mid\mathbf{z})p(\mathbf{z})$ | implicit sampler $G_\theta(\mathbf{z})$ |
| objective | ELBO (a bound on the likelihood) | a learned divergence (JS, $W_1$, any $D_f$) |
| inference network | yes, the encoder $q_\phi(\mathbf{z}\mid\mathbf{x})$ | no (the critic acts on $\mathbf{x}$) |
| likelihood of a test point | lower bound available | not available |
| training | stable: one objective, maximized | delicate: a saddle point |
| typical samples | blurry, but cover the data | sharp, but may drop modes |

Sample quality is usually judged by the Inception Score[^salimans2016] or the Fréchet Inception Distance, which compare statistics of a pretrained classifier's features on real and generated samples.

## Key takeaways

- A GAN trains an implicit sampler by minimizing a divergence that is itself estimated by a second network. Only samples are needed, never densities.
- Fenchel duality turns any f-divergence into a supremum over critics, $\mathbb{E}_P[T]-\mathbb{E}_Q[f^*(T)]$. The bound is tight at $T^*=f'(p/q)$, using $(f^*)'=(f')^{-1}$.
- The original GAN is the f-GAN for $f(u)=u\log u-(u+1)\log(u+1)$. The discriminator is $D=\sigma(V)$, the optimum is $D^*=p/(p+q)$, and the generator minimizes $2\,\mathrm{JSD}-\log4$.
- In practice: train the generator with $-\log D(G(\mathbf{z}))$, expect mode collapse and oscillation, and use a Wasserstein critic (with a gradient penalty) when supports barely overlap.

## Where this shows up next

- [[Variational Autoencoders]]: the likelihood-based alternative, and the forward-vs-reverse KL picture used here.
- [[Linear Models]]: the discriminator is a logistic-regression classifier on learned features, trained with cross-entropy.
- [[Support Vector Machines]]: the Lagrangian aside there is the same convex duality that gives the variational bound here.
- [[Policy Gradients]]: the generator gets *pathwise* gradients through the critic, while RL with a non-differentiable reward needs the score function. Adversarial imitation learning (GAIL) combines the two, using a GAN discriminator as the reward for [[Imitation Learning|imitation]].
- [[Energy-Based Models]]: noise-contrastive estimation trains an unnormalized density with a GAN-like classifier against fixed noise.
- [[Denoising Diffusion Models]] reach GAN-level sample quality with a plain regression loss and no adversary, at the price of many network passes per sample. [[Normalizing Flows]] trade the adversary for exact likelihoods; [[Flow Matching]] gives simulation-free training of continuous flows.

<!-- References: footnote definitions below render as the numbered "References" list at the bottom of the page -->

[^goodfellow2014]: Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., & Bengio, Y. (2014). *Generative Adversarial Nets*. NeurIPS. [arXiv:1406.2661](https://arxiv.org/abs/1406.2661)
[^nowozin2016]: Nowozin, S., Cseke, B., & Tomioka, R. (2016). *f-GAN: Training Generative Neural Samplers using Variational Divergence Minimization*. NeurIPS. [arXiv:1606.00709](https://arxiv.org/abs/1606.00709)
[^nguyen2010]: Nguyen, X., Wainwright, M. J., & Jordan, M. I. (2010). *Estimating Divergence Functionals and the Likelihood Ratio by Convex Risk Minimization*. IEEE Transactions on Information Theory, 56(11), 5847–5861. [doi:10.1109/TIT.2010.2068870](https://doi.org/10.1109/TIT.2010.2068870)
[^salimans2016]: Salimans, T., Goodfellow, I., Zaremba, W., Cheung, V., Radford, A., & Chen, X. (2016). *Improved Techniques for Training GANs*. NeurIPS. [arXiv:1606.03498](https://arxiv.org/abs/1606.03498)
[^arjovsky2017]: Arjovsky, M., Chintala, S., & Bottou, L. (2017). *Wasserstein Generative Adversarial Networks*. ICML. [arXiv:1701.07875](https://arxiv.org/abs/1701.07875)
[^gulrajani2017]: Gulrajani, I., Ahmed, F., Arjovsky, M., Dumoulin, V., & Courville, A. (2017). *Improved Training of Wasserstein GANs*. NeurIPS. [arXiv:1704.00028](https://arxiv.org/abs/1704.00028)
