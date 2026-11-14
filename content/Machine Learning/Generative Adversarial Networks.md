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

> **Prerequisites:** [[Variational Autoencoders]] (latent-variable generators, forward vs reverse KL) and [[Multivariable Calculus#The Lagrangian and duality|the Lagrangian and duality]] (conjugates and lower bounds are the same idea).
>
> **TL;DR:**
> - Instead of maximizing likelihood, a GAN trains a **sampler** $G_\theta$ to minimize a divergence between the data and its samples. It never evaluates a density.
> - Any **f-divergence** has a variational lower bound, obtained from the **Fenchel conjugate** $f^*$. The bound is an expectation over real samples minus an expectation over fake samples, and a **critic** network $T_\omega$ maximizes it.
> - Minimizing the bound over $\theta$ and maximizing over $\omega$ gives a saddle-point problem: adversarial training. One particular $f$ gives the original GAN, with optimal discriminator $D^*=p/(p+q)$ and value $2\,\mathrm{JSD}-\log4$.
> - In practice: the non-saturating generator loss, mode collapse, and the Wasserstein GAN for when the supports don't overlap.
> - Variants: **conditional GANs** (cGAN, AC-GAN, InfoGAN, pix2pix, CycleGAN) control what is generated; **DCGAN**, **spectral normalization** and **StyleGAN** are the architectural steps that made training reliable at high resolution.
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
- **Case 1:** on a region $\mathcal{R}_1$ where $p_X(\mathbf{x})\gg p_\theta(\mathbf{x})$ (real data the model misses), the log ratio is large and positive, so $D_{\mathrm{KL}}$ is high. This is good: missing data is penalized.
- **Case 2:** on a region $\mathcal{R}_2$ where $p_X(\mathbf{x})\ll p_\theta(\mathbf{x})$ (the model generates things that never occur in the data), the integrand is weighted by $p_X\approx0$, so the contribution is low. This is not desirable.

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

**When is the bound tight?** Suppose $\mathcal{T}$ contains all functions. The integrand of (2) is $\ell\big(T(\mathbf{x})\big)=p(\mathbf{x})T(\mathbf{x})-q(\mathbf{x})f^*\big(T(\mathbf{x})\big)$. It involves no derivative of $T$, so the variational problem decouples into independent scalar problems, one per $\mathbf{x}$. (The Euler–Lagrange equation reduces to $\partial\ell/\partial T=0$.) Setting the derivative to zero:

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
J(\theta,\omega)=\mathbb{E}_{\mathbf{x}\sim P_X}\Big[g_f\big(V_\omega(\mathbf{x})\big)\Big]-\mathbb{E}_{\mathbf{z}\sim p(\mathbf{z})}\Big[f^*\Big(g_f\big(V_\omega(G_\theta(\mathbf{z}))\big)\Big)\Big].
$$

The optimal parameters solve the saddle-point problem

$$
\theta^*,\omega^*=\arg\min_\theta\max_\omega J(\theta,\omega).
\tag{3}
$$

The critic maximizes $J$ to make the bound as tight as possible, i.e. to *estimate* the divergence. The generator minimizes $J$ to *reduce* the estimated divergence. Since the two networks respectively minimize and maximize the same objective, the procedure is called **adversarial learning**, and the pair is a **generative adversarial network**. The critic is traditionally called the **discriminator**.

![[Machine Learning/images/gan-architecture.svg]]
*Figure 3: the f-GAN. The generator turns noise into fake samples. The critic $V_\omega$ scores real and fake samples with a real number, and $g_f$ maps that score into the domain of $f^*$. Both networks play the same objective $J(\theta,\omega)$ with opposite goals.*

A few divergences and their activations:[^nowozin2016]

| divergence | $g_f(v)$ | $f^*(t)$ |
|---|---|---|
| KL | $v$ | $e^{t-1}$ |
| reverse KL | $-e^{-v}$ | $-1-\log(-t)$ |
| Pearson $\chi^2$ | $v$ | $\frac14t^2+t$ |
| GAN | $-\log(1+e^{-v})$ | $-\log(1-e^{t})$ |

> [!example]- Code: the f-GAN objective for a chosen divergence (PyTorch)
> ```python
> # assumes: v_real = V(x), v_fake = V(G(z)): critic scores (B,)
> fgan = {  # divergence: (g_f, f*), as in the table above
>     "KL":     (lambda v: v, lambda t: torch.exp(t - 1)),
>     "revKL":  (lambda v: -torch.exp(-v),
>                lambda t: -1 - torch.log(-t)),
>     "chi2":   (lambda v: v, lambda t: 0.25 * t ** 2 + t),
>     "GAN":    (lambda v: -F.softplus(-v),
>                lambda t: -torch.log1p(-t.exp())),
> }
> g_f, f_star = fgan["GAN"]
> # J(theta, omega): critic ascends, generator descends
> J_obj = g_f(v_real).mean() - f_star(g_f(v_fake)).mean()
> ```
> `F.softplus(-v)` is $\log(1+e^{-v})$, computed stably. Switching the divergence changes only the pair of functions, not the networks.

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

**Step 4: the objective.** Substituting into $J$:

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
\max_\omega J
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

> [!example]- Code: one GAN training step with the non-saturating loss (PyTorch)
> ```python
> # assumes: G, D (D returns the logit V, so D(x) = sigmoid(V));
> #   optimizers optD, optG; real batch x of size B; noise dim nz
> bce = nn.BCEWithLogitsLoss()
> ones, zeros = torch.ones(B), torch.zeros(B)
> fake = G(torch.randn(B, nz))
> # discriminator: ascend log D(x) + log(1 - D(G(z)))  ==  descend
> #   the BCE with labels 1 (real), 0 (fake)
> # detach: this step must not update G
> lossD = bce(D(x), ones) + bce(D(fake.detach()), zeros)
> optD.zero_grad(); lossD.backward(); optD.step()
> # generator, non-saturating: descend -log D(G(z))  ==  BCE of the
> #   fakes against the *real* label
> # saturating version (4): lossG = -bce(D(fake), zeros)
> lossG = bce(D(fake), ones)
> optG.zero_grad(); lossG.backward(); optG.step()
> ```
> This is the DCGAN training loop of the code repository, which uses `nn.BCELoss` on a sigmoid output and fills the label tensor with `real_label` for the generator step.

### Mode collapse and instability

- **Mode collapse.** The generator maps many different $\mathbf{z}$ to a few outputs that currently fool the discriminator. The discriminator learns to reject them, and the generator jumps to another few. Minimax ($\min_\theta\max_\omega$) is not the same as maximin, and alternating updates do not always approximate the inner maximum well. Remedies include minibatch discrimination and feature matching.[^salimans2016]
- **No convergence guarantee.** Gradient descent on a saddle-point problem can oscillate or spiral, rather than converge. The loss curves say little about sample quality.
- **Disjoint supports.** Real images lie near a low-dimensional manifold, and so do generated ones. Two such manifolds generically do not overlap. Then a perfect discriminator exists, $\mathrm{JSD}=\log2$ whatever the generator does, and the gradient is zero. This is the motivation for the next section.

## The Wasserstein GAN

**Arjovsky's example.** Let $S\sim U[0,1]$. Let the data be $P=(0,S)$, a vertical line segment at $x=0$, and the model $Q_\theta=(\theta,S)$, the same segment shifted to $x=\theta$.[^arjovsky2017] For every $\theta\ne0$ the supports are disjoint, so:
- $D_{\mathrm{KL}}(P\,\|\,Q_\theta)=\infty$;
- $\mathrm{JSD}(P\,\|\,Q_\theta)=\log2$, a constant with zero gradient in $\theta$.

Yet the model is clearly getting better as $\theta\to0$. A useful distance should say so.

The **Wasserstein-1** (earth mover's) distance measures how much mass must be moved, and how far:

$$
W_1(P,Q)=\inf_{\pi\in\Pi(P,Q)}\mathbb{E}_{(\mathbf{x},\mathbf{y})\sim\pi}\big[\lVert\mathbf{x}-\mathbf{y}\rVert\big],
$$

where $\Pi(P,Q)$ is the set of joint distributions (couplings) with marginals $P$ and $Q$. In the example, $W_1(P,Q_\theta)=|\theta|$, which is continuous and has a useful gradient everywhere.

![[Machine Learning/images/gan-js-vs-wasserstein.svg]]
*Figure 6: two parallel line segments a distance $\theta$ apart. The JS divergence is $\log2$ for every $\theta\ne0$ and drops to 0 only at $\theta=0$, so it gives the generator no gradient. The Wasserstein distance $|\theta|$ shrinks smoothly as the generator improves.*

**A critic for $W_1$.** The coupling form is intractable, but the **Kantorovich–Rubinstein duality** turns it into a supremum over critics, much like (2):

$$
W_1(P,Q)=\sup_{\lVert T\rVert_L\le1}\Big(\mathbb{E}_{\mathbf{x}\sim P}\big[T(\mathbf{x})\big]-\mathbb{E}_{\mathbf{x}\sim Q}\big[T(\mathbf{x})\big]\Big),
$$

where the supremum is over **1-Lipschitz** functions. Compare with (2): there is no $f^*$, and the constraint on $T$ is smoothness, not a range. The WGAN trains a critic network subject to this constraint, and the generator minimizes the estimated $W_1$. Two ways to enforce the Lipschitz constraint:
- **weight clipping** (the original WGAN): clamp every critic weight to $[-\delta,\delta]$. It is crude, and can make the critic too simple;
- **gradient penalty** (WGAN-GP): add $\lambda_{\mathrm{GP}}\,\mathbb{E}_{\hat{\mathbf{x}}}\big[(\lVert\nabla_{\hat{\mathbf{x}}}T(\hat{\mathbf{x}})\rVert-1)^2\big]$ at points $\hat{\mathbf{x}}$ interpolated between real and fake samples.[^gulrajani2017]

> [!example]- Code: WGAN-GP critic and generator losses (PyTorch)
> ```python
> # assumes: critic T with an unbounded scalar output; real x and
> #   fake x_f = G(z), both (B, dim); lam = lambda_GP = 10
> # (B, 1, 1, 1) for images
> eps = torch.rand(B, 1)
> x_hat = (eps * x + (1 - eps) * x_f.detach()).requires_grad_(True)
> grad, = torch.autograd.grad(T(x_hat).sum(), x_hat,
>                             create_graph=True)
> # (||grad T(x_hat)|| - 1)^2
> gp = ((grad.flatten(1).norm(dim=1) - 1) ** 2).mean()
> # critic: maximize E_P[T] - E_Q[T]
> loss_T = T(x_f.detach()).mean() - T(x).mean() + lam * gp
> # generator: minimize the W1 estimate
> loss_G = -T(x_f).mean()
> ```
> `create_graph=True` lets the penalty itself be differentiated with respect to the critic's weights. Compared with the GAN step above, the critic has no sigmoid and no log, and weight clipping would replace the `gp` line by clamping every critic weight after each step.

$W_1$ is an **integral probability metric**, not an f-divergence. It is a genuine metric, and it stays informative when the supports do not overlap.

## Conditional GANs

So far $G_\theta(\mathbf{z})$ samples from the whole data distribution. Often we want control: an image of a given class, or the photo that goes with a given sketch. The conditional variants differ in *where* the condition enters and *what* the discriminator is asked (Figure 7).

![[Machine Learning/images/gan-conditional-variants.svg]]
*Figure 7: three ways to condition a GAN. cGAN: both players see the label $y$, and the discriminator judges whether a sample is real **and** matches $y$. AC-GAN: the discriminator sees only the image, and a second head predicts its class. InfoGAN: there are no labels; the generator receives a code $c$, and a head $R$ on the discriminator's body must recover $c$ from the fake.*

### cGAN: condition both players

Mirza and Osindero give both networks the condition $y$:[^mirza2014]

$$
\min_\theta\max_\omega\;\mathbb{E}_{(\mathbf{x},y)\sim P_X}\big[\log D_\omega(\mathbf{x},y)\big]+\mathbb{E}_{\mathbf{z},\,y}\Big[\log\Big(1-D_\omega\big(G_\theta(\mathbf{z},y),y\big)\Big)\Big].
$$

Repeating Steps 5–6 for each value of $y$ gives $D^*(\mathbf{x},y)=\frac{p(\mathbf{x}\mid y)}{p(\mathbf{x}\mid y)+q_\theta(\mathbf{x}\mid y)}$, and the generator minimizes the Jensen–Shannon divergence between the class-conditional distributions, averaged over $y$. A realistic "3" presented with the label 7 counts as fake, so the generator must respect the label. The simplest way to feed $y$ in is to concatenate an embedding of it to the inputs (below). The **projection discriminator** of Miyato and Koyama instead adds an inner product between a label embedding and the image features to the discriminator's logit, $V(\mathbf{x},y)=\rho(\phi(\mathbf{x}))+\mathbf{e}_y^\top\phi(\mathbf{x})$, a form suggested by the density-ratio view of the optimal critic; it is the standard choice in large class-conditional GANs.[^miyato2018proj]

> [!example]- Code: conditioning G and D on a label (PyTorch)
> ```python
> # assumes: y (B,) integer labels;
> #   emb_G, emb_D = nn.Embedding(K, e); G, D take e extra inputs
> z = torch.randn(B, nz)
> fake = G(torch.cat([z, emb_G(y)], dim=1))          # G(z, y)
> # D(x, y): real AND consistent with y?
> logit_fake = D(torch.cat([fake, emb_D(y)], dim=1))
> # real pairs (x, y_real) from the data
> logit_real = D(torch.cat([x, emb_D(y_real)], dim=1))
> ```
> The losses are those of the GAN step above, unchanged. For images, the embedding is broadcast to a spatial map and stacked as extra channels.

### AC-GAN: an auxiliary classifier

In the **auxiliary-classifier GAN** (Odena et al.) the generator still receives $y$, but the discriminator sees only the image.[^odena2017] It has two heads: real versus fake, and a classifier over the $K$ classes. The classification log-likelihood is added to *both* players' objectives, on real images (with their labels) and on fakes (with the labels they were generated from), so the generator is rewarded for samples that are classified as their intended class. A known side effect is that the generator is also rewarded for samples that are merely *easy* to classify, which can reduce the diversity within a class.[^gong2019]

### InfoGAN: discover the codes without labels

**InfoGAN** (Chen et al.) asks for control without any labels.[^chen2016] The noise is split into an incompressible part $\mathbf{z}$ and a **latent code** $c$, for example a 10-way categorical variable plus a few uniform continuous ones. Nothing in the GAN objective forces the generator to use $c$; it could ignore it. So InfoGAN adds a regularizer that rewards a high **mutual information** $I\big(c;G(\mathbf{z},c)\big)$: it must be possible to read $c$ back off the image.

The mutual information needs the posterior $p(c\mid\mathbf{x})$, which is intractable. As in the [[Variational Autoencoders|VAE]], an auxiliary distribution $R(c\mid\mathbf{x})$ gives a lower bound. Writing $\mathbf{x}=G(\mathbf{z},c)$:

$$
\begin{aligned}
I(c;\mathbf{x})&=H(c)-H(c\mid\mathbf{x})=H(c)+\mathbb{E}_{\mathbf{x}}\Big[\mathbb{E}_{c'\sim p(c\mid\mathbf{x})}\big[\log p(c'\mid\mathbf{x})\big]\Big]\\
&=H(c)+\mathbb{E}_{\mathbf{x}}\Big[D_{\mathrm{KL}}\big(p(\cdot\mid\mathbf{x})\,\|\,R(\cdot\mid\mathbf{x})\big)\Big]+\mathbb{E}_{\mathbf{x}}\Big[\mathbb{E}_{c'\sim p(c\mid\mathbf{x})}\big[\log R(c'\mid\mathbf{x})\big]\Big]\\
&\ge H(c)+\mathbb{E}_{c\sim p(c),\,\mathbf{z}}\Big[\log R\big(c\mid G(\mathbf{z},c)\big)\Big]=:L_I(G,R).
\end{aligned}
\tag{5}
$$

The last step drops the non-negative KL and uses the fact that sampling $c$ and then $\mathbf{x}=G(\mathbf{z},c)$ produces exactly the joint distribution of $(c,\mathbf{x})$, so no posterior sample is needed. The bound is tight when $R$ equals the true posterior. InfoGAN solves

$$
\min_{G,R}\max_D\;J(D,G)-\lambda_I\,L_I(G,R),
$$

where $J$ is the GAN objective (4). $R$ is a small head on the discriminator's body, so the extra cost is negligible. For a categorical code, $-L_I+H(c)$ is just the cross-entropy of $R$'s prediction of $c$; for a continuous code with a Gaussian $R$, it is a squared error. On MNIST, the 10-way code learned to represent the digit class without supervision, and the continuous codes captured rotation and stroke width. The same bound-by-a-classifier idea reappears in the InfoNCE bound of [[Self-Supervised Representation Learning]].

> [!example]- Code: the InfoGAN information loss (PyTorch)
> ```python
> # assumes: G(z, c_onehot) generator; D_body(x) -> features h;
> #   heads D_head(h) -> logit, R_head(h) -> K logits
> # c ~ p(c), uniform categorical
> c = torch.randint(K, (B,))
> c1 = F.one_hot(c, K).float()
> fake = G(torch.cat([torch.randn(B, nz), c1], dim=1))
> h = D_body(fake)
> # -E[log R(c | G(z, c))] = H(c) - L_I
> info_loss = F.cross_entropy(R_head(h), c)
> # non-saturating GAN loss + lambda_I * info
> lossG = bce(D_head(h), ones) + lam * info_loss
> # R_head and D_body also minimize info_loss; lam = lambda_I = 1 (discrete c)
> ```
> AC-GAN uses the same cross-entropy, but with the true labels $y$ in place of the sampled code $c$, and applies it to real images as well.

### Image-to-image translation: pix2pix and CycleGAN

When the condition is itself an image, a conditional GAN becomes a learned image-to-image translator: edges to photos, maps to aerial views, day to night.

**pix2pix** (Isola et al.) uses **paired** data $(\mathbf{x}_i,\mathbf{y}_i)$.[^isola2017] The generator maps an input image $\mathbf{x}$ to an output $G(\mathbf{x})$, and the discriminator judges pairs $(\mathbf{x},\mathbf{y})$. An L1 term is added:

$$
G^*=\arg\min_G\max_D\;\mathcal{L}_{\mathrm{cGAN}}(G,D)+\lambda_{\mathrm{L1}}\,\mathbb{E}_{\mathbf{x},\mathbf{y}}\big[\lVert\mathbf{y}-G(\mathbf{x})\rVert_1\big],\qquad\lambda_{\mathrm{L1}}=100.
$$

The two terms split the work. L1 alone gives the blurry average of all plausible outputs (the same effect as the [[Variational Autoencoders#VAEs in practice|Gaussian VAE decoder]]); it gets the low frequencies right. The adversarial term supplies sharp, realistic high-frequency detail, so the discriminator only needs to judge local **patches** (the "PatchGAN" discriminator).

**CycleGAN** (Zhu et al.) drops the pairs.[^zhu2017] We have a set of horse photos and a set of zebra photos, with no correspondence between them. Two generators, $G:X\to Y$ and $F:Y\to X$, are each trained against a discriminator in the target domain. Adversarial losses alone are underdetermined: $G$ could map every horse to the same convincing zebra and still match the zebra distribution. The fix is **cycle-consistency**: translating there and back should return the input (Figure 8),

$$
\mathcal{L}_{\mathrm{cyc}}(G,F)=\mathbb{E}_{\mathbf{x}}\big[\lVert F(G(\mathbf{x}))-\mathbf{x}\rVert_1\big]+\mathbb{E}_{\mathbf{y}}\big[\lVert G(F(\mathbf{y}))-\mathbf{y}\rVert_1\big],
\tag{6}
$$

and the full objective is $\mathcal{L}_{\mathrm{GAN}}(G,D_Y)+\mathcal{L}_{\mathrm{GAN}}(F,D_X)+\lambda_{\mathrm{cyc}}\,\mathcal{L}_{\mathrm{cyc}}(G,F)$ with $\lambda_{\mathrm{cyc}}=10$. The paper uses the least-squares GAN loss for stability,[^mao2017] and an optional identity loss that keeps colours unchanged when the input already looks like the target domain.

![[Machine Learning/images/gan-cyclegan.svg]]
*Figure 8: CycleGAN. $G$ translates $X\to Y$ and $F$ translates $Y\to X$, each judged by a discriminator in its target domain. The cycle-consistency loss (6) asks that $F(G(\mathbf{x}))$ return to $\mathbf{x}$ and $G(F(\mathbf{y}))$ return to $\mathbf{y}$, which rules out translations that forget the content of the input.*

> [!example]- Code: the CycleGAN generator loss (PyTorch)
> ```python
> # assumes: G: X -> Y and Fy: Y -> X generators; D_X, D_Y return
> #   logits; batches x (from X), y (from Y); lam = lambda_cyc = 10
> fake_y, fake_x = G(x), Fy(y)
> # fool both discriminators
> adv = bce(D_Y(fake_y), ones) + bce(D_X(fake_x), ones)
> # eq. (6)
> cyc = (Fy(fake_y) - x).abs().mean() + (G(fake_x) - y).abs().mean()
> # optional identity loss
> idt = (G(y) - y).abs().mean() + (Fy(x) - x).abs().mean()
> loss_gen = adv + lam * cyc + 0.5 * lam * idt
> ```
> The paper replaces the BCE terms by least-squares ones, `((D_Y(fake_y) - 1) ** 2).mean()`. pix2pix is the paired special case: one generator, a discriminator on pairs, and `100 * (G(x) - y).abs().mean()` in place of the cycle term.

Cycle-consistency does not guarantee that the *meaning* is preserved: Chu et al. showed that a CycleGAN can hide the information it needs for the reverse trip in imperceptible high-frequency patterns.[^chu2017] Translating labelled source images into the style of an unlabelled target domain is the idea behind [[Unsupervised Domain Adaptation#Pixel-level adaptation|pixel-level domain adaptation]] (CyCADA), which adds a semantic-consistency loss for exactly this reason.[^hoffman2018]

## Architectures that made GANs work

The objective is only half the story. Much of the progress from blurry 32×32 samples to photorealistic megapixel faces came from architecture and normalization.

### DCGAN

Radford et al. found a convolutional recipe that trained reliably,[^radford2016] and it is the one used in the code repository:
- no pooling: **strided convolutions** downsample in the discriminator and **transposed convolutions** upsample in the generator;
- **batch normalization** in both networks, except the generator's output layer and the discriminator's input layer;
- no fully connected hidden layers;
- **ReLU** in the generator with a **tanh** output (images scaled to $[-1,1]$), and **LeakyReLU** (slope 0.2) in the discriminator;
- weights initialized from $\mathcal{N}(0,0.02^2)$, and Adam with learning rate $2\times10^{-4}$ and $\beta_1=0.5$.

DCGAN also showed that the latent space has linear structure: averaging the $\mathbf{z}$ vectors of a few samples per concept, "smiling woman" $-$ "neutral woman" $+$ "neutral man" decodes to smiling men.

### Spectral normalization

The WGAN needs a 1-Lipschitz critic, and even standard GANs train more stably with a smooth discriminator. For a critic $V=\mathbf{W}^{(M)}\circ a\circ\dots\circ a\circ\mathbf{W}^{(1)}$ with 1-Lipschitz activations $a$ (ReLU, LeakyReLU), the Lipschitz constant is at most the product of the layers' spectral norms (largest singular values), $\prod_m\lVert\mathbf{W}^{(m)}\rVert_2$. **Spectral normalization** (Miyato et al.) divides each weight matrix by an estimate of its spectral norm, so every layer, and hence the discriminator, is at most 1-Lipschitz.[^miyato2018sn] The estimate comes from one step of [[Eigenvalues and Eigenvectors#Power iteration and the QR algorithm|power iteration]] per training step, with the singular-vector estimate kept between steps, so it is almost free. There is no second backward pass, unlike the gradient penalty, and it works with any loss. Convolution kernels are reshaped into a matrix (out channels by everything else) first, which approximates the true operator norm of the convolution. Spectral normalization, usually with the **hinge loss**,[^lim2017] became standard in SAGAN and BigGAN.[^zhang2019]<sup>,</sup>[^brock2019]

> [!example]- Code: spectral normalization by power iteration (PyTorch)
> ```python
> # assumes: W (out, in) weight matrix; u (out,) unit vector kept
> #   from the previous training step
> with torch.no_grad():
>     # one power-iteration step
>     v = F.normalize(W.T @ u, dim=0)
>     u = F.normalize(W @ v, dim=0)
> # estimate of ||W||_2 (largest singular value), differentiable in W
> sigma = u @ W @ v
> # largest singular value about 1: the layer is 1-Lipschitz
> W_sn = W / sigma
> # the built-in version, for a whole layer:
> sn = torch.nn.utils.parametrizations.spectral_norm
> layer = sn(nn.Linear(n_in, n_out))
> ```
> The hinge loss that usually accompanies it: `lossD = F.relu(1 - V(x)).mean() + F.relu(1 + V(fake)).mean()` and `lossG = -V(fake).mean()`.

### Progressive growing and StyleGAN

**Progressive growing** (Karras et al.) trains at 4×4 first and then adds layers that double the resolution, fading each new layer in smoothly. The network learns the coarse structure before the details, and training reaches 1024×1024 faces.[^karras2018]

**StyleGAN** (Karras et al.) redesigned the generator.[^karras2019]
- A **mapping network**, an 8-layer MLP, sends $\mathbf{z}$ to an intermediate latent $\mathbf{w}$. Because $\mathbf{w}$ does not have to follow a fixed density, its space is less entangled than that of $\mathbf{z}$.
- The **synthesis network** starts from a learned constant. At every layer, $\mathbf{w}$ sets per-channel scales and biases through **adaptive instance normalization** (AdaIN),[^huang2017] $\mathrm{AdaIN}(\mathbf{h};\mathbf{s},\mathbf{b})=\mathbf{s}\odot\frac{\mathbf{h}-\mu(\mathbf{h})}{\mathrm{sd}(\mathbf{h})}+\mathbf{b}$, where $\mu$ and $\mathrm{sd}$ are the per-channel mean and standard deviation over the spatial positions, and the style $(\mathbf{s},\mathbf{b})$ (scale and bias) is a learned affine function of $\mathbf{w}$.
- Per-pixel **noise inputs** add stochastic detail (hair, freckles). Feeding different $\mathbf{w}$'s to different layers (**style mixing**) shows that the coarse layers control pose and shape, and the fine layers control colour and texture.
- StyleGAN2 replaced AdaIN by weight demodulation, which removed characteristic "droplet" artefacts, and dropped progressive growing.[^karras2020]

**The truncation trick** trades diversity for quality at sampling time. BigGAN samples $\mathbf{z}$ from a truncated normal, resampling entries whose magnitude exceeds a threshold.[^brock2019] StyleGAN pulls $\mathbf{w}$ towards its average, $\mathbf{w}'=\bar{\mathbf{w}}+\psi(\mathbf{w}-\bar{\mathbf{w}})$ with $\psi<1$. Both avoid the sparsely trained edges of the latent distribution, where samples are worst.

> [!example]- Code: AdaIN and the truncation trick (PyTorch)
> ```python
> # assumes: mapping (MLP z -> w), A (linear w -> 2C);
> #   feature map h (B, C, H, W); w_bar = mean of mapping(z); psi
> # intermediate latent w = f(z)
> w = mapping(z)
> # style: per-channel scale and bias, each (B, C)
> s, b = A(w).chunk(2, dim=1)
> mu = h.mean(dim=(2, 3), keepdim=True)
> sd = h.std(dim=(2, 3), keepdim=True) + 1e-8
> # AdaIN(h; s, b)
> h = s[:, :, None, None] * (h - mu) / sd + b[:, :, None, None]
> # truncation: psi = 1 off, psi = 0 the "average" image
> w_trunc = w_bar + psi * (w - w_bar)
> ```

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
- Conditioning: give $y$ to both players (cGAN), ask the discriminator to classify (AC-GAN), or recover an unsupervised code through a mutual-information bound (InfoGAN). Image-to-image translation adds an L1 term with pairs (pix2pix) or a cycle-consistency term without them (CycleGAN).
- Architecture matters as much as the loss: DCGAN's convolutional recipe, spectral normalization for a Lipschitz discriminator, and StyleGAN's mapping network, AdaIN styles and truncation trick.

## Where this shows up next

- [[Unsupervised Domain Adaptation]]: DANN and ADDA play this game on *features* instead of images, so that a domain discriminator cannot tell source from target and a classifier trained on the source transfers. CycleGAN translations give the pixel-level version.
- [[Variational Autoencoders]]: the likelihood-based alternative, and the forward-vs-reverse KL picture used here. Its conditional VAE parallels the cGAN, and VQGAN puts an adversarial loss on a VQ-VAE's decoder.
- [[Linear Models]]: the discriminator is a logistic-regression classifier on learned features, trained with cross-entropy.
- [[Multivariable Calculus#The Lagrangian and duality|The Lagrangian and duality]] and its application in [[Support Vector Machines]]: the same convex duality that gives the variational bound here.
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
[^mirza2014]: Mirza, M., & Osindero, S. (2014). *Conditional Generative Adversarial Nets*. arXiv. [arXiv:1411.1784](https://arxiv.org/abs/1411.1784)
[^miyato2018proj]: Miyato, T., & Koyama, M. (2018). *cGANs with Projection Discriminator*. ICLR. [arXiv:1802.05637](https://arxiv.org/abs/1802.05637)
[^odena2017]: Odena, A., Olah, C., & Shlens, J. (2017). *Conditional Image Synthesis With Auxiliary Classifier GANs*. ICML. [arXiv:1610.09585](https://arxiv.org/abs/1610.09585)

[^gong2019]: Gong, M., Xu, Y., Li, C., Zhang, K., & Batmanghelich, K. (2019). *Twin Auxiliary Classifiers GAN*. NeurIPS. [arXiv:1907.02690](https://arxiv.org/abs/1907.02690)
[^chen2016]: Chen, X., Duan, Y., Houthooft, R., Schulman, J., Sutskever, I., & Abbeel, P. (2016). *InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets*. NeurIPS. [arXiv:1606.03657](https://arxiv.org/abs/1606.03657)
[^isola2017]: Isola, P., Zhu, J.-Y., Zhou, T., & Efros, A. A. (2017). *Image-to-Image Translation with Conditional Adversarial Networks*. CVPR. [arXiv:1611.07004](https://arxiv.org/abs/1611.07004)
[^zhu2017]: Zhu, J.-Y., Park, T., Isola, P., & Efros, A. A. (2017). *Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks*. ICCV. [arXiv:1703.10593](https://arxiv.org/abs/1703.10593)
[^mao2017]: Mao, X., Li, Q., Xie, H., Lau, R. Y. K., Wang, Z., & Smolley, S. P. (2017). *Least Squares Generative Adversarial Networks*. ICCV. [arXiv:1611.04076](https://arxiv.org/abs/1611.04076)
[^chu2017]: Chu, C., Zhmoginov, A., & Sandler, M. (2017). *CycleGAN, a Master of Steganography*. NeurIPS Workshop on Machine Deception. [arXiv:1712.02950](https://arxiv.org/abs/1712.02950)
[^hoffman2018]: Hoffman, J., Tzeng, E., Park, T., Zhu, J.-Y., Isola, P., Saenko, K., Efros, A. A., & Darrell, T. (2018). *CyCADA: Cycle-Consistent Adversarial Domain Adaptation*. ICML. [arXiv:1711.03213](https://arxiv.org/abs/1711.03213)
[^radford2016]: Radford, A., Metz, L., & Chintala, S. (2016). *Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks*. ICLR. [arXiv:1511.06434](https://arxiv.org/abs/1511.06434)
[^miyato2018sn]: Miyato, T., Kataoka, T., Koyama, M., & Yoshida, Y. (2018). *Spectral Normalization for Generative Adversarial Networks*. ICLR. [arXiv:1802.05957](https://arxiv.org/abs/1802.05957)
[^lim2017]: Lim, J. H., & Ye, J. C. (2017). *Geometric GAN*. arXiv. [arXiv:1705.02894](https://arxiv.org/abs/1705.02894)
[^zhang2019]: Zhang, H., Goodfellow, I., Metaxas, D., & Odena, A. (2019). *Self-Attention Generative Adversarial Networks*. ICML. [arXiv:1805.08318](https://arxiv.org/abs/1805.08318)
[^brock2019]: Brock, A., Donahue, J., & Simonyan, K. (2019). *Large Scale GAN Training for High Fidelity Natural Image Synthesis*. ICLR. [arXiv:1809.11096](https://arxiv.org/abs/1809.11096)
[^karras2018]: Karras, T., Aila, T., Laine, S., & Lehtinen, J. (2018). *Progressive Growing of GANs for Improved Quality, Stability, and Variation*. ICLR. [arXiv:1710.10196](https://arxiv.org/abs/1710.10196)
[^karras2019]: Karras, T., Laine, S., & Aila, T. (2019). *A Style-Based Generator Architecture for Generative Adversarial Networks*. CVPR. [arXiv:1812.04948](https://arxiv.org/abs/1812.04948)
[^huang2017]: Huang, X., & Belongie, S. (2017). *Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization*. ICCV. [arXiv:1703.06868](https://arxiv.org/abs/1703.06868)
[^karras2020]: Karras, T., Laine, S., Aittala, M., Hellsten, J., Lehtinen, J., & Aila, T. (2020). *Analyzing and Improving the Image Quality of StyleGAN*. CVPR. [arXiv:1912.04958](https://arxiv.org/abs/1912.04958)
