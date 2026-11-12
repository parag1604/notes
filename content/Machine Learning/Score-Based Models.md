---
title: Score-Based Models
tags:
  - generative-models
  - score-matching
  - langevin-dynamics
  - representation-learning
date: 2026-11-12
draft: false
---

> **Prerequisites:** [[Energy-Based Models]], especially [[Energy-Based Models#Sampling with Langevin dynamics|Langevin dynamics]] and [[Energy-Based Models#From energies to scores|From energies to scores]]. Gaussian [[Non-parametric Models#Kernel density estimation|kernel density estimation]] reappears as the "noisy data" distribution.
>
> **TL;DR:**
> - The **score** $\nabla_{\mathbf{x}}\log p(\mathbf{x})$ is all Langevin dynamics needs, and it never involves the partition function. So learn the score directly with a network $\mathbf{s}_\theta(\mathbf{x})$.
> - We never observe the true score, but **score matching** fits it anyway: integration by parts (Hyvärinen), random projections (sliced), or, simplest of all, **denoising**: add Gaussian noise to the data and regress onto the noise. **Tweedie's formula** says the optimal denoiser *is* the score.
> - A score learned on clean data fails in practice. It is inaccurate wherever there is little data, and Langevin dynamics with it gets the weights of the modes wrong.
> - **Noise-conditional score networks** learn the score at many noise levels and sample by **annealed Langevin dynamics**, from very noisy to clean. Taking infinitely many noise levels gives a stochastic differential equation, the bridge to diffusion models.
>
> **Notation:** clean data $\mathbf{x}\sim p$, noisy data $\tilde{\mathbf{x}}=\mathbf{x}+\sigma\boldsymbol\epsilon$ with $\boldsymbol\epsilon\sim\mathcal{N}(\mathbf{0},\mathbf{I})$, and $q_\sigma$ the density of $\tilde{\mathbf{x}}$. Langevin step sizes are $\eta$, as in the EBM lecture.

## Motivation

The [[Energy-Based Models|EBM lecture]] ended with an observation. Langevin dynamics,

$$
\mathbf{x}_{k+1}=\mathbf{x}_k+\frac{\eta}{2}\nabla_{\mathbf{x}}\log p(\mathbf{x}_k)+\sqrt{\eta}\,\mathbf{z}_k,
\qquad \mathbf{z}_k\sim\mathcal{N}(\mathbf{0},\mathbf{I}),
\tag{1}
$$

draws samples from $p$ using only the gradient of the log-density. For an EBM, $\nabla_{\mathbf{x}}\log p_\theta=-\nabla_{\mathbf{x}}E_\theta$, and the partition function drops out. Training the EBM was still expensive: maximum likelihood needed samples from the model, and hence an MCMC run, inside every gradient step.

The idea of score-based models is to cut out the middleman. Never learn an energy. Learn the vector field $\nabla_{\mathbf{x}}\log p$ directly, from data, by a regression that needs no sampling at all. Then plug it into (1).

## The score

The **score** of a density $p$ (strictly, its *Stein score*: the gradient with respect to the data, not the parameters) is

$$
\mathbf{s}(\mathbf{x})=\nabla_{\mathbf{x}}\log p(\mathbf{x}).
$$

It is a vector field that points in the direction in which the log-density increases fastest (Figure 1). A few properties make it a convenient object to learn:
- **No normalizer.** If $p=\tilde p/Z$, then $\nabla_{\mathbf{x}}\log p=\nabla_{\mathbf{x}}\log\tilde p$. Any unnormalized model has an exact score.
- **It determines the density.** On a connected support, $\log p$ is recovered from its gradient up to an additive constant, which normalization fixes. Nothing is lost by modelling the score instead of the density.
- **Gaussians have linear scores.** For $\mathcal{N}(\boldsymbol\mu,\sigma^2\mathbf{I})$, $\mathbf{s}(\mathbf{x})=-(\mathbf{x}-\boldsymbol\mu)/\sigma^2$: a spring pulling back to the mean, stronger for narrow Gaussians.
- **Mixtures average their components' scores.** For $p=\sum_k\pi_kp_k$, $\mathbf{s}(\mathbf{x})=\sum_k r_k(\mathbf{x})\nabla\log p_k(\mathbf{x})$, where $r_k(\mathbf{x})=\pi_kp_k(\mathbf{x})/p(\mathbf{x})$ are the [[Clustering and Expectation Maximization#E-step: responsibilities|responsibilities]] from EM. Every score in this lecture's figures is computed this way, exactly.

![[Machine Learning/images/score-based-models-score-field.svg]]
*Figure 1. (a) A mixture of three Gaussians. (b) Its score field, shown as unit-length arrows coloured by magnitude. The arrows point towards the nearest mode, and they are longest far from the data, where $\log p$ drops quadratically.*

A score model is a network $\mathbf{s}_\theta:\mathbb{R}^D\to\mathbb{R}^D$ with as many outputs as inputs, typically a U-Net for images. Nothing forces $\mathbf{s}_\theta$ to be the gradient of a scalar function. That freedom is deliberate: a free-form vector field is easier to fit, and in practice the learned field is close enough to a gradient for sampling to work.

## Score matching

The natural objective is the expected squared error between the model's score and the true score, the **Fisher divergence**:

$$
J_{\text{ESM}}(\theta)=\frac12\,\mathbb{E}_{p(\mathbf{x})}\Big[\big\lVert\mathbf{s}_\theta(\mathbf{x})-\nabla_{\mathbf{x}}\log p(\mathbf{x})\big\rVert^2\Big].
\tag{2}
$$

This is **explicit score matching**, and it cannot be used as written: the target $\nabla_{\mathbf{x}}\log p$ is exactly what we don't know. We only have samples from $p$. The three methods below are three ways around this.

### Implicit score matching (Hyvärinen)

Hyvärinen showed that (2) can be rewritten without the unknown score.[^hyvarinen2005] Up to a constant that does not depend on $\theta$,

$$
J_{\text{ISM}}(\theta)=\mathbb{E}_{p(\mathbf{x})}\Big[\operatorname{tr}\big(\nabla_{\mathbf{x}}\mathbf{s}_\theta(\mathbf{x})\big)+\frac12\big\lVert\mathbf{s}_\theta(\mathbf{x})\big\rVert^2\Big].
\tag{3}
$$

Here $\nabla_{\mathbf{x}}\mathbf{s}_\theta$ is the $D\times D$ Jacobian of the score network. Every term is an expectation over data of something we can compute.

> [!proof]- Proof: integration by parts removes the unknown score
> Take one dimension first and write $\psi=(\log p)'=p'/p$. Expanding the square in (2),
>
> $$
> J_{\text{ESM}}=\mathbb{E}_p\Big[\tfrac12s_\theta^2\Big]-\mathbb{E}_p\big[s_\theta\,\psi\big]+\underbrace{\mathbb{E}_p\Big[\tfrac12\psi^2\Big]}_{\text{const}} .
> $$
>
> Only the middle term involves the unknown $\psi$. Because $p\,\psi=p'$, it is an integral we can move the derivative across:
>
> $$
> \mathbb{E}_p\big[s_\theta\,\psi\big]=\int s_\theta(x)\,p'(x)\,dx
> =\Big[s_\theta(x)\,p(x)\Big]_{-\infty}^{\infty}-\int s_\theta'(x)\,p(x)\,dx
> =-\mathbb{E}_p\big[s_\theta'\big],
> $$
>
> assuming $s_\theta(x)p(x)\to0$ as $|x|\to\infty$. Hence $J_{\text{ESM}}=\mathbb{E}_p\big[\tfrac12s_\theta^2+s_\theta'\big]+\text{const}$. In $D$ dimensions, apply the same step to each coordinate $\int s_{\theta,i}\,\partial_ip\,d\mathbf{x}$; the derivatives $\partial_is_{\theta,i}$ sum to the trace of the Jacobian, giving (3).

> [!tip] Intuition
> Minimizing (3) asks for two things. The $\tfrac12\lVert\mathbf{s}_\theta\rVert^2$ term wants the score to be **small at the data**, so data points sit at stationary points of $\log p$. The trace term wants the divergence of the field to be **negative at the data**, so the arrows converge on the data points, which makes them peaks rather than valleys. For an EBM, $\mathbf{s}_\theta=-\nabla E_\theta$ and the trace is $-\Delta E_\theta$, minus the Laplacian: (3) asks for flat, sharply convex energy minima at the data.

The catch is computational. The trace of a $D\times D$ Jacobian needs $D$ backward passes, one per diagonal entry. For a $256\times256$ colour image, $D\approx2\times10^5$.

### Sliced score matching

**Sliced score matching**[^song2019sliced] removes the trace with random projections. For any random vector $\mathbf{v}$ with $\mathbb{E}[\mathbf{v}\mathbf{v}^\top]=\mathbf{I}$ (standard Gaussian, or random signs),

$$
\mathbb{E}_{\mathbf{v}}\big[\mathbf{v}^\top\mathbf{A}\mathbf{v}\big]=\operatorname{tr}\big(\mathbf{A}\,\mathbb{E}[\mathbf{v}\mathbf{v}^\top]\big)=\operatorname{tr}(\mathbf{A}),
$$

which is Hutchinson's trace estimator.[^hutchinson1989] Replacing the trace in (3) gives

$$
J_{\text{SSM}}(\theta)=\mathbb{E}_{p(\mathbf{x})}\,\mathbb{E}_{\mathbf{v}}\Big[\mathbf{v}^\top\nabla_{\mathbf{x}}\mathbf{s}_\theta(\mathbf{x})\,\mathbf{v}+\frac12\big\lVert\mathbf{s}_\theta(\mathbf{x})\big\rVert^2\Big].
$$

The term $\mathbf{v}^\top\nabla_{\mathbf{x}}\mathbf{s}_\theta\,\mathbf{v}$ is the derivative of the scalar $\mathbf{v}^\top\mathbf{s}_\theta$ in the direction $\mathbf{v}$: one Jacobian-vector product, one extra backward pass, whatever the dimension. Geometrically, we match the scores' one-dimensional projections along random directions, the "slices".

### Denoising score matching

The third method, due to Vincent, changes the target distribution instead.[^vincent2011] Perturb each data point with Gaussian noise of scale $\sigma$:

$$
\tilde{\mathbf{x}}=\mathbf{x}+\sigma\boldsymbol\epsilon,
\qquad
q_\sigma(\tilde{\mathbf{x}}\mid\mathbf{x})=\mathcal{N}(\tilde{\mathbf{x}};\mathbf{x},\sigma^2\mathbf{I}),
\qquad
q_\sigma(\tilde{\mathbf{x}})=\int p(\mathbf{x})\,q_\sigma(\tilde{\mathbf{x}}\mid\mathbf{x})\,d\mathbf{x}.
$$

The noisy marginal $q_\sigma$ is the data density blurred by a Gaussian. For a finite training set, it is precisely the Gaussian [[Non-parametric Models#Kernel density estimation|kernel density estimate]] with bandwidth $\sigma$. Its score is unknown, but the score of the **noise kernel** is known in closed form:

$$
\nabla_{\tilde{\mathbf{x}}}\log q_\sigma(\tilde{\mathbf{x}}\mid\mathbf{x})=-\frac{\tilde{\mathbf{x}}-\mathbf{x}}{\sigma^2}=-\frac{\boldsymbol\epsilon}{\sigma}.
$$

**Denoising score matching** regresses the network onto this known target:

$$
J_{\text{DSM}}(\theta)=\frac12\,\mathbb{E}_{p(\mathbf{x})}\,\mathbb{E}_{\boldsymbol\epsilon}\Big[\Big\lVert\mathbf{s}_\theta(\mathbf{x}+\sigma\boldsymbol\epsilon)+\frac{\boldsymbol\epsilon}{\sigma}\Big\rVert^2\Big].
\tag{4}
$$

> [!note] Theorem (Vincent, 2011)
> $J_{\text{DSM}}(\theta)$ equals the explicit score-matching objective (2) **for the noisy density** $q_\sigma$, up to a constant:
>
> $$
> J_{\text{DSM}}(\theta)=\frac12\,\mathbb{E}_{q_\sigma(\tilde{\mathbf{x}})}\Big[\big\lVert\mathbf{s}_\theta(\tilde{\mathbf{x}})-\nabla_{\tilde{\mathbf{x}}}\log q_\sigma(\tilde{\mathbf{x}})\big\rVert^2\Big]+\text{const}.
> $$
>
> So minimizing (4) makes $\mathbf{s}_\theta$ learn $\nabla\log q_\sigma$, the score of the **noisy** data.

> [!proof]- Proof
> Expand both squared norms. The term $\tfrac12\mathbb{E}\lVert\mathbf{s}_\theta(\tilde{\mathbf{x}})\rVert^2$ is identical in both, because the marginal of $\tilde{\mathbf{x}}$ under $p(\mathbf{x})q_\sigma(\tilde{\mathbf{x}}\mid\mathbf{x})$ is $q_\sigma(\tilde{\mathbf{x}})$. The squared targets do not depend on $\theta$. It remains to show the cross terms agree:
>
> $$
> \begin{aligned}
> \mathbb{E}_{q_\sigma(\tilde{\mathbf{x}})}\big[\mathbf{s}_\theta^\top\nabla\log q_\sigma(\tilde{\mathbf{x}})\big]
> &=\int\mathbf{s}_\theta(\tilde{\mathbf{x}})^\top\nabla_{\tilde{\mathbf{x}}}q_\sigma(\tilde{\mathbf{x}})\,d\tilde{\mathbf{x}}\\
> &=\int\mathbf{s}_\theta(\tilde{\mathbf{x}})^\top\!\int p(\mathbf{x})\,\nabla_{\tilde{\mathbf{x}}}q_\sigma(\tilde{\mathbf{x}}\mid\mathbf{x})\,d\mathbf{x}\,d\tilde{\mathbf{x}}\\
> &=\iint p(\mathbf{x})\,q_\sigma(\tilde{\mathbf{x}}\mid\mathbf{x})\,\mathbf{s}_\theta(\tilde{\mathbf{x}})^\top\nabla_{\tilde{\mathbf{x}}}\log q_\sigma(\tilde{\mathbf{x}}\mid\mathbf{x})\,d\mathbf{x}\,d\tilde{\mathbf{x}}\\
> &=\mathbb{E}_{p(\mathbf{x})q_\sigma(\tilde{\mathbf{x}}\mid\mathbf{x})}\big[\mathbf{s}_\theta^\top\nabla\log q_\sigma(\tilde{\mathbf{x}}\mid\mathbf{x})\big].
> \end{aligned}
> $$
>
> The first line uses $q\nabla\log q=\nabla q$; the second differentiates under the integral defining $q_\sigma$; the third uses $\nabla q_\sigma(\tilde{\mathbf{x}}\mid\mathbf{x})=q_\sigma(\tilde{\mathbf{x}}\mid\mathbf{x})\nabla\log q_\sigma(\tilde{\mathbf{x}}\mid\mathbf{x})$.

Denoising score matching is the method that won. It needs no Jacobian at all: one forward pass and an ordinary mean-squared error. The price is that it learns the score of $q_\sigma$, not of $p$. For small $\sigma$ the two are close; for large $\sigma$ they differ, which turns out to be a feature (next section). Figure 2 shows it working.

![[Machine Learning/images/score-based-models-dsm-fit.svg]]
*Figure 2. Denoising score matching on a 1-D mixture (density shaded green). A model that is linear in 40 Gaussian bump features is fitted to the DSM target $-\epsilon/\sigma$ by least squares on 20 000 noisy samples. It recovers the noisy score $\nabla\log q_\sigma$ (blue), not the clean one (dotted): the relative error is 10% at $\sigma=0.3$ and 4% at $\sigma=1$, measured where $q_\sigma$ exceeds 1% of its peak. At $\sigma=0.3$ the fit is wild beyond $|x|\approx4$, where there is no data to fit.*

Rewriting (4) in terms of a **noise predictor** $\boldsymbol\epsilon_\theta(\tilde{\mathbf{x}})=-\sigma\,\mathbf{s}_\theta(\tilde{\mathbf{x}})$ turns it into

$$
J_{\text{DSM}}=\frac{1}{2\sigma^2}\,\mathbb{E}\Big[\big\lVert\boldsymbol\epsilon_\theta(\mathbf{x}+\sigma\boldsymbol\epsilon)-\boldsymbol\epsilon\big\rVert^2\Big]:
$$

**look at a noisy point and guess the noise that was added.** Remember this form: it is exactly the loss of denoising diffusion models.

## Tweedie's formula: the score is the optimal denoiser

Why should guessing noise teach us the score? Tweedie's formula gives the cleanest answer.[^efron2011] For $\tilde{\mathbf{x}}=\mathbf{x}+\sigma\boldsymbol\epsilon$ with any prior $p(\mathbf{x})$,

$$
\mathbb{E}\big[\mathbf{x}\mid\tilde{\mathbf{x}}\big]=\tilde{\mathbf{x}}+\sigma^2\,\nabla_{\tilde{\mathbf{x}}}\log q_\sigma(\tilde{\mathbf{x}}),
\qquad
\mathbb{E}\big[\boldsymbol\epsilon\mid\tilde{\mathbf{x}}\big]=-\sigma\,\nabla_{\tilde{\mathbf{x}}}\log q_\sigma(\tilde{\mathbf{x}}).
\tag{5}
$$

> [!proof]- Proof
> Differentiate the noisy marginal under the integral. Since $\nabla_{\tilde{\mathbf{x}}}\mathcal{N}(\tilde{\mathbf{x}};\mathbf{x},\sigma^2\mathbf{I})=\mathcal{N}(\tilde{\mathbf{x}};\mathbf{x},\sigma^2\mathbf{I})\,(\mathbf{x}-\tilde{\mathbf{x}})/\sigma^2$,
>
> $$
> \nabla_{\tilde{\mathbf{x}}}q_\sigma(\tilde{\mathbf{x}})
> =\int p(\mathbf{x})\,q_\sigma(\tilde{\mathbf{x}}\mid\mathbf{x})\,\frac{\mathbf{x}-\tilde{\mathbf{x}}}{\sigma^2}\,d\mathbf{x}
> =q_\sigma(\tilde{\mathbf{x}})\,\frac{\mathbb{E}[\mathbf{x}\mid\tilde{\mathbf{x}}]-\tilde{\mathbf{x}}}{\sigma^2},
> $$
>
> using Bayes' rule, $p(\mathbf{x})q_\sigma(\tilde{\mathbf{x}}\mid\mathbf{x})=q_\sigma(\tilde{\mathbf{x}})\,p(\mathbf{x}\mid\tilde{\mathbf{x}})$. Dividing by $q_\sigma(\tilde{\mathbf{x}})$ gives the first identity. The second follows from $\boldsymbol\epsilon=(\tilde{\mathbf{x}}-\mathbf{x})/\sigma$.

Read (5) three ways:
- **Score = denoising direction.** The score at a noisy point, scaled by $\sigma^2$, is the step from $\tilde{\mathbf{x}}$ to the best guess of the clean point. The minimum-MSE denoiser is a gradient step on $\log q_\sigma$.
- **DSM is consistent.** The minimizer of (4) is the conditional mean of its target, $\mathbb{E}[-\boldsymbol\epsilon/\sigma\mid\tilde{\mathbf{x}}]=\nabla\log q_\sigma(\tilde{\mathbf{x}})$, by (5). This is a second proof of Vincent's theorem.
- **Denoising autoencoders learn scores.** An autoencoder $r$ trained to map $\tilde{\mathbf{x}}$ back to $\mathbf{x}$ with squared error learns $r(\tilde{\mathbf{x}})=\mathbb{E}[\mathbf{x}\mid\tilde{\mathbf{x}}]$, so $(r(\tilde{\mathbf{x}})-\tilde{\mathbf{x}})/\sigma^2$ is a score estimate.[^alain2014]

![[Machine Learning/images/score-based-models-tweedie.svg]]
*Figure 3. The optimal denoiser $\mathbb{E}[x\mid\tilde x]$ for a 1-D mixture with modes at $-2$ and $1.5$, from Tweedie's formula. With little noise it stays near the identity and only shrinks towards the nearer mode; with $\sigma=2$ it pulls everything towards the modes. A Monte Carlo check of $\mathbb{E}[x\mid\tilde x=0.5]$ agrees with (5) to within sampling error (e.g. 0.994 vs 0.986 at $\sigma=0.5$).*

## Why naive score-based sampling fails

We now have a cheap way to learn a score. The obvious recipe is: train $\mathbf{s}_\theta$ with DSM at a small $\sigma$ (so $q_\sigma\approx p$), then run Langevin (1) from random noise. It works poorly, for three reasons.[^song2019]

**1. The manifold hypothesis.** Real data, such as natural images, concentrate near a low-dimensional manifold inside $\mathbb{R}^D$. The score $\nabla\log p$ is then undefined off the manifold, since $\log p=-\infty$ there. Implicit score matching is also inconsistent there, as its derivation assumes a density with full support.

**2. Inaccurate scores in low-density regions.** Every objective above is an expectation under the data distribution, so errors are weighted by $p(\mathbf{x})$. Where data are rare, the network receives almost no training signal and its score is arbitrary: see the edges of Figure 2's left panel. A Langevin chain started from random noise begins in exactly such a region, so its first steps follow a meaningless field.

**3. Langevin cannot see mixing weights.** Suppose $p=\pi p_1+(1-\pi)p_2$, with the two components on (nearly) disjoint regions. On the region of $p_1$,

$$
\nabla_{\mathbf{x}}\log p(\mathbf{x})=\nabla_{\mathbf{x}}\log\big(\pi\,p_1(\mathbf{x})\big)=\nabla_{\mathbf{x}}\log p_1(\mathbf{x}),
$$

and $\pi$ has vanished. The score inside each mode does not know how heavy that mode is. In principle, Langevin dynamics still converges to the right weights, by crossing between modes. In practice the low-density gap makes crossings exponentially rare, so each chain stays in the mode it fell into, and the proportions reflect where the chains started (Figure 5b).

## The fix: add noise

All three problems have the same remedy: **perturb the data with noise**.
- Noise spreads the data off the manifold, so $q_\sigma$ has full support and a well-defined score everywhere.
- Noise fills the empty regions with probability mass, so the score is trained, and hence accurate, where chains start.
- Noise **merges modes**. At large $\sigma$ the mixture becomes one blob whose shape reflects the mixing weights. Chains started there distribute themselves correctly before the modes separate.

The trade-off is that large noise destroys the data: we want samples from $p$, not from a blurred $q_\sigma$. Figure 4 shows the whole range: small $\sigma$ is accurate but has empty regions, and large $\sigma$ covers everything but has lost the structure.

![[Machine Learning/images/score-based-models-noise-levels.svg]]
*Figure 4. The noisy density $q_\sigma$ and its score directions for the three-mode mixture of Figure 1, at increasing noise. With clean data the modes are isolated and the space between them is empty; by $\sigma=1$ they are connected; by $\sigma=2$ there is one broad mode, with more mass towards the heavier component.*

The resolution is not to choose one $\sigma$ but to use **many**: start sampling with a large noise level to find the right region and the right proportions, and move to smaller noise levels to recover the detail.

## Noise-conditional score networks

A **noise-conditional score network** (NCSN)[^song2019] is a single network $\mathbf{s}_\theta(\mathbf{x},\sigma)$ that takes the noise level as an input. It is trained on a geometric sequence of noise levels

$$
\sigma_1>\sigma_2>\cdots>\sigma_L,
\qquad \frac{\sigma_{i}}{\sigma_{i+1}}=\text{const},
$$

with $\sigma_1$ large enough to wash out the data's structure and $\sigma_L$ small enough that $q_{\sigma_L}\approx p$. The loss is a weighted sum of DSM losses (4), one per level:

$$
\mathcal{L}(\theta)=\frac1L\sum_{i=1}^{L}\lambda(\sigma_i)\,\frac12\,\mathbb{E}_{p(\mathbf{x})}\mathbb{E}_{\boldsymbol\epsilon}\Big[\Big\lVert\mathbf{s}_\theta(\mathbf{x}+\sigma_i\boldsymbol\epsilon,\sigma_i)+\frac{\boldsymbol\epsilon}{\sigma_i}\Big\rVert^2\Big].
$$

How should the levels be weighted? The target $\boldsymbol\epsilon/\sigma_i$ has magnitude about $\sqrt{D}/\sigma_i$, so without weighting, the smallest noise levels would dominate the loss. Choosing $\lambda(\sigma)=\sigma^2$ makes every level contribute equally:

$$
\mathcal{L}(\theta)=\frac{1}{2L}\sum_{i=1}^{L}\mathbb{E}\Big[\big\lVert\sigma_i\,\mathbf{s}_\theta(\mathbf{x}+\sigma_i\boldsymbol\epsilon,\sigma_i)+\boldsymbol\epsilon\big\rVert^2\Big].
\tag{6}
$$

Each term is again "predict the noise", now with the noise level given. In practice we sample one random level per data point instead of summing over all $L$.

### Annealed Langevin dynamics

Sampling runs Langevin dynamics at each noise level in turn, from the noisiest to the cleanest, starting each level where the previous one ended:

> [!example] Algorithm: annealed Langevin dynamics
> 1. Initialize $\mathbf{x}$ from a broad distribution, e.g. $\mathcal{N}(\mathbf{0},\sigma_1^2\mathbf{I})$ or uniform noise.
> 2. For $i=1,\dots,L$: set the step size $\eta_i=\eta\,\sigma_i^2/\sigma_L^2$.
> 3. Within level $i$, repeat $K$ times: draw $\mathbf{z}\sim\mathcal{N}(\mathbf{0},\mathbf{I})$ and update $\mathbf{x}\leftarrow\mathbf{x}+\frac{\eta_i}{2}\,\mathbf{s}_\theta(\mathbf{x},\sigma_i)+\sqrt{\eta_i}\,\mathbf{z}$.
> 4. Optionally, finish with one denoising step from Tweedie's formula, $\mathbf{x}\leftarrow\mathbf{x}+\sigma_L^2\,\mathbf{s}_\theta(\mathbf{x},\sigma_L)$, to remove the last bit of noise.
> 5. Return $\mathbf{x}$.

The step size shrinks with $\sigma_i^2$ for a reason. The score has magnitude about $1/\sigma_i$, so the drift $\tfrac{\eta_i}{2}\mathbf{s}_\theta$ is about $\eta_i/\sigma_i$ while the noise is about $\sqrt{\eta_i}$. With $\eta_i\propto\sigma_i^2$, their ratio, the **signal-to-noise ratio** of a step, is the same at every level. Each level is then "the same amount of work", just at a different scale.

Figures 5 and 6 show annealed Langevin on two toy problems with exact (analytic) noisy scores. With the clean score, Langevin gets the mode weights badly wrong. Annealing gets them nearly right, and it covers all eight modes of a ring evenly.

![[Machine Learning/images/score-based-models-mode-weights.svg]]
*Figure 5. A mixture with weights 0.8 (lower left) and 0.2 (upper right), 1500 chains started uniformly in $[-8,8]^2$. (a) True samples: 82% lower left. (b) Langevin with the exact clean score, 1000 steps: 53%, because each chain stays in the mode nearest its start. (c) Annealed Langevin over 10 noise levels from $\sigma=10$ to $0.1$, 100 steps each: 77%, close to the true 80%.*

![[Machine Learning/images/score-based-models-annealed.svg]]
*Figure 6. Annealed Langevin dynamics on a ring of eight Gaussians, through 12 noise levels from $\sigma=8$ to $0.05$ (60 steps per level). The samples go from structureless, to a blob with the ring's overall size, to eight clusters. At the end every mode holds between 11% and 13% of the 2000 samples.*

### Practical recipe

Song and Ermon's follow-up work turned these ideas into rules of thumb that made NCSNs competitive with GANs on images:[^song2020]
- **Largest noise:** $\sigma_1$ about the largest distance between any two training points, so that one noisy distribution covers them all.
- **Spacing:** consecutive levels must overlap. In $D$ dimensions, $\mathbf{x}+\sigma\boldsymbol\epsilon$ concentrates on a thin shell of radius about $\sigma\sqrt{D}$ around $\mathbf{x}$, so the ratio $\sigma_i/\sigma_{i+1}$ must be close to 1 when $D$ is large. Image models use hundreds to thousands of levels.
- **Noise conditioning:** a simple parameterization $\mathbf{s}_\theta(\mathbf{x},\sigma)=\mathbf{s}_\theta(\mathbf{x})/\sigma$ works well, reflecting the $1/\sigma$ scale of the true score.
- **Stability:** keep an exponential moving average of the weights for sampling.

The main cost is sampling time: $L\times K$ network evaluations, often thousands, against one for a GAN or VAE.

## Infinitely many noise levels: the SDE view

What happens as the number of levels $L\to\infty$? The perturbation becomes a continuous-time process that adds noise gradually,[^song2021sde] a **stochastic differential equation** (SDE):

$$
d\mathbf{x}=\mathbf{f}(\mathbf{x},t)\,dt+g(t)\,d\mathbf{w},
\qquad t\in[0,1],
\tag{7}
$$

where $\mathbf{w}$ is a Brownian motion (so $d\mathbf{w}$ is Gaussian noise with variance $dt$). With $\mathbf{f}=\mathbf{0}$ and $g(t)^2=\tfrac{d}{dt}\sigma(t)^2$, the noise variance at time $t$ is $\sigma(t)^2$: this is the continuous version of the NCSN noise sequence, called the **variance-exploding** SDE. Let $p_t$ be the density at time $t$, so $p_0=p$ and $p_1$ is nearly Gaussian.

A classical result of Anderson says this process can be run **backwards in time**,[^anderson1982] and the reverse process is again an SDE:

$$
d\mathbf{x}=\Big[\mathbf{f}(\mathbf{x},t)-g(t)^2\,\nabla_{\mathbf{x}}\log p_t(\mathbf{x})\Big]dt+g(t)\,d\bar{\mathbf{w}},
\tag{8}
$$

with time flowing from $1$ to $0$ and $\bar{\mathbf{w}}$ a reverse-time Brownian motion. The only unknown in (8) is the score of the noisy data at every time, $\nabla\log p_t$, which a time-conditional network $\mathbf{s}_\theta(\mathbf{x},t)$ learns with denoising score matching (6), now with $t$ sampled continuously. Generating data means solving (8) numerically from noise; annealed Langevin is one particular, rather crude, way of doing so.

Two further results from the SDE view set up the next two lectures:
- A different choice of $\mathbf{f}$ and $g$, which shrinks the data while adding noise so that the total variance stays fixed, gives the **variance-preserving** SDE. Its discretization is the denoising diffusion probabilistic model (DDPM).
- There is a **deterministic** process with the same marginal densities $p_t$, the **probability flow ODE** (9) below. It maps noise to data along smooth paths, with no randomness at all. It is a continuous normalizing flow, and learning such flows directly is the subject of flow matching.

$$
\frac{d\mathbf{x}}{dt}=\mathbf{f}(\mathbf{x},t)-\frac12\,g(t)^2\,\nabla_{\mathbf{x}}\log p_t(\mathbf{x}).
\tag{9}
$$

## Key takeaways

- The score $\nabla_{\mathbf{x}}\log p$ is normalizer-free, determines the density up to a constant, and is all Langevin sampling needs. Score-based models learn it directly.
- Score matching fits the score without knowing it: by integration by parts (implicit, needs a Jacobian trace), random projections (sliced, one Jacobian-vector product), or denoising (no Jacobian at all; learns the score of the noise-blurred data).
- Tweedie's formula, $\mathbb{E}[\mathbf{x}\mid\tilde{\mathbf{x}}]=\tilde{\mathbf{x}}+\sigma^2\nabla\log q_\sigma(\tilde{\mathbf{x}})$, makes "optimal denoiser", "noise predictor" and "score" three names for one function.
- With a single small noise level, sampling fails: scores are wrong in low-density regions, and Langevin cannot recover mode weights. Many noise levels, with $\lambda(\sigma)=\sigma^2$ weighting and annealed Langevin with $\eta_i\propto\sigma_i^2$, fix both.
- In the limit of continuous noise, sampling is solving a reverse-time SDE whose only unknown is the score, or an ODE with the same marginals.

## Where this shows up next

- [[Energy-Based Models]]: an EBM is a score model whose field is constrained to be a gradient, $\mathbf{s}_\theta=-\nabla E_\theta$; implicit score matching was first proposed to train EBMs.
- [[Non-parametric Models#Kernel density estimation|Kernel density estimation]]: the noisy data distribution $q_\sigma$ of a finite dataset is a Gaussian KDE, and its exact score is a responsibility-weighted pull towards the training points.
- [[Clustering and Expectation Maximization]]: the score of a mixture is a responsibility-weighted average of component scores.
- [[Variational Autoencoders]]: denoising diffusion models are also very deep VAEs with a fixed noising encoder; the next lecture shows that their ELBO reduces to the noise-prediction loss (6).
- Next lectures (planned): **denoising diffusion models**, where the VAE route and the score route meet, and then **flow matching**, which learns the probability-flow ODE's velocity field directly.

<!-- References: footnote definitions below render as the numbered "References" list at the bottom of the page -->

[^hyvarinen2005]: Hyvärinen, A. (2005). *Estimation of Non-Normalized Statistical Models by Score Matching*. JMLR, 6, 695–709. [jmlr.org/papers/v6/hyvarinen05a](https://jmlr.org/papers/v6/hyvarinen05a.html)
[^song2019sliced]: Song, Y., Garg, S., Shi, J., & Ermon, S. (2019). *Sliced Score Matching: A Scalable Approach to Density and Score Estimation*. UAI. [arXiv:1905.07088](https://arxiv.org/abs/1905.07088)
[^hutchinson1989]: Hutchinson, M. F. (1989). *A Stochastic Estimator of the Trace of the Influence Matrix for Laplacian Smoothing Splines*. Communications in Statistics – Simulation and Computation, 18(3), 1059–1076. [doi:10.1080/03610918908812806](https://doi.org/10.1080/03610918908812806)
[^vincent2011]: Vincent, P. (2011). *A Connection Between Score Matching and Denoising Autoencoders*. Neural Computation, 23(7), 1661–1674. [doi:10.1162/NECO_a_00142](https://doi.org/10.1162/NECO_a_00142)
[^efron2011]: Efron, B. (2011). *Tweedie's Formula and Selection Bias*. Journal of the American Statistical Association, 106(496), 1602–1614. [doi:10.1198/jasa.2011.tm11181](https://doi.org/10.1198/jasa.2011.tm11181)
[^alain2014]: Alain, G., & Bengio, Y. (2014). *What Regularized Auto-Encoders Learn from the Data-Generating Distribution*. JMLR, 15, 3563–3593. [arXiv:1211.4246](https://arxiv.org/abs/1211.4246)
[^song2019]: Song, Y., & Ermon, S. (2019). *Generative Modeling by Estimating Gradients of the Data Distribution*. NeurIPS. [arXiv:1907.05600](https://arxiv.org/abs/1907.05600). See also Song's accessible [blog post](https://yang-song.net/blog/2021/score/) of the same name (2021).
[^song2020]: Song, Y., & Ermon, S. (2020). *Improved Techniques for Training Score-Based Generative Models*. NeurIPS. [arXiv:2006.09011](https://arxiv.org/abs/2006.09011)
[^song2021sde]: Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2021). *Score-Based Generative Modeling through Stochastic Differential Equations*. ICLR. [arXiv:2011.13456](https://arxiv.org/abs/2011.13456)
[^anderson1982]: Anderson, B. D. O. (1982). *Reverse-Time Diffusion Equation Models*. Stochastic Processes and their Applications, 12(3), 313–326. [doi:10.1016/0304-4149(82)90051-5](https://doi.org/10.1016/0304-4149%2882%2990051-5)
