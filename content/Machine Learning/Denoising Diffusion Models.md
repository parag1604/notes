---
title: Denoising Diffusion Models
tags:
  - generative-models
  - diffusion-models
  - variational-inference
  - score-matching
date: 2026-11-13
draft: false
prev: "[[Score-Based Models]]"
next: "[[Flow Matching]]"
---

> **Prerequisites:** [[Variational Autoencoders]] (the ELBO and [[Variational Autoencoders#The Monte Carlo ELBO|the KL divergence between two Gaussians]]), [[Score-Based Models]] (especially [[Score-Based Models#Denoising score matching|denoising score matching]] and [[Score-Based Models#Tweedie's formula: the score is the optimal denoiser|Tweedie's formula]]), [[Joint Distributions#The multivariate Gaussian|the multivariate Gaussian]] (sums and conditionals of Gaussians), and [[Markov Chains]]. The network is a [[Convolutional Neural Networks#U-Net|U-Net]] with [[Sequence Models#Order: positional encodings|sinusoidal positional encodings]] of the time step.
>
> **TL;DR:**
> - A **denoising diffusion probabilistic model** (DDPM) is a hierarchical VAE with $T$ latents of the same size as the data and a **fixed** encoder that adds a little Gaussian noise at each step, until only noise is left. Only the decoder, which removes the noise step by step, is learned.
> - Its ELBO splits into a reconstruction term, a prior-matching term that does not depend on the parameters, and $T-1$ **denoising matching** terms. Each of these is a KL divergence between Gaussians, which is a weighted regression: predict the clean data $\mathbf{x}_0$ from a noisy $\mathbf{x}_t$.
> - Predicting $\mathbf{x}_0$, predicting the added noise $\boldsymbol\epsilon$, and predicting the score $\nabla\log q_t$ are the same model in three parameterizations. Noise prediction is denoising score matching, so a DDPM is a noise-conditional score network in different coordinates.
> - **Guidance** steers sampling towards a condition $y$ by adding a classifier's gradient, or, without a classifier, by extrapolating from an unconditional to a conditional prediction.
> - **DDIM** reuses a trained DDPM with a family of non-Markovian forward processes that have the same marginals. Its deterministic member samples in tens of steps instead of a thousand and can be run backwards to encode data.
>
> **Notation in this lecture:** time runs $t=0,1,\dots,T$ with $\mathbf{x}_0$ the data and $\mathbf{x}_1,\dots,\mathbf{x}_T$ the latents. This differs from the series convention ($t=0..T-1$ for an RL horizon) and is the standard indexing in the diffusion literature. The schedule is $\alpha_t\in(0,1)$, $\beta_t=1-\alpha_t$, $\bar\alpha_t=\prod_{j=1}^{t}\alpha_j$, with $\bar\alpha_0=1$. The ELBO is $F$, as in the VAE lecture; losses to be minimized are $\mathcal{L}$; the learning rate is $\rho$, because $\alpha$ and $\beta$ are taken by the schedule and $\eta$ is Song et al.'s DDIM stochasticity parameter (23). Networks take the time step as an explicit input: $\boldsymbol\epsilon_\theta(\mathbf{x}_t,t)$, $\hat{\mathbf{x}}_\theta(\mathbf{x}_t,t)$, $\mathbf{s}_\theta(\mathbf{x}_t,t)$.
>
> **Code:** [Image-Based-Generative-Models](https://github.com/parag1604/Image-Based-Generative-Models): a DDPM on 64×64 Stanford Cars images, with a small U-Net noise predictor (residual blocks, GroupNorm, sinusoidal time embeddings), the linear schedule, EMA weights, and both the 1000-step ancestral sampler and 50-step DDIM sampling.

## Motivation

The [[Variational Autoencoders|VAE]] generates an image in one decoder pass from a low-dimensional code, and its samples are blurry. The [[Generative Adversarial Networks|GAN]] also generates in one pass, with sharp samples, but its training is a fragile two-player game and it has no likelihood. [[Score-Based Models]] showed a third way: learn the score of the data blurred by noise at many scales, then walk from noise to data with annealed Langevin dynamics.

Denoising diffusion models reach the same place from the VAE side. The idea is simple to state. Destroying an image is easy: add a little noise, then a little more, until nothing is left. If we could learn to **undo one small step** of that destruction, we could start from pure noise and undo all of them, one at a time, and end with a new image. Each step is a small, easy denoising problem, and a small step of Gaussian noise can be undone by a step that is itself approximately Gaussian.[^sohl2015] The whole model is trained by a plain regression, with no adversary and no sampling during training.

Sohl-Dickstein et al. introduced this idea in 2015,[^sohl2015] and Ho et al. made it competitive in 2020 with the DDPM.[^ho2020] They now produce most generated images and video. This lecture follows the route of Luo's tutorial,[^luo2022] which reads the DDPM as a hierarchical VAE.

## DDPMs as hierarchical VAEs

We are given data $\mathcal{D}=\{\mathbf{x}_0\}$ drawn from an unknown distribution $p_{\text{data}}$. The goal is to learn to sample from $p_{\text{data}}$.

DDPMs are latent-variable generative models, best seen as a special case of **hierarchical VAEs**. Recall that a VAE has one latent space: an encoder $q_\phi(\mathbf{z}\mid\mathbf{x})$ maps data to a latent code and a decoder $p_\theta(\mathbf{x}\mid\mathbf{z})$ maps it back. A hierarchical VAE has several latent spaces, $\mathbf{x}\to\mathbf{z}_1\to\mathbf{z}_2\to\cdots\to\mathbf{z}_T$ when encoding and $\mathbf{z}_T\to\cdots\to\mathbf{z}_1\to\mathbf{x}$ when decoding. A DDPM is a hierarchical VAE with three properties (Figure 1):
1. **There are many latent spaces**, $\mathbf{z}_1,\dots,\mathbf{z}_T$, often $T=1000$.
2. **Each latent has the same dimension as the data**: $\mathbf{z}_t\in\mathbb{R}^d$ for every $t$, if $\mathbf{x}\in\mathbb{R}^d$. There is no bottleneck.
3. **The encoder is fixed, not learned.** In a VAE both the encoder $q_\phi$ and the decoder $p_\theta$ are neural networks. In a DDPM the encoder $q$ has no parameters; only the decoder is learned.

![[Machine Learning/images/denoising-diffusion-models-hvae.svg]]
*Figure 1. (a) A VAE learns both directions between the data and one latent code. (b) A DDPM has $T$ latents of the same size as the data. The forward (encoding) direction is a fixed noising process; only the reverse (decoding) direction is learned.*

In the diffusion literature the data point is written $\mathbf{x}_0$ (the VAE's $\mathbf{x}$) and the latents are $\mathbf{x}_1,\dots,\mathbf{x}_T$ (the VAE's $\mathbf{z}_1,\dots,\mathbf{z}_T$). Subscripts are now **time steps**, not data-point indices. We write $\mathbf{x}_{1:T}=(\mathbf{x}_1,\dots,\mathbf{x}_T)$ and $\mathbf{x}_{0:T}=(\mathbf{x}_0,\dots,\mathbf{x}_T)$.

## The forward (encoding) process

The encoder of a DDPM adds noise, one step at a time. Fix $T$ (a hyperparameter) and scalars $\alpha_1,\dots,\alpha_T\in(0,1)$, the **noise schedule**. Then

$$
\mathbf{x}_t=\sqrt{\alpha_t}\,\mathbf{x}_{t-1}+\sqrt{1-\alpha_t}\,\boldsymbol\epsilon_t,
\qquad\boldsymbol\epsilon_t\sim\mathcal{N}(\mathbf{0},\mathbf{I}),
\qquad t=1,\dots,T,
\tag{1}
$$

with the noises $\boldsymbol\epsilon_1,\dots,\boldsymbol\epsilon_T$ independent of each other and of $\mathbf{x}_0$. Data $\mathbf{x}_0$, add noise, get noisy data $\mathbf{x}_1$; add noise, get noisier data $\mathbf{x}_2$; and so on, until $\mathbf{x}_T$ is noise only (Figure 2). Ho et al. write the per-step noise variance as $\beta_t=1-\alpha_t$; both names are used below. The endpoints matter: $\alpha_t=0$ would destroy all the signal in one step, and $\alpha_t=1$ would add no noise. In practice the steps are small, $\beta_t$ between $10^{-4}$ and $0.02$, and they grow with $t$.

![[Machine Learning/images/denoising-diffusion-models-forward-house.svg]]
*Figure 2. The forward process on a $48\times48$ picture of a house, with the linear schedule of Ho et al. ($T=1000$, $\beta_t$ from $10^{-4}$ to $0.02$), using the closed form (3) below with one fixed noise image. The house is still clear at $t=100$, faint at $t=400$, and gone at $t=1000$, where $\bar\alpha_T\approx4\times10^{-5}$.*

> [!tip] Intuition: why shrink as well as add noise?
> Each step scales the signal down by $\sqrt{\alpha_t}$ before adding noise of variance $1-\alpha_t$. If $\mathbf{x}_{t-1}$ has unit variance, so does $\mathbf{x}_t$: $\alpha_t\cdot1+(1-\alpha_t)=1$. The process is **variance preserving**. Data scaled to unit size stay at unit size, and the noise is not allowed to blow up. (Noise-conditional score networks do the opposite: they add noise without shrinking, so the variance explodes. The two are compared in [[#DDPM and NCSN are the same model|a later section]].)

Because $\mathbf{x}_t$ depends on the past only through $\mathbf{x}_{t-1}$ and fresh, independent noise, (1) is a first-order [[Markov Chains|Markov chain]]: $\mathbf{x}_t$ is conditionally independent of $\mathbf{x}_{t-2},\dots,\mathbf{x}_0$ given $\mathbf{x}_{t-1}$. Its transitions are Gaussian. An affine function of a Gaussian is Gaussian, so

$$
q(\mathbf{x}_t\mid\mathbf{x}_{t-1})=\mathcal{N}\big(\mathbf{x}_t;\ \underbrace{\sqrt{\alpha_t}\,\mathbf{x}_{t-1}}_{\text{mean}},\ \underbrace{(1-\alpha_t)\mathbf{I}}_{\text{covariance}}\big).
\tag{2}
$$

These are the conditional forward, or encoding, distributions. The chain is **time-inhomogeneous**, since $\alpha_t$ changes with $t$.

### Jumping to any time step

We never need to run the chain step by step. Unroll (1) twice:

$$
\begin{aligned}
\mathbf{x}_t&=\sqrt{\alpha_t}\big(\sqrt{\alpha_{t-1}}\,\mathbf{x}_{t-2}+\sqrt{1-\alpha_{t-1}}\,\boldsymbol\epsilon_{t-1}\big)+\sqrt{1-\alpha_t}\,\boldsymbol\epsilon_t\\
&=\sqrt{\alpha_t\alpha_{t-1}}\,\mathbf{x}_{t-2}+\underbrace{\sqrt{\alpha_t-\alpha_t\alpha_{t-1}}\,\boldsymbol\epsilon_{t-1}+\sqrt{1-\alpha_t}\,\boldsymbol\epsilon_t}_{\text{sum of two independent Gaussians}}.
\end{aligned}
$$

[[Joint Distributions#Sums of independent random variables|Independent Gaussians add]], with variances adding: the underbraced sum is $\mathcal{N}(\mathbf{0},v\mathbf{I})$ with $v=(\alpha_t-\alpha_t\alpha_{t-1})+(1-\alpha_t)=1-\alpha_t\alpha_{t-1}$. So $\mathbf{x}_t=\sqrt{\alpha_t\alpha_{t-1}}\,\mathbf{x}_{t-2}+\sqrt{1-\alpha_t\alpha_{t-1}}\,\bar{\boldsymbol\epsilon}$ with a single standard Gaussian $\bar{\boldsymbol\epsilon}$. This has the same form as (1), with $\alpha_t$ replaced by $\alpha_t\alpha_{t-1}$. By induction on the number of steps unrolled, all the way down to $\mathbf{x}_0$,

$$
q(\mathbf{x}_t\mid\mathbf{x}_0)=\mathcal{N}\big(\mathbf{x}_t;\ \sqrt{\bar\alpha_t}\,\mathbf{x}_0,\ (1-\bar\alpha_t)\mathbf{I}\big),
\qquad
\bar\alpha_t=\prod_{j=1}^{t}\alpha_j,
\tag{3}
$$

or, as a reparameterization,

$$
\mathbf{x}_t=\sqrt{\bar\alpha_t}\,\mathbf{x}_0+\sqrt{1-\bar\alpha_t}\,\boldsymbol\epsilon,
\qquad\boldsymbol\epsilon\sim\mathcal{N}(\mathbf{0},\mathbf{I}).
\tag{4}
$$

The $t$-th latent can be sampled directly from the data, without hopping through $t$ steps. This is what makes training cheap.

Since every $\alpha_t<1$, $\bar\alpha_t$ decreases towards 0, and $q(\mathbf{x}_T\mid\mathbf{x}_0)\approx\mathcal{N}(\mathbf{0},\mathbf{I})$ whatever $\mathbf{x}_0$ was. The approximation is very good but not exact: for the linear schedule, $\bar\alpha_T\approx4\times10^{-5}$, so a trace of $\mathbf{x}_0$, scaled by $\sqrt{\bar\alpha_T}\approx0.006$, remains. $\mathcal{N}(\mathbf{0},\mathbf{I})$ is the stationary distribution of each kernel (2), and the chain approaches it as $\bar\alpha_t\to0$ (Figure 3, and the [[Markov Chains#Towards diffusion models and state space models|Markov Chains lecture]]).

![[Machine Learning/images/denoising-diffusion-models-forward-moons.svg]]
*Figure 3. The forward process (4) on two-moons data, coloured by moon, with the linear schedule. The moons blur by $t=200$ and merge into one blob by $t=400$; at $t=1000$ the points are standard Gaussian (sample mean $(0.015,-0.001)$, variance $(0.999,0.992)$). This 2-D dataset is the running example: every later figure of samples comes from one small network trained on it.*

## The reverse (decoding) process

The model is defined through the decoder. It runs the chain backwards, from noise to data, as a first-order Markov chain with Gaussian transitions whose parameters are learned:

$$
p_\theta(\mathbf{x}_{0:T})=p(\mathbf{x}_T)\prod_{t=1}^{T}p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t),
\qquad p(\mathbf{x}_T)=\mathcal{N}(\mathbf{0},\mathbf{I}),
\tag{5}
$$

$$
p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t)=\mathcal{N}\big(\mathbf{x}_{t-1};\ \boldsymbol\mu_\theta(\mathbf{x}_t,t),\ \boldsymbol\Sigma_\theta(\mathbf{x}_t,t)\big).
$$

Like the VAE's $p_\theta(\mathbf{x},\mathbf{z})$, this is a joint distribution over data and latents. The prior $p(\mathbf{x}_T)$ is fixed, with no parameters. To generate, draw $\mathbf{x}_T\sim\mathcal{N}(\mathbf{0},\mathbf{I})$ and sample down the chain to $\mathbf{x}_0$.

Why should Gaussian reverse steps be good enough? The true reverse of a forward step, $q(\mathbf{x}_{t-1}\mid\mathbf{x}_t)$, is not Gaussian in general: it mixes over all the data that could have led to $\mathbf{x}_t$. But when the forward step adds only a little noise, the reverse step only has to undo a little noise, and it is then approximately Gaussian. This observation, which goes back to Feller's work on diffusion processes, is why diffusion models use many small steps.[^sohl2015]

$\boldsymbol\mu_\theta$ and $\boldsymbol\Sigma_\theta$ are learned by maximizing the ELBO.

## The ELBO of a DDPM

### From the VAE to the DDPM

For a VAE, $\log p_\theta(\mathbf{x})=\log\int p_\theta(\mathbf{x},\mathbf{z})\,d\mathbf{z}$, and Jensen's inequality gives the [[Variational Autoencoders#The ELBO for one data point|evidence lower bound]] with the encoder as the variational distribution. For a DDPM, replace $\mathbf{z}$ by $\mathbf{x}_{1:T}$ and the encoder by the fixed forward process:

$$
\begin{aligned}
\log p_\theta(\mathbf{x}_0)&=\log\int q(\mathbf{x}_{1:T}\mid\mathbf{x}_0)\,\frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T}\mid\mathbf{x}_0)}\,d\mathbf{x}_{1:T}\\
&\ge\mathbb{E}_{q(\mathbf{x}_{1:T}\mid\mathbf{x}_0)}\left[\log\frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T}\mid\mathbf{x}_0)}\right]=:F(\theta;\mathbf{x}_0).
\end{aligned}
\tag{6}
$$

$F$ is the ELBO for one data point. The training objective is its average over the data, $\mathbb{E}_{\mathbf{x}_0\sim p_{\text{data}}}F(\theta;\mathbf{x}_0)$, maximized over $\theta$. As for the VAE, the gap $\log p_\theta(\mathbf{x}_0)-F$ is the KL divergence from $q(\mathbf{x}_{1:T}\mid\mathbf{x}_0)$ to the model's posterior; here $q$ is fixed, so only the decoder can close it.

### Three terms

Both joints factorize along the chain. The decoder does by (5). The encoder does by the chain rule followed by the Markov property:

$$
\begin{aligned}
q(\mathbf{x}_{1:T}\mid\mathbf{x}_0)&=q(\mathbf{x}_1\mid\mathbf{x}_0)\,q(\mathbf{x}_2\mid\mathbf{x}_1,\mathbf{x}_0)\cdots q(\mathbf{x}_T\mid\mathbf{x}_{T-1},\dots,\mathbf{x}_0)\\
&=\prod_{t=1}^{T}q(\mathbf{x}_t\mid\mathbf{x}_{t-1}).
\end{aligned}
$$

Split off the $t=1$ factors, which need separate treatment:

$$
\log\frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T}\mid\mathbf{x}_0)}
=\log\frac{p(\mathbf{x}_T)\,p_\theta(\mathbf{x}_0\mid\mathbf{x}_1)}{q(\mathbf{x}_1\mid\mathbf{x}_0)}+\sum_{t=2}^{T}\log\frac{p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t)}{q(\mathbf{x}_t\mid\mathbf{x}_{t-1})}.
$$

The terms in the sum compare a reverse step with a forward step, which point in opposite directions. To compare like with like, turn the forward step around with Bayes' rule. For $t\ge2$, the Markov property lets us add $\mathbf{x}_0$ to the conditioning, and then [[Probability Basics#Bayes' theorem|Bayes' theorem]] (with everything conditioned on $\mathbf{x}_0$) gives

$$
q(\mathbf{x}_t\mid\mathbf{x}_{t-1})=q(\mathbf{x}_t\mid\mathbf{x}_{t-1},\mathbf{x}_0)=\frac{q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)\,q(\mathbf{x}_t\mid\mathbf{x}_0)}{q(\mathbf{x}_{t-1}\mid\mathbf{x}_0)}.
\tag{7}
$$

(At $t=1$ this would condition $\mathbf{x}_0$ on itself, which is why that step was split off.) $q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)$ is the distribution of $\mathbf{x}_{t-1}$ given that the forward process, started at $\mathbf{x}_0$, has landed at $\mathbf{x}_t$. Substituting (7), the sum becomes

$$
\sum_{t=2}^{T}\log\frac{p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t)}{q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)}+\sum_{t=2}^{T}\log\frac{q(\mathbf{x}_{t-1}\mid\mathbf{x}_0)}{q(\mathbf{x}_t\mid\mathbf{x}_0)}.
$$

The second sum **telescopes**: every $q(\mathbf{x}_t\mid\mathbf{x}_0)$ with $2\le t\le T-1$ appears once in a numerator and once in a denominator, leaving $\log q(\mathbf{x}_1\mid\mathbf{x}_0)-\log q(\mathbf{x}_T\mid\mathbf{x}_0)$. The $q(\mathbf{x}_1\mid\mathbf{x}_0)$ cancels the one in the first term, and the log-ratio becomes

$$
\log p_\theta(\mathbf{x}_0\mid\mathbf{x}_1)+\log\frac{p(\mathbf{x}_T)}{q(\mathbf{x}_T\mid\mathbf{x}_0)}+\sum_{t=2}^{T}\log\frac{p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t)}{q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)}.
$$

Now take the expectation over $q(\mathbf{x}_{1:T}\mid\mathbf{x}_0)$. Each term depends on one or two of the latents, so the others integrate out: the first term needs only $q(\mathbf{x}_1\mid\mathbf{x}_0)$, the second only $q(\mathbf{x}_T\mid\mathbf{x}_0)$, and the $t$-th term of the sum only $q(\mathbf{x}_{t-1},\mathbf{x}_t\mid\mathbf{x}_0)$. Factor that last one as $q(\mathbf{x}_t\mid\mathbf{x}_0)\,q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)$, so its expectation is an inner expectation over $\mathbf{x}_{t-1}$ inside an outer one over $\mathbf{x}_t$ (the tower rule of [[Joint Distributions#Conditional expectation|conditional expectation]]). The inner expectations are KL divergences, and the ELBO is

$$
\begin{aligned}
F(\theta;\mathbf{x}_0)={}&\underbrace{\mathbb{E}_{q(\mathbf{x}_1\mid\mathbf{x}_0)}\big[\log p_\theta(\mathbf{x}_0\mid\mathbf{x}_1)\big]}_{\text{(i) reconstruction}}
-\underbrace{D_{\mathrm{KL}}\big(q(\mathbf{x}_T\mid\mathbf{x}_0)\,\|\,p(\mathbf{x}_T)\big)}_{\text{(ii) prior matching}}\\
&-\sum_{t=2}^{T}\underbrace{\mathbb{E}_{q(\mathbf{x}_t\mid\mathbf{x}_0)}\Big[D_{\mathrm{KL}}\big(q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)\,\|\,p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t)\big)\Big]}_{\text{(iii) denoising matching}}.
\end{aligned}
\tag{8}
$$

The three terms:
1. **Reconstruction**: how well the last decoder step recovers the data from the first latent. It is the VAE's reconstruction term.
2. **Prior matching**: how close the final latent is to the prior. It has **no parameters**, since both $q$ and $p(\mathbf{x}_T)$ are fixed, so it can be dropped from the optimization. It is small because $\bar\alpha_T\approx0$.
3. **Denoising matching**: at each step, the learned denoising step $p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t)$ should match the "known" denoising step $q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)$, which knows where the chain started.

The objective is $\theta^*=\arg\max_\theta\mathbb{E}_{\mathbf{x}_0}F(\theta;\mathbf{x}_0)$. Everything now hinges on term (iii), and for that we need $q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)$.

## The known denoising step

Bayes' rule, as in (7), gives the reverse of a forward step when the start is known:

$$
q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)=\frac{q(\mathbf{x}_t\mid\mathbf{x}_{t-1})\,q(\mathbf{x}_{t-1}\mid\mathbf{x}_0)}{q(\mathbf{x}_t\mid\mathbf{x}_0)}.
$$

All three distributions on the right are Gaussian and known, from (2) and (3). As a function of $\mathbf{x}_{t-1}$ the result is again Gaussian, which we find by completing the square:

$$
q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)=\mathcal{N}\big(\mathbf{x}_{t-1};\ \boldsymbol\mu_q(\mathbf{x}_t,\mathbf{x}_0),\ \tilde\beta_t\mathbf{I}\big),
\tag{9}
$$

$$
\begin{aligned}
\boldsymbol\mu_q(\mathbf{x}_t,\mathbf{x}_0)&=\frac{\sqrt{\alpha_t}\,(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\,\mathbf{x}_t+\frac{\sqrt{\bar\alpha_{t-1}}\,(1-\alpha_t)}{1-\bar\alpha_t}\,\mathbf{x}_0,\\
\tilde\beta_t&=\frac{(1-\alpha_t)(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}.
\end{aligned}
\tag{10}
$$

Both the mean and the variance are computable, so $q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)$ is completely known. (We write this variance as Ho et al.'s $\tilde\beta_t$, because $\sigma_t$ is needed for the decoder's variance below.)

> [!proof]- Proof: completing the square
> The denominator $q(\mathbf{x}_t\mid\mathbf{x}_0)$ does not involve $\mathbf{x}_{t-1}$, so it is a normalizing constant. Up to terms that do not depend on $\mathbf{x}_{t-1}$, the log of the numerator is
>
> $$
> -\frac12\left[\frac{\lVert\mathbf{x}_t-\sqrt{\alpha_t}\,\mathbf{x}_{t-1}\rVert^2}{1-\alpha_t}+\frac{\lVert\mathbf{x}_{t-1}-\sqrt{\bar\alpha_{t-1}}\,\mathbf{x}_0\rVert^2}{1-\bar\alpha_{t-1}}\right].
> $$
>
> Expand and collect the terms in $\mathbf{x}_{t-1}$. The quadratic term is $-\tfrac12 a\lVert\mathbf{x}_{t-1}\rVert^2$, and the linear term is $\mathbf{b}^\top\mathbf{x}_{t-1}$, with
>
> $$
> \begin{aligned}
> a&=\frac{\alpha_t}{1-\alpha_t}+\frac{1}{1-\bar\alpha_{t-1}}=\frac{1-\bar\alpha_t}{(1-\alpha_t)(1-\bar\alpha_{t-1})},\\
> \mathbf{b}&=\frac{\sqrt{\alpha_t}}{1-\alpha_t}\,\mathbf{x}_t+\frac{\sqrt{\bar\alpha_{t-1}}}{1-\bar\alpha_{t-1}}\,\mathbf{x}_0.
> \end{aligned}
> $$
>
> The simplification of $a$ uses $\alpha_t(1-\bar\alpha_{t-1})+(1-\alpha_t)=1-\alpha_t\bar\alpha_{t-1}=1-\bar\alpha_t$. A log-density of the form $-\tfrac12a\lVert\mathbf{x}\rVert^2+\mathbf{b}^\top\mathbf{x}+\text{const}$ is that of $\mathcal{N}(\mathbf{b}/a,\ a^{-1}\mathbf{I})$, since $-\tfrac{a}{2}\lVert\mathbf{x}-\mathbf{b}/a\rVert^2$ expands to the same quadratic and linear terms. So the variance is $1/a=\tilde\beta_t$ and the mean is
>
> $$
> \frac{\mathbf{b}}{a}=\tilde\beta_t\,\mathbf{b}=\frac{\sqrt{\alpha_t}\,(1-\bar\alpha_{t-1})\,\mathbf{x}_t+\sqrt{\bar\alpha_{t-1}}\,(1-\alpha_t)\,\mathbf{x}_0}{1-\bar\alpha_t},
> $$
>
> which is (10). The same result follows from the [[Joint Distributions#The multivariate Gaussian|Gaussian conditioning formula]], since $(\mathbf{x}_{t-1},\mathbf{x}_t)$ is jointly Gaussian given $\mathbf{x}_0$.

The mean is a **precision-weighted average** of two guesses for $\mathbf{x}_{t-1}$: one from where the chain is now ($\mathbf{x}_t/\sqrt{\alpha_t}$, undoing the last shrink) and one from where it started ($\sqrt{\bar\alpha_{t-1}}\,\mathbf{x}_0$). Each is weighted by how certain it is. The posterior is narrower than either, as Figure 4 shows in one dimension.

![[Machine Learning/images/denoising-diffusion-models-posterior.svg]]
*Figure 4. Completing the square in one dimension, with an exaggerated step ($\bar\alpha_{t-1}=0.5$, $\alpha_t=0.6$, $x_0=1.5$, $x_t=0.4$). The prior $q(x_{t-1}\mid x_0)$ and the likelihood of the observed $x_t$ multiply to the posterior (10), whose mean $\mu_q$ lies between them and whose variance is smaller than either. Numerical integration of the product matches (10): mean 0.8274, variance 0.2857.*

## The denoising matching term is a regression

### Choosing the decoder's variance

The decoder steps are Gaussian, and so are the targets (9). Their KL divergence has a [[Variational Autoencoders#The Monte Carlo ELBO|closed form]]. To make it simple, Ho et al. **fix** the decoder's covariance instead of learning it: $\boldsymbol\Sigma_\theta(\mathbf{x}_t,t)=\sigma_t^2\mathbf{I}$, with $\sigma_t^2$ a constant of the schedule. Two choices are natural: $\sigma_t^2=\tilde\beta_t$, the variance of the target, and $\sigma_t^2=\beta_t$. They gave similar results,[^ho2020] and the [[#Noise schedules and variances|noise-schedule section]] explains why. (A learned variance is a refinement covered there too.)

With equal covariances, $\sigma_t^2=\tilde\beta_t$, the KL divergence between $\mathcal{N}(\boldsymbol\mu_q,\tilde\beta_t\mathbf{I})$ and $\mathcal{N}(\boldsymbol\mu_\theta,\tilde\beta_t\mathbf{I})$ in $d$ dimensions is

$$
\begin{aligned}
D_{\mathrm{KL}}&=\frac12\left[\frac{\lVert\boldsymbol\mu_\theta-\boldsymbol\mu_q\rVert^2}{\tilde\beta_t}+\operatorname{tr}(\mathbf{I})-\log\frac{|\tilde\beta_t\mathbf{I}|}{|\tilde\beta_t\mathbf{I}|}-d\right]\\
&=\frac{1}{2\tilde\beta_t}\big\lVert\boldsymbol\mu_\theta(\mathbf{x}_t,t)-\boldsymbol\mu_q(\mathbf{x}_t,\mathbf{x}_0)\big\rVert^2.
\end{aligned}
$$

The trace cancels the $-d$, and the log-determinant vanishes. With any other fixed $\sigma_t^2$, the trace and log-determinant terms no longer cancel but still do not depend on $\theta$, and the weight becomes $1/(2\sigma_t^2)$. Either way, **each denoising matching term is a regression of $\boldsymbol\mu_\theta$ onto $\boldsymbol\mu_q$**, with weight $1/(2\sigma_t^2)$, which is different for each $t$.

### The first step

The reconstruction term needs care. At $t=1$ the formula (10) gives $\tilde\beta_1=0$, because $\bar\alpha_0=1$: given $\mathbf{x}_1$ and $\mathbf{x}_0$, there is nothing left to guess about $\mathbf{x}_0$. A decoder $p_\theta(\mathbf{x}_0\mid\mathbf{x}_1)$ with zero variance would make $\log p_\theta$ infinite, so the last step gets its own fixed variance $\sigma_1^2$, for example $\beta_1$. With $p_\theta(\mathbf{x}_0\mid\mathbf{x}_1)=\mathcal{N}(\mathbf{x}_0;\boldsymbol\mu_\theta(\mathbf{x}_1,1),\sigma_1^2\mathbf{I})$,

$$
\log p_\theta(\mathbf{x}_0\mid\mathbf{x}_1)=-\frac{\lVert\mathbf{x}_0-\boldsymbol\mu_\theta(\mathbf{x}_1,1)\rVert^2}{2\sigma_1^2}-\frac d2\log(2\pi\sigma_1^2),
$$

and the second term does not depend on $\theta$. (For 8-bit images Ho et al. use a discretized Gaussian decoder, which gives a proper likelihood for integer pixel values.[^ho2020])

### The objective so far

Dropping the terms that do not depend on $\theta$ (the prior matching term and the constants),

$$
\begin{aligned}
F(\theta;\mathbf{x}_0)={}&-\,\mathbb{E}_{q(\mathbf{x}_1\mid\mathbf{x}_0)}\frac{\lVert\mathbf{x}_0-\boldsymbol\mu_\theta(\mathbf{x}_1,1)\rVert^2}{2\sigma_1^2}\\
&-\sum_{t=2}^{T}\mathbb{E}_{q(\mathbf{x}_t\mid\mathbf{x}_0)}\frac{\lVert\boldsymbol\mu_\theta(\mathbf{x}_t,t)-\boldsymbol\mu_q(\mathbf{x}_t,\mathbf{x}_0)\rVert^2}{2\sigma_t^2}+\text{const},
\end{aligned}
\tag{11}
$$

where every $\mathbf{x}_t$ is drawn from the forward process (3), not from the prior. The basic idea is now to represent the unknown $\boldsymbol\mu_\theta$ by a neural network.

**One network for all time steps.** Rather than $T$ separate networks, a DDPM uses one network that takes the noisy input $\mathbf{x}_t$ **and** the time $t$. The time tells the network how much noise to expect. It enters through a **sinusoidal embedding**, the same as a transformer's [[Sequence Models#Order: positional encodings|positional encoding]]: the integer $t$ becomes a vector $\hat{\mathbf{t}}\in\mathbb{R}^{d_{\text{emb}}}$ with

$$
\hat t_{2k}=\sin(\omega_k t),\qquad\hat t_{2k+1}=\cos(\omega_k t),\qquad\omega_k=10000^{-2k/d_{\text{emb}}},
$$

for $k=0,\dots,d_{\text{emb}}/2-1$. Low $k$ gives fast oscillations that tell nearby steps apart; high $k$ gives slow ones that locate $t$ coarsely.

For images the network is a [[Convolutional Neural Networks#U-Net|U-Net]]:[^ronneberger2015] it reduces the spatial resolution, then increases it back to the input's size, with skip connections between matching resolutions, so its output has the same shape as $\mathbf{x}_t$.

## Predicting the clean data

What should the network output? Look at the target in (10): $\boldsymbol\mu_q$ is a fixed combination of $\mathbf{x}_t$, which the network sees, and $\mathbf{x}_0$, which it does not. So let the network predict only the unknown part. Define the decoder's mean with the same formula, with $\mathbf{x}_0$ replaced by the network's estimate $\hat{\mathbf{x}}_\theta(\mathbf{x}_t,t)$:

$$
\boldsymbol\mu_\theta(\mathbf{x}_t,t)=\frac{\sqrt{\alpha_t}\,(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\,\mathbf{x}_t+\frac{\sqrt{\bar\alpha_{t-1}}\,(1-\alpha_t)}{1-\bar\alpha_t}\,\hat{\mathbf{x}}_\theta(\mathbf{x}_t,t).
\tag{12}
$$

This is a design choice, not a consequence of the ELBO; the next section makes a different one. Subtracting (10) from (12), the $\mathbf{x}_t$ terms cancel, and each denoising matching term becomes

$$
\frac{1}{2\sigma_t^2}\lVert\boldsymbol\mu_\theta-\boldsymbol\mu_q\rVert^2
=\frac{1}{2\sigma_t^2}\,\frac{\bar\alpha_{t-1}(1-\alpha_t)^2}{(1-\bar\alpha_t)^2}\,\big\lVert\hat{\mathbf{x}}_\theta(\mathbf{x}_t,t)-\mathbf{x}_0\big\rVert^2.
$$

With $\sigma_t^2=\tilde\beta_t$ the weight simplifies to a neat form. Define the **signal-to-noise ratio** of $\mathbf{x}_t$ in (4), $\mathrm{SNR}(t)=\bar\alpha_t/(1-\bar\alpha_t)$. Then

$$
\frac{\bar\alpha_{t-1}(1-\alpha_t)^2}{2\tilde\beta_t(1-\bar\alpha_t)^2}=\frac12\big(\mathrm{SNR}(t-1)-\mathrm{SNR}(t)\big).
\tag{13}
$$

The weight of step $t$ is the amount of signal-to-noise ratio that step destroys.[^kingma2021] The first step fits the same pattern: at $t=1$, (12) gives $\boldsymbol\mu_\theta(\mathbf{x}_1,1)=\hat{\mathbf{x}}_\theta(\mathbf{x}_1,1)$, because $1-\bar\alpha_1=1-\alpha_1$. So **every term of the ELBO is a weighted regression of $\hat{\mathbf{x}}_\theta(\mathbf{x}_t,t)$ onto $\mathbf{x}_0$**:

$$
F(\theta;\mathbf{x}_0)=-\sum_{t=1}^{T}\nu_t\,\mathbb{E}_{q(\mathbf{x}_t\mid\mathbf{x}_0)}\big\lVert\hat{\mathbf{x}}_\theta(\mathbf{x}_t,t)-\mathbf{x}_0\big\rVert^2+\text{const},
\qquad \nu_1=\frac{1}{2\sigma_1^2}.
$$

The network is a **denoiser** for the data point $\mathbf{x}_0$, at every noise level $t$. Unlike the decoder of a VAE or the generator of a GAN, it does **not** output a new sample from $p_{\text{data}}$. Its output is the best guess of the clean data, which for squared error is the [[Joint Distributions#The best predictor|conditional mean]] $\mathbb{E}[\mathbf{x}_0\mid\mathbf{x}_t]$. At large $t$ that is a blurry average of everything the noise could have hidden. New samples come from chaining many of its small steps.

> [!proof]- Proof of (13)
> Substitute $\tilde\beta_t=(1-\alpha_t)(1-\bar\alpha_{t-1})/(1-\bar\alpha_t)$:
>
> $$
> \frac{\bar\alpha_{t-1}(1-\alpha_t)^2}{2\tilde\beta_t(1-\bar\alpha_t)^2}=\frac{\bar\alpha_{t-1}(1-\alpha_t)}{2(1-\bar\alpha_{t-1})(1-\bar\alpha_t)}.
> $$
>
> On the other side,
>
> $$
> \begin{aligned}
> \mathrm{SNR}(t-1)-\mathrm{SNR}(t)&=\frac{\bar\alpha_{t-1}(1-\bar\alpha_t)-\bar\alpha_t(1-\bar\alpha_{t-1})}{(1-\bar\alpha_{t-1})(1-\bar\alpha_t)}\\
> &=\frac{\bar\alpha_{t-1}-\bar\alpha_t}{(1-\bar\alpha_{t-1})(1-\bar\alpha_t)},
> \end{aligned}
> $$
>
> and $\bar\alpha_{t-1}-\bar\alpha_t=\bar\alpha_{t-1}(1-\alpha_t)$.

### Training

Training needs no simulation of the chain: by (4), a noisy $\mathbf{x}_t$ at any step is one line of code away. Instead of summing over all $T$ terms for every data point, sample one term at random. Ho et al. also found that **dropping the weights** $\nu_t$ gives better samples (more in the next section).[^ho2020]

> [!example] Algorithm: training a DDPM ($\mathbf{x}_0$-prediction)
> Repeat until convergence, for a minibatch of data points:
> 1. Draw $\mathbf{x}_0$ from the data, a time step $t\sim\operatorname{Uniform}\{1,\dots,T\}$, and noise $\boldsymbol\epsilon\sim\mathcal{N}(\mathbf{0},\mathbf{I})$.
> 2. Form $\mathbf{x}_t=\sqrt{\bar\alpha_t}\,\mathbf{x}_0+\sqrt{1-\bar\alpha_t}\,\boldsymbol\epsilon$, a sample of $q(\mathbf{x}_t\mid\mathbf{x}_0)$.
> 3. Forward pass: compute $\hat{\mathbf{x}}_\theta(\mathbf{x}_t,t)$.
> 4. Backward pass: take a gradient step $\theta\leftarrow\theta-\rho\,\nabla_\theta\big\lVert\hat{\mathbf{x}}_\theta(\mathbf{x}_t,t)-\mathbf{x}_0\big\rVert^2$, averaged over the minibatch.

Each data point is used at a different random $t$ every time it is seen, so over training the network learns to denoise at every noise level.

### Sampling

To get a new sample we traverse the decoder (5): draw $\mathbf{x}_T\sim\mathcal{N}(\mathbf{0},\mathbf{I})$, then $\mathbf{x}_{T-1}\sim p_\theta(\mathbf{x}_{T-1}\mid\mathbf{x}_T)$, and so on down to $\mathbf{x}_0$. This is **ancestral sampling**.

> [!example] Algorithm: sampling from a DDPM ($\mathbf{x}_0$-prediction)
> 1. Draw $\mathbf{x}_T\sim\mathcal{N}(\mathbf{0},\mathbf{I})$.
> 2. For $t=T,T-1,\dots,1$: a forward pass gives $\hat{\mathbf{x}}_\theta(\mathbf{x}_t,t)$, and (12) gives the mean $\boldsymbol\mu_\theta(\mathbf{x}_t,t)$.
> 3. Set $\mathbf{x}_{t-1}=\boldsymbol\mu_\theta(\mathbf{x}_t,t)+\sigma_t\,\boldsymbol\xi$, with $\boldsymbol\xi\sim\mathcal{N}(\mathbf{0},\mathbf{I})$ if $t>1$ and $\boldsymbol\xi=\mathbf{0}$ if $t=1$; go back to step 2 until $t=1$ is done.
> 4. Return $\mathbf{x}_0$, a new sample.

Two details. The noise is scaled by the **standard deviation** $\sigma_t$, not the variance, since $p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t)$ has covariance $\sigma_t^2\mathbf{I}$. And the last step returns the mean without noise, as in Ho et al.'s algorithm. Sampling needs $T$ forward passes through the network, against one for a VAE or a GAN. Figure 5 shows the process on the running example.

![[Machine Learning/images/denoising-diffusion-models-reverse.svg]]
*Figure 5. Ancestral sampling with the trained 2-D model, 1000 steps, 3000 samples. The Gaussian blob changes little until about $t=300$, has the overall shape by $t=100$, and splits into two sharp moons over the last hundred steps: Figure 3 in reverse.*

## Alternate interpretations

### (a) Predicting the added noise

By (4), the noisy point determines the clean one once we know the noise that was added:

$$
\mathbf{x}_0=\frac{\mathbf{x}_t-\sqrt{1-\bar\alpha_t}\,\boldsymbol\epsilon}{\sqrt{\bar\alpha_t}}.
\tag{14}
$$

Substituting (14) into the posterior mean (10) and simplifying (the coefficient of $\mathbf{x}_t$ collapses because $\alpha_t(1-\bar\alpha_{t-1})+1-\alpha_t=1-\bar\alpha_t$) writes $\boldsymbol\mu_q$ in terms of the noise:

$$
\boldsymbol\mu_q(\mathbf{x}_t,\boldsymbol\epsilon)=\frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t-\frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\,\boldsymbol\epsilon\right).
$$

So parameterize the decoder mean by a **noise predictor** $\boldsymbol\epsilon_\theta(\mathbf{x}_t,t)$:

$$
\boldsymbol\mu_\theta(\mathbf{x}_t,t)=\frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t-\frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\,\boldsymbol\epsilon_\theta(\mathbf{x}_t,t)\right).
\tag{15}
$$

The denoising matching term becomes a regression over the added noise:

$$
\frac{1}{2\sigma_t^2}\lVert\boldsymbol\mu_q-\boldsymbol\mu_\theta\rVert^2=\frac{1}{2\sigma_t^2}\,\frac{(1-\alpha_t)^2}{\alpha_t(1-\bar\alpha_t)}\,\big\lVert\boldsymbol\epsilon-\boldsymbol\epsilon_\theta(\mathbf{x}_t,t)\big\rVert^2.
$$

This is the parameterization of Ho et al., and dropping the weights gives their **simple loss**,[^ho2020] with $t\sim\operatorname{Uniform}\{1,\dots,T\}$:

$$
\mathcal{L}_{\text{simple}}(\theta)=\mathbb{E}_{\mathbf{x}_0,\,t,\,\boldsymbol\epsilon}\Big[\big\lVert\boldsymbol\epsilon-\boldsymbol\epsilon_\theta\big(\sqrt{\bar\alpha_t}\,\mathbf{x}_0+\sqrt{1-\bar\alpha_t}\,\boldsymbol\epsilon,\ t\big)\big\rVert^2\Big].
\tag{16}
$$

Look at a noisy point and guess the noise that was added: exactly the [[Score-Based Models#Denoising score matching|denoising score matching]] loss. Compared with the ELBO, $\mathcal{L}_{\text{simple}}$ gives relatively less weight to the small-$t$ terms, which remove imperceptible amounts of noise, and more to the large-$t$ terms, which decide the content of the image. It is no longer a bound on the likelihood, but its samples are better. Training and sampling only change in one line each:

> [!example] Algorithm: DDPM with noise prediction (Ho et al., Algorithms 1 and 2)
> **Training.** Repeat: draw $\mathbf{x}_0$, $t\sim\operatorname{Uniform}\{1,\dots,T\}$ and $\boldsymbol\epsilon\sim\mathcal{N}(\mathbf{0},\mathbf{I})$; take a gradient step on $\big\lVert\boldsymbol\epsilon-\boldsymbol\epsilon_\theta(\sqrt{\bar\alpha_t}\,\mathbf{x}_0+\sqrt{1-\bar\alpha_t}\,\boldsymbol\epsilon,\,t)\big\rVert^2$.
>
> **Sampling.** Draw $\mathbf{x}_T\sim\mathcal{N}(\mathbf{0},\mathbf{I})$. For $t=T,\dots,1$, draw $\boldsymbol\xi\sim\mathcal{N}(\mathbf{0},\mathbf{I})$ if $t>1$ (else $\boldsymbol\xi=\mathbf{0}$) and set
>
> $$
> \mathbf{x}_{t-1}=\frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t-\frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\,\boldsymbol\epsilon_\theta(\mathbf{x}_t,t)\right)+\sigma_t\boldsymbol\xi.
> $$
>
> Return $\mathbf{x}_0$.

> [!example]- Code: a minimal DDPM in PyTorch
> Train a small noise predictor on 2-D data and sample it with the ancestral sampler. On the two-moons data of the figures, 20 000 training steps take a few minutes on a laptop CPU.
> ```python
> import torch, torch.nn as nn
> T = 1000
> beta = torch.linspace(1e-4, 0.02, T)               # beta[t-1] is beta_t, for t = 1..T
> alpha = 1 - beta
> abar = torch.cumprod(alpha, 0)                     # abar[t-1] is \bar alpha_t
>
> def embed(t, d=64):                                # sinusoidal embedding of the integer time t
>     k = torch.arange(d // 2)
>     ang = t[:, None].float() * 10000 ** (-2 * k / d)
>     return torch.cat([ang.sin(), ang.cos()], 1)
>
> class EpsNet(nn.Module):                           # eps_theta(x_t, t), an MLP for 2-D data
>     def __init__(self, D=2, h=256):
>         super().__init__()
>         self.net = nn.Sequential(nn.Linear(D + 64, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(),
>                                  nn.Linear(h, h), nn.SiLU(), nn.Linear(h, D))
>     def forward(self, x, t):
>         return self.net(torch.cat([x, embed(t)], 1))
>
> def train(model, data, steps=20_000, batch=512, lr=1e-3):
>     opt = torch.optim.Adam(model.parameters(), lr=lr)
>     for _ in range(steps):
>         x0 = data[torch.randint(len(data), (batch,))]
>         t = torch.randint(1, T + 1, (batch,))      # t ~ Uniform{1, ..., T}
>         eps = torch.randn_like(x0)
>         ab = abar[t - 1, None]
>         xt = ab.sqrt() * x0 + (1 - ab).sqrt() * eps        # a sample of q(x_t | x_0)
>         loss = ((model(xt, t) - eps) ** 2).mean()          # L_simple
>         opt.zero_grad(); loss.backward(); opt.step()
>     return model
>
> @torch.no_grad()
> def sample_ddpm(model, n, D=2):
>     x = torch.randn(n, D)                          # x_T ~ N(0, I)
>     for t in range(T, 0, -1):
>         eps = model(x, torch.full((n,), t))
>         mean = (x - beta[t - 1] / (1 - abar[t - 1]).sqrt() * eps) / alpha[t - 1].sqrt()
>         if t > 1:                                  # sigma_t^2 = tilde beta_t
>             var = beta[t - 1] * (1 - abar[t - 2]) / (1 - abar[t - 1])
>             x = mean + var.sqrt() * torch.randn_like(x)
>         else:
>             x = mean                               # no noise at the last step
>     return x
> ```

### (b) Predicting the score

[[Score-Based Models#Tweedie's formula: the score is the optimal denoiser|Tweedie's formula]] connects denoising to the score. In its general form: if $\mathbf{u}\mid\boldsymbol\mu\sim\mathcal{N}(\boldsymbol\mu,\boldsymbol\Sigma)$ for a random $\boldsymbol\mu$ with any prior, and $p(\mathbf{u})$ is the marginal density of $\mathbf{u}$, then $\mathbb{E}[\boldsymbol\mu\mid\mathbf{u}]=\mathbf{u}+\boldsymbol\Sigma\nabla_{\mathbf{u}}\log p(\mathbf{u})$. In a DDPM, (3) says $\mathbf{x}_t$ is Gaussian around $\boldsymbol\mu=\sqrt{\bar\alpha_t}\,\mathbf{x}_0$ with covariance $(1-\bar\alpha_t)\mathbf{I}$, and the prior on $\mathbf{x}_0$ is the data distribution. Write $q_t(\mathbf{x}_t)=\int p_{\text{data}}(\mathbf{x}_0)\,q(\mathbf{x}_t\mid\mathbf{x}_0)\,d\mathbf{x}_0$ for the **marginal** density of the noisy data. Tweedie gives the posterior mean of the clean data:

$$
\mathbb{E}[\mathbf{x}_0\mid\mathbf{x}_t]=\frac{\mathbf{x}_t+(1-\bar\alpha_t)\,\nabla_{\mathbf{x}_t}\log q_t(\mathbf{x}_t)}{\sqrt{\bar\alpha_t}}.
\tag{17}
$$

This is the best estimate of $\mathbf{x}_0$ from $\mathbf{x}_t$, a function of $\mathbf{x}_t$ alone; it is not $\mathbf{x}_0$ itself. Comparing with (14), the posterior mean of the noise is

$$
\mathbb{E}[\boldsymbol\epsilon\mid\mathbf{x}_t]=-\sqrt{1-\bar\alpha_t}\,\nabla_{\mathbf{x}_t}\log q_t(\mathbf{x}_t).
$$

For a single known $\mathbf{x}_0$ the relation holds exactly, without any average. The **conditional** score of (3) is

$$
\nabla_{\mathbf{x}_t}\log q(\mathbf{x}_t\mid\mathbf{x}_0)=-\frac{\mathbf{x}_t-\sqrt{\bar\alpha_t}\,\mathbf{x}_0}{1-\bar\alpha_t}=-\frac{\boldsymbol\epsilon}{\sqrt{1-\bar\alpha_t}}.
\tag{18}
$$

**The true score is the negatively scaled noise:** it points from $\mathbf{x}_t$ back towards the clean point, against the noise that was added. The marginal score is the average of (18) over all clean points that could have produced $\mathbf{x}_t$.

Substituting (18) into the noise form of $\boldsymbol\mu_q$ gives the score form,

$$
\boldsymbol\mu_q=\frac{1}{\sqrt{\alpha_t}}\,\mathbf{x}_t+\frac{1-\alpha_t}{\sqrt{\alpha_t}}\,\nabla_{\mathbf{x}_t}\log q(\mathbf{x}_t\mid\mathbf{x}_0),
\qquad
\boldsymbol\mu_\theta=\frac{1}{\sqrt{\alpha_t}}\,\mathbf{x}_t+\frac{1-\alpha_t}{\sqrt{\alpha_t}}\,\mathbf{s}_\theta(\mathbf{x}_t,t),
$$

and the denoising matching term becomes $\frac{(1-\alpha_t)^2}{2\sigma_t^2\alpha_t}\lVert\mathbf{s}_\theta(\mathbf{x}_t,t)-\nabla_{\mathbf{x}_t}\log q(\mathbf{x}_t\mid\mathbf{x}_0)\rVert^2$: a regression onto the **conditional** score, which we know. That is denoising score matching, and [[Score-Based Models#Denoising score matching|Vincent's theorem]] says its minimizer is the **marginal** score $\nabla\log q_t$, which we do not know.[^vincent2011] So a network trained to regress the added noise is implicitly predicting the score of the noisy data, with

$$
\mathbf{s}_\theta(\mathbf{x}_t,t)=-\frac{\boldsymbol\epsilon_\theta(\mathbf{x}_t,t)}{\sqrt{1-\bar\alpha_t}}.
\tag{19}
$$

### One network, three parameterizations

The three outputs are interchangeable. A network trained for one can be read as either of the others, and only the weighting of the loss over $t$ differs.

| Network predicts | Regression target | Conversion |
|---|---|---|
| clean data $\hat{\mathbf{x}}_\theta(\mathbf{x}_t,t)$ | $\mathbf{x}_0$ | $\hat{\mathbf{x}}_\theta=(\mathbf{x}_t-\sqrt{1-\bar\alpha_t}\,\boldsymbol\epsilon_\theta)/\sqrt{\bar\alpha_t}$ |
| noise $\boldsymbol\epsilon_\theta(\mathbf{x}_t,t)$ | $\boldsymbol\epsilon$ | $\boldsymbol\epsilon_\theta=-\sqrt{1-\bar\alpha_t}\,\mathbf{s}_\theta$ |
| score $\mathbf{s}_\theta(\mathbf{x}_t,t)$ | $\nabla\log q(\mathbf{x}_t\mid\mathbf{x}_0)$ | $\mathbf{s}_\theta=(\sqrt{\bar\alpha_t}\,\hat{\mathbf{x}}_\theta-\mathbf{x}_t)/(1-\bar\alpha_t)$ |

At the optimum all three are posterior means: $\mathbb{E}[\mathbf{x}_0\mid\mathbf{x}_t]$, $\mathbb{E}[\boldsymbol\epsilon\mid\mathbf{x}_t]$, and $\nabla\log q_t=\mathbb{E}[\nabla\log q(\mathbf{x}_t\mid\mathbf{x}_0)\mid\mathbf{x}_t]$. In practice they behave differently. Noise prediction is well scaled at every $t$ (the target always has unit variance), which is why it became the default. Clean-data prediction is easier near $t=T$, where predicting the noise means predicting almost all of $\mathbf{x}_t$. A fourth choice, **velocity** $\mathbf{v}=\sqrt{\bar\alpha_t}\,\boldsymbol\epsilon-\sqrt{1-\bar\alpha_t}\,\mathbf{x}_0$, interpolates between the two and is used in distillation and many large models.[^salimans2022]

### DDPM and NCSN are the same model

Divide (4) by $\sqrt{\bar\alpha_t}$:

$$
\tilde{\mathbf{x}}=\frac{\mathbf{x}_t}{\sqrt{\bar\alpha_t}}=\mathbf{x}_0+\bar\sigma\,\boldsymbol\epsilon,
\qquad
\bar\sigma=\sqrt{\frac{1-\bar\alpha_t}{\bar\alpha_t}}=\frac{1}{\sqrt{\mathrm{SNR}(t)}}.
$$

Here $\bar\sigma$ is the noise level of the [[Score-Based Models#Noise-conditional score networks|noise-conditional score network]] (written $\sigma$ in that lecture). The rescaled DDPM latent is exactly the noisy data $\tilde{\mathbf{x}}$ of the score-based lecture, at noise level $\bar\sigma$. A DDPM step $t$ **is** an NCSN noise level, and the linear schedule spans $\bar\sigma$ from about 0.01 at $t=1$ to about 157 at $t=T$. The scores agree up to the change of variables, $\nabla_{\mathbf{x}_t}\log q_t(\mathbf{x}_t)=\nabla_{\tilde{\mathbf{x}}}\log q_{\bar\sigma}(\tilde{\mathbf{x}})/\sqrt{\bar\alpha_t}$, so by (19) the DDPM noise predictor and the NCSN noise predictor $-\bar\sigma\,\mathbf{s}_\theta(\tilde{\mathbf{x}},\bar\sigma)$ are the same function. And each term of the DDPM loss (16) is the NCSN loss at one noise level, with the weighting $\bar\sigma^2$ of [[Score-Based Models#Noise-conditional score networks|that lecture's (6)]]; only the distribution of noise levels differs. So DDPM and NCSN learn the same thing, up to parameterization. They differ in the coordinates (DDPM shrinks the data so the variance is preserved, NCSN lets it explode), in the spacing of the noise levels, and in the sampler (ancestral sampling against annealed Langevin dynamics). The [[#The SDE view|SDE view]] below puts both in one framework.

## Noise schedules and variances

The schedule $\alpha_1,\dots,\alpha_T$ decides how fast the data are destroyed, and so which noise levels the network spends its effort on. Ho et al. used a **linear** schedule, $\beta_t$ from $10^{-4}$ to $0.02$ over $T=1000$ steps. Nichol and Dhariwal noticed that it destroys information too fast: the last part of the chain is nearly pure noise, and those steps are wasted.[^nichol2021] Their **cosine** schedule defines $\bar\alpha_t$ directly,

$$
\bar\alpha_t=\frac{h(t)}{h(0)},\qquad h(t)=\cos^2\left(\frac{t/T+s}{1+s}\cdot\frac{\pi}{2}\right),\qquad s=0.008,
$$

with $\beta_t=1-\bar\alpha_t/\bar\alpha_{t-1}$ clipped at $0.999$ to avoid a singular last step. It decreases $\bar\alpha_t$ roughly linearly in the middle and slowly at both ends (Figure 6).

> [!example]- Code: the cosine schedule
> A drop-in replacement for the linear `beta` of the DDPM code above.
> ```python
> import math, torch
> def cosine_schedule(T=1000, s=0.008):
>     t = torch.arange(T + 1, dtype=torch.float64) / T
>     f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
>     abar = f / f[0]                                         # \bar alpha_t, t = 0..T
>     beta = (1 - abar[1:] / abar[:-1]).clamp(max=0.999)      # beta_t, t = 1..T, clipped
>     return beta.float()
>
> beta = cosine_schedule()
> alpha = 1 - beta
> abar = torch.cumprod(alpha, 0)                              # recomputed after clipping
> ```

The **signal-to-noise ratio** $\mathrm{SNR}(t)=\bar\alpha_t/(1-\bar\alpha_t)$ is the cleanest way to compare schedules. It must start high (almost clean data) and end near zero, so that $q(\mathbf{x}_T\mid\mathbf{x}_0)$ is close to the prior and the prior matching term is negligible. For the linear schedule, $\bar\alpha_T\approx4.0\times10^{-5}$; for the cosine schedule, $\bar\alpha_T\approx2.4\times10^{-9}$. In between, the linear schedule has already reached $\mathrm{SNR}<10^{-2}$ for 33% of its steps, the cosine schedule for only 6%. In continuous time, the variational diffusion model analysis shows that the ELBO depends on the schedule only through the SNR at the two endpoints; the schedule in between only changes the variance of the Monte Carlo estimate, and, for the simple loss, the weighting.[^kingma2021]

![[Machine Learning/images/denoising-diffusion-models-schedules.svg]]
*Figure 6. Linear and cosine schedules, $T=1000$. (a) $\bar\alpha_t$. The linear schedule falls below 0.01 at $t=674$, the cosine one at $t=936$. (b) The log signal-to-noise ratio. (c) The ratio of the two natural decoder variances, $\tilde\beta_t/\beta_t$, is close to 1 except in the first few steps. (d) The ELBO weight on $\lVert\boldsymbol\epsilon-\boldsymbol\epsilon_\theta\rVert^2$ for $\sigma_t^2=\tilde\beta_t$. For the linear schedule it is 0.6 at $t=2$ and between 0.005 and 0.01 over most of the range, so the ELBO spends much of its effort on the first few, nearly noiseless steps; the unweighted loss (16) removes this. The cosine schedule's weight also spikes at the very end, where clipping $\beta_t$ at 0.999 makes $\alpha_t$ tiny.*

**Which decoder variance?** Panel (c) explains why $\sigma_t^2=\beta_t$ and $\sigma_t^2=\tilde\beta_t$ work equally well with $T=1000$: they differ only at the first few steps, which change the sample imperceptibly. They are the two extremes for the true reverse variance: $\beta_t$ is right when the data are themselves Gaussian noise, $\tilde\beta_t$ when the data are a single point.[^ho2020] With fewer steps the difference matters. Nichol and Dhariwal **learn** the variance as an interpolation between the two, $\boldsymbol\Sigma_\theta=\exp\big(\mathbf{v}_\theta\log\beta_t+(1-\mathbf{v}_\theta)\log\tilde\beta_t\big)$, trained with $\mathcal{L}_{\text{simple}}$ plus a small multiple of the full ELBO (since $\mathcal{L}_{\text{simple}}$ does not involve the variance). This improves log-likelihoods and allows sampling with fewer steps.[^nichol2021]

## Guided diffusion for conditional generation

Often the data come with a condition: pairs $(\mathbf{x}_0,y)$ where $y$ is a class label or a text embedding. We want to sample from $p(\mathbf{x}_0\mid y)$. How should a DDPM be modified?

### Classifier guidance

A DDPM predicts the score of the unconditional, or marginal, noisy distribution $\nabla_{\mathbf{x}_t}\log q_t(\mathbf{x}_t)$. For conditional sampling we need the **conditional score** $\nabla_{\mathbf{x}_t}\log q_t(\mathbf{x}_t\mid y)$. Bayes' theorem relates them:

$$
\begin{aligned}
\nabla_{\mathbf{x}_t}\log q_t(\mathbf{x}_t\mid y)&=\nabla_{\mathbf{x}_t}\log\frac{q_t(\mathbf{x}_t)\,p_t(y\mid\mathbf{x}_t)}{p(y)}\\
&=\underbrace{\nabla_{\mathbf{x}_t}\log q_t(\mathbf{x}_t)}_{\text{unconditional score}}+\underbrace{\nabla_{\mathbf{x}_t}\log p_t(y\mid\mathbf{x}_t)}_{\text{classifier gradient}},
\end{aligned}
$$

since $p(y)$ does not depend on $\mathbf{x}_t$. So train a classifier (or regressor) $p_\psi(y\mid\mathbf{x}_t,t)$ separately, and add its input gradient to the diffusion model's score at every sampling step.[^dhariwal2021] In noise units, by (19),

$$
\tilde{\boldsymbol\epsilon}(\mathbf{x}_t,t)=\boldsymbol\epsilon_\theta(\mathbf{x}_t,t)-\gamma\sqrt{1-\bar\alpha_t}\,\nabla_{\mathbf{x}_t}\log p_\psi(y\mid\mathbf{x}_t,t),
$$

which replaces $\boldsymbol\epsilon_\theta$ in the sampling update. The **guidance scale** $\gamma\ge0$ was introduced by Dhariwal and Nichol: $\gamma=1$ is the conditional score, and $\gamma>1$ samples from a sharpened distribution proportional to $q_t(\mathbf{x}_t)\,p_\psi(y\mid\mathbf{x}_t)^\gamma$. Larger $\gamma$ gives samples that are more clearly of class $y$, and less diverse.

> [!example]- Code: classifier guidance
> ```python
> # assumes: model(x, t) -> eps_theta; classifier(x, t) -> (B, K) logits of p_psi(y | x_t, t);
> #          abar from the DDPM code above; t (B,) integer times; y (B,) target labels
> def guided_eps(x, t, y, gamma):
>     with torch.enable_grad():                        # the samplers run under no_grad
>         x_in = x.detach().requires_grad_(True)
>         logp = torch.log_softmax(classifier(x_in, t), -1)[torch.arange(len(y)), y]
>         grad, = torch.autograd.grad(logp.sum(), x_in)   # grad_x log p_psi(y | x_t, t)
>     return model(x, t) - gamma * (1 - abar[t - 1, None]).sqrt() * grad
> ```
> Use `guided_eps` in place of `model(x, t)` in `sample_ddpm` (or `sample_ddim` below). The gradient is taken with respect to the input only; neither network is updated.

Three points about this recipe:
- **Guidance happens only at sampling time.** The diffusion model is trained as before, unconditionally, and the classifier is trained on its own. Nothing is backpropagated into $\theta$; the gradient is with respect to the **input** $\mathbf{x}_t$.
- **The classifier must work on noisy inputs.** It sees $\mathbf{x}_t$ at every noise level, so it has to be trained on noised data with $t$ as an input. An off-the-shelf image classifier cannot be used.
- **Its gradients can be poor.** A classifier's input gradient can be uninformative or adversarial, pushing $\mathbf{x}_t$ to fool the classifier rather than to look like class $y$.

### Classifier-free guidance

The last two points motivate guidance **without** a classifier.[^ho2022] Bayes' rule can be turned around to write the classifier gradient with two diffusion models:

$$
\nabla_{\mathbf{x}_t}\log p_t(y\mid\mathbf{x}_t)=\nabla_{\mathbf{x}_t}\log q_t(\mathbf{x}_t\mid y)-\nabla_{\mathbf{x}_t}\log q_t(\mathbf{x}_t).
$$

Put this "implicit classifier" in place of the real one, scaled by a guidance weight $\lambda$. The result is a **guided score**, which is not the score of $q_t(\mathbf{x}_t\mid y)$ unless $\lambda=1$:

$$
\begin{aligned}
\tilde{\mathbf{s}}_\lambda(\mathbf{x}_t,y)&=\nabla_{\mathbf{x}_t}\log q_t(\mathbf{x}_t)+\lambda\big(\nabla_{\mathbf{x}_t}\log q_t(\mathbf{x}_t\mid y)-\nabla_{\mathbf{x}_t}\log q_t(\mathbf{x}_t)\big)\\
&=\lambda\,\nabla_{\mathbf{x}_t}\log q_t(\mathbf{x}_t\mid y)+(1-\lambda)\,\nabla_{\mathbf{x}_t}\log q_t(\mathbf{x}_t).
\end{aligned}
$$

It is the score of $q_t(\mathbf{x}_t)\,p_t(y\mid\mathbf{x}_t)^\lambda$, the same sharpening as classifier guidance. In noise units, writing $\lambda=1+w$,

$$
\tilde{\boldsymbol\epsilon}=(1+w)\,\boldsymbol\epsilon_\theta(\mathbf{x}_t,t,y)-w\,\boldsymbol\epsilon_\theta(\mathbf{x}_t,t,\varnothing).
\tag{20}
$$

- **One network does both jobs.** Train a single conditional network $\boldsymbol\epsilon_\theta(\mathbf{x}_t,t,y)$, and replace $y$ by a null token $\varnothing$ ("no label", or the empty text) for a random 10–20% of the training examples. The same weights then predict both the conditional and the unconditional noise.
- **Guidance means $\lambda>1$.** $\lambda=1$ ($w=0$) is plain conditional sampling and $\lambda=0$ is unconditional; guidance **extrapolates** past the conditional prediction, away from the unconditional one. Text-to-image systems typically use $\lambda$ between about 3 and 10.
- **Each sampling step costs two forward passes**, conditional and unconditional, combined by (20) before the denoising step.
- **The samples are no longer from $p(\mathbf{x}_0\mid y)$.** They are sharper, more typical of $y$, and less diverse: guidance trades diversity for fidelity. Figure 7 shows this on the running example.

> [!example]- Code: classifier-free guidance (training and sampling changes)
> ```python
> # assumes: model(x, t, y) -> eps_theta, with label NULL meaning "no condition";
> #          x0, y (B,), t, eps and xt as in the DDPM training loop; p_uncond = 0.1-0.2
> drop = torch.rand(len(y)) < p_uncond                    # training: drop some labels
> y_in = torch.where(drop, torch.full_like(y, NULL), y)
> loss = ((model(xt, t, y_in) - eps) ** 2).mean()         # L_simple, otherwise unchanged
>
> def cfg_eps(x, t, y, w):                                # sampling: two passes, then (20)
>     eps_c = model(x, t, y)
>     eps_u = model(x, t, torch.full_like(y, NULL))
>     return (1 + w) * eps_c - w * eps_u
> ```
> Against classifier guidance: no classifier and no input gradient, but two forward passes per step. `NULL` is one extra label value (an extra row of the label embedding).

![[Machine Learning/images/denoising-diffusion-models-cfg.svg]]
*Figure 7. Classifier-free guidance on the two moons, with one network trained with 20% label dropout, sampling the upper moon (100 ancestral steps, 3000 samples; grey: data). All samples land on the right moon at every $w$; what changes is where on it. The data are uniform along the arc, and with $w=0$ the samples are too: 25% fall in the quarter of the arc nearest the other moon (the right end). Guidance pushes samples away from what the other class looks like: that quarter holds 12% of the samples at $w=1$ and 3% at $w=4$, while the far quarter grows from 25% to 31% and 39%. The samples become more typical of their class and less diverse.*

## Latent diffusion models

Running a diffusion model on high-resolution images is expensive: every one of the many sampling steps processes every pixel. Most of those bits are imperceptible detail that a good autoencoder can reproduce cheaply. **Latent diffusion** builds the diffusion model on the latent space of another encoder–decoder model:[^rombach2022]
1. **Learn a latent representation.** Train an autoencoder with encoder $E_\phi$ and decoder $D_\phi$ (parameters $\phi$, trained jointly) on the data (or on similar data), so that $D_\phi(E_\phi(\mathbf{x}_0))\approx\mathbf{x}_0$. Its latent $\mathbf{z}_0=E_\phi(\mathbf{x}_0)$ is much smaller than the image, but still spatial: an $H_z\times W_z\times C_z$ grid, typically 4 to 8 times smaller per side. The latent space is kept well behaved by a small KL penalty towards $\mathcal{N}(\mathbf{0},\mathbf{I})$, as in a [[Variational Autoencoders|VAE]], or by vector quantization, as in a VQ-VAE.[^vandenoord2017]
2. **Train a DDPM on the latents.** Encode the whole dataset, $\mathbf{z}_0=E_\phi(\mathbf{x}_0)$, and train $\boldsymbol\epsilon_\theta(\mathbf{z}_t,t,y)$ on these exactly as before. Conditions such as text enter the U-Net through cross-attention layers, with the text embeddings as keys and values.
3. **Generate.** Sample a new latent $\mathbf{z}_0$ with reverse diffusion (and guidance), then decode it: $\mathbf{x}_{\text{new}}=D_\phi(\mathbf{z}_0)$.

The new image is a sample from (approximately) $p_{\text{data}}$, to the extent that the autoencoder is accurate and the latent diffusion model fits the encoded data. The decoder carries the perceptual detail and the diffusion model the content. This split is what makes text-to-image models such as Stable Diffusion affordable to train and run.

## Denoising diffusion implicit models

### Two problems with DDPM sampling

DDPMs have two weaknesses at sampling time:
- **Sampling is slow.** It takes $T$ network evaluations, typically 1000, one per step of the chain.
- **There is no unique latent for a data point.** The forward process is stochastic: running it twice from the same $\mathbf{x}_0$ gives two different $\mathbf{x}_T$. Conversely, the reverse process adds fresh noise at every step, so one $\mathbf{x}_T$ can lead to many $\mathbf{x}_0$ (Figure 8a). There is no single $\mathbf{x}_T$ that generates a given $\mathbf{x}_0$, so a DDPM cannot be used as a deterministic encoder.

Denoising diffusion implicit models (DDIMs) solve both, **without retraining**.[^song2021ddim] The key observation is that the training loss (16) depends on the forward process only through the marginals $q(\mathbf{x}_t\mid\mathbf{x}_0)$. Any other forward process with the same marginals would be trained with the same loss, by the same network. So look for forward processes with the same marginals and better sampling properties.

DDIM is usually written with Song et al.'s convention, in which $\alpha_t$ denotes what we call $\bar\alpha_t$. We keep $\bar\alpha_t$ throughout.

### A family of non-Markovian forward processes

Choose a vector of standard deviations $\boldsymbol\sigma=(\sigma_1,\dots,\sigma_T)\in\mathbb{R}^T_{\ge0}$ and define a family of inference distributions, going **backwards** from $\mathbf{x}_T$:

$$
q_{\boldsymbol\sigma}(\mathbf{x}_{1:T}\mid\mathbf{x}_0)=q_{\boldsymbol\sigma}(\mathbf{x}_T\mid\mathbf{x}_0)\prod_{t=2}^{T}q_{\boldsymbol\sigma}(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0),
$$

where $q_{\boldsymbol\sigma}(\mathbf{x}_T\mid\mathbf{x}_0)=\mathcal{N}\big(\sqrt{\bar\alpha_T}\,\mathbf{x}_0,(1-\bar\alpha_T)\mathbf{I}\big)$ and, for $t\ge2$,

$$
q_{\boldsymbol\sigma}(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)=\mathcal{N}\left(\sqrt{\bar\alpha_{t-1}}\,\mathbf{x}_0+\sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\cdot\frac{\mathbf{x}_t-\sqrt{\bar\alpha_t}\,\mathbf{x}_0}{\sqrt{1-\bar\alpha_t}},\ \sigma_t^2\mathbf{I}\right),
\tag{21}
$$

with $\sigma_t^2\le1-\bar\alpha_{t-1}$. Read the mean as: start from the scaled clean point $\sqrt{\bar\alpha_{t-1}}\,\mathbf{x}_0$, and add back part of the noise that $\mathbf{x}_t$ contains, $(\mathbf{x}_t-\sqrt{\bar\alpha_t}\,\mathbf{x}_0)/\sqrt{1-\bar\alpha_t}$, which is exactly $\boldsymbol\epsilon$ in (4). The rest of the variance, $\sigma_t^2$, is fresh noise.

The corresponding forward step follows from Bayes' rule,

$$
q_{\boldsymbol\sigma}(\mathbf{x}_t\mid\mathbf{x}_{t-1},\mathbf{x}_0)=\frac{q_{\boldsymbol\sigma}(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)\,q_{\boldsymbol\sigma}(\mathbf{x}_t\mid\mathbf{x}_0)}{q_{\boldsymbol\sigma}(\mathbf{x}_{t-1}\mid\mathbf{x}_0)},
$$

and it depends on $\mathbf{x}_0$: $\mathbf{x}_t$ is **not** conditionally independent of $\mathbf{x}_0$ given $\mathbf{x}_{t-1}$. The forward process is not Markovian. (The generative process will still be a Markov chain.)

### The same marginals

The mean in (21) was chosen to make one thing true.

> [!note] Lemma (Song et al., 2021)
> For every $\boldsymbol\sigma$ and every $t=1,\dots,T$, $q_{\boldsymbol\sigma}(\mathbf{x}_t\mid\mathbf{x}_0)=\mathcal{N}\big(\sqrt{\bar\alpha_t}\,\mathbf{x}_0,(1-\bar\alpha_t)\mathbf{I}\big)$, the same marginal as the DDPM's (3).

> [!proof]- Proof
> By backward induction on $t$. For $t=T$ it holds by definition. Suppose it holds for $t$, so $\mathbf{x}_t=\sqrt{\bar\alpha_t}\,\mathbf{x}_0+\sqrt{1-\bar\alpha_t}\,\boldsymbol\epsilon$ with $\boldsymbol\epsilon\sim\mathcal{N}(\mathbf{0},\mathbf{I})$ given $\mathbf{x}_0$. Sampling from (21) means
>
> $$
> \mathbf{x}_{t-1}=\sqrt{\bar\alpha_{t-1}}\,\mathbf{x}_0+\sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\;\boldsymbol\epsilon+\sigma_t\,\boldsymbol\xi,
> $$
>
> with $\boldsymbol\xi\sim\mathcal{N}(\mathbf{0},\mathbf{I})$ independent of $\boldsymbol\epsilon$, because the fraction in (21) equals $\boldsymbol\epsilon$. Given $\mathbf{x}_0$, the last two terms are independent zero-mean Gaussians, so their [[Joint Distributions#Sums of independent random variables|sum]] is Gaussian with covariance $(1-\bar\alpha_{t-1}-\sigma_t^2)\mathbf{I}+\sigma_t^2\mathbf{I}=(1-\bar\alpha_{t-1})\mathbf{I}$. Hence $q_{\boldsymbol\sigma}(\mathbf{x}_{t-1}\mid\mathbf{x}_0)=\mathcal{N}\big(\sqrt{\bar\alpha_{t-1}}\,\mathbf{x}_0,(1-\bar\alpha_{t-1})\mathbf{I}\big)$.

### The same training objective

Each $q_{\boldsymbol\sigma}$ has its own ELBO, and the ELBO does depend on $\boldsymbol\sigma$, through the posteriors (21). The precise statement is Song et al.'s Theorem 1. Use a decoder of the form below, with $\hat{\mathbf{x}}_\theta$ written through a noise predictor as in (14). Then for every $\boldsymbol\sigma$ with all $\sigma_t>0$ there are weights $\zeta_1,\dots,\zeta_T>0$ (depending on $\boldsymbol\sigma$) and a constant $C$ such that

$$
-F_{\boldsymbol\sigma}(\theta)=\sum_{t=1}^{T}\zeta_t\,\mathbb{E}\big\lVert\boldsymbol\epsilon-\boldsymbol\epsilon_\theta(\mathbf{x}_t,t)\big\rVert^2+C.
$$

Every member of the family is trained by a weighted noise-prediction loss, differing from (16) only in the weights. If the network had separate parameters for each $t$, the minimizer of each term would not depend on its weight, so the minimizer of $\mathcal{L}_{\text{simple}}$ would be optimal for every $\boldsymbol\sigma$ at once. With shared parameters this holds approximately. **Training the regression network of a DDPM implicitly trains all of these non-Markovian models.** Training is the same; only sampling differs.

### Sampling with DDIM

The decoder mirrors (21), with the unknown $\mathbf{x}_0$ replaced by the network's prediction $\hat{\mathbf{x}}_\theta(\mathbf{x}_t,t)=(\mathbf{x}_t-\sqrt{1-\bar\alpha_t}\,\boldsymbol\epsilon_\theta(\mathbf{x}_t,t))/\sqrt{\bar\alpha_t}$:

$$
p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t)=
\begin{cases}
\mathcal{N}\big(\hat{\mathbf{x}}_\theta(\mathbf{x}_1,1),\ \sigma_1^2\mathbf{I}\big), & t=1,\\
q_{\boldsymbol\sigma}\big(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\hat{\mathbf{x}}_\theta(\mathbf{x}_t,t)\big), & t\ge2.
\end{cases}
$$

Putting it together, one sampling step with $\boldsymbol\xi\sim\mathcal{N}(\mathbf{0},\mathbf{I})$ is

$$
\mathbf{x}_{t-1}=\underbrace{\sqrt{\bar\alpha_{t-1}}\left(\frac{\mathbf{x}_t-\sqrt{1-\bar\alpha_t}\,\boldsymbol\epsilon_\theta(\mathbf{x}_t,t)}{\sqrt{\bar\alpha_t}}\right)}_{\text{predicted }\mathbf{x}_0\text{, rescaled}}+\underbrace{\sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\;\boldsymbol\epsilon_\theta(\mathbf{x}_t,t)}_{\text{direction pointing to }\mathbf{x}_t}+\underbrace{\sigma_t\boldsymbol\xi}_{\text{fresh noise}}.
\tag{22}
$$

The noise is scaled by $\sigma_t$, the standard deviation. Song et al. parameterize the family by one number $\eta\ge0$:

$$
\sigma_t(\eta)=\eta\,\sqrt{\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}}\,\sqrt{1-\frac{\bar\alpha_t}{\bar\alpha_{t-1}}}.
\tag{23}
$$

- **$\eta=1$ is the DDPM.** Since $1-\bar\alpha_t/\bar\alpha_{t-1}=1-\alpha_t=\beta_t$, (23) gives $\sigma_t^2=\tilde\beta_t$, the posterior variance (10). Then (21) is the DDPM posterior (9), the forward process is Markovian again, and (22) is the ancestral sampling step.
- **$\eta=0$ is the DDIM.** With $\sigma_t=0$ the update is deterministic,

$$
\mathbf{x}_{t-1}=\sqrt{\bar\alpha_{t-1}}\left(\frac{\mathbf{x}_t-\sqrt{1-\bar\alpha_t}\,\boldsymbol\epsilon_\theta(\mathbf{x}_t,t)}{\sqrt{\bar\alpha_t}}\right)+\sqrt{1-\bar\alpha_{t-1}}\;\boldsymbol\epsilon_\theta(\mathbf{x}_t,t).
\tag{24}
$$

The only randomness is in $\mathbf{x}_T\sim\mathcal{N}(\mathbf{0},\mathbf{I})$, so the generative map $\mathbf{x}_T\mapsto\mathbf{x}_0$ is a deterministic function, an **implicit** model in the sense of GANs (hence the name). Given $(\mathbf{x}_t,\mathbf{x}_0)$, the forward step to $\mathbf{x}_{t-1}$ is deterministic too; $\mathbf{x}_T$ given $\mathbf{x}_0$ is still random. Figure 8 contrasts the two kinds of paths.

![[Machine Learning/images/denoising-diffusion-models-trajectories.svg]]
*Figure 8. Six reverse trajectories from the same six starting points $\mathbf{x}_T$ (crosses), 1000 steps each, drawn every 5th step. (a) DDPM paths are jagged, and rerunning from the same $\mathbf{x}_T$ lands elsewhere: over 2000 starting points, two runs end a mean distance of 2.08 apart, on data of unit scale. (b) DDIM paths are smooth and always end at the same point.*

### Fast sampling on a subsequence

Why is DDIM fast? The derivation never used the fact that the steps are $T,T-1,\dots,1$. Take any increasing subsequence $1\le\tau_1<\tau_2<\dots<\tau_S=T$ with $S\ll T$. The family (21) can be defined on these times alone, with $\bar\alpha_{\tau_i}$ in place of $\bar\alpha_t$ and $\bar\alpha_{\tau_{i-1}}$ in place of $\bar\alpha_{t-1}$; the lemma still gives the same marginals $q(\mathbf{x}_{\tau_i}\mid\mathbf{x}_0)$, and the trained network already knows those noise levels. So the **same network** can sample with $S=50$ or even $S=10$ steps, by applying (22) from $\tau_S$ to $\tau_{S-1}$ and so on, with $\eta$ chosen freely.

With few steps, quality drops for both samplers (Figure 9). On this easy 2-D example the difference between them is modest. With 50 steps both are as good as with 1000. With 10 steps the stochastic sampler leaves 5.6% of its samples off the data and the deterministic one 4.1%; with 5 steps, 25% and 23%. Energy distances to the data favour DDIM more clearly at 10 steps (0.029 against 0.017) and 5 steps (0.15 against 0.10). On images the gap is much larger: Song et al. found the deterministic sampler far better than the stochastic one with 10 to 100 steps.[^song2021ddim] Fresh noise injected in a few big steps is hard to remove again.

![[Machine Learning/images/denoising-diffusion-models-steps.svg]]
*Figure 9. Samples from one trained network with the update (22) on evenly spaced subsequences, 5000 samples each. Top: $\eta=1$ (DDPM); bottom: $\eta=0$ (DDIM). "Off-data" is the fraction of samples farther than 0.15 from every point of a large reference sample of the data (0% for fresh data). Energy distances to the data, ×1000: DDPM 2.7, 1.4, 29; DDIM 3.7, 3.8, 17 for 1000, 50 and 10 steps (0.5 between two data samples).*

> [!example]- Code: DDIM sampling and inversion
> Uses `T`, `abar` and a trained `model` from the DDPM code above. `eta=1` with `steps=1000` reproduces ancestral sampling.
> ```python
> def times(steps):                                  # tau_1 < ... < tau_S = T
>     return torch.linspace(1, T, steps).round().long().tolist()
>
> def abar_at(t):                                    # \bar alpha_t, with \bar alpha_0 = 1
>     return abar[t - 1] if t > 0 else torch.tensor(1.0)
>
> @torch.no_grad()
> def sample_ddim(model, x, steps=50, eta=0.0):
>     ts = times(steps)[::-1] + [0]                  # T = tau_S > ... > tau_1 > 0
>     for t, s in zip(ts[:-1], ts[1:]):              # one step from time t to time s < t
>         ab_t, ab_s = abar_at(t), abar_at(s)
>         eps = model(x, torch.full((len(x),), t))
>         x0_hat = (x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()
>         sig = eta * ((1 - ab_s) / (1 - ab_t) * (1 - ab_t / ab_s)).sqrt()
>         x = ab_s.sqrt() * x0_hat + (1 - ab_s - sig**2).sqrt() * eps + sig * torch.randn_like(x)
>     return x
>
> @torch.no_grad()
> def invert_ddim(model, x0, steps=50):              # run (24) forwards: data -> latent
>     ts = [0] + times(steps)
>     x = x0
>     for s, t in zip(ts[:-1], ts[1:]):              # one step from time s to time t > s
>         ab_s, ab_t = abar_at(s), abar_at(t)
>         eps = model(x, torch.full((len(x),), max(s, 1)))   # noise estimated at the current point
>         x0_hat = (x - (1 - ab_s).sqrt() * eps) / ab_s.sqrt()
>         x = ab_t.sqrt() * x0_hat + (1 - ab_t).sqrt() * eps
>     return x
> ```

### DDIM inversion

Rearrange (24) by dividing by $\sqrt{\bar\alpha_{t-1}}$, and write $\bar{\mathbf{x}}_t=\mathbf{x}_t/\sqrt{\bar\alpha_t}$ and $\bar\sigma_t=\sqrt{(1-\bar\alpha_t)/\bar\alpha_t}$ (the NCSN noise level from before):

$$
\bar{\mathbf{x}}_{t-1}=\bar{\mathbf{x}}_t+\big(\bar\sigma_{t-1}-\bar\sigma_t\big)\,\boldsymbol\epsilon_\theta(\mathbf{x}_t,t).
$$

This is one **Euler step** of the ordinary differential equation $d\bar{\mathbf{x}}/d\bar\sigma=\boldsymbol\epsilon_\theta$, from noise level $\bar\sigma_t$ down to $\bar\sigma_{t-1}$.[^song2021ddim] An ODE can be integrated in either direction, so a DDIM can be run **backwards**: starting from a data point $\mathbf{x}_0$, apply the same steps with increasing $\sigma$ to get a latent $\mathbf{x}_T$ that maps back to (nearly) $\mathbf{x}_0$. This is **DDIM inversion**, a deterministic encoder that DDPM sampling lacks. It enables image editing: invert a real image, then regenerate it with a modified condition, or interpolate between two images by interpolating their latents.

The inversion is only **approximate**. Each backward step must use $\boldsymbol\epsilon_\theta$ evaluated at the point it starts from, $\mathbf{x}_{t-1}$, instead of the unknown end point $\mathbf{x}_t$ that the forward step used. The two agree only in the limit of small steps. On the running example, inverting and regenerating 2000 data points returns them with a mean error of 0.003 with 1000 steps, 0.07 with 50 and 0.34 with 10 (the data have unit scale). With large guidance weights the errors are much larger, and editing methods add corrections.[^mokady2023]

## The SDE view

The [[Score-Based Models#Infinitely many noise levels: the SDE view|score-based lecture]] took the number of noise levels to infinity and got a stochastic differential equation. The same limit for the DDPM gives the **variance-preserving SDE**.[^song2021sde] Let $T\to\infty$ with $\beta_t=\beta(t/T)/T$ for a smooth function $\beta(\cdot)$ on $[0,1]$. Since $\sqrt{1-\beta_t}\approx1-\tfrac12\beta_t$, the forward step (1) becomes $\mathbf{x}_t-\mathbf{x}_{t-1}\approx-\tfrac12\beta_t\mathbf{x}_{t-1}+\sqrt{\beta_t}\,\boldsymbol\epsilon_t$, which in the limit is

$$
d\mathbf{x}=-\frac12\beta(t)\,\mathbf{x}\,dt+\sqrt{\beta(t)}\,d\mathbf{W},\qquad t\in[0,1].
$$

This is a mean-reverting (Ornstein–Uhlenbeck) diffusion, the continuous-time version of the chain in [[Markov Chains#Towards diffusion models and state space models|Markov Chains]]. With $\mathbf{f}(\mathbf{x},t)=-\tfrac12\beta(t)\mathbf{x}$ and $g(t)=\sqrt{\beta(t)}$ in the general SDE of the score-based lecture, the two results from there apply:
- **The reverse-time SDE**, $d\mathbf{x}=\big[-\tfrac12\beta(t)\mathbf{x}-\beta(t)\nabla_{\mathbf{x}}\log q_t(\mathbf{x})\big]dt+\sqrt{\beta(t)}\,d\bar{\mathbf{W}}$, run from $t=1$ to $0$. DDPM ancestral sampling is a discretization of it, with the score supplied by (19).
- **The probability flow ODE**, $\dfrac{d\mathbf{x}}{dt}=-\dfrac12\beta(t)\big[\mathbf{x}+\nabla_{\mathbf{x}}\log q_t(\mathbf{x})\big]$, has the same marginals $q_t$ and no randomness. DDIM with $\eta=0$ is a discretization of it: the Euler step in $(\bar\sigma,\bar{\mathbf{x}})$ coordinates of the previous section. The ODE also gives an exact likelihood, through the change-of-variables formula of continuous normalizing flows.

So all the samplers in this lecture are numerical solvers for one of two equations whose only unknown is the score. That viewpoint has two consequences. First, better ODE solvers give good samples in far fewer steps: higher-order solvers designed for this ODE need around 10 to 20 network evaluations.[^lu2022]<sup>,</sup>[^karras2022] Second, if the goal is a deterministic map from noise to data, one can learn the ODE's velocity field directly, without a diffusion process. That is [[Flow Matching|flow matching]], the subject of the next lecture. Its usual convention runs time the other way, with $t=0$ noise and $t=1$ data.[^lipman2023]

## Diffusion models in practice

**The network.** Image diffusion models use the U-Net of Ho et al., itself the [[Convolutional Neural Networks#U-Net|segmentation U-Net]] with modern parts.[^ho2020]<sup>,</sup>[^dhariwal2021]
- **Residual blocks** with group normalization at each resolution, with skip connections from each downsampling stage to the upsampling stage of the same resolution.
- **Self-attention** layers at the low resolutions (for example $16\times16$ and $8\times8$), so distant parts of the image can be made consistent.
- **Time conditioning**: the sinusoidal embedding of $t$ passes through a small MLP, and the result is added to (or scales and shifts) the features inside every residual block. A class label is embedded and added the same way; text enters through cross-attention.
- **Data scaling**: pixels are scaled to $[-1,1]$, and the predicted $\hat{\mathbf{x}}_\theta$ is often clipped to that range during sampling.

**Exponential moving average.** Sample from an exponential moving average of the weights (decay around 0.9999), not from the latest weights. The noise-prediction loss is noisy, and the average gives visibly more coherent samples (the code repository shows both).

**Evaluation.** Likelihoods (bits per dimension, from the ELBO or the ODE) measure density estimation, and they correlate poorly with what images look like. The standard sample-quality metric is the **Fréchet inception distance** (FID).[^heusel2017] Embed real and generated images with a pretrained Inception network, fit a Gaussian to each set of embeddings, and report the Fréchet (2-Wasserstein) distance between the Gaussians:

$$
\mathrm{FID}=\lVert\boldsymbol\mu_r-\boldsymbol\mu_g\rVert^2+\operatorname{tr}\Big(\boldsymbol\Sigma_r+\boldsymbol\Sigma_g-2\big(\boldsymbol\Sigma_r\boldsymbol\Sigma_g\big)^{1/2}\Big).
$$

Lower is better. FID penalizes both poor quality (shifted means) and poor diversity (mismatched covariances), but it depends on the number of samples (usually 50 000) and on the feature network, so only numbers computed the same way are comparable. The DDPM reached an FID of 3.17 on unconditional CIFAR-10, better than most GANs of the time.[^ho2020]

**Speed against quality.** The main practical cost of diffusion is sampling. The options, from slowest to fastest:

| Sampler | Network evaluations | Notes |
|---|---|---|
| DDPM ancestral | $T$, e.g. 1000 | the reference; stochastic |
| DDIM, $\eta=0$ | 50–100 | deterministic; invertible (approximately) |
| higher-order ODE solvers | 10–35 | DPM-Solver[^lu2022], Heun's method in EDM[^karras2022] |
| distilled students | 1–8 | progressive distillation[^salimans2022], consistency models[^song2023] |

Fewer steps generally cost some quality, and stochastic samplers (some fresh noise at each step) can correct earlier errors, so the best choice depends on the budget. Guidance doubles the cost per step.

**Diffusion among the generative models.**

| | VAE | GAN | DDPM |
|---|---|---|---|
| training | ELBO, stable | adversarial game, fragile | ELBO or $\mathcal{L}_{\text{simple}}$, a regression; stable |
| likelihood | lower bound | none | lower bound; exact via the ODE |
| sampling | one decoder pass | one generator pass | many network passes |
| sample quality | blurry | sharp, may drop modes | sharp and diverse |
| latent | low-dimensional, learned | low-dimensional | same size as the data; deterministic only with DDIM |

## Key takeaways

- A DDPM is a hierarchical VAE with $T$ latents of the data's size and a fixed Gaussian noising encoder, $\mathbf{x}_t=\sqrt{\alpha_t}\,\mathbf{x}_{t-1}+\sqrt{1-\alpha_t}\,\boldsymbol\epsilon_t$. Any step is reachable in one jump, $\mathbf{x}_t=\sqrt{\bar\alpha_t}\,\mathbf{x}_0+\sqrt{1-\bar\alpha_t}\,\boldsymbol\epsilon$, and $\mathbf{x}_T$ is approximately $\mathcal{N}(\mathbf{0},\mathbf{I})$.
- Bayes' rule and a telescoping sum turn the ELBO into reconstruction, prior matching (parameter-free) and denoising matching terms. The known denoising step $q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)$ is Gaussian, so each KL is a weighted regression.
- Predicting $\mathbf{x}_0$, the noise $\boldsymbol\epsilon$, or the score are one model in three parameterizations. Dropping the ELBO weights gives the simple loss $\lVert\boldsymbol\epsilon-\boldsymbol\epsilon_\theta\rVert^2$, which is denoising score matching; DDPM and NCSN differ only in coordinates, weighting and sampler.
- The noise schedule sets which noise levels get attention; the cosine schedule wastes fewer steps than the linear one. The decoder variance matters only with few steps.
- Guidance adds a classifier's gradient, or extrapolates from the unconditional to the conditional prediction ($\lambda=1+w>1$), at sampling time. It trades diversity for fidelity.
- Latent diffusion runs the same model in an autoencoder's latent space, which makes high-resolution generation affordable.
- DDIM keeps the DDPM's marginals with a non-Markovian forward process, so the same trained network samples deterministically ($\eta=0$) in a few dozen steps and can be inverted. It is an Euler solver for the probability flow ODE.

## Where this shows up next

- [[Score-Based Models]]: the same model from the score side; Tweedie's formula and denoising score matching are what make noise prediction learn the score.
- [[Variational Autoencoders]]: the ELBO, the Gaussian KL and the reparameterization trick used throughout; latent diffusion uses a VAE as its first stage.
- [[Energy-Based Models]]: guidance multiplies densities, $q_t(\mathbf{x})\,p(y\mid\mathbf{x})^\lambda$, which is adding energies, as in a product of experts.
- [[Markov Chains]]: the forward process is a Gaussian Markov chain with stationary distribution $\mathcal{N}(\mathbf{0},\mathbf{I})$; its continuous-time limit is the Ornstein–Uhlenbeck diffusion.
- [[Convolutional Neural Networks]] and [[Sequence Models]]: the U-Net, sinusoidal encodings and cross-attention that diffusion networks are built from.
- [[Flow Matching]]: the next lecture, which learns the velocity of a deterministic noise-to-data flow directly; diffusion is the special case with a variance-preserving path. Then [[Normalizing Flows]], invertible networks with exact likelihoods.

<!-- References: footnote definitions below render as the numbered "References" list at the bottom of the page -->

[^sohl2015]: Sohl-Dickstein, J., Weiss, E. A., Maheswaranathan, N., & Ganguli, S. (2015). *Deep Unsupervised Learning using Nonequilibrium Thermodynamics*. ICML. [arXiv:1503.03585](https://arxiv.org/abs/1503.03585)
[^ho2020]: Ho, J., Jain, A., & Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models*. NeurIPS. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)
[^luo2022]: Luo, C. (2022). *Understanding Diffusion Models: A Unified Perspective*. [arXiv:2208.11970](https://arxiv.org/abs/2208.11970)
[^kingma2021]: Kingma, D. P., Salimans, T., Poole, B., & Ho, J. (2021). *Variational Diffusion Models*. NeurIPS. [arXiv:2107.00630](https://arxiv.org/abs/2107.00630)
[^ronneberger2015]: Ronneberger, O., Fischer, P., & Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation*. MICCAI. [arXiv:1505.04597](https://arxiv.org/abs/1505.04597)
[^vincent2011]: Vincent, P. (2011). *A Connection Between Score Matching and Denoising Autoencoders*. Neural Computation, 23(7), 1661–1674. [doi:10.1162/NECO_a_00142](https://doi.org/10.1162/NECO_a_00142)
[^salimans2022]: Salimans, T., & Ho, J. (2022). *Progressive Distillation for Fast Sampling of Diffusion Models*. ICLR. [arXiv:2202.00512](https://arxiv.org/abs/2202.00512)
[^nichol2021]: Nichol, A., & Dhariwal, P. (2021). *Improved Denoising Diffusion Probabilistic Models*. ICML. [arXiv:2102.09672](https://arxiv.org/abs/2102.09672)
[^dhariwal2021]: Dhariwal, P., & Nichol, A. (2021). *Diffusion Models Beat GANs on Image Synthesis*. NeurIPS. [arXiv:2105.05233](https://arxiv.org/abs/2105.05233)
[^ho2022]: Ho, J., & Salimans, T. (2022). *Classifier-Free Diffusion Guidance*. NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications. [arXiv:2207.12598](https://arxiv.org/abs/2207.12598)
[^rombach2022]: Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022). *High-Resolution Image Synthesis with Latent Diffusion Models*. CVPR. [arXiv:2112.10752](https://arxiv.org/abs/2112.10752)
[^vandenoord2017]: van den Oord, A., Vinyals, O., & Kavukcuoglu, K. (2017). *Neural Discrete Representation Learning*. NeurIPS. [arXiv:1711.00937](https://arxiv.org/abs/1711.00937)
[^song2021ddim]: Song, J., Meng, C., & Ermon, S. (2021). *Denoising Diffusion Implicit Models*. ICLR. [arXiv:2010.02502](https://arxiv.org/abs/2010.02502)
[^mokady2023]: Mokady, R., Hertz, A., Aberman, K., Pritch, Y., & Cohen-Or, D. (2023). *Null-text Inversion for Editing Real Images using Guided Diffusion Models*. CVPR. [arXiv:2211.09794](https://arxiv.org/abs/2211.09794)
[^song2021sde]: Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2021). *Score-Based Generative Modeling through Stochastic Differential Equations*. ICLR. [arXiv:2011.13456](https://arxiv.org/abs/2011.13456)
[^lu2022]: Lu, C., Zhou, Y., Bao, F., Chen, J., Li, C., & Zhu, J. (2022). *DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps*. NeurIPS. [arXiv:2206.00927](https://arxiv.org/abs/2206.00927)
[^karras2022]: Karras, T., Aittala, M., Aila, T., & Laine, S. (2022). *Elucidating the Design Space of Diffusion-Based Generative Models*. NeurIPS. [arXiv:2206.00364](https://arxiv.org/abs/2206.00364)
[^lipman2023]: Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., & Le, M. (2023). *Flow Matching for Generative Modeling*. ICLR. [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)
[^heusel2017]: Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., & Hochreiter, S. (2017). *GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium*. NeurIPS. [arXiv:1706.08500](https://arxiv.org/abs/1706.08500)
[^song2023]: Song, Y., Dhariwal, P., Chen, M., & Sutskever, I. (2023). *Consistency Models*. ICML. [arXiv:2303.01469](https://arxiv.org/abs/2303.01469)
