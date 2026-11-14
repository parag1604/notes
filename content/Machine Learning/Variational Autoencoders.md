---
title: Variational Autoencoders
tags:
  - generative-models
  - variational-inference
  - autoencoders
  - representation-learning
date: 2026-11-12
draft: false
prev: "[[Language Representations and Language Models]]"
next: "[[Generative Adversarial Networks]]"
---

> **Prerequisites:** [[Matrix Factorization and Dimensionality Reduction#Autoencoders|Autoencoders]] (encoder, bottleneck, decoder) and [[Clustering and Expectation Maximization#The evidence lower bound (ELBO)|the evidence lower bound]] from the EM lecture. [[Policy Gradients]] is helpful for the gradient-estimator aside.
>
> **TL;DR:**
> - A VAE is a latent-variable model $p_\theta(\mathbf{x}\mid\mathbf{z})p(\mathbf{z})$ trained by maximizing the **ELBO**. A second network, the **encoder** $q_\phi(\mathbf{z}\mid\mathbf{x})$, approximates the intractable posterior.
> - The ELBO splits into a **reconstruction** term and a **KL-to-prior** term. With Gaussian encoder and prior, the KL term has a closed form.
> - The **reparameterization trick** $\mathbf{z}=\boldsymbol\mu_\phi(\mathbf{x})+\boldsymbol\sigma_\phi(\mathbf{x})\odot\boldsymbol\epsilon$ moves the randomness into an input, so ordinary backpropagation gives low-variance gradients for $\phi$.
> - To generate, sample $\mathbf{z}\sim\mathcal{N}(\mathbf{0},\mathbf{I})$ and decode.
> - Variants: conditional VAEs choose what to generate, β-VAEs reweight the KL term, hierarchical VAEs stack latents, and the **VQ-VAE** family uses a discrete codebook trained with a straight-through gradient copy.
>
> **Code:** [Image-Based-Generative-Models](https://github.com/parag1604/Image-Based-Generative-Models): a convolutional VAE, a DCGAN and a DDPM (with DDIM sampling) trained on 64×64 Stanford Cars images.

## Motivation

An [[Matrix Factorization and Dimensionality Reduction#Autoencoders|autoencoder]] compresses $\mathbf{x}$ to a code $\mathbf{z}$ and reconstructs it. It learns a useful representation, but it is not a generative model: nothing tells us which codes are "valid", so decoding a random $\mathbf{z}$ usually produces garbage.

In [[Clustering and Expectation Maximization|EM]] we met the opposite situation: a proper probabilistic model with latent variables, $p_\theta(\mathbf{x},\mathbf{z})$, trained through the ELBO. EM works when the posterior $p_\theta(\mathbf{z}\mid\mathbf{x})$ can be computed by Bayes' rule, as in a Gaussian mixture. With a neural-network decoder it cannot.

The VAE joins the two ideas:[^kingma2014]<sup>,</sup>[^rezende2014]
- keep the probabilistic model and the ELBO from EM;
- replace the exact E-step by a **learned** encoder $q_\phi(\mathbf{z}\mid\mathbf{x})$, which looks exactly like the encoder of an autoencoder.

## From EM to variational inference

Let's resume our discussion from the ELBO. For any density $q(\mathbf{z})$,

$$
F_\theta(q)=\int q(\mathbf{z})\log\frac{p_\theta(\mathbf{x},\mathbf{z})}{q(\mathbf{z})}\,d\mathbf{z}
=\mathbb{E}_{q(\mathbf{z})}\big[\log p_\theta(\mathbf{x},\mathbf{z})\big]+H(q).
$$

$F_\theta(q)$ is a **functional**: a function of the function $q$. It is a lower bound on $\log p_\theta(\mathbf{x})$, and the gap is exactly a KL divergence:

$$
\log p_\theta(\mathbf{x})=F_\theta(q)+D_{\mathrm{KL}}\big(q(\mathbf{z})\,\|\,p_\theta(\mathbf{z}\mid\mathbf{x})\big).
\tag{1}
$$

- **EM** closes the gap exactly: its E-step sets $q=p_\theta(\mathbf{z}\mid\mathbf{x})$, computed analytically.
- For a general $p_\theta$ the posterior is intractable. Instead we **add parameters $\phi$ to $q$** and maximize $F$ over $\phi$ as well as $\theta$. This is **variational inference**: the best $q_\phi$ in the family is the one with the smallest KL gap in (1).

> [!note] Amortized inference
> In EM every data point gets its own $q_i(\mathbf{z})$. A VAE instead uses one network that maps any $\mathbf{x}$ to its approximate posterior, $q_\phi(\mathbf{z}\mid\mathbf{x})$, "since the latent embeddings are sampled based on $\mathbf{x}$". The cost of inference is paid once, in training, and is then *amortized* over all data points: a new $\mathbf{x}$ needs one forward pass, not an optimization.

## The VAE formulation

We are given $\mathcal{D}=\{\mathbf{x}_i\}_{i=1}^N$, drawn i.i.d. from an unknown $p_X$. We want two things:
1. a maximum-likelihood estimate of a model $p_\theta(\mathbf{x})$ that we can **sample** from;
2. **posterior inference**: an encoding $q_\phi(\mathbf{z}\mid\mathbf{x})\approx p_\theta(\mathbf{z}\mid\mathbf{x})$.

The model has the autoencoder's shape:
- the **prior** $p(\mathbf{z})=\mathcal{N}(\mathbf{0},\mathbf{I})$ is fixed, with no parameters;
- the **encoder** gives $\mathbf{z}_i\sim q_\phi(\cdot\mid\mathbf{x}_i)$;
- the **decoder** gives $\hat{\mathbf{x}}_i\sim p_\theta(\cdot\mid\mathbf{z}_i)$.

![[Machine Learning/images/vae-architecture.svg]]
*Figure 1: a VAE. The encoder outputs the mean and standard deviation of $q_\phi(\mathbf{z}\mid\mathbf{x})$. The latent code is formed from those and an independent noise input. The two ELBO terms attach to the latent code (KL) and to the output (reconstruction). To generate new data, only the decoder is used.*

### The ELBO for one data point

Apply Jensen's inequality to the log of the marginal likelihood:

$$
\begin{aligned}
\ell(\theta)=\log p_\theta(\mathbf{x})
&=\log\int q(\mathbf{z}\mid\mathbf{x})\,\frac{p_\theta(\mathbf{x},\mathbf{z})}{q(\mathbf{z}\mid\mathbf{x})}\,d\mathbf{z}\\
&\ge\mathbb{E}_{q(\mathbf{z}\mid\mathbf{x})}\left[\log\frac{p_\theta(\mathbf{x}\mid\mathbf{z})\,p(\mathbf{z})}{q(\mathbf{z}\mid\mathbf{x})}\right]\\
&=\mathbb{E}_{q(\mathbf{z}\mid\mathbf{x})}\big[\log p_\theta(\mathbf{x}\mid\mathbf{z})\big]-\mathbb{E}_{q(\mathbf{z}\mid\mathbf{x})}\left[\log\frac{q(\mathbf{z}\mid\mathbf{x})}{p(\mathbf{z})}\right].
\end{aligned}
$$

The second expectation is a KL divergence. With $q$ parameterized by $\phi$, we get the VAE objective:

$$
F(\theta,\phi;\mathbf{x})=\underbrace{\mathbb{E}_{q_\phi(\mathbf{z}\mid\mathbf{x})}\big[\log p_\theta(\mathbf{x}\mid\mathbf{z})\big]}_{\text{① reconstruction}}-\underbrace{D_{\mathrm{KL}}\big(q_\phi(\mathbf{z}\mid\mathbf{x})\,\|\,p(\mathbf{z})\big)}_{\text{② KL to the prior}}\le\log p_\theta(\mathbf{x}).
\tag{2}
$$

- $\log p_\theta(\mathbf{x}\mid\mathbf{z})$ is the **conditional data likelihood**: how well the decoder explains $\mathbf{x}$ from $\mathbf{z}$.
- $q_\phi(\mathbf{z}\mid\mathbf{x})$ is the **conditional latent distribution** (the encoder).
- $p(\mathbf{z})$ is the **prior in the latent space**.

Term ① rewards codes that carry information about $\mathbf{x}$. Term ② keeps every encoding close to the prior. The training objective is $\sum_{i=1}^N F(\theta,\phi;\mathbf{x}_i)$, maximized jointly over $\theta$ and $\phi$, with both distributions parameterized by neural networks.

## Gradients of an expectation

We need $\nabla_\theta F$ and $\nabla_\phi F$. Term ② will turn out to be analytic. Term ① is an expectation, and the two parameter sets behave very differently in it.

**Case 1: the parameters are only in the integrand.** If the sampling distribution $p_V$ does not depend on $\theta$, then

$$
\nabla_\theta\,\mathbb{E}_{v\sim p_V}\big[f_\theta(v)\big]=\int p_V(v)\,\nabla_\theta f_\theta(v)\,dv=\mathbb{E}_{v\sim p_V}\big[\nabla_\theta f_\theta(v)\big]\approx\frac{1}{L}\sum_{l=1}^{L}\nabla_\theta f_\theta(v_l),
$$

with $v_l\sim p_V$. Gradients can be computed as long as we can sample. This is the situation for **$\theta$** in term ①: the decoder appears only inside the log.

**Case 2: the parameters are in the sampling distribution.** For **$\phi$** in term ①, the expectation is taken under $q_\phi$ itself. In general, with $v\sim p_\phi(v)$:

$$
\nabla_\phi\,\mathbb{E}_{v\sim p_\phi}\big[f(v)\big]=\int f(v)\,\nabla_\phi p_\phi(v)\,dv.
$$

This is not an expectation under $p_\phi$, because $\nabla_\phi p_\phi$ is not a probability density. We cannot estimate it just by sampling from $p_\phi$ and averaging. There are two ways out.

### Way 1: the score-function estimator

Write $\nabla_\phi p_\phi=p_\phi\,\nabla_\phi\log p_\phi$ (the log-derivative trick):

$$
\nabla_\phi\,\mathbb{E}_{v\sim p_\phi}\big[f(v)\big]=\mathbb{E}_{v\sim p_\phi}\big[f(v)\,\nabla_\phi\log p_\phi(v)\big].
$$

Now it *is* an expectation, and it can be estimated by sampling. This is exactly the REINFORCE estimator from [[Policy Gradients#The objective and its gradient|policy gradients]]. It only needs $\log p_\phi$ to be differentiable, so it works even for discrete $v$ and non-differentiable $f$. Its weakness is the same as in RL: very **high variance**, because $f$ is used only as a scalar weight and its slope is ignored.

### Way 2: the reparameterization (pathwise) estimator

The second route relies on a familiar fact.

> [!note] ASIDE: LOTUS, the law of the unconscious statistician
> Let $X$ be a random variable and $Y=g(X)$ for a deterministic $g$. We may not know $p_Y$, but we don't need it:
>
> $$
> \mathbb{E}[Y]=\mathbb{E}[g(X)]=\sum_x p_X(x)\,g(x)\qquad\left(\text{or }\int p_X(x)\,g(x)\,dx\right).
> $$
>
> LOTUS expresses the expectation of $Y$ through the distribution of $X$ and the function linking them. For example, to get $\mathbb{E}[X^2]$ we never derive the distribution of $X^2$; we just average $x^2$ under $p_X$. The name is a joke: many statisticians use this formula "unconsciously", as if it were the definition of $\mathbb{E}[g(X)]$, when it is actually a theorem.

**The reparameterization trick.** Suppose we can write a sample from $p_\phi$ as a deterministic, differentiable function of $\phi$ and a noise variable whose distribution has **no parameters**:

$$
v=g_\phi(\epsilon),\qquad\epsilon\sim p(\epsilon).
$$

By LOTUS, $\mathbb{E}_{v\sim p_\phi}[f(v)]=\mathbb{E}_{\epsilon\sim p(\epsilon)}[f(g_\phi(\epsilon))]$. Now $\phi$ is only in the integrand, and we are back in Case 1:

$$
\nabla_\phi\,\mathbb{E}_{v\sim p_\phi}\big[f(v)\big]=\mathbb{E}_{\epsilon\sim p(\epsilon)}\big[\nabla_\phi f(g_\phi(\epsilon))\big]\approx\frac{1}{L}\sum_{l=1}^L\nabla_\phi f\big(g_\phi(\epsilon_l)\big).
$$

For a Gaussian, $v\sim\mathcal{N}(\mu_\phi,\sigma_\phi^2)$ is the same as $v=\mu_\phi+\sigma_\phi\,\epsilon$ with $\epsilon\sim\mathcal{N}(0,1)$. So

$$
\nabla_\phi\,\mathbb{E}_{v\sim\mathcal{N}(\mu_\phi,\sigma_\phi^2)}\big[f(v)\big]=\mathbb{E}_{\epsilon\sim\mathcal{N}(0,1)}\big[\nabla_\phi f(\mu_\phi+\sigma_\phi\epsilon)\big].
$$

The gradient now flows through $f$ along the sampling *path*, using the slope of $f$. That is where the name **pathwise** estimator comes from.

![[Machine Learning/images/vae-reparameterization-graph.svg]]
*Figure 2: left, sampling sits inside the computation graph, and there is no derivative of a random draw with respect to $\phi$. Right, after reparameterizing, the noise is an input. $\mathbf{z}$ is a deterministic function of $\phi$ and $\boldsymbol\epsilon$, so the chain rule carries $\nabla_\phi$ back from the loss.*

How much does the choice matter? Take $f(z)=z^2$ and $z\sim\mathcal{N}(\mu,1)$, so the true gradient is $\frac{\partial}{\partial\mu}\mathbb{E}[z^2]=\frac{\partial}{\partial\mu}(\mu^2+1)=2\mu$. Both estimators are unbiased, but one sample of the score-function estimator is $z^2(z-\mu)$, while one sample of the pathwise estimator is $2(\mu+\epsilon)$.

![[Machine Learning/images/vae-gradient-estimators.svg]]
*Figure 3: 20,000 single-sample estimates of the same gradient (true value 3). Both are centred on the truth, but the score-function estimator's standard deviation is about 7.0 against 2.0 for the pathwise one, so it needs about 12 times as many samples for the same accuracy.*

> [!example]- Code: score-function vs pathwise gradient (PyTorch)
> ```python
> # assumes: mu = torch.tensor(1.5, requires_grad=True);
> #   f = lambda z: z ** 2; L samples
> eps = torch.randn(L)
> # score function (REINFORCE): f(z) * d/dmu log N(z; mu, 1), with
> #   the sample z held fixed
> z = (mu + eps).detach()
> # log N(z; mu, 1) + const
> log_q = -0.5 * (z - mu) ** 2
> g_score, = torch.autograd.grad((f(z) * log_q).mean(), mu)
> # pathwise (reparameterization): differentiate f along
> #   z = mu + eps
> g_path, = torch.autograd.grad(f(mu + eps).mean(), mu)
> ```
> Both estimate $2\mu=3$. The only difference is where `detach()` sits: the score-function estimator cuts the path from $\mu$ to $z$ and differentiates the log-density instead; the pathwise estimator keeps the path and uses the slope of $f$.

> [!tip] Which estimator when?
> - **Pathwise** needs a differentiable $f$ and a reparameterizable distribution (Gaussian, logistic, anything with an invertible CDF). It usually has much lower variance. It is the VAE's choice.
> - **Score function** works for anything you can sample and whose log-density you can differentiate, including discrete latents and non-differentiable rewards. It is the RL choice, made usable by [[Policy Gradients#Baselines|baselines]].
> - For discrete variables there is a middle way: a continuous relaxation such as [[Policy Gradients#Straight-Through Gumbel-Softmax (STGS) for PG Algorithms|Gumbel-softmax]], which trades a little bias for pathwise gradients. See Mohamed et al. for a survey.[^mohamed2020]

## The Gaussian VAE

### Encoder and reparameterization

Let the encoder be a diagonal Gaussian whose mean and standard deviation are produced by a neural network with parameters $\phi$:

$$
q_\phi(\mathbf{z}\mid\mathbf{x}_i)=\mathcal{N}\big(\mathbf{z};\,\boldsymbol\mu_\phi(\mathbf{x}_i),\,\operatorname{diag}\boldsymbol\sigma^2_\phi(\mathbf{x}_i)\big).
$$

A sample is drawn as

$$
\mathbf{z}_{i,l}=\boldsymbol\mu_\phi(\mathbf{x}_i)+\boldsymbol\sigma_\phi(\mathbf{x}_i)\odot\boldsymbol\epsilon_{i,l},\qquad\boldsymbol\epsilon_{i,l}\sim\mathcal{N}(\mathbf{0},\mathbf{I}).
$$

Note that this uses the **standard deviation** $\boldsymbol\sigma$, not the variance: in general $\mathbf{z}=\boldsymbol\mu+\mathbf{L}\boldsymbol\epsilon$, where $\mathbf{L}\mathbf{L}^\top=\boldsymbol\Sigma$ (Cholesky factor). For a diagonal $\boldsymbol\Sigma$, $\mathbf{L}=\operatorname{diag}\boldsymbol\sigma$.

> [!example]- Code: the reparameterized sample (PyTorch)
> ```python
> # assumes: enc(x) -> (mu, logvar), each of shape (B, d)
> mu, logvar = enc(x)
> # sigma = exp(log(sigma^2) / 2): the standard deviation
> std = torch.exp(0.5 * logvar)
> # eps ~ N(0, I): noise with no parameters
> eps = torch.randn_like(std)
> # z = mu + sigma * eps, differentiable in mu and sigma
> z = mu + std * eps
> ```
> This is the `reparameterize` method of the VAE in the code repository. The encoder outputs $\log\boldsymbol\sigma^2$ (see [[#VAEs in practice]]).

### The Monte Carlo ELBO

With $L$ noise samples per data point, the estimate of (2) is

$$
\hat F(\theta,\phi;\mathbf{x}_i)=\frac{1}{L}\sum_{l=1}^{L}\log p_\theta(\mathbf{x}_i\mid\mathbf{z}_{i,l})-D_{\mathrm{KL}}\big(q_\phi(\mathbf{z}\mid\mathbf{x}_i)\,\|\,p(\mathbf{z})\big),
$$

and the training objective is $\sum_i\hat F(\theta,\phi;\mathbf{x}_i)$, optimized with minibatches. In practice $L=1$ is enough. Because the encoder and the prior are both Gaussian, the KL term needs no sampling at all; it is computed analytically.

> [!note]- ASIDE: the KL divergence between two Gaussians
> By definition, $D_{\mathrm{KL}}(p\,\|\,q)=\int p(x)\log\frac{p(x)}{q(x)}\,dx=\mathbb{E}_{x\sim p}\left[\log\frac{p(x)}{q(x)}\right]$.
>
> Let $p=\mathcal{N}(\mu_1,\sigma_1^2)$ and $q=\mathcal{N}(\mu_2,\sigma_2^2)$, with density $\mathcal{N}(x;\mu,\sigma^2)=\frac{1}{\sqrt{2\pi\sigma^2}}e^{-(x-\mu)^2/2\sigma^2}$. Then
>
> $$
> \log\frac{p(x)}{q(x)}=\frac12\left[-\frac{(x-\mu_1)^2}{\sigma_1^2}+\frac{(x-\mu_2)^2}{\sigma_2^2}+\log\frac{\sigma_2^2}{\sigma_1^2}\right].
> $$
>
> Take the expectation under $x\sim p$. The first term gives $\mathbb{E}[(x-\mu_1)^2]/\sigma_1^2=1$, by the definition of the variance. For the second, use $\mathbb{E}[x^2]=\mu_1^2+\sigma_1^2$:
>
> $$
> \mathbb{E}_p\big[(x-\mu_2)^2\big]=\mu_1^2+\sigma_1^2-2\mu_1\mu_2+\mu_2^2=(\mu_1-\mu_2)^2+\sigma_1^2.
> $$
>
> Hence
>
> $$
> D_{\mathrm{KL}}(p\,\|\,q)=\frac12\left[\frac{(\mu_1-\mu_2)^2}{\sigma_2^2}+\frac{\sigma_1^2}{\sigma_2^2}-\log\frac{\sigma_1^2}{\sigma_2^2}-1\right].
> $$
>
> **Against the standard-normal prior** ($\mu_2=0$, $\sigma_2=1$):
>
> $$
> D_{\mathrm{KL}}\big(\mathcal{N}(\mu,\sigma^2)\,\|\,\mathcal{N}(0,1)\big)=-\frac12\left[1+\log\sigma^2-\mu^2-\sigma^2\right].
> $$
>
> **Multivariate** ($d$ dimensions):
>
> $$
> \begin{aligned}
> &D_{\mathrm{KL}}\big(\mathcal{N}(\boldsymbol\mu_1,\boldsymbol\Sigma_1)\,\|\,\mathcal{N}(\boldsymbol\mu_2,\boldsymbol\Sigma_2)\big)\\
> &\quad=\frac12\left[(\boldsymbol\mu_2-\boldsymbol\mu_1)^\top\boldsymbol\Sigma_2^{-1}(\boldsymbol\mu_2-\boldsymbol\mu_1)+\operatorname{tr}(\boldsymbol\Sigma_2^{-1}\boldsymbol\Sigma_1)-\log\frac{|\boldsymbol\Sigma_1|}{|\boldsymbol\Sigma_2|}-d\right].
> \end{aligned}
> $$
>
> With $q=\mathcal{N}(\mathbf{0},\mathbf{I})$ (using $\operatorname{tr}\mathbf{I}_d=d$):
>
> $$
> D_{\mathrm{KL}}\big(\mathcal{N}(\boldsymbol\mu,\boldsymbol\Sigma)\,\|\,\mathcal{N}(\mathbf{0},\mathbf{I})\big)=-\frac12\left[d+\log|\boldsymbol\Sigma|-\boldsymbol\mu^\top\boldsymbol\mu-\operatorname{tr}\boldsymbol\Sigma\right].
> $$

For the diagonal encoder the KL term is therefore a simple sum over the $d$ latent dimensions:

$$
D_{\mathrm{KL}}\big(q_\phi(\mathbf{z}\mid\mathbf{x})\,\|\,\mathcal{N}(\mathbf{0},\mathbf{I})\big)=-\frac12\sum_{j=1}^{d}\Big(1+\log\sigma_j^2-\mu_j^2-\sigma_j^2\Big).
\tag{3}
$$

![[Machine Learning/images/vae-kl-landscape.svg]]
*Figure 4: the KL term (3) for one latent dimension. It is zero only at the prior ($\mu=0$, $\sigma=1$) and grows quadratically in $\mu$. It also blows up as $\sigma\to0$, so the encoder is penalized both for moving codes away from the origin and for making them too precise.*

### The reconstruction term

Term ① depends on the decoder's likelihood. With a Gaussian decoder of fixed variance, $p_\theta(\mathbf{x}\mid\mathbf{z})=\mathcal{N}\big(\mathbf{x};\hat{\mathbf{x}}_\theta(\mathbf{z}),\sigma^2\mathbf{I}\big)$, so

$$
\log p_\theta(\mathbf{x}_i\mid\mathbf{z}_i)=-\frac{1}{2\sigma^2}\big\lVert\mathbf{x}_i-\hat{\mathbf{x}}_\theta(\mathbf{z}_i)\big\rVert_2^2+\text{const}.
$$

*Maximizing* the log-likelihood is *minimizing* the squared reconstruction error, just as in [[Linear Models|least squares]]. The decoder variance $\sigma^2$ sets the exchange rate between the two terms of the ELBO: a small $\sigma^2$ makes reconstruction dominate.

Put together, the loss we minimize for one data point (with $L=1$) is

$$
\mathcal{L}(\theta,\phi;\mathbf{x}_i)=\frac{1}{2\sigma^2}\big\lVert\mathbf{x}_i-\hat{\mathbf{x}}_\theta(\mathbf{z}_i)\big\rVert^2-\frac12\sum_{j=1}^d\Big(1+\log\sigma_{j}^2-\mu_{j}^2-\sigma_{j}^2\Big),
$$

where $\mathbf{z}_i=\boldsymbol\mu_\phi(\mathbf{x}_i)+\boldsymbol\sigma_\phi(\mathbf{x}_i)\odot\boldsymbol\epsilon_i$. Backpropagation carries the reconstruction loss back to $\mathbf{z}_i$. Because of the reparameterization, it continues from $\mathbf{z}_i$ through $\boldsymbol\mu_\phi$ and $\boldsymbol\sigma_\phi$ into the encoder, all the way to the input.

> [!example] Algorithm: training a VAE
> 1. Sample a minibatch $\{\mathbf{x}_i\}_{i=1}^B$ from $\mathcal{D}$.
> 2. Encode: $\boldsymbol\mu_i,\log\boldsymbol\sigma_i^2=\mathrm{Enc}_\phi(\mathbf{x}_i)$.
> 3. Sample $\boldsymbol\epsilon_i\sim\mathcal{N}(\mathbf{0},\mathbf{I})$ and set $\mathbf{z}_i=\boldsymbol\mu_i+\boldsymbol\sigma_i\odot\boldsymbol\epsilon_i$.
> 4. Decode: $\hat{\mathbf{x}}_i=\mathrm{Dec}_\theta(\mathbf{z}_i)$.
> 5. Compute $\frac1B\sum_i\mathcal{L}(\theta,\phi;\mathbf{x}_i)$, the reconstruction loss plus the analytic KL (3).
> 6. Backpropagate and take a gradient step on $(\theta,\phi)$. Repeat until the validation ELBO stops improving.

> [!example]- Code: the negative ELBO (PyTorch)
> ```python
> # assumes: x (B, D); x_hat = dec(z) (B, D); mu, logvar (B, d) from
> #   the encoder; sigma2 = decoder variance
> # -log p(x|z) up to a constant
> rec = ((x - x_hat) ** 2).sum(dim=1) / (2 * sigma2)
> # eq. (3), analytic
> kl = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(dim=1)
> # beta = 1: minus the ELBO (2)
> loss = (rec + beta * kl).mean()
> loss.backward()
> ```
> The repository's VAE uses an L1 reconstruction, `torch.abs(recon - data).sum()`, which corresponds to a Laplace decoder, and ramps the KL weight up from 0 during training: the KL annealing described under [[#VAEs in practice|posterior collapse]].

## How do we sample?

This is where the assumption $p(\mathbf{z})=\mathcal{N}(\mathbf{0},\mathbf{I})$ comes into play. To generate a new data point:
1. sample $\mathbf{z}\sim\mathcal{N}(\mathbf{0},\mathbf{I})$;
2. decode it, $\hat{\mathbf{x}}\sim p_\theta(\mathbf{x}\mid\mathbf{z})$ (or just take the mean $\hat{\mathbf{x}}_\theta(\mathbf{z})$).

The encoder is not used at all. Why should a random draw from the prior decode to something sensible? Because of the KL term. It does **not** force each individual $q_\phi(\mathbf{z}\mid\mathbf{x})$ to equal $\mathcal{N}(\mathbf{0},\mathbf{I})$; then $\mathbf{z}$ would carry no information about $\mathbf{x}$, and reconstruction would fail. Instead, every encoding is pulled towards the origin and kept from collapsing to a point (Figure 4). The result is that the **aggregate posterior**

$$
q_\phi(\mathbf{z})=\int q_\phi(\mathbf{z}\mid\mathbf{x})\,p_X(\mathbf{x})\,d\mathbf{x}\approx\frac1N\sum_{i=1}^Nq_\phi(\mathbf{z}\mid\mathbf{x}_i)
$$

roughly fills the prior. The encodings of the training points tile the region where the prior puts its mass, with no large holes. So a typical prior sample lands near the codes of real data, and the decoder has learned what to do there.

## Which way does the KL point?

The VAE fits $q_\phi$ to the true posterior by maximizing the ELBO, which by (1) means minimizing $D_{\mathrm{KL}}(q_\phi\,\|\,p_\theta(\mathbf{z}\mid\mathbf{x}))$: the **reverse** KL, with the approximation first. Maximum-likelihood training of $p_\theta$, by contrast, minimizes the **forward** KL $D_{\mathrm{KL}}(p_X\,\|\,p_\theta)$. The two behave very differently when the approximating family is too simple.

![[Machine Learning/images/generative-forward-reverse-kl.svg]]
*Figure 5: fitting a single Gaussian $q$ to a bimodal $p$. The forward KL $D_{\mathrm{KL}}(p\,\|\,q)$ is minimized by matching moments: $q$ covers both modes and puts much of its mass in the empty valley. The reverse KL $D_{\mathrm{KL}}(q\,\|\,p)$ is minimized by locking onto one mode: $q$ almost never puts mass where $p$ has none, at the price of ignoring the other mode.*

Reading the integrands explains the pictures.
- **Forward**, $\int p\log\frac{p}{q}$: wherever $p$ is large and $q$ is small, the log ratio is huge. So $q$ must cover **all** of $p$'s mass. Where $p\approx0$, the integrand vanishes whatever $q$ does, so mass in the valley costs almost nothing.
- **Reverse**, $\int q\log\frac{q}{p}$: wherever $q$ puts mass and $p$ is small, the log ratio is huge. So $q$ must **avoid** regions where $p$ has no mass, and it is free to ignore whole modes.

Two consequences for VAEs:
- The encoder's Gaussian $q_\phi(\mathbf{z}\mid\mathbf{x})$ tends to **under-estimate** the posterior's spread and pick one mode when the true posterior has several.
- The decoder is trained by (a bound on) maximum likelihood, the forward KL, which is **mode-covering**. It is reluctant to miss any data, and it spreads probability over the gaps between data. This is one reason VAE samples look blurry (see below). The same argument motivates using *other* divergences for the generator, which is where the [[Generative Adversarial Networks#Why not just maximize likelihood?|GAN lecture]] picks up.

## VAEs in practice

**Parameterize the log-variance.** The encoder outputs $\log\boldsymbol\sigma^2$ rather than $\boldsymbol\sigma$, so that the output is unconstrained and $\boldsymbol\sigma=\exp(\tfrac12\log\boldsymbol\sigma^2)>0$ automatically. The KL formula (3) uses $\log\sigma^2$ directly.

**Choose the decoder to match the data.**
- Real-valued data: a Gaussian decoder, giving the squared-error loss above.
- Binary or $[0,1]$-valued data (e.g. MNIST pixels): a Bernoulli decoder, $p_\theta(\mathbf{x}\mid\mathbf{z})=\prod_j\hat x_j^{x_j}(1-\hat x_j)^{1-x_j}$, whose negative log-likelihood is the **binary cross-entropy**, as in [[Linear Models|logistic regression]].

**Blurry samples.** With a Gaussian decoder of fixed variance, the decoder mean is the average of all images consistent with $\mathbf{z}$. When the code cannot pin down fine detail, the average is a blur. A forward-KL (maximum-likelihood) objective also rewards spreading mass over everything plausible, rather than committing to sharp, realistic samples (Figure 5).

**Posterior collapse.** If the decoder is powerful enough to model $\mathbf{x}$ on its own (for example an autoregressive decoder), the cheapest way to maximize the ELBO is to set $q_\phi(\mathbf{z}\mid\mathbf{x})=p(\mathbf{z})$. Then term ② is zero, and the latent code is simply ignored. Bowman et al. observed this in text VAEs.[^bowman2016] Common remedies:
- **KL annealing**: multiply term ② by a weight that grows from 0 to 1 during training;
- **free bits**: don't penalize the KL below a small budget per dimension;
- weaken the decoder, or drop words from its input.

**β-VAE.** Weighting the KL term by $\beta$ gives a dial between reconstruction and regularity; see [[#β-VAE and disentanglement]] below.

**Tighter bounds.** Averaging importance weights over several samples inside the log gives the importance-weighted bound (IWAE), which is tighter than the ELBO.[^burda2016]

**Evaluating the likelihood.** The ELBO gives a lower bound on $\log p_\theta(\mathbf{x})$ for any test point. That is a real advantage over GANs, which give no likelihood at all.

## Variants of the VAE

The Gaussian VAE above is the basic recipe. Three changes to it come up again and again: telling the model what to generate, reweighting the KL term, and stacking several layers of latents.

### Conditional VAEs

Often we want to *choose* what is generated: a particular digit, an image for a given caption, or one of the many plausible outputs for a given input (Sohn et al. used this for segmentation, where several labellings of an ambiguous image are all reasonable).[^sohn2015] A **conditional VAE** conditions every distribution on the extra information $\mathbf{y}$. The derivation of (2) goes through unchanged, with $\mathbf{y}$ carried along everywhere:

$$
\log p_\theta(\mathbf{x}\mid\mathbf{y})\ge\mathbb{E}_{q_\phi(\mathbf{z}\mid\mathbf{x},\mathbf{y})}\big[\log p_\theta(\mathbf{x}\mid\mathbf{z},\mathbf{y})\big]-D_{\mathrm{KL}}\big(q_\phi(\mathbf{z}\mid\mathbf{x},\mathbf{y})\,\|\,p(\mathbf{z}\mid\mathbf{y})\big).
$$

- The prior can stay fixed, $p(\mathbf{z}\mid\mathbf{y})=\mathcal{N}(\mathbf{0},\mathbf{I})$ (the usual choice for class labels), or be a network $p_\psi(\mathbf{z}\mid\mathbf{y})$ (Sohn et al., where $\mathbf{y}$ is an input image).
- The decoder is *told* $\mathbf{y}$, so $\mathbf{z}$ only needs to encode what $\mathbf{y}$ does not say. For handwritten digits with $\mathbf{y}$ the class, $\mathbf{z}$ ends up holding the style: slant, stroke width.
- To generate, fix $\mathbf{y}$, sample $\mathbf{z}$ from the prior, and decode. Keeping $\mathbf{z}$ and changing $\mathbf{y}$ redraws the same "handwriting" as a different digit.
- If $\mathbf{y}$ is observed for only some data points, it can be treated as a partly observed latent variable, which gives a semi-supervised classifier (Kingma et al.'s M2 model).[^kingma2014semi]

> [!example]- Code: conditioning a VAE on a class label (PyTorch)
> ```python
> # assumes: enc, dec as before but with K extra input units;
> #   x (B, D); y (B,) integer labels; d latent dims
> y1 = F.one_hot(y, K).float()
> mu, logvar = enc(torch.cat([x, y1], dim=1))    # q_phi(z | x, y)
> z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
> # p_theta(x | z, y); loss as in the negative ELBO
> x_hat = dec(torch.cat([z, y1], dim=1))
> # generation: choose the labels, sample z from the prior N(0, I)
> x_new = dec(torch.cat([torch.randn(len(y), d), y1], dim=1))
> ```
> For images, the one-hot label is usually broadcast to an $h\times w$ map and stacked as extra input channels, or embedded and added to the feature maps.

### β-VAE and disentanglement

Weighting the KL term,

$$
F_\beta=\mathbb{E}_{q_\phi(\mathbf{z}\mid\mathbf{x})}\big[\log p_\theta(\mathbf{x}\mid\mathbf{z})\big]-\beta\,D_{\mathrm{KL}}\big(q_\phi(\mathbf{z}\mid\mathbf{x})\,\|\,p(\mathbf{z})\big),
$$

gives a dial between reconstruction and regularity. With $\beta=1$ it is the ELBO. With $\beta>1$, $F_\beta\le F\le\log p_\theta(\mathbf{x})$ is still a lower bound, only a looser one; with $\beta<1$ it is no longer guaranteed to be a bound. In code, it is the `beta` in the negative-ELBO snippet above.

With $\beta>1$, Higgins et al. found that individual latent dimensions often line up with interpretable factors of variation (position, scale, rotation), which they called **disentanglement**; the price is blurrier reconstructions.[^higgins2017] Why should a heavier KL do that? Average the KL term over the data and split it, writing $q_\phi(\mathbf{z})$ for the aggregate posterior from [[#How do we sample?]]:

$$
\mathbb{E}_{p_X(\mathbf{x})}\Big[D_{\mathrm{KL}}\big(q_\phi(\mathbf{z}\mid\mathbf{x})\,\|\,p(\mathbf{z})\big)\Big]=I_q(\mathbf{x};\mathbf{z})+D_{\mathrm{KL}}\big(q_\phi(\mathbf{z})\,\|\,p(\mathbf{z})\big).
$$

(Write $\log\frac{q(\mathbf{z}\mid\mathbf{x})}{p(\mathbf{z})}=\log\frac{q(\mathbf{z}\mid\mathbf{x})}{q(\mathbf{z})}+\log\frac{q(\mathbf{z})}{p(\mathbf{z})}$ and take expectations.) A large $\beta$ therefore does two things:
- it limits the **information** the code carries about $\mathbf{x}$, an information bottleneck. With few bits to spend, the model spends them on the factors that explain the most variance in the data;
- it pushes the aggregate posterior towards the **factorized** prior $\mathcal{N}(\mathbf{0},\mathbf{I})$, which favours latent dimensions that vary independently.

Burgess et al. develop this reading and propose raising the allowed KL "capacity" gradually during training, which gives sharper reconstructions for the same degree of disentanglement.[^burgess2018]

> [!warning] Disentanglement is not guaranteed
> Locatello et al. showed that unsupervised disentanglement is impossible without inductive biases on the model and the data.[^locatello2019] The reason is simple: an isotropic Gaussian prior is unchanged by any rotation of $\mathbf{z}$, so a rotated (entangled) code explains the data exactly as well as an axis-aligned one, and the likelihood cannot prefer either. In a study of more than 12,000 trained models, the random seed and the hyperparameters mattered more than the choice of method, and without labels there was no reliable way to select the disentangled models. Treat $\beta$ as a reconstruction–regularity dial; interpretable axes are a hope that depends on the data and the architecture.

### Hierarchical VAEs

A single layer of Gaussian latents limits both the prior and the posterior. A **hierarchical VAE** stacks $L$ layers,

$$
p_\theta(\mathbf{x},\mathbf{z}_{1:L})=p(\mathbf{z}_L)\prod_{l=1}^{L-1}p_\theta(\mathbf{z}_l\mid\mathbf{z}_{l+1})\;p_\theta(\mathbf{x}\mid\mathbf{z}_1),
$$

and the KL term of the ELBO becomes a sum of per-layer KLs between the encoder's and the prior's conditionals. Top layers can capture global structure (pose, identity) and bottom layers local detail. Trained naively, the upper layers tend to be ignored: the posterior collapse of [[#VAEs in practice]], layer by layer.
- **Ladder VAE** (Sønderby et al.): a deterministic bottom-up pass gathers evidence from $\mathbf{x}$; the posterior is then built top-down, in the same order as the generative model, merging the bottom-up evidence with the top-down prior at every layer.[^sonderby2016]
- **NVAE** (Vahdat and Kautz): residual cells, a posterior parameterized *relative to* the prior, and spectral regularization for stability. It was the first VAE to produce convincing 256×256 face images.[^vahdat2020]
- **Very deep VAEs** (Child): with enough stochastic layers, a VAE can represent an autoregressive model, and such VAEs outperform PixelCNN in log-likelihood on natural-image benchmarks while sampling far faster.[^child2021]

A [[Denoising Diffusion Models|diffusion model]] can be read as the extreme case: a very deep hierarchical VAE whose encoder is a fixed noising process.

## Discrete latents: the VQ-VAE family

Much of the world is discrete: words, phonemes, object categories. A Gaussian code is not, and a powerful decoder fed continuous codes tends towards posterior collapse. The **vector-quantized VAE** (VQ-VAE) of van den Oord et al. makes the latent a grid of symbols from a learned **codebook**.[^vandenoord2017] There are two payoffs. Posterior collapse is avoided in practice, because the KL term turns into a constant. And the codes are **tokens**, so any sequence model can learn a prior over them; this is how modern image and audio tokenizers work.

### Quantizing the encoder output

The codebook is a set of $K$ learned vectors $\mathbf{e}_1,\dots,\mathbf{e}_K\in\mathbb{R}^{d}$. The encoder outputs a continuous vector $\mathbf{z}_e(\mathbf{x})\in\mathbb{R}^{d}$; for an image, one such vector per position of an $h\times w$ grid, each quantized separately. Quantization replaces it by its nearest code:

$$
k^*=\arg\min_{k}\big\lVert\mathbf{z}_e(\mathbf{x})-\mathbf{e}_k\big\rVert_2,\qquad\mathbf{z}_q(\mathbf{x})=\mathbf{e}_{k^*}.
\tag{4}
$$

The decoder sees only $\mathbf{z}_q$. In VAE terms:
- the posterior is one-hot, $q(z=k\mid\mathbf{x})=\mathbb{1}[k=k^*]$, and the prior during training is uniform over the $K$ codes;
- so $D_{\mathrm{KL}}(q\,\|\,p)=\log K$ per position, a **constant**. The ELBO reduces to the reconstruction term, and the regularization comes from the bottleneck itself: each position carries at most $\log_2K$ bits.

Geometrically, this is [[Clustering and Expectation Maximization|K-means]] in the latent space: the codes are centroids, and (4) is the assignment step.

### The straight-through gradient

$\mathbf{z}_q$ is piecewise constant in $\mathbf{z}_e$: a small move of $\mathbf{z}_e$ does not change the nearest code, so $\partial\mathbf{z}_q/\partial\mathbf{z}_e=\mathbf{0}$ almost everywhere (and it is undefined on the cell boundaries). Plain backpropagation would give the encoder no signal from the reconstruction loss at all.

The fix is the **straight-through estimator** of Bengio et al.:[^bengio2013] in the backward pass, pretend the quantizer is the identity, and copy the decoder's input gradient to the encoder's output,

$$
\nabla_{\mathbf{z}_e}\mathcal{L}_{\mathrm{rec}}:=\nabla_{\mathbf{z}_q}\mathcal{L}_{\mathrm{rec}}.
$$

It is implemented with a **stop-gradient** $\mathrm{sg}[\cdot]$, which is the identity in the forward pass and has zero derivative: $\mathbf{z}_e+\mathrm{sg}[\mathbf{z}_q-\mathbf{z}_e]$ has the value $\mathbf{z}_q$, but its derivative with respect to $\mathbf{z}_e$ is the identity.

**Why does copying the gradient work?** The decoder gradient at $\mathbf{z}_q$ says which way to move the decoder's *input* to reduce the loss. If $\mathbf{z}_e$ is close to $\mathbf{z}_q$, the same direction is a good one for $\mathbf{z}_e$: when the gradient is Lipschitz, the gradients at the two points differ by $O(\lVert\mathbf{z}_q-\mathbf{z}_e\rVert)$. Following it, $\mathbf{z}_e$ eventually crosses into another code's cell, and the selected code changes. The estimator is biased, since it is the gradient of a different function (the identity), but it has low variance, and the commitment loss below keeps the bias small by keeping $\mathbf{z}_e$ near its code.

**Who receives which gradient** (Figure 6):
- the **decoder** $\theta$: the ordinary gradient of the reconstruction loss, evaluated at $\mathbf{z}_q$;
- the **encoder** $\phi$: the copied reconstruction gradient, plus the gradient of the commitment loss;
- the **codebook**: no reconstruction gradient at all (the stop-gradient blocks it). Only the codebook loss, or the EMA update, moves it.

![[Machine Learning/images/vae-vq-straight-through.svg]]
*Figure 6: the VQ-VAE computation graph. Forward (grey): the encoder output $\mathbf{z}_e$ is replaced by its nearest codebook vector $\mathbf{z}_q$, which the decoder reconstructs from. Backward (dashed orange): the argmin has no useful gradient, so the gradient arriving at $\mathbf{z}_q$ is copied unchanged to $\mathbf{z}_e$ and continues into the encoder. The codebook gets no reconstruction gradient; it is trained by the codebook loss, and the commitment loss pulls the encoder output towards its code.*

> [!example]- Code: vector quantizer with the straight-through gradient (PyTorch)
> ```python
> # assumes: z_e (B, d) encoder outputs (grid positions flattened
> #   into B); E = nn.Parameter of shape (K, d)
> # ||z_e - e_k||^2, (B, K)
> d2 = (z_e.pow(2).sum(1, keepdim=True) - 2 * z_e @ E.T
>       + E.pow(2).sum(1))
> # eq. (4): index of the nearest code
> k = d2.argmin(dim=1)
> z_q = E[k]                            # codebook lookup
> # value z_q; gradient w.r.t. z_e copied from z_q
> z_q_st = z_e + (z_q - z_e).detach()
> x_hat = dec(z_q_st)
> ```
> `.detach()` is the stop-gradient $\mathrm{sg}[\cdot]$. Since `z_q` appears only inside it, the reconstruction loss sends no gradient to `E`.

### Codebook and commitment losses

The full VQ-VAE loss for one vector is

$$
\mathcal{L}_{\mathrm{VQ}}=-\log p_\theta(\mathbf{x}\mid\mathbf{z}_q)+\big\lVert\mathrm{sg}[\mathbf{z}_e]-\mathbf{e}_{k^*}\big\rVert^2+\beta\,\big\lVert\mathbf{z}_e-\mathrm{sg}[\mathbf{e}_{k^*}]\big\rVert^2.
\tag{5}
$$

- The **codebook loss** (second term) moves the selected code towards the encoder outputs assigned to it: the K-means update step, done by gradient descent.
- The **commitment loss** (third term) moves the encoder output towards its code. Without it, nothing stops $\mathbf{z}_e$ from drifting or growing faster than the codebook can follow, and the gradient copy becomes inaccurate.
- The two terms are the same squared distance with stop-gradients on opposite sides, so the codes and the encoder are pulled towards each other at different rates. Van den Oord et al. used $\beta=0.25$ (this $\beta$ is unrelated to the β of β-VAE) and found results robust over a wide range.

> [!example]- Code: the VQ-VAE loss (5) (PyTorch)
> ```python
> # assumes: x, x_hat = dec(z_q_st), z_e, z_q from the quantizer
> #   above; beta = 0.25
> # -log p(x | z_q) for a Gaussian decoder, up to scale
> rec = F.mse_loss(x_hat, x)
> # ||sg[z_e] - e||^2: moves the codes only
> codebook = F.mse_loss(z_q, z_e.detach())
> # ||z_e - sg[e]||^2: moves the encoder only
> commit = F.mse_loss(z_e, z_q.detach())
> loss = rec + codebook + beta * commit
> ```

### EMA codebook updates

The codebook loss performs K-means by gradient descent, with a learning rate to tune. An alternative, given in the appendix of the VQ-VAE paper and by Kaiser et al., replaces it with **exponential moving averages**.[^kaiser2018] Let $n_k$ be the number of encoder outputs assigned to code $k$ in the minibatch and $\mathbf{s}_k$ their sum. Then

$$
N_k\leftarrow\gamma N_k+(1-\gamma)\,n_k,\qquad\mathbf{m}_k\leftarrow\gamma\,\mathbf{m}_k+(1-\gamma)\,\mathbf{s}_k,\qquad\mathbf{e}_k=\frac{\mathbf{m}_k}{N_k},
$$

with decay $\gamma\approx0.99$. Each code becomes a running mean of the vectors assigned to it: online K-means. The codebook loss is dropped, and the commitment loss is kept. In practice this converges faster and more stably.

### Codebook collapse

A common failure is that only a few codes are ever used. A code that is nobody's nearest neighbour receives no update, neither gradient nor EMA, so once dead it stays dead, and the effective codebook shrinks. Monitor the **perplexity** of code usage, $\exp\big(H(\bar{\mathbf{p}})\big)$, where $\bar{\mathbf{p}}$ is the fraction of vectors assigned to each code: it equals $K$ when all codes are used equally and 1 when only one is. Remedies:
- **dead-code restarts**: reset codes whose usage falls below a threshold to randomly chosen recent encoder outputs (used in Jukebox, and studied by Łańcucki et al.);[^dhariwal2020]<sup>,</sup>[^lancucki2020]
- initialize the codebook by running K-means on encoder outputs from the first batches;
- look codes up in a low-dimensional, $\ell_2$-normalized space (cosine similarity), which raises codebook usage;[^yu2022]
- avoid a learned codebook altogether: **finite scalar quantization** rounds each of a few latent dimensions to a small fixed set of levels, giving an implicit codebook that cannot collapse, trained with the same straight-through copy.[^mentzer2023]

> [!example]- Code: EMA codebook update with dead-code restarts (PyTorch)
> ```python
> # assumes: z_e (B, d), k (B,) from the quantizer;
> #   codebook E (K, d); buffers N (K,), m (K, d); gamma = 0.99
> with torch.no_grad():
>     # (B, K) assignments
>     onehot = F.one_hot(k, K).float()
>     # N_k <- gamma N_k + (1 - gamma) n_k
>     N.mul_(gamma).add_((1 - gamma) * onehot.sum(0))
>     # m_k <- gamma m_k + (1 - gamma) s_k
>     m.mul_(gamma).add_((1 - gamma) * onehot.T @ z_e)
>     # e_k = m_k / N_k: running mean
>     E.copy_(m / N.clamp_min(1e-5).unsqueeze(1))
>     # codes nobody has picked for a while
>     dead = N < 1e-3
>     # restart at random encoder outputs
>     E[dead] = z_e[torch.randint(len(z_e), (int(dead.sum()),))]
>     m[dead], N[dead] = E[dead], 1.0
>     p_bar = onehot.mean(0)
>     # K = all codes used evenly
>     H = -(p_bar * torch.log(p_bar + 1e-10)).sum()
>     perplexity = torch.exp(H)
> ```

### A prior over codes, and sampling

After training, the autoencoder maps every image to a grid of code indices, say $32\times32$ integers in $\{1,\dots,K\}$. The uniform prior was only a placeholder: decoding uniformly random indices gives incoherent images. So a **second stage** fits an autoregressive prior over the indices, $p_\psi(k_1,\dots,k_n)=\prod_ip_\psi(k_i\mid k_{<i})$, in raster order: a PixelCNN in the original paper,[^vandenoord2016] a transformer in later work. To sample, draw indices from $p_\psi$, look up their codes, and decode. The division of labour is the point: the autoencoder handles local detail and compression, and the prior models global structure over a sequence of about a thousand tokens instead of tens of thousands of pixel values.

### VQ-VAE-2 and VQGAN

- **VQ-VAE-2** (Razavi et al.) uses a hierarchy of two code grids, a coarse top level for global structure and a finer bottom level for detail, each with its own PixelCNN prior. Its ImageNet samples rivalled those of GANs in quality, with more diversity.[^razavi2019]
- **VQGAN** (Esser et al.) trains the autoencoder with a perceptual loss plus a patch-based adversarial loss from a [[Generative Adversarial Networks|GAN]] discriminator, so the decoder can synthesize sharp detail from a coarse grid (e.g. $16\times16$), and models the codes with a transformer, reaching megapixel images.[^esser2021]
- Autoencoders of this kind, VQ- or KL-regularized, are the first stage of **latent diffusion**: a [[Denoising Diffusion Models|diffusion model]] runs in their compressed latent space.[^rombach2022] And because the codes are tokens, the machinery of [[Language Representations and Language Models|language models]] can generate images and audio.

## Key takeaways

- The VAE is EM with a learned, amortized E-step: an encoder $q_\phi(\mathbf{z}\mid\mathbf{x})$ replaces the intractable posterior, and the ELBO is maximized over both $\theta$ and $\phi$.
- ELBO = reconstruction $-$ KL to the prior. The gap to $\log p_\theta(\mathbf{x})$ is $D_{\mathrm{KL}}(q_\phi\,\|\,p_\theta(\mathbf{z}\mid\mathbf{x}))$.
- Gradients with respect to the parameters of a *sampling distribution* need a trick. The score-function estimator always works but is noisy; the **reparameterization** $\mathbf{z}=\boldsymbol\mu+\boldsymbol\sigma\odot\boldsymbol\epsilon$ gives low-variance pathwise gradients.
- With Gaussian encoder and prior the KL term is closed form; with a Gaussian decoder the reconstruction term is a squared error.
- Sampling works because the KL term makes the **aggregate** posterior fill the prior, not because each encoding equals the prior.
- Reverse KL is mode-seeking, forward KL is mode-covering. Watch for blurry samples and posterior collapse.
- Variants: condition on $\mathbf{y}$ (CVAE); weight the KL by $\beta$ (a dial, not a guarantee of disentanglement); stack latents (Ladder VAE, NVAE). The VQ-VAE quantizes to a codebook, copies the decoder gradient straight through the argmin, trains the codes with a codebook (or EMA) loss plus a commitment loss, and samples by fitting an autoregressive prior over the code indices.

## Where this shows up next

- [[Generative Adversarial Networks]] drop the likelihood entirely and train a sampler against a learned critic, using the f-divergences and Fenchel duality.
- [[Policy Gradients]] use the score-function estimator from this lecture. The [[Policy Gradients#Straight-Through Gumbel-Softmax (STGS) for PG Algorithms|Gumbel-softmax]] section shows a pathwise alternative for discrete variables.
- [[Matrix Factorization and Dimensionality Reduction#Principal component analysis (PCA)|PCA]] is the linear-Gaussian limit: a VAE with linear encoder and decoder recovers probabilistic PCA.
- [[Energy-Based Models]] give up the sampler instead: a single energy network defines an unnormalized density, sampled with MCMC.
- [[Denoising Diffusion Models]] can be read as very deep hierarchical VAEs with a fixed noising encoder; latent diffusion uses a VAE (KL- or VQ-regularized) as its first stage.
- [[Language Representations and Language Models]]: VQ codes are tokens, so the same autoregressive machinery models images and audio.
- [[Generative Adversarial Networks]]: VQGAN's adversarial loss, and the conditional GANs that parallel the conditional VAE.

<!-- References: footnote definitions below render as the numbered "References" list at the bottom of the page -->

[^kingma2014]: Kingma, D. P., & Welling, M. (2014). *Auto-Encoding Variational Bayes*. ICLR. [arXiv:1312.6114](https://arxiv.org/abs/1312.6114)
[^rezende2014]: Rezende, D. J., Mohamed, S., & Wierstra, D. (2014). *Stochastic Backpropagation and Approximate Inference in Deep Generative Models*. ICML. [arXiv:1401.4082](https://arxiv.org/abs/1401.4082)
[^mohamed2020]: Mohamed, S., Rosca, M., Figurnov, M., & Mnih, A. (2020). *Monte Carlo Gradient Estimation in Machine Learning*. JMLR, 21(132). [arXiv:1906.10652](https://arxiv.org/abs/1906.10652)
[^bowman2016]: Bowman, S. R., Vilnis, L., Vinyals, O., Dai, A. M., Jozefowicz, R., & Bengio, S. (2016). *Generating Sentences from a Continuous Space*. CoNLL. [arXiv:1511.06349](https://arxiv.org/abs/1511.06349)
[^higgins2017]: Higgins, I., Matthey, L., Pal, A., Burgess, C., Glorot, X., Botvinick, M., Mohamed, S., & Lerchner, A. (2017). *β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework*. ICLR. [OpenReview](https://openreview.net/forum?id=Sy2fzU9gl)
[^burda2016]: Burda, Y., Grosse, R., & Salakhutdinov, R. (2016). *Importance Weighted Autoencoders*. ICLR. [arXiv:1509.00519](https://arxiv.org/abs/1509.00519)
[^sohn2015]: Sohn, K., Lee, H., & Yan, X. (2015). *Learning Structured Output Representation using Deep Conditional Generative Models*. NeurIPS. [proceedings](https://proceedings.neurips.cc/paper_files/paper/2015/hash/8d55a249e6baa5c06772297520da2051-Abstract.html)
[^kingma2014semi]: Kingma, D. P., Rezende, D. J., Mohamed, S., & Welling, M. (2014). *Semi-Supervised Learning with Deep Generative Models*. NeurIPS. [arXiv:1406.5298](https://arxiv.org/abs/1406.5298)
[^burgess2018]: Burgess, C. P., Higgins, I., Pal, A., Matthey, L., Watters, N., Desjardins, G., & Lerchner, A. (2018). *Understanding disentangling in β-VAE*. arXiv. [arXiv:1804.03599](https://arxiv.org/abs/1804.03599)
[^locatello2019]: Locatello, F., Bauer, S., Lucic, M., Rätsch, G., Gelly, S., Schölkopf, B., & Bachem, O. (2019). *Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations*. ICML. [arXiv:1811.12359](https://arxiv.org/abs/1811.12359)
[^sonderby2016]: Sønderby, C. K., Raiko, T., Maaløe, L., Sønderby, S. K., & Winther, O. (2016). *Ladder Variational Autoencoders*. NeurIPS. [arXiv:1602.02282](https://arxiv.org/abs/1602.02282)
[^vahdat2020]: Vahdat, A., & Kautz, J. (2020). *NVAE: A Deep Hierarchical Variational Autoencoder*. NeurIPS. [arXiv:2007.03898](https://arxiv.org/abs/2007.03898)
[^child2021]: Child, R. (2021). *Very Deep VAEs Generalize Autoregressive Models and Can Outperform Them on Images*. ICLR. [arXiv:2011.10650](https://arxiv.org/abs/2011.10650)
[^vandenoord2017]: van den Oord, A., Vinyals, O., & Kavukcuoglu, K. (2017). *Neural Discrete Representation Learning*. NeurIPS. [arXiv:1711.00937](https://arxiv.org/abs/1711.00937)
[^bengio2013]: Bengio, Y., Léonard, N., & Courville, A. (2013). *Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation*. arXiv. [arXiv:1308.3432](https://arxiv.org/abs/1308.3432)
[^kaiser2018]: Kaiser, Ł., Roy, A., Vaswani, A., Parmar, N., Bengio, S., Uszkoreit, J., & Shazeer, N. (2018). *Fast Decoding in Sequence Models using Discrete Latent Variables*. ICML. [arXiv:1803.03382](https://arxiv.org/abs/1803.03382)
[^dhariwal2020]: Dhariwal, P., Jun, H., Payne, C., Kim, J. W., Radford, A., & Sutskever, I. (2020). *Jukebox: A Generative Model for Music*. arXiv. [arXiv:2005.00341](https://arxiv.org/abs/2005.00341)
[^lancucki2020]: Łańcucki, A., Chorowski, J., Sanchez, G., Marxer, R., Chen, N., Dolfing, H. J. G. A., Khurana, S., Alumäe, T., & Laurent, A. (2020). *Robust Training of Vector Quantized Bottleneck Models*. IJCNN. [arXiv:2005.08520](https://arxiv.org/abs/2005.08520)
[^yu2022]: Yu, J., Li, X., Koh, J. Y., Zhang, H., Pang, R., Qin, J., Ku, A., Xu, Y., Baldridge, J., & Wu, Y. (2022). *Vector-quantized Image Modeling with Improved VQGAN*. ICLR. [arXiv:2110.04627](https://arxiv.org/abs/2110.04627)
[^mentzer2023]: Mentzer, F., Minnen, D., Agustsson, E., & Tschannen, M. (2024). *Finite Scalar Quantization: VQ-VAE Made Simple*. ICLR. [arXiv:2309.15505](https://arxiv.org/abs/2309.15505)
[^vandenoord2016]: van den Oord, A., Kalchbrenner, N., Vinyals, O., Espeholt, L., Graves, A., & Kavukcuoglu, K. (2016). *Conditional Image Generation with PixelCNN Decoders*. NeurIPS. [arXiv:1606.05328](https://arxiv.org/abs/1606.05328)
[^razavi2019]: Razavi, A., van den Oord, A., & Vinyals, O. (2019). *Generating Diverse High-Fidelity Images with VQ-VAE-2*. NeurIPS. [arXiv:1906.00446](https://arxiv.org/abs/1906.00446)
[^esser2021]: Esser, P., Rombach, R., & Ommer, B. (2021). *Taming Transformers for High-Resolution Image Synthesis*. CVPR. [arXiv:2012.09841](https://arxiv.org/abs/2012.09841)
[^rombach2022]: Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022). *High-Resolution Image Synthesis with Latent Diffusion Models*. CVPR. [arXiv:2112.10752](https://arxiv.org/abs/2112.10752)
