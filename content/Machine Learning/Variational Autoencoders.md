---
title: Variational Autoencoders
tags:
  - generative-models
  - variational-inference
  - autoencoders
  - representation-learning
date: 2026-11-12
draft: false
prev: "[[Clustering and Expectation Maximization]]"
next: "[[Generative Adversarial Networks]]"
---

> **Prerequisites:** [[Matrix Factorization and Dimensionality Reduction#Autoencoders|Autoencoders]] (encoder, bottleneck, decoder) and [[Clustering and Expectation Maximization#The evidence lower bound (ELBO)|the evidence lower bound]] from the EM lecture. [[Policy Gradients]] is helpful for the gradient-estimator aside.
>
> **TL;DR:**
> - A VAE is a latent-variable model $p_\theta(\mathbf{x}\mid\mathbf{z})p(\mathbf{z})$ trained by maximizing the **ELBO**. A second network, the **encoder** $q_\phi(\mathbf{z}\mid\mathbf{x})$, approximates the intractable posterior.
> - The ELBO splits into a **reconstruction** term and a **KL-to-prior** term. With Gaussian encoder and prior, the KL term has a closed form.
> - The **reparameterization trick** $\mathbf{z}=\boldsymbol\mu_\phi(\mathbf{x})+\boldsymbol\sigma_\phi(\mathbf{x})\odot\boldsymbol\epsilon$ moves the randomness into an input, so ordinary backpropagation gives low-variance gradients for $\phi$.
> - To generate, sample $\mathbf{z}\sim\mathcal{N}(\mathbf{0},\mathbf{I})$ and decode.

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

**β-VAE.** Weighting the KL term,

$$
F_\beta=\mathbb{E}_{q_\phi(\mathbf{z}\mid\mathbf{x})}\big[\log p_\theta(\mathbf{x}\mid\mathbf{z})\big]-\beta\,D_{\mathrm{KL}}\big(q_\phi(\mathbf{z}\mid\mathbf{x})\,\|\,p(\mathbf{z})\big),
$$

gives a dial between reconstruction and regularity. With $\beta>1$, Higgins et al. found that individual latent dimensions often line up with interpretable factors of variation (position, scale, rotation), which they called **disentanglement**; the price is blurrier reconstructions.[^higgins2017] With $\beta\ne1$, $F_\beta$ is no longer a bound on $\log p_\theta(\mathbf{x})$.

**Tighter bounds.** Averaging importance weights over several samples inside the log gives the importance-weighted bound (IWAE), which is tighter than the ELBO.[^burda2016]

**Evaluating the likelihood.** The ELBO gives a lower bound on $\log p_\theta(\mathbf{x})$ for any test point. That is a real advantage over GANs, which give no likelihood at all.

## Key takeaways

- The VAE is EM with a learned, amortized E-step: an encoder $q_\phi(\mathbf{z}\mid\mathbf{x})$ replaces the intractable posterior, and the ELBO is maximized over both $\theta$ and $\phi$.
- ELBO = reconstruction $-$ KL to the prior. The gap to $\log p_\theta(\mathbf{x})$ is $D_{\mathrm{KL}}(q_\phi\,\|\,p_\theta(\mathbf{z}\mid\mathbf{x}))$.
- Gradients with respect to the parameters of a *sampling distribution* need a trick. The score-function estimator always works but is noisy; the **reparameterization** $\mathbf{z}=\boldsymbol\mu+\boldsymbol\sigma\odot\boldsymbol\epsilon$ gives low-variance pathwise gradients.
- With Gaussian encoder and prior the KL term is closed form; with a Gaussian decoder the reconstruction term is a squared error.
- Sampling works because the KL term makes the **aggregate** posterior fill the prior, not because each encoding equals the prior.
- Reverse KL is mode-seeking, forward KL is mode-covering. Watch for blurry samples and posterior collapse.

## Where this shows up next

- [[Generative Adversarial Networks]] drop the likelihood entirely and train a sampler against a learned critic, using the f-divergences and Fenchel duality.
- [[Policy Gradients]] use the score-function estimator from this lecture. The [[Policy Gradients#Straight-Through Gumbel-Softmax (STGS) for PG Algorithms|Gumbel-softmax]] section shows a pathwise alternative for discrete variables.
- [[Matrix Factorization and Dimensionality Reduction#Principal component analysis (PCA)|PCA]] is the linear-Gaussian limit: a VAE with linear encoder and decoder recovers probabilistic PCA.
- [[Energy-Based Models]] give up the sampler instead: a single energy network defines an unnormalized density, sampled with MCMC.
- Later lectures (planned): VQ-VAE tokenizers, hierarchical VAEs, and diffusion models, which can be read as very deep VAEs with a fixed encoder.

<!-- References: footnote definitions below render as the numbered "References" list at the bottom of the page -->

[^kingma2014]: Kingma, D. P., & Welling, M. (2014). *Auto-Encoding Variational Bayes*. ICLR. [arXiv:1312.6114](https://arxiv.org/abs/1312.6114)
[^rezende2014]: Rezende, D. J., Mohamed, S., & Wierstra, D. (2014). *Stochastic Backpropagation and Approximate Inference in Deep Generative Models*. ICML. [arXiv:1401.4082](https://arxiv.org/abs/1401.4082)
[^mohamed2020]: Mohamed, S., Rosca, M., Figurnov, M., & Mnih, A. (2020). *Monte Carlo Gradient Estimation in Machine Learning*. JMLR, 21(132). [arXiv:1906.10652](https://arxiv.org/abs/1906.10652)
[^bowman2016]: Bowman, S. R., Vilnis, L., Vinyals, O., Dai, A. M., Jozefowicz, R., & Bengio, S. (2016). *Generating Sentences from a Continuous Space*. CoNLL. [arXiv:1511.06349](https://arxiv.org/abs/1511.06349)
[^higgins2017]: Higgins, I., Matthey, L., Pal, A., Burgess, C., Glorot, X., Botvinick, M., Mohamed, S., & Lerchner, A. (2017). *β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework*. ICLR. [OpenReview](https://openreview.net/forum?id=Sy2fzU9gl)
[^burda2016]: Burda, Y., Grosse, R., & Salakhutdinov, R. (2016). *Importance Weighted Autoencoders*. ICLR. [arXiv:1509.00519](https://arxiv.org/abs/1509.00519)
