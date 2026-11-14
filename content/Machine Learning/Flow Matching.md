---
title: Flow Matching
tags:
  - generative-models
  - flow-matching
  - rectified-flow
  - continuous-normalizing-flows
date: 2026-11-13
draft: false
prev: "[[Denoising Diffusion Models]]"
next: "[[Normalizing Flows]]"
---

> **Prerequisites:** [[Denoising Diffusion Models]] and [[Score-Based Models]], especially [[Score-Based Models#Infinitely many noise levels: the SDE view|the probability flow ODE]] and [[Score-Based Models#Tweedie's formula: the score is the optimal denoiser|Tweedie's formula]]. From the foundations: [[Joint Distributions#The best predictor|the conditional mean as the best predictor]], [[Multivariable Calculus#The Jacobian and the chain rule|Jacobians]] and [[Multivariable Calculus#Densities under a change of variables|densities under a change of variables]].
>
> **TL;DR:**
> - A **flow** moves noise to data along an ordinary differential equation $d\mathbf{x}/dt=\mathbf{v}_\theta(\mathbf{x},t)$. Flow matching learns the velocity field $\mathbf{v}_\theta$ by plain **regression**, with no ODE solves during training and no noising SDE.
> - The trick, as in denoising score matching, is to regress onto the velocity of a path towards **one** data point. Averaged over data points, the minimizer is the velocity of the whole distribution.
> - With **straight lines** from noise to data the regression target is just $\mathbf{x}_1-\mathbf{x}_0$. Straight paths are cheap to integrate, and **rectified flow** (reflow) and **minibatch optimal transport** make the learned flow straighter still, down to one or two steps.
> - **Diffusion models are flow matching with a curved path.** Velocity, noise prediction, data prediction and the score are four parameterizations of one function, and the probability flow ODE is the diffusion model's flow.
>
> **Notation and time direction:** in this lecture **$t=0$ is noise and $t=1$ is data**, the convention of the flow-matching papers and of the companion code. This is the **reverse** of the [[Denoising Diffusion Models|diffusion lecture]], where $\mathbf{x}_0$ is data and $\mathbf{x}_T$ noise. Here $\mathbf{x}_0\sim p_0=\mathcal{N}(\mathbf{0},\mathbf{I})$ is noise and $\mathbf{x}_1\sim q$ is data, so the diffusion lecture's $(\mathbf{x}_0,\boldsymbol\epsilon)$ is our $(\mathbf{x}_1,\mathbf{x}_0)$, and a continuous diffusion time $s\in[0,1]$ (or $s=\tau/T$ for the diffusion lecture's discrete step, written $\tau$ here because $t$ is taken) corresponds to $t=1-s$. Subscripts on $\mathbf{x}$ denote time, not data indices. The table in [[Flow Matching#Diffusion as flow matching|Diffusion as flow matching]] collects all conversions.
>
> **Code:** [Conditional-Flow-Matching](https://github.com/parag1604/Conditional-Flow-Matching): flow matching from scratch, from a 2-D toy (with the one-step "mean" failure and a no-time ablation) to 64×64 CelebA faces with a U-Net, EMA and Euler/midpoint/Heun/RK4 samplers, a conditional flow for 24→64 super-resolution, and minibatch-OT pairing in 2-D.

## Motivation

The [[Denoising Diffusion Models|diffusion lecture]] ended with a deterministic sampler: the probability flow ODE, whose velocity is built from a learned score. That is a roundabout route to an ODE. We defined a noising SDE, learned the score of every noisy marginal, and only then assembled a velocity field out of the score and the SDE's coefficients.

Flow matching asks the direct question. If we want an ODE that carries noise to data, why not choose a path of distributions from noise to data ourselves, and regress the velocity field that moves along it? Three things make this work:
- **Training is simulation-free.** Like denoising score matching, the loss is a regression onto a target we can compute in closed form from a (noise, data) pair. No ODE is solved during training.
- **The path is a design choice.** Straight lines from noise to data are the simplest choice. They give the simplest target, $\mathbf{x}_1-\mathbf{x}_0$, and trajectories that a solver can integrate in few steps.
- **Diffusion is a special case.** Choosing the variance-preserving path recovers the diffusion model's probability flow ODE exactly, so everything learned about diffusion carries over.

The model itself is old: a **continuous normalizing flow** (CNF), a neural ODE that transports a simple density.[^chen2018] CNFs used to be trained by maximum likelihood, which needs an ODE solve, and a trace of a Jacobian, inside every gradient step.[^grathwohl2019] Flow matching keeps the model and replaces the training. Three groups proposed it at about the same time in 2022: as **flow matching**,[^lipman2023] as **rectified flow**,[^liu2023] and as **stochastic interpolants**.[^albergo2023]

## Flows and velocity fields

A time-dependent **velocity field** $\mathbf{u}_t:\mathbb{R}^D\to\mathbb{R}^D$, $t\in[0,1]$, defines an ordinary differential equation

$$
\frac{d\mathbf{x}_t}{dt}=\mathbf{u}_t(\mathbf{x}_t).
\tag{1}
$$

Start a particle anywhere at $t=0$ and follow the arrows. The **flow** $\boldsymbol\psi_t$ maps each starting point to where it is at time $t$: $\boldsymbol\psi_0(\mathbf{x})=\mathbf{x}$ and $\tfrac{d}{dt}\boldsymbol\psi_t(\mathbf{x})=\mathbf{u}_t(\boldsymbol\psi_t(\mathbf{x}))$.
- **Existence and uniqueness.** If $\mathbf{u}_t$ is Lipschitz in $\mathbf{x}$ (true for a neural network with smooth activations, on bounded sets), the Picard–Lindelöf theorem gives one solution through every starting point.
- **Trajectories never cross.** Two trajectories that met at some time would be two solutions through the same point, contradicting uniqueness. This innocent fact drives the rectified-flow section.
- **The flow is invertible.** Run (1) backwards in time from $\mathbf{x}_1$ to recover $\mathbf{x}_0$.

Now start not one particle but a whole distribution, $\mathbf{x}_0\sim p_0$. At time $t$ the particles are distributed as some $p_t$, the **push-forward** of $p_0$ by $\boldsymbol\psi_t$. We say $\mathbf{u}_t$ **generates** the probability path $(p_t)_{t\in[0,1]}$. A generative model of this kind is a network $\mathbf{v}_\theta(\mathbf{x},t)$ with as many outputs as inputs, and sampling means drawing $\mathbf{x}_0\sim\mathcal{N}(\mathbf{0},\mathbf{I})$ and solving (1) with $\mathbf{u}_t=\mathbf{v}_\theta(\cdot,t)$ up to $t=1$. Training means choosing $\theta$ so that $p_1$ is the data distribution $q$.

## Probability paths and the continuity equation

How does a velocity field change a density? Think of probability as a fluid. The mass inside a fixed region can change only by flowing through the region's boundary. In one dimension, the mass in an interval $[x_-,x_+]$ is $\int_{x_-}^{x_+}p_t\,dx$, and the rate at which it crosses a point $x$ is the **flux** $p_t(x)\,u_t(x)$: density times speed. Mass enters at $x_-$ and leaves at $x_+$, so

$$
\frac{d}{dt}\int_{x_-}^{x_+}p_t(x)\,dx=p_t(x_-)u_t(x_-)-p_t(x_+)u_t(x_+)=-\int_{x_-}^{x_+}\frac{\partial}{\partial x}\big(p_tu_t\big)\,dx.
$$

This holds for every interval, so the integrands agree. In $D$ dimensions the derivative becomes a divergence, and we get the **continuity equation**:

$$
\frac{\partial p_t}{\partial t}+\nabla\cdot\big(p_t\,\mathbf{u}_t\big)=0.
\tag{2}
$$

Two quick checks. A constant velocity $u_t(x)=b$ in 1-D should translate the density, $p_t(x)=p_0(x-bt)$; indeed $\partial_tp_t=-b\,p_0'$ and $\partial_x(p_tb)=b\,p_0'$, which sum to zero. And a velocity field pointing outwards, $\nabla\cdot\mathbf{u}_t>0$, spreads particles apart, so the density at a moving particle drops.

> [!proof]- Derivation of the continuity equation from the ODE
> Take any smooth test function $f$ that vanishes outside a bounded set, and track its average along the flow, $\mathbb{E}[f(\mathbf{x}_t)]=\int f(\mathbf{x})\,p_t(\mathbf{x})\,d\mathbf{x}$. Differentiating the integral form,
>
> $$
> \frac{d}{dt}\mathbb{E}[f(\mathbf{x}_t)]=\int f\,\frac{\partial p_t}{\partial t}\,d\mathbf{x}.
> $$
>
> Differentiating along trajectories instead, by the chain rule and (1), $\tfrac{d}{dt}f(\mathbf{x}_t)=\nabla f(\mathbf{x}_t)^\top\mathbf{u}_t(\mathbf{x}_t)$, so
>
> $$
> \frac{d}{dt}\mathbb{E}[f(\mathbf{x}_t)]=\int\nabla f^\top\mathbf{u}_t\,p_t\,d\mathbf{x}=-\int f\,\nabla\cdot\big(p_t\mathbf{u}_t\big)\,d\mathbf{x}.
> $$
>
> The last step is integration by parts (the divergence theorem); the boundary term vanishes because $f$ does. The two expressions agree for every test function, so $\partial_tp_t=-\nabla\cdot(p_t\mathbf{u}_t)$.

> [!note] Likelihoods come for free (a preview)
> Expanding (2) along a trajectory gives $\tfrac{d}{dt}\log p_t(\mathbf{x}_t)=-\nabla\cdot\mathbf{u}_t(\mathbf{x}_t)$: the log-density at a moving particle changes at minus the rate at which the flow expands volume. Integrating from noise to data,
>
> $$
> \log p_1(\mathbf{x}_1)=\log p_0(\mathbf{x}_0)-\int_0^1\nabla\cdot\mathbf{u}_t(\mathbf{x}_t)\,dt,
> $$
>
> the continuous-time version of the change-of-variables formula, with the log-determinant of the Jacobian replaced by an integrated trace. Flow matching never trains on this quantity, but a trained flow can evaluate it, using Hutchinson's trace estimator from [[Score-Based Models#Sliced score matching|sliced score matching]]. The [[Normalizing Flows]] lecture builds models around exactly this formula.

One more fact matters below: **the velocity field generating a given path is not unique.** Adding any field $\tilde{\mathbf{u}}_t$ with $\nabla\cdot(p_t\tilde{\mathbf{u}}_t)=0$, such as a rotation around a radially symmetric density, leaves (2) unchanged. Different fields with the same marginals move individual particles differently. Flow matching picks one particular field, and the rectified-flow and optimal-transport sections are about picking a good one.

## The flow matching objective

Suppose we have chosen a probability path $p_t$ with $p_0=\mathcal{N}(\mathbf{0},\mathbf{I})$ and $p_1\approx q$, and a field $\mathbf{u}_t$ that generates it. The **flow matching** loss regresses the network onto that field:

$$
\mathcal{L}_{\text{FM}}(\theta)=\mathbb{E}_{t\sim\mathcal{U}[0,1],\ \mathbf{x}\sim p_t}\Big[\big\lVert\mathbf{v}_\theta(\mathbf{x},t)-\mathbf{u}_t(\mathbf{x})\big\rVert^2\Big].
\tag{3}
$$

As written it is useless, for the same reason as [[Score-Based Models#Score matching|explicit score matching]]: we know neither $p_t$ nor $\mathbf{u}_t$. All we have is samples from the data. The way out is also the same: condition on a single data point.

## Conditional flow matching

### Build the path from one data point at a time

Pick, for each data point $\mathbf{x}_1$, a **conditional probability path** $p_t(\mathbf{x}\mid\mathbf{x}_1)$ that starts at the noise distribution, $p_0(\mathbf{x}\mid\mathbf{x}_1)=\mathcal{N}(\mathbf{x};\mathbf{0},\mathbf{I})$, and ends concentrated at the data point, $p_1(\mathbf{x}\mid\mathbf{x}_1)=\mathcal{N}(\mathbf{x};\mathbf{x}_1,\sigma_{\min}^2\mathbf{I})$ with a small $\sigma_{\min}\ge0$. Also pick a **conditional velocity field** $\mathbf{u}_t(\mathbf{x}\mid\mathbf{x}_1)$ that generates it. Both are simple, because a single point is simple: below they are Gaussians and straight lines.

Averaging over the data gives the **marginal probability path**

$$
p_t(\mathbf{x})=\int p_t(\mathbf{x}\mid\mathbf{x}_1)\,q(\mathbf{x}_1)\,d\mathbf{x}_1,
\tag{4}
$$

which starts at $p_0=\mathcal{N}(\mathbf{0},\mathbf{I})$ and ends at $p_1=q$ blurred by $\sigma_{\min}$ (Figure 1). Which velocity field generates it? Not the average of the conditional fields, but their average **weighted by how likely each data point is to be the destination, given the current position**:

$$
\mathbf{u}_t(\mathbf{x})=\int\mathbf{u}_t(\mathbf{x}\mid\mathbf{x}_1)\,\frac{p_t(\mathbf{x}\mid\mathbf{x}_1)\,q(\mathbf{x}_1)}{p_t(\mathbf{x})}\,d\mathbf{x}_1
=\mathbb{E}\big[\mathbf{u}_t(\mathbf{x}_t\mid\mathbf{x}_1)\,\big|\,\mathbf{x}_t=\mathbf{x}\big].
\tag{5}
$$

The fraction is the posterior $p(\mathbf{x}_1\mid\mathbf{x}_t=\mathbf{x})$: Bayes' rule, with the conditional path as the likelihood. It plays the role of the [[Clustering and Expectation Maximization#E-step: responsibilities|responsibilities]] in EM, and of the weights in the score of a mixture in the [[Score-Based Models#The score|score lecture]].

![[Machine Learning/images/flow-matching-paths.svg]]
*Figure 1. The straight-line probability path on a toy problem: noise $\mathcal{N}(\mathbf{0},\mathbf{I})$ at $t=0$, eight Gaussians on a circle at $t=1$. Top: the conditional path $p_t(\mathbf{x}\mid\mathbf{x}_1)$ for one data point (star), a Gaussian that shrinks onto it as $t\to1$. Bottom: the marginal path $p_t(\mathbf{x})$, the average of the conditional paths over all data points, which morphs the Gaussian into the eight modes.*

> [!proof]- Proof that the marginal field (5) generates the marginal path (4)
> Each conditional pair satisfies the continuity equation, $\partial_tp_t(\mathbf{x}\mid\mathbf{x}_1)=-\nabla\cdot\big(p_t(\mathbf{x}\mid\mathbf{x}_1)\,\mathbf{u}_t(\mathbf{x}\mid\mathbf{x}_1)\big)$. Differentiate (4) under the integral sign (assume enough regularity to do so):
>
> $$
> \begin{aligned}
> \frac{\partial p_t(\mathbf{x})}{\partial t}
> &=\int\frac{\partial p_t(\mathbf{x}\mid\mathbf{x}_1)}{\partial t}\,q(\mathbf{x}_1)\,d\mathbf{x}_1\\
> &=-\int\nabla\cdot\big(p_t(\mathbf{x}\mid\mathbf{x}_1)\,\mathbf{u}_t(\mathbf{x}\mid\mathbf{x}_1)\big)\,q(\mathbf{x}_1)\,d\mathbf{x}_1\\
> &=-\nabla\cdot\Big(\int p_t(\mathbf{x}\mid\mathbf{x}_1)\,\mathbf{u}_t(\mathbf{x}\mid\mathbf{x}_1)\,q(\mathbf{x}_1)\,d\mathbf{x}_1\Big)\\
> &=-\nabla\cdot\big(p_t(\mathbf{x})\,\mathbf{u}_t(\mathbf{x})\big).
> \end{aligned}
> $$
>
> The last equality is the definition (5), multiplied by $p_t(\mathbf{x})$. So $(p_t,\mathbf{u}_t)$ satisfies the continuity equation (2), and $\mathbf{u}_t$ generates $p_t$. The whole proof is the linearity of (2) in the pair "density times velocity".

### The conditional loss has the same gradient

Equation (5) still contains the intractable $p_t(\mathbf{x})$. But we never need to evaluate it. Regress onto the **conditional** field instead, sampling the point $\mathbf{x}$ from the conditional path:

$$
\mathcal{L}_{\text{CFM}}(\theta)=\mathbb{E}_{t\sim\mathcal{U}[0,1],\ \mathbf{x}_1\sim q,\ \mathbf{x}\sim p_t(\cdot\mid\mathbf{x}_1)}\Big[\big\lVert\mathbf{v}_\theta(\mathbf{x},t)-\mathbf{u}_t(\mathbf{x}\mid\mathbf{x}_1)\big\rVert^2\Big].
\tag{6}
$$

Every quantity in (6) is available: draw a data point, a time and a point on its conditional path, and compute the conditional velocity. The key result of Lipman et al.:[^lipman2023] **$\mathcal{L}_{\text{CFM}}$ and $\mathcal{L}_{\text{FM}}$ differ by a constant that does not depend on $\theta$**, so they have the same gradients and the same minimizer.

> [!proof]- Proof: the two losses differ by a constant
> Fix $t$ and expand both squares. The terms $\lVert\mathbf{u}_t(\mathbf{x})\rVert^2$ and $\lVert\mathbf{u}_t(\mathbf{x}\mid\mathbf{x}_1)\rVert^2$ do not involve $\theta$. For the others:
> - **The $\lVert\mathbf{v}_\theta\rVert^2$ terms are equal.** By (4), drawing $\mathbf{x}_1\sim q$ and then $\mathbf{x}\sim p_t(\cdot\mid\mathbf{x}_1)$ is a way of drawing $\mathbf{x}\sim p_t$, so $\mathbb{E}_{q(\mathbf{x}_1)p_t(\mathbf{x}\mid\mathbf{x}_1)}\lVert\mathbf{v}_\theta(\mathbf{x},t)\rVert^2=\mathbb{E}_{p_t(\mathbf{x})}\lVert\mathbf{v}_\theta(\mathbf{x},t)\rVert^2$.
> - **The cross terms are equal.** Using (5),
>
> $$
> \begin{aligned}
> &\mathbb{E}_{p_t(\mathbf{x})}\big[\mathbf{v}_\theta(\mathbf{x},t)^\top\mathbf{u}_t(\mathbf{x})\big]\\
> &\quad=\int\mathbf{v}_\theta(\mathbf{x},t)^\top\Big(\int\mathbf{u}_t(\mathbf{x}\mid\mathbf{x}_1)\,p_t(\mathbf{x}\mid\mathbf{x}_1)\,q(\mathbf{x}_1)\,d\mathbf{x}_1\Big)d\mathbf{x}\\
> &\quad=\mathbb{E}_{q(\mathbf{x}_1)\,p_t(\mathbf{x}\mid\mathbf{x}_1)}\big[\mathbf{v}_\theta(\mathbf{x},t)^\top\mathbf{u}_t(\mathbf{x}\mid\mathbf{x}_1)\big].
> \end{aligned}
> $$
>
> Hence $\mathcal{L}_{\text{FM}}(\theta)-\mathcal{L}_{\text{CFM}}(\theta)$ is a sum of $\theta$-free terms. A shorter route to the same conclusion: (6) is a least-squares regression of the target $\mathbf{u}_t(\mathbf{x}_t\mid\mathbf{x}_1)$ on the input $(\mathbf{x}_t,t)$, and the [[Joint Distributions#The best predictor|best predictor]] under squared error is the conditional mean, which by (5) is exactly $\mathbf{u}_t(\mathbf{x})$.

This is the same move as [[Score-Based Models#Denoising score matching|denoising score matching]], where regressing onto the score of the noise kernel recovers the score of the noisy marginal. Here, regressing onto the velocity towards one data point recovers the velocity of the whole distribution. The regression target is noisy (a point $\mathbf{x}$ can be on the way to many data points), but its conditional mean is the field we want.

### Conditioning on both ends

A slightly more general view, used by rectified flow and by the optimal-transport section, conditions on a **pair** $\boldsymbol\xi=(\mathbf{x}_0,\mathbf{x}_1)$ drawn from a **coupling** $\pi(\mathbf{x}_0,\mathbf{x}_1)$, any joint distribution whose marginals are the noise $p_0$ and the data $q$.[^tong2024] Given the pair, the path is deterministic, $\mathbf{x}_t=\boldsymbol\psi_t(\mathbf{x}_0,\mathbf{x}_1)$, and so is its velocity. The proofs above go through verbatim with $\mathbf{x}_1$ replaced by $\boldsymbol\xi$. The **independent coupling**, $\pi=p_0\times q$, draws noise and data separately; it is what we have used so far.

## Gaussian conditional paths

The standard choice is a Gaussian conditional path,

$$
p_t(\mathbf{x}\mid\mathbf{x}_1)=\mathcal{N}\big(\mathbf{x};\ \boldsymbol\mu_t(\mathbf{x}_1),\ \sigma_t^2\mathbf{I}\big),
$$

with $\boldsymbol\mu_0=\mathbf{0}$, $\sigma_0=1$ (pure noise) and $\boldsymbol\mu_1=\mathbf{x}_1$, $\sigma_1=\sigma_{\min}$ (the data point). The simplest flow that produces it shifts and scales the noise:

$$
\boldsymbol\psi_t(\mathbf{x}_0)=\sigma_t\,\mathbf{x}_0+\boldsymbol\mu_t(\mathbf{x}_1),
\qquad\mathbf{x}_0\sim\mathcal{N}(\mathbf{0},\mathbf{I}).
$$

Its velocity is $\tfrac{d}{dt}\boldsymbol\psi_t(\mathbf{x}_0)=\dot\sigma_t\mathbf{x}_0+\dot{\boldsymbol\mu}_t$, where dots are time derivatives. To write it as a field of the position $\mathbf{x}$, substitute $\mathbf{x}_0=(\mathbf{x}-\boldsymbol\mu_t)/\sigma_t$:

$$
\mathbf{u}_t(\mathbf{x}\mid\mathbf{x}_1)=\frac{\dot\sigma_t}{\sigma_t}\big(\mathbf{x}-\boldsymbol\mu_t(\mathbf{x}_1)\big)+\dot{\boldsymbol\mu}_t(\mathbf{x}_1).
\tag{7}
$$

The first term contracts the cloud towards its moving centre ($\dot\sigma_t<0$), and the second moves the centre. In the loss (6) we never need (7) as a field: we sample $\mathbf{x}_0$ and use $\mathbf{x}=\boldsymbol\psi_t(\mathbf{x}_0)$ with target $\dot\sigma_t\mathbf{x}_0+\dot{\boldsymbol\mu}_t$.

### The straight-line (optimal transport) path

Interpolate the mean and the standard deviation linearly:

$$
\boldsymbol\mu_t=t\,\mathbf{x}_1,\qquad\sigma_t=1-(1-\sigma_{\min})\,t.
$$

Then each noise sample travels in a straight line at constant speed,

$$
\boldsymbol\psi_t(\mathbf{x}_0)=\big(1-(1-\sigma_{\min})t\big)\,\mathbf{x}_0+t\,\mathbf{x}_1,
\qquad
\frac{d\boldsymbol\psi_t}{dt}=\mathbf{x}_1-(1-\sigma_{\min})\,\mathbf{x}_0,
\tag{8}
$$

and the regression target does not even depend on $t$. Lipman et al. call it the **optimal transport (OT) path**, because between the two Gaussians $\mathcal{N}(\mathbf{0},\mathbf{I})$ and $\mathcal{N}(\mathbf{x}_1,\sigma_{\min}^2\mathbf{I})$ this straight-line flow is the optimal transport map, the one that moves mass the least total squared distance.[^peyre2019] The name is about the **conditional** flow only. The marginal flow it induces is generally **not** the optimal transport map between noise and data; the next two sections show how to get closer to it.

With $\sigma_{\min}=0$, (8) becomes the formula used in practice and in the companion code: $\mathbf{x}_t=(1-t)\,\mathbf{x}_0+t\,\mathbf{x}_1$, target $\mathbf{x}_1-\mathbf{x}_0$. A small positive $\sigma_{\min}$ keeps the endpoint distribution a proper density; it hardly matters for sampling.

## Training and sampling

Putting the pieces together, flow matching with the straight-line path is a few lines of code.

> [!example] Algorithm: conditional flow matching (training)
> 1. Draw a minibatch of data $\mathbf{x}_1\sim q$, noise $\mathbf{x}_0\sim\mathcal{N}(\mathbf{0},\mathbf{I})$ of the same shape, and times $t\sim\mathcal{U}[0,1]$, one per example.
> 2. Form the points on the lines, $\mathbf{x}_t=(1-t)\,\mathbf{x}_0+t\,\mathbf{x}_1$.
> 3. Take a gradient step on $\big\lVert\mathbf{v}_\theta(\mathbf{x}_t,t)-(\mathbf{x}_1-\mathbf{x}_0)\big\rVert^2$, averaged over the batch.
> 4. Repeat until the loss plateaus. The loss never reaches zero: the target is a noisy sample of its own conditional mean.

> [!example] Algorithm: sampling with Euler steps
> 1. Draw $\mathbf{x}\sim\mathcal{N}(\mathbf{0},\mathbf{I})$ and choose a number of steps $n$, with step size $h=1/n$.
> 2. For $j=0,\dots,n-1$, with $t=jh$: update $\mathbf{x}\leftarrow\mathbf{x}+h\,\mathbf{v}_\theta(\mathbf{x},t)$.
> 3. Return $\mathbf{x}$, an approximate sample from the data distribution.

> [!example]- Code: flow matching in PyTorch (training and Euler sampling)
> A minimal implementation for 2-D data. On the eight Gaussians of Figure 1, 5000 Adam steps take about half a minute on a laptop CPU; 50 Euler steps then give a sliced Wasserstein distance to the data of 0.14, against 0.10 between two independent samples of the data, and the learned field at $t=0.5$ is within 11% of the exact one.
>
> ```python
> import torch
> import torch.nn as nn
>
> class Velocity(nn.Module):
>     """v_theta(x, t) for 2-D data: an MLP on the concatenation [x, t]."""
>     def __init__(self, d=2, h=256):
>         super().__init__()
>         self.net = nn.Sequential(
>             nn.Linear(d + 1, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(),
>             nn.Linear(h, h), nn.SiLU(), nn.Linear(h, d))
>
>     def forward(self, x, t):
>         return self.net(torch.cat([x, t[:, None]], dim=1))
>
> def cfm_loss(model, x1):
>     x0 = torch.randn_like(x1)                       # noise at t = 0
>     t = torch.rand(len(x1))                         # t ~ U(0, 1)
>     xt = (1 - t[:, None]) * x0 + t[:, None] * x1    # point on the straight line
>     return ((model(xt, t) - (x1 - x0)) ** 2).sum(1).mean()
>
> @torch.no_grad()
> def sample(model, n, steps=50, d=2):
>     x = torch.randn(n, d)                           # start from noise
>     for i in range(steps):
>         t = torch.full((n,), i / steps)
>         x = x + model(x, t) / steps                 # Euler step of size 1/steps
>     return x
>
> def train(sample_data, iters=5000, batch=512):
>     model = Velocity()
>     opt = torch.optim.Adam(model.parameters(), lr=1e-3)
>     for _ in range(iters):
>         loss = cfm_loss(model, sample_data(batch))
>         opt.zero_grad()
>         loss.backward()
>         opt.step()
>     return model
> ```

Two properties of the learned field explain most practical surprises.
- **One Euler step returns the mean of the data.** At $t=0$ the point $\mathbf{x}_0$ is pure noise, independent of the data point it is paired with, so the optimal field is $\mathbf{u}_0(\mathbf{x})=\mathbb{E}[\mathbf{x}_1-\mathbf{x}_0\mid\mathbf{x}_0=\mathbf{x}]=\mathbb{E}[\mathbf{x}_1]-\mathbf{x}$. A single step of size 1 lands every sample on $\mathbb{E}[\mathbf{x}_1]$: for images, the average face. Several steps are needed, because only at later times does $\mathbf{x}_t$ carry information about its destination (Figure 5c shows this in 2-D).
- **The field must depend on $t$.** The same point means "mostly noise, keep going" at $t=0.1$ and "almost data, stop" at $t=0.9$. A network that cannot tell times apart, for example because its time embedding barely changes over $[0,1]$, learns a time-averaged field and produces blurry averages. The companion repository documents exactly this failure in an earlier version of its image code.

### ODE solvers

Sampling is numerical integration of (1), and the cost is counted in **network evaluations** (NFE). Write $h$ for the step and $\mathbf{v}(\mathbf{x},t)$ for the learned field.
- **Euler**: $\mathbf{x}_{t+h}=\mathbf{x}_t+h\,\mathbf{v}(\mathbf{x}_t,t)$. One evaluation per step; the error per step is $O(h^2)$, so the global error is $O(h)$. It is exact when trajectories are straight lines traversed at constant speed, which is the whole point of straight paths.
- **Midpoint**: take half an Euler step, evaluate the field there, and use that slope for the full step: $\mathbf{x}_{t+h}=\mathbf{x}_t+h\,\mathbf{v}\big(\mathbf{x}_t+\tfrac h2\mathbf{v}(\mathbf{x}_t,t),\,t+\tfrac h2\big)$. Two evaluations per step, global error $O(h^2)$.
- **Heun** (the trapezoidal rule as predictor–corrector): an Euler step predicts $\bar{\mathbf{x}}=\mathbf{x}_t+h\,\mathbf{v}(\mathbf{x}_t,t)$, and the update averages the two slopes, $\mathbf{x}_{t+h}=\mathbf{x}_t+\tfrac h2\big[\mathbf{v}(\mathbf{x}_t,t)+\mathbf{v}(\bar{\mathbf{x}},t+h)\big]$. Two evaluations per step, global error $O(h^2)$. Karras et al. popularized it for diffusion sampling.[^karras2022]
- **Adaptive solvers** (e.g. Dormand–Prince RK45) pick the step size from an error estimate. They are accurate but spend many evaluations where the field is stiff.

> [!example]- Code: Euler, midpoint and Heun steps
> ```python
> # assumes: model(x, t) -> v_theta(x, t), as above; x (n, D); t (n,) the current time; h the step
> def euler_step(model, x, t, h):
>     return x + h * model(x, t)
>
> def midpoint_step(model, x, t, h):
>     x_half = x + 0.5 * h * model(x, t)                  # half an Euler step
>     return x + h * model(x_half, t + 0.5 * h)           # full step with the midpoint slope
>
> def heun_step(model, x, t, h):
>     v0 = model(x, t)
>     x_pred = x + h * v0                                 # Euler predictor
>     return x + 0.5 * h * (v0 + model(x_pred, t + h))    # average the two slopes
> ```
> Replace the update line of `sample` with any of these. On a test ODE with a known solution, halving $h$ halves the Euler error and quarters the midpoint and Heun errors.

![[Machine Learning/images/flow-matching-solvers.svg]]
*Figure 2. Sample quality against cost for the flow trained on the eight Gaussians with independent pairs, measured by the sliced Wasserstein distance (SWD) between 10 000 samples and 10 000 data points; the grey band is the range of SWD between two independent data samples, so differences inside it are noise. With 2 evaluations, two Euler steps (0.55) beat one midpoint step (1.20) or one Heun step (1.65). From 8 evaluations on, all three solvers are within the noise band.*

Higher order pays off only once the step is small enough for the Taylor expansion behind it to be accurate. With very few evaluations it is better to spend them on more Euler steps, and on this toy problem the three solvers are indistinguishable from 8 evaluations on. On 64×64 faces the companion repository found Heun with 5 steps visibly noisier than midpoint with the same 10 evaluations, a difference it could not fully explain; with 50 steps both are fine. Making the trajectories themselves straighter attacks the problem at its source.

## Rectified flow

Liu, Gong and Liu reached the same algorithm from a different starting point.[^liu2023] Take any coupling of noise and data, $(\mathbf{X}_0,\mathbf{X}_1)\sim\pi$, and its linear interpolation $\mathbf{X}_t=(1-t)\mathbf{X}_0+t\mathbf{X}_1$. Its **rectified flow** is the ODE

$$
\frac{d\mathbf{Z}_t}{dt}=\mathbf{v}(\mathbf{Z}_t,t),
\qquad
\mathbf{v}(\mathbf{x},t)=\mathbb{E}\big[\mathbf{X}_1-\mathbf{X}_0\ \big|\ \mathbf{X}_t=\mathbf{x}\big],
\qquad
\mathbf{Z}_0=\mathbf{X}_0,
\tag{9}
$$

whose velocity is learned by the same regression. With the independent coupling this is flow matching with the straight-line path and $\sigma_{\min}=0$. What the rectified-flow view adds is a careful look at the **trajectories**, and a way to straighten them.

### Why the marginal flow is not straight

If the trajectories of (9) were straight lines traversed at constant speed, a single Euler step would integrate them exactly: $\mathbf{Z}_1=\mathbf{Z}_0+\mathbf{v}(\mathbf{Z}_0,0)$. The training lines are straight. The learned trajectories are not, because **the training lines cross**. With independent pairs, a noise point on the left is as likely to be paired with a data point on the right as on the left. Where two lines cross, the network sees two different targets at the same $(\mathbf{x},t)$ and learns their average. And ODE trajectories cannot cross (uniqueness), so they bend away from each other at every crossing (Figure 3).

![[Machine Learning/images/flow-matching-crossing.svg]]
*Figure 3. (a) Sixty training pairs from the independent coupling, joined by straight lines: many lines cross. (b) The trajectories of the exact marginal flow (9) from the same noise points, computed with the closed-form field of the mixture. They do not cross, so they curve; on average a trajectory is 19% longer than the straight segment between its endpoints. The flow learned from these pairs (Figure 4a) has the same excess length, 18.7%.*

The marginal flow still does its job: it transports noise to data exactly. And its endpoints define a new, **deterministic** coupling, each noise point $\mathbf{Z}_0$ paired with the data point $\mathbf{Z}_1$ it flows to. This coupling is better than the one we started with, in a precise sense:

> [!tip] Theorem (Liu et al., 2023)
> Let $\mathbf{Z}$ solve (9). Then (i) **the marginals are preserved**: $\mathbf{Z}_t$ has the same distribution as $\mathbf{X}_t$ for every $t$, in particular $\mathbf{Z}_1\sim q$; and (ii) **transport costs do not increase**: $\mathbb{E}\,c(\mathbf{Z}_1-\mathbf{Z}_0)\le\mathbb{E}\,c(\mathbf{X}_1-\mathbf{X}_0)$ for **every** convex cost $c$, such as $\lVert\cdot\rVert^2$ or $\lVert\cdot\rVert$.

> [!proof]- Proof
> (i) is the conditional flow matching result again: $\mathbf{v}$ in (9) is the field (5) for the coupling's straight-line paths, so it generates the law of $\mathbf{X}_t$.
>
> (ii) follows from Jensen's inequality, used twice. Write $\mathbf{Z}_1-\mathbf{Z}_0=\int_0^1\mathbf{v}(\mathbf{Z}_t,t)\,dt$, an average over $t$. Then
>
> $$
> \begin{aligned}
> \mathbb{E}\,c(\mathbf{Z}_1-\mathbf{Z}_0)
> &\le\int_0^1\mathbb{E}\,c\big(\mathbf{v}(\mathbf{Z}_t,t)\big)\,dt
> =\int_0^1\mathbb{E}\,c\big(\mathbf{v}(\mathbf{X}_t,t)\big)\,dt\\
> &=\int_0^1\mathbb{E}\,c\Big(\mathbb{E}\big[\mathbf{X}_1-\mathbf{X}_0\mid\mathbf{X}_t\big]\Big)\,dt\\
> &\le\int_0^1\mathbb{E}\,c\big(\mathbf{X}_1-\mathbf{X}_0\big)\,dt
> =\mathbb{E}\,c(\mathbf{X}_1-\mathbf{X}_0).
> \end{aligned}
> $$
>
> The first inequality is Jensen over $t$; the equality after it uses (i), that $\mathbf{Z}_t$ and $\mathbf{X}_t$ have the same law; the second inequality is conditional Jensen. See [[Sample Statistics#Jensen's inequality|Jensen's inequality]].

### Reflow

So: train a rectified flow, push noise through it to get pairs $(\mathbf{Z}_0,\mathbf{Z}_1)$, and train a **new** rectified flow on those pairs. This is **reflow**. The new training lines join points that the previous flow already connected without crossing, so they cross much less, and the new flow is straighter. Repeating gives the $k$-rectified flow. Liu et al. measure straightness by

$$
S(\mathbf{Z})=\int_0^1\mathbb{E}\Big[\big\lVert(\mathbf{Z}_1-\mathbf{Z}_0)-\mathbf{v}(\mathbf{Z}_t,t)\big\rVert^2\Big]dt,
$$

which is zero exactly when every trajectory is a straight line at constant speed, and they show that repeated reflow drives it to zero, at a rate of order $1/k$ for the best of the first $k$ flows. A coupling whose straight lines never cross is a fixed point: its interpolation already has non-crossing straight paths, so its rectified flow follows them exactly.

Three caveats. Each reflow needs a full pass of ODE solves to generate training pairs, and fits a new network. The pairs carry the previous model's errors, so sample quality can degrade slightly with each round. And a straight coupling is not necessarily the optimal transport coupling; it is merely one with no crossings.[^liu2022]

> [!example]- Code: one round of reflow
> ```python
> # assumes: model, a trained rectified flow v(x, t), and the Velocity class from above
> @torch.no_grad()
> def reflow_pairs(model, n, steps=100, d=2):
>     z0 = torch.randn(n, d)                              # noise
>     z1 = z0.clone()
>     for i in range(steps):                              # push it through the flow (9)
>         z1 = z1 + model(z1, torch.full((n,), i / steps)) / steps
>     return z0, z1                                       # a deterministic coupling
>
> def reflow_loss(model2, z0, z1):                        # cfm_loss, with the pairs held fixed
>     t = torch.rand(len(z0))
>     zt = (1 - t[:, None]) * z0 + t[:, None] * z1
>     return ((model2(zt, t) - (z1 - z0)) ** 2).sum(1).mean()
> ```
> Generate the pairs once, then train a fresh `Velocity()` on minibatches of them with `reflow_loss`. The only change from `cfm_loss` is where the pairs come from.

### Distillation to one step

Once the flow is nearly straight, $\mathbf{Z}_1\approx\mathbf{Z}_0+\mathbf{v}(\mathbf{Z}_0,0)$, and we can train a **one-step generator** directly: regress a network $\mathbf{g}(\mathbf{x}_0)$ onto $\mathbf{Z}_1-\mathbf{Z}_0$ over the reflow pairs. Distilling a straight flow is much easier than distilling a curved one, because the map from noise to data is smoother and the student only has to learn where each trajectory ends. InstaFlow used one reflow followed by distillation to turn Stable Diffusion into a one-step text-to-image generator.[^liu2024instaflow]

![[Machine Learning/images/flow-matching-reflow.svg]]
*Figure 4. Rectified flow on the eight Gaussians. (a) The 1-rectified flow, trained on independent pairs. (b) The 2-rectified flow, trained on 100 000 pairs generated by (a): visibly straighter. (c) Sample quality against the number of Euler steps (SWD and grey band as in Figure 2). Excess length drops from 18.7% to 0.01%, and a single Euler step goes from an SWD of 2.09 (every sample at the data mean) to 0.08, within sampling noise. The dashed curve uses the exact field of the independent coupling, which the 1-rectified flow tracks closely. One-step students distilled from either flow (diamond, star) are also within noise: in two dimensions a small network can learn even the curved map from noise to data, so the advantage of distilling a straight flow does not show here.*

## Minibatch optimal-transport couplings

Reflow straightens the flow after the fact. A cheaper idea is to start from a better coupling. The ideal coupling for squared distance is the **optimal transport plan** between noise and data. Its straight lines never cross at intermediate times (a classical property of optimal transport), so the conditional velocity at each point is already unambiguous, the marginal flow follows the straight lines, and it is the optimal transport map itself.[^peyre2019]

Computing the OT plan between two continuous distributions is out of reach, but within a minibatch it is an **assignment problem**: given $B$ noise samples and $B$ data samples, find the permutation $\rho$ minimizing $\sum_i\lVert\mathbf{x}_0^{(i)}-\mathbf{x}_1^{(\rho(i))}\rVert^2$. Solve it exactly (the Hungarian algorithm, $O(B^3)$) or approximately (Sinkhorn iterations), re-pair the batch, and train with the unchanged loss. This is **OT-CFM**,[^tong2024] proposed at the same time as **multisample flow matching**.[^pooladian2023] As $B\to\infty$ the minibatch plans approach the true OT plan. With finite batches the coupling is only approximately optimal, but it already removes most crossings, which both straightens the flow and lowers the variance of the regression target.

> [!example]- Code: minibatch OT pairing
> Drop-in before computing the loss: pair the noise batch `x0` with the data batch `x1`, then train as before.
>
> ```python
> from scipy.optimize import linear_sum_assignment
>
> def ot_pairs(x0, x1):
>     """Re-pair a minibatch so that the total squared distance is minimal."""
>     cost = torch.cdist(x0, x1) ** 2                 # (B, B) pairwise costs
>     rows, cols = linear_sum_assignment(cost.numpy())
>     return x0[rows], x1[cols]
> ```
>
> Checked against brute force over all permutations of 6-point batches. On batches of 512 from the toy problem, it cuts the mean squared distance between paired points from 10.3 to 3.5.

![[Machine Learning/images/flow-matching-ot-cfm.svg]]
*Figure 5. Independent pairs against minibatch-OT pairs (batch 512), same network and training budget. (a, b) Learned trajectories. Excess length falls from 18.7% to 0.1%. (c, d) Samples after a single Euler step (crosses: mode centres). With independent pairs every sample lands at the data mean (SWD 2.09); with OT pairs the samples are on the ring already (SWD 0.13), with some spread between the modes that disappears with two or more steps.*

The gains shrink in high dimensions. A batch of a few hundred images is a very sparse sample of a distribution in $10^4$ or more dimensions, so the minibatch plan is far from the population plan. The companion repository found little benefit on 64×64 faces with batches of 64, and kept independent pairs there. For conditional generation, where each condition has its own data distribution, pairing within the batch must respect the conditions.

## Diffusion as flow matching

Every diffusion model in the previous lectures uses a Gaussian conditional path. In flow-matching time, they are all of the form

$$
\mathbf{x}_t=a_t\,\mathbf{x}_1+\sigma_t\,\mathbf{x}_0,
\qquad a_0\approx0,\ \sigma_0\approx1,\quad a_1=1,\ \sigma_1\approx0,
$$

that is, $\boldsymbol\mu_t=a_t\mathbf{x}_1$ in the notation of (7). The straight-line path is $a_t=t$, $\sigma_t=1-t$. The others:

| | this lecture | diffusion lecture |
|---|---|---|
| data, noise | $\mathbf{x}_1$, $\mathbf{x}_0$ | $\mathbf{x}_0$, $\boldsymbol\epsilon$ |
| time | $t\in[0,1]$, data at $t=1$ | step $\tau=1,\dots,T$ or $s\in[0,1]$, data at $0$; $t=1-s$ |
| variance preserving (DDPM) | $a_t^2+\sigma_t^2=1$ | $a_t=\sqrt{\bar\alpha_\tau}$, $\sigma_t=\sqrt{1-\bar\alpha_\tau}$ |
| continuous VP | $a_t=e^{-\frac12\int_0^{1-t}\beta(r)\,dr}$ | $d\mathbf{x}=-\frac12\beta\mathbf{x}\,ds+\sqrt{\beta}\,d\mathbf{w}$ |
| variance exploding (NCSN) | $a_t=1$, $\sigma_t=\sigma(1-t)$ | $\tilde{\mathbf{x}}=\mathbf{x}+\sigma\boldsymbol\epsilon$ |
| predicted data, noise | $\hat{\mathbf{x}}_1$, $\hat{\boldsymbol\epsilon}$ | $\hat{\mathbf{x}}_\theta$, $\boldsymbol\epsilon_\theta$ |

The variance-exploding path does not start exactly at $\mathcal{N}(\mathbf{0},\mathbf{I})$ but at a very wide Gaussian, $\mathcal{N}(\mathbf{0},\sigma_{\max}^2\mathbf{I})$ up to the data.

### Four parameterizations of one function

For any such path, define the two posterior means

$$
\hat{\mathbf{x}}_1(\mathbf{x},t)=\mathbb{E}[\mathbf{x}_1\mid\mathbf{x}_t=\mathbf{x}],
\qquad
\hat{\boldsymbol\epsilon}(\mathbf{x},t)=\mathbb{E}[\mathbf{x}_0\mid\mathbf{x}_t=\mathbf{x}].
$$

Everything else is a function of either one:
- **They determine each other.** Take the conditional expectation of $\mathbf{x}_t=a_t\mathbf{x}_1+\sigma_t\mathbf{x}_0$ given $\mathbf{x}_t=\mathbf{x}$: $\mathbf{x}=a_t\hat{\mathbf{x}}_1+\sigma_t\hat{\boldsymbol\epsilon}$.
- **The score** is $\nabla_{\mathbf{x}}\log p_t(\mathbf{x})=-\hat{\boldsymbol\epsilon}/\sigma_t$. This is [[Score-Based Models#Tweedie's formula: the score is the optimal denoiser|Tweedie's formula]], applied to $\mathbf{x}_t=a_t\mathbf{x}_1+\sigma_t\mathbf{x}_0$.
- **The velocity** is, by (5) with the conditional velocity $\dot a_t\mathbf{x}_1+\dot\sigma_t\mathbf{x}_0$, the posterior mean of that: $\mathbf{u}_t=\dot a_t\hat{\mathbf{x}}_1+\dot\sigma_t\hat{\boldsymbol\epsilon}$.

Eliminating one of the predictions with the first bullet gives the conversions. In terms of the noise prediction and of the score,

$$
\mathbf{u}_t(\mathbf{x})=\frac{\dot a_t}{a_t}\,\mathbf{x}+\Big(\dot\sigma_t-\frac{\dot a_t\,\sigma_t}{a_t}\Big)\hat{\boldsymbol\epsilon}
=\frac{\dot a_t}{a_t}\,\mathbf{x}+\Big(\frac{\dot a_t\,\sigma_t^2}{a_t}-\sigma_t\dot\sigma_t\Big)\nabla_{\mathbf{x}}\log p_t(\mathbf{x}).
\tag{10}
$$

For the straight-line path ($a_t=t$, $\sigma_t=1-t$) these become memorable:

$$
\mathbf{u}_t(\mathbf{x})=\frac{\hat{\mathbf{x}}_1-\mathbf{x}}{1-t}=\frac{\mathbf{x}-\hat{\boldsymbol\epsilon}}{t}=\frac{\mathbf{x}+(1-t)\,\nabla_{\mathbf{x}}\log p_t(\mathbf{x})}{t}.
$$

The first form says the velocity points from the current position to the predicted data point, at the speed needed to arrive by $t=1$. Conversely, a trained velocity network gives $\hat{\mathbf{x}}_1=\mathbf{x}+(1-t)\mathbf{v}_\theta$ and $\hat{\boldsymbol\epsilon}=\mathbf{x}-t\mathbf{v}_\theta$ for free.

> [!example]- Code: from a velocity to the other three parameterizations
> For the straight-line path. Checked against the closed-form posterior means and score of the eight-Gaussian mixture.
>
> ```python
> def from_velocity(v, x, t):
>     """Straight-line path x_t = t x1 + (1 - t) x0, with 0 < t < 1."""
>     x1_hat = x + (1 - t) * v          # predicted data point,   E[x1 | x_t = x]
>     x0_hat = x - t * v                # predicted noise,        E[x0 | x_t = x]
>     score = -x0_hat / (1 - t)         # grad log p_t(x)
>     return x1_hat, x0_hat, score
> ```

The **losses** are related the same way. Writing a velocity network in terms of a noise network, $\mathbf{v}_\theta=\frac{\dot a_t}{a_t}\mathbf{x}+\big(\dot\sigma_t-\frac{\dot a_t\sigma_t}{a_t}\big)\boldsymbol\epsilon_\theta$, and the conditional target in the same way, the flow-matching loss becomes

$$
\big\lVert\mathbf{v}_\theta-(\dot a_t\mathbf{x}_1+\dot\sigma_t\mathbf{x}_0)\big\rVert^2
=\Big(\dot\sigma_t-\frac{\dot a_t\sigma_t}{a_t}\Big)^2\,\big\lVert\boldsymbol\epsilon_\theta-\mathbf{x}_0\big\rVert^2,
$$

the diffusion lecture's noise-prediction loss with a time-dependent weight ($1/t^2$ for the straight-line path). Flow matching, $\boldsymbol\epsilon$-prediction and $\mathbf{x}$-prediction all minimize the same family of regressions and differ only in how they weight the noise levels; Kingma and Gao show that all of these weightings are ELBOs of data with noise augmentation.[^kingma2023] One more parameterization fits here. For a variance-preserving path write $a=\cos\phi$, $\sigma=\sin\phi$; then $\tfrac{d\mathbf{x}}{d\phi}=\cos\phi\,\mathbf{x}_0-\sin\phi\,\mathbf{x}_1$, which is exactly the **v-prediction** target of Salimans and Ho.[^salimans2022] v-prediction is velocity prediction with the angle as time.

### The probability flow ODE is a flow-matching flow

For a variance-preserving path, $a_t^2+\sigma_t^2=1$, so $\sigma_t\dot\sigma_t=-a_t\dot a_t$, and the score coefficient in (10) simplifies to $\dot a_t/a_t$. With $a_t=\exp\big(-\frac12\int_0^{1-t}\beta(r)\,dr\big)$ we have $\dot a_t/a_t=\frac12\beta(1-t)$, and (10) becomes

$$
\frac{d\mathbf{x}}{dt}=\frac12\,\beta(1-t)\Big[\mathbf{x}+\nabla_{\mathbf{x}}\log p_t(\mathbf{x})\Big].
\tag{11}
$$

Substitute diffusion time $s=1-t$: this is $\tfrac{d\mathbf{x}}{ds}=-\tfrac12\beta(s)\big[\mathbf{x}+\nabla_{\mathbf{x}}\log p\big]$, the **probability flow ODE** of the variance-preserving SDE from the [[Denoising Diffusion Models|diffusion lecture]] and from Song et al.[^song2021sde] So a diffusion model sampled deterministically (DDIM, or any ODE solver) is a flow model with the VP path, and flow matching with the VP path trains the same field, up to the weighting of the loss. The two families differ in which path they choose, not in kind.

### Straight-line paths against the VP path

Two differences are visible already at the level of one data point.
- **Shape.** The straight-line conditional path is a straight segment from $\mathbf{x}_0$ to $\mathbf{x}_1$. The VP conditional path $a_t\mathbf{x}_1+\sigma_t\mathbf{x}_0$ with $a_t^2+\sigma_t^2=1$ is an arc of an ellipse in the plane of $\mathbf{x}_0$ and $\mathbf{x}_1$: a quarter circle when the two have equal length and are orthogonal, which noise and data roughly are in high dimensions.
- **Speed.** The straight-line path moves at constant speed. The VP path with the standard linear $\beta$ schedule barely moves for the first part of the time interval and then rushes, so a uniform time grid wastes steps where nothing happens and takes too few where everything does.

Lipman et al. found that the straight-line path trains faster and samples well with fewer network evaluations than VP paths on ImageNet.[^lipman2023] Figure 6 compares the two **marginal** flows on the toy problem, using the exact fields so that no learning error is involved. The result is instructive. Measured by excess length, the VP trajectories are **not** more curved than the straight-line ones; the marginal flow of the straight-line path bends just as much, for the crossing reasons of the rectified-flow section. The difference is the **speed profile**. Along the VP trajectories the speed varies by a factor of more than 1000 over $[0,1]$, against about 30 for the straight-line path, and with a uniform grid of Euler steps that is what costs accuracy.

![[Machine Learning/images/flow-matching-vp-vs-ot.svg]]
*Figure 6. The exact marginal flows of the VP path (linear $\beta$ from 0.1 to 20, as in Song et al.) and the straight-line path, for the same data and the same noise. (a, b) Trajectories: on average 17.8% (VP) and 18.9% (straight-line) longer than their chords. (c) SWD against the number of Euler steps on a uniform time grid (grey band: two independent data samples). With 4 steps the straight-line path reaches 0.12 and the VP path 0.27; with 8 steps, 0.11 against 0.14; from 16 steps on, both are within sampling noise. (d) Mean speed along the trajectories, on a log scale: between 0.002 and 4.6 for VP, between 0.1 and 3.1 for the straight-line path. The straight-line dip near $t=0.17$ is where trajectories stop heading towards the data mean (the direction of the field at $t=0$) and turn towards their own mode.*

So part of the advantage is a matter of coordinates. Any two Gaussian paths with the same range of signal-to-noise ratios $a_t/\sigma_t$ are related by rescaling $\mathbf{x}$ and reparameterizing time.[^kingma2021] Dividing the VP path by $a_t$, for example, gives $\mathbf{x}/a_t=\mathbf{x}_1+(\sigma_t/a_t)\mathbf{x}_0$, a straight line in the new variable with "time" $\sigma_t/a_t$; DDIM is exactly the Euler method in those coordinates. A well-designed time grid and a well-chosen solver recover much of the gap, as Karras et al. showed for diffusion.[^karras2022] What flow matching offers is a path that works well with the simplest choices, uniform times and Euler steps, and a framework in which the path is an explicit design decision. Making the **marginal** flow straight needs more than a good conditional path: that is what reflow and optimal-transport couplings do.

## Conditioning and classifier-free guidance

Conditional generation needs no new theory. To sample from $q(\mathbf{x}_1\mid y)$, for a class or a text prompt $y$, feed $y$ to the network, $\mathbf{v}_\theta(\mathbf{x},t,y)$, and train on (data, condition) pairs with the same loss. The companion repository's super-resolution model is an example, with the low-resolution image as $y$.

**Classifier-free guidance**[^ho2022] carries over unchanged. Train one network with the condition replaced by a null token $\varnothing$ for a random 10–20% of examples, and sample with

$$
\tilde{\mathbf{v}}_\lambda(\mathbf{x},t,y)=\mathbf{v}_\theta(\mathbf{x},t,\varnothing)+\lambda\big(\mathbf{v}_\theta(\mathbf{x},t,y)-\mathbf{v}_\theta(\mathbf{x},t,\varnothing)\big),
\tag{12}
$$

where $\lambda=0$ is unconditional, $\lambda=1$ conditional and $\lambda>1$ guided, the convention of the [[Denoising Diffusion Models|diffusion lecture]] (Ho and Salimans' guidance weight is $\lambda-1$). Why is this the same guidance? By (10), the velocity is an affine function of the score, $\mathbf{u}_t=A_t\mathbf{x}+C_t\nabla\log p_t$, with coefficients that depend only on the path. The conditional and unconditional fields share $A_t$ and $C_t$, so (12) is the velocity of the guided score $\nabla\log p_t(\mathbf{x})+\lambda\nabla\log p_t(y\mid\mathbf{x})$. As in diffusion, guided samples are not samples from $q(\mathbf{x}_1\mid y)$: guidance trades diversity for samples that are more typical of the condition.

![[Machine Learning/images/flow-matching-guidance.svg]]
*Figure 7. Classifier-free guidance on the eight Gaussians. The label $y=1$ is more likely for modes on the right: $p(y=1\mid\text{mode})$ is 0.9 for the rightmost mode, 0.1 for the leftmost, and in between for the others. One network was trained with 20% label dropout and sampled for $y=1$ (100 midpoint steps). "Right" and "left" are the fractions of samples in the three right-most and three left-most modes. Unconditional sampling gives 38% and 37% (true: 37.5% each). With $\lambda=1$ the model matches $q(\mathbf{x}_1\mid y=1)$: 62% and 13% (true: 61% and 13%). Guidance then concentrates the samples on the modes where $y=1$ is most likely: 83% and 3% at $\lambda=2$, 98% and 0.1% at $\lambda=4$. In this example the modes themselves barely sharpen; the mean distance of a sample to its mode centre stays between 0.19 and 0.20.*

## Beyond Gaussian noise: stochastic interpolants

Nothing above needs the source $p_0$ to be Gaussian. With a coupling $\pi(\mathbf{x}_0,\mathbf{x}_1)$ between **any** two distributions, for example low- and high-resolution images, or images from two domains, the same regression learns a flow between them. Albergo and Vanden-Eijnden's **stochastic interpolants** make this general.[^albergo2023]<sup>,</sup>[^albergo2023si] They interpolate

$$
\mathbf{x}_t=\boldsymbol\psi_t(\mathbf{x}_0,\mathbf{x}_1)+\gamma(t)\,\mathbf{z},
\qquad\mathbf{z}\sim\mathcal{N}(\mathbf{0},\mathbf{I}),
$$

with $\boldsymbol\psi_0(\mathbf{x}_0,\mathbf{x}_1)=\mathbf{x}_0$, $\boldsymbol\psi_1(\mathbf{x}_0,\mathbf{x}_1)=\mathbf{x}_1$ and an extra noise level $\gamma(t)$ that vanishes at both ends. They learn the velocity $\mathbf{b}(\mathbf{x},t)$ by the same kind of regression, and the score $\mathbf{s}(\mathbf{x},t)$ of $p_t$ by a denoising regression. Then a whole family of samplers has the right marginals: for any $\eta(t)\ge0$,

$$
d\mathbf{x}_t=\big[\mathbf{b}(\mathbf{x}_t,t)+\eta(t)\,\mathbf{s}(\mathbf{x}_t,t)\big]\,dt+\sqrt{2\eta(t)}\,d\mathbf{w}_t.
$$

$\eta=0$ is the flow-matching ODE; $\eta>0$ adds Langevin-like noise and correction, as in diffusion sampling. Flow matching and diffusion models are both special cases, which makes stochastic interpolants a convenient common language.

## Flow matching in practice

Flow matching has become the default training objective for large generative models.
- **Images.** Stable Diffusion 3 trains a transformer with rectified flow (the straight-line path, $\sigma_{\min}=0$). Its authors compared it with diffusion and other flow formulations at scale and found rectified flow best when the training times are drawn from a **logit-normal** distribution that concentrates on the middle of $[0,1]$, where the regression is hardest.[^esser2024]
- **Video, speech and actions.** Meta's Movie Gen video models,[^polyak2024] the Voicebox speech model[^le2023] and the $\pi_0$ robot policy[^black2024] are trained with flow matching. For robot policies the appeal is the same as for images: a flow can represent a multimodal distribution over actions, which the regression of [[Imitation Learning#Behaviour Cloning (BC)|behaviour cloning]] averages away.
- **Discrete data.** Text and other discrete data have no velocities, but the same construction works with [[Markov Chains#Continuous-time Markov chains|continuous-time Markov chains]] in place of ODEs: **discrete flow matching** learns the transition rates of a chain that carries a simple distribution to the data.[^gat2024]
- **A reference.** The "Flow Matching Guide and Code" by Lipman and coauthors covers all of this, including non-Euclidean and discrete versions, with a PyTorch library.[^lipman2024]

Practical lessons, from the papers and from the companion repository:
- Give the network a strong time embedding (scale $t$ before a sinusoidal embedding, and condition every block on it).
- Keep an exponential moving average of the weights for sampling.
- Use enough sampling steps: one or two Euler steps give averages, not samples, unless the flow has been straightened by reflow or distilled.
- Compare solvers at equal network evaluations. With very few evaluations, more Euler steps beat fewer higher-order ones; with more, the companion repository found midpoint the safest choice on images.

## Key takeaways

- A flow model generates samples by integrating $d\mathbf{x}/dt=\mathbf{v}_\theta(\mathbf{x},t)$ from noise ($t=0$) to data ($t=1$). The continuity equation $\partial_tp_t+\nabla\cdot(p_t\mathbf{u}_t)=0$ links a velocity field to the path of densities it produces.
- Flow matching regresses $\mathbf{v}_\theta$ onto the velocity of a path towards a single data point. The conditional loss has the same gradient as the intractable marginal one, because the marginal velocity is the conditional mean of the conditional velocities.
- With straight lines, the training target is $\mathbf{x}_1-\mathbf{x}_0$. The learned trajectories still curve, because training lines cross and ODE trajectories cannot.
- Reflow (training again on the flow's own noise–data pairs) and minibatch optimal-transport pairing both reduce crossings and straighten the flow, enabling few-step sampling and one-step distillation.
- Diffusion models are flow matching with a variance-preserving Gaussian path: velocity, noise prediction, data prediction and the score are interchangeable, and the probability flow ODE is the resulting flow. Classifier-free guidance carries over unchanged.

## Where this shows up next

- [[Normalizing Flows]]: the next lecture. A flow-matching model is a continuous normalizing flow, and its exact likelihood comes from the integrated divergence previewed above; normalizing flows build discrete-step versions with tractable Jacobian determinants.
- [[Denoising Diffusion Models]] and [[Score-Based Models]]: the same models seen through the VP and VE paths; their probability flow ODE is a flow-matching flow.
- [[Variational Autoencoders]] and [[Generative Adversarial Networks]]: one-step generators. Rectified flow with distillation is a way of getting a one-step generator with diffusion-like training.
- [[Multivariable Calculus#Densities under a change of variables|Densities under a change of variables]]: the discrete counterpart of the continuity equation and of the likelihood formula.
- [[Imitation Learning]]: flow-matching policies represent multimodal expert behaviour that a regression onto the mean action cannot.

<!-- References: footnote definitions below render as the numbered "References" list at the bottom of the page -->

[^chen2018]: Chen, R. T. Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D. (2018). *Neural Ordinary Differential Equations*. NeurIPS. [arXiv:1806.07366](https://arxiv.org/abs/1806.07366)
[^grathwohl2019]: Grathwohl, W., Chen, R. T. Q., Bettencourt, J., Sutskever, I., & Duvenaud, D. (2019). *FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models*. ICLR. [arXiv:1810.01367](https://arxiv.org/abs/1810.01367)
[^lipman2023]: Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., & Le, M. (2023). *Flow Matching for Generative Modeling*. ICLR. [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)
[^liu2023]: Liu, X., Gong, C., & Liu, Q. (2023). *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow*. ICLR. [arXiv:2209.03003](https://arxiv.org/abs/2209.03003)
[^albergo2023]: Albergo, M. S., & Vanden-Eijnden, E. (2023). *Building Normalizing Flows with Stochastic Interpolants*. ICLR. [arXiv:2209.15571](https://arxiv.org/abs/2209.15571)
[^tong2024]: Tong, A., Fatras, K., Malkin, N., Huguet, G., Zhang, Y., Rector-Brooks, J., et al. (2024). *Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport*. TMLR. [arXiv:2302.00482](https://arxiv.org/abs/2302.00482)
[^peyre2019]: Peyré, G., & Cuturi, M. (2019). *Computational Optimal Transport*. Foundations and Trends in Machine Learning, 11(5–6). [arXiv:1803.00567](https://arxiv.org/abs/1803.00567)
[^karras2022]: Karras, T., Aittala, M., Aila, T., & Laine, S. (2022). *Elucidating the Design Space of Diffusion-Based Generative Models*. NeurIPS. [arXiv:2206.00364](https://arxiv.org/abs/2206.00364)
[^liu2022]: Liu, Q. (2022). *Rectified Flow: A Marginal Preserving Approach to Optimal Transport*. [arXiv:2209.14577](https://arxiv.org/abs/2209.14577)
[^liu2024instaflow]: Liu, X., Zhang, X., Ma, J., Peng, J., & Liu, Q. (2024). *InstaFlow: One Step is Enough for High-Quality Diffusion-Based Text-to-Image Generation*. ICLR. [arXiv:2309.06380](https://arxiv.org/abs/2309.06380)
[^pooladian2023]: Pooladian, A.-A., Ben-Hamu, H., Domingo-Enrich, C., Amos, B., Lipman, Y., & Chen, R. T. Q. (2023). *Multisample Flow Matching: Straightening Flows with Minibatch Couplings*. ICML. [arXiv:2304.14772](https://arxiv.org/abs/2304.14772)
[^kingma2023]: Kingma, D. P., & Gao, R. (2023). *Understanding Diffusion Objectives as the ELBO with Simple Data Augmentation*. NeurIPS. [arXiv:2303.00848](https://arxiv.org/abs/2303.00848)
[^salimans2022]: Salimans, T., & Ho, J. (2022). *Progressive Distillation for Fast Sampling of Diffusion Models*. ICLR. [arXiv:2202.00512](https://arxiv.org/abs/2202.00512)
[^song2021sde]: Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2021). *Score-Based Generative Modeling through Stochastic Differential Equations*. ICLR. [arXiv:2011.13456](https://arxiv.org/abs/2011.13456)
[^kingma2021]: Kingma, D. P., Salimans, T., Poole, B., & Ho, J. (2021). *Variational Diffusion Models*. NeurIPS. [arXiv:2107.00630](https://arxiv.org/abs/2107.00630)
[^ho2022]: Ho, J., & Salimans, T. (2022). *Classifier-Free Diffusion Guidance*. NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications. [arXiv:2207.12598](https://arxiv.org/abs/2207.12598)
[^albergo2023si]: Albergo, M. S., Boffi, N. M., & Vanden-Eijnden, E. (2023). *Stochastic Interpolants: A Unifying Framework for Flows and Diffusions*. [arXiv:2303.08797](https://arxiv.org/abs/2303.08797)
[^esser2024]: Esser, P., Kulal, S., Blattmann, A., Entezari, R., Müller, J., Saini, H., et al. (2024). *Scaling Rectified Flow Transformers for High-Resolution Image Synthesis*. ICML. [arXiv:2403.03206](https://arxiv.org/abs/2403.03206)
[^polyak2024]: Polyak, A., Zohar, A., Brown, A., Tjandra, A., Sinha, A., Lee, A., et al. (2024). *Movie Gen: A Cast of Media Foundation Models*. [arXiv:2410.13720](https://arxiv.org/abs/2410.13720)
[^le2023]: Le, M., Vyas, A., Shi, B., Karrer, B., Sari, L., Moritz, R., et al. (2023). *Voicebox: Text-Guided Multilingual Universal Speech Generation at Scale*. NeurIPS. [arXiv:2306.15687](https://arxiv.org/abs/2306.15687)
[^black2024]: Black, K., Brown, N., Driess, D., Esmail, A., Equi, M., Finn, C., et al. (2024). *π0: A Vision-Language-Action Flow Model for General Robot Control*. [arXiv:2410.24164](https://arxiv.org/abs/2410.24164)
[^gat2024]: Gat, I., Remez, T., Shaul, N., Kreuk, F., Chen, R. T. Q., Synnaeve, G., et al. (2024). *Discrete Flow Matching*. NeurIPS. [arXiv:2407.15595](https://arxiv.org/abs/2407.15595)
[^lipman2024]: Lipman, Y., Havasi, M., Holderrieth, P., Shaul, N., Le, M., Karrer, B., et al. (2024). *Flow Matching Guide and Code*. [arXiv:2412.06264](https://arxiv.org/abs/2412.06264)
