---
title: Normalizing Flows
tags:
  - generative-models
  - normalizing-flows
  - continuous-normalizing-flows
  - density-estimation
date: 2026-11-13
draft: false
prev: "[[Flow Matching]]"
next: "[[Index (Machine Learning)]]"
---

> **Prerequisites:** [[Multivariable Calculus#Change of variables: the Jacobian determinant scales volume|change of variables and the Jacobian determinant]], in particular [[Multivariable Calculus#Densities under a change of variables|densities under a change of variables]] and [[Random Variables#Functions of a random variable|its one-dimensional version]]; [[Five Factorizations of a Matrix#A = LU: elimination|LU and determinants]]; [[Neural Networks#Computational graphs and reverse-mode autodiff|reverse-mode autodiff]]. From the generative series: [[Variational Autoencoders]] (latent-variable models and the direction of the KL) and [[Flow Matching]] (flows as ODEs).
>
> **TL;DR:**
> - If the generator $\mathbf{x}=f_\theta(\mathbf{z})$ is **invertible**, the density of $\mathbf{x}$ follows exactly from the density of $\mathbf{z}$ by the change-of-variables formula. Maximum likelihood becomes possible with no bound, no discriminator and no partition function.
> - The price is architectural. Every layer must be invertible, keep the dimension, and have a **cheap log-determinant**. Coupling layers (NICE, RealNVP, Glow) and autoregressive flows (MAF, IAF) get a triangular Jacobian, whose determinant is the product of its diagonal.
> - In the limit of infinitely many tiny layers, a flow becomes an ODE, a **continuous normalizing flow**. The log-determinant becomes an integral of a trace, which Hutchinson's estimator makes cheap. Training a CNF by maximum likelihood needs an ODE solve per step, and that cost is what [[Flow Matching|flow matching]] removes.
>
> **Notation in this lecture:** base variable $\mathbf{z}\sim p_Z=\mathcal{N}(\mathbf{0},\mathbf{I})$, data $\mathbf{x}\in\mathbb{R}^D$, the generative map $f=f_\theta:\mathbf{z}\mapsto\mathbf{x}$ and its inverse $g=f^{-1}$. $\mathbf{J}_f$ is the Jacobian of $f$. For continuous flows we use the convention of [[Flow Matching]]: $t=0$ is noise, $t=1$ is data, and the velocity is $\mathbf{v}_\theta(\mathbf{x},t)$. Neural ODE and FFJORD papers integrate in the opposite direction; only the sign bookkeeping changes.

## Motivation

The generative models so far each gave something up to avoid computing $p_\theta(\mathbf{x})$:
- A [[Variational Autoencoders|VAE]] has a latent variable but only a **lower bound** on $\log p_\theta(\mathbf{x})$, because the integral over $\mathbf{z}$ is intractable.
- A [[Generative Adversarial Networks|GAN]] samples in one pass but has **no density** at all.
- An [[Energy-Based Models|EBM]] has a density up to a **partition function** that nobody can compute.
- [[Score-Based Models]], [[Denoising Diffusion Models|diffusion]] and [[Flow Matching|flow matching]] learn a vector field and sample by running many steps.

A normalizing flow asks: what if the decoder of a latent-variable model were **invertible**? Then every $\mathbf{x}$ has exactly one $\mathbf{z}=g(\mathbf{x})$, the integral over $\mathbf{z}$ collapses to a single point, and the density is exact. Sampling is still one pass through $f$, and the latent code of any data point is one pass through $g$. The whole lecture is about how to build invertible maps that are both expressive and cheap to differentiate in the one way that matters: the log-determinant of the Jacobian.

The idea of turning data into Gaussian noise by a sequence of simple invertible steps goes back to Tabak and Turner.[^tabak2013] Rezende and Mohamed named it, used it to make VAE posteriors more flexible, and started the deep-learning line of work.[^rezende2015] Two reviews cover the whole area in more depth.[^papamakarios2021]<sup>,</sup>[^kobyzev2020]

## Change of variables: stretching and squashing probability

### One dimension

Push a standard Gaussian $z$ through a smooth, increasing function $x=f(z)$. Probability is conserved: the mass in a small interval $[z,z+dz]$ ends up in $[x,x+dx]$, so

$$
p_X(x)\,\lvert dx\rvert=p_Z(z)\,\lvert dz\rvert
\quad\Longrightarrow\quad
p_X(x)=\frac{p_Z(z)}{\lvert f'(z)\rvert}=p_Z\big(g(x)\big)\,\lvert g'(x)\rvert .
$$

Where $f$ **stretches** ($\lvert f'\rvert>1$), the same mass is spread over a longer interval and the density drops. Where $f$ **squashes** ($\lvert f'\rvert<1$), the density rises. Figure 1 shows this for $f(z)=z+1.5\tanh(2z)$. Its slope is $f'(0)=4$ at the centre, so the peak of the Gaussian is stretched out and $p_X(0)=0.399/4=0.100$. Away from the centre the slope returns to 1. A unimodal Gaussian has become bimodal, and every band of mass $\tfrac18$ is still a band of mass $\tfrac18$.

![[Machine Learning/images/normalizing-flows-1d-change-of-variables.svg]]
*Figure 1. (a) A standard Gaussian cut into eight bands of equal probability. (b) The increasing map $f(z)=z+1.5\tanh(2z)$ sends each band to an interval. (c) Each band keeps its mass but changes its width, so the density of $x$ is the base density divided by the slope: low where $f$ is steep, near $x=0$, and a bump on either side.*

### Many dimensions

In $D$ dimensions the slope becomes the Jacobian, and the factor by which $f$ scales small volumes is $\lvert\det\mathbf{J}_f\rvert$ (the [[Multivariable Calculus#Densities under a change of variables|prerequisite section]] derives this). For an invertible, differentiable $f$ with differentiable inverse $g$,

$$
\begin{aligned}
\log p_X(\mathbf{x})
&=\log p_Z\big(g(\mathbf{x})\big)+\log\big\lvert\det\mathbf{J}_g(\mathbf{x})\big\rvert\\
&=\log p_Z(\mathbf{z})-\log\big\lvert\det\mathbf{J}_f(\mathbf{z})\big\rvert,
\qquad \mathbf{z}=g(\mathbf{x}).
\end{aligned}
\tag{1}
$$

The two lines agree because $\mathbf{J}_g(\mathbf{x})=\mathbf{J}_f(\mathbf{z})^{-1}$ (the inverse function theorem) and $\det(\mathbf{A}^{-1})=1/\det\mathbf{A}$.

Read (1) as a tug of war when we maximize it over the training data:
- The first term wants $g$ to send every data point to where the Gaussian is high, near the origin.
- The second term rewards $g$ for **expanding** volume around the data. Without it, $g$ would map everything to $\mathbf{0}$ and claim infinite likelihood. The log-determinant is the bookkeeping that keeps total probability equal to one.

> [!note] What "normalizing" means
> Run in the inverse direction, $g$ turns the data distribution into a standard normal: it *normalizes* the data. Run forwards, $f$ lets probability mass *flow* from the simple base to the complex data distribution. Both names are in "normalizing flow".

### Training and sampling

Given data $\mathcal{D}=\{\mathbf{x}_i\}_{i=1}^N$, fit $\theta$ by **exact maximum likelihood**,

$$
\max_\theta\ \frac1N\sum_{i=1}^N\Big[\log p_Z\big(g_\theta(\mathbf{x}_i)\big)+\log\big\lvert\det\mathbf{J}_{g_\theta}(\mathbf{x}_i)\big\rvert\Big],
$$

which minimizes the **forward** KL divergence $D_{\mathrm{KL}}(p_{\text{data}}\,\|\,p_\theta)$, the mass-covering direction discussed in [[Variational Autoencoders#Which way does the KL point?|the VAE lecture]]. Sampling is $\mathbf{z}\sim\mathcal{N}(\mathbf{0},\mathbf{I})$, $\mathbf{x}=f_\theta(\mathbf{z})$.

Notice which direction each task uses. **Density evaluation** on data needs $g$ and its log-determinant. **Sampling** needs $f$. A flow used as a variational posterior, as in Rezende and Mohamed's work, needs samples together with their own densities, which only uses $f$ and $\log\lvert\det\mathbf{J}_f\rvert$, the second line of (1). Different flows make different directions cheap, and that choice shapes the whole design space.

### Composing flows

Invertible maps compose. If $f=f_K\circ\cdots\circ f_1$ with intermediate states $\mathbf{h}_0=\mathbf{z}$ and $\mathbf{h}_k=f_k(\mathbf{h}_{k-1})$, the [[Multivariable Calculus#The chain rule is a product of Jacobians|chain rule]] multiplies the Jacobians, and determinants of products multiply, so

$$
\log p_X(\mathbf{x})=\log p_Z(\mathbf{z})-\sum_{k=1}^{K}\log\big\lvert\det\mathbf{J}_{f_k}(\mathbf{h}_{k-1})\big\rvert .
\tag{2}
$$

Each layer can be simple; the composition is expressive. That is the plan for the rest of the lecture.

### The cost of a determinant, and three ways around it

A layer must be (i) invertible, (ii) cheap to run in the direction we need, (iii) cheap to take the log-determinant of, and (iv) expressive. Requirement (iii) is the hard one. A general $D\times D$ determinant costs $O(D^3)$ by [[Five Factorizations of a Matrix#A = LU: elimination|LU factorization]], and the Jacobian itself needs $D$ backward passes to form. For a $32\times32$ colour image, $D=3072$, and that is per layer, per example, per step. Three structures make it cheap (Figure 2):
1. **Triangular Jacobians.** The determinant of a triangular matrix is the product of its diagonal, $O(D)$. Coupling layers and autoregressive flows are built this way.
2. **Identity plus low rank.** By the matrix determinant lemma, $\det(\mathbf{I}+\mathbf{u}\mathbf{w}^\top)=1+\mathbf{w}^\top\mathbf{u}$, also $O(D)$. Planar flows use it.
3. **Special structure.** A $1\times1$ convolution applies one small $C\times C$ matrix at every pixel, so its determinant is that of the small matrix (Glow). An ODE needs only a **trace**, which can be estimated without forming the Jacobian at all (continuous flows).

![[Machine Learning/images/normalizing-flows-jacobian-structure.svg]]
*Figure 2. Three Jacobian structures. (a) A general dense Jacobian needs an $O(D^3)$ factorization. (b) A lower-triangular Jacobian has determinant equal to the product of its diagonal. (c) The identity (orange) plus a rank-one matrix (blue) has determinant $1+\mathbf{w}^\top\mathbf{u}$.*

> [!warning] A flow cannot change topology
> An invertible continuous map is a homeomorphism: it can bend and stretch space but cannot tear it. A Gaussian is one connected blob, so a flow can never map it onto two *disconnected* pieces exactly. It must leave a thin bridge of low density between them, which is visible in every trained model of Figure 7. Deep flows make the bridge very thin, at the cost of very large Jacobians near it. Dupont et al. discuss the same limitation for ODE flows and fix it by adding extra dimensions.[^dupont2019]

## Planar and radial flows

The first deep flows were built for variational inference.[^rezende2015] A **planar flow** adds one bent ridge to the identity:

$$
f(\mathbf{z})=\mathbf{z}+\mathbf{u}\,h(\mathbf{w}^\top\mathbf{z}+b),
\qquad h=\tanh .
\tag{3}
$$

It moves every point along the fixed direction $\mathbf{u}$, by an amount that depends on its signed position relative to the hyperplane $\mathbf{w}^\top\mathbf{z}+b=0$: a smooth step, nearly constant far from the hyperplane and steepest across it. Its Jacobian is $\mathbf{I}+\mathbf{u}\,\boldsymbol\psi(\mathbf{z})^\top$ with $\boldsymbol\psi(\mathbf{z})=h'(\mathbf{w}^\top\mathbf{z}+b)\,\mathbf{w}$, so the matrix determinant lemma gives

$$
\log\lvert\det\mathbf{J}_f(\mathbf{z})\rvert=\log\big\lvert 1+\mathbf{u}^\top\boldsymbol\psi(\mathbf{z})\big\rvert ,
$$

in $O(D)$. With $h=\tanh$, the map is invertible when $\mathbf{w}^\top\mathbf{u}\ge-1$; in practice $\mathbf{u}$ is reparameterized to enforce it. A **radial flow** instead contracts or expands around a centre $\mathbf{z}_0$: $f(\mathbf{z})=\mathbf{z}+\beta h(\alpha,r)(\mathbf{z}-\mathbf{z}_0)$ with $r=\lVert\mathbf{z}-\mathbf{z}_0\rVert$ and $h=1/(\alpha+r)$. Its Jacobian is identity plus rank one again, and $\det\mathbf{J}_f=\big[1+\beta h\big]^{D-1}\big[1+\beta h+\beta h'(r)\,r\big]$, where $h'$ is the derivative with respect to $r$; it is invertible when $\beta\ge-\alpha$.

Two limitations follow from the construction:
- There is **no closed-form inverse**. We can sample and score our own samples, but we cannot evaluate the density of a given data point cheaply. Planar and radial flows suit variational inference, not density estimation.
- Each layer bends space along a single direction, so many layers are needed in high dimension.

Figure 3 trains planar flows the way they are used in variational inference: towards an unnormalized target $p^*\propto e^{-U}$, by minimizing the **reverse** KL divergence

$$
D_{\mathrm{KL}}(q\,\|\,p^*)=\mathbb{E}_{\mathbf{x}\sim q}\big[\log q(\mathbf{x})+U(\mathbf{x})\big]+\log Z .
$$

The expectation needs only samples from the flow and their densities, from the second line of (1); the unknown $\log Z$ is a constant. Two observations:
- **Capacity.** On a curved ridge, two layers capture the bend only roughly (KL 0.21 nats), while eight layers fit it closely (0.02 nats).
- **Mode seeking.** The second target has two equal, separated lobes. Reverse KL punishes $q$ for putting mass where $p^*$ has none, but not for missing mass ([[Variational Autoencoders#Which way does the KL point?|the VAE lecture]] explains why). Even 32 layers trained directly settle on one lobe, with KL $0.70\approx\ln2$, exactly the price of covering half of the mass. **Annealing**, multiplying $U$ by a factor that grows from 0.01 to 1 over the first half of training (as Rezende and Mohamed did), lets the flow spread over both lobes while the target is still nearly flat, before the low-density gap between them becomes a barrier: KL 0.05 nats.

![[Machine Learning/images/normalizing-flows-planar.png]]
*Figure 3. Planar flows fitted by reverse KL to two unnormalized targets; each panel is a histogram of flow samples, labelled with the KL divergence in nats. Top: a sine-shaped ridge, fitted roughly by 2 layers and closely by 8. Bottom: two separated lobes. Trained directly, a 32-layer flow covers one lobe (KL near $\ln 2$); with an annealed target it covers both.*

## Coupling layers

Coupling layers give a triangular Jacobian **and** a closed-form inverse, with an arbitrary neural network inside.

### NICE: additive coupling

Split the coordinates into two parts, $\mathbf{z}=(\mathbf{z}_A,\mathbf{z}_B)$. Leave the first part alone and shift the second by any function of the first:[^dinh2014]

$$
\mathbf{x}_A=\mathbf{z}_A,\qquad \mathbf{x}_B=\mathbf{z}_B+\mathbf{t}(\mathbf{z}_A).
$$

The inverse is immediate, $\mathbf{z}_B=\mathbf{x}_B-\mathbf{t}(\mathbf{x}_A)$, because $\mathbf{x}_A=\mathbf{z}_A$ lets us recompute $\mathbf{t}$. The network $\mathbf{t}$ is never inverted, so it can be any architecture. The Jacobian is

$$
\mathbf{J}_f=\begin{bmatrix}\mathbf{I}&\mathbf{0}\\ \partial\mathbf{t}/\partial\mathbf{z}_A&\mathbf{I}\end{bmatrix},
\qquad \det\mathbf{J}_f=1 .
$$

An additive coupling is **volume-preserving**: it rearranges probability mass but cannot concentrate it. Alternating which half is updated lets every coordinate change. NICE adds one diagonal scaling layer at the end so that the model can change volume at all.

The volume-preserving limit is easy to see in numbers. If every layer has $\lvert\det\rvert=1$, then $p_X(\mathbf{x})=p_Z(g(\mathbf{x}))\le p_Z(\mathbf{0})$. In two dimensions that caps every log-density at $-\log 2\pi=-1.84$ nats. On our two-moons data (Figure 7), eight additive couplings with no scaling layer reach a test log-likelihood of $-1.99$ nats, close to that ceiling; NICE's final scaling lifts it to $-0.98$.

### RealNVP: affine coupling

RealNVP (real-valued non-volume preserving) also scales the second part:[^dinh2017]

$$
\mathbf{x}_A=\mathbf{z}_A,\qquad
\mathbf{x}_B=\mathbf{z}_B\odot\exp\big(\mathbf{s}(\mathbf{z}_A)\big)+\mathbf{t}(\mathbf{z}_A).
\tag{4}
$$

The inverse is still explicit, $\mathbf{z}_B=\big(\mathbf{x}_B-\mathbf{t}(\mathbf{x}_A)\big)\odot\exp\big(-\mathbf{s}(\mathbf{x}_A)\big)$. The Jacobian is lower block-triangular with $\operatorname{diag}(e^{\mathbf{s}})$ in the lower-right block, so

$$
\log\lvert\det\mathbf{J}_f(\mathbf{z})\rvert=\sum_j s_j(\mathbf{z}_A).
$$

The awkward off-diagonal block $\partial\mathbf{x}_B/\partial\mathbf{z}_A$ involves the derivatives of the networks, but it never enters the determinant (Figure 4).

![[Machine Learning/images/normalizing-flows-coupling-layer.svg]]
*Figure 4. An affine coupling layer. One half passes through unchanged and conditions a network that outputs a log-scale $\mathbf{s}$ and a shift $\mathbf{t}$ for the other half. Inverting it only requires recomputing $\mathbf{s}$ and $\mathbf{t}$ from the unchanged half, and the log-determinant is the sum of the log-scales.*

> [!example]- Code: an affine coupling layer and a RealNVP-style flow (PyTorch, tested)
> ```python
> import math, torch, torch.nn as nn
>
> class AffineCoupling(nn.Module):
>     """x_A = z_A,  x_B = z_B * exp(s(z_A)) + t(z_A)."""
>     def __init__(self, d_a, d_b, hidden=64):
>         super().__init__()
>         self.d_a = d_a
>         self.net = nn.Sequential(nn.Linear(d_a, hidden), nn.Tanh(),
>                                  nn.Linear(hidden, 2 * d_b))
>
>     def _s_t(self, a):
>         s, t = self.net(a).chunk(2, dim=-1)
>         return torch.tanh(s), t           # bounded log-scale, for stability
>
>     def forward(self, z):                 # z -> x, and log|det J_f(z)|
>         a, b = z[:, :self.d_a], z[:, self.d_a:]
>         s, t = self._s_t(a)
>         return torch.cat([a, b * s.exp() + t], -1), s.sum(-1)
>
>     def inverse(self, x):                 # x -> z, and log|det J_g(x)|
>         a, b = x[:, :self.d_a], x[:, self.d_a:]
>         s, t = self._s_t(a)               # a = z_A is unchanged: recompute s, t
>         return torch.cat([a, (b - t) * (-s).exp()], -1), -s.sum(-1)
>
> class Flow(nn.Module):
>     def __init__(self, D, n_layers=8):
>         super().__init__()
>         self.D = D
>         self.layers = nn.ModuleList(AffineCoupling(D // 2, D - D // 2)
>                                     for _ in range(n_layers))
>
>     def log_prob(self, x):                # change of variables, layer by layer
>         log_det = torch.zeros(x.shape[0])
>         for layer in reversed(self.layers):
>             x = x.flip(-1)                # undo the swap (|det| = 1)
>             x, ld = layer.inverse(x)
>             log_det = log_det + ld
>         log_pz = -0.5 * (x**2).sum(-1) - 0.5 * self.D * math.log(2 * math.pi)
>         return log_pz + log_det
>
>     def sample(self, n):
>         z = torch.randn(n, self.D)
>         for layer in self.layers:
>             z, _ = layer(z)
>             z = z.flip(-1)                # swap halves so every coordinate gets updated
>         return z
>
> from sklearn.datasets import make_moons
> data = torch.tensor(make_moons(5000, noise=0.05)[0], dtype=torch.float32)
>
> flow = Flow(D=2)
> opt = torch.optim.Adam(flow.parameters(), lr=1e-3)
> for step in range(3000):
>     x = data[torch.randint(len(data), (256,))]
>     loss = -flow.log_prob(x).mean()       # exact negative log-likelihood
>     opt.zero_grad(); loss.backward(); opt.step()
> ```
> Checks we ran: `inverse(forward(z))` returns `z` to $6\times10^{-8}$; the returned log-determinant equals `slogdet` of the autograd Jacobian, which is block-triangular; after training, the density integrates to 1.000 over a grid and the test log-likelihood is $-0.40$ nats per point (on raw `make_moons` coordinates, so not comparable with Figure 7, which uses rescaled data).
>
> One trap we fell into while building the demos: if the log-scale is written $\mathbf{s}=c\tanh(\text{net})$ with a learnable $c$ **initialized to zero** and a zero-initialized last layer, neither factor ever receives a gradient. The layer silently stays additive, and the model is stuck near the volume-preserving ceiling above. Initialize $c$ to 1.

Figure 5 follows a trained stack of eight affine couplings, layer by layer, as it carries the base Gaussian onto two moons.

![[Machine Learning/images/normalizing-flows-realnvp-layers.png]]
*Figure 5. A trained 8-layer affine coupling flow on two moons, run forwards. Each point is coloured by the angle of its base sample $\mathbf{z}$. Every pair of layers updates both coordinates once; the Gaussian is sheared, folded and squeezed until, after eight layers, it lies on the two moons. The thin trail between the moons is the bridge that a homeomorphism must leave.*

### Masks, squeezing and the multi-scale architecture

For images, the split into $A$ and $B$ is a **mask**. RealNVP alternates a checkerboard mask (pixels of one colour condition the other) with a channel mask (half the channels condition the rest).[^dinh2017] Between them, a **squeeze** reshapes each $2\times2\times C$ block into a $1\times1\times4C$ one, trading spatial resolution for channels so that couplings can mix nearby pixels.

The **multi-scale architecture** then factors out half of the variables after each scale and sends them straight to the Gaussian, while the other half continues through more layers. Early-exiting variables model fine detail, and the deepest ones model global structure. It also saves computation, since later layers act on fewer dimensions.

### Glow: activation normalization and invertible 1×1 convolutions

Glow replaces two parts of RealNVP's recipe with learned, invertible layers.[^kingma2018] Each step of flow is **actnorm**, then an **invertible $1\times1$ convolution**, then an affine coupling.
- **Actnorm** is a per-channel affine map $\mathbf{x}=\mathbf{s}\odot\mathbf{z}+\mathbf{b}$, initialized from the first minibatch so that every channel has zero mean and unit variance, and then trained freely. On an $H\times W$ feature map, $\log\lvert\det\rvert=HW\sum_c\log\lvert s_c\rvert$. It plays the role of batch normalization without depending on the batch.
- **Invertible 1×1 convolution.** RealNVP fixes the permutation that decides which channels condition which. Glow learns it: a $C\times C$ matrix $\mathbf{W}$ applied at every pixel, with $\log\lvert\det\rvert=HW\log\lvert\det\mathbf{W}\rvert$. To make the determinant $O(C)$ rather than $O(C^3)$, $\mathbf{W}$ is parameterized through its LU factorization,

$$
\mathbf{W}=\mathbf{P}\,\mathbf{L}\,\big(\mathbf{U}+\operatorname{diag}(\mathbf{s})\big),
\qquad
\log\lvert\det\mathbf{W}\rvert=\sum_c\log\lvert s_c\rvert,
$$

with $\mathbf{P}$ a fixed permutation, $\mathbf{L}$ unit lower-triangular and $\mathbf{U}$ strictly upper-triangular. This is the [[Five Factorizations of a Matrix#A = LU: elimination|LU factorization]], used here as a parameterization: the determinant is the product of the pivots.

Glow produced sharp $256\times256$ faces, meaningful latent interpolations, and a useful trick: sampling from a **reduced temperature**, $\mathbf{z}\sim\mathcal{N}(\mathbf{0},T^2\mathbf{I})$ with $T\approx0.7$, trades diversity for quality. On CIFAR-10 it reported 3.35 bits per dimension against RealNVP's 3.49 (bits per dimension are defined [[#Discrete data: dequantization and bits per dimension|below]]).

## Autoregressive flows

### An autoregressive model is a flow

Any density factorizes by the chain rule, $p(\mathbf{x})=\prod_{i=1}^Dp(x_i\mid\mathbf{x}_{<i})$. Make each conditional a Gaussian whose mean and log-scale are functions of the earlier coordinates. Sampling it reads

$$
x_i=z_i\,\exp\big(\alpha_i(\mathbf{x}_{<i})\big)+\mu_i(\mathbf{x}_{<i}),
\qquad z_i\sim\mathcal{N}(0,1).
\tag{5}
$$

That is an invertible map from $\mathbf{z}$ to $\mathbf{x}$: given $\mathbf{x}$, every $z_i=\big(x_i-\mu_i(\mathbf{x}_{<i})\big)e^{-\alpha_i(\mathbf{x}_{<i})}$. Since $x_i$ depends only on $z_1,\dots,z_i$, the Jacobian $\partial\mathbf{x}/\partial\mathbf{z}$ is lower triangular with diagonal $e^{\alpha_i}$, and

$$
\log\lvert\det\mathbf{J}_g(\mathbf{x})\rvert=-\sum_{i=1}^D\alpha_i(\mathbf{x}_{<i}).
$$

So a Gaussian autoregressive model **is** a one-layer normalizing flow. A linear version, with $\boldsymbol\mu$ linear in $\mathbf{x}_{<i}$ and constant $\alpha_i$, is exactly the Cholesky construction of a [[Joint Distributions#The multivariate Gaussian|multivariate Gaussian]], $\mathbf{x}=\boldsymbol\mu+\mathbf{L}\mathbf{z}$. One Gaussian layer is still limited, since each conditional is unimodal. Stacking several layers, with the order of the variables reversed or permuted between them, removes that limit.

All $\mu_i$ and $\alpha_i$ come from one network pass with **MADE**, an autoencoder whose weight matrices are masked so that output $i$ only sees inputs $1,\dots,i-1$.[^germain2015]

### MAF and IAF: two directions of one idea

The conditioner can read either side of the map, and the choice decides which operation is fast (Figure 6).
- **Masked autoregressive flow (MAF)** conditions on $\mathbf{x}$, as in (5).[^papamakarios2017] Evaluating the density of a data point is **one pass**: all of $\mathbf{x}$ is known, so all $\mu_i,\alpha_i$ come out at once. Sampling is **sequential**: $x_2$ cannot be computed until $x_1$ is, so $D$ passes.
- **Inverse autoregressive flow (IAF)** conditions on $\mathbf{z}$: $x_i=z_i\,e^{\alpha_i(\mathbf{z}_{<i})}+\mu_i(\mathbf{z}_{<i})$.[^kingma2016] Now sampling is **one pass**, and so is the density of the model's own samples, because their $\mathbf{z}$ is known. The density of an arbitrary new $\mathbf{x}$ needs $D$ sequential passes.

In our implementation, with $D=64$, five MADE layers and a batch of 256, MAF evaluated densities in 6.3 ms and drew samples in 273 ms, a factor of 43. IAF is the same network run the other way, so its two numbers swap.

![[Machine Learning/images/normalizing-flows-maf-iaf.svg]]
*Figure 6. MAF and IAF use the same masked network (MADE) on opposite sides of the map. (a) MAF reads $\mathbf{x}$, so the density of data is one pass and sampling is sequential. (b) IAF reads $\mathbf{z}$, so sampling is one pass and evaluating the density of an external point is sequential.*

This makes each one the natural choice for a different job:
- **MAF for density estimation**, where we evaluate $p(\mathbf{x})$ on data many times and rarely sample.
- **IAF for variational inference**, where a VAE samples $\mathbf{z}\sim q(\mathbf{z}\mid\mathbf{x})$ and needs $\log q$ only of its own samples. Kingma et al. introduced IAF exactly as a flexible VAE posterior.[^kingma2016]
- **Both, by distillation.** Parallel WaveNet trains a fast IAF student to match a slow autoregressive WaveNet teacher, getting real-time speech synthesis.[^vandenoord2018]

> [!note] Coupling layers are a special case
> An affine coupling layer (4) is an autoregressive layer in which the variables of $B$ depend only on $A$, not on each other. It gives up expressiveness within $B$ in exchange for being one pass in **both** directions. Coupling flows (RealNVP, Glow) are the choice when both fast sampling and fast density evaluation are needed.

Autoregressive models without a flow interpretation (PixelCNN, WaveNet, GPT-style language models) use the same chain-rule factorization with categorical rather than Gaussian conditionals. They have exact likelihoods too, but no continuous latent space and sequential sampling.

### Our demo: four flows and two baselines

We trained each model on 5000 points from a rescaled two-moons dataset and measured the average test log-likelihood on 20000 new points (higher is better; standard errors about 0.01 nats):

| Model | Test log-likelihood (nats) |
|---|---|
| Single Gaussian (fitted mean and covariance) | $-2.35$ |
| 8 additive couplings, no scaling (volume-preserving) | $-1.99$ |
| NICE: 8 additive couplings + diagonal scaling | $-0.98$ |
| CNF (FFJORD-style, exact trace, RK4) | $-0.91$ |
| RealNVP: 8 affine couplings | $-0.88$ |
| MAF: 8 MADE layers | $-0.86$ |
| Gaussian kernel density estimate (bandwidth 0.03 by cross-validation) | $-0.80$ |

![[Machine Learning/images/normalizing-flows-density-comparison.png]]
*Figure 7. Learned densities on two moons, with test log-likelihoods. All three flows capture the moons. The affine coupling flow and MAF leave visible ridges and a faint bridge between the moons; the CNF is smoother but slightly blurrier. The scale is shared with the data histogram.*

Two honest remarks:
- The CNF had a quarter of the parameters (8.7k against about 37k) and was trained for fewer steps, so this is not a ranking of methods.
- In two dimensions with 5000 points, a [[Non-parametric Models#Kernel density estimation|kernel density estimate]] beats every flow. Flows earn their keep in high dimension, where KDE's error rate degrades badly with $D$ (the [[Non-parametric Models#When trees stop helping: the curse of dimensionality|curse of dimensionality]]) and where we also want fast sampling and a latent space.

## More expressive layers

### Spline flows

An affine coupling or autoregressive layer transforms each coordinate by $z\mapsto e^{s}z+t$, a straight line. Any **monotone** one-dimensional function would do, as long as it is invertible and its derivative is cheap. Neural spline flows use **monotone rational-quadratic splines**: the conditioner outputs the widths, heights and knot derivatives of $K$ bins on an interval $[-B,B]$, with the identity outside.[^durkan2019] Each piece is a ratio of quadratics, so the derivative is closed-form and the inverse is the root of a quadratic. One spline layer can turn a unimodal coordinate into a multimodal one, which an affine layer cannot, so far fewer layers are needed.

### Residual flows

A residual block $f(\mathbf{z})=\mathbf{z}+r(\mathbf{z})$ is invertible whenever $r$ is a contraction, $\operatorname{Lip}(r)<1$. Then $\mathbf{z}$ is the unique fixed point of $\mathbf{z}\mapsto\mathbf{x}-r(\mathbf{z})$, which iteration finds by the Banach fixed-point theorem (the same argument as for the [[Dynamic Programming#The foundation: Bellman operators are contractions|Bellman operators]]). Spectral normalization keeps the Lipschitz constant below 1. The log-determinant has a power series,

$$
\log\det\big(\mathbf{I}+\mathbf{J}_r\big)=\sum_{k=1}^{\infty}\frac{(-1)^{k+1}}{k}\operatorname{tr}\big(\mathbf{J}_r^k\big),
$$

and each trace is estimated with Hutchinson's trick ([[#Hutchinson's trace estimator|below]]). Invertible ResNets truncate the series, which biases the estimate;[^behrmann2019] Residual Flows use a randomized "Russian roulette" truncation that is unbiased.[^chen2019resflow] These flows keep an unconstrained, free-form Jacobian, like the continuous flows that follow.

## Discrete data: dequantization and bits per dimension

Images are stored as integers, $x_i\in\{0,\dots,255\}$. A continuous density fitted to integer data can put arbitrarily tall spikes on the grid points and drive the log-likelihood to $+\infty$ without learning anything. The standard fix is **dequantization**: add uniform noise, $\mathbf{y}=\mathbf{x}+\mathbf{u}$ with $\mathbf{u}\sim\mathcal{U}[0,1)^D$, and fit the continuous model to $\mathbf{y}$. The discrete model $P(\mathbf{x})=\int_{[0,1)^D}p(\mathbf{x}+\mathbf{u})\,d\mathbf{u}$ is a proper distribution on the integers, and by Jensen's inequality

$$
\log P(\mathbf{x})\ \ge\ \mathbb{E}_{\mathbf{u}}\big[\log p(\mathbf{x}+\mathbf{u})\big],
$$

so the continuous log-likelihood is a lower bound on the discrete one and can no longer diverge.[^theis2016] Flow++ replaces the uniform noise by a learned noise distribution $q(\mathbf{u}\mid\mathbf{x})$, a variational bound that is tighter.[^ho2019]

Likelihoods of images are reported in **bits per dimension**, $-\log_2P(\mathbf{x})/D$: the average number of bits an ideal compressor built on the model would need per subpixel. If the dequantized data are rescaled to $[0,1)^D$, the rescaling contributes $\log_2256=8$ bits per dimension, so

$$
\text{bits/dim}=\frac{-\log p(\mathbf{y}/256)}{D\ln 2}+8,
$$

where the log-density is in nats. A uniform model on the cube scores exactly 8; every published number below that is compression won by modelling.

> [!warning] Likelihood is not sample quality
> High likelihood and good samples can disagree in both directions.[^theis2016] Worse, flows trained on one image dataset can assign **higher** likelihood to images from a different, simpler one (a model trained on CIFAR-10 scoring SVHN digits higher), so raw likelihood is a poor detector of out-of-distribution inputs.[^nalisnick2019]

## Continuous normalizing flows

### From residual layers to an ODE

A residual layer $\mathbf{h}_{k+1}=\mathbf{h}_k+\varepsilon\,\mathbf{v}(\mathbf{h}_k,t_k)$ is one Euler step of an ODE. Taking more layers with smaller steps gives a **neural ODE**:[^chen2018]

$$
\frac{d\mathbf{x}(t)}{dt}=\mathbf{v}_\theta\big(\mathbf{x}(t),t\big),
\qquad \mathbf{x}(0)=\mathbf{z}\sim\mathcal{N}(\mathbf{0},\mathbf{I}),
\qquad \mathbf{x}=\mathbf{x}(1).
$$

The map from $\mathbf{x}(0)$ to $\mathbf{x}(1)$ is automatically invertible: solve the same ODE backwards in time. If $\mathbf{v}_\theta$ is Lipschitz in $\mathbf{x}$, trajectories are unique and cannot cross. There is **no architectural constraint** on $\mathbf{v}_\theta$ at all: no masks, no triangular structure, any network with $D$ inputs (plus time) and $D$ outputs. This is the same object as in [[Flow Matching]]; the two lectures differ in how $\mathbf{v}_\theta$ is trained.

### The instantaneous change of variables

What replaces the log-determinant? Over a tiny time step, the flow map is $\mathbf{x}\mapsto\mathbf{x}+\varepsilon\mathbf{v}$, with Jacobian $\mathbf{I}+\varepsilon\,\partial\mathbf{v}/\partial\mathbf{x}$, and to first order $\det(\mathbf{I}+\varepsilon\mathbf{A})\approx1+\varepsilon\operatorname{tr}\mathbf{A}$. The determinant becomes a **trace**. Following a particle along its path,

$$
\frac{d}{dt}\log p_t\big(\mathbf{x}(t)\big)=-\operatorname{tr}\!\left(\frac{\partial\mathbf{v}_\theta}{\partial\mathbf{x}}\big(\mathbf{x}(t),t\big)\right)=-\nabla\!\cdot\mathbf{v}_\theta ,
\tag{6}
$$

and integrating from noise to data,

$$
\log p_1(\mathbf{x})=\log p_Z\big(\mathbf{x}(0)\big)-\int_0^1\nabla\!\cdot\mathbf{v}_\theta\big(\mathbf{x}(t),t\big)\,dt .
$$

The divergence is the rate at which a small volume around the particle grows. Where the flow spreads particles apart, their density falls at exactly that rate; where it squeezes them together, it rises. It is the continuous version of Figure 1. To evaluate $\log p_1(\mathbf{x})$ for a data point, solve the ODE backwards from $\mathbf{x}(1)=\mathbf{x}$ to $t=0$, carrying the running integral of the divergence along as an extra state variable.

> [!proof]- Proof of the instantaneous change of variables (6)
> **Route 1: through the discrete formula.** Fix a trajectory and a small $\varepsilon>0$. The map $\mathbf{x}(t)\mapsto\mathbf{x}(t+\varepsilon)$ is invertible and equals $\mathbf{x}+\varepsilon\mathbf{v}(\mathbf{x},t)+O(\varepsilon^2)$, with Jacobian $\mathbf{I}+\varepsilon\mathbf{A}+O(\varepsilon^2)$ where $\mathbf{A}=\partial\mathbf{v}/\partial\mathbf{x}$. By (1),
>
> $$
> \log p_{t+\varepsilon}\big(\mathbf{x}(t+\varepsilon)\big)=\log p_t\big(\mathbf{x}(t)\big)-\log\big\lvert\det(\mathbf{I}+\varepsilon\mathbf{A}+O(\varepsilon^2))\big\rvert .
> $$
>
> Expanding the determinant, the only terms of first order in $\varepsilon$ come from the diagonal, so $\det(\mathbf{I}+\varepsilon\mathbf{A})=1+\varepsilon\operatorname{tr}\mathbf{A}+O(\varepsilon^2)$, the same expansion used for [[Multivariable Calculus#Matrix variables: the trace and the log-determinant|the gradient of the log-determinant]]. Taking logs, $\log(1+\varepsilon\operatorname{tr}\mathbf{A})=\varepsilon\operatorname{tr}\mathbf{A}+O(\varepsilon^2)$. Subtract, divide by $\varepsilon$ and let $\varepsilon\to0$ to get (6).
>
> **Route 2: through the continuity equation.** Conservation of probability says $\partial_tp_t+\nabla\!\cdot(p_t\mathbf{v})=0$, the equation that [[Flow Matching]] builds its probability paths on. Expand the divergence of the product, $\nabla\!\cdot(p_t\mathbf{v})=\nabla p_t\cdot\mathbf{v}+p_t\nabla\!\cdot\mathbf{v}$. The derivative of $p_t$ along a trajectory is
>
> $$
> \frac{d}{dt}p_t\big(\mathbf{x}(t)\big)=\partial_tp_t+\nabla p_t\cdot\frac{d\mathbf{x}}{dt}=\partial_tp_t+\nabla p_t\cdot\mathbf{v}=-p_t\,\nabla\!\cdot\mathbf{v}.
> $$
>
> Dividing by $p_t$ gives (6).

Computing the trace exactly needs the diagonal of the Jacobian, which takes $D$ backward passes per function evaluation: $O(D^2)$ instead of the $O(D^3)$ determinant, but still too much for images.

### Hutchinson's trace estimator

For any random vector $\boldsymbol\epsilon$ with $\mathbb{E}[\boldsymbol\epsilon]=\mathbf{0}$ and $\mathbb{E}[\boldsymbol\epsilon\boldsymbol\epsilon^\top]=\mathbf{I}$,[^hutchinson1989]

$$
\operatorname{tr}(\mathbf{A})=\mathbb{E}_{\boldsymbol\epsilon}\big[\boldsymbol\epsilon^\top\mathbf{A}\,\boldsymbol\epsilon\big],
\tag{7}
$$

because $\mathbb{E}[\boldsymbol\epsilon^\top\mathbf{A}\boldsymbol\epsilon]=\sum_{i,j}A_{ij}\,\mathbb{E}[\epsilon_i\epsilon_j]=\sum_iA_{ii}$. The point is that $\boldsymbol\epsilon^\top\mathbf{J}$ is a **vector-Jacobian product**, exactly what one backward pass of [[Neural Networks#Computational graphs and reverse-mode autodiff|reverse-mode autodiff]] computes, without ever forming $\mathbf{J}$. FFJORD draws one probe $\boldsymbol\epsilon$ per sample, keeps it fixed along the whole trajectory, and gets an **unbiased** estimate of $\log p_1(\mathbf{x})$ (the log-density is linear in the trace integral) at the cost of about one extra backward pass per function evaluation.[^grathwohl2019] The same estimator appears in [[Score-Based Models#Sliced score matching|sliced score matching]].

Rademacher probes ($\pm1$ with equal probability) have lower variance than Gaussian ones: for symmetric $\mathbf{A}$ the variances are $2\sum_{i\ne j}A_{ij}^2$ and $2\sum_{i,j}A_{ij}^2$, so Rademacher probes remove the diagonal's contribution.

![[Machine Learning/images/normalizing-flows-cnf.svg]]
*Figure 8. The trained CNF on two moons. (a) Paths from base samples at $t=0$ (blue) to data at $t=1$ (orange), solved with RK4. They curve and cross the plane, but never cross each other. (b) Error of a single-probe Hutchinson estimate of $\log p(\mathbf{x})$ against the exact trace, over 400 test points and 50 probes each. Both estimators are unbiased (mean errors $-0.01$ and $+0.02$ nats, within their standard errors), but one probe per point is very noisy: standard deviations 4.4 nats for Gaussian and 3.1 nats for Rademacher probes. In two dimensions a Rademacher probe takes only two values up to sign, hence the two peaks. Averaging over a minibatch of 256 shrinks the noise in the loss by a factor of 16.*

> [!example]- Code: Hutchinson's divergence estimator (PyTorch, tested)
> ```python
> def divergence_hutchinson(v, x, n_probes=1):
>     """Unbiased estimate of tr(dv/dx) for each row of x; one VJP per probe.
>     v must have been computed from x (x.requires_grad = True)."""
>     est = torch.zeros(x.shape[0])
>     for _ in range(n_probes):
>         eps = torch.randint(0, 2, x.shape).to(x) * 2 - 1      # Rademacher +-1
>         eps_J = torch.autograd.grad(v, x, eps, retain_graph=True)[0]
>         est = est + (eps_J * eps).sum(-1)                      # eps^T J eps
>     return est / n_probes
> ```
> Check we ran: for a random 10-dimensional MLP, the estimate with 20 000 probes matches the exact trace of the autograd Jacobian to within 0.005. For training, pass `create_graph=True` to `torch.autograd.grad` so that the estimate itself can be differentiated.

### Training through an ODE solver: the adjoint method

The loss is defined by the ODE's endpoint, so its gradient must flow back through the solver. The direct way is to backpropagate through every solver step, which stores every intermediate state: memory grows with the number of steps. The **adjoint method** instead solves a second ODE backwards in time, and needs memory independent of the number of steps.[^chen2018]

The idea is backpropagation in continuous time. Define the **adjoint** $\mathbf{a}(t)=\partial L/\partial\mathbf{x}(t)$, the gradient of the loss with respect to the state at time $t$: what backprop would carry at "layer" $t$. It obeys its own ODE, and the parameter gradient is an integral along the path:

$$
\frac{d\mathbf{a}}{dt}=-\mathbf{a}(t)^\top\frac{\partial\mathbf{v}_\theta}{\partial\mathbf{x}},
\qquad
\frac{dL}{d\theta}=\int_{t_a}^{t_b}\mathbf{a}(t)^\top\frac{\partial\mathbf{v}_\theta}{\partial\theta}\,dt,
\tag{8}
$$

for an ODE solved from $t_a$ to $t_b$, with $L$ a function of $\mathbf{x}(t_b)$. Both right-hand sides are vector-Jacobian products, one backward pass each. In practice $\mathbf{x}$, $\mathbf{a}$ and the running integral are solved together, backwards from $t_b$, with $\mathbf{x}(t)$ reconstructed by running the original ODE in reverse instead of being stored.

> [!proof]- Derivation sketch of the adjoint equations (8)
> **The adjoint ODE.** Over a short step, $\mathbf{x}(t+\varepsilon)=\mathbf{x}(t)+\varepsilon\mathbf{v}(\mathbf{x}(t),t)+O(\varepsilon^2)$. The loss depends on $\mathbf{x}(t)$ only through $\mathbf{x}(t+\varepsilon)$, so the chain rule gives
>
> $$
> \mathbf{a}(t)^\top=\mathbf{a}(t+\varepsilon)^\top\frac{\partial\mathbf{x}(t+\varepsilon)}{\partial\mathbf{x}(t)}=\mathbf{a}(t+\varepsilon)^\top\Big(\mathbf{I}+\varepsilon\frac{\partial\mathbf{v}}{\partial\mathbf{x}}\Big)+O(\varepsilon^2).
> $$
>
> Rearranging, $\big(\mathbf{a}(t+\varepsilon)-\mathbf{a}(t)\big)/\varepsilon=-\mathbf{a}(t+\varepsilon)^\top\partial\mathbf{v}/\partial\mathbf{x}+O(\varepsilon)$, and the limit is the first equation in (8). This is the continuous form of "multiply the incoming gradient by the layer's Jacobian".
>
> **The parameter gradient.** Treat $\theta$ as if it were a separate copy $\theta_s$ in each short step $[s,s+\varepsilon]$. Changing $\theta_s$ moves $\mathbf{x}(s+\varepsilon)$ by $\varepsilon\,\partial\mathbf{v}/\partial\theta$ and hence the loss by $\varepsilon\,\mathbf{a}(s+\varepsilon)^\top\partial\mathbf{v}/\partial\theta$. The true $\theta$ is shared by all steps, so its gradient is the sum of these contributions, which becomes the integral in (8) as $\varepsilon\to0$. When the ODE runs backwards in time ($t_b<t_a$, as when we evaluate $\log p$ of data), each step has length $-\varepsilon$; the sign flips and the integral's limits flip with it, so (8) holds as an oriented integral in both cases.
>
> **The algorithm.** Set $\mathbf{a}(t_b)=\partial L/\partial\mathbf{x}(t_b)$ and $\mathbf{g}(t_b)=\mathbf{0}$. Solve $[\mathbf{x},\mathbf{a},\mathbf{g}]$ from $t_b$ back to $t_a$, with $d\mathbf{x}/dt=\mathbf{v}$, $d\mathbf{a}/dt=-\mathbf{a}^\top\partial\mathbf{v}/\partial\mathbf{x}$ and $d\mathbf{g}/dt=-\mathbf{a}^\top\partial\mathbf{v}/\partial\theta$. Then $\mathbf{g}(t_a)=\int_{t_a}^{t_b}\mathbf{a}^\top\partial\mathbf{v}/\partial\theta\,dt=dL/d\theta$. For a CNF the state also includes the accumulated divergence, which is handled the same way. Reconstructing $\mathbf{x}(t)$ backwards is cheap in memory but can be numerically inaccurate when the dynamics are stiff, which is why many implementations checkpoint instead.

Our demo is small enough that we simply backpropagated through 8 fixed RK4 steps with the exact trace. Evaluating the trained model with 64 steps instead of 8 changed its test log-likelihood by less than 0.001 nats, so the discretization was not the bottleneck.

### Why maximum-likelihood CNF training is expensive

Every training step of a CNF must:
1. **Solve an ODE** for each minibatch, from data back to noise, with tens to hundreds of evaluations of $\mathbf{v}_\theta$, each with a trace estimate;
2. **Solve a second ODE** for the gradient (adjoint), or store the whole first solve.

Worse, the number of evaluations is not fixed. An adaptive solver takes as many steps as the learned dynamics demand, and maximum likelihood has no reason to keep the dynamics simple. As training proceeds the paths can become curved and **stiff**, and each step gets slower. Regularizing the kinetic energy $\lVert\mathbf{v}_\theta\rVert^2$ and the Jacobian's Frobenius norm makes the paths straighter and training several times cheaper.[^finlay2020] In our tiny demo, one CNF training step took about 58 ms for a batch of 256, against about 9 ms for a batch of 512 for the affine coupling flow, roughly twelve times more per example with a quarter of the parameters.

The root cause is that the maximum-likelihood loss is only defined through the endpoint of a simulation. [[Flow Matching]] removes the simulation: it fixes a probability path from noise to data in advance and **regresses** $\mathbf{v}_\theta$ onto that path's velocity, with no ODE solve and no trace during training.[^lipman2023] The trained network is still a continuous normalizing flow, so everything in this section still applies at test time. Sampling solves the ODE forwards, and the exact likelihood follows from (6) with Hutchinson's estimator. The probability flow ODE of [[Score-Based Models#Infinitely many noise levels: the SDE view|score-based models]] and [[Denoising Diffusion Models|diffusion models]] is a CNF in the same sense, which is how those models report exact likelihoods.[^song2021sde]

## Flows among the generative models

| | Likelihood | Sampling | Constraint on the network | Latent space | Typical image samples |
|---|---|---|---|---|---|
| [[Variational Autoencoders\|VAE]] | lower bound (ELBO) | one pass | none (encoder + decoder) | low-dimensional, approximate encoder | blurry unless very large |
| [[Generative Adversarial Networks\|GAN]] | none | one pass | none, but unstable training | low-dimensional, no encoder | sharp; mode dropping |
| [[Energy-Based Models\|EBM]] | unnormalized ($Z$ unknown) | MCMC, many steps | scalar energy | none | moderate |
| Coupling / autoregressive flow | **exact** | one pass (coupling, IAF) or $D$ passes (MAF) | invertible, equal dimension, cheap log-det | same dimension, **exact** encoder | good, below the state of the art |
| CNF trained by max. likelihood | exact up to solver error | ODE solve | none (Lipschitz) | same dimension, exact encoder | good; costly training |
| [[Denoising Diffusion Models\|Diffusion]] / [[Score-Based Models\|score]] | bound, or exact via the probability flow ODE | many steps | none | same dimension, via the ODE | state of the art |
| [[Flow Matching]] | exact via the ODE (it is a CNF) | ODE solve, few steps if paths are straight | none | same dimension, via the ODE | state of the art |

When to reach for a classic normalizing flow:
- **You need the density itself**, exactly and fast: lossless compression, likelihood-based model comparison, or anomaly scores (with the caveat above).
- **You need both directions in one pass**: coupling flows give samples and densities of arbitrary points at the same cost.
- **As a component.** A flexible variational posterior or prior inside a VAE (IAF); a learned posterior or likelihood in simulation-based inference, a major application of MAF-style flows;[^papamakarios2021] or a sampler for a known, unnormalized physical energy, trained by reverse KL and corrected by importance weights, as in Boltzmann generators.[^noe2019]

For the highest-fidelity images, audio and video, diffusion and flow matching have largely replaced discrete flows. The equal-dimension and invertibility constraints spend parameters on bookkeeping, while a free-form velocity field does not have to.

## Key takeaways

- An invertible, differentiable generator gives an exact density: $\log p_X(\mathbf{x})=\log p_Z(g(\mathbf{x}))+\log\lvert\det\mathbf{J}_g(\mathbf{x})\rvert$. Stretching lowers density, squashing raises it, and compositions add their log-determinants.
- The whole design problem is making the log-determinant cheap: triangular Jacobians (coupling and autoregressive flows), identity plus low rank (planar flows), or special structure (Glow's LU-parameterized $1\times1$ convolutions, and traces for ODEs).
- Coupling layers (NICE, RealNVP, Glow) are fast in both directions. Autoregressive flows are more expressive per layer but fast in only one direction: MAF for density evaluation, IAF for sampling and variational inference.
- Flows cannot change topology; volume-preserving flows cannot concentrate mass. Discrete data need dequantization, and likelihood is reported in bits per dimension, which does not track sample quality.
- A continuous normalizing flow is an ODE. Its log-density changes at the rate $-\nabla\!\cdot\mathbf{v}_\theta$, and Hutchinson's estimator turns the trace into one vector-Jacobian product. Maximum-likelihood training needs ODE solves at every step; flow matching trains the same model by regression.

## Where this shows up next

- [[Flow Matching]]: the simulation-free way to train the continuous flows of this lecture, with straight paths for fast sampling.
- [[Denoising Diffusion Models]] and [[Score-Based Models]]: their probability flow ODE is a CNF, and its exact likelihood uses the instantaneous change of variables (6).
- [[Variational Autoencoders]]: planar flows and IAF make the approximate posterior $q(\mathbf{z}\mid\mathbf{x})$ more flexible, tightening the ELBO.
- [[Energy-Based Models]]: a flow trained by reverse KL towards $e^{-E}$ is a fast sampler for an energy, with exact densities for importance reweighting.
- [[Multivariable Calculus#Densities under a change of variables|Change of variables]] and [[Five Factorizations of a Matrix|LU]]: the two pieces of linear algebra and calculus that every flow is built on.

<!-- References: footnote definitions below render as the numbered "References" list at the bottom of the page -->

[^tabak2013]: Tabak, E. G., & Turner, C. V. (2013). *A Family of Nonparametric Density Estimation Algorithms*. Communications on Pure and Applied Mathematics, 66(2), 145–164. [doi:10.1002/cpa.21423](https://doi.org/10.1002/cpa.21423)
[^rezende2015]: Rezende, D. J., & Mohamed, S. (2015). *Variational Inference with Normalizing Flows*. ICML. [arXiv:1505.05770](https://arxiv.org/abs/1505.05770)
[^papamakarios2021]: Papamakarios, G., Nalisnick, E., Rezende, D. J., Mohamed, S., & Lakshminarayanan, B. (2021). *Normalizing Flows for Probabilistic Modeling and Inference*. JMLR, 22(57), 1–64. [arXiv:1912.02762](https://arxiv.org/abs/1912.02762)
[^kobyzev2020]: Kobyzev, I., Prince, S. J. D., & Brubaker, M. A. (2021). *Normalizing Flows: An Introduction and Review of Current Methods*. IEEE TPAMI, 43(11), 3964–3979. [doi:10.1109/TPAMI.2020.2992934](https://doi.org/10.1109/TPAMI.2020.2992934)
[^dupont2019]: Dupont, E., Doucet, A., & Teh, Y. W. (2019). *Augmented Neural ODEs*. NeurIPS. [arXiv:1904.01681](https://arxiv.org/abs/1904.01681)
[^dinh2014]: Dinh, L., Krueger, D., & Bengio, Y. (2015). *NICE: Non-linear Independent Components Estimation*. ICLR Workshop. [arXiv:1410.8516](https://arxiv.org/abs/1410.8516)
[^dinh2017]: Dinh, L., Sohl-Dickstein, J., & Bengio, S. (2017). *Density Estimation Using Real NVP*. ICLR. [arXiv:1605.08803](https://arxiv.org/abs/1605.08803)
[^kingma2018]: Kingma, D. P., & Dhariwal, P. (2018). *Glow: Generative Flow with Invertible 1x1 Convolutions*. NeurIPS. [arXiv:1807.03039](https://arxiv.org/abs/1807.03039)
[^germain2015]: Germain, M., Gregor, K., Murray, I., & Larochelle, H. (2015). *MADE: Masked Autoencoder for Distribution Estimation*. ICML. [arXiv:1502.03509](https://arxiv.org/abs/1502.03509)
[^papamakarios2017]: Papamakarios, G., Pavlakou, T., & Murray, I. (2017). *Masked Autoregressive Flow for Density Estimation*. NeurIPS. [arXiv:1705.07057](https://arxiv.org/abs/1705.07057)
[^kingma2016]: Kingma, D. P., Salimans, T., Jozefowicz, R., Chen, X., Sutskever, I., & Welling, M. (2016). *Improving Variational Inference with Inverse Autoregressive Flow*. NeurIPS. [arXiv:1606.04934](https://arxiv.org/abs/1606.04934)
[^vandenoord2018]: van den Oord, A., Li, Y., Babuschkin, I., et al. (2018). *Parallel WaveNet: Fast High-Fidelity Speech Synthesis*. ICML. [arXiv:1711.10433](https://arxiv.org/abs/1711.10433)
[^durkan2019]: Durkan, C., Bekasov, A., Murray, I., & Papamakarios, G. (2019). *Neural Spline Flows*. NeurIPS. [arXiv:1906.04032](https://arxiv.org/abs/1906.04032)
[^behrmann2019]: Behrmann, J., Grathwohl, W., Chen, R. T. Q., Duvenaud, D., & Jacobsen, J.-H. (2019). *Invertible Residual Networks*. ICML. [arXiv:1811.00995](https://arxiv.org/abs/1811.00995)
[^chen2019resflow]: Chen, R. T. Q., Behrmann, J., Duvenaud, D., & Jacobsen, J.-H. (2019). *Residual Flows for Invertible Generative Modeling*. NeurIPS. [arXiv:1906.02735](https://arxiv.org/abs/1906.02735)
[^theis2016]: Theis, L., van den Oord, A., & Bethge, M. (2016). *A Note on the Evaluation of Generative Models*. ICLR. [arXiv:1511.01844](https://arxiv.org/abs/1511.01844)
[^ho2019]: Ho, J., Chen, X., Srinivas, A., Duan, Y., & Abbeel, P. (2019). *Flow++: Improving Flow-Based Generative Models with Variational Dequantization and Architecture Design*. ICML. [arXiv:1902.00275](https://arxiv.org/abs/1902.00275)
[^nalisnick2019]: Nalisnick, E., Matsukawa, A., Teh, Y. W., Gorur, D., & Lakshminarayanan, B. (2019). *Do Deep Generative Models Know What They Don't Know?* ICLR. [arXiv:1810.09136](https://arxiv.org/abs/1810.09136)
[^chen2018]: Chen, R. T. Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D. (2018). *Neural Ordinary Differential Equations*. NeurIPS. [arXiv:1806.07366](https://arxiv.org/abs/1806.07366)
[^hutchinson1989]: Hutchinson, M. F. (1989). *A Stochastic Estimator of the Trace of the Influence Matrix for Laplacian Smoothing Splines*. Communications in Statistics – Simulation and Computation, 18(3), 1059–1076. [doi:10.1080/03610918908812806](https://doi.org/10.1080/03610918908812806)
[^grathwohl2019]: Grathwohl, W., Chen, R. T. Q., Bettencourt, J., Sutskever, I., & Duvenaud, D. (2019). *FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models*. ICLR. [arXiv:1810.01367](https://arxiv.org/abs/1810.01367)
[^finlay2020]: Finlay, C., Jacobsen, J.-H., Nurbekyan, L., & Oberman, A. M. (2020). *How to Train Your Neural ODE: the World of Jacobian and Kinetic Regularization*. ICML. [arXiv:2002.02798](https://arxiv.org/abs/2002.02798)
[^lipman2023]: Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., & Le, M. (2023). *Flow Matching for Generative Modeling*. ICLR. [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)
[^song2021sde]: Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2021). *Score-Based Generative Modeling through Stochastic Differential Equations*. ICLR. [arXiv:2011.13456](https://arxiv.org/abs/2011.13456)
[^noe2019]: Noé, F., Olsson, S., Köhler, J., & Wu, H. (2019). *Boltzmann Generators: Sampling Equilibrium States of Many-Body Systems with Deep Learning*. Science, 365(6457). [arXiv:1812.01729](https://arxiv.org/abs/1812.01729)
