---
title: Energy-Based Models
tags:
  - generative-models
  - energy-based-models
  - hopfield-networks
  - boltzmann-machines
  - representation-learning
date: 2026-11-12
draft: false
prev: "[[Generative Adversarial Networks]]"
next: "[[Score-Based Models]]"
---

> **Prerequisites:** [[Variational Autoencoders]] and [[Generative Adversarial Networks]] (the other two families of generative models), and the softmax from [[Linear Models#More than two classes: softmax regression|softmax regression]]. The Lagrangian aside in [[Support Vector Machines]] is used once, for the maximum-entropy derivation.
>
> **TL;DR:**
> - Any function $E_\theta(\mathbf{x})$ defines a probability distribution $p_\theta(\mathbf{x})=e^{-E_\theta(\mathbf{x})}/Z_\theta$, the **Boltzmann distribution**. Low energy means high probability. This is the most flexible way to write a model, and its price is the normalizer $Z_\theta$, which we usually cannot compute.
> - **Hopfield networks** store memories as valleys of an energy function and recall them by rolling downhill. **Boltzmann machines** make the same network stochastic, so it samples from the Boltzmann distribution instead of only descending it. **RBMs** restrict the wiring so that sampling becomes easy.
> - The likelihood gradient of every EBM has the same shape: push the energy **down on data** and **up on the model's own samples**. Contrastive losses (InfoNCE, CLIP) are this rule with the model's samples replaced by other examples in the batch.
> - Modern Hopfield networks with an exponential energy retrieve with one softmax step, and that step is transformer attention.
>
> **Notation in this lecture:** temperature is $\tau$ and inverse temperature is $\beta=1/\tau$ ($T$ is reserved for the RL horizon). $\mathbf{s}$ is the state of a network of $D$ units, not an RL state.
>
> **Code:** [Energy-Based-Models](https://github.com/parag1604/Energy-Based-Models): Langevin sampling on a 2-D toy, persistent chains in a replay buffer, an image EBM on Bitmoji faces (short-run sampling, inpainting, denoising), conditional super-resolution, and diffusion recovery likelihood. [Contrastive-Representation-Learning](https://github.com/parag1604/Contrastive-Representation-Learning): contrastive losses (InfoNCE, SimCLR, MoCo) on MNIST, for the contrastive-loss section.

## Motivation

The [[Variational Autoencoders|VAE]] and the [[Generative Adversarial Networks|GAN]] both build a generator: a network that turns noise $\mathbf{z}$ into a sample $\mathbf{x}$. Sampling is a single forward pass. The price is a restricted model: the VAE only gets a lower bound on the likelihood, and the GAN gets no density at all.

Energy-based models (EBMs) take the opposite approach. Instead of a sampler, we learn a **scalar score** $E_\theta(\mathbf{x})$ that says how "bad" or implausible $\mathbf{x}$ is. Any network with a scalar output will do. The score is turned into a probability by exponentiating and normalizing, exactly as a classifier's softmax turns logits into class probabilities. Where does that recipe come from? From physics: it is the distribution of states of a system at a fixed temperature. So we start there, as the handwritten notes do.

## The Boltzmann distribution

Consider a box of gas molecules, held at a fixed temperature by its surroundings. The molecules jiggle about (thermal motion), constantly exchanging energy through collisions. At any instant, how "excited" is a given molecule? We cannot say deterministically, so we turn to probability.

### Intuition: a staircase of energy levels

Suppose a molecule can sit in one of a ladder of states $s_k$ with energies $E_k=k\delta$, so each step up costs energy $\delta$ (Figure 1a). Assume each step up happens with probability $p\in(0,1)$, independently of the previous steps. A jump of $N$ levels, an energy increase of $\Delta E=N\delta$, then has probability

$$
P(\Delta E)=p^{N}=e^{N\ln p}=\exp\Big(-\frac{\Delta E}{\delta}\ln\frac{1}{p}\Big)=e^{-\Delta E/\tau},
\qquad \tau=\frac{\delta}{\ln(1/p)}.
$$

Because $0<p<1$, $\ln(1/p)>0$ and the **temperature** $\tau$ is positive, just as absolute temperature is. A large $p$ (jumps are easy) gives a high temperature; a small $p$ gives a low one. In physics the exponent is $-\Delta E/(k_B T)$, and the Boltzmann constant $k_B$ just converts units; we absorb it into $\tau$.

These are probabilities of energy **changes**. What they really pin down is the **ratio** between two states:

$$
\frac{P(s')}{P(s)}=e^{-(E(s')-E(s))/\tau}.
$$

The only way to satisfy this for every pair of states is $P(s)\propto e^{-E(s)/\tau}$. Normalizing so the probabilities sum to 1 gives the **Boltzmann distribution**:

$$
P(s)=\frac{1}{Z}e^{-E(s)/\tau},
\qquad
Z=\sum_{s'} e^{-E(s')/\tau}.
\tag{1}
$$

$Z$ is the **partition function** (from the German *Zustandssumme*, "sum over states").

![[Machine Learning/images/energy-based-models-boltzmann.svg]]
*Figure 1. (a) The staircase picture: each step up adds energy $\delta$ and has probability $p$, so a jump of $N$ levels has probability $p^N=e^{-\Delta E/\tau}$. (b) The resulting Boltzmann distribution over 8 equally spaced levels. At low temperature almost everything sits in the ground state; at high temperature the distribution flattens towards uniform.*

The temperature controls how peaked the distribution is (Figure 1b):
- as $\tau\to0$, all the mass goes to the lowest-energy state (the **ground state**), so sampling becomes $\arg\min_s E(s)$;
- as $\tau\to\infty$, every state becomes equally likely.

> [!note] ASIDE: the rigorous derivation (maximum entropy)
> The staircase is an intuition, not a derivation: it assumes independent steps with a fixed probability. The clean argument, due to Jaynes,[^jaynes1957] asks: of all distributions over states with a given **average energy** $\bar E$, which one assumes the least, i.e. has the maximum entropy?
>
> $$
> \max_{p}\; -\sum_i p_i\log p_i
> \quad\text{s.t.}\quad \sum_i p_i=1,\quad \sum_i p_iE_i=\bar E.
> $$
>
> With multipliers $\lambda_0$ and $\beta$ (as in the [[Support Vector Machines|Lagrangian aside of the SVM lecture]]), set the derivative of the Lagrangian to zero:
>
> $$
> \begin{aligned}
> \mathcal{L}(p)&=-\sum_j p_j\log p_j-\lambda_0\Big(\sum_j p_j-1\Big)-\beta\Big(\sum_j p_jE_j-\bar E\Big),\\
> \frac{\partial\mathcal{L}}{\partial p_i}&=-\log p_i-1-\lambda_0-\beta E_i=0 .
> \end{aligned}
> $$
>
> So $p_i\propto e^{-\beta E_i}$: the Boltzmann distribution, with the inverse temperature $\beta=1/\tau$ appearing as the **Lagrange multiplier of the energy constraint**. Entropy is strictly concave and the constraints are linear, so this stationary point is the unique maximum. A colder system (larger $\beta$) corresponds to a lower average energy.

### Softmax is a Boltzmann distribution

Compare (1) with the softmax over logits $\ell_1,\dots,\ell_K$ with a temperature:

$$
\operatorname{softmax}(\boldsymbol\ell/\tau)_k=\frac{e^{\ell_k/\tau}}{\sum_{j}e^{\ell_j/\tau}} .
$$

This is exactly (1) with **energy = negative logit**, $E_k=-\ell_k$. Higher logit means lower energy means more probable. So every classifier trained with [[Linear Models#More than two classes: softmax regression|softmax regression]] already outputs a Boltzmann distribution over classes, with $\tau=1$. The same formula appears all over this series:
- **temperature sampling** in language models divides the next-token logits by $\tau$ before the softmax;
- **Boltzmann (softmax) exploration** in [[Multi-Armed Bandits#Gradient bandit algorithms|gradient bandits]] picks arms with probability $\propto e^{h_t(a)}$;
- the **Gumbel-softmax** relaxation in [[Policy Gradients#Step 2: relax the argmax with a tempered softmax|Policy Gradients]] anneals $\tau\to0$ to approach a hard argmax;
- the GMM responsibilities in [[Clustering and Expectation Maximization#E-step: responsibilities|EM]] are a Boltzmann distribution over clusters, with energy $\lVert\mathbf{x}-\boldsymbol\mu_k\rVert^2/2$ and temperature $\sigma^2$.

## Energy-based models

An **energy-based model** applies (1) to data. Choose a parametric energy $E_\theta:\mathcal{X}\to\mathbb{R}$, any function at all, and define

$$
p_\theta(\mathbf{x})=\frac{e^{-E_\theta(\mathbf{x})}}{Z_\theta},
\qquad
Z_\theta=\int e^{-E_\theta(\mathbf{x})}\,d\mathbf{x}
\quad\Big(\text{or }\sum_{\mathbf{x}}\text{ for discrete }\mathbf{x}\Big).
\tag{2}
$$

We set $\tau=1$ from now on: a temperature can always be absorbed into $E_\theta$.

Think of $E_\theta$ as a **landscape** over the data space (Figure 2). Plausible data sit in valleys; implausible inputs sit on hills. Learning means sculpting the landscape so that the valleys line up with the training data.

![[Machine Learning/images/energy-based-models-landscape.svg]]
*Figure 2. (a) A one-dimensional energy function with a global and a local minimum. (b) The densities it defines at three temperatures. The deep valley gets the most mass; lowering $\tau$ concentrates almost all the mass there.*

Two properties make EBMs attractive:
- **Total flexibility.** $E_\theta$ need not be invertible (as flows require), need not have a latent variable (as VAEs do), and need not produce samples. Any architecture with a scalar output works.
- **Composability.** Adding energies multiplies densities: $E_1+E_2$ gives $p\propto p_1p_2$, a **product of experts**. Each expert can veto a region by assigning it high energy.[^hinton2002]

And one property makes them hard:

> [!warning] The partition function is intractable
> $Z_\theta$ is a sum or integral over the **entire** space. For binary $\mathbf{x}\in\{0,1\}^D$ it has $2^D$ terms; for images it is an integral over $\mathbb{R}^{D}$ with $D$ in the hundreds of thousands. We can evaluate $E_\theta(\mathbf{x})$, and hence **ratios** $p_\theta(\mathbf{x})/p_\theta(\mathbf{x}')=e^{E_\theta(\mathbf{x}')-E_\theta(\mathbf{x})}$, but not $p_\theta(\mathbf{x})$ itself. Every method in this lecture is a way of learning or sampling without computing $Z_\theta$.

## Hopfield networks: memories as valleys

The handwritten notes call Hopfield networks the predecessors of modern generative models. Strictly, the classical Hopfield network is not generative: it is a deterministic **associative memory**. Its stochastic version, the Boltzmann machine, is the generative one. Hopfield networks are where the energy picture began, though, so we start with them.

Intuitively, we can think of the network's memories as the **valleys of an energy landscape**. Recalling a memory from a partial or noisy cue means starting somewhere on the landscape and **rolling downhill** until we settle in the nearest valley. Kirsanov's analogy is protein folding.[^kirsanov2024hopfield] A protein never searches through its astronomically many shapes. It simply slides down its energy landscape to the folded state. Memory retrieval in a Hopfield network is the same kind of computation: no search, only descent. It is **content-addressable**: we retrieve a memory from part of its content, not from an address.

### Energy and dynamics

The network has $D$ binary units $s_i\in\{-1,+1\}$, connected by symmetric weights $w_{ij}=w_{ji}$ with no self-connections ($w_{ii}=0$), and biases $b_i$. Its energy is

$$
E(\mathbf{s})=-\frac12\sum_{i\ne j}w_{ij}s_is_j-\sum_i b_is_i
=-\frac12\mathbf{s}^\top\mathbf{W}\mathbf{s}-\mathbf{b}^\top\mathbf{s}.
\tag{3}
$$

Read each term as a pairwise **agreement score**. If $w_{ij}>0$, the edge is "happy" (lowers the energy) when $s_i$ and $s_j$ agree. If $w_{ij}<0$, it is happy when they disagree. Low energy means most edges are satisfied at once.

Inference flips one unit at a time. Unit $i$ looks at its **local field** $h_i=\sum_{j}w_{ij}s_j+b_i$ and aligns with it:

$$
s_i\leftarrow\operatorname{sign}(h_i).
\tag{4}
$$

> [!proof]- Asynchronous updates never increase the energy
> Because $\mathbf{W}$ is symmetric with a zero diagonal, the terms of (3) involving $s_i$ are $-s_i\big(\sum_j w_{ij}s_j+b_i\big)=-s_ih_i$, and $h_i$ does not depend on $s_i$. Changing $s_i$ to $s_i'$ therefore changes the energy by
>
> $$
> \Delta E=-(s_i'-s_i)\,h_i .
> $$
>
> Under (4), $s_i'$ has the sign of $h_i$. If $s_i$ flips, then $s_i'-s_i=2s_i'$ and $\Delta E=-2|h_i|\le0$; if it does not flip, $\Delta E=0$. The energy is bounded below and there are finitely many states, so after finitely many flips the network reaches a **fixed point**: a state where no single flip lowers the energy, i.e. a local minimum. Symmetry matters: with $w_{ij}\ne w_{ji}$ no energy function exists and the dynamics can cycle. So can **synchronous** updates (all units at once), which may oscillate between two states.

### Learning: digging valleys with Hebb's rule

To store patterns $\boldsymbol\xi^{(1)},\dots,\boldsymbol\xi^{(M)}\in\{-1,+1\}^D$, we want each to be a valley. Hebb's rule ("cells that fire together wire together") strengthens the connection between units that agree in the stored patterns:[^hopfield1982]

$$
w_{ij}=\frac1D\sum_{m=1}^M\xi^{(m)}_i\xi^{(m)}_j\quad(i\ne j),
\qquad w_{ii}=0 .
\tag{5}
$$

Substituting (5) into (3), with $\mathbf{b}=\mathbf{0}$, shows what this does to the landscape:

$$
E(\mathbf{s})=-\frac{1}{2D}\sum_{m=1}^M\Big(\boldsymbol\xi^{(m)\top}\mathbf{s}\Big)^2+\frac{M}{2}.
$$

Each stored pattern contributes minus its **squared overlap** with the current state. Learning a pattern literally digs a valley centred on it. Recall then works as in Figure 3: start from a corrupted pattern, update units one by one, and watch the energy fall until the stored pattern reappears.

![[Machine Learning/images/energy-based-models-hopfield-recall.svg]]
*Figure 3. A 144-unit Hopfield network storing four $12\times12$ glyphs (X, O, T, L) with Hebb's rule. Starting from the "O" with 30% of its pixels flipped, asynchronous updates restore it exactly. The energy (bottom) decreases monotonically and stops at the energy of the stored pattern.*

> [!example]- Code: Hebbian storage and asynchronous recall (NumPy)
> ```python
> # assumes: xi (M, D) stored patterns in {-1, +1}; s (D,) a noisy cue in {-1, +1}
> import numpy as np
> D = xi.shape[1]
> W = xi.T @ xi / D                          # Hebb's rule (5)
> np.fill_diagonal(W, 0)                     # no self-connections
> energy = lambda s: -0.5 * s @ W @ s        # (3) with b = 0
> for sweep in range(20):
>     changed = False
>     for i in np.random.permutation(D):     # one unit at a time, (4)
>         new = 1 if W[i] @ s >= 0 else -1
>         changed |= new != s[i]
>         s[i] = new
>     if not changed:                        # a fixed point: a local minimum of E
>         break
> ```
> Every flip lowers `energy(s)` (the proof above), so the loop stops after finitely many sweeps.

### Capacity and spurious memories

Why does a stored pattern stay put? At $\mathbf{s}=\boldsymbol\xi^{(1)}$ the local field splits into a signal term and a **crosstalk** term from the other memories:

$$
h_i=\underbrace{\frac{D-1}{D}\,\xi^{(1)}_i}_{\text{signal}}
+\underbrace{\frac1D\sum_{m\ne1}\sum_{j\ne i}\xi^{(m)}_i\xi^{(m)}_j\xi^{(1)}_j}_{\text{crosstalk}} .
$$

For random patterns the crosstalk is a sum of about $MD$ random $\pm1$ terms divided by $D$, so it is roughly Gaussian with mean 0 and variance $\approx M/D$. Bit $i$ is stable unless the crosstalk overwhelms the signal. The **load** $\alpha=M/D$ (patterns per neuron) is what matters. A careful statistical-mechanics analysis shows that recall collapses abruptly at $\alpha_c\approx0.138$:[^amit1985] a network of $D$ neurons stores only about $0.14D$ random patterns (Figure 4).

![[Machine Learning/images/energy-based-models-hopfield-capacity.svg]]
*Figure 4. Simulated capacity with $D=400$ (6 trials per point). Almost all bits of a stored pattern remain stable well past $\alpha_c$ (blue), but recall from a 10%-corrupted cue collapses near $\alpha_c$ (orange): wrong bits trigger further wrong bits, and the state drifts into a spurious minimum. In a finite network the drop is smeared slightly beyond $0.138$.*

The landscape also has valleys we never asked for:
- **reversed patterns** $-\boldsymbol\xi^{(m)}$, since $E(\mathbf{s})=E(-\mathbf{s})$ when $\mathbf{b}=\mathbf{0}$;
- **mixture states**, such as the sign of the sum of three stored patterns;
- **spin-glass states**, uncorrelated with any memory, which appear at high load.

## Boltzmann machines: Hopfield networks that dream

A Hopfield network only descends; it always lands in the nearest valley and stays there. A **Boltzmann machine**[^ackley1985] replaces the deterministic update (4) with a random one, so that the network **samples** from the Boltzmann distribution of its energy instead of minimizing it. For Boltzmann machines we use units $s_i\in\{0,1\}$, as is standard; the energy is still (3).

### The stochastic update rule

Fix all units except $s_i$. The energy gap between $s_i=0$ and $s_i=1$ is $\Delta E_i=E(s_i=0)-E(s_i=1)=h_i$. Under the Boltzmann distribution (1), only the two states matter, so

$$
p(s_i=1\mid\mathbf{s}_{-i})
=\frac{e^{-E(s_i=1)/\tau}}{e^{-E(s_i=1)/\tau}+e^{-E(s_i=0)/\tau}}
=\frac{1}{1+e^{-\Delta E_i/\tau}}
=\sigma\Big(\frac{h_i}{\tau}\Big).
\tag{6}
$$

A unit turns on with a probability that is a sigmoid of its local energy gap.[^kirsanov2024boltzmann] As $\tau\to0$ the sigmoid becomes a step and we recover the Hopfield rule. At $\tau>0$ the network occasionally moves **uphill**, which lets it escape shallow valleys. Updating units repeatedly with (6) is **Gibbs sampling**, and its stationary distribution is exactly $p(\mathbf{s})\propto e^{-E(\mathbf{s})/\tau}$. Starting hot and slowly lowering $\tau$ finds low-energy states: **simulated annealing**.[^kirkpatrick1983]

### Hidden units

Split the units into **visible** units $\mathbf{v}$ (the data) and **hidden** units $\mathbf{h}$. The model of the data is the marginal

$$
p(\mathbf{v})=\frac{1}{Z}\sum_{\mathbf{h}}e^{-E(\mathbf{v},\mathbf{h})} .
$$

Without hidden units, a quadratic energy can only capture pairwise correlations between pixels. Hidden units act as feature detectors, and by marginalizing over them the model captures higher-order structure. The same marginalization makes $p(\mathbf{v})$ a latent-variable model, like the GMM in [[Clustering and Expectation Maximization|EM]] or the [[Variational Autoencoders|VAE]].

### Learning: the wake and sleep phases

The log-likelihood of a visible vector is $\log p(\mathbf{v})=\log\sum_{\mathbf{h}}e^{-E(\mathbf{v},\mathbf{h})}-\log Z$. Differentiating both terms (the general version is derived below, in (9)) gives, for a weight,

$$
\frac{\partial\log p(\mathbf{v})}{\partial w_{ij}}
=\langle s_is_j\rangle_{\text{data}}-\langle s_is_j\rangle_{\text{model}} .
\tag{7}
$$

- $\langle s_is_j\rangle_{\text{data}}$: clamp the visible units to a training example, let the hidden units equilibrate, and measure how often units $i$ and $j$ are on together. This is the **wake** (positive) phase: plain Hebbian learning on data.
- $\langle s_is_j\rangle_{\text{model}}$: let the whole network run freely and measure the same co-activation. This is the **sleep** (negative) phase: anti-Hebbian **unlearning** of whatever the network "dreams" about.

Learning stops when the network's dreams have the same statistics as the data. The rule is beautifully **local**: each weight needs only the activity of the two units it connects. Its weakness is the negative phase, which needs samples from the model's equilibrium distribution, and Gibbs sampling in a fully connected network is painfully slow.

## Restricted Boltzmann machines

A **restricted** Boltzmann machine (RBM) removes all visible–visible and hidden–hidden connections, leaving a bipartite graph (Figure 5a):

$$
E(\mathbf{v},\mathbf{h})=-\mathbf{a}^\top\mathbf{v}-\mathbf{b}^\top\mathbf{h}-\mathbf{v}^\top\mathbf{W}\mathbf{h},
\qquad \mathbf{v}\in\{0,1\}^{D},\;\mathbf{h}\in\{0,1\}^{H}.
$$

Given $\mathbf{v}$, the energy is a sum of separate terms $-(b_j+\mathbf{v}^\top\mathbf{w}_{:j})h_j$, one per hidden unit, so the hidden units are **conditionally independent**. Applying (6) to each one gives

$$
\begin{aligned}
p(\mathbf{h}\mid\mathbf{v})&=\prod_{j}p(h_j\mid\mathbf{v}), & p(h_j=1\mid\mathbf{v})&=\sigma\big(b_j+\mathbf{v}^\top\mathbf{w}_{:j}\big),\\
p(\mathbf{v}\mid\mathbf{h})&=\prod_{i}p(v_i\mid\mathbf{h}), & p(v_i=1\mid\mathbf{h})&=\sigma\big(a_i+\mathbf{w}_{i:}\mathbf{h}\big).
\end{aligned}
$$

So we can sample a whole layer at once: **block Gibbs sampling** alternates $\mathbf{v}\to\mathbf{h}\to\mathbf{v}\to\cdots$ (Figure 5b). Summing out $\mathbf{h}$ analytically also gives the marginal energy of $\mathbf{v}$, the **free energy**:

$$
p(\mathbf{v})=\frac{e^{-F(\mathbf{v})}}{Z},
\qquad
F(\mathbf{v})=-\mathbf{a}^\top\mathbf{v}-\sum_{j=1}^{H}\log\Big(1+e^{\,b_j+\mathbf{v}^\top\mathbf{w}_{:j}}\Big).
$$

![[Machine Learning/images/energy-based-models-rbm.svg]]
*Figure 5. (a) An RBM: a layer of visible units and a layer of hidden units, with connections only between layers. (b) Block Gibbs sampling alternates between the two layers. Contrastive divergence (CD-1) starts the chain at a data point and stops after one reconstruction.*

### Contrastive divergence

Running the chain to equilibrium for every gradient step is too slow. Hinton's **contrastive divergence**[^hinton2002] starts the chain **at the data** and runs only $k$ steps:

> [!example] Algorithm: RBM training with CD-$k$
> 1. Take a minibatch of data $\mathbf{v}^{(0)}$. Compute $p(\mathbf{h}\mid\mathbf{v}^{(0)})$ and sample $\mathbf{h}^{(0)}$.
> 2. Run $k$ steps of block Gibbs sampling: $\mathbf{v}^{(1)}\sim p(\mathbf{v}\mid\mathbf{h}^{(0)})$, $\mathbf{h}^{(1)}\sim p(\mathbf{h}\mid\mathbf{v}^{(1)})$, …, ending at $\mathbf{v}^{(k)}$.
> 3. Update $\mathbf{W}\leftarrow\mathbf{W}+\alpha\Big(\mathbf{v}^{(0)}p(\mathbf{h}\mid\mathbf{v}^{(0)})^\top-\mathbf{v}^{(k)}p(\mathbf{h}\mid\mathbf{v}^{(k)})^\top\Big)$, averaged over the minibatch; similarly $\mathbf{a}\leftarrow\mathbf{a}+\alpha(\mathbf{v}^{(0)}-\mathbf{v}^{(k)})$ and $\mathbf{b}\leftarrow\mathbf{b}+\alpha\big(p(\mathbf{h}\mid\mathbf{v}^{(0)})-p(\mathbf{h}\mid\mathbf{v}^{(k)})\big)$.
> 4. Repeat for a fixed number of epochs.

This is (7) with the model expectation replaced by a short chain started at the data. Using the probabilities $p(\mathbf{h}\mid\cdot)$ instead of binary samples in step 3 reduces variance. Even $k=1$ works well in practice. CD is not, however, the gradient of the likelihood: it is biased, and it can leave spurious low-energy regions far from the data, which a short chain never visits.[^carreira2005] **Persistent CD** keeps the chains running across updates instead of restarting them at the data, and approximates the true gradient better.[^tieleman2008]

> [!example]- Code: one CD-k update of a binary RBM (PyTorch)
> ```python
> # assumes: v0 (B, D) a binary data batch; RBM parameters W (D, H), a (D,), b (H,); k; lr
> import torch
> h_given_v = lambda v: torch.sigmoid(b + v @ W)       # p(h_j = 1 | v)
> v_given_h = lambda h: torch.sigmoid(a + h @ W.T)     # p(v_i = 1 | h)
>
> ph0 = h_given_v(v0)                                  # step 1: positive (wake) phase
> h = torch.bernoulli(ph0)
> for _ in range(k):                                   # step 2: k steps of block Gibbs
>     v = torch.bernoulli(v_given_h(h))
>     ph = h_given_v(v)
>     h = torch.bernoulli(ph)
> W += lr * (v0.T @ ph0 - v.T @ ph) / len(v0)          # step 3: <v h>_data - <v h>_model, (7)
> a += lr * (v0 - v).mean(0)
> b += lr * (ph0 - ph).mean(0)
> ```
> The update is exactly $\nabla_\theta\big[F(\mathbf{v}^{(k)})-F(\mathbf{v}^{(0)})\big]$, averaged over the batch, with the free energy $F$ above and $\mathbf{v}^{(k)}$ held fixed: (9) with a $k$-step chain in place of model samples. For **persistent CD**, keep `h` (or `v`) from the previous update instead of restarting the chain at `v0`.

Stacks of RBMs, trained greedily one layer at a time, became **deep belief networks**.[^hinton2006] In 2006 these were among the first deep networks that could be trained, and they helped start the deep-learning revival. Hopfield and Hinton shared the 2024 Nobel Prize in Physics for this line of work.

## Deep EBMs: learning any energy

Modern EBMs drop the binary units and the quadratic energy: $E_\theta(\mathbf{x})$ is a deep network on continuous data, such as an image. The learning rule is the same as (7), in a general form.

### The likelihood gradient

Take the log of (2):

$$
\log p_\theta(\mathbf{x})=-E_\theta(\mathbf{x})-\log Z_\theta .
$$

The gradient of the log-partition function is an expectation under the model:

$$
\begin{aligned}
\nabla_\theta\log Z_\theta
&=\frac{1}{Z_\theta}\int\nabla_\theta e^{-E_\theta(\mathbf{x}')}\,d\mathbf{x}'
=-\int\frac{e^{-E_\theta(\mathbf{x}')}}{Z_\theta}\nabla_\theta E_\theta(\mathbf{x}')\,d\mathbf{x}'\\
&=-\mathbb{E}_{\mathbf{x}'\sim p_\theta}\big[\nabla_\theta E_\theta(\mathbf{x}')\big].
\end{aligned}
\tag{8}
$$

So the gradient of the average negative log-likelihood is

$$
\nabla_\theta\Big(-\mathbb{E}_{\mathbf{x}\sim p_{\text{data}}}\log p_\theta(\mathbf{x})\Big)
=\underbrace{\mathbb{E}_{\mathbf{x}\sim p_{\text{data}}}\big[\nabla_\theta E_\theta(\mathbf{x})\big]}_{\text{push energy down on data}}
-\underbrace{\mathbb{E}_{\mathbf{x}'\sim p_\theta}\big[\nabla_\theta E_\theta(\mathbf{x}')\big]}_{\text{push energy up on model samples}} .
\tag{9}
$$

Gradient descent on (9) lowers the energy of training points and raises the energy wherever the model currently puts mass. At the optimum the two expectations balance: the model's samples look like the data. For a Boltzmann machine, $-\partial E/\partial w_{ij}=s_is_j$, and (9) becomes (7).

### Why the second term matters: collapse

Drop the second term and minimize only the data energy, and the network will happily make the energy $-\infty$ **everywhere**: a flat landscape that explains nothing. The negative term is what carves valleys, by pushing up everything that is not data. Every practical EBM objective, including the contrastive losses below, is a way of supplying this push-up term cheaply.

### Sampling with Langevin dynamics

For the negative term we need samples from $p_\theta$. In continuous space, the standard tool is **Langevin dynamics**: gradient descent on the energy, plus noise.

$$
\mathbf{x}_{k+1}=\mathbf{x}_k-\frac{\eta}{2}\nabla_{\mathbf{x}}E_\theta(\mathbf{x}_k)+\sqrt{\eta}\,\boldsymbol\epsilon_k,
\qquad \boldsymbol\epsilon_k\sim\mathcal{N}(\mathbf{0},\mathbf{I}).
\tag{10}
$$

Without the noise, (10) would slide into the nearest valley, as a Hopfield network does. The noise lets it explore, and for small step size $\eta$ and many steps the iterates are distributed as $p_\theta$ (Figure 6). A Metropolis–Hastings correction (MALA) removes the bias of a finite $\eta$.

Notice what (10) uses: $\nabla_{\mathbf{x}}E_\theta(\mathbf{x})=-\nabla_{\mathbf{x}}\log p_\theta(\mathbf{x})$. The partition function is a constant in $\mathbf{x}$, so it **disappears** from the gradient. The quantity $\nabla_{\mathbf{x}}\log p(\mathbf{x})$ is called the **score**, and it is all a sampler needs.

![[Machine Learning/images/energy-based-models-langevin.svg]]
*Figure 6. Langevin dynamics (10) on a two-dimensional energy with three wells ($\eta=0.01$). (a) Three chains from different corners each settle into a well and then wander inside it; crossing between wells is rare. (b) 1500 chains started uniformly at random end up split roughly equally among the three wells (31%, 31%, 38%), matching the three equal-mass modes.*

Figure 6a shows the main practical difficulty: **mixing**. A chain rarely crosses a high-energy barrier, so with few steps the samples reflect where the chains started rather than the true weights of the modes. Practical deep EBMs therefore use short chains, initialize them from a **replay buffer** of earlier samples, and regularize the energy magnitudes to keep training stable.[^du2019] Song and Kingma's tutorial surveys these techniques.[^song2021]

> [!example]- Code: Langevin sampling and the maximum-likelihood loss (PyTorch)
> ```python
> # assumes: energy(x) -> (B,) the energies E_theta(x); x_real (B, D) a data batch; K, eta, alpha
> import torch
> def langevin(energy, x, K, eta):
>     """K steps of (10): x <- x - (eta/2) grad_x E(x) + sqrt(eta) eps."""
>     for _ in range(K):
>         x = x.detach().requires_grad_(True)
>         grad, = torch.autograd.grad(energy(x).sum(), x)    # gradient w.r.t. x only
>         x = x - 0.5 * eta * grad + eta ** 0.5 * torch.randn_like(x)
>     return x.detach()                                      # samples are constants for theta
>
> x_fake = langevin(energy, 8 * torch.rand_like(x_real) - 4, K, eta)   # short-run chains from noise
> e_real, e_fake = energy(x_real), energy(x_fake)
> loss = e_real.mean() - e_fake.mean() + alpha * (e_real ** 2 + e_fake ** 2).mean()
> loss.backward()               # E_data[grad E] - E_model[grad E], (9), plus a regularizer
> ```
> The first two terms of `loss` have gradient (9): energy down on data, up on samples. The small `alpha` term keeps the energies near zero so that both terms cannot drift off to infinity together; the companion repository uses the same loss. A replay buffer would replace the uniform noise by stored end points of earlier chains.

### Avoiding sampling altogether

Two classic alternatives train an EBM without any MCMC:
- **Score matching**[^hyvarinen2005] fits the model's score $\nabla_{\mathbf{x}}\log p_\theta$ to the data's, using a clever integration by parts that needs only data and derivatives of $E_\theta$. $Z_\theta$ never appears. Its denoising variant[^vincent2011] turns this into a simple regression problem. We return to both [[#From energies to scores|at the end of this lecture]].
- **Noise-contrastive estimation**[^gutmann2010] trains a logistic-regression classifier to tell data from samples of a known noise distribution, with the EBM's log-density as the logit. $\log Z_\theta$ becomes an ordinary learned parameter. It is a close cousin of the [[Generative Adversarial Networks|GAN]] discriminator, but the "generator" is fixed noise.

## Contrastive losses come from energy

Much of modern representation learning (SimCLR, CLIP) trains with a **contrastive loss**. It is an EBM trained with the partition function restricted to a **candidate set**. Take a conditional EBM $p_\theta(\mathbf{y}\mid\mathbf{x})\propto e^{-E_\theta(\mathbf{x},\mathbf{y})}$ over pairs, and replace the intractable sum over every possible $\mathbf{y}$ by a sum over the $B$ partners in a minibatch. The negative log-likelihood of the true partner becomes the **InfoNCE** loss[^oord2018]

$$
\mathcal{L}_i=E_\theta(\mathbf{x}_i,\mathbf{y}_i)+\log\sum_{j=1}^{B}e^{-E_\theta(\mathbf{x}_i,\mathbf{y}_j)} .
\tag{11}
$$

Its gradient is (9) term by term: the true pair's energy goes down, and the other candidates in the batch serve as the negative samples whose energy goes up, in place of MCMC. The details (InfoNCE and mutual information, supervised contrastive learning, SimCLR, MoCo, hard negatives, collapse and the non-contrastive methods that avoid it, CLIP) are in [[Self-Supervised Representation Learning]].

> [!tip] Intuition
> The physics picture says it all. A good energy landscape needs valleys on the data **and** hills everywhere else. "Contrastive" refers to contrasting the two: every method in this lecture differs only in where it finds the points to push up. Boltzmann machines let the network dream, CD takes one step away from the data, deep EBMs run Langevin chains, NCE uses fixed noise, and InfoNCE uses the rest of the batch.

The same idea turns a classifier into a generative model. Read the logits $\ell_y(\mathbf{x})$ of any classifier as negative energies, $E_\theta(\mathbf{x},y)=-\ell_y(\mathbf{x})$. Marginalizing over labels gives an energy for the input alone, $E_\theta(\mathbf{x})=-\log\sum_ye^{\ell_y(\mathbf{x})}$. Training this with (9) alongside the usual cross-entropy gives a **joint energy model** that both classifies and generates.[^grathwohl2020]

## Modern Hopfield networks and attention

The classical Hopfield network stores only $0.14D$ patterns because its energy, $-\sum_m(\boldsymbol\xi^{(m)\top}\mathbf{s})^2$, grows only quadratically with the overlap: valleys are wide and shallow, and they interfere. **Dense associative memories**[^krotov2016] sharpen them with a steeper function of the overlap:

$$
E(\mathbf{s})=-\sum_{m=1}^{M}F\big(\boldsymbol\xi^{(m)\top}\mathbf{s}\big).
$$

The classical network is $F(z)=z^2$. The polynomial $F(z)=z^n$ stores on the order of $D^{n-1}$ patterns, and the exponential $F(z)=e^{z}$ stores exponentially many.[^demircigil2017]

Ramsauer et al. made the state continuous, $\boldsymbol\xi\in\mathbb{R}^D$, and stored the patterns as the columns of $\mathbf{X}=[\mathbf{x}_1,\dots,\mathbf{x}_M]$.[^ramsauer2021] With the exponential interaction and a term that keeps $\boldsymbol\xi$ bounded, the energy is

$$
E(\boldsymbol\xi)=-\frac1\beta\log\sum_{m=1}^{M}e^{\beta\,\mathbf{x}_m^\top\boldsymbol\xi}+\frac12\boldsymbol\xi^\top\boldsymbol\xi+\text{const}.
$$

The first term is concave in $\boldsymbol\xi$ (a negated log-sum-exp) and the second is convex. The concave–convex procedure minimizes such an energy by repeatedly setting the gradient of the convex part equal to minus the gradient of the concave part at the current point. The gradient of the log-sum-exp is a softmax-weighted average, so the update is

$$
\boldsymbol\xi^{\text{new}}=\mathbf{X}\operatorname{softmax}\big(\beta\,\mathbf{X}^\top\boldsymbol\xi\big).
\tag{12}
$$

It is guaranteed to lower the energy, and for well-separated patterns it converges in a **single step**. Now rename things:
- the state $\boldsymbol\xi$ is a **query**;
- the stored patterns, compared against the query, are **keys**;
- the patterns returned as a weighted average are **values**;
- $\beta=1/\sqrt{d_k}$.

Then (12) is exactly the [[Sequence Models#Attention as a soft dictionary lookup|attention]] of a transformer,[^vaswani2017] $\operatorname{softmax}(\mathbf{q}^\top\mathbf{K}^\top/\sqrt{d_k})\mathbf{V}$. **One attention step is one retrieval step of an associative memory.** The temperature controls what is retrieved (Figure 7). At small $\beta$, the softmax is nearly uniform and the update returns an average of all memories, a "metastable" state. At large $\beta$, it snaps to the single closest memory. The same weighted-average-by-similarity view appeared earlier as [[Non-parametric Models#Similarity-based (dot-product) neighbours|kernel regression]].

![[Machine Learning/images/energy-based-models-modern-hopfield.svg]]
*Figure 7. One retrieval step (12) of a modern Hopfield network storing the four glyphs from Figure 3 as continuous vectors. The query is the "T" with its lower half masked (set to 0). Top: the retrieved state; bottom: the softmax weights over the stored patterns. At $\beta=0.002$ the weights are nearly uniform and the output is a blur of all four glyphs; at $\beta=0.03$ the "T" dominates; at $\beta=0.5$ the full "T" is retrieved in one step.*

> [!example]- Code: one retrieval step is one attention call (PyTorch)
> ```python
> # assumes: X (D, M) stored patterns as columns; xi (D,) the query state; beta
> import torch, torch.nn.functional as F
> xi_new = X @ torch.softmax(beta * X.T @ xi, dim=0)          # (12)
>
> # the same step as transformer attention: query xi, keys = values = stored patterns
> q, K = xi[None, None, None, :], X.T[None, None]              # shapes (1,1,1,D), (1,1,M,D)
> attn = F.scaled_dot_product_attention(q, K, K, scale=beta)[0, 0, 0]
> ```
> `attn` equals `xi_new`. With separate learned projections for queries, keys and values, this is a transformer's attention layer.

## EBMs among the generative models

| | VAE | GAN | EBM |
|---|---|---|---|
| what is learned | encoder + decoder | generator + critic | one scalar energy $E_\theta(\mathbf{x})$ |
| density | lower bound (ELBO) | none | unnormalized: $e^{-E_\theta}$ known up to $Z_\theta$ |
| sampling | one forward pass | one forward pass | iterative MCMC (Gibbs, Langevin) |
| training signal | maximize the ELBO | adversarial min–max | push data down, samples up (9) |
| main difficulty | blurry samples, posterior collapse | instability, mode collapse | intractable $Z_\theta$, slow mixing |

[[Normalizing Flows]] keep exact likelihoods by restricting the architecture to invertible maps.

## From energies to scores

Look back at what each part of this lecture actually used:
- **Sampling** with Langevin dynamics (10) used only $\nabla_{\mathbf{x}}E_\theta=-\nabla_{\mathbf{x}}\log p_\theta$, the score.
- **Training** by maximum likelihood (9) needed samples from the model, and hence the score again, inside an expensive inner loop of MCMC.
- **The energy value itself**, and the partition function that normalizes it, were never needed for either.

So why learn an energy at all? We could instead learn the **score directly**: a network $\mathbf{s}_\theta(\mathbf{x})\approx\nabla_{\mathbf{x}}\log p_{\text{data}}(\mathbf{x})$ that outputs a vector field pointing "uphill in probability". Then sample by plugging $\mathbf{s}_\theta$ into Langevin dynamics, with no $Z_\theta$ and no MCMC during training. Two questions stand in the way, and they set up the next lectures:

1. **How do we fit a score without knowing the true one?** We never observe $\nabla_{\mathbf{x}}\log p_{\text{data}}$, only samples. Score matching answers this, and its denoising version reduces it to "predict the noise that was added to a data point".
2. **Where is the learned score accurate?** Only where there is data. Between the modes of Figure 6 the density is tiny, the score is learned badly, and Langevin chains mix poorly there anyway. The fix is to **add noise to the data**: noise fills the empty regions and connects the modes. Learning scores at many noise levels and annealing from coarse to fine gives **noise-conditional score networks**.

Follow that idea to its end, with a whole sequence of noise levels that gradually destroys the data, and you arrive at [[Denoising Diffusion Models|denoising diffusion models]]. They can also be derived as a very deep [[Variational Autoencoders|VAE]], and the two routes meet. Replacing the noisy sampling path by a deterministic one gives **flow matching**.

## Key takeaways

- The Boltzmann distribution $p\propto e^{-E/\tau}$ is the maximum-entropy distribution at fixed average energy. Softmax is the same formula with energy = negative logit, and the temperature trades greedy (low $\tau$) against uniform (high $\tau$).
- An EBM is any energy network plus (2). It is maximally flexible, but $Z_\theta$ is intractable, so we can compare probabilities but not compute them.
- Hopfield networks store memories as energy minima with Hebb's rule and recall them by asynchronous descent, which provably never increases the energy. Capacity is about $0.14D$, and spurious minima appear.
- Boltzmann machines sample instead of descending: a unit turns on with probability $\sigma(h_i/\tau)$. Learning is "wake minus sleep" statistics (7). RBMs make sampling a two-step block Gibbs chain, and contrastive divergence truncates it.
- Every EBM gradient is (9): energy down on data, up on the model's samples. Langevin dynamics provides samples and needs only the score, so $Z_\theta$ drops out. InfoNCE/CLIP use the batch as the negative samples.
- Since sampling needs only the score, the natural next step is to learn the score directly. That leads to score-based models, diffusion models and flow matching.
- Modern Hopfield networks with an exponential energy retrieve in one step, and that step is attention.

## Where this shows up next

- [[Linear Models#More than two classes: softmax regression|Softmax regression]]: the softmax classifier as a Boltzmann distribution over classes, and the starting point of joint energy models.
- [[Multi-Armed Bandits#Gradient bandit algorithms|Gradient bandits]] and the Gumbel-softmax in [[Policy Gradients#Straight-Through Gumbel-Softmax (STGS) for PG Algorithms|Policy Gradients]]: Boltzmann exploration and temperature annealing. Maximum-entropy RL ([[Actor-Critic Algorithms#Soft actor-critic: maximum-entropy RL|soft actor-critic]]) gives policies $\pi(a\mid s)\propto e^{Q(s,a)/\tau}$.
- [[Clustering and Expectation Maximization]]: GMM responsibilities as a Boltzmann distribution over clusters; RBMs as latent-variable models.
- [[Generative Adversarial Networks]]: NCE as a classifier between data and noise; contrastive pushes as a fixed-generator cousin of adversarial training.
- [[Non-parametric Models#Similarity-based (dot-product) neighbours|Similarity-based neighbours]]: attention as soft nearest-neighbour retrieval, now derived from an energy.
- [[Score-Based Models]]: the next lecture, which learns the score directly. Then, in order: [[Denoising Diffusion Models]] (DDPM, and why it is the same model reached from the VAE side), then [[Flow Matching]] and [[Normalizing Flows]]. Contrastive losses as EBMs are developed in [[Self-Supervised Representation Learning]].

<!-- References: footnote definitions below render as the numbered "References" list at the bottom of the page -->

[^jaynes1957]: Jaynes, E. T. (1957). *Information Theory and Statistical Mechanics*. Physical Review, 106(4), 620–630. [doi:10.1103/PhysRev.106.620](https://doi.org/10.1103/PhysRev.106.620)
[^hinton2002]: Hinton, G. E. (2002). *Training Products of Experts by Minimizing Contrastive Divergence*. Neural Computation, 14(8), 1771–1800. [doi:10.1162/089976602760128018](https://doi.org/10.1162/089976602760128018)
[^kirsanov2024hopfield]: Kirsanov, A. (2024). *A Brain-Inspired Algorithm For Memory* [Video]. YouTube. [youtube.com/watch?v=1WPJdAW-sFo](https://www.youtube.com/watch?v=1WPJdAW-sFo)
[^hopfield1982]: Hopfield, J. J. (1982). *Neural Networks and Physical Systems with Emergent Collective Computational Abilities*. PNAS, 79(8), 2554–2558. [doi:10.1073/pnas.79.8.2554](https://doi.org/10.1073/pnas.79.8.2554)
[^amit1985]: Amit, D. J., Gutfreund, H., & Sompolinsky, H. (1985). *Storing Infinite Numbers of Patterns in a Spin-Glass Model of Neural Networks*. Physical Review Letters, 55(14), 1530–1533. [doi:10.1103/PhysRevLett.55.1530](https://doi.org/10.1103/PhysRevLett.55.1530)
[^ackley1985]: Ackley, D. H., Hinton, G. E., & Sejnowski, T. J. (1985). *A Learning Algorithm for Boltzmann Machines*. Cognitive Science, 9(1), 147–169. [doi:10.1207/s15516709cog0901_7](https://doi.org/10.1207/s15516709cog0901_7)
[^kirsanov2024boltzmann]: Kirsanov, A. (2024). *Generative Model That Won 2024 Nobel Prize* [Video]. YouTube. [youtube.com/watch?v=_bqa_I5hNAo](https://www.youtube.com/watch?v=_bqa_I5hNAo)
[^kirkpatrick1983]: Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. (1983). *Optimization by Simulated Annealing*. Science, 220(4598), 671–680. [doi:10.1126/science.220.4598.671](https://doi.org/10.1126/science.220.4598.671)
[^carreira2005]: Carreira-Perpiñán, M. Á., & Hinton, G. E. (2005). *On Contrastive Divergence Learning*. AISTATS. [proceedings.mlr.press/r5/carreira-perpinan05a](https://proceedings.mlr.press/r5/carreira-perpinan05a.html)
[^tieleman2008]: Tieleman, T. (2008). *Training Restricted Boltzmann Machines using Approximations to the Likelihood Gradient*. ICML. [doi:10.1145/1390156.1390290](https://doi.org/10.1145/1390156.1390290)
[^hinton2006]: Hinton, G. E., Osindero, S., & Teh, Y.-W. (2006). *A Fast Learning Algorithm for Deep Belief Nets*. Neural Computation, 18(7), 1527–1554. [doi:10.1162/neco.2006.18.7.1527](https://doi.org/10.1162/neco.2006.18.7.1527)
[^du2019]: Du, Y., & Mordatch, I. (2019). *Implicit Generation and Generalization in Energy-Based Models*. NeurIPS. [arXiv:1903.08689](https://arxiv.org/abs/1903.08689)
[^song2021]: Song, Y., & Kingma, D. P. (2021). *How to Train Your Energy-Based Models*. [arXiv:2101.03288](https://arxiv.org/abs/2101.03288)
[^hyvarinen2005]: Hyvärinen, A. (2005). *Estimation of Non-Normalized Statistical Models by Score Matching*. JMLR, 6, 695–709. [jmlr.org/papers/v6/hyvarinen05a](https://jmlr.org/papers/v6/hyvarinen05a.html)
[^vincent2011]: Vincent, P. (2011). *A Connection Between Score Matching and Denoising Autoencoders*. Neural Computation, 23(7), 1661–1674. [doi:10.1162/NECO_a_00142](https://doi.org/10.1162/NECO_a_00142)
[^gutmann2010]: Gutmann, M., & Hyvärinen, A. (2010). *Noise-Contrastive Estimation: A New Estimation Principle for Unnormalized Statistical Models*. AISTATS. [proceedings.mlr.press/v9/gutmann10a](https://proceedings.mlr.press/v9/gutmann10a.html)
[^oord2018]: van den Oord, A., Li, Y., & Vinyals, O. (2018). *Representation Learning with Contrastive Predictive Coding*. [arXiv:1807.03748](https://arxiv.org/abs/1807.03748)
[^grathwohl2020]: Grathwohl, W., Wang, K.-C., Jacobsen, J.-H., Duvenaud, D., Norouzi, M., & Swersky, K. (2020). *Your Classifier is Secretly an Energy Based Model and You Should Treat it Like One*. ICLR. [arXiv:1912.03263](https://arxiv.org/abs/1912.03263)
[^krotov2016]: Krotov, D., & Hopfield, J. J. (2016). *Dense Associative Memory for Pattern Recognition*. NeurIPS. [arXiv:1606.01164](https://arxiv.org/abs/1606.01164)
[^demircigil2017]: Demircigil, M., Heusel, J., Löwe, M., Upgang, S., & Vermet, F. (2017). *On a Model of Associative Memory with Huge Storage Capacity*. Journal of Statistical Physics, 168, 288–299. [doi:10.1007/s10955-017-1806-y](https://doi.org/10.1007/s10955-017-1806-y)
[^ramsauer2021]: Ramsauer, H., et al. (2021). *Hopfield Networks is All You Need*. ICLR. [arXiv:2008.02217](https://arxiv.org/abs/2008.02217)
[^vaswani2017]: Vaswani, A., et al. (2017). *Attention Is All You Need*. NeurIPS. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
