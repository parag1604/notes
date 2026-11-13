---
title: Clustering and Expectation Maximization
tags:
  - clustering
  - k-means
  - expectation-maximization
  - gaussian-mixture-models
  - spectral-clustering
  - community-detection
prev: "[[Matrix Factorization and Dimensionality Reduction]]"
next: "[[Neural Networks]]"
---

> **Prerequisites:** [[Discriminant Analysis]] (Gaussian class models, Bayes' rule), [[Non-parametric Models]] (distances, RBF similarities), [[Support Vector Machines]] (kernels, Lagrange multipliers), and [[Matrix Factorization and Dimensionality Reduction]] (eigenvectors, used by spectral clustering).
>
> **TL;DR:**
> - **Clustering** groups unlabelled points. Doing it optimally is NP-hard, so practical algorithms optimize relaxations.
> - **K-Means** alternates "assign each point to its nearest centre" with "move each centre to its points' mean"; **fuzzy c-means** and **kernel K-Means** soften or kernelize this.
> - **EM** handles latent-variable models by repeatedly maximizing a tight lower bound (the ELBO) on the log-likelihood. **Gaussian mixtures** are "QDA with unknown labels".
> - **Spectral clustering** relaxes a balanced graph cut into an eigenvector problem, and **modularity / Louvain** find communities in networks.
> - Hierarchical clustering, DBSCAN and evaluation metrics complete the toolbox.
>
> **Code:** [K-Means-Clustering](https://github.com/parag1604/K-Means-Clustering): K-means for colour quantization of an image; [Gaussian-Mixture-Models](https://github.com/parag1604/Gaussian-Mixture-Models): Gaussian mixtures fitted with EM.

## The clustering problem

We are given $\mathcal{D}=\{\mathbf{x}_i\}_{i=1}^N$ with $\mathbf{x}_i\in\mathbb{R}^D$, and **no labels**. We would like an automated way to split the data into $K$ groups. An assignment is a vector $\mathbf{c}\in\{1,\dots,K\}^N$, where $c_i$ is the cluster of point $i$. For six points forming two obvious blobs, $\mathbf{c}=(1,1,1,2,2,2)$ is a good assignment and $(1,2,1,1,2,2)$ is a poor one. Because no labels guide us, clustering is **unsupervised** learning.

Intuitively, a good clustering has two properties:

1. **low within-cluster spread:** points in the same cluster are close to each other; and
2. **high between-cluster spread:** different clusters are far apart.

With squared Euclidean distances, these are two sides of one coin. The **total scatter** $T=\sum_i\lVert\mathbf{x}_i-\bar{\mathbf{x}}\rVert^2$ does not depend on the clustering, and it splits exactly into a within-cluster part $W$ and a between-cluster part $B$. This is the same decomposition behind [[Discriminant Analysis#Fisher's linear discriminant|Fisher's criterion]]:

$$
T=\underbrace{\sum_{j=1}^K\sum_{i:c_i=j}\lVert\mathbf{x}_i-\boldsymbol{\mu}_j\rVert^2}_{W}+\underbrace{\sum_{j=1}^Kn_j\lVert\boldsymbol{\mu}_j-\bar{\mathbf{x}}\rVert^2}_{B}.
$$

Here $n_j$ is the size of cluster $j$ and $\boldsymbol{\mu}_j$ its mean. Since $T$ is fixed, **minimizing $W$ automatically maximizes $B$**: solving criterion 1 also solves criterion 2. The within-cluster spread can also be written with pairwise distances, $\sum_{i,i'\in C_j}\lVert\mathbf{x}_i-\mathbf{x}_{i'}\rVert^2=2n_j\sum_{i\in C_j}\lVert\mathbf{x}_i-\boldsymbol{\mu}_j\rVert^2$. So the ideal clustering is

$$
\mathbf{c}^*=\arg\min_{\mathbf{c}}\ \sum_{j=1}^K\frac{1}{2n_j}\sum_{i,i':\,c_i=c_{i'}=j}\lVert\mathbf{x}_i-\mathbf{x}_{i'}\rVert^2=\arg\min_{\mathbf{c}}\ W(\mathbf{c}).
$$

The factor $\frac{1}{n_j}$ matters. Without it, large clusters would be penalized more heavily. Searching over all $K^N$ assignments is hopeless: the problem is **NP-hard**, even for $K=2$ in general dimension.[^aloise2009] Practical algorithms therefore solve relaxed versions, usually iteratively.

---

## K-Means

**The algorithm.** Split the hard problem into two easy ones and alternate between them:

1. **Assign:** with the centres fixed, put each point in the cluster of its *nearest centre*.
2. **Update:** with the assignments fixed, move each centre to the *mean* of its points.

> [!example] Algorithm: K-Means (Lloyd's algorithm)
> Initialize the centres $\boldsymbol{\mu}_1,\dots,\boldsymbol{\mu}_K$, for example as $K$ distinct random data points (better: k-means++, below). **Repeat** until no assignment changes:
> 1. $c_i\leftarrow\arg\min_j\lVert\mathbf{x}_i-\boldsymbol{\mu}_j\rVert^2$ for all $i$.
> 2. $\boldsymbol{\mu}_j\leftarrow\dfrac{\sum_i\mathbb{1}\{c_i=j\}\,\mathbf{x}_i}{\sum_i\mathbb{1}\{c_i=j\}}$ for all $j$.

Both steps decrease the **cost** (or *potential*)

$$
J(\boldsymbol{\mu},\mathbf{c})=\sum_{i=1}^{N}\lVert\mathbf{x}_i-\boldsymbol{\mu}_{c_i}\rVert^2,\qquad\text{and K-Means attacks}\quad\min_{\boldsymbol{\mu}}\min_{\mathbf{c}}J(\boldsymbol{\mu},\mathbf{c}).
$$

Optimizing one block of variables while holding the other fixed is **coordinate descent**.[^lloyd1982]

**Why it converges, and why only locally.** $J$ is *not* convex ($\mathbf{c}$ is discrete), so the convexity argument for coordinate descent does not apply. The correct argument is simpler:

- the assignment step is the best possible $\mathbf{c}$ for fixed centres, so $J$ cannot increase;
- the mean is the best possible centre for a fixed cluster, since it minimizes $\sum_i\lVert\mathbf{x}_i-\boldsymbol{\mu}\rVert^2$, so $J$ cannot increase;
- $J\ge0$, and there are only finitely many assignments ($\le K^N$).

So $J$ decreases monotonically, and the algorithm must stop after finitely many steps, typically very few (Figure 1). It stops at a *local* optimum that depends on the initialization. Hence **multiple restarts** (keep the lowest $J$) and smarter seeding. Each iteration costs $O(NKD)$.

![[Machine Learning/images/clustering-em-kmeans-iterations.svg]]
*Figure 1: K-Means on three blobs. The first assignment uses three random data points as centres. After one update the centres already sit near the blobs, and the algorithm converges in three iterations. Bottom right: the cost $J$ never increases.*

**k-means++ seeding.** Pick the first centre uniformly at random. Pick each subsequent centre at random with probability proportional to $\mathrm{dist}(\mathbf{x},\text{nearest chosen centre})^2$. This spreads the initial centres out. It is guaranteed to be $O(\log K)$-competitive with the optimal $J$ in expectation, *before* any Lloyd iterations,[^kmeanspp] and it is the default in most libraries.

**Choosing $K$.** $J$ always decreases as $K$ grows ($K=N$ gives $J=0$), so we cannot simply minimize it.

- **Elbow:** plot the converged $J$ against $K$ and look for the "elbow" where the improvements suddenly flatten.
- **Silhouette:** for each point, let $a_i$ be its mean distance to its own cluster and $b_i$ its mean distance to the nearest *other* cluster. Then $s_i=\frac{b_i-a_i}{\max(a_i,b_i)}\in[-1,1]$, and we prefer the $K$ with the highest mean silhouette.[^rousseeuw1987]
- **Model selection:** for probabilistic models such as GMMs, information criteria like BIC apply (see [[#Gaussian mixture models]]).

![[Machine Learning/images/clustering-em-choosing-k.svg]]
*Figure 2: four blobs. The elbow in $J$ and the peak of the mean silhouette both point to $K=4$.*

**Limitations of K-Means.**

- **Convex, roughly spherical clusters only.** The assignments form a Voronoi partition of the space, so cluster boundaries are piecewise linear and every cluster is convex. Elongated, curved or nested clusters get cut up (Figures 4 and 7).
- **Equal-size and equal-spread bias.** It prefers clusters of similar size and variance.
- **Sensitive to scaling and outliers.** Every distance counts, and means are pulled around by outliers. *K-medoids*, which restricts the centres to data points, is more robust.
- **Initialization.** Local optima make restarts or k-means++ necessary, more so as $K$ grows. This is a separate issue from the curse of dimensionality: in high dimensions, distances themselves become less informative.

---

## Fuzzy c-means

K-Means forces each point into exactly one cluster. **Fuzzy c-means** lets point $i$ belong to cluster $k$ with a *membership weight* $r_i^k\ge0$, where $\sum_kr_i^k=1$, and minimizes[^bezdek1984]

$$
J(\boldsymbol{\mu},\mathbf{r})=\sum_{i=1}^{N}\sum_{k=1}^{K}(r_i^k)^\alpha\,\lVert\mathbf{x}_i-\boldsymbol{\mu}_k\rVert^2,\qquad\alpha>1.
$$

The **fuzzifier** $\alpha$ (commonly $2$) controls how soft the memberships are. Alternate as before.

**Centres, with the weights fixed.** Setting the gradient to zero, $\nabla_{\boldsymbol{\mu}_k}J=\sum_i(r_i^k)^\alpha(2\boldsymbol{\mu}_k-2\mathbf{x}_i)=\mathbf{0}$, gives a weighted mean:

$$
\boldsymbol{\mu}_k=\frac{\sum_i(r_i^k)^\alpha\,\mathbf{x}_i}{\sum_i(r_i^k)^\alpha}.
$$

**Weights, with the centres fixed.** The constraints only couple the memberships *of the same point*, so the problem splits into one small problem per point. Here the [[Support Vector Machines#The Lagrangian dual|Lagrangian]] does the work. We can drop $r_i^k\ge0$, because the solution below turns out positive anyway. Writing $d_{ik}=\lVert\mathbf{x}_i-\boldsymbol{\mu}_k\rVert^2$:

$$
\begin{aligned}
\mathcal{L}_i&=\sum_k(r_i^k)^\alpha d_{ik}-\nu_i\Big(\sum_kr_i^k-1\Big),\qquad\frac{\partial\mathcal{L}_i}{\partial r_i^k}=\alpha(r_i^k)^{\alpha-1}d_{ik}-\nu_i=0\\
\implies r_i^k&=\Big(\frac{\nu_i}{\alpha\,d_{ik}}\Big)^{1/(\alpha-1)}.
\end{aligned}
$$

Substituting into $\sum_kr_i^k=1$ to eliminate $\nu_i$ gives

$$
r_i^k=\frac{d_{ik}^{-1/(\alpha-1)}}{\sum_{k'=1}^{K}d_{ik'}^{-1/(\alpha-1)}}=\frac{\lVert\mathbf{x}_i-\boldsymbol{\mu}_k\rVert^{-2/(\alpha-1)}}{\sum_{k'}\lVert\mathbf{x}_i-\boldsymbol{\mu}_{k'}\rVert^{-2/(\alpha-1)}}.
$$

**The role of $\alpha$** (Figure 3):

- as $\alpha\to1^+$, the exponent blows up, the nearest centre takes all the weight, and we recover hard K-Means;
- as $\alpha\to\infty$, every membership tends to $\frac1K$.

Initialize the centres as in K-Means and the memberships as one-hot vectors, then alternate the two updates until they stop changing.

![[Machine Learning/images/clustering-em-fuzzy-memberships.svg]]
*Figure 3: fuzzy memberships of a 1-D point $x$ in three clusters with centres at $0,3,6$ (dotted). With $\alpha$ close to $1$ the memberships are almost hard. With $\alpha=2$ they blend smoothly between centres. With large $\alpha$ every point is nearly equally shared.*

---

## Kernel K-Means

K-Means can only draw straight boundaries, so it fails on data such as two interleaving moons (Figure 4, left). As in [[Support Vector Machines#Kernels|SVMs]], we can run it in a feature space $\phi(\mathbf{x})$, given only a kernel $\kappa(\mathbf{x},\mathbf{x}')=\phi(\mathbf{x})^\top\phi(\mathbf{x}')$ with $\kappa:\mathcal{X}\times\mathcal{X}\to\mathbb{R}$. (We write $\kappa$ to avoid a clash with the number of clusters $K$.)

The centre of cluster $C_j$ in feature space is $\mathbf{m}_j=\frac{1}{n_j}\sum_{l\in C_j}\phi(\mathbf{x}_l)$. It is *not* $\phi$ of any input point, so it never needs to be computed. The squared distance from a point to a centre expands into kernel evaluations only:

$$
\lVert\phi(\mathbf{x}_i)-\mathbf{m}_j\rVert^2=\kappa(\mathbf{x}_i,\mathbf{x}_i)-\frac{2}{n_j}\sum_{l\in C_j}\kappa(\mathbf{x}_i,\mathbf{x}_l)+\frac{1}{n_j^2}\sum_{l,l'\in C_j}\kappa(\mathbf{x}_l,\mathbf{x}_{l'}).
$$

> [!example] Algorithm: kernel K-Means
> Compute the Gram matrix $\mathbf{G}_{il}=\kappa(\mathbf{x}_i,\mathbf{x}_l)$ and initialize the assignments. **Repeat:** assign each point to the cluster minimizing the distance above. The update step is implicit, because the centres are defined by the assignments.

So kernel K-Means needs **no explicit projection**. Its real costs are $O(N^2)$ kernel evaluations and $O(N^2)$ memory for the Gram matrix. Two cautions, both visible in Figure 4:

- With a plain RBF kernel it still often fails on two moons, because the kernel-space clusters it finds are still "blobs".
- What works well is **spectral clustering** on a similarity graph (below). The two are closely related: weighted kernel K-Means and normalized-cut spectral clustering optimize the same objective.[^dhillon2004]

![[Machine Learning/images/clustering-em-moons.svg]]
*Figure 4: two moons. K-Means cuts straight across (ARI $0.23$). Kernel K-Means with an RBF kernel is only slightly better (ARI $0.33$). Spectral clustering on a 10-nearest-neighbour graph recovers the moons almost perfectly (ARI $0.87$). ARI is the adjusted Rand index (see [[#Evaluating a clustering]]); $1$ means perfect agreement.*

---

## Mixture models and the EM algorithm

### Motivation

Look at the 1-D data in Figure 5. They seem to come from **two Gaussians**. If we knew which Gaussian generated each point, fitting them would be trivial: compute the mean and variance of each group, exactly as in [[Discriminant Analysis#Generative classifiers and Bayes' rule|discriminant analysis]]. But the labels are unknown. A point between the bumps could belong to either Gaussian, with some probability, and those probabilities depend on the Gaussians we are trying to estimate. This chicken-and-egg structure is what EM resolves.

Consider the generative story: pick one of two Gaussians with equal probability, then sample from it. For simplicity, give both a shared, known variance $\sigma_0^2$. The likelihood of one point is

$$
p(x_i\mid\mu_1,\mu_2)=\frac12\mathcal{N}(x_i;\mu_1,\sigma_0^2)+\frac12\mathcal{N}(x_i;\mu_2,\sigma_0^2),
$$

and the maximum-likelihood problem is

$$
(\mu_1,\mu_2)_{\text{ML}}=\arg\max_{\mu_1,\mu_2}\ \sum_i\log\Big(e^{-(x_i-\mu_1)^2/2\sigma_0^2}+e^{-(x_i-\mu_2)^2/2\sigma_0^2}\Big).
$$

Setting the derivatives to zero does **not** give a closed-form solution, because the $\log$ of a *sum* couples the parameters. The MLE is still a perfectly good target. We just need an iterative way to reach a (local) maximum. That is the **expectation-maximization** (EM) algorithm, the standard tool whenever **latent variables** $z$ are needed to explain the data.[^dempster1977]

![[Machine Learning/images/clustering-em-mixture-1d.svg]]
*Figure 5: top, two hidden Gaussians (fitted) and the data (rug). Bottom, the posterior probability, or responsibility, that each Gaussian generated a point at $x$. The point $x_1=0.3$, between the bumps, is shared by both components.*

> [!note] Aside: Jensen's inequality
> For a convex function $f$, $f(\mathbb{E}[X])\le\mathbb{E}[f(X)]$: the chord lies above the curve. For a **concave** function such as $\log$, the inequality flips:
> $$
> \log\mathbb{E}[X]\ \ge\ \mathbb{E}[\log X].
> $$
> With two equally likely values $a,b$ this says $\log\frac{a+b}{2}\ge\frac{\log a+\log b}{2}$ (Figure 6, left). Equality holds when $X$ is constant.

### The evidence lower bound (ELBO)

Let $\mathbf{z}$ be the latent variables, for example the component that produced each point. The **complete-data likelihood** $p(\mathbf{x},\mathbf{z};\theta)$ would be easy to maximize. But $\mathbf{z}$ is unobserved, so we must work with the **incomplete-data** (marginal) log-likelihood:

$$
\ell(\theta)=\log p(\mathbf{x};\theta)=\log\int p(\mathbf{x},\mathbf{z};\theta)\,d\mathbf{z}.
$$

(For discrete $\mathbf{z}$ the integral is a sum.) Introduce any distribution $Q(\mathbf{z})$ over the latents (a *variational* distribution), multiply and divide by it, and apply Jensen:

$$
\begin{aligned}
\ell(\theta)&=\log\int Q(\mathbf{z})\,\frac{p(\mathbf{x},\mathbf{z};\theta)}{Q(\mathbf{z})}\,d\mathbf{z}=\log\mathbb{E}_{Q}\Big[\frac{p(\mathbf{x},\mathbf{z};\theta)}{Q(\mathbf{z})}\Big]\\
&\ge\mathbb{E}_{Q}\Big[\log\frac{p(\mathbf{x},\mathbf{z};\theta)}{Q(\mathbf{z})}\Big]=\underbrace{\mathbb{E}_Q\big[\log p(\mathbf{x},\mathbf{z};\theta)\big]+H(Q)}_{\mathrm{ELBO}(Q,\theta)}.
\end{aligned}
$$

Here $H(Q)=-\mathbb{E}_Q[\log Q]$ is the entropy. The bound holds for **every** $Q$ and $\theta$. How loose is it? Using the *correct* factorization $p(\mathbf{x},\mathbf{z};\theta)=p(\mathbf{z}\mid\mathbf{x};\theta)\,p(\mathbf{x};\theta)$:

$$
\ell(\theta)-\mathrm{ELBO}(Q,\theta)=\int Q(\mathbf{z})\log\frac{Q(\mathbf{z})}{p(\mathbf{z}\mid\mathbf{x};\theta)}\,d\mathbf{z}=D_{\mathrm{KL}}\big(Q(\mathbf{z})\,\big\|\,p(\mathbf{z}\mid\mathbf{x};\theta)\big)\ \ge0.
$$

So $\ell(\theta)=\mathrm{ELBO}(Q,\theta)+D_{\mathrm{KL}}(Q\,\|\,p(\mathbf{z}\mid\mathbf{x};\theta))$. The bound is **tight exactly when $Q$ is the posterior**, $Q(\mathbf{z})=p(\mathbf{z}\mid\mathbf{x};\theta)$. You can check this directly: with $Q=p(\mathbf{z}\mid\mathbf{x})$, the integrand becomes $\log p(\mathbf{x};\theta)$, which integrates to $\ell(\theta)$. For i.i.d. data everything decomposes over examples, with one $Q_i(z_i)$ per point.

### The algorithm

EM is **coordinate ascent on the ELBO**. Make the bound tight at the current parameters, then push the bound up.[^nealhinton1998]

> [!example] Algorithm: Expectation–Maximization
> **Repeat** until the log-likelihood stops improving:
> 1. **E-step:** $Q_i^{t}(z_i)=p(z_i\mid\mathbf{x}_i;\theta^t)$ for all $i$.
> 2. **M-step:** $\theta^{t+1}=\arg\max_{\theta}\ \sum_i\mathbb{E}_{Q_i^t}\big[\log p(\mathbf{x}_i,z_i;\theta)\big]$.
>
> The entropy term of the ELBO does not depend on $\theta$, so the M-step maximizes the **expected complete-data log-likelihood**. We optimize over $\theta$ with $Q$ held at its E-step value, *not* over $\theta^t$.

**EM never decreases the likelihood.** It is not enough to note that $\mathrm{KL}\ge0$. The argument is a chain of three steps:

$$
\ell(\theta^{t+1})\ \overset{(a)}{\ge}\ \mathrm{ELBO}(Q^t,\theta^{t+1})\ \overset{(b)}{\ge}\ \mathrm{ELBO}(Q^t,\theta^{t})\ \overset{(c)}{=}\ \ell(\theta^{t}),
$$

- **(a)** the ELBO is a lower bound for *any* $Q$;
- **(b)** the M-step maximizes over $\theta$;
- **(c)** the E-step made the bound tight at $\theta^t$.

Like K-Means, EM converges only to a **local** maximum (or a saddle), and it can sit on long plateaus (Figure 6, right). So use several random restarts, or initialize from K-Means.

![[Machine Learning/images/clustering-em-jensen-em.svg]]
*Figure 6: left, Jensen's inequality for the concave $\log$: the chord midpoint lies below the curve. Right, the average log-likelihood of a 3-component GMM during EM. It never decreases, but it can plateau near a saddle before improving again.*

> [!tip] Where else the ELBO appears
> The same bound, rewritten as $\mathbb{E}_Q[\log p(\mathbf{x}\mid\mathbf{z})]-D_{\mathrm{KL}}(Q(\mathbf{z})\,\|\,p(\mathbf{z}))$, is the objective of **variational autoencoders**. There the exact posterior is intractable, so $Q$ becomes a learned encoder instead of the exact E-step.

---

## Gaussian mixture models

### The model

Each point has a hidden label $z_i\in\{1,\dots,K\}$ drawn from a **categorical** distribution with mixing weights $\pi_j=P(z_i=j)$, where $\pi_j\ge0$ and $\sum_j\pi_j=1$. Given the label, the point is Gaussian:

$$
z_i\sim\mathrm{Categorical}(\boldsymbol{\pi}),\qquad\mathbf{x}_i\mid z_i=j\ \sim\ \mathcal{N}(\boldsymbol{\mu}_j,\boldsymbol{\Sigma}_j).
$$

The marginal log-likelihood has the troublesome log-of-sum:

$$
\ell(\boldsymbol{\pi},\boldsymbol{\mu},\boldsymbol{\Sigma})=\sum_{i=1}^N\log\sum_{j=1}^{K}\pi_j\,\mathcal{N}(\mathbf{x}_i;\boldsymbol{\mu}_j,\boldsymbol{\Sigma}_j).
$$

This is exactly the [[Discriminant Analysis#Quadratic Discriminant Analysis (QDA)|QDA]] model, except that the class labels are missing.

### E-step: responsibilities

By Bayes' rule, exactly as for the QDA posterior:

$$
q_{ij}=p(z_i=j\mid\mathbf{x}_i)=\frac{\pi_j\,\mathcal{N}(\mathbf{x}_i;\boldsymbol{\mu}_j,\boldsymbol{\Sigma}_j)}{\sum_{k=1}^{K}\pi_k\,\mathcal{N}(\mathbf{x}_i;\boldsymbol{\mu}_k,\boldsymbol{\Sigma}_k)}.
$$

### M-step: weighted maximum likelihood

With $N_j=\sum_iq_{ij}$ (the "effective number" of points in component $j$), the M-step maximizes

$$
\sum_{i,j}q_{ij}\Big[\log\pi_j-\frac12\log\lvert\boldsymbol{\Sigma}_j\rvert-\frac12(\mathbf{x}_i-\boldsymbol{\mu}_j)^\top\boldsymbol{\Sigma}_j^{-1}(\mathbf{x}_i-\boldsymbol{\mu}_j)\Big]+\text{const}.
$$

The three groups of parameters separate.

- **Means.** $\nabla_{\boldsymbol{\mu}_j}=\sum_iq_{ij}\boldsymbol{\Sigma}_j^{-1}(\mathbf{x}_i-\boldsymbol{\mu}_j)=\mathbf{0}$ gives $\boldsymbol{\mu}_j=\frac{1}{N_j}\sum_iq_{ij}\mathbf{x}_i$.
- **Covariances.** It is cleanest to differentiate with respect to the precision $\boldsymbol{\Lambda}_j=\boldsymbol{\Sigma}_j^{-1}$. Using $\partial_{\boldsymbol{\Lambda}}\log\lvert\boldsymbol{\Lambda}\rvert=\boldsymbol{\Lambda}^{-1}$ and $\partial_{\boldsymbol{\Lambda}}\,\mathbf{a}^\top\boldsymbol{\Lambda}\mathbf{a}=\mathbf{a}\mathbf{a}^\top$:
$$
\begin{aligned}
&\sum_iq_{ij}\Big[\tfrac12\boldsymbol{\Sigma}_j-\tfrac12(\mathbf{x}_i-\boldsymbol{\mu}_j)(\mathbf{x}_i-\boldsymbol{\mu}_j)^\top\Big]=\mathbf{0}\\
\Longrightarrow\ &\boldsymbol{\Sigma}_j=\frac{1}{N_j}\sum_iq_{ij}(\mathbf{x}_i-\boldsymbol{\mu}_j)(\mathbf{x}_i-\boldsymbol{\mu}_j)^\top.
\end{aligned}
$$
- **Mixing weights.** Maximize $\sum_{i,j}q_{ij}\log\pi_j$ subject to $\sum_j\pi_j=1$. With a multiplier $\nu$: $\frac{N_j}{\pi_j}=\nu$, so $\pi_j=N_j/\nu$, and the constraint gives $\nu=\sum_jN_j=N$. Hence $\pi_j=N_j/N$. The constraints $\pi_j\ge0$ can be dropped: the solution is already positive, so those constraints are inactive and their [[Support Vector Machines#The Lagrangian dual|KKT]] multipliers are zero.

> [!example] Algorithm: EM for a Gaussian mixture
> Initialize $\boldsymbol{\pi},\boldsymbol{\mu}_j,\boldsymbol{\Sigma}_j$, for example from K-Means. **Repeat:**
> 1. **E-step:** $q_{ij}\propto\pi_j\,\mathcal{N}(\mathbf{x}_i;\boldsymbol{\mu}_j,\boldsymbol{\Sigma}_j)$, normalized over $j$.
> 2. **M-step:** $N_j=\sum_iq_{ij}$, $\ \pi_j=N_j/N$, $\ \boldsymbol{\mu}_j=\frac{1}{N_j}\sum_iq_{ij}\mathbf{x}_i$, $\ \boldsymbol{\Sigma}_j=\frac{1}{N_j}\sum_iq_{ij}(\mathbf{x}_i-\boldsymbol{\mu}_j)(\mathbf{x}_i-\boldsymbol{\mu}_j)^\top$.
>
> These are the QDA estimates with **soft labels** $q_{ij}$ in place of $\mathbb{1}\{y_i=j\}$. Compute the E-step in log space with the log-sum-exp trick to avoid underflow.

> [!warning] Degeneracies
> If a component collapses onto a single point ($\boldsymbol{\mu}_j=\mathbf{x}_i$, $\boldsymbol{\Sigma}_j\to\mathbf{0}$), the likelihood goes to $+\infty$. The global "MLE" of an unconstrained GMM is therefore meaningless. Add a small ridge to every covariance ($\boldsymbol{\Sigma}_j+\epsilon\mathbf{I}$), or use a prior. Restricting the covariance type (spherical, diagonal, tied as in LDA, or full) also trades flexibility for stability.

### K-Means is a limit of GMM-EM

Fix all covariances to $\boldsymbol{\Sigma}_j=\sigma^2\mathbf{I}$ and the weights to $\pi_j=\frac1K$, then let $\sigma\to0$. The responsibilities become one-hot: each point is assigned entirely to its nearest mean. The M-step mean update becomes the K-Means mean. **K-Means is "hard EM" for a spherical GMM.** This explains both its spherical-cluster bias and why GMMs succeed where it fails (Figure 7).

![[Machine Learning/images/clustering-em-kmeans-vs-gmm.svg]]
*Figure 7: two elongated, crossing clusters. K-Means, with its convex Voronoi cells, splits the data into the wrong halves (ARI $\approx0$). A full-covariance GMM fits one elongated Gaussian per cluster (dashed: 2-standard-deviation ellipses) and recovers most of the structure. Points near the crossing remain genuinely ambiguous.*

**Choosing $K$ for a GMM.** The likelihood is a proper probability model, so information criteria apply. For example, the **BIC**, $-2\ell(\hat\theta)+p\log N$ with $p$ free parameters, penalizes extra components.[^schwarz1978] Choose the $K$ with the smallest BIC.

---

## Spectral clustering

### From data to a graph

Spectral clustering works on a **graph**: nodes are data points, and edge weights measure *similarity*. When the data already form a network (people, web pages, proteins), use it directly. Otherwise, build a similarity graph:

- **Fully connected RBF:** $w_{ij}=\exp\big(-\lVert\mathbf{x}_i-\mathbf{x}_j\rVert^2/2\tau^2\big)$, which needs a bandwidth $\tau$ (the same Gaussian as in [[Non-parametric Models#Soft windows: kernel weighting|Parzen windows]]).
- **$\varepsilon$-neighbourhood:** $w_{ij}=1$ if $\lVert\mathbf{x}_i-\mathbf{x}_j\rVert<\varepsilon$, which thresholds the similarity.
- **$k$-nearest-neighbour graph:** connect each point to its $k$ nearest neighbours. This is sparse, adapts to varying density, and is usually the most robust choice.

Write $\mathbf{A}$ for the (weighted) adjacency matrix, $d_i=\sum_jA_{ij}$ for the degree of node $i$, and $\operatorname{diag}(\mathbf{d})$ for the diagonal degree matrix. $\mathbf{A}$ is usually sparse, so store it in a sparse format.

### Cuts as quadratic forms

Split the nodes into two parts $\mathcal{V}_1,\mathcal{V}_2$ and encode the split as $s_i=+1$ for $i\in\mathcal{V}_1$ and $s_i=-1$ for $i\in\mathcal{V}_2$. For an edge between $i$ and $j$, $(s_i-s_j)^2$ is $0$ if both ends are on the same side and $4$ if the edge is **cut**. So

$$
\mathrm{cut}(\mathcal{V}_1,\mathcal{V}_2)=\frac14\sum_{\text{edges }(i,j)}A_{ij}(s_i-s_j)^2=\frac18\sum_{i,j}A_{ij}(s_i-s_j)^2.
$$

The double sum counts each undirected edge twice, hence $\frac18$. Expanding, and using $\sum_jA_{ij}=d_i$:

$$
\begin{aligned}
\frac18\sum_{i,j}A_{ij}\big(s_i^2-2s_is_j+s_j^2\big)&=\frac14\Big(\sum_id_is_i^2-\sum_{i,j}A_{ij}s_is_j\Big)\\
&=\frac14\,\mathbf{s}^\top\big(\operatorname{diag}(\mathbf{d})-\mathbf{A}\big)\mathbf{s}=\frac14\,\mathbf{s}^\top\mathbf{L}\,\mathbf{s}.
\end{aligned}
$$

$\mathbf{L}=\operatorname{diag}(\mathbf{d})-\mathbf{A}$ is the **graph Laplacian**. For any real vector $\mathbf{s}$, $\mathbf{s}^\top\mathbf{L}\mathbf{s}=\frac12\sum_{i,j}A_{ij}(s_i-s_j)^2\ge0$. So $\mathbf{L}$ is positive semi-definite and $\mathbf{L}\mathbf{1}=\mathbf{0}$. More generally, **the number of zero eigenvalues equals the number of connected components**.

### Balanced cuts and their relaxation

Minimizing the cut alone is easy: polynomial-time max-flow/min-cut algorithms solve it. But it is *useless*, because the minimum cut usually chops off a single weakly connected node. We want a **balanced** cut, $\sum_is_i=0$ (equal halves). Together with $s_i\in\{\pm1\}$, which implies $\mathbf{s}^\top\mathbf{s}=N$, this makes the problem a combinatorial (integer) program, and balanced-cut problems (minimum bisection, ratio cut, normalized cut) are **NP-hard**.[^shimalik2000]

The spectral trick is to **relax** $s_i\in\{\pm1\}$ to $s_i\in\mathbb{R}$. We keep the norm and balance constraints:

$$
\min_{\mathbf{s}\in\mathbb{R}^N}\ \mathbf{s}^\top\mathbf{L}\mathbf{s}\qquad\text{s.t.}\qquad\mathbf{s}^\top\mathbf{s}=N,\quad\mathbf{1}^\top\mathbf{s}=0.
$$

This is not a linear program. It minimizes a *quadratic* form over a sphere, i.e. a **Rayleigh quotient**. Its Lagrangian stationarity condition, $2\mathbf{L}\mathbf{s}-2\lambda\mathbf{s}=\mathbf{0}$, says $\mathbf{s}$ must be an **eigenvector** of $\mathbf{L}$, and the objective equals $N\lambda$. So we want the smallest eigenvalue allowed by the constraints.

- The smallest eigenvalue is $\lambda_1=0$, with eigenvector $\mathbf{1}$. It satisfies the norm constraint perfectly well, but it violates **balance**: it puts every node on one side, a trivial cut of size $0$.
- Eigenvectors of a symmetric matrix are orthogonal, so the constraint $\mathbf{1}^\top\mathbf{s}=0$ simply excludes $\mathbf{1}$. By the Courant–Fischer theorem, the minimizer is the eigenvector of the **second-smallest eigenvalue** $\lambda_2$: the **Fiedler vector** $\mathbf{s}_F$. (This needs a connected graph, so that $\lambda=0$ is simple.)

**Rounding.** Assign nodes by the **sign** of their entry in $\mathbf{s}_F$, or run K-Means on the 1-D values, which often works surprisingly well. For $K>2$ clusters, embed each node using the eigenvectors of the **$K$ smallest eigenvalues** of $\mathbf{L}$ and run K-Means on these $K$-dimensional embeddings. The popular Ng–Jordan–Weiss variant uses the normalized Laplacian $\mathbf{L}_{\text{sym}}=\mathbf{I}-\operatorname{diag}(\mathbf{d})^{-1/2}\mathbf{A}\operatorname{diag}(\mathbf{d})^{-1/2}$ and row-normalizes the embedding first. This corresponds to the *normalized cut*, which balances clusters by volume rather than by size.[^njw2001]<sup>,</sup>[^luxburg2007] Computing a few eigenvectors of a sparse $\mathbf{L}$ (Lanczos) is far cheaper than a dense $O(N^3)$ decomposition.

### A worked example by hand

Take the 8-node graph of Figure 8: two four-node clumps, $\{1,2,3,4\}$ and $\{5,6,7,8\}$, joined by a single bridge edge $4$–$5$. The degrees are $(2,3,3,3,3,3,3,2)$, so

$$
\mathbf{L}=\operatorname{diag}(\mathbf{d})-\mathbf{A}=\begin{pmatrix}
2&-1&-1&0&0&0&0&0\\
-1&3&-1&-1&0&0&0&0\\
-1&-1&3&-1&0&0&0&0\\
0&-1&-1&3&-1&0&0&0\\
0&0&0&-1&3&-1&-1&0\\
0&0&0&0&-1&3&-1&-1\\
0&0&0&0&-1&-1&3&-1\\
0&0&0&0&0&-1&-1&2
\end{pmatrix}.
$$

Its eigenvalues are $0,\ 0.291,\ 2,\ 2.806,\ 4,\ 4,\ 4,\ 4.903$.

- The $0$ belongs to $\mathbf{1}$: the graph is connected, so there is exactly one zero.
- The small second eigenvalue $\lambda_2\approx0.291$ signals that the graph is *almost* disconnected: there is a cheap cut.
- The Fiedler vector is
$$
\mathbf{s}_F\approx(0.43,\ 0.37,\ 0.37,\ 0.20,\ -0.20,\ -0.37,\ -0.37,\ -0.43)^\top.
$$

Taking signs gives $\mathbf{s}=(+,+,+,+,-,-,-,-)$. It is balanced ($\mathbf{1}^\top\mathbf{s}=0$) and has $\frac14\mathbf{s}^\top\mathbf{L}\mathbf{s}=1$: it cuts exactly the bridge. Notice also that the bridge nodes $4$ and $5$ have the smallest $\lvert s_F\rvert$. The Fiedler vector also says *how confidently* each node belongs to its side.

![[Machine Learning/images/clustering-em-spectral-example.svg]]
*Figure 8: the worked example. Left, the graph with nodes coloured by the sign of the Fiedler vector; only the dashed bridge is cut. Right, the Fiedler vector's entries. The bridge nodes 4 and 5 sit closest to zero.*

---

## Communities in networks: modularity and the Louvain method

Spectral clustering needs the number of clusters, and the cut objective has no notion of what a "surprisingly dense" group is. **Modularity** supplies such a notion. A good community has *more internal edges than expected by chance*, given each node's degree.

### Modularity

For a graph with $m$ edges, degrees $d_i$ and a partition $c_i$:

$$
Q=\frac{1}{2m}\sum_{i,j}\Big[A_{ij}-\frac{d_id_j}{2m}\Big]\,\mathbb{1}\{c_i=c_j\}.
$$

The term $\frac{d_id_j}{2m}$ is the expected number of edges between $i$ and $j$ in a random graph with the same degrees (the *configuration model*). So $Q$ measures the fraction of edges inside communities *minus* what chance alone would produce.

- $Q=0$ for a random grouping.
- Values around $0.3$–$0.7$ indicate clear community structure.
- Unlike the cut, putting all nodes in one community gives $Q=0$, not a "perfect" score. Modularity has balance built in.

**The spectral connection.** For two communities, encode the partition with $s_i=\pm1$ again. Since $\mathbb{1}\{c_i=c_j\}=\frac12(s_is_j+1)$, and the rows of $\mathbf{B}$ sum to zero,

$$
Q=\frac{1}{4m}\,\mathbf{s}^\top\mathbf{B}\,\mathbf{s},\qquad\mathbf{B}=\mathbf{A}-\frac{\mathbf{d}\mathbf{d}^\top}{2m}\quad(\text{the modularity matrix}).
$$

This is the same quadratic form as before, now to be *maximized*. Relaxing $s_i\in\mathbb{R}$ gives the **leading eigenvector of $\mathbf{B}$**, and its signs split the network. Repeating this inside each part builds more communities. This is Newman's spectral modularity method.[^newman2006]

### The Louvain method

Maximizing $Q$ exactly is NP-hard. **Louvain** is a fast greedy heuristic that scales to millions of nodes.[^blondel2008] For a node $i$ with degree $d_i$, write $d_{i,\text{in}}(C)$ for the total weight of edges from $i$ into community $C$, and $\Sigma_{\text{tot}}(C)$ for the total degree of $C$. Moving $i$ (currently alone) into $C$ changes the modularity by

$$
\Delta Q=\frac{1}{m}\Big[d_{i,\text{in}}(C)-\frac{\Sigma_{\text{tot}}(C)\,d_i}{2m}\Big].
$$

This is cheap to evaluate, because it involves only $i$'s neighbours.

> [!example] Algorithm: Louvain
> Start with every node in its own community. **Repeat** passes until $Q$ stops increasing:
> 1. **Local moving.** Visit the nodes in random order. For each node $i$, remove it from its community, compute $\Delta Q$ for joining each *neighbouring* community, and move it to the best one (or back, if nothing improves $Q$). Sweep repeatedly until no node moves.
> 2. **Aggregation.** Collapse each community into a single super-node. Edge weights between super-nodes are the summed weights between the communities, and internal edges become self-loops. Run phase 1 again on this smaller graph.
>
> Each pass produces a coarser level of a **hierarchy** of communities.

Each local move can only increase $Q$, and the graph shrinks with every pass, so the method terminates. In practice it needs only a handful of passes. Our implementation, verified on planted-community graphs, recovers four planted communities in three passes (Figure 9).

![[Machine Learning/images/clustering-em-louvain.svg]]
*Figure 9: a network with four planted communities of 14–24 nodes (edge probability $0.35$ inside, $0.03$ between). Left, its adjacency matrix in random order shows no structure. Right, reordering the nodes by the communities Louvain found reveals four dense diagonal blocks ($Q\approx0.52$; adjusted Rand index $0.96$ against the planted partition).*

> [!warning] Resolution limit
> Modularity cannot "see" communities smaller than a scale set by the total size of the network. Small but genuine communities may be merged together. Variants such as Leiden, or a resolution parameter $\gamma$ multiplying the null-model term, address this.

---

## Other clustering families

### Hierarchical (agglomerative) clustering

Start with each point as its own cluster. Repeatedly **merge the two closest clusters** until one remains, and record the merges as a tree, the **dendrogram**. Cutting the tree at some height yields a clustering, *for every $K$ at once* (Figure 10, left). The **linkage** defines the distance between clusters:

- **single:** the closest pair of points. It can chain clusters together along thin bridges.
- **complete:** the farthest pair. It gives compact clusters.
- **average:** the mean pairwise distance.
- **Ward:** the merge that increases the within-cluster sum of squares $W$ the least. This is K-Means' objective, built up greedily.

A naive implementation needs $O(N^2)$ memory and $O(N^2\log N)$ time, which limits it to moderate $N$.

### Density-based clustering: DBSCAN

DBSCAN defines clusters as **dense regions separated by sparse ones**.[^ester1996]

- A **core point** has at least `minPts` points within radius $\varepsilon$.
- Clusters grow by connecting core points that lie within $\varepsilon$ of each other, and absorb their nearby non-core (*border*) points.
- Everything else is labelled **noise**.

It needs no $K$, finds arbitrarily shaped clusters, and flags outliers (Figure 10, right). It struggles when clusters have very different densities, because a single $\varepsilon$ cannot fit them all. Its density view is the same one as [[Non-parametric Models#Kernel density estimation|kernel density estimation]].

![[Machine Learning/images/clustering-em-hierarchical-dbscan.svg]]
*Figure 10: left, a Ward dendrogram of three blobs; cutting at the dashed height gives three clusters. Right, DBSCAN finds both moons and labels scattered background points as noise (grey), with no need to specify $K$.*

---

## Evaluating a clustering

**Internal measures** use only the data. Examples are the cost $J$ (only comparable at a fixed $K$), the **silhouette** described above, and the model likelihood or BIC for probabilistic models.

**External measures** compare against reference labels, when they exist (for benchmarking).

- **Adjusted Rand index (ARI).** Count the pairs of points on which two partitions agree (same cluster in both, or different in both), then correct for chance.[^hubert1985] For contingency counts $n_{ab}$ (points in cluster $a$ of one partition and $b$ of the other), row sums $a_\cdot$ and column sums $b_\cdot$:
$$
\mathrm{ARI}=\frac{\sum_{ab}\binom{n_{ab}}{2}-E}{\frac12\Big[\sum_a\binom{a_\cdot}{2}+\sum_b\binom{b_\cdot}{2}\Big]-E},\qquad E=\frac{\sum_a\binom{a_\cdot}{2}\sum_b\binom{b_\cdot}{2}}{\binom{N}{2}}.
$$
It is $1$ for identical partitions, about $0$ for random ones, and invariant to relabelling the clusters.
- **Normalized mutual information (NMI).** $\mathrm{NMI}=\frac{I(U;V)}{\sqrt{H(U)H(V)}}$, where $I$ is the mutual information between the two labelings and $H$ their entropies. It is $1$ for identical partitions and $0$ for independent ones.

## Key takeaways

- Optimal clustering is NP-hard. With squared distances, minimizing within-cluster spread $W$ is the same as maximizing between-cluster spread, because $T=W+B$.
- **K-Means** is coordinate descent on $J$. It converges because $J$ decreases monotonically over finitely many assignments, but only to a local optimum. Use k-means++ and restarts, and choose $K$ by elbow, silhouette or BIC.
- **Fuzzy c-means** gives soft memberships $r_i^k\propto\lVert\mathbf{x}_i-\boldsymbol{\mu}_k\rVert^{-2/(\alpha-1)}$. **Kernel K-Means** needs only the Gram matrix, with no explicit projection.
- **EM** maximizes the ELBO, $\ell(\theta)=\mathrm{ELBO}(Q,\theta)+D_{\mathrm{KL}}(Q\,\|\,p(\mathbf{z}\mid\mathbf{x}))$, by alternating a tight E-step ($Q$ = posterior) with an M-step (maximize the expected complete log-likelihood). The likelihood never decreases.
- **GMM-EM** is QDA estimation with soft labels. K-Means is its hard, spherical limit. Guard against collapsing components.
- **Spectral clustering** relaxes the balanced cut $\frac14\mathbf{s}^\top\mathbf{L}\mathbf{s}$ to an eigenproblem; the Fiedler vector gives the split. **Modularity** compares edges with a degree-preserving null model, and **Louvain** maximizes it greedily and hierarchically.
- **Hierarchical** clustering gives all $K$ at once. **DBSCAN** finds arbitrarily shaped dense clusters and outliers. Evaluate with silhouette (internal) or ARI/NMI (external).

## Where this shows up next

- [[Discriminant Analysis]]: the supervised version of the GMM, with QDA/LDA and the same estimates under hard labels.
- [[Non-parametric Models]]: RBF similarities and density estimation. A GMM sits between one Gaussian and a full KDE.
- [[Matrix Factorization and Dimensionality Reduction]]: PCA and kernel PCA. K-Means is a matrix factorization with one-hot codes, and spectral clustering is a Laplacian eigenmap followed by K-Means.
- Graph representation learning (a planned lecture, next in the series): spectral embeddings, modularity and the graph Laplacian lead to graph neural networks and knowledge-graph embeddings.
- [[Neural Networks]]: learned, non-linear features, starting the deep-learning lectures.
- [[Variational Autoencoders]]: the ELBO with a learned, amortized E-step.

<!-- References: footnote definitions below render as the numbered "References" list at the bottom of the page -->

[^aloise2009]: Aloise, D., Deshpande, A., Hansen, P., & Popat, P. (2009). *NP-hardness of Euclidean Sum-of-Squares Clustering*. Machine Learning, 75, 245–248. [doi:10.1007/s10994-009-5103-0](https://doi.org/10.1007/s10994-009-5103-0)
[^lloyd1982]: Lloyd, S. P. (1982). *Least Squares Quantization in PCM*. IEEE Transactions on Information Theory, 28(2), 129–137. [doi:10.1109/TIT.1982.1056489](https://doi.org/10.1109/TIT.1982.1056489)
[^kmeanspp]: Arthur, D., & Vassilvitskii, S. (2007). *k-means++: The Advantages of Careful Seeding*. SODA. [pdf](https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf)
[^rousseeuw1987]: Rousseeuw, P. J. (1987). *Silhouettes: A Graphical Aid to the Interpretation and Validation of Cluster Analysis*. Journal of Computational and Applied Mathematics, 20, 53–65. [doi:10.1016/0377-0427(87)90125-7](https://doi.org/10.1016/0377-0427(87)90125-7)
[^bezdek1984]: Bezdek, J. C., Ehrlich, R., & Full, W. (1984). *FCM: The Fuzzy c-Means Clustering Algorithm*. Computers & Geosciences, 10(2–3), 191–203. [doi:10.1016/0098-3004(84)90020-7](https://doi.org/10.1016/0098-3004(84)90020-7)
[^dhillon2004]: Dhillon, I. S., Guan, Y., & Kulis, B. (2004). *Kernel k-means: Spectral Clustering and Normalized Cuts*. KDD. [doi:10.1145/1014052.1014118](https://doi.org/10.1145/1014052.1014118)
[^dempster1977]: Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). *Maximum Likelihood from Incomplete Data via the EM Algorithm*. Journal of the Royal Statistical Society: Series B, 39(1), 1–22. [doi:10.1111/j.2517-6161.1977.tb01600.x](https://doi.org/10.1111/j.2517-6161.1977.tb01600.x)
[^nealhinton1998]: Neal, R. M., & Hinton, G. E. (1998). *A View of the EM Algorithm that Justifies Incremental, Sparse, and Other Variants*. In *Learning in Graphical Models*. [doi:10.1007/978-94-011-5014-9_12](https://doi.org/10.1007/978-94-011-5014-9_12)
[^schwarz1978]: Schwarz, G. (1978). *Estimating the Dimension of a Model*. Annals of Statistics, 6(2), 461–464. [doi:10.1214/aos/1176344136](https://doi.org/10.1214/aos/1176344136)
[^shimalik2000]: Shi, J., & Malik, J. (2000). *Normalized Cuts and Image Segmentation*. IEEE TPAMI, 22(8), 888–905. [doi:10.1109/34.868688](https://doi.org/10.1109/34.868688)
[^njw2001]: Ng, A. Y., Jordan, M. I., & Weiss, Y. (2001). *On Spectral Clustering: Analysis and an Algorithm*. NeurIPS. [link](https://papers.nips.cc/paper_files/paper/2001/hash/801272ee79cfde7fa5960571fee36b9b-Abstract.html)
[^luxburg2007]: von Luxburg, U. (2007). *A Tutorial on Spectral Clustering*. Statistics and Computing, 17, 395–416. [doi:10.1007/s11222-007-9033-z](https://doi.org/10.1007/s11222-007-9033-z)
[^newman2006]: Newman, M. E. J. (2006). *Modularity and Community Structure in Networks*. PNAS, 103(23), 8577–8582. [doi:10.1073/pnas.0601602103](https://doi.org/10.1073/pnas.0601602103)
[^blondel2008]: Blondel, V. D., Guillaume, J.-L., Lambiotte, R., & Lefebvre, E. (2008). *Fast Unfolding of Communities in Large Networks*. Journal of Statistical Mechanics, P10008. [doi:10.1088/1742-5468/2008/10/P10008](https://doi.org/10.1088/1742-5468/2008/10/P10008)
[^ester1996]: Ester, M., Kriegel, H.-P., Sander, J., & Xu, X. (1996). *A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise*. KDD. [pdf](https://cdn.aaai.org/KDD/1996/KDD96-037.pdf)
[^hubert1985]: Hubert, L., & Arabie, P. (1985). *Comparing Partitions*. Journal of Classification, 2, 193–218. [doi:10.1007/BF01908075](https://doi.org/10.1007/BF01908075)
