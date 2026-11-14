---
title: Graph Neural Networks
tags:
  - neural-networks
  - deep-learning
  - graphs
  - graph-neural-networks
date: 2026-11-14
draft: false
prev: "[[Sequence Models]]"
next: "[[Self-Supervised Representation Learning]]"
---

> **Prerequisites:** [[Graph Representation Learning]] (graphs and their notation, shallow node embeddings, transductive vs inductive learning, knowledge-graph embeddings), [[Neural Networks]] (MLPs, backpropagation, normalization layers), [[Convolutional Neural Networks]] (convolution as a shared local map, equivariance and pooling) and [[Sequence Models]] (attention and transformers). The spectral sections use the graph Laplacian from [[Clustering and Expectation Maximization#Cuts as quadratic forms|spectral clustering]] and the [[Eigenvalues and Eigenvectors#The spectral theorem|spectral theorem]]; over-smoothing uses the [[Markov Chains#Convergence and the ergodic theorem|convergence of random walks]].
>
> **TL;DR:**
> - Images are grids and text is a chain; most other data (molecules, social networks, road maps, citations, knowledge bases) is a **graph**. A graph has no canonical order of its nodes and no fixed number of neighbours, so a layer on graphs must be **permutation equivariant**: relabel the nodes and the output is relabelled the same way.
> - **Spectral** graph convolution filters signals in the eigenbasis of the graph Laplacian. Polynomial filters (**ChebNet**) make it local and cheap, and their first-order version with a **renormalization trick** is the **GCN**: $\mathbf{H}'=\sigma(\tilde{\mathbf{D}}^{-1/2}\tilde{\mathbf{A}}\tilde{\mathbf{D}}^{-1/2}\mathbf{H}\mathbf{W})$, a learned low-pass filter.
> - Almost every GNN is an instance of **message passing**: each node aggregates messages from its neighbours with a permutation-invariant function and updates its state. GCN, **GraphSAGE** (inductive, sampled neighbourhoods) and **GAT** (attention over neighbours) differ only in the message and the aggregator.
> - Message-passing GNNs are at most as powerful as the **1-Weisfeiler–Lehman** test; **GIN**'s injective sum aggregation reaches that bound. Depth brings **over-smoothing** (repeated averaging converges to a random walk's stationary distribution) and **over-squashing**; **heterophily** breaks the "neighbours are alike" assumption.
> - The same encoders classify nodes, predict links and classify whole graphs, scale with sampling, and extend to **graph transformers**, **self-supervised** objectives and **knowledge graphs**.
>
> **Code:** [Graph-Representation-Learning](https://github.com/parag1604/Graph-Representation-Learning): from-scratch tutorial; versions v3–v6 (GCN vs MLP on karate, inductive GraphSAGE and GAT, WL and GIN expressiveness, over-smoothing with residual GCN and APPNP) go with this lecture.

> [!note] Local notation
> A graph is $\mathcal{G}=(\mathcal{V},\mathcal{E})$ with $N=\lvert\mathcal{V}\rvert$ nodes (in this lecture $N$ counts **nodes**, not samples) and adjacency matrix $\mathbf{A}\in\{0,1\}^{N\times N}$. $\mathcal{N}(v)$ is the set of neighbours of node $v$ and $d_v=\lvert\mathcal{N}(v)\rvert$ its degree. Because $D$ would clash with the degree matrix, node features have dimension $F$: $\mathbf{X}\in\mathbb{R}^{N\times F}$, with row $\mathbf{x}_v$. As in [[Clustering and Expectation Maximization#From data to a graph|spectral clustering]], $\mathbf{D}=\operatorname{diag}(\mathbf{d})$ is the degree matrix. Layer $\ell$ holds node states $\mathbf{H}^{(\ell)}$ with rows $\mathbf{h}_v^{(\ell)}$, and $\mathbf{H}^{(0)}=\mathbf{X}$. Graphs are undirected unless stated otherwise.

## Motivation

The last two lectures built networks for two kinds of structure. A [[Convolutional Neural Networks|CNN]] assumes a **grid**: every pixel has the same eight neighbours in the same fixed positions, so one $3\times3$ kernel can be slid everywhere. A [[Sequence Models|sequence model]] assumes a **chain** (with attention, a chain whose tokens can all see each other), where positions are ordered and a positional encoding tells the model who is where.

Now look at a molecule. Atoms are nodes, bonds are edges. A carbon atom may have four neighbours and a hydrogen one; there is no "up" or "left", and no first atom. A social network is worse: a few people have thousands of friends, most have a handful, and the graph has millions of nodes. Citation graphs, road networks, protein interaction networks, meshes, knowledge bases and recommender systems (users and items) all look like this. Grids and chains are in fact the special cases: an image is a grid graph and a sentence is a path graph.

Why not use the tools we have?

- **An MLP on the adjacency matrix.** Flatten $\mathbf{A}$ and $\mathbf{X}$ into one long vector. The input size depends on $N$, so the network only works for graphs of one size. Worse, the same graph with its nodes numbered differently gives a completely different vector: there are $N!$ numberings, and the MLP would have to learn that they are all the same graph.
- **A CNN.** Convolution needs neighbours at fixed relative positions so that weight $w_{-1,0}$ always multiplies "the pixel above". A node's neighbours form an unordered set of varying size, so there is nothing for each weight to attach to.
- **An RNN.** It needs an order to read the nodes in. Any order we choose is arbitrary, and the output would depend on it.

What we need is the graph version of the two ideas that made CNNs work: **locality** (a node is updated from its neighbourhood) and **weight sharing** (the same function at every node), plus the symmetry that graphs actually have, which is **permutation** of the node labels.

### Permutation invariance and equivariance

Renumber the nodes with a permutation matrix $\mathbf{P}$. The features become $\mathbf{P}\mathbf{X}$ and the adjacency matrix becomes $\mathbf{P}\mathbf{A}\mathbf{P}^\top$ (rows *and* columns are permuted). It is the same graph, so:

- a function $g$ that outputs one thing for the whole graph (a molecule's toxicity) should be **permutation invariant**: $g(\mathbf{P}\mathbf{X},\mathbf{P}\mathbf{A}\mathbf{P}^\top)=g(\mathbf{X},\mathbf{A})$;
- a function $f$ that outputs one vector per node (each user's interests) should be **permutation equivariant**: $f(\mathbf{P}\mathbf{X},\mathbf{P}\mathbf{A}\mathbf{P}^\top)=\mathbf{P}f(\mathbf{X},\mathbf{A})$. Relabel the input and the output rows are relabelled the same way.

This is exactly the pattern of a CNN, with translations replaced by permutations: stack **equivariant** layers (convolutions there, message passing here), then finish with an **invariant** pooling for graph-level outputs ([[Convolutional Neural Networks#Equivariance, invariance and pooling|equivariance, invariance and pooling]]). It is also what [[Sequence Models#Order: positional encodings|attention without positional encodings]] already does: self-attention is permutation equivariant, which is why a transformer needs positional encodings to know about order. A transformer is a graph network on the **complete** graph; a GNN restricts attention, or averaging, to the edges that exist.[^bronstein2021]

### Three kinds of task

| Level | Predict | Examples |
|---|---|---|
| Node | a label or value per node | the topic of a paper in a citation graph; fraud in a transaction graph; the faction of a club member |
| Edge (link) | whether an edge exists, or its type | friend and product recommendation; drug–target interaction; completing a knowledge graph |
| Graph | a label or value per graph | toxicity or solubility of a molecule; the function of a protein |

Node tasks are usually **semi-supervised**: one big graph, a few labelled nodes, and we predict the rest. Graph tasks look like ordinary supervised learning, with a dataset of many small graphs. All three use the same equivariant node encoder; only the head differs (see [[#Tasks: nodes, links and whole graphs]]).

## Signals on graphs and the graph Fourier transform

The historical route to GNNs went through signal processing.[^shuman2013] On a grid, convolution is easiest to understand in the Fourier domain: convolving is multiplying the spectrum by the filter's frequency response. Can we define "frequencies" on a graph?

### The Laplacian measures smoothness

Treat a feature column $\mathbf{x}\in\mathbb{R}^N$ as a **signal** that assigns one number to each node. The [[Clustering and Expectation Maximization#Cuts as quadratic forms|graph Laplacian]] $\mathbf{L}=\mathbf{D}-\mathbf{A}$ measures how much the signal changes across edges:

$$
\mathbf{x}^\top\mathbf{L}\,\mathbf{x}=\frac12\sum_{u,v}A_{uv}\,(x_u-x_v)^2 .
$$

It is small when neighbours carry similar values (a smooth signal) and large when the signal flips across many edges. GNNs mostly use the **normalized Laplacian** $\mathbf{L}_{\text{sym}}=\mathbf{I}-\mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2}$, which measures the same thing after dividing each value by $\sqrt{d_v}$, so that high-degree nodes do not dominate.

$\mathbf{L}_{\text{sym}}$ is symmetric and positive semi-definite, so by the spectral theorem it has an orthonormal eigenbasis, $\mathbf{L}_{\text{sym}}=\mathbf{U}\boldsymbol\Lambda\mathbf{U}^\top$, with eigenvalues $0=\lambda_1\le\lambda_2\le\dots\le\lambda_N\le2$. By the [[Eigenvalues and Eigenvectors#The Rayleigh quotient|Rayleigh quotient]], each eigenvalue is the smoothness of its eigenvector: $\mathbf{u}_i^\top\mathbf{L}_{\text{sym}}\mathbf{u}_i=\lambda_i$. So the eigenvectors are the graph's **frequencies**, ordered from smoothest to most oscillating:

- $\lambda_1=0$ belongs to the constant-like vector $\mathbf{D}^{1/2}\mathbf{1}$ (normalized), the "DC component";
- small eigenvalues belong to signals that vary slowly and change sign only across a few edges. The second one, the Fiedler vector, is what [[Clustering and Expectation Maximization#Balanced cuts and their relaxation|spectral clustering]] thresholds;
- large eigenvalues belong to signals that alternate between neighbours.

> [!tip] Intuition: the cycle graph recovers the ordinary Fourier transform
> On a cycle of $N$ nodes, every node has degree 2 and $\mathbf{L}$ is a circulant matrix. Its eigenvectors are the discrete sines and cosines $\cos(2\pi kn/N)$, $\sin(2\pi kn/N)$, with eigenvalues $2-2\cos(2\pi k/N)$, which grow with the frequency $k$. A sequence with wrap-around is a cycle graph, and on it the graph Fourier transform below *is* the discrete Fourier transform.

The **graph Fourier transform** of a signal is its coordinates in this basis, $\hat{\mathbf{x}}=\mathbf{U}^\top\mathbf{x}$, and the inverse transform is $\mathbf{x}=\mathbf{U}\hat{\mathbf{x}}$. A **spectral filter** with frequency response $g(\lambda)$ scales each frequency component:

$$
g(\mathbf{L}_{\text{sym}})\,\mathbf{x}=\mathbf{U}\,g(\boldsymbol\Lambda)\,\mathbf{U}^\top\mathbf{x},\qquad g(\boldsymbol\Lambda)=\operatorname{diag}\big(g(\lambda_1),\dots,g(\lambda_N)\big).
$$

This is the convolution theorem, used as a *definition*: on a graph we cannot slide a kernel, but we can multiply in the frequency domain.

![Top: Zachary's karate club graph coloured by two eigenvectors of its normalized Laplacian. The low-frequency eigenvector u2 is red on one side of the club and blue on the other; the high-frequency eigenvector u34 alternates between neighbouring nodes. Bottom: frequency responses against the eigenvalue. The first-order filter 2 minus lambda runs from 2 down to 0; the renormalized GCN filter 1 minus lambda and its powers squeeze everything except the lowest frequencies towards zero. Tick marks show the eigenvalues of the self-looped karate graph, which end at 1.42.](Machine%20Learning/images/graph-neural-networks-spectral.svg)
*Figure 1: frequencies on a graph. Top: two eigenvectors of $\mathbf{L}_{\text{sym}}$ on Zachary's karate club (red positive, blue negative). The smooth $\mathbf{u}_2$ ($\lambda_2=0.13$) changes sign across few edges; its sign matches the club's eventual split for 32 of the 34 members. The rough $\mathbf{u}_{34}$ ($\lambda_{34}=1.71$) flips across most edges. Bottom: frequency responses of the filters derived in the next section. The renormalized GCN filter $1-\tilde\lambda$ is a low-pass filter; stacking layers raises it to a power and keeps only the lowest frequencies. The ticks are the eigenvalues $\tilde\lambda_i$ of the karate graph with self-loops; the largest is 1.42, below 2.*

## From spectral to spatial convolutions

### Spectral networks

The first spectral GNN learned the frequency response directly: one free parameter per frequency, $g_\theta(\boldsymbol\Lambda)=\operatorname{diag}(\boldsymbol\theta)$ with $\boldsymbol\theta\in\mathbb{R}^N$.[^bruna2014] With several input and output channels, layer $\ell$ maps channel $j$ to channel $i$ with its own filter. This is a faithful translation of convolution, but it has four problems:

1. **Cost.** It needs the full eigendecomposition, $O(N^3)$, and every filter application multiplies by dense $\mathbf{U}$ and $\mathbf{U}^\top$, $O(N^2)$.
2. **Parameters.** Each filter has $N$ parameters, as many as the graph has nodes.
3. **No locality.** A free spectral filter is a dense $N\times N$ matrix in the node domain, so every node talks to every other node. A CNN kernel touches only nearby pixels.
4. **No transfer.** The filter is defined on *this* graph's eigenvectors. Another graph has another basis (and even this basis is only defined up to signs, and up to rotations within repeated eigenvalues), so a filter learned on one molecule means nothing on another.

Bruna et al. already noted that **smooth** frequency responses give filters that are localized in space, which points to the fix.

### Polynomial filters: ChebNet

Restrict the frequency response to a polynomial of degree $K$:

$$
g_\theta(\mathbf{L})=\sum_{k=0}^{K}\theta_k\,\mathbf{L}^k .
$$

Three good things happen at once.

- **Locality.** $(\mathbf{L}^k)_{uv}=0$ whenever the shortest path from $u$ to $v$ has more than $k$ edges, because each multiplication by $\mathbf{L}$ only reaches one hop further. So a degree-$K$ filter is exactly **$K$-hop local**, like a $(2K+1)\times(2K+1)$ kernel on a grid.
- **No eigendecomposition.** $g_\theta(\mathbf{L})\mathbf{x}=\sum_k\theta_k\mathbf{L}^k\mathbf{x}$ needs only $K$ sparse matrix–vector products, $O(K\lvert\mathcal{E}\rvert)$.
- **$K+1$ parameters per filter**, independent of $N$. The same coefficients define a filter on any graph, so the model can transfer.

ChebNet writes the polynomial in the **Chebyshev basis**, which is better conditioned than the monomials $\mathbf{L}^k$.[^defferrard2016] Rescale the spectrum to $[-1,1]$ with $\tilde{\mathbf{L}}=2\mathbf{L}_{\text{sym}}/\lambda_{\max}-\mathbf{I}$ and use

$$
g_\theta(\mathbf{L}_{\text{sym}})\,\mathbf{x}=\sum_{k=0}^{K}\theta_k\,T_k(\tilde{\mathbf{L}})\,\mathbf{x},\qquad T_k(y)=2y\,T_{k-1}(y)-T_{k-2}(y),
$$

with $T_0(y)=1$ and $T_1(y)=y$. The recurrence computes $T_k(\tilde{\mathbf{L}})\mathbf{x}$ from the two previous vectors, so the cost stays $O(K\lvert\mathcal{E}\rvert)$. The same Chebyshev trick was used earlier to build wavelets on graphs.[^hammond2011]

### The first-order approximation: GCN

Kipf and Welling simplified ChebNet as far as it would go and recovered expressiveness by stacking layers, as CNNs stack $3\times3$ kernels.[^kipf2017] The derivation takes five steps.

**Step 1: keep only $K=1$.** The filter becomes linear in $\mathbf{L}_{\text{sym}}$, i.e. each layer looks one hop away. Also approximate $\lambda_{\max}\approx2$ (its upper bound), so $\tilde{\mathbf{L}}=\mathbf{L}_{\text{sym}}-\mathbf{I}$:

$$
g_\theta\star\mathbf{x}\approx\theta_0\,\mathbf{x}+\theta_1(\mathbf{L}_{\text{sym}}-\mathbf{I})\,\mathbf{x}=\theta_0\,\mathbf{x}-\theta_1\,\mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2}\mathbf{x}.
$$

**Step 2: one parameter per filter.** Set $\theta=\theta_0=-\theta_1$ to reduce overfitting and computation:

$$
g_\theta\star\mathbf{x}\approx\theta\,\big(\mathbf{I}+\mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2}\big)\,\mathbf{x}.
$$

Read it in the node domain: the new value at $v$ is $\theta$ times its own value plus a degree-weighted sum of its neighbours' values.

**Step 3: notice the instability.** $\mathbf{I}+\mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2}=2\mathbf{I}-\mathbf{L}_{\text{sym}}$ has eigenvalues $2-\lambda_i\in[0,2]$ (the dashed line in Figure 1). Stacking $L$ layers applies $(2-\lambda_i)^L$ to each frequency: low frequencies are amplified up to $2^L$ and others are crushed, which gives exploding or vanishing activations and gradients in deep models.

**Step 4: the renormalization trick.** Add the self-loops *before* normalizing. With $\tilde{\mathbf{A}}=\mathbf{A}+\mathbf{I}$ and $\tilde{\mathbf{D}}=\operatorname{diag}(\tilde{\mathbf{d}})$, $\tilde d_v=d_v+1$, replace the operator by

$$
\mathbf{I}+\mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2}\ \longrightarrow\ \hat{\mathbf{A}}=\tilde{\mathbf{D}}^{-1/2}\tilde{\mathbf{A}}\tilde{\mathbf{D}}^{-1/2}.
$$

$\hat{\mathbf{A}}=\mathbf{I}-\tilde{\mathbf{L}}_{\text{sym}}$, where $\tilde{\mathbf{L}}_{\text{sym}}$ is the normalized Laplacian of the self-looped graph, so the frequency response is $1-\tilde\lambda$. The proof below shows $0\le\tilde\lambda<2$, so every eigenvalue of $\hat{\mathbf{A}}$ lies in $(-1,1]$: repeated application can no longer blow up. On the karate club, the self-loops shrink the largest eigenvalue from $1.71$ to $1.42$ (Figure 1).

**Step 5: many channels.** With an input of $F$ channels, $\mathbf{H}\in\mathbb{R}^{N\times F}$, and $F'$ output filters, the parameters form a matrix $\mathbf{W}\in\mathbb{R}^{F\times F'}$. Adding a non-linearity gives the **graph convolutional network** layer:

$$
\mathbf{H}^{(\ell+1)}=\sigma\big(\tilde{\mathbf{D}}^{-1/2}\tilde{\mathbf{A}}\tilde{\mathbf{D}}^{-1/2}\,\mathbf{H}^{(\ell)}\,\mathbf{W}^{(\ell)}\big)=\sigma\big(\hat{\mathbf{A}}\mathbf{H}^{(\ell)}\mathbf{W}^{(\ell)}\big). \tag{1}
$$

Row by row, each node takes a weighted average of itself and its neighbours, then applies a shared linear map:

$$
\mathbf{h}_v^{(\ell+1)}=\sigma\Big(\sum_{u\in\mathcal{N}(v)\cup\{v\}}\frac{1}{\sqrt{\tilde d_u\,\tilde d_v}}\,\mathbf{W}^{(\ell)\top}\mathbf{h}_u^{(\ell)}\Big).
$$

$\hat{\mathbf{A}}$ is computed once and stays sparse, so a layer costs $O(\lvert\mathcal{E}\rvert FF')$. For semi-supervised node classification, Kipf and Welling used two layers,

$$
\mathbf{Z}=\operatorname{softmax}\big(\hat{\mathbf{A}}\,\operatorname{ReLU}(\hat{\mathbf{A}}\mathbf{X}\mathbf{W}^{(0)})\,\mathbf{W}^{(1)}\big),
$$

trained with the cross-entropy over the **labelled nodes only**. The unlabelled nodes still take part: their features flow into the labelled nodes' predictions, and the gradient flows back through them.

> [!proof]- Why the renormalized spectrum lies in $(-1,1]$
> Let $\tilde{\mathbf{L}}_{\text{sym}}=\mathbf{I}-\hat{\mathbf{A}}$, take any $\mathbf{x}\ne\mathbf{0}$ and write $y_v=x_v/\sqrt{\tilde d_v}$. Using $\tilde d_v=\sum_u\tilde A_{vu}$,
>
> $$
> \mathbf{x}^\top\tilde{\mathbf{L}}_{\text{sym}}\mathbf{x}=\sum_v\tilde d_v y_v^2-\sum_{u,v}\tilde A_{uv}y_uy_v=\frac12\sum_{u,v}\tilde A_{uv}(y_u-y_v)^2\ \ge 0 .
> $$
>
> In the same way,
>
> $$
> 2\,\mathbf{x}^\top\mathbf{x}-\mathbf{x}^\top\tilde{\mathbf{L}}_{\text{sym}}\mathbf{x}=\frac12\sum_{u,v}\tilde A_{uv}(y_u+y_v)^2\ \ge 0 .
> $$
>
> So the Rayleigh quotient of $\tilde{\mathbf{L}}_{\text{sym}}$ lies in $[0,2]$. Equality at $2$ would need $y_u=-y_v$ for every pair with $\tilde A_{uv}>0$. The self-loops include the pairs $u=v$, which force $y_v=-y_v$, i.e. $\mathbf{y}=\mathbf{0}$. Hence $0\le\tilde\lambda<2$, and the eigenvalues $1-\tilde\lambda$ of $\hat{\mathbf{A}}$ lie in $(-1,1]$. Without self-loops, $\lambda=2$ is reached exactly when the graph has a bipartite component (alternate the signs across the two sides). The same argument with $\tilde{\mathbf{A}}$ replaced by $\mathbf{A}$ shows that $\mathbf{L}_{\text{sym}}$ has eigenvalues in $[0,2]$.

> [!example]- Code: a GCN layer with the renormalized adjacency (checked against the dense formula)
> Graphs are stored as an edge list `edge_index` of shape $(2,\lvert\mathcal{E}\rvert)$ with both directions of every undirected edge; row 0 is the source $u$ and row 1 the target $v$.
> ```python
> import torch
> import torch.nn as nn
>
> def gcn_norm(edge_index, N):
>     """Renormalized adjacency D~^{-1/2} (A + I) D~^{-1/2} as a sparse (N, N) tensor."""
>     loops = torch.arange(N).repeat(2, 1)
>     ei = torch.cat([edge_index, loops], dim=1)          # add self-loops: A~ = A + I
>     deg = torch.zeros(N).index_add_(0, ei[1], torch.ones(ei.shape[1]))   # d~_v
>     w = deg[ei[0]].rsqrt() * deg[ei[1]].rsqrt()         # 1 / sqrt(d~_u d~_v)
>     return torch.sparse_coo_tensor(ei.flip(0), w, (N, N)).coalesce()    # row = target v
>
>
> class GCNLayer(nn.Module):
>     def __init__(self, f_in, f_out):
>         super().__init__()
>         self.lin = nn.Linear(f_in, f_out, bias=False)
>
>     def forward(self, H, A_hat):                        # A_hat from gcn_norm
>         return torch.sparse.mm(A_hat, self.lin(H))      # A_hat H W
> ```
> The test script checks that `gcn_norm` equals $\tilde{\mathbf{D}}^{-1/2}(\mathbf{A}+\mathbf{I})\tilde{\mathbf{D}}^{-1/2}$ computed densely on the karate club, that the layer equals $\hat{\mathbf{A}}\mathbf{X}\mathbf{W}$, that the eigenvalues of $\hat{\mathbf{A}}$ lie in $(-1,1]$ (here $[-0.42,1]$), and that the layer is permutation equivariant.

### GCN is Laplacian smoothing

Equation (1) has a simple reading. Multiplying by $\hat{\mathbf{A}}$ replaces each node's features by a weighted average over its closed neighbourhood: one step of **Laplacian smoothing**, the graph analogue of blurring an image. In the frequency domain it is the filter $1-\tilde\lambda$: it keeps the smooth components ($\tilde\lambda\approx0$) and damps the rough ones (Figure 1).[^li2018] The weights $\mathbf{W}$ then mix channels, as a $1\times1$ convolution does in a CNN.

This explains why GCNs work and when they fail. They work when **neighbours tend to share labels** (homophily: papers cite papers on the same topic, friends share interests), because averaging then removes noise without mixing classes. They fail when that assumption breaks ([[#Heterophily: when neighbours differ]]) or when the averaging is repeated too often ([[#Over-smoothing]]).

The reading is so accurate that the non-linearities between layers are often unnecessary. The **simplified GCN** (SGC) removes them, so the whole network collapses to $\operatorname{softmax}(\hat{\mathbf{A}}^K\mathbf{X}\mathbf{W})$: precompute the smoothed features $\hat{\mathbf{A}}^K\mathbf{X}$ once, then fit logistic regression. It matches GCN on several citation benchmarks and trains orders of magnitude faster.[^wu2019]

### Demo: the karate club

In 1970–72, Zachary recorded which members of a university karate club met outside the club: 34 members and 78 friendships.[^zachary1977] A dispute between the instructor ("Mr. Hi") and the administrator ("Officer") split the club in two. Zachary showed that a minimum cut of the friendship graph predicts who followed whom for all but one member.

Give a GCN only the graph: the features are one-hot node identities, $\mathbf{X}=\mathbf{I}$, so they carry no information except who is who. Label just two nodes, the instructor and the administrator, and train a GCN with two graph layers ($34\to4\to2$, with $\tanh$) and a linear classifier on top, on those two labels. The two-dimensional output of the second GCN layer can be plotted directly (Figure 2).

![Four panels. Top left: the karate club graph with the two factions in blue and orange and the two labelled nodes ringed. Top right: the 2-D embedding of an untrained GCN, where the two factions already occupy different regions. Bottom: the embeddings after 25 and 300 epochs, where the factions are pulled to opposite ends of a line through the two labelled nodes.](Machine%20Learning/images/graph-neural-networks-karate.svg)
*Figure 2: semi-supervised node classification on Zachary's karate club, with one labelled node per faction (ringed). Top right: even an untrained GCN with random Glorot weights places the factions in different regions, because the embedding is a smoothed function of the graph. Bottom: training pulls the two factions apart. Over 20 seeds, the GCN classifies the 32 unlabelled members with $94.2\pm1.1\%$ accuracy; an MLP with the same features and layers gets $49.2\pm8.0\%$, chance level, because one-hot identities say nothing about unseen nodes.*

Three things are worth noticing.

- **The graph does all the work.** The MLP sees the same features and the same labels and learns nothing about the other 32 members. The GCN gets the label of a member from its neighbourhood.
- **Smoothing alone already clusters.** Before any training, a linear classifier on the random GCN embedding separates the factions with $77.5\pm12.8\%$ accuracy (20 seeds, fitted on all labels). Repeated averaging is a cheap form of [[Clustering and Expectation Maximization#Spectral clustering|spectral clustering]].
- **The embedding is one-dimensional at the end.** With two labelled nodes there is only one direction worth learning, so the trained embedding collapses onto a line: an honest reminder that two labels carry very little information.

## Message passing: one framework for spatial GNNs

The GCN layer was derived from spectra, but its node-wise form is *spatial*: gather from neighbours, combine, transform. Gilmer et al. noticed that most GNNs fit one template, **message passing**.[^gilmer2017] In layer $\ell$, every node $v$ does three things:

$$
\begin{aligned}
\mathbf{m}_{uv}&=\phi\big(\mathbf{h}_v^{(\ell)},\mathbf{h}_u^{(\ell)},\mathbf{e}_{uv}\big),\quad u\in\mathcal{N}(v),\\
\mathbf{m}_v&=\bigoplus_{u\in\mathcal{N}(v)}\mathbf{m}_{uv},\\
\mathbf{h}_v^{(\ell+1)}&=\psi\big(\mathbf{h}_v^{(\ell)},\mathbf{m}_v\big).
\end{aligned}
$$

- The **message** function $\phi$ computes what neighbour $u$ sends to $v$, possibly using an edge feature $\mathbf{e}_{uv}$ (a bond type, a distance).
- The **aggregator** $\bigoplus$ combines the incoming messages. It must be **permutation invariant** and accept any number of inputs: sum, mean, max, or an attention-weighted sum.
- The **update** function $\psi$ combines the old state with the aggregated message.

$\phi$ and $\psi$ are small neural networks **shared by all nodes** (weight sharing), and the aggregator does not care about the order of the neighbours. Together these two facts make the layer permutation equivariant. For graph-level tasks a **readout** $\mathbf{h}_{\mathcal{G}}=\bigoplus_v\mathbf{h}_v^{(L)}$, again a permutation-invariant function, makes the network invariant. The idea of a network that iterates neighbour-to-neighbour updates goes back to the original "graph neural network" of Scarselli et al., which ran one shared update to a fixed point.[^scarselli2009]

> [!proof]- Why message passing is permutation equivariant
> Relabel the nodes by a bijection $\pi$. Node $\pi(v)$ in the new graph has the old state of $v$, and its neighbourhood is $\{\pi(u):u\in\mathcal{N}(v)\}$ with the same edge features. Its messages are therefore the same vectors as those $v$ received, possibly listed in a different order. The aggregator ignores order, so $\mathbf{m}_{\pi(v)}$ equals the old $\mathbf{m}_v$, and the shared update gives $\mathbf{h}_{\pi(v)}^{(\ell+1)}=\mathbf{h}_v^{(\ell+1)}$. In matrix form this is $f(\mathbf{P}\mathbf{X},\mathbf{P}\mathbf{A}\mathbf{P}^\top)=\mathbf{P}f(\mathbf{X},\mathbf{A})$. By induction it holds for any number of layers, and an invariant readout of an equivariant map is invariant. The test script checks this numerically for every layer in this lecture with a random permutation of the karate club.

After $L$ layers, $\mathbf{h}_v^{(L)}$ depends exactly on the $L$-hop neighbourhood of $v$. Unrolling the recursion gives the **computation tree** of $v$ (Figure 3): $v$ at the root, its neighbours below, their neighbours below those. A GNN is a neural network whose architecture is different for every node, built from that node's tree, but whose weights are shared by all trees in a layer.

![Left: a six-node graph with target node A in the centre, neighbours B, C and D, and two further nodes E and F. Right: the computation tree of A for two layers. A is at the top, B, C and D are its children, and each of those has its own neighbours as children, including A itself and repeated nodes.](Machine%20Learning/images/graph-neural-networks-computation-tree.svg)
*Figure 3: a two-layer GNN computes node A's representation on its computation tree. Each arrow is a message; each internal node aggregates its children and applies the layer's shared update (self-loops are omitted for clarity). Nodes repeat (A appears again under each of its neighbours, and C appears both as A's neighbour and as B's), and the tree grows with depth roughly like (degree)$^L$: the root of both over-squashing and the cost of training on large graphs.*

How the three architectures we meet next fill the template:

| Model | Message from $u$ to $v$ | Aggregator | Update |
|---|---|---|---|
| CNN on a grid (for comparison) | $\mathbf{W}_{\text{pos}(u)}\mathbf{h}_u$, a separate weight per relative position | sum over the kernel window | add bias, non-linearity |
| GCN | $\mathbf{W}^\top\mathbf{h}_u/\sqrt{\tilde d_u\tilde d_v}$, fixed weights from degrees | sum over $\mathcal{N}(v)\cup\{v\}$ | non-linearity |
| GraphSAGE | $\mathbf{h}_u$ (or an MLP of it) | mean, max-pool or LSTM over a sample of $\mathcal{N}(v)$ | $\sigma(\mathbf{W}[\mathbf{h}_v\,\Vert\,\mathbf{m}_v])$, normalize |
| GAT | $\alpha_{vu}\mathbf{W}\mathbf{h}_u$, learned attention weights | sum over $\mathcal{N}(v)\cup\{v\}$ | non-linearity, concatenate heads |
| GIN | $\mathbf{h}_u$ | sum | $\mathrm{MLP}\big((1+\epsilon)\mathbf{h}_v+\mathbf{m}_v\big)$ |
| Transformer (for comparison) | $\alpha_{vu}\mathbf{W}_V\mathbf{h}_u$, dot-product attention | sum over **all** tokens | residual, MLP |

The CNN row shows what is lost: on a grid each neighbour has a position, so each gets its own weight matrix. In a graph the neighbours are anonymous, so all share one $\mathbf{W}$ and differ only through their states and, in GAT, their learned attention weights.

### GraphSAGE: sample and aggregate

GraphSAGE was designed for graphs that are huge and keep changing (new users and new posts every minute).[^hamilton2017] Its layer is

$$
\begin{aligned}
\mathbf{m}_v&=\operatorname{AGG}\big(\{\mathbf{h}_u^{(\ell)}:u\in\mathcal{S}(v)\}\big),\\
\mathbf{h}_v^{(\ell+1)}&=\sigma\big(\mathbf{W}^{(\ell)}\,[\,\mathbf{h}_v^{(\ell)}\,\Vert\,\mathbf{m}_v\,]\big),
\end{aligned}
$$

followed by normalizing $\mathbf{h}_v^{(\ell+1)}$ to unit length. Three design choices differ from GCN.

- **Self and neighbours are kept apart.** Concatenation, equivalently separate weight matrices $\mathbf{W}_{\text{self}}\mathbf{h}_v+\mathbf{W}_{\text{neigh}}\mathbf{m}_v$, lets the model treat a node's own features differently from its neighbourhood's. GCN mixes them in one average. This matters a lot under heterophily (Figure 6).
- **Aggregators.** *Mean* is close to GCN. *Max-pooling*, $\max_u\sigma(\mathbf{W}_{\text{pool}}\mathbf{h}_u+\mathbf{b})$ elementwise, picks out the strongest feature in the neighbourhood. *LSTM* is the most expressive but is not permutation invariant, so it is applied to a random permutation of the neighbours.
- **Sampling.** $\mathcal{S}(v)$ is a fixed-size uniform sample of the neighbours, redrawn every iteration (for example 25 in the first layer and 10 in the second). The cost per node is then bounded no matter how popular it is ([[#Training on large graphs]]).

GraphSAGE can be trained with labels, or without them by a loss that makes nodes that co-occur on short random walks similar and random pairs dissimilar: the DeepWalk objective of [[Graph Representation Learning]], now applied to the output of an encoder instead of a lookup table.

> [!example]- Code: message passing with `index_add_` (GraphSAGE and GIN, checked)
> Messages live on edges: `H[u]` gathers the source state of every edge, and `index_add_` sums them into their targets. Mean and max are built the same way.
> ```python
> import torch
> import torch.nn as nn
>
> def aggregate(msg, target, N, how="sum"):
>     """Permutation-invariant aggregation of edge messages msg (E, F) at their target nodes."""
>     out = torch.zeros(N, msg.shape[1]).index_add_(0, target, msg)
>     if how == "mean":
>         cnt = torch.zeros(N).index_add_(0, target, torch.ones(len(target)))
>         out = out / cnt.clamp(min=1).unsqueeze(1)
>     elif how == "max":
>         out = torch.full((N, msg.shape[1]), -torch.inf).index_reduce_(
>             0, target, msg, "amax")
>         out = torch.where(torch.isinf(out), torch.zeros_like(out), out)  # isolated nodes
>     return out
>
>
> class SAGELayer(nn.Module):
>     """GraphSAGE: h_v' = W_self h_v + W_neigh AGG{h_u : u in N(v)}."""
>     def __init__(self, f_in, f_out, how="mean"):
>         super().__init__()
>         self.self_lin, self.neigh_lin = nn.Linear(f_in, f_out), nn.Linear(f_in, f_out, bias=False)
>         self.how = how
>
>     def forward(self, H, edge_index):
>         u, v = edge_index
>         m = aggregate(H[u], v, H.shape[0], self.how)    # message = neighbour's state
>         return self.self_lin(H) + self.neigh_lin(m)
>
>
> class GINLayer(nn.Module):
>     """GIN: h_v' = MLP((1 + eps) h_v + sum_{u in N(v)} h_u)."""
>     def __init__(self, f_in, f_out):
>         super().__init__()
>         self.eps = nn.Parameter(torch.zeros(1))
>         self.mlp = nn.Sequential(nn.Linear(f_in, f_out), nn.ReLU(), nn.Linear(f_out, f_out))
>
>     def forward(self, H, edge_index):
>         u, v = edge_index
>         return self.mlp((1 + self.eps) * H + aggregate(H[u], v, H.shape[0], "sum"))
> ```
> The test script compares the sum, mean and max aggregators with an explicit loop over each node's neighbours on the karate club, and checks that both layers are permutation equivariant. Neighbour sampling would replace `edge_index` by a freshly sampled subset of edges at every step.

### GAT: attention over neighbours

In GCN the weight of neighbour $u$ in $v$'s average, $1/\sqrt{\tilde d_u\tilde d_v}$, is fixed by the degrees. The **graph attention network** learns it from the content, like [[Sequence Models#Attention as a soft dictionary lookup|attention]] in a transformer but only over the edges that exist.[^velickovic2018] With a shared linear map $\mathbf{W}$ and an attention vector $\mathbf{a}$:

$$
\begin{aligned}
e_{vu}&=\operatorname{LeakyReLU}\big(\mathbf{a}^\top[\,\mathbf{W}\mathbf{h}_v\,\Vert\,\mathbf{W}\mathbf{h}_u\,]\big),\\
\alpha_{vu}&=\frac{\exp(e_{vu})}{\sum_{w\in\mathcal{N}(v)\cup\{v\}}\exp(e_{vw})},\\
\mathbf{h}_v^{(\ell+1)}&=\sigma\Big(\sum_{u\in\mathcal{N}(v)\cup\{v\}}\alpha_{vu}\,\mathbf{W}\mathbf{h}_u\Big).
\end{aligned}
$$

As in [[Sequence Models#Multi-head attention|multi-head attention]], several heads run in parallel with their own $\mathbf{W}$ and $\mathbf{a}$; hidden layers concatenate them and the output layer averages them. The softmax runs over a node's neighbours, so the cost is $O(\lvert\mathcal{E}\rvert)$, not $O(N^2)$.

> [!warning] GAT's attention is "static"
> Split $\mathbf{a}=[\mathbf{a}_1;\mathbf{a}_2]$. Then $e_{vu}=\operatorname{LeakyReLU}(\mathbf{a}_1^\top\mathbf{W}\mathbf{h}_v+\mathbf{a}_2^\top\mathbf{W}\mathbf{h}_u)$. LeakyReLU is monotone and the first term is the same for all of $v$'s neighbours, so the *ranking* of neighbours is fixed by $\mathbf{a}_2^\top\mathbf{W}\mathbf{h}_u$ alone: if node $u^\star$ gets the highest score from one query, it gets the highest score from every query that sees it. Brody et al. call this **static attention** and fix it with **GATv2**, which applies the non-linearity before the dot product, $e_{vu}=\mathbf{a}^\top\operatorname{LeakyReLU}(\mathbf{W}[\mathbf{h}_v\Vert\mathbf{h}_u])$. The transformer's dot-product attention $\mathbf{q}_v^\top\mathbf{k}_u$ does not have this problem.[^brody2022]

> [!example]- Code: a multi-head GAT layer on an edge list (checked against dense masked attention)
> ```python
> import torch
> import torch.nn as nn
> import torch.nn.functional as F
>
> def edge_softmax(scores, target, N):
>     """Softmax of edge scores (E, heads) over the incoming edges of each target node."""
>     mx = torch.full((N, scores.shape[1]), -torch.inf).index_reduce_(0, target, scores, "amax")
>     e = torch.exp(scores - mx[target])                  # subtract per-node max for stability
>     return e / torch.zeros_like(mx).index_add_(0, target, e)[target]
>
>
> class GATLayer(nn.Module):
>     def __init__(self, f_in, f_out, heads=4, concat=True):
>         super().__init__()
>         self.W = nn.Linear(f_in, heads * f_out, bias=False)
>         self.a_dst = nn.Parameter(0.1 * torch.randn(heads, f_out))  # a = [a_dst ; a_src]
>         self.a_src = nn.Parameter(0.1 * torch.randn(heads, f_out))
>         self.heads, self.f_out, self.concat = heads, f_out, concat
>
>     def forward(self, H, edge_index):
>         N = H.shape[0]
>         loops = torch.arange(N).repeat(2, 1)
>         u, v = torch.cat([edge_index, loops], dim=1)    # attend to N(v) and v itself
>         Z = self.W(H).view(N, self.heads, self.f_out)   # W h, per head
>         s = F.leaky_relu((Z * self.a_dst).sum(-1)[v] + (Z * self.a_src).sum(-1)[u], 0.2)
>         alpha = edge_softmax(s, v, N)                   # (E, heads), sums to 1 per v
>         out = torch.zeros_like(Z).index_add_(0, v, alpha.unsqueeze(-1) * Z[u])
>         self.alpha = alpha.detach()
>         return out.reshape(N, -1) if self.concat else out.mean(1)
> ```
> The test checks that the attention weights sum to 1 over each closed neighbourhood, that the output equals a dense $N\times N$ softmax with non-edges masked to $-\infty$, and that the layer is permutation equivariant.

## Transductive and inductive learning with GNNs

[[Graph Representation Learning]] distinguished **transductive** methods, which embed the nodes of one fixed graph, from **inductive** ones, which can embed nodes they never saw. Shallow embeddings such as DeepWalk and node2vec are transductive by construction: they learn a lookup table $\mathbf{Z}\in\mathbb{R}^{N\times d}$ with one free row per node, so a new node has no row and the table must be retrained.

A GNN replaces the table by a **function** of a node's features and neighbourhood, with parameters shared across nodes. In principle that makes every GNN inductive. In practice it depends on how the model is trained and fed:

- **GCN as originally used is transductive.** Training is full-batch on one fixed graph: the test nodes are present during training (with features, without labels), the normalization $\hat{\mathbf{A}}$ uses the whole graph, and the reported results are on that same graph. The learned $\mathbf{W}^{(\ell)}$ can be applied to a new graph, but nothing in the training encouraged it to transfer.
- **With identity features a GNN is a shallow embedding again.** If $\mathbf{X}=\mathbf{I}$, as in the karate demo, the first layer's $\mathbf{W}^{(0)}$ has one row per node: it *is* a lookup table, and the model cannot handle a new node.
- **GraphSAGE is inductive by design.** It uses real node features, samples neighbourhoods during training so that it never relies on the full graph, and was evaluated on nodes and even whole graphs (protein–protein interaction networks of other tissues) unseen in training.

Inductive training matters whenever the graph changes (new users, new papers) or when one model must serve many graphs (molecules).

## How powerful are GNNs?

A GNN computes a node's representation from its computation tree. Two nodes with identical trees, features included, must get identical representations, whatever the weights. So what can message passing tell apart, and what can it not?

### The Weisfeiler–Lehman test

Deciding whether two graphs are isomorphic (the same up to relabelling) has no known polynomial-time algorithm in general. A classic heuristic is **colour refinement**, the 1-dimensional Weisfeiler–Lehman test (1-WL):

1. Give every node the same colour (or a colour from its features).
2. Repeat: give each node a new colour that encodes its old colour together with the **multiset** of its neighbours' colours, $c_v^{(t+1)}=\operatorname{hash}\big(c_v^{(t)},\{\!\{c_u^{(t)}:u\in\mathcal{N}(v)\}\!\}\big)$, where the hash is injective.
3. Stop when the colour classes stop splitting. If the two graphs end with different colour histograms they are certainly **not** isomorphic; if the histograms agree, the test cannot tell.

1-WL distinguishes most graphs one meets, and it decides isomorphism for all trees. The colour counts after a few rounds are also the features of the popular WL graph kernel.[^shervashidze2011] But it fails on simple examples (Figure 4):

- any two **regular** graphs with the same number of nodes and the same degree. A hexagon and two disjoint triangles are both 2-regular on 6 nodes: every node sees "my colour, plus two neighbours of my colour" forever, so 1-WL cannot tell a connected graph from a disconnected one;
- **decalin** and **bicyclopentyl**: two fused hexagons versus two pentagons joined by an edge. Both have 10 nodes and 11 edges, and 1-WL gives them the same three colour classes.

> [!example]- Code: 1-WL colour refinement (checked)
> ```python
> def wl_colours(adj, rounds=None):
>     """1-WL colour refinement. adj: list of neighbour lists. Returns the stable colour
>     of every node (comparable across graphs only when they are refined jointly)."""
>     col = [0] * len(adj)                                # every node starts with one colour
>     for _ in range(rounds or len(adj)):
>         sig = [(col[v], tuple(sorted(col[u] for u in adj[v]))) for v in range(len(adj))]
>         palette = {s: i for i, s in enumerate(sorted(set(sig)))}
>         new = [palette[s] for s in sig]                 # injective relabelling ("hash")
>         if len(set(new)) == len(set(col)):              # partition stopped refining
>             break
>         col = new
>     return col
>
>
> def wl_test(adj1, adj2):
>     """Run 1-WL on the disjoint union so colours are shared; compare the histograms."""
>     n1 = len(adj1)
>     union = [list(a) for a in adj1] + [[u + n1 for u in a] for a in adj2]
>     col = wl_colours(union)
>     h1, h2 = sorted(col[:n1]), sorted(col[n1:])
>     return h1 == h2                                     # True = "possibly isomorphic"
> ```
> The test checks that the hexagon and two triangles, and decalin and bicyclopentyl, pass as "possibly isomorphic" although they are not, that a path and a star on 4 nodes are told apart, and that on 200 random pairs of 8-node trees the test agrees with exact isomorphism every time.

### GNNs are at most as powerful as 1-WL

Compare the WL update with a message-passing layer: both combine a node's own state with an order-free summary of its neighbours' states. Xu et al. and Morris et al. proved independently that the analogy is exact in one direction:[^xu2019]<sup>,</sup>[^morris2019]

> **Theorem.** If 1-WL cannot distinguish two graphs, then no message-passing GNN maps them to different graph embeddings. A message-passing GNN is exactly as powerful as 1-WL if its aggregator, update and readout are **injective** functions of their (multiset) inputs.

> [!proof]- Proof sketch of the upper bound
> Show by induction on $t$ that, for any weights, $\mathbf{h}_v^{(t)}$ is a function of the WL colour $c_v^{(t)}$: equal colours imply equal states. At $t=0$ both come from the same input features. Suppose it holds at $t$, and let $c_v^{(t+1)}=c_w^{(t+1)}$. Since the WL hash is injective, $v$ and $w$ had the same colour and the same multiset of neighbour colours at step $t$. By the induction hypothesis they then had the same state and the same multiset of neighbour states. The aggregator only sees that multiset and the update only sees the state and the aggregate, so $\mathbf{h}_v^{(t+1)}=\mathbf{h}_w^{(t+1)}$. Finally, if two graphs have the same colour histogram, they have the same multiset of node states, and any readout over that multiset gives the same output. The converse needs every step to lose no information, which is what injectivity provides.

### GIN: an injective aggregator

Which aggregators lose information? Think of the neighbour states as a **multiset** (a set with multiplicities):

- **mean** sees only the *distribution*: $\{\!\{a,b\}\!\}$ and $\{\!\{a,a,b,b\}\!\}$ have the same mean;
- **max** sees only the *underlying set*: $\{\!\{a,b\}\!\}$ and $\{\!\{a,a,b\}\!\}$ have the same maximum;
- **sum** sees the multiset. Xu et al. show that for features from a countable set there is a function $f$ that makes $\sum_{u}f(\mathbf{h}_u)$ injective on bounded multisets.

With constant input features, mean and max cannot even count neighbours, so a mean-GNN cannot see degrees. The **graph isomorphism network** (GIN) uses the sum and lets an MLP play the role of $f$ and of the injective update (MLPs are universal approximators):

$$
\mathbf{h}_v^{(\ell+1)}=\mathrm{MLP}^{(\ell)}\Big((1+\epsilon^{(\ell)})\,\mathbf{h}_v^{(\ell)}+\sum_{u\in\mathcal{N}(v)}\mathbf{h}_u^{(\ell)}\Big).
$$

The $(1+\epsilon)$ factor keeps a node's own state distinguishable from a neighbour's. For graph classification GIN reads out with a sum at **every** layer and concatenates the results, since early layers capture small substructures that later layers may blur.

Figure 4 tests both directions of the theorem on two synthetic graph-classification tasks (12-node graphs, constant node features, three layers, readout with the layer's own aggregator).

![Top: decalin (two fused hexagons) and bicyclopentyl (two pentagons joined by an edge), each node coloured by its stable 1-WL colour; both graphs show the same three colours in the same numbers. Bottom: bar chart of test accuracy for four models on two tasks. On 3-regular versus irregular graphs, GIN and GIN with random-walk encodings reach 100 percent while mean and max aggregation stay at 50 percent. On one 12-cycle versus several shorter cycles, the three plain models stay at 50 percent and only GIN with random-walk encodings reaches 100 percent.](Machine%20Learning/images/graph-neural-networks-wl.svg)
*Figure 4: what message passing can and cannot see. Top: decalin and bicyclopentyl are not isomorphic but get identical 1-WL colourings, so every message-passing GNN embeds them identically. Bottom: test accuracy (mean ± sd over 5 seeds; 400 training and 200 test graphs). **Task 1**, 3-regular graphs versus irregular graphs with the same 12 nodes and 18 edges: 1-WL separates them by degree; GIN does too ($99.9\pm0.2\%$), while mean ($50.0\pm0.0\%$) and max ($50.0\pm0.0\%$) aggregation cannot even count neighbours. **Task 2**, a single 12-cycle versus a union of shorter cycles: every graph is 2-regular, 1-WL fails, and so does every plain GNN ($50.0\pm0.0\%$ for GIN). Adding random-walk structural encodings (below) breaks the symmetry: GIN with RWSE reaches $100.0\pm0.0\%$ on Task 2 (and $99.8\pm0.4\%$ on Task 1).*

Expressiveness is not everything. GIN can tell more graphs apart, but on node classification with informative features the mean aggregator of GCN and GraphSAGE is often just as accurate and more stable, because it normalizes away degree differences.

### Beyond 1-WL

Being bounded by 1-WL means, for example, that message-passing GNNs cannot count triangles or longer cycles in general, a real limitation for chemistry where rings matter.[^chen2020count] Ways past the bound:

- **Higher-order GNNs** pass messages between pairs or $k$-tuples of nodes, mirroring the $k$-WL hierarchy. They are strictly more powerful, but cost $O(N^k)$.[^morris2019]
- **Random node features** make every node distinguishable, so even regular graphs can be told apart. The price is that the output is now random, and invariance only holds in distribution.[^sato2021]
- **Structural and positional encodings**, such as the random-walk encodings that solve Task 2 in Figure 4, add features computed from the graph that 1-WL cannot compute ([[#Graph transformers]]).

## Pitfalls of depth and of assumptions

### Over-smoothing

A CNN gets better with depth; a GCN usually peaks at two or three layers. The reason is in the Laplacian-smoothing reading: blur an image often enough and it becomes a uniform grey.

Make it precise. $\hat{\mathbf{A}}$ is similar to the random-walk matrix $\mathbf{P}=\tilde{\mathbf{D}}^{-1}\tilde{\mathbf{A}}$ of the self-looped graph, $\hat{\mathbf{A}}=\tilde{\mathbf{D}}^{1/2}\mathbf{P}\,\tilde{\mathbf{D}}^{-1/2}$. On a connected graph the walk is irreducible, and the self-loops make it aperiodic, so by the [[Markov Chains#Convergence and the ergodic theorem|ergodic theorem]] $\mathbf{P}^k\to\mathbf{1}\boldsymbol\pi^\top$, where $\pi_v=\tilde d_v/\sum_u\tilde d_u$ is the stationary distribution. Therefore

$$
\hat{\mathbf{A}}^k\mathbf{X}\ \xrightarrow{\ k\to\infty\ }\ \frac{\mathbf{s}\,\mathbf{s}^\top\mathbf{X}}{\mathbf{s}^\top\mathbf{s}},\qquad \mathbf{s}=\tilde{\mathbf{D}}^{1/2}\mathbf{1}.
$$

Every row of the limit is the same vector $\mathbf{s}^\top\mathbf{X}/\mathbf{s}^\top\mathbf{s}$ scaled by $\sqrt{\tilde d_v}$: the nodes keep their degree and nothing else. In spectral terms, $\hat{\mathbf{A}}^k$ keeps only the eigenvector with eigenvalue 1 and damps every other component like $\lvert1-\tilde\lambda_i\rvert^k$. The speed is set by the [[Markov Chains#Aside: mixing and the spectral gap|spectral gap]]: on the karate club the second-largest $\lvert1-\tilde\lambda_i\rvert$ is $0.896$, so after 32 steps the informative components have shrunk to $0.896^{32}\approx3\%$ of their size.

This is **over-smoothing**:[^li2018] node representations become indistinguishable as depth grows. Oono and Suzuki showed that with ReLU and weight matrices of bounded norm, a deep GCN's output approaches this information-free subspace exponentially fast,[^oono2020] and Cai and Wang track it with the Dirichlet energy $\operatorname{tr}(\mathbf{H}^\top\tilde{\mathbf{L}}_{\text{sym}}\mathbf{H})$, which decays to zero.[^cai2020]

Remedies keep some of each node's own signal from being averaged away:

- **Residual connections**, $\mathbf{H}^{(\ell+1)}=\mathbf{H}^{(\ell)}+\sigma(\hat{\mathbf{A}}\mathbf{H}^{(\ell)}\mathbf{W}^{(\ell)})$, as in ResNets. **GCNII** adds an *initial* residual to $\mathbf{H}^{(0)}$ and keeps each weight matrix close to the identity, and trains 64-layer GCNs.[^chen2020gcnii] **Jumping knowledge** networks combine the outputs of all layers, so each node can use the depth that suits it.[^xu2018jk]
- **Normalization.** PairNorm rescales the features after every layer so that the total pairwise distance between nodes stays constant.[^zhao2020]
- **Decoupling propagation from transformation** (APPNP), described next.

**APPNP.** Compute $\mathbf{H}=f_\theta(\mathbf{X})$ with an MLP, then propagate with a restart:[^gasteiger2019]

$$
\mathbf{Z}^{(0)}=\mathbf{H},\qquad \mathbf{Z}^{(k+1)}=(1-\alpha)\,\hat{\mathbf{A}}\mathbf{Z}^{(k)}+\alpha\,\mathbf{H}.
$$

The spectral radius of $(1-\alpha)\hat{\mathbf{A}}$ is $1-\alpha<1$, so the iteration converges, to

$$
\mathbf{Z}^{(\infty)}=\alpha\big(\mathbf{I}-(1-\alpha)\hat{\mathbf{A}}\big)^{-1}\mathbf{H}=\sum_{k\ge0}\alpha(1-\alpha)^k\,\hat{\mathbf{A}}^k\mathbf{H}.
$$

This is **personalized PageRank**: a random walker that at every step [[Markov Chains#PageRank: the stationary distribution of the web|teleports]] back to its start node with probability $\alpha$. The weights $\alpha(1-\alpha)^k$ on $k$-hop information decay geometrically, and the node's own features always keep weight $\alpha$, so infinitely many propagation steps are safe. Only the MLP has parameters, so the depth of propagation costs no parameters at all.

> [!example]- Code: APPNP propagation (after `appnp_propagate` in version 6 of the repository)
> ```python
> import torch
>
> def appnp_propagate(H, A_hat, K=10, alpha=0.1):
>     """Z <- (1 - alpha) A_hat Z + alpha H, K times. H = MLP(X); A_hat from gcn_norm."""
>     Z = H
>     for _ in range(K):
>         Z = (1 - alpha) * torch.sparse.mm(A_hat, Z) + alpha * H   # restart to H with prob. alpha
>     return Z
>
> # the model: logits = appnp_propagate(mlp(X), A_hat); only the MLP has parameters
> ```
> After $K$ steps the distance to the closed form $\alpha(\mathbf{I}-(1-\alpha)\hat{\mathbf{A}})^{-1}\mathbf{H}$ has shrunk by at least the factor $(1-\alpha)^K$, which is $0.35$ for $\alpha=0.1$ and $K=10$, typical settings. In practice it shrinks much faster: the components that converge slowest, those along eigenvalues of $\hat{\mathbf{A}}$ near 1, start almost at their limit. On the karate club with random features the remaining error after 10 steps is about 1% of the initial one.

![Left: test accuracy against depth from 1 to 64 for GCN, GCN with residual connections and APPNP on a stochastic block model, with an MLP baseline line. GCN peaks at 2 layers and is at chance from 8 layers on; the residual GCN stays above 80 percent up to 32 layers and collapses at 64; APPNP stays near 76 percent at every depth. Right: without training, the mean cosine similarity between node feature vectors rises towards 1 as the raw features are propagated by the GCN operator, and a linear probe's accuracy rises then falls; under APPNP propagation the similarity levels off and the accuracy stays high.](Machine%20Learning/images/graph-neural-networks-oversmoothing.svg)
*Figure 5: over-smoothing on a 4-block stochastic block model (400 nodes, mean degree 8, edge homophily 0.63, noisy 16-D Gaussian features, 20 labelled nodes per class). Left: test accuracy against depth, mean ± sd over 5 seeds. Plain GCN peaks at 2 layers ($83.1\pm3.6\%$) and is at chance (25%) from 8 layers on. With residual connections, accuracy stays at 80–85% up to 32 layers, but this simple, unnormalized version fails to train at 64 ($25.2\pm0.9\%$). APPNP, whose propagation has no parameters, settles at $75.6\pm1.4\%$ for any depth. An MLP that ignores the graph gets $49.7\pm3.3\%$. Right: no training, just repeated propagation of the raw features. With $\hat{\mathbf{A}}^k\mathbf{X}$ (solid orange), the mean cosine similarity between nodes rises from 0 to 0.96 after 16 steps and to 1.00 after 32. A logistic-regression probe (dashed) peaks at 83% after 4 steps and falls to chance (24%) by 64. Under APPNP propagation with $\alpha=0.1$ (green), the similarity stays near 0 and the probe stays at 75%.*

The stochastic block model used here is the standard synthetic graph with communities: nodes in the same block are connected with probability $p_{\text{in}}$ and nodes in different blocks with $p_{\text{out}}<p_{\text{in}}$.[^holland1983]

> [!note] Over-smoothing is not the only reason deep GNNs fail
> A plain 64-layer GCN is also a 64-layer network without skip connections, so it suffers from the [[Neural Networks#Vanishing and exploding gradients|vanishing gradients]] of any deep plain network. Part of what residual connections fix in Figure 5 is optimization, not smoothing. APPNP separates the two: its propagation is deep but its trainable part is a two-layer MLP.

### Over-squashing

Over-smoothing is too much mixing; **over-squashing** is too little room. A node's computation tree grows roughly like $d^L$ with depth, and everything in it must be compressed into one fixed-size vector $\mathbf{h}_v$. When a task needs information from far away (a long-range interaction in a molecule, a distant node in a program graph), the signal from that node competes with exponentially many others on the way. Alon and Yahav showed that GNNs fail at a synthetic tree task that needs exactly this, and that making the last layer **fully adjacent** (every node can message every other) already helps on real benchmarks.[^alon2021]

Topping et al. made the bottleneck geometric. For a message-passing network whose layer derivatives are bounded by a constant $c$, the sensitivity of node $v$ after $r$ layers to the input at node $u$ obeys[^topping2022]

$$
\left\lVert\frac{\partial\mathbf{h}_v^{(r)}}{\partial\mathbf{x}_u}\right\rVert\le c^{\,r}\,\big(\hat{\mathbf{A}}^r\big)_{vu}.
$$

$(\hat{\mathbf{A}}^r)_{vu}$ is small when all $r$-step paths from $u$ to $v$ squeeze through a few edges, a **bottleneck** such as the single bridge between two communities. They identify such edges by their negative discrete (Ricci) curvature and **rewire** the graph around them. Graph transformers attack the same problem by letting every node attend to every other.

### Heterophily: when neighbours differ

GCN's averaging assumes **homophily**. Measure it with the **edge homophily ratio**, the fraction of edges that join nodes of the same class:[^zhu2020]

$$
h=\frac{\big\lvert\{(u,v)\in\mathcal{E}:y_u=y_v\}\big\rvert}{\lvert\mathcal{E}\rvert}.
$$

Standard citation networks have $h\approx0.74$–$0.81$. But many graphs are **heterophilous**: fraudsters connect to their victims, in dating networks people often link to the other gender, and in proteins amino acids of different types bind. Averaging a node with neighbours of other classes erases exactly the information we need.

Low homophily does not mean the graph is useless. With two classes and $h=0$, every neighbour is from the *other* class, which is as informative as possible: even GCN's average is then dominated by the other class and simply flips sign (Figure 6). The damage is done by **mixed** neighbourhoods, where averaging blends the classes, and by many classes, where "not my class" says little. Zhu et al. identify three designs that help: (1) **separate ego and neighbour embeddings**, as GraphSAGE's concatenation does; (2) use **higher-order neighbourhoods**, since two-hop neighbours are often of the same class again; (3) **combine intermediate representations** from all layers.[^zhu2020] Another fix is to learn the filter: GPR-GNN learns the coefficients of a polynomial $\sum_k\gamma_k\hat{\mathbf{A}}^k$, and negative coefficients give **high-pass** filters that emphasize differences between neighbours.[^chien2021]

![Test accuracy against edge homophily from 0 to 1 on two-class stochastic block models. The MLP is flat at about 78 percent. GCN and GraphSAGE are near 100 percent at both ends. In the middle GCN drops to about 64 percent, well below the MLP, while GraphSAGE with separate self and neighbour weights dips only to the MLP's level.](Machine%20Learning/images/graph-neural-networks-heterophily.svg)
*Figure 6: homophily sweep on two-class stochastic block models (400 nodes, mean degree 10, 16-D Gaussian features, 80 labelled nodes; mean ± sd over 5 seeds). The MLP ignores the edges and stays at 77–81%. GCN is near perfect at both ends: with two classes, a neighbourhood made only of the other class is as informative as one made only of the same class. In between, its averaging mixes the classes, and at $h=0.4$–$0.5$ it falls to $64\pm4\%$, well below the MLP. GraphSAGE keeps a node's own features apart from the neighbour mean and stays at the MLP's level even there ($77.9\pm2.3\%$ at $h=0.5$, against $78.6\pm2.9\%$ for the MLP).*

A caution: several popular heterophily benchmarks turned out to contain duplicated nodes and train–test leakage, and on cleaned datasets standard GNNs with ego–neighbour separation are competitive with specialized architectures.[^platonov2023]

## Training on large graphs

On a grid, a mini-batch of images is easy. On a graph, the nodes are not independent: to compute $L$ layers for one node we need its whole $L$-hop neighbourhood, which grows like $d^L$ (Figure 3). On a social network with average degree 100, a three-layer model would touch a million nodes per target node. Full-batch training, as in the original GCN, keeps every layer's $N\times F$ activations in memory, which is impossible for graphs with billions of edges. The main fixes:

- **Neighbour sampling** (GraphSAGE): sample at most $S_\ell$ neighbours per node in layer $\ell$. The tree then has at most $\prod_\ell S_\ell$ leaves, e.g. $25\times10=250$, no matter how dense the graph is. The aggregate becomes a noisy but unbiased estimate of the mean over all neighbours.[^hamilton2017]
- **Cluster-GCN**: partition the graph into dense clusters (the paper uses the METIS partitioner; community detection such as [[Clustering and Expectation Maximization#Communities in networks: modularity and the Louvain method|Louvain]] serves the same purpose) and train on one subgraph at a time. Most edges fall inside clusters, so little is lost; picking several random clusters per batch restores the edges between them and reduces the bias.[^chiang2019]
- **GraphSAINT**: sample a subgraph per step (by nodes, edges or random walks), run the full GNN on it, and reweight the aggregation and the loss by the sampling probabilities so that the estimates are unbiased.[^zeng2020]
- **Precompute the propagation.** Decoupled models such as SGC and APPNP can compute $\hat{\mathbf{A}}^K\mathbf{X}$, or the personalized PageRank features, once, offline. Training is then ordinary mini-batch learning on independent rows.[^wu2019]

## Tasks: nodes, links and whole graphs

### Node classification

Encode with a GNN, then apply a softmax head per node and the cross-entropy over the labelled nodes, as in the karate demo. The classic citation benchmarks (Cora, CiteSeer, PubMed) use 20 labelled nodes per class.

**Evaluation pitfalls.** Shchur et al. found that the ranking of GNN architectures changes completely between random train/validation/test splits of the same dataset, and that simple, carefully tuned architectures such as GCN match or beat more elaborate ones.[^shchur2018] Report the mean and standard deviation over several splits and seeds, tune the baselines as hard as the new model, and prefer benchmarks with realistic splits, such as the Open Graph Benchmark, which splits citation graphs by time and molecules by scaffold.[^hu2020]

### Link prediction

Encode the nodes, then score a candidate pair with a **decoder**, for example $s(u,v)=\sigma(\mathbf{z}_u^\top\mathbf{z}_v)$ or an MLP on $\mathbf{z}_u\odot\mathbf{z}_v$. Train with binary cross-entropy on observed edges against **negative samples** (random non-edges), as in the shallow embeddings of [[Graph Representation Learning]].

> [!example]- Code: an inner-product decoder trained against negative samples (PyTorch)
> ```python
> import torch
> import torch.nn.functional as F
>
> def link_loss(Z, pos_edges, k=1):
>     """Z: (N, d) GNN node embeddings; pos_edges: (2, E) training edges. k negatives per positive."""
>     u, v = pos_edges
>     neg_v = torch.randint(0, Z.shape[0], (k * u.numel(),))   # random targets: (almost always) non-edges
>     pos = (Z[u] * Z[v]).sum(-1)                               # logit of s(u, v) = sigma(z_u^T z_v)
>     neg = (Z[u.repeat(k)] * Z[neg_v]).sum(-1)
>     logits = torch.cat([pos, neg])
>     labels = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)])
>     return F.binary_cross_entropy_with_logits(logits, labels)
> ```
> `Z` must come from a GNN run on the graph *without* the validation and test edges (see the warning below), and the same negatives' distribution should be used when reporting AUC or MRR.

The **graph autoencoder** (GAE) reconstructs the whole adjacency matrix as $\sigma(\mathbf{Z}\mathbf{Z}^\top)$ from a GCN encoder. Its probabilistic version, the **variational graph autoencoder** (VGAE), has a GCN output a Gaussian $q(\mathbf{z}_v\mid\mathbf{X},\mathbf{A})=\mathcal{N}(\boldsymbol\mu_v,\operatorname{diag}\boldsymbol\sigma_v^2)$ per node and maximizes the evidence lower bound

$$
\mathbb{E}_q\big[\log p(\mathbf{A}\mid\mathbf{Z})\big]-D_{\mathrm{KL}}\big(q(\mathbf{Z}\mid\mathbf{X},\mathbf{A})\,\big\|\,p(\mathbf{Z})\big),
$$

with the inner-product decoder $p(A_{uv}=1\mid\mathbf{z}_u,\mathbf{z}_v)=\sigma(\mathbf{z}_u^\top\mathbf{z}_v)$ and a standard normal prior.[^kipf2016] The ELBO, the reparameterization trick and the KL term are derived in [[Variational Autoencoders]], later in this series.

> [!warning] Leakage in link prediction
> The edges used to evaluate must be **removed from the graph the GNN passes messages on**. Otherwise the encoder sees the very edge it is asked to predict, and a model can score well just by detecting that $u$ and $v$ are neighbours. Also report how negatives were sampled: random non-edges are usually far easier to reject than the hard negatives a real recommender faces.

### Graph classification

Pool the node states into one vector with a **readout** and put an MLP on top. The choice of readout matters for the same reasons as the aggregator:

- **sum** keeps counts (how many carbon atoms) and is the most expressive;
- **mean** keeps proportions and is robust to graph size;
- **max** detects the presence of a feature anywhere (does any node look like a toxic group?).

> [!example]- Code: batching many graphs and the readout (after version 5 of the repository)
> ```python
> import torch
>
> def collate(graphs):
>     """graphs: list of (X (n_i, F), edge_index (2, E_i)). One big disconnected graph plus batch ids."""
>     Xs, eis, batch, offset = [], [], [], 0
>     for i, (X, ei) in enumerate(graphs):
>         Xs.append(X); eis.append(ei + offset)                 # renumber the nodes of graph i
>         batch.append(torch.full((len(X),), i)); offset += len(X)
>     return torch.cat(Xs), torch.cat(eis, 1), torch.cat(batch)
>
> # message passing runs on the big graph unchanged (no edges cross graphs); then, per graph:
> # h_G = aggregate(H, batch, num_graphs, how="sum")   # the same aggregate as for neighbours
> ```
> Pooling a graph is aggregation with the graph id as the "target": `aggregate` from the message-passing snippet does sum, mean and max readouts as is.

Images are pooled hierarchically (pixels, then parts, then objects), and graphs can be too:

- **DiffPool** learns a soft assignment of nodes to clusters with a second GNN, $\mathbf{S}=\operatorname{softmax}(\mathrm{GNN}_{\text{pool}}(\mathbf{A},\mathbf{X}))\in\mathbb{R}^{N\times N'}$, and coarsens the graph to $\mathbf{X}'=\mathbf{S}^\top\mathbf{Z}$ and $\mathbf{A}'=\mathbf{S}^\top\mathbf{A}\mathbf{S}$. The coarse graph is dense, which limits it to small graphs.[^ying2018]
- **SAGPool** scores every node with a GCN, keeps the top fraction, and gates their features by the scores, so the pooled graph stays sparse.[^lee2019]

In an important re-evaluation with careful model selection, Errica et al. found that baselines that **ignore the graph structure** (for example, an MLP on summed node features) are competitive with GNNs on several standard graph-classification benchmarks.[^errica2020] It is a useful sanity check for any new method.

## Modern directions

### Graph transformers

A transformer is a GNN on the complete graph. Running one on the nodes gives every node a one-layer path to every other (no over-squashing) and content-based weights, at $O(N^2)$ cost. But full attention ignores the edges, so the structure must come back through **encodings**, as order came back through [[Sequence Models#Order: positional encodings|positional encodings]] in text:

- **Laplacian eigenvector positional encodings**: give node $v$ the entries $(u_{2,v},\dots,u_{k+1,v})$ of the $k$ smoothest non-trivial Laplacian eigenvectors. These are the graph's sines and cosines, the natural generalization of sinusoidal positions.[^dwivedi2020] Eigenvectors are only defined up to sign (and up to a basis within repeated eigenvalues), so training flips signs at random, or uses networks built to be invariant to these symmetries, such as SignNet.[^lim2023]
- **Random-walk structural encodings** (RWSE): give node $v$ the probabilities that a random walk returns to $v$ after $1,\dots,K$ steps, $\big((\mathbf{P})_{vv},(\mathbf{P}^2)_{vv},\dots,(\mathbf{P}^K)_{vv}\big)$ with $\mathbf{P}=\mathbf{D}^{-1}\mathbf{A}$. They have no sign ambiguity, and they see cycles: a node on a triangle can return in 3 steps, a node on a 12-cycle cannot.[^dwivedi2022] This is what solved Task 2 in Figure 4.
- **Graphormer** adds three biases to a standard transformer: a learned embedding of each node's degree (*centrality encoding*), a learned bias $b_{\varphi(u,v)}$ added to the attention logit, indexed by the shortest-path distance $\varphi(u,v)$ (*spatial encoding*), and an encoding of the edge features along the shortest path. It won a large-scale molecular property prediction challenge in 2021.[^ying2021]
- **GPS** combines the two worlds in every layer: a local message-passing layer and a global attention layer run in parallel on the same node states and are summed, with positional and structural encodings added to the input. With linear-time attention it scales to graphs with thousands of nodes.[^rampasek2022]

### Self-supervised learning on graphs

Labels on graphs are expensive (each label may be a wet-lab experiment), so the self-supervised ideas of the [[Self-Supervised Representation Learning|next lecture]] carry over:

- **Deep Graph Infomax** (DGI) trains a GNN encoder to tell real node embeddings from those of a **corrupted** graph (the same edges with the rows of $\mathbf{X}$ shuffled), using a discriminator between each node's embedding and a summary vector of the whole graph. It maximizes the mutual information between local and global views, without any labels.[^velickovic2019]
- **GraphCL** applies the SimCLR recipe to whole graphs: two random augmentations of each graph (node dropping, edge perturbation, attribute masking, subgraph sampling) are pulled together by a contrastive loss and pushed away from the other graphs in the batch. Which augmentation helps depends on the domain: removing a few edges hardly changes a social network but can turn one molecule into another.[^you2020]

Both are instances of the contrastive objectives (InfoNCE, positive and negative pairs) derived in [[Self-Supervised Representation Learning#From energy to contrast]].

### Relational GNNs for knowledge graphs

A [[Graph Representation Learning|knowledge graph]] has typed edges $(h,r,t)$: *(Marie Curie, born in, Warsaw)*. Shallow models such as TransE and DistMult learn one vector per entity and per relation. A GNN can compute the entity embeddings from their neighbourhoods instead, as long as it respects the edge types.

**R-GCN** gives every relation its own weight matrix:[^schlichtkrull2018]

$$
\mathbf{h}_v^{(\ell+1)}=\sigma\Big(\mathbf{W}_0^{(\ell)}\mathbf{h}_v^{(\ell)}+\sum_{r}\sum_{u\in\mathcal{N}_r(v)}\frac{1}{c_{v,r}}\mathbf{W}_r^{(\ell)}\mathbf{h}_u^{(\ell)}\Big).
$$

Here $\mathcal{N}_r(v)$ is the set of neighbours through relation $r$ (inverse relations count as separate types) and $c_{v,r}$ a normalizer such as $\lvert\mathcal{N}_r(v)\rvert$. With hundreds of relations this has too many parameters, so each $\mathbf{W}_r=\sum_b a_{rb}\mathbf{V}_b$ is written as a combination of a few shared **basis** matrices. For link prediction, the R-GCN encoder feeds a DistMult decoder.

**CompGCN** also embeds the relations and composes them with the neighbour before the message, $\mathbf{W}_{\text{dir}(r)}\,\varphi(\mathbf{h}_u,\mathbf{z}_r)$. The composition $\varphi$ is subtraction $\mathbf{h}_u-\mathbf{z}_r$ (as in TransE), elementwise multiplication (DistMult) or circular correlation (HolE). The relation embeddings are updated layer by layer too.[^vashishth2020]

### Graphs and large language models

Graphs are also becoming a way to organize what a language model retrieves. **GraphRAG** has an LLM extract entities and relations from a document collection into a knowledge graph, partitions it into communities with the Leiden algorithm[^traag2019] (a refinement of [[Clustering and Expectation Maximization#Communities in networks: modularity and the Louvain method|Louvain]]), and pre-writes a summary of each community. Questions about the whole corpus ("what are the main themes?") are answered by combining the community summaries, which plain passage retrieval cannot do.[^edge2024] Other systems retrieve a relevant subgraph for each question and encode it with a GNN into the language model's input.[^he2024] These are pointers only; [[Language Representations and Language Models]] covers the language-model side.

## Key takeaways

- Graphs have no node order and no fixed neighbourhood size. GNN layers are **permutation equivariant** (relabelled input, relabelled output) and graph-level readouts are **invariant**; this is the graph version of translation equivariance and pooling in CNNs.
- **Spectral** convolution filters signals in the eigenbasis of the graph Laplacian, whose eigenvectors are the graph's frequencies. Free spectral filters are expensive, non-local and tied to one graph; **polynomial** filters are $K$-hop local, cost $O(K\lvert\mathcal{E}\rvert)$ and transfer.
- **GCN** is the first-order polynomial filter with the **renormalization trick**, $\sigma(\tilde{\mathbf{D}}^{-1/2}\tilde{\mathbf{A}}\tilde{\mathbf{D}}^{-1/2}\mathbf{H}\mathbf{W})$: average over the closed neighbourhood, then a shared linear map. It is a low-pass filter, which is why it works under homophily.
- **Message passing** (message, permutation-invariant aggregate, update) covers GCN, GraphSAGE, GAT and GIN. GraphSAGE adds sampling and separate self/neighbour weights and is inductive; GAT learns neighbour weights by attention.
- Message-passing GNNs are at most as powerful as **1-WL** colour refinement; **sum** aggregation (GIN) reaches the bound, while mean and max lose multiplicities. Structural encodings, random features or higher-order networks go beyond it.
- Depth brings **over-smoothing** (repeated averaging converges to the random walk's stationary distribution; residuals, normalization and **APPNP**'s personalized PageRank help) and **over-squashing** (exponentially many nodes squeezed through bottlenecks). Under **heterophily**, keep a node's own features apart from its neighbours'.
- The same encoder serves node classification, link prediction (with a decoder, as in GAE/VGAE) and graph classification (with a readout); evaluate over many splits, avoid edge leakage, and compare against structure-free baselines.

## Where this shows up next

- [[Self-Supervised Representation Learning]]: the contrastive objectives behind DGI and GraphCL, and the collapse problems that graph methods share with image methods.
- [[Variational Autoencoders]]: the ELBO and reparameterization behind the variational graph autoencoder.
- [[Language Representations and Language Models]]: the language side of retrieval over graphs such as GraphRAG.
- Looking back: [[Graph Representation Learning]] (shallow and knowledge-graph embeddings that GNN encoders replace or feed), [[Clustering and Expectation Maximization#Spectral clustering|spectral clustering]] (the same Laplacian eigenvectors, used without learning) and [[Markov Chains]] (random walks, stationary distributions and PageRank, which explain over-smoothing and APPNP).

<!-- References: footnote definitions below render as the numbered "References" list at the bottom of the page -->

[^bronstein2021]: Bronstein, M. M., Bruna, J., Cohen, T., & Veličković, P. (2021). *Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges*. arXiv. [arXiv:2104.13478](https://arxiv.org/abs/2104.13478)
[^shuman2013]: Shuman, D. I., Narang, S. K., Frossard, P., Ortega, A., & Vandergheynst, P. (2013). *The emerging field of signal processing on graphs: Extending high-dimensional data analysis to networks and other irregular domains*. IEEE Signal Processing Magazine, 30(3). [doi:10.1109/MSP.2012.2235192](https://doi.org/10.1109/MSP.2012.2235192)
[^bruna2014]: Bruna, J., Zaremba, W., Szlam, A., & LeCun, Y. (2014). *Spectral Networks and Locally Connected Networks on Graphs*. ICLR. [arXiv:1312.6203](https://arxiv.org/abs/1312.6203)
[^defferrard2016]: Defferrard, M., Bresson, X., & Vandergheynst, P. (2016). *Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering*. NeurIPS. [arXiv:1606.09375](https://arxiv.org/abs/1606.09375)
[^hammond2011]: Hammond, D. K., Vandergheynst, P., & Gribonval, R. (2011). *Wavelets on graphs via spectral graph theory*. Applied and Computational Harmonic Analysis, 30(2). [doi:10.1016/j.acha.2010.04.005](https://doi.org/10.1016/j.acha.2010.04.005)
[^kipf2017]: Kipf, T. N., & Welling, M. (2017). *Semi-Supervised Classification with Graph Convolutional Networks*. ICLR. [arXiv:1609.02907](https://arxiv.org/abs/1609.02907)
[^li2018]: Li, Q., Han, Z., & Wu, X.-M. (2018). *Deeper Insights into Graph Convolutional Networks for Semi-Supervised Learning*. AAAI. [arXiv:1801.07606](https://arxiv.org/abs/1801.07606)
[^wu2019]: Wu, F., Zhang, T., Holanda de Souza, A., Fifty, C., Yu, T., & Weinberger, K. Q. (2019). *Simplifying Graph Convolutional Networks*. ICML. [arXiv:1902.07153](https://arxiv.org/abs/1902.07153)
[^zachary1977]: Zachary, W. W. (1977). *An Information Flow Model for Conflict and Fission in Small Groups*. Journal of Anthropological Research, 33(4). [doi:10.1086/jar.33.4.3629752](https://doi.org/10.1086/jar.33.4.3629752)
[^gilmer2017]: Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O., & Dahl, G. E. (2017). *Neural Message Passing for Quantum Chemistry*. ICML. [arXiv:1704.01212](https://arxiv.org/abs/1704.01212)
[^scarselli2009]: Scarselli, F., Gori, M., Tsoi, A. C., Hagenbuchner, M., & Monfardini, G. (2009). *The Graph Neural Network Model*. IEEE Transactions on Neural Networks, 20(1). [doi:10.1109/TNN.2008.2005605](https://doi.org/10.1109/TNN.2008.2005605)
[^hamilton2017]: Hamilton, W. L., Ying, R., & Leskovec, J. (2017). *Inductive Representation Learning on Large Graphs*. NeurIPS. [arXiv:1706.02216](https://arxiv.org/abs/1706.02216)
[^velickovic2018]: Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018). *Graph Attention Networks*. ICLR. [arXiv:1710.10903](https://arxiv.org/abs/1710.10903)
[^brody2022]: Brody, S., Alon, U., & Yahav, E. (2022). *How Attentive are Graph Attention Networks?* ICLR. [arXiv:2105.14491](https://arxiv.org/abs/2105.14491)
[^shervashidze2011]: Shervashidze, N., Schweitzer, P., van Leeuwen, E. J., Mehlhorn, K., & Borgwardt, K. M. (2011). *Weisfeiler-Lehman Graph Kernels*. JMLR, 12. [jmlr.org](https://www.jmlr.org/papers/v12/shervashidze11a.html)
[^xu2019]: Xu, K., Hu, W., Leskovec, J., & Jegelka, S. (2019). *How Powerful are Graph Neural Networks?* ICLR. [arXiv:1810.00826](https://arxiv.org/abs/1810.00826)
[^morris2019]: Morris, C., Ritzert, M., Fey, M., Hamilton, W. L., Lenssen, J. E., Rattan, G., & Grohe, M. (2019). *Weisfeiler and Leman Go Neural: Higher-order Graph Neural Networks*. AAAI. [arXiv:1810.02244](https://arxiv.org/abs/1810.02244)
[^chen2020count]: Chen, Z., Chen, L., Villar, S., & Bruna, J. (2020). *Can Graph Neural Networks Count Substructures?* NeurIPS. [arXiv:2002.04025](https://arxiv.org/abs/2002.04025)
[^sato2021]: Sato, R., Yamada, M., & Kashima, H. (2021). *Random Features Strengthen Graph Neural Networks*. SIAM International Conference on Data Mining. [arXiv:2002.03155](https://arxiv.org/abs/2002.03155)
[^oono2020]: Oono, K., & Suzuki, T. (2020). *Graph Neural Networks Exponentially Lose Expressive Power for Node Classification*. ICLR. [arXiv:1905.10947](https://arxiv.org/abs/1905.10947)
[^cai2020]: Cai, C., & Wang, Y. (2020). *A Note on Over-Smoothing for Graph Neural Networks*. ICML Graph Representation Learning workshop. [arXiv:2006.13318](https://arxiv.org/abs/2006.13318)
[^chen2020gcnii]: Chen, M., Wei, Z., Huang, Z., Ding, B., & Li, Y. (2020). *Simple and Deep Graph Convolutional Networks*. ICML. [arXiv:2007.02133](https://arxiv.org/abs/2007.02133)
[^xu2018jk]: Xu, K., Li, C., Tian, Y., Sonobe, T., Kawarabayashi, K., & Jegelka, S. (2018). *Representation Learning on Graphs with Jumping Knowledge Networks*. ICML. [arXiv:1806.03536](https://arxiv.org/abs/1806.03536)
[^zhao2020]: Zhao, L., & Akoglu, L. (2020). *PairNorm: Tackling Oversmoothing in GNNs*. ICLR. [arXiv:1909.12223](https://arxiv.org/abs/1909.12223)
[^gasteiger2019]: Gasteiger (formerly Klicpera), J., Bojchevski, A., & Günnemann, S. (2019). *Predict then Propagate: Graph Neural Networks meet Personalized PageRank*. ICLR. [arXiv:1810.05997](https://arxiv.org/abs/1810.05997)
[^holland1983]: Holland, P. W., Laskey, K. B., & Leinhardt, S. (1983). *Stochastic blockmodels: First steps*. Social Networks, 5(2). [doi:10.1016/0378-8733(83)90021-7](https://doi.org/10.1016/0378-8733(83)90021-7)
[^alon2021]: Alon, U., & Yahav, E. (2021). *On the Bottleneck of Graph Neural Networks and its Practical Implications*. ICLR. [arXiv:2006.05205](https://arxiv.org/abs/2006.05205)
[^topping2022]: Topping, J., Di Giovanni, F., Chamberlain, B. P., Dong, X., & Bronstein, M. M. (2022). *Understanding over-squashing and bottlenecks on graphs via curvature*. ICLR. [arXiv:2111.14522](https://arxiv.org/abs/2111.14522)
[^zhu2020]: Zhu, J., Yan, Y., Zhao, L., Heimann, M., Akoglu, L., & Koutra, D. (2020). *Beyond Homophily in Graph Neural Networks: Current Limitations and Effective Designs*. NeurIPS. [arXiv:2006.11468](https://arxiv.org/abs/2006.11468)
[^chien2021]: Chien, E., Peng, J., Li, P., & Milenkovic, O. (2021). *Adaptive Universal Generalized PageRank Graph Neural Network*. ICLR. [arXiv:2006.07988](https://arxiv.org/abs/2006.07988)
[^platonov2023]: Platonov, O., Kuznedelev, D., Diskin, M., Babenko, A., & Prokhorenkova, L. (2023). *A critical look at the evaluation of GNNs under heterophily: Are we really making progress?* ICLR. [arXiv:2302.11640](https://arxiv.org/abs/2302.11640)
[^chiang2019]: Chiang, W.-L., Liu, X., Si, S., Li, Y., Bengio, S., & Hsieh, C.-J. (2019). *Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks*. KDD. [arXiv:1905.07953](https://arxiv.org/abs/1905.07953)
[^zeng2020]: Zeng, H., Zhou, H., Srivastava, A., Kannan, R., & Prasanna, V. (2020). *GraphSAINT: Graph Sampling Based Inductive Learning Method*. ICLR. [arXiv:1907.04931](https://arxiv.org/abs/1907.04931)
[^shchur2018]: Shchur, O., Mumme, M., Bojchevski, A., & Günnemann, S. (2018). *Pitfalls of Graph Neural Network Evaluation*. NeurIPS Relational Representation Learning workshop. [arXiv:1811.05868](https://arxiv.org/abs/1811.05868)
[^hu2020]: Hu, W., Fey, M., Zitnik, M., Dong, Y., Ren, H., Liu, B., Catasta, M., & Leskovec, J. (2020). *Open Graph Benchmark: Datasets for Machine Learning on Graphs*. NeurIPS. [arXiv:2005.00687](https://arxiv.org/abs/2005.00687)
[^kipf2016]: Kipf, T. N., & Welling, M. (2016). *Variational Graph Auto-Encoders*. NeurIPS Bayesian Deep Learning workshop. [arXiv:1611.07308](https://arxiv.org/abs/1611.07308)
[^ying2018]: Ying, R., You, J., Morris, C., Ren, X., Hamilton, W. L., & Leskovec, J. (2018). *Hierarchical Graph Representation Learning with Differentiable Pooling*. NeurIPS. [arXiv:1806.08804](https://arxiv.org/abs/1806.08804)
[^lee2019]: Lee, J., Lee, I., & Kang, J. (2019). *Self-Attention Graph Pooling*. ICML. [arXiv:1904.08082](https://arxiv.org/abs/1904.08082)
[^errica2020]: Errica, F., Podda, M., Bacciu, D., & Micheli, A. (2020). *A Fair Comparison of Graph Neural Networks for Graph Classification*. ICLR. [arXiv:1912.09893](https://arxiv.org/abs/1912.09893)
[^dwivedi2020]: Dwivedi, V. P., & Bresson, X. (2021). *A Generalization of Transformer Networks to Graphs*. AAAI Workshop on Deep Learning on Graphs. [arXiv:2012.09699](https://arxiv.org/abs/2012.09699)
[^lim2023]: Lim, D., Robinson, J., Zhao, L., Smidt, T., Sra, S., Maron, H., & Jegelka, S. (2023). *Sign and Basis Invariant Networks for Spectral Graph Representation Learning*. ICLR. [arXiv:2202.13013](https://arxiv.org/abs/2202.13013)
[^dwivedi2022]: Dwivedi, V. P., Luu, A. T., Laurent, T., Bengio, Y., & Bresson, X. (2022). *Graph Neural Networks with Learnable Structural and Positional Representations*. ICLR. [arXiv:2110.07875](https://arxiv.org/abs/2110.07875)
[^ying2021]: Ying, C., Cai, T., Luo, S., Zheng, S., Ke, G., He, D., Shen, Y., & Liu, T.-Y. (2021). *Do Transformers Really Perform Bad for Graph Representation?* NeurIPS. [arXiv:2106.05234](https://arxiv.org/abs/2106.05234)
[^rampasek2022]: Rampášek, L., Galkin, M., Dwivedi, V. P., Luu, A. T., Wolf, G., & Beaini, D. (2022). *Recipe for a General, Powerful, Scalable Graph Transformer*. NeurIPS. [arXiv:2205.12454](https://arxiv.org/abs/2205.12454)
[^velickovic2019]: Veličković, P., Fedus, W., Hamilton, W. L., Liò, P., Bengio, Y., & Hjelm, R. D. (2019). *Deep Graph Infomax*. ICLR. [arXiv:1809.10341](https://arxiv.org/abs/1809.10341)
[^you2020]: You, Y., Chen, T., Sui, Y., Chen, T., Wang, Z., & Shen, Y. (2020). *Graph Contrastive Learning with Augmentations*. NeurIPS. [arXiv:2010.13902](https://arxiv.org/abs/2010.13902)
[^schlichtkrull2018]: Schlichtkrull, M., Kipf, T. N., Bloem, P., van den Berg, R., Titov, I., & Welling, M. (2018). *Modeling Relational Data with Graph Convolutional Networks*. ESWC. [arXiv:1703.06103](https://arxiv.org/abs/1703.06103)
[^vashishth2020]: Vashishth, S., Sanyal, S., Nitin, V., & Talukdar, P. (2020). *Composition-based Multi-Relational Graph Convolutional Networks*. ICLR. [arXiv:1911.03082](https://arxiv.org/abs/1911.03082)
[^traag2019]: Traag, V. A., Waltman, L., & van Eck, N. J. (2019). *From Louvain to Leiden: guaranteeing well-connected communities*. Scientific Reports, 9. [doi:10.1038/s41598-019-41695-z](https://doi.org/10.1038/s41598-019-41695-z)
[^edge2024]: Edge, D., Trinh, H., Cheng, N., Bradley, J., Chao, A., Mody, A., Truitt, S., & Larson, J. (2024). *From Local to Global: A Graph RAG Approach to Query-Focused Summarization*. arXiv. [arXiv:2404.16130](https://arxiv.org/abs/2404.16130)
[^he2024]: He, X., Tian, Y., Sun, Y., Chawla, N. V., Laurent, T., LeCun, Y., Bresson, X., & Hooi, B. (2024). *G-Retriever: Retrieval-Augmented Generation for Textual Graph Understanding and Question Answering*. NeurIPS. [arXiv:2402.07630](https://arxiv.org/abs/2402.07630)
