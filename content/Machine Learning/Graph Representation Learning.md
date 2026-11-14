---
title: Graph Representation Learning
tags:
  - machine-learning
  - representation-learning
  - graphs
  - node-embeddings
  - knowledge-graphs
date: 2026-11-14
draft: false
prev: "[[Clustering and Expectation Maximization]]"
next: "[[Neural Networks]]"
---

> **Prerequisites:** [[Clustering and Expectation Maximization#Spectral clustering|spectral clustering and the graph Laplacian]], [[Matrix Factorization and Dimensionality Reduction]] (the SVD, low-rank approximation, Laplacian eigenmaps), [[Markov Chains]] (random walks and stationary distributions), [[Linear Models#Logistic Regression|logistic regression]] and the [[Support Vector Machines#The hinge loss: SVM vs. logistic regression|hinge loss]].
>
> **TL;DR:**
> - A graph is not a table, so most of our models cannot use it directly. **Graph representation learning** gives every node (or entity) a vector, so that node classification, link prediction and clustering become linear problems on those vectors.
> - Classical **centralities** (degree, closeness, betweenness, eigenvector) give each node one hand-designed number saying how important it is; betweenness and the Laplacian's eigenvectors also split the graph into communities, the question learned embeddings answer in more dimensions.
> - **Shallow embeddings** are a lookup table trained as an **encoder–decoder**: the decoder turns two vectors into a similarity, and the loss asks that similarity to match one read off the graph. Spectral methods factorize a proximity matrix; **DeepWalk**, **node2vec** and **LINE** sample random walks and train skip-gram with negative sampling. These turn out to be the same thing: random-walk methods implicitly factorize a matrix (**NetMF**).
> - A lookup table is **transductive**: it has no vector for a node it never saw. Inductive methods need node features and a shared function, which is what graph neural networks provide.
> - **Knowledge graphs** store facts as triples $(h,r,t)$. **TransE** models a relation as a translation $\mathbf{h}+\mathbf{r}\approx\mathbf{t}$; **TransH**, **TransR** and **TransD** fix its trouble with symmetric and one-to-many relations; **DistMult**, **ComplEx** and **RotatE** use products and rotations instead. Which relation patterns a model can express can be proved, and shows up in experiments.
> - Evaluation is easy to get wrong: inverse-relation leakage in FB15k/WN18, test edges left in the training graph, and too-easy negatives all inflate scores.
>
> **Notation:** a graph $G=(\mathcal{V},\mathcal{E})$ with $N=|\mathcal{V}|$ nodes, adjacency matrix $\mathbf{A}$, degrees $d_v$, degree matrix $\operatorname{diag}(\mathbf{d})$, Laplacian $\mathbf{L}=\operatorname{diag}(\mathbf{d})-\mathbf{A}$ and volume $\operatorname{vol}(G)=\sum_vd_v=2|\mathcal{E}|$, as in [[Clustering and Expectation Maximization]]. Node $v$ gets an embedding $\mathbf{z}_v\in\mathbb{R}^{D}$, stacked as the rows of $\mathbf{Z}\in\mathbb{R}^{N\times D}$. In walk methods, $m$ is the window size and $k$ the number of negative samples, as in [[Language Representations and Language Models]].
>
> **Code:** [Graph-Representation-Learning](https://github.com/parag1604/Graph-Representation-Learning): from-scratch tutorial; versions v0–v2 (Laplacian eigenmaps, DeepWalk with a NetMF check, node2vec) and v7–v8 (TransE/H/R/D, DistMult/ComplEx/RotatE with filtered MRR) go with this lecture.

## Motivation

Many datasets are naturally **graphs**: things, and relations between them.

- **Social networks:** people and friendships. Who will befriend whom? Which accounts are bots?
- **Citation and web graphs:** papers citing papers, pages linking pages. What is this paper about? Which page matters? ([[Markov Chains#PageRank: the stationary distribution of the web|PageRank]] answers the last question with a random walk, an idea we will reuse.)
- **Molecules:** atoms bonded to atoms. Is this molecule toxic?
- **Knowledge graphs:** facts like (Paris, capital_of, France). Which facts are missing?
- **Recommendation:** users and the items they bought, a bipartite graph. What will this user buy next?

Every model so far in this series takes a **vector** per example. A graph gives us none: a node has no coordinates, only neighbours, and the number of neighbours varies from node to node. The adjacency matrix is a valid description, but its rows are huge ($N$ entries), sparse, and depend on an arbitrary ordering of the nodes.

The goal of **graph representation learning** is to compute, for each node, a short vector $\mathbf{z}_v\in\mathbb{R}^D$ with $D\ll N$, such that **graph structure becomes geometry**: nodes that are related in the graph end up close (or otherwise predictably placed) in $\mathbb{R}^D$. Downstream tasks then become the familiar ones.

| Task | Question | On the embeddings |
|---|---|---|
| node classification | what is the label of node $v$? | logistic regression on $\mathbf{z}_v$ |
| link prediction | is there (or will there be) an edge $u$–$v$? | a score of the pair, such as $\mathbf{z}_u^\top\mathbf{z}_v$ |
| community detection | which nodes belong together? | K-means on the $\mathbf{z}_v$ |
| visualization | what does the graph look like? | $D=2$, or t-SNE of the $\mathbf{z}_v$ |
| graph classification | is this whole molecule toxic? | pool the node vectors into one vector per graph |

This lecture covers **shallow** methods, which need no neural networks: each node's vector is a free parameter, fitted by matrix factorization or by a skip-gram objective. They are still the right tool for many large graphs, and they are the cleanest way to understand what a "good" node embedding is. The deep alternative, where a network computes $\mathbf{z}_v$ from node features and neighbours, is the subject of [[Graph Neural Networks]], which also treats graph-level tasks in depth.

## Graphs and what we can measure on them

A graph $G=(\mathcal{V},\mathcal{E})$ has nodes $\mathcal{V}=\{1,\dots,N\}$ and edges $\mathcal{E}$. Its **adjacency matrix** has $A_{uv}=1$ if $u$–$v$ is an edge (or the edge weight $w_{uv}$), and $0$ otherwise. Graphs come in several flavours.

- **Undirected** graphs (friendship) have a symmetric $\mathbf{A}$. **Directed** graphs (citations, links, "follows") do not: $A_{uv}$ means an edge from $u$ to $v$.
- **Weighted** graphs carry a strength on each edge (number of messages, similarity).
- **Heterogeneous** graphs have several node types (authors, papers, venues).
- **Multi-relational** graphs have several edge types: an edge is a triple $(u,r,v)$ with a relation $r$. A knowledge graph is the standard example.

The **degree** $d_v=\sum_uA_{uv}$ counts neighbours, and the **graph Laplacian** $\mathbf{L}=\operatorname{diag}(\mathbf{d})-\mathbf{A}$ measures how much a function on the nodes varies across edges:

$$
\mathbf{s}^\top\mathbf{L}\,\mathbf{s}=\frac12\sum_{u,v}A_{uv}(s_u-s_v)^2 .
$$

[[Clustering and Expectation Maximization#Cuts as quadratic forms|The clustering lecture]] derived this identity and used it to find balanced cuts. Here it will measure how far apart an embedding puts neighbouring nodes.

Finally, a **random walk** moves from a node to a uniformly random neighbour. Its transition matrix is $\mathbf{P}=\operatorname{diag}(\mathbf{d})^{-1}\mathbf{A}$, and $(\mathbf{P}^r)_{uv}$ is the probability of being at $v$ after $r$ steps from $u$. On a connected undirected graph its stationary distribution is $\pi_v=d_v/\operatorname{vol}(G)$, and the chain is reversible: $d_u(\mathbf{P}^r)_{uv}=d_v(\mathbf{P}^r)_{vu}$ (see [[Markov Chains#The stationary distribution]]). Both facts appear in the NetMF derivation below.

## Which nodes matter: centrality

Before learning a vector per node, it helps to see the oldest node-level summary in network science: a single number per node that says how **central** it is. A centrality is a hand-designed, one-dimensional "embedding". Comparing it with the learned ones shows what each kind captures.

Our running example is **Zachary's karate club**: 34 members of a university karate club and the 78 friendships between them, recorded just before the club split in two after a dispute between the instructor, "Mr. Hi" (member 1), and the administrator, the "Officer" (member 34).[^zachary1977] We number the members 1–34 as Zachary did.

### Four ways to be central

**Degree centrality** counts friends: $C_D(v)=d_v$, often divided by $N-1$. It is purely local: a member with many friends in one corner of the club scores as high as one with friends everywhere.

**Closeness centrality** asks how quickly a node reaches everyone else. With $\mathrm{dist}(u,v)$ the length of a shortest path on a connected graph,

$$
C_C(v)=\frac{N-1}{\sum_{u\ne v}\mathrm{dist}(v,u)},
$$

the inverse of the average distance to the other nodes.[^freeman1978]

**Betweenness centrality** asks how often a node sits *between* others. If $\sigma_{st}$ is the number of shortest paths from $s$ to $t$ and $\sigma_{st}(v)$ the number that pass through $v$,

$$
C_B(v)=\sum_{s\ne v\ne t}\frac{\sigma_{st}(v)}{\sigma_{st}},
$$

usually divided by the number of pairs, $\binom{N-1}{2}$, so that it lies in $[0,1]$.[^freeman1977] A node with high betweenness is a **broker**: remove it and many pairs drift apart. Brandes' algorithm computes it for all nodes in $O(N\lvert\mathcal{E}\rvert)$ time on an unweighted graph.[^brandes2001]

**Eigenvector centrality** makes importance recursive: a node is central if its neighbours are central.[^bonacich1972] Asking each score to be proportional to the sum of the neighbours' scores gives

$$
\lambda\,c_v=\sum_uA_{vu}\,c_u
\quad\Longleftrightarrow\quad
\mathbf{A}\mathbf{c}=\lambda\,\mathbf{c}.
$$

Which eigenvector? For a connected graph, the **Perron–Frobenius theorem** says the eigenvector of the largest eigenvalue $\lambda_1$ is the only one whose entries can all be chosen positive, so it is the only sensible set of scores. [[Eigenvalues and Eigenvectors#Power iteration and the QR algorithm|Power iteration]] finds it by repeatedly passing scores to neighbours, $\mathbf{c}\leftarrow\mathbf{A}\mathbf{c}/\lVert\mathbf{A}\mathbf{c}\rVert$. On the karate club $\lambda_1=6.73$ and $\lambda_2=4.98$, so the error shrinks by a factor $\lambda_2/\lambda_1=0.74$ per step.

> [!note] Katz centrality and PageRank
> Eigenvector centrality breaks on directed graphs, where nodes with no incoming edges get score 0 and pass nothing on. **Katz centrality** gives every node a baseline, $\mathbf{c}=\alpha\mathbf{A}\mathbf{c}+\beta\mathbf{1}$ with $\alpha<1/\lambda_1$, which sums walks of every length with weight $\alpha^r$.[^katz1953] **PageRank** also divides each node's vote among its out-neighbours: it is the stationary distribution of a random walk that teleports to a random node with probability $1-\alpha$ (see [[Markov Chains#PageRank: the stationary distribution of the web|PageRank]]). On an undirected graph without teleporting, that stationary distribution is $d_v/\operatorname{vol}(G)$, so PageRank is close to degree: on the karate club their rank correlation is 0.98 ($\alpha=0.85$).

### Centrality on the karate club

![[Machine Learning/images/graph-representation-learning-centrality.svg]]
*Figure 1: four centralities on the karate club (node area proportional to the score, rescaled within each panel; the three most central members are numbered). Circles joined Mr. Hi, squares the Officer. All four put the two leaders, members 1 and 34, near the top, but they disagree on the rest.*

| Rank | degree | closeness | betweenness | eigenvector | PageRank |
|---|---|---|---|---|---|
| 1 | 34 (17) | 1 (0.57) | 1 (0.44) | 34 (1.00) | 34 (0.101) |
| 2 | 1 (16) | 3 (0.56) | 34 (0.30) | 1 (0.95) | 1 (0.097) |
| 3 | 33 (12) | 34 (0.55) | 33 (0.15) | 3 (0.85) | 33 (0.072) |
| 4 | 3 (10) | 32 (0.54) | 3 (0.14) | 33 (0.83) | 3 (0.057) |
| 5 | 2 (9) | 14 (0.52) | 32 (0.14) | 2 (0.71) | 2 (0.053) |

The leaders top every list, which is reassuring: centrality finds them without being told about the dispute. The details differ, and each difference has a reason.

- The Officer (34) has the most friends, but Mr. Hi (1) has the higher betweenness, 0.44 against 0.30. Six of Mr. Hi's friends (5, 6, 7, 11, 12 and 17) can reach the rest of the club *only* through him: delete member 1 and they are cut off. Deleting member 34 disconnects no one, because 10 of the Officer's 16 other friends are also friends of member 33, which gives paths a second route.
- Member 3 has only 10 friends but ranks second for closeness. Member 3 has friends in both factions, so everyone is near.
- The rank correlations between the five measures range from 0.68 (eigenvector vs. PageRank) to 0.98 (degree vs. PageRank). They measure related but different things.

### From who matters to who belongs together

Centrality answers "who matters?". The karate club's real question was "who will go with whom?", which is **graph partitioning**. The two questions use the same tools.

**Betweenness finds the cut.** If two communities are joined by few edges, every shortest path between them must use one of those edges, so the edges *between* communities have high **edge betweenness** (the same formula with $\sigma_{st}(e)$ counting paths through edge $e$). The **Girvan–Newman algorithm** turns this into a divisive clustering method: repeatedly delete the edge of highest betweenness, recomputing betweenness after each deletion, and watch the graph fall apart.[^girvan2002] On the karate club, the graph splits into two pieces (15 and 19 members) after 11 deletions, and the split matches the factions for all members but two, 3 and 9 (Figure 2a).

**The eigenvectors divide the work.** Eigenvector centrality is the *top* eigenvector of $\mathbf{A}$, with all entries positive: it ranks nodes but cannot split them. [[Clustering and Expectation Maximization#Spectral clustering|Spectral clustering]] uses the eigenvector of the *second-smallest* eigenvalue of the Laplacian, the Fiedler vector, which must change sign because it is orthogonal to the constant vector. Its sign gives exactly the same split as Girvan–Newman, with the same two members misplaced. That split cuts 10 friendships, one fewer than the real faction boundary (11), so by the cut criterion it is the better partition; members 3 and 9 chose their side for reasons the friendship graph does not record.

![[Machine Learning/images/graph-representation-learning-girvan-newman.svg]]
*Figure 2: centrality and partitioning on the karate club. (a) Edge width is the edge betweenness on the full graph; the 11 dashed edges are those Girvan–Newman deletes before the graph splits. Nodes show the real factions; the two ringed members, 3 and 9, are the only ones the split puts on the other side. (b) Each member's Fiedler-vector entry (which side of the spectral cut) against eigenvector centrality (how central), with area proportional to betweenness. The two leaders are on opposite sides and at the top; the extremes of the Fiedler vector are peripheral members far from the other group. Members with entries close to 0 (3, 9, 14, 20) are the ones between the groups.*

Figure 2b puts the two views side by side. The horizontal axis is the simplest community-preserving embedding, one number per node from the Laplacian ([[#Laplacian eigenmaps]] below are a degree-weighted version of the same idea, in more dimensions); the vertical axis is a centrality. They are nearly independent descriptions of the same people: the leaders are the most central and on opposite sides, the brokers are fairly central and in the middle, and the members furthest out on either side are peripheral ones with few friends.

This is the bridge to the rest of the lecture. A centrality is a single hand-picked number per node. A learned embedding replaces it with $D$ numbers chosen by an objective, and that objective decides which of the two views it keeps: neighbourhood-preserving embeddings recover the communities (Figure 2b's horizontal axis), while [[#node2vec: tunable walks|node2vec]] with outward walks can also pick up **structural roles** such as "hub" or "broker". Centralities are also useful *input* features: graph transformers add a learned embedding of each node's degree for this reason (see [[Graph Neural Networks]]).

## Shallow embeddings as an encoder–decoder

Hamilton, Ying and Leskovec proposed a common frame that fits almost every method in this lecture.[^hamilton2017review] It has four parts (Figure 3).

1. **An encoder** maps a node to its vector. For shallow methods it is a **lookup table**: $\operatorname{enc}(v)=\mathbf{z}_v$, row $v$ of a parameter matrix $\mathbf{Z}\in\mathbb{R}^{N\times D}$.
2. **A similarity** $S_{uv}$ that we read off the graph: adjacency, $k$-step reachability, co-occurrence on random walks, or "shares a community".
3. **A decoder** maps a pair of vectors back to a similarity, usually $\operatorname{dec}(\mathbf{z}_u,\mathbf{z}_v)=\mathbf{z}_u^\top\mathbf{z}_v$ or a function of $\lVert\mathbf{z}_u-\mathbf{z}_v\rVert$.
4. **A loss** compares the two over (some of) the pairs:

$$
\min_{\mathbf{Z}}\ \sum_{(u,v)}\ \ell\big(\operatorname{dec}(\mathbf{z}_u,\mathbf{z}_v),\ S_{uv}\big).
\tag{1}
$$

![[Machine Learning/images/graph-representation-learning-encoder-decoder.svg]]
*Figure 3: the encoder–decoder view of shallow node embeddings. The encoder looks up one free vector per node. The decoder scores a pair, here nodes 1 and 5, which are not adjacent but share a neighbour. Training makes the decoded score match a similarity $S_{15}$ read off the graph.*

The methods differ in the similarity $S$ and the loss $\ell$; the encoder is the same table. This already tells us what a shallow embedding *is*: a low-dimensional summary of one chosen similarity matrix. Choose $S$ to be adjacency and you get one embedding; choose "co-occurs on short random walks" and you get another, with different downstream behaviour.

## Spectral and matrix-factorization methods

### Laplacian eigenmaps

Take $S=\mathbf{A}$ and a decoder that is a squared distance. Asking neighbours to be close means minimizing

$$
\sum_{u,v}A_{uv}\,\lVert\mathbf{z}_u-\mathbf{z}_v\rVert^2=2\operatorname{tr}\big(\mathbf{Z}^\top\mathbf{L}\,\mathbf{Z}\big),
$$

which is the Laplacian quadratic form, applied column by column. On its own it is minimized by putting every node at the same point, so we add a scale constraint, $\mathbf{Z}^\top\operatorname{diag}(\mathbf{d})\mathbf{Z}=\mathbf{I}$, and exclude the constant vector. The solution, by the same [[Eigenvalues and Eigenvectors#The Rayleigh quotient|Rayleigh-quotient]] argument as spectral clustering, is given by the generalized eigenvectors $\mathbf{L}\mathbf{y}=\lambda\operatorname{diag}(\mathbf{d})\mathbf{y}$ with the smallest non-zero eigenvalues. This is **Laplacian eigenmaps**,[^belkin2003] which [[Matrix Factorization and Dimensionality Reduction#Other dimensionality-reduction methods|the dimensionality-reduction lecture]] used on a nearest-neighbour graph built from data. Spectral clustering is Laplacian eigenmaps followed by K-means.

Laplacian eigenmaps are deterministic, need no tuning beyond $D$, and work well for **homophily**: they put nodes of the same community together. Their weakness is that they only see **first-order** proximity, through the edges, and that every eigenvector counts equally in the embedding. The higher eigenvectors are "high-frequency" functions on the graph, so with many dimensions the embedding fills up with noise (we will see this on the karate club below).

### Factorizing a proximity matrix

The other natural decoder is the inner product: ask $\mathbf{z}_u^\top\mathbf{z}_v\approx S_{uv}$. With a squared loss over all pairs, (1) becomes

$$
\min_{\mathbf{Z}}\ \lVert\mathbf{S}-\mathbf{Z}\mathbf{Z}^\top\rVert_F^2 ,
$$

which is a low-rank approximation of $\mathbf{S}$. If we allow separate "source" and "target" vectors, $\mathbf{S}\approx\mathbf{Z}\mathbf{C}^\top$, the best answer is the truncated SVD of $\mathbf{S}$ ([[Matrix Factorization and Dimensionality Reduction#Best low-rank approximation: Eckart–Young|Eckart–Young]]). Several methods are this recipe with different $\mathbf{S}$:

- **Graph factorization** uses $\mathbf{S}=\mathbf{A}$ and fits only the observed edges, with regularization, so that it scales to very large graphs.[^ahmed2013]
- **GraRep** factorizes the $r$-step transition matrices $\mathbf{P}^r$ for $r=1,\dots,R$ (after a log transform), one SVD each, and concatenates the results. Each block captures relations at a different distance.[^cao2015]
- **HOPE** targets *directed* graphs. It factorizes higher-order proximities such as the Katz index $(\mathbf{I}-\beta\mathbf{A})^{-1}\beta\mathbf{A}$, which counts paths of every length with weight $\beta^{\text{length}}$, and keeps separate source and target vectors so that $S_{uv}\neq S_{vu}$ can be represented.[^ou2016]

The appeal is that exact solvers exist. The problem is that $\mathbf{S}$ is $N\times N$ and, for multi-step proximities, **dense**: $\mathbf{P}^3$ of a sparse graph is already mostly non-zero. For a million nodes that matrix has $10^{12}$ entries. The random-walk methods of the next section avoid ever forming it, by sampling.

## Random-walk embeddings

### DeepWalk: sentences made of nodes

Perozzi, Al-Rfou and Skiena noticed that node frequencies in short random walks on social graphs follow a power law, like word frequencies in text.[^perozzi2014] So they treated **walks as sentences and nodes as words**, and trained word2vec on them.

> [!example] Algorithm: DeepWalk
> 1. From every node, start $\gamma$ random walks of length $L$ (typically $\gamma=10$, $L=40$–$80$). Each step moves to a uniformly random neighbour.
> 2. Slide a window of size $m$ over each walk. Every pair of nodes at most $m$ steps apart in a walk is a (centre, context) training pair.
> 3. Train **skip-gram**: give each node a vector $\mathbf{z}_v$ (as a centre) and a vector $\mathbf{c}_v$ (as a context), and maximize the probability of the observed context nodes.

Two nodes end up with similar vectors when they appear in similar contexts, that is, when random walks from them visit the same nodes. That is a **multi-hop** notion of similarity: two nodes need not be adjacent, only well connected through the same region of the graph.

### Skip-gram with negative sampling, in brief

Skip-gram models the probability that $v$ appears in the window of $u$ as a softmax over all nodes,

$$
p(v\mid u)=\frac{\exp(\mathbf{z}_u^\top\mathbf{c}_v)}{\sum_{w\in\mathcal{V}}\exp(\mathbf{z}_u^\top\mathbf{c}_w)} ,
$$

which is [[Linear Models#More than two classes: softmax regression|softmax regression]] with $N$ classes and learned inputs. The normalizer touches all $N$ nodes for every pair, which is far too slow. **Negative sampling** replaces the $N$-way classification by a binary one: is $(u,v)$ a real pair from the walks, or a fake pair whose context was drawn at random from a noise distribution $P_n$?[^mikolov2013] For each real pair, draw $k$ noise nodes and maximize

$$
\log\sigma(\mathbf{z}_u^\top\mathbf{c}_v)+\sum_{i=1}^{k}\log\sigma(-\mathbf{z}_u^\top\mathbf{c}_{w_i}),\qquad w_i\sim P_n .
\tag{2}
$$

The first term pulls real pairs together, the second pushes random pairs apart: the same push–pull that [[Self-Supervised Representation Learning#From energy to contrast|contrastive learning]] derives from an energy view. Word2vec uses $P_n(w)\propto(\text{frequency})^{3/4}$; on graphs the frequency of a node in the walks is proportional to its degree. [[Language Representations and Language Models#Negative sampling|The language-models lecture]] develops skip-gram, negative sampling and their connection to noise-contrastive estimation in depth.

### node2vec: tunable walks

DeepWalk's walks are uniform. Grover and Leskovec asked what *kind* of neighbourhood a walk should explore, and identified two extremes.[^grover2016]

- **Breadth-first** exploration stays close to the start node and samples its immediate neighbourhood. The idea is that this reveals a node's local *role* (a hub, a bridge, a leaf), which they linked to **structural equivalence**.
- **Depth-first** exploration wanders far, and samples a whole region of the graph. Nodes in the same region then look alike: **homophily**, or community membership.

node2vec interpolates with a **second-order** walk: the next step depends on the current node $v$ *and* the previous node $s$. For each neighbour $x$ of $v$, the unnormalized transition weight is $w_{vx}\,\alpha_{pq}(s,x)$, where $w_{vx}$ is the edge weight and

$$
\alpha_{pq}(s,x)=
\begin{cases}
1/p & \text{if } x=s\quad(\text{go back}),\\
1 & \text{if } x \text{ is a neighbour of } s\quad(\text{stay at distance 1 from } s),\\
1/q & \text{otherwise}\quad(\text{move to distance 2 from } s).
\end{cases}
\tag{3}
$$

![[Machine Learning/images/graph-representation-learning-node2vec-bias.svg]]
*Figure 4: one step of a node2vec walk that has just moved from $s$ to $v$. Going back to $s$ has weight $1/p$, a neighbour that is also adjacent to $s$ has weight $1$, and a node farther from $s$ has weight $1/q$. A large $q$ keeps the walk local (breadth-first-like); a small $q$ pushes it outward (depth-first-like). A large $p$ discourages immediate backtracking.*

With $p=q=1$, node2vec is DeepWalk. The return parameter $p$ controls backtracking; the in–out parameter $q$ controls whether the walk stays local ($q>1$) or moves outward ($q<1$). Everything after the walks (windows, skip-gram, negative sampling) is unchanged.

> [!example]- Code: the node2vec biased walk (tested)
> ```python
> import random
>
> def node2vec_walk(adj, start, length, p, q, rng=random):
>     """adj: dict node -> list of neighbours (unweighted). Returns one biased walk."""
>     nbr_sets = {u: set(vs) for u, vs in adj.items()}
>     walk = [start]
>     while len(walk) < length:
>         v = walk[-1]
>         nbrs = adj[v]
>         if not nbrs:
>             break
>         if len(walk) == 1:                     # first step: uniform
>             walk.append(rng.choice(nbrs))
>             continue
>         s = walk[-2]                           # the node we came from
>         weights = [1 / p if x == s             # go back
>                    else 1.0 if x in nbr_sets[s]  # stay at distance 1 from s
>                    else 1 / q                  # move to distance 2 from s
>                    for x in nbrs]
>         walk.append(rng.choices(nbrs, weights=weights)[0])
>     return walk
> ```
> Building `nbr_sets` once per walk keeps the code short; in practice it is built once per graph, and the original implementation precomputes the transition tables so that each step costs $O(1)$ with alias sampling.

> [!warning] Does node2vec capture structural roles?
> The breadth-first/structural-equivalence link is a heuristic. A window of nearby nodes still records *which* nodes are nearby, so two hubs in different parts of the graph see different contexts and get different vectors, however similar their roles. Methods built for structural identity, such as struc2vec, compare degree sequences of neighbourhoods instead of node identities.[^ribeiro2017] In practice $p$ and $q$ are hyperparameters tuned on validation data.

### LINE: first- and second-order proximity

LINE (Large-scale Information Network Embedding) skips the walks and works with edges directly.[^tang2015] It defines two notions of proximity and trains one embedding for each.

- **First-order proximity:** connected nodes should be similar. Model the probability of an edge as $p_1(u,v)=\sigma(\mathbf{z}_u^\top\mathbf{z}_v)$ and maximize $\sum_{(u,v)\in\mathcal{E}}w_{uv}\log p_1(u,v)$, with negative sampling. This is a logistic version of Laplacian eigenmaps.
- **Second-order proximity:** nodes with similar *neighbourhoods* should be similar, even if they are not adjacent. Give each node a context vector and model $p_2(v\mid u)$ as a softmax of $\mathbf{z}_u^\top\mathbf{c}_v$ over all $v$. This is skip-gram with a window of size 1, trained on the edges.

The two embeddings are trained separately and concatenated. Edge sampling with probability proportional to $w_{uv}$ keeps the stochastic gradients well scaled on graphs with very uneven weights.

### Random walks secretly factorize a matrix

Levy and Goldberg showed that skip-gram with negative sampling on text is, at its optimum, a factorization of a shifted pointwise-mutual-information matrix ([[Language Representations and Language Models#What skip-gram actually learns: a shifted PMI matrix|derived here]]).[^levy2014] Qiu et al. carried the argument over to graphs and computed that matrix in closed form.[^qiu2018]

> [!note] Theorem (NetMF)
> Run DeepWalk with window $m$ and $k$ negative samples drawn from the degree distribution, on a connected, undirected, non-bipartite graph, with walks long enough that pair frequencies converge. If the embedding dimension is large enough that each inner product can be set freely, the optimal embeddings satisfy
> $$
> \mathbf{Z}\mathbf{C}^\top=\log\Big(\frac{\operatorname{vol}(G)}{k\,m}\Big(\sum_{r=1}^{m}\mathbf{P}^r\Big)\operatorname{diag}(\mathbf{d})^{-1}\Big),
> \tag{4}
> $$
> with the logarithm taken entry by entry. LINE (second-order) is the case $m=1$: $\log\big(\operatorname{vol}(G)\operatorname{diag}(\mathbf{d})^{-1}\mathbf{A}\operatorname{diag}(\mathbf{d})^{-1}\big)-\log k$.

> [!proof]- Proof sketch
> 1. **Skip-gram's optimum.** For a corpus of pairs $\mathcal{D}$ and negatives from $P_n$, Levy and Goldberg's argument gives, for each pair $(u,v)$ separately,
> $$
> \mathbf{z}_u^\top\mathbf{c}_v=\log\frac{\#(u,v)}{\#(u)\,k\,P_n(v)} ,
> $$
> where $\#(u,v)$ counts the pair and $\#(u)$ counts $u$ as a centre. Set $P_n(v)=\#(v)/|\mathcal{D}|$ (the degree distribution, see step 3).
> 2. **Pair frequencies.** A walk started from the stationary distribution is at $u$ with probability $d_u/\operatorname{vol}(G)$ at every step. The pair $(u,v)$ at distance $r$ in the window arises when the walk is at $u$ and reaches $v$ after $r$ steps, or is at $v$ and reaches $u$ after $r$ steps. Averaging over $r=1,\dots,m$ and both directions,
> $$
> \frac{\#(u,v)}{|\mathcal{D}|}\to\frac{1}{2m}\sum_{r=1}^{m}\Big(\frac{d_u}{\operatorname{vol}(G)}(\mathbf{P}^r)_{uv}+\frac{d_v}{\operatorname{vol}(G)}(\mathbf{P}^r)_{vu}\Big)=\frac{1}{m}\sum_{r=1}^{m}\frac{d_u(\mathbf{P}^r)_{uv}}{\operatorname{vol}(G)} ,
> $$
> because reversibility makes the two terms equal. Long walks forget their starting node, which is why the stationary start can be assumed.
> 3. **Marginals.** Each node appears as a centre (and as a context) in proportion to its stationary probability, so $\#(u)/|\mathcal{D}|\to d_u/\operatorname{vol}(G)$, and likewise for $v$.
> 4. **Assemble.** Substituting steps 2 and 3 into step 1, the $d_u$ cancels:
> $$
> \mathbf{z}_u^\top\mathbf{c}_v=\log\Big(\frac{\operatorname{vol}(G)}{k\,m}\sum_{r=1}^{m}\frac{(\mathbf{P}^r)_{uv}}{d_v}\Big),
> $$
> which is entry $(u,v)$ of (4). With $m=1$, $(\mathbf{P})_{uv}/d_v=A_{uv}/(d_ud_v)$, which gives the LINE matrix.

What does (4) say? $\sum_r\mathbf{P}^r$ counts how likely a short walk is to connect $u$ and $v$. Dividing by $d_v$ discounts hubs, which every walk visits; this is the graph's version of PMI's division by $p(v)$. So random-walk embeddings are **low-rank factorizations of a log-transformed multi-hop proximity matrix**, and the encoder–decoder frame covers them exactly. Two practical consequences:

- For graphs small enough to form the matrix, **NetMF** computes (4) and takes a truncated SVD of $\log\max(\mathbf{M},1)$ instead of sampling walks. Qiu et al. report that it matches or beats DeepWalk and LINE on their benchmarks.[^qiu2018]
- Undefined entries ($\log 0$ for pairs further apart than $m$ steps) explain why sampled methods push those pairs to strongly negative inner products.

We can check (4) directly. Figure 5 trains DeepWalk on the karate club (34 nodes, 78 edges) with enough dimensions to fit any matrix, and compares the learned inner products with the right-hand side of (4).

![[Machine Learning/images/graph-representation-learning-netmf.svg]]
*Figure 5: DeepWalk implicitly factorizes the NetMF matrix. On the karate club (window $m=3$, $k=5$ negatives from the degree distribution, $D=34$, 3.2 million walk pairs), each dot is a node pair reachable within 3 steps (86% of pairs). Learned $\mathbf{z}_u^\top\mathbf{c}_v$ against $\log M_{uv}$ from (4): correlation 0.996, mean absolute difference 0.085. Pairs that are not reachable within 3 steps, where (4) is $-\infty$, get a mean inner product of $-7.1$.*

> [!example]- Code: the NetMF matrix and embedding (tested)
> ```python
> import numpy as np
>
> def netmf_matrix(A, m, k):
>     """The matrix inside the log of (4): DeepWalk with window m and k negatives."""
>     d = A.sum(1)
>     P = A / d[:, None]                    # random-walk transition matrix
>     S, Pr = np.zeros(A.shape), np.eye(len(A))
>     for _ in range(m):
>         Pr = Pr @ P
>         S += Pr
>     return d.sum() / (k * m) * S / d[None, :]
>
> def netmf_embed(A, m, k, D):
>     M = np.log(np.maximum(netmf_matrix(A, m, k), 1.0))   # truncated log, as in NetMF
>     U, s, _ = np.linalg.svd(M)            # M is symmetric for undirected graphs
>     return U[:, :D] * np.sqrt(s[:D])
> ```

### On the karate club

We return to the karate club from [[#Which nodes matter: centrality|the centrality section]]. We embed the friendship graph, without the faction labels, and then ask how well a **linear probe** recovers the faction: logistic regression trained on the embeddings of 20% of the members (6 people), tested on the rest, averaged over 200 random splits.

![[Machine Learning/images/graph-representation-learning-karate.svg]]
*Figure 6: karate-club embeddings coloured by the faction each member joined (labels not used in training). (a) Laplacian eigenmaps in 2-D. (b–d) DeepWalk and node2vec in 16-D, shown by their first two principal components, with $q=2$ (local walks) and $q=0.5$ (outward walks); $p=1$, window 5, 20 walks of length 40 per node. Accuracies are for a linear probe trained on 20% of the nodes (2-D for Laplacian eigenmaps, 16-D for the walk methods; walk methods averaged over 3 seeds). On this homophilous task all methods do well, and the choice of $q$ changes little.*

Some honest observations, all from the script that made Figure 6:

- Every method separates the factions: probe accuracy is 0.94 for 2-D Laplacian eigenmaps and 0.94–0.96 for the walk methods in 16-D. The karate split follows the community structure, so any homophily-preserving embedding works.
- Laplacian eigenmaps get **worse** with more dimensions: 0.61 in 16-D. The extra eigenvectors are high-frequency functions with the same scale as the useful ones, and a probe trained on six people overfits them. The walk methods weight directions by how much co-occurrence they explain, so they do not suffer from this.
- node2vec's $q$ barely matters here. Its benefits show on larger graphs with mixed structure, and are measured by tuning $p,q$ on validation data.

## Transductive versus inductive embeddings

Every method so far learns a **free vector per node**. That makes them **transductive**: they embed exactly the nodes present during training, and nothing else.

- A **new node**, such as a new user or a newly published paper, has no row in $\mathbf{Z}$. Getting one means retraining, or at least optimizing a new row with the others frozen.
- Two separate graphs (two molecules, two social networks) get embeddings in unrelated coordinate systems: the objective only depends on inner products, so any rotation of $\mathbf{Z}$ (and matching rotation of $\mathbf{C}$) is equally good. Vectors from different runs cannot be compared.
- The number of parameters is $N\times D$: it grows with the graph, and no parameter is shared between nodes. Nodes with few edges get few gradient updates and poor vectors.
- Node **features** (a user's profile, a paper's text, an atom's type) are ignored entirely.

An **inductive** method instead learns a *function* $\mathbf{z}_v=f_\theta(\mathbf{x}_v,\ \text{neighbourhood of }v)$, with features $\mathbf{x}_v$ and parameters $\theta$ shared by all nodes (Figure 7). It applies unchanged to a new node or a new graph, its parameter count does not depend on $N$, and it uses the features. GraphSAGE is the standard example: it computes $\mathbf{z}_v$ by aggregating the features of sampled neighbours with a learned function, and can be trained with the same random-walk co-occurrence loss as DeepWalk.[^hamilton2017sage] Such functions are neural networks over graphs: the subject of [[Graph Neural Networks]].

![[Machine Learning/images/graph-representation-learning-transductive-inductive.svg]]
*Figure 7: a new node 6 arrives with edges to nodes 3 and 5. (a) A transductive lookup table has no row for it. (b) An inductive encoder is a shared function of a node's features and neighbours, so it embeds node 6 the moment its features and edges are known.*

> [!tip] When is transductive fine?
> When the graph is fixed and every node of interest is in it: one big product graph to be embedded nightly, a knowledge graph to be completed, a citation graph to be visualized. Large recommender and knowledge-graph systems have used shallow embeddings at the scale of billions of edges for exactly this reason (see the scaling section).

## Knowledge graphs

### Facts as triples

A **knowledge graph** (KG) stores facts as triples $(h,r,t)$: a **head** entity, a **relation** and a **tail** entity, such as (Marie Curie, born_in, Warsaw) or (Warsaw, capital_of, Poland).[^nickel2016]<sup>,</sup>[^hogan2021] Write $\mathcal{E}$ for the set of entities and $\mathcal{R}$ for the set of relations. Large public KGs such as Freebase and Wikidata hold millions of entities and billions of facts, and they are **incomplete**: most true facts are missing. The central task is **link prediction**, also called **KG completion**: answer queries $(h,r,?)$ and $(?,r,t)$ by ranking candidate entities.

A KG embedding gives each entity a vector $\mathbf{e}$ (we write $\mathbf{h}$, $\mathbf{t}$ for the head's and tail's vectors) and each relation some parameters, and defines a **score** $f_r(h,t)$ that is high for true triples. The encoder is still a lookup table; what is new is that the decoder depends on the relation.

Relations have patterns, and a good model should be able to express them. Our toy KG (Figure 8) has one of each.

- **Symmetric:** $(h,r,t)\Rightarrow(t,r,h)$. spouse_of, sibling_of.
- **Antisymmetric:** $(h,r,t)\Rightarrow\neg(t,r,h)$. father_of.
- **Inverse:** $(h,r_1,t)\Leftrightarrow(t,r_2,h)$. father_of and child_of.
- **Composition:** $(x,r_1,y)\wedge(y,r_2,z)\Rightarrow(x,r_3,z)$. lives_in then city_in gives citizen_of.
- **Mapping cardinality:** a relation is **1-to-N** if a head has many tails (father_of), **N-to-1** if many heads share a tail (lives_in), and **N-to-N** if both (sibling_of).

![[Machine Learning/images/graph-representation-learning-toy-kg.svg]]
*Figure 8: one family of our toy knowledge graph, which has 60 such families (1 to 4 children each), 15 cities and 5 countries, and 9 relations: spouse_of, father_of, mother_of, child_of, sibling_of, lives_in, city_in, citizen_of and has_gender. Each relation pattern appears at least once, so we can test which models learn which patterns.*

### TransE: relations as translations

TransE makes the simplest geometric choice: a relation is a **translation** in the embedding space,[^bordes2013]

$$
\mathbf{h}+\mathbf{r}\approx\mathbf{t},\qquad f_r(h,t)=-\lVert\mathbf{h}+\mathbf{r}-\mathbf{t}\rVert ,
$$

with the $L_1$ or $L_2$ norm. The analogy with word vectors is direct: "king − man + woman ≈ queen" is a translation too. TransE has only $|\mathcal{E}|D+|\mathcal{R}|D$ parameters, trains fast, and handles antisymmetry, inversion ($\mathbf{r}_2=-\mathbf{r}_1$) and composition ($\mathbf{r}_3=\mathbf{r}_1+\mathbf{r}_2$) naturally.

### Training: the margin ranking loss

A KG contains only true facts, so where do negatives come from? TransE **corrupts** each training triple by replacing its head or tail with a random entity, giving $(h',r,t)$ or $(h,r,t')$, and assumes the corrupted triple is false (it usually is, because KGs are sparse). It then minimizes the **margin ranking loss**

$$
\mathcal{L}=\sum_{(h,r,t)\in\mathcal{T}}\ \sum_{(h',r,t')}\max\big(0,\ \gamma-f_r(h,t)+f_r(h',t')\big),
\tag{5}
$$

over the training triples $\mathcal{T}$ and their corruptions, with a margin $\gamma>0$. Entity vectors are kept at norm at most 1, otherwise the loss could be reduced by inflating all the vectors.

This is the **hinge loss** of the [[Support Vector Machines#The hinge loss: SVM vs. logistic regression|soft-margin SVM]], applied to the *difference* of two scores: a true triple must outscore each of its corruptions by at least $\gamma$, and pairs that already do cost nothing. It is also a **contrastive** loss: pull the positive's energy $\lVert\mathbf{h}+\mathbf{r}-\mathbf{t}\rVert$ down, push the negatives' up, in the margin form of [[Self-Supervised Representation Learning#The catch, and the fix: a candidate set|contrastive learning]]. Later models often swap the hinge for a logistic loss, $-\log\sigma(\gamma+f_r(h,t))-\sum\log\sigma(-\gamma-f_r(h',t'))$, as RotatE does,[^sun2019] which is the binary negative-sampling loss (2) again. Two practical refinements:

- **Bernoulli corruption** (from the TransH paper) replaces the head more often for 1-to-N relations and the tail more often for N-to-1 relations. This reduces "false negatives": corrupting the tail of (father, father_of, child) is likely to produce another of his children.[^wang2014]
- **Self-adversarial sampling** (from RotatE) weights each negative by its current score, focusing the gradient on hard negatives, as InfoNCE's softmax does automatically.[^sun2019]

> [!example]- Code: TransE and RotatE scores, corruption and the margin loss (tested)
> ```python
> import torch
> import torch.nn.functional as F
>
> def transe_score(h, r, t):              # real vectors, shape (..., D)
>     return -(h + r - t).norm(p=1, dim=-1)
>
> def rotate_score(h, theta, t):          # h, t complex (..., D); theta real phases (..., D)
>     return -(h * torch.exp(1j * theta) - t).abs().sum(-1)
>
> def corrupt(triples, n_ent, n_neg):
>     """(B, 3) long tensor of (h, r, t) -> (B, n_neg, 3): head or tail replaced, 50/50."""
>     neg = triples.repeat_interleave(n_neg, 0).clone()
>     new = torch.randint(0, n_ent, (len(neg),))
>     head = torch.rand(len(neg)) < 0.5
>     neg[head, 0] = new[head]
>     neg[~head, 2] = new[~head]
>     return neg.view(len(triples), n_neg, 3)
>
> def margin_ranking_loss(f_pos, f_neg, gamma=1.0):
>     """f_pos: (B,), f_neg: (B, n_neg). Each positive against its own corruptions, as in (5)."""
>     return F.relu(gamma - f_pos.unsqueeze(1) + f_neg).mean()
> ```

### Where translation breaks: symmetric and 1-to-N relations

A translation has a direction and moves every point by the same amount. Both properties cause trouble.

> [!proof]- Two things TransE cannot represent
> **Symmetric relations.** If $\mathbf{h}+\mathbf{r}=\mathbf{t}$ and $\mathbf{t}+\mathbf{r}=\mathbf{h}$, adding gives $2\mathbf{r}=\mathbf{0}$, so $\mathbf{r}=\mathbf{0}$ and then $\mathbf{h}=\mathbf{t}$. The two entities must coincide. Worse, with $\mathbf{r}=\mathbf{0}$ the score $f_r(h,h)=0$ is the best possible, so for the query (h, spouse_of, ?) the model ranks $h$ itself at the top.
>
> **1-to-N relations.** If $\mathbf{h}+\mathbf{r}=\mathbf{t}_1$ and $\mathbf{h}+\mathbf{r}=\mathbf{t}_2$, then $\mathbf{t}_1=\mathbf{t}_2$. All of a father's children are forced to the same point, and so are all of a city's residents for N-to-1. Since the children also take part in other relations (their own spouses, cities, genders), the collapse spreads errors.
>
> With a loss rather than exact equations these become "approximately equal" rather than "equal", but the pull is the same.

The fixes all give each relation a way to treat entities *differently*, so that the parts of the vectors that must coincide for one relation are free for the others (Figure 9).

![[Machine Learning/images/graph-representation-learning-translation-models.svg]]
*Figure 9: the translation family. (a) TransE translates in the entity space. (b) TransH projects both entities onto a relation-specific hyperplane, then translates within it. (c) TransR maps entities into a relation-specific space with a matrix $\mathbf{M}_r$, then translates. (d) TransD builds the map from two vectors, one belonging to the relation and one to the entity, so head and tail get different maps.*

**TransH** gives each relation a hyperplane with unit normal $\mathbf{w}_r$ and a translation $\mathbf{d}_r$ lying in it.[^wang2014] Entities are projected onto the hyperplane before translating:

$$
\begin{aligned}
\mathbf{h}_\perp&=\mathbf{h}-(\mathbf{w}_r^\top\mathbf{h})\mathbf{w}_r,\qquad
\mathbf{t}_\perp=\mathbf{t}-(\mathbf{w}_r^\top\mathbf{t})\mathbf{w}_r,\\
f_r(h,t)&=-\lVert\mathbf{h}_\perp+\mathbf{d}_r-\mathbf{t}_\perp\rVert .
\end{aligned}
$$

Now the children of one father only need the same **projection**: they may differ along $\mathbf{w}_r$. A symmetric relation needs $\mathbf{d}_r=\mathbf{0}$ and $\mathbf{h}_\perp=\mathbf{t}_\perp$, which again leaves $\mathbf{h}$ and $\mathbf{t}$ free to differ along $\mathbf{w}_r$.

**TransR** goes further: entities live in $\mathbb{R}^{D}$, each relation has its own space $\mathbb{R}^{D_r}$ and a projection matrix $\mathbf{M}_r\in\mathbb{R}^{D_r\times D}$,[^lin2015]

$$
f_r(h,t)=-\lVert\mathbf{M}_r\mathbf{h}+\mathbf{r}-\mathbf{M}_r\mathbf{t}\rVert .
$$

The motivation is that different relations care about different aspects of an entity (a person's family versus their profession). The cost is $D_rD$ parameters per relation, and a matrix–vector product per score.

**TransD** keeps the expressiveness but drops the full matrices.[^ji2015] Each entity and relation has a second "projection" vector, $\mathbf{e}_p$ and $\mathbf{r}_p$, and the mapping is built from them:

$$
\mathbf{M}_{rh}=\mathbf{r}_p\mathbf{h}_p^\top+\mathbf{I},\qquad
\mathbf{M}_{rt}=\mathbf{r}_p\mathbf{t}_p^\top+\mathbf{I},\qquad
f_r(h,t)=-\lVert\mathbf{M}_{rh}\mathbf{h}+\mathbf{r}-\mathbf{M}_{rt}\mathbf{t}\rVert .
$$

The map now depends on **both** the relation and the entity (a "dynamic" mapping matrix), and head and tail are mapped differently, which suits relations whose heads and tails are different kinds of things. Applying $\mathbf{M}_{rh}$ costs only an inner product, $\mathbf{M}_{rh}\mathbf{h}=\mathbf{h}+(\mathbf{h}_p^\top\mathbf{h})\mathbf{r}_p$, so there is no matrix to store or multiply. (Here the entity and relation spaces have equal dimension; the paper also allows different ones.)

### Products and rotations: DistMult, ComplEx and RotatE

A second family scores triples with **products** instead of distances.

**DistMult** uses a diagonal bilinear form:[^yang2015]

$$
f_r(h,t)=\langle\mathbf{h},\mathbf{r},\mathbf{t}\rangle=\sum_{i=1}^{D}h_i\,r_i\,t_i .
$$

It is a restricted version of RESCAL's full bilinear form $\mathbf{h}^\top\mathbf{W}_r\mathbf{t}$, reviewed in Nickel et al.[^nickel2016] It is cheap and strong in practice, but **always symmetric**: $\langle\mathbf{h},\mathbf{r},\mathbf{t}\rangle=\langle\mathbf{t},\mathbf{r},\mathbf{h}\rangle$, so it cannot tell (father, father_of, child) from (child, father_of, father).

**ComplEx** breaks the symmetry by moving to **complex** vectors $\mathbf{h},\mathbf{r},\mathbf{t}\in\mathbb{C}^D$ and conjugating the tail:[^trouillon2016]

$$
f_r(h,t)=\operatorname{Re}\Big(\sum_{i=1}^{D}h_i\,r_i\,\bar t_i\Big).
$$

Swapping $h$ and $t$ conjugates the sum, which flips the sign of its imaginary part. The real part of $\mathbf{r}$ therefore scores the symmetric part of a relation and the imaginary part its antisymmetric part, and inverse relations are conjugates, $\mathbf{r}_2=\bar{\mathbf{r}}_1$.

**RotatE** combines the two ideas: complex vectors, but a relation is a **rotation**, not a product with an arbitrary complex number.[^sun2019] Each coordinate of $\mathbf{r}$ has modulus 1, $r_i=e^{\mathrm{i}\theta_{r,i}}$, and

$$
\mathbf{t}\approx\mathbf{h}\circ\mathbf{r},\qquad f_r(h,t)=-\lVert\mathbf{h}\circ\mathbf{r}-\mathbf{t}\rVert ,
$$

where $\circ$ is the element-wise product: coordinate $i$ of $\mathbf{h}$ is rotated by the angle $\theta_{r,i}$ in its own complex plane (Figure 10). Rotations compose by adding angles, invert by negating them, and a rotation by $\pi$ is its own inverse: exactly the algebra of composition, inversion and symmetry.

![[Machine Learning/images/graph-representation-learning-rotate.svg]]
*Figure 10: RotatE in one complex coordinate. (a) The relation rotates the head's coordinate by $\theta_{r,i}$ to reach the tail's. (b) A rotation by $\pi$ maps $h_i$ to $t_i=-h_i$ and $t_i$ back to $h_i$, so the relation is symmetric without forcing $h_i=t_i$, unlike TransE's zero translation.*

### Which model can express which pattern

| Model | Symmetric | Antisym. | Inverse | Compose | 1-to-N |
|---|---|---|---|---|---|
| TransE | ✗ | ✓ | ✓ | ✓ | ✗ |
| TransH/R/D | ✓ | ✓ | ✗ | ✗ | ✓ |
| DistMult | ✓ | ✗ | ✗ | ✗ | ✓ |
| ComplEx | ✓ | ✓ | ✓ | ✗ | ✓ |
| RotatE | ✓ | ✓ | ✓ | ✓ | ✗ |

The first four pattern columns follow the analysis of Sun et al.;[^sun2019] "✗" means the pattern cannot be expressed for general embeddings (special cases may work: TransH can represent an inverse pair whose relations share a hyperplane, for example). The 1-to-N column asks whether several distinct tails can score perfectly for the same head and relation. RotatE's rotations are invertible maps, so, like TransE, they send one head to one point; it copes with 1-to-N only approximately, by placing the tails close together.

> [!proof]- Proofs for a few entries
> **DistMult is always symmetric**, so no antisymmetric relation can be represented: $\sum_ih_ir_it_i=\sum_it_ir_ih_i$ for every $\mathbf{h},\mathbf{r},\mathbf{t}$.
>
> **ComplEx can be antisymmetric.** Let $s=\sum_ih_ir_i\bar t_i$. Swapping $h$ and $t$ gives $\sum_it_ir_i\bar h_i$, which equals $\bar s$ when $\mathbf{r}$ is real and differs from it otherwise. With purely imaginary $\mathbf{r}$, $\sum_it_ir_i\bar h_i=-\bar s$, so the swapped score is $\operatorname{Re}(-\bar s)=-\operatorname{Re}(s)$: a high score for $(h,r,t)$ forces a low one for $(t,r,h)$.
>
> **RotatE: symmetry, inversion and composition.** Assume the facts hold exactly. *Symmetry:* $\mathbf{t}=\mathbf{h}\circ\mathbf{r}$ and $\mathbf{h}=\mathbf{t}\circ\mathbf{r}$ give $\mathbf{h}=\mathbf{h}\circ\mathbf{r}\circ\mathbf{r}$, so $r_i^2=1$ wherever $h_i\neq0$, i.e. $r_i=\pm1$. Choosing $r_i=-1$ in some coordinates allows $\mathbf{h}\neq\mathbf{t}$. *Inversion:* $\mathbf{t}=\mathbf{h}\circ\mathbf{r}_1$ and $\mathbf{h}=\mathbf{t}\circ\mathbf{r}_2$ hold for all such pairs iff $\mathbf{r}_2=\bar{\mathbf{r}}_1$ (opposite angles). *Composition:* $\mathbf{y}=\mathbf{x}\circ\mathbf{r}_1$ and $\mathbf{z}=\mathbf{y}\circ\mathbf{r}_2$ give $\mathbf{z}=\mathbf{x}\circ(\mathbf{r}_1\circ\mathbf{r}_2)$, so $\mathbf{r}_3=\mathbf{r}_1\circ\mathbf{r}_2$ (angles add).
>
> **TransH: symmetry.** With $\mathbf{d}_r=\mathbf{0}$, the score $-\lVert\mathbf{h}_\perp-\mathbf{t}_\perp\rVert$ is unchanged by swapping $h$ and $t$, and it is perfect whenever $\mathbf{h}-\mathbf{t}$ is parallel to $\mathbf{w}_r$. So $\mathbf{h}\neq\mathbf{t}$ is allowed.

### The patterns in an experiment

Figure 11 trains all seven models on the toy KG. For each of 3 random seeds we regenerate the KG and hold out about 15% of the triples of six relations; every held-out fact can be inferred from a training fact through one of the patterns (the reverse of a symmetric fact, the inverse of a parent fact, a composition through the city). All models use 32 real numbers per entity (16 complex numbers for ComplEx and RotatE), 10 corruptions per triple and the logistic loss above, trained for 300 epochs with Adam.

![[Machine Learning/images/graph-representation-learning-kg-results.svg]]
*Figure 11: filtered MRR on held-out facts of the toy knowledge graph, per relation, averaged over 3 seeds (overall MRR in brackets). About 285 entities, 1,700–1,850 triples and 175–190 test facts per seed. The four translation models score about 0.5 on the symmetric relations, DistMult and ComplEx fail on the composed relation citizen_of, and RotatE is the most even, with its weakest column on sibling_of, a relation that is both symmetric and N-to-N.*

What the numbers say, relation by relation:

- **Symmetric relations (spouse_of, sibling_of).** All four translation models sit at an MRR of about 0.5, and the reason is exactly the proof above: for 96–99% of these test queries the top-ranked candidate is the query entity *itself*, with the correct answer second. The same argument applies to the whole family. If $g$ is the map a translation model applies before translating, symmetry requires $g(\mathbf{h})+\mathbf{r}=g(\mathbf{t})$ and $g(\mathbf{t})+\mathbf{r}=g(\mathbf{h})$, hence $\mathbf{r}=\mathbf{0}$, and then $(h,r,h)$ gets the best possible score. TransH, TransR and TransD can express symmetry without forcing $\mathbf{h}=\mathbf{t}$, but they cannot stop a symmetric relation from looking reflexive. DistMult (1.00 on spouse_of), ComplEx and RotatE do not have this problem: the self-candidate comes first in only 2%, 4% and 28% of these queries.
- **Composition (citizen_of).** DistMult and ComplEx, which cannot compose relations, score 0.37 and 0.36; TransE and RotatE, which can, score 1.00, and TransR, which cannot, 0.56. TransH (1.00) and TransD (0.96) do better than the table suggests. Our toy graph offers a second route: every family member's citizenship is also in training, and family members end up with similar vectors. Real knowledge graphs are full of such side routes, which is one reason pattern tables and benchmark scores do not always agree.
- **Antisymmetric and inverse relations (father_of, mother_of, child_of).** Everyone does well, DistMult included (0.90–0.99), although it cannot represent antisymmetry. Its symmetric score would wrongly accept (child, father_of, father), but no test query asks for that: the true answer to (father, father_of, ?) is a child, and the child-to-father direction is never queried for father_of. A test set measures the patterns it contains, nothing more.
- **RotatE** has the best overall MRR (0.89 against 0.84 for TransE and TransH). Its weakest column, sibling_of (0.71), is the one where its 1-to-N limitation bites: in a family of three children, each child must be a rotation of each sibling by the *same* $\mathbf{r}$, which forces two of them to coincide.

> [!warning] Tuning matters as much as the model
> On real benchmarks, the gaps between these models shrink, and sometimes reverse, once each is given the same careful training (loss, negatives, regularization, embedding size). Kadlec et al. showed that a well-tuned DistMult beat many newer models of its time.[^kadlec2017] The pattern table says what a model *can* express; whether it does depends on optimization. Our toy runs were just as sensitive: on a smaller version of the toy graph, RotatE trained with the hinge loss (5) and a margin of 1 reached an overall MRR of only 0.34, against 0.71 with a margin of 6; the logistic loss with a margin of 6 is what Figure 11 uses.

## Evaluating link prediction

### Ranks, MRR and Hits@k

For each test triple $(h,r,t)$, score every entity as a candidate tail for $(h,r,?)$, find the **rank** of the true $t$, and do the same for the head. Report

$$
\mathrm{MRR}=\frac{1}{|\mathcal{Q}|}\sum_{q\in\mathcal{Q}}\frac{1}{\operatorname{rank}_q},\qquad
\mathrm{Hits@}k=\frac{1}{|\mathcal{Q}|}\sum_{q\in\mathcal{Q}}\mathbb{1}\{\operatorname{rank}_q\le k\},
$$

over the set of queries $\mathcal{Q}$. MRR (mean reciprocal rank) is dominated by the top of the list: rank 1 counts 1, rank 2 counts 0.5, rank 100 almost nothing. Hits@1, @3 and @10 are the fractions answered within the top 1, 3 or 10.

**Filtering.** A query may have several correct answers: (father, father_of, ?) has one per child. In the **raw** setting, the other children ranked above the test child count as errors. The **filtered** setting, introduced with TransE, removes every *other* known true triple (from training, validation and test) from the candidate list before ranking.[^bordes2013] Filtered metrics are now the standard.

> [!example]- Code: filtered MRR and Hits@k (tested)
> ```python
> import numpy as np
>
> def filtered_metrics(score_tails, test, known, ks=(1, 3, 10)):
>     """score_tails(h, r) -> scores of every entity as the tail of (h, r, ?).
>     known: set of all true (h, r, t) triples (train + valid + test)."""
>     ranks = []
>     for h, r, t in test:
>         s = np.array(score_tails(h, r), dtype=float)
>         others = [e for e in range(len(s)) if e != t and (h, r, e) in known]
>         s[others] = -np.inf                       # filter the other correct answers
>         ranks.append(1 + np.sum(s > s[t]))        # ties count in the model's favour
>     ranks = np.array(ranks)
>     return {"MRR": np.mean(1 / ranks), **{f"Hits@{k}": np.mean(ranks <= k) for k in ks}}
> ```
> Head queries $(?,r,t)$ are handled the same way with a `score_heads` function. Counting ties in the model's favour is optimistic: a model that gives every entity the same score would get rank 1. Careful evaluations break ties randomly or count them against the model.

### Test leakage in FB15k and WN18

The two benchmarks that the original TransE paper introduced, FB15k (from Freebase) and WN18 (from WordNet), turned out to leak. Many of their relations come in **inverse pairs**: Freebase stores both (film, has_actor, person) and (person, acted_in, film), and WordNet both hypernym and hyponym. A random split puts one direction of a fact in training and the other in test. A model then "predicts" the test fact by recognizing the inverse, without learning anything else.

- Toutanova and Chen showed that a simple model using such observed features rivals the embedding models on FB15k, and built **FB15k-237** by removing the near-duplicate and inverse relations.[^toutanova2015]
- Dettmers et al. showed that a rule-based model which only looks up inverse relations reaches state-of-the-art scores on WN18 and FB15k, and built **WN18RR** the same way.[^dettmers2018]

Results on the original benchmarks are therefore mostly measures of inverse-relation lookup; report FB15k-237 and WN18RR. Note that our toy experiment *deliberately* holds out facts whose inverse or symmetric partner is in training, because it tests whether a model can express those patterns. A benchmark meant to measure general knowledge-graph completion must not.

### Leakage and easy negatives in node-level link prediction

Link prediction on an ordinary graph has its own traps. The usual protocol hides a fraction of the edges as test **positives**, samples non-edges as test **negatives**, embeds the remaining graph, and scores pairs by, say, cosine similarity. Two things commonly go wrong.

1. **Leaking the test edges.** If the embedding is trained on the *full* graph, the walks traverse the test edges, and their endpoints co-occur in exactly the way the score rewards. The test edges must be removed before any part of training, including the walks and any preprocessing such as degree normalization.
2. **Easy negatives.** Random node pairs in a sparse graph are almost always far apart, so separating them from true edges is easy. Negatives at distance 2 (non-adjacent nodes with a common neighbour) are what a real recommender must reject, and they are much harder. Li et al. document both problems in published GNN link-prediction benchmarks.[^li2023]

![[Machine Learning/images/graph-representation-learning-linkpred-pitfalls.svg]]
*Figure 12: link prediction on a 400-node clustered power-law graph (Holme–Kim model, about 1,970 edges), with 10% of edges held out; mean and standard deviation over 3 graphs. Common neighbours is the classic heuristic.[^libennowell2007] DeepWalk (32-D, cosine score) trained correctly reaches 0.79 AUC against random negatives but 0.65 against 2-hop negatives. Trained on the full graph, it scores 0.98 and 0.95: leakage turns a mediocre model into an apparently excellent one.*

Two further lessons. The crude common-neighbours count beats our DeepWalk on random negatives (0.83 against 0.79), so a learned embedding must always be compared with such heuristics. And both methods lose most of their edge on 2-hop negatives, which is the regime that matters in practice.

The Holme–Kim generator adds triangles to a preferential-attachment graph, so it has hubs and clustering like real social networks.[^holme2002]

## Scaling shallow embeddings

A lookup table costs $N\times D$ numbers, and skip-gram doubles that with context vectors. For $N=10^8$ nodes and $D=128$ in 32-bit floats, one table is $10^8\times128\times4$ bytes $\approx51$ GB, before optimizer state (Adam keeps two more numbers per parameter). Three tricks make this practical:

- **Sparse updates.** Each training pair touches only $k+2$ rows, so SGD with sparse gradients (or Adagrad, which keeps one extra number per parameter) is used instead of Adam.
- **Partitioning.** PyTorch-BigGraph splits the entities into partitions and the edges into buckets by the partitions of their endpoints, then trains one bucket at a time with only two partitions in memory, on one machine or many.[^lerer2019]
- **Shared negatives.** Corrupting with entities drawn from the same batch reuses vectors already in memory, much like in-batch negatives in contrastive learning.

The parameter count is still linear in $N$, which is one more argument for inductive models, whose parameters are shared: a GNN layer has $O(D^2)$ parameters whatever the size of the graph.

## Beyond shallow embeddings

- **Graph-level tasks.** Shallow node embeddings can be averaged into one vector per graph, but the classical alternative is a **graph kernel**: for example, the Weisfeiler–Lehman kernel counts matching subtree patterns between two graphs and plugs into an [[Support Vector Machines#Kernels|SVM]].[^shervashidze2011] [[Graph Neural Networks]] learn such representations instead.
- **Heterogeneous graphs.** metapath2vec restricts random walks to follow a chosen sequence of node types (author → paper → venue → paper → author) and trains skip-gram on them, so different node types share one space meaningfully.[^dong2017]
- **Graph neural networks.** Replacing the lookup table by a learned function of features and neighbours gives GCNs, GraphSAGE and message passing, and inductive embeddings. The encoder–decoder frame and the losses of this lecture carry over unchanged; only the encoder changes.
- **Knowledge graphs and language models.** Retrieval-augmented generation fetches text for a language model to read before answering. GraphRAG-style systems first have a language model extract an entity–relation graph from a document collection, cluster it into communities, and summarize each community, so that questions about the whole collection can be answered from the summaries.[^edge2024] Explicit KGs also remain the backbone of search engines and question answering, where facts must be checkable and editable.

## Key takeaways

- **Centralities** (degree, closeness, betweenness, eigenvector, PageRank) are hand-designed one-number summaries of how important a node is. Betweenness also finds cuts (**Girvan–Newman**), and eigenvector centrality is the *top* eigenvector of $\mathbf{A}$ while partitioning uses the Laplacian's *second*: on the karate club, Girvan–Newman and the Fiedler vector give the same split, wrong only for members 3 and 9.
- A shallow node embedding is a lookup table trained as an **encoder–decoder**: choose a similarity $S$ on the graph, a decoder ($\mathbf{z}_u^\top\mathbf{z}_v$ or a distance) and a loss. The choice of $S$ decides what "similar nodes" means.
- **Laplacian eigenmaps** minimize $\operatorname{tr}(\mathbf{Z}^\top\mathbf{L}\mathbf{Z})$: neighbours close, first-order proximity only. **Factorization** methods (graph factorization, GraRep, HOPE) take a low-rank approximation of a proximity matrix.
- **DeepWalk** trains skip-gram with negative sampling on random walks; **node2vec** biases the walks with $p$ (return) and $q$ (in–out); **LINE** models first- and second-order proximity on edges.
- These are the same family: DeepWalk implicitly factorizes $\log\big(\frac{\operatorname{vol}(G)}{km}\sum_{r=1}^{m}\mathbf{P}^r\operatorname{diag}(\mathbf{d})^{-1}\big)$ (**NetMF**), which we confirmed numerically (correlation 0.996).
- Lookup tables are **transductive**: no vector for new nodes, no features, parameters growing with $N$. **Inductive** methods learn a shared function, which leads to GNNs.
- **Knowledge-graph embeddings** score triples $(h,r,t)$: translations (TransE and its H/R/D refinements), bilinear products (DistMult, ComplEx) and rotations (RotatE). They are trained with corrupted triples and a margin (hinge) or logistic loss. Which relation patterns each can express follows from short algebraic arguments.
- Evaluate with **filtered** MRR and Hits@k on leak-free benchmarks (FB15k-237, WN18RR). In node link prediction, remove test edges before training and include hard negatives.

## Where this shows up next

- [[Neural Networks]]: the next lecture; learned non-linear functions, the ingredient that turns lookup tables into inductive encoders.
- [[Graph Neural Networks]]: GCNs, GraphSAGE and message passing; inductive node embeddings and graph-level tasks.
- [[Language Representations and Language Models]]: word2vec, negative sampling and the PMI factorization behind (4), in their original setting.
- [[Self-Supervised Representation Learning]]: the contrastive view of the losses used here, with InfoNCE and hard negatives.
- [[Clustering and Expectation Maximization]] and [[Matrix Factorization and Dimensionality Reduction]]: spectral clustering, modularity and low-rank approximation, the classical side of the same ideas.

<!-- References: footnote definitions below render as the numbered "References" list at the bottom of the page -->

[^hamilton2017review]: Hamilton, W. L., Ying, R., & Leskovec, J. (2017). *Representation Learning on Graphs: Methods and Applications*. IEEE Data Engineering Bulletin. [arXiv:1709.05584](https://arxiv.org/abs/1709.05584). Also: Hamilton, W. L. (2020). *Graph Representation Learning*. Morgan & Claypool. [book website](https://www.cs.mcgill.ca/~wlh/grl_book/)
[^belkin2003]: Belkin, M., & Niyogi, P. (2003). *Laplacian Eigenmaps for Dimensionality Reduction and Data Representation*. Neural Computation, 15(6), 1373–1396. [doi:10.1162/089976603321780317](https://doi.org/10.1162/089976603321780317)
[^ahmed2013]: Ahmed, A., Shervashidze, N., Narayanamurthy, S., Josifovski, V., & Smola, A. J. (2013). *Distributed Large-Scale Natural Graph Factorization*. WWW. [doi:10.1145/2488388.2488393](https://doi.org/10.1145/2488388.2488393)
[^cao2015]: Cao, S., Lu, W., & Xu, Q. (2015). *GraRep: Learning Graph Representations with Global Structural Information*. CIKM. [doi:10.1145/2806416.2806512](https://doi.org/10.1145/2806416.2806512)
[^ou2016]: Ou, M., Cui, P., Pei, J., Zhang, Z., & Zhu, W. (2016). *Asymmetric Transitivity Preserving Graph Embedding*. KDD. [doi:10.1145/2939672.2939751](https://doi.org/10.1145/2939672.2939751)
[^perozzi2014]: Perozzi, B., Al-Rfou, R., & Skiena, S. (2014). *DeepWalk: Online Learning of Social Representations*. KDD. [arXiv:1403.6652](https://arxiv.org/abs/1403.6652)
[^mikolov2013]: Mikolov, T., Sutskever, I., Chen, K., Corrado, G., & Dean, J. (2013). *Distributed Representations of Words and Phrases and their Compositionality*. NeurIPS. [arXiv:1310.4546](https://arxiv.org/abs/1310.4546)
[^grover2016]: Grover, A., & Leskovec, J. (2016). *node2vec: Scalable Feature Learning for Networks*. KDD. [arXiv:1607.00653](https://arxiv.org/abs/1607.00653)
[^ribeiro2017]: Ribeiro, L. F. R., Savarese, P. H. P., & Figueiredo, D. R. (2017). *struc2vec: Learning Node Representations from Structural Identity*. KDD. [arXiv:1704.03165](https://arxiv.org/abs/1704.03165)
[^tang2015]: Tang, J., Qu, M., Wang, M., Zhang, M., Yan, J., & Mei, Q. (2015). *LINE: Large-scale Information Network Embedding*. WWW. [arXiv:1503.03578](https://arxiv.org/abs/1503.03578)
[^levy2014]: Levy, O., & Goldberg, Y. (2014). *Neural Word Embedding as Implicit Matrix Factorization*. NeurIPS. [proceedings](https://papers.nips.cc/paper_files/paper/2014/hash/b78666971ceae55a8e87efb7cbfd9ad4-Abstract.html)
[^qiu2018]: Qiu, J., Dong, Y., Ma, H., Li, J., Wang, K., & Tang, J. (2018). *Network Embedding as Matrix Factorization: Unifying DeepWalk, LINE, PTE, and node2vec*. WSDM. [arXiv:1710.02971](https://arxiv.org/abs/1710.02971)
[^zachary1977]: Zachary, W. W. (1977). *An Information Flow Model for Conflict and Fission in Small Groups*. Journal of Anthropological Research, 33(4), 452–473. [doi:10.1086/jar.33.4.3629752](https://doi.org/10.1086/jar.33.4.3629752)
[^hamilton2017sage]: Hamilton, W. L., Ying, R., & Leskovec, J. (2017). *Inductive Representation Learning on Large Graphs*. NeurIPS. [arXiv:1706.02216](https://arxiv.org/abs/1706.02216)
[^nickel2016]: Nickel, M., Murphy, K., Tresp, V., & Gabrilovich, E. (2016). *A Review of Relational Machine Learning for Knowledge Graphs*. Proceedings of the IEEE, 104(1), 11–33. [arXiv:1503.00759](https://arxiv.org/abs/1503.00759)
[^hogan2021]: Hogan, A., et al. (2021). *Knowledge Graphs*. ACM Computing Surveys, 54(4). [arXiv:2003.02320](https://arxiv.org/abs/2003.02320)
[^bordes2013]: Bordes, A., Usunier, N., García-Durán, A., Weston, J., & Yakhnenko, O. (2013). *Translating Embeddings for Modeling Multi-relational Data*. NeurIPS. [proceedings](https://papers.nips.cc/paper/2013/hash/1cecc7a77928ca8133fa24680a88d2f9-Abstract.html)
[^sun2019]: Sun, Z., Deng, Z.-H., Nie, J.-Y., & Tang, J. (2019). *RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space*. ICLR. [arXiv:1902.10197](https://arxiv.org/abs/1902.10197)
[^wang2014]: Wang, Z., Zhang, J., Feng, J., & Chen, Z. (2014). *Knowledge Graph Embedding by Translating on Hyperplanes*. AAAI. [doi:10.1609/aaai.v28i1.8870](https://doi.org/10.1609/aaai.v28i1.8870)
[^lin2015]: Lin, Y., Liu, Z., Sun, M., Liu, Y., & Zhu, X. (2015). *Learning Entity and Relation Embeddings for Knowledge Graph Completion*. AAAI. [doi:10.1609/aaai.v29i1.9491](https://doi.org/10.1609/aaai.v29i1.9491)
[^ji2015]: Ji, G., He, S., Xu, L., Liu, K., & Zhao, J. (2015). *Knowledge Graph Embedding via Dynamic Mapping Matrix*. ACL. [doi:10.3115/v1/P15-1067](https://doi.org/10.3115/v1/P15-1067)
[^yang2015]: Yang, B., Yih, W., He, X., Gao, J., & Deng, L. (2015). *Embedding Entities and Relations for Learning and Inference in Knowledge Bases*. ICLR. [arXiv:1412.6575](https://arxiv.org/abs/1412.6575)
[^trouillon2016]: Trouillon, T., Welbl, J., Riedel, S., Gaussier, É., & Bouchard, G. (2016). *Complex Embeddings for Simple Link Prediction*. ICML. [arXiv:1606.06357](https://arxiv.org/abs/1606.06357)
[^kadlec2017]: Kadlec, R., Bajgar, O., & Kleindienst, J. (2017). *Knowledge Base Completion: Baselines Strike Back*. ACL Workshop on Representation Learning for NLP. [arXiv:1705.10744](https://arxiv.org/abs/1705.10744)
[^toutanova2015]: Toutanova, K., & Chen, D. (2015). *Observed versus Latent Features for Knowledge Base and Text Inference*. Workshop on Continuous Vector Space Models and their Compositionality. [doi:10.18653/v1/W15-4007](https://doi.org/10.18653/v1/W15-4007)
[^dettmers2018]: Dettmers, T., Minervini, P., Stenetorp, P., & Riedel, S. (2018). *Convolutional 2D Knowledge Graph Embeddings*. AAAI. [arXiv:1707.01476](https://arxiv.org/abs/1707.01476)
[^li2023]: Li, J., Shomer, H., Mao, H., Zeng, S., Ma, Y., Shah, N., Tang, J., & Yin, D. (2023). *Evaluating Graph Neural Networks for Link Prediction: Current Pitfalls and New Benchmarking*. NeurIPS Datasets and Benchmarks. [arXiv:2306.10453](https://arxiv.org/abs/2306.10453)
[^libennowell2007]: Liben-Nowell, D., & Kleinberg, J. (2007). *The Link-Prediction Problem for Social Networks*. Journal of the American Society for Information Science and Technology, 58(7), 1019–1031. [doi:10.1002/asi.20591](https://doi.org/10.1002/asi.20591)
[^holme2002]: Holme, P., & Kim, B. J. (2002). *Growing Scale-Free Networks with Tunable Clustering*. Physical Review E, 65, 026107. [doi:10.1103/PhysRevE.65.026107](https://doi.org/10.1103/PhysRevE.65.026107)
[^lerer2019]: Lerer, A., Wu, L., Shen, J., Lacroix, T., Wehrstedt, L., Bose, A., & Peysakhovich, A. (2019). *PyTorch-BigGraph: A Large-scale Graph Embedding System*. SysML. [arXiv:1903.12287](https://arxiv.org/abs/1903.12287)
[^shervashidze2011]: Shervashidze, N., Schweitzer, P., van Leeuwen, E. J., Mehlhorn, K., & Borgwardt, K. M. (2011). *Weisfeiler-Lehman Graph Kernels*. Journal of Machine Learning Research, 12, 2539–2561. [link](https://www.jmlr.org/papers/v12/shervashidze11a.html)
[^dong2017]: Dong, Y., Chawla, N. V., & Swami, A. (2017). *metapath2vec: Scalable Representation Learning for Heterogeneous Networks*. KDD. [doi:10.1145/3097983.3098036](https://doi.org/10.1145/3097983.3098036)
[^edge2024]: Edge, D., Trinh, H., Cheng, N., Bradley, J., Chao, A., Mody, A., Truitt, S., & Larson, J. (2024). *From Local to Global: A Graph RAG Approach to Query-Focused Summarization*. [arXiv:2404.16130](https://arxiv.org/abs/2404.16130)
[^freeman1977]: Freeman, L. C. (1977). *A set of measures of centrality based on betweenness*. Sociometry, 40(1), 35–41. [doi](https://doi.org/10.2307/3033543)
[^freeman1978]: Freeman, L. C. (1978). *Centrality in social networks: conceptual clarification*. Social Networks, 1(3), 215–239. [doi](https://doi.org/10.1016/0378-8733(78)90021-7)
[^brandes2001]: Brandes, U. (2001). *A faster algorithm for betweenness centrality*. Journal of Mathematical Sociology, 25(2), 163–177. [doi](https://doi.org/10.1080/0022250X.2001.9990249)
[^bonacich1972]: Bonacich, P. (1972). *Factoring and weighting approaches to status scores and clique identification*. Journal of Mathematical Sociology, 2(1), 113–120. [doi](https://doi.org/10.1080/0022250X.1972.9989806)
[^katz1953]: Katz, L. (1953). *A new status index derived from sociometric analysis*. Psychometrika, 18(1), 39–43. [doi](https://doi.org/10.1007/BF02289026)
[^girvan2002]: Girvan, M., Newman, M. E. J. (2002). *Community structure in social and biological networks*. PNAS, 99(12), 7821–7826. [doi](https://doi.org/10.1073/pnas.122653799)
