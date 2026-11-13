---
title: Non-parametric Models
tags:
  - k-nearest-neighbors
  - parzen-windows
  - kernel-density-estimation
  - non-parametric
prev: "[[Support Vector Machines]]"
next: "[[Neural Networks]]"
---

> **Prerequisites:** basic probability (densities, Bayes' rule). [[Discriminant Analysis]] is useful for the generative view of classification.
>
> **TL;DR:**
> - **Non-parametric** models keep the training data instead of compressing it into a fixed set of parameters. Their capacity grows with the data.
> - **$k$-nearest neighbours** predicts from the $k$ closest training points. **Parzen windows** use every point within a fixed radius, or weight all points with a smooth **kernel**.
> - Both are local density estimators in disguise: $k$-NN fixes the count and lets the volume adapt, while Parzen fixes the volume and lets the count vary.
> - **Kernel density estimation** turns the same idea into a density model. Similarity-weighted averaging leads straight to attention.
>
> **Code:** [Non-Parameteric](https://github.com/parag1604/Non-Parameteric): k-nearest neighbours and Parzen windows.

## Parametric vs. non-parametric

The models in [[Linear Models]] and [[Discriminant Analysis]] are **parametric**. They extract a *fixed* number of parameters from the data (weights, means, covariances), after which the dataset can be thrown away. A **non-parametric** model does the opposite. It stores the dataset as is, in a suitable data structure, and uses it directly at prediction time. The name does not mean "no parameters". It means the effective number of parameters is not fixed in advance: it **grows with the data**.

Throughout, the data are $\{(\mathbf{x}_i,y_i)\}_{i=1}^{N}$ with $\mathbf{x}_i\in\mathbb{R}^D$ and $y_i\in\{0,1,\dots,C-1\}$. For pictures we use two classes $\{+,-\}$.

---

## $k$-Nearest Neighbours

There is no real *training* phase; this is sometimes called *lazy learning*. Everything happens at inference time. Recall that the goal of machine learning is not to do well on the training data, but to do well on **new** points drawn from the same distribution (the i.i.d. assumption). The training points carry labels, so the simplest guess for a new point is: **predict the label of the closest training point**. This is the 1-nearest-neighbour rule.

"Closest" requires a notion of distance, chosen to suit the data:

- **Euclidean** ($L_2$) for dense real-valued features;
- **Manhattan** ($L_1$) when the features are on comparable scales and should add up independently. On binary one-hot or multi-hot vectors it equals the **Hamming** distance;
- **cosine** distance for embeddings, where direction matters more than length.

Distances are sensitive to feature scales, so standardize the features first (see the [[Linear Models#Iterative methods|pre-processing aside]]).

**Noise motivates voting.** Real data are noisy: a defective sensor or a mislabelled example can put a "$-$" deep inside the "$+$" region. Then even a linearly separable problem no longer looks separable. 1-NN trusts every point completely, so each noisy point carves out its own little island in the decision regions (Figure 1, left). The fix is to look at the **$k$ nearest** training points, $\mathcal{N}_k(\mathbf{x})$, and take a **majority vote**:

$$
\hat y(\mathbf{x})=\arg\max_{c}\ \sum_{i\in\mathcal{N}_k(\mathbf{x})}\mathbb{1}\{y_i=c\}.
$$

![[Machine Learning/images/non-parametric-models-knn-boundaries.svg]]
*Figure 1: $k$-NN decision regions on noisy, nearly linearly separable data. With $k=1$ the boundary is jagged and every mislabelled point gets its own island. With larger $k$ the vote averages out the noise and the boundary becomes smooth. Too large a $k$ starts to blur genuine structure.*

$k$ controls the **bias–variance trade-off**. A small $k$ gives a flexible, low-bias but high-variance predictor that overfits the noise. A large $k$ gives a smoother, more biased but low-variance one, and at the extreme $k=N$ it always predicts the majority class. Choose $k$ by cross-validation. A remarkable classical result: as $N\to\infty$, the error rate of even 1-NN is at most **twice the Bayes-optimal error**.[^coverhart1967]

**Notes on $k$-NN.**

- **Inference cost.** A naive query compares the new point with every training point, $O(ND)$ per prediction. Spatial indexes help in low dimensions (see [[#Faster neighbour search: KD-trees and beyond|faster neighbour search]]).
- **Model size grows with the data.** That is bad where a small model is needed. It is convenient when data arrive continuously: adding new examples requires no retraining, just insertion.
- **Regression and multi-class.** For regression, predict the average of the neighbours' targets. Multi-class classification is simply the majority vote over $C$ classes.
- **Curse of dimensionality.** In high dimensions, all points become roughly equally far apart, so "nearest" carries less information. $k$-NN works best on low-dimensional or well-embedded data.

---

## Parzen Windows

![[Machine Learning/images/non-parametric-models-windows.svg]]
*Figure 2: left, $k$-NN grows a ball around the query until it contains $k=7$ points. The radius is small in dense regions and huge in empty ones. Right, a Parzen window uses a ball of fixed radius $h$. The number of points inside varies, and in an empty region there is no vote at all.*

What $k$-NN really does is find the radius of a ball, centred at the new point, that holds exactly $k$ training points. Flip this around: **fix the radius** $h$ (by some heuristic, or by cross-validation) and take the majority vote of *all* points inside the ball. This fixed region is a **Parzen window**.[^parzen1962] It adapts less well to varying density: windows in sparse regions may be empty.

The window's shape follows the distance. With the Euclidean distance it is a ball (a circle in 2D). With the Manhattan ($L_1$) distance it is a diamond, a square rotated by $45^\circ$. With the $L_\infty$ distance it is an axis-aligned square. The window function is usually called a **kernel** (more on this below).

### Soft windows: kernel weighting

A hard window gives every point inside it equal weight. In effect, it assumes a **uniform** density over the window. It is more sensible to give **nearer points more weight**. So:

1. make the window infinite, so that every training point participates; and
2. weight each point by a decreasing function of its distance to the query.

The standard choice is the **Gaussian kernel**:

$$
w_i(\mathbf{x})=\exp\Big(-\frac{\lVert\mathbf{x}-\mathbf{x}_i\rVert_2^2}{2h^2}\Big),\qquad\hat y(\mathbf{x})=\arg\max_c\sum_{i=1}^{N}w_i(\mathbf{x})\,\mathbb{1}\{y_i=c\}.
$$

Any normalizing constant cancels in the vote, so we omit it. The **bandwidth** $h$ plays the role of $k$ (or of the window radius).

- As $h\to\infty$, all weights become equal and the prediction becomes the global majority class.
- As $h\to0$, the nearest point's weight dominates all others and we recover 1-NN.

For regression the same weights give the **Nadaraya–Watson estimator**, a locally weighted average:[^nadaraya1964]

$$
\hat y(\mathbf{x})=\frac{\sum_iw_i(\mathbf{x})\,y_i}{\sum_jw_j(\mathbf{x})}.
$$

> [!note] "Kernel" means two related things
> Here a kernel is a **smoothing window**: a non-negative bump that integrates to one. In [[Support Vector Machines#Kernels|SVMs]] a kernel is a **similarity** $K(\mathbf{x},\mathbf{x}')=\phi(\mathbf{x})^\top\phi(\mathbf{x}')$, which must be symmetric and positive semi-definite. The Gaussian is both, which is why the words overlap. Not every smoothing kernel is a valid SVM kernel, and vice versa.

---

## Similarity-based (dot-product) neighbours

We need not restrict ourselves to $L_p$ distances. Any **similarity** score $s_i=s(\mathbf{x},\mathbf{x}_i)$ can rank neighbours: take the $k$ most similar instead of the $k$ closest. An inner product $s_i=\mathbf{x}^\top\mathbf{x}_i$ is the obvious choice in a vector space. Normalize it to **cosine similarity** first, otherwise long vectors look similar to everything.

The nice thing about similarities is that they give a natural **weighting** too, not just a neighbour set. The weights must be non-negative and sum to one, but raw inner products can be negative. So instead of dividing by $\sum_js_j$, pass them through a **softmax** with temperature $\tau$:

$$
w_i=\frac{\exp(s_i/\tau)}{\sum_{j=1}^{N}\exp(s_j/\tau)},\qquad\hat{\mathbf{y}}(\mathbf{x})=\sum_iw_i\,\mathbf{y}_i,
$$

where $\mathbf{y}_i$ is a one-hot label or a regression target.

- A small $\tau$ concentrates the weight on the most similar point, recovering 1-NN.
- A large $\tau$ spreads it evenly across all points.

With a Gaussian kernel this is exactly Nadaraya–Watson. With learned similarities $s_i=\mathbf{q}^\top\mathbf{k}_i$ it is the **attention** mechanism of transformers ([[Sequence Models#Attention as a soft dictionary lookup|Sequence Models]]), where a "query" softly retrieves "values" from stored "keys".[^vaswani2017]

---

## Kernel density estimation

The same windows estimate **densities**, not just labels. Place a kernel bump on every data point and average:

$$
\hat p_h(\mathbf{x})=\frac1N\sum_{i=1}^{N}\frac{1}{h^D}\,K\Big(\frac{\mathbf{x}-\mathbf{x}_i}{h}\Big),\qquad\text{e.g. }K(\mathbf{u})=(2\pi)^{-D/2}e^{-\lVert\mathbf{u}\rVert^2/2}.
$$

This is **kernel density estimation** (KDE), also called the Parzen–Rosenblatt window method.[^rosenblatt1956]<sup>,</sup>[^parzen1962] Unlike a histogram, it is smooth and does not depend on where the bin edges fall. Its single important knob is the bandwidth $h$. Too small and every data point becomes a spike; too large and real structure is smoothed away (Figure 3). A common starting point for one-dimensional Gaussian KDE is **Silverman's rule**, $h\approx1.06\,\hat\sigma N^{-1/5}$. Cross-validating the held-out log-likelihood is more reliable.

![[Machine Learning/images/non-parametric-models-kde.svg]]
*Figure 3: Gaussian KDE of 150 samples (rug at the bottom) from a two-component mixture (dotted). A tiny bandwidth chases individual points; a huge one merges the two modes. Silverman's rule is a reasonable first guess, but it slightly oversmooths this two-peaked density: it assumes the data are roughly Gaussian (see below).*

### Bias and variance of KDE

How should $h$ scale with the amount of data? Take one dimension, a symmetric kernel with $\int K=1$ and $\int uK(u)\,du=0$, and write $\mu_2=\int u^2K(u)\,du$ and $R(K)=\int K(u)^2\,du$. The estimate at a point $x$ is an average of $N$ i.i.d. terms $K_h(x-X_i)$, where $K_h(u)=\frac1hK(u/h)$.

**Bias.** Substitute $t=x-hu$ and Taylor-expand the density around $x$:

$$
\begin{aligned}
\mathbb{E}\big[\hat p_h(x)\big]&=\int\frac1hK\Big(\frac{x-t}{h}\Big)p(t)\,dt=\int K(u)\,p(x-hu)\,du\\
&=\int K(u)\Big[p(x)-hu\,p'(x)+\tfrac12h^2u^2p''(x)+O(h^3)\Big]du\\
&=p(x)+\tfrac12h^2\mu_2\,p''(x)+O(h^3).
\end{aligned}
$$

The first-order term vanishes because the kernel is symmetric. The bias is $\frac12h^2\mu_2\,p''(x)$: KDE fills in valleys ($p''>0$) and shaves off peaks ($p''<0$), more so for wide kernels.

**Variance.** The terms are i.i.d., so

$$
\begin{aligned}
\operatorname{Var}\big[\hat p_h(x)\big]&=\frac1N\operatorname{Var}\big[K_h(x-X)\big]\le\frac1N\mathbb{E}\big[K_h(x-X)^2\big]\\
&=\frac{1}{Nh}\int K(u)^2p(x-hu)\,du\approx\frac{p(x)\,R(K)}{Nh}.
\end{aligned}
$$

Narrow kernels average over few points, so the variance grows like $1/(Nh)$.

**The trade-off.** Integrating squared bias plus variance over $x$ gives the asymptotic mean integrated squared error:

$$
\mathrm{AMISE}(h)=\underbrace{\frac{h^4}{4}\,\mu_2^2\,R(p'')}_{\text{bias}^2\ \uparrow\text{ in }h}+\underbrace{\frac{R(K)}{Nh}}_{\text{variance}\ \downarrow\text{ in }h},\qquad R(p'')=\int p''(x)^2\,dx.
$$

Setting the derivative to zero gives the optimal bandwidth and the resulting error:

$$
h^*=\Big(\frac{R(K)}{\mu_2^2\,R(p'')\,N}\Big)^{1/5}\propto N^{-1/5},\qquad\mathrm{AMISE}(h^*)\propto N^{-4/5}.
$$

So the bandwidth should shrink, but slowly, as data grow. The error converges more slowly than the $N^{-1}$ rate of a correctly specified *parametric* model. That is the price of making no assumptions about the shape of $p$.

**Where Silverman's rule comes from.** $h^*$ depends on the unknown curvature $R(p'')$. Pretend $p$ is Gaussian with standard deviation $\sigma$, so $R(p'')=\frac{3}{8\sqrt\pi\,\sigma^5}$. With a Gaussian kernel ($\mu_2=1$, $R(K)=\frac{1}{2\sqrt\pi}$) this gives $h^*=(4/3)^{1/5}\sigma N^{-1/5}\approx1.06\,\sigma N^{-1/5}$. For multimodal data the true curvature is larger, so the rule **oversmooths**. Cross-validating the held-out log-likelihood, or plug-in estimates of $R(p'')$, do better.

**In $D$ dimensions** the bias is still $O(h^2)$, but the variance becomes $O\big(1/(Nh^D)\big)$, because the window now has volume $h^D$. The best rates become $h^*\propto N^{-1/(D+4)}$ and error $\propto N^{-4/(D+4)}$. To halve the error in $D=10$ dimensions you need about $2^{14/4}\approx11$ times more data. This is the curse of dimensionality made quantitative.

![[Machine Learning/images/non-parametric-models-kde-bias-variance.svg]]
*Figure 4: top, Monte-Carlo estimates of integrated squared bias and variance for the density of Figure 3 ($N=150$, 300 repetitions), with the asymptotic formulas (dotted). Bias grows like $h^4$ and variance falls like $1/h$. Their sum is minimized at $h\approx0.30$, while Silverman's rule ($h\approx0.59$) oversmooths this bimodal density. Bottom, the optimal bandwidth shrinks as $N^{-1/5}$ and the error as $N^{-4/5}$; in higher dimensions the error rate degrades to $N^{-4/(D+4)}$.*

**Classification from densities.** Fit a KDE $\hat p(\mathbf{x}\mid y=c)$ to each class, estimate the priors $\pi_c=N_c/N$, and classify with Bayes' rule, as in the [[Discriminant Analysis#Generative classifiers and Bayes' rule|generative view]] but with non-parametric class densities instead of Gaussians. Since $\pi_c\,\hat p(\mathbf{x}\mid c)\propto\sum_{i:y_i=c}K_h(\mathbf{x}-\mathbf{x}_i)$, this is *exactly* the kernel-weighted vote above.

**$k$-NN and Parzen are two sides of one estimator.** Count the $k$ training points that fall in a region of volume $V$ around $\mathbf{x}$. Then the local density estimate is

$$
\hat p(\mathbf{x})\approx\frac{k}{N\,V}.
$$

- **Parzen / KDE:** fix $V$ (the window) and count $k$.
- **$k$-NN:** fix $k$ and measure the $V$ needed to contain it.

Applying Bayes' rule to the $k$-NN version gives $\hat p(y=c\mid\mathbf{x})=k_c/k$, the fraction of neighbours in class $c$. That is exactly the majority vote.[^bishop2006]

---

## Faster neighbour search: KD-trees and beyond

**The bottleneck.** $k$-NN has no training cost, but every prediction pays for it. A naive query computes the distance to **all** $N$ stored points, so it costs $O(ND)$: *linear* in the size of the training set. With a million points and thousands of queries per second, that is hopeless. The data, however, do not change between queries. So we can pay **once** to organize them into a data structure, and then answer each query by examining only a small part of it.

**The 1-D intuition.** In one dimension the answer is familiar: sort the points once, in $O(N\log N)$, and every nearest-neighbour query becomes a **binary search**, $O(\log N)$. Each comparison discards half of the remaining candidates. The KD-tree generalizes this "split and discard" idea to $D$ dimensions.

### KD-trees

A $k$-d tree ("$k$-dimensional tree") is a binary tree in which every node splits space with an axis-aligned hyperplane, cycling through the coordinates level by level.[^bentley1975]

> [!example] Algorithm: build a KD-tree
> **build**(points $S$, depth $\ell$):
> 1. If $S$ is empty, return an empty leaf.
> 2. Choose the split coordinate $d=\ell\bmod D$, so the tree cycles through the axes.
> 3. Find the **median** point $\mathbf{p}$ of $S$ along coordinate $d$. Splitting at the median keeps the tree balanced.
> 4. Create a node storing $\mathbf{p}$ and $d$. Recursively build its **left** child from $\{\mathbf{x}\in S:x_d<p_d\}$ and its **right** child from the rest, both with depth $\ell+1$.
>
> The tree has depth $O(\log N)$. With a linear-time median it is built in $O(N\log N)$, once.

> [!example] Algorithm: nearest-neighbour search
> **search**(node, query $\mathbf{q}$): keep the best distance found so far, $r^*$, initially $\infty$.
> 1. Compute $\lVert\mathbf{q}-\mathbf{p}\rVert$ for the node's point $\mathbf{p}$, and update $r^*$ if it is smaller.
> 2. **Go near first:** recurse into the child on the same side of the split as $\mathbf{q}$.
> 3. **Prune the far side:** visit the other child *only if* $\lvert q_d-p_d\rvert<r^*$. That is, only if the ball of radius $r^*$ around $\mathbf{q}$ crosses the splitting hyperplane. Otherwise nothing on that side can beat the current best, and the whole subtree is skipped.
>
> For $k$ neighbours, keep the $k$ best in a max-heap and prune with the $k$-th best distance.

The pruning test in step 3 is what replaces "look at everything". The query first descends to its own cell, quickly finds a good candidate, and then the ball around it is usually small enough to rule out most other cells (Figure 5).

![[Machine Learning/images/non-parametric-models-kdtree.svg]]
*Figure 5: a KD-tree on 60 points in 2-D. The splitting lines alternate between vertical and horizontal, thicker lines being nearer the root. To find the nearest neighbour of the star, the search computes only 11 distances (orange). Every other cell lies entirely outside the dashed ball and is pruned without being examined.*

In low dimensions, a query visits about $O(\log N)$ nodes on average, instead of $N$.

### When trees stop helping: the curse of dimensionality

The pruning test only works if the ball around the query is **small relative to the cells**. In high dimensions that fails. Distances **concentrate**: the nearest and farthest training points end up at almost the same distance from a query. So the ball of radius $r^*$ crosses nearly every splitting hyperplane, and almost nothing can be pruned (Figure 6). In practice, exact tree search loses its advantage beyond roughly 10–20 dimensions, unless the data actually lie on a low-dimensional manifold.

![[Machine Learning/images/non-parametric-models-kdtree-scaling.svg]]
*Figure 6: distance computations per query for uniformly random data. In $D=2$ the KD-tree needs about 20 distance computations even with $N=30{,}000$, roughly logarithmic growth. In $D=8$ the cost already grows polynomially. In $D=16$ it examines essentially every point and is no better than brute force.*

### Approximate nearest neighbours

Modern applications search high-dimensional embeddings ($D$ in the hundreds) over millions or billions of items. Examples are retrieval for language models, recommendation and image search. There, the standard move is to **give up exactness**: return points that are *probably* among the nearest, in exchange for orders-of-magnitude speedups. The trade-off is measured as **recall vs. latency**.

- **Locality-sensitive hashing (LSH).** Use random hash functions under which nearby points are likely to collide. For cosine similarity, hash $\mathbf{x}\mapsto\operatorname{sign}(\mathbf{r}^\top\mathbf{x})$ with a random direction $\mathbf{r}$. Two vectors at angle $\theta$ collide with probability $1-\theta/\pi$.[^charikar2002] Concatenating several such bits gives buckets, and a query only compares against points in its own buckets.
- **Product quantization (PQ).** Split each vector into $M$ sub-vectors and replace each by the index of its nearest centroid in a small codebook learned with $k$-means. Distances are then approximated by table look-ups on compact codes, so billions of vectors fit in memory.[^jegou2011]
- **Graph-based search (HNSW).** Connect each point to a few near neighbours, in a hierarchy of increasingly sparse layers. A query greedily walks the graph towards the target, starting from the top layer. This is the state of the art for recall vs. speed in many settings.[^malkov2018]

Libraries such as FAISS combine these ideas (inverted files + PQ + GPUs) to search billions of vectors.[^johnson2017] The "vector databases" behind retrieval-augmented generation are, at heart, **approximate $k$-NN indexes**. So the humble nearest-neighbour rule of this lecture, together with the [[#Similarity-based (dot-product) neighbours|similarity-weighted]] read-out, is alive and well inside modern systems.

## Key takeaways

- Non-parametric models store the data, and their capacity grows with $N$. They have no training phase, but prediction is expensive.
- **$k$-NN**: majority vote of the $k$ nearest points. $k$ trades variance (small $k$) against bias (large $k$). Asymptotically, 1-NN error is at most twice the Bayes error.
- **Parzen windows**: a fixed radius instead of a fixed count. Soft **kernel** weights make the vote smooth. The bandwidth $h$ is the key hyper-parameter, and Nadaraya–Watson is the regression version.
- **Similarity weighting** should use a softmax (with temperature) rather than raw normalized inner products. With learned similarities, this is attention.
- **KDE** turns windows into densities. $k$-NN and Parzen are the "fix $k$" and "fix $V$" versions of $\hat p\approx k/(NV)$.
- $k$-NN queries cost $O(N)$. **KD-trees** cut this to about $O(\log N)$ by building an index once and pruning cells, but only in low dimensions. In high dimensions, **approximate** methods (LSH, product quantization, HNSW) trade a little recall for huge speedups.

## Where this shows up next

- [[Discriminant Analysis]]: parametric (Gaussian) class densities instead of KDE.
- [[Support Vector Machines]]: kernels as similarities, and a sparse alternative to storing every point (only the support vectors).
- [[Clustering and Expectation Maximization]]: $k$-means is nearest-mean classification with unknown labels, and Gaussian mixtures sit between a single Gaussian and a KDE.
- [[Neural Networks]], next in the series: instead of storing the data, learn features from it. Similarity-weighted averaging returns as attention in [[Sequence Models#Attention as a soft dictionary lookup|Sequence Models]].

<!-- References: footnote definitions below render as the numbered "References" list at the bottom of the page -->

[^coverhart1967]: Cover, T., & Hart, P. (1967). *Nearest Neighbor Pattern Classification*. IEEE Transactions on Information Theory, 13(1), 21–27. [doi:10.1109/TIT.1967.1053964](https://doi.org/10.1109/TIT.1967.1053964)
[^parzen1962]: Parzen, E. (1962). *On Estimation of a Probability Density Function and Mode*. Annals of Mathematical Statistics, 33(3), 1065–1076. [doi:10.1214/aoms/1177704472](https://doi.org/10.1214/aoms/1177704472)
[^rosenblatt1956]: Rosenblatt, M. (1956). *Remarks on Some Nonparametric Estimates of a Density Function*. Annals of Mathematical Statistics, 27(3), 832–837. [doi:10.1214/aoms/1177728190](https://doi.org/10.1214/aoms/1177728190)
[^nadaraya1964]: Nadaraya, E. A. (1964). *On Estimating Regression*. Theory of Probability & Its Applications, 9(1), 141–142. [doi:10.1137/1109020](https://doi.org/10.1137/1109020)
[^vaswani2017]: Vaswani, A., et al. (2017). *Attention Is All You Need*. NeurIPS. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
[^bentley1975]: Bentley, J. L. (1975). *Multidimensional Binary Search Trees Used for Associative Searching*. Communications of the ACM, 18(9), 509–517. [doi:10.1145/361002.361007](https://doi.org/10.1145/361002.361007)
[^charikar2002]: Charikar, M. S. (2002). *Similarity Estimation Techniques from Rounding Algorithms*. STOC. [doi:10.1145/509907.509965](https://doi.org/10.1145/509907.509965)
[^jegou2011]: Jégou, H., Douze, M., & Schmid, C. (2011). *Product Quantization for Nearest Neighbor Search*. IEEE TPAMI, 33(1), 117–128. [doi:10.1109/TPAMI.2010.57](https://doi.org/10.1109/TPAMI.2010.57)
[^malkov2018]: Malkov, Y. A., & Yashunin, D. A. (2018). *Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs*. IEEE TPAMI. [arXiv:1603.09320](https://arxiv.org/abs/1603.09320)
[^johnson2017]: Johnson, J., Douze, M., & Jégou, H. (2017). *Billion-scale Similarity Search with GPUs*. [arXiv:1702.08734](https://arxiv.org/abs/1702.08734)
[^bishop2006]: Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*, Section 2.5 (Nonparametric methods). Springer. [book page](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/)
