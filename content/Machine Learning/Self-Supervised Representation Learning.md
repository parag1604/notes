---
title: Self-Supervised Representation Learning
tags:
  - machine-learning
  - representation-learning
  - self-supervised-learning
  - contrastive-learning
date: 2026-11-13
draft: false
prev: "[[Graph Neural Networks]]"
next: "[[Language Representations and Language Models]]"
---

> **Prerequisites:** [[Neural Networks]] (encoders, backpropagation, training practice), [[Linear Models#More than two classes: softmax regression|softmax regression and cross-entropy]], [[Convolutional Neural Networks]] and [[Sequence Models]] (the image and text encoders that the losses here train).
>
> **TL;DR:**
> - Labels are expensive; raw images and text are not. **Self-supervised learning** invents a training signal from the data itself and learns an encoder whose features transfer to many downstream tasks with few labels.
> - **Contrastive learning** comes from an energy view: give compatible pairs low energy and incompatible pairs high energy, and normalize with a softmax over a small set of candidates. This is the **InfoNCE** loss. With labels it becomes **supervised contrastive learning**; with data augmentations as the only "labels" it becomes **SimCLR** and **MoCo**.
> - Without negatives, the trivial solution is to map everything to one point (**collapse**). Non-contrastive methods avoid it with asymmetry (**BYOL**, **SimSiam**, **DINO**) or by decorrelating the features (**Barlow Twins**, **VICReg**). **Predictive** methods (**MAE**, **JEPA**) fill in masked parts of the input instead.
> - Contrast an image with its caption and you get **CLIP**, which classifies images it never saw labels for.
>
> **Notation:** an encoder $f_\theta$ maps an input $\mathbf{x}$ to a representation $\mathbf{h}=f_\theta(\mathbf{x})$; a small projection head $g_\theta$ maps it to an embedding $\mathbf{z}=g_\theta(\mathbf{h})\in\mathbb{R}^d$. Similarities are cosines, $\operatorname{sim}(\mathbf{z},\mathbf{z}')=\mathbf{z}^\top\mathbf{z}'/(\lVert\mathbf{z}\rVert\lVert\mathbf{z}'\rVert)$. $B$ is the batch size, $\tau$ the temperature and $t,t'$ random augmentations.
>
> **Code:** [Contrastive-Representation-Learning](https://github.com/parag1604/Contrastive-Representation-Learning): MNIST encoders from a label-supervised pairwise loss through InfoNCE and SimCLR to MoCo, with linear probes, t-SNE plots and fine-tuning from 1000 labels. [improved-vision-encoders](https://github.com/paragduttaiisc/improved-vision-encoders): a JEPA-style masked-latent-prediction vision encoder on ImageNet (75% patch masking, variance and sliced-Wasserstein regularizers against collapse, optional mixture-of-experts layers), tracked with a linear probe.

## Motivation

A supervised image classifier trained on a million labelled photos learns features that are useful far beyond its 1000 classes: its penultimate layer is a good starting point for detection, segmentation, medical images and more. But the labels cost a lot, and most of the world's images, audio and text come with none. The question of this lecture is whether we can learn equally good features **without labels**.

The trick is to set up a task whose answer is already in the data:
- two crops of the same photo should "mean the same thing", while crops of different photos usually should not;
- a missing patch of an image, or a missing word of a sentence, can be predicted from the rest;
- a photo and its caption, scraped together from the web, describe the same scene.

None of these tasks matters in itself. They are **pretext tasks**, and what we keep is the encoder that solved them. This is **self-supervised learning (SSL)**: supervised learning, with the supervision manufactured from unlabelled data.

### What makes a representation good?

We cannot define a good representation in the abstract, so we measure it by what it makes easy. The standard yardsticks, used throughout this lecture, all freeze the encoder $f_\theta$ and ask how much work is left:
- **Linear probe.** Train a linear classifier ([[Linear Models#More than two classes: softmax regression|softmax regression]]) on the frozen features $\mathbf{h}$ with the labels of a downstream task.[^alain2016] If the classes are linearly separable in $\mathbf{h}$-space, the encoder has done the hard part.
- **k-nearest neighbours.** Classify each test point by a vote of its nearest training points in feature space ([[Non-parametric Models]]).[^wu2018] No training at all; it checks that semantically similar inputs are *close*.
- **Fine-tuning.** Use $f_\theta$ as the initialization and train everything on the downstream task, often with few labels. This measures how good a starting point the encoder is.

Behind these numbers sits one idea: **invariance**. A representation of a photo should not change when the photo is cropped a little, recoloured or blurred, because none of these changes the object in it. It should change when the object changes. Self-supervised methods differ mainly in how they encode "these two inputs should look the same" and in how they stop the encoder from cheating by making *everything* look the same.

## From energy to contrast

Here is the idea that turns "similar things should have similar features" into a loss function.

### Energies for pairs

Score how well a candidate $\mathbf{y}$ goes with an input $\mathbf{x}$ by an **energy** $E_\theta(\mathbf{x},\mathbf{y})$: **low** energy means **compatible** (an image and its own caption, two crops of the same photo), **high** energy means **incompatible**. Turn energies into probabilities the way physics does, with the Boltzmann distribution: the probability of a state falls exponentially with its energy, at a rate set by a temperature $\tau$,

$$
p_\theta(\mathbf{y}\mid\mathbf{x})
=\frac{\exp\!\big(-E_\theta(\mathbf{x},\mathbf{y})/\tau\big)}{\sum_{\mathbf{y}'}\exp\!\big(-E_\theta(\mathbf{x},\mathbf{y}')/\tau\big)} .
\tag{1}
$$

If the logits of [[Linear Models#More than two classes: softmax regression|softmax regression]] are read as negative energies, (1) is exactly the softmax. The [[Energy-Based Models]] lecture develops this view in full (where the Boltzmann form comes from, and how to train models whose normalizer is a sum over every image). Here we need only one consequence.

Training (1) by maximum likelihood makes two demands at once. The log-likelihood of an observed pair is

$$
\log p_\theta(\mathbf{y}\mid\mathbf{x})
=\underbrace{-\tfrac{1}{\tau}E_\theta(\mathbf{x},\mathbf{y})}_{\text{pull the true pair down}}
\;\underbrace{-\,\log\sum_{\mathbf{y}'}e^{-E_\theta(\mathbf{x},\mathbf{y}')/\tau}}_{\text{push every other candidate up}} .
$$

The first term alone is useless. If all we ask is low energy for compatible pairs, the encoder can give *every* pair low energy, for instance by mapping all inputs to the same vector. The second term, the **normalizer**, forbids this: making the true pair likely means making the alternatives unlikely.

### The catch, and the fix: a candidate set

The normalizer sums over **every possible** $\mathbf{y}$: every caption that could be written, every image that could be taken. That is hopeless. The fix is to **normalize over a small candidate set** instead: the true partner $\mathbf{y}^+$ plus $N-1$ **negatives** $\mathbf{y}^-_2,\dots,\mathbf{y}^-_N$, which are simply other examples, typically the rest of the minibatch. Use the energy

$$
E_\theta(\mathbf{x},\mathbf{y})=-\operatorname{sim}(\mathbf{z}_{\mathbf{x}},\mathbf{z}_{\mathbf{y}}),
$$

the negative cosine similarity of the embeddings. Then the negative log-likelihood of picking the true partner out of the candidates is

$$
\mathcal{L}_{\mathrm{InfoNCE}}
=-\log\frac{e^{\operatorname{sim}(\mathbf{z}_{\mathbf{x}},\mathbf{z}_{\mathbf{y}^+})/\tau}}
{e^{\operatorname{sim}(\mathbf{z}_{\mathbf{x}},\mathbf{z}_{\mathbf{y}^+})/\tau}+\sum_{j=2}^{N}e^{\operatorname{sim}(\mathbf{z}_{\mathbf{x}},\mathbf{z}_{\mathbf{y}^-_j})/\tau}} .
\tag{2}
$$

This is the **InfoNCE** loss.[^oord2018] Read it as an $N$-way classification problem: "which of these $N$ candidates belongs with $\mathbf{x}$?", solved by softmax regression whose logits are similarities. Everything in the first half of this lecture is (2) with different choices of what counts as a positive and where the negatives come from.

Its gradient makes the push and pull explicit. Writing $s_j=\operatorname{sim}(\mathbf{z}_{\mathbf{x}},\mathbf{z}_{\mathbf{y}_j})$ with $j=1$ the positive, and $p_j=e^{s_j/\tau}/\sum_l e^{s_l/\tau}$ the softmax,

$$
\frac{\partial\mathcal{L}_{\mathrm{InfoNCE}}}{\partial s_j}
=\frac{1}{\tau}\big(p_j-\mathbb{1}[j=1]\big).
$$

The positive's similarity is pushed **up** with force $(1-p_1)/\tau$; each negative's similarity is pushed **down** with force $p_j/\tau$, proportional to how plausible the model currently finds it. **Hard negatives**, wrong candidates that look right, get the largest push. And the more negatives there are, the better the candidate set stands in for the full normalizer. This is why contrastive methods want large batches.

> [!tip] Intuition
> Picture an energy landscape over all pairs. A good one has valleys at the compatible pairs and hills everywhere else. Pulling positives down digs the valleys; pushing negatives up raises the hills. Every method in this lecture digs valleys the same way. They differ in **where they find points to push up**: the rest of the batch (SimCLR), a queue of old embeddings (MoCo), class prototypes (cross-entropy), or nowhere at all, with some other mechanism preventing a flat landscape (BYOL, Barlow Twins).

An older, margin-based form makes the same point without a softmax.[^hadsell2006] Take the distance $r=\lVert\mathbf{z}-\mathbf{z}'\rVert$ between two embeddings as the energy, and minimize

$$
\ell(\mathbf{z},\mathbf{z}')=
\begin{cases}
\tfrac12 r^2 & \text{similar pair},\\
\tfrac12\max(0,\,\delta-r)^2 & \text{dissimilar pair}.
\end{cases}
$$

Positives are pulled into a valley; negatives are pushed away only until they clear a margin $\delta$. Without the second case, $r=0$ for all pairs is optimal: collapse again. The `version0.py` script of the code repository trains a variant of this pairwise loss on MNIST, with a binary cross-entropy on "same digit?" in place of the margin.

### NCE and InfoNCE

The name comes from **noise-contrastive estimation** (NCE).[^gutmann2010] NCE was invented to fit an unnormalized density $\tilde p_\theta(\mathbf{x})=e^{-E_\theta(\mathbf{x})}$ by turning density estimation into **binary** classification: tell data points from samples of a known noise distribution $p_{\mathrm{n}}$. With $\nu$ noise samples per data point, logistic regression with the logit

$$
\log\tilde p_\theta(\mathbf{x})+b-\log\big(\nu\,p_{\mathrm{n}}(\mathbf{x})\big)
$$

is fitted, where the log-normalizer $b$ is just another learned parameter. Gutmann and Hyvärinen proved that, as the sample grows, the fitted $\tilde p_\theta e^{b}$ converges to the true normalized density. The [[Score-Based Models]] lecture meets NCE again as an alternative to score matching.

InfoNCE changes two things:

| | NCE | InfoNCE |
|---|---|---|
| task | binary: is this point data or noise? | $N$-way: which candidate is the true partner? |
| what is modelled | a density $p(\mathbf{x})$ | a compatibility of pairs, $p(\mathbf{y}\mid\mathbf{x})/p(\mathbf{y})$ |
| negatives | from a known noise density $p_{\mathrm{n}}$, which must be evaluated | other data points, which need no density |
| goal | a consistent density estimate | a useful encoder |

The second row deserves a word. With enough capacity, the critic that minimizes (2) satisfies $e^{-E^*(\mathbf{x},\mathbf{y})/\tau}\propto p(\mathbf{y}\mid\mathbf{x})/p(\mathbf{y})$: the density *ratio* of "drawn together" to "drawn independently".[^oord2018] The negatives can come from the data precisely because the ratio's denominator is the data's own marginal.

### What InfoNCE estimates: mutual information, capped at log N

That density ratio is the integrand of the **mutual information**,

$$
I(X;Y)=\mathbb{E}_{p(\mathbf{x},\mathbf{y})}\Big[\log\frac{p(\mathbf{y}\mid\mathbf{x})}{p(\mathbf{y})}\Big],
$$

the expected number of nats that observing $\mathbf{x}$ tells us about $\mathbf{y}$ (zero iff they are independent). van den Oord et al. showed that InfoNCE bounds it from below,[^oord2018] which gave the loss its name:

$$
I(X;Y)\;\ge\;\log N-\mathcal{L}_{\mathrm{InfoNCE}} .
\tag{3}
$$

Minimizing the loss maximizes a lower bound on the information the two views share. The bound has a ceiling: since $\mathcal{L}_{\mathrm{InfoNCE}}\ge 0$, the right-hand side never exceeds $\log N$. With $N=4096$ candidates, InfoNCE cannot certify more than $\log 4096\approx 8.3$ nats, however much information the views share. This is not a weakness of InfoNCE in particular: McAllester and Stratos proved that *any* distribution-free high-confidence lower bound on MI needs a number of samples exponential in the MI.[^mcallester2018]

> [!proof]- Proof of the bound (3)
> Set up the classification problem behind (2). Draw $\mathbf{x}\sim p(\mathbf{x})$, pick the index $J$ of the positive uniformly from $\{1,\dots,N\}$, draw $\mathbf{y}_J\sim p(\mathbf{y}\mid\mathbf{x})$ and draw the other $N-1$ candidates independently from $p(\mathbf{y})$. The InfoNCE loss with critic $s(\mathbf{x},\mathbf{y})$ is the expected cross-entropy between the true index $J$ and the model's guess $\operatorname{softmax}_j s(\mathbf{x},\mathbf{y}_j)$.
>
> **Step 1: cross-entropy is at least the conditional entropy.** For any guess, the expected cross-entropy is at least $H(J\mid X,Y_{1:N})$, with equality for the Bayes posterior (cross-entropy = entropy + KL, and KL $\ge0$). So
>
> $$
> \mathcal{L}_{\mathrm{InfoNCE}}\ge H(J\mid X,Y_{1:N})=\log N-I(J;X,Y_{1:N}),
> $$
>
> because $H(J)=\log N$ for a uniform $J$.
>
> **Step 2: the index carries no more information than the pair.** Without $\mathbf{x}$, every $\mathbf{y}_j$ has marginal $p(\mathbf{y})$ whichever index is the positive, so $I(J;Y_{1:N})=0$ and, by the chain rule, $I(J;X,Y_{1:N})=I(J;X\mid Y_{1:N})$. The chain rule also gives
>
> $$
> \begin{aligned}
> I(X;J,Y_{1:N})&=I(X;Y_{1:N})+I(X;J\mid Y_{1:N})\ge I(X;J\mid Y_{1:N}),\\
> I(X;J,Y_{1:N})&=I(X;J)+I(X;Y_{1:N}\mid J)=0+I(X;Y).
> \end{aligned}
> $$
>
> In the second line, $X$ alone says nothing about $J$, and given $J$, $X$ is linked only to $\mathbf{y}_J$, whose joint law with $\mathbf{x}$ is $p(\mathbf{x},\mathbf{y})$. Together, $I(J;X,Y_{1:N})\le I(X;Y)$.
>
> **Step 3.** Combine: $\mathcal{L}_{\mathrm{InfoNCE}}\ge\log N-I(X;Y)$, which is (3). The same argument with $I(J;X,Y_{1:N})\le H(J)=\log N$ shows that the loss is non-negative, so the bound is capped at $\log N$. Poole et al. place this bound in a family of variational MI bounds and discuss its bias–variance trade-off.[^poole2019] $\blacksquare$

![Three curves of the InfoNCE estimate against the true mutual information of correlated Gaussians. Each follows the diagonal at first and then flattens at log N: 2.1 for N=8, 4.2 for N=64 and 6.2 for N=512.](Machine%20Learning/images/self-supervised-representation-learning-mi-bound.svg)

*Figure 1. InfoNCE with the **optimal** critic, $\log p(\mathbf{y}\mid\mathbf{x})/p(\mathbf{y})$, on 10-dimensional Gaussians with correlation $\rho$ per coordinate, so that the true MI is $-5\log(1-\rho^2)$. Even with a perfect critic, the estimate tracks the truth only while it is well below $\log N$ (dotted). At a true MI of 9.8 nats, the estimates are 2.08, 4.14 and 6.11 nats for $N=8,64,512$.*

> [!warning] Maximizing MI is not the whole story
> It is tempting to read contrastive learning as "maximize the information between views". But MI is invariant to any invertible transformation of the features, including ones that make them useless for a linear probe. Tschannen et al. showed that tighter MI bounds do not give better representations, and that the success of these methods depends heavily on the encoder architecture and the shape of the critic.[^tschannen2019] The geometric view below, alignment and uniformity, explains more.

### Temperature

The temperature $\tau$ in (2) controls how sharply the softmax concentrates. Dividing cosines in $[-1,1]$ by $\tau=0.1$ stretches them to logits in $[-10,10]$, so the few negatives most similar to the anchor take almost all of the gradient. A large $\tau$ spreads the push evenly over all negatives.[^wang2021]
- **Small $\tau$** is hardness-aware: it separates near-duplicates and spreads the embeddings out, but it also pushes apart **false negatives**, different images of the same kind of object, which we would rather keep close.
- **Large $\tau$** is tolerant: semantically similar samples stay together, but the embedding space is used less evenly.

Typical values are $\tau\in[0.05,0.5]$. CLIP learns $\tau$ as a parameter.

![Left: gradient weight on a negative against its similarity to the anchor, on a log scale, for temperatures 0.05, 0.2 and 1; the low-temperature curve rises by many orders of magnitude. Right: alignment, uniformity and kNN accuracy of a toy SimCLR as the temperature varies.](Machine%20Learning/images/self-supervised-representation-learning-temperature.svg)

*Figure 2. Left: the weight $p_j$ a negative receives (relative to the uniform weight $1/N$) as a function of its similarity, for 4096 negatives with typical similarities. At $\tau=0.05$ the top 1% of negatives carry 96% of the total push; at $\tau=0.2$, 17%; at $\tau=1$, only 2%. Right: SimCLR on the toy data of Figure 3 with 8-dimensional embeddings (3 seeds per point). Lowering $\tau$ from 2 to 0.1 trades alignment (0.017 → 0.164) for uniformity (−1.87 → −3.02); below 0.05 both get worse again. On this easy toy the 10-NN accuracy barely moves (97.3–97.8%); on real data the choice matters more, as the temperature ablations of the SimCLR paper show.[^chen2020]*

## Learning from labels with a contrastive loss

Before throwing the labels away, use them. Suppose we have a labelled dataset and want an encoder whose embeddings **cluster by class**. Contrast fits naturally: for an anchor of class $k$, every other example of class $k$ is a positive and everything else is a negative.

### Supervised contrastive loss

Take a batch of $2B$ embeddings (each image augmented twice, for good measure) with labels $y_i$. For anchor $i$, let $P(i)$ be the other samples with the same label and $A(i)$ all samples except $i$. The **supervised contrastive** (SupCon) loss of Khosla et al.[^khosla2020] averages the InfoNCE log-probability over all positives:

$$
\mathcal{L}_{\mathrm{SupCon}}
=\sum_{i}\frac{-1}{|P(i)|}\sum_{i^+\in P(i)}
\log\frac{e^{\operatorname{sim}(\mathbf{z}_i,\mathbf{z}_{i^+})/\tau}}{\sum_{a\in A(i)}e^{\operatorname{sim}(\mathbf{z}_i,\mathbf{z}_a)/\tau}} .
\tag{4}
$$

If each sample's only "classmate" is its own second view, $P(i)$ has one element and (4) is exactly the self-supervised loss (5) below. SupCon is the bridge: the same loss, with the labels deciding who counts as a positive.

Khosla et al. trained a ResNet with (4), froze it, and fitted a linear classifier on top. This beat end-to-end cross-entropy training on ImageNet, and was more robust to image corruptions and to hyperparameter choices.[^khosla2020]

### Cross-entropy is contrast against class prototypes

Compare with ordinary classification. A softmax classifier on normalized features, with a weight vector $\mathbf{w}_k$ per class, minimizes

$$
-\log\frac{e^{\operatorname{sim}(\mathbf{z}_i,\mathbf{w}_{y_i})/\tau}}{\sum_{k=1}^{K}e^{\operatorname{sim}(\mathbf{z}_i,\mathbf{w}_k)/\tau}} .
$$

This is InfoNCE again, with the anchor contrasted against $K$ learned **prototypes**, one per class, instead of against other samples. The two losses push towards the same geometry. In both, the optimum puts all samples of a class at one point, with the class points spread as far apart as possible on the sphere (for $d\ge K-1$, the vertices of a regular simplex). For cross-entropy this is the **neural collapse** observed at the end of training;[^papyan2020] for SupCon it was proved by Graf et al.[^graf2021]

The difference is in how they get there. The cross-entropy gradient on sample $i$ involves only the $K$ prototypes. SupCon compares each sample with many other samples, positives and negatives, and like (2) puts the most weight on the **hard** ones: the classmates that are far away and the other-class samples that are close.

![Three panels of points on a circle coloured by class. With SimCLR, each class covers an arc and the arcs tile the whole circle. With supervised contrastive learning, each class shrinks to a tight cluster. With prototype cross-entropy, the clusters are tight but less so.](Machine%20Learning/images/self-supervised-representation-learning-supcon.svg)

*Figure 3. A toy encoder (a 2-layer MLP) trained with three losses to embed 16-dimensional synthetic data on the unit circle. The data mixes 6 classes of "content" with a large random "style"; augmentations redraw the style. Each class is drawn on its own ring, slightly outside the circle, so that overlaps stay visible. SimCLR never sees a label, yet each class ends up as an arc, because only content survives augmentation. SupCon packs each class into a tight cluster (mean cosine to the class centre 0.993, against 0.966 for SimCLR and 0.978 for cross-entropy). 10-NN accuracy on held-out data, mean of 3 seeds: SimCLR 95.8%, SupCon 97.7%, cross-entropy 97.1%; the raw 16-dimensional inputs give 95.3%.*

## Self-supervised contrastive learning

Now remove the labels. Where do positives come from? From **augmentation**. Two random augmentations $t(\mathbf{x})$ and $t'(\mathbf{x})$ of the same image are declared a positive pair; augmentations of different images are negatives. This is **instance discrimination**: every image is its own class.[^dosovitskiy2014]<sup>,</sup>[^wu2018] The labels that SupCon took from annotators are now generated by the augmentation pipeline, and the choice of augmentations *defines* which invariances the encoder learns.

### SimCLR

SimCLR[^chen2020] is the simplest complete recipe, and nearly every later method is a variation of it (Figure 4, top).
1. Sample a batch of $B$ images. Augment each twice, giving $2B$ views.
2. Encode each view, $\mathbf{h}=f_\theta(t(\mathbf{x}))$, and project it with a small MLP, $\mathbf{z}=g_\theta(\mathbf{h})$.
3. For each view $i$, the other view of the same image, $i^+$, is the positive and the remaining $2B-2$ views are negatives. Minimize the loss averaged over all $2B$ anchors (**NT-Xent**, normalized temperature-scaled cross-entropy):

$$
\mathcal{L}_{\mathrm{SimCLR}}
=\frac{1}{2B}\sum_{i=1}^{2B}-\log\frac{e^{\operatorname{sim}(\mathbf{z}_i,\mathbf{z}_{i^+})/\tau}}{\sum_{l\ne i}e^{\operatorname{sim}(\mathbf{z}_i,\mathbf{z}_l)/\tau}} .
\tag{5}
$$

4. After training, throw $g_\theta$ away and use $\mathbf{h}$.

The paper's ablations taught three lessons that have held up:
- **Augmentations matter most.** No single augmentation is enough. The composition of **random cropping** with **colour distortion** was critical: with crops alone, two crops of an image can be matched by their colour histograms, a shortcut that teaches nothing about objects.
- **The projection head helps, and its input is the better representation.** A linear probe on $\mathbf{h}$ was more than 10 percentage points better than on $\mathbf{z}$. The loss makes $\mathbf{z}$ invariant to the augmentations, discarding information (colour, orientation) that downstream tasks may need; the head absorbs this loss of information.
- **More negatives help.** Batches of up to 8192 (16382 negatives per anchor) and long training kept improving results, which required large accelerator clusters.

A linear probe on SimCLR features of a ResNet-50 with four times the usual width reached 76.5% top-1 accuracy on ImageNet, matching a supervised ResNet-50.[^chen2020]

![Three pipeline schematics. SimCLR: two views pass through a shared encoder and head into an InfoNCE loss. MoCo: a query encoder and a momentum key encoder, with a queue of old keys feeding the negatives. BYOL and SimSiam: an online branch with a predictor and a target branch behind a stop-gradient, compared by a negative cosine.](Machine%20Learning/images/self-supervised-representation-learning-pipelines.svg)

*Figure 4. Three ways to train on pairs of views. SimCLR takes its negatives from the batch; MoCo from a queue of keys made by a slowly moving copy of the encoder; BYOL and SimSiam use no negatives and break the symmetry between the branches instead.*

> [!example]- Code: InfoNCE and supervised contrastive losses in PyTorch
> Both losses as single matrix operations. `supcon` with each sample's second view as its only positive reduces to `info_nce`; we checked this, and both losses, against explicit loops.
> ```python
> import torch
> import torch.nn.functional as F
>
> def info_nce(z1, z2, tau=0.1):
>     """SimCLR loss. Row i of z1 and row i of z2 embed two views of image i."""
>     z = F.normalize(torch.cat([z1, z2]), dim=1)        # 2B points on the sphere
>     B = z1.shape[0]
>     logits = z @ z.T / tau                              # cosine similarity / temperature
>     logits.fill_diagonal_(float("-inf"))                # a view is not its own negative
>     target = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)])
>     return F.cross_entropy(logits, target)              # softmax over the other 2B-1 views
>
> def supcon(z, y, tau=0.1):
>     """Supervised contrastive loss: all other samples with label y[i] are positives."""
>     z = F.normalize(z, dim=1)
>     n = z.shape[0]
>     self_mask = torch.eye(n, dtype=torch.bool, device=z.device)
>     logits = (z @ z.T / tau).masked_fill(self_mask, float("-inf"))
>     log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
>     pos = (y[:, None] == y[None, :]) & ~self_mask
>     n_pos = pos.sum(1)
>     keep = n_pos > 0                                     # anchors with at least one positive
>     loss = -log_prob.masked_fill(~pos, 0.0).sum(1)[keep] / n_pos[keep]
>     return loss.mean()
> ```

### MoCo: many negatives without a huge batch

SimCLR's appetite for large batches is a hardware problem: all $2B$ views must be encoded, with gradients, in the same step. **Momentum Contrast** (MoCo)[^he2020moco] decouples the number of negatives from the batch size (Figure 4, middle).
- Keep a **queue** of the keys (embeddings) of the last few batches, typically 65536 of them, and use them as negatives. Each step, enqueue the newest batch of keys and drop the oldest.
- Encode the anchor ("query") with the trained encoder $f_\theta$, and the positive and queued keys with a separate **key encoder** $f_\xi$ that receives no gradients.
- Update the key encoder as an exponential moving average of the query encoder:

$$
\xi\leftarrow m\,\xi+(1-m)\,\theta,\qquad m=0.999 .
$$

Why not simply let the query encoder produce the keys? Because the queue holds keys computed at many different steps. If the encoder changed quickly, the keys in the queue would come from many different encoders, and the anchor would be contrasted against a set of mutually inconsistent points: the model could tell old keys from new ones by their "age" instead of their content. He et al. found that copying the query encoder ($m=0$) made training fail, while $m=0.999$ worked best. A slowly moving key encoder makes all keys in the queue nearly comparable.

> [!example]- Code: the MoCo loss with a queue and a momentum encoder (after `version3.py`)
> ```python
> import torch
> import torch.nn.functional as F
>
> def moco_loss(encoder_q, encoder_k, x_q, x_k, queue, tau=0.07):
>     """queue: (H, d) normalized keys of earlier batches, used as the negatives."""
>     q = F.normalize(encoder_q(x_q), dim=1)                  # queries: the trained encoder
>     with torch.no_grad():
>         k = F.normalize(encoder_k(x_k), dim=1)              # keys: momentum encoder, no gradient
>     logits = torch.cat([(q * k).sum(1, keepdim=True),       # column 0: the positive key
>                         q @ queue.T], dim=1) / tau          # H negatives from the queue
>     target = torch.zeros(len(q), dtype=torch.long)
>     return F.cross_entropy(logits, target), k
>
> @torch.no_grad()
> def momentum_update(encoder_k, encoder_q, m=0.999):          # xi <- m xi + (1 - m) theta
>     for pk, pq in zip(encoder_k.parameters(), encoder_q.parameters()):
>         pk.mul_(m).add_(pq, alpha=1 - m)
>
> # each step: loss, k = moco_loss(...); loss.backward(); opt.step(); momentum_update(...)
> #            queue = torch.cat([k, queue])[:H]               # enqueue the newest keys, drop the oldest
> ```
> Compared with SimCLR's `info_nce`, the negatives come from `queue` instead of the other views in the batch, so their number, `H`, no longer depends on the batch size. The new keys are enqueued only after the loss is computed, so a key is never its own negative.

MoCo v2 added SimCLR's projection head and stronger augmentations, and then outperformed SimCLR with a batch of only 256.[^chen2020mocov2] The `version3.py` script of the code repository implements MoCo on MNIST, and `version4.py` fine-tunes the MoCo encoder with only 1000 labels, reaching about 95% test accuracy against about 93% from a random initialization.

### Hard negatives and false negatives

Not all negatives are equally useful. Random negatives are mostly easy, so far from the anchor that they contribute almost nothing to the gradient. Two refinements:
- **Hard negative sampling** reweights negatives towards those most similar to the anchor, with an importance weight $\propto e^{\beta\operatorname{sim}}$. A low temperature does a softer version of the same thing.[^robinson2021]
- **Debiasing.** Some "negatives" are really images of the same class: a different dog photo is a false negative for a dog anchor. With $K$ roughly balanced classes, about $1/K$ of random negatives are false. Debiased contrastive learning corrects the negative term for this rate.[^chuang2020]

The two pull in opposite directions: the hardest negatives are disproportionately likely to be false negatives. Robinson et al. combine both corrections.[^robinson2021]

### Alignment and uniformity

What does the InfoNCE optimum look like? Wang and Isola split the loss into two properties of normalized embeddings on the unit sphere.[^wang2020] As the number of negatives $N\to\infty$, the loss (2) minus $\log N$ converges to

$$
\begin{aligned}
&-\tfrac{1}{\tau}\,\mathbb{E}_{(\mathbf{x},\mathbf{x}^+)}\big[\mathbf{z}^\top\mathbf{z}^+\big]\\
&+\mathbb{E}_{\mathbf{x}}\Big[\log\mathbb{E}_{\mathbf{x}^-}\big[e^{\mathbf{z}^\top\mathbf{z}^-/\tau}\big]\Big].
\end{aligned}
$$

The first term rewards **alignment**: positive pairs close together. The second rewards **uniformity**: it is smallest when the embeddings spread evenly over the sphere, so that a random negative is typically far from the anchor. They proposed two direct measures,

$$
\begin{aligned}
\mathcal{L}_{\mathrm{align}}&=\mathbb{E}_{(\mathbf{x},\mathbf{x}^+)}\lVert\mathbf{z}-\mathbf{z}^+\rVert^2,\\
\mathcal{L}_{\mathrm{unif}}&=\log\mathbb{E}_{\mathbf{x},\mathbf{x}'}\,e^{-2\lVert\mathbf{z}-\mathbf{z}'\rVert^2},
\end{aligned}
$$

and found that across many trained encoders, lower values of both predict better downstream accuracy. Optimizing the two directly works as well as InfoNCE.

> [!example]- Code: the alignment and uniformity losses
> ```python
> import torch
>
> def align_loss(z, z_pos):
>     """E ||z - z+||^2 over positive pairs; rows are L2-normalized embeddings."""
>     return (z - z_pos).pow(2).sum(1).mean()
>
> def uniform_loss(z, t=2.0):
>     """log E exp(-t ||z - z'||^2) over distinct pairs in the batch."""
>     return torch.pdist(z).pow(2).mul(-t).exp().mean().log()
> ```
> Both are cheap enough to log during training. For points spread evenly on the unit circle, `uniform_loss` approaches about $-1.575$, the floor quoted in Figure 5; a collapsed embedding gives 0.

This picture also explains collapse. Alignment alone is solved perfectly by mapping every input to the same point: $\mathcal{L}_{\mathrm{align}}=0$, with no information left. Uniformity alone is solved by scattering inputs at random. It is the tension between the two that produces a useful representation: spread out as much as possible, *except* along the directions the augmentations say do not matter.

![Four panels. Top left: embeddings at initialization bunched in one part of the circle. Top right: after InfoNCE training, spread around the circle with each class on an arc. Bottom left: with the alignment loss only, all points collapse to one spot. Bottom right: training trajectories in the plane of uniformity versus alignment loss.](Machine%20Learning/images/self-supervised-representation-learning-alignment-uniformity.svg)

*Figure 5. The toy encoder of Figure 3, with 2-dimensional embeddings. At initialization the embeddings bunch together in one region of the circle. InfoNCE training lowers the alignment loss from 0.176 to 0.017 and the uniformity loss from −0.33 to −1.57, within 0.002 of its minimum on the circle ($\approx-1.575$, dotted), and 10-NN accuracy is 96.5%. Minimizing the alignment loss alone sends both losses to 0: every input lands on the same point, and 10-NN accuracy falls to 30.1% (chance is 16.7%).*

### Why should any of this help downstream?

A downstream classifier never sees the augmentations, so why should contrastive features be linearly separable? A first answer came from Arora et al.: if positive pairs are drawn from the same latent class, the contrastive loss bounds the loss of the best linear classifier on those classes.[^arora2019] A more recent one, by HaoChen et al., connects contrastive learning with spectral clustering. Build a graph whose nodes are all possible augmented images, with edges between views of the same image. A variant of the contrastive loss then recovers the bottom eigenvectors of this graph's Laplacian, exactly the embedding of [[Clustering and Expectation Maximization|spectral clustering]], so contrastive learning approximately clusters the augmentation graph: views that are linked through chains of augmentations end up together.[^haochen2021]

## Non-contrastive learning and collapse

Negatives are expensive and slightly wrong (false negatives). Can we drop them? The naive version, minimize only the distance between the embeddings of two views, fails. The failure has a name.

### What collapse is

- **Complete collapse:** every input maps to (nearly) the same embedding. The loss on positives is zero and the representation is useless.
- **Dimensional collapse:** the embeddings vary, but only within a low-dimensional subspace of $\mathbb{R}^d$. Most directions carry no information. Jing et al. showed this happens even in contrastive methods: strong augmentations and the implicit low-rank bias of gradient descent in deep linear networks squeeze the spectrum of the embedding covariance.[^jing2022] It shows up as a steep drop in the singular values of the embedding matrix.

Each non-contrastive method needs a mechanism that rules out one or both. There are two families.

### Asymmetry: BYOL and SimSiam

**BYOL** (Bootstrap Your Own Latent)[^grill2020] has two networks (Figure 4, bottom).
- An **online** network: encoder, projector, and an extra MLP, the **predictor** $q_\theta$.
- A **target** network: encoder and projector only, with weights $\xi$ that are an exponential moving average of the online weights, as in MoCo.

The online network predicts the target's embedding of the *other* view, and the loss is the squared distance of the normalized vectors, $2-2\operatorname{sim}\big(q_\theta(\mathbf{z}_\theta),\mathbf{z}'_\xi\big)$, made symmetric over the two views. Gradients flow only into the online network. There are no negatives anywhere, yet BYOL beat SimCLR, with 74.3% ImageNet linear-probe accuracy for a ResNet-50.[^grill2020]

**SimSiam**[^chen2021simsiam] stripped BYOL down to find out what matters: no moving average (the target *is* the online encoder), no negatives, no large batch. Only two ingredients remained essential: the **predictor** and a **stop-gradient** on the target branch,

$$
\mathcal{L}=-\tfrac12\operatorname{sim}\big(q(\mathbf{z}_1),\operatorname{sg}(\mathbf{z}_2)\big)-\tfrac12\operatorname{sim}\big(q(\mathbf{z}_2),\operatorname{sg}(\mathbf{z}_1)\big),
$$

where $\operatorname{sg}$ treats its argument as a constant. Remove the stop-gradient and SimSiam collapses within a few epochs; remove the predictor and it collapses too.

Why does this work, when collapse is a perfectly good minimizer of the loss? The stop-gradient means the network is **not** doing gradient descent on any fixed loss. Chen and He read SimSiam as an alternating, EM-like procedure: one step fixes the targets and fits the predictor and encoder to them; the next recomputes the targets.[^chen2021simsiam] Tian et al. analysed the learning dynamics of linear versions and showed that the predictor aligns with the eigenbasis of the embedding correlation, and that the stop-gradient and predictor together keep the training dynamics away from the collapsed fixed point.[^tian2021] There is no complete theory for the nonlinear case.

> [!example]- Code: a BYOL step in PyTorch
> The online network and predictor are trained by any optimizer; call `update_target()` after each optimizer step. Setting `m=0` and sharing weights gives SimSiam. We checked that no gradient reaches the target and that the moving-average update is exact.
> ```python
> import copy
> import torch
> import torch.nn.functional as F
>
> def make_byol(online, predictor, m=0.996):
>     target = copy.deepcopy(online).requires_grad_(False)    # no gradients ever reach it
>
>     def loss_fn(x1, x2):
>         p1, p2 = predictor(online(x1)), predictor(online(x2))
>         with torch.no_grad():                                # stop-gradient
>             t1, t2 = target(x1), target(x2)
>         return (2 - 2 * F.cosine_similarity(p1, t2, dim=1).mean()
>                 + 2 - 2 * F.cosine_similarity(p2, t1, dim=1).mean())
>
>     @torch.no_grad()
>     def update_target():                                     # after each optimizer step
>         for pt, po in zip(target.parameters(), online.parameters()):
>             pt.mul_(m).add_((1 - m) * po)
>
>     return loss_fn, update_target, target
> ```

### Decorrelation: Barlow Twins and VICReg

The second family attacks collapse directly, with a loss term that is large for collapsed embeddings. Collapse means the embedding coordinates have no variance (complete) or are redundant copies of each other (dimensional). So penalize exactly that.

**Barlow Twins**[^zbontar2021] standardizes each embedding coordinate over the batch and computes the $d\times d$ **cross-correlation matrix** $\mathbf{C}$ between the two views, $C_{uv}=\frac{1}{B}\sum_i z^{(1)}_{iu}z^{(2)}_{iv}$. The loss drives $\mathbf{C}$ towards the identity:

$$
\mathcal{L}_{\mathrm{BT}}
=\sum_{u}\big(1-C_{uu}\big)^2+\lambda\sum_{u\ne v}C_{uv}^2 .
$$

- The diagonal term makes each coordinate agree across the two views: **invariance**.
- The off-diagonal term makes different coordinates uncorrelated: **redundancy reduction**, after the neuroscientist Horace Barlow's hypothesis that sensory coding removes redundancy. Uncorrelated coordinates cannot all be copies of one direction, which rules out dimensional collapse, and since they are standardized they cannot be constant.

**VICReg**[^bardes2022] (Variance–Invariance–Covariance Regularization) splits the same idea into three explicit terms for embeddings $\mathbf{Z},\mathbf{Z}'\in\mathbb{R}^{B\times d}$ of the two views:
- **Invariance:** the mean squared distance between $\mathbf{z}_i$ and $\mathbf{z}'_i$.
- **Variance:** a hinge $\frac1d\sum_u\max\big(0,\,1-\operatorname{std}(\mathbf{Z}_{:,u})\big)$ that keeps every coordinate's standard deviation over the batch at least 1. This term alone rules out complete collapse.
- **Covariance:** the sum of squared off-diagonal entries of the covariance matrix of $\mathbf{Z}$, divided by $d$. This decorrelates the coordinates, against dimensional collapse.

Unlike Barlow Twins, the variance and covariance terms are applied to each branch separately, so the two branches need not share weights or even modalities. Both methods are close relatives of [[Matrix Factorization and Dimensionality Reduction|PCA]]: they ask for decorrelated coordinates, and the redundancy-reduction terms are a soft form of whitening.

> [!example]- Code: the Barlow Twins and VICReg losses
> ```python
> import torch
> import torch.nn.functional as F
>
> def barlow_twins(z1, z2, lam=5e-3):
>     """z1, z2: (B, d) embeddings of two views of the same B images."""
>     z1 = (z1 - z1.mean(0)) / z1.std(0, correction=0)       # standardize each coordinate over the batch
>     z2 = (z2 - z2.mean(0)) / z2.std(0, correction=0)
>     C = z1.T @ z2 / len(z1)                                 # (d, d) cross-correlation matrix
>     on = ((1 - C.diagonal()) ** 2).sum()                    # invariance
>     off = (C ** 2).sum() - (C.diagonal() ** 2).sum()        # redundancy reduction
>     return on + lam * off
>
> def vicreg(z1, z2, lam=25.0, mu=25.0, nu=1.0, eps=1e-4):
>     d = z1.shape[1]
>     def var_cov(z):                                         # applied to each branch separately
>         z = z - z.mean(0)
>         var = F.relu(1 - torch.sqrt(z.var(0) + eps)).mean() # hinge: every std >= 1
>         cov = z.T @ z / (len(z) - 1)
>         return var, ((cov ** 2).sum() - (cov.diagonal() ** 2).sum()) / d
>     (v1, c1), (v2, c2) = var_cov(z1), var_cov(z2)
>     return lam * F.mse_loss(z1, z2) + mu * (v1 + v2) / 2 + nu * (c1 + c2)
> ```
> The weights are the papers' defaults. A collapsed embedding makes the Barlow Twins diagonal term and the VICReg variance hinge large, and copies of one direction make the off-diagonal terms large; nothing in either loss needs a negative pair.

![Left: per-coordinate standard deviation of normalized embeddings during training for four variants; SimSiam and VICReg stay away from zero, while no-stop-gradient and no-predictor fall towards zero. Right: singular value spectra of the final embeddings, on a log scale.](Machine%20Learning/images/self-supervised-representation-learning-collapse.svg)

*Figure 6. A tiny SimSiam on synthetic data (32-dimensional inputs, 16-dimensional embeddings). The left panel tracks the average standard deviation of the normalized embeddings, scaled so that 1 means spread evenly over the sphere and 0 means complete collapse. Without the stop-gradient (0.011) or without the predictor (0.018), the embeddings collapse within the first hundred steps. SimSiam (0.42) and VICReg (0.89) do not. The right panel shows the singular values of the final normalized, centred embeddings. Both collapsed variants sit around $10^{-2}$ or below in every direction. SimSiam avoids complete collapse but shows **dimensional** collapse: one direction dominates (singular values 0.47, 0.08, 0.03, …). VICReg keeps nine roughly equal directions. 10-NN accuracy on the 10 classes: VICReg 99.3%, SimSiam 62.9%, no predictor 76.3%, no stop-gradient 41.1%. (Cosine 10-NN can still rank neighbours in a nearly collapsed embedding, but the tiny remaining variation is fragile.) One seed each and no batch normalization: the real methods behave better than this toy SimSiam, but the qualitative picture matches the ablations of Chen and He.*

### Self-distillation: DINO

**DINO**[^caron2021] (self-**di**stillation with **no** labels) treats the moving-average network as a teacher and the trained one as a student. Both end in a softmax over $K$ "prototype" dimensions ($K=65536$), and the student is trained to match the teacher's output distribution on a different view by cross-entropy, $-\sum_k p^{\text{teacher}}_k\log p^{\text{student}}_k$. This is the classifier-with-prototypes picture from SupCon again, except that the "labels" are the teacher's own soft predictions. It is close to the clustering-based SwAV, which assigns views to learned prototypes and predicts one view's assignment from the other.[^caron2020]

Two failure modes threaten a self-distillation loss: every image put in the same prototype, or a uniform output that says nothing. DINO counters them with a pair of opposing operations on the teacher's logits:
- **Centring:** subtract a running mean $\mathbf{c}$ of the teacher's logits over batches. No single prototype can win for every image, but on its own this pushes the output towards uniform.
- **Sharpening:** a low teacher temperature (0.04–0.07, against 0.1 for the student) makes the teacher's output peaked, pushing away from uniform.

DINO also uses **multi-crop**: the student sees two large "global" crops and several small "local" crops, the teacher only the global ones, so the student must infer the whole from a part. Trained on ViTs, DINO's features have remarkable properties: the self-attention maps of the class token segment the main object without any segmentation labels, and $k$-NN classification on the frozen features is nearly as good as a linear probe.[^caron2021]

**DINOv2**[^oquab2024] scaled the recipe up to a curated dataset of 142 million images and added a masked-token prediction loss from iBOT[^zhou2022] (next section). Its frozen features are used as general-purpose visual features, with only a linear layer on top, for classification, segmentation and depth estimation.

## Predictive self-supervised learning

A different pretext task needs no pairs at all: **hide part of the input and predict it**.

### Masked autoencoders

A **masked autoencoder** (MAE)[^he2022mae] splits an image into patches, as in the [[Sequence Models#The Vision Transformer|Vision Transformer]], and drops a random **75%** of them.
- A ViT **encoder** processes only the visible 25% of the patches. This alone makes pre-training three or more times faster.
- A small **decoder** receives the encoded visible patches plus a learned "mask token" at each missing position, and reconstructs the missing pixels (normalized per patch).
- The loss is the mean squared error on the **masked** patches only.

The high masking ratio is the key design choice. Images are spatially redundant: with 15% of patches missing, they can be filled in by interpolating their neighbours, which needs no understanding of objects. With 75% missing, the model must infer what is in the image. A ViT-Huge pre-trained this way and then fine-tuned reached 87.8% accuracy on ImageNet-1K, using no data beyond ImageNet. MAE features are best **fine-tuned**: their linear-probe accuracy lags behind contrastive and self-distillation methods, because the encoder also keeps the low-level detail that pixel reconstruction needs.

The same recipe on text, mask some of the words and predict them, is **masked language modelling**, the pre-training of BERT. It is developed in [[Language Representations and Language Models]]. (Text needs a much lower masking ratio: each word carries far more information than an image patch.)

### JEPA: predict representations, not pixels

Pixel reconstruction spends capacity on details that are inherently unpredictable, like the exact texture of grass. The **joint-embedding predictive architecture** (JEPA) idea is to predict the hidden part **in representation space** instead. **I-JEPA**[^assran2023] encodes a large context block of the image, and a predictor (a narrow ViT) predicts the representations of several target blocks, given their positions. The targets are produced by an exponential-moving-average target encoder applied to the full image, as in BYOL, and the loss is the distance between predicted and target representations.

I-JEPA combines the two families. Like MAE, it needs no hand-crafted augmentations; like BYOL, it compares representations and must avoid collapse, which it does with the moving-average target and predictor. Its linear-probe results were competitive with augmentation-based methods at lower compute. The **improved-vision-encoders** repository builds on this design: 75% random patch masking, an encoder–predictor pair, and explicit anti-collapse terms (a variance loss in the spirit of VICReg and a sliced-Wasserstein loss), monitored by a linear probe trained alongside.

## Multimodal contrastive learning: CLIP

The positive pair need not come from one modality. The web is full of images with captions, and an image and its caption describe the same scene. **CLIP**[^radford2021] (Contrastive Language–Image Pre-training) trains an image encoder and a text encoder (a transformer; see [[Sequence Models]]) on 400 million such pairs.
- In a batch of $B$ pairs, compute the $B\times B$ matrix of cosine similarities between all image embeddings and all text embeddings, scaled by a learned temperature.
- The diagonal holds the true pairs. Apply InfoNCE (2) along each **row** (for each image, find its caption among $B$ captions) and along each **column** (for each caption, find its image), and average the two. This is a **symmetric** InfoNCE.

The batch size was 32768, so each image is contrasted with 32767 wrong captions.

**Zero-shot classification** falls out for free. To classify an image among classes nobody trained it on, turn each class name into a caption ("a photo of a {class}."), embed all the captions, and predict the class whose caption is most similar to the image. The class names act as the prototypes of the cross-entropy picture, written in language instead of learned. Zero-shot CLIP matched the 76.2% ImageNet accuracy of a supervised ResNet-50 without using any of its 1.28 million labelled training images, and was much more robust to distribution shifts such as sketches and renditions.[^radford2021] The wording of the prompt matters, and averaging the embeddings of several prompts per class ("a photo of a big {class}", "a drawing of a {class}") helps.

The softmax in CLIP's loss normalizes over the whole batch, which couples all devices in a distributed run. **SigLIP**[^zhai2023] replaces it with an independent **sigmoid** (binary) loss on every image–text pair, positive on the diagonal and negative elsewhere, which is the NCE-style binary view of the same problem. It works as well or better, especially at smaller batch sizes.

> [!example]- Code: CLIP's symmetric InfoNCE, and SigLIP's sigmoid version
> ```python
> import torch
> import torch.nn.functional as F
>
> def clip_loss(img, txt, log_scale):
>     """img, txt: (B, d); row i of each is one image-caption pair. log_scale = log(1/tau), learned."""
>     I, T = F.normalize(img, dim=1), F.normalize(txt, dim=1)
>     logits = log_scale.exp() * I @ T.T                     # (B, B) scaled cosine similarities
>     target = torch.arange(len(I))                           # true pairs on the diagonal
>     return (F.cross_entropy(logits, target)                 # each image: find its caption (rows)
>             + F.cross_entropy(logits.T, target)) / 2        # each caption: find its image (columns)
>
> def siglip_loss(img, txt, log_scale, bias):
>     """An independent binary classification for each of the B*B pairs: no softmax over the batch."""
>     I, T = F.normalize(img, dim=1), F.normalize(txt, dim=1)
>     logits = log_scale.exp() * I @ T.T + bias
>     labels = 2 * torch.eye(len(I)) - 1                      # +1 on the diagonal, -1 elsewhere
>     return -F.logsigmoid(labels * logits).sum() / len(I)
>
> # zero-shot: class_txt = text_encoder(["a photo of a {c}." for c in classes])
> # pred = (F.normalize(img, dim=1) @ F.normalize(class_txt, dim=1).T).argmax(1)
> ```

## Evaluation and practice

**Evaluation protocols.**
- Report linear-probe accuracy on frozen features (the standard ImageNet benchmark), $k$-NN accuracy (cheap, no hyperparameters beyond $k$), and fine-tuning accuracy, ideally also in the **low-label** regime (1% or 10% of the labels), where pre-training matters most.
- Use the representation **before** the projection head.
- Methods rank differently under different protocols: MAE is weak under linear probing but strong after fine-tuning; DINO is strong under $k$-NN.
- Test transfer to other datasets and tasks, not only ImageNet.

**Practical choices.**
- **Augmentations** are the most important hyperparameter of a joint-embedding method, because they define the invariances. Random resized crops plus colour jitter, grayscale and blur are the image default. Do not make the encoder invariant to something a downstream task needs: rotation invariance is harmful when orientation matters.
- **Batch size and negatives:** SimCLR needs thousands; MoCo gets them from a queue; BYOL, SimSiam, Barlow Twins and VICReg need none, though the batch statistics in Barlow Twins and VICReg still need batches of a few hundred.
- **Temperature:** around 0.1 for SimCLR and SupCon, 0.07 for MoCo, learned in CLIP.
- **Projection heads** of 2–3 layers, and for Barlow Twins and VICReg, wide ones (thousands of dimensions).
- **Monitor for collapse:** track the standard deviation of the normalized embeddings, or the singular value spectrum of a batch of embeddings, as in Figure 6. The loss alone will not tell you.

Balestriero et al.'s *Cookbook of Self-Supervised Learning* collects these recipes, and many more, in one place.[^balestriero2023]

## Key takeaways

- Self-supervised learning manufactures a training signal from unlabelled data. Its product is an encoder, judged by linear probes, $k$-NN and fine-tuning.
- A contrastive loss is a softmax over energies, normalized over a small **candidate set** instead of over everything: pull the positive pair together, push the negatives apart, weighted by how plausible they look. This is InfoNCE. It lower-bounds mutual information, but never by more than $\log N$.
- With labels deciding the positives it is **supervised contrastive learning**, which, like cross-entropy against class prototypes, packs each class into a tight cluster. With augmentations deciding them it is **SimCLR**; **MoCo** takes its negatives from a queue kept consistent by a slowly moving key encoder.
- The InfoNCE optimum balances **alignment** (positives together) and **uniformity** (everything spread out). Alignment alone collapses.
- Non-contrastive methods prevent collapse with asymmetry (predictor and stop-gradient in **BYOL** and **SimSiam**; centring and sharpening in **DINO**) or with decorrelation (**Barlow Twins**, **VICReg**). Dimensional collapse is subtler than complete collapse; watch the spectrum.
- Predictive methods hide part of the input: **MAE** reconstructs pixels, **JEPA** predicts representations.
- **CLIP** is symmetric InfoNCE across images and text; class names written as captions give zero-shot classifiers.

## Where this shows up next

- [[Unsupervised Domain Adaptation]]: representations that must work on a new, unlabelled domain. Its contradistinguisher (CUDA) normalizes predictions over a batch of target points, as InfoNCE normalizes over candidates.
- [[Language Representations and Language Models]]: the same ideas on text. Word2Vec's negative sampling is a binary contrastive loss over words and contexts; masked language modelling is MAE for words; next-token prediction is the self-supervised task behind GPT.
- [[Energy-Based Models]]: the full energy-based picture behind (1), including how to train when the normalizer is a sum over all inputs and no candidate set is available.
- [[Score-Based Models]]: noise-contrastive estimation and denoising as two more ways to learn without a normalizer.
- [[Clustering and Expectation Maximization]]: spectral clustering, which contrastive learning implicitly performs on the augmentation graph, and the prototype-based clustering that SwAV and DINO build in.
- [[Matrix Factorization and Dimensionality Reduction]]: PCA and whitening, the linear ancestors of the decorrelation losses.

<!-- References: footnote definitions below render as the numbered "References" list at the bottom of the page -->

[^alain2016]: Alain, G., & Bengio, Y. (2016). *Understanding Intermediate Layers Using Linear Classifier Probes*. [arXiv:1610.01644](https://arxiv.org/abs/1610.01644)
[^wu2018]: Wu, Z., Xiong, Y., Yu, S. X., & Lin, D. (2018). *Unsupervised Feature Learning via Non-Parametric Instance Discrimination*. CVPR. [arXiv:1805.01978](https://arxiv.org/abs/1805.01978)
[^oord2018]: van den Oord, A., Li, Y., & Vinyals, O. (2018). *Representation Learning with Contrastive Predictive Coding*. [arXiv:1807.03748](https://arxiv.org/abs/1807.03748)
[^hadsell2006]: Hadsell, R., Chopra, S., & LeCun, Y. (2006). *Dimensionality Reduction by Learning an Invariant Mapping*. CVPR. [doi:10.1109/CVPR.2006.100](https://doi.org/10.1109/CVPR.2006.100)
[^gutmann2010]: Gutmann, M., & Hyvärinen, A. (2010). *Noise-Contrastive Estimation: A New Estimation Principle for Unnormalized Statistical Models*. AISTATS. [proceedings.mlr.press/v9/gutmann10a](https://proceedings.mlr.press/v9/gutmann10a.html)
[^mcallester2018]: McAllester, D., & Stratos, K. (2020). *Formal Limitations on the Measurement of Mutual Information*. AISTATS. [arXiv:1811.04251](https://arxiv.org/abs/1811.04251)
[^poole2019]: Poole, B., Ozair, S., van den Oord, A., Alemi, A. A., & Tucker, G. (2019). *On Variational Bounds of Mutual Information*. ICML. [arXiv:1905.06922](https://arxiv.org/abs/1905.06922)
[^tschannen2019]: Tschannen, M., Djolonga, J., Rubenstein, P. K., Gelly, S., & Lucic, M. (2020). *On Mutual Information Maximization for Representation Learning*. ICLR. [arXiv:1907.13625](https://arxiv.org/abs/1907.13625)
[^wang2021]: Wang, F., & Liu, H. (2021). *Understanding the Behaviour of Contrastive Loss*. CVPR. [arXiv:2012.09740](https://arxiv.org/abs/2012.09740)
[^khosla2020]: Khosla, P., Teterwak, P., Wang, C., Sarna, A., Tian, Y., Isola, P., Maschinot, A., Liu, C., & Krishnan, D. (2020). *Supervised Contrastive Learning*. NeurIPS. [arXiv:2004.11362](https://arxiv.org/abs/2004.11362)
[^papyan2020]: Papyan, V., Han, X. Y., & Donoho, D. L. (2020). *Prevalence of Neural Collapse During the Terminal Phase of Deep Learning Training*. PNAS. [arXiv:2008.08186](https://arxiv.org/abs/2008.08186)
[^graf2021]: Graf, F., Hofer, C. D., Niethammer, M., & Kwitt, R. (2021). *Dissecting Supervised Contrastive Learning*. ICML. [arXiv:2102.08817](https://arxiv.org/abs/2102.08817)
[^dosovitskiy2014]: Dosovitskiy, A., Fischer, P., Springenberg, J. T., Riedmiller, M., & Brox, T. (2014). *Discriminative Unsupervised Feature Learning with Exemplar Convolutional Neural Networks*. NeurIPS. [arXiv:1406.6909](https://arxiv.org/abs/1406.6909)
[^chen2020]: Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). *A Simple Framework for Contrastive Learning of Visual Representations*. ICML. [arXiv:2002.05709](https://arxiv.org/abs/2002.05709)
[^he2020moco]: He, K., Fan, H., Wu, Y., Xie, S., & Girshick, R. (2020). *Momentum Contrast for Unsupervised Visual Representation Learning*. CVPR. [arXiv:1911.05722](https://arxiv.org/abs/1911.05722)
[^chen2020mocov2]: Chen, X., Fan, H., Girshick, R., & He, K. (2020). *Improved Baselines with Momentum Contrastive Learning*. [arXiv:2003.04297](https://arxiv.org/abs/2003.04297)
[^robinson2021]: Robinson, J., Chuang, C.-Y., Sra, S., & Jegelka, S. (2021). *Contrastive Learning with Hard Negative Samples*. ICLR. [arXiv:2010.04592](https://arxiv.org/abs/2010.04592)
[^chuang2020]: Chuang, C.-Y., Robinson, J., Lin, Y.-C., Torralba, A., & Jegelka, S. (2020). *Debiased Contrastive Learning*. NeurIPS. [arXiv:2007.00224](https://arxiv.org/abs/2007.00224)
[^wang2020]: Wang, T., & Isola, P. (2020). *Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere*. ICML. [arXiv:2005.10242](https://arxiv.org/abs/2005.10242)
[^arora2019]: Arora, S., Khandeparkar, H., Khodak, M., Plevrakis, O., & Saunshi, N. (2019). *A Theoretical Analysis of Contrastive Unsupervised Representation Learning*. ICML. [arXiv:1902.09229](https://arxiv.org/abs/1902.09229)
[^haochen2021]: HaoChen, J. Z., Wei, C., Gaidon, A., & Ma, T. (2021). *Provable Guarantees for Self-Supervised Deep Learning with Spectral Contrastive Loss*. NeurIPS. [arXiv:2106.04156](https://arxiv.org/abs/2106.04156)
[^jing2022]: Jing, L., Vincent, P., LeCun, Y., & Tian, Y. (2022). *Understanding Dimensional Collapse in Contrastive Self-supervised Learning*. ICLR. [arXiv:2110.09348](https://arxiv.org/abs/2110.09348)
[^grill2020]: Grill, J.-B., Strub, F., Altché, F., et al. (2020). *Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning*. NeurIPS. [arXiv:2006.07733](https://arxiv.org/abs/2006.07733)
[^chen2021simsiam]: Chen, X., & He, K. (2021). *Exploring Simple Siamese Representation Learning*. CVPR. [arXiv:2011.10566](https://arxiv.org/abs/2011.10566)
[^tian2021]: Tian, Y., Chen, X., & Ganguli, S. (2021). *Understanding Self-Supervised Learning Dynamics without Contrastive Pairs*. ICML. [arXiv:2102.06810](https://arxiv.org/abs/2102.06810)
[^zbontar2021]: Zbontar, J., Jing, L., Misra, I., LeCun, Y., & Deny, S. (2021). *Barlow Twins: Self-Supervised Learning via Redundancy Reduction*. ICML. [arXiv:2103.03230](https://arxiv.org/abs/2103.03230)
[^bardes2022]: Bardes, A., Ponce, J., & LeCun, Y. (2022). *VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning*. ICLR. [arXiv:2105.04906](https://arxiv.org/abs/2105.04906)
[^caron2021]: Caron, M., Touvron, H., Misra, I., Jégou, H., Mairal, J., Bojanowski, P., & Joulin, A. (2021). *Emerging Properties in Self-Supervised Vision Transformers*. ICCV. [arXiv:2104.14294](https://arxiv.org/abs/2104.14294)
[^caron2020]: Caron, M., Misra, I., Mairal, J., Goyal, P., Bojanowski, P., & Joulin, A. (2020). *Unsupervised Learning of Visual Features by Contrasting Cluster Assignments*. NeurIPS. [arXiv:2006.09882](https://arxiv.org/abs/2006.09882)
[^oquab2024]: Oquab, M., Darcet, T., Moutakanni, T., et al. (2024). *DINOv2: Learning Robust Visual Features without Supervision*. TMLR. [arXiv:2304.07193](https://arxiv.org/abs/2304.07193)
[^zhou2022]: Zhou, J., Wei, C., Wang, H., Shen, W., Xie, C., Yuille, A., & Kong, T. (2022). *iBOT: Image BERT Pre-Training with Online Tokenizer*. ICLR. [arXiv:2111.07832](https://arxiv.org/abs/2111.07832)
[^he2022mae]: He, K., Chen, X., Xie, S., Li, Y., Dollár, P., & Girshick, R. (2022). *Masked Autoencoders Are Scalable Vision Learners*. CVPR. [arXiv:2111.06377](https://arxiv.org/abs/2111.06377)
[^assran2023]: Assran, M., Duval, Q., Misra, I., Bojanowski, P., Vincent, P., Rabbat, M., LeCun, Y., & Ballas, N. (2023). *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture*. CVPR. [arXiv:2301.08243](https://arxiv.org/abs/2301.08243)
[^radford2021]: Radford, A., Kim, J. W., Hallacy, C., et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision*. ICML. [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)
[^zhai2023]: Zhai, X., Mustafa, B., Kolesnikov, A., & Beyer, L. (2023). *Sigmoid Loss for Language Image Pre-Training*. ICCV. [arXiv:2303.15343](https://arxiv.org/abs/2303.15343)
[^balestriero2023]: Balestriero, R., Ibrahim, M., Sobal, V., et al. (2023). *A Cookbook of Self-Supervised Learning*. [arXiv:2304.12210](https://arxiv.org/abs/2304.12210)
