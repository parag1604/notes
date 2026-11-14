---
title: Unsupervised Domain Adaptation
tags:
  - machine-learning
  - domain-adaptation
  - distribution-shift
  - transfer-learning
  - adversarial-training
date: 2026-11-14
draft: false
prev: "[[Normalizing Flows]]"
next: "[[Index (Machine Learning)]]"
---

> **Prerequisites:** [[Bayesian Decision Theory#Learning as risk minimization|learning as risk minimization]] (true vs empirical risk, the Hoeffding + union bound argument), [[Generative Adversarial Networks]] (the minimax game, the optimal discriminator, the non-saturating loss), [[Support Vector Machines#Kernels|kernels]], [[Linear Models#Logistic Regression|logistic regression]] and [[Linear Models#Weighted linear regression|weighted least squares]], and [[Neural Networks#Normalization layers|batch normalization]].
>
> **TL;DR:**
> - A model trained on one distribution (the **source**) is deployed on another (the **target**), where we have inputs but no labels. Its error can grow without any warning from the source validation set.
> - If only $p(\mathbf{x})$ changes (**covariate shift**), re-weighting the source loss by $p_T(\mathbf{x})/p_S(\mathbf{x})$ fixes the bias. A source-vs-target classifier estimates that ratio. The price is variance: few effective samples. If only $p(y)$ changes (**label shift**), EM or a confusion-matrix inversion (BBSE) estimates the new class priors.
> - The **Ben-David bound** $\epsilon_T(h)\le\epsilon_S(h)+\tfrac12d_{\mathcal{H}\Delta\mathcal{H}}+\epsilon^*$ says: be accurate on the source, and make the two domains hard to tell apart. A **domain classifier** measures the second term, which motivates learning **domain-invariant features**.
> - Features are aligned with a **discrepancy** (MMD, CORAL, optimal transport) or with an **adversary** (DANN's gradient reversal layer, ADDA, CDAN, MCD).
> - Or alignment is skipped altogether. Self-training, entropy minimization and the **contradistinguisher** (CUDA) classify the target directly, with a supervised source loss plus an unsupervised target loss; batch-norm methods (AdaBN, TENT, SHOT) adapt even without the source data.
> - Invariance is not free. When the class proportions differ, aligned features **must** make errors (Zhao et al.), and without assumptions linking the domains no method can be guaranteed to work.
>
> **Notation:** labelled source data $\mathcal{D}_S=\{(\mathbf{x}_i,y_i)\}_{i=1}^{N_S}\sim p_S$, unlabelled target data $\mathcal{D}_T=\{\mathbf{x}_j\}_{j=1}^{N_T}\sim p_T$ (the target labels exist but are never seen). A feature extractor (encoder) $g$ gives $\mathbf{z}=g(\mathbf{x})$, a classifier $f$ acts on $\mathbf{z}$, and the full predictor is $h=f\circ g$. A domain discriminator $d(\mathbf{z})$ outputs the probability that $\mathbf{z}$ came from the source. The risks are $\epsilon_S(h)=\mathbb{P}_{p_S}\big(h(\mathbf{x})\ne y\big)$ and $\epsilon_T(h)=\mathbb{P}_{p_T}\big(h(\mathbf{x})\ne y\big)$. Classes are $k=1..K$; $p_S(y)$ and $q_T(y)$ are the source and target class priors.
>
> **Code:** [Unsupervised-Domain-Adaptation](https://github.com/parag1604/Unsupervised-Domain-Adaptation): from-scratch tutorial on rotated moons and shifted digits: source-only baseline, importance weighting and label shift, MMD, CORAL, DANN, ADDA, self-training, AdaBN/TENT/SHOT-IM, CUDA, and where adaptation fails.

## Motivation

Every supervised lecture so far rested on one assumption, usually unstated: the test data come from the **same distribution** as the training data. That is what makes the training error a guide to the test error ([[Bayesian Decision Theory#Learning as risk minimization|Learning as risk minimization]]). In practice the assumption fails all the time:
- a medical classifier is trained on scans from one hospital's machines and deployed at another hospital;
- a perception model is trained on synthetic, rendered images, because labels are free there, and deployed on real camera images;
- a sentiment model is trained on product reviews and used on tweets;
- a model trained last year is used this year, and the world has moved on.

In each case labelled data are plentiful in one place (the **source domain**) and unlabelled data are plentiful in another (the **target domain**). Labelling the target is expensive or impossible. **Unsupervised domain adaptation** (UDA) asks: using labelled source data and *unlabelled* target data, can we learn a classifier that is accurate on the target?[^kouw2019]<sup>,</sup>[^wilson2020]

Figure 1 is the running example. The source is the familiar two-moons problem. The target is the same problem rotated by 35°: a sensor was mounted at a slightly different angle. A small neural network trained on the source is perfect there, and gets 66% of the target right. Nothing in the source data warns us. After adaptation with DANN, a method developed below, the same network gets 92%.

![[Machine Learning/images/unsupervised-domain-adaptation-moons.svg]]
*Figure 1: the toy problem. (a) Labelled source data and the decision boundary of a network trained on them. (b) The unlabelled target domain, the source rotated by 35° (true labels shown only for evaluation): the source-only boundary cuts both moons. (c) The boundary after domain-adversarial training (DANN). Accuracies are target means over 5 seeds.*

This lecture first asks when adaptation is possible at all, and what can be corrected when the shift has a simple form. Then it builds the theory that motivates aligning features across domains, the main families of methods that do so, and the limits of the whole idea.

## The setting and kinds of shift

A **domain** is a joint distribution $p(\mathbf{x},y)$. We have two, $p_S$ and $p_T$, and we may factorize each in two ways:

$$
p(\mathbf{x},y)=p(\mathbf{x})\,p(y\mid\mathbf{x})=p(y)\,p(\mathbf{x}\mid y).
$$

If both factors may change arbitrarily, the unlabelled target sample tells us about $p_T(\mathbf{x})$ and nothing at all about $p_T(y\mid\mathbf{x})$: any labelling of the target is consistent with the data. Adaptation is only possible under an assumption about **what stays fixed**. Three assumptions have names (Figure 2):[^morenotorres2012]

- **Covariate shift:** $p_S(\mathbf{x})\ne p_T(\mathbf{x})$, but $p_S(y\mid\mathbf{x})=p_T(y\mid\mathbf{x})$. The inputs move, the rule that labels them does not. Example: a disease's symptoms-to-diagnosis relation is the same in every hospital, but each hospital sees a different mix of patients. Natural when $\mathbf{x}$ causes $y$.
- **Label shift** (prior shift): $p_S(y)\ne q_T(y)$, but $p_S(\mathbf{x}\mid y)=p_T(\mathbf{x}\mid y)$. The class proportions change, what each class looks like does not. Example: the same disease looks the same on a scan, but is ten times more common during an epidemic. Natural when $y$ causes $\mathbf{x}$.
- **Concept shift:** $p(\mathbf{x})$ is fixed, but $p(y\mid\mathbf{x})$ changes. The meaning of the labels changes (what counts as spam this year). Without target labels, nothing can be done about it.

![[Machine Learning/images/unsupervised-domain-adaptation-shift-types.svg]]
*Figure 2: three kinds of dataset shift in one dimension. (a) Covariate shift: the input density moves, the posterior $p(y=1\mid x)$ is shared. (b) Label shift: each class-conditional density keeps its shape, but the target (dashed) has 85% of class 0 instead of 50%. (c) Concept shift: same inputs, different labelling rule.*

The rotated moons fit none of these exactly: rotation changes $p(\mathbf{x})$, and at a fixed point $\mathbf{x}$ the label may change too. That is the general case deep UDA methods address, with a different kind of assumption: that **some representation** $\mathbf{z}=g(\mathbf{x})$ exists in which the domains look alike and one classifier works for both. We come back to this after the two simple cases.

## Covariate shift and importance weighting

Under covariate shift the source posterior is the right one, so a model that could represent $p(y\mid\mathbf{x})$ exactly would need no correction. The trouble is that we fit a *restricted* model, and a restricted model trades errors between regions of input space in proportion to how much data each region has. The source puts its data in the wrong places.

The fix is a one-line identity, the same one behind [[Policy Gradients#Off-policy Policy Gradients|off-policy policy gradients]]. For any loss $\ell$, with $w(\mathbf{x})=p_T(\mathbf{x})/p_S(\mathbf{x})$,

$$
\begin{aligned}
\epsilon_T(h)=\mathbb{E}_{p_T(\mathbf{x},y)}\big[\ell(h(\mathbf{x}),y)\big]
&=\int p_T(\mathbf{x})\,p(y\mid\mathbf{x})\,\ell\,d\mathbf{x}\,dy\\
&=\int p_S(\mathbf{x})\,\frac{p_T(\mathbf{x})}{p_S(\mathbf{x})}\,p(y\mid\mathbf{x})\,\ell\,d\mathbf{x}\,dy\\
&=\mathbb{E}_{p_S(\mathbf{x},y)}\big[w(\mathbf{x})\,\ell(h(\mathbf{x}),y)\big].
\end{aligned}
$$

Covariate shift was used in the first and the last step: both joints factorize with the same $p(y\mid\mathbf{x})$, so only the input densities differ. So the **importance-weighted empirical risk**

$$
\hat\epsilon_w(h)=\frac{1}{N_S}\sum_{i=1}^{N_S}w(\mathbf{x}_i)\,\ell\big(h(\mathbf{x}_i),y_i\big)
\tag{1}
$$

is an unbiased estimate of the target risk, computed from source data alone. It needs $p_S(\mathbf{x})>0$ wherever $p_T(\mathbf{x})>0$: no weighting can recover a region the source never visits. For squared loss and a linear model, minimizing (1) is [[Linear Models#Weighted linear regression|weighted least squares]]. Shimodaira introduced this correction for maximum likelihood and showed that it matters precisely when the model is misspecified: if the model family contains the true $p(y\mid\mathbf{x})$, the unweighted fit is already consistent, and more efficient.[^shimodaira2000]

### Estimating the weights with a domain classifier

We do not know either density, and [[Non-parametric Models#Kernel density estimation|estimating each one]] and dividing is unreliable in more than a few dimensions. A better route estimates the ratio directly. Pool the two samples, label source points $s=1$ and target points $s=0$, and train a probabilistic classifier for $\mathbb{P}(s=1\mid\mathbf{x})$. By Bayes' rule,

$$
\frac{\mathbb{P}(s=0\mid\mathbf{x})}{\mathbb{P}(s=1\mid\mathbf{x})}=\frac{N_T\,p_T(\mathbf{x})}{N_S\,p_S(\mathbf{x})}
\quad\Longrightarrow\quad
w(\mathbf{x})=\frac{N_S}{N_T}\cdot\frac{1-\mathbb{P}(s=1\mid\mathbf{x})}{\mathbb{P}(s=1\mid\mathbf{x})}.
$$

With [[Linear Models#Logistic Regression|logistic regression]], the log-weight is just the negated logit, plus a constant. This trick, a classifier that tells the domains apart, is the most important idea in this lecture: it reappears as the proxy $\mathcal{A}$-distance and as the discriminator of adversarial adaptation.[^bickel2009]

> [!example]- Code: importance weights from a domain classifier (tested)
> ```python
> import numpy as np
> from sklearn.linear_model import LogisticRegression
>
> def importance_weights(Xs, Xt, features=lambda X: X, C=1.0):
>     """w(x) = p_T(x) / p_S(x) at the source points, from a source-vs-target classifier."""
>     F = np.vstack([features(Xs), features(Xt)])
>     s = np.r_[np.ones(len(Xs)), np.zeros(len(Xt))]          # 1 = source, 0 = target
>     clf = LogisticRegression(C=C, max_iter=5000).fit(F, s)
>     p_src = clf.predict_proba(features(Xs))[:, 1]
>     return (len(Xs) / len(Xt)) * (1 - p_src) / p_src
>
> def effective_sample_size(w):
>     return w.sum() ** 2 / (w ** 2).sum()
> ```
> With `features=lambda X: np.c_[X, X**2]` the classifier's log-odds is quadratic, which can represent the exact log-ratio of two Gaussians.

### The variance problem

Unbiased is not the same as accurate. Where the target is dense and the source sparse, a few source points get huge weights, and the estimate (1) rests on them alone. A standard diagnostic is the **effective sample size**[^owen2013]

$$
N_{\mathrm{eff}}=\frac{\big(\sum_iw_i\big)^2}{\sum_iw_i^2}\in[1,N_S],
$$

the number of equally weighted points that would give the same variance. It equals $N_S$ when all weights are equal and approaches 1 when one weight dominates. If $p_T$ has heavier tails than $p_S$, the weights can even have infinite variance. Two pragmatic remedies: **flatten** the weights to $w^\nu$ with $\nu\in[0,1]$ (Shimodaira's suggestion: $\nu=0$ ignores the shift, $\nu=1$ is unbiased), or clip them. Both trade bias for variance, and $\nu$ can be chosen by importance-weighted cross-validation.[^sugiyama2007]

Figure 3 shows the trade-off on Shimodaira's kind of example. The data follow a cubic, $y=-x+x^3$ plus Gaussian noise $\mathcal{N}(0,0.3^2)$, but we fit a straight line. With 50 source points from $\mathcal{N}(0.5,0.5^2)$, the effective sample size under the true weights for a target $\mathcal{N}(-0.3,0.3^2)$ is only 8.6 of 50. Unweighted least squares fits the line to the wrong region and has target MSE 0.61. Full weighting removes the bias (0.149), and a flattened $\nu=0.3$ does better still (0.125), against 0.108 for the best line on the target. Weights from a logistic domain classifier on $(x,x^2)$, trained on 50 unlabelled target points, are almost as good as the true ones.

![[Machine Learning/images/unsupervised-domain-adaptation-importance-weighting.svg]]
*Figure 3: importance weighting for covariate shift. (a) One draw of 50 source points (marker size proportional to the weight) and three least-squares lines. The unweighted fit follows the source region; the weighted fits follow the target region. (b) Target test MSE against the flattening exponent $\nu$, averaged over 500 draws, with the true weights and with weights from a domain classifier. Too little weighting leaves bias, full weighting adds variance; the best $\nu$ here is about 0.3–0.4.*

> [!warning] Importance weighting is not a free lunch
> Ben-David, Lu, Luu and Pál construct pairs of adaptation tasks whose source sample and unlabelled target sample are distributed identically, yet re-weighting helps greatly on one and hurts badly on the other.[^bendavid2010impossible] The covariate shift assumption is what makes weighting valid, and it cannot be checked from unlabelled target data.

## Label shift: estimating the new class priors

Now suppose $p(\mathbf{x}\mid y)$ is shared and only the priors change. Bayes' rule gives the target posterior from the source one, with no retraining:

$$
p_T(y\mid\mathbf{x})=\frac{p(\mathbf{x}\mid y)\,q_T(y)}{\sum_{y'}p(\mathbf{x}\mid y')\,q_T(y')}
\;\propto\;p_S(y\mid\mathbf{x})\,\frac{q_T(y)}{p_S(y)}.
\tag{2}
$$

This is the prior-shift adjustment of [[Bayesian Decision Theory]]: multiply each posterior by the ratio of priors and renormalize. All we need is $q_T(y)$, and the unlabelled target sample determines it.

**Why not count predictions?** Averaging the source classifier's predictions over target points is biased: a classifier trained on balanced data pulls its predictions back towards balance. Two estimators fix this.

**EM (Saerens, Latinne and Decaestecker).**[^saerens2002] Treat the target labels as latent variables of a mixture whose components $p(\mathbf{x}\mid y)$ are known (through the source posterior) and whose mixing weights $q_T(y)$ are unknown. [[Clustering and Expectation Maximization#Mixture models and the EM algorithm|EM]] alternates:
- **E-step:** compute the adjusted posteriors (2) of every target point under the current $q_T$;
- **M-step:** set $q_T(y)$ to the average adjusted posterior of class $y$.

Each step increases the target log-likelihood. EM needs **calibrated** source posteriors: if $p_S(y\mid\mathbf{x})$ is over- or under-confident, the mixture model is wrong and so is its maximum.

> [!example]- Code: EM for the target priors (tested)
> ```python
> import numpy as np
>
> def em_priors(P, prior_s, iters=1000, tol=1e-8):
>     """P: (N_T, K) calibrated source posteriors at the target points; prior_s: (K,) source priors."""
>     q = prior_s.copy()
>     for _ in range(iters):
>         W = P * (q / prior_s)                    # E-step: adjust the posteriors with (2) ...
>         W /= W.sum(axis=1, keepdims=True)        # ... and renormalize each row
>         q_new = W.mean(axis=0)                   # M-step: new target priors
>         if np.abs(q_new - q).max() < tol:
>             return q_new, W
>         q = q_new
>     return q, W
> ```

**BBSE (black-box shift estimation, Lipton, Wang and Smola).**[^lipton2018] Take *any* fixed classifier $\hat y=h(\mathbf{x})$, calibrated or not. Under label shift, the distribution of its predictions on the target is a mixture of its per-class behaviour, which is the same in both domains:

$$
q_T(\hat y=k)=\sum_{k'}\mathbb{P}(\hat y=k\mid y=k')\,q_T(y=k')=\sum_{k'}\underbrace{\mathbb{P}_S(\hat y=k,\,y=k')}_{C_{kk'}}\,\underbrace{\frac{q_T(y=k')}{p_S(y=k')}}_{w_{k'}}.
$$

Estimate the joint confusion matrix $\mathbf{C}$ on held-out labelled source data and the prediction frequencies $\hat{\mathbf{q}}$ on the target, and solve the linear system $\mathbf{C}\mathbf{w}=\hat{\mathbf{q}}$. Then $q_T(y)=w_y\,p_S(y)$. The estimator is consistent whenever $\mathbf{C}$ is invertible, i.e. the classifier is better than useless on every class, and Lipton et al. give finite-sample error bounds.

Figure 4 tests both on sklearn's digits, made harder by average-pooling each image to 4×4. The source is balanced; the target has priors proportional to $0.7^k$ for digit $k$ (31% zeros, 1% nines), with 300 target points. A logistic regression is trained on the source and calibrated by temperature scaling on a held-out source split.

![[Machine Learning/images/unsupervised-domain-adaptation-label-shift.svg]]
*Figure 4: label shift on 4×4-pooled digits. (a) Target priors estimated in one run. (b) Target accuracy of the source classifier before and after re-weighting its posteriors with (2), mean ± sd over 20 runs. EM on the raw (uncalibrated) posteriors makes things worse; after temperature scaling it nearly matches re-weighting with the true priors. BBSE needs no calibration for its estimate.*

The numbers, averaged over 20 runs: the source classifier gets 80.9% on the target. Re-weighting with EM's priors from uncalibrated posteriors drops this to 77.0%, while EM after calibration reaches 85.9% and BBSE 86.9%, close to the 88.0% obtained with the true priors. In $L_1$ distance to the true target priors: counting predictions is off by 0.25, uncalibrated EM by 0.32, calibrated EM by 0.13 and BBSE by 0.20.

## A bound for learning across domains

Covariate and label shift are both assumptions about *distributions*. Deep UDA methods instead make an assumption about *hypotheses*: there is one classifier that does well on both domains. Ben-David and colleagues turned this into a bound.[^bendavid2007]<sup>,</sup>[^bendavid2010]

Work with binary classification and a hypothesis class $\mathcal{H}$. For two hypotheses, write $\epsilon_S(h,h')=\mathbb{P}_{p_S}\big(h(\mathbf{x})\ne h'(\mathbf{x})\big)$ for how often they disagree on source inputs, and likewise $\epsilon_T(h,h')$. Disagreement needs no labels, so it can be estimated on both domains. Define:

- the **$\mathcal{H}\Delta\mathcal{H}$-divergence**, the largest change, between domains, in how often two hypotheses disagree:

$$
d_{\mathcal{H}\Delta\mathcal{H}}(p_S,p_T)=2\sup_{h,h'\in\mathcal{H}}\big|\epsilon_S(h,h')-\epsilon_T(h,h')\big|;
$$

- the **ideal joint error** $\epsilon^*=\min_{h\in\mathcal{H}}\big[\epsilon_S(h)+\epsilon_T(h)\big]$, attained by some $h^*$.

> [!note] Theorem (Ben-David et al., 2010)
> For every $h\in\mathcal{H}$,
> $$
> \epsilon_T(h)\le\epsilon_S(h)+\tfrac12\,d_{\mathcal{H}\Delta\mathcal{H}}(p_S,p_T)+\epsilon^*.
> $$
> With samples, $\epsilon_S$ and $d_{\mathcal{H}\Delta\mathcal{H}}$ are replaced by their empirical versions, at the cost of terms of order $\sqrt{(\mathrm{VC}(\mathcal{H})\log N)/N}$, as in the [[Bayesian Decision Theory#Learning as risk minimization|ordinary generalization bound]].

> [!proof]- Proof
> The 0-1 disagreement satisfies the triangle inequality, $\epsilon(h,h'')\le\epsilon(h,h')+\epsilon(h',h'')$, in either domain: if $h$ and $h''$ disagree at $\mathbf{x}$, then $h'$ disagrees with at least one of them there. Write $\epsilon_T(h)=\epsilon_T(h,f_T)$ with $f_T$ the target labelling.
> 1. Triangle inequality through $h^*$: $\epsilon_T(h)\le\epsilon_T(h^*)+\epsilon_T(h,h^*)$.
> 2. Move the disagreement to the source: $\epsilon_T(h,h^*)=\epsilon_S(h,h^*)+\big(\epsilon_T(h,h^*)-\epsilon_S(h,h^*)\big)\le\epsilon_S(h,h^*)+\tfrac12d_{\mathcal{H}\Delta\mathcal{H}}$, because $h$ and $h^*$ are both in $\mathcal{H}$, so their change in disagreement is at most the supremum.
> 3. Triangle inequality on the source: $\epsilon_S(h,h^*)\le\epsilon_S(h)+\epsilon_S(h^*)$.
> 4. Add up: $\epsilon_T(h)\le\epsilon_S(h)+\tfrac12d_{\mathcal{H}\Delta\mathcal{H}}+\underbrace{\epsilon_S(h^*)+\epsilon_T(h^*)}_{\epsilon^*}$.

The three terms say what adaptation needs:
1. **source error** $\epsilon_S(h)$: we can measure and minimize it;
2. **divergence** $d_{\mathcal{H}\Delta\mathcal{H}}$: measurable from unlabelled data, as shown next;
3. **adaptability** $\epsilon^*$: small only if one hypothesis is good on both domains. It involves target labels, so it **cannot be estimated**. It is an assumption.

### Measuring the divergence with a domain classifier

The supremum over pairs of hypotheses looks intractable, but it is a classification problem in disguise. Take the symmetric-difference class of functions $\mathbf{x}\mapsto\mathbb{1}[h(\mathbf{x})\ne h'(\mathbf{x})]$; the divergence asks how differently often such a function fires on the two domains. Ben-David et al. show that on balanced samples it equals $2(1-2\,\mathrm{err})$, where err is the smallest error of a **source-vs-target classifier** from that class.[^bendavid2007] If no classifier can tell the domains apart (err $=\tfrac12$), the divergence is 0; if one can do it perfectly, it is 2.

In practice we train some domain classifier (a linear SVM in Ganin et al.;[^ganin2016] an RBF SVM in our experiments), measure its held-out error $\widehat{\mathrm{err}}$, and report the **proxy $\mathcal{A}$-distance**

$$
\hat d_{\mathcal{A}}=2\,(1-2\,\widehat{\mathrm{err}}).
$$

On the raw rotated moons, $\hat d_{\mathcal{A}}=$ 0.86. The proxy depends on the classifier family: a linear classifier cannot separate the rotated moons from the originals at all (the clouds overlap), while an RBF SVM can.

### Why this motivates invariant features

Apply the bound not to the inputs but to features $\mathbf{z}=g(\mathbf{x})$, with $\mathcal{H}$ a class of classifiers $f$ on $\mathbf{z}$. Then the encoder controls the middle term: if $g$ maps both domains to the **same feature distribution**, $p_S(\mathbf{z})=p_T(\mathbf{z})$, no classifier on $\mathbf{z}$ can tell them apart, the divergence vanishes, and the target error is at most the source error plus $\epsilon^*$. This gives the recipe followed by most deep UDA methods:

$$
\min_{g,f}\;\underbrace{\frac{1}{N_S}\sum_{i=1}^{N_S}\ell\big(f(g(\mathbf{x}_i)),y_i\big)}_{\text{source risk}}\;+\;\alpha\cdot\underbrace{\mathrm{dist}\big(g_\#p_S,\;g_\#p_T\big)}_{\text{feature divergence}},
\tag{3}
$$

where $g_\#p$ denotes the distribution of $g(\mathbf{x})$ when $\mathbf{x}\sim p$. Methods differ in the choice of $\mathrm{dist}$: a fixed statistic (next section) or a learned one (the section after). Keep the third term in mind, though: $g$ also changes $\epsilon^*$, and nothing in (3) stops it from making $\epsilon^*$ large. We return to this in [[#The limits of invariance]].

## Discrepancy-based alignment

### Maximum mean discrepancy

Two distributions are equal iff every bounded continuous function has the same expectation under both. **Maximum mean discrepancy** relaxes "every function" to the unit ball of a reproducing-kernel Hilbert space $\mathcal{F}$ of a [[Support Vector Machines#Kernels|kernel]] $\kappa$:[^gretton2012]

$$
\mathrm{MMD}(P,Q)=\sup_{\lVert\phi\rVert_{\mathcal{F}}\le1}\Big(\mathbb{E}_{\mathbf{z}\sim P}[\phi(\mathbf{z})]-\mathbb{E}_{\mathbf{z}'\sim Q}[\phi(\mathbf{z}')]\Big)=\big\lVert\boldsymbol\mu_P-\boldsymbol\mu_Q\big\rVert_{\mathcal{F}},
$$

where $\boldsymbol\mu_P=\mathbb{E}_P[\kappa(\mathbf{z},\cdot)]$ is the **kernel mean embedding** of $P$, the mean of its points in feature space. The supremum is attained by the witness function $\phi\propto\boldsymbol\mu_P-\boldsymbol\mu_Q$, and expanding the squared norm with the kernel trick needs only kernel evaluations:

$$
\mathrm{MMD}^2(P,Q)=\mathbb{E}\big[\kappa(\mathbf{z},\mathbf{z}')\big]+\mathbb{E}\big[\kappa(\mathbf{v},\mathbf{v}')\big]-2\,\mathbb{E}\big[\kappa(\mathbf{z},\mathbf{v})\big],
$$

with $\mathbf{z},\mathbf{z}'\sim P$ and $\mathbf{v},\mathbf{v}'\sim Q$ independent. Replacing expectations by sample averages gives an estimate from two samples. For a **characteristic** kernel, such as the Gaussian kernel, $\mathrm{MMD}(P,Q)=0$ iff $P=Q$, so MMD is a genuine metric on distributions. Compare the [[Generative Adversarial Networks#The Wasserstein GAN|Wasserstein critic]]: both are integral probability metrics, with the critic restricted to Lipschitz functions there and to an RKHS ball here, where the supremum has a closed form.

The **deep adaptation network** (DAN) of Long et al. plugs a multi-kernel MMD between source and target features of several layers into (3).[^long2015] Using a family of bandwidths avoids picking one: a bandwidth much smaller than the typical distance sees every point as unique, one much larger sees everything as identical.

> [!example]- Code: an MMD loss with a mixture of Gaussian kernels (tested)
> ```python
> import torch
>
> def mmd2(Zs, Zt, sigmas=(0.5, 1.0, 2.0)):
>     """Biased (V-statistic) estimate of MMD^2 with a sum of Gaussian kernels."""
>     Z = torch.cat([Zs, Zt])
>     D2 = torch.cdist(Z, Z).pow(2)                       # all pairwise squared distances
>     K = sum(torch.exp(-D2 / (2 * s ** 2)) for s in sigmas)
>     n = len(Zs)
>     return K[:n, :n].mean() + K[n:, n:].mean() - 2 * K[:n, n:].mean()
> ```
> The biased estimate keeps the diagonal $\kappa(\mathbf{z},\mathbf{z})$ terms; it is never negative, which makes it a convenient loss. For a two-sample *test*, the unbiased estimate (diagonals removed) is used. In our experiments the bandwidths are multiples of the median pairwise distance in the batch.

### CORAL: matching covariances

MMD with a Gaussian kernel matches all moments. **CORAL** (correlation alignment) matches only the first two, which is often most of the shift caused by lighting, contrast or sensor gain.[^sun2016] The original version needs no training at all: whiten the source features with their own covariance and re-colour them with the target's, $\mathbf{X}_S\leftarrow\mathbf{X}_S\,\boldsymbol\Sigma_S^{-1/2}\boldsymbol\Sigma_T^{1/2}$ (with a small ridge added to $\boldsymbol\Sigma_S$), then train any classifier. **Deep CORAL** turns it into a loss on the features of a network:[^sun2016deep]

$$
\mathcal{L}_{\mathrm{CORAL}}=\frac{1}{4D^2}\big\lVert\boldsymbol\Sigma_S-\boldsymbol\Sigma_T\big\rVert_F^2,
$$

where $\boldsymbol\Sigma_S$ and $\boldsymbol\Sigma_T$ are the $D\times D$ [[Joint Distributions#Mean vectors and covariance matrices|covariance matrices]] of the source and target features in the batch.

> [!example]- Code: the Deep CORAL loss (tested)
> ```python
> import torch
>
> def coral(Zs, Zt):
>     D = Zs.shape[1]
>     Cs, Ct = torch.cov(Zs.T), torch.cov(Zt.T)            # (D, D) feature covariances
>     return ((Cs - Ct) ** 2).sum() / (4 * D * D)
> ```

### Optimal transport

A third way to compare distributions is the cost of moving one onto the other. **JDOT** and **DeepJDOT** solve an optimal transport problem between source and target samples whose cost combines feature distance and label disagreement, $\alpha\lVert g(\mathbf{x}_i)-g(\mathbf{x}_j)\rVert^2+\gamma\,\ell\big(y_i,f(g(\mathbf{x}_j))\big)$.[^courty2017]<sup>,</sup>[^damodaran2018] The transport plan pairs each target point with source points that look alike *and* carry a label the classifier agrees with; training alternates between computing the plan on a minibatch and updating the network. Unlike MMD and CORAL, which align marginals, this aligns the joint distribution of features and labels.

## Adversarial alignment

MMD fixes the test in advance. A GAN instead *learns* the test: a discriminator finds whatever distinguishes the two distributions, and the other player removes it. Domain-adversarial methods play the same game, with one twist: the "generator" is the feature extractor, and it is not trying to produce realistic samples but to make source and target features indistinguishable.

### DANN and the gradient reversal layer

The **domain-adversarial neural network** (DANN) of Ganin and Lempitsky has three parts:[^ganin2015]<sup>,</sup>[^ganin2016] the encoder $g$, the label classifier $f$, and a domain discriminator $d$ that sees the features of both domains and predicts which domain they came from. Let

$$
\mathcal{L}_y(g,f)=\frac{1}{N_S}\sum_{i=1}^{N_S}\ell_{\mathrm{CE}}\big(f(g(\mathbf{x}_i)),y_i\big),
$$

$$
\mathcal{L}_d(g,d)=-\frac{1}{N_S}\sum_{i=1}^{N_S}\log d\big(g(\mathbf{x}_i)\big)-\frac{1}{N_T}\sum_{j=1}^{N_T}\log\Big(1-d\big(g(\mathbf{x}_j)\big)\Big).
$$

$\mathcal{L}_d$ is the [[Linear Models#Logistic Regression|binary cross-entropy]] of the domain classifier. DANN seeks a saddle point of

$$
E(g,f,d)=\mathcal{L}_y(g,f)-\lambda\,\mathcal{L}_d(g,d):
\qquad
\min_{g,f}\;\max_{d}\;E.
\tag{4}
$$

The discriminator maximizes $E$, i.e. minimizes its own loss $\mathcal{L}_d$: it learns to tell the domains apart. The encoder minimizes $E$, which means fitting the labels while *maximizing* the discriminator's loss: making the domains hard to tell apart.

**What is the encoder minimizing?** For a fixed encoder, the best discriminator is the one from the [[Generative Adversarial Networks#The original GAN as a special case|GAN lecture]], $d^*(\mathbf{z})=p_S(\mathbf{z})/\big(p_S(\mathbf{z})+p_T(\mathbf{z})\big)$, here on the feature distributions. Substituting it, exactly as in that derivation, gives $\min_d\mathcal{L}_d=\log4-2\,\mathrm{JSD}\big(g_\#p_S\,\|\,g_\#p_T\big)$. So with an optimal discriminator the encoder minimizes

$$
\mathcal{L}_y+2\lambda\,\mathrm{JSD}\big(g_\#p_S\,\|\,g_\#p_T\big)+\text{const},
$$

an instance of (3) with the Jensen–Shannon divergence. And a discriminator's best error is precisely what the proxy $\mathcal{A}$-distance measures: DANN minimizes the middle term of the Ben-David bound with a learned domain classifier.

**The gradient reversal layer.** Gradient steps on (4) are

$$
\begin{aligned}
\theta_g&\leftarrow\theta_g-\eta\Big(\frac{\partial\mathcal{L}_y}{\partial\theta_g}-\lambda\frac{\partial\mathcal{L}_d}{\partial\theta_g}\Big),\qquad
\theta_f\leftarrow\theta_f-\eta\frac{\partial\mathcal{L}_y}{\partial\theta_f},\\
\theta_d&\leftarrow\theta_d-\eta\frac{\partial\mathcal{L}_d}{\partial\theta_d}.
\end{aligned}
$$

Everything is plain gradient descent on $\mathcal{L}_y+\mathcal{L}_d$, **except** the sign and scale of $\partial\mathcal{L}_d/\partial\theta_g$. Ganin and Lempitsky's trick is a layer inserted between $g$ and $d$ that is the **identity in the forward pass** and **multiplies the gradient by $-\lambda$ in the backward pass**. With it, one call to backpropagation on the ordinary sum $\mathcal{L}_y+\mathcal{L}_d$ produces exactly the updates above: $d$ receives $\partial\mathcal{L}_d/\partial\theta_d$ unchanged, while everything upstream of the layer receives $-\lambda\,\partial\mathcal{L}_d/\partial\theta_g$. No alternating optimizers, no second forward pass.

> [!example]- Code: the gradient reversal layer and one DANN step (tested)
> ```python
> import torch
> import torch.nn.functional as F
>
> class GradReverse(torch.autograd.Function):
>     """Identity forward; gradient multiplied by -lam backward."""
>     @staticmethod
>     def forward(ctx, x, lam):
>         ctx.lam = lam
>         return x.view_as(x)
>
>     @staticmethod
>     def backward(ctx, grad):
>         return -ctx.lam * grad, None
>
> def dann_step(g, f, d, opt, xs, ys, xt, lam):
>     """One SGD step on L_y + L_d.  The reversal makes g ascend L_d while d descends it."""
>     z = g(torch.cat([xs, xt]))
>     loss_y = F.cross_entropy(f(z[:len(xs)]), ys)
>     dom = torch.cat([torch.ones(len(xs)), torch.zeros(len(xt))])     # 1 = source
>     loss_d = F.binary_cross_entropy_with_logits(d(GradReverse.apply(z, lam)).squeeze(1), dom)
>     opt.zero_grad()
>     (loss_y + loss_d).backward()
>     opt.step()
>     return loss_y.item(), loss_d.item()
> ```
> Here `d` outputs a logit; $d(\mathbf{z})$ in the text is its sigmoid.

**The λ schedule.** Early in training the features are random and the discriminator's signal is noise. Ganin et al. therefore ramp the reversal strength from 0 to 1,

$$
\lambda_\rho=\frac{2}{1+\exp(-10\,\rho)}-1,
$$

where $\rho\in[0,1]$ is the fraction of training done, and anneal the learning rate as $\eta_\rho=\eta_0/(1+10\rho)^{0.75}$ with SGD and momentum 0.9.[^ganin2016] The ramp applies to the gradient reaching the encoder; the discriminator always trains at full strength. Figure 5 shows the schedules and the game on the moons. The discriminator gets an early edge (its batch accuracy peaks at 0.63 around iteration 60), but as $\lambda_\rho$ grows the encoder holds it near chance: over the second half of training its accuracy averages 0.56 and its loss 0.68, close to $\log2\approx0.69$, the loss of a discriminator that can only guess.

![[Machine Learning/images/unsupervised-domain-adaptation-dann-training.svg]]
*Figure 5: (a) DANN's schedules for the reversal strength $\lambda_\rho$ and the learning rate. (b) Domain-classifier accuracy and loss on the full batch during DANN training on the rotated moons (seed 0, moving average over 50 iterations). The discriminator never gets far above chance, and its loss stays near $\log2$; it never settles exactly, because the two players keep adapting to each other.*

> [!tip] Intuition
> The discriminator is a [[Linear Models#Logistic Regression|logistic regression]] on the features, asked "source or target?". The reversed gradient tells the encoder, for every feature, which direction would make that question easier, and the encoder moves the other way. Features that carry domain information, like the angle of the moons, get suppressed; features that carry label information are protected by $\mathcal{L}_y$.

### ADDA: untied encoders and the inverted-label loss

DANN shares one encoder between the domains and uses the saturating minimax loss. **Adversarial discriminative domain adaptation** (ADDA) changes both choices, and trains in stages:[^tzeng2017]
1. **Pre-train** a source encoder $g_S$ and classifier $f$ on the labelled source data, as usual.
2. **Adapt:** initialize a separate target encoder $g_T\leftarrow g_S$. Freeze $g_S$ and $f$. Alternate between training a discriminator to tell $g_S(\mathbf{x}_S)$ from $g_T(\mathbf{x}_T)$ and training $g_T$ with **inverted labels**, $\min_{g_T}-\mathbb{E}_{\mathbf{x}\sim p_T}\big[\log d\big(g_T(\mathbf{x})\big)\big]$: make target features look like source features.
3. **Test:** classify target inputs with $f\circ g_T$.

Step 2 is a GAN in which the target encoder is the generator and the frozen source features are the "real data", and the inverted-label loss is the [[Generative Adversarial Networks#The non-saturating generator loss|non-saturating generator loss]]: the same fixed point as the minimax loss, with stronger gradients when the discriminator is winning. Untied weights give the target encoder freedom to undo a large shift, but also freedom to drift: nothing anchors $g_T$ to the labels except its initialization, so ADDA inherits the [[Generative Adversarial Networks#Mode collapse and instability|instability of GAN training]].

### Conditional alignment: CDAN

Matching the *marginal* feature distributions is weaker than what we want. Two moons can be aligned with the upper source moon on top of the *lower* target moon: perfectly invariant, perfectly wrong. **Conditional domain adversarial networks** (CDAN) give the discriminator the classifier's prediction too, so it can detect mismatched class structure.[^long2018] Its input is the outer product $\mathbf{z}\otimes\hat{\mathbf{p}}$ of the features and the softmax prediction $\hat{\mathbf{p}}=\operatorname{softmax}(f(\mathbf{z}))$, a $D\times K$ matrix that places the features in a different block for each predicted class. For large $D\times K$ a random projection replaces the full product. CDAN+E additionally down-weights uncertain (high-entropy) examples in the discriminator's loss.

### Classifier discrepancy: MCD

**Maximum classifier discrepancy** (MCD) takes the $\mathcal{H}\Delta\mathcal{H}$-divergence literally.[^saito2018] Its supremum looks for two hypotheses that agree on the source but disagree on the target. MCD trains two classifiers $f_1,f_2$ on a shared encoder $g$ and measures their disagreement on a target point by $\lVert\hat{\mathbf{p}}_1-\hat{\mathbf{p}}_2\rVert_1/K$. Each iteration has three steps:
1. train $g,f_1,f_2$ on the labelled source data;
2. with $g$ fixed, train $f_1,f_2$ to stay accurate on the source while **maximizing** their disagreement on the target: they find target points outside the source support;
3. with $f_1,f_2$ fixed, train $g$ to **minimize** the disagreement: it moves those target features to where the classifiers agree.

The adversary is now the pair of classifiers, and it attacks exactly the term of the bound.

### Pixel-level adaptation

All the methods above align features. One can also adapt the **inputs**: translate source images so that they look like target images and train on the translations, which keep their source labels. [[Generative Adversarial Networks#Image-to-image translation: pix2pix and CycleGAN|Unpaired image translation]] with cycle-consistency (CycleGAN) makes this possible without paired examples: one generator maps source to target style, another maps back, and a reconstruction loss ties them together.[^zhu2017] **CyCADA** combines this pixel-level translation with feature-level adversarial alignment and a semantic-consistency loss, which asks a source-trained classifier to give the same label before and after translation, so the translation cannot turn a 3 into an 8.[^hoffman2018] The appeal is interpretability (one can look at the translated images); the cost is training a full image-to-image GAN.

## Self-training and adaptation without the source

A different idea needs no alignment at all: let the classifier teach itself on the target.

### Pseudo-labels and entropy minimization

**Pseudo-labelling** (self-training) labels each target point with the model's own prediction, keeps the confident ones (top probability above a threshold $\tau$), and retrains on source plus pseudo-labelled target, repeating for a few rounds. **Entropy minimization** is its soft version: add $\beta\,\mathbb{E}_{\mathbf{x}\sim p_T}\big[H\big(p(y\mid\mathbf{x})\big)\big]$ to the loss, where $H$ is the entropy of the prediction.[^grandvalet2004] Both push the decision boundary away from the target points, into **low-density regions**. If classes form separate clusters in the target (the cluster assumption), that is exactly where the boundary belongs.

> [!warning] Confirmation bias
> Self-training trusts its own mistakes. A target point that is confidently misclassified becomes a training example for the wrong class, which makes the model more confident about its neighbours, and so on. Whether self-training helps depends on whether the initial errors are scattered (they get outvoted) or systematic (they get reinforced).[^arazo2020] Under a large domain shift the errors are systematic: the source boundary cuts a whole region of the target, as in Figure 1(b).

### Classifying the target directly: the contradistinguisher (CUDA)

Every alignment method in the previous two sections solves an intermediate problem first: make the two feature distributions match, then hope that a source classifier transfers. Balgi and Dukkipati argue that this is backwards.[^balgi2019]<sup>,</sup>[^balgi2021] They invoke Vapnik's principle of statistical learning: solve the problem you actually care about as directly as possible, and do not solve a more general problem on the way. The problem we care about is classifying the target. Matching two whole distributions is harder than that, may be impossible when the shift is large, and destroys domain-specific information that could help. Their **contradistinguisher** is a single network $p_\theta(y\mid\mathbf{x})$, with no discriminator and no second classifier, trained with two losses at once: the usual cross-entropy on the labelled source, and an unsupervised **contradistinguish loss** on the target.

![[Machine Learning/images/unsupervised-domain-adaptation-cuda-vs-alignment.svg]]
*Figure 6: two ways to use the unlabelled target. (a) Alignment methods learn an encoder $g$ whose source and target feature distributions match (judged by a domain discriminator or a discrepancy such as MMD), train the classifier on source labels only, and rely on the alignment for the classifier to transfer. (b) CUDA trains one classifier $p_\theta(y\mid\mathbf{x})$ jointly on the source cross-entropy and the contradistinguish loss: each class column of the target predictions is normalized over the batch, as in (5), the best label under this joint distribution becomes the pseudo-label, and the network learns to give that label to its own point and not to the other points. No feature distributions are matched.*

**The target loss.** For target points $\mathbf{x}_1,\dots,\mathbf{x}_{N_T}$, CUDA builds a joint distribution over (target point, class) out of the classifier's predictions:

$$
q_\theta(\mathbf{x}_j,y)=\frac{p_\theta(y\mid\mathbf{x}_j)}{\sum_{m=1}^{N_T}p_\theta(y\mid\mathbf{x}_m)}\;q_T(y).
\tag{5}
$$

Arrange the predictions as an $N_T\times K$ matrix. The classifier's softmax normalizes each **row**: for this point, which class? The fraction in (5) normalizes each **column**: for this class, which of the target points? Multiplying by the target prior $q_T(y)$ makes the class marginal of $q_\theta$ equal to that prior. The target objective is the log-likelihood $\sum_j\log q_\theta(\mathbf{x}_j,y_j)$, to be maximized over $\theta$ and over the unknown labels $y_j$. CUDA alternates two steps on every minibatch:
1. **Pseudo-label selection:** with $\theta$ fixed, $\hat y_j=\arg\max_{y}\,q_\theta(\mathbf{x}_j,y)$. A class that the batch already assigns to many points has a large column sum, so it needs a higher probability to win.
2. **Maximization:** with the pseudo-labels fixed, ascend

$$
\sum_{j}\Big[\log p_\theta(\hat y_j\mid\mathbf{x}_j)-\log\sum_{m}p_\theta(\hat y_j\mid\mathbf{x}_m)\Big]
$$

in $\theta$ (the $\log q_T(\hat y_j)$ term is constant). The first term makes the network confident in its pseudo-label. The second pushes the probability of class $\hat y_j$ **down on all the other target points**: the network has to find what distinguishes $\mathbf{x}_j$ from the rest of the batch, hence the name. In practice the sums run over a minibatch (128 in the paper, large enough to contain every class), with log-sum-exp for stability.

**How it relates to the rest of this lecture.**
- The first term alone is pseudo-labelling without a confidence threshold, which the paper notes is a form of entropy regularization. On its own it is prone to the collapse of entropy minimization seen in our digits experiment.
- The bracket is a log-softmax **over samples** of the scores $\log p_\theta(\hat y_j\mid\mathbf{x}_m)$, the same normalization over a batch of candidates that [[Self-Supervised Representation Learning#NCE and InfoNCE|InfoNCE]] uses. Like SHOT's diversity term, it makes "everything is class 3" a bad solution: if one class dominates the column, the denominator grows.
- Nothing is aligned. The middle term of the Ben-David bound is never estimated or minimized; the target error is attacked directly, through the cluster structure of the target data. The source loss is what decides which target cluster gets which name. The assumption is the one behind entropy minimization: that the target classes form clusters which the source classifier roughly identifies to begin with.
- The prior $q_T(y)$ is an input. When it is unknown, the paper sets it to the source prior, which is exactly wrong under label shift; an estimate from [[#Label shift: estimating the new class priors|EM or BBSE]] can be used instead. This is not tested in the papers; in our own small re-implementation (the code linked above), on the moons with 90% of the target in one class, CUDA with the source prior had 13.2% target error, and 0.0% with an EM estimate of the target prior.

CUDA adds an optional regularizer against over-fitting its own pseudo-labels: "fake" inputs (Gaussian noise in the input space, or images from a small generator trained with an MMD loss to resemble target features) on which the classifier is trained to output **every** class at once, a multi-label binary cross-entropy that keeps it from being confident away from real data.

**What the papers report.** The ICDM paper evaluates digit, traffic-sign and CIFAR/STL transfer tasks and Amazon review sentiment; the TPAMI version adds Office-31, VisDA-2017 and multi-source settings. Some of the numbers, honestly put:
- On MNIST→SVHN, a hard case for every method, CUDA reports 71.3% against 35.7% for DANN and 54.5% for DIRT-T in its table. Self-ensembling reaches 97.0% there with heavy data augmentation and was excluded from the comparison for that reason.
- On Office-31 with a ResNet-50 encoder the mean is 89.8%, against 87.6% for CDAN and 90.6% for CAN in the paper's table; with ResNet-152 it is 91.1%.
- On multi-source digits (four domains to the fifth) the mean is 94.8%, against 87.7% for the best baseline listed.
- On VisDA-2017 the authors re-ran the competing methods and report precision and recall per class, pointing out that earlier papers had reported recall as accuracy. On the validation set, mean recall is 82.0% for CUDA and 86.1% for CAN; CUDA with CAN's data augmentation and target clustering reaches 87.3%.
- The headline "CUDA" rows are the best results over hyperparameter settings (the ablation rows show the individual configurations), so the [[#Shifted digits|model-selection caveat]] below applies to them as it does to most published UDA numbers. The TPAMI paper also reports that the contradistinguish loss increases the network's **shape bias**.

### Batch-norm statistics: AdaBN

A network with [[Neural Networks#Normalization layers|batch normalization]] stores the mean and variance of every unit, computed on source data, and uses them at test time. Under shift those statistics are wrong for the target. **AdaBN** simply recomputes them on target data and changes nothing else: no gradients, no labels, no source data.[^li2016adabn] It removes the part of the shift that is a per-channel shift and scale of the activations, and nothing more.

### TENT: test-time entropy minimization

**TENT** goes one step further.[^wang2021] At test time, each batch of target inputs is normalized with its *own* statistics, and the only parameters updated are the batch-norm affine parameters (a scale and a shift per channel), by gradient descent on the mean prediction entropy of the batch. These are few parameters, so a few batches suffice and the risk of destroying the source solution is limited. Like AdaBN, TENT needs neither labels nor the source data: only the trained model and the stream of test inputs.

> [!example]- Code: TENT (tested)
> ```python
> import torch
> import torch.nn as nn
>
> def configure_tent(net, lr=1e-3):
>     """Batch statistics in every BN layer; only the BN scales and shifts are trainable."""
>     net.train()
>     params = []
>     for m in net.modules():
>         if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
>             m.track_running_stats = False
>             m.running_mean = m.running_var = None
>             params += [m.weight, m.bias]
>     for p in net.parameters():
>         p.requires_grad_(False)
>     for p in params:
>         p.requires_grad_(True)
>     return torch.optim.Adam(params, lr=lr)
>
> def tent_step(net, opt, x):
>     logits = net(x)
>     loss = -(logits.softmax(1) * logits.log_softmax(1)).sum(1).mean()   # mean entropy
>     opt.zero_grad()
>     loss.backward()
>     opt.step()
>     return logits.argmax(1)          # predictions for this batch
> ```

### Source-free adaptation: SHOT

Sometimes the source data cannot be shared (privacy, licensing, size), only the trained model. **SHOT** (source hypothesis transfer) keeps the source classifier $f$ frozen, as the "hypothesis", and adapts only the encoder $g$ on target data, so that target features fit that hypothesis.[^liang2020] Its main loss is **information maximization**:

$$
\mathcal{L}_{\mathrm{IM}}=\underbrace{\mathbb{E}_{\mathbf{x}\sim p_T}\Big[H\big(\hat{\mathbf{p}}(\mathbf{x})\big)\Big]}_{\text{each prediction confident}}-\underbrace{H\Big(\mathbb{E}_{\mathbf{x}\sim p_T}\big[\hat{\mathbf{p}}(\mathbf{x})\big]\Big)}_{\text{all classes used}}.
$$

The first term is entropy minimization. The second keeps the average prediction spread over the classes, which blocks the trivial solution of predicting one class everywhere, the same kind of [[Self-Supervised Representation Learning#What collapse is|collapse]] that self-supervised methods must prevent. The full method adds pseudo-labels from class centroids in feature space, which are more robust than the classifier's raw predictions; our experiment below uses information maximization only.

### Domain generalization

The last step removes the target data altogether: train on several source domains and hope to do well on an unseen one. This is **domain generalization**. Many algorithms (invariant risk minimization, distributionally robust training, domain-adversarial training across the sources) have been proposed; in the DomainBed study, with a fixed model-selection protocol, plain ERM with careful training matched or beat all of them on average.[^gulrajani2021] The lesson carries over to UDA: compare against a well-tuned source-only baseline, and say how hyperparameters were chosen.

## Experiments

### Rotated moons

On the moons rotated by 35° (Figure 1), with the same small network for every method (two hidden layers of 64 ReLU units, an 8-dimensional feature layer, Adam with learning rate $5\times10^{-3}$, 1000 full-batch steps, weight 1 on the MMD, CORAL and domain losses and 0.1 on the entropy), target accuracies over 5 seeds are:

| method | target accuracy | proxy $\hat d_{\mathcal{A}}$ of the features |
|---|---|---|
| source only | 66% ± 4 | 0.94 ± 0.13 |
| MMD | 81% ± 13 | 0.00 ± 0.00 |
| Deep CORAL | 70% ± 7 | 0.76 ± 0.22 |
| DANN | 92% ± 4 | 0.29 ± 0.31 |
| ADDA | 69% ± 12 | 0.39 ± 0.27 |
| entropy minimization | 82% ± 6 | 0.58 ± 0.22 |
| self-training ($\tau=0.9$) | 68% ± 5 | 0.89 ± 0.05 |

Figure 7 looks inside: the source and target features, projected to two dimensions.

![[Machine Learning/images/unsupervised-domain-adaptation-alignment.svg]]
*Figure 7: features of source (filled) and target (hollow triangles) points, coloured by the true class, projected onto their first two principal components (seed 0). Titles give mean target accuracy and proxy $\mathcal{A}$-distance over 5 seeds. Source-only features keep the domains apart. CORAL matches covariances but not the shapes. MMD and DANN pull the domains together; where they succeed, each target class lands on the source class with the same label.*

Four things stand out.
- **DANN** is the best and the most stable method here, and it lowers the proxy $\mathcal{A}$-distance of the features from 0.94 to 0.29.
- **MMD** makes the domains indistinguishable to our test: $\hat d_{\mathcal{A}}=0.00$ on every seed. Yet its target accuracy ranges from 65% to 99% across the five seeds. **Invariance is not correctness.** In the bad seeds the two feature clouds coincide, but part of each target moon sits on top of the *other* source moon. That is the $\epsilon^*$ term of the bound, which alignment does not control and the unlabelled data cannot reveal.
- **CORAL** matches only the covariances and helps little. **Entropy minimization** helps, with no alignment at all: it pushes the boundary away from dense regions of target points.
- **Self-training** with a 0.9 confidence threshold barely moves, and the reason is confirmation bias in its plainest form. The source-only network is confident (top probability at least 0.9) on 87–92% of the target points, and 80–90% of its target errors are among those confident points. The pseudo-labels therefore contain almost all the mistakes, and retraining on them teaches the network its own errors. **ADDA** is unstable: its untied target encoder starts from the source-only one, and on two of the five seeds ends up worse than where it started.

How far does adaptation reach? Figure 8 repeats the experiment for rotations from 0° to 90°.

![[Machine Learning/images/unsupervised-domain-adaptation-rotation-sweep.svg]]
*Figure 8: target accuracy against the rotation of the target moons (mean of 3 seeds per point). Below 50% (dotted) a classifier is worse than guessing: a large rotation puts much of each target moon where the other source moon was.*

For small rotations every method works, including doing nothing. At 30° MMD, DANN, ADDA and entropy minimization add 11–18 points over the source-only 74% (MMD and DANN reach 91%), while CORAL adds 2. At 45° only DANN holds up (87%, against 47% without adaptation). Beyond 60° nothing works reliably, and at 90° every method is below 50%. Self-training follows the source-only curve at every angle, and entropy minimization helps only while the source model is mostly right to begin with. For a large rotation, marginal alignment has two equally good solutions, matching the moons the right way round or the wrong way round, and nothing in the unlabelled data says which one is correct.

### Shifted digits

A harder test uses sklearn's bundled 8×8 digits (no download). Half the images, with labels, form the source. The other half is the target: each image is rotated by 30° and corrupted with Gaussian pixel noise (sd 0.15), and its labels are hidden. The target half is split into 539 unlabelled images used for adaptation and 360 held-out test images.

![[Machine Learning/images/unsupervised-domain-adaptation-digits-examples.svg]]
*Figure 9: one source image (top) and one target image (bottom) of each digit. The target is rotated by 30° and noisy.*

Every method uses the same small CNN with batch normalization (two 3×3 convolution layers, a 64-dimensional feature layer), trained with Adam (learning rate $10^{-3}$, 1500 steps, batches of 64 source and 64 target images; DANN is the exception discussed below). AdaBN, TENT (5 passes over the adaptation set) and SHOT-IM (300 steps) start from the source-only network. We fixed the hyperparameters in advance and did not tune them on target labels, with one exception noted below.

![[Machine Learning/images/unsupervised-domain-adaptation-digits-results.svg]]
*Figure 10: (a) target test accuracy on the shifted digits, mean ± sd over 5 seeds. (b) Target accuracy against the proxy $\mathcal{A}$-distance between source and target features (RBF-SVM domain classifier). MMD and ADDA reduce the divergence the most and are the most accurate; DANN improves accuracy while its features stay almost perfectly separable to the SVM.*

The source-only network is 100% accurate on its own domain and 47% ± 5 on the target. Means ± sd over 5 seeds:
- **Normalization only.** AdaBN (53% ± 4) and TENT (53% ± 3) each add about 7 points. Re-estimating batch statistics corrects shifts in the mean and scale of the activations (the clipped noise brightens the background), but not the rotation.
- **Source-free.** SHOT-IM, which never sees a source image during adaptation, reaches 73% ± 6.
- **Alignment.** MMD is the best method here (81% ± 4), followed by ADDA (77% ± 4), CORAL (67% ± 3) and DANN (64% ± 3).
- **Self-training** (43% ± 2) and **entropy minimization** (40% ± 10) are *worse* than doing nothing. Entropy minimization shows the collapse that SHOT's diversity term prevents. Measure it by the entropy of the histogram of predicted test labels, which is $\log10=2.30$ nats if every digit is predicted equally often: source-only gives 2.00, entropy minimization 1.35 (as if only about four classes were ever predicted), SHOT-IM 2.28.

Panel (b) relates accuracy to the proxy $\mathcal{A}$-distance of the features. MMD and ADDA reduce it the most and are the most accurate. DANN gains 18 points of accuracy, yet its features stay almost perfectly separable by an RBF SVM ($\hat d_{\mathcal{A}}=$ 1.98): fooling its own discriminator, a small MLP, is not the same as making the domains indistinguishable to every classifier. The ranking is specific to this small problem and these fixed hyperparameters. On standard benchmarks the published ordering differs, and the gaps between well-tuned methods are often smaller than the gaps caused by tuning.

> [!warning] Model selection without target labels
> Every number above was obtained with hyperparameters that we could not validate honestly: the target labels used to *measure* accuracy are exactly what UDA assumes we do not have. Selecting $\lambda$, the learning rate or the stopping time by target accuracy, as many papers implicitly do, overstates what a practitioner would get. Options are a small labelled target validation set (then the problem is no longer unsupervised), importance-weighted validation on the source,[^sugiyama2007] or conservative defaults. In our DANN runs, Adam was unstable across seeds (47% ± 6) while the original paper's recipe (SGD with momentum 0.9, learning rate 0.01 annealed as above) was not (64% ± 3). We report the latter, and we chose it by looking at target accuracy.

## The limits of invariance

The bound has a third term, $\epsilon^*$, and aligning features can make it large. Here is a clean case. Keep the moons unrotated, so there is **no covariate change at all**, but let the target contain mostly class 1: only label shift. A source-only classifier is then essentially perfect on the target, because $p(\mathbf{x}\mid y)$ is shared. Now force the features to be invariant. The source feature distribution is a 50/50 mixture of the two classes, the target a 10/90 mixture; to make them equal, the encoder must send some class-1 target points to where source class-0 points live, and the classifier labels them 0.

Zhao, des Combes, Zhang and Gordon made this precise.[^zhao2019] Let $d_{\mathrm{JS}}=\sqrt{\mathrm{JSD}}$ be the Jensen–Shannon *distance* (with logarithms in base 2, so it lies in $[0,1]$). For any encoder $g$ and classifier $f$, if $d_{\mathrm{JS}}\big(p_S(y),q_T(y)\big)\ge d_{\mathrm{JS}}\big(g_\#p_S,g_\#p_T\big)$, then

$$
\epsilon_S(f\circ g)+\epsilon_T(f\circ g)\ge\frac12\Big(d_{\mathrm{JS}}\big(p_S(y),q_T(y)\big)-d_{\mathrm{JS}}\big(g_\#p_S,\,g_\#p_T\big)\Big)^2.
$$

The more invariant the features (the smaller the second distance), the larger the lower bound on the **joint** error. With perfectly aligned features, the joint error is at least $\tfrac12\mathrm{JSD}\big(p_S(y),q_T(y)\big)$. Learning separate encoders, as ADDA does, does not escape it: the same bound holds for $\epsilon_S(f\circ g_S)+\epsilon_T(f\circ g_T)$.

Figure 11 shows it happening.

![[Machine Learning/images/unsupervised-domain-adaptation-invariance-hurts.svg]]
*Figure 11: two moons with no input shift, only label shift: the target contains a fraction $q_T(y=1)$ of class 1 (the source is balanced). (a) Target error: source-only stays near zero; MMD and DANN, which align the feature marginals, make more errors the more the class proportions differ. (b) The joint error and Zhao et al.'s lower bound for perfectly aligned features. Mean ± sd over 3 seeds.*

At $q_T(y=1)=0.9$ the source-only model makes 0.1% target errors. With alignment, MMD makes 6.8% and DANN 2.3% (means of 3 seeds); the source errors stay near zero. The lower bound for perfectly aligned features is $\tfrac12\mathrm{JSD}=7.3\%$ there. MMD, which aligns the most completely, ends up close to it; DANN aligns less completely and pays less. On this task the adaptation methods turn a solved problem into an unsolved one. They would have done the same on a problem with real covariate shift *and* a change in class proportions, and there the damage would be mixed in with the benefit.

**Impossibility without assumptions.** Ben-David, Lu, Luu and Pál go further.[^bendavid2010impossible] They study three assumptions: (i) the unlabelled distributions are similar, (ii) some hypothesis has low error on both domains ($\epsilon^*$ small), and (iii) covariate shift. Without (i) or (ii), the remaining assumptions do not suffice: there are tasks on which **every** algorithm that sees no target labels fails. Covariate shift in particular is a weak assumption on its own. Every UDA method therefore bets on something it cannot verify from its data: that the shift is of a known kind, or that an invariant representation preserves the labels. The bet is often good, and choosing the method means choosing the bet:
- the class proportions may differ? Use label-shift correction, or methods that align class-conditionally and re-weight (CDAN, class-weighted alignment), not plain marginal alignment;
- the shift is mostly low-level (lighting, sensor, style)? Normalization statistics (AdaBN, TENT) and moment matching are cheap and safe;
- the shift is geometric or large? Adversarial and discrepancy methods can help, but check with a labelled target sample before trusting them.

## Key takeaways

- Domain shift breaks the link between training error and test error. Adaptation from unlabelled target data is possible only under an assumption about what stays the same across domains.
- **Covariate shift:** re-weight the source loss by $p_T(\mathbf{x})/p_S(\mathbf{x})$, estimated with a source-vs-target classifier. Watch the effective sample size, and flatten the weights when it is small.
- **Label shift:** estimate the target priors (EM on calibrated posteriors, or BBSE with any classifier) and re-weight the posteriors by $q_T(y)/p_S(y)$.
- **Ben-David:** $\epsilon_T\le\epsilon_S+\tfrac12d_{\mathcal{H}\Delta\mathcal{H}}+\epsilon^*$. A domain classifier measures the divergence (proxy $\mathcal{A}$-distance), so we learn features that fool one: with a fixed statistic (MMD, CORAL, OT) or a learned adversary (DANN's gradient reversal, ADDA's inverted labels, CDAN, MCD).
- **Self-training** and entropy minimization push boundaries into low-density regions but can amplify their own errors. The **contradistinguisher** solves the target classification directly instead of aligning first (Vapnik's principle), adding a contrast over target samples that blocks collapse. **AdaBN, TENT and SHOT** adapt with the trained model and target inputs alone.
- **Invariance has a price:** aligned features force errors when class proportions differ, $\epsilon^*$ cannot be measured, and hyperparameters cannot be validated without target labels.

## Where this shows up next

- [[Generative Adversarial Networks]]: DANN and ADDA are GANs on features; the optimal discriminator, the JS divergence and the non-saturating loss all come from there.
- [[Self-Supervised Representation Learning]]: representations learned without labels, on many domains at once, are often the best starting point for adaptation; the collapse problem there is the "predict one class everywhere" problem of entropy minimization here.
- [[Imitation Learning]]: behaviour cloning suffers a covariate shift of its own making, as the learner's mistakes take it to states the expert never visited. DAgger fixes it by collecting labels on the learner's distribution, the option UDA does not have.
- [[Policy Gradients]]: importance weighting reappears for off-policy learning, with the same variance problem.
- [[Energy-Based Models]]: the density ratio estimated by a classifier here is the idea behind noise-contrastive estimation there.

<!-- References: footnote definitions below render as the numbered "References" list at the bottom of the page -->

[^kouw2019]: Kouw, W. M., & Loog, M. (2018). *An Introduction to Domain Adaptation and Transfer Learning*. [arXiv:1812.11806](https://arxiv.org/abs/1812.11806)
[^wilson2020]: Wilson, G., & Cook, D. J. (2020). *A Survey of Unsupervised Deep Domain Adaptation*. ACM Transactions on Intelligent Systems and Technology, 11(5). [doi:10.1145/3400066](https://doi.org/10.1145/3400066)
[^morenotorres2012]: Moreno-Torres, J. G., Raeder, T., Alaiz-Rodríguez, R., Chawla, N. V., & Herrera, F. (2012). *A Unifying View on Dataset Shift in Classification*. Pattern Recognition, 45(1), 521–530. [doi:10.1016/j.patcog.2011.06.019](https://doi.org/10.1016/j.patcog.2011.06.019)
[^shimodaira2000]: Shimodaira, H. (2000). *Improving Predictive Inference under Covariate Shift by Weighting the Log-Likelihood Function*. Journal of Statistical Planning and Inference, 90(2), 227–244. [doi:10.1016/S0378-3758(00)00115-4](https://doi.org/10.1016/S0378-3758(00)00115-4)
[^bickel2009]: Bickel, S., Brückner, M., & Scheffer, T. (2009). *Discriminative Learning Under Covariate Shift*. JMLR, 10, 2137–2155. [jmlr.org](https://jmlr.org/papers/v10/bickel09a.html)
[^owen2013]: Owen, A. B. (2013). *Monte Carlo Theory, Methods and Examples*, Chapter 9: Importance sampling. [artowen.su.domains/mc](https://artowen.su.domains/mc/)
[^sugiyama2007]: Sugiyama, M., Krauledat, M., & Müller, K.-R. (2007). *Covariate Shift Adaptation by Importance Weighted Cross Validation*. JMLR, 8, 985–1005. [jmlr.org](https://www.jmlr.org/papers/v8/sugiyama07a.html)
[^bendavid2010impossible]: Ben-David, S., Lu, T., Luu, T., & Pál, D. (2010). *Impossibility Theorems for Domain Adaptation*. AISTATS, PMLR 9, 129–136. [proceedings.mlr.press](https://proceedings.mlr.press/v9/david10a.html)
[^saerens2002]: Saerens, M., Latinne, P., & Decaestecker, C. (2002). *Adjusting the Outputs of a Classifier to New a Priori Probabilities: A Simple Procedure*. Neural Computation, 14(1), 21–41. [doi:10.1162/089976602753284446](https://doi.org/10.1162/089976602753284446)
[^lipton2018]: Lipton, Z. C., Wang, Y.-X., & Smola, A. (2018). *Detecting and Correcting for Label Shift with Black Box Predictors*. ICML. [arXiv:1802.03916](https://arxiv.org/abs/1802.03916)
[^bendavid2007]: Ben-David, S., Blitzer, J., Crammer, K., & Pereira, F. (2007). *Analysis of Representations for Domain Adaptation*. NeurIPS 19. [papers.nips.cc](https://papers.nips.cc/paper_files/paper/2006/hash/b1b0432ceafb0ce714426e9114852ac7-Abstract.html)
[^bendavid2010]: Ben-David, S., Blitzer, J., Crammer, K., Kulesza, A., Pereira, F., & Vaughan, J. W. (2010). *A Theory of Learning from Different Domains*. Machine Learning, 79(1–2), 151–175. [doi:10.1007/s10994-009-5152-4](https://doi.org/10.1007/s10994-009-5152-4)
[^ganin2016]: Ganin, Y., Ustinova, E., Ajakan, H., Germain, P., Larochelle, H., Laviolette, F., Marchand, M., & Lempitsky, V. (2016). *Domain-Adversarial Training of Neural Networks*. JMLR, 17(59), 1–35. [arXiv:1505.07818](https://arxiv.org/abs/1505.07818)
[^gretton2012]: Gretton, A., Borgwardt, K. M., Rasch, M. J., Schölkopf, B., & Smola, A. (2012). *A Kernel Two-Sample Test*. JMLR, 13, 723–773. [jmlr.org](https://jmlr.org/papers/v13/gretton12a.html)
[^long2015]: Long, M., Cao, Y., Wang, J., & Jordan, M. I. (2015). *Learning Transferable Features with Deep Adaptation Networks*. ICML. [arXiv:1502.02791](https://arxiv.org/abs/1502.02791)
[^sun2016]: Sun, B., Feng, J., & Saenko, K. (2016). *Return of Frustratingly Easy Domain Adaptation*. AAAI. [arXiv:1511.05547](https://arxiv.org/abs/1511.05547)
[^sun2016deep]: Sun, B., & Saenko, K. (2016). *Deep CORAL: Correlation Alignment for Deep Domain Adaptation*. ECCV Workshops. [arXiv:1607.01719](https://arxiv.org/abs/1607.01719)
[^courty2017]: Courty, N., Flamary, R., Habrard, A., & Rakotomamonjy, A. (2017). *Joint Distribution Optimal Transportation for Domain Adaptation*. NeurIPS. [arXiv:1705.08848](https://arxiv.org/abs/1705.08848)
[^damodaran2018]: Damodaran, B. B., Kellenberger, B., Flamary, R., Tuia, D., & Courty, N. (2018). *DeepJDOT: Deep Joint Distribution Optimal Transport for Unsupervised Domain Adaptation*. ECCV. [arXiv:1803.10081](https://arxiv.org/abs/1803.10081)
[^ganin2015]: Ganin, Y., & Lempitsky, V. (2015). *Unsupervised Domain Adaptation by Backpropagation*. ICML. [arXiv:1409.7495](https://arxiv.org/abs/1409.7495)
[^tzeng2017]: Tzeng, E., Hoffman, J., Saenko, K., & Darrell, T. (2017). *Adversarial Discriminative Domain Adaptation*. CVPR. [arXiv:1702.05464](https://arxiv.org/abs/1702.05464)
[^long2018]: Long, M., Cao, Z., Wang, J., & Jordan, M. I. (2018). *Conditional Adversarial Domain Adaptation*. NeurIPS. [arXiv:1705.10667](https://arxiv.org/abs/1705.10667)
[^saito2018]: Saito, K., Watanabe, K., Ushiku, Y., & Harada, T. (2018). *Maximum Classifier Discrepancy for Unsupervised Domain Adaptation*. CVPR. [arXiv:1712.02560](https://arxiv.org/abs/1712.02560)
[^zhu2017]: Zhu, J.-Y., Park, T., Isola, P., & Efros, A. A. (2017). *Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks*. ICCV. [arXiv:1703.10593](https://arxiv.org/abs/1703.10593)
[^hoffman2018]: Hoffman, J., Tzeng, E., Park, T., Zhu, J.-Y., Isola, P., Saenko, K., Efros, A. A., & Darrell, T. (2018). *CyCADA: Cycle-Consistent Adversarial Domain Adaptation*. ICML. [arXiv:1711.03213](https://arxiv.org/abs/1711.03213)
[^grandvalet2004]: Grandvalet, Y., & Bengio, Y. (2004). *Semi-supervised Learning by Entropy Minimization*. NeurIPS 17. [papers.nips.cc](https://papers.nips.cc/paper_files/paper/2004/hash/96f2b50b5d3613adf9c27049b2a888c7-Abstract.html)
[^arazo2020]: Arazo, E., Ortego, D., Albert, P., O'Connor, N. E., & McGuinness, K. (2020). *Pseudo-Labeling and Confirmation Bias in Deep Semi-Supervised Learning*. IJCNN. [arXiv:1908.02983](https://arxiv.org/abs/1908.02983)
[^li2016adabn]: Li, Y., Wang, N., Shi, J., Liu, J., & Hou, X. (2016). *Revisiting Batch Normalization For Practical Domain Adaptation*. [arXiv:1603.04779](https://arxiv.org/abs/1603.04779)
[^wang2021]: Wang, D., Shelhamer, E., Liu, S., Olshausen, B., & Darrell, T. (2021). *Tent: Fully Test-time Adaptation by Entropy Minimization*. ICLR. [arXiv:2006.10726](https://arxiv.org/abs/2006.10726)
[^liang2020]: Liang, J., Hu, D., & Feng, J. (2020). *Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation*. ICML. [arXiv:2002.08546](https://arxiv.org/abs/2002.08546)
[^gulrajani2021]: Gulrajani, I., & Lopez-Paz, D. (2021). *In Search of Lost Domain Generalization*. ICLR. [arXiv:2007.01434](https://arxiv.org/abs/2007.01434)
[^zhao2019]: Zhao, H., Tachet des Combes, R., Zhang, K., & Gordon, G. J. (2019). *On Learning Invariant Representations for Domain Adaptation*. ICML. [arXiv:1901.09453](https://arxiv.org/abs/1901.09453)
[^balgi2019]: Balgi, S., & Dukkipati, A. (2019). *CUDA: Contradistinguisher for Unsupervised Domain Adaptation*. IEEE ICDM, 21–30. [doi:10.1109/ICDM.2019.00012](https://doi.org/10.1109/ICDM.2019.00012); [arXiv:1909.03442](https://arxiv.org/abs/1909.03442)
[^balgi2021]: Balgi, S., & Dukkipati, A. (2021). *Contradistinguisher: A Vapnik's Imperative to Unsupervised Domain Adaptation*. IEEE Transactions on Pattern Analysis and Machine Intelligence. [doi:10.1109/TPAMI.2021.3071225](https://doi.org/10.1109/TPAMI.2021.3071225); [arXiv:2005.14007](https://arxiv.org/abs/2005.14007)
