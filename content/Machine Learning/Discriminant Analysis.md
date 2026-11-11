---
title: Discriminant Analysis
tags:
  - fisher-discriminant-analysis
  - linear-discriminant-analysis
  - quadratic-discriminant-analysis
  - generative-models
---

> **Prerequisites:** [[Linear Models]] (logistic regression, maximum likelihood), Bayes' rule, and the multivariate Gaussian.
>
> **TL;DR:**
> - **Fisher's discriminant** looks for the projection direction that separates two classes best: large spread *between* the projected class means, small spread *within* each class. The answer is $\mathbf{w}^*\propto\mathbf{S}_W^{-1}(\boldsymbol{\mu}_1-\boldsymbol{\mu}_0)$.
> - **Generative classifiers** instead model each class's distribution $p(\mathbf{x}\mid y=k)$ and classify with Bayes' rule. With Gaussian classes this gives **QDA** (one covariance per class, quadratic boundaries) or **LDA** (shared covariance, linear boundaries).
> - LDA recovers Fisher's direction, and its posterior is exactly a logistic regression.

## Motivation: discriminative vs. generative

Given labelled data $\{(\mathbf{x}_i,y_i)\}_{i=1}^N$ with $\mathbf{x}_i\in\mathbb{R}^D$, there are two broad ways to build a classifier.

- **Discriminative:** model $p(y\mid\mathbf{x})$ directly. [[Linear Models#Logistic Regression|Logistic regression]] does this: it never asks what the inputs of each class look like.
- **Generative:** model how each class *generates* its data, $p(\mathbf{x}\mid y=k)$, together with the class frequencies $p(y=k)$. Then invert with Bayes' rule. If the model is correct, this gives the Bayes-optimal classifier.

Both are **parametric**: we extract a few useful quantities from the data (means, covariances, weights) and can then forget the dataset itself. This lecture follows the generative road. It starts with the simplest classifier, builds up Fisher's projection idea, and ends with Gaussian discriminant analysis, where the two roads meet. For simplicity we mostly consider two classes, $y\in\{0,1\}$. Hastie et al.[^esl] and Bishop[^bishop2006] cover the same material in more depth.

**Notation.** $N_k$ is the number of points in class $k$, and $\boldsymbol{\mu}_k=\frac{1}{N_k}\sum_{i:y_i=k}\mathbf{x}_i$ is the class mean. Sums written $\sum_{i:y_i=k}$ run **only over class $k$**.

---

## A first classifier: the nearest class mean

Summarize each class by its mean, and assign a new point to the class whose mean is **closest**:

$$
\hat y(\mathbf{x})=\arg\min_k\;\lVert\mathbf{x}-\boldsymbol{\mu}_k\rVert_2^2=\arg\max_k\;\Big(\boldsymbol{\mu}_k^\top\mathbf{x}-\tfrac12\lVert\boldsymbol{\mu}_k\rVert_2^2\Big).
$$

The second form comes from expanding the square and dropping $\lVert\mathbf{x}\rVert^2$, which is the same for every class. It shows the rule is **linear** in $\mathbf{x}$. The boundary is the perpendicular bisector of the segment joining the two means. Note that "most similar mean", $\arg\max_k\langle\boldsymbol{\mu}_k,\mathbf{x}\rangle$, is only equivalent when all means have the same norm.

![[Machine Learning/images/discriminant-analysis-nearest-mean.svg]]
*Figure 1: the nearest-class-mean classifier. Each class is summarized by its mean (crosses), and the decision boundary is the perpendicular bisector between them.*

Nearest-mean ignores the **shape** of each class: how spread out it is, and in which directions. Fisher's idea, and later LDA, fix exactly that. (In fact nearest-mean is the special case of LDA with isotropic covariance $\sigma^2\mathbf{I}$ and equal priors.)

---

## Fisher's linear discriminant

Fisher asked: if we project the data onto a single direction $\mathbf{w}$, which direction preserves the class information best?[^fisher1936] We could project onto the standard basis vectors, or onto any other direction. As Figure 2 shows, the choice matters a lot.

![[Machine Learning/images/discriminant-analysis-fisher-projections.svg]]
*Figure 2: the same two-class data projected onto five directions. Each strip shows the projected points, labelled with the Fisher ratio $R$ relative to its best value $R^*$. Along $[1,1]^\top$ the classes overlap completely. Along $[1,-1]^\top$ they separate well. Fisher's direction $\mathbf{S}_W^{-1}(\boldsymbol{\mu}_1-\boldsymbol{\mu}_0)$ separates them best.*

Not all projections are equally good at keeping the classes apart, or at giving a classifier that generalizes. We want a direction that

1. **maximizes the between-class spread:** the projected means should be far apart; and
2. **minimizes the within-class spread:** each projected class should be tight.

**Projected means and spreads.** Projecting onto $\mathbf{w}$ maps $\mathbf{x}\mapsto\mathbf{w}^\top\mathbf{x}$. The projected class means are $m_k=\mathbf{w}^\top\boldsymbol{\mu}_k$, so

$$
(m_1-m_0)^2=\big(\mathbf{w}^\top(\boldsymbol{\mu}_1-\boldsymbol{\mu}_0)\big)^2=\mathbf{w}^\top\underbrace{(\boldsymbol{\mu}_1-\boldsymbol{\mu}_0)(\boldsymbol{\mu}_1-\boldsymbol{\mu}_0)^\top}_{\mathbf{S}_B}\,\mathbf{w}.
$$

$\mathbf{S}_B$ is the **between-class scatter** matrix. The within-class spread of the projected class $k$ is its *scatter* (an unnormalized variance):

$$
s_k^2=\sum_{i:y_i=k}\big(\mathbf{w}^\top\mathbf{x}_i-m_k\big)^2=\mathbf{w}^\top\underbrace{\Big[\sum_{i:y_i=k}(\mathbf{x}_i-\boldsymbol{\mu}_k)(\mathbf{x}_i-\boldsymbol{\mu}_k)^\top\Big]}_{\mathbf{S}_k}\mathbf{w}.
$$

Adding the two classes gives $s_0^2+s_1^2=\mathbf{w}^\top(\mathbf{S}_0+\mathbf{S}_1)\mathbf{w}=\mathbf{w}^\top\mathbf{S}_W\mathbf{w}$. Here $\mathbf{S}_W=\mathbf{S}_0+\mathbf{S}_1$ is the **within-class scatter**. It is *not* the covariance of the whole dataset, which would also include the between-class spread.

**The Fisher criterion** is the ratio we want to make large:

$$
R(\mathbf{w})=\frac{(m_1-m_0)^2}{s_0^2+s_1^2}=\frac{\mathbf{w}^\top\mathbf{S}_B\mathbf{w}}{\mathbf{w}^\top\mathbf{S}_W\mathbf{w}}.
$$

It does not depend on the length of $\mathbf{w}$, only on its direction. A ratio of two quadratic forms like this is a *generalized Rayleigh quotient*.

**Maximizing it.** By the quotient rule, the gradient vanishes when its numerator does:

$$
\begin{aligned}
(\mathbf{w}^\top\mathbf{S}_W\mathbf{w})\,2\mathbf{S}_B\mathbf{w}-(\mathbf{w}^\top\mathbf{S}_B\mathbf{w})\,2\mathbf{S}_W\mathbf{w}&=\mathbf{0}\\
\implies\mathbf{S}_B\mathbf{w}&=R(\mathbf{w})\,\mathbf{S}_W\mathbf{w}.
\end{aligned}
$$

This is a *generalized eigenvalue problem*. With two classes it can be solved by hand, because $\mathbf{S}_B\mathbf{w}=(\boldsymbol{\mu}_1-\boldsymbol{\mu}_0)\big[(\boldsymbol{\mu}_1-\boldsymbol{\mu}_0)^\top\mathbf{w}\big]$ always points along $\boldsymbol{\mu}_1-\boldsymbol{\mu}_0$:

$$
\begin{aligned}
R\,\mathbf{w}&=\mathbf{S}_W^{-1}\mathbf{S}_B\mathbf{w}=\underbrace{\big[(\boldsymbol{\mu}_1-\boldsymbol{\mu}_0)^\top\mathbf{w}\big]}_{\text{a scalar}}\;\mathbf{S}_W^{-1}(\boldsymbol{\mu}_1-\boldsymbol{\mu}_0)\\
\implies\mathbf{w}^*&\propto\mathbf{S}_W^{-1}(\boldsymbol{\mu}_1-\boldsymbol{\mu}_0).
\end{aligned}
$$

Compare with nearest-mean, which effectively uses $\boldsymbol{\mu}_1-\boldsymbol{\mu}_0$ itself. Fisher **whitens** that direction by the within-class scatter, so directions in which the classes are naturally spread out count for less. If $\mathbf{S}_W$ is singular (for example $D>N$), use $\mathbf{S}_W+\epsilon\mathbf{I}$.

> [!note] What Fisher does not give you
> Fisher's criterion needs **no distributional assumption**. It is a geometric statement about means and scatters. It also gives only a *direction*: to classify, we still need a threshold $c$ on $\mathbf{w}^{*\top}\mathbf{x}$, such as the midpoint of the projected means. With $K>2$ classes, the same idea, using $\mathbf{S}_B=\sum_kN_k(\boldsymbol{\mu}_k-\boldsymbol{\mu})(\boldsymbol{\mu}_k-\boldsymbol{\mu})^\top$, gives up to $K-1$ directions: the top generalized eigenvectors. This makes Fisher's method a *supervised dimensionality reduction* technique, the labelled counterpart of PCA.

---

## Generative classifiers and Bayes' rule

Now the probabilistic route. Learn a model of each class, $p(\mathbf{x}\mid y=k)$, and the **class priors** $\pi_k=p(y=k)$, with $\pi_0+\pi_1=1$. Bayes' rule then gives the posterior:

$$
p(y=1\mid\mathbf{x})=\frac{\pi_1\,p(\mathbf{x}\mid y=1)}{\pi_0\,p(\mathbf{x}\mid y=0)+\pi_1\,p(\mathbf{x}\mid y=1)}.
$$

Predicting the class with the larger posterior is the **Bayes-optimal** rule, the one with the fewest expected errors, *provided* the class models are correct. The decision boundary is where the two posteriors are equal, $\pi_0\,p(\mathbf{x}\mid y=0)=\pi_1\,p(\mathbf{x}\mid y=1)$.

**Gaussian class models.** Assume each class is Gaussian, $p(\mathbf{x}\mid y=k)=\mathcal{N}(\mathbf{x};\boldsymbol{\mu}_k,\boldsymbol{\Sigma}_k)$. Given the labels, the maximum-likelihood estimates are just per-class statistics:

$$
\begin{aligned}
\pi_k&=\frac{N_k}{N}=\frac1N\sum_{i=1}^{N}\mathbb{1}\{y_i=k\},\qquad\boldsymbol{\mu}_k=\frac{1}{N_k}\sum_{i:y_i=k}\mathbf{x}_i,\\
\boldsymbol{\Sigma}_k&=\frac{1}{N_k}\sum_{i:y_i=k}(\mathbf{x}_i-\boldsymbol{\mu}_k)(\mathbf{x}_i-\boldsymbol{\mu}_k)^\top.
\end{aligned}
$$

Dividing by $N_k-1$ instead gives the unbiased covariance estimate. The difference is negligible for reasonably large classes.

### Quadratic Discriminant Analysis (QDA)

Take logs of both sides of the boundary condition $\pi_0\,p(\mathbf{x}\mid y=0)=\pi_1\,p(\mathbf{x}\mid y=1)$. The Gaussian density $(2\pi)^{-D/2}\lvert\boldsymbol{\Sigma}_k\rvert^{-1/2}\exp\big(-\tfrac12(\mathbf{x}-\boldsymbol{\mu}_k)^\top\boldsymbol{\Sigma}_k^{-1}(\mathbf{x}-\boldsymbol{\mu}_k)\big)$ turns into a sum, and the $(2\pi)^{-D/2}$ factors cancel. What remains is a comparison of the **discriminant functions**

$$
\delta_k(\mathbf{x})=\log\pi_k-\frac12\log\lvert\boldsymbol{\Sigma}_k\rvert-\frac12(\mathbf{x}-\boldsymbol{\mu}_k)^\top\boldsymbol{\Sigma}_k^{-1}(\mathbf{x}-\boldsymbol{\mu}_k).
$$

$\delta_k(\mathbf{x})$ is the log of $\pi_k\,p(\mathbf{x}\mid y=k)$, up to a constant shared by both classes. The **QDA rule** is

$$
\hat y=1\quad\text{if}\quad\delta_1(\mathbf{x})-\delta_0(\mathbf{x})>0,\qquad\text{and }\hat y=0\text{ otherwise}.
$$

With $K$ classes, predict $\arg\max_k\delta_k(\mathbf{x})$.

> [!warning] Compare differences, not ratios, of log-scores
> It is tempting to write the rule as $\delta_1(\mathbf{x})/\delta_0(\mathbf{x})>1$, but that is wrong. The $\delta_k$ are logarithms and can have either sign, so dividing by a negative $\delta_0$ flips the inequality. Ratios are fine *before* taking logs: $\frac{\pi_1p(\mathbf{x}\mid y=1)}{\pi_0p(\mathbf{x}\mid y=0)}>1$. After taking logs, compare the difference with $0$.

Each $\delta_k$ is quadratic in $\mathbf{x}$, and the quadratic terms $-\frac12\mathbf{x}^\top\boldsymbol{\Sigma}_k^{-1}\mathbf{x}$ differ between the classes. So the boundary $\delta_1=\delta_0$ is a **quadric**: an ellipse, a parabola, a hyperbola, or even a disconnected curve (Figure 3, left).

### Linear Discriminant Analysis (LDA)

Now assume both classes share one covariance, $\boldsymbol{\Sigma}_0=\boldsymbol{\Sigma}_1=\boldsymbol{\Sigma}$. It is estimated by the **pooled within-class covariance** $\boldsymbol{\Sigma}=\frac{1}{N-2}\big(\mathbf{S}_0+\mathbf{S}_1\big)=\frac{\mathbf{S}_W}{N-2}$, not by the covariance of the whole dataset. Expanding the quadratic form,

$$
(\mathbf{x}-\boldsymbol{\mu}_k)^\top\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu}_k)=\mathbf{x}^\top\boldsymbol{\Sigma}^{-1}\mathbf{x}-2\,\mathbf{x}^\top\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}_k+\boldsymbol{\mu}_k^\top\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}_k.
$$

The terms $\log\lvert\boldsymbol{\Sigma}\rvert$ and $\mathbf{x}^\top\boldsymbol{\Sigma}^{-1}\mathbf{x}$ are now the same for both classes and cancel in $\delta_1-\delta_0$. What is left is **linear** in $\mathbf{x}$:

$$
\delta_1(\mathbf{x})-\delta_0(\mathbf{x})=\mathbf{x}^\top\boldsymbol{\Sigma}^{-1}(\boldsymbol{\mu}_1-\boldsymbol{\mu}_0)-\frac12\Big(\boldsymbol{\mu}_1^\top\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}_1-\boldsymbol{\mu}_0^\top\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}_0\Big)+\log\frac{\pi_1}{\pi_0}.
$$

So the **LDA rule** is: predict $\hat y=1$ if

$$
\mathbf{x}^\top\underbrace{\boldsymbol{\Sigma}^{-1}(\boldsymbol{\mu}_1-\boldsymbol{\mu}_0)}_{\mathbf{w}}\;>\;\underbrace{\frac12\Big(\boldsymbol{\mu}_1^\top\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}_1-\boldsymbol{\mu}_0^\top\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}_0\Big)+\log\frac{\pi_0}{\pi_1}}_{c},
$$

and $\hat y=0$ otherwise. In words: project $\mathbf{x}$ onto $\mathbf{w}=\boldsymbol{\Sigma}^{-1}(\boldsymbol{\mu}_1-\boldsymbol{\mu}_0)$ and compare with a threshold. The threshold moves towards the rarer class's mean, so a larger $\pi_1$ makes it easier to predict class $1$.

![[Machine Learning/images/discriminant-analysis-qda-vs-lda.svg]]
*Figure 3: two Gaussian classes with very different shapes (dashed ellipses mark two standard deviations of each fitted Gaussian). QDA fits one covariance per class, and its boundary is a curve; it can even enclose a separate region far from the data. LDA forces one shared covariance, and its boundary is a straight line.*

### LDA and Fisher

Since $\boldsymbol{\Sigma}\propto\mathbf{S}_W$, the LDA direction $\boldsymbol{\Sigma}^{-1}(\boldsymbol{\mu}_1-\boldsymbol{\mu}_0)$ is **the same direction** as Fisher's $\mathbf{S}_W^{-1}(\boldsymbol{\mu}_1-\boldsymbol{\mu}_0)$. The two methods reach it from different starting points.

- **Fisher** is geometric and assumption-free, and gives only a direction.
- **LDA** is probabilistic. It assumes Gaussian classes with a shared covariance, and in exchange gives the Bayes-optimal **threshold**, including the effect of the priors, and calibrated probabilities.

### LDA's posterior is a logistic regression

Plug the LDA expression into Bayes' rule:

$$
p(y=1\mid\mathbf{x})=\frac{1}{1+\exp\big(-(\delta_1(\mathbf{x})-\delta_0(\mathbf{x}))\big)}=\sigma\big(\mathbf{w}^\top\mathbf{x}+b\big),
$$

with $\mathbf{w}=\boldsymbol{\Sigma}^{-1}(\boldsymbol{\mu}_1-\boldsymbol{\mu}_0)$ and $b=-c$. This is exactly the functional form of [[Linear Models#Logistic Regression|logistic regression]]. With $K$ classes it becomes a softmax of linear functions. The two methods produce the **same model family**; they differ in how they fit its parameters.

| | LDA (generative) | Logistic regression (discriminative) |
|---|---|---|
| Fits | $p(\mathbf{x}\mid y)$ and $p(y)$, by closed-form statistics | $p(y\mid\mathbf{x})$ directly, by iterative optimization |
| Assumes | Gaussian classes with a shared covariance | only that the log-odds are linear |
| Parameters (2 classes) | $2D+\frac{D(D+1)}{2}+1$ | $D+1$ |
| Strength | data-efficient when the assumptions hold; handles missing features | robust when the Gaussian assumption fails |

Ng and Jordan made this precise:[^ngjordan2001] generative models often reach their (higher) asymptotic error with fewer samples, while discriminative models win once data is plentiful. QDA needs even more parameters, $\frac{D(D+1)}{2}$ per class, which is why LDA (or a *regularized* QDA that shrinks each $\boldsymbol{\Sigma}_k$ towards a common $\boldsymbol{\Sigma}$) is often preferred in high dimensions. Assuming **diagonal** class covariances gives **Gaussian naive Bayes**.

## Key takeaways

- **Nearest class mean** is the simplest linear classifier. It ignores class shape.
- **Fisher** maximizes $\frac{\mathbf{w}^\top\mathbf{S}_B\mathbf{w}}{\mathbf{w}^\top\mathbf{S}_W\mathbf{w}}$ (between-class over within-class scatter), giving $\mathbf{w}^*\propto\mathbf{S}_W^{-1}(\boldsymbol{\mu}_1-\boldsymbol{\mu}_0)$. It needs no distributional assumption and gives a direction, not a threshold.
- **Generative classifiers** model $p(\mathbf{x}\mid y)$ and $p(y)$ and classify with Bayes' rule. Class means and covariances are estimated **per class**.
- **QDA** uses a separate Gaussian per class and gives quadratic boundaries. The rule is $\delta_1-\delta_0>0$, a difference, never a ratio of log-scores.
- **LDA** uses a shared covariance and gives linear boundaries along Fisher's direction, with a prior-aware threshold. Its posterior is a logistic regression.

## Where this shows up next

- [[Linear Models]]: logistic regression, the discriminative counterpart of LDA.
- [[Support Vector Machines]]: a third route to a separating hyperplane, the one with maximum margin.
- [[Non-parametric Models]]: classifiers that keep the data instead of summarizing it.
- [[Clustering and Expectation Maximization#Gaussian mixture models|Gaussian mixture models]]: QDA with *unknown* labels, fitted by EM with soft, probabilistic class assignments.
- **Dimensionality reduction** (coming soon): PCA, the unsupervised cousin of Fisher's projection.

<!-- References: footnote definitions below render as the numbered "References" list at the bottom of the page -->

[^fisher1936]: Fisher, R. A. (1936). *The Use of Multiple Measurements in Taxonomic Problems*. Annals of Eugenics, 7(2), 179–188. [doi:10.1111/j.1469-1809.1936.tb02137.x](https://doi.org/10.1111/j.1469-1809.1936.tb02137.x)
[^ngjordan2001]: Ng, A. Y., & Jordan, M. I. (2001). *On Discriminative vs. Generative Classifiers: A comparison of logistic regression and naive Bayes*. NeurIPS. [link](https://papers.nips.cc/paper_files/paper/2001/hash/7b7a53e239400a13bd6be6c91c4f6c4e-Abstract.html)
[^esl]: Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.), Chapter 4.3. [book website](https://hastie.su.domains/ElemStatLearn/)
[^bishop2006]: Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*, Chapter 4.1–4.2. Springer. [book page](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/)
