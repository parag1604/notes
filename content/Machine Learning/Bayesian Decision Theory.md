---
title: Bayesian Decision Theory
tags:
  - machine-learning
  - supervised-learning
  - classification
date: 2026-11-12
draft: false
---

> **Prerequisites:** [[Probability Basics#Bayes' theorem|Bayes' theorem]], [[Random Variables]] (densities, expectation) and [[Joint Distributions#The multivariate Gaussian|the multivariate Gaussian]].
>
> **TL;DR:**
> - If we knew the class priors $\pi_k$ and the class-conditional densities $p(\mathbf{x}\mid k)$, the best possible classifier would be known: pick the class with the largest posterior $\mathbb{P}(k\mid\mathbf{x})\propto p(\mathbf{x}\mid k)\,\pi_k$. Its error, the **Bayes error**, is a floor that no classifier using the same features can beat.
> - With unequal costs of mistakes, minimize the expected loss (the **conditional risk**) instead. For two classes this is a **likelihood ratio test**: the costs and priors only move the threshold.
> - Variations: a **reject option** for uncertain cases, and the **minimax** rule when the priors are unknown.
> - With Gaussian class-conditionals the optimal boundaries are linear (equal covariances) or quadratic (different covariances). Every classifier in this series is an attempt to approximate the Bayes classifier from data.

## Motivation

A fisherman sorts his catch on a conveyor belt into two species, salmon and sea bass. A camera measures the lightness $x$ of each fish. Salmon tend to be darker, but the two ranges overlap, so some fish will be misclassified whatever rule we use. What is the **best** rule, and how good can it possibly be?

Bayesian decision theory answers this in the idealized case where all the relevant probabilities are known. It tells us what an optimal classifier looks like, and what error is unavoidable. Real classifiers must learn from data, but they are all trying to get close to this ideal, and its error is the benchmark they are measured against. This lecture follows Chapter 2 of Duda, Hart and Stork, the classic treatment of this material.[^duda2001] Bishop and Murphy cover it more briefly.[^bishop2006]<sup>,</sup>[^murphy2022]

**Notation.** Classes are $k=1,\dots,K$ with priors $\pi_k=\mathbb{P}(y=k)$. The class-conditional densities, or likelihoods, are $p(\mathbf{x}\mid k)$, and the posteriors are $\mathbb{P}(k\mid\mathbf{x})$. In the running example $K=2$: class 1 is salmon, class 2 is sea bass, and $x$ is a scalar.

## Three ways to decide

For the running example, suppose that salmon lightness is $\mathcal{N}(4,1)$ and sea bass lightness is $\mathcal{N}(7,1.5^2)$, and that this season one fish in three is a salmon: $\pi_1=1/3$, $\pi_2=2/3$.

**1. Prior only.** Without looking at the fish, the best guess is the more common class: always say "sea bass". This is right $2/3$ of the time. Its error, $\min(\pi_1,\pi_2)=1/3$, is the baseline any useful rule must beat. But every fish gets the same label.

**2. Likelihood only.** Now look at $x$, and ask which species makes this lightness more probable: decide salmon if $p(x\mid1)>p(x\mid2)$. This is the **maximum-likelihood** decision. The two densities cross at $x=5.40$ (Figure 1a), so fish darker than that are called salmon. It ignores the fact that sea bass are twice as common.

**3. Posterior.** [[Probability Basics#Bayes' theorem|Bayes' theorem]] combines both sources of information:

$$
\mathbb{P}(k\mid x)=\frac{p(x\mid k)\,\pi_k}{p(x)},\qquad p(x)=\sum_{j=1}^{K}p(x\mid j)\,\pi_j.
\tag{1}
$$

*Posterior = likelihood × prior / evidence.* The evidence $p(x)$ is the same for every class, so it only rescales. The **Bayes decision rule** picks the class with the largest posterior. Equivalently, decide salmon if

$$
p(x\mid1)\,\pi_1>p(x\mid2)\,\pi_2.
$$

Now the boundary moves to $x^*=5.05$ (Figure 1b). Because sea bass are more common, a fish has to be noticeably dark before we call it a salmon. The prior pushes the boundary towards the less likely class.

![[Machine Learning/images/bayesian-decision-theory-posteriors.svg]]
*Figure 1. The running example. (a) The class-conditional densities of lightness cross at $x=5.40$: the maximum-likelihood boundary. (b) The posteriors with priors $1/3$ (salmon) and $2/3$ (sea bass) cross at $x^*=5.05$. Fish darker than $x^*$ are called salmon (blue region), lighter ones sea bass (orange region).*

## The probability of error

Why is the posterior rule the best? Given $x$, if we decide class $k$ we are wrong with probability $1-\mathbb{P}(k\mid x)$. Choosing the largest posterior makes this conditional error as small as possible:

$$
\mathbb{P}(\text{error}\mid x)=1-\max_k\mathbb{P}(k\mid x).
$$

The overall error averages this over all $x$:

$$
\mathbb{P}(\text{error})=\int\mathbb{P}(\text{error}\mid x)\,p(x)\,dx.
$$

The integrand is non-negative, and the Bayes rule makes it as small as possible **at every single $x$**, so it also minimizes the integral. No other rule, however clever, can do better.

For two classes, the error has a nice picture (Figure 2). Plot the two curves $\pi_k\,p(x\mid k)$. At each $x$, the Bayes rule picks the higher curve, and the probability of being wrong there is the lower curve. So

$$
P_{\text{Bayes}}=\int\min\big(\pi_1\,p(x\mid1),\ \pi_2\,p(x\mid2)\big)\,dx,
$$

the area under the lower envelope of the two curves. In the running example it is 0.114. Moving the boundary away from $x^*$, say to $x'=6.35$, adds the purple sliver of Figure 2, and doubles the error to 0.225.

![[Machine Learning/images/bayesian-decision-theory-error.svg]]
*Figure 2. The Bayes error is the red area under the lower of the two curves $\pi_k\,p(x\mid k)$. Any other boundary, such as $x'$, adds an extra region (purple) in which the less likely class is chosen.*

> [!note] The Bayes error is a property of the features, not of the classifier
> The Bayes error is the error of the best possible classifier that uses the features $\mathbf{x}$. If the class-conditional densities overlap, as here, some mistakes are unavoidable. The only way to reduce the Bayes error is to measure **better features**: add the fish's width to its lightness, and the classes may separate further. A learned classifier's error splits into the Bayes error (unavoidable) plus the excess error from not knowing the true distributions.

## Losses and risk

Not all mistakes are equally bad. Missing a tumour is worse than a false alarm; letting spam through is milder than deleting an important email. Decision theory handles this with a **loss function**:
- a set of **actions** $a_1,\dots,a_A$, usually "decide class $i$", possibly with extra ones such as "reject";
- a **loss** $\lambda(a_i\mid k)$, the cost of taking action $a_i$ when the true class is $k$. We write $\lambda_{ik}$ for short.

Given $\mathbf{x}$, the expected loss of taking action $a_i$ is the **conditional risk**

$$
R(a_i\mid\mathbf{x})=\sum_{k=1}^{K}\lambda(a_i\mid k)\,\mathbb{P}(k\mid\mathbf{x}).
\tag{2}
$$

A **decision rule** $\alpha(\mathbf{x})$ chooses an action for every $\mathbf{x}$, and its **overall risk** is the average

$$
R[\alpha]=\int R\big(\alpha(\mathbf{x})\mid\mathbf{x}\big)\,p(\mathbf{x})\,d\mathbf{x}.
$$

The same argument as for the error applies: to minimize the overall risk, minimize the conditional risk separately at each $\mathbf{x}$. The **Bayes decision rule** is

$$
\alpha^*(\mathbf{x})=\arg\min_{a_i}R(a_i\mid\mathbf{x}),
$$

and its risk $R^*=R[\alpha^*]$ is the **Bayes risk**, the smallest achievable.

### 0-1 loss gives the maximum posterior

With the **0-1 loss**, every mistake costs 1 and every correct decision costs 0: $\lambda_{ik}=0$ if $i=k$ and 1 otherwise. Then (2) becomes

$$
R(a_i\mid\mathbf{x})=\sum_{k\ne i}\mathbb{P}(k\mid\mathbf{x})=1-\mathbb{P}(i\mid\mathbf{x}).
$$

Minimizing the risk means maximizing the posterior. So the posterior rule of the previous sections is the Bayes rule for 0-1 loss, and its risk is the probability of error. This rule is also called the **maximum a posteriori** (MAP) rule.

### Two classes: the likelihood ratio test

With two classes and two actions ($a_i$ = "decide class $i$"), we should decide class 1 when $R(a_1\mid\mathbf{x})<R(a_2\mid\mathbf{x})$:

$$
\lambda_{11}\,\mathbb{P}(1\mid\mathbf{x})+\lambda_{12}\,\mathbb{P}(2\mid\mathbf{x})<\lambda_{21}\,\mathbb{P}(1\mid\mathbf{x})+\lambda_{22}\,\mathbb{P}(2\mid\mathbf{x}).
$$

It is natural to assume that a mistake costs more than a correct decision: $\lambda_{21}>\lambda_{11}$ and $\lambda_{12}>\lambda_{22}$. Collecting terms, and replacing the posteriors by likelihood × prior via (1), gives the rule: decide class 1 if

$$
\psi(\mathbf{x})=\frac{p(\mathbf{x}\mid1)}{p(\mathbf{x}\mid2)}>\theta,\qquad\theta=\frac{\lambda_{12}-\lambda_{22}}{\lambda_{21}-\lambda_{11}}\cdot\frac{\pi_2}{\pi_1}.
\tag{3}
$$

This is a **likelihood ratio test**. The data enter only through the likelihood ratio $\psi(\mathbf{x})$, while the costs and the priors are folded into a threshold $\theta$ that does not depend on $\mathbf{x}$. Changing the costs or the priors does not change the classifier's "shape", only where we cut.

Figure 3 shows this for the fish. With 0-1 loss, $\theta=\pi_2/\pi_1=2$ and the boundary is at 5.05, as before. Suppose instead that wrongly calling a salmon "sea bass" costs 5 (salmon are expensive), while the opposite mistake costs 1. Then $\theta=\frac{1}{5}\cdot2=0.4$, and the boundary moves to 5.80. More fish are called salmon, because missing one is now expensive.

![[Machine Learning/images/bayesian-decision-theory-likelihood-ratio.svg]]
*Figure 3. The likelihood ratio $\psi(x)$ on a log scale, with two thresholds. With 0-1 loss we decide salmon wherever $\psi(x)>2$, that is for $x<5.05$. If missing a salmon costs five times as much, the threshold drops to 0.4 and the boundary moves to $x=5.80$.*

> [!note] Decision regions need not be connected
> Because the sea bass density is wider, it dominates again in the far left tail. The likelihood ratio drops below 2 again for $x<-1.85$, off the plot, so the exact Bayes region for salmon is the interval $(-1.85,\,5.05)$. Lightness cannot be negative, so this does not matter here. In general, though, unequal spreads make decision regions curved, or even split into pieces. We will see the same thing in two dimensions [[#Gaussian class-conditionals|below]].

> [!example]- Aside: Neyman–Pearson and ROC curves
> Sometimes we cannot put a price on mistakes, but we can cap one kind of error. A radar must keep its false-alarm rate below 1%, say, and then detect as many real aircraft as possible. Neyman and Pearson showed that the best test under such a constraint is again a likelihood ratio test (3), with the threshold chosen to meet the constraint.[^neyman1933] Sweeping $\theta$ from $\infty$ to 0 traces out all the achievable pairs of (false positive rate, true positive rate): the **ROC curve**. A classifier that outputs a score is evaluated by its whole ROC curve, and each operating point corresponds to some choice of costs and priors.

### The reject option

When the largest posterior is not much larger than the others, any decision is close to a coin flip. Often it is better not to decide, and to hand the case to a human expert or ask for more tests. Add a **reject** action with a fixed cost $c$, cheaper than a likely mistake. With 0-1 loss for the other actions, the conditional risk of deciding class $i$ is $1-\mathbb{P}(i\mid\mathbf{x})$, and the risk of rejecting is $c$. So the Bayes rule becomes:

$$
\text{decide }\arg\max_k\mathbb{P}(k\mid\mathbf{x})\ \text{ if }\ \max_k\mathbb{P}(k\mid\mathbf{x})\ge1-c,\qquad\text{otherwise reject.}
$$

The cheaper rejection is, the more cases are rejected and the lower the error on the cases we do decide. Chow analysed this error–reject trade-off.[^chow1970] Figure 4 shows the reject band for the fish with $c=0.2$: every fish with lightness between 4.23 and 5.71 is sent for manual inspection. The option only makes sense if rejecting is cheaper than guessing: with $K$ classes the largest posterior is always at least $1/K$, so for $c\ge(K-1)/K$ (for two classes, $c\ge\tfrac12$) the rule never rejects.

![[Machine Learning/images/bayesian-decision-theory-reject.svg]]
*Figure 4. A reject option with cost $c=0.2$. Where neither posterior reaches $1-c=0.8$ (the shaded band, $4.23<x<5.71$), rejecting is cheaper than the expected cost of a mistake.*

## The minimax rule: when the priors are unknown

The Bayes rule needs the priors. What if they vary, or are unknown? The salmon fraction changes with the season, and the sorting machine must work all year. Fix a decision rule, with decision regions $\mathcal{R}_1$ (decide class 1) and $\mathcal{R}_2$. Write $\pi_2=1-\pi_1$, and use $\int_{\mathcal{R}_1}p(\mathbf{x}\mid1)\,d\mathbf{x}=1-\int_{\mathcal{R}_2}p(\mathbf{x}\mid1)\,d\mathbf{x}$ and the same identity for class 2. Then the overall risk of the rule, as a function of the prior, is

$$
\begin{aligned}
R(\pi_1)&=\lambda_{22}+(\lambda_{12}-\lambda_{22})\int_{\mathcal{R}_1}p(\mathbf{x}\mid2)\,d\mathbf{x}\\
&\quad+\pi_1\Big[(\lambda_{11}-\lambda_{22})+(\lambda_{21}-\lambda_{11})\int_{\mathcal{R}_2}p(\mathbf{x}\mid1)\,d\mathbf{x}\\
&\qquad\qquad-(\lambda_{12}-\lambda_{22})\int_{\mathcal{R}_1}p(\mathbf{x}\mid2)\,d\mathbf{x}\Big].
\end{aligned}
\tag{4}
$$

Once the regions are fixed, the risk is a **straight line** in $\pi_1$. Its slope is the bracket. A rule tuned for one prior can do badly if the true prior is different (the dashed lines in Figure 5).

The **minimax** rule protects against the worst case: it minimizes the maximum risk over all priors. If we choose the regions so that the bracket in (4) is zero, the risk does not depend on the prior at all. The line is flat, at the height of its first two terms, which is the **minimax risk**. For 0-1 loss the bracket is $\int_{\mathcal{R}_2}p(\mathbf{x}\mid1)\,d\mathbf{x}-\int_{\mathcal{R}_1}p(\mathbf{x}\mid2)\,d\mathbf{x}$, so the minimax rule simply **equalizes the two conditional error rates**.

For the fish, this happens at a boundary of $x=5.20$. The minimax rule then errs on 11.5% of fish of either species, whatever the season (Figure 5). That is exactly the peak of the Bayes risk curve. This is no coincidence: the minimax rule is the Bayes rule for the **least favourable** prior, the one that makes the problem hardest. Minimax reasoning, where an adversary picks the prior, comes from game theory.

![[Machine Learning/images/bayesian-decision-theory-minimax.svg]]
*Figure 5. Risk against the true prior. The blue curve is the Bayes risk, achievable only if the prior is known. A rule tuned for one prior has a risk that is a straight line touching the curve at that prior (dashed), and it can do badly elsewhere. The minimax rule's line is flat, at the maximum of the Bayes risk curve.*

## Discriminant functions

A convenient way to describe any classifier is through **discriminant functions** $g_1(\mathbf{x}),\dots,g_K(\mathbf{x})$: assign $\mathbf{x}$ to the class with the largest $g_k(\mathbf{x})$. The Bayes classifier uses $g_k(\mathbf{x})=-R(a_k\mid\mathbf{x})$, and for 0-1 loss $g_k(\mathbf{x})=\mathbb{P}(k\mid\mathbf{x})$.

Applying the same increasing function to every $g_k$ does not change which one is largest. So the following three choices give the same classifier:

$$
g_k(\mathbf{x})=\mathbb{P}(k\mid\mathbf{x}),\qquad g_k(\mathbf{x})=p(\mathbf{x}\mid k)\,\pi_k,\qquad g_k(\mathbf{x})=\ln p(\mathbf{x}\mid k)+\ln\pi_k.
$$

The last one is the most convenient, because logarithms turn products into sums and are kind to exponential-family densities. With two classes a single discriminant suffices, $g(\mathbf{x})=g_1(\mathbf{x})-g_2(\mathbf{x})$, and we decide class 1 when $g(\mathbf{x})>0$:

$$
g(\mathbf{x})=\ln\frac{p(\mathbf{x}\mid1)}{p(\mathbf{x}\mid2)}+\ln\frac{\pi_1}{\pi_2}=\ln\frac{\mathbb{P}(1\mid\mathbf{x})}{\mathbb{P}(2\mid\mathbf{x})}.
$$

That is the **log posterior odds**, which is exactly what [[Linear Models#Logistic regression|logistic regression]] models as a linear function of $\mathbf{x}$.

## Gaussian class-conditionals

The most important special case is Gaussian class-conditional densities, $p(\mathbf{x}\mid k)=\mathcal{N}(\boldsymbol\mu_k,\boldsymbol\Sigma_k)$ in $D$ dimensions. Dropping the constant $-\frac D2\ln2\pi$, which is the same for every class, the log discriminant is

$$
g_k(\mathbf{x})=-\frac12(\mathbf{x}-\boldsymbol\mu_k)^\top\boldsymbol\Sigma_k^{-1}(\mathbf{x}-\boldsymbol\mu_k)-\frac12\ln|\boldsymbol\Sigma_k|+\ln\pi_k.
\tag{5}
$$

The first term is minus half the squared **Mahalanobis distance** from $\mathbf{x}$ to the class mean. The covariances determine the shape of the boundary, in three cases of increasing generality (Figure 6).

**Case 1: $\boldsymbol\Sigma_k=\sigma^2\mathbf{I}$.** All classes are round blobs of the same size. Then (5) is $-\|\mathbf{x}-\boldsymbol\mu_k\|^2/(2\sigma^2)+\ln\pi_k$. The term $\mathbf{x}^\top\mathbf{x}$ is common to all classes, and dropping it leaves a **linear** discriminant:

$$
g_k(\mathbf{x})=\mathbf{w}_k^\top\mathbf{x}+w_{k0},\qquad\mathbf{w}_k=\frac{\boldsymbol\mu_k}{\sigma^2},\qquad w_{k0}=-\frac{\|\boldsymbol\mu_k\|^2}{2\sigma^2}+\ln\pi_k.
$$

The boundary between two classes is a hyperplane perpendicular to $\boldsymbol\mu_1-\boldsymbol\mu_2$. With equal priors it is the perpendicular bisector of the two means, and the rule is the [[Discriminant Analysis#A first classifier: the nearest class mean|nearest class mean classifier]]. Unequal priors slide the hyperplane towards the less likely class's mean (dashed line in Figure 6a). Setting $g_1=g_2$, the boundary is $(\boldsymbol\mu_1-\boldsymbol\mu_2)^\top(\mathbf{x}-\mathbf{x}_0)=0$ with

$$
\mathbf{x}_0=\frac12(\boldsymbol\mu_1+\boldsymbol\mu_2)-\frac{\sigma^2}{\lVert\boldsymbol\mu_1-\boldsymbol\mu_2\rVert^2}\ln\frac{\pi_1}{\pi_2}\,(\boldsymbol\mu_1-\boldsymbol\mu_2).
$$

The shift grows with the noise $\sigma^2$ and shrinks as the means move apart: the more the classes overlap, the more the prior matters.

**Case 2: shared $\boldsymbol\Sigma_k=\boldsymbol\Sigma$.** All classes have the same elliptical shape. The quadratic term $\mathbf{x}^\top\boldsymbol\Sigma^{-1}\mathbf{x}$ is again common to all classes and drops out, so the discriminant is still **linear**:

$$
\mathbf{w}_k=\boldsymbol\Sigma^{-1}\boldsymbol\mu_k,\qquad w_{k0}=-\frac12\boldsymbol\mu_k^\top\boldsymbol\Sigma^{-1}\boldsymbol\mu_k+\ln\pi_k.
$$

The boundary is the hyperplane $\mathbf{w}^\top(\mathbf{x}-\mathbf{x}_0)=0$ with

$$
\begin{aligned}
\mathbf{w}&=\boldsymbol\Sigma^{-1}(\boldsymbol\mu_1-\boldsymbol\mu_2),\\
\mathbf{x}_0&=\frac12(\boldsymbol\mu_1+\boldsymbol\mu_2)-\frac{\ln(\pi_1/\pi_2)}{(\boldsymbol\mu_1-\boldsymbol\mu_2)^\top\boldsymbol\Sigma^{-1}(\boldsymbol\mu_1-\boldsymbol\mu_2)}\,(\boldsymbol\mu_1-\boldsymbol\mu_2).
\end{aligned}
$$
 This is generally **not** perpendicular to the line joining the means: it is tilted to account for the correlation (Figure 6b). The rule is "nearest mean in Mahalanobis distance". With equal priors its error is $\Phi(-\Delta/2)$, where $\Delta$ is the Mahalanobis distance between the two means. When these parameters are estimated from data, this is **linear discriminant analysis** (LDA).

**Case 3: arbitrary $\boldsymbol\Sigma_k$.** Nothing cancels, and the discriminant is **quadratic**:

$$
\begin{aligned}
g_k(\mathbf{x})&=\mathbf{x}^\top\mathbf{W}_k\mathbf{x}+\mathbf{w}_k^\top\mathbf{x}+w_{k0},\qquad\mathbf{W}_k=-\frac12\boldsymbol\Sigma_k^{-1},\qquad\mathbf{w}_k=\boldsymbol\Sigma_k^{-1}\boldsymbol\mu_k,\\
w_{k0}&=-\frac12\boldsymbol\mu_k^\top\boldsymbol\Sigma_k^{-1}\boldsymbol\mu_k-\frac12\ln|\boldsymbol\Sigma_k|+\ln\pi_k.
\end{aligned}
$$

The boundaries are quadric surfaces: ellipses, parabolas, hyperbolas, or pairs of lines. A tight class can be completely surrounded by a broad one, as in Figure 6c (and as in the fish example, where the salmon region was an interval). Estimated from data, this is **quadratic discriminant analysis** (QDA).

![[Machine Learning/images/bayesian-decision-theory-gaussian-cases.svg]]
*Figure 6. Bayes decision boundaries (red) for two Gaussian classes (blue and orange contours) with equal priors (solid) and with $\pi_1=0.8$ (dashed). (a) Equal round covariances: a straight line perpendicular to the line joining the means; the prior shifts it towards class 2. (b) A shared elliptical covariance: still a straight line, but tilted. (c) Different covariances: a quadratic curve that wraps around the tighter class.*

> [!example]- Code: the Bayes error, by integration and by simulation
> ```python
> import numpy as np
> from scipy import stats, optimize, integrate
>
> pi = np.array([1/3, 2/3])                          # priors: salmon, sea bass
> dists = [stats.norm(4.0, 1.0), stats.norm(7.0, 1.5)]
>
> def bayes_decide(x):
>     """0-1 loss: pick the class with the largest pi_k p(x|k)."""
>     scores = np.stack([np.log(p) + d.logpdf(x) for p, d in zip(pi, dists)])
>     return scores.argmax(axis=0)
>
> # Bayes error = integral of the smaller of the two curves pi_k p(x|k)
> bayes_err, _ = integrate.quad(lambda x: min(pi[0] * dists[0].pdf(x), pi[1] * dists[1].pdf(x)), -20, 30, limit=200)
>
> # check by simulation, and compare with a worse threshold
> rng = np.random.default_rng(0)
> n = 1_000_000
> y = rng.choice(2, size=n, p=pi)
> x = np.where(y == 0, dists[0].rvs(n, random_state=rng), dists[1].rvs(n, random_state=rng))
> print(bayes_err, np.mean(bayes_decide(x) != y))         # both about 0.1135
> print(np.mean((x > 6.35).astype(int) != y))             # threshold 6.35 instead: about 0.225
>
> # multivariate: shared covariance, equal priors -> Bayes error = Phi(-Delta/2)
> mu1, mu2, S = np.zeros(2), np.array([2.5, 1.5]), np.array([[1.6, 0.9], [0.9, 1.0]])
> delta = np.sqrt((mu2 - mu1) @ np.linalg.solve(S, mu2 - mu1))   # Mahalanobis distance
> print(delta, stats.norm.cdf(-delta / 2))                       # about 1.98 and 0.161
> ```

## From decision theory to learning

Everything above assumed the distributions were known. In practice we have only a labelled dataset $\mathcal{D}=\{(\mathbf{x}_i,y_i)\}_{i=1}^N$, and the rest of the supervised-learning lectures are different strategies for getting close to the Bayes classifier from data:
- **Generative (plug-in) classifiers** estimate $\pi_k$ and $p(\mathbf{x}\mid k)$ and then apply Bayes' rule. With Gaussian class-conditionals these are LDA and QDA, and with independent features, naive Bayes ([[Discriminant Analysis]]).
- **Discriminative probabilistic classifiers** model the posterior $\mathbb{P}(k\mid\mathbf{x})$ directly, as [[Linear Models#Logistic regression|logistic and softmax regression]] do.
- **Discriminant-function methods** learn the boundary itself, without probabilities, by minimizing a surrogate for the empirical risk: [[Support Vector Machines]] (hinge loss) and neural networks.
- **Non-parametric methods** estimate the posterior locally, from the training points near $\mathbf{x}$. The 1-nearest-neighbour rule's asymptotic error is at most twice the Bayes error ([[Non-parametric Models]]).

### Learning as risk minimization

The strategies above can all be described in one language, which makes precise what "getting close to the Bayes classifier" means.[^shalev2014] Data come from an unknown joint distribution, $(\mathbf{x},y)\sim P$. A predictor $h$ is judged by a loss $\ell(h(\mathbf{x}),y)$, and its quality is the **true risk**

$$
R(h)=\mathbb{E}_{(\mathbf{x},y)\sim P}\big[\ell(h(\mathbf{x}),y)\big].
$$

With 0-1 loss, $R(h)$ is the probability of error, and the smallest possible value over all functions is the Bayes error. We cannot compute $R(h)$, because $P$ is unknown. We can compute the **empirical risk**, the average loss on the training set:

$$
\hat R_N(h)=\frac1N\sum_{i=1}^{N}\ell\big(h(\mathbf{x}_i),y_i\big).
$$

For a fixed $h$, the law of large numbers says $\hat R_N(h)\to R(h)$. **Empirical risk minimization** (ERM) picks the predictor with the smallest training loss within a chosen **hypothesis class** $\mathcal{H}$: linear functions, kernel machines, trees, neural networks of a given size.

Restricting to a class is essential. Over all functions, a lookup table that memorizes the training labels has zero empirical risk and is useless on new data. And because $\hat h$ was chosen to fit the training data, its training error is **optimistically biased**. That is why performance is always measured on held-out data ([[Index (Machine Learning)#Data splits|data splits]]). The excess risk of the learned predictor over the Bayes risk $R^*$ splits into two parts:

$$
R(\hat h)-R^*=\underbrace{\Big(R(\hat h)-\inf_{h\in\mathcal{H}}R(h)\Big)}_{\text{estimation error}}+\underbrace{\Big(\inf_{h\in\mathcal{H}}R(h)-R^*\Big)}_{\text{approximation error}}.
$$

A richer class lowers the approximation error (the best member is closer to the Bayes classifier) but raises the estimation error (with finite data we may pick the wrong member). **Regularized ERM**, $\min_{h}\hat R_N(h)+\lambda\,\Omega(h)$, controls this trade-off continuously: ridge and lasso in [[Linear Models#MAP estimation and regularization|Linear Models]] are examples.

How large can the estimation error be? For a finite class of $M$ predictors and a loss bounded in $[0,1]$, Hoeffding's inequality plus the union bound ([[Sample Statistics#Concentration and the union bound|Sample Statistics]]) give, with probability at least $1-\delta$, simultaneously for every $h\in\mathcal{H}$,

$$
\begin{aligned}
\big|\hat R_N(h)-R(h)\big|&\le\sqrt{\frac{\ln(2M/\delta)}{2N}},\qquad\text{hence}\\
R(\hat h)&\le\inf_{h\in\mathcal{H}}R(h)+2\sqrt{\frac{\ln(2M/\delta)}{2N}}.
\end{aligned}
$$

The price of a larger class is only logarithmic in its size. Infinite classes need finer measures of complexity, such as the VC dimension, but the message is the same: more data, or a simpler class, makes training error a reliable guide to test error.

Finally, the 0-1 loss is neither continuous nor convex, so minimizing it directly is computationally hard. Practical methods minimize a convex **surrogate** instead: the squared loss, the logistic loss (cross-entropy), or the hinge loss of SVMs. For these, minimizing the surrogate risk over all functions still yields the Bayes classifier, so nothing is lost in the limit.[^bartlett2006]

Decision theory also separates two questions that are easy to conflate. **Inference** means estimating the probabilities. **Decision** means turning probabilities into actions using costs. A well-calibrated probabilistic model can be reused for any cost structure by moving the threshold in (3). A model trained to output hard labels cannot.

## Key takeaways

- With known priors and class-conditional densities, the Bayes rule (largest posterior, $\propto p(\mathbf{x}\mid k)\,\pi_k$) minimizes the probability of error, because it minimizes the conditional error at every $\mathbf{x}$. Its error, the Bayes error, is the area under the lower envelope of the curves $\pi_k\,p(\mathbf{x}\mid k)$. Only better features can reduce it.
- With a loss function, minimize the conditional risk $R(a\mid\mathbf{x})=\sum_k\lambda(a\mid k)\,\mathbb{P}(k\mid\mathbf{x})$. 0-1 loss gives the MAP rule. For two classes the rule is a likelihood ratio test whose threshold absorbs the costs and the priors.
- A reject option abstains when the largest posterior is below $1-c$. The minimax rule is the safe choice when priors are unknown; for 0-1 loss it equalizes the two conditional error rates.
- Discriminant functions are equivalent up to increasing transformations; $\ln p(\mathbf{x}\mid k)+\ln\pi_k$ is the convenient one. For two classes it gives the log posterior odds.
- Learning from data is (regularized) empirical risk minimization over a hypothesis class. Its excess risk over the Bayes risk splits into estimation and approximation error; held-out data are needed because training error is optimistic.
- Gaussian class-conditionals give linear boundaries with equal covariances (nearest mean, LDA) and quadratic ones otherwise (QDA). The priors shift the boundaries.

## Where this shows up next

- [[Linear Models]] models the log posterior odds directly (logistic regression), and fits regression by minimizing squared loss, the regression analogue of the risk.
- [[Discriminant Analysis]] estimates the Gaussian discriminants of this lecture from data (LDA, QDA, naive Bayes).
- [[Support Vector Machines]] and neural networks learn decision boundaries directly; [[Non-parametric Models]] relate $k$-NN's error to the Bayes error.
- In [[Introduction to Reinforcement Learning]], an agent again chooses actions to minimize expected loss (or maximize expected reward), but its actions also change the future.

<!-- References: footnote definitions below render as the numbered "References" list at the bottom of the page -->

[^duda2001]: Duda, R. O., Hart, P. E., & Stork, D. G. (2001). *Pattern Classification* (2nd ed.), Chapter 2 (Bayesian Decision Theory). Wiley. [book page](https://www.wiley.com/en-us/Pattern+Classification%2C+2nd+Edition-p-9780471056690)
[^bishop2006]: Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*, Section 1.5 (Decision theory). Springer. [book page](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/)
[^murphy2022]: Murphy, K. P. (2022). *Probabilistic Machine Learning: An Introduction*, Chapter 5 (Decision theory). MIT Press. [book page](https://probml.github.io/pml-book/book1.html)
[^neyman1933]: Neyman, J., & Pearson, E. S. (1933). *On the problem of the most efficient tests of statistical hypotheses*. Philosophical Transactions of the Royal Society A, 231, 289–337. [doi:10.1098/rsta.1933.0009](https://doi.org/10.1098/rsta.1933.0009)
[^chow1970]: Chow, C. K. (1970). *On optimum recognition error and reject tradeoff*. IEEE Transactions on Information Theory, 16(1), 41–46. [doi:10.1109/TIT.1970.1054406](https://doi.org/10.1109/TIT.1970.1054406)
[^shalev2014]: Shalev-Shwartz, S., & Ben-David, S. (2014). *Understanding Machine Learning: From Theory to Algorithms*, Chapters 2–5. Cambridge University Press. [Free online edition](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/)
[^bartlett2006]: Bartlett, P. L., Jordan, M. I., & McAuliffe, J. D. (2006). *Convexity, Classification, and Risk Bounds*. Journal of the American Statistical Association, 101(473), 138–156. [doi:10.1198/016214505000000907](https://doi.org/10.1198/016214505000000907)
