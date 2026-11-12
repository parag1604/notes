---
title: Probability Basics
tags:
  - probability
  - foundations
date: 2026-11-12
draft: false
prev: "[[Index (Probability and Linear Algebra)]]"
next: "[[Random Variables]]"
---

> **Prerequisites:** none beyond sets, sums, integrals and limits. Every other lecture in the series builds on this one.
>
> **Series:** this is part 1 of 5. **Probability Basics** → [[Random Variables]] → [[Joint Distributions]] → [[Sample Statistics]] → [[Markov Chains]].
>
> **TL;DR:**
> - Events are sets, so the language of probability is the language of sets: "or" is union, "and" is intersection, "not" is complement.
> - A probability space has three ingredients: a sample space $\Omega$ of outcomes, a σ-algebra $\mathcal{F}$ of events, and a probability $\mathbb{P}$ that obeys three axioms (normalized, non-negative, countably additive). A probability measures the *size* of an event, by counting, weighing or integrating.
> - The first consequences are the workhorses of later proofs: $\mathbb{P}(\emptyset)=0$, monotonicity, inclusion–exclusion, the **union bound**, and continuity of probability.
> - Conditioning shrinks the universe to what we know and renormalizes. The law of total probability and Bayes' rule follow, and most "paradoxes" (base rates, Monty Hall) dissolve once the conditional probabilities are written out.
>
> **Notation in this lecture series:** random variables are uppercase ($X$, $Y$) and their values lowercase ($x$, $y$), the standard convention in probability. The other lectures write vectors in bold lowercase and use $p(\cdot)$ for both PMFs and densities.

## Motivation

Probability provides a mathematical framework for reasoning under uncertainty. Every lecture in this series leans on it:
- regression assumes Gaussian noise;
- classifiers output class probabilities;
- clustering fits mixture models;
- reinforcement learning averages over random trajectories;
- generative models *are* probability distributions.

There are several interpretations of what a probability *means*. The **frequentist** view, which we use to motivate the definitions, says that $\mathbb{P}(A)$ is the long-run fraction of experiments in which $A$ happens. The **Bayesian** view treats probability as a degree of belief, which is how priors and MAP estimation are used in [[Linear Models#MAP estimation and regularization|Linear Models]].[^jaynes2003] The good news is that both obey exactly the same rules, Kolmogorov's axioms,[^kolmogorov1933] so the mathematics below does not depend on the interpretation.

This series follows two sources. Chan's *Introduction to Probability for Data Science*[^chan2026] is the source of much of the visual intuition, and Kumar's lecture notes on stochastic processes[^kumar2012] are the source of the rigorous treatment of convergence and Markov chains. For a fuller treatment, Bertsekas and Tsitsiklis[^bertsekas2008] and Blitzstein and Hwang[^blitzstein2019] are excellent first courses (both free online); Wasserman[^wasserman2004] and Murphy[^murphy2022] connect probability to statistics and machine learning.

### The frequency model

Suppose $A$ is a possible outcome of an experiment. Run the experiment $n$ times and count how often $A$ occurs, $n_A\le n$. A natural measure of how likely $A$ is, is its **relative frequency**

$$
\frac{n_A}{n}\in[0,1].
$$

It is only an estimate, which settles down as $n\to\infty$. Proving that it does settle down is the law of large numbers, one of the goals of this series ([[Sample Statistics#The law of large numbers|Sample Statistics]]). The relative frequency already has the properties we will demand of a probability:
- it lies between 0 and 1;
- the outcome "anything at all" has frequency $n/n=1$;
- if $A$ and $B$ cannot happen together, then $n_{A\cup B}=n_A+n_B$, so the frequencies add.

## Sets: the language of events

A probability is assigned to a **set** of outcomes, not to a number. "The die shows an even number" is the set $\{2,4,6\}$, and its probability is the size of that set relative to the whole, $3/6$. So before probability we need a little set theory.

A **set** is a collection of elements. We fix a **universal set** $\Omega$ containing everything under discussion; complements are taken relative to it. $A\subseteq B$ means every element of $A$ is in $B$, and $A=B$ means $A\subseteq B$ and $B\subseteq A$. The **empty set** $\emptyset$ has no elements and is a subset of every set.

The four basic operations are shown in Figure 1:
- **union** $A\cup B$: elements in $A$ **or** $B$ (or both);
- **intersection** $A\cap B$: elements in $A$ **and** $B$;
- **complement** $A^c=\Omega\setminus A$: elements **not** in $A$;
- **difference** $A\setminus B=A\cap B^c$: elements in $A$ but not in $B$. Unlike the others, it is not symmetric: $A\setminus B\ne B\setminus A$ in general.

![[Probability and Linear Algebra/images/probability-basics-set-operations.svg]]
*Figure 1. The four set operations as Venn diagrams. The rectangle is the universal set $\Omega$; the shaded region is the result.*

Union and intersection are commutative and associative, and each distributes over the other: $A\cap(B\cup C)=(A\cap B)\cup(A\cap C)$ and $A\cup(B\cap C)=(A\cup B)\cap(A\cup C)$. The most useful rules are **De Morgan's laws**,

$$
(A\cup B)^c=A^c\cap B^c,\qquad (A\cap B)^c=A^c\cup B^c,
$$

which hold for any number of sets, even infinitely many. In words: "not (A or B)" is "not A and not B". De Morgan is how we will pass between unions and intersections, for example to turn "at least one bad event happens" into "no bad event happens".

Two more definitions recur constantly:
- $A$ and $B$ are **disjoint** if $A\cap B=\emptyset$: they have nothing in common.
- Sets $B_1,\dots,B_K$ form a **partition** of $\Omega$ if they are pairwise disjoint and their union is $\Omega$. Think of a pie cut into slices: every outcome lies in exactly one slice. A partition splits any set $A$ into the disjoint pieces $A\cap B_1,\dots,A\cap B_K$.

**Infinite unions and intersections.** $\bigcup_{n=1}^{\infty}A_n$ contains the elements that belong to **at least one** $A_n$, and $\bigcap_{n=1}^{\infty}A_n$ those that belong to **every** $A_n$. Endpoints need care. For example,

$$
\bigcup_{n=1}^{\infty}\Big[-1,\;1-\tfrac1n\Big]=[-1,1),\qquad \bigcap_{n=1}^{\infty}\Big[0,\;1+\tfrac1n\Big)=[0,1].
$$

In the union, every point below 1 is eventually included, but 1 itself never is. In the intersection, every point above 1 is eventually excluded, but 1 is in every set. Infinite unions and intersections behave like limits, and [[#Continuity of probability|continuity of probability]] below makes that precise.

Kolmogorov's own dictionary translates between the two languages:

| Set language | Event language |
|---|---|
| $\Omega$ | the sure event: something happens |
| $\emptyset$ | the impossible event |
| $A\cup B$ | $A$ or $B$ happens |
| $A\cap B$ | $A$ and $B$ both happen |
| $A^c$ | $A$ does not happen |
| $A\cap B=\emptyset$ | $A$ and $B$ are incompatible |
| $A\subseteq B$ | $A$ implies $B$ |

## Probability spaces

### Sample space and events

In any experiment we fix a **universe of discourse**, the collection of all elementary outcomes we care about. This is the **sample space** $\Omega$, and its elements $\omega\in\Omega$ are the outcomes.
- One coin toss: $\Omega=\{\text{Heads},\text{Tails}\}\equiv\{1,0\}$, writing 1 for Heads.
- Three tosses: $\Omega=\{000,001,010,\dots,111\}$, eight outcomes.

An **event** $A$ is a subset of $\Omega$: the set of outcomes in which "$A$ happens". For three tosses, "exactly two heads" is the event $\{011,101,110\}$ (Figure 2).

![[Probability and Linear Algebra/images/probability-basics-coin-events.svg]]
*Figure 2. The sample space of three coin tosses. The eight outcomes are grouped into four events by the number of heads. Each group is the event $\{X=k\}$ for the random variable $X$ = number of heads (see [[Random Variables]]).*

### σ-algebras: which subsets are events

We collect the events in a family $\mathcal{F}$. We want to combine events freely ("not $A$", "$A$ or $B$", "at least one of $A_1,A_2,\dots$"), so $\mathcal{F}$ must be closed under these operations. Formally, $\mathcal{F}$ is a **σ-algebra** on $\Omega$:
1. $\Omega\in\mathcal{F}$;
2. $A\in\mathcal{F}\Rightarrow A^c\in\mathcal{F}$ (closed under complements);
3. $A_1,A_2,\ldots\in\mathcal{F}\Rightarrow\bigcup_{i=1}^{\infty}A_i\in\mathcal{F}$ (closed under countable unions).

Consequences:
- $\emptyset=\Omega^c\in\mathcal{F}$.
- Finite unions are included: pad $A\cup B$ with copies of $\emptyset$, as $A\cup B\cup\emptyset\cup\emptyset\cup\cdots$.
- Countable intersections are included, by De Morgan: $\bigcap_iA_i=\big(\bigcup_iA_i^c\big)^c$.

> [!note] Field vs σ-field
> A family closed under complements and **finite** unions is called a **field** (or algebra). Condition 3, closure under **countable** unions, is what makes it a **σ-field** (σ-algebra). The difference is real. On $\Omega=\mathbb{N}$, the sets that are finite or have a finite complement form a field. But the even numbers are a countable union of singletons from that field, and they are neither finite nor co-finite. Countable unions are needed for limits, such as "the coin eventually lands heads".

How large should $\mathcal{F}$ be?
- For a **finite or countable** $\Omega$, we can take $\mathcal{F}$ to be **all** subsets of $\Omega$ (the power set), and we always will.
- For $\Omega=\mathbb{R}$, "all subsets" is too many: there are sets so irregular that no sensible length can be assigned to them (the Vitali sets). Instead we use the **Borel σ-algebra**, the smallest σ-algebra containing all intervals. It includes every set we will ever meet in practice.

The pair $(\Omega,\mathcal{F})$ is a **measurable space**.

### Kolmogorov's axioms

A set function $\mathbb{P}:\mathcal{F}\to[0,1]$ is a **probability** if:
1. **Normalized:** $\mathbb{P}(\Omega)=1$.
2. **Non-negative:** $\mathbb{P}(A)\ge0$ for all $A\in\mathcal{F}$.
3. **σ-additive:** for every countable collection $A_1,A_2,\ldots\in\mathcal{F}$ of **pairwise disjoint** events ($A_i\cap A_j=\emptyset$ for $i\ne j$),

$$
\mathbb{P}\Big(\bigcup_{i=1}^{\infty}A_i\Big)=\sum_{i=1}^{\infty}\mathbb{P}(A_i).
$$

The triple $(\Omega,\mathcal{F},\mathbb{P})$ is a **probability space**. Each axiom is a property of relative frequencies (see above). Axiom 3 is the frequency rule $n_{A\cup B}/n=n_A/n+n_B/n$, extended to countably many events.

### Probability measures the size of a set

The axioms say *what rules* a probability obeys, not how to compute one. A useful picture is that $\mathbb{P}$ **measures the size of a set**, the slogan around which Chan organizes his treatment.[^chan2026] Depending on the sample space, "size" means one of three things (Figure 3):
- **Counting.** With finitely many equally likely outcomes, the size of $A$ is the fraction of outcomes it contains: $\mathbb{P}(A)=|A|/|\Omega|$.
- **Weighing.** If the outcomes have different weights $p(\omega)\ge0$ summing to 1, the size of $A$ is its total weight, $\mathbb{P}(A)=\sum_{\omega\in A}p(\omega)$.
- **Measuring.** On a continuum, the weight is spread out as a density $f$, and the size of $A$ is the area $\mathbb{P}(A)=\int_Af(x)\,dx$.

The same event can have different sizes under different weightings. "Even" has probability $1/2$ for a fair die and $0.65$ for the loaded one in Figure 3b. The sample space and the events are the same; only $\mathbb{P}$ changed. This is why a probability model is the whole triple $(\Omega,\mathcal{F},\mathbb{P})$, and why most of statistics is about learning $\mathbb{P}$ from data.

![[Probability and Linear Algebra/images/probability-basics-size-of-a-set.svg]]
*Figure 3. Three ways of measuring the size of the same kind of event. (a) Counting: for a fair die, "even" (orange) is 3 of 6 equally likely faces. (b) Weighing: for a loaded die, "even" has the total weight of its faces, 0.65. (c) Measuring: for a continuous quantity with density $f(x)=xe^{-x}$, the probability of an interval is the area above it.*

> [!warning] Probability zero does not mean impossible
> Pick a point uniformly at random from $[0,1]$. A single point is an interval of length zero, so $\mathbb{P}(\{0.5\})=0$, and the same holds for every other point. Yet *some* point is always picked. Probability-zero events happen all the time in continuous spaces; they are just negligible in size. An event of probability 1 is said to happen **almost surely** (a.s.). The distinction matters for the strong law of large numbers and for convergence "almost surely" in [[Sample Statistics#Four ways a sequence can converge|Sample Statistics]].

### Equally likely outcomes and counting

When all outcomes are equally likely, computing a probability means counting, and counting can be surprisingly subtle. The tools are in the aside below.

> [!note]- Aside: counting rules
> - **Multiplication principle.** A task done in $k$ stages with $n_1,\dots,n_k$ choices at each stage can be done in $n_1n_2\cdots n_k$ ways. Drawing $k$ items **with replacement** from $n$, keeping the order, gives $n^k$ sequences.
> - **Permutations.** Drawing $k$ items from $n$ **without replacement**, keeping the order: $n(n-1)\cdots(n-k+1)=\frac{n!}{(n-k)!}$ ways.
> - **Combinations.** Without replacement and ignoring the order: each unordered set of $k$ items appears in $k!$ orders, so there are $\binom{n}{k}=\frac{n!}{k!\,(n-k)!}$ sets. For example, 3 of 5 items: 60 ordered draws, but $60/3!=10$ subsets.
> - The binomial coefficient $\binom{n}{k}$ reappears in [[Random Variables#Bernoulli and binomial|the binomial distribution]] as the number of orders in which $k$ successes can occur among $n$ trials.
>
> Most counting mistakes come from mixing up "ordered vs unordered" or "with vs without replacement". Decide both before counting.

> [!example] Example: the birthday problem
> In a room of $n$ people, how likely is it that two share a birthday? Assume 365 equally likely birthdays and no twins. It is easier to count the complement, "all birthdays different": the first person can have any birthday, the second must avoid one day, the third two days, and so on. So
>
> $$
> \mathbb{P}(\text{some birthday shared})=1-\prod_{k=0}^{n-1}\Big(1-\frac{k}{365}\Big).
> $$
>
> This passes 50% at only $n=23$ people, and reaches 97% at $n=50$ (Figure 4). The intuition is that a match can occur between *any pair*, and 23 people form $\binom{23}{2}=253$ pairs. Each pair matches with probability $1/365$, so we expect about $253/365\approx0.7$ matching pairs. Using $1-x\approx e^{-x}$, the product is about $\exp\big(-\sum_kk/365\big)=\exp\big(-n(n-1)/730\big)$, the dashed curve in Figure 4.

![[Probability and Linear Algebra/images/probability-basics-birthday.svg]]
*Figure 4. The probability that at least two of $n$ people share a birthday (solid), and the approximation $1-e^{-n(n-1)/730}$ (dashed). The number of pairs grows like $n^2/2$, which is why the curve rises so quickly.*

The same $n^2$ effect governs collisions in hash tables and in the hashing trick for features: with $m$ buckets, collisions become likely after only about $\sqrt{m}$ items.

> [!example]- Code: checking the birthday problem by simulation
> ```python
> import numpy as np
> rng = np.random.default_rng(0)
>
> def p_shared_exact(n):
>     return 1 - np.prod((365 - np.arange(n)) / 365)
>
> def p_shared_sim(n, trials=100_000):
>     # each row is a room of n people; sort, then look for equal neighbours
>     b = np.sort(rng.integers(365, size=(trials, n)), axis=1)
>     return np.mean((np.diff(b, axis=1) == 0).any(axis=1))
>
> for n in (10, 23, 50):
>     print(n, round(p_shared_exact(n), 4), round(p_shared_sim(n), 4))
> # 10 0.1169 0.1157
> # 23 0.5073 0.5082
> # 50 0.9704 0.9706
> ```

## First consequences

All the rules below follow from the three axioms. Their order matters, because each proof uses only the rules before it.

**1. The empty event has probability zero: $\mathbb{P}(\emptyset)=0$.** Take $A_i=\emptyset$ for every $i$. These are pairwise disjoint and their union is $\emptyset$, so σ-additivity gives $\mathbb{P}(\emptyset)=\sum_{i=1}^{\infty}\mathbb{P}(\emptyset)$. A non-negative number equal to an infinite sum of copies of itself must be 0.

**2. Finite additivity.** For disjoint $A$ and $B$, apply σ-additivity to $A,B,\emptyset,\emptyset,\ldots$. The tail contributes $\mathbb{P}(\emptyset)+\mathbb{P}(\emptyset)+\cdots=0$, by rule 1:

$$
\mathbb{P}(A\cup B)=\mathbb{P}(A)+\mathbb{P}(B).
$$

**3. Complements: $\mathbb{P}(A^c)=1-\mathbb{P}(A)$.** $A$ and $A^c$ are disjoint with union $\Omega$, so $\mathbb{P}(A)+\mathbb{P}(A^c)=\mathbb{P}(\Omega)=1$.

**4. Monotonicity: $A\subseteq B\Rightarrow\mathbb{P}(A)\le\mathbb{P}(B)$.** Write $B=A\cup(B\setminus A)$, a disjoint union because $A\subseteq B$. Then $\mathbb{P}(B)=\mathbb{P}(A)+\mathbb{P}(B\setminus A)\ge\mathbb{P}(A)$, since the second term is non-negative.

**5. Inclusion–exclusion.** For any $A$ and $B$,

$$
\mathbb{P}(A\cup B)=\mathbb{P}(A)+\mathbb{P}(B)-\mathbb{P}(A\cap B).
$$

Split into disjoint pieces: $A\cup B=A\cup(B\setminus A)$ and $B=(B\setminus A)\cup(A\cap B)$. Subtracting the two finite-additivity equations gives the result. Adding $\mathbb{P}(A)$ and $\mathbb{P}(B)$ counts the overlap twice, so we subtract it once.

> [!warning] A note on the order of proofs
> It is tempting to prove $\mathbb{P}(\emptyset)=0$ from $\mathbb{P}(\Omega\cup\emptyset)=\mathbb{P}(\Omega)+\mathbb{P}(\emptyset)$. That uses finite additivity, which is not an axiom. Deriving it from σ-additivity by padding with copies of $\emptyset$ already needs $\mathbb{P}(\emptyset)=0$ for the padding to contribute nothing. The argument in rule 1 avoids the circle.

### The union bound

**6. Union bound (Boole's inequality).** For any events $A_1,A_2,\ldots\in\mathcal{F}$, disjoint or not,

$$
\mathbb{P}\Big(\bigcup_{i=1}^{\infty}A_i\Big)\le\sum_{i=1}^{\infty}\mathbb{P}(A_i).
\tag{1}
$$

The idea is to make the events disjoint without changing their union, and then use σ-additivity (Figure 5b). Define

$$
E_1=A_1,\qquad E_k=A_k\cap\Big(\bigcup_{i=1}^{k-1}A_i\Big)^c\quad(k\ge2),
$$

so $E_k$ keeps the outcomes of $A_k$ that no earlier event has already claimed.

> [!proof]- Proof of the union bound
> We need three facts about the $E_k$.
> - **They are pairwise disjoint.** Take $i<j$. Then $E_i\subseteq A_i$, while $E_j\subseteq A_i^c$ because $E_j$ removes everything in $A_1\cup\cdots\cup A_{j-1}$, which includes $A_i$. So $E_i\cap E_j=\emptyset$.
> - **They have the same union.** Each $E_k\subseteq A_k$, so $\bigcup_kE_k\subseteq\bigcup_kA_k$. Conversely, if $\omega\in\bigcup_kA_k$, let $k$ be the smallest index with $\omega\in A_k$. Then $\omega$ is in no earlier $A_i$, so $\omega\in E_k$.
> - **Each is smaller:** $E_k\subseteq A_k$, so $\mathbb{P}(E_k)\le\mathbb{P}(A_k)$ by monotonicity.
>
> Putting these together, with σ-additivity in the middle step:
>
> $$
> \mathbb{P}\Big(\bigcup_kA_k\Big)=\mathbb{P}\Big(\bigcup_kE_k\Big)=\sum_k\mathbb{P}(E_k)\le\sum_k\mathbb{P}(A_k).
> $$

The union bound is crude, since it ignores all overlaps, but it needs no assumptions at all. That makes it the standard way to say "with high probability, **none** of many bad events happens": bound each one, then add. The [[Multi-Armed Bandits#UCB achieves logarithmic regret|UCB regret proof]] and generalization bounds both work this way. We combine it with concentration inequalities in [[Sample Statistics#Concentration and the union bound|Sample Statistics]].

### Continuity of probability

**7. Continuity.** If $E_1\subseteq E_2\subseteq\cdots$ is an increasing sequence of events with union $E=\bigcup_kE_k$ (written $E_k\uparrow E$), then

$$
\lim_{k\to\infty}\mathbb{P}(E_k)=\mathbb{P}(E).
$$

Any countable union is the limit of finite ones in this way: $E_k=\bigcup_{i=1}^kA_i$ increases to $\bigcup_{i=1}^{\infty}A_i$ (Figure 5a). Continuity lets us compute the probability of an infinite union as a limit of probabilities of finite ones.

> [!proof]- Proof
> Peel off the new part at each step: $F_1=E_1$ and $F_k=E_k\setminus E_{k-1}$. The $F_k$ are disjoint, $E_n=\bigcup_{k\le n}F_k$ and $E=\bigcup_kF_k$. By σ-additivity,
>
> $$
> \mathbb{P}(E)=\sum_{k=1}^{\infty}\mathbb{P}(F_k)=\lim_{n\to\infty}\sum_{k=1}^{n}\mathbb{P}(F_k)=\lim_{n\to\infty}\mathbb{P}(E_n).
> $$
>
> Taking complements gives the decreasing version: $E_k\downarrow E$ implies $\mathbb{P}(E_k)\to\mathbb{P}(E)$.

![[Probability and Linear Algebra/images/probability-basics-unions.svg]]
*Figure 5. (a) Continuity: the finite unions $E_k=A_1\cup\cdots\cup A_k$ grow towards the countable union, and their probabilities converge to its probability. (b) The disjointification behind the union bound: $E_1=A_1$, $E_2$ is the part of $A_2$ outside $A_1$, and $E_3$ is the part of $A_3$ outside both. Together they cover $A_1\cup A_2\cup A_3$ exactly once.*


## Conditional probability and independence

### Conditioning on evidence

Suppose we learn that $B$ happened. Among the $n_B$ runs in which $B$ occurred, $A$ also occurred in $n_{A\cap B}$ of them, so the relevant frequency is $n_{A\cap B}/n_B=(n_{A\cap B}/n)/(n_B/n)$. This motivates the definition, for $\mathbb{P}(B)>0$:

$$
\mathbb{P}(A\mid B)=\frac{\mathbb{P}(A\cap B)}{\mathbb{P}(B)}.
$$

Conditioning **shrinks the universe to $B$** and renormalizes. Everything outside $B$ is now impossible, and the part of $A$ inside $B$ is measured relative to $B$ instead of $\Omega$. Figure 6a shows this for two dice. Knowing that the sum is at least 9 shrinks the 36 outcomes to 10, of which 4 have a 6 on the first die, so the probability of that 6 rises from $1/6$ to $4/10$. $\mathbb{P}(\cdot\mid B)$ satisfies all three axioms, so it is itself a probability, and every rule in this lecture holds for it too.

The order matters: $\mathbb{P}(A\mid B)\ne\mathbb{P}(B\mid A)$ in general. In Figure 6a, $\mathbb{P}(A\mid B)=4/10$ but $\mathbb{P}(B\mid A)=4/6$. Confusing the two is the most common error in probabilistic reasoning; the test example in the next section is the classic case.

Two rules follow immediately from the definition:
- **Chain rule:** $\mathbb{P}(A\cap B)=\mathbb{P}(A\mid B)\,\mathbb{P}(B)$, and more generally $\mathbb{P}(A_1\cap\cdots\cap A_n)=\prod_{k=1}^{n}\mathbb{P}(A_k\mid A_1\cap\cdots\cap A_{k-1})$. The trajectory probability in [[Policy Gradients#The objective and its gradient|Policy Gradients]] is exactly this product.
- **Law of total probability:** if $B_1,\ldots,B_K$ partition $\Omega$, then $A$ splits into the disjoint pieces $A\cap B_k$, and

$$
\mathbb{P}(A)=\sum_{k=1}^{K}\mathbb{P}(A\mid B_k)\,\mathbb{P}(B_k).
\tag{2}
$$

The law of total probability splits a hard problem into cases, each of which is easy once we know which case we are in.

![[Probability and Linear Algebra/images/probability-basics-conditioning.svg]]
*Figure 6. (a) Conditioning shrinks the universe. Of the 36 outcomes of two dice, 10 have a sum of at least 9 (shaded). Given that, 4 of the 10 have a 6 on the first die (circled), so $\mathbb{P}(A\mid B)=0.4$, up from $\mathbb{P}(A)=1/6$. (b) The law of total probability as areas. The unit square is cut into strips of width $\mathbb{P}(B_k)$; in each strip the shaded height is $\mathbb{P}(A\mid B_k)$. $\mathbb{P}(A)$ is the total shaded area, and the posterior $\mathbb{P}(B_k\mid A)$ is the fraction of the shaded area lying in strip $k$.*

In Figure 6b each strip's shaded area is one term $\mathbb{P}(A\mid B_k)\,\mathbb{P}(B_k)$ of (2), so $\mathbb{P}(A)=0.15+0.18+0.16=0.49$.

### Bayes' theorem

The chain rule can be applied in either order: $\mathbb{P}(B\mid A)\,\mathbb{P}(A)=\mathbb{P}(A\cap B)=\mathbb{P}(A\mid B)\,\mathbb{P}(B)$. Dividing by $\mathbb{P}(A)$, and expanding $\mathbb{P}(A)$ by the law of total probability, gives **Bayes' theorem**:

$$
\mathbb{P}(B_k\mid A)=\frac{\mathbb{P}(A\mid B_k)\,\mathbb{P}(B_k)}{\mathbb{P}(A)}=\frac{\mathbb{P}(A\mid B_k)\,\mathbb{P}(B_k)}{\sum_j\mathbb{P}(A\mid B_j)\,\mathbb{P}(B_j)}.
\tag{3}
$$

Read the $B_k$ as competing explanations (hypotheses, causes, classes) and $A$ as the evidence we observed:
- the **prior** $\mathbb{P}(B_k)$ is how plausible explanation $k$ was before the evidence;
- the **likelihood** $\mathbb{P}(A\mid B_k)$ is how well explanation $k$ predicts the evidence;
- the **posterior** $\mathbb{P}(B_k\mid A)$ is how plausible it is after the evidence;
- the denominator, the **evidence** $\mathbb{P}(A)$, is the same for every $k$ and just makes the posteriors sum to 1.

So *posterior ∝ likelihood × prior*. Bayes' theorem reverses the direction of conditioning: we usually know how causes produce evidence, $\mathbb{P}(A\mid B_k)$, and want to reason back from the evidence to the causes, $\mathbb{P}(B_k\mid A)$. In Figure 6b it asks what share of the shaded area lies in a given strip. For $B_1$ the share is $0.15/0.49\approx0.31$. $B_1$ had the largest prior, but after observing $A$ it is the *least* likely explanation, because $A$ is rare under it. [[Discriminant Analysis]] classifies with exactly (3), the classes playing the role of the $B_k$.

> [!example] Example: a rare condition and a good test
> A condition affects 1% of a population. A test detects it 95% of the time when present (its **sensitivity**), and gives a false positive 10% of the time when absent (so its **specificity** is 90%). You test positive. What is the probability that you have the condition? Let $D$ be "has the condition" and $+$ be "tests positive":
>
> $$
> \begin{aligned}
> \mathbb{P}(D\mid+)&=\frac{\mathbb{P}(+\mid D)\,\mathbb{P}(D)}{\mathbb{P}(+\mid D)\,\mathbb{P}(D)+\mathbb{P}(+\mid D^c)\,\mathbb{P}(D^c)}\\
> &=\frac{0.95\times0.01}{0.95\times0.01+0.10\times0.99}=\frac{0.0095}{0.1085}\approx0.088.
> \end{aligned}
> $$
>
> Only about 9%, although the test is "95% accurate". Picture 1000 people (Figure 7): about 10 have the condition, and about 9 of them test positive. Of the 990 healthy people, about 99 test positive. A positive result puts you among roughly $9+99=108$ people, of whom only 9 are ill. The false positives win because there are so many more healthy people to produce them.
>
> Ignoring the prior is the **base-rate fallacy**. It confuses the probability that an ill person tests positive, $\mathbb{P}(+\mid D)=0.95$, with the probability that a positive tester is ill, $\mathbb{P}(D\mid+)=0.088$.

![[Probability and Linear Algebra/images/probability-basics-base-rate.svg]]
*Figure 7. The base-rate example as 1000 people (counts rounded to whole people). The positives are the 9 ill people who test positive (blue) and the 99 healthy people who test positive (orange). Only about one positive in twelve is ill.*

> [!example] Example: an additional test updates the belief
> After the first positive test, your probability of having the condition is no longer 1% but $0.0876$. You take the test again, and it is positive again. Assume the two results are independent given your true status: the test's errors are random, not caused by something about you. Then Bayes' theorem applies again, with **the old posterior as the new prior**:
>
> $$
> \begin{aligned}
> \mathbb{P}(D\mid+,+)&=\frac{\mathbb{P}(+\mid D)\,\mathbb{P}(D\mid+)}{\mathbb{P}(+\mid D)\,\mathbb{P}(D\mid+)+\mathbb{P}(+\mid D^c)\,\mathbb{P}(D^c\mid+)}\\
> &=\frac{0.95\times0.0876}{0.95\times0.0876+0.10\times0.9124}=\frac{0.0832}{0.1744}\approx0.48.
> \end{aligned}
> $$
>
> In the picture of Figure 7, retest only the 108 people who tested positive. About $0.95\times9\approx8.6$ of the 9 ill people test positive again, but only about $0.10\times99\approx9.9$ of the 99 healthy ones do. The false positives have been cut tenfold, the true positives barely at all, and now roughly half of those who are still positive are ill. A third positive test raises the probability to $0.90$.
>
> Updating one test at a time gives the same answer as treating both tests as a single piece of evidence. With $\mathbb{P}(+,+\mid D)=0.95^2$ and $\mathbb{P}(+,+\mid D^c)=0.10^2$, one application of (3) from the original prior gives $\frac{0.95^2\times0.01}{0.95^2\times0.01+0.10^2\times0.99}\approx0.48$. Bayesian updating is consistent: the order and grouping of the evidence do not matter, only the evidence itself.

**The odds form.** Bayes' theorem is easiest to use, and to remember, in terms of odds. Dividing (3) for $D$ by (3) for $D^c$ cancels the evidence term:

$$
\underbrace{\frac{\mathbb{P}(D\mid+)}{\mathbb{P}(D^c\mid+)}}_{\text{posterior odds}}=\underbrace{\frac{\mathbb{P}(+\mid D)}{\mathbb{P}(+\mid D^c)}}_{\text{likelihood ratio}}\times\underbrace{\frac{\mathbb{P}(D)}{\mathbb{P}(D^c)}}_{\text{prior odds}}.
\tag{4}
$$

The evidence multiplies the odds by the **likelihood ratio**, which measures how much more often the evidence occurs under one explanation than the other. Here it is $0.95/0.10=9.5$. The prior odds are $1:99$, so the posterior odds are $9.5:99$, about $1:10.4$, which is the probability $0.088$ again. A positive test makes the condition 9.5 times more likely *relative to not having it*, but starting from $1:99$ that is still not much.

This explains when a positive test does mean something (Figure 8):
- **Prevalence matters.** The same test is far more informative in a clinic, where many patients are ill, than in a population screen. Being tested *because* you have symptoms raises your prior above the population rate.
- **The false-positive rate matters most for rare conditions.** Cutting it from 10% to 1% raises the likelihood ratio to 95, and $\mathbb{P}(D\mid+)$ from 0.088 to 0.49.
- **A second positive test helps a lot.** Yesterday's posterior is today's prior, as in the example above. In odds form the update is just a multiplication: if the tests' errors are independent given the true status, each positive multiplies the odds by another 9.5, and the probability goes from 0.01 to 0.088, then 0.48, then 0.90. If the errors are correlated (the same lab mistake, or a cross-reacting antibody), repeating the same test helps much less. This is why a positive screening test is usually confirmed with a *different* test.

![[Probability and Linear Algebra/images/probability-basics-bayes-tests.svg]]
*Figure 8. When does a positive test mean you are ill? (a) $\mathbb{P}(D\mid+)$ against the prevalence, for a test with 95% sensitivity and a 10% or 1% false-positive rate. At 1% prevalence (dot), a positive from the first test means only an 8.8% chance of illness. (b) Repeating a test whose errors are independent: each positive multiplies the odds by the likelihood ratio 9.5.*

> [!example] Example: the Monty Hall problem
> A prize is behind one of three doors, equally likely. You pick door 1. The host, **who knows where the prize is**, always opens a door you did not pick and that hides no prize, and offers you the chance to switch. He opens door 3. Should you switch to door 2?[^selvin1975]
>
> Most people say it does not matter, as two doors remain. Write out the conditionals instead. Let $C_k$ be "the prize is behind door $k$" and $H_3$ be "the host opens door 3". The host's rule gives the likelihoods:
> - $\mathbb{P}(H_3\mid C_1)=1/2$: the host may open door 2 or 3;
> - $\mathbb{P}(H_3\mid C_2)=1$: door 3 is his only choice;
> - $\mathbb{P}(H_3\mid C_3)=0$: he never reveals the prize.
>
> By total probability, $\mathbb{P}(H_3)=\tfrac13\big(\tfrac12+1+0\big)=\tfrac12$. By Bayes' rule,
>
> $$
> \mathbb{P}(C_1\mid H_3)=\frac{\tfrac12\cdot\tfrac13}{\tfrac12}=\frac13,\qquad \mathbb{P}(C_2\mid H_3)=\frac{1\cdot\tfrac13}{\tfrac12}=\frac23.
> $$
>
> Switching doubles your chance of winning. The intuition is that your first pick is right only $1/3$ of the time. When it is wrong, the host is *forced* to show you where the prize is not, so switching wins.
>
> The host's rule is part of the model. If the host instead opened one of the other two doors **at random**, and it happened to hide no prize, the likelihoods would be $\tfrac12,\tfrac12,0$ and both remaining doors would have posterior $\tfrac12$. The same observation can mean different things under different protocols.

> [!example]- Code: simulating Monty Hall
> ```python
> import numpy as np
> rng = np.random.default_rng(0)
>
> def play(trials=100_000):
>     stay = switch = 0
>     for _ in range(trials):
>         car, pick = rng.integers(3), rng.integers(3)
>         # the host opens a door that hides a goat and was not picked
>         opened = rng.choice([d for d in range(3) if d not in (pick, car)])
>         other = next(d for d in range(3) if d not in (pick, opened))
>         stay += pick == car
>         switch += other == car
>     return stay / trials, switch / trials
>
> print(play())   # about (0.333, 0.667)
> ```

### Independence

Events $A$ and $B$ are **independent** if

$$
\mathbb{P}(A\cap B)=\mathbb{P}(A)\,\mathbb{P}(B),
$$

equivalently $\mathbb{P}(A\mid B)=\mathbb{P}(A)$ when $\mathbb{P}(B)>0$: learning $B$ tells us nothing about $A$. In the area picture of Figure 6b, $A$ is independent of every $B_k$ exactly when the shaded height is the same in every strip. Then the shaded band is flat, and knowing which strip we are in changes nothing.

> [!warning] Disjoint is not independent
> Disjoint events are as *dependent* as events can be: if $A\cap B=\emptyset$, then learning that $B$ happened tells us that $A$ did **not**. Formally, $\mathbb{P}(A\cap B)=0$, which equals $\mathbb{P}(A)\,\mathbb{P}(B)$ only if one of them has probability zero.

For more than two events, **mutual** independence requires the product rule for **every** sub-collection. Pairwise independence is weaker. Toss two fair coins and let $A$ = "first is heads", $B$ = "second is heads", $C$ = "both tosses agree". Any two of these are independent: for example $\mathbb{P}(A\cap C)=\mathbb{P}(HH)=\tfrac14=\mathbb{P}(A)\,\mathbb{P}(C)$. But all three are not, since $\mathbb{P}(A\cap B\cap C)=\tfrac14\ne\tfrac18$: $A$ and $B$ together determine $C$.

Independence is usually an **assumption** we build into a model ("the tosses do not influence each other"), not something we derive. Repeated independent trials are the setting for the binomial distribution, the law of large numbers and the central limit theorem.

$A$ and $B$ are **conditionally independent** given $C$ if $\mathbb{P}(A\cap B\mid C)=\mathbb{P}(A\mid C)\,\mathbb{P}(B\mid C)$. Neither kind of independence implies the other. For example, two tosses of a coin whose bias is unknown are conditionally independent given the bias, but not independent: a head on the first toss makes a heads-biased coin more likely, and so makes a head on the second toss more likely. The Markov property in [[Markov Chains]] and [[Introduction to Reinforcement Learning|RL]] is a conditional independence: the future is independent of the past given the present. Naive Bayes ([[Discriminant Analysis]]) assumes features are conditionally independent given the class.

## Key takeaways

- Events are sets. Union, intersection and complement are "or", "and" and "not", and De Morgan's laws convert between them.
- A probability space is $(\Omega,\mathcal{F},\mathbb{P})$. $\mathbb{P}$ measures the size of an event, by counting, weighing or integrating, and is pinned down by three axioms.
- The consequences used everywhere later: complements, monotonicity, inclusion–exclusion, the union bound (no independence needed) and continuity of probability.
- Probability zero is not impossibility in a continuous space; "almost surely" means "except on a set of probability zero".
- Conditioning shrinks the universe and renormalizes. Total probability splits a problem into cases; Bayes' rule reverses the conditioning. Write the conditionals out, and never forget the prior.
- Independence is a product rule. Disjoint events are dependent, pairwise independence is weaker than mutual independence, and conditional independence is different again.

## Where this shows up next

- [[Random Variables]] attaches numbers to outcomes. Their distributions (PMFs, PDFs, CDFs) are the "weights" and "densities" of Figure 3, and the law of total probability becomes the law of total expectation.
- [[Joint Distributions]] conditions one random variable on another, which is Bayes' rule for random variables.
- [[Sample Statistics]] proves that the relative frequency $n_A/n$ converges to $\mathbb{P}(A)$, combining the union bound with concentration inequalities.
- [[Markov Chains]] are built entirely from the chain rule and one conditional independence assumption.
- [[Discriminant Analysis]] is Bayes' rule (3) with Gaussian likelihoods, and [[Multi-Armed Bandits]] uses the union bound in its regret proof.

<!-- References: footnote definitions below render as the numbered "References" list at the bottom of the page -->

[^jaynes2003]: Jaynes, E. T. (2003). *Probability Theory: The Logic of Science*. Cambridge University Press. [doi:10.1017/CBO9780511790423](https://doi.org/10.1017/CBO9780511790423)
[^kolmogorov1933]: Kolmogorov, A. N. (1933). *Grundbegriffe der Wahrscheinlichkeitsrechnung*. Springer. English translation: *Foundations of the Theory of Probability* (1950). [doi:10.1007/978-3-642-49888-6](https://doi.org/10.1007/978-3-642-49888-6)
[^chan2026]: Chan, S. H. (2026). *Introduction to Probability for Data Science* (2nd ed.), Chapters 2–3. Michigan Publishing. [book website, with free eBook, slides and code](https://probability4datascience.com)
[^kumar2012]: Kumar, A. (2012). *Discrete Event Stochastic Processes: Lecture Notes for an Engineering Curriculum*, Chapter 1. Indian Institute of Science. [PDF](https://ece.iisc.ac.in/~anurag/books/anurag/spqt.pdf)
[^bertsekas2008]: Bertsekas, D. P., & Tsitsiklis, J. N. (2008). *Introduction to Probability* (2nd ed.). Athena Scientific. Lectures: [MIT OCW 6.041](https://ocw.mit.edu/courses/6-041-probabilistic-systems-analysis-and-applied-probability-fall-2010/)
[^blitzstein2019]: Blitzstein, J. K., & Hwang, J. (2019). *Introduction to Probability* (2nd ed.). CRC Press. [Free online edition](https://probabilitybook.net)
[^wasserman2004]: Wasserman, L. (2004). *All of Statistics: A Concise Course in Statistical Inference*. Springer. [doi:10.1007/978-0-387-21736-9](https://doi.org/10.1007/978-0-387-21736-9)
[^murphy2022]: Murphy, K. P. (2022). *Probabilistic Machine Learning: An Introduction*, Chapters 2–3. MIT Press. [book page](https://probml.github.io/pml-book/book1.html)
[^selvin1975]: Selvin, S. (1975). *A problem in probability* (letter to the editor). The American Statistician, 29(1), 67. [doi:10.1080/00031305.1975.10479121](https://doi.org/10.1080/00031305.1975.10479121)
