---
title: Mathematical Foundations
next: "[[Multivariable Calculus]]"
---

The mathematics needed for the rest of the series: calculus, linear algebra and optimization first (they need no probability), then probability. Each lecture lists its prerequisites at the top, and every idea comes with small examples worked by hand.

## Calculus, linear algebra and optimization

Multivariable calculus is where most students struggle, so the series starts there. The linear algebra lectures follow the story of Gilbert Strang's mini-series *A Vision of Linear Algebra* (MIT OpenCourseWare): start from the columns of a matrix, and end with the factorizations that organize the whole subject. Read in order:

1. [[Multivariable Calculus]]: a single-variable recap, then gradients, directional derivatives, Jacobians and the chain rule, Hessians and second-order Taylor, classifying critical points, matrix calculus, convexity, Lagrange multipliers, and change of variables in multiple integrals.
2. [[Column Space and Rank]]: $\mathbf{A}\mathbf{x}$ as a combination of columns, the column space, independence, basis and rank, $\mathbf{A}=\mathbf{C}\mathbf{R}$ and why column rank equals row rank, and matrix multiplication as columns times rows.
3. [[Elimination and the Nullspace]]: elimination as $\mathbf{A}=\mathbf{L}\mathbf{U}$, the reduced row echelon form, special solutions of $\mathbf{A}\mathbf{x}=\mathbf{0}$, the counting theorem, and the complete solution of $\mathbf{A}\mathbf{x}=\mathbf{b}$.
4. [[The Four Fundamental Subspaces]]: the big picture of linear algebra: row space, column space and the two nullspaces, their dimensions, and their orthogonality.
5. [[Orthogonality and Least Squares]]: orthogonal matrices, projections, least squares and the normal equations, Gram–Schmidt and $\mathbf{A}=\mathbf{Q}\mathbf{R}$.
6. [[Eigenvalues and Eigenvectors]]: diagonalization and matrix powers, symmetric matrices and the spectral theorem, positive definite matrices, and the Rayleigh quotient.
7. [[First-Order Optimization]]: gradient descent and its analysis on quadratics (the condition number), convergence rates, stochastic and mini-batch gradient descent, momentum and adaptive methods such as Adam.
8. [[Five Factorizations of a Matrix]]: $\mathbf{C}\mathbf{R}$, $\mathbf{L}\mathbf{U}$, $\mathbf{Q}\mathbf{R}$, $\mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^\top$ and $\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$ side by side (the singular value decomposition itself is covered in [[Matrix Factorization and Dimensionality Reduction]]), which one to use when, and the step from linear algebra to deep learning.

## Probability

A five-part series, written for intuition first, with figures, worked examples and tested code. Read in order:

1. [[Probability Basics]]: sets as the language of events; probability spaces, σ-algebras and Kolmogorov's axioms; probability as the size of a set; counting and the birthday problem; the union bound and continuity of probability; conditioning, the law of total probability, Bayes' theorem (why a positive test may not mean you are ill), Monty Hall, and independence.
2. [[Random Variables]]: PMFs, PDFs and CDFs; expectation, LOTUS, mean vs median; variance; the law of total expectation and first-step analysis; Markov's and Chebyshev's inequalities; Bernoulli → binomial → Poisson, memoryless geometric and exponential waiting times; survival and hazard functions; changes of variables; inverse-CDF and rejection sampling; the Poisson process, and self-exciting Hawkes processes simulated by thinning.
3. [[Joint Distributions]]: joint, marginal and conditional distributions; linearity of expectation; covariance and correlation (uncorrelated is not independent); conditional expectation as the best predictor, the tower rule and total variance; sums and convolution; covariance matrices and the multivariate Gaussian.
4. [[Sample Statistics]]: the sample mean and variance; moment-generating and characteristic functions; Jensen, Chernoff and Hoeffding, and the union bound; four modes of convergence and Borel–Cantelli; the weak and strong laws of large numbers; Monte Carlo; the central limit theorem.
5. [[Markov Chains]]: the Markov property and transition matrices; first-step analysis and gambler's ruin; recurrence, transience and Pólya's random walks; stationary distributions, detailed balance and convergence; mixing and the spectral gap; PageRank; Metropolis–Hastings and Gibbs sampling; continuous-time chains (rate matrices, $e^{t\mathbf{Q}}$, Gillespie), Brownian motion, and the Markov chains behind diffusion models and state space models.
