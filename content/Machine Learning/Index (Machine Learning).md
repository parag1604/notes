---
title: Index (Machine Learning)
---

Machine learning built up from first principles: supervised learning, then unsupervised learning, from matrix factorization and clustering to deep generative models. Each lecture lists its prerequisites at the top; the probability they use is in the [[Index (Probability and Linear Algebra)|Probability and Linear Algebra]] series.

## How machine learning works in practice

A machine-learning model is a function fitted to data, so that it can make predictions on data it has not seen. Whatever the model, a practitioner goes through the same loop:

![[Machine Learning/images/ml-workflow.svg]]
*The machine-learning workflow. Evaluation sends us back to tune the model or to collect better data; after deployment, monitoring sends us back when the data change.*

1. **Collect data.** Based on the problem, collect a dataset or use a previously collected one. Split it into training, validation and test sets before looking at it closely (below).
2. **Fit a model.** From a wide range of choices (the lectures below), select a model and fit its parameters on the training set. This is the training step.
3. **Evaluate, iterate, then deploy.**
   - During development, evaluate the fitted model on the validation set. If it falls short, go back: tune the model and its hyperparameters (step 2), or collect more or better data (step 1).
   - When the model is good enough, measure it **once** on the test set, then deploy it to make predictions on new data. Monitor it in use: if the incoming data drift away from the training distribution, performance degrades silently, and the loop starts again.

### Data splits

The point of the splits is to estimate how the model will do on **new** data. Error on the data a model was fitted to is optimistic, because flexible models can memorize it (overfitting).
- The **training set** is used to fit the parameters.
- The **validation set** is used to choose among models and hyperparameters, such as the regularization strength in [[Linear Models#MAP estimation and regularization|ridge regression]] or the number of neighbours $k$ in [[Non-parametric Models|k-nearest neighbours]]. Because we choose whatever scores best on it, its score is also slightly optimistic.
- The **test set** is touched only once, at the end, to report performance. Using it to make decisions turns it into a second validation set.

Typical proportions are 60/20/20 or 80/10/10. When data are scarce, **$k$-fold cross-validation** replaces the single validation set: split the non-test data into $k$ folds, train on $k-1$ of them and validate on the remaining one, rotate through all $k$, and average the $k$ scores. Two common pitfalls:
- **Leakage.** Any preprocessing fitted to data, such as standardization statistics or feature selection, must be fitted on the training set only, then applied to the validation and test sets.
- **Non-i.i.d. data.** Splits must respect the structure of the data. Time series are split by time (train on the past, test on the future). Grouped data, such as several samples per patient, are split by group, so the same group never appears on both sides. Classification splits are usually **stratified** to keep the class proportions equal.

## Supervised learning

1. [[Linear Models]]: least squares (geometry, normal equations, MLE), gradient descent, ridge and lasso as MAP estimation, and logistic/softmax regression with cross-entropy.
2. [[Discriminant Analysis]]: generative classifiers (LDA, QDA) and Fisher's linear discriminant.
3. [[Support Vector Machines]]: maximum-margin classifiers, the dual and support vectors, kernels, soft margins and the hinge loss, and SVR.
4. [[Non-parametric Models]]: k-nearest neighbours, Parzen windows, kernel density estimation (with its bias–variance trade-off), and fast neighbour search with KD-trees and approximate indexes.

## Unsupervised learning

Learning structure from data without labels: compact representations, clusters, and generative models that can produce new data. The lectures build on each other in this order. Matrix factorization gives linear representations; EM fits latent-variable models; VAEs and GANs make those models deep; energy-based and score-based models lead to diffusion.

5. [[Matrix Factorization and Dimensionality Reduction]]: eigen-decomposition, SVD and Eckart–Young (with proof), power iteration and QR, pseudo-inverse, recommender systems, NMF and CUR; PCA (max variance, min error, via SVD), autoencoders, kernel PCA, and manifold methods.
6. [[Clustering and Expectation Maximization]]: K-Means (with k-means++ and choosing $K$), fuzzy and kernel K-Means, EM and the ELBO, Gaussian mixtures, spectral clustering, modularity and Louvain, hierarchical clustering, DBSCAN and evaluation metrics.
7. [[Variational Autoencoders]]: the ELBO as amortized EM, score-function vs reparameterization gradients, the Gaussian VAE with its closed-form KL, sampling through the aggregate posterior, forward vs reverse KL, and practical issues (blurry samples, posterior collapse, β-VAE).
8. [[Generative Adversarial Networks]]: f-divergences and Fenchel conjugates, the variational lower bound and the f-GAN, the original GAN as a special case ($D^*$, $2\,\mathrm{JSD}-\log4$), training in practice, mode collapse, and the Wasserstein GAN.
9. [[Energy-Based Models]]: the Boltzmann distribution (staircase intuition and maximum entropy), Hopfield networks as associative memory, Boltzmann machines and RBMs with contrastive divergence, the EBM likelihood gradient and Langevin sampling, contrastive losses (InfoNCE, CLIP) derived from energy, and modern Hopfield networks as attention.
10. [[Score-Based Models]]: the score and its properties, implicit, sliced and denoising score matching, Tweedie's formula (the score is the optimal denoiser), why naive score-based sampling fails, noise-conditional score networks with annealed Langevin dynamics, and a preview of the SDE view.

Coming soon, in this order: Denoising Diffusion Models, (Conditional) Flow Matching, and Contrastive Representation Learning. Also planned: Normalizing Flows.
