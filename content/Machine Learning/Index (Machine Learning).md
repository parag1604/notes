---
title: Index (Machine Learning)
next: "[[Bayesian Decision Theory]]"
---

Machine learning built up from first principles: supervised learning, then unsupervised learning, from matrix factorization and clustering to deep generative models. Each lecture lists its prerequisites at the top; the probability they use is in the [[Index (Probability and Linear Algebra)|Probability and Linear Algebra]] series.

## What kind of problem is it?

Sorting a list is not a learning problem: a fixed recipe works on every input, and having sorted lists before does not help. Telling cats from dogs in photos is: no one can write down the rule, but a program can extract it from examples, and it should get better with more of them. Machine learning is about such problems, and they come in several flavours:

| Setting | What the learner gets | Example | Where in these notes |
|---|---|---|---|
| Supervised | inputs with labels | spam filtering, house prices | Supervised learning, below |
| Unsupervised | inputs only | communities in a social network, compression | Unsupervised learning, below |
| Semi-supervised | a few labelled and many unlabelled inputs | labelling a fraction of medical scans | builds on both |
| Active | inputs, and the right to ask for chosen labels | asking an expert only about confusing cases | — |
| Transfer, domain adaptation | data from a related task or distribution | a model trained on photos applied to sketches | — |
| Few-shot, zero-shot | a handful of labelled examples, or none, for a new class | recognizing a new product from one picture | contrastive representations (planned lecture) help |
| Interactive | rewards for actions the learner chooses | adaptive clinical trials, game playing | [[Multi-Armed Bandits]], [[Introduction to Reinforcement Learning]] |

**Data and models.** Data usually arrive as vectors $\mathbf{x}\in\mathbb{R}^D$: measured features, pixel values, or embeddings of words. Their structure matters for the choice of model: grids (images), sequences (text, audio, time series), relations (tables, graphs), or several of these at once (videos, medical records). A **model** is a simplified, usually probabilistic, description of how the data were generated, with parameters fitted from data. For example, a friendship network in a school can be modelled by saying that two students in the same grade are friends with probability $p$, and students in different grades with probability $q<p$. Fitting $p$, $q$ and the grades from the network recovers its communities ([[Clustering and Expectation Maximization]]). A good model is flexible enough to capture the data, and simple enough to fit and to use.

## How machine learning works in practice

A machine-learning model is a function fitted to data, so that it can make predictions on data it has not seen. Whatever the model, a practitioner goes through the same loop:

![[Machine Learning/images/ml-workflow.svg]]
*The machine-learning workflow. Evaluation sends us back to tune the model or to collect better data; after deployment, monitoring sends us back when the data change.*

1. **Collect and prepare data.** Based on the problem, collect a dataset or use a previously collected one. Split it into training, validation and test sets before looking at it closely (below). Then clean it: fix or drop corrupted records, decide how to handle missing values and outliers, and turn raw inputs into features. Features can be designed by hand (counts, ratios, domain measurements), reduced in number ([[Matrix Factorization and Dimensionality Reduction|PCA]]), or learned by the model itself, as neural networks do.
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

The data are pairs $\mathcal{D}=\{(\mathbf{x}_i,y_i)\}_{i=1}^N$, drawn independently from an unknown joint distribution $P(\mathbf{x},y)$. When the label $y$ is a real number the task is **regression**; when it is one of $K$ classes, **classification**; when it is a structured object (a sentence, a segmentation mask) it is structured prediction. The goal is a predictor that does well on **new** draws from $P$, not just on the training pairs. Labels may be noisy (a mislabelled scan, a measurement error), so even the best possible predictor makes mistakes: that floor is the [[Bayesian Decision Theory#The probability of error|Bayes error]]. Inputs are often very high-dimensional (an $800\times600$ colour photo has $1.44$ million numbers), which is what makes the problem hard ([[Non-parametric Models]] discusses the curse of dimensionality). The first lecture describes the ideal predictor; the others are ways of approximating it from data.

1. [[Bayesian Decision Theory]]: the optimal classifier when the distributions are known: posterior decisions and the Bayes error, losses and risk, the likelihood ratio test, the reject option, the minimax rule, and Gaussian discriminants (linear and quadratic boundaries).
2. [[Linear Models]]: least squares (geometry, normal equations, MLE), gradient descent, ridge and lasso as MAP estimation, and logistic/softmax regression with cross-entropy.
3. [[Discriminant Analysis]]: generative classifiers (LDA, QDA) and Fisher's linear discriminant.
4. [[Support Vector Machines]]: maximum-margin classifiers, the dual and support vectors, kernels, soft margins and the hinge loss, and SVR.
5. [[Non-parametric Models]]: k-nearest neighbours, Parzen windows, kernel density estimation (with its bias–variance trade-off), and fast neighbour search with KD-trees and approximate indexes.

## Unsupervised learning

Learning structure from data without labels: compact representations, clusters, and generative models that can produce new data. The lectures build on each other in this order. Matrix factorization gives linear representations; EM fits latent-variable models; VAEs and GANs make those models deep; energy-based and score-based models lead to diffusion.

6. [[Matrix Factorization and Dimensionality Reduction]]: eigen-decomposition, SVD and Eckart–Young (with proof), power iteration and QR, pseudo-inverse, recommender systems, NMF and CUR; PCA (max variance, min error, via SVD), autoencoders, kernel PCA, and manifold methods.
7. [[Clustering and Expectation Maximization]]: K-Means (with k-means++ and choosing $K$), fuzzy and kernel K-Means, EM and the ELBO, Gaussian mixtures, spectral clustering, modularity and Louvain, hierarchical clustering, DBSCAN and evaluation metrics.
8. [[Variational Autoencoders]]: the ELBO as amortized EM, score-function vs reparameterization gradients, the Gaussian VAE with its closed-form KL, sampling through the aggregate posterior, forward vs reverse KL, and practical issues (blurry samples, posterior collapse, β-VAE).
9. [[Generative Adversarial Networks]]: f-divergences and Fenchel conjugates, the variational lower bound and the f-GAN, the original GAN as a special case ($D^*$, $2\,\mathrm{JSD}-\log4$), training in practice, mode collapse, and the Wasserstein GAN.
10. [[Energy-Based Models]]: the Boltzmann distribution (staircase intuition and maximum entropy), Hopfield networks as associative memory, Boltzmann machines and RBMs with contrastive divergence, the EBM likelihood gradient and Langevin sampling, contrastive losses (InfoNCE, CLIP) derived from energy, and modern Hopfield networks as attention.
11. [[Score-Based Models]]: the score and its properties, implicit, sliced and denoising score matching, Tweedie's formula (the score is the optimal denoiser), why naive score-based sampling fails, noise-conditional score networks with annealed Langevin dynamics, and a preview of the SDE view.

Coming soon, in this order: Denoising Diffusion Models, (Conditional) Flow Matching, and Contrastive Representation Learning. Also planned: Normalizing Flows.
