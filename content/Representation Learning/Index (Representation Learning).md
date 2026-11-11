---
title: Index (Representation Learning)
---

Learning representations of data, and generative models that learn to produce new data. Each lecture lists its prerequisites at the top. The linear methods (PCA, kernel PCA, autoencoders) are in [[Matrix Factorization and Dimensionality Reduction]].

## Generative models

1. [[Variational Autoencoders]]: the ELBO as amortized EM, score-function vs reparameterization gradients, the Gaussian VAE with its closed-form KL, sampling through the aggregate posterior, forward vs reverse KL, and practical issues (blurry samples, posterior collapse, β-VAE).
2. [[Generative Adversarial Networks]]: f-divergences and Fenchel conjugates, the variational lower bound and the f-GAN, the original GAN as a special case ($D^*$, $2\,\mathrm{JSD}-\log4$), training in practice, mode collapse, and the Wasserstein GAN.

Coming soon: Energy-Based Models, Normalizing Flows, Diffusion Models.
