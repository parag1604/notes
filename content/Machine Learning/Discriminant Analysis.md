---
title: Discriminant Analysis
draft: "False"
tags:
  - quadratic-discriminant-analysis
  - linear-discriminant-analysis
  - fisher-discriminant-analysis
---
## Distance-based Discriminator

Given dataset of $N$ datapoints $\mathbf{X} = \{\mathbf{x}_i\}_{i=1}^{N}$ (where $\mathbf{x}_i\in\mathbb{R}^{D}$) and corresponding labels $\mathbf{y}=\{y_i\}_{i=1}^{N}$ (where each $y_i\in\{0,1,...,C-1\}$), we can extract some useful patterns among the datapoints and then essentially forget the dataset. This procedure is called learning and the models which learn useful parameters from the dataset are known as parametric models.
For simplicity we can assume that there are only 2 classes $\{+,-\}$ ($+$ can be considered 1 and $-$ can be considered $0$ or $-1$).

![[Machine Learning/images/FDA1.png]]

In the dataset described in the above image, we already know the labels. So, we can learn the means of the class conditional datapoints, i.e.
$$
\mathbf{\mu}_j=\frac{1}{\sum_k\mathbb{1}_{\{y_i=j\}}\mathbf{x}_k}\sum_i^N\mathbb{1}_{\{y_k=j\}}\mathbf{x}_i
$$

We can create a simple discriminator function by comparing the new datapoint with the class conditional means, with the label of the new datapoint being same as the most similar mean, i.e.

$$
y_{new} = \arg\max_j~\langle \mu_j,\mathbf{x}_{new}\rangle
$$

## Fisher Discriminant Analysis
![[Machine Learning/images/FDA2.png]]

We can additionally assume the dataset to be sampled from a Gaussian. Fisher thought of trying to find a linear projection of the given dataset. This data can be projected on the standard basis or on other subspaces as shown below (linear projections):

![[Machine Learning/images/FDA3.png]]

As we can see, not all choices for the linear projection are equally got at preserving the class information and giving a classifier which might generalize well.

We want to identify a vector for linear projection that
1. maximizes the between-class spread, and
2. minimizes the within-class spread.

Let the means after the linear projection be $m_1$ and $m_2$, and the variances be $s_1$ and $s_2$.

So, we would like to make the following ratio big:
$$
R=\frac{(m_1-m_2)^2}{s_1^2+s_2^2}
$$

We can assume that $m_1=\mathbf{w}^T\mathbf{\mu}_1$ and $m_2=\mathbf{w}^T\mathbf{\mu}_2$ where $\mathbf{\mu}_1$ and $\mathbf{\mu}_2$ are the actual means of the class conditional data distributions. Therefore:
$$
m_1 - m_2 = \mathbf{w}^T(\mathbf{\mu}_1 - \mathbf{\mu}_2)
$$

Similarly $s_1^2+s_2^2$ can be written as:
$$
\mathbf{w}^T\mathbf{S}_1\mathbf{w}+\mathbf{w}^T\mathbf{S}_1\mathbf{w}=\mathbf{w}^T\mathbf{S}\mathbf{w}
$$
where $\mathbf{S}$ is the covariance matrix of the entire dataset.

Again,
$$
\left(\mathbf{w}^T(\mathbf{\mu_1-\mu_2})\right)^2=\mathbf{w}^T(\mathbf{\mu_1-\mu_2})(\mathbf{\mu_1-\mu_2})^T\mathbf{w}=\mathbf{w}^T\mathbf{M}\mathbf{w}
$$

Therefore, the objective is:
$$
\max_\mathbf{w}~R(\mathbf{w})=\frac{\mathbf{w}^T\mathbf{M}\mathbf{w}}{\mathbf{w}^T\mathbf{S}\mathbf{w}}
$$

We can take the gradient:
$$
\begin{align}
\nabla_\mathbf{w}R(\mathbf{w})=(\mathbf{w}^T\mathbf{S}\mathbf{w})\frac{\partial}{\partial\mathbf{w}}(\mathbf{w}^T\mathbf{M}\mathbf{w})-(\mathbf{w}^T\mathbf{M}\mathbf{w})\frac{\partial}{\partial\mathbf{w}}(\mathbf{w}^T\mathbf{S}\mathbf{w})&=0\\
\implies (\mathbf{w}^T\mathbf{S}\mathbf{w})(2\mathbf{M}\mathbf{w}) - (\mathbf{w}^T\mathbf{M}\mathbf{w})(2\mathbf{S}\mathbf{w}) &= 0 \\
\implies \mathbf{M}\mathbf{w} &= \frac{\mathbf{w}^T\mathbf{M}\mathbf{w}}{\mathbf{w}^T\mathbf{S}\mathbf{w}}\mathbf{S}\mathbf{w} \\
\implies \mathbf{M}\mathbf{w} &= R~\mathbf{S}\mathbf{w} \\
\implies \mathbf{R}\mathbf{w} &= \mathbf{S}^{-1}~\mathbf{M}\mathbf{w} \\
\implies \mathbf{R}\mathbf{w} &= \mathbf{S}^{-1}(\mathbf{\mu_1-\mu_2})(\mathbf{\mu_1-\mu_2})^T\mathbf{w} \\
\implies \mathbf{R}\mathbf{w} &= (\mathbf{\mu_1-\mu_2})^T\mathbf{w}~\mathbf{S}^{-1}(\mathbf{\mu_1-\mu_2}) \\
\implies \mathbf{w} &= \frac{(\mathbf{\mu_1-\mu_2})^T\mathbf{w}}{\mathbf{R}}\mathbf{S}^{-1}(\mathbf{\mu_1-\mu_2}) \\
\implies \mathbf{w} &\propto \mathbf{S}^{-1}(\mathbf{\mu_1-\mu_2}) \\
\end{align}
$$

Thus, the projection surface is along the direction $\mathbf{S}^{-1}(\mathbf{\mu_1-\mu_2})$.

---
## Probabilistic view of Discriminant Analysis

We can learn a generative model and estimate the distribution that generated the dataset for each class separately, i.e. $\mathbb{P}(\mathbf{x}_i|y_i=k)$. Given the generative model, we can use Bayes rule to design an optimal classifier.

Since we are in the binary classification domain ($y_i\in\{0,1\}$):
$$
\mathbb{P}(y_i=1|\mathbf{x}_i)=\frac{\mathbb{P}(\mathbf{x}_i|y_i=1)\mathbb{P}(y_i=1)}{\mathbb{P}(\mathbf{x}_i|y_i=1)\mathbb{P}(y_i=1)+\mathbb{P}(\mathbf{x}_i|y_i=0)\mathbb{P}(y_i=0)}
$$

We can denote prior probabilities $\mathbb{P}(y_i=k)$ as $\pi_k$ where $\pi_1+\pi_0=1$ and $\pi_k$ can be estimated from the dataset as:
$$
\pi_k=\frac{1}{N}\sum_{i=1}^N\mathbb{1}_{\{y_i=k\}}
$$
Therefore:
$$
\mathbb{P}(y_i=1|\mathbf{x}_i)=\frac{\pi_1\mathbb{P}(\mathbf{x}_i|y_i=1)}{\pi_0\mathbb{P}(\mathbf{x}_i|y_i=0)+\pi_1\mathbb{P}(\mathbf{x}_i|y_i=1)}
$$

In the context of Discriminant Analysis, we assume that the class conditional datapoints were sampled from Gaussian distributions. Since we are considering the binary classification problem, the parameters are $(\mu_1, \Sigma_1)$ and $(\mu_0, \Sigma_0)$  for the positive and negative class respectively.

Given a class label, it is easy to estimate the parameters
Class conditional sample mean: $\mu_k=\frac{1}{N}\sum_{i=1}^N\mathbf{x}_i$, and
Class conditional sample covariance matrix: $\Sigma_k=\frac{1}{N-1}\sum_{i=1}^N(\mathbf{x}_i-\mu_k)(\mathbf{x}_i-\mu_k)^T$
Additionally, we already know the priors: $\pi_k=\frac{1}{N}\sum_{i=1}^N\mathbb{1}_{\{y_i=k\}}$

Now all we need to do is find a decision rule. The decision boundary is given by the region where $\mathbb{P}(y_i=0|\mathbf{x}_i)=\mathbb{P}(y_i=1|\mathbf{x}_i)$.

### Quadratic Discriminant Analysis

For a gaussian:
$$
\begin{align}
\mathbb{P}(y_i=0|\mathbf{x}_i) &= \mathbb{P}(y_i=1|\mathbf{x}_i) \\
\implies \mathbb{P}(\mathbf{x}_i|y_i=0)\pi_0 &= \mathbb{P}(\mathbf{x}_i|y_i=1)\pi_1 \\
\implies \log\mathbb{P}(\mathbf{x}_i|y_i=0)+\log\pi_0 &= \log\mathbb{P}(\mathbf{x}_i|y_i=1)+\log\pi_1 \\
\implies \log\left((2\pi)^{-D/2}|\Sigma_0|^{-1/2}\exp\left(-0.5(\mathbf{x}_i-\mu_0)^T\Sigma_0^{-1}(\mathbf{x}_i-\mu_0)\right)\right)&+\log\pi_0 =\\
\log\left((2\pi)^{-D/2}|\Sigma_1|^{-1/2}\exp\left(-0.5(\mathbf{x}_i-\mu_1)^T\Sigma_1^{-1}(\mathbf{x}_i-\mu_1)\right)\right)&+\log\pi_1 \\
\end{align}
$$
Classification rule (for QDA):
Let the ratio be
$$
R = \frac{\log\pi_1-0.5\log|\Sigma_1|-0.5(\mathbf{x}_i-\mu_1)^T\Sigma_1^{-1}(\mathbf{x}_i-\mu_1)}{\log\pi_0-0.5\log|\Sigma_0|-0.5(\mathbf{x}_i-\mu_0)^T\Sigma_0^{-1}(\mathbf{x}_i-\mu_0)}
$$
Then $y_i=1$ if $R>1$ else $y_i=0$.

### Linear Discriminant Analysis

We can further assume that we have the same covariance matrix for both the classes, i.e. $\Sigma_0=\Sigma_1=\Sigma$.

We can expand
$$
\begin{align}
(\mathbf{x}_i-\mu_j)^T\Sigma^{-1}(\mathbf{x}_i-\mu_j)&=(\mathbf{x}_i-\mu_j)^T(\Sigma^{-1}\mathbf{x}_i-\Sigma^{-1}\mu_j)\\
&=\mathbf{x}_i^T\Sigma^{-1}\mathbf{x}_i-2\mathbf{x}_i^T\Sigma^{-1}\mu_j+\mu_j^T\Sigma^{-1}\mu_j
\end{align}
$$
Therefore,
$$
\begin{align}
\log\pi_1+\mathbf{x}_i^T\Sigma^{-1}\mu_1-0.5\mu_1^T\Sigma^{-1}\mu_1 &= \log\pi_0+\mathbf{x}_i^T\Sigma^{-1}\mu_0-0.5\mu_0^T\Sigma^{-1}\mu_0 \\
\implies \mathbf{x}_i^T\Sigma^{-1}\mu_0-\mathbf{x}_i^T\Sigma^{-1}\mu_1 &= \log\pi_1-\log\pi_0 + 0.5~\mu_0^T\Sigma^{-1}\mu_0 - 0.5~\mu_1^T\Sigma^{-1}\mu_1 \\
\implies \mathbf{x}_i^T\Sigma^{-1}(\mu_0-\mu_1) &= \log\frac{\pi_1}{\pi_0}+0.5(\mu_0^T\Sigma^{-1}\mu_0-\mu_1^T\Sigma^{-1}\mu_1)=C
\end{align}
$$
where $C$ is a constant.

Notice that the term $\mathbf{x}_i^T\Sigma^{-1}(\mu_0-\mu_1)$ is just the linear projection of $\mathbf{x}_i$ onto the direction $\Sigma^{-1}(\mu_0-\mu_1)$, which is similar to Fisher Discriminant Analysis. Actually, LDA is exactly same as FDA, just derived from the probabilistic perspective.

Updated decision rule (for LDA):
$y_i=1$ if $\mathbf{x}_i^T\Sigma^{-1}(\mu_0-\mu_1) > C$ else $y_i=0$.
