---
title: Linear Models
---


## Linear Regression

### Ordinary Least Squares

Given dataset of $N$ features $\mathbf{X} = \{\mathbf{x}_i\}_{i=1}^{N}$ (where $\mathbf{x}_i\in\mathbb{R}^{D}$) and corresponding labels $\mathbf(y)=\{y_i\}_{i=1}^{N}$, we essentially learn a vector of weights $\mathbf{w}\in\mathbb{R}^{D}$ and a scalar bias $b$ such that:$$\hat{y} = \mathbf{w}^T \mathbf{x} + b$$where $\hat{y}$ denotes the predicted output of our linear model. Notice that the bias $b$ can be included into the weights if the corresponding dimension of the features is $1$.

**Insert the reason why this is reasonable***

Our aim is to minimize the empirical risk, which we can do by taking the squared differences of the actual label $y$ and the predicted label $\hat{y}$:$$Loss = \sum_{i=1}^{N}\frac{1}{2}\left(y_i-\hat{y}_i\right)^2 = \sum_{i=1}^{N}\frac{1}{2}\left(y_i-\mathbf{w}^T\mathbf{x}_i\right)^2$$However, since the loss has to be calculated over the entire dataset, we can always write it in the vectorized form:$$Loss = \frac{1}{2}\left|\left|\mathbf{y}-\mathbf{X}\mathbf{w}\right|\right|_2^2$$where $||\cdot||$ is the $\mathrm{L}_2$ norm.

Now, in order to minimize the loss, we can take the gradient with respect to $\mathbf{w}$:
$$
\begin{align*}
\nabla_\mathbf{w}Loss &= \nabla_\mathbf{w}\frac{1}{2}\left|\left|\mathbf{y}-\mathbf{X}\mathbf{w}\right|\right|_2^2\\
&= \frac{1}{2}\cdot2\left(\mathbf{y}-\mathbf{X}\mathbf{w}\right)\cdot\left(-\mathbf{X}\right)\\
&= \mathbf{X}^T\left(\mathbf{X}\mathbf{w}-\mathbf{y}\right) ... [1]
\end{align*}
$$
Taking the Hessian, we get:
$$
\begin{align*}
\nabla^2_\mathbf{w}Loss &= \nabla_\mathbf{w}\mathbf{X}^T\left(\mathbf{X}\mathbf{w}-\mathbf{y}\right)\\
&= \mathbf{X}^T\mathbf{X}
\end{align*}
$$
which is positive definite (**assuming** *full rank*, *i.e.* rank is $D$), thus if we equate the gradient of the loss to 0, we should obtain the minima:
$$
\begin{align*}
\nabla_\mathbf{w}Loss = \mathbf{X}^T\left(\mathbf{X}\mathbf{w}-\mathbf{y}\right) &= 0\\
\mathbf{X}^T\mathbf{X}\mathbf{w} &= \mathbf{X}^T\mathbf{y}\\
\mathbf{w} &= \left(\mathbf{X}^T\mathbf{X}\right)^{-1}\mathbf{X}^T\mathbf{y}
\end{align*}
$$
Note that the inverse of the matrix $\mathbf{X}^T\mathbf{X}$ is not possible unless it is full rank, and hence our previous assumption was reasonable.

### Stochastic Gradient Descent
