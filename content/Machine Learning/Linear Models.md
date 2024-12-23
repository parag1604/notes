---
title: Linear Models
---


## Linear Regression

### Ordinary Least Squares

Given dataset of $N$ features $\mathbf{X} = \{\mathbf{x}_i\}_{i=1}^{N}$ (where $\mathbf{x}_i\in\mathbb{R}^{D}$) and corresponding labels $\mathbf(y)=\{y_i\}_{i=1}^{N}$, we essentially learn a vector of weights $\mathbf{w}\in\mathbb{R}^{D}$ and a scalar bias $b$ such that:
$$
\hat{y} = \mathbf{w}^T \mathbf{x} + b
$$
where $\hat{y}$ denotes the predicted output of our linear model. Notice that the bias $b$ can be included into the weights if the corresponding dimension of the features is $1$, which essentially compresses the expression to:
$$
\hat{y} = \mathbf{w}^T \mathbf{x}
$$
where $\mathbf{X}$ is now a $N\times (D+1)$ matrix with the first column being a vector of ones.

We can consider these as a system of linear equations. As we know a linear system may be consistent or inconsistent. In the former case where $N\leq D$, solutions do exist, however the model overfits (memorizes the training data). This is undesirable since the model does not generalize well to new examples. In the latter case however, we don't have unique solutions. For most practical purposes, $N>D$ leading the system to be inconsistent. So, what can we do?

We can think of a few ways to solve this problem:

1. We can simply take $w=\mathbf{X}^{-1}\mathbf{y}$, however since $\mathbf{X}$ is not invertible ($N>D$), this is not possible.
2. In that case, we can take the Moore-Penrose pseudo-inverse of $\mathbf{X}$. Let the Singular Value Decomposition (SVD) of $\mathbf{X}=\mathbf{U}\Sigma\mathbf{V}^T$ where $\mathbf{V}$ is the eigen vectors of $\mathbf{X}^T\mathbf{X}$, $\mathbf{U}$ is the eigen vectors of $\mathbf{X}\mathbf{X}^T$, and $\Sigma$ is the diagonal matrix of singular values which are the square roots of the eigenvalues of $\mathbf{X}^T\mathbf{X}$. Then,

$$
w=\mathbf{V}\Sigma^{-1}\mathbf{U}^T\mathbf{y}
$$

However, the eigenvalue and eigenvector calculation requires a lot of computations. Can we think of a way to reduce the computations?

**Insert Image**

We can think of the plane as the column space spanned by the dataset $C(\mathbf{X})$, and we can assume that the ground truth targets lie outside the $C(\mathbf{X})$ plane. We can then take the projection of the ground truth labels to the $C(\mathbf{X})$ plane, that should correpond to the predicted labels. Recall that the predicted labels were calculated as $\hat{y}=\mathbf{w}^T\mathbf{x}$. Therefore the vector $\mathbf{X}\mathbf{w}$ should lie in the column space of $C(\mathbf{X})$. We can estimate the weights $\mathbf{w}$ in the following 2 ways:

#### Method 1 (ERM):
Our aim is to minimize the empirical risk, which we can do by minimizing the error. We notice that the error is given by the differences of the actual label $y$ and the predicted label $\hat{y}$. We can take the squared differences (since square is a monotonically increasing function in the range [0,$\infty$)) and sum them up across all the datapoints in the dataset to get the loss function:
$$
Loss = \sum_{i=1}^{N}\frac{1}{2}\epsilon^2 = \sum_{i=1}^{N}\frac{1}{2}\left(y_i-\hat{y}_i\right)^2 = \sum_{i=1}^{N}\frac{1}{2}\left(y_i-\mathbf{w}^T\mathbf{x}_i\right)^2
$$
This objective is popularly known as the "mean squared error" (MSE).
However, since the loss has to be calculated over the entire dataset, we can always write it in the vectorized form:
$$
Loss = \frac{1}{2}\left|\left|\mathbf{y}-\mathbf{X}\mathbf{w}\right|\right|_2^2
$$
where $||\cdot||$ is the $\mathrm{L}_2$ norm.

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

#### Method 2 (Projection):

Alternatively, we can directly calculate the projection of the ground truth labels to the $C(\mathbf{X})$ plane. The error of can be expressed as $\epsilon=\mathbf{y}-\mathbf{Xw}$. Since we know that the vector $\epsilon$ is perpendicular to the plane $C(\mathbf{X})$, we get the following relation:
$$
\begin{align*}
X^T\epsilon &= 0\\
X^T(\mathbf{y}-\mathbf{X}\mathbf{w}) &= 0\\
\mathbf{X}^T\mathbf{y}-\mathbf{X}^T\mathbf{X}\mathbf{w} &= 0\\
\mathbf{X}^T\mathbf{X}\mathbf{w} &= \mathbf{X}^T\mathbf{y} \\
\mathbf{w} &= \left(\mathbf{X}^T\mathbf{X}\right)^{-1}\mathbf{X}^T\mathbf{y}
\end{align*}
$$

----

### Iterative Methods for Linear Regression

We notice that the $(\mathbf{X}^T\mathbf{X})^-1$ operation takes $\mathcal{O}(D^3)$ computations, which can be costly if D is large. Also $\mathbf{X}^T\mathbf{X}$ itself requires $\mathcal{O}(ND^2)$ FLOPs to compute, which can be costly since N is often very high. Hence it would be useful if we had a way to optimize the parameters without the requirement of inverting the $\mathbf{X}^T\mathbf{X}$ matrix.

We can use the Gradient Descent or Stochiastic Gradient Descent (SGD) to optimize the parameters when N is large. Let $f_{\mathbf{w}^t}(\mathbf{x}_i, y_i)$ be the loss function, then:

Gradient Descent:

1. Initialize $\mathbf{w}^0\sim \mathcal{N}(0,I)$
2. while $\Vert\mathbf{w}^t-\mathbf{w}^{t-1}\Vert > \epsilon$:

$$
w^{t+1} \leftarrow \mathbf{w}^t - \eta\nabla_{\mathbf{w}}f_{\mathbf{w}^t}(X, y)
$$

In Gradient descent, notice that the gradient is calculated for the entire dataset. However, that might also be computationally expensive. Hence, we can use Stochastic Gradient Descent.

Stochastic Gradient Descent:

1. Initialize $\mathbf{w}^0\sim \mathcal{N}(0,I)$
2. while $\Vert\mathbf{w}^t-\mathbf{w}^{t-1}\Vert > \epsilon$:

$$
i \sim \mathcal{U}\left(1,N\right)\\
w^{t+1} \leftarrow \mathbf{w}^t - \eta\nabla_{\mathbf{w}}f_{\mathbf{w}^t}(x_i, y_i)
$$

However, this requires us to compute the gradient for each datapoint, which might not be a good estimate of the gradient. Hence, we can use Mini-Batch Gradient Descent which is a tradeoff between the above two methods.

Mini-Batch Gradient Descent:

1. Initialize $\mathbf{w}^0\sim \mathcal{N}(0,I)$
2. while $\Vert\mathbf{w}^t-\mathbf{w}^{t-1}\Vert > \epsilon$:

$$
i_1, \ldots, i_m \sim \mathcal{U}\left(1,N\right) \text{without replacement}\\
\mathbf{X}_i = \left(\mathbf{x}_{i_1}, \ldots, \mathbf{x}_{i_m}\right), \mathbf{y}_i = \left(y_{i_1}, \ldots, y_{i_m}\right)\\
w^{t+1} \leftarrow \mathbf{w}^t - \eta\nabla_{\mathbf{w}}f_{\mathbf{w}^t}(\mathbf{X}_i, \mathbf{y}_i)
$$

In our case, the loss function $f_{\mathbf{w}^t}(\mathbf{X}_i, \mathbf{y}_i)=\frac{1}{2}\left|\left|\mathbf{y}_i-\mathbf{X}_i\mathbf{w}^t\right|\right|_2^2$ and the gradient is given by $[1]$ i.e. $\nabla_{\mathbf{w}}f_{\mathbf{w}^t}(\mathbf{X}_i, \mathbf{y}_i)=\mathbf{X}_i^T\left(\mathbf{X}_i\mathbf{w}^t-\mathbf{y}_i\right)$.

----

### Probabilistic Perspective of Linear Regression

**insert image**

----

### Regularized Linear Regression

----

## Logistic Regression

**insert image**



## Additional Topics

### Weighted Linear Regression

Let's consider a dataset $\mathcal{D}=\{(\mathbf{x}_i,y_i,r_i)\}_{i=1}^N$, where $\mathbf{x}_i\in\mathbb{R}^D$ are the features, $y_i\in\mathbb{R}$ are the labels, and $r_i>0$ are the weighing factor, i.e. some datapoints are given more weights than others (similar to [[Reinforcement Learning/Policy Gradients|policy gradient]] algorithms in [[Reinforcement Learning|reinforcment learning]]). Then the ERM objective is:

$$
w^* = \arg\min_w \sum_{i=1}^N \frac{1}{2}r_i\left(y_i-\mathbf{w}^T\mathbf{x}_i\right)^2\\
w^* = \arg\min_w \frac{1}{2}(\mathbf{Xw}-\mathbf{y})^T\mathbf{R}(\mathbf{Xw}-\mathbf{y})
$$
where $\mathbf{R}$ is a diagonal matrix with $r_i$ on the diagonal.

$$
\begin{align*}
(\mathbf{Xw}-\mathbf{y})^T\mathbf{R}(\mathbf{Xw}-\mathbf{y}) &= (\mathbf{Xw}-\mathbf{y})^T(\mathbf{RXw}-\mathbf{Ry})\\
&= (\mathbf{Xw}-\mathbf{y})^T\mathbf{RXw}-(\mathbf{Xw}-\mathbf{y})^T\mathbf{Ry}\\
&= \mathbf{w}^T\mathbf{X}^T\mathbf{R}\mathbf{Xw}-\mathbf{w}^T\mathbf{X}^T\mathbf{R}\mathbf{y}-\mathbf{y}^T\mathbf{R}\mathbf{Xw}+\mathbf{y}^T\mathbf{R}\mathbf{y}\\
&= \mathbf{w}^T\mathbf{X}^T\mathbf{R}\mathbf{Xw}-2\mathbf{w}^T\mathbf{X}^T\mathbf{R}\mathbf{y}+\mathbf{y}^T\mathbf{R}\mathbf{y}\\
\end{align*}
$$