---
title: Linear Models
---


## Linear Regression

### Ordinary Least Squares

Given dataset of $N$ datapoints $\mathbf{X} = \{\mathbf{x}_i\}_{i=1}^{N}$ (where $\mathbf{x}_i\in\mathbb{R}^{D}$) and corresponding labels $\mathbf{y}=\{y_i\}_{i=1}^{N}$, we essentially learn a vector of weights $\mathbf{w}\in\mathbb{R}^{D}$ and a scalar bias $b$ such that:
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

We can think of the plane as the column space spanned by the dataset $C(\mathbf{X})$, and we can assume that the ground truth targets lie outside the $C(\mathbf{X})$ plane. We can then take the projection of the ground truth labels to the $C(\mathbf{X})$ plane, that should correspond to the predicted labels. Recall that the predicted labels were calculated as $\hat{y}=\mathbf{w}^T\mathbf{x}$. Therefore the vector $\mathbf{X}\mathbf{w}$ should lie in the column space of $C(\mathbf{X})$. We can estimate the weights $\mathbf{w}$ in the following 2 ways:

#### Method 1 (ERM)

Our aim is to minimize the empirical risk, which we can do by minimizing the error. We notice that the error is given by the differences of the actual label $y$ and the predicted label $\hat{y}$. We can take the squared differences (since square is a monotonically increasing function in the range $\mathbb{R}^+\cup\{0\}$) and sum them up across all the datapoints in the dataset to get the loss function:
$$
Loss = \sum_{i=1}^{N}\frac{1}{2}\epsilon^2 = \sum_{i=1}^{N}\frac{1}{2}\left(y_i-\hat{y}_i\right)^2 = \sum_{i=1}^{N}\frac{1}{2}\left(y_i-\mathbf{w}^T\mathbf{x}_i\right)^2 ...[1]
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

#### Method 2 (Projection)

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

We can use the Gradient Descent or Stochastic Gradient Descent (SGD) to optimize the parameters when N is large. Let $f_{\mathbf{w}^t}(\mathbf{x}_i, y_i)$ be the loss function, then:

*Gradient Descent*:

1. Initialize $\mathbf{w}^0\sim \mathcal{N}(0,I)$
2. while $\Vert\mathbf{w}^t-\mathbf{w}^{t-1}\Vert > \epsilon$:

$$
w^{t+1} \leftarrow \mathbf{w}^t - \eta\nabla_{\mathbf{w}}f_{\mathbf{w}^t}(X, y)
$$

In Gradient descent, notice that the gradient is calculated for the entire dataset. However, that might also be computationally expensive. Hence, we can use Stochastic Gradient Descent.

*Stochastic Gradient Descent*:

1. Initialize $\mathbf{w}^0\sim \mathcal{N}(0,I)$
2. while $\Vert\mathbf{w}^t-\mathbf{w}^{t-1}\Vert > \epsilon$:

$$
\begin{align*}
i &\sim \mathcal{U}\left(1,N\right)\\
w^{t+1} &\leftarrow \mathbf{w}^t - \eta\nabla_{\mathbf{w}}f_{\mathbf{w}^t}(x_i, y_i)
\end{align*}
$$

However, this requires us to compute the gradient for each datapoint, which might not be a good estimate of the gradient. Hence, we can use Mini-Batch Gradient Descent which is a tradeoff between the above two methods and the batch size can be chosen depending on the availability of RAM.

*Mini-Batch Gradient Descent*:

1. Initialize $\mathbf{w}^0\sim \mathcal{N}(0,I)$
2. while $\Vert\mathbf{w}^t-\mathbf{w}^{t-1}\Vert > \epsilon$:

$$
\begin{align*}
i_1, \ldots, i_m &\sim \mathcal{U}\left(1,N\right) \text{without replacement}\\
\mathbf{X}_i &= \left(\mathbf{x}_{i_1}, \ldots, \mathbf{x}_{i_m}\right)\\ \mathbf{y}_i &= \left(y_{i_1}, \ldots, y_{i_m}\right)\\
w^{t+1} &\leftarrow \mathbf{w}^t - \eta\nabla_{\mathbf{w}}f_{\mathbf{w}^t}(\mathbf{X}_i, \mathbf{y}_i)
\end{align*}
$$

In our case, the loss function $f_{\mathbf{w}^t}(\mathbf{X}_i, \mathbf{y}_i)=\frac{1}{2}\left|\left|\mathbf{y}_i-\mathbf{X}_i\mathbf{w}^t\right|\right|_2^2$ and the gradient is given by $[1]$ i.e. $\nabla_{\mathbf{w}}f_{\mathbf{w}^t}(\mathbf{X}_i, \mathbf{y}_i)=\mathbf{X}_i^T\left(\mathbf{X}_i\mathbf{w}^t-\mathbf{y}_i\right)$.

---
#### Aside (Data pre-processing):
We saw that in linear regression, the each feature is multiplied with the corresponding weight ($\mathbf{w}^T\mathbf{x}=w_1x_1+w_2x_2+...+w_Dx_D$). Now in case that the units are very different, the linear regression will not give very useful results. For instance, say the height and weight of a person is used to predict age. So, the height and weight are the features $x_1, x_2$ respectively, and the age is the target $y$. Now, suppose that the features are in SI units, which means that the height is in meters and the weight is in kilograms. Say a person weighs 1.5m and weighs 50kg. Therefore: $\hat{y} = w_1*1.5+w_2*50$. Notice that the height of the person will be given less weightage since the weight (in SI units) dominates. Ideally this can be resolved by learning a very low $w_2$ that can compensate for the high value of weight, it's not a good idea (more details in regularization). We can alternatively scale the dataset (feature-wise) to ensure that the units are in the same scale as follows:

1. *Normalization*:
   Let $x^{\min}_j$ be the minimum value of the datapoints along the $j^{th}$ feature column, and similarly $x^{\max}_j$ be the maximum value. Then:
   $$
   x_{ij}=(x_{ij}-x^{\min}_{j})/(x^{\max}_{j}-x^{\min}_{j})~~\forall i\in[N],j\in[D]
   $$
2. *Standardization*: 
   Let $\mu_j$ be the average value of the datapoints along the $j^{th}$ feature column, and similarly let $\sigma_j$ be the square root of the sample variance. Then:
   $$
   x_{ij}=(x_{ij}-\mu_{j})/\sigma_j~~\forall i\in[N],j\in[D]
   $$
Note: that the nomenclature might slightly confusing, so rather it's easier to remember the former as min-max scaling, while the later can be thought of as scaling the features as if they were sampled from isotropic standard Gaussian distributions each.

----

### Probabilistic Perspective of Linear Regression

Assume that the dataset $\{\mathbf{x}_i,y_i\}_{i=1}^N$ was generated using the following process:
$$
y_i=w^T\mathbf{x}_i+\epsilon ...[2]
$$
where $\epsilon\sim\mathcal{N}(0,\Sigma)$. For simplicity, we can assume unidimensional $\mathbf{x}$, which in turn makes $\epsilon\sim\mathcal{N}(0,\sigma^2)$.

**insert image**

Taking expectation we get:
$$
\begin{align*}
\mathbb{E}[y_i] &= \mathbb{E}[w^T\mathbf{x}_i+\epsilon]\\
&= \mathbb{E}[w^T\mathbf{x}_i]+\mathbb{E}[\epsilon]\\
&= \mathbb{E}[w^T\mathbf{x}_i]
\end{align*}
$$

Which means that: if we model $y$ as $w^T\mathbf{x}$, then we should be fine because in expectation we will be getting no errors.
We can reparameterize $[2]$ by assuming that the $y_i$ is generated from a Gaussian with mean $w^Tx_i$ and variance $\sigma^2$.
Thus the likelihood of $y_i$ given $x_i$ is:
$$
L(w|x_i,y_i)=p_w(y_i|x_i)=\frac{1}{\sqrt{2\pi\sigma^2}}\exp\left\{-\frac{(y_i-w^Tx_i)^2}{2\sigma^2}\right\}
$$
And consequently, the log likelihood is:
$$
l(w|x_i,y_i)=\log p_w(y_i|x_i)=\log\frac{1}{\sqrt{2\pi\sigma^2}} - \frac{(y_i-w^Tx_i)^2}{2\sigma^2}
$$
Therefore, the log likelihood of the dataset is:
$$
\begin{align*}
l(w|\mathbf{x},\mathbf{y})&=\log \prod_{i=1}^N p_w(y_i|x_i)\\
&=\sum_{i=1}^N\log p_w(y_i|x_i)\\
&=\sum_{i=1}^N\log\frac{1}{\sqrt{2\pi\sigma^2}} - \frac{(y_i-w^Tx_i)^2}{2\sigma^2}\\
&=N\log\frac{1}{\sqrt{2\pi\sigma^2}}-\sum_{i=1}^N\frac{(y_i-w^Tx_i)^2}{2\sigma^2}\\
\implies \arg\max_w l(w|\mathbf{x},\mathbf{y}) &= \arg\max_w -\sum_{i=1}^N\frac{(y_i-w^Tx_i)^2}{2\sigma^2}\\
&= \arg\min_w \sum_{i=1}^N\frac{(y_i-w^Tx_i)^2}{2\sigma^2}\\
&= \arg\min_w \sum_{i=1}^N\frac{1}{2}(y_i-w^Tx_i)^2\\
\implies Loss &= \sum_{i=1}^N\frac{1}{2}(y_i-w^Tx_i)^2
\end{align*}
$$
which is same as the loss function in the previous section (equation $[1]$). This is easily extendable to the multi-variate case.

----

### Regularized Linear Regression

We can Impose a prior on the weights, and get the aposteriori estimates. Let's assume $w\sim\mathcal{N}(0,\xi^2)$.
$$
\begin{align*}
w_{MAP} &= \arg\max_w \prod_{i=1}^{N} \left( \frac{1}{\sqrt{2\pi\sigma^2}}\exp\left\{-\frac{(y_i-w^Tx_i)^2}{2\sigma^2}\right\} \right) \left( \frac{1}{\sqrt{2\pi\xi^2}}\exp\left\{-\frac{w^2}{2\xi^2}\right\} \right)\\
&= \arg\max_w \prod_{i=1}^{N} c_1\exp\left\{-\frac{(y_i-w^Tx_i)^2}{2\sigma^2}\right\} c_2\exp\left\{-\frac{w^2}{2\xi^2}\right\} \\
&= \arg\min_w \sum_{i=1}^{N} \frac{(y_i-w^Tx_i)^2}{2\sigma^2}+\frac{w^2}{2\xi^2}\\
\implies Loss_{MAP} &= \sum_{i=1}^{N} \frac{1}{2}(y_i-w^Tx_i)^2+\frac{\sigma^2w^2}{2\xi^2} = \sum_{i=1}^{N} \frac{1}{2}(y_i-w^Tx_i)^2+\lambda\frac{w^2}{2}\\
\end{align*}
$$
where $\lambda=\sigma^2/\xi^2$. In the multivariate case, we have:
$$
\begin{align*}
Loss_{MAP} &= \sum_{i=1}^{N} \frac{1}{2}(y_i-\mathbf{w}^T\mathbf{x}_i)^2+\frac{\lambda}{2}\mathbf{w}^T\mathbf{w}\\
&=\sum_{i=1}^{N} \frac{1}{2}(y_i-\mathbf{w}^T\mathbf{x}_i)^2+\frac{\lambda}{2}\Vert\mathbf{w}\Vert_2^2\\
&= \frac{1}{2}\Vert \mathbf{y}-\mathbf{X}\mathbf{w}\Vert_2^2+\frac{\lambda}{2}\Vert\mathbf{w}\Vert_2^2
\end{align*}
$$
This is known as $L_2$-regularized regression or **Ridge regression** and the $\lambda$ can be considered as a hyper-parameter controlling the degree of regularization.

Taking the gradient of the loss w.r.t. $\mathbf{w}$ and equating it to zero, we get:
$$
\mathbf{w}^*_{MAP}=\left(\mathbf{X}^T\mathbf{X}+\frac{\sigma^2}{\xi^2}I\right)^{-1}\mathbf{X}^T\mathbf{y}=(\mathbf{X}^T\mathbf{X}+\lambda I)^{-1}\mathbf{X}^T\mathbf{y}
$$
Notice that unlike $\mathbf{X}^T\mathbf{X}$, the matrix $(\mathbf{X}^T\mathbf{X}+\lambda I)$ is positive definite and therefore invertible without any additional assumptions.

#### Why regularize?
The objective of *regularization* is to constrain the parameters in a certain way. Regularization is often used to deal with the problem of overfitting in high capacity (over-parameterized) models.

1. **L$_2$ Regularization**: We constrain the $L_2$ norm of the parameters, which results in parameters bering "small" dimension wise. In the ridge regression (assuming appropriate scaling has been performed on the dataset), notice that the regularization can be interpreted as roughly giving equal importance to each feature. We can rewrite the loss as Total loss = Reconstruction loss + $\lambda\cdot$Regularization, where:
   Reconstruction loss: $\sum_i(y_i-\mathbf{x}_{i1}\mathbf{w}_1-\mathbf{x}_{i2}\mathbf{w}_2-...-\mathbf{x}_{iD}\mathbf{w}_D)^2$, and
   Regularization: $\mathbf{w}_1^2+\mathbf{w}_2^2+...+\mathbf{w}_D^2$
   Consequently, if any feature (say $j$) gets more weightage, i.e. the corresponding $\mathbf{w}_j$ is high, then it must justify itself by reducing the loss function by the corresponding increase. For instance, let $\mathbf{w}_j$ be the original value and $\mathbf{w}'_j$ be the new value such that $\mathbf{w}'_j>\mathbf{w}_j$, then let $\Delta\mathbf{w}_j=\mathbf{w}'_j-\mathbf{w}_j>0$. The penalty for change from $\mathbf{w}_j$ to $\mathbf{w}'_j$ is $\Delta\mathbf{w}_j^2+2\mathbf{w}_j\Delta\mathbf{w}_j$. If the change is small, we can say that the penalty $\in\mathcal{O}(\mathbf{w}_j)$. Similarly, the corresponding reduction in the reconstruction loss $\in\mathcal{O}\left(\left(\mathbf{y}-\mathbf{X}\mathbf{w}\right)^T\mathbf{X}_{j}\right)$ where $\mathbf{X}_j$ is the $j^{th}$ column of the dataset $\mathbf{X}$. Increasing $\mathbf{w}_j$ by $\Delta\mathbf{w}_j$ will only happen if this reconstruction term compensates for the corresponding penalty.
2. **L$_0$ Regularization**: An alternate approach is to select a subset of features which can explain the data. In many datasets, there are lots of features. Since inference takes $\mathcal{O}(D)$ time (due to the inner product computation), reducing the number of features that is used for the linear regression model reduces the inference time. For instance, gene expression datasets have a very high number of features, and both training and inference time can benefit from the modeling using a subset of features. However, subset selection is *np-hard* and hence L$_0$-regularization is computationally intractable.
3.  **L$_1$ Regularization**: We can relax the  L$_0$ Regularization to  L$_1$ Regularization which gives us an approximate solution to the subset selection problem.
   $$
   \mathbf{w}_{L_1}^*=\arg\min_\mathbf{w}\sum_{i=1}^{N} \frac{1}{2}(y_i-\mathbf{w}^T\mathbf{x}_i)^2+\frac{\lambda}{2}\sum_{j=1}^D\mathbf{w}_j = \arg\min_\mathbf{w} \frac{1}{2}\Vert\mathbf{y}-\mathbf{X}\mathbf{w}\Vert_2^2+\frac{\lambda}{2}\Vert\mathbf{w}\Vert_1
   $$
   Implementation details: Upon convergence, $\mathbf{w}_{L_1}^*$ often has some dimensions where the values are extremely low $(<10^{-10})$. We can remove those features from the dataset as they are not very important and removing them reduces the size of the model, thus speeding up inference time when deployed.


----

## Logistic Regression

**insert image**

The logit function/transform:
$$
\sigma(y) = \frac{1}{1+\exp(-y)} = \frac{\exp(y)}{1+\exp(y)}
$$
The logistic regression is used for classification. Say we have two classes, and each items belong to these classes $\{0, 1\}$. We can transform the output of the linear model into a conditional probability mass corresponding to each class by applying the logit transform.
$$
p_\mathbf{w}(y=1|\mathbf{x}) = \frac{1}{1+\exp(-\mathbf{w}^T\mathbf{x})}
$$
and
$$
p_\mathbf{w}(y=0|\mathbf{x}) = 1-p_\mathbf{w}(y=1|\mathbf{x})=\frac{1}{1+\exp(\mathbf{w}^T\mathbf{x})}
$$
Therefore,
Likelihood of $y_i$ given $\mathbf{x}_i$: $p_\mathbf{w}(y_i|\mathbf{x}_i)=\sigma(\mathbf{w}^T\mathbf{x}_i)^{y_i}\cdot(1-\sigma(\mathbf{w}^T\mathbf{x}_i))^{(1-y_i)}$
Likelihood of dataset: $\prod_{i=1}^N p_\mathbf{w}(y_i|\mathbf{x}_i)$
Log Likelihood of dataset: $\log\prod_{i=1}^N p_\mathbf{w}(y_i|\mathbf{x}_i)=\sum_{i=1}^N \log p_\mathbf{w}(y_i|\mathbf{x}_i)=\sum_{i=1}^N y_i\log\sigma(\mathbf{w}^T\mathbf{x}_i)+(1-y_i)\log(1-\sigma(\mathbf{w}^T\mathbf{x}_i))$
Notice that in the log likelihood, when $y=1$ the first term is non-zero and the second term is zero (due to the multiplication in the front), and similarly when $y=0$, the second term is non-zero and the first term is zero.

Maximizing the log likelihood is equivalent to minimizing the negative of the log likelihood. Therefore:
**Negative log likelihood loss** (NLL Loss) $=-\sum_{i=1}^N \left[y_i\log\sigma(\mathbf{w}^T\mathbf{x}_i)+(1-y_i)(1-\log\sigma(\mathbf{w}^T\mathbf{x}_i))\right]$

This formulation of NLL Loss is also known as the Binary Cross Entropy Loss.
Replacing $\sigma(y)= \frac{1}{1+\exp(-y)}$, we get:
$$
\begin{align*}
Loss&=-\sum_{i=1}^N \left[y_i\log\frac{1}{1+\exp(-\mathbf{w}^T\mathbf{x}_i)}+(1-y_i)\log\left(1-\frac{1}{1+\exp(-\mathbf{w}^T\mathbf{x}_i)}\right)\right]\\
&=-\sum_{i=1}^N \left[y_i\log\frac{\exp(\mathbf{w}^T\mathbf{x}_i)}{1+\exp(\mathbf{w}^T\mathbf{x}_i)}+(1-y_i)\log\left(\frac{1}{1+\exp(\mathbf{w}^T\mathbf{x}_i)}\right)\right]\\
&=-\sum_{i=1}^N \left[y_i\log\exp(\mathbf{w}^T\mathbf{x}_i)-y_i\log{(1+\exp(\mathbf{w}^T\mathbf{x}_i))}-(1-y_i)\log(1+\exp(\mathbf{w}^T\mathbf{x}_i))\right]\\
&=-\sum_{i=1}^N \left[y_i\mathbf{w}^T\mathbf{x}_i-y_i\log{(1+\exp(\mathbf{w}^T\mathbf{x}_i))}-\log(1+\exp(\mathbf{w}^T\mathbf{x}_i))+y_i\log(1+\exp(\mathbf{w}^T\mathbf{x}_i))\right]\\
&=-\sum_{i=1}^N \left[y_i\mathbf{w}^T\mathbf{x}_i-\log(1+\exp(\mathbf{w}^T\mathbf{x}_i))\right]
\end{align*}
$$
**Regularization**:

Similar to linear regression, we can regularize logistic regression by constraining the weights as follows:
$$
Loss^{L_p}=-\sum_{i=1}^N \left[y_i\mathbf{w}^T\mathbf{x}_i-\log(1+\exp(\mathbf{w}^T\mathbf{x}_i))\right]+\lambda\Vert\mathbf{w}\Vert_p^p
$$

**Generalize to multi-class classification**:

Let there be $\vert C\vert$ number of classes in which a datapoint can belong. One way to extend a binary classifier to multiple classes is to simply train $\vert C\vert$ different parallel models each modeling the probability of the datapoints belonging to one of the $C$ classes. We can relabel the $\mathbf{y}$ such that $y'_i=1$ if $y_i=c$ else $y'_i=0$, and then train a binary logistic regression classifier using the new $\mathbf{y}'$ as the ground truth.
During inference, we have $\vert C\vert$ different models. Notice that each model is a vector. So, we can indicate the $\vert C\vert$ linear models as a matrix $\mathbf{W}_{D\times \vert C\vert}$. Therefore upon performing $\mathbf{XW}$, we will get a $N\times C$ matrix as output. For each sample (row), we can simply predict the index corresponding to the maximum value as the class label.
This method is known as *One vs All* (OVA) or One vs Rest (OVR) classifier.

Additionally, for each row, we can compute the softmax of the outputs to obtain a probability distribution of the sample belonging to a given class. Let the output for $\mathbf{x}_i$ by the OVA model be $\hat{\mathbf{y}}$ which is a $\vert C\vert$ dimensional vector. Then the probability is:
$$
p_\mathbf{W}(y_i=c|\mathbf{x}_i)=\frac{\exp(\hat{\mathbf{y}}_c)}{\sum_{j=1}^{\vert C\vert}\exp(\hat{\mathbf{y}}_j)}
$$

----

## Additional Topics

### Weighted Linear Regression

Let's consider a dataset $\mathcal{D}=\{(\mathbf{x}_i,y_i,r_i)\}_{i=1}^N$, where $\mathbf{x}_i\in\mathbb{R}^D$ are the features, $y_i\in\mathbb{R}$ are the labels, and $r_i>0$ are the weighing factor, i.e. some datapoints are given more weights than others (similar to [[Reinforcement Learning/Policy Gradients|policy gradient]] algorithms in [[Reinforcement Learning/Index (Reinforcement Learning)|reinforcement learning]]). Then the ERM objective is:
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
Upon taking the gradient of the loss, we get:
$$
\begin{align*}
\nabla_w(\mathbf{Xw}-\mathbf{y})^T\mathbf{R}(\mathbf{Xw}-\mathbf{y}) &= 2\mathbf{X}^T\mathbf{R}\mathbf{X}\mathbf{w} - 2\mathbf{X}^T\mathbf{R}\mathbf{y}\\
&= 2\mathbf{X}^T\mathbf{R}(\mathbf{X}\mathbf{w}-\mathbf{y})
\end{align*}
$$
Equating to zero, we get:
$$
\begin{align*}
\mathbf{X}^T\mathbf{R}\mathbf{X}\mathbf{w} &= \mathbf{X}^T\mathbf{R}\mathbf{y}\\
\implies w^* &= (\mathbf{X}^T\mathbf{R}\mathbf{X})^{-1} \mathbf{X}^T\mathbf{R}\mathbf{y}
\end{align*}
$$
### Comments on NLL Loss

Let the label for a given datapoint $\mathbf{y}$ be a one hot vector with the $c^{th}$ index being $1$ if the label of $\mathbf{x}$ is $c$.
So, for the given datapoint $\{\mathbf{x},\mathbf{y}\}$, (cross entropy) loss is: 
$$
\sum_{j=1}^{\vert C\vert}\mathbf{y}_j\log p_\mathbf{w}(\mathbf{x})_j=\sum_{j=1}^{\vert C\vert}\mathbf{y}_j\log \frac{\exp(\mathbf{W}_j^T\mathbf{x})}{\sum_k\exp(\mathbf{W}_k^T\mathbf{x})}
$$
where $\mathbf{W}_j$ represents the $j^{th}$ column of the matrix $\mathbf{W}$, i.e. the $j^{th}$ linear model.
Note that entropy is defined as $\mathcal{H}(x)=p(x)\log p(x)$

Minimizing negative log likelihood is same as minimizing the KL divergence between the data distribution and the distribution predicted by the model.

Learned distribution: $p_\mathbf{W}(y|\mathbf{x})$
Data distribution: $p(y|\mathbf{x})$

KL Divergence:
$$
\begin{align*}
\mathbb{D}_{KL}(p||p_\mathbf{W}) &= \mathbb{E}_{p(y|\mathbf{x})} \left[ \log\frac{p(y|\mathbf{x})}{p_\mathbf{W}(y|\mathbf{x})} \right] \\
&= \sum p(y|\mathbf{x}) \log\frac{p(y|\mathbf{x})}{p_\mathbf{W}(y|\mathbf{x})}\\
&= \sum p(y|\mathbf{x}) \log p(y|\mathbf{x}) - \sum p(y|\mathbf{x}) \log p_\mathbf{W}(y|\mathbf{x})\\
\end{align*}
$$
The former term is the entropy which is independent of $\mathbf{W}$ and the latter term is the cross-entropy which is dependent on $\mathbf{W}$.
$$
\begin{align*}
\mathbf{W}^*&=\arg\min_\mathbf{W} \mathbb{D}_{KL}(p||p_\mathbf{W})\\
&=\arg\min_\mathbf{W}-\sum p(y|\mathbf{x}) \log p_\mathbf{W}(y|\mathbf{x})\\
&=\arg\min_\mathbf{W}-\sum \mathbb{1}\{y=j\} \log p_\mathbf{W}(y|\mathbf{x})\\
\end{align*}
$$
Which is essentially the NLL Loss. It is for this reason that the NLL Loss for categorical variables is also known as the Categorical *Cross Entropy Loss*. $p(y|\mathbf{x})$ can be written as the indicator since in the dataset we have no uncertainty, i.e. we know the ground truth $y$ for every $\mathbf{x}$.