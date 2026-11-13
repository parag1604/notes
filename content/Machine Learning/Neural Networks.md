---
title: Perceptrons and Neural Networks
tags:
  - neural-networks
  - perceptron
  - backpropagation
  - deep-learning
  - optimization
date: 2026-11-12
prev: "[[Non-parametric Models]]"
next: "[[Convolutional Neural Networks]]"
---

> **Prerequisites:** [[Linear Models]] (logistic and softmax regression, gradient descent, $L_2$ regularization), [[Support Vector Machines]] (hyperplanes, the hinge loss, kernels), the [[Multivariable Calculus#The Jacobian and the chain rule|multivariable chain rule]] and [[First-Order Optimization]] (study both first if they are new), and [[Random Variables|variance]].
>
> **TL;DR:**
> - The **perceptron** is stochastic gradient descent on logistic regression with the probability replaced by a hard prediction. It only updates on mistakes, and on linearly separable data it stops after at most $(R/\gamma)^2$ of them.
> - A single hyperplane cannot solve XOR. Either fix a non-linear feature map (kernels, SVMs) or **learn** the features with a **multilayer perceptron** (MLP): layers of linear maps separated by non-linear activations.
> - **Backpropagation** is the chain rule applied from the loss backwards. One backward pass gives the gradient with respect to *every* parameter at roughly the cost of a forward pass; autodiff libraries automate it on computational graphs.
> - One hidden layer is enough to approximate any continuous function (**universal approximation**), but that says nothing about how many units are needed or whether training finds them.
> - Making deep networks train well is a craft: careful initialization, ReLU-like activations, momentum and Adam, learning-rate schedules, weight decay, early stopping, dropout and normalization layers.
>
> **Code:** [Perceptron](https://github.com/parag1604/Perceptron): the learning rule, the mistake bound, pocket and XOR in numpy; [MLP-Backprop-in-NumPy](https://github.com/parag1604/MLP-Backprop-in-NumPy): hand-written backprop, gradient checking, momentum and Adam; [Scalar-Autograd](https://github.com/parag1604/Scalar-Autograd): a tiny reverse-mode autodiff engine; [NN-from-Scratch-to-PyTorch](https://github.com/parag1604/NN-from-Scratch-to-PyTorch) and [NN-from-Scratch-to-JAX](https://github.com/parag1604/NN-from-Scratch-to-JAX): from numpy to a framework; [Basic-Image-Classification](https://github.com/parag1604/Basic-Image-Classification): MLPs and CNNs on MNIST, Fashion-MNIST and CIFAR-10.

## Motivation

Every classifier we have met so far ends in a hyperplane. [[Linear Models#Logistic Regression|Logistic regression]], [[Discriminant Analysis|LDA]] and the linear [[Support Vector Machines|SVM]] all compute a score $\mathbf{w}^\top\mathbf{x}+b$ and threshold it. They differ only in how they choose $\mathbf{w}$ and $b$. When the classes are not linearly separable in the raw features, we have so far had one answer: pick a feature map $\phi(\mathbf{x})$ by hand, or implicitly through a kernel, and draw the hyperplane in feature space.

This lecture takes the other road: **learn the feature map from data**. We start from the oldest learning machine of all, Rosenblatt's perceptron,[^rosenblatt1958] derive it as a stripped-down version of logistic regression, prove that it works when a separating hyperplane exists, and see exactly where it breaks (XOR). Stacking perceptron-like units into layers fixes the problem, and the rest of the lecture is about how to train such stacks: backpropagation, what they can represent, and the practical tricks that make deep networks trainable. For book-length treatments of everything here, see Goodfellow, Bengio and Courville,[^goodfellow2016] Prince,[^prince2023] and Bishop and Bishop.[^bishop2024]

## The perceptron

### From logistic regression to a mistake-driven rule

Recall [[Linear Models#Maximum likelihood = binary cross-entropy|logistic regression]] with labels $y_i\in\{0,1\}$ and $\mu_i=\sigma(\mathbf{w}^\top\mathbf{x}_i)$ (absorb the bias into $\mathbf{w}$ by appending a constant 1 to $\mathbf{x}$). The gradient of the negative log-likelihood is

$$
\nabla_{\mathbf{w}}\mathcal{L}(\mathbf{w})=\sum_{i=1}^{N}(\mu_i-y_i)\,\mathbf{x}_i ,
$$

and gradient descent with step size $\alpha$ repeats $\mathbf{w}\leftarrow\mathbf{w}-\alpha\sum_i(\mu_i-y_i)\mathbf{x}_i$. Three small changes turn this into the perceptron.

1. **One point at a time.** Each step needs all $N$ points. Use one randomly chosen point instead, which is [[Linear Models#Iterative methods|stochastic gradient descent]]: $\mathbf{w}\leftarrow\mathbf{w}-\alpha(\mu_i-y_i)\mathbf{x}_i$.
2. **Hard predictions.** Replace the probability $\mu_i$ by the predicted label $\hat y_i=\mathbb{1}[\mathbf{w}^\top\mathbf{x}_i\ge0]$, which is the same as $\mathbb{1}[\mu_i\ge\tfrac12]$. Now $\hat y_i-y_i=0$ whenever the point is classified correctly, so the update $\mathbf{w}\leftarrow\mathbf{w}-\alpha(\hat y_i-y_i)\mathbf{x}_i$ is **mistake-driven**: the weights change only on errors.
3. **Labels in $\{-1,+1\}$.** With $\hat y_i=\operatorname{sign}(\mathbf{w}^\top\mathbf{x}_i)$ and $y_i\in\{\pm1\}$, a mistake means $\hat y_i=-y_i$, so $\hat y_i-y_i=-2y_i$ and

$$
\mathbf{w}\leftarrow\mathbf{w}-\alpha(\hat y_i-y_i)\,\mathbf{x}_i=\mathbf{w}+2\alpha\,y_i\mathbf{x}_i .
$$

Starting from $\mathbf{w}=\mathbf{0}$, every iterate is a sum of terms $2\alpha y_i\mathbf{x}_i$, so changing $\alpha$ rescales all iterates by the same positive factor and never changes a prediction $\operatorname{sign}(\mathbf{w}^\top\mathbf{x})$. We may therefore take $2\alpha=1$. Writing the bias out again, this is the **perceptron learning rule**:

> [!example] Algorithm: the perceptron (Rosenblatt, 1958)
> Data $\{(\mathbf{x}_i,y_i)\}_{i=1}^N$ with $y_i\in\{-1,+1\}$. Initialize $\mathbf{w}=\mathbf{0}$, $b=0$.
> **Repeat** for a fixed number of epochs, or until an epoch makes no mistake:
> 1. Visit the points in random order. For each $(\mathbf{x}_i,y_i)$:
> 2. If $y_i(\mathbf{w}^\top\mathbf{x}_i+b)\le0$ (a mistake, or a point on the boundary):
> 3. $\quad\mathbf{w}\leftarrow\mathbf{w}+y_i\mathbf{x}_i$ and $b\leftarrow b+y_i$.
>
> Predict with $\hat y(\mathbf{u})=\operatorname{sign}(\mathbf{w}^\top\mathbf{u}+b)$.

Why does the update help? After an update on $(\mathbf{x}_i,y_i)$, the score of that point becomes

$$
y_i(\mathbf{w}+y_i\mathbf{x}_i)^\top\mathbf{x}_i=y_i\mathbf{w}^\top\mathbf{x}_i+\lVert\mathbf{x}_i\rVert^2 ,
$$

so its margin goes up by $\lVert\mathbf{x}_i\rVert^2$ (with the bias, by $\lVert\mathbf{x}_i\rVert^2+1$). One update need not fix the point, and it can break others, but it always pushes the hyperplane the right way for the point at hand.

> [!note] The same rule as SGD on a loss
> Define the **perceptron loss** $\ell(\mathbf{w};\mathbf{x},y)=\max(0,\,-y\,\mathbf{w}^\top\mathbf{x})$. It is zero for correctly classified points and grows linearly with the violation. A subgradient is $-y\mathbf{x}$ when $y\mathbf{w}^\top\mathbf{x}\le0$ (choosing that branch at the kink) and $\mathbf{0}$ otherwise, so SGD with step 1 *is* the perceptron rule. Compare the SVM's [[Support Vector Machines#The hinge loss: SVM vs. logistic regression|hinge loss]] $\max(0,1-y\,\mathbf{w}^\top\mathbf{x})$: the perceptron loss is a hinge with the corner moved from margin 1 to margin 0, and without the $\lVert\mathbf{w}\rVert^2$ penalty. Two consequences:
> - $\mathbf{w}=\mathbf{0}$ has zero perceptron loss, which is why the algorithm counts $y_i\mathbf{w}^\top\mathbf{x}_i=0$ as a mistake. Otherwise it would never leave its starting point.
> - The perceptron stops at *any* separating hyperplane, often one that grazes the data. The SVM insists on the widest margin.

### Hyperplane geometry

The decision boundary is the hyperplane $\{\mathbf{x}:\mathbf{w}^\top\mathbf{x}+b=0\}$. It splits $\mathbb{R}^D$ into the positive half-space $\mathbf{w}^\top\mathbf{x}+b>0$ (predicted $+1$) and the negative one.

- $\mathbf{w}$ is **normal** to the hyperplane: for two points $\mathbf{x},\mathbf{x}'$ on it, $\mathbf{w}^\top(\mathbf{x}-\mathbf{x}')=0$.
- The signed distance of a point $\mathbf{x}$ from the hyperplane, measured along the unit normal $\mathbf{w}/\lVert\mathbf{w}\rVert$, is $(\mathbf{w}^\top\mathbf{x}+b)/\lVert\mathbf{w}\rVert$.
- The point of the hyperplane closest to the origin is $\mathbf{x}_0=-b\,\mathbf{w}/\lVert\mathbf{w}\rVert^2$. So the hyperplane sits at signed distance $-b/\lVert\mathbf{w}\rVert$ from the origin along $\mathbf{w}$. Increasing $b$ slides it parallel to itself in the $-\mathbf{w}$ direction, leaving the origin further inside the positive side (where $\mathbf{w}^\top\mathbf{0}+b=b>0$).

### The perceptron convergence theorem

Absorb the bias by writing $\tilde{\mathbf{x}}=(\mathbf{x},1)$ and $\tilde{\mathbf{w}}=(\mathbf{w},b)$, so that the rule is $\tilde{\mathbf{w}}\leftarrow\tilde{\mathbf{w}}+y_i\tilde{\mathbf{x}}_i$. The data are **linearly separable with margin $\gamma>0$** if some unit vector $\mathbf{u}$ ($\lVert\mathbf{u}\rVert=1$) satisfies $y_i\,\mathbf{u}^\top\tilde{\mathbf{x}}_i\ge\gamma$ for all $i$.

> [!tip] Theorem (Block, 1962; Novikoff, 1962)
> Let $\lVert\tilde{\mathbf{x}}_i\rVert\le R$ for all $i$, and suppose the data are linearly separable with margin $\gamma>0$. Then the perceptron started at $\tilde{\mathbf{w}}=\mathbf{0}$ makes **at most $(R/\gamma)^2$ mistakes**, whatever order it visits the points in. In particular it stops after finitely many epochs at a hyperplane that separates the training data.

The proof is short and worth knowing.[^block1962]<sup>,</sup>[^novikoff1962] It squeezes $\tilde{\mathbf{w}}$ between two bounds: its component along $\mathbf{u}$ grows *linearly* in the number of mistakes, while its length grows only like a *square root*.

> [!proof]- Proof of the mistake bound
> Let $\tilde{\mathbf{w}}_k$ be the weights after the $k$-th mistake, $\tilde{\mathbf{w}}_0=\mathbf{0}$, and let the $k$-th mistake happen on $(\tilde{\mathbf{x}},y)$, so $\tilde{\mathbf{w}}_k=\tilde{\mathbf{w}}_{k-1}+y\tilde{\mathbf{x}}$.
>
> **Progress along $\mathbf{u}$.** By the margin assumption,
>
> $$
> \mathbf{u}^\top\tilde{\mathbf{w}}_k=\mathbf{u}^\top\tilde{\mathbf{w}}_{k-1}+y\,\mathbf{u}^\top\tilde{\mathbf{x}}\ge\mathbf{u}^\top\tilde{\mathbf{w}}_{k-1}+\gamma ,
> $$
>
> and by induction $\mathbf{u}^\top\tilde{\mathbf{w}}_k\ge k\gamma$.
>
> **Slow growth of the norm.** A mistake means $y\,\tilde{\mathbf{w}}_{k-1}^\top\tilde{\mathbf{x}}\le0$, so
>
> $$
> \begin{aligned}
> \lVert\tilde{\mathbf{w}}_k\rVert^2&=\lVert\tilde{\mathbf{w}}_{k-1}\rVert^2+2y\,\tilde{\mathbf{w}}_{k-1}^\top\tilde{\mathbf{x}}+\lVert\tilde{\mathbf{x}}\rVert^2\\
> &\le\lVert\tilde{\mathbf{w}}_{k-1}\rVert^2+R^2 ,
> \end{aligned}
> $$
>
> and by induction $\lVert\tilde{\mathbf{w}}_k\rVert^2\le kR^2$.
>
> **Combine.** By Cauchy–Schwarz and $\lVert\mathbf{u}\rVert=1$,
>
> $$
> k\gamma\le\mathbf{u}^\top\tilde{\mathbf{w}}_k\le\lVert\tilde{\mathbf{w}}_k\rVert\le\sqrt{k}\,R ,
> $$
>
> so $\sqrt{k}\le R/\gamma$, that is $k\le(R/\gamma)^2$. Since every mistake is counted, there can be no more than $(R/\gamma)^2$ updates in total; after the last one, a full epoch passes without mistakes. $\blacksquare$

Remarks:

- The bound does not depend on $N$ or on the dimension $D$, only on the geometry: the radius of the data and the margin. The best possible $\gamma$ is exactly the margin of the hard-margin SVM through the origin on the points $\tilde{\mathbf{x}}_i$, which makes the perceptron a slow, approximate cousin of the [[Support Vector Machines|SVM]].
- The bound is a worst case and usually very loose. In Figure 1 (left) the perceptron finishes after 12 mistakes while $(R/\gamma)^2\approx3461$, because the data sit away from the origin and the appended constant 1 inflates $R$ relative to $\gamma$. Centering the data shrinks the bound.
- The theorem says nothing about *which* separator is found, and the perceptron's answer depends on the visiting order.

![[Machine Learning/images/neural-networks-perceptron.svg]]
*Figure 1: left, the perceptron on two separable Gaussian blobs, starting from $\mathbf{w}=\mathbf{0}$. Grey lines are the boundaries after each of the 12 mistakes (darker is later); green is the final separator, reached in 3 epochs. Right, on two overlapping blobs (200 points) there is no separator: the training errors of the current weights keep jumping between 26 and 98 from epoch to epoch, while the pocket algorithm's best-so-far weights settle at 23 errors.*

### When the data are not separable

Real data are rarely linearly separable. Then every hyperplane misclassifies some point, the perceptron never completes a clean epoch, and the weights wander forever (they stay bounded and cycle, but never settle). Figure 1 (right) shows the result: the quality of the *last* iterate is essentially random. Two simple fixes:

- **Pocket algorithm** (Gallant, 1990):[^gallant1990] run the perceptron, but keep "in your pocket" the weights with the fewest training errors seen so far, and return those.
- **Averaged (or voted) perceptron** (Freund and Schapire, 1999):[^freund1999] return the average of all iterates, weighted by how long each survived. The averaging smooths out the oscillation, comes with generalization guarantees in terms of the margin, and in practice behaves much like a linear SVM.

Both stop after a fixed number of epochs. Neither changes the fundamental limit: the model is still one hyperplane.

### XOR: what one hyperplane cannot do

The XOR function labels $(0,1)$ and $(1,0)$ as class 1 and $(0,0)$, $(1,1)$ as class 0 (Figure 3, left, in the next section). No line separates them. Suppose $w_1x_1+w_2x_2+b$ were positive exactly on class 1:

- class 0: $b\le0$ and $w_1+w_2+b\le0$; adding these, $w_1+w_2+2b\le0$;
- class 1: $w_1+b>0$ and $w_2+b>0$; adding these, $w_1+w_2+2b>0$.

The two conclusions contradict each other. Minsky and Papert made limitations of this kind famous,[^minsky1969] and the observation that single-layer perceptrons cannot compute such simple predicates cooled interest in neural networks for more than a decade. There are two ways out.

1. **Fix a non-linear feature map.** Adding the feature $x_1x_2$ (after centering) makes XOR separable. Choosing features by hand does not scale, but choosing them implicitly through a kernel does: the perceptron's weights are a sum of $y_i\mathbf{x}_i$ terms, so it can be kernelized exactly like the SVM. This road leads to [[Support Vector Machines#Kernels|kernel methods]].
2. **Learn the feature map.** Put a layer of perceptron-like units in front of the final one and train everything together. This is the **multilayer perceptron**, and the road to deep learning.

## Multilayer perceptrons

### Layers and activations

A multilayer perceptron (MLP, also called a fully connected feed-forward network) alternates affine maps with a fixed scalar non-linearity $\phi$ applied element-wise. With $\mathbf{h}^{(0)}=\mathbf{x}\in\mathbb{R}^{D}$ and layers $\ell=1,\dots,L$,

$$
\mathbf{z}^{(\ell)}=\mathbf{W}^{(\ell)}\mathbf{h}^{(\ell-1)}+\mathbf{b}^{(\ell)},\qquad\mathbf{h}^{(\ell)}=\phi\big(\mathbf{z}^{(\ell)}\big),
$$

where $\mathbf{W}^{(\ell)}\in\mathbb{R}^{n_\ell\times n_{\ell-1}}$ and $n_\ell$ is the **width** of layer $\ell$. The last layer is left without $\phi$: its output $\mathbf{z}^{(L)}=f_\theta(\mathbf{x})$ (the **logits**) goes into an output non-linearity inside the loss. Each row of $\mathbf{W}^{(\ell)}$ together with its bias is one "neuron": a perceptron with a soft threshold. $L$ is the **depth**, and $\theta$ collects all weights and biases.

**Why the non-linearity is essential.** Without $\phi$, two layers collapse into one:

$$
\mathbf{W}^{(2)}\big(\mathbf{W}^{(1)}\mathbf{x}+\mathbf{b}^{(1)}\big)+\mathbf{b}^{(2)}=\underbrace{\mathbf{W}^{(2)}\mathbf{W}^{(1)}}_{\text{a matrix}}\mathbf{x}+\underbrace{\mathbf{W}^{(2)}\mathbf{b}^{(1)}+\mathbf{b}^{(2)}}_{\text{a vector}} .
$$

A composition of affine maps is affine, so any depth of purely linear layers is still a single hyperplane classifier.

The common activations (Figure 2):

- **Sigmoid** $\sigma(z)=1/(1+e^{-z})$: the smooth step from [[Linear Models#From scores to probabilities|logistic regression]]. Its derivative $\sigma'(z)=\sigma(z)(1-\sigma(z))$ is at most $1/4$ and nearly zero for large $|z|$ ("saturation"), which makes deep sigmoid networks hard to train.
- **Hyperbolic tangent** $\tanh(z)=2\sigma(2z)-1$: a rescaled sigmoid, zero-centred, with $\tanh'(z)=1-\tanh^2(z)\le1$. It still saturates.
- **Rectified linear unit** $\mathrm{ReLU}(z)=\max(0,z)$: derivative 1 for $z>0$ and 0 for $z<0$. It does not saturate for positive inputs, is cheap, and is the default for hidden layers. A unit whose input is negative for every example gets zero gradient and can "die"; variants such as the leaky ReLU $\max(0.01z,z)$ or the smooth GELU avoid this.

![[Machine Learning/images/neural-networks-activations.svg]]
*Figure 2: left, the sigmoid, tanh and ReLU activations. Right, their derivatives. The sigmoid's derivative never exceeds 0.25 and both sigmoid and tanh flatten out for large $|z|$, which shrinks gradients as they pass back through many layers. ReLU passes gradients unchanged wherever it is active.*

### XOR with one hidden layer

Two ReLU units are enough for XOR.[^goodfellow2016] Take

$$
h_1=\mathrm{ReLU}(x_1+x_2),\qquad h_2=\mathrm{ReLU}(x_1+x_2-1),\qquad f(\mathbf{x})=h_1-2h_2 .
$$

The four inputs map to $(h_1,h_2)=(0,0)$, $(1,0)$, $(1,0)$ and $(2,1)$, and the outputs are $0,1,1,0$: XOR exactly. The hidden layer has moved the points so that the two class-1 inputs land on the same spot and a single line now separates the classes (Figure 3, right). The output unit is just a linear classifier in the **learned feature space**. Training discovers such features from data instead of us writing them down; that is the whole idea.

![[Machine Learning/images/neural-networks-xor.svg]]
*Figure 3: left, XOR in input space; circles are class 1 and squares class 0. The network $f=h_1-2h_2$ classifies the shaded blue band as class 1, with two parallel boundary lines (green), something no single hyperplane can do. Right, the same four points in the hidden space $(h_1,h_2)$, where $(0,1)$ and $(1,0)$ coincide and one line, $h_1-2h_2=0.5$, separates the classes.*

### Output layers and losses

The last layer of an MLP is a [[Linear Models|generalized linear model]] applied to the learned features $\mathbf{h}^{(L-1)}$. So the output non-linearity and the loss are chosen exactly as in Linear Models, as a negative log-likelihood:

| Task | Output | Loss (NLL) |
|---|---|---|
| Regression, $y\in\mathbb{R}$ | identity, $\hat y=z$ | squared error $\tfrac12(\hat y-y)^2$ (Gaussian noise) |
| Binary, $y\in\{0,1\}$ | sigmoid, $\hat y=\sigma(z)$ | binary cross-entropy |
| $K$ classes, one-hot $\mathbf{y}$ | softmax, $\hat{\mathbf{y}}=\operatorname{softmax}(\mathbf{z})$ | cross-entropy $-\sum_k y_k\log\hat y_k$ |

In all three cases the gradient of the loss with respect to the logits takes the same simple form,

$$
\frac{\partial\ell}{\partial\mathbf{z}^{(L)}}=\hat{\mathbf{y}}-\mathbf{y},
$$

"prediction minus target". This is the starting point of backpropagation below. (In code, combine the last two steps, e.g. `BCEWithLogitsLoss` or `cross_entropy` on logits, which evaluate the loss stably via log-sum-exp, as in [[Linear Models#More than two classes: softmax regression|softmax regression]].)

So an MLP for classification is "logistic regression on learned features", and training it is still empirical risk minimization of the average loss $\mathcal{L}(\theta)=\frac1N\sum_i\ell(f_\theta(\mathbf{x}_i),y_i)$, now over all layers at once. The catch is that $\mathcal{L}$ is no longer convex in $\theta$, and we need its gradient with respect to every weight in every layer.

## Backpropagation

### The chain rule, layer by layer

Fix one training example and write $\ell$ for its loss. Define the **error signal** of layer $\ell$ as the gradient of the loss with respect to that layer's pre-activations,

$$
\boldsymbol\delta^{(\ell)}=\frac{\partial\ell}{\partial\mathbf{z}^{(\ell)}}\in\mathbb{R}^{n_\ell}.
$$

Everything follows from two observations.

**Weights from errors.** Unit $j$ of layer $\ell$ computes $z^{(\ell)}_j=\sum_kW^{(\ell)}_{jk}h^{(\ell-1)}_k+b^{(\ell)}_j$, and $W^{(\ell)}_{jk}$ affects the loss only through $z^{(\ell)}_j$. Hence $\partial\ell/\partial W^{(\ell)}_{jk}=\delta^{(\ell)}_j\,h^{(\ell-1)}_k$, or in matrix form

$$
\frac{\partial\ell}{\partial\mathbf{W}^{(\ell)}}=\boldsymbol\delta^{(\ell)}\,\mathbf{h}^{(\ell-1)\top},\qquad\frac{\partial\ell}{\partial\mathbf{b}^{(\ell)}}=\boldsymbol\delta^{(\ell)} .
$$

The gradient of a weight is "error at its output times activity at its input", an outer product.

**Errors from the layer above.** $\mathbf{z}^{(\ell)}$ affects the loss only through $\mathbf{z}^{(\ell+1)}=\mathbf{W}^{(\ell+1)}\phi(\mathbf{z}^{(\ell)})+\mathbf{b}^{(\ell+1)}$, whose Jacobian is $\mathbf{W}^{(\ell+1)}\operatorname{diag}\big(\phi'(\mathbf{z}^{(\ell)})\big)$. The multivariable chain rule multiplies by the transposed Jacobian:

$$
\begin{aligned}
\boldsymbol\delta^{(\ell)}&=\Big(\frac{\partial\mathbf{z}^{(\ell+1)}}{\partial\mathbf{z}^{(\ell)}}\Big)^{\top}\boldsymbol\delta^{(\ell+1)}\\
&=\phi'\big(\mathbf{z}^{(\ell)}\big)\odot\Big(\mathbf{W}^{(\ell+1)\top}\boldsymbol\delta^{(\ell+1)}\Big) .
\end{aligned}
\tag{1}
$$

Errors flow backwards through the *transposed* weights and are gated by the local slope $\phi'$. With $\boldsymbol\delta^{(L)}=\hat{\mathbf{y}}-\mathbf{y}$ from the output layer, equation (1) gives every $\boldsymbol\delta^{(\ell)}$, and the first observation turns them into weight gradients. This is **backpropagation** (Rumelhart, Hinton and Williams, 1986).[^rumelhart1986]

> [!example] Algorithm: backpropagation for an MLP (one example)
> 1. **Forward:** for $\ell=1,\dots,L$ compute and *store* $\mathbf{z}^{(\ell)}$ and $\mathbf{h}^{(\ell)}$; then the loss $\ell$.
> 2. **Output error:** $\boldsymbol\delta^{(L)}=\hat{\mathbf{y}}-\mathbf{y}$ (for a matched output and loss; otherwise $\partial\ell/\partial\mathbf{z}^{(L)}$ directly).
> 3. **Backward:** for $\ell=L,\dots,1$: set $\nabla_{\mathbf{W}^{(\ell)}}\ell=\boldsymbol\delta^{(\ell)}\mathbf{h}^{(\ell-1)\top}$ and $\nabla_{\mathbf{b}^{(\ell)}}\ell=\boldsymbol\delta^{(\ell)}$; if $\ell>1$, compute $\boldsymbol\delta^{(\ell-1)}$ from equation (1).
>
> For a mini-batch, average the per-example gradients. Stacking examples as rows, $\boldsymbol\Delta^{(\ell)}\in\mathbb{R}^{B\times n_\ell}$ and $\mathbf{H}^{(\ell-1)}\in\mathbb{R}^{B\times n_{\ell-1}}$, the weight gradient is the single matrix product $\boldsymbol\Delta^{(\ell)\top}\mathbf{H}^{(\ell-1)}/B$.

The cost is easy to read off: the backward pass does one matrix–vector product per layer with $\mathbf{W}^{(\ell)\top}$ and one outer product, mirroring the forward pass. So **all** gradients cost about as much as one or two forward passes, however many parameters there are. The price is memory: the forward activations must be stored until the backward pass uses them.

> [!example]- Code: backpropagation, gradient checking and Adam in numpy (XOR)
> ```python
> import numpy as np
>
> sigmoid = lambda z: 1 / (1 + np.exp(-z))
>
> def init(D, H, seed=0):
>     rng = np.random.default_rng(seed)
>     return {"W1": rng.normal(0, np.sqrt(1 / D), (H, D)), "b1": np.zeros(H),
>             "W2": rng.normal(0, np.sqrt(1 / H), (1, H)), "b2": np.zeros(1)}
>
> def loss_and_grads(p, X, y):
>     """One hidden tanh layer, sigmoid output, mean binary cross-entropy."""
>     N = len(y)
>     # forward pass (rows of X are examples, so x^T W^T replaces W x)
>     Z1 = X @ p["W1"].T + p["b1"];  H = np.tanh(Z1)
>     Z2 = H @ p["W2"].T + p["b2"];  P = sigmoid(Z2[:, 0])
>     loss = -np.mean(y * np.log(P) + (1 - y) * np.log(1 - P))
>     # backward pass: delta = dLoss/dz, layer by layer
>     d2 = (P - y)[:, None] / N                    # output layer: sigmoid + BCE
>     d1 = (d2 @ p["W2"]) * (1 - H**2)             # back through W2, then tanh'
>     grads = {"W2": d2.T @ H, "b2": d2.sum(0),
>              "W1": d1.T @ X, "b1": d1.sum(0)}
>     return loss, grads
>
> def grad_check(p, X, y, eps=1e-6):
>     """Largest relative error between backprop and central differences."""
>     _, g = loss_and_grads(p, X, y); worst = 0.0
>     for k in p:
>         for i in np.ndindex(p[k].shape):
>             old = p[k][i]
>             p[k][i] = old + eps; lp, _ = loss_and_grads(p, X, y)
>             p[k][i] = old - eps; lm, _ = loss_and_grads(p, X, y)
>             p[k][i] = old
>             num = (lp - lm) / (2 * eps)
>             worst = max(worst, abs(num - g[k][i]) / max(1e-8, abs(num) + abs(g[k][i])))
>     return worst
>
> def adam(p, X, y, lr=0.05, steps=2000, b1=0.9, b2=0.999, eps=1e-8):
>     m = {k: np.zeros_like(v) for k, v in p.items()}
>     s = {k: np.zeros_like(v) for k, v in p.items()}
>     for t in range(1, steps + 1):
>         loss, g = loss_and_grads(p, X, y)
>         for k in p:
>             m[k] = b1 * m[k] + (1 - b1) * g[k]          # first moment
>             s[k] = b2 * s[k] + (1 - b2) * g[k] ** 2     # second moment
>             mh, sh = m[k] / (1 - b1**t), s[k] / (1 - b2**t)
>             p[k] -= lr * mh / (np.sqrt(sh) + eps)
>     return p, loss
>
> X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
> y = np.array([0, 1, 1, 0], float)                     # XOR
> p = init(D=2, H=4)
> print("grad check, max relative error:", grad_check(p, X, y))
> p, loss = adam(p, X, y)
> predict = lambda p, X: sigmoid(np.tanh(X @ p["W1"].T + p["b1"]) @ p["W2"].T + p["b2"])[:, 0]
> print("final loss:", loss, " predictions:", (predict(p, X) > 0.5).astype(int))
> ```
> The gradient check reports a maximum relative error of about $3\times10^{-7}$, and after 2000 Adam steps the loss is about $6\times10^{-5}$ with predictions `[0 1 1 0]`. The backprop gradients also agree with PyTorch's autograd to $10^{-10}$ in double precision.

### Computational graphs and reverse-mode autodiff

Deriving equation (1) by hand for every new architecture would be tedious and error-prone. **Automatic differentiation** does it mechanically.[^baydin2018] Any program that computes the loss is a composition of primitive operations (add, multiply, matrix product, $\exp$, $\tanh$, …), which forms a directed acyclic **computational graph**. Each primitive knows its own local derivative. For every node $v$, define its **adjoint** $\bar v=\partial\ell/\partial v$. The chain rule says that $\bar v$ collects contributions from every node $c$ that uses $v$:

$$
\bar v=\sum_{c\,:\,v\to c}\bar c\,\frac{\partial c}{\partial v},\qquad\bar\ell=1 .
$$

Visit the nodes in reverse topological order, starting from the loss, and every adjoint is ready when needed. Figure 4 does this for one sigmoid neuron with a squared-error loss.

![[Machine Learning/images/neural-networks-graph.svg]]
*Figure 4: a computational graph for $L=\frac12(\sigma(wx+b)-y)^2$ at $x=1.5$, $w=0.8$, $b=-0.2$, $y=1$. The forward pass (blue, above each node) computes $u=1.2$, $z=1.0$, $a=0.7311$, $L=0.0362$. The backward pass (vermilion, below) starts from $\bar L=1$: $\bar a=a-y=-0.2689$, $\bar z=\bar a\,a(1-a)=-0.0529$, which passes unchanged through the sum to $\bar u$ and $\bar b$, and then $\bar w=\bar u\,x=-0.0793$ and $\bar x=\bar u\,w=-0.0423$.*

Two details matter in practice:

- **Accumulate, don't overwrite.** If a node feeds several others (a weight shared across time steps, or simply $y=x+x$), its adjoint is a *sum*. Autodiff engines therefore add into `.grad` (`+=`), and you must zero the gradients before each new backward pass; this is why PyTorch training loops call `zero_grad()`.
- **Why reverse mode.** The chain rule can also be applied forwards: **forward mode** propagates $\partial v/\partial\theta_j$ for one input direction $\theta_j$ at a time, so a gradient with respect to $P$ parameters needs $P$ passes. **Reverse mode** propagates $\partial\ell/\partial v$ for one *output* at a time, so a scalar loss needs a single backward pass, at a small constant multiple of the forward cost, for all $P$ derivatives. With $P$ in the millions or billions and one scalar loss, reverse mode is the only sensible choice, and backpropagation is exactly reverse-mode autodiff applied to a network. (Forward mode wins in the opposite case: few inputs, many outputs.)

> [!example]- Code: the same XOR network in PyTorch
> ```python
> import torch
>
> torch.manual_seed(0)
> X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
> y = torch.tensor([0., 1., 1., 0.])
> model = torch.nn.Sequential(torch.nn.Linear(2, 4), torch.nn.Tanh(), torch.nn.Linear(4, 1))
> opt = torch.optim.Adam(model.parameters(), lr=0.05)
> loss_fn = torch.nn.BCEWithLogitsLoss()        # sigmoid + BCE, numerically stable
>
> for step in range(2000):
>     opt.zero_grad()                           # gradients accumulate, so reset them
>     loss = loss_fn(model(X)[:, 0], y)         # forward pass builds the graph
>     loss.backward()                           # reverse-mode autodiff = backprop
>     opt.step()                                # Adam update of every parameter
> print(loss.item(), (model(X)[:, 0] > 0).int().tolist())
> ```
> Prints a loss of about $5\times10^{-5}$ and `[0, 1, 1, 0]`. Compare it line by line with the numpy version: the forward pass is the same, and `loss.backward()` replaces the entire hand-written backward pass.

### Gradient checking

A backward pass with a bug often still trains, just badly, so verify gradients numerically when you write one by hand. Perturb one parameter at a time and use the **central difference**

$$
\frac{\partial\mathcal{L}}{\partial\theta_j}\approx\frac{\mathcal{L}(\theta+\epsilon\mathbf{e}_j)-\mathcal{L}(\theta-\epsilon\mathbf{e}_j)}{2\epsilon},
$$

whose error is $O(\epsilon^2)$, against the one-sided difference's $O(\epsilon)$. Compare with the relative error $|g_{\text{num}}-g_{\text{bp}}|/(|g_{\text{num}}|+|g_{\text{bp}}|)$. In double precision with $\epsilon\approx10^{-6}$, values around $10^{-7}$ or smaller mean the gradient is right; around $10^{-2}$ means a bug. Check on a small network and a few examples, because each parameter costs two forward passes. Kinks such as ReLU at 0 can cause isolated false alarms.

> [!tip] Aside: gradients in pictures, and why the loss is not convex
> **The gradient is perpendicular to level sets.** Along any curve $\mathbf{c}(t)$ inside a level set, $f(\mathbf{c}(t))$ is constant, so $\frac{d}{dt}f(\mathbf{c}(t))=\nabla f^\top\mathbf{c}'(t)=0$: the gradient is orthogonal to every direction within the level set.
>
> **It points in the direction of steepest ascent.** The rate of change in a unit direction $\mathbf{v}$ is $\nabla f^\top\mathbf{v}$, which by Cauchy–Schwarz is largest for $\mathbf{v}=\nabla f/\lVert\nabla f\rVert$. Gradient descent walks straight downhill across the contours (Figure 5, left).
>
> **A zero gradient is not a minimum.** For $f(x,y)=-x^2+y^2$ the gradient vanishes at the origin, but $f$ decreases along $x$ and increases along $y$: a **saddle point**, with an indefinite Hessian $\operatorname{diag}(-2,2)$ (Figure 5, right).
>
> **Neural-network losses are not convex.** Permuting the hidden units of a layer (with their incoming and outgoing weights) gives a different $\theta$ that computes the same function, so every minimum comes with $n_\ell!$ copies. The average of two such copies is generally *not* a minimum, which a convex function would require. In high dimensions most critical points turn out to be saddles rather than bad local minima,[^dauphin2014] and the noise in SGD helps to escape them. In practice, large networks trained with SGD reliably reach low training loss, though there is no general guarantee.

![[Machine Learning/images/neural-networks-gradients.svg]]
*Figure 5: normalized gradient directions (arrows) on the contours of a bowl $x^2+3y^2$ (left) and a saddle $-x^2+y^2$ (right). The arrows cross each contour at a right angle and point uphill. Both functions have zero gradient at the origin, but only the bowl has a minimum there.*

## What can a neural network represent?

### Universal approximation

> [!tip] Theorem (universal approximation)
> Let $\phi:\mathbb{R}\to\mathbb{R}$ be continuous and **not a polynomial**. For every compact set $K\subset\mathbb{R}^D$, every continuous $f:K\to\mathbb{R}$ and every $\varepsilon>0$, there are a width $H$ and parameters $a_j,b_j\in\mathbb{R}$, $\mathbf{w}_j\in\mathbb{R}^D$ such that the one-hidden-layer network $g(\mathbf{x})=\sum_{j=1}^{H}a_j\,\phi(\mathbf{w}_j^\top\mathbf{x}+b_j)$ satisfies $\sup_{\mathbf{x}\in K}|f(\mathbf{x})-g(\mathbf{x})|<\varepsilon$.

Cybenko proved this for sigmoid-shaped activations,[^cybenko1989] Hornik for any bounded non-constant activation,[^hornik1991] and Leshno, Lin, Pinkus and Schocken showed that "not a polynomial" is exactly the right condition (for a broad class of activations that includes ReLU).[^leshno1993] The polynomial exclusion makes sense: if $\phi$ is a polynomial of degree $m$, then so is every $g$, and degree-$m$ polynomials are a finite-dimensional family that cannot approximate everything.

**Intuition in one dimension.** With ReLUs the theorem is almost obvious. Three ReLUs make a "hat" (Figure 6, left):

$$
\frac{\mathrm{ReLU}(x-0.3)-2\,\mathrm{ReLU}(x-0.5)+\mathrm{ReLU}(x-0.7)}{0.2}
$$

rises linearly from 0 at $x=0.3$ to 1 at $x=0.5$ and falls back to 0 at $x=0.7$. More generally, any piecewise-linear function with knots $x_0<x_1<\dots<x_H$ on $[x_0,x_H]$ is exactly a one-hidden-layer ReLU network:

$$
g(x)=f(x_0)+\sum_{k=0}^{H-1}c_k\,\mathrm{ReLU}(x-x_k),
$$

where $c_0$ is the slope of the first piece and each later $c_k$ is the *change* of slope at knot $x_k$. Interpolate $f$ at the knots, refine the grid, and uniform continuity of $f$ on the interval makes the error as small as we like. With 4 units the maximum error in Figure 6 is 0.21; with 16 it is 0.019.

![[Machine Learning/images/neural-networks-uat.svg]]
*Figure 6: left, three shifted and scaled ReLUs (dashed) add up to a localized bump. Right, a one-hidden-layer ReLU network with $H$ units is a piecewise-linear interpolant of the target $f(x)=\sin(2\pi x)+0.6x$ with $H$ pieces; the maximum error falls from 0.21 with 4 units to 0.019 with 16.*

**What the theorem does not say.**

- **How wide.** The required $H$ can be enormous. For a general Lipschitz function on $[0,1]^D$, a grid of bumps needs on the order of $\varepsilon^{-D}$ units: exponential in the dimension, the same curse as in [[Non-parametric Models]].
- **How to find the weights.** It is a pure existence statement. It does not say that gradient descent will find a good $g$, nor how much data is needed for $g$ to generalize. Many model families (polynomials on compact sets, RBF kernel machines) are universal approximators too; universality alone does not explain why neural networks work well.

### Depth versus width

Depth can be exponentially more efficient than width. Consider the "tent" map on $[0,1]$, $t(x)=2\,\mathrm{ReLU}(x)-4\,\mathrm{ReLU}(x-\tfrac12)$, which equals $2x$ on $[0,\tfrac12]$ and $2-2x$ on $[\tfrac12,1]$. Composing it $k$ times gives a sawtooth with $2^{k}$ linear pieces, using only $2$ ReLUs per layer and $k$ layers. A one-hidden-layer ReLU network with $H$ units is piecewise linear with at most $H+1$ pieces, so it needs at least $2^{k}-1$ units to represent the same function. Telgarsky turned this idea into a theorem: there are functions computed by deep networks with $\Theta(k^3)$ layers of constant width that no network with $O(k)$ layers can approximate unless it has $\Omega(2^k)$ units.[^telgarsky2016] Real data are not sawtooth functions, but the result supports the practical experience that depth buys compositional, reusable features.

## Training neural networks in practice

### Initialization

**Break the symmetry.** If all weights of a layer start equal, every unit in it computes the same function and receives the same gradient, so they stay identical forever and the layer acts like a single unit. Weights are therefore initialized at random (biases can start at zero).

**Keep the signal's scale.** The scale of the random weights matters. Consider $z_j=\sum_{k=1}^{n_{\text{in}}}W_{jk}h_k$ with independent zero-mean weights of variance $\sigma_w^2$, independent of the inputs. Then

$$
\operatorname{Var}(z_j)=n_{\text{in}}\,\sigma_w^2\,\mathbb{E}\big[h_k^2\big].
$$

If $n_{\text{in}}\sigma_w^2$ is below 1, the signal shrinks by a constant factor at every layer and vanishes exponentially with depth; above 1, it explodes. The backward pass, equation (1), multiplies by $\mathbf{W}^\top$ instead and involves $n_{\text{out}}$ instead. Two standard choices:

- **Glorot (Xavier) initialization**[^glorot2010] for tanh-like activations, which are roughly the identity near 0, so $\mathbb{E}[h^2]\approx\operatorname{Var}(z)$ of the previous layer. Preserving the forward variance needs $\sigma_w^2=1/n_{\text{in}}$, the backward one $1/n_{\text{out}}$; the compromise is $\sigma_w^2=2/(n_{\text{in}}+n_{\text{out}})$.
- **He initialization**[^he2015] for ReLU. A ReLU zeroes half of a symmetric input, so $\mathbb{E}[\mathrm{ReLU}(z)^2]=\tfrac12\operatorname{Var}(z)$. To compensate, double the variance: $\sigma_w^2=2/n_{\text{in}}$.

Figure 7 shows what goes wrong otherwise in a 30-layer network.

![[Machine Learning/images/neural-networks-init.svg]]
*Figure 7: a 30-layer MLP of width 256 on Gaussian inputs, with four activation and initialization combinations. Left, the standard deviation of each layer's activations; right, that of the backpropagated error $\delta^{(\ell)}$ for a unit-variance error at the top. With weights that are too small (tanh, $\sigma_w=0.5/\sqrt{n}$) both signals shrink by about a factor of 2 per layer, to about $10^{-9}$. Glorot initialization keeps tanh networks in range. With ReLU, Glorot loses half the variance per layer (activations about $4\times10^{-5}$ at layer 30), while He initialization keeps both signals near 1 throughout.*

### Vanishing and exploding gradients

Unrolling equation (1) over many layers gives

$$
\begin{aligned}
\boldsymbol\delta^{(\ell)}&=\mathbf{D}^{(\ell)}\mathbf{W}^{(\ell+1)\top}\mathbf{D}^{(\ell+1)}\mathbf{W}^{(\ell+2)\top}\\
&\quad\cdots\mathbf{D}^{(L-1)}\mathbf{W}^{(L)\top}\boldsymbol\delta^{(L)},
\end{aligned}
$$

with $\mathbf{D}^{(\ell)}=\operatorname{diag}\big(\phi'(\mathbf{z}^{(\ell)})\big)$. A product of $L-\ell$ matrices tends to grow or shrink geometrically, like a number raised to the power $L-\ell$. When it shrinks, the early layers barely learn (**vanishing gradients**); when it grows, updates blow up (**exploding gradients**). Saturating activations make vanishing likely: each sigmoid layer contributes a factor of at most $1/4$. The same problem, across time steps instead of layers, is what makes plain recurrent networks hard to train.[^bengio1994]

The remedies, most of which appear in this section: initialization that keeps the scale (above); non-saturating activations (ReLU); normalization layers (below); **residual connections** $\mathbf{h}^{(\ell)}=\mathbf{h}^{(\ell-1)}+F(\mathbf{h}^{(\ell-1)})$, whose Jacobian $\mathbf{I}+\partial F/\partial\mathbf{h}$ contains an identity path for gradients;[^he2016] and, for explosions, **gradient clipping**, which rescales the gradient whenever its norm exceeds a threshold.

### Optimizers

The optimizers themselves (mini-batch SGD, momentum, Nesterov, AdaGrad, RMSProp, Adam and AdamW, learning-rate schedules, warm-up and clipping) are covered in [[First-Order Optimization]], with their analysis on quadratics and convergence rates. Study [[First-Order Optimization#Momentum and adaptive methods|momentum and adaptive methods]] and the [[First-Order Optimization#Practical recipe|practical recipe]] first if they are new. Three points are specific to neural networks:

- **Mini-batch SGD is the default.** Batch sizes of 32 to a few thousand are typical, and the gradient noise is cheap regularization that also helps escape saddle points, which are common in non-convex losses.[^bottou2018]
- **Momentum and initialization interact.** Sutskever et al. showed that a good initialization together with momentum makes a large difference when training deep networks.[^sutskever2013]
- **Adam is the usual first choice**, because its per-coordinate step sizes cope with gradients whose scales differ by orders of magnitude across layers.[^kingma2015] Well-tuned SGD with momentum sometimes generalizes slightly better, and AdamW (decoupled weight decay) is the standard variant for transformers.

### Regularization

Neural networks typically have more parameters than training examples and can fit even random labels, so regularization matters.

- **Weight decay.** Add $\frac{\lambda}{2}\lVert\mathbf{W}\rVert^2$ to the loss, exactly [[Linear Models#MAP estimation and regularization|ridge regularization]], i.e. MAP estimation under a Gaussian prior. The SGD step becomes $\mathbf{W}\leftarrow(1-\alpha\lambda)\mathbf{W}-\alpha\mathbf{g}_t$: the weights "decay" by a constant factor each step. With Adam, adding the penalty to the loss is *not* the same as decaying the weights, because the penalty's gradient gets rescaled per coordinate. The decoupled version, AdamW, applies $\mathbf{W}\leftarrow(1-\alpha\lambda)\mathbf{W}$ separately and is usually better.[^loshchilov2019]
- **Early stopping.** Monitor the loss on a validation set and keep the parameters from the epoch where it was lowest. For a quadratic loss, stopping gradient descent early behaves much like an $L_2$ penalty whose strength shrinks with training time.[^goodfellow2016]
- **Dropout.** During training, multiply each hidden unit by an independent mask $m_j\sim\mathrm{Bernoulli}(\rho)$ and rescale, $\tilde h_j=m_jh_j/\rho$, so that $\mathbb{E}[\tilde h_j]=h_j$; at test time use all units unchanged.[^srivastava2014] Here $\rho$ is the *keep* probability, often 0.5 to 0.9. A unit cannot rely on any particular other unit being present, which discourages fragile co-adaptations, and training effectively averages an exponential number of thinned sub-networks that share weights.
- **Data augmentation** (random crops, flips, noise) encodes invariances the model should have; it is often the most effective regularizer for images, and is covered with the applications.

### Normalization layers

**Batch normalization** standardizes each unit's pre-activation over the current mini-batch $\mathcal{B}$ of size $B$, then lets the network undo it with a learned scale $s_j$ and shift $t_j$ (the paper calls them $\gamma$ and $\beta$):[^ioffe2015]

$$
\hat z_{ij}=\frac{z_{ij}-\mu_j}{\sqrt{v_j+\epsilon}},\qquad\text{output}_{ij}=s_j\,\hat z_{ij}+t_j ,
$$

where $\mu_j$ and $v_j$ are the mean and variance of unit $j$ over the batch. At test time, running averages of $\mu_j$ and $v_j$ collected during training replace the batch statistics, so a prediction does not depend on the other examples. Batch norm allows larger learning rates and makes training much less sensitive to initialization. It was introduced to reduce "internal covariate shift", but later work suggests the main benefit is a smoother loss landscape.[^santurkar2018] In [Basic-Image-Classification](https://github.com/parag1604/Basic-Image-Classification) (version 7) adding batch norm is what lets a CIFAR-10 network that would not train start fitting the data.

**Layer normalization** computes the same statistics across the units of a single example instead of across the batch.[^ba2016] It behaves identically at training and test time, works with batch size 1 and with variable-length sequences, and is the standard choice in recurrent networks and transformers.

## Code

The companion repositories build everything above from scratch, one idea per file. A suggested order:

1. [Perceptron](https://github.com/parag1604/Perceptron): the learning rule on toy 2-D data; training to convergence with the bias trick and the $(R/\gamma)^2$ bound printed next to the actual mistake count; the pocket algorithm on XOR and overlapping blobs; and the feature $x_1x_2$ that makes XOR separable.
2. [MLP-Backprop-in-NumPy](https://github.com/parag1604/MLP-Backprop-in-NumPy): a two-layer network that learns XOR with a hand-written backward pass, then gradient checking, sigmoid + binary cross-entropy, and SGD vs momentum vs Adam.
3. [Scalar-Autograd](https://github.com/parag1604/Scalar-Autograd): a micrograd-style reverse-mode autodiff engine on scalars, from recording the computational graph to backpropagation in reverse topological order (with `+=` accumulation) to a small MLP library trained on XOR and Iris.
4. [NN-from-Scratch-to-PyTorch](https://github.com/parag1604/NN-from-Scratch-to-PyTorch): a numpy perceptron and MLP, then the same MLP with PyTorch autograd, then `nn.Linear` layers and `torch.optim`, ending on MNIST.
5. [NN-from-Scratch-to-JAX](https://github.com/parag1604/NN-from-Scratch-to-JAX): the same journey in JAX (`grad`, `jit`, `vmap`), ending with Flax and Optax.
6. [Basic-Image-Classification](https://github.com/parag1604/Basic-Image-Classification): MLPs and CNNs in PyTorch on MNIST, Fashion-MNIST and CIFAR-10, with ablations of the optimizer (SGD vs Adam), batch normalization and dropout.

## Key takeaways

- The perceptron is SGD on logistic regression with hard predictions, or equivalently SGD on the zero-margin hinge loss $\max(0,-y\mathbf{w}^\top\mathbf{x})$. Its update is $\mathbf{w}\leftarrow\mathbf{w}+y_i\mathbf{x}_i$ on mistakes only.
- On separable data it makes at most $(R/\gamma)^2$ mistakes (Block–Novikoff), independent of $N$ and $D$. On non-separable data it never settles; use the pocket or averaged perceptron.
- One hyperplane cannot compute XOR. Kernels fix a non-linear feature map; multilayer perceptrons learn one. Without non-linear activations, depth adds nothing.
- An MLP classifier is logistic or softmax regression on learned features. With matched output and loss, the error at the logits is $\hat{\mathbf{y}}-\mathbf{y}$.
- Backpropagation: $\boldsymbol\delta^{(\ell)}=\phi'(\mathbf{z}^{(\ell)})\odot\mathbf{W}^{(\ell+1)\top}\boldsymbol\delta^{(\ell+1)}$ and $\nabla_{\mathbf{W}^{(\ell)}}=\boldsymbol\delta^{(\ell)}\mathbf{h}^{(\ell-1)\top}$. It is reverse-mode autodiff: all gradients for about the cost of a forward pass. Check hand-written gradients with central differences.
- One hidden layer can approximate any continuous function on a compact set (non-polynomial activation), but possibly with exponentially many units and with no promise that training finds them. Depth can be exponentially more efficient than width.
- To train deep networks: random, variance-preserving initialization (Glorot for tanh, He for ReLU); ReLU-type activations; momentum or Adam with a schedule; weight decay, early stopping and dropout; batch or layer normalization; residual connections against vanishing gradients.

## Where this shows up next

- [[Convolutional Neural Networks]]: networks for images (classification, detection and segmentation), built from the pieces in this lecture plus locality and weight sharing.
- [[Sequence Models]]: networks for text and other sequences: RNNs and LSTMs, attention and transformers, mixture of experts and state space models.
- [[Support Vector Machines]]: the other way out of linear separability, via kernels and the maximum margin; the perceptron's mistake bound is governed by the same margin.
- [[Variational Autoencoders]] and [[Generative Adversarial Networks]] use neural networks as encoders, decoders, generators and critics, all trained by backpropagation.
- [[Energy-Based Models]]: Hopfield networks and Boltzmann machines, the other early family of neural networks, and the link between modern Hopfield networks and attention.
- [[Policy Gradients]] and [[Actor-Critic Algorithms]] train neural-network policies and value functions by backpropagating through the log-probabilities of actions; [[Linear Function Approximation]] is the linear special case.

<!-- References: footnote definitions below render as the numbered "References" list at the bottom of the page -->

[^rosenblatt1958]: Rosenblatt, F. (1958). *The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain*. Psychological Review, 65(6), 386–408. [doi:10.1037/h0042519](https://doi.org/10.1037/h0042519)
[^block1962]: Block, H. D. (1962). *The Perceptron: A Model for Brain Functioning. I*. Reviews of Modern Physics, 34(1), 123–135. [doi:10.1103/RevModPhys.34.123](https://doi.org/10.1103/RevModPhys.34.123)
[^novikoff1962]: Novikoff, A. B. J. (1962). *On Convergence Proofs on Perceptrons*. Proceedings of the Symposium on the Mathematical Theory of Automata, 12, 615–622. Polytechnic Institute of Brooklyn. [pdf](https://cs.uwaterloo.ca/~y328yu/classics/novikoff.pdf)
[^gallant1990]: Gallant, S. I. (1990). *Perceptron-Based Learning Algorithms*. IEEE Transactions on Neural Networks, 1(2), 179–191. [doi:10.1109/72.80230](https://doi.org/10.1109/72.80230)
[^freund1999]: Freund, Y., & Schapire, R. E. (1999). *Large Margin Classification Using the Perceptron Algorithm*. Machine Learning, 37, 277–296. [doi:10.1023/A:1007662407062](https://doi.org/10.1023/A:1007662407062)
[^minsky1969]: Minsky, M., & Papert, S. A. (1969; reissued 2017). *Perceptrons: An Introduction to Computational Geometry*. MIT Press. [doi:10.7551/mitpress/11301.001.0001](https://doi.org/10.7551/mitpress/11301.001.0001)
[^goodfellow2016]: Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*, Chapters 6–8 (XOR example in §6.1, early stopping as $L_2$ in §7.8). MIT Press. [book website](https://www.deeplearningbook.org/)
[^rumelhart1986]: Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). *Learning Representations by Back-Propagating Errors*. Nature, 323, 533–536. [doi:10.1038/323533a0](https://doi.org/10.1038/323533a0)
[^baydin2018]: Baydin, A. G., Pearlmutter, B. A., Radul, A. A., & Siskind, J. M. (2018). *Automatic Differentiation in Machine Learning: a Survey*. Journal of Machine Learning Research, 18(153), 1–43. [jmlr.org](https://jmlr.org/papers/v18/17-468.html)
[^dauphin2014]: Dauphin, Y. N., Pascanu, R., Gulcehre, C., Cho, K., Ganguli, S., & Bengio, Y. (2014). *Identifying and Attacking the Saddle Point Problem in High-Dimensional Non-Convex Optimization*. NeurIPS. [arXiv:1406.2572](https://arxiv.org/abs/1406.2572)
[^cybenko1989]: Cybenko, G. (1989). *Approximation by Superpositions of a Sigmoidal Function*. Mathematics of Control, Signals, and Systems, 2, 303–314. [doi:10.1007/BF02551274](https://doi.org/10.1007/BF02551274)
[^hornik1991]: Hornik, K. (1991). *Approximation Capabilities of Multilayer Feedforward Networks*. Neural Networks, 4(2), 251–257. [doi:10.1016/0893-6080(91)90009-T](https://doi.org/10.1016/0893-6080(91)90009-T)
[^leshno1993]: Leshno, M., Lin, V. Ya., Pinkus, A., & Schocken, S. (1993). *Multilayer Feedforward Networks with a Nonpolynomial Activation Function Can Approximate Any Function*. Neural Networks, 6(6), 861–867. [doi:10.1016/S0893-6080(05)80131-5](https://doi.org/10.1016/S0893-6080(05)80131-5)
[^telgarsky2016]: Telgarsky, M. (2016). *Benefits of Depth in Neural Networks*. COLT. [arXiv:1602.04485](https://arxiv.org/abs/1602.04485)
[^glorot2010]: Glorot, X., & Bengio, Y. (2010). *Understanding the Difficulty of Training Deep Feedforward Neural Networks*. AISTATS. [proceedings.mlr.press](https://proceedings.mlr.press/v9/glorot10a.html)
[^he2015]: He, K., Zhang, X., Ren, S., & Sun, J. (2015). *Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification*. ICCV. [arXiv:1502.01852](https://arxiv.org/abs/1502.01852)
[^bengio1994]: Bengio, Y., Simard, P., & Frasconi, P. (1994). *Learning Long-Term Dependencies with Gradient Descent is Difficult*. IEEE Transactions on Neural Networks, 5(2), 157–166. [doi:10.1109/72.279181](https://doi.org/10.1109/72.279181)
[^he2016]: He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition*. CVPR. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
[^bottou2018]: Bottou, L., Curtis, F. E., & Nocedal, J. (2018). *Optimization Methods for Large-Scale Machine Learning*. SIAM Review, 60(2), 223–311. [arXiv:1606.04838](https://arxiv.org/abs/1606.04838)
[^sutskever2013]: Sutskever, I., Martens, J., Dahl, G., & Hinton, G. (2013). *On the Importance of Initialization and Momentum in Deep Learning*. ICML. [proceedings.mlr.press](https://proceedings.mlr.press/v28/sutskever13.html)
[^kingma2015]: Kingma, D. P., & Ba, J. (2015). *Adam: A Method for Stochastic Optimization*. ICLR. [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)
[^loshchilov2019]: Loshchilov, I., & Hutter, F. (2019). *Decoupled Weight Decay Regularization*. ICLR. [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)
[^srivastava2014]: Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*. Journal of Machine Learning Research, 15(56), 1929–1958. [jmlr.org](https://jmlr.org/papers/v15/srivastava14a.html)
[^ioffe2015]: Ioffe, S., & Szegedy, C. (2015). *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift*. ICML. [arXiv:1502.03167](https://arxiv.org/abs/1502.03167)
[^santurkar2018]: Santurkar, S., Tsipras, D., Ilyas, A., & Madry, A. (2018). *How Does Batch Normalization Help Optimization?* NeurIPS. [arXiv:1805.11604](https://arxiv.org/abs/1805.11604)
[^ba2016]: Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). *Layer Normalization*. [arXiv:1607.06450](https://arxiv.org/abs/1607.06450)
[^prince2023]: Prince, S. J. D. (2023). *Understanding Deep Learning*. MIT Press. [book website](https://udlbook.github.io/udlbook/)
[^bishop2024]: Bishop, C. M., & Bishop, H. (2024). *Deep Learning: Foundations and Concepts*. Springer. [book website](https://www.bishopbook.com/)
