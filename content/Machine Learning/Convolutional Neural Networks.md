---
title: Convolutional Neural Networks
tags:
  - neural-networks
  - deep-learning
  - cnn
  - computer-vision
date: 2026-11-12
draft: false
prev: "[[Neural Networks]]"
next: "[[Sequence Models]]"
---

> **Prerequisites:** [[Neural Networks]] (MLPs, backpropagation, initialization, optimizers, dropout, batch norm) and [[Linear Models]] (softmax regression and the cross-entropy loss).
>
> **TL;DR:**
> - An MLP can in principle represent anything, but on images it needs far too many parameters and far too much data. A **convolution** builds two assumptions about images into the network: **locality** and **translation equivariance**. It is a sparse, weight-shared linear map.
> - Stacking convolutions, non-linearities and pooling grows the **receptive field** from edges to whole objects; **residual connections** keep very deep stacks trainable.
> - The same backbone, with a different head and loss, **classifies** images (cross-entropy), **localizes** and **detects** objects (box regression, IoU, NMS, mAP), and **segments** them (per-pixel prediction with U-Net, BCE + Dice).
>
> **Code:** [Basic-Image-Classification](https://github.com/parag1604/Basic-Image-Classification): MLP to CNN on MNIST, Fashion-MNIST and CIFAR-10, one idea per version; [Detection-and-Segmentation](https://github.com/parag1604/Detection-and-Segmentation): classification, head localization and U-Net segmentation on the Oxford-IIIT Pet dataset.

> [!note] Local notation
> Images are tensors of shape $C\times H\times W$ (channels, height, width), and $k$ is a kernel size; $K$ is still the number of classes.

## Motivation

The [[Neural Networks#Universal approximation|universal approximation theorem]] says that a wide enough MLP can approximate any continuous function. So why not feed raw pixels into an MLP and be done?

Take a modest $224\times224$ colour image. It has $3\cdot224\cdot224=150{,}528$ inputs. A single fully connected layer that maps it to a hidden layer of the same spatial size with 64 channels has about $150{,}528\times3{,}211{,}264\approx4.8\times10^{11}$ weights. That is hopeless to store, and hopeless to learn from any realistic dataset. And it ignores everything we know about images:

- **Locality.** A pixel is strongly related to its neighbours and only weakly to pixels far away. Edges, corners and textures are local patterns.
- **Translation.** A cat in the top-left corner is still a cat in the bottom-right. A detector for "cat ear" that is useful in one place is useful everywhere.
- **Hierarchy.** Edges compose into parts, parts into objects.

The theme of this lecture and the next is that **an architecture is a set of assumptions about the data**, called its **inductive bias**. A good bias removes parameters the data cannot pin down and makes learning feasible. A wrong or too strong bias limits what the model can learn. Convolutional networks carry a strong spatial bias: locality, translation and hierarchy are built in. [[Sequence Models]] then covers recurrent networks (a strong sequential bias) and transformers (almost no bias, so they need a lot of data or careful training).

## Convolutional neural networks

### Convolution as a sparse, shared linear map

A convolutional layer slides a small **kernel** (filter) over the input. At each position it takes the dot product between the kernel and the window of the input underneath it (Figure 1a). For an input with $C_{\text{in}}$ channels, a layer with $C_{\text{out}}$ kernels of size $k\times k$, stride $s$ and bias $b_o$ computes

$$
z_{o,i,j}=b_o+\sum_{c=1}^{C_{\text{in}}}\sum_{u=0}^{k-1}\sum_{v=0}^{k-1}w_{o,c,u,v}\,x_{c,\,si+u,\,sj+v}.
\tag{1}
$$

Strictly speaking this is a **cross-correlation**; a mathematical convolution would flip the kernel. Since the kernel is learned, the flip makes no difference, and every deep-learning library computes (1) under the name "convolution".

Equation (1) is still a **linear** map from $\mathbf{x}$ to $\mathbf{z}$, so it could be written as a (huge) matrix. That matrix is special:

- **Sparse:** each output depends only on a $k\times k$ window (locality).
- **Weight-shared:** the same $k^2C_{\text{in}}$ numbers are reused at every position (translation).

A $3\times3$ convolution from 3 to 64 channels has $64\cdot(3\cdot9+1)=1{,}792$ parameters, against the $4.8\times10^{11}$ of the dense layer above. It cannot represent most linear maps, but the ones it can represent are the ones that make sense for images.

![[Machine Learning/images/applications-of-neural-networks-convolution.svg]]
*Figure 1: (a) a $3\times3$ kernel slides over a $6\times6$ input; each output is the dot product of the kernel with one window (here the highlighted window gives $-2$). (b) Stacking stride-1, 3-wide layers grows the receptive field: a unit in layer 3 sees 7 input pixels.*

**Padding and stride.** Zero-padding the input by $P$ pixels on each side controls the output size, and the stride $s$ is the step between windows. Along each spatial dimension,

$$
H_{\text{out}}=\left\lfloor\frac{H+2P-k}{s}\right\rfloor+1.
\tag{2}
$$

With $k=3$, $P=1$, $s=1$ the size is preserved ("same" padding); with $s=2$ it is halved.[^dumoulin2016]

**Channels.** Each of the $C_{\text{out}}$ kernels produces one **feature map**. The output is again a $C\times H\times W$ tensor, so layers stack. A $1\times1$ convolution mixes channels at each pixel without looking at neighbours: it is a per-pixel linear layer.

> [!example]- Code: a convolutional layer in NumPy (checked against PyTorch)
> ```python
> import numpy as np
>
> def conv2d(x, w, b, stride=1, pad=0):
>     """x: (C_in, H, W), w: (C_out, C_in, k, k), b: (C_out,)"""
>     x = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
>     C_out, C_in, k, _ = w.shape
>     H_out = (x.shape[1] - k) // stride + 1      # eq. (2), padding already added
>     W_out = (x.shape[2] - k) // stride + 1
>     y = np.empty((C_out, H_out, W_out))
>     for i in range(H_out):
>         for j in range(W_out):
>             patch = x[:, i*stride:i*stride+k, j*stride:j*stride+k]
>             y[:, i, j] = np.tensordot(w, patch, axes=3) + b   # eq. (1)
>     return y
> ```
> This matches `torch.nn.functional.conv2d` to machine precision for several shapes, strides and paddings.

### What a filter detects

A kernel responds most strongly to windows that look like itself (a dot product is large when the two vectors align). The Sobel kernel

$$
\mathbf{w}=\begin{bmatrix}1&0&-1\\2&0&-2\\1&0&-1\end{bmatrix}
$$

computes a smoothed horizontal derivative, so it lights up on **vertical edges**, positive on dark-to-bright transitions in one direction and negative in the other (Figure 2b). Before deep learning, such filters were designed by hand. A CNN **learns** its filters, and the first layer of a trained network typically ends up with edge and colour-blob detectors much like these.[^krizhevsky2012]

![[Machine Learning/images/applications-of-neural-networks-edge-filter.svg]]
*Figure 2: (a) a synthetic image. (b) Its response to the vertical-edge Sobel kernel: orange and purple mark the two directions of a vertical edge, and horizontal edges give no response. (c) Shifting the input right by 6 pixels shifts the response map by exactly 6 pixels: convolution is translation-equivariant.*

### Equivariance, invariance and pooling

Let $T_\delta$ shift an image by $\delta$ pixels. A stride-1 convolution satisfies

$$
\operatorname{conv}(T_\delta\,\mathbf{x})=T_\delta\operatorname{conv}(\mathbf{x}),
$$

that is, it is **equivariant**: move the input and the feature map moves with it (Figure 2c). Pointwise non-linearities such as ReLU keep this property, so a whole stack of stride-1 convolutions is equivariant.

For classification we want **invariance** instead: the label should not change when the cat moves. Two ingredients provide it.

- **Pooling.** A $2\times2$ max-pool keeps the largest value in each $2\times2$ block and halves the resolution. A small shift often leaves the maximum unchanged, so pooling gives *local*, approximate invariance.
- **Global pooling.** Averaging each feature map over all positions at the end of the network removes "where" entirely and keeps "what".

Strided convolutions and pooling sub-sample the feature map, so the exact equivariance above only holds for shifts that are multiples of the total stride. Real CNNs are approximately, not exactly, shift-invariant.

### Receptive fields and depth

The **receptive field** of a unit is the set of input pixels it depends on. Stacking $L$ stride-1 layers with $k\times k$ kernels gives a receptive field of

$$
r_L=1+L\,(k-1)
$$

pixels per side (Figure 1b), and every pooling or stride-2 layer doubles the growth rate of the layers after it. Deep layers therefore see large regions and can respond to whole objects, while early layers see edges.

This also explains the design of VGG:[^simonyan2014] two stacked $3\times3$ layers have the same $5\times5$ receptive field as one $5\times5$ layer but use $2\cdot9C^2=18C^2$ weights instead of $25C^2$ (for $C$ channels in and out), and they add an extra non-linearity. Three $3\times3$ layers replace a $7\times7$ one with $27C^2$ instead of $49C^2$ weights.

> [!proof]- Backpropagation through a convolution is a convolution with the flipped kernel
> Take one channel, stride 1, no padding: $z_{i,j}=\sum_{u,v}w_{u,v}\,x_{i+u,\,j+v}$. Let $g_{i,j}=\partial\mathcal{L}/\partial z_{i,j}$ be the gradient arriving from above.
>
> **Gradient with respect to the weights.** Each $w_{u,v}$ is used at every output position, so the chain rule sums over all of them:
>
> $$
> \frac{\partial\mathcal{L}}{\partial w_{u,v}}=\sum_{i,j}g_{i,j}\,x_{i+u,\,j+v}.
> $$
>
> This is the cross-correlation of the input with the gradient map. Weight sharing means the gradients of all positions add up.
>
> **Gradient with respect to the input.** Pixel $x_{i',j'}$ influences every output $z_{i,j}$ with $i+u=i'$ and $j+v=j'$, that is, with weight $w_{i'-i,\,j'-j}$:
>
> $$
> \frac{\partial\mathcal{L}}{\partial x_{i',j'}}=\sum_{i,j}g_{i,j}\,w_{i'-i,\,j'-j}.
> $$
>
> The indices of $\mathbf{w}$ run backwards, so this is a cross-correlation of the (zero-padded) gradient map with the kernel **rotated by 180°**, a "full" convolution. With several channels the roles of $C_{\text{in}}$ and $C_{\text{out}}$ swap. This backward map is exactly the **transposed convolution** we will use for upsampling in segmentation. We checked both identities numerically against PyTorch's autograd and `conv_transpose2d`.

### Classic architectures

The history of CNNs for image classification is a story of going deeper while keeping the network trainable.

- **LeNet-5 (1998)** read handwritten digits with two convolution and pooling stages followed by fully connected layers.[^lecun1998] All the ingredients were there.
- **AlexNet (2012)** scaled the same idea to ImageNet (1.2 million images, 1000 classes) with ReLUs, dropout, data augmentation and GPUs, and cut the top-5 error of the best competing method from 26.2% to 15.3%.[^krizhevsky2012] This result started the deep-learning era in vision.
- **VGG (2014)** used only $3\times3$ convolutions, stacked 16 to 19 layers deep.[^simonyan2014]
- **ResNet (2015)** made networks with 100+ layers trainable with **residual connections**.[^he2016]

**Why residual connections help.** He et al. observed that simply stacking more layers made *training* error worse (a 56-layer plain network fit CIFAR-10 worse than a 20-layer one), so the problem was optimization, not overfitting. A residual block computes

$$
\mathbf{h}_{\ell+1}=\mathbf{h}_\ell+F_\ell(\mathbf{h}_\ell),
$$

where $F_\ell$ is a small stack of convolutions. If the best thing a block can do is nothing, it only has to learn $F_\ell\approx0$, which is easy, instead of learning an identity map through several non-linear layers. Unrolling the recursion from layer $\ell$ to the last layer $L$ gives

$$
\mathbf{h}_L=\mathbf{h}_\ell+\sum_{m=\ell}^{L-1}F_m(\mathbf{h}_m),
$$

and therefore

$$
\begin{aligned}
\frac{\partial\mathcal{L}}{\partial\mathbf{h}_\ell}
&=\frac{\partial\mathcal{L}}{\partial\mathbf{h}_L}\,\frac{\partial\mathbf{h}_L}{\partial\mathbf{h}_\ell}\\
&=\frac{\partial\mathcal{L}}{\partial\mathbf{h}_L}\left(\mathbf{I}+\frac{\partial}{\partial\mathbf{h}_\ell}\sum_{m=\ell}^{L-1}F_m(\mathbf{h}_m)\right).
\end{aligned}
$$

The identity term carries the gradient from the loss straight to every layer, however deep, without passing through a product of many Jacobians that could shrink it.[^he2016identity] We will meet the same "residual stream" in every [[Sequence Models#The transformer block|transformer block]].

> [!example]- Code: a residual block (PyTorch)
> ```python
> import torch
> import torch.nn as nn
>
> class ResidualBlock(nn.Module):
>     """h_{l+1} = ReLU(h_l + F(h_l)), with F = conv-BN-ReLU-conv-BN."""
>     def __init__(self, C):
>         super().__init__()
>         self.F = nn.Sequential(
>             nn.Conv2d(C, C, 3, padding=1, bias=False), nn.BatchNorm2d(C), nn.ReLU(),
>             nn.Conv2d(C, C, 3, padding=1, bias=False), nn.BatchNorm2d(C))
>         nn.init.zeros_(self.F[-1].weight)        # start with F = 0: the block is the identity
>
>     def forward(self, h):
>         return torch.relu(h + self.F(h))          # the skip connection is just "h +"
> ```
> When a block changes the number of channels or the resolution, the shortcut becomes a $1\times1$ convolution with the same stride so the two terms can be added. Zero-initializing the last batch-norm scale, as here, is a common trick: a freshly initialized deep ResNet then starts out as a shallow one.

### Image classification in practice

The classifier itself is the familiar one from [[Linear Models]]: the last layer produces $K$ logits, a softmax turns them into class probabilities, and training minimizes the cross-entropy. A typical network is a stack of `conv → batch norm → ReLU` blocks with occasional pooling, followed by global average pooling and a linear layer. **Accuracy** counts how often the top class is correct. With many similar classes, **top-$n$ accuracy** (is the true class among the $n$ highest-scoring ones?) is also reported; ImageNet results usually quote top-1 and top-5.

Parag's [Basic-Image-Classification](https://github.com/parag1604/Basic-Image-Classification) repository walks through this one idea at a time. Selected rows of its results table:

| Version | Dataset | Model | Params | Train acc | Test acc |
|:-:|---|---|:-:|:-:|:-:|
| 2 | Fashion-MNIST | MLP | 80K | 0.87 | 0.85 |
| 3 | Fashion-MNIST | small CNN | 9K | 0.88 | 0.87 |
| 6 | CIFAR-10 | CNN | 33K | 0.75 | 0.66 |
| 7 | CIFAR-10 | + batch norm | 33K | 0.87 | 0.66 |
| 8 | CIFAR-10 | bigger CNN | 0.6M | 0.99 | 0.76 |
| 9 | CIFAR-10 | + dropout | 0.6M | 0.95 | 0.81 |

The CNN beats the MLP with a ninth of the parameters (versions 2 and 3): that is the inductive bias paying off. Batch norm fixes the optimization of the CIFAR-10 network (training accuracy from 0.75 to 0.87), the bigger network overfits, and dropout trades training accuracy for test accuracy.

Two more tools matter in almost every application.

- **Data augmentation.** Random crops, flips, colour jitter and similar label-preserving transformations enlarge the training set and teach invariances the architecture does not have built in.
- **Transfer learning.** Features learned on a large dataset such as ImageNet are useful for other images: early layers are generic (edges, textures), later ones more task-specific.[^yosinski2014] **Fine-tuning** replaces the last layer of a pretrained network with a new head for the new task and trains on the new data, often with a small learning rate. Step 1 of [Detection-and-Segmentation](https://github.com/parag1604/Detection-and-Segmentation) does exactly this: an ImageNet-pretrained ResNet-18 with its last layer replaced by a single "dog" logit, fine-tuned with binary cross-entropy for 500 iterations to tell cats from dogs on the Oxford-IIIT Pet dataset.[^parkhi2012]

## Object localization and detection

Classification answers "what". The next question is "where".

### Single-object localization

If every image contains exactly one object of interest, localization is just classification plus **regression**: the network outputs four numbers describing a box. A common parameterization is the coordinates of the top-left and bottom-right corners as fractions of the image size, so a sigmoid output layer keeps them in $[0,1]$.

This is step 2 of [Detection-and-Segmentation](https://github.com/parag1604/Detection-and-Segmentation). The network predicts the box around the **pet's head**. It starts from the step-1 cat/dog classifier (transfer again: features that recognize pets already respond to their heads), only the last layer is replaced by a 4-output sigmoid layer, and training minimizes the mean squared error between predicted and true corners.

**Box losses.** Squared error is the simplest choice. The $L_1$ loss is less sensitive to badly wrong boxes, and the **smooth $L_1$** (Huber) loss used by the R-CNN family combines the two: quadratic for small errors $e$ and linear for large ones,

$$
\rho_\beta(e)=\begin{cases}\dfrac{e^2}{2\beta}&\text{if }|e|<\beta,\\[1ex]|e|-\dfrac{\beta}{2}&\text{otherwise.}\end{cases}
$$

### Intersection over union

All of these losses treat the four coordinates separately, but what we care about is how well the boxes **overlap**. The standard measure is the **intersection over union** (Figure 3a):

$$
\operatorname{IoU}(A,B)=\frac{|A\cap B|}{|A\cup B|}=\frac{|A\cap B|}{|A|+|B|-|A\cap B|}\in[0,1].
$$

It is 1 for a perfect match and 0 for disjoint boxes, and it does not depend on the scale of the image. Parag's repository reports the IoU of predicted and true head boxes as its test metric.

One can also train on $1-\operatorname{IoU}$ directly, but its gradient is zero whenever the boxes do not overlap. The **generalized IoU** fixes this by subtracting the fraction of the smallest enclosing box $E$ that neither box covers, $\operatorname{GIoU}=\operatorname{IoU}-|E\setminus(A\cup B)|/|E|$, which keeps pulling disjoint boxes together.[^rezatofighi2019]

![[Machine Learning/images/applications-of-neural-networks-iou-nms.svg]]
*Figure 3: (a) IoU is the area of the intersection (shaded) divided by the area of the union; these two boxes overlap with IoU 0.20. (b) Non-maximum suppression: among overlapping candidate boxes with confidence scores, the best box is kept and every box overlapping it with IoU > 0.5 is removed (dashed), then the process repeats.*

### Many objects: detection

A real image contains an unknown number of objects of several classes, so a fixed number of outputs no longer works. Detectors instead score a large set of **candidate boxes** and keep the good ones. Two families dominate.

- **Two-stage detectors (the R-CNN family).** R-CNN first proposed about 2000 candidate regions with a classical algorithm, then classified each crop with a CNN and refined its box.[^girshick2014] Faster R-CNN made the proposals themselves learned: a small **region proposal network** slides over the shared feature map and, at each position, scores a fixed set of reference boxes called **anchors** (several scales and aspect ratios).[^ren2015] Each anchor, with centre $(a_x,a_y)$, width $a_w$ and height $a_h$, predicts offsets to a box $(q_x,q_y,q_w,q_h)$ rather than its absolute coordinates, for example $t_x=(q_x-a_x)/a_w$ and $t_w=\log(q_w/a_w)$, which keeps the regression targets small and scale-free.
- **One-stage detectors.** YOLO divides the image into a grid and lets each cell directly predict a few boxes with confidences and class probabilities, in a single forward pass.[^redmon2016] SSD does the same with default boxes at several feature-map resolutions, so that small objects are found on fine maps and large objects on coarse ones.[^liu2016] One-stage detectors are faster; two-stage ones were historically more accurate.

> [!example]- Code: anchor box offsets and the smooth $L_1$ box loss (PyTorch)
> ```python
> import torch
>
> def encode(box, anchor):
>     """Boxes as (x_center, y_center, w, h), shape (N, 4) -> targets (t_x, t_y, t_w, t_h)."""
>     (x, y, w, h), (xa, ya, wa, ha) = box.unbind(-1), anchor.unbind(-1)
>     return torch.stack([(x - xa) / wa, (y - ya) / ha, torch.log(w / wa), torch.log(h / ha)], -1)
>
> def decode(t, anchor):
>     (tx, ty, tw, th), (xa, ya, wa, ha) = t.unbind(-1), anchor.unbind(-1)
>     return torch.stack([xa + tx * wa, ya + ty * ha, wa * torch.exp(tw), ha * torch.exp(th)], -1)
>
> def smooth_l1(z, beta=1.0):
>     """Quadratic for |z| < beta, linear beyond: the Huber-type box loss."""
>     a = z.abs()
>     return torch.where(a < beta, 0.5 * a ** 2 / beta, a - 0.5 * beta).sum(-1).mean()
>
> # assumes: pred_t (N, 4) from the network, gt_box and anchor (N, 4)
> # loss = smooth_l1(pred_t - encode(gt_box, anchor))
> ```
> The network regresses the offsets $t$, not the box itself; `decode` turns its outputs back into boxes before non-maximum suppression.

**Non-maximum suppression.** Neighbouring anchors or cells tend to fire on the same object, producing a cluster of overlapping boxes. Greedy **NMS** cleans this up (Figure 3b): sort the boxes by score, keep the best one, discard all remaining boxes whose IoU with it exceeds a threshold (typically 0.5), and repeat with what is left. NMS runs separately for each class.

> [!example]- Code: IoU and greedy NMS (checked against torchvision)
> ```python
> import numpy as np
>
> def iou(a, B):
>     """a: (4,) box (x1, y1, x2, y2); B: (M, 4). Returns (M,) IoUs."""
>     ix1 = np.maximum(a[0], B[:, 0]); iy1 = np.maximum(a[1], B[:, 1])
>     ix2 = np.minimum(a[2], B[:, 2]); iy2 = np.minimum(a[3], B[:, 3])
>     inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
>     area = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
>     return inter / (area(a) + area(B) - inter)
>
> def nms(boxes, scores, thresh=0.5):
>     order = np.argsort(-scores)          # best first
>     keep = []
>     while order.size > 0:
>         i = order[0]
>         keep.append(i)
>         rest = order[1:]
>         order = rest[iou(boxes[i], boxes[rest]) <= thresh]
>     return keep
> ```
> On 50 random sets of 40 boxes, `iou` matches `torchvision.ops.box_iou` and `nms` returns exactly the indices of `torchvision.ops.nms`.

**Evaluation: mean average precision.** A predicted box counts as a true positive if its IoU with an unmatched ground-truth box of the same class is at least a threshold (0.5 in PASCAL VOC).[^everingham2010] Sweeping the confidence threshold traces a precision–recall curve for each class. The **average precision** (AP) is the area under it, and **mAP** is the mean over classes. COCO additionally averages over IoU thresholds from 0.5 to 0.95, which rewards precise boxes.[^lin2014]

## Segmentation

Segmentation asks for a label for **every pixel**.

- **Semantic segmentation** assigns each pixel a class ("pet", "background") without distinguishing individual objects.
- **Instance segmentation** also separates objects of the same class ("this dog", "that dog"). Mask R-CNN does this by adding a small mask-prediction branch to Faster R-CNN, one mask per detected box.[^he2017mask]

We focus on semantic segmentation, the task of step 3 in [Detection-and-Segmentation](https://github.com/parag1604/Detection-and-Segmentation): which pixels belong to the pet?

### Fully convolutional networks and upsampling

A classification CNN shrinks the image to a small feature map and then throws away the spatial layout. For a per-pixel output we keep the layout. **Fully convolutional networks** replace the final dense layers by $1\times1$ convolutions, so the network outputs a coarse map of class scores, and then **upsample** it back to the input resolution.[^long2015] Because every layer is a convolution, such a network accepts images of any size.

Upsampling can be fixed (bilinear interpolation) or learned. The learned version is the **transposed convolution**: each input pixel scatters a copy of the kernel, scaled by its value, into the output, with stride $s$ between copies. It is the backward map of a convolution (see the aside above), and its output size inverts (2):

$$
H_{\text{out}}=(H-1)\,s-2P+k.
$$

A $2\times2$ kernel with stride 2 and no padding exactly doubles the resolution. When the kernel size is not a multiple of the stride the copies overlap unevenly, which can produce **checkerboard artifacts**; resize-then-convolve avoids them.[^odena2016]

> [!example]- Code: a transposed convolution as "scatter a scaled kernel" (NumPy, checked against PyTorch)
> ```python
> import numpy as np
>
> def conv_transpose2d(x, w, s=2):
>     """x: (H, W) input, w: (k, k) kernel, stride s, no padding (one channel)."""
>     H, W = x.shape; k = w.shape[0]
>     y = np.zeros(((H - 1) * s + k, (W - 1) * s + k))   # H_out = (H - 1) s + k
>     for i in range(H):
>         for j in range(W):
>             y[i*s:i*s+k, j*s:j*s+k] += x[i, j] * w      # each input pixel stamps the kernel
>     return y
> ```
> A convolution *gathers* a window into one output; its transpose *scatters* one input into a window. Where stamps overlap (here when $k$ is not a multiple of $s$) they add up, which is the source of checkerboard artifacts.

### U-Net

Coarse feature maps know *what* is in the image but have lost precise *where*. **U-Net** recovers the detail with **skip connections**:[^ronneberger2015] an encoder halves the resolution step by step, a decoder doubles it back, and at each resolution the decoder concatenates the encoder's feature map of the same size before convolving (Figure 4). The decoder thus combines high-level context from below with fine spatial detail from across.

![[Machine Learning/images/applications-of-neural-networks-unet.svg]]
*Figure 4: the U-Net of Parag's segmentation code. Each box is a feature map (channels, spatial size). The encoder (left) halves the resolution five times, from $224\times224$ to $7\times7$; the decoder (right) doubles it back with transposed convolutions, concatenating the encoder map of the same size (dashed skip connections); a final $1\times1$ convolution outputs one logit per pixel.*

Parag's U-Net is trained from scratch. It uses padded convolutions, so every level keeps its size exactly; the original U-Net used unpadded convolutions and cropped the skip features to match.

> [!example]- Code: one U-Net decoder step (PyTorch, after Parag's `decoder_block`)
> ```python
> import torch
> import torch.nn as nn
>
> class DecoderBlock(nn.Module):
>     def __init__(self, c_in, c_out):
>         super().__init__()
>         self.up = nn.ConvTranspose2d(c_in, c_out, 2, stride=2)    # doubles H and W
>         self.conv = nn.Sequential(
>             nn.BatchNorm2d(2 * c_out), nn.ReLU(),
>             nn.Conv2d(2 * c_out, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(),
>             nn.Conv2d(c_out, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU())
>
>     def forward(self, h, skip):            # h: coarse "what"; skip: encoder map, same size as up(h)
>         return self.conv(torch.cat([skip, self.up(h)], dim=1))   # concatenate channels
> ```
> The skip connection is the `torch.cat`: the decoder sees the upsampled context and the encoder's fine detail side by side.

### Losses: per-pixel cross-entropy and Dice

The natural loss treats each pixel as a separate binary classification. With predicted foreground probability $p_i$ and label $y_i\in\{0,1\}$ for pixel $i$ of $M$ pixels,

$$
\mathcal{L}_{\text{BCE}}=-\frac{1}{M}\sum_{i=1}^{M}\Big[y_i\log p_i+(1-y_i)\log(1-p_i)\Big].
$$

Its weakness is **class imbalance**. If the object covers 5% of the image, predicting "background" everywhere already gives a small loss, and most of the gradient comes from easy background pixels.

The **Dice coefficient** measures the overlap of the predicted mask $A$ and the true mask $B$ directly:

$$
\operatorname{Dice}(A,B)=\frac{2|A\cap B|}{|A|+|B|}.
$$

It is closely related to IoU, $\operatorname{Dice}=2\operatorname{IoU}/(1+\operatorname{IoU})$, and it depends only on the object, not on how much background there is.

**Why a "soft" Dice is needed.** The metric uses hard masks, $\hat y_i=\mathbb{1}[p_i>0.5]$. The threshold is a step function, whose derivative is zero everywhere except at the jump. So the hard Dice is piecewise constant in the network's parameters, and gradient descent gets no signal from it. The fix is to plug in the probabilities themselves. The **soft Dice** is

$$
D(\mathbf{p},\mathbf{y})=\frac{2I+\epsilon}{S+\epsilon},\qquad I=\sum_i p_iy_i,\quad S=\sum_i p_i+\sum_i y_i,
$$

where $\epsilon$ (often 1) avoids $0/0$ on empty masks.[^milletari2016] With $\epsilon=0$, its derivative with respect to one pixel's probability is

$$
\frac{\partial D}{\partial p_j}=\frac{2y_jS-2I}{S^2}.
$$

For a foreground pixel ($y_j=1$) this is $2(S-I)/S^2>0$, so the loss $1-D$ pushes $p_j$ up; for a background pixel it is $-2I/S^2<0$, so the loss pushes $p_j$ down. Every pixel gets a useful gradient, and its size is set relative to the object, not the whole image.

Parag's code trains with **BCE + (1 − soft Dice)** and reports the hard Dice (threshold 0.5) as the metric. BCE gives smooth per-pixel gradients early in training; Dice keeps the small foreground from being swamped.

> [!example]- Code: soft Dice loss (checked against autograd)
> ```python
> def soft_dice_loss(p, y, eps=1.0):
>     """p: predicted probabilities, y: {0,1} mask, same shape (H, W)."""
>     inter = (p * y).sum()
>     return 1 - (2 * inter + eps) / (p.sum() + y.sum() + eps)
> ```
> With PyTorch tensors, the autograd gradient of the $\epsilon=0$ version matches the formula above exactly, and thresholding $p$ at 0.5 first detaches it from the graph (no gradient at all).


## Key takeaways

- **Architectures encode assumptions.** Convolutions assume locality and translation equivariance; that bias is why a small CNN beats a much larger MLP on images.
- A **convolution** is a sparse, weight-shared linear map. Its output size is $\lfloor(H+2P-k)/s\rfloor+1$, stacking layers grows the receptive field, and its gradient is again a convolution (with the flipped kernel). Pooling turns equivariance into approximate invariance.
- **Residual connections** give the gradient a direct path through very deep networks. Data augmentation and **transfer learning** matter in almost every application.
- **Localization** adds box regression to a classifier; **detection** scores many candidate boxes and removes duplicates with **NMS**; boxes are judged by **IoU** and detectors by **mAP**.
- **Segmentation** predicts every pixel with an encoder–decoder such as **U-Net**, whose skip connections restore spatial detail. The **soft Dice** loss handles class imbalance and stays differentiable, unlike the thresholded metric.

## Where this shows up next

- [[Sequence Models]], the next lecture: weight sharing **in time** (recurrent networks), attention and transformers, and the [[Sequence Models#The Vision Transformer|Vision Transformer]], which drops convolutions and treats image patches as tokens.
- **Generative models.** The decoders of [[Variational Autoencoders]] and the generators of [[Generative Adversarial Networks]] for images are built from transposed convolutions, and a U-Net is the standard noise- or score-prediction network in [[Denoising Diffusion Models]].
- [[Graph Neural Networks]]: convolution generalized from grids to arbitrary graphs, where neighbours have no fixed positions.
- [[Self-Supervised Representation Learning]]: learning image features without labels, for transfer to tasks like the ones in this lecture.
- **Textbooks.** Prince's *Understanding Deep Learning*,[^prince2023] Goodfellow, Bengio and Courville's *Deep Learning*,[^goodfellow2016] and the interactive *Dive into Deep Learning*[^zhang2023] cover all of this lecture in more depth.

<!-- References: footnote definitions below render as the numbered "References" list at the bottom of the page -->

[^dumoulin2016]: Dumoulin, V., & Visin, F. (2016). *A guide to convolution arithmetic for deep learning*. arXiv. [arXiv:1603.07285](https://arxiv.org/abs/1603.07285)
[^krizhevsky2012]: Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). *ImageNet Classification with Deep Convolutional Neural Networks*. NeurIPS. [proceedings](https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)
[^simonyan2014]: Simonyan, K., & Zisserman, A. (2015). *Very Deep Convolutional Networks for Large-Scale Image Recognition*. ICLR. [arXiv:1409.1556](https://arxiv.org/abs/1409.1556)
[^lecun1998]: LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). *Gradient-Based Learning Applied to Document Recognition*. Proceedings of the IEEE, 86(11), 2278–2324. [doi:10.1109/5.726791](https://doi.org/10.1109/5.726791)
[^he2016]: He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition*. CVPR. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
[^he2016identity]: He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Identity Mappings in Deep Residual Networks*. ECCV. [arXiv:1603.05027](https://arxiv.org/abs/1603.05027)
[^yosinski2014]: Yosinski, J., Clune, J., Bengio, Y., & Lipson, H. (2014). *How transferable are features in deep neural networks?* NeurIPS. [arXiv:1411.1792](https://arxiv.org/abs/1411.1792)
[^parkhi2012]: Parkhi, O. M., Vedaldi, A., Zisserman, A., & Jawahar, C. V. (2012). *Cats and Dogs*. CVPR. [doi:10.1109/CVPR.2012.6248092](https://doi.org/10.1109/CVPR.2012.6248092); dataset: [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/)
[^rezatofighi2019]: Rezatofighi, H., Tsoi, N., Gwak, J., Sadeghian, A., Reid, I., & Savarese, S. (2019). *Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression*. CVPR. [arXiv:1902.09630](https://arxiv.org/abs/1902.09630)
[^girshick2014]: Girshick, R., Donahue, J., Darrell, T., & Malik, J. (2014). *Rich feature hierarchies for accurate object detection and semantic segmentation*. CVPR. [arXiv:1311.2524](https://arxiv.org/abs/1311.2524)
[^ren2015]: Ren, S., He, K., Girshick, R., & Sun, J. (2015). *Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks*. NeurIPS. [arXiv:1506.01497](https://arxiv.org/abs/1506.01497)
[^redmon2016]: Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). *You Only Look Once: Unified, Real-Time Object Detection*. CVPR. [arXiv:1506.02640](https://arxiv.org/abs/1506.02640)
[^liu2016]: Liu, W., Anguelov, D., Erhan, D., Szegedy, C., Reed, S., Fu, C.-Y., & Berg, A. C. (2016). *SSD: Single Shot MultiBox Detector*. ECCV. [arXiv:1512.02325](https://arxiv.org/abs/1512.02325)
[^everingham2010]: Everingham, M., Van Gool, L., Williams, C. K. I., Winn, J., & Zisserman, A. (2010). *The Pascal Visual Object Classes (VOC) Challenge*. International Journal of Computer Vision, 88, 303–338. [doi:10.1007/s11263-009-0275-4](https://doi.org/10.1007/s11263-009-0275-4)
[^lin2014]: Lin, T.-Y., Maire, M., Belongie, S., et al. (2014). *Microsoft COCO: Common Objects in Context*. ECCV. [arXiv:1405.0312](https://arxiv.org/abs/1405.0312)
[^he2017mask]: He, K., Gkioxari, G., Dollár, P., & Girshick, R. (2017). *Mask R-CNN*. ICCV. [arXiv:1703.06870](https://arxiv.org/abs/1703.06870)
[^long2015]: Long, J., Shelhamer, E., & Darrell, T. (2015). *Fully Convolutional Networks for Semantic Segmentation*. CVPR. [arXiv:1411.4038](https://arxiv.org/abs/1411.4038)
[^odena2016]: Odena, A., Dumoulin, V., & Olah, C. (2016). *Deconvolution and Checkerboard Artifacts*. Distill. [doi:10.23915/distill.00003](https://doi.org/10.23915/distill.00003)
[^ronneberger2015]: Ronneberger, O., Fischer, P., & Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation*. MICCAI. [arXiv:1505.04597](https://arxiv.org/abs/1505.04597)
[^milletari2016]: Milletari, F., Navab, N., & Ahmadi, S.-A. (2016). *V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation*. 3DV. [arXiv:1606.04797](https://arxiv.org/abs/1606.04797)
[^prince2023]: Prince, S. J. D. (2023). *Understanding Deep Learning*. MIT Press. [udlbook.github.io](https://udlbook.github.io/udlbook/)
[^goodfellow2016]: Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. [deeplearningbook.org](https://www.deeplearningbook.org/)
[^zhang2023]: Zhang, A., Lipton, Z. C., Li, M., & Smola, A. J. (2023). *Dive into Deep Learning*. Cambridge University Press. [d2l.ai](https://d2l.ai/)
