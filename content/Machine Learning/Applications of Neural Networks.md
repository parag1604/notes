---
title: Applications of Neural Networks
tags:
  - neural-networks
  - deep-learning
  - cnn
  - transformers
  - computer-vision
  - nlp
date: 2026-11-12
draft: false
prev: "[[Neural Networks]]"
next: "[[Non-parametric Models]]"
---

> **Prerequisites:** [[Neural Networks]] (MLPs, backpropagation, initialization, optimizers, dropout, batch norm) and [[Linear Models]] (softmax regression and the cross-entropy loss).
>
> **TL;DR:**
> - An MLP can in principle represent anything, but on images and text it needs far too many parameters and far too much data. Practical architectures **build assumptions about the data into the network**: locality and translation equivariance for images (**convolutions**), order for text (**recurrence**), and content-based retrieval between elements (**attention**).
> - **CNNs** classify images; the same backbone, with a different head and loss, **localizes** objects (box regression, IoU, NMS) and **segments** them (per-pixel prediction with U-Net, BCE + Dice).
> - Text becomes a sequence of **embeddings**; **LSTMs** read it in order, and **transformers** let every token attend to every other one through **scaled dot-product attention**.
> - Encoder–decoder models **caption** images, and the **Vision Transformer** treats an image as a sequence of patches. Weaker built-in assumptions mean more data or stronger regularization.
>
> **Code:** [Basic-Image-Classification](https://github.com/parag1604/Basic-Image-Classification): MLP to CNN on MNIST, Fashion-MNIST and CIFAR-10, one idea per version; [Detection-and-Segmentation](https://github.com/parag1604/Detection-and-Segmentation): classification, head localization and U-Net segmentation on the Oxford-IIIT Pet dataset; [Word-Embeddings-and-Text-Classification](https://github.com/parag1604/Word-Embeddings-and-Text-Classification): fastText vectors, then AG News classification with LSTMs and a transformer encoder; [Image-Captioning](https://github.com/parag1604/Image-Captioning): a ViT encoder with LSTM, attention-LSTM and transformer decoders on Flickr8k; [improved-vision-encoders](https://github.com/paragduttaiisc/improved-vision-encoders): a ViT encoder trained self-supervised (JEPA-style masked latent prediction) on ImageNet, evaluated by linear probes, with switchable token mixers (attention or FFT), FFN activations and mixture-of-experts layers.

> [!note] Local notation
> In this lecture $n$ is the **number of tokens** in a sequence (not the number of samples) and $d$ is the **width** of a representation. Images are tensors of shape $C\times H\times W$ (channels, height, width), and $k$ is a kernel size. Keys are the matrix $\mathbf{K}$; $K$ alone is still the number of classes.

## Motivation

The [[Neural Networks]] lecture ended with the universal approximation theorem: a wide enough MLP can approximate any continuous function. So why not feed raw pixels into an MLP and be done?

Take a modest $224\times224$ colour image. It has $3\cdot224\cdot224=150{,}528$ inputs. A single fully connected layer that maps it to a hidden layer of the same spatial size with 64 channels has about $150{,}528\times3{,}211{,}264\approx4.8\times10^{11}$ weights. That is hopeless to store, and hopeless to learn from any realistic dataset. And it ignores everything we know about images:

- **Locality.** A pixel is strongly related to its neighbours and only weakly to pixels far away. Edges, corners and textures are local patterns.
- **Translation.** A cat in the top-left corner is still a cat in the bottom-right. A detector for "cat ear" that is useful in one place is useful everywhere.
- **Hierarchy.** Edges compose into parts, parts into objects.

Text has its own structure. The meaning of a word depends on its neighbours, sentences have different lengths, and the **order** of words matters ("dog bites man" vs "man bites dog").

The theme of this lecture is that **an architecture is a set of assumptions about the data**, called its **inductive bias**. A good bias removes parameters the data cannot pin down and makes learning feasible. A wrong or too strong bias limits what the model can learn. We will see the whole spectrum: CNNs (strong spatial bias), RNNs (strong sequential bias), and transformers (almost no bias, so they need a lot of data or careful training).

## Convolutional neural networks

### Convolution as a sparse, shared linear map

A convolutional layer slides a small **kernel** (filter) over the input. At each position it takes the dot product between the kernel and the window of the input underneath it (Figure 1a). For an input with $C_{\text{in}}$ channels, a layer with $C_{\text{out}}$ kernels of size $k\times k$, stride $s$ and bias $b_o$ computes

$$
y_{o,i,j}=b_o+\sum_{c=1}^{C_{\text{in}}}\sum_{u=0}^{k-1}\sum_{v=0}^{k-1}w_{o,c,u,v}\,x_{c,\,si+u,\,sj+v}.
\tag{1}
$$

Strictly speaking this is a **cross-correlation**; a mathematical convolution would flip the kernel. Since the kernel is learned, the flip makes no difference, and every deep-learning library computes (1) under the name "convolution".

Equation (1) is still a **linear** map from $\mathbf{x}$ to $\mathbf{y}$, so it could be written as a (huge) matrix. That matrix is special:

- **Sparse:** each output depends only on a $k\times k$ window (locality).
- **Weight-shared:** the same $k^2C_{\text{in}}$ numbers are reused at every position (translation).

A $3\times3$ convolution from 3 to 64 channels has $64\cdot(3\cdot9+1)=1{,}792$ parameters, against the $4.8\times10^{11}$ of the dense layer above. It cannot represent most linear maps, but the ones it can represent are the ones that make sense for images.

![[Machine Learning/images/applications-of-neural-networks-convolution.svg]]
*Figure 1: (a) a $3\times3$ kernel slides over a $6\times6$ input; each output is the dot product of the kernel with one window (here the highlighted window gives $-2$). (b) Stacking stride-1, 3-wide layers grows the receptive field: a unit in layer 3 sees 7 input pixels.*

**Padding and stride.** Zero-padding the input by $p$ pixels on each side controls the output size, and the stride $s$ is the step between windows. Along each spatial dimension,

$$
H_{\text{out}}=\left\lfloor\frac{H+2p-k}{s}\right\rfloor+1.
\tag{2}
$$

With $k=3$, $p=1$, $s=1$ the size is preserved ("same" padding); with $s=2$ it is halved.[^dumoulin2016]

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
> Take one channel, stride 1, no padding: $y_{i,j}=\sum_{u,v}w_{u,v}\,x_{i+u,\,j+v}$. Let $g_{i,j}=\partial\mathcal{L}/\partial y_{i,j}$ be the gradient arriving from above.
>
> **Gradient with respect to the weights.** Each $w_{u,v}$ is used at every output position, so the chain rule sums over all of them:
>
> $$
> \frac{\partial\mathcal{L}}{\partial w_{u,v}}=\sum_{i,j}g_{i,j}\,x_{i+u,\,j+v}.
> $$
>
> This is the cross-correlation of the input with the gradient map. Weight sharing means the gradients of all positions add up.
>
> **Gradient with respect to the input.** Pixel $x_{a,b}$ influences every output $y_{i,j}$ with $i+u=a$ and $j+v=b$, that is, with weight $w_{a-i,\,b-j}$:
>
> $$
> \frac{\partial\mathcal{L}}{\partial x_{a,b}}=\sum_{i,j}g_{i,j}\,w_{a-i,\,b-j}.
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
\mathbf{h}_{l+1}=\mathbf{h}_l+F_l(\mathbf{h}_l),
$$

where $F_l$ is a small stack of convolutions. If the best thing a block can do is nothing, it only has to learn $F_l\approx0$, which is easy, instead of learning an identity map through several non-linear layers. Unrolling the recursion from layer $l$ to the last layer $L$ gives

$$
\mathbf{h}_L=\mathbf{h}_l+\sum_{m=l}^{L-1}F_m(\mathbf{h}_m),
$$

and therefore

$$
\begin{aligned}
\frac{\partial\mathcal{L}}{\partial\mathbf{h}_l}
&=\frac{\partial\mathcal{L}}{\partial\mathbf{h}_L}\,\frac{\partial\mathbf{h}_L}{\partial\mathbf{h}_l}\\
&=\frac{\partial\mathcal{L}}{\partial\mathbf{h}_L}\left(\mathbf{I}+\frac{\partial}{\partial\mathbf{h}_l}\sum_{m=l}^{L-1}F_m(\mathbf{h}_m)\right).
\end{aligned}
$$

The identity term carries the gradient from the loss straight to every layer, however deep, without passing through a product of many Jacobians that could shrink it.[^he2016identity] We will meet the same "residual stream" in every transformer block.

### Image classification in practice

The classifier itself is the familiar one from [[Linear Models]]: the last layer produces $K$ logits, a softmax turns them into class probabilities, and training minimizes the cross-entropy. A typical network is a stack of `conv → batch norm → ReLU` blocks with occasional pooling, followed by global average pooling and a linear layer. **Accuracy** counts how often the top class is correct. With many similar classes, **top-$k$ accuracy** (is the true class among the $k$ highest-scoring ones?) is also reported; ImageNet results usually quote top-1 and top-5.

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

If every image contains exactly one object of interest, localization is just classification plus **regression**: the network outputs four numbers describing a box. A common parameterization is the corners $(x_1,y_1,x_2,y_2)$ as fractions of the image size, so a sigmoid output layer keeps them in $[0,1]$.

This is step 2 of [Detection-and-Segmentation](https://github.com/parag1604/Detection-and-Segmentation). The network predicts the box around the **pet's head**. It starts from the step-1 cat/dog classifier (transfer again: features that recognize pets already respond to their heads), only the last layer is replaced by a 4-output sigmoid layer, and training minimizes the mean squared error between predicted and true corners.

**Box losses.** Squared error is the simplest choice. The $L_1$ loss is less sensitive to badly wrong boxes, and the **smooth $L_1$** (Huber) loss used by the R-CNN family combines the two: quadratic for small errors $z$ and linear for large ones,

$$
\ell_\beta(z)=\begin{cases}\dfrac{z^2}{2\beta}&\text{if }|z|<\beta,\\[1ex]|z|-\dfrac{\beta}{2}&\text{otherwise.}\end{cases}
$$

### Intersection over union

All of these losses treat the four coordinates separately, but what we care about is how well the boxes **overlap**. The standard measure is the **intersection over union** (Figure 3a):

$$
\operatorname{IoU}(A,B)=\frac{|A\cap B|}{|A\cup B|}=\frac{|A\cap B|}{|A|+|B|-|A\cap B|}\in[0,1].
$$

It is 1 for a perfect match and 0 for disjoint boxes, and it does not depend on the scale of the image. Parag's repository reports the IoU of predicted and true head boxes as its test metric.

One can also train on $1-\operatorname{IoU}$ directly, but its gradient is zero whenever the boxes do not overlap. The **generalized IoU** fixes this by subtracting the fraction of the smallest enclosing box $C$ that neither box covers, $\operatorname{GIoU}=\operatorname{IoU}-|C\setminus(A\cup B)|/|C|$, which keeps pulling disjoint boxes together.[^rezatofighi2019]

![[Machine Learning/images/applications-of-neural-networks-iou-nms.svg]]
*Figure 3: (a) IoU is the area of the intersection (shaded) divided by the area of the union; these two boxes overlap with IoU 0.20. (b) Non-maximum suppression: among overlapping candidate boxes with confidence scores, the best box is kept and every box overlapping it with IoU > 0.5 is removed (dashed), then the process repeats.*

### Many objects: detection

A real image contains an unknown number of objects of several classes, so a fixed number of outputs no longer works. Detectors instead score a large set of **candidate boxes** and keep the good ones. Two families dominate.

- **Two-stage detectors (the R-CNN family).** R-CNN first proposed about 2000 candidate regions with a classical algorithm, then classified each crop with a CNN and refined its box.[^girshick2014] Faster R-CNN made the proposals themselves learned: a small **region proposal network** slides over the shared feature map and, at each position, scores a fixed set of reference boxes called **anchors** (several scales and aspect ratios).[^ren2015] Each anchor $(x_a,y_a,w_a,h_a)$ predicts offsets rather than absolute coordinates, for example $t_x=(x-x_a)/w_a$ and $t_w=\log(w/w_a)$, which keeps the regression targets small and scale-free.
- **One-stage detectors.** YOLO divides the image into a grid and lets each cell directly predict a few boxes with confidences and class probabilities, in a single forward pass.[^redmon2016] SSD does the same with default boxes at several feature-map resolutions, so that small objects are found on fine maps and large objects on coarse ones.[^liu2016] One-stage detectors are faster; two-stage ones were historically more accurate.

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
H_{\text{out}}=(H-1)\,s-2p+k.
$$

A $2\times2$ kernel with stride 2 and no padding exactly doubles the resolution. When the kernel size is not a multiple of the stride the copies overlap unevenly, which can produce **checkerboard artifacts**; resize-then-convolve avoids them.[^odena2016]

### U-Net

Coarse feature maps know *what* is in the image but have lost precise *where*. **U-Net** recovers the detail with **skip connections**:[^ronneberger2015] an encoder halves the resolution step by step, a decoder doubles it back, and at each resolution the decoder concatenates the encoder's feature map of the same size before convolving (Figure 4). The decoder thus combines high-level context from below with fine spatial detail from across.

![[Machine Learning/images/applications-of-neural-networks-unet.svg]]
*Figure 4: the U-Net of Parag's segmentation code. Each box is a feature map (channels, spatial size). The encoder (left) halves the resolution five times, from $224\times224$ to $7\times7$; the decoder (right) doubles it back with transposed convolutions, concatenating the encoder map of the same size (dashed skip connections); a final $1\times1$ convolution outputs one logit per pixel.*

Parag's U-Net is trained from scratch. It uses padded convolutions, so every level keeps its size exactly; the original U-Net used unpadded convolutions and cropped the skip features to match.

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

## Sequences and text classification

We now turn from images to text. The running task is **news classification** on AG News: 120,000 training and 7,600 test articles in four classes (World, Sports, Business, Sci/Tech).[^zhang2015] This is what [Word-Embeddings-and-Text-Classification](https://github.com/parag1604/Word-Embeddings-and-Text-Classification) does.

### From words to vectors

**Tokenization.** Text is first split into **tokens**. The simplest choice is words: Parag's code lower-cases the text, keeps only letters, builds a vocabulary of the 20,000 most frequent words, and maps everything else to a special `<unk>` token. Modern language models use **subword** tokens instead (byte-pair encoding merges frequent character pairs repeatedly), so that rare words are spelled out from pieces and nothing is ever unknown.[^sennrich2016]

**One-hot vectors and embeddings.** Token $w$ out of a vocabulary of size $V$ can be written as a one-hot vector $\mathbf{e}_w\in\{0,1\}^V$. One-hot vectors are huge, and all of them are equally far apart: "cat" is no closer to "kitten" than to "carburettor". An **embedding** maps each token to a dense vector, $\mathbf{x}_w=\mathbf{E}^\top\mathbf{e}_w$, which is simply row $w$ of a learned matrix $\mathbf{E}\in\mathbb{R}^{V\times d}$. In code this is a table lookup (`nn.Embedding`), never an actual matrix product.

**Pretrained embeddings and the distributional hypothesis.** "You shall know a word by the company it keeps": words that occur in similar contexts tend to have similar meanings. Word2vec turns this into a prediction task.[^mikolov2013] The skip-gram model predicts the words around each word $w_t$ within a window of size $c$,

$$
\max\ \sum_{t}\ \sum_{\substack{-c\le j\le c\\ j\ne0}}\log p(w_{t+j}\mid w_t),
$$

where each word has an input vector $\mathbf{v}_w$ and an output vector $\mathbf{u}_w$, and

$$
p(o\mid w)=\frac{\exp(\mathbf{u}_o^\top\mathbf{v}_w)}{\sum_{o'}\exp(\mathbf{u}_{o'}^\top\mathbf{v}_w)}.
$$

The softmax sums over the whole vocabulary, which is expensive; **negative sampling** replaces it by a binary task: tell the true context word apart from a few random ones.[^mikolov2013b] This is a contrastive loss of the kind derived in [[Energy-Based Models]]. GloVe fits word vectors to global co-occurrence counts $X_{ij}$ instead, by minimizing a weighted least-squares loss on $\mathbf{w}_i^\top\tilde{\mathbf{w}}_j+b_i+\tilde b_j-\log X_{ij}$.[^pennington2014] fastText represents a word as the sum of its character $n$-gram vectors, so it can build vectors for words it has never seen.[^bojanowski2017]

The resulting spaces have striking geometry. Notebook 1 of Parag's repository loads the fastText vectors and finds that the nearest neighbours of "Bangalore" are other Indian cities (Chennai, Bengaluru, Pune, …), and that $\text{King}-\text{Man}+\text{Woman}$ lands closest to "Queen".

**A baseline: bag of embeddings.** Average the embeddings of the words in a sentence and feed the average to a linear classifier. This is fast and surprisingly strong for topic classification, but it ignores order: "not good, bad" and "not bad, good" get the same representation.

### Recurrent neural networks

A **recurrent neural network** reads the sequence $\mathbf{x}_1,\dots,\mathbf{x}_n$ one token at a time and keeps a running summary, the hidden state $\mathbf{h}_t$:

$$
\mathbf{h}_t=\tanh\big(\mathbf{W}\mathbf{x}_t+\mathbf{U}\mathbf{h}_{t-1}+\mathbf{b}\big).
$$

The same weights are used at every step: this is weight sharing **in time**, just as a convolution shares weights in space. For classification, the last hidden state $\mathbf{h}_n$ (or a pooled summary of all of them) goes to a softmax layer.

**Backpropagation through time.** Unrolled over $n$ steps, an RNN is a deep network with tied weights, and the gradient of a late hidden state with respect to an early one is a product of Jacobians:

$$
\frac{\partial\mathbf{h}_n}{\partial\mathbf{h}_t}=\prod_{m=t+1}^{n}\operatorname{diag}\big(1-\mathbf{h}_m^2\big)\,\mathbf{U}.
$$

Since $|\tanh'|\le1$, its norm is at most $\lVert\mathbf{U}\rVert^{\,n-t}$. If $\lVert\mathbf{U}\rVert<1$ the gradient **vanishes** exponentially with the distance $n-t$, so the network cannot learn long-range dependencies; if the largest singular value of $\mathbf{U}$ is large, the gradient can **explode**.[^bengio1994] Exploding gradients are tamed by **gradient clipping** (rescale the gradient when its norm exceeds a threshold). Vanishing gradients need a change of architecture.

### LSTM and GRU

The **long short-term memory** cell adds a second state, the **cell state** $\mathbf{c}_t$, which is updated *additively* and controlled by **gates**: sigmoid-valued vectors in $(0,1)$ that decide how much of something passes through.[^hochreiter1997]<sup>,</sup>[^gers2000] With $\sigma$ the sigmoid and $\odot$ the elementwise product, the forget gate $\mathbf{f}_t$, input gate $\mathbf{i}_t$, candidate $\mathbf{g}_t$ and output gate $\mathbf{o}_t$ are

$$
\begin{aligned}
\mathbf{f}_t&=\sigma(\mathbf{W}_f\mathbf{x}_t+\mathbf{U}_f\mathbf{h}_{t-1}+\mathbf{b}_f)\\
\mathbf{i}_t&=\sigma(\mathbf{W}_i\mathbf{x}_t+\mathbf{U}_i\mathbf{h}_{t-1}+\mathbf{b}_i)\\
\mathbf{g}_t&=\tanh(\mathbf{W}_g\mathbf{x}_t+\mathbf{U}_g\mathbf{h}_{t-1}+\mathbf{b}_g)\\
\mathbf{o}_t&=\sigma(\mathbf{W}_o\mathbf{x}_t+\mathbf{U}_o\mathbf{h}_{t-1}+\mathbf{b}_o)\\
\mathbf{c}_t&=\mathbf{f}_t\odot\mathbf{c}_{t-1}+\mathbf{i}_t\odot\mathbf{g}_t\\
\mathbf{h}_t&=\mathbf{o}_t\odot\tanh(\mathbf{c}_t)
\end{aligned}
$$

![[Machine Learning/images/applications-of-neural-networks-lstm-cell.svg]]
*Figure 5: an LSTM cell. The cell state (top) is only scaled by the forget gate $f_t$ and added to; the input gate $i_t$ decides how much of the candidate $g_t$ to write, and the output gate $o_t$ how much of the squashed cell state to expose as $h_t$.*

The forget gate decides what to erase, the input gate what to write, and the output gate what to reveal. The important design choice is that the cell state path has **no squashing non-linearity and no weight matrix**.

> [!proof]- Why the cell state carries gradients over long distances
> Differentiate $\mathbf{c}_t=\mathbf{f}_t\odot\mathbf{c}_{t-1}+\mathbf{i}_t\odot\mathbf{g}_t$ with respect to $\mathbf{c}_{t-1}$ along the direct path (treating the gates as given):
>
> $$
> \frac{\partial\mathbf{c}_t}{\partial\mathbf{c}_{t-1}}\Big|_{\text{direct}}=\operatorname{diag}(\mathbf{f}_t),
> \qquad
> \frac{\partial\mathbf{c}_n}{\partial\mathbf{c}_t}\Big|_{\text{direct}}=\operatorname{diag}\Big(\prod_{m=t+1}^{n}\mathbf{f}_m\Big).
> $$
>
> Compare with the vanilla RNN, where each step multiplies by $\operatorname{diag}(1-\mathbf{h}_m^2)\mathbf{U}$, a matrix the network cannot easily keep near the identity. Here each step multiplies by the forget gate, which the network *chooses*. When it keeps $\mathbf{f}_m\approx1$ on some coordinates, information and gradient pass along those coordinates almost unchanged for many steps. The full Jacobian also contains terms through the gates' dependence on $\mathbf{h}_{t-1}$, but the direct path is what prevents systematic vanishing. This is the same idea as the residual connection, $\mathbf{h}_{l+1}=\mathbf{h}_l+F(\mathbf{h}_l)$, applied in time, and it is why the forget-gate bias is often initialized to a positive value: the cell starts out remembering.

> [!example]- Code: one LSTM step from the equations (checked against PyTorch)
> ```python
> import numpy as np
>
> def lstm_cell(x, h, c, W, U, b):
>     """W: (4H, D), U: (4H, H), b: (4H,); gate order i, f, g, o (as in PyTorch)."""
>     z = W @ x + U @ h + b
>     i, f, g, o = np.split(z, 4)
>     sig = lambda t: 1 / (1 + np.exp(-t))
>     i, f, g, o = sig(i), sig(f), np.tanh(g), sig(o)
>     c = f * c + i * g
>     h = o * np.tanh(c)
>     return h, c
> ```
> With the weights of a `torch.nn.LSTMCell` (and `b = bias_ih + bias_hh`), this reproduces its $\mathbf{h}_t$ and $\mathbf{c}_t$ exactly.

The **gated recurrent unit** (GRU) is a lighter variant with two gates and no separate cell state: an update gate $\mathbf{z}_t$ interpolates between the old state and a candidate, $\mathbf{h}_t=(1-\mathbf{z}_t)\odot\mathbf{h}_{t-1}+\mathbf{z}_t\odot\tilde{\mathbf{h}}_t$, and a reset gate controls how much of $\mathbf{h}_{t-1}$ enters the candidate.[^cho2014] It often performs similarly to the LSTM.

### Batches, padding and masking

Sentences have different lengths, but a batch is a rectangular tensor. Short sequences are **padded** with a special token up to a fixed length (Parag's code uses 64 tokens), and the padding must not change the answer.

- **Where to pad for an RNN.** If the classifier reads the last hidden state, pad at the **front**, so the real tokens come last and the final state has just seen them. Notebook 2 does this. Padding at the end would make the network's last several steps read only padding.
- **Pooling must ignore padding.** Mean pooling over all positions dilutes the representation with padding vectors, and by different amounts for short and long sentences. With a mask $m_t=1$ for real tokens and $0$ for padding, use $\bar{\mathbf{h}}=\sum_t m_t\mathbf{h}_t\big/\sum_t m_t$.
- **Attention must ignore padding.** In a transformer, a padding mask stops every token from attending to padding positions (next section).

On AG News, Parag's LSTM on embeddings learned from scratch reached 91.3% test accuracy, and the same LSTM on frozen 100-dimensional GloVe embeddings 92.0% (both numbers from an earlier version of the notebooks that dropped unknown words instead of mapping them to `<unk>`). Pretrained embeddings help a little here because the training set is already large; they help much more when labelled data is scarce.

## Attention and transformers

An RNN squeezes everything it has read into one fixed-size vector $\mathbf{h}_t$. For long inputs this is a bottleneck, and information from early tokens has to survive many steps to be used. **Attention** removes the bottleneck: whenever the model needs information, it looks back at *all* stored vectors and takes a weighted average of the relevant ones. Bahdanau et al. introduced it for machine translation, where the decoder attends over all encoder states when producing each output word.[^bahdanau2015] The **transformer** then dropped recurrence altogether and built the whole network from attention.[^vaswani2017]

### Attention as a soft dictionary lookup

A Python dictionary stores (key, value) pairs; a **query** that exactly matches one key returns that key's value. Attention is a soft, differentiable version. Given a query $\mathbf{q}$ and pairs $(\mathbf{k}_1,\mathbf{v}_1),\dots,(\mathbf{k}_n,\mathbf{v}_n)$, it scores the query against every key, turns the scores into weights with a softmax, and returns the weighted average of the values:

$$
\alpha_i=\frac{\exp s(\mathbf{q},\mathbf{k}_i)}{\sum_{j=1}^{n}\exp s(\mathbf{q},\mathbf{k}_j)},\qquad
\operatorname{attn}(\mathbf{q})=\sum_{i=1}^{n}\alpha_i\mathbf{v}_i.
$$

Every step is differentiable, so the network can learn what to store as keys and values and what to ask for.

This is an old idea in new clothes. With $s(\mathbf{q},\mathbf{k})=-\lVert\mathbf{q}-\mathbf{k}\rVert^2/2h^2$ it is exactly the **Nadaraya–Watson** kernel regression of [[Non-parametric Models]], a locally weighted average of stored targets. Expanding the square,

$$
-\frac{\lVert\mathbf{q}-\mathbf{k}_i\rVert^2}{2h^2}=\frac{\mathbf{q}^\top\mathbf{k}_i}{h^2}-\frac{\lVert\mathbf{k}_i\rVert^2}{2h^2}-\frac{\lVert\mathbf{q}\rVert^2}{2h^2}.
$$

The last term is the same for every $i$ and cancels in the softmax. If all keys have the same norm, the middle term cancels too, and what remains is a **dot-product score** with temperature $h^2$. The [[Energy-Based Models]] lecture shows the same update as one retrieval step of a modern Hopfield network.[^ramsauer2021] Bahdanau's original score was a small MLP, $s(\mathbf{q},\mathbf{k})=\mathbf{v}^\top\tanh(\mathbf{W}_1\mathbf{q}+\mathbf{W}_2\mathbf{k})$ ("additive" attention); the dot product is cheaper and maps onto fast matrix multiplication.

### Scaled dot-product attention

Stack $n_q$ queries as the rows of $\mathbf{Q}\in\mathbb{R}^{n_q\times d_k}$, and $n$ keys and values as the rows of $\mathbf{K}\in\mathbb{R}^{n\times d_k}$ and $\mathbf{V}\in\mathbb{R}^{n\times d_v}$. All the lookups happen at once:

$$
\operatorname{Attention}(\mathbf{Q},\mathbf{K},\mathbf{V})=\operatorname{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V},
\tag{3}
$$

with the softmax applied to each row. Row $i$ of the $n_q\times n$ matrix $\mathbf{A}=\operatorname{softmax}(\cdot)$ holds the weights that query $i$ puts on each key, and it sums to 1.

**Why divide by $\sqrt{d_k}$?** Suppose the entries of $\mathbf{q}$ and $\mathbf{k}$ are independent with mean 0 and variance 1 (roughly what normalization layers and a good initialization give). Then

$$
\operatorname{Var}(\mathbf{q}^\top\mathbf{k})=\sum_{j=1}^{d_k}\operatorname{Var}(q_jk_j)=\sum_{j=1}^{d_k}\mathbb{E}[q_j^2]\,\mathbb{E}[k_j^2]=d_k.
$$

So raw dot products have standard deviation $\sqrt{d_k}$, which grows with the width. Large scores push the softmax into **saturation**: one weight near 1, the rest near 0. A saturated softmax has a vanishing gradient, because its Jacobian

$$
\frac{\partial\alpha_i}{\partial z_j}=\alpha_i\,(\delta_{ij}-\alpha_j)
$$

is zero when $\boldsymbol\alpha$ is one-hot. Dividing by $\sqrt{d_k}$ restores unit variance whatever the width. Figure 6 shows this numerically: with 32 keys and $d_k=512$, the largest unscaled attention weight averages 0.94 and the softmax Jacobian has a third of the norm it has with scaling.

![[Machine Learning/images/applications-of-neural-networks-attention-scaling.svg]]
*Figure 6: one query against 32 keys with standard-normal entries, averaged over 400 draws. (a) The standard deviation of raw scores grows like $\sqrt{d_k}$, while scaled scores stay at 1. (b) Without scaling the softmax becomes nearly one-hot as $d_k$ grows. (c) Its Jacobian, and with it the gradient that reaches the queries and keys, shrinks accordingly; with scaling both stay constant.*

### Multi-head attention

One attention operation produces one weighted average per query, so it can only "look for" one kind of thing at a time. **Multi-head attention** runs $h$ attentions in parallel on different learned projections and concatenates the results.[^vaswani2017] For an input $\mathbf{X}\in\mathbb{R}^{n\times d}$ and head $r=1,\dots,h$,

$$
\begin{aligned}
\operatorname{head}_r&=\operatorname{Attention}\big(\mathbf{X}\mathbf{W}_Q^{(r)},\ \mathbf{X}\mathbf{W}_K^{(r)},\ \mathbf{X}\mathbf{W}_V^{(r)}\big),\\
\operatorname{MHA}(\mathbf{X})&=\big[\operatorname{head}_1,\dots,\operatorname{head}_h\big]\,\mathbf{W}_O,
\end{aligned}
$$

with $\mathbf{W}_Q^{(r)},\mathbf{W}_K^{(r)},\mathbf{W}_V^{(r)}\in\mathbb{R}^{d\times d/h}$ and $\mathbf{W}_O\in\mathbb{R}^{d\times d}$. Each head works in a $d/h$-dimensional subspace, so the total cost and the parameter count ($4d^2$) are the same as for a single full-width head. Different heads can specialize: one may track the previous token, another a syntactic dependency, another a coreference.

- In **self-attention**, queries, keys and values all come from the same sequence $\mathbf{X}$: every token looks at every token of its own sequence.
- In **cross-attention**, the queries come from one sequence and the keys and values from another, for example a decoder attending over an encoder's outputs, or caption words attending over image patches.

> [!example]- Code: scaled dot-product and multi-head attention (checked against PyTorch)
> ```python
> import torch
>
> def attention(Q, K, V, mask=None):
>     """Q: (..., n_q, d_k), K: (..., n_k, d_k), V: (..., n_k, d_v).
>     mask: boolean (..., n_q, n_k), True = may attend."""
>     scores = Q @ K.transpose(-2, -1) / K.shape[-1] ** 0.5        # eq. (3)
>     if mask is not None:
>         scores = scores.masked_fill(~mask, float("-inf"))
>     A = torch.softmax(scores, dim=-1)            # each row sums to 1
>     return A @ V, A
>
> def multi_head(X, Wq, Wk, Wv, Wo, h, mask=None):
>     """X: (n, d). Wq, Wk, Wv, Wo: (d, d). h heads of size d/h."""
>     n, d = X.shape
>     split = lambda M: M.view(n, h, d // h).transpose(0, 1)   # (h, n, d/h)
>     Q, K, V = split(X @ Wq), split(X @ Wk), split(X @ Wv)
>     out, A = attention(Q, K, V, mask)            # (h, n, d/h)
>     return out.transpose(0, 1).reshape(n, d) @ Wo, A
> ```
> `attention` matches `torch.nn.functional.scaled_dot_product_attention` with and without a causal mask, and `multi_head` with the weights of a `torch.nn.MultiheadAttention` (no biases) reproduces its output.

### Order: positional encodings

Self-attention has no notion of order. Permute the input tokens with a permutation matrix $\mathbf{P}$: the queries, keys and values are permuted the same way, the score matrix becomes $\mathbf{P}\mathbf{Q}\mathbf{K}^\top\mathbf{P}^\top$, its row-wise softmax is $\mathbf{P}\mathbf{A}\mathbf{P}^\top$, and

$$
\operatorname{SA}(\mathbf{P}\mathbf{X})=\mathbf{P}\mathbf{A}\mathbf{P}^\top\mathbf{P}\mathbf{V}=\mathbf{P}\operatorname{SA}(\mathbf{X}).
$$

Self-attention is **permutation-equivariant**: it treats its input as a set. For text we must therefore add the position of each token to its embedding.

**Sinusoidal encodings** (the original transformer) use, for position $t$ and dimension pairs $j=0,\dots,d/2-1$,

$$
\mathrm{PE}_{t,2j}=\sin\!\left(\frac{t}{10000^{2j/d}}\right),\qquad
\mathrm{PE}_{t,2j+1}=\cos\!\left(\frac{t}{10000^{2j/d}}\right).
$$

Each pair of dimensions is a clock hand turning at its own frequency, from fast to very slow. Moving from position $t$ to $t+\delta$ rotates every pair by a fixed angle that depends only on $\delta$, so relative offsets are a linear function of the encodings and are easy for attention to pick up. Two alternatives are common:

- **Learned position embeddings** are simply a trainable vector per position (BERT and ViT use them).[^devlin2019] They cannot extrapolate beyond the longest training length.
- **Relative schemes** such as rotary embeddings (RoPE) rotate queries and keys by position-dependent angles, so the score $\mathbf{q}_i^\top\mathbf{k}_j$ depends only on the offset $i-j$.[^su2021] Most current language models use them.

### The transformer block

A transformer is a stack of identical blocks, each with two sub-layers: multi-head self-attention, which **mixes information across tokens**, and a position-wise **feed-forward network** (FFN), which **processes each token on its own**. Both sit on a residual connection with layer normalization (Figure 7a). In the now-standard **pre-norm** form,[^xiong2020]

$$
\begin{aligned}
\mathbf{X}'&=\mathbf{X}+\operatorname{MHA}\big(\operatorname{LN}(\mathbf{X})\big),\\
\mathbf{X}''&=\mathbf{X}'+\operatorname{FFN}\big(\operatorname{LN}(\mathbf{X}')\big).
\end{aligned}
$$

- The **FFN** is a two-layer MLP applied to every token separately, $\operatorname{FFN}(\mathbf{x})=\mathbf{W}_2\,\phi(\mathbf{W}_1\mathbf{x}+\mathbf{b}_1)+\mathbf{b}_2$, with a hidden width of typically $4d$ and a ReLU or GELU non-linearity $\phi$.
- **LayerNorm** normalizes each token's vector over its $d$ features, $\operatorname{LN}(\mathbf{x})=\boldsymbol\gamma\odot(\mathbf{x}-\mu)/\sqrt{\sigma^2+\epsilon}+\boldsymbol\beta$, where $\mu$ and $\sigma^2$ are the mean and variance of the entries of $\mathbf{x}$.[^ba2016] Unlike batch norm it does not depend on the other examples in the batch or on the sequence length.
- The **residual connections** form a "residual stream" that runs from the embeddings to the output; each sub-layer reads from it and adds its result back, exactly as in ResNet. The original transformer applied LayerNorm after the addition (post-norm); putting it before each sub-layer makes deep stacks easier to train without a careful learning-rate warm-up.

A block has about $4d^2$ attention parameters and $8d^2$ FFN parameters, so roughly $12d^2$ in total.

![[Machine Learning/images/applications-of-neural-networks-transformer-block.svg]]
*Figure 7: (a) a pre-norm encoder block: self-attention mixes tokens, the feed-forward network transforms each token, and each sub-layer adds its output to the residual stream. (b) A decoder block adds a causal mask to self-attention and a cross-attention sub-layer whose keys and values come from the encoder.*

### Encoders, decoders and masks

Transformers come in three arrangements.

- **Encoder-only** (BERT): every token attends to every other token in both directions. Good for understanding a whole input: classification, tagging, retrieval.
- **Decoder-only** (GPT): each token may attend only to itself and earlier tokens, so the model can be trained to predict the next token and can then generate text one token at a time.
- **Encoder–decoder** (the original transformer, translation, captioning): the decoder has masked self-attention plus cross-attention to the encoder output (Figure 7b).

The restriction to earlier tokens is a **causal mask** added to the scores before the softmax:

$$
M_{ij}=\begin{cases}0&\text{if }j\le i,\\-\infty&\text{if }j>i,\end{cases}
\qquad
\mathbf{A}=\operatorname{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}+\mathbf{M}\right).
$$

A $-\infty$ score becomes a weight of exactly 0. A **padding mask** works the same way, with $-\infty$ in the columns of padding tokens. Because of the causal mask, a decoder can be trained on all positions of a sequence **in parallel**, feeding in the true previous tokens; an RNN must step through them one by one.

**Cost.** Self-attention over $n$ tokens computes an $n\times n$ score matrix: $O(n^2d)$ time and $O(n^2)$ memory per head, against $O(nd^2)$ for an RNN layer. The payoff is that any two tokens are one step apart (an RNN needs $n$ steps) and that all positions are processed at once on parallel hardware. The quadratic cost is why long inputs are expensive and why many efficient variants, and alternative sequence models such as state-space models, exist.

### A transformer encoder for text classification

To classify a sentence with an encoder, we need one vector for the whole sequence. Two options are common:

- prepend a learnable **[CLS] token** and classify from its output vector, which can attend to every word (BERT);[^devlin2019]
- **mean-pool** the token outputs, over the real tokens only.

Notebook 4 of Parag's repository replaces the LSTM by three transformer encoder layers on frozen GloVe embeddings, with sinusoidal positional encodings, a padding mask, and masked mean pooling. It reached 92.3% test accuracy on AG News (in an earlier version that had neither positional encodings nor masked pooling), slightly above both LSTMs. For topic classification word order matters less than for, say, sentiment, so the three models are close; the transformer's advantages show on harder tasks and at scale.

## Image captioning

Captioning combines the two halves of this lecture: an image goes in, a sentence comes out.

### Encoder–decoder models

An **encoder** turns the image $I$ into features, and a **decoder** generates the caption $w_1,\dots,w_T$ one token at a time, each conditioned on the image and on the words so far:[^vinyals2015]

$$
p(w_{1:T}\mid I)=\prod_{t=1}^{T}p\big(w_t\mid w_{<t},I\big).
$$

Training maximizes the log-likelihood of human captions. With **teacher forcing**, the decoder is fed the *true* previous word at each step rather than its own prediction:

$$
\mathcal{L}=-\sum_{t=1}^{T}\log p\big(w_t^\star\mid w_{<t}^\star,I\big).
$$

This is fast and stable, but at test time the model must condition on its own (possibly wrong) earlier words, a situation it never saw in training. This mismatch is called **exposure bias**.

**Attention over the image.** Show and Tell fed one global image vector to an LSTM at the first step.[^vinyals2015] Show, Attend and Tell kept a grid of feature vectors, one per image region, and let the decoder **attend** over them at every step, so each word can look at a different part of the image.[^xu2015] This is cross-attention: the decoder state is the query, the image regions are keys and values. A transformer decoder does the same with multi-head cross-attention in every block.

Parag's [Image-Captioning](https://github.com/parag1604/Image-Captioning) repository uses a frozen, pretrained ViT-B/16 as the encoder (the next section explains ViTs): it turns a $224\times224$ image into a [CLS] vector plus 196 patch vectors. Four decoders are compared on Flickr8k:

1. a plain **LSTM**;
2. an LSTM with **dot-product attention** over the 196 patch vectors, its state initialized from the [CLS] vector;
3. an LSTM with **Bahdanau** (additive) attention;
4. a three-layer **transformer decoder** with causal self-attention and cross-attention to the patches.

![[Machine Learning/images/applications-of-neural-networks-captioning-attention.png]]
*Figure 8: attention weights of the dot-product attention LSTM (decoder 2) over the ViT patch grid, one panel per generated word, on a Flickr8k validation image. Some words attend where one would expect (the final "dog" on the white dog's body); others spread their attention over the background. Attention maps are a useful but loose explanation of what the model uses. Result from Parag's Image-Captioning repository.*

### Decoding

At test time we want a likely caption, but searching over all sentences is impossible.

- **Greedy decoding** takes the most probable word at each step. It is fast but can paint itself into a corner: a good first word may lead to poor continuations. Parag's code decodes greedily.
- **Beam search** keeps the $B$ most probable partial captions (by summed log-probability) at each step, extends each by every word, and again keeps the best $B$. A small beam ($B$ of 3 to 5) usually gives better captions than greedy decoding. Since every extra word adds a negative log-probability, scores are often length-normalized so that beam search does not favour short captions.

### Evaluation: BLEU

Captions are scored against several human references with **BLEU**, a machine-translation metric.[^papineni2002] Let $p_n$ be the **modified $n$-gram precision**: the fraction of the candidate's $n$-grams that also appear in a reference, with each $n$-gram counted at most as often as it appears in any single reference (so "the the the the" cannot score well). With candidate length $c$ and reference length $r$,

$$
\operatorname{BLEU}=\operatorname{BP}\cdot\exp\!\Big(\sum_{n=1}^{4}\tfrac14\log p_n\Big),\qquad
\operatorname{BP}=\min\big(1,\,e^{\,1-r/c}\big).
$$

The brevity penalty BP punishes captions that are too short to be judged by precision alone. BLEU is cheap and correlates with human judgment over a whole test set, but it rewards word overlap, not meaning: a correct caption phrased differently from all references scores poorly.

![[Machine Learning/images/applications-of-neural-networks-captioning-bleu.png]]
*Figure 9: validation BLEU of the four decoders on Flickr8k during training (5-epoch moving average). The Bahdanau-attention LSTM and the transformer decoder score highest; the plain LSTM is lowest for most of training. Result from Parag's Image-Captioning repository.*

## The Vision Transformer

If transformers work so well on sequences of words, can they work on images directly, without convolutions? The **Vision Transformer** (ViT) says yes, with one simple trick: **cut the image into patches and treat each patch as a token**.[^dosovitskiy2021]

### Patches as tokens

An image of size $C\times H\times W$ is split into $N=HW/P^2$ non-overlapping patches of $P\times P$ pixels (Figure 10). Each patch is flattened into a vector of length $P^2C$ and mapped to the model width $d$ by a learned matrix $\mathbf{E}\in\mathbb{R}^{P^2C\times d}$. A learnable [CLS] token is prepended, learned position embeddings are added, and the sequence goes through a standard transformer encoder:

$$
\begin{aligned}
\mathbf{z}_0&=\big[\mathbf{x}_{\text{cls}};\ \mathbf{x}_1\mathbf{E};\ \dots;\ \mathbf{x}_N\mathbf{E}\big]+\mathbf{E}_{\text{pos}},\\
\mathbf{z}_\ell&=\operatorname{Block}_\ell(\mathbf{z}_{\ell-1}),\qquad\ell=1,\dots,L,\\
\hat{\mathbf{y}}&=\operatorname{softmax}\big(\operatorname{head}(\operatorname{LN}(\mathbf{z}_L^{\text{cls}}))\big).
\end{aligned}
$$

For ViT-B/16 on a $224\times224$ image, $P=16$ gives $14\times14=196$ patch tokens. The patch embedding is the same as a convolution with kernel size and stride both equal to $P$, and that is how it is usually implemented.

![[Machine Learning/images/applications-of-neural-networks-vit-patches.svg]]
*Figure 10: (a) an image is cut into a grid of patches (here $4\times4$). (b) The patches, read row by row, become a sequence of tokens; a [CLS] token is prepended, each token is linearly embedded and gets a learned position embedding, a transformer encoder processes the sequence, and a head on the [CLS] output gives the class.*

> [!example]- Code: sinusoidal positional encodings and patchification (checked)
> ```python
> import numpy as np
>
> def sinusoidal_pe(n, d):
>     pos = np.arange(n)[:, None]
>     freq = 1.0 / 10000 ** (np.arange(0, d, 2) / d)
>     pe = np.zeros((n, d))
>     pe[:, 0::2] = np.sin(pos * freq)
>     pe[:, 1::2] = np.cos(pos * freq)
>     return pe
>
> def patchify(img, P):
>     """img: (C, H, W) -> (N, P*P*C) with N = (H/P)(W/P), row-major patches."""
>     C, H, W = img.shape
>     x = img.reshape(C, H // P, P, W // P, P)
>     return x.transpose(1, 3, 2, 4, 0).reshape((H // P) * (W // P), P * P * C)
> ```
> Checks: shifting the position by $\delta$ rotates each $(\sin,\cos)$ pair by the angle $\delta\cdot$freq; `patchify` agrees with `torch.Tensor.unfold`; and `patchify(img, P) @ E` equals a `conv2d` with kernel size and stride $P$ whose weights are $\mathbf{E}$ reshaped.

### Inductive bias and data

A ViT has much less built-in knowledge than a CNN. Only the patch embedding and the per-token MLPs are local; the 2-D layout is not given at all, and the position embeddings have to *learn* which patches are neighbours. The original paper found that, trained on ImageNet alone, ViTs fall a few points short of ResNets of similar size, but after pre-training on much larger datasets (14 to 300 million images) they match or beat them.[^dosovitskiy2021] With little data, the CNN's assumptions help; with enough data, the ViT's flexibility wins.

**Training ViTs on small datasets.** DeiT showed that a ViT can be trained competitively on ImageNet alone with a strong recipe: heavy augmentation (RandAugment, Mixup, CutMix), regularization such as label smoothing and stochastic depth, AdamW with warm-up and cosine decay, and optionally distillation from a CNN teacher through an extra token.[^touvron2021]<sup>,</sup>[^cubuk2020]<sup>,</sup>[^szegedy2016]

Parag's [improved-vision-encoders](https://github.com/paragduttaiisc/improved-vision-encoders) repository takes a different route to the same problem: instead of a heavy supervised recipe, the ViT encoder is pretrained **without labels**, by masking 75% of the patches and predicting the latent features of the masked patches from the visible ones (a joint-embedding predictive architecture). The quality of the learned features is then measured with a linear probe on ImageNet-1K. The repository also makes the encoder's parts swappable, such as attention versus an FFT-based token mixer, GELU versus SwiGLU, and dense versus mixture-of-experts feed-forward layers, so their effect on representation quality can be compared.

**Cost.** The number of tokens is $N=HW/P^2$, and attention costs $O(N^2)$. Halving the patch size quadruples $N$ and multiplies the attention cost by 16, which is why ViTs use fairly large patches and why hierarchical variants that attend within local windows exist for high-resolution tasks such as detection and segmentation.

## Key takeaways

- **Architectures encode assumptions.** Convolutions assume locality and translation equivariance, recurrence assumes sequential order, and attention assumes only that elements should exchange information based on content. Stronger assumptions need less data; weaker ones need more data or more regularization.
- A **convolution** is a sparse, weight-shared linear map. Its output size is $\lfloor(H+2p-k)/s\rfloor+1$, stacking layers grows the receptive field, and its gradient is again a convolution (with the flipped kernel). **Residual connections** give the gradient a direct path through very deep networks.
- **Localization** adds box regression to a classifier; **detection** scores many candidate boxes and removes duplicates with **NMS**; boxes are judged by **IoU** and detectors by **mAP**.
- **Segmentation** predicts every pixel with an encoder–decoder such as **U-Net**, whose skip connections restore spatial detail. The **soft Dice** loss handles class imbalance and stays differentiable, unlike the thresholded metric.
- Text is tokenized and **embedded**. **LSTMs** avoid vanishing gradients through an additive, gated cell state. Padding must be masked out of pooling and attention.
- **Attention** is a differentiable dictionary lookup. Scaling the scores by $1/\sqrt{d_k}$ keeps the softmax out of saturation. Self-attention is permutation-equivariant, so it needs **positional encodings**. A **transformer block** is attention + FFN, each on a residual stream with LayerNorm, and costs $O(n^2d)$.
- **Captioning** is conditional language modelling with an image encoder, trained with teacher forcing and decoded greedily or by beam search. The **Vision Transformer** turns patches into tokens; it needs large datasets or a strong augmentation and regularization recipe.

## Where this shows up next

- **Large language models.** A decoder-only transformer trained to predict the next token, scaled up, is a GPT.[^brown2020] [GPT-from-Scratch](https://github.com/parag1604/GPT-from-Scratch) builds one step by step, in the spirit of Karpathy's nanoGPT: a bigram model, one attention head, then a full transformer on tiny Shakespeare, a BPE tokenizer, and a key-value cache for fast generation. It then loads the real pretrained GPT-2 weights into the same code and fine-tunes them for tweet sentiment with low-rank adaptation (LoRA), training only small low-rank updates of the weight matrices.
- **Beyond attention.** State-space models process sequences in linear time while keeping long memory; [Mamba-in-JAX](https://github.com/parag1604/Mamba-in-JAX) goes from a linear SSM to Mamba's selective SSM. The [[Markov Chains]] lecture connects linear state-space models to Kalman filtering.
- **Contrastive representation learning.** CLIP trains an image encoder and a text transformer so that matching image–caption pairs have high similarity, with the contrastive loss derived in [[Energy-Based Models]].[^radford2021]
- **Generative models.** The decoders of [[Variational Autoencoders]] and the generators of [[Generative Adversarial Networks]] for images are built from transposed convolutions, and a U-Net is the standard noise- or score-prediction network in the diffusion models that follow [[Score-Based Models]].
- **The next lecture**, [[Non-parametric Models]], starts from the opposite end: no learned features at all, only distances to stored examples, which leads back to attention through Nadaraya–Watson.
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
[^zhang2015]: Zhang, X., Zhao, J., & LeCun, Y. (2015). *Character-level Convolutional Networks for Text Classification*. NeurIPS. [arXiv:1509.01626](https://arxiv.org/abs/1509.01626)
[^sennrich2016]: Sennrich, R., Haddow, B., & Birch, A. (2016). *Neural Machine Translation of Rare Words with Subword Units*. ACL. [arXiv:1508.07909](https://arxiv.org/abs/1508.07909)
[^mikolov2013]: Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). *Efficient Estimation of Word Representations in Vector Space*. ICLR Workshop. [arXiv:1301.3781](https://arxiv.org/abs/1301.3781)
[^mikolov2013b]: Mikolov, T., Sutskever, I., Chen, K., Corrado, G., & Dean, J. (2013). *Distributed Representations of Words and Phrases and their Compositionality*. NeurIPS. [arXiv:1310.4546](https://arxiv.org/abs/1310.4546)
[^pennington2014]: Pennington, J., Socher, R., & Manning, C. D. (2014). *GloVe: Global Vectors for Word Representation*. EMNLP. [ACL Anthology D14-1162](https://aclanthology.org/D14-1162/)
[^bojanowski2017]: Bojanowski, P., Grave, E., Joulin, A., & Mikolov, T. (2017). *Enriching Word Vectors with Subword Information*. TACL. [arXiv:1607.04606](https://arxiv.org/abs/1607.04606)
[^bengio1994]: Bengio, Y., Simard, P., & Frasconi, P. (1994). *Learning Long-Term Dependencies with Gradient Descent is Difficult*. IEEE Transactions on Neural Networks, 5(2), 157–166. [doi:10.1109/72.279181](https://doi.org/10.1109/72.279181)
[^hochreiter1997]: Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory*. Neural Computation, 9(8), 1735–1780. [doi:10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735)
[^gers2000]: Gers, F. A., Schmidhuber, J., & Cummins, F. (2000). *Learning to Forget: Continual Prediction with LSTM*. Neural Computation, 12(10), 2451–2471. [doi:10.1162/089976600300015015](https://doi.org/10.1162/089976600300015015)
[^cho2014]: Cho, K., van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). *Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation*. EMNLP. [arXiv:1406.1078](https://arxiv.org/abs/1406.1078)
[^bahdanau2015]: Bahdanau, D., Cho, K., & Bengio, Y. (2015). *Neural Machine Translation by Jointly Learning to Align and Translate*. ICLR. [arXiv:1409.0473](https://arxiv.org/abs/1409.0473)
[^vaswani2017]: Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). *Attention Is All You Need*. NeurIPS. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
[^ramsauer2021]: Ramsauer, H., et al. (2021). *Hopfield Networks is All You Need*. ICLR. [arXiv:2008.02217](https://arxiv.org/abs/2008.02217)
[^devlin2019]: Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL. [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)
[^su2021]: Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., & Liu, Y. (2021). *RoFormer: Enhanced Transformer with Rotary Position Embedding*. arXiv (Neurocomputing, 2024). [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)
[^xiong2020]: Xiong, R., et al. (2020). *On Layer Normalization in the Transformer Architecture*. ICML. [arXiv:2002.04745](https://arxiv.org/abs/2002.04745)
[^ba2016]: Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). *Layer Normalization*. arXiv. [arXiv:1607.06450](https://arxiv.org/abs/1607.06450)
[^vinyals2015]: Vinyals, O., Toshev, A., Bengio, S., & Erhan, D. (2015). *Show and Tell: A Neural Image Caption Generator*. CVPR. [arXiv:1411.4555](https://arxiv.org/abs/1411.4555)
[^xu2015]: Xu, K., Ba, J., Kiros, R., Cho, K., Courville, A., Salakhutdinov, R., Zemel, R., & Bengio, Y. (2015). *Show, Attend and Tell: Neural Image Caption Generation with Visual Attention*. ICML. [arXiv:1502.03044](https://arxiv.org/abs/1502.03044)
[^papineni2002]: Papineni, K., Roukos, S., Ward, T., & Zhu, W.-J. (2002). *BLEU: a Method for Automatic Evaluation of Machine Translation*. ACL. [ACL Anthology P02-1040](https://aclanthology.org/P02-1040/)
[^dosovitskiy2021]: Dosovitskiy, A., et al. (2021). *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*. ICLR. [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
[^touvron2021]: Touvron, H., Cord, M., Douze, M., Massa, F., Sablayrolles, A., & Jégou, H. (2021). *Training data-efficient image transformers & distillation through attention*. ICML. [arXiv:2012.12877](https://arxiv.org/abs/2012.12877)
[^cubuk2020]: Cubuk, E. D., Zoph, B., Shlens, J., & Le, Q. V. (2020). *RandAugment: Practical automated data augmentation with a reduced search space*. CVPR Workshops. [arXiv:1909.13719](https://arxiv.org/abs/1909.13719)
[^szegedy2016]: Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., & Wojna, Z. (2016). *Rethinking the Inception Architecture for Computer Vision* (introduces label smoothing). CVPR. [arXiv:1512.00567](https://arxiv.org/abs/1512.00567)
[^brown2020]: Brown, T. B., et al. (2020). *Language Models are Few-Shot Learners*. NeurIPS. [arXiv:2005.14165](https://arxiv.org/abs/2005.14165)
[^radford2021]: Radford, A., et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision*. ICML. [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)
[^prince2023]: Prince, S. J. D. (2023). *Understanding Deep Learning*. MIT Press. [udlbook.github.io](https://udlbook.github.io/udlbook/)
[^goodfellow2016]: Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. [deeplearningbook.org](https://www.deeplearningbook.org/)
[^zhang2023]: Zhang, A., Lipton, Z. C., Li, M., & Smola, A. J. (2023). *Dive into Deep Learning*. Cambridge University Press. [d2l.ai](https://d2l.ai/)
