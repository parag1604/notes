---
title: Sequence Models
tags:
  - neural-networks
  - deep-learning
  - transformers
  - state-space-models
  - mixture-of-experts
  - nlp
date: 2026-11-13
draft: false
prev: "[[Convolutional Neural Networks]]"
next: "[[Graph Neural Networks]]"
---

> **Prerequisites:** [[Neural Networks]] (MLPs, backpropagation, normalization layers), [[Linear Models]] (softmax regression and the cross-entropy loss) and [[Convolutional Neural Networks]] (weight sharing, residual connections). The section on editing facts uses [[Multivariable Calculus#Equality constraints|Lagrange multipliers]] and [[Orthogonality and Least Squares|least squares]]; the state space models use the matrix exponential from [[Eigenvalues and Eigenvectors#Differential equations|Eigenvalues and Eigenvectors]].
>
> **TL;DR:**
> - Text is a **sequence** of token **embeddings**. Every sequence model alternates two kinds of layer: **sequence mixing**, which moves information *between* tokens (recurrence, attention, convolution, state space models), and **channel mixing**, an MLP applied to *each* token on its own.
> - **RNNs and LSTMs** mix by reading tokens in order through a fixed-size state. **Attention** lets every token look up every other token by content; the **transformer** stacks attention and MLPs on a residual stream. Modern decoders add RoPE, RMSNorm, SwiGLU, a KV cache with grouped-query attention, and FlashAttention.
> - **Sequence mixing is only half of the story.** About two thirds of a transformer's parameters sit in its MLPs, which behave like **key–value memories**: facts can be located in them and even edited. **Mixture-of-experts** layers grow this memory without growing the compute per token.
> - Attention costs $O(n^2)$ in the sequence length. **Linear attention** and **state space models** (S4, Mamba) mix in linear time with a fixed-size state; Mamba's **selectivity** makes the state depend on content.
> - The same machinery reads images: **captioning** decodes text from image features, and the **Vision Transformer** treats image patches as tokens.
>
> **Code:** [Word-Embeddings-and-Text-Classification](https://github.com/parag1604/Word-Embeddings-and-Text-Classification): fastText vectors, then AG News classification with LSTMs and a transformer encoder; [GPT-from-Scratch](https://github.com/parag1604/GPT-from-Scratch): bigram model to GPT on tiny Shakespeare, BPE, a KV cache, GPT-2 weights loaded into the same code, and LoRA fine-tuning; [modern_llms](https://github.com/paragduttaiisc/modern_llms): nanoGPT extended to a modern billion-parameter LLM (RoPE, RMSNorm, SwiGLU, multi-head latent attention, mixture of experts, multi-token prediction with self-speculative decoding); [Mamba-in-JAX](https://github.com/parag1604/Mamba-in-JAX): from a linear SSM to S4D, the selective SSM and the full Mamba block; [Image-Captioning](https://github.com/parag1604/Image-Captioning): a ViT encoder with LSTM, attention-LSTM and transformer decoders on Flickr8k; [improved-vision-encoders](https://github.com/paragduttaiisc/improved-vision-encoders): a ViT encoder trained self-supervised (JEPA-style masked latent prediction) on ImageNet, with switchable token mixers (attention or FFT), FFN activations and mixture-of-experts layers.

> [!note] Local notation
> In this lecture $n$ is the **number of tokens** in a sequence (not the number of samples), $d$ is the **width** of a token's representation, $h$ is the number of attention heads, and $d_{\text{ff}}$ is the hidden width of an MLP. In the ViT section $N$ is the number of image patches. Keys are the matrix $\mathbf{K}$; $K$ alone is still the number of classes.

## Motivation

Images have a grid; text has an order. The meaning of a word depends on its neighbours, sentences have different lengths, and the **order** of words matters ("dog bites man" vs "man bites dog"). An MLP that takes a whole sentence as one long input vector would need a fixed length and a separate weight for every position, and it would have to relearn what "dog" means at each of them. [[Convolutional Neural Networks]] solved the analogous problem for images by sharing weights across space. Sequence models share weights across **time**.

Every architecture in this lecture is built from the same two ingredients, stacked many times (Figure 1):

- **Sequence mixing** moves information **between tokens**: the word "it" must find out what it refers to, and a verb must find its subject. Recurrence, attention, convolution and state space models are different ways of doing this.
- **Channel mixing** transforms **each token on its own**, with the same MLP at every position: it computes new features from the information the token has gathered.

Most discussions of transformers are about attention, the sequence-mixing half. One message of this lecture is that the other half matters at least as much: the MLPs hold most of the parameters, most of the compute at ordinary context lengths, and, as far as we can tell, most of the stored knowledge.

![[Machine Learning/images/sequence-models-mixing.svg]]
*Figure 1: the anatomy of one block of a modern sequence model. A residual stream carries one vector per token upward. The sequence mixer (attention, a state space model, a convolution or a recurrence) exchanges information across tokens; the channel mixer (an MLP, or a mixture of experts) processes each token separately with shared weights. Each sub-layer reads a normalized copy of the stream and adds its output back.*

The running example for the classical models is **news classification** on AG News: 120,000 training and 7,600 test articles in four classes (World, Sports, Business, Sci/Tech).[^zhang2015] This is what [Word-Embeddings-and-Text-Classification](https://github.com/parag1604/Word-Embeddings-and-Text-Classification) does. For the modern decoder we use the numbers of real models: GPT-2,[^radford2019] Llama 2[^touvron2023] and Mixtral.[^jiang2024]

## Sequences and text classification

We start with the classical pipeline: turn words into vectors, then mix them with a recurrent network.

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

The softmax sums over the whole vocabulary, which is expensive; **negative sampling** replaces it by a binary task: tell the true context word apart from a few random ones.[^mikolov2013b] This is a contrastive loss of the kind derived in [[Energy-Based Models]]. GloVe fits word vectors to global co-occurrence counts $X_{ij}$ instead, by minimizing a weighted least-squares loss on $\mathbf{w}_i^\top\tilde{\mathbf{w}}_j+b_i+\tilde b_j-\log X_{ij}$.[^pennington2014] fastText represents a word as the sum of its character $n$-gram vectors, so it can build vectors for words it has never seen.[^bojanowski2017] [[Language Representations and Language Models]] studies these objectives, and the geometry of the vectors they produce, in more depth.

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
*Figure 2: an LSTM cell. The cell state (top) is only scaled by the forget gate $f_t$ and added to; the input gate $i_t$ decides how much of the candidate $g_t$ to write, and the output gate $o_t$ how much of the squashed cell state to expose as $h_t$.*

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
\tag{1}
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

is zero when $\boldsymbol\alpha$ is one-hot. Dividing by $\sqrt{d_k}$ restores unit variance whatever the width. Figure 3 shows this numerically: with 32 keys and $d_k=512$, the largest unscaled attention weight averages 0.94 and the softmax Jacobian has a third of the norm it has with scaling.

![[Machine Learning/images/applications-of-neural-networks-attention-scaling.svg]]
*Figure 3: one query against 32 keys with standard-normal entries, averaged over 400 draws. (a) The standard deviation of raw scores grows like $\sqrt{d_k}$, while scaled scores stay at 1. (b) Without scaling the softmax becomes nearly one-hot as $d_k$ grows. (c) Its Jacobian, and with it the gradient that reaches the queries and keys, shrinks accordingly; with scaling both stay constant.*

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
>     scores = Q @ K.transpose(-2, -1) / K.shape[-1] ** 0.5        # eq. (1)
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
- **Relative schemes** make the attention score depend only on the **offset** between two tokens. The most widely used one, rotary embeddings, is next.

### Rotary position embeddings (RoPE)

Sinusoidal and learned encodings are **added** to the token embeddings, so position and content get mixed together before the queries and keys are even computed. The score between two tokens then contains cross terms between the content of one and the position of the other. What we would like is a score that depends on the two contents and on how far apart the tokens are, but not on where they sit in the sequence. **RoPE** achieves this by **rotating** queries and keys, instead of adding anything to them.[^su2021]

Start with $d=2$. Write $\mathbf{R}(\alpha)$ for the rotation of the plane by angle $\alpha$. A query at position $m$ is rotated by $m\theta$ and a key at position $n$ by $n\theta$, for a fixed frequency $\theta$. Rotations are orthogonal, $\mathbf{R}(\alpha)^\top=\mathbf{R}(-\alpha)$, and they compose by adding angles, $\mathbf{R}(\alpha)\mathbf{R}(\beta)=\mathbf{R}(\alpha+\beta)$. So the score is

$$
\big(\mathbf{R}(m\theta)\,\mathbf{q}\big)^\top\big(\mathbf{R}(n\theta)\,\mathbf{k}\big)
=\mathbf{q}^\top\mathbf{R}(-m\theta)\,\mathbf{R}(n\theta)\,\mathbf{k}
=\mathbf{q}^\top\mathbf{R}\big((n-m)\theta\big)\,\mathbf{k}.
\tag{2}
$$

It depends on the positions only through the offset $n-m$ (Figure 4a). In $d$ dimensions, RoPE splits $\mathbf{q}$ and $\mathbf{k}$ into $d/2$ pairs and rotates pair $j$ with its own frequency $\theta_j=10000^{-2j/d}$, the same geometric ladder of clock speeds as the sinusoidal encoding. The score is a sum of terms like (2), so the whole score depends only on $n-m$ (Figure 4b). Fast pairs resolve nearby positions; slow pairs distinguish long distances.

Three properties make RoPE the default in current language models (Llama, Mistral, and Parag's modern_llms):
- It is applied inside every attention layer, to queries and keys only. Values are not rotated, and nothing is added to the residual stream.
- Rotations preserve norms, so RoPE does not change the scale of the scores.
- Each step of a cached decoder only needs the absolute position of the new token, yet the scores come out relative.

**ALiBi** is a simpler relative scheme: no positional encoding at all, and a penalty proportional to distance added to every score, $s_{ij}-\lambda_r\,(i-j)$, with a fixed slope $\lambda_r$ per head forming a geometric sequence across heads.[^press2022] Distant tokens are down-weighted by design, which let models trained on short sequences run on longer ones.

![[Machine Learning/images/sequence-models-rope.svg]]
*Figure 4: rotary position embeddings. (a) In one pair of coordinates, the query at position $m=3$ is rotated by $3\theta$ and the key at $n=1$ by $\theta$; the angle between them, and with it their dot product, changes only by $(m-n)\theta$. (b) The score $\mathbf{q}_m^\top\mathbf{k}_n$ when the same random $\mathbf{q}$ and $\mathbf{k}$ ($d=64$) are placed at every position: it is constant along every diagonal (the largest spread along a diagonal is $3\times10^{-14}$, rounding error), so it depends only on $m-n$.*

> [!example]- Code: RoPE (checked numerically)
> ```python
> def rope(x, pos, base=10000.0):
>     """x: (..., n, d) with d even; pos: (n,) integer positions.
>     Rotates each pair (x[2j], x[2j+1]) by the angle pos * base**(-2j/d)."""
>     d = x.shape[-1]
>     theta = base ** (-torch.arange(0, d, 2, dtype=torch.float64) / d)   # (d/2,)
>     ang = pos[:, None].double() * theta                                  # (n, d/2)
>     cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
>     x1, x2 = x[..., 0::2], x[..., 1::2]
>     out = torch.empty_like(x)
>     out[..., 0::2] = x1 * cos - x2 * sin
>     out[..., 1::2] = x1 * sin + x2 * cos
>     return out
> ```
> With $d=64$ and random $\mathbf{q}$, $\mathbf{k}$, the scores at positions $(5,2)$, $(105,102)$ and $(1005,1002)$ agree to $10^{-9}$, position 0 leaves the vectors unchanged, and every norm is preserved.

### The transformer block

A transformer is a stack of identical blocks, each with two sub-layers: multi-head self-attention, which **mixes information across tokens**, and a position-wise **feed-forward network** (FFN), which **processes each token on its own**. Both sit on a residual connection with layer normalization (Figure 5a). In the now-standard **pre-norm** form,[^xiong2020]

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
*Figure 5: (a) a pre-norm encoder block: self-attention mixes tokens, the feed-forward network transforms each token, and each sub-layer adds its output to the residual stream. (b) A decoder block adds a causal mask to self-attention and a cross-attention sub-layer whose keys and values come from the encoder.*

### Encoders, decoders and masks

Transformers come in three arrangements.

- **Encoder-only** (BERT): every token attends to every other token in both directions. Good for understanding a whole input: classification, tagging, retrieval.
- **Decoder-only** (GPT): each token may attend only to itself and earlier tokens, so the model can be trained to predict the next token and can then generate text one token at a time.
- **Encoder–decoder** (the original transformer, translation, captioning): the decoder has masked self-attention plus cross-attention to the encoder output (Figure 5b).

The restriction to earlier tokens is a **causal mask** added to the scores before the softmax:

$$
M_{ij}=\begin{cases}0&\text{if }j\le i,\\-\infty&\text{if }j>i,\end{cases}
\qquad
\mathbf{A}=\operatorname{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}+\mathbf{M}\right).
$$

A $-\infty$ score becomes a weight of exactly 0. A **padding mask** works the same way, with $-\infty$ in the columns of padding tokens. Because of the causal mask, a decoder can be trained on all positions of a sequence **in parallel**, feeding in the true previous tokens; an RNN must step through them one by one.

> [!example]- Code: causal and padding masks, and masked mean pooling (PyTorch)
> ```python
> import torch
>
> def attention_mask(pad, causal):
>     """pad: (batch, n) bool, True at padding tokens. Returns the additive mask M, (batch, 1, n, n)."""
>     n = pad.shape[1]
>     M = torch.zeros(pad.shape[0], 1, n, n)
>     M = M.masked_fill(pad[:, None, None, :], float("-inf"))     # no query attends to padding keys
>     if causal:
>         future = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)   # entries j > i
>         M = M.masked_fill(future, float("-inf"))
>     return M                     # used as softmax(Q K^T / sqrt(d_k) + M), broadcast over heads
>
> def masked_mean(h, pad):
>     """Mean pooling over real tokens only: sum_t m_t h_t / sum_t m_t, with h of shape (batch, n, d)."""
>     m = (~pad).float().unsqueeze(-1)
>     return (h * m).sum(1) / m.sum(1)
> ```
> With a causal mask and padding at the *front*, a padding query can have every key masked, and its softmax row becomes NaN. Those rows are never used, but they must not reach the loss: pad at the end for decoders, or overwrite the masked rows.

**Cost.** Self-attention over $n$ tokens computes an $n\times n$ score matrix: $O(n^2d)$ time and $O(n^2)$ memory per head, against $O(nd^2)$ for an RNN layer. The payoff is that any two tokens are one step apart (an RNN needs $n$ steps) and that all positions are processed at once on parallel hardware. The quadratic cost is why long inputs are expensive and why many efficient variants, and alternative sequence models such as state space models, exist ([[Sequence Models#Beyond quadratic attention|below]]).

### The modern decoder block

The block of Figure 5 is the 2017 design with pre-norm. A decoder-only language model built in the last few years differs in a handful of details, each small, which together make it faster and more stable to train. Llama 2 is a representative example.[^touvron2023]

**The residual stream view.** Unroll the residual connections of an $L$-block network. The final representation of a token is its embedding plus everything every sub-layer ever added:

$$
\mathbf{x}_L=\mathbf{x}_0+\sum_{l=1}^{L}\Big(\operatorname{Attn}_l(\cdot)+\operatorname{MLP}_l(\cdot)\Big).
$$

Elhage et al. suggest reading the network this way: the residual stream is a shared **communication channel**, a $d$-dimensional vector per token, that every layer **reads** from (through a linear projection after the norm) and **writes** to (by adding a linear projection of its output).[^elhage2021] Attention heads are the only components that move information between positions; MLPs read and write at a single position. A later layer can use anything an earlier layer wrote, which is why features computed early stay available to the end. This is the view behind Figure 1, and it is the frame in which the evidence of the next section is collected.

**RMSNorm instead of LayerNorm.** LayerNorm subtracts the mean of the token's features and divides by their standard deviation. RMSNorm skips the mean and the learned shift:[^zhang2019]

$$
\operatorname{RMSNorm}(\mathbf{x})=\mathbf{g}\odot\frac{\mathbf{x}}{\sqrt{\tfrac1d\sum_{i=1}^{d}x_i^2+\epsilon}}.
$$

Zhang and Sennrich argued that the rescaling, which keeps activations at a controlled size, is what matters, and that re-centring adds computation without helping much. RMSNorm is cheaper and performed comparably in their experiments; most recent language models use it.

**SwiGLU instead of a ReLU or GELU MLP.** A **gated linear unit** multiplies two linear projections of the input elementwise, one of them passed through a non-linearity, so that one projection decides how much of the other passes.[^dauphin2017] Shazeer compared several variants and found that the one gated with the SiLU (Swish) function, $\operatorname{SiLU}(z)=z\,\sigma(z)$, gave the best language-modelling quality:[^shazeer2020]

$$
\operatorname{MLP}(\mathbf{x})=\mathbf{W}_{\text{down}}\Big(\operatorname{SiLU}(\mathbf{W}_{\text{gate}}\mathbf{x})\odot\mathbf{W}_{\text{up}}\mathbf{x}\Big).
\tag{3}
$$

It has three matrices instead of two. To keep the parameter count of the standard $4d$-wide MLP ($2\cdot4d^2=8d^2$), the hidden width is cut to $d_{\text{ff}}=\tfrac83d$, giving $3\cdot\tfrac83d^2=8d^2$. Llama 2 7B has $d=4096$ and $d_{\text{ff}}=11008$, which is $\tfrac83d$ rounded up to a multiple of 256. Shazeer offered no theoretical reason for the gain; it is an empirical finding that has held up across many models.

**No biases, and tied or untied embeddings.** Biases are dropped from the linear layers. GPT-2 reuses the input embedding matrix as the output layer ("tied" embeddings); Llama keeps them separate.

**RoPE** (above) and **grouped-query attention with a KV cache** (below) complete the picture.

> [!example]- Code: a modern decoder block with RMSNorm, RoPE, grouped-query attention, SwiGLU and a KV cache (checked)
> ```python
> class RMSNorm(torch.nn.Module):
>     def __init__(self, d, eps=1e-6):
>         super().__init__(); self.g = torch.nn.Parameter(torch.ones(d)); self.eps = eps
>     def forward(self, x):
>         return self.g * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
>
> class Block(torch.nn.Module):
>     """Pre-norm decoder block: grouped-query attention with RoPE, then a SwiGLU MLP."""
>     def __init__(self, d, n_heads, n_kv_heads, d_ff):
>         super().__init__()
>         self.h, self.g, self.dh = n_heads, n_kv_heads, d // n_heads
>         self.norm1, self.norm2 = RMSNorm(d), RMSNorm(d)
>         self.wq = torch.nn.Linear(d, d, bias=False)
>         self.wkv = torch.nn.Linear(d, 2 * n_kv_heads * self.dh, bias=False)
>         self.wo = torch.nn.Linear(d, d, bias=False)
>         self.w_gate = torch.nn.Linear(d, d_ff, bias=False)
>         self.w_up = torch.nn.Linear(d, d_ff, bias=False)
>         self.w_down = torch.nn.Linear(d_ff, d, bias=False)
>
>     def forward(self, x, cache=None):
>         """x: (n, d), the new tokens. cache: dict with past 'k', 'v' (or None)."""
>         n, d = x.shape
>         past = 0 if cache is None or "k" not in cache else cache["k"].shape[1]
>         pos = torch.arange(past, past + n)
>         a = self.norm1(x)
>         q = self.wq(a).view(n, self.h, self.dh).transpose(0, 1)             # (h, n, dh)
>         k, v = self.wkv(a).view(n, 2, self.g, self.dh).permute(1, 2, 0, 3)  # (g, n, dh) each
>         q, k = rope(q, pos), rope(k, pos)
>         if cache is not None:                                               # KV cache
>             if "k" in cache:
>                 k = torch.cat([cache["k"], k], 1); v = torch.cat([cache["v"], v], 1)
>             cache["k"], cache["v"] = k, v
>         k = k.repeat_interleave(self.h // self.g, 0)                        # share K, V
>         v = v.repeat_interleave(self.h // self.g, 0)                        # across a group
>         mask = torch.ones(n, past + n, dtype=torch.bool).tril(past)         # causal
>         o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
>         x = x + self.wo(o.transpose(0, 1).reshape(n, d))                    # sequence mixing
>         b = self.norm2(x)
>         return x + self.w_down(F.silu(self.w_gate(b)) * self.w_up(b))       # channel mixing
> ```
> Uses `rope` from the RoPE aside. With $d=64$, 8 query heads, 2 KV heads and $d_{\text{ff}}=172$, decoding 12 tokens one at a time through the cache (after a 5-token prefill) reproduces the full parallel forward pass to $4\times10^{-16}$, the cache holds only the 2 KV heads, and changing a later token leaves every earlier output unchanged.

### Where the parameters and the compute are

**Parameters.** In a block with model width $d$:
- attention has four $d\times d$ matrices, $\mathbf{W}_Q,\mathbf{W}_K,\mathbf{W}_V,\mathbf{W}_O$: about $4d^2$ parameters;
- the MLP has $2\cdot d\cdot4d=8d^2$ (or $3\cdot d\cdot\tfrac83d=8d^2$ for SwiGLU);
- the norms have a few $d$-vectors, which are negligible.

So each block has about $12d^2$ parameters, **two thirds of them in the MLP**, and a model with $L$ blocks and a vocabulary of $V$ tokens has about $12Ld^2+Vd$. For GPT-2 small ($d=768$, $L=12$, $V=50{,}257$) this gives 85M parameters in the blocks and 124M in total; for Llama 2 7B ($d=4096$, $L=32$, $V=32{,}000$, untied embeddings) it gives 6.74B. In both, the MLPs hold 66.7% of the non-embedding parameters (Figure 6a). Grouped-query attention shrinks $\mathbf{W}_K$ and $\mathbf{W}_V$ and mixture-of-experts layers multiply the MLPs, so in Mixtral the share of the MLPs rises to 97% of the non-embedding parameters (89% of the active ones).

**Compute.** A matrix–vector product with an $a\times b$ matrix costs $2ab$ floating-point operations (a multiply and an add per entry), so the forward pass costs about **2 FLOPs per parameter per token**, and training (forward plus backward) about 6.[^kaplan2020] Attention adds work that does not involve parameters: at context length $n$, the new token's query is compared with $n$ keys ($2nd$ FLOPs) and the $n$ values are averaged ($2nd$ more). Per token and layer,

$$
\underbrace{16d^2}_{\text{MLP}}+\underbrace{8d^2}_{\text{attention projections}}+\underbrace{4nd}_{\text{attention scores and sum}}.
$$

The attention term overtakes the MLP only when $n>4d$: about 3,000 tokens for GPT-2's width, and 16,000 for $d=4096$ (Figure 6b). For ordinary context lengths, the MLPs dominate the compute as well as the parameters. Beyond that, the quadratic term takes over, which is what the second half of this lecture is about.

![[Machine Learning/images/sequence-models-params.svg]]
*Figure 6: (a) the share of parameters in the embeddings, attention and MLPs, computed from the published shapes of GPT-2 small (124M), Llama 2 7B (6.74B) and Mixtral 8x7B (46.7B in total, 12.9B active per token, since each token uses 2 of 8 experts). The MLPs hold two thirds of the non-embedding parameters in the dense models and almost all of them in the mixture of experts. (b) Forward FLOPs per token in one layer of width $d=4096$, as a function of how many tokens precede it. The attention scores and weighted sum grow linearly with the context and catch up with the MLP at $n=4d=16{,}384$.*

### The KV cache and grouped-query attention

A decoder generates text one token at a time. Without care, each new token would re-run the network over the whole prefix. But with a causal mask, the keys and values of earlier tokens never change: a token cannot see the future, so nothing that comes later can alter them. Generation therefore stores them in a **KV cache**. It first runs the prompt once in parallel (the **prefill**), then pushes only the newest token through the network (the **decode** steps); each step computes one query, attends over the cached keys and values, and appends its own. Version 4 of [GPT-from-Scratch](https://github.com/parag1604/GPT-from-Scratch) implements exactly this, gets identical greedy outputs, and measures a 3.6× speed-up when generating 1,023 tokens.

The price is memory. Each token stores a key and a value of width $d$ in each of the $L$ layers: $2Ld$ numbers. For Llama 2 7B in 16-bit precision that is $2\cdot32\cdot4096\cdot2$ bytes $=512$ KiB **per token**, 2 GiB for a 4,096-token sequence, and 64 GiB for 128k tokens, far more than the 13.5 GB of weights (Figure 9b). Worse, each decode step does very little arithmetic per byte it reads: it must load all the weights and the whole cache to produce one token. Decoding is limited by **memory bandwidth**, not by compute, so a smaller cache is a faster model.

**Multi-query attention** (MQA) keeps $h$ query heads but only **one** key head and one value head, shared by all of them.[^shazeer2019] The cache shrinks by a factor $h$. **Grouped-query attention** (GQA) interpolates: the $h$ query heads are split into $g$ groups, and each group shares one key head and one value head.[^ainslie2023] With $g=h$ it is multi-head attention, with $g=1$ multi-query. Ainslie et al. found that GQA with a few groups reaches quality close to multi-head attention at a speed close to MQA, and that an existing multi-head checkpoint can be converted by mean-pooling its key and value heads within each group and training briefly ("uptraining", about 5% of the original pre-training compute). Llama 2 70B and Mistral 7B, for example, use 8 KV heads for 64 and 32 query heads.[^touvron2023]<sup>,</sup>[^jiang2023] The decoder-block aside implements GQA with `repeat_interleave`.

DeepSeek's **multi-head latent attention** (MLA) goes further: it caches one low-dimensional latent vector per token per layer, from which all heads' keys and values are reconstructed by learned up-projections.[^deepseek2024] Parag's [modern_llms](https://github.com/paragduttaiisc/modern_llms/blob/main/model/attention.py) uses MLA, with RoPE applied to a separate slice of each head (one rotated key shared by all heads), since rotated keys cannot be rebuilt from a position-free latent.

### FlashAttention: exact attention with less memory traffic

Written as three matrix operations, $\mathbf{S}=\mathbf{Q}\mathbf{K}^\top/\sqrt{d_k}$, $\mathbf{P}=\operatorname{softmax}(\mathbf{S})$, $\mathbf{O}=\mathbf{P}\mathbf{V}$, attention materializes two $n\times n$ matrices. On a GPU these are written to and read back from the large but slow main memory (HBM), and for long sequences this traffic, not the arithmetic, dominates the run time. **FlashAttention** is an **IO-aware** algorithm that computes exactly the same output without ever storing $\mathbf{S}$ or $\mathbf{P}$.[^dao2022]
- **Tiling.** It loads blocks of $\mathbf{Q}$, $\mathbf{K}$ and $\mathbf{V}$ into the small, fast on-chip memory (SRAM), and computes the contribution of one block of keys to one block of queries there.
- **Online softmax.** The softmax normalizer needs all scores of a row, which are not available block by block. The trick is to keep, for each query, a running maximum $m$, a running sum $s=\sum e^{z-m}$ and an unnormalized output $\mathbf{o}=\sum e^{z-m}\mathbf{v}$. When a new block raises the maximum to $m'$, the old $s$ and $\mathbf{o}$ are rescaled by $e^{m-m'}$, and the block's terms are added. Dividing $\mathbf{o}$ by $s$ at the end gives the exact softmax average.
- **Recomputation.** The backward pass recomputes the blocks of $\mathbf{P}$ instead of storing them.

The memory becomes linear in $n$, and the number of HBM accesses drops by a large factor. The paper reports, for example, a 3× faster GPT-2 training at sequence length 1K. It is not an approximation: the output is attention, up to floating-point rounding. PyTorch's `scaled_dot_product_attention`, which modern_llms calls, dispatches to a FlashAttention kernel when the hardware and inputs allow it.

> [!example]- Code: attention one tile at a time with an online softmax (checked)
> ```python
> def tiled_attention(q, K, V, tile=16):
>     """One query q: (d,), keys K: (n, d), values V: (n, dv). Reads K, V one tile at a time
>     and keeps only a running max m, normalizer s and unnormalized output o."""
>     m, s, o = -math.inf, 0.0, torch.zeros(V.shape[1], dtype=V.dtype)
>     for i in range(0, K.shape[0], tile):
>         z = K[i:i + tile] @ q / math.sqrt(q.shape[0])     # scores of this tile
>         m_new = max(m, z.max().item())
>         scale = math.exp(m - m_new)                      # rescale what we had so far
>         p = torch.exp(z - m_new)
>         s = s * scale + p.sum()
>         o = o * scale + p @ V[i:i + tile]
>         m = m_new
>     return o / s
> ```
> For one query against 200 keys, the result equals ordinary softmax attention to $10^{-12}$ for tile sizes 1, 7, 16 and 200. FlashAttention does the same for blocks of queries at once, inside GPU on-chip memory.

### A transformer encoder for text classification

To classify a sentence with an encoder, we need one vector for the whole sequence. Two options are common:

- prepend a learnable **[CLS] token** and classify from its output vector, which can attend to every word (BERT);[^devlin2019]
- **mean-pool** the token outputs, over the real tokens only.

Notebook 4 of Parag's repository replaces the LSTM by three transformer encoder layers on frozen GloVe embeddings, with sinusoidal positional encodings, a padding mask, and masked mean pooling. It reached 92.3% test accuracy on AG News (in an earlier version that had neither positional encodings nor masked pooling), slightly above both LSTMs. For topic classification word order matters less than for, say, sentiment, so the three models are close; the transformer's advantages show on harder tasks and at scale.

## Knowledge lives in the MLPs

Where does a language model keep the fact that the Eiffel Tower is in Paris? Attention is a natural suspect, since it is the part that relates tokens. But attention only moves information that is already in the residual stream, from one position to another. A fact that is not written anywhere in the prompt must come from the weights, and two thirds of the weights are in the MLPs. Several lines of evidence say that this is indeed where much of the knowledge is stored, although the picture is not as clean as the slogan.

### The MLP as a key–value memory

Write the MLP of one block (ignoring biases) with the rows of $\mathbf{W}_{\text{in}}\in\mathbb{R}^{d_{\text{ff}}\times d}$ as $\mathbf{k}_i^\top$ and the columns of $\mathbf{W}_{\text{out}}\in\mathbb{R}^{d\times d_{\text{ff}}}$ as $\mathbf{v}_i$:

$$
\operatorname{MLP}(\mathbf{x})=\mathbf{W}_{\text{out}}\,\phi(\mathbf{W}_{\text{in}}\mathbf{x})=\sum_{i=1}^{d_{\text{ff}}}\phi\big(\mathbf{k}_i^\top\mathbf{x}\big)\,\mathbf{v}_i.
\tag{4}
$$

Compare with attention: there, a query is compared with keys computed from the **context**, and the matching values are averaged. Here, the token's representation is compared with $d_{\text{ff}}$ keys stored in the **weights**, and the values of the keys it matches are added to the residual stream. The MLP is an attention layer over a fixed, learned memory (without the softmax normalization). With $d_{\text{ff}}=4d$, each GPT-2 small block holds 3,072 such memories.

Geva et al. tested this reading on a trained language model.[^geva2021] For each key, they collected the training prefixes that activated it most and found human-interpretable patterns: in lower layers, shallow ones (prefixes ending in the same word); in upper layers, semantic ones (prefixes about the same topic). Projecting a value vector onto the output vocabulary gave a distribution that put high probability on the tokens likely to follow the key's patterns, especially in the upper layers. The output of an MLP is a combination of many memories, refined layer by layer through the residual stream.

How much can such a memory hold? Figure 7 trains a one-hidden-layer MLP, as in (4), to map $N$ random unit vectors in $\mathbb{R}^{32}$ to $N$ random labels out of 128, a toy version of "subject → attribute". Small MLPs store everything up to a point and then fail quickly. The number of associations stored with 95% recall grows in proportion to the number of parameters, at about 0.24 associations (1.7 bits, since each label carries $\log_2128=7$ bits) per parameter. Allen-Zhu and Li measured a strikingly similar constant in language models trained on synthetic biographies: about 2 bits of factual knowledge per parameter.[^allenzhu2024] Wider MLPs are bigger memories.

![[Machine Learning/images/sequence-models-kv-memory.svg]]
*Figure 7: an MLP as a key–value memory. A ReLU MLP $\mathbb{R}^{32}\to\mathbb{R}^{128}$ with hidden width $d_{\text{ff}}$ is trained (Adam, 1,500 full-batch steps, cross-entropy) to map $N$ random unit keys to random labels. (a) The fraction of the $N$ associations it recalls. (b) The largest $N$ recalled at 95%, interpolated between the grid points of (a), against the number of parameters: it grows in proportion (log–log slope 1.09), at about 0.24 associations per parameter. One seed per point.*

### Locating facts: knowledge neurons and causal tracing

If facts are stored in MLPs, it should be possible to find them. Dai et al. attributed BERT's predictions on fill-in-the-blank facts ("Paris is the capital of [MASK]") to individual hidden units of the MLPs with integrated gradients.[^dai2022] A few **knowledge neurons** per fact stood out, mostly in the upper layers; suppressing them lowered the probability of the correct answer, and amplifying them raised it.

Meng et al. used a causal intervention on GPT models instead.[^meng2022] **Causal tracing** runs the model three times on a prompt such as "The Space Needle is located in the city of":
1. a clean run, which predicts "Seattle";
2. a corrupted run, with noise added to the embeddings of the subject tokens, which no longer does;
3. corrupted runs in which a single hidden state is restored to its clean value.

A hidden state whose restoration brings "Seattle" back must carry the information. The decisive states were the **MLP outputs in the middle layers at the last token of the subject**, plus, unsurprisingly, the states at the last position near the output. The proposed story: middle-layer MLPs, reading the subject's last token, retrieve facts about the subject and write them into the residual stream; attention later moves them to the position where the answer is predicted.

### Editing a fact: ROME

If an MLP is a memory, we should be able to write to it. Treat the output matrix $\mathbf{W}=\mathbf{W}_{\text{out}}$ of one middle-layer MLP as a **linear associative memory**: it maps keys $\mathbf{k}$ (the hidden activations $\phi(\mathbf{W}_{\text{in}}\mathbf{x})$) to values $\mathbf{v}=\mathbf{W}\mathbf{k}$. Stack many keys the model normally sees as the columns of $\mathbf{K}$, and their values as $\mathbf{V}=\mathbf{W}\mathbf{K}$. To insert a new fact we want a new matrix $\hat{\mathbf{W}}$ that maps the subject's key $\mathbf{k}_*$ to a chosen value $\mathbf{v}_*$ (one that makes the model output, say, "Rome" instead of "Seattle") while changing everything else as little as possible:

$$
\min_{\hat{\mathbf{W}}}\ \big\lVert\hat{\mathbf{W}}\mathbf{K}-\mathbf{V}\big\rVert_F^2\quad\text{subject to}\quad\hat{\mathbf{W}}\mathbf{k}_*=\mathbf{v}_*.
$$

This is least squares with one linear equality constraint, and it has a closed form. With a vector $\boldsymbol\lambda$ of Lagrange multipliers for the $d$ constraint equations, setting the gradient of the Lagrangian with respect to $\hat{\mathbf{W}}$ to zero gives

$$
(\hat{\mathbf{W}}\mathbf{K}-\mathbf{V})\mathbf{K}^\top=\boldsymbol\lambda\,\mathbf{k}_*^\top.
$$

Since $\mathbf{V}=\mathbf{W}\mathbf{K}$, this says $(\hat{\mathbf{W}}-\mathbf{W})\mathbf{C}=\boldsymbol\lambda\mathbf{k}_*^\top$ with $\mathbf{C}=\mathbf{K}\mathbf{K}^\top$, so $\hat{\mathbf{W}}=\mathbf{W}+\boldsymbol\lambda(\mathbf{C}^{-1}\mathbf{k}_*)^\top$. The constraint fixes $\boldsymbol\lambda$:

$$
\hat{\mathbf{W}}=\mathbf{W}+\frac{\big(\mathbf{v}_*-\mathbf{W}\mathbf{k}_*\big)\big(\mathbf{C}^{-1}\mathbf{k}_*\big)^\top}{\mathbf{k}_*^\top\mathbf{C}^{-1}\mathbf{k}_*}.
\tag{5}
$$

The change is a **rank-one** matrix: **R**ank-**O**ne **M**odel **E**diting.[^meng2022] The numerator is the error the old memory makes on the new key; the direction $\mathbf{C}^{-1}\mathbf{k}_*$ is the key "whitened" by the statistics of normal keys, so the edit perturbs as little as possible the outputs for keys the model usually sees. In practice $\mathbf{C}$ is estimated from activations on a large text sample, $\mathbf{k}_*$ is averaged over several prompts that end in the subject, and $\mathbf{v}_*$ is found by a few steps of gradient descent on the probability of the new answer. ROME's edits generalized to paraphrases of the prompt ("The Space Needle is in…") while leaving unrelated facts mostly intact. MEMIT extends the same update to thousands of edits spread over several layers.[^meng2023]

### Attention routes, MLPs store, with caveats

A simple picture emerges, consistent with the parameter counts and the residual stream view: **attention routes** information between positions, and **MLPs store** and transform it. Geva et al. later traced the whole recall of a fact: early MLPs enrich the representation of the subject's last token with many of its attributes; attention carries information about the relation to the final position; and attention then extracts the right attribute from the enriched subject.[^geva2023]

The slogan needs qualifications.
- **Localization is not the same as storage.** Hase et al. found that where causal tracing localizes a fact does not predict which layer is best to edit; edits work at many layers.[^hase2023] Knowledge is spread out, and causal tracing shows where it *flows*, not a single place where it *lives*.
- **Edits do not propagate.** After teaching a model a new fact, the facts that follow logically from it often stay unchanged (the "ripple effects" of an edit).[^cohen2024]
- **Attention stores too.** The heads that extract an attribute often encode subject–attribute mappings in their own weights.[^geva2023]
- **Features are not neurons.** Many more features than dimensions can be stored as nearly orthogonal directions, in **superposition**, so a single unit often responds to several unrelated things. [[Language Representations and Language Models]] explains why nearly orthogonal directions are so plentiful in high dimensions.[^elhage2022]

## Mixture of experts

If the MLPs are the memory, more knowledge needs bigger MLPs, and bigger MLPs cost compute on every token. But a given token needs only a small part of the memory: a sentence about chemistry does not need the facts about football. **Mixture-of-experts** (MoE) layers exploit this.[^shazeer2017] They replace the single MLP of a block by $E$ smaller MLPs, the **experts**, and a **router** that sends each token to only $k$ of them (Figure 8a). The number of parameters grows with $E$; the compute per token grows only with $k$. MoE is **conditional computation**: the parameters are large, but each token uses a small, input-dependent slice of them.

### Sparse MoE layers

The router is a linear layer followed by a softmax, $\mathbf{p}(\mathbf{x})=\operatorname{softmax}(\mathbf{W}_r\mathbf{x})\in\Delta^{E}$. Keep the $k$ largest probabilities, the set $\mathcal{T}(\mathbf{x})$, and combine those experts' outputs, weighted by their renormalized probabilities:

$$
\operatorname{MoE}(\mathbf{x})=\sum_{e\in\mathcal{T}(\mathbf{x})}\frac{p_e(\mathbf{x})}{\sum_{e'\in\mathcal{T}(\mathbf{x})}p_{e'}(\mathbf{x})}\,\operatorname{MLP}_e(\mathbf{x}).
$$

The top-$k$ selection is not differentiable, but the weights are: the router learns through the gradient of the output with respect to $p_e$ for the chosen experts. Only the chosen experts run, so in practice the tokens of a batch are grouped by expert and each expert processes its group as one matrix product.

### Load balancing and capacity

Left alone, routing collapses. Early in training a few experts happen to be slightly better; the router sends them more tokens; they improve faster; and the rest are starved. The model then wastes most of its parameters, and on parallel hardware the busy experts become a bottleneck. Every MoE therefore adds an **auxiliary load-balancing loss**. The Switch Transformer's version, for a batch of $T$ tokens, is[^fedus2022]

$$
\mathcal{L}_{\text{aux}}=\alpha\,E\sum_{e=1}^{E}f_e\,P_e,\qquad
f_e=\frac{\#\{\text{tokens sent to }e\}}{T},\qquad
P_e=\frac1T\sum_{\text{tokens}}p_e(\mathbf{x}).
\tag{6}
$$

Here $f_e$ is the share of tokens expert $e$ actually receives (not differentiable) and $P_e$ the average probability the router gives it (differentiable). The loss is minimized when routing is uniform, where it equals $\alpha$; its gradient pushes down the router probability of experts that receive more than their share. A small weight such as $\alpha=0.01$ is typical.

Figure 8b shows the collapse and its cure on a toy task: 8 linear experts, top-1 routing, and inputs from 8 clusters, each with its own linear target map. Without the balancing loss, 1 to 3 of the 8 experts end up receiving no tokens at all in every one of 5 runs, the busiest expert takes 25–37% of the tokens, and the test error is 0.119 (the targets have variance 4.0). With $\alpha=0.1$, every expert is used, the busiest takes at most 22%, and the error falls to 0.053 on average. One run found the ideal one-expert-per-cluster assignment (error 0.004); the others still share some clusters between experts.

**Capacity.** For efficient batched computation, each expert accepts at most

$$
\text{capacity}=\text{capacity factor}\times\frac{\text{tokens in the batch}}{E}
$$

tokens. Tokens beyond an expert's capacity are **dropped**: they skip the layer and continue on the residual stream unchanged. A capacity factor slightly above 1 (for example 1.25) leaves room for imbalance at some memory cost.

![[Machine Learning/images/sequence-models-moe.svg]]
*Figure 8: (a) top-2 routing: the router's softmax gives every expert a probability, only the two largest run, and their outputs are combined with the renormalized weights. (b) A top-1 mixture of 8 linear experts trained on a toy regression task with 8 input clusters (5 seeds, 2,000 Adam steps; bars show the mean and standard deviation of each sorted expert's share of 20,000 test tokens). Without the load-balancing loss (6), routing collapses onto a few experts and 1–3 experts are never used; with it ($\alpha=0.1$), all experts receive tokens and the test error halves.*

> [!example]- Code: a top-$k$ mixture-of-experts layer with the load-balancing loss (checked)
> ```python
> class MoE(torch.nn.Module):
>     def __init__(self, d, d_ff, n_experts, k):
>         super().__init__()
>         self.k, self.E = k, n_experts
>         self.router = torch.nn.Linear(d, n_experts, bias=False)
>         self.experts = torch.nn.ModuleList(
>             torch.nn.Sequential(torch.nn.Linear(d, d_ff), torch.nn.GELU(), torch.nn.Linear(d_ff, d))
>             for _ in range(n_experts))
>
>     def forward(self, x):                                    # x: (n, d) tokens
>         probs = torch.softmax(self.router(x), -1)            # (n, E)
>         w, idx = probs.topk(self.k, -1)                      # top-k experts per token
>         w = w / w.sum(-1, keepdim=True)                      # renormalize their weights
>         y = torch.zeros_like(x)
>         for e in range(self.E):                              # each expert sees only its tokens
>             tok, slot = (idx == e).nonzero(as_tuple=True)
>             if tok.numel():
>                 y[tok] += w[tok, slot, None] * self.experts[e](x[tok])
>         f = torch.bincount(idx.flatten(), minlength=self.E) / idx.numel()  # load fractions
>         P = probs.mean(0)                                    # mean router probabilities
>         aux = self.E * (f * P).sum()                         # load-balancing loss, 1 if balanced
>         return y, aux
> ```
> The sparse dispatch equals a dense reference that runs every expert and masks all but the top $k$, the router receives a gradient, and the auxiliary loss equals 1 for perfectly uniform routing (its value on random inputs at initialization was 1.004).

### Switch, GShard and Mixtral

- **Shazeer et al. (2017)** placed MoE layers with up to thousands of experts between LSTM layers, with a noisy top-$k$ router and balancing losses, and trained models with up to 137B parameters.[^shazeer2017]
- **GShard** brought top-2 MoE into the transformer, replacing every other MLP, with expert capacity and an auxiliary loss, and scaled a translation model beyond 600B parameters across 2,048 accelerators.[^lepikhin2021]
- **Switch Transformer** simplified routing to **top-1**: each token goes to a single expert, which halves the communication and the expert computation of top-2. It introduced the loss (6) and reported up to 7× faster pre-training than a dense T5 model with the same compute per token.[^fedus2022]
- **Mixtral 8x7B** is Mistral 7B with every MLP replaced by 8 SwiGLU experts and top-2 routing.[^jiang2024] It has **46.7B parameters in total but 12.9B active** per token (Figure 6a), and it matched or beat the dense Llama 2 70B on the reported benchmarks.

**Total versus active parameters.** An MoE's quality tracks its total parameters more than its active ones, while its speed tracks the active ones, which is the point of the design. Its memory footprint tracks the total: all experts must sit in memory even though each token uses two of them.

**Why MoE fits the "knowledge lives in the MLPs" picture.** MoE layers replace the MLPs, never the attention. If the MLPs are key–value memories, an MoE is a memory with a coarse first lookup: the router picks which bank of memories to search, and the chosen experts search it. Adding experts adds memory at almost no compute, which is exactly what a store of facts needs. Allen-Zhu and Li found MoE models to be about as efficient at storing knowledge per (total) parameter as dense ones.[^allenzhu2024] Parag's [modern_llms](https://github.com/paragduttaiisc/modern_llms/blob/main/model/feed_forward.py) implements a top-$k$ softmax router with this load-balancing loss and runs all experts of a layer as one grouped matrix product; its example 24-layer model has 2.72B parameters in total and 0.85B active.

## Beyond quadratic attention

Attention compares every token with every other one: $O(n^2d)$ time for a sequence, and a KV cache that grows linearly with the context during generation. A recurrent network does the opposite: a fixed-size state, $O(n)$ time and $O(1)$ memory per generated token, but a strictly sequential computation and a memory that must decide what to keep. The models of this section try to get the best of both: training that is parallel over the sequence like attention, and inference that is recurrent like an RNN.

### Linear attention is an RNN

Write causal attention for the $i$-th token with a general similarity function, $\operatorname{sim}(\mathbf{q},\mathbf{k})=\exp(\mathbf{q}^\top\mathbf{k}/\sqrt{d_k})$ for the usual softmax:

$$
\mathbf{y}_i=\frac{\sum_{j\le i}\operatorname{sim}(\mathbf{q}_i,\mathbf{k}_j)\,\mathbf{v}_j}{\sum_{j\le i}\operatorname{sim}(\mathbf{q}_i,\mathbf{k}_j)}.
$$

Katharopoulos et al. replaced the exponential by a dot product of feature maps, $\operatorname{sim}(\mathbf{q},\mathbf{k})=\phi(\mathbf{q})^\top\phi(\mathbf{k})$, with for example $\phi(\mathbf{z})=\operatorname{elu}(\mathbf{z})+1>0$.[^katharopoulos2020] Then $\phi(\mathbf{q}_i)$ can be pulled out of both sums:

$$
\mathbf{y}_i^\top=\frac{\phi(\mathbf{q}_i)^\top\mathbf{S}_i}{\phi(\mathbf{q}_i)^\top\mathbf{z}_i},\qquad
\mathbf{S}_i=\mathbf{S}_{i-1}+\phi(\mathbf{k}_i)\,\mathbf{v}_i^\top,\qquad
\mathbf{z}_i=\mathbf{z}_{i-1}+\phi(\mathbf{k}_i).
\tag{7}
$$

This is an **RNN** whose state is the $d_k\times d_v$ matrix $\mathbf{S}_i$ (plus the normalizer $\mathbf{z}_i$): generation costs the same at every step, whatever the context length. During training, the same outputs can be computed in parallel as a masked matrix product, $(\mathbf{L}\odot\phi(\mathbf{Q})\phi(\mathbf{K})^\top)\mathbf{V}$ with the lower-triangular all-ones mask $\mathbf{L}$, followed by the normalization. Hence the paper's title, "Transformers are RNNs".

The catch is in the state. Softmax attention keeps every key and value and can retrieve any one of them exactly. A linear-attention state is a sum of outer products in a fixed space: it **compresses** the past, and new associations interfere with old ones, much as in the classical Hopfield network of [[Energy-Based Models]]. Plain linear attention was noticeably weaker than softmax attention on language. Its successors add **decay** or **gating**, so that the state can forget: RetNet multiplies $\mathbf{S}_{i-1}$ by a fixed factor $\gamma<1$,[^sun2023] and RWKV uses learned, channel-wise decays.[^peng2023] Making the decay depend on the input is exactly the step from S4 to Mamba below.

### State space models

A **state space model** (SSM) comes from control theory. A continuous input signal $u(t)$ drives a hidden state $\mathbf{x}(t)\in\mathbb{R}^{d_s}$ through a linear differential equation, and the output reads the state:

$$
\dot{\mathbf{x}}(t)=\mathbf{A}\mathbf{x}(t)+\mathbf{B}u(t),\qquad y(t)=\mathbf{C}\mathbf{x}(t)+Du(t).
$$

The [[Markov Chains#Towards diffusion models and state space models|Markov Chains]] lecture met this as the deterministic skeleton of a Kalman filter. To apply it to a sequence of tokens, take one channel of the token vectors as the samples $u_1,u_2,\dots$ of $u(t)$ at spacing $\Delta$. If $u$ is held constant over each step (**zero-order hold**), the solution of the linear ODE over one step involves the matrix exponential, and the **discretized** system is a linear recurrence:

$$
\mathbf{x}_t=\bar{\mathbf{A}}\,\mathbf{x}_{t-1}+\bar{\mathbf{B}}\,u_t,\qquad y_t=\mathbf{C}\mathbf{x}_t+Du_t,
$$

with $\bar{\mathbf{A}}=e^{\Delta\mathbf{A}}$ and $\bar{\mathbf{B}}=(\Delta\mathbf{A})^{-1}(e^{\Delta\mathbf{A}}-\mathbf{I})\,\Delta\mathbf{B}$. The step $\Delta$ is learned: it sets how fast the state evolves relative to the tokens. A deep SSM runs one such system per channel (all $d$ channels in parallel) and mixes channels with linear layers and MLPs between SSM layers.

**Two faces of one model.** Unrolling the recurrence from $\mathbf{x}_0=\mathbf{0}$ gives $y_t=\sum_{s\le t}\mathbf{C}\bar{\mathbf{A}}^{\,t-s}\bar{\mathbf{B}}\,u_s+Du_t$. Since $\bar{\mathbf{A}}$, $\bar{\mathbf{B}}$ and $\mathbf{C}$ are the same at every step, this is a **convolution** of the input with the kernel

$$
\bar{\mathbf{K}}=\big(\mathbf{C}\bar{\mathbf{B}},\ \mathbf{C}\bar{\mathbf{A}}\bar{\mathbf{B}},\ \mathbf{C}\bar{\mathbf{A}}^2\bar{\mathbf{B}},\ \dots,\ \mathbf{C}\bar{\mathbf{A}}^{n-1}\bar{\mathbf{B}}\big).
$$

So an SSM can be **trained as a convolution**, in parallel over the sequence (with the FFT, in $O(n\log n)$), and **run as an RNN** at inference, with a fixed-size state. Version 1 of [Mamba-in-JAX](https://github.com/parag1604/Mamba-in-JAX) checks that the two agree to $5\times10^{-7}$.

**S4 and S4D.** A random $\mathbf{A}$ does not work: powers $\bar{\mathbf{A}}^{\,t}$ either explode or forget within a few steps. The **S4** model initializes $\mathbf{A}$ with the HiPPO matrix, derived so that the state is an optimal running summary of the input's history (its projection onto Legendre polynomials),[^gu2020] and uses a diagonal-plus-low-rank structure to compute the kernel stably and fast.[^gu2022] S4 was the first model to solve Path-X, the hardest task of the Long Range Arena benchmark, with sequences of 16,384 steps. Shortly after, Gupta et al. and then Gu et al. (S4D) showed that a **diagonal** $\mathbf{A}$ with a good initialization works about as well and is far simpler: each of the $d_s$ modes then evolves on its own, and the kernel is a sum of $d_s$ geometric sequences.[^gupta2022]<sup>,</sup>[^gu2022s4d]

### Selectivity: Mamba

S4 is **linear time-invariant** (LTI): the same kernel is applied whatever the input. It can learn "copy the token from 64 steps ago", a fixed shift, but not "remember the next noun and ignore the filler words", because what to remember depends on *content*. In the language of this lecture, its sequence mixing cannot depend on the tokens it mixes, while attention's weights are computed from them.

**Mamba** makes the SSM **selective**: the step $\Delta$ and the matrices $\mathbf{B}$ and $\mathbf{C}$ become functions of the current token,[^gu2023]

$$
\Delta_t=\operatorname{softplus}(\mathbf{w}_\Delta^\top\mathbf{u}_t+b_\Delta),\qquad
\mathbf{B}_t=\mathbf{W}_B\mathbf{u}_t,\qquad
\mathbf{C}_t=\mathbf{W}_C\mathbf{u}_t,
$$

and the recurrence becomes $\mathbf{x}_t=e^{\Delta_t\mathbf{A}}\mathbf{x}_{t-1}+\Delta_t\mathbf{B}_t u_t$, with a fixed diagonal $\mathbf{A}$ with negative entries. The step size is a gate:
- a **large** $\Delta_t$ makes $e^{\Delta_t\mathbf{A}}\approx\mathbf{0}$: the state is reset and overwritten by the current token ("remember this");
- a **small** $\Delta_t$ makes $e^{\Delta_t\mathbf{A}}\approx\mathbf{I}$ and $\Delta_t\mathbf{B}_t u_t\approx\mathbf{0}$: the token is ignored and the state carries on ("skip this").

This is the same idea as the LSTM's forget and input gates, now inside a linear recurrence. The price is that the system is no longer time-invariant, so there is no convolution kernel. Mamba trains with a **parallel scan** instead. One step is the affine map $\mathbf{x}\mapsto\mathbf{a}_t\odot\mathbf{x}+\mathbf{b}_t$ (diagonal $\bar{\mathbf{A}}_t$), and composing two steps gives another affine map, $(\mathbf{a}_1,\mathbf{b}_1)$ then $(\mathbf{a}_2,\mathbf{b}_2)$ equals $(\mathbf{a}_2\odot\mathbf{a}_1,\ \mathbf{a}_2\odot\mathbf{b}_1+\mathbf{b}_2)$. This composition is associative, so all $n$ prefix states can be computed in $O(\log n)$ parallel rounds, like a parallel prefix sum. Mamba's implementation keeps the large expanded state in on-chip memory, in the spirit of FlashAttention.

**The Mamba block** merges the two halves of Figure 1. The input is projected to twice the width, passed through a short causal convolution and a SiLU, then through the selective SSM, multiplied by a SiLU gate computed from a second projection, and projected back. There is no separate MLP: the gated projections around the SSM, which hold almost all of the block's parameters, do the channel mixing. Two Mamba blocks have about as many parameters as one transformer block.

**What selectivity buys.** The standard test is **selective copying**: a few data tokens are scattered at random positions among blanks, and the model must reproduce them in order at the end. Gu and Dao also used **induction heads** (recall the token that followed an earlier occurrence of the current one), the mechanism behind much of in-context learning.[^olsson2022] Parag's Mamba-in-JAX repository trains small models of each kind on CPU (test accuracy on the answer positions; chance is 12.5%):

| model | copying, $n=72$ | selective copying, $n=72$ | selective copying, $n=264$ |
|---|:-:|:-:|:-:|
| S4D (time-invariant, convolution) | 99.2% | 70.3% | 26.3% |
| Mamba (selective) | 100% | 100% | 48–51% |

The $n=72$ models have 2 layers and width 64; the $n=264$ ones have 1 layer and width 32, with a training budget that solved neither (the two Mamba numbers are two initializations). Both models copy at a fixed offset; only the selective one can decide from content what to keep. The repository also checks that Mamba's recurrent inference reproduces its parallel training outputs to $10^{-5}$ and that its state stays at 19 KiB whatever the context length.

**Not a free lunch.** A fixed-size state cannot hold an arbitrarily long context. Jelassi et al. proved that a two-layer transformer can copy strings of length exponential in its size, while any model with a fixed-size state is limited by that state, and found pretrained transformers far better than SSMs of similar size at copying and retrieving from long contexts.[^jelassi2024]

> [!example]- Code: linear attention and the selective scan, recurrent and parallel (checked)
> ```python
> def linear_attention_recurrent(Q, K, V, phi=lambda z: F.elu(z) + 1):
>     """Causal linear attention as an RNN with a d x dv matrix state S and a d-vector z."""
>     S = torch.zeros(Q.shape[1], V.shape[1], dtype=Q.dtype); z = torch.zeros(Q.shape[1], dtype=Q.dtype)
>     out = []
>     for t in range(Q.shape[0]):
>         S = S + torch.outer(phi(K[t]), V[t]); z = z + phi(K[t])
>         out.append(phi(Q[t]) @ S / (phi(Q[t]) @ z))
>     return torch.stack(out)
>
> def selective_scan(x, delta, A, B, C):
>     """Mamba-style selective SSM for one channel group.
>     x: (L, D) inputs; delta: (L, D) step sizes; A: (D, N) (negative);
>     B, C: (L, N) input-dependent. State h: (D, N). Returns y: (L, D)."""
>     h = torch.zeros(A.shape, dtype=x.dtype); ys = []
>     for t in range(x.shape[0]):
>         Abar = torch.exp(delta[t, :, None] * A)              # ZOH for A
>         h = Abar * h + delta[t, :, None] * B[t] * x[t, :, None]   # Euler for B
>         ys.append(h @ C[t])
>     return torch.stack(ys)
> ```
> Checks: the recurrent linear attention (7) equals its parallel masked form; the selective scan equals both the explicit matrix form of the next section and a parallel (Hillis–Steele) associative scan, all to $10^{-12}$.

### Mamba-2: SSMs and attention are two views of one matrix

Unroll a selective SSM whose $\bar{\mathbf{A}}_t=a_t\mathbf{I}$ is a scalar times the identity, as in Mamba-2. The output is a linear function of all inputs so far:

$$
y_i=\sum_{j\le i}\Big(\mathbf{C}_i^\top\mathbf{B}_j\Big)\Big(\prod_{k=j+1}^{i}a_k\Big)\,u_j,
\qquad\text{that is}\qquad
\mathbf{y}=\big(\mathbf{L}\odot\mathbf{C}\mathbf{B}^\top\big)\,\mathbf{u},
$$

with $L_{ij}=\prod_{k=j+1}^{i}a_k$ for $j\le i$ and 0 otherwise. Compare with causal linear attention (7) without its normalizer: $(\mathbf{L}\odot\mathbf{Q}\mathbf{K}^\top)\mathbf{V}$ with an all-ones mask. The selective SSM **is** linear attention, with $\mathbf{C}$ in the role of the queries, $\mathbf{B}$ of the keys and the input of the values, and with a data-dependent **decay mask** $\mathbf{L}$ in place of the plain causal mask. Dao and Gu call this **state space duality**: the same layer can be computed as a recurrence (linear in $n$) or as a masked attention-like matrix product (quadratic, but made of matrix multiplications that GPUs execute very fast).[^dao2024] Mamba-2 computes it blockwise, attention-style inside chunks and recurrently between them, which made its core layer 2–8× faster than Mamba's and allowed a much larger state.

### Hybrids and other recurrent models

Since fixed-state models are cheap but forgetful and attention is precise but expensive, the natural compromise is to combine them. **Jamba** interleaves Mamba layers with a few attention layers (one attention layer per eight layers in the released model) and adds mixture-of-experts MLPs; the released model fits on a single 80 GB GPU and reports strong results with contexts up to 256K tokens.[^lieber2024] Several recent large models follow the same recipe of mostly linear-time layers plus a few attention layers; the roadmap of modern_llms plans exactly this. RetNet and RWKV, mentioned above, are recurrent models with decays, trained in parallel and run as RNNs.

![[Machine Learning/images/sequence-models-cost.svg]]
*Figure 9: (a) forward FLOPs of one layer over a whole sequence, $d=4096$: causal attention mixing ($2n^2d$) against a selective SSM scan (about $9nEd\,d_s$ with expansion $E=2$ and state size $d_s=16$, a rough count) and the MLP ($16nd^2$) for scale. Attention's mixing cost overtakes the MLP at $n=8d$; the SSM's never does. (b) Memory per sequence during decoding, in 16-bit precision, for Llama 2 7B's shape (32 layers): the KV cache grows by 512 KiB per token with 32 KV heads, 128 KiB with 8 (GQA) and 16 KiB with 1 (MQA), while a Mamba-style model with 64 layers at the same width keeps a constant 19 MiB state.*

## Long context and efficient inference

Three more ideas recur in every long-context or fast-inference system.

**Sliding-window attention.** Let each token attend only to the previous $W$ tokens. The cost becomes $O(nW)$ and the cache holds at most $W$ entries (a rolling buffer). Information still travels further than $W$: after $L$ layers, a token can be influenced by tokens up to about $LW$ positions back, just as stacked convolutions grow their receptive field. Longformer combined local windows with a few global tokens;[^beltagy2020] Mistral 7B uses a window of 4,096 tokens in every layer.[^jiang2023]

**Compressing the KV cache.** Besides sharing heads (MQA, GQA) and caching a latent (MLA), the cache can be **quantized** to fewer bits, or **evicted**: StreamingLLM observed that models put large attention weights on the first few tokens ("attention sinks"), and that keeping those few tokens plus a recent window lets a model run on streams far longer than its training context.[^xiao2024]

**Speculative decoding.** Decoding is memory-bound, so checking several tokens in one forward pass costs about as much as generating one. A small **draft** model $q$ proposes $\gamma$ tokens cheaply; the large **target** model $p$ scores all of them in a single parallel pass; and each draft token $x$ is accepted with probability $\min\{1,p(x)/q(x)\}$. At the first rejection, a replacement is sampled from the normalized residual $\max\{0,p-q\}$, and the remaining drafts are discarded.[^leviathan2023]<sup>,</sup>[^chen2023] This produces **exactly** the target distribution: a token $x$ comes out either by being proposed and accepted, with probability $q(x)\min\{1,p(x)/q(x)\}=\min\{q(x),p(x)\}$, or through a rejection followed by resampling, with probability $\max\{0,p(x)-q(x)\}$; and the two add up to $p(x)$. The better the draft matches the target, the more tokens are accepted per pass; Leviathan et al. reported 2–3× faster decoding with identical outputs. Parag's [modern_llms](https://github.com/paragduttaiisc/modern_llms/blob/main/utils/spec_decode.py) trains extra heads that predict tokens $t+2,\dots,t+n$ (multi-token prediction)[^gloeckle2024] and uses them as a free draft model for **self-speculative** decoding.

> [!example]- Code: one speculative sampling step (checked)
> ```python
> def speculative_step(p_target, q_draft, gen):
>     """One draft token. p_target, q_draft: probability vectors over the vocabulary.
>     Returns a token whose distribution is exactly p_target."""
>     x = torch.multinomial(q_draft, 1, generator=gen).item()          # draft proposes
>     if torch.rand(1, generator=gen).item() < min(1.0, (p_target[x] / q_draft[x]).item()):
>         return x                                                     # accept
>     resid = (p_target - q_draft).clamp(min=0)                        # else resample from
>     return torch.multinomial(resid / resid.sum(), 1, generator=gen).item()  # the leftover mass
> ```
> With a target $p=(0.5,0.2,0.2,0.1)$ and a poor draft $q=(0.1,0.6,0.2,0.1)$, 200,000 steps give empirical frequencies $(0.501,0.198,0.201,0.099)$: the target distribution, although only $\sum_x\min\{p(x),q(x)\}=60\%$ of the draft tokens are accepted.

## Image captioning

Captioning combines the two kinds of network in these notes: an image encoder ([[Convolutional Neural Networks|a CNN]] or a Vision Transformer) and a sequence decoder. An image goes in, a sentence comes out.

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
*Figure 10: attention weights of the dot-product attention LSTM (decoder 2) over the ViT patch grid, one panel per generated word, on a Flickr8k validation image. Some words attend where one would expect (the final "dog" on the white dog's body); others spread their attention over the background. Attention maps are a useful but loose explanation of what the model uses. Result from Parag's Image-Captioning repository.*

### Decoding

At test time we want a likely caption, but searching over all sentences is impossible.

- **Greedy decoding** takes the most probable word at each step. It is fast but can paint itself into a corner: a good first word may lead to poor continuations. Parag's code decodes greedily.
- **Beam search** keeps the $B$ most probable partial captions (by summed log-probability) at each step, extends each by every word, and again keeps the best $B$. A small beam ($B$ of 3 to 5) usually gives better captions than greedy decoding. Since every extra word adds a negative log-probability, scores are often length-normalized so that beam search does not favour short captions.

> [!example]- Code: beam search (greedy decoding is $B=1$)
> ```python
> def beam_search(step_logprobs, bos, eos, B=3, max_len=20, alpha=0.0):
>     """step_logprobs(prefix) -> tensor (V,) of log p(w | prefix, I). Returns the best token list."""
>     beams, done = [([bos], 0.0)], []                    # (tokens, summed log-probability)
>     for _ in range(max_len):
>         cand = []
>         for toks, score in beams:
>             lp = step_logprobs(toks)
>             top = lp.topk(min(B, lp.numel()))           # only the B best extensions can survive
>             cand += [(toks + [w], score + l) for l, w in zip(top.values.tolist(), top.indices.tolist())]
>         cand.sort(key=lambda c: c[1], reverse=True)
>         beams = []
>         for toks, score in cand[:B]:                    # keep the B best partial captions
>             (done if toks[-1] == eos else beams).append((toks, score))
>         if not beams:
>             break
>     done += beams                                       # unfinished beams at max_len
>     return max(done, key=lambda c: c[1] / (len(c[0]) - 1) ** alpha)[0]   # alpha > 0: length-normalized
> ```
> In this simple version a finished caption leaves the beam, so the beam shrinks; production implementations keep $B$ live hypotheses and add an end-of-sequence check per step.

### Evaluation: BLEU

Captions are scored against several human references with **BLEU**, a machine-translation metric.[^papineni2002] Let $p_n$ be the **modified $n$-gram precision**: the fraction of the candidate's $n$-grams that also appear in a reference, with each $n$-gram counted at most as often as it appears in any single reference (so "the the the the" cannot score well). With candidate length $c$ and reference length $r$,

$$
\operatorname{BLEU}=\operatorname{BP}\cdot\exp\!\Big(\sum_{n=1}^{4}\tfrac14\log p_n\Big),\qquad
\operatorname{BP}=\min\big(1,\,e^{\,1-r/c}\big).
$$

The brevity penalty BP punishes captions that are too short to be judged by precision alone. BLEU is cheap and correlates with human judgment over a whole test set, but it rewards word overlap, not meaning: a correct caption phrased differently from all references scores poorly.

![[Machine Learning/images/applications-of-neural-networks-captioning-bleu.png]]
*Figure 11: validation BLEU of the four decoders on Flickr8k during training (5-epoch moving average). The Bahdanau-attention LSTM and the transformer decoder score highest; the plain LSTM is lowest for most of training. Result from Parag's Image-Captioning repository.*

## The Vision Transformer

If transformers work so well on sequences of words, can they work on images directly, without convolutions? The **Vision Transformer** (ViT) says yes, with one simple trick: **cut the image into patches and treat each patch as a token**.[^dosovitskiy2021]

### Patches as tokens

An image of size $C\times H\times W$ is split into $N=HW/P^2$ non-overlapping patches of $P\times P$ pixels (Figure 12). Each patch is flattened into a vector of length $P^2C$ and mapped to the model width $d$ by a learned matrix $\mathbf{E}\in\mathbb{R}^{P^2C\times d}$. A learnable [CLS] token is prepended, learned position embeddings are added, and the sequence goes through a standard transformer encoder:

$$
\begin{aligned}
\mathbf{z}_0&=\big[\mathbf{x}_{\text{cls}};\ \mathbf{x}_1\mathbf{E};\ \dots;\ \mathbf{x}_N\mathbf{E}\big]+\mathbf{E}_{\text{pos}},\\
\mathbf{z}_\ell&=\operatorname{Block}_\ell(\mathbf{z}_{\ell-1}),\qquad\ell=1,\dots,L,\\
\hat{\mathbf{y}}&=\operatorname{softmax}\big(\operatorname{head}(\operatorname{LN}(\mathbf{z}_L^{\text{cls}}))\big).
\end{aligned}
$$

For ViT-B/16 on a $224\times224$ image, $P=16$ gives $14\times14=196$ patch tokens. The patch embedding is the same as a convolution with kernel size and stride both equal to $P$, and that is how it is usually implemented.

![[Machine Learning/images/applications-of-neural-networks-vit-patches.svg]]
*Figure 12: (a) an image is cut into a grid of patches (here $4\times4$). (b) The patches, read row by row, become a sequence of tokens; a [CLS] token is prepended, each token is linearly embedded and gets a learned position embedding, a transformer encoder processes the sequence, and a head on the [CLS] output gives the class.*

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

A ViT has much less built-in knowledge than a [[Convolutional Neural Networks|CNN]]. Only the patch embedding and the per-token MLPs are local; the 2-D layout is not given at all, and the position embeddings have to *learn* which patches are neighbours. The original paper found that, trained on ImageNet alone, ViTs fall a few points short of ResNets of similar size, but after pre-training on much larger datasets (14 to 300 million images) they match or beat them.[^dosovitskiy2021] With little data, the CNN's assumptions help; with enough data, the ViT's flexibility wins.

**Training ViTs on small datasets.** DeiT showed that a ViT can be trained competitively on ImageNet alone with a strong recipe: heavy augmentation (RandAugment, Mixup, CutMix), regularization such as label smoothing and stochastic depth, AdamW with warm-up and cosine decay, and optionally distillation from a CNN teacher through an extra token.[^touvron2021]<sup>,</sup>[^cubuk2020]<sup>,</sup>[^szegedy2016]

Parag's [improved-vision-encoders](https://github.com/paragduttaiisc/improved-vision-encoders) repository takes a different route to the same problem: instead of a heavy supervised recipe, the ViT encoder is pretrained **without labels**, by masking 75% of the patches and predicting the latent features of the masked patches from the visible ones (a joint-embedding predictive architecture). The quality of the learned features is then measured with a linear probe on ImageNet-1K. The repository also makes the encoder's parts swappable, such as attention versus an FFT-based token mixer, GELU versus SwiGLU, and dense versus mixture-of-experts feed-forward layers, so their effect on representation quality can be compared.

**Cost.** The number of tokens is $N=HW/P^2$, and attention costs $O(N^2)$. Halving the patch size quadruples $N$ and multiplies the attention cost by 16, which is why ViTs use fairly large patches and why hierarchical variants that attend within local windows exist for high-resolution tasks such as detection and segmentation.

## Key takeaways

- **Sequence models alternate sequence mixing and channel mixing.** Recurrence, attention, convolution and state space models differ in how tokens exchange information; all of them leave the per-token computation to MLPs.
- Text is tokenized and **embedded**. **LSTMs** avoid vanishing gradients through an additive, gated cell state. Padding must be masked out of pooling and attention.
- **Attention** is a differentiable dictionary lookup. Scaling the scores by $1/\sqrt{d_k}$ keeps the softmax out of saturation. Self-attention is permutation-equivariant, so it needs **positional information**; **RoPE** rotates queries and keys so that scores depend only on relative position.
- A **modern decoder block** is pre-norm (RMSNorm) attention with RoPE and grouped-query attention, followed by a SwiGLU MLP, on a residual stream. It has about $12d^2$ parameters, **two thirds in the MLP**, and the MLP also dominates the compute until the context exceeds about $4d$ tokens.
- The **KV cache** makes generation linear per step but costs $2Ld$ numbers per token; MQA, GQA and latent attention shrink it. **FlashAttention** computes exact attention tile by tile with an online softmax and never stores the $n\times n$ matrix.
- **MLPs act as key–value memories.** Facts can be located in middle-layer MLPs and edited with a rank-one update (ROME), though knowledge is distributed and edits do not propagate to related facts.
- **Mixture of experts** replaces the MLP by many experts and a router: total parameters grow, active parameters do not. A load-balancing loss prevents collapse onto a few experts.
- **Linear attention** is an RNN with a matrix state. **SSMs** train as convolutions and run as RNNs; **Mamba** makes them selective, at the cost of a parallel scan; **Mamba-2** shows that selective SSMs are linear attention with a decay mask. Fixed-size states are cheap but cannot recall arbitrary context, hence hybrids.
- **Captioning** is conditional language modelling with an image encoder, trained with teacher forcing and decoded greedily or by beam search. The **Vision Transformer** turns patches into tokens; it needs large datasets or a strong augmentation and regularization recipe.

## Where this shows up next

- [[Graph Neural Networks]], the next lecture: convolution and attention generalized from grids and chains to arbitrary graphs (a transformer is a GNN on the complete graph).
- [[Self-Supervised Representation Learning]]: learning representations without labels, for images and text, with the encoders of this lecture and the contrastive loss derived in [[Energy-Based Models]]. CLIP, for instance, trains an image encoder and a text transformer so that matching image–caption pairs have high similarity.[^radford2021]
- [[Language Representations and Language Models]]: what the embeddings and the next-token objective learn: word-vector geometry, tokenization, masked and next-token prediction, scaling laws and in-context learning. A decoder-only transformer trained to predict the next token, scaled up, is a GPT.[^brown2020] [GPT-from-Scratch](https://github.com/parag1604/GPT-from-Scratch) builds one step by step, in the spirit of Karpathy's nanoGPT, then loads the real pretrained GPT-2 weights into the same code and fine-tunes them with LoRA.
- [[Non-parametric Models]]: attention with a Gaussian score is Nadaraya–Watson kernel regression; [[Energy-Based Models]]: one attention step is one retrieval step of a modern Hopfield network.
- **Generative models.** [[Denoising Diffusion Models]] embed the noise level with the sinusoidal encoding of this lecture and condition on text through cross-attention.
- [[Policy Gradients]]: fine-tuning a language model from human feedback treats it as a policy over tokens.
- **Textbooks.** Prince's *Understanding Deep Learning*,[^prince2023] Goodfellow, Bengio and Courville's *Deep Learning*,[^goodfellow2016] and the interactive *Dive into Deep Learning*[^zhang2023] cover the classical part of this lecture in more depth.

<!-- References: footnote definitions below render as the numbered "References" list at the bottom of the page -->

[^zhang2015]: Zhang, X., Zhao, J., & LeCun, Y. (2015). *Character-level Convolutional Networks for Text Classification*. NeurIPS. [arXiv:1509.01626](https://arxiv.org/abs/1509.01626)
[^radford2019]: Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). *Language Models are Unsupervised Multitask Learners*. OpenAI technical report. [PDF](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
[^touvron2023]: Touvron, H., Martin, L., Stone, K., et al. (2023). *Llama 2: Open Foundation and Fine-Tuned Chat Models*. arXiv. [arXiv:2307.09288](https://arxiv.org/abs/2307.09288)
[^jiang2024]: Jiang, A. Q., Sablayrolles, A., Roux, A., Mensch, A., et al. (2024). *Mixtral of Experts*. arXiv. [arXiv:2401.04088](https://arxiv.org/abs/2401.04088)
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
[^press2022]: Press, O., Smith, N. A., & Lewis, M. (2022). *Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation*. ICLR. [arXiv:2108.12409](https://arxiv.org/abs/2108.12409)
[^xiong2020]: Xiong, R., et al. (2020). *On Layer Normalization in the Transformer Architecture*. ICML. [arXiv:2002.04745](https://arxiv.org/abs/2002.04745)
[^ba2016]: Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). *Layer Normalization*. arXiv. [arXiv:1607.06450](https://arxiv.org/abs/1607.06450)
[^elhage2021]: Elhage, N., Nanda, N., Olsson, C., et al. (2021). *A Mathematical Framework for Transformer Circuits*. Transformer Circuits Thread. [transformer-circuits.pub](https://transformer-circuits.pub/2021/framework/index.html)
[^zhang2019]: Zhang, B., & Sennrich, R. (2019). *Root Mean Square Layer Normalization*. NeurIPS. [arXiv:1910.07467](https://arxiv.org/abs/1910.07467)
[^dauphin2017]: Dauphin, Y. N., Fan, A., Auli, M., & Grangier, D. (2017). *Language Modeling with Gated Convolutional Networks*. ICML. [arXiv:1612.08083](https://arxiv.org/abs/1612.08083)
[^shazeer2020]: Shazeer, N. (2020). *GLU Variants Improve Transformer*. arXiv. [arXiv:2002.05202](https://arxiv.org/abs/2002.05202)
[^kaplan2020]: Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., et al. (2020). *Scaling Laws for Neural Language Models*. arXiv. [arXiv:2001.08361](https://arxiv.org/abs/2001.08361)
[^shazeer2019]: Shazeer, N. (2019). *Fast Transformer Decoding: One Write-Head is All You Need*. arXiv. [arXiv:1911.02150](https://arxiv.org/abs/1911.02150)
[^ainslie2023]: Ainslie, J., Lee-Thorp, J., de Jong, M., Zemlyanskiy, Y., Lebrón, F., & Sanghai, S. (2023). *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*. EMNLP. [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)
[^jiang2023]: Jiang, A. Q., Sablayrolles, A., Mensch, A., Bamford, C., et al. (2023). *Mistral 7B*. arXiv. [arXiv:2310.06825](https://arxiv.org/abs/2310.06825)
[^deepseek2024]: DeepSeek-AI (2024). *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model*. arXiv. [arXiv:2405.04434](https://arxiv.org/abs/2405.04434)
[^dao2022]: Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Ré, C. (2022). *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*. NeurIPS. [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
[^geva2021]: Geva, M., Schuster, R., Berant, J., & Levy, O. (2021). *Transformer Feed-Forward Layers Are Key-Value Memories*. EMNLP. [arXiv:2012.14913](https://arxiv.org/abs/2012.14913)
[^allenzhu2024]: Allen-Zhu, Z., & Li, Y. (2024). *Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws*. arXiv. [arXiv:2404.05405](https://arxiv.org/abs/2404.05405)
[^dai2022]: Dai, D., Dong, L., Hao, Y., Sui, Z., Chang, B., & Wei, F. (2022). *Knowledge Neurons in Pretrained Transformers*. ACL. [arXiv:2104.08696](https://arxiv.org/abs/2104.08696)
[^meng2022]: Meng, K., Bau, D., Andonian, A., & Belinkov, Y. (2022). *Locating and Editing Factual Associations in GPT*. NeurIPS. [arXiv:2202.05262](https://arxiv.org/abs/2202.05262)
[^meng2023]: Meng, K., Sen Sharma, A., Andonian, A., Belinkov, Y., & Bau, D. (2023). *Mass-Editing Memory in a Transformer*. ICLR. [arXiv:2210.07229](https://arxiv.org/abs/2210.07229)
[^geva2023]: Geva, M., Bastings, J., Filippova, K., & Globerson, A. (2023). *Dissecting Recall of Factual Associations in Auto-Regressive Language Models*. EMNLP. [arXiv:2304.14767](https://arxiv.org/abs/2304.14767)
[^hase2023]: Hase, P., Bansal, M., Kim, B., & Ghandeharioun, A. (2023). *Does Localization Inform Editing? Surprising Differences in Causality-Based Localization vs. Knowledge Editing in Language Models*. NeurIPS. [arXiv:2301.04213](https://arxiv.org/abs/2301.04213)
[^cohen2024]: Cohen, R., Biran, E., Yoran, O., Globerson, A., & Geva, M. (2024). *Evaluating the Ripple Effects of Knowledge Editing in Language Models*. TACL. [arXiv:2307.12976](https://arxiv.org/abs/2307.12976)
[^elhage2022]: Elhage, N., Hume, T., Olsson, C., Schiefer, N., et al. (2022). *Toy Models of Superposition*. Transformer Circuits Thread. [arXiv:2209.10652](https://arxiv.org/abs/2209.10652)
[^shazeer2017]: Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G., & Dean, J. (2017). *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*. ICLR. [arXiv:1701.06538](https://arxiv.org/abs/1701.06538)
[^fedus2022]: Fedus, W., Zoph, B., & Shazeer, N. (2022). *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity*. JMLR, 23(120). [arXiv:2101.03961](https://arxiv.org/abs/2101.03961)
[^lepikhin2021]: Lepikhin, D., Lee, H., Xu, Y., Chen, D., et al. (2021). *GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding*. ICLR. [arXiv:2006.16668](https://arxiv.org/abs/2006.16668)
[^katharopoulos2020]: Katharopoulos, A., Vyas, A., Pappas, N., & Fleuret, F. (2020). *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention*. ICML. [arXiv:2006.16236](https://arxiv.org/abs/2006.16236)
[^sun2023]: Sun, Y., Dong, L., Huang, S., Ma, S., et al. (2023). *Retentive Network: A Successor to Transformer for Large Language Models*. arXiv. [arXiv:2307.08621](https://arxiv.org/abs/2307.08621)
[^peng2023]: Peng, B., Alcaide, E., Anthony, Q., Albalak, A., et al. (2023). *RWKV: Reinventing RNNs for the Transformer Era*. Findings of EMNLP. [arXiv:2305.13048](https://arxiv.org/abs/2305.13048)
[^gu2020]: Gu, A., Dao, T., Ermon, S., Rudra, A., & Ré, C. (2020). *HiPPO: Recurrent Memory with Optimal Polynomial Projections*. NeurIPS. [arXiv:2008.07669](https://arxiv.org/abs/2008.07669)
[^gu2022]: Gu, A., Goel, K., & Ré, C. (2022). *Efficiently Modeling Long Sequences with Structured State Spaces*. ICLR. [arXiv:2111.00396](https://arxiv.org/abs/2111.00396)
[^gupta2022]: Gupta, A., Gu, A., & Berant, J. (2022). *Diagonal State Spaces are as Effective as Structured State Spaces*. NeurIPS. [arXiv:2203.14343](https://arxiv.org/abs/2203.14343)
[^gu2022s4d]: Gu, A., Gupta, A., Goel, K., & Ré, C. (2022). *On the Parameterization and Initialization of Diagonal State Space Models*. NeurIPS. [arXiv:2206.11893](https://arxiv.org/abs/2206.11893)
[^gu2023]: Gu, A., & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. arXiv (COLM 2024). [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)
[^olsson2022]: Olsson, C., Elhage, N., Nanda, N., Joseph, N., et al. (2022). *In-context Learning and Induction Heads*. Transformer Circuits Thread. [arXiv:2209.11895](https://arxiv.org/abs/2209.11895)
[^jelassi2024]: Jelassi, S., Brandfonbrener, D., Kakade, S. M., & Malach, E. (2024). *Repeat After Me: Transformers are Better than State Space Models at Copying*. ICML. [arXiv:2402.01032](https://arxiv.org/abs/2402.01032)
[^dao2024]: Dao, T., & Gu, A. (2024). *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality*. ICML. [arXiv:2405.21060](https://arxiv.org/abs/2405.21060)
[^lieber2024]: Lieber, O., Lenz, B., Bata, H., Cohen, G., et al. (2024). *Jamba: A Hybrid Transformer-Mamba Language Model*. arXiv. [arXiv:2403.19887](https://arxiv.org/abs/2403.19887)
[^beltagy2020]: Beltagy, I., Peters, M. E., & Cohan, A. (2020). *Longformer: The Long-Document Transformer*. arXiv. [arXiv:2004.05150](https://arxiv.org/abs/2004.05150)
[^xiao2024]: Xiao, G., Tian, Y., Chen, B., Han, S., & Lewis, M. (2024). *Efficient Streaming Language Models with Attention Sinks*. ICLR. [arXiv:2309.17453](https://arxiv.org/abs/2309.17453)
[^leviathan2023]: Leviathan, Y., Kalman, M., & Matias, Y. (2023). *Fast Inference from Transformers via Speculative Decoding*. ICML. [arXiv:2211.17192](https://arxiv.org/abs/2211.17192)
[^chen2023]: Chen, C., Borgeaud, S., Irving, G., Lespiau, J.-B., Sifre, L., & Jumper, J. (2023). *Accelerating Large Language Model Decoding with Speculative Sampling*. arXiv. [arXiv:2302.01318](https://arxiv.org/abs/2302.01318)
[^gloeckle2024]: Gloeckle, F., Youbi Idrissi, B., Rozière, B., Lopez-Paz, D., & Synnaeve, G. (2024). *Better & Faster Large Language Models via Multi-token Prediction*. ICML. [arXiv:2404.19737](https://arxiv.org/abs/2404.19737)
[^vinyals2015]: Vinyals, O., Toshev, A., Bengio, S., & Erhan, D. (2015). *Show and Tell: A Neural Image Caption Generator*. CVPR. [arXiv:1411.4555](https://arxiv.org/abs/1411.4555)
[^xu2015]: Xu, K., Ba, J., Kiros, R., Cho, K., Courville, A., Salakhutdinov, R., Zemel, R., & Bengio, Y. (2015). *Show, Attend and Tell: Neural Image Caption Generation with Visual Attention*. ICML. [arXiv:1502.03044](https://arxiv.org/abs/1502.03044)
[^papineni2002]: Papineni, K., Roukos, S., Ward, T., & Zhu, W.-J. (2002). *BLEU: a Method for Automatic Evaluation of Machine Translation*. ACL. [ACL Anthology P02-1040](https://aclanthology.org/P02-1040/)
[^dosovitskiy2021]: Dosovitskiy, A., et al. (2021). *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*. ICLR. [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
[^touvron2021]: Touvron, H., Cord, M., Douze, M., Massa, F., Sablayrolles, A., & Jégou, H. (2021). *Training data-efficient image transformers & distillation through attention*. ICML. [arXiv:2012.12877](https://arxiv.org/abs/2012.12877)
[^cubuk2020]: Cubuk, E. D., Zoph, B., Shlens, J., & Le, Q. V. (2020). *RandAugment: Practical automated data augmentation with a reduced search space*. CVPR Workshops. [arXiv:1909.13719](https://arxiv.org/abs/1909.13719)
[^szegedy2016]: Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., & Wojna, Z. (2016). *Rethinking the Inception Architecture for Computer Vision* (introduces label smoothing). CVPR. [arXiv:1512.00567](https://arxiv.org/abs/1512.00567)
[^radford2021]: Radford, A., et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision*. ICML. [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)
[^brown2020]: Brown, T. B., et al. (2020). *Language Models are Few-Shot Learners*. NeurIPS. [arXiv:2005.14165](https://arxiv.org/abs/2005.14165)
[^prince2023]: Prince, S. J. D. (2023). *Understanding Deep Learning*. MIT Press. [udlbook.github.io](https://udlbook.github.io/udlbook/)
[^goodfellow2016]: Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. [deeplearningbook.org](https://www.deeplearningbook.org/)
[^zhang2023]: Zhang, A., Lipton, Z. C., Li, M., & Smola, A. J. (2023). *Dive into Deep Learning*. Cambridge University Press. [d2l.ai](https://d2l.ai/)
