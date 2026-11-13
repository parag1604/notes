---
title: Language Representations and Language Models
tags:
  - machine-learning
  - self-supervised-learning
  - word-embeddings
  - language-models
  - representation-learning
date: 2026-11-13
draft: false
prev: "[[Self-Supervised Representation Learning]]"
next: "[[Variational Autoencoders]]"
---

> **Prerequisites:** [[Self-Supervised Representation Learning]] (NCE and InfoNCE: learning by telling real pairs from fake ones), [[Matrix Factorization and Dimensionality Reduction]] (SVD and low-rank approximation), [[Sequence Models]] (attention, the transformer block, causal masks), [[Linear Models#More than two classes: softmax regression|softmax regression]] and cross-entropy, and the [[Sample Statistics#Chernoff bounds|Chernoff bound]] with the [[Sample Statistics#Concentration and the union bound|union bound]] for the near-orthogonality argument.
>
> **TL;DR:**
> - Text is the largest unlabelled dataset there is, and it supervises itself: **hide part of it and predict the hidden part from the rest**. Three choices of what to hide give three families: neighbouring words (**word2vec**, **GloVe**), masked words (**BERT**), and the next word (**GPT**).
> - Word vectors store **concepts as directions**: $\text{king}-\text{man}+\text{woman}\approx\text{queen}$. Independent concepts can sit in orthogonal directions, but only $d$ of those fit in $d$ dimensions. If directions only need to be **almost** orthogonal, **exponentially many** fit ($\approx e^{c\,d\,\varepsilon^2}$), which is how networks store more features than they have neurons (**superposition**).
> - Next-token prediction with cross-entropy is **compression**. Its loss falls as a **power law** in parameters, data and compute, and at the scale of billions of parameters the model starts to **learn from examples in its prompt** without any weight update (**in-context learning**). Instruction tuning and RLHF turn the predictor into an assistant.
>
> **Notation:** a vocabulary of $V$ tokens; a token sequence $x_1,\dots,x_T$ (indices from 1 here, as in most of the language-model literature); a word $w$ with **input vector** $\mathbf{v}_w\in\mathbb{R}^d$ and context word $c$ with **output vector** $\mathbf{u}_c\in\mathbb{R}^d$; $\#(\cdot)$ for counts in the corpus; $\sigma$ the logistic sigmoid; $\log$ natural, so losses are in nats.
>
> **Code:** [Word-Embeddings-and-Text-Classification](https://github.com/parag1604/Word-Embeddings-and-Text-Classification): exploring pretrained fastText vectors (neighbours, analogies), then AG News classification with LSTMs and a transformer encoder on learned and GloVe embeddings. [GPT-from-Scratch](https://github.com/parag1604/GPT-from-Scratch): a bigram model grown into a GPT on tiny Shakespeare, a BPE tokenizer, a KV cache, GPT-2 weights loaded into the same code, and LoRA fine-tuning for tweet sentiment. [modern_llms](https://github.com/paragduttaiisc/modern_llms): nanoGPT rebuilt on Hugging Face and extended with the techniques of current frontier models (multi-head latent attention, mixture of experts, the Muon optimizer, multi-token prediction with self-speculative decoding, RoPE/SwiGLU/RMSNorm) to train billion-parameter LLMs on local hardware and serve them fast.

## Motivation

Labelled data is expensive: somebody has to look at every example. Text is the opposite. Trillions of words are freely available, and every sentence carries its own supervision, because the words of a sentence constrain each other. If we hide one word, the others tell us a lot about it. "The cat sat on the ___" has few sensible completions.

This lecture follows one idea through three increasingly powerful forms.

> [!tip] Self-supervision
> Split each example into a visible part and a hidden part, and train a model to **predict the hidden part from the visible part**. No labels are needed; the data labels itself. The [[Self-Supervised Representation Learning|previous lecture]] applied the idea to images (predict which two crops come from the same photo). In language it is even more natural.

Why should predicting words teach anything useful? The **distributional hypothesis** answers: words that occur in similar contexts tend to have similar meanings.[^harris1954] "Coffee" and "tea" appear next to "cup", "drink", "hot" and "morning"; a model that must predict those neighbours is pushed to give the two words similar representations. Meaning is not given to the model; it is the cheapest way to predict context.

The three forms differ in what is hidden (Figure 1):

1. **Neighbouring words** (word2vec, GloVe): predict the words in a small window around a word. The result is one vector per word.
2. **Masked words** (BERT): hide random words and predict them from both sides. The result is a vector per word *in context*.
3. **The next word** (GPT): predict each word from the words before it. The result is a model that can **generate** text, and at scale, one that can follow instructions and learn from examples in its prompt.

![[Machine Learning/images/language-models-objectives.svg]]
*Figure 1. Three self-supervised objectives on the same sentence. Orange boxes are prediction targets, blue boxes are masked inputs, dotted boxes are unused. (a) Skip-gram predicts the neighbours of a word. (b) A masked language model predicts hidden words from both sides. (c) An autoregressive language model predicts the next word from the left only.*

Before any of this, text has to be cut into pieces a model can index.

## Tokenization

A model sees a sequence of integers, one per **token**. What a token is matters more than it seems.

| unit | vocabulary | sequence length | unknown words |
|---|---|---|---|
| characters (or bytes) | tiny (~100, or 256 bytes) | long | none |
| words | huge (hundreds of thousands) | short | many: names, typos, new words, morphology ("unhappiness") |
| **subwords** | chosen, typically 30k–200k | medium | none, if the base alphabet is bytes |

Words waste parameters on rare forms and still meet unknown words at test time. Characters never fail but make sequences several times longer, and attention cost grows with length (see [[Sequence Models]]). **Subwords** sit in between: frequent words are single tokens, and rare words are spelled out from frequent pieces ("tokenization" → "token" + "ization").

### Byte-pair encoding

Byte-pair encoding (BPE) was a data-compression trick; Sennrich et al. turned it into a subword tokenizer for translation.[^sennrich2016] Start from characters, and repeatedly **merge the most frequent adjacent pair** into a new symbol.

> [!example] Algorithm: learning BPE merges
> **Input:** word counts from a corpus, and a number of merges $M$.
> 1. Write each word as a sequence of characters followed by an end-of-word marker `_`.
> 2. Count every adjacent pair of symbols, weighting each word by its count.
> 3. Merge the most frequent pair everywhere into one new symbol, and record the merge.
> 4. Repeat steps 2–3 $M$ times.
>
> To **encode** new text, split it into characters and apply the recorded merges in the order they were learned. The final vocabulary is the base alphabet plus the $M$ merged symbols.

A tiny worked example. Take a corpus with the words *tea* (6 times), *ten* (4), *tent* (3), *net* (5) and *tenet* (2).
- **Merge 1.** The pair `t e` occurs $6+4+3+2=15$ times (in *tea*, *ten*, *tent*, and at the start of *tenet*), more than any other pair, so `te` becomes a symbol.
- **Merge 2.** Now `t _` (a word-final *t*) is the most frequent pair, with $3+5+2=10$ occurrences, giving `t_`.
- **Merge 3.** `te n` occurs $4+3+2=9$ times, giving `ten`.
- **Merge 4.** `e t_` occurs 7 times (5 in *net*, 2 in *tenet*), giving `et_`.

After four merges the corpus reads `te a _`, `ten _`, `ten t_`, `n et_`, `ten et_`. An unseen word still encodes: *tents* becomes `ten t s _`, built from learned pieces plus single characters. Nothing is ever out of vocabulary.

> [!example]- Code: BPE in 30 lines (tested)
> ```python
> from collections import Counter
>
> def bpe_train(word_counts, num_merges):
>     """word_counts: dict word -> count. Returns the merges and the segmented corpus."""
>     corpus = {tuple(w) + ("_",): c for w, c in word_counts.items()}  # "_" marks the end of a word
>     merges = []
>     for _ in range(num_merges):
>         pairs = Counter()
>         for symbols, c in corpus.items():
>             for a, b in zip(symbols, symbols[1:]):
>                 pairs[a, b] += c
>         if not pairs:
>             break
>         best = max(pairs, key=pairs.get)          # most frequent adjacent pair
>         merges.append(best)
>         new_corpus = {}
>         for symbols, c in corpus.items():
>             out, i = [], 0
>             while i < len(symbols):
>                 if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == best:
>                     out.append(symbols[i] + symbols[i + 1]); i += 2
>                 else:
>                     out.append(symbols[i]); i += 1
>             new_corpus[tuple(out)] = c
>         corpus = new_corpus
>     return merges, corpus
>
> merges, corpus = bpe_train({"tea": 6, "ten": 4, "tent": 3, "net": 5, "tenet": 2}, 4)
> # merges: [('t','e'), ('t','_'), ('te','n'), ('e','t_')]
> ```
> Ties are broken by first occurrence. The encoder (in the test script) applies the merges in order.

### Other tokenizers

- **Byte-level BPE** (GPT-2): run BPE on UTF-8 **bytes** instead of Unicode characters. The base alphabet has only 256 symbols, so any string, in any script, with any emoji, can be encoded. GPT-2's vocabulary has 50,257 tokens.[^radford2019]
- **WordPiece** (BERT): like BPE, but each merge is chosen to most increase the likelihood of the training data under a simple language model, rather than by raw pair count.[^schuster2012]<sup>,</sup>[^wu2016]
- **SentencePiece**: treats the input as a raw character stream, with the space as an ordinary symbol (written `▁`), so no language-specific pre-splitting into words is needed. It implements BPE and a unigram-language-model tokenizer.[^kudo2018]

### Why tokenization matters

The model never sees characters, only token ids, and this leaks into behaviour.
- **Arithmetic.** GPT-3.5 and GPT-4 tokenize numbers into chunks of up to three digits, read left to right, so "12345" splits as `123|45` and digits of equal place value land in different positions of different tokens. Forcing right-to-left chunking (by inserting commas: "12,345") markedly improves their addition accuracy.[^singh2024] Many newer models simply split numbers into single digits.
- **Spelling and counting.** "How many r's are in *strawberry*?" is hard for a model that sees two or three opaque tokens, not ten letters.
- **Languages.** A tokenizer trained mostly on English cuts other languages into many more pieces: the same text translated into different languages can differ in token count by up to 15 times.[^petrov2023] Longer sequences cost more, run slower, and fit less content into the context window.

## Static word embeddings

The first form of self-supervision gives each word type one vector. An embedding table maps token $w$ to row $w$ of a matrix (see [[Sequence Models]] for how embeddings feed a network). The question is how to train the table without labels.

### Word2vec: skip-gram and CBOW

Mikolov et al. proposed two shallow models.[^mikolov2013a] The **skip-gram** model predicts each word's neighbours, within a window of $\pm m$ positions, from the word itself:

$$
\max_{\mathbf{U},\mathbf{V}}\ \sum_{t=1}^{T}\ \sum_{\substack{-m\le j\le m\\ j\ne0}}\log p(x_{t+j}\mid x_t),
\qquad
p(c\mid w)=\frac{\exp(\mathbf{u}_c^\top\mathbf{v}_w)}{\sum_{c'=1}^{V}\exp(\mathbf{u}_{c'}^\top\mathbf{v}_w)} .
\tag{1}
$$

Each word has two vectors: $\mathbf{v}_w$ when it is the centre word and $\mathbf{u}_c$ when it is a context word. This is [[Linear Models#More than two classes: softmax regression|softmax regression]] with $V$ classes, whose input features are themselves learned. **CBOW** (continuous bag of words) flips the direction: average the vectors of the context words and predict the centre word. CBOW is faster; skip-gram does better on rare words, because every occurrence of a rare word generates $2m$ training pairs for it.

**The cost.** The normalizer in (1) sums over the whole vocabulary. Its gradient,

$$
\nabla_{\mathbf{v}_w}\log p(c\mid w)=\mathbf{u}_c-\sum_{c'=1}^{V}p(c'\mid w)\,\mathbf{u}_{c'},
$$

touches all $V$ output vectors for every training pair: with $V=10^5$ and billions of pairs, this is far too slow. The structure is the familiar one: pull the true context up, push every other word down in proportion to how probable the model thinks it is.

### Negative sampling

The second word2vec paper replaced the $V$-way classification by a **binary** one: is $(w,c)$ a real (word, context) pair from the corpus, or a fake pair whose context word was drawn at random?[^mikolov2013b] For each real pair, draw $k$ noise words $c_1,\dots,c_k$ from a noise distribution $P_n$ and maximize

$$
\log\sigma(\mathbf{u}_c^\top\mathbf{v}_w)+\sum_{i=1}^{k}\log\sigma(-\mathbf{u}_{c_i}^\top\mathbf{v}_w),
\qquad c_i\sim P_n .
\tag{2}
$$

Each step now touches only $k+1$ output vectors, with $k$ between 2 and 20. The authors found that the unigram distribution raised to the power $3/4$, $P_n(c)\propto\#(c)^{3/4}$, works best: it samples rare words more often than their raw frequency.

This is **noise-contrastive estimation** in a simplified form: learn a distribution by training a classifier to separate data from noise, as derived in [[Self-Supervised Representation Learning]].[^gutmann2010]<sup>,</sup>[^mnih2012] Full NCE includes the term $\log kP_n(c)$ in the classifier's logit, which makes the model's scores converge to true log-probabilities. Negative sampling drops that term, so it no longer estimates $p(c\mid w)$, but it still learns good vectors, and exactly what it learns is the subject of the next subsection.[^goldberg2014]

**Subsampling frequent words.** Words like "the" and "of" occur in nearly every window and say little about their neighbours. Word2vec discards each occurrence of word $w$ with probability $1-\sqrt{t/f(w)}$, where $f(w)$ is the word's frequency and $t\approx10^{-5}$. Frequent words are thinned out, training is faster, and the windows effectively widen around content words.[^mikolov2013b]

### What skip-gram actually learns: a shifted PMI matrix

Levy and Goldberg showed that skip-gram with negative sampling is, at its optimum, a **matrix factorization**.[^levy2014] Collect the objective (2) over the corpus and look at a single (word, context) pair. Write $x=\mathbf{u}_c^\top\mathbf{v}_w$. The pair appears $\#(w,c)$ times as a positive, and as a negative about $k\,\#(w)\,P_n(c)$ times, so its share of the objective is

$$
\ell(x)=\#(w,c)\log\sigma(x)+k\,\#(w)\,P_n(c)\log\sigma(-x).
$$

If the vectors are free to set each dot product independently (dimension $d$ large enough), each $x$ is maximized separately. Setting $\ell'(x)=0$, using $\tfrac{d}{dx}\log\sigma(x)=\sigma(-x)$, gives

$$
\mathbf{u}_c^\top\mathbf{v}_w=\log\frac{\#(w,c)}{\#(w)\,P_n(c)}-\log k .
\tag{3}
$$

With the plain unigram noise $P_n(c)=\#(c)/|\mathcal{D}|$, where $|\mathcal{D}|$ is the number of pairs, the first term is the **pointwise mutual information**

$$
\operatorname{PMI}(w,c)=\log\frac{p(w,c)}{p(w)\,p(c)},
$$

so skip-gram factorizes the matrix $\operatorname{PMI}-\log k$: the word vectors are its rows' factor and the context vectors its columns' factor. PMI is positive when $w$ and $c$ co-occur more often than chance and negative when less often.

> [!proof]- Proof of (3)
> $\ell'(x)=\#(w,c)\,\sigma(-x)-k\,\#(w)P_n(c)\,\sigma(x)=0$, so $\sigma(x)/\sigma(-x)=\#(w,c)/(k\,\#(w)P_n(c))$. Since $\sigma(x)/\sigma(-x)=e^x$, taking logs gives (3). The second derivative, $-\big(\#(w,c)+k\#(w)P_n(c)\big)\sigma(x)\sigma(-x)$, is negative, so this is the maximum. With $P_n(c)=\#(c)/|\mathcal{D}|$ and $p(w,c)=\#(w,c)/|\mathcal{D}|$, $p(w)=\#(w)/|\mathcal{D}|$, the first term of (3) is $\log\frac{\#(w,c)|\mathcal{D}|}{\#(w)\#(c)}=\operatorname{PMI}(w,c)$.

In practice $d$ (a few hundred) is much smaller than $V$, so skip-gram finds a **low-rank** approximation of the shifted PMI matrix, weighted by how often each pair occurs. The best low-rank approximation of a matrix is the business of the singular value decomposition and the Eckart–Young theorem from [[Matrix Factorization and Dimensionality Reduction]]; indeed, an SVD of the (positive) PMI matrix gives word vectors competitive with word2vec.[^levy2014] Figure 2(b) checks (3) on a toy corpus.

### GloVe: regression on log co-occurrence counts

GloVe (global vectors) starts from the same co-occurrence statistics but fits them directly.[^pennington2014] Let $X_{ij}$ count how often word $j$ appears in the context of word $i$, and $P_{ij}=X_{ij}/\sum_kX_{ik}$.

**Why ratios.** A single probability $P_{ik}$ mixes meaning with frequency. Ratios cancel the frequency. Compare $i=$ *king* and $j=$ *queen* through probe words $k$:
- $k=$ *he*: much more likely near *king*, so $P_{ik}/P_{jk}\gg1$;
- $k=$ *she*: $P_{ik}/P_{jk}\ll1$;
- $k=$ *crown*, related to both, and $k=$ *fish*, related to neither: $P_{ik}/P_{jk}\approx1$.

The ratio isolates exactly the dimension along which *king* and *queen* differ (gender) and ignores everything they share. GloVe asks the vectors to reproduce these ratios through **differences**: $\exp\big((\mathbf{w}_i-\mathbf{w}_j)^\top\tilde{\mathbf{w}}_k\big)=P_{ik}/P_{jk}$. This holds if $\mathbf{w}_i^\top\tilde{\mathbf{w}}_k=\log P_{ik}=\log X_{ik}-\log X_i$. The term $\log X_i$ does not depend on $k$, so it is absorbed into a bias $b_i$, and a context bias $\tilde b_k$ is added for symmetry. The result is a weighted **log-bilinear regression**:

$$
J=\sum_{i,j:\,X_{ij}>0}f(X_{ij})\big(\mathbf{w}_i^\top\tilde{\mathbf{w}}_j+b_i+\tilde b_j-\log X_{ij}\big)^2 .
\tag{4}
$$

The **weighting function** $f$ is

$$
f(x)=\begin{cases}(x/x_{\max})^{\alpha}&x<x_{\max},\\1&\text{otherwise,}\end{cases}
\qquad x_{\max}=100,\ \alpha=3/4 .
$$

It has three jobs: $f(0)=0$, so the $\log0$ of unseen pairs never enters; rare co-occurrences, which are noisy, get small weight; and very frequent pairs ("of the") are capped so they cannot dominate. Compared with (3), GloVe fits $\log X_{ij}$ (PMI up to the biases) with an explicit weighted least-squares loss over the whole count matrix, instead of streaming through the corpus with SGD.

## Concepts as directions

Now Parag's favourite fact about word vectors. Train skip-gram on a large corpus, take the vectors for *king*, *man* and *woman*, and compute $\mathbf{v}_{\text{king}}-\mathbf{v}_{\text{man}}+\mathbf{v}_{\text{woman}}$. The nearest word vector to the result is $\mathbf{v}_{\text{queen}}$.[^mikolov2013c] The same arithmetic gives *Paris* − *France* + *Italy* ≈ *Rome* and *walking* − *walk* + *swim* ≈ *swimming*.

**The geometric reading.** The difference $\mathbf{v}_{\text{woman}}-\mathbf{v}_{\text{man}}$ is a direction that means "female instead of male". Adding it to *king* moves along that direction while keeping everything else (royalty, adulthood) fixed. The four words form a **parallelogram**: $\mathbf{v}_{\text{queen}}-\mathbf{v}_{\text{king}}\approx\mathbf{v}_{\text{woman}}-\mathbf{v}_{\text{man}}$. A **concept is a direction**, and an analogy is a parallelogram.

**Why it happens.** Equation (3) explains it. Suppose that for every context word $c$, *king* and *queen* differ in their context statistics the same way *man* and *woman* do:

$$
\operatorname{PMI}(\text{king},c)-\operatorname{PMI}(\text{queen},c)\approx\operatorname{PMI}(\text{man},c)-\operatorname{PMI}(\text{woman},c)\quad\text{for all }c.
$$

Gender words (*he*, *his*, *she*, *her*) shift both pairs by the same amount, and words like *crown* or *work* do not distinguish within a pair. If $\mathbf{u}_c^\top\mathbf{v}_w\approx\operatorname{PMI}(w,c)$, this says $\mathbf{u}_c^\top(\mathbf{v}_{\text{king}}-\mathbf{v}_{\text{queen}}-\mathbf{v}_{\text{man}}+\mathbf{v}_{\text{woman}})\approx0$ for all $c$. When the context vectors span $\mathbb{R}^d$, the vector in brackets must be near zero: the parallelogram.[^arora2016]

Figure 2 reproduces this on a toy corpus. Every "sentence" contains one of eight nouns, which differ in three binary attributes (gender, royalty, age), plus context words drawn from attribute-specific pools (*he*/*she*, *crown*/*field*, *work*/*toy*) and filler words. Skip-gram with negative sampling, $d=20$, learns vectors in which all four analogies tested come out right.

![[Machine Learning/images/language-models-sgns.svg]]
*Figure 2. Skip-gram with negative sampling ($k=5$, $d=20$) on a synthetic corpus of 30,000 sentences. (a) The eight noun vectors projected onto their top two principal components; the male→female arrows (orange) are nearly parallel, so king − man + woman lands on queen. The analogies king − man + woman → queen, prince − boy + girl → princess, king − queen + princess → prince and man − boy + girl → woman are all solved. (b) Learned dot products $\mathbf{v}_w^\top\mathbf{u}_c$ against the shifted PMI of (3) for the 1,850 (word, context) pairs seen at least 20 times: correlation 0.95, mean absolute difference 0.08.*

### Caveats: analogies are less clean than they look

- **The exclusion trick.** The standard test (called 3CosAdd) returns $\arg\max_{d}\cos(\mathbf{v}_d,\ \mathbf{v}_b-\mathbf{v}_a+\mathbf{v}_c)$ over all words **except $a$, $b$ and $c$**. Without that exclusion, the nearest word to $\text{king}-\text{man}+\text{woman}$ in real embeddings is very often *king* itself: the offset is small compared with the word vector. Linzen showed that on some analogy categories, simply returning the nearest neighbour of $c$ (ignoring $a$ and $b$ altogether) scores almost as well as the full arithmetic.[^linzen2016] (In the toy corpus of Figure 2, the offsets are large and the answer is right even without the exclusion; real corpora are messier.)
- **Selection bias.** The famous examples are the ones that work. Accuracy varies widely across relation types: inflections and country–capital pairs work well, while many lexical-semantic relations do not.[^rogers2017]
- **Biases come along.** The same arithmetic surfaces stereotypes present in the training text, which the embeddings faithfully encode.

### The linear representation hypothesis

The idea that concepts are directions extends well beyond static word vectors. The **linear representation hypothesis** says that the internal activations of language models represent many high-level concepts (gender, language, tense, truth) as directions, so that a concept can be read off with a linear probe and steered by adding a vector. Park et al. make the claim precise and show that the right notion of "orthogonal" for such directions is not always the Euclidean one: it depends on an inner product induced by the model's unembedding matrix.[^park2024]

### Independent concepts, orthogonal directions

If concepts are directions, how should the model arrange them? Concepts that vary independently ought not to interfere. Gender and royalty are independent: changing one should not change the reading of the other. With a linear readout $\mathbf{r}_j^\top\mathbf{h}$ for concept $j$, adding the gender direction $\mathbf{g}$ to a representation changes the royalty readout by $\mathbf{r}_{\text{royal}}^\top\mathbf{g}$, which is zero exactly when the two directions are **orthogonal**. In Figure 2(a) the gender arrows are roughly perpendicular to the royalty separation.

But in $\mathbb{R}^d$ there are **at most $d$** mutually orthogonal directions. Word vectors have a few hundred dimensions, and the residual stream of a large language model a few thousand. There are many more concepts than that: every named entity, every topic, every syntactic feature, every fact. Does a model really have to choose only $d$ of them?

## Almost orthogonal: exponentially many concepts

No. The way out is to accept **almost orthogonal** directions, with $|\cos|\le\varepsilon$ for some small $\varepsilon$ instead of exactly zero. In high dimensions, almost orthogonal directions are not just a little more plentiful than orthogonal ones; there are **exponentially many** of them.

### Random directions are nearly orthogonal

Draw two independent random unit vectors $\mathbf{a},\mathbf{b}$ in $\mathbb{R}^d$. By symmetry $\mathbb{E}[\mathbf{a}^\top\mathbf{b}]=0$, and since $\mathbb{E}[(\mathbf{a}^\top\mathbf{b})^2]=\sum_i\mathbb{E}[a_i^2]\,\mathbb{E}[b_i^2]=d\cdot\tfrac1d\cdot\tfrac1d$,

$$
\operatorname{Var}(\mathbf{a}^\top\mathbf{b})=\frac1d .
$$

The cosine concentrates around zero with standard deviation $1/\sqrt d$ (Figure 3). In 3 dimensions random directions are often nearly parallel; in 3,000 dimensions the largest $|\cos|$ among half a million random pairs is 0.08. Most directions in a high-dimensional space are nearly perpendicular to any given one.

![[Machine Learning/images/language-models-random-cosines.svg]]
*Figure 3. Cosine similarity between 1,000 random unit vectors (499,500 pairs) in $d$ dimensions, log density. The standard deviation matches $1/\sqrt d$ to three decimals (0.577, 0.183, 0.058, 0.018). The largest $|\cos|$ among all pairs drops from 1.00 at $d=3$ to 0.76, 0.28 and 0.08.*

The variance alone gives Chebyshev's bound, which decays only like $1/(d\varepsilon^2)$. The tail is in fact **exponentially** small, like a Gaussian with variance $1/d$:

$$
P\big(|\mathbf{a}^\top\mathbf{b}|>\varepsilon\big)\le2\,e^{-d\varepsilon^2/2}\qquad\big(0<\varepsilon\le\tfrac{1}{\sqrt2}\big).
\tag{5}
$$

### Counting the directions

Now draw $N$ random unit vectors. A **union bound** over the fewer than $N^2/2$ pairs gives

$$
P\big(\text{some pair has }|\cos|>\varepsilon\big)<\frac{N^2}{2}\cdot2e^{-d\varepsilon^2/2}=N^2e^{-d\varepsilon^2/2}.
$$

This is below 1 whenever $N<e^{d\varepsilon^2/4}$. A probability below 1 means that at least one draw succeeds, so:

> [!note] Near-orthogonal capacity
> For $0<\varepsilon\le1/\sqrt2$, $\mathbb{R}^d$ contains at least $\big\lceil e^{d\varepsilon^2/4}\big\rceil-1$ unit vectors whose pairwise cosines all satisfy $|\cos|\le\varepsilon$. Exactly orthogonal: at most $d$. Almost orthogonal: **exponentially many in $d$**.

For example, with $d=1{,}000$ and $\varepsilon=0.2$ the guarantee is $e^{10}\approx22{,}000$ directions, and with $d=4{,}096$ (a typical LLM width) it is $e^{41}\approx6\times10^{17}$. The constant $1/4$ is loose: a greedy search finds many more (Figure 4).

> [!proof]- Proof of the tail bound (5)
> By rotational symmetry we may fix $\mathbf{b}=\mathbf{e}_1$, so $\mathbf{a}^\top\mathbf{b}=a_1$, the first coordinate of a uniform point on the unit sphere. The set $\{a_1\ge\varepsilon\}$ is a **spherical cap**. Its share of the sphere's surface equals the share of the unit ball's volume taken by the cone $K$ from the origin to the cap.
>
> Every point of that cone lies within distance $\sqrt{1-\varepsilon^2}$ of the point $\varepsilon\mathbf{e}_1$ (for $\varepsilon\le1/\sqrt2$). To see this, a point on the cap's rim is $\varepsilon\mathbf{e}_1+\sqrt{1-\varepsilon^2}\,\mathbf{n}$ with $\mathbf{n}\perp\mathbf{e}_1$; points on the cap itself are at most that far from $\varepsilon\mathbf{e}_1$; and the segments from the origin to the cap stay within that ball because the origin itself is at distance $\varepsilon\le\sqrt{1-\varepsilon^2}$ and balls are convex. So $K$ sits inside a ball of radius $\sqrt{1-\varepsilon^2}$, and
>
> $$
> P(a_1\ge\varepsilon)=\frac{\operatorname{vol}(K)}{\operatorname{vol}(\text{unit ball})}\le\big(1-\varepsilon^2\big)^{d/2}\le e^{-d\varepsilon^2/2},
> $$
>
> using $1-u\le e^{-u}$. The event $a_1\le-\varepsilon$ is symmetric, which gives the factor 2. (This is the cap bound of Ball's notes on convex geometry; a Chernoff argument on $a_1=g_1/\lVert\mathbf{g}\rVert$ with Gaussian $\mathbf{g}$ gives the same rate with different constants.)

### The Johnson–Lindenstrauss view

The same fact appears in the classical **Johnson–Lindenstrauss lemma**: any $n$ points in any dimension can be mapped linearly into $k=O(\log n/\varepsilon^2)$ dimensions while preserving all pairwise distances, and hence inner products of unit vectors, up to about $\varepsilon$.[^johnson1984] A random Gaussian projection does it with high probability, and the proof is again concentration plus a union bound.[^dasgupta2003] Apply it to the $n$ standard basis vectors of $\mathbb{R}^n$, which are exactly orthogonal: their images are $n$ unit-ish vectors in $k$ dimensions with pairwise $|\cos|\lesssim\varepsilon$. Solving $k\approx C\log n/\varepsilon^2$ for $n$ gives $n\approx e^{k\varepsilon^2/C}$, the same exponential capacity. Johnson–Lindenstrauss says that **$n$ independent features can be compressed into $O(\log n)$ dimensions** at the price of a small, controlled amount of interference.

![[Machine Learning/images/language-models-packing.svg]]
*Figure 4. How many directions with all pairwise $|\cos|\le\varepsilon$ fit in $d$ dimensions. Solid: a greedy search that draws 200,000 random unit vectors and keeps each one that is nearly orthogonal to all kept so far (a lower bound on the true capacity; capped at 5,000). Dashed: the guarantee $e^{d\varepsilon^2/4}$ from the union bound. Dotted: the $d$ exactly orthogonal directions. With $\varepsilon=0.5$ the greedy search finds 29 directions in $d=8$, 157 in $d=16$, 1,923 in $d=32$ and hits the cap at $d=48$; with $\varepsilon=0.3$ it finds 24, 83, 476 and 2,190 directions in $d=16$, 32, 64 and 96. Both grow exponentially in $d$ (straight lines on the log scale), far above $d$; the union-bound guarantee has the right shape but a pessimistic constant.*

## Superposition

The capacity argument suggests what a network might do when it needs to represent more features than it has dimensions: give each feature a direction, and let the directions overlap slightly. Elhage et al. called this **superposition** and studied it in a toy model small enough to understand completely.[^elhage2022]

**The toy model.** A vector $\mathbf{x}\in\mathbb{R}^n$ holds $n$ features. Each feature is zero with probability $1-p$, and otherwise uniform on $[0,1]$; feature $i$ has importance $I_i$. A linear map compresses $\mathbf{x}$ into $m<n$ dimensions, and the transpose with a ReLU reconstructs it:

$$
\mathbf{h}=\mathbf{W}\mathbf{x},\qquad
\hat{\mathbf{x}}=\operatorname{ReLU}\big(\mathbf{W}^\top\mathbf{h}+\mathbf{b}\big),\qquad
\mathcal{L}=\mathbb{E}\Big[\sum_{i=1}^{n}I_i\,(x_i-\hat x_i)^2\Big].
$$

Column $\mathbf{W}_i\in\mathbb{R}^m$ is the direction assigned to feature $i$. Feature $i$ is read back as $\mathbf{W}_i^\top\mathbf{h}=\lVert\mathbf{W}_i\rVert^2x_i+\sum_{j\ne i}(\mathbf{W}_i^\top\mathbf{W}_j)\,x_j$: its own value plus **interference** from every other active feature.

**Sparsity decides.** If features are dense (most are active at once), interference is paid on almost every input and is too costly, so the model behaves like PCA: it represents the $m$ most important features in orthogonal directions and ignores the rest. If features are **sparse**, two features are rarely active together, interference is rarely paid, and the ReLU with a negative bias can clip the small interference that remains. Then it pays to squeeze in more features than dimensions (Figure 5). In 2 dimensions with 5 sparse features, the model places them at the vertices of a regular pentagon, the most nearly orthogonal arrangement of 5 directions in a plane. In larger models, pairs of features often share one direction with opposite signs (antipodal pairs), since a ReLU can separate $+x_i$ from $-x_j$.

![[Machine Learning/images/language-models-superposition.svg]]
*Figure 5. The toy model of superposition, trained with Adam for 6,000 steps (importances $I_i=0.9^{\,i-1}$). (a) $n=5$ dense features ($p=1$) in $m=2$ dimensions: only the two most important features get a direction (column norms 1.00, 1.00, the rest below 0.01). (b) Sparse features ($p=0.05$): all five are stored, as a pentagon (column norms 1.13–1.14). (c) $n=20$ features in $m=5$ dimensions: the number of features with $\lVert\mathbf{W}_i\rVert>0.5$ grows from 5 (dense) to 18 (at $p=0.01$) as the features become sparser.*

**Polysemantic neurons.** If features are stored along directions that are not aligned with the neuron basis, a single neuron responds to several unrelated features: it is **polysemantic**. Real language models are full of such neurons, which is why reading a network neuron by neuron rarely works. Superposition explains why: the network has many more features than neurons. A similar picture appeared earlier for word senses: the vector of a polysemous word like *bank* is close to a weighted sum of vectors for its separate senses, which sparse coding can pull apart.[^arora2018]

**Recovering the features: sparse autoencoders.** If activations $\mathbf{h}\in\mathbb{R}^m$ are sparse combinations of many feature directions, then **dictionary learning** should find them. A sparse autoencoder maps $\mathbf{h}$ to a much wider code, $\mathbf{f}=\operatorname{ReLU}(\mathbf{W}_e\mathbf{h}+\mathbf{b}_e)\in\mathbb{R}^{F}$ with $F\gg m$, reconstructs $\hat{\mathbf{h}}=\mathbf{W}_d\mathbf{f}$, and penalizes $\lVert\mathbf{h}-\hat{\mathbf{h}}\rVert^2+\lambda\lVert\mathbf{f}\rVert_1$. The $\ell_1$ penalty (as in the lasso) keeps few features active at a time. The columns of $\mathbf{W}_d$ are candidate feature directions, and in practice many of them are far more interpretable (monosemantic) than individual neurons.[^cunningham2023]<sup>,</sup>[^bricken2023]

> [!tip] Intuition
> Exactly orthogonal directions are like seats in a room with $d$ chairs. Almost orthogonal directions are like a crowd on a dance floor: there is no fixed limit, only a trade-off between how many people fit and how often they bump into each other. Sparse features rarely dance at the same time, so the floor can hold far more of them.

## From static to contextual embeddings

A static embedding gives *bank* one vector, whether the sentence is about rivers or money. That vector is a compromise between the senses, and the same is true for every ambiguous word, which is most common words. What we want is a representation of each word **in its context**.

**ELMo** (embeddings from language models) obtained it by running a deep bidirectional LSTM language model over the sentence (one LSTM reads left to right, another right to left, each predicting the next word in its direction) and taking a learned weighted sum of its layers as the representation of each token.[^peters2018] Plugged into existing task models, ELMo vectors improved the state of the art across question answering, entailment, sentiment and named-entity recognition. The lower layers captured syntax and the higher ones word sense. The idea that won, though, was to replace the LSTMs by transformers and to **fine-tune the whole pretrained network** instead of only extracting features from it.

## Masked language modelling

BERT (bidirectional encoder representations from transformers) is a transformer encoder trained to fill in blanks.[^devlin2019] Choose a random 15% of the positions $\mathcal{M}$ of each input sequence, corrupt them, and train the network to predict the original tokens from the corrupted sequence $\tilde{\mathbf{x}}$:

$$
\mathcal{L}_{\text{MLM}}(\theta)=-\sum_{i\in\mathcal{M}}\log p_\theta\big(x_i\mid\tilde{\mathbf{x}}\big).
\tag{6}
$$

Of the chosen positions, **80%** are replaced by a special `[MASK]` token, **10%** by a random token, and **10%** are left unchanged. The reason for the mix: `[MASK]` never appears when the model is later fine-tuned, so a model trained only on masks might learn useful representations only at masked positions. Random and unchanged tokens force it to build a good representation of **every** token, since it can never be sure which ones are wrong. BERT also used a next-sentence prediction task, which later work found unnecessary.

**What MLM learns.** To fill a blank, the model must use both sides of the context ("The ___ barked at the mailman" needs the right side), resolve syntax and coreference, and know facts ("Paris is the capital of ___"). Because attention is **bidirectional** (no causal mask), each output vector is a summary of its token in the context of the entire sentence.

**How encoder-only models are used.**
- **Fine-tuning:** add a small head (a linear layer on the `[CLS]` vector for classification, or on every token for tagging) and train everything on the labelled task. A few thousand labelled examples go a long way, because the pretrained network already knows the language.
- **Embeddings:** use the output vectors directly, pooled over tokens, for retrieval, clustering or similarity search. Sentence-embedding models are usually fine-tuned further with the contrastive losses of [[Self-Supervised Representation Learning]].

**ELECTRA** made pretraining more sample-efficient.[^clark2020] MLM learns only from the 15% of masked positions. ELECTRA instead trains a small MLM **generator** to fill the blanks with plausible tokens, and the main model, the **discriminator**, to decide for **every** token whether it is original or replaced. Learning from all positions gave the same downstream accuracy as BERT-style models with a fraction of the compute.

**The catch.** An MLM does not define a joint distribution over sequences that one can sample from, since it predicts each masked token given the others, never the whole sequence. Generation needs a different factorization.

## Next-token prediction

### The chain rule as a model

Any distribution over sequences factorizes, by the chain rule of probability, into next-token conditionals:

$$
p(x_1,\dots,x_T)=\prod_{t=1}^{T}p(x_t\mid x_1,\dots,x_{t-1}).
\tag{7}
$$

This is not an assumption; it is an identity. An **autoregressive language model** parameterizes each conditional with one network, a transformer decoder whose causal mask lets position $t$ see only $x_{<t}$ (see [[Sequence Models]]). Training maximizes the log-likelihood, i.e. minimizes the cross-entropy

$$
\mathcal{L}(\theta)=-\frac1T\sum_{t=1}^{T}\log p_\theta\big(x_t\mid x_{<t}\big).
\tag{8}
$$

**Teacher forcing.** During training, the context $x_{<t}$ is always the **true** prefix from the data, never the model's own predictions. Combined with the causal mask, all $T$ predictions of a sequence are computed in one parallel forward pass. At generation time the model must instead condition on its own samples, a mismatch called **exposure bias**: one early mistake can push it into contexts it never saw during training.

This is the GPT recipe: generative pretraining of a decoder on a large corpus, then fine-tuning.[^radford2018] GPT-2 scaled it to 1.5 billion parameters and showed that the pretrained model alone could perform tasks like summarization and translation **zero-shot**, when the task was phrased as text to continue.[^radford2019] Earlier neural language models[^bengio2003] already used (8); what changed was the architecture, the data and the scale.

### Cross-entropy is compression

The loss (8) has a direct meaning. By Shannon's source coding theorem, a code that assigns $-\log_2q(x)$ bits to an outcome $x$ is (up to rounding) the shortest code for data from $q$, and **arithmetic coding** achieves this in practice, using the model's predicted distribution at each step. So:

- the loss in nats divided by $\ln2$ is the number of **bits per token** a compressor built from the model would use;
- minimizing cross-entropy is literally learning to **compress** the training text;
- the **perplexity**, $\exp(\mathcal{L})$, is the effective number of equally likely choices the model is torn between at each step. A uniform guess over GPT-2's 50,257 tokens has loss $\ln 50{,}257\approx10.8$ nats (15.6 bits) per token and perplexity 50,257.

Delétang et al. made the link concrete: large language models, used with arithmetic coding, are strong general-purpose compressors, even for images and audio they were not trained on.[^deletang2023] A model that compresses text well must capture its regularities, from spelling and grammar to facts and reasoning patterns. This is the argument for why next-token prediction learns so much.

Because the loss depends on the tokenizer (more tokens per text means each is easier to predict), models with different tokenizers are compared in **bits per byte** or bits per character instead of per token.

### Why autoregressive models can generate

Factorization (7) gives an exact sampler for free: draw $x_1$, then $x_2$ given $x_1$, and so on (**ancestral sampling**). Each step needs only one forward pass, and with a key–value cache (see [[Sequence Models]]) previous positions are never recomputed. Masked models and the contrastive objectives of the previous lecture have no such sampler.

**Decoding.** How we pick each token from $p_\theta(\cdot\mid x_{<t})$ changes the text a lot. With logits $\ell_i$:
- **Greedy** decoding takes $\arg\max_i\ell_i$. It is deterministic, and it tends to get stuck in repetitive loops.
- **Temperature** samples from $p_i\propto\exp(\ell_i/\tau)$. $\tau<1$ sharpens the distribution (safer, duller), $\tau>1$ flattens it (more diverse, more errors), and $\tau\to0$ is greedy. This is the Boltzmann distribution of [[Energy-Based Models]] with energy $-\ell_i$.
- **Top-$k$** sampling keeps only the $k$ most likely tokens and renormalizes.[^fan2018]
- **Top-$p$ (nucleus)** sampling keeps the smallest set of tokens whose total probability exceeds $p$, so the candidate set adapts: small when the model is confident, large when it is not.[^holtzman2020] This cuts off the long tail of unlikely tokens, whose combined mass is what derails pure sampling.

### Encoder–decoder versus decoder-only

An **encoder–decoder** model (the original transformer, and T5) encodes an input bidirectionally and decodes an output autoregressively with cross-attention to the encoder. T5 cast every task (translation, classification, question answering) as text-to-text and pretrained with **span corruption**: replace random spans of the input by sentinel tokens and have the decoder generate the missing spans.[^raffel2020] A **decoder-only** model handles the same tasks by putting input and output in one sequence, "input, then output". It is simpler, it uses every token of every document as a training target, and it scales well, which is why almost all large language models today are decoder-only.

## Scaling laws

How good does a language model get as it grows? Kaplan et al. trained hundreds of transformers and found that the test loss falls as a **power law** in each resource when the other two are not the bottleneck: parameters $N$, training tokens $D$, and compute $C$, over many orders of magnitude.[^kaplan2020] The training compute of a transformer is about

$$
C\approx6ND\ \text{FLOPs},
$$

two per parameter per token for the forward pass and four for the backward pass. Architecture details (depth versus width, number of heads) mattered much less than these three totals.

**The compute-optimal trade-off.** With a fixed budget $C$, should one train a bigger model on fewer tokens or a smaller one on more? Kaplan et al. concluded that the parameter count should grow fastest, $N^*\propto C^{0.73}$, and GPT-3 (175B parameters, 300B tokens) followed that advice. Hoffmann et al. revisited the question, tuning the learning-rate schedule to each run's length, which Kaplan et al. had not, and found that **parameters and tokens should grow equally**, $N^*\propto C^{0.5}$ and $D^*\propto C^{0.5}$, at roughly **20 tokens per parameter**.[^hoffmann2022] Their 70B-parameter Chinchilla, trained on 1.4T tokens with the same compute as the 280B Gopher, outperformed Gopher and GPT-3. One of their three methods fits the loss directly as

$$
L(N,D)=E+\frac{A}{N^{\alpha}}+\frac{B}{D^{\beta}},
\tag{9}
$$

with fitted constants $E=1.69$, $A=406.4$, $B=410.7$, $\alpha=0.34$ and $\beta=0.28$ (loss in nats per token).

The three terms have a clean reading: $E$ is the irreducible entropy of text, which no model removes; $A/N^\alpha$ is the cost of a finite model; $B/D^\beta$ the cost of finite data. Minimizing (9) subject to $6ND=C$ gives $N^*\propto C^{\beta/(\alpha+\beta)}=C^{0.45}$ (Figure 6).

![[Machine Learning/images/language-models-scaling.svg]]
*Figure 6. The parametric loss (9) from Hoffmann et al. (2022), plotted from their published constants. (a) At fixed compute $C=6ND$, the predicted loss is U-shaped in model size; dots mark the optimum. (b) Compute-optimal parameters and tokens from (9). Markers: GPT-3 (175B, 300B tokens), Gopher (280B, 300B) and Chinchilla (70B, 1.4T), at their training compute. Equation (9) predicts losses 2.00, 1.99 and 1.94 for them. With these constants the optimal ratio drifts from 32 to 98 tokens per parameter between $10^{19}$ and $10^{24}$ FLOPs, more than the 20 found by the paper's other two methods; a later re-analysis traced this to the fit itself and found a refitted (9) consistent with the other methods.*

Two caveats. First, Besiroglu et al. could not reproduce the fitted constants of (9) from the paper's data and found the reported confidence intervals implausibly narrow; their refit agrees with the other two methods.[^besiroglu2024] Second, "compute-optimal" counts only training. A model that will serve billions of requests is cheaper overall if it is smaller and trained on **many more** than 20 tokens per parameter, and most recent open models are deliberately "over-trained" this way.

## In-context learning

GPT-3 had 175 billion parameters and was trained only to predict the next token.[^brown2020] Its most surprising ability was not in its weights but in its **prompt**. Write a few input–output examples of a task, then a new input:

```
cheese → fromage
house → maison
dog → chien
apple →
```

and the model continues with *pomme*. Nothing was fine-tuned; no gradient was computed; the "training set" is the prompt itself. Brown et al. called this **in-context learning**: zero-shot (instruction only), one-shot and few-shot prompting. Few-shot accuracy improved steadily with model size, and the gap between few-shot and zero-shot widened, so the largest models were also the best in-context learners. ICL is an **emergent** phenomenon of scale: nobody designed it, and smaller models trained the same way showed far less of it.

Why should next-token prediction produce a learner? Four complementary explanations.

**1. Implicit Bayesian inference.** Pretraining documents are each about something: a topic, a style, a format. Call that latent concept $\theta$. To predict the next token well, the model must infer $\theta$ from the text so far. A prompt of examples is just a document whose concept is the task, so the model's prediction is a posterior predictive,

$$
p(y\mid\text{prompt})=\int p(y\mid\text{prompt},\theta)\,p(\theta\mid\text{prompt})\,d\theta,
$$

and each extra example sharpens $p(\theta\mid\text{prompt})$. Xie et al. proved that ICL emerges in this way when pretraining data is a mixture of hidden Markov models, even though the prompt format (concatenated examples) is unlike any pretraining document.[^xie2022] Consistent with this view, Min et al. found that replacing the labels in the demonstrations by random ones barely hurts on many classification tasks: the demonstrations mainly tell the model the label space, the input distribution and the format, i.e. *which* task, not how to do it.[^min2022]

**2. ICL as implicit gradient descent.** Garg et al. trained transformers from scratch on prompts $(\mathbf{x}_1,f(\mathbf{x}_1),\dots,\mathbf{x}_n,f(\mathbf{x}_n),\mathbf{x}_q)$ with a fresh random linear function $f$ in each prompt.[^garg2022] The trained models predicted $f(\mathbf{x}_q)$ about as well as least squares on the in-context examples: they had learned a learning algorithm. Why can attention implement one? Take one step of gradient descent from $\mathbf{w}=\mathbf{0}$ on the in-context loss $\frac1{2n}\sum_i(y_i-\mathbf{w}^\top\mathbf{x}_i)^2$, with step size $\eta$:

$$
\mathbf{w}_1=\frac{\eta}{n}\sum_{i=1}^{n}y_i\,\mathbf{x}_i,
\qquad
\hat y_q=\mathbf{w}_1^\top\mathbf{x}_q=\frac{1}{n}\sum_{i=1}^{n}\underbrace{\eta\,y_i}_{\text{value}}\,\underbrace{\mathbf{x}_i^\top}_{\text{key}}\underbrace{\mathbf{x}_q}_{\text{query}} .
$$

That is exactly one layer of **linear** self-attention (attention without the softmax) over tokens $(\mathbf{x}_i,y_i)$, with keys $\mathbf{x}_i$, values $\eta y_i$ and query $\mathbf{x}_q$. Von Oswald et al. showed that trained linear self-attention layers do learn such weights, so that $L$ layers run $L$ steps of (preconditioned) gradient descent on the prompt.[^vonoswald2023] Akyürek et al. found that, depending on depth and width, trained transformers match gradient descent, ridge regression or exact least squares.[^akyurek2023]

> [!example]- Code: one linear-attention layer is one gradient step (tested)
> ```python
> import numpy as np
> rng = np.random.default_rng(0)
> d, n, eta = 4, 32, 0.5
> X = rng.standard_normal((n, d)); y = X @ rng.standard_normal(d)
> x_q = rng.standard_normal(d)
>
> w1 = eta / n * X.T @ y                            # one GD step from w = 0
> pred_gd = w1 @ x_q
>
> Z = np.hstack([X, y[:, None]])                    # tokens z_i = (x_i, y_i)
> z_q = np.append(x_q, 0.0)                         # query token (x_q, 0)
> W_K = np.hstack([np.eye(d), np.zeros((d, 1))])    # key   = x_i
> W_Q = W_K.copy()                                  # query = x_q
> W_V = np.zeros((1, d + 1)); W_V[0, d] = eta       # value = eta * y_i
> keys, values, q = Z @ W_K.T, Z @ W_V.T, W_Q @ z_q
> pred_attn = (values[:, 0] * (keys @ q)).sum() / n # linear attention, no softmax
> assert np.isclose(pred_gd, pred_attn)             # both 0.100333
> ```

**3. Induction heads.** Olsson et al. looked for the mechanism inside real transformers.[^olsson2022] An **induction head** is a pair of attention heads in consecutive layers that completes patterns of the form `[A][B] … [A] → [B]`: the first head marks each token with the token before it, and the second, at the current token `A`, attends to the position just after the earlier `A` and copies what it finds there (`B`). It is the simplest possible "learn from the context" circuit. Induction heads form abruptly during training, at the same moment as a visible bump in the training loss and a sharp jump in in-context learning, measured as how much the loss on late tokens of a document improves over early tokens. The evidence is causal for small attention-only models and correlational for large ones.

**4. Scale, and whether "emergence" is real.** Wei et al. catalogued **emergent abilities**: tasks on which models perform at chance up to some scale and well above it beyond, so that the ability could not have been predicted by extrapolating smaller models.[^wei2022a] Schaeffer et al. argued that much of this sharpness comes from the **metric**.[^schaeffer2023] If per-token accuracy $q$ improves smoothly with scale, an all-or-nothing metric like exact match on an answer of $L$ tokens scores $q^L$, which looks like a sudden jump: with $L=10$, going from $q=0.8$ to $q=0.95$ moves exact match from 0.11 to 0.60. Under continuous metrics (per-token log-likelihood, edit distance), many "emergent" curves become smooth. Both observations are useful: the underlying loss improves predictably, but what users care about is often a thresholded outcome, and that can still change abruptly.

**Chain of thought.** A related prompting discovery: if the few-shot examples include worked reasoning steps before each answer, large models also write out steps, and their accuracy on arithmetic and multi-step word problems rises sharply.[^wei2022b] The gains appeared only in models of roughly 100 billion parameters or more. Generating intermediate tokens gives the model more computation per answer, and each step conditions the next.

## From pretrained models to assistants

A pretrained model continues text; it does not follow instructions. Asked "Explain the moon landing to a six-year-old", it may well continue with more questions of the same kind, as such text appears on the web. Three steps turn it into an assistant.

**1. Instruction tuning (supervised fine-tuning).** Fine-tune on many tasks phrased as instructions with good responses. FLAN fine-tuned a 137B model on over 60 tasks phrased as natural-language instructions, which greatly improved zero-shot performance on held-out task types.[^wei2021] InstructGPT fine-tuned GPT-3 on demonstrations written by human labellers.[^ouyang2022]

**2. Reward modelling.** Humans compare two responses $y_w$ (preferred) and $y_l$ to the same prompt $x$, which is easier than writing ideal answers. A reward model $r_\phi(x,y)$ is fit with the Bradley–Terry likelihood

$$
p(y_w\succ y_l\mid x)=\sigma\big(r_\phi(x,y_w)-r_\phi(x,y_l)\big),
$$

which is logistic regression on reward differences.[^christiano2017]

**3. Reinforcement learning from human feedback (RLHF).** Treat the language model as a policy $\pi_\theta(y\mid x)$ over responses: each token is an action, and the reward arrives at the end. Maximize the reward while staying close to the supervised model $\pi_{\text{ref}}$:

$$
\max_\theta\ \mathbb{E}_{x,\ y\sim\pi_\theta(\cdot\mid x)}\big[r_\phi(x,y)\big]-\beta\,D_{\mathrm{KL}}\big(\pi_\theta(\cdot\mid x)\,\big\|\,\pi_{\text{ref}}(\cdot\mid x)\big).
\tag{10}
$$

The KL penalty stops the policy from drifting into text that fools the reward model but is not actually good (reward hacking). InstructGPT optimized (10) with **PPO**, the clipped importance-weighted [[Policy Gradients|policy-gradient]] method (version 12 of the [Policy-Gradients](https://github.com/parag1604/Policy-Gradients) repository implements PPO from scratch).[^schulman2017] Its human evaluators preferred the outputs of the 1.3B-parameter InstructGPT to those of the 175B GPT-3: alignment with what people want mattered more than a hundredfold increase in size.[^ouyang2022]

**Direct preference optimization (DPO)** removes the reward model and the RL loop.[^rafailov2023] The optimal policy of (10) has a closed form, $\pi^*(y\mid x)\propto\pi_{\text{ref}}(y\mid x)\exp\big(r(x,y)/\beta\big)$, so a reward can be read off from a policy: $r(x,y)=\beta\log\frac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)}+\beta\log Z(x)$. In the Bradley–Terry likelihood only reward **differences** appear, so the intractable $\log Z(x)$ cancels, leaving a simple classification loss on the policy itself:

$$
\mathcal{L}_{\text{DPO}}(\theta)=-\log\sigma\!\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)}-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\right).
$$

> [!proof]- Why the optimal policy of (10) is a Boltzmann reweighting of the reference
> For a fixed prompt, (10) is $\sum_y\pi(y)\,r(y)-\beta\sum_y\pi(y)\log\frac{\pi(y)}{\pi_{\text{ref}}(y)}$. Define $\pi^*(y)=\pi_{\text{ref}}(y)e^{r(y)/\beta}/Z$ with $Z=\sum_y\pi_{\text{ref}}(y)e^{r(y)/\beta}$. Then $r(y)=\beta\log\frac{\pi^*(y)}{\pi_{\text{ref}}(y)}+\beta\log Z$, and substituting gives
>
> $$
> \text{objective}=\beta\log Z-\beta\,D_{\mathrm{KL}}(\pi\,\|\,\pi^*).
> $$
>
> The KL divergence is non-negative and zero only at $\pi=\pi^*$, so $\pi^*$ is the maximizer. It is the [[Energy-Based Models|Boltzmann distribution]] with energy $-r$ and temperature $\beta$, tilted by the prior $\pi_{\text{ref}}$.

## Key takeaways

- **Self-supervision in language** = predict hidden text from visible text. Neighbours (word2vec, GloVe) give static word vectors; masked tokens (BERT) give contextual encoders; next tokens (GPT) give generators.
- **Tokenization** comes first. BPE greedily merges frequent pairs; byte-level BPE never meets an unknown token. Token boundaries explain odd failures in arithmetic and spelling and unequal costs across languages.
- **Skip-gram with negative sampling** is a simplified NCE, and at its optimum it factorizes the **shifted PMI matrix**, $\mathbf{u}_c^\top\mathbf{v}_w=\operatorname{PMI}(w,c)-\log k$. **GloVe** regresses on $\log X_{ij}$ with a weighting that caps frequent pairs; ratios of co-occurrence probabilities isolate meaning.
- **Concepts are directions**, and analogies are parallelograms, because PMI differences are shared across word pairs. Test analogies honestly: exclude the query words, and don't cherry-pick.
- Only $d$ directions can be exactly orthogonal, but at least $e^{d\varepsilon^2/4}$ can be **almost** orthogonal ($|\cos|\le\varepsilon$), by concentration of random inner products and a union bound (Johnson–Lindenstrauss). With **sparse** features, networks exploit this: **superposition**, polysemantic neurons, and sparse autoencoders to untangle them.
- **Next-token cross-entropy is compression** (bits per token, perplexity). Autoregressive models sample exactly by ancestral sampling; decoding (temperature, top-$k$, top-$p$) shapes the output.
- **Scaling laws:** loss falls as a power law in parameters, data and compute; compute-optimal training grows parameters and tokens together (Chinchilla: about 20 tokens per parameter).
- **In-context learning** emerges at scale: implicit Bayesian inference over latent tasks, attention implementing gradient descent, and induction heads; whether abilities "emerge" suddenly depends partly on the metric.
- **Assistants** = pretraining + instruction tuning + preference optimization (RLHF with PPO, or DPO).

## Where this shows up next

- [[Sequence Models]]: the transformer and state-space architectures that implement every model in this lecture, where most of the knowledge sits in the MLP layers, and how mixture-of-experts layers scale them.
- [[Self-Supervised Representation Learning]]: the contrastive and non-contrastive objectives behind sentence embeddings and CLIP's text encoder.
- [[Matrix Factorization and Dimensionality Reduction]] (earlier): the SVD and Eckart–Young explain why a low-rank factorization of the PMI matrix is the best rank-$d$ summary of co-occurrence statistics.
- [[Energy-Based Models]]: temperature sampling and the DPO solution are Boltzmann distributions; contrastive learning is EBM training with the batch as negatives.
- [[Policy Gradients]]: the REINFORCE and PPO machinery used in RLHF. [[Actor-Critic Algorithms]] covers the value-function baselines that PPO uses in practice.
- **Textbooks and courses.** Jurafsky and Martin's *Speech and Language Processing* covers embeddings and language models in depth, and Prince's *Understanding Deep Learning* covers transformers and their training.

<!-- References: footnote definitions below render as the numbered "References" list at the bottom of the page -->

[^harris1954]: Harris, Z. S. (1954). *Distributional Structure*. WORD, 10(2–3), 146–162. [doi:10.1080/00437956.1954.11659520](https://doi.org/10.1080/00437956.1954.11659520)
[^sennrich2016]: Sennrich, R., Haddow, B., & Birch, A. (2016). *Neural Machine Translation of Rare Words with Subword Units*. ACL. [arXiv:1508.07909](https://arxiv.org/abs/1508.07909)
[^radford2019]: Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). *Language Models are Unsupervised Multitask Learners*. OpenAI technical report. [PDF](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
[^schuster2012]: Schuster, M., & Nakajima, K. (2012). *Japanese and Korean Voice Search*. ICASSP. [doi:10.1109/ICASSP.2012.6289079](https://doi.org/10.1109/ICASSP.2012.6289079)
[^wu2016]: Wu, Y., Schuster, M., Chen, Z., Le, Q. V., et al. (2016). *Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation*. arXiv. [arXiv:1609.08144](https://arxiv.org/abs/1609.08144)
[^kudo2018]: Kudo, T., & Richardson, J. (2018). *SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing*. EMNLP (demo). [arXiv:1808.06226](https://arxiv.org/abs/1808.06226)
[^singh2024]: Singh, A. K., & Strouse, D. J. (2024). *Tokenization counts: the impact of tokenization on arithmetic in frontier LLMs*. arXiv. [arXiv:2402.14903](https://arxiv.org/abs/2402.14903)
[^petrov2023]: Petrov, A., La Malfa, E., Torr, P. H. S., & Bibi, A. (2023). *Language Model Tokenizers Introduce Unfairness Between Languages*. NeurIPS. [arXiv:2305.15425](https://arxiv.org/abs/2305.15425)
[^mikolov2013a]: Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). *Efficient Estimation of Word Representations in Vector Space*. ICLR workshop. [arXiv:1301.3781](https://arxiv.org/abs/1301.3781)
[^mikolov2013b]: Mikolov, T., Sutskever, I., Chen, K., Corrado, G., & Dean, J. (2013). *Distributed Representations of Words and Phrases and their Compositionality*. NeurIPS. [arXiv:1310.4546](https://arxiv.org/abs/1310.4546)
[^gutmann2010]: Gutmann, M., & Hyvärinen, A. (2010). *Noise-contrastive estimation: A new estimation principle for unnormalized statistical models*. AISTATS. [PMLR](https://proceedings.mlr.press/v9/gutmann10a.html)
[^mnih2012]: Mnih, A., & Teh, Y. W. (2012). *A Fast and Simple Algorithm for Training Neural Probabilistic Language Models*. ICML. [arXiv:1206.6426](https://arxiv.org/abs/1206.6426)
[^goldberg2014]: Goldberg, Y., & Levy, O. (2014). *word2vec Explained: deriving Mikolov et al.'s negative-sampling word-embedding method*. arXiv. [arXiv:1402.3722](https://arxiv.org/abs/1402.3722)
[^levy2014]: Levy, O., & Goldberg, Y. (2014). *Neural Word Embedding as Implicit Matrix Factorization*. NeurIPS. [proceedings](https://papers.nips.cc/paper_files/paper/2014/hash/b78666971ceae55a8e87efb7cbfd9ad4-Abstract.html)
[^pennington2014]: Pennington, J., Socher, R., & Manning, C. D. (2014). *GloVe: Global Vectors for Word Representation*. EMNLP. [doi:10.3115/v1/D14-1162](https://doi.org/10.3115/v1/D14-1162)
[^mikolov2013c]: Mikolov, T., Yih, W., & Zweig, G. (2013). *Linguistic Regularities in Continuous Space Word Representations*. NAACL-HLT. [ACL Anthology](https://aclanthology.org/N13-1090/)
[^arora2016]: Arora, S., Li, Y., Liang, Y., Ma, T., & Risteski, A. (2016). *A Latent Variable Model Approach to PMI-based Word Embeddings*. TACL. [arXiv:1502.03520](https://arxiv.org/abs/1502.03520)
[^linzen2016]: Linzen, T. (2016). *Issues in evaluating semantic spaces using word analogies*. RepEval workshop. [arXiv:1606.07736](https://arxiv.org/abs/1606.07736)
[^rogers2017]: Rogers, A., Drozd, A., & Li, B. (2017). *The (too Many) Problems of Analogical Reasoning with Word Vectors*. *SEM. [doi:10.18653/v1/S17-1017](https://doi.org/10.18653/v1/S17-1017)
[^park2024]: Park, K., Choe, Y. J., & Veitch, V. (2024). *The Linear Representation Hypothesis and the Geometry of Large Language Models*. ICML. [arXiv:2311.03658](https://arxiv.org/abs/2311.03658)
[^johnson1984]: Johnson, W. B., & Lindenstrauss, J. (1984). *Extensions of Lipschitz mappings into a Hilbert space*. Contemporary Mathematics, 26, 189–206. [doi:10.1090/conm/026/737400](https://doi.org/10.1090/conm/026/737400)
[^dasgupta2003]: Dasgupta, S., & Gupta, A. (2003). *An elementary proof of a theorem of Johnson and Lindenstrauss*. Random Structures & Algorithms, 22(1), 60–65. [doi:10.1002/rsa.10073](https://doi.org/10.1002/rsa.10073)
[^elhage2022]: Elhage, N., Hume, T., Olsson, C., Schiefer, N., et al. (2022). *Toy Models of Superposition*. Transformer Circuits Thread. [arXiv:2209.10652](https://arxiv.org/abs/2209.10652)
[^arora2018]: Arora, S., Li, Y., Liang, Y., Ma, T., & Risteski, A. (2018). *Linear Algebraic Structure of Word Senses, with Applications to Polysemy*. TACL. [arXiv:1601.03764](https://arxiv.org/abs/1601.03764)
[^cunningham2023]: Cunningham, H., Ewart, A., Riggs, L., Huben, R., & Sharkey, L. (2024). *Sparse Autoencoders Find Highly Interpretable Features in Language Models*. ICLR. [arXiv:2309.08600](https://arxiv.org/abs/2309.08600)
[^bricken2023]: Bricken, T., Templeton, A., Batson, J., Chen, B., Jermyn, A., et al. (2023). *Towards Monosemanticity: Decomposing Language Models With Dictionary Learning*. Transformer Circuits Thread. [link](https://transformer-circuits.pub/2023/monosemantic-features)
[^peters2018]: Peters, M. E., Neumann, M., Iyyer, M., Gardner, M., Clark, C., Lee, K., & Zettlemoyer, L. (2018). *Deep contextualized word representations*. NAACL. [arXiv:1802.05365](https://arxiv.org/abs/1802.05365)
[^devlin2019]: Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL. [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)
[^clark2020]: Clark, K., Luong, M.-T., Le, Q. V., & Manning, C. D. (2020). *ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators*. ICLR. [arXiv:2003.10555](https://arxiv.org/abs/2003.10555)
[^radford2018]: Radford, A., Narasimhan, K., Salimans, T., & Sutskever, I. (2018). *Improving Language Understanding by Generative Pre-Training*. OpenAI technical report. [PDF](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
[^bengio2003]: Bengio, Y., Ducharme, R., Vincent, P., & Jauvin, C. (2003). *A Neural Probabilistic Language Model*. JMLR, 3, 1137–1155. [JMLR](https://www.jmlr.org/papers/v3/bengio03a.html)
[^deletang2023]: Delétang, G., Ruoss, A., Duquenne, P.-A., Catt, E., et al. (2024). *Language Modeling Is Compression*. ICLR. [arXiv:2309.10668](https://arxiv.org/abs/2309.10668)
[^fan2018]: Fan, A., Lewis, M., & Dauphin, Y. (2018). *Hierarchical Neural Story Generation*. ACL. [arXiv:1805.04833](https://arxiv.org/abs/1805.04833)
[^holtzman2020]: Holtzman, A., Buys, J., Du, L., Forbes, M., & Choi, Y. (2020). *The Curious Case of Neural Text Degeneration*. ICLR. [arXiv:1904.09751](https://arxiv.org/abs/1904.09751)
[^raffel2020]: Raffel, C., Shazeer, N., Roberts, A., Lee, K., et al. (2020). *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*. JMLR. [arXiv:1910.10683](https://arxiv.org/abs/1910.10683)
[^kaplan2020]: Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., et al. (2020). *Scaling Laws for Neural Language Models*. arXiv. [arXiv:2001.08361](https://arxiv.org/abs/2001.08361)
[^hoffmann2022]: Hoffmann, J., Borgeaud, S., Mensch, A., Buchatskaya, E., et al. (2022). *Training Compute-Optimal Large Language Models*. NeurIPS. [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)
[^besiroglu2024]: Besiroglu, T., Erdil, E., Barnett, M., & You, J. (2024). *Chinchilla Scaling: A replication attempt*. arXiv. [arXiv:2404.10102](https://arxiv.org/abs/2404.10102)
[^brown2020]: Brown, T. B., Mann, B., Ryder, N., Subbiah, M., et al. (2020). *Language Models are Few-Shot Learners*. NeurIPS. [arXiv:2005.14165](https://arxiv.org/abs/2005.14165)
[^xie2022]: Xie, S. M., Raghunathan, A., Liang, P., & Ma, T. (2022). *An Explanation of In-context Learning as Implicit Bayesian Inference*. ICLR. [arXiv:2111.02080](https://arxiv.org/abs/2111.02080)
[^min2022]: Min, S., Lyu, X., Holtzman, A., Artetxe, M., Lewis, M., Hajishirzi, H., & Zettlemoyer, L. (2022). *Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?* EMNLP. [arXiv:2202.12837](https://arxiv.org/abs/2202.12837)
[^garg2022]: Garg, S., Tsipras, D., Liang, P., & Valiant, G. (2022). *What Can Transformers Learn In-Context? A Case Study of Simple Function Classes*. NeurIPS. [arXiv:2208.01066](https://arxiv.org/abs/2208.01066)
[^vonoswald2023]: von Oswald, J., Niklasson, E., Randazzo, E., Sacramento, J., Mordvintsev, A., Zhmoginov, A., & Vladymyrov, M. (2023). *Transformers learn in-context by gradient descent*. ICML. [arXiv:2212.07677](https://arxiv.org/abs/2212.07677)
[^akyurek2023]: Akyürek, E., Schuurmans, D., Andreas, J., Ma, T., & Zhou, D. (2023). *What learning algorithm is in-context learning? Investigations with linear models*. ICLR. [arXiv:2211.15661](https://arxiv.org/abs/2211.15661)
[^olsson2022]: Olsson, C., Elhage, N., Nanda, N., Joseph, N., et al. (2022). *In-context Learning and Induction Heads*. Transformer Circuits Thread. [arXiv:2209.11895](https://arxiv.org/abs/2209.11895)
[^wei2022a]: Wei, J., Tay, Y., Bommasani, R., Raffel, C., et al. (2022). *Emergent Abilities of Large Language Models*. TMLR. [arXiv:2206.07682](https://arxiv.org/abs/2206.07682)
[^schaeffer2023]: Schaeffer, R., Miranda, B., & Koyejo, S. (2023). *Are Emergent Abilities of Large Language Models a Mirage?* NeurIPS. [arXiv:2304.15004](https://arxiv.org/abs/2304.15004)
[^wei2022b]: Wei, J., Wang, X., Schuurmans, D., Bosma, M., et al. (2022). *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*. NeurIPS. [arXiv:2201.11903](https://arxiv.org/abs/2201.11903)
[^wei2021]: Wei, J., Bosma, M., Zhao, V. Y., Guu, K., et al. (2022). *Finetuned Language Models Are Zero-Shot Learners*. ICLR. [arXiv:2109.01652](https://arxiv.org/abs/2109.01652)
[^ouyang2022]: Ouyang, L., Wu, J., Jiang, X., Almeida, D., et al. (2022). *Training language models to follow instructions with human feedback*. NeurIPS. [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)
[^christiano2017]: Christiano, P., Leike, J., Brown, T. B., Martic, M., Legg, S., & Amodei, D. (2017). *Deep reinforcement learning from human preferences*. NeurIPS. [arXiv:1706.03741](https://arxiv.org/abs/1706.03741)
[^schulman2017]: Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). *Proximal Policy Optimization Algorithms*. arXiv. [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
[^rafailov2023]: Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., & Finn, C. (2023). *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*. NeurIPS. [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)
