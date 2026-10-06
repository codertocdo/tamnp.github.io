---
title: "Feature Engineering"
date: 2026-10-01 09:00:00 +0700
categories: [Artificial Intelligence, Machine Learning, Machine Learning Pipeline, Data Preprocessing]
tags: [feature-engineering,feature-scaling,bag-of-words,dimensionality-reduction,random-projection,transfer-learning,scikit-learn,cpp]
math: true
published: false
---

Here is an experiment that surprised me when I first ran it. Take a small, clean dataset of 178 wines, 12 chemical measurements each. Train a k-nearest-neighbours classifier on it. Accuracy: **68.6%**. Now change nothing about the model, the data, or the hyperparameters — just rescale each column so they all live on comparable ranges. Accuracy: **97.2%**.

No new data. No deeper network. No hyperparameter search. Just a different *representation* of exactly the same numbers.

That jump is the whole story of **feature engineering**: the craft of deciding *what numbers a model actually gets to see*. It is the least glamorous part of machine learning and, very often, the part that decides whether a project works at all. A line widely attributed to Andrew Ng puts it bluntly: *"Applied machine learning is basically feature engineering."*

In this post I document what I learned about the topic from my course, then go further: I re-derive the math, run every claim as real code (C++ from scratch and Python with scikit-learn), and flag the places where the sources are imprecise or wrong. By the end you should be able to *see* feature engineering — as geometry — rather than just recite a list of techniques.

## 1. What Is a Feature, Mathematically?

A **feature** is a single measurable property of a data point: a pixel intensity, a word count, a patient's age, the proline concentration of a wine. Stack $D$ features of one data point together and you get a **feature vector**:

$$\mathbf{x} = \begin{bmatrix} x_1 & x_2 & \cdots & x_D \end{bmatrix}^\top \in \mathbb{R}^D$$

The space $\mathbb{R}^D$ is the **feature space**, and each data point is literally a *point* in it. Stack $N$ feature vectors as rows and you get the **design matrix** (also called the data matrix):

$$X = \begin{bmatrix} \mathbf{x}^{(1)\top} \\ \mathbf{x}^{(2)\top} \\ \vdots \\ \mathbf{x}^{(N)\top} \end{bmatrix} \in \mathbb{R}^{N \times D}$$

Two constraints make this matrix possible, and both are the reason feature engineering exists:
 
1. **Fixed length.** Every row must have the same $D$. Matrix products like $X\mathbf{w}$ are undefined otherwise. But raw data almost never arrives that way: a tweet has 12 words, a novel has 120,000; one photo is 640×427 pixels, another is 4000×3000.
2. **Meaningful geometry.** Most learning algorithms reason with distances, dot products, or hyperplanes in $\mathbb{R}^D$. If the coordinates are on wildly different scales, or the relevant information is hidden in a nonlinear combination, the geometry lies to the model.

So I find this the most useful working definition:

> **Feature engineering** is the design of a function $\phi$ that maps raw inputs of arbitrary shape into fixed-length vectors $\phi(\text{raw}) \in \mathbb{R}^D$ whose *geometry* reflects what matters for the task.

## 2. The Five Jobs of Feature Engineering

![Feature Engineering Mindmap](/assets/img/AI/ML/Pipeline/Preprocessing/3_feature_engineering/feature_engineering_mindmap.png)

| Job                | Question it answers                     | Changes $D$?      | Typical example                            |
| ------------------ | --------------------------------------- | ----------------- | ------------------------------------------ |
| **Creation**       | What useful quantity is *missing*?      | ↑ adds columns    | `price_per_m2 = price / area`              |
| **Transformation** | Is each column in a usable *form*?      | same or ↑         | one-hot encoding a `color` column          |
| **Extraction**     | Can I *summarise* raw data compactly?   | usually ↓         | 273,280 pixels → 10-bin texture histogram  |
| **Selection**      | Which existing columns actually *help*? | ↓ removes columns | drop features with zero mutual information |
| **Scaling**        | Are columns on *comparable* ranges?     | same              | z-score every column                       |


## 3. The Big Picture: Where Feature Engineering Sits

Here is the standard two-phase pipeline:

![Machine Learning Phases](/assets/img/AI/ML/Pipeline/Preprocessing/3_feature_engineering/machine_learning_phase.png)

Three things in this diagram are easy to miss and important:

- **The extractor can use labels.** For a shape classifier, a good $\phi$ keeps the number of edges and throws away colour; for a colour classifier, the opposite. Same raw data, different optimal features - the *task* defines what "informative" means.
-  

### Failure 1: One-Hot Encoding

The simplest possible scheme. Take your vocabulary, assign each word an index, and represent word $i$ as a vector that is $1$ at position $i$ and $0$ everywhere else.

With a four-word vocabulary `["cat", "dog", "kitten", "airplane"]`:

| Word     | Vector         |
| -------- | -------------- |
| cat      | `[1, 0, 0, 0]` |
| dog      | `[0, 1, 0, 0]` |
| kitten   | `[0, 0, 1, 0]` |
| airplane | `[0, 0, 0, 1]` |

Clean, unambiguous, easy to build, and completely useless for meaning. Here's what happens when we compare any two of them:

```python
import numpy as np

vocab = ["cat", "dog", "kitten", "airplane"]
I = np.eye(len(vocab))  # each row is a one-hot vector

def cos(a, b):
  return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

for i in range(len(vocab)):
  for j in range(i + 1, len(vocab)):
    print(f"cos({vocab[i]:8s}, {vocab[j]:8s}) = {cos(I[i], I[j]):.1f}")
```
Output:
```text
cos(cat     , dog     ) = 0.0
cos(cat     , kitten  ) = 0.0
cos(cat     , airplane) = 0.0
cos(dog     , kitten  ) = 0.0
cos(dog     , airplane) = 0.0
cos(kitten  , airplane) = 0.0
```
Every pair scores **exactly zero**. In geometric terms, every one-hot vector is orthogonal (perpendicular) to every other one. The representation asserts that "cat" is precisely as unrelated to "kitten" as it is to "airplane". All semantic structure has been destroyed by the encoding itself.

There's a second, more practical problem - **dimensionality**. A realistic vocabulary is around 50,000 words, so each word needs a 50,000-dimensional vector:

```text
one-hot : 50000 x 50000 floats = 10.0 GB
dense   : 50000 x 1536  floats = 307.2 MB
```
10 GB to store a table that is 99.99% zeros, and which still tells you nothing about meaning. This is the classic **curse of dimensionality** in its most wasteful form: the vectors are enormous *and* sparse *and* uninformative.

### Failure 2: Raw Pixels

The same problem appears in vision. A $224 \times 224$ RGB image is just $224 \times 224 \times 3 = 150{,}528$ numbers. Why not compare images by comparing those numbers directly?

Because raw pixels values encode **appearance**, not **content**. Consider:

- A photo of a black cat and a photo of a white cat: *semantically almost identical*, numerically almost opposite (every pixel inverted).
- A photo of a cat, and the same photo shifted three pixels to the right: *semantically identical*, but numerically every single value has changed.
- A photo of a cat on a grass, and a photo of a dog on grass: *semantically different*, but numerically very similar because most of the frame is same shade of green.

Raw pixels are sensitive to brightness, translation, rotation, and background - and blind to the thing we actually care about. Same disease as one-hot encoding: the representation measures the wrong property.

### The Diagnosis

Both failures share one root cause:

> **The geometry of the representation doesn't match the semantics of the data.**

We need a representation where *geometric closeness means semantic closeness*. That's precisely what an embedding is.

## 2. Vector Embedding

A **vector embedding** is a mapping from some object - a word, a sentence, an image, a user, a product - to a point in a continuous, relatively low-dimensional space, chosen so that **similar objects land near each other**.

More formally, an embedding is a learned function

$$f: \mathcal{X} \rightarrow \mathbb{R}^{d}$$

that maps objects from a domain $\mathcal{X}$ (all English sentence, all images) into $d$-dimensional real space, where $d$ is typically a few hundred to a few thousand. The defining property is that $f$ is trained so that:

$$\text{semantic\_similarity}(x_1, x_2) \;\approx\; \text{geometric\_similarity}\big(f(x_1), f(x_2)\big)$$

Three properties distinguish embeddings from the failed representations above:

- **Dense, not sparse.** Nearly every coordinate carries information, instead of a single 1 in a sea of zeros.
- **Low-dimensional.** Hundreds or thousands of dimensions instead of ten of thousands. As a reference point, `text-embedding-3-small` outputs 1,536 dimensions; many open models like BGE and E5 use 768; Google's `gemini-embedding-001` goes up to 3,072.
- **Learned, not assigned.** Nobody hand-writes these numbers. They emerge from training on huge amounts of data.

### Where Do the Numbers Come From?

This is the part that usually gets hand-waved, so let's be concrete. Almost all embedding methods rest on the **distributional hypothesis**, summarized by linguist J.R. Firth in 1957:

> *You shall know a word by the company it keeps.*

The claim is that a word's meaning can be recovered from the statistics of the words that appear near it. "Cat" and "kitten" appear alongside *purr*, *whiskers*, *milk*, *fur*; "airplane" appears alongside *runway*, *flight*, *airport*. Track those co-occurrence patterns across billions of words, compress them, and we get vectors whose geometry mirrrors meaning.

Modern models (BERT, CLIP, the Qwen3-Embedding family) replace the raw counting with contrastive training on neural networks, but the underlying bet is the same: **context determines meaning**. We'll actually implement the counting version in Part 6, and it works well enough to be convincing.

### Comparison: One-Hot vs. Raw Pixels vs. Embeddings

| Property              | One-Hot                   | Raw Pixels                 | Embedding                             |
| --------------------- | ------------------------- | -------------------------- | ------------------------------------- |
| Dimensionality        | Vocabulary size (~50k)    | Width × Height × 3 (~150k) | 256 – 3072                            |
| Density               | Sparse (one non-zero)     | Dense                      | Dense                                 |
| Encodes similarity?   | No — all pairs orthogonal | Accidentally, and wrongly  | **Yes, by construction**              |
| Handles unseen items? | No                        | N/A                        | Yes (via subwords / learned features) |
| Source of values      | Assigned by index         | Measured by sensor         | **Learned from data**                 |
| Storage (50k items)   | ~10 GB                    | —                          | ~300 MB                               |

## 3. Geometric Intuition

Here's where it clicks. Forget "list of numbers" and think **arrows in space**.

### Starts With Two Dimensions

Suppose we build a toy embedding with exactly two dimensions, and suppose - purely for intuition - that those dimensions happen to mean something interpretable:

- **Axis 1 (x):** how *animal-like* the concept is
- **Axis 2 (y):** how *mechanical* the concept is

Plot a few words:

| Word       | x (animal) | y (mechanical) |
| ---------- | ---------- | -------------- |
| cat        | 0.9        | 0.1            |
| kitten     | 0.95       | 0.05           |
| dog        | 0.85       | 0.15           |
| airplane   | 0.05       | 0.95           |
| helicopter | 0.1        | 0.9            |

You can see that "cat", "kitten", and "dog" form a tight cluster hugging the x-axis. "Airplane" and "helicopter" form their own cluster hugging the y-axis. The animal arrows all point in roughly the same direction; the machine arrows point in a very different one. **The angle between arrows encode relatedness**.

### Now Scale Up

Real embeddings do exactly this, except with 768 or 1,536 axes instead of 2 - and crucially, **the axes have no individual human-readable meaning**. There is no "animal-ness" dimension in a real model. Meaning is distributed across all coordinates at once, and only *directions* in the aggregate carry semantics.

This is the single hardest thing to internalize, so let me put it plainly: you cannot visualize 1,536 dimensions, and you should stop trying. What you *can* do is trust that the operatons are identical. Dot products, norms, and angles are defined the same way in $\mathbb{R}^{1536}$ as in $\mathbb{R}^{2}$. The algebra scales even though your visual cortex doesn't.

### But High Dimensions Are Genuinely Weird

Here's something that surprised me, and it's worth knowing because it explains *why* high-dimensional embeddings work so well.

In 2D, if you pick two random directions, the angle between them is uniformly distributed - 90° is nothing special. In high dimensions, that changes completely. I measured this over 20,000 random pairs per dimension:

```python
import numpy as np
rng = np.random.default_rng(42)

def random_pair_stats(dim, trials = 20000):
  a = rng.normal(size = (trials, dim))
  b = rng.normal(size = (trials, dim))
  
  a /= np.linalg.norm(a, axis = 1, keepdims = True)
  b /= np.linalg.norm(b, axis = 1, keepdims = True)

  cos = np.sum(a * b, axis = 1)

  return np.abs(cos).mean(), np.degrees(np.arccos(cos)).mean()

for d in [2, 3, 10, 100, 768, 1536, 3072]:
  m, arg = random_pair_stats(d)
  print(f"dim = {d:>5}  mean|cos| = {m:.4f}  mean angle = {arg:.2f}°")
```
Result:

| Dimensions | Mean \|cos θ\| | Mean angle |
| ---------- | -------------- | ---------- |
| 2          | 0.6339         | 90.68°     |
| 3          | 0.4984         | 89.86°     |
| 10         | 0.2601         | 90.13°     |
| 100        | 0.0800         | 89.99°     |
| 768        | 0.0286         | 90.01°     |
| 1536       | 0.0206         | 89.99°     |
| 3072       | 0.0144         | 89.99°     |

As dimension grows, random vectors become **almost perfectly orthogonal**. The mean absolute cosine shrinks toward zero, following $\mathbb{E}[|\cos\theta|] \approx \sqrt{2/(\pi d)}$. (Check it: for $d = 768$, that formula predicts $0.0288$ - the measurement says $0.0286$.)

This phenomenon is called **concentration of measure**, and it is a *gift*, not a curse. It means high-dimensional space has room for an enormous number of nearly-independent concepts. In 3D you can fit only 3 mutually perpendicular directions; in 768D you can pack in vastly more *nearly-perpendicular* ones. Unrelatedness is the default, so when two embeddings *do* score 0.8 similarity, that's a strong, meaningful signal rather than a 

It also gives you a practical calibration rule: **a cosine score of 0.1 between two 1,536-dimensional embeddings is not "slightly similar" - it is statistical noise**.

## 4. Cosine Similarity

**Cosine similarity** between two non-zero vectors $\mathbf{a}, \mathbf{b} \in \mathbb{R}^{n}$ is the cosine of the angle $\theta$ between them:

$$\cos(\theta) = \frac{\mathbf{a} \cdot \mathbf{b}}{\lVert \mathbf{a} \rVert \, \lVert \mathbf{b} \rVert} = \frac{\sum_{i=1}^{n} a_i b_i}{\sqrt{\sum_{i=1}^{n} a_i^2} \; \sqrt{\sum_{i=1}^{n} b_i^2}}$$

Read it as three steps:

1. **Numerator - the dot product** $\mathbf{a} \cdot \mathbf{b}$ measures raw alignment. It grows when the vectors agree in sign and magnitude across coordinates.
2. **Denominator - the two norms** $\lVert \mathbf{a} \rVert$ and $\lVert \mathbf{b} \rVert$ measures the lengths.
3. **Dividing** cancels out length entirely, leaving *pure direction*.

The output is bounded in $[-1, 1]$:

| Score | Angle | Interpretation                            |
| ----- | ----- | ----------------------------------------- |
| $+1$  | 0°    | Same direction — maximally similar        |
| $0$   | 90°   | Orthogonal — unrelated                    |
| $-1$  | 180°  | Opposite direction — maximally dissimilar |

A practical note: although the theoretical range is $[-1, 1]$, most modern text embedding models produce **non-negative** scores in practice, so real-world similarities cluster in $[0, 1]$. Don't be surprised if you never see a negative value.

You'll also encounter **cosine distance**, define as $1 - \cos(\theta)$, which converts the similarity into a proper distance-like quantity where smaller means closer. Vector databases often expose this instead.

### Implementation in C++ 

Let's build it from the arithmetic up, so nothing is hidden behind an abstraction:

```cpp
#include <cstdio>
#include <cmath>
#include <cstdlib>

// Dot product: sum of element-wise products
double dot(const double* a, const double* b, int n) {
  double acc = 0.0;
  for (int i = 0; i < n; i++) {
    acc += a[i] * b[i];
  }
  return acc;
}

// L2 norm (magnitude): sqrt of the vector dotted with itself
double norm(const double* v, int n) {
  return std::sqrt(dot(v, v, n));
}

// cos(theta) = (a . b) / (||a|| * ||b||)
double consine_similarity(const double* a, const double* b, int n) {
  double na = norm(a, n);
  double nb = norm(b, n);
  if (na == 0.0 || nb == 0.0) return 0.0;
  return dot(a, b, n) / (na * nb);
}

//Euclidean (L2) distance, for comparison
double euclidean_distance(const double* a, const double* b, int n) {
  double acc = 0.0;
  for (int i = 0; i < n; i++) {
    double d = a[i] - b[i];
    acc += d * d;
  }
  return std::sqrt(acc);
}

int main() {
  const int DIM = 3;

  // Toy "document": word counts of [ai, model, cooking]
  // short_note and long_essay discuss the SAME topic
  // but the essay is 10x longer
  double* short_note = (double*)std::malloc(DIM * sizeof(double));
  double* long_essay = (double*)std::malloc(DIM * sizeof(double));
  double* recipe     = (double*)std::malloc(DIM * sizeof(double));

  short_note[0] = 2;  short_node[1] = 3;  short_node[2] = 0;
  long_essay[0] = 20; long_essay[1] = 30; long_essay[2] = 0;
  recipe[0]     = 0;  recipe[1]     = 1;  recipe[2] = 9;  


}
```

### Cosine Similarity vs. Euclidean Distance

**Euclidean distance** (the L2 norm of the difference) is the straight-line distance between two points:

$$d(\mathbf{a}, \mathbf{b}) = \lVert \mathbf{a} - \mathbf{b} \rVert = \sqrt{\sum_{i=1}^{n} (a_i - b_i)^2}$$

Now re-read the C++ output. Two documents on the *identical* topic, differing only in length, scored:

- **Cosine: 1.0000** — perfectly similar. Correct.
- **Euclidean: 32.45** — extremely far apart.

Meanwhile the totally unrelated recipe scored a Euclidean distance of only **9.43** - over three times *closer*. Ranked by Euclidean distance, this toy search engine would confidently return the cooking recipe ahead of the article about the same subject.

That's the core distinction:

> **Euclidean distance measures how far apart. Cosine similarity measures how aligned.**

And for semantic tasks, **direction carries the meaning while magnitude carries mostly irrelevant nuisance information** - document length, word frequency, image brigtness, how chatty a user is. Cosine similarity divides that nuisance out for free.

|                         | Cosine Similarity                           | Euclidean Distance                                                          |
| ----------------------- | ------------------------------------------- | --------------------------------------------------------------------------- |
| Measures                | Angle / orientation                         | Straight-line separation                                                    |
| Range                   | $[-1, 1]$                                   | $[0, \infty)$                                                               |
| Sensitive to magnitude? | **No**                                      | **Yes**                                                                     |
| Good for                | Text, semantic search, RAG, recommendations | Physical coordinates, clustering raw features, images with meaningful scale |
| Cost                    | Dot product + 2 norms + division            | Subtract, square, sum, sqrt                                                 |

However, here's a fact that ties the whole section together. If your vectors are **L2-normalized** (rescaled to unit length, $\lVert \mathbf{v} \rVert = 1$), then the two metrics are algebraically linked:

$$d(\hat{\mathbf{a}}, \hat{\mathbf{b}})^2 = \lVert \hat{\mathbf{a}} \rVert^2 + \lVert \hat{\mathbf{b}} \rVert^2 - 2(\hat{\mathbf{a}} \cdot \hat{\mathbf{b}}) = 2\big(1 - \cos\theta\big)$$

Squared Euclidean distance is just a decreasing linear function of cosine similarity. **They produce identical rankings**. And since $\lVert \hat{\mathbf{a}} \rVert = \lVert \hat{\mathbf{b}} \rVert = 1$, the cosine formula collapses to a bare dot product:

$$\cos(\theta) = \hat{\mathbf{a}} \cdot \hat{\mathbf{b}}$$

This explains a production practice you'll see everywhere: **normalize once at index time, then use the dot product forever**. You get cosine semantics at the cost of a single multiply-accumulate loop, with no square roots and no divisions in the hot path - which maps beautifully onto SIMD and GPU hardware. It's also part of why the Transformer's attention mechanism uses a scaled dot product rather than a full cosine.

So when people argue about cosine vs. Euclidean vs. dot product for normalized embeddings, they're often arguing about nothing. **The real rule: use whatever metric your embedding model was trained with**. That information is in the model card. Choosing differently from the training objective is the actual mistake.

## 5. Building Embeddings From Scratch

Time to make this real. Let's derive embeddings ourselves from a tiny corpus, using nothing but NumPy.

The pipleline follows the classic count-based recipe: **co-occurrence counts → PPMI weighting → SVD compression → cosine similarity**.

```python
import numpy as np

corpus = """
the cat sat on the mat and the cat drank milk
the dog sat on the rug and the dog drank water
the cat chased the dog around the garden
a kitten is a young cat and a puppy is a young dog
the cat purred softly while the dog barked loudly
milk and water are drinks that the cat and the dog like
the king ruled the kingdom and the queen ruled beside him
the king wore a crown and the queen wore a crown too
a man became king and a woman became queen
the kingdom loved the king and the kingdom loved the queen
""".split()

vocab = sorted(set(corpus))
idx = {w: i for i, w in enumerate(vocab)}
V = len(vocab)
WINDOW = 2

# 1. Co-occurence matrix: how often does word i appear near word j
C = np.zeros((V, V))
for i, w in enumerate(corpus):
  for j in range(max(0, 1 - WINDOW), min(len(corpus), i + WINDOW + 1)):
    if i != j:
      C[idx[w], idx[corpus[j]]] += 1

# 2. Positive Pointwise Mutual Information
#    Raw counts over-reward common words like "the"; PMI asks instead: 
#    do these two words co-occur MORE than chance would predict?
total = C.sum()
p_wc = C / total
p_w = p_wc.sum(axis = 1, keepdims = True)
p_c = p_wc.sum(axis = 0, keepdims = True)
with np.errstate(divide = "ignore", invalid = "ignore"):
  pmi = np.log(p_wc / (p_w * p_c))
ppmi = np.nan_to_num(np.maximum(pmi, 0), nan = 0.0, posinf = 0.0, neginf = 0.0)

# 3. Compress V dimensions down to 8 via truncated SVD.
#    This is the step that turns a sparse table into a dense embedding.
U, S, Vt = np.linalg.svd(ppmi)
DIM = 8
E = U[:, :DIM] * S[:DIM]
E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)   # L2-normalize

# 4. With normalized rows, cosine similarity IS the dot product.
def cos(a, b):
  return float(np.dot(a, b))

def nearest(word, k=4):
  scores = E @ E[idx[word]]
  order = np.argsort(-scores)
  return [(vocab[i], scores[i]) for i in order if vocab[i] != word][:k]

print("Vocab size:", V, "| embedding dim:", DIM, "\n")
for w in ["cat", "king", "milk"]:
  print(f"nearest to '{w}':", ", ".join(f"{n} ({s:.3f})" for n, s in nearest(w)))

print("\nPairwise cosine similarity:")
for a, b in [("cat", "dog"), ("king", "queen"), ("cat", "king"), ("milk", "water")]:
  print(f"  cos({a:5s}, {b:6s}) = {cos(E[idx[a]], E[idx[b]]):+.4f}")
```

Actual output:
```text
Vocab size: 41 | embedding dim: 8
 
nearest to 'cat': mat (0.700), rug (0.700), while (0.689), sat (0.683)
nearest to 'king': queen (0.935), woman (0.909), became (0.853), loved (0.848)
nearest to 'milk': loudly (0.946), drank (0.926), barked (0.916), dog (0.854)
 
Pairwise cosine similarity:
  cos(cat  , dog   ) = +0.6599
  cos(king , queen ) = +0.9352
  cos(cat  , king  ) = +0.1754
  cos(milk , water ) = +0.4017
```

Look at what emerged from ten lines of text and some matrix algebra:

- **`king` and `queen` score 0.935** - the model discovered they're related purely from the company they keep.
- **`cat` and `dog` score 0.660** - related, but less tightly than the royal pair, which is right: in this corpus they're often *contrasted* with each other.
- **`cat` and `king` score 0.175** - essentially unrelated, exactly as they should be. Two different semantic regions of the space.

And now the honest part, which I think the most instructive thing here: **it also produces nonsenes**. The nearest neighbour of `milk` is `loudly`, which is meaningless. `cat` is closet to `mat` and `rug`, which is rhyme-and-proximity, not semantics.

That's not a bug in the code - it's a demonstration of the method's actual requirement. The distributional hypothesis needs *volume*. Ten sentences produce coincidences; ten billion produce meaning. When you read that models are trained on trillions of tokens, this is why. Scale isn't a luxury here, it's the load-bearing ingredient.

### The Full Pipeline

Here's how these pieces assemble into a real system:

![Embedding Pipeline](/assets/img/AI/ML/Fundamentals/1-vector-cosine/pipeline.png)

## 6. Where This Actually Gets Used

The reason this one idea deserves the attention is its absurd breadth of application.

### Semantic Search and RAG

Traditional keyword search fails when the user's words differ from the document's words. Search "how do I fix my laptop's battery draining fast" and a keyword engine looks for those literal tokens; it will miss a perfect article titled "Improving power efficiency on portable computers".

Embedding-based search doesn't care about the words. It embeds the query, embeds every document, and ranks by cosine similarity. Different vocabulary, same direction in vector space.

This is the retrieval half of **Retrieval-Augmented Generation (RAG)**, the standard architecture for grounding an LLM in your own documents: chunk the documents, embed the chunks, store them in a vector database, then at query time retrieve the top-$k$ chunks by cosine similarity and paste them into the model's context window. Every RAG system you've used runs this loop.

### Recommendation System

Represent both users and items as vectors in a shared space. A user's vector is often just the (normalized) average of the items they've engaged with. Recommendation then becomes: find items whose vectors have high cosine similarity to the user's vector. Same math, different nouns.

### Cross-Modal Search (Text ↔ Image)

This is the genuinely magical one. **CLIP** - style models train an image encoder and a text encoder *jointly*, so that a photo of a dog and the caption "a photo of a dog" land at nearly the same point in one shared space.

Once that shared space exists, you can search images with text - embed the sentence, embed every image, rank by cosine similarity. That's how Google Photos finds "beach sunset" without anyone tagging your photos, and it's the backbone of modern video retrieval systems.

### Face Recognition and Verification

**FaceNet** and its successors embed faces into a compact space where all photos of the same person cluster tightly. Verification becomes a threshold test: compute cosine similarity between two face embeddings, and if it exceeds some threshold, it's the same person. Note the elegance — the system can recognize people it was never trained on, because it learned a *metric* rather than a set of classes.

### Clustering, Deduplication, and Classification
 
- **Clustering:** group articles by topic without labels, using cosine distance as the metric for k-means.
- **Deduplication:** near-identical documents have cosine similarity near 1.0, catching paraphrases that exact hashing misses.
- **Zero-shot classification:** embed your candidate labels as text, embed the input, assign whichever label scores highest. No training required.



