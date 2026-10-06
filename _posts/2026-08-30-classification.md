---
title: "Classification & Regression In Machine Learning"
date: 2026-08-30 09:00:00 +0700
categories: [Artificial Intelligence, Machine Learning, Supervised Learning]
tags: [supervised-learning, classification, regression, machine-learning, model-evaluation, scikit-learn]
math: true
published: false
---

Here is a deceptively simple pair of questions about the same student:

1. *What score will this student get on the final exam?*
2. *Will this student pass?*

Same person. Same data. Same underlying reality. And yet these two questions send you down two completely different branches of machine learning, with different models, different loss functions, different geometry, and different ways of being wrong.

The first is **regression**. The second is **classification**. Together they cover the overwhelming majority of supervised learning you'll ever build - and if you look under the hood of a modern object detector or a large language model, you'll find both of them hiding in there.

## 1. The Common Ground: Supervised Learning as Function Approximation

Before we split, let's be precise about what the two tasks share. Both classification and regression are forms of **supervised learning**, which means both start with the same three ingredients.

**Ingredient 1 - Labelled data**. A dataset of $n$ examples:

$$D=\{(\mathbf{x}_1,y_1),(\mathbf{x}_2,y_2),\dots,(\mathbf{x}_n,y_n)\}$$

where each $\mathbf{x}_i\in\mathcal{X}\subseteq\mathbb{R}^d$ is a **feature vector** (the $d$ measurable things we know: hours studied, pixel intensities, word counts) and each $y_i\in\mathcal{Y}$ is the **target** (the thing we want to predict). The word *supervised* means every input comes pre-paired with its correct answer — a supervisor has already done the work.

**Ingredient 2 - Hypothesis space**. We assume some unknown true fuction $f:\mathcal{X}\to\mathcal{Y}$ generated this data, and we search for an approximation $\hat{f}$ inside a restricted family $\mathcal{H}$ (all straight lines, all decision trees of depth 5, all neural networks with a given architecture). We can never search "all possible functions" — that way lies overfitting and computational death.

**Ingredient 3 - Loss function**. A function $L(\hat{y},y)$ that assigns a *cost* to predicting $\hat{y}$ when the truth was $y$. Training is then the optimization problem known as **Empirical Risk Minimization (ERM)**:

$$\hat{f}=\arg\min_{f\in\mathcal{H}}\frac{1}{n}\sum_{i=1}^{n}L\big(f(\mathbf{x}_i),y_i\big)$$

Regardless of whether we're solving a classification or a regression problem, the job of the learning algorithm is to find the best mapping function given the available resources. That framework is identical for both tasks.

So where do they actually diverge? In exactly two places:

- The **codomain** $\mathcal{Y}$ — the set the answer is allowed to live in.
- The **loss function** $L$ — what "wrong" means in that set.

Everything else in the classification-vs-regression debate is downstream of those two choices. Their distinction hinges on something more fundamental than discrete-versus-continuous: the formulation of the problem and how their loss functions are approximated.

## 2. Regression: Predicting a Quantity

### Formal Definition

**Regression** is a supervised learning technique used to predict continuous numerical values by learning relationships between input variables (features) and an output variable (target). It helps understand how changes in one or more factors influence a measurable outcome.

In regression, the target space is the real line:

$$\mathcal{Y}=\mathbb{R}\quad\text{(or }\mathbb{R}^k\text{ for multi-output regression)}$$

The crucial property isn't just "it's a number" - it's that $\mathcal{Y}$ carries a **metric and an ordering**. For example, the statement $\lvert 72-70\rvert<\lvert 72-40\rvert$ is meaningful and useful. Predicting 70 when the truth is 72 is a *small* mistake; predicting 40 is a *large* one. Regression models are built to exploit that structure.

Regression analysis is a statistical methodology generally used for the numeric prediction of continuous-valued functions.

### Linear Regression

The simplest hypothesis space is the set of affine functions:

$$\hat{y}=f(\mathbf{x})=\mathbf{w}^\top\mathbf{x}+b=w_1x_1+w_2x_2+\dots+w_dx_d+b$$

Here $\mathbf{w}\in\mathbb{R}^d$ are the **weights** (how much each feature pushes the prediction) and $b$ is the **bias** or intercept.

The canonical loss is **Mean Squared Error (MSE)**:

$$L_{\text{MSE}}=\frac{1}{n}\sum_{i=1}^{n}\big(y_i-\hat{y}_i\big)^2$$

## 3. Classification: Predicting a Category

### Formal Definition

**Classification** is a supervised machine learning technique used to predict labels or categories from input data. It assigns each data point to a predefined class based on learned patterns.

For example, a classification model might be trained on dataset of images labeled as cats, dogs or birds; subsequently, the model can be used to predict whether new, previously unseen images belong to the cat, dog, or bird class based on features such as colour, texture, or shape.

![Classification Example](/assets/img/AI/ML/Supervised-Learning/1-classification-regression/classification-example.png)

In classification, the target space is a **finite, unordered set**:

$$\mathcal{Y}=\{C_1,C_2,\dots,C_K\}$$

- $K=2$ → **binary classification** (spam / not spam, malignant / benign)
- $K>2$ → **multi-class classification** (which of 1000 ImageNet categories)
- Each sample can hold several labels at once → **multi-label classification** (an article tagged both *politics* and *economics*)

The defining property is what's *missing*: there's no meaningful distance between labels. If your classes are $\{\text{cat},\text{dog},\text{bird}\}$ and you natively encode them as $\{0,1,2\}$, you've silently told the model that a cat is "closer" to a dog than to a bird, and that a dog is the *average* of a cat and a bird. That's nonsense, and it's precisely why we use **one-hot encoding** instead: cat $\to[1,0,0]$, dog $\to[0,1,0]$, bird $\to[0,0,1]$. Every class sits at an equal distance from every other.

Classification extracts models - called classifiers - that predict discrete categorical class labels.

### The Problem With The "Natural" Loss

The loss we actually care about in classification is the **0-1 loss**:

$$L_{0/1}(\hat{y},y)=\mathbb{1}[\hat{y}\neq y]=\begin{cases}0 & \hat{y}=y\\ 1 & \hat{y}\neq y\end{cases}$$

### Logistic Regression

### Cross-entropy

### How We Measure Classifiers
