---
title: "Introduction to Machine Learning (ML)"
date: 2026-05-22 12:00:00 +0700
categories: [Artificial Intelligence, Machine Learning, Machine Learning Pipeline, Data Preprocessing]
tags: [cpp, machine learning, AI]
math: true
---


## 1. What is AA Tree?

An AA Tree simplifies the Red-Black Tree by eliminating left-leaning red edges. Instead of colors (Red or Black), it uses the concept of a **level** for each node. We have some definitions when working with AA Tree.

**Understanding Levels and Links**

In an AA Tree, the **level** is a specific property assigned to each node. 
* The level of every leaf node is one.
* The level of any node lacking a left child is also strictly one.
* The level of a node with a left child is always exactly one greater than the level of that left child.

Nodes are connected by two types of **links**:
* **Normal Link**: A connection where the child's level is one less than the parent's.
* **Horizontal Link**: A connection where both the parent and the child share the exact same level.

**Strict Properties of an AA Tree**

By applying these concepts, an AA Tree must adhere to these precise rules:
1. **Only right horizontal links are allowed.** The level of a right child can be equal to or one less than its parent. Conversely, the level of a left child must strictly be one less than its parent, meaning left horizontal links are forbidden.
2. There cannot be two consecutive right horizontal links.
3. Any node with a level strictly greater than 1 must possess exactly two children.
4. If a node does not have a right horizontal links, both of its children must be at the same level.

![AA Tree Example](/assets/img/CS/DSA/1_aa_tree/aa-tree.png)

* In the image above, node 30 is the root of the AA tree. The red arrows represent the horizontal links, connecting parent nodes to their right children at the exact same level. 
* Nodes 5, 10, 20, 35, 40, 55, 65, 80, and 90 are all leaf nodes; therefore, they have a level of 1. 
* Nodes 15, 50, 60, and 85 have a level of 2 because they all have a left child at level 1. 
* Similarly, nodes 30 and 70 have a level of 3.

## 2. Implementing the AA Tree

To maintain the tree, we only need a basic node structure containing pointers to the children, the data, and an integer representing the level. We will handle memory management manually to understand exactly how the nodes are linked.

```cpp
#include <iostream>

struct TreeNode {
    int data;
    int level;
    TreeNode* left;
    TreeNode* right;
    TreeNode(int val) : data(val), level(1), left(nullptr), right(nullptr) {} 
};
```

To keep the tree balanced, the AA Tree relies on two fundamental utility operations:
* **skew**: Fixes the issue of a left horizontal link (a left child with the same level as its parent) by performing a **right rotation**.
* **split**: Fixes the issue of two consecutive right horizontal links (a right grandchild with the same level as its grandparent) by performing a **left rotation** and increasing the level of the new root.

**Skew**
* Node C is the leftchild of node N.
* Node C and node N share the same level, creating a left horizontal link.
* Skew operation is used by right rotation at node C and node N.


This is the source code of **skew** operation using C++ pointers:
```cpp
#include <iostream>

TreeNode* skew(TreeNode* root) {
  if (root == nullptr || root->left == nullptr) 
    return root;
  if (root->left->level == root->level) {
    TreeNode* leftChild = root->left;
    root->left = leftChild->right;
    leftChild->right = root;
    return leftChild;
  }
  return root;
}
```
**Split**
