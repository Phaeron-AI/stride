# Component 05 — Complexity Reasoner

## Identity

```
Type:                   Transformer — Encoder with Classification Heads
Parameters:             ~83M
Role:                   Reasons about algorithmic complexity
                        Derives time and space complexity from code structure
                        Localizes performance bottlenecks
```

---

## Hyperparameters

```
d_model:                512
n_heads:                8
d_head:                 64         (d_model / n_heads)
n_layers:               16
d_ff:                   2048       (4 × d_model)
dropout:                0.1
max_seq_len:            512
vocab_size:             32,000
activation:             GELU
norm:                   LayerNorm
bias:                   False

complexity_classes:     8
  0  O(1)
  1  O(log n)
  2  O(n)
  3  O(n log n)
  4  O(n²)
  5  O(n³)
  6  O(2^n)
  7  O(n!)
```

---

## Parameter Count Verification

```
Embedding layer
  token embedding          32,000 × 512               =  16.384M
  position embedding       512 × 512                  =   0.262M
  embedding total                                     =  16.646M

Per transformer layer
  Attention
    Q projection           512 × 512                  =   0.262M
    K projection           512 × 512                  =   0.262M
    V projection           512 × 512                  =   0.262M
    output projection      512 × 512                  =   0.262M
    attention total                                   =   1.048M
  FFN
    up projection          512 × 2048                 =   1.049M
    down projection        2048 × 512                 =   1.049M
    ffn total                                         =   2.098M
  Layer norms (×2)         2 × (2 × 512)              =   0.002M
  Per layer total                                     ≈   3.148M

All 16 layers              16 × 3.148M                =  50.368M
Output head                512 × 32,000               =  16.384M

Complexity heads
  time complexity          512 × 8                    =   0.004M
  space complexity         512 × 8                    =   0.004M
  bottleneck locator       512 × 512                  =   0.262M
  heads total                                         =   0.270M

Grand Total                                           ≈  83.668M ✓
```

---

## Dimensional Walkthrough

```
INPUT
  code token ids           (8, 512)
    B = 8, seq_len = 512

EMBEDDING
  token embeddings         (8, 512, 512)
  position embeddings      (1, 512, 512)
  combined                 (8, 512, 512)
    B, seq_len, d_model

PER ATTENTION LAYER (×16)

  Layer Norm               (8, 512, 512)

  Q, K, V Projections
    Q                      (8, 512, 512)
    K                      (8, 512, 512)
    V                      (8, 512, 512)

  Reshape to Heads
    Q                      (8, 8, 512, 64)
    K                      (8, 8, 512, 64)
    V                      (8, 8, 512, 64)
      B, n_heads, seq_len, d_head

  Attention Scores
    QK^T / sqrt(64)        (8, 8, 512, 512)
    softmax                (8, 8, 512, 512)

  Attention Output
    scores × V             (8, 8, 512, 64)
    reshape                (8, 512, 512)
    output projection      (8, 512, 512)
    residual               (8, 512, 512)

  FFN
    Layer Norm             (8, 512, 512)
    up projection          (8, 512, 2048)
    GELU                   (8, 512, 2048)
    down projection        (8, 512, 512)
    residual               (8, 512, 512)

AFTER ALL 16 LAYERS
  hidden states            (8, 512, 512)

FINAL LAYER NORM
  output                   (8, 512, 512)

POOLED REPRESENTATION
  mean pool over seq_len   (8, 512)

COMPLEXITY HEADS
  time complexity
    linear                 (8, 512) → (8, 8)
    softmax                (8, 8)
      → probability over 8 time complexity classes

  space complexity
    linear                 (8, 512) → (8, 8)
    softmax                (8, 8)
      → probability over 8 space complexity classes

  bottleneck locator
    linear                 (8, 512) → (8, 512)
    softmax over seq dim   (8, 512)
      → attention over token positions
      → highlights where bottleneck occurs in code
```

---

## Training Objective

```
Primary loss:             Cross entropy on time complexity class
Secondary loss:           Cross entropy on space complexity class
Tertiary loss:            KL divergence on bottleneck location
                          (soft labels from profiling data)
Ground truth:             Empirical benchmarking at multiple input sizes
                          + known complexity labels from LeetCode / competitive programming
```

---

## Outputs Fed To

```
→  Component 07   Reasoning Graph Builder
     pooled representation (8, 512)
     time complexity       (8, 8)
     space complexity      (8, 8)
     bottleneck locations  (8, 512)
```

---

## Notes

```
- Smaller model justified: complexity reasoning is a
  narrower classification task than execution or causal reasoning
- Bottleneck locator output is a soft attention map
  not a hard index — preserves gradient flow
- Complexity labels derived empirically by running functions
  with input sizes [10, 100, 1000, 10000] and fitting curve
```