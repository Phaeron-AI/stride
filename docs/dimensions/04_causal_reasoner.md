# Component 04 — Causal Reasoner

## Identity

```
Type:                   Transformer — Encoder with Graph Output Heads
Parameters:             ~219M
Role:                   Reasons about why code behaves as it does
                        Builds dependency graphs from code structure
                        Predicts causal relationships between code elements
```

---

## Hyperparameters

```
d_model:                768
n_heads:                12
d_head:                 64
n_layers:               24
d_ff:                   3072
dropout:                0.1
max_seq_len:            512
vocab_size:             32,000
activation:             GELU
norm:                   LayerNorm
bias:                   False

ast_feature_dim:        64         AST structural features per token
edge_types:             6
  0  supports
  1  contradicts
  2  refines
  3  branches
  4  depends_on
  5  produces
```

---

## Parameter Count Verification

```
Base transformer
  identical to Execution Reasoner                    ≈ 219.489M

Additional heads
  AST input projection   (768 + 64) × 768            =   0.640M
  edge prediction head   768 × 512                   =   0.393M
  edge type head         768 × (512 × 6)             =   2.359M
  additional total                                   ≈   3.392M

Grand Total                                          ≈ 222.881M
  (reported as ~219M base, heads add ~3M)
```

---

## Dimensional Walkthrough

```
INPUT
  code token ids           (8, 512)
    B = 8, seq_len = 512
  AST dependency features  (8, 512, 64)
    structural signal per token position
    extracted from Abstract Syntax Tree

EMBEDDING
  token embeddings         (8, 512, 768)
  AST features             (8, 512, 64)   already continuous

INPUT PROJECTION
  concatenate              (8, 512, 832)
    along feature dim: 768 + 64
  linear project           (8, 512, 768)
    back to d_model

POSITION EMBEDDINGS
  add positions            (8, 512, 768)

THROUGH 24 LAYERS
  (identical to Execution Reasoner internals)
  per layer attention      (8, 12, 512, 512)
  per layer output         (8, 512, 768)
  final hidden states      (8, 512, 768)

CAUSAL GRAPH OUTPUT HEADS

  Node Embeddings
    final hidden states    (8, 512, 768)
    one embedding per token position

  Edge Prediction Head
    pairwise scores        (8, 512, 512)
      for all token pairs i, j
    sigmoid                (8, 512, 512)
      → P(causal edge exists from i to j)
    threshold at 0.5       (8, 512, 512)   binary adjacency

  Edge Type Head
    edge type logits       (8, 512, 512, 6)
      for each possible edge, classify type
    softmax over dim=3     (8, 512, 512, 6)
      → probability distribution over 6 types

POOLED REPRESENTATION
  mean pool over seq_len   (8, 768)
    → fed to downstream components
```

---

## Training Objective

```
Primary loss:             BCE on edge existence prediction
Secondary loss:           Cross entropy on edge type classification
Ground truth:             Dependency graphs extracted from AST + runtime profiling
```

---

## Outputs Fed To

```
→  Component 07   Reasoning Graph Builder
     node embeddings  (8, 512, 768)
     edge predictions (8, 512, 512)
     edge types       (8, 512, 512, 6)
     pooled           (8, 768) projected to (8, 512)
```

---

## Notes

```
- AST features extracted using Python ast module
- Runtime profiling adds dynamic dependency signal
  on top of static AST structure
- Edge prediction is all-pairs O(seq_len²) — memory intensive
  512 × 512 × 8 batches = manageable but monitor VRAM
- Only edges above threshold fed to graph builder
  typical sparsity: ~5% of possible edges active
```