# Component 07 — Reasoning Graph Builder

## Identity

```
Type:                   Transformer — Encoder with Graph Prediction Heads
Parameters:             ~44M
Role:                   Converts linear reasoning traces into
                        structured dependency graphs
                        Central connective tissue of the DERIVE suite
                        Receives from all reasoners, feeds PRM and Generation Head
```

---

## Hyperparameters

```
d_model:                512
n_heads:                8
d_head:                 64
n_layers:               8          shallower — structural classification task
d_ff:                   2048
dropout:                0.1
max_seq_len:            512
vocab_size:             32,000
activation:             GELU
norm:                   LayerNorm
bias:                   False

max_nodes:              64         maximum reasoning steps in graph
tokens_per_node:        8          512 seq_len / 64 nodes
edge_types:             6
  0  supports
  1  contradicts
  2  refines
  3  branches
  4  depends_on
  5  produces
node_types:             4
  0  observation
  1  inference
  2  hypothesis
  3  conclusion
```

---

## Parameter Count Verification

```
Embedding layer
  token embedding          32,000 × 512               =  16.384M
  position embedding       512 × 512                  =   0.262M
  embedding total                                     =  16.646M

Per transformer layer
  per layer total                                     ≈   3.148M
8 layers                   8 × 3.148M                 =  25.184M

Input fusion layer
  execution pooled proj    768 × 512                  =   0.393M
  causal pooled proj       768 × 512                  =   0.393M
  complexity pooled proj   512 × 512                  =   0.262M
  fusion linear            (512 × 4) × 512            =   1.049M
    (text + 3 specialist signals)
  fusion total                                        =   2.097M

Graph prediction heads
  adjacency head           512 × 64                   =   0.033M
    (node_dim → max_nodes, then outer product)
  edge type head           512 × (64 × 6)             =   0.197M
  node type head           512 × 4                    =   0.002M
  heads total                                         =   0.232M

Grand Total                                           ≈  44.159M ✓
```

---

## Dimensional Walkthrough

```
INPUT A — Reasoning Text
  token ids                (8, 512)

INPUT B — Specialist Signals
  execution pooled         (8, 768)    from Component 03
  causal pooled            (8, 768)    from Component 04
  causal edges             (8, 512, 512)   from Component 04
  causal edge types        (8, 512, 512, 6) from Component 04
  complexity pooled        (8, 512)    from Component 05
  complexity time          (8, 8)      from Component 05
  complexity space         (8, 8)      from Component 05
  complexity bottleneck    (8, 512)    from Component 05

SPECIALIST SIGNAL PROJECTION
  execution pooled         (8, 768) → (8, 512)   linear
  causal pooled            (8, 768) → (8, 512)   linear
  complexity pooled        (8, 512)               no projection needed
  concatenate signals      (8, 1536)
    512 + 512 + 512
  fuse to d_model          (8, 512)               linear

TEXT EMBEDDING
  token embeddings         (8, 512, 512)
  position embeddings      (1, 512, 512)
  combined                 (8, 512, 512)

SIGNAL INJECTION
  fused signal             (8, 512)
  expand                   (8, 1, 512)
  broadcast add            (8, 512, 512)
    adds specialist context to every token position

THROUGH 8 TRANSFORMER LAYERS
  per layer attention      (8, 8, 512, 512)
  per layer output         (8, 512, 512)
  final hidden states      (8, 512, 512)

FINAL LAYER NORM
  output                   (8, 512, 512)

SEGMENT INTO NODES
  reshape                  (8, 64, 8, 512)
    64 nodes, 8 tokens per node
  mean pool within nodes   (8, 64, 512)
    → one embedding per reasoning step node

GRAPH PREDICTION HEADS

  Node Type Head
    linear                 (8, 64, 512) → (8, 64, 4)
    softmax over dim=2     (8, 64, 4)
      → node type probabilities per node

  Adjacency Head
    project nodes          (8, 64, 512) → (8, 64, 64)
    outer product          (8, 64, 64)
      node_i · node_j^T for all pairs
    sigmoid                (8, 64, 64)
      → P(edge exists from node i to node j)
    threshold at 0.5       (8, 64, 64)   binary adjacency matrix

  Edge Type Head
    for existing edges only
    concatenate node pairs (8, E, 1024)
      E = number of edges above threshold
      concat node_i and node_j embeddings
    linear                 (8, E, 1024) → (8, E, 6)
    softmax                (8, E, 6)
      → edge type distribution per edge

OUTPUT GRAPH
  node embeddings          (8, 64, 512)
  node types               (8, 64, 4)
  adjacency matrix         (8, 64, 64)    binary
  edge types               (8, E, 6)      sparse, E = active edges
```

---

## Training Objective

```
Adjacency loss:           BCE on edge existence
Edge type loss:           Cross entropy on edge type classification
Node type loss:           Cross entropy on node type classification
Ground truth:             Human-annotated dependency graphs (small set)
                          + synthetic graphs from structured reasoning traces
```

---

## Outputs Fed To

```
→  Component 06   Process Reward Model
     node embeddings  (8, 64, 512)
     adjacency        (8, 64, 64)

→  Component 08   Code Generation Head
     node embeddings  (8, 64, 512)
     projected to     (8, 64, 768) before cross attention
```

---

## Notes

```
- Shallowest model in suite (8 layers) because task is
  structural classification, not generative reasoning
- Specialist signals injected via broadcast add —
  simple but effective fusion strategy
  alternative: cross attention to specialist outputs (more expensive)
- Edge type head operates only on existing edges (sparse)
  not all 64×64 pairs — saves compute
- Causal edges from Component 04 are auxiliary signal
  graph builder learns its own structure on top
```