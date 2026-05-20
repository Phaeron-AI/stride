# Component 03 — Execution Reasoner

## Identity

```
Type:                   Transformer — Encoder
Parameters:             ~219M
Role:                   Reasons about what code does
                        Predicts program state at each execution step
                        without running the interpreter
```

---

## Hyperparameters

```
d_model:                768
n_heads:                12
d_head:                 64         (d_model / n_heads)
n_layers:               24
d_ff:                   3072       (4 × d_model)
dropout:                0.1
max_seq_len:            512
vocab_size:             32,000
activation:             GELU
norm:                   LayerNorm
bias:                   False
```

---

## Parameter Count Verification

```
Embedding layer
  token embedding          32,000 × 768               =  24.576M
  position embedding       512 × 768                  =   0.393M
  embedding total                                     =  24.969M

Per transformer layer
  Attention
    Q projection           768 × 768                  =   0.590M
    K projection           768 × 768                  =   0.590M
    V projection           768 × 768                  =   0.590M
    output projection      768 × 768                  =   0.590M
    attention total                                   =   2.360M
  FFN
    up projection          768 × 3072                 =   2.359M
    down projection        3072 × 768                 =   2.359M
    ffn total                                         =   4.718M
  Layer norms (×2)         2 × (2 × 768)              =   0.003M
  Per layer total                                     ≈   7.081M

All 24 layers              24 × 7.081M                = 169.944M
Output head                768 × 32,000               =  24.576M

Grand Total                                           ≈ 219.489M ✓
```

---

## Dimensional Walkthrough

```
INPUT
  token ids                (8, 512)
    B = 8, seq_len = 512

EMBEDDING LAYER
  token embeddings         (8, 512, 768)
  position embeddings      (1, 512, 768)   broadcast
  combined                 (8, 512, 768)
    B, seq_len, d_model

PER ATTENTION LAYER (×24)

  Layer Norm
    input                  (8, 512, 768)
    output                 (8, 512, 768)

  Q, K, V Projections
    Q                      (8, 512, 768)
    K                      (8, 512, 768)
    V                      (8, 512, 768)

  Reshape to Heads
    Q                      (8, 12, 512, 64)
    K                      (8, 12, 512, 64)
    V                      (8, 12, 512, 64)
      B, n_heads, seq_len, d_head

  Attention Scores
    QK^T / sqrt(d_head)    (8, 12, 512, 512)
      B, n_heads, seq_len, seq_len
      memory per layer:    8×12×512×512×2 bytes ≈ 50MB
    softmax                (8, 12, 512, 512)

  Attention Output
    scores × V             (8, 12, 512, 64)
    reshape                (8, 512, 768)
    output projection      (8, 512, 768)

  Residual
    input + attn output    (8, 512, 768)

  FFN
    Layer Norm             (8, 512, 768)
    up projection          (8, 512, 3072)
    GELU                   (8, 512, 3072)
    down projection        (8, 512, 768)
    residual               (8, 512, 768)

AFTER ALL 24 LAYERS
  hidden states            (8, 512, 768)

FINAL LAYER NORM
  output                   (8, 512, 768)

OUTPUT HEAD
  logits                   (8, 512, 32000)
    B, seq_len, vocab_size

POOLED REPRESENTATION
  mean pool over seq_len   (8, 768)
    → fed to downstream components
```

---

## Training Objective

```
Primary loss:             Cross entropy on next execution step prediction
Secondary loss:           Variable state prediction at each line
Ground truth:             Execution traces from Component 01
```

---

## Outputs Fed To

```
→  Component 07   Reasoning Graph Builder
     pooled representation (8, 768)
     projected to (8, 512) before entry
```

---

## Notes

```
- Causal (decoder-style) masking not used — full bidirectional attention
- Pooled representation is mean of all token positions
- Attention memory is the main VRAM constraint at this size
  50MB per layer × 24 layers = 1.2GB attention maps alone
  use gradient checkpointing during training
```