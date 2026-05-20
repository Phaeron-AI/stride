# Component 09 — Syntax Verifier

## Identity

```
Type:                   Transformer — Encoder with Classification Heads
Parameters:             ~22M
Role:                   Fast structural validity checker
                        Runs on rolling window of partially generated code
                        Catches syntax errors before they propagate
                        Lightest model in suite — designed for speed
```

---

## Hyperparameters

```
d_model:                256
n_heads:                4
d_head:                 64
n_layers:               6
d_ff:                   1024       (4 × d_model)
dropout:                0.1
max_seq_len:            256        rolling window size
vocab_size:             32,000
activation:             GELU
norm:                   LayerNorm
bias:                   False

syntax_error_classes:   12
  0   no_error
  1   unexpected_indent
  2   unexpected_dedent
  3   invalid_syntax
  4   missing_colon
  5   unmatched_bracket
  6   unmatched_paren
  7   unmatched_brace
  8   invalid_token
  9   unterminated_string
  10  undefined_name
  11  type_error
```

---

## Parameter Count Verification

```
Embedding layer
  token embedding          32,000 × 256               =   8.192M
  position embedding       256 × 256                  =   0.066M
  embedding total                                     =   8.258M

Per transformer layer
  Attention
    Q projection           256 × 256                  =   0.066M
    K projection           256 × 256                  =   0.066M
    V projection           256 × 256                  =   0.066M
    output projection      256 × 256                  =   0.066M
    attention total                                   =   0.264M
  FFN
    up projection          256 × 1024                 =   0.262M
    down projection        1024 × 256                 =   0.262M
    ffn total                                         =   0.524M
  Layer norms (×2)         2 × (2 × 256)              =   0.001M
  Per layer total                                     ≈   0.789M

All 6 layers               6 × 0.789M                 =   4.734M
Output head                256 × 32,000               =   8.192M

Classification heads
  syntax valid             256 × 1                    =   0.000M
  error location           256 × 256                  =   0.066M
  error type               256 × 12                   =   0.003M
  heads total                                         =   0.069M

Grand Total                                           ≈  21.253M ✓
```

---

## Dimensional Walkthrough

```
INPUT
  partial code tokens      (8, 256)
    rolling window of currently generated code
    window slides forward as generation proceeds
    B = 8, window_len = 256

EMBEDDING
  token embeddings         (8, 256, 256)
  position embeddings      (1, 256, 256)
  combined                 (8, 256, 256)
    B, seq_len, d_model

PER ATTENTION LAYER (×6)

  Layer Norm               (8, 256, 256)

  Q, K, V Projections
    Q                      (8, 256, 256)
    K                      (8, 256, 256)
    V                      (8, 256, 256)

  Reshape to Heads
    Q                      (8, 4, 256, 64)
    K                      (8, 4, 256, 64)
    V                      (8, 4, 256, 64)
      B, n_heads, seq_len, d_head

  Attention Scores
    QK^T / sqrt(64)        (8, 4, 256, 256)
    softmax                (8, 4, 256, 256)
    no causal mask — full bidirectional for error detection

  Attention Output
    scores × V             (8, 4, 256, 64)
    reshape                (8, 256, 256)
    output projection      (8, 256, 256)
    residual               (8, 256, 256)

  FFN
    Layer Norm             (8, 256, 256)
    up projection          (8, 256, 1024)
    GELU                   (8, 256, 1024)
    down projection        (8, 256, 256)
    residual               (8, 256, 256)

AFTER ALL 6 LAYERS
  hidden states            (8, 256, 256)

FINAL LAYER NORM
  output                   (8, 256, 256)

POOLED REPRESENTATION
  mean pool over seq_len   (8, 256)

CLASSIFICATION HEADS

  Syntax Valid Head
    linear + sigmoid       (8, 256) → (8, 1)
      1.0 = valid syntax
      0.0 = syntax error detected

  Error Location Head
    linear                 (8, 256) → (8, 256)
    softmax over seq dim   (8, 256)
      → attention distribution over token positions
      → peak indicates where error likely occurs
      (only meaningful when syntax_valid < 0.5)

  Error Type Head
    linear                 (8, 256) → (8, 12)
    softmax                (8, 12)
      → probability distribution over 12 error classes
      (only meaningful when syntax_valid < 0.5)
```

---

## Training Objective

```
Primary loss:             BCE on syntax_valid binary label
Secondary loss:           Cross entropy on error_type (masked to error cases only)
Tertiary loss:            KL divergence on error_location
                          (soft labels from parser error positions)
Ground truth:             Python ast.parse() — zero cost, perfect labels
                          run on millions of code snippets from GitHub
```

---

## Outputs Fed To

```
→  Component 08   Code Generation Head
     syntax_valid  (8, 1)    gate — halt generation if < 0.5
     error_location (8, 256)  inform correction
     error_type    (8, 12)   classify error for targeted correction

→  Component 06   Process Reward Model
     syntax_valid used as hard constraint in PRM scoring
     syntactically invalid steps scored 0.0 regardless of other dims
```

---

## Rolling Window Mechanics

```
Generation step T:
  current sequence         (8, T)
  take last 256 tokens     (8, 256)   if T > 256
  take all T tokens        (8, T)     if T ≤ 256, pad to (8, 256)

Run verifier every N tokens
  N = 16 (default)
  verify every 16 generated tokens — not every token
  balance between error detection speed and compute overhead

If syntax_valid < 0.5:
  pause generation
  pass error_location and error_type to Generation Head
  Generation Head conditions on error signal to correct
  resume generation
```

---

## Notes

```
- Smallest and fastest model in suite by design
  22M params, 6 layers — runs in milliseconds
  can run synchronously during generation without bottleneck
- Bidirectional attention used (no causal mask)
  needs to see full window to detect bracket mismatches etc
- Ground truth is free — Python parser is perfect label source
  train on all of GitHub Python files (billions of examples)
- Error location is a soft attention map not a hard index
  preserves gradient flow during training
```